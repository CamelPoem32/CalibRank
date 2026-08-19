'''KAIST VLP-16 scan loading and pairwise ICP odometry.

Raw KAIST 3D LiDAR scans are headerless little-endian float32 binary files with
four values per point: ``[x, y, z, reflectance]``. This module keeps that binary
parsing local and exposes the same ``LidarData`` representation used by the
higher-level calibration pipeline.
'''

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import LidarData, timestamps_ns_to_s


##################################################
# Raw KAIST VLP scan convention
##################################################


SAFE_PARALLELIZM = True
LIDAR_POINT_DTYPE = np.dtype('<f4')
LIDAR_VALUES_PER_POINT = 4

if SAFE_PARALLELIZM:
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


##################################################
# Public scan discovery and loading interface
##################################################


def discover_lidar_archives(dataset_root: str | Path) -> list[Path]:
    '''Find tar archives below a caller-supplied KAIST LiDAR directory.'''

    root = Path(dataset_root).expanduser()
    if root.is_file():
        return [root] if _is_tar_archive(root) else []
    if not root.exists():
        return []

    return sorted(path for path in root.rglob('*') if path.is_file() and _is_tar_archive(path))


def discover_lidar_scans(
    dataset_root: str | Path,
    *,
    extract_if_needed: bool = True,
) -> list[Path]:
    '''Find KAIST VLP ``.bin`` scans below a caller-supplied directory.

    The function does not assume names such as ``sensor_data`` or ``VLP_left``.
    High-level code is free to pass the exact LiDAR folder it wants to use.
    '''

    root = Path(dataset_root).expanduser()

    if root.is_file() and root.suffix.lower() == '.bin':
        return [root]
    if not root.exists():
        return []

    scans = sorted(path for path in root.rglob('*.bin') if path.is_file())
    if scans or not extract_if_needed:
        return scans

    # Preserve the old loader's optional archive-extraction behavior, but keep
    # it generic to the directory supplied by the caller.
    for archive in discover_lidar_archives(root):
        target_dir = _archive_extract_dir(archive)
        target_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_tar(archive, target_dir)

    return sorted(path for path in root.rglob('*.bin') if path.is_file())


def sort_by_filename(scans: list[Path]) -> list[Path]:
    '''Sort KAIST LiDAR scans by the integer nanosecond timestamp filename.'''

    def timestamp_key(path: Path) -> tuple[int, str]:
        timestamp_ns = _timestamp_ns_from_path(path)
        if timestamp_ns is None:
            raise ValueError(
                f'Could not parse LiDAR timestamp from filename {path.name!r}; '
                'expected <timestamp_ns>.bin.'
            )
        return timestamp_ns, str(path)

    return sorted(scans, key=timestamp_key)


def load_lidar_relative_poses(
    dataset_root: str | Path,
    *,
    voxel_size_m: float = 0.5,
    max_correspondence_distance_m: float = 1.5,
    max_iterations: int = 50,
    max_scans: int | None = None,
    scan_period_s: float | None = None,
    extract_if_needed: bool = True,
    number_of_workers: int = 1,
    stamp_file: str | Path | None = None,
) -> LidarData:
    '''Compute pairwise LiDAR relative poses from ordered KAIST VLP scans.

    Args:
        dataset_root: Directory containing the desired ``.bin`` scans. No fixed
            KAIST folder hierarchy is assumed.
        voxel_size_m: Voxel size for scan downsampling in meters.
        max_correspondence_distance_m: ICP correspondence gate in meters.
        max_iterations: Maximum ICP iterations per scan pair.
        max_scans: Optional cap for quick notebook checks.
        scan_period_s: Fallback period used only if timestamps cannot be parsed.
        extract_if_needed: Extract tar archives below ``dataset_root`` if no
            already-extracted ``.bin`` scans are found.
        number_of_workers: Number of process workers for chunked ICP.
        stamp_file: Optional KAIST ``VLP_*_stamp.csv``. When supplied, it is
            used as the authoritative scan order and validated against filenames.

    Returns:
        ``LidarData`` containing interval midpoint timestamps and relative SE(3)
        transforms computed with the same current-to-previous ICP convention as
        the previous New College implementation.
    '''

    scans = discover_lidar_scans(dataset_root, extract_if_needed=extract_if_needed)
    if stamp_file is not None:
        scans, scan_timestamps_s = _order_scans_from_stamp_file(scans, Path(stamp_file).expanduser())
    else:
        scans = sort_by_filename(scans)
        scan_timestamps_s = _timestamps_from_paths(scans, scan_period_s=scan_period_s)

    if max_scans is not None:
        scans = scans[:max_scans]
        scan_timestamps_s = scan_timestamps_s[:max_scans]

    if len(scans) < 2:
        raise ValueError('At least two KAIST VLP scans are required to compute relative poses.')

    if number_of_workers <= 1:
        relative_poses, fitness, inlier_rmse = _register_scan_pairs_serial(
            scans,
            voxel_size_m=voxel_size_m,
            max_correspondence_distance_m=max_correspondence_distance_m,
            max_iterations=max_iterations,
        )
    else:
        relative_poses, fitness, inlier_rmse = _register_scan_pairs_parallel(
            scans,
            voxel_size_m=voxel_size_m,
            max_correspondence_distance_m=max_correspondence_distance_m,
            max_iterations=max_iterations,
            number_of_workers=number_of_workers,
        )

    poses = np.stack(relative_poses)
    interval_timestamps_s = 0.5 * (scan_timestamps_s[:-1] + scan_timestamps_s[1:])

    return LidarData(
        timestamps_s=interval_timestamps_s,
        relative_poses_se3=poses,
        scan_timestamps_s=scan_timestamps_s,
        source_scan_paths=[str(path) for path in scans],
        fitness=np.asarray(fitness),
        inlier_rmse=np.asarray(inlier_rmse),
        metadata={
            'raw_format': 'KAIST VLP float32 [x, y, z, reflectance]',
            'voxel_size_m': voxel_size_m,
            'max_correspondence_distance_m': max_correspondence_distance_m,
            'max_iterations': max_iterations,
            'stamp_file': str(stamp_file) if stamp_file is not None else None,
        },
    )


##################################################
# KAIST scan and timestamp parsing
##################################################


def _load_lidar_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    '''Load one KAIST VLP binary scan as XYZ and reflectance arrays.'''

    raw_values = np.fromfile(path, dtype=LIDAR_POINT_DTYPE)
    if raw_values.size == 0:
        raise ValueError(f'LiDAR scan is empty: {path}')
    if raw_values.size % LIDAR_VALUES_PER_POINT != 0:
        raise ValueError(
            f'LiDAR scan {path} contains {raw_values.size} float32 values, which '
            'is not divisible by four [x, y, z, reflectance] values per point.'
        )

    points = raw_values.reshape(-1, LIDAR_VALUES_PER_POINT)
    finite_mask = np.all(np.isfinite(points), axis=1)
    points = points[finite_mask]
    if points.size == 0:
        raise ValueError(f'LiDAR scan contains no finite points: {path}')

    xyz = points[:, :3].astype(np.float64, copy=False)
    reflectance = points[:, 3].astype(np.float64, copy=False)
    return xyz, reflectance


def _order_scans_from_stamp_file(scans: list[Path], stamp_file: Path) -> tuple[list[Path], np.ndarray]:
    '''Order scan paths according to a one-column KAIST nanosecond stamp CSV.'''

    if not stamp_file.is_file():
        raise FileNotFoundError(f'LiDAR timestamp file does not exist: {stamp_file}')

    raw_timestamps = np.genfromtxt(stamp_file, delimiter=',', dtype=np.int64)
    raw_timestamps = np.asarray(raw_timestamps, dtype=np.int64).reshape(-1)
    if raw_timestamps.size == 0:
        raise ValueError(f'LiDAR timestamp file is empty: {stamp_file}')
    if raw_timestamps.size > 1 and np.any(np.diff(raw_timestamps) <= 0):
        raise ValueError(f'LiDAR timestamp file must be strictly increasing: {stamp_file}')

    scans_by_timestamp: dict[int, Path] = {}
    for path in scans:
        timestamp_ns = _timestamp_ns_from_path(path)
        if timestamp_ns is None:
            continue
        if timestamp_ns in scans_by_timestamp:
            raise ValueError(f'Duplicate LiDAR scan timestamp {timestamp_ns} in {path.parent}.')
        scans_by_timestamp[timestamp_ns] = path

    missing_timestamps = [int(ts) for ts in raw_timestamps if int(ts) not in scans_by_timestamp]
    if missing_timestamps:
        preview = missing_timestamps[:5]
        raise ValueError(
            f'{len(missing_timestamps)} timestamps from {stamp_file} have no matching .bin scan; '
            f'first missing values: {preview}'
        )

    ordered_scans = [scans_by_timestamp[int(timestamp_ns)] for timestamp_ns in raw_timestamps]
    return ordered_scans, timestamps_ns_to_s(raw_timestamps)


def _timestamps_from_paths(scans: list[Path], *, scan_period_s: float | None) -> np.ndarray:
    '''Read absolute scan timestamps from integer nanosecond filenames.'''

    timestamp_values = [_timestamp_ns_from_path(path) for path in scans]
    if timestamp_values and all(value is not None for value in timestamp_values):
        timestamps_ns = np.asarray(timestamp_values, dtype=np.int64)
        if timestamps_ns.size <= 1 or np.all(np.diff(timestamps_ns) > 0):
            return timestamps_ns_to_s(timestamps_ns)

    if scan_period_s is None:
        raise ValueError(
            'Could not parse strictly increasing nanosecond timestamps from LiDAR filenames. '
            'Provide a valid KAIST stamp_file or set scan_period_s.'
        )

    return np.arange(len(scans), dtype=float) * float(scan_period_s)


def _timestamp_components_from_path(path: Path) -> tuple[int, int] | None:
    '''Return ``(seconds, nanoseconds)`` parsed from ``<timestamp_ns>.bin``.'''

    timestamp_ns = _timestamp_ns_from_path(path)
    if timestamp_ns is None:
        return None
    return timestamp_ns // 1_000_000_000, timestamp_ns % 1_000_000_000


def _timestamp_from_path(path: Path) -> float:
    '''Parse a KAIST nanosecond scan filename into absolute seconds.'''

    timestamp_ns = _timestamp_ns_from_path(path)
    if timestamp_ns is None:
        return np.nan
    return float(timestamps_ns_to_s(np.asarray([timestamp_ns], dtype=np.int64))[0])


def _timestamp_ns_from_path(path: Path) -> int | None:
    '''Return the integer nanosecond timestamp encoded by a KAIST scan filename.'''

    if path.suffix.lower() != '.bin' or not path.stem.isdigit():
        return None
    try:
        value = int(path.stem)
    except ValueError:
        return None
    if value < 0 or value > np.iinfo(np.int64).max:
        return None
    return value


##################################################
# Open3D cloud construction and ICP
##################################################


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            'open3d is required for KAIST LiDAR ICP loading. '
            'Install it in the notebook environment with `pip install open3d`.'
        ) from exc
    return o3d


def _load_and_preprocess_cloud(open3d, path: Path, voxel_size_m: float):
    xyz, _ = _load_lidar_bin(path)

    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(xyz)
    if cloud.is_empty():
        raise ValueError(f'Point cloud is empty: {path}')

    # Keep the previous point-to-plane ICP preprocessing: voxel downsampling and
    # local normal estimation. Reflectance is not needed by this registration.
    if voxel_size_m > 0.0:
        cloud = cloud.voxel_down_sample(voxel_size_m)

    radius = max(voxel_size_m * 2.0, 1.0)
    cloud.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    return cloud


def _register_pair(
    open3d,
    *,
    source,
    target,
    max_correspondence_distance_m: float,
    max_iterations: int,
):
    estimation = open3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iterations))

    return open3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_m,
        np.eye(4),
        estimation,
        criteria,
    )


def _register_scan_pairs_serial(
    scans: list[Path],
    *,
    voxel_size_m: float,
    max_correspondence_distance_m: float,
    max_iterations: int,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    '''Register all consecutive scan pairs in one process.'''

    open3d = _import_open3d()
    relative_poses: list[np.ndarray] = []
    fitness: list[float] = []
    inlier_rmse: list[float] = []

    previous_cloud = _load_and_preprocess_cloud(open3d, scans[0], voxel_size_m)

    for scan_path in tqdm(scans[1:], desc='LiDAR ICP'):
        current_cloud = _load_and_preprocess_cloud(open3d, scan_path, voxel_size_m)

        # Preserve the previous convention: transform the current scan into the
        # previous scan frame by registering source=current, target=previous.
        result = _register_pair(
            open3d,
            source=current_cloud,
            target=previous_cloud,
            max_correspondence_distance_m=max_correspondence_distance_m,
            max_iterations=max_iterations,
        )

        relative_poses.append(np.asarray(result.transformation, dtype=float))
        fitness.append(float(result.fitness))
        inlier_rmse.append(float(result.inlier_rmse))
        previous_cloud = current_cloud

    return relative_poses, fitness, inlier_rmse


##################################################
# Parallel scan-pair registration
##################################################


def _scan_pair_chunks(number_of_scans: int, number_of_workers: int) -> list[tuple[int, int]]:
    '''Split consecutive scan-pair indices into overlapping worker chunks.'''

    number_of_pairs = number_of_scans - 1
    if number_of_pairs <= 0:
        return []

    worker_count = min(max(int(number_of_workers), 1), number_of_pairs)
    boundaries = np.linspace(0, number_of_pairs, worker_count + 1, dtype=int)

    return [
        (int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
        if start < stop
    ]


def _register_scan_chunk(
    payload: tuple[list[str], int, float, float, int],
) -> tuple[int, list[np.ndarray], list[float], list[float]]:
    '''Load and register one contiguous chunk of KAIST VLP scans.'''

    scan_path_strings, first_pair_index, voxel_size_m, max_correspondence_distance_m, max_iterations = payload
    open3d = _import_open3d()
    scan_paths = [Path(path) for path in scan_path_strings]

    relative_poses: list[np.ndarray] = []
    fitness: list[float] = []
    inlier_rmse: list[float] = []

    previous_cloud = _load_and_preprocess_cloud(open3d, scan_paths[0], voxel_size_m)

    for scan_path in scan_paths[1:]:
        current_cloud = _load_and_preprocess_cloud(open3d, scan_path, voxel_size_m)
        result = _register_pair(
            open3d,
            source=current_cloud,
            target=previous_cloud,
            max_correspondence_distance_m=max_correspondence_distance_m,
            max_iterations=max_iterations,
        )

        relative_poses.append(np.asarray(result.transformation, dtype=float))
        fitness.append(float(result.fitness))
        inlier_rmse.append(float(result.inlier_rmse))
        previous_cloud = current_cloud

    return first_pair_index, relative_poses, fitness, inlier_rmse


def _register_scan_pairs_parallel(
    scans: list[Path],
    *,
    voxel_size_m: float,
    max_correspondence_distance_m: float,
    max_iterations: int,
    number_of_workers: int,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    '''Register consecutive scans in overlapping process chunks.'''

    chunks = _scan_pair_chunks(len(scans), number_of_workers)
    payloads = [
        (
            [str(path) for path in scans[pair_start:pair_stop + 1]],
            pair_start,
            voxel_size_m,
            max_correspondence_distance_m,
            max_iterations,
        )
        for pair_start, pair_stop in chunks
    ]

    process_context = mp.get_context('spawn')
    chunk_results = []

    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=process_context) as executor:
        futures = [executor.submit(_register_scan_chunk, payload) for payload in payloads]
        for future in tqdm(as_completed(futures), total=len(futures), desc='LiDAR ICP chunks'):
            chunk_results.append(future.result())

    # Restore the global pair order after workers finish out of order.
    chunk_results.sort(key=lambda result: result[0])

    relative_poses = [pose for _, chunk_poses, _, _ in chunk_results for pose in chunk_poses]
    fitness = [value for _, _, chunk_fitness, _ in chunk_results for value in chunk_fitness]
    inlier_rmse = [value for _, _, _, chunk_rmse in chunk_results for value in chunk_rmse]

    expected_pair_count = len(scans) - 1
    if len(relative_poses) != expected_pair_count:
        raise RuntimeError(
            'Parallel ICP produced an unexpected number of scan pairs: '
            f'{len(relative_poses)} instead of {expected_pair_count}'
        )

    return relative_poses, fitness, inlier_rmse


##################################################
# Optional archive extraction
##################################################


def _is_tar_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith('.tar') or name.endswith('.tar.gz') or name.endswith('.tgz')


def _archive_extract_dir(archive: Path) -> Path:
    name = archive.name
    for suffix in ('.tar.gz', '.tgz', '.tar'):
        if name.lower().endswith(suffix):
            return archive.with_name(name[:-len(suffix)])
    return archive.with_suffix('')


def _safe_extract_tar(archive: Path, target_dir: Path) -> None:
    '''Extract one tar archive while rejecting paths outside ``target_dir``.'''

    target_root = target_dir.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            destination = (target_dir / member.name).resolve()
            try:
                destination.relative_to(target_root)
            except ValueError as exc:
                raise ValueError(f'Unsafe path {member.name!r} in archive {archive}.') from exc
        tar.extractall(target_dir)


##################################################
# Normalized LidarData CSV cache
##################################################


def _json_compatible(value: Any) -> Any:
    '''Convert common NumPy and path objects into JSON-compatible values.'''

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)

    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def save_lidar_data_csv(
    lidar_data: LidarData,
    filepath: str | Path,
) -> Path:
    '''Save a normalized ``LidarData`` entity into one CSV without raw scan paths.'''

    output_path = Path(filepath).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    interval_timestamps = np.asarray(lidar_data.timestamps_s, dtype=float).reshape(-1)
    relative_poses = np.asarray(lidar_data.relative_poses_se3, dtype=float)
    scan_timestamps = np.asarray(lidar_data.scan_timestamps_s, dtype=float).reshape(-1)
    fitness = np.asarray(lidar_data.fitness, dtype=float).reshape(-1)
    inlier_rmse = np.asarray(lidar_data.inlier_rmse, dtype=float).reshape(-1)

    number_of_intervals = interval_timestamps.size

    # Validate all arrays before serializing the normalized cache.
    if number_of_intervals == 0:
        raise ValueError('lidar_data must contain at least one relative-pose interval')
    if relative_poses.shape != (number_of_intervals, 4, 4):
        raise ValueError(
            'relative_poses_se3 must have shape (N, 4, 4), where '
            'N is the number of interval timestamps'
        )
    if scan_timestamps.shape != (number_of_intervals + 1,):
        raise ValueError('scan_timestamps_s must have shape (N + 1,)')
    if fitness.shape != (number_of_intervals,):
        raise ValueError('fitness must have shape (N,)')
    if inlier_rmse.shape != (number_of_intervals,):
        raise ValueError('inlier_rmse must have shape (N,)')

    arrays_to_check = (interval_timestamps, relative_poses, scan_timestamps, fitness, inlier_rmse)
    if not all(np.all(np.isfinite(array)) for array in arrays_to_check):
        raise ValueError('lidar_data contains non-finite numerical values')

    metadata_json = json.dumps(
        dict(lidar_data.metadata),
        default=_json_compatible,
        ensure_ascii=False,
        separators=(',', ':'),
    )

    pose_columns = [f'T_{row}{column}' for row in range(4) for column in range(4)]
    records: list[dict[str, object]] = []

    for interval_index in range(number_of_intervals):
        flattened_pose = relative_poses[interval_index].reshape(-1)
        record: dict[str, object] = {
            'interval_index': interval_index,
            'timestamp_s': interval_timestamps[interval_index],
            'scan_start_timestamp_s': scan_timestamps[interval_index],
            'scan_end_timestamp_s': scan_timestamps[interval_index + 1],
            'fitness': fitness[interval_index],
            'inlier_rmse': inlier_rmse[interval_index],
        }

        for column_name, value in zip(pose_columns, flattened_pose):
            record[column_name] = float(value)

        record['metadata_json'] = metadata_json if interval_index == 0 else ''
        records.append(record)

    column_order = [
        'interval_index',
        'timestamp_s',
        'scan_start_timestamp_s',
        'scan_end_timestamp_s',
        'fitness',
        'inlier_rmse',
        *pose_columns,
        'metadata_json',
    ]

    pd.DataFrame(records, columns=column_order).to_csv(
        output_path,
        index=False,
        float_format='%.17g',
    )
    return output_path


def load_lidar_data_csv(filepath: str | Path) -> LidarData:
    '''Load a normalized ``LidarData`` cache created by ``save_lidar_data_csv``.'''

    source_path = Path(filepath).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f'LiDAR CSV does not exist: {source_path}')

    table = pd.read_csv(source_path)
    pose_columns = [f'T_{row}{column}' for row in range(4) for column in range(4)]
    required_columns = [
        'interval_index',
        'timestamp_s',
        'scan_start_timestamp_s',
        'scan_end_timestamp_s',
        'fitness',
        'inlier_rmse',
        *pose_columns,
        'metadata_json',
    ]

    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f'LiDAR CSV is missing columns: {missing_columns}')
    if table.empty:
        raise ValueError('LiDAR CSV contains no interval rows')

    table = table.sort_values('interval_index', kind='stable').reset_index(drop=True)
    expected_indices = np.arange(table.shape[0], dtype=int)
    stored_indices = table['interval_index'].to_numpy(dtype=int)
    if not np.array_equal(stored_indices, expected_indices):
        raise ValueError('interval_index must contain consecutive values starting at zero')

    interval_timestamps = table['timestamp_s'].to_numpy(dtype=float)
    scan_start_timestamps = table['scan_start_timestamp_s'].to_numpy(dtype=float)
    scan_end_timestamps = table['scan_end_timestamp_s'].to_numpy(dtype=float)

    if not np.allclose(
        scan_start_timestamps[1:],
        scan_end_timestamps[:-1],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError('Consecutive interval rows contain inconsistent scan timestamps')

    relative_poses = table.loc[:, pose_columns].to_numpy(dtype=float).reshape(-1, 4, 4)
    scan_timestamps = np.concatenate([scan_start_timestamps[:1], scan_end_timestamps])
    fitness = table['fitness'].to_numpy(dtype=float)
    inlier_rmse = table['inlier_rmse'].to_numpy(dtype=float)

    numerical_arrays = (interval_timestamps, scan_timestamps, relative_poses, fitness, inlier_rmse)
    if not all(np.all(np.isfinite(array)) for array in numerical_arrays):
        raise ValueError('LiDAR CSV contains non-finite numerical values')

    metadata_json = '{}'
    for value in table['metadata_json']:
        if isinstance(value, str) and value.strip():
            metadata_json = value
            break

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError('metadata_json contains invalid JSON') from exc

    if not isinstance(metadata, dict):
        raise ValueError('metadata_json must encode a JSON object')

    return LidarData(
        timestamps_s=interval_timestamps,
        relative_poses_se3=relative_poses,
        scan_timestamps_s=scan_timestamps,
        source_scan_paths=[],
        fitness=fitness,
        inlier_rmse=inlier_rmse,
        metadata=metadata,
    )