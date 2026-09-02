'''LiDAR scan loading and robust relative-pose odometry with Open3D Tensor ICP on CPU or CUDA.

The public API keeps the same ``LidarData`` representation used by the higher-level calibration pipeline. Raw scan-format handling remains local to this module, while registration is dataset-independent once XYZ points are loaded.

Supported raw scan conventions are KAIST VLP ``<timestamp_ns>.bin`` files containing little-endian float32 ``[x, y, z, reflectance]`` points and New College-style ``.pcd`` point clouds with timestamps encoded in their filenames.

Relative poses preserve the existing convention: every returned transform maps points from the current LiDAR scan into the immediately previous LiDAR scan frame. When ``local_map_scans > 1``, several recent already-registered scans are merged into a bounded local map expressed in the previous-scan frame, but the returned result is still the same consecutive ``T_Lprevious_Lcurrent`` transform expected by the rest of the pipeline.

Registration uses Open3D's tensor pipeline for both CPU and CUDA. Only the latest ``local_map_scans`` preprocessed scans are cached, so memory usage is bounded with sequence length. No deskewing is performed in this module.
'''

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import tarfile
from typing import Any, Sequence
import warnings
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import LidarData

try:
    from .data import timestamps_ns_to_s
except ImportError:
    def timestamps_ns_to_s(timestamps_ns: np.ndarray) -> np.ndarray:
        '''Convert integer nanosecond timestamps to float seconds.'''
        timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        seconds = timestamps_ns // 1_000_000_000
        nanoseconds = timestamps_ns % 1_000_000_000
        return seconds.astype(float) + nanoseconds.astype(float) * 1e-9


##################################################
# Raw scan conventions and registration defaults
##################################################


SAFE_PARALLELIZM = True
LIDAR_POINT_DTYPE = np.dtype('<f4')
LIDAR_VALUES_PER_POINT = 4
DEFAULT_LOCAL_MAP_SCANS = 5
DEFAULT_VOXEL_SCALE_FACTORS = (2.0, 1.0, 0.5)
DEFAULT_DEVICE = 'cpu'
DEFAULT_CUDA_DEVICE_ID = 0

if SAFE_PARALLELIZM:
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


@dataclass(frozen=True)
class _RegistrationConfig:
    '''Internal tensor-registration configuration.'''
    voxel_size_m: float
    max_correspondence_distance_m: float
    max_iterations: int
    local_map_scans: int
    voxel_scale_factors: tuple[float, ...]
    use_previous_motion_initialization: bool
    robust_kernel: str | None
    robust_kernel_scale_factor: float
    bidirectional_check: bool
    device: str
    cuda_device_id: int
    normal_radius_factor: float
    normal_max_nn: int


@dataclass
class _CachedScan:
    '''One preprocessed scan retained in the bounded local-map cache.'''
    cloud: Any
    T_O_L: np.ndarray


##################################################
# Public scan discovery and loading interface
##################################################


def discover_lidar_archives(dataset_root: str | Path) -> list[Path]:
    '''Find supported scan archives below a caller-supplied LiDAR directory.'''
    root = Path(dataset_root).expanduser()
    if root.is_file():
        return [root] if _is_supported_archive(root) else []
    if not root.exists():
        return []
    return sorted(path for path in root.rglob('*') if path.is_file() and _is_supported_archive(path))


def discover_lidar_scans(dataset_root: str | Path, *, extract_if_needed: bool = True) -> list[Path]:
    '''Find supported ``.bin`` or ``.pcd`` LiDAR scans below one caller-supplied directory.'''
    root = Path(dataset_root).expanduser()
    if root.is_file() and root.suffix.lower() in {'.bin', '.pcd'}:
        return [root]
    if not root.exists():
        return []
    scans = _discover_scan_files(root)
    if scans or not extract_if_needed:
        return scans
    for archive in discover_lidar_archives(root):
        target_dir = _archive_extract_dir(archive)
        target_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_archive(archive, target_dir)
    return _discover_scan_files(root)


def sort_by_filename(scans: list[Path]) -> list[Path]:
    '''Sort scans by timestamps encoded in their filenames.'''
    def timestamp_key(path: Path) -> tuple[int, str]:
        timestamp_ns = _timestamp_ns_from_path(path)
        if timestamp_ns is None:
            raise ValueError(f'Could not parse a LiDAR timestamp from filename {path.name!r}. Expected KAIST <timestamp_ns>.bin or a filename ending in <seconds>_<nanoseconds>.pcd.')
        return timestamp_ns, str(path)
    return sorted(scans, key=timestamp_key)


def load_lidar_scan_xyz(path: str | Path) -> np.ndarray:
    '''Load one supported raw LiDAR scan and return finite float32 XYZ points.

    This public wrapper intentionally delegates to the module's private KAIST
    ``.bin`` and Open3D ``.pcd`` parsers so callers can reuse the established
    raw-scan convention without duplicating binary decoding logic.
    '''
    open3d = _import_open3d()
    return _load_lidar_scan_xyz(open3d, Path(path).expanduser())


def scan_timestamps_from_paths(scans: Sequence[str | Path], *, scan_period_s: float | None = None) -> np.ndarray:
    '''Read scan timestamps from supported filenames using the existing parser.'''
    return _timestamps_from_paths([Path(path).expanduser() for path in scans], scan_period_s=scan_period_s)


def timestamp_from_scan_path(path: str | Path) -> float:
    '''Parse one supported scan filename into seconds, or NaN when unavailable.'''
    return _timestamp_from_path(Path(path).expanduser())


def load_lidar_relative_poses(dataset_root: str | Path, *, voxel_size_m: float = 0.5, max_correspondence_distance_m: float = 1.5, max_iterations: int = 50, max_scans: int | None = None, scan_period_s: float | None = None, extract_if_needed: bool = True, number_of_workers: int = 1, stamp_file: str | Path | None = None, local_map_scans: int = DEFAULT_LOCAL_MAP_SCANS, voxel_scale_factors: tuple[float, ...] = DEFAULT_VOXEL_SCALE_FACTORS, use_previous_motion_initialization: bool = True, robust_kernel: str | None = 'tukey', robust_kernel_scale_factor: float = 1.0, bidirectional_check: bool = False, device: str = DEFAULT_DEVICE, cuda_device_id: int = DEFAULT_CUDA_DEVICE_ID, normal_radius_factor: float = 2.0, normal_max_nn: int = 30) -> LidarData:
    '''Compute robust consecutive LiDAR relative poses with Open3D Tensor ICP on CPU or CUDA.

    ``device='cpu'`` uses ``CPU:0`` and ``device='cuda'`` uses ``CUDA:<cuda_device_id>``. The mathematical registration path is otherwise the same. CUDA requires an Open3D build with CUDA enabled.

    ``local_map_scans`` controls how many latest already-registered scans form the target map. ``local_map_scans=1`` reproduces the old scan-to-previous-scan topology while still allowing multi-scale ICP, a robust kernel, and a motion initial guess. The default value is five scans, which is short enough for hand-carried LiDAR motion but gives sparse vehicle-mounted scans more geometric support than a single pair.

    Only the latest ``local_map_scans`` preprocessed scans are retained in memory. Each cached scan is voxel-downsampled once at the finest requested scale and has normals estimated once. The local map is rebuilt only from this bounded cache, so memory usage does not grow with the duration of the sequence.
    '''
    config = _validated_registration_config(voxel_size_m=voxel_size_m, max_correspondence_distance_m=max_correspondence_distance_m, max_iterations=max_iterations, local_map_scans=local_map_scans, voxel_scale_factors=voxel_scale_factors, use_previous_motion_initialization=use_previous_motion_initialization, robust_kernel=robust_kernel, robust_kernel_scale_factor=robust_kernel_scale_factor, bidirectional_check=bidirectional_check, device=device, cuda_device_id=cuda_device_id, normal_radius_factor=normal_radius_factor, normal_max_nn=normal_max_nn)
    scans = discover_lidar_scans(dataset_root, extract_if_needed=extract_if_needed)
    if stamp_file is not None:
        scans, scan_timestamps_s = _order_scans_from_stamp_file(scans, Path(stamp_file).expanduser())
    else:
        scans = sort_by_filename(scans)
        scan_timestamps_s = _timestamps_from_paths(scans, scan_period_s=scan_period_s)
    if max_scans is not None:
        if int(max_scans) < 2:
            raise ValueError('max_scans must be at least two when provided')
        scans = scans[:int(max_scans)]
        scan_timestamps_s = scan_timestamps_s[:int(max_scans)]
    if len(scans) < 2:
        raise ValueError('At least two LiDAR scans are required to compute relative poses.')

    open3d = _import_open3d()
    open3d_device = _resolve_open3d_device(open3d, config)

    sequential_registration_required = config.device == 'cuda' or config.local_map_scans > 1 or config.use_previous_motion_initialization or config.bidirectional_check
    workers_used = max(int(number_of_workers), 1)
    if workers_used > 1 and sequential_registration_required:
        warnings.warn('CUDA registration, local-map registration, previous-motion initialization, and bidirectional checks require sequential odometry state. number_of_workers is therefore reduced to 1. Independent CPU multiprocessing remains available only with device="cpu", local_map_scans=1, use_previous_motion_initialization=False, and bidirectional_check=False.', RuntimeWarning, stacklevel=2)
        workers_used = 1

    if workers_used <= 1:
        relative_poses, fitness, inlier_rmse, diagnostics = _register_scan_sequence_serial(scans, config=config, open3d=open3d, open3d_device=open3d_device)
    else:
        relative_poses, fitness, inlier_rmse, diagnostics = _register_scan_pairs_parallel_independent(scans, config=config, number_of_workers=workers_used)

    poses = np.stack(relative_poses)
    interval_timestamps_s = 0.5 * (scan_timestamps_s[:-1] + scan_timestamps_s[1:])
    metadata = {'raw_format': _describe_scan_formats(scans), 'registration': 'open3d_tensor_multiscale_point_to_plane_icp', 'relative_pose_convention': 'current_scan_to_previous_scan', 'device': config.device, 'open3d_device': str(open3d_device), 'cuda_device_id': config.cuda_device_id if config.device == 'cuda' else None, 'voxel_size_m': config.voxel_size_m, 'max_correspondence_distance_m': config.max_correspondence_distance_m, 'max_iterations_total': config.max_iterations, 'local_map_scans': config.local_map_scans, 'voxel_scale_factors': config.voxel_scale_factors, 'cache_voxel_size_m': _cache_voxel_size(config), 'use_previous_motion_initialization': config.use_previous_motion_initialization, 'robust_kernel': config.robust_kernel, 'robust_kernel_scale_factor': config.robust_kernel_scale_factor, 'bidirectional_check': config.bidirectional_check, 'normal_radius_factor': config.normal_radius_factor, 'normal_max_nn': config.normal_max_nn, 'number_of_workers_requested': int(number_of_workers), 'number_of_workers_used': workers_used, 'stamp_file': str(stamp_file) if stamp_file is not None else None, **_summarize_registration_diagnostics(diagnostics)}
    return LidarData(timestamps_s=interval_timestamps_s, relative_poses_se3=poses, scan_timestamps_s=scan_timestamps_s, source_scan_paths=[str(path) for path in scans], fitness=np.asarray(fitness, dtype=float), inlier_rmse=np.asarray(inlier_rmse, dtype=float), metadata=metadata)


##################################################
# Registration configuration and device validation
##################################################


def _validated_registration_config(*, voxel_size_m: float, max_correspondence_distance_m: float, max_iterations: int, local_map_scans: int, voxel_scale_factors: tuple[float, ...], use_previous_motion_initialization: bool, robust_kernel: str | None, robust_kernel_scale_factor: float, bidirectional_check: bool, device: str, cuda_device_id: int, normal_radius_factor: float, normal_max_nn: int) -> _RegistrationConfig:
    '''Validate public registration controls and normalize them once.'''
    voxel_size_m = float(voxel_size_m)
    max_correspondence_distance_m = float(max_correspondence_distance_m)
    max_iterations = int(max_iterations)
    local_map_scans = int(local_map_scans)
    robust_kernel_scale_factor = float(robust_kernel_scale_factor)
    cuda_device_id = int(cuda_device_id)
    normal_radius_factor = float(normal_radius_factor)
    normal_max_nn = int(normal_max_nn)
    normalized_device = str(device).strip().lower()
    if normalized_device not in {'cpu', 'cuda'}:
        raise ValueError("device must be either 'cpu' or 'cuda'")
    if cuda_device_id < 0:
        raise ValueError('cuda_device_id must be non-negative')
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError('voxel_size_m must be finite and positive')
    if not np.isfinite(max_correspondence_distance_m) or max_correspondence_distance_m <= 0.0:
        raise ValueError('max_correspondence_distance_m must be finite and positive')
    if max_iterations <= 0:
        raise ValueError('max_iterations must be positive')
    if local_map_scans <= 0:
        raise ValueError('local_map_scans must be at least one')
    if not np.isfinite(robust_kernel_scale_factor) or robust_kernel_scale_factor <= 0.0:
        raise ValueError('robust_kernel_scale_factor must be finite and positive')
    if not np.isfinite(normal_radius_factor) or normal_radius_factor <= 0.0:
        raise ValueError('normal_radius_factor must be finite and positive')
    if normal_max_nn <= 0:
        raise ValueError('normal_max_nn must be positive')
    scale_factors = tuple(float(value) for value in voxel_scale_factors)
    if not scale_factors:
        raise ValueError('voxel_scale_factors must contain at least one value')
    if any(not np.isfinite(value) or value <= 0.0 for value in scale_factors):
        raise ValueError('voxel_scale_factors must contain only finite positive values')
    if any(scale_factors[index] <= scale_factors[index + 1] for index in range(len(scale_factors) - 1)):
        raise ValueError('voxel_scale_factors must be strictly decreasing from coarse to fine')
    normalized_kernel = None if robust_kernel is None else str(robust_kernel).strip().lower()
    if normalized_kernel not in {None, 'tukey', 'huber', 'cauchy', 'l2'}:
        raise ValueError("robust_kernel must be one of None, 'l2', 'tukey', 'huber', or 'cauchy'")
    return _RegistrationConfig(voxel_size_m=voxel_size_m, max_correspondence_distance_m=max_correspondence_distance_m, max_iterations=max_iterations, local_map_scans=local_map_scans, voxel_scale_factors=scale_factors, use_previous_motion_initialization=bool(use_previous_motion_initialization), robust_kernel=normalized_kernel, robust_kernel_scale_factor=robust_kernel_scale_factor, bidirectional_check=bool(bidirectional_check), device=normalized_device, cuda_device_id=cuda_device_id, normal_radius_factor=normal_radius_factor, normal_max_nn=normal_max_nn)


def _resolve_open3d_device(open3d, config: _RegistrationConfig):
    '''Resolve ``CPU:0`` or ``CUDA:n`` and fail early when requested CUDA is unavailable.'''
    if config.device == 'cpu':
        return open3d.core.Device('CPU:0')
    if not open3d.core.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but Open3D reports that CUDA is unavailable")
    device_count = int(open3d.core.cuda.device_count())
    if config.cuda_device_id >= device_count:
        raise ValueError(f'cuda_device_id={config.cuda_device_id} is invalid because Open3D reports {device_count} CUDA device(s)')
    return open3d.core.Device(f'CUDA:{config.cuda_device_id}')


def _cache_voxel_size(config: _RegistrationConfig) -> float:
    '''Return the finest voxel size used to bound cached scan memory.'''
    return float(config.voxel_size_m * min(config.voxel_scale_factors))


##################################################
# Scan and timestamp parsing
##################################################


def _discover_scan_files(root: Path) -> list[Path]:
    '''Return supported raw scan files below ``root`` without ordering assumptions.'''
    direct_scans: list[Path] = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_file() and Path(entry.name).suffix.lower() in {'.bin', '.pcd'}:
                    direct_scans.append(Path(entry.path))
    except OSError:
        direct_scans = []
    if direct_scans:
        return sorted(direct_scans)
    return sorted(path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in {'.bin', '.pcd'})


def _load_lidar_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    '''Load one KAIST VLP binary scan as XYZ and reflectance arrays.'''
    raw_values = np.fromfile(path, dtype=LIDAR_POINT_DTYPE)
    if raw_values.size == 0:
        raise ValueError(f'LiDAR scan is empty: {path}')
    if raw_values.size % LIDAR_VALUES_PER_POINT != 0:
        raise ValueError(f'LiDAR scan {path} contains {raw_values.size} float32 values, which is not divisible by four [x, y, z, reflectance] values per point.')
    points = raw_values.reshape(-1, LIDAR_VALUES_PER_POINT)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.size == 0:
        raise ValueError(f'LiDAR scan contains no finite points: {path}')
    return points[:, :3].astype(np.float32, copy=False), points[:, 3].astype(np.float32, copy=False)


def _load_lidar_scan_xyz(open3d, path: Path) -> np.ndarray:
    '''Load one supported raw scan and return finite float32 XYZ points.'''
    suffix = path.suffix.lower()
    if suffix == '.bin':
        xyz, _ = _load_lidar_bin(path)
        return xyz
    if suffix == '.pcd':
        cloud = open3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            raise ValueError(f'Point cloud is empty or unreadable: {path}')
        xyz = np.asarray(cloud.points, dtype=np.float32)
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
        if xyz.size == 0:
            raise ValueError(f'Point cloud contains no finite XYZ points: {path}')
        return xyz
    raise ValueError(f'Unsupported LiDAR scan format: {path.suffix!r}')


def _order_scans_from_stamp_file(scans: list[Path], stamp_file: Path) -> tuple[list[Path], np.ndarray]:
    '''Order scan paths according to a one-column nanosecond stamp CSV.'''
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
    missing_timestamps = [int(timestamp_ns) for timestamp_ns in raw_timestamps if int(timestamp_ns) not in scans_by_timestamp]
    if missing_timestamps:
        raise ValueError(f'{len(missing_timestamps)} timestamps from {stamp_file} have no matching scan; first missing values: {missing_timestamps[:5]}')
    return [scans_by_timestamp[int(timestamp_ns)] for timestamp_ns in raw_timestamps], timestamps_ns_to_s(raw_timestamps)


def _timestamps_from_paths(scans: list[Path], *, scan_period_s: float | None) -> np.ndarray:
    '''Read scan timestamps from filenames, with an optional fixed-period fallback.'''
    timestamp_values = [_timestamp_ns_from_path(path) for path in scans]
    if timestamp_values and all(value is not None for value in timestamp_values):
        timestamps_ns = np.asarray(timestamp_values, dtype=np.int64)
        if timestamps_ns.size <= 1 or np.all(np.diff(timestamps_ns) > 0):
            return timestamps_ns_to_s(timestamps_ns)
    if scan_period_s is None:
        raise ValueError('Could not parse strictly increasing timestamps from LiDAR filenames. Provide a valid stamp_file or set scan_period_s.')
    scan_period_s = float(scan_period_s)
    if not np.isfinite(scan_period_s) or scan_period_s <= 0.0:
        raise ValueError('scan_period_s must be finite and positive')
    return np.arange(len(scans), dtype=float) * scan_period_s


def _timestamp_components_from_path(path: Path) -> tuple[int, int] | None:
    '''Return ``(seconds, nanoseconds)`` parsed from a supported scan filename.'''
    timestamp_ns = _timestamp_ns_from_path(path)
    if timestamp_ns is None:
        return None
    return timestamp_ns // 1_000_000_000, timestamp_ns % 1_000_000_000


def _timestamp_from_path(path: Path) -> float:
    '''Parse a supported scan filename into seconds.'''
    timestamp_ns = _timestamp_ns_from_path(path)
    if timestamp_ns is None:
        return np.nan
    return float(timestamps_ns_to_s(np.asarray([timestamp_ns], dtype=np.int64))[0])


def _timestamp_ns_from_path(path: Path) -> int | None:
    '''Return the integer nanosecond timestamp encoded by a KAIST or New College scan filename.'''
    suffix = path.suffix.lower()
    if suffix == '.bin' and path.stem.isdigit():
        value = int(path.stem)
        return value if 0 <= value <= np.iinfo(np.int64).max else None
    if suffix == '.pcd':
        parts = path.stem.split('_')
        if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
            seconds = int(parts[-2])
            nanoseconds = int(parts[-1])
            if seconds >= 0 and 0 <= nanoseconds < 1_000_000_000:
                value = seconds * 1_000_000_000 + nanoseconds
                return value if value <= np.iinfo(np.int64).max else None
    return None


def _describe_scan_formats(scans: list[Path]) -> str:
    '''Return a compact raw-format description for metadata.'''
    suffixes = sorted({path.suffix.lower() for path in scans})
    descriptions = []
    if '.bin' in suffixes:
        descriptions.append('KAIST VLP float32 [x, y, z, reflectance]')
    if '.pcd' in suffixes:
        descriptions.append('Open3D PCD')
    return ' + '.join(descriptions) if descriptions else 'unknown'


##################################################
# Open3D Tensor cloud preprocessing and ICP
##################################################


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError('open3d is required for LiDAR ICP loading. Install it in the notebook environment with `pip install open3d`.') from exc
    return o3d


def _preprocess_scan_tensor(open3d, path: Path, open3d_device, config: _RegistrationConfig):
    '''Load one raw scan, move it to the selected device, downsample once at the finest requested scale, and estimate normals once.'''
    xyz = _load_lidar_scan_xyz(open3d, path)
    positions = open3d.core.Tensor(np.ascontiguousarray(xyz, dtype=np.float32), dtype=open3d.core.Dtype.Float32, device=open3d_device)
    cloud = open3d.t.geometry.PointCloud(positions)
    cache_voxel_size = _cache_voxel_size(config)
    cloud = cloud.voxel_down_sample(cache_voxel_size)
    if int(cloud.point.positions.shape[0]) == 0:
        raise ValueError(f'Voxel downsampling produced an empty point cloud: {path}')
    cloud.estimate_normals(max_nn=config.normal_max_nn, radius=max(config.normal_radius_factor * cache_voxel_size, 1e-3))
    return cloud


def _transform_tensor_cloud(open3d, cloud, transformation: np.ndarray, open3d_device):
    '''Clone one cached tensor cloud and transform its points and normals on the selected device.'''
    transformed = cloud.clone()
    transformation_tensor = open3d.core.Tensor(np.asarray(transformation, dtype=np.float32), dtype=open3d.core.Dtype.Float32, device=open3d_device)
    transformed.transform(transformation_tensor)
    return transformed


def _build_local_map_tensor(open3d, history: deque[_CachedScan], T_O_L_reference: np.ndarray, open3d_device, config: _RegistrationConfig):
    '''Build a bounded target map from only the latest cached scans, expressed in the immediately previous LiDAR frame.'''

    if not history:
        raise ValueError('Cannot build a local map without previous scans')

    T_Lreference_O = np.linalg.inv(np.asarray(T_O_L_reference, dtype=float))

    transformed_positions = []
    transformed_normals = []

    for cached_scan in history:
        T_Lreference_Lscan = T_Lreference_O @ np.asarray(cached_scan.T_O_L, dtype=float)

        transformed_cloud = _transform_tensor_cloud(open3d, cached_scan.cloud, T_Lreference_Lscan, open3d_device)

        transformed_positions.append(transformed_cloud.point.positions)
        transformed_normals.append(transformed_cloud.point.normals)

    ##################################################
    # Open3D core.concatenate has special behavior for a
    # single input tensor and can flatten an (N, 3) tensor.
    # Preserve the tensor directly when the local map contains
    # only one scan.
    ##################################################

    if len(transformed_positions) == 1:
        positions = transformed_positions[0]
        normals = transformed_normals[0]
    else:
        positions = open3d.core.concatenate(transformed_positions, axis=0)
        normals = open3d.core.concatenate(transformed_normals, axis=0)

    if len(positions.shape) != 2 or int(positions.shape[1]) != 3:
        raise RuntimeError(f'Local-map positions have invalid shape {positions.shape}; expected (N, 3)')

    if len(normals.shape) != 2 or int(normals.shape[1]) != 3:
        raise RuntimeError(f'Local-map normals have invalid shape {normals.shape}; expected (N, 3)')

    if int(positions.shape[0]) != int(normals.shape[0]):
        raise RuntimeError(f'Local-map positions and normals have different lengths: {positions.shape[0]} vs {normals.shape[0]}')

    local_map = open3d.t.geometry.PointCloud({'positions': positions, 'normals': normals})

    local_map = local_map.voxel_down_sample(_cache_voxel_size(config))

    if int(local_map.point.positions.shape[0]) == 0:
        raise ValueError('Local-map voxel downsampling produced an empty point cloud')

    return local_map


def _iterations_per_scale(total_iterations: int, number_of_scales: int) -> tuple[int, ...]:
    '''Split one total ICP iteration budget from coarse to fine.'''
    if number_of_scales == 1:
        return (int(total_iterations),)
    weights = np.linspace(number_of_scales, 1.0, number_of_scales)
    weights /= np.sum(weights)
    allocations = np.maximum(1, np.floor(weights * int(total_iterations)).astype(int))
    difference = int(total_iterations) - int(np.sum(allocations))
    allocation_index = 0
    while difference != 0:
        index = allocation_index % number_of_scales
        if difference > 0:
            allocations[index] += 1
            difference -= 1
        elif allocations[index] > 1:
            allocations[index] -= 1
            difference += 1
        allocation_index += 1
    return tuple(int(value) for value in allocations)


def _tensor_robust_kernel(open3d, config: _RegistrationConfig):
    '''Create the tensor robust kernel used by point-to-plane ICP.'''
    robust_kernel_module = open3d.t.pipelines.registration.robust_kernel
    method_map = {'l2': robust_kernel_module.RobustKernelMethod.L2Loss, 'huber': robust_kernel_module.RobustKernelMethod.HuberLoss, 'cauchy': robust_kernel_module.RobustKernelMethod.CauchyLoss, 'tukey': robust_kernel_module.RobustKernelMethod.TukeyLoss}
    method = robust_kernel_module.RobustKernelMethod.L2Loss if config.robust_kernel is None else method_map[config.robust_kernel]
    kernel_scale_m = max(config.robust_kernel_scale_factor * config.voxel_size_m, 1e-6)
    return robust_kernel_module.RobustKernel(method, kernel_scale_m, 1.0)


def _register_tensor_multiscale(open3d, *, source, target, initial_transform: np.ndarray, config: _RegistrationConfig) -> tuple[np.ndarray, float, float, dict[str, float]]:
    '''Register source into target with Open3D Tensor multi-scale point-to-plane ICP on the source/target device.'''
    registration = open3d.t.pipelines.registration
    voxel_sizes = open3d.utility.DoubleVector([config.voxel_size_m * scale_factor for scale_factor in config.voxel_scale_factors])
    max_correspondence_distances = open3d.utility.DoubleVector([config.max_correspondence_distance_m * scale_factor for scale_factor in config.voxel_scale_factors])
    criteria_list = [registration.ICPConvergenceCriteria(max_iteration=iterations) for iterations in _iterations_per_scale(config.max_iterations, len(config.voxel_scale_factors))]
    estimation = registration.TransformationEstimationPointToPlane(_tensor_robust_kernel(open3d, config))
    initial_transform_tensor = open3d.core.Tensor(np.asarray(initial_transform, dtype=np.float64), dtype=open3d.core.Dtype.Float64, device=open3d.core.Device('CPU:0'))
    result = registration.multi_scale_icp(source, target, voxel_sizes, criteria_list, max_correspondence_distances, initial_transform_tensor, estimation)
    transformation = np.asarray(result.transformation.cpu().numpy(), dtype=float)
    if transformation.shape != (4, 4) or not np.all(np.isfinite(transformation)):
        raise RuntimeError('Open3D Tensor ICP did not produce a finite 4x4 transformation')
    finest_correspondence_distance = float(config.max_correspondence_distance_m * min(config.voxel_scale_factors))
    information_function = getattr(registration, 'get_information_matrix', None)
    if information_function is None:
        information_function = getattr(registration, 'get_information_matrix_from_point_clouds')
    information_tensor = information_function(source, target, finest_correspondence_distance, result.transformation)
    information_matrix = np.asarray(information_tensor.cpu().numpy(), dtype=float)
    information_matrix = 0.5 * (information_matrix + information_matrix.T)
    information_eigenvalues = np.linalg.eigvalsh(information_matrix)
    positive_eigenvalues = information_eigenvalues[information_eigenvalues > 1e-12]
    if positive_eigenvalues.size == 0:
        information_min_eigenvalue = 0.0
        information_condition_number = np.inf
    else:
        information_min_eigenvalue = float(np.min(positive_eigenvalues))
        information_condition_number = float(np.max(positive_eigenvalues) / information_min_eigenvalue)
    diagnostics = {'information_min_eigenvalue': information_min_eigenvalue, 'information_condition_number': information_condition_number}
    return transformation, float(result.fitness), float(result.inlier_rmse), diagnostics


def _bidirectional_closure_diagnostics(open3d, *, current_cloud, target_map, forward_transform: np.ndarray, config: _RegistrationConfig) -> dict[str, float]:
    '''Independently solve the reverse registration and report SE(3) closure; this does not modify the accepted forward odometry.'''
    reverse_transform, _, _, _ = _register_tensor_multiscale(open3d, source=target_map, target=current_cloud, initial_transform=np.linalg.inv(forward_transform), config=config)
    closure = forward_transform @ reverse_transform
    translation_closure_m = float(np.linalg.norm(closure[:3, 3]))
    trace_value = np.clip((np.trace(closure[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    rotation_closure_deg = float(np.rad2deg(np.arccos(trace_value)))
    return {'bidirectional_translation_closure_m': translation_closure_m, 'bidirectional_rotation_closure_deg': rotation_closure_deg}


##################################################
# Sequential bounded-cache local-map odometry
##################################################


def _register_scan_sequence_serial(scans: list[Path], *, config: _RegistrationConfig, open3d, open3d_device) -> tuple[list[np.ndarray], list[float], list[float], list[dict[str, float]]]:
    '''Register a scan sequence while retaining only the latest local-map scans on the chosen Open3D device.'''
    relative_poses: list[np.ndarray] = []
    fitness: list[float] = []
    inlier_rmse: list[float] = []
    diagnostics: list[dict[str, float]] = []
    first_cloud = _preprocess_scan_tensor(open3d, scans[0], open3d_device, config)
    T_O_L_previous = np.eye(4)
    previous_relative_pose: np.ndarray | None = None
    history: deque[_CachedScan] = deque(maxlen=config.local_map_scans)
    history.append(_CachedScan(cloud=first_cloud, T_O_L=T_O_L_previous.copy()))
    for scan_path in tqdm(scans[1:], desc=f'LiDAR ICP [{config.device}]'):
        current_cloud = _preprocess_scan_tensor(open3d, scan_path, open3d_device, config)
        target_map = _build_local_map_tensor(open3d, history, T_O_L_previous, open3d_device, config)
        initial_transform = previous_relative_pose if config.use_previous_motion_initialization and previous_relative_pose is not None else np.eye(4)
        relative_pose, pair_fitness, pair_rmse, pair_diagnostics = _register_tensor_multiscale(open3d, source=current_cloud, target=target_map, initial_transform=initial_transform, config=config)
        if config.bidirectional_check:
            pair_diagnostics.update(_bidirectional_closure_diagnostics(open3d, current_cloud=current_cloud, target_map=target_map, forward_transform=relative_pose, config=config))
        relative_poses.append(relative_pose)
        fitness.append(pair_fitness)
        inlier_rmse.append(pair_rmse)
        diagnostics.append(pair_diagnostics)
        T_O_L_current = T_O_L_previous @ relative_pose
        history.append(_CachedScan(cloud=current_cloud, T_O_L=T_O_L_current.copy()))
        previous_relative_pose = relative_pose
        T_O_L_previous = T_O_L_current
    if config.device == 'cuda':
        open3d.core.cuda.synchronize(open3d_device)
    return relative_poses, fitness, inlier_rmse, diagnostics


##################################################
# Optional independent CPU multiprocessing path
##################################################


def _scan_pair_chunks(number_of_scans: int, number_of_workers: int) -> list[tuple[int, int]]:
    '''Split consecutive independent pair indices into overlapping worker chunks.'''
    number_of_pairs = number_of_scans - 1
    if number_of_pairs <= 0:
        return []
    worker_count = min(max(int(number_of_workers), 1), number_of_pairs)
    boundaries = np.linspace(0, number_of_pairs, worker_count + 1, dtype=int)
    return [(int(start), int(stop)) for start, stop in zip(boundaries[:-1], boundaries[1:]) if start < stop]


def _register_scan_chunk_independent(payload: tuple[list[str], int, _RegistrationConfig]) -> tuple[int, list[np.ndarray], list[float], list[float], list[dict[str, float]]]:
    '''Register one CPU chunk as independent current-to-previous scan pairs.'''
    scan_path_strings, first_pair_index, config = payload
    open3d = _import_open3d()
    open3d_device = open3d.core.Device('CPU:0')
    scan_paths = [Path(path) for path in scan_path_strings]
    relative_poses: list[np.ndarray] = []
    fitness: list[float] = []
    inlier_rmse: list[float] = []
    diagnostics: list[dict[str, float]] = []
    previous_cloud = _preprocess_scan_tensor(open3d, scan_paths[0], open3d_device, config)
    for scan_path in scan_paths[1:]:
        current_cloud = _preprocess_scan_tensor(open3d, scan_path, open3d_device, config)
        relative_pose, pair_fitness, pair_rmse, pair_diagnostics = _register_tensor_multiscale(open3d, source=current_cloud, target=previous_cloud, initial_transform=np.eye(4), config=config)
        relative_poses.append(relative_pose)
        fitness.append(pair_fitness)
        inlier_rmse.append(pair_rmse)
        diagnostics.append(pair_diagnostics)
        previous_cloud = current_cloud
    return first_pair_index, relative_poses, fitness, inlier_rmse, diagnostics


def _register_scan_pairs_parallel_independent(scans: list[Path], *, config: _RegistrationConfig, number_of_workers: int) -> tuple[list[np.ndarray], list[float], list[float], list[dict[str, float]]]:
    '''Register independent CPU current-to-previous pairs in process chunks.'''
    if config.device != 'cpu':
        raise ValueError('Independent multiprocessing is supported only with device="cpu"')
    chunks = _scan_pair_chunks(len(scans), number_of_workers)
    payloads = [([str(path) for path in scans[pair_start:pair_stop + 1]], pair_start, config) for pair_start, pair_stop in chunks]
    process_context = mp.get_context('spawn')
    chunk_results = []
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=process_context) as executor:
        futures = [executor.submit(_register_scan_chunk_independent, payload) for payload in payloads]
        for future in tqdm(as_completed(futures), total=len(futures), desc='LiDAR ICP chunks [cpu]'):
            chunk_results.append(future.result())
    chunk_results.sort(key=lambda result: result[0])
    relative_poses = [pose for _, chunk_poses, _, _, _ in chunk_results for pose in chunk_poses]
    fitness = [value for _, _, chunk_fitness, _, _ in chunk_results for value in chunk_fitness]
    inlier_rmse = [value for _, _, _, chunk_rmse, _ in chunk_results for value in chunk_rmse]
    diagnostics = [value for _, _, _, _, chunk_diagnostics in chunk_results for value in chunk_diagnostics]
    expected_pair_count = len(scans) - 1
    if len(relative_poses) != expected_pair_count:
        raise RuntimeError(f'Parallel ICP produced an unexpected number of scan pairs: {len(relative_poses)} instead of {expected_pair_count}')
    return relative_poses, fitness, inlier_rmse, diagnostics


##################################################
# Registration diagnostic summaries
##################################################


def _summarize_registration_diagnostics(diagnostics: list[dict[str, float]]) -> dict[str, float | None]:
    '''Store compact diagnostic summaries without bloating the normalized CSV.'''
    if not diagnostics:
        return {}
    summary: dict[str, float | None] = {}
    diagnostic_names = sorted({name for pair_diagnostics in diagnostics for name in pair_diagnostics})
    for diagnostic_name in diagnostic_names:
        values = np.asarray([pair_diagnostics.get(diagnostic_name, np.nan) for pair_diagnostics in diagnostics], dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            summary[f'{diagnostic_name}_median'] = None
            summary[f'{diagnostic_name}_p90'] = None
        else:
            summary[f'{diagnostic_name}_median'] = float(np.median(finite_values))
            summary[f'{diagnostic_name}_p90'] = float(np.percentile(finite_values, 90))
    return summary


##################################################
# Optional archive extraction
##################################################


def _is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith('.tar') or name.endswith('.tar.gz') or name.endswith('.tgz') or name.endswith('.zip')


def _archive_extract_dir(archive: Path) -> Path:
    name = archive.name
    for suffix in ('.tar.gz', '.tgz', '.tar', '.zip'):
        if name.lower().endswith(suffix):
            return archive.with_name(name[:-len(suffix)])
    return archive.with_suffix('')


def _safe_extract_archive(archive: Path, target_dir: Path) -> None:
    '''Extract a supported archive while rejecting paths outside the target directory.'''
    if archive.name.lower().endswith('.zip'):
        _safe_extract_zip(archive, target_dir)
    else:
        _safe_extract_tar(archive, target_dir)


def _safe_extract_tar(archive: Path, target_dir: Path) -> None:
    '''Safely extract one tar archive.'''
    target_root = target_dir.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            destination = (target_dir / member.name).resolve()
            try:
                destination.relative_to(target_root)
            except ValueError as exc:
                raise ValueError(f'Unsafe path {member.name!r} in archive {archive}.') from exc
        tar.extractall(target_dir)


def _safe_extract_zip(archive: Path, target_dir: Path) -> None:
    '''Safely extract one ZIP archive.'''
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            destination = (target_dir / member.filename).resolve()
            try:
                destination.relative_to(target_root)
            except ValueError as exc:
                raise ValueError(f'Unsafe path {member.filename!r} in archive {archive}.') from exc
        zip_file.extractall(target_dir)


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


def save_lidar_data_csv(lidar_data: LidarData, filepath: str | Path) -> Path:
    '''Save a normalized ``LidarData`` entity into one CSV without raw scan paths.'''
    output_path = Path(filepath).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interval_timestamps = np.asarray(lidar_data.timestamps_s, dtype=float).reshape(-1)
    relative_poses = np.asarray(lidar_data.relative_poses_se3, dtype=float)
    scan_timestamps = np.asarray(lidar_data.scan_timestamps_s, dtype=float).reshape(-1)
    fitness = np.asarray(lidar_data.fitness, dtype=float).reshape(-1)
    inlier_rmse = np.asarray(lidar_data.inlier_rmse, dtype=float).reshape(-1)
    number_of_intervals = interval_timestamps.size
    if number_of_intervals == 0:
        raise ValueError('lidar_data must contain at least one relative-pose interval')
    if relative_poses.shape != (number_of_intervals, 4, 4):
        raise ValueError('relative_poses_se3 must have shape (N, 4, 4), where N is the number of interval timestamps')
    if scan_timestamps.shape != (number_of_intervals + 1,):
        raise ValueError('scan_timestamps_s must have shape (N + 1,)')
    if fitness.shape != (number_of_intervals,):
        raise ValueError('fitness must have shape (N,)')
    if inlier_rmse.shape != (number_of_intervals,):
        raise ValueError('inlier_rmse must have shape (N,)')
    arrays_to_check = (interval_timestamps, relative_poses, scan_timestamps, fitness, inlier_rmse)
    if not all(np.all(np.isfinite(array)) for array in arrays_to_check):
        raise ValueError('lidar_data contains non-finite numerical values')
    metadata_json = json.dumps(dict(lidar_data.metadata), default=_json_compatible, ensure_ascii=False, separators=(',', ':'))
    pose_columns = [f'T_{row}{column}' for row in range(4) for column in range(4)]
    records: list[dict[str, object]] = []
    for interval_index in range(number_of_intervals):
        flattened_pose = relative_poses[interval_index].reshape(-1)
        record: dict[str, object] = {'interval_index': interval_index, 'timestamp_s': interval_timestamps[interval_index], 'scan_start_timestamp_s': scan_timestamps[interval_index], 'scan_end_timestamp_s': scan_timestamps[interval_index + 1], 'fitness': fitness[interval_index], 'inlier_rmse': inlier_rmse[interval_index]}
        for column_name, value in zip(pose_columns, flattened_pose):
            record[column_name] = float(value)
        record['metadata_json'] = metadata_json if interval_index == 0 else ''
        records.append(record)
    column_order = ['interval_index', 'timestamp_s', 'scan_start_timestamp_s', 'scan_end_timestamp_s', 'fitness', 'inlier_rmse', *pose_columns, 'metadata_json']
    pd.DataFrame(records, columns=column_order).to_csv(output_path, index=False, float_format='%.17g')
    return output_path


def load_lidar_data_csv(filepath: str | Path) -> LidarData:
    '''Load a normalized ``LidarData`` cache created by ``save_lidar_data_csv``.'''
    source_path = Path(filepath).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f'LiDAR CSV does not exist: {source_path}')
    table = pd.read_csv(source_path)
    pose_columns = [f'T_{row}{column}' for row in range(4) for column in range(4)]
    required_columns = ['interval_index', 'timestamp_s', 'scan_start_timestamp_s', 'scan_end_timestamp_s', 'fitness', 'inlier_rmse', *pose_columns, 'metadata_json']
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
    if not np.allclose(scan_start_timestamps[1:], scan_end_timestamps[:-1], rtol=0.0, atol=1e-12):
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
    return LidarData(timestamps_s=interval_timestamps, relative_poses_se3=relative_poses, scan_timestamps_s=scan_timestamps, source_scan_paths=[], fitness=fitness, inlier_rmse=inlier_rmse, metadata=metadata)