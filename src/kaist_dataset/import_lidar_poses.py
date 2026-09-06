'''Command-line KAIST absolute LiDAR pose importer.

This script independently registers every selected raw KAIST VLP scan against
the reconstructed LAS map and saves absolute LiDAR sensor poses T_W_L.

The reference body trajectory is used only to initialize each scan-to-map
registration and to choose the relevant LAS map region. The saved measurement
remains T_W_L and is not converted to T_W_B.

The script intentionally contains no short-test run, plotting, trajectory
comparison, registration-overlay visualization, or CSV reload verification.

Typical usage:

    python -m kaist_dataset.import_lidar_poses /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --side left --device cuda

    python -m kaist_dataset.import_lidar_poses /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --side right --device cuda --cuda-device-id 0 --map-cache-cell-size-m 15 --map-cache-size 21 --cuda-prefetch-workers 8 --cuda-prefetch-depth 64

The output defaults to:

    <project_root>/data/KAISTDataset/<UrbanXX>/lidar_map_poses_vlp_<side>.csv
'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


##################################################
# Support both module execution and direct execution
##################################################

if __package__ in {None, ''}:
    SRC_ROOT = Path(__file__).resolve().parents[1]

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from kaist_dataset.data import import_true_trajectory
    from kaist_dataset.lidar import discover_lidar_scans, scan_timestamps_from_paths, sort_by_filename
    from kaist_dataset.lidar_map import LidarReferenceMap, bounds_around_positions, choose_open3d_device, discover_las_files, initial_lidar_positions_from_reference, localize_lidar_scans_to_map, save_lidar_map_pose_csv
else:
    from .data import import_true_trajectory
    from .lidar import discover_lidar_scans, scan_timestamps_from_paths, sort_by_filename
    from .lidar_map import LidarReferenceMap, bounds_around_positions, choose_open3d_device, discover_las_files, initial_lidar_positions_from_reference, localize_lidar_scans_to_map, save_lidar_map_pose_csv


##################################################
# KAIST VLP extrinsics
##################################################
#
# T_A_B maps coordinates from B into A.
#
# Vehicle2LeftVLP.txt and Vehicle2RightVLP.txt are used here as:
#
#     T_B_LL : Left LiDAR -> body
#     T_B_RL : Right LiDAR -> body
#
##################################################

T_B_LL = np.array([
    [-0.514066, -0.702201, -0.492595, -0.440699],
    [0.486485, -0.711672, 0.506809, 0.397052],
    [-0.706447, 0.0208933, 0.707457, 1.90953],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

T_B_RL = np.array([
    [-0.512152, 0.699241, -0.498761, -0.449885],
    [-0.494811, -0.714859, -0.494104, -0.416713],
    [-0.702041, -0.0062642, 0.712109, 1.91294],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

T_B_LIDAR_BY_SIDE = {
    'left': T_B_LL,
    'right': T_B_RL,
}


##################################################
# CLI helpers
##################################################


def _parse_robust_kernel(value: str) -> str | None:
    value = str(value).strip().lower()

    if value in {'none', 'null', 'off'}:
        return None

    if value not in {'tukey', 'huber', 'cauchy', 'l2'}:
        raise argparse.ArgumentTypeError("robust kernel must be one of: tukey, huber, cauchy, l2, none")

    return value


def _dataset_name(dataset_root: Path) -> str:
    name = dataset_root.name

    if not name.lower().startswith('urban'):
        raise ValueError(f"Expected dataset_root to point to a KAIST UrbanXX directory, got {dataset_root}")

    return name


def _dataset_paths(dataset_root: Path, side: str) -> tuple[Path, Path, Path]:
    dataset_name = _dataset_name(dataset_root)
    dataset_slug = dataset_name.lower()

    sensor_data_root = dataset_root / f'{dataset_slug}_data' / dataset_slug / 'sensor_data'
    las_map_root = dataset_root / f'{dataset_slug}_las'
    reference_pose_root = dataset_root / f'{dataset_slug}_pose' / dataset_slug / 'global_pose.csv'

    if not sensor_data_root.is_dir():
        raise FileNotFoundError(f'KAIST sensor_data directory was not found: {sensor_data_root}')

    if not las_map_root.exists():
        raise FileNotFoundError(f'KAIST LAS map directory was not found: {las_map_root}')

    if not reference_pose_root.exists():
        raise FileNotFoundError(f'KAIST reference-pose directory was not found: {reference_pose_root}')

    direct_lidar_root = sensor_data_root / f'VLP_{side}'

    if direct_lidar_root.is_dir():
        lidar_root = direct_lidar_root
    else:
        matches = sorted(path for path in sensor_data_root.glob(f'*VLP*{side}*') if path.is_dir())

        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one {side} VLP directory below {sensor_data_root}, found {len(matches)}: {matches}")

        lidar_root = matches[0]

    return lidar_root, las_map_root, reference_pose_root


def _default_output_path(dataset_root: Path, side: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    dataset_name = _dataset_name(dataset_root)
    return project_root / 'data' / 'KAISTDataset' / dataset_name / f'lidar_map_poses_vlp_{side}.csv'


def _nearest_reference_mismatches(scan_timestamps_s: np.ndarray, reference_timestamps_s: np.ndarray) -> np.ndarray:
    scan_timestamps_s = np.asarray(scan_timestamps_s, dtype=float).reshape(-1)
    reference_timestamps_s = np.asarray(reference_timestamps_s, dtype=float).reshape(-1)

    insertion_indices = np.searchsorted(reference_timestamps_s, scan_timestamps_s, side='left')
    right_indices = np.clip(insertion_indices, 0, reference_timestamps_s.size - 1)
    left_indices = np.clip(insertion_indices - 1, 0, reference_timestamps_s.size - 1)

    left_mismatches = np.abs(scan_timestamps_s - reference_timestamps_s[left_indices])
    right_mismatches = np.abs(reference_timestamps_s[right_indices] - scan_timestamps_s)

    return np.minimum(left_mismatches, right_mismatches)


def _select_full_scan_sequence(scan_paths: list[Path], scan_timestamps_s: np.ndarray, reference_timestamps_s: np.ndarray, *, scan_step: int, max_reference_mismatch_s: float | None) -> tuple[list[Path], np.ndarray]:
    scan_timestamps_s = np.asarray(scan_timestamps_s, dtype=float)
    reference_timestamps_s = np.asarray(reference_timestamps_s, dtype=float)

    if scan_step <= 0:
        raise ValueError('scan_step must be positive')

    mask = (scan_timestamps_s >= reference_timestamps_s[0]) & (scan_timestamps_s <= reference_timestamps_s[-1])

    if max_reference_mismatch_s is not None:
        mismatch_s = _nearest_reference_mismatches(scan_timestamps_s, reference_timestamps_s)
        mask &= mismatch_s <= float(max_reference_mismatch_s)

    indices = np.flatnonzero(mask)[::scan_step]

    return [scan_paths[index] for index in indices], scan_timestamps_s[indices]


def _log(verbosity: int, level: int, *values) -> None:
    if verbosity >= level:
        print(*values)


##################################################
# Argument parser
##################################################


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Register KAIST VLP scans to the reconstructed LAS map and save absolute T_W_L poses.')

    parser.add_argument('dataset_root', type=Path, help='Path to one KAIST UrbanXX directory, for example /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16.')
    parser.add_argument('--side', choices=('left', 'right'), default='left', help='KAIST VLP sensor to process.')
    parser.add_argument('--output', type=Path, default=None, help='Output CSV path. Default: data/KAISTDataset/<UrbanXX>/lidar_map_poses_vlp_<side>.csv.')
    parser.add_argument('--verbosity', '-v', type=int, choices=(0, 1, 2), default=1, help='Console verbosity level.')

    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda', help='Open3D registration device.')
    parser.add_argument('--cuda-device-id', type=int, default=0, help='CUDA device index.')

    parser.add_argument('--map-voxel-size-m', type=float, default=0.25, help='Voxel size used when loading/downsampling the LAS reference map.')
    parser.add_argument('--scan-voxel-size-m', type=float, default=0.25, help='Base scan voxel size used by ICP.')
    parser.add_argument('--map-crop-radius-m', type=float, default=50.0, help='XY radius of each local LAS registration crop.')
    parser.add_argument('--map-crop-z-margin-m', type=float, default=10.0, help='Vertical half-margin of each local LAS registration crop.')
    parser.add_argument('--max-correspondence-distance-m', type=float, default=1.5, help='Base ICP correspondence gate.')
    parser.add_argument('--max-iterations', type=int, default=50, help='Total ICP iteration budget.')
    parser.add_argument('--voxel-scale-factors', type=float, nargs='+', default=(2.0, 1.0, 0.5), metavar='SCALE', help='Coarse-to-fine ICP scale factors.')
    parser.add_argument('--robust-kernel', type=_parse_robust_kernel, default='tukey', help='ICP robust kernel: tukey, huber, cauchy, l2, or none.')
    parser.add_argument('--robust-kernel-scale-factor', type=float, default=1.0, help='Robust-kernel scale relative to the current voxel size.')

    parser.add_argument('--max-reference-mismatch-s', type=float, default=5.0, help='Maximum allowed nearest reference-timestamp mismatch for a scan.')
    parser.add_argument('--scan-step', type=int, default=1, help='Process every Nth valid scan. Default 1 processes the complete valid sequence.')
    parser.add_argument('--extract-if-needed', action=argparse.BooleanOptionalAction, default=False, help='Extract LiDAR archives if raw scans are not already available.')

    parser.add_argument('--map-load-mode', choices=('selected_scans', 'full'), default='selected_scans', help='Load only the LAS region around selected scan initializations, or load the complete LAS map.')
    parser.add_argument('--map-load-xy-margin-m', type=float, default=None, help='Additional XY margin used when loading LAS bounds. Default: map_crop_radius_m + 10 m.')
    parser.add_argument('--map-load-z-margin-m', type=float, default=None, help='Additional Z margin used when loading LAS bounds. Default: map_crop_z_margin_m + 5 m.')
    parser.add_argument('--map-chunk-size-points', type=int, default=5_000_000, help='LAS chunk size used while loading the reference map.')

    parser.add_argument('--normal-radius-factor', type=float, default=2.0, help='Normal-estimation radius relative to voxel size.')
    parser.add_argument('--normal-max-nn', type=int, default=30, help='Maximum neighbours used for normal estimation.')

    parser.add_argument('--cpu-workers', type=int, default=None, help='CPU localization workers. None lets the backend choose.')
    parser.add_argument('--cpu-threads-per-worker', type=int, default=1, help='Open3D/BLAS thread budget per CPU worker.')

    parser.add_argument('--map-cache-cell-size-m', type=float, default=10.0, help='Spatial cell size used by the prepared local-map target cache.')
    parser.add_argument('--map-cache-size', type=int, default=4, help='Maximum number of prepared map targets retained in the cache.')

    parser.add_argument('--cuda-prefetch-workers', type=int, default=2, help='CPU threads that prefetch raw scans for CUDA registration.')
    parser.add_argument('--cuda-prefetch-depth', type=int, default=8, help='Maximum CUDA scan-prefetch queue depth.')
    parser.add_argument('--cleanup-interval-scans', type=int, default=0, help='Optional periodic cleanup interval. Zero disables forced cleanup.')

    return parser


##################################################
# Processing
##################################################


def run(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {dataset_root}')

    lidar_root, las_map_root, reference_pose_root = _dataset_paths(dataset_root, args.side)
    output_path = _default_output_path(dataset_root, args.side) if args.output is None else args.output.expanduser()
    T_B_LIDAR = T_B_LIDAR_BY_SIDE[args.side]

    progress = args.verbosity >= 1
    effective_device = choose_open3d_device(args.device, args.cuda_device_id)

    ##################################################
    # Discover the complete raw sequence and reference data
    ##################################################

    raw_scans = discover_lidar_scans(lidar_root, extract_if_needed=args.extract_if_needed)
    raw_scans = sort_by_filename(raw_scans)

    if not raw_scans:
        raise RuntimeError(f'No raw {args.side} LiDAR scans were found below {lidar_root}')

    scan_timestamps_s = scan_timestamps_from_paths(raw_scans)
    reference_timestamps_s, reference_poses_T_W_B = import_true_trajectory(reference_pose_root)
    las_files = discover_las_files(las_map_root)

    if not las_files:
        raise RuntimeError(f'No LAS/LAZ map files were found below {las_map_root}')

    selected_scan_paths, selected_scan_timestamps_s = _select_full_scan_sequence(raw_scans, scan_timestamps_s, reference_timestamps_s, scan_step=args.scan_step, max_reference_mismatch_s=args.max_reference_mismatch_s)

    if not selected_scan_paths:
        raise RuntimeError('No LiDAR scans remain after reference-time overlap and mismatch filtering.')

    _log(args.verbosity, 1, f'Dataset: {dataset_root}')
    _log(args.verbosity, 1, f'LiDAR side: {args.side}')
    _log(args.verbosity, 1, f'Raw LiDAR root: {lidar_root}')
    _log(args.verbosity, 1, f'LAS map root: {las_map_root}')
    _log(args.verbosity, 1, f'Reference pose root: {reference_pose_root}')
    _log(args.verbosity, 1, f'Output CSV: {output_path}')
    _log(args.verbosity, 1, f'Raw scans discovered: {len(raw_scans)}')
    _log(args.verbosity, 1, f'Scans selected for full localization: {len(selected_scan_paths)}')
    _log(args.verbosity, 1, f'Requested device: {args.device}; effective device: {effective_device}')

    if args.verbosity >= 2:
        print(f'Selected timestamp range: {selected_scan_timestamps_s[0]:.9f} .. {selected_scan_timestamps_s[-1]:.9f}')
        print(f'Reference timestamp range: {reference_timestamps_s[0]:.9f} .. {reference_timestamps_s[-1]:.9f}')
        print(f'LAS files: {len(las_files)}')

        for las_file in las_files:
            print(f'  {las_file}')

    ##################################################
    # Load only the map region needed by this full sequence
    ##################################################

    map_bounds_min = None
    map_bounds_max = None

    if args.map_load_mode == 'selected_scans':
        initial_positions_W, _ = initial_lidar_positions_from_reference(selected_scan_timestamps_s, reference_timestamps_s, reference_poses_T_W_B, T_B_LIDAR, max_reference_mismatch_s=args.max_reference_mismatch_s)

        map_load_xy_margin_m = args.map_crop_radius_m + 10.0 if args.map_load_xy_margin_m is None else args.map_load_xy_margin_m
        map_load_z_margin_m = args.map_crop_z_margin_m + 5.0 if args.map_load_z_margin_m is None else args.map_load_z_margin_m

        map_bounds_min, map_bounds_max = bounds_around_positions(initial_positions_W, xy_margin_m=map_load_xy_margin_m, z_margin_m=map_load_z_margin_m)

        _log(args.verbosity, 2, f'Map bounds min XYZ: {map_bounds_min}')
        _log(args.verbosity, 2, f'Map bounds max XYZ: {map_bounds_max}')

    reference_map = LidarReferenceMap.from_las_files(
        las_files,
        voxel_size_m=args.map_voxel_size_m,
        bounds_min_xyz=map_bounds_min,
        bounds_max_xyz=map_bounds_max,
        chunk_size_points=args.map_chunk_size_points,
        progress=progress,
    )

    _log(args.verbosity, 1, f'Loaded reference-map points: {reference_map.point_count}')

    ##################################################
    # Perform one full absolute scan-to-map localization pass
    ##################################################

    results = localize_lidar_scans_to_map(
        reference_map,
        selected_scan_paths,
        reference_timestamps_s,
        reference_poses_T_W_B,
        T_B_LIDAR,
        initial_guess_source='reference',
        max_reference_mismatch_s=args.max_reference_mismatch_s,
        scan_voxel_size_m=args.scan_voxel_size_m,
        map_crop_radius_m=args.map_crop_radius_m,
        map_crop_z_margin_m=args.map_crop_z_margin_m,
        max_correspondence_distance_m=args.max_correspondence_distance_m,
        max_iterations=args.max_iterations,
        voxel_scale_factors=tuple(args.voxel_scale_factors),
        robust_kernel=args.robust_kernel,
        robust_kernel_scale_factor=args.robust_kernel_scale_factor,
        device=effective_device,
        cuda_device_id=args.cuda_device_id,
        max_scans=None,
        scan_step=1,
        progress=progress,
        normal_radius_factor=args.normal_radius_factor,
        normal_max_nn=args.normal_max_nn,
        cpu_workers=args.cpu_workers,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
        map_cache_cell_size_m=args.map_cache_cell_size_m,
        map_cache_size=args.map_cache_size,
        cuda_prefetch_workers=args.cuda_prefetch_workers,
        cuda_prefetch_depth=args.cuda_prefetch_depth,
        cleanup_interval_scans=args.cleanup_interval_scans,
    )

    ##################################################
    # Save exactly the T_W_L CSV consumed by notebook 19
    ##################################################

    saved_path = save_lidar_map_pose_csv(results, output_path)

    successful_count = sum(bool(result.success) for result in results)

    _log(args.verbosity, 1, f'Saved absolute LiDAR poses: {saved_path}')
    _log(args.verbosity, 1, f'Successful registrations: {successful_count} / {len(results)}')

    if args.verbosity >= 2 and results:
        fitness = np.asarray([result.fitness for result in results if result.success and np.isfinite(result.fitness)], dtype=float)
        inlier_rmse = np.asarray([result.inlier_rmse for result in results if result.success and np.isfinite(result.inlier_rmse)], dtype=float)

        if fitness.size:
            print(f'Fitness median: {float(np.median(fitness)):.6f}')

        if inlier_rmse.size:
            print(f'Inlier RMSE median: {float(np.median(inlier_rmse)):.6f} m')

    return saved_path


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

# python ./kaist_dataset/import_lidar_poses.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --side left --device cuda --cuda-device-id 0 --map-cache-cell-size-m 15 --map-cache-size 21 --cuda-prefetch-workers 8 --cuda-prefetch-depth 64 --verbosity 1
# python ./kaist_dataset/import_lidar_poses.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --side right --device cuda --cuda-device-id 0 --map-cache-cell-size-m 15 --map-cache-size 21 --cuda-prefetch-workers 8 --cuda-prefetch-depth 64 --verbosity 1