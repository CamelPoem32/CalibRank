'''Command-line KAIST LiDAR odometry importer.

This script computes relative LiDAR odometry for one KAIST VLP sensor and saves
the resulting LidarData object using the project's normalized CSV format.

The script intentionally contains no IMU loading, plotting, short-test
subsequences, trajectory comparison, or CSV reload diagnostics.

Typical usage:

    python -m kaist_dataset.import_lidar_odometry /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --side left

    python -m kaist_dataset.import_lidar_odometry /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --side right --device cpu --local-map-scans 5 --workers 24

The output defaults to:

    <project_root>/data/KAISTDataset/<UrbanXX>/lidar_odometry_vlp_<side>.csv
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

    from kaist_dataset.lidar import load_lidar_relative_poses, save_lidar_data_csv
else:
    from .lidar import load_lidar_relative_poses, save_lidar_data_csv


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


def _lidar_root(dataset_root: Path, side: str) -> Path:
    dataset_name = _dataset_name(dataset_root)
    dataset_slug = dataset_name.lower()
    sensor_data_root = dataset_root / f'{dataset_slug}_data' / dataset_slug / 'sensor_data'

    if not sensor_data_root.is_dir():
        raise FileNotFoundError(f'KAIST sensor_data directory was not found: {sensor_data_root}')

    direct_path = sensor_data_root / f'VLP_{side}'

    if direct_path.is_dir():
        return direct_path

    matches = sorted(path for path in sensor_data_root.glob(f'*VLP*{side}*') if path.is_dir())

    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {side} VLP directory below {sensor_data_root}, found {len(matches)}: {matches}")

    return matches[0]


def _default_output_path(dataset_root: Path, side: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    dataset_name = _dataset_name(dataset_root)
    return project_root / 'data' / 'KAISTDataset' / dataset_name / f'lidar_odometry_vlp_{side}.csv'


def _log(verbosity: int, level: int, *values) -> None:
    if verbosity >= level:
        print(*values)


##################################################
# Argument parser
##################################################


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compute and save relative KAIST VLP LiDAR odometry.')

    parser.add_argument('dataset_root', type=Path, help='Path to one KAIST UrbanXX directory, for example /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16.')
    parser.add_argument('--side', choices=('left', 'right'), default='left', help='KAIST VLP sensor to process.')
    parser.add_argument('--output', type=Path, default=None, help='Output CSV path. Default: data/KAISTDataset/<UrbanXX>/lidar_odometry_vlp_<side>.csv.')
    parser.add_argument('--verbosity', '-v', type=int, choices=(0, 1, 2), default=1, help='Console verbosity level.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='Registration device.')

    parser.add_argument('--voxel-size-m', type=float, default=0.5, help='Base voxel size used by multiscale ICP.')
    parser.add_argument('--max-correspondence-distance-m', type=float, default=1.5, help='Base ICP correspondence distance.')
    parser.add_argument('--max-iterations', type=int, default=50, help='Total ICP iteration budget.')
    parser.add_argument('--local-map-scans', type=int, default=5, help='Number of recent registered scans used as the local ICP submap.')
    parser.add_argument('--voxel-scale-factors', type=float, nargs='+', default=(2.0, 1.0, 0.5), metavar='SCALE', help='Coarse-to-fine voxel/correspondence scale factors.')
    parser.add_argument('--robust-kernel', type=_parse_robust_kernel, default='tukey', help='ICP robust kernel: tukey, huber, cauchy, l2, or none.')
    parser.add_argument('--robust-kernel-scale-factor', type=float, default=1.0, help='Robust-kernel scale relative to the current ICP voxel size.')
    parser.add_argument('--workers', type=int, default=1, help='Requested CPU worker count. Sequential local-map odometry may force this to one.')

    parser.add_argument('--previous-motion-initialization', action=argparse.BooleanOptionalAction, default=True, help='Initialize each registration from the previous accepted relative motion.')
    parser.add_argument('--bidirectional-check', action=argparse.BooleanOptionalAction, default=False, help='Also solve reverse registration for closure diagnostics.')
    parser.add_argument('--extract-if-needed', action=argparse.BooleanOptionalAction, default=True, help='Extract supported LiDAR archives if no already-extracted scans are available.')

    return parser


##################################################
# Processing
##################################################


def run(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {dataset_root}')

    lidar_root = _lidar_root(dataset_root, args.side)
    output_path = _default_output_path(dataset_root, args.side) if args.output is None else args.output.expanduser()

    _log(args.verbosity, 1, f'Dataset: {dataset_root}')
    _log(args.verbosity, 1, f'LiDAR side: {args.side}')
    _log(args.verbosity, 1, f'Raw LiDAR root: {lidar_root}')
    _log(args.verbosity, 1, f'Output CSV: {output_path}')

    _log(args.verbosity, 2, f'Device: {args.device}')
    _log(args.verbosity, 2, f'Voxel size: {args.voxel_size_m} m')
    _log(args.verbosity, 2, f'Max correspondence distance: {args.max_correspondence_distance_m} m')
    _log(args.verbosity, 2, f'Max iterations: {args.max_iterations}')
    _log(args.verbosity, 2, f'Local map scans: {args.local_map_scans}')
    _log(args.verbosity, 2, f'Voxel scale factors: {tuple(args.voxel_scale_factors)}')
    _log(args.verbosity, 2, f'Previous-motion initialization: {args.previous_motion_initialization}')
    _log(args.verbosity, 2, f'Robust kernel: {args.robust_kernel}')
    _log(args.verbosity, 2, f'Robust kernel scale factor: {args.robust_kernel_scale_factor}')
    _log(args.verbosity, 2, f'Bidirectional check: {args.bidirectional_check}')
    _log(args.verbosity, 2, f'Requested workers: {args.workers}')

    lidar_data = load_lidar_relative_poses(
        lidar_root,
        voxel_size_m=args.voxel_size_m,
        max_correspondence_distance_m=args.max_correspondence_distance_m,
        max_iterations=args.max_iterations,
        extract_if_needed=args.extract_if_needed,
        number_of_workers=args.workers,
        local_map_scans=args.local_map_scans,
        voxel_scale_factors=tuple(args.voxel_scale_factors),
        use_previous_motion_initialization=args.previous_motion_initialization,
        robust_kernel=args.robust_kernel,
        robust_kernel_scale_factor=args.robust_kernel_scale_factor,
        bidirectional_check=args.bidirectional_check,
        device=args.device,
    )

    saved_path = save_lidar_data_csv(lidar_data, output_path)

    _log(args.verbosity, 1, f'Saved LiDAR odometry: {saved_path}')
    _log(args.verbosity, 1, f'Interval poses: {lidar_data.relative_poses_se3.shape[0]}')

    if args.verbosity >= 2:
        if lidar_data.fitness is not None and len(lidar_data.fitness):
            print(f'ICP fitness median: {float(np.nanmedian(lidar_data.fitness)):.6f}')

        if lidar_data.inlier_rmse is not None and len(lidar_data.inlier_rmse):
            print(f'ICP RMSE median: {float(np.nanmedian(lidar_data.inlier_rmse)):.6f} m')

        if lidar_data.scan_timestamps_s is not None and len(lidar_data.scan_timestamps_s):
            print(f'Scan timestamp range: {float(lidar_data.scan_timestamps_s[0]):.9f} .. {float(lidar_data.scan_timestamps_s[-1]):.9f}')

        if lidar_data.metadata:
            workers_used = lidar_data.metadata.get('number_of_workers_used')

            if workers_used is not None:
                print(f'Workers actually used: {workers_used}')

    return saved_path


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

# python ./kaist_dataset/import_lidar_odometry.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --side left --device cpu --voxel-size-m 0.5 --max-correspondence-distance-m 1.5 --max-iterations 50 --local-map-scans 5 --voxel-scale-factors 2.0 1.0 0.5 --previous-motion-initialization --robust-kernel tukey --workers 24 --verbosity 1
# python ./kaist_dataset/import_lidar_odometry.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --side right --device cpu --local-map-scans 5 --workers 24 --verbosity 1