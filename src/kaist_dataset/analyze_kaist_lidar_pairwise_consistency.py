'''Diagnose pairwise consistency of KAIST left/right LiDAR absolute map-pose streams.

The script compares the two independently estimated LiDAR world-pose streams without imposing any rigidity constraint on the factor graph.

For synchronized timestamps it computes

    T_L0_L1(t) = inv(T_W_L0(t)) @ T_W_L1(t)

and measures how much this relative transform varies over the sequence.

This is only a diagnostic. A changing relative transform is not treated as an error by assumption: it may reflect a real sensor motion, inconsistent scan-to-map measurements, timing mismatch, or another sequence-dependent effect.

Default output for one dataset:

    <dataset_root>/tmp/pairwise_lidar_consistency/
        pairwise_consistency_samples.csv
        pairwise_consistency_summary.csv
        pairwise_relative_transform_center.csv
        pairwise_consistency.png

Example:

    python analyze_kaist_lidar_pairwise_consistency.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban08
'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


##################################################
# Project discovery
##################################################


def find_project_root(start_path: Path) -> Path:
    '''Find the repository root containing src/kaist_dataset.'''

    start_path = start_path.resolve()
    candidates = [start_path, *start_path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    visited = set()

    for candidate in candidates:
        if candidate in visited:
            continue

        visited.add(candidate)

        if (candidate / 'src' / 'kaist_dataset').is_dir():
            return candidate

    raise FileNotFoundError("Could not locate the project root containing 'src/kaist_dataset'. Run this script from inside the project or place it inside the repository.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
SRC_ROOT = PROJECT_ROOT / 'src'

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


##################################################
# Project imports
##################################################


from kaist_dataset.lidar_map import load_lidar_map_pose_csv


##################################################
# Generic helpers
##################################################


def load_successful_lidar_map_poses(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    '''Load successful finite absolute LiDAR poses T_W_L from one map-pose CSV.'''

    timestamps_s, poses_T_W_L, dataframe = load_lidar_map_pose_csv(csv_path)
    timestamps_s = np.asarray(timestamps_s, dtype=float).reshape(-1)
    poses_T_W_L = np.asarray(poses_T_W_L, dtype=float)

    if poses_T_W_L.ndim != 3 or poses_T_W_L.shape[1:] != (4, 4):
        raise ValueError(f'Expected LiDAR poses with shape (N, 4, 4), got {poses_T_W_L.shape} in {csv_path}')

    if len(timestamps_s) != len(poses_T_W_L):
        raise ValueError(f'LiDAR timestamp/pose count mismatch in {csv_path}: {len(timestamps_s)} timestamps, {len(poses_T_W_L)} poses')

    success_mask = dataframe['success'].to_numpy(dtype=bool) if 'success' in dataframe.columns else np.ones(len(dataframe), dtype=bool)
    finite_mask = np.all(np.isfinite(poses_T_W_L), axis=(1, 2))
    mask = success_mask & finite_mask & np.isfinite(timestamps_s)

    selected_timestamps = timestamps_s[mask]
    selected_poses = poses_T_W_L[mask]

    if len(selected_timestamps) < 2:
        raise ValueError(f'LiDAR map-pose CSV contains fewer than two successful finite poses: {csv_path}')

    if np.any(np.diff(selected_timestamps) <= 0.0):
        raise ValueError(f'LiDAR map-pose timestamps must be strictly increasing: {csv_path}')

    return selected_timestamps, selected_poses


def processed_data_directory_has_pair(path: Path) -> bool:
    '''Return whether a directory contains both left and right LiDAR map-pose CSVs.'''

    return (path / 'lidar_map_poses_vlp_left.csv').is_file() and (path / 'lidar_map_poses_vlp_right.csv').is_file()


def resolve_processed_data_directory(dataset_root: Path, requested_path: Path | None) -> Path:
    '''Resolve the directory containing the paired LiDAR map-pose CSV files.'''

    if requested_path is not None:
        processed_data_dir = requested_path.expanduser()

        if not processed_data_dir.is_absolute():
            processed_data_dir = dataset_root / processed_data_dir

        processed_data_dir = processed_data_dir.resolve()

        if not processed_data_directory_has_pair(processed_data_dir):
            raise FileNotFoundError(f'Processed-data directory does not contain both LiDAR map-pose CSV files: {processed_data_dir}')

        return processed_data_dir

    dataset_name = dataset_root.name
    candidates = [dataset_root, dataset_root / 'data', PROJECT_ROOT / 'data' / 'KAISTDataset' / dataset_name]

    for candidate in candidates:
        if processed_data_directory_has_pair(candidate):
            return candidate.resolve()

    candidate_text = '\n'.join(f'  {candidate}' for candidate in candidates)
    raise FileNotFoundError(f'Could not find both lidar_map_poses_vlp_left.csv and lidar_map_poses_vlp_right.csv.\nSearched:\n{candidate_text}\nPass the directory explicitly using --processed-data-dir.')


def resolve_output_directory(dataset_root: Path, requested_path: Path | None) -> Path:
    '''Resolve the diagnostic output directory.'''

    if requested_path is None:
        return (dataset_root / 'tmp' / 'pairwise_lidar_consistency').resolve()

    output_dir = requested_path.expanduser()

    if not output_dir.is_absolute():
        output_dir = dataset_root / output_dir

    return output_dir.resolve()


def robust_summary(values: np.ndarray) -> dict[str, float]:
    '''Calculate compact robust and conventional statistics for one non-negative deviation array.'''

    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {name: np.nan for name in ('mean', 'median', 'rmse', 'p90', 'p95', 'max', 'mad', 'robust_sigma')}

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    return {
        'mean': float(np.mean(values)),
        'median': median,
        'rmse': float(np.sqrt(np.mean(values**2))),
        'p90': float(np.percentile(values, 90.0)),
        'p95': float(np.percentile(values, 95.0)),
        'max': float(np.max(values)),
        'mad': mad,
        'robust_sigma': 1.4826 * mad,
    }


##################################################
# Pose interpolation
##################################################


def typical_sample_period(timestamps_s: np.ndarray) -> float:
    '''Return the median positive timestamp spacing.'''

    time_differences = np.diff(np.asarray(timestamps_s, dtype=float).reshape(-1))
    positive_time_differences = time_differences[time_differences > 0.0]

    if len(positive_time_differences) == 0:
        raise ValueError('Cannot estimate sample period from fewer than two increasing timestamps.')

    return float(np.median(positive_time_differences))


def interpolate_pose_with_gap_check(timestamps_s: np.ndarray, poses: np.ndarray, query_timestamp_s: float, maximum_gap_s: float) -> tuple[np.ndarray | None, float]:
    '''Interpolate one SE(3) pose while rejecting interpolation across a large source-data gap.'''

    timestamps_s = np.asarray(timestamps_s, dtype=float).reshape(-1)
    poses = np.asarray(poses, dtype=float)

    if query_timestamp_s < timestamps_s[0] or query_timestamp_s > timestamps_s[-1]:
        return None, np.nan

    right_index = int(np.searchsorted(timestamps_s, query_timestamp_s, side='left'))

    if right_index < len(timestamps_s) and np.isclose(timestamps_s[right_index], query_timestamp_s, rtol=0.0, atol=1e-9):
        return poses[right_index].copy(), 0.0

    if right_index == 0 or right_index >= len(timestamps_s):
        return None, np.nan

    left_index = right_index - 1
    time_left = float(timestamps_s[left_index])
    time_right = float(timestamps_s[right_index])
    interpolation_gap_s = time_right - time_left

    if interpolation_gap_s <= 0.0 or interpolation_gap_s > maximum_gap_s:
        return None, interpolation_gap_s

    alpha = (query_timestamp_s - time_left) / interpolation_gap_s
    interpolated_pose = np.eye(4, dtype=float)
    interpolated_pose[:3, 3] = (1.0 - alpha) * poses[left_index, :3, 3] + alpha * poses[right_index, :3, 3]

    rotations = Rotation.from_matrix(np.stack((poses[left_index, :3, :3], poses[right_index, :3, :3]), axis=0))
    interpolated_pose[:3, :3] = Slerp([time_left, time_right], rotations)([query_timestamp_s]).as_matrix()[0]

    return interpolated_pose, interpolation_gap_s


def synchronize_lidar_pose_streams(left_timestamps_s: np.ndarray, left_poses: np.ndarray, right_timestamps_s: np.ndarray, right_poses: np.ndarray, sampling_stream: str, maximum_interpolation_gap_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Synchronize the two absolute pose streams by interpolating the opposite stream at timestamps of the selected sampling stream.'''

    overlap_start_s = max(float(left_timestamps_s[0]), float(right_timestamps_s[0]))
    overlap_end_s = min(float(left_timestamps_s[-1]), float(right_timestamps_s[-1]))

    if overlap_end_s <= overlap_start_s:
        raise ValueError('Left and right LiDAR map-pose streams do not overlap in time.')

    if sampling_stream == 'left':
        source_timestamps = left_timestamps_s
        source_poses = left_poses
        other_timestamps = right_timestamps_s
        other_poses = right_poses
    elif sampling_stream == 'right':
        source_timestamps = right_timestamps_s
        source_poses = right_poses
        other_timestamps = left_timestamps_s
        other_poses = left_poses
    else:
        raise ValueError(f'Unknown sampling stream: {sampling_stream}')

    source_mask = (source_timestamps >= overlap_start_s) & (source_timestamps <= overlap_end_s)
    query_timestamps = source_timestamps[source_mask]
    query_source_poses = source_poses[source_mask]

    selected_timestamps = []
    synchronized_left_poses = []
    synchronized_right_poses = []
    interpolation_gaps_s = []

    for query_timestamp_s, source_pose in zip(query_timestamps, query_source_poses):
        interpolated_pose, interpolation_gap_s = interpolate_pose_with_gap_check(other_timestamps, other_poses, float(query_timestamp_s), maximum_interpolation_gap_s)

        if interpolated_pose is None:
            continue

        selected_timestamps.append(float(query_timestamp_s))
        interpolation_gaps_s.append(float(interpolation_gap_s))

        if sampling_stream == 'left':
            synchronized_left_poses.append(source_pose)
            synchronized_right_poses.append(interpolated_pose)
        else:
            synchronized_left_poses.append(interpolated_pose)
            synchronized_right_poses.append(source_pose)

    if len(selected_timestamps) < 2:
        raise RuntimeError('Fewer than two synchronized left/right LiDAR pose pairs remained after overlap and interpolation-gap filtering.')

    return np.asarray(selected_timestamps, dtype=float), np.asarray(synchronized_left_poses, dtype=float), np.asarray(synchronized_right_poses, dtype=float), np.asarray(interpolation_gaps_s, dtype=float)


##################################################
# Pairwise consistency
##################################################


def calculate_pairwise_relative_poses(left_poses: np.ndarray, right_poses: np.ndarray) -> np.ndarray:
    '''Calculate T_L0_L1 = inv(T_W_L0) @ T_W_L1 for every synchronized pose pair.'''

    relative_poses = np.empty_like(left_poses)

    for pose_index, (left_pose, right_pose) in enumerate(zip(left_poses, right_poses)):
        relative_poses[pose_index] = np.linalg.inv(left_pose) @ right_pose

    return relative_poses


def central_pairwise_transform(relative_poses: np.ndarray) -> np.ndarray:
    '''Calculate a descriptive sequence-level central relative transform.'''

    central_pose = np.eye(4, dtype=float)
    central_pose[:3, 3] = np.median(relative_poses[:, :3, 3], axis=0)
    central_pose[:3, :3] = Rotation.from_matrix(relative_poses[:, :3, :3]).mean().as_matrix()

    return central_pose


def calculate_consistency_residuals(relative_poses: np.ndarray, central_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Calculate component and magnitude deviations from the descriptive central relative transform.'''

    central_inverse = np.linalg.inv(central_pose)
    residual_rotation_vectors_deg = np.empty((len(relative_poses), 3), dtype=float)
    residual_translations_m = np.empty((len(relative_poses), 3), dtype=float)

    for pose_index, relative_pose in enumerate(relative_poses):
        residual_pose = central_inverse @ relative_pose
        residual_rotation_vectors_deg[pose_index] = np.rad2deg(Rotation.from_matrix(residual_pose[:3, :3]).as_rotvec())
        residual_translations_m[pose_index] = residual_pose[:3, 3]

    rotation_deviation_deg = np.linalg.norm(residual_rotation_vectors_deg, axis=1)
    translation_deviation_m = np.linalg.norm(residual_translations_m, axis=1)

    return residual_rotation_vectors_deg, residual_translations_m, rotation_deviation_deg, translation_deviation_m


##################################################
# Saving
##################################################


def save_sample_table(path: Path, dataset_name: str, timestamps_s: np.ndarray, interpolation_gaps_s: np.ndarray, relative_poses: np.ndarray, residual_rotation_vectors_deg: np.ndarray, residual_translations_m: np.ndarray, rotation_deviation_deg: np.ndarray, translation_deviation_m: np.ndarray) -> Path:
    '''Save synchronized pairwise relative-pose samples and consistency residuals.'''

    data: dict[str, Any] = {
        'dataset': np.repeat(dataset_name, len(timestamps_s)),
        'timestamp_s': timestamps_s,
        'time_from_start_s': timestamps_s - timestamps_s[0],
        'interpolation_gap_s': interpolation_gaps_s,
        'rotation_deviation_deg': rotation_deviation_deg,
        'translation_deviation_m': translation_deviation_m,
        'residual_rx_deg': residual_rotation_vectors_deg[:, 0],
        'residual_ry_deg': residual_rotation_vectors_deg[:, 1],
        'residual_rz_deg': residual_rotation_vectors_deg[:, 2],
        'residual_tx_m': residual_translations_m[:, 0],
        'residual_ty_m': residual_translations_m[:, 1],
        'residual_tz_m': residual_translations_m[:, 2],
    }

    for row_index in range(4):
        for column_index in range(4):
            data[f'T_L0_L1_{row_index}{column_index}'] = relative_poses[:, row_index, column_index]

    pd.DataFrame(data).to_csv(path, index=False, float_format='%.12g')
    return path


def save_summary_table(path: Path, dataset_name: str, timestamps_s: np.ndarray, interpolation_gaps_s: np.ndarray, rotation_deviation_deg: np.ndarray, translation_deviation_m: np.ndarray) -> Path:
    '''Save sequence-level pairwise consistency statistics.'''

    rows = []

    for metric_name, unit, values in (('pairwise_rotation_deviation_deg', 'deg', rotation_deviation_deg), ('pairwise_translation_deviation_m', 'm', translation_deviation_m)):
        statistics = robust_summary(values)
        rows.append({'dataset': dataset_name, 'metric': metric_name, 'unit': unit, 'samples': int(len(values)), 'duration_s': float(timestamps_s[-1] - timestamps_s[0]), 'interpolation_gap_median_s': float(np.median(interpolation_gaps_s)), 'interpolation_gap_max_s': float(np.max(interpolation_gaps_s)), **statistics})

    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.12g')
    return path


def save_central_transform(path: Path, dataset_name: str, central_pose: np.ndarray) -> Path:
    '''Save the descriptive central T_L0_L1 transform as one flattened CSV row.'''

    row: dict[str, Any] = {'dataset': dataset_name}

    for row_index in range(4):
        for column_index in range(4):
            row[f'T_L0_L1_{row_index}{column_index}'] = float(central_pose[row_index, column_index])

    pd.DataFrame([row]).to_csv(path, index=False, float_format='%.12g')
    return path


def plot_pairwise_consistency(path: Path, dataset_name: str, timestamps_s: np.ndarray, residual_rotation_vectors_deg: np.ndarray, residual_translations_m: np.ndarray, rotation_deviation_deg: np.ndarray, translation_deviation_m: np.ndarray, dpi: int) -> Path:
    '''Plot pairwise LiDAR consistency magnitudes and residual components over time.'''

    plot_times_s = timestamps_s - timestamps_s[0]
    fig, axes = plt.subplots(4, 1, figsize=(14.0, 13.0), sharex=True)

    axes[0].plot(plot_times_s, rotation_deviation_deg)
    axes[0].set_ylabel('deviation [deg]')
    axes[0].set_title('Pairwise LiDAR rotation deviation from sequence center')
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(plot_times_s, translation_deviation_m)
    axes[1].set_ylabel('deviation [m]')
    axes[1].set_title('Pairwise LiDAR translation deviation from sequence center')
    axes[1].grid(True, alpha=0.25)

    component_names = ('x', 'y', 'z')

    for component_index, component_name in enumerate(component_names):
        axes[2].plot(plot_times_s, residual_rotation_vectors_deg[:, component_index], label=component_name)

    axes[2].set_ylabel('residual [deg]')
    axes[2].set_title('Pairwise relative-rotation residual components')
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    for component_index, component_name in enumerate(component_names):
        axes[3].plot(plot_times_s, residual_translations_m[:, component_index], label=component_name)

    axes[3].set_ylabel('residual [m]')
    axes[3].set_xlabel('time from start [s]')
    axes[3].set_title('Pairwise relative-translation residual components')
    axes[3].grid(True, alpha=0.25)
    axes[3].legend()

    fig.suptitle(f'Left/right LiDAR map-pose pairwise consistency: {dataset_name}')
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


##################################################
# Argument parser
##################################################


def build_argument_parser() -> argparse.ArgumentParser:
    '''Construct the command-line interface.'''

    parser = argparse.ArgumentParser(description='Diagnose pairwise consistency of KAIST left/right LiDAR absolute map-pose streams.')
    parser.add_argument('dataset_root', type=Path, help='Path to one KAIST Urban dataset root, for example /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban08.')
    parser.add_argument('--processed-data-dir', type=Path, default=None, help='Directory containing lidar_map_poses_vlp_left.csv and lidar_map_poses_vlp_right.csv. Relative paths are interpreted relative to dataset_root.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory. Default: <dataset_root>/tmp/pairwise_lidar_consistency. Relative paths are interpreted relative to dataset_root.')
    parser.add_argument('--sampling-stream', choices=('left', 'right'), default='left', help='Use timestamps of this stream as synchronization query timestamps.')
    parser.add_argument('--max-gap-factor', type=float, default=5.0, help='Reject interpolation across a gap larger than this factor times the median sample period of the interpolated stream.')
    parser.add_argument('--max-interpolation-gap-s', type=float, default=None, help='Explicit maximum interpolation gap in seconds. Overrides --max-gap-factor.')
    parser.add_argument('--dpi', type=int, default=180, help='Saved plot resolution.')
    return parser


##################################################
# Main processing
##################################################


def run(args: argparse.Namespace) -> None:
    '''Load paired map-pose streams, calculate consistency diagnostics, and save results.'''

    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {dataset_root}')

    dataset_name = dataset_root.name
    processed_data_dir = resolve_processed_data_directory(dataset_root, args.processed_data_dir)
    output_dir = resolve_output_directory(dataset_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_csv_path = processed_data_dir / 'lidar_map_poses_vlp_left.csv'
    right_csv_path = processed_data_dir / 'lidar_map_poses_vlp_right.csv'

    left_timestamps_s, left_poses = load_successful_lidar_map_poses(left_csv_path)
    right_timestamps_s, right_poses = load_successful_lidar_map_poses(right_csv_path)

    interpolated_stream_timestamps = right_timestamps_s if args.sampling_stream == 'left' else left_timestamps_s
    default_maximum_gap_s = args.max_gap_factor * typical_sample_period(interpolated_stream_timestamps)
    maximum_interpolation_gap_s = float(args.max_interpolation_gap_s) if args.max_interpolation_gap_s is not None else float(default_maximum_gap_s)

    timestamps_s, synchronized_left_poses, synchronized_right_poses, interpolation_gaps_s = synchronize_lidar_pose_streams(left_timestamps_s, left_poses, right_timestamps_s, right_poses, args.sampling_stream, maximum_interpolation_gap_s)

    relative_poses = calculate_pairwise_relative_poses(synchronized_left_poses, synchronized_right_poses)
    central_pose = central_pairwise_transform(relative_poses)
    residual_rotation_vectors_deg, residual_translations_m, rotation_deviation_deg, translation_deviation_m = calculate_consistency_residuals(relative_poses, central_pose)

    sample_path = output_dir / 'pairwise_consistency_samples.csv'
    summary_path = output_dir / 'pairwise_consistency_summary.csv'
    central_transform_path = output_dir / 'pairwise_relative_transform_center.csv'
    plot_path = output_dir / 'pairwise_consistency.png'

    save_sample_table(sample_path, dataset_name, timestamps_s, interpolation_gaps_s, relative_poses, residual_rotation_vectors_deg, residual_translations_m, rotation_deviation_deg, translation_deviation_m)
    save_summary_table(summary_path, dataset_name, timestamps_s, interpolation_gaps_s, rotation_deviation_deg, translation_deviation_m)
    save_central_transform(central_transform_path, dataset_name, central_pose)
    plot_pairwise_consistency(plot_path, dataset_name, timestamps_s, residual_rotation_vectors_deg, residual_translations_m, rotation_deviation_deg, translation_deviation_m, args.dpi)

    rotation_statistics = robust_summary(rotation_deviation_deg)
    translation_statistics = robust_summary(translation_deviation_m)

    print()
    print('KAIST LiDAR pairwise consistency diagnostics')
    print('============================================')
    print(f'dataset:                     {dataset_name}')
    print(f'processed data:              {processed_data_dir}')
    print(f'sampling stream:             {args.sampling_stream}')
    print(f'synchronized pairs:          {len(timestamps_s)}')
    print(f'duration [s]:                {timestamps_s[-1] - timestamps_s[0]:.3f}')
    print(f'max interpolation gap [s]:   {maximum_interpolation_gap_s:.6g}')
    print()
    print('Pairwise rotation deviation')
    print(f'  median [deg]:               {rotation_statistics["median"]:.6g}')
    print(f'  RMSE [deg]:                 {rotation_statistics["rmse"]:.6g}')
    print(f'  p90 [deg]:                  {rotation_statistics["p90"]:.6g}')
    print(f'  robust sigma [deg]:         {rotation_statistics["robust_sigma"]:.6g}')
    print()
    print('Pairwise translation deviation')
    print(f'  median [m]:                 {translation_statistics["median"]:.6g}')
    print(f'  RMSE [m]:                   {translation_statistics["rmse"]:.6g}')
    print(f'  p90 [m]:                    {translation_statistics["p90"]:.6g}')
    print(f'  robust sigma [m]:           {translation_statistics["robust_sigma"]:.6g}')
    print()
    print('Saved outputs')
    print('-------------')
    print(sample_path)
    print(summary_path)
    print(central_transform_path)
    print(plot_path)


##################################################
# Entrypoint
##################################################


def main() -> None:
    '''Parse arguments and run the diagnostic.'''

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.max_gap_factor <= 0.0:
        parser.error('--max-gap-factor must be positive.')

    if args.max_interpolation_gap_s is not None and args.max_interpolation_gap_s <= 0.0:
        parser.error('--max-interpolation-gap-s must be positive.')

    if args.dpi <= 0:
        parser.error('--dpi must be positive.')

    run(args)


if __name__ == '__main__':
    main()

# python analyze_kaist_lidar_pairwise_consistency.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban08