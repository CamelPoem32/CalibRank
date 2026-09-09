'''Aggregate KAIST multi-sequence pipeline results and compare calibration against fixed calibration parameters.

The script inspects only the direct child directories of DATASET_ROOT. Each
child directory is treated as one dataset, for example:

    KAISTDataset/
        Urban13/
        Urban14/
        Urban15/
        Urban16/
        Urban17/

For a selected LiDAR mode, the expected input layout inside every dataset is:

    <dataset>/
        experiment_results/
            errors/
                poses/
                    with_calibration/
                        trajectory_errors.pkl
                        trajectory.csv
                    without_calibration/
                        trajectory_errors.pkl
                        trajectory.csv

or:

    <dataset>/
        experiment_results/
            errors/
                odometry/
                    with_calibration/
                        trajectory_errors.pkl
                        trajectory.csv
                    without_calibration/
                        trajectory_errors.pkl
                        trajectory.csv

Only direct children of DATASET_ROOT are inspected. Dataset discovery is not
recursive.

The raw trajectory errors stored in the pickle are reduced to one set of
statistics per dataset. This keeps each driving sequence as the primary
experimental unit when averaging improvements across datasets.

The six localization benchmark metrics are calculated from trajectory.csv in
the same directory as each pickle. Each CSV row stores one timestamp together
with the estimated and timestamp-aligned reference 4x4 SE(3) matrices.

For an error metric e, calibration gain is defined as

    absolute_gain = e_without_calibration - e_with_calibration

and

    relative_gain_percent =
        100 * absolute_gain / e_without_calibration.

Therefore positive gain always means that calibration improved the result.

Default output:

    <dataset_root>/pipeline_results_summary/<lidar_mode>/
        with_calibration_metrics.csv
        without_calibration_metrics.csv
        calibration_gains.csv
        calibration_gains_long.csv
        calibration_gain_summary.csv
        plots/
            comparison_median.png
            comparison_rmse.png
            comparison_p90.png
            gain_median.png
            gain_rmse.png
            gain_p90.png
            average_gain_median.png
            average_gain_rmse.png
            average_gain_p90.png
        plots_metrics/
            localization_metrics.csv
            localization_metrics_summary.csv
            comparison_<statistic>.png
            gain_<statistic>.png
            average_values_<statistic>.png
            average_gain_<statistic>.png

Example:

    python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset

Example for LiDAR odometry:

    python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset --lidar-mode odometry

Example with a custom output directory:

    python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset --lidar-mode poses --output-dir /home/camel/Skoltech/phd_proposal/kaist_summary

Example selecting different statistics:

    python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset --statistics median mean rmse p90 p10
'''

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import re
import warnings
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from kaist_dataset.data import import_true_trajectory


##################################################
# Error definitions
##################################################


ERROR_TYPES = {
    'full_se3': {
        'title': 'Full SE(3) tangent error',
        'ylabel': r'$\|\mathrm{Log}(T_{ref}^{-1}T_{est})\|_2$',
    },
    'rotation_deg': {
        'title': 'Rotational error',
        'ylabel': 'error [deg]',
    },
    'translation_m': {
        'title': 'Translational error',
        'ylabel': 'error [m]',
    },
}

LOCALIZATION_METRIC_TYPES = {
    'absolute_translation_m': {
        'title': 'Absolute translation error',
        'ylabel': 'error [m]',
    },
    'absolute_rotation_deg': {
        'title': 'Absolute orientation error',
        'ylabel': 'error [deg]',
    },
    'translation_rpe_m': {
        'title': 'Translational RPE',
        'ylabel': 'error [m]',
    },
    'rotation_rpe_deg': {
        'title': 'Rotational RPE',
        'ylabel': 'error [deg]',
    },
    'translation_rte_m': {
        'title': 'Translational distance RTE',
        'ylabel': 'error [m]',
    },
    'rotation_rte_deg': {
        'title': 'Rotational distance RTE',
        'ylabel': 'error [deg]',
    },
}

DEFAULT_RPE_DELTA_TIME_S = 1.0
DEFAULT_RTE_DISTANCE_M = 100.0

CALIBRATION_DEVIATION_TYPES = {
    'extrinsic_rotation_deg': {
        'title': 'Extrinsic rotation deviation',
        'ylabel': 'deviation [deg]',
        'unit': 'deg',
    },
    'extrinsic_translation_m': {
        'title': 'Extrinsic translation deviation',
        'ylabel': 'deviation [m]',
        'unit': 'm',
    },
    'time_offset_ms': {
        'title': 'Time-offset deviation',
        'ylabel': 'deviation [ms]',
        'unit': 'ms',
    },
    'gyro_bias_radps': {
        'title': 'Gyroscope-bias deviation',
        'ylabel': 'deviation [rad/s]',
        'unit': 'rad/s',
    },
}

AVAILABLE_STATISTICS = ('median', 'mean', 'rmse', 'p90', 'p10')

RUN_CONFIG_COMPARE_KEYS = (
    'lidar_mode',
    'acc_mode',
    'window_size_s',
    'step_size_s',
    'imu_frequency_hz',
    'map_pose_factor_stride',
    'gyro_information',
    'accel_information',
    'lidar_rotation_information',
    'lidar_translation_information',
    'map_pose_rotation_information',
    'map_pose_translation_information',
    'gravity_z_mps2',
)


##################################################
# Generic helpers
##################################################


def natural_sort_key(value: str) -> list[Any]:
    '''Return a natural sorting key so Urban2 appears before Urban10.'''

    return [int(component) if component.isdigit() else component.lower() for component in re.split(r'(\d+)', value)]


def resolve_output_directory(dataset_root: Path, requested_path: Path | None, lidar_mode: str) -> Path:
    '''Resolve the summary output directory.'''

    if requested_path is None:
        return (dataset_root / 'pipeline_results_summary' / lidar_mode).resolve()

    output_dir = requested_path.expanduser()

    if not output_dir.is_absolute():
        output_dir = dataset_root / output_dir

    return output_dir.resolve()


def load_result_pickle(path: Path) -> dict[str, Any]:
    '''Load one trusted pipeline result pickle.'''

    with path.open('rb') as stream:
        payload = pickle.load(stream)

    if not isinstance(payload, dict):
        raise ValueError(f'Expected dictionary payload in {path}, got {type(payload).__name__}')

    return payload


def finite_error_array(payload: dict[str, Any], error_name: str, path: Path) -> np.ndarray:
    '''Extract one finite one-dimensional error array.'''

    if 'errors' not in payload or not isinstance(payload['errors'], dict):
        raise ValueError(f'Pickle does not contain an errors dictionary: {path}')

    if error_name not in payload['errors']:
        raise ValueError(f'Pickle does not contain errors["{error_name}"]: {path}')

    values = np.asarray(payload['errors'][error_name], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(f'Error array "{error_name}" contains no finite samples: {path}')

    return values


def compute_statistic(values: np.ndarray, statistic: str) -> float:
    '''Calculate one requested summary statistic.'''

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    if statistic == 'median':
        return float(np.median(values))

    if statistic == 'mean':
        return float(np.mean(values))

    if statistic == 'rmse':
        return float(np.sqrt(np.mean(values**2)))

    if statistic.startswith('p'):
        percentile = float(statistic[1:])
        return float(np.percentile(values, percentile))

    raise ValueError(f'Unknown statistic: {statistic}')


def values_match(value_a: Any, value_b: Any) -> bool:
    '''Compare simple run-configuration values robustly.'''

    if isinstance(value_a, (int, float, np.integer, np.floating)) and isinstance(value_b, (int, float, np.integer, np.floating)):
        return bool(np.isclose(float(value_a), float(value_b), rtol=1e-10, atol=1e-12, equal_nan=True))

    return value_a == value_b


def compare_run_configs(with_payload: dict[str, Any], without_payload: dict[str, Any]) -> list[str]:
    '''Return important configuration mismatches between paired runs.'''

    with_config = with_payload.get('run_config', {})
    without_config = without_payload.get('run_config', {})

    if not isinstance(with_config, dict) or not isinstance(without_config, dict):
        return []

    mismatches = []

    for key in RUN_CONFIG_COMPARE_KEYS:
        if key not in with_config or key not in without_config:
            continue

        if not values_match(with_config[key], without_config[key]):
            mismatches.append(f'{key}: with={with_config[key]!r}, without={without_config[key]!r}')

    return mismatches


def validate_result_payload(payload: dict[str, Any], *, path: Path, dataset_name: str, calibration_mode: str, lidar_mode: str) -> None:
    '''Validate one pipeline result pickle against its expected location.'''

    expected_payload_mode = calibration_mode.replace('_', '-')

    stored_dataset_name = payload.get('dataset_name')

    if stored_dataset_name is not None and str(stored_dataset_name) != dataset_name:
        warnings.warn(f'Dataset name mismatch in {path}: directory={dataset_name!r}, pickle={stored_dataset_name!r}')

    stored_mode = payload.get('mode')

    if stored_mode is not None and str(stored_mode) not in {calibration_mode, expected_payload_mode}:
        warnings.warn(f'Calibration mode mismatch in {path}: expected {calibration_mode!r}, pickle={stored_mode!r}')

    stored_lidar_mode = payload.get('lidar_mode')

    if stored_lidar_mode is not None and str(stored_lidar_mode) != lidar_mode:
        raise ValueError(f'LiDAR mode mismatch in {path}: expected {lidar_mode!r}, pickle={stored_lidar_mode!r}')

    for error_name in ERROR_TYPES:
        finite_error_array(payload, error_name, path)


##################################################
# Per-dataset metrics
##################################################


def result_to_metric_row(dataset_name: str, payload: dict[str, Any], path: Path, statistics: tuple[str, ...]) -> dict[str, Any]:
    '''Reduce one trajectory-error pickle to one paper-style dataset row.'''

    row: dict[str, Any] = {
        'dataset': dataset_name,
    }

    timestamps = np.asarray(payload.get('timestamps_s', []), dtype=float).reshape(-1)
    finite_timestamps = timestamps[np.isfinite(timestamps)]

    error_arrays = {error_name: finite_error_array(payload, error_name, path) for error_name in ERROR_TYPES}

    sample_counts = [len(values) for values in error_arrays.values()]
    row['samples'] = int(min(sample_counts))

    if len(finite_timestamps) >= 2:
        row['duration_s'] = float(finite_timestamps[-1] - finite_timestamps[0])
    else:
        row['duration_s'] = np.nan

    for error_name, values in error_arrays.items():
        for statistic in statistics:
            row[f'{error_name}_{statistic}'] = compute_statistic(values, statistic)

    return row


##################################################
# Localization benchmark metrics
##################################################


def trajectory_matrix_columns(prefix: str) -> list[str]:
    '''Return the sixteen CSV column names of one flattened 4x4 SE(3) matrix.'''

    return [f'{prefix}_T{row_index}{column_index}' for row_index in range(4) for column_index in range(4)]


def load_trajectory_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Load aligned estimated/reference SE(3) trajectories from trajectory.csv.

    Each row contains one timestamp, one estimated 4x4 pose, and the reference
    4x4 pose interpolated at the same timestamp. This keeps the metric processor
    independent of the original KAIST dataset files once the pipeline run has
    completed.
    '''

    if not path.is_file():
        raise FileNotFoundError(f'Trajectory CSV was not found next to the result pickle: {path}')

    dataframe = pd.read_csv(path)
    estimated_columns = trajectory_matrix_columns('estimated')
    reference_columns = trajectory_matrix_columns('reference')
    required_columns = ['timestamp_s', *estimated_columns, *reference_columns]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(f'Trajectory CSV {path} is missing columns: {", ".join(missing_columns)}')

    timestamps = dataframe['timestamp_s'].to_numpy(dtype=float)
    estimated_poses = dataframe[estimated_columns].to_numpy(dtype=float).reshape(-1, 4, 4)
    reference_poses = dataframe[reference_columns].to_numpy(dtype=float).reshape(-1, 4, 4)

    finite_mask = np.isfinite(timestamps) & np.all(np.isfinite(estimated_poses), axis=(1, 2)) & np.all(np.isfinite(reference_poses), axis=(1, 2))
    timestamps = timestamps[finite_mask]
    estimated_poses = estimated_poses[finite_mask]
    reference_poses = reference_poses[finite_mask]

    if len(timestamps) < 2:
        raise ValueError(f'Trajectory CSV {path} must contain at least two finite trajectory samples')

    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f'Trajectory timestamps in {path} must be strictly increasing')

    return timestamps, estimated_poses, reference_poses


def interpolate_pose_trajectory(source_timestamps: np.ndarray, source_poses: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    '''Interpolate an SE(3) trajectory with linear translation and SO(3) SLERP.'''

    query_timestamps = np.asarray(query_timestamps, dtype=float).reshape(-1)

    if len(query_timestamps) == 0:
        return np.empty((0, 4, 4), dtype=float)

    if np.any(query_timestamps < source_timestamps[0]) or np.any(query_timestamps > source_timestamps[-1]):
        raise ValueError('Pose interpolation query lies outside the source trajectory support')

    time_origin = float(source_timestamps[0])
    source_times_relative = source_timestamps - time_origin
    query_times_relative = query_timestamps - time_origin

    rotation_interpolator = Slerp(source_times_relative, Rotation.from_matrix(source_poses[:, :3, :3]))
    interpolated_rotations = rotation_interpolator(query_times_relative).as_matrix()

    interpolated_translations = np.column_stack([
        np.interp(query_times_relative, source_times_relative, source_poses[:, axis_index, 3])
        for axis_index in range(3)
    ])

    interpolated_poses = np.repeat(np.eye(4, dtype=float)[None, :, :], len(query_timestamps), axis=0)
    interpolated_poses[:, :3, :3] = interpolated_rotations
    interpolated_poses[:, :3, 3] = interpolated_translations

    return interpolated_poses


def invert_se3_array(poses: np.ndarray) -> np.ndarray:
    '''Invert an array of rigid transforms.'''

    poses = np.asarray(poses, dtype=float)
    inverse_poses = np.repeat(np.eye(4, dtype=float)[None, :, :], len(poses), axis=0)

    rotations_transposed = np.transpose(poses[:, :3, :3], (0, 2, 1))
    inverse_poses[:, :3, :3] = rotations_transposed
    inverse_poses[:, :3, 3] = -np.einsum('nij,nj->ni', rotations_transposed, poses[:, :3, 3])

    return inverse_poses


def rotation_angle_deg(rotation_matrices: np.ndarray) -> np.ndarray:
    '''Return SO(3) geodesic rotation angles in degrees.'''

    traces = np.trace(rotation_matrices, axis1=1, axis2=2)
    cosine_angles = np.clip(0.5 * (traces - 1.0), -1.0, 1.0)

    return np.rad2deg(np.arccos(cosine_angles))


def relative_motion_error_arrays(estimated_start_poses: np.ndarray, estimated_end_poses: np.ndarray, reference_start_poses: np.ndarray, reference_end_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Return translational and rotational errors between estimated and reference relative motions.'''

    estimated_deltas = invert_se3_array(estimated_start_poses) @ estimated_end_poses
    reference_deltas = invert_se3_array(reference_start_poses) @ reference_end_poses
    relative_errors = invert_se3_array(reference_deltas) @ estimated_deltas

    translation_errors_m = np.linalg.norm(relative_errors[:, :3, 3], axis=1)
    rotation_errors_deg = rotation_angle_deg(relative_errors[:, :3, :3])

    return translation_errors_m, rotation_errors_deg

def kaist_reference_trajectory_path(dataset_dir: Path) -> Path:
    '''Return the original KAIST global reference trajectory path.'''

    dataset_slug = dataset_dir.name.lower()
    return dataset_dir / f'{dataset_slug}_pose' / dataset_slug / 'global_pose.csv'


def reference_gap_threshold(reference_timestamps: np.ndarray) -> float:
    '''Return a conservative threshold used to identify unusually large reference-trajectory gaps.'''

    time_differences = np.diff(np.asarray(reference_timestamps, dtype=float))
    positive_time_differences = time_differences[time_differences > 0.0]

    if len(positive_time_differences) == 0:
        raise ValueError('Reference trajectory must contain increasing timestamps.')

    return float(5.0 * np.median(positive_time_differences))


def reference_interval_is_continuous(reference_timestamps: np.ndarray, start_timestamp: float, end_timestamp: float, maximum_gap_s: float) -> bool:
    '''Return whether one requested interval avoids unusually large reference timestamp gaps.'''

    first_index = max(0, int(np.searchsorted(reference_timestamps, start_timestamp, side='right')) - 1)
    last_index = min(len(reference_timestamps) - 1, int(np.searchsorted(reference_timestamps, end_timestamp, side='left')))

    if last_index <= first_index:
        return True

    return bool(np.all(np.diff(reference_timestamps[first_index:last_index + 1]) <= maximum_gap_s))


def calculate_temporal_rpe(estimated_timestamps: np.ndarray, estimated_poses: np.ndarray, reference_timestamps: np.ndarray, reference_poses: np.ndarray, delta_time_s: float) -> tuple[np.ndarray, np.ndarray]:
    '''Calculate translational and rotational RPE over a fixed temporal interval.'''

    valid_start_mask = estimated_timestamps + delta_time_s <= min(estimated_timestamps[-1], reference_timestamps[-1])
    start_indices = np.flatnonzero(valid_start_mask)

    if len(start_indices) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    start_timestamps = estimated_timestamps[start_indices]
    end_timestamps = start_timestamps + delta_time_s
    maximum_reference_gap_s = reference_gap_threshold(reference_timestamps)
    continuous_mask = np.asarray([reference_interval_is_continuous(reference_timestamps, start_timestamp, end_timestamp, maximum_reference_gap_s) for start_timestamp, end_timestamp in zip(start_timestamps, end_timestamps)], dtype=bool)

    start_indices = start_indices[continuous_mask]
    start_timestamps = start_timestamps[continuous_mask]
    end_timestamps = end_timestamps[continuous_mask]

    if len(start_indices) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    estimated_start_poses = estimated_poses[start_indices]
    estimated_end_poses = interpolate_pose_trajectory(estimated_timestamps, estimated_poses, end_timestamps)
    reference_start_poses = interpolate_pose_trajectory(reference_timestamps, reference_poses, start_timestamps)
    reference_end_poses = interpolate_pose_trajectory(reference_timestamps, reference_poses, end_timestamps)

    return relative_motion_error_arrays(estimated_start_poses, estimated_end_poses, reference_start_poses, reference_end_poses)


def reference_cumulative_distance(reference_poses: np.ndarray) -> np.ndarray:
    '''Return cumulative 3D distance along a reference trajectory.'''

    segment_distances = np.linalg.norm(np.diff(reference_poses[:, :3, 3], axis=0), axis=1)

    return np.concatenate(([0.0], np.cumsum(segment_distances)))


def timestamp_at_reference_distance(reference_timestamps: np.ndarray, cumulative_distance_m: np.ndarray, target_distance_m: float) -> float | None:
    '''Interpolate the timestamp at which the reference reaches one cumulative distance.'''

    if target_distance_m < cumulative_distance_m[0] or target_distance_m > cumulative_distance_m[-1]:
        return None

    upper_index = int(np.searchsorted(cumulative_distance_m, target_distance_m, side='left'))

    if upper_index == 0:
        return float(reference_timestamps[0])

    if upper_index >= len(cumulative_distance_m):
        return None

    lower_index = upper_index - 1
    lower_distance = float(cumulative_distance_m[lower_index])
    upper_distance = float(cumulative_distance_m[upper_index])

    while upper_index < len(cumulative_distance_m) - 1 and upper_distance <= lower_distance:
        upper_index += 1
        upper_distance = float(cumulative_distance_m[upper_index])

    if upper_distance <= lower_distance:
        return None

    fraction = (target_distance_m - lower_distance) / (upper_distance - lower_distance)

    return float(reference_timestamps[lower_index] + fraction * (reference_timestamps[upper_index] - reference_timestamps[lower_index]))


def calculate_distance_rte(estimated_timestamps: np.ndarray, estimated_poses: np.ndarray, reference_timestamps: np.ndarray, reference_poses: np.ndarray, distance_m: float) -> tuple[np.ndarray, np.ndarray]:
    '''Calculate translational and rotational relative error over fixed reference travel distance.'''

    cumulative_distance_m = reference_cumulative_distance(reference_poses)
    start_distances_m = np.interp(estimated_timestamps, reference_timestamps, cumulative_distance_m)
    maximum_reference_gap_s = reference_gap_threshold(reference_timestamps)

    start_indices = []
    end_timestamps = []

    for start_index, start_distance_m in enumerate(start_distances_m):
        target_distance_m = float(start_distance_m + distance_m)
        end_timestamp = timestamp_at_reference_distance(reference_timestamps, cumulative_distance_m, target_distance_m)

        if end_timestamp is None:
            break

        if end_timestamp > estimated_timestamps[-1]:
            break

        if not reference_interval_is_continuous(reference_timestamps, float(estimated_timestamps[start_index]), end_timestamp, maximum_reference_gap_s):
            continue

        start_indices.append(start_index)
        end_timestamps.append(end_timestamp)

    if not start_indices:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    start_indices = np.asarray(start_indices, dtype=int)
    end_timestamps = np.asarray(end_timestamps, dtype=float)
    start_timestamps = estimated_timestamps[start_indices]

    estimated_start_poses = estimated_poses[start_indices]
    estimated_end_poses = interpolate_pose_trajectory(estimated_timestamps, estimated_poses, end_timestamps)
    reference_start_poses = interpolate_pose_trajectory(reference_timestamps, reference_poses, start_timestamps)
    reference_end_poses = interpolate_pose_trajectory(reference_timestamps, reference_poses, end_timestamps)

    return relative_motion_error_arrays(estimated_start_poses, estimated_end_poses, reference_start_poses, reference_end_poses)

def calculate_localization_metric_arrays(trajectory_path: Path, reference_timestamps: np.ndarray, reference_poses: np.ndarray, rpe_delta_time_s: float, rte_distance_m: float) -> dict[str, np.ndarray]:
    '''Calculate the six localization metrics from one saved trajectory CSV.'''

    timestamps, estimated_poses, aligned_reference_poses = load_trajectory_csv(trajectory_path)

    absolute_translation_m = np.linalg.norm(estimated_poses[:, :3, 3] - aligned_reference_poses[:, :3, 3], axis=1)
    absolute_rotation_matrices = np.transpose(aligned_reference_poses[:, :3, :3], (0, 2, 1)) @ estimated_poses[:, :3, :3]
    absolute_rotation_deg = rotation_angle_deg(absolute_rotation_matrices)

    translation_rpe_m, rotation_rpe_deg = calculate_temporal_rpe(timestamps, estimated_poses, reference_timestamps, reference_poses, rpe_delta_time_s)
    translation_rte_m, rotation_rte_deg = calculate_distance_rte(timestamps, estimated_poses, reference_timestamps, reference_poses, rte_distance_m)

    metric_arrays = {
        'absolute_translation_m': absolute_translation_m,
        'absolute_rotation_deg': absolute_rotation_deg,
        'translation_rpe_m': translation_rpe_m,
        'rotation_rpe_deg': rotation_rpe_deg,
        'translation_rte_m': translation_rte_m,
        'rotation_rte_deg': rotation_rte_deg,
    }

    for metric_name, values in metric_arrays.items():
        if len(values) == 0:
            raise ValueError(f'Calculated metric {metric_name!r} contains no samples for {trajectory_path}')

    return metric_arrays


def localization_metric_title(metric_name: str, rpe_delta_time_s: float, rte_distance_m: float) -> str:
    '''Return a plot title including the RPE/RTE interval when applicable.'''

    title = LOCALIZATION_METRIC_TYPES[metric_name]['title']

    if metric_name in {'translation_rpe_m', 'rotation_rpe_deg'}:
        return f'{title} over {rpe_delta_time_s:g} s'

    if metric_name in {'translation_rte_m', 'rotation_rte_deg'}:
        return f'{title} over {rte_distance_m:g} m'

    return title


def collect_localization_metrics(dataset_root: Path, experiment_dir_name: str, lidar_mode: str, pickle_name: str, trajectory_name: str, statistics: tuple[str, ...], rpe_delta_time_s: float, rte_distance_m: float, strict: bool) -> pd.DataFrame:
    '''Calculate six localization metrics for all complete calibration pairs.'''

    dataset_dirs = sorted((path for path in dataset_root.iterdir() if path.is_dir()), key=lambda path: natural_sort_key(path.name))
    rows = []

    for dataset_dir in dataset_dirs:
        with_path, without_path = dataset_result_paths(dataset_dir, experiment_dir_name, lidar_mode, pickle_name)

        if not with_path.is_file() and not without_path.is_file():
            continue

        if not with_path.is_file() or not without_path.is_file():
            if strict:
                missing_paths = [str(path) for path in (with_path, without_path) if not path.is_file()]
                raise FileNotFoundError(f'Incomplete localization-metric pair for {dataset_dir.name}. Missing: {", ".join(missing_paths)}')

            continue

        with_trajectory_path = with_path.parent / trajectory_name
        without_trajectory_path = without_path.parent / trajectory_name

        if not with_trajectory_path.is_file() or not without_trajectory_path.is_file():
            missing_paths = [str(path) for path in (with_trajectory_path, without_trajectory_path) if not path.is_file()]
            message = f'Skipping localization metrics for {dataset_dir.name}: missing trajectory CSV. Missing: {", ".join(missing_paths)}'

            if strict:
                raise FileNotFoundError(message)

            warnings.warn(message)
            continue

        reference_pose_path = kaist_reference_trajectory_path(dataset_dir)
        if not reference_pose_path.is_file():
            raise FileNotFoundError(f'KAIST reference trajectory was not found: {reference_pose_path}')

        reference_timestamps, reference_poses = import_true_trajectory(reference_pose_path)
        reference_timestamps = np.asarray(reference_timestamps, dtype=float)
        reference_poses = np.asarray(reference_poses, dtype=float)

        with_arrays = calculate_localization_metric_arrays(with_trajectory_path, reference_timestamps, reference_poses, rpe_delta_time_s, rte_distance_m)
        without_arrays = calculate_localization_metric_arrays(without_trajectory_path, reference_timestamps, reference_poses, rpe_delta_time_s, rte_distance_m)

        for metric_name in LOCALIZATION_METRIC_TYPES:
            with_values = with_arrays[metric_name]
            without_values = without_arrays[metric_name]

            for statistic in statistics:
                with_value = compute_statistic(with_values, statistic)
                without_value = compute_statistic(without_values, statistic)
                absolute_gain = without_value - with_value
                relative_gain_percent = 100.0 * absolute_gain / without_value if np.isfinite(without_value) and without_value != 0.0 else np.nan

                rows.append({
                    'dataset': dataset_dir.name,
                    'metric': metric_name,
                    'statistic': statistic,
                    'rpe_delta_time_s': rpe_delta_time_s,
                    'rte_distance_m': rte_distance_m,
                    'samples_without_calibration': int(len(without_values)),
                    'samples_with_calibration': int(len(with_values)),
                    'without_calibration': without_value,
                    'with_calibration': with_value,
                    'absolute_gain': absolute_gain,
                    'relative_gain_percent': relative_gain_percent,
                    'improved': bool(absolute_gain > 0.0),
                })

    if not rows:
        raise RuntimeError('No localization metrics could be calculated from the available result pairs')

    dataframe = pd.DataFrame(rows)
    dataset_order = sorted(dataframe['dataset'].unique(), key=natural_sort_key)
    dataset_rank = {dataset_name: index for index, dataset_name in enumerate(dataset_order)}
    metric_rank = {metric_name: index for index, metric_name in enumerate(LOCALIZATION_METRIC_TYPES)}
    statistic_rank = {statistic: index for index, statistic in enumerate(statistics)}

    dataframe['_dataset_rank'] = dataframe['dataset'].map(dataset_rank)
    dataframe['_metric_rank'] = dataframe['metric'].map(metric_rank)
    dataframe['_statistic_rank'] = dataframe['statistic'].map(statistic_rank)
    dataframe = dataframe.sort_values(['_dataset_rank', '_metric_rank', '_statistic_rank']).drop(columns=['_dataset_rank', '_metric_rank', '_statistic_rank']).reset_index(drop=True)

    return dataframe


def build_localization_metric_summary(metrics_dataframe: pd.DataFrame) -> pd.DataFrame:
    '''Build sequence-balanced aggregate values for the six localization metrics.'''

    rows = []

    for metric_name in LOCALIZATION_METRIC_TYPES:
        metric_dataframe = metrics_dataframe[metrics_dataframe['metric'] == metric_name]

        for statistic in metric_dataframe['statistic'].drop_duplicates():
            selected = metric_dataframe[metric_dataframe['statistic'] == statistic]
            without_values = selected['without_calibration'].to_numpy(dtype=float)
            with_values = selected['with_calibration'].to_numpy(dtype=float)
            absolute_gains = selected['absolute_gain'].to_numpy(dtype=float)
            relative_gains = selected['relative_gain_percent'].to_numpy(dtype=float)
            finite_relative_gains = relative_gains[np.isfinite(relative_gains)]

            rows.append({
                'metric': metric_name,
                'statistic': statistic,
                'datasets': int(len(selected)),
                'rpe_delta_time_s': float(selected['rpe_delta_time_s'].iloc[0]),
                'rte_distance_m': float(selected['rte_distance_m'].iloc[0]),
                'without_calibration_mean': float(np.mean(without_values)),
                'without_calibration_std': float(np.std(without_values, ddof=1)) if len(without_values) > 1 else 0.0,
                'with_calibration_mean': float(np.mean(with_values)),
                'with_calibration_std': float(np.std(with_values, ddof=1)) if len(with_values) > 1 else 0.0,
                'absolute_gain_mean': float(np.mean(absolute_gains)),
                'absolute_gain_median': float(np.median(absolute_gains)),
                'relative_gain_percent_mean': float(np.mean(finite_relative_gains)) if len(finite_relative_gains) else np.nan,
                'relative_gain_percent_median': float(np.median(finite_relative_gains)) if len(finite_relative_gains) else np.nan,
                'relative_gain_percent_std': float(np.std(finite_relative_gains, ddof=1)) if len(finite_relative_gains) > 1 else 0.0,
                'improved_datasets': int(np.sum(absolute_gains > 0.0)),
                'degraded_datasets': int(np.sum(absolute_gains < 0.0)),
                'unchanged_datasets': int(np.sum(np.isclose(absolute_gains, 0.0))),
                'improved_fraction': float(np.mean(absolute_gains > 0.0)),
            })

    return pd.DataFrame(rows)


def plot_localization_metric_comparison(metrics_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, rpe_delta_time_s: float, rte_distance_m: float, dpi: int) -> None:
    '''Plot with/without-calibration values for the six localization metrics.'''

    selected_statistic = metrics_dataframe[metrics_dataframe['statistic'] == statistic]
    datasets = sorted(selected_statistic['dataset'].unique(), key=natural_sort_key)
    x = np.arange(len(datasets), dtype=float)
    width = 0.38

    fig, axes = plt.subplots(len(LOCALIZATION_METRIC_TYPES), 1, figsize=(max(10.0, 1.25 * len(datasets)), 18.0), sharex=True)

    for axis, metric_name in zip(axes, LOCALIZATION_METRIC_TYPES):
        selected = selected_statistic[selected_statistic['metric'] == metric_name].set_index('dataset')
        without_values = np.asarray([selected.loc[dataset_name, 'without_calibration'] for dataset_name in datasets], dtype=float)
        with_values = np.asarray([selected.loc[dataset_name, 'with_calibration'] for dataset_name in datasets], dtype=float)

        axis.bar(x - width / 2.0, without_values, width=width, label='Without calibration')
        axis.bar(x + width / 2.0, with_values, width=width, label='With calibration')
        axis.set_ylabel(LOCALIZATION_METRIC_TYPES[metric_name]['ylabel'])
        axis.set_title(f'{localization_metric_title(metric_name, rpe_delta_time_s, rte_distance_m)}: {statistic.upper()}')
        axis.grid(True, axis='y', alpha=0.25)
        axis.legend()

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[-1].set_xlabel('KAIST sequence')

    fig.suptitle(f'Localization metric comparison, LiDAR mode: {lidar_mode}')
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_localization_metric_gain(metrics_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, rpe_delta_time_s: float, rte_distance_m: float, dpi: int) -> None:
    '''Plot percentage gain from adaptive calibration for the six localization metrics.'''

    selected_statistic = metrics_dataframe[metrics_dataframe['statistic'] == statistic]
    datasets = sorted(selected_statistic['dataset'].unique(), key=natural_sort_key)
    x = np.arange(len(datasets), dtype=float)

    fig, axes = plt.subplots(len(LOCALIZATION_METRIC_TYPES), 1, figsize=(max(10.0, 1.25 * len(datasets)), 18.0), sharex=True)

    for axis, metric_name in zip(axes, LOCALIZATION_METRIC_TYPES):
        selected = selected_statistic[selected_statistic['metric'] == metric_name].set_index('dataset')
        gain_values = np.asarray([selected.loc[dataset_name, 'relative_gain_percent'] for dataset_name in datasets], dtype=float)

        axis.bar(x, gain_values)
        axis.axhline(0.0, linewidth=1.0)
        axis.set_ylabel('gain [%]')
        axis.set_title(f'{localization_metric_title(metric_name, rpe_delta_time_s, rte_distance_m)}: {statistic.upper()}')
        axis.grid(True, axis='y', alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[-1].set_xlabel('KAIST sequence')

    fig.suptitle(f'Localization metric calibration gain, LiDAR mode: {lidar_mode}\nPositive values mean lower error with calibration')
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_localization_metric_average_values(metrics_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, rpe_delta_time_s: float, rte_distance_m: float, dpi: int) -> None:
    '''Plot sequence-balanced average values with between-sequence standard deviations.'''

    selected_statistic = metrics_dataframe[metrics_dataframe['statistic'] == statistic]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
    axes = axes.reshape(-1)

    for axis, metric_name in zip(axes, LOCALIZATION_METRIC_TYPES):
        selected = selected_statistic[selected_statistic['metric'] == metric_name]
        without_values = selected['without_calibration'].to_numpy(dtype=float)
        with_values = selected['with_calibration'].to_numpy(dtype=float)

        means = np.asarray([np.mean(without_values), np.mean(with_values)], dtype=float)
        standard_deviations = np.asarray([
            np.std(without_values, ddof=1) if len(without_values) > 1 else 0.0,
            np.std(with_values, ddof=1) if len(with_values) > 1 else 0.0,
        ])

        axis.bar(np.arange(2), means, yerr=standard_deviations, capsize=5)
        axis.set_xticks(np.arange(2))
        axis.set_xticklabels(['Without', 'With'])
        axis.set_ylabel(LOCALIZATION_METRIC_TYPES[metric_name]['ylabel'])
        axis.set_title(localization_metric_title(metric_name, rpe_delta_time_s, rte_distance_m))
        axis.grid(True, axis='y', alpha=0.25)

    fig.suptitle(f'Sequence-balanced average localization metrics: {statistic.upper()}, LiDAR mode: {lidar_mode}')
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_localization_metric_average_gain(metrics_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, rpe_delta_time_s: float, rte_distance_m: float, dpi: int) -> None:
    '''Plot the sequence-balanced average percentage gain of all six metrics.'''

    selected_statistic = metrics_dataframe[metrics_dataframe['statistic'] == statistic]
    labels = []
    mean_gains = []
    std_gains = []

    for metric_name in LOCALIZATION_METRIC_TYPES:
        values = selected_statistic[selected_statistic['metric'] == metric_name]['relative_gain_percent'].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        labels.append(localization_metric_title(metric_name, rpe_delta_time_s, rte_distance_m))
        mean_gains.append(float(np.mean(values)) if len(values) else np.nan)
        std_gains.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

    x = np.arange(len(labels), dtype=float)
    fig, axis = plt.subplots(figsize=(11.5, 5.8))

    axis.bar(x, mean_gains, yerr=std_gains, capsize=5)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, ha='right')
    axis.set_ylabel('mean per-sequence gain [%]')
    axis.set_title(f'Average localization metric gain: {statistic.upper()}, LiDAR mode: {lidar_mode}')
    axis.grid(True, axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

##################################################
# Calibration deviation metrics
##################################################


def calibration_variable_deviation_rows(
    dataset_name: str,
    payload: dict[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    '''Convert one calibration history pickle into scalar deviation rows.

    Extrinsic SE(3) variables are represented by two physically meaningful
    magnitudes:

        rotation:
            geodesic angle between reference and estimated rotations [deg]

        translation:
            Euclidean translation difference [m]

    Time offsets are stored as absolute deviation [ms].

    Gyroscope bias is stored as the Euclidean norm of the bias-vector
    difference [rad/s].
    '''

    variables = payload.get('variables', {})
    windows = payload.get('windows', [])

    if not isinstance(variables, dict):
        raise ValueError(f'Calibration pickle does not contain a variables dictionary: {path}')

    if not isinstance(windows, list):
        raise ValueError(f'Calibration pickle does not contain a windows list: {path}')

    rows = []

    for window in windows:
        window_index = int(window['window_index'])
        window_start_s = float(window['window_start_s'])
        window_end_s = float(window['window_end_s'])
        window_midpoint_s = float(window['window_midpoint_s'])

        window_variables = window.get('variables', {})

        if not isinstance(window_variables, dict):
            raise ValueError(f'Window {window_index} in {path} does not contain a variables dictionary')

        for variable_name, values in window_variables.items():
            metadata = variables.get(variable_name, {})

            variable_type = str(metadata.get('variable_type', 'unknown'))
            fixed = bool(metadata.get('fixed', False))

            if not isinstance(values, dict):
                continue

            estimated = values.get('estimated')
            reference = values.get('reference', metadata.get('reference'))

            if estimated is None or reference is None:
                continue

            ##################################################
            # Extrinsic SE(3)
            ##################################################

            if variable_type == 'extrinsic':
                estimated_pose = np.asarray(
                    estimated,
                    dtype=float,
                )

                reference_pose = np.asarray(
                    reference,
                    dtype=float,
                )

                if estimated_pose.shape != (4, 4):
                    raise ValueError(
                        f'Estimated extrinsic {variable_name} in {path} '
                        f'must have shape (4, 4), got {estimated_pose.shape}'
                    )

                if reference_pose.shape != (4, 4):
                    raise ValueError(
                        f'Reference extrinsic {variable_name} in {path} '
                        f'must have shape (4, 4), got {reference_pose.shape}'
                    )

                relative_pose = (
                    np.linalg.inv(reference_pose)
                    @ estimated_pose
                )

                rotation_deviation_deg = float(
                    np.rad2deg(
                        Rotation.from_matrix(
                            relative_pose[:3, :3]
                        ).magnitude()
                    )
                )

                translation_deviation_m = float(
                    np.linalg.norm(
                        estimated_pose[:3, 3]
                        - reference_pose[:3, 3]
                    )
                )

                rows.append({
                    'dataset': dataset_name,
                    'window_index': window_index,
                    'window_start_s': window_start_s,
                    'window_end_s': window_end_s,
                    'window_midpoint_s': window_midpoint_s,
                    'variable': variable_name,
                    'variable_type': variable_type,
                    'fixed': fixed,
                    'deviation_type': 'extrinsic_rotation_deg',
                    'value': rotation_deviation_deg,
                    'unit': 'deg',
                })

                rows.append({
                    'dataset': dataset_name,
                    'window_index': window_index,
                    'window_start_s': window_start_s,
                    'window_end_s': window_end_s,
                    'window_midpoint_s': window_midpoint_s,
                    'variable': variable_name,
                    'variable_type': variable_type,
                    'fixed': fixed,
                    'deviation_type': 'extrinsic_translation_m',
                    'value': translation_deviation_m,
                    'unit': 'm',
                })

            ##################################################
            # Time offset
            ##################################################

            elif variable_type == 'time_offset':
                estimated_value = float(
                    np.asarray(
                        estimated,
                        dtype=float,
                    ).reshape(())
                )

                reference_value = float(
                    np.asarray(
                        reference,
                        dtype=float,
                    ).reshape(())
                )

                deviation_ms = 1000.0 * abs(
                    estimated_value
                    - reference_value
                )

                rows.append({
                    'dataset': dataset_name,
                    'window_index': window_index,
                    'window_start_s': window_start_s,
                    'window_end_s': window_end_s,
                    'window_midpoint_s': window_midpoint_s,
                    'variable': variable_name,
                    'variable_type': variable_type,
                    'fixed': fixed,
                    'deviation_type': 'time_offset_ms',
                    'value': deviation_ms,
                    'unit': 'ms',
                })

            ##################################################
            # Gyroscope bias
            ##################################################

            elif variable_type == 'gyro_bias':
                estimated_bias = np.asarray(estimated, dtype=float).reshape(-1)
                reference_bias = np.asarray(reference, dtype=float).reshape(-1)

                if estimated_bias.shape != reference_bias.shape:
                    raise ValueError(f'Bias shape mismatch for {variable_name} in {path}: estimated={estimated_bias.shape}, reference={reference_bias.shape}')

                bias_difference = estimated_bias - reference_bias
                bias_deviation_radps = float(np.linalg.norm(bias_difference))

                rows.append({
                    'dataset': dataset_name,
                    'window_index': window_index,
                    'window_start_s': window_start_s,
                    'window_end_s': window_end_s,
                    'window_midpoint_s': window_midpoint_s,
                    'variable': variable_name,
                    'variable_type': variable_type,
                    'fixed': fixed,
                    'deviation_type': 'gyro_bias_radps',
                    'value': bias_deviation_radps,
                    'unit': 'rad/s',
                })

                component_names = ('x', 'y', 'z')

                for component_index, component_name in enumerate(component_names):
                    rows.append({
                        'dataset': dataset_name,
                        'window_index': window_index,
                        'window_start_s': window_start_s,
                        'window_end_s': window_end_s,
                        'window_midpoint_s': window_midpoint_s,
                        'variable': variable_name,
                        'variable_type': variable_type,
                        'fixed': fixed,
                        'deviation_type': f'gyro_bias_{component_name}_radps',
                        'value': float(abs(bias_difference[component_index])),
                        'unit': 'rad/s',
                    })

    return rows


def collect_calibration_deviations(dataset_root: Path, experiment_dir_name: str, lidar_mode: str, strict: bool) -> pd.DataFrame:
    '''Load with-calibration histories from all available KAIST datasets.

    Only direct child directories containing the expected experiment-results
    structure are treated as datasets. This prevents output directories such as
    pipeline_results_summary from being interpreted as KAIST sequences.

    The without-calibration runs are intentionally not used here because their
    calibration variables are fixed to the supplied reference values and their
    deviations should therefore be identically zero.
    '''

    dataset_dirs = sorted((path for path in dataset_root.iterdir() if path.is_dir() and (path / experiment_dir_name / 'errors' / lidar_mode).is_dir()), key=lambda path: natural_sort_key(path.name))

    rows = []

    for dataset_dir in dataset_dirs:
        calibration_dir = dataset_dir / experiment_dir_name / 'errors' / lidar_mode / 'with_calibration'
        calibration_path = calibration_dir / 'calibration_variables.pkl'

        if not calibration_dir.is_dir():
            continue

        if not calibration_path.is_file():
            message = f'Skipping calibration-deviation analysis for {dataset_dir.name}: {calibration_path} was not found'

            if strict:
                raise FileNotFoundError(message)

            warnings.warn(message)
            continue

        payload = load_result_pickle(calibration_path)

        stored_dataset_name = payload.get('dataset_name')

        if stored_dataset_name is not None and str(stored_dataset_name) != dataset_dir.name:
            warnings.warn(f'Dataset name mismatch in {calibration_path}: directory={dataset_dir.name!r}, pickle={stored_dataset_name!r}')

        stored_lidar_mode = payload.get('lidar_mode')

        if stored_lidar_mode is not None and str(stored_lidar_mode) != lidar_mode:
            raise ValueError(f'LiDAR mode mismatch in {calibration_path}: expected {lidar_mode!r}, pickle={stored_lidar_mode!r}')

        rows.extend(calibration_variable_deviation_rows(dataset_name=dataset_dir.name, payload=payload, path=calibration_path))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_calibration_deviation_dataset_summary(
    calibration_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    '''Reduce rolling-window deviations to one value per dataset and variable.'''

    rows = []

    group_columns = [
        'dataset',
        'variable',
        'variable_type',
        'fixed',
        'deviation_type',
        'unit',
    ]

    for group_values, group in calibration_dataframe.groupby(
        group_columns,
        sort=False,
    ):
        (
            dataset_name,
            variable_name,
            variable_type,
            fixed,
            deviation_type,
            unit,
        ) = group_values

        values = group['value'].to_numpy(
            dtype=float
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            continue

        rows.append({
            'dataset': dataset_name,
            'variable': variable_name,
            'variable_type': variable_type,
            'fixed': bool(fixed),
            'deviation_type': deviation_type,
            'unit': unit,
            'windows': int(len(values)),
            'mean': float(np.mean(values)),
            'std': float(
                np.std(values, ddof=1)
            ) if len(values) > 1 else 0.0,
            'median': float(
                np.median(values)
            ),
            'rmse': float(
                np.sqrt(
                    np.mean(values**2)
                )
            ),
            'p90': float(
                np.percentile(
                    values,
                    90.0,
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_calibration_deviation_summary(
    dataset_summary_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    '''Calculate sequence-balanced mean calibration deviation per variable.

    Each dataset contributes one mean value regardless of its trajectory length
    or number of rolling windows.
    '''

    rows = []

    group_columns = [
        'variable',
        'variable_type',
        'fixed',
        'deviation_type',
        'unit',
    ]

    for group_values, group in dataset_summary_dataframe.groupby(
        group_columns,
        sort=False,
    ):
        (
            variable_name,
            variable_type,
            fixed,
            deviation_type,
            unit,
        ) = group_values

        dataset_means = group['mean'].to_numpy(
            dtype=float
        )

        dataset_means = dataset_means[
            np.isfinite(dataset_means)
        ]

        if len(dataset_means) == 0:
            continue

        rows.append({
            'variable': variable_name,
            'variable_type': variable_type,
            'fixed': bool(fixed),
            'deviation_type': deviation_type,
            'unit': unit,
            'datasets': int(
                len(dataset_means)
            ),
            'mean': float(
                np.mean(dataset_means)
            ),
            'std': float(
                np.std(
                    dataset_means,
                    ddof=1,
                )
            ) if len(dataset_means) > 1 else 0.0,
            'median': float(
                np.median(dataset_means)
            ),
            'p90': float(
                np.percentile(
                    dataset_means,
                    90.0,
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def plot_calibration_deviation_per_dataset(
    dataset_summary_dataframe: pd.DataFrame,
    output_path: Path,
    lidar_mode: str,
    dpi: int,
) -> None:
    '''Plot mean rolling-window calibration deviation for every dataset.'''

    dataframe = dataset_summary_dataframe[
        ~dataset_summary_dataframe['fixed']
    ].copy()

    if dataframe.empty:
        warnings.warn(
            'No free calibration variables are available for '
            'calibration-deviation plotting.'
        )
        return

    datasets = sorted(
        dataframe['dataset'].unique(),
        key=natural_sort_key,
    )

    x = np.arange(
        len(datasets),
        dtype=float,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            max(
                12.0,
                1.35 * len(datasets),
            ),
            10.0,
        ),
    )

    axes = axes.reshape(-1)

    for axis, (
        deviation_type,
        deviation_info,
    ) in zip(
        axes,
        CALIBRATION_DEVIATION_TYPES.items(),
    ):
        selected = dataframe[
            dataframe['deviation_type']
            == deviation_type
        ]

        variable_names = sorted(
            selected['variable'].unique()
        )

        if len(variable_names) == 0:
            axis.text(
                0.5,
                0.5,
                'No active variables',
                transform=axis.transAxes,
                ha='center',
                va='center',
            )

            axis.set_title(
                deviation_info['title']
            )

            axis.grid(
                True,
                axis='y',
                alpha=0.25,
            )

            continue

        width = 0.8 / len(
            variable_names
        )

        for variable_index, variable_name in enumerate(
            variable_names
        ):
            variable_data = selected[
                selected['variable']
                == variable_name
            ].set_index(
                'dataset'
            )

            values = np.asarray(
                [
                    variable_data.loc[
                        dataset_name,
                        'mean',
                    ]
                    if dataset_name
                    in variable_data.index
                    else np.nan
                    for dataset_name
                    in datasets
                ],
                dtype=float,
            )

            offset = (
                variable_index
                - 0.5 * (
                    len(variable_names)
                    - 1
                )
            ) * width

            axis.bar(
                x + offset,
                values,
                width=width,
                label=variable_name,
            )

        axis.set_ylabel(
            deviation_info['ylabel']
        )

        axis.set_title(
            deviation_info['title']
        )

        axis.grid(
            True,
            axis='y',
            alpha=0.25,
        )

        axis.legend(
            fontsize='small'
        )

    for axis in axes:
        axis.set_xticks(
            x
        )

        axis.set_xticklabels(
            datasets,
            rotation=45,
            ha='right',
        )

    fig.suptitle(
        f'Mean calibration deviation per KAIST sequence, '
        f'LiDAR mode: {lidar_mode}'
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
    )

    plt.close(
        fig
    )


def plot_mean_calibration_deviation(
    summary_dataframe: pd.DataFrame,
    output_path: Path,
    lidar_mode: str,
    dpi: int,
) -> None:
    '''Plot sequence-balanced mean calibration deviation per variable.'''

    dataframe = summary_dataframe[
        ~summary_dataframe['fixed']
    ].copy()

    if dataframe.empty:
        warnings.warn(
            'No free calibration variables are available for '
            'mean calibration-deviation plotting.'
        )
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.0),
    )

    axes = axes.reshape(-1)

    for axis, (
        deviation_type,
        deviation_info,
    ) in zip(
        axes,
        CALIBRATION_DEVIATION_TYPES.items(),
    ):
        if deviation_type == 'gyro_bias_radps':
            component_types = ('gyro_bias_x_radps', 'gyro_bias_y_radps', 'gyro_bias_z_radps')
            component_labels = ('X', 'Y', 'Z')

            means = []
            standard_deviations = []

            for component_type in component_types:
                component_data = dataframe[dataframe['deviation_type'] == component_type]

                if component_data.empty:
                    means.append(np.nan)
                    standard_deviations.append(np.nan)
                else:
                    means.append(float(component_data['mean'].iloc[0]))
                    standard_deviations.append(float(component_data['std'].iloc[0]))

            x = np.arange(3, dtype=float)

            axis.bar(x, np.asarray(means, dtype=float), yerr=np.asarray(standard_deviations, dtype=float), capsize=5)
            axis.set_xticks(x)
            axis.set_xticklabels(component_labels)
            axis.set_ylabel(deviation_info['ylabel'])
            axis.set_title(deviation_info['title'])
            axis.grid(True, axis='y', alpha=0.25)
            continue

        selected = dataframe[
            dataframe['deviation_type']
            == deviation_type
        ].copy()

        if selected.empty:
            axis.text(
                0.5,
                0.5,
                'No active variables',
                transform=axis.transAxes,
                ha='center',
                va='center',
            )

            axis.set_title(
                deviation_info['title']
            )

            axis.grid(
                True,
                axis='y',
                alpha=0.25,
            )

            continue

        selected = selected.sort_values(
            'variable'
        )

        labels = selected[
            'variable'
        ].tolist()

        means = selected[
            'mean'
        ].to_numpy(
            dtype=float
        )

        standard_deviations = selected[
            'std'
        ].to_numpy(
            dtype=float
        )

        x = np.arange(
            len(labels),
            dtype=float,
        )

        axis.bar(
            x,
            means,
            yerr=standard_deviations,
            capsize=5,
        )

        axis.set_xticks(
            x
        )

        axis.set_xticklabels(
            labels,
            rotation=25,
            ha='right',
        )

        axis.set_ylabel(
            deviation_info['ylabel']
        )

        axis.set_title(
            deviation_info['title']
        )

        axis.grid(
            True,
            axis='y',
            alpha=0.25,
        )

    fig.suptitle(
        f'Sequence-balanced mean calibration deviation, '
        f'LiDAR mode: {lidar_mode}'
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
    )

    plt.close(
        fig
    )

##################################################
# Dataset discovery
##################################################


def dataset_result_paths(dataset_dir: Path, experiment_dir_name: str, lidar_mode: str, pickle_name: str) -> tuple[Path, Path]:
    '''Construct the calibrated and uncalibrated pickle paths for one dataset.'''

    base_dir = dataset_dir / experiment_dir_name / 'errors' / lidar_mode

    return (
        base_dir / 'with_calibration' / pickle_name,
        base_dir / 'without_calibration' / pickle_name,
    )


def collect_results(dataset_root: Path, experiment_dir_name: str, lidar_mode: str, pickle_name: str, statistics: tuple[str, ...], strict: bool, strict_config: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Load paired calibration results from every direct dataset child directory.'''

    dataset_dirs = sorted((path for path in dataset_root.iterdir() if path.is_dir()), key=lambda path: natural_sort_key(path.name))

    with_rows = []
    without_rows = []

    for dataset_dir in dataset_dirs:
        with_path, without_path = dataset_result_paths(dataset_dir, experiment_dir_name, lidar_mode, pickle_name)

        if not with_path.is_file() and not without_path.is_file():
            continue

        if not with_path.is_file() or not without_path.is_file():
            missing_paths = [str(path) for path in (with_path, without_path) if not path.is_file()]
            message = f'Skipping dataset {dataset_dir.name}: incomplete calibration pair. Missing: {", ".join(missing_paths)}'

            if strict:
                raise FileNotFoundError(message)

            warnings.warn(message)
            continue

        with_payload = load_result_pickle(with_path)
        without_payload = load_result_pickle(without_path)

        validate_result_payload(with_payload, path=with_path, dataset_name=dataset_dir.name, calibration_mode='with_calibration', lidar_mode=lidar_mode)
        validate_result_payload(without_payload, path=without_path, dataset_name=dataset_dir.name, calibration_mode='without_calibration', lidar_mode=lidar_mode)

        config_mismatches = compare_run_configs(with_payload, without_payload)

        if config_mismatches:
            mismatch_text = '; '.join(config_mismatches)
            message = f'Run-configuration mismatch for {dataset_dir.name}: {mismatch_text}'

            if strict_config:
                raise ValueError(message)

            warnings.warn(message)

        with_row = result_to_metric_row(dataset_dir.name, with_payload, with_path, statistics)
        without_row = result_to_metric_row(dataset_dir.name, without_payload, without_path, statistics)

        with_rows.append(with_row)
        without_rows.append(without_row)

    if not with_rows:
        raise RuntimeError(f'No complete with/without-calibration result pairs were found under {dataset_root} for LiDAR mode "{lidar_mode}".')

    with_dataframe = pd.DataFrame(with_rows)
    without_dataframe = pd.DataFrame(without_rows)

    with_dataframe = with_dataframe.sort_values('dataset', key=lambda series: series.map(natural_sort_key)).reset_index(drop=True)
    without_dataframe = without_dataframe.sort_values('dataset', key=lambda series: series.map(natural_sort_key)).reset_index(drop=True)

    if with_dataframe['dataset'].tolist() != without_dataframe['dataset'].tolist():
        raise RuntimeError('Internal error: calibrated and uncalibrated dataset ordering differs.')

    return with_dataframe, without_dataframe


##################################################
# Calibration-gain tables
##################################################


def build_gain_tables(with_dataframe: pd.DataFrame, without_dataframe: pd.DataFrame, statistics: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Build wide and long per-dataset calibration gain tables.'''

    datasets = with_dataframe['dataset'].tolist()

    wide_rows = []
    long_rows = []

    for row_index, dataset_name in enumerate(datasets):
        wide_row: dict[str, Any] = {
            'dataset': dataset_name,
        }

        for error_name in ERROR_TYPES:
            for statistic in statistics:
                metric_name = f'{error_name}_{statistic}'

                with_value = float(with_dataframe.loc[row_index, metric_name])
                without_value = float(without_dataframe.loc[row_index, metric_name])

                absolute_gain = without_value - with_value
                relative_gain_percent = 100.0 * absolute_gain / without_value if np.isfinite(without_value) and without_value != 0.0 else np.nan

                wide_row[f'{metric_name}_without'] = without_value
                wide_row[f'{metric_name}_with'] = with_value
                wide_row[f'{metric_name}_gain'] = absolute_gain
                wide_row[f'{metric_name}_gain_percent'] = relative_gain_percent

                long_rows.append({
                    'dataset': dataset_name,
                    'error_type': error_name,
                    'statistic': statistic,
                    'without_calibration': without_value,
                    'with_calibration': with_value,
                    'absolute_gain': absolute_gain,
                    'relative_gain_percent': relative_gain_percent,
                    'improved': bool(absolute_gain > 0.0),
                })

        wide_rows.append(wide_row)

    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def build_gain_summary(gain_long_dataframe: pd.DataFrame, statistics: tuple[str, ...]) -> pd.DataFrame:
    '''Average calibration gains across datasets while preserving sequence-level weighting.'''

    rows = []

    for error_name in ERROR_TYPES:
        for statistic in statistics:
            selected = gain_long_dataframe[(gain_long_dataframe['error_type'] == error_name) & (gain_long_dataframe['statistic'] == statistic)].copy()

            without_values = selected['without_calibration'].to_numpy(dtype=float)
            with_values = selected['with_calibration'].to_numpy(dtype=float)
            absolute_gains = selected['absolute_gain'].to_numpy(dtype=float)
            relative_gains = selected['relative_gain_percent'].to_numpy(dtype=float)

            finite_relative_gains = relative_gains[np.isfinite(relative_gains)]

            improved_count = int(np.sum(absolute_gains > 0.0))
            degraded_count = int(np.sum(absolute_gains < 0.0))
            unchanged_count = int(np.sum(np.isclose(absolute_gains, 0.0)))

            rows.append({
                'error_type': error_name,
                'statistic': statistic,
                'datasets': int(len(selected)),
                'without_calibration_mean': float(np.mean(without_values)),
                'without_calibration_std': float(np.std(without_values, ddof=1)) if len(without_values) > 1 else 0.0,
                'with_calibration_mean': float(np.mean(with_values)),
                'with_calibration_std': float(np.std(with_values, ddof=1)) if len(with_values) > 1 else 0.0,
                'absolute_gain_mean': float(np.mean(absolute_gains)),
                'absolute_gain_median': float(np.median(absolute_gains)),
                'relative_gain_percent_mean': float(np.mean(finite_relative_gains)) if len(finite_relative_gains) else np.nan,
                'relative_gain_percent_median': float(np.median(finite_relative_gains)) if len(finite_relative_gains) else np.nan,
                'relative_gain_percent_std': float(np.std(finite_relative_gains, ddof=1)) if len(finite_relative_gains) > 1 else 0.0,
                'improved_datasets': improved_count,
                'degraded_datasets': degraded_count,
                'unchanged_datasets': unchanged_count,
                'improved_fraction': improved_count / len(selected) if len(selected) else np.nan,
            })

    return pd.DataFrame(rows)


##################################################
# Plotting
##################################################


def plot_metric_comparison(with_dataframe: pd.DataFrame, without_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, dpi: int) -> None:
    '''Plot calibrated and uncalibrated values for every dataset and error type.'''

    datasets = with_dataframe['dataset'].tolist()
    x = np.arange(len(datasets), dtype=float)
    width = 0.38

    fig, axes = plt.subplots(len(ERROR_TYPES), 1, figsize=(max(10.0, 1.25 * len(datasets)), 11.0), sharex=True)

    if len(ERROR_TYPES) == 1:
        axes = np.asarray([axes])

    for axis, (error_name, error_info) in zip(axes, ERROR_TYPES.items()):
        metric_name = f'{error_name}_{statistic}'

        without_values = without_dataframe[metric_name].to_numpy(dtype=float)
        with_values = with_dataframe[metric_name].to_numpy(dtype=float)

        axis.bar(x - width / 2.0, without_values, width=width, label='Without calibration')
        axis.bar(x + width / 2.0, with_values, width=width, label='With calibration')

        axis.set_ylabel(error_info['ylabel'])
        axis.set_title(f"{error_info['title']}: {statistic.upper()}")
        axis.grid(True, axis='y', alpha=0.25)
        axis.legend()

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[-1].set_xlabel('KAIST sequence')

    fig.suptitle(f'Calibration comparison, LiDAR mode: {lidar_mode}')
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_metric_gain(gain_long_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, dpi: int) -> None:
    '''Plot per-dataset percentage gain produced by calibration.'''

    datasets = list(dict.fromkeys(gain_long_dataframe['dataset'].tolist()))
    x = np.arange(len(datasets), dtype=float)

    fig, axes = plt.subplots(len(ERROR_TYPES), 1, figsize=(max(10.0, 1.25 * len(datasets)), 11.0), sharex=True)

    if len(ERROR_TYPES) == 1:
        axes = np.asarray([axes])

    for axis, (error_name, error_info) in zip(axes, ERROR_TYPES.items()):
        selected = gain_long_dataframe[(gain_long_dataframe['error_type'] == error_name) & (gain_long_dataframe['statistic'] == statistic)].set_index('dataset')
        gain_values = np.asarray([selected.loc[dataset_name, 'relative_gain_percent'] for dataset_name in datasets], dtype=float)

        axis.bar(x, gain_values)
        axis.axhline(0.0, linewidth=1.0)

        axis.set_ylabel('gain [%]')
        axis.set_title(f"{error_info['title']}: {statistic.upper()}")
        axis.grid(True, axis='y', alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[-1].set_xlabel('KAIST sequence')

    fig.suptitle(f'Calibration gain, LiDAR mode: {lidar_mode}\nPositive values mean lower error with calibration')
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_average_gain(gain_summary_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, dpi: int) -> None:
    '''Plot sequence-balanced mean percentage calibration gain for all error types.'''

    selected = gain_summary_dataframe[gain_summary_dataframe['statistic'] == statistic].set_index('error_type')

    error_names = list(ERROR_TYPES.keys())
    labels = ['Full SE(3)', 'Rotation', 'Translation']
    mean_gains = np.asarray([selected.loc[error_name, 'relative_gain_percent_mean'] for error_name in error_names], dtype=float)
    std_gains = np.asarray([selected.loc[error_name, 'relative_gain_percent_std'] for error_name in error_names], dtype=float)

    x = np.arange(len(error_names), dtype=float)

    fig, axis = plt.subplots(figsize=(8.5, 5.5))

    axis.bar(x, mean_gains, yerr=std_gains, capsize=5)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel('mean per-sequence gain [%]')
    axis.set_title(f'Average calibration gain: {statistic.upper()}, LiDAR mode: {lidar_mode}')
    axis.grid(True, axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


##################################################
# CSV formatting
##################################################


def ordered_metric_columns(statistics: tuple[str, ...]) -> list[str]:
    '''Return paper-friendly metric column ordering.'''

    columns = ['dataset', 'samples', 'duration_s']

    for error_name in ERROR_TYPES:
        for statistic in statistics:
            columns.append(f'{error_name}_{statistic}')

    return columns


def save_tables(with_dataframe: pd.DataFrame, without_dataframe: pd.DataFrame, gain_wide_dataframe: pd.DataFrame, gain_long_dataframe: pd.DataFrame, gain_summary_dataframe: pd.DataFrame, output_dir: Path, statistics: tuple[str, ...]) -> dict[str, Path]:
    '''Save all result tables.'''

    output_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = ordered_metric_columns(statistics)

    with_path = output_dir / 'with_calibration_metrics.csv'
    without_path = output_dir / 'without_calibration_metrics.csv'
    gain_wide_path = output_dir / 'calibration_gains.csv'
    gain_long_path = output_dir / 'calibration_gains_long.csv'
    gain_summary_path = output_dir / 'calibration_gain_summary.csv'

    with_dataframe[metric_columns].to_csv(with_path, index=False, float_format='%.10g')
    without_dataframe[metric_columns].to_csv(without_path, index=False, float_format='%.10g')
    gain_wide_dataframe.to_csv(gain_wide_path, index=False, float_format='%.10g')
    gain_long_dataframe.to_csv(gain_long_path, index=False, float_format='%.10g')
    gain_summary_dataframe.to_csv(gain_summary_path, index=False, float_format='%.10g')

    return {
        'with_calibration': with_path,
        'without_calibration': without_path,
        'gain_wide': gain_wide_path,
        'gain_long': gain_long_path,
        'gain_summary': gain_summary_path,
    }


##################################################
# Console summary
##################################################


def print_compact_summary(gain_summary_dataframe: pd.DataFrame, statistic: str) -> None:
    '''Print the most useful aggregate comparison to the console.'''

    selected = gain_summary_dataframe[gain_summary_dataframe['statistic'] == statistic]

    print()
    print(f'Calibration comparison using {statistic.upper()}')
    print('========================================')

    for _, row in selected.iterrows():
        error_name = str(row['error_type'])
        title = ERROR_TYPES[error_name]['title']

        print()
        print(title)
        print(f"  datasets:                    {int(row['datasets'])}")
        print(f"  without calibration mean:    {row['without_calibration_mean']:.6g}")
        print(f"  with calibration mean:       {row['with_calibration_mean']:.6g}")
        print(f"  mean absolute gain:           {row['absolute_gain_mean']:.6g}")
        print(f"  mean relative gain:           {row['relative_gain_percent_mean']:.3f}%")
        print(f"  median relative gain:         {row['relative_gain_percent_median']:.3f}%")
        print(f"  improved datasets:            {int(row['improved_datasets'])}/{int(row['datasets'])}")


##################################################
# Argument parser
##################################################


def build_argument_parser() -> argparse.ArgumentParser:
    '''Construct the command-line interface.'''

    parser = argparse.ArgumentParser(description='Aggregate KAIST pipeline trajectory-error pickles across multiple dataset directories and compare calibration against fixed parameters.')

    parser.add_argument('dataset_root', type=Path, help='Directory whose direct child directories are datasets such as Urban13, Urban14, Urban15, Urban16, and Urban17.')
    parser.add_argument('--lidar-mode', choices=('poses', 'odometry'), default='poses', help='LiDAR result mode to process.')
    parser.add_argument('--experiment-dir-name', type=str, default='experiment_results', help='Experiment-results directory name inside every dataset.')
    parser.add_argument('--pickle-name', type=str, default='trajectory_errors.pkl', help='Result pickle filename.')
    parser.add_argument('--trajectory-name', type=str, default='trajectory.csv', help='Trajectory CSV filename stored next to each result pickle.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Summary output directory. Default: <dataset_root>/pipeline_results_summary/<lidar_mode>. Relative paths are interpreted relative to dataset_root.')
    parser.add_argument('--statistics', nargs='+', choices=AVAILABLE_STATISTICS, default=['median', 'rmse', 'p90'], help='Per-sequence statistics included in the tables and plots.')
    parser.add_argument('--strict', action=argparse.BooleanOptionalAction, default=False, help='Fail instead of skipping datasets with only one calibration mode available.')
    parser.add_argument('--strict-config', action=argparse.BooleanOptionalAction, default=False, help='Fail when important run parameters differ between calibrated and uncalibrated runs.')
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True, help='Generate comparison and gain plots.')
    parser.add_argument('--dpi', type=int, default=180, help='Saved plot resolution.')
    parser.add_argument('--console-statistic', choices=AVAILABLE_STATISTICS, default='rmse', help='Statistic used for the compact final console summary.')
    parser.add_argument('--rpe-delta-time-s', type=float, default=DEFAULT_RPE_DELTA_TIME_S, help='Temporal interval used for translational and rotational RPE.')
    parser.add_argument('--rte-distance-m', type=float, default=DEFAULT_RTE_DISTANCE_M, help='Reference travel distance used for translational and rotational distance RTE.')

    return parser


##################################################
# Main processing
##################################################


def run(args: argparse.Namespace) -> None:
    '''Aggregate all direct dataset directories and save tables and plots.'''

    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {dataset_root}')

    statistics = tuple(dict.fromkeys(args.statistics))

    output_dir = resolve_output_directory(dataset_root, args.output_dir, args.lidar_mode)
    plots_dir = output_dir / 'plots'
    plots_metrics_dir = output_dir / 'plots_metrics'

    with_dataframe, without_dataframe = collect_results(
        dataset_root=dataset_root,
        experiment_dir_name=args.experiment_dir_name,
        lidar_mode=args.lidar_mode,
        pickle_name=args.pickle_name,
        statistics=statistics,
        strict=args.strict,
        strict_config=args.strict_config,
    )

    gain_wide_dataframe, gain_long_dataframe = build_gain_tables(with_dataframe, without_dataframe, statistics)
    gain_summary_dataframe = build_gain_summary(gain_long_dataframe, statistics)

    table_paths = save_tables(
        with_dataframe=with_dataframe,
        without_dataframe=without_dataframe,
        gain_wide_dataframe=gain_wide_dataframe,
        gain_long_dataframe=gain_long_dataframe,
        gain_summary_dataframe=gain_summary_dataframe,
        output_dir=output_dir,
        statistics=statistics,
    )

    localization_metrics_dataframe = collect_localization_metrics(
        dataset_root=dataset_root,
        experiment_dir_name=args.experiment_dir_name,
        lidar_mode=args.lidar_mode,
        pickle_name=args.pickle_name,
        trajectory_name=args.trajectory_name,
        statistics=statistics,
        rpe_delta_time_s=args.rpe_delta_time_s,
        rte_distance_m=args.rte_distance_m,
        strict=args.strict,
    )
    localization_metric_summary_dataframe = build_localization_metric_summary(localization_metrics_dataframe)

    plots_metrics_dir.mkdir(parents=True, exist_ok=True)
    localization_metrics_path = plots_metrics_dir / 'localization_metrics.csv'
    localization_metric_summary_path = plots_metrics_dir / 'localization_metrics_summary.csv'
    localization_metrics_dataframe.to_csv(localization_metrics_path, index=False, float_format='%.10g')
    localization_metric_summary_dataframe.to_csv(localization_metric_summary_path, index=False, float_format='%.10g')

    ##################################################
    # Calibration deviation analysis
    ##################################################

    calibration_deviation_dataframe = collect_calibration_deviations(
        dataset_root=dataset_root,
        experiment_dir_name=args.experiment_dir_name,
        lidar_mode=args.lidar_mode,
        strict=args.strict,
    )

    calibration_deviation_windows_path = None
    calibration_deviation_dataset_path = None
    calibration_deviation_summary_path = None

    if not calibration_deviation_dataframe.empty:
        calibration_deviation_dataset_dataframe = build_calibration_deviation_dataset_summary(
            calibration_deviation_dataframe
        )

        calibration_deviation_summary_dataframe = build_calibration_deviation_summary(
            calibration_deviation_dataset_dataframe
        )

        calibration_deviation_windows_path = (
            plots_metrics_dir
            / 'calibration_deviations_windows.csv'
        )

        calibration_deviation_dataset_path = (
            plots_metrics_dir
            / 'calibration_deviations_per_dataset.csv'
        )

        calibration_deviation_summary_path = (
            plots_metrics_dir
            / 'calibration_deviation_summary.csv'
        )

        calibration_deviation_dataframe.to_csv(
            calibration_deviation_windows_path,
            index=False,
            float_format='%.10g',
        )

        calibration_deviation_dataset_dataframe.to_csv(
            calibration_deviation_dataset_path,
            index=False,
            float_format='%.10g',
        )

        calibration_deviation_summary_dataframe.to_csv(
            calibration_deviation_summary_path,
            index=False,
            float_format='%.10g',
        )

    #############################################
    # Plots
    #############################################

    if args.plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

        for statistic in statistics:
            plot_metric_comparison(with_dataframe, without_dataframe, statistic, plots_dir / f'comparison_{statistic}.png', args.lidar_mode, args.dpi)
            plot_metric_gain(gain_long_dataframe, statistic, plots_dir / f'gain_{statistic}.png', args.lidar_mode, args.dpi)
            plot_average_gain(gain_summary_dataframe, statistic, plots_dir / f'average_gain_{statistic}.png', args.lidar_mode, args.dpi)

            plot_localization_metric_comparison(localization_metrics_dataframe, statistic, plots_metrics_dir / f'comparison_{statistic}.png', args.lidar_mode, args.rpe_delta_time_s, args.rte_distance_m, args.dpi)
            plot_localization_metric_gain(localization_metrics_dataframe, statistic, plots_metrics_dir / f'gain_{statistic}.png', args.lidar_mode, args.rpe_delta_time_s, args.rte_distance_m, args.dpi)
            plot_localization_metric_average_values(localization_metrics_dataframe, statistic, plots_metrics_dir / f'average_values_{statistic}.png', args.lidar_mode, args.rpe_delta_time_s, args.rte_distance_m, args.dpi)
            plot_localization_metric_average_gain(localization_metrics_dataframe, statistic, plots_metrics_dir / f'average_gain_{statistic}.png', args.lidar_mode, args.rpe_delta_time_s, args.rte_distance_m, args.dpi)

        if not calibration_deviation_dataframe.empty:
            plot_calibration_deviation_per_dataset(calibration_deviation_dataset_dataframe, plots_metrics_dir / 'calibration_deviation_per_dataset.png', args.lidar_mode, args.dpi,)
            plot_mean_calibration_deviation(calibration_deviation_summary_dataframe, plots_metrics_dir / 'calibration_deviation_mean.png', args.lidar_mode, args.dpi,)

    console_statistic = args.console_statistic

    if console_statistic not in statistics:
        console_statistic = 'rmse' if 'rmse' in statistics else statistics[0]

    print()
    print('Processed datasets')
    print('==================')
    print(', '.join(with_dataframe['dataset'].tolist()))

    print_compact_summary(gain_summary_dataframe, console_statistic)

    print()
    print('Saved tables')
    print('============')

    for path in table_paths.values():
        print(path)

    print(localization_metrics_path)
    print(localization_metric_summary_path)    
    if calibration_deviation_windows_path is not None:
        print(calibration_deviation_windows_path)
        print(calibration_deviation_dataset_path)
        print(calibration_deviation_summary_path)

    if args.plots:
        print()
        print('Saved plots')
        print('===========')
        print(plots_dir)
        print(plots_metrics_dir)


##################################################
# Entrypoint
##################################################


def main() -> None:
    '''Parse CLI arguments and process all datasets.'''

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error('--dpi must be positive.')

    if args.rpe_delta_time_s <= 0.0:
        parser.error('--rpe-delta-time-s must be positive.')

    if args.rte_distance_m <= 0.0:
        parser.error('--rte-distance-m must be positive.')

    run(args)


if __name__ == '__main__':
    main()

# python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset
# python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset --experiment-dir-name experiment_results_new --output-dir "/mnt/d/Downloads/MobRobLab/KAISTDataset/pipeline_results_new/plots/poses"