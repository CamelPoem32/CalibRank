'''Run the KAIST multi-sensor rolling factor-graph pipeline from the command line.

This script is a runnable counterpart of notebook 19. It loads one KAIST Urban
dataset, imports precomputed left/right LiDAR odometry and scan-to-map pose
measurements, constructs the same IMU and LiDAR streams used by the notebook,
runs multi_sensor_pipeline.RollingGraph, evaluates the estimated trajectory
against the KAIST reference trajectory, and saves the requested plots.

Two calibration modes are supported:

    with-calibration
        IMU extrinsic, IMU time offset, IMU gyro bias, right-LiDAR extrinsic,
        and right-LiDAR time offset are optimized. Left-LiDAR extrinsic and
        time offset remain fixed and define the main LiDAR reference sensor.

    without-calibration
        All calibration parameters are fixed.

The graph operates in a translated local world frame O to avoid optimizing
georeferenced KAIST poses whose translations are millions of meters from the
world origin. Estimated body poses are transformed back into the original KAIST
world frame W before evaluation and plotting.

Default output layout:

    <dataset_root>/outputs/
        plots/
            with_calibration/
                trajectory_vs_reference.png
                trajectory_errors.png
                benchmark_errors.png
                calibration_errors.png

            without_calibration/
                trajectory_vs_reference.png
                trajectory_errors.png
                benchmark_errors.png
                calibration_errors.png

        errors/
            <lidar_mode>/
                with_calibration/
                    trajectory_errors.pkl
                    trajectory.csv
                    calibration_variables.pkl
                without_calibration/
                    trajectory_errors.pkl
                    trajectory.csv
                    calibration_variables.pkl

Examples:

    python run_kaist_multi_sensor_pipeline.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --mode with-calibration

    python run_kaist_multi_sensor_pipeline.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --mode without-calibration

    python run_kaist_multi_sensor_pipeline.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --mode with-calibration --output-dir /home/camel/Skoltech/phd_proposal/outputs/urban16

    python run_kaist_multi_sensor_pipeline.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --mode with-calibration --window-size 50 --step-size 25
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import pickle

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mrob
import numpy as np



##################################################
# Project discovery
##################################################


def find_project_root(start_path: Path) -> Path:
    '''Find the repository root containing src/multi_sensor_pipeline.'''

    start_path = start_path.resolve()
    candidates = [start_path, *start_path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    visited = set()

    for candidate in candidates:
        if candidate in visited:
            continue

        visited.add(candidate)

        if (candidate / 'src' / 'multi_sensor_pipeline').is_dir():
            return candidate

    raise FileNotFoundError("Could not locate the project root containing 'src/multi_sensor_pipeline'. Run this script from inside the project or place it inside the repository.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
SRC_ROOT = PROJECT_ROOT / 'src'

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


##################################################
# Project imports
##################################################


import data_processing

from kaist_dataset import load_imus
from kaist_dataset.data import import_true_trajectory
from kaist_dataset.lidar import load_lidar_data_csv
from kaist_dataset.lidar_map import load_lidar_map_pose_csv
from multi_sensor_pipeline import (
    ComplexAccelStream,
    GyroStream,
    LidarOdometryStream,
    LidarPoseStream,
    RollingGraph,
    Sensor,
    SimpleAccelStream,
    SolverConfig,
    TrajectoryConfig,
    VariableConfig,
    VariableKey,
    VariableType,
    plot_calibration_estimates,
    plot_lidar_localization_benchmark_errors,
    plot_rolling_trajectory,
    plot_trajectory_errors,
    print_rolling_result_summary,
)


##################################################
# KAIST sensor extrinsics
##################################################
#
# Convention:
#
#     T_A_B maps coordinates from frame B into frame A.
#
# Vehicle2LeftVLP:
#
#     T_B_LL : left LiDAR -> body
#
# Vehicle2RightVLP:
#
#     T_B_RL : right LiDAR -> body
#
##################################################


T_B_LL_INITIAL = np.array([
    [-0.514066, -0.702201, -0.492595, -0.440699],
    [0.486485, -0.711672, 0.506809, 0.397052],
    [-0.706447, 0.0208933, 0.707457, 1.90953],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

T_B_RL_INITIAL = np.array([
    [-0.512152, 0.699241, -0.498761, -0.449885],
    [-0.494811, -0.714859, -0.494104, -0.416713],
    [-0.702041, -0.0062642, 0.712109, 1.91294],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

T_I_B_INITIAL = np.array([
    [1.0, 0.0, 0.0, -0.07],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.7],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

T_LL_B_INITIAL = np.linalg.inv(T_B_LL_INITIAL)
T_RL_B_INITIAL = np.linalg.inv(T_B_RL_INITIAL)
T_B_I_INITIAL_FOR_DATASET = np.linalg.inv(T_I_B_INITIAL)

T_B_LL_INITIAL_TANGENT = mrob.SE3(T_B_LL_INITIAL).Ln()
T_B_RL_INITIAL_TANGENT = mrob.SE3(T_B_RL_INITIAL).Ln()
T_B_I_INITIAL_TANGENT = mrob.SE3(T_B_I_INITIAL_FOR_DATASET).Ln()

GYRO_BIAS_INITIAL_FOR_DATASET = np.zeros(3, dtype=float)
TAU_I_INITIAL_FOR_DATASET = 0.0
TAU_L_INITIAL_FOR_DATASET = 0.0


##################################################
# Generic helpers
##################################################


def log(verbosity: int, level: int, *values: Any) -> None:
    '''Print values when the requested verbosity includes this level.'''

    if verbosity >= level:
        print(*values)


def accumulate_lidar_poses(relative_poses: np.ndarray) -> np.ndarray:
    '''Accumulate consecutive LiDAR SE(3) increments into an absolute local trajectory.'''

    relative_poses = np.asarray(relative_poses, dtype=float)
    poses = np.empty((len(relative_poses) + 1, 4, 4), dtype=float)
    poses[0] = np.eye(4)

    for measurement_index, relative_pose in enumerate(relative_poses):
        poses[measurement_index + 1] = poses[measurement_index] @ relative_pose

    return poses


def lidar_scan_timestamps_for_pipeline(lidar_data, scan_period_s: float | None) -> np.ndarray:
    '''Return one timestamp per LiDAR scan.'''

    if lidar_data.scan_timestamps_s is not None:
        return np.asarray(lidar_data.scan_timestamps_s, dtype=float)

    midpoint_times = np.asarray(lidar_data.timestamps_s, dtype=float)

    if midpoint_times.size == 0:
        raise ValueError('LiDAR data has no timestamps.')

    if midpoint_times.size == 1:
        if scan_period_s is None:
            raise ValueError('Set --lidar-scan-period-s when only one LiDAR midpoint timestamp is available.')

        half_period = 0.5 * float(scan_period_s)
        return np.array([midpoint_times[0] - half_period, midpoint_times[0] + half_period])

    boundaries = np.empty(midpoint_times.size + 1, dtype=float)
    boundaries[1:-1] = 0.5 * (midpoint_times[:-1] + midpoint_times[1:])
    boundaries[0] = midpoint_times[0] - (boundaries[1] - midpoint_times[0])
    boundaries[-1] = midpoint_times[-1] + (midpoint_times[-1] - boundaries[-2])

    return boundaries


def load_successful_lidar_map_poses(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    '''Load successful finite absolute LiDAR sensor poses T_W_L from one CSV.'''

    timestamps_s, poses_T_W_L, dataframe = load_lidar_map_pose_csv(csv_path)
    timestamps_s = np.asarray(timestamps_s, dtype=float)
    poses_T_W_L = np.asarray(poses_T_W_L, dtype=float)

    success_mask = dataframe['success'].to_numpy(dtype=bool) if 'success' in dataframe.columns else np.ones(len(dataframe), dtype=bool)
    finite_mask = np.all(np.isfinite(poses_T_W_L[:, :3, :]), axis=(1, 2))
    mask = success_mask & finite_mask

    selected_timestamps = timestamps_s[mask]
    selected_poses = poses_T_W_L[mask]

    if len(selected_timestamps) == 0:
        raise ValueError(f'LiDAR map pose CSV contains no successful finite poses: {csv_path}')

    if np.any(np.diff(selected_timestamps) <= 0.0):
        raise ValueError(f'LiDAR map pose timestamps must be strictly increasing: {csv_path}')

    return selected_timestamps, selected_poses


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    '''Save and close one Matplotlib figure.'''

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def compute_trajectory_error_series(estimated_timestamps: np.ndarray, estimated_poses: np.ndarray, reference_timestamps: np.ndarray, reference_poses: np.ndarray) -> dict[str, np.ndarray]:
    '''Calculate the raw trajectory error series saved for cross-sequence analysis.

    The definitions intentionally match plot_trajectory_errors:

        full_se3:
            ||Log(T_reference^{-1} T_estimated)||_2

        rotation_deg:
            norm of the rotational part of the SE(3) logarithm, in degrees

        translation_m:
            Euclidean distance between estimated and reference positions
    '''

    estimated_timestamps = np.asarray(estimated_timestamps, dtype=float).reshape(-1)
    estimated_poses = np.asarray(estimated_poses, dtype=float)
    reference_timestamps = np.asarray(reference_timestamps, dtype=float).reshape(-1)
    reference_poses = np.asarray(reference_poses, dtype=float)

    overlap_mask = (estimated_timestamps >= reference_timestamps[0]) & (estimated_timestamps <= reference_timestamps[-1])

    error_timestamps = estimated_timestamps[overlap_mask]
    error_estimated_poses = estimated_poses[overlap_mask]

    if len(error_timestamps) == 0:
        raise ValueError('Estimated trajectory does not overlap the reference trajectory.')

    interpolated_reference_poses = np.asarray([data_processing._interpolate_pose(reference_timestamps, reference_poses, timestamp) for timestamp in error_timestamps], dtype=float)

    full_se3_errors = np.empty(len(error_timestamps), dtype=float)
    rotational_errors_deg = np.empty(len(error_timestamps), dtype=float)
    translational_errors_m = np.empty(len(error_timestamps), dtype=float)

    for pose_index, (estimated_pose, reference_pose) in enumerate(zip(error_estimated_poses, interpolated_reference_poses)):
        relative_error_pose = np.linalg.inv(reference_pose) @ estimated_pose
        relative_error_tangent = np.asarray(mrob.SE3(relative_error_pose).Ln(), dtype=float).reshape(6)

        full_se3_errors[pose_index] = np.linalg.norm(relative_error_tangent)
        rotational_errors_deg[pose_index] = np.rad2deg(np.linalg.norm(relative_error_tangent[:3]))
        translational_errors_m[pose_index] = np.linalg.norm(estimated_pose[:3, 3] - reference_pose[:3, 3])

    return {
        'timestamps_s': error_timestamps,
        'relative_timestamps_s': error_timestamps - error_timestamps[0],
        'estimated_poses': error_estimated_poses,
        'reference_poses': interpolated_reference_poses,
        'full_se3': full_se3_errors,
        'rotation_deg': rotational_errors_deg,
        'translation_m': translational_errors_m,
    }


def save_trajectory_error_pickle(path: Path, *, dataset_name: str, mode: str, lidar_mode: str, error_series: dict[str, np.ndarray], trajectory_statistics: dict[str, Any], benchmark_statistics: dict[str, Any], run_config: dict[str, Any]) -> Path:
    '''Save raw trajectory errors and run metadata for later multi-sequence analysis.'''

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'schema_version': 1,
        'dataset_name': dataset_name,
        'mode': mode,
        'lidar_mode': lidar_mode,
        'timestamps_s': np.asarray(error_series['timestamps_s'], dtype=float),
        'relative_timestamps_s': np.asarray(error_series['relative_timestamps_s'], dtype=float),
        'errors': {
            'full_se3': np.asarray(error_series['full_se3'], dtype=float),
            'rotation_deg': np.asarray(error_series['rotation_deg'], dtype=float),
            'translation_m': np.asarray(error_series['translation_m'], dtype=float),
        },
        'trajectory_statistics': trajectory_statistics,
        'benchmark_statistics': benchmark_statistics,
        'run_config': run_config,
    }

    with path.open('wb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    return path

def calibration_value_copy(value: Any) -> Any:
    '''Return one calibration value in a pickle-friendly NumPy/scalar representation.'''

    if value is None:
        return None

    array = np.asarray(value, dtype=float)

    if array.ndim == 0:
        return float(array)

    return array.copy()


def save_calibration_variables_pickle(
    path: Path,
    *,
    dataset_name: str,
    mode: str,
    lidar_mode: str,
    results: list[Any],
    calibration_keys: list[VariableKey],
    reference_values: dict[VariableKey, Any],
    variable_configs: dict[VariableKey, VariableConfig],
    run_config: dict[str, Any],
) -> Path:
    '''Save estimated and reference calibration variables for every rolling window.

    The output intentionally uses string labels instead of VariableKey objects so
    it can be inspected without reconstructing the pipeline variable classes.

    Each rolling-window entry stores:

        window index
        start / end / midpoint timestamp
        estimated calibration value
        reference calibration value
        variable type
        sensor id
        whether the variable was fixed during optimization
    '''

    path.parent.mkdir(parents=True, exist_ok=True)

    ##################################################
    # Describe the calibration variables once
    ##################################################

    variables = {}

    for key in calibration_keys:
        config = variable_configs.get(key)

        variables[key.label] = {
            'sensor_id': key.sensor_id,
            'variable_type': key.variable_type.value,
            'fixed': bool(config.fixed) if config is not None else None,
            'reference': calibration_value_copy(reference_values.get(key)),
        }

    ##################################################
    # Save one estimated calibration state per window
    ##################################################

    windows = []

    for result in results:
        window_variables = {}

        for key in calibration_keys:
            estimated_value = result.calibration_value(key)

            window_variables[key.label] = {
                'estimated': calibration_value_copy(estimated_value),
                'reference': calibration_value_copy(reference_values.get(key)),
            }

        windows.append({
            'window_index': int(result.window_index),
            'window_start_s': float(result.window_start),
            'window_end_s': float(result.window_end),
            'window_midpoint_s': 0.5 * (float(result.window_start) + float(result.window_end)),
            'variables': window_variables,
        })

    ##################################################
    # Build a self-contained calibration-history payload
    ##################################################

    payload = {
        'schema_version': 1,
        'dataset_name': dataset_name,
        'mode': mode,
        'lidar_mode': lidar_mode,
        'variables': variables,
        'windows': windows,
        'run_config': run_config,
    }

    with path.open('wb') as stream:
        pickle.dump(
            payload,
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return path


def save_trajectory_csv(path: Path, error_series: dict[str, np.ndarray]) -> Path:
    '''Save aligned estimated/reference trajectories as timestamped 4x4 SE(3) matrices.'''

    path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = np.asarray(error_series['timestamps_s'], dtype=float).reshape(-1)
    estimated_poses = np.asarray(error_series['estimated_poses'], dtype=float)
    reference_poses = np.asarray(error_series['reference_poses'], dtype=float)

    if estimated_poses.shape != (len(timestamps), 4, 4):
        raise ValueError(f'estimated_poses must have shape ({len(timestamps)}, 4, 4), got {estimated_poses.shape}')

    if reference_poses.shape != (len(timestamps), 4, 4):
        raise ValueError(f'reference_poses must have shape ({len(timestamps)}, 4, 4), got {reference_poses.shape}')

    matrix_columns = [f'T{row_index}{column_index}' for row_index in range(4) for column_index in range(4)]
    columns = ['timestamp_s', *[f'estimated_{column}' for column in matrix_columns], *[f'reference_{column}' for column in matrix_columns]]
    values = np.column_stack((timestamps, estimated_poses.reshape(len(timestamps), 16), reference_poses.reshape(len(timestamps), 16)))

    np.savetxt(path, values, delimiter=',', header=','.join(columns), comments='', fmt='%.17g')

    return path


def jsonable(value: Any) -> Any:
    '''Convert NumPy-heavy structures into JSON-compatible values.'''

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def parse_scheduler(text: str) -> list[tuple[float, int]]:
    '''Parse lambda:iterations,lambda:iterations scheduler syntax.'''

    scheduler = []

    for component in str(text).split(','):
        component = component.strip()

        if not component:
            continue

        pieces = component.split(':')

        if len(pieces) != 2:
            raise argparse.ArgumentTypeError("Scheduler must use lambda:iterations pairs, for example '1e-5:100,1e-5:100,1e-9:100'.")

        damping = float(pieces[0])
        iterations = int(pieces[1])

        if not np.isfinite(damping) or damping <= 0.0:
            raise argparse.ArgumentTypeError('Scheduler damping values must be finite and positive.')

        if iterations <= 0:
            raise argparse.ArgumentTypeError('Scheduler iteration counts must be positive.')

        scheduler.append((damping, iterations))

    if not scheduler:
        raise argparse.ArgumentTypeError('Solver scheduler cannot be empty.')

    return scheduler


def normalize_mode_name(mode: str) -> str:
    '''Convert a CLI mode to a filesystem-safe directory name.'''

    return str(mode).replace('-', '_')


##################################################
# Dataset path helpers
##################################################


def kaist_dataset_layout(dataset_root: Path) -> tuple[str, Path, Path]:
    '''Return dataset name, sensor-data root, and reference-pose path.'''

    dataset_root = dataset_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {dataset_root}')

    dataset_name = dataset_root.name
    dataset_slug = dataset_name.lower()
    sensor_data_root = dataset_root / f'{dataset_slug}_data' / dataset_slug / 'sensor_data'
    reference_pose_path = dataset_root / f'{dataset_slug}_pose' / dataset_slug / 'global_pose.csv'

    if not sensor_data_root.is_dir():
        raise FileNotFoundError(f'KAIST sensor_data directory was not found: {sensor_data_root}')

    if not reference_pose_path.exists():
        raise FileNotFoundError(f'KAIST reference trajectory path was not found: {reference_pose_path}')

    return dataset_name, sensor_data_root, reference_pose_path


def processed_data_directory_has_required_files(path: Path, lidar_mode: str, lidar_side: str = 'left') -> bool:
    '''Return whether a directory contains the selected LiDAR CSV.'''

    if lidar_side not in ('left', 'right'):
        raise ValueError(f'Unknown LiDAR side: {lidar_side}')

    if lidar_mode == 'poses':
        return (path / f'lidar_map_poses_vlp_{lidar_side}.csv').is_file()

    if lidar_mode == 'odometry':
        return (path / f'lidar_odometry_vlp_{lidar_side}.csv').is_file()

    raise ValueError(f'Unknown LiDAR mode: {lidar_mode}')


def resolve_processed_data_directory(dataset_root: Path, dataset_name: str, requested_path: Path | None, lidar_mode: str, lidar_side: str = 'left') -> Path:
    '''Resolve the directory containing the LiDAR CSV required by the selected LiDAR mode.'''

    if requested_path is not None:
        processed_data_dir = requested_path.expanduser()

        if not processed_data_dir.is_absolute():
            processed_data_dir = dataset_root / processed_data_dir

        processed_data_dir = processed_data_dir.resolve()

        if not processed_data_dir.is_dir():
            raise FileNotFoundError(f'Processed-data directory does not exist: {processed_data_dir}')

        if not processed_data_directory_has_required_files(processed_data_dir, lidar_mode, lidar_side):
            required_filename = f'lidar_map_poses_vlp_{lidar_side}.csv' if lidar_mode == 'poses' else f'lidar_odometry_vlp_{lidar_side}.csv'
            raise FileNotFoundError(f'Processed-data directory does not contain the file required by LiDAR mode "{lidar_mode}":\n  {required_filename}\nDirectory: {processed_data_dir}')

        return processed_data_dir

    candidates = [dataset_root, dataset_root / 'data', PROJECT_ROOT / 'data' / 'KAISTDataset' / dataset_name]

    for candidate in candidates:
        if processed_data_directory_has_required_files(candidate, lidar_mode, lidar_side):
            return candidate.resolve()

    candidate_text = '\n'.join(f'  {candidate}' for candidate in candidates)
    required_filename = f'lidar_map_poses_vlp_{lidar_side}.csv' if lidar_mode == 'poses' else f'lidar_odometry_vlp_{lidar_side}.csv'

    raise FileNotFoundError(f'Could not find processed LiDAR data for mode "{lidar_mode}".\nRequired file:\n  {required_filename}\nSearched:\n{candidate_text}\nPass its directory explicitly using --processed-data-dir.')


def resolve_output_directory(dataset_root: Path, requested_path: Path | None) -> Path:
    '''Resolve the configurable output root.

    Absolute paths are used directly. Relative paths are interpreted relative
    to the dataset root.
    '''

    if requested_path is None:
        return (dataset_root / 'outputs').resolve()

    output_dir = requested_path.expanduser()

    if not output_dir.is_absolute():
        output_dir = dataset_root / output_dir

    return output_dir.resolve()


##################################################
# Calibration-variable configuration
##################################################


def build_variable_configs(
    mode: str,
    imu_extrinsic_key: VariableKey,
    imu_tau_key: VariableKey,
    imu_bias_key: VariableKey,
    lidar_extrinsic_key: VariableKey,
    lidar_tau_key: VariableKey,
    lidar_right_extrinsic_key: VariableKey,
    lidar_right_tau_key: VariableKey,
    right_lidar_active: bool,
    imu_prior_source: str,
    imu_extrinsic_rotation_prior_information: float,
    imu_extrinsic_translation_prior_information: float,
    imu_tau_initial: float,
    imu_tau_prior_information: float,
    lidar_extrinsic_rotation_prior_information: float,
    lidar_extrinsic_translation_prior_information: float,
    lidar_tau_prior_information: float,
    right_lidar_tau_initial: float,
    bias_prior_information: float,
) -> dict[VariableKey, VariableConfig]:
    '''Create calibration-variable configuration for one execution mode.'''

    T_B_I_prior_information = np.diag([
        imu_extrinsic_rotation_prior_information,
        imu_extrinsic_rotation_prior_information,
        imu_extrinsic_rotation_prior_information,
        imu_extrinsic_translation_prior_information,
        imu_extrinsic_translation_prior_information,
        imu_extrinsic_translation_prior_information,
    ])

    T_B_L_prior_information = np.diag([
        lidar_extrinsic_rotation_prior_information,
        lidar_extrinsic_rotation_prior_information,
        lidar_extrinsic_rotation_prior_information,
        lidar_extrinsic_translation_prior_information,
        lidar_extrinsic_translation_prior_information,
        lidar_extrinsic_translation_prior_information,
    ])

    bias_estimate_initial = np.zeros(3, dtype=float)

    if mode == 'with-calibration':
        variable_configs = {
            imu_extrinsic_key: VariableConfig(
                initial_source='optimized',
                initial_value=T_B_I_INITIAL_FOR_DATASET,
                fixed=False,
                prior_source=imu_prior_source,
                prior_value=T_B_I_INITIAL_FOR_DATASET,
                prior_information=T_B_I_prior_information,
            ),
            imu_tau_key: VariableConfig(
                initial_source='constant',
                initial_value=imu_tau_initial,
                fixed=False,
                prior_source=imu_prior_source,
                prior_value=imu_tau_initial,
                prior_information=imu_tau_prior_information,
            ),
            imu_bias_key: VariableConfig(
                initial_source='optimized',
                initial_value=bias_estimate_initial,
                fixed=False,
                prior_source='constant',
                prior_value=bias_estimate_initial,
                prior_information=bias_prior_information,
            ),
            lidar_extrinsic_key: VariableConfig(initial_source='constant', initial_value=T_B_LL_INITIAL, fixed=True),
            lidar_tau_key: VariableConfig(initial_source='constant', initial_value=0.0, fixed=True),
        }

        if right_lidar_active:
            variable_configs[lidar_right_extrinsic_key] = VariableConfig(
                initial_source='optimized',
                initial_value=T_B_RL_INITIAL,
                fixed=False,
                prior_source='optimized',
                prior_value=T_B_RL_INITIAL,
                prior_information=T_B_L_prior_information,
            )

            variable_configs[lidar_right_tau_key] = VariableConfig(
                initial_source='optimized',
                initial_value=right_lidar_tau_initial,
                fixed=False,
                prior_source='constant',
                prior_value=0.0,
                prior_information=lidar_tau_prior_information,
            )

        return variable_configs

    if mode == 'without-calibration':
        variable_configs = {
            imu_extrinsic_key: VariableConfig(initial_source='constant', initial_value=T_B_I_INITIAL_FOR_DATASET, fixed=True),
            imu_tau_key: VariableConfig(initial_source='constant', initial_value=imu_tau_initial, fixed=True),
            imu_bias_key: VariableConfig(initial_source='constant', initial_value=bias_estimate_initial, fixed=True),
            lidar_extrinsic_key: VariableConfig(initial_source='constant', initial_value=T_B_LL_INITIAL, fixed=True),
            lidar_tau_key: VariableConfig(initial_source='constant', initial_value=0.0, fixed=True),
        }

        if right_lidar_active:
            variable_configs[lidar_right_extrinsic_key] = VariableConfig(initial_source='constant', initial_value=T_B_RL_INITIAL, fixed=True)
            variable_configs[lidar_right_tau_key] = VariableConfig(initial_source='constant', initial_value=right_lidar_tau_initial, fixed=True)

        return variable_configs

    raise ValueError(f'Unknown calibration mode: {mode}')


##################################################
# Argument parser
##################################################


def build_argument_parser() -> argparse.ArgumentParser:
    '''Construct the command-line interface.'''

    parser = argparse.ArgumentParser(description='Run the KAIST rolling LiDAR-IMU factor-graph pipeline and save trajectory, benchmark, and calibration plots.')

    parser.add_argument('dataset_root', type=Path, help='Path to one KAIST Urban dataset root, for example /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16.')

    parser.add_argument('--mode', choices=('with-calibration', 'without-calibration'), default='with-calibration', help='Run with online calibration enabled or with all calibration parameters fixed.')
    parser.add_argument('--processed-data-dir', type=Path, default=None, help='Directory containing processed LiDAR CSVs for the selected --lidar-mode. Relative paths are interpreted relative to dataset_root.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output root. Default: <dataset_root>/outputs. Relative paths are interpreted relative to dataset_root.')
    parser.add_argument('--verbosity', '-v', type=int, choices=(0, 1, 2), default=0, help='Console verbosity.')
    parser.add_argument('--graph-verbosity', type=int, choices=(0, 1, 2), default=0, help='Verbosity passed to RollingGraph.generate_filter_iterative.')
    parser.add_argument('--dpi', type=int, default=160, help='DPI used for saved figures.')

    parser.add_argument('--imu-frequency-hz', type=float, default=100.0, help='Target IMU resampling frequency.')
    parser.add_argument('--max-imu-files', type=int, default=None, help='Optional maximum number of IMU files.')
    parser.add_argument('--lidar-scan-period-s', type=float, default=None, help='Fallback LiDAR scan period when original scan timestamps are unavailable.')

    parser.add_argument('--acc-mode', choices=('simple', 'complex'), default='simple', help='Accelerometer factor type.')
    parser.add_argument('--imu-samples-per-factor', type=int, default=None, help='Optional IMU samples-per-factor setting.')
    parser.add_argument('--lidar-samples-per-factor', type=int, default=None, help='Optional LiDAR samples-per-factor setting.')
    parser.add_argument('--map-pose-factor-stride', type=int, default=1, help='Use one absolute LiDAR map-pose factor every N trajectory poses.')
    parser.add_argument('--lidar-mode', choices=('poses', 'odometry'), default='poses', help='LiDAR measurement mode. "poses" uses absolute scan-to-map LidarPoseStream factors. "odometry" uses relative LidarOdometryStream factors. The modes are mutually exclusive.')
    parser.add_argument('--imu-time-offset-margin', type=float, default=5.0, help='Measurement support margin for IMU temporal calibration.')
    parser.add_argument('--lidar-time-offset-margin', type=float, default=5.0, help='Measurement support margin for LiDAR temporal calibration.')

    parser.add_argument('--gyro-information', type=float, default=1e6, help='Gyroscope factor information.')
    parser.add_argument('--accel-information', type=float, default=1.0, help='Accelerometer factor information.')
    parser.add_argument('--lidar-rotation-information', type=float, default=1e-2, help='Rotation information for relative LiDAR odometry factors.')
    parser.add_argument('--lidar-translation-information', type=float, default=1.0, help='Translation information for relative LiDAR odometry factors.')
    parser.add_argument('--map-pose-rotation-information', type=float, default=1e4, help='Rotation information for absolute LiDAR map-pose factors.')
    parser.add_argument('--map-pose-translation-information', type=float, default=1e-3, help='Translation information for absolute LiDAR map-pose factors.')

    parser.add_argument('--imu-prior-source', choices=('optimized', 'constant', 'numerical', 'default'), default='optimized', help='Soft-prior source for free IMU calibration variables.')
    parser.add_argument('--imu-extrinsic-rotation-prior-information', type=float, default=1e1, help='Rotational information applied to each of the three IMU-extrinsic rotation components.')
    parser.add_argument('--imu-extrinsic-translation-prior-information', type=float, default=1e3, help='Translational information applied to each of the three IMU-extrinsic translation components.')
    parser.add_argument('--imu-tau-initial', type=float, default=0.0, help='Initial IMU time offset.')
    parser.add_argument('--imu-tau-prior-information', type=float, default=1e2, help='Information of the IMU temporal-offset prior.')
    parser.add_argument('--lidar-extrinsic-rotation-prior-information', type=float, default=1e1, help='Rotational information applied to each of the three free LiDAR-extrinsic rotation components.')
    parser.add_argument('--lidar-extrinsic-translation-prior-information', type=float, default=1e3, help='Translational information applied to each of the three free LiDAR-extrinsic translation components.')
    parser.add_argument('--right-lidar-tau-initial', type=float, default=0.0, help='Initial right-LiDAR temporal offset.')
    parser.add_argument('--lidar-tau-prior-information', '--right-lidar-tau-prior-information', dest='lidar_tau_prior_information', type=float, default=1e2, help='Information of the free LiDAR temporal-offset prior. Currently this applies to the right LiDAR. --right-lidar-tau-prior-information is retained as a backwards-compatible alias.')
    parser.add_argument('--bias-prior-information', type=float, default=1e2, help='Gyroscope-bias prior information.')
    parser.add_argument('--gravity-z', type=float, default=9.81, help='World gravity Z component passed to the accelerometer stream.')

    parser.add_argument('--window-size', type=float, default=500.0, help='Rolling window size in seconds.')
    parser.add_argument('--step-size', type=float, default=250.0, help='Rolling-window step in seconds.')
    parser.add_argument('--solver-method', type=str, default='LM', help='MROB solver method.')
    parser.add_argument('--solver-scheduler', type=parse_scheduler, default=parse_scheduler('1e-5:100,1e-5:100,1e-9:100'), help="Solver scheduler, for example '1e-5:100,1e-5:100,1e-9:100'.")
    parser.add_argument('--solver-verbose', action=argparse.BooleanOptionalAction, default=False, help='Enable verbose output inside the graph solver.')

    return parser


##################################################
# Pipeline
##################################################


def run_pipeline(args: argparse.Namespace) -> None:
    '''Load KAIST data, solve the rolling graph, and save all requested plots.'''

    dataset_root = args.dataset_root.expanduser().resolve()
    dataset_name, sensor_data_root, reference_pose_path = kaist_dataset_layout(dataset_root)
    processed_data_dir = resolve_processed_data_directory(dataset_root, dataset_name, args.processed_data_dir, args.lidar_mode)
    output_dir = resolve_output_directory(dataset_root, args.output_dir)

    mode_output_name = normalize_mode_name(args.mode)
    lidar_mode_output_name = normalize_mode_name(args.lidar_mode)

    plots_dir = output_dir / 'plots' / lidar_mode_output_name / mode_output_name
    errors_dir = output_dir / 'errors' / lidar_mode_output_name / mode_output_name

    plots_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    lidar_odometry_csv_left = processed_data_dir / 'lidar_odometry_vlp_left.csv'
    lidar_odometry_csv_right = processed_data_dir / 'lidar_odometry_vlp_right.csv'
    lidar_map_pose_csv_left = processed_data_dir / 'lidar_map_poses_vlp_left.csv'
    lidar_map_pose_csv_right = processed_data_dir / 'lidar_map_poses_vlp_right.csv'

    if args.lidar_mode == 'poses' and not lidar_map_pose_csv_left.is_file():
        raise FileNotFoundError(f'LiDAR pose mode requires the left map-pose CSV:\n{lidar_map_pose_csv_left}')

    if args.lidar_mode == 'odometry' and not lidar_odometry_csv_left.is_file():
        raise FileNotFoundError(f'LiDAR odometry mode requires the left odometry CSV:\n{lidar_odometry_csv_left}')

    log(args.verbosity, 1, f'Dataset: {dataset_name}')
    log(args.verbosity, 1, f'Dataset root: {dataset_root}')
    log(args.verbosity, 1, f'Sensor data root: {sensor_data_root}')
    log(args.verbosity, 1, f'Reference poses: {reference_pose_path}')
    log(args.verbosity, 1, f'LiDAR mode: {args.lidar_mode}')
    log(args.verbosity, 1, f'Processed LiDAR data: {processed_data_dir}')
    log(args.verbosity, 1, f'Calibration mode: {args.mode}')
    log(args.verbosity, 1, f'Output root: {output_dir}')
    log(args.verbosity, 1, f'Plots: {plots_dir}')
    log(args.verbosity, 1, f'Error data: {errors_dir}')

    ##################################################
    # Load IMU
    ##################################################

    imu_streams_raw = load_imus(
        sensor_data_root,
        target_frequency_hz=args.imu_frequency_hz,
        max_files=args.max_imu_files,
    )

    if not imu_streams_raw:
        raise RuntimeError(f'No IMU streams were loaded from {sensor_data_root}')

    primary_imu_name = list(imu_streams_raw.keys())[0]
    primary_imu = imu_streams_raw[primary_imu_name]

    ##################################################
    # Prepare IMU measurements and reference trajectory
    ##################################################

    original_imu_timestamps = np.asarray(primary_imu.timestamps_s, dtype=float).copy()
    original_gyrs = np.asarray(primary_imu.gyro_radps, dtype=float).copy()
    original_accs = np.asarray(primary_imu.accel_mps2, dtype=float).copy()

    true_timestamps, true_poses = import_true_trajectory(reference_pose_path)
    true_timestamps = np.asarray(true_timestamps, dtype=float)
    true_poses = np.asarray(true_poses, dtype=float)

    T_B_I_REFERENCE = T_B_I_INITIAL_FOR_DATASET.copy()
    TAU_I_REFERENCE = float(TAU_I_INITIAL_FOR_DATASET)
    BIAS_REFERENCE = GYRO_BIAS_INITIAL_FOR_DATASET.copy()
    T_B_L_REFERENCE = T_B_LL_INITIAL.copy()
    TAU_L_REFERENCE = float(TAU_L_INITIAL_FOR_DATASET)

    log(args.verbosity, 1, 'Primary IMU:', primary_imu_name, primary_imu.timestamps_s.shape)
    log(args.verbosity, 1, {'imu_start': float(original_imu_timestamps[0]), 'imu_end': float(original_imu_timestamps[-1]), 'imu_samples': len(original_imu_timestamps)})
    log(args.verbosity, 2, 'Reference T_B_I Ln:', mrob.SE3(T_B_I_REFERENCE).Ln())
    log(args.verbosity, 2, 'Reference tau_I:', TAU_I_REFERENCE)

    ##################################################
    # Initialize mode-dependent LiDAR variables
    ##################################################

    lidar_pose_timestamps_left = None
    lidar_pose_timestamps_right = None
    lidar_poses_left = None
    lidar_poses_right = None
    lidar_poses_left_graph = None
    lidar_poses_right_graph = None

    lidar_timestamps = None
    lidar_relative_poses = None
    lidar_poses = None
    lidar_right_timestamps = None
    lidar_right_relative_poses = None
    lidar_right_poses = None

    ##################################################
    # LiDAR mode: absolute sensor poses
    ##################################################

    if args.lidar_mode == 'poses':
        lidar_pose_timestamps_left, lidar_poses_left = load_successful_lidar_map_poses(lidar_map_pose_csv_left)

        if lidar_map_pose_csv_right.is_file():
            lidar_pose_timestamps_right, lidar_poses_right = load_successful_lidar_map_poses(lidar_map_pose_csv_right)
        else:
            lidar_pose_timestamps_right = np.empty(0, dtype=float)
            lidar_poses_right = np.empty((0, 4, 4), dtype=float)


        ##################################################
        # Local graph world frame
        ##################################################
        #
        # O and W have identical orientation and differ only by translation.
        #
        # Absolute map poses therefore remain physically unchanged except for
        # removal of the large KAIST world-coordinate offset.
        ##################################################

        GRAPH_ORIGIN_W = lidar_poses_left[0, :3, 3].copy()

        T_O_W = np.eye(4, dtype=float)
        T_O_W[:3, 3] = -GRAPH_ORIGIN_W

        T_W_O = np.linalg.inv(T_O_W)

        lidar_poses_left_graph = T_O_W @ lidar_poses_left
        lidar_poses_right_graph = T_O_W @ lidar_poses_right if len(lidar_poses_right) else np.empty((0, 4, 4), dtype=float)

        trajectory_source_timestamps = lidar_pose_timestamps_left
        trajectory_source_sensor_poses = lidar_poses_left_graph
        right_lidar_timestamps_for_overlap = lidar_pose_timestamps_right

        log(args.verbosity, 1, 'LiDAR input: absolute scan-to-map poses.')
        log(args.verbosity, 1, 'Left map poses:', len(lidar_poses_left))

        if len(lidar_poses_right):
            log(args.verbosity, 1, 'Right map poses:', len(lidar_poses_right))
        else:
            log(args.verbosity, 1, 'Right map-pose CSV not found. Running with the left LiDAR only.')

        log(args.verbosity, 2, 'Left map pose timestamp range:', lidar_pose_timestamps_left[0], '...', lidar_pose_timestamps_left[-1])
        log(args.verbosity, 2, 'KAIST graph origin in W:', GRAPH_ORIGIN_W)
        log(args.verbosity, 2, 'Maximum graph-frame Left LiDAR position norm [m]:', np.max(np.linalg.norm(lidar_poses_left_graph[:, :3, 3], axis=1)))

    ##################################################
    # LiDAR mode: relative odometry
    ##################################################

    elif args.lidar_mode == 'odometry':
        lidar_data = load_lidar_data_csv(lidar_odometry_csv_left)
        lidar_timestamps = lidar_scan_timestamps_for_pipeline(lidar_data, args.lidar_scan_period_s)
        lidar_relative_poses = np.asarray(lidar_data.relative_poses_se3, dtype=float)

        if len(lidar_timestamps) != len(lidar_relative_poses) + 1:
            raise ValueError('Expected one more left-LiDAR scan timestamp than relative-pose measurements.')

        lidar_poses = accumulate_lidar_poses(lidar_relative_poses)

        if lidar_odometry_csv_right.is_file():
            lidar_right_data = load_lidar_data_csv(lidar_odometry_csv_right)
            lidar_right_timestamps = lidar_scan_timestamps_for_pipeline(lidar_right_data, args.lidar_scan_period_s)
            lidar_right_relative_poses = np.asarray(lidar_right_data.relative_poses_se3, dtype=float)

            if len(lidar_right_timestamps) != len(lidar_right_relative_poses) + 1:
                raise ValueError('Expected one more right-LiDAR scan timestamp than relative-pose measurements.')

            lidar_right_poses = accumulate_lidar_poses(lidar_right_relative_poses)
        else:
            lidar_right_timestamps = np.empty(0, dtype=float)
            lidar_right_relative_poses = np.empty((0, 4, 4), dtype=float)
            lidar_right_poses = np.empty((0, 4, 4), dtype=float)

        ##################################################
        # Align the local Left-LiDAR odometry frame to W
        ##################################################
        #
        # Raw accumulated odometry is:
        #
        #     T_LL0_LL(t)
        #
        # where the first LiDAR scan defines the local odometry origin.
        #
        # For trajectory initialization and later evaluation we need:
        #
        #     T_O_LL(t)
        #
        # with O parallel to the KAIST world frame W.
        #
        # We use one reference body pose only to establish the global alignment
        # of the odometry coordinate frame. The reference trajectory is NOT
        # added to the factor graph.
        ##################################################

        alignment_start_time = max(float(lidar_timestamps[0]), float(original_imu_timestamps[0]), float(true_timestamps[0]))
        alignment_index = int(np.searchsorted(lidar_timestamps, alignment_start_time, side='left'))

        if alignment_index >= len(lidar_timestamps):
            raise ValueError('Left-LiDAR odometry does not overlap the IMU/reference trajectory.')

        alignment_timestamp = float(lidar_timestamps[alignment_index])
        T_W_B_ALIGNMENT = np.asarray(data_processing._interpolate_pose(true_timestamps, true_poses, alignment_timestamp), dtype=float)
        T_W_LL_ALIGNMENT = T_W_B_ALIGNMENT @ T_B_LL_INITIAL
        T_LL0_LL_ALIGNMENT = lidar_poses[alignment_index]
        # Since
        #
        #     T_W_LL_ALIGNMENT = T_W_LL0 @ T_LL0_LL_ALIGNMENT,
        #
        # recover the world pose of the odometry origin.
        T_W_LL0 = T_W_LL_ALIGNMENT @ np.linalg.inv(T_LL0_LL_ALIGNMENT)

        GRAPH_ORIGIN_W = T_W_LL0[:3, 3].copy()

        T_O_W = np.eye(4, dtype=float)
        T_O_W[:3, 3] = -GRAPH_ORIGIN_W

        T_W_O = np.linalg.inv(T_O_W)
        T_O_LL0 = T_O_W @ T_W_LL0
        # Convert the complete accumulated odometry trajectory into graph-frame
        # absolute Left-LiDAR poses. These poses are used only for trajectory
        # initialization. LidarOdometryStream still receives the raw accumulated
        # LiDAR-frame odometry below.
        lidar_poses_left_graph = T_O_LL0 @ lidar_poses

        trajectory_source_timestamps = lidar_timestamps
        trajectory_source_sensor_poses = lidar_poses_left_graph
        right_lidar_timestamps_for_overlap = lidar_right_timestamps

        log(args.verbosity, 1, 'LiDAR input: relative odometry.')
        log(args.verbosity, 1, 'Left LiDAR relative poses:', len(lidar_relative_poses))

        if len(lidar_right_poses):
            log(args.verbosity, 1, 'Right LiDAR relative poses:', len(lidar_right_relative_poses))
        else:
            log(args.verbosity, 1, 'Right odometry CSV not found. Running with the left LiDAR only.')

        log(args.verbosity, 2, 'Left odometry timestamp range:', lidar_timestamps[0], '...', lidar_timestamps[-1])
        log(args.verbosity, 2, 'Odometry alignment timestamp:', alignment_timestamp)
        log(args.verbosity, 2, 'Recovered T_W_LL0:', T_W_LL0)
        log(args.verbosity, 2, 'KAIST graph origin in W:', GRAPH_ORIGIN_W)

    else:
        raise ValueError(f'Unknown LiDAR mode: {args.lidar_mode}')

    ##################################################
    # Physical sensors
    ##################################################

    imu = Sensor('imu_0', kind='imu')
    lidar = Sensor('lidar_0', kind='lidar')
    lidar_right = Sensor('lidar_1', kind='lidar')

    ##################################################
    # IMU streams
    ##################################################

    gravity_world = np.array([0.0, 0.0, float(args.gravity_z)])

    gyro_stream = GyroStream(
        sensor=imu,
        timestamps=original_imu_timestamps,
        angular_velocity=original_gyrs,
        samples_per_factor=args.imu_samples_per_factor,
        time_offset_margin=args.imu_time_offset_margin,
        information=args.gyro_information,
    )

    if args.acc_mode == 'complex':
        accel_stream = ComplexAccelStream(
            sensor=imu,
            timestamps=original_imu_timestamps,
            acceleration=original_accs,
            angular_velocity=original_gyrs,
            samples_per_factor=args.imu_samples_per_factor,
            time_offset_margin=args.imu_time_offset_margin,
            gravity_world=gravity_world,
            information=args.accel_information,
        )
    else:
        accel_stream = SimpleAccelStream(
            sensor=imu,
            timestamps=original_imu_timestamps,
            acceleration=original_accs,
            samples_per_factor=args.imu_samples_per_factor,
            time_offset_margin=args.imu_time_offset_margin,
            gravity_world=gravity_world,
            information=args.accel_information,
        )

    ##################################################
    # LiDAR factor information
    ##################################################

    lidar_information = np.diag([
        args.lidar_rotation_information,
        args.lidar_rotation_information,
        args.lidar_rotation_information,
        args.lidar_translation_information,
        args.lidar_translation_information,
        args.lidar_translation_information,
    ])

    lidar_map_pose_information = np.diag([
        args.map_pose_rotation_information,
        args.map_pose_rotation_information,
        args.map_pose_rotation_information,
        args.map_pose_translation_information,
        args.map_pose_translation_information,
        args.map_pose_translation_information,
    ])
    ##################################################
    # LiDAR streams
    ##################################################

    lidar_pose_left_stream = None
    lidar_pose_right_stream = None
    lidar_stream = None
    lidar_right_stream = None

    streams = [gyro_stream, accel_stream]

    ##################################################
    # Absolute LiDAR-pose mode
    ##################################################

    if args.lidar_mode == 'poses':
        lidar_pose_left_stream = LidarPoseStream(
            sensor=lidar,
            timestamps=lidar_pose_timestamps_left,
            poses=lidar_poses_left_graph,
            samples_per_factor=args.lidar_samples_per_factor,
            time_offset_margin=args.lidar_time_offset_margin,
            information=lidar_map_pose_information,
            factor_stride=args.map_pose_factor_stride,
            stream_name='lidar_0.map_pose_left',
        )

        streams.append(lidar_pose_left_stream)

        if len(lidar_poses_right):
            lidar_pose_right_stream = LidarPoseStream(
                sensor=lidar_right,
                timestamps=lidar_pose_timestamps_right,
                poses=lidar_poses_right_graph,
                samples_per_factor=args.lidar_samples_per_factor,
                time_offset_margin=args.lidar_time_offset_margin,
                information=lidar_map_pose_information,
                factor_stride=args.map_pose_factor_stride,
                stream_name='lidar_1.map_pose_right',
            )

            streams.append(lidar_pose_right_stream)

    ##################################################
    # Relative LiDAR-odometry mode
    ##################################################

    elif args.lidar_mode == 'odometry':
        lidar_stream = LidarOdometryStream(
            sensor=lidar,
            timestamps=lidar_timestamps,
            odometry_poses=lidar_poses,
            samples_per_factor=args.lidar_samples_per_factor,
            time_offset_margin=args.lidar_time_offset_margin,
            information=lidar_information,
            stream_name='lidar_0.odometry_left',
        )

        streams.append(lidar_stream)

        if len(lidar_right_poses):
            lidar_right_stream = LidarOdometryStream(
                sensor=lidar_right,
                timestamps=lidar_right_timestamps,
                odometry_poses=lidar_right_poses,
                samples_per_factor=args.lidar_samples_per_factor,
                time_offset_margin=args.lidar_time_offset_margin,
                information=lidar_information,
                stream_name='lidar_1.odometry_right',
            )

            streams.append(lidar_right_stream)

    else:
        raise ValueError(f'Unknown LiDAR mode: {args.lidar_mode}')

    log(args.verbosity, 1, 'Active streams:')

    for stream in streams:
        log(args.verbosity, 1, f'  {stream.stream_name}: {stream.stream_type} / {stream.sensor.sensor_id}')

    ##################################################
    # Calibration-variable keys
    ##################################################

    imu_extrinsic_key = VariableKey('imu_0', VariableType.EXTRINSIC)
    imu_tau_key = VariableKey('imu_0', VariableType.TIME_OFFSET)
    imu_bias_key = VariableKey('imu_0', VariableType.GYRO_BIAS)
    lidar_extrinsic_key = VariableKey('lidar_0', VariableType.EXTRINSIC)
    lidar_tau_key = VariableKey('lidar_0', VariableType.TIME_OFFSET)
    lidar_right_extrinsic_key = VariableKey('lidar_1', VariableType.EXTRINSIC)
    lidar_right_tau_key = VariableKey('lidar_1', VariableType.TIME_OFFSET)

    right_lidar_active = lidar_pose_right_stream is not None or lidar_right_stream is not None
    ##################################################
    # Calibration mode
    ##################################################

    variable_configs = build_variable_configs(
        mode=args.mode,
        imu_extrinsic_key=imu_extrinsic_key,
        imu_tau_key=imu_tau_key,
        imu_bias_key=imu_bias_key,
        lidar_extrinsic_key=lidar_extrinsic_key,
        lidar_tau_key=lidar_tau_key,
        lidar_right_extrinsic_key=lidar_right_extrinsic_key,
        lidar_right_tau_key=lidar_right_tau_key,
        right_lidar_active=right_lidar_active,
        imu_prior_source=args.imu_prior_source,
        imu_extrinsic_rotation_prior_information=args.imu_extrinsic_rotation_prior_information,
        imu_extrinsic_translation_prior_information=args.imu_extrinsic_translation_prior_information,
        imu_tau_initial=args.imu_tau_initial,
        imu_tau_prior_information=args.imu_tau_prior_information,
        lidar_extrinsic_rotation_prior_information=args.lidar_extrinsic_rotation_prior_information,
        lidar_extrinsic_translation_prior_information=args.lidar_extrinsic_translation_prior_information,
        lidar_tau_prior_information=args.lidar_tau_prior_information,
        right_lidar_tau_initial=args.right_lidar_tau_initial,
        bias_prior_information=args.bias_prior_information,
    )

    active_sensors = [imu, lidar]

    if right_lidar_active:
        active_sensors.append(lidar_right)

    ##################################################
    # Rolling graph
    ##################################################

    rolling_graph = RollingGraph(
        streams=streams,
        sensors=active_sensors,
        variable_configs=variable_configs,
        solver_config=SolverConfig(method=args.solver_method, scheduler=args.solver_scheduler, solver_verbose=args.solver_verbose),
        trajectory_config=TrajectoryConfig(anchor_first_pose=True, anchor_first_pose_each_window=True, anchor_last_pose=False, anchor_all_poses=False, use_imu_gyr=True),
    )

    ##################################################
    # Initial body trajectory
    ##################################################
    #
    # Both LiDAR modes expose:
    #
    #     trajectory_source_timestamps
    #     trajectory_source_sensor_poses
    #
    # where trajectory_source_sensor_poses contains T_O_LL.
    #
    # Therefore body initialization is always:
    #
    #     T_O_B = T_O_LL @ T_LL_B
    ##################################################

    pose_time_start = max(float(trajectory_source_timestamps[0]), float(original_imu_timestamps[0]), float(true_timestamps[0]))
    pose_time_end = min(float(trajectory_source_timestamps[-1]), float(original_imu_timestamps[-1]), float(true_timestamps[-1]))

    if right_lidar_active and len(right_lidar_timestamps_for_overlap):
        pose_time_start = max(pose_time_start, float(right_lidar_timestamps_for_overlap[0]))
        pose_time_end = min(pose_time_end, float(right_lidar_timestamps_for_overlap[-1]))

    if pose_time_end <= pose_time_start:
        raise ValueError(f'No common time interval remains for LiDAR mode "{args.lidar_mode}".')

    pose_mask = (trajectory_source_timestamps >= pose_time_start) & (trajectory_source_timestamps <= pose_time_end)
    window_pose_timestamps = trajectory_source_timestamps[pose_mask]

    if len(window_pose_timestamps) == 0:
        raise ValueError(f'No left-LiDAR trajectory samples overlap the common interval for mode "{args.lidar_mode}".')

    init_lidar_poses = trajectory_source_sensor_poses[pose_mask] @ T_LL_B_INITIAL
    init_lidar_poses_world = T_W_O @ init_lidar_poses

    first_lidar_true_index = int(np.argmin(np.abs(true_timestamps - window_pose_timestamps[0])))

    log(args.verbosity, 1, 'Initial-pose timestamp mismatch [s]:', true_timestamps[first_lidar_true_index] - window_pose_timestamps[0])
    log(args.verbosity, 1, 'Trajectory initialization poses:', init_lidar_poses.shape)
    log(args.verbosity, 1, 'Rolling window / step [s]:', args.window_size, '/', args.step_size)
    ##################################################
    # Solve rolling graph
    ##################################################

    results = rolling_graph.generate_filter_iterative(
        window_size=args.window_size,
        step_size=args.step_size,
        pose_timestamps=window_pose_timestamps,
        states=init_lidar_poses,
        first_pose=init_lidar_poses[0],
        clear_previous=True,
        verbose=args.graph_verbosity,
    )

    estimated_timestamps, estimated_poses = rolling_graph.rolling_state.rolling_trajectory
    estimated_timestamps = np.asarray(estimated_timestamps, dtype=float)
    estimated_poses = np.asarray(estimated_poses, dtype=float)

    if len(results) == 0:
        raise RuntimeError('RollingGraph produced no solved windows.')

    if len(estimated_timestamps) == 0:
        raise RuntimeError('RollingGraph produced no stitched trajectory.')

    estimated_poses_world = T_W_O @ estimated_poses

    if args.verbosity > 0:
        print()
        print('Rolling solve complete')
        print('----------------------')
        print(f'mode: {args.mode}')
        print(f'accelerometer mode: {args.acc_mode}')
        print(f'solved windows: {len(results)}')
        print(f'stitched estimated poses: {estimated_poses.shape}')

        print()
        print('Last rolling window')
        print('-------------------')
        print_rolling_result_summary(results[-1])

    ##################################################
    # Plot trajectory against reference
    ##################################################

    result_reference_mask = (true_timestamps >= estimated_timestamps[0]) & (true_timestamps <= estimated_timestamps[-1])

    trajectory_figure, _ = plot_rolling_trajectory(
        estimated_timestamps,
        estimated_poses_world,
        reference_timestamps=true_timestamps[result_reference_mask],
        reference_poses=true_poses[result_reference_mask],
        initial_poses=init_lidar_poses_world,
        title=f'KAIST RollingGraph trajectory, {args.lidar_mode}, {args.mode}',
    )

    trajectory_plot_path = plots_dir / 'trajectory_vs_reference.png'
    save_figure(trajectory_figure, trajectory_plot_path, args.dpi)

    ##################################################
    # Plot trajectory errors
    ##################################################

    trajectory_error_figure, _, trajectory_error_statistics = plot_trajectory_errors(
        estimated_timestamps=estimated_timestamps,
        estimated_poses=estimated_poses_world,
        reference_timestamps=true_timestamps,
        reference_poses=true_poses,
        title=f'RollingGraph Trajectory Errors, {args.lidar_mode}, {args.mode}',
        relative_time=True,
    )

    trajectory_error_plot_path = plots_dir / 'trajectory_errors.png'
    save_figure(trajectory_error_figure, trajectory_error_plot_path, args.dpi)

    trajectory_error_series = compute_trajectory_error_series(estimated_timestamps, estimated_poses_world, true_timestamps, true_poses)

    ##################################################
    # Plot localization benchmark errors
    ##################################################

    benchmark_figure, _, benchmark_statistics = plot_lidar_localization_benchmark_errors(
        estimated_timestamps,
        estimated_poses_world,
        reference_timestamps=true_timestamps,
        reference_poses=true_poses,
    )

    benchmark_plot_path = plots_dir / 'benchmark_errors.png'
    save_figure(benchmark_figure, benchmark_plot_path, args.dpi)

    run_config = {
        'dataset_root': str(dataset_root),
        'processed_data_dir': str(processed_data_dir),
        'mode': args.mode,
        'acc_mode': args.acc_mode,
        'window_size_s': float(args.window_size),
        'step_size_s': float(args.step_size),
        'imu_frequency_hz': float(args.imu_frequency_hz),
        'lidar_mode': args.lidar_mode,
        'map_pose_factor_stride': int(args.map_pose_factor_stride),
        'gyro_information': float(args.gyro_information),
        'accel_information': float(args.accel_information),
        'lidar_rotation_information': float(args.lidar_rotation_information),
        'lidar_translation_information': float(args.lidar_translation_information),
        'map_pose_rotation_information': float(args.map_pose_rotation_information),
        'map_pose_translation_information': float(args.map_pose_translation_information),
        'imu_tau_initial_s': float(args.imu_tau_initial),
        'right_lidar_tau_initial_s': float(args.right_lidar_tau_initial),
        'imu_extrinsic_rotation_prior_information': float(args.imu_extrinsic_rotation_prior_information),
        'imu_extrinsic_translation_prior_information': float(args.imu_extrinsic_translation_prior_information),
        'imu_tau_prior_information': float(args.imu_tau_prior_information),
        'lidar_extrinsic_rotation_prior_information': float(args.lidar_extrinsic_rotation_prior_information),
        'lidar_extrinsic_translation_prior_information': float(args.lidar_extrinsic_translation_prior_information),
        'lidar_tau_prior_information': float(args.lidar_tau_prior_information),
        'bias_prior_information': float(args.bias_prior_information),
        'gravity_z_mps2': float(args.gravity_z),
        'estimated_pose_count': int(len(estimated_timestamps)),
        'error_sample_count': int(len(trajectory_error_series['timestamps_s'])),
        'trajectory_start_time_s': float(estimated_timestamps[0]),
        'trajectory_end_time_s': float(estimated_timestamps[-1]),
    }

    trajectory_error_pickle_path = errors_dir / 'trajectory_errors.pkl'
    trajectory_csv_path = errors_dir / 'trajectory.csv'

    save_trajectory_error_pickle(
        trajectory_error_pickle_path,
        dataset_name=dataset_name,
        mode=args.mode,
        lidar_mode=args.lidar_mode,
        error_series=trajectory_error_series,
        trajectory_statistics=trajectory_error_statistics,
        benchmark_statistics=benchmark_statistics,
        run_config=run_config,
    )

    save_trajectory_csv(trajectory_csv_path, trajectory_error_series)

    ##################################################
    # Plot calibration estimates / errors
    ##################################################
    calibration_keys = [imu_extrinsic_key, imu_tau_key, imu_bias_key, lidar_extrinsic_key, lidar_tau_key]

    reference_values = {
        imu_extrinsic_key: T_B_I_REFERENCE,
        imu_tau_key: TAU_I_REFERENCE,
        imu_bias_key: BIAS_REFERENCE,
        lidar_extrinsic_key: T_B_L_REFERENCE,
        lidar_tau_key: TAU_L_REFERENCE,
    }

    if right_lidar_active:
        calibration_keys.extend([lidar_right_extrinsic_key, lidar_right_tau_key])
        reference_values[lidar_right_extrinsic_key] = T_B_RL_INITIAL
        reference_values[lidar_right_tau_key] = TAU_L_REFERENCE

    ##################################################
    # Save calibration-variable histories
    ##################################################

    calibration_variables_pickle_path = errors_dir / 'calibration_variables.pkl'

    save_calibration_variables_pickle(
        calibration_variables_pickle_path,
        dataset_name=dataset_name,
        mode=args.mode,
        lidar_mode=args.lidar_mode,
        results=results,
        calibration_keys=calibration_keys,
        reference_values=reference_values,
        variable_configs=variable_configs,
        run_config=run_config,
    )

    calibration_figure, _ = plot_calibration_estimates(results, calibration_keys, reference_values=reference_values)
    calibration_plot_path = plots_dir / 'calibration_errors.png'
    save_figure(calibration_figure, calibration_plot_path, args.dpi)

    ##################################################
    # Console report
    ##################################################

    if args.verbosity > 0:
        print()
        print('Trajectory error statistics')
        print('---------------------------')
        print(json.dumps(jsonable(trajectory_error_statistics), indent=2))

        print()
        print('Localization benchmark statistics')
        print('---------------------------------')
        print(json.dumps(jsonable(benchmark_statistics), indent=2))

    if args.verbosity > -1:
        print()
        print('Saved outputs')
        print('-------------')
        print(trajectory_plot_path)
        print(trajectory_error_plot_path)
        print(benchmark_plot_path)
        print(calibration_plot_path)
        print(trajectory_error_pickle_path)
        print(trajectory_csv_path)
        print(calibration_variables_pickle_path)


##################################################
# Entrypoint
##################################################


def main() -> None:
    '''Parse CLI arguments and run the KAIST pipeline.'''

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.window_size <= 0.0:
        parser.error('--window-size must be positive.')

    if args.step_size <= 0.0:
        parser.error('--step-size must be positive.')

    if args.step_size > args.window_size:
        parser.error('--step-size must not exceed --window-size.')

    if args.map_pose_factor_stride <= 0:
        parser.error('--map-pose-factor-stride must be positive.')

    if args.imu_frequency_hz <= 0.0:
        parser.error('--imu-frequency-hz must be positive.')

    prior_information_arguments = {
        '--imu-extrinsic-rotation-prior-information': args.imu_extrinsic_rotation_prior_information,
        '--imu-extrinsic-translation-prior-information': args.imu_extrinsic_translation_prior_information,
        '--imu-tau-prior-information': args.imu_tau_prior_information,
        '--lidar-extrinsic-rotation-prior-information': args.lidar_extrinsic_rotation_prior_information,
        '--lidar-extrinsic-translation-prior-information': args.lidar_extrinsic_translation_prior_information,
        '--lidar-tau-prior-information': args.lidar_tau_prior_information,
        '--bias-prior-information': args.bias_prior_information,
    }

    for argument_name, information in prior_information_arguments.items():
        if not np.isfinite(information) or information <= 0.0:
            parser.error(f'{argument_name} must be finite and positive.')

    run_pipeline(args)


if __name__ == '__main__':
    main()

# python run_kaist_multi_sensor_pipeline.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --lidar-mode poses --mode with-calibration --output-dir experiment_results
# python run_kaist_multi_sensor_pipeline.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --lidar-mode poses --mode without-calibration --output-dir experiment_results
# python run_kaist_multi_sensor_pipeline.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --lidar-mode odometry --mode with-calibration --output-dir experiment_results
# python run_kaist_multi_sensor_pipeline.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --lidar-mode odometry --mode without-calibration --output-dir experiment_results
