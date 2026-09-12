'''Estimate dataset-wide KAIST measurement-noise covariances for observability analysis.

The script estimates effective measurement noise from all selected KAIST Urban
sequences and prints the resulting covariance matrices to the console.

IMU noise:
    Low-dynamics intervals are detected from the resampled IMU stream. For each
    contiguous interval, the per-axis median is removed before pooling samples.
    This removes local gyro bias and the approximately constant gravity vector.
    The resulting covariance therefore describes high-frequency measurement
    noise plus vehicle vibration at the selected IMU sampling frequency.

LiDAR scan-to-map pose noise:
    Each successful absolute LiDAR map pose T_W_L is compared with the pose
    predicted from the KAIST reference body trajectory and the nominal LiDAR
    extrinsic. A per-sequence/per-LiDAR median residual is removed before
    pooling, so fixed calibration/world-alignment offsets are not counted as
    random measurement noise. Rotation residuals are represented as SO(3)
    rotation vectors and translation residuals in the local relative-error
    transform.

The printed "isotropic sigma" is sqrt(trace(covariance) / 3), useful when the
observability configuration accepts one scalar standard deviation per 3-vector.

Examples:

    python estimate_kaist_measurement_noise.py /mnt/d/Downloads/MobRobLab/KAISTDataset

    python estimate_kaist_measurement_noise.py /mnt/d/Downloads/MobRobLab/KAISTDataset --datasets Urban08 Urban13 Urban16 Urban38 Urban39

    python estimate_kaist_measurement_noise.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16
'''

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation


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

    raise FileNotFoundError("Could not locate the project root containing 'src/kaist_dataset'. Run this script from inside the repository or place it inside the repository.")


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
from kaist_dataset.lidar_map import load_lidar_map_pose_csv


##################################################
# KAIST nominal LiDAR extrinsics
##################################################


T_B_LL = np.array([[-0.514066, -0.702201, -0.492595, -0.440699], [0.486485, -0.711672, 0.506809, 0.397052], [-0.706447, 0.0208933, 0.707457, 1.90953], [0.0, 0.0, 0.0, 1.0]], dtype=float)

T_B_RL = np.array([[-0.512152, 0.699241, -0.498761, -0.449885], [-0.494811, -0.714859, -0.494104, -0.416713], [-0.702041, -0.0062642, 0.712109, 1.91294], [0.0, 0.0, 0.0, 1.0]], dtype=float)


##################################################
# Xsens MTi-300 datasheet reference
##################################################


MTI300_GYRO_NOISE_DENSITY_DEGPS_SQRT_HZ = 0.01
MTI300_ACCEL_NOISE_DENSITY_UG_SQRT_HZ = 60.0
STANDARD_GRAVITY_MPS2 = 9.80665


##################################################
# Dataset helpers
##################################################


def natural_sort_key(value: str) -> list[object]:
    '''Return a natural sorting key.'''

    return [int(component) if component.isdigit() else component.lower() for component in re.split(r'(\d+)', str(value))]


def discover_dataset_directories(root: Path, requested_names: list[str] | None) -> list[Path]:
    '''Return one or more KAIST Urban dataset directories.'''

    root = root.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f'Dataset path does not exist: {root}')

    if re.fullmatch(r'Urban\d+', root.name, flags=re.IGNORECASE):
        dataset_dirs = [root]
    else:
        dataset_dirs = sorted([path for path in root.iterdir() if path.is_dir() and re.fullmatch(r'Urban\d+', path.name, flags=re.IGNORECASE)], key=lambda path: natural_sort_key(path.name))

    if requested_names:
        requested_set = set(requested_names)
        dataset_dirs = [path for path in dataset_dirs if path.name in requested_set]
        missing = sorted(requested_set.difference(path.name for path in dataset_dirs), key=natural_sort_key)

        if missing:
            raise FileNotFoundError(f'Requested dataset directories were not found: {missing}')

    if not dataset_dirs:
        raise FileNotFoundError(f'No UrbanXX dataset directories found under {root}')

    return dataset_dirs


def dataset_layout(dataset_dir: Path) -> tuple[Path, Path]:
    '''Return the KAIST sensor-data directory and reference trajectory path.'''

    dataset_slug = dataset_dir.name.lower()
    sensor_data_root = dataset_dir / f'{dataset_slug}_data' / dataset_slug / 'sensor_data'
    reference_pose_path = dataset_dir / f'{dataset_slug}_pose' / dataset_slug / 'global_pose.csv'

    if not sensor_data_root.is_dir():
        raise FileNotFoundError(f'Sensor-data directory was not found: {sensor_data_root}')

    if not reference_pose_path.is_file():
        raise FileNotFoundError(f'Reference trajectory was not found: {reference_pose_path}')

    return sensor_data_root, reference_pose_path


def resolve_processed_data_directory(dataset_dir: Path) -> Path | None:
    '''Find a directory containing at least one LiDAR map-pose CSV.'''

    candidates = [dataset_dir, dataset_dir / 'data', PROJECT_ROOT / 'data' / 'KAISTDataset' / dataset_dir.name]

    for candidate in candidates:
        if (candidate / 'lidar_map_poses_vlp_left.csv').is_file() or (candidate / 'lidar_map_poses_vlp_right.csv').is_file():
            return candidate.resolve()

    return None


##################################################
# Statistics
##################################################


def covariance_and_sigma(samples: np.ndarray) -> tuple[np.ndarray, float]:
    '''Return sample covariance and equivalent isotropic one-axis sigma.'''

    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 2:
        raise ValueError(f'Expected at least two 3D samples, got {samples.shape}')

    covariance = np.cov(samples, rowvar=False, ddof=1)
    isotropic_sigma = float(np.sqrt(np.trace(covariance) / 3.0))
    return covariance, isotropic_sigma


def robust_inlier_mask(samples: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    '''Return a component-wise MAD inlier mask for pooled residual samples.'''

    samples = np.asarray(samples, dtype=float)
    center = np.median(samples, axis=0)
    absolute_deviation = np.abs(samples - center)
    mad = np.median(absolute_deviation, axis=0)
    robust_sigma = 1.4826 * mad
    fallback_sigma = np.std(samples, axis=0, ddof=1)
    scale = np.where(robust_sigma > 1e-12, robust_sigma, np.maximum(fallback_sigma, 1e-12))
    return np.all(absolute_deviation <= threshold * scale, axis=1)


##################################################
# IMU noise estimation
##################################################


def contiguous_true_segments(mask: np.ndarray, minimum_samples: int) -> list[slice]:
    '''Return contiguous True regions with at least minimum_samples samples.'''

    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    segments = []

    for start, stop in zip(changes[0::2], changes[1::2]):
        if stop - start >= minimum_samples:
            segments.append(slice(int(start), int(stop)))

    return segments


def estimate_imu_residuals(dataset_dir: Path, imu_frequency_hz: float, minimum_segment_s: float, gyro_threshold_radps: float, accel_norm_threshold_mps2: float) -> tuple[np.ndarray, np.ndarray]:
    '''Estimate centered gyro and accelerometer residuals from low-dynamics intervals.'''

    sensor_data_root, _ = dataset_layout(dataset_dir)
    imu_streams = load_imus(sensor_data_root, target_frequency_hz=imu_frequency_hz)

    if not imu_streams:
        raise RuntimeError(f'No IMU streams were loaded from {sensor_data_root}')

    imu = imu_streams[list(imu_streams.keys())[0]]
    gyro = np.asarray(imu.gyro_radps, dtype=float)
    accel = np.asarray(imu.accel_mps2, dtype=float)

    gyro_norm = np.linalg.norm(gyro, axis=1)
    accel_norm = np.linalg.norm(accel, axis=1)
    low_dynamics_mask = (gyro_norm <= gyro_threshold_radps) & (np.abs(accel_norm - STANDARD_GRAVITY_MPS2) <= accel_norm_threshold_mps2)
    minimum_samples = max(2, int(round(minimum_segment_s * imu_frequency_hz)))
    segments = contiguous_true_segments(low_dynamics_mask, minimum_samples)

    gyro_residuals = []
    accel_residuals = []

    for segment in segments:
        gyro_segment = gyro[segment]
        accel_segment = accel[segment]
        gyro_residuals.append(gyro_segment - np.median(gyro_segment, axis=0))
        accel_residuals.append(accel_segment - np.median(accel_segment, axis=0))

    if not gyro_residuals:
        raise RuntimeError(f'No low-dynamics interval of at least {minimum_segment_s:g} s was found in {dataset_dir.name}. Try relaxing --gyro-threshold-radps or --accel-norm-threshold-mps2.')

    return np.concatenate(gyro_residuals, axis=0), np.concatenate(accel_residuals, axis=0)


##################################################
# LiDAR scan-to-map noise estimation
##################################################


def load_successful_map_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    '''Load successful finite map-pose estimates.'''

    timestamps, poses, dataframe = load_lidar_map_pose_csv(path)
    timestamps = np.asarray(timestamps, dtype=float)
    poses = np.asarray(poses, dtype=float)
    success_mask = dataframe['success'].to_numpy(dtype=bool) if 'success' in dataframe.columns else np.ones(len(dataframe), dtype=bool)
    finite_mask = np.all(np.isfinite(poses), axis=(1, 2))
    mask = success_mask & finite_mask & np.isfinite(timestamps)
    return timestamps[mask], poses[mask]


def estimate_lidar_residuals(dataset_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    '''Return centered scan-to-map rotation and translation residual samples for all available LiDAR streams.'''

    processed_data_dir = resolve_processed_data_directory(dataset_dir)

    if processed_data_dir is None:
        return [], []

    _, reference_pose_path = dataset_layout(dataset_dir)
    reference_timestamps, reference_poses = import_true_trajectory(reference_pose_path)
    reference_timestamps = np.asarray(reference_timestamps, dtype=float)
    reference_poses = np.asarray(reference_poses, dtype=float)
    rotation_groups = []
    translation_groups = []

    for filename, T_B_L in (('lidar_map_poses_vlp_left.csv', T_B_LL), ('lidar_map_poses_vlp_right.csv', T_B_RL)):
        path = processed_data_dir / filename

        if not path.is_file():
            continue

        timestamps, measured_poses = load_successful_map_poses(path)
        overlap_mask = (timestamps >= reference_timestamps[0]) & (timestamps <= reference_timestamps[-1])
        timestamps = timestamps[overlap_mask]
        measured_poses = measured_poses[overlap_mask]

        if len(timestamps) < 2:
            continue

        rotation_residuals = np.empty((len(timestamps), 3), dtype=float)
        translation_residuals = np.empty((len(timestamps), 3), dtype=float)

        for sample_index, (timestamp, measured_pose) in enumerate(zip(timestamps, measured_poses)):
            T_W_B_reference = np.asarray(data_processing._interpolate_pose(reference_timestamps, reference_poses, float(timestamp)), dtype=float)
            T_W_L_reference = T_W_B_reference @ T_B_L
            relative_error = np.linalg.inv(T_W_L_reference) @ measured_pose
            rotation_residuals[sample_index] = Rotation.from_matrix(relative_error[:3, :3]).as_rotvec()
            translation_residuals[sample_index] = relative_error[:3, 3]

        rotation_residuals -= np.median(rotation_residuals, axis=0)
        translation_residuals -= np.median(translation_residuals, axis=0)
        rotation_groups.append(rotation_residuals)
        translation_groups.append(translation_residuals)

    return rotation_groups, translation_groups


##################################################
# Console reporting
##################################################


def print_covariance(name: str, covariance: np.ndarray, sigma: float, unit: str, samples: int) -> None:
    '''Print one covariance estimate.'''

    print()
    print(name)
    print('-' * len(name))
    print(f'samples: {samples}')
    print(f'covariance [{unit}^2]:')
    print(np.array2string(covariance, precision=8, suppress_small=False))
    print(f'isotropic sigma [{unit}]: {sigma:.8g}')


##################################################
# Main
##################################################


def build_argument_parser() -> argparse.ArgumentParser:
    '''Construct the CLI.'''

    parser = argparse.ArgumentParser(description='Estimate pooled KAIST IMU and LiDAR scan-to-map measurement-noise covariances.')
    parser.add_argument('dataset_root', type=Path, help='KAISTDataset root or one UrbanXX dataset directory.')
    parser.add_argument('--datasets', nargs='*', default=None, help='Optional UrbanXX names to include when dataset_root is the KAISTDataset parent directory.')
    parser.add_argument('--imu-frequency-hz', type=float, default=100.0, help='IMU frequency at which per-sample covariance is estimated. Default: 100 Hz, matching the observability pipeline.')
    parser.add_argument('--minimum-low-dynamics-segment-s', type=float, default=2.0, help='Minimum contiguous low-dynamics interval length.')
    parser.add_argument('--gyro-threshold-radps', type=float, default=0.02, help='Maximum gyro norm used to identify low-dynamics samples.')
    parser.add_argument('--accel-norm-threshold-mps2', type=float, default=0.15, help='Maximum |norm(a)-g| used to identify low-dynamics samples.')
    parser.add_argument('--lidar-outlier-sigma', type=float, default=5.0, help='Component-wise MAD threshold used before calculating LiDAR covariance.')
    return parser


def main() -> None:
    '''Estimate and print pooled measurement-noise covariances.'''

    args = build_argument_parser().parse_args()
    dataset_dirs = discover_dataset_directories(args.dataset_root, args.datasets)

    gyro_groups = []
    accel_groups = []
    lidar_rotation_groups = []
    lidar_translation_groups = []

    print('Datasets:', ', '.join(path.name for path in dataset_dirs))

    for dataset_dir in dataset_dirs:
        try:
            gyro_residuals, accel_residuals = estimate_imu_residuals(dataset_dir, args.imu_frequency_hz, args.minimum_low_dynamics_segment_s, args.gyro_threshold_radps, args.accel_norm_threshold_mps2)
            gyro_groups.append(gyro_residuals)
            accel_groups.append(accel_residuals)
            print(f'{dataset_dir.name}: IMU low-dynamics samples = {len(gyro_residuals)}')
        except Exception as error:
            print(f'{dataset_dir.name}: IMU skipped: {error}')

        try:
            rotation_groups, translation_groups = estimate_lidar_residuals(dataset_dir)
            lidar_rotation_groups.extend(rotation_groups)
            lidar_translation_groups.extend(translation_groups)
            print(f'{dataset_dir.name}: LiDAR residual groups = {len(rotation_groups)}')
        except Exception as error:
            print(f'{dataset_dir.name}: LiDAR skipped: {error}')

    if not gyro_groups or not accel_groups:
        raise RuntimeError('No IMU residual samples were collected.')

    gyro_residuals = np.concatenate(gyro_groups, axis=0)
    accel_residuals = np.concatenate(accel_groups, axis=0)
    gyro_covariance, gyro_sigma = covariance_and_sigma(gyro_residuals)
    accel_covariance, accel_sigma = covariance_and_sigma(accel_residuals)

    print_covariance('Empirical gyro measurement noise', gyro_covariance, gyro_sigma, 'rad/s', len(gyro_residuals))
    print_covariance('Empirical accelerometer measurement noise', accel_covariance, accel_sigma, 'm/s^2', len(accel_residuals))

    gyro_noise_density_radps_sqrt_hz = np.deg2rad(MTI300_GYRO_NOISE_DENSITY_DEGPS_SQRT_HZ)
    accel_noise_density_mps2_sqrt_hz = MTI300_ACCEL_NOISE_DENSITY_UG_SQRT_HZ * 1e-6 * STANDARD_GRAVITY_MPS2
    datasheet_gyro_sigma = gyro_noise_density_radps_sqrt_hz * np.sqrt(args.imu_frequency_hz / 2.0)
    datasheet_accel_sigma = accel_noise_density_mps2_sqrt_hz * np.sqrt(args.imu_frequency_hz / 2.0)

    print()
    print('MTi-300 datasheet white-noise reference')
    print('---------------------------------------')
    print(f'gyro noise density:        {MTI300_GYRO_NOISE_DENSITY_DEGPS_SQRT_HZ:g} deg/s/sqrt(Hz)')
    print(f'accel noise density:       {MTI300_ACCEL_NOISE_DENSITY_UG_SQRT_HZ:g} ug/sqrt(Hz)')
    print(f'per-sample gyro sigma at {args.imu_frequency_hz:g} Hz:  {datasheet_gyro_sigma:.8g} rad/s')
    print(f'per-sample accel sigma at {args.imu_frequency_hz:g} Hz: {datasheet_accel_sigma:.8g} m/s^2')
    print(f'empirical / datasheet gyro sigma ratio:  {gyro_sigma / datasheet_gyro_sigma:.4g}')
    print(f'empirical / datasheet accel sigma ratio: {accel_sigma / datasheet_accel_sigma:.4g}')

    if lidar_rotation_groups and lidar_translation_groups:
        lidar_rotation_residuals = np.concatenate(lidar_rotation_groups, axis=0)
        lidar_translation_residuals = np.concatenate(lidar_translation_groups, axis=0)
        rotation_inliers = robust_inlier_mask(lidar_rotation_residuals, args.lidar_outlier_sigma)
        translation_inliers = robust_inlier_mask(lidar_translation_residuals, args.lidar_outlier_sigma)
        joint_inliers = rotation_inliers & translation_inliers
        lidar_rotation_inliers = lidar_rotation_residuals[joint_inliers]
        lidar_translation_inliers = lidar_translation_residuals[joint_inliers]
        lidar_rotation_covariance, lidar_rotation_sigma = covariance_and_sigma(lidar_rotation_inliers)
        lidar_translation_covariance, lidar_translation_sigma = covariance_and_sigma(lidar_translation_inliers)

        print()
        print(f'LiDAR robust inliers: {len(lidar_rotation_inliers)} / {len(lidar_rotation_residuals)} ({100.0 * len(lidar_rotation_inliers) / len(lidar_rotation_residuals):.2f}%)')
        print_covariance('Empirical LiDAR scan-to-map rotation noise', lidar_rotation_covariance, lidar_rotation_sigma, 'rad', len(lidar_rotation_inliers))
        print_covariance('Empirical LiDAR scan-to-map translation noise', lidar_translation_covariance, lidar_translation_sigma, 'm', len(lidar_translation_inliers))
        print()
        print('Suggested temporary scalar LiDAR values from this run')
        print('-----------------------------------------------------')
        print(f'lidar_rotation_noise_std = {lidar_rotation_sigma:.8g}')
        print(f'lidar_translation_noise_std = {lidar_translation_sigma:.8g}')
    else:
        print()
        print('No LiDAR map-pose residuals were collected. The IMU estimates above are still valid.')


if __name__ == '__main__':
    main()

# python estimate_kaist_measurement_noise.py /mnt/d/Downloads/MobRobLab/KAISTDataset

'''
Empirical gyro measurement noise
--------------------------------
samples: 492985
covariance [rad/s^2]:
[[ 2.96367932e-06 -1.94438078e-07 -4.83643437e-07]
 [-1.94438078e-07  1.89162212e-06  1.09062693e-07]
 [-4.83643437e-07  1.09062693e-07  1.88843506e-06]]
isotropic sigma [rad/s]: 0.0014993039

Empirical accelerometer measurement noise
-----------------------------------------
samples: 492985
covariance [m/s^2^2]:
[[ 4.85003041e-03 -9.25262862e-05  2.14219676e-04]
 [-9.25262862e-05  1.51980006e-03 -1.13182999e-03]
 [ 2.14219676e-04 -1.13182999e-03  2.29777687e-03]]
isotropic sigma [m/s^2]: 0.053751302

MTi-300 datasheet white-noise reference
---------------------------------------
gyro noise density:        0.01 deg/s/sqrt(Hz)
accel noise density:       60 ug/sqrt(Hz)
per-sample gyro sigma at 100 Hz:  0.0012341341 rad/s
per-sample accel sigma at 100 Hz: 0.0041606092 m/s^2
empirical / datasheet gyro sigma ratio:  1.215
empirical / datasheet accel sigma ratio: 12.92

LiDAR robust inliers: 300896 / 302763 (99.38%)

Empirical LiDAR scan-to-map rotation noise
------------------------------------------
samples: 300896
covariance [rad^2]:
[[1.10442767e-05 9.46809587e-07 3.78536515e-06]
 [9.46809587e-07 3.35041997e-05 1.90008728e-06]
 [3.78536515e-06 1.90008728e-06 1.10553483e-05]]
isotropic sigma [rad]: 0.0043051839

Empirical LiDAR scan-to-map translation noise
---------------------------------------------
samples: 300896
covariance [m^2]:
[[0.00843466 0.00266052 0.00702046]
 [0.00266052 0.0144485  0.00225705]
 [0.00702046 0.00225705 0.00704268]]
isotropic sigma [m]: 0.099876337

Suggested temporary scalar LiDAR values from this run
-----------------------------------------------------
lidar_rotation_noise_std = 0.0043051839
lidar_translation_noise_std = 0.099876337
'''