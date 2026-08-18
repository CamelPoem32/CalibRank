
import copy
import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mrob
import numpy as np
from numba import njit
try:
    from tools_ahrs import plot
    import transform
except ImportError:  # pragma: no cover - supports importing this module as src.data_processing.
    from src.tools_ahrs import plot
    from src import transform

def subsample_indices(indices: Iterable[int], final_length: Optional[int]) -> np.ndarray:
    """
    Select almost equally spaced values from a sequence of integer indices.

    The first and last input indices are always preserved. If final_length is
    None or is not smaller than the input length, all indices are returned.
    """
    indices = np.asarray(list(indices), dtype=np.int64).reshape(-1)

    if len(indices) == 0:
        return indices.copy()

    if final_length is None or final_length >= len(indices):
        return indices.copy()

    if final_length < 2:
        raise ValueError("final_length must be at least 2 when subsampling measurement support")

    positions = np.rint(np.linspace(0, len(indices) - 1, int(final_length))).astype(np.int64)
    positions[0] = 0
    positions[-1] = len(indices) - 1

    if len(np.unique(positions)) != len(positions):
        raise RuntimeError("Failed to generate unique equally spaced subsampling positions")

    return indices[positions]


def select_time_support_indices(timestamps: Sequence[float], support_start: float, support_end: float, final_length: Optional[int]) -> np.ndarray:
    """
    Return subsampled indices whose first and last timestamps bracket a time interval.

    One sample before support_start and one sample after support_end are included
    whenever they exist. This is important because the C++ factors interpolate
    measurements at shifted query times.
    """
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)

    if len(timestamps) < 2:
        raise ValueError("At least two timestamped measurements are required")

    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Measurement timestamps must be finite")

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Measurement timestamps must be strictly increasing")

    if support_end <= support_start:
        raise ValueError("support_end must be greater than support_start")

    if support_start < timestamps[0] or support_end > timestamps[-1]:
        raise IndexError(f"Requested support [{support_start}, {support_end}] lies outside measurement timestamps [{timestamps[0]}, {timestamps[-1]}]")

    i_start = max(int(np.searchsorted(timestamps, support_start, side="right")) - 1, 0)
    i_end = min(int(np.searchsorted(timestamps, support_end, side="left")) + 1, len(timestamps))

    indices = np.arange(i_start, i_end, dtype=np.int64)

    if len(indices) < 2:
        raise ValueError("Selected measurement support contains fewer than two samples")

    return subsample_indices(indices, final_length)


def _as_vector3(values: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")

    return values.copy()


def _as_pose_matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, mrob.SE3):
        return np.asarray(pose.T(), dtype=float).copy()

    pose = np.asarray(pose, dtype=float)

    if pose.shape == (6,):
        return np.asarray(mrob.SE3(pose).T(), dtype=float)

    if pose.shape != (4, 4):
        raise ValueError("A pose must be an mrob.SE3 object, a six-dimensional tangent vector, or a 4x4 transformation matrix")

    if not np.all(np.isfinite(pose)):
        raise ValueError("Pose matrix must be finite")

    if not mrob.isSE3(pose):
        raise ValueError("Pose matrix must be a valid SE(3) transformation")

    return pose.copy()


def _as_mrob_se3(pose: Any) -> mrob.SE3:
    if isinstance(pose, mrob.SE3):
        return mrob.SE3(pose)

    pose = np.asarray(pose, dtype=float)

    if pose.shape == (6,):
        return mrob.SE3(pose)

    return mrob.SE3(_as_pose_matrix(pose))


def _interpolate_vector(timestamps: Sequence[float], values: np.ndarray, query_time: float) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)

    if values.shape != (len(timestamps), 3):
        raise ValueError("Interpolated vector measurements must have shape (N, 3)")

    if query_time < timestamps[0] or query_time > timestamps[-1]:
        raise IndexError(f"Query time {query_time} lies outside measurement support [{timestamps[0]}, {timestamps[-1]}]")

    i_upper = int(np.searchsorted(timestamps, query_time, side="left"))

    if i_upper == 0:
        return values[0].copy()

    if i_upper >= len(timestamps):
        return values[-1].copy()

    if timestamps[i_upper] == query_time:
        return values[i_upper].copy()

    i_lower = i_upper - 1
    alpha = (query_time - timestamps[i_lower]) / (timestamps[i_upper] - timestamps[i_lower])
    return (1.0 - alpha) * values[i_lower] + alpha * values[i_upper]

def _interpolate_pose(timestamps: Sequence[float], poses: Sequence[Any], query_time: float) -> np.ndarray:
    """
    Interpolate an SE(3) pose along the relative Lie-algebra increment.
    """
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)

    if query_time < timestamps[0] or query_time > timestamps[-1]:
        raise IndexError(f"Query time {query_time} lies outside pose support [{timestamps[0]}, {timestamps[-1]}]")

    i_upper = int(np.searchsorted(timestamps, query_time, side="left"))

    if i_upper == 0:
        return _as_pose_matrix(poses[0])

    if i_upper >= len(timestamps):
        return _as_pose_matrix(poses[-1])

    if timestamps[i_upper] == query_time:
        return _as_pose_matrix(poses[i_upper])

    i_lower = i_upper - 1
    alpha = (query_time - timestamps[i_lower]) / (timestamps[i_upper] - timestamps[i_lower])

    T_left = _as_mrob_se3(poses[i_lower])
    T_right = _as_mrob_se3(poses[i_upper])
    relative_xi = T_left.inv().mul(T_right).Ln()

    return np.asarray(T_left.mul(mrob.SE3(alpha * relative_xi)).T(), dtype=float)

def _integrate_vector_interval(timestamps: Sequence[float], values: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    """
    Integrate piecewise-linearly interpolated three-dimensional measurements.
    """
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)

    if values.shape != (len(timestamps), 3):
        raise ValueError("Integrated vector measurements must have shape (N, 3)")

    if start_time < timestamps[0] or end_time > timestamps[-1]:
        raise IndexError(f"Integration interval [{start_time}, {end_time}] lies outside measurement support [{timestamps[0]}, {timestamps[-1]}]")

    interior_times = timestamps[(timestamps > start_time) & (timestamps < end_time)]
    integration_times = np.concatenate(([start_time], interior_times, [end_time]))
    integration_values = np.vstack([_interpolate_vector(timestamps, values, time) for time in integration_times])
    dt = np.diff(integration_times)

    return np.sum(0.5 * dt[:, None] * (integration_values[:-1] + integration_values[1:]), axis=0)


def _information_matrix(value: Any, dimension: int, factor_index: int = 0) -> np.ndarray:
    """
    Convert a scalar, one matrix, or a sequence of matrices to one information matrix.
    """
    if value is None:
        return np.eye(dimension)

    value = np.asarray(value, dtype=float)

    if value.ndim == 0:
        matrix = np.eye(dimension) * float(value)
    elif value.shape == (dimension, dimension):
        matrix = value
    elif value.ndim == 3 and value.shape[1:] == (dimension, dimension):
        matrix = value[min(factor_index, len(value) - 1)]
    else:
        raise ValueError(f"Information must be a scalar, a ({dimension}, {dimension}) matrix, or an (N, {dimension}, {dimension}) array")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Information matrix must be finite")

    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Information matrix must be symmetric")

    return matrix.copy()

def _initialize_trajectory_poses(
    pose_timestamps: Sequence[float],
    states: Optional[Sequence[Any]],
    first_pose: Any,
    imu_timestamps: Optional[Sequence[float]],
    angular_velocity_imu: Optional[np.ndarray],
    T_B_I_initial: Any,
    bias_initial: Sequence[float],
    tau_I_initial: float,
    lidar_timestamps: Optional[Sequence[float]] = None,
    lidar_odometry_poses: Optional[Sequence[Any]] = None,
    T_B_L_initial: Any = None,
    tau_L_initial: float = 0.0,
    use_imu_gyr: bool = False,
) -> np.ndarray:
    '''
    Use supplied poses first, then initialize unavailable trajectory poses.

    When use_imu_gyr is False, propagate both rotation and translation using
    the accumulated LiDAR odometry trajectory T_O_L(t).

    When use_imu_gyr is True, propagate rotation using gyroscope integration
    and propagate only translation using the LiDAR odometry trajectory.
    '''

    pose_timestamps = np.asarray(pose_timestamps, dtype=float).reshape(-1)
    supplied_states = [] if states is None else list(states)
    initial_poses = [_as_pose_matrix(state) for state in supplied_states[:len(pose_timestamps)]]

    if len(initial_poses) == 0: initial_poses.append(_as_pose_matrix(first_pose))

    if len(initial_poses) == len(pose_timestamps): return np.asarray(initial_poses)

    use_imu_gyr = bool(use_imu_gyr)
    use_lidar_odometry = (lidar_timestamps is not None and lidar_odometry_poses is not None)

    if not use_imu_gyr and not use_lidar_odometry: raise ValueError("lidar_timestamps and lidar_odometry_poses are required when use_imu_gyr is False")

    # Prepare gyroscope measurements only when IMU rotation propagation is used.
    if use_imu_gyr:
        if imu_timestamps is None or angular_velocity_imu is None: raise ValueError("imu_timestamps and angular_velocity_imu are required when use_imu_gyr is True")

        imu_timestamps = np.asarray(imu_timestamps, dtype=float).reshape(-1)

        angular_velocity_imu = np.asarray(angular_velocity_imu, dtype=float)

        if angular_velocity_imu.shape != (len(imu_timestamps), 3): raise ValueError("angular_velocity_imu must have shape (len(imu_timestamps), 3)")

        bias_initial = _as_vector3(bias_initial,"bias_initial",)
        T_B_I_initial = (np.eye(4) if T_B_I_initial is None else _as_pose_matrix(T_B_I_initial))

        C = T_B_I_initial[:3, :3]
        corrected_angular_velocity = (angular_velocity_imu - bias_initial[None, :])

    # Prepare the accumulated LiDAR odometry trajectory.
    if use_lidar_odometry:
        lidar_timestamps = np.asarray(lidar_timestamps,dtype=float,).reshape(-1)
        lidar_odometry_poses = np.asarray([_as_pose_matrix(lidar_pose) for lidar_pose in lidar_odometry_poses])

        if len(lidar_odometry_poses) != len(lidar_timestamps): raise ValueError("lidar_odometry_poses must have the same length as lidar_timestamps")

        if len(lidar_timestamps) < 2: raise ValueError("At least two LiDAR odometry poses are required")

        if np.any(np.diff(lidar_timestamps) <= 0): raise ValueError("lidar_timestamps must be strictly increasing")

        T_B_L_initial = (np.eye(4) if T_B_L_initial is None else _as_pose_matrix(T_B_L_initial))

        T_L_B_initial = np.linalg.inv(T_B_L_initial)

    for pose_index in range(len(initial_poses), len(pose_timestamps)):
        previous_pose = initial_poses[-1]
        next_pose = previous_pose.copy()

        # In IMU mode, initialize the new body orientation by integrating the bias-corrected angular velocity over the trajectory-pose interval.
        if use_imu_gyr:
            imu_time_origin = float(pose_timestamps[pose_index - 1] + tau_I_initial)
            imu_time_target = float(pose_timestamps[pose_index] + tau_I_initial)

            phi_I = _integrate_vector_interval(imu_timestamps, corrected_angular_velocity, imu_time_origin, imu_time_target,)

            delta_rotation_body = (C @ mrob.SO3(phi_I).R() @ C.T)

            next_pose[:3, :3] = (previous_pose[:3, :3] @ delta_rotation_body)

        # Query the accumulated LiDAR trajectory and recover the relative LiDAR motion over the current body-pose interval.
        if use_lidar_odometry:
            lidar_time_origin = float(pose_timestamps[pose_index - 1] + tau_L_initial)
            lidar_time_target = float(pose_timestamps[pose_index] + tau_L_initial)

            lidar_interval_supported = (lidar_timestamps[0] <= lidar_time_origin and lidar_time_target <= lidar_timestamps[-1])

            if lidar_interval_supported:
                T_O_L_origin = _interpolate_pose(lidar_timestamps, lidar_odometry_poses, lidar_time_origin,)
                T_O_L_target = _interpolate_pose(lidar_timestamps, lidar_odometry_poses, lidar_time_target,)

                relative_lidar_pose = (np.linalg.inv(T_O_L_origin) @ T_O_L_target)

                relative_body_pose = (T_B_L_initial @ relative_lidar_pose @ T_L_B_initial)

                if use_imu_gyr:
                    # Keep the IMU-propagated orientation and use only the translation component of the LiDAR relative motion.
                    next_pose[:3, 3] = (previous_pose[:3, 3] + previous_pose[:3, :3] @ relative_body_pose[:3, 3])
                else:
                    # Propagate the complete SE(3) body motion measured by LiDAR, including both rotation and translation.
                    next_pose = (previous_pose @ relative_body_pose)

            elif not use_imu_gyr:
                raise ValueError(f"LiDAR odometry does not cover the shifted interval [{lidar_time_origin}, {lidar_time_target}]")

        initial_poses.append(next_pose)

    return np.asarray(initial_poses)

@njit
def apply_timeshift(data, i_shift, trim_end=False):
    '''
    Shifting data in time to apply time_offset, calculated by TwistnSync
    Positive i_shift belongs to earlier start of recording

    param: data - array of time or data
    param: i_shift - time_offset in terms of indices in times array
    param: trim_end - if True, trim end of data with i_shift < 0, to change arrays len the same

    return: data_sync - input array, starting from i_shift (not generally true)
    '''

    if(i_shift >= 0):
        data_sync = data[i_shift:]         # need to delete tail of data moved to beginning
    else:
        if trim_end:
            data_sync = data[:len(data) - i_shift]
        else:
            data_sync = data
    return data_sync.copy()

def trim_to_min_length(*arrays):
    """
    Trims all input arrays along the first dimension to match the minimal common length.
    
    param: *arrays - Arbitrary number of numpy arrays with different shapes.
    
    return: list of np.ndarray: List of trimmed arrays with the same first-dimension length.
    """
    if not arrays:
        raise ValueError("At least one array must be provided.")
    
    # Find the minimum length along the first dimension
    max_common_len = min(len(arr) for arr in arrays)
    
    # Trim all arrays to this length
    trimmed_arrays = [arr[:max_common_len].copy() for arr in arrays]
    
    return trimmed_arrays
