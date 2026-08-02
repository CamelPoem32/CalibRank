"""Simulation dataset and local observability-Jacobian assembly.

The global Jacobian is partitioned as `J = [J_T  J_C]`: trajectory pose
columns first, then the active calibration columns. When IMU and LiDAR factors
are both included, the calibration block contains 17 parameters:
`T_B_I` (6), `b_g` (3), `tau_I` (1), `T_B_L` (6), and `tau_L` (1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..accelerometer import AccelerometerFactorTerms, linearize_accelerometer_factor
from ..assembly import (
    JacobianBlock,
    JacobianBundle,
    VariableLayout,
    assemble_jacobian_dense,
    assemble_jacobian_sparse,
    make_residual_blocks,
)
from ..finite_difference import central_difference_vector, finite_difference_left_jacobian_se3
from ..jacobians import (
    extrinsic_prior_jacobian_left,
    linearize_gyro_factor,
    linearize_lidar_factor,
)
from ..lie_se3 import se3_inverse
from ..lie_so3 import so3_exp, so3_log
from ..residuals import (
    gyro_increment_from_signal,
    relative_body_motion,
    relative_pose_residual_prediction_first,
    sensor_relative_prediction,
)
from ..types import (
    DEFAULT_PRACTICAL_RANK_POLICY,
    AccelerometerOptions,
    FixedExtrinsic,
    JacobianCheckResult,
    JacobianOptions,
    PracticalRankPolicy,
    normalized_jacobian_options,
    validate_accelerometer_mode,
    validate_fixed_extrinsic,
)
from ..scaling import (
    build_parameter_scaling_dense,
    build_parameter_scaling_sparse,
    scale_jacobian_dense,
    scale_jacobian_sparse,
)
from ..whitening import whiten_residual_and_jacobian_dense
from .sensors import ImuData, LidarOdometryData


SE3_COMPONENT_LABELS = ("roll", "pitch", "yaw", "x", "y", "z")

##################################################
# Trajectory reframing
##################################################
@dataclass
class ReframedTrajectory:
    '''Continuously query a trajectory after a constant body-frame change.
    
    `T_W_B_new(t) = T_W_B_old(t) @ T_old_new`; spatial twists are unchanged by this constant right multiplication.
    
    Attributes:
        base_trajectory: Underlying continuous trajectory.
        old_body_from_new_body: Constant right-multiplied frame transform.
        mode: Descriptive reframed trajectory mode.
    '''

    base_trajectory: object
    old_body_from_new_body: NDArray[np.float64]
    mode: str

    @property
    def start_time(self) -> float:
        '''Return the first valid trajectory time.
        
        Returns:
            float: First valid time in seconds.
        '''
        return float(getattr(self.base_trajectory, "start_time"))

    @property
    def end_time(self) -> float:
        '''Return the last valid trajectory time.
        
        Returns:
            float: Last valid time in seconds.
        '''
        return float(getattr(self.base_trajectory, "end_time"))

    def pose_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return the reframed body pose at one time.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: Reframed transform `T_W_B`, shape `(4, 4)`.
        '''
        return np.asarray(self.base_trajectory.pose_at(time_seconds), dtype=float) @ self.old_body_from_new_body

    def poses_at(self, query_times: ArrayLike) -> NDArray[np.float64]:
        '''Return reframed body poses at multiple times.
        
        Args:
            query_times (ArrayLike): Trajectory query times.
        
        Returns:
            NDArray[np.float64]: Reframed transforms with shape `(N, 4, 4)`.
        '''
        times = np.asarray(query_times, dtype=float).reshape(-1)
        return np.stack([self.pose_at(float(time_seconds)) for time_seconds in times], axis=0) if times.size else np.zeros((0, 4, 4))

    def position_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return the reframed body origin in world coordinates.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World position with shape `(3,)`.
        '''
        return self.pose_at(time_seconds)[:3, 3].copy()

    def euler_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return reframed ZYX Euler angles.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: ZYX angles `[roll, pitch, yaw]`, shape `(3,)`.
        '''
        rotation = self.pose_at(time_seconds)[:3, :3]
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        return np.array([roll, pitch, yaw], dtype=float)

    def yaw_at(self, time_seconds: float) -> float:
        '''Return the reframed yaw angle.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            float: Yaw angle in radians.
        '''
        return float(self.euler_at(time_seconds)[2])

    def sample(self, num: int = 400) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        '''Sample the reframed trajectory for plotting and inspection.
        
        Args:
            num (int): Number of uniformly spaced samples to return.
        
        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: Tuple `(times, positions, eulers)`.
        '''
        sample_times = np.linspace(self.start_time, self.end_time, int(num))
        positions = np.vstack([self.position_at(float(time_seconds)) for time_seconds in sample_times])
        eulers = np.vstack([self.euler_at(float(time_seconds)) for time_seconds in sample_times])
        return sample_times, positions, eulers


def reframe_dataset_to_fixed_extrinsic(
    dataset: "CalibrationSimulationDataset",
    fixed_extrinsic: FixedExtrinsic,
) -> "CalibrationSimulationDataset":
    '''Express a simulation dataset in a requested fixed-extrinsic body frame.
    
    Args:
        dataset ('CalibrationSimulationDataset'): Simulation dataset to inspect or reframe.
        fixed_extrinsic (FixedExtrinsic): Selected fixed-extrinsic body-frame convention.
    
    Returns:
        'CalibrationSimulationDataset': Dataset expressed in the selected body frame.
    
    Raises:
        ValueError: The fixed-extrinsic convention is invalid.
    
    Notes:
        Sensor measurements are not regenerated because the represented world sensor poses remain unchanged.
    '''

    selected = validate_fixed_extrinsic(fixed_extrinsic)
    if selected == "none":
        return dataset
    if selected == "T_B_L":
        old_body_from_new_body = np.asarray(dataset.T_B_L_true, dtype=float)
        new_body_from_old_body = se3_inverse(old_body_from_new_body)
        return replace(
            dataset,
            trajectory=ReframedTrajectory(
                dataset.trajectory,
                old_body_from_new_body,
                mode=f"{getattr(dataset.trajectory, 'mode', 'trajectory')}_B_equals_L",
            ),
            T_B_L_true=np.eye(4),
            T_B_I_true=new_body_from_old_body @ np.asarray(dataset.T_B_I_true, dtype=float),
        )
    old_body_from_new_body = np.asarray(dataset.T_B_I_true, dtype=float)
    new_body_from_old_body = se3_inverse(old_body_from_new_body)
    return replace(
        dataset,
        trajectory=ReframedTrajectory(
            dataset.trajectory,
            old_body_from_new_body,
            mode=f"{getattr(dataset.trajectory, 'mode', 'trajectory')}_B_equals_I",
        ),
        T_B_I_true=np.eye(4),
        T_B_L_true=new_body_from_old_body @ np.asarray(dataset.T_B_L_true, dtype=float),
    )

##################################################
# Dataset compatibility and signal helpers
##################################################
def _labels_for_layout(layout: VariableLayout) -> list[str]:
    '''Build calibration-coordinate labels in layout order.
    
    Args:
        layout (VariableLayout): Global variable layout containing trajectory and calibration blocks.
    
    Returns:
        list[str]: Calibration labels in global column order.
    '''

    labels: list[str] = []
    for block in layout.calibration_blocks:
        if block.name in {"T_B_I", "T_B_L"}:
            labels.extend(f"{block.name}_{component}" for component in SE3_COMPONENT_LABELS)
        elif block.name == "b_g":
            labels.extend(("b_gx", "b_gy", "b_gz"))
        elif block.dimension == 1:
            labels.append(block.name)
        else:
            labels.extend(f"{block.name}_{index}" for index in range(block.dimension))
    return labels


def _array_attribute(instance: object, names: tuple[str, ...]) -> NDArray[np.float64]:
    '''Read the first available finite array attribute.
    
    Args:
        instance (object): Object from which a compatible array attribute is read.
        names (tuple[str, ...]): Candidate attribute names in priority order.
    
    Returns:
        NDArray[np.float64]: First matching finite array.
    
    Raises:
        ValueError: A found attribute contains non-finite values.
        AttributeError: None of the compatibility attributes exists.
    '''

    for name in names:
        if hasattr(instance, name):
            value = np.asarray(getattr(instance, name), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{type(instance).__name__}.{name} contains non-finite values")
            return value
    joined_names = ", ".join(names)
    raise AttributeError(f"{type(instance).__name__} must expose one of: {joined_names}")


def _imu_arrays(imu: ImuData) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Extract validated gyroscope timestamps and samples.
    
    Args:
        imu (ImuData): Simulated IMU data container.
    
    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple `(sensor_timestamps, gyroscope_samples)`.
    
    Raises:
        ValueError: The IMU arrays have invalid shape, length, or timestamp ordering.
    '''

    sensor_timestamps = _array_attribute(imu, ("sensor_timestamps", "timestamps", "times", "t")).reshape(-1)
    gyroscope_samples = _array_attribute(imu, ("gyroscope", "gyro", "angular_velocity", "omega"))
    if gyroscope_samples.ndim != 2 or gyroscope_samples.shape[1] != 3:
        raise ValueError("IMU gyroscope array must have shape (N, 3)")
    if sensor_timestamps.shape[0] != gyroscope_samples.shape[0]:
        raise ValueError("IMU timestamps and gyroscope arrays must have the same length")
    if sensor_timestamps.size < 2 or np.any(np.diff(sensor_timestamps) <= 0.0):
        raise ValueError("IMU timestamps must be strictly increasing")
    return sensor_timestamps, gyroscope_samples


def _accelerometer_arrays(imu: ImuData) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Extract validated accelerometer timestamps and samples.
    
    Args:
        imu (ImuData): Simulated IMU data container.
    
    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple `(sensor_timestamps, accelerometer_samples)`.
    
    Raises:
        ValueError: The accelerometer arrays have invalid shape or length.
    '''

    sensor_timestamps = _array_attribute(imu, ("sensor_timestamps", "timestamps", "times", "t")).reshape(-1)
    accelerometer_samples = _array_attribute(imu, ("accelerometer", "accel", "specific_force", "acceleration"))
    if accelerometer_samples.ndim != 2 or accelerometer_samples.shape[1] != 3:
        raise ValueError("IMU accelerometer array must have shape (N, 3)")
    if sensor_timestamps.shape[0] != accelerometer_samples.shape[0]:
        raise ValueError("IMU timestamps and accelerometer arrays must have the same length")
    return sensor_timestamps, accelerometer_samples


def _interpolate_vector_signal(
    sample_times: NDArray[np.float64],
    samples: NDArray[np.float64],
    query_time: float,
) -> NDArray[np.float64]:
    '''Linearly interpolate a vector signal at one timestamp.
    
    Args:
        sample_times (NDArray[np.float64]): Strictly increasing sample timestamps.
        samples (NDArray[np.float64]): Vector-valued samples aligned with `sample_times`.
        query_time (float): Timestamp at which the signal is evaluated.
    
    Returns:
        NDArray[np.float64]: Interpolated vector with one value per sample axis.
    '''

    return np.array([np.interp(query_time, sample_times, samples[:, axis]) for axis in range(samples.shape[1])], dtype=float)

##################################################
# Accelerometer selection structures
##################################################
@dataclass(frozen=True)
class AccelerometerSelection:
    '''Store accepted accelerometer samples and gate diagnostics.
    
    Attributes:
        indices: Accepted accelerometer sample indices.
        support_times: Trajectory support times keyed by sample index.
        candidate_count: Number of samples considered before gating.
        rejection_reasons: Counts grouped by rejection reason.
        gate_mask: Boolean acceptance mask over candidates.
    '''

    indices: NDArray[np.int64]
    support_times: dict[int, tuple[float, ...]]
    candidate_count: int
    rejection_reasons: dict[str, int]
    gate_mask: NDArray[np.bool_]


def _time_key(value: float) -> float:
    '''Create a stable rounded timestamp key.
    
    Args:
        value (float): Value to validate, differentiate, or convert.
    
    Returns:
        float: Rounded timestamp used for dictionary lookup.
    '''

    return round(float(value), 12)


def _finite_difference_scalar_residual(
    residual_function: Callable[[float], ArrayLike],
    value: float,
    *,
    epsilon: float,
) -> NDArray[np.float64]:
    '''Differentiate a vector residual with respect to one scalar.
    
    Args:
        residual_function (Callable[[float], ArrayLike]): Vector residual evaluated as a function of one scalar.
        value (float): Value to validate, differentiate, or convert.
        epsilon (float): Positive central finite-difference step.
    
    Returns:
        NDArray[np.float64]: Jacobian with shape `(residual_dimension, 1)`.
    '''

    return central_difference_vector(
        lambda scalar_array: np.asarray(residual_function(float(scalar_array[0])), dtype=float),
        np.array([float(value)], dtype=float),
        epsilon=epsilon,
    )


def _gyro_factor_residual(
    start_body_pose: NDArray[np.float64],
    end_body_pose: NDArray[np.float64],
    body_from_imu: NDArray[np.float64],
    gyro_bias: ArrayLike,
    imu_time_offset: float,
    true_start_time: float,
    true_end_time: float,
    imu_sensor_timestamps: NDArray[np.float64],
    gyroscope_samples: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Evaluate the gyroscope propagation residual for two body poses.
    
    Args:
        start_body_pose (NDArray[np.float64]): Body pose at the start of the factor interval.
        end_body_pose (NDArray[np.float64]): Body pose at the end of the factor interval.
        body_from_imu (NDArray[np.float64]): IMU extrinsic transform `T_B_I`.
        gyro_bias (ArrayLike): Constant gyroscope bias vector.
        imu_time_offset (float): IMU temporal offset in seconds.
        true_start_time (float): Factor start time on the trajectory clock.
        true_end_time (float): Factor end time on the trajectory clock.
        imu_sensor_timestamps (NDArray[np.float64]): IMU sample timestamps on the sensor clock.
        gyroscope_samples (NDArray[np.float64]): Gyroscope samples with shape `(N, 3)`.
    
    Returns:
        NDArray[np.float64]: SO(3) residual vector with shape `(3,)`.
    '''

    # IMU samples live on the sensor clock while pose nodes live in trajectory time.
    # With `true_time = sensor_time + tau_I`, integration uses `-tau_I` in the
    # existing gyro helper, whose convention is `sample_time = pose_time + tau`.
    imu_rotation_vector = gyro_increment_from_signal(
        imu_sensor_timestamps,
        gyroscope_samples,
        true_start_time,
        true_end_time,
        -float(imu_time_offset),
        gyro_bias,
        interpolation="linear",
    )

    # Rotate the IMU-frame increment through the IMU-to-body extrinsic rotation.
    imu_rotation_increment = so3_exp(imu_rotation_vector)
    body_from_imu_rotation = np.asarray(body_from_imu[:3, :3], dtype=float)
    body_rotation_increment = body_from_imu_rotation @ imu_rotation_increment @ body_from_imu_rotation.T

    start_body_rotation = np.asarray(start_body_pose[:3, :3], dtype=float)
    end_body_rotation = np.asarray(end_body_pose[:3, :3], dtype=float)
    residual_rotation = start_body_rotation @ body_rotation_increment @ end_body_rotation.T
    return so3_log(residual_rotation)

##################################################
# Window Jacobian assembly
##################################################
@dataclass
class CalibrationSimulationDataset:
    '''Store a simulated trajectory, sensor streams, and true calibration.
    
    Attributes:
        trajectory: Continuously queryable body trajectory.
        imu: Simulated IMU stream.
        lidar: Simulated LiDAR odometry stream.
        T_B_L_true: True body-from-LiDAR transform.
        T_B_I_true: True body-from-IMU transform.
        tau_I_true: True IMU temporal offset.
        tau_L_true: True LiDAR temporal offset.
        gyro_bias_true: True gyroscope bias.
    '''

    trajectory: object
    imu: ImuData
    lidar: LidarOdometryData
    T_B_L_true: NDArray[np.float64]
    T_B_I_true: NDArray[np.float64]
    tau_I_true: float
    tau_L_true: float
    gyro_bias_true: NDArray[np.float64]

    @property
    def start_time(self) -> float:
        '''Return the first valid dataset time.
        
        Returns:
            float: First valid trajectory time.
        '''
        return float(getattr(self.trajectory, "start_time"))

    @property
    def end_time(self) -> float:
        '''Return the last valid dataset time.
        
        Returns:
            float: Last valid trajectory time.
        '''
        return float(getattr(self.trajectory, "end_time"))

    def window_jacobians(
        self,
        start: float,
        end: float,
        pose_provider: object,
        *,
        include_imu: bool = True,
        include_lidar: bool = True,
        include_priors: bool = False,
        include_smoothness: bool = False,
        use_sparse: bool = True,
        parameter_scaling: object | None = None,
        pose_node_rate_hz: float = 5.0,
        imu_rotation_noise_std: float = 0.01,
        finite_difference_epsilon: float = 1e-7,
        jacobian_options: JacobianOptions | None = None,
        fixed_extrinsic: FixedExtrinsic = "T_B_L",
        practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
        accelerometer_options: AccelerometerOptions | None = None,
    ) -> tuple[JacobianBundle, list[NDArray[np.float64]], dict[str, int]]:
        '''Assemble a whitened local observability Jacobian for one time window.
        
        Args:
            start (float): Requested window start time.
            end (float): Requested window end time.
            pose_provider (object): Continuous pose and twist query interface.
            include_imu (bool): Whether gyroscope factors are included.
            include_lidar (bool): Whether LiDAR factors are included.
            include_priors (bool): Whether active calibration priors are appended.
            include_smoothness (bool): Whether smoothness was requested for the window.
            use_sparse (bool): Whether the returned global Jacobian uses sparse storage.
            parameter_scaling (object | None): Optional physical parameter scaling specification.
            pose_node_rate_hz (float): Pose-node rate used by gyro-only windows.
            imu_rotation_noise_std (float): Gyroscope residual standard deviation.
            finite_difference_epsilon (float): Positive finite-difference perturbation size.
            jacobian_options (JacobianOptions | None): Analytic or numerical Jacobian configuration.
            fixed_extrinsic (FixedExtrinsic): Selected fixed-extrinsic body-frame convention.
            practical_rank_policy (PracticalRankPolicy): Practical-rank thresholds saved in bundle metadata.
            accelerometer_options (AccelerometerOptions | None): Optional accelerometer factor configuration.
        
        Returns:
            tuple[JacobianBundle, list[NDArray[np.float64]], dict[str, int]]: Tuple `(bundle, body_motions, counts)`.
        
        Raises:
            ValueError: The window, sensor selection, provider output, or numerical configuration is invalid.
        
        Notes:
            Trajectory columns precede active calibration columns in `J = [J_T  J_C]`.
            Whitening is applied to complete local factor blocks before global insertion.
        '''

        window_start_time = max(float(start), self.start_time)
        window_end_time = min(float(end), self.end_time)
        valid_window_bounds = np.isfinite([window_start_time, window_end_time]).all()
        if not valid_window_bounds or window_end_time <= window_start_time:
            raise ValueError("window must satisfy start < end inside the trajectory range")
        if not include_imu and not include_lidar:
            raise ValueError("at least one of include_imu/include_lidar must be True")
        validated_jacobian_options = normalized_jacobian_options(jacobian_options)
        selected_fixed_extrinsic = validate_fixed_extrinsic(fixed_extrinsic)
        accel_options = AccelerometerOptions() if accelerometer_options is None else accelerometer_options
        accelerometer_mode = validate_accelerometer_mode(accel_options.mode)
        finite_difference_epsilon = validated_jacobian_options.finite_difference_epsilon
        if finite_difference_epsilon <= 0.0:
            raise ValueError("finite_difference_epsilon must be positive")
        if include_smoothness:
            warnings.warn(
                "Spatial smoothness requires multiple time-indexed extrinsic nodes. "
                "This dataset currently has one constant T_B_I and one constant T_B_L, "
                "so no smoothness factor is added. A one-node smoothness term would "
                "be a prior and would misrepresent measurement observability.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Select LiDAR factors in true trajectory time, but keep sensor-clock
        # endpoints for the time-offset derivative column.
        lidar_indices = np.empty(0, dtype=int)
        lidar_sensor_start_times = np.asarray(self.lidar.relative_start_times, dtype=float)
        lidar_sensor_end_times = np.asarray(self.lidar.relative_end_times, dtype=float)
        if include_lidar:
            lidar_true_start_times = lidar_sensor_start_times + self.tau_L_true
            lidar_true_end_times = lidar_sensor_end_times + self.tau_L_true
            lidar_window_mask = (lidar_true_start_times >= window_start_time) & (
                lidar_true_end_times <= window_end_time
            )
            lidar_time_offset_margin_mask = (lidar_true_start_times - finite_difference_epsilon >= self.start_time) & (
                lidar_true_end_times + finite_difference_epsilon <= self.end_time
            )
            lidar_window_mask = lidar_window_mask & lidar_time_offset_margin_mask
            lidar_indices = np.flatnonzero(lidar_window_mask)

        accel_sensor_timestamps = np.zeros(0, dtype=float)
        accel_measurements = np.zeros((0, 3), dtype=float)
        accel_selection = AccelerometerSelection(np.zeros(0, dtype=int), {}, 0, {}, np.zeros(0, dtype=bool))
        if accelerometer_mode != "disabled":
            accel_sensor_timestamps, accel_measurements = _accelerometer_arrays(self.imu)
            gyro_times_for_gate, gyro_samples_for_gate = _imu_arrays(self.imu)
            accel_selection = self._select_accelerometer_factors(
                accel_sensor_timestamps,
                accel_measurements,
                gyro_times_for_gate,
                gyro_samples_for_gate,
                window_start_time,
                window_end_time,
                accel_options,
                finite_difference_epsilon,
            )

        accel_support_times = np.asarray(
            [time for index in accel_selection.indices for time in accel_selection.support_times[int(index)]],
            dtype=float,
        )

        # LiDAR and accepted accelerometer timestamps define shared pose nodes
        # when present; otherwise gyro-only windows use a regular pose grid.
        if lidar_indices.size or accel_support_times.size:
            pose_time_parts = []
            if lidar_indices.size:
                pose_time_parts.extend(
                    [
                        lidar_sensor_start_times[lidar_indices] + self.tau_L_true,
                        lidar_sensor_end_times[lidar_indices] + self.tau_L_true,
                    ]
                )
            if accel_support_times.size:
                pose_time_parts.append(accel_support_times)
            pose_times = np.unique(np.concatenate(pose_time_parts))
        elif include_imu:
            pose_node_rate = float(pose_node_rate_hz)
            if not np.isfinite(pose_node_rate) or pose_node_rate <= 0.0:
                raise ValueError("pose_node_rate_hz must be finite and positive")
            pose_node_step = 1.0 / pose_node_rate
            pose_times = np.arange(window_start_time, window_end_time + 0.5 * pose_node_step, pose_node_step)
            pose_times = pose_times[pose_times <= window_end_time + 1e-12]
        else:
            pose_times = np.zeros(0, dtype=float)

        if pose_times.size < 2:
            return self._empty_bundle(use_sparse), [], {"imu": 0, "lidar": 0}

        # Decide which sensor families actually contribute factors in this window.
        imu_sensor_timestamps = np.zeros(0, dtype=float)
        gyroscope_samples = np.zeros((0, 3), dtype=float)
        imu_interval_indices = np.zeros(0, dtype=int)
        if include_imu:
            imu_sensor_timestamps, gyroscope_samples = _imu_arrays(self.imu)
            imu_interval_indices = self._usable_imu_interval_indices(
                pose_times,
                imu_sensor_timestamps,
                finite_difference_epsilon,
            )
        lidar_is_active = bool(include_lidar and lidar_indices.size > 0)
        imu_is_active = bool(include_imu and imu_interval_indices.size > 0)
        accelerometer_is_active = bool(accelerometer_mode != "disabled" and accel_selection.indices.size > 0)
        if not lidar_is_active and not imu_is_active and not accelerometer_is_active:
            return self._empty_bundle(use_sparse), [], {"imu": 0, "lidar": 0}

        # Build the global variable layout: trajectory first, then only the
        # calibration variables touched by the selected sensor families.
        pose_index_by_time = {_time_key(pose_time): pose_index for pose_index, pose_time in enumerate(pose_times)}
        pose_variable_names = [f"T_W_B_{pose_index}" for pose_index in range(pose_times.size)]
        variable_specs = [(pose_name, 6, "trajectory") for pose_name in pose_variable_names]
        if imu_is_active or accelerometer_is_active:
            variable_specs.append(("T_B_I", 6, "calibration"))
            if imu_is_active:
                variable_specs.append(("b_g", 3, "calibration"))
            variable_specs.append(("tau_I", 1, "calibration"))
        if lidar_is_active:
            if selected_fixed_extrinsic != "T_B_L":
                variable_specs.append(("T_B_L", 6, "calibration"))
            variable_specs.append(("tau_L", 1, "calibration"))
        layout = VariableLayout.from_specs(variable_specs)

        residual_specs: list[tuple[str, int, NDArray[np.float64], str]] = []
        residual_values: dict[str, NDArray[np.float64]] = {}
        jacobian_blocks: list[JacobianBlock] = []
        provider_body_poses = np.asarray(pose_provider.poses_at(pose_times), dtype=float)
        if provider_body_poses.shape != (pose_times.size, 4, 4):
            raise ValueError("pose_provider.poses_at must return shape (N, 4, 4)")
        jacobian_check_results: list[JacobianCheckResult] = []

        lidar_count = self._append_lidar_factors(
            lidar_indices,
            lidar_sensor_start_times,
            lidar_sensor_end_times,
            pose_index_by_time,
            pose_variable_names,
            provider_body_poses,
            pose_provider,
            residual_specs,
            residual_values,
            jacobian_blocks,
            finite_difference_epsilon,
            validated_jacobian_options,
            jacobian_check_results,
            selected_fixed_extrinsic,
        ) if lidar_is_active else 0

        imu_count = self._append_imu_factors(
            pose_times,
            imu_interval_indices,
            imu_sensor_timestamps,
            gyroscope_samples,
            pose_variable_names,
            provider_body_poses,
            residual_specs,
            residual_values,
            jacobian_blocks,
            imu_rotation_noise_std,
            finite_difference_epsilon,
            validated_jacobian_options,
            jacobian_check_results,
            selected_fixed_extrinsic,
        ) if imu_is_active else 0

        accelerometer_check_results: list[JacobianCheckResult] = []
        accelerometer_terms: list[AccelerometerFactorTerms] = []
        accelerometer_count = self._append_accelerometer_factors(
            accel_selection,
            accel_sensor_timestamps,
            accel_measurements,
            pose_index_by_time,
            pose_variable_names,
            provider_body_poses,
            pose_provider,
            residual_specs,
            residual_values,
            jacobian_blocks,
            accel_options,
            validated_jacobian_options,
            accelerometer_check_results,
            accelerometer_terms,
        ) if accelerometer_is_active else 0

        if not residual_specs:
            return self._empty_bundle(use_sparse), [], {"imu": 0, "lidar": 0}

        if include_priors:
            self._append_priors(layout, residual_specs, residual_values, jacobian_blocks)

        residual_blocks = make_residual_blocks(residual_specs)
        # Return motions for the pose intervals that carry measurement support.
        # LiDAR windows use their own consecutive odometry intervals; IMU-only
        # windows may drop boundary intervals that lack finite-difference margin.
        motion_interval_indices = (
            np.arange(pose_times.size - 1, dtype=int)
            if lidar_is_active
            else imu_interval_indices
        )
        body_motions = [
            relative_body_motion(provider_body_poses[index], provider_body_poses[index + 1])
            for index in motion_interval_indices
        ]
        metadata = self._metadata(
            layout,
            pose_times,
            window_start_time,
            window_end_time,
            imu_count,
            lidar_count,
            validated_jacobian_options,
            jacobian_check_results,
            selected_fixed_extrinsic,
            practical_rank_policy,
            accel_options,
            accelerometer_count,
            accel_selection.candidate_count,
            accel_selection.rejection_reasons,
            accel_selection.gate_mask,
            accelerometer_check_results,
            accelerometer_terms,
        )

        # Assemble a dense or sparse global Jacobian, then optionally apply
        # physical parameter scaling without changing the active block layout.
        if use_sparse:
            bundle = assemble_jacobian_sparse(
                layout,
                residual_blocks,
                jacobian_blocks,
                residual_values,
                metadata=metadata,
            )
        else:
            bundle = assemble_jacobian_dense(
                layout,
                residual_blocks,
                jacobian_blocks,
                residual_values,
                metadata=metadata,
            )
        if parameter_scaling is not None:
            bundle = self._scaled_bundle(bundle, layout, parameter_scaling, use_sparse)
        counts = {"imu": imu_count, "lidar": lidar_count}
        if accelerometer_mode != "disabled":
            counts.update(
                {
                    "gyro_factor_count": imu_count,
                    "lidar_factor_count": lidar_count,
                    "accelerometer_factor_count": accelerometer_count,
                    "accelerometer_candidate_count": accel_selection.candidate_count,
                    "accelerometer_rejected_count": int(sum(accel_selection.rejection_reasons.values())),
                }
            )
        return bundle, body_motions, counts

    def _usable_imu_interval_indices(
        self,
        pose_times: NDArray[np.float64],
        imu_sensor_timestamps: NDArray[np.float64],
        finite_difference_epsilon: float,
    ) -> NDArray[np.int64]:
        '''Find pose intervals whose shifted IMU limits are fully sampled.
        
        Args:
            pose_times (NDArray[np.float64]): Trajectory times associated with pose nodes.
            imu_sensor_timestamps (NDArray[np.float64]): IMU sample timestamps on the sensor clock.
            finite_difference_epsilon (float): Positive finite-difference perturbation size.
        
        Returns:
            NDArray[np.int64]: Accepted interval indices.
        '''

        usable_indices = []
        for interval_index in range(pose_times.size - 1):
            true_start_time = float(pose_times[interval_index])
            true_end_time = float(pose_times[interval_index + 1])
            imu_sensor_start_time = true_start_time - self.tau_I_true
            imu_sensor_end_time = true_end_time - self.tau_I_true
            integration_interval_is_sampled = (
                imu_sensor_start_time - finite_difference_epsilon >= imu_sensor_timestamps[0]
                and imu_sensor_end_time + finite_difference_epsilon <= imu_sensor_timestamps[-1]
            )
            if integration_interval_is_sampled:
                usable_indices.append(interval_index)
        return np.asarray(usable_indices, dtype=int)

    def _select_accelerometer_factors(
        self,
        accel_sensor_timestamps: NDArray[np.float64],
        accel_measurements: NDArray[np.float64],
        gyro_sensor_timestamps: NDArray[np.float64],
        gyro_measurements: NDArray[np.float64],
        window_start_time: float,
        window_end_time: float,
        options: AccelerometerOptions,
        finite_difference_epsilon: float,
    ) -> AccelerometerSelection:
        '''Select accelerometer factors and record rejection reasons.
        
        Args:
            accel_sensor_timestamps (NDArray[np.float64]): Accelerometer timestamps on the IMU sensor clock.
            accel_measurements (NDArray[np.float64]): Accelerometer specific-force measurements.
            gyro_sensor_timestamps (NDArray[np.float64]): Gyroscope timestamps used by the low-dynamic gate.
            gyro_measurements (NDArray[np.float64]): Gyroscope measurements used by the low-dynamic gate.
            window_start_time (float): Clipped window start time.
            window_end_time (float): Clipped window end time.
            options (AccelerometerOptions): Validated factor or Jacobian configuration.
            finite_difference_epsilon (float): Positive finite-difference perturbation size.
        
        Returns:
            AccelerometerSelection: Accepted samples and rejection diagnostics.
        '''

        mode = validate_accelerometer_mode(options.mode)
        if mode == "disabled":
            return AccelerometerSelection(np.zeros(0, dtype=int), {}, 0, {}, np.zeros(0, dtype=bool))
        gravity_world = np.asarray(getattr(self.imu, "gravity_world", np.array([0.0, 0.0, -9.81])), dtype=float)
        gravity_norm = float(np.linalg.norm(gravity_world))
        h = float(options.support_half_width_seconds)
        stride = max(int(options.sample_stride), 1)
        candidate_indices = np.arange(0, accel_sensor_timestamps.size, stride, dtype=int)
        if options.factor_rate_hz is not None and candidate_indices.size:
            period = 1.0 / float(options.factor_rate_hz)
            kept = []
            next_sensor_time = -np.inf
            for index in candidate_indices:
                sensor_time = float(accel_sensor_timestamps[index])
                if sensor_time + 1e-12 >= next_sensor_time:
                    kept.append(index)
                    next_sensor_time = sensor_time + period
            candidate_indices = np.asarray(kept, dtype=int)

        accepted: list[int] = []
        support_times: dict[int, tuple[float, ...]] = {}
        gate_mask = np.zeros(candidate_indices.size, dtype=bool)
        reasons = {"outside_window": 0, "unavailable_support": 0, "gravity_norm_gate_failed": 0, "gyro_norm_gate_failed": 0}
        for local_index, sample_index in enumerate(candidate_indices):
            sensor_time = float(accel_sensor_timestamps[sample_index])
            true_time = sensor_time + self.tau_I_true
            if true_time < window_start_time - 1e-12 or true_time > window_end_time + 1e-12:
                reasons["outside_window"] += 1
                continue
            times = (true_time,) if mode == "simple" else (true_time - h, true_time, true_time + h)
            if min(times) - finite_difference_epsilon < self.start_time or max(times) + finite_difference_epsilon > self.end_time:
                reasons["unavailable_support"] += 1
                continue
            if mode == "simple" and options.require_low_dynamic_gate:
                force_norm = float(np.linalg.norm(accel_measurements[sample_index]))
                gyro_value = _interpolate_vector_signal(gyro_sensor_timestamps, gyro_measurements, sensor_time)
                gyro_norm = float(np.linalg.norm(gyro_value))
                if abs(force_norm - gravity_norm) > float(options.gravity_norm_tolerance_m_s2):
                    reasons["gravity_norm_gate_failed"] += 1
                    continue
                if gyro_norm > float(options.low_dynamic_gyro_threshold_rad_s):
                    reasons["gyro_norm_gate_failed"] += 1
                    continue
            accepted.append(int(sample_index))
            support_times[int(sample_index)] = tuple(float(time) for time in times)
            gate_mask[local_index] = True
        return AccelerometerSelection(
            np.asarray(accepted, dtype=int),
            support_times,
            int(candidate_indices.size),
            {key: int(value) for key, value in reasons.items() if value},
            gate_mask,
        )

    def _append_accelerometer_factors(
        self,
        selection: AccelerometerSelection,
        accel_sensor_timestamps: NDArray[np.float64],
        accel_measurements: NDArray[np.float64],
        pose_index_by_time: dict[float, int],
        pose_variable_names: list[str],
        provider_body_poses: NDArray[np.float64],
        pose_provider: object,
        residual_specs: list[tuple[str, int, NDArray[np.float64], str]],
        residual_values: dict[str, NDArray[np.float64]],
        jacobian_blocks: list[JacobianBlock],
        accelerometer_options: AccelerometerOptions,
        jacobian_options: JacobianOptions,
        accelerometer_check_results: list[JacobianCheckResult],
        accelerometer_terms: list[AccelerometerFactorTerms],
    ) -> int:
        '''Append accepted accelerometer factors to global assembly buffers.
        
        Args:
            selection (AccelerometerSelection): Accepted accelerometer samples and support times.
            accel_sensor_timestamps (NDArray[np.float64]): Accelerometer timestamps on the IMU sensor clock.
            accel_measurements (NDArray[np.float64]): Accelerometer specific-force measurements.
            pose_index_by_time (dict[float, int]): Mapping from rounded timestamps to local pose indices.
            pose_variable_names (list[str]): Global variable names of the local pose nodes.
            provider_body_poses (NDArray[np.float64]): Body poses queried from the trajectory provider.
            pose_provider (object): Continuous pose and twist query interface.
            residual_specs (list[tuple[str, int, NDArray[np.float64], str]]): Mutable residual-block specifications for global assembly.
            residual_values (dict[str, NDArray[np.float64]]): Mutable residual values keyed by residual name.
            jacobian_blocks (list[JacobianBlock]): Mutable local Jacobian blocks for global assembly.
            accelerometer_options (AccelerometerOptions): Optional accelerometer factor configuration.
            jacobian_options (JacobianOptions): Analytic or numerical Jacobian configuration.
            accelerometer_check_results (list[JacobianCheckResult]): Mutable analytic-versus-numerical check results.
            accelerometer_terms (list[AccelerometerFactorTerms]): Mutable saved accelerometer intermediate terms.
        
        Returns:
            int: Number of appended accelerometer factors.
        '''

        mode = validate_accelerometer_mode(accelerometer_options.mode)
        if mode == "disabled":
            return 0
        accel_covariance = accelerometer_options.covariance(self.imu.accel_covariance)
        gravity_world = np.asarray(getattr(self.imu, "gravity_world", np.array([0.0, 0.0, -9.81])), dtype=float)
        count = 0
        for sample_index in selection.indices:
            sample_index_int = int(sample_index)
            sensor_time = float(accel_sensor_timestamps[sample_index_int])
            support_times = selection.support_times[sample_index_int]
            pose_indices = [pose_index_by_time[_time_key(time)] for time in support_times]
            poses = tuple(provider_body_poses[index] for index in pose_indices)
            _, _, spatial_twists = pose_provider.poses_and_twists_at(np.asarray(support_times, dtype=float))
            residual_name = f"accel_{mode}_{count}"
            linearization = linearize_accelerometer_factor(
                mode,
                poses,
                self.T_B_I_true,
                accel_measurements[sample_index_int],
                gravity_world,
                accelerometer_options.support_half_width_seconds,
                tuple(spatial_twists),
                pose_provider=pose_provider,
                sensor_time=sensor_time,
                tau_I=self.tau_I_true,
                jacobian_options=jacobian_options,
                factor_name=residual_name,
            )
            accelerometer_check_results.extend(linearization.check_results)
            if accelerometer_options.save_factor_terms and linearization.terms is not None:
                accelerometer_terms.append(linearization.terms)

            local_blocks = [block.matrix for block in linearization.pose_blocks] + [linearization.H_T_B_I, linearization.H_tau_I]
            H_local = np.hstack(local_blocks)
            whitened_residual, whitened_local = whiten_residual_and_jacobian_dense(
                linearization.residual,
                H_local,
                accel_covariance,
            )
            residual_specs.append((residual_name, 3, np.eye(3), "measurement"))
            residual_values[residual_name] = whitened_residual
            column_start = 0
            for pose_block, pose_index in zip(linearization.pose_blocks, pose_indices):
                column_stop = column_start + 6
                jacobian_blocks.append(JacobianBlock(residual_name, pose_variable_names[pose_index], whitened_local[:, column_start:column_stop]))
                column_start = column_stop
            jacobian_blocks.append(JacobianBlock(residual_name, "T_B_I", whitened_local[:, column_start:column_start + 6]))
            column_start += 6
            jacobian_blocks.append(JacobianBlock(residual_name, "tau_I", whitened_local[:, column_start:column_start + 1]))
            count += 1
        return count

    def _append_lidar_factors(
        self,
        lidar_indices: NDArray[np.int64],
        lidar_sensor_start_times: NDArray[np.float64],
        lidar_sensor_end_times: NDArray[np.float64],
        pose_index_by_time: dict[float, int],
        pose_variable_names: list[str],
        provider_body_poses: NDArray[np.float64],
        pose_provider: object,
        residual_specs: list[tuple[str, int, NDArray[np.float64], str]],
        residual_values: dict[str, NDArray[np.float64]],
        jacobian_blocks: list[JacobianBlock],
        finite_difference_epsilon: float,
        jacobian_options: JacobianOptions,
        jacobian_check_results: list[JacobianCheckResult],
        fixed_extrinsic: FixedExtrinsic,
    ) -> int:
        '''Append LiDAR relative-pose factors to global assembly buffers.
        
        Args:
            lidar_indices (NDArray[np.int64]): Indices of LiDAR measurements accepted in the window.
            lidar_sensor_start_times (NDArray[np.float64]): LiDAR interval start timestamps on the sensor clock.
            lidar_sensor_end_times (NDArray[np.float64]): LiDAR interval end timestamps on the sensor clock.
            pose_index_by_time (dict[float, int]): Mapping from rounded timestamps to local pose indices.
            pose_variable_names (list[str]): Global variable names of the local pose nodes.
            provider_body_poses (NDArray[np.float64]): Body poses queried from the trajectory provider.
            pose_provider (object): Continuous pose and twist query interface.
            residual_specs (list[tuple[str, int, NDArray[np.float64], str]]): Mutable residual-block specifications for global assembly.
            residual_values (dict[str, NDArray[np.float64]]): Mutable residual values keyed by residual name.
            jacobian_blocks (list[JacobianBlock]): Mutable local Jacobian blocks for global assembly.
            finite_difference_epsilon (float): Positive finite-difference perturbation size.
            jacobian_options (JacobianOptions): Analytic or numerical Jacobian configuration.
            jacobian_check_results (list[JacobianCheckResult]): Mutable factor Jacobian check results.
            fixed_extrinsic (FixedExtrinsic): Selected fixed-extrinsic body-frame convention.
        
        Returns:
            int: Number of appended LiDAR factors.
        '''

        lidar_count = 0
        for local_measurement_number, measurement_index in enumerate(lidar_indices):
            sensor_start_time = float(lidar_sensor_start_times[measurement_index])
            sensor_end_time = float(lidar_sensor_end_times[measurement_index])
            true_start_time = sensor_start_time + self.tau_L_true
            true_end_time = sensor_end_time + self.tau_L_true
            start_pose_index = pose_index_by_time[_time_key(true_start_time)]
            end_pose_index = pose_index_by_time[_time_key(true_end_time)]
            start_body_pose = provider_body_poses[start_pose_index]
            end_body_pose = provider_body_poses[end_pose_index]
            lidar_measurement = np.asarray(self.lidar.measurements[measurement_index], dtype=float)
            lidar_covariance = np.asarray(self.lidar.covariances[measurement_index], dtype=float)
            residual_name = f"lidar_{local_measurement_number}"

            # Query trajectory twists at the same true endpoints used by the
            # LiDAR residual. These twists feed the analytic tau_L block.
            query_times = np.array([true_start_time, true_end_time], dtype=float)
            _, _, spatial_twists = pose_provider.poses_and_twists_at(query_times)

            lidar_linearization = linearize_lidar_factor(
                start_body_pose,
                end_body_pose,
                np.eye(4) if fixed_extrinsic == "T_B_L" else self.T_B_L_true,
                lidar_measurement,
                spatial_twists[0],
                spatial_twists[1],
                pose_provider=pose_provider,
                sensor_start_time=sensor_start_time,
                sensor_end_time=sensor_end_time,
                lidar_time_offset=self.tau_L_true,
                jacobian_options=jacobian_options,
            )
            jacobian_check_results.extend(lidar_linearization.check_results)

            # H_local_L: (6, 19) = [H_T0(6), H_T1(6), H_T_B_L(6), H_tau_L(1)].
            # Whitening is applied once to preserve correlations between the
            # local blocks before splitting the whitened matrix back apart.
            H_local = np.hstack(
                [
                    lidar_linearization.H_start_pose,
                    lidar_linearization.H_end_pose,
                    lidar_linearization.H_T_B_L,
                    lidar_linearization.H_tau_L,
                ]
            )
            whitened_residual, whitened_local = whiten_residual_and_jacobian_dense(
                lidar_linearization.residual,
                H_local,
                lidar_covariance,
            )
            start_pose_slice = slice(0, 6)
            end_pose_slice = slice(6, 12)
            extrinsic_slice = slice(12, 18)
            time_offset_slice = slice(18, 19)

            residual_specs.append((residual_name, 6, np.eye(6), "measurement"))
            residual_values[residual_name] = whitened_residual
            jacobian_blocks.extend(
                [
                    JacobianBlock(residual_name, pose_variable_names[start_pose_index], whitened_local[:, start_pose_slice]),
                    JacobianBlock(residual_name, pose_variable_names[end_pose_index], whitened_local[:, end_pose_slice]),
                    JacobianBlock(residual_name, "tau_L", whitened_local[:, time_offset_slice]),
                ]
            )
            if fixed_extrinsic != "T_B_L":
                jacobian_blocks.append(JacobianBlock(residual_name, "T_B_L", whitened_local[:, extrinsic_slice]))
            lidar_count += 1
        return lidar_count

    def _append_imu_factors(
        self,
        pose_times: NDArray[np.float64],
        imu_interval_indices: NDArray[np.int64],
        imu_sensor_timestamps: NDArray[np.float64],
        gyroscope_samples: NDArray[np.float64],
        pose_variable_names: list[str],
        provider_body_poses: NDArray[np.float64],
        residual_specs: list[tuple[str, int, NDArray[np.float64], str]],
        residual_values: dict[str, NDArray[np.float64]],
        jacobian_blocks: list[JacobianBlock],
        imu_rotation_noise_std: float,
        finite_difference_epsilon: float,
        jacobian_options: JacobianOptions,
        jacobian_check_results: list[JacobianCheckResult],
        fixed_extrinsic: FixedExtrinsic,
    ) -> int:
        '''Append gyroscope propagation factors to global assembly buffers.
        
        Args:
            pose_times (NDArray[np.float64]): Trajectory times associated with pose nodes.
            imu_interval_indices (NDArray[np.int64]): Pose-interval indices supported by the IMU samples.
            imu_sensor_timestamps (NDArray[np.float64]): IMU sample timestamps on the sensor clock.
            gyroscope_samples (NDArray[np.float64]): Gyroscope samples with shape `(N, 3)`.
            pose_variable_names (list[str]): Global variable names of the local pose nodes.
            provider_body_poses (NDArray[np.float64]): Body poses queried from the trajectory provider.
            residual_specs (list[tuple[str, int, NDArray[np.float64], str]]): Mutable residual-block specifications for global assembly.
            residual_values (dict[str, NDArray[np.float64]]): Mutable residual values keyed by residual name.
            jacobian_blocks (list[JacobianBlock]): Mutable local Jacobian blocks for global assembly.
            imu_rotation_noise_std (float): Gyroscope residual standard deviation.
            finite_difference_epsilon (float): Positive finite-difference perturbation size.
            jacobian_options (JacobianOptions): Analytic or numerical Jacobian configuration.
            jacobian_check_results (list[JacobianCheckResult]): Mutable factor Jacobian check results.
            fixed_extrinsic (FixedExtrinsic): Selected fixed-extrinsic body-frame convention.
        
        Returns:
            int: Number of appended gyroscope factors.
        '''

        imu_residual_covariance = (float(imu_rotation_noise_std) ** 2) * np.eye(3)
        imu_count = 0
        for interval_index in imu_interval_indices:
            true_start_time = float(pose_times[interval_index])
            true_end_time = float(pose_times[interval_index + 1])
            start_body_pose = provider_body_poses[interval_index]
            end_body_pose = provider_body_poses[interval_index + 1]
            residual_name = f"imu_{imu_count}"

            imu_linearization = linearize_gyro_factor(
                start_body_pose,
                end_body_pose,
                self.T_B_I_true,
                self.gyro_bias_true,
                self.tau_I_true,
                true_start_time,
                true_end_time,
                imu_sensor_timestamps,
                gyroscope_samples,
                interpolation="linear",
                time_offset_sign=-1.0,
                jacobian_options=jacobian_options,
            )
            jacobian_check_results.extend(imu_linearization.check_results)

            # H_local_I: (3, 22) = [H_Tk(6), H_Tk1(6), H_T_B_I(6), H_b_g(3), H_tau_I(1)].
            # Whitening is applied once to the complete local Jacobian, then
            # slices are inserted into the global sparse/dense assembly layout.
            H_local = np.hstack(
                [
                    imu_linearization.H_start_pose,
                    imu_linearization.H_end_pose,
                    imu_linearization.H_T_B_I,
                    imu_linearization.H_b_g,
                    imu_linearization.H_tau_I,
                ]
            )
            whitened_residual, whitened_local = whiten_residual_and_jacobian_dense(
                imu_linearization.residual,
                H_local,
                imu_residual_covariance,
            )
            start_pose_slice = slice(0, 6)
            end_pose_slice = slice(6, 12)
            extrinsic_slice = slice(12, 18)
            bias_slice = slice(18, 21)
            time_offset_slice = slice(21, 22)

            residual_specs.append((residual_name, 3, np.eye(3), "measurement"))
            residual_values[residual_name] = whitened_residual
            jacobian_blocks.extend(
                [
                    JacobianBlock(residual_name, pose_variable_names[interval_index], whitened_local[:, start_pose_slice]),
                    JacobianBlock(residual_name, pose_variable_names[interval_index + 1], whitened_local[:, end_pose_slice]),
                    JacobianBlock(residual_name, "T_B_I", whitened_local[:, extrinsic_slice]),
                    JacobianBlock(residual_name, "b_g", whitened_local[:, bias_slice]),
                    JacobianBlock(residual_name, "tau_I", whitened_local[:, time_offset_slice]),
                ]
            )
            imu_count += 1
        return imu_count

    def _append_priors(
        self,
        layout: VariableLayout,
        residual_specs: list[tuple[str, int, NDArray[np.float64], str]],
        residual_values: dict[str, NDArray[np.float64]],
        jacobian_blocks: list[JacobianBlock],
    ) -> None:
        '''Append priors for active calibration variables.
        
        Args:
            layout (VariableLayout): Global variable layout containing trajectory and calibration blocks.
            residual_specs (list[tuple[str, int, NDArray[np.float64], str]]): Mutable residual-block specifications for global assembly.
            residual_values (dict[str, NDArray[np.float64]]): Mutable residual values keyed by residual name.
            jacobian_blocks (list[JacobianBlock]): Mutable local Jacobian blocks for global assembly.
        '''

        active_calibration_names = {block.name for block in layout.calibration_blocks}
        for variable_name, nominal_transform in (("T_B_I", self.T_B_I_true), ("T_B_L", self.T_B_L_true)):
            if variable_name not in active_calibration_names:
                continue
            residual_name = f"prior_{variable_name}"
            prior_covariance = np.diag([0.05**2] * 3 + [0.1**2] * 3)
            prior_jacobian = extrinsic_prior_jacobian_left(nominal_transform, nominal_transform)
            whitened_residual, whitened_jacobian = whiten_residual_and_jacobian_dense(
                prior_jacobian.residual,
                prior_jacobian.H_X,
                prior_covariance,
            )
            residual_specs.append((residual_name, 6, np.eye(6), "prior"))
            residual_values[residual_name] = whitened_residual
            jacobian_blocks.append(JacobianBlock(residual_name, variable_name, whitened_jacobian))

        additive_prior_specs: tuple[tuple[str, int, NDArray[np.float64]], ...] = (
            ("b_g", 3, (0.02**2) * np.eye(3)),
            ("tau_I", 1, np.array([[0.02**2]])),
            ("tau_L", 1, np.array([[0.02**2]])),
        )
        for variable_name, variable_dimension, prior_covariance in additive_prior_specs:
            if variable_name not in active_calibration_names:
                continue
            residual_name = f"prior_{variable_name}"
            prior_residual = np.zeros(variable_dimension, dtype=float)
            prior_jacobian = -np.eye(variable_dimension)
            whitened_residual, whitened_jacobian = whiten_residual_and_jacobian_dense(
                prior_residual,
                prior_jacobian,
                prior_covariance,
            )
            residual_specs.append((residual_name, variable_dimension, np.eye(variable_dimension), "prior"))
            residual_values[residual_name] = whitened_residual
            jacobian_blocks.append(JacobianBlock(residual_name, variable_name, whitened_jacobian))

    @staticmethod
    def _metadata(
        layout: VariableLayout,
        pose_times: NDArray[np.float64],
        window_start_time: float,
        window_end_time: float,
        imu_count: int,
        lidar_count: int,
        jacobian_options: JacobianOptions,
        jacobian_check_results: list[JacobianCheckResult],
        fixed_extrinsic: FixedExtrinsic,
        practical_rank_policy: PracticalRankPolicy,
        accelerometer_options: AccelerometerOptions,
        accelerometer_count: int,
        accelerometer_candidate_count: int,
        accelerometer_rejection_reasons: dict[str, int],
        accelerometer_gate_mask: NDArray[np.bool_],
        accelerometer_check_results: list[JacobianCheckResult],
        accelerometer_terms: list[AccelerometerFactorTerms],
    ) -> dict[str, object]:
        '''Build metadata describing the assembled window and diagnostics.
        
        Args:
            layout (VariableLayout): Global variable layout containing trajectory and calibration blocks.
            pose_times (NDArray[np.float64]): Trajectory times associated with pose nodes.
            window_start_time (float): Clipped window start time.
            window_end_time (float): Clipped window end time.
            imu_count (int): Number of appended gyroscope factors.
            lidar_count (int): Number of appended LiDAR factors.
            jacobian_options (JacobianOptions): Analytic or numerical Jacobian configuration.
            jacobian_check_results (list[JacobianCheckResult]): Mutable factor Jacobian check results.
            fixed_extrinsic (FixedExtrinsic): Selected fixed-extrinsic body-frame convention.
            practical_rank_policy (PracticalRankPolicy): Practical-rank thresholds saved in bundle metadata.
            accelerometer_options (AccelerometerOptions): Optional accelerometer factor configuration.
            accelerometer_count (int): Number of appended accelerometer factors.
            accelerometer_candidate_count (int): Number of accelerometer candidates before gating.
            accelerometer_rejection_reasons (dict[str, int]): Counts of rejected accelerometer candidates.
            accelerometer_gate_mask (NDArray[np.bool_]): Boolean mask of accepted accelerometer candidates.
            accelerometer_check_results (list[JacobianCheckResult]): Mutable analytic-versus-numerical check results.
            accelerometer_terms (list[AccelerometerFactorTerms]): Mutable saved accelerometer intermediate terms.
        
        Returns:
            dict[str, object]: Metadata dictionary for the assembled bundle.
        '''

        trajectory_dimension = sum(block.dimension for block in layout.trajectory_blocks)
        calibration_dimension = sum(block.dimension for block in layout.calibration_blocks)
        max_check_error = max((result.max_absolute_error for result in jacobian_check_results), default=0.0)
        all_checks_passed = all(result.passed for result in jacobian_check_results)
        max_accel_check_error = max((result.max_absolute_error for result in accelerometer_check_results), default=0.0)
        all_accel_checks_passed = all(result.passed for result in accelerometer_check_results)
        return {
            "window": (window_start_time, window_end_time),
            "pose_times": pose_times,
            "trajectory_blocks": [(block.name, block.dimension) for block in layout.trajectory_blocks],
            "calibration_blocks": [(block.name, block.dimension) for block in layout.calibration_blocks],
            "calibration_labels": _labels_for_layout(layout),
            "counts": {"imu": imu_count, "lidar": lidar_count, "accelerometer": accelerometer_count},
            "trajectory_dimension": trajectory_dimension,
            "calibration_dimension": calibration_dimension,
            "total_dimension": layout.total_dim,
            "fixed_extrinsic": fixed_extrinsic,
            "fixed_extrinsic_value": np.eye(4).tolist() if fixed_extrinsic != "none" else None,
            "body_frame_definition": "B == L and T_B_L == I_4" if fixed_extrinsic == "T_B_L" else ("B == I and T_B_I == I_4" if fixed_extrinsic == "T_B_I" else "free sensor extrinsics"),
            "practical_rank_policy": practical_rank_policy.as_metadata(),
            "sensor_time_convention": "true_time = sensor_time + tau_sensor",
            "dimension_explanation": (
                "J columns = 6 * number_of_pose_nodes + number_of_active_calibration_parameters. "
                "When fixed_extrinsic='T_B_L' and both sensors are active, the calibration vector has 11 columns: "
                "T_B_I(6), b_g(3), tau_I(1), tau_L(1); free T_B_L adds 6 more columns. "
                "Use J_C or projected O_C for calibration-only observability analysis."
            ),
            "jacobian_method": jacobian_options.method,
            "finite_difference_epsilon": jacobian_options.finite_difference_epsilon,
            "jacobian_check_results": tuple(jacobian_check_results),
            "maximum_jacobian_check_error": float(max_check_error),
            "all_jacobian_checks_passed": bool(all_checks_passed),
            "accelerometer_mode": accelerometer_options.mode,
            "accelerometer_options": accelerometer_options.as_metadata(),
            "accelerometer_factor_count": int(accelerometer_count),
            "accelerometer_candidate_count": int(accelerometer_candidate_count),
            "accelerometer_rejection_reasons": dict(accelerometer_rejection_reasons),
            "accelerometer_gate_mask": np.asarray(accelerometer_gate_mask, dtype=bool),
            "accelerometer_jacobian_check_results": tuple(accelerometer_check_results),
            "maximum_accelerometer_jacobian_check_error": float(max_accel_check_error),
            "all_accelerometer_jacobian_checks_passed": bool(all_accel_checks_passed),
            "accelerometer_factor_terms": tuple(accelerometer_terms),
        }

    @staticmethod
    def _scaled_bundle(
        bundle: JacobianBundle,
        layout: VariableLayout,
        parameter_scaling: object,
        use_sparse: bool,
    ) -> JacobianBundle:
        '''Apply parameter scaling while preserving bundle partitions.
        
        Args:
            bundle (JacobianBundle): Assembled Jacobian bundle.
            layout (VariableLayout): Global variable layout containing trajectory and calibration blocks.
            parameter_scaling (object): Optional physical parameter scaling specification.
            use_sparse (bool): Whether the returned global Jacobian uses sparse storage.
        
        Returns:
            JacobianBundle: Bundle with scaled Jacobian columns.
        '''

        trajectory_dimension = sum(block.dimension for block in layout.trajectory_blocks)
        if use_sparse:
            scaling_matrix = build_parameter_scaling_sparse(layout.blocks, parameter_scaling)
            scaled_jacobian = scale_jacobian_sparse(bundle.J, scaling_matrix)  # type: ignore[arg-type]
            trajectory_jacobian = scaled_jacobian[:, :trajectory_dimension].tocsr()
            calibration_jacobian = scaled_jacobian[:, trajectory_dimension:].tocsr()
        else:
            scaling_matrix = build_parameter_scaling_dense(layout.blocks, parameter_scaling)
            scaled_jacobian = scale_jacobian_dense(bundle.J, scaling_matrix)  # type: ignore[arg-type]
            trajectory_jacobian = scaled_jacobian[:, :trajectory_dimension]
            calibration_jacobian = scaled_jacobian[:, trajectory_dimension:]
        return JacobianBundle(
            J=scaled_jacobian,
            J_T=trajectory_jacobian,
            J_C=calibration_jacobian,
            residual=bundle.residual,
            row_slices=bundle.row_slices,
            trajectory_column_slices=bundle.trajectory_column_slices,
            calibration_column_slices=bundle.calibration_column_slices,
            metadata=bundle.metadata,
        )

    @staticmethod
    def _empty_bundle(use_sparse: bool) -> JacobianBundle:
        '''Create an empty dense or sparse Jacobian bundle.
        
        Args:
            use_sparse (bool): Whether the returned global Jacobian uses sparse storage.
        
        Returns:
            JacobianBundle: Empty Jacobian bundle.
        '''

        layout = VariableLayout.from_specs([])
        residual_blocks = make_residual_blocks([])
        if use_sparse:
            return assemble_jacobian_sparse(layout, residual_blocks, [], {})
        return assemble_jacobian_dense(layout, residual_blocks, [], {})