"""Sensor simulation for IMU and LiDAR odometry flows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..lie_se3 import se3_exp, se3_inverse
from ..lie_so3 import so3_hat

##################################################
# Sensor data structures
##################################################
@dataclass(frozen=True)
class ImuData:
    '''Store simulated IMU timestamps, measurements, covariances, and convention.
    
    Attributes:
        sensor_timestamps: IMU timestamps on the sensor clock.
        true_times: Corresponding trajectory-clock times.
        gyroscope: Gyroscope measurements with shape `(N, 3)`.
        accelerometer: Specific-force measurements with shape `(N, 3)`.
        gyro_covariance: Gyroscope covariance matrix.
        accel_covariance: Accelerometer covariance matrix.
        gravity_world: World gravity vector.
        accelerometer_convention: Text identifier of the measurement convention.
    '''

    sensor_timestamps: NDArray[np.float64]
    true_times: NDArray[np.float64]
    gyroscope: NDArray[np.float64]
    accelerometer: NDArray[np.float64]
    gyro_covariance: NDArray[np.float64]
    accel_covariance: NDArray[np.float64]
    gravity_world: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=float))
    accelerometer_convention: str = "specific_force_imu_frame_R_IW_times_a_minus_g"


@dataclass(frozen=True)
class LidarOdometryData:
    '''Store simulated LiDAR relative-pose odometry.
    
    Attributes:
        sensor_timestamps: LiDAR samples on the sensor clock.
        true_times: Corresponding trajectory-clock times.
        measurements: Relative-pose measurements with shape `(M, 4, 4)`.
        covariances: Measurement covariances with shape `(M, 6, 6)`.
        relative_start_times: Interval starts on the sensor clock.
        relative_end_times: Interval ends on the sensor clock.
    '''

    sensor_timestamps: NDArray[np.float64]
    true_times: NDArray[np.float64]
    measurements: NDArray[np.float64]
    covariances: NDArray[np.float64]
    relative_start_times: NDArray[np.float64]
    relative_end_times: NDArray[np.float64]

##################################################
# Sensor-clock construction
##################################################
def _sensor_clock_samples(
    trajectory_start_time: float,
    trajectory_end_time: float,
    rate_hz: float,
    temporal_offset: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Build aligned sensor-clock and trajectory-clock samples.
    
    Args:
        trajectory_start_time (float): First valid trajectory time.
        trajectory_end_time (float): Last valid trajectory time.
        rate_hz (float): Sensor sampling rate in hertz.
        temporal_offset (float): Sensor-to-trajectory clock offset in seconds.
    
    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple `(sensor_timestamps, true_timestamps)`.
    
    Raises:
        ValueError: `rate_hz` is not positive.
    '''

    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    sample_period = 1.0 / float(rate_hz)
    sensor_start_time = trajectory_start_time - float(temporal_offset)
    sensor_end_time = trajectory_end_time - float(temporal_offset)
    sensor_timestamps = np.arange(sensor_start_time, sensor_end_time + 0.5 * sample_period, sample_period)
    true_timestamps = sensor_timestamps + float(temporal_offset)
    inside_trajectory = (true_timestamps >= trajectory_start_time - 1e-12) & (
        true_timestamps <= trajectory_end_time + 1e-12
    )
    return sensor_timestamps[inside_trajectory], true_timestamps[inside_trajectory]

##################################################
# IMU simulation
##################################################
def generate_imu_data(
    trajectory: object,
    rate_hz: float,
    tau_I_true: float,
    gyro_bias_true: ArrayLike,
    gyro_noise_std: float,
    accel_noise_std: float,
    T_B_I_true: ArrayLike | None = None,
    *,
    seed: int = 0,
) -> ImuData:
    '''Generate synthetic gyroscope and accelerometer measurements.
    
    Args:
        trajectory (object): Continuously queryable body trajectory.
        rate_hz (float): Sensor sampling rate in hertz.
        tau_I_true (float): True IMU clock offset in seconds.
        gyro_bias_true (ArrayLike): True gyroscope bias vector.
        gyro_noise_std (float): Gyroscope white-noise standard deviation.
        accel_noise_std (float): Accelerometer white-noise standard deviation.
        T_B_I_true (ArrayLike | None): True body-from-IMU extrinsic transform, or `None` for identity.
        seed (int): Random seed used for deterministic simulation.
    
    Returns:
        ImuData: Simulated IMU data.
    
    Raises:
        ValueError: The bias or IMU extrinsic has an invalid shape.
    
    Notes:
        `true_time = sensor_timestamp + tau_I_true`.
        Accelerometer samples follow `f_m^I = R_IW (a_I^W - g_W) + noise` and include lever-arm acceleration.
    '''

    random_generator = np.random.default_rng(seed)
    gyro_bias_vector = np.asarray(gyro_bias_true, dtype=float)
    if gyro_bias_vector.shape != (3,):
        raise ValueError("gyro_bias_true must be (3,)")
    body_from_imu_true = np.eye(4) if T_B_I_true is None else np.asarray(T_B_I_true, dtype=float)
    if body_from_imu_true.shape != (4, 4):
        raise ValueError("T_B_I_true must be None or a transform with shape (4, 4)")
    imu_rotation_in_body = body_from_imu_true[:3, :3]
    imu_position_in_body = body_from_imu_true[:3, 3]

    trajectory_start_time = float(getattr(trajectory, "start_time"))
    trajectory_end_time = float(getattr(trajectory, "end_time"))

    # Build the IMU clock so its samples cover the requested true trajectory span.
    sensor_timestamps, true_timestamps = _sensor_clock_samples(
        trajectory_start_time,
        trajectory_end_time,
        rate_hz,
        tau_I_true,
    )

    # Simulate gyroscope measurements from trajectory angular velocity plus bias and noise.
    gyroscope_without_noise = np.vstack(
        [trajectory.angular_velocity_body_at(float(true_time)) for true_time in true_timestamps]
    )
    gyroscope_measurements = gyroscope_without_noise + gyro_bias_vector + random_generator.normal(
        0.0,
        gyro_noise_std,
        size=gyroscope_without_noise.shape,
    )

    # Simulate accelerometer specific force at the IMU origin. The body-origin
    # term contains gravity, and the lever-arm term adds tangential and
    # centripetal accelerations before rotating the result into the IMU frame.
    gravity_world = np.array([0.0, 0.0, -9.81], dtype=float)
    accelerometer_rows = []
    angular_step = min(1e-3, 0.25 / float(rate_hz))
    for true_time in true_timestamps:
        body_pose = trajectory.pose_at(float(true_time))
        body_rotation_world = body_pose[:3, :3]
        acceleration_world = trajectory.acceleration_at(float(true_time))
        omega_body = trajectory.angular_velocity_body_at(float(true_time))
        earlier_time = max(trajectory_start_time, float(true_time) - angular_step)
        later_time = min(trajectory_end_time, float(true_time) + angular_step)
        if later_time > earlier_time:
            alpha_body = (
                trajectory.angular_velocity_body_at(later_time)
                - trajectory.angular_velocity_body_at(earlier_time)
            ) / (later_time - earlier_time)
        else:
            alpha_body = np.zeros(3, dtype=float)
        lever_arm_matrix = so3_hat(alpha_body) + so3_hat(omega_body) @ so3_hat(omega_body)
        body_specific_force = body_rotation_world.T @ (acceleration_world - gravity_world) + lever_arm_matrix @ imu_position_in_body
        accelerometer_rows.append(imu_rotation_in_body.T @ body_specific_force)
    accelerometer_measurements = np.vstack(accelerometer_rows) + random_generator.normal(
        0.0,
        accel_noise_std,
        size=(true_timestamps.size, 3),
    )

    return ImuData(
        sensor_timestamps=sensor_timestamps,
        true_times=true_timestamps,
        gyroscope=gyroscope_measurements,
        accelerometer=accelerometer_measurements,
        gyro_covariance=(gyro_noise_std**2) * np.eye(3),
        accel_covariance=(accel_noise_std**2) * np.eye(3),
        gravity_world=gravity_world,
    )

##################################################
# LiDAR odometry simulation
##################################################
def generate_lidar_odometry(
    trajectory: object,
    rate_hz: float,
    tau_L_true: float,
    T_B_L_true: ArrayLike,
    pose_noise_std: ArrayLike,
    *,
    seed: int = 0,
) -> LidarOdometryData:
    '''Generate synthetic LiDAR relative-pose measurements.
    
    Args:
        trajectory (object): Continuously queryable body trajectory.
        rate_hz (float): Sensor sampling rate in hertz.
        tau_L_true (float): True LiDAR clock offset in seconds.
        T_B_L_true (ArrayLike): True body-from-LiDAR extrinsic transform.
        pose_noise_std (ArrayLike): Scalar or six-dimensional LiDAR pose-noise standard deviation.
        seed (int): Random seed used for deterministic simulation.
    
    Returns:
        LidarOdometryData: Simulated LiDAR odometry data.
    
    Raises:
        ValueError: The LiDAR extrinsic or noise vector has an invalid shape.
    
    Notes:
        `true_time = sensor_timestamp + tau_L_true` and measurement noise is applied as `Exp(noise_xi) @ Z`.
    '''

    random_generator = np.random.default_rng(seed)
    body_from_lidar_true = np.asarray(T_B_L_true, dtype=float)
    if body_from_lidar_true.shape != (4, 4):
        raise ValueError("T_B_L_true must be (4, 4)")
    pose_noise_std_vector = np.asarray(pose_noise_std, dtype=float)
    if pose_noise_std_vector.shape == ():
        pose_noise_std_vector = np.full(6, float(pose_noise_std_vector))
    if pose_noise_std_vector.shape != (6,):
        raise ValueError("pose_noise_std must be scalar or shape (6,)")

    trajectory_start_time = float(getattr(trajectory, "start_time"))
    trajectory_end_time = float(getattr(trajectory, "end_time"))

    # Build the LiDAR clock and keep interval endpoints in sensor time for time-offset Jacobians.
    sensor_timestamps, true_timestamps = _sensor_clock_samples(
        trajectory_start_time,
        trajectory_end_time,
        rate_hz,
        tau_L_true,
    )

    # Simulate LiDAR odometry by comparing consecutive LiDAR poses in the world frame.
    relative_pose_measurements = []
    measurement_covariances = []
    relative_start_sensor_times = []
    relative_end_sensor_times = []
    for interval_index in range(sensor_timestamps.size - 1):
        relative_start_sensor_time = float(sensor_timestamps[interval_index])
        relative_end_sensor_time = float(sensor_timestamps[interval_index + 1])
        relative_start_true_time = float(true_timestamps[interval_index])
        relative_end_true_time = float(true_timestamps[interval_index + 1])
        start_body_pose = trajectory.pose_at(relative_start_true_time)
        end_body_pose = trajectory.pose_at(relative_end_true_time)
        start_lidar_pose = start_body_pose @ body_from_lidar_true
        end_lidar_pose = end_body_pose @ body_from_lidar_true
        noiseless_relative_pose = se3_inverse(start_lidar_pose) @ end_lidar_pose
        noise_tangent = random_generator.normal(0.0, pose_noise_std_vector)
        relative_pose_measurements.append(se3_exp(noise_tangent) @ noiseless_relative_pose)
        measurement_covariances.append(np.diag(pose_noise_std_vector**2))
        relative_start_sensor_times.append(relative_start_sensor_time)
        relative_end_sensor_times.append(relative_end_sensor_time)

    return LidarOdometryData(
        sensor_timestamps=sensor_timestamps,
        true_times=true_timestamps,
        measurements=(
            np.stack(relative_pose_measurements, axis=0)
            if relative_pose_measurements
            else np.zeros((0, 4, 4))
        ),
        covariances=np.stack(measurement_covariances, axis=0) if measurement_covariances else np.zeros((0, 6, 6)),
        relative_start_times=np.asarray(relative_start_sensor_times, dtype=float),
        relative_end_times=np.asarray(relative_end_sensor_times, dtype=float),
    )