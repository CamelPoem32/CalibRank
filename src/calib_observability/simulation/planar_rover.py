"""Planar rover simulation embedded in full SE(3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..lie_se3 import se3_exp
from .dataset import CalibrationSimulationDataset
from .sensors import generate_imu_data, generate_lidar_odometry
from .trajectory import AnalyticTrajectory

##################################################
# Simulation configuration
##################################################
@dataclass(frozen=True)
class PlanarRoverConfig:
    '''Configure rover geometry, motion, sensors, and true calibration.
    
    Attributes:
        rectangle_width: Rectangle width in metres.
        rectangle_height: Rectangle height in metres.
        straight_speed: Straight-segment speed in metres per second.
        turn_duration: Duration of each quarter turn.
        acceleration_duration: Nominal acceleration phase duration.
        total_laps: Requested number of rectangle laps.
        imu_rate_hz: IMU sampling rate.
        lidar_rate_hz: LiDAR odometry rate.
        tau_I_true: True IMU temporal offset.
        tau_L_true: True LiDAR temporal offset.
        gyro_bias_true: True gyroscope bias.
        T_B_I_true: Rotation-first tangent used to construct `T_B_I`.
        T_B_L_true: Rotation-first tangent used to construct `T_B_L`.
        gyro_noise_std: Gyroscope white-noise standard deviation.
        accel_noise_std: Accelerometer white-noise standard deviation.
        lidar_pose_noise_std: LiDAR pose-noise standard deviations.
        random_seed: Base random seed.
        mode: Default trajectory mode.
    '''

    rectangle_width: float = 8.0
    rectangle_height: float = 5.0
    straight_speed: float = 1.0
    turn_duration: float = 1.5
    acceleration_duration: float = 0.8
    total_laps: int = 1
    imu_rate_hz: float = 50.0
    lidar_rate_hz: float = 5.0
    tau_I_true: float = 0.01
    tau_L_true: float = -0.02
    gyro_bias_true: tuple[float, float, float] = (0.005, -0.003, 0.01)
    T_B_I_true: tuple[float, float, float, float, float, float] = (0.01, -0.02, 0.03, 0.12, 0.0, 0.08)
    T_B_L_true: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.04, 0.35, 0.05, 0.12)
    gyro_noise_std: float = 0.002
    accel_noise_std: float = 0.03
    lidar_pose_noise_std: tuple[float, float, float, float, float, float] = (0.005, 0.005, 0.005, 0.02, 0.02, 0.02)
    random_seed: int = 7
    mode: str = "one_rectangle"

##################################################
# Trajectory construction helpers
##################################################
VectorFunction = Callable[[float], NDArray[np.float64]]
ScalarFunction = Callable[[float], float]


def _positive(value: float, name: str) -> float:
    '''Validate and return a finite positive scalar.
    
    Args:
        value (float): Value to validate, differentiate, or convert.
        name (str): Name used in validation errors.
    
    Returns:
        float: Validated positive scalar.
    
    Raises:
        ValueError: The value is non-finite or not positive.
    '''
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _zero_vector(_: float) -> NDArray[np.float64]:
    '''Return a three-dimensional zero vector.
    
    Args:
        _ (float): Value supplied for `_`.
    
    Returns:
        NDArray[np.float64]: Zero vector with shape `(3,)`.
    '''
    return np.zeros(3, dtype=float)


def _make_trajectory(
    duration: float,
    mode: str,
    position_function: VectorFunction,
    euler_function: VectorFunction,
    velocity_function: VectorFunction,
    acceleration_function: VectorFunction,
    yaw_rate_function: ScalarFunction | None = None,
) -> AnalyticTrajectory:
    '''Construct an analytic trajectory from coordinate functions.
    
    Args:
        duration (float): Trajectory duration in seconds.
        mode (str): Textual trajectory or simulation mode.
        position_function (VectorFunction): Analytic world-position function.
        euler_function (VectorFunction): Analytic roll-pitch-yaw function.
        velocity_function (VectorFunction): Analytic world-velocity function.
        acceleration_function (VectorFunction): Analytic world-acceleration function.
        yaw_rate_function (ScalarFunction | None): Optional analytic yaw-rate function.
    
    Returns:
        AnalyticTrajectory: Configured analytic trajectory.
    '''
    return AnalyticTrajectory(
        start_time=0.0,
        end_time=_positive(duration, "duration"),
        mode=mode,
        position_function=position_function,
        euler_function=euler_function,
        velocity_function=velocity_function,
        acceleration_function=acceleration_function,
        yaw_rate_function=yaw_rate_function,
    )

##################################################
# Planar motion modes
##################################################
def _straight_trajectory(config: PlanarRoverConfig, accelerating: bool) -> AnalyticTrajectory:
    '''Construct a straight constant-speed or accelerating trajectory.
    
    Args:
        config (PlanarRoverConfig): Planar rover simulation configuration.
        accelerating (bool): Whether the straight trajectory includes an acceleration phase.
    
    Returns:
        AnalyticTrajectory: Straight analytic trajectory.
    '''
    travel_distance = _positive(config.rectangle_width, "rectangle_width")
    target_speed = _positive(config.straight_speed, "straight_speed")

    if not accelerating:
        duration = travel_distance / target_speed

        def position_function(time_seconds: float) -> NDArray[np.float64]:
            # Build trajectory coordinates for a literal constant-velocity line.
            '''Return straight-line world position.
            
            Args:
                time_seconds (float): Trajectory query time in seconds.
            
            Returns:
                NDArray[np.float64]: World position with shape `(3,)`.
            '''
            return np.array([target_speed * time_seconds, 0.0, 0.0], dtype=float)

        def velocity_function(_: float) -> NDArray[np.float64]:
            '''Return straight-line world velocity.
            
            Args:
                _ (float): Value supplied for `_`.
            
            Returns:
                NDArray[np.float64]: World velocity with shape `(3,)`.
            '''
            return np.array([target_speed, 0.0, 0.0], dtype=float)

        return _make_trajectory(
            duration,
            "straight_constant_velocity",
            position_function,
            _zero_vector,
            velocity_function,
            _zero_vector,
            lambda _: 0.0,
        )

    acceleration_duration = _positive(config.acceleration_duration, "acceleration_duration")
    longitudinal_acceleration = target_speed / acceleration_duration
    distance_during_full_acceleration = 0.5 * longitudinal_acceleration * acceleration_duration**2
    if travel_distance <= distance_during_full_acceleration:
        used_acceleration_duration = float(np.sqrt(2.0 * travel_distance / longitudinal_acceleration))
        cruise_duration = 0.0
    else:
        used_acceleration_duration = acceleration_duration
        cruise_duration = (travel_distance - distance_during_full_acceleration) / target_speed
    duration = used_acceleration_duration + cruise_duration
    distance_after_acceleration = 0.5 * longitudinal_acceleration * used_acceleration_duration**2
    speed_after_acceleration = longitudinal_acceleration * used_acceleration_duration

    def traveled_distance(time_seconds: float) -> float:
        '''Return distance travelled during accelerate-then-cruise motion.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            float: Distance travelled in metres.
        '''
        if time_seconds <= used_acceleration_duration:
            return 0.5 * longitudinal_acceleration * time_seconds**2
        return distance_after_acceleration + speed_after_acceleration * (time_seconds - used_acceleration_duration)

    def position_function(time_seconds: float) -> NDArray[np.float64]:
        # Build trajectory coordinates for a literal accelerate-then-cruise line.
        '''Return straight-line world position.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World position with shape `(3,)`.
        '''
        return np.array([traveled_distance(time_seconds), 0.0, 0.0], dtype=float)

    def velocity_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return straight-line world velocity.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World velocity with shape `(3,)`.
        '''
        if time_seconds <= used_acceleration_duration:
            speed = longitudinal_acceleration * time_seconds
        else:
            speed = speed_after_acceleration
        return np.array([speed, 0.0, 0.0], dtype=float)

    def acceleration_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return straight-line world acceleration.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World acceleration with shape `(3,)`.
        '''
        acceleration = longitudinal_acceleration if time_seconds < used_acceleration_duration else 0.0
        return np.array([acceleration, 0.0, 0.0], dtype=float)

    return _make_trajectory(
        duration,
        "straight_accelerating",
        position_function,
        _zero_vector,
        velocity_function,
        acceleration_function,
        lambda _: 0.0,
    )


def _single_turn_trajectory(config: PlanarRoverConfig) -> AnalyticTrajectory:
    '''Construct one constant-speed circular quarter turn.
    
    Args:
        config (PlanarRoverConfig): Planar rover simulation configuration.
    
    Returns:
        AnalyticTrajectory: Quarter-turn analytic trajectory.
    '''
    duration = _positive(config.turn_duration, "turn_duration")
    speed = _positive(config.straight_speed, "straight_speed")
    yaw_rate = 0.5 * np.pi / duration
    turn_radius = speed / yaw_rate

    def yaw_angle(time_seconds: float) -> float:
        '''Return yaw accumulated during the quarter turn.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            float: Yaw angle in radians.
        '''
        return yaw_rate * time_seconds

    def position_function(time_seconds: float) -> NDArray[np.float64]:
        # Build trajectory coordinates for one literal circular quarter-turn.
        '''Return circular-turn world position.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World position with shape `(3,)`.
        '''
        yaw = yaw_angle(time_seconds)
        return np.array([turn_radius * np.sin(yaw), turn_radius * (1.0 - np.cos(yaw)), 0.0], dtype=float)

    def euler_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return circular-turn Euler angles.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: Euler angles with shape `(3,)`.
        '''
        return np.array([0.0, 0.0, yaw_angle(time_seconds)], dtype=float)

    def velocity_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return circular-turn world velocity.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World velocity with shape `(3,)`.
        '''
        yaw = yaw_angle(time_seconds)
        return np.array([turn_radius * yaw_rate * np.cos(yaw), turn_radius * yaw_rate * np.sin(yaw), 0.0], dtype=float)

    def acceleration_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return circular-turn centripetal acceleration.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World acceleration with shape `(3,)`.
        '''
        yaw = yaw_angle(time_seconds)
        return np.array(
            [-turn_radius * yaw_rate**2 * np.sin(yaw), turn_radius * yaw_rate**2 * np.cos(yaw), 0.0],
            dtype=float,
        )

    return _make_trajectory(
        duration,
        "single_turn",
        position_function,
        euler_function,
        velocity_function,
        acceleration_function,
        lambda _: yaw_rate,
    )


def _rectangle_trajectory(config: PlanarRoverConfig, laps: int) -> AnalyticTrajectory:
    '''Construct a literal rectangle from straight and in-place turn phases.
    
    Args:
        config (PlanarRoverConfig): Planar rover simulation configuration.
        laps (int): Number of rectangular laps.
    
    Returns:
        AnalyticTrajectory: Rectangular analytic trajectory.
    
    Notes:
        The rectangle is literal: each edge is followed by an in-place quarter turn.
    '''
    rectangle_width = _positive(config.rectangle_width, "rectangle_width")
    rectangle_height = _positive(config.rectangle_height, "rectangle_height")
    straight_speed = _positive(config.straight_speed, "straight_speed")
    turn_duration = _positive(config.turn_duration, "turn_duration")
    lap_count = max(int(laps), 1)
    quarter_turn_rate = 0.5 * np.pi / turn_duration

    # A rectangle is represented literally: straight edge, in-place quarter-turn,
    # then the next straight edge. No spline is fitted through the corner points.
    edge_definitions = (
        (rectangle_width, np.array([1.0, 0.0], dtype=float), 0.0),
        (rectangle_height, np.array([0.0, 1.0], dtype=float), 0.5 * np.pi),
        (rectangle_width, np.array([-1.0, 0.0], dtype=float), np.pi),
        (rectangle_height, np.array([0.0, -1.0], dtype=float), 1.5 * np.pi),
    )
    single_lap_duration = sum(edge_length / straight_speed + turn_duration for edge_length, _, _ in edge_definitions)
    total_duration = lap_count * single_lap_duration

    def rectangle_state(
        time_seconds: float,
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], float]:
        '''Evaluate the active rectangle phase and its kinematic state.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], float]: Tuple `(position_xy, yaw, velocity_xy, acceleration_xy, yaw_rate)`.
        '''
        elapsed_time = float(np.clip(time_seconds, 0.0, total_duration))
        if elapsed_time >= total_duration:
            return (
                np.zeros(2, dtype=float),
                2.0 * np.pi * lap_count,
                np.zeros(2, dtype=float),
                np.zeros(2, dtype=float),
                0.0,
            )

        lap_index = min(int(elapsed_time // single_lap_duration), lap_count - 1)
        time_inside_lap = elapsed_time - lap_index * single_lap_duration
        yaw_offset = 2.0 * np.pi * lap_index
        position_xy = np.zeros(2, dtype=float)

        for edge_length, edge_direction, edge_yaw in edge_definitions:
            # Checking straight segment
            edge_duration = edge_length / straight_speed
            if time_inside_lap <= edge_duration:
                position = position_xy + edge_direction * straight_speed * time_inside_lap
                velocity = edge_direction * straight_speed
                return position, yaw_offset + edge_yaw, velocity, np.zeros(2, dtype=float), 0.0

            position_xy = position_xy + edge_direction * edge_length
            time_inside_lap -= edge_duration
            # Checking corner segment
            if time_inside_lap <= turn_duration:
                yaw = yaw_offset + edge_yaw + quarter_turn_rate * time_inside_lap
                return position_xy.copy(), yaw, np.zeros(2, dtype=float), np.zeros(2, dtype=float), quarter_turn_rate

            time_inside_lap -= turn_duration

        return (
            np.zeros(2, dtype=float),
            yaw_offset + 2.0 * np.pi,
            np.zeros(2, dtype=float),
            np.zeros(2, dtype=float),
            0.0,
        )

    def position_function(time_seconds: float) -> NDArray[np.float64]:
        # Build trajectory coordinates by evaluating the active rectangle phase.
        '''Return rectangle-path world position.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World position with shape `(3,)`.
        '''
        position_xy, _, _, _, _ = rectangle_state(time_seconds)
        return np.array([position_xy[0], position_xy[1], 0.0], dtype=float)

    def euler_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return rectangle-path Euler angles.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: Euler angles with shape `(3,)`.
        '''
        _, yaw, _, _, _ = rectangle_state(time_seconds)
        return np.array([0.0, 0.0, yaw], dtype=float)

    def velocity_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return rectangle-path world velocity.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World velocity with shape `(3,)`.
        '''
        _, _, velocity_xy, _, _ = rectangle_state(time_seconds)
        return np.array([velocity_xy[0], velocity_xy[1], 0.0], dtype=float)

    def acceleration_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return rectangle-path world acceleration.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World acceleration with shape `(3,)`.
        '''
        _, _, _, acceleration_xy, _ = rectangle_state(time_seconds)
        return np.array([acceleration_xy[0], acceleration_xy[1], 0.0], dtype=float)

    def yaw_rate_function(time_seconds: float) -> float:
        '''Return rectangle-path yaw rate.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            float: Yaw rate in radians per second.
        '''
        _, _, _, _, yaw_rate = rectangle_state(time_seconds)
        return yaw_rate

    return _make_trajectory(
        total_duration,
        "multiple_rectangles" if lap_count > 1 else "one_rectangle",
        position_function,
        euler_function,
        velocity_function,
        acceleration_function,
        yaw_rate_function,
    )

##################################################
# Reference and 3D motion modes
##################################################
def _stationary_trajectory() -> AnalyticTrajectory:
    '''Construct a stationary reference trajectory.
    
    Returns:
        AnalyticTrajectory: Stationary analytic trajectory.
    '''
    return _make_trajectory(
        8.0,
        "stationary",
        _zero_vector,
        _zero_vector,
        _zero_vector,
        _zero_vector,
        lambda _: 0.0,
    )


def _multi_axis_3d_trajectory() -> AnalyticTrajectory:
    '''Construct the multi-axis sinusoidal 3D reference trajectory.
    
    Returns:
        AnalyticTrajectory: Multi-axis analytic trajectory.
    '''
    duration = 16.0

    def position_function(time_seconds: float) -> NDArray[np.float64]:
        # Build trajectory coordinates from the stated sinusoidal 3D reference.
        '''Return the sinusoidal 3D world position.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World position with shape `(3,)`.
        '''
        return np.array(
            [
                2.5 * np.sin(0.4 * time_seconds),
                2.0 * np.sin(0.7 * time_seconds + 0.3),
                0.8 * np.sin(0.5 * time_seconds),
            ],
            dtype=float,
        )

    def euler_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return the multi-axis roll-pitch-yaw trajectory.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: Euler angles with shape `(3,)`.
        '''
        return np.array(
            [
                0.25 * np.sin(0.6 * time_seconds),
                0.22 * np.cos(0.45 * time_seconds),
                0.8 * np.sin(0.35 * time_seconds) + 0.2 * time_seconds,
            ],
            dtype=float,
        )

    def velocity_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return the analytic 3D world velocity.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World velocity with shape `(3,)`.
        '''
        return np.array(
            [
                2.5 * 0.4 * np.cos(0.4 * time_seconds),
                2.0 * 0.7 * np.cos(0.7 * time_seconds + 0.3),
                0.8 * 0.5 * np.cos(0.5 * time_seconds),
            ],
            dtype=float,
        )

    def acceleration_function(time_seconds: float) -> NDArray[np.float64]:
        '''Return the analytic 3D world acceleration.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World acceleration with shape `(3,)`.
        '''
        return np.array(
            [
                -2.5 * 0.4**2 * np.sin(0.4 * time_seconds),
                -2.0 * 0.7**2 * np.sin(0.7 * time_seconds + 0.3),
                -0.8 * 0.5**2 * np.sin(0.5 * time_seconds),
            ],
            dtype=float,
        )

    def yaw_rate_function(time_seconds: float) -> float:
        '''Return the analytic yaw rate.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            float: Yaw rate in radians per second.
        '''
        return float(0.8 * 0.35 * np.cos(0.35 * time_seconds) + 0.2)

    return _make_trajectory(
        duration,
        "multi-axis_3d_reference_motion",
        position_function,
        euler_function,
        velocity_function,
        acceleration_function,
        yaw_rate_function,
    )

##################################################
# Public simulation interface
##################################################
def create_trajectory(config: PlanarRoverConfig, mode: str | None = None) -> AnalyticTrajectory:
    '''Create the configured trajectory mode.
    
    Args:
        config (PlanarRoverConfig): Planar rover simulation configuration.
        mode (str | None): Textual trajectory or simulation mode.
    
    Returns:
        AnalyticTrajectory: Selected analytic trajectory.
    
    Raises:
        ValueError: The requested mode is unknown.
    '''

    selected_mode = config.mode if mode is None else mode
    if selected_mode == "stationary":
        return _stationary_trajectory()
    if selected_mode == "straight_constant_velocity":
        return _straight_trajectory(config, accelerating=False)
    if selected_mode == "straight_accelerating":
        return _straight_trajectory(config, accelerating=True)
    if selected_mode == "single_turn":
        return _single_turn_trajectory(config)
    if selected_mode == "one_rectangle":
        return _rectangle_trajectory(config, 1)
    if selected_mode == "multiple_rectangles":
        return _rectangle_trajectory(config, max(config.total_laps, 2))
    if selected_mode == "multi-axis_3d_reference_motion":
        return _multi_axis_3d_trajectory()
    raise ValueError(f"unknown motion mode {selected_mode!r}")


def simulate_planar_rover(
    config: PlanarRoverConfig | None = None,
    *,
    mode: str | None = None,
) -> CalibrationSimulationDataset:
    '''Generate a complete synthetic rover calibration dataset.
    
    Args:
        config (PlanarRoverConfig | None): Planar rover simulation configuration.
        mode (str | None): Textual trajectory or simulation mode.
    
    Returns:
        CalibrationSimulationDataset: Synthetic calibration dataset.
    '''

    simulation_config = PlanarRoverConfig() if config is None else config

    # Build the body trajectory coordinates directly from the selected mode.
    trajectory = create_trajectory(simulation_config, mode)

    # Convert true calibration tangent vectors into SE(3) extrinsic transforms.
    body_from_imu_true = se3_exp(np.asarray(simulation_config.T_B_I_true, dtype=float))
    body_from_lidar_true = se3_exp(np.asarray(simulation_config.T_B_L_true, dtype=float))

    # Simulate IMU measurements from the trajectory and true IMU clock offset.
    imu = generate_imu_data(
        trajectory,
        simulation_config.imu_rate_hz,
        simulation_config.tau_I_true,
        np.asarray(simulation_config.gyro_bias_true, dtype=float),
        simulation_config.gyro_noise_std,
        simulation_config.accel_noise_std,
        body_from_imu_true,
        seed=simulation_config.random_seed,
    )

    # Simulate LiDAR relative odometry from the same trajectory and extrinsics.
    lidar = generate_lidar_odometry(
        trajectory,
        simulation_config.lidar_rate_hz,
        simulation_config.tau_L_true,
        body_from_lidar_true,
        np.asarray(simulation_config.lidar_pose_noise_std, dtype=float),
        seed=simulation_config.random_seed + 1,
    )
    return CalibrationSimulationDataset(
        trajectory=trajectory,
        imu=imu,
        lidar=lidar,
        T_B_L_true=body_from_lidar_true,
        T_B_I_true=body_from_imu_true,
        tau_I_true=simulation_config.tau_I_true,
        tau_L_true=simulation_config.tau_L_true,
        gyro_bias_true=np.asarray(simulation_config.gyro_bias_true, dtype=float),
    )