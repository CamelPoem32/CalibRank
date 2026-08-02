"""Continuously queryable analytic trajectories for simulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..lie_so3 import so3_log

##################################################
# Trajectory function types
##################################################
VectorFunction = Callable[[float], NDArray[np.float64]]
ScalarFunction = Callable[[float], float]

##################################################
# Pose construction helpers
##################################################
def euler_zyx_to_rotation(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    '''Convert ZYX Euler angles to a rotation matrix.
    
    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.
    
    Returns:
        NDArray[np.float64]: Rotation matrix with shape `(3, 3)`.
    '''

    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_roll, -sin_roll], [0.0, sin_roll, cos_roll]])
    rotation_y = np.array([[cos_pitch, 0.0, sin_pitch], [0.0, 1.0, 0.0], [-sin_pitch, 0.0, cos_pitch]])
    rotation_z = np.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def pose_from_xyz_rpy(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> NDArray[np.float64]:
    '''Build an SE(3) pose from world position and ZYX Euler angles.
    
    Args:
        x (float): World-frame x coordinate.
        y (float): World-frame y coordinate.
        z (float): World-frame z coordinate.
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.
    
    Returns:
        NDArray[np.float64]: Homogeneous transform with shape `(4, 4)`.
    '''

    pose = np.eye(4)
    pose[:3, :3] = euler_zyx_to_rotation(roll, pitch, yaw)
    pose[:3, 3] = np.array([x, y, z], dtype=float)
    return pose


def _as_vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    '''Validate a finite three-dimensional trajectory output.
    
    Args:
        value (ArrayLike): Value to validate, differentiate, or convert.
        name (str): Name used in validation errors.
    
    Returns:
        NDArray[np.float64]: Validated vector with shape `(3,)`.
    
    Raises:
        ValueError: The value is not a finite vector with shape `(3,)`.
    '''
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must return a finite vector with shape (3,)")
    return vector

##################################################
# Continuously queryable trajectory
##################################################
@dataclass
class AnalyticTrajectory:
    '''Represent a continuously queryable analytic body trajectory.
    
    Attributes:
        start_time: First valid trajectory time.
        end_time: Last valid trajectory time.
        mode: Descriptive motion mode.
        position_function: World-position function.
        euler_function: ZYX Euler-angle function.
        velocity_function: World-velocity function.
        acceleration_function: World-acceleration function.
        yaw_rate_function: Optional analytic yaw-rate function.
    '''

    start_time: float
    end_time: float
    mode: str
    position_function: VectorFunction
    euler_function: VectorFunction
    velocity_function: VectorFunction
    acceleration_function: VectorFunction
    yaw_rate_function: ScalarFunction | None = None

    def __post_init__(self) -> None:
        '''Validate trajectory bounds and analytic coordinate functions.
        
        Raises:
            ValueError: Trajectory bounds or analytic outputs are invalid.
        '''
        self.start_time = float(self.start_time)
        self.end_time = float(self.end_time)
        if not np.isfinite([self.start_time, self.end_time]).all() or self.end_time <= self.start_time:
            raise ValueError("trajectory times must be finite and strictly increasing")

        # Validate the analytic formulas once, early, so later sensor loops only
        # have to query already-checked trajectory coordinates.
        validation_time = 0.5 * (self.start_time + self.end_time)
        _as_vector3(self.position_function(validation_time), "position_function")
        _as_vector3(self.euler_function(validation_time), "euler_function")
        _as_vector3(self.velocity_function(validation_time), "velocity_function")
        _as_vector3(self.acceleration_function(validation_time), "acceleration_function")

    def _bounded_time(self, time_seconds: float) -> float:
        '''Clamp a query time to the trajectory interval.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            float: Clamped query time.
        '''
        return float(np.clip(float(time_seconds), self.start_time, self.end_time))

    def pose_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return the body pose at one time.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: Transform `T_W_B`, shape `(4, 4)`.
        '''

        bounded_time = self._bounded_time(time_seconds)
        position_xyz = self.position_at(bounded_time)
        euler_rpy = self.euler_at(bounded_time)
        return pose_from_xyz_rpy(
            float(position_xyz[0]),
            float(position_xyz[1]),
            float(position_xyz[2]),
            float(euler_rpy[0]),
            float(euler_rpy[1]),
            float(euler_rpy[2]),
        )

    def poses_at(self, query_times: ArrayLike) -> NDArray[np.float64]:
        '''Return body poses at multiple times.
        
        Args:
            query_times (ArrayLike): Trajectory query times.
        
        Returns:
            NDArray[np.float64]: Transforms with shape `(N, 4, 4)`.
        '''

        query_time_array = np.asarray(query_times, dtype=float)
        return np.stack([self.pose_at(float(time_seconds)) for time_seconds in query_time_array], axis=0)

    def position_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return world position at one time.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World position with shape `(3,)`.
        '''

        bounded_time = self._bounded_time(time_seconds)
        return _as_vector3(self.position_function(bounded_time), "position_function")

    def euler_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return ZYX Euler angles at one time.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: ZYX Euler angles with shape `(3,)`.
        '''

        bounded_time = self._bounded_time(time_seconds)
        return _as_vector3(self.euler_function(bounded_time), "euler_function")

    def velocity_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return world velocity at one time.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World velocity with shape `(3,)`.
        '''

        bounded_time = self._bounded_time(time_seconds)
        return _as_vector3(self.velocity_function(bounded_time), "velocity_function")

    def acceleration_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return world acceleration at one time.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            NDArray[np.float64]: World acceleration with shape `(3,)`.
        '''

        bounded_time = self._bounded_time(time_seconds)
        return _as_vector3(self.acceleration_function(bounded_time), "acceleration_function")

    def yaw_at(self, time_seconds: float) -> float:
        '''Return yaw at one time.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            float: Yaw angle in radians.
        '''

        return float(self.euler_at(time_seconds)[2])

    def yaw_rate_at(self, time_seconds: float) -> float:
        '''Return analytic or finite-difference yaw rate.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
        
        Returns:
            float: Yaw rate in radians per second.
        '''

        bounded_time = self._bounded_time(time_seconds)
        if self.yaw_rate_function is not None:
            return float(self.yaw_rate_function(bounded_time))

        finite_difference_step = 1e-4
        earlier_time = max(self.start_time, bounded_time - finite_difference_step)
        later_time = min(self.end_time, bounded_time + finite_difference_step)
        if later_time <= earlier_time:
            return 0.0
        return float((self.yaw_at(later_time) - self.yaw_at(earlier_time)) / (later_time - earlier_time))

    def angular_velocity_body_at(
        self,
        time_seconds: float,
        finite_difference_step: float = 1e-4,
    ) -> NDArray[np.float64]:
        '''Return body-frame angular velocity.
        
        Args:
            time_seconds (float): Trajectory query time in seconds.
            finite_difference_step (float): Time step used by the angular-velocity approximation.
        
        Returns:
            NDArray[np.float64]: Body angular velocity with shape `(3,)`.
        
        Notes:
            Planar trajectories use the yaw-rate function; general 3D trajectories use an SO(3) finite difference.
        '''

        bounded_time = self._bounded_time(time_seconds)
        euler_rpy = self.euler_at(bounded_time)
        if abs(euler_rpy[0]) < 1e-12 and abs(euler_rpy[1]) < 1e-12:
            return np.array([0.0, 0.0, self.yaw_rate_at(bounded_time)], dtype=float)

        earlier_time = max(self.start_time, bounded_time - finite_difference_step)
        later_time = min(self.end_time, bounded_time + finite_difference_step)
        if later_time <= earlier_time:
            return np.zeros(3)
        earlier_rotation = self.pose_at(earlier_time)[:3, :3]
        later_rotation = self.pose_at(later_time)[:3, :3]
        return so3_log(earlier_rotation.T @ later_rotation) / (later_time - earlier_time)

    def sample(
        self,
        num: int = 400,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        '''Sample trajectory times, positions, and Euler angles.
        
        Args:
            num (int): Number of uniformly spaced samples to return.
        
        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: Tuple `(sample_times, positions, euler_angles)`.
        '''

        sample_times = np.linspace(self.start_time, self.end_time, num)
        sampled_positions = np.vstack([self.position_at(float(time_seconds)) for time_seconds in sample_times])
        sampled_euler_angles = np.vstack([self.euler_at(float(time_seconds)) for time_seconds in sample_times])
        return sample_times, sampled_positions, sampled_euler_angles