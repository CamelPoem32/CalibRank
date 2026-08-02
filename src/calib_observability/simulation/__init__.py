"""Simulation helpers for calibration observability experiments."""

# Re-export the public simulation API used by tests and notebooks.
from .dataset import CalibrationSimulationDataset, ReframedTrajectory, reframe_dataset_to_fixed_extrinsic
from .planar_rover import PlanarRoverConfig, create_trajectory, simulate_planar_rover
from .sensors import ImuData, LidarOdometryData, generate_imu_data, generate_lidar_odometry
from .trajectory import AnalyticTrajectory

__all__ = [
    "AnalyticTrajectory",
    "PlanarRoverConfig",
    "create_trajectory",
    "simulate_planar_rover",
    "ImuData",
    "LidarOdometryData",
    "generate_imu_data",
    "generate_lidar_odometry",
    "CalibrationSimulationDataset",
    "ReframedTrajectory",
    "reframe_dataset_to_fixed_extrinsic",
]
