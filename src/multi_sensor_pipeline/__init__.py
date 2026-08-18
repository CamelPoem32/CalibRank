"""Modular rolling multi-sensor calibration pipeline.

The package separates physical sensors, measurement streams, shared calibration variables, rolling graph orchestration, priors, initialization sources, and rolling state. It intentionally leaves the existing monolithic calibration implementation in place while providing a composable path for future graph visualization.
"""

from .rolling_graph import RollingGraph, SolverConfig, TrajectoryConfig, WindowResult
from .rolling_state import RollingState
from .sensors import Sensor
from .streams import AccelStream, ComplexAccelStream, GyroStream, LidarOdometryStream, LidarPoseStream, PoseObservationStream, RadarPoseStream, SimpleAccelStream
from .variables import ValueSource, VariableConfig, VariableKey, VariableType
from .plotting import plot_calibration_estimates, plot_rolling_trajectory, plot_stream_measurements, print_rolling_result_summary

__all__ = [
    "Sensor",
    "VariableKey",
    "VariableType",
    "VariableConfig",
    "ValueSource",
    "RollingState",
    "RollingGraph",
    "SolverConfig",
    "TrajectoryConfig",
    "WindowResult",
    "GyroStream",
    "AccelStream",
    "SimpleAccelStream",
    "ComplexAccelStream",
    "LidarOdometryStream",
    "PoseObservationStream",
    "LidarPoseStream",
    "RadarPoseStream",
    "plot_stream_measurements",
    "plot_rolling_trajectory",
    "plot_calibration_estimates",
    "print_rolling_result_summary",
]
