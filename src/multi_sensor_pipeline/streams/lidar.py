"""LiDAR/radar pose and odometry streams for the modular calibration pipeline.

This file owns conversion of LiDAR odometry and already-body-frame pose observations into existing mrob factors. It deliberately does not own shared calibration-variable creation, rolling state, or any future graph visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:
    import data_processing
except ImportError:  # pragma: no cover
    from src import data_processing

from ..sensors import Sensor, ensure_sensor
from ..variables import VariableKey, VariableRequirement, VariableType
from .base import MeasurementStream, StreamContext, normalize_indices, validate_timestamps


def _validate_pose_stack(poses: Any, timestamps: np.ndarray, name: str) -> list[np.ndarray]:
    """Validate a pose measurement stack using the repository's existing SE(3) validator."""

    pose_list = [data_processing._as_pose_matrix(pose) for pose in poses]
    if len(pose_list) != timestamps.size:
        raise ValueError(f"{name} must contain one pose per timestamp")
    return pose_list


@dataclass
class LidarOdometryStream(MeasurementStream):
    """LiDAR relative-motion calibration stream using ``add_factor_lidar_calib_odometry``."""

    sensor: Sensor | str
    timestamps: Sequence[float]
    odometry_poses: Sequence[Any]
    samples_per_factor: int | None = 32
    time_offset_margin: float = 0.25
    information: Any = 1.0
    factor_stride: int = 1
    stream_name: str | None = None
    stream_type: str = field(default="lidar_odometry", init=False)

    def __post_init__(self) -> None:
        self.sensor = ensure_sensor(self.sensor, default_kind="lidar")
        self.stream_name = f"{self.sensor.sensor_id}.odometry" if self.stream_name is None else str(self.stream_name)
        self.timestamps = validate_timestamps(self.timestamps, f"{self.stream_name}.timestamps")
        self.odometry_poses = _validate_pose_stack(self.odometry_poses, self.timestamps, f"{self.stream_name}.odometry_poses")
        self.time_offset_margin = float(self.time_offset_margin)
        self.factor_stride = int(self.factor_stride)
        if self.samples_per_factor is not None and int(self.samples_per_factor) < 2:
            raise ValueError(f"{self.stream_name}.samples_per_factor must be at least 2 or None")
        if self.time_offset_margin < 0.0:
            raise ValueError(f"{self.stream_name}.time_offset_margin must be nonnegative")
        if self.factor_stride < 1:
            raise ValueError(f"{self.stream_name}.factor_stride must be positive")

    @property
    def extrinsic_key(self) -> VariableKey:
        return VariableKey(self.sensor.sensor_id, VariableType.EXTRINSIC)

    @property
    def time_offset_key(self) -> VariableKey:
        return VariableKey(self.sensor.sensor_id, VariableType.TIME_OFFSET)

    def required_variables(self) -> tuple[VariableRequirement, ...]:
        return (
            VariableRequirement(self.extrinsic_key, self.stream_name, "LiDAR odometry factor maps LiDAR relative motion into the body frame"),
            VariableRequirement(self.time_offset_key, self.stream_name, "LiDAR odometry factor queries odometry at body time plus tau_L"),
        )

    def valid_time_interval(self) -> tuple[float, float]:
        return float(self.timestamps[0] + self.time_offset_margin), float(self.timestamps[-1] - self.time_offset_margin)

    def trajectory_lidar_data(self) -> dict[str, Any]:
        return {"sensor_id": self.sensor.sensor_id, "lidar_timestamps": self.timestamps, "lidar_odometry_poses": self.odometry_poses}

    def numerical_lidar_data(self) -> dict[str, Any]:
        return {"sensor_id": self.sensor.sensor_id, "lidar_timestamps": self.timestamps, "lidar_odometry_poses": self.odometry_poses}

    def add_factors(self, context: StreamContext) -> None:
        T_B_L_node = context.node_for(self.extrinsic_key)
        tau_L_node = context.node_for(self.time_offset_key)

        # Add LiDAR relative-pose factors between selected consecutive trajectory poses using fixed interpolation support.
        for factor_index, pose_index in enumerate(range(0, len(context.pose_nodes) - 1, self.factor_stride)):
            target_index = pose_index + self.factor_stride
            if target_index >= len(context.pose_nodes):
                break

            pose_time_origin = float(context.pose_timestamps[pose_index])
            pose_time_target = float(context.pose_timestamps[target_index])
            support_indices = data_processing.select_time_support_indices(self.timestamps, pose_time_origin - self.time_offset_margin, pose_time_target + self.time_offset_margin, self.samples_per_factor)
            timestamps = self.timestamps[support_indices]
            measurements = [self.odometry_poses[index] for index in support_indices]
            information = data_processing._information_matrix(self.information, 6, factor_index)

            factor_id = context.graph.add_factor_lidar_calib_odometry(pose_time_origin, pose_time_target, timestamps, measurements, context.pose_nodes[pose_index], context.pose_nodes[target_index], T_B_L_node, tau_L_node, information)
            context.record_factor(
                factor_id=factor_id,
                factor_type="lidar_calib_odometry",
                stream_name=self.stream_name,
                sensor_id=self.sensor.sensor_id,
                node_ids=(context.pose_nodes[pose_index], context.pose_nodes[target_index], T_B_L_node, tau_L_node),
                pose_indices=(pose_index, target_index),
                measurement_indices=normalize_indices(support_indices),
                variable_keys=(self.extrinsic_key, self.time_offset_key),
            )


@dataclass
class PoseObservationStream(MeasurementStream):
    """Absolute pose observation stream using ``add_factor_1pose_3d`` on trajectory nodes.

    The supplied poses are assumed to already be observations of the body trajectory ``T_W_B`` or to have been converted into that frame upstream. This class therefore adds one-pose factors only; it does not estimate a sensor extrinsic.
    """

    sensor: Sensor | str
    timestamps: Sequence[float]
    poses: Sequence[Any]
    information: Any = 1.0
    factor_stride: int = 1
    stream_name: str | None = None
    stream_type: str = field(default="pose_observation", init=False)

    def __post_init__(self) -> None:
        self.sensor = ensure_sensor(self.sensor)
        self.stream_name = f"{self.sensor.sensor_id}.pose" if self.stream_name is None else str(self.stream_name)
        self.timestamps = validate_timestamps(self.timestamps, f"{self.stream_name}.timestamps")
        self.poses = _validate_pose_stack(self.poses, self.timestamps, f"{self.stream_name}.poses")
        self.factor_stride = int(self.factor_stride)
        if self.factor_stride < 1:
            raise ValueError(f"{self.stream_name}.factor_stride must be positive")

    def valid_time_interval(self) -> tuple[float, float]:
        return float(self.timestamps[0]), float(self.timestamps[-1])

    def add_factors(self, context: StreamContext) -> None:
        # Add absolute pose observations at selected trajectory timestamps. Pose interpolation reuses the repository SE(3) interpolation helper.
        for factor_index, pose_index in enumerate(range(0, len(context.pose_nodes), self.factor_stride)):
            pose_time = float(context.pose_timestamps[pose_index])
            if pose_time < self.timestamps[0] or pose_time > self.timestamps[-1]:
                raise IndexError(f"{self.stream_name}: pose time {pose_time} lies outside observation support [{self.timestamps[0]}, {self.timestamps[-1]}]")
            observed_pose = data_processing._interpolate_pose(self.timestamps, self.poses, pose_time)
            information = data_processing._information_matrix(self.information, 6, factor_index)
            factor_id = context.graph.add_factor_1pose_3d(data_processing._as_mrob_se3(observed_pose), context.pose_nodes[pose_index], information)
            context.record_factor(
                factor_id=factor_id,
                factor_type="pose_observation",
                stream_name=self.stream_name,
                sensor_id=self.sensor.sensor_id,
                node_ids=(context.pose_nodes[pose_index],),
                pose_indices=(pose_index,),
                details={"observation_time": pose_time},
            )


class LidarPoseStream(PoseObservationStream):
    """LiDAR pose observations already expressed as body poses."""


class RadarPoseStream(PoseObservationStream):
    """Radar pose observations already expressed as body poses."""

