"""LiDAR/radar pose and odometry streams for the modular calibration pipeline.

This file owns conversion of LiDAR odometry, generic already-body-frame pose observations, and absolute calibrated LiDAR sensor-pose observations into existing mrob factors. It deliberately does not own shared calibration-variable creation, rolling state, trajectory initialization, or graph visualization.

The absolute LiDAR pose convention is:

    T_A_B maps coordinates from frame B to frame A.

For LidarPoseStream:

    trajectory node: T_O_B
    extrinsic node:  T_B_L
    measurement:     T_O_L

where O is the world frame used internally by the graph. O may be the original map/world frame or a translated local graph world frame.

The predicted sensor pose is:

    T_O_L_predicted = T_O_B @ T_B_L

and the temporal offset follows the same convention as FactorLidarCalibOdometry:

    sensor_query_time = pose_time + tau_L.
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
    time_offset_margin: float = 1.0
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
        return (VariableRequirement(self.extrinsic_key, self.stream_name, "LiDAR odometry factor maps LiDAR relative motion into the body frame"), 
                VariableRequirement(self.time_offset_key, self.stream_name, "LiDAR odometry factor queries odometry at body time plus tau_L"))

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

            context.record_factor(factor_id=factor_id, factor_type="lidar_calib_odometry", stream_name=self.stream_name, sensor_id=self.sensor.sensor_id, node_ids=(context.pose_nodes[pose_index], context.pose_nodes[target_index], T_B_L_node, tau_L_node), pose_indices=(pose_index, target_index), measurement_indices=normalize_indices(support_indices), variable_keys=(self.extrinsic_key, self.time_offset_key))


@dataclass
class PoseObservationStream(MeasurementStream):
    """Absolute BODY-pose observation stream using ``add_factor_1pose_3d``.

    The supplied poses must already be observations of the trajectory body pose T_W_B, or T_O_B when the graph uses a local world frame.

    This generic stream does not contain a sensor-extrinsic model and therefore does not estimate an extrinsic or temporal offset.
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
        T_B_L_node = context.node_for(self.extrinsic_key)
        tau_L_node = context.node_for(self.time_offset_key)

        # Add calibrated absolute LiDAR sensor-pose observations at selected trajectory timestamps. The C++ factor interpolates T_O_L at pose_time + tau_L, so Python supplies a fixed measurement-support window rather than a pre-interpolated pose.
        for factor_index, pose_index in enumerate(range(0, len(context.pose_nodes), self.factor_stride)):
            pose_time = float(context.pose_timestamps[pose_index])

            if pose_time < self.timestamps[0] + self.time_offset_margin or pose_time > self.timestamps[-1] - self.time_offset_margin:
                raise IndexError(f"{self.stream_name}: pose time {pose_time} lies outside valid observation support [{self.timestamps[0] + self.time_offset_margin}, {self.timestamps[-1] - self.time_offset_margin}] for time_offset_margin={self.time_offset_margin}")

            support_indices = data_processing.select_time_support_indices(self.timestamps, pose_time - self.time_offset_margin, pose_time + self.time_offset_margin, self.samples_per_factor)

            if len(support_indices) < 2:
                raise ValueError(f"{self.stream_name}: fewer than two LiDAR pose measurements cover pose time {pose_time} with time_offset_margin={self.time_offset_margin}")

            timestamps = self.timestamps[support_indices]
            measurements = [self.poses[index] for index in support_indices]
            information = data_processing._information_matrix(self.information, 6, factor_index)

            factor_id = context.graph.add_factor_lidar_calib_pose(pose_time, timestamps, measurements, context.pose_nodes[pose_index], T_B_L_node, tau_L_node, information)

            context.record_factor(factor_id=factor_id, factor_type="lidar_calib_pose", stream_name=self.stream_name, sensor_id=self.sensor.sensor_id, node_ids=(context.pose_nodes[pose_index], T_B_L_node, tau_L_node), pose_indices=(pose_index,), measurement_indices=normalize_indices(support_indices), variable_keys=(self.extrinsic_key, self.time_offset_key), details={"pose_time": pose_time, "measurement_query_convention": "pose_time + tau_L", "measurement_frame": "T_O_L"})

@dataclass
class LidarPoseStream(MeasurementStream):
    """Absolute calibrated LiDAR SENSOR-pose observation stream.

    The supplied poses are LiDAR sensor poses T_O_L, not body poses.

    For trajectory body pose T_O_B and LiDAR extrinsic T_B_L, the predicted LiDAR pose is

        T_O_L_predicted = T_O_B @ T_B_L.

    The LiDAR temporal offset uses the same convention as the existing calibrated LiDAR odometry factor:

        sensor_query_time = pose_time + tau_L.

    Therefore the measurement used by a factor at trajectory time t is the interpolated absolute LiDAR pose

        T_O_L_measured(t + tau_L).

    This stream requires an MROB binding named ``add_factor_lidar_calib_pose`` connecting one trajectory pose node, one extrinsic node, and one scalar time-offset node.
    """

    sensor: Sensor | str
    timestamps: Sequence[float]
    poses: Sequence[Any]
    samples_per_factor: int | None = 32
    time_offset_margin: float = 1.0
    information: Any = 1.0
    factor_stride: int = 1
    stream_name: str | None = None
    stream_type: str = field(default="pose_observation", init=False)

    def __post_init__(self) -> None:
        self.sensor = ensure_sensor(self.sensor, default_kind="lidar")

        self.stream_name = f"{self.sensor.sensor_id}.map_pose" if self.stream_name is None else str(self.stream_name)

        self.timestamps = validate_timestamps(self.timestamps, f"{self.stream_name}.timestamps")
        self.poses = _validate_pose_stack(self.poses, self.timestamps, f"{self.stream_name}.poses")
        self.time_offset_margin = float(self.time_offset_margin)
        self.factor_stride = int(self.factor_stride)

        if self.samples_per_factor is not None:
            self.samples_per_factor = int(self.samples_per_factor)

            if self.samples_per_factor < 2:
                raise ValueError(f"{self.stream_name}.samples_per_factor must be at least 2 or None")

        if self.time_offset_margin <= 0.0:
            raise ValueError(f"{self.stream_name}.time_offset_margin must be positive because the calibrated pose factor interpolates measurements while optimizing tau_L")

        if self.factor_stride < 1:
            raise ValueError(f"{self.stream_name}.factor_stride must be positive")

        valid_start, valid_end = self.valid_time_interval()

        if valid_end <= valid_start:
            raise ValueError(f"{self.stream_name}.time_offset_margin leaves no valid measurement interval")

    @property
    def extrinsic_key(self) -> VariableKey:
        return VariableKey(self.sensor.sensor_id, VariableType.EXTRINSIC)

    @property
    def time_offset_key(self) -> VariableKey:
        return VariableKey(self.sensor.sensor_id, VariableType.TIME_OFFSET)

    def required_variables(self) -> tuple[VariableRequirement, ...]:
        return (VariableRequirement(self.extrinsic_key, self.stream_name, "Absolute LiDAR sensor-pose factor predicts T_O_L = T_O_B @ T_B_L"), 
                VariableRequirement(self.time_offset_key, self.stream_name, "Absolute LiDAR sensor-pose factor queries T_O_L measurements at pose_time + tau_L"))

    def valid_time_interval(self) -> tuple[float, float]:
        return float(self.timestamps[0] + self.time_offset_margin), float(self.timestamps[-1] - self.time_offset_margin)

    def trajectory_lidar_data(self) -> dict[str, Any]:
        # Absolute sensor poses T_O_L(t) are accumulated poses; the trajectory initializer forms relative updates internally.
        return {"sensor_id": self.sensor.sensor_id, "lidar_timestamps": self.timestamps, "lidar_odometry_poses": self.poses}

    def numerical_lidar_data(self) -> dict[str, Any]:
        # Numerical calibration derives local angular velocity from T_O_L(t) with transform.se3_to_angvels, preserving T_A_B convention.
        return {"sensor_id": self.sensor.sensor_id, "lidar_timestamps": self.timestamps, "lidar_odometry_poses": self.poses}

    def add_factors(self, context: StreamContext) -> None:
        add_factor = getattr(context.graph, "add_factor_lidar_calib_pose", None)

        if add_factor is None:
            raise RuntimeError("LidarPoseStream requires mrob.FGraph.add_factor_lidar_calib_pose, but the imported mrob module does not expose that binding. The old add_factor_1pose_3d factor cannot be used here because it would not depend on T_B_L or tau_L.")

        T_B_L_node = context.node_for(self.extrinsic_key)
        tau_L_node = context.node_for(self.time_offset_key)

        # Add calibrated absolute LiDAR-pose factors at selected trajectory timestamps using fixed measurement support for temporal-offset interpolation.
        for factor_index, pose_index in enumerate(range(0, len(context.pose_nodes), self.factor_stride)):
            pose_time = float(context.pose_timestamps[pose_index])

            support_start = pose_time - self.time_offset_margin
            support_end = pose_time + self.time_offset_margin

            support_indices = data_processing.select_time_support_indices(self.timestamps, support_start, support_end, self.samples_per_factor)

            if len(support_indices) < 2:
                raise ValueError(f"{self.stream_name}: fewer than two LiDAR pose measurements cover pose time {pose_time} with time_offset_margin={self.time_offset_margin}")

            timestamps = self.timestamps[support_indices]
            measurements = [self.poses[index] for index in support_indices]
            information = data_processing._information_matrix(self.information, 6, factor_index)

            factor_id = add_factor(pose_time, timestamps, measurements, context.pose_nodes[pose_index], T_B_L_node, tau_L_node, information)

            context.record_factor(factor_id=factor_id, factor_type="lidar_calib_pose", stream_name=self.stream_name, sensor_id=self.sensor.sensor_id, node_ids=(context.pose_nodes[pose_index], T_B_L_node, tau_L_node), pose_indices=(pose_index,), measurement_indices=normalize_indices(support_indices), variable_keys=(self.extrinsic_key, self.time_offset_key), details={"pose_time": pose_time, "measurement_query_convention": "pose_time + tau_L", "measurement_frame": "T_O_L"})


class RadarPoseStream(PoseObservationStream):
    """Radar pose observations already expressed as body poses."""