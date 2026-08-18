"""IMU measurement streams for gyro and accelerometer calibration factors.

This file owns the translation from timestamped IMU measurements into existing mrob gyro/accelerometer calibration factors. It deliberately does not own physical sensor identity beyond referencing ``Sensor``, shared calibration-variable creation, rolling state, or solver orchestration.
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
from .base import MeasurementStream, StreamContext, normalize_indices, validate_timestamps, validate_vector_measurements


def _stream_name(default: str, sensor: Sensor, stream_name: str | None) -> str:
    return f"{sensor.sensor_id}.{default}" if stream_name is None else str(stream_name)


@dataclass
class GyroStream(MeasurementStream):
    """Gyroscope propagation stream using ``add_factor_gyro_calib_prop``."""

    sensor: Sensor | str
    timestamps: Sequence[float]
    angular_velocity: Any
    samples_per_factor: int | None = 64
    time_offset_margin: float = 0.25
    information: Any = 1.0
    factor_stride: int = 1
    stream_name: str | None = None
    stream_type: str = field(default="gyro", init=False)

    def __post_init__(self) -> None:
        self.sensor = ensure_sensor(self.sensor, default_kind="imu")
        self.stream_name = _stream_name("gyro", self.sensor, self.stream_name)
        self.timestamps = validate_timestamps(self.timestamps, f"{self.stream_name}.timestamps")
        self.angular_velocity = validate_vector_measurements(self.angular_velocity, self.timestamps, f"{self.stream_name}.angular_velocity")
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

    @property
    def bias_key(self) -> VariableKey:
        return VariableKey(self.sensor.sensor_id, VariableType.GYRO_BIAS)

    def required_variables(self) -> tuple[VariableRequirement, ...]:
        return (
            VariableRequirement(self.extrinsic_key, self.stream_name, "gyro propagation rotates IMU angular velocity into the body frame"),
            VariableRequirement(self.time_offset_key, self.stream_name, "gyro propagation queries IMU samples at body time plus tau_I"),
            VariableRequirement(self.bias_key, self.stream_name, "gyro propagation subtracts a constant gyroscope bias"),
        )

    def valid_time_interval(self) -> tuple[float, float]:
        return float(self.timestamps[0] + self.time_offset_margin), float(self.timestamps[-1] - self.time_offset_margin)

    def trajectory_imu_data(self) -> dict[str, Any]:
        return {"sensor_id": self.sensor.sensor_id, "imu_timestamps": self.timestamps, "angular_velocity_imu": self.angular_velocity}

    def numerical_imu_data(self, sensor_id: str) -> dict[str, Any] | None:
        if str(sensor_id) != self.sensor.sensor_id:
            return None
        return {"imu_timestamps": self.timestamps, "angular_velocity_imu": self.angular_velocity}

    def add_factors(self, context: StreamContext) -> None:
        T_B_I_node = context.node_for(self.extrinsic_key)
        tau_I_node = context.node_for(self.time_offset_key)
        bias_node = context.node_for(self.bias_key)

        # Add gyroscope propagation factors between selected consecutive trajectory poses using fixed measurement support around the nominal pose times.
        for factor_index, pose_index in enumerate(range(0, len(context.pose_nodes) - 1, self.factor_stride)):
            target_index = pose_index + self.factor_stride
            if target_index >= len(context.pose_nodes):
                break

            pose_time_origin = float(context.pose_timestamps[pose_index])
            pose_time_target = float(context.pose_timestamps[target_index])
            support_indices = data_processing.select_time_support_indices(self.timestamps, pose_time_origin - self.time_offset_margin, pose_time_target + self.time_offset_margin, self.samples_per_factor)
            timestamps = self.timestamps[support_indices]
            measurements = self.angular_velocity[support_indices]
            information = data_processing._information_matrix(self.information, 3, factor_index)

            factor_id = context.graph.add_factor_gyro_calib_prop(pose_time_origin, pose_time_target, timestamps, measurements, context.pose_nodes[pose_index], context.pose_nodes[target_index], T_B_I_node, bias_node, tau_I_node, information)
            context.record_factor(
                factor_id=factor_id,
                factor_type="gyro_calib_prop",
                stream_name=self.stream_name,
                sensor_id=self.sensor.sensor_id,
                node_ids=(context.pose_nodes[pose_index], context.pose_nodes[target_index], T_B_I_node, bias_node, tau_I_node),
                pose_indices=(pose_index, target_index),
                measurement_indices=normalize_indices(support_indices),
                variable_keys=(self.extrinsic_key, self.bias_key, self.time_offset_key),
            )


@dataclass
class SimpleAccelStream(MeasurementStream):
    """Simple gravity-alignment accelerometer stream using ``add_factor_accel_gravity_calib``."""

    sensor: Sensor | str
    timestamps: Sequence[float]
    acceleration: Any
    samples_per_factor: int | None = 64
    time_offset_margin: float = 0.25
    gravity_world: Sequence[float] = (0.0, 0.0, -9.81)
    information: Any = 1.0
    factor_stride: int = 1
    accel_norm_tolerance: float | None = None
    accel_gyro_threshold: float | None = None
    angular_velocity_for_gating: Any | None = None
    stream_name: str | None = None
    stream_type: str = field(default="simple_accel", init=False)

    def __post_init__(self) -> None:
        self.sensor = ensure_sensor(self.sensor, default_kind="imu")
        self.stream_name = _stream_name("accel", self.sensor, self.stream_name)
        self.timestamps = validate_timestamps(self.timestamps, f"{self.stream_name}.timestamps")
        self.acceleration = validate_vector_measurements(self.acceleration, self.timestamps, f"{self.stream_name}.acceleration")
        self.gravity_world = data_processing._as_vector3(self.gravity_world, f"{self.stream_name}.gravity_world")
        self.time_offset_margin = float(self.time_offset_margin)
        self.factor_stride = int(self.factor_stride)
        if self.samples_per_factor is not None and int(self.samples_per_factor) < 2:
            raise ValueError(f"{self.stream_name}.samples_per_factor must be at least 2 or None")
        if self.time_offset_margin < 0.0:
            raise ValueError(f"{self.stream_name}.time_offset_margin must be nonnegative")
        if self.factor_stride < 1:
            raise ValueError(f"{self.stream_name}.factor_stride must be positive")
        if self.angular_velocity_for_gating is not None:
            self.angular_velocity_for_gating = validate_vector_measurements(self.angular_velocity_for_gating, self.timestamps, f"{self.stream_name}.angular_velocity_for_gating")
        if self.accel_gyro_threshold is not None and self.angular_velocity_for_gating is None:
            raise ValueError(f"{self.stream_name}.accel_gyro_threshold requires angular_velocity_for_gating")

    @property
    def extrinsic_key(self) -> VariableKey:
        return VariableKey(self.sensor.sensor_id, VariableType.EXTRINSIC)

    @property
    def time_offset_key(self) -> VariableKey:
        return VariableKey(self.sensor.sensor_id, VariableType.TIME_OFFSET)

    def required_variables(self) -> tuple[VariableRequirement, ...]:
        return (
            VariableRequirement(self.extrinsic_key, self.stream_name, "accelerometer gravity factor uses the IMU-to-body rotation"),
            VariableRequirement(self.time_offset_key, self.stream_name, "accelerometer gravity factor queries acceleration at body time plus tau_I"),
        )

    def valid_time_interval(self) -> tuple[float, float]:
        return float(self.timestamps[0] + self.time_offset_margin), float(self.timestamps[-1] - self.time_offset_margin)

    def _measurement_accepted(self, pose_time: float, tau_I_initial: float) -> tuple[bool, str]:
        query_time = float(pose_time) + float(tau_I_initial)
        specific_force = data_processing._interpolate_vector(self.timestamps, self.acceleration, query_time)

        # Optional stationary-ish gating preserves the old gravity-norm and gyro-threshold behavior.
        if self.accel_norm_tolerance is not None:
            magnitude_error = abs(np.linalg.norm(specific_force) - np.linalg.norm(self.gravity_world))
            if magnitude_error > float(self.accel_norm_tolerance):
                return False, f"gravity magnitude error {magnitude_error}"

        if self.accel_gyro_threshold is not None:
            angular_velocity = data_processing._interpolate_vector(self.timestamps, self.angular_velocity_for_gating, query_time)
            if np.linalg.norm(angular_velocity) > float(self.accel_gyro_threshold):
                return False, f"angular velocity norm {np.linalg.norm(angular_velocity)}"

        return True, ""

    def add_factors(self, context: StreamContext) -> None:
        T_B_I_node = context.node_for(self.extrinsic_key)
        tau_I_node = context.node_for(self.time_offset_key)
        tau_initial = float(np.asarray(context.graph.get_estimated_state()[tau_I_node], dtype=float).reshape(-1)[0])

        # Add one gravity-alignment factor per selected trajectory pose, retaining the existing sign and gravity convention.
        for factor_index, pose_index in enumerate(range(0, len(context.pose_nodes), self.factor_stride)):
            pose_time = float(context.pose_timestamps[pose_index])
            accepted, rejection_reason = self._measurement_accepted(pose_time, tau_initial)
            if not accepted:
                context.record_factor(
                    factor_id=None,
                    factor_type="accel_gravity_calib",
                    stream_name=self.stream_name,
                    sensor_id=self.sensor.sensor_id,
                    node_ids=(context.pose_nodes[pose_index], T_B_I_node, tau_I_node),
                    pose_indices=(pose_index,),
                    variable_keys=(self.extrinsic_key, self.time_offset_key),
                    accepted=False,
                    details={"mode": "simple", "reason": rejection_reason},
                )
                continue

            support_indices = data_processing.select_time_support_indices(self.timestamps, pose_time - self.time_offset_margin, pose_time + self.time_offset_margin, self.samples_per_factor)
            timestamps = self.timestamps[support_indices]
            measurements = self.acceleration[support_indices]
            information = data_processing._information_matrix(self.information, 3, factor_index)

            factor_id = context.graph.add_factor_accel_gravity_calib(pose_time, timestamps, measurements, self.gravity_world, context.pose_nodes[pose_index], T_B_I_node, tau_I_node, information)
            context.record_factor(
                factor_id=factor_id,
                factor_type="accel_gravity_calib",
                stream_name=self.stream_name,
                sensor_id=self.sensor.sensor_id,
                node_ids=(context.pose_nodes[pose_index], T_B_I_node, tau_I_node),
                pose_indices=(pose_index,),
                measurement_indices=normalize_indices(support_indices),
                variable_keys=(self.extrinsic_key, self.time_offset_key),
                details={"mode": "simple"},
            )


@dataclass
class ComplexAccelStream(MeasurementStream):
    """Dynamic accelerometer stream using ``add_factor_accel_lever_arm_calib``."""

    sensor: Sensor | str
    timestamps: Sequence[float]
    acceleration: Any
    angular_velocity: Any
    samples_per_factor: int | None = 64
    time_offset_margin: float = 0.25
    gravity_world: Sequence[float] = (0.0, 0.0, -9.81)
    information: Any = 1.0
    factor_stride: int = 1
    stream_name: str | None = None
    stream_type: str = field(default="complex_accel", init=False)

    def __post_init__(self) -> None:
        self.sensor = ensure_sensor(self.sensor, default_kind="imu")
        self.stream_name = _stream_name("complex_accel", self.sensor, self.stream_name)
        self.timestamps = validate_timestamps(self.timestamps, f"{self.stream_name}.timestamps")
        self.acceleration = validate_vector_measurements(self.acceleration, self.timestamps, f"{self.stream_name}.acceleration")
        self.angular_velocity = validate_vector_measurements(self.angular_velocity, self.timestamps, f"{self.stream_name}.angular_velocity")
        self.gravity_world = data_processing._as_vector3(self.gravity_world, f"{self.stream_name}.gravity_world")
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

    @property
    def bias_key(self) -> VariableKey:
        return VariableKey(self.sensor.sensor_id, VariableType.GYRO_BIAS)

    def required_variables(self) -> tuple[VariableRequirement, ...]:
        return (
            VariableRequirement(self.extrinsic_key, self.stream_name, "lever-arm accelerometer factor uses the IMU-to-body transform"),
            VariableRequirement(self.time_offset_key, self.stream_name, "lever-arm accelerometer factor queries IMU samples at body time plus tau_I"),
            VariableRequirement(self.bias_key, self.stream_name, "lever-arm accelerometer factor uses bias-corrected gyroscope samples"),
        )

    def valid_time_interval(self) -> tuple[float, float]:
        return float(self.timestamps[0] + self.time_offset_margin), float(self.timestamps[-1] - self.time_offset_margin)

    def add_factors(self, context: StreamContext) -> None:
        if len(context.pose_nodes) < 3:
            raise ValueError(f"{self.stream_name} requires at least three trajectory poses")

        T_B_I_node = context.node_for(self.extrinsic_key)
        tau_I_node = context.node_for(self.time_offset_key)
        bias_node = context.node_for(self.bias_key)

        # Add dynamic accelerometer factors centered on each selected interior trajectory pose.
        for factor_index, pose_index in enumerate(range(1, len(context.pose_nodes) - 1, self.factor_stride)):
            pose_time_previous = float(context.pose_timestamps[pose_index - 1])
            pose_time = float(context.pose_timestamps[pose_index])
            pose_time_next = float(context.pose_timestamps[pose_index + 1])
            support_indices = data_processing.select_time_support_indices(self.timestamps, pose_time - self.time_offset_margin, pose_time + self.time_offset_margin, self.samples_per_factor)
            timestamps = self.timestamps[support_indices]
            accelerometer_measurements = self.acceleration[support_indices]
            gyroscope_measurements = self.angular_velocity[support_indices]
            information = data_processing._information_matrix(self.information, 3, factor_index)

            factor_id = context.graph.add_factor_accel_lever_arm_calib(pose_time_previous, pose_time, pose_time_next, timestamps, accelerometer_measurements, gyroscope_measurements, self.gravity_world, context.pose_nodes[pose_index - 1], context.pose_nodes[pose_index], context.pose_nodes[pose_index + 1], T_B_I_node, bias_node, tau_I_node, information)
            context.record_factor(
                factor_id=factor_id,
                factor_type="accel_lever_arm_calib",
                stream_name=self.stream_name,
                sensor_id=self.sensor.sensor_id,
                node_ids=(context.pose_nodes[pose_index - 1], context.pose_nodes[pose_index], context.pose_nodes[pose_index + 1], T_B_I_node, bias_node, tau_I_node),
                pose_indices=(pose_index - 1, pose_index, pose_index + 1),
                measurement_indices=normalize_indices(support_indices),
                variable_keys=(self.extrinsic_key, self.bias_key, self.time_offset_key),
                details={"mode": "complex"},
            )


AccelStream = SimpleAccelStream

