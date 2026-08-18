"""Measurement-stream base protocol and stream construction context.

This file defines the small interface between ``RollingGraph`` and factor-producing streams. It deliberately does not implement any sensor mathematics; concrete stream modules own the mrob factor calls and measurement sampling rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..metadata import FactorMetadata, GraphMetadata
from ..variables import VariableKey, VariableRequirement


def validate_timestamps(timestamps: Sequence[float], name: str) -> np.ndarray:
    """Validate one strictly increasing timestamp vector."""

    values = np.asarray(timestamps, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError(f"{name} must contain at least two timestamps")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return values.copy()


def validate_vector_measurements(values: Any, timestamps: np.ndarray, name: str) -> np.ndarray:
    """Validate one ``Nx3`` vector measurement stream."""

    array = np.asarray(values, dtype=float)
    if array.shape != (timestamps.size, 3):
        raise ValueError(f"{name} must have shape ({timestamps.size}, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def normalize_indices(indices: Sequence[int]) -> tuple[int, ...]:
    """Convert numpy integer arrays into stable metadata tuples."""

    return tuple(int(index) for index in np.asarray(indices, dtype=np.int64).reshape(-1))


@dataclass
class StreamContext:
    """Read-only graph-construction context passed from ``RollingGraph`` to streams."""

    graph: Any
    pose_timestamps: np.ndarray
    pose_nodes: Sequence[int]
    variable_nodes: dict[VariableKey, int]
    metadata: GraphMetadata

    def node_for(self, key: VariableKey) -> int:
        """Return the shared mrob node id for one calibration variable."""

        if key not in self.variable_nodes:
            raise KeyError(f"Calibration variable {key.label} has not been created")
        return self.variable_nodes[key]

    def record_factor(
        self,
        *,
        factor_id: int | None,
        factor_type: str,
        stream_name: str,
        sensor_id: str | None,
        node_ids: Sequence[int],
        pose_indices: Sequence[int] = (),
        measurement_indices: Sequence[int] = (),
        variable_keys: Sequence[VariableKey] = (),
        accepted: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record structured metadata beside the opaque mrob factor."""

        self.metadata.factors.append(
            FactorMetadata(
                factor_id=None if factor_id is None else int(factor_id),
                factor_type=str(factor_type),
                stream_name=str(stream_name),
                sensor_id=None if sensor_id is None else str(sensor_id),
                node_ids=tuple(int(node_id) for node_id in node_ids),
                pose_indices=tuple(int(index) for index in pose_indices),
                measurement_indices=normalize_indices(measurement_indices),
                variable_keys=tuple(variable_keys),
                accepted=bool(accepted),
                details={} if details is None else dict(details),
            )
        )


class MeasurementStream:
    """Small interface implemented by all factor-producing measurement streams."""

    sensor: Any
    stream_name: str
    stream_type: str

    def required_variables(self) -> tuple[VariableRequirement, ...]:
        """Return calibration variables this stream needs to build its factors."""

        return ()

    def add_factors(self, context: StreamContext) -> None:
        """Add this stream's factors to the mrob graph."""

        raise NotImplementedError

    def valid_time_interval(self) -> tuple[float, float] | None:
        """Return the safe body-time interval implied by fixed interpolation support, if any."""

        return None

    def trajectory_imu_data(self) -> dict[str, Any] | None:
        """Return IMU data suitable for trajectory initialization, when this stream provides it."""

        return None

    def trajectory_lidar_data(self) -> dict[str, Any] | None:
        """Return LiDAR odometry data suitable for trajectory initialization, when this stream provides it."""

        return None

    def numerical_imu_data(self, sensor_id: str) -> dict[str, Any] | None:
        """Return IMU angular velocity data for numerical calibration of ``sensor_id``, when available."""

        return None

    def numerical_lidar_data(self) -> dict[str, Any] | None:
        """Return LiDAR/body angular velocity source data for numerical calibration, when available."""

        return None

