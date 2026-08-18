"""Physical sensor identities for the modular calibration pipeline.

This file owns only lightweight sensor naming and validation. It deliberately does not own measurement data, factor construction, or optimization variables; streams and the variable registry use these identities to agree on shared calibration nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Sensor:
    """A physical sensor identity that can produce one or more measurement streams."""

    sensor_id: str
    kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sensor_id = str(self.sensor_id).strip()
        if not sensor_id:
            raise ValueError("sensor_id must be a nonempty string")
        object.__setattr__(self, "sensor_id", sensor_id)
        if self.kind is not None:
            object.__setattr__(self, "kind", str(self.kind).strip().lower())


def ensure_sensor(sensor: Sensor | str, *, default_kind: str | None = None) -> Sensor:
    """Return a validated ``Sensor`` from either a ``Sensor`` object or a sensor id string."""

    if isinstance(sensor, Sensor):
        if sensor.kind is None and default_kind is not None:
            return Sensor(sensor.sensor_id, kind=default_kind, metadata=dict(sensor.metadata))
        return sensor
    return Sensor(str(sensor), kind=default_kind)

