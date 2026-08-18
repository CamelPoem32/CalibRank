"""Optimization-variable keys, configurations, and mrob node creation.

This file defines how calibration variables are named, initialized, fixed, and regularized. It deliberately does not decide where numerical calibration values come from; ``RollingGraph`` resolves value sources and then asks this module to create the corresponding mrob nodes and prior factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import mrob
import numpy as np

try:  # Support both ``import multi_sensor_pipeline`` and ``import src.multi_sensor_pipeline`` in notebooks/tests.
    import data_processing
except ImportError:  # pragma: no cover - exercised only when the repository root, rather than src/, is on sys.path.
    from src import data_processing


class VariableType(str, Enum):
    """Calibration variables currently supported by the mrob calibration bindings."""

    EXTRINSIC = "extrinsic"
    TIME_OFFSET = "time_offset"
    GYRO_BIAS = "gyro_bias"


class ValueSource(str, Enum):
    """Explicit source for a node initial value or a soft-prior target."""

    DEFAULT = "default"
    CONSTANT = "constant"
    OPTIMIZED = "optimized"
    NUMERICAL = "numerical"


_VALUE_SOURCE_ALIASES = {
    "configured": ValueSource.CONSTANT,
    "config": ValueSource.CONSTANT,
    "user": ValueSource.CONSTANT,
    "provided": ValueSource.CONSTANT,
    "constant": ValueSource.CONSTANT,
    "previous": ValueSource.OPTIMIZED,
    "rolling": ValueSource.OPTIMIZED,
    "optimized": ValueSource.OPTIMIZED,
    "identity": ValueSource.DEFAULT,
    "zero": ValueSource.DEFAULT,
    "default": ValueSource.DEFAULT,
    "numerical": ValueSource.NUMERICAL,
}


def normalize_variable_type(variable_type: VariableType | str) -> VariableType:
    """Normalize a public variable-type value and produce readable validation errors."""

    if isinstance(variable_type, VariableType):
        return variable_type
    try:
        return VariableType(str(variable_type).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(variable.value for variable in VariableType)
        raise ValueError(f"Unknown variable_type {variable_type!r}; expected one of {allowed}") from exc


def normalize_value_source(source: ValueSource | str | None, *, allow_none: bool = False) -> ValueSource | None:
    """Normalize public source strings, including notebook-friendly aliases such as ``previous`` and ``configured``."""

    if source is None:
        if allow_none:
            return None
        return ValueSource.DEFAULT
    if isinstance(source, ValueSource):
        return source
    text = str(source).strip().lower()
    if text in _VALUE_SOURCE_ALIASES:
        return _VALUE_SOURCE_ALIASES[text]
    allowed = ", ".join(sorted(_VALUE_SOURCE_ALIASES))
    raise ValueError(f"Unknown value source {source!r}; expected one of {allowed}")


@dataclass(frozen=True, order=True)
class VariableKey:
    """Uniquely identifies one calibration variable owned by one physical sensor."""

    sensor_id: str
    variable_type: VariableType | str

    def __post_init__(self) -> None:
        sensor_id = str(self.sensor_id).strip()
        if not sensor_id:
            raise ValueError("VariableKey.sensor_id must be a nonempty string")
        object.__setattr__(self, "sensor_id", sensor_id)
        object.__setattr__(self, "variable_type", normalize_variable_type(self.variable_type))

    @property
    def label(self) -> str:
        return f"{self.sensor_id}:{self.variable_type.value}"


@dataclass(frozen=True)
class VariableRequirement:
    """A stream request for a shared calibration variable."""

    key: VariableKey
    stream_name: str
    reason: str = ""


@dataclass
class VariableConfig:
    """Configuration for one optimizable calibration variable.

    ``initial_source`` controls only the node value used to start optimization. ``prior_source`` controls only the optional soft-prior target, and no prior factor is added unless both ``prior_source`` and ``prior_information`` are supplied.
    """

    initial_source: ValueSource | str = ValueSource.DEFAULT
    initial_value: Any = None
    fixed: bool = False
    prior_source: ValueSource | str | None = None
    prior_value: Any = None
    prior_information: Any = None

    def normalized(self) -> "VariableConfig":
        """Return a shallow normalized copy without mutating the caller-owned config."""

        return VariableConfig(
            initial_source=normalize_value_source(self.initial_source),
            initial_value=self.initial_value,
            fixed=bool(self.fixed),
            prior_source=normalize_value_source(self.prior_source, allow_none=True),
            prior_value=self.prior_value,
            prior_information=self.prior_information,
        )

    def validate(self, key: VariableKey) -> None:
        """Catch source/prior combinations that are otherwise easy to misread in notebooks."""

        config = self.normalized()
        if config.prior_information is not None and config.prior_source is None:
            raise ValueError(f"{key.label} supplies prior_information but no prior_source")
        if config.prior_source is not None and config.prior_information is None:
            raise ValueError(f"{key.label} supplies prior_source but no prior_information")
        if config.initial_source == ValueSource.CONSTANT and config.initial_value is None:
            raise ValueError(f"{key.label} uses initial_source='constant' but initial_value is missing")
        if config.prior_source == ValueSource.CONSTANT and config.prior_value is None:
            raise ValueError(f"{key.label} uses prior_source='constant' but prior_value is missing")


@dataclass
class VariableNode:
    """A created mrob node and the calibration variable it represents."""

    key: VariableKey
    node_id: int
    initial_value: Any
    fixed: bool
    initial_source: ValueSource


def normalize_variable_config_map(configs: Mapping[Any, VariableConfig | Mapping[str, Any]] | None) -> dict[VariableKey, VariableConfig]:
    """Normalize config dictionaries keyed by ``VariableKey`` or by ``(sensor_id, variable_type)`` tuples."""

    normalized: dict[VariableKey, VariableConfig] = {}
    for raw_key, raw_config in ({} if configs is None else dict(configs)).items():
        if isinstance(raw_key, VariableKey):
            key = raw_key
        elif isinstance(raw_key, tuple) and len(raw_key) == 2:
            key = VariableKey(raw_key[0], raw_key[1])
        else:
            raise ValueError("variable_configs keys must be VariableKey objects or (sensor_id, variable_type) tuples")

        config = raw_config if isinstance(raw_config, VariableConfig) else VariableConfig(**dict(raw_config))
        config.validate(key)
        normalized[key] = config.normalized()
    return normalized


def default_value_for(variable_type: VariableType | str) -> Any:
    """Return the identity/zero default for a supported calibration variable."""

    variable_type = normalize_variable_type(variable_type)
    if variable_type == VariableType.EXTRINSIC:
        return np.eye(4)
    if variable_type == VariableType.TIME_OFFSET:
        return 0.0
    if variable_type == VariableType.GYRO_BIAS:
        return np.zeros(3)
    raise AssertionError(f"Unhandled variable type {variable_type}")


def copy_variable_value(value: Any) -> Any:
    """Copy calibration values without forcing scalars into arrays."""

    if isinstance(value, np.ndarray):
        return value.copy()
    if np.isscalar(value):
        return float(value)
    return np.asarray(value, dtype=float).copy()


def coerce_variable_value(key: VariableKey, value: Any, *, role: str) -> Any:
    """Validate and shape one calibration value for mrob node/factor APIs."""

    if key.variable_type == VariableType.EXTRINSIC:
        return data_processing._as_pose_matrix(default_value_for(key.variable_type) if value is None else value)
    if key.variable_type == VariableType.TIME_OFFSET:
        if value is None:
            value = 0.0
        value = float(np.asarray(value, dtype=float).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError(f"{key.label} {role} must be finite")
        return value
    if key.variable_type == VariableType.GYRO_BIAS:
        return data_processing._as_vector3(default_value_for(key.variable_type) if value is None else value, f"{key.label} {role}")
    raise AssertionError(f"Unhandled variable type {key.variable_type}")


def information_dimension(variable_type: VariableType | str) -> int:
    """Return the residual dimension used by a variable's existing mrob prior factor."""

    variable_type = normalize_variable_type(variable_type)
    if variable_type == VariableType.EXTRINSIC:
        return 6
    if variable_type == VariableType.TIME_OFFSET:
        return 1
    if variable_type == VariableType.GYRO_BIAS:
        return 3
    raise AssertionError(f"Unhandled variable type {variable_type}")


def add_variable_node(graph: mrob.FGraph, key: VariableKey, initial_value: Any, *, fixed: bool) -> int:
    """Create the mrob node matching one calibration variable."""

    mode = mrob.NODE_ANCHOR if fixed else mrob.NODE_STANDARD
    if key.variable_type == VariableType.EXTRINSIC:
        return graph.add_node_pose_3d(data_processing._as_mrob_se3(initial_value), mode=mode)
    if key.variable_type == VariableType.TIME_OFFSET:
        return graph.add_node_scalar(float(initial_value), mode=mode)
    if key.variable_type == VariableType.GYRO_BIAS:
        return graph.add_node_landmark_3d(data_processing._as_vector3(initial_value, key.label), mode=mode)
    raise AssertionError(f"Unhandled variable type {key.variable_type}")


def add_variable_prior(graph: mrob.FGraph, key: VariableKey, node_id: int, prior_value: Any, prior_information: Any) -> int:
    """Add a soft prior factor using the existing mrob observation bindings."""

    if key.variable_type == VariableType.EXTRINSIC:
        return graph.add_factor_1pose_3d(data_processing._as_mrob_se3(prior_value), node_id, data_processing._information_matrix(prior_information, 6))
    if key.variable_type == VariableType.TIME_OFFSET:
        return graph.add_factor_1_scalar_obs(float(prior_value), node_id, data_processing._information_matrix(prior_information, 1))
    if key.variable_type == VariableType.GYRO_BIAS:
        return graph.add_factor_1_landmark_3d(data_processing._as_vector3(prior_value, key.label), node_id, data_processing._information_matrix(prior_information, 3))
    raise AssertionError(f"Unhandled variable type {key.variable_type}")

