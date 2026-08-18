"""Structured graph metadata for later inspection and visualization.

This file records what RollingGraph constructed beside the opaque mrob graph. It deliberately has no plotting or graph-drawing dependency, leaving future visualization modules free to consume these records however they want.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .variables import ValueSource, VariableKey


@dataclass
class TrajectoryNodeMetadata:
    """Metadata for one trajectory pose node ``T_W_B[k]``."""

    pose_index: int
    timestamp: float
    node_id: int
    fixed: bool


@dataclass
class CalibrationVariableMetadata:
    """Metadata for one calibration variable node shared by streams."""

    key: VariableKey
    node_id: int
    fixed: bool
    initial_source: ValueSource
    initial_value: Any


@dataclass
class FactorMetadata:
    """Metadata for one factor added to the mrob graph."""

    factor_id: int | None
    factor_type: str
    stream_name: str
    sensor_id: str | None
    node_ids: tuple[int, ...]
    pose_indices: tuple[int, ...] = ()
    measurement_indices: tuple[int, ...] = ()
    variable_keys: tuple[VariableKey, ...] = ()
    accepted: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorMetadata:
    """Metadata for one optional soft prior on a calibration variable."""

    factor_id: int | None
    key: VariableKey
    node_id: int
    prior_source: ValueSource
    prior_value: Any
    fixed_node: bool
    added: bool
    information: Any = None


@dataclass
class StreamMetadata:
    """Metadata for one configured measurement stream."""

    stream_name: str
    stream_type: str
    sensor_id: str
    required_variables: tuple[VariableKey, ...]


@dataclass
class GraphMetadata:
    """All side-channel graph structure required by future visualizers and notebook diagnostics."""

    trajectory_nodes: list[TrajectoryNodeMetadata] = field(default_factory=list)
    calibration_variables: list[CalibrationVariableMetadata] = field(default_factory=list)
    priors: list[PriorMetadata] = field(default_factory=list)
    streams: list[StreamMetadata] = field(default_factory=list)
    factors: list[FactorMetadata] = field(default_factory=list)
    numerical_results: dict[str, Any] = field(default_factory=dict)
    window: dict[str, Any] = field(default_factory=dict)

    @property
    def factor_counts(self) -> dict[str, int]:
        """Count accepted, actually added factors by type."""

        counts: dict[str, int] = {}
        for factor in self.factors:
            if factor.factor_id is None or not factor.accepted:
                continue
            counts[factor.factor_type] = counts.get(factor.factor_type, 0) + 1
        for prior in self.priors:
            if not prior.added or prior.factor_id is None:
                continue
            key = f"{prior.key.variable_type.value}_prior"
            counts[key] = counts.get(key, 0) + 1
        return counts

