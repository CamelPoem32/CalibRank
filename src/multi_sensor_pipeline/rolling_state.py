"""Rolling-window state shared across calibration windows.

This file stores solved trajectory poses and calibration-variable estimates for warm-starting future windows. It deliberately does not own measurement streams or factor creation; streams remain stateless descriptions of measurements, while ``RollingState`` stores the optimizer's evolving solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .variables import VariableKey, copy_variable_value


@dataclass
class RollingState:
    """Warm-start state carried from one rolling graph window to the next."""

    calibration_values: dict[VariableKey, Any] = field(default_factory=dict)
    pose_cache: dict[float, np.ndarray] = field(default_factory=dict)
    output_pose_cache: dict[float, np.ndarray] = field(default_factory=dict)
    output_pose_times: dict[float, float] = field(default_factory=dict)
    window_results: list[Any] = field(default_factory=list)
    window_metadata: list[dict[str, Any]] = field(default_factory=list)

    def clear(self) -> None:
        """Drop all warm-start and output state."""

        self.calibration_values.clear()
        self.pose_cache.clear()
        self.output_pose_cache.clear()
        self.output_pose_times.clear()
        self.window_results.clear()
        self.window_metadata.clear()

    @staticmethod
    def time_key(timestamp: float) -> float:
        """Use stable rounded floating-time keys for overlapping windows."""

        return round(float(timestamp), 9)

    def get_calibration(self, key: VariableKey) -> Any | None:
        """Return a copy of the latest optimized value for one calibration variable."""

        if key not in self.calibration_values:
            return None
        return copy_variable_value(self.calibration_values[key])

    def set_calibration(self, key: VariableKey, value: Any) -> None:
        """Store a copy of one optimized calibration value."""

        self.calibration_values[key] = copy_variable_value(value)

    def pose_prefix(self, pose_timestamps: Sequence[float]) -> list[np.ndarray]:
        """Return the consecutive solved pose prefix available for the given timestamps."""

        poses: list[np.ndarray] = []
        for timestamp in pose_timestamps:
            key = self.time_key(timestamp)
            if key not in self.pose_cache:
                break
            poses.append(self.pose_cache[key].copy())
        return poses

    def store_window_solution(
        self,
        pose_timestamps: Sequence[float],
        trajectory_poses: Sequence[Any],
        calibration_values: dict[VariableKey, Any],
        *,
        result: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store optimized poses and calibration variables after a solved window."""

        for timestamp, pose in zip(pose_timestamps, trajectory_poses):
            self.pose_cache[self.time_key(timestamp)] = np.asarray(pose, dtype=float).copy()
        for key, value in calibration_values.items():
            self.set_calibration(key, value)
        if result is not None:
            self.window_results.append(result)
        if metadata is not None:
            self.window_metadata.append(dict(metadata))

    def commit_output_segment(self, pose_timestamps: Sequence[float], trajectory_poses: Sequence[Any], *, commit_end: float, include_end: bool) -> None:
        """Commit the non-overlapping output segment of a rolling window."""

        for timestamp, pose in zip(pose_timestamps, trajectory_poses):
            should_commit = timestamp <= commit_end if include_end else timestamp < commit_end
            if should_commit:
                key = self.time_key(timestamp)
                self.output_pose_cache[key] = np.asarray(pose, dtype=float).copy()
                self.output_pose_times[key] = float(timestamp)

    @property
    def rolling_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Return committed rolling trajectory poses ordered by timestamp."""

        if len(self.output_pose_cache) == 0:
            return np.empty(0), np.empty((0, 4, 4))
        keys = sorted(self.output_pose_cache)
        timestamps = np.array([self.output_pose_times[key] for key in keys])
        poses = np.array([self.output_pose_cache[key] for key in keys])
        return timestamps, poses

