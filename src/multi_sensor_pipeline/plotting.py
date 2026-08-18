"""Notebook plotting helpers for the modular multi-sensor calibration pipeline.

This file owns lightweight plots for measurement streams and rolling results produced by ``RollingGraph``. It deliberately does not visualize graph topology; future graph visualization should consume ``GraphMetadata`` in a separate module.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mrob
import numpy as np

from .streams import ComplexAccelStream, GyroStream, LidarOdometryStream, SimpleAccelStream
from .variables import VariableKey, VariableType


def _as_window_times(results: Sequence[Any]) -> np.ndarray:
    """Return one midpoint timestamp per rolling-window result."""

    return np.asarray([0.5 * (float(result.window_start) + float(result.window_end)) for result in results], dtype=float)


def plot_stream_measurements(streams: Sequence[Any]) -> tuple[plt.Figure, np.ndarray]:
    """Plot measurements owned by configured pipeline streams.

    Args:
        streams: Measurement streams passed to ``RollingGraph``.

    Returns:
        Matplotlib figure and axes.
    """

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    component_names = ("x", "y", "z")

    # Plot IMU stream variables from the actual stream objects used by RollingGraph.
    for stream in streams:
        if isinstance(stream, GyroStream):
            for axis, component in enumerate(component_names):
                axes[0].plot(stream.timestamps, stream.angular_velocity[:, axis], label=f"{stream.stream_name} {component}", alpha=0.85)
        if isinstance(stream, (SimpleAccelStream, ComplexAccelStream)):
            for axis, component in enumerate(component_names):
                axes[1].plot(stream.timestamps, stream.acceleration[:, axis], label=f"{stream.stream_name} {component}", alpha=0.85)
        if isinstance(stream, LidarOdometryStream):
            positions = np.asarray([pose[:3, 3] for pose in stream.odometry_poses], dtype=float)
            for axis, component in enumerate(component_names):
                axes[2].plot(stream.timestamps, positions[:, axis], label=f"{stream.stream_name} p_{component}", alpha=0.85)

    axes[0].set_title("Gyroscope Streams")
    axes[0].set_ylabel("rad/s")
    axes[1].set_title("Accelerometer Streams")
    axes[1].set_ylabel("m/s^2")
    axes[2].set_title("Accumulated LiDAR Odometry Translation")
    axes[2].set_ylabel("m")
    axes[2].set_xlabel("time, s")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=3, fontsize="small")
    fig.tight_layout()
    return fig, axes


def plot_rolling_trajectory(
    estimated_timestamps: Sequence[float],
    estimated_poses: Sequence[Any],
    *,
    reference_timestamps: Sequence[float] | None = None,
    reference_poses: Sequence[Any] | None = None,
    initial_poses: Sequence[Any] | None = None,
    title: str = "RollingGraph Trajectory",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot estimated XY trajectory and height over time.

    Args:
        estimated_timestamps: Timestamps associated with stitched rolling output poses.
        estimated_poses: Estimated ``T_W_B`` pose matrices.
        reference_timestamps: Optional timestamps for a reference trajectory.
        reference_poses: Optional reference pose matrices.
        initial_poses: Optional trajectory initialization poses.
        title: Figure title.

    Returns:
        Matplotlib figure and axes.
    """

    estimated_timestamps = np.asarray(estimated_timestamps, dtype=float)
    estimated_poses = np.asarray(estimated_poses, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # XY plan view is the most compact way to see whether rolling windows stitch coherently.
    if reference_poses is not None:
        reference_poses = np.asarray(reference_poses, dtype=float)
        axes[0].plot(reference_poses[:, 0, 3], reference_poses[:, 1, 3], label="reference", color="black", linewidth=1.5)
    if initial_poses is not None:
        initial_poses = np.asarray(initial_poses, dtype=float)
        axes[0].plot(initial_poses[:, 0, 3], initial_poses[:, 1, 3], label="initial", linestyle="--", alpha=0.8)
    if estimated_poses.size:
        axes[0].plot(estimated_poses[:, 0, 3], estimated_poses[:, 1, 3], label="estimated", alpha=0.9)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x, m")
    axes[0].set_ylabel("y, m")
    axes[0].set_title("XY Trajectory")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    # Height over time catches frame-rotation mistakes that are hard to see in a planar plot.
    if reference_timestamps is not None and reference_poses is not None:
        axes[1].plot(reference_timestamps, reference_poses[:, 2, 3], label="reference z", color="black", linewidth=1.5)
    if estimated_poses.size:
        axes[1].plot(estimated_timestamps, estimated_poses[:, 2, 3], label="estimated z", alpha=0.9)
    axes[1].set_xlabel("time, s")
    axes[1].set_ylabel("z, m")
    axes[1].set_title("Height")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_calibration_estimates(
    results: Sequence[Any],
    variable_keys: Sequence[VariableKey],
    *,
    reference_values: Mapping[VariableKey, Any] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot calibration-variable estimates over rolling-window midpoints.

    Args:
        results: ``WindowResult`` objects returned by ``RollingGraph``.
        variable_keys: Calibration variables to plot.
        reference_values: Optional reference values keyed by ``VariableKey``.

    Returns:
        Matplotlib figure and axes.
    """

    if len(results) == 0:
        raise ValueError("results must contain at least one rolling-window result")
    reference_values = {} if reference_values is None else dict(reference_values)
    window_times = _as_window_times(results)
    fig, axes = plt.subplots(len(variable_keys), 1, figsize=(14, max(3, 2.8 * len(variable_keys))), sharex=True, squeeze=False)
    axes = axes[:, 0]

    # Convert each variable into a readable Euclidean series: SE(3) tangent, scalar tau, or 3D bias.
    for axis, key in zip(axes, variable_keys):
        values = [result.calibration_value(key) for result in results]
        if key.variable_type == VariableType.EXTRINSIC:
            series = np.asarray([mrob.SE3(value).Ln() for value in values], dtype=float)
            labels = ("rx", "ry", "rz", "tx", "ty", "tz")
        elif key.variable_type == VariableType.TIME_OFFSET:
            series = np.asarray(values, dtype=float).reshape(-1, 1)
            labels = ("tau",)
        elif key.variable_type == VariableType.GYRO_BIAS:
            series = np.asarray(values, dtype=float).reshape(len(results), 3)
            labels = ("bx", "by", "bz")
        else:
            continue

        for component_index, label in enumerate(labels):
            axis.plot(window_times, series[:, component_index], marker="o", markersize=3, label=label)

        if key in reference_values:
            reference = reference_values[key]
            if key.variable_type == VariableType.EXTRINSIC:
                reference = mrob.SE3(reference).Ln()
            elif key.variable_type == VariableType.TIME_OFFSET:
                reference = np.asarray([float(reference)])
            else:
                reference = np.asarray(reference, dtype=float).reshape(-1)
            for component_index, label in enumerate(labels[: reference.size]):
                axis.axhline(float(reference[component_index]), linestyle="--", linewidth=0.8, alpha=0.5)

        axis.set_ylabel(key.label)
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=min(6, len(labels)), fontsize="small")

    axes[-1].set_xlabel("window midpoint time, s")
    fig.tight_layout()
    return fig, axes


def print_rolling_result_summary(result: Any) -> None:
    """Print one concise rolling-window result summary."""

    print(f"window {result.window_index}: [{result.window_start:.3f}, {result.window_end:.3f}]")
    print(f"poses: {len(result.pose_timestamps)}")
    print(f"chi2: {result.chi2_before:.6e} -> {result.chi2_after:.6e}")
    print("factor counts:", result.factor_counts)
    for key, value in sorted(result.calibration_values.items(), key=lambda item: item[0].label):
        if key.variable_type == VariableType.EXTRINSIC:
            print(f"{key.label} Ln:", mrob.SE3(value).Ln())
        else:
            print(f"{key.label}:", value)

