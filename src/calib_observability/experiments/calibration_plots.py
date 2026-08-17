"""Plotting helpers for real-data calibration injection notebooks."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .calibration_results import CalibrationResultSeries


AXIS_LABELS = ("x", "y", "z")


def plot_raw_imu_stream(imu_data, *, title: str = "Raw IMU measurements"):
    """Plot accelerometer and gyroscope components from an IMU stream.

    Args:
        imu_data: Object with ``timestamps_s``, ``accel_mps2``, and
            ``gyro_radps`` arrays.
        title: Figure title.

    Returns:
        Matplotlib ``(figure, axes)`` tuple.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    timestamps = np.asarray(imu_data.timestamps_s, dtype=float)

    for axis in range(3):
        axes[0].plot(timestamps, imu_data.accel_mps2[:, axis], label=f"a_{AXIS_LABELS[axis]}")
        axes[1].plot(timestamps, imu_data.gyro_radps[:, axis], label=f"w_{AXIS_LABELS[axis]}")

    axes[0].set_ylabel("accel [m/s^2]")
    axes[1].set_ylabel("gyro [rad/s]")
    axes[1].set_xlabel("sensor time [s]")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    return fig, axes


def plot_lidar_diagnostics(lidar_data, *, title: str = "LiDAR ICP diagnostics"):
    """Plot LiDAR scan cadence, ICP fitness, and inlier RMSE.

    Args:
        lidar_data: ``LidarData`` with scan timestamps and optional ICP metrics.
        title: Figure title.

    Returns:
        Matplotlib ``(figure, axes)`` tuple.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
    interval_timestamps = np.asarray(lidar_data.timestamps_s, dtype=float)
    scan_timestamps = lidar_data.scan_timestamps_s
    if scan_timestamps is not None:
        axes[0].plot(scan_timestamps[:-1], np.diff(scan_timestamps), marker=".", linestyle="-")
    else:
        axes[0].plot(interval_timestamps, np.full_like(interval_timestamps, np.nan), marker=".")

    fitness = np.full_like(interval_timestamps, np.nan)
    rmse = np.full_like(interval_timestamps, np.nan)
    if lidar_data.fitness is not None:
        fitness = np.asarray(lidar_data.fitness, dtype=float)
    if lidar_data.inlier_rmse is not None:
        rmse = np.asarray(lidar_data.inlier_rmse, dtype=float)

    axes[1].plot(interval_timestamps, fitness, marker=".")
    axes[2].plot(interval_timestamps, rmse, marker=".")
    axes[0].set_title(title)
    axes[0].set_ylabel("scan dt [s]")
    axes[1].set_ylabel("ICP fitness")
    axes[2].set_ylabel("ICP RMSE [m]")
    axes[2].set_xlabel("LiDAR time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_timestamp_alignment(
    *,
    imu_timestamps_s: Sequence[float],
    lidar_timestamps_s: Sequence[float],
    title: str = "IMU and LiDAR timestamp alignment",
):
    """Plot IMU and LiDAR sample times on one horizontal timeline.

    Args:
        imu_timestamps_s: IMU timestamps, shape ``(N,)``.
        lidar_timestamps_s: LiDAR timestamps, shape ``(M,)``.
        title: Figure title.

    Returns:
        Matplotlib ``(figure, axis)`` tuple.
    """
    import matplotlib.pyplot as plt

    imu_timestamps = np.asarray(imu_timestamps_s, dtype=float)
    lidar_timestamps = np.asarray(lidar_timestamps_s, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(imu_timestamps, np.zeros_like(imu_timestamps), ".", markersize=1.5, label="IMU")
    ax.plot(lidar_timestamps, np.ones_like(lidar_timestamps), "|", markersize=14, label="LiDAR scans")
    ax.set_yticks([0, 1], ["IMU", "LiDAR"])
    ax.set_xlabel("time [s]")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_xy_trajectories(
    trajectories: Mapping[str, tuple[Sequence[float], np.ndarray]],
    *,
    title: str = "Trajectory comparison",
):
    """Plot XY paths for reference, baseline, and injected trajectories.

    Args:
        trajectories: Mapping from label to ``(timestamps, poses_se3)``.
        title: Figure title.

    Returns:
        Matplotlib ``(figure, axis)`` tuple.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, (_, poses) in trajectories.items():
        poses = np.asarray(poses, dtype=float)
        if poses.size == 0:
            continue
        ax.plot(poses[:, 0, 3], poses[:, 1, 3], label=label)
        ax.scatter(poses[0, 0, 3], poses[0, 1, 3], s=30)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_calibration_series(
    baseline: CalibrationResultSeries,
    injected: CalibrationResultSeries | None = None,
    *,
    title: str = "Estimated calibration series",
):
    """Plot window-wise bias and time-offset estimates.

    Args:
        baseline: Baseline result series.
        injected: Optional injected result series.
        title: Figure title.

    Returns:
        Matplotlib ``(figure, axes)`` tuple.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    _plot_bias_components(axes[0], baseline, linestyle="-")
    axes[1].plot(baseline.window_midpoints_s, baseline.tau_I_s, label=f"{baseline.label} tau_I")
    axes[2].plot(baseline.window_midpoints_s, baseline.tau_L_s, label=f"{baseline.label} tau_L")

    if injected is not None:
        _plot_bias_components(axes[0], injected, linestyle="--")
        axes[1].plot(injected.window_midpoints_s, injected.tau_I_s, "--", label=f"{injected.label} tau_I")
        axes[2].plot(injected.window_midpoints_s, injected.tau_L_s, "--", label=f"{injected.label} tau_L")

    axes[0].set_title(title)
    axes[0].set_ylabel("bias_g [rad/s]")
    axes[1].set_ylabel("tau_I [s]")
    axes[2].set_ylabel("tau_L [s]")
    axes[2].set_xlabel("window midpoint [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    return fig, axes


def plot_baseline_relative_deltas(
    deltas: Mapping[str, np.ndarray],
    *,
    injected_targets: Mapping[str, np.ndarray | float] | None = None,
    title: str = "Baseline-relative injected calibration deltas",
):
    """Plot estimated injected-minus-baseline calibration deltas.

    Args:
        deltas: Dictionary returned by ``calibration_delta_from_baseline``.
        injected_targets: Optional independently known injected deltas.
        title: Figure title.

    Returns:
        Matplotlib ``(figure, axes)`` tuple.
    """
    import matplotlib.pyplot as plt

    t = np.asarray(deltas["window_midpoints_s"], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    bias = np.asarray(deltas["bias_g_delta_radps"], dtype=float)
    for axis in range(3):
        axes[0].plot(t, bias[:, axis], label=f"bias_{AXIS_LABELS[axis]}")
    tau_I = np.asarray(deltas["tau_I_delta_s"], dtype=float)
    tau_L = np.asarray(deltas["tau_L_delta_s"], dtype=float)
    axes[1].plot(t, tau_I, label="tau_I")
    axes[1].plot(t, tau_L, label="tau_L")

    T_B_I_delta_vectors = np.asarray(deltas["T_B_I_delta_vectors"], dtype=float)
    for axis in range(6):
        axes[2].plot(t, T_B_I_delta_vectors[:, axis], label=f"T_B_I[{axis}]")

    if injected_targets:
        _draw_target_lines(axes[0], injected_targets.get("gyro_bias_delta_radps"))
        if "tau_I_delta_s" in injected_targets:
            axes[1].axhline(float(injected_targets["tau_I_delta_s"]), color="C0", linestyle=":", label="target tau_I")
        if "tau_L_delta_s" in injected_targets:
            axes[1].axhline(float(injected_targets["tau_L_delta_s"]), color="C1", linestyle=":", label="target tau_L")
        if "T_B_I_delta_vector" in injected_targets:
            target = np.asarray(injected_targets["T_B_I_delta_vector"], dtype=float).reshape(6)
            for axis in range(6):
                axes[2].axhline(target[axis], color=f"C{axis % 10}", linestyle=":")

    axes[0].set_title(title)
    axes[0].set_ylabel("bias delta [rad/s]")
    axes[1].set_ylabel("tau delta [s]")
    axes[2].set_ylabel("T_B_I delta tangent")
    axes[2].set_xlabel("window midpoint [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    return fig, axes


def plot_factor_counts(series: CalibrationResultSeries, *, title: str = "Factor counts"):
    """Plot per-window factor counts by family.

    Args:
        series: Window-wise result series.
        title: Figure title.

    Returns:
        Matplotlib ``(figure, axis)`` tuple.
    """
    import matplotlib.pyplot as plt

    families = sorted({family for counts in series.factor_counts for family in counts})
    fig, ax = plt.subplots(figsize=(12, 4))
    for family in families:
        ax.plot(
            series.window_midpoints_s,
            [counts.get(family, 0) for counts in series.factor_counts],
            marker=".",
            label=family,
        )
    ax.set_title(title)
    ax.set_xlabel("window midpoint [s]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=3)
    fig.tight_layout()
    return fig, ax


def _plot_bias_components(axis, series: CalibrationResultSeries, *, linestyle: str) -> None:
    """Plot the three gyroscope-bias components on one axis."""
    for component in range(3):
        axis.plot(
            series.window_midpoints_s,
            series.bias_g_radps[:, component],
            linestyle=linestyle,
            label=f"{series.label} b_{AXIS_LABELS[component]}",
        )


def _draw_target_lines(axis, target) -> None:
    """Draw horizontal target lines for optional vector targets."""
    if target is None:
        return
    values = np.asarray(target, dtype=float).reshape(3)
    for component in range(3):
        axis.axhline(values[component], color=f"C{component}", linestyle=":")
