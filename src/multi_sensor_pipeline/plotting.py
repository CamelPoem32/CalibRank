"""Notebook plotting helpers for the modular multi-sensor calibration pipeline.

This file owns lightweight plots for measurement streams and rolling results produced by ``RollingGraph``. It deliberately does not visualize graph topology; future graph visualization should consume ``GraphMetadata`` in a separate module.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mrob
import numpy as np

from .streams import ComplexAccelStream, GyroStream, LidarOdometryStream, PoseObservationStream, SimpleAccelStream
from .variables import VariableKey, VariableType
import data_processing


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
        if isinstance(stream, PoseObservationStream):
            positions = np.asarray([pose[:3, 3] for pose in stream.poses], dtype=float)
            for axis, component in enumerate(component_names):
                axes[2].plot(stream.timestamps, positions[:, axis], label=f"{stream.stream_name} p_{component}", alpha=0.85, linestyle="--")

    axes[0].set_title("Gyroscope Streams")
    axes[0].set_ylabel("rad/s")
    axes[1].set_title("Accelerometer Streams")
    axes[1].set_ylabel("m/s^2")
    axes[2].set_title("LiDAR Odometry / Absolute Pose Translation")
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
    """Plot calibration-variable estimates over rolling-window midpoints."""

    if len(results) == 0:
        raise ValueError("results must contain at least one rolling-window result")

    reference_values = {} if reference_values is None else dict(reference_values)
    window_times = _as_window_times(results)

    fig, axes = plt.subplots(
        len(variable_keys),
        1,
        figsize=(14, max(3, 2.8 * len(variable_keys))),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    ##################################################
    # Convert and plot each available calibration history
    ##################################################

    for axis, key in zip(axes, variable_keys):
        raw_values = [
            result.calibration_value(key)
            for result in results
        ]

        valid_mask = np.asarray(
            [value is not None for value in raw_values],
            dtype=bool,
        )

        if not np.any(valid_mask):
            axis.text(
                0.5,
                0.5,
                f"No stored values for {key.label}",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            axis.set_ylabel(key.label)
            axis.grid(True, alpha=0.25)
            continue

        values = [
            value
            for value in raw_values
            if value is not None
        ]

        variable_times = window_times[valid_mask]

        if key.variable_type == VariableType.EXTRINSIC:
            series = np.asarray(
                [
                    mrob.SE3(
                        np.asarray(value, dtype=float)
                    ).Ln()
                    for value in values
                ],
                dtype=float,
            )
            labels = ("rx", "ry", "rz", "tx", "ty", "tz")

        elif key.variable_type == VariableType.TIME_OFFSET:
            series = np.asarray(
                values,
                dtype=float,
            ).reshape(-1, 1)
            labels = ("tau",)

        elif key.variable_type == VariableType.GYRO_BIAS:
            series = np.asarray(
                values,
                dtype=float,
            ).reshape(len(values), 3)
            labels = ("bx", "by", "bz")

        else:
            continue

        for component_index, label in enumerate(labels):
            axis.plot(
                variable_times,
                series[:, component_index],
                marker="o",
                markersize=3,
                label=label,
            )

        ##################################################
        # Overlay optional reference values
        ##################################################

        if key in reference_values:
            reference = reference_values[key]

            if key.variable_type == VariableType.EXTRINSIC:
                reference = mrob.SE3(
                    np.asarray(reference, dtype=float)
                ).Ln()

            elif key.variable_type == VariableType.TIME_OFFSET:
                reference = np.asarray(
                    [float(reference)]
                )

            else:
                reference = np.asarray(
                    reference,
                    dtype=float,
                ).reshape(-1)

            for component_index in range(
                min(reference.size, len(labels))
            ):
                axis.axhline(
                    float(reference[component_index]),
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.5,
                )

        axis.set_ylabel(key.label)
        axis.grid(True, alpha=0.25)
        axis.legend(
            ncol=min(6, len(labels)),
            fontsize="small",
        )

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

def plot_trajectory_errors(estimated_timestamps: Sequence[float], estimated_poses: Sequence[Any], reference_timestamps: Sequence[float], reference_poses: Sequence[Any], *, title: str = "Trajectory Estimation Errors", relative_time: bool = True) -> tuple[plt.Figure, np.ndarray, dict[str, dict[str, float]]]:
    """Plot full SE(3), rotational, and translational trajectory errors against an interpolated reference trajectory.

    The reference trajectory is interpolated independently at every estimated timestamp.

    The relative pose error is

        T_error = T_reference^{-1} T_estimated.

    The full SE(3) error is the Euclidean norm of the MROB logarithmic coordinates

        ||Log(T_error)||_2.

    MROB uses rotation-first tangent coordinates [rx, ry, rz, tx, ty, tz]. The rotational error is therefore the norm of the first three logarithmic coordinates, converted to degrees.

    The translational error is the Euclidean position error

        ||p_estimated - p_reference||_2.

    The returned statistics contain median and RMSE for all three metrics.
    """

    estimated_timestamps = np.asarray(estimated_timestamps, dtype=float).reshape(-1)
    reference_timestamps = np.asarray(reference_timestamps, dtype=float).reshape(-1)
    estimated_poses = np.asarray(estimated_poses, dtype=float)
    reference_poses = np.asarray(reference_poses, dtype=float)

    if estimated_timestamps.size == 0:
        raise ValueError("estimated_timestamps must contain at least one timestamp")

    if reference_timestamps.size < 2:
        raise ValueError("reference_timestamps must contain at least two timestamps for pose interpolation")

    if estimated_poses.shape != (estimated_timestamps.size, 4, 4):
        raise ValueError(f"estimated_poses must have shape ({estimated_timestamps.size}, 4, 4), got {estimated_poses.shape}")

    if reference_poses.shape != (reference_timestamps.size, 4, 4):
        raise ValueError(f"reference_poses must have shape ({reference_timestamps.size}, 4, 4), got {reference_poses.shape}")

    if np.any(np.diff(estimated_timestamps) <= 0.0):
        raise ValueError("estimated_timestamps must be strictly increasing")

    if np.any(np.diff(reference_timestamps) <= 0.0):
        raise ValueError("reference_timestamps must be strictly increasing")

    ##################################################
    # Keep only estimated poses covered by the reference trajectory
    ##################################################

    overlap_mask = (estimated_timestamps >= reference_timestamps[0]) & (estimated_timestamps <= reference_timestamps[-1])

    error_timestamps = estimated_timestamps[overlap_mask]
    error_estimated_poses = estimated_poses[overlap_mask]

    if error_timestamps.size == 0:
        raise ValueError(f"Estimated trajectory [{estimated_timestamps[0]}, {estimated_timestamps[-1]}] does not overlap reference trajectory [{reference_timestamps[0]}, {reference_timestamps[-1]}]")

    ##################################################
    # Interpolate the reference pose at every estimated timestamp
    ##################################################

    interpolated_reference_poses = np.asarray([data_processing._interpolate_pose(reference_timestamps, reference_poses, timestamp) for timestamp in error_timestamps], dtype=float)

    ##################################################
    # Calculate trajectory errors
    ##################################################

    full_se3_errors = np.empty(error_timestamps.size, dtype=float)
    rotational_errors_deg = np.empty(error_timestamps.size, dtype=float)
    translational_errors_m = np.empty(error_timestamps.size, dtype=float)

    for pose_index, (estimated_pose, reference_pose) in enumerate(zip(error_estimated_poses, interpolated_reference_poses)):
        relative_error_pose = np.linalg.inv(reference_pose) @ estimated_pose
        relative_error_tangent = np.asarray(mrob.SE3(relative_error_pose).Ln(), dtype=float).reshape(6)

        full_se3_errors[pose_index] = np.linalg.norm(relative_error_tangent)
        rotational_errors_deg[pose_index] = np.rad2deg(np.linalg.norm(relative_error_tangent[:3]))
        translational_errors_m[pose_index] = np.linalg.norm(estimated_pose[:3, 3] - reference_pose[:3, 3])

    ##################################################
    # Calculate summary statistics
    ##################################################

    def metric_statistics(values: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "rmse": float(np.sqrt(np.mean(values**2))),
        }

    statistics = {
        "se3": metric_statistics(full_se3_errors),
        "rotation_deg": metric_statistics(rotational_errors_deg),
        "translation_m": metric_statistics(translational_errors_m),
    }

    ##################################################
    # Use elapsed time for readable KAIST plots
    ##################################################

    if relative_time:
        plot_times = error_timestamps - error_timestamps[0]
        x_label = "time from start, s"
    else:
        plot_times = error_timestamps
        x_label = "time, s"

    ##################################################
    # Plot all three error metrics
    ##################################################

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(plot_times, full_se3_errors)
    axes[0].set_ylabel(r"$\|\mathrm{Log}(T_{ref}^{-1}T_{est})\|_2$")
    axes[0].set_title("Full SE(3) Tangent Error")

    axes[1].plot(plot_times, rotational_errors_deg)
    axes[1].set_ylabel("rotation error, deg")
    axes[1].set_title("Rotational Error")

    axes[2].plot(plot_times, translational_errors_m)
    axes[2].set_ylabel("translation error, m")
    axes[2].set_xlabel(x_label)
    axes[2].set_title("Translational Error")

    ##################################################
    # Add one compact statistics box to every subplot
    ##################################################

    statistics_texts = [
        f"Median: {statistics['se3']['median']:.6g}\nRMSE: {statistics['se3']['rmse']:.6g}",
        f"Median: {statistics['rotation_deg']['median']:.6g} deg\nRMSE: {statistics['rotation_deg']['rmse']:.6g} deg",
        f"Median: {statistics['translation_m']['median']:.6g} m\nRMSE: {statistics['translation_m']['rmse']:.6g} m",
    ]

    for axis, statistics_text in zip(axes, statistics_texts):
        axis.grid(True, alpha=0.25)
        axis.text(0.985, 0.95, statistics_text, transform=axis.transAxes, ha="right", va="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes, statistics

def _metric_statistics(values: np.ndarray) -> dict[str, float]:
    """Return median and RMSE over finite metric values."""

    values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return {
            "median": float("nan"),
            "rmse": float("nan"),
        }

    return {
        "median": float(np.median(finite_values)),
        "rmse": float(np.sqrt(np.mean(finite_values**2))),
    }


def _interpolate_trajectory_poses(timestamps: np.ndarray, poses: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    """Interpolate one SE(3) trajectory at the requested timestamps."""

    return np.asarray([data_processing._interpolate_pose(timestamps, poses, float(timestamp)) for timestamp in query_timestamps], dtype=float)


def _relative_pose_errors(estimated_start_pose: np.ndarray, estimated_end_pose: np.ndarray, reference_start_pose: np.ndarray, reference_end_pose: np.ndarray) -> tuple[float, float]:
    """Return translational and rotational relative-pose error for one motion interval."""

    estimated_delta_pose = np.linalg.inv(estimated_start_pose) @ estimated_end_pose
    reference_delta_pose = np.linalg.inv(reference_start_pose) @ reference_end_pose

    relative_motion_error = np.linalg.inv(reference_delta_pose) @ estimated_delta_pose
    relative_motion_error_tangent = np.asarray(mrob.SE3(relative_motion_error).Ln(), dtype=float).reshape(6)

    translational_error_m = float(np.linalg.norm(relative_motion_error[:3, 3]))
    rotational_error_deg = float(np.rad2deg(np.linalg.norm(relative_motion_error_tangent[:3])))

    return translational_error_m, rotational_error_deg


def plot_lidar_localization_benchmark_errors(
    estimated_timestamps: Sequence[float],
    estimated_poses: Sequence[Any],
    reference_timestamps: Sequence[float],
    reference_poses: Sequence[Any],
    *,
    rpe_delta_time_s: float = 1.0,
    rte_distance_m: float = 100.0,
    min_ate_normalization_distance_m: float = 100.0,
    title: str = "LiDAR Localization Benchmark Errors",
    relative_time: bool = True,
) -> tuple[plt.Figure, np.ndarray, dict[str, dict[str, float]]]:
    """Plot five localization metrics used for LiDAR / LiDAR-inertial trajectory evaluation.

    The five plotted metrics are:

        1. Running absolute trajectory error RMSE (ATE), in metres.
        2. Translational relative pose error (RPE) over ``rpe_delta_time_s``, in metres.
        3. Rotational relative pose error (RPE) over ``rpe_delta_time_s``, in degrees.
        4. Translational relative trajectory error (RTE) over ``rte_distance_m`` of reference travel, in metres.
        5. Running ATE normalized by travelled reference distance, in metres per kilometre.

    The absolute error is computed directly in the supplied world frame without
    post-hoc SE(3) alignment. The reference trajectory is interpolated independently
    at every estimated timestamp.

    For temporal RPE, both estimated and reference trajectories are interpolated at
    the exact interval end time. For distance-based RTE, the interval end time is
    obtained by interpolating reference cumulative path length to the requested
    travelled distance.

    The final value of the running ATE curve is the full-sequence ATE RMSE. The final
    value of the normalized ATE curve is the full-sequence ATE RMSE divided by total
    travelled reference distance in kilometres.
    """

    estimated_timestamps = np.asarray(estimated_timestamps, dtype=float).reshape(-1)
    reference_timestamps = np.asarray(reference_timestamps, dtype=float).reshape(-1)
    estimated_poses = np.asarray(estimated_poses, dtype=float)
    reference_poses = np.asarray(reference_poses, dtype=float)

    if estimated_timestamps.size == 0:
        raise ValueError("estimated_timestamps must contain at least one timestamp")

    if reference_timestamps.size < 2:
        raise ValueError("reference_timestamps must contain at least two timestamps for pose interpolation")

    if estimated_poses.shape != (estimated_timestamps.size, 4, 4):
        raise ValueError(f"estimated_poses must have shape ({estimated_timestamps.size}, 4, 4), got {estimated_poses.shape}")

    if reference_poses.shape != (reference_timestamps.size, 4, 4):
        raise ValueError(f"reference_poses must have shape ({reference_timestamps.size}, 4, 4), got {reference_poses.shape}")

    if np.any(np.diff(estimated_timestamps) <= 0.0):
        raise ValueError("estimated_timestamps must be strictly increasing")

    if np.any(np.diff(reference_timestamps) <= 0.0):
        raise ValueError("reference_timestamps must be strictly increasing")

    if rpe_delta_time_s <= 0.0:
        raise ValueError("rpe_delta_time_s must be positive")

    if rte_distance_m <= 0.0:
        raise ValueError("rte_distance_m must be positive")

    if min_ate_normalization_distance_m <= 0.0:
        raise ValueError("min_ate_normalization_distance_m must be positive")

    ##################################################
    # Keep only estimated poses covered by the reference trajectory
    ##################################################

    overlap_mask = (estimated_timestamps >= reference_timestamps[0]) & (estimated_timestamps <= reference_timestamps[-1])

    error_timestamps = estimated_timestamps[overlap_mask]
    error_estimated_poses = estimated_poses[overlap_mask]

    if error_timestamps.size == 0:
        raise ValueError(
            f"Estimated trajectory [{estimated_timestamps[0]}, {estimated_timestamps[-1]}] does not overlap "
            f"reference trajectory [{reference_timestamps[0]}, {reference_timestamps[-1]}]"
        )

    interpolated_reference_poses = _interpolate_trajectory_poses(reference_timestamps, reference_poses, error_timestamps)

    ##################################################
    # Calculate absolute translation errors and running ATE
    ##################################################

    absolute_translation_errors_m = np.linalg.norm(error_estimated_poses[:, :3, 3] - interpolated_reference_poses[:, :3, 3], axis=1)

    running_ate_rmse_m = np.sqrt(
        np.cumsum(absolute_translation_errors_m**2)
        / np.arange(1, absolute_translation_errors_m.size + 1, dtype=float)
    )

    ##################################################
    # Calculate reference travelled distance
    ##################################################

    reference_position_deltas = np.diff(interpolated_reference_poses[:, :3, 3], axis=0)
    reference_segment_distances_m = np.linalg.norm(reference_position_deltas, axis=1)
    cumulative_reference_distance_m = np.concatenate(([0.0], np.cumsum(reference_segment_distances_m)))

    ##################################################
    # Calculate fixed-time translational and rotational RPE
    ##################################################

    rpe_start_mask = error_timestamps + rpe_delta_time_s <= error_timestamps[-1]
    rpe_start_indices = np.flatnonzero(rpe_start_mask)
    rpe_end_timestamps = error_timestamps[rpe_start_indices] + rpe_delta_time_s

    translational_rpe_m = np.empty(rpe_start_indices.size, dtype=float)
    rotational_rpe_deg = np.empty(rpe_start_indices.size, dtype=float)

    if rpe_start_indices.size:
        rpe_estimated_end_poses = _interpolate_trajectory_poses(error_timestamps, error_estimated_poses, rpe_end_timestamps)
        rpe_reference_end_poses = _interpolate_trajectory_poses(reference_timestamps, reference_poses, rpe_end_timestamps)

        for output_index, (start_index, estimated_end_pose, reference_end_pose) in enumerate(zip(rpe_start_indices, rpe_estimated_end_poses, rpe_reference_end_poses)):
            translational_rpe_m[output_index], rotational_rpe_deg[output_index] = _relative_pose_errors(
                error_estimated_poses[start_index],
                estimated_end_pose,
                interpolated_reference_poses[start_index],
                reference_end_pose,
            )

    ##################################################
    # Calculate fixed-distance translational RTE
    ##################################################

    rte_end_timestamps_list = []
    rte_errors_m_list = []

    for start_index in range(error_timestamps.size):
        target_distance_m = cumulative_reference_distance_m[start_index] + rte_distance_m

        if target_distance_m > cumulative_reference_distance_m[-1]:
            break

        upper_index = int(np.searchsorted(cumulative_reference_distance_m, target_distance_m, side="left"))

        if upper_index <= start_index:
            continue

        lower_index = upper_index - 1

        lower_distance_m = cumulative_reference_distance_m[lower_index]
        upper_distance_m = cumulative_reference_distance_m[upper_index]

        if upper_distance_m <= lower_distance_m:
            continue

        interpolation_fraction = (target_distance_m - lower_distance_m) / (upper_distance_m - lower_distance_m)

        rte_end_timestamp = error_timestamps[lower_index] + interpolation_fraction * (error_timestamps[upper_index] - error_timestamps[lower_index])

        estimated_end_pose = data_processing._interpolate_pose(
            error_timestamps,
            error_estimated_poses,
            float(rte_end_timestamp),
        )

        reference_end_pose = data_processing._interpolate_pose(
            reference_timestamps,
            reference_poses,
            float(rte_end_timestamp),
        )

        translational_rte_m, _ = _relative_pose_errors(
            error_estimated_poses[start_index],
            estimated_end_pose,
            interpolated_reference_poses[start_index],
            reference_end_pose,
        )

        rte_end_timestamps_list.append(float(rte_end_timestamp))
        rte_errors_m_list.append(translational_rte_m)

    rte_end_timestamps = np.asarray(rte_end_timestamps_list, dtype=float)
    rte_errors_m = np.asarray(rte_errors_m_list, dtype=float)

    ##################################################
    # Calculate running ATE normalized by travelled distance
    ##################################################

    running_ate_per_km = np.full(error_timestamps.size, np.nan, dtype=float)

    normalization_mask = cumulative_reference_distance_m >= min_ate_normalization_distance_m

    running_ate_per_km[normalization_mask] = (
        running_ate_rmse_m[normalization_mask]
        / (cumulative_reference_distance_m[normalization_mask] / 1000.0)
    )

    total_reference_distance_km = float(cumulative_reference_distance_m[-1] / 1000.0)

    if total_reference_distance_km > 0.0:
        final_ate_per_km = float(running_ate_rmse_m[-1] / total_reference_distance_km)
    else:
        final_ate_per_km = float("nan")

    ##################################################
    # Calculate summary statistics
    ##################################################

    statistics = {
        "ate_m": _metric_statistics(absolute_translation_errors_m),
        "translation_rpe_m": _metric_statistics(translational_rpe_m),
        "rotation_rpe_deg": _metric_statistics(rotational_rpe_deg),
        "rte_m": _metric_statistics(rte_errors_m),
        "ate_per_km": {
            "value": final_ate_per_km,
            "trajectory_distance_km": total_reference_distance_km,
        },
    }

    ##################################################
    # Convert all timestamps to one common plotting time basis
    ##################################################

    plot_time_origin = error_timestamps[0] if relative_time else 0.0

    plot_times = error_timestamps - plot_time_origin
    rpe_plot_times = rpe_end_timestamps - plot_time_origin
    rte_plot_times = rte_end_timestamps - plot_time_origin

    if relative_time:
        x_label = "time from start, s"
    else:
        x_label = "time, s"

    ##################################################
    # Plot all five benchmark metrics
    ##################################################

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 16),
        sharex=True,
    )

    axes[0].plot(
        plot_times,
        running_ate_rmse_m,
    )
    axes[0].set_ylabel("ATE RMSE, m")
    axes[0].set_title("Running Absolute Trajectory Error")

    axes[1].plot(
        rpe_plot_times,
        translational_rpe_m,
    )
    axes[1].set_ylabel("translation RPE, m")
    axes[1].set_title(f"Translational RPE over {rpe_delta_time_s:g} s")

    axes[2].plot(
        rpe_plot_times,
        rotational_rpe_deg,
    )
    axes[2].set_ylabel("rotation RPE, deg")
    axes[2].set_title(f"Rotational RPE over {rpe_delta_time_s:g} s")

    axes[3].plot(
        rte_plot_times,
        rte_errors_m,
    )
    axes[3].set_ylabel("translation RTE, m")
    axes[3].set_title(f"Translational RTE over {rte_distance_m:g} m")

    axes[4].plot(
        plot_times,
        running_ate_per_km,
    )
    axes[4].set_ylabel("ATE, m/km")
    axes[4].set_xlabel(x_label)
    axes[4].set_title("Running ATE Normalized by Travelled Distance")

    ##################################################
    # Add one compact statistics box to every subplot
    ##################################################

    statistics_texts = [
        (
            f"ATE RMSE: {statistics['ate_m']['rmse']:.6g} m\n"
            f"Median abs.: {statistics['ate_m']['median']:.6g} m"
        ),
        (
            f"Median: {statistics['translation_rpe_m']['median']:.6g} m\n"
            f"RMSE: {statistics['translation_rpe_m']['rmse']:.6g} m"
        ),
        (
            f"Median: {statistics['rotation_rpe_deg']['median']:.6g} deg\n"
            f"RMSE: {statistics['rotation_rpe_deg']['rmse']:.6g} deg"
        ),
        (
            f"Median: {statistics['rte_m']['median']:.6g} m\n"
            f"RMSE: {statistics['rte_m']['rmse']:.6g} m"
        ),
        (
            f"Final: {statistics['ate_per_km']['value']:.6g} m/km\n"
            f"Distance: {statistics['ate_per_km']['trajectory_distance_km']:.6g} km"
        ),
    ]

    for axis, statistics_text in zip(axes, statistics_texts):
        axis.grid(True, alpha=0.25)

        axis.text(
            0.985,
            0.95,
            statistics_text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.8,
            },
        )

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes, statistics
