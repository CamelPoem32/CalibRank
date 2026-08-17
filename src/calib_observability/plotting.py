'''Small plotting helpers for validation notebooks.'''

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


##################################################
# Validation plot writers
##################################################
def save_singular_values_plot(values: ArrayLike, path: str | Path, title: str) -> Path:
    '''Save a singular-value stem plot.
    
    Args:
        values: Singular values, shape `(N,)`.
        path: Output image path.
        title: Figure title.
    
    Returns:
        pathlib.Path: Saved figure path.
    '''

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.semilogy(np.arange(1, s.size + 1), np.maximum(s, 1e-16), marker="o")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_sparsity_plot(J: object, path: str | Path, title: str) -> Path:
    '''Save a matrix sparsity plot.
    
    Args:
        J: Dense or sparse matrix accepted by ``matplotlib.axes.Axes.spy``.
        path: Output image path.
        title: Figure title.
    
    Returns:
        Path to the saved image.
    '''

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spy(J, markersize=2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out

def plot_trajectory_2d(
    poses: ArrayLike,
    *,
    title: str = "Trajectory",
    label: str = "trajectory",
    color: str = "tab:blue",
    linestyle: str = "-",
    linewidth: float = 1.8,
    show_start_end: bool = True,
):
    '''Plot a single SE(3) trajectory in the XY plane.

    Args:
        poses: SE(3) poses with shape (N, 4, 4).
        title: Figure title.
        label: Trajectory label shown in the legend.
        color: Matplotlib color used for the trajectory.
        linestyle: Matplotlib line style.
        linewidth: Trajectory line width.
        show_start_end: Whether to mark the first and last trajectory poses.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figure and
        trajectory axis.

    Raises:
        ValueError: If ``poses`` does not have shape ``(N, 4, 4)`` or contains
            no poses.
    '''
    pose_array = np.asarray(poses, dtype=float)

    if pose_array.ndim != 3 or pose_array.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")
    if pose_array.shape[0] == 0:
        raise ValueError("poses must contain at least one pose")

    trajectory_xy = pose_array[:, :2, 3]

    ##################################################
    # Plot the planar trajectory
    ##################################################

    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], label=label, color=color, linestyle=linestyle, linewidth=linewidth)

    if show_start_end:
        axis.scatter(trajectory_xy[0, 0], trajectory_xy[0, 1], marker="o", color="tab:green", label="start", zorder=3)
        axis.scatter(trajectory_xy[-1, 0], trajectory_xy[-1, 1], marker="x", color="tab:red", label="end", zorder=3)

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()

    fig.tight_layout()
    plt.show()
    return fig, axis


def plot_trajectory_comparison(
    true_poses: ArrayLike,
    estimated_poses: ArrayLike,
    *,
    true_timestamps: ArrayLike | None = None,
    estimated_timestamps: ArrayLike | None = None,
    initial_poses: ArrayLike | None = None,
    title: str = 'Trajectory comparison',
):
    '''Plot true, initial, and estimated planar trajectories.

    Args:
        true_poses: Ground-truth SE(3) poses with shape (N, 4, 4).
        estimated_poses: Estimated SE(3) poses with shape (M, 4, 4).
        true_timestamps: Optional timestamps for ground-truth poses, shape (N,).
        estimated_timestamps: Optional timestamps for estimated poses, shape (M,).
        initial_poses: Optional initial SE(3) guesses with shape (K, 4, 4).
        title: Figure title.

    Returns:
        tuple[matplotlib.figure.Figure, numpy.ndarray]: Figure and the (2,)
        axes array. The first axis shows XY position and the second axis shows
        planar position error when timestamps are supplied.
    '''

    true_pose_array = np.asarray(true_poses, dtype=float)
    estimated_pose_array = np.asarray(estimated_poses, dtype=float)
    if true_pose_array.ndim != 3 or true_pose_array.shape[1:] != (4, 4):
        raise ValueError('true_poses must have shape (N, 4, 4)')
    if estimated_pose_array.ndim != 3 or estimated_pose_array.shape[1:] != (4, 4):
        raise ValueError('estimated_poses must have shape (M, 4, 4)')

    true_xy = true_pose_array[:, :2, 3]
    estimated_xy = estimated_pose_array[:, :2, 3]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Draw the map-view trajectory first; equal aspect keeps geometry honest.
    axes[0].plot(true_xy[:, 0], true_xy[:, 1], label='true', color='black', linewidth=2.0)
    if initial_poses is not None:
        initial_pose_array = np.asarray(initial_poses, dtype=float)
        if initial_pose_array.ndim != 3 or initial_pose_array.shape[1:] != (4, 4):
            raise ValueError('initial_poses must have shape (K, 4, 4)')
        initial_xy = initial_pose_array[:, :2, 3]
        axes[0].plot(initial_xy[:, 0], initial_xy[:, 1], label='initial', color='tab:orange', linestyle='--', alpha=0.8)
    axes[0].plot(estimated_xy[:, 0], estimated_xy[:, 1], label='estimated', color='tab:blue', linewidth=1.8)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # If timestamps are available, interpolate true XY to estimated times and plot error.
    if true_timestamps is not None and estimated_timestamps is not None and estimated_pose_array.size:
        true_times = np.asarray(true_timestamps, dtype=float).reshape(-1)
        estimated_times = np.asarray(estimated_timestamps, dtype=float).reshape(-1)
        if true_times.shape != (true_pose_array.shape[0],):
            raise ValueError('true_timestamps must have shape (N,)')
        if estimated_times.shape != (estimated_pose_array.shape[0],):
            raise ValueError('estimated_timestamps must have shape (M,)')
        true_x = np.interp(estimated_times, true_times, true_xy[:, 0])
        true_y = np.interp(estimated_times, true_times, true_xy[:, 1])
        errors = np.linalg.norm(estimated_xy - np.column_stack([true_x, true_y]), axis=1)
        axes[1].plot(estimated_times, errors, color='tab:red', linewidth=1.8)
        axes[1].set_xlabel('time [s]')
        axes[1].set_ylabel('XY error [m]')
        axes[1].set_title('Estimated trajectory error')
    else:
        axes[1].axis('off')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_calibration_window_chi2(results: object):
    '''Plot before/after chi2 values for rolling factor-graph windows.

    Args:
        results: Iterable of CalibrationWindowResult objects returned by
            FactorGraphCalibration.generate_filter_iterative.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figure and axis
        containing one line for pre-solve chi2 and one line for post-solve chi2.
    '''

    result_list = list(results)
    window_indices = np.asarray([result.window_index for result in result_list], dtype=int)
    chi2_before = np.asarray([result.chi2_before for result in result_list], dtype=float)
    chi2_after = np.asarray([result.chi2_after for result in result_list], dtype=float)

    fig, axis = plt.subplots(figsize=(8, 3.5))
    axis.semilogy(window_indices, np.maximum(chi2_before, 1e-16), marker='o', label='before')
    axis.semilogy(window_indices, np.maximum(chi2_after, 1e-16), marker='o', label='after')
    axis.set_xlabel('window index')
    axis.set_ylabel('chi2')
    axis.set_title('Rolling factor-graph objective')
    axis.grid(True, which='both', alpha=0.3)
    axis.legend()
    fig.tight_layout()
    return fig, axis



def plot_imu_calibration_emulation_diagnostics(
    reference_timestamps: ArrayLike,
    original_sensor_timestamps: ArrayLike,
    emulated_sensor_timestamps: ArrayLike,
    original_gyroscope: ArrayLike,
    emulated_gyroscope: ArrayLike,
    original_accelerometer: ArrayLike,
    emulated_accelerometer: ArrayLike,
    T_B_I_truth: ArrayLike,
    tau_I_truth: ArrayLike,
):
    '''Plot compact diagnostics for artificial IMU calibration emulation.

    Args:
        reference_timestamps: Physical/reference timestamps with shape ``(N,)``.
        original_sensor_timestamps: Original sensor timestamps with shape ``(N,)``.
        emulated_sensor_timestamps: Warped sensor timestamps with shape ``(N,)``.
        original_gyroscope: Original gyroscope samples, shape ``(N, 3)``.
        emulated_gyroscope: Emulated gyroscope samples, shape ``(N, 3)``.
        original_accelerometer: Original accelerometer samples, shape ``(N, 3)``.
        emulated_accelerometer: Emulated accelerometer samples, shape ``(N, 3)``.
        T_B_I_truth: Artificial body-from-IMU truth, shape ``(N, 4, 4)``.
        tau_I_truth: Artificial temporal truth, shape ``(N,)``.

    Returns:
        tuple[matplotlib.figure.Figure, numpy.ndarray]: Figure and axes.
    '''

    from .lie_se3 import se3_log

    reference_times = np.asarray(reference_timestamps, dtype=float).reshape(-1)
    original_times = np.asarray(original_sensor_timestamps, dtype=float).reshape(-1)
    emulated_times = np.asarray(emulated_sensor_timestamps, dtype=float).reshape(-1)
    gyro_original = np.asarray(original_gyroscope, dtype=float)
    gyro_emulated = np.asarray(emulated_gyroscope, dtype=float)
    accel_original = np.asarray(original_accelerometer, dtype=float)
    accel_emulated = np.asarray(emulated_accelerometer, dtype=float)
    T_truth = np.asarray(T_B_I_truth, dtype=float)
    tau_truth = np.asarray(tau_I_truth, dtype=float).reshape(-1)

    if reference_times.size == 0:
        raise ValueError("reference_timestamps must not be empty")
    if gyro_original.shape != gyro_emulated.shape or gyro_original.shape != (reference_times.size, 3):
        raise ValueError("gyroscope arrays must have shape (N, 3)")
    if accel_original.shape != accel_emulated.shape or accel_original.shape != (reference_times.size, 3):
        raise ValueError("accelerometer arrays must have shape (N, 3)")
    if T_truth.shape != (reference_times.size, 4, 4):
        raise ValueError("T_B_I_truth must have shape (N, 4, 4)")
    if tau_truth.shape != reference_times.shape or original_times.shape != reference_times.shape or emulated_times.shape != reference_times.shape:
        raise ValueError("timestamp and tau arrays must have matching shape (N,)")

    relative_time = reference_times - reference_times[0]
    T_vectors = np.vstack([se3_log(transform) for transform in T_truth])
    timestamp_warp = emulated_times - original_times

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    axes[0].plot(relative_time, tau_truth, color="tab:blue")
    axes[0].set_ylabel("tau_I [s]")
    axes[0].set_title("Artificial temporal calibration truth")

    labels = ("rx", "ry", "rz", "tx", "ty", "tz")
    for component_index, label in enumerate(labels):
        axes[1].plot(relative_time, T_vectors[:, component_index], label=label)
    axes[1].set_ylabel("Ln(T_B_I)")
    axes[1].set_title("Artificial spatial calibration truth")
    axes[1].legend(loc="best", ncol=3, fontsize=8)

    axes[2].plot(relative_time, np.linalg.norm(gyro_original, axis=1), label="original")
    axes[2].plot(relative_time, np.linalg.norm(gyro_emulated, axis=1), label="emulated", alpha=0.85)
    axes[2].set_ylabel("||gyro|| [rad/s]")
    axes[2].set_title("Gyroscope norm")
    axes[2].legend(loc="best")

    axes[3].plot(relative_time, np.linalg.norm(accel_original, axis=1), label="original")
    axes[3].plot(relative_time, np.linalg.norm(accel_emulated, axis=1), label="emulated", alpha=0.85)
    axes[3].set_ylabel("||accel|| [m/s^2]")
    axes[3].set_title("Accelerometer norm")
    axes[3].legend(loc="best")

    axes[4].plot(relative_time, timestamp_warp, color="tab:purple")
    axes[4].set_ylabel("s_new - s_old [s]")
    axes[4].set_xlabel("time from dataset start [s]")
    axes[4].set_title("Timestamp warp")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_time_varying_calibration_tracking(
    results: object,
    *,
    T_B_I_truth: ArrayLike,
    tau_I_truth: ArrayLike,
    bias_truth: ArrayLike,
    T_B_L_truth: ArrayLike,
    tau_L_truth: float,
):
    '''Plot rolling calibration estimates against time-varying truth.

    Args:
        results: Iterable of ``CalibrationWindowResult`` objects.
        T_B_I_truth: Artificial ``T_B_I`` truth at window midpoints, shape ``(N, 4, 4)``.
        tau_I_truth: Artificial ``tau_I`` truth at window midpoints, shape ``(N,)``.
        bias_truth: Bias reference with shape ``(3,)``.
        T_B_L_truth: Constant ``T_B_L`` reference with shape ``(4, 4)``.
        tau_L_truth: Constant ``tau_L`` reference.

    Returns:
        tuple: ``(estimate_figure, estimate_axes, error_figure, error_axes)``.
    '''

    from .lie_se3 import se3_inverse, se3_log

    result_list = list(results)
    if len(result_list) == 0:
        raise ValueError("results must contain at least one window")

    window_numbers = np.asarray([result.window_index for result in result_list], dtype=int)
    window_midpoints = np.asarray([0.5 * (result.window_start + result.window_end) for result in result_list], dtype=float)
    T_I_truth = np.asarray(T_B_I_truth, dtype=float)
    tau_truth = np.asarray(tau_I_truth, dtype=float).reshape(-1)
    bias_ref = np.asarray(bias_truth, dtype=float).reshape(3)
    T_L_truth = np.asarray(T_B_L_truth, dtype=float)

    if T_I_truth.shape != (len(result_list), 4, 4):
        raise ValueError("T_B_I_truth must have shape (len(results), 4, 4)")
    if tau_truth.shape != (len(result_list),):
        raise ValueError("tau_I_truth must have shape (len(results),)")
    if T_L_truth.shape != (4, 4):
        raise ValueError("T_B_L_truth must have shape (4, 4)")

    # Extract the solved/end-of-window estimates already plotted by the previous
    # version of this helper.
    bias_estimates = np.vstack([result.bias_g for result in result_list])
    tau_estimates = np.column_stack([[result.tau_I for result in result_list], [result.tau_L for result in result_list]])
    tau_truth_block = np.column_stack([tau_truth, np.full_like(tau_truth, float(tau_L_truth))])
    T_B_I_vectors = np.vstack([se3_log(result.T_B_I) for result in result_list])
    T_B_I_truth_vectors = np.vstack([se3_log(transform) for transform in T_I_truth])
    T_B_L_vectors = np.vstack([se3_log(result.T_B_L) for result in result_list])
    T_B_L_truth_vector = se3_log(T_L_truth)

    # Extract optional numerical coarse-calibration initial values. Missing
    # values are represented by NaN so old result objects and disabled numerical
    # calibration runs remain plot-compatible.
    tau_I_initial = np.asarray(
        [
            np.nan if getattr(result, "numerical_tau_I_initial", None) is None else float(result.numerical_tau_I_initial)
            for result in result_list
        ],
        dtype=float,
    )
    tau_initial_block = np.column_stack([tau_I_initial, np.full_like(tau_I_initial, np.nan)])

    T_B_I_initial_vectors = np.full((len(result_list), 6), np.nan, dtype=float)
    for result_index, result in enumerate(result_list):
        numerical_T_B_I_initial = getattr(result, "numerical_T_B_I_initial", None)
        if numerical_T_B_I_initial is not None:
            T_B_I_initial_vectors[result_index] = se3_log(numerical_T_B_I_initial)

    has_tau_initial = np.any(np.isfinite(tau_I_initial))
    has_T_B_I_initial = np.any(np.isfinite(T_B_I_initial_vectors))

    def _plot_block(axis, estimates, truth, labels, title, ylabel, truth_is_time_varying, initials=None, initial_label="numerical initial"):
        for component_index, label in enumerate(labels):
            color = f"C{component_index % 10}"

            # Final estimate is the solved value after factor-graph optimization.
            axis.plot(window_numbers, estimates[:, component_index], marker="o", markersize=3, linewidth=1.4, color=color, label=f"{label} estimate")

            # Truth may be time-varying for the artificial IMU calibration, or a
            # constant reference for fixed quantities such as T_B_L and tau_L.
            if truth_is_time_varying:
                axis.plot(window_numbers, truth[:, component_index], linestyle="--", linewidth=1.2, color=color, alpha=0.8, label=f"{label} truth")
            else:
                axis.axhline(truth[component_index], linestyle="--", linewidth=1.1, color=color, alpha=0.8, label=f"{label} truth")

            # Numerical initial is the coarse-calibration starting point supplied
            # to the factor graph before local refinement. Plot only components
            # that are available for at least one window.
            if initials is not None and np.any(np.isfinite(initials[:, component_index])):
                axis.plot(
                    window_numbers,
                    initials[:, component_index],
                    marker="x",
                    markersize=5,
                    linestyle=":",
                    linewidth=1.2,
                    color=color,
                    alpha=0.95,
                    label=f"{label} {initial_label}",
                )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", ncol=2, fontsize=8)

    se3_labels = ("rx", "ry", "rz", "tx", "ty", "tz")
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    _plot_block(axes[0], bias_estimates, bias_ref, ("b_gx", "b_gy", "b_gz"), "Gyroscope bias estimate vs reference", "rad/s", False)
    _plot_block(
        axes[1],
        tau_estimates,
        tau_truth_block,
        ("tau_I", "tau_L"),
        "Temporal offset: numerical initial vs estimate vs truth",
        "s",
        True,
        initials=tau_initial_block if has_tau_initial else None,
    )
    _plot_block(
        axes[2],
        T_B_I_vectors,
        T_B_I_truth_vectors,
        tuple(f"T_B_I {label}" for label in se3_labels),
        "Body-from-IMU extrinsic Ln(T_B_I): numerical initial vs estimate vs truth",
        "SE(3) tangent [rad, m]",
        True,
        initials=T_B_I_initial_vectors if has_T_B_I_initial else None,
    )
    _plot_block(axes[3], T_B_L_vectors, T_B_L_truth_vector, tuple(f"T_B_L {label}" for label in se3_labels), "Body-from-LiDAR extrinsic Ln(T_B_L) estimate vs reference", "SE(3) tangent [rad, m]", False)
    axes[-1].set_xlabel("rolling window number")

    if window_numbers.size > 1:
        top_axis = axes[0].secondary_xaxis("top")
        top_axis.set_xlabel("window midpoint time [s]")
        tick_indices = np.linspace(0, window_numbers.size - 1, min(window_numbers.size, 8), dtype=int)
        top_axis.set_xticks(window_numbers[tick_indices])
        top_axis.set_xticklabels([f"{window_midpoints[index]:.1f}" for index in tick_indices])
    fig.tight_layout()

    # Error plots compare starting point and final estimate against truth. The
    # starting point is shown only when numerical initial values were recorded.
    rotation_errors = []
    translation_errors = []
    initial_rotation_errors = []
    initial_translation_errors = []
    for result, truth in zip(result_list, T_I_truth):
        delta = se3_inverse(truth) @ result.T_B_I
        xi = se3_log(delta)
        rotation_errors.append(np.linalg.norm(xi[:3]))
        translation_errors.append(np.linalg.norm(result.T_B_I[:3, 3] - truth[:3, 3]))

        numerical_T_B_I_initial = getattr(result, "numerical_T_B_I_initial", None)
        if numerical_T_B_I_initial is None:
            initial_rotation_errors.append(np.nan)
            initial_translation_errors.append(np.nan)
        else:
            initial_delta = se3_inverse(truth) @ numerical_T_B_I_initial
            initial_xi = se3_log(initial_delta)
            initial_rotation_errors.append(np.linalg.norm(initial_xi[:3]))
            initial_translation_errors.append(np.linalg.norm(numerical_T_B_I_initial[:3, 3] - truth[:3, 3]))

    error_fig, error_axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    error_axes[0].plot(window_numbers, tau_estimates[:, 0] - tau_truth, marker="o", label="estimate - truth")
    if has_tau_initial:
        error_axes[0].plot(window_numbers, tau_I_initial - tau_truth, marker="x", linestyle=":", label="numerical initial - truth")
    error_axes[0].set_ylabel("tau_I error [s]")
    error_axes[0].set_title("Temporal tracking error")
    error_axes[0].legend()

    error_axes[1].plot(window_numbers, rotation_errors, marker="o", color="tab:orange", label="estimate")
    if has_T_B_I_initial:
        error_axes[1].plot(window_numbers, initial_rotation_errors, marker="x", linestyle=":", color="tab:orange", alpha=0.85, label="numerical initial")
    error_axes[1].set_ylabel("rotation error [rad]")
    error_axes[1].set_title("T_B_I rotation tracking error")
    error_axes[1].legend()

    error_axes[2].plot(window_numbers, translation_errors, marker="o", color="tab:green", label="estimate")
    if has_T_B_I_initial:
        error_axes[2].plot(window_numbers, initial_translation_errors, marker="x", linestyle=":", color="tab:green", alpha=0.85, label="numerical initial")
    error_axes[2].set_ylabel("translation error [m]")
    error_axes[2].set_xlabel("rolling window number")
    error_axes[2].set_title("T_B_I translation tracking error")
    error_axes[2].legend()

    for axis in error_axes:
        axis.grid(True, alpha=0.3)
    error_fig.tight_layout()

    return fig, axes, error_fig, error_axes


def plot_numerical_calibration_window(result, *, tau_truth=None, title=None):
    '''Plot one numerical TwistnSync/SciPy calibration window diagnostic figure.

    Args:
        result: ``NumericalCalibrationResult`` returned by numerical calibration.
        tau_truth: Optional artificial truth tau used only for evaluation text.
        title: Optional figure title.

    Returns:
        Tuple ``(fig, axes)`` with a compact multi-panel diagnostic plot.
    '''

    # Prepare synchronized signals and norms. Failed windows still receive a small
    # placeholder figure so notebook loops do not need special plotting branches.
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=False)
    if not getattr(result, "success", False):
        axes[0].text(0.5, 0.5, f"Numerical calibration failed: {result.message}", ha="center", va="center", transform=axes[0].transAxes)
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        return fig, axes

    sync_t = np.asarray(result.synchronized_timestamps, dtype=float)
    relative_t = sync_t - sync_t[0]
    source_norm = np.linalg.norm(result.source_angvels, axis=1)
    reference_norm = np.linalg.norm(result.reference_angvels, axis=1)
    source_sync_norm = np.linalg.norm(result.source_angvels_synchronized, axis=1)
    reference_sync_norm = np.linalg.norm(result.reference_angvels_synchronized, axis=1)

    # Panel A: speed norms before temporal calibration, shown against local time axes.
    axes[0].plot(result.source_timestamps - result.source_timestamps[0], source_norm, label="||omega_I|| before")
    axes[0].plot(result.reference_timestamps - result.reference_timestamps[0], reference_norm, label="||omega_B|| before")
    axes[0].set_title("Before temporal calibration")
    axes[0].set_ylabel("rad/s")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Panel B: norms after applying tau_I in the factor convention.
    label = f"estimated tau_I={result.tau_I:+.4f}s"
    if tau_truth is not None:
        label += f", truth={float(tau_truth):+.4f}s"
    axes[1].plot(relative_t, source_sync_norm, label="||omega_I(t + tau_I)||")
    axes[1].plot(relative_t, reference_sync_norm, label="||omega_B(t)||")
    axes[1].set_title(label)
    axes[1].set_ylabel("rad/s")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Panel C: optional temporal score curve from fallback correlation diagnostics.
    diagnostics = getattr(result, "diagnostics", {}) or {}
    tau_grid = diagnostics.get("fallback_tau_grid")
    scores = diagnostics.get("fallback_scores")
    if tau_grid is not None and scores is not None:
        axes[2].plot(tau_grid, scores, color="tab:purple")
        axes[2].axvline(result.tau_I, color="black", linestyle="--", label="selected tau_I")
        if tau_truth is not None:
            axes[2].axvline(float(tau_truth), color="tab:green", linestyle=":", label="truth tau_I")
        axes[2].set_xlabel("tau_I [s]")
        axes[2].set_ylabel("score")
        axes[2].set_title("Temporal norm-correlation diagnostic")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No temporal score curve available", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Temporal diagnostic")
    axes[2].grid(True, alpha=0.3)

    # Panel D: synchronized vector components before/after spatial alignment.
    component_labels = ("x", "y", "z")
    for component_index, label_component in enumerate(component_labels):
        axes[3].plot(relative_t, result.reference_angvels_synchronized[:, component_index], color=f"C{component_index}", linewidth=1.3, label=f"omega_B {label_component}")
        axes[3].plot(relative_t, result.source_angvels_synchronized[:, component_index], color=f"C{component_index}", linestyle=":", alpha=0.55, label=f"omega_I {label_component}")
        axes[3].plot(relative_t, result.source_angvels_aligned[:, component_index], color=f"C{component_index}", linestyle="--", alpha=0.85, label=f"R_B_I omega_I {label_component}")
    axes[3].set_title("Synchronized angular velocity components")
    axes[3].set_ylabel("rad/s")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(ncol=3, fontsize=8)

    # Panel E: residual norm and excitation text.
    residual_norm = np.linalg.norm(result.residuals, axis=1)
    axes[4].plot(relative_t, residual_norm, color="tab:red", label="||omega_B - R_B_I omega_I||")
    s = result.excitation_singular_values
    r = result.excitation_ratios
    axes[4].set_title(f"Residual and excitation: s=[{s[0]:.3g}, {s[1]:.3g}, {s[2]:.3g}], s2/s1={r[0]:.3g}, s3/s1={r[1]:.3g}")
    axes[4].set_xlabel("time inside window [s]")
    axes[4].set_ylabel("rad/s")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    if title is not None:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_numerical_calibration_summary(summary_table):
    '''Plot five-window numerical calibration summary diagnostics.

    Args:
        summary_table: DataFrame-like object with tau, rotation-error, RSSD/RMSE,
            and excitation-ratio columns produced by notebook 16.

    Returns:
        Tuple ``(fig, axes)``.
    '''

    # Access through ``np.asarray`` so pandas DataFrames and dict-like tables both work.
    windows = np.asarray(summary_table["window"], dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Tau comparison answers whether the temporal coarse stage recovers each plateau.
    axes[0].plot(windows, np.asarray(summary_table["tau_truth"], dtype=float), marker="o", label="truth")
    axes[0].plot(windows, np.asarray(summary_table["tau_numerical"], dtype=float), marker="o", label="numerical")
    axes[0].set_ylabel("tau_I [s]")
    axes[0].set_title("Numerical temporal calibration")
    axes[0].legend()

    # Geodesic rotation error is the primary spatial metric.
    axes[1].plot(windows, np.asarray(summary_table["rotation_error_deg"], dtype=float), marker="o", color="tab:orange")
    axes[1].set_ylabel("rotation error [deg]")
    axes[1].set_title("R_B_I geodesic error")

    # Excitation ratios show whether the interval has multi-axis rotational content.
    axes[2].plot(windows, np.asarray(summary_table["s2_over_s1"], dtype=float), marker="o", label="s2/s1")
    axes[2].plot(windows, np.asarray(summary_table["s3_over_s1"], dtype=float), marker="o", label="s3/s1")
    axes[2].set_xlabel("window index")
    axes[2].set_ylabel("ratio")
    axes[2].set_title("Rotational excitation")
    axes[2].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_factor_graph_numerical_prior_comparison(comparison_table):
    '''Plot numerical-only, FG baseline, and FG numerical-prior comparison.

    Args:
        comparison_table: DataFrame-like object with truth/numerical/baseline/prior
            tau and rotation-error columns produced by notebook 16.

    Returns:
        Tuple ``(fig, axes)``.
    '''

    # The first two panels show the desired basin-improvement effect directly;
    # the last panel checks whether the graph optimization quality changed.
    windows = np.asarray(comparison_table["window"], dtype=int)
    fig, axes = plt.subplots(4, 1, figsize=(13, 15), sharex=True)

    for column, label in (
        ("tau_truth", "truth"),
        ("tau_numerical", "numerical"),
        ("tau_fg_baseline", "FG baseline"),
        ("tau_fg_numerical_prior", "FG numerical prior"),
    ):
        axes[0].plot(windows, np.asarray(comparison_table[column], dtype=float), marker="o", label=label)
    axes[0].set_ylabel("tau_I [s]")
    axes[0].set_title("Temporal calibration comparison")
    axes[0].legend()

    for column, label in (
        ("rotation_error_numerical_deg", "numerical"),
        ("rotation_error_fg_baseline_deg", "FG baseline"),
        ("rotation_error_fg_numerical_prior_deg", "FG numerical prior"),
    ):
        axes[1].plot(windows, np.asarray(comparison_table[column], dtype=float), marker="o", label=label)
    axes[1].set_ylabel("rotation error [deg]")
    axes[1].set_title("R_B_I geodesic error comparison")
    axes[1].legend()

    axes[2].plot(windows, np.asarray(comparison_table["spatial_rssd"], dtype=float), marker="o", label="RSSD")
    axes[2].plot(windows, np.asarray(comparison_table["vector_rmse"], dtype=float), marker="o", label="vector RMSE")
    axes[2].set_ylabel("rad/s")
    axes[2].set_title("Numerical calibration residual quality")
    axes[2].legend()

    for prefix, label in (("baseline", "FG baseline"), ("numerical_prior", "FG numerical prior")):
        axes[3].plot(windows, np.asarray(comparison_table[f"chi2_before_{prefix}"], dtype=float), marker="o", linestyle=":", label=f"{label} chi2 before")
        axes[3].plot(windows, np.asarray(comparison_table[f"chi2_after_{prefix}"], dtype=float), marker="o", linestyle="-", label=f"{label} chi2 after")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("window index")
    axes[3].set_ylabel("chi2")
    axes[3].set_title("Factor graph chi2")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes