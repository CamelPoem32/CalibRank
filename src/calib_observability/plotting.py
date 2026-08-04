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
