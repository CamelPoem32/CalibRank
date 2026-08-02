"""Rigid-transform helpers for converting relative SE(3) poses to rates."""

from __future__ import annotations

import numpy as np


def _as_relative_pose_array(relative_poses_se3: np.ndarray) -> np.ndarray:
    poses = np.asarray(relative_poses_se3, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("relative_poses_se3 must have shape (N, 4, 4).")
    return poses


def _rotation_log(rotation_matrix: np.ndarray) -> np.ndarray:
    """Return the SO(3) logarithm vector for one 3x3 rotation matrix."""
    trace = float(np.trace(rotation_matrix))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    # First-order approximation keeps tiny rotations numerically stable.
    skew = 0.5 * (rotation_matrix - rotation_matrix.T)
    vee = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float)
    if theta < 1e-12:
        return vee

    sin_theta = float(np.sin(theta))
    if abs(sin_theta) < 1e-12:
        return theta * vee / max(np.linalg.norm(vee), 1e-12)
    return theta * vee / sin_theta


def _pose_durations(timestamps_s: np.ndarray, n_poses: int) -> np.ndarray:
    """Infer one positive duration per relative pose from scan or interval times."""
    times = np.asarray(timestamps_s, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("timestamps_s must not be empty.")
    if not np.all(np.isfinite(times)):
        raise ValueError("timestamps_s must contain only finite values.")

    # Prefer scan-boundary timestamps: N relative poses come from N+1 scans.
    if times.size == n_poses + 1:
        durations = np.diff(times)
    elif times.size == n_poses:
        if n_poses == 1:
            raise ValueError(
                "A single interval timestamp is ambiguous; pass two scan timestamps "
                "or provide velocities from a LidarData object with scan timestamps."
            )
        interval_steps = np.diff(times)
        typical_step = float(np.median(interval_steps[interval_steps > 0]))
        durations = np.concatenate([[typical_step], interval_steps])
    else:
        raise ValueError(
            "timestamps_s must have length N interval timestamps or N+1 scan timestamps."
        )

    if np.any(durations <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing.")
    return durations


def se3_to_angvels(relative_poses_se3: np.ndarray, timestamps_s: np.ndarray) -> np.ndarray:
    """Convert relative SE(3) rotations to angular velocity vectors.

    Args:
        relative_poses_se3: Array with shape ``(N, 4, 4)``. Each matrix is the
            relative motion from scan ``i`` to scan ``i + 1``.
        timestamps_s: Either ``N + 1`` scan timestamps in seconds or ``N``
            interval timestamps in seconds. Scan timestamps are preferred.

    Returns:
        ``(N, 3)`` array of angular velocity vectors in radians per second.
    """
    poses = _as_relative_pose_array(relative_poses_se3)
    durations = _pose_durations(timestamps_s, poses.shape[0])

    # Convert each incremental rotation to an axis-angle vector, then divide by dt.
    rotation_vectors = np.vstack([_rotation_log(pose[:3, :3]) for pose in poses])
    return rotation_vectors / durations[:, None]


def se3_to_velocities(relative_poses_se3: np.ndarray, timestamps_s: np.ndarray) -> np.ndarray:
    """Convert relative SE(3) translations to linear velocity vectors.

    Args:
        relative_poses_se3: Array with shape ``(N, 4, 4)``. Each matrix is the
            relative motion from scan ``i`` to scan ``i + 1``.
        timestamps_s: Either ``N + 1`` scan timestamps in seconds or ``N``
            interval timestamps in seconds. Scan timestamps are preferred.

    Returns:
        ``(N, 3)`` array of linear velocity vectors in meters per second.
    """
    poses = _as_relative_pose_array(relative_poses_se3)
    durations = _pose_durations(timestamps_s, poses.shape[0])

    # The translational part is already the scan-to-scan displacement.
    translations = poses[:, :3, 3]
    return translations / durations[:, None]
