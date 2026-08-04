"""SO(3) and SE(3) helpers for poses, relative motion, and rates."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import mrob
import numpy as np


def _as_timestamps(timestamps_s: Sequence[float], expected_length: Optional[int] = None) -> np.ndarray:
    """Convert timestamps to one strictly increasing one-dimensional array."""
    timestamps = np.asarray(timestamps_s, dtype=float).reshape(-1)

    if expected_length is not None and len(timestamps) != expected_length:
        raise ValueError(f"timestamps_s must contain {expected_length} values")

    if len(timestamps) == 0:
        raise ValueError("timestamps_s must not be empty")

    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps_s must contain only finite values")

    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing")

    return timestamps


def _as_so3_array(so3s: Sequence[Any]) -> np.ndarray:
    """Convert SO(3) matrices or mrob.SO3 objects to shape (N, 3, 3)."""
    if isinstance(so3s, mrob.SO3):
        return np.asarray(so3s.R(), dtype=float)[None, ...]

    rotations = []

    for so3 in so3s:
        if isinstance(so3, mrob.SO3):
            rotations.append(np.asarray(so3.R(), dtype=float))
        else:
            rotations.append(np.asarray(so3, dtype=float))

    rotations = np.asarray(rotations, dtype=float)

    if rotations.ndim == 2 and rotations.shape == (3, 3):
        rotations = rotations[None, ...]

    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("so3s must have shape (N, 3, 3)")

    if not np.all(np.isfinite(rotations)):
        raise ValueError("so3s must contain only finite values")

    return rotations


def _as_se3_array(se3s: Sequence[Any]) -> np.ndarray:
    """Convert SE(3) matrices or mrob.SE3 objects to shape (N, 4, 4)."""
    if isinstance(se3s, mrob.SE3):
        return np.asarray(se3s.T(), dtype=float)[None, ...]

    transformations = []

    for se3 in se3s:
        if isinstance(se3, mrob.SE3):
            transformations.append(np.asarray(se3.T(), dtype=float))
        else:
            transformations.append(np.asarray(se3, dtype=float))

    transformations = np.asarray(transformations, dtype=float)

    if transformations.ndim == 2 and transformations.shape == (4, 4):
        transformations = transformations[None, ...]

    if transformations.ndim != 3 or transformations.shape[1:] != (4, 4):
        raise ValueError("se3s must have shape (N, 4, 4)")

    if not np.all(np.isfinite(transformations)):
        raise ValueError("se3s must contain only finite values")

    return transformations


def _as_vectors(values: Sequence[float], dimension: int, name: str) -> np.ndarray:
    """Convert a vector sequence to shape (N, dimension)."""
    vectors = np.asarray(values, dtype=float)

    if vectors.ndim == 1 and vectors.shape == (dimension,):
        vectors = vectors[None, ...]

    if vectors.ndim != 2 or vectors.shape[1] != dimension:
        raise ValueError(f"{name} must have shape (N, {dimension})")

    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} must contain only finite values")

    return vectors


def _relative_durations(timestamps_s: Sequence[float], number_relative_states: int) -> np.ndarray:
    """Infer one positive duration for every relative state."""
    timestamps = _as_timestamps(timestamps_s)

    # Prefer absolute-state timestamps: M relative motions come from M + 1 states.
    if len(timestamps) == number_relative_states + 1:
        return np.diff(timestamps)

    # Preserve support for one timestamp associated with every relative motion.
    if len(timestamps) == number_relative_states:
        if number_relative_states == 1:
            raise ValueError("One relative state requires two absolute-state timestamps")

        interval_steps = np.diff(timestamps)
        first_step = float(np.median(interval_steps))
        return np.concatenate(([first_step], interval_steps))

    raise ValueError("timestamps_s must contain M or M + 1 values for M relative states")


def so3_to_so3_vectors(so3s: Sequence[Any]) -> np.ndarray:
    """Map SO(3) matrices to three-dimensional rotation vectors."""
    rotations = _as_so3_array(so3s)

    return np.asarray([mrob.SO3(rotation).Ln() for rotation in rotations])


def so3_vectors_to_so3(so3_vectors: Sequence[float]) -> np.ndarray:
    """Map three-dimensional rotation vectors to SO(3) matrices."""
    so3_vectors = _as_vectors(so3_vectors, 3, "so3_vectors")

    return np.asarray([mrob.SO3(vector).R() for vector in so3_vectors])


def se3_to_se3_vectors(se3s: Sequence[Any]) -> np.ndarray:
    """Map SE(3) matrices to six-dimensional tangent vectors."""
    transformations = _as_se3_array(se3s)

    return np.asarray([mrob.SE3(transformation).Ln() for transformation in transformations])


def se3_vectors_to_se3(se3_vectors: Sequence[float]) -> np.ndarray:
    """Map six-dimensional tangent vectors to SE(3) matrices."""
    se3_vectors = _as_vectors(se3_vectors, 6, "se3_vectors")

    return np.asarray([mrob.SE3(vector).T() for vector in se3_vectors])


def se3_to_poses(se3s: Sequence[Any]) -> np.ndarray:
    """Extract Cartesian pose coordinates from SE(3) matrices."""
    transformations = _as_se3_array(se3s)

    return transformations[:, :3, 3].copy()


def se3_vectors_to_poses(se3_vectors: Sequence[float]) -> np.ndarray:
    """Convert SE(3) tangent vectors to Cartesian pose coordinates."""
    return se3_to_poses(se3_vectors_to_se3(se3_vectors))


def poses_to_se3(poses: Sequence[float], so3s: Optional[Sequence[Any]] = None) -> np.ndarray:
    """Create SE(3) matrices from Cartesian coordinates and optional SO(3) matrices."""
    poses = _as_vectors(poses, 3, "poses")

    if so3s is None:
        rotations = np.repeat(np.eye(3)[None, ...], len(poses), axis=0)
    else:
        rotations = _as_so3_array(so3s)

    if len(rotations) != len(poses):
        raise ValueError("poses and so3s must have equal lengths")

    transformations = np.repeat(np.eye(4)[None, ...], len(poses), axis=0)
    transformations[:, :3, :3] = rotations
    transformations[:, :3, 3] = poses

    return transformations


def poses_to_se3_vectors(poses: Sequence[float], so3s: Optional[Sequence[Any]] = None) -> np.ndarray:
    """Create SE(3) tangent vectors from Cartesian coordinates and optional rotations."""
    return se3_to_se3_vectors(poses_to_se3(poses, so3s))


def so3_to_relative_so3(so3s: Sequence[Any]) -> np.ndarray:
    """Convert absolute SO(3) matrices to consecutive body-frame rotations."""
    rotations = _as_so3_array(so3s)

    if len(rotations) < 2:
        return np.empty((0, 3, 3))

    return np.asarray(
        [
            rotations[i].T @ rotations[i + 1]
            for i in range(len(rotations) - 1)
        ]
    )


def so3_vectors_to_relative_so3(so3_vectors: Sequence[float]) -> np.ndarray:
    """Convert absolute SO(3) vectors to consecutive relative SO(3) matrices."""
    return so3_to_relative_so3(so3_vectors_to_so3(so3_vectors))


def relative_so3_to_so3(relative_so3s: Sequence[Any], first_so3: Optional[Any] = None) -> np.ndarray:
    """Accumulate relative SO(3) matrices into absolute SO(3) matrices."""
    relative_rotations = _as_so3_array(relative_so3s)

    if first_so3 is None:
        first_rotation = np.eye(3)
    else:
        first_rotation = _as_so3_array(first_so3)[0]

    rotations = np.empty((len(relative_rotations) + 1, 3, 3))
    rotations[0] = first_rotation

    for i, relative_rotation in enumerate(relative_rotations):
        rotations[i + 1] = rotations[i] @ relative_rotation

    return rotations


def relative_so3_vectors_to_so3(
    relative_so3_vectors: Sequence[float],
    first_so3: Optional[Any] = None,
) -> np.ndarray:
    """Accumulate relative SO(3) vectors into absolute SO(3) matrices."""
    relative_so3s = so3_vectors_to_so3(relative_so3_vectors)

    return relative_so3_to_so3(relative_so3s, first_so3)


def se3_to_relative_se3(se3s: Sequence[Any]) -> np.ndarray:
    """Convert absolute SE(3) matrices to consecutive body-frame relative matrices."""
    transformations = _as_se3_array(se3s)

    if len(transformations) < 2:
        return np.empty((0, 4, 4))

    relative_transformations = np.empty((len(transformations) - 1, 4, 4))

    for i in range(len(relative_transformations)):
        relative_transformations[i] = np.linalg.inv(transformations[i]) @ transformations[i + 1]

    return relative_transformations


def se3_vectors_to_relative_se3(se3_vectors: Sequence[float]) -> np.ndarray:
    """Convert absolute SE(3) vectors to consecutive relative SE(3) matrices."""
    return se3_to_relative_se3(se3_vectors_to_se3(se3_vectors))


def se3_to_relative_se3_vectors(se3s: Sequence[Any]) -> np.ndarray:
    """Convert absolute SE(3) matrices to consecutive relative tangent vectors."""
    relative_se3s = se3_to_relative_se3(se3s)

    return se3_to_se3_vectors(relative_se3s)


def relative_se3_to_se3(
    relative_se3s: Sequence[Any],
    first_se3: Optional[Any] = None,
) -> np.ndarray:
    """Accumulate relative SE(3) matrices into absolute SE(3) matrices."""
    relative_transformations = _as_se3_array(relative_se3s)

    if first_se3 is None:
        first_transformation = np.eye(4)
    else:
        first_transformation = _as_se3_array(first_se3)[0]

    transformations = np.empty((len(relative_transformations) + 1, 4, 4))
    transformations[0] = first_transformation

    for i, relative_transformation in enumerate(relative_transformations):
        transformations[i + 1] = transformations[i] @ relative_transformation

    return transformations


def relative_se3_vectors_to_se3(
    relative_se3_vectors: Sequence[float],
    first_se3: Optional[Any] = None,
) -> np.ndarray:
    """Accumulate relative SE(3) vectors into absolute SE(3) matrices."""
    relative_se3s = se3_vectors_to_se3(relative_se3_vectors)

    return relative_se3_to_se3(relative_se3s, first_se3)


def relative_se3_to_poses(
    relative_se3s: Sequence[Any],
    first_se3: Optional[Any] = None,
) -> np.ndarray:
    """Accumulate relative SE(3) matrices and return Cartesian pose coordinates."""
    se3s = relative_se3_to_se3(relative_se3s, first_se3)

    return se3_to_poses(se3s)


def relative_se3_vectors_to_poses(
    relative_se3_vectors: Sequence[float],
    first_se3: Optional[Any] = None,
) -> np.ndarray:
    """Accumulate relative SE(3) vectors and return Cartesian pose coordinates."""
    se3s = relative_se3_vectors_to_se3(relative_se3_vectors, first_se3)

    return se3_to_poses(se3s)


def relative_so3_to_angvels(
    relative_so3s: Sequence[Any],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert relative SO(3) matrices to body-frame angular velocities."""
    relative_rotations = _as_so3_array(relative_so3s)
    durations = _relative_durations(timestamps_s, len(relative_rotations))
    rotation_vectors = so3_to_so3_vectors(relative_rotations)

    return rotation_vectors / durations[:, None]


def relative_so3_vectors_to_angvels(
    relative_so3_vectors: Sequence[float],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert relative SO(3) vectors to body-frame angular velocities."""
    relative_so3_vectors = _as_vectors(
        relative_so3_vectors,
        3,
        "relative_so3_vectors",
    )
    durations = _relative_durations(timestamps_s, len(relative_so3_vectors))

    return relative_so3_vectors / durations[:, None]


def so3_to_angvels(
    so3s: Sequence[Any],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert absolute SO(3) matrices to body-frame angular velocities."""
    rotations = _as_so3_array(so3s)
    _as_timestamps(timestamps_s, len(rotations))

    relative_rotations = so3_to_relative_so3(rotations)

    return relative_so3_to_angvels(relative_rotations, timestamps_s)


def so3_vectors_to_angvels(
    so3_vectors: Sequence[float],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert absolute SO(3) vectors to body-frame angular velocities."""
    so3s = so3_vectors_to_so3(so3_vectors)

    return so3_to_angvels(so3s, timestamps_s)


def relative_se3_to_angvels(
    relative_se3s: Sequence[Any],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert relative SE(3) matrices to body-frame angular velocities."""
    relative_transformations = _as_se3_array(relative_se3s)
    relative_rotations = relative_transformations[:, :3, :3]

    return relative_so3_to_angvels(relative_rotations, timestamps_s)


def relative_se3_vectors_to_angvels(
    relative_se3_vectors: Sequence[float],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert relative SE(3) vectors to body-frame angular velocities."""
    relative_se3_vectors = _as_vectors(
        relative_se3_vectors,
        6,
        "relative_se3_vectors",
    )
    durations = _relative_durations(timestamps_s, len(relative_se3_vectors))

    return relative_se3_vectors[:, :3] / durations[:, None]


def se3_to_angvels(
    se3s: Sequence[Any],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert absolute SE(3) matrices to body-frame angular velocities."""
    transformations = _as_se3_array(se3s)
    _as_timestamps(timestamps_s, len(transformations))

    relative_transformations = se3_to_relative_se3(transformations)

    return relative_se3_to_angvels(relative_transformations, timestamps_s)


def se3_vectors_to_angvels(
    se3_vectors: Sequence[float],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert absolute SE(3) vectors to body-frame angular velocities."""
    se3s = se3_vectors_to_se3(se3_vectors)

    return se3_to_angvels(se3s, timestamps_s)


def relative_se3_to_velocities(
    relative_se3s: Sequence[Any],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert relative SE(3) matrices to source-frame linear velocities."""
    relative_transformations = _as_se3_array(relative_se3s)
    durations = _relative_durations(timestamps_s, len(relative_transformations))

    return relative_transformations[:, :3, 3] / durations[:, None]


def relative_se3_vectors_to_velocities(
    relative_se3_vectors: Sequence[float],
    timestamps_s: Sequence[float],
) -> np.ndarray:
    """Convert relative SE(3) vectors to source-frame linear velocities."""
    relative_se3s = se3_vectors_to_se3(relative_se3_vectors)

    return relative_se3_to_velocities(relative_se3s, timestamps_s)


def se3_to_velocities(
    se3s: Sequence[Any],
    timestamps_s: Sequence[float],
    frame: str = "world",
) -> np.ndarray:
    """Convert absolute SE(3) matrices to linear velocities.

    Args:
        se3s: Absolute SE(3) matrices with shape ``(N, 4, 4)``.
        timestamps_s: One timestamp for every absolute transformation.
        frame: ``"world"`` for world-frame finite differences or ``"body"``
            for the same displacement expressed in the source body frame.

    Returns:
        Linear velocities with shape ``(N - 1, 3)``.
    """
    transformations = _as_se3_array(se3s)
    timestamps = _as_timestamps(timestamps_s, len(transformations))

    durations = np.diff(timestamps)
    displacements_world = np.diff(transformations[:, :3, 3], axis=0)

    if frame == "world":
        return displacements_world / durations[:, None]

    if frame == "body":
        rotations_world_body = transformations[:-1, :3, :3]
        displacements_body = np.einsum(
            "nij,nj->ni",
            rotations_world_body.transpose(0, 2, 1),
            displacements_world,
        )

        return displacements_body / durations[:, None]

    raise ValueError("frame must be 'world' or 'body'")


def se3_vectors_to_velocities(
    se3_vectors: Sequence[float],
    timestamps_s: Sequence[float],
    frame: str = "world",
) -> np.ndarray:
    """Convert absolute SE(3) vectors to linear velocities."""
    se3s = se3_vectors_to_se3(se3_vectors)

    return se3_to_velocities(se3s, timestamps_s, frame)


def angvels_to_so3(
    timestamps_s: Sequence[float],
    angular_velocities: Sequence[float],
    first_so3: Optional[Any] = None,
) -> np.ndarray:
    """Integrate body-frame angular velocities into absolute SO(3) matrices."""
    angular_velocities = _as_vectors(
        angular_velocities,
        3,
        "angular_velocities",
    )
    timestamps = _as_timestamps(
        timestamps_s,
        len(angular_velocities) + 1,
    )

    if first_so3 is None:
        first_rotation = np.eye(3)
    else:
        first_rotation = _as_so3_array(first_so3)[0]

    rotations = np.empty((len(timestamps), 3, 3))
    rotations[0] = first_rotation

    for i, dt in enumerate(np.diff(timestamps)):
        delta_rotation = mrob.SO3(angular_velocities[i] * dt).R()
        rotations[i + 1] = rotations[i] @ delta_rotation

    return rotations


def angvels_to_so3_vectors(
    timestamps_s: Sequence[float],
    angular_velocities: Sequence[float],
    first_so3: Optional[Any] = None,
) -> np.ndarray:
    """Integrate angular velocities and return absolute SO(3) vectors."""
    so3s = angvels_to_so3(
        timestamps_s,
        angular_velocities,
        first_so3,
    )

    return so3_to_so3_vectors(so3s)


def velocities_to_se3(
    timestamps_s: Sequence[float],
    linear_velocities: Sequence[float],
    angular_velocities: Optional[Sequence[float]] = None,
    first_se3: Optional[Any] = None,
    velocity_frame: str = "world",
) -> np.ndarray:
    """Integrate linear and optional angular velocities into absolute SE(3) matrices."""
    linear_velocities = _as_vectors(
        linear_velocities,
        3,
        "linear_velocities",
    )
    timestamps = _as_timestamps(
        timestamps_s,
        len(linear_velocities) + 1,
    )

    if angular_velocities is None:
        angular_velocities = np.zeros_like(linear_velocities)
    else:
        angular_velocities = _as_vectors(
            angular_velocities,
            3,
            "angular_velocities",
        )

    if len(angular_velocities) != len(linear_velocities):
        raise ValueError(
            "linear_velocities and angular_velocities must have equal lengths"
        )

    if first_se3 is None:
        first_transformation = np.eye(4)
    else:
        first_transformation = _as_se3_array(first_se3)[0]

    transformations = np.empty((len(timestamps), 4, 4))
    transformations[0] = first_transformation

    for i, dt in enumerate(np.diff(timestamps)):
        transformations[i + 1] = transformations[i]

        delta_rotation = mrob.SO3(angular_velocities[i] * dt).R()
        transformations[i + 1, :3, :3] = (
            transformations[i, :3, :3] @ delta_rotation
        )

        if velocity_frame == "world":
            displacement_world = linear_velocities[i] * dt
        elif velocity_frame == "body":
            displacement_world = (
                transformations[i, :3, :3]
                @ linear_velocities[i]
                * dt
            )
        else:
            raise ValueError(
                "velocity_frame must be 'world' or 'body'"
            )

        transformations[i + 1, :3, 3] = (
            transformations[i, :3, 3]
            + displacement_world
        )

    return transformations


def velocities_to_se3_vectors(
    timestamps_s: Sequence[float],
    linear_velocities: Sequence[float],
    angular_velocities: Optional[Sequence[float]] = None,
    first_se3: Optional[Any] = None,
    velocity_frame: str = "world",
) -> np.ndarray:
    """Integrate velocities and return absolute SE(3) tangent vectors."""
    se3s = velocities_to_se3(
        timestamps_s,
        linear_velocities,
        angular_velocities,
        first_se3,
        velocity_frame,
    )

    return se3_to_se3_vectors(se3s)