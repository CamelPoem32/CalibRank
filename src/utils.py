# -*- coding: utf-8 -*-
"""
@file utils

Utilities for comparing absolute and relative SE(3) trajectories.
"""

from __future__ import annotations

from typing import Any, Sequence

import mrob
import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R

import transform


def _as_se3_array(se3s: Sequence[Any], name: str = "se3s") -> np.ndarray:
    """
    Convert SE(3) matrices or mrob.SE3 objects to an array with shape (N, 4, 4).
    """
    if isinstance(se3s, mrob.SE3):
        transformations = np.asarray(se3s.T(), dtype=float)[None, ...]
    else:
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
        raise ValueError(f"{name} must have shape (N, 4, 4)")

    if not np.all(np.isfinite(transformations)):
        raise ValueError(f"{name} must contain only finite values")

    return transformations


def _validate_se3_pair(se3s1: Sequence[Any], se3s2: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert and validate two equally sized SE(3) trajectories.
    """
    transformations1 = _as_se3_array(se3s1, "se3s1")
    transformations2 = _as_se3_array(se3s2, "se3s2")

    if len(transformations1) != len(transformations2):
        raise ValueError("se3s1 and se3s2 must have equal lengths")

    return transformations1, transformations2


def _validate_increment(increment: int, number_poses: int) -> int:
    """
    Validate the number of trajectory steps used for relative-pose errors.
    """
    increment = int(increment)

    if increment < 1:
        raise ValueError("increment must be a positive integer")

    if increment >= number_poses:
        raise ValueError("increment must be smaller than the trajectory length")

    return increment


def _se3_log(se3: np.ndarray) -> np.ndarray:
    """
    Map one SE(3) matrix to its six-dimensional tangent vector [omega, v].
    """
    return np.asarray(mrob.SE3(se3).Ln(), dtype=float).reshape(6)


def _so3_log(so3: np.ndarray) -> np.ndarray:
    """
    Map one SO(3) matrix to its three-dimensional rotation vector.
    """
    return np.asarray(mrob.SO3(so3).Ln(), dtype=float).reshape(3)


def se3_error_matrices(se3s1: Sequence[Any], se3s2: Sequence[Any]) -> np.ndarray:
    """
    Calculate pointwise right-invariant SE(3) error matrices.

    The error is defined as inv(T1) @ T2 for every pair of poses.
    """
    transformations1, transformations2 = _validate_se3_pair(se3s1, se3s2)

    return np.asarray([
        np.linalg.inv(T1) @ T2
        for T1, T2 in zip(transformations1, transformations2)
    ])


def se3_errors(se3s1: Sequence[Any], se3s2: Sequence[Any]) -> np.ndarray:
    """
    Calculate pointwise six-dimensional SE(3) errors [omega, v].

    Each row is Log(inv(T1) @ T2), using the MROB tangent ordering
    [rotation, translation].
    """
    error_matrices = se3_error_matrices(se3s1, se3s2)

    return np.asarray([
        _se3_log(error_matrix)
        for error_matrix in error_matrices
    ])


def calculate_se3_distances(se3s1: Sequence[Any], se3s2: Sequence[Any]) -> np.ndarray:
    """
    Calculate the full tangent-space distance between two SE(3) trajectories.

    The returned value is the Euclidean norm of Log(inv(T1) @ T2).
    Rotation is measured in radians and translation in meters, so this combined
    distance is mainly useful when both components have compatible weighting.
    """
    errors = se3_errors(se3s1, se3s2)

    return np.linalg.norm(errors, axis=1)


def calculate_se3_rotation_errors(se3s1: Sequence[Any], se3s2: Sequence[Any]) -> np.ndarray:
    """
    Calculate pointwise rotation-error vectors between two SE(3) trajectories.
    """
    transformations1, transformations2 = _validate_se3_pair(se3s1, se3s2)

    return np.asarray([
        _so3_log(T1[:3, :3].T @ T2[:3, :3])
        for T1, T2 in zip(transformations1, transformations2)
    ])


def calculate_se3_rotation_distances(se3s1: Sequence[Any], se3s2: Sequence[Any]) -> np.ndarray:
    """
    Calculate pointwise rotation distances in radians.
    """
    rotation_errors = calculate_se3_rotation_errors(se3s1, se3s2)

    return np.linalg.norm(rotation_errors, axis=1)


def calculate_se3_translation_errors(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    frame: str = "world",
) -> np.ndarray:
    """
    Calculate pointwise translation-error vectors.

    Args:
        se3s1: First SE(3) trajectory with shape (N, 4, 4).
        se3s2: Second SE(3) trajectory with shape (N, 4, 4).
        frame: ``"world"`` returns p2 - p1 in world coordinates.
            ``"body"`` expresses the same displacement in the frame of T1.

    Returns:
        Translation errors with shape (N, 3).
    """
    transformations1, transformations2 = _validate_se3_pair(se3s1, se3s2)
    translation_errors_world = transformations2[:, :3, 3] - transformations1[:, :3, 3]

    if frame == "world":
        return translation_errors_world

    if frame == "body":
        rotations_world_body = transformations1[:, :3, :3]
        return np.einsum(
            "nij,nj->ni",
            rotations_world_body.transpose(0, 2, 1),
            translation_errors_world,
        )

    raise ValueError("frame must be 'world' or 'body'")


def calculate_se3_translation_distances(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    frame: str = "world",
) -> np.ndarray:
    """
    Calculate pointwise translation distances in meters.
    """
    translation_errors = calculate_se3_translation_errors(se3s1, se3s2, frame=frame)

    return np.linalg.norm(translation_errors, axis=1)


def se3_to_relative_se3(se3s: Sequence[Any], increment: int = 1) -> np.ndarray:
    """
    Convert an absolute SE(3) trajectory to relative pose changes.

    Each output is inv(T[i]) @ T[i + increment].
    """
    transformations = _as_se3_array(se3s)
    increment = _validate_increment(increment, len(transformations))

    return np.asarray([
        np.linalg.inv(transformations[i]) @ transformations[i + increment]
        for i in range(len(transformations) - increment)
    ])


def relative_se3_error_matrices(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate relative-pose error matrices between two SE(3) trajectories.

    For each interval, the error is inv(delta_T1) @ delta_T2.
    """
    transformations1, transformations2 = _validate_se3_pair(se3s1, se3s2)
    increment = _validate_increment(increment, len(transformations1))

    relative1 = se3_to_relative_se3(transformations1, increment)
    relative2 = se3_to_relative_se3(transformations2, increment)

    return np.asarray([
        np.linalg.inv(delta_T1) @ delta_T2
        for delta_T1, delta_T2 in zip(relative1, relative2)
    ])


def relative_se3_errors(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate six-dimensional relative-pose errors [omega, v].
    """
    error_matrices = relative_se3_error_matrices(se3s1, se3s2, increment)

    return np.asarray([
        _se3_log(error_matrix)
        for error_matrix in error_matrices
    ])


def calculate_relative_se3_distances(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate full tangent-space relative-pose distances.
    """
    errors = relative_se3_errors(se3s1, se3s2, increment)

    return np.linalg.norm(errors, axis=1)


def calculate_relative_se3_rotation_errors(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate relative rotation-error vectors in radians.
    """
    error_matrices = relative_se3_error_matrices(se3s1, se3s2, increment)

    return np.asarray([
        _so3_log(error_matrix[:3, :3])
        for error_matrix in error_matrices
    ])


def calculate_relative_se3_rotation_distances(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate relative rotation distances in radians.
    """
    rotation_errors = calculate_relative_se3_rotation_errors(se3s1, se3s2, increment)

    return np.linalg.norm(rotation_errors, axis=1)


def calculate_relative_se3_translation_errors(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate translation components of the relative-pose error matrices.

    Each vector is expressed in the source frame of the first relative motion.
    """
    error_matrices = relative_se3_error_matrices(se3s1, se3s2, increment)

    return error_matrices[:, :3, 3].copy()


def calculate_relative_se3_translation_distances(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate relative translation distances in meters.
    """
    translation_errors = calculate_relative_se3_translation_errors(se3s1, se3s2, increment)

    return np.linalg.norm(translation_errors, axis=1)


def APE_SE3(se3s1: Sequence[Any], se3s2: Sequence[Any]) -> np.ndarray:
    """
    Calculate full absolute pose error as a tangent-space norm.
    """
    return calculate_se3_distances(se3s1, se3s2)


def APE_SE3_rotation(se3s1: Sequence[Any], se3s2: Sequence[Any]) -> np.ndarray:
    """
    Calculate absolute rotation error in radians.
    """
    return calculate_se3_rotation_distances(se3s1, se3s2)


def APE_SE3_translation(se3s1: Sequence[Any], se3s2: Sequence[Any]) -> np.ndarray:
    """
    Calculate absolute translation error in meters.
    """
    return calculate_se3_translation_distances(se3s1, se3s2)


def RPE_SE3(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate full relative pose error as a tangent-space norm.
    """
    return calculate_relative_se3_distances(se3s1, se3s2, increment)


def RPE_SE3_rotation(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate relative rotation error in radians.
    """
    return calculate_relative_se3_rotation_distances(se3s1, se3s2, increment)


def RPE_SE3_translation(
    se3s1: Sequence[Any],
    se3s2: Sequence[Any],
    increment: int = 1,
) -> np.ndarray:
    """
    Calculate relative translation error in meters.
    """
    return calculate_relative_se3_translation_distances(se3s1, se3s2, increment)

@njit
def wrap_angle(angle):
    """
    Wraps the given angle to the range [-pi, +pi].

    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """

    pi2 = 2 * np.pi

    while angle < -np.pi:
        angle += pi2

    while angle >= np.pi:
        angle -= pi2

    return angle

@njit
def wrap_angles(angles):
    for i in range(len(angles)):
        angles[i] = wrap_angle(angles[i])
    return angles

def scipy_rotation(w1, w2):
    '''
    Finds best rotation matrix 1to2 (det=1): w2 = M @ w1

    param: w1, w2 (array) - arrays of the same shape

    return: (array 3x3) M
    '''
    M_1_to_2, rssd = R.align_vectors(w2, w1)
    M_1_to_2 = M_1_to_2.as_matrix()

    return M_1_to_2

def steady_samples_number(data, max_idle_val=5e-2, axis=-1):
    norms = np.linalg.norm(data, axis=axis)
    last_steady = np.argmax(norms >= max_idle_val)
    if last_steady == 0 and np.all(norms < max_idle_val): last_steady = len(data)
    return last_steady

@njit
def RMSE(errors):
    '''
    Calculate Root Mean Squared of the given errors array
    '''
    pow = np.power(errors, 2)
    mean = np.array([np.mean(pow[:, i]) for i in range(len(pow[0]))])
    return np.sqrt(mean)

def rotate_quats(quats, R, right=False):
    '''
    Left multiplication of SO3(quat) by R -> R @ SO3(quat)
    
    param: quats (array Nx4) - quaternions [w, x, y ,z]
    param: R (array 3x3) - rotation matrix
    param: right (bool) - is matmul right (SO3(q) @ R) or left (R @ SO3(q))? Default=False

    return: (array Nx4) qs [w, x, y ,z]
    '''
    qs = np.empty_like(quats)
    Rs = transform.quats_to_so3s(quats)
    Rs_rot = np.empty_like(Rs)
    for i, q in enumerate(quats):
        if right: 
            Rs_rot[i] = Rs[i] @ R
        else:
            Rs_rot[i] = R @ Rs[i]
        
    qs = transform.so3s_to_quats(Rs_rot)
    
    return qs