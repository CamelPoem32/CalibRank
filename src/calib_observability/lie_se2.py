'''SE(2) Lie-group utilities for reduced planar reference analysis.

The tangent convention is ``xi = [omega, v_x, v_y]``. All adjoints,
Jacobians and perturbations follow the left-action convention
``T_perturbed = Exp(delta_xi) @ T``.
'''

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve

from .conventions import as_matrix, as_vector
from .lie_so3 import _mrob, _mrob_geometry_is_usable


##################################################
# Planar constants and MROB embedding helpers
##################################################
def _J2() -> NDArray[np.float64]:
    '''Return the 2D ninety-degree rotation generator.

    Returns:
        Matrix ``[[0, -1], [1, 0]]`` with shape ``(2, 2)``.
    '''

    return np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)


def _se2_to_se3_matrix(T: ArrayLike) -> NDArray[np.float64]:
    '''Embed an SE(2) homogeneous matrix into SE(3).

    Args:
        T: Planar transform with shape ``(3, 3)``.

    Returns:
        Embedded rigid transform with shape ``(4, 4)``.

    Raises:
        ValueError: If ``T`` is not a finite 3x3 matrix.
    '''

    M = as_matrix(T, (3, 3), "T")

    embedded_transform = np.eye(4)
    embedded_transform[:2, :2] = M[:2, :2]
    embedded_transform[:2, 3] = M[:2, 2]

    return embedded_transform


def _se3_to_se2_matrix(T: ArrayLike) -> NDArray[np.float64]:
    '''Extract the planar part of an embedded SE(3) transform.

    Args:
        T: Embedded transform with shape ``(4, 4)``.

    Returns:
        Planar homogeneous transform with shape ``(3, 3)``.

    Raises:
        ValueError: If ``T`` is not a finite 4x4 matrix.
    '''

    M = as_matrix(T, (4, 4), "T_SE3")

    planar_transform = np.eye(3)
    planar_transform[:2, :2] = M[:2, :2]
    planar_transform[:2, 2] = M[:2, 3]

    return planar_transform


def _se2_to_se3_tangent(xi: ArrayLike) -> NDArray[np.float64]:
    '''Embed a planar tangent into rotation-first SE(3) ordering.

    Args:
        xi: Planar tangent ``[omega, v_x, v_y]`` with shape ``(3,)``.

    Returns:
        SE(3) tangent ``[0, 0, omega, v_x, v_y, 0]`` with shape ``(6,)``.

    Raises:
        ValueError: If ``xi`` is not a finite three-dimensional vector.
    '''

    x = as_vector(xi, 3, "xi")

    embedded_tangent = np.zeros(6, dtype=float)
    embedded_tangent[2] = x[0]
    embedded_tangent[3] = x[1]
    embedded_tangent[4] = x[2]

    return embedded_tangent


def _se3_to_se2_tangent(xi: ArrayLike) -> NDArray[np.float64]:
    '''Extract a planar tangent from rotation-first SE(3) ordering.

    Args:
        xi: Embedded SE(3) tangent with shape ``(6,)``.

    Returns:
        Planar tangent ``[omega_z, v_x, v_y]`` with shape ``(3,)``.

    Raises:
        ValueError: If ``xi`` is not a finite six-dimensional vector.
    '''

    x = as_vector(xi, 6, "xi_SE3")
    return np.array([x[2], x[3], x[4]], dtype=float)


def _mrob_se3_from_planar_matrix(T: ArrayLike):
    '''Construct an MROB SE(3) object from an embedded planar matrix.

    Args:
        T: Planar transform accepted by :func:`_se2_to_se3_matrix`.

    Returns:
        An MROB ``SE3`` instance, or ``None`` when the optional backend is
        unavailable or rejects the value.
    '''

    if not _mrob_geometry_is_usable("SE3"):
        return None

    try:
        return _mrob.SE3(_se2_to_se3_matrix(T).astype(np.float64))
    except Exception:
        return None


def _mrob_se3_from_planar_tangent(xi: ArrayLike):
    '''Construct an MROB SE(3) object from an embedded planar tangent.

    Args:
        xi: Planar tangent accepted by :func:`_se2_to_se3_tangent`.

    Returns:
        An MROB ``SE3`` instance, or ``None`` when the optional backend is
        unavailable or rejects the value.
    '''

    if not _mrob_geometry_is_usable("SE3"):
        return None

    try:
        return _mrob.SE3(_se2_to_se3_tangent(xi).astype(np.float64))
    except Exception:
        return None


##################################################
# SO(2) and SE(2) algebra mappings
##################################################
def so2_exp(theta: float) -> NDArray[np.float64]:
    '''Compute an SO(2) rotation matrix.

    Args:
        theta: Rotation angle in radians.

    Returns:
        Rotation matrix with shape ``(2, 2)``.

    Raises:
        ValueError: If ``theta`` is not finite.
    '''

    if not np.isfinite(theta):
        raise ValueError("theta must be finite")

    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def se2_hat(xi: ArrayLike) -> NDArray[np.float64]:
    '''Map an SE(2) tangent to its 3x3 algebra matrix.

    Args:
        xi: Tangent ``[omega, v_x, v_y]`` with shape ``(3,)``.

    Returns:
        SE(2) algebra matrix with shape ``(3, 3)``.

    Raises:
        ValueError: If ``xi`` is not a finite three-dimensional vector.
    '''

    x = as_vector(xi, 3, "xi")

    # MROB does not expose a native SE(2) hat operation in this Python build.
    return np.array(
        [
            [0.0, -x[0], x[1]],
            [x[0], 0.0, x[2]],
            [0.0, 0.0, 0.0],
        ]
    )


def se2_vee(Xi: ArrayLike) -> NDArray[np.float64]:
    '''Map an SE(2) algebra matrix to ``[omega, v_x, v_y]``.

    Args:
        Xi: SE(2) algebra matrix with shape ``(3, 3)``.

    Returns:
        Planar tangent with shape ``(3,)``.

    Raises:
        ValueError: If ``Xi`` is not a finite 3x3 matrix.
    '''

    M = as_matrix(Xi, (3, 3), "Xi")

    # MROB does not expose a native SE(2) vee operation in Python.
    return np.array([M[1, 0], M[0, 2], M[1, 2]], dtype=float)


def _se2_V(theta: float) -> NDArray[np.float64]:
    '''Return the planar translation Jacobian used by the SE(2) exponential.

    Args:
        theta: Rotation component of the planar tangent.

    Returns:
        Matrix ``V(theta)`` with shape ``(2, 2)``.
    '''

    # Use Taylor coefficients close to zero to avoid removable singularities.
    if abs(theta) < 1e-9:
        A = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
        B = theta / 2.0 - theta**3 / 24.0 + theta**5 / 720.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta

    return np.array([[A, -B], [B, A]], dtype=float)


##################################################
# SE(2) group operations
##################################################
def se2_exp(xi: ArrayLike) -> NDArray[np.float64]:
    '''Compute the SE(2) exponential map ``Exp(xi)``.

    Args:
        xi: Tangent ``[omega, v_x, v_y]`` with shape ``(3,)``.

    Returns:
        Homogeneous planar transform with shape ``(3, 3)``.

    Raises:
        ValueError: If ``xi`` is not a finite three-dimensional vector.
    '''

    x = as_vector(xi, 3, "xi")
    mrob_transform = _mrob_se3_from_planar_tangent(x)

    if mrob_transform is not None:
        try:
            return _se3_to_se2_matrix(mrob_transform.T())
        except Exception:
            pass

    # Construct the planar rotation and translated exponential coordinates.
    T = np.eye(3)
    T[:2, :2] = so2_exp(float(x[0]))
    V = _se2_V(float(x[0]))
    T[:2, 2] = V @ x[1:]

    return T


def se2_log(T: ArrayLike) -> NDArray[np.float64]:
    '''Compute the principal SE(2) logarithm ``Log(T)``.

    Args:
        T: Homogeneous planar transform with shape ``(3, 3)``.

    Returns:
        Tangent ``[omega, v_x, v_y]`` with shape ``(3,)``.

    Raises:
        ValueError: If ``T`` is not a finite 3x3 matrix.
    '''

    M = as_matrix(T, (3, 3), "T")
    mrob_transform = _mrob_se3_from_planar_matrix(M)

    if mrob_transform is not None:
        try:
            mrob_log = np.asarray(mrob_transform.Ln(), dtype=float)
            if mrob_log.shape == (4, 4):
                return np.array(
                    [mrob_log[1, 0], mrob_log[0, 3], mrob_log[1, 3]],
                    dtype=float,
                )
            return _se3_to_se2_tangent(mrob_log.reshape(6))
        except Exception:
            pass

    # Recover rotation first, then invert the translation Jacobian.
    theta = float(np.arctan2(M[1, 0], M[0, 0]))
    V = _se2_V(theta)
    v = solve(V, M[:2, 2], assume_a="gen")

    return np.r_[theta, v]


def se2_inverse(T: ArrayLike) -> NDArray[np.float64]:
    '''Compute the analytic inverse of an SE(2) transform.

    Args:
        T: Homogeneous planar transform with shape ``(3, 3)``.

    Returns:
        Inverse transform with shape ``(3, 3)``.

    Raises:
        ValueError: If ``T`` is not a finite 3x3 matrix.
    '''

    M = as_matrix(T, (3, 3), "T")
    mrob_transform = _mrob_se3_from_planar_matrix(M)

    if mrob_transform is not None:
        try:
            return _se3_to_se2_matrix(mrob_transform.inv().T())
        except Exception:
            pass

    R = M[:2, :2]
    t = M[:2, 2]

    Ti = np.eye(3)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -(R.T @ t)

    return Ti


def se2_adjoint(T: ArrayLike) -> NDArray[np.float64]:
    '''Compute ``Adj(T)`` for tangents ``[omega, v_x, v_y]``.

    Args:
        T: Homogeneous planar transform with shape ``(3, 3)``.

    Returns:
        Group adjoint with shape ``(3, 3)``.

    Raises:
        ValueError: If ``T`` is not a finite 3x3 matrix.
    '''

    M = as_matrix(T, (3, 3), "T")
    mrob_transform = _mrob_se3_from_planar_matrix(M)

    if mrob_transform is not None:
        try:
            # Embed and extract only the planar coordinates of the SE(3) adjoint.
            embed_planar_tangent = np.zeros((6, 3), dtype=float)
            embed_planar_tangent[2, 0] = 1.0
            embed_planar_tangent[3, 1] = 1.0
            embed_planar_tangent[4, 2] = 1.0
            extract_planar_tangent = embed_planar_tangent.T

            return (
                extract_planar_tangent
                @ np.asarray(mrob_transform.adj(), dtype=float)
                @ embed_planar_tangent
            )
        except Exception:
            pass

    # Assemble the planar adjoint directly from rotation and translation.
    R = M[:2, :2]
    t = M[:2, 2]

    Adj = np.zeros((3, 3), dtype=float)
    Adj[0, 0] = 1.0
    Adj[1:, 0] = -_J2() @ t
    Adj[1:, 1:] = R

    return Adj


##################################################
# SE(2) algebra adjoint and left Jacobians
##################################################
def se2_little_adjoint(xi: ArrayLike) -> NDArray[np.float64]:
    '''Compute the SE(2) algebra adjoint ``ad_xi``.

    Args:
        xi: Tangent ``[omega, v_x, v_y]`` with shape ``(3,)``.

    Returns:
        Algebra adjoint with shape ``(3, 3)``.

    Raises:
        ValueError: If ``xi`` is not a finite three-dimensional vector.
    '''

    x = as_vector(xi, 3, "xi")

    ad = np.zeros((3, 3), dtype=float)
    ad[1:, 0] = -_J2() @ x[1:]
    ad[1:, 1:] = x[0] * _J2()

    return ad


def _se2_left_jacobian_series(
    ad: NDArray[np.float64],
    *,
    tolerance: float,
    max_terms: int,
) -> tuple[NDArray[np.float64], bool]:
    '''Evaluate the algebra-adjoint series for the SE(2) left Jacobian.

    Args:
        ad: Algebra adjoint with shape ``(3, 3)``.
        tolerance: Frobenius-norm convergence threshold.
        max_terms: Maximum number of series indices considered.

    Returns:
        Tuple ``(J, converged)`` containing the accumulated Jacobian and the
        convergence status.
    '''

    J = np.eye(3)
    power = np.eye(3)
    factorial = 1.0

    for n in range(1, max_terms):
        power = power @ ad
        factorial *= float(n + 1)
        term = power / factorial
        J = J + term

        if np.linalg.norm(term, ord="fro") < tolerance:
            return J, True

    return J, False


def se2_left_jacobian(
    xi: ArrayLike, *, tolerance: float = 1e-14, max_terms: int = 60
) -> NDArray[np.float64]:
    '''Compute the SE(2) left Jacobian by an algebra-adjoint series.

    Args:
        xi: Tangent ``[omega, v_x, v_y]`` with shape ``(3,)``.
        tolerance: Frobenius-norm convergence threshold.
        max_terms: Maximum number of series indices considered.

    Returns:
        Left Jacobian with shape ``(3, 3)``.

    Raises:
        ValueError: If ``xi`` is invalid or the requested series does not
            converge for a non-negligible tangent.
    '''

    x = as_vector(xi, 3, "xi")
    ad = se2_little_adjoint(x)
    J, converged = _se2_left_jacobian_series(
        ad,
        tolerance=tolerance,
        max_terms=max_terms,
    )

    if converged:
        return J

    # Preserve the accumulated small-tangent limit from the original code.
    if np.linalg.norm(x) < 1e-10:
        return J

    raise ValueError("SE(2) left-Jacobian series did not converge")


def se2_left_jacobian_inverse(xi: ArrayLike) -> NDArray[np.float64]:
    '''Compute the inverse SE(2) left Jacobian using a linear solve.

    Args:
        xi: Tangent ``[omega, v_x, v_y]`` with shape ``(3,)``.

    Returns:
        Inverse left Jacobian with shape ``(3, 3)``.

    Raises:
        ValueError: If ``xi`` is invalid or the Jacobian series does not
            converge.
    '''

    J = se2_left_jacobian(xi)
    return solve(J, np.eye(3), assume_a="gen")