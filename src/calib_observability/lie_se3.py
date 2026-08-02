'''SE(3) Lie-group utilities with rotation-first tangents.

The tangent convention is
``xi = [omega_x, omega_y, omega_z, v_x, v_y, v_z]``. Every operation is
compatible with the left-perturbation rule
``T_perturbed = Exp(delta_xi) @ T``.
'''

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve

from .conventions import as_matrix, as_vector
from .lie_so3 import (
    _mrob,
    _mrob_geometry_is_usable,
    so3_exp,
    so3_hat,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
)


##################################################
# Optional MROB geometry backend
##################################################
def _mrob_se3_from_matrix(T: ArrayLike):
    '''Construct an MROB SE(3) object from a transform when possible.

    Args:
        T: Homogeneous transform accepted by the MROB constructor.

    Returns:
        An MROB ``SE3`` instance, or ``None`` when the backend is unavailable
        or rejects the input.
    '''

    if not _mrob_geometry_is_usable("SE3"):
        return None

    try:
        return _mrob.SE3(np.asarray(T, dtype=np.float64))
    except Exception:
        return None


def _mrob_se3_from_tangent(xi: ArrayLike):
    '''Construct an MROB SE(3) object from a tangent when possible.

    Args:
        xi: Rotation-first tangent accepted by the MROB constructor.

    Returns:
        An MROB ``SE3`` instance, or ``None`` when the backend is unavailable
        or rejects the input.
    '''

    if not _mrob_geometry_is_usable("SE3"):
        return None

    try:
        return _mrob.SE3(np.asarray(xi, dtype=np.float64))
    except Exception:
        return None


##################################################
# SE(3) algebra mappings
##################################################
def se3_hat(xi: ArrayLike) -> NDArray[np.float64]:
    '''Map a rotation-first SE(3) tangent to a 4x4 algebra matrix.

    Args:
        xi: Tangent ``[omega, v]`` with shape ``(6,)``.

    Returns:
        SE(3) algebra matrix with shape ``(4, 4)``.

    Raises:
        ValueError: If ``xi`` is not a finite six-dimensional vector.
    '''

    x = as_vector(xi, 6, "xi")

    if _mrob_geometry_is_usable("SE3"):
        try:
            return np.asarray(_mrob.hat6(x), dtype=float)
        except Exception:
            pass

    # Explicit fallback for unavailable or unsafe MROB bindings.
    Xi = np.zeros((4, 4), dtype=float)
    Xi[:3, :3] = so3_hat(x[:3])
    Xi[:3, 3] = x[3:]

    return Xi


def se3_vee(Xi: ArrayLike) -> NDArray[np.float64]:
    '''Map a 4x4 SE(3) algebra matrix to a rotation-first tangent.

    Args:
        Xi: SE(3) algebra matrix with shape ``(4, 4)``.

    Returns:
        Tangent ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]`` with shape
        ``(6,)``.

    Raises:
        ValueError: If ``Xi`` is not a finite 4x4 matrix.
    '''

    M = as_matrix(Xi, (4, 4), "Xi")

    # MROB exposes ``hat6`` but no corresponding Python ``vee`` helper.
    omega = np.array([M[2, 1], M[0, 2], M[1, 0]])
    return np.r_[omega, M[:3, 3]]


##################################################
# SE(3) group operations
##################################################
def se3_exp(xi: ArrayLike) -> NDArray[np.float64]:
    '''Compute the SE(3) exponential map ``Exp(xi)``.

    Args:
        xi: Rotation-first tangent ``[omega, v]`` with shape ``(6,)``.

    Returns:
        Homogeneous rigid transform with shape ``(4, 4)``.

    Raises:
        ValueError: If ``xi`` is not a finite six-dimensional vector.
    '''

    x = as_vector(xi, 6, "xi")
    mrob_transform = _mrob_se3_from_tangent(x)

    if mrob_transform is not None:
        try:
            return np.asarray(mrob_transform.T(), dtype=float)
        except Exception:
            pass

    # Rotation and translation share the SO(3) left Jacobian ``V``.
    R = so3_exp(x[:3])
    V = so3_left_jacobian(x[:3])
    t = V @ x[3:]

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def se3_log(T: ArrayLike) -> NDArray[np.float64]:
    '''Compute the principal SE(3) logarithm ``Log(T)``.

    Args:
        T: Homogeneous rigid transform with shape ``(4, 4)``.

    Returns:
        Rotation-first tangent ``[omega, v]`` with shape ``(6,)``.

    Raises:
        ValueError: If ``T`` is not a finite 4x4 matrix.
    '''

    M = as_matrix(T, (4, 4), "T")
    mrob_transform = _mrob_se3_from_matrix(M)

    if mrob_transform is not None:
        try:
            mrob_log = np.asarray(mrob_transform.Ln(), dtype=float)
            if mrob_log.shape == (4, 4):
                return se3_vee(mrob_log)
            return mrob_log.reshape(6)
        except Exception:
            pass

    # Recover rotation first, then invert the SO(3) translation Jacobian.
    omega = so3_log(M[:3, :3])
    Vinv = so3_left_jacobian_inverse(omega)
    v = Vinv @ M[:3, 3]

    return np.r_[omega, v]


def se3_inverse(T: ArrayLike) -> NDArray[np.float64]:
    '''Compute the analytic inverse of an SE(3) transform.

    Args:
        T: Homogeneous rigid transform with shape ``(4, 4)``.

    Returns:
        Inverse transform with shape ``(4, 4)``.

    Raises:
        ValueError: If ``T`` is not a finite 4x4 matrix.
    '''

    M = as_matrix(T, (4, 4), "T")
    mrob_transform = _mrob_se3_from_matrix(M)

    if mrob_transform is not None:
        try:
            return np.asarray(mrob_transform.inv().T(), dtype=float)
        except Exception:
            pass

    R = M[:3, :3]
    t = M[:3, 3]

    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)

    return Ti


def se3_adjoint(T: ArrayLike) -> NDArray[np.float64]:
    '''Compute ``Adj(T)`` for rotation-first SE(3) tangents.

    Args:
        T: Homogeneous rigid transform with shape ``(4, 4)``.

    Returns:
        Group adjoint with shape ``(6, 6)``.

    Raises:
        ValueError: If ``T`` is not a finite 4x4 matrix.
    '''

    M = as_matrix(T, (4, 4), "T")
    mrob_transform = _mrob_se3_from_matrix(M)

    if mrob_transform is not None:
        try:
            return np.asarray(mrob_transform.adj(), dtype=float)
        except Exception:
            pass

    R = M[:3, :3]
    t = M[:3, 3]

    Adj = np.zeros((6, 6), dtype=float)
    Adj[:3, :3] = R
    Adj[3:, :3] = so3_hat(t) @ R
    Adj[3:, 3:] = R

    return Adj


def transform_point(T: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    '''Transform a three-dimensional point by an SE(3) matrix.

    Args:
        T: Homogeneous rigid transform with shape ``(4, 4)``.
        p: Point with shape ``(3,)``.

    Returns:
        Transformed point ``R @ p + t`` with shape ``(3,)``.

    Raises:
        ValueError: If ``T`` or ``p`` has an invalid shape or non-finite value.
    '''

    M = as_matrix(T, (4, 4), "T")
    q = as_vector(p, 3, "p")
    mrob_transform = _mrob_se3_from_matrix(M)

    if mrob_transform is not None:
        try:
            return np.asarray(mrob_transform.transform(q), dtype=float).reshape(3)
        except Exception:
            pass

    return M[:3, :3] @ q + M[:3, 3]


##################################################
# SE(3) algebra adjoint and left Jacobians
##################################################
def se3_little_adjoint(xi: ArrayLike) -> NDArray[np.float64]:
    '''Compute the SE(3) Lie-algebra adjoint ``ad_xi``.

    Args:
        xi: Rotation-first tangent ``[omega, v]`` with shape ``(6,)``.

    Returns:
        Algebra adjoint with shape ``(6, 6)``.

    Raises:
        ValueError: If ``xi`` is not a finite six-dimensional vector.
    '''

    x = as_vector(xi, 6, "xi")

    if _mrob_geometry_is_usable("SE3") and hasattr(_mrob, "curley_wedge"):
        try:
            return np.asarray(_mrob.curley_wedge(x), dtype=float)
        except Exception:
            pass

    w_hat = so3_hat(x[:3])
    v_hat = so3_hat(x[3:])

    ad = np.zeros((6, 6), dtype=float)
    ad[:3, :3] = w_hat
    ad[3:, :3] = v_hat
    ad[3:, 3:] = w_hat

    return ad


def _se3_left_jacobian_series(
    ad: NDArray[np.float64],
    *,
    tolerance: float,
    max_terms: int,
) -> tuple[NDArray[np.float64], bool]:
    '''Evaluate the algebra-adjoint series for the SE(3) left Jacobian.

    Args:
        ad: Algebra adjoint with shape ``(6, 6)``.
        tolerance: Frobenius-norm convergence threshold.
        max_terms: Maximum number of series indices considered.

    Returns:
        Tuple ``(J, converged)`` containing the accumulated Jacobian and the
        convergence status.
    '''

    J = np.eye(6)
    power = np.eye(6)
    factorial = 1.0

    for n in range(1, max_terms):
        power = power @ ad
        factorial *= float(n + 1)
        term = power / factorial
        J = J + term

        if np.linalg.norm(term, ord="fro") < tolerance:
            return J, True

    return J, False


def se3_left_jacobian(
    xi: ArrayLike, *, tolerance: float = 1e-14, max_terms: int = 80
) -> NDArray[np.float64]:
    '''Compute the SE(3) left Jacobian by an algebra-adjoint series.

    Args:
        xi: Rotation-first tangent ``[omega, v]`` with shape ``(6,)``.
        tolerance: Positive Frobenius-norm convergence threshold.
        max_terms: Maximum number of series indices, at least two.

    Returns:
        Left Jacobian with shape ``(6, 6)``.

    Raises:
        ValueError: If the series settings are invalid, ``xi`` is invalid, or
            the requested series does not converge for a non-negligible tangent.
    '''

    x = as_vector(xi, 6, "xi")
    if tolerance <= 0.0 or max_terms < 2:
        raise ValueError("tolerance must be positive and max_terms must be >= 2")

    ad = se3_little_adjoint(x)
    J, converged = _se3_left_jacobian_series(
        ad,
        tolerance=tolerance,
        max_terms=max_terms,
    )

    if converged:
        return J

    # Preserve the accumulated small-tangent limit from the original code.
    if np.linalg.norm(x) < 1e-10:
        return J

    raise ValueError("SE(3) left-Jacobian series did not converge")


def se3_left_jacobian_inverse(xi: ArrayLike) -> NDArray[np.float64]:
    '''Compute the inverse SE(3) left Jacobian using a linear solve.

    Args:
        xi: Rotation-first tangent ``[omega, v]`` with shape ``(6,)``.

    Returns:
        Inverse left Jacobian with shape ``(6, 6)``.

    Raises:
        ValueError: If ``xi`` is invalid or the Jacobian series does not
            converge.
    '''

    J = se3_left_jacobian(xi)
    return solve(J, np.eye(6), assume_a="gen")