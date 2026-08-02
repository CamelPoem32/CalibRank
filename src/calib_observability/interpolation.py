'''Geodesic interpolation helpers on SO(3) and SE(3).

Each interpolation interval is represented by one Lie-group geodesic. The
resulting pose trajectory is continuous inside and across adjacent intervals,
but body or spatial velocities may be discontinuous where separately defined
segments meet.
'''

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import as_matrix
from .lie_so3 import so3_exp, so3_log
from .lie_se3 import se3_adjoint, se3_exp, se3_inverse, se3_log


##################################################
# Input validation helpers
##################################################
def _check_alpha(alpha: float) -> float:
    '''Validate a normalized interpolation fraction.

    Args:
        alpha: Interpolation fraction expected inside the closed interval
            ``[0, 1]``.

    Returns:
        The validated interpolation fraction as a float.

    Raises:
        ValueError: If ``alpha`` is not finite or lies outside ``[0, 1]``.
    '''

    # Convert scalar-like inputs before checking the valid interpolation range.
    value = float(alpha)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")

    return value


def _check_interval(
    t_0: float,
    t_1: float,
    t: float,
    *,
    allow_extrapolation: bool,
) -> tuple[float, float, float]:
    '''Validate interpolation support times and the requested query time.

    Args:
        t_0: Start time of the interpolation interval.
        t_1: End time of the interpolation interval.
        t: Time at which the interpolated pose is requested.
        allow_extrapolation: Whether ``t`` may lie outside ``[t_0, t_1]``.

    Returns:
        A tuple ``(start, end, query)`` containing validated float times.

    Raises:
        ValueError: If a time is non-finite, the interval is not increasing,
            or extrapolation is disabled and ``t`` lies outside the interval.
    '''

    # Convert all time values once so subsequent arithmetic uses plain floats.
    start = float(t_0)
    end = float(t_1)
    query = float(t)

    # Reject invalid support intervals before checking the query location.
    if not np.isfinite([start, end, query]).all() or end <= start:
        raise ValueError("t_1 must be greater than t_0 and all times must be finite")

    if not allow_extrapolation and (
        query < start - 1e-12
        or query > end + 1e-12
    ):
        raise ValueError("interpolation time must lie inside [t_0, t_1]")

    return start, end, query


def _check_rotation(R: ArrayLike, name: str) -> NDArray[np.float64]:
    '''Validate one finite proper rotation matrix.

    Args:
        R: Candidate rotation matrix with shape ``(3, 3)``.
        name: Argument name used in validation messages.

    Returns:
        The validated rotation matrix as a floating-point array.

    Raises:
        ValueError: If ``R`` is not finite, has an invalid shape, is not
            orthogonal, or has a non-positive determinant.
    '''

    # Reuse the package matrix validator before checking SO(3) constraints.
    matrix = as_matrix(R, (3, 3), name)
    is_orthogonal = np.allclose(
        matrix.T @ matrix,
        np.eye(3),
        atol=1e-8,
    )
    if not is_orthogonal or np.linalg.det(matrix) <= 0.0:
        raise ValueError(f"{name} must be a valid rotation matrix")

    return matrix


##################################################
# Relative Lie-group displacement helpers
##################################################
def _relative_so3_vector(
    R_start: NDArray[np.float64],
    R_end: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Calculate the tangent vector from one rotation to another.

    Args:
        R_start: Starting rotation matrix with shape ``(3, 3)``.
        R_end: Ending rotation matrix with shape ``(3, 3)``.

    Returns:
        The relative SO(3) logarithm ``Log(R_start.T @ R_end)``, shape ``(3,)``.
    '''

    # Express the endpoint relative to the starting rotation.
    relative_rotation = R_start.T @ R_end
    return so3_log(relative_rotation)


def _relative_se3_vector(
    T_start: NDArray[np.float64],
    T_end: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Calculate the tangent vector from one pose to another.

    Args:
        T_start: Starting homogeneous transform with shape ``(4, 4)``.
        T_end: Ending homogeneous transform with shape ``(4, 4)``.

    Returns:
        The relative SE(3) logarithm ``Log(inv(T_start) @ T_end)``, shape
        ``(6,)`` in rotation-first tangent ordering.
    '''

    # Express the endpoint pose in the local frame of the starting pose.
    relative_transform = se3_inverse(T_start) @ T_end
    return se3_log(relative_transform)


##################################################
# SO(3) interpolation
##################################################
def interpolate_so3(
    R_0: ArrayLike,
    R_1: ArrayLike,
    alpha: float,
) -> NDArray[np.float64]:
    '''Interpolate two rotations along one SO(3) geodesic.

    The interpolation is

    ``R(alpha) = R_0 @ Exp(alpha * Log(R_0.T @ R_1))``.

    Args:
        R_0: Starting rotation matrix with shape ``(3, 3)``.
        R_1: Ending rotation matrix with shape ``(3, 3)``.
        alpha: Interpolation fraction inside ``[0, 1]``.

    Returns:
        The interpolated rotation matrix with shape ``(3, 3)``.

    Raises:
        ValueError: If either rotation is invalid or ``alpha`` lies outside
            ``[0, 1]``.

    Notes:
        The SO(3) logarithm is assumed not to be evaluated at its branch
        singularity near a relative rotation angle of pi.
    '''

    # Validate endpoint rotations before evaluating the interpolation fraction.
    R_start = _check_rotation(R_0, "R_0")
    R_end = _check_rotation(R_1, "R_1")
    interpolation_fraction = _check_alpha(alpha)

    # Move along the relative tangent vector from the starting rotation.
    phi = _relative_so3_vector(R_start, R_end)
    R_alpha = R_start @ so3_exp(interpolation_fraction * phi)

    return R_alpha


def interpolate_so3_with_angular_velocity(
    R_0: ArrayLike,
    R_1: ArrayLike,
    t_0: float,
    t_1: float,
    t: float,
    *,
    allow_extrapolation: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''Interpolate an SO(3) pose and return body and spatial angular velocity.

    For one interval,

    ``R(t) = R_0 @ Exp(alpha * phi)``,
    ``alpha = (t - t_0) / (t_1 - t_0)``,
    ``phi = Log(R_0.T @ R_1)``.

    Args:
        R_0: Starting rotation matrix with shape ``(3, 3)``.
        R_1: Ending rotation matrix with shape ``(3, 3)``.
        t_0: Start time of the interpolation interval.
        t_1: End time of the interpolation interval.
        t: Requested interpolation time.
        allow_extrapolation: Whether ``t`` may lie outside ``[t_0, t_1]``.

    Returns:
        A tuple containing:

        - ``R_t``: interpolated rotation matrix, shape ``(3, 3)``.
        - ``omega_body``: constant body angular velocity, shape ``(3,)``.
        - ``omega_spatial``: spatial angular velocity at ``R_t``, shape ``(3,)``.

    Raises:
        ValueError: If the interval, query time, or endpoint rotations are
            invalid.

    Notes:
        Piecewise interpolation remains continuous in rotation, but angular
        velocity may jump at knots between independently constructed segments.
    '''

    # Validate the interval first to preserve the original error precedence.
    start, end, query = _check_interval(
        t_0,
        t_1,
        t,
        allow_extrapolation=allow_extrapolation,
    )
    R_start = _check_rotation(R_0, "R_0")
    R_end = _check_rotation(R_1, "R_1")

    # Interpolate along the relative rotation vector at the requested time.
    interval_duration = end - start
    alpha = (query - start) / interval_duration
    phi = _relative_so3_vector(R_start, R_end)
    R_t = R_start @ so3_exp(alpha * phi)

    # Convert the constant body velocity to its spatial representation.
    omega_body = phi / interval_duration
    omega_spatial = R_t @ omega_body

    return R_t, omega_body, omega_spatial


##################################################
# SE(3) interpolation
##################################################
def interpolate_se3(
    T_0: ArrayLike,
    T_1: ArrayLike,
    alpha: float,
) -> NDArray[np.float64]:
    '''Interpolate two poses along one SE(3) geodesic.

    The interpolation is

    ``T(alpha) = T_0 @ Exp(alpha * Log(inv(T_0) @ T_1))``.

    Args:
        T_0: Starting homogeneous transform with shape ``(4, 4)``.
        T_1: Ending homogeneous transform with shape ``(4, 4)``.
        alpha: Interpolation fraction inside ``[0, 1]``.

    Returns:
        The interpolated homogeneous transform with shape ``(4, 4)``.

    Raises:
        ValueError: If a transform is non-finite, has an invalid shape, or
            ``alpha`` lies outside ``[0, 1]``.

    Notes:
        SE(3) tangents use rotation-first ordering. The logarithm inherits the
        SO(3) branch singularity near a relative rotation angle of pi.
    '''

    # Validate both endpoint matrices before checking the interpolation fraction.
    T_start = as_matrix(T_0, (4, 4), "T_0")
    T_end = as_matrix(T_1, (4, 4), "T_1")
    interpolation_fraction = _check_alpha(alpha)

    # Move from T_start along the relative SE(3) tangent vector.
    xi = _relative_se3_vector(T_start, T_end)
    T_alpha = T_start @ se3_exp(interpolation_fraction * xi)

    return T_alpha


def interpolate_se3_with_twist(
    T_0: ArrayLike,
    T_1: ArrayLike,
    t_0: float,
    t_1: float,
    t: float,
    *,
    allow_extrapolation: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''Interpolate an SE(3) pose and return body and spatial twists.

    For one interval,

    ``T(t) = T_0 @ Exp(alpha * xi)``,
    ``alpha = (t - t_0) / (t_1 - t_0)``,
    ``xi = Log(inv(T_0) @ T_1)``.

    Args:
        T_0: Starting homogeneous transform with shape ``(4, 4)``.
        T_1: Ending homogeneous transform with shape ``(4, 4)``.
        t_0: Start time of the interpolation interval.
        t_1: End time of the interpolation interval.
        t: Requested interpolation time.
        allow_extrapolation: Whether ``t`` may lie outside ``[t_0, t_1]``.

    Returns:
        A tuple containing:

        - ``T_t``: interpolated homogeneous transform, shape ``(4, 4)``.
        - ``xi_body``: constant body twist, shape ``(6,)``.
        - ``xi_spatial``: spatial twist at ``T_t``, shape ``(6,)``.

    Raises:
        ValueError: If the interval, query time, or endpoint transforms are
            invalid.

    Notes:
        Piecewise interpolation remains continuous in pose, but twist may jump
        at knots between independently constructed segments.
    '''

    # Validate the interval first to preserve the original error precedence.
    start, end, query = _check_interval(
        t_0,
        t_1,
        t,
        allow_extrapolation=allow_extrapolation,
    )
    T_start = as_matrix(T_0, (4, 4), "T_0")
    T_end = as_matrix(T_1, (4, 4), "T_1")

    # Interpolate along the relative pose tangent at the requested time.
    interval_duration = end - start
    alpha = (query - start) / interval_duration
    xi = _relative_se3_vector(T_start, T_end)
    T_t = T_start @ se3_exp(alpha * xi)

    # Convert the constant body twist to the spatial frame of T_t.
    xi_body = xi / interval_duration
    xi_spatial = se3_adjoint(T_t) @ xi_body

    return T_t, xi_body, xi_spatial