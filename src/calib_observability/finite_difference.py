'''Finite-difference utilities using the package left-perturbation convention.'''

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import as_matrix, as_vector
from .lie_so3 import so3_exp
from .lie_se2 import se2_exp
from .lie_se3 import se3_exp


##################################################
# Shared validation and perturbation helpers
##################################################
def _check_positive_epsilon(epsilon: float) -> None:
    '''Validate a finite-difference perturbation magnitude.

    Args:
        epsilon: Perturbation magnitude used on both sides of the central
            difference.

    Raises:
        ValueError: If ``epsilon`` is not strictly positive.
    '''

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")


def _finite_difference_left_jacobian(
    residual_fn: Callable[[NDArray[np.float64]], ArrayLike],
    transformation: NDArray[np.float64],
    tangent_dimension: int,
    exp_fn: Callable[[ArrayLike], NDArray[np.float64]],
    epsilon: float,
) -> NDArray[np.float64]:
    '''Calculate a central Jacobian using left Lie-group perturbations.

    Args:
        residual_fn: Function mapping a perturbed transformation to a residual
            vector.
        transformation: Transformation at the linearization point.
        tangent_dimension: Dimension of the corresponding Lie algebra.
        exp_fn: Exponential map from a tangent vector to a transformation.
        epsilon: Positive central-difference perturbation magnitude.

    Returns:
        The residual Jacobian with shape ``(m, tangent_dimension)``.
    '''

    # Evaluate once to determine the residual dimension without changing the
    # original finite-difference behavior or imposing extra output validation.
    residual_zero = np.asarray(residual_fn(transformation), dtype=float)
    J = np.zeros((residual_zero.size, tangent_dimension), dtype=float)

    # Perturb each tangent coordinate on the left and form one Jacobian column.
    for i in range(tangent_dimension):
        perturbation = np.zeros(tangent_dimension)
        perturbation[i] = epsilon

        transformation_plus = exp_fn(perturbation) @ transformation
        transformation_minus = exp_fn(-perturbation) @ transformation

        residual_plus = np.asarray(residual_fn(transformation_plus))
        residual_minus = np.asarray(residual_fn(transformation_minus))
        J[:, i] = (residual_plus - residual_minus) / (2.0 * epsilon)

    return J


##################################################
# Euclidean central differences
##################################################
def central_difference_scalar(
    f: Callable[[float], float],
    x: float,
    epsilon: float = 1e-7,
) -> float:
    '''Calculate the central finite difference of a scalar function.

    Args:
        f: Scalar-valued function of one scalar argument.
        x: Point at which the derivative is evaluated.
        epsilon: Positive perturbation magnitude.

    Returns:
        The central finite-difference derivative of ``f`` at ``x``.

    Raises:
        ValueError: If ``x`` or ``epsilon`` is non-finite, or ``epsilon`` is
            not strictly positive.
    '''

    # Validate both scalar values together to retain the original error message.
    if epsilon <= 0.0 or not np.isfinite([x, epsilon]).all():
        raise ValueError("x and epsilon must be finite and epsilon > 0")

    value_plus = float(f(x + epsilon))
    value_minus = float(f(x - epsilon))

    return (value_plus - value_minus) / (2.0 * epsilon)


def central_difference_vector(
    f: Callable[[NDArray[np.float64]], ArrayLike],
    x: ArrayLike,
    epsilon: float = 1e-7,
) -> NDArray[np.float64]:
    '''Calculate the central finite-difference Jacobian of a vector function.

    Args:
        f: Function mapping an input vector with shape ``(n,)`` to an output
            vector with shape ``(m,)``.
        x: Evaluation vector with shape ``(n,)``.
        epsilon: Positive perturbation magnitude applied to each coordinate.

    Returns:
        The Jacobian matrix with shape ``(m, n)``.

    Raises:
        ValueError: If ``x`` is not a finite one-dimensional vector,
            ``epsilon`` is not positive, or ``f(x)`` is not a finite
            one-dimensional vector.
    '''

    # Validate the evaluation vector before calling the supplied function.
    x0 = np.asarray(x, dtype=float)
    if x0.ndim != 1 or not np.all(np.isfinite(x0)):
        raise ValueError("x must be a finite one-dimensional vector")
    _check_positive_epsilon(epsilon)

    # Evaluate once to determine the output dimension and validate its format.
    y0 = np.asarray(f(x0), dtype=float)
    if y0.ndim != 1 or not np.all(np.isfinite(y0)):
        raise ValueError("f(x) must be a finite one-dimensional vector")

    J = np.zeros((y0.size, x0.size), dtype=float)

    # Perturb one Euclidean input coordinate at a time.
    for i in range(x0.size):
        step = np.zeros_like(x0)
        step[i] = epsilon

        y_plus = np.asarray(f(x0 + step), dtype=float)
        y_minus = np.asarray(f(x0 - step), dtype=float)
        J[:, i] = (y_plus - y_minus) / (2.0 * epsilon)

    return J


##################################################
# Lie-group left-perturbation Jacobians
##################################################
def finite_difference_left_jacobian_so3(
    residual_fn: Callable[[NDArray[np.float64]], ArrayLike],
    R: ArrayLike,
    epsilon: float = 1e-7,
) -> NDArray[np.float64]:
    '''Finite-difference a residual under SO(3) left perturbations.

    The perturbations are

    ``R_plus = Exp(+epsilon * e_i) @ R`` and
    ``R_minus = Exp(-epsilon * e_i) @ R``.

    Args:
        residual_fn: Function accepting a rotation matrix with shape ``(3, 3)``
            and returning a residual vector with shape ``(m,)``.
        R: Rotation matrix at the linearization point, shape ``(3, 3)``.
        epsilon: Positive tangent-space perturbation magnitude.

    Returns:
        The Jacobian with shape ``(m, 3)``.

    Raises:
        ValueError: If ``R`` is not a finite ``(3, 3)`` matrix or ``epsilon``
            is not strictly positive.
    '''

    # Validate the matrix before the perturbation size to preserve error order.
    R0 = as_matrix(R, (3, 3), "R")
    _check_positive_epsilon(epsilon)

    return _finite_difference_left_jacobian(
        residual_fn,
        R0,
        3,
        so3_exp,
        epsilon,
    )


def finite_difference_left_jacobian_se3(
    residual_fn: Callable[[NDArray[np.float64]], ArrayLike],
    T: ArrayLike,
    epsilon: float = 1e-7,
) -> NDArray[np.float64]:
    '''Finite-difference a residual under SE(3) left perturbations.

    The perturbations are

    ``T_plus = Exp(+epsilon * e_i) @ T`` and
    ``T_minus = Exp(-epsilon * e_i) @ T``.

    Args:
        residual_fn: Function accepting a homogeneous transform with shape
            ``(4, 4)`` and returning a residual vector with shape ``(m,)``.
        T: Transform at the linearization point, shape ``(4, 4)``.
        epsilon: Positive tangent-space perturbation magnitude.

    Returns:
        The Jacobian with shape ``(m, 6)`` in rotation-first tangent ordering.

    Raises:
        ValueError: If ``T`` is not a finite ``(4, 4)`` matrix or ``epsilon``
            is not strictly positive.
    '''

    # Validate the matrix before the perturbation size to preserve error order.
    T0 = as_matrix(T, (4, 4), "T")
    _check_positive_epsilon(epsilon)

    return _finite_difference_left_jacobian(
        residual_fn,
        T0,
        6,
        se3_exp,
        epsilon,
    )


def finite_difference_left_jacobian_se2(
    residual_fn: Callable[[NDArray[np.float64]], ArrayLike],
    T: ArrayLike,
    epsilon: float = 1e-7,
) -> NDArray[np.float64]:
    '''Finite-difference a residual under SE(2) left perturbations.

    The perturbations are

    ``T_plus = Exp(+epsilon * e_i) @ T`` and
    ``T_minus = Exp(-epsilon * e_i) @ T``.

    Args:
        residual_fn: Function accepting a homogeneous transform with shape
            ``(3, 3)`` and returning a residual vector with shape ``(m,)``.
        T: Transform at the linearization point, shape ``(3, 3)``.
        epsilon: Positive tangent-space perturbation magnitude.

    Returns:
        The Jacobian with shape ``(m, 3)``.

    Raises:
        ValueError: If ``T`` is not a finite ``(3, 3)`` matrix or ``epsilon``
            is not strictly positive.
    '''

    # Validate the matrix before the perturbation size to preserve error order.
    T0 = as_matrix(T, (3, 3), "T")
    _check_positive_epsilon(epsilon)

    return _finite_difference_left_jacobian(
        residual_fn,
        T0,
        3,
        se2_exp,
        epsilon,
    )