'''Analytic and numerical Jacobian utilities from the proposal.'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import as_matrix, as_vector
from .finite_difference import (
    central_difference_vector,
    finite_difference_left_jacobian_se2,
    finite_difference_left_jacobian_se3,
)
from .lie_so3 import (
    so3_exp,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
)
from .lie_se2 import se2_adjoint, se2_inverse, se2_left_jacobian_inverse
from .lie_se3 import se3_adjoint, se3_inverse, se3_left_jacobian_inverse, se3_log
from .types import (
    JacobianCheckError,
    JacobianCheckResult,
    JacobianOptions,
    compare_jacobians,
    normalized_jacobian_options,
)
from .residuals import (
    _gyro_interpolator,
    gyro_increment_from_signal,
    gyro_propagation_residual,
    relative_body_motion,
    relative_body_motion_se2,
    relative_pose_residual_prediction_first,
    relative_pose_residual_prediction_first_se2,
    sensor_relative_prediction,
    sensor_relative_prediction_se2,
    spatial_smoothness_residual,
    spatial_smoothness_residual_se2,
    extrinsic_prior_residual,
    extrinsic_prior_residual_se2,
)


##################################################
# General Jacobian output structures
##################################################
@dataclass(frozen=True)
class SmoothnessJacobian:
    '''
    Spatial smoothness residual and left-perturbation Jacobians.

    Attributes:
        residual (NDArray[np.float64]): Factor residual vector.
        H_X_m (NDArray[np.float64]): Jacobian with respect to the current transform.
        H_X_m1 (NDArray[np.float64]): Jacobian with respect to the next transform.
    '''

    residual: NDArray[np.float64]
    H_X_m: NDArray[np.float64]
    H_X_m1: NDArray[np.float64]


@dataclass(frozen=True)
class PriorJacobian:
    '''
    Extrinsic-prior residual and left-perturbation Jacobian.

    Attributes:
        residual (NDArray[np.float64]): Factor residual vector.
        H_X (NDArray[np.float64]): Jacobian with respect to the calibration transform.
    '''

    residual: NDArray[np.float64]
    H_X: NDArray[np.float64]


@dataclass(frozen=True)
class CalibrationJacobian:
    '''
    Conjugation residual and calibration left-perturbation Jacobian.

    Attributes:
        residual (NDArray[np.float64]): Factor residual vector.
        H_X (NDArray[np.float64]): Jacobian with respect to the calibration transform.
    '''

    residual: NDArray[np.float64]
    H_X: NDArray[np.float64]


##################################################
# General SO(3) and SE(3) analytic Jacobians
##################################################
def gyro_temporal_offset_jacobian(
    R_k: ArrayLike,
    R_k1: ArrayLike,
    sample_times: ArrayLike,
    omega_samples: ArrayLike,
    t_k: float,
    t_k1: float,
    tau: float,
    b_g: ArrayLike | None = None,
    *,
    interpolation: str = "linear",
) -> NDArray[np.float64]:
    '''
    Compute `dr_k / d tau` for the gyroscope propagation residual.

    Implements the proposal formula `J_l^{-1}(r_k) @ R_k @ J_l(phi_k) @ [omega_b(t_k1+tau) -
    omega_b(t_k+tau)]`.

    Args:
        R_k (ArrayLike): Start rotation of the propagation interval.
        R_k1 (ArrayLike): End rotation of the propagation interval.
        sample_times (ArrayLike): Gyroscope sample timestamps.
        omega_samples (ArrayLike): Gyroscope angular-velocity samples.
        t_k (float): Nominal start time of the interval.
        t_k1 (float): Nominal end time of the interval.
        tau (float): Temporal offset applied to the signal query.
        b_g (ArrayLike | None): Constant gyroscope bias.
        interpolation (str): Gyroscope interpolation method.

    Returns:
        NDArray[np.float64]: Use `result[:, None]` during assembly for a `(3, 1)` block.

    Notes:
        Perturbation convention: Pose residuals use left perturbations.
    '''

    R0 = as_matrix(R_k, (3, 3), "R_k")
    _ = as_matrix(R_k1, (3, 3), "R_k1")
    bias = np.zeros(3) if b_g is None else as_vector(b_g, 3, "b_g")
    phi = gyro_increment_from_signal(
        sample_times,
        omega_samples,
        t_k,
        t_k1,
        tau,
        bias,
        interpolation=interpolation,
    )
    residual = gyro_propagation_residual(
        R_k,
        R_k1,
        sample_times,
        omega_samples,
        t_k,
        t_k1,
        tau,
        bias,
        interpolation=interpolation,
    )
    f = _gyro_interpolator(sample_times, omega_samples, interpolation)
    q_k = (f(t_k1 + tau) - bias) - (f(t_k + tau) - bias)
    # Jinv: (3, 3), R0: (3, 3), Jphi: (3, 3), q_k: (3,) -> H_tau: (3,)
    return so3_left_jacobian_inverse(residual) @ R0 @ so3_left_jacobian(phi) @ q_k


def spatial_smoothness_jacobians_left(X_m: ArrayLike, X_m1: ArrayLike) -> SmoothnessJacobian:
    '''
    Jacobian of `Log(X_m^{-1} X_m1)` under SE(3) left perturbations.

    Args:
        X_m (ArrayLike): Current calibration transform.
        X_m1 (ArrayLike): Next calibration transform.

    Returns:
        SmoothnessJacobian: `residual` shape `(6,)`, `H_X_m` and `H_X_m1` shape `(6, 6)`.

    Notes:
        Perturbation convention: `X' = Exp(delta_xi) @ X`, matching the final derivation in the proposal.
    '''

    X0 = as_matrix(X_m, (4, 4), "X_m")
    X1 = as_matrix(X_m1, (4, 4), "X_m1")
    r = spatial_smoothness_residual(X0, X1)
    Jinv = se3_left_jacobian_inverse(r)
    Adj_X_inv = se3_adjoint(se3_inverse(X0))
    # Jinv: (6, 6), Adj_X_inv: (6, 6) -> H_common: (6, 6)
    H_common = Jinv @ Adj_X_inv
    return SmoothnessJacobian(residual=r, H_X_m=-H_common, H_X_m1=H_common)


def extrinsic_prior_jacobian_left(X: ArrayLike, X_0: ArrayLike) -> PriorJacobian:
    '''
    Jacobian of `Log(X^{-1} X_0)` under an SE(3) left perturbation.

    Args:
        X (ArrayLike): Current calibration or extrinsic transform.
        X_0 (ArrayLike): Nominal calibration transform.

    Returns:
        PriorJacobian: `residual` shape `(6,)`, `H_X` shape `(6, 6)`.

    Notes:
        Perturbation convention: `X' = Exp(delta_xi) @ X`.
    '''

    Xc = as_matrix(X, (4, 4), "X")
    Xn = as_matrix(X_0, (4, 4), "X_0")
    r = extrinsic_prior_residual(Xc, Xn)
    # Jinv: (6, 6), Adj(inv(X)): (6, 6) -> H_X: (6, 6)
    H_X = -se3_left_jacobian_inverse(r) @ se3_adjoint(se3_inverse(Xc))
    return PriorJacobian(residual=r, H_X=H_X)


def pose_residual_calibration_jacobian_left(
    T_W_B_m: ArrayLike,
    T_W_B_m1: ArrayLike,
    X: ArrayLike,
    Z_m: ArrayLike,
) -> CalibrationJacobian:
    '''
    Calibration Jacobian of `Log((X^{-1} A_m X) Z_m^{-1})`.

    Args:
        T_W_B_m (ArrayLike): Body pose at the start of the interval.
        T_W_B_m1 (ArrayLike): Body pose at the end of the interval.
        X (ArrayLike): Current calibration or extrinsic transform.
        Z_m (ArrayLike): Measured sensor relative pose.

    Returns:
        CalibrationJacobian: `residual` shape `(6,)`, `H_X` shape `(6, 6)`.

    Notes:
        Perturbation convention: `X' = Exp(delta_xi) @ X` and prediction-first residuals.
    '''

    T0 = as_matrix(T_W_B_m, (4, 4), "T_W_B_m")
    T1 = as_matrix(T_W_B_m1, (4, 4), "T_W_B_m1")
    X0 = as_matrix(X, (4, 4), "X")
    Z = as_matrix(Z_m, (4, 4), "Z_m")
    A_m = relative_body_motion(T0, T1)
    Z_hat = sensor_relative_prediction(T0, T1, X0)
    r = relative_pose_residual_prediction_first(Z_hat, Z)
    # Jinv: (6, 6), Adj(inv(X)): (6, 6), motion: (6, 6) -> H_X: (6, 6)
    H_X = se3_left_jacobian_inverse(r) @ se3_adjoint(se3_inverse(X0)) @ (
        se3_adjoint(A_m) - np.eye(6)
    )
    return CalibrationJacobian(residual=r, H_X=H_X)


##################################################
# Numerical pose Jacobian interfaces
##################################################
def finite_difference_left_jacobian_se3_pose(
    residual_fn: Callable[[NDArray[np.float64]], ArrayLike],
    T: ArrayLike,
    epsilon: float = 1e-7,
) -> NDArray[np.float64]:
    '''
    Reference SE(3) pose Jacobian using left perturbations.

    This is a public alias kept in `jacobians.py` so future MROB analytic blocks can replace
    numerical blocks at the same call sites.

    Args:
        residual_fn (Callable[[NDArray[np.float64]], ArrayLike]): Residual function evaluated under perturbations.
        T (ArrayLike): Linearization transform.
        epsilon (float): Positive central finite-difference step.

    Returns:
        NDArray[np.float64]: Numerical left-perturbation Jacobian.
    '''

    return finite_difference_left_jacobian_se3(residual_fn, T, epsilon)


def finite_difference_two_pose_jacobians_left(
    residual_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike],
    T_W_B_m: ArrayLike,
    T_W_B_m1: ArrayLike,
    epsilon: float = 1e-7,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    Reference Jacobians for a residual depending on two SE(3) poses.

    Args:
        residual_fn (Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]): Residual function evaluated under perturbations.
        T_W_B_m (ArrayLike): Body pose at the start of the interval.
        T_W_B_m1 (ArrayLike): Body pose at the end of the interval.
        epsilon (float): Positive central finite-difference step.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Two arrays, each shape `(m, 6)`.

    Notes:
        Perturbation convention: Both pose blocks are perturbed as `Exp(delta_xi) @ T`.
    '''

    T0 = as_matrix(T_W_B_m, (4, 4), "T_W_B_m")
    T1 = as_matrix(T_W_B_m1, (4, 4), "T_W_B_m1")

    return _finite_difference_two_pose_blocks(
        residual_fn,
        T0,
        T1,
        finite_difference_left_jacobian_se3,
        epsilon,
    )


def _finite_difference_two_pose_blocks(
    residual_fn: Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        ArrayLike,
    ],
    T0: NDArray[np.float64],
    T1: NDArray[np.float64],
    finite_difference_fn: Callable[
        [
            Callable[[NDArray[np.float64]], ArrayLike],
            ArrayLike,
            float,
        ],
        NDArray[np.float64],
    ],
    epsilon: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    Calculate left-perturbation Jacobians for a two-pose residual.

    Args:
        residual_fn: Residual function accepting the first and second poses.
        T0: Validated first pose matrix.
        T1: Validated second pose matrix.
        finite_difference_fn: Lie-group finite-difference function for one pose.
        epsilon: Positive central finite-difference step.

    Returns:
        Jacobian blocks with respect to ``T0`` and ``T1``.
    '''

    # Perturb one pose at a time while keeping the other support pose fixed.
    H0 = finite_difference_fn(
        lambda T_perturbed: residual_fn(T_perturbed, T1),
        T0,
        epsilon,
    )
    H1 = finite_difference_fn(
        lambda T_perturbed: residual_fn(T0, T_perturbed),
        T1,
        epsilon,
    )

    return H0, H1


def get_pose_jacobians_from_mrob(*args: object, **kwargs: object) -> None:
    '''
    Placeholder for future MROB pose Jacobian extraction.

    Expected future return shape for a two-pose SE(3) residual: `(H_T_m, H_T_m1)`, where
    each block is `(d, 6)` in the package rotation-first tangent ordering and left-
    perturbation convention. MROB's tangent ordering and residual convention must be
    verified before this is enabled.

    Args:
        *args (object): Positional arguments forwarded to the factor implementation.
        **kwargs (object): Keyword arguments forwarded to the factor implementation.

    Raises:
        NotImplementedError: Always, until an MROB extraction interface is defined.
    '''

    _ = args, kwargs
    raise NotImplementedError(
        "Future MROB integration point: extract analytic pose Jacobian blocks "
        "and convert them to rotation-first left-perturbation ordering."
    )


##################################################
# Planar SE(2) Jacobian variants
##################################################
def spatial_smoothness_jacobians_left_se2(
    X_m: ArrayLike, X_m1: ArrayLike
) -> SmoothnessJacobian:
    '''
    SE(2) Jacobian of `Log(X_m^{-1} X_m1)` under left perturbations.

    Args:
        X_m (ArrayLike): Current calibration transform.
        X_m1 (ArrayLike): Next calibration transform.

    Returns:
        SmoothnessJacobian: Residual and Jacobian blocks for consecutive SE(2) transforms.
    '''

    X0 = as_matrix(X_m, (3, 3), "X_m")
    X1 = as_matrix(X_m1, (3, 3), "X_m1")
    r = spatial_smoothness_residual_se2(X0, X1)
    # Jinv: (3, 3), Adj(inv(X0)): (3, 3) -> H_common: (3, 3)
    H_common = se2_left_jacobian_inverse(r) @ se2_adjoint(se2_inverse(X0))
    return SmoothnessJacobian(residual=r, H_X_m=-H_common, H_X_m1=H_common)


def extrinsic_prior_jacobian_left_se2(X: ArrayLike, X_0: ArrayLike) -> PriorJacobian:
    '''
    SE(2) Jacobian of `Log(X^{-1} X_0)` under a left perturbation.

    Args:
        X (ArrayLike): Current calibration or extrinsic transform.
        X_0 (ArrayLike): Nominal calibration transform.

    Returns:
        PriorJacobian: Planar prior residual and calibration Jacobian.
    '''

    Xc = as_matrix(X, (3, 3), "X")
    Xn = as_matrix(X_0, (3, 3), "X_0")
    r = extrinsic_prior_residual_se2(Xc, Xn)
    # Jinv: (3, 3), Adj(inv(Xc)): (3, 3) -> H_X: (3, 3)
    H_X = -se2_left_jacobian_inverse(r) @ se2_adjoint(se2_inverse(Xc))
    return PriorJacobian(residual=r, H_X=H_X)


def pose_residual_calibration_jacobian_left_se2(
    T_W_B_m: ArrayLike,
    T_W_B_m1: ArrayLike,
    X: ArrayLike,
    Z_m: ArrayLike,
) -> CalibrationJacobian:
    '''
    SE(2) calibration Jacobian of `Log((X^{-1} A_m X) Z_m^{-1})`.

    Args:
        T_W_B_m (ArrayLike): Body pose at the start of the interval.
        T_W_B_m1 (ArrayLike): Body pose at the end of the interval.
        X (ArrayLike): Current calibration or extrinsic transform.
        Z_m (ArrayLike): Measured sensor relative pose.

    Returns:
        CalibrationJacobian: Planar prediction-first residual and calibration Jacobian.
    '''

    T0 = as_matrix(T_W_B_m, (3, 3), "T_W_B_m")
    T1 = as_matrix(T_W_B_m1, (3, 3), "T_W_B_m1")
    X0 = as_matrix(X, (3, 3), "X")
    Z = as_matrix(Z_m, (3, 3), "Z_m")
    A_m = relative_body_motion_se2(T0, T1)
    Z_hat = sensor_relative_prediction_se2(T0, T1, X0)
    r = relative_pose_residual_prediction_first_se2(Z_hat, Z)
    # Jinv: (3, 3), Adj(inv(X)): (3, 3), motion: (3, 3) -> H_X: (3, 3)
    H_X = se2_left_jacobian_inverse(r) @ se2_adjoint(se2_inverse(X0)) @ (
        se2_adjoint(A_m) - np.eye(3)
    )
    return CalibrationJacobian(residual=r, H_X=H_X)


def finite_difference_two_pose_jacobians_left_se2(
    residual_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike],
    T_W_B_m: ArrayLike,
    T_W_B_m1: ArrayLike,
    epsilon: float = 1e-7,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    Reference Jacobians for a residual depending on two SE(2) poses.

    Args:
        residual_fn (Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]): Residual function evaluated under perturbations.
        T_W_B_m (ArrayLike): Body pose at the start of the interval.
        T_W_B_m1 (ArrayLike): Body pose at the end of the interval.
        epsilon (float): Positive central finite-difference step.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Planar Jacobian blocks for the first and second poses.
    '''

    T0 = as_matrix(T_W_B_m, (3, 3), "T_W_B_m")
    T1 = as_matrix(T_W_B_m1, (3, 3), "T_W_B_m1")

    return _finite_difference_two_pose_blocks(
        residual_fn,
        T0,
        T1,
        finite_difference_left_jacobian_se2,
        epsilon,
    )


##################################################
# IMU and LiDAR factor output structures
##################################################
@dataclass(frozen=True)
class GyroFactorTerms:
    '''
    Intermediate terms for the IMU gyroscope propagation residual.

    Formula from the proposal/current implementation: `E_I = R_k C Exp(phi) C^{-1}
    R_k1^{-1}`, `r_I = Log(E_I)`, where `phi = integral (omega(s) - b_g) ds`. Inputs are
    rotations `(3,3)`, a body-from-IMU transform `(4,4)`, gyro samples `(N,3)`, and scalar
    times. Outputs are reusable terms for `(3,*)` Jacobian blocks. Left perturbations are
    used for poses and `T_B_I`; only the rotation block of `T_B_I` participates.
    Interpolation is the same gyro interpolation used by the residual integral. Avoid
    intervals near the SO(3) logarithm branch at pi.

    Attributes:
        residual (NDArray[np.float64]): Factor residual vector.
        E_I (NDArray[np.float64]): Gyroscope propagation error rotation.
        R_k (NDArray[np.float64]): Start body rotation.
        R_k1 (NDArray[np.float64]): End body rotation.
        C (NDArray[np.float64]): IMU rotation relative to the body.
        phi (NDArray[np.float64]): Integrated bias-corrected gyro increment.
        Delta_R (NDArray[np.float64]): Rotation exponential of the integrated gyro increment.
        Q (NDArray[np.float64]): Gyro increment conjugated into the body frame.
        delta_t (float): Duration of the propagation interval.
        omega_start_shifted (NDArray[np.float64]): Bias-corrected angular velocity at the shifted start time.
        omega_end_shifted (NDArray[np.float64]): Bias-corrected angular velocity at the shifted end time.
    '''

    residual: NDArray[np.float64]
    E_I: NDArray[np.float64]
    R_k: NDArray[np.float64]
    R_k1: NDArray[np.float64]
    C: NDArray[np.float64]
    phi: NDArray[np.float64]
    Delta_R: NDArray[np.float64]
    Q: NDArray[np.float64]
    delta_t: float
    omega_start_shifted: NDArray[np.float64]
    omega_end_shifted: NDArray[np.float64]


@dataclass(frozen=True)
class GyroFactorLinearization:
    '''
    Complete local IMU gyro factor linearization.

    Dimensions are `residual (3,)`, start/end pose blocks `(3,6)`, `T_B_I` block `(3,6)`,
    gyro-bias block `(3,3)`, and `tau_I` block `(3,1)`. The residual is prediction-first
    rotation propagation, with left perturbations and rotation-first SE(3) tangent ordering.

    Attributes:
        residual (NDArray[np.float64]): Factor residual vector.
        H_start_pose (NDArray[np.float64]): Jacobian with respect to the start body pose.
        H_end_pose (NDArray[np.float64]): Jacobian with respect to the end body pose.
        H_T_B_I (NDArray[np.float64]): Jacobian with respect to the IMU extrinsic transform.
        H_b_g (NDArray[np.float64]): Jacobian with respect to gyroscope bias.
        H_tau_I (NDArray[np.float64]): Jacobian with respect to the IMU temporal offset.
        intermediate_terms (GyroFactorTerms): Reusable residual and kinematic terms.
        check_results (tuple[JacobianCheckResult, ...]): Optional analytic-versus-numerical comparison results.
    '''

    residual: NDArray[np.float64]
    H_start_pose: NDArray[np.float64]
    H_end_pose: NDArray[np.float64]
    H_T_B_I: NDArray[np.float64]
    H_b_g: NDArray[np.float64]
    H_tau_I: NDArray[np.float64]
    intermediate_terms: GyroFactorTerms
    check_results: tuple[JacobianCheckResult, ...] = ()


@dataclass(frozen=True)
class LidarFactorTerms:
    '''
    Intermediate terms for the LiDAR relative-pose residual.

    Formula: `A = inverse(T_0) T_1`, `Z_hat = inverse(X_L) A X_L`, `E_L = Z_hat
    inverse(Z_measurement)`, and `r_L = Log(E_L)`. Inputs are SE(3) transforms `(4,4)` and
    outputs feed `(6,*)` Jacobian blocks. All group perturbations are left perturbations
    with rotation-first tangent ordering. Singular cases are inherited from the SE(3)
    logarithm and degenerate zero-duration trajectory intervals.

    Attributes:
        residual (NDArray[np.float64]): Factor residual vector.
        E_L (NDArray[np.float64]): LiDAR prediction-first residual transform.
        T_0 (NDArray[np.float64]): Start body pose.
        T_1 (NDArray[np.float64]): End body pose.
        X_L (NDArray[np.float64]): LiDAR extrinsic transform.
        A (NDArray[np.float64]): Relative body motion.
        Z_hat (NDArray[np.float64]): Predicted sensor relative pose.
    '''

    residual: NDArray[np.float64]
    E_L: NDArray[np.float64]
    T_0: NDArray[np.float64]
    T_1: NDArray[np.float64]
    X_L: NDArray[np.float64]
    A: NDArray[np.float64]
    Z_hat: NDArray[np.float64]


@dataclass(frozen=True)
class LidarFactorLinearization:
    '''
    Complete local LiDAR relative-pose factor linearization.

    Dimensions are `residual (6,)`, start/end pose blocks `(6,6)`, `T_B_L` block `(6,6)`,
    and `tau_L` block `(6,1)`. The residual convention is prediction-first: `Log(Z_hat @
    inverse(Z_measurement))`.

    Attributes:
        residual (NDArray[np.float64]): Factor residual vector.
        H_start_pose (NDArray[np.float64]): Jacobian with respect to the start body pose.
        H_end_pose (NDArray[np.float64]): Jacobian with respect to the end body pose.
        H_T_B_L (NDArray[np.float64]): Jacobian with respect to the LiDAR extrinsic transform.
        H_tau_L (NDArray[np.float64]): Jacobian with respect to the LiDAR temporal offset.
        intermediate_terms (LidarFactorTerms): Reusable residual and kinematic terms.
        check_results (tuple[JacobianCheckResult, ...]): Optional analytic-versus-numerical comparison results.
    '''

    residual: NDArray[np.float64]
    H_start_pose: NDArray[np.float64]
    H_end_pose: NDArray[np.float64]
    H_T_B_L: NDArray[np.float64]
    H_tau_L: NDArray[np.float64]
    intermediate_terms: LidarFactorTerms
    check_results: tuple[JacobianCheckResult, ...] = ()


##################################################
# IMU gyroscope factor
##################################################
def gyro_factor_residual_and_terms(
    start_body_pose: ArrayLike,
    end_body_pose: ArrayLike,
    body_from_imu: ArrayLike,
    gyro_bias: ArrayLike,
    imu_time_offset: float,
    true_start_time: float,
    true_end_time: float,
    imu_sensor_timestamps: ArrayLike,
    gyroscope_samples: ArrayLike,
    *,
    interpolation: str = "linear",
    time_offset_sign: float = -1.0,
) -> GyroFactorTerms:
    '''
    Evaluate the IMU residual and reusable analytic terms.

    The package stores IMU samples on the sensor clock. Therefore dataset assembly uses
    `sample_time = true_time - tau_I`, represented here by the default
    `time_offset_sign=-1`. With a signal already expressed in the shifted integration clock,
    pass `time_offset_sign=+1`. All outputs are SO(3) terms with shapes documented by
    `GyroFactorTerms`.

    Args:
        start_body_pose (ArrayLike): Body pose at the start of the factor interval.
        end_body_pose (ArrayLike): Body pose at the end of the factor interval.
        body_from_imu (ArrayLike): IMU extrinsic transform relative to the body.
        gyro_bias (ArrayLike): Constant gyroscope bias vector.
        imu_time_offset (float): IMU temporal offset.
        true_start_time (float): Start time in the trajectory clock.
        true_end_time (float): End time in the trajectory clock.
        imu_sensor_timestamps (ArrayLike): Gyroscope timestamps in the sensor clock.
        gyroscope_samples (ArrayLike): Gyroscope angular-velocity samples.
        interpolation (str): Gyroscope interpolation method.
        time_offset_sign (float): Sign mapping the stored offset to the signal query clock.

    Returns:
        GyroFactorTerms: Gyroscope residual and reusable analytic terms.
    '''

    T0 = as_matrix(start_body_pose, (4, 4), "start_body_pose")
    T1 = as_matrix(end_body_pose, (4, 4), "end_body_pose")
    X = as_matrix(body_from_imu, (4, 4), "body_from_imu")
    bias = as_vector(gyro_bias, 3, "gyro_bias")
    tau_for_signal = float(time_offset_sign) * float(imu_time_offset)
    R_k = T0[:3, :3]
    R_k1 = T1[:3, :3]
    C = X[:3, :3]
    phi = gyro_increment_from_signal(
        imu_sensor_timestamps,
        gyroscope_samples,
        true_start_time,
        true_end_time,
        tau_for_signal,
        bias,
        interpolation=interpolation,
    )
    Delta_R = so3_exp(phi)
    # C: (3, 3), Delta_R: (3, 3), C.T: (3, 3) -> Q: (3, 3).
    Q = C @ Delta_R @ C.T
    # R_k: (3, 3), Q: (3, 3), R_k1.T: (3, 3) -> E_I: (3, 3).
    E_I = R_k @ Q @ R_k1.T
    residual = so3_log(E_I)
    f = _gyro_interpolator(imu_sensor_timestamps, gyroscope_samples, interpolation)
    omega_start = np.asarray(f(true_start_time + tau_for_signal), dtype=float).reshape(3) - bias
    omega_end = np.asarray(f(true_end_time + tau_for_signal), dtype=float).reshape(3) - bias
    return GyroFactorTerms(
        residual=residual,
        E_I=E_I,
        R_k=R_k,
        R_k1=R_k1,
        C=C,
        phi=phi,
        Delta_R=Delta_R,
        Q=Q,
        delta_t=float(true_end_time - true_start_time),
        omega_start_shifted=omega_start,
        omega_end_shifted=omega_end,
    )


def gyro_factor_start_pose_jacobian_left(terms: GyroFactorTerms) -> NDArray[np.float64]:
    '''
    Return `H_T_k = [J_l^{-1}(r_I), 0]`, shape `(3,6)`.

    Args:
        terms (GyroFactorTerms): Precomputed factor residual and reusable intermediate terms.

    Returns:
        NDArray[np.float64]: Start-pose Jacobian with shape ``(3, 6)``.
    '''

    J_l_inv_r = so3_left_jacobian_inverse(terms.residual)
    H = np.zeros((3, 6), dtype=float)
    H[:, :3] = J_l_inv_r
    return H


def gyro_factor_end_pose_jacobian_left(terms: GyroFactorTerms) -> NDArray[np.float64]:
    '''
    Return `H_T_k1 = [-J_l^{-1}(r_I) E_I, 0]`, shape `(3,6)`.

    Args:
        terms (GyroFactorTerms): Precomputed factor residual and reusable intermediate terms.

    Returns:
        NDArray[np.float64]: End-pose Jacobian with shape ``(3, 6)``.
    '''

    J_l_inv_r = so3_left_jacobian_inverse(terms.residual)
    H = np.zeros((3, 6), dtype=float)
    # J_l_inv_r: (3, 3), E_I: (3, 3) -> rotation block: (3, 3).
    H[:, :3] = -J_l_inv_r @ terms.E_I
    return H


def gyro_factor_extrinsic_jacobian_left(terms: GyroFactorTerms) -> NDArray[np.float64]:
    '''
    Return the analytic `T_B_I` block, shape `(3,6)`.

    Formula: `J_l^{-1}(r_I) R_k (I - Q)` for rotation columns and exact zero translation
    columns because the current gyro-only residual does not depend on the lever arm.

    Args:
        terms (GyroFactorTerms): Precomputed factor residual and reusable intermediate terms.

    Returns:
        NDArray[np.float64]: IMU-extrinsic Jacobian with shape ``(3, 6)``.
    '''

    J_l_inv_r = so3_left_jacobian_inverse(terms.residual)
    H = np.zeros((3, 6), dtype=float)
    # J_l_inv_r: (3, 3), R_k: (3, 3), (I - Q): (3, 3) -> H_rot: (3, 3).
    H[:, :3] = J_l_inv_r @ terms.R_k @ (np.eye(3) - terms.Q)
    return H


def gyro_factor_bias_jacobian(terms: GyroFactorTerms) -> NDArray[np.float64]:
    '''
    Return `H_b_g`, shape `(3,3)`, for constant gyro bias.

    Args:
        terms (GyroFactorTerms): Precomputed factor residual and reusable intermediate terms.

    Returns:
        NDArray[np.float64]: Gyroscope-bias Jacobian with shape ``(3, 3)``.
    '''

    J_l_inv_r = so3_left_jacobian_inverse(terms.residual)
    J_l_phi = so3_left_jacobian(terms.phi)
    # J_l_inv_r: (3, 3), R_k: (3, 3), C: (3, 3), J_l_phi: (3, 3)
    # dphi/db_g = -delta_t I_3, so multiplication by delta_t scales columns.
    return -J_l_inv_r @ terms.R_k @ terms.C @ J_l_phi * terms.delta_t


def gyro_factor_temporal_offset_jacobian(
    terms: GyroFactorTerms,
    *,
    time_offset_sign: float = -1.0,
) -> NDArray[np.float64]:
    '''
    Return `H_tau_I`, shape `(3,1)`, using the endpoint Leibniz rule.

    Args:
        terms (GyroFactorTerms): Precomputed factor residual and reusable intermediate terms.
        time_offset_sign (float): Sign mapping the stored offset to the signal query clock.

    Returns:
        NDArray[np.float64]: IMU temporal-offset Jacobian with shape ``(3, 1)``.
    '''

    J_l_inv_r = so3_left_jacobian_inverse(terms.residual)
    J_l_phi = so3_left_jacobian(terms.phi)
    q_k = terms.omega_end_shifted - terms.omega_start_shifted
    # Existing dataset convention uses sample_time = true_time - tau_I.
    q_k = float(time_offset_sign) * q_k
    # J_l_inv_r: (3, 3), R_k: (3, 3), C: (3, 3), J_l_phi: (3, 3), q_k: (3,)
    # H_tau_I_vector: (3,).
    H_tau_I_vector = J_l_inv_r @ terms.R_k @ terms.C @ J_l_phi @ q_k
    return H_tau_I_vector[:, None]


def linearize_gyro_factor_analytic(*args: object, **kwargs: object) -> GyroFactorLinearization:
    '''
    Compute all analytic IMU gyro factor blocks with documented dimensions.

    Args:
        *args (object): Positional arguments forwarded to the factor implementation.
        **kwargs (object): Keyword arguments forwarded to the factor implementation.

    Returns:
        GyroFactorLinearization: Complete analytic gyroscope-factor linearization.
    '''

    terms = gyro_factor_residual_and_terms(*args, **kwargs)
    time_offset_sign = float(kwargs.get("time_offset_sign", -1.0))
    return GyroFactorLinearization(
        residual=terms.residual,
        H_start_pose=gyro_factor_start_pose_jacobian_left(terms),
        H_end_pose=gyro_factor_end_pose_jacobian_left(terms),
        H_T_B_I=gyro_factor_extrinsic_jacobian_left(terms),
        H_b_g=gyro_factor_bias_jacobian(terms),
        H_tau_I=gyro_factor_temporal_offset_jacobian(terms, time_offset_sign=time_offset_sign),
        intermediate_terms=terms,
    )


def linearize_gyro_factor_finite_difference(
    start_body_pose: ArrayLike,
    end_body_pose: ArrayLike,
    body_from_imu: ArrayLike,
    gyro_bias: ArrayLike,
    imu_time_offset: float,
    true_start_time: float,
    true_end_time: float,
    imu_sensor_timestamps: ArrayLike,
    gyroscope_samples: ArrayLike,
    *,
    interpolation: str = "linear",
    time_offset_sign: float = -1.0,
    finite_difference_epsilon: float = 1e-7,
) -> GyroFactorLinearization:
    '''
    Compute reference IMU blocks by central finite differences.

    Args:
        start_body_pose (ArrayLike): Body pose at the start of the factor interval.
        end_body_pose (ArrayLike): Body pose at the end of the factor interval.
        body_from_imu (ArrayLike): IMU extrinsic transform relative to the body.
        gyro_bias (ArrayLike): Constant gyroscope bias vector.
        imu_time_offset (float): IMU temporal offset.
        true_start_time (float): Start time in the trajectory clock.
        true_end_time (float): End time in the trajectory clock.
        imu_sensor_timestamps (ArrayLike): Gyroscope timestamps in the sensor clock.
        gyroscope_samples (ArrayLike): Gyroscope angular-velocity samples.
        interpolation (str): Gyroscope interpolation method.
        time_offset_sign (float): Sign mapping the stored offset to the signal query clock.
        finite_difference_epsilon (float): Positive finite-difference step.

    Returns:
        GyroFactorLinearization: Complete numerical gyroscope-factor linearization.
    '''

    terms = gyro_factor_residual_and_terms(
        start_body_pose,
        end_body_pose,
        body_from_imu,
        gyro_bias,
        imu_time_offset,
        true_start_time,
        true_end_time,
        imu_sensor_timestamps,
        gyroscope_samples,
        interpolation=interpolation,
        time_offset_sign=time_offset_sign,
    )

    def residual_fn(
        start_pose: ArrayLike,
        end_pose: ArrayLike,
        extrinsic: ArrayLike,
        bias: ArrayLike,
        tau: float,
    ) -> NDArray[np.float64]:
        '''Evaluate the gyro residual for one perturbed parameter block.

        Args:
            start_pose: Start body pose used in the residual.
            end_pose: End body pose used in the residual.
            extrinsic: IMU extrinsic transform.
            bias: Gyroscope bias vector.
            tau: IMU temporal offset.

        Returns:
            Gyroscope propagation residual.
        '''

        # Reuse the source-of-truth residual while replacing one variable block.
        return gyro_factor_residual_and_terms(
            start_pose,
            end_pose,
            extrinsic,
            bias,
            tau,
            true_start_time,
            true_end_time,
            imu_sensor_timestamps,
            gyroscope_samples,
            interpolation=interpolation,
            time_offset_sign=time_offset_sign,
        ).residual

    # Perturb the two trajectory poses and IMU extrinsics on SE(3).
    H_start = finite_difference_left_jacobian_se3(
        lambda T: residual_fn(
            T,
            end_body_pose,
            body_from_imu,
            gyro_bias,
            imu_time_offset,
        ),
        start_body_pose,
        finite_difference_epsilon,
    )
    H_end = finite_difference_left_jacobian_se3(
        lambda T: residual_fn(
            start_body_pose,
            T,
            body_from_imu,
            gyro_bias,
            imu_time_offset,
        ),
        end_body_pose,
        finite_difference_epsilon,
    )
    H_extrinsic = finite_difference_left_jacobian_se3(
        lambda X: residual_fn(
            start_body_pose,
            end_body_pose,
            X,
            gyro_bias,
            imu_time_offset,
        ),
        body_from_imu,
        finite_difference_epsilon,
    )

    # Bias and temporal offset are Euclidean parameter blocks.
    H_bias = central_difference_vector(
        lambda b: residual_fn(
            start_body_pose,
            end_body_pose,
            body_from_imu,
            b,
            imu_time_offset,
        ),
        gyro_bias,
        finite_difference_epsilon,
    )
    H_tau = central_difference_vector(
        lambda tau: residual_fn(
            start_body_pose,
            end_body_pose,
            body_from_imu,
            gyro_bias,
            float(tau[0]),
        ),
        np.array([imu_time_offset], dtype=float),
        finite_difference_epsilon,
    )

    return GyroFactorLinearization(
        residual=terms.residual,
        H_start_pose=H_start,
        H_end_pose=H_end,
        H_T_B_I=H_extrinsic,
        H_b_g=H_bias,
        H_tau_I=H_tau,
        intermediate_terms=terms,
    )


##################################################
# LiDAR relative-pose factor
##################################################
def lidar_factor_residual_and_terms(
    start_body_pose: ArrayLike,
    end_body_pose: ArrayLike,
    body_from_lidar: ArrayLike,
    lidar_measurement: ArrayLike,
) -> LidarFactorTerms:
    '''
    Evaluate LiDAR prediction-first residual terms.

    Args:
        start_body_pose (ArrayLike): Body pose at the start of the factor interval.
        end_body_pose (ArrayLike): Body pose at the end of the factor interval.
        body_from_lidar (ArrayLike): LiDAR extrinsic transform relative to the body.
        lidar_measurement (ArrayLike): Measured LiDAR relative pose.

    Returns:
        LidarFactorTerms: LiDAR residual and reusable relative-pose terms.
    '''

    T0 = as_matrix(start_body_pose, (4, 4), "start_body_pose")
    T1 = as_matrix(end_body_pose, (4, 4), "end_body_pose")
    X = as_matrix(body_from_lidar, (4, 4), "body_from_lidar")
    Z = as_matrix(lidar_measurement, (4, 4), "lidar_measurement")
    A = relative_body_motion(T0, T1)
    Z_hat = sensor_relative_prediction(T0, T1, X)
    E_L = Z_hat @ se3_inverse(Z)
    residual = se3_log(E_L)
    return LidarFactorTerms(residual, E_L, T0, T1, X, A, Z_hat)


def _lidar_pose_common_jacobian(
    terms: LidarFactorTerms,
) -> NDArray[np.float64]:
    '''
    Calculate the common LiDAR start/end pose Jacobian block.

    Args:
        terms: LiDAR residual and reusable relative-pose terms.

    Returns:
        Common ``(6, 6)`` block before applying the start-pose sign.
    '''

    # Both pose blocks share the same residual logarithm and frame transport.
    J_l_inv_r = se3_left_jacobian_inverse(terms.residual)
    transform = se3_inverse(terms.X_L) @ se3_inverse(terms.T_0)

    return J_l_inv_r @ se3_adjoint(transform)


def lidar_factor_start_pose_jacobian_left(
    terms: LidarFactorTerms,
) -> NDArray[np.float64]:
    '''
    Calculate the LiDAR Jacobian with respect to the start pose.

    Args:
        terms: LiDAR residual and reusable relative-pose terms.

    Returns:
        Start-pose Jacobian ``H_T_0`` with shape ``(6, 6)``.
    '''

    return -_lidar_pose_common_jacobian(terms)


def lidar_factor_end_pose_jacobian_left(
    terms: LidarFactorTerms,
) -> NDArray[np.float64]:
    '''
    Calculate the LiDAR Jacobian with respect to the end pose.

    Args:
        terms: LiDAR residual and reusable relative-pose terms.

    Returns:
        End-pose Jacobian ``H_T_1`` with shape ``(6, 6)``.
    '''

    return _lidar_pose_common_jacobian(terms)


def lidar_factor_extrinsic_jacobian_left(terms: LidarFactorTerms) -> NDArray[np.float64]:
    '''
    Return `H_T_B_L = J_l^{-1}(r_L) Adj(inverse(X_L)) (Adj(A)-I_6)`, `(6,6)`.

    Args:
        terms (LidarFactorTerms): Precomputed factor residual and reusable intermediate terms.

    Returns:
        NDArray[np.float64]: LiDAR-extrinsic Jacobian with shape ``(6, 6)``.
    '''

    J_l_inv_r = se3_left_jacobian_inverse(terms.residual)
    # Adj(inv(X_L)): (6, 6), (Adj(A): (6, 6) - I_6) -> motion block: (6, 6).
    return (
        J_l_inv_r
        @ se3_adjoint(se3_inverse(terms.X_L))
        @ (se3_adjoint(terms.A) - np.eye(6))
    )


def lidar_factor_temporal_offset_jacobian_spatial(
    terms: LidarFactorTerms,
    start_spatial_twist: ArrayLike,
    end_spatial_twist: ArrayLike,
) -> NDArray[np.float64]:
    '''
    Return spatial-twist `H_tau_L`, shape `(6,1)`.

    Args:
        terms (LidarFactorTerms): Precomputed factor residual and reusable intermediate terms.
        start_spatial_twist (ArrayLike): Spatial twist at the start pose.
        end_spatial_twist (ArrayLike): Spatial twist at the end pose.

    Returns:
        NDArray[np.float64]: LiDAR temporal-offset Jacobian from spatial twists.
    '''

    xi_0_spatial = as_vector(start_spatial_twist, 6, "start_spatial_twist")
    xi_1_spatial = as_vector(end_spatial_twist, 6, "end_spatial_twist")
    # The time shift moves both endpoint poses along their spatial twists.
    H_tau = _lidar_pose_common_jacobian(terms) @ (
        xi_1_spatial - xi_0_spatial
    )

    return H_tau[:, None]


def lidar_factor_temporal_offset_jacobian_body(
    terms: LidarFactorTerms,
    start_body_twist: ArrayLike,
    end_body_twist: ArrayLike,
) -> NDArray[np.float64]:
    '''
    Return body-twist equivalent `H_tau_L`, shape `(6,1)`.

    Args:
        terms (LidarFactorTerms): Precomputed factor residual and reusable intermediate terms.
        start_body_twist (ArrayLike): Body-frame twist at the start pose.
        end_body_twist (ArrayLike): Body-frame twist at the end pose.

    Returns:
        NDArray[np.float64]: LiDAR temporal-offset Jacobian from body twists.
    '''

    xi_0_body = as_vector(start_body_twist, 6, "start_body_twist")
    xi_1_body = as_vector(end_body_twist, 6, "end_body_twist")
    J_l_inv_r = se3_left_jacobian_inverse(terms.residual)
    # Adj(A): (6, 6), xi_1_body: (6,) -> transported end twist: (6,).
    bracket = se3_adjoint(terms.A) @ xi_1_body - xi_0_body
    H_tau = J_l_inv_r @ se3_adjoint(se3_inverse(terms.X_L)) @ bracket
    return H_tau[:, None]


def linearize_lidar_factor_analytic(
    start_body_pose: ArrayLike,
    end_body_pose: ArrayLike,
    body_from_lidar: ArrayLike,
    lidar_measurement: ArrayLike,
    start_spatial_twist: ArrayLike,
    end_spatial_twist: ArrayLike,
) -> LidarFactorLinearization:
    '''
    Compute all analytic LiDAR factor blocks with prediction-first residuals.

    Args:
        start_body_pose (ArrayLike): Body pose at the start of the factor interval.
        end_body_pose (ArrayLike): Body pose at the end of the factor interval.
        body_from_lidar (ArrayLike): LiDAR extrinsic transform relative to the body.
        lidar_measurement (ArrayLike): Measured LiDAR relative pose.
        start_spatial_twist (ArrayLike): Spatial twist at the start pose.
        end_spatial_twist (ArrayLike): Spatial twist at the end pose.

    Returns:
        LidarFactorLinearization: Complete analytic LiDAR-factor linearization.
    '''

    terms = lidar_factor_residual_and_terms(
        start_body_pose,
        end_body_pose,
        body_from_lidar,
        lidar_measurement,
    )
    return LidarFactorLinearization(
        residual=terms.residual,
        H_start_pose=lidar_factor_start_pose_jacobian_left(terms),
        H_end_pose=lidar_factor_end_pose_jacobian_left(terms),
        H_T_B_L=lidar_factor_extrinsic_jacobian_left(terms),
        H_tau_L=lidar_factor_temporal_offset_jacobian_spatial(
            terms,
            start_spatial_twist,
            end_spatial_twist,
        ),
        intermediate_terms=terms,
    )


def linearize_lidar_factor_finite_difference(
    start_body_pose: ArrayLike,
    end_body_pose: ArrayLike,
    body_from_lidar: ArrayLike,
    lidar_measurement: ArrayLike,
    pose_provider: object,
    sensor_start_time: float,
    sensor_end_time: float,
    lidar_time_offset: float,
    *,
    finite_difference_epsilon: float = 1e-7,
) -> LidarFactorLinearization:
    '''
    Compute reference LiDAR blocks by central finite differences.

    Args:
        start_body_pose (ArrayLike): Body pose at the start of the factor interval.
        end_body_pose (ArrayLike): Body pose at the end of the factor interval.
        body_from_lidar (ArrayLike): LiDAR extrinsic transform relative to the body.
        lidar_measurement (ArrayLike): Measured LiDAR relative pose.
        pose_provider (object): Trajectory interface used to query poses and twists.
        sensor_start_time (float): Start timestamp in the LiDAR sensor clock.
        sensor_end_time (float): End timestamp in the LiDAR sensor clock.
        lidar_time_offset (float): LiDAR temporal offset.
        finite_difference_epsilon (float): Positive finite-difference step.

    Returns:
        LidarFactorLinearization: Complete numerical LiDAR-factor linearization.
    '''

    shifted_times = np.array(
        [
            sensor_start_time + lidar_time_offset,
            sensor_end_time + lidar_time_offset,
        ],
        dtype=float,
    )
    poses, _, spatial_twists = pose_provider.poses_and_twists_at(shifted_times)
    terms = lidar_factor_residual_and_terms(
        start_body_pose,
        end_body_pose,
        body_from_lidar,
        lidar_measurement,
    )

    def residual_from(
        start_pose: ArrayLike,
        end_pose: ArrayLike,
        extrinsic: ArrayLike,
    ) -> NDArray[np.float64]:
        '''Evaluate the LiDAR residual for one perturbed SE(3) block.

        Args:
            start_pose: Start body pose used in the residual.
            end_pose: End body pose used in the residual.
            extrinsic: LiDAR extrinsic transform.

        Returns:
            Prediction-first LiDAR relative-pose residual.
        '''

        # Reuse the source-of-truth residual while replacing one SE(3) block.
        return lidar_factor_residual_and_terms(
            start_pose,
            end_pose,
            extrinsic,
            lidar_measurement,
        ).residual

    # Perturb the two body poses and the LiDAR extrinsic transform separately.
    H_start = finite_difference_left_jacobian_se3(
        lambda T: residual_from(T, end_body_pose, body_from_lidar),
        start_body_pose,
        finite_difference_epsilon,
    )
    H_end = finite_difference_left_jacobian_se3(
        lambda T: residual_from(start_body_pose, T, body_from_lidar),
        end_body_pose,
        finite_difference_epsilon,
    )
    H_extrinsic = finite_difference_left_jacobian_se3(
        lambda X: residual_from(start_body_pose, end_body_pose, X),
        body_from_lidar,
        finite_difference_epsilon,
    )

    def residual_at_tau(tau: float) -> NDArray[np.float64]:
        '''Evaluate the LiDAR residual at one temporal-offset value.

        Args:
            tau: LiDAR temporal offset applied to both trajectory query times.

        Returns:
            Prediction-first LiDAR relative-pose residual.
        '''

        # Shift both trajectory query times by the same LiDAR clock offset.
        query_times = np.array(
            [
                sensor_start_time + tau,
                sensor_end_time + tau,
            ],
            dtype=float,
        )
        shifted_poses = np.asarray(
            pose_provider.poses_at(query_times),
            dtype=float,
        )

        return lidar_factor_residual_and_terms(
            shifted_poses[0],
            shifted_poses[1],
            body_from_lidar,
            lidar_measurement,
        ).residual

    H_tau = central_difference_vector(
        lambda tau: residual_at_tau(float(tau[0])),
        np.array([lidar_time_offset], dtype=float),
        finite_difference_epsilon,
    )

    return LidarFactorLinearization(
        residual=terms.residual,
        H_start_pose=H_start,
        H_end_pose=H_end,
        H_T_B_L=H_extrinsic,
        H_tau_L=H_tau,
        intermediate_terms=terms,
    )


##################################################
# Analytic and finite-difference checks
##################################################
def _gyro_linearization_blocks(
    linearization: GyroFactorLinearization,
) -> dict[str, NDArray[np.float64]]:
    '''
    Return named IMU Jacobian blocks for analytic checks.

    Args:
        linearization: Complete local gyroscope-factor linearization.

    Returns:
        Mapping from variable names to Jacobian blocks.
    '''

    return {
        "T_start": linearization.H_start_pose,
        "T_end": linearization.H_end_pose,
        "T_B_I": linearization.H_T_B_I,
        "b_g": linearization.H_b_g,
        "tau_I": linearization.H_tau_I,
    }


def _lidar_linearization_blocks(
    linearization: LidarFactorLinearization,
) -> dict[str, NDArray[np.float64]]:
    '''
    Return named LiDAR Jacobian blocks for analytic checks.

    Args:
        linearization: Complete local LiDAR-factor linearization.

    Returns:
        Mapping from variable names to Jacobian blocks.
    '''

    return {
        "T_start": linearization.H_start_pose,
        "T_end": linearization.H_end_pose,
        "T_B_L": linearization.H_T_B_L,
        "tau_L": linearization.H_tau_L,
    }


def _check_or_raise(
    checks: list[JacobianCheckResult],
    options: JacobianOptions,
) -> None:
    '''Raise a compact error when a requested Jacobian check fails.

    Args:
        checks: Analytic-versus-numerical comparison results.
        options: Jacobian checking tolerances and failure behavior.

    Raises:
        JacobianCheckError: If at least one check fails and raising is enabled.
    '''

    failed = [check for check in checks if not check.passed]
    if failed and options.raise_on_check_failure:
        first = failed[0]
        raise JacobianCheckError(
            f"Jacobian check failed for {first.factor_name}/{first.variable_name}: "
            f"max_abs={first.max_absolute_error:.3e}, rel_fro={first.relative_frobenius_error:.3e}"
        )


def _compare_linearization_blocks(
    analytic_blocks: dict[str, NDArray[np.float64]],
    finite_difference_blocks: dict[str, NDArray[np.float64]],
    *,
    factor_name: str,
    options: JacobianOptions,
) -> tuple[JacobianCheckResult, ...]:
    '''Compare corresponding analytic and numerical Jacobian blocks.

    Args:
        analytic_blocks: Analytic Jacobian blocks keyed by variable name.
        finite_difference_blocks: Numerical reference blocks using the same keys.
        factor_name: Factor identifier stored in each comparison result.
        options: Comparison tolerances and failure behavior.

    Returns:
        One comparison result for every analytic Jacobian block.
    '''

    checks = [
        compare_jacobians(
            analytic_blocks[name],
            finite_difference_blocks[name],
            factor_name=factor_name,
            variable_name=name,
            atol=options.check_atol,
            rtol=options.check_rtol,
        )
        for name in analytic_blocks
    ]
    _check_or_raise(checks, options)
    return tuple(checks)


##################################################
# Public factor linearization dispatch
##################################################
def linearize_gyro_factor(
    *args: object,
    jacobian_options: JacobianOptions | None = None,
    **kwargs: object,
) -> GyroFactorLinearization:
    '''
    Dispatch IMU gyro factor linearization according to `JacobianOptions`.

    Args:
        jacobian_options (JacobianOptions | None): Jacobian method and checking configuration.
        *args (object): Positional arguments forwarded to the factor implementation.
        **kwargs (object): Keyword arguments forwarded to the factor implementation.

    Returns:
        GyroFactorLinearization: Gyroscope-factor linearization selected by the configured method.
    '''

    options = normalized_jacobian_options(jacobian_options)
    if options.method == "analytic":
        return linearize_gyro_factor_analytic(*args, **kwargs)
    if options.method == "finite_difference":
        return linearize_gyro_factor_finite_difference(
            *args,
            finite_difference_epsilon=options.finite_difference_epsilon,
            **kwargs,
        )
    # Checked mode evaluates both paths and attaches the comparison results to
    # the analytic linearization returned to the caller.
    analytic = linearize_gyro_factor_analytic(*args, **kwargs)
    finite_difference = linearize_gyro_factor_finite_difference(
        *args,
        finite_difference_epsilon=options.finite_difference_epsilon,
        **kwargs,
    )
    checks = _compare_linearization_blocks(
        _gyro_linearization_blocks(analytic),
        _gyro_linearization_blocks(finite_difference),
        factor_name="imu_gyro",
        options=options,
    )

    return replace(analytic, check_results=checks)


def linearize_lidar_factor(
    *args: object,
    jacobian_options: JacobianOptions | None = None,
    **kwargs: object,
) -> LidarFactorLinearization:
    '''
    Dispatch LiDAR factor linearization according to `JacobianOptions`.

    Args:
        jacobian_options (JacobianOptions | None): Jacobian method and checking configuration.
        *args (object): Positional arguments forwarded to the factor implementation.
        **kwargs (object): Keyword arguments forwarded to the factor implementation.

    Returns:
        LidarFactorLinearization: LiDAR-factor linearization selected by the configured method.
    '''

    options = normalized_jacobian_options(jacobian_options)
    analytic_args = args[:6]
    if options.method == "analytic":
        return linearize_lidar_factor_analytic(*analytic_args)
    if options.method == "finite_difference":
        return linearize_lidar_factor_finite_difference(
            *args[:4],
            kwargs["pose_provider"],
            kwargs["sensor_start_time"],
            kwargs["sensor_end_time"],
            kwargs["lidar_time_offset"],
            finite_difference_epsilon=options.finite_difference_epsilon,
        )
    # Checked mode uses the analytic result and stores one comparison result per
    # local variable block.
    analytic = linearize_lidar_factor_analytic(*analytic_args)
    finite_difference = linearize_lidar_factor_finite_difference(
        *args[:4],
        kwargs["pose_provider"],
        kwargs["sensor_start_time"],
        kwargs["sensor_end_time"],
        kwargs["lidar_time_offset"],
        finite_difference_epsilon=options.finite_difference_epsilon,
    )
    checks = _compare_linearization_blocks(
        _lidar_linearization_blocks(analytic),
        _lidar_linearization_blocks(finite_difference),
        factor_name="lidar_relative_pose",
        options=options,
    )

    return replace(analytic, check_results=checks)