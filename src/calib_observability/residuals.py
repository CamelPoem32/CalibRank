'''Residual functions from `phd_proposal_draft.tex`.

The canonical pose residual is prediction-first:
`r = Log(prediction @ inverse(measurement))`.
'''

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline, interp1d

from .conventions import as_matrix, as_vector, ensure_same_length
from .lie_so3 import so3_exp, so3_log
from .lie_se2 import se2_adjoint, se2_inverse, se2_log
from .lie_se3 import se3_adjoint, se3_inverse, se3_log


##################################################
# Gyroscope interpolation and integration
##################################################
def _gyro_interpolator(
    sample_times: ArrayLike,
    omega_samples: ArrayLike,
    interpolation: str,
) -> Callable[[ArrayLike], NDArray[np.float64]]:
    '''Build an interpolator for a three-axis gyroscope signal.
    
    Args:
        sample_times: Strictly increasing sample timestamps, shape ``(N,)``.
        omega_samples: Angular-velocity samples, shape ``(N, 3)``.
        interpolation: Interpolation method, either ``"linear"`` or ``"cubic"``.
    
    Returns:
        Callable that evaluates angular velocity at scalar or array timestamps.
    
    Raises:
        ValueError: If shapes, values, timestamp ordering, or the interpolation
            method are invalid.
    '''
    times = np.asarray(sample_times, dtype=float)
    omega = np.asarray(omega_samples, dtype=float)
    if times.ndim != 1 or omega.ndim != 2 or omega.shape[1] != 3:
        raise ValueError("sample_times must be (N,), omega_samples must be (N, 3)")
    ensure_same_length([times, omega], ["sample_times", "omega_samples"])
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(omega)):
        raise ValueError("sample_times and omega_samples must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("sample_times must be strictly increasing")
    if interpolation == "linear":
        f = interp1d(times, omega, axis=0, kind="linear", fill_value="extrapolate")
        return lambda t: np.asarray(f(t), dtype=float)
    if interpolation == "cubic":
        spline = CubicSpline(times, omega, axis=0, extrapolate=True)
        return lambda t: np.asarray(spline(t), dtype=float)
    raise ValueError("interpolation must be 'linear' or 'cubic'")


def _integrate_gyro_signal(
    sample_times: ArrayLike,
    omega_samples: ArrayLike,
    lower_time: float,
    upper_time: float,
    bias: NDArray[np.float64],
    interpolation: str,
) -> NDArray[np.float64]:
    '''Integrate a bias-corrected interpolated gyroscope signal.
    
    Linear interpolation is integrated exactly by splitting the interval at sample
    timestamps. Cubic interpolation uses the analytic spline antiderivative.
    
    Args:
        sample_times: Gyroscope timestamps, shape ``(N,)``.
        omega_samples: Angular-velocity samples, shape ``(N, 3)``.
        lower_time: Lower integration bound.
        upper_time: Upper integration bound.
        bias: Constant gyroscope bias, shape ``(3,)``.
        interpolation: Interpolation method, ``"linear"`` or ``"cubic"``.
    
    Returns:
        Integrated rotation vector, shape ``(3,)``.
    
    Raises:
        ValueError: If the interpolation method is unsupported or the sampled
            signal is invalid.
    '''

    times = np.asarray(sample_times, dtype=float)
    omega = np.asarray(omega_samples, dtype=float)
    if interpolation == "linear":
        # Linear interpolation is piecewise linear with a kink at every sample.
        # Splitting at those samples gives the exact trapezoidal integral and
        # avoids SciPy adaptive quadrature spending subdivisions on each kink.
        interpolator = _gyro_interpolator(times, omega, interpolation)
        interior_times = times[(times > lower_time) & (times < upper_time)]
        integration_times = np.r_[lower_time, interior_times, upper_time]
        interpolated_omega = interpolator(integration_times)
        corrected_omega = interpolated_omega - bias[None, :]
        time_steps = np.diff(integration_times)
        return np.sum(
            0.5 * time_steps[:, None] * (corrected_omega[:-1] + corrected_omega[1:]),
            axis=0,
        )
    if interpolation == "cubic":
        _gyro_interpolator(times, omega, interpolation)
        spline = CubicSpline(times, omega, axis=0, extrapolate=True)
        spline_integral = np.asarray(spline.integrate(lower_time, upper_time), dtype=float)
        return spline_integral - bias * (upper_time - lower_time)
    raise ValueError("interpolation must be 'linear' or 'cubic'")


def gyro_increment_from_signal(
    sample_times: ArrayLike,
    omega_samples: ArrayLike,
    t_k: float,
    t_k1: float,
    tau: float,
    b_g: ArrayLike | None = None,
    *,
    interpolation: str = "linear",
) -> NDArray[np.float64]:
    '''Integrate the bias-corrected gyroscope signal over shifted limits.
    
    Implements `phi_k(tau) = integral_{t_k+tau}^{t_k1+tau}
    (omega(s) - b_g) ds` from the gyroscope propagation subsection.
    
    Args:
        sample_times: Gyroscope sample times, shape `(N,)`.
        omega_samples: Gyroscope samples, shape `(N, 3)`.
        t_k, t_k1: Nominal interval endpoints.
        tau: Temporal offset in seconds.
        b_g: Gyroscope bias, shape `(3,)`; zero when omitted.
        interpolation: `"linear"` or `"cubic"`.
    
    Returns:
        ndarray, shape `(3,)`
    
    Raises:
        ValueError: If dimensions, finite checks, or interval ordering are invalid.
    
    Notes:
        Perturbation convention: This scalar time-offset factor is paired with left-perturbation pose residuals.
    '''

    if not np.all(np.isfinite([t_k, t_k1, tau])):
        raise ValueError("t_k, t_k1, and tau must be finite")
    if t_k1 <= t_k:
        raise ValueError("t_k1 must be greater than t_k")
    bias = np.zeros(3) if b_g is None else as_vector(b_g, 3, "b_g")
    lower_time = float(t_k + tau)
    upper_time = float(t_k1 + tau)
    return _integrate_gyro_signal(sample_times, omega_samples, lower_time, upper_time, bias, interpolation)


##################################################
# Gyroscope propagation residual
##################################################
def gyro_propagation_residual(
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
    '''Compute the SO(3) gyroscope propagation residual.
    
    Implements `r_k = Log(R_k Delta_R_k(tau) R_{k+1}^{-1})`.
    
    Args:
        R_k, R_k1: Body rotations at interval endpoints, each shape `(3, 3)`.
        sample_times, omega_samples: Gyroscope samples with shapes `(N,)` and `(N, 3)`.
        t_k, t_k1, tau: Nominal endpoints and temporal offset.
        b_g: Gyroscope bias, shape `(3,)`.
        interpolation: `"linear"` or `"cubic"`.
    
    Returns:
        ndarray, shape `(3,)`
    
    Raises:
        ValueError: If inputs are invalid.
    
    Notes:
        Perturbation convention: Pose perturbations are left perturbations.
    '''

    R0 = as_matrix(R_k, (3, 3), "R_k")
    R1 = as_matrix(R_k1, (3, 3), "R_k1")
    phi = gyro_increment_from_signal(
        sample_times, omega_samples, t_k, t_k1, tau, b_g, interpolation=interpolation
    )
    Delta_R = so3_exp(phi)
    # R0: (3, 3), Delta_R: (3, 3), R1.T: (3, 3) -> R_r: (3, 3)
    R_r = R0 @ Delta_R @ R1.T
    return so3_log(R_r)


##################################################
# SE(3) relative-pose residuals
##################################################
def relative_body_motion(T_W_B_m: ArrayLike, T_W_B_m1: ArrayLike) -> NDArray[np.float64]:
    '''Compute body relative motion `A_m = T_W_B_m^{-1} T_W_B_m1`.
    
    Args:
        T_W_B_m, T_W_B_m1: SE(3) body poses, each shape `(4, 4)`.
    
    Returns:
        ndarray, shape `(4, 4)`
    
    Raises:
        ValueError: If inputs are invalid.
    
    Notes:
        Perturbation convention: Used by residuals linearized with left pose perturbations.
    '''

    T0 = as_matrix(T_W_B_m, (4, 4), "T_W_B_m")
    T1 = as_matrix(T_W_B_m1, (4, 4), "T_W_B_m1")
    # inv(T0): (4, 4), T1: (4, 4) -> A_m: (4, 4)
    return se3_inverse(T0) @ T1


def sensor_relative_prediction(
    T_W_B_m: ArrayLike, T_W_B_m1: ArrayLike, X: ArrayLike
) -> NDArray[np.float64]:
    '''Predict a sensor relative pose `Z_hat = X^{-1} A_m X`.
    
    Args:
        T_W_B_m, T_W_B_m1: Body poses, shape `(4, 4)`.
        X: Calibration transform such as `T_B_L`, shape `(4, 4)`.
    
    Returns:
        ndarray, shape `(4, 4)`
    
    Raises:
        ValueError: If inputs are invalid.
    
    Notes:
        Perturbation convention: Calibration perturbations are left perturbations.
    '''

    X0 = as_matrix(X, (4, 4), "X")
    A_m = relative_body_motion(T_W_B_m, T_W_B_m1)
    # inv(X0): (4, 4), A_m: (4, 4), X0: (4, 4) -> Z_hat: (4, 4)
    return se3_inverse(X0) @ A_m @ X0


def relative_pose_residual_prediction_first(
    Z_hat: ArrayLike, Z: ArrayLike
) -> NDArray[np.float64]:
    '''Canonical relative-pose residual `Log(Z_hat @ inverse(Z))`.
    
    Args:
        Z_hat: Predicted SE(3) relative pose, shape `(4, 4)`.
        Z: Measured SE(3) relative pose, shape `(4, 4)`.
    
    Returns:
        ndarray, shape `(6,)`
    
    Raises:
        ValueError: If inputs are invalid.
    
    Notes:
        Perturbation convention: Used everywhere by default for SE(3) residuals.
    '''

    Zh = as_matrix(Z_hat, (4, 4), "Z_hat")
    Zm = as_matrix(Z, (4, 4), "Z")
    # Zh: (4, 4), inv(Zm): (4, 4) -> residual_transform: (4, 4)
    return se3_log(Zh @ se3_inverse(Zm))


def relative_pose_residual_measurement_first(
    Z_hat: ArrayLike, Z: ArrayLike
) -> NDArray[np.float64]:
    '''Alternative residual `Log(inverse(Z) @ Z_hat)`.
    
    This is implemented only under an explicit name because the package
    canonical convention is prediction-first.
    
    Args:
        Z_hat, Z: Predicted and measured SE(3) relative poses, each shape `(4, 4)`.
    
    Returns:
        ndarray, shape `(6,)`
    
    Raises:
        ValueError: If inputs are invalid.
    '''

    Zh = as_matrix(Z_hat, (4, 4), "Z_hat")
    Zm = as_matrix(Z, (4, 4), "Z")
    # inv(Zm): (4, 4), Zh: (4, 4) -> residual_transform: (4, 4)
    return se3_log(se3_inverse(Zm) @ Zh)


def spatial_smoothness_residual(X_m: ArrayLike, X_m1: ArrayLike) -> NDArray[np.float64]:
    '''Compute `r_smooth = Log(X_m^{-1} X_m1)` in SE(3).
    
    Args:
        X_m, X_m1: Consecutive calibration transforms, each shape `(4, 4)`.
    
    Returns:
        ndarray, shape `(6,)`
    
    Raises:
        ValueError: If inputs are invalid.
    
    Notes:
        Perturbation convention: Jacobians use left perturbations on both transforms.
    '''

    X0 = as_matrix(X_m, (4, 4), "X_m")
    X1 = as_matrix(X_m1, (4, 4), "X_m1")
    # inv(X0): (4, 4), X1: (4, 4) -> residual_transform: (4, 4)
    return se3_log(se3_inverse(X0) @ X1)


def extrinsic_prior_residual(X: ArrayLike, X_0: ArrayLike) -> NDArray[np.float64]:
    '''Compute `r_prior = Log(X^{-1} X_0)` in SE(3).
    
    Args:
        X: Current calibration transform, shape `(4, 4)`.
        X_0: Nominal calibration transform, shape `(4, 4)`.
    
    Returns:
        ndarray, shape `(6,)`
    
    Raises:
        ValueError: If inputs are invalid.
    
    Notes:
        Perturbation convention: The Jacobian is with respect to a left perturbation of `X`.
    '''

    Xc = as_matrix(X, (4, 4), "X")
    Xn = as_matrix(X_0, (4, 4), "X_0")
    # inv(Xc): (4, 4), Xn: (4, 4) -> residual_transform: (4, 4)
    return se3_log(se3_inverse(Xc) @ Xn)


def motion_only_sensitivity_block(A_m: ArrayLike) -> NDArray[np.float64]:
    '''Return `Adj(A_m) - I_6` for fixed-motion extrinsic sensitivity.
    
    Args:
        A_m: Body relative motion, shape `(4, 4)`.
    
    Returns:
        ndarray, shape `(6, 6)`
    
    Raises:
        ValueError: If `A_m` is invalid.
    '''

    A = as_matrix(A_m, (4, 4), "A_m")
    return se3_adjoint(A) - np.eye(6)


##################################################
# SE(2) relative-pose residuals
##################################################
def relative_pose_residual_prediction_first_se2(
    Z_hat: ArrayLike, Z: ArrayLike
) -> NDArray[np.float64]:
    '''SE(2) canonical residual `Log(Z_hat @ inverse(Z))`.
    
    Args:
        Z_hat, Z: Predicted and measured SE(2) relative poses, each shape `(3, 3)`.
    
    Returns:
        ndarray, shape `(3,)`
    
    Raises:
        ValueError: If inputs are invalid.
    
    Notes:
        Perturbation convention: Left perturbations with tangent `[omega, v_x, v_y]`.
    '''

    Zh = as_matrix(Z_hat, (3, 3), "Z_hat")
    Zm = as_matrix(Z, (3, 3), "Z")
    # Zh: (3, 3), inv(Zm): (3, 3) -> residual_transform: (3, 3)
    return se2_log(Zh @ se2_inverse(Zm))


def relative_pose_residual_measurement_first_se2(
    Z_hat: ArrayLike, Z: ArrayLike
) -> NDArray[np.float64]:
    '''Calculate the alternative measurement-first SE(2) residual.
    
    Args:
        Z_hat: Predicted planar relative pose, shape ``(3, 3)``.
        Z: Measured planar relative pose, shape ``(3, 3)``.
    
    Returns:
        Residual ``Log(inverse(Z) @ Z_hat)``, shape ``(3,)``.
    '''

    Zh = as_matrix(Z_hat, (3, 3), "Z_hat")
    Zm = as_matrix(Z, (3, 3), "Z")
    # inv(Zm): (3, 3), Zh: (3, 3) -> residual_transform: (3, 3)
    return se2_log(se2_inverse(Zm) @ Zh)


def relative_body_motion_se2(T_W_B_m: ArrayLike, T_W_B_m1: ArrayLike) -> NDArray[np.float64]:
    '''Calculate planar body motion between two world poses.
    
    Args:
        T_W_B_m: Body pose at the first timestamp, shape ``(3, 3)``.
        T_W_B_m1: Body pose at the second timestamp, shape ``(3, 3)``.
    
    Returns:
        Relative motion ``inverse(T_W_B_m) @ T_W_B_m1``.
    '''

    T0 = as_matrix(T_W_B_m, (3, 3), "T_W_B_m")
    T1 = as_matrix(T_W_B_m1, (3, 3), "T_W_B_m1")
    # inv(T0): (3, 3), T1: (3, 3) -> A_m: (3, 3)
    return se2_inverse(T0) @ T1


def sensor_relative_prediction_se2(
    T_W_B_m: ArrayLike, T_W_B_m1: ArrayLike, X: ArrayLike
) -> NDArray[np.float64]:
    '''Predict planar sensor motion from body motion and extrinsics.
    
    Args:
        T_W_B_m: Body pose at the first timestamp, shape ``(3, 3)``.
        T_W_B_m1: Body pose at the second timestamp, shape ``(3, 3)``.
        X: Sensor extrinsic transform in the body frame, shape ``(3, 3)``.
    
    Returns:
        Predicted sensor motion ``inverse(X) @ A_m @ X``.
    '''

    X0 = as_matrix(X, (3, 3), "X")
    A_m = relative_body_motion_se2(T_W_B_m, T_W_B_m1)
    # inv(X0): (3, 3), A_m: (3, 3), X0: (3, 3) -> Z_hat: (3, 3)
    return se2_inverse(X0) @ A_m @ X0


def spatial_smoothness_residual_se2(X_m: ArrayLike, X_m1: ArrayLike) -> NDArray[np.float64]:
    '''Calculate the planar calibration smoothness residual.
    
    Args:
        X_m: Calibration transform at the first index, shape ``(3, 3)``.
        X_m1: Calibration transform at the next index, shape ``(3, 3)``.
    
    Returns:
        Residual ``Log(inverse(X_m) @ X_m1)``, shape ``(3,)``.
    '''

    X0 = as_matrix(X_m, (3, 3), "X_m")
    X1 = as_matrix(X_m1, (3, 3), "X_m1")
    # inv(X0): (3, 3), X1: (3, 3) -> residual_transform: (3, 3)
    return se2_log(se2_inverse(X0) @ X1)


def extrinsic_prior_residual_se2(X: ArrayLike, X_0: ArrayLike) -> NDArray[np.float64]:
    '''Calculate the planar extrinsic-prior residual.
    
    Args:
        X: Current calibration transform, shape ``(3, 3)``.
        X_0: Nominal calibration transform, shape ``(3, 3)``.
    
    Returns:
        Residual ``Log(inverse(X) @ X_0)``, shape ``(3,)``.
    '''

    Xc = as_matrix(X, (3, 3), "X")
    Xn = as_matrix(X_0, (3, 3), "X_0")
    # inv(Xc): (3, 3), Xn: (3, 3) -> residual_transform: (3, 3)
    return se2_log(se2_inverse(Xc) @ Xn)


def motion_only_sensitivity_block_se2(A_m: ArrayLike) -> NDArray[np.float64]:
    '''Calculate one planar motion-only sensitivity block.
    
    Args:
        A_m: Planar body relative motion, shape ``(3, 3)``.
    
    Returns:
        Matrix ``Adj(A_m) - I_3``, shape ``(3, 3)``.
    '''

    A = as_matrix(A_m, (3, 3), "A_m")
    return se2_adjoint(A) - np.eye(3)