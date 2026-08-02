from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calib_observability.finite_difference import finite_difference_left_jacobian_se2, finite_difference_left_jacobian_se3
from src.calib_observability.jacobians import (
    extrinsic_prior_jacobian_left,
    extrinsic_prior_jacobian_left_se2,
    gyro_temporal_offset_jacobian,
    pose_residual_calibration_jacobian_left,
    pose_residual_calibration_jacobian_left_se2,
    spatial_smoothness_jacobians_left,
    spatial_smoothness_jacobians_left_se2,
)
from src.calib_observability.lie_se2 import se2_exp
from src.calib_observability.lie_se3 import se3_exp
from src.calib_observability.residuals import (
    gyro_propagation_residual,
    sensor_relative_prediction,
    sensor_relative_prediction_se2,
    relative_pose_residual_prediction_first,
    relative_pose_residual_prediction_first_se2,
    spatial_smoothness_residual,
    spatial_smoothness_residual_se2,
    extrinsic_prior_residual,
    extrinsic_prior_residual_se2,
)


def _errors(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    max_abs = float(np.max(np.abs(A - B)))
    rel = float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-15))
    return max_abs, rel


def test_spatial_smoothness_jacobians_se3_finite_difference() -> None:
    X0 = se3_exp(np.array([0.05, -0.02, 0.03, 0.2, -0.1, 0.3]))
    X1 = se3_exp(np.array([0.06, -0.01, 0.035, 0.22, -0.09, 0.32]))
    jac = spatial_smoothness_jacobians_left(X0, X1)
    H0 = finite_difference_left_jacobian_se3(lambda X: spatial_smoothness_residual(X, X1), X0)
    H1 = finite_difference_left_jacobian_se3(lambda X: spatial_smoothness_residual(X0, X), X1)
    assert _errors(jac.H_X_m, H0)[0] < 2e-6
    assert _errors(jac.H_X_m1, H1)[0] < 2e-6


def test_prior_jacobian_se3_finite_difference() -> None:
    X = se3_exp(np.array([0.05, -0.02, 0.03, 0.2, -0.1, 0.3]))
    X0 = se3_exp(np.array([0.052, -0.019, 0.031, 0.21, -0.095, 0.31]))
    jac = extrinsic_prior_jacobian_left(X, X0)
    H = finite_difference_left_jacobian_se3(lambda Y: extrinsic_prior_residual(Y, X0), X)
    assert _errors(jac.H_X, H)[0] < 2e-6


def test_pose_calibration_jacobian_se3_finite_difference() -> None:
    T0 = se3_exp(np.array([0.02, -0.01, 0.03, 0.1, 0.0, 0.0]))
    T1 = se3_exp(np.array([0.04, -0.015, 0.08, 0.5, 0.2, 0.05]))
    X = se3_exp(np.array([0.01, 0.02, -0.02, 0.3, -0.1, 0.2]))
    Z_true = sensor_relative_prediction(T0, T1, X)
    Z = se3_exp(np.array([1e-4, -2e-4, 1.5e-4, 3e-4, -1e-4, 2e-4])) @ Z_true
    jac = pose_residual_calibration_jacobian_left(T0, T1, X, Z)
    H = finite_difference_left_jacobian_se3(
        lambda Y: relative_pose_residual_prediction_first(sensor_relative_prediction(T0, T1, Y), Z),
        X,
    )
    assert _errors(jac.H_X, H)[0] < 5e-6


def test_gyro_temporal_offset_jacobian_finite_difference() -> None:
    times = np.linspace(0.0, 2.0, 200)
    omega = np.c_[0.01 * times, 0.02 * np.sin(times), 0.1 + 0.03 * times**2]
    b_g = np.array([0.001, -0.002, 0.003])
    t0, t1, tau = 0.3, 1.1, 0.02
    R0 = np.eye(3)
    from src.calib_observability.lie_so3 import so3_exp
    from src.calib_observability.residuals import gyro_increment_from_signal

    R1 = R0 @ so3_exp(gyro_increment_from_signal(times, omega, t0, t1, tau, b_g, interpolation="cubic"))
    H = gyro_temporal_offset_jacobian(R0, R1, times, omega, t0, t1, tau, b_g, interpolation="cubic")
    eps = 1e-7
    rp = gyro_propagation_residual(R0, R1, times, omega, t0, t1, tau + eps, b_g, interpolation="cubic")
    rm = gyro_propagation_residual(R0, R1, times, omega, t0, t1, tau - eps, b_g, interpolation="cubic")
    H_fd = (rp - rm) / (2 * eps)
    assert _errors(H, H_fd)[0] < 2e-6


def test_se2_jacobians_finite_difference() -> None:
    X0 = se2_exp(np.array([0.1, 0.2, -0.1]))
    X1 = se2_exp(np.array([0.12, 0.23, -0.08]))
    sm = spatial_smoothness_jacobians_left_se2(X0, X1)
    H0 = finite_difference_left_jacobian_se2(lambda X: spatial_smoothness_residual_se2(X, X1), X0)
    assert _errors(sm.H_X_m, H0)[0] < 2e-6

    pr = extrinsic_prior_jacobian_left_se2(X0, X1)
    Hp = finite_difference_left_jacobian_se2(lambda X: extrinsic_prior_residual_se2(X, X1), X0)
    assert _errors(pr.H_X, Hp)[0] < 2e-6

    T0 = se2_exp(np.array([0.02, 0.0, 0.0]))
    T1 = se2_exp(np.array([0.4, 1.0, 0.4]))
    Z = sensor_relative_prediction_se2(T0, T1, X0)
    cal = pose_residual_calibration_jacobian_left_se2(T0, T1, X0, Z)
    Hc = finite_difference_left_jacobian_se2(
        lambda X: relative_pose_residual_prediction_first_se2(sensor_relative_prediction_se2(T0, T1, X), Z),
        X0,
    )
    assert _errors(cal.H_X, Hc)[0] < 5e-6
