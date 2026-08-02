from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calib_observability.lie_se3 import se3_exp
from src.calib_observability.residuals import (
    relative_body_motion,
    relative_pose_residual_measurement_first,
    relative_pose_residual_prediction_first,
    sensor_relative_prediction,
    spatial_smoothness_residual,
    extrinsic_prior_residual,
)


def test_prediction_first_residual_zero_for_matching_measurement() -> None:
    T0 = se3_exp(np.array([0.1, -0.05, 0.02, 0.2, 0.0, 0.1]))
    T1 = se3_exp(np.array([0.2, 0.03, -0.01, 0.8, 0.2, -0.1]))
    X = se3_exp(np.array([0.02, 0.01, -0.03, 0.4, -0.1, 0.2]))
    Z = sensor_relative_prediction(T0, T1, X)
    assert np.allclose(relative_pose_residual_prediction_first(Z, Z), np.zeros(6), atol=1e-12)
    assert np.allclose(relative_pose_residual_measurement_first(Z, Z), np.zeros(6), atol=1e-12)
    assert np.allclose(relative_body_motion(T0, T1), np.linalg.solve(T0, T1))


def test_smoothness_and_prior_zero() -> None:
    X = se3_exp(np.array([0.02, 0.01, -0.03, 0.4, -0.1, 0.2]))
    assert np.allclose(spatial_smoothness_residual(X, X), np.zeros(6), atol=1e-12)
    assert np.allclose(extrinsic_prior_residual(X, X), np.zeros(6), atol=1e-12)
