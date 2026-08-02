from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calib_observability.lie_se3 import (
    se3_adjoint,
    se3_exp,
    se3_inverse,
    se3_left_jacobian,
    se3_left_jacobian_inverse,
    se3_log,
)


def test_se3_exp_log_inverse_round_trip() -> None:
    rng = np.random.default_rng(2)
    for _ in range(25):
        xi = rng.normal(0.0, [0.25, 0.2, 0.15, 0.5, 0.4, 0.3])
        T = se3_exp(xi)
        assert np.allclose(se3_log(T), xi, atol=1e-9)
        assert np.allclose(se3_exp(se3_log(T)), T, atol=1e-10)
        assert np.allclose(se3_inverse(T) @ T, np.eye(4), atol=1e-11)


def test_se3_adjoint_identity() -> None:
    xi = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2])
    eta = np.array([-0.03, 0.04, 0.02, 0.1, 0.05, -0.02])
    T = se3_exp(xi)
    lhs = T @ se3_exp(eta) @ se3_inverse(T)
    rhs = se3_exp(se3_adjoint(T) @ eta)
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_se3_left_jacobian_inverse() -> None:
    xi = np.array([0.1, -0.08, 0.05, 0.2, 0.1, -0.15])
    J = se3_left_jacobian(xi)
    Jinv = se3_left_jacobian_inverse(xi)
    assert np.allclose(Jinv @ J, np.eye(6), atol=1e-10)
