from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calib_observability.lie_se2 import (
    se2_adjoint,
    se2_exp,
    se2_inverse,
    se2_left_jacobian,
    se2_left_jacobian_inverse,
    se2_log,
)


def test_se2_exp_log_inverse_round_trip() -> None:
    rng = np.random.default_rng(3)
    for _ in range(25):
        xi = rng.normal(0.0, [0.4, 0.5, 0.3])
        T = se2_exp(xi)
        assert np.allclose(se2_log(T), xi, atol=1e-10)
        assert np.allclose(se2_exp(se2_log(T)), T, atol=1e-10)
        assert np.allclose(se2_inverse(T) @ T, np.eye(3), atol=1e-11)


def test_se2_adjoint_identity() -> None:
    xi = np.array([0.3, 0.2, -0.1])
    eta = np.array([-0.05, 0.04, 0.02])
    T = se2_exp(xi)
    lhs = T @ se2_exp(eta) @ se2_inverse(T)
    rhs = se2_exp(se2_adjoint(T) @ eta)
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_se2_left_jacobian_inverse() -> None:
    xi = np.array([0.2, 0.4, -0.1])
    assert np.allclose(se2_left_jacobian_inverse(xi) @ se2_left_jacobian(xi), np.eye(3), atol=1e-10)
