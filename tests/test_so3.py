from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calib_observability.lie_so3 import (
    so3_exp,
    so3_hat,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
    so3_vee,
)


def test_so3_exp_log_round_trip() -> None:
    rng = np.random.default_rng(1)
    for _ in range(30):
        omega = rng.normal(0.0, 0.4, 3)
        assert np.allclose(so3_log(so3_exp(omega)), omega, atol=1e-10)
        R = so3_exp(omega)
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-10)


def test_so3_hat_vee_and_left_jacobian_inverse() -> None:
    omega = np.array([0.2, -0.1, 0.05])
    assert np.allclose(so3_vee(so3_hat(omega)), omega)
    J = so3_left_jacobian(omega)
    Jinv = so3_left_jacobian_inverse(omega)
    assert np.allclose(Jinv @ J, np.eye(3), atol=1e-11)
