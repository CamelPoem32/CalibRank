from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calib_observability.whitening import whiten_residual_and_jacobian_dense


def test_whitening_matches_information_norm() -> None:
    r = np.array([0.3, -0.2, 0.1])
    H = np.arange(12, dtype=float).reshape(3, 4) / 10.0
    Sigma = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, 0.0], [0.1, 0.0, 1.0]])
    r_bar, H_bar = whiten_residual_and_jacobian_dense(r, H, Sigma)
    assert H_bar.shape == H.shape
    lhs = float(r_bar @ r_bar)
    rhs = float(r @ np.linalg.inv(Sigma) @ r)
    assert np.allclose(lhs, rhs, atol=1e-12)
