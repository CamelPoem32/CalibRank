from __future__ import annotations

import numpy as np

from src.calib_observability.interpolation import (
    interpolate_se3,
    interpolate_se3_with_twist,
    interpolate_so3,
    interpolate_so3_with_angular_velocity,
)
from src.calib_observability.lie_so3 import so3_exp, so3_log
from src.calib_observability.lie_se3 import se3_adjoint, se3_exp, se3_inverse, se3_log


def test_so3_geodesic_endpoints_and_angular_velocity() -> None:
    R0 = so3_exp(np.array([0.1, -0.05, 0.03]))
    R1 = R0 @ so3_exp(np.array([0.2, 0.1, -0.04]))
    assert np.allclose(interpolate_so3(R0, R1, 0.0), R0)
    assert np.allclose(interpolate_so3(R0, R1, 1.0), R1)
    Rt, omega_body, omega_spatial = interpolate_so3_with_angular_velocity(R0, R1, 1.0, 3.0, 1.7)
    assert np.allclose(Rt.T @ Rt, np.eye(3), atol=1e-10)
    assert np.allclose(omega_spatial, Rt @ omega_body)
    dt = 1e-6
    Rp = interpolate_so3_with_angular_velocity(R0, R1, 1.0, 3.0, 1.7 + dt)[0]
    Rm = interpolate_so3_with_angular_velocity(R0, R1, 1.0, 3.0, 1.7 - dt)[0]
    omega_fd = so3_log(Rm.T @ Rp) / (2.0 * dt)
    assert np.allclose(omega_fd, omega_body, atol=1e-6)


def test_se3_geodesic_endpoints_and_twists() -> None:
    T0 = se3_exp(np.array([0.1, -0.05, 0.03, 0.2, -0.1, 0.05]))
    T1 = T0 @ se3_exp(np.array([0.15, 0.08, -0.04, 0.3, 0.2, -0.1]))
    assert np.allclose(interpolate_se3(T0, T1, 0.0), T0)
    assert np.allclose(interpolate_se3(T0, T1, 1.0), T1)
    Tt, body_twist, spatial_twist = interpolate_se3_with_twist(T0, T1, 1.0, 3.0, 1.7)
    assert np.allclose(spatial_twist, se3_adjoint(Tt) @ body_twist)
    dt = 1e-6
    Tp = interpolate_se3_with_twist(T0, T1, 1.0, 3.0, 1.7 + dt)[0]
    Tm = interpolate_se3_with_twist(T0, T1, 1.0, 3.0, 1.7 - dt)[0]
    twist_fd = se3_log(se3_inverse(Tm) @ Tp) / (2.0 * dt)
    assert np.allclose(twist_fd, body_twist, atol=1e-6)
