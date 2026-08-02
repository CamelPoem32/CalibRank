from __future__ import annotations

import numpy as np
import pytest

from src.calib_observability.backend import estimate_poses_dummy
from src.calib_observability.jacobians import linearize_gyro_factor, linearize_lidar_factor
from src.calib_observability.observability import effective_observability_dense, effective_observability_sparse_lsmr
from src.calib_observability.simulation import PlanarRoverConfig, simulate_planar_rover
from src.calib_observability.types import JacobianCheckError, JacobianOptions, compare_jacobians, validate_jacobian_method


def _dataset():
    config = PlanarRoverConfig(imu_rate_hz=80.0, lidar_rate_hz=5.0, total_laps=1, random_seed=61)
    dataset = simulate_planar_rover(config, mode="one_rectangle")
    return dataset, estimate_poses_dummy(dataset)


def test_invalid_jacobian_method_raises() -> None:
    with pytest.raises(ValueError):
        validate_jacobian_method("banana")


def test_compare_jacobians_detects_corruption() -> None:
    check = compare_jacobians(np.eye(2), np.eye(2) + 1.0, factor_name="f", variable_name="x", atol=1e-6, rtol=1e-5)
    assert not check.passed
    with pytest.raises(JacobianCheckError):
        from src.calib_observability.jacobians import _check_or_raise
        _check_or_raise([check], JacobianOptions(method="analytic_checked", raise_on_check_failure=True))


def test_all_three_modes_return_equal_factor_residuals_and_checked_results() -> None:
    dataset, provider = _dataset()
    measurement_index = 4
    true_times = np.array([
        dataset.lidar.relative_start_times[measurement_index] + dataset.tau_L_true,
        dataset.lidar.relative_end_times[measurement_index] + dataset.tau_L_true,
    ], dtype=float)
    poses, _, spatial_twists = provider.poses_and_twists_at(true_times)
    analytic = linearize_lidar_factor(poses[0], poses[1], dataset.T_B_L_true, dataset.lidar.measurements[measurement_index], spatial_twists[0], spatial_twists[1], pose_provider=provider, sensor_start_time=float(dataset.lidar.relative_start_times[measurement_index]), sensor_end_time=float(dataset.lidar.relative_end_times[measurement_index]), lidar_time_offset=dataset.tau_L_true, jacobian_options=JacobianOptions(method="analytic"))
    finite = linearize_lidar_factor(poses[0], poses[1], dataset.T_B_L_true, dataset.lidar.measurements[measurement_index], spatial_twists[0], spatial_twists[1], pose_provider=provider, sensor_start_time=float(dataset.lidar.relative_start_times[measurement_index]), sensor_end_time=float(dataset.lidar.relative_end_times[measurement_index]), lidar_time_offset=dataset.tau_L_true, jacobian_options=JacobianOptions(method="finite_difference"))
    checked = linearize_lidar_factor(poses[0], poses[1], dataset.T_B_L_true, dataset.lidar.measurements[measurement_index], spatial_twists[0], spatial_twists[1], pose_provider=provider, sensor_start_time=float(dataset.lidar.relative_start_times[measurement_index]), sensor_end_time=float(dataset.lidar.relative_end_times[measurement_index]), lidar_time_offset=dataset.tau_L_true, jacobian_options=JacobianOptions(method="analytic_checked"))
    assert np.allclose(analytic.residual, finite.residual)
    assert np.allclose(analytic.H_T_B_L, finite.H_T_B_L, atol=1e-6)
    assert np.allclose(checked.H_T_B_L, analytic.H_T_B_L)
    assert checked.check_results and all(result.passed for result in checked.check_results)


def test_pipeline_matrices_agree_for_analytic_and_finite_difference() -> None:
    dataset, provider = _dataset()
    start = 1.0
    end = 3.0
    analytic_bundle, _, _ = dataset.window_jacobians(start, end, provider, use_sparse=False, jacobian_options=JacobianOptions(method="analytic"))
    finite_bundle, _, _ = dataset.window_jacobians(start, end, provider, use_sparse=False, jacobian_options=JacobianOptions(method="finite_difference"))
    checked_bundle, _, _ = dataset.window_jacobians(start, end, provider, use_sparse=False, jacobian_options=JacobianOptions(method="analytic_checked"))
    assert np.allclose(analytic_bundle.residual, finite_bundle.residual, atol=1e-10)
    assert np.allclose(analytic_bundle.J, finite_bundle.J, atol=2e-5)
    assert np.allclose(analytic_bundle.J_T, finite_bundle.J_T, atol=2e-5)
    assert np.allclose(analytic_bundle.J_C, finite_bundle.J_C, atol=2e-5)
    assert checked_bundle.metadata["all_jacobian_checks_passed"] is True
    O_analytic = effective_observability_dense(analytic_bundle.J_T, analytic_bundle.J_C)
    O_finite = effective_observability_dense(finite_bundle.J_T, finite_bundle.J_C)
    # Projection through P_T_perp can amplify tiny local finite-difference noise.\n    assert np.allclose(O_analytic, O_finite, atol=3e-4)



def test_sparse_pipeline_uses_sparse_projection_result() -> None:
    dataset, provider = _dataset()
    start = 1.0
    end = 3.0
    analytic_bundle, _, _ = dataset.window_jacobians(
        start,
        end,
        provider,
        use_sparse=True,
        jacobian_options=JacobianOptions(method="analytic"),
    )
    finite_bundle, _, _ = dataset.window_jacobians(
        start,
        end,
        provider,
        use_sparse=True,
        jacobian_options=JacobianOptions(method="finite_difference"),
    )

    analytic_projection = effective_observability_sparse_lsmr(analytic_bundle.J_T, analytic_bundle.J_C)
    finite_projection = effective_observability_sparse_lsmr(finite_bundle.J_T, finite_bundle.J_C)

    assert np.allclose(analytic_bundle.residual, finite_bundle.residual, atol=1e-10)
    assert np.allclose(analytic_bundle.J.toarray(), finite_bundle.J.toarray(), atol=2e-5)
    assert np.allclose(analytic_projection.O_C.toarray(), finite_projection.O_C.toarray(), atol=3e-4)
    assert np.allclose(analytic_projection.S_C, finite_projection.S_C, atol=1e-2, rtol=1e-5)
