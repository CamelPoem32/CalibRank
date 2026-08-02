from __future__ import annotations

import numpy as np

from src.calib_observability.accelerometer import (
    complex_accelerometer_terms,
    linearize_complex_accelerometer_factor,
    linearize_simple_accelerometer_factor,
)
from src.calib_observability.backend import estimate_poses_dummy
from src.calib_observability.factor_observability import (
    build_accelerometer_motion_sensitivity,
    effective_target_observability_from_bundle_dense,
    effective_target_observability_from_bundle_sparse,
)
from src.calib_observability.lie_se3 import se3_exp
from src.calib_observability.simulation import PlanarRoverConfig, reframe_dataset_to_fixed_extrinsic, simulate_planar_rover
from src.calib_observability.types import AccelerometerOptions, JacobianOptions, PracticalRankPolicy
from src.calib_observability.visualization.quasi_realtime_rover import _rank_dashboard_text, build_observability_visualization_series


def _dataset(mode: str = "one_rectangle"):
    config = PlanarRoverConfig(
        rectangle_width=2.0,
        rectangle_height=1.0,
        straight_speed=1.0,
        turn_duration=0.5,
        imu_rate_hz=50.0,
        lidar_rate_hz=5.0,
        gyro_noise_std=1e-6,
        accel_noise_std=1e-6,
        random_seed=123,
        mode=mode,
    )
    return reframe_dataset_to_fixed_extrinsic(simulate_planar_rover(config, mode=mode), "T_B_L")


def test_disabled_mode_matches_omitted_accelerometer_options() -> None:
    dataset = _dataset()
    provider = estimate_poses_dummy(dataset)
    omitted, _, omitted_counts = dataset.window_jacobians(1.0, 2.5, provider, use_sparse=False, fixed_extrinsic="T_B_L")
    disabled, _, disabled_counts = dataset.window_jacobians(
        1.0,
        2.5,
        provider,
        use_sparse=False,
        fixed_extrinsic="T_B_L",
        accelerometer_options=AccelerometerOptions(mode="disabled"),
    )
    assert omitted.J.shape == disabled.J.shape
    assert omitted.J_T.shape == disabled.J_T.shape
    assert omitted.J_C.shape == disabled.J_C.shape
    assert np.allclose(omitted.J, disabled.J)
    assert omitted.row_slices == disabled.row_slices
    assert omitted_counts == disabled_counts
    assert disabled.metadata["accelerometer_factor_count"] == 0


def test_stationary_accelerometer_specific_force_has_gravity_norm() -> None:
    config = PlanarRoverConfig(mode="stationary", imu_rate_hz=20.0, lidar_rate_hz=5.0, accel_noise_std=1e-9, gyro_noise_std=1e-9)
    dataset = simulate_planar_rover(config, mode="stationary")
    norms = np.linalg.norm(dataset.imu.accelerometer, axis=1)
    assert np.allclose(norms, np.linalg.norm(dataset.imu.gravity_world), atol=1e-6)
    assert dataset.imu.accelerometer_convention == "specific_force_imu_frame_R_IW_times_a_minus_g"


def test_simple_accelerometer_jacobians_and_zero_translation_columns() -> None:
    dataset = _dataset()
    provider = estimate_poses_dummy(dataset)
    index = 30
    sensor_time = float(dataset.imu.sensor_timestamps[index])
    true_time = sensor_time + dataset.tau_I_true
    pose = provider.poses_at(np.array([true_time]))[0]
    _, _, spatial_twists = provider.poses_and_twists_at(np.array([true_time]))
    checked = linearize_simple_accelerometer_factor(
        pose,
        dataset.T_B_I_true,
        dataset.imu.accelerometer[index],
        dataset.imu.gravity_world,
        spatial_twists[0],
        pose_provider=provider,
        sensor_time=sensor_time,
        tau_I=dataset.tau_I_true,
        jacobian_options=JacobianOptions(method="analytic_checked"),
    )
    assert checked.check_results and all(result.passed for result in checked.check_results)
    assert np.allclose(checked.pose_blocks[0].matrix[:, 3:6], 0.0)
    assert np.allclose(checked.H_T_B_I[:, 3:6], 0.0)


def test_simple_gating_accepts_low_dynamic_and_rejects_turning_samples() -> None:
    dataset = _dataset()
    provider = estimate_poses_dummy(dataset)
    options = AccelerometerOptions(mode="simple", factor_rate_hz=5.0, measurement_std_m_s2=0.05)
    bundle, _, counts = dataset.window_jacobians(0.0, dataset.end_time, provider, use_sparse=False, fixed_extrinsic="T_B_L", accelerometer_options=options)
    assert counts["accelerometer_candidate_count"] > counts["accelerometer_factor_count"] > 0
    reasons = bundle.metadata["accelerometer_rejection_reasons"]
    assert "gyro_norm_gate_failed" in reasons or "gravity_norm_gate_failed" in reasons


def test_complex_accelerometer_checked_jacobians_and_planar_lever_arm_columns() -> None:
    dataset = _dataset()
    provider = estimate_poses_dummy(dataset)
    h = 0.2
    index = 90
    sensor_time = float(dataset.imu.sensor_timestamps[index])
    true_time = sensor_time + dataset.tau_I_true
    support_times = np.array([true_time - h, true_time, true_time + h])
    poses, _, spatial_twists = provider.poses_and_twists_at(support_times)
    checked = linearize_complex_accelerometer_factor(
        poses[0],
        poses[1],
        poses[2],
        dataset.T_B_I_true,
        dataset.imu.accelerometer[index],
        dataset.imu.gravity_world,
        h,
        spatial_twists[0],
        spatial_twists[1],
        spatial_twists[2],
        pose_provider=provider,
        sensor_time=sensor_time,
        tau_I=dataset.tau_I_true,
        jacobian_options=JacobianOptions(method="analytic_checked"),
    )
    assert checked.check_results and all(result.passed for result in checked.check_results)
    assert checked.H_T_B_I.shape == (3, 6)
    assert np.linalg.matrix_rank(checked.H_T_B_I) <= 3
    assert np.linalg.norm(checked.H_T_B_I[:, 3:5]) > 0.0
    assert np.linalg.norm(checked.H_T_B_I[:, 5]) <= 1e-8


def test_complex_terms_recover_constant_yaw_discrete_kinematics() -> None:
    h = 0.2
    yaw_rate = 0.7
    def yaw_pose(time_seconds: float):
        pose = np.eye(4)
        c, s = np.cos(yaw_rate * time_seconds), np.sin(yaw_rate * time_seconds)
        pose[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return pose
    terms = complex_accelerometer_terms(
        yaw_pose(-h),
        yaw_pose(0.0),
        yaw_pose(h),
        se3_exp([0.0, 0.0, 0.0, 0.2, 0.1, 0.0]),
        np.array([0.0, 0.0, 9.81]),
        np.array([0.0, 0.0, -9.81]),
        h,
    )
    assert np.allclose(terms.omega_zero_body, np.array([0.0, 0.0, yaw_rate]), atol=1e-3)
    assert np.linalg.norm(terms.alpha_zero_body) <= 1e-10
    assert np.linalg.norm(terms.centripetal_acceleration[:2]) > 0.0


def test_global_dense_sparse_accelerometer_outputs_agree() -> None:
    dataset = _dataset()
    provider = estimate_poses_dummy(dataset)
    options = AccelerometerOptions(mode="complex", factor_rate_hz=5.0, measurement_std_m_s2=0.05)
    dense_bundle, _, dense_counts = dataset.window_jacobians(1.0, 2.5, provider, use_sparse=False, fixed_extrinsic="T_B_L", accelerometer_options=options)
    sparse_bundle, _, sparse_counts = dataset.window_jacobians(1.0, 2.5, provider, use_sparse=True, fixed_extrinsic="T_B_L", accelerometer_options=options)
    assert dense_bundle.J_C.shape[1] == 11
    assert "T_B_L" not in dense_bundle.calibration_column_slices
    assert dense_counts["accelerometer_factor_count"] == sparse_counts["accelerometer_factor_count"]
    assert np.allclose(dense_bundle.J, sparse_bundle.J.toarray())
    assert np.allclose(dense_bundle.J_T, sparse_bundle.J_T.toarray())
    assert np.allclose(dense_bundle.J_C, sparse_bundle.J_C.toarray())
    dense_accel = build_accelerometer_motion_sensitivity(dense_bundle)
    sparse_accel = build_accelerometer_motion_sensitivity(sparse_bundle)
    assert np.allclose(dense_accel.C_X_I_accel_physical, sparse_accel.C_X_I_accel_physical.toarray())
    assert dense_accel.practical_rank == sparse_accel.practical_rank
    for variable_name in ("T_B_I", "b_g", "tau_I", "tau_L"):
        dense_result = effective_target_observability_from_bundle_dense(dense_bundle, variable_name, tau_target_std_seconds=0.2)
        sparse_result = effective_target_observability_from_bundle_sparse(sparse_bundle, variable_name, tau_target_std_seconds=0.2)
        assert np.allclose(dense_result.singular_values_O_X, sparse_result.singular_values_O_X, atol=1e-5, rtol=1e-4)
        assert dense_result.practical_rank_diagnostics.practical_rank == sparse_result.practical_rank_diagnostics.practical_rank
        assert np.allclose(
            dense_result.physical_information_diagnostics.information_matrix,
            sparse_result.physical_information_diagnostics.information_matrix,
            atol=1e-5,
            rtol=1e-4,
        )


def test_visualization_series_accepts_accelerometer_options() -> None:
    dataset = _dataset()
    provider = estimate_poses_dummy(dataset)
    series = build_observability_visualization_series(
        dataset,
        provider,
        window_duration=1.5,
        window_step=0.75,
        accelerometer_options=AccelerometerOptions(mode="complex", factor_rate_hz=5.0, measurement_std_m_s2=0.05),
        practical_rank_policy=PracticalRankPolicy(),
        use_sparse=False,
    )
    valid = [snapshot for snapshot in series.snapshots if snapshot.is_valid]
    assert valid
    dashboard_text = _rank_dashboard_text(valid[-1], tuple(series.ranks))
    assert "C_X_I accel:" in dashboard_text
    assert "O_T_B_I:" in dashboard_text and "cond=" in dashboard_text
    assert np.nanmax(series.C_X_I_accel_rank) <= 5
    assert np.max(series.accelerometer_factor_count) > 0
