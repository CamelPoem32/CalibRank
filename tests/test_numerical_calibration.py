import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calib_observability.lie_se3 import se3_exp
from numerical_calibration import (
    NumericalCalibrationConfig,
    NumericalCalibrationResult,
    derive_body_angular_velocity_from_lidar,
    estimate_imu_calibration_numerical,
    estimate_rotation_align_vectors,
    estimate_temporal_offset_twistnsync,
    synchronize_angular_velocity_signals,
)
from factor_graph_calibration import FactorGraphCalibration


def _rich_body_angvel(timestamps):
    return np.column_stack([
        np.sin(2.0 * np.pi * 0.31 * timestamps) + 0.2 * np.cos(2.0 * np.pi * 0.17 * timestamps),
        0.7 * np.cos(2.0 * np.pi * 0.43 * timestamps + 0.2),
        0.5 * np.sin(2.0 * np.pi * 0.71 * timestamps + 0.4),
    ])


def _accumulated_lidar_poses(timestamps, omega_body):
    poses = np.repeat(np.eye(4)[None, :, :], timestamps.size, axis=0)
    for index in range(timestamps.size - 1):
        dt = timestamps[index + 1] - timestamps[index]
        poses[index + 1] = poses[index] @ se3_exp(np.r_[omega_body[index] * dt, [0.0, 0.0, 0.0]])
    return poses


def _successful_result(T_B_I, tau_I):
    return NumericalCalibrationResult(
        tau_I=tau_I,
        R_B_I=T_B_I[:3, :3].copy(),
        T_B_I=T_B_I.copy(),
        bias_g_used=np.zeros(3),
        temporal_delay_raw=0.0,
        spatial_rssd=0.0,
        source_timestamps=np.array([0.0, 1.0]),
        reference_timestamps=np.array([0.0, 1.0]),
        synchronized_timestamps=np.array([0.0, 1.0]),
        source_angvels=np.zeros((2, 3)),
        reference_angvels=np.zeros((2, 3)),
        source_angvels_synchronized=np.zeros((2, 3)),
        reference_angvels_synchronized=np.zeros((2, 3)),
        source_angvels_aligned=np.zeros((2, 3)),
        residuals=np.zeros((2, 3)),
        residual_rmse=np.zeros(3),
        residual_vector_rmse=0.0,
        residual_vector_median=0.0,
        excitation_singular_values=np.ones(3),
        excitation_ratios=(1.0, 1.0),
        success=True,
        message="ok",
    )


def _failed_result():
    return NumericalCalibrationResult(
        tau_I=None,
        R_B_I=None,
        T_B_I=None,
        bias_g_used=None,
        temporal_delay_raw=None,
        spatial_rssd=None,
        source_timestamps=np.empty(0),
        reference_timestamps=np.empty(0),
        synchronized_timestamps=np.empty(0),
        source_angvels=np.empty((0, 3)),
        reference_angvels=np.empty((0, 3)),
        source_angvels_synchronized=np.empty((0, 3)),
        reference_angvels_synchronized=np.empty((0, 3)),
        source_angvels_aligned=np.empty((0, 3)),
        residuals=np.empty((0, 3)),
        residual_rmse=np.full(3, np.nan),
        residual_vector_rmse=None,
        residual_vector_median=None,
        excitation_singular_values=np.full(3, np.nan),
        excitation_ratios=(np.nan, np.nan),
        success=False,
        message="mock failure",
    )


def test_scipy_alignment_direction_convention_recovers_R_B_I():
    timestamps = np.linspace(0.0, 20.0, 300)
    source = _rich_body_angvel(timestamps)
    R_B_I = se3_exp([0.2, -0.15, 0.35, 0.0, 0.0, 0.0])[:3, :3]
    reference = (R_B_I @ source.T).T

    estimated, rssd, aligned, residuals, residual_rmse, vector_rmse, vector_median = estimate_rotation_align_vectors(source, reference)

    assert np.allclose(estimated, R_B_I, atol=1e-10)
    assert rssd < 1e-6
    assert np.max(np.abs(aligned - reference)) < 1e-10
    assert np.max(np.abs(residuals)) < 1e-10
    assert np.max(residual_rmse) < 1e-10
    assert vector_rmse < 1e-10
    assert vector_median < 1e-10


def test_identity_spatial_calibration():
    timestamps = np.linspace(0.0, 10.0, 120)
    vectors = _rich_body_angvel(timestamps)
    estimated, *_ = estimate_rotation_align_vectors(vectors, vectors)
    assert np.allclose(estimated, np.eye(3), atol=1e-10)


def test_temporal_shift_sign_matches_factor_graph_tau_convention():
    tau_I_true = 0.18
    body_timestamps = np.arange(0.0, 30.0, 0.02)
    imu_timestamps = body_timestamps + tau_I_true
    body_angvels = _rich_body_angvel(body_timestamps)
    imu_angvels = body_angvels.copy()

    tau_I, raw_delay, diagnostics = estimate_temporal_offset_twistnsync(
        imu_timestamps,
        imu_angvels,
        body_timestamps,
        body_angvels,
        prefer_twistnsync=False,
        max_abs_tau_s=0.5,
        tau_grid_step_s=0.005,
    )

    assert abs(tau_I - tau_I_true) <= 0.006
    assert raw_delay is None or np.isfinite(raw_delay)
    assert diagnostics["temporal_method"] == "norm_correlation_fallback"


def test_no_input_mutation_and_translation_preserved():
    R_B_I = se3_exp([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])[:3, :3]
    tau_I_true = 0.12
    body_timestamps = np.arange(0.0, 12.0, 0.05)
    omega_body = _rich_body_angvel(body_timestamps)
    lidar_poses = _accumulated_lidar_poses(body_timestamps, omega_body)
    imu_timestamps = body_timestamps + tau_I_true
    omega_imu = (R_B_I.T @ omega_body.T).T
    previous_T_B_I = np.eye(4)
    previous_T_B_I[:3, 3] = [1.0, -2.0, 0.5]

    imu_timestamps_copy = imu_timestamps.copy()
    omega_imu_copy = omega_imu.copy()
    lidar_poses_copy = lidar_poses.copy()

    result = estimate_imu_calibration_numerical(
        window_start=1.0,
        window_end=10.0,
        imu_timestamps=imu_timestamps,
        angular_velocity_imu=omega_imu,
        lidar_timestamps=body_timestamps,
        lidar_odometry_poses=lidar_poses,
        T_B_L=np.eye(4),
        bias_g=np.zeros(3),
        T_B_I_previous=previous_T_B_I,
        config=NumericalCalibrationConfig(margin_s=0.5, resample_step_s=0.05, max_abs_tau_s=0.5, tau_grid_step_s=0.005, prefer_twistnsync=False),
    )

    assert result.success, result.message
    assert abs(result.tau_I - tau_I_true) <= 0.03
    assert np.allclose(result.T_B_I[:3, 3], previous_T_B_I[:3, 3])
    assert np.allclose(result.R_B_I, R_B_I, atol=2e-2)
    assert np.array_equal(imu_timestamps, imu_timestamps_copy)
    assert np.array_equal(omega_imu, omega_imu_copy)
    assert np.array_equal(lidar_poses, lidar_poses_copy)


def test_too_few_samples_failure():
    result = estimate_imu_calibration_numerical(
        window_start=0.0,
        window_end=1.0,
        imu_timestamps=[0.0, 1.0],
        angular_velocity_imu=np.zeros((2, 3)),
        lidar_timestamps=[0.0, 1.0],
        lidar_odometry_poses=np.repeat(np.eye(4)[None, :, :], 2, axis=0),
        config=NumericalCalibrationConfig(min_samples=5),
    )
    assert not result.success
    assert "too few" in result.message


def test_invalid_no_overlap_failure():
    body_timestamps = np.arange(10.0, 12.0, 0.1)
    imu_timestamps = np.arange(0.0, 2.0, 0.1)
    omega_body = _rich_body_angvel(body_timestamps)
    lidar_poses = _accumulated_lidar_poses(body_timestamps, omega_body)

    result = estimate_imu_calibration_numerical(
        window_start=10.2,
        window_end=11.0,
        imu_timestamps=imu_timestamps,
        angular_velocity_imu=_rich_body_angvel(imu_timestamps),
        lidar_timestamps=body_timestamps,
        lidar_odometry_poses=lidar_poses,
        config=NumericalCalibrationConfig(margin_s=0.1, min_samples=5, max_abs_tau_s=0.2),
    )
    assert not result.success


def test_derive_body_angular_velocity_from_lidar_uses_local_relative_rotations():
    timestamps = np.linspace(0.0, 1.0, 11)
    omega_body = np.tile([0.0, 0.0, 0.4], (timestamps.size, 1))
    lidar_poses = _accumulated_lidar_poses(timestamps, omega_body)
    midpoints, derived = derive_body_angular_velocity_from_lidar(timestamps, lidar_poses)
    assert midpoints.shape == (timestamps.size - 1,)
    assert np.allclose(derived[:, 2], 0.4, atol=1e-8)


def test_numerical_prior_disabled_preserves_old_initials_in_build_problem():
    graph = FactorGraphCalibration(
        imu_samples_per_factor=None,
        imu_time_offset_margin=1.0,
        include_gyro_factors=True,
        include_accel_factors=False,
        include_lidar_factors=False,
        T_B_I_regularization_information=1.0,
        tau_I_regularization_information=1.0,
        use_numerical_calibration_prior=False,
        use_numerical_calibration_initial=False,
        maxIters=0,
    )
    T_initial = se3_exp([0.1, 0.0, 0.0, 0.2, 0.3, 0.4])
    with patch.object(FactorGraphCalibration, "solve_problem", lambda self: self.states):
        result = graph.generate_filter_window(
            window_index=0,
            window_start=0.0,
            window_end=1.0,
            pose_timestamps=[0.0, 1.0],
            states=[np.eye(4), np.eye(4)],
            imu_timestamps=[-1.0, 0.0, 0.5, 1.0, 2.0],
            angular_velocity_imu=np.zeros((5, 3)),
            T_B_I_initial=T_initial,
            tau_I_initial=0.33,
        )
    assert result.numerical_calibration_success is None
    assert np.allclose(graph.states_init[graph.node_T_B_I], T_initial)
    assert np.isclose(float(np.asarray(graph.states_init[graph.node_tau_I]).reshape(-1)[0]), 0.33)


def test_numerical_prior_failure_falls_back_to_previous_initials():
    graph = FactorGraphCalibration(
        imu_samples_per_factor=None,
        imu_time_offset_margin=1.0,
        include_gyro_factors=True,
        include_accel_factors=False,
        include_lidar_factors=False,
        use_numerical_calibration_prior=True,
        maxIters=0,
    )
    T_initial = se3_exp([0.2, 0.0, 0.0, 0.1, 0.2, 0.3])
    with patch("factor_graph_calibration.estimate_imu_calibration_numerical", return_value=_failed_result()):
        with patch.object(FactorGraphCalibration, "solve_problem", lambda self: self.states):
            result = graph.generate_filter_window(
                window_index=0,
                window_start=0.0,
                window_end=1.0,
                pose_timestamps=[0.0, 1.0],
                states=[np.eye(4), np.eye(4)],
                imu_timestamps=[-1.0, 0.0, 0.5, 1.0, 2.0],
                angular_velocity_imu=np.zeros((5, 3)),
                T_B_I_initial=T_initial,
                tau_I_initial=-0.2,
            )
    assert result.numerical_calibration_success is False
    assert np.allclose(graph.states_init[graph.node_T_B_I], T_initial)
    assert np.isclose(float(np.asarray(graph.states_init[graph.node_tau_I]).reshape(-1)[0]), -0.2)


def test_numerical_prior_success_sets_initial_and_prior_diagnostics():
    graph = FactorGraphCalibration(
        imu_samples_per_factor=None,
        imu_time_offset_margin=1.0,
        include_gyro_factors=True,
        include_accel_factors=False,
        include_lidar_factors=False,
        use_numerical_calibration_prior=True,
        maxIters=0,
    )
    T_prior = se3_exp([-0.1, 0.2, -0.3, 0.7, -0.4, 0.2])
    with patch("factor_graph_calibration.estimate_imu_calibration_numerical", return_value=_successful_result(T_prior, 0.41)):
        with patch.object(FactorGraphCalibration, "solve_problem", lambda self: self.states):
            result = graph.generate_filter_window(
                window_index=0,
                window_start=0.0,
                window_end=1.0,
                pose_timestamps=[0.0, 1.0],
                states=[np.eye(4), np.eye(4)],
                imu_timestamps=[-1.0, 0.0, 0.5, 1.0, 2.0],
                angular_velocity_imu=np.zeros((5, 3)),
                T_B_I_initial=np.eye(4),
                tau_I_initial=0.0,
            )
    assert result.numerical_calibration_success is True
    assert np.allclose(result.numerical_T_B_I_prior, T_prior)
    assert np.isclose(result.numerical_tau_I_prior, 0.41)
    assert np.allclose(graph.states_init[graph.node_T_B_I], T_prior)
    assert np.isclose(float(np.asarray(graph.states_init[graph.node_tau_I]).reshape(-1)[0]), 0.41)
