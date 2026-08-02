from __future__ import annotations

import numpy as np

from src.calib_observability.backend import estimate_poses_dummy
from src.calib_observability.diagnostics import (
    PracticalRankPolicy,
    physical_information_diagnostics,
    practical_rank_diagnostics,
    scalar_time_offset_diagnostics,
    validate_stored_rank_against_matrix,
)
from src.calib_observability.lie_se3 import se3_inverse
from src.calib_observability.simulation import (
    PlanarRoverConfig,
    reframe_dataset_to_fixed_extrinsic,
    simulate_planar_rover,
)
from src.calib_observability.visualization.quasi_realtime_rover import build_observability_visualization_series


def _small_fixed_dataset():
    config = PlanarRoverConfig(
        rectangle_width=1.0,
        rectangle_height=1.0,
        straight_speed=1.0,
        turn_duration=0.2,
        imu_rate_hz=20.0,
        lidar_rate_hz=5.0,
        random_seed=99,
    )
    original = simulate_planar_rover(config, mode="one_rectangle")
    fixed = reframe_dataset_to_fixed_extrinsic(original, "T_B_L")
    return original, fixed


def test_fixed_lidar_reframing_preserves_world_sensor_poses() -> None:
    original, fixed = _small_fixed_dataset()
    for time_seconds in np.linspace(original.start_time + 0.2, original.end_time - 0.2, 4):
        old_body_pose = original.trajectory.pose_at(float(time_seconds))
        new_body_pose = fixed.trajectory.pose_at(float(time_seconds))
        assert np.allclose(new_body_pose @ fixed.T_B_L_true, old_body_pose @ original.T_B_L_true, atol=1e-10)
        assert np.allclose(new_body_pose @ fixed.T_B_I_true, old_body_pose @ original.T_B_I_true, atol=1e-10)
    assert np.allclose(fixed.T_B_L_true, np.eye(4))
    assert np.allclose(fixed.T_B_I_true, se3_inverse(original.T_B_L_true) @ original.T_B_I_true)


def test_fixed_lidar_window_layout_has_no_lidar_extrinsic_columns_or_blocks() -> None:
    _, dataset = _small_fixed_dataset()
    provider = estimate_poses_dummy(dataset)
    bundle, body_motions, counts = dataset.window_jacobians(
        1.0,
        2.4,
        provider,
        use_sparse=False,
        fixed_extrinsic="T_B_L",
    )
    assert counts["lidar"] > 0 and counts["imu"] > 0
    assert bundle.J_C.shape[1] == 11
    assert bundle.metadata["calibration_dimension"] == 11
    assert "T_B_L" not in bundle.calibration_column_slices
    assert list(bundle.calibration_column_slices) == ["T_B_I", "b_g", "tau_I", "tau_L"]
    assert all("T_B_L" not in name for name in bundle.calibration_column_slices)
    assert body_motions


def test_canonical_practical_rank_policy_rejects_columns_and_tiny_matrices() -> None:
    policy = PracticalRankPolicy(
        column_absolute_threshold=1e-6,
        column_relative_threshold=1e-5,
        matrix_absolute_threshold=1e-5,
        singular_absolute_threshold=1e-5,
        singular_relative_threshold=1e-5,
    )
    tiny = np.diag([1e-12, 8e-13, 6e-13])
    tiny_rank = practical_rank_diagnostics(tiny, policy=policy)
    assert tiny_rank.practical_rank == 0
    assert tiny_rank.matrix_passed_absolute_gate is False
    assert np.isnan(tiny_rank.practical_condition_number)

    meaningful = np.diag([2.0, 1e-4, 1e-7])
    meaningful_rank = practical_rank_diagnostics(meaningful, policy=policy)
    assert meaningful_rank.matrix_passed_absolute_gate is True
    assert meaningful_rank.practical_rank == 2
    assert meaningful_rank.zero_column_mask.tolist() == [False, False, True]
    validate_stored_rank_against_matrix(meaningful, meaningful_rank, policy)


def test_physical_information_and_scalar_tau_use_unnormalized_matrix() -> None:
    policy = PracticalRankPolicy()
    O = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=float)
    rank = practical_rank_diagnostics(O, policy=policy)
    info = physical_information_diagnostics(O, practical_rank_result=rank, policy=policy)
    assert np.allclose(info.information_matrix, O.T @ O)
    assert np.isclose(info.trace_information, 25.0)
    assert np.allclose(info.information_eigenvalues, np.array([16.0, 9.0]))

    tau = np.array([[3.0], [4.0]], dtype=float)
    tau_info = scalar_time_offset_diagnostics(tau, policy=policy, target_std_seconds=0.2)
    assert np.isclose(tau_info.sensitivity_tau, 5.0)
    assert np.isclose(tau_info.information_tau, 25.0)
    assert np.isclose(tau_info.local_std_bound_tau_seconds, 0.2)
    assert tau_info.meets_target is True


def test_visualization_series_uses_canonical_rank_and_keeps_c_x_l_diagnostic() -> None:
    _, dataset = _small_fixed_dataset()
    provider = estimate_poses_dummy(dataset)
    policy = PracticalRankPolicy()
    series = build_observability_visualization_series(
        dataset,
        provider,
        window_duration=1.4,
        window_step=0.7,
        fixed_extrinsic="T_B_L",
        practical_rank_policy=policy,
        use_sparse=False,
    )
    valid = next(snapshot for snapshot in series.snapshots if snapshot.is_valid)
    assert "T_B_L" not in valid.bundle.calibration_column_slices
    assert "T_B_L" not in valid.target_results
    assert np.isfinite(series.C_X_L_rank).any()
    for variable_name, result in valid.target_results.items():
        recomputed = validate_stored_rank_against_matrix(result.O_X_physical, result.practical_rank_diagnostics, policy)
        snapshot_index = series.snapshots.index(valid)
        assert series.ranks[variable_name][snapshot_index] == recomputed.practical_rank
