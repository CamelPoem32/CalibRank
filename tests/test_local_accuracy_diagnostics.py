from __future__ import annotations

from pathlib import Path

import numpy as np

from src.calib_observability.backend import estimate_poses_dummy
from src.calib_observability.diagnostics import (
    coordinate_metadata_for_variable,
    local_accuracy_diagnostics,
    practical_rank_diagnostics,
)
from src.calib_observability.factor_observability import (
    effective_target_observability_dense,
    effective_target_observability_from_bundle_dense,
    effective_target_observability_from_bundle_sparse,
)
from src.calib_observability.simulation import PlanarRoverConfig, reframe_dataset_to_fixed_extrinsic, simulate_planar_rover
from src.calib_observability.types import AccelerometerOptions, PracticalRankPolicy
from src.calib_observability.visualization.quasi_realtime_rover import _rank_dashboard_text, build_observability_visualization_series


def test_full_rank_covariance_equals_svd_inverse_information() -> None:
    O = np.array([[2.0, 0.0], [0.0, 4.0], [1.0, 0.5]], dtype=float)
    rank = practical_rank_diagnostics(O, policy=PracticalRankPolicy())
    labels = ("x", "y")
    units = ("native", "native")
    accuracy = local_accuracy_diagnostics(O, variable_name="X", coordinate_labels=labels, coordinate_units=units, practical_rank_result=rank)
    assert accuracy.covariance_kind == "full"
    assert accuracy.full_covariance is not None
    expected = np.linalg.inv(O.T @ O)
    assert np.allclose(accuracy.full_covariance, expected)
    assert np.all(np.isfinite(accuracy.coordinate_std_bounds))


def test_scalar_tau_bound_seconds_frames_and_target_match_norm() -> None:
    O_tau = np.array([[3.0], [4.0]], dtype=float)
    rank = practical_rank_diagnostics(O_tau, policy=PracticalRankPolicy())
    accuracy = local_accuracy_diagnostics(
        O_tau,
        variable_name="tau_I",
        coordinate_labels=("tau_I",),
        coordinate_units=("s",),
        practical_rank_result=rank,
        lidar_rate_hz=5.0,
        target_std_seconds=0.25,
    )
    assert np.isclose(accuracy.scalar_std_bound, 1.0 / np.linalg.norm(O_tau[:, 0]))
    assert np.isclose(accuracy.scalar_std_bound_lidar_frames, 0.2 * 5.0)
    assert np.isclose(accuracy.target_ratio, 0.2 / 0.25)
    assert accuracy.meets_target is True


def test_rank_zero_target_has_infinite_coordinate_bounds_and_identity_null_projector() -> None:
    O = np.zeros((5, 3), dtype=float)
    rank = practical_rank_diagnostics(O, policy=PracticalRankPolicy())
    accuracy = local_accuracy_diagnostics(O, variable_name="b_g", coordinate_labels=("x", "y", "z"), coordinate_units=("rad/s",) * 3, practical_rank_result=rank)
    assert accuracy.covariance_kind == "unobservable"
    assert accuracy.full_covariance is None
    assert accuracy.observable_subspace_pseudocovariance is None
    assert np.all(np.isinf(accuracy.coordinate_std_bounds))
    assert np.allclose(accuracy.observable_projector, np.zeros((3, 3)))
    assert np.allclose(accuracy.null_projector, np.eye(3))


def test_rank_deficient_subspace_bounds_only_retained_coordinates() -> None:
    O = np.diag([2.0, 0.0, 3.0])
    rank = practical_rank_diagnostics(O, policy=PracticalRankPolicy())
    accuracy = local_accuracy_diagnostics(O, variable_name="b_g", coordinate_labels=("x", "y", "z"), coordinate_units=("rad/s",) * 3, practical_rank_result=rank)
    assert accuracy.covariance_kind == "observable_subspace_only"
    assert accuracy.observable_subspace_pseudocovariance is not None
    assert np.isfinite(accuracy.coordinate_std_bounds[0])
    assert np.isinf(accuracy.coordinate_std_bounds[1])
    assert np.isfinite(accuracy.coordinate_std_bounds[2])
    assert np.allclose(accuracy.observable_projector + accuracy.null_projector, np.eye(3))
    assert np.allclose(accuracy.observable_projector, accuracy.observable_projector.T)
    assert np.allclose(accuracy.observable_projector @ accuracy.observable_projector, accuracy.observable_projector)
    assert np.allclose(accuracy.coordinate_observable_fraction + accuracy.coordinate_null_fraction, np.ones(3))
    assert np.isclose(accuracy.worst_retained_mode_std_bound, 1.0 / rank.singular_values[rank.retained_mask][-1])


def _fixed_dataset(mode: str = "one_rectangle"):
    config = PlanarRoverConfig(rectangle_width=2.0, rectangle_height=1.0, straight_speed=1.0, turn_duration=0.5, imu_rate_hz=50.0, lidar_rate_hz=5.0, gyro_noise_std=1e-6, accel_noise_std=1e-6, random_seed=123, mode=mode)
    return reframe_dataset_to_fixed_extrinsic(simulate_planar_rover(config, mode=mode), "T_B_L")


def test_dense_sparse_accuracy_diagnostics_match_for_bundle_target() -> None:
    dataset = _fixed_dataset()
    provider = estimate_poses_dummy(dataset)
    options = AccelerometerOptions(mode="complex", factor_rate_hz=5.0, measurement_std_m_s2=0.05)
    dense_bundle, _, _ = dataset.window_jacobians(1.0, 2.5, provider, use_sparse=False, fixed_extrinsic="T_B_L", accelerometer_options=options)
    sparse_bundle, _, _ = dataset.window_jacobians(1.0, 2.5, provider, use_sparse=True, fixed_extrinsic="T_B_L", accelerometer_options=options)
    dense = effective_target_observability_from_bundle_dense(dense_bundle, "T_B_I", tau_target_std_seconds=0.2, lidar_rate_hz=5.0)
    sparse = effective_target_observability_from_bundle_sparse(sparse_bundle, "T_B_I", tau_target_std_seconds=0.2, lidar_rate_hz=5.0)
    da = dense.local_accuracy_diagnostics
    sa = sparse.local_accuracy_diagnostics
    assert np.allclose(da.coordinate_std_bounds, sa.coordinate_std_bounds, equal_nan=True)
    assert np.allclose(da.coordinate_null_fraction, sa.coordinate_null_fraction)
    assert da.covariance_kind == sa.covariance_kind


def test_rover_accelerometer_coordinate_boundedness_expectations() -> None:
    dataset = _fixed_dataset()
    provider = estimate_poses_dummy(dataset)
    disabled = build_observability_visualization_series(dataset, provider, window_duration=1.5, window_step=0.75, accelerometer_options=AccelerometerOptions(mode="disabled"), lidar_rate_hz=5.0, use_sparse=False)
    simple = build_observability_visualization_series(dataset, provider, window_duration=1.5, window_step=0.75, accelerometer_options=AccelerometerOptions(mode="simple", factor_rate_hz=5.0, measurement_std_m_s2=0.05), lidar_rate_hz=5.0, use_sparse=False)
    complex_series = build_observability_visualization_series(dataset, provider, window_duration=1.5, window_step=0.75, accelerometer_options=AccelerometerOptions(mode="complex", factor_rate_hz=5.0, measurement_std_m_s2=0.05), lidar_rate_hz=5.0, use_sparse=False)
    assert np.all(~disabled.coordinate_bounded_mask["T_B_I"][:, 3:6])
    assert np.all(~simple.coordinate_bounded_mask["T_B_I"][:, 3:6])
    valid_complex = np.where(np.isfinite(complex_series.coordinate_null_fraction["T_B_I"][:, 5]))[0]
    assert valid_complex.size > 0
    assert np.all(~complex_series.coordinate_bounded_mask["T_B_I"][valid_complex, 5])
    assert np.any(complex_series.coordinate_bounded_mask["T_B_I"][valid_complex, 3])
    assert np.any(complex_series.coordinate_bounded_mask["T_B_I"][valid_complex, 4])


def test_bg_bounds_are_finite_and_animation_text_uses_shared_accuracy() -> None:
    dataset = _fixed_dataset()
    provider = estimate_poses_dummy(dataset)
    series = build_observability_visualization_series(dataset, provider, window_duration=1.5, window_step=0.75, lidar_rate_hz=5.0, use_sparse=False)
    valid = next(
        snapshot
        for snapshot in series.snapshots
        if snapshot.is_valid
        and "b_g" in snapshot.local_accuracy_by_variable
        and snapshot.local_accuracy_by_variable["b_g"].covariance_kind == "full"
    )
    index = series.snapshots.index(valid)
    assert np.all(np.isfinite(series.coordinate_std_bounds["b_g"][index]))
    text = _rank_dashboard_text(valid, tuple(series.ranks), show_local_accuracy_summary=True)
    assert "Local CRLB-like bounds:" in text
    assert "O_b_g" in text
    assert "O_tau_I" in text
    assert "target" in text


def test_accuracy_uses_physical_not_column_normalized_matrix() -> None:
    physical = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)
    normalized = physical / np.linalg.norm(physical, axis=0)
    result = effective_target_observability_dense(physical, np.array([0, 1]), np.array([], dtype=int), normalization="none", variable_name="target")
    assert result.local_accuracy_diagnostics.full_covariance is not None
    assert not np.allclose(result.local_accuracy_diagnostics.full_covariance, np.linalg.inv(normalized.T @ normalized))
    labels, units = coordinate_metadata_for_variable("T_B_I", 6)
    assert labels[0] == "rot_x" and units[3] == "m"


def test_notebooks_consume_shared_local_accuracy_series() -> None:
    notebook04 = Path("notebooks/04_quasi_realtime_planar_rover_observability.ipynb").read_text(encoding="utf-8")
    notebook08 = Path("notebooks/08_accelerometer_observability_validation.ipynb").read_text(encoding="utf-8")
    for notebook_text in (notebook04, notebook08):
        assert "coordinate_std_bounds" in notebook_text
        assert "coordinate_bounded_mask" in notebook_text
        assert "coordinate_null_fraction" in notebook_text
        assert "retained_mode_std_bounds" in notebook_text
        assert "CRLB-like" in notebook_text
    assert "quasi_realtime_local_accuracy_by_variable.csv" in notebook04
    assert "accelerometer_local_accuracy_by_variable.csv" in notebook08
