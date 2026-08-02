from __future__ import annotations

import numpy as np
from scipy import sparse

from src.calib_observability.backend import estimate_poses_dummy
from src.calib_observability.factor_observability import SUPPORTED_CALIBRATION_VARIABLES
from src.calib_observability.simulation import PlanarRoverConfig, simulate_planar_rover
from src.calib_observability.visualization.quasi_realtime_rover import (
    QuasiRealtimeConfig,
    _condition_plot_variables,
    _rank_dashboard_text,
    _update_heatmap,
    compute_quasi_realtime_snapshots,
    dashboard_series,
    matrix_for_display,
    rolling_window_bounds,
)


def test_rolling_window_bounds_prefix_then_fixed_length() -> None:
    early_start, early_end = rolling_window_bounds(2.0, 0.0, 5.0)
    assert early_start == 0.0
    assert early_end == 2.0

    later_start, later_end = rolling_window_bounds(8.0, 0.0, 5.0)
    assert later_start == 3.0
    assert later_end == 8.0
    assert later_end - later_start == 5.0


def test_snapshot_generation_handles_startup_and_later_valid_window() -> None:
    config = PlanarRoverConfig(
        rectangle_width=1.0,
        rectangle_height=1.0,
        straight_speed=1.0,
        turn_duration=0.2,
        total_laps=1,
        imu_rate_hz=20.0,
        lidar_rate_hz=5.0,
        random_seed=44,
    )
    dataset = simulate_planar_rover(config, mode="one_rectangle")
    pose_provider = estimate_poses_dummy(dataset)
    realtime_config = QuasiRealtimeConfig(window_length=1.4, frame_step=0.7, use_sparse=False)

    snapshots = compute_quasi_realtime_snapshots(dataset, pose_provider, realtime_config)

    assert snapshots[0].is_valid is False
    assert "waiting" in snapshots[0].status
    assert any(snapshot.is_valid for snapshot in snapshots)
    valid_snapshot = next(snapshot for snapshot in snapshots if snapshot.is_valid)
    assert valid_snapshot.counts.get("imu", 0) > 0
    assert valid_snapshot.J_C_display.ndim == 2


def test_dashboard_series_has_all_supported_variables_with_nan_for_missing_values() -> None:
    config = PlanarRoverConfig(
        rectangle_width=1.0,
        rectangle_height=1.0,
        straight_speed=1.0,
        turn_duration=0.2,
        total_laps=1,
        imu_rate_hz=20.0,
        lidar_rate_hz=5.0,
        random_seed=45,
    )
    dataset = simulate_planar_rover(config, mode="one_rectangle")
    pose_provider = estimate_poses_dummy(dataset)
    realtime_config = QuasiRealtimeConfig(window_length=1.4, frame_step=0.7, use_sparse=False)

    snapshots = compute_quasi_realtime_snapshots(dataset, pose_provider, realtime_config)
    series = dashboard_series(snapshots, SUPPORTED_CALIBRATION_VARIABLES)

    assert set(series["effective_ranks"]) == set(SUPPORTED_CALIBRATION_VARIABLES)
    assert set(series["condition_numbers"]) == set(SUPPORTED_CALIBRATION_VARIABLES)
    for variable_name in SUPPORTED_CALIBRATION_VARIABLES:
        ranks = series["effective_ranks"][variable_name]
        condition_numbers = series["condition_numbers"][variable_name]
        assert ranks.shape == series["times"].shape
        assert condition_numbers.shape == series["times"].shape
        assert np.isnan(ranks[0])


def test_matrix_for_display_accepts_dense_sparse_and_downsamples_rows() -> None:
    dense_matrix = np.arange(60, dtype=float).reshape(10, 6)
    sparse_matrix = sparse.csr_matrix(dense_matrix)

    dense_display = matrix_for_display(dense_matrix, max_rows=4, max_cols=3)
    sparse_display = matrix_for_display(sparse_matrix, max_rows=4, max_cols=3)

    assert dense_display.shape == (4, 3)
    assert sparse_display.shape == (4, 3)
    assert np.all(np.isfinite(dense_display))
    assert np.allclose(dense_display, sparse_display)

def test_heatmap_update_refreshes_extent_when_frame_shape_changes() -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    image = axis.imshow(np.zeros((1, 1)), aspect="auto")
    matrix = np.arange(12, dtype=float).reshape(4, 3)

    _update_heatmap(image, axis, matrix, "J_C")

    assert image.get_array().shape == (4, 3)
    assert image.get_extent() == [-0.5, 2.5, 3.5, -0.5]
    plt.close(fig)

def test_rank_dashboard_uses_variable_dimension_denominator() -> None:
    config = PlanarRoverConfig(
        rectangle_width=1.0,
        rectangle_height=1.0,
        straight_speed=1.0,
        turn_duration=0.2,
        total_laps=1,
        imu_rate_hz=20.0,
        lidar_rate_hz=5.0,
        random_seed=47,
    )
    dataset = simulate_planar_rover(config, mode="one_rectangle")
    pose_provider = estimate_poses_dummy(dataset)
    realtime_config = QuasiRealtimeConfig(window_length=1.4, frame_step=0.7, use_sparse=False)

    snapshots = compute_quasi_realtime_snapshots(dataset, pose_provider, realtime_config)
    valid_snapshot = next(snapshot for snapshot in snapshots if snapshot.is_valid)
    dashboard_text = _rank_dashboard_text(valid_snapshot, SUPPORTED_CALIBRATION_VARIABLES)

    assert "ranks (practical/max):" in dashboard_text
    assert "O_T_B_I:" in dashboard_text and " / 6," in dashboard_text
    assert "O_T_B_L:" in dashboard_text and "nan / 6," in dashboard_text
    assert "O_b_g:" in dashboard_text and " / 3," in dashboard_text
    assert "O_tau_I: tau std=" in dashboard_text
    assert "O_tau_L: tau std=" in dashboard_text

def test_animation_condition_plot_excludes_scalar_tau_variables() -> None:
    variables = ("T_B_I", "b_g", "tau_I", "tau_L")
    assert _condition_plot_variables(variables) == ("T_B_I", "b_g")
