import numpy as np

from src.calib_observability.experiments.calibration_results import (
    calibration_delta_from_baseline,
    series_from_results,
)
from src.factor_graph_calibration import CalibrationWindowResult


def _window(index, tau_I, bias):
    T_B_I = np.eye(4)
    T_B_I[:3, 3] = np.array([0.1 * index, 0.0, 0.0])
    T_B_L = np.eye(4)
    return CalibrationWindowResult(
        window_index=index,
        window_start=float(index),
        window_end=float(index + 1),
        pose_timestamps=np.array([index, index + 1], dtype=float),
        trajectory_poses=np.repeat(np.eye(4)[None, ...], 2, axis=0),
        T_B_I=T_B_I,
        T_B_L=T_B_L,
        bias_g=np.asarray(bias, dtype=float),
        tau_I=float(tau_I),
        tau_L=0.0,
        chi2_before=10.0 + index,
        chi2_after=5.0 + index,
        factor_counts={"gyro": index + 1, "lidar": index + 2},
    )


def test_series_from_results_extracts_window_arrays():
    series = series_from_results([_window(0, 0.01, [1, 2, 3]), _window(1, 0.02, [4, 5, 6])], label="baseline")

    assert series.label == "baseline"
    np.testing.assert_array_equal(series.window_indices, np.array([0, 1]))
    np.testing.assert_allclose(series.window_midpoints_s, np.array([0.5, 1.5]))
    np.testing.assert_allclose(series.bias_g_radps, np.array([[1, 2, 3], [4, 5, 6]]))
    np.testing.assert_allclose(series.tau_I_s, np.array([0.01, 0.02]))
    assert series.factor_counts[1]["lidar"] == 3


def test_calibration_delta_from_baseline_uses_shared_windows():
    baseline = series_from_results([_window(0, 0.01, [0, 0, 0]), _window(1, 0.02, [1, 1, 1])], label="baseline")
    injected = series_from_results([_window(1, 0.05, [3, 4, 5])], label="injected")

    deltas = calibration_delta_from_baseline(baseline, injected)

    np.testing.assert_array_equal(deltas["window_indices"], np.array([1]))
    np.testing.assert_allclose(deltas["tau_I_delta_s"], np.array([0.03]))
    np.testing.assert_allclose(deltas["bias_g_delta_radps"], np.array([[2.0, 3.0, 4.0]]))
