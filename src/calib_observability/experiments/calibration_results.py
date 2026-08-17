"""Result extraction helpers for rolling calibration experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import mrob
import numpy as np

from src.factor_graph_calibration import CalibrationWindowResult
from src.transform import se3_to_se3_vectors


@dataclass(frozen=True)
class CalibrationResultSeries:
    """Window-wise calibration estimates extracted from rolling graph results.

    Args:
        label: Human-readable run label.
        window_indices: Rolling window indices, shape ``(N,)``.
        window_midpoints_s: Window midpoint timestamps in seconds, shape ``(N,)``.
        T_B_I: Body-from-IMU estimates, shape ``(N, 4, 4)``.
        T_B_L: Body-from-LiDAR estimates, shape ``(N, 4, 4)``.
        T_B_I_vectors: ``mrob.SE3.Ln`` tangent vectors for ``T_B_I``, shape
            ``(N, 6)``. The project convention is
            ``[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]``.
        T_B_L_vectors: ``mrob.SE3.Ln`` tangent vectors for ``T_B_L``, shape
            ``(N, 6)``.
        bias_g_radps: Gyroscope bias estimates in rad/s, shape ``(N, 3)``.
        tau_I_s: IMU clock-offset estimates in seconds, shape ``(N,)``.
        tau_L_s: LiDAR clock-offset estimates in seconds, shape ``(N,)``.
        chi2_before: Window objective before optimization, shape ``(N,)``.
        chi2_after: Window objective after optimization, shape ``(N,)``.
        factor_counts: Per-window factor family counts.
    """

    label: str
    window_indices: np.ndarray
    window_midpoints_s: np.ndarray
    T_B_I: np.ndarray
    T_B_L: np.ndarray
    T_B_I_vectors: np.ndarray
    T_B_L_vectors: np.ndarray
    bias_g_radps: np.ndarray
    tau_I_s: np.ndarray
    tau_L_s: np.ndarray
    chi2_before: np.ndarray
    chi2_after: np.ndarray
    factor_counts: list[dict[str, int]]


def series_from_results(results: Sequence[CalibrationWindowResult], *, label: str = "run") -> CalibrationResultSeries:
    """Convert rolling window result objects into plottable arrays.

    Args:
        results: Sequence returned by ``FactorGraphCalibration.generate_filter_iterative``.
        label: Human-readable run label.

    Returns:
        ``CalibrationResultSeries`` with one row per solved window.
    """
    n_results = len(results)
    window_indices = np.asarray([result.window_index for result in results], dtype=int)
    window_midpoints = np.asarray(
        [0.5 * (result.window_start + result.window_end) for result in results],
        dtype=float,
    )

    T_B_I = _stack_optional_se3(result.T_B_I for result in results)
    T_B_L = _stack_optional_se3(result.T_B_L for result in results)
    T_B_I_vectors = _se3_vectors_with_nan(T_B_I)
    T_B_L_vectors = _se3_vectors_with_nan(T_B_L)

    bias_g = np.vstack(
        [
            np.full(3, np.nan) if result.bias_g is None else np.asarray(result.bias_g, dtype=float).reshape(3)
            for result in results
        ]
    ) if n_results else np.empty((0, 3))
    tau_I = np.asarray([np.nan if result.tau_I is None else float(result.tau_I) for result in results], dtype=float)
    tau_L = np.asarray([np.nan if result.tau_L is None else float(result.tau_L) for result in results], dtype=float)

    return CalibrationResultSeries(
        label=label,
        window_indices=window_indices,
        window_midpoints_s=window_midpoints,
        T_B_I=T_B_I,
        T_B_L=T_B_L,
        T_B_I_vectors=T_B_I_vectors,
        T_B_L_vectors=T_B_L_vectors,
        bias_g_radps=bias_g,
        tau_I_s=tau_I,
        tau_L_s=tau_L,
        chi2_before=np.asarray([result.chi2_before for result in results], dtype=float),
        chi2_after=np.asarray([result.chi2_after for result in results], dtype=float),
        factor_counts=[dict(result.factor_counts) for result in results],
    )


def calibration_delta_from_baseline(
    baseline: CalibrationResultSeries,
    injected: CalibrationResultSeries,
) -> dict[str, np.ndarray]:
    """Compute injected-minus-baseline calibration series on shared windows.

    Args:
        baseline: Result series from the unmodified real measurements.
        injected: Result series from the injected measurement copy.

    Returns:
        Dictionary with baseline-relative deltas for timestamps, ``T_B_I``,
        ``T_B_L``, gyroscope bias, and time offsets.
    """
    base_indices = {int(index): row for row, index in enumerate(baseline.window_indices)}
    injected_rows = []
    baseline_rows = []
    for injected_row, index in enumerate(injected.window_indices):
        index = int(index)
        if index in base_indices:
            injected_rows.append(injected_row)
            baseline_rows.append(base_indices[index])

    if not injected_rows:
        raise ValueError("baseline and injected series do not share window indices")

    b = np.asarray(baseline_rows, dtype=int)
    i = np.asarray(injected_rows, dtype=int)

    return {
        "window_indices": injected.window_indices[i].copy(),
        "window_midpoints_s": injected.window_midpoints_s[i].copy(),
        "T_B_I_delta": _relative_se3_series(baseline.T_B_I[b], injected.T_B_I[i]),
        "T_B_L_delta": _relative_se3_series(baseline.T_B_L[b], injected.T_B_L[i]),
        "T_B_I_delta_vectors": _relative_se3_vectors(baseline.T_B_I[b], injected.T_B_I[i]),
        "T_B_L_delta_vectors": _relative_se3_vectors(baseline.T_B_L[b], injected.T_B_L[i]),
        "bias_g_delta_radps": injected.bias_g_radps[i] - baseline.bias_g_radps[b],
        "tau_I_delta_s": injected.tau_I_s[i] - baseline.tau_I_s[b],
        "tau_L_delta_s": injected.tau_L_s[i] - baseline.tau_L_s[b],
        "chi2_after_delta": injected.chi2_after[i] - baseline.chi2_after[b],
    }


def result_summary_rows(series: CalibrationResultSeries) -> list[dict[str, float]]:
    """Build compact dictionary rows for DataFrame display.

    Args:
        series: Window-wise calibration result series.

    Returns:
        List of scalar dictionaries suitable for ``pandas.DataFrame``.
    """
    rows = []
    for row, window_index in enumerate(series.window_indices):
        rows.append(
            {
                "window_index": int(window_index),
                "t_mid_s": float(series.window_midpoints_s[row]),
                "bias_g_x": float(series.bias_g_radps[row, 0]),
                "bias_g_y": float(series.bias_g_radps[row, 1]),
                "bias_g_z": float(series.bias_g_radps[row, 2]),
                "tau_I_s": float(series.tau_I_s[row]),
                "tau_L_s": float(series.tau_L_s[row]),
                "chi2_before": float(series.chi2_before[row]),
                "chi2_after": float(series.chi2_after[row]),
            }
        )
    return rows


def _stack_optional_se3(values: Sequence[np.ndarray | None]) -> np.ndarray:
    """Stack optional SE(3) matrices and preserve missing values as NaNs."""
    matrices = []
    for value in values:
        if value is None:
            matrices.append(np.full((4, 4), np.nan))
        else:
            matrices.append(np.asarray(value, dtype=float).reshape(4, 4))
    return np.stack(matrices, axis=0) if matrices else np.empty((0, 4, 4))


def _se3_vectors_with_nan(se3s: np.ndarray) -> np.ndarray:
    """Map valid SE(3) rows to tangent vectors while preserving NaN rows."""
    vectors = np.full((len(se3s), 6), np.nan)
    for row, se3 in enumerate(se3s):
        if np.all(np.isfinite(se3)):
            vectors[row] = se3_to_se3_vectors(se3)[0]
    return vectors


def _relative_se3_series(baseline: np.ndarray, injected: np.ndarray) -> np.ndarray:
    """Compute baseline-inverse times injected transforms row by row."""
    deltas = np.full_like(injected, np.nan)
    for row, (base, inj) in enumerate(zip(baseline, injected)):
        if np.all(np.isfinite(base)) and np.all(np.isfinite(inj)):
            deltas[row] = np.asarray(mrob.SE3(base).inv().mul(mrob.SE3(inj)).T(), dtype=float)
    return deltas


def _relative_se3_vectors(baseline: np.ndarray, injected: np.ndarray) -> np.ndarray:
    """Compute tangent vectors of baseline-relative SE(3) deltas."""
    return _se3_vectors_with_nan(_relative_se3_series(baseline, injected))
