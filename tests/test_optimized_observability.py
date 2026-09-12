"""Regression coverage for optional observability acceleration."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calib_observability import factor_observability as factor
from calib_observability.diagnostics import validate_stored_rank_against_matrix
from calib_observability.observability import effective_observability_dense
from calib_observability.types import PracticalRankPolicy
from kaist_dataset import observability as kaist
from kaist_dataset.run_kaist_observability import build_argument_parser


def compare_targets(expected, actual):
    np.testing.assert_allclose(actual.O_X_physical, expected.O_X_physical, atol=2e-10, rtol=2e-8)
    assert actual.effective_rank_O_X == expected.effective_rank_O_X
    assert actual.machine_rank_O_X == expected.machine_rank_O_X
    for field in ("singular_values_O_X", "zero_target_column_mask", "retained_singular_value_mask"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=2e-10, rtol=2e-8)
    # Singular vectors are sign/basis ambiguous; compare their projectors.
    np.testing.assert_allclose(
        actual.null_space_O_X @ actual.null_space_O_X.T,
        expected.null_space_O_X @ expected.null_space_O_X.T, atol=2e-8,
    )
    a, b = actual.local_accuracy_diagnostics, expected.local_accuracy_diagnostics
    assert a.covariance_kind == b.covariance_kind
    for field in ("observable_projector", "null_projector", "coordinate_std_bounds",
                  "coordinate_is_fully_bounded", "retained_mode_std_bounds",
                  "physical_information_matrix"):
        np.testing.assert_allclose(getattr(a, field), getattr(b, field), atol=2e-8, rtol=2e-7)
    for field in ("full_covariance", "observable_subspace_pseudocovariance"):
        av, bv = getattr(a, field), getattr(b, field)
        if bv is None:
            assert av is None
        else:
            np.testing.assert_allclose(av, bv, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("shape", [(40, 6), (6, 6), (3, 6), (1, 1)])
@pytest.mark.parametrize("kind", ["full", "dependent", "zero", "filtered"])
def test_target_diagnostics_reuse_preserves_results(shape, kind):
    rng = np.random.default_rng(413)
    matrix = rng.normal(size=shape)
    if kind == "dependent" and shape[1] > 1:
        matrix[:, -1] = matrix[:, 0] + matrix[:, 1]
    elif kind == "zero":
        matrix[:] = 0
    elif kind == "filtered":
        matrix[:, -1] *= 1e-14
    saved = matrix.copy()
    args = (matrix, np.arange(shape[1]), [])
    baseline = factor.effective_target_observability_dense(*args)
    optimized = factor.effective_target_observability_dense(*args, optimize=True)
    compare_targets(baseline, optimized)
    validate_stored_rank_against_matrix(optimized.O_X_physical, optimized.practical_rank_diagnostics, PracticalRankPolicy())
    np.testing.assert_array_equal(matrix, saved)


@pytest.mark.parametrize("rows,nuisance_columns", [(40, 8), (8, 12), (40, 0), (0, 3)])
@pytest.mark.parametrize("dependent", [False, True])
def test_projection_handles_rank_deficient_wide_empty_nuisance(rows, nuisance_columns, dependent):
    rng = np.random.default_rng(42)
    nuisance = rng.normal(size=(rows, nuisance_columns))
    if dependent and nuisance_columns:
        nuisance[:, -1] = nuisance[:, 0] * 2
    target = rng.normal(size=(rows, 3))
    expected = effective_observability_dense(nuisance, target)
    actual = effective_observability_dense(nuisance, target, optimize=True)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(nuisance.T @ actual, 0, atol=1e-11)


def test_projection_preserves_pinv_cutoff_and_small_target_residual():
    nuisance = np.diag([1.0, 2e-15, 0.5e-15, 0.0])
    target = np.eye(4)
    expected = effective_observability_dense(nuisance, target)
    actual = effective_observability_dense(nuisance, target, optimize=True)
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_dense_target_with_nuisance_scaling_and_scalar_timing():
    rng = np.random.default_rng(98)
    matrix = rng.normal(size=(50, 8))
    scaling = np.diag(np.linspace(0.5, 2, 8))
    for target in ([0], [0, 1, 2]):
        kwargs = dict(parameter_scaling=scaling, tau_target_std_seconds=0.2, lidar_rate_hz=5)
        nuisance = np.arange(len(target), 8)
        a = factor.effective_target_observability_dense(matrix, target, nuisance, **kwargs)
        b = factor.effective_target_observability_dense(matrix, target, nuisance, optimize=True, **kwargs)
        compare_targets(a, b)


def test_sparse_optimization_retains_lsmr_projection():
    rng = np.random.default_rng(21)
    matrix = sparse.csr_matrix(rng.normal(size=(30, 8)))
    a = factor.effective_target_observability_sparse_lsmr(matrix, [0, 1], np.arange(2, 8))
    b = factor.effective_target_observability_sparse_lsmr(matrix, [0, 1], np.arange(2, 8), optimize=True)
    # Both branches use the exact same LSMR path.
    np.testing.assert_array_equal(a.O_X_physical.toarray(), b.O_X_physical.toarray())
    np.testing.assert_allclose(a.local_accuracy_diagnostics.coordinate_std_bounds,
                               b.local_accuracy_diagnostics.coordinate_std_bounds)


def test_optimized_target_uses_one_right_svd_and_no_large_projector(monkeypatch):
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(100, 10))
    original_svd, original_eye = np.linalg.svd, np.eye
    calls = []
    def svd(a, *args, **kwargs):
        calls.append((a.shape, kwargs.get("full_matrices", True)))
        return original_svd(a, *args, **kwargs)
    def eye(n, *args, **kwargs):
        assert n != 100, "must not construct the residual-space identity/projector"
        return original_eye(n, *args, **kwargs)
    monkeypatch.setattr(np.linalg, "svd", svd)
    monkeypatch.setattr(np, "eye", eye)
    factor.effective_target_observability_dense(matrix, [0, 1], np.arange(2, 10), optimize=True)
    assert calls.count(((100, 2), False)) == 1
    assert not any(full and rows == 100 for (rows, cols), full in calls)


def test_kaist_flag_cli_and_shared_analysis_helper(monkeypatch, tmp_path):
    assert not kaist.KaistObservabilityConfig(tmp_path).optimize
    parser = build_argument_parser()
    assert parser.parse_args(["Urban16", "--optimize"]).optimize
    assert not parser.parse_args(["Urban16", "--no-optimize"]).optimize
    from types import SimpleNamespace
    prepared = SimpleNamespace(dataset=object(), metadata={"effective_rates_hz": {"lidar": 10.0}})
    monkeypatch.setattr(kaist, "estimate_poses_dummy", lambda dataset: "provider")
    captured = []
    def analyze(dataset, provider, **kwargs):
        captured.append(kwargs)
        assert dataset is prepared.dataset and provider == "provider"
        return "series"
    monkeypatch.setattr(kaist, "run_rolling_observability_analysis", analyze)
    config = kaist.KaistObservabilityConfig(tmp_path)
    for optimize in (False, True):
        assert kaist.analyze_observability_inputs(prepared, replace(config, optimize=optimize)) == "series"
        assert captured[-1]["optimize"] is optimize
    assert {k: v for k, v in captured[0].items() if k != "optimize"} == {
        k: v for k, v in captured[1].items() if k != "optimize"}


@pytest.mark.parametrize("use_sparse", [False, True])
def test_optimized_rolling_serial_and_parallel(use_sparse):
    from calib_observability.backend import estimate_poses_dummy
    from calib_observability.simulation import PlanarRoverConfig, simulate_planar_rover
    from calib_observability.visualization.quasi_realtime_rover import (
        QuasiRealtimeConfig, compute_quasi_realtime_snapshots,
    )
    dataset = simulate_planar_rover(PlanarRoverConfig(
        rectangle_width=1, rectangle_height=1, straight_speed=1,
        turn_duration=0.2, total_laps=1, imu_rate_hz=20, lidar_rate_hz=5, random_seed=44,
    ), mode="one_rectangle")
    provider = estimate_poses_dummy(dataset)
    config = QuasiRealtimeConfig(window_length=1.4, frame_step=0.7,
                                 max_analysis_windows=3, use_sparse=use_sparse)
    reference = compute_quasi_realtime_snapshots(dataset, provider, config)
    for workers in (1, 2):
        actual = compute_quasi_realtime_snapshots(
            dataset, provider, replace(config, optimize=True, n_processes=workers))
        assert any(snapshot.is_valid for snapshot in actual)
        for a, b in zip(reference, actual, strict=True):
            assert a.current_time == b.current_time
            assert a.is_valid == b.is_valid
            assert a.target_results.keys() == b.target_results.keys()
            for variable in a.target_results:
                ar, br = a.target_results[variable], b.target_results[variable]
                assert ar.effective_rank_O_X == br.effective_rank_O_X
                np.testing.assert_allclose(ar.local_accuracy_diagnostics.observable_projector,
                                           br.local_accuracy_diagnostics.observable_projector, atol=1e-7)


def test_ill_conditioned_nuisance_preserves_legacy_roundoff():
    # Ill-conditioned retained modes occur in real KAIST windows. The exact
    # orthogonal projector need not agree with the legacy finite-precision pinv.
    rng = np.random.default_rng(73)
    left, _ = np.linalg.qr(rng.normal(size=(35, 6)))
    right, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    nuisance = (left * np.array([1, 0.1, 0.01, 1e-6, 1e-9, 2e-15])) @ right.T
    target = rng.normal(size=(35, 3))
    expected = effective_observability_dense(nuisance, target)
    actual = effective_observability_dense(nuisance, target, optimize=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
