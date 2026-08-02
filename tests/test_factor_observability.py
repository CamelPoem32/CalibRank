from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import sparse

from src.calib_observability.assembly import JacobianBlock, VariableLayout, assemble_jacobian_dense, make_residual_blocks
from src.calib_observability.factor_observability import (
    build_imu_gyro_motion_sensitivity,
    build_lidar_motion_only_matrix_dense,
    covariance_from_physical_information,
    effective_rank_threshold_sweep,
    effective_target_observability_dense,
    effective_target_observability_sparse_lsmr,
    extract_nuisance_columns,
    extract_variable_columns,
    normalize_jacobian_columns_dense,
    normalize_jacobian_columns_sparse,
    rank_diagnostics_dense,
)
from src.calib_observability.lie_se3 import se3_exp


def _planar_motion(yaw: float, x: float, y: float) -> np.ndarray:
    return se3_exp(np.array([0.0, 0.0, yaw, x, y, 0.0], dtype=float))


def _full_rank_motion(yaw: float, pitch: float, roll: float, x: float, y: float, z: float) -> np.ndarray:
    return se3_exp(np.array([roll, pitch, yaw, x, y, z], dtype=float))


def _imu_only_bundle() -> object:
    layout = VariableLayout.from_specs([("T_B_I", 6, "calibration")])
    residual_blocks = make_residual_blocks([("imu_0", 3, np.eye(3), "measurement")])
    sensitivity = np.array(
        [
            [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.1, 1.0, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return assemble_jacobian_dense(
        layout,
        residual_blocks,
        [JacobianBlock("imu_0", "T_B_I", sensitivity)],
        {"imu_0": np.zeros(3)},
        metadata={"calibration_labels": ["T_B_I_roll", "T_B_I_pitch", "T_B_I_yaw", "T_B_I_x", "T_B_I_y", "T_B_I_z"]},
    )


def test_planar_lidar_motion_only_has_generic_rank_five_and_z_nullspace() -> None:
    body_motions = [
        _planar_motion(0.6, 1.0, 0.0),
        _planar_motion(-0.4, 0.0, 1.5),
        _planar_motion(0.9, 1.0, 1.0),
    ]
    result = build_lidar_motion_only_matrix_dense(body_motions, relative_rank_threshold=1e-8)
    assert result.C_X_L_raw.shape == (18, 6)
    assert result.machine_rank == 5
    assert result.effective_rank == 5
    assert np.allclose(np.asarray(result.C_X_L_raw)[:, 5], 0.0, atol=1e-12)
    z_translation = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    projection = result.null_space_basis @ (result.null_space_basis.T @ z_translation)
    assert np.allclose(projection, z_translation, atol=1e-8)


def test_gyro_only_imu_translation_columns_are_zero() -> None:
    result = build_imu_gyro_motion_sensitivity(_imu_only_bundle())
    assert np.allclose(result.translation_column_norms, 0.0)
    assert result.structural_zero_column_report["translation_columns_are_zero"] is True
    assert result.rotation_only_sensitivity_matrix.shape == (3, 3)
    assert result.full_machine_rank == result.rotation_machine_rank


def test_full_rank_lidar_motion_only_does_not_imply_joint_target_rank() -> None:
    rich_motions = [
        _full_rank_motion(0.4, 0.2, 0.1, 1.0, 0.0, 0.5),
        _full_rank_motion(-0.3, 0.5, -0.2, 0.0, 1.2, 0.3),
        _full_rank_motion(0.8, -0.4, 0.3, 0.7, -0.5, 1.0),
    ]
    motion_result = build_lidar_motion_only_matrix_dense(rich_motions, relative_rank_threshold=1e-8)
    assert motion_result.machine_rank == 6

    target_block = np.eye(6)
    J = np.hstack([target_block, target_block])
    joint_result = effective_target_observability_dense(
        J,
        np.arange(6),
        np.arange(6, 12),
        normalization="column",
        relative_rank_threshold=1e-8,
    )
    assert joint_result.machine_rank_O_X == 0
    assert joint_result.effective_rank_O_X == 0


def test_dense_and_sparse_per_variable_results_agree_by_information_matrix() -> None:
    rng = np.random.default_rng(12)
    J = rng.normal(size=(30, 8))
    target_columns = np.array([0, 1, 2])
    nuisance_columns = np.array([3, 4, 5, 6, 7])
    dense_result = effective_target_observability_dense(
        J,
        target_columns,
        nuisance_columns,
        normalization="physical_then_column",
        relative_rank_threshold=1e-7,
    )
    sparse_result = effective_target_observability_sparse_lsmr(
        sparse.csr_matrix(J),
        target_columns,
        nuisance_columns,
        normalization="physical_then_column",
        relative_rank_threshold=1e-7,
    )
    dense_O = np.asarray(dense_result.O_X_normalized)
    sparse_O = sparse_result.O_X_normalized.toarray()
    assert np.allclose(dense_O.T @ dense_O, sparse_O.T @ sparse_O, atol=1e-8)
    assert dense_result.machine_rank_O_X == sparse_result.machine_rank_O_X
    assert dense_result.effective_rank_O_X == sparse_result.effective_rank_O_X
    assert np.allclose(dense_result.singular_values_O_X, sparse_result.singular_values_O_X, atol=1e-8)


def test_column_normalization_preserves_rank_and_keeps_zero_columns_zero() -> None:
    J = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
    before = np.linalg.matrix_rank(J)
    normalized = normalize_jacobian_columns_dense(J)
    after = np.linalg.matrix_rank(normalized.normalized_jacobian)
    assert before == after
    assert normalized.zero_column_mask.tolist() == [False, False, True]
    assert np.allclose(normalized.normalized_jacobian[:, 2], 0.0)

    sparse_normalized = normalize_jacobian_columns_sparse(sparse.csr_matrix(J))
    assert sparse_normalized.zero_column_mask.tolist() == [False, False, True]
    assert np.allclose(sparse_normalized.normalized_jacobian.toarray()[:, 2], 0.0)


def test_effective_rank_threshold_sweep_and_separate_machine_rank() -> None:
    singular_values = np.array([1.0, 1e-4, 1e-8])
    sweep = effective_rank_threshold_sweep(singular_values, [1e-3, 1e-5, 1e-9])
    assert sweep[1e-3] == 1
    assert sweep[1e-5] == 2
    assert sweep[1e-9] == 3

    matrix = np.diag([1.0, 1e-8])
    diagnostics = rank_diagnostics_dense(matrix, relative_rank_threshold=1e-5)
    assert diagnostics.machine_rank == 2
    assert diagnostics.effective_rank == 1
    assert diagnostics.machine_rank_threshold != diagnostics.relative_rank_threshold


def test_column_unit_conversion_changes_raw_spectrum_not_column_normalized_effective_rank() -> None:
    J = np.array([[1.0, 0.2], [0.0, 1.0], [0.1, 0.0]], dtype=float)
    converted = J.copy()
    converted[:, 1] *= 1e6
    assert not np.allclose(np.linalg.svd(J, compute_uv=False), np.linalg.svd(converted, compute_uv=False))
    rank_original = rank_diagnostics_dense(normalize_jacobian_columns_dense(J).normalized_jacobian).effective_rank
    rank_converted = rank_diagnostics_dense(normalize_jacobian_columns_dense(converted).normalized_jacobian).effective_rank
    assert rank_original == rank_converted


def test_covariance_uses_physical_matrix_not_column_normalized_matrix() -> None:
    physical = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)
    covariance, ordinary_condition, note = covariance_from_physical_information(physical)
    assert covariance is not None
    assert np.allclose(covariance, np.diag([0.25, 1.0]))
    assert np.isfinite(ordinary_condition)
    normalized_covariance, _, _ = covariance_from_physical_information(normalize_jacobian_columns_dense(physical).normalized_jacobian)
    assert normalized_covariance is not None
    assert not np.allclose(covariance, normalized_covariance)
    assert "physical" in note or "covariance" in note


def test_effective_condition_and_ordinary_condition_for_rank_deficient_matrix() -> None:
    J = np.diag([1.0, 1e-3, 0.0])
    result = effective_target_observability_dense(
        J,
        np.array([0, 1, 2]),
        np.array([], dtype=int),
        normalization="none",
        relative_rank_threshold=1e-5,
    )
    assert result.effective_rank_O_X == 2
    assert np.isclose(result.effective_condition_number, 1e3)
    assert np.isinf(result.ordinary_condition_number)
    assert result.covariance_from_physical_information is None


def test_calibration_block_extraction_and_nuisance_cover_all_columns() -> None:
    layout = VariableLayout.from_specs([("T_B_I", 6, "calibration"), ("b_g", 3, "calibration"), ("tau_I", 1, "calibration")])
    residual_blocks = make_residual_blocks([("imu_0", 3, np.eye(3), "measurement")])
    bundle = assemble_jacobian_dense(
        layout,
        residual_blocks,
        [JacobianBlock("imu_0", "T_B_I", np.ones((3, 6))), JacobianBlock("imu_0", "b_g", np.ones((3, 3)))],
        {"imu_0": np.zeros(3)},
        metadata={"calibration_labels": ["r", "p", "y", "x", "yy", "z", "bgx", "bgy", "bgz", "tau"]},
    )
    target = extract_variable_columns(bundle, "T_B_I")
    nuisance = extract_nuisance_columns(bundle, "T_B_I")
    assert target.column_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert nuisance.column_indices.tolist() == [6, 7, 8, 9]
    assert np.intersect1d(target.column_indices, nuisance.column_indices).size == 0
    assert np.union1d(target.column_indices, nuisance.column_indices).size == bundle.J_C.shape[1]
