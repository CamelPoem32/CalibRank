import numpy as np
from scipy import sparse

from src.calib_observability.diagnostics import (
    common_rank_diagnostics,
    physical_information_diagnostics,
    reject_and_normalize_columns_dense,
    reject_and_normalize_columns_sparse,
)


def test_reject_and_normalize_columns_dense_keeps_zero_and_rejects_dust():
    J = np.array(
        [
            [0.0, 1e-13, 3.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    result = reject_and_normalize_columns_dense(
        J,
        absolute_zero_tolerance=1e-12,
        relative_zero_tolerance=1e-10,
    )
    assert result.zero_column_mask.tolist() == [True, True, False]
    assert np.allclose(result.normalized_matrix[:, 0], 0.0)
    assert np.allclose(result.normalized_matrix[:, 1], 0.0)
    assert np.isclose(np.linalg.norm(result.normalized_matrix[:, 2]), 1.0)
    assert np.all(np.isfinite(result.normalized_matrix))


def test_reject_and_normalize_columns_sparse_matches_dense():
    J = np.array(
        [
            [0.0, 1e-13, 3.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    dense = reject_and_normalize_columns_dense(
        J,
        absolute_zero_tolerance=1e-12,
        relative_zero_tolerance=1e-10,
    )
    sparse_result = reject_and_normalize_columns_sparse(
        sparse.csr_matrix(J),
        absolute_zero_tolerance=1e-12,
        relative_zero_tolerance=1e-10,
    )
    assert np.allclose(sparse_result.normalized_matrix.toarray(), dense.normalized_matrix)
    assert np.allclose(sparse_result.original_column_norms, dense.original_column_norms)
    assert sparse_result.zero_column_mask.tolist() == dense.zero_column_mask.tolist()


def test_common_rank_and_information_are_consistent():
    J = np.diag([4.0, 2.0, 1e-8])
    rank = common_rank_diagnostics(J, relative_rank_threshold=1e-5)
    info = physical_information_diagnostics(J, relative_rank_threshold=1e-5)
    assert rank.machine_rank == 2
    assert rank.effective_rank == 2
    assert rank.maximum_possible_rank == 3
    assert np.allclose(info.information_eigenvalues, rank.singular_values**2)
    assert np.isclose(info.total_information, np.sum(rank.singular_values**2))
    assert np.isclose(info.maximum_retained_standard_deviation_bound, 1.0 / 2.0)
