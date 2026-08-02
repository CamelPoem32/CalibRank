from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import sparse

from src.calib_observability.linalg import numerical_rank_dense
from src.calib_observability.observability import (
    effective_observability_dense,
    effective_observability_dense_qr,
    effective_observability_sparse_lsmr,
    reduced_information_dense,
    schur_complement_dense,
)


def test_dense_sparse_projection_invariants_match() -> None:
    rng = np.random.default_rng(5)
    J_T = rng.normal(size=(40, 8))
    J_C = rng.normal(size=(40, 5))
    O = effective_observability_dense(J_T, J_C)
    O_qr = effective_observability_dense_qr(J_T, J_C)
    sparse_result = effective_observability_sparse_lsmr(sparse.csr_matrix(J_T), sparse.csr_matrix(J_C))
    S_dense = reduced_information_dense(O)
    assert np.allclose(S_dense, sparse_result.S_C, atol=1e-8)
    assert np.allclose(S_dense, O_qr.T @ O_qr, atol=1e-8)
    assert numerical_rank_dense(O) == numerical_rank_dense(sparse_result.O_C.toarray())
    assert np.allclose(np.linalg.svd(O, compute_uv=False), np.linalg.svd(sparse_result.O_C.toarray(), compute_uv=False), atol=1e-8)


def test_schur_equals_projected_information_full_rank() -> None:
    rng = np.random.default_rng(6)
    J_T = rng.normal(size=(50, 7))
    J_C = rng.normal(size=(50, 4))
    S_projected = reduced_information_dense(effective_observability_dense(J_T, J_C))
    S_schur = schur_complement_dense(J_T, J_C)
    assert np.allclose(S_projected, S_schur, atol=1e-9)


def test_rank_deficient_trajectory_projection_and_schur_error() -> None:
    rng = np.random.default_rng(7)
    J_T = rng.normal(size=(20, 5))
    J_T[:, 4] = J_T[:, 0] + J_T[:, 1]
    J_C = rng.normal(size=(20, 3))
    O = effective_observability_dense(J_T, J_C)
    assert O.shape == (20, 3)
    import pytest

    with pytest.raises(ValueError):
        schur_complement_dense(J_T, J_C)
