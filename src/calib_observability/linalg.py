'''Linear algebra utilities for dense and sparse observability analysis.'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import svds


##################################################
# Dense singular-value diagnostics
##################################################
@dataclass(frozen=True)
class SingularValueDiagnostics:
    '''Store numerical singular-value diagnostics for a dense matrix.

    Attributes:
        singular_values: Singular values in descending order.
        rank: Number of singular values above ``tolerance``.
        tolerance: Absolute threshold used for numerical rank.
        condition_number: Ratio of the largest to smallest retained singular
            value, or infinity when no singular value is retained.
    '''

    singular_values: NDArray[np.float64]
    rank: int
    tolerance: float
    condition_number: float


def _rank_tolerance(
    s: NDArray[np.float64],
    shape: tuple[int, int],
    tolerance: float | None,
) -> float:
    '''Resolve the absolute singular-value threshold used for numerical rank.

    Args:
        s: Singular values in descending order.
        shape: Shape of the analyzed matrix.
        tolerance: Explicit threshold, or ``None`` for the default
            ``max(shape) * eps * sigma_max`` rule.

    Returns:
        Absolute singular-value threshold.
    '''

    if tolerance is not None:
        return float(tolerance)

    sigma_max = float(s[0]) if s.size else 0.0
    return max(shape) * np.finfo(float).eps * sigma_max


def gram_matrix_dense(J: ArrayLike) -> NDArray[np.float64]:
    '''Compute the dense Gram matrix ``J.T @ J``.

    Args:
        J: Finite matrix with shape ``(m, n)``.

    Returns:
        Gram matrix with shape ``(n, n)``.

    Raises:
        ValueError: If ``J`` is not a finite two-dimensional matrix.
    '''

    A = np.asarray(J, dtype=float)
    if A.ndim != 2 or not np.all(np.isfinite(A)):
        raise ValueError("J must be a finite matrix")

    # A.T: (n, m), A: (m, n) -> gram: (n, n).
    return A.T @ A


def cross_gram_matrix_dense(
    J_A: ArrayLike,
    J_B: ArrayLike,
) -> NDArray[np.float64]:
    '''Compute the dense cross-Gram matrix ``J_A.T @ J_B``.

    Args:
        J_A: Matrix with shape ``(m, n_A)``.
        J_B: Matrix with shape ``(m, n_B)``.

    Returns:
        Cross-Gram matrix with shape ``(n_A, n_B)``.

    Raises:
        ValueError: If either input is not a matrix or their row counts differ.
    '''

    A = np.asarray(J_A, dtype=float)
    B = np.asarray(J_B, dtype=float)

    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
        raise ValueError("J_A and J_B must be matrices with the same row count")

    # A.T: (n_A, m), B: (m, n_B) -> cross: (n_A, n_B).
    return A.T @ B


def pseudoinverse_dense(
    J: ArrayLike,
    rcond: float | None = None,
) -> NDArray[np.float64]:
    '''Compute the Moore-Penrose pseudoinverse of a dense matrix.

    Args:
        J: Matrix with shape ``(m, n)``.
        rcond: Relative singular-value cutoff forwarded to ``numpy.linalg.pinv``.

    Returns:
        Pseudoinverse with shape ``(n, m)``.

    Raises:
        ValueError: If ``J`` is not two-dimensional.
    '''

    A = np.asarray(J, dtype=float)
    if A.ndim != 2:
        raise ValueError("J must be a matrix")

    return np.linalg.pinv(A, rcond=rcond)


def numerical_rank_dense(J: ArrayLike, tolerance: float | None = None) -> int:
    '''Compute dense numerical rank from singular values.

    Args:
        J: Matrix with shape ``(m, n)``.
        tolerance: Absolute singular-value threshold, or ``None`` for the
            package default.

    Returns:
        Number of singular values greater than the selected threshold.
    '''

    A = np.asarray(J, dtype=float)
    singular_values = np.linalg.svd(A, compute_uv=False)
    rank_tolerance = _rank_tolerance(singular_values, A.shape, tolerance)

    return int(np.sum(singular_values > rank_tolerance))


def null_space_dense(
    J: ArrayLike,
    tolerance: float | None = None,
) -> NDArray[np.float64]:
    '''Compute an orthonormal basis for the dense numerical null space.

    Args:
        J: Matrix with shape ``(m, n)``.
        tolerance: Absolute singular-value threshold, or ``None`` for the
            package default.

    Returns:
        Matrix with shape ``(n, n-rank)`` whose columns span the numerical null
        space.
    '''

    A = np.asarray(J, dtype=float)
    _, singular_values, Vt = np.linalg.svd(A, full_matrices=True)
    rank_tolerance = _rank_tolerance(singular_values, A.shape, tolerance)
    rank = int(np.sum(singular_values > rank_tolerance))

    return Vt[rank:].T.copy()


def singular_value_diagnostics_dense(
    J: ArrayLike,
    tolerance: float | None = None,
) -> SingularValueDiagnostics:
    '''Compute singular values, numerical rank, tolerance and condition number.

    Args:
        J: Matrix with shape ``(m, n)``.
        tolerance: Absolute singular-value threshold, or ``None`` for the
            package default.

    Returns:
        Dense singular-value diagnostics.
    '''

    A = np.asarray(J, dtype=float)
    singular_values = np.linalg.svd(A, compute_uv=False)
    rank_tolerance = _rank_tolerance(singular_values, A.shape, tolerance)
    rank = int(np.sum(singular_values > rank_tolerance))

    retained_values = singular_values[singular_values > rank_tolerance]
    condition_number = (
        float(retained_values[0] / retained_values[-1])
        if retained_values.size
        else np.inf
    )

    return SingularValueDiagnostics(
        singular_values=singular_values,
        rank=rank,
        tolerance=rank_tolerance,
        condition_number=condition_number,
    )


def condition_number_dense(
    J: ArrayLike,
    tolerance: float | None = None,
) -> float:
    '''Compute the numerical condition number above a rank tolerance.

    Args:
        J: Matrix with shape ``(m, n)``.
        tolerance: Absolute singular-value threshold, or ``None`` for the
            package default.

    Returns:
        Ratio of the largest to smallest retained singular value, or infinity
        when none are retained.
    '''

    return singular_value_diagnostics_dense(J, tolerance).condition_number


##################################################
# Sparse products and structural diagnostics
##################################################
def gram_matrix_sparse(J: sparse.spmatrix) -> sparse.csr_matrix:
    '''Compute the sparse Gram matrix ``J.T @ J`` without densifying ``J``.

    Args:
        J: Sparse matrix with shape ``(m, n)``.

    Returns:
        CSR Gram matrix with shape ``(n, n)``.

    Raises:
        ValueError: If ``J`` is not sparse.
    '''

    if not sparse.issparse(J):
        raise ValueError("J must be sparse")

    # J.T: (n, m), J: (m, n) -> gram: (n, n).
    return (J.T @ J).tocsr()


def cross_gram_matrix_sparse(
    J_A: sparse.spmatrix,
    J_B: sparse.spmatrix,
) -> sparse.csr_matrix:
    '''Compute the sparse cross-Gram matrix ``J_A.T @ J_B``.

    Args:
        J_A: Sparse matrix with shape ``(m, n_A)``.
        J_B: Sparse matrix with shape ``(m, n_B)``.

    Returns:
        CSR cross-Gram matrix with shape ``(n_A, n_B)``.

    Raises:
        ValueError: If either matrix is dense or their row counts differ.
    '''

    if not sparse.issparse(J_A) or not sparse.issparse(J_B):
        raise ValueError("J_A and J_B must be sparse")
    if J_A.shape[0] != J_B.shape[0]:
        raise ValueError("J_A and J_B must have the same row count")

    # J_A.T: (n_A, m), J_B: (m, n_B) -> cross: (n_A, n_B).
    return (J_A.T @ J_B).tocsr()


def structural_rank_sparse(J: sparse.spmatrix) -> int:
    '''Compute sparse structural rank rather than numerical rank.

    Args:
        J: Sparse matrix.

    Returns:
        Structural rank determined only by the sparsity pattern.

    Raises:
        ValueError: If ``J`` is not sparse.
    '''

    if not sparse.issparse(J):
        raise ValueError("J must be sparse")

    return int(csgraph.structural_rank(J))


##################################################
# Sparse extremal singular values
##################################################
def smallest_singular_values_sparse(
    J: sparse.spmatrix,
    k: int,
) -> NDArray[np.float64]:
    '''Estimate the ``k`` smallest singular values of a sparse matrix.

    Args:
        J: Sparse matrix with shape ``(m, n)``.
        k: Requested number of smallest singular values.

    Returns:
        Singular values sorted in ascending order. Very small matrices are
        densified because ``scipy.sparse.linalg.svds`` requires ``k < min(m,n)``.

    Raises:
        ValueError: If ``J`` is not sparse.
    '''

    if not sparse.issparse(J):
        raise ValueError("J must be sparse")

    max_k = max(1, min(k, min(J.shape) - 1))
    if min(J.shape) <= 2:
        return np.linalg.svd(J.toarray(), compute_uv=False)[-k:]

    values = svds(
        J,
        k=max_k,
        which="SM",
        return_singular_vectors=False,
    )
    return np.sort(values)


def largest_singular_values_sparse(
    J: sparse.spmatrix,
    k: int,
) -> NDArray[np.float64]:
    '''Estimate the ``k`` largest singular values of a sparse matrix.

    Args:
        J: Sparse matrix with shape ``(m, n)``.
        k: Requested number of largest singular values.

    Returns:
        Singular values sorted in descending order. Very small matrices are
        densified because ``scipy.sparse.linalg.svds`` requires ``k < min(m,n)``.

    Raises:
        ValueError: If ``J`` is not sparse.
    '''

    if not sparse.issparse(J):
        raise ValueError("J must be sparse")

    max_k = max(1, min(k, min(J.shape) - 1))
    if min(J.shape) <= 2:
        return np.linalg.svd(J.toarray(), compute_uv=False)[:k]

    values = svds(
        J,
        k=max_k,
        which="LM",
        return_singular_vectors=False,
    )
    return np.sort(values)[::-1]