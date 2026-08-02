'''Residual and Jacobian whitening without explicit covariance inversion.

Each whitening operation factors the covariance as ``Sigma = L L.T`` and
solves with the lower-triangular Cholesky factor. This is numerically preferable
to forming ``Sigma^{-1}`` explicitly and applies independently of the Lie-group
perturbation convention used to construct a factor.
'''

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cholesky, solve_triangular
from scipy import sparse

from .conventions import as_matrix


##################################################
# Covariance validation
##################################################
def _as_covariance(Sigma: ArrayLike, d: int) -> NDArray[np.float64]:
    '''Validate and return a symmetric covariance matrix.

    Args:
        Sigma (ArrayLike): Candidate covariance matrix.
        d (int): Expected residual dimension.

    Returns:
        NDArray[np.float64]: Validated covariance with shape ``(d, d)``.

    Raises:
        ValueError: If the matrix has the wrong shape, contains non-finite
            values, or is not symmetric within numerical tolerance.
    '''

    # Reuse the central finite-matrix validator before checking covariance
    # symmetry. Positive definiteness is checked later by Cholesky factorization.
    cov = as_matrix(Sigma, (d, d), "Sigma")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("Sigma must be symmetric")
    return cov


##################################################
# Dense residual and Jacobian whitening
##################################################
def whiten_residual_dense(r: ArrayLike, Sigma: ArrayLike) -> NDArray[np.float64]:
    '''Whiten a dense residual using a Cholesky triangular solve.

    Args:
        r (ArrayLike): Residual vector with shape ``(d,)``.
        Sigma (ArrayLike): Residual covariance with shape ``(d, d)``.

    Returns:
        NDArray[np.float64]: Whitened residual ``L^{-1} r``, shape ``(d,)``.

    Raises:
        ValueError: If the residual or covariance is invalid, or if the
            covariance is not positive definite.

    Notes:
        Whitening is applied after residual linearization and is independent of
        whether the underlying variables use left or right perturbations.
    '''

    # Validate the residual before matching and factoring its covariance.
    rv = np.asarray(r, dtype=float)
    if rv.ndim != 1 or not np.all(np.isfinite(rv)):
        raise ValueError("r must be a finite vector")
    cov = _as_covariance(Sigma, rv.size)

    # Sigma = L L.T and r_bar = L^{-1} r. No covariance inverse is formed.
    L = cholesky(cov, lower=True)
    return solve_triangular(L, rv, lower=True)  # L: (d, d), r: (d,) -> r_bar: (d,)


def whiten_jacobian_dense(H: ArrayLike, Sigma: ArrayLike) -> NDArray[np.float64]:
    '''Whiten a dense Jacobian block using its residual covariance.

    Args:
        H (ArrayLike): Jacobian matrix with shape ``(d, n)``.
        Sigma (ArrayLike): Residual covariance with shape ``(d, d)``.

    Returns:
        NDArray[np.float64]: Whitened Jacobian ``L^{-1} H``, shape ``(d, n)``.

    Raises:
        ValueError: If the Jacobian or covariance is invalid, or if the
            covariance is not positive definite.
    '''

    # Validate the local Jacobian and factor the covariance of its residual rows.
    J = np.asarray(H, dtype=float)
    if J.ndim != 2 or not np.all(np.isfinite(J)):
        raise ValueError("H must be a finite matrix")
    cov = _as_covariance(Sigma, J.shape[0])

    # Apply the same residual-space whitening transform to every Jacobian column.
    L = cholesky(cov, lower=True)
    return solve_triangular(L, J, lower=True)  # L: (d, d), H: (d, n) -> J_bar: (d, n)


def whiten_residual_and_jacobian_dense(
    r: ArrayLike,
    H: ArrayLike,
    Sigma: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Whiten a residual and its matching dense Jacobian together.

    Args:
        r (ArrayLike): Residual vector with shape ``(d,)``.
        H (ArrayLike): Jacobian matrix with shape ``(d, n)``.
        Sigma (ArrayLike): Residual covariance with shape ``(d, d)``.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: ``(r_bar, J_bar)``
        with shapes ``(d,)`` and ``(d, n)``.

    Raises:
        ValueError: If residual and Jacobian dimensions are inconsistent, the
            covariance is invalid, or the covariance is not positive definite.
    '''

    # Validate the shared residual dimension before computing one covariance
    # factorization for both outputs.
    rv = np.asarray(r, dtype=float)
    J = np.asarray(H, dtype=float)
    if rv.ndim != 1 or J.ndim != 2 or J.shape[0] != rv.size:
        raise ValueError("r must be (d,) and H must be (d, n)")
    cov = _as_covariance(Sigma, rv.size)
    L = cholesky(cov, lower=True)

    # Apply the identical lower-triangular solve to residual and Jacobian rows.
    r_bar = solve_triangular(L, rv, lower=True)  # L: (d, d), r: (d,) -> r_bar: (d,)
    J_bar = solve_triangular(L, J, lower=True)  # L: (d, d), H: (d, n) -> J_bar: (d, n)
    return r_bar, J_bar


##################################################
# Sparse factor-block preparation
##################################################
def whiten_factor_blocks_sparse(
    blocks: Sequence[tuple[ArrayLike, ArrayLike, ArrayLike]],
) -> list[tuple[NDArray[np.float64], sparse.csr_matrix]]:
    '''Whiten small factor blocks before sparse global insertion.

    Args:
        blocks (Sequence[tuple[ArrayLike, ArrayLike, ArrayLike]]): Sequence of
            ``(residual, local_jacobian, covariance)`` tuples. Each residual has
            shape ``(d_i,)``, each Jacobian has shape ``(d_i, n_i)``, and each
            covariance has shape ``(d_i, d_i)``.

    Returns:
        list[tuple[NDArray[np.float64], sparse.csr_matrix]]: Whitened residuals
        and CSR Jacobian blocks in the same input order.

    Raises:
        ValueError: If any local residual, Jacobian, or covariance is invalid.

    Notes:
        Small local sparse Jacobians are temporarily densified for triangular
        solves, then converted back to CSR before global matrix assembly.
    '''

    whitened: list[tuple[NDArray[np.float64], sparse.csr_matrix]] = []

    # Process each independent factor with the same dense whitening routine,
    # preserving the input ordering used by later sparse assembly.
    for r, H, Sigma in blocks:
        H_dense = H.toarray() if sparse.issparse(H) else np.asarray(H, dtype=float)
        r_bar, J_bar = whiten_residual_and_jacobian_dense(r, H_dense, Sigma)
        whitened.append((r_bar, sparse.csr_matrix(J_bar)))

    return whitened