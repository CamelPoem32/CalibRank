'''Dense and sparse observability projection and reduction utilities.'''

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.linalg import solve
from scipy.sparse.linalg import lsmr, spsolve

from .linalg import numerical_rank_dense
from .lie_se2 import se2_adjoint
from .lie_se3 import se3_adjoint
from .types import JacobianOptions


##################################################
# Sparse projection result
##################################################
@dataclass(frozen=True)
class SparseProjectionResult:
    '''Store the result of sparse trajectory-nuisance projection.
    
    Attributes:
        O_C: Projected calibration matrix in CSR format, shape ``(m, n_C)``.
        S_C: Reduced calibration information matrix, shape ``(n_C, n_C)``.
        solve_iterations: Number of LSMR iterations used for each calibration column.
        runtime_s: Total sparse projection runtime in seconds.
        shape: Shape of ``O_C``.
        nnz: Number of explicitly stored nonzero entries in ``O_C``.
        density: Fraction of nonzero entries in ``O_C``.
    '''

    O_C: sparse.csr_matrix
    S_C: NDArray[np.float64]
    solve_iterations: tuple[int, ...]
    runtime_s: float
    shape: tuple[int, int]
    nnz: int
    density: float


##################################################
# Dense trajectory projection
##################################################
def trajectory_projector_dense(J_T: ArrayLike) -> NDArray[np.float64]:
    '''Return dense `P_T_perp = I - J_T @ pinv(J_T)`.
    
    Args:
        J_T: Trajectory Jacobian, shape `(m, n_T)`.
    
    Returns:
        ndarray, shape `(m, m)`
    
    Raises:
        ValueError: If `J_T` is invalid.
    
    Notes:
        Perturbation convention: `J_T` columns are assumed to already use left perturbations.
    '''

    A = np.asarray(J_T, dtype=float)
    if A.ndim != 2 or not np.all(np.isfinite(A)):
        raise ValueError("J_T must be a finite matrix")
    pinv = np.linalg.pinv(A)
    # A: (m, n_T), pinv: (n_T, m) -> P_T: (m, m)
    P_T = A @ pinv
    return np.eye(A.shape[0]) - P_T


def effective_observability_dense(J_T: ArrayLike, J_C: ArrayLike) -> NDArray[np.float64]:
    '''Project calibration sensitivity away from the trajectory column space.
    
    Args:
        J_T: Trajectory Jacobian, shape ``(m, n_T)``.
        J_C: Calibration Jacobian, shape ``(m, n_C)``.
    
    Returns:
        Projected calibration matrix ``O_C``, shape ``(m, n_C)``.
    
    Raises:
        ValueError: If the inputs are not matrices with the same row count.
    '''

    T = np.asarray(J_T, dtype=float)
    C = np.asarray(J_C, dtype=float)
    if T.ndim != 2 or C.ndim != 2 or T.shape[0] != C.shape[0]:
        raise ValueError("J_T and J_C must be matrices with the same row count")
    P = trajectory_projector_dense(T)
    # P: (m, m), C: (m, n_C) -> O_C: (m, n_C)
    return P @ C


def effective_observability_dense_qr(J_T: ArrayLike, J_C: ArrayLike) -> NDArray[np.float64]:
    '''Build the QR-reduced calibration observability matrix.
    
    The returned matrix has the same nonzero singular values and rank as the
    explicit projection ``P_T_perp @ J_C`` without retaining redundant zero rows.
    
    Args:
        J_T: Trajectory Jacobian, shape ``(m, n_T)``.
        J_C: Calibration Jacobian, shape ``(m, n_C)``.
    
    Returns:
        Reduced matrix ``Q_2.T @ J_C``, shape ``(m-r, n_C)``, where ``r`` is the
        numerical rank of ``J_T``.
    
    Raises:
        ValueError: If the inputs do not have the same row count.
    '''

    T = np.asarray(J_T, dtype=float)
    C = np.asarray(J_C, dtype=float)
    if T.ndim != 2 or C.ndim != 2 or T.shape[0] != C.shape[0]:
        raise ValueError("J_T and J_C must have the same row count")
    Q, R = np.linalg.qr(T, mode="complete")
    rank = numerical_rank_dense(R)
    Q2 = Q[:, rank:]
    # Q2.T: (m-rank, m), C: (m, n_C) -> O_tilde: (m-rank, n_C)
    return Q2.T @ C


##################################################
# Sparse trajectory projection
##################################################
def effective_observability_sparse_lsmr(
    J_T: sparse.spmatrix,
    J_C: sparse.spmatrix,
    *,
    atol: float = 1e-14,
    btol: float = 1e-14,
    maxiter: int | None = None,
) -> SparseProjectionResult:
    '''Project sparse calibration columns with independent LSMR solves.
    
    For each calibration column ``c``, the function solves
    ``min_x ||J_T x - c||`` and stores the unreproducible component
    ``c - J_T x``. The dense residual-space projector is never constructed.
    
    Args:
        J_T: Sparse trajectory Jacobian, shape ``(m, n_T)``.
        J_C: Sparse calibration Jacobian, shape ``(m, n_C)``.
        atol: Absolute LSMR stopping tolerance.
        btol: Relative LSMR stopping tolerance.
        maxiter: Maximum iterations for each column solve. When omitted, a
            matrix-size-dependent default is used.
    
    Returns:
        Sparse projection matrix, reduced information matrix, solver statistics,
        and sparsity metadata.
    
    Raises:
        ValueError: If either input is dense or their row counts differ.
    '''

    if not sparse.issparse(J_T) or not sparse.issparse(J_C):
        raise ValueError("J_T and J_C must be sparse")
    if J_T.shape[0] != J_C.shape[0]:
        raise ValueError("J_T and J_C must have the same row count")
    Jt = J_T.tocsr()
    Jc = J_C.tocsc()
    start = perf_counter()
    columns = []
    iterations: list[int] = []
    for j in range(Jc.shape[1]):
        c = np.asarray(Jc[:, j].toarray()).ravel()
        sol = lsmr(Jt, c, atol=atol, btol=btol, maxiter=maxiter or 5 * max(Jt.shape))
        x = sol[0]
        iterations.append(int(sol[2]))
        # Jt: (m, n_T), x: (n_T,) -> projection_in_trajectory: (m,)
        o_col = c - Jt @ x
        columns.append(sparse.csr_matrix(o_col[:, None]))
    O_C = sparse.hstack(columns, format="csr") if columns else sparse.csr_matrix((J_T.shape[0], 0))
    # O_C.T: (n_C, m), O_C: (m, n_C) -> S_C: (n_C, n_C)
    S_C = (O_C.T @ O_C).toarray()
    runtime = perf_counter() - start
    nnz = int(O_C.nnz)
    density = nnz / (O_C.shape[0] * O_C.shape[1]) if O_C.shape[0] * O_C.shape[1] else 0.0
    return SparseProjectionResult(O_C=O_C, S_C=S_C, solve_iterations=tuple(iterations), runtime_s=runtime, shape=O_C.shape, nnz=nnz, density=density)


##################################################
# Reduced information and Schur complements
##################################################
def reduced_information_dense(O_C: ArrayLike) -> NDArray[np.float64]:
    '''Build the reduced calibration information matrix.
    
    Args:
        O_C: Projected calibration matrix, shape ``(m, n_C)``.
    
    Returns:
        Matrix ``S_C = O_C.T @ O_C``, shape ``(n_C, n_C)``.
    
    Raises:
        ValueError: If ``O_C`` is not a matrix.
    '''

    O = np.asarray(O_C, dtype=float)
    if O.ndim != 2:
        raise ValueError("O_C must be a matrix")
    # O.T: (n_C, m), O: (m, n_C) -> S_C: (n_C, n_C)
    return O.T @ O


def reduced_information_sparse_lsmr(J_T: sparse.spmatrix, J_C: sparse.spmatrix) -> NDArray[np.float64]:
    '''Build reduced information through sparse LSMR projection.
    
    Args:
        J_T: Sparse trajectory Jacobian, shape ``(m, n_T)``.
        J_C: Sparse calibration Jacobian, shape ``(m, n_C)``.
    
    Returns:
        Dense reduced information matrix, shape ``(n_C, n_C)``.
    '''

    return effective_observability_sparse_lsmr(J_T, J_C).S_C


def schur_complement_dense(J_T: ArrayLike, J_C: ArrayLike, *, rank_tolerance: float | None = None) -> NDArray[np.float64]:
    '''Calculate the dense calibration Schur complement.
    
    This ordinary-inverse formulation requires ``J_T`` to have full column rank.
    
    Args:
        J_T: Trajectory Jacobian, shape ``(m, n_T)``.
        J_C: Calibration Jacobian, shape ``(m, n_C)``.
        rank_tolerance: Optional numerical-rank threshold for ``J_T``.
    
    Returns:
        Dense Schur complement, shape ``(n_C, n_C)``.
    
    Raises:
        ValueError: If ``J_T`` is rank deficient.
    '''

    T = np.asarray(J_T, dtype=float)
    Cj = np.asarray(J_C, dtype=float)
    rank = numerical_rank_dense(T, rank_tolerance)
    if rank < T.shape[1]:
        raise ValueError("Schur complement with ordinary inverse requires full column rank J_T")
    # T.T: (n_T, m), T: (m, n_T) -> A: (n_T, n_T)
    A = T.T @ T
    # T.T: (n_T, m), Cj: (m, n_C) -> B: (n_T, n_C)
    B = T.T @ Cj
    # Cj.T: (n_C, m), Cj: (m, n_C) -> C: (n_C, n_C)
    C = Cj.T @ Cj
    X = solve(A, B, assume_a="pos")  # A: (n_T, n_T), B: (n_T, n_C) -> X: (n_T, n_C)
    # B.T: (n_C, n_T), X: (n_T, n_C) -> correction: (n_C, n_C)
    return C - B.T @ X


def schur_complement_sparse(J_T: sparse.spmatrix, J_C: sparse.spmatrix) -> NDArray[np.float64]:
    '''Calculate the sparse-normal-equation Schur complement.
    
    Args:
        J_T: Sparse trajectory Jacobian, shape ``(m, n_T)``.
        J_C: Sparse calibration Jacobian, shape ``(m, n_C)``.
    
    Returns:
        Dense calibration Schur complement, shape ``(n_C, n_C)``.
    
    Raises:
        ValueError: If either input is not sparse.
    '''

    if not sparse.issparse(J_T) or not sparse.issparse(J_C):
        raise ValueError("J_T and J_C must be sparse")
    # J_T.T: (n_T, m), J_T: (m, n_T) -> A: (n_T, n_T)
    A = (J_T.T @ J_T).tocsc()
    # J_T.T: (n_T, m), J_C: (m, n_C) -> B: (n_T, n_C)
    B = (J_T.T @ J_C).tocsc()
    # J_C.T: (n_C, m), J_C: (m, n_C) -> C: (n_C, n_C)
    C = (J_C.T @ J_C).toarray()
    X = np.column_stack([spsolve(A, B[:, j]) for j in range(B.shape[1])])
    # B.T: (n_C, n_T), X: (n_T, n_C) -> correction: (n_C, n_C)
    return C - B.T.toarray() @ X


##################################################
# Projection dispatch
##################################################
def effective_observability(
    J_T: ArrayLike | sparse.spmatrix,
    J_C: ArrayLike | sparse.spmatrix,
    *,
    method: str = "dense",
) -> NDArray[np.float64] | SparseProjectionResult:
    '''Dispatch calibration projection to a dense or sparse implementation.
    
    Args:
        J_T: Trajectory Jacobian.
        J_C: Calibration Jacobian with the same row count as ``J_T``.
        method: Projection method: ``"dense"``, ``"dense_qr"``, or
            ``"sparse_lsmr"``.
    
    Returns:
        Projected matrix for dense methods or a ``SparseProjectionResult`` for
        sparse LSMR projection.
    
    Raises:
        ValueError: If the method is unknown or sparse projection receives dense
            inputs.
    '''

    if method == "dense":
        return effective_observability_dense(J_T, J_C)  # type: ignore[arg-type]
    if method == "dense_qr":
        return effective_observability_dense_qr(J_T, J_C)  # type: ignore[arg-type]
    if method == "sparse_lsmr":
        if not sparse.issparse(J_T) or not sparse.issparse(J_C):
            raise ValueError("sparse_lsmr requires sparse inputs")
        return effective_observability_sparse_lsmr(J_T, J_C)
    raise ValueError("method must be 'dense', 'dense_qr', or 'sparse_lsmr'")


##################################################
# Motion-only sensitivity matrices
##################################################
def build_motion_only_matrix_dense(A_list: list[ArrayLike]) -> NDArray[np.float64]:
    '''Stack dense SE(3) motion-only extrinsic-sensitivity blocks.
    
    Args:
        A_list: Body relative motions, each shape ``(4, 4)``.
    
    Returns:
        Matrix ``vstack(Adj(A_m) - I_6)``, shape ``(6M, 6)``.
    '''

    blocks = [se3_adjoint(np.asarray(A, dtype=float)) - np.eye(6) for A in A_list]
    return np.vstack(blocks) if blocks else np.zeros((0, 6))


def build_motion_only_matrix_sparse(A_list: list[ArrayLike]) -> sparse.csr_matrix:
    '''Stack sparse SE(3) motion-only extrinsic-sensitivity blocks.
    
    Args:
        A_list: Body relative motions, each shape ``(4, 4)``.
    
    Returns:
        CSR matrix ``vstack(Adj(A_m) - I_6)``, shape ``(6M, 6)``.
    '''

    blocks = [sparse.csr_matrix(se3_adjoint(np.asarray(A, dtype=float)) - np.eye(6)) for A in A_list]
    return sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, 6))


def build_motion_only_matrix_dense_se2(A_list: list[ArrayLike]) -> NDArray[np.float64]:
    '''Stack dense SE(2) motion-only extrinsic-sensitivity blocks.
    
    Args:
        A_list: Planar body relative motions, each shape ``(3, 3)``.
    
    Returns:
        Matrix ``vstack(Adj(A_m) - I_3)``, shape ``(3M, 3)``.
    '''

    blocks = [se2_adjoint(np.asarray(A, dtype=float)) - np.eye(3) for A in A_list]
    return np.vstack(blocks) if blocks else np.zeros((0, 3))


def build_motion_only_matrix_sparse_se2(A_list: list[ArrayLike]) -> sparse.csr_matrix:
    '''Stack sparse SE(2) motion-only extrinsic-sensitivity blocks.
    
    Args:
        A_list: Planar body relative motions, each shape ``(3, 3)``.
    
    Returns:
        CSR matrix ``vstack(Adj(A_m) - I_3)``, shape ``(3M, 3)``.
    '''

    blocks = [sparse.csr_matrix(se2_adjoint(np.asarray(A, dtype=float)) - np.eye(3)) for A in A_list]
    return sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, 3))


##################################################
# Sliding-window observability analysis
##################################################
def analyze_observability_over_time(
    dataset: object,
    pose_provider: object,
    window_duration: float,
    window_step: float,
    include_priors: bool = False,
    include_smoothness: bool = False,
    use_sparse: bool = True,
    parameter_scaling: object | None = None,
    rank_tolerance: float | None = None,
    jacobian_options: JacobianOptions | None = None,
) -> list[dict[str, object]]:
    '''Evaluate observability diagnostics over sliding dataset windows.
    
    The dataset must provide ``start_time``, ``end_time``, and a
    ``window_jacobians`` method returning a Jacobian bundle, body motions, and
    factor counts.
    
    Args:
        dataset: Dataset-like object supplying window Jacobians.
        pose_provider: Trajectory provider passed to the dataset.
        window_duration: Window duration in seconds.
        window_step: Shift between consecutive windows in seconds.
        include_priors: Whether dataset assembly should include prior factors.
        include_smoothness: Whether dataset assembly should include smoothness factors.
        use_sparse: Use sparse LSMR projection when ``True``.
        parameter_scaling: Optional parameter scaling passed to dataset assembly.
        rank_tolerance: Optional numerical-rank tolerance.
        jacobian_options: Optional Jacobian calculation and checking settings.
    
    Returns:
        One diagnostic dictionary per complete sliding window.
    
    Raises:
        ValueError: If the window duration or step is non-positive.
    '''

    if window_duration <= 0.0 or window_step <= 0.0:
        raise ValueError("window_duration and window_step must be positive")
    t0 = float(getattr(dataset, "start_time"))
    t1 = float(getattr(dataset, "end_time"))
    results: list[dict[str, object]] = []
    start = t0
    while start + window_duration <= t1 + 1e-12:
        end = start + window_duration
        bundle, motions, counts = dataset.window_jacobians(
            start,
            end,
            pose_provider,
            include_priors=include_priors,
            include_smoothness=include_smoothness,
            use_sparse=use_sparse,
            parameter_scaling=parameter_scaling,
            jacobian_options=jacobian_options,
        )
        if use_sparse:
            proj = effective_observability_sparse_lsmr(bundle.J_T, bundle.J_C)  # type: ignore[arg-type]
            S_C = proj.S_C
            singular = np.linalg.svd(proj.O_C.toarray(), compute_uv=False)
            O_rank = numerical_rank_dense(proj.O_C.toarray(), rank_tolerance)
            nnz = int(bundle.J.nnz)  # type: ignore[attr-defined]
            runtime = proj.runtime_s
        else:
            O_C = effective_observability_dense(bundle.J_T, bundle.J_C)  # type: ignore[arg-type]
            S_C = reduced_information_dense(O_C)
            singular = np.linalg.svd(O_C, compute_uv=False)
            O_rank = numerical_rank_dense(O_C, rank_tolerance)
            nnz = int(np.count_nonzero(bundle.J))
            runtime = 0.0
        C_X = build_motion_only_matrix_dense(motions)
        nonzero = singular[singular > (_rank_tolerance_for_values(singular, rank_tolerance) if singular.size else 0.0)]
        results.append(
            {
                "window_start": start,
                "window_end": end,
                "num_imu_factors": counts.get("imu", 0),
                "num_lidar_factors": counts.get("lidar", 0),
                "rank_J": numerical_rank_dense(bundle.J.toarray() if sparse.issparse(bundle.J) else bundle.J, rank_tolerance),
                "rank_J_T": numerical_rank_dense(bundle.J_T.toarray() if sparse.issparse(bundle.J_T) else bundle.J_T, rank_tolerance),
                "rank_J_C": numerical_rank_dense(bundle.J_C.toarray() if sparse.issparse(bundle.J_C) else bundle.J_C, rank_tolerance),
                "rank_O_C": O_rank,
                "rank_C_X": numerical_rank_dense(C_X, rank_tolerance),
                "singular_values_O_C": singular,
                "smallest_nonzero_singular_value": float(nonzero[-1]) if nonzero.size else 0.0,
                "condition_number": float(nonzero[0] / nonzero[-1]) if nonzero.size else np.inf,
                "marginal_standard_deviations": _std_if_full_rank(S_C, rank_tolerance),
                "null_space_directions": None if O_rank == S_C.shape[0] else np.linalg.svd(S_C)[2][O_rank:].T,
                "runtime_s": runtime,
                "matrix_shape": bundle.J.shape,
                "matrix_nnz": nnz,
            }
        )
        start += window_step
    return results


##################################################
# Numerical helper functions
##################################################
def _rank_tolerance_for_values(values: NDArray[np.float64], tolerance: float | None) -> float:
    '''Calculate the numerical threshold used for singular values.
    
    Args:
        values: Singular values ordered from largest to smallest.
        tolerance: Explicit threshold. When omitted, a machine-precision threshold
            is calculated from the largest value.
    
    Returns:
        Numerical rank threshold.
    '''
    if tolerance is not None:
        return float(tolerance)
    vmax = float(values[0]) if values.size else 0.0
    return max(values.size, 1) * np.finfo(float).eps * vmax


def _std_if_full_rank(S_C: NDArray[np.float64], tolerance: float | None) -> NDArray[np.float64] | None:
    '''Calculate marginal standard deviations for a full-rank information matrix.
    
    Args:
        S_C: Reduced calibration information matrix, shape ``(n_C, n_C)``.
        tolerance: Numerical rank tolerance.
    
    Returns:
        Marginal standard deviations when ``S_C`` is full rank; otherwise ``None``.
    '''
    rank = numerical_rank_dense(S_C, tolerance)
    if rank < S_C.shape[0]:
        return None
    cov = solve(S_C, np.eye(S_C.shape[0]), assume_a="pos")
    return np.sqrt(np.maximum(np.diag(cov), 0.0))