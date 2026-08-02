'''Factor-specific calibration observability diagnostics.

The module keeps the observability pipeline explicit:

    assembled whitened Jacobian
        -> optional physical parameter scaling
        -> target and nuisance extraction
        -> nuisance projection
        -> optional column normalization for display
        -> practical-rank and local-accuracy diagnostics

Whitening is performed before these helpers receive a Jacobian. Physical
scaling changes parameter coordinates and therefore affects the projected
physical information. Column normalization is only a numerical display
operation and is never substituted for the physical matrix used for covariance
or CRLB-like calculations.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.linalg import cho_factor, cho_solve

from .assembly import JacobianBundle
from .lie_se3 import se3_adjoint
from .scaling import scale_jacobian_dense, scale_jacobian_sparse
from .diagnostics import (
    DEFAULT_PRACTICAL_RANK_POLICY,
    LocalAccuracyDiagnostics,
    PhysicalInformationDiagnostics,
    PracticalRankDiagnostics,
    PracticalRankPolicy,
    ScalarTimeOffsetDiagnostics,
    coordinate_metadata_for_variable,
    local_accuracy_diagnostics,
    physical_information_diagnostics,
    practical_rank_diagnostics,
    scalar_time_offset_diagnostics,
)
from .types import JacobianOptions

SUPPORTED_CALIBRATION_VARIABLES = (
    "T_B_I",
    "T_B_L",
    "b_g",
    "tau_I",
    "tau_L",
)
FIXED_T_B_L_CALIBRATION_VARIABLES = (
    "T_B_I",
    "b_g",
    "tau_I",
    "tau_L",
)
NormalizationMode = Literal[
    "none",
    "column",
    "physical_only",
    "physical_then_column",
]

##################################################
# Diagnostic result structures
##################################################
@dataclass(frozen=True)
class VariableColumnExtraction:
    '''Store one calibration-variable column extraction.

    Attributes:
        column_indices: Local column indices in ``bundle.J_C``.
        labels: Labels corresponding to the selected columns.
        jacobian_block: Selected Jacobian block with shape ``(m, n_X)``.
    '''

    column_indices: NDArray[np.int64]
    labels: list[str]
    jacobian_block: NDArray[np.float64] | sparse.csr_matrix

@dataclass(frozen=True)
class ColumnNormalizationResult:
    '''Store a Jacobian and its column-normalization metadata.

    For ``J`` with shape ``(m, n)``, each active column is divided by its original
    2-norm. Structurally zero columns remain exactly zero.

    Attributes:
        normalized_jacobian: Column-normalized dense or sparse Jacobian.
        original_column_norms: Original column 2-norms, shape ``(n,)``.
        zero_column_mask: Mask of columns rejected as zero.
        active_column_mask: Complement of ``zero_column_mask``.
    '''

    normalized_jacobian: NDArray[np.float64] | sparse.csr_matrix
    original_column_norms: NDArray[np.float64]
    zero_column_mask: NDArray[np.bool_]
    active_column_mask: NDArray[np.bool_]

@dataclass(frozen=True)
class RankDiagnostics:
    '''Store machine-rank and relative effective-rank diagnostics.

    Attributes:
        machine_rank: Rank obtained from the floating-point machine threshold.
        machine_rank_threshold: Threshold used for machine-rank calculation.
        effective_rank: Rank retained by the selected relative threshold.
        relative_rank_threshold: Relative singular-value threshold.
        singular_values: Singular values in descending order.
        normalized_singular_values: Singular values divided by ``sigma_max``.
        retained_singular_value_mask: Mask of relatively retained singular values.
    '''

    machine_rank: int
    machine_rank_threshold: float
    effective_rank: int
    relative_rank_threshold: float
    singular_values: NDArray[np.float64]
    normalized_singular_values: NDArray[np.float64]
    retained_singular_value_mask: NDArray[np.bool_]

@dataclass(frozen=True)
class EffectiveTargetObservabilityResult:
    '''Store projected observability diagnostics for one target variable.

    ``O_X_raw`` and ``O_X_physical`` are the physically meaningful projected
    matrices. ``O_X_normalized`` is intended only for numerical display and
    relative-direction diagnostics.

    Attributes:
        O_X_raw: Projected target matrix after nuisance compensation.
        O_X_physical: Physical projected target matrix.
        O_X_normalized: Optional column-normalized diagnostic matrix.
        machine_rank_O_X: Machine-precision rank of the physical matrix.
        effective_rank_O_X: Canonical practical rank of the physical matrix.
        machine_rank_threshold: Machine singular-value threshold.
        relative_rank_threshold: Requested legacy relative threshold.
        singular_values_O_X: Singular values of the physical projected matrix.
        normalized_singular_values_O_X: Singular values normalized by ``sigma_max``.
        null_space_O_X: Machine-null-space basis.
        effective_condition_number: Practical retained-subspace condition number.
        effective_condition_note: Interpretation of the practical condition number.
        ordinary_condition_number: Full physical Jacobian condition number.
        covariance_from_physical_information: Full covariance when column rank is full.
        covariance_note: Interpretation of the covariance field.
        practical_rank_diagnostics: Canonical practical-rank diagnostics.
        physical_information_diagnostics: Physical information diagnostics.
        scalar_time_offset_diagnostics: Scalar timing diagnostics when applicable.
        local_accuracy_diagnostics: Local CRLB-like accuracy diagnostics.
        target_labels: Target coordinate labels.
        nuisance_labels: Nuisance coordinate labels.
        original_target_column_norms: Physical target column norms.
        normalized_target_column_norms: Diagnostic matrix column norms.
        zero_target_column_mask: Negligible physical target columns.
        retained_singular_value_mask: Practically retained singular values.
        matrix_dimensions: Named matrix dimensions used by notebook checks.
        sparse_nnz: Number of stored sparse entries, or ``None`` for dense results.
    '''

    O_X_raw: NDArray[np.float64] | sparse.csr_matrix
    O_X_physical: NDArray[np.float64] | sparse.csr_matrix
    O_X_normalized: NDArray[np.float64] | sparse.csr_matrix
    machine_rank_O_X: int
    effective_rank_O_X: int
    machine_rank_threshold: float
    relative_rank_threshold: float
    singular_values_O_X: NDArray[np.float64]
    normalized_singular_values_O_X: NDArray[np.float64]
    null_space_O_X: NDArray[np.float64]
    effective_condition_number: float
    effective_condition_note: str
    ordinary_condition_number: float
    covariance_from_physical_information: NDArray[np.float64] | None
    covariance_note: str
    practical_rank_diagnostics: PracticalRankDiagnostics
    physical_information_diagnostics: PhysicalInformationDiagnostics
    scalar_time_offset_diagnostics: ScalarTimeOffsetDiagnostics | None
    local_accuracy_diagnostics: LocalAccuracyDiagnostics
    target_labels: list[str]
    nuisance_labels: list[str]
    original_target_column_norms: NDArray[np.float64]
    normalized_target_column_norms: NDArray[np.float64]
    zero_target_column_mask: NDArray[np.bool_]
    retained_singular_value_mask: NDArray[np.bool_]
    matrix_dimensions: dict[str, tuple[int, int]]
    sparse_nnz: int | None

@dataclass(frozen=True)
class MotionOnlyObservabilityResult:
    '''Store LiDAR motion-only extrinsic sensitivity diagnostics.

    The result assumes body motions are known. It does not establish joint
    observability when trajectory and other calibration variables may compensate.

    Attributes:
        C_X_L_raw: Stacked ``Adj(A_m) - I_6`` matrix.
        C_X_L_column_normalized: Column-normalized display matrix.
        machine_rank: Machine-precision rank.
        effective_rank: Practical rank retained by the canonical policy.
        practical_rank: Compatibility alias of ``effective_rank``.
        machine_rank_threshold: Machine singular-value threshold.
        relative_rank_threshold: Requested legacy relative threshold.
        singular_values: Physical singular values.
        normalized_singular_values: Singular values divided by ``sigma_max``.
        null_space_basis: Machine-null-space basis.
        original_column_norms: Original column norms.
        zero_column_mask: Structurally or numerically zero columns.
        matrix_dimensions: Named matrix dimensions.
        sparse_nnz: Number of sparse entries, or ``None`` for dense results.
        practical_rank_diagnostics: Canonical practical-rank diagnostics.
    '''

    C_X_L_raw: NDArray[np.float64] | sparse.csr_matrix
    C_X_L_column_normalized: NDArray[np.float64] | sparse.csr_matrix
    machine_rank: int
    effective_rank: int
    practical_rank: int
    machine_rank_threshold: float
    relative_rank_threshold: float
    singular_values: NDArray[np.float64]
    normalized_singular_values: NDArray[np.float64]
    null_space_basis: NDArray[np.float64]
    original_column_norms: NDArray[np.float64]
    zero_column_mask: NDArray[np.bool_]
    matrix_dimensions: dict[str, tuple[int, int]]
    sparse_nnz: int | None
    practical_rank_diagnostics: PracticalRankDiagnostics | None = None

@dataclass(frozen=True)
class ImuGyroMotionSensitivityResult:
    '''Store gyro-only sensitivity of the ``T_B_I`` calibration block.

    Attributes:
        full_T_B_I_sensitivity_matrix: Six-column gyro sensitivity matrix.
        full_T_B_I_normalized: Column-normalized six-column matrix.
        rotation_only_sensitivity_matrix: First three rotation columns.
        rotation_only_normalized: Column-normalized rotation block.
        translation_column_norms: Norms of translation columns 3 through 5.
        structural_zero_column_report: Translation-zero consistency report.
        full_machine_rank: Machine rank of the full six-column block.
        full_effective_rank: Practical rank of the full block.
        full_practical_rank: Compatibility alias of ``full_effective_rank``.
        rotation_machine_rank: Machine rank of the rotation block.
        rotation_effective_rank: Practical rank of the rotation block.
        rotation_practical_rank: Compatibility alias of ``rotation_effective_rank``.
        full_singular_values: Singular values of the full block.
        full_normalized_singular_values: Normalized full-block singular values.
        rotation_singular_values: Singular values of the rotation block.
        rotation_normalized_singular_values: Normalized rotation singular values.
        original_column_norms: Full-block column norms.
        zero_column_mask: Full-block zero-column mask.
    '''

    full_T_B_I_sensitivity_matrix: NDArray[np.float64] | sparse.csr_matrix
    full_T_B_I_normalized: NDArray[np.float64] | sparse.csr_matrix
    rotation_only_sensitivity_matrix: NDArray[np.float64] | sparse.csr_matrix
    rotation_only_normalized: NDArray[np.float64] | sparse.csr_matrix
    translation_column_norms: NDArray[np.float64]
    structural_zero_column_report: dict[str, object]
    full_machine_rank: int
    full_effective_rank: int
    full_practical_rank: int
    rotation_machine_rank: int
    rotation_effective_rank: int
    rotation_practical_rank: int
    full_singular_values: NDArray[np.float64]
    full_normalized_singular_values: NDArray[np.float64]
    rotation_singular_values: NDArray[np.float64]
    rotation_normalized_singular_values: NDArray[np.float64]
    original_column_norms: NDArray[np.float64]
    zero_column_mask: NDArray[np.bool_]

@dataclass(frozen=True)
class AccelerometerSensitivityResult:
    '''Store accelerometer-only ``T_B_I`` factor sensitivity.

    No nuisance projection is applied. The raw, whitened, and physical fields
    currently reference the same already-whitened ``T_B_I`` block.

    Attributes:
        C_X_I_accel_raw: Extracted accelerometer sensitivity matrix.
        C_X_I_accel_whitened: Already-whitened sensitivity matrix.
        C_X_I_accel_physical: Physical-coordinate sensitivity matrix.
        C_X_I_accel_column_normalized: Column-normalized display matrix.
        machine_rank: Machine-precision rank.
        practical_rank: Canonical practical rank.
        singular_values: Physical singular values.
        normalized_singular_values: Singular values divided by ``sigma_max``.
        original_column_norms: Original column norms.
        zero_column_mask: Negligible whole-column mask.
        practical_rank_diagnostics: Canonical practical-rank diagnostics.
        matrix_dimensions: Named matrix dimensions.
        sparse_nnz: Number of sparse entries, or ``None`` for dense results.
    '''

    C_X_I_accel_raw: NDArray[np.float64] | sparse.csr_matrix
    C_X_I_accel_whitened: NDArray[np.float64] | sparse.csr_matrix
    C_X_I_accel_physical: NDArray[np.float64] | sparse.csr_matrix
    C_X_I_accel_column_normalized: NDArray[np.float64] | sparse.csr_matrix
    machine_rank: int
    practical_rank: int
    singular_values: NDArray[np.float64]
    normalized_singular_values: NDArray[np.float64]
    original_column_norms: NDArray[np.float64]
    zero_column_mask: NDArray[np.bool_]
    practical_rank_diagnostics: PracticalRankDiagnostics
    matrix_dimensions: dict[str, tuple[int, int]]
    sparse_nnz: int | None

@dataclass(frozen=True)
class SlidingWindowFactorDiagnostics:
    '''Store all factor-specific diagnostics for one time window.

    Attributes:
        window_start: Inclusive window start time.
        window_end: Window end time.
        counts: Accepted factor counts reported by the dataset.
        motion_lidar: LiDAR motion-only diagnostics when LiDAR factors exist.
        motion_imu: Gyro-only IMU diagnostics when the required block exists.
        target_results: Joint projected diagnostics keyed by calibration variable.
    '''

    window_start: float
    window_end: float
    counts: dict[str, int]
    motion_lidar: MotionOnlyObservabilityResult | None
    motion_imu: ImuGyroMotionSensitivityResult | None
    target_results: dict[str, EffectiveTargetObservabilityResult]

##################################################
# Matrix conversion and validation helpers
##################################################
def _dense(matrix: ArrayLike | sparse.spmatrix) -> NDArray[np.float64]:
    '''Convert a matrix to a finite two-dimensional dense array.

    Args:
        matrix: Dense or sparse matrix to validate.

    Returns:
        Dense floating-point matrix.

    Raises:
        ValueError: If the input is not a finite two-dimensional matrix.
    '''

    # Convert sparse inputs only at diagnostic boundaries where dense linear
    # algebra is explicitly required.
    value = (
        matrix.toarray()
        if sparse.issparse(matrix)
        else np.asarray(matrix, dtype=float)
    )

    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("matrix must be a finite 2D array")

    return value


def _labels_from_bundle(bundle: JacobianBundle) -> list[str]:
    '''Return calibration-column labels aligned with ``bundle.J_C``.

    Args:
        bundle: Assembled Jacobian bundle.

    Returns:
        Metadata labels when their count matches ``J_C`` columns; otherwise
        generated labels ``c_0``, ``c_1``, and so on.
    '''

    labels = list(bundle.metadata.get("calibration_labels", []))
    if len(labels) == bundle.J_C.shape[1]:
        return labels

    return [
        f"c_{column_index}"
        for column_index in range(bundle.J_C.shape[1])
    ]


def _check_relative_threshold(relative_rank_threshold: float) -> float:
    '''Validate a relative rank threshold.

    Args:
        relative_rank_threshold: Candidate threshold.

    Returns:
        Validated floating-point threshold.

    Raises:
        ValueError: If the threshold is not finite or lies outside ``(0, 1)``.
    '''
    threshold = float(relative_rank_threshold)
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("relative_rank_threshold must satisfy 0 < threshold < 1")
    return threshold


def _check_columns(
    total_columns: int,
    columns: ArrayLike,
    name: str,
) -> NDArray[np.int64]:
    '''Validate and normalize a sequence of matrix column indices.

    Args:
        total_columns: Number of available columns.
        columns: Column indices to validate.
        name: Argument name used in validation errors.

    Returns:
        One-dimensional ``int64`` array of unique valid indices.

    Raises:
        ValueError: If an index is out of range or duplicated.
    '''

    indices = np.asarray(columns, dtype=int).reshape(-1)

    # Validate matrix bounds before checking uniqueness, preserving the original
    # error priority for malformed inputs.
    if indices.size and (
        np.any(indices < 0)
        or np.any(indices >= total_columns)
    ):
        raise ValueError(
            f"{name} contains indices outside [0, {total_columns})"
        )

    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicate indices")

    return indices.astype(np.int64, copy=False)


def _check_target_nuisance(
    total_columns: int,
    target_columns: ArrayLike,
    nuisance_columns: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    '''Validate disjoint target and nuisance column selections.

    Args:
        total_columns: Number of available matrix columns.
        target_columns: Target-variable columns.
        nuisance_columns: Nuisance-variable columns.

    Returns:
        Tuple ``(target, nuisance)`` of validated ``int64`` arrays.

    Raises:
        ValueError: If the target is empty or the two selections overlap.
    '''

    target = _check_columns(
        total_columns,
        target_columns,
        "target_columns",
    )
    nuisance = _check_columns(
        total_columns,
        nuisance_columns,
        "nuisance_columns",
    )

    if target.size == 0:
        raise ValueError("target_columns must be nonempty")

    if np.intersect1d(target, nuisance).size:
        raise ValueError(
            "target_columns and nuisance_columns must be disjoint"
        )

    return target, nuisance


def _slice_columns(
    matrix: ArrayLike | sparse.spmatrix,
    columns: NDArray[np.int64],
) -> NDArray[np.float64] | sparse.csr_matrix:
    '''Extract selected columns while preserving matrix sparsity.

    Args:
        matrix: Dense or sparse source matrix.
        columns: Validated local column indices.

    Returns:
        Dense array or CSR sparse matrix containing the selected columns.
    '''

    if sparse.issparse(matrix):
        return matrix.tocsr()[:, columns].tocsr()

    return np.asarray(matrix, dtype=float)[:, columns]

def _column_norms(
    matrix: ArrayLike | sparse.spmatrix,
) -> NDArray[np.float64]:
    '''Calculate column 2-norms without unnecessarily densifying a matrix.

    Args:
        matrix: Dense or sparse matrix.

    Returns:
        Column 2-norms with shape ``(n,)``.
    '''

    if sparse.issparse(matrix):
        squared_norms = matrix.power(2).sum(axis=0)
        return np.sqrt(np.asarray(squared_norms).ravel())

    return np.linalg.norm(
        np.asarray(matrix, dtype=float),
        axis=0,
    )


def _machine_null_space(
    matrix: ArrayLike | sparse.spmatrix,
    machine_rank: int,
) -> NDArray[np.float64]:
    '''Return the right-singular-vector basis below the machine rank.

    Args:
        matrix: Matrix whose right null space is required.
        machine_rank: Number of singular directions treated as nonzero.

    Returns:
        Null-space basis with shape ``(n, n - machine_rank)``.
    '''

    _, _, Vt = np.linalg.svd(
        _dense(matrix),
        full_matrices=True,
    )

    return Vt[machine_rank:].T.copy()

##################################################
# Calibration column extraction
##################################################
def extract_variable_columns(
    bundle: JacobianBundle,
    variable_name: str,
) -> VariableColumnExtraction:
    '''Extract the active calibration block of one target variable.

    Args:
        bundle: Assembled Jacobian bundle.
        variable_name: Supported calibration variable to select.

    Returns:
        Selected local indices, labels, and ``J_C`` block.

    Raises:
        ValueError: If the variable is unsupported or its active block is empty.
        KeyError: If the variable is not active in the bundle.
    '''

    # Resolve the active block in local calibration-column coordinates.
    if variable_name not in SUPPORTED_CALIBRATION_VARIABLES:
        raise ValueError(
            f"unsupported calibration variable {variable_name!r}"
        )

    if variable_name not in bundle.calibration_column_slices:
        raise KeyError(f"{variable_name!r} is not active in this bundle")

    block_slice = bundle.calibration_column_slices[variable_name]
    columns = np.arange(
        block_slice.start or 0,
        block_slice.stop or 0,
        dtype=np.int64,
    )

    if columns.size == 0:
        raise ValueError(
            f"{variable_name!r} has an empty calibration block"
        )

    columns = _check_columns(
        bundle.J_C.shape[1],
        columns,
        "target_columns",
    )

    # Keep labels and matrix columns in exactly the same local order.
    labels = _labels_from_bundle(bundle)
    selected_labels = [labels[column] for column in columns]
    jacobian_block = _slice_columns(bundle.J_C, columns)

    return VariableColumnExtraction(
        columns,
        selected_labels,
        jacobian_block,
    )


def extract_nuisance_columns(
    bundle: JacobianBundle,
    excluded_variable_name: str,
) -> VariableColumnExtraction:
    '''Extract all active calibration columns except one target block.

    Args:
        bundle: Assembled Jacobian bundle.
        excluded_variable_name: Target variable omitted from nuisance columns.

    Returns:
        Nuisance indices, labels, and ``J_C`` block.

    Raises:
        ValueError: If target and nuisance columns overlap or fail to cover all
            active calibration columns.
    '''

    target = extract_variable_columns(
        bundle,
        excluded_variable_name,
    )

    # Form the ordered complement of the target block in local J_C coordinates.
    total_columns = bundle.J_C.shape[1]
    all_columns = np.arange(total_columns, dtype=np.int64)
    nuisance = np.setdiff1d(
        all_columns,
        target.column_indices,
        assume_unique=True,
    )

    # Keep explicit partition checks close to the extraction operation.
    if np.intersect1d(target.column_indices, nuisance).size:
        raise ValueError("target and nuisance columns overlap")

    if np.union1d(target.column_indices, nuisance).size != total_columns:
        raise ValueError(
            "target and nuisance columns do not cover all active "
            "calibration columns"
        )

    labels = _labels_from_bundle(bundle)
    nuisance_labels = [labels[column] for column in nuisance]
    jacobian_block = _slice_columns(bundle.J_C, nuisance)

    return VariableColumnExtraction(
        nuisance,
        nuisance_labels,
        jacobian_block,
    )

##################################################
# Column normalization for diagnostics
##################################################
def normalize_jacobian_columns_dense(
    J: ArrayLike,
    *,
    zero_tolerance: float = 0.0,
) -> ColumnNormalizationResult:
    '''Normalize active columns of a dense Jacobian for diagnostics.

    Args:
        J: Dense matrix with shape ``(m, n)``.
        zero_tolerance: Columns with norm at or below this value remain zero.

    Returns:
        Normalized matrix, original norms, and zero/active masks.

    Raises:
        ValueError: If ``zero_tolerance`` is negative or ``J`` is invalid.
    '''

    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be nonnegative")

    matrix = _dense(J)
    column_norms = _column_norms(matrix)
    zero_mask = column_norms <= float(zero_tolerance)
    active_mask = ~zero_mask

    # Multiplication by D_inv preserves the original implementation and keeps
    # rejected columns exactly zero.
    inverse_norms = np.zeros_like(column_norms)
    inverse_norms[active_mask] = 1.0 / column_norms[active_mask]
    normalized = matrix @ np.diag(inverse_norms)

    return ColumnNormalizationResult(
        normalized,
        column_norms,
        zero_mask,
        active_mask,
    )


def normalize_jacobian_columns_sparse(
    J: sparse.spmatrix,
    *,
    zero_tolerance: float = 0.0,
) -> ColumnNormalizationResult:
    '''Normalize active columns of a sparse Jacobian without densifying it.

    Args:
        J: Sparse matrix with shape ``(m, n)``.
        zero_tolerance: Columns with norm at or below this value remain zero.

    Returns:
        CSR-normalized matrix, original norms, and zero/active masks.

    Raises:
        ValueError: If ``J`` is not sparse or ``zero_tolerance`` is negative.
    '''

    if not sparse.issparse(J):
        raise ValueError("J must be sparse")

    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be nonnegative")

    matrix = J.tocsc(copy=True)
    column_norms = _column_norms(matrix)
    zero_mask = column_norms <= float(zero_tolerance)
    active_mask = ~zero_mask

    # Apply a sparse diagonal scaling and remove explicit stored zeros.
    inverse_norms = np.zeros_like(column_norms)
    inverse_norms[active_mask] = 1.0 / column_norms[active_mask]
    D_inv = sparse.diags(inverse_norms, format="csc")
    normalized = (matrix @ D_inv).tocsr()
    normalized.eliminate_zeros()

    return ColumnNormalizationResult(
        normalized,
        column_norms,
        zero_mask,
        active_mask,
    )

##################################################
# Singular-value and rank diagnostics
##################################################
def rank_diagnostics_from_singular_values(
    singular_values: ArrayLike,
    matrix_shape: tuple[int, int],
    *,
    relative_rank_threshold: float = 1e-5,
) -> RankDiagnostics:
    '''Compute machine and relative ranks from singular values.

    Args:
        singular_values: Singular values ordered from largest to smallest.
        matrix_shape: Shape of the matrix that produced the singular values.
        relative_rank_threshold: Relative cutoff applied to ``sigma_max``.

    Returns:
        Machine-rank and relative effective-rank diagnostics.

    Raises:
        ValueError: If the relative threshold is invalid.
    '''

    threshold = _check_relative_threshold(relative_rank_threshold)
    singular = np.asarray(
        singular_values,
        dtype=float,
    ).reshape(-1)

    # Empty matrices carry no singular directions.
    if singular.size == 0:
        return RankDiagnostics(
            0,
            0.0,
            0,
            threshold,
            singular,
            singular.copy(),
            np.zeros(0, dtype=bool),
        )

    sigma_max = float(singular[0])

    # A zero matrix must not be normalized by an artificial scale.
    if sigma_max <= 0.0:
        return RankDiagnostics(
            0,
            0.0,
            0,
            threshold,
            singular,
            np.zeros_like(singular),
            np.zeros_like(singular, dtype=bool),
        )

    machine_threshold = (
        max(matrix_shape)
        * np.finfo(float).eps
        * sigma_max
    )
    normalized = singular / sigma_max
    retained = normalized > threshold

    return RankDiagnostics(
        machine_rank=int(np.sum(singular > machine_threshold)),
        machine_rank_threshold=float(machine_threshold),
        effective_rank=int(np.sum(retained)),
        relative_rank_threshold=threshold,
        singular_values=singular,
        normalized_singular_values=normalized,
        retained_singular_value_mask=retained,
    )


def rank_diagnostics_dense(
    J: ArrayLike | sparse.spmatrix,
    *,
    relative_rank_threshold: float = 1e-5,
) -> RankDiagnostics:
    '''Compute SVD rank diagnostics for a small matrix.

    Args:
        J: Dense or sparse matrix that may be safely densified.
        relative_rank_threshold: Relative singular-value cutoff.

    Returns:
        Rank diagnostics derived from the matrix singular values.
    '''

    matrix = _dense(J)
    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    return rank_diagnostics_from_singular_values(
        singular_values,
        matrix.shape,
        relative_rank_threshold=relative_rank_threshold,
    )


def effective_rank_threshold_sweep(
    singular_values: ArrayLike,
    relative_thresholds: ArrayLike | None = None,
) -> dict[float, int]:
    '''Evaluate effective rank across several relative thresholds.

    Args:
        singular_values: Singular values ordered from largest to smallest.
        relative_thresholds: Thresholds to evaluate. The standard logarithmic
            sweep is used when omitted.

    Returns:
        Mapping from each validated threshold to its effective rank.

    Raises:
        ValueError: If any supplied threshold is invalid.
    '''

    default_thresholds = [
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
        1e-9,
    ]
    thresholds = np.asarray(
        default_thresholds
        if relative_thresholds is None
        else relative_thresholds,
        dtype=float,
    ).reshape(-1)

    singular = np.asarray(
        singular_values,
        dtype=float,
    ).reshape(-1)
    sigma_max = float(singular[0]) if singular.size else 0.0

    ranks: dict[float, int] = {}
    for threshold in thresholds:
        checked = _check_relative_threshold(float(threshold))

        if sigma_max <= 0.0:
            ranks[float(checked)] = 0
        else:
            ranks[float(checked)] = int(
                np.sum(singular > checked * sigma_max)
            )

    return ranks

##################################################
# Physical scaling and projection preparation
##################################################
def _scaling_matrix(
    matrix: ArrayLike | sparse.spmatrix,
    parameter_scaling: object | None,
) -> NDArray[np.float64] | sparse.csr_matrix | None:
    '''Build a dense or sparse parameter-scaling matrix.

    Args:
        matrix: Jacobian whose column count defines the scaling dimension.
        parameter_scaling: ``None``, a length-``n`` vector, or an ``(n, n)``
            matrix.

    Returns:
        Scaling matrix matching the Jacobian storage type, or ``None``.

    Raises:
        ValueError: If the scaling shape is incompatible with the Jacobian.
    '''

    if parameter_scaling is None:
        return None

    total_columns = matrix.shape[1]
    values = np.asarray(parameter_scaling, dtype=float)

    # A vector specifies independent scaling for every Jacobian column.
    if values.ndim == 1:
        if values.shape != (total_columns,):
            raise ValueError(
                "parameter_scaling vector must have one entry per column"
            )
        values = np.diag(values)

    if values.shape != (total_columns, total_columns):
        raise ValueError(
            "parameter_scaling must have shape (n,) or (n, n)"
        )

    if sparse.issparse(matrix):
        return sparse.csr_matrix(values)

    return values


def _apply_physical_scaling(
    matrix: ArrayLike | sparse.spmatrix,
    parameter_scaling: object | None,
) -> NDArray[np.float64] | sparse.csr_matrix:
    '''Apply physical parameter scaling to a Jacobian.

    Args:
        matrix: Dense or sparse Jacobian.
        parameter_scaling: Optional parameter-scaling vector or matrix.

    Returns:
        Physically scaled matrix with the same dense/sparse representation.
    '''

    scaling = _scaling_matrix(matrix, parameter_scaling)

    if scaling is None:
        if sparse.issparse(matrix):
            return matrix.tocsr()
        return np.asarray(matrix, dtype=float)

    # J: (m, n), D: (n, n), J_scaled = J @ D: (m, n).
    if sparse.issparse(matrix):
        return scale_jacobian_sparse(matrix, scaling)

    return scale_jacobian_dense(matrix, scaling)


def _prepare_projection_matrix(
    J: ArrayLike | sparse.spmatrix,
    normalization: NormalizationMode,
    parameter_scaling: object | None,
) -> NDArray[np.float64] | sparse.csr_matrix:
    '''Prepare a Jacobian for target/nuisance projection.

    Args:
        J: Dense or sparse input Jacobian.
        normalization: Requested scaling and display-normalization mode.
        parameter_scaling: Optional physical parameter scaling.

    Returns:
        Matrix after optional physical scaling. Column normalization is deferred
        until after nuisance projection.

    Raises:
        ValueError: If ``normalization`` is unknown.
    '''

    valid_modes = {
        "none",
        "column",
        "physical_only",
        "physical_then_column",
    }
    if normalization not in valid_modes:
        raise ValueError("unknown normalization mode")

    if normalization in {"physical_only", "physical_then_column"}:
        return _apply_physical_scaling(J, parameter_scaling)

    if sparse.issparse(J):
        return J.tocsr()

    return np.asarray(J, dtype=float)


def _normalize_projected_matrix(
    O_X_raw: ArrayLike | sparse.spmatrix,
    normalization: NormalizationMode,
) -> ColumnNormalizationResult:
    '''Apply optional post-projection column normalization.

    Args:
        O_X_raw: Projected physical target matrix.
        normalization: Requested normalization mode.

    Returns:
        Column-normalization result. For non-column modes the original matrix is
        returned with measured column norms and masks.
    '''

    if normalization in {"column", "physical_then_column"}:
        if sparse.issparse(O_X_raw):
            return normalize_jacobian_columns_sparse(O_X_raw)
        return normalize_jacobian_columns_dense(O_X_raw)

    # Even without display normalization, expose the same norm and zero-column
    # metadata expected by downstream diagnostics.
    if sparse.issparse(O_X_raw):
        matrix = O_X_raw.tocsr()
        norms = _column_norms(matrix)
        zero_mask = norms <= 0.0
        return ColumnNormalizationResult(
            matrix,
            norms,
            zero_mask,
            ~zero_mask,
        )

    matrix = np.asarray(O_X_raw, dtype=float)
    norms = _column_norms(matrix)
    zero_mask = norms <= 0.0

    return ColumnNormalizationResult(
        matrix,
        norms,
        zero_mask,
        ~zero_mask,
    )

##################################################
# Physical covariance and target result assembly
##################################################
def covariance_from_physical_information(
    physical_matrix: ArrayLike | sparse.spmatrix,
) -> tuple[NDArray[np.float64] | None, float, str]:
    '''Compute covariance from a full-column-rank physical Jacobian.

    Args:
        physical_matrix: Whitened Jacobian in the parameter coordinates in which
            covariance is requested.

    Returns:
        Tuple ``(covariance, condition_number, note)``. ``covariance`` is
        ``None`` and the condition number is infinite when the Jacobian is rank
        deficient.
    '''

    J_p = _dense(physical_matrix)

    # Use the physical matrix directly. A merely column-normalized matrix would
    # discard the relative parameter information required by covariance.
    _, singular_values, Vt = np.linalg.svd(
        J_p,
        full_matrices=False,
    )
    diagnostics = rank_diagnostics_from_singular_values(
        singular_values,
        J_p.shape,
    )

    n_parameters = J_p.shape[1]
    if diagnostics.machine_rank < n_parameters:
        return (
            None,
            float("inf"),
            "rank-deficient physical Jacobian; ordinary covariance is "
            "undefined",
        )

    # For full column rank, (J_p.T J_p)^-1 = V diag(1 / sigma_i^2) V.T.
    V = Vt.T
    inverse_variances = 1.0 / np.square(singular_values)
    covariance = (V * inverse_variances) @ V.T
    jacobian_condition_number = float(
        singular_values[0] / singular_values[-1]
    )

    return (
        covariance,
        jacobian_condition_number,
        "full-rank covariance computed from the SVD of the physical Jacobian",
    )


def _effective_condition(
    singular_values: NDArray[np.float64],
    retained_mask: NDArray[np.bool_],
) -> tuple[float, str]:
    '''Compute the condition number of retained singular values.

    Args:
        singular_values: Full singular-value array.
        retained_mask: Mask selecting retained values.

    Returns:
        Tuple containing the condition number and an interpretation note.
    '''

    retained = singular_values[retained_mask]

    if retained.size < 2:
        return (
            float("nan"),
            "fewer than two singular values retained by the relative "
            "threshold",
        )

    return (
        float(retained[0] / retained[-1]),
        "sigma_max divided by the smallest retained singular value",
    )


def _make_target_result(
    O_X_raw: NDArray[np.float64] | sparse.csr_matrix,
    normalization: NormalizationMode,
    relative_rank_threshold: float,
    practical_rank_policy: PracticalRankPolicy,
    tau_target_std_seconds: float | None,
    target_labels: list[str],
    nuisance_labels: list[str],
    sparse_nnz: int | None,
    *,
    variable_name: str = "target",
    lidar_rate_hz: float | None = None,
    coordinate_null_fraction_tolerance: float = 1e-6,
) -> EffectiveTargetObservabilityResult:
    '''Assemble all diagnostics for one projected target matrix.

    Args:
        O_X_raw: Projected physical target matrix.
        normalization: Post-projection normalization mode.
        relative_rank_threshold: Legacy relative threshold stored in the result.
        practical_rank_policy: Canonical practical-rank policy.
        tau_target_std_seconds: Optional scalar timing accuracy requirement.
        target_labels: Incoming target labels.
        nuisance_labels: Nuisance-variable labels.
        sparse_nnz: Number of sparse entries, or ``None``.
        variable_name: Target calibration variable name.
        lidar_rate_hz: Optional LiDAR rate for timing bounds in frame units.
        coordinate_null_fraction_tolerance: Null-projection tolerance used for
            coordinate-wise boundedness.

    Returns:
        Complete projected target observability result.

    Raises:
        AssertionError: If independently calculated scalar timing diagnostics
            disagree.
    '''

    # Column normalization is a display operation. All information, rank, and
    # local-accuracy calculations below use the unnormalized physical matrix.
    normalized = _normalize_projected_matrix(
        O_X_raw,
        normalization,
    )
    diagnostic_matrix = _dense(normalized.normalized_jacobian)
    physical_matrix = _dense(O_X_raw)

    practical = practical_rank_diagnostics(
        physical_matrix,
        policy=practical_rank_policy,
    )
    information = physical_information_diagnostics(
        physical_matrix,
        practical_rank_result=practical,
        policy=practical_rank_policy,
    )

    # Scalar time-offset variables expose a dedicated physical sensitivity and
    # standard-deviation bound when a target requirement is supplied.
    tau_diagnostics = None
    if (
        physical_matrix.shape[1] == 1
        and tau_target_std_seconds is not None
    ):
        tau_diagnostics = scalar_time_offset_diagnostics(
            physical_matrix,
            policy=practical_rank_policy,
            target_std_seconds=tau_target_std_seconds,
        )

    # Central coordinate metadata supersedes arbitrary incoming labels in the
    # final result. The argument remains present for API compatibility.
    _ = target_labels
    coordinate_labels, coordinate_units = coordinate_metadata_for_variable(
        variable_name,
        physical_matrix.shape[1],
    )
    accuracy = local_accuracy_diagnostics(
        physical_matrix,
        variable_name=variable_name,
        coordinate_labels=coordinate_labels,
        coordinate_units=coordinate_units,
        practical_rank_result=practical,
        lidar_rate_hz=lidar_rate_hz,
        target_std_seconds=tau_target_std_seconds,
        coordinate_null_fraction_tolerance=(
            coordinate_null_fraction_tolerance
        ),
    )

    # Keep the generic and scalar timing diagnostics locked to the same physical
    # interpretation.
    if tau_diagnostics is not None:
        if not np.isclose(
            accuracy.scalar_std_bound,
            tau_diagnostics.local_std_bound_tau_seconds,
            equal_nan=True,
        ):
            raise AssertionError(
                "local tau accuracy must match scalar time-offset diagnostics"
            )

        if accuracy.meets_target != tau_diagnostics.meets_target:
            raise AssertionError(
                "local tau target decision must match scalar time-offset "
                "diagnostics"
            )

    null_space = _machine_null_space(
        physical_matrix,
        practical.machine_rank,
    )
    (
        covariance,
        ordinary_condition_number,
        covariance_note,
    ) = covariance_from_physical_information(O_X_raw)

    machine_rank_threshold = (
        max(physical_matrix.shape)
        * np.finfo(float).eps
        * practical.sigma_max
    )

    return EffectiveTargetObservabilityResult(
        O_X_raw=O_X_raw,
        O_X_physical=O_X_raw,
        O_X_normalized=normalized.normalized_jacobian,
        machine_rank_O_X=practical.machine_rank,
        effective_rank_O_X=practical.practical_rank,
        machine_rank_threshold=machine_rank_threshold,
        relative_rank_threshold=float(relative_rank_threshold),
        singular_values_O_X=practical.singular_values,
        normalized_singular_values_O_X=(
            practical.normalized_singular_values
        ),
        null_space_O_X=null_space,
        effective_condition_number=(
            practical.practical_condition_number
        ),
        effective_condition_note=(
            "canonical practical condition from unnormalized physical O_X"
        ),
        ordinary_condition_number=ordinary_condition_number,
        covariance_from_physical_information=covariance,
        covariance_note=covariance_note,
        practical_rank_diagnostics=practical,
        physical_information_diagnostics=information,
        scalar_time_offset_diagnostics=tau_diagnostics,
        local_accuracy_diagnostics=accuracy,
        target_labels=list(coordinate_labels),
        nuisance_labels=nuisance_labels,
        original_target_column_norms=practical.column_norms,
        normalized_target_column_norms=_column_norms(diagnostic_matrix),
        zero_target_column_mask=practical.zero_column_mask,
        retained_singular_value_mask=practical.retained_mask,
        matrix_dimensions={
            "O_X_physical": physical_matrix.shape,
            "O_X_normalized": diagnostic_matrix.shape,
        },
        sparse_nnz=sparse_nnz,
    )

##################################################
# Target observability projection
##################################################
def effective_target_observability_dense(
    J: ArrayLike,
    target_columns: ArrayLike,
    nuisance_columns: ArrayLike,
    *,
    normalization: NormalizationMode = "physical_then_column",
    relative_rank_threshold: float = 1e-5,
    parameter_scaling: object | None = None,
    target_labels: list[str] | None = None,
    nuisance_labels: list[str] | None = None,
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
    tau_target_std_seconds: float | None = None,
    variable_name: str = "target",
    lidar_rate_hz: float | None = None,
    coordinate_null_fraction_tolerance: float = 1e-6,
) -> EffectiveTargetObservabilityResult:
    '''Project a dense target block away from nuisance-variable sensitivity.

    Args:
        J: Whitened dense Jacobian.
        target_columns: Target-variable column indices.
        nuisance_columns: Nuisance-variable column indices.
        normalization: Physical scaling and display-normalization mode.
        relative_rank_threshold: Legacy relative singular-value threshold.
        parameter_scaling: Optional physical parameter scaling.
        target_labels: Optional target labels.
        nuisance_labels: Optional nuisance labels.
        practical_rank_policy: Canonical practical-rank policy.
        tau_target_std_seconds: Optional scalar timing accuracy requirement.
        variable_name: Target calibration variable name.
        lidar_rate_hz: Optional LiDAR rate for timing conversion.
        coordinate_null_fraction_tolerance: Coordinate boundedness tolerance.

    Returns:
        Complete dense target observability diagnostics.
    '''

    # Physical scaling, when requested, must happen before target/nuisance
    # extraction because it changes the parameter coordinates being projected.
    prepared = _prepare_projection_matrix(
        J,
        normalization,
        parameter_scaling,
    )
    matrix = _dense(prepared)

    target, nuisance = _check_target_nuisance(
        matrix.shape[1],
        target_columns,
        nuisance_columns,
    )
    J_X = matrix[:, target]
    J_N = matrix[:, nuisance]

    # Remove residual changes reproducible by the nuisance variables.
    if J_N.shape[1] == 0:
        O_X_raw = J_X.copy()
    else:
        P_N = J_N @ np.linalg.pinv(J_N)
        O_X_raw = (
            np.eye(matrix.shape[0])
            - P_N
        ) @ J_X

    resolved_target_labels = (
        target_labels
        or [f"target_{index}" for index in range(target.size)]
    )
    resolved_nuisance_labels = (
        nuisance_labels
        or [f"nuisance_{index}" for index in range(nuisance.size)]
    )

    return _make_target_result(
        O_X_raw,
        normalization,
        relative_rank_threshold,
        practical_rank_policy,
        tau_target_std_seconds,
        resolved_target_labels,
        resolved_nuisance_labels,
        sparse_nnz=None,
        variable_name=variable_name,
        lidar_rate_hz=lidar_rate_hz,
        coordinate_null_fraction_tolerance=(
            coordinate_null_fraction_tolerance
        ),
    )


def effective_target_observability_sparse_lsmr(
    J: sparse.spmatrix,
    target_columns: ArrayLike,
    nuisance_columns: ArrayLike,
    *,
    normalization: NormalizationMode = "physical_then_column",
    relative_rank_threshold: float = 1e-5,
    parameter_scaling: object | None = None,
    target_labels: list[str] | None = None,
    nuisance_labels: list[str] | None = None,
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
    tau_target_std_seconds: float | None = None,
    variable_name: str = "target",
    lidar_rate_hz: float | None = None,
    coordinate_null_fraction_tolerance: float = 1e-6,
) -> EffectiveTargetObservabilityResult:
    '''Project a sparse target block using independent LSMR nuisance solves.

    Args:
        J: Whitened sparse Jacobian.
        target_columns: Target-variable column indices.
        nuisance_columns: Nuisance-variable column indices.
        normalization: Physical scaling and display-normalization mode.
        relative_rank_threshold: Legacy relative singular-value threshold.
        parameter_scaling: Optional physical parameter scaling.
        target_labels: Optional target labels.
        nuisance_labels: Optional nuisance labels.
        practical_rank_policy: Canonical practical-rank policy.
        tau_target_std_seconds: Optional scalar timing accuracy requirement.
        variable_name: Target calibration variable name.
        lidar_rate_hz: Optional LiDAR rate for timing conversion.
        coordinate_null_fraction_tolerance: Coordinate boundedness tolerance.

    Returns:
        Complete sparse target observability diagnostics.

    Raises:
        ValueError: If ``J`` is not sparse.
    '''

    if not sparse.issparse(J):
        raise ValueError("J must be sparse")

    prepared = _prepare_projection_matrix(
        J,
        normalization,
        parameter_scaling,
    )
    matrix = (
        prepared.tocsr()
        if sparse.issparse(prepared)
        else sparse.csr_matrix(prepared)
    )

    target, nuisance = _check_target_nuisance(
        matrix.shape[1],
        target_columns,
        nuisance_columns,
    )
    J_X = matrix[:, target].tocsc()
    J_N = matrix[:, nuisance].tocsr()

    # Solve one least-squares nuisance compensation problem for each target
    # column, avoiding construction of a dense residual-space projector.
    residual_columns: list[sparse.csr_matrix] = []
    for local_column_index in range(J_X.shape[1]):
        c_j = np.asarray(
            J_X[:, local_column_index].toarray()
        ).ravel()

        if J_N.shape[1] == 0:
            q_j = c_j
        else:
            solution = lsmr(
                J_N,
                c_j,
                atol=1e-15,
                btol=1e-15,
                conlim=1e15,
                maxiter=50 * max(J_N.shape),
            )[0]
            q_j = c_j - J_N @ solution

        residual_columns.append(
            sparse.csr_matrix(q_j[:, None])
        )

    if residual_columns:
        O_X_raw = sparse.hstack(
            residual_columns,
            format="csr",
        )
    else:
        O_X_raw = sparse.csr_matrix(
            (matrix.shape[0], 0)
        )

    O_X_raw.eliminate_zeros()

    resolved_target_labels = (
        target_labels
        or [f"target_{index}" for index in range(target.size)]
    )
    resolved_nuisance_labels = (
        nuisance_labels
        or [f"nuisance_{index}" for index in range(nuisance.size)]
    )

    return _make_target_result(
        O_X_raw,
        normalization,
        relative_rank_threshold,
        practical_rank_policy,
        tau_target_std_seconds,
        resolved_target_labels,
        resolved_nuisance_labels,
        sparse_nnz=int(O_X_raw.nnz),
        variable_name=variable_name,
        lidar_rate_hz=lidar_rate_hz,
        coordinate_null_fraction_tolerance=(
            coordinate_null_fraction_tolerance
        ),
    )

##################################################
# Jacobian bundle projection interfaces
##################################################
def _bundle_projection_inputs(
    bundle: JacobianBundle,
    variable_name: str,
    include_trajectory_nuisance: bool,
) -> tuple[
    NDArray[np.float64] | sparse.spmatrix,
    NDArray[np.int64],
    NDArray[np.int64],
    list[str],
    list[str],
]:
    '''Prepare target/nuisance columns for a Jacobian bundle.

    Args:
        bundle: Assembled Jacobian bundle.
        variable_name: Active calibration target.
        include_trajectory_nuisance: Include all trajectory columns when
            projecting the target block.

    Returns:
        Tuple containing the source matrix, target columns, nuisance columns,
        target labels, and nuisance labels.
    '''

    target = extract_variable_columns(
        bundle,
        variable_name,
    )
    nuisance = extract_nuisance_columns(
        bundle,
        variable_name,
    )

    if not include_trajectory_nuisance:
        return (
            bundle.J_C,
            target.column_indices,
            nuisance.column_indices,
            target.labels,
            nuisance.labels,
        )

    # Calibration columns follow trajectory columns in the full Jacobian.
    calibration_offset = bundle.J_T.shape[1]
    target_columns = target.column_indices + calibration_offset
    nuisance_columns = np.r_[
        np.arange(calibration_offset, dtype=np.int64),
        nuisance.column_indices + calibration_offset,
    ]
    nuisance_labels = [
        *bundle.trajectory_column_slices.keys(),
        *nuisance.labels,
    ]

    return (
        bundle.J,
        target_columns,
        nuisance_columns,
        target.labels,
        nuisance_labels,
    )


def effective_target_observability_from_bundle_dense(
    bundle: JacobianBundle,
    variable_name: str,
    *,
    include_trajectory_nuisance: bool = True,
    normalization: NormalizationMode = "physical_then_column",
    relative_rank_threshold: float = 1e-5,
    parameter_scaling: object | None = None,
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
    tau_target_std_seconds: float | None = None,
    lidar_rate_hz: float | None = None,
    coordinate_null_fraction_tolerance: float = 1e-6,
) -> EffectiveTargetObservabilityResult:
    '''Analyze one active variable from a dense Jacobian bundle.

    Args:
        bundle: Assembled Jacobian bundle.
        variable_name: Active calibration variable to analyze.
        include_trajectory_nuisance: Include trajectory columns as nuisance.
        normalization: Physical scaling and display-normalization mode.
        relative_rank_threshold: Legacy relative rank threshold.
        parameter_scaling: Optional physical parameter scaling.
        practical_rank_policy: Canonical practical-rank policy.
        tau_target_std_seconds: Optional scalar timing accuracy requirement.
        lidar_rate_hz: Optional LiDAR rate for timing conversion.
        coordinate_null_fraction_tolerance: Coordinate boundedness tolerance.

    Returns:
        Projected target observability diagnostics.
    '''

    (
        matrix,
        target_columns,
        nuisance_columns,
        target_labels,
        nuisance_labels,
    ) = _bundle_projection_inputs(
        bundle,
        variable_name,
        include_trajectory_nuisance,
    )

    return effective_target_observability_dense(
        matrix,
        target_columns,
        nuisance_columns,
        normalization=normalization,
        relative_rank_threshold=relative_rank_threshold,
        parameter_scaling=parameter_scaling,
        target_labels=target_labels,
        nuisance_labels=nuisance_labels,
        practical_rank_policy=practical_rank_policy,
        tau_target_std_seconds=tau_target_std_seconds,
        variable_name=variable_name,
        lidar_rate_hz=lidar_rate_hz,
        coordinate_null_fraction_tolerance=(
            coordinate_null_fraction_tolerance
        ),
    )


def effective_target_observability_from_bundle_sparse(
    bundle: JacobianBundle,
    variable_name: str,
    *,
    include_trajectory_nuisance: bool = True,
    normalization: NormalizationMode = "physical_then_column",
    relative_rank_threshold: float = 1e-5,
    parameter_scaling: object | None = None,
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
    tau_target_std_seconds: float | None = None,
    lidar_rate_hz: float | None = None,
    coordinate_null_fraction_tolerance: float = 1e-6,
) -> EffectiveTargetObservabilityResult:
    '''Analyze one active variable from a sparse Jacobian bundle.

    Args:
        bundle: Assembled Jacobian bundle.
        variable_name: Active calibration variable to analyze.
        include_trajectory_nuisance: Include trajectory columns as nuisance.
        normalization: Physical scaling and display-normalization mode.
        relative_rank_threshold: Legacy relative rank threshold.
        parameter_scaling: Optional physical parameter scaling.
        practical_rank_policy: Canonical practical-rank policy.
        tau_target_std_seconds: Optional scalar timing accuracy requirement.
        lidar_rate_hz: Optional LiDAR rate for timing conversion.
        coordinate_null_fraction_tolerance: Coordinate boundedness tolerance.

    Returns:
        Projected target observability diagnostics.
    '''

    (
        matrix,
        target_columns,
        nuisance_columns,
        target_labels,
        nuisance_labels,
    ) = _bundle_projection_inputs(
        bundle,
        variable_name,
        include_trajectory_nuisance,
    )

    sparse_matrix = (
        matrix
        if sparse.issparse(matrix)
        else sparse.csr_matrix(matrix)
    )

    return effective_target_observability_sparse_lsmr(
        sparse_matrix,
        target_columns,
        nuisance_columns,
        normalization=normalization,
        relative_rank_threshold=relative_rank_threshold,
        parameter_scaling=parameter_scaling,
        target_labels=target_labels,
        nuisance_labels=nuisance_labels,
        practical_rank_policy=practical_rank_policy,
        tau_target_std_seconds=tau_target_std_seconds,
        variable_name=variable_name,
        lidar_rate_hz=lidar_rate_hz,
        coordinate_null_fraction_tolerance=(
            coordinate_null_fraction_tolerance
        ),
    )

##################################################
# LiDAR motion-only sensitivity
##################################################
def build_lidar_motion_only_matrix_dense(
    body_motions: list[ArrayLike],
    *,
    relative_rank_threshold: float = 1e-5,
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
) -> MotionOnlyObservabilityResult:
    '''Build dense LiDAR motion-only extrinsic diagnostics.

    For each body motion ``A_m``, the function stacks
    ``Adj(A_m) - I_6``. Body poses are treated as known.

    Args:
        body_motions: Relative body transformations.
        relative_rank_threshold: Legacy relative threshold stored in the result.
        practical_rank_policy: Canonical practical-rank policy.

    Returns:
        Dense LiDAR motion-only observability diagnostics.
    '''

    # Every relative motion contributes one 6x6 extrinsic sensitivity block.
    identity_6 = np.eye(6)
    blocks = [
        se3_adjoint(body_motion) - identity_6
        for body_motion in body_motions
    ]
    C_raw = (
        np.vstack(blocks)
        if blocks
        else np.zeros((0, 6), dtype=float)
    )

    # Rank uses the physical matrix; normalization is retained only for display.
    normalized = normalize_jacobian_columns_dense(C_raw)
    ranks = practical_rank_diagnostics(
        C_raw,
        policy=practical_rank_policy,
    )
    null_basis = _machine_null_space(
        C_raw,
        ranks.machine_rank,
    )
    machine_rank_threshold = (
        max(C_raw.shape)
        * np.finfo(float).eps
        * ranks.sigma_max
    )

    return MotionOnlyObservabilityResult(
        C_X_L_raw=C_raw,
        C_X_L_column_normalized=normalized.normalized_jacobian,
        machine_rank=ranks.machine_rank,
        effective_rank=ranks.practical_rank,
        practical_rank=ranks.practical_rank,
        machine_rank_threshold=machine_rank_threshold,
        relative_rank_threshold=float(relative_rank_threshold),
        singular_values=ranks.singular_values,
        normalized_singular_values=ranks.normalized_singular_values,
        null_space_basis=null_basis,
        original_column_norms=normalized.original_column_norms,
        zero_column_mask=normalized.zero_column_mask,
        matrix_dimensions={
            "Adj(A_m)-I_6": (6, 6),
            "C_X_L": C_raw.shape,
        },
        sparse_nnz=None,
        practical_rank_diagnostics=ranks,
    )


def build_lidar_motion_only_matrix_sparse(
    body_motions: list[ArrayLike],
    *,
    relative_rank_threshold: float = 1e-5,
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
) -> MotionOnlyObservabilityResult:
    '''Build sparse LiDAR motion-only diagnostics.

    Args:
        body_motions: Relative body transformations.
        relative_rank_threshold: Legacy relative threshold stored in the result.
        practical_rank_policy: Canonical practical-rank policy.

    Returns:
        Sparse result numerically equivalent to the dense implementation.
    '''

    # Reuse one canonical dense SVD diagnostic because the motion-only matrix
    # has only six columns, then expose sparse matrices for downstream storage.
    dense_result = build_lidar_motion_only_matrix_dense(
        body_motions,
        relative_rank_threshold=relative_rank_threshold,
        practical_rank_policy=practical_rank_policy,
    )

    C_sparse = sparse.csr_matrix(dense_result.C_X_L_raw)
    normalized = normalize_jacobian_columns_sparse(C_sparse)

    return MotionOnlyObservabilityResult(
        C_X_L_raw=C_sparse,
        C_X_L_column_normalized=normalized.normalized_jacobian,
        machine_rank=dense_result.machine_rank,
        effective_rank=dense_result.effective_rank,
        practical_rank=dense_result.practical_rank,
        machine_rank_threshold=dense_result.machine_rank_threshold,
        relative_rank_threshold=dense_result.relative_rank_threshold,
        singular_values=dense_result.singular_values,
        normalized_singular_values=(
            dense_result.normalized_singular_values
        ),
        null_space_basis=dense_result.null_space_basis,
        original_column_norms=normalized.original_column_norms,
        zero_column_mask=normalized.zero_column_mask,
        matrix_dimensions=dense_result.matrix_dimensions,
        sparse_nnz=int(C_sparse.nnz),
        practical_rank_diagnostics=(
            dense_result.practical_rank_diagnostics
        ),
    )

##################################################
# IMU and accelerometer factor sensitivity
##################################################
def _factor_row_indices(
    bundle: JacobianBundle,
    prefixes: tuple[str, ...],
) -> NDArray[np.int64]:
    '''Collect global row indices for residual names with selected prefixes.

    Args:
        bundle: Assembled Jacobian bundle.
        prefixes: Residual-name prefixes to include.

    Returns:
        Ordered row indices matching the insertion order of ``row_slices``.
    '''

    selected_slices = [
        row_slice
        for name, row_slice in bundle.row_slices.items()
        if name.startswith(prefixes)
    ]

    if not selected_slices:
        return np.zeros(0, dtype=int)

    return np.concatenate([
        np.arange(
            row_slice.start or 0,
            row_slice.stop or 0,
        )
        for row_slice in selected_slices
    ])


def _calibration_block_columns(
    bundle: JacobianBundle,
    variable_name: str,
) -> NDArray[np.int64]:
    '''Return local ``J_C`` columns for one active calibration block.

    Args:
        bundle: Assembled Jacobian bundle.
        variable_name: Active calibration variable.

    Returns:
        Ordered local calibration-column indices.
    '''

    block_slice = bundle.calibration_column_slices[variable_name]

    return np.arange(
        block_slice.start or 0,
        block_slice.stop or 0,
        dtype=np.int64,
    )


def _factor_calibration_sensitivity(
    bundle: JacobianBundle,
    prefixes: tuple[str, ...],
    variable_name: str,
) -> NDArray[np.float64] | sparse.csr_matrix:
    '''Extract factor rows and one calibration-variable block from ``J_C``.

    Args:
        bundle: Assembled Jacobian bundle.
        prefixes: Residual-name prefixes to include.
        variable_name: Active calibration variable to extract.

    Returns:
        Dense or CSR factor-sensitivity matrix.
    '''

    rows = _factor_row_indices(
        bundle,
        prefixes,
    )
    columns = _calibration_block_columns(
        bundle,
        variable_name,
    )

    if sparse.issparse(bundle.J_C):
        matrix = bundle.J_C.tocsr()
        return matrix[rows, :][:, columns].tocsr()

    matrix = np.asarray(bundle.J_C, dtype=float)
    return matrix[rows, :][:, columns]


def build_imu_gyro_motion_sensitivity(
    bundle: JacobianBundle,
    *,
    relative_rank_threshold: float = 1e-5,
    translation_zero_tolerance: float = 1e-10,
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
) -> ImuGyroMotionSensitivityResult:
    '''Extract gyro-only ``T_B_I`` sensitivity from assembled IMU rows.

    Args:
        bundle: Assembled Jacobian bundle.
        relative_rank_threshold: Legacy relative threshold.
        translation_zero_tolerance: Tolerance for structural translation columns.
        practical_rank_policy: Canonical practical-rank policy.

    Returns:
        Full and rotation-only gyro sensitivity diagnostics.

    Raises:
        KeyError: If ``T_B_I`` is not active in the bundle.
    '''

    if "T_B_I" not in bundle.calibration_column_slices:
        raise KeyError("T_B_I is not active in this bundle")

    # The legacy threshold remains part of the public API; practical rank is
    # governed by the canonical policy used throughout the package.
    _ = relative_rank_threshold

    full_matrix = _factor_calibration_sensitivity(
        bundle,
        ("imu_",),
        "T_B_I",
    )
    rotation_matrix = _slice_columns(
        full_matrix,
        np.arange(3, dtype=np.int64),
    )
    translation_matrix = _slice_columns(
        full_matrix,
        np.arange(3, 6, dtype=np.int64),
    )
    translation_norms = _column_norms(translation_matrix)

    # Keep full and rotation-only normalized matrices for notebook display.
    if sparse.issparse(full_matrix):
        full_normalized = normalize_jacobian_columns_sparse(full_matrix)
    else:
        full_normalized = normalize_jacobian_columns_dense(full_matrix)

    if sparse.issparse(rotation_matrix):
        rotation_normalized = normalize_jacobian_columns_sparse(
            rotation_matrix
        )
    else:
        rotation_normalized = normalize_jacobian_columns_dense(
            rotation_matrix
        )

    full_ranks = practical_rank_diagnostics(
        full_matrix,
        policy=practical_rank_policy,
    )
    rotation_ranks = practical_rank_diagnostics(
        rotation_matrix,
        policy=practical_rank_policy,
    )

    translation_columns_are_zero = bool(
        np.all(
            translation_norms
            <= translation_zero_tolerance
        )
    )

    return ImuGyroMotionSensitivityResult(
        full_T_B_I_sensitivity_matrix=full_matrix,
        full_T_B_I_normalized=full_normalized.normalized_jacobian,
        rotation_only_sensitivity_matrix=rotation_matrix,
        rotation_only_normalized=(
            rotation_normalized.normalized_jacobian
        ),
        translation_column_norms=translation_norms,
        structural_zero_column_report={
            "translation_column_indices_rotation_first": [3, 4, 5],
            "translation_columns_are_zero": (
                translation_columns_are_zero
            ),
            "tolerance": float(translation_zero_tolerance),
            "interpretation": (
                "gyro-only residuals do not depend on the translation part "
                "of T_B_I"
            ),
        },
        full_machine_rank=full_ranks.machine_rank,
        full_effective_rank=full_ranks.practical_rank,
        full_practical_rank=full_ranks.practical_rank,
        rotation_machine_rank=rotation_ranks.machine_rank,
        rotation_effective_rank=rotation_ranks.practical_rank,
        rotation_practical_rank=rotation_ranks.practical_rank,
        full_singular_values=full_ranks.singular_values,
        full_normalized_singular_values=(
            full_ranks.normalized_singular_values
        ),
        rotation_singular_values=rotation_ranks.singular_values,
        rotation_normalized_singular_values=(
            rotation_ranks.normalized_singular_values
        ),
        original_column_norms=full_normalized.original_column_norms,
        zero_column_mask=full_normalized.zero_column_mask,
    )

def build_accelerometer_motion_sensitivity(
    bundle: JacobianBundle,
    *,
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
) -> AccelerometerSensitivityResult:
    '''Extract accelerometer-only ``T_B_I`` factor sensitivity.

    Args:
        bundle: Assembled Jacobian bundle.
        practical_rank_policy: Canonical practical-rank policy.

    Returns:
        Accelerometer factor-sensitivity diagnostics before nuisance projection.

    Raises:
        KeyError: If ``T_B_I`` is not active in the bundle.
    '''

    if "T_B_I" not in bundle.calibration_column_slices:
        raise KeyError("T_B_I is not active in this bundle")

    raw = _factor_calibration_sensitivity(
        bundle,
        ("accel_simple_", "accel_complex_"),
        "T_B_I",
    )

    # Normalize only for display and retain sparse storage when available.
    if sparse.issparse(raw):
        normalized = normalize_jacobian_columns_sparse(raw)
        dense_raw = raw.toarray()
        sparse_nnz = int(raw.nnz)
    else:
        normalized = normalize_jacobian_columns_dense(raw)
        dense_raw = np.asarray(raw, dtype=float)
        sparse_nnz = None

    rank = practical_rank_diagnostics(
        dense_raw,
        policy=practical_rank_policy,
    )

    return AccelerometerSensitivityResult(
        C_X_I_accel_raw=raw,
        C_X_I_accel_whitened=raw,
        C_X_I_accel_physical=raw,
        C_X_I_accel_column_normalized=(
            normalized.normalized_jacobian
        ),
        machine_rank=rank.machine_rank,
        practical_rank=rank.practical_rank,
        singular_values=rank.singular_values,
        normalized_singular_values=(
            rank.normalized_singular_values
        ),
        original_column_norms=rank.column_norms,
        zero_column_mask=rank.zero_column_mask,
        practical_rank_diagnostics=rank,
        matrix_dimensions={
            "C_X_I_accel": dense_raw.shape,
        },
        sparse_nnz=sparse_nnz,
    )

##################################################
# Sliding-window diagnostics
##################################################
def sliding_window_factor_diagnostics(
    dataset: object,
    pose_provider: object,
    *,
    window_duration: float,
    step: float,
    relative_rank_threshold: float = 1e-5,
    normalization: NormalizationMode = "physical_then_column",
    use_sparse: bool = False,
    jacobian_options: JacobianOptions | None = None,
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
    fixed_extrinsic: str = "none",
    tau_target_std_seconds: float | None = None,
) -> list[SlidingWindowFactorDiagnostics]:
    '''Run factor-specific diagnostics over sliding time windows.

    Args:
        dataset: Dataset exposing ``start_time``, ``end_time``, and
            ``window_jacobians``.
        pose_provider: Trajectory pose provider forwarded to the dataset.
        window_duration: Window duration in seconds.
        step: Window-start increment in seconds.
        relative_rank_threshold: Legacy relative threshold.
        normalization: Physical scaling and display-normalization mode.
        use_sparse: Use sparse target projection when ``True``.
        jacobian_options: Optional Jacobian calculation options.
        practical_rank_policy: Canonical practical-rank policy.
        fixed_extrinsic: Fixed-extrinsic configuration forwarded to the dataset.
        tau_target_std_seconds: Optional scalar timing accuracy requirement.

    Returns:
        Ordered diagnostics for every complete window.

    Raises:
        ValueError: If ``window_duration`` or ``step`` is not positive.
    '''

    if window_duration <= 0.0 or step <= 0.0:
        raise ValueError("window_duration and step must be positive")

    results: list[SlidingWindowFactorDiagnostics] = []
    current_start = float(dataset.start_time)
    dataset_end = float(dataset.end_time)

    # Process only complete windows. The small epsilon protects the final
    # boundary against floating-point accumulation in repeated step additions.
    while current_start + window_duration <= dataset_end + 1e-12:
        current_end = min(
            current_start + window_duration,
            dataset_end,
        )

        bundle, body_motions, counts = dataset.window_jacobians(
            current_start,
            current_end,
            pose_provider,
            use_sparse=use_sparse,
            jacobian_options=jacobian_options,
            fixed_extrinsic=fixed_extrinsic,
        )

        # Motion-only summaries are included only when the corresponding factor
        # family contributes rows to the current window.
        if counts.get("lidar", 0):
            lidar_motion = build_lidar_motion_only_matrix_dense(
                body_motions,
                relative_rank_threshold=relative_rank_threshold,
                practical_rank_policy=practical_rank_policy,
            )
        else:
            lidar_motion = None

        if (
            counts.get("imu", 0)
            and "T_B_I" in bundle.calibration_column_slices
        ):
            imu_motion = build_imu_gyro_motion_sensitivity(
                bundle,
                relative_rank_threshold=relative_rank_threshold,
                practical_rank_policy=practical_rank_policy,
            )
        else:
            imu_motion = None

        # Jointly project every active supported calibration variable against
        # all remaining calibration and trajectory nuisance directions.
        target_results: dict[
            str,
            EffectiveTargetObservabilityResult,
        ] = {}

        for variable_name in SUPPORTED_CALIBRATION_VARIABLES:
            if variable_name not in bundle.calibration_column_slices:
                continue

            if use_sparse:
                target_result = (
                    effective_target_observability_from_bundle_sparse(
                        bundle,
                        variable_name,
                        normalization=normalization,
                        relative_rank_threshold=relative_rank_threshold,
                        practical_rank_policy=practical_rank_policy,
                        tau_target_std_seconds=tau_target_std_seconds,
                    )
                )
            else:
                target_result = (
                    effective_target_observability_from_bundle_dense(
                        bundle,
                        variable_name,
                        normalization=normalization,
                        relative_rank_threshold=relative_rank_threshold,
                        practical_rank_policy=practical_rank_policy,
                        tau_target_std_seconds=tau_target_std_seconds,
                    )
                )

            target_results[variable_name] = target_result

        results.append(
            SlidingWindowFactorDiagnostics(
                current_start,
                current_end,
                counts,
                lidar_motion,
                imu_motion,
                target_results,
            )
        )
        current_start += step

    return results