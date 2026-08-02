'''Diagnostics for calibration observability matrices.

The module separates structural observability, practical numerical rank,
physical information, and local CRLB-like accuracy diagnostics. Physical
information is always calculated from whitened, unnormalized matrices; column
normalization is reserved for display and relative-direction inspection.
'''

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.linalg import solve

from .linalg import null_space_dense, numerical_rank_dense, singular_value_diagnostics_dense
from .observability import (
    build_motion_only_matrix_dense,
    effective_observability_dense,
    effective_observability_sparse_lsmr,
    reduced_information_dense,
)
from .types import DEFAULT_PRACTICAL_RANK_POLICY, PracticalRankPolicy


##################################################
# Matrix-level observability diagnostics
##################################################
@dataclass(frozen=True)
class MatrixAnalysis:
    '''Store rank, singular-value, null-space, and covariance diagnostics.

    Attributes:
        name (str): Human-readable name of the analyzed matrix.
        shape (tuple[int, int]): Matrix dimensions as ``(rows, columns)``.
        rank (int): Numerical rank under the selected tolerance.
        singular_values (NDArray[np.float64]): Singular values in descending order.
        tolerance (float): Absolute threshold used for the numerical-rank decision.
        condition_number (float): Condition number reported by the SVD diagnostics.
        null_space_basis (NDArray[np.float64]): Basis vectors spanning the numerical
            right null space.
        is_full_joint_calibration_rank (bool): Whether all matrix columns are
            numerically independent.
        covariance_C (NDArray[np.float64] | None): Conventional local calibration
            covariance when the reduced information matrix is full rank.
        observable_subspace_pseudocovariance (NDArray[np.float64] | None):
            Pseudocovariance restricted to the observable subspace when rank
            deficient.
        notes (tuple[str, ...]): Explanatory notes attached to the result.
    '''

    name: str
    shape: tuple[int, int]
    rank: int
    singular_values: NDArray[np.float64]
    tolerance: float
    condition_number: float
    null_space_basis: NDArray[np.float64]
    is_full_joint_calibration_rank: bool
    covariance_C: NDArray[np.float64] | None
    observable_subspace_pseudocovariance: NDArray[np.float64] | None
    notes: tuple[str, ...]


def _analyze_matrix(name: str, M: NDArray[np.float64], tolerance: float | None = None) -> MatrixAnalysis:
    '''Calculate common dense-matrix observability diagnostics.

    Args:
        name (str): Human-readable matrix name stored in the result.
        M (NDArray[np.float64]): Dense matrix to analyze.
        tolerance (float | None): Optional absolute singular-value tolerance.

    Returns:
        MatrixAnalysis: Rank, singular-value, condition-number, and null-space
        diagnostics. Covariance fields are left empty.
    '''

    # Calculate singular-value rank diagnostics and the matching right null space.
    diag = singular_value_diagnostics_dense(M, tolerance)
    null_basis = null_space_dense(M, diag.tolerance)

    # Package the common matrix result without covariance interpretation.
    return MatrixAnalysis(
        name=name,
        shape=M.shape,
        rank=diag.rank,
        singular_values=diag.singular_values,
        tolerance=diag.tolerance,
        condition_number=diag.condition_number,
        null_space_basis=null_basis,
        is_full_joint_calibration_rank=diag.rank == M.shape[1],
        covariance_C=None,
        observable_subspace_pseudocovariance=None,
        notes=(),
    )


def analyze_fixed_trajectory_calibration(
    J_C: ArrayLike, tolerance: float | None = None
) -> MatrixAnalysis:
    '''Analyze calibration sensitivity with a fixed trajectory.

    Args:
        J_C (ArrayLike): Calibration Jacobian with shape ``(m, n_C)``.
        tolerance (float | None): Optional absolute singular-value tolerance.

    Returns:
        MatrixAnalysis: Diagnostics for ``rank(J_C)``.
    '''

    # Fixed-trajectory observability depends only on the calibration columns.
    M = np.asarray(J_C, dtype=float)
    analysis = _analyze_matrix("fixed_trajectory_calibration_J_C", M, tolerance)

    return analysis


def analyze_motion_only_extrinsic(
    A_list: list[ArrayLike], tolerance: float | None = None
) -> MatrixAnalysis:
    '''Analyze motion-only extrinsic sensitivity.

    This diagnostic evaluates ``rank(C_X)`` while treating body motions as fixed
    and known. It does not establish joint calibration observability when trajectory,
    bias, or timing variables can move.

    Args:
        A_list (list[ArrayLike]): Relative-motion matrices used to construct
            ``C_X``.
        tolerance (float | None): Optional absolute singular-value tolerance.

    Returns:
        MatrixAnalysis: Motion-only extrinsic diagnostics with interpretation notes.
    '''

    # Stack the relative-motion constraints before applying common diagnostics.
    C_X = build_motion_only_matrix_dense(A_list)
    analysis = _analyze_matrix("motion_only_C_X", C_X, tolerance)

    # Attach the interpretation boundary specific to the motion-only test.
    return replace(
        analysis,
        notes=(
            "rank(C_X) assumes fixed known body motions.",
            "It does not prove joint observability when poses, bias, and timing can also move.",
        ),
    )


def analyze_joint_calibration_dense(
    J_T: ArrayLike, J_C: ArrayLike, tolerance: float | None = None
) -> MatrixAnalysis:
    '''Analyze joint calibration observability with a dense projection.

    The effective matrix is ``O_C = P_T_perp @ J_C``, which removes calibration
    directions explainable by trajectory perturbations.

    Args:
        J_T (ArrayLike): Dense trajectory Jacobian with shape ``(m, n_T)``.
        J_C (ArrayLike): Dense calibration Jacobian with shape ``(m, n_C)``.
        tolerance (float | None): Optional absolute singular-value tolerance.

    Returns:
        MatrixAnalysis: Effective calibration rank and reduced-information
        covariance diagnostics.
    '''

    # Project calibration columns away from the trajectory column space.
    O_C = effective_observability_dense(J_T, J_C)
    S_C = reduced_information_dense(O_C)

    # Analyze the projected matrix and interpret its reduced information.
    analysis = _analyze_matrix("joint_calibration_O_C", O_C, tolerance)
    cov, pseudocov, notes = _covariance_from_reduced_info(
        S_C,
        analysis.rank,
        tolerance,
    )

    return replace(
        analysis,
        covariance_C=cov,
        observable_subspace_pseudocovariance=pseudocov,
        notes=notes,
    )


def analyze_joint_calibration_sparse(
    J_T: sparse.spmatrix, J_C: sparse.spmatrix, tolerance: float | None = None
) -> MatrixAnalysis:
    '''Analyze joint calibration observability with sparse LSMR projection.

    Args:
        J_T (sparse.spmatrix): Sparse trajectory Jacobian.
        J_C (sparse.spmatrix): Sparse calibration Jacobian.
        tolerance (float | None): Optional absolute singular-value tolerance.

    Returns:
        MatrixAnalysis: Effective calibration diagnostics together with sparse
        projection density information.
    '''

    # Compute the sparse trajectory projection, then densify only the reduced
    # calibration matrix needed by the current dense SVD diagnostics.
    proj = effective_observability_sparse_lsmr(J_T, J_C)
    O_dense = proj.O_C.toarray()

    # Reuse the same rank and covariance interpretation as the dense path.
    analysis = _analyze_matrix(
        "joint_calibration_O_C_sparse_lsmr",
        O_dense,
        tolerance,
    )
    cov, pseudocov, notes = _covariance_from_reduced_info(
        proj.S_C,
        analysis.rank,
        tolerance,
    )

    return replace(
        analysis,
        covariance_C=cov,
        observable_subspace_pseudocovariance=pseudocov,
        notes=notes + (
            f"sparse projection nnz={proj.nnz}, density={proj.density:.3g}",
        ),
    )


def analyze_full_system(J: ArrayLike | sparse.spmatrix, tolerance: float | None = None) -> MatrixAnalysis:
    '''Analyze numerical rank of the complete system Jacobian.

    Args:
        J (ArrayLike | sparse.spmatrix): Dense or sparse full-system Jacobian.
        tolerance (float | None): Optional absolute singular-value tolerance.

    Returns:
        MatrixAnalysis: Diagnostics for ``rank(J)``.
    '''

    # Use the common dense analyzer after converting only when necessary.
    M = J.toarray() if sparse.issparse(J) else np.asarray(J, dtype=float)

    return _analyze_matrix("full_system_J", M, tolerance)


def _covariance_from_reduced_info(
    S_C: NDArray[np.float64], rank: int, tolerance: float | None
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, tuple[str, ...]]:
    '''Build covariance diagnostics from reduced calibration information.

    Args:
        S_C (NDArray[np.float64]): Reduced information matrix with shape
            ``(n_C, n_C)``.
        rank (int): Numerical rank of the corresponding effective observability
            matrix.
        tolerance (float | None): Cutoff passed to ``numpy.linalg.pinv`` when the
            system is rank deficient.

    Returns:
        tuple[NDArray[np.float64] | None, NDArray[np.float64] | None,
        tuple[str, ...]]: Conventional covariance, observable-subspace
        pseudocovariance, and explanatory notes.
    '''

    n = S_C.shape[0]

    # A full-rank positive information matrix supports a conventional covariance.
    if rank == n:
        cov = solve(S_C, np.eye(n), assume_a="pos")
        return (
            cov,
            None,
            ("S_C is full rank; covariance_C is a conventional local covariance.",),
        )

    # A rank-deficient information matrix supports only an observable-subspace
    # pseudocovariance; rejected directions remain unbounded.
    pseudocov = np.linalg.pinv(S_C, rcond=tolerance)
    return (
        None,
        pseudocov,
        (
            "S_C is rank deficient; covariance_C is None.",
            (
                "The pseudoinverse is only an observable-subspace "
                "pseudocovariance, not finite covariance in null directions."
            ),
        ),
    )


##################################################
# Column and practical-rank result structures
##################################################
@dataclass(frozen=True)
class ColumnRejectionNormalizationResult:
    '''Store near-zero column rejection and diagnostic normalization results.

    The input matrix is assumed to be whitened and expressed in physical units.
    Rejected columns remain exactly zero instead of being divided by an artificial
    norm. The normalized matrix is intended only for display and relative-direction
    diagnostics, never for physical information calculations.

    Attributes:
        normalized_matrix (NDArray[np.float64] | sparse.csr_matrix): Matrix after
            rejecting negligible columns and normalizing active columns.
        original_column_norms (NDArray[np.float64]): Original Euclidean column norms.
        zero_threshold (float): Threshold used to classify negligible columns.
        zero_column_mask (NDArray[np.bool_]): Mask of rejected columns.
        active_column_mask (NDArray[np.bool_]): Mask of retained columns.
    '''

    normalized_matrix: NDArray[np.float64] | sparse.csr_matrix
    original_column_norms: NDArray[np.float64]
    zero_threshold: float
    zero_column_mask: NDArray[np.bool_]
    active_column_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class PracticalRankDiagnostics:
    '''Store the three-stage practical-rank decision.

    Stage 1 rejects negligible whole columns, Stage 2 applies an absolute matrix
    gate, and Stage 3 retains singular values above the larger of the absolute and
    relative singular thresholds.

    Attributes:
        matrix_shape (tuple[int, int]): Shape of the physical projected matrix.
        column_norms (NDArray[np.float64]): Original physical column norms.
        column_threshold (float): Whole-column rejection threshold.
        zero_column_mask (NDArray[np.bool_]): Rejected-column mask.
        active_column_mask (NDArray[np.bool_]): Retained-column mask.
        filtered_matrix (NDArray[np.float64]): Physical matrix with rejected columns
            set exactly to zero.
        singular_values (NDArray[np.float64]): Singular values of the filtered matrix.
        normalized_singular_values (NDArray[np.float64]): Singular values divided by
            ``sigma_max``.
        sigma_max (float): Largest singular value.
        matrix_absolute_threshold (float): Absolute gate applied to ``sigma_max``.
        matrix_passed_absolute_gate (bool): Whether the matrix contains sufficient
            absolute sensitivity.
        singular_threshold (float): Final singular-value retention threshold.
        retained_mask (NDArray[np.bool_]): Retained singular-direction mask.
        machine_rank (int): Floating-point machine rank.
        relative_only_rank (int): Rank under only the relative singular threshold.
        practical_rank (int): Canonical practical rank.
        maximum_possible_rank (int): Minimum matrix dimension.
        sigma_min_retained (float): Smallest retained singular value.
        practical_condition_number (float): Condition number over retained modes.
        worst_retained_std_bound (float): Reciprocal of ``sigma_min_retained``.
    '''

    matrix_shape: tuple[int, int]
    column_norms: NDArray[np.float64]
    column_threshold: float
    zero_column_mask: NDArray[np.bool_]
    active_column_mask: NDArray[np.bool_]
    filtered_matrix: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    normalized_singular_values: NDArray[np.float64]
    sigma_max: float
    matrix_absolute_threshold: float
    matrix_passed_absolute_gate: bool
    singular_threshold: float
    retained_mask: NDArray[np.bool_]
    machine_rank: int
    relative_only_rank: int
    practical_rank: int
    maximum_possible_rank: int
    sigma_min_retained: float
    practical_condition_number: float
    worst_retained_std_bound: float


@dataclass(frozen=True)
class CommonRankDiagnostics:
    '''Store backward-compatible rank diagnostic aliases.

    Attributes:
        machine_rank (int): Floating-point machine rank.
        effective_rank (int): Canonical practical rank.
        maximum_possible_rank (int): Minimum matrix dimension.
        machine_threshold (float): Machine-precision singular-value threshold.
        relative_rank_threshold (float): Requested relative threshold.
        singular_values (NDArray[np.float64]): Singular values of the filtered matrix.
        normalized_singular_values (NDArray[np.float64]): Singular values normalized
            by the largest value.
        retained_mask (NDArray[np.bool_]): Canonically retained singular directions.
    '''

    machine_rank: int
    effective_rank: int
    maximum_possible_rank: int
    machine_threshold: float
    relative_rank_threshold: float
    singular_values: NDArray[np.float64]
    normalized_singular_values: NDArray[np.float64]
    retained_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class PhysicalInformationDiagnostics:
    '''Store information diagnostics in unnormalized physical coordinates.

    ``S_X = O_physical.T @ O_physical`` uses ``D = I``. Rotations remain radians,
    translations metres, gyro bias radians per second, and time offsets seconds.
    Consequently, mixed SE(3) singular directions should not be interpreted as if
    all columns shared one unit.

    Attributes:
        information_matrix (NDArray[np.float64]): Physical information matrix.
        information_eigenvalues (NDArray[np.float64]): Squared physical singular
            values.
        trace_information (float): Trace of the physical information matrix.
        frobenius_norm_O (float): Frobenius norm of the physical projected matrix.
        sigma_max_physical (float): Largest physical singular value.
        all_physical_singular_values (NDArray[np.float64]): All singular values of
            the original physical matrix.
        observable_subspace_std_bounds (NDArray[np.float64]): Reciprocal retained
            singular values.
        worst_retained_std_bound (float): Largest retained-mode bound.
        full_covariance (NDArray[np.float64] | None): Conventional covariance for a
            full-column-rank target.
        observable_subspace_pseudocovariance (NDArray[np.float64] | None):
            Pseudocovariance for a rank-deficient target.
        notes (tuple[str, ...]): Interpretation notes.
        singular_values (NDArray[np.float64]): Compatibility alias.
        total_information (float): Compatibility alias for trace information.
        sigma_max (float): Compatibility alias for the largest singular value.
        sigma_min_retained (float): Smallest retained singular value.
        effective_condition_number (float): Condition number over retained modes.
        standard_deviation_bounds (NDArray[np.float64]): Compatibility alias.
        maximum_retained_standard_deviation_bound (float): Compatibility alias.
    '''

    information_matrix: NDArray[np.float64]
    information_eigenvalues: NDArray[np.float64]
    trace_information: float
    frobenius_norm_O: float
    sigma_max_physical: float
    all_physical_singular_values: NDArray[np.float64]
    observable_subspace_std_bounds: NDArray[np.float64]
    worst_retained_std_bound: float
    full_covariance: NDArray[np.float64] | None
    observable_subspace_pseudocovariance: NDArray[np.float64] | None
    notes: tuple[str, ...]
    # Compatibility aliases used by earlier notebooks/tests.
    singular_values: NDArray[np.float64]
    total_information: float
    sigma_max: float
    sigma_min_retained: float
    effective_condition_number: float
    standard_deviation_bounds: NDArray[np.float64]
    maximum_retained_standard_deviation_bound: float


@dataclass(frozen=True)
class ScalarTimeOffsetDiagnostics:
    '''Store scalar temporal-offset sensitivity diagnostics.

    Attributes:
        sensitivity_tau (float): Euclidean norm of the physical timing column.
        information_tau (float): Scalar information ``sensitivity_tau**2``.
        local_std_bound_tau_seconds (float): Local CRLB-like standard-deviation
            bound in seconds.
        target_std_seconds (float): Requested timing standard-deviation target.
        meets_target (bool): Whether the finite local bound meets the target.
        matrix_passed_absolute_gate (bool): Whether timing sensitivity passes the
            absolute matrix gate.
    '''

    sensitivity_tau: float
    information_tau: float
    local_std_bound_tau_seconds: float
    target_std_seconds: float
    meets_target: bool
    matrix_passed_absolute_gate: bool


##################################################
# Calibration coordinate metadata
##################################################
COORDINATE_NULL_FRACTION_TOLERANCE = 1e-6

VARIABLE_COORDINATE_LABELS: dict[str, tuple[str, ...]] = {
    "T_B_I": ("rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"),
    "T_B_L": ("rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"),
    "b_g": ("b_gx", "b_gy", "b_gz"),
    "tau_I": ("tau_I",),
    "tau_L": ("tau_L",),
}
VARIABLE_COORDINATE_UNITS: dict[str, tuple[str, ...]] = {
    "T_B_I": ("rad", "rad", "rad", "m", "m", "m"),
    "T_B_L": ("rad", "rad", "rad", "m", "m", "m"),
    "b_g": ("rad/s", "rad/s", "rad/s"),
    "tau_I": ("s",),
    "tau_L": ("s",),
}


##################################################
# Local accuracy result structure
##################################################
@dataclass(frozen=True)
class LocalAccuracyDiagnostics:
    '''Store local CRLB-like diagnostics for one calibration target.

    The projected matrix must be whitened, unnormalized, and expressed in physical
    ``D = I`` coordinates. Full-rank targets receive a conventional covariance.
    Rank-deficient targets receive only an observable-subspace pseudocovariance,
    while coordinates with non-negligible null-space projection are marked
    unbounded.

    Attributes:
        variable_name (str): Calibration variable represented by the matrix columns.
        matrix_shape (tuple[int, int]): Shape of the physical projected matrix.
        coordinate_labels (tuple[str, ...]): Human-readable coordinate names.
        coordinate_units (tuple[str, ...]): Physical coordinate units.
        practical_rank (int): Canonical retained rank.
        maximum_rank (int): Number of target coordinates.
        nullity (int): Number of rejected or unobservable directions.
        physical_information_matrix (NDArray[np.float64]): ``O_X.T @ O_X``.
        physical_singular_values (NDArray[np.float64]): Singular values of the
            filtered physical matrix.
        retained_mask (NDArray[np.bool_]): Retained singular directions.
        retained_mode_std_bounds (NDArray[np.float64]): Reciprocal retained singular
            values.
        retained_mode_directions (NDArray[np.float64]): Right singular vectors of
            retained modes.
        worst_retained_mode_std_bound (float): Largest retained-mode bound.
        retained_mode_rotation_component_norms (NDArray[np.float64]): Rotation-part
            norms for retained SE(3) directions.
        retained_mode_translation_component_norms (NDArray[np.float64]):
            Translation-part norms for retained SE(3) directions.
        retained_mode_kinds (tuple[str, ...]): Rotation, translation, mixed, or
            native labels for retained modes.
        full_covariance (NDArray[np.float64] | None): Full local covariance when all
            coordinates are retained.
        observable_subspace_pseudocovariance (NDArray[np.float64] | None):
            Pseudocovariance restricted to retained modes.
        observable_projector (NDArray[np.float64]): Projector onto retained modes.
        null_projector (NDArray[np.float64]): Complementary null-space projector.
        coordinate_observable_fraction (NDArray[np.float64]): Diagonal observable
            projector entries.
        coordinate_null_fraction (NDArray[np.float64]): Diagonal null-projector
            entries.
        coordinate_is_fully_bounded (NDArray[np.bool_]): Coordinates with negligible
            null-space projection.
        coordinate_std_bounds (NDArray[np.float64]): Finite or infinite coordinate
            bounds.
        scalar_std_bound (float | None): Scalar target bound when ``n_X == 1``.
        scalar_std_bound_lidar_frames (float | None): Scalar bound converted to
            LiDAR frames.
        target_std_bound (float | None): Optional requested scalar bound.
        target_ratio (float | None): Scalar bound divided by the requested target.
        meets_target (bool | None): Optional scalar target decision.
        covariance_kind (Literal): Full, observable-subspace-only, or unobservable.
        notes (tuple[str, ...]): Interpretation notes.
    '''

    variable_name: str
    matrix_shape: tuple[int, int]
    coordinate_labels: tuple[str, ...]
    coordinate_units: tuple[str, ...]
    practical_rank: int
    maximum_rank: int
    nullity: int
    physical_information_matrix: NDArray[np.float64]
    physical_singular_values: NDArray[np.float64]
    retained_mask: NDArray[np.bool_]
    retained_mode_std_bounds: NDArray[np.float64]
    retained_mode_directions: NDArray[np.float64]
    worst_retained_mode_std_bound: float
    retained_mode_rotation_component_norms: NDArray[np.float64]
    retained_mode_translation_component_norms: NDArray[np.float64]
    retained_mode_kinds: tuple[str, ...]
    full_covariance: NDArray[np.float64] | None
    observable_subspace_pseudocovariance: NDArray[np.float64] | None
    observable_projector: NDArray[np.float64]
    null_projector: NDArray[np.float64]
    coordinate_observable_fraction: NDArray[np.float64]
    coordinate_null_fraction: NDArray[np.float64]
    coordinate_is_fully_bounded: NDArray[np.bool_]
    coordinate_std_bounds: NDArray[np.float64]
    scalar_std_bound: float | None
    scalar_std_bound_lidar_frames: float | None
    target_std_bound: float | None
    target_ratio: float | None
    meets_target: bool | None
    covariance_kind: Literal["full", "observable_subspace_only", "unobservable"]
    notes: tuple[str, ...]


##################################################
# Local accuracy and retained-mode diagnostics
##################################################
def coordinate_metadata_for_variable(variable_name: str, dimension: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    '''Return coordinate labels and units for one calibration variable.

    Known variables use radians for rotation, metres for translation, radians per
    second for gyro bias, and seconds for temporal offsets. Unknown variables fall
    back to indexed native-coordinate labels.

    Args:
        variable_name (str): Calibration variable name.
        dimension (int): Number of variable coordinates.

    Returns:
        tuple[tuple[str, ...], tuple[str, ...]]: Coordinate labels and units, each
        with length ``dimension``.
    '''

    # Use the central physical metadata when the requested dimension matches.
    labels = VARIABLE_COORDINATE_LABELS.get(variable_name)
    units = VARIABLE_COORDINATE_UNITS.get(variable_name)
    if (
        labels is not None
        and units is not None
        and len(labels) == dimension
        and len(units) == dimension
    ):
        return labels, units

    # Unknown or dimension-mismatched variables retain explicit native labels.
    fallback_labels = tuple(
        f"{variable_name}_{index}"
        for index in range(dimension)
    )
    fallback_units = tuple("native" for _ in range(dimension))

    return fallback_labels, fallback_units


def local_accuracy_diagnostics(
    O_X_physical: ArrayLike | sparse.spmatrix,
    *,
    variable_name: str,
    coordinate_labels: tuple[str, ...] | list[str],
    coordinate_units: tuple[str, ...] | list[str],
    practical_rank_result: PracticalRankDiagnostics,
    lidar_rate_hz: float | None = None,
    target_std_seconds: float | None = None,
    coordinate_null_fraction_tolerance: float = COORDINATE_NULL_FRACTION_TOLERANCE,
) -> LocalAccuracyDiagnostics:
    '''Compute local CRLB-like bounds from a physical projected matrix.

    The input must be the unnormalized, whitened matrix used for
    ``S_X = O_X.T @ O_X``. The retained singular mask is taken directly from
    ``practical_rank_result`` so rank, condition, information, and accuracy
    diagnostics refer to the same observable subspace.

    Args:
        O_X_physical (ArrayLike | sparse.spmatrix): Physical projected matrix with
            shape ``(m, n_X)``.
        variable_name (str): Calibration variable represented by the columns.
        coordinate_labels (tuple[str, ...] | list[str]): Coordinate labels with
            length ``n_X``.
        coordinate_units (tuple[str, ...] | list[str]): Coordinate units with
            length ``n_X``.
        practical_rank_result (PracticalRankDiagnostics): Canonical rank result for
            the same matrix.
        lidar_rate_hz (float | None): Optional LiDAR frequency used to express a
            scalar time bound in frames.
        target_std_seconds (float | None): Optional scalar timing target in seconds.
        coordinate_null_fraction_tolerance (float): Maximum null-space fraction for
            treating a coordinate as fully bounded.

    Returns:
        LocalAccuracyDiagnostics: Retained-mode, covariance, projector, coordinate,
        and optional scalar timing diagnostics.

    Raises:
        ValueError: If metadata, matrix shape, or tolerance is invalid.
        AssertionError: If SVD values or retained-bound identities disagree with the
            supplied practical-rank result.
    '''

    matrix = _as_dense_matrix(O_X_physical)
    labels = tuple(coordinate_labels)
    units = tuple(coordinate_units)
    if len(labels) != matrix.shape[1] or len(units) != matrix.shape[1]:
        raise ValueError("coordinate labels and units must match O_X_physical columns")
    if practical_rank_result.matrix_shape != tuple(matrix.shape):
        raise ValueError("practical_rank_result shape must match O_X_physical")
    tolerance = float(coordinate_null_fraction_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("coordinate_null_fraction_tolerance must be finite and nonnegative")

    # The practical-rank policy may zero negligible whole columns before SVD.
    # Use that same filtered physical matrix, never a normalized display matrix,
    # so retained masks and singular directions are canonical and shared.
    filtered_matrix = np.asarray(practical_rank_result.filtered_matrix, dtype=float)
    U, singular_values, Vt = np.linalg.svd(filtered_matrix, full_matrices=True)
    _ = U
    if not np.allclose(singular_values, practical_rank_result.singular_values, atol=1e-10, rtol=1e-8):
        raise AssertionError("local accuracy singular values must match practical-rank diagnostics")
    retained_mask = np.asarray(practical_rank_result.retained_mask, dtype=bool).copy()
    n_parameters = matrix.shape[1]
    practical_rank = int(practical_rank_result.practical_rank)
    retained_singular_values = singular_values[retained_mask]
    V = Vt.T
    # V_retained: (n_X, r). Columns span only the practically retained
    # observable subspace. Rejected singular directions are treated as
    # unbounded rather than assigned a large but finite covariance.
    V_retained = V[:, retained_mask] if retained_mask.size else np.zeros((n_parameters, 0), dtype=float)

    information_matrix = matrix.T @ matrix
    information_matrix = 0.5 * (information_matrix + information_matrix.T)
    identity = np.eye(n_parameters, dtype=float)
    observable_projector = np.zeros((n_parameters, n_parameters), dtype=float)
    null_projector = identity.copy()
    full_covariance = None
    observable_pseudocovariance = None
    covariance_kind: Literal["full", "observable_subspace_only", "unobservable"] = "unobservable"
    notes = [
        "local CRLB-like standard-deviation bound from unnormalized whitened physical O_X",
        "D=I physical coordinates; no requirement-based parameter scaling is applied",
    ]

    if practical_rank > 0:
        inverse_variances = 1.0 / np.square(retained_singular_values)
        Sigma_observable = (V_retained * inverse_variances) @ V_retained.T
        Sigma_observable = 0.5 * (Sigma_observable + Sigma_observable.T)
        observable_projector = V_retained @ V_retained.T
        observable_projector = 0.5 * (observable_projector + observable_projector.T)
        null_projector = identity - observable_projector
        null_projector = 0.5 * (null_projector + null_projector.T)
        if practical_rank == n_parameters:
            full_covariance = Sigma_observable
            covariance_kind = "full"
            notes.append("full_covariance is conventional because every coordinate direction is retained")
        else:
            observable_pseudocovariance = Sigma_observable
            covariance_kind = "observable_subspace_only"
            notes.append(
                "rank deficient target; pseudocovariance is only valid "
                "inside the retained observable subspace"
            )
    else:
        Sigma_observable = None
        notes.append("rank-zero target; all coordinate directions are unbounded")

    observable_fraction = np.clip(np.diag(observable_projector), 0.0, 1.0)
    null_fraction = np.clip(np.diag(null_projector), 0.0, 1.0)
    coordinate_is_bounded = null_fraction <= tolerance
    coordinate_std_bounds = np.full(n_parameters, float("inf"), dtype=float)
    if practical_rank > 0 and Sigma_observable is not None:
        # A finite pseudocovariance diagonal is not sufficient to claim that a
        # coordinate is bounded. The coordinate must have negligible projection
        # onto the rejected/null subspace.
        diagonal_variances = np.maximum(np.diag(Sigma_observable), 0.0)
        coordinate_std_bounds[coordinate_is_bounded] = np.sqrt(diagonal_variances[coordinate_is_bounded])

    retained_mode_std_bounds = (
        1.0 / retained_singular_values
        if retained_singular_values.size
        else np.asarray([], dtype=float)
    )
    worst_retained = float("inf") if retained_mode_std_bounds.size == 0 else float(np.max(retained_mode_std_bounds))
    if retained_mode_std_bounds.size:
        expected = 1.0 / float(retained_singular_values[-1])
        if not np.isclose(worst_retained, expected, atol=1e-12, rtol=1e-10):
            raise AssertionError("worst retained bound must equal 1 / sigma_min_retained")

    rotation_norms, translation_norms, mode_kinds = _classify_retained_modes(variable_name, V_retained)

    scalar_std_bound = None
    scalar_frames = None
    target_ratio = None
    meets_target = None
    if n_parameters == 1:
        sigma_tau = float(np.linalg.norm(matrix[:, 0]))
        passed_gate = bool(sigma_tau > float(practical_rank_result.matrix_absolute_threshold))
        scalar_std_bound = float("inf") if sigma_tau == 0.0 or not passed_gate else float(1.0 / sigma_tau)
        if lidar_rate_hz is not None:
            scalar_frames = (
                float(scalar_std_bound * float(lidar_rate_hz))
                if np.isfinite(scalar_std_bound)
                else float("inf")
            )
        if target_std_seconds is not None:
            target_ratio = (
                float(scalar_std_bound / float(target_std_seconds))
                if np.isfinite(scalar_std_bound)
                else float("inf")
            )
            meets_target = bool(target_ratio <= 1.0)

    return LocalAccuracyDiagnostics(
        variable_name=str(variable_name),
        matrix_shape=tuple(matrix.shape),
        coordinate_labels=labels,
        coordinate_units=units,
        practical_rank=practical_rank,
        maximum_rank=n_parameters,
        nullity=max(n_parameters - practical_rank, 0),
        physical_information_matrix=information_matrix,
        physical_singular_values=singular_values,
        retained_mask=retained_mask,
        retained_mode_std_bounds=retained_mode_std_bounds,
        retained_mode_directions=V_retained,
        worst_retained_mode_std_bound=worst_retained,
        retained_mode_rotation_component_norms=rotation_norms,
        retained_mode_translation_component_norms=translation_norms,
        retained_mode_kinds=mode_kinds,
        full_covariance=full_covariance,
        observable_subspace_pseudocovariance=observable_pseudocovariance,
        observable_projector=observable_projector,
        null_projector=null_projector,
        coordinate_observable_fraction=observable_fraction,
        coordinate_null_fraction=null_fraction,
        coordinate_is_fully_bounded=coordinate_is_bounded,
        coordinate_std_bounds=coordinate_std_bounds,
        scalar_std_bound=scalar_std_bound,
        scalar_std_bound_lidar_frames=scalar_frames,
        target_std_bound=None if target_std_seconds is None else float(target_std_seconds),
        target_ratio=target_ratio,
        meets_target=meets_target,
        covariance_kind=covariance_kind,
        notes=tuple(notes),
    )


def _classify_retained_modes(
    variable_name: str,
    V_retained: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[str, ...],
]:
    '''Classify retained singular directions for an SE(3) variable.

    Args:
        variable_name (str): Calibration variable name.
        V_retained (NDArray[np.float64]): Retained right singular vectors with shape
            ``(n_X, practical_rank)``.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
        Rotation-component norms, translation-component norms, and mode labels.
    '''

    rotation_norms = np.full(V_retained.shape[1], np.nan, dtype=float)
    translation_norms = np.full(V_retained.shape[1], np.nan, dtype=float)
    kinds: list[str] = []
    if variable_name not in {"T_B_I", "T_B_L"} or V_retained.shape[0] < 6:
        return rotation_norms, translation_norms, tuple("native" for _ in range(V_retained.shape[1]))
    for mode_index in range(V_retained.shape[1]):
        direction = V_retained[:, mode_index]
        rotation_norm = float(np.linalg.norm(direction[:3]))
        translation_norm = float(np.linalg.norm(direction[3:6]))
        rotation_norms[mode_index] = rotation_norm
        translation_norms[mode_index] = translation_norm
        if translation_norm <= COORDINATE_NULL_FRACTION_TOLERANCE:
            kinds.append("rotation")
        elif rotation_norm <= COORDINATE_NULL_FRACTION_TOLERANCE:
            kinds.append("translation")
        elif rotation_norm >= translation_norm:
            kinds.append("rotation-dominant mixed")
        else:
            kinds.append("translation-dominant mixed")
    return rotation_norms, translation_norms, tuple(kinds)


##################################################
# Matrix validation and column helpers
##################################################
def _as_dense_matrix(matrix: ArrayLike | sparse.spmatrix) -> NDArray[np.float64]:
    '''Convert a dense or sparse input to a finite 2D dense matrix.

    Args:
        matrix (ArrayLike | sparse.spmatrix): Matrix to convert and validate.

    Returns:
        NDArray[np.float64]: Finite dense floating-point matrix.

    Raises:
        ValueError: If the converted input is not two-dimensional or contains
            non-finite values.
    '''

    # Preserve values exactly while normalizing the storage representation.
    dense_matrix = (
        matrix.toarray()
        if sparse.issparse(matrix)
        else np.asarray(matrix, dtype=float)
    )
    if dense_matrix.ndim != 2 or not np.all(np.isfinite(dense_matrix)):
        raise ValueError("matrix must be a finite 2D matrix")

    return dense_matrix


def _column_norms(matrix: NDArray[np.float64] | sparse.spmatrix) -> NDArray[np.float64]:
    '''Calculate Euclidean norms of dense or sparse matrix columns.

    Args:
        matrix (NDArray[np.float64] | sparse.spmatrix): Input matrix.

    Returns:
        NDArray[np.float64]: One Euclidean norm per matrix column.
    '''

    # Avoid densifying sparse matrices when only per-column energy is needed.
    if sparse.issparse(matrix):
        squared_norms = np.asarray(matrix.power(2).sum(axis=0)).ravel()
        return np.sqrt(squared_norms)

    return np.linalg.norm(np.asarray(matrix, dtype=float), axis=0)


def _zero_threshold_from_norms(
    column_norms: ArrayLike,
    *,
    absolute_zero_tolerance: float,
    relative_zero_tolerance: float,
) -> float:
    '''Calculate the whole-column rejection threshold.

    Args:
        column_norms (ArrayLike): Column norms used to determine the relative scale.
        absolute_zero_tolerance (float): Absolute rejection component.
        relative_zero_tolerance (float): Fraction of the maximum column norm.

    Returns:
        float: Combined absolute and relative rejection threshold.

    Raises:
        ValueError: If either tolerance is negative.
    '''

    # Both tolerance components represent nonnegative column magnitudes.
    if absolute_zero_tolerance < 0.0 or relative_zero_tolerance < 0.0:
        raise ValueError("zero tolerances must be nonnegative")

    norms = np.asarray(column_norms, dtype=float).reshape(-1)
    maximum_column_norm = float(np.max(norms)) if norms.size else 0.0

    return float(
        absolute_zero_tolerance
        + relative_zero_tolerance * maximum_column_norm
    )


##################################################
# Column rejection and diagnostic normalization
##################################################
def reject_and_normalize_columns_dense(
    J: ArrayLike,
    *,
    absolute_zero_tolerance: float,
    relative_zero_tolerance: float,
) -> ColumnRejectionNormalizationResult:
    '''Reject near-zero dense columns before normalizing active columns.

    Rejected columns remain exactly zero. Active columns are divided by their
    original Euclidean norms, preventing tiny numerical leakage from becoming a
    unit-strength diagnostic direction.

    Args:
        J (ArrayLike): Dense matrix with shape ``(m, n)``.
        absolute_zero_tolerance (float): Absolute column rejection tolerance.
        relative_zero_tolerance (float): Relative column rejection tolerance.

    Returns:
        ColumnRejectionNormalizationResult: Normalized matrix, original norms, and
        active/rejected column masks.
    '''

    matrix = _as_dense_matrix(J)
    column_norms = _column_norms(matrix)
    zero_threshold = _zero_threshold_from_norms(
        column_norms,
        absolute_zero_tolerance=absolute_zero_tolerance,
        relative_zero_tolerance=relative_zero_tolerance,
    )
    zero_mask = column_norms <= zero_threshold
    active_mask = ~zero_mask
    normalized = np.zeros_like(matrix, dtype=float)
    if np.any(active_mask):
        normalized[:, active_mask] = matrix[:, active_mask] / column_norms[active_mask]
    return ColumnRejectionNormalizationResult(
        normalized_matrix=normalized,
        original_column_norms=column_norms,
        zero_threshold=zero_threshold,
        zero_column_mask=zero_mask,
        active_column_mask=active_mask,
    )


def reject_and_normalize_columns_sparse(
    J: sparse.spmatrix,
    *,
    absolute_zero_tolerance: float,
    relative_zero_tolerance: float,
) -> ColumnRejectionNormalizationResult:
    '''Reject near-zero sparse columns before normalizing active columns.

    Args:
        J (sparse.spmatrix): Sparse matrix with shape ``(m, n)``.
        absolute_zero_tolerance (float): Absolute column rejection tolerance.
        relative_zero_tolerance (float): Relative column rejection tolerance.

    Returns:
        ColumnRejectionNormalizationResult: Sparse normalized matrix, original
        norms, and active/rejected column masks.

    Raises:
        ValueError: If ``J`` is not a SciPy sparse matrix.
    '''

    if not sparse.issparse(J):
        raise ValueError("J must be sparse")
    matrix = J.tocsc(copy=True)
    column_norms = _column_norms(matrix)
    zero_threshold = _zero_threshold_from_norms(
        column_norms,
        absolute_zero_tolerance=absolute_zero_tolerance,
        relative_zero_tolerance=relative_zero_tolerance,
    )
    zero_mask = column_norms <= zero_threshold
    active_mask = ~zero_mask
    inverse_norms = np.zeros_like(column_norms)
    inverse_norms[active_mask] = 1.0 / column_norms[active_mask]
    normalized = (matrix @ sparse.diags(inverse_norms, format="csc")).tocsr()
    normalized.eliminate_zeros()
    return ColumnRejectionNormalizationResult(
        normalized_matrix=normalized,
        original_column_norms=column_norms,
        zero_threshold=zero_threshold,
        zero_column_mask=zero_mask,
        active_column_mask=active_mask,
    )


##################################################
# Practical rank and physical information
##################################################
def practical_rank_diagnostics(
    O_physical: ArrayLike | sparse.spmatrix,
    *,
    policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
) -> PracticalRankDiagnostics:
    '''Apply the canonical practical-rank policy to a physical matrix.

    The matrix must be whitened, unnormalized, and expressed in physical ``D = I``
    coordinates. Whole columns are rejected before SVD, but individual matrix
    entries are never thresholded.

    Args:
        O_physical (ArrayLike | sparse.spmatrix): Physical projected matrix with
            shape ``(m, n_X)``.
        policy (PracticalRankPolicy): Column, matrix, and singular-value thresholds.

    Returns:
        PracticalRankDiagnostics: Column rejection, SVD, rank, condition-number,
        and retained-mode uncertainty diagnostics.
    '''

    matrix = _as_dense_matrix(O_physical)
    column_norms = _column_norms(matrix)
    maximum_column_norm = float(np.max(column_norms)) if column_norms.size else 0.0
    column_threshold = float(policy.column_absolute_threshold + policy.column_relative_threshold * maximum_column_norm)
    zero_column_mask = column_norms <= column_threshold
    active_column_mask = ~zero_column_mask
    filtered_matrix = matrix.copy()
    if zero_column_mask.size:
        filtered_matrix[:, zero_column_mask] = 0.0

    # singular_values: (min(m, n_X),), descending by construction.
    singular_values = np.linalg.svd(filtered_matrix, compute_uv=False)
    maximum_possible_rank = min(matrix.shape)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    normalized = singular_values / sigma_max if sigma_max > 0.0 else np.zeros_like(singular_values)
    machine_threshold = max(matrix.shape, default=0) * np.finfo(float).eps * sigma_max
    machine_rank = int(np.sum(singular_values > machine_threshold)) if singular_values.size else 0
    relative_only_rank = (
        int(
            np.sum(
                singular_values
                > float(policy.singular_relative_threshold) * sigma_max
            )
        )
        if sigma_max > 0.0
        else 0
    )

    matrix_passed = bool(sigma_max > float(policy.matrix_absolute_threshold))
    if not matrix_passed:
        retained_mask = np.zeros_like(singular_values, dtype=bool)
        singular_threshold = float("nan")
        practical_rank = 0
        sigma_min_retained = float("nan")
        condition = float("nan")
        worst_std = float("inf")
    else:
        singular_threshold = float(
            max(
                policy.singular_absolute_threshold,
                policy.singular_relative_threshold * sigma_max,
            )
        )
        retained_mask = singular_values > singular_threshold
        practical_rank = int(np.sum(retained_mask))
        if practical_rank == 0:
            sigma_min_retained = float("nan")
            condition = float("nan")
            worst_std = float("inf")
        else:
            retained_values = singular_values[retained_mask]
            sigma_min_retained = float(retained_values[-1])
            condition = 1.0 if practical_rank == 1 else float(sigma_max / sigma_min_retained)
            worst_std = float(1.0 / sigma_min_retained) if sigma_min_retained > 0.0 else float("inf")

    return PracticalRankDiagnostics(
        matrix_shape=tuple(matrix.shape),
        column_norms=column_norms,
        column_threshold=column_threshold,
        zero_column_mask=zero_column_mask,
        active_column_mask=active_column_mask,
        filtered_matrix=filtered_matrix,
        singular_values=singular_values,
        normalized_singular_values=normalized,
        sigma_max=sigma_max,
        matrix_absolute_threshold=float(policy.matrix_absolute_threshold),
        matrix_passed_absolute_gate=matrix_passed,
        singular_threshold=singular_threshold,
        retained_mask=retained_mask,
        machine_rank=machine_rank,
        relative_only_rank=relative_only_rank,
        practical_rank=practical_rank,
        maximum_possible_rank=maximum_possible_rank,
        sigma_min_retained=sigma_min_retained,
        practical_condition_number=condition,
        worst_retained_std_bound=worst_std,
    )


def common_rank_diagnostics(
    matrix: ArrayLike | sparse.spmatrix,
    *,
    relative_rank_threshold: float,
) -> CommonRankDiagnostics:
    '''Return backward-compatible common rank diagnostics.

    Args:
        matrix (ArrayLike | sparse.spmatrix): Matrix to analyze.
        relative_rank_threshold (float): Relative singular-value threshold.

    Returns:
        CommonRankDiagnostics: Compatibility fields backed by the canonical
        practical-rank implementation.
    '''

    policy = PracticalRankPolicy(singular_relative_threshold=float(relative_rank_threshold))
    rank = practical_rank_diagnostics(matrix, policy=policy)
    machine_threshold = max(rank.matrix_shape, default=0) * np.finfo(float).eps * rank.sigma_max
    return CommonRankDiagnostics(
        machine_rank=rank.machine_rank,
        effective_rank=rank.practical_rank,
        maximum_possible_rank=rank.maximum_possible_rank,
        machine_threshold=float(machine_threshold),
        relative_rank_threshold=float(relative_rank_threshold),
        singular_values=rank.singular_values,
        normalized_singular_values=rank.normalized_singular_values,
        retained_mask=rank.retained_mask,
    )


def physical_information_diagnostics(
    O_physical: ArrayLike | sparse.spmatrix,
    *,
    practical_rank_result: PracticalRankDiagnostics | None = None,
    policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
    relative_rank_threshold: float | None = None,
) -> PhysicalInformationDiagnostics:
    '''Calculate physical information and retained-mode uncertainty diagnostics.

    The information matrix is always formed from the original unnormalized
    physical matrix, not from the column-filtered matrix used only for practical
    rank interpretation.

    Args:
        O_physical (ArrayLike | sparse.spmatrix): Physical projected matrix with
            shape ``(m, n_X)``.
        practical_rank_result (PracticalRankDiagnostics | None): Optional canonical
            rank result for the same matrix.
        policy (PracticalRankPolicy): Rank policy used when no result is supplied.
        relative_rank_threshold (float | None): Backward-compatible override for
            the policy relative threshold.

    Returns:
        PhysicalInformationDiagnostics: Information matrix, singular values,
        covariance or pseudocovariance, and compatibility aliases.
    '''

    if relative_rank_threshold is not None:
        policy = PracticalRankPolicy(singular_relative_threshold=float(relative_rank_threshold))
    matrix = _as_dense_matrix(O_physical)
    rank = (
        practical_rank_result
        if practical_rank_result is not None
        else practical_rank_diagnostics(matrix, policy=policy)
    )
    # O_X_physical.T: (n_X, m), O_X_physical: (m, n_X) -> S_X: (n_X, n_X).
    information_matrix = matrix.T @ matrix
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    information_eigenvalues = singular_values**2
    trace_information = float(np.trace(information_matrix))
    frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    retained_singular_values = rank.singular_values[rank.retained_mask]
    if retained_singular_values.size:
        std_bounds = 1.0 / retained_singular_values
        worst_std = float(np.max(std_bounds))
        sigma_min = float(retained_singular_values[-1])
        effective_condition = rank.practical_condition_number
    else:
        std_bounds = np.asarray([], dtype=float)
        worst_std = float("inf")
        sigma_min = float("nan")
        effective_condition = float("nan")

    full_covariance = None
    pseudocovariance = None
    notes: list[str] = [
        "information uses unnormalized physical O_X with D=I",
        "SE(3) directions may mix radians and metres; compare within variable families",
    ]
    n_parameters = matrix.shape[1]
    full_rank_tol = max(matrix.shape, default=0) * np.finfo(float).eps * sigma_max
    is_full_column_rank = (
        singular_values.size >= n_parameters
        and int(np.sum(singular_values > full_rank_tol)) == n_parameters
    )
    if is_full_column_rank and n_parameters:
        full_covariance = np.linalg.pinv(information_matrix, rcond=np.finfo(float).eps)
        notes.append("full_covariance is a conventional local covariance because O_X is full column rank")
    elif n_parameters:
        pseudocovariance = np.linalg.pinv(information_matrix, rcond=np.finfo(float).eps)
        notes.append("rank deficient target; pseudocovariance is only observable-subspace diagnostic")

    return PhysicalInformationDiagnostics(
        information_matrix=information_matrix,
        information_eigenvalues=information_eigenvalues,
        trace_information=trace_information,
        frobenius_norm_O=frobenius_norm,
        sigma_max_physical=sigma_max,
        all_physical_singular_values=singular_values,
        observable_subspace_std_bounds=std_bounds,
        worst_retained_std_bound=worst_std,
        full_covariance=full_covariance,
        observable_subspace_pseudocovariance=pseudocovariance,
        notes=tuple(notes),
        singular_values=singular_values,
        total_information=trace_information,
        sigma_max=sigma_max,
        sigma_min_retained=sigma_min,
        effective_condition_number=effective_condition,
        standard_deviation_bounds=std_bounds,
        maximum_retained_standard_deviation_bound=worst_std,
    )


def scalar_time_offset_diagnostics(
    O_tau_physical: ArrayLike | sparse.spmatrix,
    *,
    policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
    target_std_seconds: float,
) -> ScalarTimeOffsetDiagnostics:
    '''Calculate scalar temporal-offset sensitivity and local uncertainty.

    Args:
        O_tau_physical (ArrayLike | sparse.spmatrix): Physical timing column with
            shape ``(m, 1)``.
        policy (PracticalRankPolicy): Absolute sensitivity gate.
        target_std_seconds (float): Requested standard-deviation target in seconds.

    Returns:
        ScalarTimeOffsetDiagnostics: Scalar sensitivity, information, bound, and
        target decision.

    Raises:
        ValueError: If the input does not contain exactly one column.
    '''

    matrix = _as_dense_matrix(O_tau_physical)
    if matrix.shape[1] != 1:
        raise ValueError("O_tau_physical must have exactly one column")
    sensitivity = float(np.linalg.norm(matrix[:, 0]))
    passed_gate = bool(sensitivity > float(policy.matrix_absolute_threshold))
    std_bound = float("inf") if sensitivity == 0.0 or not passed_gate else float(1.0 / sensitivity)
    information = sensitivity**2
    return ScalarTimeOffsetDiagnostics(
        sensitivity_tau=sensitivity,
        information_tau=information,
        local_std_bound_tau_seconds=std_bound,
        target_std_seconds=float(target_std_seconds),
        meets_target=bool(np.isfinite(std_bound) and std_bound <= float(target_std_seconds)),
        matrix_passed_absolute_gate=passed_gate,
    )


##################################################
# Stored diagnostic consistency checks
##################################################
def validate_stored_rank_against_matrix(
    O_physical: ArrayLike | sparse.spmatrix,
    stored_diagnostics: PracticalRankDiagnostics,
    policy: PracticalRankPolicy,
) -> PracticalRankDiagnostics:
    '''Recompute rank diagnostics and verify a stored result.

    Args:
        O_physical (ArrayLike | sparse.spmatrix): Physical projected matrix.
        stored_diagnostics (PracticalRankDiagnostics): Previously stored result.
        policy (PracticalRankPolicy): Policy used for direct recomputation.

    Returns:
        PracticalRankDiagnostics: Newly recomputed diagnostics after all checks pass.

    Raises:
        AssertionError: If singular values, masks, gate decisions, rank, or condition
            number differ from direct recomputation.
    '''

    # Recompute every canonical diagnostic directly from the physical matrix.
    recomputed = practical_rank_diagnostics(O_physical, policy=policy)

    # Check the continuous SVD quantities before discrete masks and rank.
    if not np.allclose(
        recomputed.singular_values,
        stored_diagnostics.singular_values,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise AssertionError("stored singular values do not match direct SVD")
    if not np.array_equal(
        recomputed.zero_column_mask,
        stored_diagnostics.zero_column_mask,
    ):
        raise AssertionError("stored zero-column mask does not match direct column rejection")
    # The gate, retained mask, and final rank must be exactly reproducible.
    if (
        recomputed.matrix_passed_absolute_gate
        != stored_diagnostics.matrix_passed_absolute_gate
    ):
        raise AssertionError("stored matrix gate decision does not match direct recomputation")
    if not np.array_equal(
        recomputed.retained_mask,
        stored_diagnostics.retained_mask,
    ):
        raise AssertionError("stored retained singular mask does not match direct recomputation")
    if recomputed.practical_rank != stored_diagnostics.practical_rank:
        raise AssertionError("stored practical rank does not match direct recomputation")
    stored_condition = stored_diagnostics.practical_condition_number
    recomputed_condition = recomputed.practical_condition_number
    conditions_match = (
        np.isnan(stored_condition) and np.isnan(recomputed_condition)
    ) or np.isclose(stored_condition, recomputed_condition, atol=1e-12, rtol=1e-10)
    if not conditions_match:
        raise AssertionError("stored practical condition number does not match direct recomputation")
    return recomputed