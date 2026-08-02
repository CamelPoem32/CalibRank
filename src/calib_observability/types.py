'''Shared type aliases and configuration objects for calibration observability.

This module centralizes array aliases, supported operating modes, numerical
thresholds, and Jacobian-checking configuration. Group perturbations elsewhere
in the package use left multiplication and rotation-first tangent ordering.
'''

from __future__ import annotations

from numpy.typing import NDArray
import numpy as np

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
SO2Matrix = NDArray[np.float64]
SE2Matrix = NDArray[np.float64]
SO3Matrix = NDArray[np.float64]
SE3Matrix = NDArray[np.float64]


from dataclasses import dataclass, asdict
from typing import Literal
from numpy.typing import ArrayLike


##################################################
# Public mode aliases and default thresholds
##################################################
JacobianMethod = Literal["analytic", "finite_difference", "analytic_checked"]
FixedExtrinsic = Literal["none", "T_B_L", "T_B_I"]
AccelerometerMode = Literal["disabled", "simple", "complex"]

FIXED_EXTRINSIC: FixedExtrinsic = "T_B_L"
COLUMN_ABSOLUTE_THRESHOLD = 1e-6
COLUMN_RELATIVE_THRESHOLD = 1e-5
MATRIX_ABSOLUTE_THRESHOLD = 1e-5
SINGULAR_ABSOLUTE_THRESHOLD = 1e-5
SINGULAR_RELATIVE_THRESHOLD = 1e-5
TAU_TARGET_FRAMES = 1.0


##################################################
# Practical-rank configuration
##################################################
@dataclass(frozen=True)
class PracticalRankPolicy:
    '''Store thresholds used by the canonical practical-rank policy.

    The policy acts on an unnormalized, whitened observability matrix in native
    physical coordinates. It first rejects negligible whole columns, then
    rejects an entirely negligible matrix, and finally combines absolute and
    relative singular-value thresholds. Entrywise thresholding is intentionally
    excluded because it can change cancellations and singular directions.

    Attributes:
        column_absolute_threshold (float): Absolute whole-column rejection
            threshold.
        column_relative_threshold (float): Whole-column threshold relative to
            the largest column norm.
        matrix_absolute_threshold (float): Absolute gate for the complete
            observability matrix.
        singular_absolute_threshold (float): Absolute retained-singular-value
            threshold.
        singular_relative_threshold (float): Retained-singular-value threshold
            relative to the largest singular value.
    '''

    column_absolute_threshold: float = COLUMN_ABSOLUTE_THRESHOLD
    column_relative_threshold: float = COLUMN_RELATIVE_THRESHOLD
    matrix_absolute_threshold: float = MATRIX_ABSOLUTE_THRESHOLD
    singular_absolute_threshold: float = SINGULAR_ABSOLUTE_THRESHOLD
    singular_relative_threshold: float = SINGULAR_RELATIVE_THRESHOLD

    def __post_init__(self) -> None:
        '''Validate all practical-rank thresholds after construction.

        Raises:
            ValueError: If any threshold is non-finite or negative, or if the
                relative singular-value threshold is not smaller than one.
        '''

        # Collect all dataclass fields into one vector for a shared finite and
        # nonnegative validation pass.
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("practical-rank thresholds must be finite and nonnegative")

        # A relative threshold of one would reject even the leading singular
        # value because retained values are compared using a strict inequality.
        if not 0.0 <= float(self.singular_relative_threshold) < 1.0:
            raise ValueError("singular_relative_threshold must satisfy 0 <= threshold < 1")

    def as_metadata(self) -> dict[str, float]:
        '''Return a JSON-friendly representation of the rank policy.

        Returns:
            dict[str, float]: Policy fields converted to plain Python floats.
        '''

        return {key: float(value) for key, value in asdict(self).items()}


DEFAULT_PRACTICAL_RANK_POLICY = PracticalRankPolicy()


##################################################
# Public mode validators
##################################################
def validate_fixed_extrinsic(value: str | None) -> FixedExtrinsic:
    '''Validate the selected fixed-extrinsic convention.

    Args:
        value (str | None): Requested convention. ``None`` selects ``none``.

    Returns:
        FixedExtrinsic: One of ``none``, ``T_B_L``, or ``T_B_I``.

    Raises:
        ValueError: If the requested convention is unsupported.
    '''

    selected = "none" if value is None else str(value)
    if selected not in {"none", "T_B_L", "T_B_I"}:
        raise ValueError("fixed_extrinsic must be 'none', 'T_B_L', or 'T_B_I'")
    return selected  # type: ignore[return-value]


def validate_accelerometer_mode(value: str | None) -> AccelerometerMode:
    '''Validate the selected accelerometer-factor mode.

    Args:
        value (str | None): Requested mode. ``None`` selects ``disabled``.

    Returns:
        AccelerometerMode: One of ``disabled``, ``simple``, or ``complex``.

    Raises:
        ValueError: If the requested mode is unsupported.
    '''

    selected = "disabled" if value is None else str(value)
    if selected not in {"disabled", "simple", "complex"}:
        raise ValueError("accelerometer mode must be 'disabled', 'simple', or 'complex'")
    return selected  # type: ignore[return-value]


##################################################
# Accelerometer factor configuration
##################################################
@dataclass(frozen=True)
class AccelerometerOptions:
    '''Store options for optional accelerometer factor construction.

    The accelerometer shares ``T_B_I`` and ``tau_I`` with the gyroscope and
    does not introduce accelerometer bias, scale, axis misalignment, velocity,
    or a separate temporal-offset variable. Measurements are interpreted as
    IMU-frame specific force including gravity:

        f_m^I = R_IW (a_I^W - g_W) + noise.

    Attributes:
        mode (AccelerometerMode): Disabled, simple, or complex factor model.
        factor_rate_hz (float | None): Optional factor construction rate.
        sample_stride (int): Number of candidate samples skipped per factor.
        support_half_width_seconds (float): Half-width of the complex
            three-pose support interval.
        gravity_norm_tolerance_m_s2 (float): Allowed gravity-norm mismatch for
            factor acceptance.
        low_dynamic_gyro_threshold_rad_s (float): Gyroscope threshold used by
            the low-dynamic gate.
        require_low_dynamic_gate (bool): Whether low-dynamic gating is active.
        measurement_std_m_s2 (float | tuple[float, float, float] | None):
            Optional scalar or per-axis measurement standard deviation.
        save_factor_terms (bool): Whether intermediate factor terms are stored.
    '''

    mode: AccelerometerMode = "disabled"
    factor_rate_hz: float | None = None
    sample_stride: int = 1
    support_half_width_seconds: float = 0.2
    gravity_norm_tolerance_m_s2: float = 0.5
    low_dynamic_gyro_threshold_rad_s: float = 0.2
    require_low_dynamic_gate: bool = True
    measurement_std_m_s2: float | tuple[float, float, float] | None = None
    save_factor_terms: bool = False

    def __post_init__(self) -> None:
        '''Validate accelerometer factor configuration after construction.

        Raises:
            ValueError: If a mode, rate, stride, support width, gate threshold,
                or measurement standard deviation is invalid.
        '''

        # Validate the discrete mode and positive sampling/support parameters.
        validate_accelerometer_mode(self.mode)
        if not np.isfinite(self.support_half_width_seconds) or self.support_half_width_seconds <= 0.0:
            raise ValueError("support_half_width_seconds must be finite and positive")
        if self.factor_rate_hz is not None and (not np.isfinite(self.factor_rate_hz) or self.factor_rate_hz <= 0.0):
            raise ValueError("factor_rate_hz must be None or finite and positive")
        if int(self.sample_stride) < 1:
            raise ValueError("sample_stride must be at least 1")

        # Gate tolerances are allowed to be zero but must remain finite.
        if self.gravity_norm_tolerance_m_s2 < 0.0 or not np.isfinite(self.gravity_norm_tolerance_m_s2):
            raise ValueError("gravity_norm_tolerance_m_s2 must be finite and nonnegative")
        if self.low_dynamic_gyro_threshold_rad_s < 0.0 or not np.isfinite(self.low_dynamic_gyro_threshold_rad_s):
            raise ValueError("low_dynamic_gyro_threshold_rad_s must be finite and nonnegative")

        # Accept either one positive standard deviation for all axes or three
        # positive axis-specific values.
        if self.measurement_std_m_s2 is not None:
            values = np.asarray(self.measurement_std_m_s2, dtype=float)
            if values.shape == ():
                values = np.full(3, float(values))
            if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError("measurement_std_m_s2 must be positive scalar or length-3 tuple")

    def covariance(self, fallback_covariance: ArrayLike) -> NDArray[np.float64]:
        '''Return the accelerometer measurement covariance.

        Args:
            fallback_covariance (ArrayLike): Covariance used when no explicit
                standard deviation is configured, shape ``(3, 3)``.

        Returns:
            NDArray[np.float64]: Accelerometer covariance, shape ``(3, 3)``.

        Raises:
            ValueError: If the fallback covariance is not finite with shape
                ``(3, 3)``.
        '''

        # Reuse the provided covariance when no explicit noise model overrides
        # it in this options object.
        if self.measurement_std_m_s2 is None:
            covariance = np.asarray(fallback_covariance, dtype=float)
            if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
                raise ValueError("fallback accelerometer covariance must be finite with shape (3, 3)")
            return covariance

        # Expand a scalar standard deviation to all axes and convert standard
        # deviations to a diagonal covariance.
        std = np.asarray(self.measurement_std_m_s2, dtype=float)
        if std.shape == ():
            std = np.full(3, float(std))
        return np.diag(std**2)

    def as_metadata(self) -> dict[str, object]:
        '''Return a JSON-friendly representation of accelerometer options.

        Returns:
            dict[str, object]: Dataclass fields with tuple-valued standard
                deviations converted to a list.
        '''

        data = asdict(self)
        if isinstance(data.get("measurement_std_m_s2"), tuple):
            data["measurement_std_m_s2"] = list(data["measurement_std_m_s2"])
        return data


##################################################
# Jacobian construction and checking configuration
##################################################
@dataclass(frozen=True)
class JacobianOptions:
    '''Store options controlling local factor Jacobian construction.

    All SO(3) and SE(3) variables use left perturbations,
    ``T_perturbed = Exp(delta_xi) @ T``, with rotation-first tangent ordering.
    ``analytic_checked`` evaluates analytic and central finite-difference blocks
    against the same residual convention, then returns the analytic blocks when
    all comparisons pass.

    Attributes:
        method (JacobianMethod): Analytic, finite-difference, or checked mode.
        finite_difference_epsilon (float): Central perturbation magnitude.
        check_atol (float): Absolute Jacobian comparison tolerance.
        check_rtol (float): Relative Jacobian comparison tolerance.
        raise_on_check_failure (bool): Whether a failed comparison raises.

    Notes:
        These checks validate one local linearization numerically. They do not
        prove global observability and may become ill-conditioned near Lie-log
        branch singularities.
    '''

    method: JacobianMethod = "analytic"
    finite_difference_epsilon: float = 1e-7
    check_atol: float = 1e-6
    check_rtol: float = 1e-5
    raise_on_check_failure: bool = True


class JacobianCheckError(RuntimeError):
    '''Report a failed Jacobian comparison in ``analytic_checked`` mode.'''


@dataclass(frozen=True)
class JacobianCheckResult:
    '''Store numerical diagnostics for one analytic Jacobian block.

    The comparison uses the maximum absolute element error and relative
    Frobenius error
    ``||H_analytic - H_fd||_F / max(||H_fd||_F, eps)``.

    Attributes:
        factor_name (str): Name of the factor being checked.
        variable_name (str): Name of the perturbed variable block.
        analytic_shape (tuple[int, int]): Shape of the analytic block.
        finite_difference_shape (tuple[int, int]): Shape of the numerical block.
        max_absolute_error (float): Largest absolute element difference.
        relative_frobenius_error (float): Relative Frobenius-norm difference.
        passed (bool): Whether the configured tolerances were satisfied.
    '''

    factor_name: str
    variable_name: str
    analytic_shape: tuple[int, int]
    finite_difference_shape: tuple[int, int]
    max_absolute_error: float
    relative_frobenius_error: float
    passed: bool


def validate_jacobian_method(method: str) -> JacobianMethod:
    '''Validate and return a public Jacobian evaluation mode.

    Args:
        method (str): Requested Jacobian method.

    Returns:
        JacobianMethod: Validated analytic, finite-difference, or checked mode.

    Raises:
        ValueError: If the method is unsupported.
    '''

    if method not in {"analytic", "finite_difference", "analytic_checked"}:
        raise ValueError("jacobian method must be 'analytic', 'finite_difference', or 'analytic_checked'")
    return method  # type: ignore[return-value]


def normalized_jacobian_options(options: JacobianOptions | None = None) -> JacobianOptions:
    '''Return validated Jacobian options, inserting defaults when omitted.

    Args:
        options (JacobianOptions | None): User-provided options or ``None``.

    Returns:
        JacobianOptions: Validated configuration object.

    Raises:
        ValueError: If the method, finite-difference step, or check tolerances
            are invalid.
    '''

    result = JacobianOptions() if options is None else options
    validate_jacobian_method(result.method)
    if result.finite_difference_epsilon <= 0.0:
        raise ValueError("finite_difference_epsilon must be positive")
    if result.check_atol < 0.0 or result.check_rtol < 0.0:
        raise ValueError("check tolerances must be nonnegative")
    return result


def compare_jacobians(
    analytic: ArrayLike,
    finite_difference: ArrayLike,
    *,
    factor_name: str,
    variable_name: str,
    atol: float,
    rtol: float,
) -> JacobianCheckResult:
    '''Compare an analytic Jacobian block with a finite-difference reference.

    The caller supplies blocks expressed under the same residual and
    perturbation conventions. This helper only computes shape and numerical
    comparison diagnostics.

    Args:
        analytic (ArrayLike): Analytic Jacobian block.
        finite_difference (ArrayLike): Numerical reference block.
        factor_name (str): Factor name included in the result.
        variable_name (str): Variable-block name included in the result.
        atol (float): Absolute comparison tolerance.
        rtol (float): Relative comparison tolerance.

    Returns:
        JacobianCheckResult: Shape, error, and pass/fail diagnostics.

    Raises:
        ValueError: If either input is not two-dimensional.
    '''

    # Convert both blocks once so shape and numerical diagnostics use the same
    # floating-point representation.
    analytic_matrix = np.asarray(analytic, dtype=float)
    finite_difference_matrix = np.asarray(finite_difference, dtype=float)
    if analytic_matrix.ndim != 2 or finite_difference_matrix.ndim != 2:
        raise ValueError("Jacobian blocks must be two-dimensional")

    # A shape mismatch is represented as a failed check rather than an
    # exception, allowing callers to aggregate diagnostics consistently.
    if analytic_matrix.shape != finite_difference_matrix.shape:
        return JacobianCheckResult(
            factor_name=factor_name,
            variable_name=variable_name,
            analytic_shape=analytic_matrix.shape,
            finite_difference_shape=finite_difference_matrix.shape,
            max_absolute_error=float("inf"),
            relative_frobenius_error=float("inf"),
            passed=False,
        )

    # Calculate absolute and normalized global discrepancies for equal-shaped
    # blocks, including well-defined zero-size matrix behavior.
    difference = analytic_matrix - finite_difference_matrix
    max_absolute_error = float(np.max(np.abs(difference))) if difference.size else 0.0
    denominator = max(float(np.linalg.norm(finite_difference_matrix, ord="fro")), np.finfo(float).eps)
    relative_frobenius_error = float(np.linalg.norm(difference, ord="fro") / denominator)
    passed = bool(max_absolute_error <= float(atol) + float(rtol) * float(np.max(np.abs(finite_difference_matrix)) if finite_difference_matrix.size else 0.0))

    return JacobianCheckResult(
        factor_name=factor_name,
        variable_name=variable_name,
        analytic_shape=analytic_matrix.shape,
        finite_difference_shape=finite_difference_matrix.shape,
        max_absolute_error=max_absolute_error,
        relative_frobenius_error=relative_frobenius_error,
        passed=passed,
    )