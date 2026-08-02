'''Accelerometer residuals and Jacobians for calibration observability.

The accelerometer measurement convention is IMU-frame specific force including
gravity in the usual accelerometer sense:

    f_m^I = R_IW (a_I^W - g_W) + noise.

No accelerometer bias, velocity state, scale, axis-misalignment, or separate
accelerometer time offset is introduced. The accelerometer shares T_B_I and
tau_I with the gyroscope. All pose and extrinsic perturbations are left
perturbations in rotation-first SE(3) tangent ordering.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .finite_difference import central_difference_vector, finite_difference_left_jacobian_se3
from .lie_se3 import se3_exp
from .lie_so3 import so3_hat, so3_left_jacobian_inverse, so3_log
from .types import (
    AccelerometerMode,
    JacobianCheckError,
    JacobianCheckResult,
    JacobianOptions,
    compare_jacobians,
    normalized_jacobian_options,
    validate_accelerometer_mode,
)


##################################################
# Accelerometer factor output structures
##################################################
@dataclass(frozen=True)
class AccelerometerPoseBlock:
    '''Store one pose Jacobian block of an accelerometer factor.

    name is ``zero`` for the simple one-pose factor and one of ``minus``,
    ``zero`` or ``plus`` for the complex three-pose factor. matrix uses
    rotation-first left perturbations of the corresponding T_W_B pose.

    Attributes:
        name (str): pose position inside the factor support
        matrix (array 3x6): Jacobian with respect to the selected pose
    '''

    name: str
    matrix: NDArray[np.float64]


@dataclass(frozen=True)
class AccelerometerFactorTerms:
    '''Store intermediate quantities used by the accelerometer factors.

    All vector-valued fields have shape (3,). The complex factor stores the
    translational and rotational kinematic terms required for the complete
    specific-force prediction. For simple mode, complex-only fields contain
    zero vectors or zero matrices while keeping the same output structure.
    '''

    mode: AccelerometerMode
    sensor_time: float
    tau_I: float
    t_minus: float
    t_zero: float
    t_plus: float
    T_minus: NDArray[np.float64]
    T_zero: NDArray[np.float64]
    T_plus: NDArray[np.float64]
    T_B_I: NDArray[np.float64]
    gravity_world: NDArray[np.float64]
    a_zero_world: NDArray[np.float64]
    phi_minus: NDArray[np.float64]
    phi_plus: NDArray[np.float64]
    omega_zero_body: NDArray[np.float64]
    alpha_zero_body: NDArray[np.float64]
    specific_force_body_origin: NDArray[np.float64]
    lever_arm_matrix: NDArray[np.float64]
    tangential_acceleration: NDArray[np.float64]
    centripetal_acceleration: NDArray[np.float64]
    predicted_specific_force: NDArray[np.float64]
    measured_specific_force: NDArray[np.float64]
    residual: NDArray[np.float64]
    G_s_theta_minus: NDArray[np.float64]
    G_s_theta_zero: NDArray[np.float64]
    G_s_theta_plus: NDArray[np.float64]
    G_s_rho_minus: NDArray[np.float64]
    G_s_rho_zero: NDArray[np.float64]
    G_s_rho_plus: NDArray[np.float64]
    L_minus: NDArray[np.float64]
    L_plus: NDArray[np.float64]
    G_omega_minus: NDArray[np.float64]
    G_omega_zero: NDArray[np.float64]
    G_omega_plus: NDArray[np.float64]
    G_alpha_minus: NDArray[np.float64]
    G_alpha_zero: NDArray[np.float64]
    G_alpha_plus: NDArray[np.float64]
    B_omega: NDArray[np.float64]
    B_alpha: NDArray[np.float64]


@dataclass(frozen=True)
class AccelerometerFactorLinearization:
    '''Store an accelerometer residual and its local Jacobian blocks.

    residual has shape (3,). pose_blocks contains one block for simple mode or
    three blocks for complex mode. H_T_B_I has shape (3, 6), and H_tau_I has
    shape (3, 1). check_results is populated only when analytic Jacobians are
    explicitly checked against finite differences.
    '''

    residual: NDArray[np.float64]
    pose_blocks: tuple[AccelerometerPoseBlock, ...]
    H_T_B_I: NDArray[np.float64]
    H_tau_I: NDArray[np.float64]
    check_results: tuple[JacobianCheckResult, ...]
    maximum_check_error: float | None
    all_checks_passed: bool | None
    terms: AccelerometerFactorTerms | None


##################################################
# Input validation helpers
##################################################
def _as_transform(value: ArrayLike, name: str) -> NDArray[np.float64]:
    '''Convert an input value to a finite 4x4 transformation matrix.

    Args:
        value (array-like 4x4): transformation matrix to validate
        name (str): argument name used in the error message

    Returns:
        array 4x4: validated floating-point transformation
    '''

    # Convert first, then validate both matrix shape and numerical values.
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite SE(3) matrix with shape (4, 4)")

    return matrix


def _as_vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    '''Convert an input value to a finite three-dimensional vector.

    Args:
        value (array-like 3): vector to validate
        name (str): argument name used in the error message

    Returns:
        array 3: validated floating-point vector
    '''

    # Reshape to the expected vector representation and reject invalid values.
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite with shape (3,)")

    return vector


##################################################
# Simple one-pose accelerometer factor
##################################################
def simple_accelerometer_residual(
    T_W_B: ArrayLike,
    T_B_I: ArrayLike,
    measured_specific_force_I: ArrayLike,
    gravity_world: ArrayLike,
) -> NDArray[np.float64]:
    '''Calculate the simple gravity-alignment accelerometer residual.

    The predicted measurement is

        f_hat_I = C.T @ (-R.T @ gravity_world),

    where R is the body orientation in the world frame and C is the IMU
    orientation in the body frame. Translational acceleration, angular motion
    and the IMU lever arm are ignored by this factor.

    Args:
        T_W_B (array 4x4): body pose in the world frame
        T_B_I (array 4x4): IMU pose in the body frame
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF

    Returns:
        array 3: predicted minus measured specific force
    '''

    # Validate the input transformations and measurement vectors.
    T = _as_transform(T_W_B, "T_W_B")
    X = _as_transform(T_B_I, "T_B_I")
    measured = _as_vector3(measured_specific_force_I, "measured_specific_force_I")
    gravity = _as_vector3(gravity_world, "gravity_world")

    # Rotate gravity from world RF to body RF and then to IMU RF.
    R = T[:3, :3]
    C = X[:3, :3]
    predicted = C.T @ (-R.T @ gravity)

    return predicted - measured


def _simple_terms(
    T_W_B: NDArray[np.float64],
    T_B_I: NDArray[np.float64],
    measured_specific_force_I: NDArray[np.float64],
    gravity_world: NDArray[np.float64],
    sensor_time: float,
    tau_I: float,
) -> AccelerometerFactorTerms:
    '''Collect all output terms of the simple accelerometer factor.

    Complex-factor quantities are filled with zeros so both factor modes share
    one output data structure.

    Args:
        T_W_B (array 4x4): validated body pose in world RF
        T_B_I (array 4x4): validated IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        sensor_time (float): original measurement timestamp
        tau_I (float): IMU temporal offset

    Returns:
        AccelerometerFactorTerms: residual and intermediate values
    '''

    # Calculate the gravity-only specific-force prediction.
    R = T_W_B[:3, :3]
    C = T_B_I[:3, :3]
    y_g = -R.T @ gravity_world
    predicted = C.T @ y_g
    residual = predicted - measured_specific_force_I

    # Keep the complete complex-factor interface using zero placeholders.
    z3 = np.zeros(3)
    Z = np.zeros((3, 3))

    return AccelerometerFactorTerms(
        mode="simple",
        sensor_time=float(sensor_time),
        tau_I=float(tau_I),
        t_minus=float(sensor_time + tau_I),
        t_zero=float(sensor_time + tau_I),
        t_plus=float(sensor_time + tau_I),
        T_minus=T_W_B,
        T_zero=T_W_B,
        T_plus=T_W_B,
        T_B_I=T_B_I,
        gravity_world=gravity_world,
        a_zero_world=z3,
        phi_minus=z3,
        phi_plus=z3,
        omega_zero_body=z3,
        alpha_zero_body=z3,
        specific_force_body_origin=y_g,
        lever_arm_matrix=Z,
        tangential_acceleration=z3,
        centripetal_acceleration=z3,
        predicted_specific_force=predicted,
        measured_specific_force=measured_specific_force_I,
        residual=residual,
        G_s_theta_minus=Z,
        G_s_theta_zero=Z,
        G_s_theta_plus=Z,
        G_s_rho_minus=Z,
        G_s_rho_zero=Z,
        G_s_rho_plus=Z,
        L_minus=Z,
        L_plus=Z,
        G_omega_minus=Z,
        G_omega_zero=Z,
        G_omega_plus=Z,
        G_alpha_minus=Z,
        G_alpha_zero=Z,
        G_alpha_plus=Z,
        B_omega=Z,
        B_alpha=Z,
    )


def simple_accelerometer_analytic_blocks(
    T_W_B: ArrayLike,
    T_B_I: ArrayLike,
    measured_specific_force_I: ArrayLike,
    gravity_world: ArrayLike,
    spatial_twist_zero: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], AccelerometerFactorTerms]:
    '''Calculate analytic Jacobian blocks of the simple accelerometer factor.

    Translation columns are zero because this factor uses only gravity
    alignment and ignores the IMU lever arm and body translational acceleration.
    H_tau_I is obtained by applying the pose Jacobian to the spatial trajectory
    twist at the queried timestamp.

    Args:
        T_W_B (array 4x4): body pose in world RF
        T_B_I (array 4x4): IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        spatial_twist_zero (array 6): spatial pose derivative at factor time

    Returns:
        tuple: A tuple containing:
            H_T (array 3x6): Jacobian with respect to T_W_B
            H_T_B_I (array 3x6): Jacobian with respect to T_B_I
            H_tau_I (array 3x1): Jacobian with respect to temporal offset
            terms (AccelerometerFactorTerms): residual intermediate values
    '''

    # Validate factor inputs and convert the spatial twist to a 6D vector.
    T = _as_transform(T_W_B, "T_W_B")
    X = _as_transform(T_B_I, "T_B_I")
    measured = _as_vector3(measured_specific_force_I, "measured_specific_force_I")
    gravity = _as_vector3(gravity_world, "gravity_world")
    xi_s = np.asarray(spatial_twist_zero, dtype=float).reshape(6)

    # Extract rotations and gravity expressed in the body reference frame.
    R = T[:3, :3]
    C = X[:3, :3]
    y_g = -R.T @ gravity

    # Assemble pose and extrinsic blocks in rotation-first tangent ordering.
    H_T_rotation = -C.T @ R.T @ so3_hat(gravity)
    H_T_translation = np.zeros((3, 3))
    H_T = np.hstack([H_T_rotation, H_T_translation])

    H_T_B_I_rotation = C.T @ so3_hat(y_g)
    H_T_B_I_translation = np.zeros((3, 3))
    H_T_B_I = np.hstack([H_T_B_I_rotation, H_T_B_I_translation])

    # A time shift moves the queried pose along the local trajectory twist.
    H_tau_I = (H_T @ xi_s).reshape(3, 1)
    terms = _simple_terms(T, X, measured, gravity, 0.0, 0.0)

    return H_T, H_T_B_I, H_tau_I, terms


##################################################
# Complex three-pose accelerometer factor
##################################################
def complex_accelerometer_terms(
    T_minus: ArrayLike,
    T_zero: ArrayLike,
    T_plus: ArrayLike,
    T_B_I: ArrayLike,
    measured_specific_force_I: ArrayLike,
    gravity_world: ArrayLike,
    support_half_width_seconds: float,
    *,
    sensor_time: float = 0.0,
    tau_I: float = 0.0,
) -> AccelerometerFactorTerms:
    '''Calculate all terms of the complex three-pose accelerometer factor.

    T_minus, T_zero and T_plus correspond to T_W_B(t-h), T_W_B(t) and
    T_W_B(t+h). Central finite differences provide body-origin translational
    acceleration, angular velocity and angular acceleration. The IMU lever arm
    then contributes tangential and centripetal acceleration.

    Args:
        T_minus (array 4x4): body pose at t-h
        T_zero (array 4x4): body pose at t
        T_plus (array 4x4): body pose at t+h
        T_B_I (array 4x4): IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        support_half_width_seconds (float): half-width h of pose support
        sensor_time (float): original measurement timestamp
        tau_I (float): IMU temporal offset

    Returns:
        AccelerometerFactorTerms: prediction, residual and derivatives
    '''

    # Validate pose support, extrinsics and measurement vectors.
    Tm = _as_transform(T_minus, "T_minus")
    T0 = _as_transform(T_zero, "T_zero")
    Tp = _as_transform(T_plus, "T_plus")
    X = _as_transform(T_B_I, "T_B_I")
    measured = _as_vector3(measured_specific_force_I, "measured_specific_force_I")
    gravity = _as_vector3(gravity_world, "gravity_world")

    h = float(support_half_width_seconds)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("support_half_width_seconds must be finite and positive")

    # Extract rotations, translations and the IMU lever arm.
    Rm, R0, Rp = Tm[:3, :3], T0[:3, :3], Tp[:3, :3]
    qm, q0, qp = Tm[:3, 3], T0[:3, 3], Tp[:3, 3]
    C = X[:3, :3]
    ell = X[:3, 3]

    # Estimate body-origin translational and angular motion at the middle pose.
    a_zero_world = (qp - 2.0 * q0 + qm) / h**2
    phi_minus = so3_log(R0.T @ Rm)
    phi_plus = so3_log(R0.T @ Rp)
    omega_zero_body = (phi_plus - phi_minus) / (2.0 * h)
    alpha_zero_body = (phi_plus + phi_minus) / h**2

    # Predict specific force at the body origin and at the displaced IMU point.
    specific_force_body_origin = R0.T @ (a_zero_world - gravity)
    alpha_hat = so3_hat(alpha_zero_body)
    omega_hat = so3_hat(omega_zero_body)
    lever_arm_matrix = alpha_hat + omega_hat @ omega_hat

    tangential_acceleration = alpha_hat @ ell
    centripetal_acceleration = omega_hat @ omega_hat @ ell
    y_zero = specific_force_body_origin + lever_arm_matrix @ ell
    predicted = C.T @ y_zero
    residual = predicted - measured

    # Linearize relative rotations through the inverse left SO(3) Jacobian.
    L_minus = so3_left_jacobian_inverse(phi_minus) @ R0.T
    L_plus = so3_left_jacobian_inverse(phi_plus) @ R0.T

    # Map left pose rotations and translations to body-origin specific force.
    G_s_theta_minus = -(1.0 / h**2) * R0.T @ so3_hat(qm)
    G_s_theta_zero = R0.T @ (
        so3_hat(a_zero_world - gravity)
        + (2.0 / h**2) * so3_hat(q0)
    )
    G_s_theta_plus = -(1.0 / h**2) * R0.T @ so3_hat(qp)

    G_s_rho_minus = (1.0 / h**2) * R0.T
    G_s_rho_zero = -(2.0 / h**2) * R0.T
    G_s_rho_plus = (1.0 / h**2) * R0.T

    # Map pose rotations to the central angular velocity and acceleration.
    G_omega_minus = -(1.0 / (2.0 * h)) * L_minus
    G_omega_zero = (1.0 / (2.0 * h)) * (L_minus - L_plus)
    G_omega_plus = (1.0 / (2.0 * h)) * L_plus

    G_alpha_minus = (1.0 / h**2) * L_minus
    G_alpha_zero = -(1.0 / h**2) * (L_minus + L_plus)
    G_alpha_plus = (1.0 / h**2) * L_plus

    # Map changes of angular motion to lever-arm specific force.
    B_alpha = -so3_hat(ell)
    B_omega = (
        -so3_hat(np.cross(omega_zero_body, ell))
        - so3_hat(omega_zero_body) @ so3_hat(ell)
    )

    return AccelerometerFactorTerms(
        mode="complex",
        sensor_time=float(sensor_time),
        tau_I=float(tau_I),
        t_minus=float(sensor_time + tau_I - h),
        t_zero=float(sensor_time + tau_I),
        t_plus=float(sensor_time + tau_I + h),
        T_minus=Tm,
        T_zero=T0,
        T_plus=Tp,
        T_B_I=X,
        gravity_world=gravity,
        a_zero_world=a_zero_world,
        phi_minus=phi_minus,
        phi_plus=phi_plus,
        omega_zero_body=omega_zero_body,
        alpha_zero_body=alpha_zero_body,
        specific_force_body_origin=specific_force_body_origin,
        lever_arm_matrix=lever_arm_matrix,
        tangential_acceleration=tangential_acceleration,
        centripetal_acceleration=centripetal_acceleration,
        predicted_specific_force=predicted,
        measured_specific_force=measured,
        residual=residual,
        G_s_theta_minus=G_s_theta_minus,
        G_s_theta_zero=G_s_theta_zero,
        G_s_theta_plus=G_s_theta_plus,
        G_s_rho_minus=G_s_rho_minus,
        G_s_rho_zero=G_s_rho_zero,
        G_s_rho_plus=G_s_rho_plus,
        L_minus=L_minus,
        L_plus=L_plus,
        G_omega_minus=G_omega_minus,
        G_omega_zero=G_omega_zero,
        G_omega_plus=G_omega_plus,
        G_alpha_minus=G_alpha_minus,
        G_alpha_zero=G_alpha_zero,
        G_alpha_plus=G_alpha_plus,
        B_omega=B_omega,
        B_alpha=B_alpha,
    )


def complex_accelerometer_residual(
    T_minus: ArrayLike,
    T_zero: ArrayLike,
    T_plus: ArrayLike,
    T_B_I: ArrayLike,
    measured_specific_force_I: ArrayLike,
    gravity_world: ArrayLike,
    support_half_width_seconds: float,
) -> NDArray[np.float64]:
    '''Calculate the complex three-pose accelerometer residual.

    Args:
        T_minus (array 4x4): body pose at t-h
        T_zero (array 4x4): body pose at t
        T_plus (array 4x4): body pose at t+h
        T_B_I (array 4x4): IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        support_half_width_seconds (float): half-width h of pose support

    Returns:
        array 3: predicted minus measured specific force
    '''

    # Reuse the complete term builder to keep one residual implementation.
    terms = complex_accelerometer_terms(
        T_minus,
        T_zero,
        T_plus,
        T_B_I,
        measured_specific_force_I,
        gravity_world,
        support_half_width_seconds,
    )

    return terms.residual


def _complex_pose_block(
    terms: AccelerometerFactorTerms,
    C: NDArray[np.float64],
    G_s_theta: NDArray[np.float64],
    G_s_rho: NDArray[np.float64],
    G_omega: NDArray[np.float64],
    G_alpha: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Assemble one complex pose Jacobian block.

    The rotational part combines direct specific-force sensitivity with the
    angular-velocity and angular-acceleration lever-arm terms. The translational
    part contains body-origin acceleration sensitivity.

    Args:
        terms (AccelerometerFactorTerms): complex factor intermediate terms
        C (array 3x3): IMU orientation in body RF
        G_s_theta (array 3x3): rotation to body-origin force mapping
        G_s_rho (array 3x3): translation to body-origin force mapping
        G_omega (array 3x3): rotation to angular-velocity mapping
        G_alpha (array 3x3): rotation to angular-acceleration mapping

    Returns:
        array 3x6: rotation-first pose Jacobian block
    '''

    # Assemble rotational and translational columns separately for readability.
    H_rotation = C.T @ (
        G_s_theta
        + terms.B_omega @ G_omega
        + terms.B_alpha @ G_alpha
    )
    H_translation = C.T @ G_s_rho

    return np.hstack([H_rotation, H_translation])


def complex_accelerometer_analytic_blocks(
    T_minus: ArrayLike,
    T_zero: ArrayLike,
    T_plus: ArrayLike,
    T_B_I: ArrayLike,
    measured_specific_force_I: ArrayLike,
    gravity_world: ArrayLike,
    support_half_width_seconds: float,
    spatial_twist_minus: ArrayLike,
    spatial_twist_zero: ArrayLike,
    spatial_twist_plus: ArrayLike,
    *,
    sensor_time: float = 0.0,
    tau_I: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], AccelerometerFactorTerms]:
    '''Calculate analytic Jacobian blocks of the complex accelerometer factor.

    Each pose block has shape (3, 6). The extrinsic translation columns describe
    direct lever-arm sensitivity and therefore disappear when angular velocity
    and angular acceleration are zero. The temporal-offset block shifts all
    three support poses along their spatial trajectory twists.

    Args:
        T_minus (array 4x4): body pose at t-h
        T_zero (array 4x4): body pose at t
        T_plus (array 4x4): body pose at t+h
        T_B_I (array 4x4): IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        support_half_width_seconds (float): half-width h of pose support
        spatial_twist_minus (array 6): spatial twist at t-h
        spatial_twist_zero (array 6): spatial twist at t
        spatial_twist_plus (array 6): spatial twist at t+h
        sensor_time (float): original measurement timestamp
        tau_I (float): IMU temporal offset

    Returns:
        tuple: A tuple containing:
            H_minus (array 3x6): Jacobian with respect to T_minus
            H_zero (array 3x6): Jacobian with respect to T_zero
            H_plus (array 3x6): Jacobian with respect to T_plus
            H_T_B_I (array 3x6): Jacobian with respect to T_B_I
            H_tau_I (array 3x1): Jacobian with respect to temporal offset
            terms (AccelerometerFactorTerms): residual intermediate values
    '''

    # Calculate all kinematic terms and derivative mappings first.
    terms = complex_accelerometer_terms(
        T_minus,
        T_zero,
        T_plus,
        T_B_I,
        measured_specific_force_I,
        gravity_world,
        support_half_width_seconds,
        sensor_time=sensor_time,
        tau_I=tau_I,
    )

    C = terms.T_B_I[:3, :3]
    ell = terms.T_B_I[:3, 3]
    y_zero = terms.specific_force_body_origin + terms.lever_arm_matrix @ ell

    # Build Jacobians for the three supporting body poses.
    H_minus = _complex_pose_block(
        terms,
        C,
        terms.G_s_theta_minus,
        terms.G_s_rho_minus,
        terms.G_omega_minus,
        terms.G_alpha_minus,
    )
    H_zero = _complex_pose_block(
        terms,
        C,
        terms.G_s_theta_zero,
        terms.G_s_rho_zero,
        terms.G_omega_zero,
        terms.G_alpha_zero,
    )
    H_plus = _complex_pose_block(
        terms,
        C,
        terms.G_s_theta_plus,
        terms.G_s_rho_plus,
        terms.G_omega_plus,
        terms.G_alpha_plus,
    )

    # Extrinsic rotation changes both sensor orientation and the body lever arm.
    H_T_B_I_rotation = (
        so3_hat(y_zero)
        - terms.lever_arm_matrix @ so3_hat(ell)
    )
    H_T_B_I_translation = terms.lever_arm_matrix
    H_T_B_I = C.T @ np.hstack([
        H_T_B_I_rotation,
        H_T_B_I_translation,
    ])

    # Shift every support pose according to its spatial trajectory twist.
    xi_minus = np.asarray(spatial_twist_minus, dtype=float).reshape(6)
    xi_zero = np.asarray(spatial_twist_zero, dtype=float).reshape(6)
    xi_plus = np.asarray(spatial_twist_plus, dtype=float).reshape(6)

    H_tau_I = (
        H_minus @ xi_minus
        + H_zero @ xi_zero
        + H_plus @ xi_plus
    ).reshape(3, 1)

    return H_minus, H_zero, H_plus, H_T_B_I, H_tau_I, terms


##################################################
# Finite-difference Jacobian references
##################################################
def _finite_difference_simple_blocks(
    T_W_B: NDArray[np.float64],
    T_B_I: NDArray[np.float64],
    measured_specific_force_I: NDArray[np.float64],
    gravity_world: NDArray[np.float64],
    sensor_time: float,
    tau_I: float,
    pose_provider: object | None,
    epsilon: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''Calculate finite-difference blocks of the simple accelerometer factor.

    The temporal-offset Jacobian is zero when no trajectory pose provider is
    supplied because the fixed T_W_B input cannot be shifted in time.

    Args:
        T_W_B (array 4x4): validated body pose in world RF
        T_B_I (array 4x4): validated IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        sensor_time (float): original measurement timestamp
        tau_I (float): IMU temporal offset
        pose_provider (object or None): trajectory pose query interface
        epsilon (float): central finite-difference step

    Returns:
        tuple: A tuple containing:
            H_T (array 3x6): numerical pose Jacobian
            H_T_B_I (array 3x6): numerical extrinsic Jacobian
            H_tau (array 3x1): numerical temporal-offset Jacobian
    '''

    # Perturb the body pose and IMU extrinsics independently on SE(3).
    H_T = finite_difference_left_jacobian_se3(
        lambda Tp: simple_accelerometer_residual(
            Tp,
            T_B_I,
            measured_specific_force_I,
            gravity_world,
        ),
        T_W_B,
        epsilon,
    )
    H_T_B_I = finite_difference_left_jacobian_se3(
        lambda Xp: simple_accelerometer_residual(
            T_W_B,
            Xp,
            measured_specific_force_I,
            gravity_world,
        ),
        T_B_I,
        epsilon,
    )

    # Query a shifted trajectory pose only when a provider is available.
    if pose_provider is None:
        H_tau = np.zeros((3, 1), dtype=float)
    else:
        H_tau = central_difference_vector(
            lambda value: simple_accelerometer_residual(
                pose_provider.poses_at(
                    np.array([sensor_time + float(value[0])])
                )[0],
                T_B_I,
                measured_specific_force_I,
                gravity_world,
            ),
            np.array([tau_I], dtype=float),
            epsilon,
        )

    return H_T, H_T_B_I, H_tau


def _finite_difference_complex_blocks(
    T_minus: NDArray[np.float64],
    T_zero: NDArray[np.float64],
    T_plus: NDArray[np.float64],
    T_B_I: NDArray[np.float64],
    measured_specific_force_I: NDArray[np.float64],
    gravity_world: NDArray[np.float64],
    support_half_width_seconds: float,
    sensor_time: float,
    tau_I: float,
    pose_provider: object | None,
    epsilon: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''Calculate finite-difference blocks of the complex accelerometer factor.

    Each supporting pose and the IMU extrinsics are perturbed separately. The
    temporal-offset block shifts all three trajectory query times together.

    Args:
        T_minus (array 4x4): validated body pose at t-h
        T_zero (array 4x4): validated body pose at t
        T_plus (array 4x4): validated body pose at t+h
        T_B_I (array 4x4): validated IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        support_half_width_seconds (float): half-width h of pose support
        sensor_time (float): original measurement timestamp
        tau_I (float): IMU temporal offset
        pose_provider (object or None): trajectory pose query interface
        epsilon (float): central finite-difference step

    Returns:
        tuple: A tuple containing:
            H_minus (array 3x6): numerical Jacobian for T_minus
            H_zero (array 3x6): numerical Jacobian for T_zero
            H_plus (array 3x6): numerical Jacobian for T_plus
            H_T_B_I (array 3x6): numerical extrinsic Jacobian
            H_tau (array 3x1): numerical temporal-offset Jacobian
    '''

    h = float(support_half_width_seconds)

    # Perturb each supporting pose while keeping the remaining inputs fixed.
    H_minus = finite_difference_left_jacobian_se3(
        lambda Tp: complex_accelerometer_residual(
            Tp,
            T_zero,
            T_plus,
            T_B_I,
            measured_specific_force_I,
            gravity_world,
            h,
        ),
        T_minus,
        epsilon,
    )
    H_zero = finite_difference_left_jacobian_se3(
        lambda Tp: complex_accelerometer_residual(
            T_minus,
            Tp,
            T_plus,
            T_B_I,
            measured_specific_force_I,
            gravity_world,
            h,
        ),
        T_zero,
        epsilon,
    )
    H_plus = finite_difference_left_jacobian_se3(
        lambda Tp: complex_accelerometer_residual(
            T_minus,
            T_zero,
            Tp,
            T_B_I,
            measured_specific_force_I,
            gravity_world,
            h,
        ),
        T_plus,
        epsilon,
    )

    # Perturb IMU extrinsics after all pose blocks are evaluated.
    H_T_B_I = finite_difference_left_jacobian_se3(
        lambda Xp: complex_accelerometer_residual(
            T_minus,
            T_zero,
            T_plus,
            Xp,
            measured_specific_force_I,
            gravity_world,
            h,
        ),
        T_B_I,
        epsilon,
    )

    # Shift all three pose support times through the trajectory provider.
    if pose_provider is None:
        H_tau = np.zeros((3, 1), dtype=float)
    else:
        H_tau = central_difference_vector(
            lambda value: complex_accelerometer_residual(
                *pose_provider.poses_at(
                    np.array([
                        sensor_time + float(value[0]) - h,
                        sensor_time + float(value[0]),
                        sensor_time + float(value[0]) + h,
                    ])
                ),
                T_B_I,
                measured_specific_force_I,
                gravity_world,
                h,
            ),
            np.array([tau_I], dtype=float),
            epsilon,
        )

    return H_minus, H_zero, H_plus, H_T_B_I, H_tau


##################################################
# Analytic and numerical Jacobian comparison
##################################################
def _checked_blocks(
    analytic_blocks: tuple[NDArray[np.float64], ...],
    finite_difference_blocks: tuple[NDArray[np.float64], ...],
    block_names: tuple[str, ...],
    factor_name: str,
    options: JacobianOptions,
) -> tuple[JacobianCheckResult, ...]:
    '''Compare corresponding analytic and finite-difference Jacobian blocks.

    Args:
        analytic_blocks (tuple of arrays): analytic Jacobian blocks
        finite_difference_blocks (tuple of arrays): numerical references
        block_names (tuple of str): variable name assigned to each block
        factor_name (str): factor identifier included in check results
        options (JacobianOptions): tolerances and failure behavior

    Returns:
        tuple: one JacobianCheckResult for every compared block
    '''

    # Compare blocks in the same order as their variable names.
    results = tuple(
        compare_jacobians(
            analytic,
            finite_difference,
            factor_name=factor_name,
            variable_name=name,
            atol=options.check_atol,
            rtol=options.check_rtol,
        )
        for name, analytic, finite_difference in zip(
            block_names,
            analytic_blocks,
            finite_difference_blocks,
        )
    )

    # Raise one compact error containing all failed variable blocks when requested.
    if options.raise_on_check_failure and not all(result.passed for result in results):
        details = ", ".join(
            f"{result.variable_name}: {result.max_absolute_error:.3e}"
            for result in results
            if not result.passed
        )
        raise JacobianCheckError(
            f"accelerometer Jacobian check failed for {factor_name}: {details}"
        )

    return results


##################################################
# Factor linearization interface
##################################################
def linearize_simple_accelerometer_factor(
    T_W_B: ArrayLike,
    T_B_I: ArrayLike,
    measured_specific_force_I: ArrayLike,
    gravity_world: ArrayLike,
    spatial_twist_zero: ArrayLike,
    *,
    pose_provider: object | None = None,
    sensor_time: float = 0.0,
    tau_I: float = 0.0,
    jacobian_options: JacobianOptions | None = None,
    factor_name: str = "accel_simple",
) -> AccelerometerFactorLinearization:
    '''Linearize the simple gravity-alignment accelerometer factor.

    Analytic and finite-difference Jacobians are always calculated. The selected
    JacobianOptions method decides which blocks are returned and whether the two
    variants are compared.

    Args:
        T_W_B (array 4x4): body pose in world RF
        T_B_I (array 4x4): IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        spatial_twist_zero (array 6): spatial twist at factor time
        pose_provider (object or None): trajectory pose query interface
        sensor_time (float): original measurement timestamp
        tau_I (float): IMU temporal offset
        jacobian_options (JacobianOptions or None): Jacobian calculation mode
        factor_name (str): identifier included in check results

    Returns:
        AccelerometerFactorLinearization: residual and blocks
    '''

    # Normalize configuration and validate all direct factor inputs.
    options = normalized_jacobian_options(jacobian_options)
    T = _as_transform(T_W_B, "T_W_B")
    X = _as_transform(T_B_I, "T_B_I")
    measured = _as_vector3(measured_specific_force_I, "measured_specific_force_I")
    gravity = _as_vector3(gravity_world, "gravity_world")

    # Calculate analytic blocks and terms carrying the actual factor timestamp.
    H_T_a, H_X_a, H_tau_a, _ = simple_accelerometer_analytic_blocks(
        T,
        X,
        measured,
        gravity,
        spatial_twist_zero,
    )
    terms = _simple_terms(T, X, measured, gravity, sensor_time, tau_I)
    analytic_blocks = (H_T_a, H_X_a, H_tau_a)

    # Calculate finite-difference references for selection or validation.
    fd_blocks = _finite_difference_simple_blocks(
        T,
        X,
        measured,
        gravity,
        sensor_time,
        tau_I,
        pose_provider,
        options.finite_difference_epsilon,
    )

    # Select returned blocks and optionally compare analytic against numerical values.
    check_results: tuple[JacobianCheckResult, ...] = ()
    if options.method == "finite_difference":
        H_T, H_X, H_tau = fd_blocks
    else:
        H_T, H_X, H_tau = analytic_blocks
        if options.method == "analytic_checked":
            check_results = _checked_blocks(
                analytic_blocks,
                fd_blocks,
                ("T_zero", "T_B_I", "tau_I"),
                factor_name,
                options,
            )

    # Summarize optional check results for convenient inspection.
    max_error = max(
        (result.max_absolute_error for result in check_results),
        default=None,
    )
    all_passed = (
        None
        if not check_results
        else all(result.passed for result in check_results)
    )

    return AccelerometerFactorLinearization(
        residual=terms.residual,
        pose_blocks=(AccelerometerPoseBlock("zero", H_T),),
        H_T_B_I=H_X,
        H_tau_I=H_tau,
        check_results=check_results,
        maximum_check_error=max_error,
        all_checks_passed=all_passed,
        terms=terms,
    )


def linearize_complex_accelerometer_factor(
    T_minus: ArrayLike,
    T_zero: ArrayLike,
    T_plus: ArrayLike,
    T_B_I: ArrayLike,
    measured_specific_force_I: ArrayLike,
    gravity_world: ArrayLike,
    support_half_width_seconds: float,
    spatial_twist_minus: ArrayLike,
    spatial_twist_zero: ArrayLike,
    spatial_twist_plus: ArrayLike,
    *,
    pose_provider: object | None = None,
    sensor_time: float = 0.0,
    tau_I: float = 0.0,
    jacobian_options: JacobianOptions | None = None,
    factor_name: str = "accel_complex",
) -> AccelerometerFactorLinearization:
    '''Linearize the complex three-pose accelerometer factor.

    Analytic and finite-difference Jacobians are always calculated. The selected
    JacobianOptions method decides which blocks are returned and whether the two
    variants are compared.

    Args:
        T_minus (array 4x4): body pose at t-h
        T_zero (array 4x4): body pose at t
        T_plus (array 4x4): body pose at t+h
        T_B_I (array 4x4): IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        support_half_width_seconds (float): half-width h of pose support
        spatial_twist_minus (array 6): spatial twist at t-h
        spatial_twist_zero (array 6): spatial twist at t
        spatial_twist_plus (array 6): spatial twist at t+h
        pose_provider (object or None): trajectory pose query interface
        sensor_time (float): original measurement timestamp
        tau_I (float): IMU temporal offset
        jacobian_options (JacobianOptions or None): Jacobian calculation mode
        factor_name (str): identifier included in check results

    Returns:
        AccelerometerFactorLinearization: residual and blocks
    '''

    # Normalize configuration and validate the complete three-pose support.
    options = normalized_jacobian_options(jacobian_options)
    Tm = _as_transform(T_minus, "T_minus")
    T0 = _as_transform(T_zero, "T_zero")
    Tp = _as_transform(T_plus, "T_plus")
    X = _as_transform(T_B_I, "T_B_I")
    measured = _as_vector3(measured_specific_force_I, "measured_specific_force_I")
    gravity = _as_vector3(gravity_world, "gravity_world")

    # Calculate analytic blocks together with the complex kinematic terms.
    analytic_blocks_with_terms = complex_accelerometer_analytic_blocks(
        Tm,
        T0,
        Tp,
        X,
        measured,
        gravity,
        support_half_width_seconds,
        spatial_twist_minus,
        spatial_twist_zero,
        spatial_twist_plus,
        sensor_time=sensor_time,
        tau_I=tau_I,
    )
    Hm_a, H0_a, Hp_a, HX_a, Htau_a, terms = analytic_blocks_with_terms
    analytic_blocks = (Hm_a, H0_a, Hp_a, HX_a, Htau_a)

    # Calculate finite-difference references for selection or validation.
    fd_blocks = _finite_difference_complex_blocks(
        Tm,
        T0,
        Tp,
        X,
        measured,
        gravity,
        support_half_width_seconds,
        sensor_time,
        tau_I,
        pose_provider,
        options.finite_difference_epsilon,
    )

    # Select returned blocks and optionally compare analytic against numerical values.
    check_results: tuple[JacobianCheckResult, ...] = ()
    if options.method == "finite_difference":
        Hm, H0, Hp, HX, Htau = fd_blocks
    else:
        Hm, H0, Hp, HX, Htau = analytic_blocks
        if options.method == "analytic_checked":
            check_results = _checked_blocks(
                analytic_blocks,
                fd_blocks,
                ("T_minus", "T_zero", "T_plus", "T_B_I", "tau_I"),
                factor_name,
                options,
            )

    # Summarize optional check results for convenient inspection.
    max_error = max(
        (result.max_absolute_error for result in check_results),
        default=None,
    )
    all_passed = (
        None
        if not check_results
        else all(result.passed for result in check_results)
    )

    return AccelerometerFactorLinearization(
        residual=terms.residual,
        pose_blocks=(
            AccelerometerPoseBlock("minus", Hm),
            AccelerometerPoseBlock("zero", H0),
            AccelerometerPoseBlock("plus", Hp),
        ),
        H_T_B_I=HX,
        H_tau_I=Htau,
        check_results=check_results,
        maximum_check_error=max_error,
        all_checks_passed=all_passed,
        terms=terms,
    )


def linearize_accelerometer_factor(
    mode: str,
    poses: tuple[ArrayLike, ...],
    T_B_I: ArrayLike,
    measured_specific_force_I: ArrayLike,
    gravity_world: ArrayLike,
    support_half_width_seconds: float,
    spatial_twists: tuple[ArrayLike, ...],
    *,
    pose_provider: object | None = None,
    sensor_time: float = 0.0,
    tau_I: float = 0.0,
    jacobian_options: JacobianOptions | None = None,
    factor_name: str = "accel",
) -> AccelerometerFactorLinearization:
    '''Dispatch accelerometer factor linearization according to selected mode.

    Simple mode requires one pose and one spatial twist. Complex mode requires
    three poses and three spatial twists ordered as minus, zero and plus.

    Args:
        mode (str): simple, complex or disabled accelerometer mode
        poses (tuple of arrays): one or three body poses depending on mode
        T_B_I (array 4x4): IMU pose in body RF
        measured_specific_force_I (array 3): measured specific force in IMU RF
        gravity_world (array 3): gravity vector in world RF
        support_half_width_seconds (float): half-width h of pose support
        spatial_twists (tuple of arrays): one or three trajectory twists
        pose_provider (object or None): trajectory pose query interface
        sensor_time (float): original measurement timestamp
        tau_I (float): IMU temporal offset
        jacobian_options (JacobianOptions or None): Jacobian calculation mode
        factor_name (str): identifier included in check results

    Returns:
        AccelerometerFactorLinearization: residual and blocks
    '''

    # Validate the requested mode before checking mode-specific support sizes.
    selected = validate_accelerometer_mode(mode)

    if selected == "simple":
        if len(poses) != 1 or len(spatial_twists) != 1:
            raise ValueError(
                "simple accelerometer factor requires one pose and one spatial twist"
            )

        return linearize_simple_accelerometer_factor(
            poses[0],
            T_B_I,
            measured_specific_force_I,
            gravity_world,
            spatial_twists[0],
            pose_provider=pose_provider,
            sensor_time=sensor_time,
            tau_I=tau_I,
            jacobian_options=jacobian_options,
            factor_name=factor_name,
        )

    if selected == "complex":
        if len(poses) != 3 or len(spatial_twists) != 3:
            raise ValueError(
                "complex accelerometer factor requires three poses and three spatial twists"
            )

        return linearize_complex_accelerometer_factor(
            poses[0],
            poses[1],
            poses[2],
            T_B_I,
            measured_specific_force_I,
            gravity_world,
            support_half_width_seconds,
            spatial_twists[0],
            spatial_twists[1],
            spatial_twists[2],
            pose_provider=pose_provider,
            sensor_time=sensor_time,
            tau_I=tau_I,
            jacobian_options=jacobian_options,
            factor_name=factor_name,
        )

    raise ValueError("disabled mode has no accelerometer factor to linearize")


##################################################
# Public module interface
##################################################
__all__ = [
    "AccelerometerFactorLinearization",
    "AccelerometerFactorTerms",
    "AccelerometerPoseBlock",
    "complex_accelerometer_residual",
    "complex_accelerometer_terms",
    "complex_accelerometer_analytic_blocks",
    "linearize_accelerometer_factor",
    "linearize_complex_accelerometer_factor",
    "linearize_simple_accelerometer_factor",
    "simple_accelerometer_residual",
    "simple_accelerometer_analytic_blocks",
]