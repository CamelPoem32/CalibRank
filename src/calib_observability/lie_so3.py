'''SO(3) Lie-group utilities.

All functions use 3x3 rotation matrices and rotation vectors
``omega = [omega_x, omega_y, omega_z]``. The module follows the package-wide
left-perturbation convention and uses MROB only when its Python geometry
bindings pass an isolated safety probe.
'''

from __future__ import annotations

from functools import lru_cache
import subprocess
import sys

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import as_matrix, as_vector

try:  # pragma: no cover - exercised only when mrob is installed.
    import mrob as _mrob
except Exception:  # pragma: no cover - optional dependency.
    _mrob = None

_EPS = 1e-12
_MROB_PROBE_TIMEOUT_SECONDS = 2.0


##################################################
# Optional MROB geometry backend
##################################################
def _mrob_probe_code(group_name: str) -> str | None:
    '''Return child-process code used to probe one MROB geometry binding.

    Args:
        group_name: Lie-group name, currently ``"SO3"`` or ``"SE3"``.

    Returns:
        Python source for the requested safety probe, or ``None`` when the
        group name is unsupported.
    '''

    normalized_group_name = group_name.upper()

    if normalized_group_name == "SO3":
        return (
            "import mrob, numpy as np; "
            "omega=np.array([0.01,-0.02,0.03], dtype=np.float64); "
            "rotation=mrob.SO3(omega); rotation.R(); rotation.Ln(); rotation.adj(); "
            "mrob.hat3(omega); "
            "getattr(mrob, 'left_jacobian_SO3', lambda x: np.eye(3))(omega); "
            "getattr(mrob, 'inv_left_jacobian_SO3', lambda x: np.eye(3))(omega)"
        )

    if normalized_group_name == "SE3":
        return (
            "import mrob, numpy as np; "
            "xi=np.array([0.01,-0.02,0.03,0.1,-0.2,0.3], dtype=np.float64); "
            "transform=mrob.SE3(xi); transform.T(); transform.Ln(); transform.adj(); "
            "mrob.hat6(xi)"
        )

    return None


@lru_cache(maxsize=None)
def _mrob_geometry_is_usable(group_name: str = "SO3") -> bool:
    '''Check whether an installed MROB geometry binding is safe to call.

    Some Windows and Python combinations expose the :mod:`mrob` module but can
    terminate the interpreter when SO(3) or SE(3) constructors receive NumPy
    arrays. The probe therefore runs in a child process and its result is
    cached for the remainder of the program.

    Args:
        group_name: Geometry binding to test, ``"SO3"`` or ``"SE3"``.

    Returns:
        ``True`` when the child process completes successfully, otherwise
        ``False``.
    '''

    if _mrob is None:
        return False

    probe_code = _mrob_probe_code(group_name)
    if probe_code is None:
        return False

    # Keep a potentially unsafe binary binding outside the main process.
    try:
        completed_process = subprocess.run(
            [sys.executable, "-c", probe_code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_MROB_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except Exception:
        return False

    return completed_process.returncode == 0


def _mrob_so3_from_rotation(R: ArrayLike):
    '''Construct an MROB SO(3) object from a rotation matrix when possible.

    Args:
        R: Rotation matrix accepted by the MROB constructor.

    Returns:
        An MROB ``SO3`` instance, or ``None`` when the backend is unavailable
        or rejects the input.
    '''

    if not _mrob_geometry_is_usable("SO3"):
        return None

    try:
        return _mrob.SO3(np.asarray(R, dtype=np.float64))
    except Exception:
        return None


def _mrob_so3_from_tangent(omega: ArrayLike):
    '''Construct an MROB SO(3) object from a rotation vector when possible.

    Args:
        omega: Rotation vector accepted by the MROB constructor.

    Returns:
        An MROB ``SO3`` instance, or ``None`` when the backend is unavailable
        or rejects the input.
    '''

    if not _mrob_geometry_is_usable("SO3"):
        return None

    try:
        return _mrob.SO3(np.asarray(omega, dtype=np.float64))
    except Exception:
        return None


##################################################
# SO(3) algebra mappings
##################################################
def so3_hat(omega: ArrayLike) -> NDArray[np.float64]:
    '''Map a rotation vector to its 3x3 skew-symmetric matrix.

    Args:
        omega: Rotation vector with shape ``(3,)``.

    Returns:
        Skew-symmetric matrix ``omega^`` with shape ``(3, 3)``.

    Raises:
        ValueError: If ``omega`` is not a finite three-dimensional vector.
    '''

    w = as_vector(omega, 3, "omega")

    # Prefer the verified MROB implementation when the binding is available.
    if _mrob_geometry_is_usable("SO3"):
        try:
            return np.asarray(_mrob.hat3(w), dtype=float)
        except Exception:
            pass

    # Explicit fallback used when MROB is absent or unsafe.
    return np.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ],
        dtype=float,
    )


def so3_vee(Omega: ArrayLike) -> NDArray[np.float64]:
    '''Map a 3x3 skew-symmetric matrix to a rotation vector.

    Args:
        Omega: SO(3) algebra matrix with shape ``(3, 3)``.

    Returns:
        Rotation vector ``[Omega_32, Omega_13, Omega_21]`` with shape ``(3,)``.

    Raises:
        ValueError: If ``Omega`` is not a finite 3x3 matrix.
    '''

    M = as_matrix(Omega, (3, 3), "Omega")

    # MROB exposes ``hat3`` but no corresponding Python ``vee`` helper.
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


##################################################
# SO(3) exponential, logarithm and projection
##################################################
def so3_exp(omega: ArrayLike) -> NDArray[np.float64]:
    '''Compute the SO(3) exponential map ``Exp(omega)``.

    Rodrigues' formula is used by the fallback path. Near zero, the scalar
    coefficients are evaluated with truncated Taylor series to avoid division
    by a very small rotation angle.

    Args:
        omega: Rotation vector with shape ``(3,)``.

    Returns:
        Rotation matrix with shape ``(3, 3)``.

    Raises:
        ValueError: If ``omega`` is not a finite three-dimensional vector.
    '''

    w = as_vector(omega, 3, "omega")
    mrob_rotation = _mrob_so3_from_tangent(w)
    if mrob_rotation is not None:
        return np.asarray(mrob_rotation.R(), dtype=float)

    # Build Rodrigues coefficients with a stable small-angle branch.
    theta = float(np.linalg.norm(w))
    W = so3_hat(w)
    W2 = W @ W

    if theta < 1e-8:
        a = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
        b = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / theta**2

    return np.eye(3) + a * W + b * W2


def so3_project_to_rotation(R: ArrayLike) -> NDArray[np.float64]:
    '''Project a near-rotation matrix onto SO(3) using an SVD.

    Args:
        R: Finite matrix with shape ``(3, 3)``.

    Returns:
        Closest proper orthogonal matrix under the Frobenius norm.

    Raises:
        ValueError: If ``R`` is not a finite 3x3 matrix.
    '''

    M = as_matrix(R, (3, 3), "R")

    # Orthogonal Procrustes projection followed by determinant correction.
    U, _, Vt = np.linalg.svd(M)
    R_proj = U @ Vt

    if np.linalg.det(R_proj) < 0.0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt

    return R_proj


def so3_log(R: ArrayLike) -> NDArray[np.float64]:
    '''Compute the principal SO(3) logarithm ``Log(R)``.

    The input is first projected onto SO(3). The fallback implementation uses
    separate branches near zero and near angle pi, where the ordinary
    ``theta / sin(theta)`` expression is numerically fragile.

    Args:
        R: Rotation or near-rotation matrix with shape ``(3, 3)``.

    Returns:
        Principal rotation vector with shape ``(3,)``.

    Raises:
        ValueError: If ``R`` is not a finite 3x3 matrix.
    '''

    Rm = so3_project_to_rotation(R)
    mrob_rotation = _mrob_so3_from_rotation(Rm)
    if mrob_rotation is not None:
        return np.asarray(mrob_rotation.Ln(), dtype=float).reshape(3)

    # Recover the principal rotation angle from the matrix trace.
    cos_theta = np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    # Around identity, the antisymmetric part is already first-order accurate.
    if theta < 1e-8:
        return so3_vee(0.5 * (Rm - Rm.T))

    # Around pi, extract a stable axis from the dominant diagonal element.
    if np.pi - theta < 1e-6:
        A = (Rm + np.eye(3)) / 2.0
        axis = np.empty(3)
        idx = int(np.argmax(np.diag(A)))
        axis[idx] = np.sqrt(max(A[idx, idx], 0.0))
        j = (idx + 1) % 3
        k = (idx + 2) % 3

        if axis[idx] > _EPS:
            axis[j] = A[j, idx] / axis[idx]
            axis[k] = A[k, idx] / axis[idx]
        else:
            axis = np.array([1.0, 0.0, 0.0])

        axis = axis / np.linalg.norm(axis)
        return theta * axis

    return theta / (2.0 * np.sin(theta)) * so3_vee(Rm - Rm.T)


##################################################
# SO(3) left Jacobians
##################################################
def so3_left_jacobian(omega: ArrayLike) -> NDArray[np.float64]:
    '''Compute the SO(3) left Jacobian ``J_l(omega)``.

    Args:
        omega: Rotation vector with shape ``(3,)``.

    Returns:
        Left Jacobian with shape ``(3, 3)``.

    Raises:
        ValueError: If ``omega`` is not a finite three-dimensional vector.
    '''

    w = as_vector(omega, 3, "omega")

    if _mrob_geometry_is_usable("SO3") and hasattr(_mrob, "left_jacobian_SO3"):
        try:
            return np.asarray(_mrob.left_jacobian_SO3(w), dtype=float)
        except Exception:
            pass

    # Evaluate the closed form with stable coefficients around zero.
    theta = float(np.linalg.norm(w))
    W = so3_hat(w)
    W2 = W @ W

    if theta < 1e-8:
        a = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
        b = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040.0
    else:
        a = (1.0 - np.cos(theta)) / theta**2
        b = (theta - np.sin(theta)) / theta**3

    return np.eye(3) + a * W + b * W2


def so3_left_jacobian_inverse(omega: ArrayLike) -> NDArray[np.float64]:
    '''Compute the inverse SO(3) left Jacobian ``J_l^{-1}(omega)``.

    Args:
        omega: Rotation vector with shape ``(3,)``.

    Returns:
        Inverse left Jacobian with shape ``(3, 3)``.

    Raises:
        ValueError: If ``omega`` is not a finite three-dimensional vector.
    '''

    w = as_vector(omega, 3, "omega")

    if _mrob_geometry_is_usable("SO3") and hasattr(
        _mrob,
        "inv_left_jacobian_SO3",
    ):
        try:
            return np.asarray(_mrob.inv_left_jacobian_SO3(w), dtype=float)
        except Exception:
            pass

    # Use the Bernoulli-series limit near zero and the closed form elsewhere.
    theta = float(np.linalg.norm(w))
    W = so3_hat(w)
    W2 = W @ W

    if theta < 1e-8:
        return np.eye(3) - 0.5 * W + (1.0 / 12.0) * W2

    half = 0.5 * theta
    cot_half = 1.0 / np.tan(half)
    coeff = (1.0 - 0.5 * theta * cot_half) / theta**2

    return np.eye(3) - 0.5 * W + coeff * W2