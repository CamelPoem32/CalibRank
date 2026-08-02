'''Central mathematical conventions and validators.

The package follows ``phd_proposal_draft.tex`` and uses homogeneous
transformations, rotation-first SE(3) tangent vectors and left perturbations:

    T_perturbed = Exp(delta_xi) @ T

MROB follows the same SE(3) tangent ordering, ``xi = [omega, v]``, and exposes
left-side updates through ``SE3.update_lhs``. Tangent conversion between MROB
and this package therefore validates the input without permuting components.
'''

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


##################################################
# Package-wide mathematical conventions
##################################################
SE3_TANGENT_ORDER: tuple[str, ...] = (
    "omega_x",
    "omega_y",
    "omega_z",
    "v_x",
    "v_y",
    "v_z",
)
SE2_TANGENT_ORDER: tuple[str, ...] = ("omega", "v_x", "v_y")
LEFT_PERTURBATION = "T_perturbed = Exp(delta_xi) @ T"


##################################################
# Array validation helpers
##################################################
def as_vector(x: ArrayLike, dim: int, name: str) -> NDArray[np.float64]:
    '''Convert an input value to a finite vector of the requested dimension.

    Args:
        x: Array-like tangent, state or residual vector.
        dim: Expected vector dimension.
        name: Argument name included in validation errors.

    Returns:
        Floating-point vector with shape ``(dim,)``.

    Raises:
        ValueError: If ``x`` does not have shape ``(dim,)`` or contains
            non-finite values.
    '''

    # Convert to the package floating-point representation before validation.
    arr = np.asarray(x, dtype=float)

    # Keep vector shape strict to avoid silently flattening matrices or batches.
    if arr.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")

    return arr


def as_matrix(
    x: ArrayLike,
    shape: tuple[int, int],
    name: str,
) -> NDArray[np.float64]:
    '''Convert an input value to a finite matrix of the requested shape.

    Args:
        x: Array-like matrix.
        shape: Expected matrix shape ``(rows, columns)``.
        name: Argument name included in validation errors.

    Returns:
        Floating-point matrix with the requested shape.

    Raises:
        ValueError: If ``x`` has an incorrect shape or contains non-finite
            values.
    '''

    # Convert first so all later computations receive one numerical dtype.
    arr = np.asarray(x, dtype=float)

    # Validate shape and numerical contents independently for clearer errors.
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")

    return arr


def ensure_same_length(arrays: Iterable[ArrayLike], names: Iterable[str]) -> None:
    '''Validate that all arrays have the same first dimension.

    Args:
        arrays: Arrays whose first dimensions must match.
        names: Names paired with ``arrays`` for the validation message.

    Raises:
        ValueError: If the arrays do not share the same first dimension.
    '''

    # Read only the first dimension because trailing dimensions may differ.
    lengths = [np.asarray(array).shape[0] for array in arrays]

    # Report every supplied name and its observed length when a mismatch exists.
    if len(set(lengths)) > 1:
        joined = ", ".join(
            f"{name}={length}"
            for name, length in zip(names, lengths)
        )
        raise ValueError(
            f"arrays must have matching first dimensions: {joined}"
        )


##################################################
# MROB tangent conversion
##################################################
def tangent_from_mrob(x: ArrayLike) -> NDArray[np.float64]:
    '''Convert an MROB SE(3) tangent vector to package ordering.

    MROB defines its six-dimensional tangent vector as ``xi = [omega, v]``:
    the first three entries are rotational and the final three entries are
    translational. This is identical to ``SE3_TANGENT_ORDER``, so no component
    permutation is required.

    Args:
        x: MROB tangent vector ordered as
            ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]``.

    Returns:
        Validated tangent vector in package rotation-first ordering, with shape
        ``(6,)``.

    Raises:
        ValueError: If ``x`` does not have shape ``(6,)`` or contains
            non-finite values.
    '''

    # MROB and the package use the same [omega, v] component ordering.
    return as_vector(x, len(SE3_TANGENT_ORDER), "x")


def tangent_to_mrob(xi: ArrayLike) -> NDArray[np.float64]:
    '''Convert a package SE(3) tangent vector to MROB ordering.

    The package and MROB both use rotation-first tangents
    ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]``. MROB's ``update_lhs`` also
    matches the package left-perturbation convention. Conversion is therefore
    an identity mapping after shape and finite-value validation.

    Args:
        xi: Package tangent vector in rotation-first ordering.

    Returns:
        Validated tangent vector accepted by MROB, with shape ``(6,)``.

    Raises:
        ValueError: If ``xi`` does not have shape ``(6,)`` or contains
            non-finite values.
    '''

    # Preserve the component order expected by MROB's SE3 constructor and updates.
    return as_vector(xi, len(SE3_TANGENT_ORDER), "xi")