'''Parameter scaling for observability Jacobians.'''

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse


##################################################
# Parameter scale configuration
##################################################
@dataclass(frozen=True)
class ParameterScales:
    '''Store characteristic parameter scales used for observability analysis.
    
    The scaling matrix satisfies ``delta_x = D @ delta_y`` and changes numerical
    units without changing the package left-perturbation convention.
    
    Attributes:
        rotation_scale_rad: Characteristic rotation scale in radians.
        translation_scale_m: Characteristic translation scale in metres.
        gyro_bias_scale_rad_s: Characteristic gyroscope-bias scale in radians per second.
        time_offset_scale_s: Characteristic temporal-offset scale in seconds.
    '''

    rotation_scale_rad: float = 1.0
    translation_scale_m: float = 1.0
    gyro_bias_scale_rad_s: float = 1.0
    time_offset_scale_s: float = 1.0

    def validate(self) -> None:
        '''Validate that every parameter scale is finite and positive.
        
        Raises:
            ValueError: If any stored scale is non-finite or non-positive.
        '''
        vals = np.array(
            [
                self.rotation_scale_rad,
                self.translation_scale_m,
                self.gyro_bias_scale_rad_s,
                self.time_offset_scale_s,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(vals)) or np.any(vals <= 0.0):
            raise ValueError("all parameter scales must be finite and positive")


##################################################
# Variable-block scale selection
##################################################
def _block_scale_values(block: object, scales: ParameterScales) -> NDArray[np.float64]:
    '''Return diagonal scaling values for one variable block.
    
    Args:
        block: Object with ``name`` and ``dimension`` attributes, or a
            ``(name, dimension)`` tuple.
        scales: Characteristic scales for supported calibration variables.
    
    Returns:
        One scaling value per coordinate in the block.
    '''
    name = str(getattr(block, "name", block[0] if isinstance(block, tuple) else ""))
    dim = int(getattr(block, "dimension", block[1] if isinstance(block, tuple) else 0))
    lower = name.lower()
    if dim == 6:
        return np.r_[
            np.full(3, scales.rotation_scale_rad),
            np.full(3, scales.translation_scale_m),
        ]
    if dim == 3 and ("b_g" in lower or "bias" in lower or "gyro" in lower):
        return np.full(3, scales.gyro_bias_scale_rad_s)
    if dim == 1 and ("tau" in lower or "time" in lower or "offset" in lower):
        return np.full(1, scales.time_offset_scale_s)
    return np.ones(dim)


##################################################
# Scaling-matrix construction
##################################################
def build_parameter_scaling_dense(
    variable_blocks: Sequence[object], scales: ParameterScales | None = None
) -> NDArray[np.float64]:
    '''Build dense diagonal `D` such that `delta_x = D @ delta_y`.
    
    Args:
        variable_blocks: Blocks with `name` and `dimension`, or `(name, dimension)` tuples.
        scales: Characteristic scales.
    
    Returns:
        ndarray, shape `(n, n)`
    
    Raises:
        ValueError: If scales are invalid.
    
    Notes:
        Perturbation convention: Scaling preserves the left-perturbation tangent coordinates and changes only their numerical units.
    '''

    s = ParameterScales() if scales is None else scales
    s.validate()
    diag = np.concatenate([_block_scale_values(block, s) for block in variable_blocks])
    return np.diag(diag)


def build_parameter_scaling_sparse(
    variable_blocks: Sequence[object], scales: ParameterScales | None = None
) -> sparse.csr_matrix:
    '''Build a sparse diagonal parameter-scaling matrix.
    
    Args:
        variable_blocks: Blocks with ``name`` and ``dimension`` attributes, or
            ``(name, dimension)`` tuples.
        scales: Characteristic scales. Unit scales are used when omitted.
    
    Returns:
        CSR diagonal matrix ``D`` satisfying ``delta_x = D @ delta_y``.
    
    Raises:
        ValueError: If any scale is non-finite or non-positive.
    '''

    s = ParameterScales() if scales is None else scales
    s.validate()
    diag = np.concatenate([_block_scale_values(block, s) for block in variable_blocks])
    return sparse.diags(diag, format="csr")


##################################################
# Jacobian scaling
##################################################
def scale_jacobian_dense(J: ArrayLike, D: ArrayLike) -> NDArray[np.float64]:
    '''Apply parameter scaling to a dense Jacobian.
    
    Args:
        J: Dense Jacobian, shape ``(m, n)``.
        D: Dense square scaling matrix, shape ``(n, n)``.
    
    Returns:
        Scaled Jacobian ``J @ D``, shape ``(m, n)``.
    
    Raises:
        ValueError: If the matrix dimensions are incompatible.
    '''

    A = np.asarray(J, dtype=float)
    S = np.asarray(D, dtype=float)
    if A.ndim != 2 or S.shape != (A.shape[1], A.shape[1]):
        raise ValueError("J must be (m, n) and D must be (n, n)")
    # A: (m, n), S: (n, n) -> J_scaled: (m, n)
    return A @ S


def scale_jacobian_sparse(J: sparse.spmatrix, D: sparse.spmatrix) -> sparse.csr_matrix:
    '''Apply parameter scaling to a sparse Jacobian.
    
    Args:
        J: Sparse Jacobian, shape ``(m, n)``.
        D: Sparse square scaling matrix, shape ``(n, n)``.
    
    Returns:
        Scaled Jacobian ``J @ D`` in CSR format.
    
    Raises:
        ValueError: If either input is dense or the dimensions are incompatible.
    '''

    if not sparse.issparse(J) or not sparse.issparse(D):
        raise ValueError("J and D must be sparse matrices")
    if D.shape != (J.shape[1], J.shape[1]):
        raise ValueError("D must be square with dimension equal to J columns")
    # J: (m, n), D: (n, n) -> J_scaled: (m, n)
    return (J @ D).tocsr()