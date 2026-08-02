'''Dense and sparse Jacobian block assembly.

The module places local residual Jacobians into a common global perturbation
vector and separates the resulting matrix into trajectory and calibration
columns. Dense and sparse assembly follow the same block layout and residual
ordering.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse


# Supported variable and residual classifications.
VariableCategory = Literal["trajectory", "calibration"]
ResidualKind = Literal["measurement", "smoothness", "prior"]


##################################################
# Variable layout structures
##################################################
@dataclass(frozen=True)
class VariableBlock:
    '''Store one variable block in the global perturbation vector.

    Attributes:
        name (str): Unique variable name used to address the block.
        dimension (int): Number of perturbation coordinates in the block.
        sl (slice): Global column slice assigned to the variable.
        category (VariableCategory): Trajectory or calibration partition.
    '''

    name: str
    dimension: int
    sl: slice
    category: VariableCategory


@dataclass(frozen=True)
class VariableLayout:
    '''Store the global variable layout and column partitioning.

    Attributes:
        blocks (tuple[VariableBlock, ...]): Ordered variable blocks.
        total_dim (int): Total number of global perturbation coordinates.
    '''

    blocks: tuple[VariableBlock, ...]
    total_dim: int

    @classmethod
    def from_specs(
        cls,
        specs: list[tuple[str, int, VariableCategory]],
    ) -> "VariableLayout":
        '''Create a variable layout from ordered block specifications.

        Args:
            specs (list[tuple[str, int, VariableCategory]]): Tuples containing
                variable name, dimension and category in global column order.

        Returns:
            VariableLayout: Layout with consecutive global column slices.

        Raises:
            ValueError: If any variable dimension is not positive.
        '''

        blocks: list[VariableBlock] = []
        start = 0

        # Assign one consecutive global column slice to every variable.
        for name, dim, category in specs:
            if dim <= 0:
                raise ValueError("variable dimensions must be positive")

            sl = slice(start, start + dim)
            blocks.append(
                VariableBlock(
                    name=name,
                    dimension=dim,
                    sl=sl,
                    category=category,
                )
            )
            start += dim

        return cls(
            blocks=tuple(blocks),
            total_dim=start,
        )

    def block(self, name: str) -> VariableBlock:
        '''Return a variable block by name.

        Args:
            name (str): Name of the requested variable block.

        Returns:
            VariableBlock: Matching block from the global layout.

        Raises:
            KeyError: If the layout does not contain the requested name.
        '''

        # Preserve layout order while searching for the requested block.
        for block in self.blocks:
            if block.name == name:
                return block

        raise KeyError(name)

    @property
    def trajectory_blocks(self) -> tuple[VariableBlock, ...]:
        '''Return all trajectory blocks in global layout order.

        Returns:
            tuple[VariableBlock, ...]: Variable blocks classified as trajectory.
        '''

        return tuple(
            block
            for block in self.blocks
            if block.category == "trajectory"
        )

    @property
    def calibration_blocks(self) -> tuple[VariableBlock, ...]:
        '''Return all calibration blocks in global layout order.

        Returns:
            tuple[VariableBlock, ...]: Variable blocks classified as calibration.
        '''

        return tuple(
            block
            for block in self.blocks
            if block.category == "calibration"
        )


##################################################
# Residual and Jacobian structures
##################################################
@dataclass(frozen=True)
class ResidualBlock:
    '''Store one block in the stacked residual vector.

    Attributes:
        name (str): Unique residual name used to address the block.
        dimension (int): Number of scalar residual entries in the block.
        row_slice (slice): Global row slice assigned to the residual.
        covariance (NDArray[np.float64]): Residual covariance matrix.
        kind (ResidualKind): Measurement, smoothness or prior classification.
    '''

    name: str
    dimension: int
    row_slice: slice
    covariance: NDArray[np.float64]
    kind: ResidualKind


@dataclass(frozen=True)
class JacobianBlock:
    '''Store one local Jacobian block before global assembly.

    Attributes:
        residual_name (str): Residual block receiving the local rows.
        variable_name (str): Variable block receiving the local columns.
        matrix (NDArray[np.float64] | sparse.spmatrix): Local Jacobian matrix.
    '''

    residual_name: str
    variable_name: str
    matrix: NDArray[np.float64] | sparse.spmatrix


@dataclass(frozen=True)
class JacobianBundle:
    '''Store the assembled Jacobian, partitions and aligned metadata.

    Attributes:
        J (NDArray[np.float64] | sparse.csr_matrix): Complete global Jacobian.
        J_T (NDArray[np.float64] | sparse.csr_matrix): Trajectory columns.
        J_C (NDArray[np.float64] | sparse.csr_matrix): Calibration columns.
        residual (NDArray[np.float64]): Residual vector aligned with Jacobian rows.
        row_slices (dict[str, slice]): Global residual rows keyed by block name.
        trajectory_column_slices (dict[str, slice]): Local slices inside J_T.
        calibration_column_slices (dict[str, slice]): Local slices inside J_C.
        metadata (dict[str, Any]): Additional caller-provided information.
    '''

    J: NDArray[np.float64] | sparse.csr_matrix
    J_T: NDArray[np.float64] | sparse.csr_matrix
    J_C: NDArray[np.float64] | sparse.csr_matrix
    residual: NDArray[np.float64]
    row_slices: dict[str, slice]
    trajectory_column_slices: dict[str, slice]
    calibration_column_slices: dict[str, slice]
    metadata: dict[str, Any]


##################################################
# Block layout construction
##################################################
def make_residual_blocks(
    specs: list[tuple[str, int, ArrayLike, ResidualKind]]
) -> tuple[ResidualBlock, ...]:
    '''Create ordered residual blocks from block specifications.

    Args:
        specs (list[tuple[str, int, ArrayLike, ResidualKind]]): Tuples
            containing residual name, dimension, covariance and kind in global
            row order.

    Returns:
        tuple[ResidualBlock, ...]: Residual blocks with consecutive row slices.

    Raises:
        ValueError: If a residual dimension is not positive or its covariance
            does not have shape ``(dimension, dimension)``.
    '''

    blocks: list[ResidualBlock] = []
    start = 0

    # Validate each covariance and assign a consecutive global row slice.
    for name, dim, cov, kind in specs:
        if dim <= 0:
            raise ValueError("residual dimensions must be positive")

        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.shape != (dim, dim):
            raise ValueError(
                f"covariance for {name} must have shape ({dim}, {dim})"
            )

        blocks.append(
            ResidualBlock(
                name=name,
                dimension=dim,
                row_slice=slice(start, start + dim),
                covariance=cov_arr,
                kind=kind,
            )
        )
        start += dim

    return tuple(blocks)


##################################################
# Internal assembly helpers
##################################################
def _partition_columns(
    layout: VariableLayout,
    category: VariableCategory,
) -> tuple[list[int], dict[str, slice]]:
    '''Collect global columns belonging to one variable category.

    Args:
        layout (VariableLayout): Complete global variable layout.
        category (VariableCategory): Category selected for the partition.

    Returns:
        tuple[list[int], dict[str, slice]]: Global column indices and local
            slices inside the selected partition, keyed by variable name.
    '''

    cols: list[int] = []
    slices: dict[str, slice] = {}
    start = 0

    # Preserve global variable order while compressing selected blocks locally.
    for block in layout.blocks:
        if block.category != category:
            continue

        local = slice(start, start + block.dimension)
        slices[block.name] = local
        cols.extend(
            range(
                block.sl.start or 0,
                block.sl.stop or 0,
            )
        )
        start += block.dimension

    return cols, slices


def _residual_vector(
    residual_blocks: tuple[ResidualBlock, ...],
    residual_values: dict[str, ArrayLike] | None,
) -> NDArray[np.float64]:
    '''Assemble the residual vector in the configured block order.

    Missing residual values remain zero. Values with names that do not appear
    in ``residual_blocks`` are ignored, matching the original assembly
    interface.

    Args:
        residual_blocks (tuple[ResidualBlock, ...]): Ordered residual layout.
        residual_values (dict[str, ArrayLike] | None): Optional residual values
            keyed by residual block name.

    Returns:
        NDArray[np.float64]: Stacked residual vector with one entry per
            Jacobian row.

    Raises:
        ValueError: If a provided residual has an incompatible shape.
    '''

    total_rows = sum(
        block.dimension
        for block in residual_blocks
    )
    residual = np.zeros(total_rows, dtype=float)

    if residual_values is None:
        return residual

    # Insert only residual blocks explicitly provided by the caller.
    for block in residual_blocks:
        if block.name not in residual_values:
            continue

        r = np.asarray(
            residual_values[block.name],
            dtype=float,
        )
        if r.shape != (block.dimension,):
            raise ValueError(
                f"residual {block.name} must have shape ({block.dimension},)"
            )

        residual[block.row_slice] = r

    return residual


def _block_maps(
    layout: VariableLayout,
    residual_blocks: tuple[ResidualBlock, ...],
) -> tuple[dict[str, ResidualBlock], dict[str, VariableBlock], int]:
    '''Create name lookup tables shared by dense and sparse assembly.

    Args:
        layout (VariableLayout): Complete global variable layout.
        residual_blocks (tuple[ResidualBlock, ...]): Ordered residual layout.

    Returns:
        tuple[dict[str, ResidualBlock], dict[str, VariableBlock], int]:
            Residual lookup, variable lookup and total Jacobian row count.
    '''

    row_by_name = {
        block.name: block
        for block in residual_blocks
    }
    var_by_name = {
        block.name: block
        for block in layout.blocks
    }
    total_rows = sum(
        block.dimension
        for block in residual_blocks
    )

    return row_by_name, var_by_name, total_rows


##################################################
# Dense Jacobian assembly
##################################################
def assemble_jacobian_dense(
    layout: VariableLayout,
    residual_blocks: tuple[ResidualBlock, ...],
    jacobian_blocks: list[JacobianBlock],
    residual_values: dict[str, ArrayLike] | None = None,
    metadata: dict[str, Any] | None = None,
) -> JacobianBundle:
    '''Assemble a dense global Jacobian and its variable partitions.

    Local Jacobians are assumed to use left-perturbation tangent coordinates.
    Multiple local blocks assigned to the same global location are accumulated.

    Args:
        layout (VariableLayout): Variable layout with global column slices.
        residual_blocks (tuple[ResidualBlock, ...]): Residual blocks with global
            row slices.
        jacobian_blocks (list[JacobianBlock]): Local Jacobian blocks to insert.
        residual_values (dict[str, ArrayLike] | None): Optional residual vectors
            keyed by residual block name.
        metadata (dict[str, Any] | None): Optional metadata attached to the
            returned bundle.

    Returns:
        JacobianBundle: Dense global Jacobian, trajectory and calibration
            partitions, aligned residual vector, slices and metadata.

    Raises:
        KeyError: If a Jacobian block references an unknown residual or
            variable name.
        ValueError: If a local Jacobian has an incompatible shape.
    '''

    # Prepare name-based block lookup and allocate the complete dense matrix.
    row_by_name, var_by_name, total_rows = _block_maps(
        layout,
        residual_blocks,
    )
    J = np.zeros(
        (total_rows, layout.total_dim),
        dtype=float,
    )

    # Validate and accumulate every local Jacobian at its global block location.
    for block in jacobian_blocks:
        rb = row_by_name[block.residual_name]
        vb = var_by_name[block.variable_name]

        if sparse.issparse(block.matrix):
            M = block.matrix.toarray()
        else:
            M = np.asarray(
                block.matrix,
                dtype=float,
            )

        if M.shape != (rb.dimension, vb.dimension):
            raise ValueError(
                f"block ({block.residual_name}, {block.variable_name}) "
                f"has shape {M.shape}; "
                f"expected ({rb.dimension}, {vb.dimension})"
            )

        J[rb.row_slice, vb.sl] += M

    # Extract trajectory and calibration columns in their original block order.
    traj_cols, traj_local = _partition_columns(
        layout,
        "trajectory",
    )
    calib_cols, calib_local = _partition_columns(
        layout,
        "calibration",
    )

    J_T = (
        J[:, traj_cols]
        if traj_cols
        else np.zeros((total_rows, 0))
    )
    J_C = (
        J[:, calib_cols]
        if calib_cols
        else np.zeros((total_rows, 0))
    )

    # Package the assembled matrices with aligned residual and slice metadata.
    return JacobianBundle(
        J=J,
        J_T=J_T,
        J_C=J_C,
        residual=_residual_vector(
            residual_blocks,
            residual_values,
        ),
        row_slices={
            block.name: block.row_slice
            for block in residual_blocks
        },
        trajectory_column_slices=traj_local,
        calibration_column_slices=calib_local,
        metadata=metadata or {},
    )


##################################################
# Sparse Jacobian assembly
##################################################
def assemble_jacobian_sparse(
    layout: VariableLayout,
    residual_blocks: tuple[ResidualBlock, ...],
    jacobian_blocks: list[JacobianBlock],
    residual_values: dict[str, ArrayLike] | None = None,
    metadata: dict[str, Any] | None = None,
) -> JacobianBundle:
    '''Assemble a sparse global Jacobian directly from local blocks.

    Local blocks are converted to COO triplets and placed directly into the
    global sparse matrix. The complete Jacobian and its partitions are never
    densified. Duplicate coordinates are summed during conversion to CSR.

    Args:
        layout (VariableLayout): Variable layout with global column slices.
        residual_blocks (tuple[ResidualBlock, ...]): Residual blocks with global
            row slices.
        jacobian_blocks (list[JacobianBlock]): Local Jacobian blocks to insert.
        residual_values (dict[str, ArrayLike] | None): Optional residual vectors
            keyed by residual block name.
        metadata (dict[str, Any] | None): Optional metadata attached to the
            returned bundle.

    Returns:
        JacobianBundle: Sparse CSR Jacobian, trajectory and calibration
            partitions, aligned residual vector, slices and metadata.

    Raises:
        KeyError: If a Jacobian block references an unknown residual or
            variable name.
        ValueError: If a local Jacobian has an incompatible shape.
    '''

    # Prepare name-based block lookup and global COO storage.
    row_by_name, var_by_name, total_rows = _block_maps(
        layout,
        residual_blocks,
    )
    data: list[float] = []
    rows: list[int] = []
    cols: list[int] = []

    # Convert local matrices to COO and shift their coordinates globally.
    for block in jacobian_blocks:
        rb = row_by_name[block.residual_name]
        vb = var_by_name[block.variable_name]
        local = sparse.coo_matrix(block.matrix)

        if local.shape != (rb.dimension, vb.dimension):
            raise ValueError(
                f"block ({block.residual_name}, {block.variable_name}) "
                f"has shape {local.shape}; "
                f"expected ({rb.dimension}, {vb.dimension})"
            )

        row_start = rb.row_slice.start or 0
        col_start = vb.sl.start or 0

        data.extend(local.data.tolist())
        rows.extend(
            (local.row + row_start).tolist()
        )
        cols.extend(
            (local.col + col_start).tolist()
        )

    # Build the global CSR matrix and retain sparse trajectory partitions.
    J = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(total_rows, layout.total_dim),
    ).tocsr()

    traj_cols, traj_local = _partition_columns(
        layout,
        "trajectory",
    )
    calib_cols, calib_local = _partition_columns(
        layout,
        "calibration",
    )

    J_T = (
        J[:, traj_cols].tocsr()
        if traj_cols
        else sparse.csr_matrix((total_rows, 0))
    )
    J_C = (
        J[:, calib_cols].tocsr()
        if calib_cols
        else sparse.csr_matrix((total_rows, 0))
    )

    # Package the sparse matrices with aligned residual and slice metadata.
    return JacobianBundle(
        J=J,
        J_T=J_T,
        J_C=J_C,
        residual=_residual_vector(
            residual_blocks,
            residual_values,
        ),
        row_slices={
            block.name: block.row_slice
            for block in residual_blocks
        },
        trajectory_column_slices=traj_local,
        calibration_column_slices=calib_local,
        metadata=metadata or {},
    )