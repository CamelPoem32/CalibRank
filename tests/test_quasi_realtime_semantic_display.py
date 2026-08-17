import numpy as np
from scipy import sparse

from src.calib_observability.visualization.quasi_realtime_rover import (
    MatrixDisplayLayout,
    QuasiRealtimeConfig,
    _c_x_display_result,
    _empty_snapshot,
    _semantic_j_c_display,
)


CALIBRATION_SLICES = {
    "T_B_I": slice(0, 6),
    "b_g": slice(6, 9),
    "tau_I": slice(9, 10),
    "tau_L": slice(10, 11),
}


def _family_slices(lidar: int, gyro: int, accel: int) -> dict[str, tuple[slice, ...]]:
    start = 0
    families: dict[str, tuple[slice, ...]] = {}
    if lidar:
        families["lidar"] = (slice(start, start + lidar),)
        start += lidar
    if gyro:
        families["gyro"] = (slice(start, start + gyro),)
        start += gyro
    if accel:
        families["accelerometer"] = (slice(start, start + accel),)
    return families


def test_j_c_semantic_column_layout() -> None:
    matrix = np.ones((3, 11), dtype=float)

    result = _semantic_j_c_display(
        matrix,
        CALIBRATION_SLICES,
        {"lidar": (slice(0, 3),)},
        max_rows=20,
        max_cols=20,
        normalize_factor_blocks=False,
    )

    assert result.layout.column_boundaries == (6, 9, 10)
    assert result.layout.column_centers == (2.5, 7.0, 9.0, 10.0)
    assert result.layout.column_labels == ("T_B_I", "b_g", "tau_I", "tau_L")


def test_j_c_factor_family_rows_without_downsampling() -> None:
    matrix = np.arange(27 * 11, dtype=float).reshape(27, 11)

    result = _semantic_j_c_display(
        matrix,
        CALIBRATION_SLICES,
        _family_slices(12, 9, 6),
        max_rows=40,
        max_cols=20,
        normalize_factor_blocks=False,
    )

    assert result.matrix.shape == (27, 11)
    assert result.layout.row_boundaries == (12, 21)
    assert result.layout.row_labels == ("LiDAR", "gyro", "accel")
    assert np.allclose(result.matrix, matrix)


def test_j_c_block_aware_downsampling_keeps_every_family() -> None:
    matrix = np.arange(27 * 11, dtype=float).reshape(27, 11)

    result = _semantic_j_c_display(
        matrix,
        CALIBRATION_SLICES,
        _family_slices(12, 9, 6),
        max_rows=8,
        max_cols=20,
        normalize_factor_blocks=False,
    )

    lengths = (
        result.layout.row_boundaries[0],
        result.layout.row_boundaries[1] - result.layout.row_boundaries[0],
        result.matrix.shape[0] - result.layout.row_boundaries[1],
    )
    assert result.matrix.shape[0] <= 8
    assert all(length >= 1 for length in lengths)
    assert result.layout.row_boundaries == (lengths[0], lengths[0] + lengths[1])


def test_j_c_per_family_normalization_enabled() -> None:
    matrix = np.ones((6, 11), dtype=float)
    matrix[0:2] *= 1000.0
    matrix[2:4] *= 2.0
    matrix[4:6] *= 0.01

    result = _semantic_j_c_display(
        matrix,
        CALIBRATION_SLICES,
        _family_slices(2, 2, 2),
        max_rows=20,
        max_cols=20,
        normalize_factor_blocks=True,
    )

    assert np.isclose(np.max(np.abs(result.matrix[0:2])), 1.0)
    assert np.isclose(np.max(np.abs(result.matrix[2:4])), 1.0)
    assert np.isclose(np.max(np.abs(result.matrix[4:6])), 1.0)
    assert result.block_scales == {
        "lidar": 1000.0,
        "gyro": 2.0,
        "accelerometer": 0.01,
    }


def test_j_c_per_family_normalization_disabled_preserves_values() -> None:
    matrix = np.arange(6 * 11, dtype=float).reshape(6, 11)

    result = _semantic_j_c_display(
        matrix,
        CALIBRATION_SLICES,
        _family_slices(2, 2, 2),
        max_rows=20,
        max_cols=20,
        normalize_factor_blocks=False,
    )

    assert result.block_scales == {}
    assert np.allclose(result.matrix, matrix)


def test_j_c_dense_and_sparse_display_equivalence() -> None:
    matrix = np.arange(27 * 11, dtype=float).reshape(27, 11)

    dense = _semantic_j_c_display(
        matrix,
        CALIBRATION_SLICES,
        _family_slices(12, 9, 6),
        max_rows=8,
        max_cols=20,
        normalize_factor_blocks=True,
    )
    sparse_result = _semantic_j_c_display(
        sparse.csr_matrix(matrix),
        CALIBRATION_SLICES,
        _family_slices(12, 9, 6),
        max_rows=8,
        max_cols=20,
        normalize_factor_blocks=True,
    )

    assert np.allclose(dense.matrix, sparse_result.matrix)
    assert dense.layout == sparse_result.layout
    assert dense.block_scales == sparse_result.block_scales


def test_j_c_empty_family_combinations() -> None:
    for lidar, gyro, accel in [(3, 0, 0), (0, 3, 0), (0, 0, 3), (3, 0, 3), (0, 3, 3)]:
        matrix = np.ones((lidar + gyro + accel, 11), dtype=float)
        result = _semantic_j_c_display(
            matrix,
            CALIBRATION_SLICES,
            _family_slices(lidar, gyro, accel),
            max_rows=20,
            max_cols=20,
            normalize_factor_blocks=True,
        )
        assert result.matrix.shape[0] == lidar + gyro + accel
        assert len(result.layout.row_boundaries) == max(len(result.layout.row_labels) - 1, 0)


def test_c_x_layouts() -> None:
    lidar = _c_x_display_result(np.zeros((5, 6)), "lidar", max_rows=20, max_cols=20)
    gyro = _c_x_display_result(np.zeros((5, 3)), "gyro", max_rows=20, max_cols=20)

    assert lidar.layout.column_boundaries == (3,)
    assert lidar.layout.column_labels == ("rotation", "translation")
    assert gyro.layout.column_boundaries == ()
    assert gyro.layout.column_labels == ("rotation",)


def test_empty_snapshot_has_valid_layouts() -> None:
    config = QuasiRealtimeConfig()
    snapshot = _empty_snapshot(
        0.0,
        0.0,
        0.0,
        "waiting",
        config,
        {},
        {},
        {},
    )

    assert isinstance(snapshot.J_C_display_layout, MatrixDisplayLayout)
    assert isinstance(snapshot.C_X_L_display_layout, MatrixDisplayLayout)
    assert isinstance(snapshot.C_X_I_gyro_display_layout, MatrixDisplayLayout)
    assert isinstance(snapshot.C_X_I_accel_display_layout, MatrixDisplayLayout)
    assert snapshot.J_C_display_block_scales == {}
