from pathlib import Path

import numpy as np
import pandas as pd

from src.kaist_dataset.lidar_map import (
    LIDAR_MAP_POSE_CSV_COLUMNS,
    LidarMapRegistrationResult,
    LidarReferenceMap,
    compose_initial_lidar_pose,
    csv_row_to_pose_matrix,
    load_lidar_map_pose_csv,
    pose_matrix_to_csv_values,
    registration_inputs_to_local_origin,
    restore_local_origin_transform,
    save_lidar_map_pose_csv,
    bounds_around_positions,
    initial_lidar_positions_from_reference,
    interpolate_reference_body_pose,
)


def test_csv_matrix_serialization_order(tmp_path: Path):
    T_W_L = np.eye(4)
    T_W_L[:3, :] = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ])
    assert pose_matrix_to_csv_values(T_W_L) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    result = LidarMapRegistrationResult(timestamp_s=123.25, scan_path='scan.bin', T_W_L_initial=T_W_L, T_W_L_estimated=T_W_L, fitness=0.8, inlier_rmse=0.2, success=True)
    csv_path = save_lidar_map_pose_csv([result], tmp_path / 'poses.csv')
    dataframe = pd.read_csv(csv_path)

    assert list(dataframe.columns) == LIDAR_MAP_POSE_CSV_COLUMNS
    assert list(dataframe.columns[:13]) == ['timestamp', 'r11', 'r12', 'r13', 't11', 'r21', 'r22', 'r23', 't21', 'r31', 'r32', 'r33', 't31']
    assert dataframe.iloc[0, :13].tolist() == [123.25, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


def test_csv_pose_reconstruction_round_trip(tmp_path: Path):
    T_W_L = np.eye(4)
    T_W_L[:3, :] = np.array([
        [0.0, -1.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
    ])
    row = dict(zip(['timestamp', 'r11', 'r12', 'r13', 't11', 'r21', 'r22', 'r23', 't21', 'r31', 'r32', 'r33', 't31'], [1.0, *pose_matrix_to_csv_values(T_W_L)]))
    np.testing.assert_allclose(csv_row_to_pose_matrix(row), T_W_L)

    result = LidarMapRegistrationResult(timestamp_s=1.0, scan_path='scan.bin', T_W_L_initial=T_W_L, T_W_L_estimated=T_W_L, success=True)
    csv_path = save_lidar_map_pose_csv([result], tmp_path / 'round_trip.csv')
    timestamps, poses, dataframe = load_lidar_map_pose_csv(csv_path)

    np.testing.assert_allclose(timestamps, [1.0])
    np.testing.assert_allclose(poses[0], T_W_L)
    assert dataframe.shape[0] == 1


def test_local_registration_origin_conversion_restores_global_transform():
    T_W_L = np.eye(4)
    T_W_L[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    T_W_L[:3, 3] = [450000.0, 4020000.0, 85.0]
    map_points_W = np.array([
        [450001.0, 4020002.0, 86.0],
        [449999.0, 4019998.0, 84.0],
    ])

    local = registration_inputs_to_local_origin(T_W_L, map_points_W)

    np.testing.assert_allclose(local.origin_xyz, T_W_L[:3, 3])
    np.testing.assert_allclose(local.target_points_O, map_points_W - T_W_L[:3, 3])
    np.testing.assert_allclose(local.T_O_L_initial[:3, :3], T_W_L[:3, :3])
    np.testing.assert_allclose(local.T_O_L_initial[:3, 3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(restore_local_origin_transform(local.T_O_L_initial, local.origin_xyz), T_W_L)


def test_map_crop_preserves_global_coordinates_and_excludes_outside_radius():
    points = np.array([
        [100.0, 100.0, 10.0],
        [103.0, 104.0, 12.0],
        [110.0, 100.0, 10.0],
        [101.0, 101.0, 30.0],
    ])
    reference_map = LidarReferenceMap.from_points(points)

    crop = reference_map.query_local_map([100.0, 100.0, 10.0], radius_m=5.1, z_margin_m=5.0)

    assert crop.shape == (2, 3)
    assert {tuple(row) for row in crop} == {(100.0, 100.0, 10.0), (103.0, 104.0, 12.0)}


def test_compose_initial_lidar_pose_uses_T_W_B_times_T_B_L():
    T_W_B = np.eye(4)
    T_W_B[:3, 3] = [10.0, 20.0, 30.0]
    T_B_L = np.eye(4)
    T_B_L[:3, 3] = [1.0, 2.0, 3.0]

    T_W_L = compose_initial_lidar_pose(T_W_B, T_B_L)

    np.testing.assert_allclose(T_W_L, T_W_B @ T_B_L)
    np.testing.assert_allclose(T_W_L[:3, 3], [11.0, 22.0, 33.0])


def test_failed_registration_writes_nan_pose_entries(tmp_path: Path):
    T_initial = np.eye(4)
    failed = LidarMapRegistrationResult(timestamp_s=2.0, scan_path='bad.bin', T_W_L_initial=T_initial, T_W_L_estimated=None, success=False, fitness=np.nan, inlier_rmse=np.nan, reference_timestamp_mismatch_s=0.01, scan_points_used=12, map_points_used=34)

    csv_path = save_lidar_map_pose_csv([failed], tmp_path / 'failed.csv')
    dataframe = pd.read_csv(csv_path)

    assert bool(dataframe.loc[0, 'success']) is False
    assert dataframe.loc[0, 'timestamp'] == 2.0
    assert dataframe.loc[0, ['r11', 'r12', 'r13', 't11', 'r21', 'r22', 'r23', 't21', 'r31', 'r32', 'r33', 't31']].isna().all()
    assert dataframe.loc[0, 'scan_points_used'] == 12
    assert dataframe.loc[0, 'map_points_used'] == 34

def test_cpu_scan_to_map_registration_synthetic_identity(tmp_path: Path):
    pytest = __import__('pytest')
    try:
        __import__('open3d')
    except ImportError:
        pytest.skip('Open3D is not installed in this environment.')

    rng = np.random.default_rng(7)
    scan_points_L = rng.normal(size=(300, 3))
    scan_points_L /= np.maximum(np.linalg.norm(scan_points_L, axis=1, keepdims=True), 1e-9)
    scan_points_L *= rng.uniform(0.5, 2.0, size=(300, 1))
    raw = np.column_stack([scan_points_L, np.ones(scan_points_L.shape[0])]).astype('<f4')
    scan_path = tmp_path / '1000000000.bin'
    raw.tofile(scan_path)

    T_W_L = np.eye(4)
    T_W_L[:3, 3] = [315421.5, 4154182.0, 15.0]
    map_points_W = scan_points_L @ T_W_L[:3, :3].T + T_W_L[:3, 3]
    reference_map = LidarReferenceMap.from_points(map_points_W)

    from src.kaist_dataset.lidar_map import localize_lidar_scans_to_map

    results = localize_lidar_scans_to_map(
        reference_map,
        [scan_path],
        np.array([1.0]),
        T_W_L[None, :, :],
        np.eye(4),
        max_reference_mismatch_s=0.0,
        scan_voxel_size_m=0.1,
        map_crop_radius_m=5.0,
        map_crop_z_margin_m=5.0,
        max_correspondence_distance_m=0.5,
        max_iterations=5,
        voxel_scale_factors=(1.0,),
        robust_kernel='l2',
        device='cpu',
        progress=False,
    )

    assert len(results) == 1
    assert results[0].success
    assert results[0].diagnostics['used_local_origin_compensation'] is True
    np.testing.assert_allclose(results[0].T_W_L_estimated, T_W_L, atol=1e-5)

def test_bounds_around_positions_expands_xy_and_z():
    positions = np.array([[10.0, 20.0, 3.0], [14.0, 25.0, 8.0]])

    bounds_min, bounds_max = bounds_around_positions(positions, xy_margin_m=2.0, z_margin_m=1.5)

    np.testing.assert_allclose(bounds_min, [8.0, 18.0, 1.5])
    np.testing.assert_allclose(bounds_max, [16.0, 27.0, 9.5])


def test_initial_lidar_positions_from_reference_uses_extrinsic_convention():
    T_W_B = np.eye(4)
    T_W_B[:3, 3] = [100.0, 200.0, 300.0]
    T_B_L = np.eye(4)
    T_B_L[:3, 3] = [1.0, 2.0, 3.0]

    positions, mismatches = initial_lidar_positions_from_reference([10.0], [10.0], T_W_B[None, :, :], T_B_L, max_reference_mismatch_s=0.0)

    np.testing.assert_allclose(positions, [[101.0, 202.0, 303.0]])
    np.testing.assert_allclose(mismatches, [0.0])

def test_reference_interpolation_clamps_endpoint_inside_mismatch_tolerance():
    first_pose = np.eye(4)
    first_pose[:3, 3] = [1.0, 2.0, 3.0]
    second_pose = np.eye(4)
    second_pose[:3, 3] = [4.0, 5.0, 6.0]

    T, mismatch = interpolate_reference_body_pose([10.0, 11.0], np.stack([first_pose, second_pose]), 9.96, max_mismatch_s=0.05)

    np.testing.assert_allclose(T, first_pose)
    assert np.isclose(mismatch, 0.04)