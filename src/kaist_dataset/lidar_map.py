'''LAS-map scan-to-map localization utilities for KAIST Urban16 LiDAR diagnostics.

The transform convention follows the rest of the project: ``T_A_B`` maps points
from frame B into frame A. Scan-to-map ICP registers raw Left VLP scan points in
LiDAR coordinates directly against a LAS crop in the world/map frame, so the ICP
result is an absolute ``T_W_L``. For KAIST initial poses, ``T_B_L`` maps LiDAR to
body and therefore ``T_W_L_initial = T_W_B_initial @ T_B_L``.
'''

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import gc
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from .lidar import load_lidar_scan_xyz, scan_timestamps_from_paths, sort_by_filename

try:
    from data_processing import _interpolate_pose as _project_interpolate_pose
except ImportError:  # pragma: no cover - package import style depends on caller path setup.
    try:
        from src.data_processing import _interpolate_pose as _project_interpolate_pose
    except ImportError:  # pragma: no cover
        _project_interpolate_pose = None


POSE_CSV_COLUMNS = [
    'timestamp',
    'r11', 'r12', 'r13', 't11',
    'r21', 'r22', 'r23', 't21',
    'r31', 'r32', 'r33', 't31',
]
DIAGNOSTIC_CSV_COLUMNS = [
    'success',
    'fitness',
    'inlier_rmse',
    'reference_timestamp_mismatch_s',
    'translation_correction_m',
    'rotation_correction_deg',
    'scan_points_used',
    'map_points_used',
]
LIDAR_MAP_POSE_CSV_COLUMNS = POSE_CSV_COLUMNS + DIAGNOSTIC_CSV_COLUMNS
DEFAULT_T_B_LL = np.array([
    [-0.514066, -0.702201, -0.492595, -0.440699],
    [0.486485, -0.711672, 0.506809, 0.397052],
    [-0.706447, 0.0208933, 0.707457, 1.90953],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)


@dataclass(frozen=True)
class LocalOriginRegistrationInput:
    '''ICP inputs expressed in a temporary local map frame O near the scan.'''

    origin_xyz: np.ndarray
    target_points_O: np.ndarray
    T_O_L_initial: np.ndarray


@dataclass
class LidarMapRegistrationResult:
    '''Diagnostic result for one independent raw-scan-to-LAS-map registration.'''

    timestamp_s: float
    scan_path: Path | str
    T_W_L_initial: np.ndarray | None
    T_W_L_estimated: np.ndarray | None
    fitness: float = np.nan
    inlier_rmse: float = np.nan
    success: bool = False
    translation_correction_m: float = np.nan
    rotation_correction_deg: float = np.nan
    map_points_used: int = 0
    scan_points_used: int = 0
    reference_timestamp_mismatch_s: float = np.nan
    message: str = ''
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LidarReferenceMap:
    '''A LAS/LAZ reference map with one persistent CPU spatial index for crops.'''

    points_xyz: np.ndarray
    intensity: np.ndarray | None = None
    voxel_size_m: float | None = None
    source_files: tuple[Path, ...] = ()
    original_point_count: int | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points_xyz must have shape (N, 3).')
        finite_mask = np.all(np.isfinite(points), axis=1)
        if not np.all(finite_mask):
            points = points[finite_mask]
            if self.intensity is not None:
                self.intensity = np.asarray(self.intensity)[finite_mask]
        if points.size == 0:
            raise ValueError('Reference map contains no finite XYZ points.')
        self.points_xyz = np.ascontiguousarray(points, dtype=np.float64)
        if self.intensity is not None:
            self.intensity = np.asarray(self.intensity)
        if self.original_point_count is None:
            self.original_point_count = int(self.points_xyz.shape[0])
        self._xy_tree = cKDTree(self.points_xyz[:, :2])

    @classmethod
    def from_las_path(cls, path: str | Path, *, voxel_size_m: float | None = None, bounds_min_xyz: Sequence[float] | None = None, bounds_max_xyz: Sequence[float] | None = None, chunk_size_points: int = 5_000_000, progress: bool = False) -> 'LidarReferenceMap':
        '''Discover and load one LAS/LAZ file or a recursive LAS/LAZ directory.'''
        return cls.from_las_files(discover_las_files(path), voxel_size_m=voxel_size_m, bounds_min_xyz=bounds_min_xyz, bounds_max_xyz=bounds_max_xyz, chunk_size_points=chunk_size_points, progress=progress)

    @classmethod
    def from_las_files(cls, las_files: Sequence[str | Path], *, voxel_size_m: float | None = None, bounds_min_xyz: Sequence[float] | None = None, bounds_max_xyz: Sequence[float] | None = None, chunk_size_points: int = 5_000_000, progress: bool = False) -> 'LidarReferenceMap':
        '''Load LAS/LAZ files once, streaming large files before building one KD-tree.'''
        files = tuple(Path(path).expanduser() for path in las_files)
        if not files:
            raise FileNotFoundError('No LAS/LAZ files were provided.')
        points_chunks: list[np.ndarray] = []
        intensity_chunks: list[np.ndarray] = []
        has_intensity = True
        original_count = 0
        for file_path in files:
            chunk_iterator = iter_las_point_chunks(file_path, bounds_min_xyz=bounds_min_xyz, bounds_max_xyz=bounds_max_xyz, chunk_size_points=chunk_size_points)
            if progress:
                chunk_iterator = tqdm(chunk_iterator, desc=f'Loading {file_path.name}', unit='chunk')
            for points, intensity in chunk_iterator:
                original_count += int(points.shape[0])
                if voxel_size_m is not None:
                    points, intensity = voxel_downsample_points(points, voxel_size_m, intensity=intensity)
                points_chunks.append(points)
                if intensity is None:
                    has_intensity = False
                else:
                    intensity_chunks.append(intensity)
        if not points_chunks:
            raise ValueError('LAS files contain no finite XYZ points.')
        points_xyz = np.vstack(points_chunks)
        intensity = np.concatenate(intensity_chunks) if has_intensity and len(intensity_chunks) == len(points_chunks) else None
        if voxel_size_m is not None:
            points_xyz, intensity = voxel_downsample_points(points_xyz, voxel_size_m, intensity=intensity)
        return cls(points_xyz=points_xyz, intensity=intensity, voxel_size_m=voxel_size_m, source_files=files, original_point_count=original_count)

    @classmethod
    def from_points(cls, points_xyz: np.ndarray, *, intensity: np.ndarray | None = None, voxel_size_m: float | None = None) -> 'LidarReferenceMap':
        '''Build a reference-map object from synthetic or already-loaded points.'''
        points = np.asarray(points_xyz, dtype=np.float64)
        intensities = None if intensity is None else np.asarray(intensity)
        original_count = int(points.shape[0])
        if voxel_size_m is not None:
            points, intensities = voxel_downsample_points(points, voxel_size_m, intensity=intensities)
        return cls(points_xyz=points, intensity=intensities, voxel_size_m=voxel_size_m, original_point_count=original_count)

    @property
    def point_count(self) -> int:
        return int(self.points_xyz.shape[0])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.min(self.points_xyz, axis=0), np.max(self.points_xyz, axis=0)

    def query_local_map(self, center_xyz: Sequence[float], radius_m: float, z_margin_m: float | None = None) -> np.ndarray:
        '''Return a world-frame LAS crop around ``center_xyz`` without recentering it.'''
        center = np.asarray(center_xyz, dtype=float).reshape(3)
        radius_m = float(radius_m)
        if not np.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError('radius_m must be finite and positive.')
        indices = self._xy_tree.query_ball_point(center[:2], radius_m)
        if not indices:
            return np.empty((0, 3), dtype=np.float64)
        crop = self.points_xyz[np.asarray(indices, dtype=np.int64)]
        if z_margin_m is not None:
            z_margin_m = float(z_margin_m)
            if not np.isfinite(z_margin_m) or z_margin_m < 0.0:
                raise ValueError('z_margin_m must be finite and non-negative when provided.')
            crop = crop[np.abs(crop[:, 2] - center[2]) <= z_margin_m]
        return np.ascontiguousarray(crop, dtype=np.float64)

    def sample_points(self, max_points: int = 200_000, *, seed: int = 0) -> np.ndarray:
        '''Return a random plotting subset without modifying the stored map.'''
        max_points = int(max_points)
        if max_points <= 0 or self.point_count <= max_points:
            return self.points_xyz
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.point_count, size=max_points, replace=False)
        return self.points_xyz[np.sort(indices)]


def discover_las_files(path: str | Path) -> list[Path]:
    '''Discover LAS/LAZ map files from one file or a recursive directory.'''
    root = Path(path).expanduser()
    suffixes = {'.las', '.laz'}
    if root.is_file():
        if root.suffix.lower() not in suffixes:
            raise ValueError(f'Expected a .las or .laz file, got: {root}')
        return [root]
    if not root.exists():
        raise FileNotFoundError(f'LAS map path does not exist: {root}')
    if not root.is_dir():
        raise ValueError(f'LAS map path is neither a file nor a directory: {root}')
    files = sorted(path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in suffixes)
    if not files:
        raise FileNotFoundError(f'No .las or .laz files found below: {root}')
    return files


def iter_las_point_chunks(path: str | Path, *, bounds_min_xyz: Sequence[float] | None = None, bounds_max_xyz: Sequence[float] | None = None, chunk_size_points: int = 5_000_000):
    '''Yield finite XYZ/intensity chunks from one LAS/LAZ file.'''
    try:
        import laspy
    except ImportError as exc:
        raise ImportError('laspy is required to load LAS/LAZ reference maps. Install it in the notebook environment with `pip install laspy`.') from exc
    file_path = Path(path).expanduser()
    bounds_min, bounds_max = _normalize_bounds(bounds_min_xyz, bounds_max_xyz)
    chunk_size_points = int(chunk_size_points)
    if chunk_size_points <= 0:
        raise ValueError('chunk_size_points must be positive.')
    with laspy.open(file_path) as reader:
        has_intensity = 'intensity' in set(reader.header.point_format.dimension_names)
        for las_points in reader.chunk_iterator(chunk_size_points):
            points = np.column_stack((np.asarray(las_points.x, dtype=np.float64), np.asarray(las_points.y, dtype=np.float64), np.asarray(las_points.z, dtype=np.float64)))
            finite_mask = np.all(np.isfinite(points), axis=1)
            if bounds_min is not None:
                finite_mask &= np.all(points >= bounds_min, axis=1)
            if bounds_max is not None:
                finite_mask &= np.all(points <= bounds_max, axis=1)
            points = points[finite_mask]
            if points.size == 0:
                continue
            intensity = np.asarray(las_points.intensity)[finite_mask] if has_intensity else None
            yield np.ascontiguousarray(points, dtype=np.float64), intensity


def load_las_points(path: str | Path, *, bounds_min_xyz: Sequence[float] | None = None, bounds_max_xyz: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    '''Load one LAS/LAZ file as world/map XYZ plus optional intensity.'''
    chunks = list(iter_las_point_chunks(path, bounds_min_xyz=bounds_min_xyz, bounds_max_xyz=bounds_max_xyz))
    if not chunks:
        raise ValueError(f'LAS file contains no finite XYZ points: {Path(path).expanduser()}')
    points = np.vstack([chunk[0] for chunk in chunks])
    has_intensity = all(chunk[1] is not None for chunk in chunks)
    intensity = np.concatenate([chunk[1] for chunk in chunks]) if has_intensity else None
    return np.ascontiguousarray(points, dtype=np.float64), intensity


def voxel_downsample_points(points_xyz: np.ndarray, voxel_size_m: float | None, *, intensity: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    '''Voxel-downsample points by keeping the first point in each occupied voxel.'''
    points = np.asarray(points_xyz, dtype=np.float64)
    if voxel_size_m is None:
        return np.ascontiguousarray(points, dtype=np.float64), intensity
    voxel_size_m = float(voxel_size_m)
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError('voxel_size_m must be finite and positive.')
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), None if intensity is None else np.asarray(intensity)[:0]
    finite_mask = np.all(np.isfinite(points), axis=1)
    points = points[finite_mask]
    intensities = None if intensity is None else np.asarray(intensity)[finite_mask]
    voxel_keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, first_indices = np.unique(voxel_keys, axis=0, return_index=True)
    first_indices = np.sort(first_indices)
    downsampled_points = np.ascontiguousarray(points[first_indices], dtype=np.float64)
    downsampled_intensity = None if intensities is None else intensities[first_indices]
    return downsampled_points, downsampled_intensity


def compose_initial_lidar_pose(T_W_B: np.ndarray, T_B_L: np.ndarray) -> np.ndarray:
    '''Compose the initial LiDAR pose with project convention ``T_A_B``.

    ``T_W_B`` maps body-frame points into the world/map frame. ``T_B_L`` maps
    LiDAR-frame points into the body frame. Their product maps LiDAR-frame
    points directly into the world/map frame: ``T_W_L = T_W_B @ T_B_L``.
    '''
    return _as_transform(T_W_B, 'T_W_B') @ _as_transform(T_B_L, 'T_B_L')


def interpolate_reference_body_pose(reference_timestamps_s: Sequence[float], reference_poses_T_W_B: np.ndarray, query_time_s: float, *, max_mismatch_s: float | None = None) -> tuple[np.ndarray, float]:
    '''Interpolate a reference body pose and report nearest timestamp mismatch.'''
    timestamps = np.asarray(reference_timestamps_s, dtype=float)
    poses = np.asarray(reference_poses_T_W_B, dtype=float)
    if timestamps.ndim != 1 or poses.shape != (timestamps.size, 4, 4):
        raise ValueError('reference_timestamps_s and reference_poses_T_W_B must have shapes (N,) and (N, 4, 4).')
    if timestamps.size == 0:
        raise ValueError('Reference trajectory is empty.')
    query_time_s = float(query_time_s)
    nearest_index = int(np.clip(np.searchsorted(timestamps, query_time_s), 0, timestamps.size - 1))
    candidates = {nearest_index}
    if nearest_index > 0:
        candidates.add(nearest_index - 1)
    mismatch = min(abs(float(timestamps[index]) - query_time_s) for index in candidates)
    if max_mismatch_s is not None and mismatch > float(max_mismatch_s):
        raise ValueError(f'Reference timestamp mismatch {mismatch:.6f} s exceeds {float(max_mismatch_s):.6f} s.')
    if query_time_s <= timestamps[0]:
        return poses[0].copy(), float(mismatch)
    if query_time_s >= timestamps[-1]:
        return poses[-1].copy(), float(mismatch)
    if _project_interpolate_pose is not None and timestamps.size >= 2:
        return np.asarray(_project_interpolate_pose(timestamps, poses, query_time_s), dtype=float), float(mismatch)
    return poses[min(candidates, key=lambda index: abs(float(timestamps[index]) - query_time_s))].copy(), float(mismatch)


def registration_inputs_to_local_origin(T_W_L_initial: np.ndarray, target_points_W: np.ndarray, *, origin_xyz: Sequence[float] | None = None) -> LocalOriginRegistrationInput:
    '''Move a global map crop and initial pose into a local frame for ICP.

    The raw source scan remains in LiDAR coordinates and is not pre-transformed.
    Target map points become ``p_O = p_W - origin``. The initial transform keeps
    the same rotation and subtracts the same origin from its translation, so it
    maps LiDAR coordinates into this temporary local map frame. Restoring the ICP
    result only adds the origin back to the translation.
    '''
    T_W_L_initial = _as_transform(T_W_L_initial, 'T_W_L_initial')
    target_points = np.asarray(target_points_W, dtype=np.float64)
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError('target_points_W must have shape (N, 3).')
    origin = np.asarray(T_W_L_initial[:3, 3] if origin_xyz is None else origin_xyz, dtype=np.float64).reshape(3)
    T_O_L_initial = T_W_L_initial.copy()
    T_O_L_initial[:3, 3] = T_W_L_initial[:3, 3] - origin
    return LocalOriginRegistrationInput(origin_xyz=origin, target_points_O=np.ascontiguousarray(target_points - origin, dtype=np.float64), T_O_L_initial=T_O_L_initial)


def restore_local_origin_transform(T_O_L: np.ndarray, origin_xyz: Sequence[float]) -> np.ndarray:
    '''Restore a local-origin ICP transform to an absolute world/map ``T_W_L``.'''
    T_W_L = _as_transform(T_O_L, 'T_O_L').copy()
    T_W_L[:3, 3] = T_W_L[:3, 3] + np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    return T_W_L


def open3d_cuda_is_available(cuda_device_id: int = 0) -> bool:
    '''Return whether Open3D Tensor CUDA is available for a given device id.'''
    try:
        import open3d as o3d
    except ImportError:
        return False
    if not o3d.core.cuda.is_available():
        return False
    return int(cuda_device_id) < int(o3d.core.cuda.device_count())


def choose_open3d_device(requested_device: str = 'cpu', cuda_device_id: int = 0) -> str:
    '''Use CUDA when requested and available, otherwise return CPU with a warning.'''
    requested = str(requested_device).strip().lower()
    if requested not in {'cpu', 'cuda'}:
        raise ValueError("requested_device must be 'cpu' or 'cuda'.")
    if requested == 'cuda' and not open3d_cuda_is_available(cuda_device_id):
        warnings.warn("Open3D CUDA is unavailable; falling back to device='cpu'.", RuntimeWarning, stacklevel=2)
        return 'cpu'
    return requested


def select_scans_overlapping_reference(scan_paths: Sequence[str | Path], scan_timestamps_s: Sequence[float], reference_timestamps_s: Sequence[float], *, max_scans: int | None = None, scan_step: int = 1, margin_s: float = 0.0, max_reference_mismatch_s: float | None = None) -> tuple[list[Path], np.ndarray]:
    '''Select LiDAR scans supported by the reference trajectory.

    ``margin_s`` expands the overall start/end support of the reference trajectory.

    ``max_reference_mismatch_s`` additionally requires every selected LiDAR timestamp to have an actual reference sample within the requested temporal distance. This rejects scans that fall inside large internal gaps of the reference trajectory.
    '''

    paths = [Path(path).expanduser() for path in scan_paths]
    timestamps = np.asarray(scan_timestamps_s, dtype=float)
    reference_timestamps = np.asarray(reference_timestamps_s, dtype=float)

    if len(paths) != timestamps.size:
        raise ValueError('scan_paths and scan_timestamps_s must have matching lengths.')

    if reference_timestamps.ndim != 1 or reference_timestamps.size == 0:
        raise ValueError('reference_timestamps_s must be a non-empty one-dimensional array.')

    if timestamps.ndim != 1:
        raise ValueError('scan_timestamps_s must be one-dimensional.')

    if np.any(np.diff(reference_timestamps) <= 0.0):
        raise ValueError('reference_timestamps_s must be strictly increasing.')

    scan_step = int(scan_step)

    if scan_step <= 0:
        raise ValueError('scan_step must be positive.')

    ##################################################
    # First keep timestamps inside the overall reference support.
    ##################################################

    valid_mask = (
        (timestamps >= reference_timestamps[0] - float(margin_s))
        & (timestamps <= reference_timestamps[-1] + float(margin_s))
    )

    ##################################################
    # Optionally reject timestamps lying inside large gaps in the
    # reference trajectory.
    ##################################################

    if max_reference_mismatch_s is not None:
        max_reference_mismatch_s = float(max_reference_mismatch_s)

        if not np.isfinite(max_reference_mismatch_s) or max_reference_mismatch_s < 0.0:
            raise ValueError('max_reference_mismatch_s must be finite and non-negative when provided.')

        right_indices = np.searchsorted(reference_timestamps, timestamps, side='left')
        right_indices = np.clip(right_indices, 0, reference_timestamps.size - 1)
        left_indices = np.clip(right_indices - 1, 0, reference_timestamps.size - 1)

        right_mismatches = np.abs(reference_timestamps[right_indices] - timestamps)
        left_mismatches = np.abs(reference_timestamps[left_indices] - timestamps)

        nearest_mismatches = np.minimum(left_mismatches, right_mismatches)

        valid_mask &= nearest_mismatches <= max_reference_mismatch_s

    indices = np.nonzero(valid_mask)[0]

    ##################################################
    # Apply downsampling only after validity filtering.
    ##################################################

    indices = indices[::scan_step]

    if max_scans is not None:
        max_scans = int(max_scans)

        if max_scans < 0:
            raise ValueError('max_scans must be non-negative when provided.')

        indices = indices[:max_scans]

    return [paths[index] for index in indices], timestamps[indices]


def initial_lidar_positions_from_reference(scan_timestamps_s: Sequence[float], reference_timestamps_s: Sequence[float], reference_poses_T_W_B: np.ndarray, T_B_L: np.ndarray, *, max_reference_mismatch_s: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    '''Compute reference-initialized LiDAR positions for map-load bounds and diagnostics.'''
    positions = []
    mismatches = []
    for timestamp_s in np.asarray(scan_timestamps_s, dtype=float):
        T_W_B, mismatch_s = interpolate_reference_body_pose(reference_timestamps_s, reference_poses_T_W_B, float(timestamp_s), max_mismatch_s=max_reference_mismatch_s)
        T_W_L = compose_initial_lidar_pose(T_W_B, T_B_L)
        positions.append(T_W_L[:3, 3].copy())
        mismatches.append(float(mismatch_s))
    return np.asarray(positions, dtype=float), np.asarray(mismatches, dtype=float)


def bounds_around_positions(positions_xyz: np.ndarray, *, xy_margin_m: float, z_margin_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    '''Return XYZ bounds around positions, expanding XY and optionally Z by margins.'''
    positions = np.asarray(positions_xyz, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError('positions_xyz must have shape (N, 3) with N > 0.')
    min_xyz = np.min(positions, axis=0)
    max_xyz = np.max(positions, axis=0)
    xy_margin_m = float(xy_margin_m)
    if not np.isfinite(xy_margin_m) or xy_margin_m < 0.0:
        raise ValueError('xy_margin_m must be finite and non-negative.')
    min_xyz[:2] -= xy_margin_m
    max_xyz[:2] += xy_margin_m
    if z_margin_m is None:
        min_xyz[2] = -np.inf
        max_xyz[2] = np.inf
    else:
        z_margin_m = float(z_margin_m)
        if not np.isfinite(z_margin_m) or z_margin_m < 0.0:
            raise ValueError('z_margin_m must be finite and non-negative when provided.')
        min_xyz[2] -= z_margin_m
        max_xyz[2] += z_margin_m
    return min_xyz, max_xyz


@dataclass(frozen=True)
class _RegistrationConfig:
    voxel_size_m: float
    max_correspondence_distance_m: float
    max_iterations: int
    voxel_scale_factors: tuple[float, ...]
    robust_kernel: str | None
    robust_kernel_scale_factor: float
    device: str
    cuda_device_id: int
    normal_radius_factor: float
    normal_max_nn: int

    def __post_init__(self) -> None:
        if self.device not in {'cpu', 'cuda'}:
            raise ValueError("device must be 'cpu' or 'cuda'.")
        if not np.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError('voxel_size_m must be finite and positive.')
        if not np.isfinite(self.max_correspondence_distance_m) or self.max_correspondence_distance_m <= 0.0:
            raise ValueError('max_correspondence_distance_m must be finite and positive.')
        if int(self.max_iterations) <= 0:
            raise ValueError('max_iterations must be positive.')
        factors = tuple(float(value) for value in self.voxel_scale_factors)
        if not factors or any(not np.isfinite(value) or value <= 0.0 for value in factors):
            raise ValueError('voxel_scale_factors must contain finite positive values.')
        if any(factors[index] <= factors[index + 1] for index in range(len(factors) - 1)):
            raise ValueError('voxel_scale_factors must be strictly decreasing from coarse to fine.')
        kernel = None if self.robust_kernel is None else str(self.robust_kernel).strip().lower()
        if kernel not in {None, 'l2', 'tukey', 'huber', 'cauchy'}:
            raise ValueError("robust_kernel must be one of None, 'l2', 'tukey', 'huber', or 'cauchy'.")


@dataclass(frozen=True)
class _LocalizationTask:
    sequence_index: int
    scan_path: Path
    timestamp_s: float
    T_W_L_initial: np.ndarray | None
    reference_timestamp_mismatch_s: float
    preparation_error: str = ''


@dataclass
class _CachedMapTarget:
    origin_xyz: np.ndarray
    target: Any
    map_points_used: int


class _MapTargetCache:
    """Bounded LRU cache of preprocessed LAS target crops."""

    def __init__(self, reference_map: LidarReferenceMap, *, config: _RegistrationConfig, open3d, open3d_device, map_crop_radius_m: float, map_crop_z_margin_m: float | None, cell_size_m: float, max_entries: int):
        self.reference_map = reference_map
        self.config = config
        self.open3d = open3d
        self.open3d_device = open3d_device
        self.map_crop_radius_m = float(map_crop_radius_m)
        self.map_crop_z_margin_m = None if map_crop_z_margin_m is None else float(map_crop_z_margin_m)
        self.cell_size_m = float(cell_size_m)
        self.max_entries = max(int(max_entries), 1)
        if not np.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise ValueError('map_cache_cell_size_m must be finite and positive.')
        self._entries: OrderedDict[tuple[int, int, int], _CachedMapTarget] = OrderedDict()

    def _key_and_origin(self, center_xyz: np.ndarray) -> tuple[tuple[int, int, int], np.ndarray]:
        center = np.asarray(center_xyz, dtype=np.float64).reshape(3)
        cell_indices = np.floor(center / self.cell_size_m).astype(np.int64)
        if self.map_crop_z_margin_m is None:
            cell_indices[2] = 0
            origin = (cell_indices.astype(np.float64) + 0.5) * self.cell_size_m
            origin[2] = center[2]
        else:
            origin = (cell_indices.astype(np.float64) + 0.5) * self.cell_size_m
        return tuple(int(value) for value in cell_indices), origin

    def get(self, T_W_L_initial: np.ndarray) -> _CachedMapTarget:
        center = _as_transform(T_W_L_initial, 'T_W_L_initial')[:3, 3]
        key, origin = self._key_and_origin(center)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached
        xy_extra = np.sqrt(2.0) * 0.5 * self.cell_size_m
        z_extra = 0.5 * self.cell_size_m
        crop_radius_m = self.map_crop_radius_m + xy_extra
        crop_z_margin_m = None if self.map_crop_z_margin_m is None else self.map_crop_z_margin_m + z_extra
        map_crop_W = self.reference_map.query_local_map(origin, crop_radius_m, z_margin_m=crop_z_margin_m)
        if map_crop_W.shape[0] == 0:
            raise ValueError('Local LAS crop is empty.')
        target_points_O = np.ascontiguousarray(map_crop_W - origin, dtype=np.float64)
        target = _make_tensor_cloud(self.open3d, target_points_O, self.open3d_device)
        target = _preprocess_tensor_cloud(self.open3d, target, self.config, voxel_size_m=self.config.voxel_size_m * min(self.config.voxel_scale_factors), estimate_normals=True)
        map_points_used = int(target.point.positions.shape[0])
        if map_points_used == 0:
            raise ValueError('Voxel-downsampled LAS crop is empty.')
        cached = _CachedMapTarget(origin_xyz=origin, target=target, map_points_used=map_points_used)
        self._entries[key] = cached
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return cached


_CPU_REFERENCE_MAP: LidarReferenceMap | None = None
_CPU_WORKER_CONFIG: _RegistrationConfig | None = None
_CPU_WORKER_OPEN3D = None
_CPU_WORKER_DEVICE = None
_CPU_WORKER_MAP_CACHE: _MapTargetCache | None = None
_CPU_WORKER_CLEANUP_INTERVAL = 0


def localize_lidar_scans_to_map(reference_map: LidarReferenceMap, scan_paths: Sequence[str | Path], reference_timestamps_s: Sequence[float], reference_poses_T_W_B: np.ndarray, T_B_L: np.ndarray, *, initial_guess_source: str = 'reference', max_reference_mismatch_s: float | None = 0.05, scan_voxel_size_m: float = 0.25, map_crop_radius_m: float = 50.0, map_crop_z_margin_m: float | None = 10.0, max_correspondence_distance_m: float = 1.5, max_iterations: int = 50, voxel_scale_factors: tuple[float, ...] = (2.0, 1.0, 0.5), robust_kernel: str | None = 'tukey', robust_kernel_scale_factor: float = 1.0, device: str = 'cpu', cuda_device_id: int = 0, max_scans: int | None = None, scan_step: int = 1, scan_period_s: float | None = None, progress: bool = True, normal_radius_factor: float = 2.0, normal_max_nn: int = 30, cpu_workers: int | None = None, cpu_threads_per_worker: int = 1, map_cache_cell_size_m: float = 10.0, map_cache_size: int = 4, cuda_prefetch_workers: int = 2, cuda_prefetch_depth: int = 8, cleanup_interval_scans: int = 0) -> list[LidarMapRegistrationResult]:
    """Independently localize raw LiDAR scans with device-specific throughput optimizations.

    CPU mode uses multiple forked worker processes on Linux/WSL because every scan is independently seeded from the reference trajectory. CUDA mode keeps one ICP stream on one GPU, reuses preprocessed target-map crops on the GPU, and overlaps raw scan file loading with GPU work through a small CPU thread prefetch queue.
    """
    if str(initial_guess_source).strip().lower() != 'reference':
        raise ValueError("Only initial_guess_source='reference' is implemented for this diagnostic notebook.")
    config = _RegistrationConfig(voxel_size_m=scan_voxel_size_m, max_correspondence_distance_m=max_correspondence_distance_m, max_iterations=max_iterations, voxel_scale_factors=voxel_scale_factors, robust_kernel=robust_kernel, robust_kernel_scale_factor=robust_kernel_scale_factor, device=choose_open3d_device(device, cuda_device_id), cuda_device_id=cuda_device_id, normal_radius_factor=normal_radius_factor, normal_max_nn=normal_max_nn)
    scans = sort_by_filename([Path(path).expanduser() for path in scan_paths])
    scan_timestamps = scan_timestamps_from_paths(scans, scan_period_s=scan_period_s)
    scan_step = int(scan_step)
    if scan_step <= 0:
        raise ValueError('scan_step must be positive.')
    indices = np.arange(0, len(scans), scan_step, dtype=int)
    if max_scans is not None:
        indices = indices[:int(max_scans)]
    scans = [scans[index] for index in indices]
    scan_timestamps = scan_timestamps[indices]
    if not scans:
        return []
    tasks = _prepare_localization_tasks(scans, scan_timestamps, reference_timestamps_s, reference_poses_T_W_B, T_B_L, max_reference_mismatch_s=max_reference_mismatch_s)
    if config.device == 'cuda':
        return _localize_tasks_cuda(reference_map, tasks, config=config, map_crop_radius_m=map_crop_radius_m, map_crop_z_margin_m=map_crop_z_margin_m, map_cache_cell_size_m=map_cache_cell_size_m, map_cache_size=map_cache_size, cuda_prefetch_workers=cuda_prefetch_workers, cuda_prefetch_depth=cuda_prefetch_depth, cleanup_interval_scans=cleanup_interval_scans, progress=progress)
    return _localize_tasks_cpu(reference_map, tasks, config=config, map_crop_radius_m=map_crop_radius_m, map_crop_z_margin_m=map_crop_z_margin_m, map_cache_cell_size_m=map_cache_cell_size_m, map_cache_size=map_cache_size, cpu_workers=cpu_workers, cpu_threads_per_worker=cpu_threads_per_worker, cleanup_interval_scans=cleanup_interval_scans, progress=progress)


def _prepare_localization_tasks(scans: Sequence[Path], scan_timestamps: np.ndarray, reference_timestamps_s: Sequence[float], reference_poses_T_W_B: np.ndarray, T_B_L: np.ndarray, *, max_reference_mismatch_s: float | None) -> list[_LocalizationTask]:
    tasks: list[_LocalizationTask] = []
    for sequence_index, (scan_path, timestamp_s) in enumerate(zip(scans, scan_timestamps)):
        try:
            T_W_B_initial, mismatch_s = interpolate_reference_body_pose(reference_timestamps_s, reference_poses_T_W_B, float(timestamp_s), max_mismatch_s=max_reference_mismatch_s)
            T_W_L_initial = compose_initial_lidar_pose(T_W_B_initial, T_B_L)
            tasks.append(_LocalizationTask(sequence_index=sequence_index, scan_path=scan_path, timestamp_s=float(timestamp_s), T_W_L_initial=T_W_L_initial, reference_timestamp_mismatch_s=float(mismatch_s)))
        except Exception as exc:
            tasks.append(_LocalizationTask(sequence_index=sequence_index, scan_path=scan_path, timestamp_s=float(timestamp_s), T_W_L_initial=None, reference_timestamp_mismatch_s=np.nan, preparation_error=str(exc)))
    return tasks


def _localize_tasks_cuda(reference_map: LidarReferenceMap, tasks: Sequence[_LocalizationTask], *, config: _RegistrationConfig, map_crop_radius_m: float, map_crop_z_margin_m: float | None, map_cache_cell_size_m: float, map_cache_size: int, cuda_prefetch_workers: int, cuda_prefetch_depth: int, cleanup_interval_scans: int, progress: bool) -> list[LidarMapRegistrationResult]:
    o3d = _import_open3d()
    open3d_device = _resolve_open3d_device(o3d, config)
    cache = _MapTargetCache(reference_map, config=config, open3d=o3d, open3d_device=open3d_device, map_crop_radius_m=map_crop_radius_m, map_crop_z_margin_m=map_crop_z_margin_m, cell_size_m=map_cache_cell_size_m, max_entries=map_cache_size)
    iterator = _prefetched_scan_iterator(tasks, workers=max(int(cuda_prefetch_workers), 0), depth=max(int(cuda_prefetch_depth), 1))
    if progress:
        iterator = tqdm(iterator, total=len(tasks), desc='LAS scan-to-map ICP [cuda]')
    results: list[LidarMapRegistrationResult] = []
    for processed_count, (task, scan_xyz_L, load_error) in enumerate(iterator, start=1):
        results.append(_localize_one_task(task, cache=cache, config=config, open3d=o3d, open3d_device=open3d_device, scan_xyz_L=scan_xyz_L, scan_load_error=load_error))
        if cleanup_interval_scans > 0 and processed_count % int(cleanup_interval_scans) == 0:
            gc.collect()
    return results


def _prefetched_scan_iterator(tasks: Sequence[_LocalizationTask], *, workers: int, depth: int):
    if workers <= 0:
        for task in tasks:
            if task.preparation_error:
                yield task, None, None
            else:
                try:
                    yield task, load_lidar_scan_xyz(task.scan_path), None
                except Exception as exc:
                    yield task, None, str(exc)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='lidar-scan-prefetch') as executor:
        pending: deque[tuple[_LocalizationTask, Any]] = deque()
        task_iterator = iter(tasks)
        def submit_until_full() -> None:
            while len(pending) < depth:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    break
                future = None if task.preparation_error else executor.submit(load_lidar_scan_xyz, task.scan_path)
                pending.append((task, future))
        submit_until_full()
        while pending:
            task, future = pending.popleft()
            if future is None:
                yield task, None, None
            else:
                try:
                    yield task, future.result(), None
                except Exception as exc:
                    yield task, None, str(exc)
            submit_until_full()


def _localize_tasks_cpu(reference_map: LidarReferenceMap, tasks: Sequence[_LocalizationTask], *, config: _RegistrationConfig, map_crop_radius_m: float, map_crop_z_margin_m: float | None, map_cache_cell_size_m: float, map_cache_size: int, cpu_workers: int | None, cpu_threads_per_worker: int, cleanup_interval_scans: int, progress: bool) -> list[LidarMapRegistrationResult]:
    workers = _resolve_cpu_worker_count(cpu_workers)
    if workers <= 1:
        o3d = _import_open3d()
        if hasattr(o3d.utility, 'set_max_threads') and int(cpu_threads_per_worker) > 0:
            o3d.utility.set_max_threads(int(cpu_threads_per_worker))
        open3d_device = _resolve_open3d_device(o3d, config)
        cache = _MapTargetCache(reference_map, config=config, open3d=o3d, open3d_device=open3d_device, map_crop_radius_m=map_crop_radius_m, map_crop_z_margin_m=map_crop_z_margin_m, cell_size_m=map_cache_cell_size_m, max_entries=map_cache_size)
        iterator: Iterable[_LocalizationTask] = tqdm(tasks, total=len(tasks), desc='LAS scan-to-map ICP [cpu]') if progress else tasks
        results = []
        for processed_count, task in enumerate(iterator, start=1):
            results.append(_localize_one_task(task, cache=cache, config=config, open3d=o3d, open3d_device=open3d_device))
            if cleanup_interval_scans > 0 and processed_count % int(cleanup_interval_scans) == 0:
                gc.collect()
        return results
    if 'fork' not in mp.get_all_start_methods():
        warnings.warn("Parallel CPU localization needs multiprocessing 'fork' to share the large read-only LAS map without duplicating it. Falling back to one CPU process.", RuntimeWarning, stacklevel=2)
        return _localize_tasks_cpu(reference_map, tasks, config=config, map_crop_radius_m=map_crop_radius_m, map_crop_z_margin_m=map_crop_z_margin_m, map_cache_cell_size_m=map_cache_cell_size_m, map_cache_size=map_cache_size, cpu_workers=1, cpu_threads_per_worker=0, cleanup_interval_scans=cleanup_interval_scans, progress=progress)
    global _CPU_REFERENCE_MAP
    _CPU_REFERENCE_MAP = reference_map
    chunks = [list(chunk) for chunk in np.array_split(np.asarray(tasks, dtype=object), min(workers, len(tasks))) if len(chunk)]
    results_by_index: dict[int, LidarMapRegistrationResult] = {}
    try:
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=mp.get_context('fork'), initializer=_initialize_cpu_worker, initargs=(config, map_crop_radius_m, map_crop_z_margin_m, map_cache_cell_size_m, map_cache_size, int(cpu_threads_per_worker), int(cleanup_interval_scans))) as executor:
            futures = {executor.submit(_cpu_worker_localize_chunk, chunk): len(chunk) for chunk in chunks}
            progress_bar = tqdm(total=len(tasks), desc=f'LAS scan-to-map ICP [cpu x{len(chunks)}]') if progress else None
            try:
                for future in as_completed(futures):
                    for sequence_index, result in future.result():
                        results_by_index[int(sequence_index)] = result
                    if progress_bar is not None:
                        progress_bar.update(futures[future])
            finally:
                if progress_bar is not None:
                    progress_bar.close()
    finally:
        _CPU_REFERENCE_MAP = None
    return [results_by_index[index] for index in range(len(tasks))]


def _resolve_cpu_worker_count(cpu_workers: int | None) -> int:
    if cpu_workers is not None:
        return max(int(cpu_workers), 1)
    physical_cores = None
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
    except Exception:
        physical_cores = None
    if not physical_cores:
        physical_cores = max(1, (os.cpu_count() or 1) // 2)
    return max(1, min(int(physical_cores), 8))


def _initialize_cpu_worker(config: _RegistrationConfig, map_crop_radius_m: float, map_crop_z_margin_m: float | None, map_cache_cell_size_m: float, map_cache_size: int, cpu_threads_per_worker: int, cleanup_interval_scans: int) -> None:
    global _CPU_WORKER_CONFIG, _CPU_WORKER_OPEN3D, _CPU_WORKER_DEVICE, _CPU_WORKER_MAP_CACHE, _CPU_WORKER_CLEANUP_INTERVAL
    if _CPU_REFERENCE_MAP is None:
        raise RuntimeError('CPU worker did not inherit the reference map through fork.')
    o3d = _import_open3d()
    if hasattr(o3d.utility, 'set_max_threads') and int(cpu_threads_per_worker) > 0:
        o3d.utility.set_max_threads(int(cpu_threads_per_worker))
    _CPU_WORKER_CONFIG = config
    _CPU_WORKER_OPEN3D = o3d
    _CPU_WORKER_DEVICE = _resolve_open3d_device(o3d, config)
    _CPU_WORKER_MAP_CACHE = _MapTargetCache(_CPU_REFERENCE_MAP, config=config, open3d=o3d, open3d_device=_CPU_WORKER_DEVICE, map_crop_radius_m=map_crop_radius_m, map_crop_z_margin_m=map_crop_z_margin_m, cell_size_m=map_cache_cell_size_m, max_entries=map_cache_size)
    _CPU_WORKER_CLEANUP_INTERVAL = int(cleanup_interval_scans)


def _cpu_worker_localize_chunk(tasks: Sequence[_LocalizationTask]) -> list[tuple[int, LidarMapRegistrationResult]]:
    if _CPU_WORKER_CONFIG is None or _CPU_WORKER_OPEN3D is None or _CPU_WORKER_MAP_CACHE is None:
        raise RuntimeError('CPU worker was not initialized.')
    indexed_results = []
    for processed_count, task in enumerate(tasks, start=1):
        result = _localize_one_task(task, cache=_CPU_WORKER_MAP_CACHE, config=_CPU_WORKER_CONFIG, open3d=_CPU_WORKER_OPEN3D, open3d_device=_CPU_WORKER_DEVICE)
        indexed_results.append((task.sequence_index, result))
        if _CPU_WORKER_CLEANUP_INTERVAL > 0 and processed_count % _CPU_WORKER_CLEANUP_INTERVAL == 0:
            gc.collect()
    return indexed_results


def _localize_one_task(task: _LocalizationTask, *, cache: _MapTargetCache, config: _RegistrationConfig, open3d, open3d_device, scan_xyz_L: np.ndarray | None = None, scan_load_error: str | None = None) -> LidarMapRegistrationResult:
    if task.preparation_error:
        return _make_failure_result(task, task.preparation_error)
    if scan_load_error is not None:
        return _make_failure_result(task, scan_load_error)
    try:
        cached_target = cache.get(task.T_W_L_initial)
        if scan_xyz_L is None:
            scan_xyz_L = load_lidar_scan_xyz(task.scan_path)
        T_W_L_estimated, fitness, inlier_rmse, scan_points_used, diagnostics = _register_preloaded_scan_to_cached_target(scan_xyz_L, cached_target, task.T_W_L_initial, config=config, open3d=open3d, open3d_device=open3d_device)
        diagnostics.update({'map_cache_origin_xyz': cached_target.origin_xyz.copy(), 'map_cache_cell_size_m': cache.cell_size_m, 'used_cached_target_map': True})
        return _make_success_result(timestamp_s=task.timestamp_s, scan_path=task.scan_path, T_W_L_initial=task.T_W_L_initial, T_W_L_estimated=T_W_L_estimated, fitness=fitness, inlier_rmse=inlier_rmse, scan_points_used=scan_points_used, map_points_used=cached_target.map_points_used, reference_timestamp_mismatch_s=task.reference_timestamp_mismatch_s, diagnostics=diagnostics)
    except Exception as exc:
        return _make_failure_result(task, str(exc))


def _make_failure_result(task: _LocalizationTask, message: str) -> LidarMapRegistrationResult:
    return LidarMapRegistrationResult(timestamp_s=float(task.timestamp_s), scan_path=task.scan_path, T_W_L_initial=task.T_W_L_initial, T_W_L_estimated=None, fitness=np.nan, inlier_rmse=np.nan, success=False, translation_correction_m=np.nan, rotation_correction_deg=np.nan, map_points_used=0, scan_points_used=0, reference_timestamp_mismatch_s=float(task.reference_timestamp_mismatch_s), message=str(message))


def _register_preloaded_scan_to_cached_target(scan_xyz_L: np.ndarray, cached_target: _CachedMapTarget, T_W_L_initial: np.ndarray, *, config: _RegistrationConfig, open3d, open3d_device) -> tuple[np.ndarray, float, float, int, dict[str, Any]]:
    source = _make_tensor_cloud(open3d, scan_xyz_L, open3d_device)
    source = _preprocess_tensor_cloud(open3d, source, config, voxel_size_m=config.voxel_size_m * min(config.voxel_scale_factors), estimate_normals=False)
    scan_points_used = int(source.point.positions.shape[0])
    if scan_points_used == 0:
        raise ValueError('Voxel-downsampled scan is empty.')
    T_O_L_initial = _as_transform(T_W_L_initial, 'T_W_L_initial').copy()
    T_O_L_initial[:3, 3] -= cached_target.origin_xyz
    T_O_L_estimated, fitness, inlier_rmse, diagnostics = _register_tensor_multiscale(open3d, source=source, target=cached_target.target, initial_transform=T_O_L_initial, config=config)
    T_W_L_estimated = restore_local_origin_transform(T_O_L_estimated, cached_target.origin_xyz)
    diagnostics.update({"local_origin_xyz": cached_target.origin_xyz.copy(), "used_local_origin_compensation": True})
    return T_W_L_estimated, fitness, inlier_rmse, scan_points_used, diagnostics


def register_lidar_scan_to_map_crop(scan_path: str | Path, map_crop_W: np.ndarray, T_W_L_initial: np.ndarray, *, config: _RegistrationConfig, open3d=None, open3d_device=None) -> tuple[np.ndarray, float, float, int, int, dict[str, Any]]:
    """Register one raw LiDAR scan to one world-frame LAS crop and return absolute ``T_W_L``."""
    o3d = _import_open3d() if open3d is None else open3d
    device_obj = _resolve_open3d_device(o3d, config) if open3d_device is None else open3d_device
    scan_xyz_L = load_lidar_scan_xyz(scan_path)
    local_input = registration_inputs_to_local_origin(T_W_L_initial, map_crop_W)
    source = _make_tensor_cloud(o3d, scan_xyz_L, device_obj)
    target = _make_tensor_cloud(o3d, local_input.target_points_O, device_obj)
    source = _preprocess_tensor_cloud(o3d, source, config, voxel_size_m=config.voxel_size_m * min(config.voxel_scale_factors), estimate_normals=False)
    target = _preprocess_tensor_cloud(o3d, target, config, voxel_size_m=config.voxel_size_m * min(config.voxel_scale_factors), estimate_normals=True)
    scan_points_used = int(source.point.positions.shape[0])
    map_points_used = int(target.point.positions.shape[0])
    if scan_points_used == 0:
        raise ValueError('Voxel-downsampled scan is empty.')
    if map_points_used == 0:
        raise ValueError('Voxel-downsampled LAS crop is empty.')
    T_O_L_estimated, fitness, inlier_rmse, diagnostics = _register_tensor_multiscale(o3d, source=source, target=target, initial_transform=local_input.T_O_L_initial, config=config)
    T_W_L_estimated = restore_local_origin_transform(T_O_L_estimated, local_input.origin_xyz)
    diagnostics.update({'local_origin_xyz': local_input.origin_xyz.copy(), 'used_local_origin_compensation': True})
    return T_W_L_estimated, fitness, inlier_rmse, scan_points_used, map_points_used, diagnostics


def pose_matrix_to_csv_values(T_W_L: np.ndarray | None) -> list[float]:
    '''Serialize the upper 3x4 transform row-major for the required CSV order.'''
    if T_W_L is None:
        return [float('nan')] * 12
    T = _as_transform(T_W_L, 'T_W_L')
    return [float(value) for value in T[:3, :].reshape(-1)]


def csv_row_to_pose_matrix(row: pd.Series | dict[str, Any] | Sequence[float]) -> np.ndarray:
    '''Reconstruct a 4x4 transform from the required row-major 3x4 CSV values.'''
    if isinstance(row, (pd.Series, dict)):
        values = [row[column] for column in POSE_CSV_COLUMNS[1:]]
    else:
        values = list(row)
        if len(values) == 13:
            values = values[1:13]
    if len(values) != 12:
        raise ValueError('Expected 12 pose values, or 13 values including timestamp.')
    T = np.eye(4, dtype=float)
    T[:3, :] = np.asarray(values, dtype=float).reshape(3, 4)
    return T


def save_lidar_map_pose_csv(results: Sequence[LidarMapRegistrationResult], path: str | Path) -> Path:
    '''Save one row per scan, keeping the first 13 columns exactly as specified.'''
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_registration_result_to_pose_csv_row(result) for result in results]
    pd.DataFrame(rows, columns=LIDAR_MAP_POSE_CSV_COLUMNS).to_csv(output_path, index=False, float_format='%.12g')
    return output_path


def load_lidar_map_pose_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    '''Load saved scan-to-map poses and reconstruct ``T_W_L`` matrices.'''
    csv_path = Path(path).expanduser()
    dataframe = pd.read_csv(csv_path)
    missing = [column for column in POSE_CSV_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f'Pose CSV is missing required columns: {missing}')
    timestamps = dataframe['timestamp'].to_numpy(dtype=float)
    poses = np.stack([csv_row_to_pose_matrix(row) for _, row in dataframe.iterrows()]) if len(dataframe) else np.empty((0, 4, 4), dtype=float)
    return timestamps, poses, dataframe


def registration_results_to_dataframe(results: Sequence[LidarMapRegistrationResult]) -> pd.DataFrame:
    '''Convert registration results to a notebook-friendly diagnostics table.'''
    rows = []
    for result in results:
        row = {
            'timestamp_s': result.timestamp_s,
            'scan_path': str(result.scan_path),
            'success': bool(result.success),
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'reference_timestamp_mismatch_s': result.reference_timestamp_mismatch_s,
            'translation_correction_m': result.translation_correction_m,
            'rotation_correction_deg': result.rotation_correction_deg,
            'scan_points_used': result.scan_points_used,
            'map_points_used': result.map_points_used,
            'message': result.message,
        }
        if result.T_W_L_initial is not None:
            row.update({'initial_x': result.T_W_L_initial[0, 3], 'initial_y': result.T_W_L_initial[1, 3], 'initial_z': result.T_W_L_initial[2, 3]})
        if result.T_W_L_estimated is not None:
            row.update({'estimated_x': result.T_W_L_estimated[0, 3], 'estimated_y': result.T_W_L_estimated[1, 3], 'estimated_z': result.T_W_L_estimated[2, 3]})
        rows.append(row)
    return pd.DataFrame(rows)


def transform_points(T_A_B: np.ndarray, points_B: np.ndarray) -> np.ndarray:
    '''Apply ``T_A_B`` to an ``(N, 3)`` point array expressed in frame B.'''
    T = _as_transform(T_A_B, 'T_A_B')
    points = np.asarray(points_B, dtype=float)
    return points @ T[:3, :3].T + T[:3, 3]


def plot_reference_map_overview(reference_map: LidarReferenceMap, *, reference_poses_T_W_B: np.ndarray | None = None, ax=None, max_map_points: int = 200_000, map_color: str = '0.65', trajectory_color: str = 'tab:orange', title: str = 'LAS map and reference trajectory'):
    '''Plot LAS map XY and optionally overlay the reference body trajectory.'''
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    points = reference_map.sample_points(max_map_points)
    ax.scatter(points[:, 0], points[:, 1], s=0.2, c=map_color, alpha=0.35, linewidths=0, label='LAS map')
    if reference_poses_T_W_B is not None and len(reference_poses_T_W_B):
        trajectory = np.asarray(reference_poses_T_W_B, dtype=float)[:, :3, 3]
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=trajectory_color, linewidth=1.5, label='Reference body trajectory')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], s=20, color='tab:green', label='Reference start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=20, color='tab:red', label='Reference end')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x_W [m]')
    ax.set_ylabel('y_W [m]')
    ax.set_title(title)
    ax.legend(loc='best')
    return ax


def plot_registration_result_projections(reference_map: LidarReferenceMap, result: LidarMapRegistrationResult, *, crop_radius_m: float = 50.0, crop_z_margin_m: float | None = 10.0, scan_voxel_size_m: float | None = 0.25, max_map_points: int = 80_000, max_scan_points: int = 80_000):
    '''Plot XY and XZ projections of a local LAS crop and one transformed scan.'''
    import matplotlib.pyplot as plt
    if result.T_W_L_estimated is None:
        raise ValueError('Cannot plot a failed registration without an estimated pose.')
    center = result.T_W_L_estimated[:3, 3]
    crop = reference_map.query_local_map(center, crop_radius_m, z_margin_m=crop_z_margin_m)
    if crop.shape[0] > max_map_points:
        crop = _sample_array_rows(crop, max_map_points, seed=0)
    scan = load_lidar_scan_xyz(result.scan_path)
    if scan_voxel_size_m is not None:
        scan, _ = voxel_downsample_points(scan, scan_voxel_size_m)
    scan_W = transform_points(result.T_W_L_estimated, scan)
    if scan_W.shape[0] > max_scan_points:
        scan_W = _sample_array_rows(scan_W, max_scan_points, seed=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].scatter(crop[:, 0], crop[:, 1], s=0.5, c='0.65', alpha=0.45, linewidths=0, label='LAS crop')
    axes[0].scatter(scan_W[:, 0], scan_W[:, 1], s=0.8, c='tab:blue', alpha=0.65, linewidths=0, label='Registered scan')
    axes[0].set_xlabel('x_W [m]')
    axes[0].set_ylabel('y_W [m]')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].legend(loc='best')
    axes[1].scatter(crop[:, 0], crop[:, 2], s=0.5, c='0.65', alpha=0.45, linewidths=0, label='LAS crop')
    axes[1].scatter(scan_W[:, 0], scan_W[:, 2], s=0.8, c='tab:blue', alpha=0.65, linewidths=0, label='Registered scan')
    axes[1].set_xlabel('x_W [m]')
    axes[1].set_ylabel('z_W [m]')
    axes[1].legend(loc='best')
    fig.suptitle(f'Scan-to-map overlay at {result.timestamp_s:.6f} s; fitness={result.fitness:.3f}, RMSE={result.inlier_rmse:.3f} m')
    return fig, axes


def plot_localization_trajectories(reference_map: LidarReferenceMap, results: Sequence[LidarMapRegistrationResult], *, reference_poses_T_W_B: np.ndarray | None = None, max_map_points: int = 200_000):
    '''Plot initial/reference-seeded LiDAR poses, estimated poses, and reference body trajectory.'''
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(8, 8))
    plot_reference_map_overview(reference_map, reference_poses_T_W_B=reference_poses_T_W_B, ax=ax, max_map_points=max_map_points, title='Scan-to-map LiDAR trajectory')
    initial = np.asarray([result.T_W_L_initial[:3, 3] for result in results if result.T_W_L_initial is not None], dtype=float)
    estimated = np.asarray([result.T_W_L_estimated[:3, 3] for result in results if result.success and result.T_W_L_estimated is not None], dtype=float)
    if initial.size:
        ax.plot(initial[:, 0], initial[:, 1], color='tab:cyan', linewidth=1.2, label='Initial T_W_L')
    if estimated.size:
        ax.plot(estimated[:, 0], estimated[:, 1], color='tab:blue', linewidth=1.5, label='ICP T_W_L')
        ax.scatter(estimated[:, 0], estimated[:, 1], s=10, color='tab:blue')
    ax.legend(loc='best')
    return ax


def plot_registration_diagnostics(results: Sequence[LidarMapRegistrationResult]):
    '''Plot correction magnitudes, fitness, and RMSE over LiDAR scan time.'''
    import matplotlib.pyplot as plt
    dataframe = registration_results_to_dataframe(results)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    axes[0].plot(dataframe['timestamp_s'], dataframe['translation_correction_m'], marker='.', linewidth=1.0)
    axes[0].set_ylabel('translation [m]')
    axes[0].set_title('ICP correction relative to reference initialization')
    axes[1].plot(dataframe['timestamp_s'], dataframe['rotation_correction_deg'], marker='.', linewidth=1.0)
    axes[1].set_ylabel('rotation [deg]')
    axes[2].plot(dataframe['timestamp_s'], dataframe['fitness'], marker='.', linewidth=1.0)
    axes[2].set_ylabel('fitness')
    axes[3].plot(dataframe['timestamp_s'], dataframe['inlier_rmse'], marker='.', linewidth=1.0)
    axes[3].set_ylabel('RMSE [m]')
    axes[3].set_xlabel('LiDAR timestamp [s]')
    for ax in axes:
        if 'success' in dataframe:
            failed = dataframe[~dataframe['success'].astype(bool)]
            if not failed.empty:
                ax.scatter(failed['timestamp_s'], np.full(len(failed), ax.get_ylim()[0]), marker='x', color='tab:red', label='failed')
        ax.grid(True, alpha=0.25)
    return fig, axes


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError('open3d is required for LiDAR scan-to-map ICP. Install it in the notebook environment with `pip install open3d`.') from exc
    return o3d

def _make_tensor_cloud(open3d, points_xyz: np.ndarray, open3d_device):
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Point cloud must have shape (N, 3).')
    tensor = open3d.core.Tensor(np.ascontiguousarray(points), dtype=open3d.core.Dtype.Float32, device=open3d_device)
    return open3d.t.geometry.PointCloud(tensor)


def _preprocess_tensor_cloud(open3d, cloud, config: _RegistrationConfig, *, voxel_size_m: float, estimate_normals: bool):
    processed = cloud.voxel_down_sample(float(voxel_size_m))
    if int(processed.point.positions.shape[0]) == 0:
        return processed
    if estimate_normals:
        processed.estimate_normals(max_nn=int(config.normal_max_nn), radius=max(float(config.normal_radius_factor) * float(voxel_size_m), 1e-3))
    return processed


def _register_tensor_multiscale(open3d, *, source, target, initial_transform: np.ndarray, config: _RegistrationConfig) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    registration = open3d.t.pipelines.registration
    voxel_sizes = open3d.utility.DoubleVector([float(config.voxel_size_m) * float(scale) for scale in config.voxel_scale_factors])
    max_correspondence_distances = open3d.utility.DoubleVector([float(config.max_correspondence_distance_m) * float(scale) for scale in config.voxel_scale_factors])
    criteria_list = [registration.ICPConvergenceCriteria(max_iteration=iterations) for iterations in _iterations_per_scale(int(config.max_iterations), len(config.voxel_scale_factors))]
    estimation = registration.TransformationEstimationPointToPlane(_tensor_robust_kernel(open3d, config))
    initial_tensor = open3d.core.Tensor(np.asarray(initial_transform, dtype=np.float64), dtype=open3d.core.Dtype.Float64, device=open3d.core.Device('CPU:0'))
    result = registration.multi_scale_icp(source, target, voxel_sizes, criteria_list, max_correspondence_distances, initial_tensor, estimation)
    transformation = np.asarray(result.transformation.cpu().numpy(), dtype=float)
    if transformation.shape != (4, 4) or not np.all(np.isfinite(transformation)):
        raise RuntimeError('Open3D Tensor ICP did not produce a finite 4x4 transformation.')
    return transformation, float(result.fitness), float(result.inlier_rmse), {'open3d_converged': True}


def _tensor_robust_kernel(open3d, config: _RegistrationConfig):
    robust_kernel_module = open3d.t.pipelines.registration.robust_kernel
    method_map = {'l2': robust_kernel_module.RobustKernelMethod.L2Loss, 'huber': robust_kernel_module.RobustKernelMethod.HuberLoss, 'cauchy': robust_kernel_module.RobustKernelMethod.CauchyLoss, 'tukey': robust_kernel_module.RobustKernelMethod.TukeyLoss}
    kernel = None if config.robust_kernel is None else str(config.robust_kernel).strip().lower()
    method = robust_kernel_module.RobustKernelMethod.L2Loss if kernel is None else method_map[kernel]
    scale = max(float(config.robust_kernel_scale_factor) * float(config.voxel_size_m), 1e-6)
    return robust_kernel_module.RobustKernel(method, scale, 1.0)


def _resolve_open3d_device(open3d, config: _RegistrationConfig):
    if config.device == 'cpu':
        return open3d.core.Device('CPU:0')
    if not open3d.core.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but Open3D reports that CUDA is unavailable")
    device_count = int(open3d.core.cuda.device_count())
    if int(config.cuda_device_id) >= device_count:
        raise ValueError(f'cuda_device_id={config.cuda_device_id} is invalid because Open3D reports {device_count} CUDA device(s)')
    return open3d.core.Device(f'CUDA:{int(config.cuda_device_id)}')


def _iterations_per_scale(total_iterations: int, number_of_scales: int) -> tuple[int, ...]:
    if number_of_scales == 1:
        return (int(total_iterations),)
    weights = np.linspace(number_of_scales, 1.0, number_of_scales)
    weights /= np.sum(weights)
    allocations = np.maximum(1, np.floor(weights * int(total_iterations)).astype(int))
    difference = int(total_iterations) - int(np.sum(allocations))
    index = 0
    while difference != 0:
        target = index % number_of_scales
        if difference > 0:
            allocations[target] += 1
            difference -= 1
        elif allocations[target] > 1:
            allocations[target] -= 1
            difference += 1
        index += 1
    return tuple(int(value) for value in allocations)




def _normalize_bounds(bounds_min_xyz: Sequence[float] | None, bounds_max_xyz: Sequence[float] | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if bounds_min_xyz is None and bounds_max_xyz is None:
        return None, None
    if bounds_min_xyz is None or bounds_max_xyz is None:
        raise ValueError('bounds_min_xyz and bounds_max_xyz must be provided together.')
    bounds_min = np.asarray(bounds_min_xyz, dtype=np.float64).reshape(3)
    bounds_max = np.asarray(bounds_max_xyz, dtype=np.float64).reshape(3)
    if np.any(bounds_min > bounds_max):
        raise ValueError('bounds_min_xyz must be less than or equal to bounds_max_xyz.')
    return bounds_min, bounds_max


def _make_success_result(*, timestamp_s: float, scan_path: Path, T_W_L_initial: np.ndarray, T_W_L_estimated: np.ndarray, fitness: float, inlier_rmse: float, scan_points_used: int, map_points_used: int, reference_timestamp_mismatch_s: float, diagnostics: dict[str, Any]) -> LidarMapRegistrationResult:
    translation_correction_m = float(np.linalg.norm(T_W_L_estimated[:3, 3] - T_W_L_initial[:3, 3]))
    rotation_correction_deg = _rotation_difference_deg(T_W_L_initial[:3, :3], T_W_L_estimated[:3, :3])
    return LidarMapRegistrationResult(timestamp_s=timestamp_s, scan_path=scan_path, T_W_L_initial=T_W_L_initial, T_W_L_estimated=T_W_L_estimated, fitness=float(fitness), inlier_rmse=float(inlier_rmse), success=bool(np.isfinite(fitness) and np.isfinite(inlier_rmse)), translation_correction_m=translation_correction_m, rotation_correction_deg=rotation_correction_deg, map_points_used=int(map_points_used), scan_points_used=int(scan_points_used), reference_timestamp_mismatch_s=float(reference_timestamp_mismatch_s), diagnostics=diagnostics)


def _registration_result_to_pose_csv_row(result: LidarMapRegistrationResult) -> dict[str, Any]:
    pose_values = pose_matrix_to_csv_values(result.T_W_L_estimated if result.success else None)
    values = [float(result.timestamp_s), *pose_values, bool(result.success), float(result.fitness), float(result.inlier_rmse), float(result.reference_timestamp_mismatch_s), float(result.translation_correction_m), float(result.rotation_correction_deg), int(result.scan_points_used), int(result.map_points_used)]
    return dict(zip(LIDAR_MAP_POSE_CSV_COLUMNS, values))


def _rotation_difference_deg(R_initial: np.ndarray, R_estimated: np.ndarray) -> float:
    R_delta = np.asarray(R_estimated, dtype=float) @ np.asarray(R_initial, dtype=float).T
    cos_angle = np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def _as_transform(transform: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(transform, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f'{name} must have shape (4, 4).')
    if not np.all(np.isfinite(T)):
        raise ValueError(f'{name} contains non-finite values.')
    return T


def _sample_array_rows(values: np.ndarray, max_rows: int, *, seed: int = 0) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] <= int(max_rows):
        return values
    rng = np.random.default_rng(seed)
    indices = rng.choice(values.shape[0], size=int(max_rows), replace=False)
    return values[np.sort(indices)]