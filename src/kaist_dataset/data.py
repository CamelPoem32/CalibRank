'''Data containers and low-level KAIST trajectory/calibration importers.

The dataclasses in this module keep the same normalized representations used by
higher-level calibration code. Dataset-specific parsing is limited to the raw
KAIST CSV/TXT formats so downstream optimization code can stay dataset-agnostic.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
import re
from typing import Any, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IMUData:
    '''Synchronized accelerometer and gyroscope measurements for one IMU.

    Args:
        timestamps_s: Monotonic timestamps in seconds with shape ``(N,)``.
        accel_mps2: Accelerometer samples in meters per second squared, shape
            ``(N, 3)``.
        gyro_radps: Gyroscope samples in radians per second, shape ``(N, 3)``.
        name: Human-readable stream name.
        frame_id: Optional sensor frame name from the source data.
        frequency_hz: Nominal resampled frequency in hertz.
        metadata: Extra loader details such as source path and raw format.
    '''

    timestamps_s: np.ndarray
    accel_mps2: np.ndarray
    gyro_radps: np.ndarray
    name: str = 'imu'
    frame_id: str | None = None
    frequency_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'timestamps_s', np.asarray(self.timestamps_s, dtype=float))
        object.__setattr__(self, 'accel_mps2', np.asarray(self.accel_mps2, dtype=float))
        object.__setattr__(self, 'gyro_radps', np.asarray(self.gyro_radps, dtype=float))
        _validate_sensor_triplet(self.timestamps_s, self.accel_mps2, self.gyro_radps, self.name)


@dataclass(frozen=True)
class LidarData:
    '''Pairwise LiDAR odometry estimates as relative SE(3) matrices.

    Args:
        timestamps_s: Timestamps in seconds for each relative-pose interval,
            usually midpoint times, shape ``(N,)``.
        relative_poses_se3: Relative transforms computed between consecutive
            scans with shape ``(N, 4, 4)``.
        scan_timestamps_s: Optional original scan timestamps with shape
            ``(N + 1,)``. Use these for exact velocity durations when available.
        source_scan_paths: Ordered raw scan paths used to compute the poses.
        fitness: ICP fitness values, shape ``(N,)``.
        inlier_rmse: ICP inlier RMSE values, shape ``(N,)``.
        metadata: Extra loader and ICP settings.
    '''

    timestamps_s: np.ndarray
    relative_poses_se3: np.ndarray
    scan_timestamps_s: np.ndarray | None = None
    source_scan_paths: list[str] = field(default_factory=list)
    fitness: np.ndarray | None = None
    inlier_rmse: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'timestamps_s', np.asarray(self.timestamps_s, dtype=float))
        object.__setattr__(self, 'relative_poses_se3', np.asarray(self.relative_poses_se3, dtype=float))

        if self.scan_timestamps_s is not None:
            object.__setattr__(self, 'scan_timestamps_s', np.asarray(self.scan_timestamps_s, dtype=float))
        if self.fitness is not None:
            object.__setattr__(self, 'fitness', np.asarray(self.fitness, dtype=float))
        if self.inlier_rmse is not None:
            object.__setattr__(self, 'inlier_rmse', np.asarray(self.inlier_rmse, dtype=float))

        _validate_lidar(self)


##################################################
# Shared validation helpers
##################################################


def _validate_sensor_triplet(
    timestamps_s: np.ndarray,
    accel_mps2: np.ndarray,
    gyro_radps: np.ndarray,
    name: str,
) -> None:
    if timestamps_s.ndim != 1:
        raise ValueError(f'{name}: timestamps_s must have shape (N,).')
    if accel_mps2.shape != (timestamps_s.size, 3):
        raise ValueError(f'{name}: accel_mps2 must have shape (N, 3).')
    if gyro_radps.shape != (timestamps_s.size, 3):
        raise ValueError(f'{name}: gyro_radps must have shape (N, 3).')
    if timestamps_s.size and not np.all(np.isfinite(timestamps_s)):
        raise ValueError(f'{name}: timestamps_s contains non-finite values.')
    if timestamps_s.size > 1 and np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError(f'{name}: timestamps_s must be strictly increasing.')


def _validate_lidar(lidar_data: LidarData) -> None:
    poses = lidar_data.relative_poses_se3
    n_poses = poses.shape[0] if poses.ndim == 3 else -1

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError('relative_poses_se3 must have shape (N, 4, 4).')
    if lidar_data.timestamps_s.shape != (n_poses,):
        raise ValueError('timestamps_s must have shape (N,) for N relative poses.')
    if n_poses and not np.all(np.isfinite(lidar_data.timestamps_s)):
        raise ValueError('timestamps_s contains non-finite values.')
    if n_poses > 1 and np.any(np.diff(lidar_data.timestamps_s) <= 0.0):
        raise ValueError('timestamps_s must be strictly increasing.')

    if lidar_data.scan_timestamps_s is not None:
        if lidar_data.scan_timestamps_s.shape != (n_poses + 1,):
            raise ValueError('scan_timestamps_s must have shape (N + 1,).')
        if np.any(np.diff(lidar_data.scan_timestamps_s) <= 0.0):
            raise ValueError('scan_timestamps_s must be strictly increasing.')


##################################################
# KAIST timestamp conversion
##################################################


def timestamps_ns_to_s(timestamps_ns: np.ndarray) -> np.ndarray:
    '''Convert integer ROS timestamps in nanoseconds to floating-point seconds.

    KAIST stores timestamps as integer nanoseconds. Splitting each timestamp into
    integer seconds and nanoseconds avoids first converting the approximately
    1e18-valued integer directly to float.
    '''

    timestamps_ns = np.asarray(timestamps_ns)
    if timestamps_ns.ndim != 1:
        raise ValueError('timestamps_ns must have shape (N,).')

    try:
        timestamps_ns = timestamps_ns.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('timestamps_ns must contain valid int64 nanosecond timestamps.') from exc

    seconds = timestamps_ns // 1_000_000_000
    nanoseconds = timestamps_ns % 1_000_000_000
    return seconds.astype(np.float64) + nanoseconds.astype(np.float64) * 1e-9


##################################################
# KAIST global-pose trajectory importer
##################################################


def import_true_trajectory(file_path: Any) -> Tuple[np.ndarray, np.ndarray]:
    '''Import a KAIST ``global_pose.csv`` trajectory as timestamped SE(3) poses.

    The expected row format is::

        timestamp_ns,
        P00,P01,P02,P03,
        P10,P11,P12,P13,
        P20,P21,P22,P23

    where the 12 pose values are the first three rows of a 4x4 homogeneous
    transformation matrix. The matrix is returned exactly in the convention
    stored by KAIST; this importer does not invert, rebase, or recenter it.

    Returns
    -------
    trajectory_timestamps
        Array of shape ``(N,)`` containing absolute timestamps in seconds.

    trajectory_poses
        Array of shape ``(N, 4, 4)`` containing homogeneous transformations.
    '''

    file_path = Path(file_path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f'Trajectory file does not exist: {file_path}')

    # Read the timestamp column as int64 so nanosecond precision is not lost by
    # first parsing the approximately 1e18-valued field as floating point.
    try:
        trajectory_data = pd.read_csv(
            file_path,
            header=None,
            comment='#',
            skip_blank_lines=True,
            dtype={0: 'int64'},
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Could not parse KAIST trajectory CSV: {file_path}') from exc

    trajectory_data = trajectory_data.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if trajectory_data.empty:
        raise ValueError(f'Trajectory file is empty: {file_path}')
    if trajectory_data.shape[1] != 13:
        raise ValueError(
            'Expected 13 columns: timestamp_ns followed by a 3x4 pose matrix; '
            f'received {trajectory_data.shape[1]}'
        )

    timestamps_ns = trajectory_data.iloc[:, 0].to_numpy(dtype=np.int64)
    pose_values = trajectory_data.iloc[:, 1:13].apply(
        pd.to_numeric,
        errors='coerce',
    ).to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(pose_values)):
        raise ValueError('Trajectory file contains non-finite pose values.')

    trajectory_timestamps = timestamps_ns_to_s(timestamps_ns)
    if trajectory_timestamps.size > 1 and np.any(np.diff(trajectory_timestamps) <= 0.0):
        raise ValueError('Trajectory timestamps must be strictly increasing.')

    # Reconstruct the bottom homogeneous row instead of modifying the supplied
    # 3x4 matrix values or projecting the rotation onto SO(3).
    number_poses = trajectory_timestamps.size
    trajectory_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], number_poses, axis=0)
    trajectory_poses[:, :3, :] = pose_values.reshape(-1, 3, 4)

    return trajectory_timestamps, trajectory_poses


##################################################
# KAIST extrinsic-calibration importer
##################################################


_FLOAT_PATTERN = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')


def import_extrinsic_calibration(file_path: Any) -> np.ndarray:
    '''Import one KAIST ``Vehicle2*.txt`` extrinsic as a 4x4 matrix.

    KAIST calibration files contain human-readable ``RPY:``, ``R:``, and ``T:``
    lines. This function uses the supplied rotation matrix and translation
    directly and returns the represented transform without inverting it.

    The function intentionally does not assign frame-direction semantics beyond
    what is encoded by the source filename. Higher-level code can therefore
    decide whether a particular matrix should be used directly or inverted for
    its own ``T_A_B`` convention.
    '''

    file_path = Path(file_path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f'Calibration file does not exist: {file_path}')

    text = unescape(file_path.read_text(encoding='utf-8', errors='replace'))
    rotation_values = _numbers_after_prefix(text, 'R:', expected_count=9)
    translation_values = _numbers_after_prefix(text, 'T:', expected_count=3)

    rotation = np.asarray(rotation_values, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_values, dtype=np.float64)

    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError(f'Calibration file contains non-finite values: {file_path}')

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def _numbers_after_prefix(text: str, prefix: str, *, expected_count: int) -> list[float]:
    '''Extract numeric values from the first line beginning with ``prefix``.'''

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue

        values = [float(token) for token in _FLOAT_PATTERN.findall(line[len(prefix):])]
        if len(values) != expected_count:
            raise ValueError(
                f'Expected {expected_count} numeric values after {prefix!r}; '
                f'received {len(values)}.'
            )
        return values

    raise ValueError(f'Could not find a {prefix!r} line in calibration file.')