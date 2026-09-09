"""Regression tests for KAIST ingestion, coordinate conversion and early limits."""

from pathlib import Path
from types import SimpleNamespace
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kaist_dataset.observability import (
    KaistObservabilityConfig, prepare_observability_inputs, uniform_sample_indices,
)
from kaist_dataset.run_kaist_multi_sensor_pipeline import (
    resolve_processed_data_directory, T_B_LL_INITIAL, T_B_RL_INITIAL,
)
from calib_observability.lie_se3 import se3_exp, se3_log
from calib_observability.visualization import quasi_realtime_rover as visual
from calib_observability.workflows import DiscretePoseTrajectory


@pytest.mark.parametrize("side", ["left", "right"])
def test_discovery_requires_only_selected_lidar(tmp_path, side):
    csv = tmp_path / f"lidar_map_poses_vlp_{side}.csv"
    csv.touch()
    assert resolve_processed_data_directory(tmp_path, "UrbanTest", tmp_path, "poses", side) == tmp_path
    with pytest.raises(FileNotFoundError):
        resolve_processed_data_directory(tmp_path, "UrbanTest", tmp_path, "poses",
                                         "right" if side == "left" else "left")


@pytest.mark.parametrize("side,extrinsic", [("left", T_B_LL_INITIAL), ("right", T_B_RL_INITIAL)])
def test_preparation_preserves_frames_times_and_skipped_motion(tmp_path, side, extrinsic):
    epoch = 1_520_000_000.0
    times = epoch + np.arange(21, dtype=float)
    T_B_L = se3_exp(se3_log(extrinsic))
    body = np.stack([se3_exp([0, 0, t / 100, t, 0, 0]) for t in range(21)])
    body[:, :3, 3] += [1e6, 2e6, 100]
    lidar = body @ T_B_L
    imu_times = epoch + np.arange(0, 20.01, 0.1)
    values = np.column_stack((imu_times - epoch, np.zeros(len(imu_times)), np.ones(len(imu_times))))
    imu = SimpleNamespace(timestamps_s=imu_times, gyro_radps=values, accel_mps2=values + 10)
    config = KaistObservabilityConfig(tmp_path, lidar=side, start_time=2, end_time=18,
        max_lidar_poses=5, max_ground_truth_poses=7, max_imu_samples=31)
    prepared = prepare_observability_inputs(imu, times, lidar, times, body, config)
    dataset = prepared.dataset
    assert dataset.start_time == 0
    assert dataset.end_time == 16
    assert len(dataset.lidar.measurements) == 4
    assert len(dataset.imu.gyroscope) <= 31
    assert len(prepared.reference_trajectory.timestamps) <= 7
    np.testing.assert_allclose(dataset.imu.accelerometer, dataset.imu.gyroscope + 10)
    np.testing.assert_allclose(dataset.imu.gyroscope[:, 0],
                               dataset.imu.sensor_timestamps + 2, atol=1e-6)
    np.testing.assert_allclose(dataset.T_B_L_true, np.eye(4))
    # Reframing once returns sensor poses, not body poses with a double lever arm.
    chosen = np.array([2, 6, 10, 14, 18])
    local = prepared.metadata["T_local_world"] @ lidar[chosen]
    for t, expected in zip(dataset.lidar.sensor_timestamps, local):
        np.testing.assert_allclose(dataset.trajectory.pose_at(t), expected, atol=1e-7)
    for index, relative in enumerate(dataset.lidar.measurements):
        np.testing.assert_allclose(local[index] @ relative, local[index + 1], atol=1e-7)
    # Reference in the displayed sensor frame agrees at exact shared knots.
    np.testing.assert_allclose(prepared.reference_trajectory.pose_at(0),
                               dataset.trajectory.pose_at(0), atol=1e-6)


def test_sample_limits_and_invalid_configuration(tmp_path):
    assert np.array_equal(uniform_sample_indices(101, 3), [0, 50, 100])
    for kwargs in ({"max_lidar_poses": 1}, {"mp4_fps": 0},
                   {"end_time": -1}, {"gyro_noise_std": np.nan}):
        with pytest.raises(ValueError):
            KaistObservabilityConfig(tmp_path, **kwargs)


def test_analysis_cap_applies_before_snapshot_construction(monkeypatch):
    assert visual.QuasiRealtimeConfig(5.0, 1.0, True).use_sparse is True
    calls = []
    def record(dataset, provider, **kwargs):
        calls.append(kwargs["current_time"])
        return kwargs
    monkeypatch.setattr(visual, "build_window_snapshot", record)
    dataset = SimpleNamespace(start_time=0.0, end_time=1e9)
    config = visual.QuasiRealtimeConfig(frame_step=0.01, max_analysis_windows=5)
    result = visual.compute_quasi_realtime_snapshots(dataset, None, config)
    assert len(result) == 5
    assert calls[0] == 0 and calls[-1] == 1e9
    assert np.all(np.diff(calls) > 0)
    np.testing.assert_allclose(visual.analysis_frame_times(0, 1.6, 1), [0, 1, 1.6])
    with pytest.raises(ValueError):
        visual.analysis_frame_times(0, 10, 1, 0)


def test_subprocess_payload_is_capped_and_contains_reference(tmp_path, monkeypatch):
    snapshots = [SimpleNamespace(current_time=float(i)) for i in range(20)]
    monkeypatch.setattr(visual, "_pickleable_snapshot_for_animation", lambda snapshot: snapshot)
    trajectory = DiscretePoseTrajectory(np.array([0, 19]), np.stack([np.eye(4), se3_exp([0, 0, 0, 19, 0, 0])]))
    seen = {}
    def run(command, **kwargs):
        with open(command[command.index("--payload") + 1], "rb") as handle:
            seen.update(pickle.load(handle))
        Path(command[command.index("--output-mp4") + 1]).write_bytes(b"test")
        return SimpleNamespace(stdout="", stderr="")
    monkeypatch.setattr(visual, "_run_renderer_subprocess", run)
    visual.save_quasi_realtime_rover_animation_mp4_subprocess(
        SimpleNamespace(trajectory=trajectory), snapshots, tmp_path / "movie.mp4",
        max_rendered_frames=3, trajectory_samples=4, reference_trajectory=trajectory,
    )
    assert [s.current_time for s in seen["snapshots"]] == [0, 10, 19]
    assert len(seen["dataset"].trajectory.timestamps) <= 7
    assert len(seen["render_kwargs"]["reference_trajectory"].timestamps) <= 7
    assert not list(tmp_path.glob("*.html"))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_cli_verbosity_and_live_subprocess_output(level, capsys):
    from kaist_dataset.run_kaist_observability import build_argument_parser

    args = build_argument_parser().parse_args(["Urban16", "--verbose", str(level), "--n-processes", "2"])
    assert args.verbose == level
    assert args.n_processes == 2
    visual._run_renderer_subprocess(
        [sys.executable, "-u", "-c", "import sys; print('frame progress', file=sys.stderr)"],
        verbose=level,
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert ("frame progress" in output.err) == (level > 0)


def test_silent_renderer_failure_retains_diagnostics(capsys):
    import subprocess

    with pytest.raises(subprocess.CalledProcessError) as error:
        visual._run_renderer_subprocess(
            [sys.executable, "-c", "import sys; print('encoder failure'); sys.exit(7)"],
            verbose=0,
        )
    assert error.value.returncode == 7
    assert "encoder failure" in error.value.output
    assert capsys.readouterr() == ("", "")
