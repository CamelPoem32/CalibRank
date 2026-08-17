from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from src.calib_observability.visualization.quasi_realtime_rover import _pickleable_trajectory_proxy
from src.calib_observability.types import AccelerometerOptions
from src.calib_observability.workflows import planar_rover_observability as workflow


def test_simple_accelerometer_dashboard_mp4_primary_path_uses_subprocess(monkeypatch, tmp_path):
    calls = {"html": 0, "mp4": 0}
    snapshots = [SimpleNamespace(current_time=0.0)]

    def fake_series(*args, **kwargs):
        return SimpleNamespace(snapshots=snapshots)

    def fake_html(*args, **kwargs):
        calls["html"] += 1
        return Path(args[2])

    def fake_mp4(*args, **kwargs):
        calls["mp4"] += 1
        return Path(args[2])

    monkeypatch.setattr(workflow, "build_observability_visualization_series", fake_series)
    monkeypatch.setattr(workflow, "save_quasi_realtime_rover_animation", fake_html)
    monkeypatch.setattr(workflow, "save_quasi_realtime_rover_animation_mp4_subprocess", fake_mp4)

    _, animation_path = workflow.save_simple_accelerometer_dashboard(
        dataset=object(),
        pose_provider=object(),
        output_html=tmp_path / "dashboard.mp4",
        window_duration=5.0,
        window_step=1.0,
        accelerometer_options=AccelerometerOptions(mode="simple"),
        max_display_rows=2,
        max_display_cols=2,
        trajectory_samples=3,
        save_html=True,
    )

    assert calls == {"html": 0, "mp4": 1}
    assert animation_path == tmp_path / "dashboard.mp4"


def test_simple_accelerometer_dashboard_html_path_keeps_html_and_optional_mp4(monkeypatch, tmp_path):
    calls = {"html": 0, "mp4": 0}
    snapshots = [SimpleNamespace(current_time=0.0), SimpleNamespace(current_time=1.0)]

    def fake_series(*args, **kwargs):
        return SimpleNamespace(snapshots=snapshots)

    def fake_html(*args, **kwargs):
        calls["html"] += 1
        assert kwargs["output_mp4"] is None
        return Path(args[2])

    def fake_mp4(*args, **kwargs):
        calls["mp4"] += 1
        assert args[1] == snapshots
        return Path(args[2])

    monkeypatch.setattr(workflow, "build_observability_visualization_series", fake_series)
    monkeypatch.setattr(workflow, "save_quasi_realtime_rover_animation", fake_html)
    monkeypatch.setattr(workflow, "save_quasi_realtime_rover_animation_mp4_subprocess", fake_mp4)

    _, animation_path = workflow.save_simple_accelerometer_dashboard(
        dataset=object(),
        pose_provider=object(),
        output_html=tmp_path / "dashboard.html",
        output_mp4=tmp_path / "dashboard.mp4",
        window_duration=5.0,
        window_step=1.0,
        accelerometer_options=AccelerometerOptions(mode="simple"),
        max_display_rows=2,
        max_display_cols=2,
        trajectory_samples=3,
        save_html=True,
    )

    assert calls == {"html": 1, "mp4": 1}
    assert animation_path == tmp_path / "dashboard.html"


def test_pickleable_trajectory_proxy_does_not_pickle_local_trajectory_closure():
    def position_function(time):
        return np.array([time, time**2, 0.0], dtype=float)

    class LocalClosureTrajectory:
        def __init__(self):
            self.position_function = position_function

        def sample(self, number_samples):
            times = np.linspace(0.0, 1.0, int(number_samples))
            positions = np.vstack([self.position_function(time) for time in times])
            yaws = np.zeros_like(times)
            return times, positions, yaws

        def position_at(self, time):
            return self.position_function(float(time))

        def yaw_at(self, time):
            return 0.0

    proxy = _pickleable_trajectory_proxy(
        LocalClosureTrajectory(),
        trajectory_samples=5,
        frame_times=np.array([0.25, 0.75]),
    )

    payload = SimpleNamespace(trajectory=proxy)
    round_trip = pickle.loads(pickle.dumps(payload))

    np.testing.assert_allclose(round_trip.trajectory.position_at(0.5), [0.5, 0.25, 0.0])
