"""Check ordered, closure-safe process-pool analysis and window progress."""

from dataclasses import replace
import os

import numpy as np
import pytest

from src.calib_observability.backend import estimate_poses_dummy
from src.calib_observability.simulation import PlanarRoverConfig, simulate_planar_rover
from src.calib_observability.visualization import quasi_realtime_rover as visual


def test_parallel_simulation_matches_serial_with_complete_snapshots():
    dataset = simulate_planar_rover(PlanarRoverConfig(
        rectangle_width=1.0, rectangle_height=1.0, straight_speed=1.0,
        turn_duration=0.2, total_laps=1, imu_rate_hz=20.0,
        lidar_rate_hz=5.0, random_seed=44,
    ), mode="one_rectangle")
    provider = estimate_poses_dummy(dataset)
    config = visual.QuasiRealtimeConfig(
        window_length=1.4, frame_step=0.7, max_analysis_windows=4,
    )
    serial = visual.compute_quasi_realtime_snapshots(dataset, provider, config)
    parallel = visual.compute_quasi_realtime_snapshots(
        dataset, provider, replace(config, n_processes=2),
    )
    assert any(snapshot.is_valid for snapshot in parallel)
    for expected, actual in zip(serial, parallel, strict=True):
        assert actual.current_time == expected.current_time
        assert actual.is_valid == expected.is_valid
        assert actual.counts == expected.counts
        np.testing.assert_allclose(actual.J_C_display, expected.J_C_display, atol=1e-10)
        assert actual.effective_ranks == expected.effective_ranks or not actual.is_valid
        if actual.is_valid:
            assert actual.bundle is not None
            assert actual.target_results.keys() == expected.target_results.keys()


def test_completed_window_progress_and_order(monkeypatch):
    # A local class exercises cloudpickle rather than relying on fork inheritance.
    class DelayedDataset:
        start_time = 0.0
        end_time = 6.0

        def window_jacobians(self, start, end, provider, **kwargs):
            import os
            import time
            time.sleep(0.8 if end == 1 else 0.05)
            raise ValueError(f"worker-pid={os.getpid()}")

    updates = []
    class Progress:
        def __init__(self, **kwargs):
            assert kwargs["total"] == 7
            assert kwargs["unit"] == "window"
            assert kwargs["disable"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, count):
            updates.append(count)

    monkeypatch.setattr(visual, "tqdm", Progress)
    results = visual.compute_quasi_realtime_snapshots(
        DelayedDataset(), None,
        visual.QuasiRealtimeConfig(frame_step=1.0, n_processes=2), verbose=1,
    )
    assert [result.current_time for result in results] == list(range(7))
    assert updates == [1] * 7
    pids = {int(result.status.split("worker-pid=")[1]) for result in results[1:]}
    assert len(pids) == 2 and os.getpid() not in pids


def test_worker_exception_propagates():
    class BrokenDataset:
        start_time = 0.0
        end_time = 1.0

        def window_jacobians(self, *args, **kwargs):
            raise RuntimeError("intentional worker failure")

    with pytest.raises(RuntimeError, match="intentional worker failure"):
        visual.compute_quasi_realtime_snapshots(
            BrokenDataset(), None, visual.QuasiRealtimeConfig(n_processes=2),
        )


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_worker_count(count):
    with pytest.raises(ValueError, match="n_processes"):
        visual.compute_quasi_realtime_snapshots(
            None, None, visual.QuasiRealtimeConfig(n_processes=count),
        )

