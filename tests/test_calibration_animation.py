import subprocess
import sys

import numpy as np

from src.calib_observability.visualization.calibration_animation import save_calibration_animation_payload


def _poses(points):
    poses = np.repeat(np.eye(4)[None, ...], len(points), axis=0)
    poses[:, :3, 3] = np.asarray(points, dtype=float)
    return poses


def test_child_process_html_animation_references_external_frames(tmp_path):
    timestamps = np.array([0.0, 1.0, 2.0])
    payload = save_calibration_animation_payload(
        tmp_path / "payload.npz",
        reference_timestamps_s=timestamps,
        reference_poses_se3=_poses([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        baseline_timestamps_s=timestamps,
        baseline_poses_se3=_poses([[0, 0, 0], [1, 0.1, 0], [2, 0.1, 0]]),
        injected_timestamps_s=timestamps,
        injected_poses_se3=_poses([[0, 0, 0], [1, -0.1, 0], [2, -0.1, 0]]),
        baseline_tau_I_s=np.array([0.0, 0.0, 0.0]),
        injected_tau_I_s=np.array([0.01, 0.01, 0.01]),
        calibration_timestamps_s=timestamps,
        target_tau_I_delta_s=0.01,
    )
    output = tmp_path / "animation.html"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/render_calibration_animation.py",
            "--payload",
            str(payload),
            "--output",
            str(output),
            "--backend",
            "html",
            "--max-rendered-frames",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "backend=html" in completed.stdout
    html = output.read_text(encoding="utf-8")
    assert "frame_00000.png" in html
    assert "data:image" not in html
    assert (tmp_path / "animation_frames" / "frame_00002.png").exists()
