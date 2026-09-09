#!/usr/bin/env python3
"""Run KAIST observability with a bounded MP4 dashboard.

Example:
    python src/kaist_dataset/run_kaist_observability.py /path/to/Urban16 --lidar left

Relative --output-dir and --processed-data-dir paths are relative to dataset_root.
Caps sample the whole selected overlap. Increasing them increases analysis cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kaist_dataset.observability import KaistObservabilityConfig, run_observability


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the KAIST observability command line.

    Returns:
        ArgumentParser with dataset, sampling, analysis and MP4 options.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="One KAIST Urban dataset directory.")
    parser.add_argument("--lidar", choices=("left", "right"), default="left")
    parser.add_argument("--processed-data-dir", type=Path, help="Precomputed map-pose CSV directory.")
    parser.add_argument("--output-dir", type=Path, help="Default: dataset_root/outputs/calib_observability/<side>.")
    parser.add_argument("--start-time", type=float, default=0.0, help="Seconds from common overlap start.")
    parser.add_argument("--end-time", type=float, help="Seconds from common overlap start.")
    parser.add_argument("--use-sparse", action=argparse.BooleanOptionalAction, default=False)
    defaults = KaistObservabilityConfig(Path("."))
    parser.add_argument("--verbose", "--verbosity", "-v", type=int, choices=(0, 1, 2),
                        default=defaults.verbose,
                        help=f"0: silent, 1: status and progress, 2: also JSON metadata. Default: {defaults.verbose}.")
    settings = {
        "n-processes": (int, "Process-pool workers for analysis windows; 1 runs sequentially."),
        "imu-frequency-hz": (float, "Target IMU rate before applying the sample cap [Hz]."),
        "window-size": (float, "Analysis window duration [s]."),
        "step-size": (float, "Requested spacing before the analysis-window cap [s]."),
        "max-imu-samples": (int, "Maximum paired IMU samples across the selected interval."),
        "max-lidar-poses": (int, "Maximum absolute poses, sampled before forming odometry."),
        "max-ground-truth-poses": (int, "Maximum reference trajectory support poses."),
        "max-analysis-windows": (int, "Maximum windows computed across the full interval."),
        "max-rendered-frames": (int, "Maximum frames serialized and rendered."),
        "trajectory-samples": (int, "Samples used to draw each full trajectory."),
        "mp4-fps": (float, "MP4 playback frame rate."),
        "mp4-dpi": (int, "MP4 resolution for the 17 x 10 inch dashboard."),
        "gyro-noise-std": (float, "Gyroscope whitening standard deviation [rad/s]."),
        "accel-noise-std": (float, "Accelerometer whitening standard deviation [m/s^2]."),
        "lidar-rotation-noise-std": (float, "Relative LiDAR rotation standard deviation [rad]."),
        "lidar-translation-noise-std": (float, "Relative LiDAR translation standard deviation [m]."),
        "simple-accel-noise-std": (float, "Simple-accelerometer factor standard deviation [m/s^2]."),
        "gravity-z": (float, "World gravity Z component [m/s^2]."),
    }
    # Keep configuration defaults shared between CLI and Python callers.
    for option, (value_type, help_text) in settings.items():
        default = getattr(defaults, option.replace("-", "_"))
        parser.add_argument("--" + option, type=value_type, default=default,
                            help=f"{help_text} Default: {default}.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse options and export KAIST observability results.

    Args:
        argv: Optional argument sequence; None reads the process command line.

    Returns:
        Zero after a successful run.
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = KaistObservabilityConfig(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    run_observability(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python src/kaist_dataset/run_kaist_observability.py /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 --lidar left --n-processes 8
