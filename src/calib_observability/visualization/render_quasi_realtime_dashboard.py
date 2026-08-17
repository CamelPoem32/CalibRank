#!/usr/bin/env python3
"""Render a quasi-realtime rover dashboard MP4 from a pickled payload."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    """Load a dashboard payload and render the MP4 in this process.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, help="Pickle payload path.")
    parser.add_argument("--output-mp4", required=True, help="Destination MP4 path.")
    args = parser.parse_args(argv)

    src_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(src_root))

    from calib_observability.visualization.quasi_realtime_rover import (
        save_quasi_realtime_rover_animation,
    )

    with Path(args.payload).expanduser().open("rb") as handle:
        payload = pickle.load(handle)

    output_mp4 = Path(args.output_mp4).expanduser().resolve()
    temporary_html = output_mp4.with_suffix(output_mp4.suffix + ".unused.html")
    render_kwargs = dict(payload["render_kwargs"])

    save_quasi_realtime_rover_animation(
        payload["dataset"],
        payload["snapshots"],
        temporary_html,
        output_mp4=output_mp4,
        save_html=False,
        **render_kwargs,
    )

    print(f"Saved MP4 in subprocess: {output_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
