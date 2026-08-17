#!/usr/bin/env python3
"""Render calibration-injection animations in a separate Python process."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and render an animation.

    Args:
        argv: Optional command-line argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, help="Input .npz animation payload.")
    parser.add_argument("--output", required=True, help="Output .mp4 or .html path.")
    parser.add_argument("--backend", choices=("mp4", "html"), default="mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--every-nth-frame", type=int, default=1)
    parser.add_argument("--max-rendered-frames", type=int, default=None)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--no-fallback-html", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.calib_observability.visualization.calibration_animation import render_calibration_animation

    result = render_calibration_animation(
        args.payload,
        args.output,
        backend=args.backend,
        fps=args.fps,
        dpi=args.dpi,
        every_nth_frame=args.every_nth_frame,
        max_rendered_frames=args.max_rendered_frames,
        codec=args.codec,
        fallback_html=not args.no_fallback_html,
    )
    print(
        f"backend={result['backend']} output_path={result['output_path']} "
        f"frame_count={result['frame_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
