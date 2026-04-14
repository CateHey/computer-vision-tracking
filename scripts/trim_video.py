#!/usr/bin/env python3
"""
Trim a video to a specified duration using OpenCV.

Usage:
    python scripts/trim_video.py data/raw/original.mp4 --duration 300
    python scripts/trim_video.py data/raw/original.mp4 --duration 300 --output data/clips/original_5min.mp4
    python scripts/trim_video.py data/raw/original.mp4 --start 60 --duration 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def trim_video(
    input_path: str,
    duration_sec: float,
    output_path: str | None = None,
    start_sec: float = 0.0,
) -> Path:
    """Trim video to a given duration.

    Args:
        input_path: Path to source video.
        duration_sec: Duration in seconds to keep.
        output_path: Output path. If None, auto-generates as <name>_<dur>s.<ext>.
        start_sec: Start time in seconds (default: 0).

    Returns:
        Path to the trimmed video.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_sec = total_frames / fps if fps > 0 else 0

    print(f"Source: {src}")
    print(f"  {width}x{height} @ {fps:.1f} FPS, {total_frames} frames ({total_sec:.1f}s)")

    # Determine output path
    if output_path is None:
        dur_label = f"{int(duration_sec)}s"
        out = src.parent / f"{src.stem}_{dur_label}{src.suffix}"
    else:
        out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Seek to start frame
    start_frame = int(start_sec * fps)
    max_frames = int(duration_sec * fps)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"  Start: frame {start_frame} ({start_sec:.1f}s)")

    # Use same codec as source where possible, fallback to XVID
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path_str = str(out.with_suffix(".avi")) if out.suffix == ".mp4" else str(out)
    if out.suffix == ".mp4":
        out = out.with_suffix(".avi")

    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        count += 1
        if count % 1000 == 0:
            print(f"  {count}/{max_frames} frames ({count/fps:.1f}s)")

    cap.release()
    writer.release()

    print(f"Output: {out}")
    print(f"  {count} frames ({count/fps:.1f}s)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim a video to a specified duration.")
    parser.add_argument("input", help="Path to input video.")
    parser.add_argument(
        "--duration", type=float, default=300,
        help="Duration in seconds (default: 300 = 5 min).",
    )
    parser.add_argument(
        "--start", type=float, default=0,
        help="Start time in seconds (default: 0).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: auto-generated).",
    )
    args = parser.parse_args()
    trim_video(args.input, args.duration, args.output, args.start)


if __name__ == "__main__":
    main()
