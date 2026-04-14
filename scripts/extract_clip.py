#!/usr/bin/env python3
"""
Extract a short clip from a raw video file.

Uses ffmpeg (must be installed) or falls back to OpenCV if ffmpeg is unavailable.

Usage:
    python scripts/extract_clip.py \
        --input data/raw/original.mp4 \
        --output data/clips/test-5s.mp4 \
        --start 0 --duration 5

    # Extract 10 seconds starting at 30s:
    python scripts/extract_clip.py \
        --input data/raw/original.mp4 \
        --output data/clips/interaction-10s.mp4 \
        --start 30 --duration 10
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.config_loader import setup_logging

logger = logging.getLogger(__name__)


def extract_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
) -> None:
    """Extract a clip using ffmpeg (fast, codec-copy when possible)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(input_path),
        "-t", str(duration_sec),
        "-c", "copy",
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg stderr: %s", result.stderr)
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}")
    logger.info("Clip saved: %s", output_path)


def extract_with_opencv(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
) -> None:
    """Extract a clip using OpenCV (slower, re-encodes)."""
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Clip saved: %s (%d frames)", output_path, frame_idx - start_frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a clip from a video.")
    parser.add_argument("--input", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", type=Path, required=True, help="Output clip path.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds.")
    args = parser.parse_args()

    setup_logging()

    if not args.input.exists():
        logger.error("Input video not found: %s", args.input)
        sys.exit(1)

    if shutil.which("ffmpeg"):
        extract_with_ffmpeg(args.input, args.output, args.start, args.duration)
    else:
        logger.warning("ffmpeg not found, falling back to OpenCV (slower)")
        extract_with_opencv(args.input, args.output, args.start, args.duration)


if __name__ == "__main__":
    main()
