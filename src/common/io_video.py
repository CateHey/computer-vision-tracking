"""
Video I/O utilities: reading, writing, frame extraction, and metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


def open_video_reader(path: str | Path) -> cv2.VideoCapture:
    """Open a video file for reading.

    Args:
        path: Path to the video file.

    Returns:
        An opened cv2.VideoCapture object.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If OpenCV cannot open the video.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def get_video_properties(cap: cv2.VideoCapture) -> Dict[str, Any]:
    """Extract video properties from an opened VideoCapture.

    Returns:
        Dict with keys: fps, width, height, total_frames.
    """
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def create_video_writer(
    path: str | Path,
    fps: float,
    width: int,
    height: int,
    codec: str = "XVID",
) -> cv2.VideoWriter:
    """Create a video writer.

    The default codec is XVID (.avi) which works reliably across all platforms
    without extra libraries. If you need .mp4 output, install OpenH264 and
    use codec="avc1".

    Args:
        path: Output video file path. Extension should match codec:
            XVID -> .avi, mp4v/avc1 -> .mp4, MJPG -> .avi
        fps: Frames per second.
        width: Frame width.
        height: Frame height.
        codec: FourCC codec string. Recommended: "XVID" (universal),
            "avc1" (H.264, needs OpenH264), "mp4v" (MPEG-4 Part 2).

    Returns:
        An opened cv2.VideoWriter object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.warning("VideoWriter failed to open with codec '%s', falling back to XVID", codec)
        avi_path = path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(avi_path), fourcc, fps, (width, height))
    return writer


def iter_frames(
    cap: cv2.VideoCapture,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Iterator[Tuple[int, Any]]:
    """Iterate over video frames.

    Args:
        cap: An opened cv2.VideoCapture.
        max_frames: Stop after this many frames (from start_frame). None = read all.
        start_frame: First frame index to yield (0-based). Frames before this
            are skipped via seek. Default 0.
        end_frame: Stop before this frame index (exclusive). None = read to end.
            When both max_frames and end_frame are set, the stricter limit wins.

    Yields:
        (frame_index, frame_bgr) tuples. frame_index is the absolute position
        in the video (not relative to start_frame).
    """
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame
    count = 0
    while True:
        if max_frames is not None and count >= max_frames:
            break
        if end_frame is not None and idx >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1
        count += 1
