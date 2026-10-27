"""
Video frame access for the SAM3 / Cutie composite pipelines (torch-free).

  - extract_frames_to_memory: load a frame range into a list (RGB).
  - video_frame_count:        planned number of frames of a run (no decoding).
  - iter_frames_from_video:   stream the same frames one at a time (RGB), so a
                              long video never has to fit in RAM.

All three apply the same start / end / max_frames bounds: ``max_frames`` is
relative to ``start_frame`` and ``end_frame`` is exclusive.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


def _planned_end(
    total: int,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: Optional[int],
) -> int:
    """Exclusive end frame of a run (shared bounds logic)."""
    if end_frame is None:
        end_frame = total
    if max_frames is not None:
        end_frame = min(end_frame, start_frame + max_frames)
    return end_frame


def extract_frames_to_memory(
    video_path: str,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end_frame = _planned_end(total, start_frame, end_frame, max_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    idx = start_frame
    while idx < end_frame:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        idx += 1

    cap.release()
    return frames, fps, width, height


def video_frame_count(
    video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> int:
    """Number of frames extract_frames_to_memory would load, from the container metadata.

    CAP_PROP_FRAME_COUNT is unreliable for some codecs, so this is a plan: the real
    number of frames read can be smaller (the stream ends early).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    end_frame = _planned_end(total, start_frame, end_frame, max_frames)
    return max(0, end_frame - start_frame)


def iter_frames_from_video(
    video_path: str,
    start_frame: int,
    end_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, frame_rgb) for the frames extract_frames_to_memory would load.

    frame_idx is the index in the video (it starts at start_frame). Seeks once to
    start_frame, then reads sequentially. Stops at end_frame / start_frame + max_frames,
    or as soon as the stream stops delivering frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = _planned_end(total, start_frame, end_frame, max_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        idx = start_frame
        while idx < end_frame:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            yield idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()
