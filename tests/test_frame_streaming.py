"""Streaming frames (iter_frames_from_video) must match the preloaded frames exactly."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.sam3_composite.frames import (  # noqa: E402
    extract_frames_to_memory,
    iter_frames_from_video,
    video_frame_count,
)

NUM_FRAMES = 40
SIZE = 64


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    """40-frame 64x64 MJPG video with distinct, structured content in every frame."""
    path = tmp_path_factory.mktemp("video") / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (SIZE, SIZE))
    assert writer.isOpened()
    for i in range(NUM_FRAMES):
        base = (i * 6) % 256                                               # per-frame brightness
        frame = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
        frame[:, :, 0] = (base + 40) % 256                                 # colour channels differ
        frame[:, :, 1] = base
        frame[:, :, 2] = (base + 120) % 256
        x = 4 + i                                                          # moving bright square
        frame[10:20, x:x + 10] = (255, 255, 255)
        writer.write(frame)
    writer.release()
    return str(path)


CASES = [
    pytest.param(0, None, None, id="whole_video"),
    pytest.param(5, 20, None, id="start5_end20"),
    pytest.param(3, None, 10, id="start3_max10"),
    pytest.param(5, 20, 4, id="start5_end20_max4"),
    pytest.param(10, None, 1000, id="max_larger_than_video"),
]


@pytest.mark.parametrize("start_frame,end_frame,max_frames", CASES)
def test_stream_matches_preloaded_frames(video_path, start_frame, end_frame, max_frames):
    frames, _fps, _w, _h = extract_frames_to_memory(
        video_path, start_frame, end_frame, max_frames=max_frames)
    streamed = list(iter_frames_from_video(
        video_path, start_frame, end_frame=end_frame, max_frames=max_frames))

    assert len(frames) > 0
    assert len(streamed) == len(frames)
    for offset, (frame_idx, frame) in enumerate(streamed):
        assert frame_idx == start_frame + offset
        assert np.array_equal(frame, frames[offset])

    count = video_frame_count(
        video_path, start_frame=start_frame, end_frame=end_frame, max_frames=max_frames)
    assert count == len(frames)


def test_frames_are_distinct(video_path):
    """Guards the test itself: identical frames would make the comparison meaningless."""
    frames, _fps, width, height = extract_frames_to_memory(video_path, 0, None)
    assert (width, height) == (SIZE, SIZE)
    assert len(frames) == NUM_FRAMES
    for a, b in zip(frames[:-1], frames[1:]):
        assert not np.array_equal(a, b)


def test_split_handoff_equals_single_pass(video_path):
    """Phase 1 loaded + phase 2 streamed reproduces the whole run, as in cutie_composite."""
    start_frame, phase1 = 4, 7
    whole, _fps, _w, _h = extract_frames_to_memory(video_path, start_frame, None)
    head, _fps, _w, _h = extract_frames_to_memory(video_path, start_frame, None, max_frames=phase1)
    tail = list(iter_frames_from_video(
        video_path, start_frame + phase1, end_frame=start_frame + len(whole)))

    assert len(head) == phase1
    assert len(head) + len(tail) == len(whole)
    for gidx, (vidx, frame) in enumerate(tail, start=phase1):
        assert vidx - start_frame == gidx
        assert np.array_equal(frame, whole[gidx])


def test_stream_past_end_of_video_stops_cleanly(video_path):
    assert list(iter_frames_from_video(video_path, NUM_FRAMES + 5)) == []
    assert video_frame_count(video_path, start_frame=NUM_FRAMES + 5) == 0


def test_unreadable_video_raises(tmp_path):
    missing = str(tmp_path / "missing.avi")
    with pytest.raises(RuntimeError):
        video_frame_count(missing)
    with pytest.raises(RuntimeError):
        next(iter_frames_from_video(missing, 0))
