"""SAM3 initialisation retries (pure logic, SAM3 replaced by a fake)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.cutie_composite.sam3_init import select_sam3_init  # noqa: E402

FRAMES = list(range(125))  # stand-ins for frames


def _fake(counts):
    """counts[(start_frame, threshold)] -> objects found; returns a run_fn and a call log."""
    calls = []

    def run_fn(frames, thr, pts):
        start = 125 - len(frames)
        calls.append((start, thr, pts is not None))
        n = counts.get((start, thr), 0)
        return {k: {"masks": [object()] * n} for k in range(len(frames))}
    return run_fn, calls


def test_first_attempt_wins_when_all_animals_are_found():
    run_fn, calls = _fake({(0, 0.5): 6})
    r = select_sam3_init(FRAMES, run_fn, 6, 0.5, [0.4, 0.3], [30, 60], init_points=["pts"])
    assert (r["count"], r["start_frame"], r["threshold"], r["attempts"]) == (6, 0, 0.5, 1)
    assert calls == [(0, 0.5, True)]
    assert sorted(r["results"])[:3] == [0, 1, 2]


def test_retries_thresholds_then_later_start_frames():
    run_fn, calls = _fake({(0, 0.5): 5, (0, 0.4): 5, (0, 0.3): 5, (30, 0.5): 6})
    r = select_sam3_init(FRAMES, run_fn, 6, 0.5, [0.4, 0.3], [30, 60], init_points=["pts"])
    assert (r["count"], r["start_frame"], r["threshold"], r["attempts"]) == (6, 30, 0.5, 4)
    assert calls[-1] == (30, 0.5, False)          # init points only apply to frame 0
    assert min(r["results"]) == 30                 # frames before the start get no masks


def test_keeps_the_best_attempt_when_none_is_complete():
    run_fn, calls = _fake({(0, 0.5): 4, (0, 0.4): 5, (0, 0.3): 5, (30, 0.5): 4, (30, 0.4): 4, (30, 0.3): 4})
    r = select_sam3_init(FRAMES, run_fn, 6, 0.5, [0.4, 0.3], [30], init_points=None)
    assert (r["count"], r["start_frame"], r["threshold"], r["attempts"]) == (5, 0, 0.4, 6)
