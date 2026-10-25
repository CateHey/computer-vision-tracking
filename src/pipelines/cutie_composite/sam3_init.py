"""SAM3 initialisation policy for cutie_composite (pure logic, no torch): retry the SAM3
phase with lower score thresholds and later start frames until every animal is found.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def select_sam3_init(frames, run_fn, num_slots, base_threshold, retry_thresholds, retry_start_frames,
                     init_points=None):
    """Run the SAM3 phase until some start frame / score threshold shows every animal.

    SAM3's text-prompt detection is not deterministic and merges touching animals, so the
    same clip can hand 6/6 or 5/6 animals to CUTIE (seen on Bunya, 2026-10-23 vs 10-25).
    Attempts, in order: frame 0 with base_threshold, then each retry threshold, then each
    later start frame with every threshold (frames before it get no masks). The first
    attempt whose first frames contain num_slots objects wins; otherwise the attempt with
    the most objects is kept. Results are re-keyed to the original frame indices.

    run_fn(frames, threshold, init_points) -> {local_frame_idx: {"masks": [...], ...}}.
    Returns {"results", "count", "start_frame", "threshold", "attempts"}.
    """
    thresholds = [float(base_threshold)] + [float(t) for t in retry_thresholds if float(t) != float(base_threshold)]
    starts = [0] + [int(s) for s in retry_start_frames if 0 < int(s) < len(frames) - 1]
    best = None
    attempts = 0
    for start in starts:
        for thr in thresholds:
            res = run_fn(frames[start:], thr, init_points if start == 0 else None)
            attempts += 1
            res = {k + start: v for k, v in res.items()}
            first_keys = sorted(res)[:5]
            count = max((len(res[k]["masks"]) for k in first_keys), default=0)
            logger.info("SAM3 init attempt %d: start frame %d, score threshold %.2f -> %d/%d animals",
                        attempts, start, thr, count, num_slots)
            if best is None or count > best["count"]:
                best = {"results": res, "count": count, "start_frame": start, "threshold": thr}
            if count >= num_slots:
                best["attempts"] = attempts
                return best
    best["attempts"] = attempts
    return best
