"""Mask deduplication by IoU — shared by all tracking strategies.

Provides a single implementation of the "sort by area, greedily keep
non-overlapping" deduplication logic used by both SlotTracker and
IdentityMatcher.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .metrics import mask_iou


def deduplicate_masks(
    masks: List[np.ndarray],
    max_count: int,
    iou_threshold: float = 0.5,
    scores: Optional[List[float]] = None,
) -> Tuple[List[np.ndarray], Optional[List[float]]]:
    """Remove redundant masks by IoU, keeping the largest first.

    Sorts masks by area (descending), then greedily keeps masks that
    don't overlap with already-kept masks above the IoU threshold.

    If len(masks) <= max_count, returns all masks unmodified.

    Args:
        masks: Boolean masks (H, W).
        max_count: Maximum number of masks to keep.
        iou_threshold: IoU above which a mask is considered a duplicate.
        scores: Optional confidence scores parallel to masks. If provided,
            the corresponding scores for kept masks are also returned.

    Returns:
        (unique_masks, unique_scores) — unique_scores is None if scores
        was not provided.
    """
    if len(masks) <= max_count:
        return masks, scores

    areas = [m.sum() for m in masks]
    sorted_idx = np.argsort(areas)[::-1]

    unique_masks: List[np.ndarray] = []
    unique_scores: Optional[List[float]] = [] if scores is not None else None

    for idx in sorted_idx:
        is_dup = any(mask_iou(masks[idx], um) > iou_threshold for um in unique_masks)
        if not is_dup:
            unique_masks.append(masks[idx])
            if unique_scores is not None and scores is not None:
                unique_scores.append(scores[idx])
            if len(unique_masks) >= max_count:
                break

    return unique_masks, unique_scores
