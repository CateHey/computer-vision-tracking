"""Geometric utilities for tracking computations."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def euclidean_distance(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
) -> float:
    """Euclidean distance between two 2D points.

    Returns float('inf') if either point is None.
    """
    if a is None or b is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def match_dets_to_slots(
    detections: list,
    slot_centroids: list,
    max_animals: int,
    max_distance: float = 200.0,
) -> List[Optional[object]]:
    """Match YOLO detections to identity slots by centroid proximity.

    Returns a list of length *max_animals* where each element is the best
    matching Detection (or ``None``) for that slot.

    Each detection must expose a ``.center()`` method returning ``(x, y)``.
    """
    result: List[Optional[object]] = [None] * max_animals
    if not detections:
        return result

    det_centers = [d.center() for d in detections]
    used: set = set()

    for slot_idx in range(max_animals):
        sc = slot_centroids[slot_idx]
        if sc is None:
            continue

        best_di = None
        best_dist = float("inf")
        for di, dc in enumerate(det_centers):
            if di in used:
                continue
            dist = ((sc[0] - dc[0]) ** 2 + (sc[1] - dc[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_di = di

        if best_di is not None and best_dist < max_distance:
            result[slot_idx] = detections[best_di]
            used.add(best_di)

    return result


def resolve_overlaps(
    slot_masks: List[Optional[np.ndarray]],
    slot_centroids: List[Optional[Tuple[float, float]]],
) -> None:
    """Resolve overlapping pixels between slot masks by centroid proximity.

    For each pair of active masks, contested pixels (where both masks are True)
    are assigned to the mask whose centroid is closer. Modifies masks in-place.

    Args:
        slot_masks: List of boolean masks per slot (None for inactive).
        slot_centroids: List of (x, y) centroids per slot (None for inactive).
    """
    active = [
        (i, slot_masks[i], slot_centroids[i])
        for i in range(len(slot_masks))
        if slot_masks[i] is not None and slot_centroids[i] is not None
    ]

    for a_idx in range(len(active)):
        i, mi, ci = active[a_idx]
        for b_idx in range(a_idx + 1, len(active)):
            j, mj, cj = active[b_idx]

            overlap = mi & mj
            if not overlap.any():
                continue

            n_overlap = int(overlap.sum())
            logger.debug(
                "Resolving %d overlapping pixels between slot %d and %d", n_overlap, i, j,
            )

            ys, xs = np.where(overlap)
            di = (xs - ci[0]) ** 2 + (ys - ci[1]) ** 2
            dj = (xs - cj[0]) ** 2 + (ys - cj[1]) ** 2

            closer_to_j = dj < di
            mi[ys[closer_to_j], xs[closer_to_j]] = False

            closer_to_i = ~closer_to_j
            mj[ys[closer_to_i], xs[closer_to_i]] = False
