"""Shared cost computation for mask-to-slot assignment.

Used by both SlotTracker (sam2_yolo pipeline) and IdentityMatcher (reference
pipeline) to compute the soft cost between a current detection and a tracked slot.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .metrics import mask_iou

# Cost for impossible assignments (hard gate for truly pathological cases)
INF_COST = 1e6
# Hard veto only for extreme area changes (5x ratio = 400% change)
HARD_AREA_RATIO_VETO = 5.0


def compute_assignment_cost(
    centroid: Tuple[float, float],
    area: float,
    mask: np.ndarray,
    prev_centroid: Optional[Tuple[float, float]],
    prev_area: Optional[float],
    prev_mask: Optional[np.ndarray],
    max_distance: float,
    w_dist: float = 0.4,
    w_iou: float = 0.4,
    w_area: float = 0.2,
) -> float:
    """Compute soft assignment cost between a detection and a previous slot state.

    Three components:
    - Distance cost: normalized by max_distance, capped at 1.0
    - Mask IoU cost: 1.0 - mask_iou (shape continuity)
    - Area ratio cost: soft penalty for size change, capped at 1.0

    Args:
        centroid: Current detection centroid (x, y). May be a velocity-predicted
            position (caller handles prediction before calling this function).
        area: Current detection mask area.
        mask: Current detection boolean mask.
        prev_centroid: Previous slot centroid (None → INF_COST).
        prev_area: Previous slot mask area (None or 0 → neutral 0.5).
        prev_mask: Previous slot mask (None → neutral 0.5).
        max_distance: Normalization constant for distance cost (pixels).
        w_dist: Weight for distance cost.
        w_iou: Weight for mask IoU cost.
        w_area: Weight for area ratio cost.

    Returns:
        Cost in [0, ~1.0], or INF_COST for impossible assignments.
    """
    if prev_centroid is None:
        return INF_COST

    # --- Distance cost (normalized) ---
    dx = centroid[0] - prev_centroid[0]
    dy = centroid[1] - prev_centroid[1]
    dist = (dx * dx + dy * dy) ** 0.5
    dist_cost = min(dist / max_distance, 1.0)

    # --- Mask IoU cost ---
    if prev_mask is not None:
        iou = mask_iou(mask, prev_mask)
        iou_cost = 1.0 - iou
    else:
        iou_cost = 0.5  # neutral if no previous mask

    # --- Area change cost (soft, not hard veto) ---
    if prev_area is not None and prev_area > 0 and area > 0:
        ratio = max(area, prev_area) / min(area, prev_area)
        if ratio > HARD_AREA_RATIO_VETO:
            return INF_COST  # only veto extreme cases (5x change)
        area_cost = min(ratio - 1.0, 1.0)
    else:
        area_cost = 0.5

    return w_dist * dist_cost + w_iou * iou_cost + w_area * area_cost
