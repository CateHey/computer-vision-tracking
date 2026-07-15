"""
Safety systems for the Cutie composite pipeline.

Cutie tracks continuously, but can drift or lose rats over long videos. These
safety nets detect failures and trigger SAM3 re-segmentation only when needed.

Three systems:

  1. PROTECTION (every frame) — Cutie returned fewer than N masks (a rat is lost)
  2. FUSION (every frame)    — a mask vanished AND another grew ~its area
                               (two rats merged into one)
  3. CHECKPOINT (every K frames) — compare Cutie vs SAM3 image (count + IoU)

Decision hierarchy (checkpoint):
  - Cutie==N, SAM3==N, IoU>thr        -> all good, Cutie continues
  - Cutie==N, SAM3<N                  -> trust Cutie (SAM3 likely fused close rats)
  - Cutie<N                           -> SAM3 wins, recover the missing rat
  - Cutie==N, SAM3==N, IoU<thr        -> drift/swap, SAM3 wins

Recovery: run SAM3 image up to `max_search_frames` (default 30) until N rats are
found; identify the missing one by comparing centroids against the PREVIOUS good
frame; reassign identity; re-inject to Cutie (without reset).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# FAILURE REPORT
# ============================================================================

@dataclass
class SafetyEvent:
    """Describes a safety-system trigger."""
    triggered: bool
    system: str                     # "protection" | "fusion" | "checkpoint" | "none"
    reason: str = ""
    missing_slots: List[int] = field(default_factory=list)
    detail: str = ""


# ============================================================================
# SYSTEM 1 — PROTECTION (missing masks)
# ============================================================================

def check_protection(
    slot_masks: List[Optional[np.ndarray]],
    num_slots: int,
    min_area: int = 50,
) -> SafetyEvent:
    """Detect if Cutie lost one or more rats (fewer than N valid masks).

    Returns SafetyEvent with the list of missing slots.
    """
    missing = []
    for s in range(num_slots):
        m = slot_masks[s] if s < len(slot_masks) else None
        if m is None or int(m.sum()) < min_area:
            missing.append(s)

    if missing:
        return SafetyEvent(
            triggered=True,
            system="protection",
            reason="missing_masks",
            missing_slots=missing,
            detail=f"{len(missing)} slot(s) missing: {missing}",
        )
    return SafetyEvent(triggered=False, system="none")


# ============================================================================
# SYSTEM 2 — FUSION (two rats merged)
# ============================================================================

def check_fusion(
    prev_masks: List[Optional[np.ndarray]],
    curr_masks: List[Optional[np.ndarray]],
    num_slots: int,
    area_match_tolerance: float = 0.15,
    min_growth_ratio: float = 1.5,
    min_area: int = 50,
) -> SafetyEvent:
    """Detect a fusion: one mask vanished AND another grew by ~its area.

    Three conditions (all required, to avoid false positives):
      1. A slot that had a mask before is now missing (vanished)
      2. Another slot grew significantly (>= min_growth_ratio)
      3. The grown area ≈ vanished area (within area_match_tolerance)

    Returns SafetyEvent listing the vanished slot(s).
    """
    # Find vanished slots (had mask, now gone)
    vanished = []
    for s in range(num_slots):
        prev = prev_masks[s] if s < len(prev_masks) else None
        curr = curr_masks[s] if s < len(curr_masks) else None
        prev_ok = prev is not None and int(prev.sum()) >= min_area
        curr_ok = curr is not None and int(curr.sum()) >= min_area
        if prev_ok and not curr_ok:
            vanished.append((s, int(prev.sum())))

    if not vanished:
        return SafetyEvent(triggered=False, system="none")

    # Find grown slots (grew >= min_growth_ratio)
    grown = []
    for s in range(num_slots):
        prev = prev_masks[s] if s < len(prev_masks) else None
        curr = curr_masks[s] if s < len(curr_masks) else None
        if prev is None or curr is None:
            continue
        prev_area = int(prev.sum())
        curr_area = int(curr.sum())
        if prev_area < min_area:
            continue
        if curr_area >= prev_area * min_growth_ratio:
            growth = curr_area - prev_area   # how much it grew
            grown.append((s, growth))

    if not grown:
        return SafetyEvent(triggered=False, system="none")

    # Check if any vanished area ≈ any growth
    for van_slot, van_area in vanished:
        for grown_slot, growth in grown:
            ratio = growth / max(van_area, 1)
            if abs(ratio - 1.0) <= area_match_tolerance:
                return SafetyEvent(
                    triggered=True,
                    system="fusion",
                    reason="fusion_detected",
                    missing_slots=[van_slot],
                    detail=(f"slot {van_slot} (area {van_area}) fused into "
                            f"slot {grown_slot} (grew {growth})"),
                )

    return SafetyEvent(triggered=False, system="none")


# ============================================================================
# SYSTEM 3 — CHECKPOINT (periodic Cutie vs SAM3 comparison)
# ============================================================================

def build_union_mask(masks: List[Optional[np.ndarray]], shape: Tuple[int, int]) -> np.ndarray:
    """Union of all masks into one binary image."""
    union = np.zeros(shape, dtype=bool)
    for m in masks:
        if m is not None:
            union = union | m
    return union


def compute_iou(
    masks_a: List[Optional[np.ndarray]],
    masks_b: List[np.ndarray],
    shape: Tuple[int, int],
) -> float:
    """Global IoU over the union of masks (ignores background)."""
    union_a = build_union_mask(masks_a, shape)
    union_b = build_union_mask(list(masks_b), shape)
    inter = np.logical_and(union_a, union_b).sum()
    uni = np.logical_or(union_a, union_b).sum()
    if uni == 0:
        return 1.0
    return float(inter) / float(uni)


def checkpoint_decision(
    cutie_masks: List[Optional[np.ndarray]],
    sam3_masks: np.ndarray,
    num_slots: int,
    shape: Tuple[int, int],
    iou_threshold: float = 0.85,
    min_area: int = 50,
) -> SafetyEvent:
    """Apply the decision hierarchy comparing Cutie vs SAM3 image.

    Returns SafetyEvent: triggered=True means "SAM3 wins, re-inject".
    """
    n_cutie = sum(1 for m in cutie_masks if m is not None and int(m.sum()) >= min_area)
    n_sam3 = len(sam3_masks)

    # Case 3: Cutie lost rats -> SAM3 wins
    if n_cutie < num_slots:
        return SafetyEvent(
            triggered=True,
            system="checkpoint",
            reason="cutie_lost_rats",
            detail=f"Cutie has {n_cutie}/{num_slots}, SAM3 has {n_sam3}",
        )

    # Case 2: Cutie has all, SAM3 has fewer -> trust Cutie
    if n_cutie == num_slots and n_sam3 < num_slots:
        return SafetyEvent(
            triggered=False,
            system="checkpoint",
            reason="trust_cutie",
            detail=f"Cutie {n_cutie} OK, SAM3 only {n_sam3} (likely fused close rats)",
        )

    # Case 1 & 4: both have N -> check IoU
    if n_cutie == num_slots and n_sam3 >= num_slots:
        iou = compute_iou(cutie_masks, sam3_masks, shape)
        if iou < iou_threshold:
            return SafetyEvent(
                triggered=True,
                system="checkpoint",
                reason="low_iou_drift",
                detail=f"IoU={iou:.2f} < {iou_threshold} (drift/swap)",
            )
        return SafetyEvent(
            triggered=False,
            system="checkpoint",
            reason="all_good",
            detail=f"IoU={iou:.2f}",
        )

    return SafetyEvent(triggered=False, system="none")


# ============================================================================
# RECOVERY — SAM3 search + identity reassignment
# ============================================================================

def reassign_by_previous_frame(
    sam3_masks: np.ndarray,
    prev_centroids: List[Optional[Tuple[float, float]]],
    num_slots: int,
    mask_centroid_fn,
    assign_fn,
    max_dist: Optional[float] = None,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[Tuple[float, float]]]]:
    """Assign SAM3 masks to slots by matching centroids against the previous
    good frame (where identities were correct).

    Args:
        sam3_masks: (M, H, W) fresh masks from SAM3 image
        prev_centroids: last-known-good centroid per slot
        num_slots: N
        mask_centroid_fn: function mask -> (x, y) or None
        assign_fn: assign_identities_by_centroid-style function
        max_dist: optional max assignment distance

    Returns: (slot_masks, slot_centroids)
    """
    slot_masks: List[Optional[np.ndarray]] = [None] * num_slots
    slot_centroids: List[Optional[Tuple[float, float]]] = [None] * num_slots

    if len(sam3_masks) == 0:
        return slot_masks, slot_centroids

    sam3_centroids = [mask_centroid_fn(m) for m in sam3_masks]

    assignment = assign_fn(
        list(sam3_masks), sam3_centroids, prev_centroids, num_slots,
        max_dist=max_dist,
    )
    for new_i, slot_idx in enumerate(assignment):
        if slot_idx >= 0:
            slot_masks[slot_idx] = sam3_masks[new_i]
            slot_centroids[slot_idx] = sam3_centroids[new_i]

    return slot_masks, slot_centroids


class RecoveryController:
    """Manages the SAM3 recovery search across frames.

    When a rat is lost, SAM3 image runs on the current frame. If it doesn't find
    all N rats (e.g. rats still overlapping), the search continues on subsequent
    frames up to max_search_frames. During the search, mask carry-over fills gaps.
    """

    def __init__(self, max_search_frames: int = 30):
        self.max_search_frames = max_search_frames
        self.searching = False
        self.search_start_frame = -1
        self.frames_searched = 0

    def start(self, frame_idx: int) -> None:
        self.searching = True
        self.search_start_frame = frame_idx
        self.frames_searched = 0
        logger.info("Recovery search started at frame %d", frame_idx)

    def should_continue(self) -> bool:
        return self.searching and self.frames_searched < self.max_search_frames

    def tick(self) -> None:
        self.frames_searched += 1

    def stop(self, success: bool, frame_idx: int) -> None:
        status = "SUCCESS" if success else "GAVE UP"
        logger.info("Recovery search %s at frame %d (searched %d frames)",
                    status, frame_idx, self.frames_searched)
        self.searching = False
        self.search_start_frame = -1
        self.frames_searched = 0