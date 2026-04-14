"""
Multi-object tracking with Hungarian assignment and soft costs.

Provides:
- Smart duplicate mask filtering (only when num_masks > max_count).
- SlotTracker: fixed identity slots using YOLO track IDs as primary identity,
  with Hungarian (optimal) assignment for SAM2 masks ↔ YOLO tracks.
  Uses soft cost combining centroid distance + mask IoU + area change.
  No hard area veto. Missing-frame tolerance prevents color flicker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cost import INF_COST, compute_assignment_cost
from .mask_dedup import deduplicate_masks
from .metrics import compute_centroid

logger = logging.getLogger(__name__)


def filter_masks(
    masks: List[np.ndarray],
    iou_threshold: float,
    max_count: int,
) -> List[np.ndarray]:
    """Keep at most `max_count` masks, removing duplicates by IoU.

    If num_masks <= max_count, returns all masks unmodified.
    """
    result, _ = deduplicate_masks(masks, max_count, iou_threshold)
    return result


@dataclass
class SlotState:
    """Persistent state for one tracking slot."""
    centroid: Optional[Tuple[float, float]] = None
    area: int = 0
    mask: Optional[np.ndarray] = None
    missing_count: int = 0
    active: bool = False
    yolo_track_id: Optional[int] = None


class SlotTracker:
    """Fixed-slot tracker using YOLO track IDs + Hungarian assignment.

    Identity flow:
    1. YOLO model.track() provides track IDs via BoT-SORT/ByteTrack.
    2. YOLO track IDs are mapped to fixed color slots (slot 0 = green, etc.).
    3. SAM2 masks are matched to YOLO detections using Hungarian assignment
       with a soft cost matrix combining:
       - Centroid distance (normalized by max_distance)
       - 1 - mask IoU with previous frame
       - Area change ratio (soft penalty, NOT hard veto)
    4. Slots survive missing frames (up to max_missing_frames) to prevent
       color flicker at borders.

    The key insight: YOLO track IDs provide temporal identity, but they can
    still switch. The soft-cost Hungarian matching makes the mask↔track
    association globally optimal, avoiding the cascading errors of greedy
    matching.
    """

    def __init__(
        self,
        num_slots: int,
        max_distance: float = 150.0,
        max_missing_frames: int = 5,
        w_dist: float = 0.4,
        w_iou: float = 0.4,
        w_area: float = 0.2,
        cost_threshold: float = 0.85,
    ):
        """
        Args:
            num_slots: Number of tracking slots (= max_animals).
            max_distance: Normalization constant for distance cost (pixels).
            max_missing_frames: Frames before a slot is released.
            w_dist: Weight for centroid distance in cost matrix.
            w_iou: Weight for (1 - mask_iou) in cost matrix.
            w_area: Weight for area change in cost matrix.
            cost_threshold: Assignments with cost > this are rejected.
        """
        self.num_slots = num_slots
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.w_dist = w_dist
        self.w_iou = w_iou
        self.w_area = w_area
        self.cost_threshold = cost_threshold
        self.slots: List[SlotState] = [SlotState() for _ in range(num_slots)]

        # Maps YOLO track_id → slot index for stable identity
        self._track_to_slot: Dict[int, int] = {}

    def update(
        self,
        masks: List[np.ndarray],
        track_ids: List[Optional[int]],
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[Tuple[float, float]]]]:
        """Match masks to slots using YOLO track IDs + Hungarian assignment.

        Args:
            masks: Filtered SAM2 masks for this frame.
            track_ids: YOLO track ID for each mask (parallel list, may contain None).

        Returns:
            (slot_masks, slot_centroids) — lists of length num_slots.
            None entries for slots without a match this frame.
        """
        # Compute centroids and areas
        curr_centroids = []
        curr_areas = []
        for m in masks:
            c = compute_centroid(m)
            curr_centroids.append(c if c is not None else (0.0, 0.0))
            curr_areas.append(int(m.sum()))

        active_slots = [i for i in range(self.num_slots) if self.slots[i].active]

        # --- First frame: assign in order ---
        if len(active_slots) == 0:
            return self._init_slots(masks, curr_centroids, curr_areas, track_ids)

        # --- No masks: increment missing ---
        if len(masks) == 0:
            self._increment_missing_all()
            return [None] * self.num_slots, [None] * self.num_slots

        # --- Step 1: Try to match by YOLO track ID (primary identity) ---
        matched_masks, matched_slots, unmatched_mask_idx = self._match_by_track_id(
            masks, track_ids, curr_centroids, curr_areas,
        )

        # --- Step 2: Hungarian assignment for remaining masks ↔ unmatched slots ---
        unmatched_active = [i for i in active_slots if i not in matched_slots]

        if unmatched_mask_idx and unmatched_active:
            self._hungarian_assign(
                masks, curr_centroids, curr_areas,
                unmatched_mask_idx, unmatched_active,
                matched_masks, matched_slots, track_ids,
            )

        # --- Step 3: Place still-unmatched masks into free slots ---
        remaining = [mi for mi in unmatched_mask_idx if mi not in matched_masks]
        for mi in remaining:
            free = self._find_free_slot()
            if free is None:
                break
            self._assign_to_slot(free, masks[mi], curr_centroids[mi],
                                 curr_areas[mi], track_ids[mi])
            matched_masks.add(mi)

        # --- Step 4: Increment missing for unmatched active slots ---
        for si in active_slots:
            if si not in matched_slots:
                self.slots[si].missing_count += 1
                if self.slots[si].missing_count > self.max_missing_frames:
                    self._release_slot(si)

        # Build output
        result_masks: List[Optional[np.ndarray]] = [None] * self.num_slots
        result_centroids: List[Optional[Tuple[float, float]]] = [None] * self.num_slots
        for i, s in enumerate(self.slots):
            if s.active and s.missing_count == 0:
                result_masks[i] = s.mask
                result_centroids[i] = s.centroid

        return result_masks, result_centroids

    def get_debug_info(self) -> str:
        """Return a debug string summarizing slot states."""
        parts = []
        for i, s in enumerate(self.slots):
            if s.active:
                parts.append(
                    f"slot{i}:tid={s.yolo_track_id}(miss={s.missing_count})"
                )
            else:
                parts.append(f"slot{i}:free")
        return " | ".join(parts)

    # --- Internal methods ---

    def _init_slots(self, masks, centroids, areas, track_ids):
        """First-frame initialization."""
        result_masks: List[Optional[np.ndarray]] = [None] * self.num_slots
        result_centroids: List[Optional[Tuple[float, float]]] = [None] * self.num_slots
        for mi, m in enumerate(masks):
            if mi >= self.num_slots:
                break
            self._assign_to_slot(mi, m, centroids[mi], areas[mi], track_ids[mi])
            result_masks[mi] = m
            result_centroids[mi] = centroids[mi]
        return result_masks, result_centroids

    def _match_by_track_id(self, masks, track_ids, centroids, areas):
        """Match masks to slots by YOLO track ID, with swap guard.

        When all masks map to known track IDs, validate that the YOLO-proposed
        assignment is spatially consistent. For N=2, compare straight vs swapped
        cost — if swapped is cheaper, YOLO has likely swapped IDs during a
        crossing, so reject all track_id matches and let Hungarian resolve it.
        """
        # Collect masks that have a known track_id → slot mapping
        tid_matches = []  # list of (mask_idx, slot_idx, track_id)
        no_tid = []       # mask indices with no known track_id

        for mi in range(len(masks)):
            tid = track_ids[mi]
            if tid is not None and tid in self._track_to_slot:
                si = self._track_to_slot[tid]
                tid_matches.append((mi, si, tid))
            else:
                no_tid.append(mi)

        # --- Swap guard: validate when we have exactly 2 track_id matches ---
        if len(tid_matches) == 2:
            mi_a, si_a, tid_a = tid_matches[0]
            mi_b, si_b, tid_b = tid_matches[1]

            # Only check if they map to different slots (normal case)
            if si_a != si_b:
                # Cost of YOLO's proposed assignment (straight)
                cost_straight = (
                    self._compute_cost(centroids[mi_a], areas[mi_a], masks[mi_a], self.slots[si_a])
                    + self._compute_cost(centroids[mi_b], areas[mi_b], masks[mi_b], self.slots[si_b])
                )
                # Cost of swapped assignment
                cost_swapped = (
                    self._compute_cost(centroids[mi_a], areas[mi_a], masks[mi_a], self.slots[si_b])
                    + self._compute_cost(centroids[mi_b], areas[mi_b], masks[mi_b], self.slots[si_a])
                )

                if cost_swapped < cost_straight:
                    # YOLO swapped IDs — reject all track_id matches
                    logger.info(
                        "Swap guard triggered: straight_cost=%.3f > swapped_cost=%.3f "
                        "(tids %s→slots %s). Delegating to Hungarian.",
                        cost_straight, cost_swapped,
                        [tid_a, tid_b], [si_a, si_b],
                    )
                    return set(), set(), list(range(len(masks)))

        # --- No swap detected: accept track_id matches ---
        matched_masks: set = set()
        matched_slots: set = set()
        unmatched_mask_idx: list = list(no_tid)

        for mi, si, tid in tid_matches:
            if si not in matched_slots:
                self._assign_to_slot(si, masks[mi], centroids[mi], areas[mi], tid)
                matched_masks.add(mi)
                matched_slots.add(si)
            else:
                unmatched_mask_idx.append(mi)

        return matched_masks, matched_slots, unmatched_mask_idx

    def _hungarian_assign(self, masks, centroids, areas,
                          unmatched_mask_idx, unmatched_slots,
                          matched_masks, matched_slots, track_ids):
        """Optimal assignment using soft cost matrix."""
        n_masks = len(unmatched_mask_idx)
        n_slots = len(unmatched_slots)
        cost = np.full((n_masks, n_slots), INF_COST, dtype=np.float64)

        for row, mi in enumerate(unmatched_mask_idx):
            for col, si in enumerate(unmatched_slots):
                slot = self.slots[si]
                cost[row, col] = self._compute_cost(
                    centroids[mi], areas[mi], masks[mi], slot,
                )

        row_ind, col_ind = linear_sum_assignment(cost)

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self.cost_threshold:
                continue
            mi = unmatched_mask_idx[r]
            si = unmatched_slots[c]
            self._assign_to_slot(si, masks[mi], centroids[mi], areas[mi], track_ids[mi])
            matched_masks.add(mi)
            matched_slots.add(si)

    def _compute_cost(self, centroid, area, mask, slot: SlotState) -> float:
        """Compute soft assignment cost between a mask and a slot.

        Returns a value in [0, ~1.0], or INF_COST for pathological cases.
        """
        if not slot.active or slot.centroid is None:
            return INF_COST

        return compute_assignment_cost(
            centroid, area, mask,
            slot.centroid, float(slot.area), slot.mask,
            self.max_distance, self.w_dist, self.w_iou, self.w_area,
        )

    def _assign_to_slot(self, slot_idx, mask, centroid, area, track_id):
        """Assign a mask to a slot, updating all state."""
        slot = self.slots[slot_idx]

        # Update track_id ↔ slot mapping
        old_tid = slot.yolo_track_id
        if old_tid is not None and old_tid in self._track_to_slot:
            del self._track_to_slot[old_tid]
        if track_id is not None:
            # Remove any other slot claiming this track_id
            if track_id in self._track_to_slot:
                other_slot = self._track_to_slot[track_id]
                if other_slot != slot_idx:
                    self.slots[other_slot].yolo_track_id = None
            self._track_to_slot[track_id] = slot_idx

        slot.centroid = centroid
        slot.area = area
        slot.mask = mask
        slot.missing_count = 0
        slot.active = True
        slot.yolo_track_id = track_id

    def _increment_missing_all(self):
        """Increment missing count for all active slots."""
        for i in range(self.num_slots):
            if self.slots[i].active:
                self.slots[i].missing_count += 1
                if self.slots[i].missing_count > self.max_missing_frames:
                    self._release_slot(i)

    def _release_slot(self, slot_idx: int) -> None:
        """Release a slot, clearing its state."""
        slot = self.slots[slot_idx]
        if slot.yolo_track_id is not None:
            self._track_to_slot.pop(slot.yolo_track_id, None)
        slot.centroid = None
        slot.area = 0
        slot.mask = None
        slot.missing_count = 0
        slot.active = False
        slot.yolo_track_id = None

    def _find_free_slot(self) -> Optional[int]:
        """Find the first inactive slot index."""
        for i, s in enumerate(self.slots):
            if not s.active:
                return i
        return None
