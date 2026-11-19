"""SAM3 recovery for cutie_composite (no torch here: SAM3 is passed in as segment_fn).

When the area guard raises its flag, one recovery attempt runs on the current frame:

1. Group: the flagged animals plus the animals whose mask touches them (the likely
   "eater"), using each flagged animal's current mask and its last healthy mask moved
   by its last velocity.
2. SAM3 on a crop around the group (the animals look bigger to SAM3 than in the full
   frame). Candidates that mostly lie on an animal outside the group are dropped.
3. Verified: every flagged animal is "low" and SAM3 sees the same mask as CUTIE
   (IoU >= verify_iou), e.g. an animal rearing. Nothing changes.
4. Assignment: candidates are matched to the group by overlap with each animal's last
   healthy mask (moved by velocity) plus centroid distance, with the Hungarian method.
   A candidate whose area is outside [low_ratio, high_ratio] of that animal's median is
   never assigned (a merged two-animal blob cannot be given to one identity).
5. Residual: a flagged low/missing animal still without a mask is looked for in the
   background residual (pixels that differ from the background plate and no other
   animal covers), optionally refined with SAM3 on a crop around the blob.
6. Recovered only if every animal ends with a mask and every animal in the group has a
   plausible area; then the corrected index mask is re-injected into CUTIE. Otherwise
   the attempt failed and the guard schedules a retry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.pipelines.cutie_composite.area_guard import (
    KIND_HIGH, KIND_LOW, KIND_MISSING, AreaGuard, GuardStatus,
)

logger = logging.getLogger(__name__)

SegmentFn = Callable[[np.ndarray], Tuple[Sequence[np.ndarray], Sequence[float]]]
_BIG = 1e6


@dataclass
class RecoveryConfig:
    crop_margin: float = 0.30
    crop_min_size: int = 192
    crop_max_frame_fraction: float = 0.60
    touch_px: int = 6
    centroid_weight: float = 0.5
    centroid_scale_px: float = 165.0
    max_predict_px: float = 120.0
    verify_iou: float = 0.80
    foreign_overlap: float = 0.50
    min_candidate_px: int = 20
    residual_enabled: bool = True
    residual_diff_threshold: int = 25
    residual_open_px: int = 3
    residual_cover_dilate_px: int = 5
    residual_refine_with_sam3: bool = True
    residual_refine_min_iou: float = 0.30

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "RecoveryConfig":
        cfg = cfg or {}
        res = cfg.get("residual", {}) or {}
        d = cls()
        return cls(
            crop_margin=float(cfg.get("crop_margin", d.crop_margin)),
            crop_min_size=int(cfg.get("crop_min_size", d.crop_min_size)),
            crop_max_frame_fraction=float(cfg.get("crop_max_frame_fraction", d.crop_max_frame_fraction)),
            touch_px=int(cfg.get("touch_px", d.touch_px)),
            centroid_weight=float(cfg.get("centroid_weight", d.centroid_weight)),
            centroid_scale_px=float(cfg.get("centroid_scale_px", d.centroid_scale_px)),
            max_predict_px=float(cfg.get("max_predict_px", d.max_predict_px)),
            verify_iou=float(cfg.get("verify_iou", d.verify_iou)),
            foreign_overlap=float(cfg.get("foreign_overlap", d.foreign_overlap)),
            min_candidate_px=int(cfg.get("min_candidate_px", d.min_candidate_px)),
            residual_enabled=bool(res.get("enabled", d.residual_enabled)),
            residual_diff_threshold=int(res.get("diff_threshold", d.residual_diff_threshold)),
            residual_open_px=int(res.get("open_px", d.residual_open_px)),
            residual_cover_dilate_px=int(res.get("cover_dilate_px", d.residual_cover_dilate_px)),
            residual_refine_with_sam3=bool(res.get("refine_with_sam3", d.residual_refine_with_sam3)),
            residual_refine_min_iou=float(res.get("refine_min_iou", d.residual_refine_min_iou)),
        )


@dataclass
class StepResult:
    slot_masks: List[Optional[np.ndarray]]
    slot_centroids: List[Optional[Tuple[float, float]]]
    sources: List[Optional[str]]                      # "recovered" where SAM3 replaced CUTIE
    extras: List[Tuple[np.ndarray, Optional[float]]] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    reinject_index_mask: Optional[np.ndarray] = None
    sam3_calls: int = 0


# ---------------------------------------------------------------------- helpers
def mask_centroid(mask: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return mask.copy()
    h, w = mask.shape
    m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(mask.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def crop_box(region: np.ndarray, margin: float, min_size: int, max_fraction: float) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) around `region`, padded; the full frame when the box gets too big."""
    h, w = region.shape
    ys, xs = np.nonzero(region)
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    pad_x, pad_y = int(round(margin * bw)), int(round(margin * bh))
    x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y
    if x1 - x0 < min_size:
        extra = (min_size - (x1 - x0) + 1) // 2
        x0, x1 = x0 - extra, x1 + extra
    if y1 - y0 < min_size:
        extra = (min_size - (y1 - y0) + 1) // 2
        y0, y1 = y0 - extra, y1 + extra
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if (x1 - x0) * (y1 - y0) > max_fraction * w * h:
        return 0, 0, w, h
    return x0, y0, x1, y1


def compose_index_mask(slot_masks: Sequence[Optional[np.ndarray]], shape: Tuple[int, int],
                       priority: Sequence[int] = ()) -> np.ndarray:
    """Index mask (0 = background, i+1 = slot i); slots in `priority` are painted last."""
    idx = np.zeros(shape, dtype=np.int32)
    prio = set(priority)
    order = [s for s in range(len(slot_masks)) if s not in prio] + [s for s in priority]
    for s in order:
        m = slot_masks[s]
        if m is not None:
            idx[m] = s + 1
    return idx


# ---------------------------------------------------------------------- recovery
class MaskRecovery:
    def __init__(self, num_slots: int, guard: AreaGuard, cfg: RecoveryConfig,
                 segment_fn: SegmentFn, background_bgr: Optional[np.ndarray]):
        self.n = int(num_slots)
        self.guard = guard
        self.cfg = cfg
        self.segment_fn = segment_fn
        self.bg_gray = (cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
                        if background_bgr is not None else None)
        self.attempts = 0
        self.recoveries = 0

    # -- one frame of phase 2 --------------------------------------------------
    def step(self, frame: int, frame_rgb: np.ndarray, slot_masks: Sequence[Optional[np.ndarray]],
             slot_centroids: Sequence[Optional[Tuple[float, float]]]) -> StepResult:
        res = StepResult(slot_masks=list(slot_masks), slot_centroids=list(slot_centroids),
                         sources=[None] * self.n)
        status = self.guard.update(frame, slot_masks, slot_centroids, act=True)
        n_closed = len(self.guard.episodes)
        opening = self.guard.episode is None and bool(status.triggered)
        attempt = self.guard.should_attempt(status)
        if len(self.guard.episodes) > n_closed:
            res.events.append({"type": "episode_end", **self.guard.episodes[-1].as_dict()})
        if opening:
            res.events.append({"type": "flag", "slots": [s + 1 for s in sorted(status.triggered)],
                               "kinds": {str(s + 1): k for s, k in sorted(status.triggered.items())},
                               "onsets": {str(s + 1): int(f) for s, f in sorted(status.onsets.items())}})
        if not attempt:
            return res

        self.attempts += 1
        outcome, detail, new_masks, changed, extras, calls, group = self._attempt(
            frame, frame_rgb, slot_masks, status)
        res.sam3_calls = calls
        res.extras = extras
        n_closed = len(self.guard.episodes)
        self.guard.record_attempt(frame, outcome, detail, slots=group)
        res.events.append({"type": "attempt", "outcome": outcome,
                           "slots": [s + 1 for s in sorted(group)], "detail": detail})
        if len(self.guard.episodes) > n_closed:
            res.events.append({"type": "episode_end", **self.guard.episodes[-1].as_dict()})
        logger.info("Recovery frame %d: %s (%s)", frame, outcome, detail)
        if outcome == "recovered":
            self.recoveries += 1
            res.slot_masks = new_masks
            res.slot_centroids = [mask_centroid(m) for m in new_masks]
            for s in changed:
                res.sources[s] = "recovered"
            res.reinject_index_mask = compose_index_mask(new_masks, frame_rgb.shape[:2], priority=sorted(changed))
        return res

    # -- one attempt -----------------------------------------------------------
    def _attempt(self, frame, frame_rgb, slot_masks, status: GuardStatus):
        cfg, n = self.cfg, self.n
        shape = frame_rgb.shape[:2]
        low, high = self.guard.cfg.low_ratio, self.guard.cfg.high_ratio
        baselines = self.guard.baselines()
        triggered = dict(status.triggered)
        calls = 0

        # last healthy masks, moved by their last velocity (clamped)
        pred_masks: Dict[int, np.ndarray] = {}
        pred_cents: Dict[int, Tuple[float, float]] = {}
        for s in range(n):
            ref = self.guard.healthy[s]
            if ref is None:
                continue
            dt = frame - ref.frame
            dx, dy = ref.velocity[0] * dt, ref.velocity[1] * dt
            d = float(np.hypot(dx, dy))
            if d > cfg.max_predict_px > 0:
                dx, dy = dx * cfg.max_predict_px / d, dy * cfg.max_predict_px / d
            pred_masks[s] = shift_mask(ref.mask, dx, dy)
            if ref.centroid is not None:
                pred_cents[s] = (ref.centroid[0] + dx, ref.centroid[1] + dy)

        # group: flagged animals + animals touching them
        zone = np.zeros(shape, dtype=bool)
        for s in triggered:
            if slot_masks[s] is not None:
                zone |= slot_masks[s]
            if s in pred_masks:
                zone |= pred_masks[s]
        zone = dilate(zone, cfg.touch_px)
        group = set(triggered)
        for s in range(n):
            if s not in group and slot_masks[s] is not None and (slot_masks[s] & zone).any():
                group.add(s)
        group_sorted = sorted(group)

        region = np.zeros(shape, dtype=bool)
        for s in group_sorted:
            if slot_masks[s] is not None:
                region |= slot_masks[s]
            if s in pred_masks:
                region |= pred_masks[s]
        box = crop_box(region, cfg.crop_margin, cfg.crop_min_size, cfg.crop_max_frame_fraction)

        cands, scores = self._segment(frame_rgb, box)
        calls += 1
        outside = np.zeros(shape, dtype=bool)
        for s in range(n):
            if s not in group and slot_masks[s] is not None:
                outside |= slot_masks[s]
        keep = [i for i, c in enumerate(cands) if (c & outside).sum() <= cfg.foreign_overlap * c.sum()]
        cands = [cands[i] for i in keep]
        scores = [scores[i] for i in keep]
        detail = [f"crop {box[2] - box[0]}x{box[3] - box[1]}", f"{len(cands)} SAM3 candidate(s)"]

        # verified: CUTIE was right (only "low" animals can be verified)
        if triggered and all(k == KIND_LOW for k in triggered.values()):
            if all(max((iou(c, slot_masks[s]) for c in cands), default=0.0) >= cfg.verify_iou
                   for s in triggered):
                return "verified", "; ".join(detail + ["SAM3 agrees with CUTIE"]), None, [], \
                    self._extras(cands, scores, set()), calls, group_sorted

        # assignment by overlap with the last healthy mask + centroid distance
        assigned: Dict[int, np.ndarray] = {}
        used = set()
        if cands:
            cost = np.full((len(group_sorted), len(cands)), _BIG)
            for i, s in enumerate(group_sorted):
                for j, c in enumerate(cands):
                    area = int(c.sum())
                    if baselines[s] and not (low <= area / baselines[s] <= high):
                        continue
                    ov = iou(c, pred_masks.get(s))
                    cc = mask_centroid(c)
                    if s in pred_cents and cc is not None:
                        dist = np.hypot(cc[0] - pred_cents[s][0], cc[1] - pred_cents[s][1]) / cfg.centroid_scale_px
                    else:
                        dist = 1.0
                    cost[i, j] = (1.0 - ov) + cfg.centroid_weight * min(dist, 2.0)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < _BIG:
                    assigned[group_sorted[i]] = cands[j]
                    used.add(j)
        detail.append("assigned " + (",".join(f"R{s + 1}" for s in sorted(assigned)) or "none"))

        # residual for flagged low / missing animals still without a mask
        need = [s for s in group_sorted if s not in assigned and triggered.get(s) in (KIND_LOW, KIND_MISSING)]
        if need and cfg.residual_enabled and self.bg_gray is not None:
            found, extra_calls = self._residual(frame_rgb, slot_masks, assigned, need, pred_cents, baselines, low, high)
            calls += extra_calls
            assigned.update(found)
            detail.append("residual " + (",".join(f"R{s + 1}" for s in sorted(found)) or "none"))

        # compose: assigned masks win, everyone else loses the pixels they overlap
        taken = np.zeros(shape, dtype=bool)
        for m in assigned.values():
            taken |= m
        new_masks: List[Optional[np.ndarray]] = []
        for s in range(n):
            if s in assigned:
                new_masks.append(assigned[s].copy())
            elif slot_masks[s] is not None:
                m = slot_masks[s] & ~taken
                new_masks.append(m if m.any() else None)
            else:
                new_masks.append(None)
        # assigned candidates may overlap each other: paint in order, later ones keep their pixels
        painted = np.zeros(shape, dtype=bool)
        for s in sorted(assigned, reverse=True):
            new_masks[s] = new_masks[s] & ~painted
            painted |= new_masks[s]

        complete = all(m is not None and m.any() for m in new_masks)
        plausible = all(not baselines[s] or (new_masks[s] is not None and
                        low <= new_masks[s].sum() / baselines[s] <= high) for s in group_sorted)
        extras = self._extras(cands, scores, used)
        if complete and plausible and assigned:
            changed = [s for s in group_sorted if slot_masks[s] is None or new_masks[s] is None
                       or iou(new_masks[s], slot_masks[s]) < 0.999]
            return "recovered", "; ".join(detail), new_masks, changed, extras, calls, group_sorted
        why = "an animal still has no mask" if not complete else "an area is still out of range"
        return "failed", "; ".join(detail + [why]), None, [], extras, calls, group_sorted

    # -- pieces ----------------------------------------------------------------
    def _segment(self, frame_rgb: np.ndarray, box) -> Tuple[List[np.ndarray], List[Optional[float]]]:
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(frame_rgb[y0:y1, x0:x1])
        masks, scores = self.segment_fn(crop)
        out, out_scores = [], []
        for k, m in enumerate(masks):
            m = np.asarray(m).astype(bool)
            if m.shape != crop.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (crop.shape[1], crop.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.sum() < self.cfg.min_candidate_px:
                continue
            full = np.zeros(frame_rgb.shape[:2], dtype=bool)
            full[y0:y1, x0:x1] = m
            out.append(full)
            out_scores.append(float(scores[k]) if scores is not None and k < len(scores) else None)
        return out, out_scores

    @staticmethod
    def _extras(cands, scores, used) -> List[Tuple[np.ndarray, Optional[float]]]:
        return [(c, scores[j]) for j, c in enumerate(cands) if j not in used]

    def _residual(self, frame_rgb, slot_masks, assigned, need, pred_cents, baselines, low, high):
        """Blobs that differ from the background and no other animal covers."""
        cfg = self.cfg
        shape = frame_rgb.shape[:2]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        fg = cv2.absdiff(gray, self.bg_gray) > cfg.residual_diff_threshold
        covered = np.zeros(shape, dtype=bool)
        for s in range(self.n):
            if s in need:
                continue
            m = assigned.get(s, slot_masks[s])
            if m is not None:
                covered |= m
        fg &= ~dilate(covered, cfg.residual_cover_dilate_px)
        if cfg.residual_open_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.residual_open_px + 1,) * 2)
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        n_lab, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
        blobs = [labels == i for i in range(1, n_lab)]
        found: Dict[int, np.ndarray] = {}
        calls = 0
        if not blobs:
            return found, calls
        cost = np.full((len(need), len(blobs)), _BIG)
        for i, s in enumerate(need):
            for j, b in enumerate(blobs):
                area = int(b.sum())
                if baselines[s] and not (low <= area / baselines[s] <= high):
                    continue
                cb = mask_centroid(b)
                if s in pred_cents and cb is not None:
                    cost[i, j] = np.hypot(cb[0] - pred_cents[s][0], cb[1] - pred_cents[s][1])
                else:
                    cost[i, j] = 1.0
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] >= _BIG:
                continue
            s, blob = need[i], blobs[j]
            mask = blob
            if cfg.residual_refine_with_sam3:
                box = crop_box(blob, self.cfg.crop_margin, self.cfg.crop_min_size, self.cfg.crop_max_frame_fraction)
                cands, _ = self._segment(frame_rgb, box)
                calls += 1
                best = max(cands, key=lambda c: iou(c, blob), default=None)
                if best is not None and iou(best, blob) >= cfg.residual_refine_min_iou and \
                        (not baselines[s] or low <= best.sum() / baselines[s] <= high):
                    mask = best & ~covered
            found[s] = mask
        return found, calls
