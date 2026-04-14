"""
Social contact classification for lab rat tracking.

Classifies pairwise interactions between tracked animals using keypoint
geometry, SAM2 mask overlap, and velocity estimation. Produces per-frame
contact events and groups them into temporal bouts.

Contact types (priority order — specific → general):
  1. N2N  — nose-to-nose (most specific keypoint check)
  2. N2AG — nose-to-anogenital (specific keypoint pair)
  3. T2T  — tail-to-tail (specific keypoint pair)
  4. FOL  — following (requires motion + spatial pattern)
  5. SBS  — side-by-side (requires mask overlap + posture)
  6. N2B  — nose-to-body (catch-all for remaining nose contact)

N2B is last because its mask-containment check is very broad (any nose
inside any part of the other rat's mask). Placing it before FOL/SBS/T2T
would absorb those more specific contact types.

See docs/contacts/design.md for full specification.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .metrics import mask_iou as compute_mask_iou
from .utils import Detection, Keypoint

logger = logging.getLogger(__name__)


# ── Enums & dataclasses ─────────────────────────────────────────────────

class ContactType(str, Enum):
    N2N = "N2N"
    N2AG = "N2AG"
    N2B = "N2B"
    T2T = "T2T"
    FOL = "FOL"
    SBS = "SBS"


class Zone(str, Enum):
    CONTACT = "contact"
    PROXIMITY = "proximity"
    INDEPENDENT = "independent"


@dataclass
class ContactEvent:
    """A single contact observation for one pair in one frame."""
    frame_idx: int
    time_sec: float
    rat_a_slot: int
    rat_b_slot: int
    rat_a_track_id: Optional[int]
    rat_b_track_id: Optional[int]
    zone: str
    contact_type: Optional[str] = None
    investigator_slot: Optional[int] = None
    nose_nose_dist_px: Optional[float] = None
    nose_nose_dist_bl: Optional[float] = None
    nose_tailbase_dist_px: Optional[float] = None
    nose_tailbase_dist_bl: Optional[float] = None
    centroid_dist_px: Optional[float] = None
    centroid_dist_bl: Optional[float] = None
    mask_iou: float = 0.0
    body_length_a_px: Optional[float] = None
    body_length_b_px: Optional[float] = None
    velocity_a_px: Optional[float] = None
    velocity_b_px: Optional[float] = None
    orientation_cos: Optional[float] = None
    quality_flag: Optional[str] = None
    bout_id: Optional[int] = None


@dataclass
class Bout:
    """A temporal grouping of consecutive same-type contact events."""
    bout_id: int
    contact_type: str
    rat_a_slot: int
    rat_b_slot: int
    investigator_slot: Optional[int]
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_frames: int = 0
    total_frames: int = 0
    # Accumulation buffers (not exported directly)
    _nose_dists: List[float] = field(default_factory=list, repr=False)
    _mask_ious: List[float] = field(default_factory=list, repr=False)
    _vel_a: List[float] = field(default_factory=list, repr=False)
    _vel_b: List[float] = field(default_factory=list, repr=False)
    _quality_flags: List[str] = field(default_factory=list, repr=False)

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "bout_id": self.bout_id,
            "contact_type": self.contact_type,
            "rat_a_slot": self.rat_a_slot,
            "rat_b_slot": self.rat_b_slot,
            "investigator_slot": self.investigator_slot if self.investigator_slot is not None else "",
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": round(self.start_time_sec, 4),
            "end_time_sec": round(self.end_time_sec, 4),
            "duration_sec": round(self.duration_sec, 4),
            "duration_frames": self.duration_frames,
            "total_frames": self.total_frames,
            "mean_nose_dist_px": round(sum(self._nose_dists) / len(self._nose_dists), 1) if self._nose_dists else "",
            "mean_mask_iou": round(sum(self._mask_ious) / len(self._mask_ious), 4) if self._mask_ious else 0.0,
            "mean_velocity_a_px": round(sum(self._vel_a) / len(self._vel_a), 1) if self._vel_a else "",
            "mean_velocity_b_px": round(sum(self._vel_b) / len(self._vel_b), 1) if self._vel_b else "",
            "quality_flags": ",".join(sorted(set(self._quality_flags))) if self._quality_flags else "",
        }


# ── Geometry helpers ─────────────────────────────────────────────────────

def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _get_kp(det: Detection, index: int, min_conf: float) -> Optional[Tuple[float, float]]:
    """Get keypoint (x, y) if it exists and has sufficient confidence."""
    if det.keypoints is None or index >= len(det.keypoints):
        return None
    kp = det.keypoints[index]
    if kp.conf < min_conf:
        return None
    return (kp.x, kp.y)


def _valid_kp_count(det: Detection, min_conf: float) -> int:
    """Count keypoints with confidence >= min_conf."""
    if det.keypoints is None:
        return 0
    return sum(1 for kp in det.keypoints if kp.conf >= min_conf)


def estimate_body_length(det: Detection, min_conf: float) -> Optional[float]:
    """Estimate body length as dist(nose, tail_base).

    Falls back to bbox diagonal * 0.6 if keypoints unavailable.
    """
    nose = _get_kp(det, 4, min_conf)
    tail_base = _get_kp(det, 1, min_conf)
    if nose is not None and tail_base is not None:
        d = _euclidean(nose, tail_base)
        if d > 5.0:  # sanity: at least 5px
            return d
    # Fallback: bbox diagonal * 0.6
    diag = math.sqrt((det.x2 - det.x1) ** 2 + (det.y2 - det.y1) ** 2)
    if diag > 10.0:
        return diag * 0.6
    return None


def _body_orientation(det: Detection, min_conf: float) -> Optional[Tuple[float, float]]:
    """Unit vector from tail_base to nose (body direction)."""
    nose = _get_kp(det, 4, min_conf)
    tail_base = _get_kp(det, 1, min_conf)
    if nose is None or tail_base is None:
        return None
    dx = nose[0] - tail_base[0]
    dy = nose[1] - tail_base[1]
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1.0:
        return None
    return (dx / mag, dy / mag)


def _cos_angle(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Cosine of angle between two vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (m1 * m2)))


# ── Per-frame contact classification ────────────────────────────────────

def classify_pair_contacts(
    det_a: Detection,
    det_b: Detection,
    slot_a: int,
    slot_b: int,
    mask_a: Optional[np.ndarray],
    mask_b: Optional[np.ndarray],
    vel_a: Optional[float],
    vel_b: Optional[float],
    vel_vec_a: Optional[Tuple[float, float]],
    vel_vec_b: Optional[Tuple[float, float]],
    body_length: float,
    frame_idx: int,
    time_sec: float,
    cfg: Dict[str, Any],
) -> ContactEvent:
    """Classify the contact between a pair of detections in one frame.

    Args:
        det_a, det_b: Detection objects with keypoints.
        slot_a, slot_b: Slot indices for stable identity.
        mask_a, mask_b: SAM2 boolean masks (or None).
        vel_a, vel_b: Centroid displacement in px since previous frame.
        vel_vec_a, vel_vec_b: Velocity vectors (dx, dy) since previous frame.
        body_length: Average body length for this pair (px).
        frame_idx: Current frame index.
        time_sec: Current timestamp.
        cfg: contacts config dict.

    Returns:
        ContactEvent describing the pair's interaction this frame.
    """
    min_conf = cfg.get("min_keypoint_conf", 0.3)
    contact_zone_bl = cfg.get("contact_zone_bl", 0.3)
    proximity_zone_bl = cfg.get("proximity_zone_bl", 1.0)
    sbs_iou_min = cfg.get("sbs_mask_iou_min", 0.02)
    sbs_cos_min = cfg.get("sbs_parallel_cos_min", 0.7)
    follow_radius_bl = cfg.get("follow_radius_bl", 0.5)
    follow_cos_min = cfg.get("follow_alignment_cos", 0.7)
    mask_overlap_warn = cfg.get("mask_overlap_warning", 0.5)

    # Velocity thresholds: prefer body-length units, fall back to legacy px
    if "sbs_max_velocity_bl" in cfg:
        sbs_max_vel = cfg["sbs_max_velocity_bl"] * body_length
    else:
        sbs_max_vel = cfg.get("sbs_max_velocity_px", 5.0)
    if "follow_min_speed_bl" in cfg:
        follow_min_speed = cfg["follow_min_speed_bl"] * body_length
    else:
        follow_min_speed = cfg.get("follow_min_speed_px", 3.0)

    contact_radius = body_length * contact_zone_bl
    follow_radius = body_length * follow_radius_bl

    # Compute distances
    centroid_a = det_a.center()
    centroid_b = det_b.center()
    centroid_dist = _euclidean(centroid_a, centroid_b)
    centroid_dist_bl = centroid_dist / body_length if body_length > 0 else None

    # Keypoint distances
    nose_a = _get_kp(det_a, 4, min_conf)
    nose_b = _get_kp(det_b, 4, min_conf)
    tail_base_a = _get_kp(det_a, 1, min_conf)
    tail_base_b = _get_kp(det_b, 1, min_conf)
    mid_body_a = _get_kp(det_a, 3, min_conf)
    mid_body_b = _get_kp(det_b, 3, min_conf)

    nose_nose_dist = _euclidean(nose_a, nose_b) if nose_a and nose_b else None
    nose_nose_bl = nose_nose_dist / body_length if nose_nose_dist is not None and body_length > 0 else None

    # Nose-to-tail_base: check both directions, take minimum
    n2tb_ab = _euclidean(nose_a, tail_base_b) if nose_a and tail_base_b else None
    n2tb_ba = _euclidean(nose_b, tail_base_a) if nose_b and tail_base_a else None
    nose_tb_dist = None
    if n2tb_ab is not None and n2tb_ba is not None:
        nose_tb_dist = min(n2tb_ab, n2tb_ba)
    elif n2tb_ab is not None:
        nose_tb_dist = n2tb_ab
    elif n2tb_ba is not None:
        nose_tb_dist = n2tb_ba
    nose_tb_bl = nose_tb_dist / body_length if nose_tb_dist is not None and body_length > 0 else None

    # Mask IoU
    m_iou = 0.0
    if mask_a is not None and mask_b is not None:
        m_iou = compute_mask_iou(mask_a, mask_b)

    # Body lengths per rat
    bl_a = estimate_body_length(det_a, min_conf)
    bl_b = estimate_body_length(det_b, min_conf)

    # Orientation
    orient_a = _body_orientation(det_a, min_conf)
    orient_b = _body_orientation(det_b, min_conf)
    orient_cos = _cos_angle(orient_a, orient_b) if orient_a and orient_b else None

    # Quality flags
    quality_flag = None
    # Check if either detection has carried-over (stale) keypoints
    if getattr(det_a, "is_carried_over", False) or getattr(det_b, "is_carried_over", False):
        quality_flag = "stale_keypoints"
    elif m_iou > mask_overlap_warn:
        quality_flag = "high_mask_overlap"
    elif _valid_kp_count(det_a, min_conf) < 2 or _valid_kp_count(det_b, min_conf) < 2:
        quality_flag = "missing_keypoints"

    # Determine zone
    if centroid_dist_bl is not None:
        if centroid_dist_bl < contact_zone_bl:
            zone = Zone.CONTACT
        elif centroid_dist_bl < proximity_zone_bl:
            zone = Zone.PROXIMITY
        else:
            zone = Zone.INDEPENDENT
    else:
        zone = Zone.INDEPENDENT

    # Build base event
    event = ContactEvent(
        frame_idx=frame_idx,
        time_sec=round(time_sec, 4),
        rat_a_slot=slot_a,
        rat_b_slot=slot_b,
        rat_a_track_id=det_a.track_id,
        rat_b_track_id=det_b.track_id,
        zone=zone.value,
        nose_nose_dist_px=round(nose_nose_dist, 1) if nose_nose_dist is not None else None,
        nose_nose_dist_bl=round(nose_nose_bl, 4) if nose_nose_bl is not None else None,
        nose_tailbase_dist_px=round(nose_tb_dist, 1) if nose_tb_dist is not None else None,
        nose_tailbase_dist_bl=round(nose_tb_bl, 4) if nose_tb_bl is not None else None,
        centroid_dist_px=round(centroid_dist, 1),
        centroid_dist_bl=round(centroid_dist_bl, 4) if centroid_dist_bl is not None else None,
        mask_iou=round(m_iou, 4),
        body_length_a_px=round(bl_a, 1) if bl_a is not None else None,
        body_length_b_px=round(bl_b, 1) if bl_b is not None else None,
        velocity_a_px=round(vel_a, 1) if vel_a is not None else None,
        velocity_b_px=round(vel_b, 1) if vel_b is not None else None,
        orientation_cos=round(orient_cos, 4) if orient_cos is not None else None,
        quality_flag=quality_flag,
    )

    # ── Classification in priority order (specific → general) ──
    # N2B is last because its mask-containment check is very broad and would
    # absorb T2T, FOL, and SBS if evaluated first.

    # 1. Nose-to-nose (most specific: both noses close)
    if nose_nose_dist is not None and nose_nose_dist < contact_radius:
        event.contact_type = ContactType.N2N.value
        event.zone = Zone.CONTACT.value
        return event

    # 2. Nose-to-anogenital (asymmetric: pick closest direction first)
    n2ag_a = n2tb_ab is not None and n2tb_ab < contact_radius
    n2ag_b = n2tb_ba is not None and n2tb_ba < contact_radius
    if n2ag_a and n2ag_b:
        # Both directions qualify — pick the closer one (no A-first bias)
        if n2tb_ab <= n2tb_ba:
            event.investigator_slot = slot_a
        else:
            event.investigator_slot = slot_b
        event.contact_type = ContactType.N2AG.value
        event.zone = Zone.CONTACT.value
        return event
    elif n2ag_a:
        event.contact_type = ContactType.N2AG.value
        event.investigator_slot = slot_a
        event.zone = Zone.CONTACT.value
        return event
    elif n2ag_b:
        event.contact_type = ContactType.N2AG.value
        event.investigator_slot = slot_b
        event.zone = Zone.CONTACT.value
        return event

    # 3. Tail-to-tail: both tail bases close (rear-to-rear contact)
    tb_tb_dist = _euclidean(tail_base_a, tail_base_b) if tail_base_a and tail_base_b else None
    if tb_tb_dist is not None and tb_tb_dist < contact_radius:
        event.contact_type = ContactType.T2T.value
        event.zone = Zone.CONTACT.value
        return event

    # 4. Following: nose near tail_base of other + both moving + aligned velocity
    # Pick closer direction first (no A-first bias)
    if vel_vec_a is not None and vel_vec_b is not None:
        fol_a = (n2tb_ab is not None and n2tb_ab < follow_radius
                 and vel_a is not None and vel_b is not None
                 and vel_a > follow_min_speed and vel_b > follow_min_speed)
        fol_b = (n2tb_ba is not None and n2tb_ba < follow_radius
                 and vel_a is not None and vel_b is not None
                 and vel_a > follow_min_speed and vel_b > follow_min_speed)
        if fol_a or fol_b:
            v_cos = _cos_angle(vel_vec_a, vel_vec_b)
            if v_cos > follow_cos_min:
                if fol_a and fol_b:
                    inv = slot_a if n2tb_ab <= n2tb_ba else slot_b
                elif fol_a:
                    inv = slot_a
                else:
                    inv = slot_b
                event.contact_type = ContactType.FOL.value
                event.investigator_slot = inv
                event.zone = Zone.CONTACT.value
                return event

    # 5. Side-by-side: mask overlap + low velocity + parallel orientation
    if m_iou > sbs_iou_min:
        low_vel_a = vel_a is not None and vel_a < sbs_max_vel
        low_vel_b = vel_b is not None and vel_b < sbs_max_vel
        # Also accept if velocity is unknown (first frame)
        vel_ok = (low_vel_a and low_vel_b) or (vel_a is None and vel_b is None)
        parallel = orient_cos is not None and abs(orient_cos) > sbs_cos_min
        if vel_ok and parallel:
            event.contact_type = ContactType.SBS.value
            event.zone = Zone.CONTACT.value
            return event

    # 6. Nose-to-body (catch-all: nose inside other's mask or near mid_body)
    # Guard: skip if nose is near the other's tail_base (N2AG territory) or
    # nose (N2N territory) — those were already checked above and didn't fire,
    # meaning they were just outside threshold. Only N2B if the nose is near
    # the trunk/flank region.
    n2b_detected = False
    n2b_investigator = None
    for nose_x, mask_other, slot_inv, tb_other, nose_other, mid_other in [
        (nose_a, mask_b, slot_a, tail_base_b, nose_b, mid_body_b),
        (nose_b, mask_a, slot_b, tail_base_a, nose_a, mid_body_a),
    ]:
        if n2b_detected:
            break
        if nose_x is None:
            continue
        # Guard: if nose is close to the other's tail_base or nose, skip
        # (those are N2AG/N2N territory — they didn't fire because they were
        # just outside contact_radius, so N2B shouldn't catch them either)
        if tb_other is not None and _euclidean(nose_x, tb_other) < contact_radius * 1.5:
            continue
        if nose_other is not None and _euclidean(nose_x, nose_other) < contact_radius * 1.5:
            continue
        # Mask containment check
        if mask_other is not None:
            ny, nx = round(nose_x[1]), round(nose_x[0])
            if 0 <= ny < mask_other.shape[0] and 0 <= nx < mask_other.shape[1]:
                if mask_other[ny, nx]:
                    n2b_detected = True
                    n2b_investigator = slot_inv
                    break
        # Fallback: keypoint distance to mid_body
        if mid_other is not None:
            if _euclidean(nose_x, mid_other) < contact_radius * 1.5:
                n2b_detected = True
                n2b_investigator = slot_inv
                break
    if n2b_detected:
        event.contact_type = ContactType.N2B.value
        event.investigator_slot = n2b_investigator
        event.zone = Zone.CONTACT.value
        return event

    # No specific contact type detected — zone already set
    return event


# ── ContactTracker — multi-frame state + bout management ────────────────

# CSV column order for per-frame output
FRAME_CSV_COLUMNS = [
    "frame_idx", "time_sec", "rat_a_slot", "rat_b_slot",
    "rat_a_track_id", "rat_b_track_id", "zone", "contact_type",
    "investigator_slot", "nose_nose_dist_px", "nose_nose_dist_bl",
    "nose_tailbase_dist_px", "nose_tailbase_dist_bl",
    "centroid_dist_px", "centroid_dist_bl", "mask_iou",
    "body_length_a_px", "body_length_b_px",
    "velocity_a_px", "velocity_b_px", "orientation_cos",
    "quality_flag", "bout_id",
]

BOUT_CSV_COLUMNS = [
    "bout_id", "contact_type", "rat_a_slot", "rat_b_slot",
    "investigator_slot", "start_frame", "end_frame",
    "start_time_sec", "end_time_sec", "duration_sec",
    "duration_frames", "total_frames",
    "mean_nose_dist_px", "mean_mask_iou",
    "mean_velocity_a_px", "mean_velocity_b_px", "quality_flags",
]


class ContactTracker:
    """Accumulates per-frame contact events into bouts and produces outputs.

    Maintains velocity estimation from slot centroids across frames.
    Writes per-frame CSV incrementally. Generates bout CSV, session JSON,
    and PDF report on finalize().
    """

    def __init__(
        self,
        output_dir: Path,
        fps: float,
        num_slots: int,
        video_path: str,
        config: Dict[str, Any],
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.num_slots = num_slots
        self.video_path = video_path
        self.cfg = config.get("contacts", {})

        self.fallback_bl = self.cfg.get("fallback_body_length_px", 120.0)
        self.bout_max_gap = self.cfg.get("bout_max_gap_frames", 3)
        self.bout_min_dur = self.cfg.get("bout_min_duration_frames", 2)
        self.follow_min_frames = self.cfg.get("follow_min_frames", 5)
        # Match pipeline's max_distance (200px) so detections aren't silently dropped
        self.det_slot_match_radius = self.cfg.get("det_slot_match_radius", 200.0)

        # Body length EMA per slot
        self._bl_ema: Dict[int, float] = {}
        self._bl_alpha = 0.1

        # Velocity tracking: previous centroids per slot + EMA smoothing
        self._prev_centroids: Dict[int, Optional[Tuple[float, float]]] = {}
        self._vel_ema: Dict[int, float] = {}           # smoothed speed per slot
        self._vel_vec_ema: Dict[int, Tuple[float, float]] = {}  # smoothed vector per slot
        self._vel_alpha = 0.3  # EMA weight for new velocity (0.3 = moderate smoothing)

        # Bout tracking
        self._bouts: List[Bout] = []
        self._active_bouts: Dict[str, Bout] = {}  # key: "slotA_slotB_type"
        self._next_bout_id = 0

        # Incremental counters for summary (don't accumulate events in memory)
        self._total_frames = 0
        self._zone_counts = {"contact": 0, "proximity": 0, "independent": 0}
        self._quality_counts = {
            "high_mask_overlap": 0,
            "missing_keypoints": 0,
            "stale_keypoints": 0,
            "single_detection": 0,
            "merged_state": 0,
        }

        # Open per-frame CSV for streaming writes
        self._csv_path = self.output_dir / "contacts_per_frame.csv"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=FRAME_CSV_COLUMNS)
        self._csv_writer.writeheader()

    def update(
        self,
        detections: List[Detection],
        slot_masks: List[Optional[np.ndarray]],
        slot_centroids: List[Optional[Tuple[float, float]]],
        frame_idx: int,
    ) -> List[ContactEvent]:
        """Process one frame's detections and masks.

        Args:
            detections: YOLO detections with keypoints for this frame.
            slot_masks: Per-slot SAM2 masks (None if slot empty).
            slot_centroids: Per-slot centroids (None if slot empty).
            frame_idx: Current frame number.

        Returns:
            List of ContactEvent (one per pair).
        """
        time_sec = frame_idx / self.fps
        self._total_frames += 1

        # Build slot → detection mapping
        # Priority 1: use track_id (pipelines set track_id = slot_idx + 1)
        # Priority 2: fall back to centroid proximity for unmatched detections
        slot_dets: Dict[int, Detection] = {}
        unmatched_dets = []
        for det in detections:
            if det is None:
                continue
            if det.track_id is not None:
                si = det.track_id - 1  # track_id is 1-based
                if 0 <= si < self.num_slots and si not in slot_dets:
                    slot_dets[si] = det
                    continue
            unmatched_dets.append(det)

        # Proximity fallback for detections without track_id
        for det in unmatched_dets:
            det_center = det.center()
            best_si = None
            best_dist = self.det_slot_match_radius
            for si in range(self.num_slots):
                if si in slot_dets:
                    continue
                if slot_centroids[si] is None or slot_masks[si] is None:
                    continue
                d = _euclidean(det_center, slot_centroids[si])
                if d < best_dist:
                    best_dist = d
                    best_si = si
            if best_si is not None:
                slot_dets[best_si] = det

        # Compute velocities from previous centroids (EMA-smoothed)
        velocities: Dict[int, Optional[float]] = {}
        vel_vectors: Dict[int, Optional[Tuple[float, float]]] = {}
        alpha = self._vel_alpha
        for si in range(self.num_slots):
            curr = slot_centroids[si]
            prev = self._prev_centroids.get(si)
            if curr is not None and prev is not None:
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                raw_speed = math.sqrt(dx * dx + dy * dy)
                # EMA smoothing to reduce centroid jitter
                if si in self._vel_ema:
                    self._vel_ema[si] = self._vel_ema[si] * (1 - alpha) + raw_speed * alpha
                    old_vx, old_vy = self._vel_vec_ema[si]
                    self._vel_vec_ema[si] = (
                        old_vx * (1 - alpha) + dx * alpha,
                        old_vy * (1 - alpha) + dy * alpha,
                    )
                else:
                    self._vel_ema[si] = raw_speed
                    self._vel_vec_ema[si] = (dx, dy)
                velocities[si] = self._vel_ema[si]
                vel_vectors[si] = self._vel_vec_ema[si]
            else:
                velocities[si] = None
                vel_vectors[si] = None

        # Update previous centroids
        for si in range(self.num_slots):
            self._prev_centroids[si] = slot_centroids[si]

        # Update body length EMA
        min_conf = self.cfg.get("min_keypoint_conf", 0.3)
        for si, det in slot_dets.items():
            bl = estimate_body_length(det, min_conf)
            if bl is not None:
                if si in self._bl_ema:
                    self._bl_ema[si] = self._bl_ema[si] * (1 - self._bl_alpha) + bl * self._bl_alpha
                else:
                    self._bl_ema[si] = bl

        # Generate pairs and classify
        events: List[ContactEvent] = []
        active_slots = [si for si in range(self.num_slots)
                        if slot_centroids[si] is not None and si in slot_dets]

        if len(active_slots) < 2:
            # Single or no detection — record all pairs as independent
            for i in range(self.num_slots):
                for j in range(i + 1, self.num_slots):
                    ev = ContactEvent(
                        frame_idx=frame_idx,
                        time_sec=round(time_sec, 4),
                        rat_a_slot=i,
                        rat_b_slot=j,
                        rat_a_track_id=None,
                        rat_b_track_id=None,
                        zone=Zone.INDEPENDENT.value,
                        quality_flag="single_detection",
                    )
                    events.append(ev)
        else:
            for i in range(len(active_slots)):
                for j in range(i + 1, len(active_slots)):
                    si_a = active_slots[i]
                    si_b = active_slots[j]
                    det_a = slot_dets[si_a]
                    det_b = slot_dets[si_b]

                    # Average body length for the pair
                    bl_a = self._bl_ema.get(si_a, self.fallback_bl)
                    bl_b = self._bl_ema.get(si_b, self.fallback_bl)
                    avg_bl = (bl_a + bl_b) / 2.0

                    event = classify_pair_contacts(
                        det_a=det_a,
                        det_b=det_b,
                        slot_a=si_a,
                        slot_b=si_b,
                        mask_a=slot_masks[si_a],
                        mask_b=slot_masks[si_b],
                        vel_a=velocities.get(si_a),
                        vel_b=velocities.get(si_b),
                        vel_vec_a=vel_vectors.get(si_a),
                        vel_vec_b=vel_vectors.get(si_b),
                        body_length=avg_bl,
                        frame_idx=frame_idx,
                        time_sec=time_sec,
                        cfg=self.cfg,
                    )
                    events.append(event)

        # Update bouts, counters, and write CSV (no event accumulation)
        for event in events:
            self._update_bout(event, frame_idx)
            self._zone_counts[event.zone] = self._zone_counts.get(event.zone, 0) + 1
            if event.quality_flag and event.quality_flag in self._quality_counts:
                self._quality_counts[event.quality_flag] += 1
            self._write_frame_row(event)

        return events

    def write_merged_placeholder(
        self,
        slot_centroids: List[Optional[Tuple[float, float]]],
        frame_idx: int,
    ) -> None:
        """Write a placeholder row for a MERGED-state frame.

        Maintains frame_idx continuity in CSV and keeps velocity + body length
        EMA updated, but does NOT classify contacts (identity is ambiguous).
        """
        time_sec = frame_idx / self.fps
        self._total_frames += 1

        # Update velocity EMA so it doesn't spike when exiting MERGED state
        alpha = self._vel_alpha
        for si in range(self.num_slots):
            curr = slot_centroids[si]
            prev = self._prev_centroids.get(si)
            if curr is not None and prev is not None:
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                raw_speed = math.sqrt(dx * dx + dy * dy)
                if si in self._vel_ema:
                    self._vel_ema[si] = self._vel_ema[si] * (1 - alpha) + raw_speed * alpha
                    old_vx, old_vy = self._vel_vec_ema[si]
                    self._vel_vec_ema[si] = (
                        old_vx * (1 - alpha) + dx * alpha,
                        old_vy * (1 - alpha) + dy * alpha,
                    )
                else:
                    self._vel_ema[si] = raw_speed
                    self._vel_vec_ema[si] = (dx, dy)

        # Update previous centroids
        for si in range(self.num_slots):
            self._prev_centroids[si] = slot_centroids[si]

        # Write one placeholder row per pair (N=2 → single pair 0,1)
        event = ContactEvent(
            frame_idx=frame_idx,
            time_sec=round(time_sec, 4),
            rat_a_slot=0,
            rat_b_slot=1,
            rat_a_track_id=None,
            rat_b_track_id=None,
            zone=Zone.PROXIMITY.value,
            contact_type=None,
            quality_flag="merged_state",
        )

        self._zone_counts["proximity"] = self._zone_counts.get("proximity", 0) + 1
        self._quality_counts["merged_state"] += 1

        # Let bout tracker close active bouts after bout_max_gap of no contact
        self._update_bout(event, frame_idx)
        self._write_frame_row(event)

    def _update_bout(self, event: ContactEvent, frame_idx: int) -> None:
        """Update bout tracking for a contact event."""
        if event.contact_type is None:
            # No contact — check if any active bout for this pair should close
            pair_key_prefix = f"{event.rat_a_slot}_{event.rat_b_slot}_"
            to_close = [k for k in self._active_bouts if k.startswith(pair_key_prefix)]
            for key in to_close:
                bout = self._active_bouts[key]
                gap = frame_idx - bout.end_frame
                if gap > self.bout_max_gap + 1:
                    self._close_bout(key)
            return

        key = f"{event.rat_a_slot}_{event.rat_b_slot}_{event.contact_type}"
        if key in self._active_bouts:
            bout = self._active_bouts[key]
            gap = frame_idx - bout.end_frame
            if gap <= self.bout_max_gap + 1:
                # Continue bout
                bout.end_frame = frame_idx
                bout.end_time_sec = event.time_sec
                bout.duration_frames += 1
                bout.total_frames = frame_idx - bout.start_frame + 1
                self._accumulate_bout_metrics(bout, event)
                event.bout_id = bout.bout_id
            else:
                # Gap too large — close and start new
                self._close_bout(key)
                self._start_bout(event, frame_idx)
        else:
            self._start_bout(event, frame_idx)

    def _start_bout(self, event: ContactEvent, frame_idx: int) -> None:
        bout = Bout(
            bout_id=self._next_bout_id,
            contact_type=event.contact_type,
            rat_a_slot=event.rat_a_slot,
            rat_b_slot=event.rat_b_slot,
            investigator_slot=event.investigator_slot,
            start_frame=frame_idx,
            end_frame=frame_idx,
            start_time_sec=event.time_sec,
            end_time_sec=event.time_sec,
            duration_frames=1,
            total_frames=1,
        )
        self._accumulate_bout_metrics(bout, event)
        event.bout_id = self._next_bout_id
        key = f"{event.rat_a_slot}_{event.rat_b_slot}_{event.contact_type}"
        self._active_bouts[key] = bout
        self._next_bout_id += 1

    def _close_bout(self, key: str) -> None:
        bout = self._active_bouts.pop(key)
        min_dur = self.bout_min_dur
        # FOL requires more consecutive frames to be valid (follow_min_frames)
        if bout.contact_type == ContactType.FOL.value:
            min_dur = max(min_dur, self.follow_min_frames)
        if bout.duration_frames >= min_dur:
            self._bouts.append(bout)

    def _accumulate_bout_metrics(self, bout: Bout, event: ContactEvent) -> None:
        if event.nose_nose_dist_px is not None:
            bout._nose_dists.append(event.nose_nose_dist_px)
        bout._mask_ious.append(event.mask_iou)
        if event.velocity_a_px is not None:
            bout._vel_a.append(event.velocity_a_px)
        if event.velocity_b_px is not None:
            bout._vel_b.append(event.velocity_b_px)
        if event.quality_flag:
            bout._quality_flags.append(event.quality_flag)

    def _write_frame_row(self, event: ContactEvent) -> None:
        row = {}
        for col in FRAME_CSV_COLUMNS:
            val = getattr(event, col, None)
            row[col] = val if val is not None else ""
        self._csv_writer.writerow(row)

    def finalize(self) -> Dict[str, Any]:
        """Close all open bouts and produce final outputs.

        Returns:
            Session summary dict (also written to JSON file).
        """
        # Close remaining active bouts
        for key in list(self._active_bouts.keys()):
            self._close_bout(key)

        # Close per-frame CSV
        self._csv_file.close()
        logger.info("Per-frame CSV: %s (%d rows)", self._csv_path, self._total_frames)

        # Write bout CSV
        bout_path = self._write_bout_csv()

        # Write session summary JSON
        summary = self._build_summary()
        summary_path = self.output_dir / "session_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Session summary: %s", summary_path)

        # Generate PDF report
        self._generate_report(summary)

        return summary

    def _write_bout_csv(self) -> Path:
        bout_path = self.output_dir / "contact_bouts.csv"
        with bout_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BOUT_CSV_COLUMNS)
            writer.writeheader()
            for bout in self._bouts:
                writer.writerow(bout.to_csv_row())
        logger.info("Bout CSV: %s (%d bouts)", bout_path, len(self._bouts))
        return bout_path

    def _build_summary(self) -> Dict[str, Any]:
        total = self._total_frames
        if total == 0:
            total = 1  # avoid division by zero

        duration_sec = self._total_frames / self.fps if self.fps > 0 else 0

        # Zone summary
        zone_summary = {}
        for z in ["contact", "proximity", "independent"]:
            count = self._zone_counts.get(z, 0)
            zone_summary[f"{z}_frames"] = count
            zone_summary[f"{z}_pct"] = round(count / total * 100, 2)

        # Contact type summary
        type_summary = {}
        for ct in ContactType:
            ct_bouts = [b for b in self._bouts if b.contact_type == ct.value]
            if not ct_bouts:
                type_summary[ct.value] = {
                    "total_bouts": 0,
                    "total_duration_sec": 0.0,
                    "total_frames": 0,
                    "pct_of_session": 0.0,
                }
                continue

            durations = [b.duration_sec for b in ct_bouts]
            total_dur = sum(durations)
            total_fr = sum(b.duration_frames for b in ct_bouts)

            entry = {
                "total_bouts": len(ct_bouts),
                "total_duration_sec": round(total_dur, 2),
                "total_frames": total_fr,
                "pct_of_session": round(total_fr / total * 100, 2),
                "mean_bout_duration_sec": round(total_dur / len(ct_bouts), 2),
                "median_bout_duration_sec": round(median(durations), 2) if durations else 0.0,
                "max_bout_duration_sec": round(max(durations), 2) if durations else 0.0,
            }

            # Asymmetric types: per-investigator breakdown
            if ct in (ContactType.N2AG, ContactType.N2B, ContactType.FOL):
                by_inv = {}
                for b in ct_bouts:
                    inv_key = f"slot_{b.investigator_slot}" if b.investigator_slot is not None else "unknown"
                    if inv_key not in by_inv:
                        by_inv[inv_key] = {"bouts": 0, "duration_sec": 0.0}
                    by_inv[inv_key]["bouts"] += 1
                    by_inv[inv_key]["duration_sec"] = round(
                        by_inv[inv_key]["duration_sec"] + b.duration_sec, 2
                    )
                entry["by_investigator"] = by_inv

            type_summary[ct.value] = entry

        # Per-pair summary
        pair_summary = {}
        pairs = set()
        for b in self._bouts:
            pairs.add((b.rat_a_slot, b.rat_b_slot))
        for a, b in sorted(pairs):
            pair_bouts = [bout for bout in self._bouts
                          if bout.rat_a_slot == a and bout.rat_b_slot == b]
            total_contact = sum(bout.duration_sec for bout in pair_bouts)
            by_type = {}
            for ct in ContactType:
                ct_bouts = [bout for bout in pair_bouts if bout.contact_type == ct.value]
                if ct_bouts:
                    by_type[ct.value] = {
                        "bouts": len(ct_bouts),
                        "duration_sec": round(sum(bout.duration_sec for bout in ct_bouts), 2),
                    }
            pair_summary[f"pair_{a}_{b}"] = {
                "total_contact_sec": round(total_contact, 2),
                "total_contact_pct": round(total_contact / duration_sec * 100, 2) if duration_sec > 0 else 0.0,
                "contact_types": by_type,
            }

        # Quality (from incremental counters — no event list needed)
        quality_counts = {
            "high_mask_overlap_frames": self._quality_counts.get("high_mask_overlap", 0),
            "missing_keypoints_frames": self._quality_counts.get("missing_keypoints", 0),
            "single_detection_frames": self._quality_counts.get("single_detection", 0),
            "merged_state_frames": self._quality_counts.get("merged_state", 0),
        }
        flagged = sum(quality_counts.values())
        quality_counts["total_flagged_pct"] = round(flagged / total * 100, 2)

        # Time bins (1-minute bins)
        bin_dur = 60.0
        num_bins = max(1, int(math.ceil(duration_sec / bin_dur)))
        bins = []
        for bi in range(num_bins):
            bin_start = bi * bin_dur
            bin_end = min((bi + 1) * bin_dur, duration_sec)
            bin_entry = {
                "bin_start_sec": round(bin_start, 1),
                "bin_end_sec": round(bin_end, 1),
            }
            # Count contact seconds per type in this bin
            for ct in ContactType:
                ct_sec = 0.0
                for bout in self._bouts:
                    if bout.contact_type != ct.value:
                        continue
                    # Overlap between bout and bin
                    overlap_start = max(bout.start_time_sec, bin_start)
                    overlap_end = min(bout.end_time_sec, bin_end)
                    if overlap_end > overlap_start:
                        ct_sec += overlap_end - overlap_start
                bin_entry[f"{ct.value}_sec"] = round(ct_sec, 2)
            bin_entry["contact_sec"] = round(
                sum(bin_entry.get(f"{ct.value}_sec", 0) for ct in ContactType), 2
            )
            bins.append(bin_entry)

        return {
            "metadata": {
                "video_path": self.video_path,
                "video_duration_sec": round(duration_sec, 2),
                "total_frames": self._total_frames,
                "fps": self.fps,
                "num_rats": self.num_slots,
                "analysis_date": datetime.now().isoformat(timespec="seconds"),
            },
            "parameters": {
                k: v for k, v in self.cfg.items()
            },
            "zone_summary": zone_summary,
            "contact_type_summary": type_summary,
            "per_pair_summary": pair_summary,
            "quality": quality_counts,
            "time_bins": {
                "bin_duration_sec": bin_dur,
                "bins": bins,
            },
        }

    def _generate_report(self, summary: Dict[str, Any]) -> None:
        """Generate PDF report with matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not available — skipping PDF report")
            return

        pdf_path = self.output_dir / "report.pdf"

        # Color scheme for contact types
        ct_colors = {
            "N2N": "#1f77b4",
            "N2AG": "#ff7f0e",
            "N2B": "#2ca02c",
            "T2T": "#8c564b",
            "SBS": "#9467bd",
            "FOL": "#d62728",
        }

        with PdfPages(str(pdf_path)) as pdf:
            # ── Page 1: Timeline Ethogram ──
            fig, ax = plt.subplots(figsize=(14, 3))
            for bout in self._bouts:
                color = ct_colors.get(bout.contact_type, "#888888")
                ax.barh(
                    0, bout.duration_sec, left=bout.start_time_sec,
                    height=0.6, color=color, edgecolor="none",
                )
            ax.set_yticks([0])
            ax.set_yticklabels([f"Pair 0-1"])
            ax.set_xlabel("Time (seconds)")
            ax.set_title("Contact Ethogram")
            patches = [mpatches.Patch(color=c, label=t) for t, c in ct_colors.items()]
            ax.legend(handles=patches, loc="upper right", ncol=5, fontsize=8)
            ax.set_xlim(0, summary["metadata"]["video_duration_sec"])
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ── Page 2: Duration Histograms ──
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            axes_flat = axes.flatten()
            all_durations = []
            for idx, ct in enumerate(ContactType):
                ax = axes_flat[idx]
                durations = [b.duration_sec for b in self._bouts if b.contact_type == ct.value]
                all_durations.extend(durations)
                if durations:
                    ax.hist(durations, bins=min(20, max(5, len(durations))),
                            color=ct_colors[ct.value], edgecolor="white")
                ax.set_title(ct.value)
                ax.set_xlabel("Duration (s)")
                ax.set_ylabel("Count")
            # Last subplot: all contacts
            ax = axes_flat[5]
            if all_durations:
                ax.hist(all_durations, bins=min(20, max(5, len(all_durations))),
                        color="#888888", edgecolor="white")
            ax.set_title("ALL")
            ax.set_xlabel("Duration (s)")
            ax.set_ylabel("Count")
            plt.suptitle("Bout Duration Distributions", fontsize=13)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ── Page 3: Pie Chart ──
            fig, ax = plt.subplots(figsize=(8, 6))
            type_sums = summary["contact_type_summary"]
            labels = []
            sizes = []
            colors = []
            for ct in ContactType:
                dur = type_sums[ct.value].get("total_duration_sec", 0)
                if dur > 0:
                    labels.append(ct.value)
                    sizes.append(dur)
                    colors.append(ct_colors[ct.value])
            if sizes:
                ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                       startangle=90)
                ax.set_title("Contact Time by Type")
            else:
                ax.text(0.5, 0.5, "No contacts detected", ha="center", va="center",
                        fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ── Page 4: Time-Binned Contact Rates ──
            time_bins = summary.get("time_bins", {})
            bins_data = time_bins.get("bins", [])
            if bins_data:
                fig, ax = plt.subplots(figsize=(12, 5))
                x = [b["bin_start_sec"] / 60.0 for b in bins_data]
                bottom = [0.0] * len(x)
                for ct in ContactType:
                    vals = [b.get(f"{ct.value}_sec", 0) for b in bins_data]
                    ax.bar(x, vals, bottom=bottom, width=0.9,
                           color=ct_colors[ct.value], label=ct.value)
                    bottom = [b + v for b, v in zip(bottom, vals)]
                ax.set_xlabel("Time (minutes)")
                ax.set_ylabel("Contact seconds per minute")
                ax.set_title("Contact Rate Over Time")
                ax.legend(loc="upper right", ncol=5, fontsize=8)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # ── Page 5: Summary Table ──
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis("off")
            table_data = []
            headers = ["Pair", "Total (s)"] + [ct.value for ct in ContactType]
            for pair_key, pair_info in summary.get("per_pair_summary", {}).items():
                row = [pair_key.replace("pair_", "")]
                row.append(f"{pair_info['total_contact_sec']:.1f}")
                for ct in ContactType:
                    ct_info = pair_info.get("contact_types", {}).get(ct.value, {})
                    dur = ct_info.get("duration_sec", 0)
                    bouts = ct_info.get("bouts", 0)
                    row.append(f"{dur:.1f}s ({bouts}b)")
                table_data.append(row)
            if table_data:
                table = ax.table(cellText=table_data, colLabels=headers,
                                 loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
            ax.set_title("Per-Pair Contact Summary", fontsize=13, pad=20)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ── Load per-frame CSV for data-driven pages ──
            try:
                import pandas as pd
                df = pd.read_csv(self._csv_path)
            except Exception:
                df = None

            if df is not None and not df.empty and "time_sec" in df.columns:
                # ── Page 6: Distance Over Time ──
                fig, ax = plt.subplots(figsize=(14, 4))
                zone_colors = {"contact": "#2ca02c", "proximity": "#ffcc00", "independent": "#dddddd"}
                # Paint zone background as contiguous spans (fast)
                if "zone" in df.columns:
                    zones = df["zone"].values
                    times = df["time_sec"].values
                    i = 0
                    while i < len(zones):
                        z = zones[i]
                        j = i
                        while j < len(zones) and zones[j] == z:
                            j += 1
                        t0 = times[i]
                        t1 = times[min(j, len(times) - 1)]
                        ax.axvspan(t0, t1, color=zone_colors.get(z, "#dddddd"),
                                   alpha=0.3, linewidth=0)
                        i = j
                if "nose_nose_dist_bl" in df.columns:
                    valid = df.dropna(subset=["nose_nose_dist_bl"])
                    ax.plot(valid["time_sec"], valid["nose_nose_dist_bl"],
                            color="#1f77b4", linewidth=0.5, alpha=0.8, label="Nose-Nose (BL)")
                if "centroid_dist_bl" in df.columns:
                    valid = df.dropna(subset=["centroid_dist_bl"])
                    ax.plot(valid["time_sec"], valid["centroid_dist_bl"],
                            color="#ff7f0e", linewidth=0.5, alpha=0.8, label="Centroid (BL)")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Distance (body lengths)")
                ax.set_title("Inter-Rat Distance Over Time")
                line_handles = ax.get_legend_handles_labels()[0]
                zone_patches = [mpatches.Patch(color=c, alpha=0.3, label=z.capitalize())
                                for z, c in zone_colors.items()]
                ax.legend(handles=line_handles + zone_patches,
                          loc="upper right", fontsize=7, ncol=2)
                ax.set_xlim(0, df["time_sec"].max())
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                # ── Page 7: Investigator Role Breakdown ──
                asym_types = ["N2AG", "N2B", "FOL"]
                inv_data = {}
                for ct_val in asym_types:
                    ct_bouts = [b for b in self._bouts if b.contact_type == ct_val]
                    slot_counts = {}
                    for b in ct_bouts:
                        s = b.investigator_slot if b.investigator_slot is not None else -1
                        slot_counts[s] = slot_counts.get(s, 0) + 1
                    inv_data[ct_val] = slot_counts

                if any(inv_data.values()):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    y_pos = list(range(len(asym_types)))
                    slot_ids = sorted({s for sc in inv_data.values() for s in sc.keys()})
                    slot_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                    for idx, slot_id in enumerate(slot_ids):
                        vals = [inv_data[ct].get(slot_id, 0) for ct in asym_types]
                        label = f"Rat {slot_id}" if slot_id >= 0 else "Unknown"
                        color = slot_colors[idx % len(slot_colors)]
                        left = [0] * len(asym_types)
                        if idx > 0:
                            for prev in range(idx):
                                prev_slot = slot_ids[prev]
                                for j, ct in enumerate(asym_types):
                                    left[j] += inv_data[ct].get(prev_slot, 0)
                        ax.barh(y_pos, vals, left=left, height=0.5,
                                color=color, label=label, edgecolor="white")
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(asym_types)
                    ax.set_xlabel("Number of Bouts")
                    ax.set_title("Investigator Role Breakdown (who initiates)")
                    ax.legend(loc="lower right", fontsize=9)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                # ── Page 8: Cumulative Contact Time ──
                fig, ax = plt.subplots(figsize=(14, 5))
                for ct in ContactType:
                    ct_frames = df[df["contact_type"] == ct.value]
                    if ct_frames.empty:
                        continue
                    times = ct_frames["time_sec"].values
                    dt = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
                    cumulative = np.arange(1, len(times) + 1) * dt
                    ax.plot(times, cumulative, color=ct_colors[ct.value],
                            linewidth=1.5, label=ct.value)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Cumulative contact time (seconds)")
                ax.set_title("Cumulative Contact Time by Type")
                ax.legend(loc="upper left", fontsize=9)
                ax.set_xlim(0, df["time_sec"].max())
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        logger.info("PDF report: %s", pdf_path)
