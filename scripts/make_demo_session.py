#!/usr/bin/env python3
"""
Generate a demo contact session without running YOLO or SAM2.

Synthesises a choreographed 120 s encounter between two rats — keypoints and
masks — and drives it through the real ContactTracker, the real temporal
post-processing and the real Excel consolidation. The output is a genuine
run directory, so it doubles as an end-to-end test of the contact chain and as
a sample file for the HTML report viewer.

Because it uses the production code path (including resolve_overlaps on the
masks), whatever this script produces is what the real pipeline would produce
for the same geometry.

Usage:
    python scripts/make_demo_session.py
    python scripts/make_demo_session.py --duration 60 --out outputs/runs/my_demo
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.geometry import resolve_overlaps
from src.common.utils import Detection, Keypoint

# contacts_v2 only exists on the develop branch. Fall back to v1 when absent.
try:
    from src.common.contacts_v2 import ContactTrackerV2 as _TrackerV2
except Exception:
    _TrackerV2 = None
from src.common.contacts import ContactTracker as _TrackerV1

logger = logging.getLogger("demo")

# ── Scene constants ────────────────────────────────────────────────────────

W, H = 1920, 1080
FPS = 30.0
BL = 120.0                 # body length in px, matching fallback_body_length_px
SEMI_MAJOR = 0.55 * BL     # mask ellipse along the body axis
SEMI_MINOR = 0.20 * BL     # mask ellipse across it
BLEND_SEC = 1.2            # approach / retreat time between segments

# Offsets along the body axis as a fraction of body length, from the body
# centre. Ears also carry a lateral offset.
#
# contacts_v2 treats `tail_start` (the body-tail junction) as the rear /
# anogenital point and `tail_base` as a point further down the tail.
KP_LAYOUT_FULL = [
    ("tail_tip",   -0.90,  0.00),
    ("tail_base",  -0.50,  0.00),
    ("tail_start", -0.35,  0.00),   # rear / anogenital, per contacts_v2
    ("mid_body",    0.00,  0.00),
    ("nose",        0.50,  0.00),
    ("right_ear",   0.34,  0.10),
    ("left_ear",    0.34, -0.10),
]

# What every v2 config actually ships: five points, and no tail_start.
KP_NAMES_SHIPPED = ["nose", "left_ear", "right_ear", "mid_body", "tail_base"]

# Distance from the body centre to whichever keypoint is used as the rear.
REAR_OFFSET = {"tail_start": 0.35, "tail_base": 0.50}

# Which fraction the running configuration uses. Set by generate().
REAR_FRAC = 0.50

# The choreography. Each entry is (start_sec, end_sec, kind), scaled to the
# requested duration. Kinds are defined in POSE_FNS below.
SCRIPT = [
    (0,   12,  "apart"),
    (12,  20,  "n2n"),
    (20,  30,  "n2ag"),
    (30,  38,  "apart"),
    (38,  54,  "fol"),
    (54,  62,  "apart"),
    (62,  78,  "sbs"),
    (78,  86,  "apart"),
    (86,  96,  "t2t"),
    (96,  104, "apart"),
    (104, 114, "n2b"),
    (114, 120, "apart"),
]


# ── Choreography ───────────────────────────────────────────────────────────

def _anchor(t: float) -> Tuple[float, float]:
    """Slowly drifting centre for an interaction, so contacts are not static."""
    return (960.0 + 90.0 * math.sin(t * 0.20), 540.0 + 60.0 * math.cos(t * 0.17))


def _theta(t: float) -> float:
    """Slowly rotating orientation for the interacting pair."""
    return 0.30 * math.sin(t * 0.15)


def _rot(v: Tuple[float, float], a: float) -> Tuple[float, float]:
    ca, sa = math.cos(a), math.sin(a)
    return (v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca)


def _apart(t: float):
    """Two rats wandering independently, well over a body length apart."""
    def p0(u):
        return (470.0 + 200.0 * math.sin(u * 0.36), 380.0 + 140.0 * math.cos(u * 0.27))

    def p1(u):
        return (1430.0 + 175.0 * math.cos(u * 0.31), 720.0 + 155.0 * math.sin(u * 0.23))

    dt = 0.12
    c0, c0n = p0(t), p0(t + dt)
    c1, c1n = p1(t), p1(t + dt)
    h0 = math.atan2(c0n[1] - c0[1], c0n[0] - c0[0])
    h1 = math.atan2(c1n[1] - c1[1], c1n[0] - c1[0])
    return c0, h0, c1, h1


def _n2n(t: float):
    """Head to head. Nose gap 0.18 BL, inside the 0.3 BL contact radius."""
    a, th = _anchor(t), _theta(t)
    d = 1.18 * BL
    off = _rot((d / 2.0, 0.0), th)
    return (a[0] - off[0], a[1] - off[1]), th, (a[0] + off[0], a[1] + off[1]), th + math.pi


def _n2ag(t: float):
    """Rat 0's nose at rat 1's rear. Same heading, near stationary.

    Centre separation is chosen so the nose-to-rear gap is 0.15 BL, inside the
    contact radius, whichever keypoint the taxonomy uses as the rear.
    """
    a, th = _anchor(t), _theta(t)
    d = (0.50 + REAR_FRAC + 0.15) * BL
    off = _rot((d / 2.0, 0.0), th)
    return (a[0] - off[0], a[1] - off[1]), th, (a[0] + off[0], a[1] + off[1]), th


def _fol(t: float):
    """Rat 0 trails rat 1 at 1.40 BL — outside N2AG range, inside follow range,
    with both animals moving fast enough and in the same direction."""
    # A wide arc keeps the pair in frame at ~4.5 px/frame (135 px/s).
    speed, radius = 135.0, 380.0
    w = speed / radius
    cx = 960.0 + radius * math.cos(w * t)
    cy = 540.0 + radius * math.sin(w * t)
    th = math.atan2(math.cos(w * t), -math.sin(w * t))  # tangent to the arc
    d = (0.50 + REAR_FRAC + 0.38) * BL
    off = _rot((d / 2.0, 0.0), th)
    return (cx - off[0], cy - off[1]), th, (cx + off[0], cy + off[1]), th


def _sbs(t: float):
    """Flank to flank, parallel and near stationary, with overlapping masks.

    Lateral gap 40 px: wide enough that nose-nose (36 px contact radius) does
    not fire first, narrow enough that the mask ellipses genuinely overlap.
    """
    a, th = _anchor(t), _theta(t)
    perp = _rot((0.0, 1.0), th)
    g = 20.0
    return ((a[0] - perp[0] * g, a[1] - perp[1] * g), th,
            (a[0] + perp[0] * g, a[1] + perp[1] * g), th)


def _t2t(t: float):
    """Rear to rear — both rear points meet in the middle. (v1 only; contacts_v2
    removed T2T, where this segment simply reads as no contact.)"""
    a, th = _anchor(t), _theta(t)
    d = (2 * REAR_FRAC + 0.15) * BL
    off = _rot((d / 2.0, 0.0), th)
    return ((a[0] - off[0], a[1] - off[1]), th + math.pi,
            (a[0] + off[0], a[1] + off[1]), th)


def _n2b(t: float):
    """Rat 0 approaches rat 1's flank perpendicularly, nose at its mid body."""
    a, th = _anchor(t), _theta(t)
    perp = _rot((0.0, 1.0), th)
    c0 = (a[0] - perp[0] * 0.5 * BL, a[1] - perp[1] * 0.5 * BL)
    return c0, math.atan2(perp[1], perp[0]), a, th


POSE_FNS: Dict[str, Callable] = {
    "apart": _apart, "n2n": _n2n, "n2ag": _n2ag,
    "fol": _fol, "sbs": _sbs, "t2t": _t2t, "n2b": _n2b,
}


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _blend(pa, pb, w: float):
    """Blend two poses. Headings blend as direction vectors so the shorter arc
    is taken and there is no wrap-around spin."""
    out = []
    for i in (0, 2):  # centres
        out.append((pa[i][0] * (1 - w) + pb[i][0] * w,
                    pa[i][1] * (1 - w) + pb[i][1] * w))
    hs = []
    for i in (1, 3):  # headings
        vx = math.cos(pa[i]) * (1 - w) + math.cos(pb[i]) * w
        vy = math.sin(pa[i]) * (1 - w) + math.sin(pb[i]) * w
        hs.append(math.atan2(vy, vx) if (vx or vy) else pa[i])
    return out[0], hs[0], out[1], hs[1]


def build_script(duration: float) -> List[Tuple[float, float, str]]:
    """Scale the choreography to the requested duration."""
    total = SCRIPT[-1][1]
    k = duration / total
    return [(s * k, e * k, kind) for s, e, kind in SCRIPT]


def pose_at(t: float, script) -> Tuple[Tuple[float, float], float, Tuple[float, float], float]:
    """Pose of both rats at time t, eased across segment boundaries."""
    idx = 0
    for i, (s, e, _) in enumerate(script):
        if s <= t < e:
            idx = i
            break
    else:
        idx = len(script) - 1

    s, e, kind = script[idx]
    own = POSE_FNS[kind](t)

    if t - s < BLEND_SEC and idx > 0:
        prev = POSE_FNS[script[idx - 1][2]](t)
        return _blend(prev, own, _smoothstep((t - s) / BLEND_SEC))
    if e - t < BLEND_SEC and idx < len(script) - 1:
        nxt = POSE_FNS[script[idx + 1][2]](t)
        return _blend(nxt, own, _smoothstep((e - t) / BLEND_SEC))
    return own


# ── Rat rendering ──────────────────────────────────────────────────────────

def make_keypoints(c, h, rng: random.Random, layout) -> Tuple[List[Keypoint], Tuple[float, float, float, float]]:
    """Body keypoints plus the enclosing bounding box, with detector-like noise."""
    ca, sa = math.cos(h), math.sin(h)
    kps: List[Keypoint] = []
    xs, ys = [], []

    # A whole-animal confidence dip stands in for a partly occluded detection.
    dip = rng.random() < 0.06

    for name, along, across in layout:
        ax, ay = along * BL, across * BL
        x = c[0] + ax * ca - ay * sa + rng.gauss(0, 1.1)
        y = c[1] + ax * sa + ay * ca + rng.gauss(0, 1.1)
        conf = rng.uniform(0.82, 0.97)
        if dip and name in ("nose", "tail_base", "mid_body"):
            conf = rng.uniform(0.05, 0.28)          # below keypoint_min_conf
        elif rng.random() < 0.02:
            conf = rng.uniform(0.10, 0.29)
        kps.append(Keypoint(x=x, y=y, conf=conf, name=name))
        xs.append(x)
        ys.append(y)

    pad = 0.12 * BL
    box = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    return kps, box


def make_mask(c, h) -> np.ndarray:
    """Body mask as an oriented ellipse, rasterised only inside its bounding box."""
    mask = np.zeros((H, W), dtype=bool)

    r = int(math.ceil(SEMI_MAJOR)) + 3
    x0, x1 = max(0, int(c[0]) - r), min(W, int(c[0]) + r + 1)
    y0, y1 = max(0, int(c[1]) - r), min(H, int(c[1]) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return mask

    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = xs - c[0]
    dy = ys - c[1]
    ca, sa = math.cos(-h), math.sin(-h)
    u = dx * ca - dy * sa          # along the body
    v = dx * sa + dy * ca          # across it
    mask[y0:y1, x0:x1] = (u / SEMI_MAJOR) ** 2 + (v / SEMI_MINOR) ** 2 <= 1.0
    return mask


def centroid_of(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# ── Main ───────────────────────────────────────────────────────────────────

def generate(out_dir: Path, duration: float, seed: int, faithful: bool,
             taxonomy: str, shipped_keypoints: bool) -> Path:
    """Run the choreography through the real contact tracker.

    Args:
        out_dir: Run directory to create.
        duration: Session length in seconds.
        seed: Seed for the detector-like noise.
        faithful: Make the two masks disjoint, as both production pipelines do
            (CUTIE emits an index mask; the centroid pipeline calls
            resolve_overlaps). Mask IoU is then always zero.
        taxonomy: "v2" for contacts_v2, "v1" for contacts.py.
        shipped_keypoints: Use the five keypoints every v2 config actually
            declares, which omit `tail_start`.
    """
    global REAR_FRAC

    rng = random.Random(seed)

    use_v2 = taxonomy == "v2"
    if use_v2 and _TrackerV2 is None:
        raise SystemExit(
            "contacts_v2 is not on this branch. Run from a checkout of "
            "origin/develop, or pass --taxonomy v1."
        )

    if shipped_keypoints:
        layout = [kp for kp in KP_LAYOUT_FULL if kp[0] in KP_NAMES_SHIPPED]
        rear_name = "tail_base"
    else:
        layout = list(KP_LAYOUT_FULL)
        rear_name = "tail_start" if use_v2 else "tail_base"
    REAR_FRAC = REAR_OFFSET[rear_name]

    script = build_script(duration)
    n_frames = int(round(duration * FPS))

    contacts_dir = out_dir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)

    if use_v2:
        # Mirrors the contacts block of configs/local_cutie_composite.yaml.
        config = {"contacts": {
            "enabled": True,
            "contact_zone_bl_enter": 0.30, "contact_zone_bl_exit": 0.45,
            "proximity_zone_bl": 1.0,
            "sbs_mask_iou_enter": 0.05, "sbs_mask_iou_exit": 0.02,
            "follow_radius_bl": 0.4, "follow_radius_bl_exit": 0.6,
            "follow_min_speed_bls": 0.15, "follow_alignment_cos": 0.6,
            "activation_threshold": 0.5, "activation_threshold_rare": 0.35,
            "secondary_threshold": 0.4,
            "bout_max_gap_frames": 3,
            "bout_min_frames_n2n": 6, "bout_min_frames_n2ag": 6,
            "bout_min_frames_t2t": 6, "bout_min_frames_fol": 12,
            "bout_min_frames_sbs": 6, "bout_min_frames_n2b": 6,
        }}
        tracker = _TrackerV2(output_dir=contacts_dir, fps=FPS, num_slots=2,
                             video_path=f"demo://synthetic_{int(duration)}s.avi",
                             config=config)
    else:
        config = {"contacts": {
            "min_keypoint_conf": 0.3, "contact_zone_bl": 0.3,
            "proximity_zone_bl": 1.0, "fallback_body_length_px": 120,
            "sbs_mask_iou_min": 0.02, "sbs_max_velocity_bl": 0.04,
            "sbs_parallel_cos_min": 0.7, "follow_radius_bl": 0.5,
            "follow_min_speed_bl": 0.025, "follow_alignment_cos": 0.7,
            "follow_min_frames": 30, "bout_max_gap_frames": 3,
            "bout_min_duration_frames": 9, "mask_overlap_warning": 0.5,
            "det_slot_match_radius": 200.0,
        }}
        tracker = _TrackerV1(output_dir=contacts_dir, fps=FPS, num_slots=2,
                             video_path=f"demo://synthetic_{int(duration)}s.avi",
                             config=config)

    logger.info("Taxonomy: %s | keypoints: %d (%s) | rear point: %s",
                taxonomy, len(layout),
                "as shipped" if shipped_keypoints else "full 7-point set", rear_name)
    logger.info("Masks: %s", "disjoint (as in production)" if faithful else "overlap kept")
    logger.info("Synthesising %d frames (%.0f s at %.0f fps)", n_frames, duration, FPS)

    for i in range(n_frames):
        t = i / FPS
        c0, h0, c1, h1 = pose_at(t, script)

        dets: List[Optional[Detection]] = []
        masks: List[Optional[np.ndarray]] = []
        cents: List[Optional[Tuple[float, float]]] = []

        for slot, (c, h) in enumerate(((c0, h0), (c1, h1))):
            kps, box = make_keypoints(c, h, rng, layout)
            det = Detection(
                x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                conf=rng.uniform(0.75, 0.95),
                class_name="rat", keypoints=kps, track_id=slot + 1,
            )
            if rng.random() < 0.03:
                det.is_carried_over = True
            m = make_mask(c, h)
            dets.append(det)
            masks.append(m)
            cents.append(centroid_of(m))

        if faithful:
            resolve_overlaps(masks, cents)

        tracker.update(detections=[d for d in dets if d is not None],
                       slot_masks=masks, slot_centroids=cents, frame_idx=i)

        if (i + 1) % 300 == 0:
            logger.info("  %d / %d frames", i + 1, n_frames)

    logger.info("Finalising — bouts, session summary, PDF report")
    tracker.finalize()

    from scripts.postprocess_contacts_simple import run_postprocess
    run_postprocess(contacts_dir, fps=FPS, consolidate=True)

    return contacts_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default=None,
                    help="Output run directory (default: outputs/runs/<timestamp>_demo)")
    ap.add_argument("--duration", type=float, default=120.0, help="Session length in seconds")
    ap.add_argument("--seed", type=int, default=11, help="Random seed for detector noise")
    ap.add_argument("--keep-mask-overlap", action="store_true",
                    help="Keep the mask overlap instead of making the masks disjoint. "
                         "Production makes them disjoint, so SBS cannot fire there; use "
                         "this to see what SBS would produce once that is fixed.")
    ap.add_argument("--taxonomy", choices=["v1", "v2"], default="v2",
                    help="Contact module to drive: v2 = contacts_v2 (cutie_composite), "
                         "v1 = contacts.py (centroid). Default v2.")
    ap.add_argument("--shipped-keypoints", action="store_true",
                    help="Use the five keypoints the v2 configs actually declare, which "
                         "omit tail_start. Reproduces the shipped behaviour, where N2AG "
                         "cannot be computed.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.out:
        out_dir = Path(args.out)
    else:
        from datetime import datetime
        out_dir = Path("outputs/runs") / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_demo")

    contacts_dir = generate(out_dir, args.duration, args.seed,
                            faithful=not args.keep_mask_overlap,
                            taxonomy=args.taxonomy,
                            shipped_keypoints=args.shipped_keypoints)

    xlsx = contacts_dir / "results.xlsx"
    print("\n" + "=" * 62)
    print("  DEMO SESSION READY")
    print("  Workbook:  " + str(xlsx))
    print("  PDF:       " + str(contacts_dir / "report.pdf"))
    print("  Event log: " + str(contacts_dir / "event_log.txt"))
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
