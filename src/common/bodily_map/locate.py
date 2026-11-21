"""Data layer for the bodily map of social contact (offline post-processor).

Reads ``tracks/tracks.jsonl.gz`` (keypoints) and ``contacts/results.xlsx``
(PerFrame sheet) of a finished run and places every bodily mouse-to-mouse
contact on the TOUCHED mouse's body in a pose-normalised coordinate.

Dependencies: standard library, numpy and openpyxl only.

Canonical coordinate
--------------------
Midline polyline of the receiver: nose -> mid_body -> tail_start -> tail_base
-> tail_tip (low-confidence optional points are dropped).

* ``t``: arc-length-normalised position of the projection foot, nose=0,
  tail_tip=1 (monotone along the body, continuous heatmap axis).
* ``s``: signed perpendicular offset of the query point from the midline,
  divided by body length. **s > 0 = the mouse's RIGHT side, s < 0 = LEFT**
  (image coordinates, y down; calibrated so right_ear gives s > 0).
* ``u``: secondary axis used only for region bins, the arc position along
  nose -> mid_body -> tail_start normalised by that length (u=1 at tail_start,
  u>1 on the tail). It is the same axis contacts_v2 uses for ``n2b_zone``
  (head u<0.20, torso u<0.75, rear otherwise).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MIN_CONF = 0.3

# Region cuts (on u, the nose->mid_body->tail_start axis, and |s|).
U_NOSE = 0.06        # u < 0.06: nose tip
U_HEAD = 0.20        # u < 0.20: head (same cut as contacts_v2 n2b_zone head)
U_TORSO = 0.75       # u < 0.75: torso (same cut as n2b_zone torso)
U_ANOGENITAL = 0.90  # 0.75..0.90: rump; 0.90..U_TAIL: anogenital
U_TAIL = 1.15        # u beyond this: tail
S_FLANK = 0.08       # |s| (in body lengths) beyond this: flank, else mid_back
SBS_FLANK_S = 0.12   # nominal lateral offset of the flank point for SBS

SLOT_COLORS_BGR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def slot_color_rgb(slot: int) -> Tuple[float, float, float]:
    b, g, r = SLOT_COLORS_BGR[slot % len(SLOT_COLORS_BGR)]
    return (r / 255.0, g / 255.0, b / 255.0)


# ----------------------------------------------------------------- loading
def find_run_files(run_dir) -> Tuple[Path, Path]:
    """Locate tracks.jsonl.gz and contacts/results.xlsx anywhere under run_dir."""
    run_dir = Path(run_dir)
    tr = sorted(run_dir.rglob("tracks.jsonl.gz"))
    xl = sorted(p for p in run_dir.rglob("results.xlsx") if p.parent.name == "contacts")
    if not tr or not xl:
        raise FileNotFoundError(f"tracks.jsonl.gz / contacts/results.xlsx not found under {run_dir}")
    return tr[0], xl[0]


def load_tracks(path) -> Tuple[dict, Dict[int, Dict[int, dict]]]:
    """Return (meta, {frame: {slot: mouse_dict}}). Masks (rle) are dropped."""
    frames: Dict[int, Dict[int, dict]] = {}
    meta: dict = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj:
                meta = obj["meta"]
                continue
            frames[int(obj["f"])] = {
                int(m["id"]) - 1: {"c": m.get("c"), "kp": m.get("kp")} for m in obj.get("rats", [])
            }
    return meta, frames


def load_perframe(path) -> List[dict]:
    """Read the PerFrame sheet as a list of dicts keyed by header name."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["PerFrame"]
    it = ws.iter_rows(values_only=True)
    header = list(next(it))
    rows = [dict(zip(header, r)) for r in it]
    wb.close()
    return rows


# --------------------------------------------------------------- projection
def _kp_dict(kp: Sequence, names: Sequence[str], min_conf: float = MIN_CONF) -> Dict[str, np.ndarray]:
    out = {}
    for n, p in zip(names, kp or []):
        if p is not None and len(p) >= 3 and p[2] >= min_conf:
            out[n] = np.array([float(p[0]), float(p[1])])
    return out


def project_to_midline(kp: Dict[str, np.ndarray], q, body_length: Optional[float] = None) -> Optional[dict]:
    """Project pixel ``q`` onto the receiver's midline.

    ``kp`` maps keypoint name -> xy (already confidence-filtered). Returns None
    if nose or tail_start is missing. Result keys: t, s, u, region, foot.
    Sign convention: s > 0 = mouse's right, s < 0 = left (right_ear -> s > 0).
    """
    if "nose" not in kp or "tail_start" not in kp:
        return None
    order = ["nose", "mid_body", "tail_start", "tail_base", "tail_tip"]
    pts = [(n, kp[n]) for n in order if n in kp]
    P = np.array([p for _, p in pts])
    seg = P[1:] - P[:-1]
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    if total < 1e-6:
        return None
    qv = np.asarray(q, float)
    best = None
    for i in range(len(seg)):
        if seglen[i] < 1e-9:
            continue
        d = seg[i] / seglen[i]
        a = float(np.clip(np.dot(qv - P[i], d), 0.0, seglen[i]))
        foot = P[i] + d * a
        dist = float(np.hypot(*(qv - foot)))
        if best is None or dist < best[0]:
            best = (dist, i, a, foot, d)
    dist, i, a, foot, d = best
    arc = cum[i] + a
    # Right-hand normal. d points nose->tail, so the facing direction is -d;
    # in image coords (y down) the mouse's right of facing f is (-f_y, f_x) = (d_y, -d_x).
    right = np.array([d[1], -d[0]])
    bl = body_length
    ts_idx = [n for n, _ in pts].index("tail_start")
    axis_len = float(cum[ts_idx])
    if not bl or bl <= 0:
        bl = axis_len
    s = float(np.dot(qv - foot, right) / bl)
    u = float(arc / axis_len)
    return {"t": float(arc / total), "s": s, "u": u, "foot": foot, "region": region_from(u, s)}


def region_from(u: float, s: float) -> str:
    """Discrete region label from axis position u and signed lateral offset s."""
    if u >= U_TAIL:
        return "tail"
    if u >= U_ANOGENITAL:
        return "anogenital"
    if u < U_NOSE:
        return "nose"
    if u < U_HEAD:
        return "head"
    if u < U_TORSO:
        if abs(s) >= S_FLANK:
            return "flank_R" if s > 0 else "flank_L"
        return "mid_back"
    return "rump"


def region_with_zone(computed: str, s: float, zone: Optional[str]) -> str:
    """Prefer the pipeline's n2b_zone (head/torso/rear) for N2B rows."""
    if zone == "head":
        return computed if computed == "nose" else "head"
    if zone == "torso":
        return "flank_R" if s >= S_FLANK else ("flank_L" if s <= -S_FLANK else "mid_back")
    if zone == "rear":
        return computed if computed in ("anogenital", "tail") else "rump"
    return computed


def flank_facing(kp: Dict[str, np.ndarray], other_centroid, body_length: Optional[float] = None) -> Optional[dict]:
    """SBS placement: lateral flank at mid-body on the side facing the other mouse.

    Side is picked from the sign of the other centroid relative to this mouse's
    body axis (same right-hand convention as project_to_midline).
    """
    if "nose" not in kp or "tail_start" not in kp:
        return None
    mid = kp.get("mid_body", (kp["nose"] + kp["tail_start"]) / 2.0)
    r = project_to_midline(kp, other_centroid, body_length)
    base = project_to_midline(kp, mid, body_length)
    if r is None or base is None:
        return None
    sign = 1.0 if r["s"] >= 0 else -1.0
    return {"t": base["t"], "s": sign * SBS_FLANK_S, "u": base["u"],
            "region": "flank_R" if sign > 0 else "flank_L"}


# ------------------------------------------------------------ contact table
def _xy(row, a, b):
    x, y = row.get(a), row.get(b)
    if x is None or y is None:
        return None
    return (float(x), float(y))


def locate_contacts(tracks_meta: dict, frames: Dict[int, Dict[int, dict]], perframe: List[dict]):
    """Return (records, stats). One record per (contact frame, touched mouse)."""
    names = tracks_meta["keypoint_names"]
    fps = float(tracks_meta.get("fps", 30.0))
    stats = {"rows": len(perframe), "not_contact": 0, "fol_skipped": 0, "no_investigator": 0,
             "receiver_absent": 0, "unresolved_keypoints": 0, "no_query_point": 0,
             "located_rows": 0, "records": 0}
    records: List[dict] = []

    def add(row, recv_slot, other_slot, ctype, res, zone=None):
        region = region_with_zone(res["region"], res["s"], zone) if zone else res["region"]
        records.append({
            "frame_idx": int(row["frame_idx"]), "time_sec": float(row.get("time_sec") or 0.0),
            "focal_mouse": f"R{recv_slot + 1}", "other_mouse": f"R{other_slot + 1}",
            "direction": "passive", "contact_type": ctype, "region": region,
            "t": res["t"], "s": res["s"], "dur_sec": 1.0 / fps, "u": res["u"]})

    for row in perframe:
        ctype = row.get("real_type") or row.get("contact_type")
        if ctype in (None, "", "NC", "none"):
            stats["not_contact"] += 1
            continue
        if ctype == "FOL":
            stats["fol_skipped"] += 1
            continue
        try:
            si, sj = (int(x) for x in str(row["pair_key"]).split("_"))
        except Exception:
            stats["not_contact"] += 1
            continue
        fr = frames.get(int(row["frame_idx"]), {})
        before = len(records)
        mice = {}
        for slot, tag in ((si, "i"), (sj, "j")):
            m = fr.get(slot)
            if m is None or not m.get("kp"):
                mice[tag] = None
                continue
            cen = _xy(row, f"centroid_{tag}_x", f"centroid_{tag}_y")
            if cen is None and m.get("c"):
                cen = (float(m["c"][0]), float(m["c"][1]))
            mice[tag] = {"kp": _kp_dict(m["kp"], names), "bl": row.get(f"body_length_{tag}_px"),
                         "nose": _xy(row, f"nose_{tag}_x", f"nose_{tag}_y"), "cen": cen}
        slots = {"i": si, "j": sj}

        def place(recv, other, query, zone=None):
            """Project query onto receiver ``recv`` ('i'/'j'); count skips."""
            m = mice[recv]
            if m is None:
                stats["receiver_absent"] += 1
                return
            if query is None:
                stats["no_query_point"] += 1
                return
            res = project_to_midline(m["kp"], query, m["bl"])
            if res is None:
                stats["unresolved_keypoints"] += 1
                return
            add(row, slots[recv], slots[other], ctype, res, zone)

        if ctype == "N2N":
            # Symmetric: each mouse is touched at the nose by the other's nose.
            place("i", "j", mice["j"]["nose"] if mice["j"] else None)
            place("j", "i", mice["i"]["nose"] if mice["i"] else None)
        elif ctype in ("N2AG", "N2B"):
            role = row.get("investigator_role")
            if role not in ("i", "j"):
                stats["no_investigator"] += 1
                continue
            recv = "j" if role == "i" else "i"
            q = mice[role]["nose"] if mice[role] else None
            zone = row.get("n2b_zone") if ctype == "N2B" and row.get("n2b_zone") else None
            place(recv, role, q, zone)
        elif ctype == "SBS":
            for recv, other in (("i", "j"), ("j", "i")):
                m, o = mice[recv], mice[other]
                if m is None or o is None:
                    stats["receiver_absent"] += 1
                    continue
                if o["cen"] is None:
                    stats["no_query_point"] += 1
                    continue
                res = flank_facing(m["kp"], o["cen"], m["bl"])
                if res is None:
                    stats["unresolved_keypoints"] += 1
                    continue
                add(row, slots[recv], slots[other], ctype, res)
        else:
            stats["not_contact"] += 1
            continue
        if len(records) > before:
            stats["located_rows"] += 1
    stats["records"] = len(records)
    return records, stats


def aggregate_by_region(records: List[dict], focal: str) -> Dict[str, dict]:
    """Per-region count and total duration for rows where ``focal`` is touched."""
    agg: Dict[str, dict] = {}
    for r in records:
        if r["focal_mouse"] != focal:
            continue
        a = agg.setdefault(r["region"], {"count": 0, "dur_sec": 0.0})
        a["count"] += 1
        a["dur_sec"] += r["dur_sec"]
    return agg
