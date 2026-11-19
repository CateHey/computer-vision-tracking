"""contact_rules_v3: body masks without tail, refined nose tip, head circle, and the rule
changes built on them (N2AG evidence, FOL guard, N2B vs SBS, N2N subtypes, low confidence,
ambiguous, suspect masks, approach / withdrawal subtypes, brief and following tables)."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.contacts_v2 import (  # noqa: E402
    ContactClassifier, ContactEvent, ContactTrackerV2, ContactType, Dynamics, DynamicsTracker,
    body_mask, mask_touches_disk, refined_nose_tip,
)
from src.common.utils import Detection, Keypoint  # noqa: E402

H, W = 300, 600
BL = 100.0            # body length passed to the classifier (nose -> tail_start = 100 px)
V3 = {"contact_rules_v3": True}


def _rat(cx, cy, heading=0.0, nose_inset=0.0, tail=True):
    """Detection + full mask of a rat centred at (cx, cy) facing `heading` (radians).

    Body: ellipse 50 x 16 px semi-axes; tail: 4 px wide, 70 px long behind tail_start.
    The nose keypoint sits `nose_inset` px inside the snout."""
    ux, uy = math.cos(heading), math.sin(heading)
    px, py = -uy, ux

    def at(a, b=0.0):
        return (cx + a * ux + b * px, cy + a * uy + b * py)

    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (50, 16), math.degrees(heading), 0, 360, 1, -1)
    if tail:
        p0, p1 = at(-50), at(-120)
        cv2.line(mask, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))), 1, 4)
    names = {"nose": at(50 - nose_inset), "left_ear": at(36, -7), "right_ear": at(36, 7),
             "mid_body": at(0), "tail_start": at(-50), "tail_base": at(-85), "tail_tip": at(-120)}
    kps = [Keypoint(x, y, 0.9, n) for n, (x, y) in names.items()]
    xs, ys = [x for x, _ in names.values()], [y for _, y in names.values()]
    det = Detection(min(xs), min(ys), max(xs), max(ys), 0.9, "rat", kps)
    return det, mask.astype(bool), (cx, cy)


def _run(a, b, cfg=None, vel_a=(0.0, 0.0), vel_b=(0.0, 0.0), frames=1, clf=None):
    """Classify the pair (a, b) for `frames` identical frames; returns (last event, classifier)."""
    clf = clf or ContactClassifier("0_1", 0, 1, dict(cfg or {}))
    ev = None
    for f in range(frames):
        ev = clf.classify(a[0], b[0], a[1], b[1], a[2], b[2], vel_a, vel_b, BL, BL,
                          frame_idx=f, time_sec=f / 30.0)
    return ev, clf


# ── geometry ──

def test_body_mask_drops_the_tail_and_keeps_the_body():
    _, full, _ = _rat(300, 150)
    _, no_tail, _ = _rat(300, 150, tail=False)
    body = body_mask(full, BL, 0.10)
    tail_px = full & ~no_tail
    assert tail_px.sum() > 200
    assert not (body & tail_px)[:, :235].any()                   # nothing of the tail survives
    assert (body & no_tail).sum() >= 0.97 * no_tail.sum()        # border restored
    assert body_mask(None, BL) is None


def test_refined_nose_tip_moves_to_the_snout():
    det, full, _ = _rat(300, 150, nose_inset=8)
    tip = refined_nose_tip((342.0, 150.0), (1.0, 0.0), full, 0.15 * BL)
    assert tip[0] == pytest.approx(350, abs=1.5) and tip[1] == pytest.approx(150, abs=2)
    assert refined_nose_tip((342.0, 150.0), None, full, 15) == (342.0, 150.0)


def test_mask_touches_disk_is_euclidean():
    m = np.zeros((50, 50), bool)
    m[10, 20] = True
    assert mask_touches_disk((13, 13), m, 10.0) is True         # 7.6 px
    assert mask_touches_disk((13, 18), m, 10.0) is False        # 10.6 px: inside the square, outside the disk


# ── rules ──

def test_nose_on_the_tail_is_a_flag_not_nose_to_body():
    b = _rat(300, 150)                                          # tail from x=250 back to x=180
    a = _rat(205, 204, heading=-math.pi / 2)                   # nose at (205, 154), 2 px below the tail line
    ev, _ = _run(a, b, {**V3, "n2b_nose_mask_radius_px": 6})
    assert ev.nose_on_tail and not ev.head_touch_ij
    assert ev.contact_type != ContactType.N2B
    old, _ = _run(a, b, {"n2b_nose_mask_radius_px": 6})        # before v3: tail counted as body
    assert old.contact_type == ContactType.N2B


def test_n2ag_needs_evidence_of_touching():
    b = _rat(300, 150)                                          # rear (tail_start) at (250, 150)
    # nose tip 27 px (0.27 BL) behind the rear, off the body, walking: no evidence
    a = _rat(173, 150 + 0.0, heading=0.0)
    a = (a[0], a[1] & (np.arange(W)[None, :] < 224), a[2])     # keep a's mask clear of b
    ev, _ = _run(a, b, V3, vel_a=(30.0, 0.0), vel_b=(30.0, 0.0))
    assert ev.nose_tailbase_ij_bl == pytest.approx(0.27, abs=0.02)
    assert ev.contact_type != ContactType.N2AG and ev.scores.n2ag < 0.5
    old, _ = _run(a, b, {}, vel_a=(30.0, 0.0), vel_b=(30.0, 0.0))
    assert old.contact_type == ContactType.N2AG                 # v2: distance alone was enough
    # head circle on the rear: N2AG, and both walking -> subtype following
    a2 = _rat(198, 150)                                         # nose tip 2 px from the rear
    ev, _ = _run(a2, b, V3, vel_a=(30.0, 0.0), vel_b=(30.0, 0.0))
    assert ev.n2ag_subtype == "classic"                         # 0.3 BL/s: below n2ag_walking_bls
    ev, _ = _run(a2, b, V3, vel_a=(90.0, 0.0), vel_b=(90.0, 0.0))
    assert ev.head_touch_ij and ev.contact_type == ContactType.N2AG
    assert ev.n2ag_subtype == "following" and ev.investigator_role == "i"
    # slow for several frames at 0.27 BL also counts as evidence
    ev, _ = _run(a, b, V3, frames=6)
    assert ev.contact_type == ContactType.N2AG and ev.n2ag_subtype == "classic"


def test_fol_is_annulled_only_by_a_real_touch():
    cfg = {**V3}
    b = _rat(300, 150)
    a = _rat(175, 150)                                          # nose 25 px behind b's rear, no touch
    clf = ContactClassifier("0_1", 0, 1, cfg)
    old_clf = ContactClassifier("0_1", 0, 1, {})
    for f in range(8):                                          # both walk right, a behind b
        dx = 3 * f
        a_f = _rat(175 + dx, 150)
        a_f = (a_f[0], a_f[1] & (np.arange(W)[None, :] < 236 + dx), a_f[2])
        b_f = _rat(300 + dx, 150)
        ev = clf.classify(a_f[0], b_f[0], a_f[1], b_f[1], a_f[2], b_f[2], (90.0, 0), (90.0, 0),
                          BL, BL, frame_idx=f, time_sec=f / 30)
        old = old_clf.classify(a_f[0], b_f[0], a_f[1], b_f[1], a_f[2], b_f[2], (90.0, 0), (90.0, 0),
                               BL, BL, frame_idx=f, time_sec=f / 30)
    assert not ev.head_touch_ij and ev.mask_contact_bl < 0.02
    assert ev.scores.fol > 0 and old.scores.fol == 0.0          # v2 annulled it below 0.5 BL


def test_n2b_competes_with_sbs_and_records_the_zone():
    b = _rat(300, 150)                                          # body top edge at y = 134
    a = _rat(300, 82, heading=math.pi / 2)                      # facing down, nose at (300, 132)
    ev, _ = _run(a, b, V3)
    assert ev.head_touch_ij and not ev.head_touch_ji
    assert ev.contact_type == ContactType.N2B and ev.investigator_role == "i"
    assert ev.n2b_zone == "torso"
    # with a strong SBS score N2B still scores (v3) instead of being zeroed (v2)
    new = ContactClassifier("0_1", 0, 1, V3)
    old = ContactClassifier("0_1", 0, 1, {"n2b_nose_mask_radius_px": 6})
    kw = dict(nose_i=(300.0, 132.0), nose_j=None, mask_i=a[1], mask_j=b[1], nose_nose_bl=1.0,
              nose_i_rear_j_bl=1.0, nose_j_rear_i_bl=1.0, centroid_i=a[2], centroid_j=b[2],
              orient_i=(0.0, 1.0), orient_j=(1.0, 0.0), sbs_score=0.9)
    assert new._score_n2b(**kw, head_touch_ij=True)[0] == pytest.approx(1.0)
    assert old._score_n2b(**kw)[0] == 0.0


def test_n2n_subtypes_and_lateral_is_barely_penalised():
    b = _rat(300, 150, heading=math.pi)                         # faces left, nose at (250, 150)
    a = _rat(198, 150)                                          # faces right, nose at (248, 150)
    ev, _ = _run(a, b, V3)
    assert ev.contact_type == ContactType.N2N and ev.n2n_subtype == "frontal"
    a_lat = _rat(300, 150 - 34)                                 # same direction, heads side by side
    b_lat = _rat(300, 150)
    ev, _ = _run(a_lat, b_lat, V3)
    assert ev.n2n_subtype in ("lateral", "") or ev.contact_type != ContactType.N2N
    clf = ContactClassifier("0_1", 0, 1, V3)
    old = ContactClassifier("0_1", 0, 1, {})
    same = ((1.0, 0.0), (1.0, 0.0))
    assert clf._score_n2n(0.1, *same) == pytest.approx(0.85)
    assert old._score_n2n(0.1, *same) == pytest.approx(0.3)
    assert ContactClassifier._n2n_subtype((1, 0), (0, 1)) == "perpendicular"


def test_ambiguous_and_low_confidence_flags():
    # parallel and touching: SBS ~0.99 with lateral N2N as secondary; a lateral factor of 0.95
    # brings N2N within 0.10 of SBS -> ambiguous
    b, a = _rat(300, 150), _rat(300, 120)
    ev, _ = _run(a, b, V3, frames=3)
    assert ev.contact_type == ContactType.SBS and ev.secondary_type == ContactType.N2N
    assert not ev.ambiguous                                     # 0.99 vs 0.85
    ev, _ = _run(a, b, {**V3, "n2n_lateral_factor": 0.95}, frames=3)
    assert ev.contact_type == ContactType.SBS and ev.ambiguous
    # the bodies touch (nose on the flank) but no type reaches the activation threshold:
    # the best type is kept and flagged
    a = _rat(300, 82, heading=math.pi / 2)
    ev, _ = _run(a, b, {**V3, "activation_threshold": 1.01})
    assert ev.contact_type == ContactType.N2B and ev.low_confidence
    old, _ = _run(a, b, {"activation_threshold": 1.01, "n2b_nose_mask_radius_px": 6})
    assert old.contact_type == ContactType.NONE and not old.low_confidence


def test_suspect_mask_flag(tmp_path):
    tr = ContactTrackerV2(tmp_path, fps=30, num_slots=2, video_path="x",
                          config={**V3, "fallback_body_length_px": BL})
    a, b = _rat(150, 150), _rat(450, 150)
    for f in range(35):
        tr.update([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], f)
    big = a[1].copy()
    big[100:300, 50:250] = True                                 # leak into the background
    tr.update([a[0], b[0]], [big, b[1]], [a[2], b[2]], 35)
    last = tr._all_events[-1]
    assert last.suspect_mask_i and not last.suspect_mask_j
    assert not tr._all_events[-2].suspect_mask_i


def _dyn(pair, f, dyn, mover=None, ct=ContactType.NONE):
    ev = ContactEvent(frame_idx=f, time_sec=f / 30, pair_key=pair, contact_type=ct)
    ev.dynamics, ev.mover, ev.centroid_dist_bl = dyn, mover, 1.0
    if ct != ContactType.NONE:
        from src.common.contacts_v2 import CONTACT_FAMILY
        ev.family = CONTACT_FAMILY[ct]
    return ev


def test_dynamics_subkinds():
    dt = DynamicsTracker(30, {**V3, "dynamics_min_frames": 3})
    f = 0
    for _ in range(6):                                          # i approaches alone -> search, ends in contact
        dt.process_event(_dyn("0_1", f, Dynamics.CLOSING, "i")); f += 1
    for _ in range(5):
        dt.process_event(_dyn("0_1", f, Dynamics.STABLE, None, ContactType.N2N)); f += 1
    for _ in range(6):                                          # j leaves right after -> withdrawal
        dt.process_event(_dyn("0_1", f, Dynamics.SEPARATING, "j")); f += 1
    for _ in range(30):
        dt.process_event(_dyn("0_1", f, Dynamics.STABLE)); f += 1
    for _ in range(6):                                          # both close in -> mutual approach
        dt.process_event(_dyn("0_1", f, Dynamics.CLOSING, None)); f += 1
    for _ in range(6):                                          # i leaves, no contact before -> flight
        dt.process_event(_dyn("0_1", f, Dynamics.SEPARATING, "i")); f += 1
    out = [(e.kind, e.subkind, e.mover, e.after_contact, e.ends_in_contact) for e in dt.close_all()]
    assert out == [
        ("approach", "search", "i", False, True),
        ("avoidance", "withdrawal", "j", True, False),
        ("approach", "mutual_approach", None, False, False),
        ("avoidance", "flight", "i", False, False),
    ]


# ── post-processing: Events columns and following episodes ──

def test_events_carry_v3_columns_and_following_episodes_merge():
    from scripts.postprocess_contacts_simple import (
        DEFAULT_CONFIG, compute_labels, extract_events, following_episodes)
    rows = []
    for f in range(120):
        ct, sub, role = "none", "", ""
        if 10 <= f < 40:
            ct, role = "FOL", "i"
        elif 40 <= f < 70:
            ct, sub, role = "N2AG", "following", "i"
        elif 90 <= f < 110:
            ct, sub, role = "N2AG", "classic", "j"
        rows.append({"frame_idx": f, "time_sec": f / 30, "pair_key": "0_1", "contact_type": ct,
                     "investigator_role": role, "n2ag_subtype": sub, "secondary_type": "N2B" if ct == "N2AG" else "",
                     "ambiguous": int(f >= 90 and ct != "none"), "low_confidence": 0,
                     "tail_contact_bl": 0.0, "n2n_subtype": "", "n2b_zone": "",
                     "nose_nose_dist_bl": 1.0, "centroid_dist_bl": 0.8})
    df = pd.DataFrame(rows)
    _, real = compute_labels(df, DEFAULT_CONFIG, 30.0)
    ev = extract_events(df, real, 30.0)
    ev = ev[ev.contact_type != "NC"].reset_index(drop=True)
    assert list(ev.contact_type) == ["FOL", "N2AG", "N2AG"]
    assert list(ev.subtype) == ["", "following", "classic"]
    assert list(ev.secondary_type) == ["", "N2B", "N2B"] and list(ev.ambiguous) == [0, 0, 1]
    fol = following_episodes(df, extract_events(df, real, 30.0), 30.0, DEFAULT_CONFIG)
    assert len(fol) == 1                                        # FOL + N2AG following, same follower
    r = fol.iloc[0]
    assert (r.start_frame, r.end_frame, r.composition, r.n_events) == (10, 69, "FOL → N2AG", 2)
    assert r.sequence == "FOL 10-39 → N2AG 40-69"
    assert following_episodes(df.drop(columns=["n2ag_subtype"]), ev, 30.0, DEFAULT_CONFIG) is None


def test_n2ag_gaze_is_ignored_with_the_nose_on_the_rear():
    """At <= 0.20 BL the direction nose -> rear is a few px long and points anywhere: the
    gaze factor must not sink N2AG (it turned annotated N2AG into N2B 'rear')."""
    new = ContactClassifier("0_1", 0, 1, V3)
    old = ContactClassifier("0_1", 0, 1, {})
    kw = dict(nose_j_rear_i_bl=2.0, nose_i=(100.0, 100.0), nose_j=(400.0, 100.0),
              rear_i=(300.0, 100.0), rear_j=(103.0, 100.0), orient_i=(-1.0, 0.0), orient_j=(1.0, 0.0))
    on_rear = new._score_n2ag(0.05, **kw)                       # 3 px "behind" the nose
    assert on_rear == (pytest.approx(1.0), "i")
    assert old._score_n2ag(0.05, **kw)[0] == pytest.approx(0.4)    # v2 kept its 0.4 floor
    # further away the gaze still counts: looking away at 0.25 BL scores low
    far = new._score_n2ag(0.25, **{**kw, "rear_j": (125.0, 100.0)})
    assert far[0] == pytest.approx(0.2)


def test_groups_join_alike_types_and_similarity_table():
    from scripts.postprocess_contacts_simple import (
        DEFAULT_CONFIG, add_event_groups, contact_group, group_summary, group_table, type_similarity)
    g = group_table(DEFAULT_CONFIG)
    assert contact_group("N2N", "lateral", g) == "Lado a lado"                  # level 1 joins ...
    assert contact_group("SBS", "", g) == "Lado a lado"
    assert contact_group("N2N", "frontal", g) == "Cara a cara"                  # ... level 3 splits
    assert contact_group("N2N", None, g) == "Cara a cara"                       # no subtype: default
    assert contact_group("N2AG", "following", g) == "Persecución"
    assert contact_group("FOL", "", g) == "Persecución"
    assert contact_group("N2AG", "classic", g) == "Olfateo anogenital"
    assert contact_group("N2B", "rear", g) == "Olfateo corporal"
    custom = group_table({"groups": {"n2b": {"default": "Olfateo"}, "N2AG": {"default": "Olfateo"}}})
    assert contact_group("N2AG", "classic", custom) == "Olfateo"                # editable in the config
    ev = pd.DataFrame({"event_id": [0, 1, 2, 3], "pair_key": ["0_1", "0_1", "1_2", "0_1"],
                       "contact_type": ["N2N", "SBS", "N2N", "NC"],
                       "subtype": ["lateral", "", "frontal", ""], "duration_sec": [1.0, 2.0, 0.5, 9.0]})
    ev = add_event_groups(ev, DEFAULT_CONFIG)
    assert list(ev.group) == ["Lado a lado", "Lado a lado", "Cara a cara", ""]
    gs = group_summary(ev)
    assert list(gs.group) == ["Cara a cara", "Lado a lado"]
    assert gs.set_index("group").loc["Lado a lado", "total_sec"] == 3.0       # no double counting
    df = pd.DataFrame({"contact_type": ["N2AG"] * 4 + ["SBS"] * 2 + ["none"],
                       "secondary_type": ["N2B", "N2B", "N2B", "", "N2N", "", "SBS"]})
    sim = type_similarity(df).set_index("type")
    assert sim.loc["N2AG", "N2B_%"] == 75.0 and sim.loc["N2AG", "no_second_%"] == 25.0
    assert sim.loc["SBS", "N2N_%"] == 50.0 and sim.loc["N2N", "frames"] == 0
