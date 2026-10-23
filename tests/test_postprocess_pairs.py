"""Post-proceso multi-par: las ventanas temporales deben aplicarse por par."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.postprocess_contacts_simple import (  # noqa: E402
    DEFAULT_CONFIG, assign_event_ids, compute_labels, extract_events,
    normalize_labels, session_frames,
)

PAIRS = ["0_1", "0_2", "1_2"]
FPS = 30.0


def _table(n_frames=200, run=(50, 110), ct_run="N2N", odd=None):
    """Interleaved per-frame table (frame0: 0_1,0_2,1_2; frame1: ...). Pair 0_1 carries
    ct_run over `run`; `odd` = {frame: (type, role)} overrides single raw frames."""
    odd = odd or {}
    rows = []
    for f in range(n_frames):
        for pk in PAIRS:
            ct = ct_run if pk == "0_1" and run[0] <= f < run[1] else "none"
            role = "i" if ct != "none" else ""
            if pk == "0_1" and f in odd:
                ct, role = odd[f]
            rows.append({
                "frame_idx": f, "time_sec": f / FPS, "zone": "independent",
                "pair_key": pk, "contact_type": ct,
                "investigator_role": role,
                "nose_nose_dist_bl": 0.2 if ct == "N2N" else float("inf"),
                "centroid_dist_bl": 0.8, "score_n2n": 0.9 if ct == "N2N" else 0.0,
                "score_n2ag": 0.9 if ct == "N2AG" else 0.0,
            })
    return pd.DataFrame(rows)


def test_interleaved_pairs_keep_run():
    df = _table()
    raw, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)
    n2n = ev[ev.contact_type == "N2N"]
    assert len(n2n) == 1
    r = n2n.iloc[0]
    assert r.pair_key == "0_1" and r.pair_label == "R1-R2"
    assert r.duration_frames == 60
    assert r.start_frame == 50 and r.end_frame == 109
    assert r.investigator_slot == ""   # N2N is symmetric: no investigator
    assert r.peak_score == 0.9
    assert abs(r.mean_nose_nose_dist_bl - 0.2) < 1e-6
    assert set(ev.event_id) == set(range(len(ev)))  # ids unicos y densos
    assert session_frames(df) == 200


def test_event_ids_align_with_rows():
    df = _table()
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ids = assign_event_ids(df, real)
    ev = extract_events(df, real, FPS)
    n2n_id = int(ev[ev.contact_type == "N2N"].event_id.iloc[0])
    assert (ids == n2n_id).sum() == 60
    assert set(df.pair_key[ids == n2n_id]) == {"0_1"}


def test_none_normalization():
    vals = normalize_labels(pd.Series(["none", None, "NC", "N2N", float("nan"), ""]))
    assert list(vals) == ["", "", "", "N2N", "", ""]
    df = _table()
    raw, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    assert (raw == "").sum() == len(df) - 60
    assert (real == "NC").sum() > 0 and "none" not in set(real)


def test_no_pair_key_behaves_as_before():
    df = _table()
    df = df[df.pair_key == "0_1"].drop(columns=["pair_key"]).reset_index(drop=True)
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)
    assert "pair_key" not in ev.columns
    assert len(ev[ev.contact_type == "N2N"]) == 1


def test_investigator_votes_only_over_frames_of_the_event_type():
    # Two raw N2B frames (role j) inside an N2AG run (role i) are folded into the run by
    # smoothing; the investigator must come from the N2AG frames only.
    df = _table(ct_run="N2AG", odd={70: ("N2B", "j"), 71: ("N2B", "j")})
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)
    n2ag = ev[ev.contact_type == "N2AG"]
    assert len(n2ag) == 1 and n2ag.iloc[0].duration_frames == 60
    assert n2ag.iloc[0].investigator_slot == 0


def test_symmetric_types_never_get_an_investigator():
    # One raw N2AG frame (role i) inside an SBS run must not give SBS an investigator
    # (seen on Bunya 2026-10-23: event R4-R5 SBS reported investigator R4).
    df = _table(ct_run="SBS", odd={80: ("N2AG", "i")})
    df.loc[df.pair_key == "0_1", "investigator_role"] = df.loc[df.pair_key == "0_1", "contact_type"].map(
        lambda c: "i" if c == "N2AG" else "")
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)
    sbs = ev[ev.contact_type == "SBS"]
    assert len(sbs) == 1 and sbs.iloc[0].investigator_slot == ""
