"""Post-proceso multi-par: las ventanas temporales deben aplicarse por par."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.postprocess_contacts_simple import (  # noqa: E402
    DEFAULT_CONFIG, assign_event_ids, calibration_line, compute_labels, extract_events,
    load_calibration, normalize_labels, per_animal_summary, run_postprocess, session_frames,
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


def test_per_type_minimum_bout_duration():
    df = _table(run=(50, 60), ct_run="FOL")           # 10 frames of FOL = 0.33 s at 30 fps
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    assert "FOL" in set(real)                        # survives the global 0.3 s
    cfg = {**DEFAULT_CONFIG, "min_bout": {"duration_sec": 0.3, "by_type": {"FOL": 0.4}}}
    _, real = compute_labels(df, cfg, FPS)
    assert "FOL" not in set(real)                    # removed by the 0.4 s FOL minimum


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


def _multi_table(runs, n_frames=200):
    """Interleaved per-frame table for 3 animals. runs = {pair: (start, stop, type, role)}."""
    rows = []
    for f in range(n_frames):
        for pk in PAIRS:
            ct, role = "none", ""
            if pk in runs and runs[pk][0] <= f < runs[pk][1]:
                ct, role = runs[pk][2], runs[pk][3]
            rows.append({
                "frame_idx": f, "time_sec": f / FPS, "zone": "independent",
                "pair_key": pk, "contact_type": ct, "investigator_role": role,
                "nose_nose_dist_bl": 0.2 if ct == "N2N" else float("inf"),
                "centroid_dist_bl": 0.8, "score_n2n": 0.9 if ct == "N2N" else 0.0,
                "score_n2ag": 0.9 if ct == "N2AG" else 0.0,
            })
    return pd.DataFrame(rows)


RUNS_3 = {"0_1": (50, 110, "N2N", "i"), "0_2": (80, 140, "N2AG", "i")}


def test_per_animal_summary_union_overlap_and_roles():
    df = _multi_table(RUNS_3)
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)
    pa = per_animal_summary(df, real, ev, FPS)
    assert list(pa.animal) == ["R1", "R2", "R3"]
    pa = pa.set_index("animal")

    r1 = pa.loc["R1"]
    assert r1.n_events_total == 2 and r1.n_events_N2N == 1 and r1.n_events_N2AG == 1
    assert r1.n_events_N2B == 0 and r1.n_events_SBS == 0 and r1.n_events_FOL == 0
    assert r1.contact_time_sec == pytest.approx(3.0)            # frames 50-139, a union
    assert r1.time_2plus_contacts_sec == pytest.approx(1.0)     # frames 80-109
    assert r1.n_as_investigator == 1 and r1.n_as_receiver == 0
    assert r1.fol_time_sec == 0.0

    r2 = pa.loc["R2"]                                            # only in the symmetric N2N
    assert r2.n_events_total == 1 and r2.contact_time_sec == pytest.approx(2.0)
    assert r2.time_2plus_contacts_sec == 0.0
    assert r2.n_as_investigator == 0 and r2.n_as_receiver == 0

    r3 = pa.loc["R3"]
    assert r3.n_events_total == 1 and r3.n_events_N2AG == 1
    assert r3.contact_time_sec == pytest.approx(2.0)
    assert r3.n_as_investigator == 0 and r3.n_as_receiver == 1


def test_per_animal_summary_fol_is_not_contact_time():
    df = _multi_table({"1_2": (20, 70, "FOL", "j")})
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)
    pa = per_animal_summary(df, real, ev, FPS).set_index("animal")
    assert pa.loc["R1"].n_events_total == 0 and pa.loc["R1"].fol_time_sec == 0.0
    for animal in ("R2", "R3"):
        assert pa.loc[animal].n_events_FOL == 1
        assert pa.loc[animal].contact_time_sec == 0.0           # FOL is non-contact
        assert pa.loc[animal].fol_time_sec == pytest.approx(50 / FPS, abs=1e-3)
    assert pa.loc["R3"].n_as_investigator == 1                  # role j of pair 1_2 = slot 2 follows
    assert pa.loc["R2"].n_as_receiver == 1


def test_per_animal_summary_without_pair_key_is_empty():
    df = _table()
    df = df[df.pair_key == "0_1"].drop(columns=["pair_key"]).reset_index(drop=True)
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)
    assert per_animal_summary(df, real, ev, FPS).empty


def test_calibration_line():
    cal = {"contact_bl": 0.30, "contact_cm": 2.7, "body_length_px_median": 164.0,
           "cm_per_px": 9.5 / 164.0, "cm_per_px_source": "estimated_from_body_length"}
    assert calibration_line(cal) == (
        "Contact distance: 0.30 BL (\u2248 2.7 cm; cm/px estimated from body length 164 px = 9.5 cm)")
    assert "measured" in calibration_line({**cal, "cm_per_px_source": "config"})
    assert calibration_line({"contact_bl": 0.30}) == "Contact distance: 0.30 BL"   # no cm metadata
    assert calibration_line({}) is None


def test_run_postprocess_writes_per_animal_outputs(tmp_path):
    import json
    _multi_table(RUNS_3).to_csv(tmp_path / "contacts_per_frame.csv", index=False)
    (tmp_path / "session_summary.json").write_text(json.dumps({
        "metadata": {"fps": FPS, "body_length_px_median": 164.0, "cm_per_px": 0.0579,
                     "cm_per_px_source": "estimated_from_body_length",
                     "thresholds_cm": {"contact_enter": 2.7, "contact_exit": 4.1, "proximity": 9.5}},
        "parameters": {"contact_zone_bl_enter": 0.30}}), encoding="utf-8")
    assert load_calibration(tmp_path)["contact_cm"] == pytest.approx(2.7)

    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=False)

    pa = pd.read_csv(tmp_path / "per_animal_summary.csv").set_index("animal")
    assert pa.loc["R1", "contact_time_sec"] == pytest.approx(3.0)
    log = (tmp_path / "event_log.txt").read_text(encoding="utf-8")
    assert "Contact distance: 0.30 BL (\u2248 2.7 cm; cm/px estimated from body length 164 px = 9.5 cm)" in log
    assert log.index("--- Summary by type ---") < log.index("--- Per-animal summary ---") \
        < log.index("--- Filtering impact ---")
    summary = json.loads((tmp_path / "session_summary_real.json").read_text(encoding="utf-8"))
    assert summary["per_animal"]["R1"]["n_events_total"] == 2
    assert summary["per_animal"]["R3"]["n_as_receiver"] == 1


def test_consolidation_adds_per_animal_and_calibration_sheets(tmp_path):
    import json
    pytest.importorskip("openpyxl")
    _multi_table(RUNS_3).to_csv(tmp_path / "contacts_per_frame.csv", index=False)
    (tmp_path / "session_summary.json").write_text(json.dumps({
        "quality_flags": {"nose_off_mask": 4},
        "metadata": {"fps": FPS, "body_length_px_median": 164.0, "cm_per_px": 0.0579,
                     "cm_per_px_source": "estimated_from_body_length",
                     "thresholds_cm": {"contact_enter": 2.7, "contact_exit": 4.1, "proximity": 9.5}},
        "parameters": {"contact_zone_bl_enter": 0.30}}), encoding="utf-8")

    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=True)

    xlsx = tmp_path / "results.xlsx"
    assert xlsx.exists() and not (tmp_path / "per_animal_summary.csv").exists()   # absorbed
    sheets = pd.read_excel(xlsx, sheet_name=None)
    assert "PerAnimal" in sheets and len(sheets["PerAnimal"]) == 3
    params = sheets["Parameters"]
    cal = params[params.section == "calibration"].set_index("parameter")["value"]
    assert float(cal["cm_per_px"]) == pytest.approx(0.0579)
    assert float(cal["thresholds_cm.contact_enter"]) == pytest.approx(2.7)
    assert cal["cm_per_px_source"] == "estimated_from_body_length"
    assert "cm_per_px" not in set(params[params.section == "video"].parameter)
    flags = params[params.section == "quality_flags"].set_index("parameter")["value"]
    assert int(flags["nose_off_mask"]) == 4

    # Re-running on the consolidated directory keeps the calibration (summary JSON is gone)
    assert load_calibration(tmp_path)["contact_cm"] == pytest.approx(2.7)
    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=True)
    params = pd.read_excel(xlsx, sheet_name="Parameters")
    assert "thresholds_cm.contact_enter" in set(params[params.section == "calibration"].parameter)
