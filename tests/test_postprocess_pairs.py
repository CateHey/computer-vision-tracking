"""Post-proceso multi-par: las ventanas temporales deben aplicarse por par."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.excel_report import DIAGNOSTICS_README  # noqa: E402
from scripts.postprocess_contacts_simple import (  # noqa: E402
    CONTACT_TYPES, DEFAULT_CONFIG, INVESTIGATION_COLUMNS, PAIR_MATRIX_COLUMNS, PER_ANIMAL_COLUMNS,
    assign_event_ids, calibration_line, compute_labels, extract_events, investigation_bouts,
    load_calibration, n_animals_in_table, normalize_labels, pair_contact_matrix,
    per_animal_summary, run_postprocess, session_frames,
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
    assert r1.n_investigation_bouts == 2                         # one bout in each of its two pairs
    # 200 frames at 30 fps = 6.667 s = 0.1111 min
    assert r1.session_min == pytest.approx(200 / FPS / 60, abs=1e-4)
    assert r1.events_per_min == pytest.approx(2 / (200 / FPS / 60), abs=1e-2)       # 18.0
    assert r1.contact_sec_per_min == pytest.approx(3.0 / (200 / FPS / 60), abs=1e-2)  # 27.0
    assert r1.investigation_bouts_per_min == pytest.approx(2 / (200 / FPS / 60), abs=1e-2)   # 18.0
    assert set(pa.session_min) == {r1.session_min}     # one session length for every animal
    assert list(pa.reset_index().columns) == PER_ANIMAL_COLUMNS

    r2 = pa.loc["R2"]                                            # only in the symmetric N2N
    assert r2.n_events_total == 1 and r2.contact_time_sec == pytest.approx(2.0)
    assert r2.time_2plus_contacts_sec == 0.0
    assert r2.n_as_investigator == 0 and r2.n_as_receiver == 0
    assert r2.events_per_min == pytest.approx(9.0, abs=1e-2)
    assert r2.contact_sec_per_min == pytest.approx(18.0, abs=1e-2)
    assert r2.n_investigation_bouts == 1 and r2.investigation_bouts_per_min == pytest.approx(9.0, abs=1e-2)

    r3 = pa.loc["R3"]
    assert r3.n_events_total == 1 and r3.n_events_N2AG == 1
    assert r3.contact_time_sec == pytest.approx(2.0)
    assert r3.n_as_investigator == 0 and r3.n_as_receiver == 1
    assert r3.n_investigation_bouts == 1


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
    assert (pa.n_investigation_bouts == 0).all()                # FOL is not investigation
    assert (pa.investigation_bouts_per_min == 0).all()


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
    assert pa.loc["R1", "events_per_min"] == pytest.approx(18.0, abs=1e-2)
    assert {"session_min", "events_per_min", "contact_sec_per_min"} <= set(pa.columns)
    assert pa.loc["R1", "n_investigation_bouts"] == 2 and pa.loc["R2", "n_investigation_bouts"] == 1
    assert pa.loc["R1", "investigation_bouts_per_min"] == pytest.approx(18.0, abs=1e-2)
    log = (tmp_path / "event_log.txt").read_text(encoding="utf-8")
    assert "Contact distance: 0.30 BL (\u2248 2.7 cm; cm/px estimated from body length 164 px = 9.5 cm)" in log
    assert log.index("--- Summary by type ---") < log.index("--- Investigation bouts") \
        < log.index("--- Per-animal summary ---") \
        < log.index("Per-pair matrix: pair_contact_matrix.csv") < log.index("--- Filtering impact ---")
    summary = json.loads((tmp_path / "session_summary_real.json").read_text(encoding="utf-8"))
    assert summary["per_animal"]["R1"]["n_events_total"] == 2
    assert summary["per_animal"]["R3"]["n_as_receiver"] == 1
    assert summary["per_animal"]["R1"]["n_investigation_bouts"] == 2
    assert summary["investigation"]["n_bouts"] == 2 and summary["investigation"]["total_sec"] == 4.0
    assert summary["investigation"]["by_dominant_type"] == {"N2N": 1, "N2AG": 1, "N2B": 0}
    assert summary["parameters"]["investigation_max_gap"] == 3
    ib = pd.read_csv(tmp_path / "investigation_bouts.csv")
    assert list(ib.columns) == INVESTIGATION_COLUMNS and list(ib.pair_label) == ["R1-R2", "R1-R3"]
    assert summary["per_animal"]["R1"]["events_per_min"] == pytest.approx(18.0, abs=1e-2)

    pm = pd.read_csv(tmp_path / "pair_contact_matrix.csv").set_index("pair_label")
    assert list(pm.index) == ["R1-R2", "R1-R3", "R2-R3"]                # every pair, empty ones too
    assert pm.loc["R1-R3", "n_events_N2AG"] == 1 and pm.loc["R2-R3", "n_events_total"] == 0
    assert summary["pair_matrix"]["R1-R2"]["n_events_N2N"] == 1
    assert summary["pair_matrix"]["R1-R2"]["contact_sec"] == pytest.approx(2.0)
    assert summary["pair_matrix"]["R2-R3"]["animal_a"] == "R2"
    assert summary["pair_matrix"]["R2-R3"]["n_events_total"] == 0


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
    assert not (tmp_path / "pair_contact_matrix.csv").exists()                      # absorbed
    sheets = pd.read_excel(xlsx, sheet_name=None)
    assert "PerAnimal" in sheets and len(sheets["PerAnimal"]) == 3
    assert "PairMatrix" in sheets and len(sheets["PairMatrix"]) == 3               # 3 animals = 3 pairs
    names = list(sheets)
    assert names.index("PairMatrix") == names.index("PerAnimal") + 1                # right after PerAnimal
    assert names.index("Investigation") == names.index("PairMatrix") + 1            # right after PairMatrix
    assert names == ["Summary", "Events", "PerAnimal", "PairMatrix", "Investigation",
                     "PerFrame", "Parameters"]                                       # no reports -> no ByType / Global
    assert not (tmp_path / "diagnostics.xlsx").exists()                              # no diagnostic inputs
    assert len(sheets["Investigation"]) == 2 and not (tmp_path / "investigation_bouts.csv").exists()
    assert list(sheets["Investigation"].columns) == INVESTIGATION_COLUMNS
    assert list(sheets["PairMatrix"].columns) == PAIR_MATRIX_COLUMNS
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


# ── results.xlsx holds only validated tables; diagnostics.xlsx holds the rest ──

RESULTS_SHEETS = ["Summary", "Events", "PerAnimal", "PairMatrix", "Investigation",
                  "PerFrame", "ByType", "Global", "Parameters"]
DIAGNOSTIC_SHEETS = ["Bouts", "Dynamics", "Individual"]


def _diagnostic_tables():
    """Stand-ins for the tracker's raw tables (contact_bouts, dynamics_no_contact, individual)."""
    bouts = pd.DataFrame({"id": [0, 1, 2], "pair_key": ["0_1", "0_1", "0_2"],
                          "contact_type": ["N2N", "N2AG", "N2AG"], "initiator": [0, 1, 0]})
    dynamics = pd.DataFrame({"id": [0, 1], "pair_key": ["0_1", "1_2"],
                             "kind": ["approach", "avoid"], "mover": [0, 2]})
    individual = {"metadata": {"num_slots": 3},
                  "individual_metrics": {"R1": {"slot_idx": 0, "total_events_initiated": 2}}}
    return bouts, dynamics, individual


def _write_diagnostic_sources(path):
    import json
    bouts, dynamics, individual = _diagnostic_tables()
    bouts.to_csv(path / "contact_bouts.csv", index=False)
    dynamics.to_csv(path / "dynamics_no_contact.csv", index=False)
    (path / "individual_summary.json").write_text(json.dumps(individual), encoding="utf-8")


def _consolidated_run(path):
    """3 animals, N2N + N2AG, tracker diagnostics present, reports on, workbooks built."""
    pytest.importorskip("openpyxl")
    pytest.importorskip("matplotlib")
    _multi_table(RUNS_3).to_csv(path / "contacts_per_frame.csv", index=False)
    _write_diagnostic_sources(path)
    run_postprocess(path, FPS, make_reports=True, consolidate=True)


def _check_diagnostics_workbook(path):
    diag = pd.read_excel(path / "diagnostics.xlsx", sheet_name=None)
    assert list(diag) == ["README"] + DIAGNOSTIC_SHEETS
    assert list(diag["README"].columns) == ["README"] and list(diag["README"]["README"]) == [DIAGNOSTICS_README]
    assert "Not validated by the clip review" in DIAGNOSTICS_README
    assert len(diag["Bouts"]) == 3 and list(diag["Bouts"].columns) == ["id", "pair_key", "contact_type", "initiator"]
    assert len(diag["Dynamics"]) == 2 and "mover" in diag["Dynamics"].columns
    assert len(diag["Individual"]) == 2                                              # metadata + individual_metrics
    return diag


def test_results_workbook_has_only_validated_sheets(tmp_path):
    _consolidated_run(tmp_path)

    sheets = pd.read_excel(tmp_path / "results.xlsx", sheet_name=None)
    assert list(sheets) == RESULTS_SHEETS
    assert not set(DIAGNOSTIC_SHEETS) & set(sheets)

    _check_diagnostics_workbook(tmp_path)
    for src in ("contact_bouts.csv", "dynamics_no_contact.csv", "individual_summary.json"):
        assert not (tmp_path / src).exists()                                         # absorbed


def test_by_type_and_global_count_contact_types_only(tmp_path):
    _consolidated_run(tmp_path)
    sheets = pd.read_excel(tmp_path / "results.xlsx", sheet_name=None)

    by_type = sheets["ByType"]
    assert list(by_type.contact_type) == CONTACT_TYPES                               # five rows, no NC
    assert (by_type.removed >= 0).all()

    summary = sheets["Summary"].set_index("contact_type")
    glob = sheets["Global"].set_index("metric")["value"]
    events = sheets["Events"]
    n_nc = int((events.contact_type == "NC").sum())
    assert n_nc == 5                                                                 # Events keeps its NC rows
    assert summary.loc["TOTAL", "events"] == 2
    assert glob["real_events_total"] == summary.loc["TOTAL", "events"] == len(events) - n_nc
    assert glob["real_events_total"] == by_type.real_events.sum()
    assert glob["raw_bouts_total"] == by_type.raw_bouts.sum()
    assert glob["real_contact_frames"] == 120                                        # 2 events x 60 frames; NC not counted
    # the one NC figure left: pair-time without contact, 3 pairs x 200 frames - 120 contact frames
    assert glob["no_contact_sec"] == pytest.approx((3 * 200 - 120) / FPS, abs=0.05)


def test_rerun_on_a_consolidated_run_keeps_the_diagnostics(tmp_path):
    _consolidated_run(tmp_path)
    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=True)            # sources are gone now
    _check_diagnostics_workbook(tmp_path)
    assert not set(DIAGNOSTIC_SHEETS) & set(pd.read_excel(tmp_path / "results.xlsx", sheet_name=None))


def test_diagnostics_are_carried_out_of_a_pre_split_results_workbook(tmp_path):
    pytest.importorskip("openpyxl")
    bouts, dynamics, _ = _diagnostic_tables()
    individual = pd.DataFrame({"animal": ["metadata", "individual_metrics"], "num_slots": [3, None]})
    with pd.ExcelWriter(tmp_path / "results.xlsx", engine="openpyxl") as w:          # layout before the split
        _multi_table(RUNS_3).to_excel(w, sheet_name="PerFrame", index=False)
        bouts.to_excel(w, sheet_name="Bouts", index=False)
        dynamics.to_excel(w, sheet_name="Dynamics", index=False)
        individual.to_excel(w, sheet_name="Individual", index=False)

    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=True)

    assert not set(DIAGNOSTIC_SHEETS) & set(pd.read_excel(tmp_path / "results.xlsx", sheet_name=None))
    diag = _check_diagnostics_workbook(tmp_path)
    assert list(diag["Individual"].columns) == ["animal", "num_slots"]


def test_session_summary_and_event_log_do_not_count_nc_spans_as_events(tmp_path):
    import json
    _multi_table(RUNS_3).to_csv(tmp_path / "contacts_per_frame.csv", index=False)
    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=False)

    real = json.loads((tmp_path / "session_summary_real.json").read_text(encoding="utf-8"))["real_summary"]
    assert set(real["events_by_type"]) == set(CONTACT_TYPES)                         # no NC entry
    assert real["total_events"] == 2 == sum(e["count"] for e in real["events_by_type"].values())
    assert real["no_contact"]["spans"] == 5
    impact = json.loads((tmp_path / "session_summary_real.json").read_text(encoding="utf-8"))["filtering_impact"]
    raw_bouts = sum(json.loads((tmp_path / "session_summary_real.json").read_text(
        encoding="utf-8"))["raw_summary"]["bouts_by_type"].values())
    assert impact["bouts_removed"] == raw_bouts - 2

    log = (tmp_path / "event_log.txt").read_text(encoding="utf-8")
    nc_line = next(l for l in log.splitlines() if l.startswith("  NC "))
    assert " 5 spans," in nc_line and "events" not in nc_line
    assert "N2N  (Nose-to-nose):  1 events," in log
    assert f"Raw bouts: {raw_bouts} -> Real contact events: 2 " in log


# ── pair contact matrix ────────────────────────────────────────────────────

RUNS_3_FOL = {**RUNS_3, "1_2": (20, 70, "FOL", "j")}


def test_pair_contact_matrix_counts_durations_and_fol():
    df = _multi_table(RUNS_3_FOL)
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)
    session_sec = session_frames(df) / FPS
    assert n_animals_in_table(df) == 3

    pm = pair_contact_matrix(ev, n_animals_in_table(df), FPS, session_sec)
    assert list(pm.columns) == PAIR_MATRIX_COLUMNS
    assert list(pm.pair_key) == ["0_1", "0_2", "1_2"]
    pm = pm.set_index("pair_label")
    assert (pm.loc["R1-R2", "animal_a"], pm.loc["R1-R2", "animal_b"]) == ("R1", "R2")
    assert (pm.loc["R2-R3", "animal_a"], pm.loc["R2-R3", "animal_b"]) == ("R2", "R3")
    min_ = session_sec / 60

    n2n = pm.loc["R1-R2"]                                   # one N2N of 60 frames = 2 s
    assert n2n.n_events_total == 1 and n2n.n_events_N2N == 1 and n2n.n_events_N2AG == 0
    assert n2n.contact_sec == pytest.approx(2.0)
    assert n2n.contact_pct_of_session == pytest.approx(2.0 / session_sec * 100, abs=1e-2)
    assert n2n.fol_sec == 0.0
    assert n2n.events_per_min == pytest.approx(1 / min_, abs=1e-2)
    assert n2n.contact_sec_per_min == pytest.approx(2.0 / min_, abs=1e-2)

    assert pm.loc["R1-R3", "n_events_N2AG"] == 1 and pm.loc["R1-R3", "contact_sec"] == pytest.approx(2.0)

    fol = pm.loc["R2-R3"]                                   # FOL counts as an event, not as contact time
    assert fol.n_events_total == 1 and fol.n_events_FOL == 1
    assert fol.contact_sec == 0.0 and fol.contact_pct_of_session == 0.0
    assert fol.fol_sec == pytest.approx(50 / FPS, abs=1e-3)
    assert fol.contact_sec_per_min == 0.0 and fol.events_per_min == pytest.approx(1 / min_, abs=1e-2)


def test_pair_contact_matrix_lists_pairs_without_events():
    df = _multi_table(RUNS_3)
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)                      # NC spans are in here too
    pm = pair_contact_matrix(ev, 4, FPS, 200 / FPS)         # a 4th animal that never touched anyone
    assert len(pm) == 6                                     # C(4, 2)
    quiet = pm[pm.pair_key.isin(["0_3", "1_3", "2_3"])]
    assert len(quiet) == 3
    assert (quiet.n_events_total == 0).all() and (quiet.contact_sec == 0).all()
    assert (quiet.events_per_min == 0).all()
    # NC spans never count as events
    assert pm.n_events_total.sum() == 2 and (ev.contact_type == "NC").sum() > 0


def test_pair_contact_matrix_without_events_or_animals():
    empty = pair_contact_matrix(pd.DataFrame(), 3, FPS, 10.0)
    assert len(empty) == 3 and (empty.n_events_total == 0).all()
    assert list(empty.columns) == PAIR_MATRIX_COLUMNS
    assert pair_contact_matrix(pd.DataFrame(), 1, FPS, 10.0).empty
    assert pair_contact_matrix(pd.DataFrame(), 0, FPS, 10.0).empty
    df = _table()
    assert n_animals_in_table(df.drop(columns=["pair_key"])) == 0
    assert pair_contact_matrix(pd.DataFrame(), 3, FPS, 0.0).contact_pct_of_session.eq(0).all()


def test_event_log_header_counts_contact_events_not_nc_spans(tmp_path):
    _multi_table(RUNS_3).to_csv(tmp_path / "contacts_per_frame.csv", index=False)
    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=False)
    df = _multi_table(RUNS_3)
    _, real = compute_labels(df, DEFAULT_CONFIG, FPS)
    ev = extract_events(df, real, FPS)
    n_nc = int((ev.contact_type == "NC").sum())
    assert n_nc == 5                                        # 2 + 2 + 1 gaps around the 2 contact events

    log = (tmp_path / "event_log.txt").read_text(encoding="utf-8")
    assert f"Contact events: 2 (plus {n_nc} no-contact spans) | " in log
    assert "Total events" not in log
    # 4 s of contact (2 events x 2 s) over 3 pairs x 6.667 s = 20 s of pair-time
    assert "Contact time: 4.0 s = 20.0% of pair-time" in log
    assert "Pairs: 3 (contact time is summed over pairs; % is of pair-time = session x pairs)" in log


def test_event_log_header_two_animals_is_percent_of_session(tmp_path):
    df = _multi_table(RUNS_3)
    df = df[df.pair_key == "0_1"].reset_index(drop=True)    # 2 animals: one pair
    df.to_csv(tmp_path / "contacts_per_frame.csv", index=False)
    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=False)
    log = (tmp_path / "event_log.txt").read_text(encoding="utf-8")
    assert "Contact events: 1 (plus 2 no-contact spans) | Contact time: 2.0 s = 30.0% of session" in log
    assert "Pairs:" not in log
    pm = pd.read_csv(tmp_path / "pair_contact_matrix.csv")
    assert list(pm.pair_label) == ["R1-R2"] and pm.n_events_total.iloc[0] == 1


# ── investigation bouts ────────────────────────────────────────────────────

def _label_table(seqs, n_frames=100, role="i"):
    """Interleaved per-frame table plus the cleaned labels, built directly (no smoothing).

    seqs = {pair: [(label, n_frames), ...]} laid out from frame 0, "" = no contact; each pair
    is padded with "" up to n_frames. Returns (df, real_types) aligned row by row; the raw
    contact_type equals the cleaned label, N2AG / N2B frames carry investigator role `role`.
    """
    labels = {}
    for pk in PAIRS:
        seq = [lab for lab, n in seqs.get(pk, []) for _ in range(n)]
        labels[pk] = seq + [""] * (n_frames - len(seq))
    rows, real = [], []
    for f in range(n_frames):
        for pk in PAIRS:
            lab = labels[pk][f]
            rows.append({
                "frame_idx": f, "time_sec": f / FPS, "zone": "independent", "pair_key": pk,
                "contact_type": lab or "none",
                "investigator_role": role if lab in ("N2AG", "N2B") else "",
                "score_n2n": 0.7 if lab == "N2N" else 0.0,
                "score_n2ag": 0.9 if lab == "N2AG" else 0.0,
                "score_n2b": 0.5 if lab == "N2B" else 0.0,
            })
            real.append(lab)
    return pd.DataFrame(rows), np.array(real, dtype=object)


def test_investigation_bout_merges_n2ag_n2b_n2ag_into_one_bout():
    df, real = _label_table({"0_1": [("", 10), ("N2AG", 20), ("N2B", 5), ("N2AG", 20)]})
    ev = extract_events(df, real, FPS)
    assert [c for c in ev.contact_type if c != "NC"] == ["N2AG", "N2B", "N2AG"]   # events are fragmented

    bouts = investigation_bouts(df, real, FPS)
    assert list(bouts.columns) == INVESTIGATION_COLUMNS
    assert len(bouts) == 1
    b = bouts.iloc[0]
    assert b.bout_id == 0 and b.pair_key == "0_1" and b.pair_label == "R1-R2"
    assert b.start_frame == 10 and b.end_frame == 54
    assert b.start_time_sec == pytest.approx(10 / FPS, abs=1e-4)
    assert b.end_time_sec == pytest.approx(55 / FPS, abs=1e-4)
    assert b.duration_sec == pytest.approx(45 / FPS, abs=1e-4)
    assert b.n2ag_sec == pytest.approx(40 / FPS, abs=1e-4)
    assert b.n2b_sec == pytest.approx(5 / FPS, abs=1e-4)
    assert b.n2n_sec == 0.0 and b.other_sec == 0.0
    assert b.dominant_type == "N2AG" and b.n_subtype_switches == 2
    assert b.investigator_slot == 0                       # role "i" of pair 0_1
    assert b.peak_score == 0.9                            # the dominant type's score column


def test_investigation_bout_bridges_short_gaps_only():
    df, real = _label_table({"0_1": [("N2AG", 20), ("", 2), ("N2AG", 20)]})
    bouts = investigation_bouts(df, real, FPS)
    assert len(bouts) == 1
    b = bouts.iloc[0]
    assert b.start_frame == 0 and b.end_frame == 41
    assert b.duration_sec == pytest.approx(42 / FPS, abs=1e-4)
    assert b.other_sec == pytest.approx(2 / FPS, abs=1e-4)          # the bridged gap
    assert b.n2ag_sec == pytest.approx(40 / FPS, abs=1e-4)
    assert b.n_subtype_switches == 0

    df, real = _label_table({"0_1": [("N2AG", 20), ("", 5), ("N2AG", 20)]})
    bouts = investigation_bouts(df, real, FPS)
    assert len(bouts) == 2
    assert list(bouts.start_frame) == [0, 25] and list(bouts.end_frame) == [19, 44]
    assert (bouts.other_sec == 0).all()                              # a bout never ends on a gap

    # the limit itself: <= max_gap frames are bridged, one more splits
    for gap, expected in ((3, 1), (4, 2)):
        df, real = _label_table({"0_1": [("N2B", 10), ("", gap), ("N2B", 10)]})
        assert len(investigation_bouts(df, real, FPS, max_gap=3)) == expected
    df, real = _label_table({"0_1": [("N2B", 10), ("", 5), ("N2B", 10)]})
    assert len(investigation_bouts(df, real, FPS, max_gap=5)) == 1
    assert len(investigation_bouts(df, real, FPS, max_gap=4)) == 2
    assert len(investigation_bouts(df, real, FPS, max_gap=0)) == 2


def test_investigation_bout_nc_label_is_a_gap_too():
    df, real = _label_table({"0_1": [("N2AG", 20), ("", 2), ("N2AG", 20)]})
    real = np.where(real == "", "NC", real).astype(object)          # run_postprocess writes NC
    bouts = investigation_bouts(df, real, FPS)
    assert len(bouts) == 1 and bouts.iloc[0].other_sec == pytest.approx(2 / FPS, abs=1e-4)


def test_investigation_bouts_never_include_sbs_or_fol_frames():
    df, real = _label_table({"0_1": [("SBS", 40)], "0_2": [("FOL", 40)]})
    bouts = investigation_bouts(df, real, FPS)
    assert bouts.empty and list(bouts.columns) == INVESTIGATION_COLUMNS

    # a long SBS stretch between two investigations separates them; none of it ends up in a bout
    df, real = _label_table({"0_1": [("N2AG", 10), ("SBS", 30), ("N2B", 10)]})
    bouts = investigation_bouts(df, real, FPS)
    assert len(bouts) == 2
    assert list(zip(bouts.start_frame, bouts.end_frame)) == [(0, 9), (40, 49)]
    assert (bouts.other_sec == 0).all() and list(bouts.dominant_type) == ["N2AG", "N2B"]

    # a short SBS stretch is only a bridged gap: counted as other_sec, not as a sub-type
    df, real = _label_table({"0_1": [("N2AG", 10), ("SBS", 3), ("N2B", 10)]})
    bouts = investigation_bouts(df, real, FPS)
    assert len(bouts) == 1
    b = bouts.iloc[0]
    assert b.other_sec == pytest.approx(3 / FPS, abs=1e-4) and b.n_subtype_switches == 1
    assert b.n2ag_sec + b.n2b_sec + b.n2n_sec == pytest.approx(20 / FPS, abs=1e-4)


def test_investigation_bout_ids_are_unique_and_ordered_by_start_then_pair():
    df, real = _label_table({
        "0_1": [("", 40), ("N2N", 10), ("", 20), ("N2B", 10)],
        "0_2": [("", 10), ("N2AG", 15)],
        "1_2": [("", 10), ("N2B", 15), ("", 30), ("N2N", 10)],
    })
    bouts = investigation_bouts(df, real, FPS)
    assert len(bouts) == 5
    assert bouts.bout_id.is_unique and list(bouts.bout_id) == list(range(5))
    keys = list(zip(bouts.start_frame, bouts.pair_key))
    # start frame first, ties broken by pair (0_2 before 1_2), not by the order the pairs are listed
    assert keys == [(10, "0_2"), (10, "1_2"), (40, "0_1"), (55, "1_2"), (70, "0_1")]


def test_investigation_bout_dominant_type_ties_and_investigator():
    # N2N > N2AG > N2B when frames tie
    df, real = _label_table({"0_1": [("N2B", 10), ("N2N", 10)]})
    assert investigation_bouts(df, real, FPS).iloc[0].dominant_type == "N2N"
    df, real = _label_table({"0_1": [("N2B", 10), ("N2AG", 10)]})
    assert investigation_bouts(df, real, FPS).iloc[0].dominant_type == "N2AG"
    # more N2B frames than N2AG: N2B dominates and its score column gives the peak
    df, real = _label_table({"0_1": [("N2B", 11), ("N2AG", 10)]})
    b = investigation_bouts(df, real, FPS).iloc[0]
    assert b.dominant_type == "N2B" and b.peak_score == 0.5

    # N2N is symmetric: a pure N2N bout has no investigator; role "j" of pair 1_2 is slot 2
    df, real = _label_table({"0_1": [("N2N", 12)], "1_2": [("N2B", 8), ("N2AG", 8)]}, role="j")
    bouts = investigation_bouts(df, real, FPS).set_index("pair_key")
    assert bouts.loc["0_1", "investigator_slot"] == ""
    assert bouts.loc["1_2", "investigator_slot"] == 2


def test_investigation_bouts_without_pair_key_and_empty_input():
    df, real = _label_table({"0_1": [("N2AG", 15), ("N2B", 15)]})
    keep = (df.pair_key == "0_1").to_numpy()
    solo = df[keep].drop(columns=["pair_key"]).reset_index(drop=True)
    bouts = investigation_bouts(solo, real[keep], FPS)
    assert len(bouts) == 1 and bouts.iloc[0].pair_key == "" and bouts.iloc[0].pair_label == ""
    assert bouts.iloc[0].n_subtype_switches == 1

    df, real = _label_table({})
    assert investigation_bouts(df, real, FPS).empty


def test_run_postprocess_investigation_block_for_a_sliding_nose(tmp_path):
    import json
    # N2AG, then the nose slides to the flank (N2B), then back: 3 events but one investigation
    df, _ = _label_table({"0_1": [("", 10), ("N2AG", 20), ("N2B", 12), ("N2AG", 20)]})
    df.to_csv(tmp_path / "contacts_per_frame.csv", index=False)
    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=False)

    ev = pd.read_csv(tmp_path / "contacts_real_events.csv")
    assert list(ev[ev.contact_type != "NC"].contact_type) == ["N2AG", "N2B", "N2AG"]   # unchanged
    ib = pd.read_csv(tmp_path / "investigation_bouts.csv")
    assert len(ib) == 1 and ib.n_subtype_switches.iloc[0] == 2 and ib.dominant_type.iloc[0] == "N2AG"

    log = (tmp_path / "event_log.txt").read_text(encoding="utf-8")
    assert "--- Investigation bouts (N2N + N2AG + N2B merged, gaps <= 3 frames) ---\n" \
           "  Bouts: 1 | total 1.7s | dominant: N2AG 1 | mean sub-type switches per bout: 2.0\n" \
           "  Time by sub-type inside bouts: N2N 0.0s | N2AG 1.3s | N2B 0.4s\n" in log
    assert log.index("--- Summary by type ---") < log.index("--- Investigation bouts") \
        < log.index("--- Per-animal summary ---")
    summary = json.loads((tmp_path / "session_summary_real.json").read_text(encoding="utf-8"))
    inv = summary["investigation"]
    assert inv["n_bouts"] == 1 and inv["total_sec"] == pytest.approx(1.73, abs=1e-2)
    assert inv["by_dominant_type"] == {"N2N": 0, "N2AG": 1, "N2B": 0}
    assert inv["mean_subtype_switches"] == 2.0
    assert inv["time_by_subtype_sec"]["N2B"] == pytest.approx(0.4, abs=1e-2)


def test_investigation_max_gap_is_read_from_the_config(tmp_path):
    import json
    df, _ = _label_table({"0_1": [("N2AG", 15), ("", 5), ("N2AG", 15)]})
    df.to_csv(tmp_path / "contacts_per_frame.csv", index=False)
    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=False)
    # with the default gap of 3 frames the 5-frame pause splits the investigation
    assert len(pd.read_csv(tmp_path / "investigation_bouts.csv")) == 2

    run_postprocess(tmp_path, FPS, make_reports=False, consolidate=False,
                    config_overrides={"gap_bridging": {"max_gap": 0}, "investigation": {"max_gap": 5}})
    ib = pd.read_csv(tmp_path / "investigation_bouts.csv")
    assert len(ib) == 1 and ib.other_sec.iloc[0] == pytest.approx(5 / FPS, abs=1e-4)
    log = (tmp_path / "event_log.txt").read_text(encoding="utf-8")
    assert "gaps <= 5 frames" in log
    summary = json.loads((tmp_path / "session_summary_real.json").read_text(encoding="utf-8"))
    assert summary["parameters"]["investigation_max_gap"] == 5
