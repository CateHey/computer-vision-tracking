"""Tests for scripts/validate_contacts.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import validate_contacts as vc  # noqa: E402


def _events(n):
    return pd.DataFrame({"start_frame": np.arange(n) * 100, "end_frame": np.arange(n) * 100 + 10,
                         "duration_sec": np.linspace(0.1, 3.0, n)})


def test_tertiles_even():
    t = vc.assign_tertiles(pd.Series(np.linspace(0.1, 3, 30)))
    assert t.value_counts().to_dict() == {"short": 10, "medium": 10, "long": 10}


def test_sample_stratified_and_deterministic():
    df = _events(60)
    a = vc.stratified_sample(df, 15, 42)
    b = vc.stratified_sample(df, 15, 42)
    c = vc.stratified_sample(df, 15, 7)
    assert len(a) == 15
    assert a["duration_tertile"].value_counts().to_dict() == {"short": 5, "medium": 5, "long": 5}
    assert list(a.index) == list(b.index)
    assert list(a.index) != list(c.index)


def test_sample_takes_all_when_few():
    assert len(vc.stratified_sample(_events(5), 15, 1)) == 5


def test_sample_quota_when_uneven():
    assert len(vc.stratified_sample(_events(20), 19, 1)) == 19


def test_wilson():
    lo, hi = vc.wilson_ci(8, 10)
    assert lo == pytest.approx(0.490, abs=0.01) and hi == pytest.approx(0.943, abs=0.01)
    assert np.isnan(vc.wilson_ci(0, 0)[0])
    assert vc.wilson_ci(10, 10)[1] == 1.0


def test_precision_uses_only_the_stratified_sample():
    rows = []
    for i, (v, ins) in enumerate([("real", True), ("real", True), ("flicker", True),
                                  ("flicker", False), ("flicker", False)]):
        rows.append({"sample_id": f"S01.{i + 1}", "contact_type": "N2N", "verdict": v, "in_sample": ins,
                     "duration_sec": 0.5, "peak_score": 0.9, "duration_tertile": "short"})
    s = vc.compute_summary(pd.DataFrame(rows), vc.DEFAULT_CONFIG)
    assert s["n_reviewed"] == 3 and s["per_type"]["N2N"]["precision"] == pytest.approx(2 / 3)
    assert s["extras"]["n_reviewed"] == 2 and s["extras"]["per_type"]["N2N"]["precision"] == 0.0


def test_score_on_synthetic_review(tmp_path):
    verdicts = ["real", "real", "flicker", "wrong_type", "unsure", "tracking_error", "real", "flicker"]
    rows = []
    for i in range(len(verdicts)):
        rows.append({"sample_id": f"V{i + 1:03d}", "clip": f"clips/{i}.mp4", "thumbnail": "t.png",
                     "pair_label": "R1-R2", "contact_type": "N2N" if i < 5 else "SBS",
                     "contact_label": "x", "start_time": "00:00.0", "end_time": "00:01.0",
                     "duration_sec": 0.2 + 0.4 * i, "duration_frames": 5,
                     "duration_tertile": vc.TERTILES[i % 3], "peak_score": 0.5 + 0.05 * i})
    p = tmp_path / "review.xlsx"
    vc.write_review_xlsx(rows, p)
    df = pd.read_excel(p, sheet_name="review")
    df["verdict"] = verdicts
    df["correct_type"] = [None, None, None, "N2B", None, None, None, None]
    with pd.ExcelWriter(p) as w:
        df.to_excel(w, sheet_name="review", index=False)
    assert vc.main(["score", str(p)]) == 0
    s = json.loads((tmp_path / "validation_summary.json").read_text())
    n2n = s["per_type"]["N2N"]
    assert n2n["n_reviewed"] == 5 and n2n["n_real"] == 2 and n2n["n_decided"] == 4
    assert n2n["precision"] == 0.5
    assert s["wrong_type_confusion"] == {"N2N": {"N2B": 1}}
    assert (tmp_path / "validation_summary.md").exists()
    assert (tmp_path / "validation_precision.png").exists()


def _ev(pk, ct, a, b):
    return {"pair_key": pk, "contact_type": ct, "start_frame": a, "end_frame": b, "n_frames": b - a + 1,
            "duration_sec": (b - a + 1) / 30, "peak_score": 0.9, "duration_tertile": "short"}


def test_build_scenes_groups_overlapping_events_and_adds_extras():
    events = pd.DataFrame([_ev("0_1", "N2N", 10, 40), _ev("1_2", "SBS", 30, 60),
                           _ev("0_2", "N2AG", 45, 80), _ev("0_1", "N2N", 200, 230)])
    sample = events.iloc[[0, 1, 3]]            # the N2AG was not sampled
    scenes = vc.build_scenes(sample, events, pad=15, nframes=1000)
    assert [s["scene_id"] for s in scenes] == ["S01", "S02"]
    s1 = scenes[0]
    assert (s1["a"], s1["b"]) == (0, 95)       # grown to cover the extra N2AG (80 + 15)
    assert list(s1["events"]["contact_type"]) == ["N2N", "SBS", "N2AG"]
    assert list(s1["events"]["in_sample"]) == [True, True, False]
    assert not s1["events"]["partial"].any()
    assert len(scenes[1]["events"]) == 1 and (scenes[1]["a"], scenes[1]["b"]) == (185, 245)


def test_build_scenes_splits_a_long_chain_into_non_overlapping_segments():
    events = pd.DataFrame([_ev("0_1", "N2N", 10, 40), _ev("1_2", "SBS", 30, 60), _ev("0_2", "N2AG", 45, 80)])
    scenes = vc.build_scenes(events.iloc[[0, 1]], events, pad=15, nframes=1000, max_frames=76)
    # chain [0, 95] = 96 frames > 76 -> two segments of 48 frames, no repeated footage
    assert [(s["a"], s["b"]) for s in scenes] == [(0, 47), (48, 95)]
    s1, s2 = scenes
    assert list(s1["events"]["contact_type"]) == ["N2N", "SBS"]      # most of each lies in segment 1
    assert list(s1["events"]["partial"]) == [False, True]            # SBS continues into segment 2
    assert list(s2["events"]["contact_type"]) == ["N2AG"]
    assert bool(s2["events"].iloc[0]["partial"]) is True and not bool(s2["events"].iloc[0]["in_sample"])
    assert list(s2["visible"]["contact_type"]) == ["SBS", "N2AG"]   # SBS still highlighted there
    # each event has exactly one review row
    keys = [k for s in scenes for k in vc._event_keys(s["events"])]
    assert len(keys) == len(set(keys)) == 3


def test_build_scenes_splits_sampled_events_too():
    events = pd.DataFrame([_ev("0_1", "N2N", 10, 40), _ev("1_2", "SBS", 30, 60)])
    scenes = vc.build_scenes(events, events, pad=15, nframes=1000, max_frames=50)
    assert [(s["a"], s["b"]) for s in scenes] == [(0, 37), (38, 75)]
    assert [list(s["events"]["contact_type"]) for s in scenes] == [["N2N"], ["SBS"]]
    assert all(s["events"]["partial"].all() for s in scenes)


# ------------------------------------------------------------------ NC windows (recall proxy)
FPS = 30.0


def _raw_events_table():
    """Events table as the post-processing writes it: NC rows fill the gaps between contacts.

    Pair 1_2 is deliberately inconsistent: one NC row covers a contact event (N2AG 300-360),
    which the sampler must still never hand out.
    """
    rows = [
        ("0_1", "NC", 0, 99), ("0_1", "N2N", 100, 159), ("0_1", "NC", 160, 399),
        ("0_1", "SBS", 400, 430), ("0_1", "NC", 431, 445), ("0_1", "FOL", 446, 480),
        ("0_1", "NC", 481, 799),
        ("0_2", "NC", 0, 799),
        ("1_2", "NC", 0, 799), ("1_2", "N2AG", 300, 360),
    ]
    return pd.DataFrame([{"pair_key": pk, "contact_type": ct, "start_frame": a, "end_frame": b,
                          "duration_sec": (b - a + 1) / FPS, "peak_score": 0.9 if ct != "NC" else np.nan}
                         for pk, ct, a, b in rows])


def _overlaps(w, contacts):
    same = contacts[contacts["pair_key"] == w["pair_key"]]
    return bool(((same["start_frame"] <= w["end_frame"]) & (same["end_frame"] >= w["start_frame"])).any())


def test_nc_spans_are_the_nc_rows_and_load_events_drops_them(tmp_path):
    raw = _raw_events_table()
    spans = vc.nc_spans(raw)
    assert len(spans) == 6 and set(spans.columns) == {"pair_key", "start_frame", "end_frame"}
    cdir = tmp_path / "contacts"
    cdir.mkdir()
    raw.to_csv(cdir / "contacts_real_events.csv", index=False)
    events, src = vc.load_events(tmp_path)
    assert src == "events" and set(events["contact_type"]) == {"N2N", "SBS", "FOL", "N2AG"}
    assert len(vc.load_nc_spans(tmp_path)) == 6


@pytest.mark.parametrize("seed", range(12))
def test_nc_windows_never_overlap_a_contact_event_of_the_same_pair(seed):
    raw = _raw_events_table()
    contacts, nc = vc._normalise_events(raw), vc.nc_spans(raw)
    wins = vc.sample_nc_windows(nc, contacts, None, FPS, 40, 3.0, 2.0, seed)
    assert len(wins) > 5 and (wins["contact_type"] == "NC").all()
    assert wins["nc_near"].isna().all()                          # no per-frame table: uniform
    for _, w in wins.iterrows():
        assert not _overlaps(w, contacts), dict(w)
        assert 60 <= w["n_frames"] <= 90                         # >= 2 * pad_sec, <= nc_window_sec
        assert not (w["pair_key"] == "0_1" and 431 <= w["start_frame"] <= 445)   # 15-frame span: too short
    for _, g in wins.groupby("pair_key"):                        # windows of a pair do not overlap
        g = g.sort_values("start_frame")
        assert (g["start_frame"].to_numpy()[1:] > g["end_frame"].to_numpy()[:-1]).all()


def test_nc_windows_are_seeded_and_capped():
    raw = _raw_events_table()
    contacts, nc = vc._normalise_events(raw), vc.nc_spans(raw)
    a = vc.sample_nc_windows(nc, contacts, None, FPS, 5, 3.0, 2.0, 1)
    b = vc.sample_nc_windows(nc, contacts, None, FPS, 5, 3.0, 2.0, 1)
    c = vc.sample_nc_windows(nc, contacts, None, FPS, 5, 3.0, 2.0, 2)
    assert len(a) == 5 and a.equals(b) and not a.equals(c)
    assert len(vc.sample_nc_windows(nc, contacts, None, FPS, 1000, 3.0, 2.0, 1)) < 1000   # only what exists
    assert vc.sample_nc_windows(nc, contacts, None, FPS, 0, 3.0, 2.0, 1).empty


def test_nc_windows_prefer_pairs_that_come_close():
    raw = _raw_events_table()
    contacts, nc = vc._normalise_events(raw), vc.nc_spans(raw)
    pf = pd.DataFrame([{"pair_key": pk, "frame_idx": f,
                        "centroid_dist_bl": 0.5 if (pk == "0_2" and 600 <= f <= 620) else 2.5}
                       for pk in ("0_1", "0_2", "1_2") for f in range(800)])
    pf_pairs = vc.index_per_frame(pf)
    for seed in range(5):
        one = vc.sample_nc_windows(nc, contacts, pf_pairs, FPS, 1, 3.0, 2.0, seed, near_bl=1.0)
        assert len(one) == 1 and bool(one.iloc[0]["nc_near"])
        assert one.iloc[0]["pair_key"] == "0_2" and one.iloc[0]["start_frame"] <= 600 <= one.iloc[0]["end_frame"]
        assert one.iloc[0]["min_centroid_dist_bl"] == pytest.approx(0.5)
    many = vc.sample_nc_windows(nc, contacts, pf_pairs, FPS, 6, 3.0, 2.0, 0)
    assert len(many) == 6 and int(many["nc_near"].sum()) == 1     # the single near window, then filled up
    assert not any(_overlaps(w, contacts) for _, w in many.iterrows())


def _nc_review_rows():
    """3 contact rows (N2N: real, real, flicker) and 4 NC windows (2 correct, 1 missed, 1 unsure)."""
    rows = []
    for i, (ct, v) in enumerate([("N2N", "real"), ("N2N", "real"), ("N2N", "flicker"),
                                 ("NC", "nc_correct"), ("NC", "nc_correct"), ("NC", "missed_contact"),
                                 ("NC", "unsure")]):
        nc = ct == "NC"
        rows.append({"sample_id": f"N{i - 2:02d}.1" if nc else f"S01.{i + 1}", "clip": f"clips/{i}.mp4",
                     "thumbnail": "t.png", "in_sample": True, "pair_label": "R1-R2", "contact_type": ct,
                     "contact_label": "x", "start_time": "00:00.0", "end_time": "00:01.0",
                     "duration_sec": 3.0 if nc else 0.5, "duration_frames": 15,
                     "duration_tertile": "" if nc else "short", "peak_score": None if nc else 0.9,
                     "nc_near": True if nc else None, "_verdict": v})
    return rows


def _fill(path, rows, verdicts, correct_type=None):
    vc.write_review_xlsx([{k: v for k, v in r.items() if k != "_verdict"} for r in rows], path)
    df = pd.read_excel(path, sheet_name="review")
    df["verdict"] = verdicts
    if correct_type is not None:
        df["correct_type"] = correct_type
    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name="review", index=False)


def test_review_workbook_offers_the_nc_verdicts(tmp_path):
    from openpyxl import load_workbook
    p = tmp_path / "review.xlsx"
    vc.write_review_xlsx([{k: v for k, v in r.items() if k != "_verdict"} for r in _nc_review_rows()], p)
    wb = load_workbook(p)
    lists = [dv.formula1 for dv in wb["review"].data_validations.dataValidation]
    union = [f for f in lists if "real" in f]
    assert len(union) == 1
    for v in vc.ALL_VERDICTS:
        assert v in union[0]
    assert {"nc_correct", "missed_contact", "unsure"} <= set(vc.ALL_VERDICTS)
    assert "missed_contact" in "\n".join(str(c.value) for c in wb["Instructions"]["A"])


def test_nc_windows_give_a_recall_proxy_and_leave_precision_alone(tmp_path):
    rows = _nc_review_rows()
    p = tmp_path / "review.xlsx"
    _fill(p, rows, [r["_verdict"] for r in rows], [None, None, None, None, None, "N2B", None])
    assert vc.main(["score", str(p)]) == 0
    s = json.loads((tmp_path / "validation_summary.json").read_text())
    rp = s["recall_proxy"]
    assert (rp["n_nc_windows"], rp["n_missed"], rp["n_unsure"], rp["n_decided"]) == (4, 1, 1, 3)
    assert rp["miss_rate"] == pytest.approx(1 / 3)
    # Wilson 95% interval of 1/3, by hand: centre (p + z2/2n) / (1 + z2/n) = 0.4269, half-width 0.3654
    assert rp["ci_low"] == pytest.approx(0.0615, abs=1e-4) and rp["ci_high"] == pytest.approx(0.7923, abs=1e-4)
    assert rp["missed_by_type"] == {"N2B": 1} and rp["n_near_windows"] == 4
    # contact figures see only the three contact rows
    assert s["n_reviewed"] == 3 and s["per_type"]["N2N"]["precision"] == pytest.approx(2 / 3)
    assert "NC" not in s["per_type"] and s["extras"]["n_reviewed"] == 0
    md = (tmp_path / "validation_summary.md").read_text()
    assert "Recall proxy" in md and "33%" in md and "N2B x1" in md


def test_score_with_only_nc_rows(tmp_path):
    rows = [r for r in _nc_review_rows() if r["contact_type"] == "NC"]
    p = tmp_path / "review.xlsx"
    _fill(p, rows, [r["_verdict"] for r in rows])
    assert vc.main(["score", str(p)]) == 0
    s = json.loads((tmp_path / "validation_summary.json").read_text())
    assert s["n_reviewed"] == 0 and s["recall_proxy"]["miss_rate"] == pytest.approx(1 / 3)
    assert s["recall_proxy"]["missed_by_type"] == {"UNSPECIFIED": 1}


def test_score_without_nc_rows_has_no_recall_proxy():
    rows = [{"sample_id": "S01.1", "contact_type": "N2N", "verdict": "real", "in_sample": True,
             "duration_sec": 0.5, "peak_score": 0.9, "duration_tertile": "short"}]
    s = vc.compute_summary(pd.DataFrame(rows), vc.DEFAULT_CONFIG)
    assert s["recall_proxy"] is None and "Recall proxy" not in vc.render_md(s)


# ------------------------------------------------------------------ inter-rater agreement
def _agreement_rows():
    """(sample_id, contact_type, rater A, rater B): 8 contact rows with 6 binary agreements, one
    row rater B called unsure, and two NC rows."""
    return [
        ("S01.1", "N2N", "real", "real"), ("S01.2", "N2N", "real", "real"),
        ("S01.3", "N2N", "real", "flicker"), ("S01.4", "N2N", "flicker", "wrong_type"),
        ("S02.1", "SBS", "real", "real"), ("S02.2", "SBS", "real", "real"),
        ("S02.3", "SBS", "flicker", "real"), ("S02.4", "SBS", "tracking_error", "flicker"),
        ("S03.1", "FOL", "real", "unsure"),
        ("N01.1", "NC", "nc_correct", "nc_correct"), ("N02.1", "NC", "nc_correct", "nc_correct"),
    ]


def _write_review(path, rows, col):
    """review.xlsx filled with column `col` of `rows` (2 = rater A, 3 = rater B)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = [{"sample_id": r[0], "clip": "c.mp4", "pair_label": "R1-R2", "contact_type": r[1],
             "duration_sec": 0.5, "peak_score": 0.9, "duration_tertile": "short", "in_sample": True} for r in rows]
    vc.write_review_xlsx(recs, path)
    df = pd.read_excel(path, sheet_name="review")
    df["verdict"] = [r[col] for r in rows]
    df["reviewer"] = "AA" if col == 2 else "BB"
    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name="review", index=False)


def test_cohen_kappa_matches_the_hand_computation():
    from fractions import Fraction
    a = [True, True, True, True, True, False, False, False]      # 5 yes, 3 no
    b = [True, True, True, True, False, True, False, False]      # 5 yes, 3 no; 6 of 8 agree
    st = vc.cohen_kappa(a, b)
    assert st["table"] == [[4, 1], [1, 2]] and st["n_agree"] == 6 and st["raw_agreement"] == 0.75
    p_o, p_a, p_b = Fraction(6, 8), Fraction(5, 8), Fraction(5, 8)
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)                      # 34/64
    expected = (p_o - p_e) / (1 - p_e)
    assert expected == Fraction(7, 15)
    assert st["kappa"] == pytest.approx(float(expected)) and st["p_expected"] == pytest.approx(float(p_e))
    assert vc.cohen_kappa([True, True], [True, True])["kappa"] is None        # one class only: undefined
    assert vc.cohen_kappa([], [])["kappa"] is None
    assert vc.cohen_kappa([True, False, True], [True, False, True])["kappa"] == 1.0
    assert vc.cohen_kappa([True, False], [False, True])["kappa"] == -1.0


def test_compare_two_review_workbooks(tmp_path):
    rows = _agreement_rows()
    pa, pb = tmp_path / "a" / "review.xlsx", tmp_path / "b" / "review.xlsx"
    _write_review(pa, rows, 2)
    _write_review(pb, rows[:-3] + rows[-2:], 3)         # rater B never saw S03.1
    assert vc.main(["score", "--compare", str(pa), str(pb)]) == 0
    r = json.loads((tmp_path / "a" / "interrater_summary.json").read_text())
    assert (tmp_path / "a" / "interrater_summary.md").exists()
    assert not (tmp_path / "b" / "interrater_summary.json").exists()           # written next to review A
    assert (r["n_matched"], r["n_only_a"], r["n_only_b"], r["n_unsure_excluded"]) == (10, 1, 0, 0)
    assert (r["reviewer_a"], r["reviewer_b"]) == ("AA", "BB")
    ov = r["overall"]                                    # contact rows only (NC rows are their own group)
    assert ov["n"] == 8 and ov["n_agree"] == 6 and ov["raw_agreement"] == pytest.approx(0.75)
    assert ov["table"] == [[4, 1], [1, 2]]
    assert ov["kappa"] == pytest.approx(7 / 15)         # (0.75 - 34/64) / (1 - 34/64), by hand
    assert r["per_type"]["N2N"]["kappa"] == pytest.approx(0.5)     # p_o 3/4, p_e 1/2
    assert r["per_type"]["SBS"]["kappa"] == pytest.approx(0.5)
    assert r["per_type"]["N2N"]["raw_agreement_full_verdict"] == pytest.approx(0.5)   # flicker vs wrong_type differ
    assert r["per_type"]["NC"]["raw_agreement"] == 1.0 and r["per_type"]["NC"]["kappa"] is None
    assert sorted(d["sample_id"] for d in r["disagreements"]) == ["S01.3", "S02.3"]
    md = (tmp_path / "a" / "interrater_summary.md").read_text()
    assert "0.467" in md and "undefined" in md and "S02.3" in md


def test_compare_excludes_unsure_rows(tmp_path):
    rows = _agreement_rows()
    pa, pb = tmp_path / "a.xlsx", tmp_path / "b.xlsx"
    _write_review(pa, rows, 2)
    _write_review(pb, rows, 3)
    r = vc.compare_reviews(pd.read_excel(pa, sheet_name="review"), pd.read_excel(pb, sheet_name="review"))
    assert r["n_matched"] == 11 and r["n_unsure_excluded"] == 1 and r["overall"]["n"] == 8
    assert "FOL" not in r["per_type"]


def test_score_needs_exactly_one_of_review_and_compare(tmp_path):
    with pytest.raises(SystemExit):
        vc.main(["score"])
    with pytest.raises(SystemExit):
        vc.main(["score", str(tmp_path / "a.xlsx"), "--compare", "x.xlsx", "y.xlsx"])
