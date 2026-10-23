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
