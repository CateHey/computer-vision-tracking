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
