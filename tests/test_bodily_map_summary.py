"""Deterministic tests for the bodily-map summary numbers (no pixel assertions)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.bodily_map import summary as S  # noqa: E402


def rec(focal, other, ctype, region="nose", dur=1.0):
    return {"frame_idx": 0, "time_sec": 0.0, "focal_mouse": focal, "other_mouse": other, "direction": "x",
            "contact_type": ctype, "region": region, "t": 0.0, "s": 0.0, "dur_sec": dur}


RECS = [rec("R1", "R2", "N2AG", "anogenital", 3.0),   # R2 active on R1
        rec("R2", "R1", "N2B", "mid_back", 1.0),      # R1 active on R2
        rec("R1", "R2", "N2B", "flank_R", 1.0),       # R2 active on R1
        rec("R1", "R2", "N2N", "nose", 10.0),         # mutual
        rec("R2", "R1", "SBS", "flank_L", 5.0)]       # mutual
MICE = ["R1", "R2"]


def test_active_passive_directional_only():
    ap = S.active_passive(RECS, MICE)
    assert ap["R2"]["active"] == 4.0 and ap["R2"]["passive"] == 1.0
    assert ap["R1"]["active"] == 1.0 and ap["R1"]["passive"] == 4.0


def test_pct_initiated():
    ap = S.active_passive(RECS, MICE)
    assert abs(ap["R2"]["pct_initiated"] - 80.0) < 1e-9
    assert abs(ap["R1"]["pct_initiated"] - 20.0) < 1e-9
    assert S.active_passive([rec("R1", "R2", "N2N")], MICE)["R1"]["pct_initiated"] is None


def test_mutual_is_n2n_plus_sbs():
    ap = S.active_passive(RECS, MICE)
    assert ap["R1"]["mutual"] == 15.0 and ap["R2"]["mutual"] == 15.0


def test_region_preference_normalized():
    pref = S.region_preference(RECS, MICE)
    for m in MICE:
        assert abs(sum(pref[m].values()) - 1.0) < 1e-9
    assert "flank" in pref["R1"] and "flank_R" not in pref["R1"]
    assert S.region_preference([], ["R1"])["R1"]["nose"] == 0.0


def test_contacts_per_mouse_credits_both():
    d = S.contacts_per_mouse(RECS, MICE)
    assert d["R1"]["N2N"]["dur"] == 10.0 and d["R2"]["N2N"]["dur"] == 10.0
