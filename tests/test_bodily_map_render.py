"""Deterministic tests for the bodily-map template, filtering and marker logic."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.bodily_map import render as R  # noqa: E402
from src.common.bodily_map import template as T  # noqa: E402


def rec(focal, other, region="nose", ctype="N2N", t=0.0, s=0.0, dur=0.1):
    return {"frame_idx": 0, "time_sec": 0.0, "focal_mouse": focal, "other_mouse": other, "direction": "passive",
            "contact_type": ctype, "region": region, "t": t, "s": s, "dur_sec": dur}


RECS = [rec("R1", "R2"), rec("R1", "R3", "head"), rec("R2", "R1", "flank_R", "SBS", 0.3, 0.1),
        rec("R3", "R1", "rump", "N2B", 0.5), rec("R3", "R2", "tail", "N2AG", 0.9), rec("R2", "R3")]


def test_template_regions_nonempty_and_in_range():
    for name in T.ALL_REGIONS:
        polys = T.REGION_POLYS[name]
        assert polys, name
        for v in polys:
            assert v.ndim == 2 and v.shape[1] == 2 and len(v) >= 3, name
            assert v[:, 0].min() >= T.XLIM[0] and v[:, 0].max() <= T.XLIM[1], name
            assert v[:, 1].min() >= T.YLIM[0] and v[:, 1].max() <= T.YLIM[1], name


def test_template_geometry_is_consistent():
    assert T.REGION_POLYS["flank_R"][0][:, 1].mean() > 0 > T.REGION_POLYS["flank_L"][0][:, 1].mean()
    assert T.REGION_POLYS["ear_R"][0][:, 1].mean() > 0 > T.REGION_POLYS["ear_L"][0][:, 1].mean()
    assert T.REGION_POLYS["tail"][0][:, 0].max() <= 1.0 + 1e-9
    # head/nose are 2D areas, not a point
    head = np.vstack([T.REGION_POLYS["nose"][0], T.REGION_POLYS["head"][0]])
    assert np.ptp(head[:, 1]) > 0.1
    # region order along the body
    xs = [T.REGION_POLYS[r][0][:, 0].mean() for r in ("nose", "head", "rump", "anogenital", "tail")]
    assert xs == sorted(xs)


def test_region_of_point_matches_locate_cuts():
    assert T.region_of_point(0.0, 0.0) == "nose"
    assert T.region_of_point(0.3, 0.2) == "flank_R"
    assert T.region_of_point(0.3, -0.2) == "flank_L"
    assert T.region_of_point(0.3, 0.0) == "mid_back"
    assert T.region_of_point(0.9, 0.0) == "tail"


def test_passive_filter():
    rows = R.select_passive(RECS, "R1")
    assert len(rows) == 2 and all(r["focal_mouse"] == "R1" for r in rows)


def test_active_filter():
    rows = R.select_active(RECS, "R1", "R2")
    assert len(rows) == 1 and rows[0]["other_mouse"] == "R1" and rows[0]["focal_mouse"] == "R2"
    assert R.select_active(RECS, "R2", "R2") == []


def test_active_combined_filter():
    rows = R.select_active_combined(RECS, "R1")
    assert len(rows) == 2 and all(r["other_mouse"] == "R1" for r in rows)
    assert {r["focal_mouse"] for r in rows} == {"R2", "R3"}


def test_region_table_sums():
    tab = R.region_table(R.select_passive(RECS, "R1"))
    assert tab["nose"]["count"] == 1 and tab["head"]["count"] == 1
    assert abs(sum(v["dur_sec"] for v in tab.values()) - 0.2) < 1e-9


def test_marker_assignment_stable_and_distinct():
    mice = [f"R{i}" for i in range(1, 7)]
    shapes = [R.marker_for(m) for m in mice]
    assert shapes == ["o", "s", "^", "D", "*", "P"]
    assert len(set(shapes)) == 6
    assert shapes == [R.marker_for(m) for m in reversed(mice)][::-1]
    assert R.color_for("R1") == (1.0, 0.0, 0.0) and R.color_for("R3") == (0.0, 0.0, 1.0)
