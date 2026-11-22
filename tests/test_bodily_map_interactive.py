import json

from src.common.bodily_map import template as T
from src.common.bodily_map.interactive import (TYPE_CODES, build_payload, payload_script, session_end,
                                               template_geometry, ts_to_px, widget)


def _r(f, o, ct, region="head", t=0.1, s=0.02, time=1.0, d=0.1):
    return dict(frame_idx=int(time * 30), time_sec=time, focal_mouse=f, other_mouse=o, direction="passive",
                contact_type=ct, region=region, t=t, s=s, dur_sec=d)


RECS = [_r("R2", "R1", "N2B", "flank_R", 0.3, 0.1, 5.0), _r("R1", "R2", "N2N", "nose", 0.01, 0, 2.0),
        _r("R3", "R1", "SBS", "flank_L", 0.3, -0.1, 3.0), _r("R1", "R3", "N2AG", "anogenital", 0.55, 0, 4.0)]


def test_payload_keys_and_columns():
    p = build_payload(RECS, ["R1", "R2", "R3"])
    assert {"t_end", "mice", "colors", "types", "records", "template"} <= set(p)
    cols = p["records"]
    assert set(cols) == {"time", "focal", "other", "type", "region", "tt", "ss", "dur"}
    assert len({len(v) for v in cols.values()}) == 1 and len(cols["time"]) == 4
    assert cols["time"] == sorted(cols["time"])
    assert len(p["colors"]) == 3 and all(c.startswith("#") and len(c) == 7 for c in p["colors"])
    assert p["t_end"] >= 5.0


def test_type_and_mouse_codes():
    p = build_payload(RECS, ["R1", "R2", "R3"])
    assert TYPE_CODES == ["N2N", "N2AG", "N2B", "SBS"]
    c = p["records"]  # sorted by time: N2N(2), SBS(3), N2AG(4), N2B(5)
    assert c["type"] == [0, 3, 1, 2]
    assert c["focal"][0] == 0 and c["other"][0] == 1
    assert p["regions"][c["region"][0]] == "nose"


def test_template_extents_match_template_py():
    g = template_geometry()
    assert (g["tmin"], g["tmax"]) == T.XLIM and (g["smin"], g["smax"]) == T.YLIM and g["aspect"] == T.ASPECT
    assert len(g["body"]) > 50 and len(g["tail"]) > 10 and len(g["ears"]) == 2
    assert set(g["regions"]) >= {"nose", "head", "rump", "anogenital"}


def test_ts_to_px_corners():
    w = 1100.0
    x0, y0 = ts_to_px(T.XLIM[0], T.YLIM[1], w)
    x1, y1 = ts_to_px(T.XLIM[1], T.YLIM[0], w)
    assert (x0, y0) == (0, 0) and abs(x1 - w) < 1e-9
    assert abs(y1 - w * T.ASPECT * 0.84 / 1.10) < 1e-6
    assert ts_to_px(0.0, 0.0, w)[1] == y1 / 2  # s=0 on the midline


def test_empty_and_script_safe(tmp_path):
    p = build_payload([], ["R1"])
    assert p["records"]["time"] == [] and p["t_end"] == 0
    s = payload_script(build_payload(RECS, ["R1", "R2", "R3"]))
    assert s.startswith("<script>const BM_DATA=") and json.loads(s[len("<script>const BM_DATA="):-len(";</script>")])
    (tmp_path / "provenance.json").write_text('{"duration_sec": 99.5}')
    assert session_end(tmp_path, RECS) == 99.5
    assert 'data-focal="R2"' in widget("R2")
