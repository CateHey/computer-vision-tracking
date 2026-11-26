"""Structure tests for the unified report hub (no pixel assertions, synthetic fixture)."""
import json
import re
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.bodily_map import gallery  # noqa: E402
from src.common.bodily_map.hub import build_hub  # noqa: E402


def _make_run(tmp_path, with_validation=True):
    run = tmp_path / "2026-01-01_000000_x"
    (run / "contacts").mkdir(parents=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["contact_type", "contact_label", "events", "total_sec", "mean_duration_sec", "pct_of_session"])
    ws.append(["N2N", "Nose-to-nose", 3, 2.0, 0.7, 1.5])
    ws.append(["TOTAL", "All contact types", 3, 2.0, None, 1.5])
    ws = wb.create_sheet("Events")
    ws.append(["event_id", "pair_label", "contact_type", "start_time", "duration_sec"])
    ws.append([0, "R1-R2", "N2N", "0:01.0", 0.7])
    ws = wb.create_sheet("Parameters")
    ws.append(["section", "parameter", "value"])
    ws.append(["video", "fps", 30])
    wb.create_sheet("PerAnimal").append(["animal"])
    wb["PerAnimal"].append(["R1"])
    wb["PerAnimal"].append(["R2"])
    wb.save(run / "contacts" / "results.xlsx")
    (run / "provenance.json").write_text(json.dumps({"timestamp_utc": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
    if with_validation:
        (run / "validation").mkdir()
        (run / "validation" / "manifest.json").write_text(json.dumps({"n_samples": 5, "params": {}}), encoding="utf-8")
    return run


def test_hub_structure(tmp_path):
    run = _make_run(tmp_path)
    page = build_hub(run, run / "index.html").read_text(encoding="utf-8")
    for label in ("Overview", "Bodily Map", "Session Report", "Validation"):
        assert f">{label}</button>" in page
    assert 'id="hub-data"' in page and '"R1-R2"' in page and '"fps"' in page
    assert "Not yet reviewed" in page or "Review rows" in page
    assert "bm-theme" in page
    assert re.search(r"(src|href)\s*=\s*[\"']https?:", page) is None
    assert "://" not in re.sub(r"<footer>.*?</footer>", "", page, flags=re.S).replace("http://www.w3.org", "")


def test_hub_iframe_embed_when_gallery_present(tmp_path):
    run = _make_run(tmp_path)
    (run / "bodily_map").mkdir()
    (run / "bodily_map" / "index.html").write_text("x", encoding="utf-8")
    page = build_hub(run, run / "index.html").read_text(encoding="utf-8")
    assert 'src="bodily_map/index.html?embed=1"' in page


def test_hub_missing_pieces_are_graceful(tmp_path):
    run = tmp_path / "bare"
    run.mkdir()
    page = build_hub(run, run / "index.html").read_text(encoding="utf-8")
    assert "No validation/ folder" in page and "Events sheet is not" in page


def test_gallery_embed_hook():
    assert "embed=1" in gallery.JS + gallery.CSS + Path(gallery.__file__).read_text(encoding="utf-8")
    assert "html.embed .topin{display:none}" in gallery.CSS
