"""README_WHAT_TO_REVIEW.md, its section updates and the mouse file names."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.common.run_readme import README_NAME, update_section, write_run_readme  # noqa: E402
import validate_contacts as vc  # noqa: E402

LOG = """2026-11-20 10:00:00 [INFO] x: SAM3 init attempt 1: start frame 0, score threshold 0.50 -> 5/6 animals
2026-11-20 10:00:09 [INFO] x: === PHASE 1 complete. Handoff: 5/6 mice ===
2026-11-20 10:30:00 [INFO] x: Area guard: episode 100-250 unresolved after 3 attempt(s) (R2)
2026-11-20 10:30:01 [INFO] x: Recovery: 1 episode(s) {'unresolved': 1}, 3 attempt(s)
2026-11-20 10:30:02 [INFO] x: Mouse 1: pose detected in 900/900 frames (100.0%)
2026-11-20 10:30:02 [INFO] x: Mouse 2: pose detected in 700/900 frames (77.8%)
"""


def _run(tmp_path):
    run = tmp_path / "2026-11-20_100000_cutie_composite"
    for d in ("logs", "overlays", "tracks", "contacts/reports", "validation/identity"):
        (run / d).mkdir(parents=True)
    (run / "logs" / "run.log").write_text(LOG, encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps(
        {"git_commit": "abcdef0123456789", "timestamp_utc": "2026-11-20T00:00:00+00:00"}))
    (run / "config_used.yaml").write_text("video_path: data/raw/clip.avi\n")
    for f in ("contacts/results.xlsx", "contacts/diagnostics.xlsx", "contacts/event_log.txt",
              "tracks/tracks.jsonl.gz", "background.png", "overlays/cutie_mouse1_2026-11-20.avi"):
        (run / f).write_bytes(b"x")
    return run


def test_readme_sections_files_and_flags(tmp_path):
    run = _run(tmp_path)
    p = write_run_readme(run, 6, 30.0, 300.0, {"contact_rules_v3": True, "recovery_enabled": True})
    t = p.read_text(encoding="utf-8")
    assert p.name == README_NAME
    assert t.index("What this is") < t.index("## Review in this order") < t.index("## Files in this folder") \
        < t.index("## Red flags")
    assert "6 mice" in t and "5 min 00 s" in t and "30 fps" in t and "abcdef01" in t
    assert "contact_rules_v3 on" in t and "area-guard recovery on" in t
    assert "clip.avi" in t
    for name in ("results.xlsx", "diagnostics.xlsx", "event_log.txt", "tracks.jsonl.gz", "background.png",
                 "sheet `Investigation`", "sheet `Following`", "validation/", "logs/"):
        assert name in t
    assert "report.pdf" not in t.split("## Red flags")[0].split("| File")[1]   # only files that exist
    assert "Handoff 5/6" in t and "mouse 2 in only 77.8" in t and "unresolved" in t
    assert "score <run>/validation/review.xlsx" in t


def test_readme_reads_old_log_wording(tmp_path):
    run = _run(tmp_path)
    (run / "logs" / "run.log").write_text(
        "x: === PHASE 1 complete. Handoff: 4/6 rats ===\nx: Rat 3: pose detected in 1/10 frames (10.0%)\n")
    t = write_run_readme(run, 6, 30.0, 10.0, {}).read_text(encoding="utf-8")
    assert "Handoff 4/6" in t and "mouse 3 in only 10.0" in t


def test_update_section_replaces_and_appends(tmp_path):
    run = _run(tmp_path)
    assert update_section(run, "Identity check", "x") is False          # no README yet
    write_run_readme(run, 6, 30.0, 300.0, {})
    assert update_section(run, "Identity check", "OLD-TEXT")
    assert update_section(run, "Blind clip review", "clips")
    assert update_section(run, "Identity check", "second")
    t = (run / README_NAME).read_text(encoding="utf-8")
    assert t.count("## Identity check") == 1 and "second" in t and "OLD-TEXT" not in t
    assert "## Blind clip review\n\nclips" in t and "## Red flags" in t


def test_overlay_names_use_mouse():
    src = (ROOT / "src" / "pipelines" / "cutie_composite" / "run.py").read_text(encoding="utf-8")
    assert 'f"cutie_mouse{i+1}_{today}{ext}"' in src and "unitary_rat" not in src
    assert "area_per_mouse.png" in src and "area_per_rat" not in src


def test_find_overlay_accepts_both_names(tmp_path):
    for per_mouse in ("cutie_mouse1_2026-11-20.avi", "cutie_unitary_rat1_2026-11-12.avi"):
        run = tmp_path / per_mouse
        (run / "overlays").mkdir(parents=True)
        (run / "overlays" / per_mouse).write_bytes(b"x")
        (run / "overlays" / "cutie_composite_2026-11-20.avi").write_bytes(b"x")
        assert vc.find_overlay(run).name == "cutie_composite_2026-11-20.avi"
    # without a composite file the per-mouse videos are still not taken for the main overlay
    run = tmp_path / "only"
    (run / "overlays").mkdir(parents=True)
    (run / "overlays" / "cutie_mouse1_2026-11-20.avi").write_bytes(b"x")
    try:
        vc.find_overlay(run)
        assert False
    except FileNotFoundError:
        pass
