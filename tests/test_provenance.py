"""provenance.json: what code, models and software produced a run."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.common import provenance  # noqa: E402
from src.common.provenance import write_provenance  # noqa: E402

CONFIG = REPO / "configs" / "local_cutie_composite.yaml"


def test_write_provenance_records_the_environment_and_the_md5s(tmp_path):
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"not really weights" * 5000)
    run = tmp_path / "run"
    run.mkdir()
    (run / "config_used.yaml").write_text("video_path: x.avi\n")
    path = write_provenance(run, CONFIG, extra={
        "yolo_path": str(weights), "video_path": "data/raw/x.avi", "cli_overrides": ["scan.max_frames=30"]})
    assert path == run / "provenance.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    for key in ("timestamp_utc", "hostname", "python_version", "git_commit", "git_dirty", "git_dirty_files",
                "cutie_commit", "packages", "gpu", "yolo_md5", "config_path", "config_md5", "config_used_md5"):
        assert key in d, key
    assert d["timestamp_utc"].endswith("+00:00") and d["hostname"] and d["python_version"].count(".") == 2
    assert d["yolo_md5"] == hashlib.md5(weights.read_bytes()).hexdigest()
    assert d["config_md5"] == hashlib.md5(CONFIG.read_bytes()).hexdigest()
    assert d["config_used_md5"] == hashlib.md5((run / "config_used.yaml").read_bytes()).hexdigest()
    assert d["config_path"] == str(CONFIG)
    assert (d["yolo_path"], d["video_path"], d["cli_overrides"]) == (str(weights), "data/raw/x.avi", ["scan.max_frames=30"])
    assert set(d["packages"]) == {"torch", "transformers", "ultralytics", "opencv", "numpy"}
    assert d["packages"]["numpy"] == np.__version__
    assert d["git_commit"] is None or len(d["git_commit"]) == 40      # the repo is a git checkout
    assert d["git_dirty"] in (True, False, None) and isinstance(d["git_dirty_files"], list)


def test_missing_yolo_file_and_extra_are_tolerated(tmp_path):
    d = json.loads(write_provenance(tmp_path / "r", CONFIG, extra={"yolo_path": str(tmp_path / "nope.pt")})
                   .read_text(encoding="utf-8"))
    assert d["yolo_md5"] is None and d["config_used_md5"] is None
    d = json.loads(write_provenance(tmp_path / "r2", CONFIG).read_text(encoding="utf-8"))
    assert d["yolo_md5"] is None and d["config_md5"]


def test_missing_packages_are_reported_not_raised(monkeypatch):
    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(provenance.importlib.metadata, "version", absent)
    v = provenance.package_versions()
    assert [v[k] for k in ("torch", "transformers", "ultralytics", "numpy")] == ["not installed"] * 4
    assert v["opencv"] != ""


def test_git_state_outside_a_repo_is_unknown(tmp_path):
    assert provenance.git_state(tmp_path) == (None, None, [])
