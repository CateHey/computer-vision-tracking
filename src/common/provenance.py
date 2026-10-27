"""Provenance of a run: what code, models and software produced its outputs.

`write_provenance(run_dir, config_path, extra)` writes `<run_dir>/provenance.json` with the
UTC time, host, Python and package versions, the git commit of the repo (and of the CUTIE
checkout), the GPU, the MD5 of the YOLO weights and of the config. Every field is collected
on its own: one that cannot be determined (no git, no torch, ...) is null or
"not installed" and never stops the run.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import logging
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUTIE_DIR = PROJECT_ROOT / "models" / "cutie"
NOT_INSTALLED = "not installed"
MAX_DIRTY_FILES = 50

# label in provenance.json -> distribution name on PyPI (read from the package metadata, so
# slow libraries such as torch or ultralytics are not imported just to read their version)
PACKAGES = {"torch": "torch", "transformers": "transformers", "ultralytics": "ultralytics", "numpy": "numpy"}


def file_md5(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    """MD5 hex digest of a file, read in chunks (the YOLO weights are about 86 MB)."""
    h = hashlib.md5()  # noqa: S324  a fingerprint, not a security measure
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> Optional[str]:
    """Raw stdout of `git -C repo <args>`; None if git is missing or the command fails."""
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def git_state(repo: Path = PROJECT_ROOT) -> Tuple[Optional[str], Optional[bool], List[str]]:
    """(commit, dirty, modified paths) of the repo.

    `dirty` is True when a tracked file differs from HEAD (`git status --porcelain
    --untracked-files=no`, so run outputs and other untracked files do not count); None
    when the state cannot be read.
    """
    commit = (_git(repo, "rev-parse", "HEAD") or "").strip() or None
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if status is None:
        return commit, None, []
    files = [line[3:] for line in status.splitlines() if line.strip()]
    return commit, bool(files), files[:MAX_DIRTY_FILES]


def package_versions() -> Dict[str, str]:
    """Versions of torch, transformers, ultralytics, opencv and numpy ("not installed" if absent)."""
    out: Dict[str, str] = {}
    for label, dist in PACKAGES.items():
        try:
            out[label] = importlib.metadata.version(dist)
        except ImportError:  # PackageNotFoundError is a ModuleNotFoundError
            out[label] = NOT_INSTALLED
        except Exception as exc:  # noqa: BLE001  a broken install must not stop the run
            out[label] = f"unavailable ({type(exc).__name__})"
    try:
        cv2 = importlib.import_module("cv2")
        out["opencv"] = str(cv2.__version__)
    except ImportError:
        out["opencv"] = NOT_INSTALLED
    except Exception as exc:  # noqa: BLE001
        out["opencv"] = f"unavailable ({type(exc).__name__})"
    return out


def gpu_name() -> Optional[str]:
    """Name of CUDA device 0, or None without torch / without a visible GPU."""
    try:
        torch = importlib.import_module("torch")
        return str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001  no torch, broken install or driver problem
        return None


def _resolve(path: Union[str, Path]) -> Path:
    """A path as given, else relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else PROJECT_ROOT / p


def collect_provenance(run_dir: Union[str, Path], config_path: Union[str, Path],
                       extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The provenance dict that write_provenance() saves (see the module docstring)."""
    commit, dirty, dirty_files = git_state(PROJECT_ROOT)
    cutie_commit = None
    if (CUTIE_DIR / ".git").exists():
        cutie_commit = (_git(CUTIE_DIR, "rev-parse", "HEAD") or "").strip() or None
    extra = dict(extra or {})
    cfg = _resolve(config_path).resolve()
    data: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
        "git_dirty_files": dirty_files,
        "cutie_commit": cutie_commit,
        "packages": package_versions(),
        "gpu": gpu_name(),
        "yolo_md5": None,
        "config_path": str(cfg),
        "config_md5": None,
        "config_used_md5": None,
    }
    yolo = extra.get("yolo_path")
    if yolo and _resolve(yolo).is_file():
        data["yolo_md5"] = file_md5(_resolve(yolo))
    if cfg.is_file():
        data["config_md5"] = file_md5(cfg)
    used = Path(run_dir) / "config_used.yaml"  # the config after CLI overrides
    if used.is_file():
        data["config_used_md5"] = file_md5(used)
    data.update(extra)
    return data


def write_provenance(run_dir: Union[str, Path], config_path: Union[str, Path],
                     extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write `<run_dir>/provenance.json` and return its path.

    Args:
        run_dir: the run directory (created if missing).
        config_path: the YAML config the run started from (path and MD5 are recorded; the MD5 of
            `config_used.yaml`, which has the CLI overrides applied, too when it exists).
        extra: merged into the top level of the file. `extra["yolo_path"]`, when the file
            exists, also gives `yolo_md5`.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "provenance.json"
    path.write_text(json.dumps(collect_provenance(run_dir, config_path, extra), indent=2, default=str),
                    encoding="utf-8")
    return path
