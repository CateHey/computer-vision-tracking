"""
Configuration loading, path resolution, run directory setup, and logging.

All scripts and pipelines use this module to load YAML configs and
set up structured output directories under outputs/runs/<run_id>/.
"""

from __future__ import annotations

import copy
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Project root: two levels up from src/common/config_loader.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(
    yaml_path: str | Path,
    cli_overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load a YAML config file and apply optional CLI overrides.

    Args:
        yaml_path: Path to the YAML config file (absolute or relative to cwd).
        cli_overrides: List of "dotted.key=value" strings, e.g.
            ["detection.confidence=0.5", "scan.max_frames=300"].

    Returns:
        Fully resolved configuration dictionary.
    """
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Apply CLI overrides
    if cli_overrides:
        for override in cli_overrides:
            if "=" not in override:
                logger.warning("Ignoring malformed override (no '='): %s", override)
                continue
            key, value = override.split("=", 1)
            _set_nested(config, key.strip(), _parse_value(value.strip()))

    # Resolve relative paths against project root
    _resolve_paths(config)

    # Store metadata
    config["_meta"] = {
        "config_file": str(yaml_path),
        "project_root": str(PROJECT_ROOT),
    }

    return config


def setup_run_dir(config: Dict[str, Any], tag: Optional[str] = None) -> Path:
    """Create a timestamped run directory and save the config used.

    Args:
        config: The loaded configuration dictionary.
        tag: Optional tag appended to run_id (e.g., "scan", "pipeline").

    Returns:
        Path to the created run directory.
    """
    output_dir = Path(config.get("output_dir", PROJECT_ROOT / "outputs" / "runs"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{timestamp}_{tag}" if tag else timestamp
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config used for this run
    config_copy = copy.deepcopy(config)
    config_copy.pop("_meta", None)
    config_out = run_dir / "config_used.yaml"
    with config_out.open("w", encoding="utf-8") as f:
        yaml.dump(config_copy, f, default_flow_style=False, sort_keys=False)

    logger.info("Run directory created: %s", run_dir)
    return run_dir


def setup_logging(
    run_dir: Optional[Path] = None,
    level: str = "INFO",
) -> None:
    """Configure Python logging to console and optionally to a file.

    Args:
        run_dir: If provided, also log to <run_dir>/logs/run.log.
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Reset root logger
    root = logging.getLogger()
    root.setLevel(log_level)
    # Remove existing handlers to avoid duplicates
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # File handler (if run_dir provided)
    if run_dir is not None:
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)
        logger.info("Logging to file: %s", log_dir / "run.log")


def get_device(config: Dict[str, Any]) -> str:
    """Resolve the device string from config.

    Returns "cuda" or "cpu" based on config and hardware availability.
    """
    import torch

    device = config.get("models", {}).get("device", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Keys whose values are filesystem paths and should be resolved.
_PATH_KEYS = {"video_path", "output_dir", "yolo_path", "sam2_checkpoint", "sam3_checkpoint", "samurai_checkpoint", "sleap_checkpoint"}


def _resolve_paths(config: Dict[str, Any]) -> None:
    """Resolve relative paths in the config against PROJECT_ROOT (in-place)."""
    for key, value in config.items():
        if isinstance(value, dict):
            _resolve_paths(value)
        elif isinstance(value, str) and key in _PATH_KEYS:
            p = Path(value)
            if not p.is_absolute():
                config[key] = str(PROJECT_ROOT / p)


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _parse_value(value_str: str) -> Any:
    """Parse a CLI override value string into the appropriate Python type."""
    if value_str.lower() == "null" or value_str.lower() == "none":
        return None
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str
