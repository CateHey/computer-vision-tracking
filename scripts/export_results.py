#!/usr/bin/env python3
"""
Export and organize run results for sharing or archiving.

Copies key outputs from a run directory into a clean export folder.

Usage:
    python scripts/export_results.py \
        --run-dir outputs/runs/2025-12-09_143022_sam2_yolo \
        --export-dir exports/run_2025-12-09
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.config_loader import setup_logging

logger = logging.getLogger(__name__)


def export_run(run_dir: Path, export_dir: Path) -> None:
    """Copy key artifacts from a run directory to an export location."""
    run_dir = run_dir.resolve()
    export_dir = export_dir.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    config_file = run_dir / "config_used.yaml"
    if config_file.exists():
        shutil.copy2(config_file, export_dir / "config_used.yaml")
        logger.info("Copied config")

    # Copy overlays
    overlays_dir = run_dir / "overlays"
    if overlays_dir.exists():
        dest = export_dir / "overlays"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(overlays_dir, dest)
        logger.info("Copied overlays")

    # Copy frames
    frames_dir = run_dir / "frames"
    if frames_dir.exists():
        dest = export_dir / "frames"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(frames_dir, dest)
        logger.info("Copied frames")

    # Copy scan artifacts
    scan_dir = run_dir / "scan"
    if scan_dir.exists():
        dest = export_dir / "scan"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(scan_dir, dest)
        logger.info("Copied scan results")

    # Copy logs
    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        dest = export_dir / "logs"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(logs_dir, dest)
        logger.info("Copied logs")

    logger.info("Export complete: %s", export_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export run results for sharing.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory to export.")
    parser.add_argument("--export-dir", type=Path, required=True, help="Destination directory.")
    args = parser.parse_args()

    setup_logging()
    export_run(args.run_dir, args.export_dir)


if __name__ == "__main__":
    main()
