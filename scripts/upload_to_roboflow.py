#!/usr/bin/env python3
"""
Upload exported frames to a Roboflow project for annotation.

Reads project settings from config; API key must come from the
ROBOFLOW_API_KEY environment variable (never stored in config files).

Usage:
    export ROBOFLOW_API_KEY="your-key-here"

    python scripts/upload_to_roboflow.py \
        --config configs/local_quick.yaml \
        --frames-root outputs/runs/2025-12-09_143022_extract_frames/frames

    # Override split via CLI:
    python scripts/upload_to_roboflow.py \
        --config configs/local_quick.yaml \
        --frames-root frames/ \
        roboflow.split=valid
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.config_loader import load_config, setup_logging

logger = logging.getLogger(__name__)


class RoboflowUploader:
    """Upload images from a folder structure to a Roboflow project."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project_id: str,
        split: str = "train",
        batch_name: Optional[str] = None,
        tag_names: Optional[List[str]] = None,
        num_retry_uploads: int = 2,
    ) -> None:
        from roboflow import Roboflow

        self.split = split
        self.batch_name = batch_name
        self.tag_names = tag_names
        self.num_retry_uploads = num_retry_uploads

        logger.info("Connecting to Roboflow: workspace=%s, project=%s", workspace, project_id)
        rf = Roboflow(api_key=api_key)
        self.project = rf.workspace(workspace).project(project_id)
        logger.info("Connected successfully")

    def _find_images(self, frames_root: Path) -> List[Path]:
        """Recursively find all image files under frames_root."""
        frames_root = frames_root.resolve()
        if not frames_root.exists():
            raise FileNotFoundError(f"Frames root does not exist: {frames_root}")

        images = sorted(
            p for p in frames_root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        logger.info("Found %d images under %s", len(images), frames_root)
        return images

    def upload_folder(self, frames_root: Path) -> Dict[str, Any]:
        """Upload all images under frames_root.

        Returns:
            Summary dict with total, successful, failed counts.
        """
        images = self._find_images(frames_root)
        total = len(images)
        successful = 0
        failed = 0
        failed_paths: List[str] = []

        if total == 0:
            logger.warning("No images found. Nothing to upload.")
            return {"total": 0, "successful": 0, "failed": 0, "failed_paths": []}

        logger.info("Uploading %d images (split=%s)", total, self.split)

        for idx, img_path in enumerate(images, start=1):
            img_str = str(img_path)
            try:
                logger.info("(%d/%d) Uploading: %s", idx, total, img_path.name)
                self.project.upload(
                    image_path=img_str,
                    split=self.split,
                    num_retry_uploads=self.num_retry_uploads,
                    batch_name=self.batch_name,
                    tag_names=self.tag_names,
                )
                successful += 1
            except Exception as e:
                failed += 1
                failed_paths.append(img_str)
                logger.error("Failed to upload %s: %s", img_str, e)

        logger.info("Upload complete: %d/%d successful, %d failed", successful, total, failed)
        if failed_paths:
            for p in failed_paths:
                logger.error("  Failed: %s", p)

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "failed_paths": failed_paths,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload frames to Roboflow for annotation.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--frames-root", type=Path, required=True,
        help="Root folder containing images to upload.",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Roboflow API key (overrides ROBOFLOW_API_KEY env var).",
    )
    parser.add_argument(
        "--tag", dest="tags", action="append", default=None,
        help="Tag to apply to uploads (can be passed multiple times).",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides as key=value pairs.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides or None)
    setup_logging()

    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        logger.error("No API key. Set ROBOFLOW_API_KEY or use --api-key.")
        sys.exit(1)

    rf_cfg = config.get("roboflow", {})

    uploader = RoboflowUploader(
        api_key=api_key,
        workspace=rf_cfg.get("workspace", ""),
        project_id=rf_cfg.get("project", ""),
        split=rf_cfg.get("split", "train"),
        batch_name=rf_cfg.get("batch_name"),
        tag_names=args.tags,
    )

    uploader.upload_folder(args.frames_root)


if __name__ == "__main__":
    main()
