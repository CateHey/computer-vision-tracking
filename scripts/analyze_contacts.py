#!/usr/bin/env python3
"""
Standalone social contact analysis on existing pipeline output.

Runs YOLO detection (with keypoints) + contact classification on a video
without SAM2 segmentation. Produces the same contact outputs (CSV, JSON,
PDF report) as the inline pipeline mode.

This is useful for:
  - Analyzing videos that were already processed (re-run contacts only)
  - Quick contact analysis without the full SAM2 pipeline
  - Batch processing multiple videos

Usage:
    python scripts/analyze_contacts.py --config configs/local_quick.yaml
    python scripts/analyze_contacts.py --config configs/local_quick.yaml contacts.enabled=true
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.contacts import ContactTracker, classify_pair_contacts, estimate_body_length, ContactEvent, Zone
from src.common.io_video import open_video_reader, get_video_properties, iter_frames
from src.common.metrics import evaluate_closeness
from src.common.utils import Detection, Keypoint

logger = logging.getLogger(__name__)


def run_contact_analysis(
    config_path: str | Path,
    cli_overrides: List[str] | None = None,
) -> Path:
    """Run contact classification using YOLO detections (no SAM2).

    Uses evaluate_closeness() as a pre-filter, then classifies contact
    types using keypoint geometry on close frames.
    """
    config = load_config(config_path, cli_overrides)

    # Force contacts enabled
    if "contacts" not in config:
        config["contacts"] = {}
    config["contacts"]["enabled"] = True

    run_dir = setup_run_dir(config, tag="contacts")
    setup_logging(run_dir)

    logger.info("=== Social Contact Analysis ===")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)

    # Load YOLO only (no SAM2 needed)
    from ultralytics import YOLO

    video_path = config["video_path"]
    models_cfg = config["models"]
    det_cfg = config.get("detection", {})
    close_cfg = config.get("closeness", {})
    contacts_cfg = config.get("contacts", {})
    scan_cfg = config.get("scan", {})

    conf_thr = det_cfg.get("confidence", 0.25)
    max_animals = det_cfg.get("max_animals", 2)
    dist_thr = close_cfg.get("distance_threshold_norm", 0.15)
    iou_thr = close_cfg.get("iou_threshold", 0.1)
    max_frames = scan_cfg.get("max_frames")
    min_kp_conf = contacts_cfg.get("min_keypoint_conf", 0.3)
    fallback_bl = contacts_cfg.get("fallback_body_length_px", 120.0)

    # Keypoint names
    kpt_names = det_cfg.get("keypoint_names") or [
        "tail_tip", "tail_base", "tail_start", "mid_body",
        "nose", "right_ear", "left_ear",
    ]

    # Load YOLO model
    model_path = models_cfg["yolo_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    model = YOLO(str(model_path))
    logger.info("YOLO model loaded from %s", model_path)

    # Open video
    cap = open_video_reader(video_path)
    props = get_video_properties(cap)
    fps = props["fps"]
    width = props["width"]
    height = props["height"]
    logger.info("Video: %dx%d @ %.1f FPS, %d frames", width, height, fps, props["total_frames"])

    # Create contact tracker
    contact_tracker = ContactTracker(
        output_dir=run_dir / "contacts",
        fps=fps,
        num_slots=max_animals,
        video_path=str(video_path),
        config=config,
    )

    # Body length EMA per detection (by track_id)
    bl_ema: Dict[int, float] = {}
    bl_alpha = 0.1
    prev_centroids: Dict[int, Optional[tuple]] = {}

    frame_count = 0
    close_count = 0

    for frame_idx, frame_bgr in iter_frames(cap, max_frames=max_frames):
        # Run YOLO detection
        results = model(frame_bgr, conf=conf_thr, verbose=False)
        detections: List[Detection] = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            has_kpts = r.keypoints is not None and len(r.keypoints) > 0
            kpts_data = r.keypoints.data.cpu().numpy() if has_kpts else None

            for i, (bbox, conf) in enumerate(zip(xyxy, confs)):
                x1, y1, x2, y2 = bbox.tolist()
                kps = None
                if kpts_data is not None:
                    kps = []
                    for kp_idx in range(kpts_data.shape[1]):
                        kx, ky, kc = kpts_data[i][kp_idx]
                        name = kpt_names[kp_idx] if kp_idx < len(kpt_names) else f"kp{kp_idx}"
                        kps.append(Keypoint(x=float(kx), y=float(ky), conf=float(kc), name=name))

                detections.append(Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    conf=float(conf), keypoints=kps, track_id=i,
                ))

        time_sec = frame_idx / fps

        # Pre-filter: is this a close frame?
        is_close, _, _ = evaluate_closeness(
            detections, width, height, dist_thr, iou_thr,
        )

        # Build slot-like structures (use detection index as slot)
        slot_masks: List[Optional[np.ndarray]] = [None] * max_animals
        slot_centroids: List[Optional[tuple]] = [None] * max_animals
        for di, det in enumerate(detections[:max_animals]):
            slot_centroids[di] = det.center()

        # Run contact classification
        contact_tracker.update(detections, slot_masks, slot_centroids, frame_idx)

        if is_close:
            close_count += 1

        frame_count += 1
        if frame_count % 200 == 0:
            logger.info(
                "Processed %d frames (%d close, %.1f%%)",
                frame_count, close_count,
                close_count / frame_count * 100 if frame_count > 0 else 0,
            )

    cap.release()

    # Finalize
    summary = contact_tracker.finalize()

    total_bouts = sum(
        v.get("total_bouts", 0)
        for v in summary.get("contact_type_summary", {}).values()
    )

    logger.info("=== Analysis Complete ===")
    logger.info("Frames processed: %d (%d close)", frame_count, close_count)
    logger.info("Contact bouts: %d", total_bouts)
    logger.info("Outputs: %s", run_dir / "contacts")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run social contact analysis on a video (YOLO-only, no SAM2).",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides as key=value pairs.",
    )
    args = parser.parse_args()

    run_contact_analysis(args.config, args.overrides or None)


if __name__ == "__main__":
    main()
