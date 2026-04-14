"""
YOLO-only debug overlay — no SAM2, no tracking.

Renders YOLO bounding boxes + keypoints on every frame to isolate
detection quality from segmentation and tracking issues.

Usage:
    python scripts/debug_yolo_only.py --config configs/local_reference.yaml
    python scripts/debug_yolo_only.py --config configs/hpc_reference.yaml \
        video_path=data/raw/my_video.avi
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.io_video import (
    open_video_reader, get_video_properties, create_video_writer, iter_frames,
)
from src.common.model_loaders import load_yolo
from src.common.config_loader import get_device
from src.common.visualization import draw_detections, draw_keypoints, draw_text
from src.pipelines.sam2_yolo.infer_yolo import detect_only

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-only debug overlay (no SAM2).")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Chunk identifier for parallel processing.")
    args, extra = parser.parse_known_args()

    config = load_config(args.config, extra or None)
    tag = f"yolo_debug_chunk{args.chunk_id}" if args.chunk_id is not None else "yolo_debug"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)
    logger.info("Starting YOLO-only debug overlay")
    logger.info("Run directory: %s", run_dir)

    # Load only YOLO (no SAM2)
    device = get_device(config)
    yolo = load_yolo(config["models"]["yolo_path"], device)

    det_cfg = config.get("detection", {})
    det_conf = det_cfg.get("confidence", 0.25)
    kpt_names = det_cfg.get("keypoint_names")
    kpt_min_conf = det_cfg.get("keypoint_min_conf", 0.3)
    filter_class = det_cfg.get("filter_class")
    nms_iou = det_cfg.get("nms_iou")
    max_animals = det_cfg.get("max_animals", 2)

    video_path = config["video_path"]
    cap = open_video_reader(video_path)
    props = get_video_properties(cap)

    colors = config.get("output", {}).get("overlay_colors")
    if colors:
        colors = [tuple(c) for c in colors]

    codec = config.get("output", {}).get("video_codec", "XVID")
    out_path = run_dir / "overlays" / f"yolo_debug_{date.today()}.avi"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_video_writer(out_path, props["fps"], props["width"], props["height"], codec)

    logger.info("Video: %s (%d frames, %.1f FPS)", video_path, props["total_frames"], props["fps"])

    frame_count = 0
    for frame_idx, frame_bgr in iter_frames(
        cap, start_frame=args.start_frame, end_frame=args.end_frame,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detections = detect_only(
            yolo, frame_rgb, det_conf,
            keypoint_names=kpt_names,
            filter_class=filter_class,
            nms_iou=nms_iou,
        )

        # Assign sequential IDs for color coding
        for i, det in enumerate(detections[:max_animals]):
            det.track_id = i + 1

        render_dets = detections[:max_animals]

        frame_out = np.copy(frame_rgb)
        if render_dets:
            frame_out = draw_detections(frame_out, render_dets, colors=colors)
            frame_out = draw_keypoints(frame_out, render_dets, colors=colors, min_conf=kpt_min_conf)

        confs = ", ".join(f"{d.conf:.2f}" for d in render_dets) if render_dets else "-"
        status = f"YOLO: {len(render_dets)} det | conf=[{confs}] | frame {frame_idx}"
        frame_out = draw_text(frame_out, status)

        writer.write(cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))
        frame_count += 1
        if frame_count % 50 == 0:
            logger.info("Processed %d frames", frame_count)

    cap.release()
    writer.release()
    logger.info("Pipeline complete. %d frames processed.", frame_count)
    logger.info("Overlay video: %s", out_path)


if __name__ == "__main__":
    main()
