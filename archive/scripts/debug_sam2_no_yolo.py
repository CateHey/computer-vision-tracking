"""
SAM2 without YOLO — centroid propagation debug overlay.

Uses YOLO ONLY on frame 0 to initialize, then SAM2 tracks both rats
using previous-frame centroids as point prompts. No YOLO after init.

This isolates whether YOLO's per-frame intermittency is causing SAM2 failures.

Usage:
    python scripts/debug_sam2_no_yolo.py --config configs/local_reference.yaml

    # Use a specific frame for initialization (e.g., frame where both rats are clearly visible):
    python scripts/debug_sam2_no_yolo.py --config configs/local_reference.yaml --init-frame 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

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
from src.common.metrics import compute_centroid
from src.common.visualization import (
    apply_masks_overlay, draw_centroids, draw_text,
)
from src.pipelines.sam2_yolo.infer_yolo import detect_only
from src.pipelines.sam2_yolo.models_io import load_models

logger = logging.getLogger(__name__)


def _segment_from_boxes(
    predictor, frame_rgb: np.ndarray, detections, sam_threshold: float,
) -> Tuple[List[np.ndarray], List[float]]:
    """Segment using YOLO boxes (initialization only)."""
    predictor.set_image(frame_rgb)
    masks, scores = [], []
    for det in detections:
        box = np.array([det.x1, det.y1, det.x2, det.y2])
        raw, sc, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=box, multimask_output=False,
        )
        masks.append(raw[0] > sam_threshold)
        scores.append(float(sc[0]))
    return masks, scores


def _segment_from_centroids(
    predictor, frame_rgb: np.ndarray,
    centroids: List[Tuple[float, float]],
    sam_threshold: float,
) -> Tuple[List[np.ndarray], List[float]]:
    """Segment using previous-frame centroids as point prompts.

    Each rat gets its own centroid as a positive prompt and the other
    rat's centroid as a negative prompt to help SAM2 distinguish them.
    """
    predictor.set_image(frame_rgb)
    masks, scores = [], []

    for i, (cx, cy) in enumerate(centroids):
        # Positive: this rat's centroid
        pos = [[cx, cy]]
        # Negative: all other rats' centroids (helps separate when close)
        neg = [[ox, oy] for j, (ox, oy) in enumerate(centroids) if j != i]

        all_points = pos + neg
        all_labels = [1] * len(pos) + [0] * len(neg)

        raw, sc, _ = predictor.predict(
            point_coords=np.array(all_points, dtype=np.float32),
            point_labels=np.array(all_labels, dtype=np.int32),
            box=None,
            multimask_output=False,
        )
        masks.append(raw[0] > sam_threshold)
        scores.append(float(sc[0]))

    return masks, scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM2 without YOLO — centroid propagation (YOLO only on init frame).",
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Chunk identifier for parallel processing.")
    parser.add_argument("--init-frame", type=int, default=None,
                        help="Frame to use for YOLO initialization (default: first frame with 2 detections).")
    args, extra = parser.parse_known_args()

    config = load_config(args.config, extra or None)
    tag = f"sam2_no_yolo_chunk{args.chunk_id}" if args.chunk_id is not None else "sam2_no_yolo"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)
    logger.info("Starting SAM2 no-YOLO debug (centroid propagation)")
    logger.info("Run directory: %s", run_dir)

    yolo, sam = load_models(config)

    det_cfg = config.get("detection", {})
    det_conf = det_cfg.get("confidence", 0.25)
    kpt_names = det_cfg.get("keypoint_names")
    filter_class = det_cfg.get("filter_class")
    nms_iou = det_cfg.get("nms_iou")
    max_animals = det_cfg.get("max_animals", 2)
    sam_thr = config.get("segmentation", {}).get("sam_threshold", 0.0)

    video_path = config["video_path"]
    cap = open_video_reader(video_path)
    props = get_video_properties(cap)

    colors = config.get("output", {}).get("overlay_colors")
    if colors:
        colors = [tuple(c) for c in colors]

    codec = config.get("output", {}).get("video_codec", "XVID")
    out_path = run_dir / "overlays" / f"sam2_no_yolo_pure_{date.today()}.avi"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_video_writer(out_path, props["fps"], props["width"], props["height"], codec)

    logger.info("Video: %s (%d frames, %.1f FPS)", video_path, props["total_frames"], props["fps"])

    # State: previous frame centroids (None until initialized)
    prev_centroids: Optional[List[Tuple[float, float]]] = None
    initialized = False
    init_frame_idx = args.init_frame  # None = auto-detect first good frame
    yolo_uses = 0

    frame_count = 0
    for frame_idx, frame_bgr in iter_frames(
        cap, start_frame=args.start_frame, end_frame=args.end_frame,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mode = "none"

        # Decide whether to use YOLO this frame
        use_yolo = False
        if not initialized:
            if init_frame_idx is None or frame_idx == init_frame_idx:
                use_yolo = True

        all_masks: List[np.ndarray] = []
        all_scores: List[float] = []

        if use_yolo:
            # YOLO initialization
            detections = detect_only(
                yolo, frame_rgb, det_conf,
                keypoint_names=kpt_names,
                filter_class=filter_class,
                nms_iou=nms_iou,
            )
            dets = detections[:max_animals]

            if len(dets) >= max_animals or (init_frame_idx is not None and frame_idx == init_frame_idx):
                # Good init frame (or user-specified)
                all_masks, all_scores = _segment_from_boxes(sam, frame_rgb, dets, sam_thr)
                yolo_uses += 1
                initialized = True
                mode = f"YOLO-INIT ({len(dets)} det)"
                logger.info("Initialized from YOLO at frame %d (%d detections)", frame_idx, len(dets))
            elif not initialized:
                # Not enough detections, skip and try next frame
                mode = f"WAITING ({len(dets)}/{max_animals} det)"

        elif initialized and prev_centroids is not None:
            # Pure centroid propagation — no YOLO
            all_masks, all_scores = _segment_from_centroids(
                sam, frame_rgb, prev_centroids, sam_thr,
            )
            mode = "CENTROID"

        # Update centroids from this frame's masks
        if all_masks:
            new_centroids = []
            for i, m in enumerate(all_masks):
                c = compute_centroid(m)
                if c is not None:
                    new_centroids.append(c)
                elif prev_centroids is not None and i < len(prev_centroids):
                    # Mask was empty — keep previous centroid
                    new_centroids.append(prev_centroids[i])
            if len(new_centroids) == max_animals:
                prev_centroids = new_centroids

        # Render
        frame_out = np.copy(frame_rgb)

        if all_masks:
            render_colors = colors[:len(all_masks)] if colors else None
            frame_out = apply_masks_overlay(frame_out, all_masks, colors=render_colors)

            centroids = [compute_centroid(m) for m in all_masks]
            valid = [c for c in centroids if c is not None]
            if valid:
                c_colors = render_colors[:len(valid)] if render_colors else None
                frame_out = draw_centroids(frame_out, valid, colors=c_colors)

        scores_str = ", ".join(f"{s:.2f}" for s in all_scores) if all_scores else "-"
        n_masks = len(all_masks)
        status = (
            f"SAM2-noYOLO: {n_masks} masks | mode={mode} | "
            f"scores=[{scores_str}] | YOLO used {yolo_uses}x | frame {frame_idx}"
        )
        frame_out = draw_text(frame_out, status)

        writer.write(cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))
        frame_count += 1
        if frame_count % 50 == 0:
            logger.info("Processed %d frames (mode=%s)", frame_count, mode)

    cap.release()
    writer.release()
    logger.info("Pipeline complete. %d frames processed. YOLO used %d times.", frame_count, yolo_uses)
    logger.info("Overlay video: %s", out_path)


if __name__ == "__main__":
    main()
