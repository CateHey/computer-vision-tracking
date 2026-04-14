"""
SAM2-only debug overlay — raw masks without identity matching.

Runs YOLO → SAM2 segment_frame() and renders the raw masks in detection order
(mask 0 = green, mask 1 = red, always). No IdentityMatcher assignment, no
filter_duplicates, no resolve_overlaps.

This isolates SAM2 mask quality from tracking/identity issues.

Usage:
    python scripts/debug_sam2_only.py --config configs/local_reference.yaml

    # Without centroid fallback (pure YOLO-box prompts only):
    python scripts/debug_sam2_only.py --config configs/local_reference.yaml --no-fallback
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
    apply_masks_overlay, draw_detections, draw_keypoints,
    draw_centroids, draw_text,
)
from src.pipelines.sam2_yolo.infer_yolo import detect_only
from src.pipelines.sam2_yolo.models_io import load_models
from src.pipelines.reference.sam2_processor import segment_frame
from src.pipelines.reference.identity_matcher import IdentityMatcher

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2-only debug overlay (no identity matching).")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Chunk identifier for parallel processing.")
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Disable centroid fallback — only use YOLO boxes as SAM2 prompts.",
    )
    args, extra = parser.parse_known_args()

    config = load_config(args.config, extra or None)
    tag = f"sam2_debug_chunk{args.chunk_id}" if args.chunk_id is not None else "sam2_debug"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)
    logger.info("Starting SAM2-only debug overlay (fallback=%s)", not args.no_fallback)
    logger.info("Run directory: %s", run_dir)

    # Load YOLO + SAM2
    yolo, sam = load_models(config)

    det_cfg = config.get("detection", {})
    det_conf = det_cfg.get("confidence", 0.25)
    kpt_names = det_cfg.get("keypoint_names")
    kpt_min_conf = det_cfg.get("keypoint_min_conf", 0.3)
    filter_class = det_cfg.get("filter_class")
    nms_iou = det_cfg.get("nms_iou")
    max_animals = det_cfg.get("max_animals", 2)
    sam_thr = config.get("segmentation", {}).get("sam_threshold", 0.0)

    # Matcher is used internally by segment_frame() for centroid fallback.
    # We create it but do NOT use its match() output for rendering.
    matcher = IdentityMatcher(
        max_entities=max_animals,
        proximity_threshold=config.get("identity_matcher", {}).get("proximity_threshold", 150.0),
    )

    video_path = config["video_path"]
    cap = open_video_reader(video_path)
    props = get_video_properties(cap)

    colors = config.get("output", {}).get("overlay_colors")
    if colors:
        colors = [tuple(c) for c in colors]

    codec = config.get("output", {}).get("video_codec", "XVID")
    suffix = "no_fallback" if args.no_fallback else "with_fallback"
    out_path = run_dir / "overlays" / f"sam2_debug_{suffix}_{date.today()}.avi"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_video_writer(out_path, props["fps"], props["width"], props["height"], codec)

    logger.info("Video: %s (%d frames, %.1f FPS)", video_path, props["total_frames"], props["fps"])

    frame_count = 0
    for frame_idx, frame_bgr in iter_frames(
        cap, start_frame=args.start_frame, end_frame=args.end_frame,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Step 1: YOLO detection
        detections = detect_only(
            yolo, frame_rgb, det_conf,
            keypoint_names=kpt_names,
            filter_class=filter_class,
            nms_iou=nms_iou,
        )

        # Step 2: SAM2 segmentation (raw, no identity matching)
        if args.no_fallback:
            # Override prev_centroids so segment_frame skips centroid fallback
            matcher.prev_centroids = [None] * max_animals

        all_masks, all_scores = segment_frame(
            sam, frame_rgb, detections, matcher,
            sam_threshold=sam_thr,
            max_entities=max_animals,
            kpt_min_conf=kpt_min_conf,
            interaction_state="SEPARATE",  # always SEPARATE to allow fallback if enabled
        )

        # Feed masks to matcher so it updates prev_centroids for next-frame fallback
        # (but we don't use the identity output for rendering)
        if not args.no_fallback and all_masks:
            matcher.match(all_masks, all_scores)

        # Step 3: Render raw masks in detection order (no identity assignment)
        frame_out = np.copy(frame_rgb)

        if all_masks:
            render_colors = colors[:len(all_masks)] if colors else None
            frame_out = apply_masks_overlay(frame_out, all_masks, colors=render_colors)

            # Draw centroids of raw masks
            centroids = [compute_centroid(m) for m in all_masks]
            valid_centroids = [c for c in centroids if c is not None]
            if valid_centroids:
                centroid_colors = render_colors[:len(valid_centroids)] if render_colors else None
                frame_out = draw_centroids(frame_out, valid_centroids, colors=centroid_colors)

        # Draw YOLO detections on top
        for i, det in enumerate(detections[:max_animals]):
            det.track_id = i + 1
        render_dets = detections[:max_animals]
        if render_dets:
            frame_out = draw_detections(frame_out, render_dets, colors=colors)
            frame_out = draw_keypoints(frame_out, render_dets, colors=colors, min_conf=kpt_min_conf)

        # Status bar with mask info
        n_masks = len(all_masks)
        n_dets = len(detections)
        scores_str = ", ".join(f"{s:.2f}" for s in all_scores) if all_scores else "-"
        fallback_label = "fallback=OFF" if args.no_fallback else "fallback=ON"
        n_from_yolo = min(n_dets, max_animals)
        n_from_fallback = max(0, n_masks - n_from_yolo)
        status = (
            f"SAM2: {n_masks} masks ({n_from_yolo} box + {n_from_fallback} fallback) | "
            f"YOLO: {n_dets} det | scores=[{scores_str}] | {fallback_label}"
        )
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
