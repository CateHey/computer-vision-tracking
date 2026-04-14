"""
Centroid pipeline — SAM2 centroid propagation + YOLO keypoints (read-only) + contacts.

Architecture:
  - YOLO runs on frame 0 to initialize (box prompts → 2 masks)
  - After init, SAM2 propagates using centroid point prompts (multimask_output=False)
  - YOLO runs every frame ONLY to extract keypoints (read-only — never touches SAM2)
  - Keypoints assigned to SAM2 masks by spatial overlap (containment + nearest centroid)
  - ContactTracker uses the assigned keypoints + SAM2 masks for social contact classification

CRITICAL: multimask_output must be False. True degrades tracking quality.
CRITICAL: YOLO keypoints are READ-ONLY consumers of SAM2 identity. YOLO detection
order must NEVER influence SAM2 prompting. SAM2 centroids are the sole identity source.

Usage:
    python -m src.pipelines.centroid.run --config configs/local_centroid.yaml
    python -m src.pipelines.centroid.run --config configs/local_centroid.yaml contacts.enabled=true
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.contacts import ContactTracker
from src.common.io_video import (
    open_video_reader, get_video_properties, create_video_writer, iter_frames,
)
from src.common.metrics import compute_centroid
from src.common.geometry import euclidean_distance, resolve_overlaps
from src.common.visualization import (
    apply_masks_overlay, draw_centroids, draw_keypoints, draw_text,
)
from src.common.yolo_inference import detect_only
from src.common.model_loaders import load_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM2 segmentation helpers (identity source — do NOT modify prompting logic)
# ---------------------------------------------------------------------------

def _segment_from_boxes(predictor, frame_rgb, detections, sam_threshold):
    """Segment using YOLO boxes (initialization only — frame 0)."""
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


def _segment_from_centroids(predictor, frame_rgb, centroids, sam_threshold):
    """Segment using previous-frame centroids as point prompts.

    multimask_output=False is critical — True degrades tracking quality.
    """
    predictor.set_image(frame_rgb)
    masks, scores = [], []

    for i, (cx, cy) in enumerate(centroids):
        pos = [[cx, cy]]
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


# ---------------------------------------------------------------------------
# Keypoint assignment (YOLO → SAM2 masks, read-only)
# ---------------------------------------------------------------------------

def _assign_keypoints_to_masks(detections, masks, centroids):
    """Assign YOLO detections to SAM2 masks by spatial overlap.

    Two-pass: (1) containment — YOLO box center inside mask,
    (2) fallback — nearest centroid within 200px.

    YOLO detection order does NOT matter here. The mask determines
    which rat gets which keypoints.
    """
    n_slots = len(masks)
    slot_dets = [None] * n_slots
    used_dets = set()

    # Pass 1: containment
    for det in detections:
        cx, cy = det.center()
        ix, iy = int(round(cx)), int(round(cy))
        for i, mask in enumerate(masks):
            if slot_dets[i] is not None:
                continue
            h, w = mask.shape
            if 0 <= iy < h and 0 <= ix < w and mask[iy, ix]:
                slot_dets[i] = det
                used_dets.add(id(det))
                break

    # Pass 2: nearest centroid fallback
    for det in detections:
        if id(det) in used_dets:
            continue
        cx, cy = det.center()
        best_slot, best_dist = None, float("inf")
        for i in range(n_slots):
            if slot_dets[i] is not None:
                continue
            d = euclidean_distance((cx, cy), centroids[i])
            if d < best_dist:
                best_dist = d
                best_slot = i
        if best_slot is not None and best_dist < 200.0:
            slot_dets[best_slot] = det
            used_dets.add(id(det))

    # Set track_id for rendering
    for i, det in enumerate(slot_dets):
        if det is not None:
            det.track_id = i + 1

    return slot_dets


def _carry_over_keypoints(slot_dets, prev_slot_dets, prev_centroids, curr_centroids):
    """Fill missing detections with previous frame's keypoints shifted by centroid delta."""
    if prev_slot_dets is None:
        return slot_dets

    for i in range(len(slot_dets)):
        if slot_dets[i] is not None:
            continue
        if prev_slot_dets[i] is None:
            continue
        if prev_centroids[i] is None or curr_centroids[i] is None:
            continue

        dx = curr_centroids[i][0] - prev_centroids[i][0]
        dy = curr_centroids[i][1] - prev_centroids[i][1]

        carried = copy.deepcopy(prev_slot_dets[i])
        carried.x1 += dx
        carried.y1 += dy
        carried.x2 += dx
        carried.y2 += dy
        carried.track_id = i + 1
        carried.is_carried_over = True
        if carried.keypoints:
            for kp in carried.keypoints:
                kp.x += dx
                kp.y += dy
        slot_dets[i] = carried

    return slot_dets


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str | Path,
    cli_overrides: List[str] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    chunk_id: int | None = None,
) -> Path:
    config = load_config(config_path, cli_overrides)
    tag = f"centroid_chunk{chunk_id}" if chunk_id is not None else "centroid"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)

    logger.info("Starting Centroid pipeline (Approach A: YOLO keypoints read-only)")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)
    if start_frame > 0 or end_frame is not None:
        logger.info("Frame range: %d -> %s", start_frame, end_frame or "end")

    # Load models
    yolo, sam = load_models(config)

    # Open video
    video_path = config["video_path"]
    cap = open_video_reader(video_path)
    props = get_video_properties(cap)
    logger.info(
        "Video: %s (%dx%d @ %.1f FPS, %d frames)",
        video_path, props["width"], props["height"],
        props["fps"], props["total_frames"],
    )

    # Output video
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    codec = config.get("output", {}).get("video_codec", "XVID")
    ext = ".avi" if codec == "XVID" else ".mp4"
    today = date.today().strftime("%Y-%m-%d")
    out_video_path = overlays_dir / f"centroid_{today}{ext}"
    writer = create_video_writer(
        out_video_path, props["fps"], props["width"], props["height"], codec=codec,
    )

    # Parse config
    det_cfg = config.get("detection", {})
    det_conf = det_cfg.get("confidence", 0.25)
    max_animals = det_cfg.get("max_animals", 2)
    kpt_names = det_cfg.get("keypoint_names")
    kpt_min_conf = det_cfg.get("keypoint_min_conf", 0.3)
    filter_class = det_cfg.get("filter_class")
    nms_iou = det_cfg.get("nms_iou")
    sam_thr = config.get("segmentation", {}).get("sam_threshold", 0.0)

    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else None
    max_frames = config.get("scan", {}).get("max_frames")

    # Contact classification (optional)
    contacts_enabled = config.get("contacts", {}).get("enabled", False)
    contact_tracker = None
    if contacts_enabled:
        contact_tracker = ContactTracker(
            output_dir=run_dir / "contacts",
            fps=props["fps"],
            num_slots=max_animals,
            video_path=str(video_path),
            config=config,
        )
        logger.info("Contact classification enabled -> %s", run_dir / "contacts")

    # Centroid propagation state
    prev_centroids: Optional[List[Tuple[float, float]]] = None
    prev_slot_dets = None
    initialized = False
    yolo_uses = 0

    frame_count = 0
    carried_count = 0

    for frame_idx, frame_bgr in iter_frames(
        cap, max_frames=max_frames, start_frame=start_frame, end_frame=end_frame,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mode = "none"

        all_masks: List[np.ndarray] = []
        all_scores: List[float] = []
        slot_masks: List[Optional[np.ndarray]] = [None] * max_animals
        slot_centroids: List[Optional[Tuple[float, float]]] = [None] * max_animals
        slot_dets = [None] * max_animals
        has_carry_over = False

        # ==============================================================
        # STEP 1: SAM2 segmentation (identity source — untouchable)
        # ==============================================================
        if not initialized:
            detections = detect_only(
                yolo, frame_rgb, det_conf,
                keypoint_names=kpt_names, filter_class=filter_class,
                nms_iou=nms_iou,
            )[:max_animals]

            if len(detections) >= max_animals:
                all_masks, all_scores = _segment_from_boxes(
                    sam, frame_rgb, detections, sam_thr,
                )
                yolo_uses += 1
                initialized = True
                mode = f"YOLO-INIT ({len(detections)} det)"
                logger.info("Initialized at frame %d (%d detections)", frame_idx, len(detections))

                # Assign init keypoints directly
                for i, det in enumerate(detections[:max_animals]):
                    det.track_id = i + 1
                    slot_dets[i] = det
            else:
                mode = f"WAITING ({len(detections)}/{max_animals} det)"

        else:
            # SAM2 centroid propagation — YOLO does NOT influence this
            all_masks, all_scores = _segment_from_centroids(
                sam, frame_rgb, prev_centroids, sam_thr,
            )
            mode = "CENTROID"

        # ==============================================================
        # STEP 2: Update centroids from masks
        # ==============================================================
        new_centroids = list(prev_centroids) if prev_centroids else [None] * max_animals
        if all_masks:
            for i, m in enumerate(all_masks):
                c = compute_centroid(m)
                if c is not None:
                    new_centroids[i] = c
                    slot_masks[i] = m
                    slot_centroids[i] = c
                elif prev_centroids is not None and i < len(prev_centroids):
                    slot_centroids[i] = prev_centroids[i]

            # Resolve overlapping mask pixels
            resolve_overlaps(slot_masks, slot_centroids)

        # ==============================================================
        # STEP 3: YOLO keypoints (read-only — does NOT touch SAM2)
        # ==============================================================
        if initialized and mode == "CENTROID":
            detections = detect_only(
                yolo, frame_rgb, det_conf,
                keypoint_names=kpt_names, filter_class=filter_class,
                nms_iou=nms_iou,
            )

            # Assign keypoints to masks by spatial overlap
            valid_masks = [m for m in slot_masks if m is not None]
            valid_centroids = [c for c in slot_centroids if c is not None]
            if valid_masks and valid_centroids:
                slot_dets = _assign_keypoints_to_masks(
                    detections, slot_masks, slot_centroids,
                )

            # Carry over missing keypoints from previous frame
            slot_dets = _carry_over_keypoints(
                slot_dets, prev_slot_dets, prev_centroids, new_centroids,
            )

            n_fresh = sum(1 for d in slot_dets if d is not None and not getattr(d, 'is_carried_over', False))
            n_carried = sum(1 for d in slot_dets if d is not None and getattr(d, 'is_carried_over', False))
            has_carry_over = n_carried > 0
            if has_carry_over:
                carried_count += 1

            mode = f"CENTROID ({len(detections)} det, {n_fresh} assigned, {n_carried} carried)"

        prev_centroids = new_centroids
        prev_slot_dets = list(slot_dets)

        # ==============================================================
        # STEP 4: Contact classification (if enabled)
        # ==============================================================
        contact_events = []
        if contact_tracker is not None and initialized:
            contact_events = contact_tracker.update(
                slot_dets, slot_masks, slot_centroids, frame_idx,
            )

        # ==============================================================
        # STEP 5: Render overlay
        # ==============================================================
        frame_out = np.copy(frame_rgb)

        # Masks
        render_masks = [m for m in slot_masks if m is not None]
        render_colors_list = []
        for i in range(max_animals):
            if slot_masks[i] is not None and colors:
                render_colors_list.append(colors[i % len(colors)])
        if not render_colors_list:
            render_colors_list = None

        if render_masks:
            frame_out = apply_masks_overlay(frame_out, render_masks, colors=render_colors_list)

        # Centroids
        render_centroids = [c for c in slot_centroids if c is not None]
        if render_centroids:
            c_colors = render_colors_list[:len(render_centroids)] if render_colors_list else None
            frame_out = draw_centroids(frame_out, render_centroids, colors=c_colors)

        # Keypoints
        ordered_dets = [d for d in slot_dets if d is not None]
        if ordered_dets:
            frame_out = draw_keypoints(
                frame_out, ordered_dets, colors=render_colors_list,
                min_conf=kpt_min_conf,
            )

        # Contact lines
        if contact_events:
            for ev in contact_events:
                if ev.contact_type is not None:
                    ca = slot_centroids[ev.rat_a_slot]
                    cb = slot_centroids[ev.rat_b_slot]
                    if ca is not None and cb is not None:
                        cv2.line(
                            frame_out,
                            (int(ca[0]), int(ca[1])),
                            (int(cb[0]), int(cb[1])),
                            (255, 255, 0), 2,
                        )

        # Status bar
        active_count = sum(1 for m in slot_masks if m is not None)
        status = f"Animals: {active_count}/{max_animals} | {mode}"
        if contact_events:
            for ev in contact_events:
                if ev.contact_type is not None:
                    status += f" | {ev.contact_type}"
        frame_out = draw_text(frame_out, status)

        writer.write(cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))
        frame_count += 1
        if frame_count % 50 == 0:
            logger.info("Processed %d frames (mode=%s)", frame_count, mode)

        if frame_count % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    cap.release()
    writer.release()

    # Finalize contacts
    if contact_tracker is not None:
        summary = contact_tracker.finalize()
        total_bouts = sum(
            v.get("total_bouts", 0)
            for v in summary.get("contact_type_summary", {}).values()
        )
        logger.info("Contact analysis: %d bouts detected", total_bouts)

        try:
            from scripts.postprocess_contacts_simple import run_postprocess
            run_postprocess(run_dir / "contacts", fps=props["fps"])
        except Exception as e:
            logger.warning("Contact post-processing failed: %s", e)

    logger.info("Pipeline complete. %d frames processed. YOLO used %d times for init.", frame_count, yolo_uses)
    logger.info("Frames with keypoint carry-over: %d/%d (%.1f%%)",
                carried_count, frame_count,
                100 * carried_count / max(frame_count, 1))
    logger.info("Overlay video: %s", out_video_path)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centroid pipeline: SAM2 centroid propagation + YOLO keypoints + contacts.",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Chunk identifier for parallel processing.")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides as key=value.")
    args = parser.parse_args()

    run_pipeline(
        args.config,
        args.overrides or None,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        chunk_id=args.chunk_id,
    )


if __name__ == "__main__":
    main()
