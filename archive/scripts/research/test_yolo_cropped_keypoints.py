"""
Test: Do YOLO keypoints survive on per-rat crops?

Runs SAM2 centroid propagation for ~4 seconds, then for each frame:
  1. Runs YOLO on the FULL image → gets keypoints (baseline)
  2. Crops each rat using SAM2 mask bounding box + padding, blacks out non-mask pixels
  3. Runs YOLO on each CROP → gets keypoints (test)
  4. Compares: did YOLO find the rat? how many keypoints? confidence?

This tests whether the "mask-conditioned YOLO" approach from the proposed
pipeline refactor is viable before committing to building it.

Usage:
    python scripts/research/test_yolo_cropped_keypoints.py --config configs/local_reference.yaml
    python scripts/research/test_yolo_cropped_keypoints.py --config configs/local_reference.yaml --end-frame 120
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
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
    apply_masks_overlay, draw_centroids, draw_text, draw_keypoints,
)
from src.pipelines.sam2_yolo.infer_yolo import detect_only
from src.pipelines.sam2_yolo.models_io import load_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM2 helpers (same as debug_sam2_no_yolo.py)
# ---------------------------------------------------------------------------

def _segment_from_boxes(predictor, frame_rgb, detections, sam_threshold):
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
            box=None, multimask_output=False,
        )
        masks.append(raw[0] > sam_threshold)
        scores.append(float(sc[0]))
    return masks, scores


# ---------------------------------------------------------------------------
# Crop + YOLO test
# ---------------------------------------------------------------------------

def _mask_bbox(mask: np.ndarray, padding: int = 20) -> Tuple[int, int, int, int]:
    """Get bounding box of a boolean mask with padding."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    h, w = mask.shape
    x1 = max(0, int(np.min(xs)) - padding)
    y1 = max(0, int(np.min(ys)) - padding)
    x2 = min(w, int(np.max(xs)) + padding)
    y2 = min(h, int(np.max(ys)) + padding)
    return x1, y1, x2, y2


def _crop_rat(frame_rgb: np.ndarray, mask: np.ndarray, padding: int = 20) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a rat region from the frame, blacking out non-mask pixels."""
    x1, y1, x2, y2 = _mask_bbox(mask, padding)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((64, 64, 3), dtype=np.uint8), (0, 0, 0, 0)

    crop = frame_rgb[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    crop[~mask_crop] = 0  # black out non-mask pixels
    return crop, (x1, y1, x2, y2)


def _run_yolo_on_crop(yolo, crop, det_conf, kpt_names, filter_class, nms_iou):
    """Run YOLO on a cropped image and return detections."""
    # Resize crop to a reasonable size for YOLO (at least 128px)
    h, w = crop.shape[:2]
    if max(h, w) < 128:
        scale = 128 / max(h, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)))

    return detect_only(yolo, crop, det_conf, keypoint_names=kpt_names,
                       filter_class=filter_class, nms_iou=nms_iou)


def _keypoint_summary(detections) -> Dict:
    """Summarize keypoint detection quality."""
    if not detections:
        return {"detected": False, "n_dets": 0, "n_kpts": 0, "kpt_confs": [], "kpt_names": []}

    # Take the best detection (highest confidence)
    best = max(detections, key=lambda d: d.conf)
    kpts = best.keypoints or []
    return {
        "detected": True,
        "n_dets": len(detections),
        "det_conf": round(best.conf, 3),
        "n_kpts": len([k for k in kpts if k.conf >= 0.3]),
        "n_kpts_total": len(kpts),
        "kpt_confs": {k.name: round(k.conf, 3) for k in kpts if k.name},
        "kpt_names": [k.name for k in kpts if k.conf >= 0.3],
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_crop_comparison(frame_rgb, masks, full_dets, crop_dets_list, crop_bboxes, frame_idx):
    """Draw a comparison frame: left = full image YOLO, right = crop info."""
    h, w = frame_rgb.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)

    # Left: full image with masks + full-image YOLO keypoints
    left = frame_rgb.copy()
    if masks:
        colors = [(0, 255, 0, 150), (255, 0, 0, 150)]
        left = apply_masks_overlay(left, masks, colors=colors[:len(masks)])
    if full_dets:
        left = draw_keypoints(left, full_dets)
    left = draw_text(left, f"FULL IMAGE — {len(full_dets)} det | frame {frame_idx}")
    canvas[:, :w] = left

    # Right: show each crop with its YOLO keypoints
    right = frame_rgb.copy()
    if masks:
        colors = [(0, 255, 0, 150), (255, 0, 0, 150)]
        right = apply_masks_overlay(right, masks, colors=colors[:len(masks)])

    for i, (crop_dets, bbox) in enumerate(zip(crop_dets_list, crop_bboxes)):
        x1, y1, x2, y2 = bbox
        if x2 <= x1:
            continue
        # Draw crop bounding box
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        cv2.rectangle(right, (x1, y1), (x2, y2), color, 2)

        # Draw keypoints from crop (offset to full image coords)
        for det in crop_dets:
            if det.keypoints:
                for kp in det.keypoints:
                    if kp.conf >= 0.3:
                        px = int(kp.x + x1)
                        py = int(kp.y + y1)
                        cv2.circle(right, (px, py), 5, color, -1)
                        cv2.circle(right, (px, py), 5, (255, 255, 255), 1)

        n_kpts = sum(1 for d in crop_dets for k in (d.keypoints or []) if k.conf >= 0.3)
        label = f"Crop {i}: {len(crop_dets)} det, {n_kpts} kpts"
        cv2.putText(right, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    right = draw_text(right, f"CROPPED YOLO — frame {frame_idx}")
    canvas[:, w:] = right

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test YOLO keypoint quality on SAM2 mask crops vs full image.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=120,
                        help="End frame (default 120 = ~4s at 30fps).")
    parser.add_argument("--padding", type=int, default=20,
                        help="Padding around mask bbox for crop (px).")
    parser.add_argument("--init-frame", type=int, default=None)
    args, extra = parser.parse_known_args()

    config = load_config(args.config, extra or None)
    run_dir = setup_run_dir(config, tag="crop_keypoint_test")
    setup_logging(run_dir)
    logger.info("Starting crop keypoint test")

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

    codec = config.get("output", {}).get("video_codec", "XVID")
    out_path = run_dir / "overlays" / f"crop_keypoint_test_{date.today()}.avi"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_video_writer(out_path, props["fps"], props["width"] * 2, props["height"], codec)

    # SAM2 centroid propagation state
    prev_centroids: Optional[List[Tuple[float, float]]] = None
    initialized = False
    init_frame_idx = args.init_frame

    # Results tracking
    results = {"frames": [], "summary": {}}
    frame_count = 0
    total_full_kpts = 0
    total_crop_kpts = 0
    total_full_dets = 0
    total_crop_dets = 0
    frames_full_detected = 0
    frames_crop_detected = 0

    for frame_idx, frame_bgr in iter_frames(
        cap, start_frame=args.start_frame, end_frame=args.end_frame,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Step 1: SAM2 centroid propagation (get masks) ---
        use_yolo_init = False
        if not initialized:
            if init_frame_idx is None or frame_idx == init_frame_idx:
                use_yolo_init = True

        all_masks = []
        if use_yolo_init:
            dets = detect_only(yolo, frame_rgb, det_conf,
                               keypoint_names=kpt_names, filter_class=filter_class,
                               nms_iou=nms_iou)[:max_animals]
            if len(dets) >= max_animals or (init_frame_idx is not None and frame_idx == init_frame_idx):
                all_masks, _ = _segment_from_boxes(sam, frame_rgb, dets, sam_thr)
                initialized = True
                logger.info("SAM2 initialized at frame %d", frame_idx)
        elif initialized and prev_centroids is not None:
            all_masks, _ = _segment_from_centroids(sam, frame_rgb, prev_centroids, sam_thr)

        # Update centroids
        if all_masks:
            new_c = []
            for i, m in enumerate(all_masks):
                c = compute_centroid(m)
                if c is not None:
                    new_c.append(c)
                elif prev_centroids and i < len(prev_centroids):
                    new_c.append(prev_centroids[i])
            if len(new_c) == max_animals:
                prev_centroids = new_c

        if not all_masks or len(all_masks) < max_animals:
            frame_count += 1
            continue

        # --- Step 2: Run YOLO on FULL image (baseline) ---
        full_dets = detect_only(yolo, frame_rgb, det_conf,
                                keypoint_names=kpt_names, filter_class=filter_class,
                                nms_iou=nms_iou)

        # --- Step 3: Crop each rat and run YOLO on crops ---
        crop_dets_list = []
        crop_bboxes = []
        for mask in all_masks:
            crop, bbox = _crop_rat(frame_rgb, mask, padding=args.padding)
            crop_dets = _run_yolo_on_crop(yolo, crop, det_conf, kpt_names,
                                          filter_class, nms_iou)
            crop_dets_list.append(crop_dets)
            crop_bboxes.append(bbox)

        # --- Step 4: Compare results ---
        full_summary = _keypoint_summary(full_dets)
        crop_summaries = [_keypoint_summary(cd) for cd in crop_dets_list]

        frame_result = {
            "frame": frame_idx,
            "full_image": full_summary,
            "crops": crop_summaries,
        }
        results["frames"].append(frame_result)

        # Accumulate stats
        total_full_dets += full_summary["n_dets"]
        total_full_kpts += full_summary["n_kpts"]
        if full_summary["detected"]:
            frames_full_detected += 1

        for cs in crop_summaries:
            total_crop_dets += cs["n_dets"]
            total_crop_kpts += cs["n_kpts"]
            if cs["detected"]:
                frames_crop_detected += 1

        # --- Step 5: Render comparison video ---
        comp = _draw_crop_comparison(frame_rgb, all_masks, full_dets,
                                     crop_dets_list, crop_bboxes, frame_idx)
        writer.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

        frame_count += 1
        if frame_count % 30 == 0:
            logger.info("Frame %d — full: %d det %d kpts | crops: %s",
                         frame_idx, full_summary["n_dets"], full_summary["n_kpts"],
                         [(cs["n_dets"], cs["n_kpts"]) for cs in crop_summaries])

    cap.release()
    writer.release()

    # --- Summary ---
    n_crops_expected = frame_count * max_animals
    results["summary"] = {
        "total_frames": frame_count,
        "full_image": {
            "frames_with_detection": frames_full_detected,
            "detection_rate": round(frames_full_detected / max(frame_count, 1), 3),
            "avg_keypoints_per_frame": round(total_full_kpts / max(frame_count, 1), 1),
        },
        "cropped": {
            "crops_with_detection": frames_crop_detected,
            "detection_rate": round(frames_crop_detected / max(n_crops_expected, 1), 3),
            "avg_keypoints_per_crop": round(total_crop_kpts / max(n_crops_expected, 1), 1),
        },
    }

    # Save results JSON
    json_path = run_dir / "crop_keypoint_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Frames processed: %d", frame_count)
    logger.info("")
    logger.info("FULL IMAGE YOLO:")
    logger.info("  Detection rate: %d/%d (%.1f%%)",
                frames_full_detected, frame_count,
                100 * frames_full_detected / max(frame_count, 1))
    logger.info("  Avg keypoints/frame: %.1f",
                total_full_kpts / max(frame_count, 1))
    logger.info("")
    logger.info("CROPPED YOLO (per-rat mask crops):")
    logger.info("  Detection rate: %d/%d crops (%.1f%%)",
                frames_crop_detected, n_crops_expected,
                100 * frames_crop_detected / max(n_crops_expected, 1))
    logger.info("  Avg keypoints/crop: %.1f",
                total_crop_kpts / max(n_crops_expected, 1))
    logger.info("")
    logger.info("Video: %s", out_path)
    logger.info("JSON:  %s", json_path)


if __name__ == "__main__":
    main()
