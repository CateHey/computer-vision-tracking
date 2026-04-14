"""
Interaction frame extraction for YOLO retraining.

Two-phase pipeline:
  Phase 1 (scan):    SAM2 centroid propagation → per-frame interaction scores CSV
  Phase 2 (select):  Merge CSVs → select ~200 well-distributed frames → extract JPEGs

Usage:
    # Phase 1: scan a chunk (GPU required)
    python roboflow/extract_interaction_frames.py scan \
        --config configs/hpc_extract.yaml \
        --start-frame 0 --end-frame 900 --chunk-id 0

    # Phase 2: select and extract (CPU only)
    python roboflow/extract_interaction_frames.py select \
        --config configs/hpc_extract.yaml \
        --scores-dir outputs/runs/<batch>/chunks \
        --output-dir roboflow/frames

    # Both phases (local, single GPU)
    python roboflow/extract_interaction_frames.py all \
        --config configs/local_extract.yaml
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Add project root to path so src imports work
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.io_video import open_video_reader, get_video_properties, iter_frames
from src.common.metrics import compute_centroid, mask_iou
from src.common.geometry import euclidean_distance
from src.pipelines.sam2_yolo.infer_yolo import detect_only
from src.pipelines.sam2_yolo.models_io import load_models
from src.pipelines.centroid.run import _segment_from_boxes, _segment_from_centroids

logger = logging.getLogger(__name__)

SCORES_HEADER = [
    "frame_idx", "centroid_dist_px", "mask_iou",
    "mask_area_0", "mask_area_1",
    "centroid_0_x", "centroid_0_y", "centroid_1_x", "centroid_1_y",
    "num_masks",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_row(writer, frame_idx, dist, iou_val, areas, centroids, num_masks):
    """Write a single row to the scores CSV."""
    c0x = round(centroids[0][0], 1) if centroids[0] is not None else ""
    c0y = round(centroids[0][1], 1) if centroids[0] is not None else ""
    c1x = round(centroids[1][0], 1) if len(centroids) > 1 and centroids[1] is not None else ""
    c1y = round(centroids[1][1], 1) if len(centroids) > 1 and centroids[1] is not None else ""
    dist_str = "" if (isinstance(dist, float) and math.isnan(dist)) else round(dist, 1)
    writer.writerow([
        frame_idx, dist_str, round(iou_val, 4),
        areas[0], areas[1],
        c0x, c0y, c1x, c1y,
        num_masks,
    ])


# ---------------------------------------------------------------------------
# Phase 1: Scan
# ---------------------------------------------------------------------------

def scan_chunk(
    config_path: str,
    cli_overrides: Optional[List[str]] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    chunk_id: Optional[int] = None,
) -> Path:
    """Phase 1: Run SAM2 centroid propagation and compute interaction scores.

    Mirrors the centroid pipeline loop but stripped down — no visualization,
    no keypoint assignment, no contacts. Only computes centroid distance
    and mask IoU per frame.

    Returns:
        Path to the output run directory containing scores.csv.
    """
    config = load_config(config_path, cli_overrides)
    tag = f"extract_chunk{chunk_id}" if chunk_id is not None else "extract_scan"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)

    logger.info("=== Interaction Frame Scan (Phase 1) ===")
    logger.info("Config: %s", config_path)
    logger.info("Run dir: %s", run_dir)

    # Load models
    yolo, sam = load_models(config)

    # Open video
    cap = open_video_reader(config["video_path"])
    props = get_video_properties(cap)
    logger.info("Video: %s (%d frames, %.1f FPS)",
                config["video_path"], props["total_frames"], props["fps"])

    # Config sections
    det_cfg = config.get("detection", {})
    det_conf = det_cfg.get("confidence", 0.25)
    max_animals = det_cfg.get("max_animals", 2)
    kpt_names = det_cfg.get("keypoint_names")
    filter_class = det_cfg.get("filter_class")
    nms_iou = det_cfg.get("nms_iou")
    sam_thr = config.get("segmentation", {}).get("sam_threshold", 0.0)
    max_frames = config.get("scan", {}).get("max_frames")
    max_init_retries = 50

    # Output CSV
    csv_path = run_dir / "scores.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(SCORES_HEADER)

    prev_centroids = None
    initialized = False
    init_retries = 0
    frame_count = 0

    for frame_idx, frame_bgr in iter_frames(
        cap, max_frames=max_frames, start_frame=start_frame, end_frame=end_frame,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_count += 1

        if not initialized:
            # YOLO detection to initialize SAM2
            detections = detect_only(
                yolo, frame_rgb, det_conf,
                keypoint_names=kpt_names,
                filter_class=filter_class,
                nms_iou=nms_iou,
            )[:max_animals]

            if len(detections) >= max_animals:
                masks, scores = _segment_from_boxes(sam, frame_rgb, detections, sam_thr)
                centroids = [compute_centroid(m) for m in masks]
                if all(c is not None for c in centroids):
                    dist = euclidean_distance(centroids[0], centroids[1])
                    iou_val = mask_iou(masks[0], masks[1])
                    areas = [int(m.sum()) for m in masks]
                    _write_row(writer, frame_idx, dist, iou_val, areas, centroids, 2)
                    prev_centroids = list(centroids)
                    initialized = True
                    logger.info("Initialized at frame %d", frame_idx)
                    continue

            init_retries += 1
            _write_row(writer, frame_idx, float("nan"), 0.0, [0, 0],
                       [None, None], 0)
            if init_retries >= max_init_retries:
                logger.error("Failed to initialize after %d frames", max_init_retries)
                break
            continue

        # Centroid propagation
        masks, scores = _segment_from_centroids(sam, frame_rgb, prev_centroids, sam_thr)

        centroids = []
        valid_count = 0
        for i, m in enumerate(masks):
            c = compute_centroid(m)
            if c is not None:
                centroids.append(c)
                valid_count += 1
            else:
                centroids.append(prev_centroids[i] if i < len(prev_centroids) else None)

        if valid_count == 2:
            dist = euclidean_distance(centroids[0], centroids[1])
            iou_val = mask_iou(masks[0], masks[1])
            areas = [int(masks[0].sum()), int(masks[1].sum())]
        else:
            dist = float("nan")
            iou_val = 0.0
            areas = [
                int(masks[0].sum()) if len(masks) > 0 and compute_centroid(masks[0]) else 0,
                int(masks[1].sum()) if len(masks) > 1 and compute_centroid(masks[1]) else 0,
            ]

        _write_row(writer, frame_idx, dist, iou_val, areas, centroids, valid_count)

        # Update prev centroids for valid masks
        for i in range(min(max_animals, len(centroids))):
            if centroids[i] is not None:
                prev_centroids[i] = centroids[i]

        if frame_count % 50 == 0:
            logger.info("Processed %d frames (frame_idx=%d)", frame_count, frame_idx)
        if frame_count % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    csv_file.close()
    cap.release()
    logger.info("Scan complete. %d frames → %s", frame_count, csv_path)
    print(f"Run directory: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Phase 2: Select and Extract
# ---------------------------------------------------------------------------

def _select_interaction_frames(
    rows: List[Dict[str, str]],
    max_frames: int,
    min_gap_frames: int,
    interaction_threshold_px: float,
) -> List[dict]:
    """Select well-distributed interaction frames across the video.

    Algorithm (temporal windows):
      1. Filter to frames with 2 valid masks and centroid_dist <= threshold
      2. Score each: (1 - dist/threshold)*0.6 + mask_iou*0.4
      3. Divide video into max_frames temporal windows
      4. Pick the best-scoring candidate from each window
      → guarantees even distribution across video duration
    """
    # --- Step 1: Filter and score candidates ---
    candidates = []
    for row in rows:
        num_masks = int(row["num_masks"])
        if num_masks < 2:
            continue
        dist_str = row["centroid_dist_px"]
        if dist_str == "" or dist_str == "nan":
            continue
        dist = float(dist_str)
        if dist > interaction_threshold_px:
            continue
        iou_val = float(row["mask_iou"])
        frame_idx = int(row["frame_idx"])
        score = (1.0 - dist / interaction_threshold_px) * 0.6 + iou_val * 0.4
        candidates.append({
            **row,
            "_dist": dist,
            "_iou": iou_val,
            "_frame_idx": frame_idx,
            "_score": score,
        })

    logger.info("Interaction candidates: %d / %d frames (threshold=%dpx)",
                len(candidates), len(rows), int(interaction_threshold_px))

    if not candidates:
        return []

    # --- Step 2: Temporal window selection ---
    # Divide the full scan range into max_frames windows
    all_frame_indices = [int(r["frame_idx"]) for r in rows]
    scan_start = min(all_frame_indices)
    scan_end = max(all_frame_indices)
    scan_range = scan_end - scan_start + 1
    window_size = max(1, scan_range // max_frames)

    selected = []
    for w in range(max_frames):
        win_start = scan_start + w * window_size
        win_end = win_start + window_size
        # Find best candidate in this window
        win_cands = [c for c in candidates if win_start <= c["_frame_idx"] < win_end]
        if win_cands:
            best = max(win_cands, key=lambda r: r["_score"])
            selected.append(best)

    # Enforce min_gap between selected frames
    if min_gap_frames > 1:
        filtered = []
        for s in selected:
            fi = s["_frame_idx"]
            too_close = any(abs(fi - f["_frame_idx"]) < min_gap_frames for f in filtered)
            if not too_close:
                filtered.append(s)
        selected = filtered

    selected.sort(key=lambda r: r["_frame_idx"])
    logger.info("Selected %d frames across %d windows (window=%d frames, min_gap=%d)",
                len(selected), max_frames, window_size, min_gap_frames)
    return selected


def select_and_extract(
    config_path: str,
    scores_dir: str,
    output_dir: str,
    cli_overrides: Optional[List[str]] = None,
) -> Path:
    """Phase 2: Merge score CSVs, select interaction frames, extract JPEGs.

    Returns:
        Path to the output directory containing extracted frames.
    """
    config = load_config(config_path, cli_overrides)
    extract_cfg = config.get("extract", {})
    max_frames = extract_cfg.get("max_frames", 200)
    min_gap = extract_cfg.get("min_gap_frames", 15)
    threshold_px = extract_cfg.get("interaction_threshold_px", 200)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Step 1: Merge CSVs ---
    scores_path = Path(scores_dir)
    csv_files = sorted(scores_path.glob("**/scores.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores.csv found under {scores_path}")

    all_rows = []
    for csv_file in csv_files:
        with csv_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    all_rows.sort(key=lambda r: int(r["frame_idx"]))
    logger.info("Merged %d rows from %d score CSVs", len(all_rows), len(csv_files))

    # --- Step 2: Select frames ---
    selected = _select_interaction_frames(all_rows, max_frames, min_gap, threshold_px)
    if not selected:
        logger.warning("No interaction frames found. Try increasing interaction_threshold_px.")
        return Path(output_dir)

    # --- Step 3: Extract frames from video ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_frames = {s["_frame_idx"] for s in selected}
    max_target = max(target_frames)

    video_path = config["video_path"]
    cap = open_video_reader(video_path)
    logger.info("Extracting %d frames from %s", len(target_frames), video_path)

    saved = 0
    for frame_idx, frame_bgr in iter_frames(cap):
        if frame_idx in target_frames:
            fname = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path / fname), frame_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        if frame_idx > max_target:
            break

    cap.release()
    logger.info("Extracted %d frames to %s", saved, out_path)

    # --- Step 4: Write metadata ---
    meta_csv = out_path / "metadata.csv"
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "centroid_dist_px", "mask_iou", "score"])
        for s in selected:
            writer.writerow([
                s["_frame_idx"],
                round(s["_dist"], 1),
                round(s["_iou"], 4),
                round(s["_score"], 4),
            ])

    logger.info("Metadata: %s", meta_csv)
    logger.info("Done. %d interaction frames ready for Roboflow labeling.", saved)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract interaction frames for YOLO retraining.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan (Phase 1)
    scan_p = subparsers.add_parser("scan",
        help="Phase 1: SAM2 centroid propagation → interaction scores CSV")
    scan_p.add_argument("--config", type=str, required=True)
    scan_p.add_argument("--start-frame", type=int, default=0)
    scan_p.add_argument("--end-frame", type=int, default=None)
    scan_p.add_argument("--chunk-id", type=int, default=None)
    scan_p.add_argument("overrides", nargs="*")

    # select (Phase 2)
    sel_p = subparsers.add_parser("select",
        help="Phase 2: Merge CSVs → select frames → extract JPEGs")
    sel_p.add_argument("--config", type=str, required=True)
    sel_p.add_argument("--scores-dir", type=str, required=True,
        help="Directory containing chunk run directories with scores.csv")
    sel_p.add_argument("--output-dir", type=str, default="roboflow/frames")
    sel_p.add_argument("overrides", nargs="*")

    # all (both phases, single GPU)
    all_p = subparsers.add_parser("all",
        help="Run both phases (single GPU, local)")
    all_p.add_argument("--config", type=str, required=True)
    all_p.add_argument("--output-dir", type=str, default="roboflow/frames")
    all_p.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    if args.command == "scan":
        scan_chunk(args.config, args.overrides or None,
                   start_frame=args.start_frame,
                   end_frame=args.end_frame,
                   chunk_id=args.chunk_id)

    elif args.command == "select":
        select_and_extract(args.config, args.scores_dir,
                           args.output_dir, args.overrides or None)

    elif args.command == "all":
        run_dir = scan_chunk(args.config, args.overrides or None)
        select_and_extract(args.config, str(run_dir),
                           args.output_dir, args.overrides or None)


if __name__ == "__main__":
    main()
