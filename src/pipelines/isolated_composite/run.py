"""
Isolated-composite pipeline — SAMURAI masks + "erase other rat" YOLO pose.

Architecture:
  - Compute an empty-scene background from the video via temporal median.
  - SAM2 video predictor (SAMURAI mode) propagates identity + masks.
  - For each slot i, build a composite where the OTHER rat is replaced by
    background pixels — arena context and rat i stay untouched.
  - Run YOLO pose on each composite. Keypoints come out in original frame
    coordinates because rat i was never moved.
  - Write overlay video on the original frame + per-rat isolated videos
    (for visual inspection).

Multi-GPU: same --start-frame / --end-frame / --chunk-id contract as
samurai and centroid. Each worker independently computes its own background
using the same deterministic sampling, so no synchronization is needed.

Usage:
    python -m src.pipelines.isolated_composite.run \
        --config configs/hpc_isolated_composite.yaml
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.common.config_loader import (
    load_config, setup_run_dir, setup_logging, get_device,
)
from src.common.constants import DEFAULT_KEYPOINT_NAMES
from src.common.io_video import create_video_writer
from src.common.metrics import compute_centroid
from src.common.model_loaders import load_yolo
from src.common.utils import Detection
from src.common.visualization import (
    apply_masks_overlay, draw_centroids, draw_keypoints, draw_text,
)
from src.common.yolo_inference import detect_only
from src.pipelines.isolated_composite.composition import (
    compose_isolated_video_frame,
    erase_other_rat,
    load_or_compute_background,
    pick_detection_for_slot,
)
from src.pipelines.samurai.run import (
    DEFAULT_CHUNK_SIZE,
    _extract_chunk_frames,
    extract_frames,
    get_init_bboxes,
    load_samurai_predictor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keypoint carry-over (shifted by centroid delta)
# ---------------------------------------------------------------------------

def _carry_over_keypoints(
    slot_dets: List[Optional[Detection]],
    prev_slot_dets: Optional[List[Optional[Detection]]],
    prev_centroids: Optional[List[Optional[Tuple[float, float]]]],
    curr_centroids: List[Optional[Tuple[float, float]]],
) -> List[Optional[Detection]]:
    """Fill missing detections with the previous frame's keypoints shifted by centroid delta."""
    if prev_slot_dets is None or prev_centroids is None:
        return slot_dets

    for i in range(len(slot_dets)):
        if slot_dets[i] is not None:
            continue
        if prev_slot_dets[i] is None or not prev_slot_dets[i].keypoints:
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
        if carried.keypoints:
            for kp in carried.keypoints:
                kp.x += dx
                kp.y += dy
                kp.conf *= 0.9
        slot_dets[i] = carried

    return slot_dets


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def _extract_slot_masks(
    obj_ids,
    mask_logits,
    max_animals: int,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[Tuple[float, float]]]]:
    """Convert SAMURAI predictor output into per-slot boolean masks + centroids.

    Note: unlike samurai/run.py we do NOT call resolve_overlaps here. Preserving
    raw masks — including overlapping pixels — is essential so that the erase
    step never deletes pixels that belong to the rat of interest.
    """
    slot_masks: List[Optional[np.ndarray]] = [None] * max_animals
    slot_centroids: List[Optional[Tuple[float, float]]] = [None] * max_animals

    for i, obj_id in enumerate(obj_ids):
        mask = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
        slot_idx = int(obj_id) - 1
        if 0 <= slot_idx < max_animals:
            slot_masks[slot_idx] = mask
            c = compute_centroid(mask)
            if c is not None:
                slot_centroids[slot_idx] = c

    return slot_masks, slot_centroids


def _run_yolo_on_composites(
    yolo_model,
    composites_rgb: List[np.ndarray],
    confidence: float,
    keypoint_names: List[str],
) -> List[List[Detection]]:
    """Run YOLO on a list of RGB composites. Batches when possible."""
    if not composites_rgb:
        return []
    if len(composites_rgb) == 1:
        return [detect_only(
            yolo_model, composites_rgb[0],
            confidence=confidence, keypoint_names=keypoint_names,
        )]

    # Ultralytics YOLO accepts a list of arrays for batched inference.
    try:
        results = yolo_model(composites_rgb, conf=confidence, verbose=False)
    except Exception:
        # Fallback: per-composite call
        return [
            detect_only(
                yolo_model, comp,
                confidence=confidence, keypoint_names=keypoint_names,
            )
            for comp in composites_rgb
        ]

    from src.common.yolo_inference import _parse_results  # lazy import

    per_composite: List[List[Detection]] = []
    for r in results:
        per_composite.append(_parse_results([r], keypoint_names, None))
    return per_composite


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str | Path,
    cli_overrides: Optional[List[str]] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    chunk_id: Optional[int] = None,
    interactive: bool = False,
) -> Path:
    config = load_config(config_path, cli_overrides)
    tag = (
        f"isolated_composite_chunk{chunk_id}"
        if chunk_id is not None else "isolated_composite"
    )
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)

    logger.info("Starting Isolated-Composite pipeline (SAM2 erase + YOLO pose)")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)
    if start_frame > 0 or end_frame is not None:
        logger.info("Frame range: %d -> %s", start_frame, end_frame or "end")
    if chunk_id is not None:
        logger.info("Chunk ID: %d", chunk_id)

    device = get_device(config)
    logger.info("Using device: %s", device)

    video_path = config["video_path"]
    max_frames = config.get("scan", {}).get("max_frames")
    max_animals = config.get("detection", {}).get("max_animals", 2)
    kpt_names = config.get("detection", {}).get("keypoint_names", DEFAULT_KEYPOINT_NAMES)
    kpt_min_conf = config.get("detection", {}).get("keypoint_min_conf", 0.3)
    yolo_conf = config.get("detection", {}).get("confidence", 0.25)
    chunk_size = config.get("scan", {}).get("chunk_size", DEFAULT_CHUNK_SIZE)

    comp_cfg = config.get("composition", {}) or {}
    erase_dilate_px = int(comp_cfg.get("erase_dilate_px", 15))
    erase_feather_px = int(comp_cfg.get("feather_px", 5))

    out_cfg = config.get("output", {}) or {}
    colors_raw = out_cfg.get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else None
    codec = out_cfg.get("video_codec", "XVID")
    write_individual_videos = bool(out_cfg.get("write_individual_videos", True))

    # ==================================================================
    # Phase 0: Compute / load background
    # ==================================================================
    background_bgr = load_or_compute_background(config, video_path, run_dir)
    bg_h, bg_w = background_bgr.shape[:2]
    logger.info("Background ready (%dx%d)", bg_w, bg_h)

    # ==================================================================
    # Phase 1: Extract video frames to disk (SAMURAI needs a JPEG dir)
    # ==================================================================
    frames_dir = run_dir / "frames_tmp"
    num_frames, fps, width, height = extract_frames(
        video_path, frames_dir,
        start_frame=start_frame, end_frame=end_frame, max_frames=max_frames,
    )
    logger.info(
        "Video: %s (%dx%d @ %.1f FPS, %d frames extracted)",
        video_path, width, height, fps, num_frames,
    )

    if num_frames == 0:
        logger.error("No frames extracted. Check video path: %s", video_path)
        return run_dir

    if (bg_h, bg_w) != (height, width):
        logger.warning(
            "Background size (%dx%d) does not match video (%dx%d); resizing background",
            bg_w, bg_h, width, height,
        )
        background_bgr = cv2.resize(background_bgr, (width, height))

    # ==================================================================
    # Phase 2: Get initial bounding boxes
    # ==================================================================
    bboxes = get_init_bboxes(config, str(video_path), interactive=interactive)
    if len(bboxes) < max_animals:
        logger.warning(
            "Only %d bboxes provided but max_animals=%d", len(bboxes), max_animals,
        )

    # ==================================================================
    # Phase 3: Load YOLO
    # ==================================================================
    models_cfg = config["models"]
    yolo_path = models_cfg.get("yolo_path", "models/yolo/best.pt")
    yolo_model = load_yolo(yolo_path, device)
    logger.info("YOLO model loaded from %s", yolo_path)

    # ==================================================================
    # Phase 4: Prepare output writers
    # ==================================================================
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    ext = ".avi" if codec == "XVID" else ".mp4"
    today = date.today().strftime("%Y-%m-%d")
    out_video_path = overlays_dir / f"isolated_composite_{today}{ext}"
    writer = create_video_writer(out_video_path, fps, width, height, codec=codec)

    individual_writers: List[Optional[cv2.VideoWriter]] = []
    if write_individual_videos:
        for i in range(max_animals):
            path = overlays_dir / f"rat_{i+1}_{today}{ext}"
            individual_writers.append(
                create_video_writer(path, fps, width, height, codec=codec)
            )

    # ==================================================================
    # Phase 5: Chunked streaming — SAM2 propagate → erase → YOLO → render
    # ==================================================================
    n_chunks = (num_frames + chunk_size - 1) // chunk_size
    logger.info(
        "Processing %d frames in %d chunks of %d frames",
        num_frames, n_chunks, chunk_size,
    )

    gpu_tag = f"GPU{chunk_id}" if chunk_id is not None else "GPU"

    frame_count = 0
    yolo_hit_counts = [0] * max_animals
    carried_frames = 0
    last_centroids: Optional[List[Optional[Tuple[float, float]]]] = None
    prev_slot_dets: Optional[List[Optional[Detection]]] = None
    prev_centroids: Optional[List[Optional[Tuple[float, float]]]] = None

    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(c_start + chunk_size, num_frames)
        c_len = c_end - c_start

        logger.info(
            "Chunk %d/%d: frames %d-%d (%d frames)",
            chunk_idx + 1, n_chunks, c_start, c_end - 1, c_len,
        )

        chunk_frames_dir = run_dir / f"chunk_tmp_{chunk_idx}"
        n_chunk_frames = _extract_chunk_frames(
            frames_dir, chunk_frames_dir, c_start, c_end,
        )
        if n_chunk_frames == 0:
            logger.warning("No frames in chunk %d, skipping", chunk_idx)
            continue

        predictor = load_samurai_predictor(config, device)

        use_amp = device == "cuda"
        amp_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.inference_mode()
        )

        with torch.inference_mode(), amp_ctx:
            state = predictor.init_state(str(chunk_frames_dir))

            if chunk_idx == 0:
                for i, bb in enumerate(bboxes):
                    obj_id = i + 1
                    box = np.array(
                        [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                        dtype=np.float32,
                    )
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=obj_id, box=box,
                    )
                    logger.info(
                        "  Registered Rat %d: bbox=[%d, %d, %d, %d]",
                        obj_id, bb["x"], bb["y"],
                        bb["x"] + bb["w"], bb["y"] + bb["h"],
                    )
            else:
                if last_centroids and any(c is not None for c in last_centroids):
                    for i in range(max_animals):
                        obj_id = i + 1
                        if last_centroids[i] is not None:
                            cx, cy = last_centroids[i]
                            point = np.array([[cx, cy]], dtype=np.float32)
                            label = np.array([1], dtype=np.int32)
                            predictor.add_new_points_or_box(
                                state, frame_idx=0, obj_id=obj_id,
                                points=point, labels=label,
                            )
                            logger.info(
                                "  Re-init Rat %d from centroid (%.0f, %.0f)",
                                obj_id, cx, cy,
                            )
                        elif i < len(bboxes):
                            bb = bboxes[i]
                            box = np.array(
                                [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                                dtype=np.float32,
                            )
                            predictor.add_new_points_or_box(
                                state, frame_idx=0, obj_id=obj_id, box=box,
                            )
                else:
                    for i, bb in enumerate(bboxes):
                        obj_id = i + 1
                        box = np.array(
                            [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                            dtype=np.float32,
                        )
                        predictor.add_new_points_or_box(
                            state, frame_idx=0, obj_id=obj_id, box=box,
                        )

            pbar = tqdm(
                predictor.propagate_in_video(state),
                total=n_chunk_frames,
                desc=f"[{gpu_tag}] Chunk {chunk_idx+1}/{n_chunks}",
                unit="f",
            )
            for local_frame_idx, obj_ids, mask_logits in pbar:
                global_frame_idx = c_start + local_frame_idx

                frame_path = chunk_frames_dir / f"{local_frame_idx:05d}.jpg"
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    continue

                slot_masks, slot_centroids = _extract_slot_masks(
                    obj_ids, mask_logits, max_animals,
                )
                last_centroids = list(slot_centroids)

                # Build one erased composite per slot
                composites_bgr: List[Optional[np.ndarray]] = [None] * max_animals
                composites_rgb_for_yolo: List[np.ndarray] = []
                composite_slot_index: List[int] = []

                for i in range(max_animals):
                    mi = slot_masks[i]
                    if mi is None:
                        continue
                    mo_parts = [
                        slot_masks[j]
                        for j in range(max_animals)
                        if j != i and slot_masks[j] is not None
                    ]
                    if mo_parts:
                        mask_other = np.any(np.stack(mo_parts, axis=0), axis=0)
                    else:
                        mask_other = np.zeros_like(mi)

                    comp_bgr = erase_other_rat(
                        frame_bgr, background_bgr,
                        mask_self=mi, mask_other=mask_other,
                        dilate_px=erase_dilate_px,
                        feather_px=erase_feather_px,
                    )
                    composites_bgr[i] = comp_bgr
                    composites_rgb_for_yolo.append(
                        cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)
                    )
                    composite_slot_index.append(i)

                # YOLO pass over all composites at once
                per_composite_dets = _run_yolo_on_composites(
                    yolo_model,
                    composites_rgb_for_yolo,
                    confidence=yolo_conf,
                    keypoint_names=kpt_names,
                )

                slot_dets: List[Optional[Detection]] = [None] * max_animals
                for comp_idx, slot_idx in enumerate(composite_slot_index):
                    dets = per_composite_dets[comp_idx] if comp_idx < len(per_composite_dets) else []
                    picked = pick_detection_for_slot(
                        dets, mask_self=slot_masks[slot_idx],
                    )
                    if picked is not None:
                        picked.track_id = slot_idx + 1
                        slot_dets[slot_idx] = picked
                        yolo_hit_counts[slot_idx] += 1

                # Carry-over for missing slots
                missing_before = sum(1 for d in slot_dets if d is None)
                slot_dets = _carry_over_keypoints(
                    slot_dets, prev_slot_dets, prev_centroids, slot_centroids,
                )
                missing_after = sum(1 for d in slot_dets if d is None)
                if missing_before > missing_after:
                    carried_frames += 1

                prev_slot_dets = [copy.deepcopy(d) for d in slot_dets]
                prev_centroids = list(slot_centroids)

                # --- Render overlay on the original frame ---
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_out = frame_rgb.copy()

                render_masks = [m for m in slot_masks if m is not None]
                render_colors_list = None
                if colors:
                    render_colors_list = [
                        colors[i % len(colors)]
                        for i in range(max_animals)
                        if slot_masks[i] is not None
                    ]
                if render_masks:
                    frame_out = apply_masks_overlay(
                        frame_out, render_masks, colors=render_colors_list,
                    )

                render_centroids = [c for c in slot_centroids if c is not None]
                if render_centroids:
                    frame_out = draw_centroids(
                        frame_out, render_centroids, colors=render_colors_list,
                    )

                ordered_dets = [d for d in slot_dets if d is not None]
                if ordered_dets:
                    det_colors = None
                    if colors:
                        det_colors = [
                            colors[i % len(colors)]
                            for i in range(max_animals)
                            if slot_dets[i] is not None
                        ]
                    frame_out = draw_keypoints(
                        frame_out, ordered_dets, colors=det_colors,
                        min_conf=kpt_min_conf,
                    )

                active_count = sum(1 for m in slot_masks if m is not None)
                kpt_count = sum(1 for d in slot_dets if d is not None)
                status = (
                    f"F{global_frame_idx} | Animals: {active_count}/{max_animals}"
                    f" | Pose: {kpt_count}"
                )
                frame_out = draw_text(frame_out, status)
                writer.write(cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))

                # --- Write per-rat isolated videos ---
                if write_individual_videos:
                    for i in range(max_animals):
                        iw = individual_writers[i]
                        if iw is None:
                            continue
                        if slot_masks[i] is None:
                            iw.write(background_bgr)
                        else:
                            iso_bgr = compose_isolated_video_frame(
                                frame_bgr, background_bgr, slot_masks[i],
                                feather_px=3,
                            )
                            iw.write(iso_bgr)

                frame_count += 1
                if frame_count % 30 == 0:
                    hit_strs = "/".join(str(h) for h in yolo_hit_counts)
                    pbar.set_postfix_str(
                        f"pose_hits={hit_strs} carry={carried_frames}"
                    )
                if frame_count % 100 == 0:
                    logger.info(
                        "Processed %d frames | pose_hits=%s carry=%d",
                        frame_count,
                        "/".join(str(h) for h in yolo_hit_counts),
                        carried_frames,
                    )

        del predictor, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        shutil.rmtree(str(chunk_frames_dir), ignore_errors=True)
        logger.info(
            "Chunk %d complete. Total frames rendered: %d",
            chunk_idx + 1, frame_count,
        )

    writer.release()
    for iw in individual_writers:
        if iw is not None:
            iw.release()

    cleanup_frames = out_cfg.get("cleanup_frames", True)
    if cleanup_frames:
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        logger.info("Cleaned up extracted frames")

    logger.info(
        "Pipeline complete. %d frames processed in %d chunks.",
        frame_count, n_chunks,
    )
    for i in range(max_animals):
        logger.info(
            "Rat %d: pose detected in %d/%d frames (%.1f%%)",
            i + 1, yolo_hit_counts[i], frame_count,
            100 * yolo_hit_counts[i] / max(frame_count, 1),
        )
    logger.info(
        "Frames with any carry-over: %d/%d (%.1f%%)",
        carried_frames, frame_count,
        100 * carried_frames / max(frame_count, 1),
    )
    logger.info("Overlay video: %s", out_video_path)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated-composite pipeline: erase the other rat with computed background, then YOLO pose.",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Chunk identifier for parallel processing.")
    parser.add_argument("--interactive", action="store_true",
                        help="Use interactive bbox selection on first frame.")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides as key=value.")
    args = parser.parse_args()

    run_pipeline(
        args.config,
        args.overrides or None,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        chunk_id=args.chunk_id,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
