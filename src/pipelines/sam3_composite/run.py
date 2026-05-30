"""
SAM3 composite pipeline.

Same architecture as isolated_composite but using SAM3 (HuggingFace) instead
of SAM2 video predictor:

  - SAM3 detects "mouse" automatically via text prompt (no manual bboxes)
  - Background median computed once (shared with isolated_composite)
  - Per-rat composites: erase other rats using background
  - YOLO runs on composites (single-rat frames) for keypoint detection
  - Keypoint carry-over from previous frame when YOLO fails (mirrors isolated)
  - Unitary videos use compose_isolated_video_frame (rat on clean background)
  - ContactTrackerV2 for analysis

Usage:
  export HF_TOKEN="hf_..."
  python -m src.pipelines.sam3_composite.run \
      --config configs/hpc_sam3_composite.yaml \
      contacts.enabled=true
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.common.config_loader import load_config, setup_run_dir, setup_logging, get_device
from src.common.constants import DEFAULT_KEYPOINT_NAMES
from src.common.utils import Detection
from src.common.io_video import create_video_writer
from src.common.model_loaders import load_yolo
from src.common.yolo_inference import detect_only
from src.common.contacts_v2 import ContactTrackerV2
from src.pipelines.isolated_composite.composition import (
    load_or_compute_background,
    erase_other_rat,
    pick_detection_for_slot,
    compose_isolated_video_frame,
)

DEFAULT_CHUNK_SIZE = 150

logger = logging.getLogger(__name__)


# ============================================================================
# SAM3 LOADER
# ============================================================================

def load_sam3_model(device: str):
    """Load SAM3 model + processor from HuggingFace."""
    from huggingface_hub import login
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        logger.info("HuggingFace login successful")
    else:
        logger.warning("HF_TOKEN not set in environment. Model load may fail.")

    logger.info("Loading SAM3 model from facebook/sam3...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Sam3VideoModel.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    logger.info("SAM3 model loaded.")
    return model, processor


# ============================================================================
# FRAME EXTRACTION
# ============================================================================

def extract_frames_to_memory(
    video_path: str,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total
    if max_frames is not None:
        end_frame = min(end_frame, start_frame + max_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    idx = start_frame
    while idx < end_frame:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        idx += 1

    cap.release()
    return frames, fps, width, height


# ============================================================================
# IDENTITY ASSIGNMENT
# ============================================================================

def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def assign_identities_by_centroid(
    new_masks: List[np.ndarray],
    new_centroids: List[Optional[Tuple[float, float]]],
    prev_centroids: List[Optional[Tuple[float, float]]],
    num_slots: int,
    max_dist: Optional[float] = None,
) -> List[int]:
    """Greedy nearest-neighbor assignment with optional max distance threshold.

    If max_dist is provided, assignments above that distance are rejected (slot -1).
    Helps with N>=3 rats where spurious far-away detections could be misassigned.
    """
    n_new = len(new_centroids)
    assignment = [-1] * n_new

    dist = np.full((n_new, num_slots), float("inf"))
    for i in range(n_new):
        if new_centroids[i] is None:
            continue
        for s in range(num_slots):
            if prev_centroids[s] is None:
                continue
            dx = new_centroids[i][0] - prev_centroids[s][0]
            dy = new_centroids[i][1] - prev_centroids[s][1]
            d = np.sqrt(dx * dx + dy * dy)
            # Apply max_dist filter: reject assignments above threshold
            if max_dist is not None and d > max_dist:
                continue
            dist[i, s] = d

    used_new = set()
    used_slots = set()
    while len(used_slots) < num_slots and len(used_new) < n_new:
        masked = dist.copy()
        for i in used_new:
            masked[i, :] = float("inf")
        for s in used_slots:
            masked[:, s] = float("inf")
        if not np.isfinite(masked).any():
            break
        i, s = np.unravel_index(np.argmin(masked), masked.shape)
        if not np.isfinite(masked[i, s]):
            break
        assignment[i] = s
        used_new.add(i)
        used_slots.add(s)

    return assignment


# ============================================================================
# KEYPOINT CARRY-OVER (mirrored from isolated_composite)
# ============================================================================

def _carry_over_keypoints(
    slot_dets: List[Optional[Detection]],
    prev_slot_dets: Optional[List[Optional[Detection]]],
    prev_frame_centroids: Optional[List[Optional[Tuple[float, float]]]],
    curr_centroids: List[Optional[Tuple[float, float]]],
) -> List[Optional[Detection]]:
    if prev_slot_dets is None or prev_frame_centroids is None:
        return slot_dets

    for i in range(len(slot_dets)):
        if slot_dets[i] is not None:
            continue
        if prev_slot_dets[i] is None or not prev_slot_dets[i].keypoints:
            continue
        if prev_frame_centroids[i] is None or curr_centroids[i] is None:
            continue

        dx = curr_centroids[i][0] - prev_frame_centroids[i][0]
        dy = curr_centroids[i][1] - prev_frame_centroids[i][1]

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


# ============================================================================
# SAM3 PROCESSING
# ============================================================================

def process_chunk_with_sam3(
    model, processor,
    frames_chunk: List[np.ndarray],
    text_prompt: str,
    device: str,
    score_threshold: float = 0.5,
) -> Dict[int, Dict[str, Any]]:
    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames_chunk]

    session = processor.init_video_session(
        video=pil_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    session = processor.add_text_prompt(inference_session=session, text=text_prompt)

    results = {}
    for model_out in model.propagate_in_video_iterator(inference_session=session):
        processed = processor.postprocess_outputs(session, model_out)
        masks = processed['masks'].cpu().numpy()
        scores = processed['scores'].cpu().tolist()

        keep = [i for i, s in enumerate(scores) if s >= score_threshold]
        if not keep:
            results[model_out.frame_idx] = {
                'masks': np.zeros((0,) + masks.shape[1:], dtype=bool),
                'scores': [],
            }
            continue

        masks_filtered = masks[keep] > 0.5
        scores_filtered = [scores[i] for i in keep]
        results[model_out.frame_idx] = {
            'masks': masks_filtered,
            'scores': scores_filtered,
        }

    del session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    config_path,
    cli_overrides=None,
    start_frame=0,
    end_frame=None,
    chunk_id=None,
):
    config = load_config(config_path, cli_overrides)
    tag = f"sam3_composite_chunk{chunk_id}" if chunk_id is not None else "sam3_composite"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)

    logger.info("Starting SAM3 composite pipeline")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)

    device = get_device(config)
    logger.info("Using device: %s", device)

    video_path = config["video_path"]
    max_frames = config.get("scan", {}).get("max_frames")
    num_slots = config.get("detection", {}).get("max_animals", 2)
    kpt_names = config.get("detection", {}).get("keypoint_names", DEFAULT_KEYPOINT_NAMES)
    kpt_min_conf = config.get("detection", {}).get("keypoint_min_conf", 0.3)
    yolo_conf = config.get("detection", {}).get("confidence", 0.25)
    chunk_size = config.get("scan", {}).get("chunk_size", DEFAULT_CHUNK_SIZE)
    # Bug 6 fix: max distance for centroid-based identity matching
    max_assignment_dist = config.get("detection", {}).get("max_assignment_dist", None)
    if max_assignment_dist is not None:
        max_assignment_dist = float(max_assignment_dist)

    sam3_cfg = config.get("sam3", {})
    text_prompt = sam3_cfg.get("text_prompt", "mouse")
    score_threshold = float(sam3_cfg.get("score_threshold", 0.5))

    comp_cfg = config.get("composition", {}) or {}
    erase_dilate_px = int(comp_cfg.get("erase_dilate_px", 15))
    erase_feather_px = int(comp_cfg.get("feather_px", 5))
    mask_dilate_for_pick = int(comp_cfg.get("mask_dilate_for_pick", 7))
    unitary_feather_px = int(comp_cfg.get("unitary_feather_px", 3))

    out_cfg = config.get("output", {}) or {}
    colors_raw = out_cfg.get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
    ]
    codec = out_cfg.get("video_codec", "XVID")
    ext = ".avi" if codec == "XVID" else ".mp4"
    write_individual_videos = bool(out_cfg.get("write_individual_videos", True))
    today = date.today().strftime("%Y-%m-%d")

    # Phase 1: extract frames
    logger.info("Extracting frames from %s ...", video_path)
    all_frames, fps, width, height = extract_frames_to_memory(
        video_path, start_frame=start_frame, end_frame=end_frame, max_frames=max_frames,
    )
    num_frames = len(all_frames)
    logger.info("Loaded %d frames @ %.1f fps (%dx%d)", num_frames, fps, width, height)

    if num_frames == 0:
        logger.error("No frames extracted.")
        return run_dir

    # Phase 2: background
    background = load_or_compute_background(config, video_path, run_dir)
    bg_h, bg_w = background.shape[:2]
    if (bg_h, bg_w) != (height, width):
        background = cv2.resize(background, (width, height))
    logger.info("Background ready.")

    # Phase 3: models
    sam3_model, sam3_processor = load_sam3_model(device)
    yolo_path = config.get("models", {}).get("yolo_path", "models/yolo/best.pt")
    yolo_model = load_yolo(yolo_path, device)
    logger.info("YOLO loaded from %s", yolo_path)

    # Phase 4: writers
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = overlays_dir / f"sam3_composite_{today}{ext}"
    main_writer = create_video_writer(out_video_path, fps, width, height, codec=codec)

    unitary_writers = []
    if write_individual_videos:
        for i in range(num_slots):
            path = overlays_dir / f"sam3_unitary_rat{i+1}_{today}{ext}"
            unitary_writers.append(create_video_writer(path, fps, width, height, codec=codec))
    else:
        unitary_writers = [None] * num_slots

    # Phase 5: contacts
    contacts_enabled = config.get("contacts", {}).get("enabled", False)
    contact_tracker = None
    if contacts_enabled:
        contacts_dir = run_dir / "contacts"
        contact_tracker = ContactTrackerV2(
            output_dir=contacts_dir,
            fps=fps,
            num_slots=num_slots,
            video_path=str(video_path),
            config=config,
        )
        logger.info("ContactTrackerV2 initialized")

    # Phase 6: process chunks
    n_chunks = (num_frames + chunk_size - 1) // chunk_size
    logger.info("Processing %d frames in %d chunks of %d frames",
                num_frames, n_chunks, chunk_size)

    # State (3 separate variables for distinct purposes)
    prev_centroids: List[Optional[Tuple[float, float]]] = [None] * num_slots
    prev_frame_centroids: Optional[List[Optional[Tuple[float, float]]]] = None
    prev_slot_dets: Optional[List[Optional[Detection]]] = None

    # Bug 1 fix: robust first-frame init flag (works with multi-GPU / chunk_id)
    first_assignment_done = False

    yolo_hit_counts = [0] * num_slots
    carried_frames = 0
    global_frame_idx = 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_frames)
        chunk_frames = all_frames[chunk_start:chunk_end]
        logger.info("Chunk %d/%d (frames %d-%d, %d frames)",
                    chunk_idx + 1, n_chunks, chunk_start, chunk_end - 1, len(chunk_frames))

        sam3_results = process_chunk_with_sam3(
            sam3_model, sam3_processor, chunk_frames,
            text_prompt=text_prompt, device=device,
            score_threshold=score_threshold,
        )

        for local_idx in range(len(chunk_frames)):
            frame_rgb = chunk_frames[local_idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Bug 2 fix: even when SAM3 returns nothing for this frame, we still
            # need to update tracker state with empty data so carry-over and
            # contact_tracker don't lose frame continuity.
            if local_idx not in sam3_results:
                # Update tracker with empty state (no detections this frame)
                slot_masks: List[Optional[np.ndarray]] = [None] * num_slots
                slot_centroids: List[Optional[Tuple[float, float]]] = [None] * num_slots
                slot_detections: List[Optional[Detection]] = [None] * num_slots

                # Carry-over: try to recover keypoints from prev frame
                slot_detections = _carry_over_keypoints(
                    slot_detections, prev_slot_dets, prev_frame_centroids, slot_centroids,
                )

                # Save state for next frame
                prev_slot_dets = [copy.deepcopy(d) for d in slot_detections]
                prev_frame_centroids = list(slot_centroids)

                # Update contact tracker with empty state to keep continuity
                if contact_tracker is not None:
                    contact_tracker.update(
                        detections=[d for d in slot_detections if d is not None],
                        slot_masks=slot_masks,
                        slot_centroids=slot_centroids,
                        frame_idx=global_frame_idx,
                    )

                main_writer.write(frame_bgr)
                for w in unitary_writers:
                    if w is not None:
                        w.write(background)
                global_frame_idx += 1
                continue

            raw_masks = sam3_results[local_idx]['masks']
            new_centroids = [mask_centroid(m) for m in raw_masks]

            slot_masks: List[Optional[np.ndarray]] = [None] * num_slots
            slot_centroids: List[Optional[Tuple[float, float]]] = [None] * num_slots

            # Bug 1 fix: use flag instead of global_frame_idx == 0 (robust to
            # multi-GPU and frame skips)
            if not first_assignment_done:
                # First valid frame: assign by detection order
                for i in range(min(len(raw_masks), num_slots)):
                    slot_masks[i] = raw_masks[i]
                    slot_centroids[i] = new_centroids[i]
                if any(c is not None for c in slot_centroids):
                    first_assignment_done = True
            else:
                assignment = assign_identities_by_centroid(
                    list(raw_masks), new_centroids, prev_centroids, num_slots,
                    max_dist=max_assignment_dist,
                )
                for new_i, slot_idx in enumerate(assignment):
                    if slot_idx >= 0:
                        slot_masks[slot_idx] = raw_masks[new_i]
                        slot_centroids[slot_idx] = new_centroids[new_i]

            # Update "last known" centroids (for identity continuity)
            for s in range(num_slots):
                if slot_centroids[s] is not None:
                    prev_centroids[s] = slot_centroids[s]

            # Build composites and run YOLO
            slot_detections: List[Optional[Detection]] = [None] * num_slots
            composites: List[Optional[np.ndarray]] = [None] * num_slots

            for slot_idx in range(num_slots):
                if slot_masks[slot_idx] is None:
                    continue

                other_masks = [slot_masks[j] for j in range(num_slots)
                               if j != slot_idx and slot_masks[j] is not None]
                if other_masks:
                    union_other = np.zeros_like(slot_masks[slot_idx], dtype=bool)
                    for m in other_masks:
                        union_other = union_other | m
                else:
                    union_other = np.zeros_like(slot_masks[slot_idx], dtype=bool)

                composite = erase_other_rat(
                    frame_bgr=frame_bgr,
                    background_bgr=background,
                    mask_self=slot_masks[slot_idx],
                    mask_other=union_other,
                    dilate_px=erase_dilate_px,
                    feather_px=erase_feather_px,
                )
                composites[slot_idx] = composite

                composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                dets = detect_only(
                    yolo_model, composite_rgb,
                    confidence=yolo_conf,
                    keypoint_names=kpt_names,
                )

                if dets:
                    if mask_dilate_for_pick > 0:
                        k = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (mask_dilate_for_pick * 2 + 1, mask_dilate_for_pick * 2 + 1),
                        )
                        mask_for_pick = cv2.dilate(
                            slot_masks[slot_idx].astype(np.uint8), k,
                        ).astype(bool)
                    else:
                        mask_for_pick = slot_masks[slot_idx]

                    chosen = pick_detection_for_slot(
                        detections=dets,
                        mask_self=mask_for_pick,
                    )
                    if chosen is not None:
                        chosen.track_id = slot_idx + 1
                        slot_detections[slot_idx] = chosen
                        yolo_hit_counts[slot_idx] += 1

            # Carry-over for missing slots
            missing_before = sum(1 for d in slot_detections if d is None)
            slot_detections = _carry_over_keypoints(
                slot_detections, prev_slot_dets, prev_frame_centroids, slot_centroids,
            )
            missing_after = sum(1 for d in slot_detections if d is None)
            if missing_before > missing_after:
                carried_frames += 1

            # Save state for next frame's carry-over
            prev_slot_dets = [copy.deepcopy(d) for d in slot_detections]
            prev_frame_centroids = list(slot_centroids)

            # Contact tracker
            if contact_tracker is not None:
                contact_tracker.update(
                    detections=[d for d in slot_detections if d is not None],
                    slot_masks=slot_masks,
                    slot_centroids=slot_centroids,
                    frame_idx=global_frame_idx,
                )

            # Draw helper
            def _draw_slot_overlay(canvas, slot_idx):
                if slot_masks[slot_idx] is None:
                    return canvas
                color = colors[slot_idx % len(colors)]
                overlay_mask = np.zeros_like(canvas)
                overlay_mask[slot_masks[slot_idx]] = color
                canvas = cv2.addWeighted(canvas, 1.0, overlay_mask, 0.4, 0)
                if slot_detections[slot_idx] is not None:
                    det = slot_detections[slot_idx]
                    if det.keypoints is not None:
                        for kp in det.keypoints:
                            if kp.conf >= kpt_min_conf:
                                cv2.circle(canvas, (int(kp.x), int(kp.y)), 4, color, -1)
                if slot_centroids[slot_idx] is not None:
                    cx, cy = int(slot_centroids[slot_idx][0]), int(slot_centroids[slot_idx][1])
                    cv2.putText(canvas, f"R{slot_idx+1}", (cx - 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                return canvas

            # Main overlay
            overlay = frame_bgr.copy()
            for slot_idx in range(num_slots):
                overlay = _draw_slot_overlay(overlay, slot_idx)
            main_writer.write(overlay)

            # Unitary videos (clean background + only rat i)
            if write_individual_videos:
                for slot_idx in range(num_slots):
                    iw = unitary_writers[slot_idx]
                    if iw is None:
                        continue
                    if slot_masks[slot_idx] is None:
                        iw.write(background)
                    else:
                        unitary_clean = compose_isolated_video_frame(
                            frame_bgr=frame_bgr,
                            background_bgr=background,
                            mask_self=slot_masks[slot_idx],
                            feather_px=unitary_feather_px,
                        )
                        unitary_clean = _draw_slot_overlay(unitary_clean, slot_idx)
                        iw.write(unitary_clean)

            global_frame_idx += 1

        del sam3_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Phase 7: finalize
    main_writer.release()
    for w in unitary_writers:
        if w is not None:
            w.release()
    logger.info("Output videos written: %s", overlays_dir)

    if contact_tracker is not None:
        summary = contact_tracker.finalize()
        total_bouts = sum(
            v.get("total_bouts", 0)
            for v in summary.get("contact_type_summary", {}).values()
        )
        logger.info("ContactTracker finalized. Total bouts: %d", total_bouts)

        try:
            from scripts.postprocess_contacts_simple import run_postprocess
            run_postprocess(run_dir / "contacts", fps=fps)
        except Exception as e:
            logger.warning("Contact post-processing failed: %s", e)

    for i in range(num_slots):
        logger.info(
            "Rat %d: pose detected in %d/%d frames (%.1f%%)",
            i + 1, yolo_hit_counts[i], global_frame_idx,
            100 * yolo_hit_counts[i] / max(global_frame_idx, 1),
        )
    logger.info(
        "Frames with any carry-over: %d/%d (%.1f%%)",
        carried_frames, global_frame_idx,
        100 * carried_frames / max(global_frame_idx, 1),
    )

    logger.info("Pipeline complete. Output directory: %s", run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="SAM3 composite pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        cli_overrides=args.overrides,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        chunk_id=args.chunk_id,
    )


if __name__ == "__main__":
    main()