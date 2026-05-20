"""
SAM3 composite pipeline.

Same architecture as isolated_composite but using SAM3 (HuggingFace) instead
of SAM2 video predictor:

  - SAM3 detects "mouse" automatically via text prompt (no manual bboxes)
  - Background median computed once (shared with isolated_composite)
  - Per-rat composites: erase other rats using background
  - YOLO runs on composites (single-rat frames) for keypoint detection
  - ContactTrackerV2 + IndividualMetricsCalculator for analysis
  - Output: overlay video + N unitary videos + CSVs + JSONs + report.pdf

Inter-chunk identity assignment:
  - SAM3 detects N "mouse" objects per chunk but their order is arbitrary
  - For chunks > 0: use Hungarian-like centroid matching with last frame
    of previous chunk to maintain identity continuity

Token:
  - Reads HF_TOKEN from environment variable
  - Login is silent if HF_TOKEN is missing (will fail to load model)

Usage:
  export HF_TOKEN="hf_..."
  python -m src.pipelines.sam3_composite.run \\
      --config configs/hpc_sam3_composite.yaml \\
      contacts.enabled=true
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.common.config_loader import load_config, setup_run_dir, setup_logging, get_device
from src.common.utils import Detection
from src.common.io_video import create_video_writer
from src.common.model_loaders import load_yolo
from src.common.yolo_inference import detect_only
from src.common.contacts_v2 import ContactTrackerV2
from src.pipelines.isolated_composite.composition import (
    load_or_compute_background,
    erase_other_rat,
    pick_detection_for_slot,
)

# Default chunk size if not specified in YAML
DEFAULT_CHUNK_SIZE = 150

logger = logging.getLogger(__name__)


# ============================================================================
# SAM3 LOADER
# ============================================================================

def load_sam3_model(device: str):
    """Load SAM3 model + processor from HuggingFace.

    Requires HF_TOKEN env variable to be set.
    Applies the necessary forward patch for transformers compatibility.
    """
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

    # Apply the patch needed for transformers compatibility
    # (text_embeds.pooler_output extraction)
    import transformers.models.sam3.modeling_sam3 as sam3_module
    original_forward = sam3_module.Sam3Model.forward




    def patched_forward(self, pixel_values=None, vision_embeds=None, input_ids=None,
                        attention_mask=None, text_embeds=None, input_boxes=None,
                        input_boxes_labels=None, **kwargs):
        # Wrap text_embeds tensor in a fake object so internal `.pooler_output` access works
        if text_embeds is not None and not hasattr(text_embeds, 'pooler_output'):
            class _Wrapper:
                def __init__(self, tensor):
                    self.pooler_output = tensor
                    self._tensor = tensor
                def __getattr__(self, name):
                    return getattr(self._tensor, name)
            text_embeds = _Wrapper(text_embeds)
        return original_forward(self, pixel_values, vision_embeds, input_ids,
                                attention_mask, text_embeds, input_boxes,
                                input_boxes_labels, **kwargs)



    sam3_module.Sam3Model.forward = patched_forward
    logger.info("SAM3 model loaded and patched.")

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
    """Extract frames from video into memory (RGB numpy arrays).

    Returns: (frames_list, fps, width, height)
    """
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
    """Compute centroid of a binary mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def assign_identities_by_centroid(
    new_masks: List[np.ndarray],
    new_centroids: List[Optional[Tuple[float, float]]],
    prev_centroids: List[Optional[Tuple[float, float]]],
    num_slots: int,
) -> List[int]:
    """Greedy nearest-neighbor assignment of new detections to existing slots.

    Returns: list of slot indices (length = len(new_masks)).
    A value of -1 means "no slot assigned" (extra detection, drop it).
    """
    n_new = len(new_centroids)
    assignment = [-1] * n_new

    # Build distance matrix [n_new x num_slots]
    dist = np.full((n_new, num_slots), float("inf"))
    for i in range(n_new):
        if new_centroids[i] is None:
            continue
        for s in range(num_slots):
            if prev_centroids[s] is None:
                continue
            dx = new_centroids[i][0] - prev_centroids[s][0]
            dy = new_centroids[i][1] - prev_centroids[s][1]
            dist[i, s] = np.sqrt(dx * dx + dy * dy)

    # Greedy assignment (smallest distance first)
    used_new = set()
    used_slots = set()
    while len(used_slots) < num_slots and len(used_new) < n_new:
        # Find minimum distance among unassigned
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
# SAM3 PROCESSING
# ============================================================================

def process_chunk_with_sam3(
    model,
    processor,
    frames_chunk: List[np.ndarray],
    text_prompt: str,
    device: str,
    score_threshold: float = 0.5,
) -> Dict[int, Dict[str, Any]]:
    """Run SAM3 on a chunk of frames.

    Returns dict: frame_idx (local to chunk) -> {'masks': np.ndarray, 'scores': list}
    """
    from PIL import Image

    # Convert frames to PIL Images (SAM3 expects PIL)
    pil_frames = [Image.fromarray(f) for f in frames_chunk]

    session = processor.init_video_session(
        video=pil_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    session = processor.add_text_prompt(
        inference_session=session,
        text=text_prompt,
    )

    results = {}
    for model_out in model.propagate_in_video_iterator(inference_session=session):
        processed = processor.postprocess_outputs(session, model_out)
        masks = processed['masks'].cpu().numpy()  # (N_obj, H, W) float
        scores = processed['scores'].cpu().tolist()

        # Filter by score
        keep = [i for i, s in enumerate(scores) if s >= score_threshold]
        if not keep:
            results[model_out.frame_idx] = {'masks': np.zeros((0,) + masks.shape[1:], dtype=bool),
                                            'scores': []}
            continue

        masks_filtered = masks[keep] > 0.5  # binarize
        scores_filtered = [scores[i] for i in keep]
        results[model_out.frame_idx] = {
            'masks': masks_filtered,
            'scores': scores_filtered,
        }

    del session
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    config_path: str | Path,
    cli_overrides: List[str] | None = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    chunk_id: Optional[int] = None,
) -> Path:
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
    kpt_min_conf = config.get("detection", {}).get("keypoint_min_conf", 0.3)
    chunk_size = config.get("scan", {}).get("chunk_size", DEFAULT_CHUNK_SIZE)

    sam3_cfg = config.get("sam3", {})
    text_prompt = sam3_cfg.get("text_prompt", "mouse")
    score_threshold = float(sam3_cfg.get("score_threshold", 0.5))

    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
    ]
    codec = config.get("output", {}).get("video_codec", "XVID")
    ext = ".avi" if codec == "XVID" else ".mp4"
    today = date.today().strftime("%Y-%m-%d")

    # ==================================================================
    # Phase 1: Get video info + extract all frames into memory
    # ==================================================================
    logger.info("Extracting frames from %s ...", video_path)
    all_frames, fps, width, height = extract_frames_to_memory(
        video_path, start_frame=start_frame, end_frame=end_frame, max_frames=max_frames,
    )
    num_frames = len(all_frames)
    logger.info("Loaded %d frames @ %.1f fps (%dx%d)", num_frames, fps, width, height)

    if num_frames == 0:
        logger.error("No frames extracted.")
        return run_dir

    # ==================================================================
    # Phase 2: Compute background median (shared helper)
    # ==================================================================
    background = load_or_compute_background(config, video_path, run_dir)
    logger.info("Background computed/loaded.")

    # ==================================================================
    # Phase 3: Load SAM3 and YOLO
    # ==================================================================
    sam3_model, sam3_processor = load_sam3_model(device)
    yolo_path = config.get("models", {}).get("yolo_path")
    yolo_model = load_yolo(yolo_path, device)

    # ==================================================================
    # Phase 4: Prepare output writers
    # ==================================================================
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Main overlay video
    out_video_path = overlays_dir / f"sam3_composite_{today}{ext}"
    main_writer = create_video_writer(out_video_path, fps, width, height, codec=codec)

    # Per-slot unitary videos
    unitary_writers = []
    for i in range(num_slots):
        path = overlays_dir / f"sam3_unitary_rat{i+1}_{today}{ext}"
        unitary_writers.append(create_video_writer(path, fps, width, height, codec=codec))

    # ==================================================================
    # Phase 5: Contact tracker
    # ==================================================================
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
        logger.info("ContactTrackerV2 initialized (output: %s)", contacts_dir)

    # ==================================================================
    # Phase 6: Process in chunks
    # ==================================================================
    n_chunks = (num_frames + chunk_size - 1) // chunk_size
    logger.info("Processing %d frames in %d chunks of %d frames",
                num_frames, n_chunks, chunk_size)

    prev_centroids: List[Optional[Tuple[float, float]]] = [None] * num_slots
    global_frame_idx = 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_frames)
        chunk_frames = all_frames[chunk_start:chunk_end]
        logger.info("Chunk %d/%d (frames %d-%d, %d frames)",
                    chunk_idx + 1, n_chunks, chunk_start, chunk_end - 1, len(chunk_frames))

        # --- Run SAM3 on chunk ---
        sam3_results = process_chunk_with_sam3(
            sam3_model, sam3_processor, chunk_frames,
            text_prompt=text_prompt, device=device,
            score_threshold=score_threshold,
        )

        # --- For each frame in chunk: process ---
        for local_idx in range(len(chunk_frames)):
            frame_rgb = chunk_frames[local_idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Get masks from SAM3
            if local_idx not in sam3_results:
                # Missing frame - write empty
                main_writer.write(frame_bgr)
                for w in unitary_writers:
                    w.write(frame_bgr)
                global_frame_idx += 1
                continue

            raw_masks = sam3_results[local_idx]['masks']  # (N, H, W) bool

            # Compute centroids of new masks
            new_centroids = [mask_centroid(m) for m in raw_masks]

            # Assign identities
            slot_masks: List[Optional[np.ndarray]] = [None] * num_slots
            slot_centroids: List[Optional[Tuple[float, float]]] = [None] * num_slots

            if chunk_idx == 0 and local_idx == 0:
                # First frame ever: assign by detection order (limited to num_slots)
                for i in range(min(len(raw_masks), num_slots)):
                    slot_masks[i] = raw_masks[i]
                    slot_centroids[i] = new_centroids[i]
            else:
                # Use centroid matching against previous frame
                assignment = assign_identities_by_centroid(
                    list(raw_masks), new_centroids, prev_centroids, num_slots,
                )
                for new_i, slot_idx in enumerate(assignment):
                    if slot_idx >= 0:
                        slot_masks[slot_idx] = raw_masks[new_i]
                        slot_centroids[slot_idx] = new_centroids[new_i]

            # Update prev_centroids for next frame (keep last known if missing)
            for s in range(num_slots):
                if slot_centroids[s] is not None:
                    prev_centroids[s] = slot_centroids[s]

            # --- Build composites and run YOLO ---
            slot_detections: List[Optional[Detection]] = [None] * num_slots
            composites: List[np.ndarray] = []

            for slot_idx in range(num_slots):
                if slot_masks[slot_idx] is None:
                    composites.append(frame_bgr.copy())
                    continue

                # Erase OTHER rats
                other_masks = [slot_masks[j] for j in range(num_slots)
                               if j != slot_idx and slot_masks[j] is not None]
                if other_masks:
                    union_other = np.zeros_like(slot_masks[slot_idx], dtype=bool)
                    for m in other_masks:
                        union_other = union_other | m
                    composite = erase_other_rat(frame_bgr, union_other, background)
                else:
                    composite = frame_bgr.copy()
                composites.append(composite)

                # YOLO on composite
                dets = detect_only(yolo_model, composite, device=device)
                if dets:
                    chosen = pick_detection_for_slot(
                        dets, slot_masks[slot_idx], slot_centroids[slot_idx],
                    )
                    slot_detections[slot_idx] = chosen

            # --- ContactTracker update ---
            if contact_tracker is not None:
                contact_tracker.update(
                    detections=[d for d in slot_detections if d is not None],
                    slot_masks=slot_masks,
                    slot_centroids=slot_centroids,
                    frame_idx=global_frame_idx,
                )

            # --- Helper: draw mask + keypoints + label for a single slot ---
            def _draw_slot_overlay(canvas: np.ndarray, slot_idx: int) -> np.ndarray:
                """Draw mask overlay + keypoints + label for slot_idx onto canvas.

                Uses the same color as the main overlay video for consistency.
                Returns the modified canvas (in-place compatible).
                """
                if slot_masks[slot_idx] is None:
                    return canvas
                color = colors[slot_idx % len(colors)]

                # Mask overlay (semi-transparent)
                overlay_mask = np.zeros_like(canvas)
                overlay_mask[slot_masks[slot_idx]] = color
                canvas = cv2.addWeighted(canvas, 1.0, overlay_mask, 0.4, 0)

                # Keypoints
                if slot_detections[slot_idx] is not None:
                    det = slot_detections[slot_idx]
                    if det.keypoints is not None:
                        for kp in det.keypoints:
                            if kp.conf >= kpt_min_conf:
                                cv2.circle(canvas, (int(kp.x), int(kp.y)), 4, color, -1)

                # Label
                if slot_centroids[slot_idx] is not None:
                    cx, cy = int(slot_centroids[slot_idx][0]), int(slot_centroids[slot_idx][1])
                    cv2.putText(canvas, f"R{slot_idx+1}", (cx - 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                return canvas

            # --- Render main overlay video (all rats on the original frame) ---
            overlay = frame_bgr.copy()
            for slot_idx in range(num_slots):
                overlay = _draw_slot_overlay(overlay, slot_idx)
            main_writer.write(overlay)

            # --- Render unitary videos (composite + mask + keypoints for that slot) ---
            for slot_idx in range(num_slots):
                unitary_frame = composites[slot_idx].copy()
                unitary_frame = _draw_slot_overlay(unitary_frame, slot_idx)
                unitary_writers[slot_idx].write(unitary_frame)

            global_frame_idx += 1

        # Cleanup chunk memory
        del sam3_results
        torch.cuda.empty_cache()
        gc.collect()

    # ==================================================================
    # Phase 7: Finalize
    # ==================================================================
    main_writer.release()
    for w in unitary_writers:
        w.release()
    logger.info("Output videos written: %s", overlays_dir)

    if contact_tracker is not None:
        summary = contact_tracker.finalize()
        logger.info("ContactTracker finalized. Total bouts: %d", summary.get("total_bouts", 0))

    logger.info("Pipeline complete. Output directory: %s", run_dir)
    return run_dir


def main() -> None:
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