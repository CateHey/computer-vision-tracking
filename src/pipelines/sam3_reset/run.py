"""
SAM3 reset pipeline.

Same as sam3_composite but with automatic failure detection and recovery:

  Loop chunks → SAM3 video tracking
    Per frame:
      - Detect failures (area growth/shrink/overlap)
      - If failure detected (and not in cooldown):
          - SAM3 IMAGE on failed frame (no context, fresh segmentation)
          - Reassign identities using last K healthy frames' average centroids
          - Re-init SAM3 video session with fresh masks (cascade fallbacks)
          - Continue tracking
      - YOLO on composites
      - ContactTrackerV2 update
      - Render videos

Outputs (same as sam3_composite + reset log):
  - sam3_reset_<date>.avi              (main overlay)
  - sam3_reset_unitary_rat{i}_<date>.avi  (per-rat unitary)
  - contacts/                          (full contact analysis)
  - resets_log.csv                     (audit trail of all resets)
"""

from __future__ import annotations

import argparse
import csv
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
from src.common.constants import DEFAULT_CHUNK_SIZE
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
from src.pipelines.sam3_reset.failure_detection import (
    detect_frame_failure,
    HealthyFrameHistory,
    FailureReport,
)
from src.pipelines.sam3_reset.reset_strategies import (
    sam3_image_segment,
    reassign_identities,
    init_new_session_cascade,
)
from src.pipelines.sam3_composite.run import (
    load_sam3_model,
    extract_frames_to_memory,
    mask_centroid,
    assign_identities_by_centroid,
)

logger = logging.getLogger(__name__)


# ============================================================================
# RESET LOG WRITER
# ============================================================================

class ResetLogger:
    """Writes resets_log.csv with audit trail of all reset events."""

    HEADERS = [
        "frame_idx", "time_sec", "reason", "affected_slots",
        "area_before", "area_after", "overlap_value", "overlap_pair",
        "strategy_used", "success", "skipped_reason",
    ]

    def __init__(self, output_path: Path):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADERS)
        self._writer.writeheader()
        self._count = 0

    def log(self, **kwargs):
        row = {h: kwargs.get(h, "") for h in self.HEADERS}
        # Normalize lists/tuples to strings
        for k in ("affected_slots", "overlap_pair"):
            if isinstance(row[k], (list, tuple)):
                row[k] = str(row[k])
        self._writer.writerow(row)
        self._count += 1

    def close(self):
        self._file.close()
        logger.info("Reset log written: %s (%d events)", self.path, self._count)


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
    tag = f"sam3_reset_chunk{chunk_id}" if chunk_id is not None else "sam3_reset"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)

    logger.info("Starting SAM3 reset pipeline")
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

    # ===== Reset / failure detection config =====
    reset_cfg = config.get("reset", {})
    reset_enabled = reset_cfg.get("enabled", True)
    cooldown_frames = int(reset_cfg.get("cooldown_frames", 10))
    min_history_frames = int(reset_cfg.get("min_history_frames", 30))
    lookback_frames = int(reset_cfg.get("lookback_frames", 5))

    fd_cfg = config.get("failure_detection", {})
    area_growth_thr = float(fd_cfg.get("area_growth_threshold", 2.5))
    area_shrink_thr = float(fd_cfg.get("area_shrink_threshold", 0.10))
    overlap_thr = float(fd_cfg.get("overlap_threshold", 0.90))

    rs_cfg = config.get("reset_strategies", {})
    try_masks = bool(rs_cfg.get("try_masks", True))
    try_points = bool(rs_cfg.get("try_points", True))
    try_bboxes = bool(rs_cfg.get("try_bboxes", True))
    points_per_mask = int(rs_cfg.get("points_per_mask", 7))

    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
    ]
    codec = config.get("output", {}).get("video_codec", "XVID")
    ext = ".avi" if codec == "XVID" else ".mp4"
    today = date.today().strftime("%Y-%m-%d")

    # ==================================================================
    # Phase 1: Extract frames
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
    # Phase 2: Background + models
    # ==================================================================
    background = load_or_compute_background(config, video_path, run_dir)
    logger.info("Background computed/loaded.")

    sam3_model, sam3_processor = load_sam3_model(device)
    yolo_path = config.get("models", {}).get("yolo_path")
    yolo_model = load_yolo(yolo_path, device)

    # ==================================================================
    # Phase 3: Output writers
    # ==================================================================
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = overlays_dir / f"sam3_reset_{today}{ext}"
    main_writer = create_video_writer(out_video_path, fps, width, height, codec=codec)
    unitary_writers = []
    for i in range(num_slots):
        path = overlays_dir / f"sam3_reset_unitary_rat{i+1}_{today}{ext}"
        unitary_writers.append(create_video_writer(path, fps, width, height, codec=codec))

    # ==================================================================
    # Phase 4: Trackers + history
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

    history = HealthyFrameHistory(lookback=lookback_frames)
    reset_logger = ResetLogger(run_dir / "resets_log.csv")
    frames_since_last_reset = cooldown_frames + 1  # start ready to reset

    prev_centroids: List[Optional[Tuple[float, float]]] = [None] * num_slots
    prev_masks: List[Optional[np.ndarray]] = [None] * num_slots
    global_frame_idx = 0

    # ==================================================================
    # Phase 5: Process chunks
    # ==================================================================
    n_chunks = (num_frames + chunk_size - 1) // chunk_size
    logger.info("Processing %d frames in %d chunks of %d frames",
                num_frames, n_chunks, chunk_size)

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_frames)
        chunk_frames = all_frames[chunk_start:chunk_end]
        logger.info("Chunk %d/%d (frames %d-%d)",
                    chunk_idx + 1, n_chunks, chunk_start, chunk_end - 1)

        # ---- Run SAM3 video on chunk ----
        sam3_results = _run_sam3_video_chunk(
            sam3_model, sam3_processor, chunk_frames,
            text_prompt=text_prompt, device=device,
            score_threshold=score_threshold,
        )

        # ---- Process each frame ----
        local_idx = 0
        while local_idx < len(chunk_frames):
            frame_rgb = chunk_frames[local_idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if local_idx not in sam3_results:
                # No SAM3 output for this frame
                main_writer.write(frame_bgr)
                for w in unitary_writers:
                    w.write(frame_bgr)
                local_idx += 1
                global_frame_idx += 1
                continue

            raw_masks = sam3_results[local_idx]['masks']
            new_centroids = [mask_centroid(m) for m in raw_masks]

            # ---- Assign identities to slots ----
            slot_masks: List[Optional[np.ndarray]] = [None] * num_slots
            slot_centroids: List[Optional[Tuple[float, float]]] = [None] * num_slots

            if global_frame_idx == 0:
                # First frame: assign by detection order
                for i in range(min(len(raw_masks), num_slots)):
                    slot_masks[i] = raw_masks[i]
                    slot_centroids[i] = new_centroids[i]
            else:
                assignment = assign_identities_by_centroid(
                    list(raw_masks), new_centroids, prev_centroids, num_slots,
                )
                for new_i, slot_idx in enumerate(assignment):
                    if slot_idx >= 0:
                        slot_masks[slot_idx] = raw_masks[new_i]
                        slot_centroids[slot_idx] = new_centroids[new_i]

            # ==========================================================
            # FAILURE DETECTION
            # ==========================================================
            failure_report = FailureReport(is_failed=False, reason="none", affected_slots=[])
            if reset_enabled and global_frame_idx >= min_history_frames:
                failure_report = detect_frame_failure(
                    prev_masks, slot_masks,
                    growth_thr=area_growth_thr,
                    shrink_thr=area_shrink_thr,
                    overlap_thr=overlap_thr,
                )

            # ==========================================================
            # RESET LOGIC
            # ==========================================================
            if failure_report.is_failed:
                time_sec = global_frame_idx / fps
                if frames_since_last_reset <= cooldown_frames:
                    # In cooldown — log skipped
                    reset_logger.log(
                        frame_idx=global_frame_idx,
                        time_sec=round(time_sec, 3),
                        reason=failure_report.reason,
                        affected_slots=failure_report.affected_slots,
                        area_before=failure_report.area_before,
                        area_after=failure_report.area_after,
                        overlap_value=failure_report.overlap_value,
                        overlap_pair=failure_report.overlap_pair,
                        strategy_used="",
                        success=False,
                        skipped_reason="cooldown",
                    )
                    logger.info("Frame %d: failure detected (%s) but in cooldown, skipping reset",
                                global_frame_idx, failure_report.reason)
                else:
                    logger.info("Frame %d: %s — triggering RESET",
                                global_frame_idx, str(failure_report))

                    # ---- Perform reset ----
                    reset_success, strategy_used, new_slot_masks, new_slot_centroids = _do_reset(
                        sam3_model=sam3_model,
                        sam3_processor=sam3_processor,
                        frame_rgb=frame_rgb,
                        text_prompt=text_prompt,
                        device=device,
                        score_threshold=score_threshold,
                        history=history,
                        num_slots=num_slots,
                        try_masks=try_masks,
                        try_points=try_points,
                        try_bboxes=try_bboxes,
                        points_per_mask=points_per_mask,
                    )

                    if reset_success and any(m is not None for m in new_slot_masks):
                        # Replace current frame's masks with fresh ones
                        slot_masks = new_slot_masks
                        slot_centroids = new_slot_centroids
                        # Now we need to re-run SAM3 video for the rest of the chunk
                        # using these new masks as the starting point
                        remaining_frames = chunk_frames[local_idx:]
                        sam3_results = _run_sam3_video_chunk_with_init(
                            sam3_model, sam3_processor, remaining_frames,
                            init_masks=slot_masks,
                            text_prompt=text_prompt, device=device,
                            score_threshold=score_threshold,
                            try_masks=try_masks, try_points=try_points,
                            try_bboxes=try_bboxes, points_per_mask=points_per_mask,
                        )
                        # Re-index sam3_results so it aligns with local_idx
                        # (sam3_results now starts from current local_idx as 0)
                        sam3_results = {(local_idx + k): v for k, v in sam3_results.items()}

                        frames_since_last_reset = 0
                        reset_logger.log(
                            frame_idx=global_frame_idx,
                            time_sec=round(time_sec, 3),
                            reason=failure_report.reason,
                            affected_slots=failure_report.affected_slots,
                            area_before=failure_report.area_before,
                            area_after=failure_report.area_after,
                            overlap_value=failure_report.overlap_value,
                            overlap_pair=failure_report.overlap_pair,
                            strategy_used=strategy_used,
                            success=True,
                            skipped_reason="",
                        )
                    else:
                        # Reset failed — use last good masks if available
                        logger.warning("Frame %d: RESET FAILED, falling back to previous masks",
                                       global_frame_idx)
                        slot_masks = [m for m in prev_masks]
                        slot_centroids = [c for c in prev_centroids]
                        reset_logger.log(
                            frame_idx=global_frame_idx,
                            time_sec=round(time_sec, 3),
                            reason=failure_report.reason,
                            affected_slots=failure_report.affected_slots,
                            area_before=failure_report.area_before,
                            area_after=failure_report.area_after,
                            overlap_value=failure_report.overlap_value,
                            overlap_pair=failure_report.overlap_pair,
                            strategy_used=strategy_used,
                            success=False,
                            skipped_reason="all_strategies_failed",
                        )

            # ---- Update history if frame is healthy ----
            if not failure_report.is_failed:
                history.push(slot_centroids)

            # ---- Update prev_* ----
            for s in range(num_slots):
                if slot_centroids[s] is not None:
                    prev_centroids[s] = slot_centroids[s]
                if slot_masks[s] is not None:
                    prev_masks[s] = slot_masks[s].copy()

            frames_since_last_reset += 1

            # ==========================================================
            # COMPOSITES + YOLO + CONTACTS + RENDER
            # ==========================================================
            slot_detections: List[Optional[Detection]] = [None] * num_slots
            composites: List[np.ndarray] = []

            for slot_idx in range(num_slots):
                if slot_masks[slot_idx] is None:
                    composites.append(frame_bgr.copy())
                    continue

                other_masks = [slot_masks[j] for j in range(num_slots)
                               if j != slot_idx and slot_masks[j] is not None]
                if other_masks:
                    union_other = np.zeros_like(slot_masks[slot_idx], dtype=bool)
                    for m in other_masks:
                        union_other = union_other | m
                    composite = erase_other_rat(
                        frame_bgr=frame_bgr,
                        background_bgr=background,
                        mask_self=slot_masks[slot_idx],
                        mask_other=union_other,
                    )
                else:
                    composite = frame_bgr.copy()
                composites.append(composite)

                composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                dets = detect_only(yolo_model, composite_rgb, confidence=0.25)
                if dets:
                    chosen = pick_detection_for_slot(
                        detections=dets,
                        mask_self=slot_masks[slot_idx],
                    )
                    slot_detections[slot_idx] = chosen

            if contact_tracker is not None:
                contact_tracker.update(
                    detections=[d for d in slot_detections if d is not None],
                    slot_masks=slot_masks,
                    slot_centroids=slot_centroids,
                    frame_idx=global_frame_idx,
                )

            # Helper for rendering (same as sam3_composite)
            def _draw_slot_overlay(canvas: np.ndarray, slot_idx: int) -> np.ndarray:
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

            # Render main + unitary videos
            overlay = frame_bgr.copy()
            for slot_idx in range(num_slots):
                overlay = _draw_slot_overlay(overlay, slot_idx)
            main_writer.write(overlay)

            for slot_idx in range(num_slots):
                unitary_frame = composites[slot_idx].copy()
                unitary_frame = _draw_slot_overlay(unitary_frame, slot_idx)
                unitary_writers[slot_idx].write(unitary_frame)

            local_idx += 1
            global_frame_idx += 1

        # Cleanup chunk
        del sam3_results
        torch.cuda.empty_cache()
        gc.collect()

    # ==================================================================
    # Phase 6: Finalize
    # ==================================================================
    main_writer.release()
    for w in unitary_writers:
        w.release()
    logger.info("Output videos written: %s", overlays_dir)

    reset_logger.close()

    if contact_tracker is not None:
        summary = contact_tracker.finalize()
        logger.info("ContactTracker finalized. Total bouts: %d", summary.get("total_bouts", 0))

    logger.info("Pipeline complete. Output directory: %s", run_dir)
    return run_dir


# ============================================================================
# HELPER: SAM3 video sobre chunk (sin init customizada)
# ============================================================================

def _run_sam3_video_chunk(
    model, processor, frames_chunk, text_prompt, device, score_threshold,
) -> Dict[int, Dict[str, Any]]:
    """Wrapper alrededor de process_chunk_with_sam3 importado, sin inicialización custom."""
    from src.pipelines.sam3_composite.run import process_chunk_with_sam3
    return process_chunk_with_sam3(
        model, processor, frames_chunk,
        text_prompt=text_prompt, device=device,
        score_threshold=score_threshold,
    )


# ============================================================================
# HELPER: SAM3 video sobre chunk con masks de inicialización (después de reset)
# ============================================================================

def _run_sam3_video_chunk_with_init(
    model, processor, frames_chunk, init_masks,
    text_prompt, device, score_threshold,
    try_masks=True, try_points=True, try_bboxes=True, points_per_mask=7,
) -> Dict[int, Dict[str, Any]]:
    """Como _run_sam3_video_chunk pero inicializa la sesión con masks específicas."""
    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames_chunk]

    session = processor.init_video_session(
        video=pil_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    # Try to init with the provided masks (cascade)
    success, strategy = init_new_session_cascade(
        processor, session, init_masks, frame_idx=0,
        try_masks=try_masks, try_points=try_points, try_bboxes=try_bboxes,
        points_per_mask=points_per_mask,
    )

    if not success:
        # Fallback: use text prompt
        logger.warning("All init strategies failed, falling back to text prompt")
        session = processor.add_text_prompt(inference_session=session, text=text_prompt)

    results = {}
    for model_out in model.propagate_in_video_iterator(inference_session=session):
        processed = processor.postprocess_outputs(session, model_out)
        masks = processed['masks'].cpu().numpy()
        scores = processed['scores'].cpu().tolist()

        keep = [i for i, s in enumerate(scores) if s >= score_threshold]
        if not keep:
            results[model_out.frame_idx] = {'masks': np.zeros((0,) + masks.shape[1:], dtype=bool),
                                            'scores': []}
            continue
        masks_filtered = masks[keep] > 0.5
        scores_filtered = [scores[i] for i in keep]
        results[model_out.frame_idx] = {'masks': masks_filtered, 'scores': scores_filtered}

    del session
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ============================================================================
# HELPER: ejecutar reset completo
# ============================================================================

def _do_reset(
    sam3_model, sam3_processor, frame_rgb, text_prompt, device, score_threshold,
    history: HealthyFrameHistory, num_slots: int,
    try_masks: bool, try_points: bool, try_bboxes: bool, points_per_mask: int,
) -> Tuple[bool, str, List[Optional[np.ndarray]], List[Optional[Tuple[float, float]]]]:
    """Ejecuta el reset completo:
       1) SAM3 image sobre frame
       2) Reasignar identidades vs historia
       3) Retornar masks reasignadas (la nueva sesión video se inicializará después)

    Returns:
        (success, strategy_used, slot_masks, slot_centroids)
        strategy_used aquí solo indica si las máscaras frescas se obtuvieron OK.
        La estrategia real de inicialización se decide en _run_sam3_video_chunk_with_init.
    """
    # Step 1: SAM3 image segmentation (no context)
    try:
        fresh_masks, fresh_scores = sam3_image_segment(
            sam3_model, sam3_processor, frame_rgb,
            text_prompt=text_prompt, device=device,
            score_threshold=score_threshold,
        )
    except Exception as e:
        logger.error("SAM3 image segmentation failed during reset: %s", e)
        return False, "none", [None] * num_slots, [None] * num_slots

    if len(fresh_masks) == 0:
        logger.warning("SAM3 image returned no masks during reset")
        return False, "none", [None] * num_slots, [None] * num_slots

    # Step 2: reassign identities using historical average centroids
    avg_centroids = history.get_average_centroids(num_slots)
    if all(c is None for c in avg_centroids):
        # No history yet — assign by detection order
        slot_masks = [None] * num_slots
        slot_centroids = [None] * num_slots
        for i in range(min(len(fresh_masks), num_slots)):
            slot_masks[i] = fresh_masks[i]
            slot_centroids[i] = mask_centroid(fresh_masks[i])
    else:
        slot_masks, slot_centroids = reassign_identities(
            fresh_masks, avg_centroids, num_slots,
        )

    return True, "fresh_masks_ready", slot_masks, slot_centroids


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 reset pipeline")
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