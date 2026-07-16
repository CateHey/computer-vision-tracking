"""
Cutie composite pipeline — TWO-PHASE architecture.

Phase 1 (SAM3):   frames [0, sam3_phase_frames) tracked with sam3_composite's
                  exact logic (SAM3 video + Forma B init points). Reliable
                  detection + identity from the start.

Phase 2 (Cutie):  Cutie is seeded with the masks from the LAST frame of phase 1
                  (frame sam3_phase_frames - 1) and tracks continuously for the
                  rest of the video. Safety nets (protection, fusion, checkpoint)
                  trigger SAM3 recovery when Cutie fails.

Both phases share the same per-frame processing (composites -> YOLO -> keypoints
-> carry-over -> contacts -> render) via _process_frame_outputs().

Usage:
  export HF_TOKEN="hf_..."
  python -m src.pipelines.cutie_composite.run \\
      --config configs/local_cutie_composite.yaml \\
      video_path=data/raw/multirat.avi detection.max_animals=6 contacts.enabled=true
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import logging
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
# Reuse SAM3 helpers + shared utilities from sam3_composite
from src.pipelines.sam3_composite.run import (
    load_sam3_model,
    sam3_image_segment,
    process_chunk_with_sam3,
    get_chunk_init_points,
    extract_frames_to_memory,
    mask_centroid,
    assign_identities_by_centroid,
    shift_mask_by_velocity,
    _carry_over_keypoints,
)
from src.pipelines.cutie_composite.cutie_tracker import (
    CutieTracker,
    slot_masks_to_index_mask,
)
from src.pipelines.cutie_composite.safety import (
    SafetyEvent,
    check_protection,
    check_fusion,
    checkpoint_decision,
    reassign_by_previous_frame,
    RecoveryController,
)

logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================

def select_first_n_masks(
    masks: np.ndarray,
    scores: List[float],
    n: int,
) -> Tuple[np.ndarray, List[float]]:
    """Keep first n masks by detection order (mirrors sam3_composite)."""
    if len(masks) <= n:
        return masks, scores
    logger.info("SAM3 detected %d rats, kept first %d by detection order", len(masks), n)
    return masks[:n], scores[:n]


class SafetyLogger:
    """Writes safety_log.csv with all safety-system triggers (phase 2)."""
    HEADERS = ["frame_idx", "time_sec", "system", "reason",
               "missing_slots", "detail", "sam3_ran", "recovered"]

    def __init__(self, output_path: Path):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADERS)
        self._writer.writeheader()
        self._count = 0

    def log(self, **kwargs):
        row = {h: kwargs.get(h, "") for h in self.HEADERS}
        if isinstance(row["missing_slots"], (list, tuple)):
            row["missing_slots"] = str(row["missing_slots"])
        self._writer.writerow(row)
        self._count += 1

    def close(self):
        self._file.close()
        logger.info("Safety log written: %s (%d events)", self.path, self._count)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config_path, cli_overrides=None, start_frame=0, end_frame=None):
    config = load_config(config_path, cli_overrides)
    run_dir = setup_run_dir(config, tag="cutie_composite")
    setup_logging(run_dir)

    logger.info("Starting Cutie composite pipeline (TWO-PHASE: SAM3 -> Cutie)")
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
    max_assignment_dist = config.get("detection", {}).get("max_assignment_dist", None)
    if max_assignment_dist is not None:
        max_assignment_dist = float(max_assignment_dist)

    sam3_cfg = config.get("sam3", {})
    text_prompt = sam3_cfg.get("text_prompt", "mouse")
    score_threshold = float(sam3_cfg.get("score_threshold", 0.5))
    # NEW: how many frames Phase 1 (SAM3) tracks before handing off to Cutie
    sam3_phase_frames = int(sam3_cfg.get("sam3_phase_frames", 125))

    cutie_cfg = config.get("cutie", {}) or {}
    cutie_max_internal = int(cutie_cfg.get("max_internal_size", 720))
    cutie_mem_every = int(cutie_cfg.get("mem_every", 5))
    cutie_max_mem = int(cutie_cfg.get("max_mem_frames", 5))
    cutie_long_term = bool(cutie_cfg.get("use_long_term", True))
    cutie_weights = cutie_cfg.get("weights_path", None)

    safety_cfg = config.get("safety", {}) or {}
    checkpoint_interval = int(safety_cfg.get("checkpoint_interval", 150))
    checkpoint_iou = float(safety_cfg.get("checkpoint_iou_threshold", 0.85))
    protection_enabled = bool(safety_cfg.get("protection_enabled", True))
    fusion_enabled = bool(safety_cfg.get("fusion_enabled", True))
    max_search_frames = int(safety_cfg.get("max_search_frames", 30))
    fusion_area_tol = float(safety_cfg.get("fusion_area_tolerance", 0.15))
    fusion_growth = float(safety_cfg.get("fusion_min_growth", 1.5))

    comp_cfg = config.get("composition", {}) or {}
    erase_dilate_px = int(comp_cfg.get("erase_dilate_px", 15))
    erase_feather_px = int(comp_cfg.get("feather_px", 5))
    mask_dilate_for_pick = int(comp_cfg.get("mask_dilate_for_pick", 7))
    unitary_feather_px = int(comp_cfg.get("unitary_feather_px", 3))
    max_mask_carry = int(comp_cfg.get("max_mask_carry_frames", 5))

    out_cfg = config.get("output", {}) or {}
    colors_raw = out_cfg.get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
        (255, 0, 255), (0, 255, 255)]
    codec = out_cfg.get("video_codec", "XVID")
    ext = ".avi" if codec == "XVID" else ".mp4"
    write_individual_videos = bool(out_cfg.get("write_individual_videos", True))
    today = date.today().strftime("%Y-%m-%d")

    # ---------------- Phase 1: frames ----------------
    logger.info("Extracting frames from %s ...", video_path)
    all_frames, fps, width, height = extract_frames_to_memory(
        video_path, start_frame=start_frame, end_frame=end_frame, max_frames=max_frames)
    num_frames = len(all_frames)
    logger.info("Loaded %d frames @ %.1f fps (%dx%d)", num_frames, fps, width, height)
    if num_frames == 0:
        logger.error("No frames extracted.")
        return run_dir
    frame_shape = (height, width)

    # Clamp phase split
    sam3_phase_frames = min(sam3_phase_frames, num_frames)
    logger.info("Phase split: SAM3 tracks frames [0, %d), Cutie tracks [%d, %d)",
                sam3_phase_frames, sam3_phase_frames - 1, num_frames)

    # ---------------- Background ----------------
    background = load_or_compute_background(config, video_path, run_dir)
    if background.shape[:2] != frame_shape:
        background = cv2.resize(background, (width, height))
    logger.info("Background ready.")

    # ---------------- Models ----------------
    sam3_model, sam3_processor = load_sam3_model(device)
    yolo_path = config.get("models", {}).get("yolo_path", "models/yolo/best.pt")
    yolo_model = load_yolo(yolo_path, device)
    logger.info("SAM3 + YOLO loaded (Cutie loaded after Phase 1).")

    # ---------------- Writers ----------------
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = overlays_dir / f"cutie_composite_{today}{ext}"
    main_writer = create_video_writer(out_video_path, fps, width, height, codec=codec)
    unitary_writers = []
    if write_individual_videos:
        for i in range(num_slots):
            path = overlays_dir / f"cutie_unitary_rat{i+1}_{today}{ext}"
            unitary_writers.append(create_video_writer(path, fps, width, height, codec=codec))
    else:
        unitary_writers = [None] * num_slots

    # ---------------- Trackers/loggers ----------------
    contacts_enabled = config.get("contacts", {}).get("enabled", False)
    contact_tracker = None
    if contacts_enabled:
        contact_tracker = ContactTrackerV2(
            output_dir=run_dir / "contacts", fps=fps, num_slots=num_slots,
            video_path=str(video_path), config=config)
        logger.info("ContactTrackerV2 initialized")

    safety_logger = SafetyLogger(run_dir / "safety_log.csv")

    # ---------------- Shared state ----------------
    state = {
        "prev_centroids": [None] * num_slots,
        "prev_frame_centroids": None,
        "prev_slot_dets": None,
        "prev_masks": [None] * num_slots,
        "prev_velocities": [(0.0, 0.0)] * num_slots,
        "missing_counter": [0] * num_slots,
        "yolo_hit_counts": [0] * num_slots,
        "carried_kpt_frames": 0,
        "carried_mask_frames": 0,
        "global_frame_idx": 0,
    }

    # Per-frame output processor (shared by both phases)
    def process_frame_outputs(frame_bgr, slot_masks, slot_centroids):
        """Composites -> YOLO -> keypoint carry-over -> contacts -> render.
        Mutates state for carry-over continuity. Writes video frames."""
        gidx = state["global_frame_idx"]

        # Mask carry-over (fill short gaps)
        frame_had_mask_carry = False
        for s in range(num_slots):
            if slot_masks[s] is not None:
                state["missing_counter"][s] = 0
                continue
            if state["prev_masks"][s] is None:
                continue
            state["missing_counter"][s] += 1
            if state["missing_counter"][s] > max_mask_carry:
                continue
            dx, dy = state["prev_velocities"][s]
            carried = shift_mask_by_velocity(state["prev_masks"][s], dx, dy)
            if carried.any():
                slot_masks[s] = carried
                slot_centroids[s] = mask_centroid(carried)
                frame_had_mask_carry = True
        if frame_had_mask_carry:
            state["carried_mask_frames"] += 1

        # Velocity update
        for s in range(num_slots):
            if slot_centroids[s] is not None and state["prev_centroids"][s] is not None:
                dx = slot_centroids[s][0] - state["prev_centroids"][s][0]
                dy = slot_centroids[s][1] - state["prev_centroids"][s][1]
                state["prev_velocities"][s] = (max(-50.0, min(50.0, dx)),
                                               max(-50.0, min(50.0, dy)))
        for s in range(num_slots):
            if slot_centroids[s] is not None:
                state["prev_centroids"][s] = slot_centroids[s]

        # Composites + YOLO
        slot_detections: List[Optional[Detection]] = [None] * num_slots
        for slot_idx in range(num_slots):
            if slot_masks[slot_idx] is None:
                continue
            other_masks = [slot_masks[j] for j in range(num_slots)
                           if j != slot_idx and slot_masks[j] is not None]
            union_other = np.zeros_like(slot_masks[slot_idx], dtype=bool)
            for m in other_masks:
                union_other = union_other | m
            composite = erase_other_rat(
                frame_bgr=frame_bgr, background_bgr=background,
                mask_self=slot_masks[slot_idx], mask_other=union_other,
                dilate_px=erase_dilate_px, feather_px=erase_feather_px)
            composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            dets = detect_only(yolo_model, composite_rgb, confidence=yolo_conf, keypoint_names=kpt_names)
            if dets:
                if mask_dilate_for_pick > 0:
                    k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (mask_dilate_for_pick * 2 + 1, mask_dilate_for_pick * 2 + 1))
                    mask_for_pick = cv2.dilate(slot_masks[slot_idx].astype(np.uint8), k).astype(bool)
                else:
                    mask_for_pick = slot_masks[slot_idx]
                chosen = pick_detection_for_slot(detections=dets, mask_self=mask_for_pick)
                if chosen is not None:
                    chosen.track_id = slot_idx + 1
                    slot_detections[slot_idx] = chosen
                    state["yolo_hit_counts"][slot_idx] += 1

        # Keypoint carry-over
        missing_before = sum(1 for d in slot_detections if d is None)
        slot_detections = _carry_over_keypoints(
            slot_detections, state["prev_slot_dets"],
            state["prev_frame_centroids"], slot_centroids)
        missing_after = sum(1 for d in slot_detections if d is None)
        if missing_before > missing_after:
            state["carried_kpt_frames"] += 1

        state["prev_slot_dets"] = [copy.deepcopy(d) for d in slot_detections]
        state["prev_frame_centroids"] = list(slot_centroids)

        # Contacts
        if contact_tracker is not None:
            contact_tracker.update(
                detections=[d for d in slot_detections if d is not None],
                slot_masks=slot_masks, slot_centroids=slot_centroids, frame_idx=gidx)

        # Render
        def _draw(canvas, slot_idx):
            if slot_masks[slot_idx] is None:
                return canvas
            color = colors[slot_idx % len(colors)]
            om = np.zeros_like(canvas)
            om[slot_masks[slot_idx]] = color
            canvas = cv2.addWeighted(canvas, 1.0, om, 0.4, 0)
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

        overlay = frame_bgr.copy()
        for slot_idx in range(num_slots):
            overlay = _draw(overlay, slot_idx)
        main_writer.write(overlay)

        if write_individual_videos:
            for slot_idx in range(num_slots):
                iw = unitary_writers[slot_idx]
                if iw is None:
                    continue
                if slot_masks[slot_idx] is None:
                    iw.write(background)
                else:
                    uc = compose_isolated_video_frame(
                        frame_bgr=frame_bgr, background_bgr=background,
                        mask_self=slot_masks[slot_idx], feather_px=unitary_feather_px)
                    uc = _draw(uc, slot_idx)
                    iw.write(uc)

        # Update prev_masks
        for s in range(num_slots):
            if slot_masks[s] is not None:
                state["prev_masks"][s] = slot_masks[s].copy()

    # ==================================================================
    # PHASE 1 — SAM3 (frames [0, sam3_phase_frames))
    # Uses sam3_composite's exact logic: SAM3 video + Forma B init points.
    # ==================================================================
    logger.info("=== PHASE 1: SAM3 tracking frames [0, %d) ===", sam3_phase_frames)
    phase1_frames = all_frames[:sam3_phase_frames]

    # Forma B: init points from SAM3 image on frame 0 (first chunk => detection order)
    chunk_init_points = get_chunk_init_points(
        sam3_model, sam3_processor,
        first_frame_rgb=phase1_frames[0],
        text_prompt=text_prompt, device=device, score_threshold=score_threshold,
        num_slots=num_slots, prev_centroids=state["prev_centroids"],
        is_first_chunk=True, manual_init_points=None, max_dist=max_assignment_dist)

    # SAM3 video over the whole phase-1 clip (basic, no checkpoints)
    sam3_results = process_chunk_with_sam3(
        sam3_model, sam3_processor, phase1_frames,
        text_prompt=text_prompt, device=device,
        score_threshold=score_threshold, init_points=chunk_init_points)

    first_assignment_done = False
    last_good_slot_masks: List[Optional[np.ndarray]] = [None] * num_slots

    for local_idx in range(len(phase1_frames)):
        frame_rgb = phase1_frames[local_idx]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if local_idx not in sam3_results:
            # SAM3 returned nothing this frame — empty state, keep continuity
            slot_masks = [None] * num_slots
            slot_centroids = [None] * num_slots
            process_frame_outputs(frame_bgr, slot_masks, slot_centroids)
            state["global_frame_idx"] += 1
            continue

        raw_masks = sam3_results[local_idx]['masks']
        new_centroids = [mask_centroid(m) for m in raw_masks]
        slot_masks = [None] * num_slots
        slot_centroids = [None] * num_slots

        if not first_assignment_done:
            for i in range(min(len(raw_masks), num_slots)):
                slot_masks[i] = raw_masks[i]
                slot_centroids[i] = new_centroids[i]
            if any(c is not None for c in slot_centroids):
                first_assignment_done = True
        else:
            assignment = assign_identities_by_centroid(
                list(raw_masks), new_centroids, state["prev_centroids"],
                num_slots, max_dist=max_assignment_dist)
            for new_i, slot_idx in enumerate(assignment):
                if slot_idx >= 0:
                    slot_masks[slot_idx] = raw_masks[new_i]
                    slot_centroids[slot_idx] = new_centroids[new_i]

        process_frame_outputs(frame_bgr, slot_masks, slot_centroids)

        # Remember the last frame's masks (for Cutie handoff)
        last_good_slot_masks = [m.copy() if m is not None else None for m in slot_masks]
        state["global_frame_idx"] += 1

    del sam3_results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info("=== PHASE 1 complete. Handoff masks: %d/%d rats ===",
                sum(1 for m in last_good_slot_masks if m is not None), num_slots)

    # ==================================================================
    # HANDOFF — seed Cutie with the last SAM3 frame's masks
    # Cutie starts AT frame (sam3_phase_frames - 1) and re-emits it, then
    # continues from sam3_phase_frames onward.
    # ==================================================================
    handoff_idx = sam3_phase_frames - 1
    handoff_frame_rgb = all_frames[handoff_idx]

    index_mask, num_objects = slot_masks_to_index_mask(last_good_slot_masks, frame_shape)
    if num_objects == 0:
        logger.error("Phase 1 produced no masks for handoff. Cannot start Cutie.")
        # finalize what we have
        main_writer.release()
        for w in unitary_writers:
            if w is not None:
                w.release()
        safety_logger.close()
        return run_dir

    logger.info("Loading Cutie and seeding with handoff masks...")
    cutie = CutieTracker(
        device=device, max_internal_size=cutie_max_internal,
        mem_every=cutie_mem_every, max_mem_frames=cutie_max_mem,
        use_long_term=cutie_long_term, weights_path=cutie_weights)
    cutie.init_with_masks(handoff_frame_rgb, index_mask, num_slots)
    logger.info("Cutie seeded at frame %d.", handoff_idx)

    # ==================================================================
    # PHASE 2 — Cutie (frames [sam3_phase_frames, num_frames))
    # Continuous tracking + safety nets.
    # ==================================================================
    logger.info("=== PHASE 2: Cutie tracking frames [%d, %d) ===",
                sam3_phase_frames, num_frames)

    sam3_calls = 0
    recovery = RecoveryController(max_search_frames=max_search_frames)

    for gidx in range(sam3_phase_frames, num_frames):
        frame_rgb = all_frames[gidx]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        time_sec = gidx / fps

        # Cutie propagates
        id_mask = cutie.track(frame_rgb)
        slot_masks = cutie.id_mask_to_slot_masks(id_mask, num_slots)
        slot_centroids = [mask_centroid(m) if m is not None else None for m in slot_masks]

        # ---- Safety ----
        sam3_ran = False
        recovered = False

        protection_event = check_protection(slot_masks, num_slots) if protection_enabled \
            else SafetyEvent(triggered=False, system="none")
        fusion_event = check_fusion(
            state["prev_masks"], slot_masks, num_slots,
            area_match_tolerance=fusion_area_tol, min_growth_ratio=fusion_growth) \
            if fusion_enabled else SafetyEvent(triggered=False, system="none")

        checkpoint_event = SafetyEvent(triggered=False, system="none")
        do_checkpoint = (gidx % checkpoint_interval == 0)
        need_sam3 = protection_event.triggered or fusion_event.triggered

        if do_checkpoint and not need_sam3:
            sam3_masks_ck, sam3_scores_ck = sam3_image_segment(
                sam3_model, sam3_processor, frame_rgb,
                text_prompt=text_prompt, device=device, score_threshold=score_threshold)
            sam3_masks_ck, sam3_scores_ck = select_first_n_masks(sam3_masks_ck, sam3_scores_ck, num_slots)
            sam3_calls += 1
            sam3_ran = True
            checkpoint_event = checkpoint_decision(
                slot_masks, sam3_masks_ck, num_slots, frame_shape, iou_threshold=checkpoint_iou)
            if checkpoint_event.triggered:
                new_masks, new_cents = reassign_by_previous_frame(
                    sam3_masks_ck, state["prev_centroids"], num_slots,
                    mask_centroid, assign_identities_by_centroid, max_dist=max_assignment_dist)
                if sum(1 for m in new_masks if m is not None) >= num_slots:
                    slot_masks, slot_centroids = new_masks, new_cents
                    im, no = slot_masks_to_index_mask(slot_masks, frame_shape)
                    cutie.reinject_masks(frame_rgb, im, num_slots)
                    recovered = True

        if (protection_event.triggered or fusion_event.triggered) and not recovered:
            event = protection_event if protection_event.triggered else fusion_event
            sam3_masks_rec, sam3_scores_rec = sam3_image_segment(
                sam3_model, sam3_processor, frame_rgb,
                text_prompt=text_prompt, device=device, score_threshold=score_threshold)
            sam3_masks_rec, sam3_scores_rec = select_first_n_masks(sam3_masks_rec, sam3_scores_rec, num_slots)
            sam3_calls += 1
            sam3_ran = True
            new_masks, new_cents = reassign_by_previous_frame(
                sam3_masks_rec, state["prev_centroids"], num_slots,
                mask_centroid, assign_identities_by_centroid, max_dist=max_assignment_dist)
            n_found = sum(1 for m in new_masks if m is not None)
            if n_found >= num_slots:
                slot_masks, slot_centroids = new_masks, new_cents
                im, no = slot_masks_to_index_mask(slot_masks, frame_shape)
                cutie.reinject_masks(frame_rgb, im, num_slots)
                recovered = True
                if recovery.searching:
                    recovery.stop(True, gidx)
            else:
                if not recovery.searching:
                    recovery.start(gidx)
                for s in range(num_slots):
                    if new_masks[s] is not None:
                        slot_masks[s] = new_masks[s]
                        slot_centroids[s] = new_cents[s]
            safety_logger.log(
                frame_idx=gidx, time_sec=round(time_sec, 3),
                system=event.system, reason=event.reason,
                missing_slots=event.missing_slots, detail=event.detail,
                sam3_ran=sam3_ran, recovered=recovered)
        elif checkpoint_event.triggered:
            safety_logger.log(
                frame_idx=gidx, time_sec=round(time_sec, 3),
                system=checkpoint_event.system, reason=checkpoint_event.reason,
                missing_slots=[], detail=checkpoint_event.detail,
                sam3_ran=sam3_ran, recovered=recovered)

        if recovery.searching:
            recovery.tick()
            if not recovery.should_continue():
                recovery.stop(False, gidx)

        # ---- Shared per-frame outputs ----
        process_frame_outputs(frame_bgr, slot_masks, slot_centroids)
        state["global_frame_idx"] += 1

        if state["global_frame_idx"] % 500 == 0:
            logger.info("Processed %d/%d frames (Phase 2 SAM3 calls: %d)",
                        state["global_frame_idx"], num_frames, sam3_calls)

    # ==================================================================
    # Finalize
    # ==================================================================
    main_writer.release()
    for w in unitary_writers:
        if w is not None:
            w.release()
    logger.info("Output videos written: %s", overlays_dir)

    safety_logger.close()
    cutie.cleanup()

    if contact_tracker is not None:
        summary = contact_tracker.finalize()
        total_bouts = sum(v.get("total_bouts", 0)
                          for v in summary.get("contact_type_summary", {}).values())
        logger.info("ContactTracker finalized. Total bouts: %d", total_bouts)
        try:
            from scripts.postprocess_contacts_simple import run_postprocess
            run_postprocess(run_dir / "contacts", fps=fps)
        except Exception as e:
            logger.warning("Contact post-processing failed: %s", e)

    gidx_total = state["global_frame_idx"]
    for i in range(num_slots):
        logger.info("Rat %d: pose detected in %d/%d frames (%.1f%%)",
                    i + 1, state["yolo_hit_counts"][i], gidx_total,
                    100 * state["yolo_hit_counts"][i] / max(gidx_total, 1))
    logger.info("Frames with keypoint carry-over: %d/%d (%.1f%%)",
                state["carried_kpt_frames"], gidx_total,
                100 * state["carried_kpt_frames"] / max(gidx_total, 1))
    logger.info("Frames with mask carry-over: %d/%d (%.1f%%)",
                state["carried_mask_frames"], gidx_total,
                100 * state["carried_mask_frames"] / max(gidx_total, 1))
    logger.info("Phase 2 SAM3 calls: %d", sam3_calls)
    logger.info("Pipeline complete. Output directory: %s", run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Cutie composite pipeline (two-phase)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    args = parser.parse_args()
    run_pipeline(config_path=args.config, cli_overrides=args.overrides,
                 start_frame=args.start_frame, end_frame=args.end_frame)


if __name__ == "__main__":
    main()