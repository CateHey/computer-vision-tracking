"""
SAMURAI + SLEAP pipeline — SAMURAI video propagation for masks + SLEAP keypoints.

Architecture:
  - SAMURAI (SAM2 video predictor) provides masks and identity via propagate_in_video
  - SLEAP bottom-up model provides 7 keypoints per rat
  - Keypoints assigned to SAMURAI masks by spatial overlap (containment + nearest centroid)
  - No contact classification yet (future addition)

Strategy:
  - Phase 1: Extract video frames to disk (SAMURAI needs JPEG directory)
  - Phase 2: Initialize SAMURAI with bounding boxes (interactive or config-provided)
  - Phase 3+4: Stream processing — propagate masks AND render each frame immediately
             (avoids storing all masks in memory → prevents OOM)
  - SLEAP keypoints applied per-frame during streaming if available

Usage:
    python -m src.pipelines.samurai_sleap.run --config configs/local_samurai_sleap.yaml
    python -m src.pipelines.samurai_sleap.run --config configs/local_samurai_sleap.yaml --interactive
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

from src.common.config_loader import load_config, setup_run_dir, setup_logging, get_device
from src.common.io_video import create_video_writer
from src.common.metrics import compute_centroid
from src.common.geometry import euclidean_distance, resolve_overlaps
from src.common.visualization import (
    apply_masks_overlay, draw_centroids, draw_keypoints, draw_text,
)
from src.common.utils import Detection, Keypoint

logger = logging.getLogger(__name__)

KEYPOINT_NAMES = [
    "tail_tip", "tail_base", "tail_start",
    "mid_body", "nose", "right_ear", "left_ear",
]

# Default chunk size for video processing (limits GPU memory usage)
DEFAULT_CHUNK_SIZE = 300


# ---------------------------------------------------------------------------
# Phase 1: Frame extraction (SAMURAI video predictor needs JPEG directory)
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str | Path,
    frames_dir: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int | None = None,
) -> Tuple[int, float, int, int]:
    """Extract video frames as JPEGs to a directory.

    Returns:
        (num_frames_extracted, fps, width, height)
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = 0
    abs_idx = start_frame
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        if end_frame is not None and abs_idx >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), frame)
        idx += 1
        abs_idx += 1

    cap.release()
    logger.info("Extracted %d frames to %s", idx, frames_dir)
    return idx, fps, width, height


# ---------------------------------------------------------------------------
# Phase 2: Initial bounding boxes
# ---------------------------------------------------------------------------

def get_init_bboxes(config: Dict[str, Any], video_path: str, interactive: bool = False) -> List[Dict]:
    """Get initial bounding boxes for SAMURAI initialization."""
    cfg_bboxes = config.get("init_bboxes")
    if cfg_bboxes and not interactive:
        bboxes = []
        for bb in cfg_bboxes:
            bboxes.append({
                "x": int(bb[0]), "y": int(bb[1]),
                "w": int(bb[2]), "h": int(bb[3]),
            })
        logger.info("Using %d bounding boxes from config", len(bboxes))
        return bboxes

    return _select_bboxes_interactive(video_path)


def _select_bboxes_interactive(video_path: str) -> List[Dict]:
    """Show first frame and let user draw bounding boxes with the mouse."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")

    logger.info("Interactive bbox selection — draw a rectangle around each rat, press ENTER. Press 'q'/ESC when done.")

    bboxes = []
    clone = frame.copy()
    colors = [(0, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255)]
    window_name = "Select rats - Draw box + ENTER | 'q' to finish"

    while True:
        display = clone.copy()
        for i, bb in enumerate(bboxes):
            c = colors[i % len(colors)]
            cv2.rectangle(display, (bb["x"], bb["y"]),
                          (bb["x"] + bb["w"], bb["y"] + bb["h"]), c, 2)
            cv2.putText(display, f"Rat {i+1}", (bb["x"], bb["y"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

        roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi

        if w == 0 or h == 0:
            break

        bboxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        logger.info("  Rat %d: x=%d, y=%d, w=%d, h=%d", len(bboxes), x, y, w, h)

    cv2.destroyAllWindows()

    if not bboxes:
        raise RuntimeError("No bounding boxes selected. Cannot initialize SAMURAI.")

    logger.info("%d rat(s) selected.", len(bboxes))
    return bboxes


# ---------------------------------------------------------------------------
# SAMURAI model loading
# ---------------------------------------------------------------------------

def load_samurai_predictor(config: Dict[str, Any], device: str):
    """Load SAM2 video predictor (SAMURAI mode)."""
    from sam2.build_sam import build_sam2_video_predictor

    models_cfg = config["models"]
    checkpoint = models_cfg["samurai_checkpoint"]
    sam2_config = models_cfg["sam2_config"]

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"SAMURAI/SAM2 checkpoint not found: {checkpoint}")

    logger.info("Loading SAMURAI (SAM2 video predictor) from %s", checkpoint)
    predictor = build_sam2_video_predictor(sam2_config, str(checkpoint), device=device)
    return predictor


# ---------------------------------------------------------------------------
# SLEAP keypoint detection
# ---------------------------------------------------------------------------

def load_sleap_model(config: Dict[str, Any], device: str):
    """Load SLEAP bottom-up model from checkpoint (pure PyTorch, no sleap-nn needed)."""
    from src.pipelines.samurai_sleap.sleap_model import load_sleap_checkpoint

    models_cfg = config["models"]
    ckpt_path = models_cfg.get("sleap_checkpoint")

    if not ckpt_path or not Path(ckpt_path).exists():
        logger.warning("SLEAP checkpoint not found at %s — running without keypoints", ckpt_path)
        return None

    try:
        model = load_sleap_checkpoint(ckpt_path, device=device)
        return model
    except Exception as e:
        logger.warning("Failed to load SLEAP model: %s — running without keypoints", e)
        return None


def sleap_predict_peaks(
    sleap_model,
    frame_rgb: np.ndarray,
    keypoint_names: List[str],
    min_confidence: float = 0.3,
    device: str = "cpu",
) -> list:
    """Run SLEAP inference and return raw peaks: list of (kpt_idx, x, y, conf)."""
    if sleap_model is None:
        return []

    try:
        from src.pipelines.samurai_sleap.sleap_model import predict_keypoints

        _, peaks = predict_keypoints(
            sleap_model, frame_rgb, keypoint_names,
            min_confidence=min_confidence, device=device,
        )
        return peaks

    except Exception as e:
        logger.warning("SLEAP inference failed on frame: %s", e, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Peak-to-mask assignment — assign individual SLEAP peaks directly to masks
# ---------------------------------------------------------------------------

def assign_peaks_to_masks(
    peaks: list,
    keypoint_names: List[str],
    masks: List[Optional[np.ndarray]],
    centroids: List[Optional[Tuple[float, float]]],
) -> List[Optional[Detection]]:
    """Assign individual SLEAP peaks to SAMURAI masks by containment/proximity.

    Instead of clustering peaks into instances first (which mixes rats when they're
    close together), each peak is assigned directly to whichever mask contains it.
    This uses the reliable SAMURAI masks as the source of identity.
    """
    n_slots = len(masks)
    # Per-slot keypoints: slot_kpts[i] = list of Keypoint
    slot_kpts: List[List[Keypoint]] = [[] for _ in range(n_slots)]

    for kpt_idx, x, y, conf in peaks:
        ix, iy = int(round(x)), int(round(y))
        kp_name = keypoint_names[kpt_idx] if kpt_idx < len(keypoint_names) else f"kpt_{kpt_idx}"
        kp = Keypoint(x=x, y=y, conf=conf, name=kp_name)

        # Pass 1: containment — is this peak inside a mask?
        assigned = False
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            h, w = mask.shape
            if 0 <= iy < h and 0 <= ix < w and mask[iy, ix]:
                # If this keypoint type already assigned to this slot, keep the higher-conf one
                existing = [k for k in slot_kpts[i] if k.name == kp_name]
                if existing:
                    if conf > existing[0].conf:
                        slot_kpts[i] = [k for k in slot_kpts[i] if k.name != kp_name]
                        slot_kpts[i].append(kp)
                else:
                    slot_kpts[i].append(kp)
                assigned = True
                break

        # Pass 2: nearest centroid fallback (for peaks just outside mask boundary)
        if not assigned:
            best_slot, best_dist = None, float("inf")
            for i in range(n_slots):
                if centroids[i] is None:
                    continue
                d = euclidean_distance((x, y), centroids[i])
                if d < best_dist:
                    best_dist = d
                    best_slot = i
            if best_slot is not None and best_dist < 150.0:
                existing = [k for k in slot_kpts[best_slot] if k.name == kp_name]
                if existing:
                    if conf > existing[0].conf:
                        slot_kpts[best_slot] = [k for k in slot_kpts[best_slot] if k.name != kp_name]
                        slot_kpts[best_slot].append(kp)
                else:
                    slot_kpts[best_slot].append(kp)

    # Build Detection objects from assigned keypoints
    slot_dets: List[Optional[Detection]] = [None] * n_slots
    for i in range(n_slots):
        kpts = slot_kpts[i]
        if not kpts:
            continue

        xs = [kp.x for kp in kpts]
        ys = [kp.y for kp in kpts]
        pad = 20
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = max(xs) + pad
        y2 = max(ys) + pad

        slot_dets[i] = Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            conf=float(np.mean([kp.conf for kp in kpts])),
            class_name="rat",
            keypoints=kpts,
            track_id=i + 1,
        )

    assigned_count = sum(1 for d in slot_dets if d is not None)
    total_kpts = sum(len(slot_kpts[i]) for i in range(n_slots))
    logger.debug("SLEAP: %d peaks → %d assigned to %d/%d masks",
                 len(peaks), total_kpts, assigned_count, n_slots)

    return slot_dets


# ---------------------------------------------------------------------------
# Temporal smoothing — stabilize keypoints across frames
# ---------------------------------------------------------------------------

def _smooth_keypoints(
    slot_dets: List[Optional[Detection]],
    prev_slot_dets: List[Optional[Detection]],
    alpha: float = 0.5,
) -> List[Optional[Detection]]:
    """Exponential moving average on keypoint positions to reduce jitter.

    Each keypoint position is blended: new_pos = alpha * current + (1-alpha) * previous.
    Only smooths keypoints that exist in both current and previous frame for the same slot.
    """
    if prev_slot_dets is None:
        return slot_dets

    for i in range(len(slot_dets)):
        if slot_dets[i] is None or prev_slot_dets[i] is None:
            continue
        if not slot_dets[i].keypoints or not prev_slot_dets[i].keypoints:
            continue

        prev_by_name = {kp.name: kp for kp in prev_slot_dets[i].keypoints}

        for kp in slot_dets[i].keypoints:
            prev_kp = prev_by_name.get(kp.name)
            if prev_kp is not None:
                kp.x = alpha * kp.x + (1 - alpha) * prev_kp.x
                kp.y = alpha * kp.y + (1 - alpha) * prev_kp.y

    return slot_dets


def _carry_over_keypoints(
    slot_dets: List[Optional[Detection]],
    prev_slot_dets: List[Optional[Detection]],
    prev_centroids: List[Optional[Tuple[float, float]]],
    curr_centroids: List[Optional[Tuple[float, float]]],
) -> List[Optional[Detection]]:
    """Fill missing keypoints with previous frame's keypoints shifted by centroid delta."""
    if prev_slot_dets is None:
        return slot_dets

    for i in range(len(slot_dets)):
        if prev_slot_dets[i] is None or not prev_slot_dets[i].keypoints:
            continue
        if prev_centroids[i] is None or curr_centroids[i] is None:
            continue

        dx = curr_centroids[i][0] - prev_centroids[i][0]
        dy = curr_centroids[i][1] - prev_centroids[i][1]

        if slot_dets[i] is None:
            # No detection this frame — carry over entire previous detection
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
                    kp.conf *= 0.9  # decay confidence on carry-over
            slot_dets[i] = carried
        else:
            # Detection exists but might be missing some keypoints — fill gaps
            existing_names = {kp.name for kp in slot_dets[i].keypoints}
            for prev_kp in prev_slot_dets[i].keypoints:
                if prev_kp.name not in existing_names:
                    filled_kp = Keypoint(
                        x=prev_kp.x + dx,
                        y=prev_kp.y + dy,
                        conf=prev_kp.conf * 0.9,
                        name=prev_kp.name,
                    )
                    slot_dets[i].keypoints.append(filled_kp)

    return slot_dets


# ---------------------------------------------------------------------------
# Chunked frame directory helpers
# ---------------------------------------------------------------------------

def _extract_chunk_frames(
    all_frames_dir: Path,
    chunk_dir: Path,
    chunk_start: int,
    chunk_end: int,
) -> int:
    """Symlink (or copy) a range of frames into a chunk-specific directory.

    SAM2 video predictor expects frames named 00000.jpg, 00001.jpg, ...
    so we re-index them starting from 0 inside each chunk directory.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
        src = all_frames_dir / f"{global_idx:05d}.jpg"
        if not src.exists():
            break
        dst = chunk_dir / f"{local_idx:05d}.jpg"
        # Copy instead of symlink for cross-platform compatibility
        shutil.copy2(str(src), str(dst))
        count += 1
    return count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str | Path,
    cli_overrides: List[str] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    chunk_id: int | None = None,
    interactive: bool = False,
) -> Path:
    config = load_config(config_path, cli_overrides)
    tag = f"samurai_sleap_chunk{chunk_id}" if chunk_id is not None else "samurai_sleap"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)

    logger.info("Starting SAMURAI + SLEAP pipeline")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)
    if start_frame > 0 or end_frame is not None:
        logger.info("Frame range: %d -> %s", start_frame, end_frame or "end")

    device = get_device(config)
    logger.info("Using device: %s", device)

    video_path = config["video_path"]
    max_frames = config.get("scan", {}).get("max_frames")
    max_animals = config.get("detection", {}).get("max_animals", 2)
    kpt_names = config.get("detection", {}).get("keypoint_names", KEYPOINT_NAMES)
    kpt_min_conf = config.get("detection", {}).get("keypoint_min_conf", 0.3)
    chunk_size = config.get("scan", {}).get("chunk_size", DEFAULT_CHUNK_SIZE)

    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else None
    codec = config.get("output", {}).get("video_codec", "XVID")

    # ==================================================================
    # Phase 1: Extract ALL frames to disk
    # ==================================================================
    frames_dir = run_dir / "frames_tmp"
    num_frames, fps, width, height = extract_frames(
        video_path, frames_dir,
        start_frame=start_frame, end_frame=end_frame, max_frames=max_frames,
    )
    logger.info("Video: %s (%dx%d @ %.1f FPS, %d frames extracted)",
                video_path, width, height, fps, num_frames)

    if num_frames == 0:
        logger.error("No frames extracted. Check video path: %s", video_path)
        return run_dir

    # ==================================================================
    # Phase 2: Get initial bounding boxes
    # ==================================================================
    bboxes = get_init_bboxes(config, str(video_path), interactive=interactive)
    if len(bboxes) < max_animals:
        logger.warning("Only %d bboxes provided but max_animals=%d", len(bboxes), max_animals)

    # ==================================================================
    # Phase 3: Load SLEAP model (lightweight, keep in memory)
    # ==================================================================
    sleap_model = load_sleap_model(config, device)
    has_sleap = sleap_model is not None

    # ==================================================================
    # Phase 4: Prepare output video
    # ==================================================================
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    ext = ".avi" if codec == "XVID" else ".mp4"
    today = date.today().strftime("%Y-%m-%d")
    out_video_path = overlays_dir / f"samurai_sleap_{today}{ext}"
    writer = create_video_writer(out_video_path, fps, width, height, codec=codec)

    # ==================================================================
    # Phase 5: Chunked streaming — track + render in one pass per chunk
    # ==================================================================
    # Process video in chunks to limit GPU memory.
    # Each chunk: load frames into SAM2 → propagate → render → free memory.
    # Bounding boxes are only registered on chunk 0 (frame 0).
    # For subsequent chunks, we use the last known centroids to re-init.

    n_chunks = (num_frames + chunk_size - 1) // chunk_size
    logger.info("Processing %d frames in %d chunks of %d frames (chunk_size=%d)",
                num_frames, n_chunks, chunk_size, chunk_size)

    frame_count = 0
    sleap_detect_count = 0
    carried_count = 0
    last_centroids = None  # Carry centroids across chunks for re-init
    prev_slot_dets: List[Optional[Detection]] = None
    prev_centroids: List[Optional[Tuple[float, float]]] = None

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_frames)
        chunk_len = chunk_end - chunk_start

        logger.info("Chunk %d/%d: frames %d-%d (%d frames)",
                     chunk_idx + 1, n_chunks, chunk_start, chunk_end - 1, chunk_len)

        # --- Prepare chunk frame directory ---
        chunk_frames_dir = run_dir / f"chunk_tmp_{chunk_idx}"
        n_chunk_frames = _extract_chunk_frames(frames_dir, chunk_frames_dir, chunk_start, chunk_end)

        if n_chunk_frames == 0:
            logger.warning("No frames in chunk %d, skipping", chunk_idx)
            continue

        # --- Load SAMURAI for this chunk ---
        predictor = load_samurai_predictor(config, device)

        use_amp = device == "cuda"
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.inference_mode()

        with torch.inference_mode(), ctx:
            state = predictor.init_state(str(chunk_frames_dir))

            if chunk_idx == 0:
                # First chunk: register bounding boxes from config
                for i, bb in enumerate(bboxes):
                    obj_id = i + 1
                    box = np.array(
                        [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                        dtype=np.float32,
                    )
                    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=box)
                    logger.info("  Registered Rat %d: bbox=[%d, %d, %d, %d]",
                                obj_id, bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"])
            else:
                # Subsequent chunks: re-init from last known centroids as point prompts,
                # OR fall back to original bboxes if centroids are unavailable
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
                            logger.info("  Re-init Rat %d from centroid (%.0f, %.0f)",
                                        obj_id, cx, cy)
                        else:
                            # Fall back to original bbox
                            bb = bboxes[i] if i < len(bboxes) else bboxes[0]
                            box = np.array(
                                [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                                dtype=np.float32,
                            )
                            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=box)
                else:
                    for i, bb in enumerate(bboxes):
                        obj_id = i + 1
                        box = np.array(
                            [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                            dtype=np.float32,
                        )
                        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=box)

            # --- Stream: propagate + render each frame ---
            for local_frame_idx, obj_ids, mask_logits in tqdm(
                predictor.propagate_in_video(state),
                total=n_chunk_frames,
                desc=f"Chunk {chunk_idx+1}/{n_chunks}",
            ):
                global_frame_idx = chunk_start + local_frame_idx

                # Read frame from disk
                frame_path = chunk_frames_dir / f"{local_frame_idx:05d}.jpg"
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # --- Extract masks ---
                slot_masks: List[Optional[np.ndarray]] = [None] * max_animals
                slot_centroids: List[Optional[Tuple[float, float]]] = [None] * max_animals

                for i, obj_id in enumerate(obj_ids):
                    mask = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    slot_idx = obj_id - 1
                    if slot_idx < max_animals:
                        slot_masks[slot_idx] = mask
                        c = compute_centroid(mask)
                        if c is not None:
                            slot_centroids[slot_idx] = c

                resolve_overlaps(slot_masks, slot_centroids)

                # Update centroids after overlap resolution
                for i in range(max_animals):
                    if slot_masks[i] is not None:
                        c = compute_centroid(slot_masks[i])
                        if c is not None:
                            slot_centroids[i] = c

                # Save last centroids for chunk carry-over
                last_centroids = list(slot_centroids)

                # --- SLEAP keypoints ---
                slot_dets: List[Optional[Detection]] = [None] * max_animals
                if has_sleap:
                    peaks = sleap_predict_peaks(sleap_model, frame_rgb, kpt_names, kpt_min_conf, device=device)
                    if peaks:
                        sleap_detect_count += 1
                        slot_dets = assign_peaks_to_masks(peaks, kpt_names, slot_masks, slot_centroids)

                    # Carry over missing keypoints from previous frame
                    slot_dets = _carry_over_keypoints(
                        slot_dets, prev_slot_dets, prev_centroids, slot_centroids,
                    )
                    n_carried = sum(1 for d in slot_dets if d is not None
                                    and any(kp.conf < kpt_min_conf * 0.9 for kp in (d.keypoints or [])))
                    if n_carried > 0:
                        carried_count += 1

                    # Smooth keypoint positions to reduce jitter
                    slot_dets = _smooth_keypoints(slot_dets, prev_slot_dets, alpha=0.6)

                    # Save state for next frame
                    prev_slot_dets = [copy.deepcopy(d) for d in slot_dets]
                    prev_centroids = list(slot_centroids)

                # --- Render overlay ---
                frame_out = np.copy(frame_rgb)

                render_masks = [m for m in slot_masks if m is not None]
                render_colors_list = []
                for i in range(max_animals):
                    if slot_masks[i] is not None and colors:
                        render_colors_list.append(colors[i % len(colors)])
                if not render_colors_list:
                    render_colors_list = None

                if render_masks:
                    frame_out = apply_masks_overlay(frame_out, render_masks, colors=render_colors_list)

                render_centroids = [c for c in slot_centroids if c is not None]
                if render_centroids:
                    c_colors = render_colors_list[:len(render_centroids)] if render_colors_list else None
                    frame_out = draw_centroids(frame_out, render_centroids, colors=c_colors)

                ordered_dets = [d for d in slot_dets if d is not None]
                if ordered_dets:
                    frame_out = draw_keypoints(
                        frame_out, ordered_dets, colors=render_colors_list,
                        min_conf=kpt_min_conf,
                    )

                active_count = sum(1 for m in slot_masks if m is not None)
                kpt_count = sum(1 for d in slot_dets if d is not None)
                status = f"F{global_frame_idx} | Animals: {active_count}/{max_animals} | SAMURAI"
                if has_sleap:
                    status += f" | SLEAP: {kpt_count} det"
                frame_out = draw_text(frame_out, status)

                writer.write(cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))
                frame_count += 1

        # --- Cleanup chunk ---
        del predictor, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Remove chunk frames directory
        shutil.rmtree(str(chunk_frames_dir), ignore_errors=True)

        logger.info("Chunk %d complete. Total frames rendered: %d", chunk_idx + 1, frame_count)

    writer.release()

    # ==================================================================
    # Cleanup: remove extracted frames
    # ==================================================================
    cleanup_frames = config.get("output", {}).get("cleanup_frames", True)
    if cleanup_frames:
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        logger.info("Cleaned up extracted frames")

    # ==================================================================
    # Summary
    # ==================================================================
    logger.info("Pipeline complete. %d frames processed in %d chunks.", frame_count, n_chunks)
    if has_sleap:
        logger.info("Frames with SLEAP detections: %d/%d (%.1f%%)",
                    sleap_detect_count, frame_count,
                    100 * sleap_detect_count / max(frame_count, 1))
        logger.info("Frames with keypoint carry-over: %d/%d (%.1f%%)",
                    carried_count, frame_count,
                    100 * carried_count / max(frame_count, 1))
    logger.info("Overlay video: %s", out_video_path)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAMURAI + SLEAP pipeline: SAMURAI mask propagation + SLEAP keypoints.",
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
