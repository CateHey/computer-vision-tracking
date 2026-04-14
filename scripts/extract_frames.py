#!/usr/bin/env python3
"""
Close-contact detection and frame extraction workflow.

Runs a multi-step pipeline to find frames where animals are interacting
and exports them for annotation in Roboflow.

Steps:
  1. scan       - Run YOLO over the video, classify each frame as close/not close
  2. encounters - Group consecutive close frames into encounter events
  3. plan       - Build a sampling plan (which frames to export)
  4. export     - Save selected frames to disk
  5. all        - Run all steps 1-4 sequentially

Usage:
    python scripts/extract_frames.py scan --config configs/local_quick.yaml
    python scripts/extract_frames.py all --config configs/local_quick.yaml
    python scripts/extract_frames.py scan --config configs/local_quick.yaml detection.confidence=0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

# Add project root to path so src imports work when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.io_video import open_video_reader, get_video_properties, iter_frames
from src.common.metrics import evaluate_closeness
from src.common.utils import Detection, FrameAnalysis, Encounter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STEP 1: Scan
# ---------------------------------------------------------------------------

def step_scan(config: Dict[str, Any], run_dir: Path) -> None:
    """Run YOLO detection over the video and classify frames as close/not close.

    Outputs:
        <run_dir>/scan/close_frames.jsonl   (one JSON line per frame)
        <run_dir>/scan/scan_metadata.json   (fps, counts, thresholds)
    """
    from ultralytics import YOLO

    video_path = config["video_path"]
    models_cfg = config["models"]
    det_cfg = config.get("detection", {})
    close_cfg = config.get("closeness", {})
    scan_cfg = config.get("scan", {})

    conf_thr = det_cfg.get("confidence", 0.25)
    dist_thr = close_cfg.get("distance_threshold_norm", 0.15)
    iou_thr = close_cfg.get("iou_threshold", 0.1)
    max_frames = scan_cfg.get("max_frames")

    logger.info("=== Step 1: Scan ===")
    logger.info("Video: %s", video_path)
    logger.info("Thresholds: conf=%.2f, dist_norm=%.2f, iou=%.2f", conf_thr, dist_thr, iou_thr)

    # Load YOLO
    model_path = models_cfg["yolo_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    model = YOLO(str(model_path))
    logger.info("YOLO model loaded from %s", model_path)

    # Open video
    cap = open_video_reader(video_path)
    props = get_video_properties(cap)
    fps = props["fps"]
    width = props["width"]
    height = props["height"]
    logger.info("Video: %dx%d @ %.1f FPS, %d frames", width, height, fps, props["total_frames"])

    # Output directory
    scan_dir = run_dir / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    frames_file = scan_dir / "close_frames.jsonl"
    meta_file = scan_dir / "scan_metadata.json"

    num_processed = 0
    num_close = 0

    with frames_file.open("w", encoding="utf-8") as f_out:
        for frame_idx, frame_bgr in iter_frames(cap, max_frames=max_frames):
            result = model(frame_bgr, verbose=False)[0]
            detections: List[Detection] = []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for bbox, conf in zip(xyxy, confs):
                    if conf < conf_thr:
                        continue
                    x1, y1, x2, y2 = bbox.tolist()
                    detections.append(Detection(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        conf=float(conf),
                    ))

            time_sec = frame_idx / fps
            is_close, min_dist, max_iou = evaluate_closeness(
                detections, width, height, dist_thr, iou_thr,
            )
            if is_close:
                num_close += 1

            analysis = FrameAnalysis(
                frame_idx=frame_idx,
                time_sec=round(time_sec, 4),
                num_detections=len(detections),
                detections=detections,
                is_close=is_close,
                min_distance_norm=round(min_dist, 6),
                max_iou=round(max_iou, 6),
            )
            f_out.write(json.dumps(analysis.to_dict()) + "\n")

            num_processed += 1
            if num_processed % 100 == 0:
                logger.info("Processed %d frames (%d close)", num_processed, num_close)

    cap.release()

    metadata = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": props["total_frames"],
        "num_processed_frames": num_processed,
        "num_close_frames": num_close,
        "thresholds": {
            "confidence": conf_thr,
            "distance_threshold_norm": dist_thr,
            "iou_threshold": iou_thr,
        },
    }
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Scan complete: %d/%d close frames", num_close, num_processed)
    logger.info("Frames file: %s", frames_file)
    logger.info("Metadata: %s", meta_file)


# ---------------------------------------------------------------------------
# STEP 2: Encounters
# ---------------------------------------------------------------------------

def step_encounters(config: Dict[str, Any], run_dir: Path) -> None:
    """Group consecutive close frames into encounters.

    Reads: <run_dir>/scan/close_frames.jsonl, scan_metadata.json
    Writes: <run_dir>/scan/encounters_summary.json
    """
    enc_cfg = config.get("encounters", {})
    max_gap_sec = enc_cfg.get("max_gap_seconds", 2.0)
    min_dur_sec = enc_cfg.get("min_duration_seconds", 0.5)

    logger.info("=== Step 2: Encounters ===")
    logger.info("Max gap: %.1fs, Min duration: %.1fs", max_gap_sec, min_dur_sec)

    scan_dir = run_dir / "scan"
    frames_file = scan_dir / "close_frames.jsonl"
    meta_file = scan_dir / "scan_metadata.json"

    if not frames_file.exists():
        raise FileNotFoundError(f"Run scan first. Missing: {frames_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Run scan first. Missing: {meta_file}")

    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    fps = float(meta["fps"])

    max_gap_frames = int(max_gap_sec * fps)
    min_dur_frames = int(min_dur_sec * fps)

    encounters: List[Encounter] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_count = 0
    last_close_frame: Optional[int] = None
    total_close = 0

    with frames_file.open("r", encoding="utf-8") as f:
        for line in f:
            fr = json.loads(line)
            if not fr.get("is_close", False):
                continue

            frame_idx = int(fr["frame_idx"])
            total_close += 1

            if current_start is None:
                current_start = frame_idx
                current_end = frame_idx
                current_count = 1
            else:
                gap = frame_idx - last_close_frame
                if gap <= max_gap_frames:
                    current_end = frame_idx
                    current_count += 1
                else:
                    duration = current_end - current_start + 1
                    if duration >= min_dur_frames:
                        encounters.append(Encounter(
                            encounter_id=len(encounters),
                            start_frame=current_start,
                            end_frame=current_end,
                            start_time_sec=current_start / fps,
                            end_time_sec=current_end / fps,
                            num_close_frames=current_count,
                        ))
                    current_start = frame_idx
                    current_end = frame_idx
                    current_count = 1

            last_close_frame = frame_idx

    # Flush last encounter
    if current_start is not None:
        duration = current_end - current_start + 1
        if duration >= min_dur_frames:
            encounters.append(Encounter(
                encounter_id=len(encounters),
                start_frame=current_start,
                end_frame=current_end,
                start_time_sec=current_start / fps,
                end_time_sec=current_end / fps,
                num_close_frames=current_count,
            ))

    summary = {
        "video_path": meta["video_path"],
        "fps": fps,
        "total_frames": meta["total_frames"],
        "total_close_frames": total_close,
        "num_encounters": len(encounters),
        "parameters": {
            "max_gap_seconds": max_gap_sec,
            "min_duration_seconds": min_dur_sec,
        },
        "encounters": [e.to_dict() for e in encounters],
    }

    out_file = scan_dir / "encounters_summary.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Found %d encounters from %d close frames", len(encounters), total_close)
    for e in encounters:
        logger.info(
            "  Encounter %d: %.1fs - %.1fs (%d close frames)",
            e.encounter_id, e.start_time_sec, e.end_time_sec, e.num_close_frames,
        )
    logger.info("Summary: %s", out_file)


# ---------------------------------------------------------------------------
# STEP 3: Sampling plan
# ---------------------------------------------------------------------------

def step_plan(config: Dict[str, Any], run_dir: Path) -> None:
    """Build a sampling plan deciding which frames to export per encounter.

    Reads: scan/close_frames.jsonl, scan/encounters_summary.json, scan/scan_metadata.json
    Writes: scan/sampling_plan.json
    """
    samp_cfg = config.get("sampling", {})
    target_total = samp_cfg.get("target_total_frames", 200)
    min_per_enc = samp_cfg.get("min_per_encounter", 10)

    logger.info("=== Step 3: Sampling Plan ===")
    logger.info("Target: %d frames, min %d per encounter", target_total, min_per_enc)

    scan_dir = run_dir / "scan"
    frames_file = scan_dir / "close_frames.jsonl"
    encounters_file = scan_dir / "encounters_summary.json"
    meta_file = scan_dir / "scan_metadata.json"

    for f in [frames_file, encounters_file, meta_file]:
        if not f.exists():
            raise FileNotFoundError(f"Run previous steps first. Missing: {f}")

    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    fps = float(meta["fps"])

    with encounters_file.open("r", encoding="utf-8") as f:
        enc_summary = json.load(f)

    enc_data = enc_summary["encounters"]
    num_enc = len(enc_data)
    if num_enc == 0:
        raise RuntimeError("No encounters found. Check thresholds or run encounters step.")

    # Map encounter -> frame ranges
    enc_ranges = [(int(e["encounter_id"]), int(e["start_frame"]), int(e["end_frame"])) for e in enc_data]
    frames_by_enc: Dict[int, List[int]] = {eid: [] for eid, _, _ in enc_ranges}

    with frames_file.open("r", encoding="utf-8") as f:
        for line in f:
            fr = json.loads(line)
            if not fr.get("is_close", False):
                continue
            fidx = int(fr["frame_idx"])
            for eid, sf, ef in enc_ranges:
                if sf <= fidx <= ef:
                    frames_by_enc[eid].append(fidx)
                    break

    # Compute desired frames per encounter
    base_per_enc = max(min_per_enc, target_total // num_enc)

    selected: List[Dict[str, Any]] = []
    per_enc_info: Dict[str, Any] = {}

    for eid, sf, ef in enc_ranges:
        available = frames_by_enc[eid]
        n_avail = len(available)

        if n_avail == 0:
            per_enc_info[str(eid)] = {"available": 0, "selected": 0}
            continue

        desired = min(base_per_enc, n_avail)
        if desired <= 0:
            per_enc_info[str(eid)] = {"available": n_avail, "selected": 0}
            continue

        # Spread selection evenly across available frames
        indices = []
        for i in range(desired):
            pos = int(round(i * (n_avail - 1) / max(1, desired - 1)))
            indices.append(available[pos])
        indices = sorted(set(indices))

        per_enc_info[str(eid)] = {"available": n_avail, "selected": len(indices)}

        for fidx in indices:
            selected.append({
                "encounter_id": eid,
                "frame_idx": fidx,
                "time_sec": round(fidx / fps, 4),
            })

    selected.sort(key=lambda d: d["frame_idx"])

    plan = {
        "target_total_frames": target_total,
        "base_per_encounter": base_per_enc,
        "actual_total_frames": len(selected),
        "per_encounter": per_enc_info,
        "selected_frames": selected,
    }

    out_file = scan_dir / "sampling_plan.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    logger.info("Plan: %d frames selected across %d encounters", len(selected), num_enc)
    for eid_str, info in per_enc_info.items():
        logger.info("  Encounter %s: %d/%d selected", eid_str, info["selected"], info["available"])
    logger.info("Plan saved: %s", out_file)


# ---------------------------------------------------------------------------
# STEP 4: Export frames
# ---------------------------------------------------------------------------

def step_export(config: Dict[str, Any], run_dir: Path) -> None:
    """Export frames from the video according to the sampling plan.

    Reads: scan/sampling_plan.json + original video
    Writes: frames/encounter_NNN/<prefix>_eNNN_fNNNNNN.jpg
    """
    export_cfg = config.get("export", {})
    prefix = export_cfg.get("filename_prefix", "rat_close")
    custom_output = export_cfg.get("output_dir")

    logger.info("=== Step 4: Export Frames ===")

    scan_dir = run_dir / "scan"
    plan_file = scan_dir / "sampling_plan.json"
    if not plan_file.exists():
        raise FileNotFoundError(f"Run plan step first. Missing: {plan_file}")

    with plan_file.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    selected = plan.get("selected_frames", [])
    if not selected:
        logger.warning("No frames in plan. Nothing to export.")
        return

    # Determine output directory
    if custom_output:
        output_root = Path(custom_output)
    else:
        output_root = run_dir / "frames"
    output_root.mkdir(parents=True, exist_ok=True)

    # Group by encounter
    frames_by_enc: Dict[int, List[int]] = {}
    for item in selected:
        eid = int(item["encounter_id"])
        frames_by_enc.setdefault(eid, []).append(int(item["frame_idx"]))
    for eid in frames_by_enc:
        frames_by_enc[eid] = sorted(set(frames_by_enc[eid]))

    all_indices = sorted({f for flist in frames_by_enc.values() for f in flist})
    target_set = set(all_indices)

    # Open video and extract frames
    video_path = config["video_path"]
    cap = open_video_reader(video_path)
    logger.info("Exporting %d frames from %s", len(all_indices), video_path)

    total_saved = 0
    for frame_idx, frame_bgr in iter_frames(cap):
        if frame_idx in target_set:
            for eid, flist in frames_by_enc.items():
                if frame_idx in flist:
                    enc_dir = output_root / f"encounter_{eid:03d}"
                    enc_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{prefix}_e{eid:03d}_f{frame_idx:06d}.jpg"
                    cv2.imwrite(str(enc_dir / fname), frame_bgr)
                    total_saved += 1

        # Stop early if past all needed frames
        if frame_idx > max(all_indices):
            break

    cap.release()
    logger.info("Exported %d frames to %s", total_saved, output_root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Close-contact detection and frame extraction.",
    )
    parser.add_argument(
        "command",
        choices=["scan", "encounters", "plan", "export", "all"],
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existing run directory (for steps 2-4). If omitted, creates a new one.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides as key=value pairs.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_config(args.config, args.overrides or None)

    # Use existing run dir or create new one
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    else:
        run_dir = setup_run_dir(config, tag="extract_frames")

    setup_logging(run_dir)

    logger.info("Run directory: %s", run_dir)

    if args.command == "scan":
        step_scan(config, run_dir)
    elif args.command == "encounters":
        step_encounters(config, run_dir)
    elif args.command == "plan":
        step_plan(config, run_dir)
    elif args.command == "export":
        step_export(config, run_dir)
    elif args.command == "all":
        step_scan(config, run_dir)
        step_encounters(config, run_dir)
        step_plan(config, run_dir)
        step_export(config, run_dir)

    logger.info("Done. Run directory: %s", run_dir)


if __name__ == "__main__":
    main()
