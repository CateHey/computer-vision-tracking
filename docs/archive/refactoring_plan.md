# Refactoring Plan (Historical Record)

> **Note:** This is an archived historical document recording the initial codebase
> restructuring. For current architecture, see [Architecture Overview](../architecture/overview.md).

This document records the refactoring performed to restructure the repository from its initial state into the target architecture.

## Motivation

The codebase had working code in two disconnected locations:
- `src/pipelines/sam2_yolo_deprecated/` — YOLO+SAM2 video pipeline
- `src/roboflow_close_contact_detect_to_export_frames/` — close-contact frame extraction + upload

Problems addressed:
- Hardcoded thresholds scattered across files
- No shared utility layer (duplicate code, e.g., `compute_iou` in two places)
- No config system (couldn't switch between local/HPC without editing code)
- No standardized output structure
- No documentation beyond basic READMEs

## What Was Done

### Phase A: Foundation
1. Created directory skeleton matching target architecture
2. Wrote YAML config files (`configs/local_quick.yaml`, `configs/hpc_full.yaml`)
3. Built config loader (`src/common/config_loader.py`) with YAML loading, path resolution, CLI overrides, run directory setup, and logging

### Phase B: Shared Utilities
Extracted reusable code from deprecated modules into `src/common/`:
4. `metrics.py` — `bbox_iou`, `mask_iou`, `compute_centroid`, `evaluate_closeness`
5. `io_video.py` — video reader/writer, frame iterator, property extraction
6. `visualization.py` — mask overlay, centroid drawing, text annotation
7. `tracking.py` — mask filtering (NMS), centroid-based identity matching
8. `utils.py` — `Detection`, `FrameAnalysis`, `Encounter` data classes

### Phase C: SAM2+YOLO Pipeline
Rebuilt the pipeline as `src/pipelines/sam2_yolo/`:
9. `models_io.py` — config-driven YOLO + SAM2 loading
10. `infer_yolo.py` — YOLO detection returning `Detection` objects
11. `infer_sam2.py` — SAM2 segmentation from box prompts
12. `postprocess.py` — mask filtering + identity tracking
13. `run.py` — full pipeline orchestration with CLI

### Phase D: Scripts
14. `scripts/extract_frames.py` — 5-step close-contact workflow (scan/encounters/plan/export/all)
15. `scripts/upload_to_roboflow.py` — Roboflow upload with config-driven settings
16. `scripts/extract_clip.py` — ffmpeg/OpenCV clip extraction
17. `scripts/export_results.py` — run artifact export

### Phase E: Infrastructure
18. Slurm templates (`slurm/run_infer.sbatch`, `run_train_yolo.sbatch`, `run_extract_frames.sbatch`)
19. `pyproject.toml` (modern packaging) + updated `requirements.txt`
20. Updated `.gitignore` for new structure

### Phase F: Documentation & Cleanup
21. Documentation in `docs/`: overview, data notes, local/HPC guides, labeling guide, evaluation plan
22. Updated root `README.md`
23. Deleted deprecated code

## Key Design Decisions

### Config-Driven Everything
All tunable parameters live in YAML configs. CLI overrides allow one-off changes without editing files. Every run saves a `config_used.yaml` snapshot for reproducibility.

### Shared vs Pipeline-Specific Code
- `src/common/` contains model-agnostic utilities (video I/O, geometry, visualization)
- `src/pipelines/<name>/` contains model-specific logic (YOLO inference, SAM2 prompting)
- This separation allows adding new pipelines (e.g., SAM3) without duplicating utilities

### Output Structure
Standardized `outputs/runs/<run_id>/` with config snapshot, logs, and pipeline-specific artifacts. Makes comparison across runs straightforward.

### Logging Over Print
Replaced all `print()` calls with Python `logging`. Logs go to both console and file (`<run_dir>/logs/run.log`).

## Parameter Reference

All tunable parameters and their defaults are documented inline in `configs/local_quick.yaml`.
For detailed tuning guidance, see [parameter_tuning.md](parameter_tuning.md).

Key parameters:

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| YOLO confidence | `detection.confidence` | 0.25 | Min detection confidence |
| Max animals | `detection.max_animals` | 2 | Max tracked instances |
| Edge margin | `detection.edge_margin` | 0 | Reject border detections (px) |
| Custom NMS | `detection.nms_iou` | null | Custom NMS IoU threshold |
| Border padding | `detection.yolo_border_padding_px` | 0 | Mirror-pad frame borders (px) |
| SAM threshold | `segmentation.sam_threshold` | 0.0 | Mask logit threshold |
| Mask IoU | `segmentation.mask_iou_threshold` | 0.5 | Dedup threshold |
| Tracker | `tracking.tracker_config` | botsort.yaml | BoT-SORT or ByteTrack |
| Tracking distance | `tracking.max_centroid_distance` | 150.0 | Distance normalization constant |
| Missing frames | `tracking.max_missing_frames` | 5 | Frames before slot release |
| Cost weights | `tracking.w_dist/w_iou/w_area` | 0.4/0.4/0.2 | Hungarian cost weights |
| Cost threshold | `tracking.cost_threshold` | 0.85 | Max cost for valid assignment |
| Close distance | `closeness.distance_threshold_norm` | 0.15 | Normalized by diagonal |
| Close IoU | `closeness.iou_threshold` | 0.1 | Bbox overlap for "close" |
| Encounter gap | `encounters.max_gap_seconds` | 2.0 | Max gap to merge |
| Encounter min | `encounters.min_duration_seconds` | 0.5 | Min encounter length |
| Sample target | `sampling.target_total_frames` | 50/200 | Frames to export |
