# Project Overview

## Purpose

This project develops an AI-based workflow to analyze **laboratory rat movement** from real-world lab videos. The objective is to detect and track rat movement accurately, with emphasis on **body-part localization** (7 keypoints) to enable robust trajectory extraction and social behavior analysis (nose-touching, side-by-side, following).

## Key Deliverable

A practical, repeatable pipeline that turns raw video into structured outputs:
detections, masks, tracks, social contact classifications, and summary reports.

## Success Criteria

### Primary (movement + parts)
- For each frame, accurately identify rat instances (bounding box + silhouette mask)
- Localize 7 body parts: **tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear**
- Maintain consistent identity across time (tracking), especially during close interactions

### Secondary (research readiness)
- Reproducible runs on laptop (short clips) and UQ Bunya HPC (full videos)
- Clear outputs: overlay videos, contact CSVs, bout reports, session summaries
- Social contact classification: nose-to-nose, nose-to-body, side-by-side, following

## Models

### YOLO26 (Ultralytics)
- **Role:** Fast pose detection — bounding boxes + 7 keypoints per rat
- **Architecture:** NMS-free (end-to-end), dual detection head, C2PSA attention
- **Weights:** `models/yolo/best.pt` (trained on custom rat dataset)
- **Previous:** YOLOv8l-pose (archived as `yolov8lrata.pt`)
- See [YOLO26 Migration](../models/yolo26_migration.md) for upgrade details

### SAM2 (Meta)
- **Role:** High-quality promptable segmentation
- **Use case:** Turn YOLO detections (boxes/points) into pixel-accurate masks
- **Variants:** tiny (local testing), large (HPC production)

## Pipeline Overview

```
1. Video → frames
2. YOLO26 detect-only → rat bounding boxes + 7 keypoints
3. SAM2 segmentation (box prompts + centroid fallback) → masks
4. IdentityMatcher → consistent slot assignment (SEPARATE ↔ MERGED state machine)
5. ContactTracker → social contact classification + bout detection
6. Outputs → overlay video + contact CSVs + session summary
```

### Three Pipelines

| Pipeline | Module | Best for |
|----------|--------|----------|
| **reference** (recommended) | `src/pipelines/reference/` | Most videos — best identity stability |
| sam2_yolo | `src/pipelines/sam2_yolo/` | Simple scenes without close interactions |
| sam2_video | `src/pipelines/sam2_video/` | Short videos with heavy occlusion |

See [Pipeline Comparison](pipelines.md) for detailed differences.

## Mental Model

- **YOLO26** finds where and what (rats + body parts) — fast, NMS-free
- **SAM2** refines detections into pixel-accurate masks
- **IdentityMatcher** keeps IDs stable through crossings and close interactions
- **ContactTracker** classifies social behaviors from keypoints + masks
- **Outputs** provide quantitative data for behavioral analysis
