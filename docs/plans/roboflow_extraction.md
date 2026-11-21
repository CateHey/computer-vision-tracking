# Plan: Roboflow Interaction Frame Extraction for YOLO Retraining

## Context

The centroid pipeline proves SAM2 masks are stable even during close interactions, but YOLO misses keypoints when rats are close (showing "1 carried" in status bar). Retraining YOLO with close-interaction frames should improve keypoint detection during interactions.

**Goal:** Extract ~200 well-distributed frames where rats are interacting closely, save as JPEGs for Roboflow labeling, then retrain YOLO.

## Architecture: Two-Phase Approach

**Phase 1 — Scan (GPU, parallelizable):**
- SAM2 centroid propagation through video (YOLO only on init frame)
- For each frame: compute centroid distance + mask IoU
- Output: CSV with per-frame interaction scores

**Phase 2 — Select & Extract (CPU, fast):**
- Merge all chunk CSVs
- Select ~200 best interaction frames with minimum spacing
- Seek to those frames in video → save as JPEGs

## New Files

| File | Description |
|------|-------------|
| `roboflow/extract_interaction_frames.py` | Main script (scan + select subcommands) |
| `roboflow/run_extract.sh` | Bunya batch wrapper (parallel scan → merge → extract → scp) |
| `configs/hpc_extract.yaml` | HPC config (SAM2 large, cuda) |
| `configs/local_extract.yaml` | Local config (SAM2 tiny, auto device, max_frames=20) |

## Usage

```bash
# Local (single GPU, 10s clip)
python roboflow/extract_interaction_frames.py all --config configs/local_extract.yaml

# Bunya HPC (multi-GPU parallel, 5 min video)
bash roboflow/run_extract.sh data/raw/original.mp4 4

# Download frames to local machine (PowerShell)
scp -r s4948012@bunya.rcc.uq.edu.au:/path/roboflow/frames `
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\roboflow"
```

## Selection Algorithm

```
1. Filter: num_masks == 2 AND centroid_dist_px <= threshold (200px default)
2. Score: score = (1 - dist/threshold) * 0.6 + mask_iou * 0.4
3. Sort by score descending
4. Greedy select with min_gap (15 frames = ~0.5s):
   - Accept frame only if no selected frame within min_gap
   - Stop at max_frames (200)
5. Re-sort by frame_idx for sequential extraction
```

## Output

```
roboflow/frames/
├── frame_000123.jpg
├── frame_000456.jpg
├── ...  (~200 JPEGs)
└── metadata.csv   ← frame_idx, centroid_dist, mask_iou, score
```

## YOLO Retraining Workflow

1. Run extraction on Bunya → get ~200 interaction frames
2. SCP frames to local machine
3. Upload to Roboflow → label keypoints (7 per rat)
4. Mix ~30-40% interaction frames with ~60-70% normal/separated frames
5. Export dataset from Roboflow in YOLOv8 format
6. Retrain YOLO26 on augmented dataset
7. Replace `models/yolo/best.pt` with new weights
8. Re-run centroid pipeline → fewer "carried" frames during interactions
