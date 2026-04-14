# Debugging: Isolating YOLO vs SAM2 Problems

When the reference pipeline produces bad output (ID swaps, disappearing masks,
intermittent masks), use these debug scripts to identify the root cause.

## Decision Tree

```
Problem in overlay video?
  ├─ Is YOLO detecting both rats? → Run debug_yolo_only.py
  │    ├─ YOLO misses a rat often → YOLO problem (retrain, lower confidence)
  │    ├─ YOLO boxes overlap a lot → Boxes confuse SAM2 → verify with sam2_only
  │    ├─ YOLO is intermittent/swaps → Try sam2_no_yolo (bypass YOLO)
  │    └─ YOLO detections look clean → SAM2 or IdentityMatcher problem
  │
  ├─ Are SAM2 masks correct? → Run debug_sam2_only.py
  │    ├─ Masks merge into one blob → SAM2 can't separate with current boxes
  │    ├─ Masks flicker/disappear → Centroid fallback unreliable
  │    ├─ Masks look good → IdentityMatcher is the problem (tune params)
  │    └─ Try --no-fallback → Isolates box-only vs centroid fallback quality
  │
  └─ Is YOLO the bottleneck? → Run debug_sam2_no_yolo.py
       ├─ Masks are stable without YOLO → YOLO is the problem, not SAM2
       ├─ Masks still bad → SAM2 can't track from centroids alone
       └─ Try --reinit-every 300 → Periodic YOLO correction helps?
```

## YOLO-Only Debug

Shows only YOLO bounding boxes and keypoints, no segmentation.

```bash
# Local
python scripts/debug_yolo_only.py --config configs/local_reference.yaml

# HPC
python scripts/debug_yolo_only.py --config configs/hpc_reference.yaml \
    video_path=data/raw/original_120s.avi

# Specific frame range
python scripts/debug_yolo_only.py --config configs/local_reference.yaml \
    --start-frame 100 --end-frame 200
```

**What to look for:**
- Status bar shows `YOLO: N det` — should be 2 when both rats are visible
- Frames where detection count drops to 0 or 1
- Boxes that overlap heavily during interactions
- Keypoints landing on the wrong rat

## SAM2-Only Debug

Shows raw SAM2 masks without identity matching. Colors are by detection order
(mask 0 = green, mask 1 = red), NOT by identity.

```bash
# With centroid fallback (default)
python scripts/debug_sam2_only.py --config configs/local_reference.yaml

# Without centroid fallback (pure YOLO boxes)
python scripts/debug_sam2_only.py --config configs/local_reference.yaml --no-fallback

# HPC
python scripts/debug_sam2_only.py --config configs/hpc_reference.yaml \
    video_path=data/raw/original_120s.avi
```

**What to look for:**
- Status bar shows `N masks (X box + Y fallback)` — see how many come from boxes vs centroid
- Masks merging into a single blob during interactions
- Mask quality when using `--no-fallback` vs default
- Score values: low scores (<0.5) indicate SAM2 is uncertain

## SAM2 without YOLO (Centroid Propagation)

Uses YOLO **only on the first frame** to initialize, then SAM2 tracks both rats
using previous-frame centroids as point prompts. No YOLO after initialization.

This answers the question: **is YOLO's per-frame intermittency causing SAM2 failures?**

```bash
# Pure centroid propagation (YOLO only on frame 0)
python scripts/debug_sam2_no_yolo.py --config configs/local_reference.yaml

# Specify which frame to initialize from (e.g., frame where both rats are visible)
python scripts/debug_sam2_no_yolo.py --config configs/local_reference.yaml --init-frame 50

# HPC parallel
bash scripts/run_debug_parallel.sh sam2_no_yolo data/raw/original_120s.avi
```

**How it works:**
1. Frame 0: YOLO detects 2 rats → SAM2 segments with boxes → gets 2 centroids
2. Frame N: Uses previous centroids as positive point prompts + other rat's centroid
   as negative prompt → SAM2 segments → updates centroids
3. No YOLO dependency after initialization

**What to look for:**
- Status bar shows `mode=CENTROID` (propagation) vs `mode=YOLO-INIT` (initialization)
- `YOLO used 1x` — should always be 1
- If masks are stable → YOLO was the bottleneck in the reference pipeline
- If masks drift or merge → SAM2 can't track from centroids alone

## HPC Parallel Execution

All debug scripts support parallel chunk execution:

```bash
# YOLO-only parallel
bash scripts/run_debug_parallel.sh yolo data/raw/original_120s.avi

# SAM2-only parallel
bash scripts/run_debug_parallel.sh sam2 data/raw/original_120s.avi

# SAM2 no-YOLO parallel
bash scripts/run_debug_parallel.sh sam2_no_yolo data/raw/original_120s.avi

# With extra flags
bash scripts/run_debug_parallel.sh sam2 data/raw/original_120s.avi 4 configs/hpc_reference.yaml "--no-fallback"
```

## Output

All scripts create a run directory under `outputs/runs/` with:
- `overlays/<mode>_debug_*.avi`
- `logs/run.log`
