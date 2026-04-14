# Pipeline Comparison

This project has 5 pipelines for rat segmentation and tracking.
All use YOLO (detection + keypoints) paired with a segmentation model
(SAM2 or SAM3), but differ in how they handle identity and what happens
when YOLO fails.

## Best Pipeline (without SAM3)

> **Use the `reference` pipeline.** It has the best balance of speed,
> robustness, and batch support. The centroid fallback + IdentityMatcher
> state machine handles interactions where YOLO loses a rat, without the
> BoT-SORT ID switching problems of sam2_yolo.

### Decision Tree

```
Start
  ├─ Need batch/parallel processing?
  │    ├─ Yes → reference (default), centroid, or sam3
  │    └─ No  → Any pipeline works
  ├─ Frequent close interactions / occlusions?
  │    ├─ Yes → centroid (SAM2 propagation, most stable masks)
  │    │        or reference (state machine handles merges)
  │    └─ No  → sam2_yolo is fine (simpler, same speed)
  ├─ Want most stable masks with least YOLO dependency?
  │    └─ Yes → centroid (YOLO only on frame 0, SAM2 drives the rest)
  ├─ Short video + heavy occlusion?
  │    └─ Yes → sam2_video (temporal memory, but slow + no batch)
  └─ Want to compare SAM2 vs SAM3 segmentation quality?
       └─ Yes → Run reference + sam3 on same clip, compare overlays
```

See also: [Risks and Known Limitations](risks.md)

---

## Comparison Table

| Aspect | sam2_yolo | sam2_video | reference | centroid | sam3 |
|--------|-----------|------------|-----------|----------|------|
| **Segmentation** | SAM2 ImagePredictor | SAM2 VideoPredictor | SAM2 ImagePredictor | SAM2 ImagePredictor (centroid propagation) | **SAM3** Sam3Processor |
| **YOLO** | detect + track (BoT-SORT) | detect-only | detect-only | detect-only (init + keypoints) | detect-only |
| **Tracking** | SlotTracker (YOLO IDs) | SAM2 temporal memory | IdentityMatcher (state machine) | SAM2 centroid ordering (inherent) | IdentityMatcher (state machine) |
| **Centroid fallback** | No | Yes (SAM2 temporal) | Yes (SAM2 point prompt) | N/A (centroids are primary) | Yes (SAM3 point prompt) |
| **Batch/chunks** | Yes | No | Yes | Yes | Yes |
| **Contacts** | Yes | Yes | Yes | Yes | Yes |
| **Speed** | Fast (~25 FPS) | Slow (~5-10 FPS) | Fast (~25 FPS) | Fast (~25 FPS) | TBD (depends on SAM3) |
| **Occlusion robustness** | Medium | High | Medium-High | High (SAM2 drives masks) | Medium-High |
| **Recommended** | No frequent interactions | Short + heavy occlusion | **Default choice** | Experimental — most stable masks | Evaluation / comparison |

---

## 1. sam2_yolo

**Module:** `src/pipelines/sam2_yolo/`
**Configs:** `configs/local_quick.yaml`, `configs/hpc_full.yaml`
**Slurm:** `slurm/run_infer.sbatch`, `slurm/run_chunks.sbatch`

### Per-frame flow

```
Frame → YOLO.track() (BoT-SORT) → boxes + track_ids + keypoints
      → SAM2ImagePredictor.predict(box=...) → masks
      → SlotTracker (YOLO IDs + Hungarian) → fixed slots
      → ContactTracker (optional)
      → Render overlay
```

### How tracking works

1. YOLO uses BoT-SORT internally → assigns track_id to each detection
2. SlotTracker maps track_ids to fixed slots (slot 0 = green, slot 1 = red)
3. If YOLO swaps IDs, a swap guard compares straight vs swapped cost
4. Fallback: Hungarian assignment with soft cost (distance + IoU + area)

### Batch support

```bash
# Job array with 4 parallel chunks
sbatch --array=0-3 slurm/run_chunks.sbatch

# Each chunk processes a range of frames
python -m src.pipelines.sam2_yolo.run --config configs/hpc_full.yaml \
    --start-frame 0 --end-frame 750 --chunk-id 0

# Merge at the end
python scripts/merge_chunks.py outputs/runs/*_chunk*/ -o outputs/runs/merged
```

### Weaknesses

- Depends on YOLO every frame. If YOLO loses a rat → empty slot, rat "disappears"
- BoT-SORT can cause ID switches during crossings/interactions
- No fallback: if YOLO doesn't detect, SAM2 has no prompt

### When to use

- Videos without frequent rat interactions
- When you need batch processing for long videos
- When speed is the priority

---

## 2. sam2_video

**Module:** `src/pipelines/sam2_video/`
**Configs:** `configs/local_sam2video.yaml`, `configs/hpc_sam2video.yaml`
**Slurm:** `slurm/run_sam2video.sbatch`

### Per-segment flow (2000 frames)

```
Segment:
  1. Extract frames to temporary JPEGs
  2. YOLO detect_only() on first frame → boxes for initialization
  3. SAM2VideoPredictor.init_state(frames_dir)
  4. SAM2.add_new_points_or_box(frame_idx=0, box=...) per rat
  5. SAM2.propagate_in_video() → generates masks for all frames
     Per frame:
       a. YOLO detect_only() → keypoints
       b. Match keypoints ↔ SAM2 masks (nose-in-mask or centroid)
       c. ContactTracker (optional)
       d. Render overlay
  6. Clean up JPEGs + GPU memory
```

### How tracking works

- SAM2VideoPredictor has a memory bank of 7 frames (memory_attention + memory_encoder)
- Once initialized with boxes, propagates masks automatically
- Doesn't need YOLO on intermediate frames (only for keypoints)
- Identity maintained by SAM2 internally (object pointers)
- Between segments: centroid matching to preserve colors

### Batch support

**No batch support.** Processes the full video at once (divided into
internal segments of ~2000 frames for memory management, but cannot
be parallelized across GPUs).

### Weaknesses

- Slow: ~3-5x slower than other pipelines
- Requires extracting frames to JPEG (additional I/O)
- Higher memory usage (GPU + RAM for frames)
- Cannot parallelize with chunks

### When to use

- Short videos with heavy interactions/occlusions
- When tracking quality matters more than speed
- When other pipelines fail on specific frames

---

## 3. reference (recommended)

**Module:** `src/pipelines/reference/`
**Configs:** `configs/local_reference.yaml`, `configs/hpc_reference.yaml`
**Slurm:** `slurm/run_reference.sbatch`

### Per-frame flow

```
Frame → YOLO detect_only() → boxes + keypoints (NO BoT-SORT)
      → SAM2ImagePredictor.set_image(frame)
      → For each detection: SAM2.predict(box=...) → mask
      → If YOLO detected fewer rats than active slots:
          For each unmatched slot:
            SAM2.predict(point_coords=prev_centroid) → fallback mask
      → filter_duplicates() → remove redundant masks
      → IdentityMatcher.match() → assign to fixed slots
      → ContactTracker (optional)
      → Render overlay
```

### How tracking works

1. IdentityMatcher uses Hungarian assignment with 3-component soft cost:
   - Distance cost (weight 0.4): normalized by `proximity_threshold`
   - Mask IoU cost (weight 0.4): shape continuity between frames
   - Area ratio cost (weight 0.2): size consistency
2. Velocity prediction with EMA smoothing (0.7/0.3 blend)
3. Swap guard for N=2 crossings: compares straight vs swapped total cost
4. **State machine** (SEPARATE ↔ MERGED) for close interaction handling:
   - MERGED: freezes identity, coasts with velocity, skips centroid fallback
   - Split resolution: 2×2 Hungarian using pre-merge evidence

See [Identity Matcher Design](identity_matcher.md) for full details.

### Centroid fallback (key feature)

When YOLO detects only 1 rat but IdentityMatcher knows there are 2 active slots:

```python
# The lost slot has a previous centroid
prev_centroid = matcher.prev_centroids[slot_idx]

# Used as point prompt for SAM2
SAM2.predict(point_coords=prev_centroid, point_labels=[1])
```

This recovers the lost rat's mask using SAM2 with a point instead of a box.
This is what makes the pipeline work during occlusions.

### Batch support

```bash
# Same chunk args as sam2_yolo
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
    --start-frame 0 --end-frame 750 --chunk-id 0

# Or use run_parallel.sh (recommended):
bash scripts/run_parallel.sh data/raw/original_120s.avi
```

### Weaknesses

- No SAM2 temporal memory → identity can flip with very fast movement
- proximity_threshold needs tuning per resolution
- Centroid fallback depends on previous centroid being close to current position

### When to use

- **Default choice** for most videos
- Videos with interactions where YOLO loses rats momentarily
- When you need batch processing + robustness
- Direct alternative to sam2_yolo when ID switches are a problem

---

## 4. centroid (SAM2 centroid propagation — experimental)

**Module:** `src/pipelines/centroid/`
**Configs:** `configs/local_centroid.yaml`, `configs/hpc_centroid.yaml`

### Architecture

Inverts the reference pipeline's architecture:
- **SAM2 centroid propagation** drives masks and identity (stable backbone)
- **YOLO** is demoted to a keypoint provider on the full image (noisy signal)
- **No IdentityMatcher** — identity is inherent from SAM2 propagation ordering
- **Temporal carry-over** fills missing keypoints when YOLO misses a rat

### Per-frame flow

```
Frame 0 (init):
  YOLO detect_only() → boxes + keypoints
  → SAM2.predict(box=...) per detection → init masks + centroids
  → Store keypoints

Frame N (propagation):
  1. SAM2.predict(
       point_coords=prev_centroid,      # positive: this rat
       point_coords=other_centroid,      # negative: other rat
     ) → 2 stable masks
  2. Update centroids from new masks
  3. YOLO detect_only() on full image → 0-2 detections with keypoints
  4. Assign keypoints to masks:
     - Check if detection center falls inside a mask
     - Fallback: assign to nearest mask centroid
  5. Temporal carry-over (for missing YOLO detections):
     - Use previous frame's keypoints shifted by centroid delta
  6. ContactTracker.update()
  7. Render: masks + keypoints + contacts
```

### Key differences from reference

| Aspect | Reference | Centroid |
|--------|-----------|----------|
| SAM2 prompts | YOLO boxes + keypoints every frame | Centroids from previous frame (YOLO only on init) |
| Identity source | IdentityMatcher (Hungarian matching) | Centroid ordering (inherent from propagation) |
| Keypoint assignment | `match_dets_to_slots()` by centroid distance | Mask containment (detection center inside mask) |
| Missing detections | Centroid fallback (skipped during MERGED) | Temporal carry-over of previous keypoints |
| State machine | SEPARATE/MERGED | Not needed (SAM2 handles interactions) |
| YOLO role | Drives everything (boxes → SAM2 → masks) | Keypoint-only provider; boxes ignored after init |

### Batch support

```bash
# Using run_parallel.sh with pipeline argument
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_centroid.yaml "" centroid

# Direct chunk execution
python -m src.pipelines.centroid.run --config configs/hpc_centroid.yaml \
    --start-frame 0 --end-frame 750 --chunk-id 0
```

### Strengths

- Most stable masks: SAM2 centroid propagation proven at ~1 blink in 3600 frames
- No YOLO dependency for masks (only needs YOLO on frame 0 for initialization)
- Identity cannot swap during interactions (SAM2 negative prompts handle separation)
- Simpler architecture: no IdentityMatcher, no state machine

### Weaknesses

- Keypoints still come from YOLO (noisy, can miss rats on some frames)
- Temporal carry-over is approximate (shifts by centroid delta, not true tracking)
- If SAM2 loses a mask entirely, no recovery mechanism (would need re-initialization)
- Experimental: not yet validated on full-length videos

### When to use

- Videos with frequent close interactions where reference pipeline identity swaps
- When mask stability is the priority over keypoint accuracy
- Comparing SAM2-driven vs YOLO-driven identity approaches

---

## 5. sam3 (SAM3 evaluation)

**Module:** `src/pipelines/sam3/`
**Configs:** `configs/local_sam3.yaml`, `configs/hpc_sam3.yaml`

### Per-frame flow

```
Frame → YOLO detect_only() → boxes + keypoints (NO BoT-SORT)
      → Sam3Processor.set_image(frame)
      → For each detection: SAM3.predict(box=..., points=...) → mask
        (all coords normalized to [0,1])
      → If YOLO detected fewer rats than active slots:
          For each unmatched slot:
            SAM3.predict(point_coords=prev_centroid_normalized) → fallback mask
      → filter_duplicates() → remove redundant masks
      → IdentityMatcher.match() → assign to fixed slots
      → ContactTracker (optional)
      → Render overlay
```

### How tracking works

Identical to the reference pipeline — same IdentityMatcher, same state machine,
same cost function. The only difference is the segmentation model (SAM3 vs SAM2).

### Key difference: coordinate normalization

SAM3 uses **normalized [0,1] coordinates** instead of pixel coordinates.
The `sam3_processor.py` converts all YOLO-space pixel prompts:

```python
# Box: [x1, y1, x2, y2] pixels → [x1/w, y1/h, x2/w, y2/h]
# Points: [[x, y]] pixels → [[x/w, y/h]]
```

### Requirements

- Python 3.12, PyTorch 2.7+, CUDA 12.6
- SAM3 package: `pip install -e ./models/sam3/sam3`
- Checkpoint: `models/sam3/sam3.pt` (~3.2GB from HuggingFace)
- See [Local Setup](../setup/local.md) and [HPC Setup](../setup/hpc.md) for installation

### Weaknesses

- Same as reference pipeline (no SAM temporal memory, resolution-dependent thresholds)
- SAM3 checkpoint is ~3.2GB (larger than SAM2 tiny/small)
- SAM3 requires Python 3.12 (may conflict with existing SAM2 venv)
- API not yet battle-tested in this project

### When to use

- Evaluating SAM3 vs SAM2 segmentation quality
- If SAM3 produces better masks on your specific videos
- Future-proofing: SAM3 will likely improve over time

---

## Component Comparison

### Models

| Component | sam2_yolo | sam2_video | reference | centroid | sam3 |
|-----------|-----------|------------|-----------|----------|------|
| YOLO | `model.track()` | `model()` detect-only | `model()` detect-only | `model()` detect-only (init + keypoints) | `model()` detect-only |
| Segmentation | `SAM2ImagePredictor` | `SAM2VideoPredictor` | `SAM2ImagePredictor` | `SAM2ImagePredictor` (centroid prompts) | `Sam3Processor` |
| Model build | `build_sam2()` | `build_sam2_video_predictor()` | `build_sam2()` | `build_sam2()` | `build_sam3_image_model()` |

### Tracking / Identity

| Component | sam2_yolo | sam2_video | reference | centroid | sam3 |
|-----------|-----------|------------|-----------|----------|------|
| Class | `SlotTracker` | SAM2 internal | `IdentityMatcher` | None (inherent) | `IdentityMatcher` |
| File | `src/common/tracking.py` | N/A | `src/pipelines/reference/identity_matcher.py` | N/A | (same as reference) |
| Identity based on | YOLO track IDs + Hungarian | SAM2 object pointers | Hungarian + soft cost + state machine | SAM2 centroid propagation ordering | (same as reference) |
| Uses YOLO track IDs | Yes (primary) | No | No | No | No |
| Missing frames | 5 frame tolerance | N/A (SAM2 propagates) | Centroid fallback + velocity coasting | Temporal keypoint carry-over | Centroid fallback + velocity coasting |
| Swap guard | Yes | N/A | Yes | N/A (SAM2 negative prompts) | Yes |

### Config Files

| Pipeline | Local | HPC | Slurm |
|----------|-------|-----|-------|
| sam2_yolo | `configs/local_quick.yaml` | `configs/hpc_full.yaml` | `slurm/run_infer.sbatch`, `slurm/run_chunks.sbatch` |
| sam2_video | `configs/local_sam2video.yaml` | `configs/hpc_sam2video.yaml` | `slurm/run_sam2video.sbatch` |
| reference | `configs/local_reference.yaml` | `configs/hpc_reference.yaml` | `slurm/run_reference.sbatch` |
| centroid | `configs/local_centroid.yaml` | `configs/hpc_centroid.yaml` | — (use `run_parallel.sh`) |
| sam3 | `configs/local_sam3.yaml` | `configs/hpc_sam3.yaml` | — |

### Shared Modules

All pipelines reuse these common modules:

| Module | File | Purpose |
|--------|------|---------|
| Video I/O | `src/common/io_video.py` | Read/write video, iterate frames |
| Config | `src/common/config_loader.py` | YAML + CLI overrides |
| Metrics | `src/common/metrics.py` | mask_iou, compute_centroid |
| Visualization | `src/common/visualization.py` | Overlays, keypoints, centroids |
| Contacts | `src/common/contacts.py` | ContactTracker with bout detection |
| Data classes | `src/common/utils.py` | Detection, Keypoint dataclasses |

---

## Quick Commands

### Local test (10s clip)

```bash
# reference (recommended)
python -m src.pipelines.reference.run --config configs/local_reference.yaml

# centroid (experimental — SAM2-driven)
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml

# sam3 (requires sam3 package installed)
python -m src.pipelines.sam3.run --config configs/local_sam3.yaml

# sam2_yolo
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml

# sam2_video
python -m src.pipelines.sam2_video.run --config configs/local_sam2video.yaml
```

### Bunya HPC (2 min video)

```bash
# reference (recommended, default)
bash scripts/run_parallel.sh data/raw/original_120s.avi

# centroid (experimental)
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_centroid.yaml "" centroid

# sam3
python -m src.pipelines.sam3.run --config configs/hpc_sam3.yaml \
    video_path=data/raw/original_120s.avi

# sam2_yolo
sbatch --export=ALL,CONFIG=configs/hpc_full.yaml,OVERRIDES="video_path=data/raw/original_120s.avi" slurm/run_infer.sbatch

# sam2_video
sbatch --export=ALL,CONFIG=configs/hpc_sam2video.yaml,OVERRIDES="video_path=data/raw/original_120s.avi" slurm/run_sam2video.sbatch
```

### With contacts enabled

```bash
# reference with contacts
python -m src.pipelines.reference.run --config configs/local_reference.yaml contacts.enabled=true

# centroid with contacts
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml contacts.enabled=true

# On Bunya (reference):
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_reference.yaml "contacts.enabled=true"

# On Bunya (centroid):
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_centroid.yaml "contacts.enabled=true" centroid
```
