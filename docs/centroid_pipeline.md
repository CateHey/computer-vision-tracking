# Centroid Pipeline — Technical Documentation

> Production pipeline for laboratory rat tracking and social contact classification.
> SAM2 centroid propagation (identity) + YOLO keypoints (read-only, for contacts).

---

## Overview

Tracks 2 laboratory rats in video using:

| Model | Role |
|-------|------|
| **YOLO26** | Frame 0: bounding boxes for init. Every frame: keypoints for contacts (read-only) |
| **SAM2** | Segments each rat every frame (pixel-precise mask via centroid propagation) |

**Key principle**: SAM2 masks are the sole source of identity. YOLO provides keypoints as a read-only overlay — it never influences SAM2 prompting.

---

## Architecture

```
Frame 0 (initialization):
  YOLO detect → 2 boxes → SAM2 segment (box prompts) → 2 masks → centroids

Frame 1..N (steady state):
  For each rat:
    SAM2.predict(
      point_coords = [[this_centroid], [other_centroid]],
      point_labels = [1, 0],         # positive, negative
      box = None,
      multimask_output = False,      # CRITICAL: must be False
    )
  → compute centroids from new masks
  → resolve_overlaps for contested pixels

  YOLO detect → keypoints (read-only, detection order irrelevant)
  → assign keypoints to masks by containment
  → carry-over missing keypoints from previous frame
  → ContactTracker.update() for social contacts
  → render overlay
```

---

## How It Works

### Step 1: YOLO Initialization (frame 0 only)

```python
detections = detect_only(yolo, frame_rgb, confidence=0.25)
masks, scores = _segment_from_boxes(sam, frame_rgb, detections, threshold)
centroids = [compute_centroid(m) for m in masks]
```

- Runs once to get initial bounding boxes
- SAM2 segments each box → 2 masks
- Centroids computed from masks
- YOLO never runs again after this

### Step 2: SAM2 Centroid Propagation (every frame after init)

```python
masks, scores = _segment_from_centroids(sam, frame_rgb, prev_centroids, threshold)
```

- Previous frame's centroid as **positive prompt** ("segment here")
- Other rat's centroid as **negative prompt** ("NOT here")
- `multimask_output=False` → single direct mask (no candidate selection)
- Centroids update from new masks; if mask empty, keep previous centroid

### Step 3: Overlap Resolution

```python
resolve_overlaps(slot_masks, slot_centroids)
```

- When two masks share pixels, each contested pixel goes to nearest centroid
- Ensures no pixel belongs to both rats

---

## Validated Findings

### Finding 1: `multimask_output=False` is critical

| Setting | Result |
|---------|--------|
| `multimask_output=True` (3 candidates, pick best score) | **Poor tracking** — masks degrade, identity unstable |
| `multimask_output=False` (1 direct mask) | **Stable tracking** — masks clean, identity maintained |

Why True fails: the "best score" candidate is not the best for temporal consistency. The score reflects single-frame quality, not tracking continuity. Masks jump between interpretations frame-to-frame.

**Rule: Always use `multimask_output=False`. Do not change this.**

### Finding 2: YOLO every frame destroys identity

| Approach | Result |
|----------|--------|
| YOLO every frame → boxes as SAM2 prompts | **Identity swaps** — permanent, unrecoverable |
| YOLO init only → SAM2 centroid propagation | **Best results** — stable identity throughout |

Why: YOLO detection order is arbitrary (no tracking memory). `detection[0]` might be either rat. When used as SAM2 prompts, wrong box → wrong slot → permanent swap.

**Rule: YOLO runs ONLY on frame 0. Never after.**

### Finding 3: SAM2 centroid propagation maintains identity

Validated on 3600 frames (120s video): only 1 blink in entire video.

What it handles well:
- Rats crossing paths
- Prolonged close interactions (negative prompts keep masks separate)
- Rapid posture changes
- Temporary occlusion

### Finding 4: Reference pipeline problems (why we switched)

The reference pipeline (YOLO every frame + IdentityMatcher) had 4 failure modes:

1. **Mask bleeding** — overlapping YOLO boxes → SAM2 segments both rats as one blob
2. **ID swaps** — IdentityMatcher MERGED state timeout → wrong assignment
3. **Rat disappears** — YOLO intermittent detection → missing masks for 10-50+ frames
4. **Flickering** — YOLO detection count fluctuates between 1 and 2 every frame

All traced to YOLO's per-frame instability. SAM2 centroid propagation eliminates all 4.

---

## What Was Removed and Why

| Removed Feature | Why |
|----------------|-----|
| YOLO box prompts after init | Caused identity swaps via YOLO ordering |
| `multimask_output=True` | Degraded tracking quality |
| YOLO every-frame detection | Unnecessary — SAM2 alone tracks better |
| IdentityMatcher (Hungarian assignment) | SAM2 propagation inherently maintains identity |
| SEPARATE/MERGED state machine | No longer needed |
| Keypoint assignment/stabilization | Removed with YOLO (to be re-added for contacts) |

---

## Contact Detection (Approach A — IMPLEMENTED, TESTING)

**Status: Implemented, awaiting test results to evaluate.**

YOLO runs every frame as a read-only keypoint provider. SAM2 prompting is untouched.

```
Frame N:
  1. SAM2 centroid propagation → 2 masks (identity preserved, untouched)
  2. YOLO detect → keypoints (detection order irrelevant)
  3. Assign keypoints to masks by containment (box center inside mask)
  4. Carry-over missing keypoints from previous frame (shifted by centroid delta)
  5. ContactTracker.update(slot_dets, masks, centroids)
```

**Why it should be safe**: YOLO never touches SAM2 prompting. YOLO is a read-only consumer of SAM2 identity. The previous swaps came from YOLO *boxes* as SAM2 prompts, not from keypoint extraction.

**Risk to SAM2**: LOW — YOLO ordering never influences identity.

### What to watch for during testing

1. **Identity swaps** — Do the masks (green/red) ever swap between rats? If yes, SAM2 prompting is somehow affected (should NOT happen with this approach)
2. **Keypoint assignment errors** — Are keypoints from rat A appearing on rat B's mask? Check during close interactions when YOLO boxes overlap
3. **Carried-over keypoint quality** — When YOLO misses a rat, carried-over keypoints shift by centroid delta. Are they accurate enough for contacts?
4. **Contact classification accuracy** — Are N2N, N2AG, etc. firing at the right moments? Are there false positives or missed contacts?
5. **Performance** — Does YOLO running every frame slow things down significantly on HPC?
6. **Carry-over percentage** — Log shows `Frames with keypoint carry-over: X/Y (Z%)`. High % means YOLO is missing rats often

### If Approach A fails

If identity swaps appear → something is wrong in the implementation (debug).
If keypoints are too unreliable during interactions → try **Approach B** (mask geometry).
If YOLO is too slow → consider running YOLO every N frames instead of every frame.

### Approach B: Mask Geometry (Skeleton-Free)

Use SAM2 mask shape to infer body parts without any pose model:
- Medial axis transform → skeleton of mask shape
- PCA orientation → head-tail direction
- Extremity detection → narrower end = head

Paper: [Self-Supervised Body Part Segmentation for Rats](https://arxiv.org/abs/2405.04650)

**Risk to SAM2**: NONE (purely post-processing on masks)
**Limitation**: Less precise than keypoints, struggles with curled postures

### Approach C: DeepLabCut / SLEAP

Replace YOLO with dedicated pose estimation ([DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), [SLEAP](https://sleap.ai/)). Requires training data.

**Risk to SAM2**: NONE (separate model)
**When to use**: If YOLO keypoints are consistently bad

### Approach D: SAM2 Multi-Prompt Body Parts — NOT RECOMMENDED

Multiple SAM2 prompts per rat for body parts. **Risk: HIGH** — could interfere with centroid propagation.

### What ContactTracker Needs

| Contact | Keypoints Required | Masks Required |
|---------|-------------------|----------------|
| N2N | nose (both rats) | No |
| N2AG | nose + tail_base | No |
| T2T | tail_base (both) | No |
| FOL | nose + tail_base + velocity | No |
| SBS | (optional: orientation) | YES (mask IoU) |
| N2B | nose | YES (containment) |

7 keypoints: tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear

---

## Configuration

```yaml
video_path: data/clips/output-10s.mp4
output_dir: outputs/runs

models:
  yolo_path: models/yolo/best.pt
  sam2_checkpoint: models/sam2/.../sam2.1_hiera_tiny.pt
  sam2_config: configs/sam2.1/sam2.1_hiera_t.yaml
  device: auto

detection:
  confidence: 0.25
  max_animals: 2
  keypoint_names: [tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear]
  keypoint_min_conf: 0.3

segmentation:
  sam_threshold: 0.0

contacts:
  enabled: false
  min_keypoint_conf: 0.3
  contact_zone_bl: 0.3
  proximity_zone_bl: 1.0
  fallback_body_length_px: 120
  sbs_mask_iou_min: 0.02
  sbs_max_velocity_bl: 0.04
  sbs_parallel_cos_min: 0.7
  follow_radius_bl: 0.5
  follow_min_speed_bl: 0.025
  follow_alignment_cos: 0.7
  follow_min_frames: 30
  bout_max_gap_frames: 3
  bout_min_duration_frames: 9
  mask_overlap_warning: 0.5
  det_slot_match_radius: 200.0

output:
  video_codec: XVID
  overlay_colors:
    - [0, 255, 0, 150]    # Rat 0: Green
    - [255, 0, 0, 150]    # Rat 1: Red

scan:
  max_frames: null
```

---

## Running

### Local

```bash
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml
```

### HPC (Bunya — multi-GPU parallel)

```bash
ssh s4948012@bunya.rcc.uq.edu.au
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

salloc --partition=gpu_cuda --qos=gpu --gres=gpu:3 --cpus-per-task=8 --mem=64G --time=24:00:00
srun --pty bash

bash scripts/run_parallel.sh data/raw/original_120s.avi
```

---

## Output

```
outputs/runs/<timestamp>_centroid/
├── overlays/
│   └── centroid_2026-03-07.avi      # Overlay video with masks + centroids
├── contacts/                         # (only if contacts.enabled=true)
│   ├── contacts_raw.csv
│   ├── contacts_real_events.csv
│   ├── session_summary_real.json
│   ├── event_log.txt
│   ├── contacts_timeline.png
│   └── contacts_report.tar.gz
└── logs/
    └── run.log
```

---

## Test Checklist

### Test 1: Rats far apart
- [ ] Both masks clean, separate
- [ ] Status bar shows `CENTROID`

### Test 2: Rats approaching
- [ ] Masks remain separate

### Test 3: Rats interacting (touching/overlapping)
- [ ] Both rats have masks (resolve_overlaps handles bleeding)
- [ ] **No identity swap** (same color stays on same rat)

### Test 4: Rats separating after interaction
- [ ] Masks recover cleanly
- [ ] No identity swap

### Test 5: Full video (120s)
- [ ] Pipeline completes without crash
- [ ] Overlay video shows masks on both rats at all times

---

## Rules for Future Development

1. **Never use `multimask_output=True`** for centroid propagation
2. **Never use YOLO boxes as SAM2 prompts** after initialization
3. **Never let YOLO detection order influence SAM2 identity**
4. **SAM2 centroid prompts are the sole source of identity**
5. Any additions (keypoints, contacts) must work *on top of* SAM2 masks without modifying SAM2 prompting

---

## Research Sources

- [Self-Supervised Body Part Segmentation for Rats](https://arxiv.org/abs/2405.04650)
- [Multi-animal pose estimation with DeepLabCut](https://www.nature.com/articles/s41592-022-01443-0)
- [SLEAP: Multi-animal pose tracking](https://www.nature.com/articles/s41592-022-01426-1)
- [Multi-animal 3D social pose estimation](https://www.nature.com/articles/s42256-023-00776-5)
- [SAM2MOT: Multi-Object Tracking by Segmentation](https://arxiv.org/html/2504.04519v1)
