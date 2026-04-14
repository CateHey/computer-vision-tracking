# Tracking Techniques Analysis — BoT-SORT & ByteTrack

Analysis of the Ultralytics tracker YAML configurations and their impact
on lab rat tracking with occlusions, wall interactions, and close contact.

All information sourced from the installed `ultralytics==8.4.14` package.

---

## 1. Tracker Overview

Our pipeline uses `model.track(persist=True, tracker="botsort.yaml")` which
delegates to the Ultralytics BoT-SORT implementation. BoT-SORT extends
ByteTrack with Global Motion Compensation (GMC) and optional ReID.

Both trackers share the same two-stage association algorithm from ByteTrack.
BoT-SORT adds camera motion handling and appearance features on top.

---

## 2. Parameter Analysis

### 2.1 `track_buffer` — Lost Track Lifetime

| | |
|---|---|
| **Default** | `30` |
| **Type** | int (frames) |
| **Classification** | **Pipeline Improvement** (already parametrizable) |

**What it does:** Number of frames a lost track survives before permanent deletion.
When YOLO misses a detection (e.g., rat partially behind the other), the track
enters "Lost" state and its Kalman filter continues predicting position. If matched
again within `track_buffer` frames, the track resumes with the same ID.

**Important detail:** The actual frame count is `int(frame_rate / 30.0 * track_buffer)`.
Ultralytics hardcodes `frame_rate=30`, so `track_buffer: 30` always = 30 frames
regardless of actual video FPS.

**Track lifecycle:**
```
New → Tracked → [detection lost] → Lost → [matched again] → Tracked
                                        → [buffer exceeded] → Removed (permanent)
```

**Tradeoffs for lab rats:**

| Value | Behavior | When to use |
|-------|----------|-------------|
| 15-20 | Short memory, quick ID release | Fast-moving animals, no prolonged occlusions |
| 30 (default) | ~1 second at 25fps | General purpose, brief wall interactions |
| 60-90 | 2-3 seconds | Rats that hide behind each other frequently |
| 120+ | 4+ seconds | Extended grooming/huddling where one rat is hidden |

**Risk of high values:** If a track is lost and the rat reappears far away, the
old track may incorrectly match to a different rat. The Kalman prediction
diverges from reality after many frames, but the Hungarian algorithm may
still find a "least bad" match.

**Recommendation for lab rats:** `track_buffer: 60` — rats in 640x480 lab videos
rarely disappear for more than 2 seconds, and this gives enough buffer for
brief wall occlusions without excessive ID drift risk.

---

### 2.2 Thresholds — Two-Stage Matching

#### How two-stage matching works

```
All YOLO detections
  │
  ├── conf >= track_high_thresh (0.25)
  │     → FIRST PASS: match against ALL tracks (active + lost)
  │       Uses: IoU distance + fuse_score + ReID (if enabled)
  │       Hungarian threshold: match_thresh (0.8)
  │       Unmatched detections with conf >= new_track_thresh → create new track
  │
  ├── track_low_thresh (0.1) < conf < track_high_thresh (0.25)
  │     → SECOND PASS: match against REMAINING active tracks only
  │       Uses: raw IoU distance only (no fuse_score, no ReID)
  │       Hungarian threshold: hardcoded 0.5 (stricter, NOT configurable)
  │
  └── conf <= track_low_thresh (0.1)
        → DISCARDED entirely
```

#### `track_high_thresh` (default: 0.25)

| | |
|---|---|
| **Classification** | **Pipeline Improvement** (already parametrizable) |

Detections with confidence >= this enter the primary association pass with the
full cost matrix. Aligns with our `detection.confidence: 0.25`.

**Recommendation:** Keep at `0.25` to stay synchronized with the pipeline's
confidence threshold.

#### `track_low_thresh` (default: 0.1)

| | |
|---|---|
| **Classification** | **Pipeline Improvement** (already parametrizable) |

Absolute floor — detections below this are never used. Detections between
`low` and `high` thresholds enter the second pass (IoU-only, stricter).

**Why this matters for rats:** When a rat is partially occluded by the wall or
the other rat, YOLO confidence drops (e.g., from 0.6 to 0.15). The second stage
ensures these low-confidence detections can still match existing active tracks.

**Recommendation:** Keep at `0.1`.

#### `new_track_thresh` (default: 0.25)

| | |
|---|---|
| **Classification** | **Pipeline Improvement** (already parametrizable) |

After both passes, unmatched first-pass detections with confidence >= this create
new tracks. Since default equals `track_high_thresh`, every unmatched high-conf
detection creates a track. Setting higher (e.g., 0.3) creates a gap where
medium-confidence unmatched detections don't spawn spurious tracks.

**Recommendation:** `new_track_thresh: 0.3` — reduces false track creation from
transient noise while still picking up new rats quickly.

---

### 2.3 `match_thresh` — Association Cost Threshold

| | |
|---|---|
| **Default** | `0.8` |
| **Classification** | **Pipeline Improvement** (already parametrizable) |

Maximum cost for Hungarian assignment in the first pass. Cost = IoU distance
(1 - IoU), optionally fused with confidence.

- `0.8` → allows matches with IoU as low as 0.2 (very permissive)
- `0.5` → requires IoU >= 0.5 (stricter)

**Note:** The second pass and unconfirmed track association use hardcoded
thresholds (0.5 and 0.7 respectively) — NOT configurable via YAML.

**Recommendation:** Keep at `0.8`. Frame-to-frame IoU for lab rats is usually
high (>0.5) but can drop during fast movement or near walls. The permissive
threshold lets the Kalman filter bridge these gaps.

---

### 2.4 `fuse_score` — Confidence-IoU Fusion

| | |
|---|---|
| **Default** | `True` |
| **Classification** | **Pipeline Improvement** (already parametrizable) |

Multiplies IoU similarity by detection confidence: `fused_cost = 1 - (IoU * confidence)`.

**Effect:**

| Detection conf | IoU | Cost without fuse | Cost with fuse |
|---------------|-----|-------------------|----------------|
| 0.9 | 0.6 | 0.40 | 0.46 |
| 0.3 | 0.6 | 0.40 | 0.82 |

Biases matching toward high-confidence detections. Prevents low-confidence noise
from stealing tracks from legitimate detections.

**Recommendation:** Keep `True`.

---

### 2.5 `gmc_method` — Global Motion Compensation (BoT-SORT only)

| | |
|---|---|
| **Default** | `sparseOptFlow` |
| **Classification** | **Pipeline Improvement** (already parametrizable) |

Estimates camera motion between frames and compensates Kalman filter predictions.

| Method | Description | Use case |
|--------|-------------|----------|
| `sparseOptFlow` | Lucas-Kanade optical flow | Moving/shaking cameras |
| `orb` | ORB feature matching | Moving cameras, moderate speed |
| `sift` | SIFT feature matching | Moving cameras, best accuracy |
| `ecc` | Enhanced Correlation Coefficient | Small camera vibrations |
| `none` | Disabled (identity transform) | **Static cameras** |

**For lab rat tracking:** Our cameras are static (fixed overhead mount). Camera
motion compensation adds ~5-10% compute with no benefit.

**Recommendation:** `gmc_method: none`.

---

### 2.6 `with_reid` — Re-Identification (BoT-SORT only)

| | |
|---|---|
| **Default** | `False` |
| **Classification** | **Technique Analyzed** (available but requires evaluation) |

Enables appearance-based feature extraction for track matching. When enabled,
the final cost is the element-wise minimum of IoU cost and appearance cost:

```
final_cost = min(IoU_cost, appearance_cost)
```

If either IoU OR appearance produces a good match, the pair is associated.
Helps recover tracks after occlusion when IoU is poor but appearance is
recognizable.

**Two modes:**
- `model: auto` — uses YOLO's own intermediate features as embeddings.
  Zero extra model overhead (hooks into the Detect layer input).
- `model: yolov8n-cls.pt` — separate classification model as encoder.
  Crops each detection, runs through encoder. Adds significant latency.

**Related parameters (only active when `with_reid: True`):**
- `proximity_thresh: 0.5` — spatial gate: only consider ReID for pairs with
  IoU >= 0.5. Prevents matching across the frame.
- `appearance_thresh: 0.8` — minimum embedding similarity. Rejects pairs
  with low appearance match.

**Expected impact for visually similar lab rats:**

| Scenario | Expected benefit |
|----------|-----------------|
| Different colored rats | High — ReID can distinguish by appearance |
| Same color, different size | Medium — subtle size features may help |
| Identical appearance (our case) | **Low** — features may not discriminate |
| After prolonged occlusion | Medium — even partial appearance matching helps |

**Risk:** For visually identical rats, ReID might cause MORE swaps by confidently
matching the wrong rat based on similar appearance.

**Recommendation:** Start with `with_reid: False` (current). If ID switches
persist during prolonged occlusions, test `with_reid: True, model: auto` as
it adds minimal overhead. Expect limited benefit for visually identical rats.

---

## 3. Classification Summary

| Parameter | Default | Classification | Recommended |
|-----------|---------|---------------|-------------|
| `track_buffer` | 30 | **Pipeline Improvement** | 60 |
| `track_high_thresh` | 0.25 | **Pipeline Improvement** | 0.25 (keep) |
| `track_low_thresh` | 0.1 | **Pipeline Improvement** | 0.1 (keep) |
| `new_track_thresh` | 0.25 | **Pipeline Improvement** | 0.3 |
| `match_thresh` | 0.8 | **Pipeline Improvement** | 0.8 (keep) |
| `fuse_score` | True | **Pipeline Improvement** | True (keep) |
| `gmc_method` | sparseOptFlow | **Pipeline Improvement** | `none` |
| `with_reid` | False | **Technique Analyzed** | False (test if needed) |
| `proximity_thresh` | 0.5 | **Pipeline Improvement** | 0.5 (keep) |
| `appearance_thresh` | 0.8 | **Pipeline Improvement** | 0.8 (keep) |

---

## 4. Recommended BoT-SORT Config for Lab Rat Tracking

Save as `configs/trackers/lab_rats_botsort.yaml`:

```yaml
# BoT-SORT config optimized for lab rat tracking
# Static overhead camera, 2 visually similar rats, 25fps, 640x480

tracker_type: botsort

# --- Two-stage association ---
track_high_thresh: 0.25    # First pass: confident detections
track_low_thresh: 0.1      # Second pass floor: recover partial occlusions
new_track_thresh: 0.3      # Slightly above high_thresh to reduce false new tracks

# --- Track persistence ---
track_buffer: 60           # ~2 seconds at 25fps. Survives brief wall occlusions
                           # without excessive ID drift risk.

# --- Association ---
match_thresh: 0.8          # Permissive IoU matching (Kalman bridges gaps)
fuse_score: True           # Bias toward high-confidence detections

# --- Camera motion ---
gmc_method: none           # Static camera — no compensation needed.
                           # Saves ~5-10% compute per frame.

# --- ReID (disabled by default) ---
# Enable with_reid if ID switches persist during prolonged rat-on-rat occlusion.
# Expect limited benefit for visually identical rats.
proximity_thresh: 0.5      # Spatial gate for ReID
appearance_thresh: 0.8     # Appearance similarity gate
with_reid: False           # Set True to test ReID
model: auto                # Uses YOLO features (no extra model)
```

**Usage:**

```yaml
# In your pipeline config (local_quick.yaml or hpc_full.yaml):
tracking:
  tracker_config: configs/trackers/lab_rats_botsort.yaml
```

```bash
# Or via CLI override:
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    tracking.tracker_config=configs/trackers/lab_rats_botsort.yaml
```

---

## 5. BoT-SORT vs ByteTrack

ByteTrack is BoT-SORT without GMC and ReID. Shares all 7 base parameters.

| Feature | BoT-SORT | ByteTrack |
|---------|----------|-----------|
| Two-stage matching | Yes | Yes |
| Kalman filter | XYWH (width+height independent) | XYAH (aspect ratio coupled) |
| Global Motion Compensation | Yes (`gmc_method`) | No |
| ReID appearance features | Yes (`with_reid`) | No |
| Compute overhead | Higher (GMC + optional ReID) | Lower |

With our recommended config (`gmc_method: none`, `with_reid: False`), BoT-SORT
is nearly identical to ByteTrack. The remaining difference is Kalman filter
parameterization — BoT-SORT's independent width/height (XYWH) is slightly
better for rats whose aspect ratio changes (stretching, curling up).

**Recommendation:** Stay with BoT-SORT + the optimized config.

---

## 6. SlotTracker Tuning (Post-BoT-SORT Layer)

After BoT-SORT assigns track IDs, our `SlotTracker` (`src/common/tracking.py`)
maps those IDs to fixed color slots using Hungarian assignment with a soft
cost matrix.

### 6.1 Changes Applied

| Parameter | Previous | New | Rationale |
|-----------|----------|-----|-----------|
| `cost_threshold` | 0.85 | **0.75** | Accept slightly imperfect matches during occlusion rather than dropping assignments |
| `nms_iou` | null (YOLO default ~0.7) | **0.5** | Suppress merged/duplicate boxes when rats touch or overlap |
| `tracker_config` | `botsort.yaml` (default) | `configs/trackers/lab_rats_botsort.yaml` | Custom BoT-SORT tuned for static camera, 2 rats |

### 6.2 Swap Guard

**Problem:** During crossings and close contact, BoT-SORT can swap track IDs
between the two rats. Since `_match_by_track_id()` previously trusted YOLO
track IDs unconditionally, a swap at the BoT-SORT level propagated directly
into our color slot assignments, causing the overlay colors to flip.

**Solution:** Before accepting track_id-based matches, the swap guard computes
the cost of two possible assignments:

1. **Straight** — assign each mask to the slot its YOLO track_id maps to
2. **Swapped** — assign each mask to the other rat's slot

If the swapped assignment has lower total cost (centroid distance + mask IoU +
area change), we reject the track_id matches entirely and delegate to the
Hungarian solver, which uses spatial/shape continuity instead of YOLO IDs.

```
Frame N:  mask_A → slot_0 (green),  mask_B → slot_1 (red)     [via track IDs]
Frame N+1 (crossing): YOLO swaps IDs
  Straight cost:  mask_A→slot_0 = 0.72,  mask_B→slot_1 = 0.68  → total 1.40
  Swapped cost:   mask_A→slot_1 = 0.25,  mask_B→slot_0 = 0.22  → total 0.47  ← CHEAPER
  → Swap guard triggers, Hungarian assigns mask_A→slot_1, mask_B→slot_0
  → Colors remain stable ✓
```

**When it triggers:** Only when exactly 2 masks both have known track_id→slot
mappings. For fewer masks (0 or 1), there's nothing to swap-check. For >2
masks, a generalized permutation check could be added but isn't needed for
our 2-rat setup.

**Logging:** When the swap guard fires, it logs at INFO level:
```
Swap guard triggered: straight_cost=1.400 > swapped_cost=0.470 (tids [3, 5]→slots [0, 1]). Delegating to Hungarian.
```

---

## 7. Known Tracking Failure Modes

These are the failure modes observed in lab rat videos and how each layer
addresses them:

| Failure Mode | BoT-SORT Mitigation | SlotTracker Mitigation |
|-------------|--------------------|-----------------------|
| **ID swap during crossing** | `new_track_thresh: 0.3` reduces new-ID creation | Swap guard detects and corrects |
| **Duplicate/overlapping boxes** | Default NMS (now `iou: 0.5`) | `filter_masks` keeps top-N by area |
| **Lost track at walls** | `track_buffer: 60` survives 2s occlusion | `max_missing_frames: 5` holds slot |
| **New ID created mid-video** | Higher `new_track_thresh` suppresses | `_assign_to_slot` remaps new tid→existing slot |
| **Box size explosion during contact** | Two-stage matching recovers low-conf detections | `HARD_AREA_RATIO_VETO = 5.0` rejects extreme changes |
| **Keypoint confusion** | Not addressed (upstream YOLO issue) | Not addressed (keypoints pass through) |

### Remaining limitations

- **Keypoint confusion:** When rats overlap, YOLO may assign keypoints from
  rat A to rat B's bounding box. This is a YOLO pose model limitation that
  no amount of tracking tuning can fix. A retrained YOLO model with more
  occlusion examples would help.
- **Extended occlusion (>2s):** If one rat hides behind the other for more
  than `track_buffer` frames (60 = 2.4s at 25fps), the track is deleted and
  a new ID is created on reappearance. The SlotTracker's free-slot assignment
  may or may not recover the correct color.
