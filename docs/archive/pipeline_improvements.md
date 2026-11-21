# Pipeline Architecture

This document describes the pipeline architecture, what each component does,
why it reduces ID switching / color flicker, and how to tune parameters.

---

## Pipeline Flow

```
Frame (RGB)
  │
  ├─ [Optional] Mirror-pad borders (yolo_border_padding_px)
  │
  ▼
YOLO model.track() (BoT-SORT / ByteTrack)
  │  → boxes + keypoints + class names + TRACK IDs
  │  → filter by class (filter_class)
  │  → correct coordinates if padded
  │  → [Optional] custom NMS IoU (nms_iou)
  │  BoT-SORT provides: Kalman prediction, motion compensation,
  │  track buffer (survives lost detections), optional ReID
  │
  ▼
Post-Detection Filters (infer_yolo.py)
  │  → [Optional] Edge margin: reject centroids within N px of border
  │
  ▼
SAM2 Segmentation (per box, on original unpadded frame)
  │  → predictor.set_image(frame)
  │  → predict(box=..., multimask_output=False)
  │  → mask = masks[0] > sam_threshold
  │
  ▼
Smart Duplicate Filter (only if num_masks > max_animals)
  │  → sort by area (largest first)
  │  → keep if IoU with all kept < mask_iou_threshold
  │
  ▼
Slot Tracker (Hungarian assignment + soft costs)
  │  → N fixed color slots (slot 0 = green, slot 1 = red, ...)
  │
  │  Step 1: Match by YOLO track ID (O(1) lookup)
  │    If track ID already assigned to a slot → instant match
  │
  │  Step 2: Hungarian assignment for unmatched masks ↔ slots
  │    Cost matrix = w_dist × centroid_dist + w_iou × (1-mask_iou) + w_area × area_change
  │    scipy.optimize.linear_sum_assignment → globally optimal
  │    Area change is SOFT cost (not hard veto)
  │    Only veto if area ratio > 5x (truly pathological)
  │
  │  Step 3: Unmatched masks → first free slot
  │  Step 4: Unmatched slots → missing_count++
  │    Release slot after max_missing_frames consecutive misses
  │
  ▼
Render Overlay (persistent buffer)
  │  → colored masks (by slot index)
  │  → bounding boxes + labels + confidence
  │  → keypoints (7-point rat pose)
  │  → centroids
  │
  ▼
Output Video (.avi)
```

---

## Component Details

### 1. YOLO Detection + Tracking (`infer_yolo.py`)

Uses `model.track()` with BoT-SORT for stable track IDs across frames.

**Detection flow:**
1. Optional mirror-padding (`yolo_border_padding_px`) to help detect rats at frame edges
2. `model.track(frame, conf=confidence, persist=True, tracker=tracker_config)` — BoT-SORT maintains Kalman-filter state across frames
3. Optional custom NMS IoU (`nms_iou`) — tighter NMS suppresses merged boxes when animals touch
4. Parse results into `Detection` objects with track IDs, boxes, keypoints, class names
5. Coordinate correction if frame was padded
6. Optional edge margin filter — reject detections whose centroid is within N pixels of frame border

**Available YOLO-level filters:**

| Filter | Config key | Default | Purpose |
|--------|-----------|---------|---------|
| Confidence | `detection.confidence` | 0.25 | Min detection score |
| Class filter | `detection.filter_class` | null | Only keep this class |
| Border padding | `detection.yolo_border_padding_px` | 0 | Mirror-pad for edge detections |
| Edge margin | `detection.edge_margin` | 0 | Reject centroids near frame border |
| Custom NMS | `detection.nms_iou` | null | Tighter NMS IoU (null = YOLO default ~0.7) |

**Where:** [infer_yolo.py](../src/pipelines/sam2_yolo/infer_yolo.py) — `detect_and_track()`.

---

### 2. SAM2 Segmentation (`infer_sam2.py`)

Uses YOLO bounding boxes as prompts for SAM2 mask prediction.

- One SAM2 call per detection (box prompt, `multimask_output=False`)
- Operates on the original unpadded frame
- Mask threshold applied to raw logits: `mask = raw_masks[0] > sam_threshold`

**Where:** [infer_sam2.py](../src/pipelines/sam2_yolo/infer_sam2.py) — `segment_from_boxes()`.

---

### 3. Post-processing + Tracking (`postprocess.py` + `tracking.py`)

**Mask deduplication:** When more masks than `max_animals` exist, keeps the largest non-overlapping masks (IoU-based filtering).

**SlotTracker** assigns masks to fixed color slots:
- **Stage 1:** Match by YOLO track ID (O(1) lookup from `_track_to_slot` dict)
- **Stage 2:** Hungarian optimal assignment for remaining masks using soft cost matrix
- **Stage 3:** Unmatched masks go to free slots
- **Stage 4:** Unmatched slots increment missing counter, released after `max_missing_frames`

**Where:** [postprocess.py](../src/pipelines/sam2_yolo/postprocess.py), [tracking.py](../src/common/tracking.py).

---

## Key Design Decisions and Why

### Why `model.track()` instead of `model()`

`model.track()` with BoT-SORT provides:
- **Kalman prediction:** Predicts where each object will be next frame. Even if YOLO misses for 1-2 frames, the track stays alive (`track_buffer` in tracker YAML, default=30).
- **Motion-aware assignment:** Uses motion prediction, not just nearest-neighbor, so more robust to crossing paths.
- **Global Motion Compensation:** `sparseOptFlow` handles camera vibration.
- **Stable track IDs:** Primary identity signal for the SlotTracker.

### Why Hungarian assignment instead of greedy matching

Greedy (slot 0 picks first, slot 1 gets leftovers) causes cascading errors when two objects are close. Hungarian compares both possible pairings:
- `cost(mask_A→slot_0) + cost(mask_B→slot_1)` vs
- `cost(mask_A→slot_1) + cost(mask_B→slot_0)`

and picks whichever is cheaper overall.

### Why soft area cost instead of hard area veto

Hard area veto (reject if area changes >40%) drops valid matches near borders where boxes are clipped. Soft cost (weight `w_area=0.2`) makes large area changes more expensive but not impossible. Only truly pathological cases (area ratio >5x) are vetoed.

### Why edge margin exists but area filter was removed

**Edge margin** (reject centroids near frame border) is a simple, reliable filter that catches flickering partial detections at walls without dropping real objects.

**Adaptive area filter** was implemented and tested but **removed** because it consistently dropped valid wall-adjacent detections — clipped boxes near walls are legitimately smaller, triggering the area filter. The tracker's soft area cost handles area changes more gracefully than a hard YOLO-level filter.

---

## BoT-SORT Tracker Configuration

The Ultralytics BoT-SORT tracker has its own config. Default values work
well for most cases. Key parameters you might tune:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `track_buffer` | 30 | Frames to keep a lost track alive. Raise to 60-120 for slow animals. |
| `track_high_thresh` | 0.25 | Min confidence for first-pass matching. |
| `track_low_thresh` | 0.1 | Min confidence for second-pass (low-conf) matching. |
| `match_thresh` | 0.8 | Max cost for IoU-based association. |
| `with_reid` | false | Enable appearance ReID. Adds compute but helps after occlusion. |
| `gmc_method` | sparseOptFlow | Global Motion Compensation. |

To customize, create a YAML file (e.g., `configs/trackers/custom_botsort.yaml`):

```yaml
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 60          # raised for slow-moving rats
match_thresh: 0.8
fuse_score: true
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: false
```

Then reference it in your config:
```yaml
tracking:
  tracker_config: configs/trackers/custom_botsort.yaml
```

Or via CLI: `tracking.tracker_config=configs/trackers/custom_botsort.yaml`

---

## Parameter Recommendations

### Default settings (start here)

```yaml
detection:
  confidence: 0.25
  filter_class: null
  yolo_border_padding_px: 0
  edge_margin: 0
  nms_iou: null

tracking:
  tracker_config: botsort.yaml
  max_centroid_distance: 150.0
  max_missing_frames: 5
  w_dist: 0.4
  w_iou: 0.4
  w_area: 0.2
  cost_threshold: 0.85
```

### If you see border flicker

```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    detection.yolo_border_padding_px=64 \
    detection.edge_margin=10 \
    detection.confidence=0.20 \
    tracking.max_missing_frames=10
```

### If you see ID swaps during close contact

```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    tracking.w_iou=0.5 \
    tracking.w_dist=0.3 \
    tracking.w_area=0.2 \
    tracking.max_centroid_distance=100 \
    detection.nms_iou=0.5
```

This prioritizes mask shape (IoU) over centroid distance, which helps when
centroids are close but mask shapes are distinguishable. Lower NMS catches
merged boxes.

### If detections are noisy

```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    detection.confidence=0.40 \
    detection.filter_class=ratas
```

---

## Debug Signals

Every 50 frames, the log shows slot status:

```
Processed 100 frames | slot0:tid=1(miss=0) | slot1:tid=2(miss=0)
```

| Signal | Meaning | Action |
|--------|---------|--------|
| `tid=1(miss=0)` | Track 1 assigned to slot 0, matched this frame | Stable |
| `tid=2(miss=3)` | Track 2 not matched for 3 frames | Check if near border |
| `slot1:free` | Slot released after max_missing exceeded | Rat left frame or tracking lost |
| `tid` changes (e.g., tid=1 → tid=3) | BoT-SORT created a new track | Check `track_buffer` in tracker config |

---

## Known Risks & Limitations

1. **BoT-SORT ID switches during prolonged occlusion:** If two rats overlap
   for many frames and then separate, BoT-SORT may assign new track IDs.
   The slot tracker handles this via Hungarian fallback, but colors may swap.
   **Mitigation:** Raise `track_buffer` in BoT-SORT config to 60-120.

2. **Border padding + large values:** Padding >128px may create "ghost"
   reflected detections. Stick to 32-64px.

3. **Soft area cost with extreme partial views:** If a rat is 90% off-screen,
   the area change is huge. The soft cost will be high but the 5x hard veto
   threshold will still allow it. Combined with mask IoU cost, the assignment
   should still be correct.

4. **Cost weight tuning:** The defaults (0.4/0.4/0.2) work for lab videos.
   For very different scenarios (different resolution, speed, occlusion patterns),
   you may need to re-tune. Use the CLI overrides for quick experiments.

5. **No cross-gap re-identification:** If a slot is released and re-assigned,
   the new occupant gets the same color. There's no appearance-based re-ID
   across long gaps (could enable `with_reid: true` in BoT-SORT but this
   adds compute and may not help for visually identical rats).

---

## Future Improvements

- **Frame skip / stride:** Process every Nth frame for speed.
- **Temporal smoothing:** EMA on box/keypoint coordinates.
- **SAM2 VideoPredictor:** Use temporal mask propagation for more stable
  mask tracking (major architectural change).
- **SAM2 point prompts:** Use previous centroid as point prompt when box prompt
  fails (e.g., during close contact). Requires tracker → segmentation feedback loop.
- **YOLO 26:** Upgrade from YOLOv8 to YOLO26 for better detections.
  YOLO26 supports NMS-free end-to-end inference.
