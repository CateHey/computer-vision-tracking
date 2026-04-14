# Reference Codebase Comparison

Comparison of our SAM2+YOLO pipeline against the reference `yolov8_sam2_toolkit`
documented in `reference/`. Covers what we adopted, what we tested and removed,
and what we chose not to adopt.

---

## 1. Architecture Differences

| Aspect | Our pipeline | Reference toolkit |
|--------|-------------|-------------------|
| Pipeline pattern | Sequential function calls in a loop | Data Bus: shared `frame_data` dict through processor chain |
| YOLO inference | `model.track()` with BoT-SORT (temporal state) | `model()` per-frame (stateless detection) |
| Identity source | YOLO track IDs (primary) + Hungarian assignment (fallback) | SAM2-side `IdentityMatcher` with greedy distance + area gating |
| Tracking location | `SlotTracker` in `src/common/tracking.py` | `IdentityMatcher` in `tracking/identity_matcher.py` |
| Mask dedup | IoU-based, sort by area, keep top N | Same algorithm |
| YOLO filtering | Class filter + border padding + edge margin + custom NMS | 7-step chain: confidence, class, area, edge margin, ROI, NMS, entity count |
| Config system | YAML + CLI overrides | Constructor parameters (code-level) |

---

## 2. Techniques — Status After Evaluation

### 2.1 Adopted: Edge margin filter

**Reference:** `YOLOProcessor(edge_margin=20)` — rejects detections near frame border.

**Our implementation:** `detection.edge_margin` config param → `_filter_edge_margin()` in `infer_yolo.py`. Rejects detections whose centroid is within N pixels of the frame border. Applied after coordinate correction (compatible with border padding).

**Status:** Implemented, disabled by default (`edge_margin: 0`). Set to 10-15 for lab videos with wall issues.

---

### 2.2 Adopted: Custom NMS IoU threshold

**Reference:** `YOLOProcessor(max_overlap=0.5)` — post-YOLO NMS with configurable IoU.

**Our implementation:** `detection.nms_iou` config param → passed directly to `model.track(iou=nms_iou)`. Ultralytics handles the NMS internally with the custom threshold.

**Status:** Implemented, disabled by default (`nms_iou: null` = YOLO default ~0.7). Use 0.4-0.5 when animals touch and produce merged boxes.

---

### 2.3 Tested and Removed: Adaptive area filtering

**Reference:** `YOLOProcessor(area=None, area_error=0.2)` — learns reference area from first frame, rejects boxes deviating >20%.

**Our implementation:** Was implemented as `_filter_by_area()` with EMA-based reference learning and configurable tolerance. Tested in production.

**Why removed:** Consistently dropped valid wall-adjacent detections. Rats near walls produce legitimately smaller (clipped) bounding boxes that triggered the area filter. The tracker's soft area cost (`w_area=0.2`) handles area changes more gracefully without dropping detections.

**Status:** Removed from code entirely. Not in config, not in `infer_yolo.py`.

---

### 2.4 Tested and Removed: Max entity cap at YOLO level

**Reference:** `YOLOProcessor(max_entities=2)` — keep top N by confidence after filtering.

**Our implementation:** Was implemented as `_cap_max_detections()`. Tested with `max_detections=max_animals`.

**Why removed:** Redundant with our mask deduplication in `postprocess.py` which already caps maspruebasratslabs-c02t9/3ks to `max_animals` using IoU-based filtering. The YOLO-level cap was too aggressive — it could drop a valid detection before SAM2 even ran.

**Status:** Removed from code entirely.

---

### 2.5 Not Adopted: Min-entities confidence retry

**Reference:** `YOLOProcessor(min_entities=2)` — if fewer than N detections, recursively lowers confidence by 5% and retries.

**Why not adopted:** BoT-SORT's `track_low_thresh=0.1` already performs second-pass matching for low-confidence detections, achieving a similar effect without re-running the model. Lowering `detection.confidence` to 0.15-0.20 is the simpler equivalent.

---

### 2.6 Not Adopted: ROI filter

**Reference:** `YOLOProcessor(roi=[100, 100, 800, 600])` — only keep detections within a rectangle.

**Why not adopted:** Our current videos use the full frame as the arena. Low priority but easy to add if needed (~10 lines).

---

### 2.7 Not Adopted: SAM2 point prompts / centroid tracking

**Reference:** `SAM2Processor` supports manual point prompts and centroid-based tracking between frames (previous mask centroid as point prompt for next frame).

**Why not adopted:** Our pipeline uses YOLO boxes as SAM2 prompts, which works well when YOLO produces clean detections. Point prompts would require the tracker to feed centroid predictions back to segmentation — a significant architectural change. Deferred as future improvement.

---

### 2.8 Not Adopted: Reference tracking algorithm

**Reference:** `IdentityMatcher` — greedy per-slot matching with hard distance + area gates.

**Why not adopted:** Our `SlotTracker` with Hungarian optimal assignment + soft costs + YOLO track ID mapping is strictly superior:

| Feature | Reference | Ours |
|---------|-----------|------|
| Matching | Greedy (sequential) | Hungarian (globally optimal) |
| Area handling | Hard veto (±40%) | Soft cost (weight 0.2), hard veto only at 5x |
| Identity signal | Centroid distance only | YOLO track ID (primary) + centroid+IoU+area (fallback) |
| IoU in matching | Not used | Mask IoU (weight 0.4) |
| Missing tolerance | None | `max_missing_frames` with counter |

---

### 2.9 Not Adopted: Data Bus architecture

**Reference:** `frame_data` dictionary passed through a processor chain with `validate()` contracts.

**Why not adopted:** Would require rewriting the entire pipeline for no functional benefit. Our direct function-call approach is simpler and equally effective for the current use case.

---

### 2.10 Not Adopted: Per-frame `model()` instead of `model.track()`

**Reference:** Uses stateless `model()` per frame — no temporal state in YOLO.

**Why not adopted:** `model.track()` with BoT-SORT provides Kalman prediction, motion compensation, stable track IDs, and track buffer. This is a major advantage over stateless detection.

---

## 3. Summary

| Technique | Origin | Status | Reason |
|-----------|--------|--------|--------|
| Edge margin | Reference | **Adopted** | Reliable wall-artifact suppression |
| Custom NMS IoU | Reference | **Adopted** | Catches merged boxes during contact |
| Adaptive area filter | Reference | **Removed** | Dropped valid wall-adjacent detections |
| Max entity cap | Reference | **Removed** | Redundant with mask dedup |
| Min-entities retry | Reference | Not adopted | BoT-SORT low_thresh achieves same effect |
| ROI filter | Reference | Not adopted | Not needed for current videos |
| SAM2 point prompts | Reference | Not adopted | Architectural change, deferred |
| IdentityMatcher | Reference | Not adopted | Our Hungarian tracking is superior |
| Data Bus pattern | Reference | Not adopted | No functional benefit for refactor cost |
| Stateless detection | Reference | Not adopted | `model.track()` is strictly better |

**Where we're stronger:** Tracking (Hungarian + BoT-SORT vs greedy), identity stability (track IDs + missing tolerance), border handling (padding + edge margin combined).

**Where reference is stronger:** SAM2 flexibility (point prompts, centroid fallback), richer class filtering (include/exclude lists), entity count enforcement.

---

## 4. Parameter Mapping

| Reference parameter | Our equivalent | Notes |
|---------------------|---------------|-------|
| `confidence` | `detection.confidence` | Same |
| `entities` / `max_entities` | `detection.max_animals` | Enforced at mask level, not YOLO level |
| `min_entities` | (not implemented) | BoT-SORT low_thresh is the equivalent |
| `area` / `area_error` | (removed) | Tested, caused wall-detection drops |
| `classes` | `detection.filter_class` | Theirs: class ID list, ours: single class name |
| `edge_margin` | `detection.edge_margin` | Same concept, same implementation |
| `max_overlap` | `detection.nms_iou` | Theirs: post-filter NMS, ours: YOLO-internal NMS |
| `roi` | (not implemented) | Low priority |
| `proximity_threshold` | `tracking.max_centroid_distance` | Theirs: hard gate, ours: normalization constant |
| `area_tolerance` | `tracking.w_area` | Theirs: hard ±40% gate, ours: soft cost weight |
| `iou_threshold` | `segmentation.mask_iou_threshold` | Same — mask dedup IoU |
