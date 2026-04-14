# Social Contact Classification — Design

## 1. Context

Our SAM2+YOLO pipeline already produces per-frame:
- **Detections** with bounding boxes, confidence, track_id (stable via BoT-SORT)
- **7 keypoints** per rat: `tail_tip`(0), `tail_base`(1), `tail_start`(2), `mid_body`(3), `nose`(4), `right_ear`(5), `left_ear`(6)
- **SAM2 instance masks** (boolean arrays per slot)
- **Slot centroids** with stable identity across frames

We also have `evaluate_closeness()` in `src/common/metrics.py` that classifies
frames as close/not-close using centroid distance + bbox IoU, and
`scripts/extract_frames.py` that groups close frames into encounter events.

This module extends that foundation to classify **what type** of social contact
is occurring, using the standard ethological categories from rodent behavioral
neuroscience.

---

## 2. Contact Types

Six contact types plus NC (No Contact), ordered by classification priority:

### 2.1 Nose-to-nose (N2N)

| | |
|---|---|
| **Behavior** | Snouts directed at each other. Social recognition / greeting. |
| **Geometry** | `dist(nose_A, nose_B) < contact_radius` |
| **Keypoints used** | `nose` (index 4) of both rats |
| **Confidence** | **High** — nose is the most reliably detected keypoint |
| **Fallback** | If one nose is missing: check `dist(nose_A, head_centroid_B) < contact_radius` where head centroid = midpoint of `(right_ear + left_ear)` |

**Detection rule:**
```
nose_A = det_A.keypoints[4]  # (x, y, conf)
nose_B = det_B.keypoints[4]
if nose_A.conf >= min_kp_conf and nose_B.conf >= min_kp_conf:
    d = euclidean(nose_A, nose_B)
    if d < contact_radius:
        → NOSE_TO_NOSE
```

### 2.2 Nose-to-anogenital (N2AG)

| | |
|---|---|
| **Behavior** | Snout of rat A near tail base of rat B. Primary individual recognition via pheromones. **The single most important metric for social phenotyping.** |
| **Geometry** | `dist(nose_A, tail_base_B) < contact_radius` OR `dist(nose_B, tail_base_A) < contact_radius` |
| **Keypoints used** | `nose` (4) of one rat, `tail_base` (1) of the other |
| **Confidence** | **High** — both keypoints are well-detected in our model |
| **Asymmetric** | Records which rat is the investigator (nose owner) and which is the target (tail_base owner) |

**Detection rule:**
```
nose_A, tail_base_B = det_A.keypoints[4], det_B.keypoints[1]
nose_B, tail_base_A = det_B.keypoints[4], det_A.keypoints[1]

if nose_A.conf >= min_kp_conf and tail_base_B.conf >= min_kp_conf:
    if dist(nose_A, tail_base_B) < contact_radius:
        → NOSE_TO_ANOGENITAL (investigator=A, target=B)

if nose_B.conf >= min_kp_conf and tail_base_A.conf >= min_kp_conf:
    if dist(nose_B, tail_base_A) < contact_radius:
        → NOSE_TO_ANOGENITAL (investigator=B, target=A)
```

### 2.3 Nose-to-body (N2B)

| | |
|---|---|
| **Behavior** | Snout of rat A near trunk/flank of rat B. General social investigation. |
| **Geometry** | `nose_A` is within `contact_radius` of rat B's body but NOT near nose or tail_base of B |
| **Keypoints used** | `nose` (4) of one rat, `mid_body` (3) of the other |
| **Confidence** | **Medium** — mid_body is reliable but the "body zone" is broad |
| **Catch-all** | This is the fallback when nose-A is near rat-B but doesn't qualify as N2N or N2AG |

**Detection rule:**
```
# Only check N2B if N2N and N2AG were NOT triggered for this pair
nose_A, mid_body_B = det_A.keypoints[4], det_B.keypoints[3]

# Option 1: keypoint distance
if dist(nose_A, mid_body_B) < contact_radius * 1.5:
    → NOSE_TO_BODY (investigator=A, target=B)

# Option 2: mask containment (more accurate)
if mask_B[int(nose_A.y), int(nose_A.x)] == True:
    → NOSE_TO_BODY (investigator=A, target=B)
```

**Mask-based check is preferred** when SAM2 masks are available: if the nose
keypoint of rat A falls inside the mask of rat B, that's definitive body contact.

### 2.4 Tail-to-tail (T2T)

| | |
|---|---|
| **Behavior** | Both tail bases close together. Rear-to-rear contact. |
| **Geometry** | `dist(tail_base_A, tail_base_B) < contact_radius` |
| **Keypoints used** | `tail_base` (index 1) of both rats |
| **Confidence** | **High** — tail_base is reliably detected |
| **Symmetric** | No investigator assigned |

**Detection rule:**
```
tail_base_A = det_A.keypoints[1]
tail_base_B = det_B.keypoints[1]
if tail_base_A.conf >= min_kp_conf and tail_base_B.conf >= min_kp_conf:
    d = euclidean(tail_base_A, tail_base_B)
    if d < contact_radius:
        → TAIL_TO_TAIL
```

### 2.5 Side-by-side (SBS)

| | |
|---|---|
| **Behavior** | Bodies parallel, in contact, low velocity. Affiliative resting / huddling. |
| **Geometry** | `mask_iou(A, B) > sbs_iou_threshold` AND both rats have low velocity |
| **Keypoints used** | Centroids (from masks), body orientation from `nose` → `tail_base` vector |
| **Confidence** | **Medium** — relies on mask quality and velocity estimation |

**Detection rule:**
```
iou = mask_iou(mask_A, mask_B)
vel_A = dist(centroid_A_now, centroid_A_prev) / dt
vel_B = dist(centroid_B_now, centroid_B_prev) / dt

# Body orientation vectors
orient_A = vector(tail_base_A → nose_A)
orient_B = vector(tail_base_B → nose_B)
cos_angle = dot(orient_A, orient_B) / (|orient_A| * |orient_B|)

if iou > sbs_iou_threshold:
    if vel_A < low_velocity_threshold and vel_B < low_velocity_threshold:
        if abs(cos_angle) > parallel_threshold:  # same or opposite direction
            → SIDE_BY_SIDE
```

### 2.6 Following (FOL)

| | |
|---|---|
| **Behavior** | Rat A moves in same direction as rat B at close range. Social pursuit. |
| **Geometry** | `dist(nose_A, tail_base_B) < follow_radius` AND both moving AND velocity vectors aligned |
| **Keypoints used** | `nose` (4), `tail_base` (1), centroids across frames |
| **Confidence** | **Low-Medium** — requires multi-frame velocity estimation, sensitive to tracking jitter |
| **Temporal** | Must persist for `min_follow_frames` to count |

**Detection rule:**
```
# Rat A's nose is near rat B's tail (A is behind B)
if dist(nose_A, tail_base_B) < follow_radius:
    vel_A = centroid_A_now - centroid_A_prev
    vel_B = centroid_B_now - centroid_B_prev
    speed_A = |vel_A|
    speed_B = |vel_B|

    if speed_A > min_speed and speed_B > min_speed:
        cos_angle = dot(vel_A, vel_B) / (|vel_A| * |vel_B|)
        if cos_angle > follow_alignment:  # both moving ~same direction
            → FOLLOWING (follower=A, leader=B)
```

---

## 3. Proximity Zones

All distances are normalized by **estimated body length** for scale-invariance.

### Body length estimation

```
body_length = dist(nose, tail_base)   # keypoints 4 → 1
```

If keypoints are unavailable, fallback:
```
body_length = bbox_diagonal * 0.6     # empirical ratio
```

The body length is smoothed per rat with an EMA across frames to avoid
jitter from keypoint noise.

### Zone definitions

| Zone | Distance (body-lengths) | Meaning |
|------|------------------------|---------|
| **Contact** | < 0.3 BL | Physical contact — keypoints overlap or near-touching |
| **Proximity** | 0.3 - 1.0 BL | Close approach — within reach but not touching |
| **Independent** | > 1.0 BL | No social interaction |

The `contact_radius` used in Section 2 corresponds to the **Contact zone**
(< 0.3 body-lengths). In absolute pixels (640x480 video, typical rat BL ~120px):

| Zone | Approx. pixels |
|------|---------------|
| Contact | < 36 px |
| Proximity | 36 - 120 px |
| Independent | > 120 px |

These are configurable via YAML:

```yaml
contacts:
  # Proximity zones (in body-lengths)
  contact_zone_bl: 0.3       # Contact: < 0.3 BL
  proximity_zone_bl: 1.0     # Proximity: 0.3 - 1.0 BL
  # Fallback when body length can't be estimated from keypoints
  fallback_body_length_px: 120
```

---

## 4. Priority and Mutual Exclusion

Per pair per frame, contacts are evaluated in priority order. **Higher-priority
contacts suppress lower-priority ones** to avoid double-counting:

```
1. NOSE_TO_NOSE        (most specific — both noses within contact radius)
2. NOSE_TO_ANOGENITAL  (specific — nose near tail_base, picks closer direction)
3. TAIL_TO_TAIL        (specific — both tail bases within contact radius)
4. FOLLOWING            (behavioral — requires motion + spatial pattern)
5. SIDE_BY_SIDE         (postural — requires mask overlap + low velocity + parallel)
6. NOSE_TO_BODY        (catch-all — nose inside mask or near mid_body)
```

**N2B is last** because its mask-containment check (nose inside the other rat's
SAM2 mask) is very broad. Placing it before FOL/SBS/T2T would absorb those more
specific contact types, since a nose is almost always inside the other rat's mask
during any close interaction.

**N2B has guards** to prevent absorbing near-miss N2N/N2AG contacts: if the nose
is within 1.5× contact_radius of the other rat's nose or tail_base, N2B skips
(those regions are N2N/N2AG territory).

**N2AG picks the closer direction** when both A→B and B→A qualify simultaneously,
eliminating systematic A-first bias in investigator statistics.

**No contact:** If none of the above trigger, the pair is classified by zone:
`PROXIMITY` (within proximity zone) or `INDEPENDENT`.

---

## 5. Integration with Existing Pipeline

### 5.1 Where it fits

```
                     EXISTING PIPELINE
                     ─────────────────
  YOLO detect_and_track() → detections (with keypoints, track_ids)
  SAM2 segment_from_boxes() → masks
  postprocess_frame() → slot_masks, slot_centroids
                     │
                     │  NEW MODULE
                     ▼  ───────────
  classify_contacts() → per-frame contact events
  ContactTracker.update() → accumulate bouts, compute velocities
                     │
                     ▼
  Output: CSV + JSON + PDF report
```

### 5.2 Relationship to existing code

| Existing | Contact module usage |
|----------|---------------------|
| `evaluate_closeness()` | **Kept as fast pre-filter.** Only run contact classification on frames where `is_close=True`. Saves compute on the 80%+ of frames where rats are apart. |
| `extract_frames.py` encounters | **Extended.** Encounters are enriched with contact type breakdown. The encounter grouping logic stays the same. |
| `SlotTracker` | **Read-only dependency.** Contact module reads slot states (centroids, masks, track_ids) but doesn't modify them. |
| `Detection.keypoints` | **Primary input.** All geometric rules use the 7 keypoints from YOLO. |
| `mask_iou()` | **Reused.** For mask overlap detection (SBS, N2B fallback). |
| `compute_centroid()` | **Reused.** For velocity estimation. |
| Config YAML | **New section `contacts:`** added alongside existing `closeness:` and `encounters:`. |

### 5.3 New module location

```
src/common/contacts.py        # Contact classification logic
    classify_frame_contacts()  # Per-frame: detections + masks → contact events
    ContactTracker             # Multi-frame: accumulate bouts, velocities

scripts/analyze_contacts.py   # CLI: run contact analysis on pipeline output
    OR integrated into run.py  # Option: real-time contact logging during pipeline
```

### 5.4 Integration options

**Option A — Post-hoc analysis script (recommended for v1):**
Run contact analysis on the JSONL output of `extract_frames.py scan`, adding
keypoint data to the scan output. Minimal changes to existing pipeline.

**Option B — Inline in run.py:**
Add `classify_frame_contacts()` call after `postprocess_frame()` in the main
loop. Produces real-time contact CSV alongside the overlay video. Requires
passing masks + detections to the contact classifier.

Both options use the same `src/common/contacts.py` module.

---

## 6. Edge Cases

### 6.1 Missing keypoints

| Scenario | Handling |
|----------|---------|
| One nose missing (conf < min_kp_conf) | Skip N2N. N2AG/N2B only check the rat with a valid nose. |
| Both noses missing | No keypoint-based contacts. Fall back to mask_iou for SBS. |
| tail_base missing | Skip N2AG for that pair direction. Use `tail_start` (index 2) as fallback with wider radius. |
| All keypoints missing | Zone classification only (centroid distance → PROXIMITY/INDEPENDENT). |

### 6.2 Mask overlap anomalies

| Scenario | Flag |
|----------|------|
| `mask_iou > 0.5` | **Tracking error flag.** Two rats can't physically overlap >50%. Likely: identity swap, merged detection, or SAM2 error. Log as `quality_flag: "high_mask_overlap"`. |
| `mask_iou > 0` but `< 0.1` | Border contact. Valid for SBS / physical contact. |
| `mask_iou > 0.1` but `< 0.5` | Significant overlap. Likely close huddling or partial tracking error. |

### 6.3 Partial visibility (wall occlusion)

When a rat is near the frame border:
- Bounding box is clipped → bbox area is artificially small
- Some keypoints may be outside the frame (conf drops)
- **Rule:** Require at least 2 valid keypoints per rat to classify contacts.
  Otherwise, fall back to centroid distance only.

### 6.4 Rapid ID switches

If YOLO track_id changes between frames for the same physical rat:
- The SlotTracker already handles this (Hungarian assignment maintains slot identity)
- Contact bouts use **slot index** (not track_id) as the identity key
- A bout is not broken by a track_id switch if the slot remains the same

### 6.5 Single rat detected

When only 1 of 2 rats is detected:
- No pair exists → no contacts possible
- Frame is recorded as `zone: INDEPENDENT` with note `"single_detection"`

---

## 7. Parameters

All configurable via the `contacts:` YAML section:

```yaml
contacts:
  # Enable/disable contact classification in the pipeline
  enabled: false

  # Minimum keypoint confidence to use a keypoint for contact detection
  min_keypoint_conf: 0.3

  # Proximity zones (in body-lengths, BL)
  contact_zone_bl: 0.3         # < 0.3 BL = contact
  proximity_zone_bl: 1.0       # 0.3 - 1.0 BL = proximity
  fallback_body_length_px: 120 # when keypoints can't estimate BL

  # Side-by-side detection
  sbs_mask_iou_min: 0.02       # minimum mask overlap for SBS
  sbs_max_velocity_bl: 0.04    # max centroid displacement per frame (in BL)
  sbs_parallel_cos_min: 0.7    # min |cos(angle)| between body orientations

  # Following detection
  follow_radius_bl: 0.5        # nose-to-tail_base distance (in BL)
  follow_min_speed_bl: 0.025   # min centroid displacement per frame (in BL)
  follow_alignment_cos: 0.7    # min cos(angle) between velocity vectors
  follow_min_frames: 30        # min frames to count as following (~1.0s at 30fps)

  # Bout grouping
  bout_max_gap_frames: 3       # max frames gap within a bout
  bout_min_duration_frames: 9  # min frames for a bout (~300ms at 30fps)

  # Quality flags
  mask_overlap_warning: 0.5    # mask_iou above this = tracking error flag
```

---

## 8. Data Flow Summary

```
Per frame:
  detections[]       ← YOLO (with keypoints + track_ids)
  slot_masks[]       ← SAM2 + SlotTracker
  slot_centroids[]   ← SlotTracker

  evaluate_closeness() → is_close?
    │
    ├─ False → zone=INDEPENDENT, skip classification
    │
    └─ True → classify_frame_contacts():
               1. Estimate body lengths from keypoints
               2. Compute pairwise keypoint distances
               3. Check contact types in priority order:
                  N2N → N2AG → N2B → FOL → SBS
               4. Check mask overlap (SBS, N2B fallback, quality flags)
               5. Return list of ContactEvent per pair

  ContactTracker.update(events):
    - Accumulate into bouts (group consecutive same-type events)
    - Update velocity buffer (for FOL, SBS)
    - Write per-frame row to CSV
    - Flag quality issues

End of video:
  ContactTracker.finalize():
    - Close open bouts
    - Write bout CSV
    - Write session JSON summary
    - Generate PDF report (ethogram, histograms, pie chart)
```
