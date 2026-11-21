# YOLO Limitations for Multi-Animal Pose Tracking

Research notes on why YOLO underperforms in our rat tracking scenario and what
alternatives exist.

---

## 1. Our Problem

YOLO26 in detect-only mode is intermittent and unstable when tracking 2 identical
rats in a cage. Running `debug_yolo_only.py` on 120s of video showed:

- Detection count fluctuates between 0, 1, and 2 unpredictably
- Detections appear and disappear every second
- Box identity swaps frame-to-frame (box 0 ↔ box 1)
- Overlapping/merged boxes during close interactions
- Even when rats are clearly separated, YOLO sometimes detects only 1

This instability cascades through the entire reference pipeline — SAM2 gets bad
prompts, IdentityMatcher gets unstable inputs, masks bleed/flicker/swap.

---

## 2. Why YOLO Struggles Here

### 2.1 No temporal modeling

YOLO is a **pure single-frame detector**. It has no concept of what it detected
in the previous frame. Every frame is an independent prediction from scratch.

This means:
- Small appearance changes (rat turns slightly) can cause a detection to appear
  or disappear
- There is no smoothing — confidence can jump from 0.9 to 0.1 between frames
- Keypoint positions can jump or drift with no continuity
- Detection ordering (which rat is "box 0") is arbitrary each frame

Recent "temporal YOLO" research adds spatiotemporal modules specifically because
vanilla YOLO has keypoint drift and temporal instability. Standard YOLO was not
designed to maintain identity or consistency across frames.

### 2.2 Identical appearance

Both rats are the same species, color, and size. YOLO relies on appearance
features to distinguish detections — when two objects look identical, the model
has no reliable signal to decide which is which.

This is fundamentally different from typical YOLO demos (cars, people, potatoes)
where objects have visible differences (color, size, shape, clothing).

### 2.3 Posture changes

Rats contract, stretch, curl up, stand on hind legs, and groom — their
silhouette changes drastically between frames. A curled-up rat looks very
different from a stretched-out one. If the training data was biased toward
certain postures, the model's confidence will fluctuate when the rat assumes
under-represented poses.

### 2.4 Close interactions

During nose-to-nose, side-by-side, or climbing interactions:
- Bounding boxes overlap heavily or merge into one
- Keypoints from both rats fall within overlapping regions
- The model can't reliably assign keypoints to the correct rat
- One rat may be partially occluded by the other

### 2.5 YOLO26's NMS-free architecture

YOLO26 uses a one-to-one detection head (no NMS post-processing). This is
generally an improvement, but it means:
- The `nms_iou` parameter in our config **does nothing** — it's silently ignored
- We cannot tune NMS behavior to control how overlapping boxes are handled
- The one-to-one head is designed for distinguishable objects — two identical
  rats push it into an ambiguous regime

### 2.6 Training distribution

Our YOLO26 model was trained at `imgsz=640` with 7 keypoints. If the training
set was sparse on:
- Occluded rats (one behind the other)
- Close interactions (nose-to-nose, climbing)
- Edge poses (curled up, standing, grooming)
- Low-contrast positions (rat color similar to cage floor)

...then the model's confidence in those situations will be low and flickery.

---

## 3. Why the Potato Demo Works

Ultralytics demos show YOLO counting potatoes on a conveyor belt with no
identity swaps. This works because the task is fundamentally easier:

| Factor | Potatoes on conveyor | Rats in cage |
|--------|---------------------|--------------|
| Movement | One direction, constant speed | Free, unpredictable |
| Overlap | Never touch | Nose-to-nose, climb over |
| Shape change | Same shape always | Contract, stretch, curl |
| Count | Dozens, briefly visible | Always 2, always present |
| Identity | Don't care which is which | Must track which is which |
| Tracking | BoT-SORT/ByteTrack (`model.track()`) | Detection-only (`model()`) |

The potato demo also uses **tracking** (`model.track()` with BoT-SORT or
ByteTrack), not detection-only. Each potato gets a persistent ID across frames.
Our pipeline uses `detect_only()` — zero temporal memory.

Even with tracking enabled, identical animals cause identity swaps. The potato
scenario simply never hits this limitation because potatoes don't interact.

---

## 4. What We Can Tune (Limited)

### 4.1 Confidence threshold

Currently `detection.confidence=0.25` (default). Options:
- Lower (0.10–0.20): fewer missed detections, but more false positives
- Higher (0.30–0.40): fewer false positives, but more missed rats

Worth testing on HPC to see if detection count stabilizes at lower thresholds.

### 4.2 Input size

Training was at `imgsz=640`. Video is 960×960, auto-scaled by YOLO. Could
try explicit `imgsz=960` to see if higher resolution helps.

### 4.3 Retraining

More training data with:
- Occluded poses
- Close interactions
- Edge cases (curled, standing, grooming)

This would help confidence stability but won't fix the fundamental
no-temporal-modeling problem.

---

## 5. The Research Landscape

Multi-animal identity tracking is an active research problem. Several tools
exist because **vanilla YOLO + tracker fails on identical animals**:

### AlphaTracker
- YOLO detection + single-animal pose estimation + IoU-based identity tracking
  with Kalman filter error correction
- Captures hierarchical visual information to maintain identity of nearly
  identical animals
- [Paper](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2023.1111908/full)

### DIPLOMAT
- Builds on DeepLabCut and SLEAP per-frame predictions
- Applies novel methods to tolerate occlusion and preserve identity
- Reduces identity swaps by >75% on the MABe mouse benchmark
- Remaining errors require manual correction — even specialized tools don't
  eliminate swaps entirely
- [Paper](https://www.biorxiv.org/content/10.1101/2025.08.11.669786v2.full)

### Identity-stable bidirectional segmentation
- Uses bidirectional video object segmentation with object-level memory
- Corrects identity swaps with less manual annotation effort
- Addresses the cost of identity correction in long recordings
- [Paper](https://www.biorxiv.org/content/10.1101/2025.09.22.677949v1.full)

### YOLO-BYTE (dairy cows)
- YOLO + ByteTrack for multi-object tracking of dairy cows
- Similar problem: identical-looking animals in close proximity
- Uses tracking to maintain identity, not detection-only
- [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0168169923002454)

### Key insight from the literature

All these tools acknowledge that YOLO-based detection alone is insufficient for
identical-animal identity tracking. Every solution adds **temporal modeling** on
top of the detector — whether through Kalman filters, bidirectional segmentation,
or specialized identity modules.

---

## 6. Experiment: YOLO on Per-Rat Crops (2026-03-04)

**Script:** `scripts/research/test_yolo_cropped_keypoints.py`

**Idea:** Run YOLO on cropped, mask-isolated rat regions instead of the full
image. Each crop contains only one rat (non-mask pixels blacked out), so YOLO
can't confuse identities. This was the "short-term pragmatic option" from the
proposed pipeline refactor.

**Result:** YOLO keypoints degrade significantly on crops.

- Full image: both rats detected, all 7 keypoints with labels and high confidence
- Cropped: rat detected but only ~3-4 keypoints, lower confidence, some missing

**Why it fails:**
- **Out of distribution**: YOLO was trained on full 640x640 images with cage
  context (floor, walls, both rats visible). A black background with just a rat
  silhouette is something the model has never seen.
- **Small crops**: rat bounding boxes are ~120-200px, upscaled to fit YOLO's
  input — loss of detail, blurry keypoints.
- **Missing context**: YOLO uses surrounding context (cage edges, other rat) as
  cues for keypoint localization. Removing context removes information.

**Conclusion:** Running YOLO on per-rat crops is **not viable** with the current
model. A dedicated pose model trained on cropped instances would be needed, which
is a larger effort (data collection, labeling, training).

---

## 7. What This Means for Us

### What we confirmed

Our centroid-propagation experiment (`debug_sam2_no_yolo.py`) already proved the
core insight: SAM2 with temporal prompts (centroids from previous frame) produces
almost perfect masks (1 blink in 3600 frames). The instability comes entirely
from YOLO's per-frame interference.

### The remaining contradiction

- **Masks**: SAM2 centroid propagation works — YOLO not needed after init
- **Keypoints**: ContactTracker needs per-rat keypoints (nose, ears, tail) for
  contact classification — only YOLO provides these
- **Problem**: YOLO's keypoints are unreliable (intermittent, swapping), but we
  need them for contacts

### Ruled out

- [x] ~~YOLO on per-rat crops~~ — keypoints degrade, out of distribution (Section 6)
- [x] ~~Periodic YOLO re-initialization~~ — YOLO swaps break stable propagation
- [x] ~~SAM2 multimask for tail coverage~~ — picks up background/cage

### Open questions

- [ ] Can we use YOLO keypoints on the full image with temporal filtering
  (average across N frames, reject outliers) to smooth out the noise?
- [ ] Can we extract keypoint-like positions from SAM2 masks? (e.g., nose as
  the mask point closest to the other rat, tail as the furthest point)
- [ ] Should we look at DeepLabCut or SLEAP for keypoints instead of YOLO?
- [ ] Would retraining YOLO with more interaction/occlusion data help?
- [ ] Can the multimask IoU selection (pick candidate by prev-mask IoU) improve
  tail coverage without the background problem?
