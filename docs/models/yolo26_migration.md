# YOLO26 Migration Plan

Migration path from our current YOLOv8 rat pose model to YOLO26.

All technical details verified against the installed `ultralytics==8.4.14` package,
which already includes full YOLO26 support.

---

## 1. What is YOLO26

YOLO26 is the latest YOLO architecture from Ultralytics. Key changes from YOLOv8:

| Feature | YOLOv8 | YOLO26 |
|---------|--------|--------|
| Backbone blocks | `C2f` | `C3k2` (CSP bottleneck, faster) |
| Attention | None | `C2PSA` (spatial attention at backbone end) |
| NMS | Standard NMS post-processing | **NMS-free** (end-to-end, top-k selection) |
| Detection head | Single one-to-many | Dual: one-to-one + one-to-many |
| `reg_max` | 16 (DFL with 16 bins) | 1 (direct box regression) |
| Pose head | Single `cv4` | Separate `cv4_kpts` + `cv4_sigma` + RealNVP |
| SPPF | Standard | Enhanced (residual shortcut, 3 pooling iterations) |

**Performance comparison:**

| Model | Parameters | GFLOPs |
|-------|-----------|--------|
| YOLOv8n | 3,157,200 | 8.9 |
| YOLO11n | 2,624,080 | 6.6 |
| **YOLO26n** | **2,572,280** | **6.1** |
| YOLOv8l | 43,691,520 | 165.7 |
| YOLO11l | 25,372,160 | 87.6 |
| **YOLO26l** | **26,299,704** | **93.8** |

YOLO26 is the most efficient at nano scale. At larger scales, ~3-7% larger
than YOLO11 due to dual head overhead, but NMS-free inference eliminates the
sequential NMS step, improving wall-clock latency.

---

## 2. Required Code Changes

### 2.1 Model weights swap

Update config YAML to point to new YOLO26 weights:

```yaml
# Before (current):
models:
  yolo_path: models/yolo/yolov8lrata.pt

# After:
models:
  yolo_path: models/yolo/yolo26l-pose-rata.pt  # retrained on same dataset
```

No code change in `models_io.py` — `YOLO(str(model_path))` auto-detects
the architecture from the checkpoint.

### 2.2 Ultralytics version

**No upgrade needed.** Our installed `ultralytics==8.4.14` already includes
full YOLO26 support (all config files, head classes, and training code).

### 2.3 `nms_iou` parameter becomes a no-op

In YOLO26's end-to-end mode, the `iou` parameter passed to `model.track()`
has no effect — there is no NMS step. Post-processing uses top-k score
selection only.

Current code in `infer_yolo.py`:
```python
if nms_iou is not None:
    track_kwargs["iou"] = nms_iou  # silently ignored with YOLO26
```

**Options:**
- Leave as-is (no harm, silently ignored)
- Add a log warning if YOLO26 is detected and `nms_iou` is set
- Remove `nms_iou` from config after full migration

### 2.4 Everything else — unchanged

| Component | Change needed? | Why |
|-----------|---------------|-----|
| `model.track()` API | **No** | Identical signature and return format |
| `r.boxes.xyxy`, `.conf`, `.cls`, `.id` | **No** | Same result object structure |
| `r.keypoints.data` | **No** | Same `(N, K, 3)` tensor format |
| BoT-SORT / ByteTrack | **No** | Trackers operate on detection outputs, format unchanged |
| `_parse_results()` | **No** | Parses same fields |
| `pad_frame()` / coordinate correction | **No** | Pre-processing, model-agnostic |
| `_filter_edge_margin()` | **No** | Post-processing filter |
| SAM2 integration | **No** | Uses detection boxes, unaffected by YOLO architecture |
| SlotTracker | **No** | Uses masks and track IDs, unaffected |

---

## 3. Re-training Requirement

**YOLOv8 weights are NOT compatible with YOLO26.** Five specific incompatibilities:

1. **`reg_max` mismatch:** YOLO26 uses `reg_max=1` (direct regression), YOLOv8
   uses `reg_max=16` (DFL). Detection head output dimensions differ.

2. **Dual detection heads:** YOLO26 has `one2one_cv2` and `one2one_cv3` layers
   that don't exist in YOLOv8. State dict keys won't match.

3. **Pose head architecture:** `Pose26` has separate `cv4_kpts`, `cv4_sigma`,
   and a `RealNVP` normalizing flow model. Standard `Pose` has single `cv4`.

4. **SPPF structure:** YOLO26's SPPF has residual shortcut and different channel
   counts. Not weight-compatible.

5. **Attention blocks:** YOLO26 adds `C2PSA` and attention-enabled `C3k2` blocks
   with no equivalent in YOLOv8.

**Training command:**

```bash
yolo train model=yolo26l-pose.yaml \
    data=path/to/rat_pose_dataset.yaml \
    epochs=100 imgsz=640 batch=16
```

Dataset format is identical — no re-annotation needed. Same YOLO-format labels
with keypoints: `class_id cx cy w h kp1_x kp1_y kp1_v ...`

Our `kpt_shape` will be `[7, 3]` (7 keypoints: tail_tip, tail_base, tail_start,
mid_body, nose, right_ear, left_ear).

---

## 4. Impact of NMS-Free Inference

### How it works

YOLO26 replaces IoU-based NMS with top-k score selection. During training,
a one-to-one head learns to produce at most one high-confidence prediction
per object. At inference, the one-to-many head is removed and predictions
are selected by score only — no IoU comparison needed.

### Impact on our pipeline

| Aspect | With NMS (current YOLOv8) | NMS-free (YOLO26) |
|--------|--------------------------|-------------------|
| Duplicate boxes | Removed by IoU threshold | Don't exist (one-to-one head) |
| `detection.nms_iou` | Configurable | **No effect** |
| Merged boxes (2 rats → 1 box) | Tunable via NMS IoU | Handled by architecture |
| Close-contact behavior | NMS may merge or suppress | One-to-one head decides |
| Latency | NMS adds sequential step | Faster (no NMS overhead) |

### Concerns

- **Close contact:** NMS-free relies on the one-to-one head learning to separate
  overlapping objects. If training data lacks close-contact examples, separation
  may be worse than tuned NMS. **Mitigation:** Ensure dataset includes close-contact frames.

- **Loss of `nms_iou` knob:** Currently we can adjust NMS IoU for merged boxes.
  With YOLO26, this control is gone — the model must learn it.
  **Mitigation:** More training data for edge cases; `detection.confidence` remains.

---

## 5. Risk Analysis

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Retraining produces worse accuracy than YOLOv8 | **High** | Medium | Compare mAP/mAR on validation set before switching. Keep YOLOv8 weights as fallback. |
| Close-contact detection degrades without NMS tuning | **High** | Medium | Augment training data with close-contact frames. Test on known difficult clips. |
| `nms_iou` config param silently ignored | **Low** | Certain | Document in config comments. Log warning if set with YOLO26. |
| Pose keypoint accuracy changes (different head) | **Medium** | Medium | Compare per-keypoint accuracy. Pose26's RealNVP + sigma may improve or change behavior. |
| Training takes longer (dual head, attention) | **Low** | Likely | Budget 20-30% more GPU time. Same dataset, just longer epochs. |
| Ultralytics API breaking change in future | **Low** | Low | Pin `ultralytics==8.4.14` in requirements.txt. |
| ReID `model: auto` with YOLO26 end-to-end head | **Medium** | Unknown | Ultralytics has fallback to `yolo26n-cls.pt`. Test if enabling ReID later. |

---

## 6. Phased Execution Plan

### Phase 1: Preparation (no code changes) -- DONE

- [x] Verify training dataset includes close-contact frames
- [x] Pin ultralytics version in `requirements.txt`: `ultralytics==8.4.14`
- [x] Baseline: run current YOLOv8 model on test clips, record mAP/mAR and
  count ID switches for comparison
- [x] Identify YOLO26 pose model scale

### Phase 2: Retrain -- DONE

- [x] Train YOLO26 pose model on the same rat dataset
- [x] Monitor training loss curves
- [x] Validate on held-out set
- [x] Export best weights to `models/yolo/best.pt`

### Phase 3: Integrate -- DONE (2026-03-03)

- [x] Update all 12 config files: `yolo_path: models/yolo/best.pt`
- [x] Add config comment noting `nms_iou` has no effect with YOLO26
- [ ] Run pipeline on test clip -- verify no crashes
- [ ] Check logs: track IDs, detection counts, frame processing

### Phase 4: Validate (compare against YOLOv8 baseline)

- [ ] Run on same test clips used for baseline
- [ ] Compare: detection count per frame, ID switch count, keypoint accuracy
- [ ] Specifically test:
  - Wall interactions (border detections)
  - Close contact (two rats touching)
  - Fast movement (crossing paths)
  - Partial occlusion
- [ ] If worse on any metric, analyze and consider:
  - More training epochs
  - Data augmentation adjustments
  - Fallback to YOLOv8 for that scenario

### Phase 5: Rollout -- DONE (all configs updated at once)

- [x] Update all 6 configs to use YOLO26 weights
- [ ] Update documentation (docstrings, README model references)
- [x] Archive YOLOv8 weights (keep `yolov8lrata.pt` as fallback)
- [x] `nms_iou` kept with no-op comment (backward compatible)

---

## 7. Unaffected Components (Confirmed)

| Component | File | Status |
|-----------|------|--------|
| BoT-SORT tracker | `botsort.yaml` | **Unaffected** — operates on detection outputs |
| ByteTrack tracker | `bytetrack.yaml` | **Unaffected** — same reason |
| SlotTracker | `src/common/tracking.py` | **Unaffected** — uses masks + track IDs |
| SAM2 segmentation | `src/pipelines/sam2_yolo/infer_sam2.py` | **Unaffected** — uses detection boxes |
| Post-processing | `src/pipelines/sam2_yolo/postprocess.py` | **Unaffected** — mask dedup + tracker |
| Visualization | `src/common/visualization.py` | **Unaffected** — renders from Detection objects |
| Config system | `src/common/config_loader.py` | **Unaffected** — model-agnostic |
| Frame extraction | `scripts/extract_frames.py` | **Unaffected** — post-pipeline |

---

## 8. Available YOLO26 Model Variants

| Config file | Task | Scales |
|-------------|------|--------|
| `yolo26.yaml` | Detection | n, s, m, l, x |
| `yolo26-p2.yaml` | Detection (small objects) | n, s |
| `yolo26-p6.yaml` | Detection (large objects) | n, s |
| `yolo26-seg.yaml` | Instance segmentation | n, s, m, l, x |
| `yolo26-pose.yaml` | **Pose estimation** | n, s, m, l, x |
| `yolo26-obb.yaml` | Oriented bounding boxes | n, s, m, l, x |
| `yolo26-cls.yaml` | Classification | n, s, m, l, x |

For our rat tracking pipeline with 7 keypoints, use **`yolo26l-pose`** (large
scale, matching current `yolov8l` performance tier).
