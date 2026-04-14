# Plan: Centroid Pipeline — SAM2 Centroid Propagation + YOLO Keypoints + Contacts

## Context

We've proven through experiments that:
1. **SAM2 centroid propagation** produces almost perfect masks (1 blink in 3600 frames)
2. **YOLO per-frame detection** is the root cause of reference pipeline instability
3. **YOLO on crops** doesn't work (keypoints degrade out of distribution)
4. **IdentityMatcher** already compensates for most of YOLO's noise in reference — the failures are edge cases during interactions

The new pipeline inverts the architecture:
- **SAM2 centroid propagation** drives masks and identity (stable backbone)
- **YOLO** is demoted to a keypoint provider on the full image (noisy signal)
- Keypoints assigned to masks by spatial overlap, not YOLO box ordering
- Temporal carry-over fills in keypoints when YOLO misses a rat
- **ContactTracker** runs on the stable masks + assigned keypoints

### Key differences from reference pipeline

| Aspect | Reference | Centroid (new) |
|--------|-----------|----------------|
| SAM2 prompts | YOLO boxes + keypoints every frame | Centroids from previous frame (YOLO only on init) |
| Identity source | IdentityMatcher (Hungarian matching) | Centroid ordering (inherent from propagation) |
| Keypoint assignment | `match_dets_to_slots()` by centroid distance | Mask containment (detection center inside mask) |
| Missing detections | Centroid fallback (skipped during MERGED) | Temporal carry-over of previous keypoints |
| State machine | SEPARATE/MERGED | Not needed (SAM2 handles interactions via negative prompts) |
| YOLO role | Drives everything (boxes → SAM2 → masks) | Keypoint-only provider; boxes ignored after init |

---

## Pipeline Flow

```
Frame 0 (init):
  YOLO detect → SAM2 segment with boxes → 2 masks + 2 centroids + keypoints
  Initialize slots: slot_masks, slot_centroids, slot_keypoints

Frame N (every subsequent frame):
  1. SAM2 centroid propagation → 2 stable masks
     - Positive prompt: this rat's prev centroid
     - Negative prompt: other rat's prev centroid
     - Update centroids from new masks

  2. YOLO detect on full image → 0-2 detections with keypoints (noisy)

  3. Assign keypoints to masks:
     - For each YOLO detection, check which mask contains its box center
     - If center inside mask_i → assign detection to slot i
     - Fallback: assign to nearest mask centroid

  4. Temporal carry-over (when YOLO misses a rat):
     - Slots without fresh YOLO keypoints → use previous frame's keypoints
     - Shift carried-over keypoints by centroid delta (frame-to-frame movement)
     - Flag as "carried_over" for quality tracking

  5. ContactTracker.update(slot_dets, slot_masks, slot_centroids, frame_idx)
     - Always runs (no MERGED state gating)
     - Quality flags propagated for reliability reporting

  6. Render: masks (SAM2) + keypoints (YOLO/carried) + contacts + status bar
```

---

## New Files

### 1. `src/pipelines/centroid/run.py` — Main pipeline (~200 lines)

The core pipeline loop. Reuses SAM2 segmentation functions from `debug_sam2_no_yolo.py`
and ContactTracker from `src/common/contacts.py`.

**Key functions to implement:**

```python
def _assign_keypoints_to_masks(
    detections: List[Detection],
    masks: List[np.ndarray],
    centroids: List[Tuple[float, float]],
) -> List[Optional[Detection]]:
    """Assign YOLO detections to SAM2 masks by spatial overlap.

    For each detection, check if its bounding box center falls inside
    a mask. If yes, assign to that slot. If no, assign to nearest
    mask centroid. Returns slot-aligned list (length = max_animals).
    """

def _carry_over_keypoints(
    slot_dets: List[Optional[Detection]],
    prev_slot_dets: List[Optional[Detection]],
    prev_centroids: List[Tuple[float, float]],
    curr_centroids: List[Tuple[float, float]],
) -> List[Optional[Detection]]:
    """Fill missing detections with previous frame's keypoints.

    Shifts carried-over keypoint positions by the centroid delta
    (curr_centroid - prev_centroid) to approximate movement.
    """

def run_pipeline(config_path, cli_overrides=None, start_frame=0,
                 end_frame=None, chunk_id=None):
    """Main pipeline loop."""
```

**Main loop structure:**
```
load config → load YOLO + SAM2 → open video → init ContactTracker

for frame in video:
    if not initialized:
        YOLO detect → segment_from_boxes → init centroids + keypoints
    else:
        segment_from_centroids → update centroids
        YOLO detect (full image) → assign to masks → carry over missing
        ContactTracker.update()

    render + write frame

ContactTracker.finalize()
```

### 2. `src/pipelines/centroid/__init__.py` — Empty module init

### 3. `configs/local_centroid.yaml` — Local config

Copy of `local_reference.yaml` with adjustments:
- Remove `identity_matcher` section (not used)
- Add `centroid` section with carry-over settings
- Keep `contacts` section (same parameters)
- Keep `detection` section (YOLO still runs for keypoints)

### 4. `configs/hpc_centroid.yaml` — HPC config

Same as local but with HPC paths (large SAM2 model, cuda device, full video path).

---

## Reused Modules (no changes needed)

| Module | What we use |
|--------|------------|
| `src/common/contacts.py` | ContactTracker (unchanged) |
| `src/common/config_loader.py` | load_config, setup_run_dir, setup_logging |
| `src/common/io_video.py` | open_video_reader, get_video_properties, create_video_writer, iter_frames |
| `src/common/visualization.py` | apply_masks_overlay, draw_centroids, draw_keypoints, draw_text |
| `src/common/metrics.py` | compute_centroid, mask_iou |
| `src/common/utils.py` | Detection, Keypoint dataclasses |
| `src/pipelines/sam2_yolo/infer_yolo.py` | detect_only() |
| `src/pipelines/sam2_yolo/models_io.py` | load_models() |

**SAM2 segmentation functions:** Copied from `scripts/debug_sam2_no_yolo.py` into `run.py`
(same `_segment_from_boxes` and `_segment_from_centroids`). These are small (~30 lines each)
and specific to this pipeline's prompting strategy, so inlining is cleaner than a shared module.

---

## File Summary

| File | Action | Est. lines |
|------|--------|-----------|
| `src/pipelines/centroid/__init__.py` | NEW | 1 |
| `src/pipelines/centroid/run.py` | NEW | ~200 |
| `configs/local_centroid.yaml` | NEW | ~70 |
| `configs/hpc_centroid.yaml` | NEW | ~70 |

No existing files modified.

---

## Verification

1. **Syntax check:** `python -c "import ast; ast.parse(open('src/pipelines/centroid/run.py').read())"`
2. **CLI test:** `python -m src.pipelines.centroid.run --help`
3. **Local run (10s clip, no contacts):**
   ```bash
   python -m src.pipelines.centroid.run --config configs/local_centroid.yaml
   ```
4. **Local run with contacts:**
   ```bash
   python -m src.pipelines.centroid.run --config configs/local_centroid.yaml contacts.enabled=true
   ```
5. **Compare output video** against reference pipeline on same clip
6. **Check contact CSV** matches expected format for ContactTracker
7. **Reference pipeline unaffected** (no files modified)
