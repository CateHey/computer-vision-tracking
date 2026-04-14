# Risks and Known Limitations

Consolidated list of known failure modes, limitations, and workarounds
across all pipeline components. Use this document to understand what can
go wrong and how to mitigate it.

---

## Tracking / Identity

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| **ID swap during fast crossings** | Rats swap colors for 1-2 frames | Swap guard compares straight vs swapped cost (N=2) | Implemented |
| **Linear velocity assumption** | During curves/turns, predicted centroid overshoots | EMA smoothing (0.7/0.3) + clamping at ±half proximity_threshold | Implemented |
| **N=2 only** | State machine assumes exactly 2 entities | Would need N×N Hungarian + cluster detection for N>2 | Design limitation |
| **Resolution-dependent thresholds** | proximity_threshold=150px assumes ~640px width | Tune per resolution or normalize to frame diagonal | Manual tuning needed |
| **Long occlusion (>30 frames)** | Force-resets MERGED→SEPARATE, may assign wrong IDs | max_merged_frames is configurable; increase for slower interactions | Configurable |
| **No cross-gap re-identification** | If both rats leave frame and return, IDs may swap | Would need appearance features (Re-ID); not available for identical rats | Unsolved |

## Detection (YOLO)

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| **Identical appearance** | YOLO cannot distinguish rat A from rat B | We don't rely on YOLO identity — IdentityMatcher handles this | By design |
| **Border/edge hallucinations** | YOLO detects partial reflections at cage borders | `edge_margin` config parameter to ignore detections near frame edges | Available in sam2_yolo, not in reference |
| **Keypoint errors during occlusion** | When rats overlap, keypoints may land on wrong rat | Negative keypoints from overlapping detections fed to SAM2/SAM3 | Implemented |
| **YOLO misses during fast movement** | Rat moves too fast between frames → no detection | Centroid fallback in reference/SAM3 pipeline keeps tracking | Implemented |
| **NMS-free YOLO26 double detections** | YOLO26 occasionally outputs overlapping boxes | filter_duplicates removes redundant masks by IoU | Implemented |

## Segmentation (SAM2 / SAM3)

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| **Mask quality depends on prompt** | Bad YOLO box → bad SAM mask | Keypoint prompts + negative points improve precision | Implemented |
| **Centroid fallback accuracy** | Point prompt less precise than box prompt | Only used when YOLO misses; results are filtered by IoU dedup | Implemented |
| **Merged blob segmentation** | When rats overlap, SAM segments them as one blob | MERGED state skips centroid fallback to avoid duplicating blob | Implemented |
| **SAM3 coordinate normalization** | SAM3 uses [0,1] coords vs SAM2 pixel coords | sam3_processor.py converts all prompts to normalized space | Implemented |
| **SAM3 API not yet validated** | Sam3Processor.predict() interface may differ from expected | Processor designed to be adaptable; verify after installing sam3 package | Needs testing |

## Contact Classification

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| **Depends on keypoint quality** | Wrong keypoints → wrong contact type (e.g., N2N vs N2AG) | `min_keypoint_conf` threshold filters low-confidence keypoints | Configurable |
| **Skipped during MERGED state** | Contacts not classified when rats are close/overlapping | By design: identity is ambiguous during merge | Design decision |
| **FOL sensitivity to jitter** | Following detection depends on velocity direction, tracking jitter causes false positives | `follow_min_frames` and `follow_min_speed_px` thresholds | Configurable |
| **SBS false positives** | Side-by-side detected when rats are close but not aligned | `sbs_parallel_cos_min` checks directional alignment | Configurable |
| **Bout gap tolerance** | Short interruptions split one interaction into multiple bouts | `bout_max_gap_frames` bridges small gaps (default 3) | Configurable |

## HPC / Parallel Processing

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| **Chunk boundary discontinuity** | Identity may re-initialize at chunk boundaries | Each chunk starts fresh; contacts merged by frame offset | Known limitation |
| **GPU memory per chunk** | SAM2-large + YOLO on MIG partition may OOM | Use SAM2-tiny for testing; large only on full A100 | Manual selection |
| **TMPDIR filling up** | pip install on login node fills /tmp | `export TMPDIR=$HOME/tmp && mkdir -p $TMPDIR` | Documented |
| **Git conflicts on Bunya** | Local changes conflict with `git pull` | `git stash && git pull && git stash pop` | Documented |
| **SAM3 checkpoint size** | sam3.pt is ~3.2GB, may be slow to download | Cache in models/sam3/; use HuggingFace CLI with auth | Documented |

## General

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| **No ground truth for tracking** | Cannot quantitatively measure ID accuracy | Visual inspection of overlay videos + observation log | Manual process |
| **Domain-specific tuning** | Parameters tuned for 640px cage videos; may not generalize | All thresholds are configurable via YAML | By design |
| **Single-species assumption** | System assumes all detected objects are the same species | `filter_class` config can restrict YOLO to specific classes | Available |

---

## Risk Priority

**High** — actively affects output quality:
- ID swap during fast crossings (mitigated but not eliminated)
- Long occlusion force-reset
- Chunk boundary discontinuity

**Medium** — occasionally affects output:
- Keypoint errors during overlap
- Centroid fallback accuracy
- Contact classification during marginal interactions

**Low** — rare or well-mitigated:
- Border hallucinations
- Resolution-dependent thresholds
- YOLO double detections
