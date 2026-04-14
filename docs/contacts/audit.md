# Social Contact System — Full Audit (updated 2026-03-05)

Comprehensive review of classification logic, pipeline integration, post-processing,
configuration, failure points, and risks.

---

## 1. Architecture Overview

```
YOLO26 (detect-only)
  |
  +-- detections: bbox + 7 keypoints + confidence
  |
  +-- Pipeline (reference / centroid / sam3)
  |     |
  |     +-- SAM2/SAM3 segmentation --> masks (pixel boolean H x W)
  |     +-- Identity resolution --> slot_masks, slot_centroids
  |     +-- match_dets_to_slots() --> slot_dets (detection per slot)
  |     |
  |     +-- ContactTracker.update(slot_dets, slot_masks, slot_centroids, frame_idx)
  |     |     |
  |     |     +-- Velocity from centroid delta (px/frame)
  |     |     +-- Body length EMA from keypoints (nose->tail_base)
  |     |     +-- classify_pair_contacts() per pair
  |     |     +-- Bout tracking (gap-tolerant grouping)
  |     |     +-- Stream CSV row to disk
  |     |
  |     +-- ContactTracker.finalize()
  |           +-- Close open bouts
  |           +-- Write bout CSV, summary JSON, PDF report
  |
  +-- Post-processing (scripts/postprocess_contacts_simple.py)
        +-- Load contacts_per_frame.csv
        +-- Rule 1: Majority-vote smoothing (window=7)
        +-- Rule 2: Gap bridging (max_gap=5 frames)
        +-- Rule 3: Min bout filter (0.5s / 15 frames at 30fps)
        +-- Inject NC (No Contact) for remaining empty frames
        +-- Write real_per_frame.csv, real_events.csv, event_log.txt,
            session_summary_real.json, reports/
```

### Key files

| File | Role |
|------|------|
| `src/common/contacts.py` | ContactTracker class + classify_pair_contacts() |
| `scripts/postprocess_contacts_simple.py` | Temporal filtering + human-readable output |
| `configs/contacts_postprocess_simple.yaml` | Post-processing parameters |
| `src/common/geometry.py` | match_dets_to_slots() (NOT used by contacts.py) |
| `src/pipelines/reference/run.py` | Reference pipeline integration |
| `src/pipelines/centroid/run.py` | Centroid pipeline integration |
| `src/pipelines/sam3/run.py` | SAM3 pipeline integration |
| `src/pipelines/reference/identity_matcher.py` | SEPARATE/MERGED state machine |

---

## 2. Contact Types (6 types + NC)

Classification runs per-pair per-frame in hard-coded priority order.
Higher-priority types suppress lower-priority ones (first match wins).

### Priority 1: N2N (Nose-to-nose)

| | |
|---|---|
| **Behavior** | Snouts directed at each other. Social recognition / greeting. |
| **Condition** | `dist(nose_A, nose_B) < contact_radius` |
| **Keypoints** | nose (index 4) of both rats |
| **Symmetric** | Yes — no investigator assigned |
| **Confidence** | High — nose is the most reliably detected keypoint |
| **Threshold** | `contact_radius = body_length * contact_zone_bl` (default 0.3 BL) |

### Priority 2: N2AG (Nose-to-anogenital)

| | |
|---|---|
| **Behavior** | Snout of one rat near tail base of the other. Primary individual recognition via pheromones. **Most important metric for social phenotyping.** |
| **Condition** | `dist(nose_A, tail_base_B) < contact_radius` OR `dist(nose_B, tail_base_A) < contact_radius` |
| **Keypoints** | nose (4) of one rat, tail_base (1) of the other |
| **Asymmetric** | Yes — investigator = nose owner |
| **Both directions** | Checked independently; first match wins |

### Priority 3: T2T (Tail-to-tail)

| | |
|---|---|
| **Behavior** | Both tail bases close together. Rear-to-rear contact. |
| **Condition** | `dist(tail_base_A, tail_base_B) < contact_radius` |
| **Keypoints** | tail_base (1) of both rats |
| **Symmetric** | Yes — no investigator assigned |

### Priority 4: FOL (Following)

| | |
|---|---|
| **Behavior** | One rat moves behind the other at close range in same direction. |
| **Conditions (all must hold)** | 1. `dist(nose_A, tail_base_B) < follow_radius` (0.5 BL) |
| | 2. Both speeds > `follow_min_speed_bl` (0.025 BL/frame) |
| | 3. `cos(vel_vec_A, vel_vec_B) > follow_alignment_cos` (0.7) |
| **Keypoints** | nose (4), tail_base (1), centroids across frames |
| **Asymmetric** | Yes — investigator = follower (nose owner) |
| **Temporal** | Cannot fire on frame 0 (no velocity yet) |

### Priority 5: SBS (Side-by-side)

| | |
|---|---|
| **Behavior** | Bodies parallel, in contact, low velocity. Affiliative resting / huddling. |
| **Conditions (all must hold)** | 1. `mask_iou(A, B) > sbs_mask_iou_min` (0.02) |
| | 2. Both velocities < `sbs_max_velocity_bl` (0.04 BL/frame) or unknown |
| | 3. `|cos(orient_A, orient_B)| > sbs_parallel_cos_min` (0.7) |
| **Requires** | SAM2 masks for both rats (IoU = 0 without masks → SBS impossible) |
| **Symmetric** | Yes — no investigator assigned |

### Priority 6: N2B (Nose-to-body) — catch-all

| | |
|---|---|
| **Behavior** | Snout near trunk/flank. General social investigation. |
| **Primary check** | Nose pixel inside other rat's SAM2 mask (mask containment) |
| **Fallback** | `dist(nose, mid_body) < contact_radius * 1.5` (wider radius) |
| **Keypoints** | nose (4), mid_body (3) |
| **Asymmetric** | Yes — investigator = nose owner |
| **Guards** | Skips if nose is within 1.5× contact_radius of the other's tail_base or nose (prevents absorbing N2AG/N2N) |
| **Note** | Last in priority order because mask containment is very broad — would absorb T2T, FOL, SBS if checked earlier |

### NC (No Contact)

| | |
|---|---|
| **Behavior** | Rats are independent — no social interaction. |
| **Generated by** | Post-processing only (not raw ContactTracker) |
| **Injection point** | After all 3 temporal filtering rules: `np.where(types == "", "NC", types)` |
| **Purpose** | Complete behavioral timeline — shows WHEN rats are independent |

### Zone classification (independent of contact type)

| Zone | Distance | Meaning |
|------|----------|---------|
| Contact | < 0.3 BL | Physical contact range |
| Proximity | 0.3 - 1.0 BL | Close approach, within reach |
| Independent | > 1.0 BL | No social interaction |

Note: Zone is based on centroid distance; contact type is based on keypoint/mask geometry.
A pair can theoretically be in INDEPENDENT zone but still trigger N2N if noses happen to
be close while centroids are far (unusual but possible with elongated body postures).

---

## 3. Pipeline Integration

### 3.1 Data passed to ContactTracker per frame

| Data | Source | Format |
|------|--------|--------|
| `slot_dets` | YOLO detections matched to slots by track_id (fallback: centroid proximity) | List[Detection] with .keypoints, .track_id |
| `slot_masks` | SAM2/SAM3 segmentation | List[Optional[ndarray]] — boolean (H, W) |
| `slot_centroids` | Mask centroid or bbox center | List[Optional[Tuple[float, float]]] |
| `frame_idx` | Pipeline frame counter | int (0-based) |

### 3.2 Per-pipeline differences

| Aspect | Reference | Centroid | SAM3 |
|--------|-----------|----------|------|
| **Segmentation** | SAM2 ImagePredictor | SAM2 centroid propagation | SAM3 Sam3Processor |
| **Identity** | IdentityMatcher (Hungarian) | SAM2 propagation (inherent) | IdentityMatcher (Hungarian) |
| **Keypoint assignment** | `match_dets_to_slots()` proximity | Mask overlap + proximity + carry-over | `match_dets_to_slots()` proximity |
| **MERGED handling** | `write_merged_placeholder()` | Lightweight merge detection (mask IoU > 0.5 → `write_merged_placeholder()`) | `write_merged_placeholder()` |
| **Coords** | All pixel-space | All pixel-space | SAM3 normalizes [0,1] internally, outputs pixel masks |

### 3.3 Reference pipeline (SEPARATE vs MERGED)

```python
if matcher.state == "SEPARATE":
    contact_events = contact_tracker.update(slot_dets, slot_masks, slot_centroids, frame_idx)
else:  # MERGED
    contact_tracker.write_merged_placeholder(slot_centroids, frame_idx)
```

**MERGED state** (animals overlapping, identity ambiguous):
- No contact classification — identity is unknown
- Placeholder row: `zone=proximity, contact_type=None, quality_flag=merged_state`
- Velocity tracking updated (centroids still available)
- Active bouts may close if gap exceeds `bout_max_gap_frames`

### 3.4 Centroid pipeline — keypoint carry-over + overlap resolution + merge detection

When YOLO misses a detection, the centroid pipeline carries forward the previous
frame's detection shifted by centroid delta:

```python
carried.x1 += dx; carried.y1 += dy; carried.x2 += dx; carried.y2 += dy
carried.is_carried_over = True  # quality flag for ContactTracker
for kp in carried.keypoints:
    kp.x += dx; kp.y += dy
```

**Stale keypoints (FIXED):** Carried-over detections are now tagged with
`is_carried_over=True`. `classify_pair_contacts()` checks this and sets
`quality_flag="stale_keypoints"` on the contact event.

**Mask overlap resolution (FIXED):** `resolve_overlaps()` imported from
`identity_matcher.py` is called after computing centroids. Contested pixels
(where both masks are True) are assigned to the nearest centroid. Prevents
inflated mask IoU → false SBS.

**Lightweight merge detection (FIXED):** After overlap resolution, pairwise mask
IoU is checked. If IoU > `mask_overlap_warning` (0.5), `write_merged_placeholder()`
is called instead of `update()` — no contact classification when identity is ambiguous.

```python
# Step 4b in centroid pipeline
for i, j in active_mask_pairs:
    if compute_mask_iou(slot_masks[i], slot_masks[j]) > merge_iou_thr:
        identity_ambiguous = True
        break
# Step 5
if identity_ambiguous:
    contact_tracker.write_merged_placeholder(slot_centroids, frame_idx)
else:
    contact_events = contact_tracker.update(...)
```

### 3.5 Post-processing trigger

All pipelines call post-processing at finalize:
```python
from scripts.postprocess_contacts_simple import run_postprocess
run_postprocess(run_dir / "contacts", fps=props["fps"])
```

Non-fatal — logs warning if it fails. Does not block pipeline completion.

---

## 4. Body Length Estimation

Body length is critical — all distance thresholds scale with it.

### 4.1 Estimation methods

1. **Primary:** `dist(nose, tail_base)` — keypoints 4 and 1
   - Requires both keypoints with confidence >= `min_keypoint_conf` (0.3)
   - Sanity check: distance > 5.0 px (rejects bogus overlapping keypoints)

2. **Fallback:** `bbox_diagonal * 0.6` — empirical ratio
   - Used when keypoints unavailable
   - Sanity check: diagonal > 10.0 px

3. **Config fallback:** `fallback_body_length_px` (default 120)
   - Used when EMA not yet populated (first frames)

### 4.2 EMA smoothing

```python
_bl_ema[slot] = _bl_ema[slot] * 0.9 + new_bl * 0.1  # alpha = 0.1
```

- 10% weight for new measurements (conservative, ~10-frame time constant)
- Per-slot tracking
- **Not configurable** — alpha is hard-coded

### 4.3 Impact of misestimation

| If BL is 2x too low | If BL is 2x too high |
|---|---|
| contact_radius halved → many contacts missed | contact_radius doubled → false contacts |
| Zone thresholds halved → rats appear farther apart | Zone thresholds doubled → rats appear closer |
| FOL: follow_radius halved → following rarely detected | FOL: follow_radius doubled → false following |

**Sensitivity: CRITICAL.** A 2x error propagates directly to all distance-based decisions.

---

## 5. Bugs Found and Fixed (2026-03-05)

### BUG 1: `follow_min_frames` was dead code (HIGH) — FIXED

**All config files** defined `follow_min_frames` but the code never read it.
FOL was classified per-frame with no temporal minimum.

**Fix:** `_close_bout()` now enforces `follow_min_frames` as the minimum duration
for FOL bouts. FOL bouts shorter than 30 frames (1.0s) are discarded even if they
pass the general `bout_min_duration_frames` (9 frames / 300ms) check.

**Updated thresholds (2026-03-05):** `follow_min_frames` increased from 5 to 30
(1.0s at 30fps) and `bout_min_duration_frames` from 2 to 9 (300ms at 30fps) based
on literature review. See `docs/contacts/threshold_research.md`.

### BUG 2: Bout gap threshold — NOT A BUG

`gap <= self.bout_max_gap + 1` is correct. `gap = frame_idx - end_frame` is 1
for adjacent frames (0 actual gap frames), so `gap <= 3 + 1 = 4` allows exactly
3 gap frames as configured. The `+ 1` compensates for the 1-based gap arithmetic.

### BUG 3: MERGED bout closing inconsistency (MEDIUM) — FIXED

During MERGED state, bouts closed with `gap > bout_max_gap` (stricter by 1 frame
than the SEPARATE path which uses `gap <= bout_max_gap + 1`).

**Fix:** Changed MERGED path to `gap > self.bout_max_gap + 1` for consistency.

### BUG 4: `merged_state` not in summary JSON (LOW) — FIXED

Quality counter for "merged_state" was tracked but not included in `_build_summary()`.

**Fix:** Added `merged_state_frames` to the quality section of the JSON summary.

### BUG 5: Multi-slot placeholder only generates pair (0,1) (LOW for N=2) — FIXED

When < 2 active slots, the code only generated a placeholder row for pair (0, 1)
with hardcoded slot indices. For N>2 pipelines, this skipped other pairs.

**Fix:** Now iterates all `num_slots` pairs with correct slot indices `(i, j)`.

### BUG 6: Post-processing real_event_id mixed types (LOW) — FIXED

`event_ids` list mixed int values and empty strings (""). DataFrame had object dtype.

**Fix:** Uses `np.full(len, -1, dtype=int)` — all values are int. All events
(including NC) get sequential IDs consistent with `contacts_real_events.csv`.
After NC injection, every frame belongs to an event (contact or NC).

---

## 6. Risks & Failure Modes

### 6.1 CRITICAL: Body length estimation failure

| Scenario | What happens | Mitigation |
|---|---|---|
| Both keypoints missing for 20+ frames | EMA stale, uses last good value | Acceptable — EMA decays slowly |
| First 10 frames have no keypoints | Uses `fallback_body_length_px=120` | May be wrong for different camera setups |
| Rat is curled up (nose near tail_base) | BL underestimated → all radii shrink | EMA smoothing reduces impact |
| Very different rat sizes | avg_bl splits the difference | Acceptable for same-strain pairs |

**Recommendation:** Validate `fallback_body_length_px` matches your camera/resolution setup.

### 6.2 ~~HIGH~~ DONE: Pixel-based thresholds don't scale with resolution

**Fixed:** `sbs_max_velocity_bl: 0.04` and `follow_min_speed_bl: 0.025` now use
body-length units. Converted at runtime: `threshold_px = threshold_bl × body_length`.
Backward-compatible: old `_px` keys still work as fallback.
All 10 config files updated. `det_slot_match_radius` remains in px (match pipeline scale).

### 6.3 HIGH: SBS impossible without masks

SBS requires `mask_iou > 0.02`. If SAM2/SAM3 fails to produce masks (OOM, model error),
mask_iou defaults to 0.0 and SBS is never triggered.

**Consequence:** All SBS events silently disappear. No warning or quality flag.

### 6.4 MEDIUM: MERGED state creates classification blind spot

During MERGED state (IdentityMatcher), **no contact is classified** despite the rats
being physically close (likely interacting). This can last 5-50+ frames.

**Consequence:** Contact time underreported during close interactions — exactly when
contacts are most relevant.

**Mitigations:**
- Post-processing gap bridging may recover some events (if same type before/after)
- MERGED frames show as NC in post-processed output
- Quality flag `merged_state` is set on these frames

### 6.5 ~~MEDIUM~~ DONE: Centroid pipeline stale keypoints

**Fixed:** Carried-over detections now have `is_carried_over=True` attribute.
`classify_pair_contacts()` checks this and sets `quality_flag="stale_keypoints"`.
ContactTracker counts these in `_quality_counts["stale_keypoints"]` and reports
in session summary. Frames with stale keypoints are visible in CSV and event log.

### 6.6 MEDIUM: FOL single-frame sensitivity

Without `follow_min_frames` enforcement (Bug 1), a single frame where both rats
happen to move in the same direction with a nose near a tail_base triggers FOL.

**Consequence:** Brief co-directional movements near each other count as following.
Post-processing min-bout filter (0.3s) helps, but the inline bout tracker only
requires 2 frames (`bout_min_duration_frames`).

### 6.7 LOW: Post-processing rule order matters

The 3 rules are applied in fixed order: smooth → gap-bridge → min-bout.
Changing the order would produce different results. This is by design but not
obvious to users.

### 6.8 LOW: FPS cascade fallback

If FPS can't be resolved from CLI, session_summary.json, or video file,
the post-processing falls back to `fps_fallback: 30.0` with a warning.
Wrong FPS causes all time-based calculations to be incorrect (but frame indices
remain correct).

---

## 7. Parameter Reference

### 7.1 Contact classification parameters (in all pipeline configs)

| Parameter | Default | Unit | Sensitivity | Notes |
|---|---|---|---|---|
| `enabled` | false | bool | — | Gate for entire module |
| `min_keypoint_conf` | 0.3 | — | LOW | Filters very low-conf keypoints |
| `contact_zone_bl` | 0.3 | body-lengths | **CRITICAL** | Scales all contact radii |
| `proximity_zone_bl` | 1.0 | body-lengths | **CRITICAL** | Zone boundary |
| `fallback_body_length_px` | 120 | pixels | **CRITICAL** | Default for first frames |
| `sbs_mask_iou_min` | 0.02 | ratio | LOW | Very permissive (2% overlap) |
| `sbs_max_velocity_bl` | 0.04 | body-lengths/frame | MEDIUM | Scales with BL (was `_px: 5.0`) |
| `sbs_parallel_cos_min` | 0.7 | cosine | LOW | Geometric invariant |
| `follow_radius_bl` | 0.5 | body-lengths | HIGH | Scales with BL |
| `follow_min_speed_bl` | 0.025 | body-lengths/frame | MEDIUM | Scales with BL (was `_px: 3.0`) |
| `follow_alignment_cos` | 0.7 | cosine | LOW | Geometric invariant |
| `follow_min_frames` | 30 | frames | HIGH | 1.0s at 30fps — enforced in `_close_bout()` |
| `bout_max_gap_frames` | 3 | frames | MEDIUM | Temporal continuity |
| `bout_min_duration_frames` | 9 | frames | HIGH | 300ms at 30fps — min meaningful contact |
| `mask_overlap_warning` | 0.5 | ratio | MEDIUM | Quality flag + centroid merge detection threshold |
| `det_slot_match_radius` | 200.0 | pixels | **CRITICAL** | Det-to-slot matching |

### 7.2 Post-processing parameters (contacts_postprocess_simple.yaml)

| Parameter | Default | Unit | Notes |
|---|---|---|---|
| `smoothing.window` | 7 | frames (odd) | ~233ms at 30fps |
| `gap_bridging.max_gap` | 5 | frames | ~167ms at 30fps |
| `min_bout.duration_sec` | 0.5 | seconds | 15 frames at 30fps |
| `fps_fallback` | 30.0 | Hz | Used only if no other FPS source |

### 7.3 Hard-coded values in code (not configurable)

| Value | Location | Purpose |
|---|---|---|
| `_bl_alpha = 0.1` | contacts.py:481 | Body length EMA weight |
| `N2B radius * 1.5` | contacts.py:369 | Wider radius for nose-to-body fallback |
| `BL sanity > 5.0 px` | contacts.py:162 | Reject bad keypoint estimates |
| `bbox fallback * 0.6` | contacts.py:167 | Empirical BL from bbox diagonal |
| `match_dist = 200.0 px` | geometry.py:26 | match_dets_to_slots max distance |

### 7.4 Config consistency

All 6 main pipeline configs (local/hpc x reference/centroid/sam3) have **identical**
contacts parameters with all 15 keys present.

Legacy configs (`hpc_full.yaml`, `local_sam2video.yaml`, `hpc_sam2video.yaml`)
are missing `det_slot_match_radius` — falls back to code default (100.0).

---

## 8. Output Files

### Raw outputs (from ContactTracker)

| File | Format | Content |
|------|--------|---------|
| `contacts_per_frame.csv` | CSV, 23 columns | One row per pair per frame. Streamed to disk. |
| `contact_bouts.csv` | CSV, 17 columns | One row per temporal bout (gap-tolerant grouping) |
| `session_summary.json` | JSON | Zone/type/pair/quality statistics, 1-min time bins |
| `report.pdf` | PDF, 8 pages | Timeline, histograms, pie chart, time-binned rates, etc. |

### Post-processed outputs (from postprocess_contacts_simple.py)

| File | Format | Content |
|------|--------|---------|
| `contacts_real_per_frame.csv` | CSV | Original + real_type, real_zone, real_event_id |
| `contacts_real_events.csv` | CSV, 16 columns | One row per clean event (chronological) |
| `event_log.txt` | Text | Human-readable legend, event table, summary, validation tips |
| `session_summary_real.json` | JSON | Filtering impact, events_by_type (incl. NC) |
| `reports/timeline_comparison.png` | PNG | Raw vs real timeline |
| `reports/duration_by_type.png` | PNG | Raw vs real duration bars (contacts only) |
| `reports/events_by_type.png` | PNG | Raw vs real event count bars (contacts only) |
| `reports/event_duration_distribution.png` | PNG | Histogram per type (incl. NC) |
| `reports/comparison_by_type.csv` | CSV | Per-type raw vs real stats (incl. NC) |
| `reports/comparison_global.csv` | CSV | Global metrics and flicker rates |

---

## 9. Post-Processing Filtering Rules

### Rule 1: Majority-vote smoothing

- Slides centered window (default 5 frames) over per-frame labels
- Each frame adopts the most frequent label in its window
- Window shrinks at edges (asymmetric)
- Tie-breaking: keep current value > prefer non-empty > priority order
- Empty strings ("") participate in voting
- Effect: corrects 1-2 frame flickers

### Rule 2: Gap bridging

- Scans left-to-right for contiguous runs of same non-empty type
- If a gap of <= `max_gap` empty frames separates two runs of the **same type**, fills the gap
- Different types are NOT bridged
- Effect: merges bouts separated by brief tracking failures

### Rule 3: Minimum bout filter

- After gap bridging, removes contiguous runs shorter than `min_frames`
- Replaced with empty string (later becomes NC)
- Effect: eliminates brief noise events

### NC injection

After all 3 rules: `real_types = np.where(types == "", "NC", types)`

NC is never present during filtering — it's a post-hoc label for the output.

---

## 10. Recommendations

### Immediate fixes — ALL DONE (2026-03-05)

1. ~~**`follow_min_frames` was dead code**~~ — FIXED: now enforced in `_close_bout()`, updated to 30 frames (1.0s)
2. ~~**Bout gap off-by-one**~~ — NOT A BUG: `+1` is correct (gap=1 for adjacent frames)
3. ~~**`merged_state_frames` missing from summary JSON**~~ — FIXED
4. ~~**`det_slot_match_radius` mismatch**~~ — FIXED: default 100→200px to match pipeline's match_dets_to_slots() max_distance
5. ~~**Threshold values too low for rat behavior**~~ — FIXED: `bout_min_duration_frames` 2→9, `follow_min_frames` 5→30, postprocess thresholds updated. See `docs/contacts/threshold_research.md`

### Classification fixes — ALL DONE (2026-03-05)

6. ~~**N2B absorbing FOL/SBS/T2T**~~ — FIXED: priority reordered to N2N → N2AG → T2T → FOL → SBS → N2B (catch-all last). N2B mask-containment was firing before the more specific types could be evaluated.
7. ~~**N2B near-miss absorption**~~ — FIXED: N2B now guards against nose being within 1.5× contact_radius of the other's tail_base or nose (N2AG/N2N territory). Also uses `round()` instead of `int()` for mask pixel lookup.
8. ~~**A-first bias in N2AG/FOL**~~ — FIXED: when both directions qualify, picks the closer distance instead of always choosing slot A.
9. ~~**Velocity noise causing false FOL/missed SBS**~~ — FIXED: velocity is now EMA-smoothed (alpha=0.3) instead of raw frame-to-frame delta. Reduces centroid jitter effects.

### Post-processing & merge fixes — ALL DONE (2026-03-05)

10. ~~**NC event_id inconsistency**~~ — FIXED: `write_real_per_frame()` now assigns event_ids to NC events too, matching `extract_events()`. Previously NC frames got `-1` in per-frame CSV while having proper IDs in events CSV.
11. ~~**T2T missing from merge report + SBS overwritten**~~ — FIXED: `merge_chunks.py` report now includes all 6 contact types (was missing T2T). Subplot grid dynamically sized with `squeeze=False` to prevent overwrite and crash.
12. ~~**Bout ID collision across chunks**~~ — FIXED: `_renumber_bout_ids()` assigns globally unique bout_ids after chunk merge. Per-frame CSV bout_ids updated to match.
13. ~~**ZeroDivisionError on empty CSV**~~ — FIXED: CLI path (`main()`) returns early on empty CSV instead of continuing to divide by zero. Also guarded `total_sec` division in `write_event_log()`.
14. ~~**Negative bouts_removed**~~ — FIXED: `bouts_removed` now compares raw bouts vs real *contact* events only (excludes NC). Previously NC events inflated the real count, making `raw - real` negative.
15. ~~**Chart 4 axes crash**~~ — FIXED: added `squeeze=False` to `plt.subplots()` in duration distribution chart. Without it, `axes[row][col]` crashes when `n_rows==1`.

### Known limitations

16. **Orphaned bout_ids in raw per-frame CSV** — ContactTracker assigns bout_ids during active bouts. When `_close_bout()` filters a short bout, those frames retain stale bout_ids. Post-processing addresses this via `real_event_id` in the cleaned CSV. Not fixed in ContactTracker to avoid complexity.
17. **Bout splitting at chunk boundaries** — bouts spanning chunk boundaries are split into two sub-bouts. If either sub-bout fails the minimum duration filter, that portion is lost. Mitigated by `_renumber_bout_ids()` but not fully resolved.

### Centroid pipeline fixes — ALL DONE (2026-03-05)

18. ~~**ContactTracker re-matching detections (CRITICAL)**~~ — FIXED: `ContactTracker.update()` now respects `track_id` (set by pipelines as `slot_idx + 1`) for slot mapping. Previously re-matched by proximity with first-match-wins, causing slot 0 to always win during close contact → wrong keypoints on wrong rat → incorrect investigator roles. Proximity fallback retained only for detections without `track_id`.
19. ~~**Add `resolve_overlaps()` to centroid pipeline**~~ — FIXED: imported from `identity_matcher.py`, called after computing centroids from masks. Contested pixels assigned to nearest centroid. Prevents inflated mask IoU → false SBS contacts.
20. ~~**Add quality flag for carried keypoints**~~ — FIXED: `_carry_over_keypoints()` sets `is_carried_over=True` on deep-copied detections. `classify_pair_contacts()` detects this via `getattr()` and sets `quality_flag="stale_keypoints"`. Counted in `_quality_counts` and reported in session summary JSON.
21. ~~**Add lightweight merge detector for centroid pipeline**~~ — FIXED: checks pairwise mask IoU after overlap resolution. If IoU > `mask_overlap_warning` (0.5), uses `write_merged_placeholder()` instead of `update()`. Prevents contact classification during identity-ambiguous periods. Status bar shows `[AMBIGUOUS]`.
22. ~~**Dead carry-over detection code**~~ — FIXED: removed first `has_carry_over` check that used `is` comparison on deep-copied objects (always `False`). Kept only the working `id()`-based check.
23. ~~**Normalize pixel-based thresholds**~~ — FIXED: `sbs_max_velocity_bl: 0.04` and `follow_min_speed_bl: 0.025` now use body-length units. Converted at runtime: `threshold_px = threshold_bl × body_length`. Backward-compatible: old `_px` keys still work as fallback. All 10 config files updated.

### Additional bugs found (re-audit 2026-03-05)

24. ~~**Velocity EMA not updated during MERGED state (HIGH)**~~ — FIXED: `write_merged_placeholder()` only updated `_prev_centroids` but never updated `_vel_ema` or `_vel_vec_ema`. On MERGED → SEPARATE transition, velocity EMA jumped to the centroid delta accumulated across the entire MERGED period → false FOL triggers or missed SBS. Now computes velocity EMA in `write_merged_placeholder()` identically to `update()`.
25. ~~**Type hint `List[Optional[object]]` in centroid pipeline (LOW)**~~ — FIXED: changed to `List[Optional["Detection"]]` for clarity. No runtime impact.

### Remaining improvements — SKIPPED (not needed)

26. ~~**Make `_bl_alpha` configurable**~~ — SKIPPED: alpha=0.1 is a stable internal constant, not a behavioral parameter. No practical scenario requires tuning it.
27. ~~**Add explicit warning when SBS disabled**~~ — SKIPPED: if SAM2/SAM3 fails to produce masks, the pipeline has much bigger visible problems (no overlay, no identity). An SBS-only silent failure is unrealistic.

### Future considerations

28. **MERGED state contact inference** — use pre-merge contact type to infer during MERGED
29. **N>2 support** — fix single-detection placeholder for multi-animal experiments
30. **Body length validation** — warn if estimated BL differs significantly from fallback

---

## 11. Research Findings (2025-2026 literature, 2026-03-05)

### 11.1 Actionable improvements (ranked by effort-to-impact)

#### 1. Keypoint validity filtering via SAM2 masks (TRIVIAL effort)

**Source:** ADPT (eLife, March 2025)

YOLO keypoints that fall **outside their rat's SAM2 mask** are likely errant detections.
Adding a point-in-mask check before contact classification would reject bad keypoints
at the source — more principled than relying on EMA smoothing alone.

```python
# Proposed check in _assign_keypoints_to_masks() or contacts.py
if mask is not None:
    for kp in det.keypoints:
        ny, nx = round(kp.y), round(kp.x)
        if not (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx]):
            kp.conf = 0.0  # suppress errant keypoint
```

**Impact:** Reduces false contacts from errant keypoints, especially during occlusion.

#### 2. Mask intersection region for contact classification (MODERATE effort)

**Source:** s-DANNCE (Cell, 2025) — 3D mesh-based social touch detection

Instead of pure keypoint-to-keypoint distances, use the **overlap region of two SAM2 masks**
to determine where on each body the contact occurs:
1. Compute mask intersection pixels
2. Find which keypoints of each rat are closest to the intersection
3. Use keypoint identity (nose, tail_base, mid_body) to classify contact type

This is more robust than point distances because it uses visible physical overlap.
Our `resolve_overlaps()` already computes the overlap — we just need to use it
for classification instead of only for pixel assignment.

**Impact:** More robust contact type classification, especially for N2B and SBS.

#### 3. HMM-based bout detection (MODERATE effort)

**Source:** Infinite hidden semi-Markov model (Nature Neuroscience, Late 2025),
Keypoint-MoSeq (Nature Methods, 2024), Sticky Poisson HMM (2025)

Replace gap-bridging + min-bout with a 2-3 state HMM per contact type:
- States: `CONTACT`, `GAP`, `NO_CONTACT`
- Sticky HMM variant prevents over-segmentation (like our gap bridging but probabilistic)
- `hmmlearn` or `ssm` Python libraries, minimal code

**Impact:** Better bout boundary detection, handles ambiguous transitions naturally.

#### 4. DAM4SAM — distractor-aware memory for SAM2 (MODERATE effort)

**Source:** DAM4SAM (CVPR 2025) — plug-in module, no SAM2 retraining

Maintains **reference frames** of well-separated rats. When masks overlap (MERGED state),
uses these reference frames as additional SAM2 memory to distinguish the two animals.
Sets new state-of-the-art on 6 benchmarks.

For our pipeline: store 3-5 "clean separation" frames, use during merge detection
and split resolution. Would help `_resolve_split()` in `identity_matcher.py`.

**Impact:** Fewer identity swaps during close interactions.

#### 5. Bidirectional SAM2 for identity swap detection (HIGH effort)

**Source:** Bidirectional VOS (bioRxiv, September 2025)

Run SAM2 forward + backward on the same clip. Per-frame IoU between forward and
backward masks drops at identity swap points. Flag these frames for re-resolution.

**Impact:** Catches identity swaps that current state machine misses.

### 11.2 New tools/methods (for reference only)

| Tool | Year | Key concept | Relevance |
|------|------|-------------|-----------|
| ViT-RSI | 2025 | Vision transformer for social interaction classification | Alternative paradigm, ensemble validator |
| InteBOMB | 2025 | Dual long/short-term memory for tracking | Memory concept for IdentityMatcher |
| STCS | 2024 | Optical flow alongside segmentation | Denser motion for FOL/SBS |
| Seg2Track-SAM2 | 2025 | Sliding-window SAM2 memory (75% less VRAM) | Long video processing |
| DIPLOMAT | 2025 | Skeleton constraints, 75% fewer ID swaps | Skeleton integrity check |
| New idtracker.ai | 2025 | Contrastive learning for identity | Visual identity embeddings |
| vmTracking | 2025 | Virtual markers for occlusion handling | Multi-animal identity |
| PPAP | CVPR 2025 | Probabilistic prompts for pose estimation | Uncertainty in SAM2 prompts |
| SuperAnimal | 2024 | Confidence-based keypoint filtering | Already partial (min_conf=0.3) |
