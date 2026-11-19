# Observation Log

Chronological record of video analysis observations, problems found, and fixes applied.
Each entry documents a test run and what was learned from it.

---

## How to Use This Log

After watching an overlay video, add an entry below with:

1. **Date and video** — what was tested
2. **Pipeline and config** — which pipeline, which settings
3. **Observations** — what you saw (good and bad)
4. **Problem** — root cause analysis if something went wrong
5. **Action** — what was changed (or planned)
6. **Result** — did it improve? (fill in after re-testing)

### Observation Categories

| Tag | Meaning |
|-----|---------|
| `ID-SWAP` | Identity swap — colors flip between rats |
| `ID-LOST` | Identity lost — rat disappears from tracking |
| `MERGE` | Merged detection — two rats become one blob |
| `SPLIT` | Split error — wrong IDs assigned after separation |
| `FALSE-DET` | False detection — wall, shadow, or artifact detected as rat |
| `KEYPOINT` | Keypoint error — wrong body part position |
| `CONTACT` | Contact classification error — wrong type or missed contact |
| `MASK` | Mask quality — SAM2 mask doesn't match rat shape |
| `PERF` | Performance — speed, memory, GPU usage |

---

## Entries

### Entry 001 — Baseline (pre-YOLO26, pre-state machine)

- **Date:** 2026-02 (approx)
- **Video:** `data/raw/original_120s.avi`
- **Pipeline:** sam2_yolo (`configs/hpc_full.yaml`)
- **Config:** YOLOv8 + BoT-SORT + SlotTracker

**Observations:**
- `ID-SWAP`: Colors flip during close interactions (BoT-SORT swaps IDs)
- `MERGE`: Two rats merge into one blob when touching → one slot gets corrupted
- `SPLIT`: After separation, IDs don't recover correctly
- General tracking works well when rats are apart

**Problem:** BoT-SORT relies on appearance features, but both rats look identical.
SlotTracker tries to compensate but the corrupted state cascades.

**Action:** Designed reference pipeline with IdentityMatcher + state machine.
Switched to YOLO26 model.

**Result:** _Pending validation with new pipeline._

---

### Entry 002 — Contact rules v3 after the manual comparison of the 6-rat clip

- **Date:** 2026-11-19
- **Video:** 6-rat clip, 10 s (300 frames, 960x960, 30 fps); run `2026-10-28_091052_cutie_composite`
- **Pipeline:** cutie_composite (ContactTrackerV2 + `postprocess_contacts_simple.py`)
- **Config:** `configs/local_cutie_composite.yaml` / `hpc_cutie_composite.yaml` with
  `contact_rules_v3: true` (`false` reproduces the earlier rules);
  `configs/contacts_postprocess_simple.yaml` (new `brief` and `following` blocks)
- **Commit:** this commit ("Arreglo en interacciones")

**Observations** (38 manual annotations, frame by frame, against `results.xlsx`; Events kappa ~0.62):
- `CONTACT`: when and between whom are found well; the type is weak, mostly N2B and FOL.
- `CONTACT`: ~30 % of the manual interactions last less than the 0.3 s minimum of Events
  and vanish from every sheet.
- `CONTACT` / `MASK`: a nose on the other rat's tail counted as nose-to-body, because the
  tail is part of the CUTIE mask; thin leaks of background inflate mask contact.
- `CONTACT`: N2B was zeroed whenever SBS was strong; FOL was annulled whenever the bodies
  were < 0.5 BL apart, with no touch; N2AG fired on distance alone (nose < 0.30 BL from the rear).
- `CONTACT`: one rat trailing another while sniffing alternated N2AG / FOL and split one
  chase into many events.

**Problem:** the touch tests used the full mask (tail included) and a fixed 6 px window
around the nose keypoint, which sits ~5 px inside the head; several guards annulled a type
from distance alone.

**Action** (`src/common/contacts_v2.py` unless noted; all behind `contact_rules_v3`):
1. Body mask for detection: opening with a disk of 0.10 BL (`body_mask_open_bl`), largest
   component, border restored. Removes the tail and thin leaks. Internal only: the video
   still shows full masks. `suspect_mask_i/j` flags a mask > 1.4x the rat's median area.
2. Refined nose tip from the own mask (most forward border pixel within 0.15 BL of the
   nose keypoint) and a head circle of 0.08 BL (`head_circle_bl`) instead of the fixed 6 px.
3. N2AG: nose tip <= 0.30 BL from the rear **and** evidence of touching (head circle on the
   body, or <= 0.20 BL, or slow there >= 5 frames); otherwise score x 0.45. Gaze factor
   0.2 + 0.8 (cos + 1) / 2.
4. FOL is annulled only by a real touch (body masks or a head circle on the body).
5. N2B no longer zeroed by SBS; it competes (primary / secondary). Zone touched:
   head (t < 0.2) / torso (< 0.75) / rear. A nose on the tail only is `nose_on_tail`, not contact.
6. SBS measured on body masks; `tail_contact_bl` reported apart.
7. N2N subtypes frontal / perpendicular / lateral (head directions); lateral keeps 85 %.
8. `low_confidence`: bodies touch, no type >= 0.5, best type >= 0.35 is assigned and flagged.
9. `ambiguous`: primary - secondary < 0.10. Events now carry `secondary_type`, `subtype`,
   `ambiguous`, `low_confidence` (`scripts/postprocess_contacts_simple.py`).
10. N2AG subtype following (both walking) / classic; new sheet **Following** in
    `diagnostics.xlsx`: FOL + N2AG-while-walking events of the same pair and follower,
    merged when <= 0.5 s apart.
11. New sheet **Brief** in `diagnostics.xlsx`: runs of 3-8 frames (FOL up to 11) that the
    0.3 s minimum keeps out of Events; `near_event` says when one is glued to an event of
    the same pair. Events itself is unchanged.
12. Dynamics `subkind`: approach = search / mutual_approach; avoidance = withdrawal (right
    after a contact) / flight (no contact before) / mutual_separation; plus `after_contact`
    and `ends_in_contact`.

Unchanged: the 5 types, the hierarchy, smoothing and gap bridging.

**Calibration of the head circle** (masks rebuilt from the per-rat overlay videos, centroid
error median 1.5 px; 409 directed frames annotated as N2N / N2AG / N2B against ~7 900
frames with no annotation):

| Touch rule | Annotated frames detected (of 409) | False | Precision | F1 |
|---|---|---|---|---|
| Old: 6 px square at the nose keypoint, full mask | 268 | 82 | 0.77 | 0.71 |
| New, 8 px (0.05 BL) | 269 | 39 | 0.87 | 0.75 |
| **New, 13 px (0.08 BL), kept** | **314** | **52** | **0.86** | **0.81** |
| New, 20 px | 347 | 75 | 0.82 | 0.84 |

The confirmed non-contacts (R1->R4 243-252 walking past a tail, R4-R5 184-194 tail,
R5->R6 29-35) give no touch with the new rule; the old rule fired on R1->R4 in 4 frames.

**Result:** 133 tests pass. Brief on the real `results.xlsx`: Events unchanged (21),
Brief 39 runs (26 isolated), including R2-R6 207-210 and R3-R4 207-210. On the synthetic
demo the scripted N2B scene (frames 1560-1710) is found at 1593-1675 (old rules: 1644-1652).
_Pending: rerun the tracker on the real clip with v3 and compare with the manual annotations._
Watch: lateral N2N competing with SBS when two rats walk side by side with heads level
(noses ~0.30 BL apart); `social_anogenital_split` 0.65-0.7; SBS subtypes.

---

### Entry 003 — N2AG gaze factor ignored with the nose on the rear (fix to Entry 002)

- **Date:** 2026-11-19
- **Video:** same 6-rat clip, run `2026-10-28_091052_cutie_composite`
- **Pipeline:** cutie_composite, `contact_rules_v3: true`
- **Commit:** this commit

**How it was tested:** the run saved no masks or keypoints, so both were rebuilt from the
per-rat overlay videos (the mask is the tinted area; the keypoints are the pure-colour dots)
and ContactTrackerV2 + post-processing were rerun. With the old rules the rebuilt run gives
the same type as the original `results.xlsx` in 98.8 % of the pair-frames, and its 21 Events
match within 1-3 frames (2 lost, 2 gained), so the inputs are faithful.

**Observation:**
- `CONTACT`: with the Entry 002 rules, 88 of the 190 frames annotated as N2AG came out as
  N2B zone "rear": same pair, frames and investigator, wrong type.

**Problem:** the stronger N2AG gaze factor (0.2 + 0.8 (cos + 1) / 2) compares the head
direction with the direction nose -> rear. With the nose on the rear that vector is a few px
long and points anywhere, so the N2AG score fell (e.g. 0.75 -> 0.35) and N2B won.

**Action:** `_score_n2ag` skips the gaze factor when the nose is within `n2ag_close_bl`
(0.20 BL) of the rear; further away it still applies. Test:
`test_n2ag_gaze_is_ignored_with_the_nose_on_the_rear`.

**Result** (frame by frame against the 35 non-tentative manual annotations):

| | Old rules (rebuilt) | Entry 002 | Entry 003 |
|---|---|---|---|
| Contact frames found (recall) | 0.656 | 0.695 | **0.717** |
| Contact frames correct (precision) | 0.719 | 0.745 | 0.735 |
| Same type when both see contact | 0.845 | 0.533 | 0.837 |
| Cohen's kappa (6 classes) | 0.609 | 0.524 | **0.645** |
| Annotations with the same type in Events | 17/35 | 16/35 | **19/35** |
| Annotated N2AG frames labelled N2AG | 129/190 | 32/190 | **140/190** |

134 tests pass. _Pending: a fresh run on the HPC (the comparison above uses rebuilt inputs)._

---

### Entry 004 — Contact groups, similarity table, full following sequence, walking threshold

- **Date:** 2026-11-19
- **Video:** same 6-rat clip (rebuilt inputs, see Entry 003)
- **Pipeline:** cutie_composite, `contact_rules_v3: true`
- **Commit:** this commit

**Observations:**
- `CONTACT`: some types look alike. In this clip the second type of N2AG frames is N2B in
  78 % of them; N2N has SBS as second type in 29 % and SBS has N2N in 31 %. Of the 20 frames
  annotated as SBS that the program called N2N, 19 are lateral N2N.
- `CONTACT`: the Following sheet listed only the distinct types of an episode, so two N2AG
  events merged into one chase read as "N2AG".
- `CONTACT`: `follow_min_speed_bls` (0.15 BL/s) also decided the N2AG subtype; it is below
  90 % of all speeds (tracking jitter alone reaches ~0.25 BL/s), so almost every N2AG was
  "following".

**Action:**
1. Three levels: **group** (safest) > type > subtype. Table `groups` in
   `configs/contacts_postprocess_simple.yaml`, editable:

   | Group | Label | Type | Subtype |
   |---|---|---|---|
   | FTF | Face-to-face | N2N | frontal, perpendicular |
   | S2S | Side-to-side | N2N / SBS | lateral / — |
   | AGS | Anogenital sniffing | N2AG | classic |
   | BDS | Body sniffing | N2B | head / torso / rear |
   | CHS | Chase | N2AG / FOL | following / — |

   Columns `group` and `group_label` in Events and Brief, placed before `contact_type`
   (hierarchy order: group, group_label, contact_type, contact_label, subtype), from the
   event's type and most frequent subtype; `real_group` in PerFrame, before `real_type`
   (the group of the event the frame belongs to). One group per event, so totals by group
   never count a frame twice. New sheet **Groups** in `diagnostics.xlsx` with the totals.
   Names and codes are editable (`groups`, `group_labels`); all in English.
2. New sheet **Similarity** in `diagnostics.xlsx`: per type, % of its frames whose second
   type is each other type (how alike the types are in that video).
3. Following: `composition` keeps every event in order ("N2AG → N2AG") and the new column
   `sequence` adds the frames ("N2AG 84-101 → N2AG 111-123").
4. New `n2ag_walking_bls: 0.5` (cutie configs) decides N2AG following / classic; FOL keeps
   `follow_min_speed_bls`.

**Result:** 135 tests pass. Contact detection and the Entry 003 metrics are unchanged
(kappa 0.645). Groups in the clip: FTF 5 events (3.1 s), S2S 7 (3.3 s), AGS 3 (3.1 s),
CHS 6 (3.6 s), BDS 4 (1.6 s). The Following sheet is the CHS group seen as episodes. With 0.5
BL/s, 3 of the 7 N2AG events are classic (were all following); Following 6 -> 5 episodes.
_Pending: calibrate `n2ag_walking_bls` with the annotator marking which N2AG had both rats walking._

---

<!-- TEMPLATE: Copy this for new entries

### Entry NNN — [Short description]

- **Date:** YYYY-MM-DD
- **Video:** `path/to/video`
- **Pipeline:** reference / sam2_yolo / sam2_video
- **Config:** `configs/xxx.yaml` + any overrides
- **Commit:** `abc1234` (git short hash)

**Observations:**
- `TAG`: Description of what was seen
- `TAG`: Description of what was seen

**Problem:** Root cause analysis

**Action:** What was changed (code, config, or training)

**Result:** Did it improve? Better/worse/same on specific scenarios.
Metrics if available (detection count, ID switches, contact accuracy).

-->
