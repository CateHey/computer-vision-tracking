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
