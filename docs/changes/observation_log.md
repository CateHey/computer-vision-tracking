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
