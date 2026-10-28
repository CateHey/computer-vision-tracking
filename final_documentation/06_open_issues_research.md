# 06 — Open issues checked against the literature and reference tools

What the published tools do about each point in [OPEN_ISSUES.md](../OPEN_ISSUES.md), what
that implies for this pipeline, what was implemented (switchable, counts unchanged by
default) and what still needs the lab's decision. Checked 2026-10-25.

## Sources actually read

| Source | What it is | What we took from it |
|--------|------------|----------------------|
| de Chaumont et al. 2019, *Live Mouse Tracker*, Nature Biomedical Engineering, **Supplementary Information** (`41551_2019_396_MOESM1_ESM.pdf`, "Behavioural event extraction", pp. 15–18) | The most used open system for groups of up to 4 mice; peer-reviewed event definitions | exact geometric definitions and units of contact, nose-nose, nose-anogenital, side-side, follow, group2/3/4 |
| `fdechaumont/lmt-analysis`, `LMT/lmtanalysis/ParametersMouse.py`, `BuildEventOralOralContact.py`, `BuildEventOralGenitalContact.py`, `BuildEventSideBySide.py`, `BuildEventFollowZone.py`, `BuildEventGroup3.py`, `Animal.py`, `Measure.py` | LMT's open-source event code | thresholds as implemented (15 px, 30 px, scale 10/57 cm/px), follow minimum 7 frames and gap merge 10 frames, how groups and dyads coexist |
| Segalin et al. 2021, *MARS*, eLife 63720 (Methods, Table 3, Fig. 8) | Pose-based classifier for pairs | annotator definitions of close investigation (face / body / genital), the body-region boundaries, HMM + 3-frame smoothing |
| SimBA docs, `Scenario1.md` step 8 | Random-forest behaviour classifier | "minimum bout length (ms)" exists as a user setting, no default |
| A-SOiD, GitHub README and docs | Active-learning classifier | no bout minimum stated in the docs (the 200 / 400 ms figures in `docs/contacts/threshold_research.md` come from the paper, not re-verified here) |
| JAX Mouse Phenome Database (body length, C57BL/6J) | Reference strain measurements | nose-to-tail-base ≈ 9.7–10.1 cm at 20 months (young adults are a little shorter) |

Not reachable: de Chaumont 2012 (ResearchGate 403), the eNeuro 2024 snout-tracking paper (file over the fetch limit).

## What Live Mouse Tracker defines (verbatim units)

- *contact*: "the two masks have one common pixel".
- *contact nose-nose*: "the two nose points are less than 15 pixels (26 mm) from one another".
- *contact nose-anogenital*: "the nose point of the first mouse is within 15 pixels (26 mm) from the tail point of the other mouse".
- *contact side-side*: "the side of a mouse is within 30 pixels (52 mm) from the side of another mouse; the two animals are oriented in the same direction" (and *side-side opp* for opposite directions).
- *follow*: "Both mice move at a speed higher than 5 pixels per frame, with an angle between their direction vectors smaller than 45°, with the distance between their centres of mass shorter than two mean body lengths". In the analysis code: follower's centre within 2.5 cm of the followed's past position, speed ≥ 10 cm/s, events shorter than 7 frames removed, gaps ≤ 10 frames merged.
- *group2 / group3 / group4*: "the mouse is in contact with one and only one other mouse" / "with two and only two other mice" / "with three other mice". Dyadic events keep being produced while a group exists; LMT additionally offers "exclusive" timelines where overlapping dyadic types are removed.
- No minimum duration for the sniffing contacts: an event is any run of frames.
- Scale: 1 px = 10/57 cm ≈ 0.175 cm; `Measure.oneSecond = 30` (30 fps).

MARS does not use fixed distances: annotators label "close investigation" by eye ("actively sniffing the intruder anywhere on its body or tail") and the classifier learns from 270 pairwise-distance features. To split face / body / genital investigation MARS uses the resident's nose position relative to two boundaries: midway between the intruder's nose and neck, and midway between its hips and tail base. Classifier output is smoothed with an HMM and a 3-frame moving average; no minimum bout length is stated.

## Item by item

### A1 — what distance counts as contact

**Finding.** LMT's threshold for nose-nose and nose-anogenital is 26 mm between keypoints. Ours is 0.30 body lengths to enter and 0.45 to leave. With the body length measured on the 6-mouse video (164 px) and a mouse of about 9 cm nose-to-tail-base, 0.30 BL ≈ 2.7 cm: the same number as LMT. The exit threshold (≈ 4 cm) only keeps a contact alive once started (hysteresis), which LMT does not have. MARS gives no distance at all.

**Implication.** The current threshold is defensible by precedent. What is missing is the px/cm calibration of this arena to state it in cm in the paper.

**Calibration cross-check (2026-10-27).** The arena floor spans 822 px horizontally in
`multimicevideo.avi` (columns 60–882 of the 960 px frame, found from the brightness
profile). For a 50 × 50 cm arena (the Live Mouse Tracker standard) that gives
1 px = 0.061 cm, so the measured body length of 164 px is 10.0 cm, which is exactly the
JAX reference for an adult C57BL/6J. Two independent estimates agree, so the body-length
calibration is trustworthy. Resulting thresholds: contact enters at 0.30 BL = 3.0 cm, leaves
at 0.45 BL = 4.5 cm, proximity 1.0 BL = 10 cm; LMT's nose rule is 2.6 cm. If the arena is
40 cm instead, every figure scales by 0.8 (BL = 8 cm, contact 2.4 cm).

**Action.** Confirm the arena's physical side length and set `contacts.cm_per_px` (50 cm →
0.061) so the reports print exact cm; otherwise the body-length estimate (`reference_body_length_cm`)
is used. To match LMT's 26 mm exactly, set `contact_zone_bl_enter: 0.26` (and `_exit: 0.40`);
keeping 0.30 BL means "within 3 cm", which must then be the wording in the methods. *Decision: lab.*

### A2 — several animals: per-pair counting

**Finding.** LMT does exactly what we do: every pair is scored on its own, and dyadic events are not suppressed when three animals touch. Group configurations are reported as separate per-animal events (group2, group3, group4).

**Implemented.** Each per-frame row now carries `n_contacts_i` / `n_contacts_j`: how many animals each of the two touches in that frame (FOL excluded, as it is non-contact). `session_summary.json` adds `multi_contact` (frames in which some animal touches two or more). This is LMT's group2/group3 information with our labels; pair events are unchanged.

**Still to decide.** Whether the paper reports per-pair events (LMT default), per-animal participation, or cluster events. The data for all three is now in the table. *Decision: lab.*

### A3 — two minimum durations

**Finding.** No consensus: LMT applies none to sniffing contacts and 7 frames (0.23 s) to following, with gaps ≤ 10 frames merged; MARS relies on an HMM; SimBA leaves the minimum to the user; our own review (`docs/contacts/threshold_research.md`) cites 200–400 ms. Our 0.3 s uniform minimum sits inside that range; the inconsistency is only that the tracker's own FOL minimum (0.4 s) is bypassed by the event stage.

**Implemented.** `configs/contacts_postprocess_simple.yaml` → `min_bout.by_type` (seconds per type) applied after the global minimum. Empty by default, so nothing changes; `by_type: {FOL: 0.4}` aligns the reported events with the tracker. The values applied are written to the `Parameters` sheet.

**Decision (2026-10-27).** `by_type: {FOL: 0.4}` is now the default: it makes the reported
following events consistent with the tracker's own 12-frame bout minimum and with A-SOiD's
400 ms default, while LMT's 7 frames would be looser. Sniffing contacts keep 0.3 s. The
only FOL event of the 6-mouse 10 s run (10 frames) no longer counts; it was below the
tracker's own definition anyway.

### A4 — nose-to-body almost never fires

**Finding.** LMT has no body-sniffing type; its generic "contact" is "masks share a pixel", i.e. touching masks, not a nose inside the other mask. MARS splits investigation into face / body / genital by where the nose falls along the other animal's body, with the genital region starting midway between hips and tail base. Our N2B needs the nose *pixel* inside the other mask, which with disjoint CUTIE masks requires true overlap (34 raw frames in 4 500; 0 events), and our anogenital region starts at 55 % of the nose→tail_start axis, i.e. at mid-body, earlier than MARS's boundary. Together they explain 7 N2AG events and 0 N2B in the run: body sniffing is either dropped or counted as anogenital.

**Implemented.** `contacts: n2b_nose_mask_radius_px` (default 0 = current behaviour). With r > 0, a nose within r px of the other mask counts as touching it, LMT-style. `social_anogenital_split` is now an explicit config key (0.55).

**Decision (2026-10-27), pending one validation round.** Defaults set to
`n2b_nose_mask_radius_px: 6` (0.37 cm at 0.061 cm/px, the scale of the keypoint error; LMT's
"masks touching" criterion applied to a nose that cannot fall inside a disjoint mask) and
`social_anogenital_split: 0.8` (MARS's genital boundary is midway between the hips and the
tail base; on our nose→mid_body→tail_start axis the hips sit near 0.75, so that midpoint is
about 0.85–0.9; the ramp 0.7–0.9 around 0.8 reproduces it). Both are config keys; the
previous behaviour is `0` and `0.55`. Validate on the 6-mouse clip: N2B events should appear
and some N2AG should move to N2B; keep the setting the reviewer confirms. The old run
(`2026-10-23_191936`, 7 N2AG / 0 N2B) is the baseline.

**Validation run (2026-10-28, job 30737285, run `2026-10-28_091052`).** Tracking was practically
identical to the baseline (4 389 of 4 500 pair-frames carry the same raw label; centroid and
nose distances agree within 0.02 BL in 90 % of the frames; the rest are the first frames, where
the body length now warm-starts), so the differences are the configuration's. Raw labels: 42
frames `none → N2B` and 34 `N2AG → N2B` (body sniffing moved out of the anogenital class, as
intended), 9 `N2AG → none`, 7 `N2AG → SBS`, only 6 frames newly N2AG. Events: N2B 0 → 2
(0.9 s), SBS 5 → 6, N2N 3 → 3, FOL 1 → 0 (the 0.4 s floor), **N2AG 7 → 10 although its time
fell from 7.6 s to 7.0 s**: the nose sliding between flank and rear now alternates N2AG and N2B,
which cuts one long anogenital event into two or three short ones. The new split is doing what
MARS does, but counting sub-types as separate events inflates the count whenever the nose
crosses the boundary. MARS avoids this by scoring one "close investigation" class and only
then labelling frames face / body / genital. **Decision:** report *investigation bouts* (N2N,
N2AG and N2B merged, gaps ≤ 3 frames) as the event count, with the seconds per sub-type inside
each bout; keep the sub-type events for the duration figures. Implemented as an additional
table (`investigation_bouts.csv`, sheet `Investigation`), so nothing existing changes.


### B1 — provisional body length

**Finding.** LMT normalises approach distances by the animal's *mean body length* over the recording; MARS works in cm. Nobody uses a fixed fallback.

**Implemented.** The estimator now returns the median of the first real observations as soon as three exist, instead of the fallback, so the provisional phase is a few frames instead of 25. `fallback_body_length_px` is set to 165 in the HPC config (the value measured on the 6-mouse video) and remains the last resort.

### B2 — keypoints during contact

**Finding.** Occlusion in close contact is the known weak point of every multi-animal pose tool (maDLC and SLEAP papers both say so). Our composite approach trades identity robustness for keypoint error at the contact point. No reference tool solves it; MARS and LMT validate by manual review, which is what `validate_contacts.py` provides.

**Action.** None in code. Report the reviewed precision per type.

### B3 — cross-pair consistency

Covered by A2: `n_contacts_*` makes multi-contact frames visible, and the review clips show all pairs of a scene. A hard rule (one nose, one N2N) has no precedent in LMT or MARS; not implemented.

### B4, B5 — inert IoU column, missing config key

`min_keypoint_conf` is now explicit in both cutie configs (0.3). The IoU column stays, documented as inert with CUTIE.

## Validation plan for the proposed values

**Sensitivity found on the 6-mouse clip (2026-10-28, `scripts/sensitivity_analysis.py`).** N2N and
N2AG counts are stable across the activation threshold 0.40–0.70 (3 and 7–9 events), but all 5
SBS events vanish at 0.55: their scores sit at 0.50–0.57 because the side-by-side latch floors
the score at exactly the activation threshold (`_score_sbs`: `max(score, 0.5)` while the
Schmitt trigger is on). SBS is therefore a latch-driven label, not a score-driven one. Options:
keep it and say so in the methods, raise the latch floor to e.g. 0.6 so SBS survives a stricter
threshold, or lower the SBS activation. Minimum-duration sensitivity is as expected for a 10 s
clip (N2AG 2–13 events over 0.2–1.0 s). *Decision: lab; report the table either way.*

1. Re-run the 6-mouse 10 s clip with the candidate settings (`n2b_nose_mask_radius_px: 6`, `social_anogenital_split: 0.75`, `by_type: {FOL: 0.4}`) via `OVERRIDES` or a copy of the config.
2. `validate_contacts.py sample` on both runs; review all N2B and N2AG rows.
3. Compare per-type precision and the N2B / N2AG split; adopt the setting with the better reviewed precision and write the chosen cm thresholds into the methods.
