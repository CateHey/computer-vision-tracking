# 05 — How each contact is computed: a code-level reference

This document walks through the exact computation behind every contact label, from the
per-frame inputs to the evidence clips, and points at the code that does it. It is the
technical companion of [02_contact_detection.md](02_contact_detection.md), which gives the
design rationale. Everything here was checked against the code on `develop` on
2026-10-25 and against the first multi-animal run on Bunya (6 mice, 10 s, run
`2026-10-23_191936_cutie_composite`).

File references are `src/common/contacts_v2.py` unless another file is named; line numbers
are those of the commit this document was written for and will drift slightly.
Parameter values are the ones in `configs/hpc_cutie_composite.yaml` (`contacts:` block) or,
when a key is absent there, the default hard-wired in `ContactClassifier.__init__`
(line 979). Lengths are in **body lengths (BL)** and speeds in **BL/s** unless stated.

---

## 1. Inputs, once per frame and per animal

`cutie_composite` calls `ContactTrackerV2.update()` (line 1938) once per frame
(`src/pipelines/cutie_composite/run.py`, line 253) with three things per animal slot:

| Input | Source | Used for |
|-------|--------|----------|
| Binary mask | CUTIE index mask, one integer id per pixel, split per slot | mask contact, nose-in-mask, centroid |
| Centroid | Centre of the mask | distances, velocity, zone, dynamics |
| 7 keypoints with confidence | YOLO pose on the per-animal composite (`yolo26_v11.pt`) | every nose/rear/orientation measure |

Keypoint names, in model index order: `tail_tip, tail_base, tail_start, mid_body, nose,
right_ear, left_ear`. The classifier reads them by name (`get_keypoint`, line 598) and
ignores any keypoint below `min_keypoint_conf` = 0.3. The anatomical mapping matters:

- **rear / anogenital point = `tail_start`** (body–tail junction), not `tail_base`, which sits
  half-way down the tail. `tail_base` is only used for the informational tail–tail distance.
- **body axis** = `nose → mid_body → tail_start`.

YOLO detections are matched to slots by nearest mask centroid (`_map_detections_to_slots`);
a slot with no detection in a frame gets `single_detection=1`, and fewer than two confident
keypoints gets `missing_keypoints=1`. Both flags are written per frame.

---

## 2. Normalisation and state kept across frames

**Body length (`BodyLengthEstimator`, line 742).** The nose–`tail_start` distance of frames
where both keypoints have confidence ≥ 0.5 is collected (up to 500 values); the body length
is the 80th percentile, recomputed every 10 new observations. Until 25 frames have been seen
*and* 10 observations exist, the estimator returns `fallback_body_length_px` (120 px).
For a pair, `bl_ref = (BL_i + BL_j) / 2`, floored at 1 px (line 1061 onwards). Every
distance below is divided by `bl_ref`.

> Measured on the 6-mouse video: 164 px once warmed up. The first 25 frames (0.8 s at
> 30 fps) therefore divide by 120 px and overstate every BL distance by about 37 %.
> Set `fallback_body_length_px` in the config to the typical value of the arena.

**Velocity (`VelocityEstimator`, line 800).** Centroids are buffered (window 11) and the
velocity is the first derivative of a Savitzky–Golay fit (order 3), in px/s, then divided
by `bl_ref`. With fewer than 11 frames the last two centroids give a finite difference.

**Orientation (`best_orientation`, line 679).** Head direction = unit vector from the ear
midpoint to the nose (`head_orientation`, line 627), used when both ears are confident and
at least 4 px apart; otherwise body direction = unit vector `mid_body → nose` (or
`tail_start → nose`) (`body_orientation`, line 655).

**Schmitt triggers (`SchmittTrigger`, line 857).** A two-threshold latch per type and
direction: for distances it turns on below `enter` and off above `exit`, so a value that
hovers near one threshold does not flicker. Every latch is per pair and persists across
frames.

**Score shapes (lines 905–975).** Three piecewise-linear maps to [0, 1]:

| Function | Shape |
|----------|-------|
| `reversed_trapezoidal_score(x, near, far)` | 1 for x ≤ near, 0 for x ≥ far, linear between |
| `ramp_up_score(x, low, high)` | 0 for x ≤ low, 1 for x ≥ high, linear between |
| `trapezoidal_score(x, a, b, c, d)` | 0 outside (a, d), 1 on [b, c], linear ramps |

Non-finite inputs (a missing keypoint gives distance = ∞) always score 0.

**Shared mask border (`mask_contact_px`, line 936).** Boundary pixels of mask *i* that lie
within 4 px (`mask_contact_dilate_px`, elliptical dilation) of mask *j*, and vice versa,
averaged, then divided by `bl_ref` → `mask_contact_bl`. It is only computed when the
centroids are closer than 1.5 BL, otherwise it is 0. Mask IoU is still written but is
always 0 with CUTIE (masks are disjoint by construction); `high_mask_overlap` can therefore
never fire in this pipeline.

---

## 3. Per-pair geometry (`ContactClassifier.classify`, line 1061)

For *N* animals there are *N(N−1)/2* classifiers, one per unordered pair `i_j`
(slot indices, 0-based; the reports print them as `R(i+1)-R(j+1)`). Each frame, each
classifier computes, for its two animals only:

| Column | Definition |
|--------|------------|
| `nose_nose_dist_bl` | ‖nose_i − nose_j‖ / bl_ref |
| `centroid_dist_bl` | ‖centroid_i − centroid_j‖ / bl_ref |
| `nose_tailbase_ij_bl` | ‖nose_i − tail_start_j‖ / bl_ref (i's nose at j's rear) |
| `nose_tailbase_ji_bl` | ‖nose_j − tail_start_i‖ / bl_ref |
| `tail_tail_dist_bl` | ‖tail_base_i − tail_base_j‖ / bl_ref (informational) |
| `mask_contact_bl` | shared border length / bl_ref (section 2) |
| `velocity_i_bls`, `velocity_j_bls` | speeds |
| `velocity_alignment_cos` | cosine between the two velocity vectors |
| `orientation_alignment_cos` | cosine between the two orientation vectors |
| `zone` | `contact` if centroid distance < 0.30, `proximity` if < 1.0, else `independent` (`_determine_zone`, line 1272) |
| `t_i_on_j`, `t_j_on_i` | where the investigator's nose projects on the target's body axis: 0 at the nose, 0.5 at `mid_body`, 1 at `tail_start` (`_nose_axis_param`, line 1281) |

The body-axis parameter is what separates "sniffing the flank" from "sniffing the
anogenital area": the nose is projected on the two segments nose→mid_body and
mid_body→tail_start and the closer projection wins.

---

## 4. The five scores

Each type gets a score in [0, 1] every frame. Below, `Rt(x, a, b)` is the reversed
trapezoid and `Ru(x, a, b)` the ramp-up of section 2.

### 4.1 N2N — nose-to-nose (`_score_n2n`, line 1341)

```
soft  = Rt(nose_nose_dist_bl, 0.30, 0.45)          # contact_zone_bl_enter / _exit
latch = Schmitt(nose_nose_dist_bl, enter 0.30, exit 0.45)
soft  = max(soft, 0.5)  if latch is on
face  = 0.3 + 0.7 * (cos(orient_i, -orient_j) + 1) / 2   # 1 when they face each other
score = soft * face
```

Symmetric: no investigator is assigned.

### 4.2 N2AG — nose-to-anogenital (`_score_n2ag`, line 1360)

Computed in both directions; "i investigates j" uses i's nose and j's rear:

```
s_ij  = Rt(nose_tailbase_ij_bl, 0.30, 0.45), raised to >= 0.5 while its own latch is on
s_ij *= gaze_i      = 0.4 + 0.6 * (cos(orient_i, nose_i -> rear_j) + 1) / 2   # (_gaze_alignment, line 1320)
s_ij *= rear_zone_i = Ru(t_i_on_j, 0.45, 0.65)      # social_anogenital_split 0.55 +- 0.10
```

The larger of `s_ij`, `s_ji` is the score and its direction is the **investigator**
(`investigator_role` = `i` or `j`). The rear-zone factor is what stops a nose at the
flank, which can be within 0.45 BL of `tail_start`, from counting as anogenital.

### 4.3 FOL — following (`_score_fol`, line 1418; `_score_fol_direction`, line 1454)

Following is defined as a **non-contact** behaviour, so it is zero whenever the animals
touch: if the smallest of the three nose–body distances is < 0.5 BL
(`follow_no_contact_bl`) or `mask_contact_bl` > 0.35, FOL = 0. Otherwise, per direction
("i follows j"):

```
path_j = last 12 positions of j's tail_start (centroid when missing -> fol_used_centroid=1)
dist   = Rt(min distance from nose_i to path_j in BL, 0.40, 0.60)   # follow_radius_bl / _exit
orient = Ru(cos(orient_i, nose_i -> centroid_j), 0.40, 0.80)         # follow_alignment_cos 0.6 +- 0.2
speed  = 1 if speed_i >= 0.15 BL/s else Ru(speed_i, 0, 0.15)         # follow_min_speed_bls
s_ij   = dist * orient * speed
```

The larger direction wins and names the follower. FOL is the only "rare" type: it is
accepted at 0.35 instead of 0.50 (section 5).

### 4.4 SBS — side-by-side (`_score_sbs`, line 1494)

```
dist   = Rt(centroid_dist_bl, 0.30, 1.00)
speed  = Rt(max(speed_i, speed_j), 0.25, 0.75)            # sbs_max_velocity_bls 0.5 x 0.5 / x 1.5
align  = Ru(|orientation_alignment_cos|, 0.55, 0.85)      # sbs_parallel_cos_min 0.7 +- 0.15 (either way round)
border = Ru(mask_contact_bl, 0.20, 0.70)                  # sbs_contact_bl_exit, 2 x sbs_contact_bl_enter
score  = geometric mean(border, dist, speed, align)       # each floored at 1e-6
score  = 0 if dist == 0 or align == 0                      # the "hard" factors
latch  = Schmitt(mask_contact_bl, enter > 0.35, exit < 0.20), fed only when dist >= 0.5 and both hard factors pass
score  = max(score, 0.5) if latch is on
```

The geometric mean lets one weak factor lower the score without zeroing it; the hard
factors stop a tail-to-tail or perpendicular touch from being called side-by-side.
Symmetric: no investigator.

### 4.5 N2B — nose-to-body, the catch-all (`_score_n2b`, line 1532)

```
if sbs_score >= 0.5: score = 0                             # n2b_sbs_suppress
in_ij = 1 if the pixel under nose_i is inside mask_j else 0
guard = 0.7 if any of nose_nose, nose_tailbase_ij, nose_tailbase_ji < 0.45 BL else 1.0
s_ij  = in_ij * guard * gaze_i(toward centroid_j)
```

Larger direction wins and names the investigator. Because the test is "nose pixel inside
the other's mask", N2B needs the masks to be accurate at the contact point.

---

## 5. One label per pair per frame (`ScoreMap`, line 119)

`argmax_type` (line 140) walks the types in priority order `N2N, N2AG, FOL, SBS, N2B` and
keeps the highest score that reaches its threshold: 0.50 (`activation_threshold`), or 0.35
for FOL (`activation_threshold_rare`). Equal scores go to the earlier type. If nothing
reaches its threshold the frame is `none` (printed as NC later).

`secondary_type` (line 197) is the best *other* type scoring ≥ 0.40
(`secondary_threshold`; 0.35 for FOL). It is informational: it lets a 0.52 N2N / 0.49 N2AG
frame be seen as ambiguous.

The frame also records: `family` (investigative: N2N, N2AG, N2B; affiliative: SBS;
non-contact: FOL), `investigator_role` (N2AG, FOL, N2B only), and the **dynamics** channel
(`_compute_dynamics`, line 1584): the change of centroid distance per second; |Δ| < 0.10 BL/s
is `stable`, negative is `closing`, positive `separating`; the **mover** is the animal whose
velocity projected on the line joining the centroids explains ≥ 60 % of the change
(`_dynamics_mover`, line 1630); during contact, separating while the non-investigator is
the mover is written as `rejects`, otherwise `accepts` (`_reciprocity`, line 1667).

All of this is one row of `contacts_per_frame.csv` per pair and frame (`PerFrame` sheet of
`results.xlsx`), including the five `score_*` columns, the distances, the quality flags and,
since October 2026, the pixel positions of both centroids and noses used by the evidence clips.

---

## 6. Frames to bouts (`BoutManager`, line 1697)

The tracker groups consecutive frames of the same `(pair, type)`: a bout stays open across a
gap of up to 3 frames of another label (`bout_max_gap_frames`) and is kept when closed only if
it has at least `bout_min_frames_<type>` frames: 6 for N2N, N2AG, SBS, N2B and 12 for FOL
(`_close_bout`, line 1775). The bout's **initiator** is the majority of the frame-level
investigator roles (`Bout.finalize_metrics`, line 452). Bouts go to `contact_bouts.csv`
(`Bouts` sheet). They are a diagnostic view; the reportable events come from section 7.

---

## 7. Frames to events (`scripts/postprocess_contacts_simple.py`)

`run_postprocess` (line 1178) reads `configs/contacts_postprocess_simple.yaml` and applies,
**per pair** (`compute_labels`, line 296), three rules to the raw per-frame labels:

1. **Majority vote** over a sliding window of 5 frames (`apply_majority_vote`, line 355);
   ties keep the current label, then prefer a contact over none by priority.
2. **Gap bridging**: a gap of ≤ 3 frames between two runs of the same type is filled
   (`apply_gap_bridging`, line 403).
3. **Minimum duration**: runs shorter than 0.3 s are dropped (9 frames at 30 fps, 8 at 25)
   (`apply_min_bout_filter`, line 443).

What survives becomes the cleaned label (`real_type`), and every maximal run of one label
for one pair is an **event** (`extract_events`, line 520): `start_frame`, `end_frame`,
`duration_sec = (end − start + 1) / fps`, `peak_score` = the maximum of that type's score
column over the run, `investigator_slot` = majority of the frame roles **among frames whose
raw label is the event's type** (symmetric types N2N and SBS never get one). Runs of `none`
are written too, as NC events, so every pair's timeline is fully covered. `event_id` is unique
over the session and ordered by start frame, then pair.

> Two minimum durations therefore exist: the tracker's per-type bout minimum (6 / 12 frames)
> and the post-processing minimum (9 frames at 30 fps). The events are derived from the raw
> labels, not from the bouts, so a 10-frame FOL run is an event although the tracker would
> have discarded it as a bout. Pick one definition before reporting.

---

## 8. Several animals: what pairwise independence means

With six animals every frame yields 15 rows, one per pair, and each classifier sees only
its own two masks and keypoint sets. Consequences, all visible in the 6-mouse run:

- One animal can be in several contacts at once. Frames 182–202: R3-R4 N2N, R2-R4 N2N and
  R2-R3 SBS overlap, with R4's nose 0.17–0.18 BL from both other noses. No rule says a nose
  belongs to one contact only; whether a three-way huddle is one event or three is a
  decision the pipeline does not make.
- An investigator in one pair can be a receiver in another at the same time
  (R3 sniffs R5 while R2 sniffs R3, frames 80–100).
- Contact time in the summaries is summed over pairs ("8.8 % of pair-time" in the event log
  means 13.3 s of contact over 15 pairs × 10 s).
- `merged_state` (two animals in one mask) is never set by `cutie_composite`, because CUTIE
  emits disjoint masks; identity errors show up as swaps, not merges, and only the overlay
  video reveals them.

---

## 9. Evidence that the computation did what this says

**Per frame.** `PerFrame` carries, for every pair and frame, the five scores, the raw and
cleaned labels, the distances behind them and the flags; any event can be re-derived from
it by hand. Example from run `2026-10-23_191936` (6 mice, 300 frames at 30 fps):

| Event | Pair | Frames | Peak / mean score | Supporting measure |
|-------|------|--------|-------------------|--------------------|
| N2N | R4-R5 | 124–161 (1.27 s) | 1.00 / 0.90 | mean nose–nose distance 0.16 BL |
| N2AG | R3-R5 | 82–191 (3.67 s) | 1.00 / 0.74 | investigator R3; secondary SBS |
| SBS | R4-R5 | 225–267 (1.43 s) | 0.90 / 0.57 | mean nose–nose 0.34 BL; secondary N2N |
| FOL | R2-R3 | 239–248 (0.33 s) | 1.00 / 0.80 | mean nose–nose 1.33 BL, i.e. no contact, as FOL requires |

**Per run.** `logs/run.log` reports the tracking quality the contacts depend on:
`Handoff: 6/6 rats` (every animal reached CUTIE), pose detected in 300/300 frames for each
animal, 0 % keypoint carry-over, 0.3 % mask carry-over. 86 raw bouts became 27 tracker bouts
and 46 cleaned events (30 NC, 7 N2AG, 5 SBS, 3 N2N, 1 FOL). The `Parameters` sheet records
the smoothing, gap and minimum-duration values that were applied.

**Per event, by eye.** `scripts/validate_contacts.py sample` draws a stratified sample per
type (`stratified_sample`, line 244: duration tertiles, seeded), groups events that overlap
in time into scenes (`build_scenes`, line 501) and writes one slow-motion clip per scene:
the two animals of every active event are circled in their overlay colours and named in
the banner, the raw per-frame label and score are printed at the bottom, and `review.xlsx`
has one row per event with `clip_from` / `clip_to`. A reviewer marks each row `real`,
`flicker`, `wrong_type` (+ `correct_type`), `tracking_error` or `unsure`;
`validate_contacts.py score` turns that into precision per type with Wilson 95 % intervals
(`wilson_ci`, line 122; `compute_summary`, line 878), flicker rate by duration and the
precision that minimum-duration or minimum-score thresholds would give. Only the rows of
the stratified sample enter those figures; events listed because they overlap a sampled
one (`in_sample = FALSE`) are summarised apart, since they over-represent busy moments.

**Per-animal view, cm calibration and a keypoint check (added 2026-10-27).**

- `per_animal_summary.csv` (`PerAnimal` sheet, "Per-animal summary" block of `event_log.txt`,
  `per_animal` in `session_summary_real.json`) has one row per animal, built from the cleaned
  labels of every pair that includes it: events by type (`n_events_total`, `n_events_<TYPE>`),
  `contact_time_sec` (frames in which at least one of its pairs is in contact, FOL excluded; a
  union, so two simultaneous contacts count once), `time_2plus_contacts_sec` (frames with two or
  more pairs in contact at once, LMT's group3), `n_as_investigator` / `n_as_receiver` (asymmetric
  events N2AG, N2B, FOL; the follower counts as investigator for FOL) and `fol_time_sec`. Pair
  events are unchanged. Function: `per_animal_summary` in `scripts/postprocess_contacts_simple.py`.
- Contact thresholds are set in BL; `session_summary.json` metadata now carries
  `body_length_px_median` (median of the per-animal estimates at the end of the run), `cm_per_px`,
  `cm_per_px_source` and `thresholds_cm` (`contact_enter`, `contact_exit`, `proximity` =
  BL threshold x body length in px x `cm_per_px`). `cm_per_px` is `contacts.cm_per_px` when it is
  set (`config`) and `contacts.reference_body_length_cm` / `body_length_px_median` otherwise
  (`estimated_from_body_length`, 9.5 cm for an adult C57BL/6J nose to tail base). The estimate
  assumes the animals are of the reference size; a measured arena scale replaces it. The values
  appear in the `calibration` section of the `Parameters` sheet and in the `event_log.txt` header
  (`Contact distance: 0.30 BL (≈ 2.7 cm; ...)`).
- `nose_off_mask_i`, `nose_off_mask_j` and `nose_off_mask` (1 if either) are per-frame quality
  flags: the nose keypoint of an animal has no pixel of that animal's own mask within
  `nose_off_mask_radius_px` (3) px, i.e. the keypoint is misplaced, which is most likely during
  contact (B2 in OPEN_ISSUES). The count is in `quality_flags` of the summary and
  `scripts/validate_contacts.py` reports it as `frac_nose_off_mask` per sampled event. A missing
  nose or mask is not flagged.

**What the 6-mouse run says about sensitivity.** Of 4 500 pair-frames, 579 carry a raw
contact label (298 N2AG, 118 SBS, 98 N2N, 34 N2B, 31 FOL). N2B is the rarest and none of
its 34 frames survived post-processing. 14 frames have the two masks physically touching
(shared border > 0.2 BL) and no label at all; in half of them a nose is within 0.34 BL of
the other body. Those are the borderline cases a reviewer should expect to find missing
rather than mislabelled.

---

## 10. Known limitations and open decisions

| Item | Where | Status |
|------|-------|--------|
| First 25 frames use a 120 px body length (37 % off on the 6-mouse video) | `BodyLengthEstimator.current`, line 790 | set `fallback_body_length_px`, or warm up from the first observations |
| Bout minimum (6 / 12 frames) ≠ event minimum (9 frames) | sections 6–7 | choose one; the events are what is reported |
| No cross-pair consistency (one nose in two N2N) | section 8 | a rule or a per-frame flag for multi-contact frames |
| SBS can latch on the very first frames (velocity history empty, body length provisional) | `_score_sbs` + section 2 | review such events with extra care; a warm-up exclusion is an option |
| Mask IoU and `high_mask_overlap` are meaningless with CUTIE | section 2 | keep as zero or drop the column |
| `min_keypoint_conf` for the contacts is the code default 0.3 (not in the YAML) | `ContactClassifier.__init__` | add to the config if it is to be tuned |
| `stale_keypoints` was never set: carried-over keypoints looked like detections | `_carry_over_keypoints`, `classify` | fixed 2026-10-25: `Detection.carried` is propagated to the flag |
| N2B needs the nose *pixel* inside the other mask; with disjoint CUTIE masks it almost never fires, so flank sniffing is under-counted | `_score_n2b`, line 1532 | use the distance from the nose to the other mask (e.g. within 4 px, as `mask_contact_px` does) |
| Contact = nose within 0.30 BL (about 2.7 cm for a 9.5 cm mouse), latched until 0.45 BL; this is proximity, not touch | `contact_zone_bl_enter/_exit` | the cm values are reported (section 9); measure `cm_per_px` for the arena and decide the distance the paper calls contact |
| One pair-event per pair: an animal in two simultaneous contacts contributes two events and twice the contact time | section 8 | `per_animal_summary.csv` gives the per-animal union (section 9); define the per-group count before reporting |
