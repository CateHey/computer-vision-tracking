# Open issues and decisions

Living checklist of what still needs a decision or a fix before the contact counts are
reported. Each item says why it matters, where it lives and what we propose. Tick items off
here when they are resolved (and say how). Technical background:
[final_documentation/05_contact_computation.md](final_documentation/05_contact_computation.md).

Last update: 2026-10-25. Literature check and proposed fixes per item:
[final_documentation/06_open_issues_research.md](final_documentation/06_open_issues_research.md).

## A. Decisions that change the contact counts (need the lab's call)

| # | Item | Why it matters | Where | Proposed action | Status |
|---|------|----------------|-------|-----------------|--------|
| A1 | **What distance counts as contact.** A contact starts when the nose is within 0.30 body lengths (about 2 cm for an 8 cm mouse) and lasts until 0.45. That is proximity, not touch: two noses 2 cm apart count as nose-to-nose. | Moves every count and duration more than any bug. | `configs/hpc_cutie_composite.yaml` → `contact_zone_bl_enter` / `_exit`; `05` §4 | Calibrate px per cm for the arena, decide the distance the paper calls "contact", set both thresholds, re-run. **Research:** 0.30 BL ≈ 2.7 cm matches Live Mouse Tracker's 26 mm nose-nose / nose-anogenital rule (06 §A1); only the px/cm calibration is missing. | open (lab) |
| A2 | **Per-pair counting with several animals.** Each pair is classified on its own. An animal in two simultaneous contacts (R4 with R2 and R3 at 6.1 s in the 6-mouse run) yields two events and twice the contact time. | Per-animal totals and "contact time" depend on the definition. | `src/common/contacts_v2.py` (one classifier per pair); `05` §8 | Define, in writing, the per-animal count (events the animal takes part in) and the per-group count (distinct simultaneous clusters as one event?). **Research:** LMT also scores pairs independently and reports group2/group3 separately (06 §A2). **Done:** per-frame `n_contacts_i/j` and `multi_contact` in the summary. Choose the paper's definition. | data ready, decision open |
| A3 | **Two minimum durations.** The tracker keeps bouts of ≥ 6 frames (FOL 12); post-processing keeps runs of ≥ 0.3 s (9 frames at 30 fps) and the reported events come from the latter. A 10-frame following run is reported although the tracker would drop it. | Inconsistent definition of an event between `Bouts` and `Events`. | `contacts: bout_min_frames_*` vs `configs/contacts_postprocess_simple.yaml` → `min_bout.duration_sec`; `05` §6–7 | Choose one: either report only post-processed events (and align the per-type minimums there, e.g. FOL 0.4 s), or make post-processing per-type. **Research:** no consensus (LMT none / FOL 7 frames; MARS HMM; SimBA user-set) (06 §A3). **Done:** `min_bout.by_type` in the YAML, empty by default; `{FOL: 0.4}` aligns events with the tracker. | mechanism ready, decision open |
| A4 | **Nose-to-body almost never fires.** N2B requires the nose *pixel* to fall inside the other animal's mask; CUTIE masks are disjoint, so this needs real overlap. 6-mouse run: 34 raw frames in 4 500, 0 events; 14 frames with masks touching and no label. Flank sniffing is under-counted or absorbed. | A whole contact type is effectively missing from the counts. | `ContactClassifier._score_n2b`; `05` §4.5 and §9 | Score N2B by distance from the nose to the other mask (nose within ~4 px / 0.05 BL of the mask, as `mask_contact_px` already does for SBS) instead of pixel membership. **Research:** LMT contact = masks touching; MARS genital region starts midway hips–tail base (06 §A4). **Done:** `n2b_nose_mask_radius_px` (0 = old behaviour) and explicit `social_anogenital_split`. Try 6 px and 0.75, then review clips. | mechanism ready, validation open |

## B. Known measurement limits (fix or keep in mind)

| # | Item | Why it matters | Where | Proposed action | Status |
|---|------|----------------|-------|-----------------|--------|
| B1 | **Body length is provisional for the first 25 frames** (120 px fallback vs 164 px measured on the 6-mouse video). Distances in the first 0.8 s are overstated by ~37 %. The SBS event at 0.00–0.33 s falls in that window. | Early events are less reliable; a warm-up exclusion or a better fallback removes the bias. | `BodyLengthEstimator.current`; `contacts: fallback_body_length_px` | Set `fallback_body_length_px: 165` for this arena, or start from the median of the first observations. **Done:** estimator warm-starts from the median of the first 3 observations; HPC config sets `fallback_body_length_px: 165`. | fixed 2026-10-25 |
| B2 | **Keypoints are least reliable exactly during contact.** YOLO sees each animal on a composite where the others are erased, and the erased region runs through the contact point. | Contact-frame geometry carries more error than free-moving frames. | `src/pipelines/isolated_composite/composition.py` | No code fix; the review clips are the evidence that matters. Report the reviewed precision per type with its confidence interval. | by design |
| B3 | **No cross-pair consistency check.** Nothing prevents one nose from being nose-to-nose with two animals at once (see A2). | Impossible configurations are not flagged. | `ContactTrackerV2.update` | **Done** via `n_contacts_i/j` (A2). A hard exclusivity rule has no precedent in LMT or MARS; not implemented. | flag done |
| B4 | **Mask IoU and `high_mask_overlap` are always 0** with CUTIE (disjoint masks). | The column suggests a check that cannot trigger. | `05` §2 | Drop the column or document it as inert. | cosmetic |
| B6 | **SAM3 initialisation is not deterministic.** The same 6-mouse clip handed 6/6 animals to CUTIE on 2026-10-23 and 5/6 on 2026-10-25 (job 30588112): touching mice get merged at the start and the slot is never filled later. A 5/6 run silently drops that animal's 5 pairs. | Counts of a partial run are wrong without warning. | `select_sam3_init` in `src/pipelines/cutie_composite/run.py`; `sam3:` block in the configs | **Done 2026-10-25:** the SAM3 phase is retried with lower thresholds (0.4, 0.3) and later start frames (30, 60) until every animal is found, and the run aborts (`require_all_animals: true`) if the handoff still misses one. Log lines `SAM3 init attempt ...`. | fixed, to be confirmed on Bunya |
| B5 | **`min_keypoint_conf` for contacts is the code default (0.3)** and is not in the YAML. | Low-confidence noses (0.3–0.5) enter the distances; the body-length estimator uses 0.5. | `ContactClassifier.__init__` | **Done:** `min_keypoint_conf: 0.3` explicit in both cutie configs. | fixed 2026-10-25 |

## C. Done in October 2026 (for the record)

| # | Item | Fixed in |
|---|------|----------|
| C1 | `stale_keypoints` was never set: carried-over keypoints looked like detections | commit `c25838a` |
| C2 | Precision mixed the stratified sample with overlapping "extra" events | `c25838a` |
| C3 | SBS events got a spurious investigator from one stray N2AG frame; symmetric types never have one now | `8895a27` |
| C4 | The post-processing YAML was ignored by the pipeline (built-in 5 / 3 / 0.3 s applied); it is read now, with the values that were in use | `8895a27` |
| C5 | Background config keys (`num_samples`, `seed`) were silently ignored | `8895a27` |
| C6 | Evidence clips: one clip per scene of overlapping events, pair circled in its colours, R-labelled file names, one review row per event | `6705a44` |
| C7 | Colour names in clips were wrong for R4 (cyan) and R6 (yellow) | `6705a44` |

## D. Operational

| # | Item | Action | Status |
|---|------|--------|--------|
| D1 | The Hugging Face token was pasted in a chat log | Delete it at https://huggingface.co/settings/tokens, create a new Read token, rewrite `~/.hf_token` on Bunya | pending |
| D2 | Untracked copies of files deleted in the cleanup reappeared in the laptop checkout (2026-10-23 14:57: `docs/archive`, `configs/deprecated`, debug scripts, spare YOLO weights, `docs/setup/hpc.md`) | They are not in git. `git clean -n` lists them; delete when sure. Never `git add -A`. | pending |
| D3 | Re-run the 6-mouse 10 s job to get clips with the pair highlighted (positions are only saved by runs after 2026-10-23) | `git pull && sbatch slurm/run_cutie.sbatch data/raw/multimicevideo.avi 6 10` | pending |
| D4 | Full-length runs (`multimicevideo.avi` 10:50, `twomousevideo.mp4` 20:00) held every frame in RAM (54 GB and 100 GB at 960x960, 30 fps) | **Done:** phase 2 streams the video (`scan.preload_all_frames: false`, the default) and only the first 125 frames are in RAM, so memory is now about constant. Time still grows with the length: `sbatch --time=12:00:00 slurm/run_cutie.sbatch data/raw/multimicevideo.avi 6` | done (full-length run on Bunya not yet checked) |
| D5 | Repo still says "rat" in names (`rattrack`, R1…R6, titles); the animals are mice | Rename only if wanted for the paper; the SAM3 prompt is already "mouse" | decide |
