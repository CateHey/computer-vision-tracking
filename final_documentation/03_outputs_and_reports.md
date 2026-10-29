# 3. Outputs and Reports

Every run writes a self-contained, timestamped directory. Nothing is overwritten, and the
exact configuration used is saved alongside the results.

---

## 3.1 Run directory

```
outputs/runs/<YYYY-MM-DD_HHMMSS>_cutie_composite/
├── config_used.yaml              # exact parameters for this run
├── overlays/
│   ├── cutie_composite_<date>.avi   # overlay: masks, keypoints, ids
│   └── cutie_unitary_ratN_<date>.avi # one isolated video per animal
├── logs/
│   └── run.log
└── contacts/                     # only when contacts.enabled=true
    ├── results.xlsx              # the reported tables, one workbook  <-- the file to keep
    ├── diagnostics.xlsx          # tracker diagnostics: not validated, not for reporting
    ├── report.pdf                # 8-page PDF (the tracker's own diagnostic output)
    ├── event_log.txt             # human-readable event list
    └── reports/
        ├── timeline_comparison.png
        ├── duration_by_type.png
        ├── events_by_type.png
        └── event_duration_distribution.png
```

The per-run tables used to be six separate CSV and two JSON files. They are now
consolidated into `results.xlsx` (the reported tables) and `diagnostics.xlsx` (the
tracker diagnostics) at the end of the run, and the source files are removed. Pass
`--no_consolidate` to
[scripts/postprocess_contacts_simple.py](../scripts/postprocess_contacts_simple.py) to
keep them instead.

`cutie_composite` has no chunk mode, so there is no batch/merge layout for it. The
parallel runner and chunk merger belong to the older `centroid` pipeline.

### The videos

Two kinds, both for visual validation:

- **`cutie_composite_<date>.avi`** — the original frame with each animal's mask
  tinted in its own colour, its keypoints, and an `R1`…`RN` label at its centroid.
- **`cutie_unitary_ratN_<date>.avi`** — one video per animal showing only that
  animal against the background plate. This is the direct way to check the
  composite step: if a second animal is visible in one of these, the erase step
  failed and that animal's keypoints are suspect.

---

## 3.2 The data tables

The tables are split over two workbooks by one rule: `results.xlsx` holds only tables whose
numbers the clip-review validation (`scripts/validate_contacts.py`) can check, and
`diagnostics.xlsx` holds the raw tracker tables that it does not. Both are written by the same
call (`build_excel_report`).

### `results.xlsx` (reported, every table is covered by the validation)

One table per sheet, in this order:

| Sheet | Rows | What it holds |
|-------|------|---------------|
| `Summary` | 7 | Headline numbers per contact type, a `TOTAL` row and an `ANY` row. For the type rows and `TOTAL`, `total_sec` is summed over pairs (pair-seconds) and `pct_of_session` is a share of pair-time (session x number of pairs). `ANY` ("Any contact (union over pairs)") is the time with at least one pair in contact, FOL excluded, with `events` blank and `pct_of_session` a share of the session; it needs the per-frame table |
| `Events` | one per event | The cleaned behavioural events — **the reportable result** |
| `PerAnimal` | one per animal | Events and contact time per animal against any partner, plus the same per session minute (`session_min`, `events_per_min`, `contact_sec_per_min`), and the investigation bouts of the animal's pairs (`n_investigation_bouts`, `investigation_bouts_per_min`); `per_animal_summary.csv`, see 05 §9) |
| `PairMatrix` | one per pair of animals | Events by type, contact time (FOL excluded), `fol_sec`, `contact_pct_of_session` and rates per minute for every pair, including pairs with no events: the table behind the social-network figure (`pair_contact_matrix.csv`) |
| `Investigation` | one per investigation bout | N2N, N2AG and N2B merged into one bout per pair (gaps of up to `investigation.max_gap` frames bridged), with `n2n_sec` / `n2ag_sec` / `n2b_sec` / `other_sec`, `dominant_type`, `n_subtype_switches`, `investigator_slot` and `peak_score` (`investigation_bouts.csv`, see 05 §7). Additional to `Events`, which is unchanged |
| `PerFrame` | one per frame | Complete geometry with both raw and cleaned labels (raw per-frame data, not a claim) |
| `ByType` | 5 | Raw versus cleaned, per contact type (N2N, N2AG, N2B, FOL, SBS). No `NC` row: no-contact spans are not events and have no raw bouts to compare |
| `Global` | 13 | Raw versus cleaned, session totals. `raw_bouts_total` and `real_events_total` count contact types only, so `real_events_total` equals the `events` total of `Summary` and the contact rows of `Events`. `no_contact_sec` is the no-contact time, summed over pairs |
| `Parameters` | varies | Every parameter actually applied, plus video provenance |

### `diagnostics.xlsx` (tracker diagnostics, not validated, not for reporting)

Written next to `results.xlsx`, only when at least one of these tables exists. Its first
sheet, `README`, says so in one cell. The other sheets keep the content and columns they had
when they were still in `results.xlsx`:

| Sheet | Rows | What it holds |
|-------|------|---------------|
| `README` | 1 | "Diagnostic tables from the tracker (raw bouts, approach/avoid episodes, per-animal initiator statistics). Not validated by the clip review — not for reporting. The reported tables are in results.xlsx." |
| `Bouts` | one per bout | Bouts straight from `contacts_v2`, with the voted `initiator` |
| `Dynamics` | one per episode | Approach / avoid episodes **without** contact ("Hoja B"), with `mover` |
| `Individual` | varies | Per-animal initiator statistics flattened from `individual_summary.json`; built from the tracker bouts and their `initiator`, a different basis from the validated `PerAnimal` sheet |

Why they are apart: none of the three is a cleaned event table, so a clip review of `Events`
says nothing about them (`PerAnimal` and `Individual` can disagree for the same animal because
they count different things). Re-running the post-processing on a run that is already
consolidated keeps them: they are read back from `diagnostics.xlsx`, or from the `Bouts`,
`Dynamics` and `Individual` sheets of a `results.xlsx` written before the split.

### `PerFrame` — one row per frame per pair

The complete measurement record, grouped as `contacts_v2` writes it:

| Group | Columns |
|-------|---------|
| Identity | `frame_idx`, `time_str`, `time_sec`, `pair_key` |
| Classification | `family`, `contact_type`, `name_contact`, `secondary_type`, `secondary_name`, `secondary_score` |
| Dynamics | `dynamics`, `mover`, `dist_delta_bls`, `reciprocity` |
| Zone | `zone` |
| Geometry (body lengths) | `nose_nose_dist_bl`, `centroid_dist_bl`, `nose_tailbase_ij_bl`, `nose_tailbase_ji_bl`, `tail_tail_dist_bl`, `mask_iou`, `mask_contact_bl` |
| Kinematics | `velocity_i_bls`, `velocity_j_bls`, `velocity_alignment_cos`, `orientation_alignment_cos` |
| Body length | `body_length_i_px`, `body_length_j_px` |
| Roles | `investigator_role`, `initiator`, `bout_id` |
| Soft scores | one column per contact type, the continuous score before thresholding |
| Quality flags | `stale_keypoints`, `high_mask_overlap`, `missing_keypoints`, `single_detection`, `merged_state`, `fol_used_centroid`, `nose_off_mask_i/_j` and `nose_off_mask` (nose keypoint outside its own mask) |

This is the sheet for any custom analysis. Two columns need care: `mask_iou` is
identically 0 with CUTIE's disjoint masks (kept only for the `high_mask_overlap`
flag) — use `mask_contact_bl`, the shared mask border length in body lengths, which
drives SBS. `nose_tailbase_ij_bl` is meaningful only when `tail_start` was detected.
The soft score columns show how close a frame was to a different label.
Multi-rat tables hold one row per (frame, `pair_key`); count session time from
unique `frame_idx`, never from row count.

### `Events` — the behavioural event table

One row per cleaned event, covering the whole session including `NC` intervals (the validation
samples its no-contact windows from them; count events with `contact_type != "NC"`):

`event_id`, `start_time_sec`, `end_time_sec`, `duration_sec`, `start_time`, `end_time`,
`duration` (all three human-readable as `MM:SS.d`), `contact_type`, `contact_label`
(the full name, e.g. "Nose-to-anogenital"), `start_frame`, `end_frame`, `duration_frames`,
`investigator_slot`, and the mean nose–nose distance, centroid distance and mask IoU over
the event.

**This is the sheet to report and to use for statistics.**

### `event_log.txt`

The same events as plain text with timestamps — built for scrubbing to a time in the
overlay video and confirming the label by eye. The header gives the contact distance in BL
and cm, and an "Investigation bouts" block and a "Per-animal summary" block follow the summary
by type. The summary by type lists the no-contact intervals as `NC ... spans`, not as events,
and "Raw bouts -> Real contact events" counts contact events only. The log is built from the
cleaned events and carries none of the tracker diagnostics (approach/avoid episodes, initiator
statistics): those are in `diagnostics.xlsx`.

### `Parameters` and the session-level numbers

The `Global` and `ByType` sheets carry the session-level result, and `Parameters`
records what produced it:

`Parameters` is a flat `section / parameter / value` table with these sections:

| Section | What it records |
|---------|-----------------|
| `video` | Source file, FPS, frame count, duration, number of animals |
| `calibration` | Median body length in px, `cm_per_px` and where it came from, and the contact thresholds in cm |
| `detection_thresholds` | The contact zone and bout thresholds applied during the run |
| `quality_flags` | Frame counts per quality flag |
| `postprocessing` | The smoothing, gap-bridging and minimum-duration values **actually applied** |
| `postprocessing_metadata` | FPS and how it was determined, total frames and duration |
| `filtering_impact` | Frames and bouts removed, and the flicker rate |

`filtering_impact` is worth reporting: it quantifies how much of the raw signal was noise.
A high flicker rate signals unreliable keypoints rather than unusual behaviour.

The `video` section is the run's provenance, which is what makes a single `results.xlsx`
sufficient to reconstruct how a number was produced. It is preserved across repeated
post-processing passes even though its original JSON source no longer exists.

---

## 3.3 How the report is generated

Two report generators run, both with matplotlib. Neither requires a separate command —
both are invoked automatically at the end of a run.

### `report.pdf` — from `ContactTrackerV2.finalize()`

`report.pdf` is the tracker's own diagnostic output. It is drawn from the raw tracker bouts
and per-frame table, before the post-processing cleans them, so its counts are not the
reported numbers and the clip review does not validate it. Report from `results.xlsx`.

| Page | Figure | Reads |
|------|--------|-------|
| 1 | **Ethogram** — bouts as coloured bars on a time axis, one colour per contact type | bouts |
| 2 | **Bout duration histograms** — one panel per type, plus all types combined | bouts |
| 3 | **Pie chart** — share of total contact time by type | summary |
| 4 | **Contact rate over time** — stacked bars, contact seconds per one-minute bin | summary |
| 5 | **Per-pair summary table** — total contact seconds and per-type duration and bout count | summary |
| 6 | **Inter-rat distance over time** — nose–nose and centroid distance in body lengths, on a background shaded by proximity zone | per-frame table |
| 7 | **Investigator breakdown** — who initiated, stacked by rat, for N2AG, N2B and FOL | bouts |
| 8 | **Cumulative contact time** by type | per-frame table |

Pages 6–8 are produced only if pandas can read the per-frame table; the first five need only
the in-memory bouts and summary. Colours are fixed per contact type and shared across every
figure and both generators, so the type identity is consistent throughout.

### `contacts/reports/` — from the post-processing script

Four PNGs, all built on the **cleaned** events, and framed as a before/after comparison
against the raw labels:

- `timeline_comparison.png` — raw versus cleaned label timeline, showing what the temporal
  filtering removed
- `duration_by_type.png` — total time per contact type
- `events_by_type.png` — event counts per contact type
- `event_duration_distribution.png` — distribution of event durations

The same comparison as numbers lives in the `ByType` and `Global` sheets of
`results.xlsx`, ready to drop into a table.

For a merged parallel run, `merge_chunks.py` regenerates `report.pdf` from the concatenated
CSVs and re-runs the post-processing over the merged per-frame table, so the batch root
carries a complete session-level report.

---

## 3.4 Running the system

### Local — a short clip

```bash
export HF_TOKEN="hf_..."          # SAM3 is fetched from Hugging Face

python -m src.pipelines.cutie_composite.run     --config configs/local_cutie_composite.yaml
```

Contacts are enabled in the config. Any key can be overridden inline:

```bash
python -m src.pipelines.cutie_composite.run     --config configs/local_cutie_composite.yaml     video_path=data/raw/multirat.avi detection.max_animals=6 scan.max_frames=900
```

### HPC (UQ Bunya)

```bash
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=24:00:00
srun --pty bash

python -m src.pipelines.cutie_composite.run     --config configs/hpc_cutie_composite.yaml video_path=data/raw/<video>
```

One GPU, one process — this pipeline has no chunk mode. Only the SAM3 phase (the
first 125 frames) is held in memory and CUTIE streams the rest of the video, so
memory does not grow with the session length and `--mem=32G` is enough; a longer
session only needs a longer `--time`. Watch the `Phase 1 complete. Handoff: n/N rats`
line in the log: if `n < N`, the missing animals were never handed to CUTIE and will
be absent for the whole run.

### Re-running the post-processing on its own

Useful for exploring how sensitive the results are to the temporal thresholds, without
re-running inference:

```bash
python scripts/postprocess_contacts_simple.py outputs/runs/<run>/contacts/ \
    --config configs/contacts_postprocess_simple.yaml --make_reports

# or override directly
python scripts/postprocess_contacts_simple.py outputs/runs/<run>/contacts/ \
    smoothing.window=7 min_bout.duration_sec=0.5
```

FPS is taken from `--fps` if given, otherwise from `session_summary.json`, otherwise from
the `Parameters` sheet of an existing `results.xlsx`, otherwise from the video, otherwise
from the configured fallback of 30 — and which source was used is recorded in the
`postprocessing_metadata` section.

Re-running works whether or not the run has already been consolidated: the per-frame table
is read from `contacts_per_frame.csv` when it is there, and from the `PerFrame` sheet of
`results.xlsx` when it is not. The diagnostic tables of a consolidated run survive the re-run
(see §3.2).

---

## 3.5 The HTML viewer

[scripts/report_viewer.html](../scripts/report_viewer.html) reads a `results.xlsx` and
renders the session as a page. Open it, drop the workbook in, and it shows:

- **Stat tiles** — session duration, contact time as pair-seconds summed over pairs with its
  share of pair-time (the `TOTAL` row of `Summary`), time with any contact (the `ANY` row: frames
  in which at least one pair is in contact, as seconds and as a share of the session), event
  count, longest event, and the percentage of raw bouts removed by filtering
- **Ethogram** — every event on a time axis, with the inter-rat distance trace underneath
  so it is visible *why* an event fired. Hovering gives the type and timecode; clicking
  jumps to that row in the event table
- **Time by type** and **filtering impact** — the `ByType` and `Global` sheets as charts
- **Event table** — searchable and filterable by type, with timecodes for scrubbing
- **Parameters** — the full provenance, collapsed by default

It reads `results.xlsx` only and needs just the `Events` sheet; a missing sheet only drops
the part that needs it, it does not fail. It does not open `diagnostics.xlsx`, so approach/avoid
episodes are not shown (there is no dynamics panel).

The file is parsed in the browser and never uploaded anywhere. Contact-type colours match
`contacts.py` and the PDF exactly, and every colour is paired with its label, so identity
never depends on colour alone.

A published copy is at
<https://claude.ai/code/artifact/de07d163-e63b-46fb-9906-263ece405b52> — same page, no
local file needed.

---

## 3.6 The demo session

[scripts/make_demo_session.py](../scripts/make_demo_session.py) synthesises a
choreographed 120 s encounter — keypoints and masks, no YOLO, no SAM3, no CUTIE —
and drives it through the **real** `ContactTrackerV2`, the real post-processing
and the real Excel consolidation. It is both a sample file for the viewer and an
end-to-end test of the contact chain, with a known ground truth: the six
behaviours are scripted at known times.

Regenerate the demo with the fixed pipeline (masks disjoint, as in production;
the full 7-keypoint set is used):

```bash
python scripts/make_demo_session.py --out outputs/runs/demo_fixed --taxonomy v2
```

`--shipped-keypoints` and `--keep-mask-overlap` remain only to reproduce the old
defective (pre-2026-09-30) behaviour for comparison. Old `demo_as_shipped` /
`demo_fixed` workbooks predate the fix and are obsolete.

`--taxonomy v1` drives the older `contacts.py` instead, which is what runs on
`main`.

---

## 3.7 Validation checklist before reporting a session

1. **Watch the overlay video.** Each animal keeps its colour and its `R1`…`RN`
   label for the whole session. Any swap invalidates every per-animal measure
   downstream.
2. **Check the carry-over rate** in `run.log`. A high percentage means YOLO was frequently
   missing a rat and much of the geometry is interpolated.
3. **Check the filtering impact** in the `Parameters` sheet (`filtering_impact` rows) or
   the viewer's "Raw bouts removed" tile. A high value points to unreliable keypoints,
   not to unusual behaviour.
4. **Check the quality-flag totals** in the `Parameters` sheet — particularly
   `high_mask_overlap_frames`, where the two masks may have merged.
5. **Spot-check three or four events** by scrubbing to their timestamps from
   `event_log.txt` and confirming the label by eye.
6. **Confirm which post-processing parameters were applied** in the `Parameters` sheet;
   they must match `configs/contacts_postprocess_simple.yaml`, which the pipeline reads.
7. **Check the handoff** in `run.log`: `Phase 1 complete. Handoff: n/N rats`. If
   `n < N`, re-run — the missing animals cannot be recovered mid-session.
8. **Watch a unitary video.** If a second animal appears in
   `cutie_unitary_ratN_*.avi`, the composite failed and that animal's keypoints
   are unreliable.
9. **Confirm the run post-dates the 2026-09-30 fixes.** Anything produced earlier
   (e.g. `outputs/2026-07-28_*`) used wrong keypoint names, could not fire SBS, and
   mixed pairs in post-processing — re-run it
   ([02 §2.6](02_contact_detection.md#26-defects-found-and-fixed-2026-09-30)).

---

## 3.8 Sensitivity analysis

How much do the reported counts depend on the thresholds? `scripts/sensitivity_analysis.py`
answers it offline, without a GPU, from the per-frame table of a finished run
(`contacts_per_frame.csv`, or the `PerFrame` sheet of `results.xlsx`). The table holds the five
scores per frame and pair, and everything after the scores is deterministic, so the script rebuilds
the raw label (`ScoreMap.argmax_type`), the cleaned label (`compute_labels`) and the events
(`extract_events`) for each value of one parameter at a time, the others at their baseline.

```bash
python scripts/sensitivity_analysis.py outputs/runs/<run>                     # grid in configs/sensitivity.yaml
python scripts/sensitivity_analysis.py outputs/runs/<run> --out some/dir      # other output folder
python scripts/sensitivity_analysis.py outputs/runs/<run> min_bout_sec=[0.3,0.5] smoothing_window=7
```

| Parameter | Default grid | Baseline |
|-----------|--------------|----------|
| `activation_threshold` (N2N, N2AG, SBS, N2B) | 0.40, 0.45, 0.50, 0.55, 0.60, 0.70 | 0.50 |
| `activation_threshold_rare` (FOL) | 0.25, 0.35, 0.45 | 0.35 |
| `smoothing_window` (frames) | 3, 5, 7 | `configs/contacts_postprocess_simple.yaml` |
| `gap_bridging_max_gap` (frames) | 1, 3, 5 | same file |
| `min_bout_sec` (global minimum duration) | 0.2, 0.3, 0.4, 0.5, 1.0 | same file; `min_bout.by_type` stays in force |

Output in `<run>/sensitivity/` (or `--out`):

- `sensitivity_long.csv`: `parameter, value, contact_type, n_events, total_sec, mean_sec`, plus
  `n_pairs` (pairs with an event of that type), `n_pair_frames` (pair-frames carrying the type after
  cleaning) and `is_baseline`. NC is not counted.
- `sensitivity_summary.md`: one table per parameter (columns = types, cells = `n_events (total_sec)`,
  baseline row in bold), the robustness paragraph, and the list of what cannot be varied offline.
- `sensitivity_<parameter>.png`: events against value, one line per type, the baseline as a dashed line.

The script also prints a one-paragraph robustness statement: for each type, the minimum and maximum
event count over the activation-threshold grid and over the minimum-duration grid (FOL does not move
with the first, because it uses the rare threshold). It checks the rebuilt raw labels against
`ScoreMap` on a row sample and reports how often they equal the stored `contact_type` at the baseline
thresholds; anything below 100 % means the stored label was not the plain argmax of the stored scores.

**Not covered.** The Schmitt enter/exit distances (`contact_zone_bl_enter/_exit`,
`sbs_contact_bl_enter/_exit`, `follow_radius_bl*`) and the per-type score internals are fixed at
tracking time and already folded into the stored scores; varying them needs a new tracker run.
