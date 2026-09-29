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
    ├── results.xlsx              # every table, one workbook  <-- the file to keep
    ├── report.pdf                # 8-page PDF
    ├── event_log.txt             # human-readable event list
    └── reports/
        ├── timeline_comparison.png
        ├── duration_by_type.png
        ├── events_by_type.png
        └── event_duration_distribution.png
```

The per-run tables used to be six separate CSV and two JSON files. They are now
consolidated into a single `results.xlsx` at the end of the run, and the source files
are removed. Pass `--no_consolidate` to
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

Everything lives in `results.xlsx`, one table per sheet.

| Sheet | Rows | What it holds |
|-------|------|---------------|
| `Summary` | 6 | Headline numbers per contact type, plus a total row |
| `Events` | one per event | The cleaned behavioural events — **the reportable result** |
| `PerFrame` | one per frame | Complete geometry with both raw and cleaned labels |
| `Bouts` | one per bout | Bouts from `contacts_v2` (diagnostic) |
| `Dynamics` | one per episode | Approach / avoid episodes **without** contact ("Hoja B") |
| `Individual` | one per animal | Per-animal metrics — **currently never written, see 02 §2.6 C** |
| `ByType` | 7 | Raw versus cleaned, per contact type |
| `Global` | 11 | Raw versus cleaned, session totals |
| `Parameters` | varies | Every parameter actually applied, plus video provenance |

### `PerFrame` — one row per frame per pair

The complete measurement record, grouped as `contacts_v2` writes it:

| Group | Columns |
|-------|---------|
| Identity | `frame_idx`, `time_str`, `time_sec`, `pair_key` |
| Classification | `family`, `contact_type`, `name_contact`, `secondary_type`, `secondary_name`, `secondary_score` |
| Dynamics | `dynamics`, `mover`, `dist_delta_bls`, `reciprocity` |
| Zone | `zone` |
| Geometry (body lengths) | `nose_nose_dist_bl`, `centroid_dist_bl`, `nose_tailbase_ij_bl`, `nose_tailbase_ji_bl`, `tail_tail_dist_bl`, `mask_iou` |
| Kinematics | `velocity_i_bls`, `velocity_j_bls`, `velocity_alignment_cos`, `orientation_alignment_cos` |
| Body length | `body_length_i_px`, `body_length_j_px` |
| Roles | `investigator_role`, `initiator`, `bout_id` |
| Soft scores | one column per contact type, the continuous score before thresholding |
| Quality flags | `stale_keypoints`, `high_mask_overlap`, `missing_keypoints`, `single_detection`, `merged_state`, `fol_used_centroid` |

This is the sheet for any custom analysis. Two columns are worth checking first:
`mask_iou` (identically 0 — see [02 §2.6 B](02_contact_detection.md)) and
`nose_tailbase_ij_bl` (its missing sentinel throughout — see 02 §2.6 A). The soft
score columns show how close a frame was to a different label.

### `Events` — the behavioural event table

One row per cleaned event, covering the whole session including `NC` intervals:

`event_id`, `start_time_sec`, `end_time_sec`, `duration_sec`, `start_time`, `end_time`,
`duration` (all three human-readable as `MM:SS.d`), `contact_type`, `contact_label`
(the full name, e.g. "Nose-to-anogenital"), `start_frame`, `end_frame`, `duration_frames`,
`investigator_slot`, and the mean nose–nose distance, centroid distance and mask IoU over
the event.

**This is the sheet to report and to use for statistics.**

### `event_log.txt`

The same events as plain text with timestamps — built for scrubbing to a time in the
overlay video and confirming the label by eye.

### `Parameters` and the session-level numbers

The `Global` and `ByType` sheets carry the session-level result, and `Parameters`
records what produced it:

`Parameters` is a flat `section / parameter / value` table with these sections:

| Section | What it records |
|---------|-----------------|
| `video` | Source file, FPS, frame count, duration, number of animals |
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
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00
srun --pty bash

python -m src.pipelines.cutie_composite.run     --config configs/hpc_cutie_composite.yaml video_path=data/raw/<video>
```

One GPU, one process — this pipeline has no chunk mode. Because every frame is
held in memory, request memory in proportion to the session length and
resolution, and watch the `Phase 1 complete. Handoff: n/N rats` line in the log:
if `n < N`, the missing animals were never handed to CUTIE and will be absent for
the whole run.

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
`results.xlsx` when it is not.

---

## 3.5 The HTML viewer

[scripts/report_viewer.html](../scripts/report_viewer.html) reads a `results.xlsx` and
renders the session as a page. Open it, drop the workbook in, and it shows:

- **Stat tiles** — session duration, total contact time and its share of the session,
  event count, longest event, and the percentage of raw bouts removed by filtering
- **Ethogram** — every event on a time axis, with the inter-rat distance trace underneath
  so it is visible *why* an event fired. Hovering gives the type and timecode; clicking
  jumps to that row in the event table
- **Time by type** and **filtering impact** — the `ByType` and `Global` sheets as charts
- **Event table** — searchable and filterable by type, with timecodes for scrubbing
- **Parameters** — the full provenance, collapsed by default

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

Two workbooks ship with this documentation, under `outputs/runs/`:

| File | What it is |
|------|-----------|
| `demo_as_shipped/contacts/results.xlsx` | 5 keypoints and disjoint masks, exactly as the pipeline runs today. N2AG and SBS are absent. |
| `demo_fixed/contacts/results.xlsx` | 7 keypoints including `tail_start`, masks allowed to overlap. All types recoverable. |

Opening both in the viewer is the quickest way to see the defects in §2.6: the
same choreography, the same code, and two of the five behaviours appear only in
the second file.

```bash
# regenerate (needs origin/develop for contacts_v2)
python scripts/make_demo_session.py --out outputs/runs/demo_as_shipped     --taxonomy v2 --shipped-keypoints
python scripts/make_demo_session.py --out outputs/runs/demo_fixed     --taxonomy v2 --keep-mask-overlap
```

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
6. **Confirm which post-processing parameters were applied** — read the `Parameters`
   sheet rather than assuming the YAML values (see §2.3).
7. **Check the handoff** in `run.log`: `Phase 1 complete. Handoff: n/N rats`. If
   `n < N`, re-run — the missing animals cannot be recovered mid-session.
8. **Watch a unitary video.** If a second animal appears in
   `cutie_unitary_ratN_*.avi`, the composite failed and that animal's keypoints
   are unreliable.
9. **Do not report N2AG or SBS counts** from the current pipeline — see
   [02 §2.6](02_contact_detection.md#26-three-defects-that-suppress-contact-types).
