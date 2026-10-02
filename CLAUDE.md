# Rat Tracking Vision Pipeline

## Project

SAM3 + CUTIE + YOLO pose pipeline for laboratory rat tracking (6 animals) and social
contact classification. Runs locally (short clips) and on UQ Bunya HPC (full videos, 1 GPU).

## Production Pipeline

**`cutie_composite`** (`src/pipelines/cutie_composite/`) — production default.
1. SAM3 text-prompt bootstrap on the first 125 frames (needs `HF_TOKEN`)
2. CUTIE propagates one labelled index mask (identity comes from CUTIE)
3. Per-animal composite: other animals are erased using a background plate
4. YOLO pose (`models/yolo/yolo26_v11.pt`) on each composite
5. `ContactTrackerV2` (`src/common/contacts_v2.py`): 5 types N2N / N2AG / N2B / SBS / FOL + dynamics
6. `scripts/postprocess_contacts_simple.py`: per-pair temporal cleaning
7. `results.xlsx` (`src/common/excel_report.py`) + `scripts/report_viewer.html`

No chunk mode: `scripts/run_parallel.sh` and `scripts/merge_chunks.py` belong to the old centroid pipeline.

### Keypoints (CRITICAL)

7 keypoints, model index order: `tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear`
(verified 2026-09-30 by running the model). Configs must list names in this order.
The 5-name lists used before were wrong and **invalidated all contact outputs produced before 2026-09-30**.
Rear/anogenital point = `tail_start` (not `tail_base`).

### Contacts

- SBS uses `mask_contact_bl` (shared-border length in body lengths): CUTIE masks are disjoint, so mask IoU is always 0.
- Contact types: N2N nose-to-nose, N2AG nose-to-anogenital, N2B nose-to-body, SBS side-by-side, FOL following; NC = none.

## Other Pipelines

`centroid` (older SAM2 centroid pipeline), `sam3_composite`, `sam3_reset`, `isolated_composite`, `samurai*`
in `src/pipelines/`. Older ones (reference, sam3, sam2_yolo, sam2_video) are in `src/pipelines/deprecated/`.

**Never use YOLO box prompts for SAM2** (arbitrary detection order causes identity swaps; centroid-only prompting
was swap-free). See `docs/centroid_pipeline.md` "Lessons Learned".

## Key Paths

| What | Path |
|------|------|
| Production pipeline | `src/pipelines/cutie_composite/` |
| Contact tracker (current) | `src/common/contacts_v2.py` |
| Post-processing | `scripts/postprocess_contacts_simple.py` |
| Excel report / viewer | `src/common/excel_report.py`, `scripts/report_viewer.html` |
| Validation | `scripts/validate_contacts.py`, `configs/validation.yaml` |
| YOLO inference / weights | `src/common/yolo_inference.py`, `models/yolo/yolo26_v11.pt` |
| Configs | `configs/local_cutie_composite.yaml`, `configs/hpc_cutie_composite.yaml` |
| Old centroid pipeline / parallel runner | `src/pipelines/centroid/`, `scripts/run_parallel.sh`, `scripts/merge_chunks.py` |
| Deprecated pipelines / configs | `src/pipelines/deprecated/`, `configs/deprecated/` |

## Running

```bash
export HF_TOKEN="hf_..."   # SAM3 download

# Local
python -m src.pipelines.cutie_composite.run --config configs/local_cutie_composite.yaml
python -m src.pipelines.cutie_composite.run --config configs/local_cutie_composite.yaml \
    video_path=data/raw/multirat.avi detection.max_animals=6 scan.max_frames=900

# HPC (Bunya, one GPU, one process; see final_documentation/03 section 3.4)
python -m src.pipelines.cutie_composite.run --config configs/hpc_cutie_composite.yaml video_path=data/raw/<video>

# Re-run post-processing only
python scripts/postprocess_contacts_simple.py outputs/runs/<run>/contacts/ --make_reports
```

## Validation workflow

```bash
python scripts/validate_contacts.py sample outputs/runs/<run> --per-type 15   # clips + review.xlsx + index.html
# reviewer fills verdict / correct_type in <run>/validation/review.xlsx
python scripts/validate_contacts.py score outputs/runs/<run>/validation/review.xlsx
```

Verdicts: real, flicker, wrong_type, tracking_error, unsure. `score` writes precision with Wilson CIs per type,
flicker rate by duration, and min-duration / min-score threshold tables. Tests: `pytest tests/`.

## Documentation

`final_documentation/` is the current reference (01 pipeline, 02 contact detection, 03 outputs and reports,
04 model comparison). `docs/` is older design history (see `docs/README.md`).

## Conventions

- Branches: work on `develop`
- All configs in `configs/` — no hardcoded thresholds in code
- CLI overrides: `key.subkey=value` (e.g., `detection.confidence=0.4`)
- Output to `outputs/runs/<timestamp>_<tag>/`
- Logs go to `outputs/runs/<run>/logs/run.log`
