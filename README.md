# Mouse Social-Contact Tracking Pipeline

A markerless computer-vision pipeline that tracks six group-housed, visually identical mice from a
single overhead camera and scores their social contacts — nose-to-nose, nose-to-anogenital,
nose-to-body, side-by-side and following — without implants, markers or a labelled training set of
behaviour. Identity is carried by video object segmentation (SAM3 + CUTIE), pose by YOLO26, and
every behaviour by interpretable geometric rules that are validated against blinded human review.

Runs locally on short clips and on UQ's Bunya HPC on full sessions. The production pipeline is
[`cutie_composite`](src/pipelines/cutie_composite/); the methods reference for the paper is
[`final_documentation/`](final_documentation/README.md).

## What it does

1. **Detect** the mice with SAM3 (text prompt "mouse") in the first seconds.
2. **Track** each one as a labelled mask with CUTIE, so identity comes from mask memory rather than
   per-frame re-detection; an **area guard** calls SAM3 to re-separate masks when the tracker merges
   two touching animals.
3. **Pose** — seven keypoints per mouse with YOLO26, estimated on a per-animal composite that erases
   the others so keypoints are never confused.
4. **Classify** five contact behaviours per pair and frame from touch and orientation, in
   body-length units (`ContactTrackerV2`).
5. **Clean and count** — smoothing, events, groups and bouts; per-mouse and per-pair summaries.
6. **Validate** — sample events into blinded clips and score precision, recall, the investigator,
   the event boundaries and identity persistence.

## Pipelines evaluated (the research record)

Several tracking strategies were built and compared on this data before `cutie_composite` was chosen
as the production pipeline. The alternatives are kept in the repository on purpose — the comparison is
part of the research. Why each was or was not selected:
[`final_documentation/04_model_comparison.md`](final_documentation/04_model_comparison.md); the
chronological test log is [`docs/changes/observation_log.md`](docs/changes/observation_log.md).

| Pipeline | Approach | Status |
|----------|----------|--------|
| [`cutie_composite`](src/pipelines/cutie_composite/) | SAM3 bootstrap → CUTIE mask propagation (identity by construction) → YOLO26 pose on per-animal composites → `ContactTrackerV2` (5 contact types + recovery) | **Production — the paper pipeline.** One GPU, one process |
| [`centroid`](src/pipelines/centroid/) | SAM2 prompted with each animal's previous centroid, YOLO pose on the full frame, v1 contacts | Earlier production pipeline; the only one with chunked multi-GPU runs |
| `sam3_composite`, `isolated_composite` | SAM3 / SAM2-video with per-animal composites | Building blocks imported by `cutie_composite` |
| `sam3_reset`, `samurai`, `samurai_sleap` | SAM3 with reset strategies; SAMURAI (and +SLEAP pose) | Exploratory; not selected (see the comparison) |

Earlier approaches, debug scripts and deprecated configs are preserved under
[`archive/`](archive/README.md). `main` is behind this branch and does not contain `cutie_composite`;
active work is on `develop`.

## Quick start (local)

```bash
git clone https://github.com/CateHey/computer-vision-tracking.git && cd computer-vision-tracking
python -m venv .venv && source .venv/bin/activate          # PowerShell: .venv/Scripts/Activate.ps1
pip install torch torchvision torchaudio                     # CUDA build from pytorch.org if you have a GPU
git clone https://github.com/facebookresearch/sam2.git models/sam2/segment-anything-2
git clone https://github.com/hkchengrex/Cutie.git models/cutie && pip install -e models/cutie
pip install -r requirements.txt                              # SAM3 comes through transformers

export HF_TOKEN="hf_..."                                     # SAM3 is fetched from Hugging Face
python -m src.pipelines.cutie_composite.run --config configs/local_cutie_composite.yaml
python -m src.pipelines.cutie_composite.run --config configs/local_cutie_composite.yaml \
    video_path=data/raw/session.avi detection.max_animals=6 scan.max_frames=900
```

Full setup: [docs/setup/local.md](docs/setup/local.md). Configs must list the 7 keypoint names in
model index order (`tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear`); see
[CLAUDE.md](CLAUDE.md).

## Quick start (Bunya HPC)

```bash
ssh uqcvaras@bunya.rcc.uq.edu.au                             # UQ password + Okta push
cd ~/computer-vision-tracking && git pull
sbatch slurm/run_cutie.sbatch data/raw/session.avi 6         # paper pipeline, 1 GPU; args: video, number of mice
sbatch slurm/run_cutie.sbatch data/raw/session.avi 6 120     # optional 3rd arg: only the first N seconds
```

First-time setup from zero, one verified command at a time:
[docs/setup/bunya_step_by_step.md](docs/setup/bunya_step_by_step.md).

## What a run produces

Each run writes `outputs/runs/<timestamp>_cutie_composite/`:

- **`README_WHAT_TO_REVIEW.md`** — what to open and in what order, with the red flags; start here.
- **`contacts/results.xlsx`** — the **reported** tables (events, per-mouse, per-pair, summary); every
  table is covered by the clip review.
- **`contacts/diagnostics.xlsx`** — the tracker's raw diagnostic tables; **not** validated, not for
  reporting.
- **`overlays/`** — the full session video with all masks, plus one video per mouse.
- **`validation/`** — blinded evidence clips, `review.xlsx`, `index.html`, and `identity/` (one
  labelled frame per minute to confirm identities never swapped).
- **`tracks/`** — per-frame mask history (`tracks.jsonl.gz`) and `area_per_mouse.png` (recovery
  episodes).
- **`provenance.json`, `config_used.yaml`, `logs/`** — the exact configuration, commit, versions and
  weights checksum for reproducibility.

## Validation workflow

```bash
python scripts/postprocess_contacts_simple.py outputs/runs/<run>/contacts/ \
    --config configs/contacts_postprocess_simple.yaml --make_reports
python scripts/validate_contacts.py sample outputs/runs/<run> --per-type 15 --nc-per-run 20
# a reviewer fills verdict / correct_type / investigator_ok / boundaries_ok / subtype_ok in review.xlsx
python scripts/validate_contacts.py score  outputs/runs/<run>/validation/review.xlsx
python scripts/validate_contacts.py identity outputs/runs/<run> --every 60   # identity checkpoints
python scripts/sensitivity_analysis.py outputs/runs/<run>                     # threshold sensitivity
pytest tests/
```

Open [`scripts/report_viewer.html`](scripts/report_viewer.html) in a browser and drop in
`results.xlsx` for an interactive report. **Before reporting contact counts, read
[OPEN_ISSUES.md](OPEN_ISSUES.md)** for the decisions still pending.

## Repository map

```
computer-vision-tracking/
  src/
    common/                 # config, video I/O, geometry, metrics, contacts v1/v2, Excel report, run README
    pipelines/              # cutie_composite (production) + sam3_composite / isolated_composite (building blocks)
                            #   + centroid (earlier) + sam3_reset / samurai* (exploratory)
  configs/                  # YAML configs: local_* / hpc_* per pipeline, extraction, post-processing, validation
  scripts/                  # post-processing, validation, sensitivity, report viewer, demo session
  slurm/                    # sbatch jobs (run_cutie.sbatch = paper pipeline, 1 GPU)
  tests/                    # pytest for the post-processing and validation tooling
  final_documentation/      # methods reference for the paper (pipeline, contacts, outputs, model comparison)
  docs/                     # design history and guides (index: docs/README.md)
  archive/                  # earlier approaches, debug scripts and deprecated configs, kept for the record
  roboflow/                 # frame extraction for labelling + YOLO pose training
  data/, models/, outputs/  # large content (gitignored; .gitkeep preserves the folders)
```

## Configuration

Every parameter lives in YAML; nothing is hardcoded. Override any key on the command line as
`key.subkey=value` (e.g. `detection.max_animals=6`).

| Pipeline / tool | Local | HPC |
|-----------------|-------|-----|
| cutie_composite (production) | `configs/local_cutie_composite.yaml` | `configs/hpc_cutie_composite.yaml` |
| centroid | `configs/local_centroid.yaml` | `configs/hpc_centroid.yaml` |
| sam3_composite / sam3_reset | `configs/local_sam3_*.yaml` | `configs/hpc_sam3_*.yaml` |
| isolated_composite | `configs/local_isolated_composite.yaml` | `configs/hpc_isolated_composite.yaml` |
| samurai / samurai_sleap | `configs/local_samurai_sleap.yaml` | `configs/hpc_samurai*.yaml` |
| contact post-processing | `configs/contacts_postprocess_simple.yaml` | same |
| validation | `configs/validation.yaml` | same |

## Documentation

- [`final_documentation/`](final_documentation/README.md) — the paper's methods reference (pipeline,
  contact detection, outputs, model comparison, the November 2026 contact update).
- [`docs/`](docs/README.md) — design history and guides:
  [HPC setup](docs/setup/bunya_step_by_step.md), [local setup](docs/setup/local.md),
  [parameter tuning](docs/guides/parameter_tuning.md),
  [observation log](docs/changes/observation_log.md),
  [risks & limitations](docs/architecture/risks.md).
- [`archive/`](archive/README.md) — exploratory scripts and earlier approaches, kept for the record.

## Status and citation

Research software under active development on `develop`. The contact rules and the mask-recovery
module are validated against a hand-annotated clip; a full-length native run and a two-rater
validation are the work before the counts are reported (see [OPEN_ISSUES.md](OPEN_ISSUES.md)).

**[Citation / DOI to be added on publication. Code, configuration and the annotated clip will be
released at the repository URL.]**
