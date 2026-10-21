# Rat Tracking Vision Pipeline

Segmentation + pose pipeline for laboratory rat tracking and social-contact classification
from overhead video. Runs locally on short clips and on UQ's Bunya HPC on full sessions.

## Pipelines on this branch (`develop`)

| Pipeline | What it does | Status |
|----------|--------------|--------|
| `cutie_composite` | SAM3 text-prompt bootstrap on the first 125 frames, then CUTIE propagates one labelled mask (identity by construction, N animals). YOLO pose runs on per-animal composites so it never sees two animals. `ContactTrackerV2`: 5 contact types + dynamics. Outputs `results.xlsx`, a PDF report and an HTML viewer | **Production: the paper pipeline.** One process, one GPU |
| `centroid` | SAM2 prompted with each animal's previous centroid, YOLO pose on the full frame, 2 rats, v1 contacts | Older production pipeline; the only one with chunked multi-GPU runs |
| `sam3_composite`, `isolated_composite` | SAM3 / SAM2-video with per-animal composites | Building blocks imported by `cutie_composite` |
| `sam3_reset`, `samurai`, `samurai_sleap` | SAM3 with reset strategies, SAMURAI (+SLEAP) | Exploratory; why they were not selected: [`final_documentation/04_model_comparison.md`](final_documentation/04_model_comparison.md) |

`main` is behind this branch and does not contain `cutie_composite`. The methods reference for the
paper is [`final_documentation/`](final_documentation/README.md).

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
    video_path=data/raw/multirat.avi detection.max_animals=6 scan.max_frames=900

# older centroid pipeline on the committed 10 s clip
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml contacts.enabled=true
```

Details: [docs/setup/local.md](docs/setup/local.md). Configs must list the 7 keypoint names in model
index order (`tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear`); see `CLAUDE.md`.

## Quick start (Bunya HPC)

```bash
ssh uqcvaras@bunya.rcc.uq.edu.au                                        # UQ password + Okta push
cd ~/computer-vision-tracking && source scripts/bunya_env.sh            # account string + ~/.hf_token
sbatch slurm/run_cutie.sbatch data/raw/session.avi 2                    # paper pipeline, 1 GPU; args: video, number of animals
sbatch scripts/run_pipeline.sbatch data/raw/session.avi                 # centroid pipeline: 3 GPUs, chunked
```

Account string, first-time setup (`scripts/bunya_setup.sh`), memory sizing and downloads:
[docs/setup/hpc.md](docs/setup/hpc.md).

## Project structure

```
computer-vision-tracking/
  configs/                  # YAML configs: local_* / hpc_* per pipeline, extraction, post-processing, validation
  docs/                     # Older design documentation (index: docs/README.md)
  final_documentation/      # Methods reference for the paper (pipeline, contacts v2, outputs, model comparison)
  data/clips/               # 10 s test clip (committed); data/raw/ holds full videos (gitignored)
  models/yolo/              # YOLO pose weights: yolo26_v11.pt (paper), best.pt (centroid); SAM2/CUTIE are cloned
  roboflow/                 # Frame extraction for labelling + YOLO pose training (local and Bunya)
  scripts/                  # post-processing, validation, report viewer, Bunya helpers, centroid parallel runner
  slurm/                    # sbatch files: run_cutie.sbatch (paper pipeline, 1 GPU), centroid single run, centroid array
  src/common/               # config loading, video I/O, geometry, metrics, contacts v1/v2, Excel report
  src/pipelines/            # cutie_composite, sam3_composite, sam3_reset, isolated_composite, centroid, samurai*
  tests/                    # pytest for post-processing and validation tooling
  outputs/runs/             # run outputs: <timestamp>_<tag>/ (gitignored)
```

## Configuration

Every parameter lives in YAML; nothing is hardcoded. Override any key on the command line as
`key.subkey=value`.

| Pipeline / tool | Local | HPC |
|-----------------|-------|-----|
| cutie_composite | `configs/local_cutie_composite.yaml` | `configs/hpc_cutie_composite.yaml` |
| centroid | `configs/local_centroid.yaml` | `configs/hpc_centroid.yaml` |
| sam3_composite / sam3_reset | `configs/local_sam3_*.yaml` | `configs/hpc_sam3_*.yaml` |
| isolated_composite | `configs/local_isolated_composite.yaml` | `configs/hpc_isolated_composite.yaml` |
| samurai / samurai_sleap | `configs/local_samurai_sleap.yaml` | `configs/hpc_samurai.yaml`, `configs/hpc_samurai_sleap.yaml` |
| frame extraction (labelling) | `configs/local_extract.yaml` | `configs/hpc_extract.yaml` |
| contact post-processing | `configs/contacts_postprocess_simple.yaml` | same |
| validation | `configs/validation.yaml` | same |

## Post-processing, validation, reports

```bash
python scripts/postprocess_contacts_simple.py outputs/runs/<run>/contacts/ \
    --config configs/contacts_postprocess_simple.yaml --make_reports
python scripts/validate_contacts.py sample outputs/runs/<run> --per-type 15
python scripts/validate_contacts.py score outputs/runs/<run>/validation/review.xlsx
pytest tests/
```

Open `scripts/report_viewer.html` in a browser and drop in `results.xlsx`.

## Documentation

- [final_documentation/README.md](final_documentation/README.md): the paper's methods reference
- [docs/README.md](docs/README.md): index of the older design documentation. Key pages:
  [HPC Guide](docs/setup/hpc.md),
  [Local Setup](docs/setup/local.md),
  [Parameter Tuning](docs/guides/parameter_tuning.md),
  [Centroid Pipeline](docs/centroid_pipeline.md),
  [Risks & Limitations](docs/architecture/risks.md)
