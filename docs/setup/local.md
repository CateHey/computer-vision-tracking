# Running Locally

Local runs are for short clips and config checks. Full sessions run on Bunya ([bunya_step_by_step.md](bunya_step_by_step.md)).

## Prerequisites

- Python 3.10+
- ffmpeg on PATH (needed by `scripts/merge_chunks.py`; optional for single runs)
- A CUDA GPU is strongly recommended for `cutie_composite`; the centroid pipeline runs the 10 s clip on CPU with the tiny SAM2 model
- A Hugging Face token (`HF_TOKEN`) for the SAM3 download used by `cutie_composite`

## Setup

```bash
git clone https://github.com/CateHey/computer-vision-tracking.git
cd computer-vision-tracking

python -m venv .venv
source .venv/bin/activate              # PowerShell: .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip

# PyTorch: pick the build for your CUDA version at https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# SAM2 (requirements.txt installs it as an editable package from this clone)
git clone https://github.com/facebookresearch/sam2.git models/sam2/segment-anything-2

# CUTIE (video object segmentation; weights are downloaded into ./weights/ on first use)
git clone https://github.com/hkchengrex/Cutie.git models/cutie
pip install -e models/cutie

# everything else, including transformers for SAM3
pip install -r requirements.txt

# SAM2.1 checkpoints: local configs use hiera_tiny, HPC configs use hiera_large
(cd models/sam2/segment-anything-2/checkpoints && bash download_ckpts.sh)

python -c "import torch, sam2, cutie, transformers, ultralytics; print('cuda:', torch.cuda.is_available())"
```

The YOLO pose weights (`models/yolo/yolo26_v11.pt` for the paper pipeline, `models/yolo/best.pt`
for the centroid pipeline) come with the repository.

## Quick test: paper pipeline (`cutie_composite`)

```bash
export HF_TOKEN="hf_..."               # SAM3 is fetched from Hugging Face on first run

python -m src.pipelines.cutie_composite.run --config configs/local_cutie_composite.yaml
python -m src.pipelines.cutie_composite.run --config configs/local_cutie_composite.yaml \
    video_path=data/raw/multirat.avi detection.max_animals=6 scan.max_frames=900
```

Contacts are enabled in the config. Watch for `Phase 1 complete. Handoff: n/N rats` in the log:
n must equal N, otherwise the missing animals are absent for the whole run.

## Quick test: centroid pipeline

`configs/local_centroid.yaml` points at the committed 10 s clip `data/clips/output-10s.mp4`
and the tiny SAM2 model.

```bash
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml contacts.enabled=true
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml models.device=cuda scan.max_frames=100
```

Any YAML key can be overridden as `key.subkey=value`; see [parameter_tuning.md](../guides/parameter_tuning.md).

## Cut a clip from a raw video

```bash
python scripts/trim_video.py data/raw/original.mp4 --duration 120                                   # -> data/raw/original_120s.avi
python scripts/trim_video.py data/raw/original.mp4 --start 60 --duration 30 --output data/clips/minute2.mp4
```

## Post-processing, validation and reports

```bash
# temporal cleaning of the per-frame contacts into events, results.xlsx and the PDF report
python scripts/postprocess_contacts_simple.py outputs/runs/<run>/contacts/ \
    --config configs/contacts_postprocess_simple.yaml --make_reports

# validation: stratified sample of events for review, then scoring of the filled review.xlsx
python scripts/validate_contacts.py sample outputs/runs/<run> --per-type 15
python scripts/validate_contacts.py score outputs/runs/<run>/validation/review.xlsx

pytest tests/
```

Open `scripts/report_viewer.html` in a browser and drop in `results.xlsx`. Details:
`final_documentation/03_outputs_and_reports.md` and [contacts_postprocess_simple.md](../contacts_postprocess_simple.md).

## Labelling and retraining YOLO

Frame extraction, Roboflow upload and training: [labeling.md](../guides/labeling.md) and
`roboflow/TRAINING_GUIDE.md`.

## Output structure

```
outputs/runs/<timestamp>_<tag>/
  config_used.yaml                  # resolved config, for reproducibility
  logs/run.log
  overlays/<tag>_<model>_<date>.avi
  contacts/                         # per-frame CSV, events, session summary
  results.xlsx, report.pdf          # after post-processing (cutie_composite)
```

Chunked centroid runs on Bunya add a `chunks/` level and a merged batch directory; see
[output_structure.md](../data/output_structure.md).
