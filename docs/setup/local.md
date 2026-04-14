# Running Locally

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, CPU works for short clips)
- ffmpeg (optional, for clip extraction)

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd yolo-sam2-lab-tracking

# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install PyTorch (match your CUDA version)
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install SAM2 (required for reference, sam2_yolo, sam2_video)

```bash
# Clone SAM2 into the models directory
git clone https://github.com/facebookresearch/segment-anything-2.git models/sam2/segment-anything-2

# Install SAM2 as editable package
pip install -e ./models/sam2/segment-anything-2

# Install project dependencies (filter out SAM2 editable line to avoid conflicts)
grep -v "^-e ./models/sam2/segment-anything-2" requirements.txt > requirements.fixed.txt
pip install -r requirements.fixed.txt

# Download SAM2 checkpoints (tiny for local, large for HPC)
cd models/sam2/segment-anything-2/checkpoints && bash download_ckpts.sh
cd -
```

### Install SAM3 (optional — for sam3 pipeline)

SAM3 requires **Python 3.12** and **PyTorch 2.7+** (CUDA 12.6).
If your existing venv uses Python 3.10, create a separate venv.

```bash
# Option A: Same venv (if you already have Python 3.12 + PyTorch 2.7+)
git clone https://github.com/facebookresearch/sam3.git models/sam3/sam3
pip install -e ./models/sam3/sam3

# Option B: Separate venv (if your main venv is Python 3.10)
python3.12 -m venv .venv-sam3
source .venv-sam3/bin/activate  # Linux/Mac
# .venv-sam3\Scripts\activate   # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/facebookresearch/sam3.git models/sam3/sam3
pip install -e ./models/sam3/sam3
grep -v "^-e ./models/sam2/segment-anything-2" requirements.txt > requirements.fixed.txt
pip install -r requirements.fixed.txt

# Download SAM3 checkpoint (~3.2GB, requires HuggingFace authentication)
pip install huggingface_hub
huggingface-cli login   # paste your HF token when prompted
huggingface-cli download facebook/sam3 sam3.pt --local-dir models/sam3/
```

### Verify installation

```bash
# SAM2
python -c "from sam2.build_sam import build_sam2; print('SAM2 OK')"

# SAM3 (only if installed)
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"

# YOLO
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

## Quick Test: Reference Pipeline (recommended)

Process a 10-second clip with overlay output:

```bash
python -m src.pipelines.reference.run --config configs/local_reference.yaml
```

With contact classification:

```bash
python -m src.pipelines.reference.run --config configs/local_reference.yaml contacts.enabled=true
```

### Alternative: SAM3 Pipeline (if installed)

```bash
python -m src.pipelines.sam3.run --config configs/local_sam3.yaml
```

### Alternative: SAM2+YOLO Pipeline

```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml
```

Override parameters on the fly:

```bash
python -m src.pipelines.reference.run --config configs/local_reference.yaml \
    detection.confidence=0.3 scan.max_frames=50
```

## Quick Test: Frame Extraction

Find close-contact frames and export them:

```bash
# Run all steps (scan + encounters + plan + export)
python scripts/extract_frames.py all --config configs/local_quick.yaml

# Or run steps individually
python scripts/extract_frames.py scan --config configs/local_quick.yaml
python scripts/extract_frames.py encounters --config configs/local_quick.yaml --run-dir outputs/runs/<your-run-dir>
python scripts/extract_frames.py plan --config configs/local_quick.yaml --run-dir outputs/runs/<your-run-dir>
python scripts/extract_frames.py export --config configs/local_quick.yaml --run-dir outputs/runs/<your-run-dir>
```

## Extract a Clip from Raw Video

```bash
python scripts/extract_clip.py \
    --input data/raw/original.mp4 \
    --output data/clips/test-5s.mp4 \
    --start 0 --duration 5
```

## Upload Frames to Roboflow

```bash
export ROBOFLOW_API_KEY="your-key"

python scripts/upload_to_roboflow.py \
    --config configs/local_quick.yaml \
    --frames-root outputs/runs/<run-dir>/frames
```

## Output Structure

Each run creates a timestamped directory:

```
outputs/runs/<timestamp>_<tag>/
  config_used.yaml          # Config snapshot for reproducibility
  logs/
    run.log                 # Full execution log
  overlays/
    overlay.mp4             # Annotated video (pipeline runs)
  scan/
    close_frames.jsonl      # Per-frame analysis (extract_frames)
    scan_metadata.json
    encounters_summary.json
    sampling_plan.json
  frames/
    encounter_000/
      rat_close_e000_f000123.jpg
    encounter_001/
      ...
```
