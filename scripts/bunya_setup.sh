#!/bin/bash
# ============================================================================
# First-time setup of this project on Bunya. Idempotent: safe to re-run.
# ============================================================================
#
# Run it INSIDE an interactive job (RCC kills builds on the login nodes):
#
#   ssh uqcvaras@bunya.rcc.uq.edu.au
#   git clone -b develop https://github.com/CateHey/computer-vision-tracking.git ~/computer-vision-tracking
#   cd ~/computer-vision-tracking && source scripts/bunya_env.sh
#   bunya_gpu_shell 1 2                      # 1 GPU for 2 h: enough to build + verify
#   cd ~/computer-vision-tracking && bash scripts/bunya_setup.sh
#
# SAM3 (facebook/sam3) is a gated Hugging Face model: put a read token whose
# account accepted the terms in ~/.hf_token (or export HF_TOKEN) before running,
# otherwise step 4 skips the SAM3 download and the first job does it instead.
#
# What it does:
#   1. Creates the scratch dirs and symlinks data/raw and outputs/runs into
#      /scratch/user/<you>/computer-vision-tracking (300 GB, purged @ 90 days).
#      outputs/slurm (job logs) stays in home: slurm/run_cutie.sbatch runs this
#      script on the first job, and moving its open log file would orphan it.
#   2. Creates the conda env (python 3.11 + ffmpeg) under ~/.conda via miniforge
#   3. Installs PyTorch (CUDA 12.6 wheels), SAM2 (required by requirements.txt),
#      requirements.txt, transformers>=5 (SAM3) and CUTIE
#   4. Downloads the weights: SAM2.1 checkpoints, CUTIE, SAM3 (if a token exists)
#   5. Verifies CUDA and every import the cutie_composite pipeline, the report
#      and the validation script need
#
# Roughly 15-30 minutes, dominated by the downloads.
# ============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_DIR="$(pwd)"
# shellcheck disable=SC1091
source scripts/bunya_env.sh

# --- 0. Guards --------------------------------------------------------------
case "$(hostname -s)" in
    bunya[0-9]*)
        echo "ERROR: this is a login node ($(hostname -s)); RCC kills builds here." >&2
        echo "       Get a compute node first:  bunya_gpu_shell 1 2   then re-run this script." >&2
        exit 1
        ;;
esac
if [ "$PROJECT_DIR" != "$BUNYA_PROJECT_DIR" ]; then
    echo "NOTE: project is at $PROJECT_DIR but BUNYA_PROJECT_DIR=$BUNYA_PROJECT_DIR."
    echo "      Export BUNYA_PROJECT_DIR=$PROJECT_DIR before sourcing bunya_env.sh, or move the clone."
fi

# --- 1. Scratch layout + symlinks ---------------------------------------------
echo "=== [1/5] Scratch layout: $BUNYA_SCRATCH_DIR"
mkdir -p "$BUNYA_SCRATCH_DIR/data/raw" "$BUNYA_SCRATCH_DIR/outputs/runs" outputs/slurm

# link_to_scratch <relative dir>: replace a repo dir with a symlink into scratch,
# moving any existing contents across first.
link_to_scratch() {
    local rel="$1" target="$BUNYA_SCRATCH_DIR/$1"
    mkdir -p "$target" "$(dirname "$rel")"
    if [ -L "$rel" ]; then
        echo "  $rel -> $(readlink "$rel")  (already linked)"
        return
    fi
    if [ -d "$rel" ]; then
        find "$rel" -mindepth 1 -maxdepth 1 -exec mv -n -t "$target" {} +
        rmdir "$rel"
    fi
    ln -s "$target" "$rel"
    echo "  $rel -> $target"
}
link_to_scratch data/raw
link_to_scratch outputs/runs

# data/raw/.gitkeep is tracked by git, so keep git quiet about the symlink.
touch "$BUNYA_SCRATCH_DIR/data/raw/.gitkeep"
git update-index --skip-worktree data/raw/.gitkeep 2>/dev/null || true
if ! grep -qx '/data/raw' .git/info/exclude 2>/dev/null; then
    echo '/data/raw' >> .git/info/exclude
fi

# --- 2. Conda env -------------------------------------------------------------
echo "=== [2/5] Conda env '$BUNYA_CONDA_ENV' (module $BUNYA_CONDA_MODULE)"
bunya_load_conda
if conda env list | awk '{print $1}' | grep -qx "$BUNYA_CONDA_ENV"; then
    echo "  env exists, reusing it"
else
    # 3.11: Python 3.10 reached end of life in October 2026 and new numpy/opencv wheels no longer target it.
    conda create -y -n "$BUNYA_CONDA_ENV" -c conda-forge python=3.11 ffmpeg
fi
set +u; conda activate "$BUNYA_CONDA_ENV"; set -u
echo "  python: $(command -v python)"

# --- 3. Python packages -------------------------------------------------------
echo "=== [3/5] PyTorch, SAM2, requirements.txt, transformers (SAM3), CUTIE"
python -m pip install --upgrade --no-cache-dir pip
# CUDA 12.6 wheels: they run on any driver >= 525, which is what an HPC cluster with
# A100 / H100 / L40S nodes can be expected to have. The default PyPI wheels of recent
# torch releases bundle CUDA 13 and need a newer driver. If the verify step still
# reports cuda=False, compare `nvidia-smi` (driver) with the index and pick another:
#   https://download.pytorch.org/whl/cu130   (driver >= 580)
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
python -m pip install --no-cache-dir torch torchvision torchaudio --index-url "$TORCH_INDEX"
python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("  ERROR: this torch build cannot see the GPU. Check `nvidia-smi` and TORCH_INDEX (see comment above).")
PY

# SAM2: requirements.txt installs it from this checkout (used by the centroid pipeline).
SAM2_DIR="models/sam2/segment-anything-2"
if [ ! -d "$SAM2_DIR/.git" ]; then
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
fi
# Skip SAM2's optional CUDA extension (needs nvcc, only used for mask post-processing).
export SAM2_BUILD_CUDA=0
python -m pip install --no-cache-dir -r requirements.txt

# SAM3 ships in transformers 5.x; 4.x fails with KeyError: 'sam3_video'.
python -m pip install --no-cache-dir "transformers>=5.0" accelerate

# CUTIE (hkchengrex/Cutie): the continuous tracker of the cutie_composite pipeline.
CUTIE_DIR="models/cutie"
if [ ! -d "$CUTIE_DIR/.git" ]; then
    git clone https://github.com/hkchengrex/Cutie.git "$CUTIE_DIR"
fi
# Cutie's full dependency list pulls in GUI/training packages (PySide6, gradio,
# cchardet, netifaces, ...) that may fail to build. Inference needs only a few,
# so fall back to --no-deps plus those when the full install fails.
if ! python -m pip install --no-cache-dir -e "$CUTIE_DIR"; then
    echo "  full CUTIE install failed; installing inference-only dependencies instead"
    python -m pip install --no-cache-dir --no-deps -e "$CUTIE_DIR"
    python -m pip install --no-cache-dir hydra-core omegaconf einops easydict gitpython requests tqdm gdown
fi

# --- 4. Model weights ---------------------------------------------------------
echo "=== [4/5] Model weights"
echo "  -- SAM2.1 checkpoints (centroid pipeline)"
CKPT_DIR="$SAM2_DIR/checkpoints"
mkdir -p "$CKPT_DIR"
for ck in sam2.1_hiera_large.pt sam2.1_hiera_tiny.pt; do
    if [ -s "$CKPT_DIR/$ck" ]; then
        echo "  $ck present"
    else
        wget -q --show-progress -O "$CKPT_DIR/$ck.part" \
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/$ck"
        mv "$CKPT_DIR/$ck.part" "$CKPT_DIR/$ck"
    fi
done

echo "  -- CUTIE weights -> $CUTIE_DIR/weights/ (where Cutie's loader looks)"
python - <<'PY'
from cutie.utils.download_models import download_models_if_needed
download_models_if_needed()
PY

echo "  -- SAM3 (facebook/sam3, gated) -> ~/.cache/huggingface"
if [ -n "${HF_TOKEN:-}" ]; then
    python - <<'PY' || echo "  WARNING: SAM3 download failed. Accept the terms at https://huggingface.co/facebook/sam3 with the account that owns the token, then re-run this script."
import os
from huggingface_hub import snapshot_download
path = snapshot_download("facebook/sam3", token=os.environ["HF_TOKEN"])
print("  SAM3 cached at", path)
PY
else
    echo "  WARNING: HF_TOKEN is empty; SAM3 was NOT downloaded."
    echo "           1. https://huggingface.co/facebook/sam3   -> accept the terms"
    echo "           2. https://huggingface.co/settings/tokens -> create a read token"
    echo "           3. echo hf_... > ~/.hf_token && chmod 600 ~/.hf_token"
    echo "           then re-run this script (or let the first job download it)."
fi

# --- 5. Verify ----------------------------------------------------------------
echo "=== [5/5] Verify"
python - <<'PY'
import torch, ultralytics, cv2, transformers
from sam2.build_sam import build_sam2                          # noqa: F401  centroid pipeline
from transformers import Sam3VideoModel, Sam3VideoProcessor    # noqa: F401  SAM3
from cutie.utils.get_default_model import get_default_model    # noqa: F401  CUTIE
from cutie.inference.inference_core import InferenceCore       # noqa: F401
import pandas, openpyxl, matplotlib, scipy, yaml, tqdm         # noqa: F401  contacts / report / validation
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f"  python {__import__('sys').version.split()[0]}  numpy {__import__('numpy').__version__}")
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpu={gpu}")
print(f"  transformers {transformers.__version__}  ultralytics {ultralytics.__version__}  opencv {cv2.__version__}")
print("  imports OK: sam2, Sam3VideoModel, cutie, pandas, openpyxl, matplotlib")
PY
ls -la models/yolo/yolo26_v11.pt "$CUTIE_DIR/weights/cutie-base-mega.pth"
echo "  ffmpeg: $(command -v ffmpeg)"
echo "  driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"

cat <<EOT

============================================================
  SETUP COMPLETE
  Code / env :  $PROJECT_DIR   (conda env: $BUNYA_CONDA_ENV)
  Data       :  $BUNYA_SCRATCH_DIR/data/raw      (= data/raw)
  Outputs    :  $BUNYA_SCRATCH_DIR/outputs/runs  (= outputs/runs)
============================================================

Next steps
  1. Upload a video from your laptop (PowerShell):
     scp "${BUNYA_LOCAL_DIR}/data/raw/original.mp4" ${BUNYA_USER}@${BUNYA_HOST}:computer-vision-tracking/data/raw/
  2. Submit the production pipeline + validation sample from a login node:
     cd ${BUNYA_PROJECT_DIR} && source scripts/bunya_env.sh
     sbatch slurm/run_cutie.sbatch data/raw/original.mp4 2 10     # video, number of animals, first N seconds
  3. Watch it:   squeue -u \$USER      tail -f outputs/slurm/cutie_<JOBID>.out
  4. Quotas:     rquota
EOT
