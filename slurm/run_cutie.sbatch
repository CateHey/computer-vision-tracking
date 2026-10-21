#!/bin/bash --login
#SBATCH --job-name=cutie-clip
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/cutie_%j.out
#SBATCH --error=outputs/slurm/cutie_%j.err

# ============================================================================
# Production pipeline on Bunya, end to end, in one job:
#   0. first run only: installs the environment (scripts/bunya_setup.sh, ~20-30 min)
#   1. cutie_composite (SAM3 -> CUTIE -> YOLO pose -> contacts_v2 -> results.xlsx + PDF)
#   2. validation sample (scripts/validate_contacts.py sample: clips, thumbs, review.xlsx)
#   3. zips the run directory and prints the scp line to download it
# One GPU, one process: the pipeline holds every frame in memory and has no chunk mode.
#
# Submit from the project dir on a login node after sourcing scripts/bunya_env.sh
# (it exports the Slurm account string and reads ~/.hf_token):
#
#   cd ~/computer-vision-tracking && source scripts/bunya_env.sh
#   sbatch slurm/run_cutie.sbatch data/raw/original.mp4 2 10     # video, animals, first N seconds
#
# Arguments
#   $1  video path                                 (default: data/raw/original.mp4)
#   $2  number of animals -> detection.max_animals (default: the config value, 6)
#   $3  seconds to process from the start -> scan.max_frames (default: the whole video)
#
# Env knobs (prefix the sbatch command)
#   CONFIG=configs/hpc_cutie_composite.yaml
#   OVERRIDES="sam3.sam3_phase_frames=50 cutie.max_internal_size=480"
#   VALIDATE=0          skip the validation sample (default 1)
#   PER_TYPE=15         events per contact type in the sample
#   ZIP=0               do not zip the run directory (default 1)
#   SETUP=1             re-run scripts/bunya_setup.sh even if the env exists
#
# Long sessions: every frame is held in memory, so raise the resources:
#   sbatch --mem=128G --time=24:00:00 slurm/run_cutie.sbatch data/raw/original.mp4 2
#
# Monitor:  squeue -u $USER      tail -f outputs/slurm/cutie_<JOBID>.out
# Results:  outputs/runs/<timestamp>_cutie_composite/  and  outputs/runs/<timestamp>_cutie_composite.zip
# ============================================================================

set -euo pipefail

VIDEO="${1:-data/raw/original.mp4}"
N_ANIMALS="${2:-}"
RUN_SECONDS="${3:-}"
CONFIG="${CONFIG:-configs/hpc_cutie_composite.yaml}"
OVERRIDES="${OVERRIDES:-}"
VALIDATE="${VALIDATE:-1}"
PER_TYPE="${PER_TYPE:-15}"
ZIP="${ZIP:-1}"
SETUP="${SETUP:-0}"

cd "${BUNYA_PROJECT_DIR:-$HOME/computer-vision-tracking}"
# shellcheck disable=SC1091
source scripts/bunya_env.sh

echo "============================================"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  Account:   ${SLURM_JOB_ACCOUNT:-?}"
echo "  GPU:       ${CUDA_VISIBLE_DEVICES:-none}  ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))"
echo "  Video:     $VIDEO  (animals: ${N_ANIMALS:-config}, seconds: ${RUN_SECONDS:-all})"
echo "  Config:    $CONFIG"
echo "  Overrides: ${OVERRIDES:-<none>}"
echo "  Validate:  $VALIDATE (per_type=$PER_TYPE)   Zip: $ZIP"
echo "  Date:      $(date)"
echo "============================================"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is empty, so SAM3 (facebook/sam3) cannot be loaded." >&2
    echo "       echo hf_... > ~/.hf_token && chmod 600 ~/.hf_token   (see scripts/bunya_env.sh)" >&2
    exit 1
fi

# ---- 0. Environment (the first job installs it) --------------------------------
if [ "$SETUP" = "1" ] || ! bunya_activate 2>/dev/null; then
    echo ""
    echo "=== Environment '$BUNYA_CONDA_ENV' missing or SETUP=1: running scripts/bunya_setup.sh"
    bash scripts/bunya_setup.sh
    bunya_activate
fi
echo "  Python:    $(python --version 2>&1)"
echo "  PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())')"

# ---- 1. Inputs -------------------------------------------------------------------
# (after setup, because setup moves data/raw to scratch)
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: video not found: $VIDEO  (upload it to ~/computer-vision-tracking/data/raw/)" >&2
    exit 1
fi
if [ -n "$N_ANIMALS" ]; then
    OVERRIDES="detection.max_animals=$N_ANIMALS $OVERRIDES"
fi
if [ -n "$RUN_SECONDS" ]; then
    MAX_FRAMES="$(python - "$VIDEO" "$RUN_SECONDS" <<'PY'
import sys, cv2
cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n = min(int(round(fps * float(sys.argv[2]))), total) if total > 0 else int(round(fps * float(sys.argv[2])))
print(n)
print(f"  video: {fps:.2f} fps, {total} frames -> processing the first {n} frames", file=sys.stderr)
PY
)"
    OVERRIDES="scan.max_frames=$MAX_FRAMES $OVERRIDES"
fi
echo "  Overrides: $OVERRIDES"

# ---- 2. Pipeline -----------------------------------------------------------------
echo ""
echo "=== Pipeline: cutie_composite"
# shellcheck disable=SC2086
python -m src.pipelines.cutie_composite.run --config "$CONFIG" video_path="$VIDEO" $OVERRIDES

# Run dirs are named <YYYY-MM-DD_HHMMSS>_cutie_composite, so the last one by name is this job's.
RUN_DIR="$(ls -d outputs/runs/*_cutie_composite 2>/dev/null | sort | tail -1)"
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: no *_cutie_composite run directory under outputs/runs/" >&2
    exit 1
fi
echo ""
echo "=== Run directory: $RUN_DIR"
grep -h -E "Handoff|carry-over|Total bouts|pose detected" "$RUN_DIR"/logs/*.log 2>/dev/null || true

# ---- 3. Validation sample (evidence clips + review workbook) ---------------------
if [ "$VALIDATE" = "1" ]; then
    echo ""
    echo "=== Validation sample (per_type=$PER_TYPE)"
    if python scripts/validate_contacts.py sample "$RUN_DIR" --per-type "$PER_TYPE"; then
        echo "  review workbook: $RUN_DIR/validation/review.xlsx"
    else
        echo "  WARNING: validation sample failed; usually the clip produced no contact events." >&2
        echo "           Check 'Total bouts' above and the contacts/ folder of the run." >&2
    fi
fi

# ---- 4. Package + download commands ----------------------------------------------
ABS_RUN="$(readlink -f "$RUN_DIR")"
RUN_NAME="$(basename "$RUN_DIR")"
ZIP_PATH=""
if [ "$ZIP" = "1" ]; then
    echo ""
    echo "=== Zipping $RUN_NAME"
    ZIP_PATH="$(python - "$ABS_RUN" <<'PY'
import pathlib, shutil, sys
run = pathlib.Path(sys.argv[1])
print(shutil.make_archive(str(run), "zip", root_dir=run.parent, base_dir=run.name))
PY
)"
    echo "  $ZIP_PATH  ($(du -h "$ZIP_PATH" | cut -f1))"
fi

cat <<EOF

============================================================
  DONE: $RUN_DIR
  overlays/     cutie_composite_<date>.avi + one unitary video per animal
  contacts/     results.xlsx, CSV tables, event_log.txt, PDF report
  validation/   review.xlsx, index.html, clips/, thumbs/   <- the evidence; fill review.xlsx
  logs/         run.log
============================================================

Download to the laptop (PowerShell; one password + Okta push):
  scp ${BUNYA_USER}@${BUNYA_HOST}:${ZIP_PATH:-$ABS_RUN} "${BUNYA_LOCAL_DIR}\outputs\\"
  Expand-Archive -Force "${BUNYA_LOCAL_DIR}\outputs\\${RUN_NAME}.zip" "${BUNYA_LOCAL_DIR}\outputs\\"
  -> ${BUNYA_LOCAL_DIR}\outputs\\${RUN_NAME}\validation\index.html

Score the filled review (upload it back, then on a login node):
  scp "${BUNYA_LOCAL_DIR}\outputs\\${RUN_NAME}\validation\review.xlsx" ${BUNYA_USER}@${BUNYA_HOST}:${ABS_RUN}/validation/
  cd ${BUNYA_PROJECT_DIR} && source scripts/bunya_env.sh && bunya_activate
  python scripts/validate_contacts.py score ${RUN_DIR}/validation/review.xlsx
EOF

echo "Job finished at: $(date)"
