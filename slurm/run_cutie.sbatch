#!/bin/bash --login
#SBATCH --job-name=cutie
#SBATCH --account=a_qbi
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/cutie_%j.out
#SBATCH --error=outputs/slurm/cutie_%j.err

# cutie_composite + validation sample + zip, on one GPU.
# Requires the conda env "rattrack" (docs/setup/bunya_step_by_step.md) and ~/.hf_token.
#
# Submit from ~/computer-vision-tracking:
#   mkdir -p outputs/slurm
#   sbatch slurm/run_cutie.sbatch data/raw/original.mp4 2      # video, number of animals
#   sbatch slurm/run_cutie.sbatch data/raw/original.mp4 2 5    # only the first 5 seconds
#
# Watch:  squeue -u $USER      tail -f outputs/slurm/cutie_<JOBID>.out

set -euo pipefail

VIDEO="$1"        # e.g. data/raw/original.mp4
ANIMALS="$2"      # e.g. 2
SECONDS_TO_RUN="${3:-}"   # optional: only the first N seconds

cd "$HOME/computer-vision-tracking"
module load miniforge/26.3.2-1
eval "$(conda shell.bash hook)"
conda activate rattrack
export HF_TOKEN="$(cat "$HOME/.hf_token")"

nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 1. Pipeline -> outputs/runs/<timestamp>_cutie_composite/
# Short test: limit the frames and give SAM3 half of them so CUTIE also runs
# (SAM3 normally takes the first 125 frames).
LIMIT=""
if [ -n "$SECONDS_TO_RUN" ]; then
    FRAMES="$(python -c "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); print(int(round((c.get(cv2.CAP_PROP_FPS) or 25)*float(sys.argv[2]))))" "$VIDEO" "$SECONDS_TO_RUN")"
    SAM3_FRAMES=$(( FRAMES / 2 < 125 ? FRAMES / 2 : 125 ))
    LIMIT="scan.max_frames=$FRAMES sam3.sam3_phase_frames=$SAM3_FRAMES"
    echo "Short run: $SECONDS_TO_RUN s = $FRAMES frames (SAM3 $SAM3_FRAMES, CUTIE the rest)"
fi

# shellcheck disable=SC2086
python -m src.pipelines.cutie_composite.run --config configs/hpc_cutie_composite.yaml \
    video_path="$VIDEO" detection.max_animals="$ANIMALS" $LIMIT

RUN="$(ls -d outputs/runs/*_cutie_composite | sort | tail -1)"
echo "Run: $RUN"
grep -h -E "Handoff|carry-over|Total bouts" "$RUN"/logs/*.log || true

# 2. Validation sample -> $RUN/validation/ (review.xlsx, index.html, clips/)
python scripts/validate_contacts.py sample "$RUN" --per-type 15 || echo "No events to sample."

# 3. Zip the whole run
cd outputs/runs
python -c "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', '.', sys.argv[1])" "$(basename "$RUN")"
ls -lh "$(basename "$RUN").zip"

echo ""
echo "Download (PowerShell on the laptop):"
echo "  scp uqcvaras@bunya.rcc.uq.edu.au:computer-vision-tracking/outputs/runs/$(basename "$RUN").zip C:\\Users\\uqcvaras\\Downloads\\computer-vision-tracking\\outputs\\"
echo "  Expand-Archive C:\\Users\\uqcvaras\\Downloads\\computer-vision-tracking\\outputs\\$(basename "$RUN").zip C:\\Users\\uqcvaras\\Downloads\\computer-vision-tracking\\outputs\\"
