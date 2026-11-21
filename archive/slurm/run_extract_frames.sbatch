#!/bin/bash
#SBATCH --job-name=extract-frames
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/extract_%j.out
#SBATCH --error=outputs/slurm/extract_%j.err

# ============================================================================
# Close-Contact Frame Extraction on Bunya HPC
# ============================================================================
#
# Usage:
#   sbatch slurm/run_extract_frames.sbatch
#
# ============================================================================

set -euo pipefail

CONFIG="${CONFIG:-configs/hpc_full.yaml}"

echo "=== Frame Extraction ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Config:    $CONFIG"
echo "=========================="

cd ~/yolo-sam2-lab-tracking
source .venv/bin/activate
module load cuda 2>/dev/null || true

mkdir -p outputs/slurm

python scripts/extract_frames.py all --config "$CONFIG"

echo "=== Job complete ==="
