#!/bin/bash
#SBATCH --job-name=sam2-video-infer
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/sam2video_%j.out
#SBATCH --error=outputs/slurm/sam2video_%j.err

# ============================================================================
# SAM2 Video Pipeline on Bunya HPC
# ============================================================================
#
# Uses SAM2VideoPredictor with temporal memory for robust tracking.
# Slower than sam2_yolo but much better tracking through occlusions.
#
# Usage:
#   sbatch slurm/run_sam2video.sbatch
#
#   # Override config and parameters:
#   sbatch --export=ALL,CONFIG=configs/hpc_sam2video.yaml,OVERRIDES="video_path=data/raw/other.mp4 contacts.enabled=true" slurm/run_sam2video.sbatch
#
# ============================================================================

set -euo pipefail

# Default config (can be overridden via --export=CONFIG=...)
CONFIG="${CONFIG:-configs/hpc_sam2video.yaml}"

# Optional config overrides (space-separated key=value pairs)
OVERRIDES="${OVERRIDES:-}"

echo "=== SAM2 Video Pipeline ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Config:    $CONFIG"
echo "Overrides: $OVERRIDES"
echo "==========================="

# Navigate to project root
cd ~/Balbi/yolo-sam2-lab-tracking

# Load modules and activate virtual environment
module load python/3.10.4-gcccore-11.3.0 2>/dev/null || true
source .venv/bin/activate

# Create slurm output directory
mkdir -p outputs/slurm

# Use node-local SSD for temp JPEG frames (much faster I/O)
export TMPDIR="${TMPDIR:-/tmp}"
echo "TMPDIR:    $TMPDIR"

# Run pipeline
if [ -n "$OVERRIDES" ]; then
    python -m src.pipelines.sam2_video.run --config "$CONFIG" $OVERRIDES
else
    python -m src.pipelines.sam2_video.run --config "$CONFIG"
fi

echo "=== Job complete ==="
