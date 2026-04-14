#!/bin/bash
#SBATCH --job-name=sam2-yolo-infer
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/infer_%j.out
#SBATCH --error=outputs/slurm/infer_%j.err

# ============================================================================
# SAM2+YOLO Inference on Bunya HPC
# ============================================================================
#
# Usage:
#   sbatch slurm/run_infer.sbatch
#
#   # Override config:
#   sbatch slurm/run_infer.sbatch --export=CONFIG=configs/local_quick.yaml
#
# ============================================================================

set -euo pipefail

# Default config (can be overridden via --export=CONFIG=...)
CONFIG="${CONFIG:-configs/hpc_full.yaml}"

echo "=== SAM2+YOLO Inference ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Config:    $CONFIG"
echo "=========================="

# Navigate to project root
cd ~/Balbi/yolo-sam2-lab-tracking

# Load modules and activate virtual environment
module load python/3.10.4-gcccore-11.3.0 2>/dev/null || true
source .venv/bin/activate

# Create slurm output directory
mkdir -p outputs/slurm

# Run pipeline
python -m src.pipelines.sam2_yolo.run --config "$CONFIG"

echo "=== Job complete ==="
