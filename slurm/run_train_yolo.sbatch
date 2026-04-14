#!/bin/bash
#SBATCH --job-name=yolo-train
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/train_%j.out
#SBATCH --error=outputs/slurm/train_%j.err

# ============================================================================
# YOLO Training on Bunya HPC
# ============================================================================
#
# Usage:
#   sbatch slurm/run_train_yolo.sbatch
#
# Prerequisites:
#   - Roboflow dataset exported to data/roboflow_export/
#   - data.yaml path configured in configs/yolo_train.yaml
#
# ============================================================================

set -euo pipefail

echo "=== YOLO Training ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "=========================="

cd ~/yolo-sam2-lab-tracking
source .venv/bin/activate
module load cuda 2>/dev/null || true

mkdir -p outputs/slurm

# Example training command (adjust paths as needed):
# yolo detect train \
#   data=data/roboflow_export/data.yaml \
#   model=yolov8l.pt \
#   epochs=100 \
#   imgsz=640 \
#   batch=16 \
#   project=outputs/runs \
#   name=yolo_train

echo "=== Training job template ==="
echo "Edit this script to configure your training run."
echo "See: https://docs.ultralytics.com/modes/train/"

echo "=== Job complete ==="
