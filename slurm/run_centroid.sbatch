#!/bin/bash --login
#SBATCH --job-name=centroid-infer
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/centroid_%j.out
#SBATCH --error=outputs/slurm/centroid_%j.err

# ============================================================================
# Centroid Pipeline on Bunya HPC (single GPU)
# ============================================================================
#
# IMPORTANT: You must pass your account with --account or -A:
#   sbatch --account=a_YOUR_GROUP slurm/run_centroid.sbatch
#
# With overrides:
#   OVERRIDES="video_path=data/raw/original_120s.avi contacts.enabled=true" \
#     sbatch --account=a_YOUR_GROUP slurm/run_centroid.sbatch
#
# For parallel (3 GPUs), use run_chunks.sbatch instead:
#   sbatch --array=0-2 slurm/run_chunks.sbatch
#
# ============================================================================

set -euo pipefail

# Default config (can be overridden via --export=CONFIG=...)
CONFIG="${CONFIG:-configs/hpc_centroid.yaml}"

# Optional config overrides (space-separated key=value pairs)
OVERRIDES="${OVERRIDES:-}"

echo "=== Centroid Pipeline ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Config:    $CONFIG"
echo "Overrides: $OVERRIDES"
echo "=========================="

# Navigate to project root
cd ~/Balbi/yolo-sam2-lab-tracking

# Load modules and activate virtual environment
module load python/3.10.4-gcccore-11.3.0 2>/dev/null || true
source .venv/bin/activate

# Create slurm output directory
mkdir -p outputs/slurm

# Run pipeline
if [ -n "$OVERRIDES" ]; then
    python -m src.pipelines.centroid.run --config "$CONFIG" $OVERRIDES
else
    python -m src.pipelines.centroid.run --config "$CONFIG"
fi

echo "=== Job complete ==="
