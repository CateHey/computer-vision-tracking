#!/bin/bash --login
#SBATCH --job-name=centroid-infer
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/centroid_%j.out
#SBATCH --error=outputs/slurm/centroid_%j.err

# ============================================================================
# Centroid Pipeline on Bunya HPC (single GPU, one process)
# ============================================================================
#
# SBATCH_ACCOUNT, so --account is not needed on the command line):
#
#   cd ~/computer-vision-tracking
#   sbatch slurm/run_centroid.sbatch
#
# With overrides (any key=value the pipeline accepts):
#   OVERRIDES="video_path=data/raw/original_120s.avi contacts.enabled=true" \
#     sbatch slurm/run_centroid.sbatch
#
# Different config:
#   CONFIG=configs/hpc_centroid.yaml sbatch slurm/run_centroid.sbatch
#
#
# For a full video use the multi-GPU runner instead:
#   sbatch scripts/run_pipeline.sbatch data/raw/original.mp4
# ============================================================================

set -euo pipefail

CONFIG="${CONFIG:-configs/hpc_centroid.yaml}"
OVERRIDES="${OVERRIDES:-}"

cd "$HOME/computer-vision-tracking"
module load miniforge/26.3.2-1
eval "$(conda shell.bash hook)"
conda activate rattrack

echo "=== Centroid Pipeline ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Account:   ${SLURM_JOB_ACCOUNT:-?}"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "Config:    $CONFIG"
echo "Overrides: ${OVERRIDES:-<none>}"
echo "Python:    $(python --version 2>&1)"
echo "CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================="

# shellcheck disable=SC2086
python -m src.pipelines.centroid.run --config "$CONFIG" $OVERRIDES

echo "=== Job complete ==="
