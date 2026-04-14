#!/bin/bash
###############################################################################
# SLURM BATCH SCRIPT — Centroid pipeline (3 GPU parallel)
###############################################################################
#
# Submit:    sbatch scripts/run_pipeline.sbatch data/raw/original_120s.avi
# Monitor:   squeue -u $USER
# Cancel:    scancel <JOB_ID>
# Logs:      tail -f outputs/slurm/pipeline_<JOBID>.out
#
###############################################################################

# ---- Slurm resource requests ----
#SBATCH --job-name=centroid-pipeline
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3                   # 3 GPUs for parallel chunks
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/pipeline_%j.out
#SBATCH --error=outputs/slurm/pipeline_%j.err

###############################################################################
# 1. ENVIRONMENT
###############################################################################

mkdir -p outputs/slurm

cd ~/Balbi/yolo-sam2-lab-tracking

module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

VIDEO="${1:-data/raw/original_120s.avi}"

echo "============================================"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  GPUs:      $CUDA_VISIBLE_DEVICES"
echo "  Video:     $VIDEO"
echo "  Python:    $(python --version)"
echo "  PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Date:      $(date)"
echo "============================================"

###############################################################################
# 2. RUN PIPELINE (3 GPUs, 3 chunks)
###############################################################################

bash scripts/run_parallel.sh "$VIDEO" 3

echo ""
echo "Job finished at: $(date)"
