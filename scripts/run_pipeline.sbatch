#!/bin/bash --login
###############################################################################
# SLURM BATCH SCRIPT — centroid pipeline, N GPUs in parallel on one node
###############################################################################
#
# Submit from the project dir on a login node (no salloc needed):
#   cd ~/computer-vision-tracking && source scripts/bunya_env.sh
#   sbatch scripts/run_pipeline.sbatch data/raw/original_120s.avi
#
# Arguments:
#   $1  video path                       (default: data/raw/original_120s.avi)
#   $2  number of chunks = GPUs to use   (default: every GPU in the job, 3)
#
# Resources: edit the #SBATCH lines or override on the command line. --qos=gpu
# allows at most 4 GPUs per job:
#   sbatch --gres=gpu:4 --cpus-per-task=16 scripts/run_pipeline.sbatch data/raw/original.mp4 4
#
# Env knobs (prefix the sbatch command):
#   CONFIG=configs/hpc_centroid.yaml  OVERRIDES="detection.confidence=0.3"  PIPELINE=centroid
#
# Account: scripts/bunya_env.sh exports SBATCH_ACCOUNT, so --account is picked
# up automatically. Without it:  sbatch --account=a_<group> scripts/run_pipeline.sbatch ...
#
# Monitor:  squeue -u $USER      tail -f outputs/slurm/pipeline_<JOBID>.out
# Cancel:   scancel <JOB_ID>
# Results:  outputs/runs/<timestamp>_centroid_batch/  (merged video + contacts;
#           the end of the job log prints the scp commands to download them)
###############################################################################

#SBATCH --job-name=centroid-pipeline
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/pipeline_%j.out
#SBATCH --error=outputs/slurm/pipeline_%j.err

set -euo pipefail

VIDEO="${1:-data/raw/original_120s.avi}"
NUM_CHUNKS="${2:-${SLURM_GPUS_ON_NODE:-3}}"
CONFIG="${CONFIG:-configs/hpc_centroid.yaml}"
OVERRIDES="${OVERRIDES:-}"
PIPELINE="${PIPELINE:-centroid}"

cd "${BUNYA_PROJECT_DIR:-$HOME/computer-vision-tracking}"
# shellcheck disable=SC1091
source scripts/bunya_env.sh
bunya_activate

echo "============================================"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  Account:   ${SLURM_JOB_ACCOUNT:-?}"
echo "  GPUs:      ${CUDA_VISIBLE_DEVICES:-none}  ($(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true) visible)"
echo "  Video:     $VIDEO"
echo "  Chunks:    $NUM_CHUNKS"
echo "  Config:    $CONFIG"
echo "  Pipeline:  $PIPELINE"
echo "  Python:    $(python --version 2>&1)"
echo "  PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  TMPDIR:    ${TMPDIR:-unset}"
echo "  Date:      $(date)"
echo "============================================"

bash scripts/run_parallel.sh "$VIDEO" "$NUM_CHUNKS" "$CONFIG" "$OVERRIDES" "$PIPELINE"

echo ""
echo "Job finished at: $(date)"
