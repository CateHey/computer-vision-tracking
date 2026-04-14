#!/bin/bash
#SBATCH --job-name=sam2-chunk
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/chunk_%a_%j.out
#SBATCH --error=outputs/slurm/chunk_%a_%j.err

# ============================================================================
# Parallel chunk processing via Slurm array jobs
# ============================================================================
#
# Splits a video into N chunks and processes each on a separate GPU.
# Each array task computes its frame range from SLURM_ARRAY_TASK_ID.
#
# Usage (3 GPUs):
#   sbatch --array=0-2 slurm/run_chunks.sbatch
#
# With overrides:
#   OVERRIDES="video_path=data/raw/original_120s.avi contacts.enabled=true" \
#     sbatch --array=0-2 slurm/run_chunks.sbatch
#
# Limit concurrent GPU jobs (e.g., max 2 at a time):
#   sbatch --array=0-2%2 slurm/run_chunks.sbatch
#
# After all chunks finish, merge results:
#   python scripts/merge_chunks.py outputs/runs/*_chunk*/
#
# ============================================================================

set -euo pipefail

# --- Configuration ---
CONFIG="${CONFIG:-configs/hpc_centroid.yaml}"
PIPELINE="${PIPELINE:-src.pipelines.centroid.run}"
TOTAL_FRAMES="${TOTAL_FRAMES:-30000}"    # Total frames in the video
                                          # 20 min @ 25fps = 30000
                                          # Override: --export=...,TOTAL_FRAMES=15000
OVERRIDES="${OVERRIDES:-}"               # Config overrides (space-separated key=value)
                                          # Override: --export=...,OVERRIDES="video_path=... contacts.enabled=true"

# --- Compute frame range for this chunk ---
NUM_CHUNKS=$SLURM_ARRAY_TASK_COUNT       # Total tasks in the array
CHUNK_ID=$SLURM_ARRAY_TASK_ID            # This task's index (0-based)

FRAMES_PER_CHUNK=$(( (TOTAL_FRAMES + NUM_CHUNKS - 1) / NUM_CHUNKS ))
START_FRAME=$(( CHUNK_ID * FRAMES_PER_CHUNK ))
END_FRAME=$(( START_FRAME + FRAMES_PER_CHUNK ))

# Clamp end_frame to total
if [ $END_FRAME -gt $TOTAL_FRAMES ]; then
    END_FRAME=$TOTAL_FRAMES
fi

echo "=== Chunk Processing ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array task:   $CHUNK_ID / $NUM_CHUNKS"
echo "Node:         $SLURM_NODELIST"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Pipeline:     $PIPELINE"
echo "Config:       $CONFIG"
echo "Overrides:    ${OVERRIDES:-<none>}"
echo "Frame range:  $START_FRAME → $END_FRAME ($FRAMES_PER_CHUNK frames)"
echo "========================="

# --- Setup environment ---
cd ~/Balbi/yolo-sam2-lab-tracking

module load python/3.10.4-gcccore-11.3.0 2>/dev/null || true
source .venv/bin/activate

mkdir -p outputs/slurm

# --- Run pipeline for this chunk ---
# shellcheck disable=SC2086
python -m $PIPELINE \
    --config "$CONFIG" \
    --start-frame $START_FRAME \
    --end-frame $END_FRAME \
    --chunk-id $CHUNK_ID \
    $OVERRIDES

echo "=== Chunk $CHUNK_ID complete ==="
