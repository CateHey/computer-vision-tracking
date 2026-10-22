#!/bin/bash --login
#SBATCH --job-name=centroid-chunk
#SBATCH --account=a_qbi
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/chunk_%a_%j.out
#SBATCH --error=outputs/slurm/chunk_%a_%j.err

# ============================================================================
# Parallel chunk processing via Slurm array jobs (one GPU per array task)
# ============================================================================
#
# Alternative to scripts/run_pipeline.sbatch for when 3-4 GPUs on a single
# node are hard to get: Slurm schedules each chunk independently. Each array
# task computes its own frame range from SLURM_ARRAY_TASK_ID.
#
# Usage (4 chunks; --qos=gpu runs at most 4 at once):
#   cd ~/computer-vision-tracking
#   VIDEO=data/raw/original.mp4 sbatch --array=0-3 slurm/run_chunks.sbatch
#
# Extra overrides:
#   VIDEO=data/raw/original.mp4 OVERRIDES="detection.confidence=0.3" \
#     sbatch --array=0-3 slurm/run_chunks.sbatch
#
# Skip the frame-count probe by giving it directly:
#   VIDEO=data/raw/original.mp4 TOTAL_FRAMES=30000 sbatch --array=0-3 slurm/run_chunks.sbatch
#
# After every task finishes, merge the chunk run dirs (check names with ls -lt):
#   python scripts/merge_chunks.py outputs/runs/<date>_*_chunk*/ -o outputs/runs/<date>_merged
# ============================================================================

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit with: sbatch --array=0-<N-1> slurm/run_chunks.sbatch}"

CONFIG="${CONFIG:-configs/hpc_centroid.yaml}"
PIPELINE="${PIPELINE:-src.pipelines.centroid.run}"
VIDEO="${VIDEO:-}"
TOTAL_FRAMES="${TOTAL_FRAMES:-}"
OVERRIDES="${OVERRIDES:-}"

# Same default as scripts/run_parallel.sh: contacts on for the centroid pipeline.
CONTACTS_FLAG=""
case "$PIPELINE" in *centroid*) CONTACTS_FLAG="contacts.enabled=true" ;; esac

cd "$HOME/computer-vision-tracking"
module load miniforge/26.3.2-1
eval "$(conda shell.bash hook)"
conda activate rattrack

if [ -n "$VIDEO" ]; then
    OVERRIDES="video_path=$VIDEO $OVERRIDES"
fi
if [ -z "$TOTAL_FRAMES" ]; then
    if [ -z "$VIDEO" ]; then
        echo "ERROR: set VIDEO=<path> (frame count is read from it) or TOTAL_FRAMES=<n>" >&2
        exit 1
    fi
    TOTAL_FRAMES=$(python -c "
import cv2, sys
cap = cv2.VideoCapture('$VIDEO')
if not cap.isOpened():
    sys.exit('ERROR: cannot open $VIDEO')
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
")
fi

NUM_CHUNKS=$SLURM_ARRAY_TASK_COUNT
CHUNK_ID=$SLURM_ARRAY_TASK_ID
FRAMES_PER_CHUNK=$(( (TOTAL_FRAMES + NUM_CHUNKS - 1) / NUM_CHUNKS ))
START_FRAME=$(( CHUNK_ID * FRAMES_PER_CHUNK ))
END_FRAME=$(( START_FRAME + FRAMES_PER_CHUNK ))
if [ "$END_FRAME" -gt "$TOTAL_FRAMES" ]; then
    END_FRAME=$TOTAL_FRAMES
fi

echo "=== Chunk Processing ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array task:   $CHUNK_ID / $NUM_CHUNKS"
echo "Node:         $SLURM_NODELIST"
echo "Account:      ${SLURM_JOB_ACCOUNT:-?}"
echo "GPU:          ${CUDA_VISIBLE_DEVICES:-none}"
echo "Pipeline:     $PIPELINE"
echo "Config:       $CONFIG"
echo "Overrides:    ${CONTACTS_FLAG} ${OVERRIDES:-<none>}"
echo "Frame range:  $START_FRAME -> $END_FRAME  (of $TOTAL_FRAMES)"
echo "========================="

# shellcheck disable=SC2086
python -m "$PIPELINE" \
    --config "$CONFIG" \
    --start-frame "$START_FRAME" \
    --end-frame "$END_FRAME" \
    --chunk-id "$CHUNK_ID" \
    $CONTACTS_FLAG $OVERRIDES

echo "=== Chunk $CHUNK_ID complete ==="
