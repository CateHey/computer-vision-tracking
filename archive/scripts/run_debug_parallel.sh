#!/bin/bash
# ============================================================================
# Run debug scripts (yolo_only or sam2_only) in parallel chunks on HPC
# ============================================================================
#
# Usage (inside salloc with GPUs):
#   bash scripts/run_debug_parallel.sh yolo data/raw/original_120s.avi
#   bash scripts/run_debug_parallel.sh sam2 data/raw/original_120s.avi
#   bash scripts/run_debug_parallel.sh sam2_no_yolo data/raw/original_120s.avi
#   bash scripts/run_debug_parallel.sh sam2 data/raw/original_120s.avi 4
#   bash scripts/run_debug_parallel.sh sam2 data/raw/original_120s.avi 4 configs/hpc_reference.yaml
#   bash scripts/run_debug_parallel.sh sam2 data/raw/original_120s.avi 4 configs/hpc_reference.yaml "--no-fallback"
#   bash scripts/run_debug_parallel.sh sam2_no_yolo data/raw/original_120s.avi 4 configs/hpc_reference.yaml "--reinit-every 300"
#
# Arguments:
#   $1 - Mode: "yolo", "sam2", or "sam2_no_yolo" (required)
#   $2 - Video path (required)
#   $3 - Number of chunks/GPUs (default: auto-detect)
#   $4 - Config file (default: configs/hpc_reference.yaml)
#   $5 - Extra flags for the debug script (optional, e.g., "--no-fallback")
#
# ============================================================================

set -euo pipefail

MODE="${1:?Usage: bash scripts/run_debug_parallel.sh <yolo|sam2|sam2_no_yolo> <video_path> [num_chunks] [config] [extra_flags]}"
VIDEO="${2:?Usage: bash scripts/run_debug_parallel.sh <yolo|sam2|sam2_no_yolo> <video_path> [num_chunks] [config] [extra_flags]}"
NUM_CHUNKS="${3:-0}"
CONFIG="${4:-configs/hpc_reference.yaml}"
EXTRA_FLAGS="${5:-}"

# Validate mode
case "$MODE" in
    yolo)
        SCRIPT="scripts/debug_yolo_only.py"
        TAG="yolo_debug_batch"
        ;;
    sam2)
        SCRIPT="scripts/debug_sam2_only.py"
        TAG="sam2_debug_batch"
        ;;
    sam2_no_yolo)
        SCRIPT="scripts/debug_sam2_no_yolo.py"
        TAG="sam2_no_yolo_batch"
        ;;
    *)
        echo "ERROR: Mode must be 'yolo', 'sam2', or 'sam2_no_yolo', got '$MODE'"
        exit 1
        ;;
esac

# --- Step 0: Validate GPU count ---
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected. Are you inside a salloc with --gres=gpu:N?"
    exit 1
fi

if [ "$NUM_CHUNKS" -eq 0 ]; then
    NUM_CHUNKS=$NUM_GPUS
fi
if [ "$NUM_CHUNKS" -gt "$NUM_GPUS" ]; then
    echo "WARNING: Requested $NUM_CHUNKS chunks but only $NUM_GPUS GPU(s). Reducing."
    NUM_CHUNKS=$NUM_GPUS
fi
echo "  Mode: $MODE ($SCRIPT)"
echo "  GPUs: $NUM_GPUS, Chunks: $NUM_CHUNKS"

# Load ffmpeg for video merging
module load ffmpeg 2>/dev/null || true

# --- Step 1: Create batch directory ---
BATCH_ID="$(date +%Y-%m-%d_%H%M%S)_${TAG}"
BATCH_DIR="outputs/runs/$BATCH_ID"
CHUNKS_DIR="$BATCH_DIR/chunks"
LOGS_DIR="$BATCH_DIR/logs"
mkdir -p "$CHUNKS_DIR" "$LOGS_DIR"

# --- Step 2: Get total frames ---
echo "Analyzing video: $VIDEO"
TOTAL_FRAMES=$(python -c "
import cv2, sys
cap = cv2.VideoCapture('$VIDEO')
if not cap.isOpened():
    print('ERROR: cannot open video', file=sys.stderr)
    sys.exit(1)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total / fps if fps > 0 else 0
print(total)
print(f'  Resolution: {w}x{h}', file=sys.stderr)
print(f'  FPS: {fps:.1f}', file=sys.stderr)
print(f'  Total frames: {total}', file=sys.stderr)
print(f'  Duration: {duration:.1f}s ({duration/60:.1f} min)', file=sys.stderr)
cap.release()
")

if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" -eq 0 ]; then
    echo "ERROR: Could not read frame count from video"
    exit 1
fi

# --- Step 3: Calculate frame ranges ---
FRAMES_PER_CHUNK=$(( (TOTAL_FRAMES + NUM_CHUNKS - 1) / NUM_CHUNKS ))

echo ""
echo "=== Debug Parallel Execution ==="
echo "  Batch ID:   $BATCH_ID"
echo "  Script:     $SCRIPT"
echo "  Config:     $CONFIG"
echo "  Video:      $VIDEO"
echo "  Frames:     $TOTAL_FRAMES"
echo "  Chunks:     $NUM_CHUNKS"
echo "  Per chunk:  ~$FRAMES_PER_CHUNK frames"
if [ -n "$EXTRA_FLAGS" ]; then
    echo "  Flags:      $EXTRA_FLAGS"
fi
echo ""

BATCH_START=$(date +%s)

# --- Step 4: Launch chunks in parallel ---
PIDS=()
LOGS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    START=$((i * FRAMES_PER_CHUNK))
    END=$(( (i + 1) * FRAMES_PER_CHUNK ))
    if [ $END -gt $TOTAL_FRAMES ]; then
        END=$TOTAL_FRAMES
    fi

    LOG="$LOGS_DIR/chunk_${i}.log"
    LOGS+=("$LOG")
    echo "  Chunk $i: frames $START -> $END (GPU $i)"

    CUDA_VISIBLE_DEVICES=$i python "$SCRIPT" \
        --config "$CONFIG" \
        --start-frame $START \
        --end-frame $END \
        --chunk-id $i \
        output_dir="$CHUNKS_DIR" \
        video_path="$VIDEO" $EXTRA_FLAGS \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $NUM_CHUNKS chunks launched."
echo ""

# --- Step 5: Progress monitor (reuse same pattern as run_parallel.sh) ---
_progress_bar() {
    local current=$1 total=$2 width=30
    local pct=0
    if [ "$total" -gt 0 ]; then
        pct=$(( current * 100 / total ))
    fi
    local filled=$(( pct * width / 100 ))
    local empty=$(( width - filled ))
    local bar=""
    for ((b=0; b<filled; b++)); do bar+="█"; done
    for ((b=0; b<empty;  b++)); do bar+="░"; done
    printf "%s %3d%%" "$bar" "$pct"
}

_get_chunk_frames() {
    local log="$1"
    if [ ! -f "$log" ]; then
        echo 0
        return
    fi
    local n
    n=$(grep -oP 'Processed \K[0-9]+(?= frames)' "$log" 2>/dev/null | tail -1)
    if [ -z "$n" ]; then
        n=$(grep -oP 'Pipeline complete\. \K[0-9]+(?= frames)' "$log" 2>/dev/null | tail -1)
    fi
    echo "${n:-0}"
}

_chunk_status() {
    local pid=$1 log=$2
    if ! kill -0 "$pid" 2>/dev/null; then
        if wait "$pid" 2>/dev/null; then
            echo "done"
        else
            echo "failed"
        fi
    elif grep -q "Pipeline complete" "$log" 2>/dev/null; then
        echo "done"
    else
        echo "running"
    fi
}

ALL_DONE=0
ITER=0
CHUNK_STATUS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    CHUNK_STATUS+=("running")
done

while [ "$ALL_DONE" -eq 0 ]; do
    sleep 2

    ELAPSED=$(( $(date +%s) - BATCH_START ))
    E_MIN=$((ELAPSED / 60))
    E_SEC=$((ELAPSED % 60))

    ALL_DONE=1
    TOTAL_PROCESSED=0
    STATUS_LINES=""

    for i in $(seq 0 $((NUM_CHUNKS - 1))); do
        LOG="${LOGS[$i]}"
        CHUNK_FRAMES=$(( FRAMES_PER_CHUNK ))
        if [ $i -eq $((NUM_CHUNKS - 1)) ]; then
            CHUNK_FRAMES=$(( TOTAL_FRAMES - i * FRAMES_PER_CHUNK ))
        fi

        PROCESSED=$(_get_chunk_frames "$LOG")
        TOTAL_PROCESSED=$((TOTAL_PROCESSED + PROCESSED))

        if [ "${CHUNK_STATUS[$i]}" = "running" ]; then
            CHUNK_STATUS[$i]=$(_chunk_status "${PIDS[$i]}" "$LOG")
        fi

        cur_status="${CHUNK_STATUS[$i]}"
        case "$cur_status" in
            running)
                ALL_DONE=0
                BAR=$(_progress_bar "$PROCESSED" "$CHUNK_FRAMES")
                STATUS_LINES+="  Chunk $i: $BAR  ($PROCESSED/$CHUNK_FRAMES frames)\n"
                ;;
            done)
                STATUS_LINES+="  Chunk $i: ████████████████████████████████ 100%  DONE\n"
                ;;
            failed)
                STATUS_LINES+="  Chunk $i: FAILED — see $LOG\n"
                ;;
        esac
    done

    OVERALL=$(_progress_bar "$TOTAL_PROCESSED" "$TOTAL_FRAMES")

    ETA_STR=""
    if [ "$TOTAL_PROCESSED" -gt 0 ] && [ "$ELAPSED" -gt 5 ]; then
        REMAINING=$(( TOTAL_FRAMES - TOTAL_PROCESSED ))
        FPS_100=$(( TOTAL_PROCESSED * 100 / ELAPSED ))
        if [ "$FPS_100" -gt 0 ]; then
            ETA_SECS=$(( REMAINING * 100 / FPS_100 ))
            ETA_MIN=$((ETA_SECS / 60))
            ETA_SEC=$((ETA_SECS % 60))
            REAL_FPS=$(( FPS_100 / 100 ))
            REAL_FPS_DEC=$(( (FPS_100 % 100) / 10 ))
            ETA_STR="  ETA: ${ETA_MIN}m ${ETA_SEC}s  (~${REAL_FPS}.${REAL_FPS_DEC} FPS)"
        fi
    fi

    if [ "$ITER" -gt 0 ]; then
        DISPLAY_LINES=$(( NUM_CHUNKS + 3 ))
        for ((l=0; l<DISPLAY_LINES; l++)); do
            printf "\033[A\033[2K"
        done
    fi

    printf "  === Progress [%dm %ds elapsed] ===\n" "$E_MIN" "$E_SEC"
    printf "%b" "$STATUS_LINES"
    printf "  Overall: %s  (%d/%d frames)\n" "$OVERALL" "$TOTAL_PROCESSED" "$TOTAL_FRAMES"
    printf "%s\n" "$ETA_STR"

    ITER=$((ITER + 1))
done

# --- Step 6: Check results ---
FAILED=0
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    if [ "${CHUNK_STATUS[$i]}" = "failed" ]; then
        FAILED=$((FAILED + 1))
    fi
done

TOTAL_ELAPSED=$(( $(date +%s) - BATCH_START ))
TOTAL_MINS=$((TOTAL_ELAPSED / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

echo ""
echo "  Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "=== $FAILED chunk(s) failed. Check logs in $LOGS_DIR/ ==="
    exit 1
fi

# --- Step 7: Find chunk videos and merge ---
echo ""
echo "=== All chunks completed. Merging video... ==="

CHUNK_VIDEOS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    LOG="${LOGS[$i]}"
    RUN_DIR=$(grep -oP 'Run directory: \K.*' "$LOG" 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ]; then
        # Find overlay video in this chunk's run dir
        VID=$(find "$RUN_DIR/overlays" -name "*.avi" 2>/dev/null | head -1)
        if [ -n "$VID" ]; then
            CHUNK_VIDEOS+=("$VID")
            echo "  Chunk $i: $VID"
        else
            echo "  Chunk $i: WARNING — no overlay video found in $RUN_DIR"
        fi
    else
        echo "  Chunk $i: WARNING — could not find run directory in log"
    fi
done

if [ ${#CHUNK_VIDEOS[@]} -eq 0 ]; then
    echo "ERROR: No chunk videos found."
    exit 1
fi

# Merge with ffmpeg (stream copy, no re-encode)
MERGED_DIR="$BATCH_DIR/overlays"
mkdir -p "$MERGED_DIR"
MERGED_VIDEO="$MERGED_DIR/${MODE}_debug_merged.avi"

if [ ${#CHUNK_VIDEOS[@]} -eq 1 ]; then
    cp "${CHUNK_VIDEOS[0]}" "$MERGED_VIDEO"
else
    FILELIST="$BATCH_DIR/_filelist.txt"
    for v in "${CHUNK_VIDEOS[@]}"; do
        echo "file '$(realpath "$v")'" >> "$FILELIST"
    done
    ffmpeg -f concat -safe 0 -i "$FILELIST" -c copy "$MERGED_VIDEO" -y 2>/dev/null
    rm -f "$FILELIST"
fi

ABS_MERGED="$(cd "$(dirname "$MERGED_VIDEO")" && pwd)/$(basename "$MERGED_VIDEO")"

# --- Step 8: Download commands ---
REMOTE_USER="${USER:-s4948012}"
REMOTE_HOST="bunya.rcc.uq.edu.au"
LOCAL_DIR="C:\\Users\\CatherineVaras\\Downloads\\yolo-sam2-lab-tracking\\outputs"

echo ""
echo "============================================================"
echo "  DEBUG BATCH COMPLETE: $BATCH_ID"
echo "  MODE:           $MODE"
echo "  TOTAL TIME:     ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "  MERGED VIDEO:   $ABS_MERGED"
echo "============================================================"
echo ""
echo "# Download (PowerShell):"
echo "scp ${REMOTE_USER}@${REMOTE_HOST}:${ABS_MERGED} \\"
echo "  \"${LOCAL_DIR}\\\""
echo ""
