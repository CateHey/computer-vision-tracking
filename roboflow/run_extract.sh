#!/bin/bash
# ============================================================================
# Extract interaction frames in parallel on Bunya HPC
# ============================================================================
#
# Usage (inside salloc with GPUs):
#   bash roboflow/run_extract.sh data/raw/original.mp4
#   bash roboflow/run_extract.sh data/raw/original.mp4 2 configs/hpc_extract.yaml 4
#
# Arguments:
#   $1 - Video path (required)
#   $2 - Number of chunks/GPUs (default: 2)
#   $3 - Config file (default: configs/hpc_extract.yaml)
#   $4 - Minutes to scan (default: 4) — limits how much video to process
#
# Output:
#   roboflow/frames/
#   ├── frame_000123.jpg
#   ├── frame_001456.jpg
#   └── metadata.csv
#
# ============================================================================

set -euo pipefail

VIDEO="${1:?Usage: bash roboflow/run_extract.sh <video_path> [num_chunks] [config] [minutes]}"
NUM_CHUNKS="${2:-2}"
CONFIG="${3:-configs/hpc_extract.yaml}"
SCAN_MINUTES="${4:-4}"

# --- Step 0: Validate GPU count ---
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected. Are you inside a salloc with --gres=gpu:N?"
    exit 1
fi
if [ "$NUM_CHUNKS" -gt "$NUM_GPUS" ]; then
    echo "WARNING: Requested $NUM_CHUNKS chunks but only $NUM_GPUS GPU(s) available."
    echo "         Reducing to $NUM_GPUS chunks."
    NUM_CHUNKS=$NUM_GPUS
fi
echo "  GPUs detected: $NUM_GPUS"

# --- Step 1: Create batch directory ---
BATCH_ID="$(date +%Y-%m-%d_%H%M%S)_extract_batch"
BATCH_DIR="outputs/runs/$BATCH_ID"
CHUNKS_DIR="$BATCH_DIR/chunks"
LOGS_DIR="$BATCH_DIR/logs"
mkdir -p "$CHUNKS_DIR" "$LOGS_DIR"

# --- Step 2: Get total frames from video ---
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

# Cap to requested scan duration
# Estimate FPS from video (default 30 if unknown)
VIDEO_FPS=$(python -c "
import cv2
cap = cv2.VideoCapture('$VIDEO')
fps = cap.get(cv2.CAP_PROP_FPS)
print(int(fps) if fps > 0 else 30)
cap.release()
")
MAX_SCAN_FRAMES=$(( SCAN_MINUTES * 60 * VIDEO_FPS ))
if [ "$MAX_SCAN_FRAMES" -lt "$TOTAL_FRAMES" ]; then
    echo "  Limiting scan to ${SCAN_MINUTES} min ($MAX_SCAN_FRAMES frames of $TOTAL_FRAMES)"
    TOTAL_FRAMES=$MAX_SCAN_FRAMES
fi

# --- Step 3: Calculate frame ranges ---
FRAMES_PER_CHUNK=$(( (TOTAL_FRAMES + NUM_CHUNKS - 1) / NUM_CHUNKS ))

echo ""
echo "=== Interaction Frame Extraction Plan ==="
echo "  Batch ID:   $BATCH_ID"
echo "  Batch dir:  $BATCH_DIR"
echo "  Config:     $CONFIG"
echo "  Video:      $VIDEO"
echo "  Scan:       ${SCAN_MINUTES} min ($TOTAL_FRAMES frames)"
echo "  Chunks:     $NUM_CHUNKS"
echo "  Per chunk:  ~$FRAMES_PER_CHUNK frames"
echo ""

BATCH_START=$(date +%s)

# --- Step 4: Launch scan chunks in parallel ---
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

    CUDA_VISIBLE_DEVICES=$i python roboflow/extract_interaction_frames.py scan \
        --config "$CONFIG" \
        --start-frame $START \
        --end-frame $END \
        --chunk-id $i \
        output_dir="$CHUNKS_DIR" \
        video_path="$VIDEO" \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $NUM_CHUNKS scan chunks launched."
echo ""

# --- Step 5: Live progress monitor ---
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
        n=$(grep -oP 'Scan complete\. \K[0-9]+(?= frames)' "$log" 2>/dev/null | tail -1)
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
    elif grep -q "Scan complete" "$log" 2>/dev/null; then
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

# --- Final status check ---
FAILED=0
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    if [ "${CHUNK_STATUS[$i]}" = "failed" ]; then
        FAILED=$((FAILED + 1))
    fi
done

SCAN_ELAPSED=$(( $(date +%s) - BATCH_START ))
SCAN_MINS=$((SCAN_ELAPSED / 60))
SCAN_SECS=$((SCAN_ELAPSED % 60))

echo ""
echo "  Scan time: ${SCAN_MINS}m ${SCAN_SECS}s"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "=== $FAILED chunk(s) failed. Check logs in $LOGS_DIR/ ==="
    exit 1
fi

# --- Step 6: Select and extract frames ---
echo ""
echo "=== All chunks completed. Selecting and extracting frames... ==="
FRAMES_DIR="roboflow/frames"
python roboflow/extract_interaction_frames.py select \
    --config "$CONFIG" \
    --scores-dir "$CHUNKS_DIR" \
    --output-dir "$FRAMES_DIR" \
    video_path="$VIDEO"

TOTAL_ELAPSED=$(( $(date +%s) - BATCH_START ))
TOTAL_MINS=$((TOTAL_ELAPSED / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

# --- Step 7: Print summary and download commands ---
REMOTE_USER="${USER:-s4948012}"
REMOTE_HOST="bunya.rcc.uq.edu.au"
ABS_FRAMES_DIR="$(cd "$FRAMES_DIR" && pwd)"
FRAME_COUNT=$(ls -1 "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)
LOCAL_DIR="C:\\Users\\CatherineVaras\\Downloads\\yolo-sam2-lab-tracking\\roboflow"

echo ""
echo "============================================================"
echo "  EXTRACTION COMPLETE"
echo "  TOTAL TIME:   ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "  FRAMES DIR:   $ABS_FRAMES_DIR"
echo "  FRAME COUNT:  $FRAME_COUNT"
echo "============================================================"
echo ""
echo "# Download frames (PowerShell):"
echo "scp -r ${REMOTE_USER}@${REMOTE_HOST}:${ABS_FRAMES_DIR} \`"
echo "  \"${LOCAL_DIR}\""
echo ""
