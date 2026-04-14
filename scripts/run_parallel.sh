#!/bin/bash
# ============================================================================
# Run a pipeline in parallel chunks within a salloc session
# ============================================================================
#
# Usage (inside salloc with multiple GPUs):
#   bash scripts/run_parallel.sh data/raw/original_120s.avi
#   bash scripts/run_parallel.sh data/raw/original_120s.avi 3
#   bash scripts/run_parallel.sh data/raw/my_video.avi 3 configs/hpc_centroid.yaml
#   bash scripts/run_parallel.sh data/raw/my_video.avi 3 configs/hpc_centroid.yaml "detection.confidence=0.3"
#   bash scripts/run_parallel.sh data/raw/my_video.avi 4 configs/hpc_isolated_composite.yaml "" isolated_composite
#
# Arguments:
#   $1 - Video path (required)
#   $2 - Number of chunks/GPUs (default: 3)
#   $3 - Config file (default: configs/hpc_centroid.yaml)
#   $4 - Extra overrides (optional, space-separated key=value)
#   $5 - Pipeline name (default: centroid) — resolves to src.pipelines.<name>.run
#
# Output structure:
#   outputs/runs/<BATCH_ID>/
#   ├── chunks/
#   │   ├── chunk_0/  (overlays, contacts, logs per chunk)
#   │   ├── chunk_1/
#   │   ├── chunk_2/
#   │   └── chunk_3/
#   ├── overlays/
#   │   └── <pipeline>_<model>_<date>_merged.avi   ← FINAL VIDEO
#   ├── contacts/
#   │   ├── contacts_per_frame.csv
#   │   ├── contact_bouts.csv
#   │   └── session_summary.json
#   └── logs/
#       └── run.log
#
# ============================================================================

set -euo pipefail

VIDEO="${1:?Usage: bash scripts/run_parallel.sh <video_path> [num_chunks] [config] [overrides] [pipeline]}"
NUM_CHUNKS="${2:-3}"
CONFIG="${3:-configs/hpc_centroid.yaml}"
EXTRA_OVERRIDES="${4:-}"
PIPELINE="${5:-centroid}"
PIPELINE_MODULE="src.pipelines.${PIPELINE}.run"

# Pipelines that expose the contacts feature (injected as a default override).
# Other pipelines should not receive contacts.enabled=true.
PIPELINE_CONTACTS_FLAG=""
case "$PIPELINE" in
    centroid)
        PIPELINE_CONTACTS_FLAG="contacts.enabled=true"
        ;;
esac

# --- Step 0: Validate GPU count ---
# Count only "GPU N:" lines, not MIG sub-device lines
# Note: grep -c returns exit code 1 when count=0, so || true prevents pipefail
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

# Load ffmpeg for video merging (stream copy, no re-encode)
module load ffmpeg 2>/dev/null || true

# --- Step 1: Create batch directory ---
BATCH_ID="$(date +%Y-%m-%d_%H%M%S)_${PIPELINE}_batch"
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

# --- Step 3: Calculate frame ranges ---
FRAMES_PER_CHUNK=$(( (TOTAL_FRAMES + NUM_CHUNKS - 1) / NUM_CHUNKS ))

echo ""
echo "=== Parallel Execution Plan ==="
echo "  Pipeline:   $PIPELINE ($PIPELINE_MODULE)"
echo "  Batch ID:   $BATCH_ID"
echo "  Batch dir:  $BATCH_DIR"
echo "  Config:     $CONFIG"
echo "  Video:      $VIDEO"
echo "  Frames:     $TOTAL_FRAMES"
echo "  Chunks:     $NUM_CHUNKS"
echo "  Per chunk:  ~$FRAMES_PER_CHUNK frames"
if [ -n "$EXTRA_OVERRIDES" ]; then
    echo "  Overrides:  $EXTRA_OVERRIDES"
fi
echo ""

BATCH_START=$(date +%s)

# --- Step 4: Launch chunks in parallel ---
PIDS=()
CHUNK_STARTS=()
LOGS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    START=$((i * FRAMES_PER_CHUNK))
    END=$(( (i + 1) * FRAMES_PER_CHUNK ))
    if [ $END -gt $TOTAL_FRAMES ]; then
        END=$TOTAL_FRAMES
    fi

    LOG="$LOGS_DIR/chunk_${i}.log"
    LOGS+=("$LOG")
    CHUNK_STARTS+=("$(date +%s)")
    echo "  Chunk $i: frames $START -> $END (GPU $i)"

    CUDA_VISIBLE_DEVICES=$i python -m $PIPELINE_MODULE \
        --config "$CONFIG" \
        --start-frame $START \
        --end-frame $END \
        --chunk-id $i \
        output_dir="$CHUNKS_DIR" \
        video_path="$VIDEO" $PIPELINE_CONTACTS_FLAG $EXTRA_OVERRIDES \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $NUM_CHUNKS chunks launched."
echo ""

# --- Step 5: Live progress monitor ---
# Parse "Processed N frames" from each chunk log every 2s
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
    # Get the last "Processed N frames" or "Pipeline complete. N frames"
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
        # Process ended — check if success or failure
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

# Monitor loop — refresh every 2 seconds until all chunks finish
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

    # Build status lines
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

        # Update status if still running
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

    # Calculate ETA
    ETA_STR=""
    if [ "$TOTAL_PROCESSED" -gt 0 ] && [ "$ELAPSED" -gt 5 ]; then
        REMAINING=$(( TOTAL_FRAMES - TOTAL_PROCESSED ))
        FPS_100=$(( TOTAL_PROCESSED * 100 / ELAPSED ))  # FPS x 100 for integer math
        if [ "$FPS_100" -gt 0 ]; then
            ETA_SECS=$(( REMAINING * 100 / FPS_100 ))
            ETA_MIN=$((ETA_SECS / 60))
            ETA_SEC=$((ETA_SECS % 60))
            REAL_FPS=$(( FPS_100 / 100 ))
            REAL_FPS_DEC=$(( (FPS_100 % 100) / 10 ))
            ETA_STR="  ETA: ${ETA_MIN}m ${ETA_SEC}s  (~${REAL_FPS}.${REAL_FPS_DEC} FPS)"
        fi
    fi

    # Move cursor up and overwrite previous output (skip on first iteration)
    # Lines: header(1) + chunks(N) + overall(1) + eta(1)
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

# Final status check
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

# --- Step 6: Find chunk directories and merge ---
echo ""
echo "=== All chunks completed. Merging... ==="

CHUNK_DIRS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    RUN_DIR=$(grep -oP 'Run directory: \K.*' "${LOGS[$i]}" 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ]; then
        CHUNK_DIRS+=("$RUN_DIR")
        echo "  Chunk $i: $RUN_DIR"
    else
        echo "  Chunk $i: WARNING - could not find run directory in log"
    fi
done

if [ ${#CHUNK_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No run directories found in logs."
    exit 1
fi

echo ""
python scripts/merge_chunks.py "${CHUNK_DIRS[@]}" -o "$BATCH_DIR"

# --- Step 7: Compress contacts into tar.gz ---
CONTACTS_TAR=""
if [ -d "$BATCH_DIR/contacts" ]; then
    CONTACTS_TAR="$BATCH_DIR/contacts_${BATCH_ID}.tar.gz"
    tar -czf "$CONTACTS_TAR" -C "$BATCH_DIR" contacts/
    echo "Contacts compressed: $CONTACTS_TAR"
fi

# --- Step 8: Print download commands ---
REMOTE_USER="${USER:-s4948012}"
REMOTE_HOST="bunya.rcc.uq.edu.au"
ABS_BATCH_DIR="$(cd "$BATCH_DIR" && pwd)"

# Find the merged video
MERGED_VIDEO=$(find "$BATCH_DIR/overlays" -name "*_merged.*" 2>/dev/null | head -1)
ABS_MERGED=""
if [ -n "$MERGED_VIDEO" ]; then
    ABS_MERGED="$(cd "$(dirname "$MERGED_VIDEO")" && pwd)/$(basename "$MERGED_VIDEO")"
fi

ABS_CONTACTS=""
if [ -n "$CONTACTS_TAR" ]; then
    ABS_CONTACTS="$(cd "$(dirname "$CONTACTS_TAR")" && pwd)/$(basename "$CONTACTS_TAR")"
fi

LOCAL_DIR="C:\\Users\\CatherineVaras\\Downloads\\yolo-sam2-lab-tracking\\outputs"

echo ""
echo "============================================================"
echo "  BATCH COMPLETE: $BATCH_ID"
echo "  TOTAL TIME:     ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "  BATCH DIR:      $ABS_BATCH_DIR"
if [ -n "$ABS_MERGED" ]; then
    echo "  MERGED VIDEO:   $ABS_MERGED"
fi
if [ -n "$ABS_CONTACTS" ]; then
    echo "  CONTACTS:       $ABS_CONTACTS"
fi
echo "============================================================"
echo ""
echo "# Download results (PowerShell):"
if [ -n "$ABS_MERGED" ]; then
    echo "scp ${REMOTE_USER}@${REMOTE_HOST}:${ABS_MERGED} \`"
    echo "  \"${LOCAL_DIR}\""
fi
if [ -n "$ABS_CONTACTS" ]; then
    echo "scp ${REMOTE_USER}@${REMOTE_HOST}:${ABS_CONTACTS} \`"
    echo "  \"${LOCAL_DIR}\""
fi
ABS_EVENT_LOG=""
if [ -f "$BATCH_DIR/contacts/event_log.txt" ]; then
    ABS_EVENT_LOG="$(cd "$BATCH_DIR/contacts" && pwd)/event_log.txt"
    echo ""
    echo "# Quick-view event log (PowerShell):"
    echo "scp ${REMOTE_USER}@${REMOTE_HOST}:${ABS_EVENT_LOG} \`"
    echo "  \"${LOCAL_DIR}\""
fi
echo ""
