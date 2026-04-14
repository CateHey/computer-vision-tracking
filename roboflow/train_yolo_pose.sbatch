#!/bin/bash
###############################################################################
# SLURM BATCH SCRIPT - YOLO Pose Training on Bunya HPC
###############################################################################
#
# Submit with:   sbatch train_yolo_pose.sbatch
# Monitor with:  squeue -u $USER
# Cancel with:   scancel <JOB_ID>
# View logs:     tail -f outputs/slurm/yolo_train_<JOBID>.out
#
###############################################################################

# ---- Slurm resource requests ----
#SBATCH --job-name=yolo-pose-train
#SBATCH --partition=gpu_cuda           # GPU partition on Bunya
#SBATCH --qos=gpu                      # GPU quality of service
#SBATCH --gres=gpu:1                   # 1 GPU (YOLO trains on single GPU)
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --mem=64G                      # RAM (64GB for image caching)
#SBATCH --time=12:00:00                # max wall time (12 hours)
#SBATCH --output=outputs/slurm/yolo_train_%j.out   # stdout log
#SBATCH --error=outputs/slurm/yolo_train_%j.err    # stderr log

###############################################################################
# 1. ENVIRONMENT SETUP
###############################################################################

# Create output directory for Slurm logs
mkdir -p outputs/slurm

# Navigate to project directory
cd ~/Balbi/yolo-sam2-lab-tracking

# Load Python module (Bunya-specific)
module load python/3.10.4-gcccore-11.3.0

# Activate the project virtual environment
source .venv/bin/activate

# Set temp directory (avoids /tmp quota issues on HPC)
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

# Print environment info for debugging
echo "============================================"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  GPUs:      $CUDA_VISIBLE_DEVICES"
echo "  Python:    $(python --version)"
echo "  PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU name:  $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "  Date:      $(date)"
echo "============================================"

###############################################################################
# 2. COPY TRAINING SCRIPT TO BUNYA (if not already there)
###############################################################################

# If you haven't copied train_yolo_pose.py to Bunya yet, do it first:
#   scp train_yolo_pose.py s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/

###############################################################################
# 3. RUN TRAINING
###############################################################################

# Train with optimized batch size for A100 GPUs
# Adjust --batch based on your GPU memory:
#   A100 40GB  -> batch 16-32
#   A100 80GB  -> batch 32-64
#   V100 32GB  -> batch 8-16

python roboflow/train_yolo_pose.py \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --model yolo11s-pose.pt \
    --workers 8 \
    --patience 20

###############################################################################
# 4. COMPRESS RESULTS FOR DOWNLOAD
###############################################################################

echo ""
echo "Compressing training results..."

# Find the results directory
RESULTS_DIR="runs/pose/train_ratslabs"

if [ -d "$RESULTS_DIR" ]; then
    # Create a tarball of the trained model and results
    ARCHIVE="outputs/yolo_pose_trained_${SLURM_JOB_ID}.tar.gz"
    tar -czf "$ARCHIVE" \
        "$RESULTS_DIR/weights/best.pt" \
        "$RESULTS_DIR/weights/last.pt" \
        "$RESULTS_DIR/results.csv" \
        "$RESULTS_DIR/results.png" \
        "$RESULTS_DIR/confusion_matrix.png" \
        "$RESULTS_DIR/args.yaml" \
        2>/dev/null

    echo ""
    echo "============================================================"
    echo "  TRAINING COMPLETE"
    echo "  Results:  $RESULTS_DIR"
    echo "  Archive:  $ARCHIVE"
    echo "============================================================"
    echo ""
    echo "# Download trained model to your local machine (PowerShell):"
    echo "scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/$RESULTS_DIR/weights/best.pt \\"
    echo "  \"C:\\Users\\CatherineVaras\\Downloads\\roboflow\\\""
    echo ""
    echo "# Download full results archive:"
    echo "scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/$ARCHIVE \\"
    echo "  \"C:\\Users\\CatherineVaras\\Downloads\\roboflow\\\""
else
    echo "WARNING: Results directory not found at $RESULTS_DIR"
fi

echo ""
echo "Job finished at: $(date)"
