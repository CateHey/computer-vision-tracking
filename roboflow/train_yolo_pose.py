"""
YOLO Pose Training Script for Bunya HPC
========================================
Trains a YOLO pose estimation model using a Roboflow dataset.
Designed to run on UQ's Bunya HPC with GPU acceleration.

Usage (standalone):
    python train_yolo_pose.py

Usage (via Slurm):
    sbatch train_yolo_pose.sbatch
"""

import os
import sys
import time
import argparse

# ---------------------------------------------------------------------------
# 1. CONFIGURATION  (edit these values as needed)
# ---------------------------------------------------------------------------

# Roboflow settings
ROBOFLOW_API_KEY = "kD254mWPx6WlP2phIeC4"
ROBOFLOW_WORKSPACE = "modelos-yolo"
ROBOFLOW_PROJECT = "pruebasratslabs-c02t9"
ROBOFLOW_VERSION = 7
DATASET_FORMAT = "yolov8"          # ultralytics-compatible format

# Training hyperparameters
MODEL_NAME = "yolo11s-pose.pt"     # pre-trained pose model (small variant)
EPOCHS = 100                       # total training epochs
IMG_SIZE = 640                     # input image resolution
BATCH_SIZE = 16                    # adjust based on GPU memory (16 for 40GB A100)
WORKERS = 8                        # dataloader workers (match --cpus-per-task)
PATIENCE = 20                      # early stopping patience (0 = disabled)

# Output directory (on Bunya, use scratch or home)
PROJECT_DIR = "runs/pose"          # results saved under this folder
RUN_NAME = "train_ratslabs"        # subfolder name for this run


def parse_args():
    """Allow overriding config from the command line."""
    parser = argparse.ArgumentParser(description="Train YOLO pose model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (skip download)")
    return parser.parse_args()


def download_dataset():
    """Download the dataset from Roboflow (only if not already present)."""
    from roboflow import Roboflow

    # Check if dataset folder already exists to avoid re-downloading
    expected_dir = f"Pruebasratslabs-{ROBOFLOW_VERSION}"
    if os.path.isdir(expected_dir):
        print(f"Dataset already exists at ./{expected_dir}, skipping download.")
        return os.path.join(expected_dir, "data.yaml")

    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(DATASET_FORMAT)

    # dataset.location contains the path to the downloaded folder
    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"Dataset downloaded to: {dataset.location}")
    return data_yaml


def make_progress_bar(current, total, bar_len=40, label=""):
    """Print a visual progress bar to the terminal.

    Example output:
      Epoch 23/100 [===============>........................] 23% | loss: 0.0345
    """
    fraction = current / total if total > 0 else 0
    filled = int(bar_len * fraction)
    bar = "=" * filled + ">" + "." * (bar_len - filled - 1)
    pct = fraction * 100
    sys.stdout.write(f"\r  {label} [{bar}] {pct:5.1f}%")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def register_progress_callbacks(model, total_epochs):
    """Register callbacks that print a progress bar after each epoch.

    Ultralytics fires 'on_train_epoch_end' after every epoch. We hook into
    that to display a live bar showing how far training has progressed, plus
    the current loss value.
    """
    state = {"start_time": None}

    def on_train_start(trainer):
        state["start_time"] = time.time()
        print("\n" + "=" * 60)
        print("  TRAINING STARTED")
        print(f"  Total epochs: {total_epochs}")
        print("=" * 60 + "\n")

    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1  # 0-indexed -> 1-indexed
        # Extract current loss from trainer
        loss_val = trainer.loss.item() if hasattr(trainer.loss, 'item') else trainer.loss
        elapsed = time.time() - state["start_time"]
        # Estimate remaining time
        per_epoch = elapsed / epoch
        remaining = per_epoch * (total_epochs - epoch)
        mins_left = int(remaining // 60)
        secs_left = int(remaining % 60)

        label = f"Epoch {epoch:3d}/{total_epochs}"
        make_progress_bar(epoch, total_epochs, bar_len=40, label=label)
        print(f"    loss: {loss_val:.4f} | elapsed: {int(elapsed//60)}m{int(elapsed%60):02d}s | remaining: ~{mins_left}m{secs_left:02d}s")

    def on_train_end(trainer):
        total_time = time.time() - state["start_time"]
        mins = int(total_time // 60)
        secs = int(total_time % 60)
        print(f"\n  Total training time: {mins}m {secs:02d}s")

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def train(args):
    """Run YOLO pose model training."""
    from ultralytics import YOLO

    # -----------------------------------------------------------------------
    # 2. DOWNLOAD DATASET (or use provided path)
    # -----------------------------------------------------------------------
    if args.data:
        data_yaml = args.data
    else:
        data_yaml = download_dataset()

    if not os.path.isfile(data_yaml):
        print(f"ERROR: data.yaml not found at {data_yaml}")
        sys.exit(1)

    print(f"Using data config: {data_yaml}")

    # -----------------------------------------------------------------------
    # 3. LOAD MODEL
    # -----------------------------------------------------------------------
    if args.resume:
        # Resume from last checkpoint
        checkpoint = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "last.pt")
        if not os.path.isfile(checkpoint):
            print(f"ERROR: No checkpoint found at {checkpoint}")
            sys.exit(1)
        print(f"Resuming training from: {checkpoint}")
        model = YOLO(checkpoint)
    else:
        # Start from pre-trained model
        print(f"Loading pre-trained model: {args.model}")
        model = YOLO(args.model)

    # -----------------------------------------------------------------------
    # 4. REGISTER PROGRESS BAR + TRAIN
    # -----------------------------------------------------------------------
    register_progress_callbacks(model, args.epochs)

    # Key parameters explained:
    #   data       -> path to data.yaml with train/val/test splits
    #   epochs     -> number of full passes through the training set
    #   imgsz      -> images resized to this resolution (square)
    #   batch      -> images per batch (higher = faster but more VRAM)
    #   workers    -> parallel data loading threads
    #   patience   -> stop early if no improvement for N epochs
    #   device     -> auto-detects GPU; set to "0" for specific GPU
    #   project    -> parent directory for saving results
    #   name       -> subdirectory name for this specific run
    #   exist_ok   -> True = overwrite previous run with same name
    #   amp        -> mixed precision training (faster on modern GPUs)
    #   cache      -> cache images in RAM for faster training
    #   plots      -> generate training plots (loss curves, PR curves)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=0,                  # use GPU 0 (use "cpu" if no GPU)
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        amp=True,                  # mixed precision for speed
        cache=True,                # cache images in RAM
        plots=True,                # save training plots
        save=True,                 # save checkpoints
        save_period=10,            # save checkpoint every 10 epochs
        resume=args.resume,
    )

    # -----------------------------------------------------------------------
    # 5. RESULTS
    # -----------------------------------------------------------------------
    # Use the trainer's actual save directory (avoids path nesting issues)
    save_dir = str(results.save_dir) if hasattr(results, 'save_dir') else str(model.trainer.save_dir)
    best_weights = os.path.join(save_dir, "weights", "best.pt")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best weights: {best_weights}")
    print(f"  Results dir:  {save_dir}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 6. VALIDATE with best weights
    # -----------------------------------------------------------------------
    if os.path.isfile(best_weights):
        print("\nRunning validation with best weights...")
        best_model = YOLO(best_weights)
        metrics = best_model.val(data=data_yaml)
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
    else:
        print(f"\nWARNING: best.pt not found at {best_weights}")
        print("Training validation was already done above. Check the results directory.")

    return best_weights


if __name__ == "__main__":
    args = parse_args()
    train(args)
