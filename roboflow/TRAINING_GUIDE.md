# YOLO Pose Training Guide - Bunya HPC

Complete guide to configure, train, and retrieve a YOLO pose model on Bunya.

---

## Files

| File | What it does |
|------|-------------|
| `train_yolo_pose.py` | Python script: downloads Roboflow data, trains model, validates |
| `train_yolo_pose.sbatch` | Slurm job script: allocates GPU, sets up env, runs training |

---

## Quick Start

### From your local machine (PowerShell)

```powershell
# 1. SSH into Bunya FIRST (nothing works without this)
ssh s4948012@bunya.rcc.uq.edu.au
```

### On Bunya (after SSH)

```bash
# 2. Pull latest code (includes roboflow/train_yolo_pose.py)
cd ~/Balbi/yolo-sam2-lab-tracking
git pull

# 3. Request a GPU with salloc (interactive session)
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00
srun --pty bash

# 4. Setup environment
cd ~/Balbi/yolo-sam2-lab-tracking
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate
export TMPDIR=$HOME/tmp && mkdir -p $TMPDIR

# 5. Run training (script is inside roboflow/ folder)
python roboflow/train_yolo_pose.py --epochs 100 --batch 16 --imgsz 640

# 6. When done, the trained model will be at:
#    ~/Balbi/yolo-sam2-lab-tracking/runs/pose/train_ratslabs/weights/best.pt
```

### Alternative: sbatch (non-interactive, runs in background)

If you want to disconnect and let it run:

```bash
# On Bunya (after SSH, no salloc needed)
cd ~/Balbi/yolo-sam2-lab-tracking
mkdir -p outputs/slurm
sbatch roboflow/train_yolo_pose.sbatch

# Monitor from anywhere
squeue -u $USER
tail -f outputs/slurm/yolo_train_*.out
```

### Progress bar during training

You'll see a live bar after each epoch:

```
  TRAINING STARTED
  Total epochs: 100

  Epoch  12/100 [============>............................] 12.0%
    loss: 0.0452 | elapsed: 8m23s | remaining: ~61m45s
  Epoch  13/100 [=============>...........................] 13.0%
    loss: 0.0431 | elapsed: 9m05s | remaining: ~60m50s
```

---

## Configuration

### What to change in `train_yolo_pose.py`

Open the file and edit the top section:

```python
# Roboflow settings
ROBOFLOW_API_KEY = "kD254mWPx6WlP2phIeC4"   # your API key
ROBOFLOW_WORKSPACE = "modelos-yolo"           # your workspace name
ROBOFLOW_PROJECT = "pruebasratslabs-c02t9"    # your project slug
ROBOFLOW_VERSION = 6                          # dataset version number

# Training hyperparameters
MODEL_NAME = "yolo11s-pose.pt"   # which model to train (see table below)
EPOCHS = 100                     # how many times to go through all images
IMG_SIZE = 640                   # image resolution (640 is standard)
BATCH_SIZE = 16                  # images per batch (depends on GPU memory)
WORKERS = 8                      # data loading threads
PATIENCE = 20                    # stop early if no improvement for 20 epochs
```

### Model sizes (pick one)

| Model | VRAM needed | Speed | When to use |
|-------|------------|-------|-------------|
| `yolo11n-pose.pt` | ~4 GB | Fastest | Quick experiments, small datasets |
| `yolo11s-pose.pt` | ~6 GB | Fast | Good default, balanced |
| `yolo11m-pose.pt` | ~12 GB | Medium | Larger datasets, better accuracy |
| `yolo11l-pose.pt` | ~18 GB | Slow | When accuracy matters most |
| `yolo11x-pose.pt` | ~28 GB | Slowest | Maximum accuracy, needs A100 |

### Batch size by GPU

| GPU | VRAM | Recommended batch |
|-----|------|-------------------|
| A100 40GB | 40 GB | 16 - 32 |
| A100 80GB | 80 GB | 32 - 64 |
| V100 32GB | 32 GB | 8 - 16 |
| T4 16GB | 16 GB | 4 - 8 |

If you get `CUDA out of memory`, reduce `BATCH_SIZE` (or use `--batch 8` from CLI).

### salloc resource flags explained

```bash
salloc \
  --partition=gpu_cuda \   # Bunya's GPU partition
  --qos=gpu \              # GPU quality of service (required)
  --gres=gpu:1 \           # 1 GPU (YOLO trains on single GPU)
  --cpus-per-task=8 \      # 8 CPU cores for data loading (match WORKERS)
  --mem=64G \              # 64GB RAM (for image caching)
  --time=12:00:00          # max 12 hours (increase for 200+ epochs)
```

---

## Command line overrides

You can override any setting without editing the file:

```bash
# Change epochs and batch size
python roboflow/train_yolo_pose.py --epochs 200 --batch 32

# Use a bigger model
python roboflow/train_yolo_pose.py --model yolo11m-pose.pt

# Skip Roboflow download, use existing data
python roboflow/train_yolo_pose.py --data /path/to/data.yaml

# Resume interrupted training (picks up from last.pt)
python roboflow/train_yolo_pose.py --resume
```

---

## Where Results Are Saved (on Bunya)

Training saves everything under the project directory on Bunya:

```
~/Balbi/yolo-sam2-lab-tracking/runs/pose/train_ratslabs/
```

You can check it after training:

```bash
ls ~/Balbi/yolo-sam2-lab-tracking/runs/pose/train_ratslabs/weights/
# best.pt   last.pt
```

---

## Download Trained Model (SCP)

Run these from your **local machine** (PowerShell), NOT from Bunya:

```powershell
# Best weights only (this is the model you use for inference)
scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/runs/pose/train_ratslabs/weights/best.pt `
  "C:\Users\CatherineVaras\Downloads\roboflow\"

# Last checkpoint (to resume training later)
scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/runs/pose/train_ratslabs/weights/last.pt `
  "C:\Users\CatherineVaras\Downloads\roboflow\"

# Full results folder (plots, metrics, everything)
scp -r s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/runs/pose/train_ratslabs/ `
  "C:\Users\CatherineVaras\Downloads\roboflow\"
```

---

## Adding More Images to Roboflow

### Step 1: Upload images

1. Go to [app.roboflow.com](https://app.roboflow.com)
2. Open workspace **modelos-yolo** > project **pruebasratslabs-c02t9**
3. Click **Upload** (top right)
4. Drag and drop images or folders
5. You can also upload video clips (Roboflow extracts frames)

### Step 2: Annotate

- Click on each uploaded image to annotate
- Use the **same keypoint labels** as your existing images (consistency is critical)
- Tip: use Roboflow's auto-label to speed up, then correct mistakes manually
- Aim for variety: different angles, lighting, positions

### Step 3: Generate a new version

1. Go to **Versions** tab
2. Click **Generate New Version**
3. Configure preprocessing and augmentation (see below)
4. Click **Generate**

### Step 4: Recommended preprocessing/augmentation

**Preprocessing** (always apply):
- Auto-Orient: ON
- Resize: Stretch to 640x640

**Augmentation** (recommended for <500 images):
- Flip: Horizontal
- Rotation: Between -15 and +15 degrees
- Brightness: Between -15% and +15%
- Blur: Up to 1px
- Cutout: 3 boxes, 5% size each (simulates occlusion)

**Do NOT over-augment** if you have 1000+ images. Disable augmentation or use minimal settings.

**Train/Validation/Test split**: 70% / 20% / 10% (Roboflow default is fine)

### Step 5: Update the script and retrain

```python
# In train_yolo_pose.py, change the version number:
ROBOFLOW_VERSION = 7   # new version number from Roboflow
```

On Bunya, delete the old dataset before re-downloading:

```bash
cd ~/Balbi/yolo-sam2-lab-tracking
rm -rf Pruebasratslabs-6
```

Then retrain (salloc method):

```bash
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00
srun --pty bash
cd ~/Balbi/yolo-sam2-lab-tracking
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate
python roboflow/train_yolo_pose.py --epochs 100 --batch 16
```

---

## Training Output Files

After training, `runs/pose/train_ratslabs/` contains:

```
runs/pose/train_ratslabs/
  weights/
    best.pt              <- best model (use this for inference)
    last.pt              <- last checkpoint (use to resume training)
  results.csv            <- epoch-by-epoch metrics
  results.png            <- loss/metric curves plot
  confusion_matrix.png   <- class confusion matrix
  P_curve.png            <- precision curve
  R_curve.png            <- recall curve
  PR_curve.png           <- precision-recall curve
  F1_curve.png           <- F1 score curve
  args.yaml              <- all training arguments used
  val_batch0_pred.jpg    <- sample validation predictions
```

---

## Monitoring Commands

```bash
# If using salloc (interactive) - you see output directly in terminal
# Open another terminal on the same node to check GPU:
nvidia-smi

# If using sbatch (background):
squeue -u $USER                              # check job status
tail -f outputs/slurm/yolo_train_*.out       # watch training log
scancel <JOB_ID>                             # cancel a job
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `--batch` (try 8 or 4) |
| `No module named ultralytics` | Run `pip install ultralytics` in your venv |
| `Dataset not found` | Check `ROBOFLOW_VERSION` matches your version |
| Job stays `PD` (pending) | Wait for GPUs, or try `--time=06:00:00` (shorter jobs get scheduled faster) |
| Training too slow | Use `--model yolo11n-pose.pt` (nano) or increase `--batch` |
| `resume` fails | Make sure `last.pt` exists at `runs/pose/train_ratslabs/weights/last.pt` |
| Low accuracy | Add more diverse images, increase epochs, try a bigger model |
