# Bunya HPC Guide

Complete guide for running pipelines on UQ's Bunya HPC cluster.

## Table of Contents

- [Quick Start (automated)](#quick-start-automated)
- [Quick Start (manual)](#quick-start-manual)
- [Single Run (1 GPU)](#single-run-1-gpu)
- [Parallel Batch (sbatch)](#parallel-batch-sbatch)
- [run_parallel.sh Reference](#run_parallelsh-reference)
- [Output Structure](#output-structure)
- [First-Time Setup](#first-time-setup)
- [Transfer Files](#transfer-files)
- [Monitoring](#monitoring)
- [Other Run Options](#other-run-options)
- [Slurm Admin Commands](#slurm-admin-commands)
- [Recovery (Finding Last Batch)](#recovery-finding-last-batch)
- [Disk Quota](#disk-quota)
- [Roboflow Extraction](#roboflow-extraction)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (automated)

The fastest way to process a full video. One script does everything.

### Pre-flight: sync code from local machine

```bash
# On your local machine (PowerShell or bash)
git add .
git commit -m "documentation updated"
git push origin
```

### On Bunya

```bash
# 1. Connect to Bunya
ssh s4948012@bunya.rcc.uq.edu.au

# 2. Request 4 GPUs
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=06:00:00
srun --pty bash

# 3. Setup environment
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

# 4. Run (one command does everything: split, process, merge, print download)
bash scripts/run_parallel.sh data/raw/original_120s.avi
```

---

## Quick Start (manual)

If you want to control each chunk manually within a salloc session.

```bash
# 1. Connect and get 4 GPUs
ssh s4948012@bunya.rcc.uq.edu.au
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=06:00:00
srun --pty bash

# 2. Setup
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate
mkdir -p outputs/slurm

# 3. Launch 4 chunks in parallel, each on its own GPU
CUDA_VISIBLE_DEVICES=0 python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 0 --end-frame 7500 --chunk-id 0 \
  video_path=data/raw/original_120s.avi contacts.enabled=true > outputs/slurm/chunk_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 7500 --end-frame 15000 --chunk-id 1 \
  video_path=data/raw/original_120s.avi contacts.enabled=true > outputs/slurm/chunk_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 15000 --end-frame 22500 --chunk-id 2 \
  video_path=data/raw/original_120s.avi contacts.enabled=true > outputs/slurm/chunk_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 22500 --end-frame 30000 --chunk-id 3 \
  video_path=data/raw/original_120s.avi contacts.enabled=true > outputs/slurm/chunk_3.log 2>&1 &

# 4. Wait for all to finish
wait
echo "All chunks done!"

# 5. Check which run directories were created
ls -d outputs/runs/*reference_chunk*

# 6. Merge (replace paths with actual directories from step 5)
python scripts/merge_chunks.py outputs/runs/*_reference_chunk*/
```

**Key**: `CUDA_VISIBLE_DEVICES=N` assigns each process to a different GPU. The `&` runs them in background. `wait` blocks until all finish.

---

## Single Run (1 GPU)

For quick tests or short videos. No chunking needed.

```bash
# 1. Connect and get 1 GPU
ssh s4948012@bunya.rcc.uq.edu.au
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --time=06:00:00 --mem=64G
srun --pty bash

# 2. Setup
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

# 3a. Quick test (2 min clip)
python scripts/trim_video.py data/raw/original.mp4 --duration 120
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  video_path=data/raw/original_120s.avi contacts.enabled=true

# 3b. Or run sam2_yolo pipeline instead
python -m src.pipelines.sam2_yolo.run --config configs/hpc_full.yaml \
  video_path=data/raw/original_120s.avi \
  models.sam2_checkpoint=models/sam2/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt \
  models.sam2_config=configs/sam2.1/sam2.1_hiera_t.yaml \
  contacts.enabled=true
```

---

## Parallel Batch (sbatch)

Non-interactive mode. Submit from the login node and let Slurm schedule it.
No need to stay connected.

```bash
# 1. Connect (login node only, no salloc needed)
ssh s4948012@bunya.rcc.uq.edu.au

# 2. Setup
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

# 3. Trim video if needed (runs on login node, no GPU)
python scripts/trim_video.py data/raw/original.mp4 --duration 120

# 4. Submit 4 parallel chunks (Slurm assigns GPUs automatically)
PIPELINE=src.pipelines.reference.run \
CONFIG=configs/hpc_reference.yaml \
TOTAL_FRAMES=30000 \
OVERRIDES="video_path=data/raw/original_120s.avi contacts.enabled=true" \
  sbatch --array=0-3 slurm/run_chunks.sbatch

# 5. Monitor (can disconnect and check later)
squeue -u $USER
#  JOBID  PARTITION  NAME        ARRAY_TASK  STATE    NODE
#  12345  gpu_cuda   sam2-chunk  0           RUNNING  gpu-node-1
#  12345  gpu_cuda   sam2-chunk  1           RUNNING  gpu-node-2
#  12345  gpu_cuda   sam2-chunk  2           PENDING  (waiting for GPU)
#  12345  gpu_cuda   sam2-chunk  3           PENDING  (waiting for GPU)

# View chunk progress:
cat outputs/slurm/chunk_0_*.out

# 6. When all chunks finish, merge
python scripts/merge_chunks.py outputs/runs/*_reference_chunk*/

# 7. Download results (see Transfer Files section)
```

**Difference vs salloc**: sbatch runs in the background. You can disconnect from Bunya and come back later. Slurm handles GPU scheduling. But you need to merge manually after all chunks finish.

### Alternative: sbatch per-chunk (without array)

If `slurm/run_chunks.sbatch` isn't available, submit each chunk individually:

```bash
cd ~/Balbi/yolo-sam2-lab-tracking

sbatch --job-name=chunk0 --partition=gpu_cuda --qos=gpu --gres=gpu:1 \
  --cpus-per-task=8 --mem=32G --time=06:00:00 \
  --output=outputs/slurm/chunk_0_%j.out --error=outputs/slurm/chunk_0_%j.err \
  --wrap='module load python/3.10.4-gcccore-11.3.0 && source .venv/bin/activate && \
  python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 0 --end-frame 7500 --chunk-id 0 \
  video_path=data/raw/original_120s.avi contacts.enabled=true'

sbatch --job-name=chunk1 --partition=gpu_cuda --qos=gpu --gres=gpu:1 \
  --cpus-per-task=8 --mem=32G --time=06:00:00 \
  --output=outputs/slurm/chunk_1_%j.out --error=outputs/slurm/chunk_1_%j.err \
  --wrap='module load python/3.10.4-gcccore-11.3.0 && source .venv/bin/activate && \
  python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 7500 --end-frame 15000 --chunk-id 1 \
  video_path=data/raw/original_120s.avi contacts.enabled=true'

# Repeat for chunks 2, 3, etc. adjusting --start-frame, --end-frame, --chunk-id

# Monitor
squeue -u $USER
tail -f outputs/slurm/chunk_0_*.out

# When all done, merge
ls -d outputs/runs/*chunk*
python scripts/merge_chunks.py outputs/runs/*_reference_chunk*/
```

---

## `run_parallel.sh` Reference

### Arguments

| Arg | Description | Default |
|-----|-------------|---------|
| `$1` | Video path | (required) |
| `$2` | Number of chunks/GPUs | `4` |
| `$3` | Config file | `configs/hpc_reference.yaml` |
| `$4` | Extra overrides (space-separated) | none |
| `$5` | Pipeline name | `reference` |

### Examples

```bash
# Default: 4 chunks, reference pipeline
bash scripts/run_parallel.sh data/raw/original_120s.avi

# 2 chunks (if only 2 GPUs available)
bash scripts/run_parallel.sh data/raw/original_120s.avi 2

# Custom config
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_reference.yaml

# With extra overrides
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_reference.yaml "detection.confidence=0.3"

# Centroid pipeline
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_centroid.yaml "" centroid

# Centroid with contacts
bash scripts/run_parallel.sh data/raw/original_120s.avi 2 configs/hpc_centroid.yaml "contacts.enabled=true" centroid
```

### What It Does (step by step)

1. **Creates batch directory** with timestamp: `outputs/runs/2026-02-27_141500_reference_batch/`
2. **Reads video metadata** (total frames, FPS, resolution) using OpenCV
3. **Divides frames** into N equal chunks (e.g., 30000 / 4 = 7500 per chunk)
4. **Launches N processes** in parallel, each with `CUDA_VISIBLE_DEVICES=<gpu_id>`
5. **Waits** for all chunks, reports timing per chunk
6. **Finds run directories** by parsing logs (no collisions with previous runs)
7. **Merges results** via `scripts/merge_chunks.py` (video + contacts + logs)
8. **Compresses contacts** into `contacts_<batch_id>.tar.gz`
9. **Prints download commands** (scp) for the merged video and contacts

### Sample Output

```
Analyzing video: data/raw/original_120s.avi
  Resolution: 640x480
  FPS: 25.0
  Total frames: 30000
  Duration: 1200.0s (20.0 min)

=== Parallel Execution Plan ===
  Batch ID:   2026-02-27_141500_reference_batch
  Batch dir:  outputs/runs/2026-02-27_141500_reference_batch
  Config:     configs/hpc_reference.yaml
  Video:      data/raw/original_120s.avi
  Frames:     30000
  Chunks:     4
  Per chunk:  ~7500 frames

  Chunk 0: frames 0 -> 7500 (GPU 0)
  Chunk 1: frames 7500 -> 15000 (GPU 1)
  Chunk 2: frames 15000 -> 22500 (GPU 2)
  Chunk 3: frames 22500 -> 30000 (GPU 3)

All 4 chunks launched. Waiting...
  Monitor: tail -f outputs/runs/.../logs/chunk_0.log

  Chunk 0: DONE (12m 34s)
  Chunk 1: DONE (11m 58s)
  Chunk 2: DONE (12m 15s)
  Chunk 3: DONE (11m 42s)

  Total time: 12m 34s

=== All chunks completed. Merging... ===

============================================================
  BATCH COMPLETE: 2026-02-27_141500_reference_batch
  TOTAL TIME:     12m 34s
  BATCH DIR:      /home/s4948012/Balbi/.../2026-02-27_141500_reference_batch
  MERGED VIDEO:   /home/s4948012/Balbi/.../reference_sam2.1_hiera_large_2026-02-27_merged.avi
  CONTACTS:       /home/s4948012/Balbi/.../contacts_2026-02-27_141500_reference_batch.tar.gz
============================================================

# Download results (PowerShell):
scp s4948012@bunya.rcc.uq.edu.au:/home/.../reference_sam2.1_hiera_large_2026-02-27_merged.avi \
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
scp s4948012@bunya.rcc.uq.edu.au:/home/.../contacts_2026-02-27_141500_reference_batch.tar.gz \
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
```

---

## Output Structure

```
outputs/runs/2026-02-27_141500_reference_batch/
├── chunks/
│   ├── *_reference_chunk0/
│   │   ├── config_used.yaml
│   │   ├── overlays/
│   │   │   └── reference_sam2.1_hiera_large_2026-02-27.avi
│   │   ├── contacts/
│   │   │   ├── contacts_per_frame.csv
│   │   │   ├── contact_bouts.csv
│   │   │   └── session_summary.json
│   │   └── logs/
│   │       └── run.log
│   ├── *_reference_chunk1/
│   ├── *_reference_chunk2/
│   └── *_reference_chunk3/
├── overlays/
│   └── reference_sam2.1_hiera_large_2026-02-27_merged.avi   <- FINAL VIDEO
├── contacts/
│   ├── contacts_per_frame.csv          <- all frames combined (raw)
│   ├── contact_bouts.csv               <- all bouts combined (raw)
│   ├── session_summary.json            <- aggregated totals (raw)
│   ├── contacts_real_per_frame.csv     <- post-processed per-frame labels
│   ├── contacts_real_events.csv        <- clean events with MM:SS timestamps
│   ├── session_summary_real.json       <- post-processing metadata + comparison
│   ├── event_log.txt                   <- human-readable event log for validation
│   └── reports/
│       ├── timeline_comparison.png     <- raw vs real ethogram timeline
│       ├── duration_by_type.png        <- total seconds per contact type
│       ├── events_by_type.png          <- event count per contact type
│       ├── event_duration_distribution.png <- duration histograms
│       ├── comparison_by_type.csv      <- per-type statistics
│       └── comparison_global.csv       <- global statistics
├── contacts_2026-02-27_141500_reference_batch.tar.gz        <- compressed (includes all above)
├── config_used.yaml
└── logs/
    ├── chunk_0.log
    ├── chunk_1.log
    ├── chunk_2.log
    ├── chunk_3.log
    └── run.log                     <- concatenated logs from all chunks
```

---

## First-Time Setup

Only needed once when setting up the project on Bunya.

### Python + venv

```bash
cd ~/Balbi/yolo-sam2-lab-tracking
module load python/3.10.4-gcccore-11.3.0
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### PyTorch (CUDA 11.8)

```bash
export TMPDIR=$HOME/tmp && mkdir -p $TMPDIR
python -m pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

# Validate
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### SAM2 (required for reference, sam2_yolo, sam2_video)

```bash
# Clone SAM2 into models directory
git clone https://github.com/facebookresearch/segment-anything-2.git models/sam2/segment-anything-2

# Install SAM2 package
python -m pip install ./models/sam2/segment-anything-2

# Install project dependencies (filter out SAM2 editable line)
grep -v "^-e ./models/sam2/segment-anything-2" requirements.txt > requirements.fixed.txt
python -m pip install -r requirements.fixed.txt
pip install ultralytics opencv-python roboflow

# Download SAM2 checkpoints (tiny + large)
cd models/sam2/segment-anything-2/checkpoints && bash download_ckpts.sh
cd ~/Balbi/yolo-sam2-lab-tracking

# Verify
python -c "from sam2.build_sam import build_sam2; print('SAM2 OK')"
```

### SAM3 (optional — for sam3 pipeline)

SAM3 requires **Python 3.12** and **PyTorch 2.7+** (CUDA 12.6).
If Bunya's Python module is 3.10, you need a separate venv.

```bash
# Option A: Same venv (if Bunya has Python 3.12 module)
module load python/3.12.x-gcccore-xx.x.x  # check: module avail python
git clone https://github.com/facebookresearch/sam3.git models/sam3/sam3
python -m pip install ./models/sam3/sam3

# Option B: Use conda for Python 3.12 (if no system module)
module load anaconda3
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/facebookresearch/sam3.git models/sam3/sam3
pip install -e ./models/sam3/sam3
grep -v "^-e ./models/sam2/segment-anything-2" requirements.txt > requirements.fixed.txt
pip install -r requirements.fixed.txt

# Download SAM3 checkpoint (~3.2GB)
export TMPDIR=$HOME/tmp && mkdir -p $TMPDIR
pip install huggingface_hub
huggingface-cli login   # paste HF token
huggingface-cli download facebook/sam3 sam3.pt --local-dir models/sam3/

# Verify
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"
```

### Running SAM3 on Bunya

```bash
# Single GPU test (2 min clip)
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --time=02:00:00 --mem=64G
srun --pty bash
cd ~/Balbi/yolo-sam2-lab-tracking
# activate the SAM3 environment (conda or venv)
python -m src.pipelines.sam3.run --config configs/hpc_sam3.yaml \
    video_path=data/raw/original_120s.avi scan.max_frames=300
```

---

## Transfer Files

### Upload video to Bunya (PowerShell)

```powershell
scp C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\data\raw\original.mp4 `
  s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/data/raw/
```

### Check available runs (Bunya)

```bash
ls -lt ~/Balbi/yolo-sam2-lab-tracking/outputs/runs/
```

### Download overlay video (PowerShell)

```powershell
# The scp command is printed automatically by run_parallel.sh
# Example:
scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/outputs/runs/2026-02-27_141500_reference_batch/overlays/reference_sam2.1_hiera_large_2026-02-27_merged.avi `
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
```

### Download contacts (PowerShell)

```powershell
scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/outputs/runs/2026-02-27_141500_reference_batch/contacts_2026-02-27_141500_reference_batch.tar.gz `
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
```

### Download entire batch (PowerShell)

```powershell
scp -r s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/outputs/runs/2026-02-27_141500_reference_batch `
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
```

---

## Monitoring

### During `run_parallel.sh`

If nothing appears after launching, the script is analyzing the video (reading frame count with OpenCV). Wait a moment.

```bash
# Check if Python processes are running
ps aux | grep python | grep reference

# Find latest batch directory
LAST=$(ls -dt outputs/runs/*_batch/ 2>/dev/null | head -1)
echo "Latest batch: $LAST"

# Check if chunk logs exist yet
ls -la $LAST/logs/

# Watch a chunk log in real-time
tail -f $LAST/logs/chunk_0.log

# Quick status of all chunks
for f in $LAST/logs/chunk_*.log; do echo "=== $(basename $f) ==="; tail -3 "$f"; done

# Check GPU utilization
nvidia-smi
```

### During `sbatch` jobs

```bash
# Check job status
squeue -u $USER

# View chunk output
cat outputs/slurm/chunk_0_*.out

# Follow a log in real-time
tail -f outputs/slurm/chunk_0_*.out

# Check disk usage
du -sh outputs/runs/
```

---

## GPU vs CPU — Choosing `models.device`

The `models.device` parameter in the YAML config controls where models run:

| Value | Behavior |
|-------|----------|
| `"auto"` | Uses GPU if CUDA is available, falls back to CPU |
| `"cuda"` | Forces GPU (fails if no CUDA) |
| `"cpu"` | Forces CPU (always works, but SAM2 is very slow) |

### If you got CPU on Bunya

If you see `Using device: cpu`, check:
1. You're on a **GPU node** (not the login node) — request one with `salloc`
2. CUDA module is loaded — run `module load cuda` before activating venv
3. PyTorch has CUDA — reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu118`

Quick diagnostic:
```bash
nvidia-smi                                                    # Should show GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"  # Should say True
```

## CLI Overrides

Any YAML parameter can be overridden with `key.subkey=value`:

```bash
# GPU + tiny model + only 300 frames
python -m src.pipelines.reference.run --config configs/local_reference.yaml \
    models.device=cuda scan.max_frames=300

# Change YOLO confidence
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
    detection.confidence=0.4

# Use a different video
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
    video_path=data/raw/another_video.mp4

# Enable contacts
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
    contacts.enabled=true
```

### Common overrides

| Override | Effect |
|----------|--------|
| `models.device=cuda` | Force GPU |
| `models.device=cpu` | Force CPU |
| `scan.max_frames=300` | Process only 300 frames |
| `scan.max_frames=null` | Process entire video |
| `detection.confidence=0.4` | Raise YOLO detection threshold |
| `detection.keypoint_min_conf=0.5` | Only high-confidence keypoints |
| `video_path=data/raw/video.mp4` | Use a different input video |
| `contacts.enabled=true` | Enable contact classification |

---

## Other Run Options

```bash
# 5 min clip test
python scripts/trim_video.py data/raw/original.mp4 --duration 300
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  video_path=data/raw/original_300s.avi contacts.enabled=true

# Full video (no trimming, single GPU)
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  video_path=data/raw/original.mp4 contacts.enabled=true

# sam2_yolo pipeline (uses BoT-SORT tracking instead of IdentityMatcher)
python -m src.pipelines.sam2_yolo.run --config configs/hpc_full.yaml \
  video_path=data/raw/original_120s.avi contacts.enabled=true

# Contacts only analysis (fast, no SAM2)
python scripts/analyze_contacts.py --config configs/hpc_full.yaml contacts.enabled=true
```

---

## Slurm Admin Commands

```bash
# See available partitions and nodes
sinfo -p gpu_cuda

# See GPUs across cluster nodes
sinfo -p gpu_cuda -N -l

# See your running/pending jobs
squeue -u $USER

# Cancel a specific job
scancel <JOB_ID>

# Interactive shell on allocated node
srun --pty bash

# Check GPU status
nvidia-smi

# List GPU devices
nvidia-smi -L

# Count available GPUs (not MIG sub-devices)
nvidia-smi -L | grep -c "^GPU"
```

---

## Recovery (Finding Last Batch)

If a process was accidentally stopped, or you need to find previous results:

```bash
# Find the latest batch directory
LAST=$(ls -dt outputs/runs/*_batch/ 2>/dev/null | head -1)
echo "Batch: $LAST"

# Check if merged outputs exist
ls "$LAST/overlays/" 2>/dev/null && ls "$LAST/contacts/" 2>/dev/null || echo "No merged yet"

# Check chunk logs for errors
for f in $LAST/logs/chunk_*.log; do echo "=== $(basename $f) ==="; tail -3 "$f"; done

# Verify video integrity
source .venv/bin/activate
python -c "import cv2; c=cv2.VideoCapture('$LAST/overlays/reference_sam2.1_hiera_large_2026-02-27_merged.avi'); print(f'{int(c.get(cv2.CAP_PROP_FRAME_COUNT))} frames, {c.get(cv2.CAP_PROP_FRAME_COUNT)/c.get(cv2.CAP_PROP_FPS):.1f}s')"

# Manually compress contacts if needed
tar -czf "$LAST/contacts.tar.gz" -C "$LAST" contacts/
ls -lh "$LAST/contacts.tar.gz"

# Check if Python processes are still running
ps aux | grep python | grep reference
```

---

## Disk Quota

```bash
# Check disk usage
du -sh $HOME
du -sh $HOME/.cache $HOME/.local

# Check pipeline outputs size
du -sh outputs/runs/
```

> GPU flags (`--gres`, `--time`, `--mem`) do not affect disk quota.

If quota is full — **nuclear reset** (last resort, destroys everything):

```bash
rm -rf ./* ./.??*
mkdir -p ~/Balbi && cd ~/Balbi
git clone https://github.com/catherinevaraspadilla-commits/yolo-sam2-lab-tracking.git
```

---

## Roboflow Extraction

Extract annotated frames from a video for Roboflow labeling:

```bash
bash roboflow/run_extract.sh data/raw/original.mp4 2
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `PD` in squeue | Waiting for GPU | Wait, or reduce `--gres=gpu:N` |
| `Exit 1` on a chunk | Python error | Check `logs/chunk_N.log` |
| `CUDA out of memory` | Too many processes per GPU | Use 1 chunk per GPU (default) |
| `git pull` conflicts | Local changes on Bunya | `git stash && git pull && git stash pop` |
| `salloc` hangs | No GPUs available | Try fewer GPUs or off-peak hours |
| Video not found in merge | Old naming | Fixed: merge finds any `*.avi`/`*.mp4` |
| `sbatch: error: Unable to open file` | Wrong directory | `cd ~/Balbi/yolo-sam2-lab-tracking` first |
| `AttributeError: NoneType` | Old code on Bunya | `git pull` to get latest fixes |
