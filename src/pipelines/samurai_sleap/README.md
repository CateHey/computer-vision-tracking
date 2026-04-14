# SAMURAI + SLEAP Pipeline

SAMURAI (SAM2 video predictor) for mask tracking + SLEAP for keypoint detection.

## Prerequisites

### Models on Bunya

Make sure these files exist on Bunya before running:

```
~/Balbi/yolo-sam2-lab-tracking/
  models/
    sam2/segment-anything-2/checkpoints/sam2.1_hiera_large.pt
    sleap/sleap.ckpt
```

Upload SLEAP checkpoint from local (PowerShell):

```bash
ssh s4948012@bunya.rcc.uq.edu.au "mkdir -p ~/Balbi/yolo-sam2-lab-tracking/models/sleap"
scp C:\Users\lucer\Downloads\mouse\yolo-sam2-lab-tracking\models\sleap\sleap.ckpt s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/models/sleap/
```

### Python dependencies

```bash
pip install opencv-python numpy torch torchvision sleap-nn sleap-io scipy tqdm
```

### Video

Place your 2-minute video at the path configured in the YAML (default: `data/raw/original.mp4`).

## Step 1: Get initial bounding boxes (local)

Run this locally on the 10s clip to select rat positions on frame 0:

```bash
python -m src.pipelines.samurai_sleap.get_init_bboxes --video data/clips/output-10s.mp4
```

Draw a box around each rat, press ENTER after each, ESC when done. It prints a YAML snippet like:

```yaml
init_bboxes:
  - [424, 701, 309, 147]   # Rat 1
  - [105, 709, 383, 179]   # Rat 2
```

These boxes are already set in `configs/hpc_samurai_sleap.yaml`. Since all videos start with rats in the same position, these work for any video from the same rig.

## Step 2: Run on Bunya (2-min video)

SSH into Bunya and run:

```bash
cd ~/Balbi/yolo-sam2-lab-tracking

python -m src.pipelines.samurai_sleap.run \
    --config configs/hpc_samurai_sleap.yaml \
    video_path=data/raw/original_120s.avi
```

### With a frame limit (test first 300 frames):

```bash
python -m src.pipelines.samurai_sleap.run \
    --config configs/hpc_samurai_sleap.yaml \
    video_path=data/raw/your_2min_video.avi \
    scan.max_frames=300
```

### With chunk processing (parallel):

```bash
python -m src.pipelines.samurai_sleap.run \
    --config configs/hpc_samurai_sleap.yaml \
    --start-frame 0 --end-frame 1800 --chunk-id 0

python -m src.pipelines.samurai_sleap.run \
    --config configs/hpc_samurai_sleap.yaml \
    --start-frame 1800 --end-frame 3600 --chunk-id 1
```

## Step 3: Run locally (optional)

For local testing with SAM2 tiny:

```bash
python -m src.pipelines.samurai_sleap.run \
    --config configs/local_samurai_sleap.yaml \
    --interactive
```

Or with the pre-set bboxes (add `init_bboxes` to `local_samurai_sleap.yaml` first):

```bash
python -m src.pipelines.samurai_sleap.run --config configs/local_samurai_sleap.yaml
```

## Output

Results go to `outputs/runs/<timestamp>_samurai_sleap/`:

```
outputs/runs/2026-03-24_183000_samurai_sleap/
  overlays/
    samurai_sleap_2026-03-24.avi    # Video with masks + keypoints
  logs/
    run.log                         # Execution log
  config_used.yaml                  # Config snapshot
```

## Pipeline flow

1. **Extract frames** — video frames saved as JPEGs (SAMURAI needs a frame directory)
2. **Init bboxes** — from config or interactive selection
3. **SAMURAI tracking** — SAM2 video predictor propagates masks across all frames
4. **SLEAP keypoints** — per-frame bottom-up pose detection (7 keypoints per rat)
5. **Assignment** — keypoints matched to masks by spatial overlap
6. **Render** — overlay video with colored masks, centroids, and keypoint labels

## Notes

- If `sleap-nn` is not installed, the pipeline still runs with **masks only** (no keypoints)
- The `init_bboxes` don't need to be pixel-perfect — SAMURAI refines the segmentation
- Extracted frames are cleaned up after processing (configurable via `output.cleanup_frames`)
- Contact classification is not yet implemented in this pipeline
