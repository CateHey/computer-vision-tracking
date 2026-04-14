# Rat Tracking Vision Pipeline (YOLO26 + SAM2/SAM3)

AI-based workflow for analyzing laboratory rat movement from real-world lab videos. Combines YOLO26 pose detection with SAM2 or SAM3 segmentation for accurate body-part localization (7 keypoints), identity tracking, and social contact classification.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run reference pipeline on a short clip (recommended)
python -m src.pipelines.reference.run --config configs/local_reference.yaml

# Or run sam3 pipeline (requires SAM3 installed)
python -m src.pipelines.sam3.run --config configs/local_sam3.yaml

# Or run sam2_yolo pipeline
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml

# With contact classification
python -m src.pipelines.reference.run --config configs/local_reference.yaml contacts.enabled=true
```

## How It Works

1. **YOLO26** detects rat bounding boxes and 7 body-part keypoints (NMS-free architecture)
2. **SAM2 or SAM3** segments each detection into pixel-accurate masks
3. **IdentityMatcher** maintains consistent identity via Hungarian assignment + state machine
4. **ContactTracker** classifies social contacts (nose-to-nose, side-by-side, following, etc.)
5. **Outputs** — overlay videos, contact CSVs, bout reports, session summaries

## Pipelines

| Pipeline | Best for | Tracking | Speed |
|----------|----------|----------|-------|
| **reference** (recommended) | Most videos | IdentityMatcher + centroid fallback | ~25 FPS |
| sam3 | SAM3 evaluation | IdentityMatcher + centroid fallback | TBD |
| sam2_yolo | Simple scenes | SlotTracker + BoT-SORT | ~25 FPS |
| sam2_video | Heavy occlusion | SAM2 temporal memory | ~5-10 FPS |

See [Pipeline Comparison](docs/architecture/pipelines.md) for detailed differences.

## Project Structure

```
yolo-sam2-lab-tracking/
  configs/                           # YAML configs (6 files: local/hpc × 3 pipelines)
  docs/                              # Documentation (see docs/README.md for index)
  data/
    raw/                             # Full videos (gitignored)
    clips/                           # Short test clips
  models/
    yolo/                            # YOLO weights (best.pt)
    sam2/                            # SAM2 checkpoints
    sam3/                            # SAM3 checkpoint (sam3.pt)
  src/
    common/                          # Shared utilities
      config_loader.py               #   YAML loading, run setup, logging
      io_video.py                    #   Video read/write/iterate
      metrics.py                     #   mask_iou, compute_centroid
      visualization.py               #   Mask overlays, keypoints
      tracking.py                    #   SlotTracker (sam2_yolo pipeline)
      contacts.py                    #   ContactTracker + bout detection
      geometry.py                    #   Euclidean distance, detection-slot matching
      cost.py                        #   Assignment cost computation
      mask_dedup.py                  #   Mask deduplication by IoU
      model_loaders.py               #   YOLO model loading
      utils.py                       #   Detection, Keypoint dataclasses
    pipelines/
      reference/                     # Reference pipeline (recommended)
        run.py                       #   Entry point
        identity_matcher.py          #   Hungarian + state machine tracking
        sam2_processor.py            #   SAM2 prompts + centroid fallback
      sam2_yolo/                     # SAM2+YOLO pipeline
        run.py                       #   Entry point
        models_io.py                 #   Model loading
        infer_yolo.py                #   YOLO detection
        infer_sam2.py                #   SAM2 segmentation
        postprocess.py               #   Filtering + tracking
      sam2_video/                    # SAM2 Video pipeline
        run.py                       #   Entry point
      sam3/                          # SAM3 pipeline (evaluation)
        run.py                       #   Entry point
        models_io.py                 #   SAM3 model loading
        sam3_processor.py            #   SAM3 prompts + coord normalization
  scripts/
    run_parallel.sh                  # Multi-GPU parallel execution
    merge_chunks.py                  # Merge chunk outputs (video + contacts)
    extract_frames.py                # Close-contact frame extraction
    trim_video.py                    # Cut clips from raw video
    analyze_contacts.py              # Contact analysis utilities
  slurm/                             # HPC Slurm job scripts
  outputs/runs/                      # Structured run outputs
```

## Configuration

All parameters are in YAML configs — no hardcoded thresholds:

| Pipeline | Local | HPC |
|----------|-------|-----|
| reference | `configs/local_reference.yaml` | `configs/hpc_reference.yaml` |
| sam3 | `configs/local_sam3.yaml` | `configs/hpc_sam3.yaml` |
| sam2_yolo | `configs/local_quick.yaml` | `configs/hpc_full.yaml` |
| sam2_video | `configs/local_sam2video.yaml` | `configs/hpc_sam2video.yaml` |

Override any parameter via CLI:
```bash
python -m src.pipelines.reference.run --config configs/local_reference.yaml \
    detection.confidence=0.4 contacts.enabled=true
```

## Documentation

See [docs/README.md](docs/README.md) for the full documentation index.

Key docs:
- [Pipeline Comparison](docs/architecture/pipelines.md) — which pipeline to use
- [Risks & Limitations](docs/architecture/risks.md) — known failure modes and workarounds
- [Identity Matcher Design](docs/architecture/identity_matcher.md) — how tracking works
- [Local Setup](docs/setup/local.md) — SAM2/SAM3 installation
- [HPC Guide](docs/setup/hpc.md) — running on Bunya
- [Parameter Tuning](docs/guides/parameter_tuning.md) — config reference
- [YOLO26 Migration](docs/models/yolo26_migration.md) — model upgrade details
- [Contact Design](docs/contacts/design.md) — social contact classification

## Environments

| Environment | Config | Video | SAM2 Model | Device |
|-------------|--------|-------|-----------|--------|
| Local laptop | `local_reference.yaml` | 10s clips | tiny | auto |
| UQ Bunya HPC | `hpc_reference.yaml` | full video | large | cuda |

## HPC (Bunya) Quick Start

```bash
# SSH into Bunya, request GPUs, then:
bash scripts/run_parallel.sh data/raw/original_120s.avi
```

This splits the video across available GPUs, processes in parallel, merges results,
and prints download commands. See [HPC Guide](docs/setup/hpc.md) for details.
# computer-vision-tracking
