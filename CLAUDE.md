# Rat Tracking Vision Pipeline

## Project

YOLO + SAM2 pipeline for laboratory rat tracking and social contact classification.
Runs locally (short clips) and on UQ Bunya HPC (full videos, multi-GPU parallel).

## Production Pipeline

**Centroid pipeline** (`src/pipelines/centroid/`) — production default.
- SAM2 centroid propagation drives masks and identity (YOLO only on init frame)
- SAM2 uses centroid prompts ALWAYS (positive + negative points, no YOLO box prompts)
- YOLO demoted to keypoint-only provider on full image
- Keypoints assigned to masks by spatial overlap (not YOLO box ordering)
- Temporal carry-over fills missing keypoints using centroid delta
- No IdentityMatcher — identity inherent from SAM2 propagation

**CRITICAL: Do NOT reintroduce YOLO box prompts for SAM2.**
YOLO detection order is arbitrary and causes identity swaps when used as SAM2 prompts.
SAM2 centroid-only prompting was validated swap-free. See `docs/centroid_pipeline.md`
"Lessons Learned" section for full analysis.

All other pipelines (reference, sam3, sam2_yolo, sam2_video) are in `src/pipelines/deprecated/`.

7 keypoints: tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear

## Key Paths

| What | Path |
|------|------|
| Centroid pipeline | `src/pipelines/centroid/` |
| Contact tracker | `src/common/contacts.py` |
| Shared geometry | `src/common/geometry.py` |
| Shared metrics | `src/common/metrics.py` |
| YOLO inference | `src/common/yolo_inference.py` |
| Model loaders | `src/common/model_loaders.py` |
| YOLO weights | `models/yolo/best.pt` |
| Centroid configs | `configs/local_centroid.yaml`, `configs/hpc_centroid.yaml` |
| Parallel runner | `scripts/run_parallel.sh` |
| Chunk merger | `scripts/merge_chunks.py` |
| Deprecated pipelines | `src/pipelines/deprecated/` |
| Deprecated configs | `configs/deprecated/` |

## Running

```bash
# Local (10s clip)
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml

# Local with contacts
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml contacts.enabled=true

# HPC (multi-GPU parallel)
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_centroid.yaml "" centroid
```

## Documentation

See `docs/README.md` for the full index. Key docs:

| Folder | Content |
|--------|---------|
| `docs/setup/` | Local and HPC setup |
| `docs/architecture/` | System design, pipeline comparison |
| `docs/guides/` | Parameter tuning, labeling, evaluation |
| `docs/models/` | YOLO migration plan |
| `docs/contacts/` | Social contact design and output format |
| `docs/data/` | Data notes and output structure |
| `docs/changes/` | Observation log — pipeline improvements tracked by date |
| `docs/archive/` | Historical documents |

## Current Status

- Centroid pipeline is production default (2026-03-06)
- YOLO weights: `models/yolo/best.pt` (Roboflow-trained)
- ContactTracker with 6 contact types + NC operational
- 25 bugs/improvements fixed in contact system audit
- Parallel execution on Bunya via `scripts/run_parallel.sh`
- Reference, SAM3, sam2_yolo, sam2_video pipelines deprecated

## Conventions

- All configs in `configs/` — no hardcoded thresholds in code
- CLI overrides: `key.subkey=value` (e.g., `detection.confidence=0.4`)
- Output to `outputs/runs/<timestamp>_<tag>/`
- Logs go to `outputs/runs/<run>/logs/run.log`
