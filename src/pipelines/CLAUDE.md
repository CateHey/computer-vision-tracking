# src/pipelines/

Each subfolder is a self-contained pipeline. All share the same architecture:
YOLO detect → segmentation model → identity tracking → optional contacts → overlay video.

## Pipelines

| Pipeline | Segmentation | Tracking | Status |
|----------|-------------|----------|--------|
| `reference/` | SAM2 ImagePredictor | IdentityMatcher | **Recommended** |
| `sam3/` | SAM3 Sam3Processor | IdentityMatcher | Evaluation |
| `sam2_yolo/` | SAM2 ImagePredictor | SlotTracker (BoT-SORT) | Legacy |
| `sam2_video/` | SAM2 VideoPredictor | SAM2 internal | Special case |

## Shared Components

All pipelines reuse modules from `src/common/`. The `reference/` and `sam3/` pipelines
also share `identity_matcher.py` (defined in `reference/`, imported by `sam3/`).

## Adding a New Pipeline

1. Create `src/pipelines/<name>/` with `__init__.py`, `run.py`, `models_io.py`
2. Reuse `src/common/` modules (config, video I/O, visualization, contacts)
3. Add configs in `configs/local_<name>.yaml` and `configs/hpc_<name>.yaml`
4. Update `docs/architecture/pipelines.md` comparison table
