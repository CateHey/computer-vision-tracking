# src/

All Python source code. Organized into `common/` (shared utilities) and `pipelines/` (pipeline implementations).

## Conventions

- Imports use `src.` prefix: `from src.common.metrics import mask_iou`
- Run pipelines as modules: `python -m src.pipelines.reference.run`
- No hardcoded thresholds — all parameters come from YAML configs
- Shared code goes in `src/common/`, pipeline-specific code in `src/pipelines/<name>/`

## Structure

- `common/` — Shared utilities (config, video I/O, metrics, visualization, contacts, geometry, cost, tracking)
- `pipelines/reference/` — Recommended pipeline (SAM2 + IdentityMatcher)
- `pipelines/sam3/` — SAM3 evaluation pipeline (mirrors reference)
- `pipelines/sam2_yolo/` — Original pipeline (BoT-SORT tracking)
- `pipelines/sam2_video/` — Video predictor pipeline (temporal memory)
