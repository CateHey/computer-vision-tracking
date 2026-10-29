# src/common/

Shared utilities used by all pipelines. When adding functionality that several pipelines need, put it here.

## Files

| File | Purpose |
|------|---------|
| `config_loader.py` | YAML config loading, CLI overrides, run directory setup, logging, `get_device()` |
| `io_video.py` | Video read/write, frame iteration, video properties |
| `metrics.py` | `mask_iou()`, `compute_centroid()`, `evaluate_closeness()` |
| `visualization.py` | Mask overlays, keypoint rendering, status text |
| `contacts_v2.py` | `ContactTrackerV2`: continuous scores + Schmitt triggers, 5 types (N2N, N2AG, N2B, SBS, FOL) + NC, families, dynamics. **Used by `cutie_composite`** |
| `individual_metrics.py` | Per-animal and directed-pair metrics computed from v2 bouts (who initiates, active vs passive); extra PDF pages |
| `contacts_v21.py` | Variant of v2 (hysteresis rule engine, multi-scale temporal windows). Not imported by any pipeline |
| `contacts.py` | `ContactTracker` v1 (centroid pipeline): 6 contact types + NC, bout detection, CSV/JSON/PDF |
| `excel_report.py` | `build_excel_report()`: consolidates run outputs into `results.xlsx` (reported tables) and `diagnostics.xlsx` (tracker diagnostics) |
| `geometry.py` | `euclidean_distance()`, `resolve_overlaps()`, detection-to-slot matching |
| `model_loaders.py` | `load_yolo()`, `load_models()` |
| `yolo_inference.py` | `detect_only()` and YOLO result parsing (keypoints by name) |
| `provenance.py` | `write_provenance()`: `<run>/provenance.json` (git commit, package versions, GPU, MD5 of YOLO weights and config); called by `cutie_composite` |
| `utils.py` | `Detection`, `Keypoint`, `FrameAnalysis` dataclasses |
| `constants.py` | `DEFAULT_KEYPOINT_NAMES` and other shared constants |

Removed in October 2026 (git history keeps them): `cost.py`, `mask_dedup.py`, `tracking.py`
(assignment cost, mask dedup and the BoT-SORT `SlotTracker` of the deleted sam2_yolo / reference pipelines).

## Key Rules

- `_PATH_KEYS` in `config_loader.py` lists the config keys that resolve as file paths
- Keypoints are addressed by name (`constants.DEFAULT_KEYPOINT_NAMES`), never by index;
  the rear/anogenital point is `tail_start`
- SBS in v2 uses `mask_contact_bl` (shared border length): CUTIE masks are disjoint, so mask IoU is always 0
