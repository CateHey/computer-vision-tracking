# src/common/

Shared utilities used by all pipelines. When adding functionality that multiple pipelines need, put it here.

## Files

| File | Purpose |
|------|---------|
| `config_loader.py` | YAML config loading, CLI overrides, run directory setup, logging |
| `io_video.py` | Video read/write, frame iteration |
| `metrics.py` | `mask_iou()`, `compute_centroid()` |
| `visualization.py` | Mask overlays, keypoint rendering, status bar |
| `contacts.py` | ContactTracker — 5 contact types, bout detection, CSV/JSON/PDF reports |
| `geometry.py` | `euclidean_distance()`, `match_dets_to_slots()` |
| `cost.py` | `compute_assignment_cost()` — 3-component soft cost for identity matching |
| `mask_dedup.py` | `deduplicate_masks()` — remove redundant SAM masks by IoU |
| `model_loaders.py` | `load_yolo()` — YOLO model loading |
| `tracking.py` | `SlotTracker` — used only by sam2_yolo pipeline |
| `utils.py` | `Detection`, `Keypoint` dataclasses |
| `constants.py` | Shared constants |

## Key Rules

- `_PATH_KEYS` in `config_loader.py` lists config keys that resolve as file paths (includes `sam3_checkpoint`)
- `match_dets_to_slots()` in `geometry.py` is used by both reference and sam3 pipelines
- `cost.py` supports area veto: if area ratio > 5.0x, cost is set to INF
