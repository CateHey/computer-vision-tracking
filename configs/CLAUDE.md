# configs/

YAML configuration files. Most pipelines have a local variant (short clip, small models, auto device)
and an HPC variant (full video, large models, cuda).

## Files

| Config | Used by | Status |
|--------|---------|--------|
| `local_cutie_composite.yaml`, `hpc_cutie_composite.yaml` | `src.pipelines.cutie_composite.run` | **Production (paper pipeline)** |
| `local_sam3_composite.yaml`, `hpc_sam3_composite.yaml` | `src.pipelines.sam3_composite.run` (also the SAM3 phase of `cutie_composite`) | Building block |
| `local_sam3_reset.yaml`, `hpc_sam3_reset.yaml` | `src.pipelines.sam3_reset.run` | Exploratory |
| `local_centroid.yaml`, `hpc_centroid.yaml` | `src.pipelines.centroid.run` | Older production pipeline (chunked runs) |
| `local_isolated_composite.yaml`, `hpc_isolated_composite.yaml` | `src.pipelines.isolated_composite.run` | Exploratory |
| `hpc_samurai.yaml` | `src.pipelines.samurai.run` | Exploratory |
| `local_samurai_sleap.yaml`, `hpc_samurai_sleap.yaml` | `src.pipelines.samurai_sleap.run` | Exploratory |
| `local_extract.yaml`, `hpc_extract.yaml` | `roboflow/extract_interaction_frames.py`, `roboflow/run_extract.sh`; the `roboflow:` block is read by `scripts/upload_to_roboflow.py` | Labelling |
| `contacts_postprocess_simple.yaml` | `scripts/postprocess_contacts_simple.py` | Post-processing |
| `validation.yaml` | `scripts/validate_contacts.py` | Validation |

The configs of the removed pipelines (reference, sam3, sam2_yolo, sam2_video) and the BoT-SORT
tracker config were deleted in October 2026; git history keeps them.

## Conventions

- All thresholds are configurable: no hardcoded values in code
- CLI overrides: `key.subkey=value` (e.g., `detection.confidence=0.4`)
- Path keys (`video_path`, `yolo_path`, `sam2_checkpoint`, `sam3_checkpoint`, ...) resolve relative to the project root
- `detection.keypoint_names` must list the 7 names in model index order:
  `tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear` (see `CLAUDE.md`)
- Every run writes the resolved config to `outputs/runs/<run>/config_used.yaml`
