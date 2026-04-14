# configs/

YAML configuration files. Each pipeline has a local and HPC variant.

## Files

| Config | Pipeline | Environment |
|--------|----------|-------------|
| `local_reference.yaml` | reference | Local (tiny SAM2, auto device) |
| `hpc_reference.yaml` | reference | HPC (large SAM2, cuda) |
| `local_sam3.yaml` | sam3 | Local (SAM3, auto device) |
| `hpc_sam3.yaml` | sam3 | HPC (SAM3, cuda) |
| `local_quick.yaml` | sam2_yolo | Local |
| `hpc_full.yaml` | sam2_yolo | HPC |
| `local_sam2video.yaml` | sam2_video | Local |
| `hpc_sam2video.yaml` | sam2_video | HPC |

## Conventions

- All thresholds are configurable — no hardcoded values in code
- CLI overrides: `key.subkey=value` (e.g., `detection.confidence=0.4`)
- Path keys (`video_path`, `yolo_path`, `sam2_checkpoint`, `sam3_checkpoint`) resolve relative to project root
- SAM3 configs use `sam3_checkpoint` instead of `sam2_checkpoint` + `sam2_config`
