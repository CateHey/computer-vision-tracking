# scripts/

Utility scripts for execution, merging, data extraction, and upload.

## Files

| Script | Purpose |
|--------|---------|
| `run_parallel.sh` | Multi-GPU parallel execution — splits video into chunks, runs on separate GPUs, merges |
| `run_pipeline.sbatch` | Slurm batch wrapper around `run_parallel.sh` (centroid pipeline only; the production job is `slurm/run_cutie.sbatch`) |
| `bunya_env.sh` | Bunya settings: user, account string (auto from `groups`), paths, `bunya_activate`, `bunya_gpu_shell` |
| `bunya_setup.sh` | One-time Bunya setup: scratch symlinks, conda env, PyTorch, SAM2, transformers 5 (SAM3), CUTIE, all weights |
| `merge_chunks.py` | Merges chunk outputs: concatenates overlay videos (ffmpeg stream copy), combines contact CSVs |
| `trim_video.py` | Alternative clip extraction |
| `upload_to_roboflow.py` | Uploads extracted frames to Roboflow for labeling |

Removed in October 2026 (git history keeps them): `debug_yolo_only.py`, `debug_sam2_only.py`, `debug_sam2_no_yolo.py`, `run_debug_parallel.sh`, `research/`, `check_gpu.py`, `export_results.py`, `extract_clip.py`, `extract_frames.py`, `analyze_contacts.py`.

## Key Notes

- `run_parallel.sh` auto-detects GPU count and divides frames evenly
- `merge_chunks.py` uses ffmpeg stream copy (no re-encoding) for fast video merge
- Contact CSVs are merged with frame offset adjustment per chunk
