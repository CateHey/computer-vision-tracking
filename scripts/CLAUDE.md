# scripts/

Utility scripts for execution, merging, data extraction, and upload.

## Files

| Script | Purpose |
|--------|---------|
| `run_parallel.sh` | Multi-GPU parallel execution — splits video into chunks, runs on separate GPUs, merges |
| `merge_chunks.py` | Merges chunk outputs: concatenates overlay videos (ffmpeg stream copy), combines contact CSVs |
| `extract_frames.py` | Finds close-contact frames and exports them (scan → encounters → plan → export) |
| `extract_clip.py` | Cuts a clip from raw video (ffmpeg) |
| `trim_video.py` | Alternative clip extraction |
| `upload_to_roboflow.py` | Uploads extracted frames to Roboflow for labeling |
| `analyze_contacts.py` | Contact analysis utilities |

## Key Notes

- `run_parallel.sh` auto-detects GPU count and divides frames evenly
- `merge_chunks.py` uses ffmpeg stream copy (no re-encoding) for fast video merge
- Contact CSVs are merged with frame offset adjustment per chunk
