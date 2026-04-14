# Output Structure

All pipeline runs write to `outputs/runs/`. Each run creates a timestamped
directory. Old/manual outputs in `outputs/olds/` are git-ignored.

---

## Single Run (any pipeline)

Every pipeline produces the same base structure. The `<TAG>` and video filename
change depending on the pipeline used.

```
outputs/runs/<YYYY-MM-DD_HHMMSS>_<TAG>/
├── config_used.yaml                          # exact config for this run
├── logs/
│   └── run.log
├── overlays/
│   └── <pipeline>_<model>_<YYYY-MM-DD>.avi   # overlay video
└── contacts/                                 # only if contacts.enabled=true
    ├── contacts_per_frame.csv
    ├── contact_bouts.csv
    ├── session_summary.json
    └── report.pdf
```

### Pipeline tags and video names

| Pipeline      | `<TAG>`         | Video filename example                              |
|---------------|-----------------|-----------------------------------------------------|
| Reference     | `reference`     | `reference_sam2.1_hiera_large_2026-02-27.avi`       |
| SAM2 + YOLO   | `sam2_yolo`     | `sam2_yolo_sam2.1_hiera_large_2026-02-27.avi`       |
| SAM2 Video    | `sam2_video`    | `sam2_video_sam2.1_hiera_large_2026-02-27.avi`      |

- Video format depends on `output.video_codec`: `XVID` produces `.avi`, anything
  else produces `.mp4`.
- Model name is taken from the SAM2 checkpoint filename stem.

### Chunk runs (parallel processing)

When `--chunk-id` is passed, the tag becomes `<pipeline>_chunk<ID>`:

```
outputs/runs/<YYYY-MM-DD_HHMMSS>_reference_chunk0/
outputs/runs/<YYYY-MM-DD_HHMMSS>_reference_chunk1/
...
```

Each chunk directory has the same internal structure as a single run.

---

## Parallel Batch (`scripts/run_parallel.sh`)

The parallel script groups everything under a single batch directory:

```
outputs/runs/<YYYY-MM-DD_HHMMSS>_reference_batch/
│
├── chunks/
│   ├── <TIMESTAMP>_reference_chunk0/
│   │   ├── config_used.yaml
│   │   ├── logs/run.log
│   │   ├── overlays/
│   │   │   └── reference_<model>_<date>.avi
│   │   └── contacts/
│   │       ├── contacts_per_frame.csv
│   │       ├── contact_bouts.csv
│   │       ├── session_summary.json
│   │       └── report.pdf
│   ├── <TIMESTAMP>_reference_chunk1/
│   ├── <TIMESTAMP>_reference_chunk2/
│   └── <TIMESTAMP>_reference_chunk3/
│
├── overlays/
│   └── <pipeline>_<model>_<date>_merged.avi    # final concatenated video
│
├── contacts/
│   ├── contacts_per_frame.csv                  # merged from all chunks
│   ├── contact_bouts.csv                       # merged from all chunks
│   └── session_summary.json                    # aggregated statistics
│
├── logs/
│   ├── chunk_0.log                             # stdout+stderr per chunk
│   ├── chunk_1.log
│   ├── chunk_2.log
│   ├── chunk_3.log
│   └── run.log                                 # concatenated chunk logs
│
└── contacts_<BATCH_ID>.tar.gz                  # compressed contacts for download
```

### How the script works

1. Reads total frames from the video with OpenCV.
2. Divides frames evenly across `NUM_CHUNKS` GPUs.
3. Launches each chunk with `CUDA_VISIBLE_DEVICES=$i` in background.
4. Waits for all chunks; reports per-chunk and total timing.
5. Parses `"Run directory: ..."` from each chunk log to find the exact paths.
6. Calls `scripts/merge_chunks.py` to merge videos, CSVs, and logs.
7. Compresses `contacts/` into a `.tar.gz`.
8. Prints `scp` download commands.

---

## Merged Output (`scripts/merge_chunks.py`)

When called standalone (outside `run_parallel.sh`):

```
python scripts/merge_chunks.py <chunk_dir_1> <chunk_dir_2> ... -o <output_dir>
```

```
<output_dir>/
├── config_used.yaml                          # copied from first chunk
├── overlays/
│   └── <base_name>_merged.avi                # concatenated video
├── contacts/
│   ├── contacts_per_frame.csv                # rows appended sequentially
│   ├── contact_bouts.csv                     # rows appended sequentially
│   └── session_summary.json                  # aggregated metadata
└── logs/
    └── run.log                               # concatenated with chunk separators
```

- Video merging uses `ffmpeg concat` (stream copy, no re-encode) when available;
  falls back to OpenCV re-encoding.
- CSV merging keeps the header from the first chunk and appends data rows from
  all chunks in order.

---

## SLURM Batch (`slurm/run_chunks.sbatch`)

When using `sbatch --array=0-3`:

```
outputs/slurm/
├── chunk_0_<JOBID>.out                       # stdout per array task
├── chunk_0_<JOBID>.err                       # stderr per array task
├── chunk_1_<JOBID>.out
├── chunk_1_<JOBID>.err
├── chunk_2_<JOBID>.out
├── chunk_2_<JOBID>.err
├── chunk_3_<JOBID>.out
└── chunk_3_<JOBID>.err
```

The actual run directories are created inside `outputs/runs/` (same structure as
chunk runs above). After all array tasks finish, merge manually:

```bash
python scripts/merge_chunks.py outputs/runs/*_reference_chunk* -o outputs/runs/merged
```

---

## Contact Files Detail

All three pipelines produce the same contact output format when
`contacts.enabled=true`.

### `contacts_per_frame.csv`

One row per frame per rat pair. Key columns:

| Column                 | Description                                      |
|------------------------|--------------------------------------------------|
| `frame_idx`            | 0-based frame number                             |
| `time_sec`             | Timestamp in seconds                             |
| `rat_a_slot`           | Identity slot (0 or 1)                           |
| `rat_b_slot`           | Identity slot (0 or 1)                           |
| `zone`                 | `contact` / `proximity` / `independent`          |
| `contact_type`         | `N2N` / `N2AG` / `N2B` / `SBS` / `FOL` / null   |
| `investigator_slot`    | Which rat initiates (for asymmetric types)       |
| `nose_nose_dist_px`    | Nose-to-nose distance in pixels                  |
| `nose_nose_dist_bl`    | Same, in body lengths                            |
| `centroid_dist_px`     | Centroid distance in pixels                       |
| `mask_iou`             | Intersection-over-union of masks                 |
| `body_length_a_px`     | Body length of rat A in pixels                   |
| `velocity_a_px`        | Velocity of rat A in pixels/frame                |
| `orientation_cos`      | Cosine of angle between body vectors             |
| `quality_flag`         | `high_mask_overlap` / `missing_keypoints` / null |
| `bout_id`              | Assigned during finalization                     |

### `contact_bouts.csv`

Temporal groupings of consecutive same-type contacts:

| Column              | Description                                |
|---------------------|--------------------------------------------|
| `bout_id`           | Unique bout identifier                     |
| `contact_type`      | Type of contact                            |
| `start_frame`       | First frame of bout                        |
| `end_frame`         | Last frame of bout                         |
| `duration_sec`      | Duration in seconds                        |
| `duration_frames`   | Duration in frames                         |
| `mean_nose_dist_px` | Average nose distance during bout          |
| `mean_mask_iou`     | Average mask overlap during bout           |

### `session_summary.json`

Aggregated session statistics:

```json
{
  "metadata": {
    "video_path": "data/raw/original_120s.avi",
    "total_frames": 3600,
    "fps": 30.0,
    "duration_sec": 120.0
  },
  "zone_summary": {
    "contact_frames": 450,
    "proximity_frames": 800,
    "independent_frames": 2350
  },
  "contact_type_summary": {
    "N2N":  { "total_bouts": 12, "total_duration_sec": 8.5 },
    "N2AG": { "total_bouts": 25, "total_duration_sec": 18.3 },
    "SBS":  { "total_bouts": 5,  "total_duration_sec": 12.0 }
  }
}
```

### `report.pdf`

Visual contact analysis with 5 pages:

1. Timeline ethogram (colored bars by contact type)
2. Duration histograms per contact type
3. Pie chart of contact time distribution
4. Time-binned contact rate (1-minute bins)
5. Per-pair summary table

---

## Contact Types

| Code   | Name              | Symmetric | Description                                   |
|--------|-------------------|-----------|-----------------------------------------------|
| `N2N`  | Nose-to-Nose      | Yes       | Both noses within contact zone                |
| `N2AG` | Nose-to-Anogenital| No        | One nose near other's tail base               |
| `N2B`  | Nose-to-Body      | No        | One nose touching other's body mask           |
| `SBS`  | Side-by-Side      | Yes       | High mask overlap, low velocity, parallel     |
| `FOL`  | Following         | No        | One nose near other's tail, both moving, aligned |

For asymmetric types, `investigator_slot` indicates which rat initiates.

---

## `outputs/olds/`

Legacy manual outputs. **Git-ignored** — not tracked in the repository.

```
outputs/olds/
├── overlay.avi
├── overlay 2.avi
├── overlay 6s.avi
├── overlay10sgood.avi
├── overlay_26-2.avi
└── sam2_video_26-2.avi
```
