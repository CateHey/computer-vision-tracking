# Archive — earlier approaches and exploratory work

Material kept for the research record. **None of it is part of the production pipeline**
(`src/pipelines/cutie_composite/`); it documents what was tried on the way there. Why the main
tracking approaches were or were not selected is in
[`../final_documentation/04_model_comparison.md`](../final_documentation/04_model_comparison.md), and
the chronological test log is
[`../docs/changes/observation_log.md`](../docs/changes/observation_log.md).

## Contents

| Path | What it is |
|------|------------|
| `scripts/` | one-off debug and inspection scripts (SAM2- or YOLO-only runs, contact analysis, frame/clip extraction, GPU check) |
| `scripts/research/` | small experiments (e.g. YOLO on cropped keypoints) |
| `configs/deprecated/` | configs for pipelines no longer used (reference, sam2-video, sam3, full, quick) |
| `configs/trackers/` | BoT-SORT tracker config from the appearance-tracking experiments |
| `slurm/` | old sbatch jobs (frame extraction, inference, sam2-video, YOLO training) |
| `common/` | `src/common` modules written during exploration and not imported by any current pipeline (`cost.py`, `mask_dedup.py`, `tracking.py`) |

These files are not maintained and may not run against the current code. They are kept so the
evolution of the project stays visible.
