# src/pipelines/

Each subfolder is a self-contained pipeline with a `run.py` entry point:
`python -m src.pipelines.<name>.run --config configs/<env>_<name>.yaml [key=value ...]`.

## Pipelines

| Pipeline | Segmentation / identity | Pose | Status |
|----------|-------------------------|------|--------|
| `cutie_composite/` | SAM3 text-prompt bootstrap on the first 125 frames (through `sam3_composite`), then CUTIE propagates one labelled index mask (one id per pixel, N animals) | YOLO on per-animal composites (other animals erased), `ContactTrackerV2` | **Production, the paper pipeline.** One process, one GPU, no chunk mode |
| `sam3_composite/` | SAM3 via Hugging Face `transformers` (text prompt "mouse") + composites | YOLO on composites | Building block: `cutie_composite` imports its SAM3 phase. Standalone runs are exploratory |
| `isolated_composite/` | SAM2 video predictor + composites; `composition.py` (background plate, erase other animals) is imported by `cutie_composite` and `sam3_*` | YOLO on composites | Exploratory |
| `sam3_reset/` | `sam3_composite` plus failure detection and reset strategies | YOLO on composites | Exploratory, superseded by CUTIE |
| `centroid/` | SAM2 image predictor prompted with each animal's previous centroid, 2 rats | YOLO on the full frame, v1 contacts | Older production pipeline; the only one with chunk mode (`scripts/run_parallel.sh`) |
| `samurai/`, `samurai_sleap/` | SAMURAI (SAM2 video predictor with motion-aware memory) | YOLO / SLEAP | Exploratory |

Why CUTIE was selected over the others: `final_documentation/04_model_comparison.md`.

Removed in October 2026: `reference`, `sam3`, `sam2_yolo`, `sam2_video` (git history keeps them).

## Chunk-mode contract (centroid only; `scripts/run_parallel.sh`, `slurm/run_chunks.sbatch`)

- CLI flags `--start-frame`, `--end-frame`, `--chunk-id N` (run dir tag becomes `<name>_chunkN`)
- Log lines the runner parses: `Run directory: <path>`, `Processed N frames`, `Pipeline complete. N frames processed`
- Per-chunk contact CSVs stay unconsolidated so `scripts/merge_chunks.py` can offset and concatenate them

## Adding a new pipeline

1. Create `src/pipelines/<name>/` with `__init__.py` and `run.py`
2. Reuse `src/common/` (config, video I/O, visualization, contacts v2) and `isolated_composite/composition.py`
3. Add `configs/local_<name>.yaml` and `configs/hpc_<name>.yaml`; list the 7 keypoint names in model index order
4. If it should run chunked on Bunya, honour the contract above
