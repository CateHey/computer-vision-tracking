# Data Notes

## Input Format

- Raw video format: `.h264` / `.mp4`
- Typical length: ~20 minutes (full recordings)
- Resolution: varies by camera setup

## Execution Modes

### Quick Local Tests
- Process short clips: 5-10 seconds
- Config: `configs/local_quick.yaml`
- Good for qualitative checks and iteration

### Full HPC Runs
- Process entire videos: ~20 minutes
- Config: `configs/hpc_full.yaml`
- Uses Bunya GPU nodes via Slurm

## Data Layout

```
data/
  raw/                  # Full-length source videos (gitignored)
  clips/                # Short test clips (committed)
    output-6s.mp4
    output-10s.mp4
  frames/               # Manually extracted frames (optional)
  roboflow_export/      # Roboflow dataset exports (gitignored)
```

## Domain Gap

Our lab conditions differ from benchmark datasets:
- Custom lighting (IR / visible)
- Cage / arena backgrounds
- Specific camera angles (top-down, angled)
- Frequent occlusions (mice overlapping)

Expect to fine-tune:
- YOLO on Roboflow-labeled frames
- SAM prompting strategy for our specific conditions

## Key Challenge: Interactions

The hardest frames are those with:
- Two mice in close contact (nose-to-nose, climbing, nesting)
- Strong motion blur
- Self-occlusion (mouse body overlapping its own parts)

These are prioritized in the labeling strategy (see [labeling_guide.md](labeling_guide.md)).
