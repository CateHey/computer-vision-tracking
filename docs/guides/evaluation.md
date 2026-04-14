# Evaluation Plan

## Qualitative Checks

### Overlay Video Inspection
- Do nose/head/tail points sit correctly on the animal?
- Do masks follow the mouse without drifting?
- Are identities maintained across frames (no color swaps)?
- How does tracking behave during interactions?

### What to Look For
- Clean mask boundaries (no jagged edges, no bleeding into background)
- Stable centroid positions (no jitter between frames)
- Correct identity persistence (green mouse stays green)

## Quantitative Metrics (when labels exist)

### Body-Part Localization Error
- Distance (in pixels) between predicted and ground-truth body part positions
- Reported per part: nose, head, tail

### Mask Quality
- **IoU:** intersection over union between predicted and ground-truth masks
- **Boundary quality:** precision of mask edges

### Track Stability
- **Missing frames:** percentage of frames without a confident detection for nose/tail
- **Identity switches:** number of times identity assignments swap between animals
- **Jitter:** frame-to-frame displacement spikes (sudden jumps in centroid position)

## Comparison Framework

When comparing SAM2 vs SAM3 pipelines:

1. Run both on the same video/clip
2. Compare overlay videos side by side
3. Compare quantitative metrics on labeled frames
4. Document findings in the run's output directory

## Current Priority

Focus on **qualitative evaluation** first:
- Does the pipeline produce reasonable overlays?
- Are close-contact frames correctly identified?
- Does tracking maintain identity during simple motion?

Quantitative metrics will become important once:
- Labeled dataset exists (Roboflow annotations)
- Multiple pipeline configurations need comparison
- Fine-tuning experiments are underway
