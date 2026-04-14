# Labeling Guide (Roboflow)

## Goal

Create a labeled dataset that supports **body-part localization** for lab mice.

## Minimum Viable Labels

Every annotated frame should have:

| Class | Type | Description |
|-------|------|-------------|
| **Mouse** | Instance (bbox) | Full body bounding box |
| **Nose** | Keypoint / small bbox | Tip of the snout |
| **Head** | Keypoint / small bbox | Center of the head |
| **Tail** | Keypoint / small bbox | Tail tip |

## Recommended Additions

If time allows, also label:

- **Tail base** — helps determine body orientation
- **Torso center** — useful for trajectory smoothing
- **Ears** — useful for head pose estimation

## Where to Focus Labeling

Prioritize frames with:

1. **Strong motion blur** — tests detection robustness
2. **Self-occlusion** — mouse body overlapping its own parts
3. **Interaction scenarios** — two mice nose-to-nose, tail contact, climbing
4. **Edge cases** — partial visibility, unusual poses

These "hard cases" usually determine tracking quality.

## Workflow

### 1. Extract Candidate Frames

Use the close-contact detection pipeline to find interaction frames:

```bash
python scripts/extract_frames.py all --config configs/local_quick.yaml
```

### 2. Upload to Roboflow

```bash
export ROBOFLOW_API_KEY="your-key"

python scripts/upload_to_roboflow.py \
    --config configs/local_quick.yaml \
    --frames-root outputs/runs/<run-dir>/frames \
    --tag close_contact
```

### 3. Annotate in Roboflow

- Use the Roboflow web interface to draw bounding boxes / keypoints
- Aim for consistent annotation style across frames
- Use the "Smart Polygon" tool for precise mouse outlines (optional)

### 4. Export Dataset

- Export from Roboflow in YOLOv8 format
- Download to `data/roboflow_export/`
- The export should include a `data.yaml` file for training

## Annotation Tips

- Label **every visible mouse** in the frame, even partially occluded ones
- For body parts, use small tight bounding boxes (~20x20 px) centered on the feature
- When a body part is occluded, skip it (don't guess)
- Maintain consistent class names across all annotations
