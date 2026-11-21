# Pipeline Comparison Log

Side-by-side comparison of pipeline runs on the same video.
Use this to decide which pipeline/config works best for a given scenario.

---

## How to Use

1. Run 2+ pipelines on the **same video clip**
2. Watch both overlay videos
3. Fill in a comparison entry below
4. Mark which pipeline "wins" for each criterion

### Quick comparison commands

```bash
# Run all 3 pipelines on the same clip
python -m src.pipelines.reference.run --config configs/local_reference.yaml \
    video_path=data/clips/test.mp4 contacts.enabled=true

python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    video_path=data/clips/test.mp4 contacts.enabled=true

python -m src.pipelines.sam2_video.run --config configs/local_sam2video.yaml \
    video_path=data/clips/test.mp4 contacts.enabled=true
```

---

## Comparison Criteria

| Criterion | What to check |
|-----------|---------------|
| **Identity stability** | Do colors stay consistent? Count ID swaps. |
| **Detection coverage** | Are both rats detected in every frame? Count drops. |
| **Close interaction** | What happens when rats touch/overlap? |
| **Fast crossing** | What happens when rats cross paths quickly? |
| **Wall interaction** | False detections near cage borders? |
| **Mask quality** | Do SAM2 masks fit the rat shape well? |
| **Contact accuracy** | Are N2N, N2B, SBS, FOL contacts correct? |
| **Speed** | FPS, total processing time |

---

## Comparisons

### Comparison 001 — [Pending: first comparison after YOLO26 + state machine]

- **Date:** _TBD_
- **Video:** `data/raw/original_120s.avi`
- **Commit:** _TBD_

| Criterion | reference | sam2_yolo | sam2_video | Winner |
|-----------|-----------|-----------|------------|--------|
| Identity stability | | | | |
| Detection coverage | | | | |
| Close interaction | | | | |
| Fast crossing | | | | |
| Wall interaction | | | | |
| Mask quality | | | | |
| Contact accuracy | | | | |
| Speed (FPS) | | | | |

**Notes:** _Fill after running comparison_

**Conclusion:** _Which pipeline to use and why_

---

<!-- TEMPLATE: Copy this for new comparisons

### Comparison NNN — [Short description]

- **Date:** YYYY-MM-DD
- **Video:** `path/to/video`
- **Commit:** `abc1234`

| Criterion | reference | sam2_yolo | sam2_video | Winner |
|-----------|-----------|-----------|------------|--------|
| Identity stability | | | | |
| Detection coverage | | | | |
| Close interaction | | | | |
| Fast crossing | | | | |
| Wall interaction | | | | |
| Mask quality | | | | |
| Contact accuracy | | | | |
| Speed (FPS) | | | | |

**Notes:**

**Conclusion:**

-->
