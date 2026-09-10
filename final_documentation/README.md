# Rat Social Behaviour Tracking — Final Documentation

Automated tracking and social-contact classification for pairs of laboratory rats
from overhead video. Written as the methods reference for the paper.

> **Read this first.** The current pipeline lives on `origin/develop`, not `main`.
> The in-repo `CLAUDE.md` still names the older *centroid* pipeline as the
> production default; that has been out of date since `cutie_composite` landed.

## The four documents

| Document | Content |
|----------|---------|
| [01_pipeline.md](01_pipeline.md) | The selected pipeline: SAM3 → CUTIE with per-animal composites |
| [02_contact_detection.md](02_contact_detection.md) | How contacts are scored, classified into 5 types, and cleaned into events — **including three defects that currently suppress two of them** |
| [03_outputs_and_reports.md](03_outputs_and_reports.md) | Every file produced, the report, the viewer, and how to run it |
| [04_model_comparison.md](04_model_comparison.md) | Why CUTIE: what each segmentation backbone did and how it failed |

---

## System in one page

**Input.** A video of rats in an arena, single fixed overhead camera. The shipped
configs are set up for **6 animals**.

**Output.** An overlay video, a single `results.xlsx` holding every table (per-frame
geometry, classified events with timestamps, session summary, parameters), a PDF report,
and an HTML viewer that reads the workbook.

**Three models, strictly separated roles:**

| Model | Role | Runs on |
|-------|------|---------|
| SAM3 | Bootstrap — turns the text prompt `"mouse"` into initial masks | First 125 frames |
| CUTIE | Segmentation **and identity** — propagates one labelled index mask | Every frame after the handoff |
| YOLO pose (`yolo26_v11.pt`) | 5 body keypoints | Every frame, on a per-animal composite |

Two design decisions carry the system. **Identity is propagated, not re-derived**:
CUTIE holds object memory and emits one integer id per pixel, so two animals can
never share a pixel and there is no per-frame assignment step to get wrong. And
**pose never sees two animals at once**: for each animal, every other animal is
painted out using a background plate, so the keypoints come back unmixed and
already in frame coordinates.

**Processing chain:**

```
video
  → SAM2 centroid propagation          → 2 masks + 2 centroids per frame  (identity)
  → YOLO pose on full frame            → 7 keypoints per rat              (geometry)
  → keypoint-to-mask assignment        → per-rat keypoint sets
  → pairwise geometric classification  → one contact label per frame
  → bout grouping + temporal filtering → discrete behavioural events
  → results.xlsx, PDF report, HTML viewer
```

Processing chain:

```
video
  → SAM3 (bootstrap)  → CUTIE (propagate)   → one labelled mask per animal
  → erase other animals → YOLO pose          → keypoints per animal
  → continuous scoring + Schmitt triggers    → one contact label per frame
  → bouts + temporal filtering               → discrete behavioural events
  → results.xlsx, PDF report, HTML viewer
```

**Five social contact types**, plus `NC`: nose-to-nose (N2N), nose-to-anogenital
(N2AG), nose-to-body (N2B), side-by-side (SBS) and following (FOL). They are
grouped into families — investigative, affiliative, non-contact — and each frame
also carries a **dynamics** label (closing / stable / separating).

> ⚠️ **Two of the five cannot currently fire.** N2AG depends on a keypoint the
> shipped model does not have, and SBS depends on mask overlap that CUTIE's
> output makes structurally impossible. Their frames are absorbed by other types
> rather than left unlabelled. See
> [02 §2.6](02_contact_detection.md#26-three-defects-that-suppress-contact-types)
> — this must be resolved before any behavioural result is reported.

**Scale.** `cutie_composite` loads the whole video into memory and runs as a single
process — it has no chunk mode, so the parallel runner and chunk merger do not
apply to it. Those belong to the older `centroid` pipeline.

---

## Source of truth

This documentation describes the code as it stands in this repository. Key files:

All paths below are on `origin/develop` unless marked.

| Component | File |
|-----------|------|
| Pipeline | `src/pipelines/cutie_composite/run.py` |
| CUTIE wrapper | `src/pipelines/cutie_composite/cutie_tracker.py` |
| Composite ("erase the other animal") | `src/pipelines/isolated_composite/composition.py` |
| Contact scoring and bouts | `src/common/contacts_v2.py` |
| Temporal post-processing | [scripts/postprocess_contacts_simple.py](../scripts/postprocess_contacts_simple.py) |
| Parameters | `configs/local_cutie_composite.yaml`, `configs/hpc_cutie_composite.yaml` |
| Demo generator (this branch) | [scripts/make_demo_session.py](../scripts/make_demo_session.py) |
| Excel consolidation | [src/common/excel_report.py](../src/common/excel_report.py) |
| HTML report viewer | [scripts/report_viewer.html](../scripts/report_viewer.html) |
