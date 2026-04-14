# Architecture

System design documentation. How each pipeline works, how tracking and identity
assignment are implemented, and why specific design decisions were made.

## Files

- `overview.md` — Project purpose, models (YOLO26, SAM2), success criteria
- `pipelines.md` — Comparison of the 4 pipelines (reference, sam3, sam2_yolo, sam2_video)
- `risks.md` — Known failure modes, workarounds, and risk priorities
- `identity_matcher.md` — Hungarian matching + SEPARATE/MERGED state machine
- `tracking_analysis.md` — BoT-SORT and ByteTrack parameter analysis
- `pipeline_improvements.md` — sam2_yolo pipeline design details, BoT-SORT config
- `reference_comparison.md` — What was adopted/rejected from the reference toolkit
