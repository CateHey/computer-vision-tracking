# Documentation Index

## Core

| Document | Description |
|----------|-------------|
| [Centroid Pipeline](centroid_pipeline.md) | **Main reference** — architecture, findings, contact research, test checklist, rules |

## Setup

| Document | Description |
|----------|-------------|
| [Local Setup](setup/local.md) | Prerequisites, venv, quick tests |
| [HPC Setup (Bunya)](setup/hpc.md) | Cluster setup, salloc, parallel runs, file transfer |

## Architecture

| Document | Description |
|----------|-------------|
| [Overview](architecture/overview.md) | Project purpose, models, success criteria |
| [Pipeline Comparison](architecture/pipelines.md) | All pipelines compared (centroid is production) |
| [Risks & Limitations](architecture/risks.md) | Known failure modes and workarounds |
| [Reference Comparison](architecture/reference_comparison.md) | What we adopted/rejected from reference toolkit |

## Guides

| Document | Description |
|----------|-------------|
| [Parameter Tuning](guides/parameter_tuning.md) | Config reference with tuning guidance |
| [Labeling](guides/labeling.md) | Minimum viable labels, Roboflow workflow |
| [Evaluation](guides/evaluation.md) | Qualitative checks, quantitative metrics |
| [Debugging](guides/debugging.md) | Debug scripts to isolate pipeline problems |

## Contacts

| Document | Description |
|----------|-------------|
| [Contact Design](contacts/design.md) | 6 contact types, detection rules |
| [Output Format](contacts/output_format.md) | CSV columns, bout format, session JSON |
| [System Audit](contacts/audit.md) | Full audit: bugs, risks, parameters |
| [Threshold Research](contacts/threshold_research.md) | Literature review: minimum bout durations |
| [Post-Processing](contacts_postprocess_simple.md) | 3-rule temporal filtering pipeline |

## Models

| Document | Description |
|----------|-------------|
| [YOLO26 Migration](models/yolo26_migration.md) | Migration from YOLOv8 to YOLO26 |

## Data

| Document | Description |
|----------|-------------|
| [Data Notes](data/notes.md) | Input format, domain gap, key challenges |
| [Output Structure](data/output_structure.md) | Run directories, chunk outputs, merged results |

## Research

| Document | Description |
|----------|-------------|
| [YOLO Limitations](research/yolo_research.md) | Why YOLO fails on identical animals |

## Changes

| Document | Description |
|----------|-------------|
| [Observation Log](changes/observation_log.md) | Chronological log of observations and fixes |
| [Pipeline Comparison](changes/pipeline_comparison.md) | Side-by-side pipeline comparison template |

## Archive (Deprecated)

| Document | Description |
|----------|-------------|
| [Identity Matcher](archive/identity_matcher.md) | Hungarian matching + SEPARATE/MERGED (reference pipeline) |
| [Tracking Analysis](archive/tracking_analysis.md) | BoT-SORT/ByteTrack parameters (sam2_yolo) |
| [Pipeline Improvements](archive/pipeline_improvements.md) | sam2_yolo pipeline design |
| [Centroid Pipeline Plan](archive/centroid_pipeline.md) | Original design plan (executed) |
| [Refactoring Plan](archive/refactoring_plan.md) | Historical codebase restructuring |
