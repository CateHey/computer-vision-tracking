# Documentation Index

The methods reference for the paper is [`final_documentation/`](../final_documentation/README.md)
(pipeline `cutie_composite`, contacts v2, reports). `docs/` is older design history, mostly written
for the `centroid` pipeline; it is kept for the parts that still apply (setup, tuning, contact research).

## Core

| Document | Description |
|----------|-------------|
| [Centroid Pipeline](centroid_pipeline.md) | Reference for the older `centroid` pipeline: architecture, findings, contact research, lessons learned |

## Setup

| Document | Description |
|----------|-------------|
| [Local Setup](setup/local.md) | venv, PyTorch, SAM2, CUTIE, HF token, quick tests |
| [HPC Setup (Bunya)](setup/bunya_paso_a_paso.md) | Account string, first-time setup, `cutie_composite` and `centroid` runs, file transfer, troubleshooting |

## Architecture

| Document | Description |
|----------|-------------|
| [Overview](architecture/overview.md) | Project purpose, models, success criteria |
| [Pipeline Comparison](architecture/pipelines.md) | Historical comparison; several pipelines listed there were removed in October 2026 |
| [Risks & Limitations](architecture/risks.md) | Known failure modes and workarounds |

## Guides

| Document | Description |
|----------|-------------|
| [Parameter Tuning](guides/parameter_tuning.md) | Config reference with tuning guidance (the SlotTracker section refers to the removed sam2_yolo pipeline) |
| [Labeling](guides/labeling.md) | Minimum viable labels, frame extraction, Roboflow workflow |
| [Evaluation](guides/evaluation.md) | Qualitative checks, quantitative metrics |

## Contacts

| Document | Description |
|----------|-------------|
| [Contact Design](contacts/design.md) | v1 contact types and rules (`src/common/contacts.py`); v2 is described in `final_documentation/02` |
| [Output Format](contacts/output_format.md) | CSV columns, bout format, session JSON |
| [System Audit](contacts/audit.md) | Audit of the v1 contact system: bugs, risks, parameters |
| [Threshold Research](contacts/threshold_research.md) | Literature review: minimum bout durations |
| [Post-Processing](contacts_postprocess_simple.md) | 3-rule temporal filtering pipeline |

## Models

| Document | Description |
|----------|-------------|
| [YOLO26 Migration](models/yolo26_migration.md) | Migration from YOLOv8 to YOLO26 (historical) |

## Data

| Document | Description |
|----------|-------------|
| [Data Notes](data/notes.md) | Input format, domain gap, key challenges |
| [Output Structure](data/output_structure.md) | Run directories, chunk outputs, merged results |

## Changes

| Document | Description |
|----------|-------------|
| [Observation Log](changes/observation_log.md) | Chronological log of observations and fixes |
