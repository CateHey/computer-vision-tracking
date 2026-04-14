# Bout Post-Processing for Social Contact Classification

## Research basis for bout post-processing

Frame-by-frame behavioral classifiers (whether manual scoring or automated) produce noisy binary labels that need temporal post-processing before they become meaningful "bouts" or "events." This section reviews how established tools and literature handle this problem, and justifies the default parameters for our ContactTracker.

---

### Source 1: SimBA (Simple Behavioral Analysis)

**Nilsson et al., 2020; Goodwin et al., 2024**

- Provides a **"Minimum behavior bout length (ms)"** parameter applied after binary classification.
- Example from docs: at 50 fps, predictions `1,1,1,1,0,1,1,1,1` normally yield two 80 ms bouts. Setting minimum bout to 20 ms merges them into a single 180 ms bout.
- Default is **0 ms** — the documentation recommends using **Kleinberg burst detection** (sigma=2, gamma=0.3) for all temporal smoothing instead of heuristic thresholds.
- The minimum bout parameter effectively **bridges short gaps** by overwriting interruptions shorter than the threshold.

Links:
- [SimBA validation tutorial](https://github.com/sgoldenlab/simba/blob/master/docs/validation_tutorial.md)
- [SimBA Kleinberg smoothing docs](https://simba-docs.readthedocs.io/en/latest/docs/tutorials/kleinberg.html)
- [Goodwin et al. 2024, Nature Neuroscience](https://www.nature.com/articles/s41593-024-01649-9)

### Source 2: JAABA (Janelia Automatic Animal Behavior Annotator)

**Kabra et al., 2013, Nature Methods; Robie et al., 2020**

- Implements a clean **two-pass** post-processing pipeline:
  1. **Maximum gap filling**: non-behavior frames "sandwiched" between behavior frames are filled in (gap bridging).
  2. **Minimum bout length**: after gap filling, bouts shorter than a threshold are removed.
- Both parameters are **swept per-behavior** in recall-precision analysis — no universal default, but the two-pass architecture (gap-fill first, then min-bout) is the recommended pattern.
- The paper notes that "fragmentation of a seemingly single behavioral bout happens when a bout is relatively short (lunges, headbutts) or at the edge of a JAABA-defined bout."

Links:
- [Kabra et al. 2013, Nature Methods](https://www.nature.com/articles/nmeth.2281)
- [Robie et al. 2020, bioRxiv](https://www.biorxiv.org/content/10.1101/2020.06.16.153130v1.full)

### Source 3: MARS (Mouse Action Recognition System)

**Segalin et al., 2021, eLife**

- After XGBoost frame-level classification, MARS applies:
  1. **HMM smoothing** — enforces one-hot labeling and smooths transitions.
  2. **Three-frame moving average** — additional pass after HMM.
- Authors noted that "brief sequences (1-3 frames) of false negatives or false positives shorter than typical annotation bouts" were their most common classifier error — motivating the smoothing pipeline.
- The MARS UI provides a **"merge bouts occurring less than X seconds apart"** gap-bridging parameter.
- Inter-annotator study: a major source of disagreement was "the extent to which annotators merged together consecutive bouts of the same behavior."

Links:
- [Segalin et al. 2021, eLife](https://elifesciences.org/articles/63720)

### Source 4: A-SOiD / B-SOiD

**Schweihoff et al., 2024, Nature Methods; Hsu & Bhatt et al., 2021**

- A-SOiD defines a **"bout length"** parameter: the shortest duration a definable component of the designated behavior is expected to last.
- **Default: 400 ms (12 frames at 30 fps)**, based on analysis of the CalMS21 dataset.
- At the **10th percentile** of annotated bout durations, 200 ms (6 frames at 30 fps) was sufficient to resolve behavioral changes.
- B-SOiD downsamples to 10 fps and applies 20x frame-shifted segmentation for inherent temporal smoothing.

Links:
- [Schweihoff et al. 2024, Nature Methods](https://www.nature.com/articles/s41592-024-02200-1)
- [Hsu & Bhatt et al. 2021, Nature Communications](https://www.nature.com/articles/s41467-021-25420-x)

### Source 5: DeepEthogram

**Bohnslav et al., 2021, eLife**

- Has a `postprocessor.min_bout_length` config parameter.
- **Default: 1–2 frames** (minimal filtering), because the architecture uses a **temporal sequence model** with large receptive field that already enforces coherence.
- Configuration: `postprocessor: { min_bout_length: 2, type: min_bout }`.
- Supports per-behavior thresholds via `type: min_bout_per_behavior`.

Links:
- [Bohnslav et al. 2021, eLife](https://elifesciences.org/articles/63377)

### Source 6: Classical freezing literature + BehaviorDEPOT

**Gabriel et al., 2022, eLife; IMPReSS standard; VideoFreeze**

- BehaviorDEPOT uses a three-stage pipeline: (1) **sliding window convolution**, (2) **low-pass velocity filter**, (3) **minimum duration threshold**.
- Well-established field standards for freezing (immobility) bouts:
  - **IMPReSS (International Mouse Phenotyping)**: minimum **2 seconds**
  - **VideoFreeze system**: minimum **1 second**
  - **Manual observation standard**: **2 seconds** of continuous immobility
- Social contact behaviors (sniffing, following) are faster and shorter than freezing, so 0.25–1.0 s is more appropriate.

Links:
- [Gabriel et al. 2022, eLife](https://elifesciences.org/articles/74314)
- [IMPReSS Fear Conditioning Protocol](https://web.mousephenotype.org/impress/ProcedureInfo?action=list&procID=199)

### Supplementary: Bout theory (Berdoy, 1993)

**"Defining bouts of behaviour: A three-process model," Animal Behaviour**

- Formalizes the **bout-ending criterion (BEC)**: the time gap that separates within-bout intervals from between-bout intervals.
- Method: plot **log-frequency distribution** of inter-event intervals. The distribution shows a mixture of exponential processes (short within-bout gaps vs. long between-bout gaps). The crossover point defines the BEC.
- Recommends log-frequency over log-survivorship plots because data points are independent.
- **No universal time value** — BEC is empirically derived per behavior and species.

Link:
- [Berdoy 1993, Animal Behaviour](https://www.sciencedirect.com/science/article/abs/pii/S0003347283712017)

---

## Summary table

| Tool | Parameter | Default / typical value |
|------|-----------|------------------------|
| SimBA | Minimum bout length | 0 ms (use Kleinberg instead) |
| SimBA | Kleinberg sigma / gamma | 2 / 0.3 |
| JAABA | Gap fill + min bout | User-tunable per behavior (two-pass) |
| MARS | Smoothing | HMM + 3-frame moving average |
| A-SOiD | Bout length (feature window) | 400 ms; 200 ms at 10th percentile |
| DeepEthogram | `min_bout_length` | 1–2 frames |
| BehaviorDEPOT | Sliding window + min duration | All tunable |
| Freezing (IMPReSS) | Minimum bout | 2 seconds |
| Freezing (VideoFreeze) | Minimum bout | 1 second |
| Bout theory (Berdoy) | Inter-bout criterion | Empirically derived from log-frequency plots |

---

## Justification for our default parameters

Our ContactTracker post-processing applies **3 simple rules** in sequence. The defaults are justified by the literature above:

### Rule 1: Majority-vote smoothing (window = 7 frames)

- **What it does**: Slides a 7-frame window over per-frame contact labels. Each frame adopts the most frequent label in its window.
- **Why 7 frames**: At 30 fps, 7 frames = 233 ms. MARS found that 1–3 frame glitches were their most common classifier error. A 7-frame window corrects isolated 2–3 frame flickers while staying short enough to preserve real behavioral transitions (~300ms+). This is wider than MARS's "3-frame average" to account for the noisier keypoint-based classification in our pipeline.
- **Literature support**: MARS (3-frame average), BehaviorDEPOT (sliding window convolution), B-SOiD (inherent temporal smoothing via frame-shifting).

### Rule 2: Gap bridging (max_gap = 5 frames, ~167 ms at 30 fps)

- **What it does**: If the same contact type reappears within 5 frames of ending, the gap is filled and the bout continues.
- **Why 5 frames / 167 ms**: MARS explicitly found "brief sequences (1-3 frames) of false negatives" as the dominant error mode, but our keypoint-based classifier can produce 3–5 frame dropouts when a keypoint confidence briefly dips below threshold. 167 ms is well below any reasonable behavioral transition time for rodent social contacts, so bridging at this level only merges within-bout gaps without merging distinct events.
- **Literature support**: JAABA (gap-fill as first pass), SimBA (minimum bout bridges gaps), MARS (1–3 frame FN as dominant error).

### Rule 3: Minimum bout duration (default = 0.5 s = 15 frames at 30 fps)

- **What it does**: After smoothing and gap bridging, bouts shorter than 0.5 s are discarded as noise.
- **Why 0.5 s**: A-SOiD found that 200 ms (10th percentile of CalMS21 annotated bouts) was the floor for resolving real behavioral events, with 400 ms as the default. Research on rat social behavior indicates that meaningful social contacts require sustained interaction — a brief nose-touch under 300 ms is more likely detector noise than intentional social investigation. The 500 ms threshold ensures only sustained contacts are reported, sitting above the A-SOiD 400 ms default while remaining practical for rat social contacts. See [Threshold Research](contacts/threshold_research.md) for detailed literature review.
- **Literature support**: A-SOiD (200–400 ms range), SimBA (per-classifier thresholds, nosing=50ms to adjacent lying=750ms), freezing literature (1–2 s upper bound), rat behavioral studies (300 ms minimum for meaningful social contact).

### Why not Kleinberg / HMM?

SimBA recommends Kleinberg burst detection, and MARS uses HMM smoothing. Both are effective but add algorithmic complexity. Our 3-rule pipeline (majority vote → gap bridge → min bout) is:
- **Transparent**: each rule has a clear, interpretable effect
- **Tunable**: parameters map directly to observable quantities (frames, seconds)
- **Sufficient**: covers the same error modes (flicker, short gaps, spurious bouts) that motivate the more complex approaches
- **Consistent with JAABA's architecture**: gap-fill first, then minimum bout — the cleanest conceptual model

For a 2-rat social contact system with 5 contact types at 30 fps, the 3-rule approach matches the accuracy needs without the overhead of fitting HMM transition matrices or Kleinberg hyperparameters.

---

## Usage

### Basic usage

```bash
# Post-process contacts from a pipeline run (uses built-in defaults)
python scripts/postprocess_contacts_simple.py outputs/runs/2026-03-04_ref/contacts/

# With explicit FPS
python scripts/postprocess_contacts_simple.py path/to/contacts/ --fps 30

# With config file
python scripts/postprocess_contacts_simple.py path/to/contacts/ \
    --config configs/contacts_postprocess_simple.yaml

# With reports (4 PNGs + 2 CSVs)
python scripts/postprocess_contacts_simple.py path/to/contacts/ --make_reports

# Override parameters via CLI
python scripts/postprocess_contacts_simple.py path/to/contacts/ \
    smoothing.window=7 min_bout.duration_sec=0.5

# Separate output directory
python scripts/postprocess_contacts_simple.py path/to/contacts/ \
    --output_dir results/postprocessed/ --make_reports
```

### FPS resolution

The script resolves FPS using a 4-level priority cascade:

1. `--fps` CLI argument (highest priority)
2. `session_summary.json` in the input directory (`metadata.fps`)
3. `--video_path` argument (reads FPS via OpenCV)
4. Config fallback value `fps_fallback: 30.0` (with warning)

### Input requirements

- `contacts_per_frame.csv` must exist in the input directory (required)
- Required columns: `frame_idx`, `time_sec`, `zone`, `contact_type`
- Optional but used if present: `investigator_slot`, `nose_nose_dist_px`, `centroid_dist_px`, `mask_iou`
- `session_summary.json` (optional, used for FPS auto-detection)

---

## Outputs

### Always produced (4 files)

| File | Description |
|------|-------------|
| `contacts_real_per_frame.csv` | Original CSV + 3 new columns: `real_type`, `real_zone`, `real_event_id` |
| `contacts_real_events.csv` | One row per clean event with human-readable timestamps |
| `event_log.txt` | Human-readable event log with MM:SS timestamps for quick validation |
| `session_summary_real.json` | Metadata, parameters, raw vs real comparison, filtering impact |

### With `--make_reports` (6 files in reports/)

| File | Description |
|------|-------------|
| `timeline_comparison.png` | Raw vs real contact labels over time (X-axis MM:SS) |
| `duration_by_type.png` | Bar chart: total seconds per contact type, raw vs real |
| `events_by_type.png` | Bar chart: event count per type, raw vs real |
| `event_duration_distribution.png` | Histogram of real event durations per type |
| `comparison_by_type.csv` | Per-type statistics: bouts, durations, means |
| `comparison_global.csv` | Global statistics: contact time, flicker rate, parameters |

---

## Event table format (`contacts_real_events.csv`)

This is the key validation table. Each row is one behavioral event with
human-readable timestamps that can be looked up directly in the video:

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `event_id` | int | 1 | Sequential event ID |
| `contact_type` | str | "N2AG" | N2N, N2AG, N2B, FOL, or SBS |
| `start_frame` | int | 375 | First frame of the event |
| `end_frame` | int | 412 | Last frame (inclusive) |
| `start_time` | str | "0:12.5" | Human-readable start (MM:SS.d or H:MM:SS.d) |
| `end_time` | str | "0:13.7" | Human-readable end |
| `duration` | str | "1.2s" | Human-readable duration |
| `start_time_sec` | float | 12.500 | Raw seconds (for programmatic use) |
| `end_time_sec` | float | 13.733 | Raw seconds |
| `duration_sec` | float | 1.233 | Raw seconds |
| `duration_frames` | int | 37 | Number of frames |
| `investigator_slot` | int/empty | 0 | Which rat initiated (for asymmetric types) |
| `mean_nose_nose_dist_px` | float/null | 42.3 | Mean nose-nose distance during event |
| `mean_centroid_dist_px` | float/null | 65.1 | Mean centroid distance during event |
| `mean_mask_iou` | float/null | 0.02 | Mean mask overlap during event |

### How to validate

1. Open `event_log.txt` for a quick overview
2. Pick an event (e.g., event #3: N2B at 0:22.1)
3. Open the video, seek to 0:22
4. Observe through the end time — the contact type and investigator should match
5. If events look wrong, adjust parameters (e.g., increase `min_bout.duration_sec`)
