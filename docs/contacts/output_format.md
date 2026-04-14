# Social Contact Classification — Output Format

## 1. Output Files Overview

Each pipeline run produces four contact analysis outputs:

```
<run_dir>/contacts/
    contacts_per_frame.csv      # Every frame, every pair
    contact_bouts.csv           # Grouped temporal events
    session_summary.json        # Aggregated statistics
    report.pdf                  # Visual report (ethogram + charts)
```

---

## 2. Per-Frame CSV (`contacts_per_frame.csv`)

One row per animal pair per frame. For 2 rats this means one row per frame
(pair 0-1). For N rats this would be N*(N-1)/2 rows per frame.

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `frame_idx` | int | Frame number (0-indexed) |
| `time_sec` | float | Timestamp in seconds (`frame_idx / fps`) |
| `rat_a_slot` | int | Slot index of rat A (0 = green, 1 = red) |
| `rat_b_slot` | int | Slot index of rat B |
| `rat_a_track_id` | int/null | YOLO track ID of rat A (may change across video) |
| `rat_b_track_id` | int/null | YOLO track ID of rat B |
| `zone` | str | `"contact"`, `"proximity"`, or `"independent"` |
| `contact_type` | str/null | `"N2N"`, `"N2AG"`, `"N2B"`, `"T2T"`, `"FOL"`, `"SBS"`, or null if no contact |
| `investigator_slot` | int/null | For asymmetric contacts (N2AG, N2B, FOL): the active rat's slot |
| `nose_nose_dist_px` | float/null | Distance between noses in pixels (null if keypoints missing) |
| `nose_nose_dist_bl` | float/null | Same, normalized by body length |
| `nose_tailbase_dist_px` | float/null | Min distance nose_A↔tail_base_B or nose_B↔tail_base_A |
| `nose_tailbase_dist_bl` | float/null | Same, normalized |
| `centroid_dist_px` | float | Centroid-to-centroid distance in pixels |
| `centroid_dist_bl` | float/null | Same, normalized (null if BL unavailable) |
| `mask_iou` | float | SAM2 mask IoU between the two rats (0 if no masks) |
| `body_length_a_px` | float/null | Estimated body length of rat A (nose→tail_base) |
| `body_length_b_px` | float/null | Estimated body length of rat B |
| `velocity_a_px` | float/null | Rat A centroid displacement since previous frame |
| `velocity_b_px` | float/null | Rat B centroid displacement since previous frame |
| `orientation_cos` | float/null | Cosine of angle between body orientation vectors |
| `quality_flag` | str/null | `"high_mask_overlap"`, `"missing_keypoints"`, `"single_detection"`, or null |
| `bout_id` | int/null | ID of the contact bout this frame belongs to (null if no contact) |

### Example rows

```csv
frame_idx,time_sec,rat_a_slot,rat_b_slot,rat_a_track_id,rat_b_track_id,zone,contact_type,investigator_slot,nose_nose_dist_px,nose_nose_dist_bl,nose_tailbase_dist_px,nose_tailbase_dist_bl,centroid_dist_px,centroid_dist_bl,mask_iou,body_length_a_px,body_length_b_px,velocity_a_px,velocity_b_px,orientation_cos,quality_flag,bout_id
0,0.0000,0,1,1,2,independent,,,,,,,,185.3,1.52,0.0,122.1,118.5,,,
125,5.0000,0,1,1,2,contact,N2N,,28.5,0.24,,,,35.2,0.29,0.03,119.8,121.0,4.2,3.8,0.92,,7
126,5.0400,0,1,1,2,contact,N2AG,0,45.1,0.37,22.3,0.18,,42.1,0.35,0.01,120.5,120.8,5.1,2.3,-0.85,,8
400,16.0000,0,1,1,2,contact,SBS,,,,,,68.2,0.56,0.08,121.5,119.2,1.2,0.8,0.95,,15
```

---

## 3. Contact Bouts CSV (`contact_bouts.csv`)

A **bout** is a continuous period of the same contact type between the same pair,
allowing small gaps (up to `bout_max_gap_frames`).

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `bout_id` | int | Unique bout identifier |
| `contact_type` | str | `"N2N"`, `"N2AG"`, `"N2B"`, `"T2T"`, `"FOL"`, `"SBS"` |
| `rat_a_slot` | int | Slot index of rat A |
| `rat_b_slot` | int | Slot index of rat B |
| `investigator_slot` | int/null | For asymmetric types: the active rat |
| `start_frame` | int | First frame of the bout |
| `end_frame` | int | Last frame of the bout |
| `start_time_sec` | float | Start timestamp |
| `end_time_sec` | float | End timestamp |
| `duration_sec` | float | `end_time_sec - start_time_sec` |
| `duration_frames` | int | Number of frames with active contact |
| `total_frames` | int | `end_frame - start_frame + 1` (includes gap frames) |
| `mean_nose_dist_px` | float/null | Average nose-nose distance during bout |
| `mean_mask_iou` | float | Average mask IoU during bout |
| `mean_velocity_a_px` | float/null | Average velocity of rat A during bout |
| `mean_velocity_b_px` | float/null | Average velocity of rat B during bout |
| `quality_flags` | str/null | Comma-separated list of any quality flags raised during bout |

### Example rows

```csv
bout_id,contact_type,rat_a_slot,rat_b_slot,investigator_slot,start_frame,end_frame,start_time_sec,end_time_sec,duration_sec,duration_frames,total_frames,mean_nose_dist_px,mean_mask_iou,mean_velocity_a_px,mean_velocity_b_px,quality_flags
0,N2N,0,1,,125,138,5.00,5.52,0.52,12,14,30.2,0.02,3.8,4.1,
1,N2AG,0,1,0,145,162,5.80,6.48,0.68,15,18,48.1,0.01,5.2,2.1,
2,SBS,0,1,,400,475,16.00,19.00,3.00,72,76,,0.07,1.1,0.9,
3,FOL,0,1,0,520,548,20.80,21.92,1.12,25,29,,0.0,8.5,7.2,
```

---

## 4. Session Summary JSON (`session_summary.json`)

Aggregated statistics for the entire video session.

```json
{
  "metadata": {
    "video_path": "data/clips/output-10s.mp4",
    "video_duration_sec": 1200.0,
    "total_frames": 30000,
    "fps": 25.0,
    "num_rats": 2,
    "analysis_date": "2026-02-25T13:45:00",
    "pipeline_version": "1.0.0",
    "config_hash": "abc123"
  },

  "parameters": {
    "min_keypoint_conf": 0.3,
    "contact_zone_bl": 0.3,
    "proximity_zone_bl": 1.0,
    "fallback_body_length_px": 120,
    "bout_max_gap_frames": 3,
    "bout_min_duration_frames": 2
  },

  "zone_summary": {
    "contact_frames": 4500,
    "contact_pct": 15.0,
    "proximity_frames": 3200,
    "proximity_pct": 10.67,
    "independent_frames": 22300,
    "independent_pct": 74.33
  },

  "contact_type_summary": {
    "N2N": {
      "total_bouts": 45,
      "total_duration_sec": 28.4,
      "total_frames": 710,
      "pct_of_session": 2.37,
      "mean_bout_duration_sec": 0.63,
      "median_bout_duration_sec": 0.48,
      "max_bout_duration_sec": 2.1,
      "mean_bout_nose_dist_px": 25.3
    },
    "N2AG": {
      "total_bouts": 62,
      "total_duration_sec": 58.2,
      "total_frames": 1455,
      "pct_of_session": 4.85,
      "mean_bout_duration_sec": 0.94,
      "median_bout_duration_sec": 0.72,
      "max_bout_duration_sec": 4.8,
      "by_investigator": {
        "slot_0": { "bouts": 35, "duration_sec": 32.1 },
        "slot_1": { "bouts": 27, "duration_sec": 26.1 }
      }
    },
    "N2B": {
      "total_bouts": 38,
      "total_duration_sec": 22.8,
      "total_frames": 570,
      "pct_of_session": 1.90,
      "mean_bout_duration_sec": 0.60,
      "median_bout_duration_sec": 0.44,
      "max_bout_duration_sec": 1.9,
      "by_investigator": {
        "slot_0": { "bouts": 20, "duration_sec": 12.5 },
        "slot_1": { "bouts": 18, "duration_sec": 10.3 }
      }
    },
    "SBS": {
      "total_bouts": 12,
      "total_duration_sec": 85.6,
      "total_frames": 2140,
      "pct_of_session": 7.13,
      "mean_bout_duration_sec": 7.13,
      "median_bout_duration_sec": 5.2,
      "max_bout_duration_sec": 18.4
    },
    "FOL": {
      "total_bouts": 8,
      "total_duration_sec": 12.0,
      "total_frames": 300,
      "pct_of_session": 1.00,
      "mean_bout_duration_sec": 1.50,
      "median_bout_duration_sec": 1.2,
      "max_bout_duration_sec": 3.1,
      "by_investigator": {
        "slot_0": { "bouts": 5, "duration_sec": 7.5 },
        "slot_1": { "bouts": 3, "duration_sec": 4.5 }
      }
    }
  },

  "per_pair_summary": {
    "pair_0_1": {
      "total_contact_sec": 207.0,
      "total_contact_pct": 17.25,
      "contact_types": {
        "N2N": { "bouts": 45, "duration_sec": 28.4 },
        "N2AG": { "bouts": 62, "duration_sec": 58.2 },
        "N2B": { "bouts": 38, "duration_sec": 22.8 },
        "SBS": { "bouts": 12, "duration_sec": 85.6 },
        "FOL": { "bouts": 8, "duration_sec": 12.0 }
      }
    }
  },

  "quality": {
    "high_mask_overlap_frames": 15,
    "missing_keypoints_frames": 230,
    "single_detection_frames": 85,
    "total_flagged_pct": 1.1
  },

  "time_bins": {
    "bin_duration_sec": 60,
    "bins": [
      {
        "bin_start_sec": 0,
        "bin_end_sec": 60,
        "contact_sec": 8.2,
        "N2N_sec": 1.2,
        "N2AG_sec": 3.5,
        "N2B_sec": 1.0,
        "SBS_sec": 2.0,
        "FOL_sec": 0.5
      }
    ]
  }
}
```

---

## 5. PDF Report (`report.pdf`)

Generated with matplotlib. Single PDF file, one page per figure.

### Page 1: Timeline Ethogram

Horizontal bar chart spanning the full video duration.

```
Time (seconds) →
0          60         120        180        ...

Rat pair   ████░░░████████░░░░░████░░░░████████████
0-1:       N2N  N2AG        N2B       SBS

Legend:  ■ N2N (blue)  ■ N2AG (orange)  ■ N2B (green)  ■ SBS (purple)  ■ FOL (red)
```

Each contact bout is a colored segment on the timeline. Gaps between bouts
are empty. Stacked if multiple types overlap (rare with mutual exclusion).

### Page 2: Contact Duration Histograms

One subplot per contact type (2x3 grid). Each histogram shows the distribution
of bout durations in seconds.

```
N2N                    N2AG                   N2B
 ██                     █                      ██
████                   ████                   ████
█████                 ██████                 ██████
0  1  2  3s          0  2  4  6s           0  1  2  3s

SBS                    FOL                    ALL CONTACTS
     ████                ██
    ██████              ████
   █████████           ██████
0  5  10  15s         0  1  2  3s          0  5  10  15s
```

### Page 3: Contact Type Pie Chart

Pie chart showing proportion of total contact time per type.

```
        ┌───────┐
       /  N2AG   \      N2N:    13.7%
      │  28.1%    │     N2AG:   28.1%
      │           │     N2B:    11.0%
       \  SBS    /      SBS:    41.3%
        │ 41.3% │       FOL:     5.8%
        └───────┘
```

### Page 4: Time-Binned Contact Rates

Stacked area chart showing contact seconds per minute over time.
Useful for detecting habituation (decreasing social interaction over time).

```
Contact
sec/min
  12 │  ████████
  10 │ ██████████
   8 │████████████████
   6 │████████████████████
   4 │██████████████████████████
   2 │████████████████████████████████
   0 └──────────────────────────────────
     0    5    10   15   20  minutes
```

### Page 5: Per-Pair Summary Table (for N > 2 rats)

For 2-rat experiments this is a single row. For multi-rat experiments,
a table showing contact statistics per pair.

| Pair | Total contact (s) | N2N | N2AG | N2B | SBS | FOL |
|------|-------------------|-----|------|-----|-----|-----|
| 0-1  | 207.0            | 28.4| 58.2 | 22.8| 85.6| 12.0|

---

## 6. File Size Estimates

For a 20-minute video at 25 fps (30,000 frames):

| File | Estimated size |
|------|---------------|
| `contacts_per_frame.csv` | ~4-6 MB (30k rows, ~150 bytes/row) |
| `contact_bouts.csv` | ~10-50 KB (100-500 bouts, ~200 bytes/row) |
| `session_summary.json` | ~5-10 KB |
| `report.pdf` | ~200-500 KB (5 pages with charts) |

---

## 7. Compatibility Notes

- **CSV files** use standard comma separation, UTF-8 encoding, with header row.
  Compatible with pandas, R, Excel, SPSS, Prism.
- **JSON** uses standard format with 2-space indentation. Compatible with Python
  `json.load()`, R `jsonlite`, JavaScript.
- **PDF** generated by matplotlib with `PdfPages`. Requires `matplotlib` (already
  in our environment for visualization).
- **Null values** in CSV are empty strings (`,,`). In JSON they are `null`.
- **Timestamps** are in seconds from video start (float, 4 decimal places).
  Frame indices are 0-based integers.
