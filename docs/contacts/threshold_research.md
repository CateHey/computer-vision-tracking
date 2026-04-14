# Temporal Threshold Research for Rat Social Contact Classification

## 1. Context

Our ContactTracker classifies social contacts per-frame at 30 fps and groups them into
"bouts" (continuous events). Two layers of temporal filtering apply:

1. **Inline** (in `src/common/contacts.py`): `bout_min_duration_frames` and `follow_min_frames`
2. **Post-processing** (in `scripts/postprocess_contacts_simple.py`): majority-vote smoothing,
   gap bridging, and minimum bout duration

The question: **how long must a social contact last to be considered real behavior
rather than detector noise?** This document reviews the literature and justifies our
chosen thresholds.

---

## 2. Distance Thresholds in Real-World Units

For reference, using adult lab rat body length (BL) ~ 22 cm (nose to tail base):

| Zone | Threshold (BL) | Real distance | Meaning |
|------|----------------|---------------|---------|
| Contact | < 0.3 BL | < 6.6 cm | Physical contact — noses or bodies touching |
| Proximity | 0.3 – 1.0 BL | 6.6 – 22 cm | Close approach, within reach |
| Independent | > 1.0 BL | > 22 cm | No social interaction |

Contact-type specific distances:
- **N2N** (nose-to-nose): both noses within 6.6 cm
- **N2AG** (nose-to-anogenital): nose within 6.6 cm of other's tail base
- **N2B** (nose-to-body): nose within 9.9 cm of mid-body (1.5× contact radius)
- **T2T** (tail-to-tail): both tail bases within 6.6 cm
- **FOL** (following): nose within 11 cm of leader's tail base + both moving + velocity aligned
- **SBS** (side-by-side): mask overlap + low velocity + parallel bodies

---

## 3. Literature Review: Minimum Bout Durations

### 3.1 SimBA (Simple Behavioral Analysis)

**Nilsson et al., 2020; Goodwin et al., 2024 (Nature Neuroscience)**

- Default minimum bout: **0 ms** (no filtering)
- Recommended approach: **Kleinberg burst detection** (sigma=2, gamma=0.3)
- Per-classifier thresholds from SimBA documentation:
  - Nosing: **50 ms** minimum
  - Adjacent lying (huddling): **750 ms** minimum
  - These are not universal — users tune per behavior
- SimBA treats temporal smoothing as behavior-specific, not one-size-fits-all

### 3.2 MARS (Mouse Action Recognition System)

**Segalin et al., 2021 (eLife)**

- Uses HMM smoothing + 3-frame moving average after XGBoost classification
- Found **1–3 frame false negatives** as the dominant classifier error
- MARS UI provides gap-bridging ("merge bouts < X seconds apart")
- No explicit minimum bout duration — HMM handles temporal coherence
- Designed for mouse behaviors (attack, mount, investigation) at 30 fps

### 3.3 A-SOiD (Active learning SOiD)

**Schweihoff et al., 2024 (Nature Methods)**

- Defines a **"bout length"** parameter: shortest expected duration of the behavior
- **Default: 400 ms** (12 frames at 30 fps), based on CalMS21 mouse dataset
- **10th percentile** of annotated bouts: **200 ms** (6 frames at 30 fps)
- This means 90% of real behavioral bouts are longer than 200 ms
- Used for feature window sizing, not just filtering

### 3.4 DeepEthogram

**Bohnslav et al., 2021 (eLife)**

- `min_bout_length`: **1–2 frames** (minimal filtering)
- Rationale: temporal sequence model already enforces coherence
- Supports per-behavior thresholds via `min_bout_per_behavior`

### 3.5 BehaviorDEPOT / Freezing Literature

**Gabriel et al., 2022 (eLife)**

- Freezing (immobility) standards:
  - IMPReSS: **2 seconds** minimum
  - VideoFreeze: **1 second** minimum
  - Manual observation: **2 seconds**
- These are for passive behaviors — social contacts are faster

### 3.6 Bout Theory

**Berdoy, 1993 (Animal Behaviour)**

- Formalizes the **bout-ending criterion (BEC)**: gap duration that separates
  within-bout pauses from between-bout intervals
- Method: log-frequency distribution of inter-event intervals reveals mixture
  of exponential processes
- **No universal time value** — BEC must be empirically derived per behavior/species

### 3.7 Rat-Specific Social Behavior Timing

From behavioral neuroscience literature on rat social interaction:

- **Social recognition test** (Kogan et al., 2000): investigation bouts measured
  with manual scoring typically range **0.5–5 seconds**
- **Resident-intruder paradigm**: social investigation bouts averaged **1–3 seconds**
- **Anogenital sniffing** is the most stereotyped: typically **0.5–2 seconds** per bout
- **Brief nose touches** (< 300 ms) are often incidental — two rats passing near each
  other, not deliberate social investigation
- **Following** requires sustained pursuit: literature uses **1–3 seconds** minimum

---

## 4. Summary of Findings

| Source | Behavior type | Minimum bout | Notes |
|--------|--------------|-------------|-------|
| SimBA | Nosing | 50 ms | Very permissive, relies on Kleinberg |
| SimBA | Adjacent lying | 750 ms | Passive behavior needs longer |
| A-SOiD | General (CalMS21) | 200–400 ms | 200ms floor, 400ms default |
| DeepEthogram | General | 33–67 ms | Model handles temporal coherence |
| Freezing (IMPReSS) | Immobility | 2,000 ms | Passive behavior |
| Freezing (VideoFreeze) | Immobility | 1,000 ms | Passive behavior |
| Rat social (literature) | Investigation | 500–5,000 ms | Manual scoring standard |
| Rat social (literature) | Following | 1,000–3,000 ms | Sustained pursuit |

**Key insight**: There is no single universal minimum. But for **rat social contacts
classified from keypoint distances**, the consensus points to:

- **< 200 ms**: Almost certainly noise (even A-SOiD's floor is 200 ms)
- **200–300 ms**: Borderline — could be real but often incidental
- **300–500 ms**: Conservative threshold — captures most real contacts
- **> 500 ms**: Safe zone — virtually all sustained social interactions

---

## 5. Chosen Thresholds and Rationale

### 5.1 Inline filtering (contacts.py)

| Parameter | Old | New | Real time (30fps) | Rationale |
|-----------|-----|-----|-------------------|-----------|
| `bout_min_duration_frames` | 2 | **9** | 67ms → **300ms** | Matches A-SOiD floor (200ms) with margin. Removes incidental proximity that isn't deliberate social contact. |
| `follow_min_frames` | 5 | **30** | 167ms → **1.0s** | Literature consensus for following/pursuit behavior. Brief same-direction movement is common in an enclosed arena — 1s ensures sustained pursuit. |

### 5.2 Post-processing filtering (contacts_postprocess_simple.yaml)

| Parameter | Old | New | Real time (30fps) | Rationale |
|-----------|-----|-----|-------------------|-----------|
| `smoothing.window` | 5 | **7** | 167ms → **233ms** | Wider window catches 2–3 frame flickers without smoothing over real transitions (300ms+). |
| `gap_bridging.max_gap` | 3 | **5** | 100ms → **167ms** | Keypoint confidence drops can cause 3–5 frame gaps within a real contact bout. 167ms is still well below any real behavioral transition. |
| `min_bout.duration_sec` | 0.3 | **0.5** | — | Final safety net. At 500ms, only sustained interactions survive. This is the post-processing layer, so it catches anything the inline filter missed. |

### 5.3 Combined effect

The two filtering layers work together:

1. **Inline** (`bout_min_duration_frames=9`): Bouts < 300ms are never written to the raw CSV
2. **Post-processing** (`min_bout.duration_sec=0.5`): After smoothing and gap bridging,
   events < 500ms are removed from the clean output

This means the raw CSV has a 300ms floor and the final events have a 500ms floor.
The 200ms difference between layers allows the post-processing to potentially merge
two 300ms bouts separated by a short gap into a single longer event that survives
the 500ms filter.

### 5.4 Following (FOL) has a stricter minimum

FOL requires `follow_min_frames=30` (1.0s), which is higher than the general
`bout_min_duration_frames=9` (300ms). This is because:

- Brief same-direction movement in a small arena is extremely common
- True following requires sustained pursuit over multiple body lengths
- Literature consistently uses 1–3 second windows for following behavior
- 1.0s is conservative — could increase to 1.5s if FOL is still noisy

---

## 6. Validation Strategy

After running with the new thresholds, check:

1. **Event durations in `event_log.txt`**: All events should be ≥ 0.5s (post-processed) or ≥ 0.3s (raw)
2. **FOL events**: Should be ≥ 1.0s. If many 1.0–1.5s FOL events appear, they may still be noise — consider increasing `follow_min_frames` to 45 (1.5s)
3. **N2N events**: Brief nose-to-nose greetings typically last 0.5–2s. If legitimate greetings are being filtered, consider reducing `min_bout.duration_sec` to 0.4s
4. **SBS events**: Huddling/resting contacts are long (5–30s). These should never be affected by minimum bout filtering
5. **Total contact time**: Should decrease compared to old thresholds. If total contact drops > 50%, thresholds may be too aggressive — check the filtered events

---

## 7. References

1. Nilsson SR et al. (2020). Simple Behavioral Analysis (SimBA). [GitHub](https://github.com/sgoldenlab/simba)
2. Goodwin NL et al. (2024). Nature Neuroscience. doi:10.1038/s41593-024-01649-9
3. Segalin C et al. (2021). MARS. eLife. doi:10.7554/eLife.63720
4. Schweihoff JF et al. (2024). A-SOiD. Nature Methods. doi:10.1038/s41592-024-02200-1
5. Hsu AI & Bhatt PP et al. (2021). B-SOiD. Nature Communications. doi:10.1038/s41467-021-25420-x
6. Bohnslav JP et al. (2021). DeepEthogram. eLife. doi:10.7554/eLife.63377
7. Gabriel CJ et al. (2022). BehaviorDEPOT. eLife. doi:10.7554/eLife.74314
8. Kabra M et al. (2013). JAABA. Nature Methods. doi:10.1038/nmeth.2281
9. Berdoy M (1993). Defining bouts of behaviour. Animal Behaviour. doi:10.1006/anbe.1993.1201
10. Kogan JH et al. (2000). Long-term memory underlying hippocampus-dependent social recognition. Hippocampus.
