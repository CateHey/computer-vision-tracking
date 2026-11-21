# Identity Matcher Design

The `IdentityMatcher` is the tracking component of the reference pipeline.
It maintains consistent rat identities across frames without relying on
YOLO's built-in tracker (BoT-SORT).

**File:** `src/pipelines/reference/identity_matcher.py`

---

## Core Approach

Hungarian assignment with soft cost matrix, velocity prediction, and an
N=2 interaction state machine (SEPARATE ↔ MERGED).

### Why not use YOLO's tracker?

BoT-SORT (YOLO's built-in tracker) assigns track IDs based on appearance
+ motion. In our domain (two similar-looking rats in a cage), it frequently
swaps IDs during close interactions because the rats look identical.

IdentityMatcher avoids this by:
- Using mask shape continuity (IoU) as the primary identity signal
- Tracking velocity to predict position during crossings
- Explicitly handling merge/split events via a state machine

---

## Hungarian Assignment

Each frame, we build a cost matrix (active_slots × valid_masks) and solve
with `scipy.optimize.linear_sum_assignment` for globally optimal assignment.

### 3-Component Soft Cost

| Component | Weight | Computation | Purpose |
|-----------|--------|-------------|---------|
| Distance | 0.4 | `min(dist / proximity_threshold, 1.0)` | Spatial continuity |
| Mask IoU | 0.4 | `1.0 - mask_iou(current, previous)` | Shape continuity |
| Area ratio | 0.2 | `min(abs(ratio - 1.0), 1.0)` | Size consistency |

- **Reject** if total_cost > `cost_threshold` (0.85)
- **Hard veto** only for 5× area change (prevents assigning a merged blob)

### Velocity Prediction

Predicted position = centroid + velocity (used for distance cost):

```
velocity = 0.7 * (current_centroid - prev_centroid) + 0.3 * prev_velocity
```

Clamped to `proximity_threshold / 2` to prevent runaway predictions.

### Swap Guard (N=2)

When 2 masks match 2 slots, compare straight vs swapped total cost:
- If swapped is cheaper → use swapped assignment
- If costs within 0.1 → prefer continuity (keep previous assignment)

---

## State Machine: SEPARATE ↔ MERGED

```
  SEPARATE ──→ MERGED ──→ SEPARATE
     ↑            │
     └────────────┘
```

### States

| State | Condition | Behavior |
|-------|-----------|----------|
| **SEPARATE** | 2 distinct masks | Normal Hungarian matching |
| **MERGED** | 1 blob where 2 expected | Freeze identity, coast with velocity |

### SEPARATE → MERGED Transition

Triggered when:
- Only 1 mask after dedup AND its area > 1.5× average single-rat area
- AND centroids of both slots are close to the merged blob

### MERGED State Behavior

1. Don't assign the merged blob to any slot
2. Coast each slot's centroid using pre-merge velocity
3. Don't update prev_masks or prev_centroids (keep pre-merge state)
4. Increment frames_merged counter
5. SAM2 centroid fallback is **suppressed** (would re-segment the same blob)
6. Contact classification is **skipped** (identity is ambiguous)

If `frames_merged > max_merged_frames` (default: 30) → force reset to SEPARATE.

### MERGED → SEPARATE (Split Resolution)

When 2 distinct masks reappear, assign IDs using pre-merge evidence:

```
cost_matrix[slot][mask] = 0.3 * distance_cost + 0.4 * mask_iou_cost + 0.3 * area_cost
```

- Distance: from predicted position (pre-merge centroid + velocity × frames_merged)
- Mask IoU: compared against pre-merge masks (shape continuity)
- Area: compared against pre-merge areas

Mask IoU is weighted highest (0.4) because shape is the most reliable
identity signal after a merge — position may have drifted.

### Pre-Merge Snapshot

At the SEPARATE → MERGED transition, we save:
- `pre_merge_centroids` — last known position per slot
- `pre_merge_velocities` — motion direction per slot
- `pre_merge_masks` — last mask per slot (for IoU at split)
- `pre_merge_areas` — last area per slot

---

## Area Tracking

Running average of single-rat mask area, updated only during SEPARATE state:

```python
alpha = min(0.1, 1.0 / n_samples)
avg_area = (1 - alpha) * avg_area + alpha * current_area
```

Used for merge detection (is this blob 1.5× a single rat?).

---

## Configuration

```yaml
identity_matcher:
  proximity_threshold: 150.0    # soft normalization distance (px)
  area_tolerance: 0.4           # used in soft area cost
  max_merged_frames: 30         # max frames in MERGED before force-reset
```

---

## Integration Points

| Component | How it interacts |
|-----------|-----------------|
| `sam2_processor.py` | Receives `interaction_state` param; skips centroid fallback during MERGED |
| `run.py` | Passes `matcher.state` to segment_frame; skips contacts during MERGED |
| `ContactTracker` | Not called during MERGED (identity is ambiguous) |
| `filter_duplicates()` | Runs before match(); unchanged |
| `resolve_overlaps()` | Runs before match(); unchanged |

---

## Known Limitations

1. **Linear velocity assumption**: During MERGED, velocity is extrapolated
   linearly. Long merges (>30 frames) may cause drift.

2. **No multi-merge**: State machine handles N=2 only. If a third entity
   appears (e.g., YOLO false positive), it's handled by the cost threshold.

3. **Resolution-dependent**: `proximity_threshold` needs tuning per video
   resolution (150px works for 640×480).

4. **Robustness fixes designed but not yet applied**: Multi-condition merge
   guard, split validation, exponential velocity decay, adaptive IoU gating.
   These are waiting for test results from the base implementation.
