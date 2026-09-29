# 2. Contact Detection and Classification

Implemented in `src/common/contacts_v2.py` (`ContactTrackerV2`) on
`origin/develop`, with temporal cleaning in
[scripts/postprocess_contacts_simple.py](../scripts/postprocess_contacts_simple.py).

`contacts_v2` is a rewrite, not a revision of the older `contacts.py`. The
differences that matter:

| | `contacts.py` (v1, centroid) | `contacts_v2.py` (current) |
|---|---|---|
| Types | 6, including T2T | **5** — T2T removed |
| Decision | first rule to match wins, fixed priority | **continuous scores 0–1, argmax over a threshold** |
| Stability | none per frame | **Schmitt triggers** (separate enter / exit thresholds) |
| Keypoints | by **index** (hardcoded 4 = nose) | by **name** |
| Grouping | — | **families**: investigative / affiliative / non-contact |
| Motion | — | **dynamics**: closing / stable / separating, plus who moved |
| Secondary label | — | second-best type recorded alongside the winner |
| Non-contact output | — | `dynamics_no_contact.csv` — approach/avoid without contact |

---

## 2.1 Per-frame measurements

For every frame and every pair:

| Quantity | How |
|----------|-----|
| **Body length (BL)** | Per animal, from the pose. The pair's reference is the **mean** of the two. |
| **Velocity** | Mask-centroid displacement, expressed in **body lengths per second** (`bls`). |
| **Orientation** | Body axis; `orientation_alignment_cos` between the two. |
| **Distances** | nose–nose, nose–rear (both directions), tail–tail, centroid–centroid — all in body lengths. |
| **Mask IoU** | Intersection over union of the two masks. |
| **Head centre** | Midpoint of the two ears, or whichever ear is confident. |

All distances are in body lengths, so the rules are invariant to camera height,
resolution and animal size.

### Anatomy — a naming trap

`contacts_v2` defines the rear (anogenital) reference as **`tail_start`**, the
body–tail junction, and treats **`tail_base`** as a point further down the tail:

```python
KP_TAIL_START = "tail_start"   # body-tail junction = rear / anogenital
KP_TAIL_BASE  = "tail_base"    # mid-tail point (NOT the rear)
```

The older `contacts.py` used `tail_base` as the anogenital reference. **The two
modules mean different things by the same word**, which matters when comparing
results across pipelines — and, as §2.6 shows, it has broken N2AG in practice.

### Zones

From centroid distance in body lengths, with hysteresis:

| Zone | Threshold |
|------|-----------|
| `contact` | enters below 0.30 BL, leaves above 0.45 BL |
| `proximity` | below 1.0 BL |
| `independent` | beyond that |

---

## 2.2 The five contact types

| Code | Name | Family | Depends on |
|------|------|--------|------------|
| **N2N** | Nose-to-nose | investigative | nose of both |
| **N2AG** | Nose-to-anogenital | investigative | nose + **`tail_start`** of the other |
| **N2B** | Nose-to-body | investigative | nose inside the other's **mask**, gated by gaze alignment |
| **SBS** | Side-by-side | affiliative | **mask IoU** + parallel orientation + low speed |
| **FOL** | Following | **non-contact** | rear-path pursuit + aligned velocity + both moving |

FOL sits in a *non-contact* family: following is a social behaviour but not
physical contact, and grouping it as such keeps "contact time" honest.

T2T (tail-to-tail) existed in v1 and was removed in v2.

---

## 2.3 Scoring instead of priority

v1 tested rules in a fixed order and the first match won. v2 computes a
**continuous score in [0, 1] for every type**, then takes the argmax subject to a
threshold:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `activation_threshold` | 0.50 | A type must reach this to be selected |
| `activation_threshold_rare` | 0.35 | Lower bar, applied only to **FOL** |
| `secondary_threshold` | 0.40 | The runner-up is recorded if it reaches this |

Scores are built from smooth ramps and trapezoids over the geometric quantities,
combined as a **geometric mean** rather than a raw product, so one weak factor
degrades a score instead of zeroing it. Each type also has a **Schmitt trigger**
with separate enter and exit thresholds, so a borderline frame does not flicker
between labels.

Two explicit guards remain:
- **N2B is suppressed** when SBS scores ≥ 0.5, so a side-by-side posture is not
  relabelled as nose-to-body.
- **N2B is attenuated** (×0.7) when a more specific contact is near its threshold.

Recording the runner-up (`secondary_type`, `secondary_score`) is a genuine
improvement for analysis: a frame scoring 0.52 N2N and 0.49 N2AG is visible as
ambiguous rather than silently reported as clean N2N.

---

## 2.4 Dynamics — the new non-contact channel

Independently of contact type, each frame gets a **dynamics** label from the rate
of change of centroid distance: `closing`, `stable`, or `separating`. It also
records the **mover** (whose velocity along the line joining the animals explains
the change) and, during contact, a **reciprocity** judgement: separating while
the *receiver* is the mover reads as "repels"; stable or closing reads as
"accepts".

Approach/avoid episodes that never reach contact are written to a separate table,
`dynamics_no_contact.csv` — which is exactly the "Hoja B" you have been referring
to. This captures avoidance, which a contact-only ethogram misses entirely.

---

## 2.5 From frames to events

**Bouts (in `contacts_v2`).** Consecutive same-type frames group into bouts, with
a maximum gap of 3 frames and a per-type minimum length — 6 frames (200 ms) for
N2N, N2AG, SBS and N2B, and 12 frames (400 ms) for FOL. Per-type minimums are an
improvement on v1's single global threshold.

**Post-processing.** `postprocess_contacts_simple.py` then applies three temporal
rules to the per-frame labels — majority-vote smoothing, gap bridging, and a
minimum duration — and emits the final event table with human-readable
timestamps. See [03_outputs_and_reports.md](03_outputs_and_reports.md) for the
parameter values actually used, which are **not** the ones in the YAML.

Threshold rationale (literature review in
`docs/contacts/threshold_research.md`): A-SOiD's CalMS21-derived 400 ms default
with a 200 ms floor is the closest reference point; manual scoring of rat social
investigation reports bouts of 0.5–5 s; contacts under ~300 ms are typically
incidental. v2's 200 ms per-type floor sits at that floor, with FOL held to
400 ms because brief same-direction movement is unavoidable in a small arena.

---

## 2.6 Three defects that suppress contact types

These were found by running the choreographed demo
([03 §3.5](03_outputs_and_reports.md)) through the real `ContactTrackerV2`. All
six behaviours were scripted; only three were ever detected.

### A. N2AG can never fire — the rear keypoint does not exist

`contacts_v2` reads the rear from `tail_start`:

```python
rear_i = get_keypoint(det_i, KP_TAIL_START, self.min_kp_conf)   # rear of i
```

`get_keypoint` is a strict name match with **no alias and no fallback**. But every
v2 config declares only five keypoints:

```yaml
keypoint_names: [nose, left_ear, right_ear, mid_body, tail_base]
```

There is no `tail_start`. So `rear_i` and `rear_j` are always `None`, the
nose-to-rear distances stay at their "missing" sentinel, and the N2AG score is
always 0. This affects `cutie_composite`, `sam3_composite` and `sam3_reset`
alike.

FOL degrades rather than dies: it falls back to centroid paths and sets the
`fol_used_centroid` flag — good design, and the flag makes it auditable.

**Most likely cause:** `contacts_v2` was written against the 7-point model, where
`tail_start` was the body–tail junction. The 5-point model that replaced it has
no such point, and its `tail_base` is probably that same junction under a
different name. The fix is a one-line rename or an alias — but it must be
verified against the model's actual training labels, not assumed.

### B. SBS can never fire — mask IoU is structurally zero

SBS is gated on mask overlap, with a Schmitt trigger entering at 0.05:

```python
active = self.trig_sbs.update(mask_iou)
iou_score = ramp_up_score(mask_iou, self.sbs_iou_exit, self.sbs_iou_enter * 3)
```

But CUTIE emits **one integer id per pixel**, and the masks are split from it:

```python
m = (id_mask == obj_id)
```

Two masks can never share a pixel, so `mask_iou` is identically 0. With
`iou_score` floored at 1e-6 inside the geometric mean, the SBS score cannot
exceed **exp(ln(1e-6)/4) ≈ 0.032**, far below the 0.50 activation threshold. The
trigger never activates either.

The same holds for the older `centroid` pipeline, which calls `resolve_overlaps`
to make its masks disjoint *before* classifying.

The existing comment `Fix BUG-3: geometric mean ... so it doesn't collapse to 0`
shows the symptom was noticed; the root cause — disjoint masks — was not.

### C. Individual metrics never generate

`individual_metrics.py` still references `ContactType.T2T`, which v2 removed, so
it raises `AttributeError` and is caught by a `try/except` that logs a warning
and continues. `individual_summary.json` is never written.

### Measured effect

Same 120 s choreography, same code, two configurations:

| Type | Scripted | **As shipped** | With A + B fixed |
|------|----------|----------------|------------------|
| N2N | ✓ | 211 frames | 209 frames |
| N2AG | ✓ | **0** | 309 frames |
| SBS | ✓ | **0** | 798 frames |
| FOL | ✓ | 416 frames | 21 frames |
| N2B | ✓ | 134 frames | 6 frames |

Two of the five types are unreachable. Worse than that: their frames do not go
unlabelled, they are **absorbed by other types**. The scripted side-by-side and
following segments were reported as N2B and FOL as shipped, and were reclaimed by
SBS and N2AG once the defects were addressed. A behavioural result computed from
the current output would be not merely incomplete but **mis-attributed**.

**Nothing here should be reported from the current pipeline output until A and B
are resolved.**

---

## 2.7 Other limitations

- **Rule-based, not learned.** Rearing, grooming and mounting are not
  distinguished; they fall into N2B or go unlabelled.
- **2D overhead view.** One animal on top of another reads as high mask overlap —
  which, given defect B, is currently not visible at all.
- **`none` versus empty.** `contacts_v2` writes the string `"none"` for no
  contact, while the post-processor expects an empty cell and converts it to
  `NC`. Both labels end up in the same output column. Cosmetic, but it inflates
  the event count and must be normalised before analysis (the HTML viewer does
  this).
- **No ground truth.** Thresholds are justified from the literature and by visual
  inspection. No frame-level agreement against manual scoring has been computed.
  That remains the main gap before publication.
