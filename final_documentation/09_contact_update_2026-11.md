# 09 — Contact detection, November 2026: from the October state to now

For someone who knew the system as it was at the end of October 2026 (run
`2026-10-28_091052_cutie_composite`, the 6-mouse 10 s clip). This page says what was wrong,
what the system does now and why, and how much it improved on that clip. It describes only the
final state; the step-by-step history is in `docs/changes/observation_log.md` (Entries 002–005).

---

## 1. Where we started

The pipeline already scored five contact types per pair and frame (N2N nose-to-nose, N2AG
nose-to-anogenital, N2B nose-to-body, SBS side-by-side, FOL following) and cleaned them into
`Events` (5-frame majority vote, gaps ≤ 3 frames bridged, runs < 0.3 s dropped; FOL < 0.4 s).

The clip was annotated by hand, frame by frame (35 firm annotations; 2 tentative ones left out)
and compared with the program. The program found **when** and **between whom** well, but:

| Problem | Cause |
|---------|-------|
| A nose on the other mouse's **tail** counted as contact (R1–R4 243–252 was reported as nose-to-anogenital; R4–R5 184–194 as side-by-side) | Touch was tested on the full mask, tail included |
| Nose-to-anogenital fired on distance alone | No evidence of touch was required |
| Following was cancelled whenever the bodies were < 0.5 BL apart, even without touching | Distance used as a stand-in for touch |
| Nose-to-body was cancelled whenever side-by-side was strong | A fixed guard in favour of SBS |
| The nose keypoint sits ~5 px inside the head, and the touch window was a fixed 6 px | Touch measured from the wrong point with a size that ignores the animal |
| ~30 % of the annotated interactions were shorter than 0.3 s and vanished from every sheet | The minimum duration of `Events` |
| Look-alike types (heads side by side vs side-by-side; anogenital vs body sniffing) were confused, with no way to report them together | One flat label per event |

---

## 2. What the system does now, and why

All rule changes are behind one switch, `contact_rules_v3: true` in
`configs/local_cutie_composite.yaml` and `hpc_cutie_composite.yaml` (`false` reproduces the
October rules). Details and formulas: [02 §2.8](02_contact_detection.md#28-contact-rules-v3-and-groups-2026-11),
[05 §4.6](05_contact_computation.md#46-contact-rules-v3-contact_rules_v3-true).

**How touch is measured**

| Now | Why |
|-----|-----|
| Every touch test uses a **body mask without tail** (thin parts removed with a 0.10 BL opening). The video still shows the full masks | A tail touch is not a social contact |
| The nose is the **tip of the snout** taken from the mouse's own mask, not the keypoint | The keypoint lies inside the head |
| A **head circle** of 0.08 BL (~13 px) around the tip touching the other's body = touch | Scales with the animal; chosen against the manual annotation: it finds 314 of 409 annotated nose-contact frames with 52 false, where the old 6 px rule found 268 with 82 false |

**Per type**

| Type | Now | Why |
|------|-----|-----|
| N2AG | Nose ≤ 0.30 BL from the rear **and** evidence of touch (head circle on the body, ≤ 0.20 BL, or staying slow there) | Proximity alone over-counted |
| N2B | Head circle on the other's body; competes with SBS instead of being cancelled; records the **zone** touched (head / torso / rear). A nose on the tail only is a flag (`nose_on_tail`), not contact | Flank sniffing was lost |
| SBS | Shared border measured without tails | Tails touching are not side-by-side |
| FOL | Cancelled only by a real touch | A follower walking close behind is still following |
| N2N | Heads side by side keep most of the score; records **frontal / perpendicular / lateral** | All three are nose-to-nose |

**Confidence.** Each frame and event says how sure the label is: `secondary_type` (the type
that almost won), `ambiguous` (the two were within 0.10) and `low_confidence` (the bodies
touch but no type reached its threshold, so the best one ≥ 0.35 was used).

**Short interactions are kept.** Interactions of 3–8 frames that `Events` drops are listed in
the `Brief` sheet. `Events` itself is unchanged, so its counts stay comparable.

---

## 3. Three levels: group > type > subtype

Each event now carries three labels, from the safest to the most specific:

| Group | Name | Type | Subtype |
|-------|------|------|---------|
| FTF | Face-to-face | N2N | frontal, perpendicular |
| S2S | Side-to-side | N2N / SBS | lateral / — |
| AGS | Anogenital sniffing | N2AG | classic (the other mouse still) |
| BDS | Body sniffing | N2B | head, torso, rear |
| CHS | Chase | N2AG / FOL | following (both walking) / — |
| NC | No contact | NC | — |

- **Group** joins the types the program confuses with each other. On the clip, 19 of the 20
  frames annotated as side-by-side that the program called nose-to-nose were *lateral*
  nose-to-nose, i.e. the same group (S2S). Each type + subtype belongs to one group only, so
  totals by group never count a frame twice.
- **Type** is the five types, unchanged.
- **Subtype** is the finest detail and the least checked: the manual annotation has no
  subtypes, so they are not validated yet (the clip review now asks for them, §5).

`Following` (diagnostics) is the CHS group seen as episodes: a mouse that follows, sniffs the
rear while walking and follows again is one chase episode (`sequence`:
`FOL 10-39 → N2AG 40-69`). `Similarity` (diagnostics) shows how often each type has each other
type as its runner-up in that video.

---

## 4. Result on the 10 s clip

Against the 35 firm manual annotations. The "now" column comes from re-running the current
code on masks and keypoints rebuilt from the run's overlay videos; with the October rules that
rebuild reproduces the original `results.xlsx` in 98.8 % of the pair-frames.

**Is what `Events` reports right?**

| | October | Now |
|---|---|---|
| Contact events in `Events` | 21 | 25 |
| Match an annotated interaction, same type | 16 (76 %) | **21 (84 %)** |
| Match an annotated interaction, other type | 3 | 3 |
| No annotated interaction | 2 — both are the tail cases the annotator rejected (R1–R4 243–252, R4–R5 184–194) | 1 — R2→R5 following 176–187, not annotated, not checked |

Both errors the annotator had marked are gone. Every event now corresponds to an annotated
interaction except that one unchecked following. The 3 of another type: R2–R5 17–28 (N2N lateral
for an annotated side-by-side, same group S2S), R3–R5 102–110 (body sniffing inside an annotated
anogenital sniffing), R2–R3 176–195 (N2N for an annotated body sniffing).

**How much of what was annotated is found?**

| Of the 35 annotations | October | Now |
|---|---|---|
| Found in `Events` | 23 (16 with the same type) | **26 (19 with the same type)** |
| Found in `Events` or `Brief` | — (no `Brief`) | **34 (30 with the same type)** |
| Not found anywhere | 12 | **1** (R1–R5 nose-to-body, frames 0–2: the first frames of the video, before the program has any history) |

Overall frame-by-frame agreement (Cohen's κ, six classes) went from 0.58 to 0.65.

---

## 5. Where to look

`results.xlsx` (reported, validated by the clip review):
- `Events`: `group`, `group_label`, `contact_type`, `contact_label`, `subtype`, then frames,
  investigator, `secondary_type`, `ambiguous`, `low_confidence`.
- `PerFrame`: `real_group` before `real_type`; per-frame subtypes and flags
  (`nose_on_tail`, `head_touch_ij/_ji`, `tail_contact_bl`, `suspect_mask_i/_j`).

`diagnostics.xlsx` (not validated, not for reporting): `Brief` (filter `near_event` empty for
isolated interactions; filled ones are pieces next to an event, mostly type flicker),
`Following`, `Groups`, `Similarity`, `Bouts`, `Dynamics` (now with `subkind`: search,
mutual approach, withdrawal, flight, mutual separation), `Individual`.

`validation/review.xlsx` (clip review): rows show group, type and subtype; the reviewer also
fills `subtype_ok`, and `score` reports `subtype_accuracy`.

---

## 6. Limits and next steps

- One 10 s clip, one annotator, rebuilt inputs. Re-run the clip on the HPC with the current
  code and annotate more video before reporting.
- Subtypes are not validated (OPEN_ISSUES B9): fill `subtype_ok` in the next clip review.
- The walking threshold that separates classic from following N2AG (0.5 BL/s) is set by eye
  (B8).
- Lateral nose-to-nose can take time from side-by-side (B7); both are S2S, so group totals are
  not affected.
- Outputs still say "rat" in places; the animals are mice (D5, not changed yet).
