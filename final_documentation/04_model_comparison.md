# 4. Segmentation Backbones Compared

Why the tracker ended up on CUTIE. Written for the methods section: what each
backbone does, how it failed here, and what the failure implies.

---

## 4.1 The task the backbone has to do

Not "segment a rat". The backbone must **keep the same label on the same animal
for the whole session**, through touching, crossing, mounting and occlusion,
between animals that are visually identical.

That framing is what separates the candidates. Segmentation quality was never the
bottleneck — every model tested produces a good mask of a rat. Identity over time
was the bottleneck.

Two properties decide it:

1. **Does the model carry object memory across frames**, or is it re-prompted from
   scratch each frame?
2. **Can two objects claim the same pixel?** A model emitting independent
   per-object masks can; one emitting a single labelled index mask cannot.

---

## 4.2 The candidates

### YOLO + a motion tracker (BoT-SORT) — `sam2_yolo`

Detection every frame; a tracker associates detections across frames using motion
and appearance.

**Failed.** Appearance re-identification is the mechanism, and the animals are
identical, so it contributes nothing. On every close interaction the IDs swapped,
and the downstream slot state stayed corrupted afterwards. This is a property of
the study animals, not a tuning problem.

### SAM2, prompted per frame — `reference`, `centroid`

SAM2 segments from a prompt. Two ways of prompting it were tried.

**With YOLO boxes as prompts (`reference`) — failed.** Detector output order is
arbitrary, so `detection[0]` may be either animal; a wrong box becomes a wrong
mask becomes a permanent swap. Overlapping boxes also made SAM2 return both
animals as one blob.

**With the previous centroid as a prompt (`centroid`) — worked, with limits.**
Each animal's own previous centroid is the positive point and the other's is a
negative point. This was the production default from March 2026 and was validated
swap-free over 3600 frames. Its limits: SAM2 has **no object memory** — it is
re-prompted from two points every frame, so nothing links frame *t* to frame
*t−1* except those coordinates; it does not scale naturally past two animals; and
pose still ran on the full frame, so keypoints could be mixed between overlapping
bodies.

`multimask_output` had to be `False`. With `True`, SAM2 returns three candidates
and the highest-scoring one was selected each frame — but that score measures
single-frame plausibility, not temporal consistency, so the mask jumped between
interpretations of the same animal.

### SAM3 — `sam3_composite`, `sam3_reset`

Open-vocabulary: the text prompt `"mouse"` yields masks with no manual
initialisation, which is a real operational gain.

**Kept, but only for bootstrapping.** As a full-session tracker it needed explicit
failure detection and reset strategies — a whole `sam3_reset` pipeline exists for
that. Needing a reset system is the symptom: the model was losing lock and had to
be re-grounded. Its actual strength, zero-shot initialisation, is what
`cutie_composite` retained.

### SAMURAI (SAM2 video predictor) — `samurai`, `samurai_sleap`, `isolated_composite`

SAM2's video mode with motion-aware memory. Better temporal behaviour than
per-frame SAM2, and `isolated_composite` paired it with the "erase the other
animal" idea that survives in the current pipeline. Contact classification was
never implemented on these, so they stayed exploratory.

### CUTIE — `cutie_composite` (selected)

A video object segmentation model built for exactly this task: propagate a
labelled mask through a video with an explicit object memory
(`mem_every=5`, `max_mem_frames=5`, long-term memory on).

**Selected**, for two structural reasons:

1. **Object memory.** It propagates a labelled mask rather than re-deriving one
   from prompts, so identity is carried by the model rather than reconstructed
   each frame by geometry. No reset machinery is needed.
2. **One id per pixel.** Its output is an index mask, so two animals cannot
   overlap. Identity is unambiguous by construction, and it extends to N animals
   at no extra cost — the shipped configs run 6.

---

## 4.3 Summary

| Backbone | Object memory | Output | Identity holds? | Verdict |
|---|---|---|---|---|
| YOLO + BoT-SORT | motion only | boxes | No — appearance re-ID is useless on identical animals | Rejected |
| SAM2, box prompts | none | per-object masks | No — arbitrary detector order | Rejected |
| SAM2, centroid prompts | none | per-object masks | Yes, for 2 animals | Superseded |
| SAM3 | limited | per-object masks | Needed reset strategies | Kept for init only |
| SAMURAI | yes | per-object masks | Promising, never completed | Exploratory |
| **CUTIE** | **yes, explicit** | **index mask (1 id/pixel)** | **Yes, by construction** | **Selected** |

**The general lesson**, and the one worth stating in the paper: for identical
animals, identity cannot be recovered by appearance and should not be
reconstructed frame by frame from geometry. It has to be *propagated* by a model
that holds object state. Everything that worked here moved in that direction;
everything that failed tried to re-derive identity each frame.

---

## 4.4 The cost of the index mask

The same property that makes CUTIE's identity unambiguous — one id per pixel —
means **mask overlap between animals is always zero**. Side-by-side detection is
gated on mask IoU, so it can never fire. See
[02 §2.6 B](02_contact_detection.md#b-sbs-can-never-fire--mask-iou-is-structurally-zero).

This is worth stating plainly because it is a design trade-off, not a bug in
CUTIE: a representation that guarantees disjoint identity cannot also express
physical overlap. Any contact rule that needs overlap must be rebuilt on a
different signal — mask *adjacency* or boundary distance, the gap between mask
contours, or a dilated-mask intersection — rather than on IoU.
