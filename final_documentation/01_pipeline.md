# 1. The Tracking Pipeline

The selected pipeline is **`cutie_composite`**
(`src/pipelines/cutie_composite/run.py` on `origin/develop`).

> **Branch note.** The current work lives on `origin/develop`, not `main`. `main`
> does not contain this pipeline at all. The in-repo `CLAUDE.md` on both branches
> still names the older *centroid* pipeline as the production default — that has
> been out of date since this pipeline landed. Read the code, not the docs.

---

## 1.1 The problem it solves

Rats in an arena are visually identical. Any tracker that re-detects them
independently each frame cannot know which detection is which — detector output
order is arbitrary. When animals touch, cross, or occlude one another the
identities swap, and once swapped they never recover. Every measure of *who did
what to whom* is wrong from that point on.

There is a second problem specific to pose estimation. When two animals overlap,
a pose model run on the raw frame mixes their body parts: a nose from one animal
gets attached to the other's skeleton. Masking alone does not fix this, because
the pose model still sees both bodies.

`cutie_composite` addresses both: a video object segmentation model owns identity
end to end, and the pose model is shown a frame in which only one animal is
present.

---

## 1.2 Architecture

Two phases, sharing one per-frame processing routine.

```
PHASE 1 — SAM3, frames [0, sam3_phase_frames)         default 125 frames
  SAM3 open-vocabulary video segmentation, text prompt "mouse"
  → per-frame masks
  → identities assigned by centroid proximity to the previous frame

HANDOFF at frame sam3_phase_frames - 1
  the last good per-slot masks become a single index mask
  → CUTIE is seeded with it

PHASE 2 — CUTIE, frames [sam3_phase_frames, N)        the rest of the video
  CUTIE propagates the index mask forward, continuously
  → one integer id per pixel, so masks are disjoint by construction

PER FRAME (both phases)
  1. mask carry-over      missing mask ≤ 5 frames → shift the previous one by velocity
  2. composite            for each animal, paint every OTHER animal out using a
                          background plate, leaving the arena and this animal intact
  3. YOLO pose            run on that composite → keypoints already in frame coords
  4. pick detection       the detection whose box overlaps this animal's own mask
  5. keypoint carry-over  fill a missing pose from the previous frame
  6. contacts             ContactTrackerV2.update(...)
  7. render               overlay video + one isolated video per animal
```

### Why the composite step matters

For animal *i*, `erase_other_rat` replaces the pixels of every other animal with
the corresponding pixels of a background plate (a temporal median of the video),
dilated by 15 px and feathered by 5 px. Animal *i* and the arena are untouched,
so the keypoints YOLO returns are already in original frame coordinates — no
transform back, no cropping artefacts. The pose model simply never sees a second
animal.

The result is written out as one video per animal, which makes the step
auditable by eye.

### Why identity holds

CUTIE is a video object segmentation model with an explicit object memory
(`mem_every=5`, `max_mem_frames=5`, long-term memory on). It propagates a
labelled mask rather than re-detecting each frame, and it emits **one integer id
per pixel**. Two animals therefore cannot share a pixel, and there is no
assignment step to get wrong after the handoff.

SAM3 is used only to bootstrap: it turns the text prompt "mouse" into initial
masks without any manual box drawing, which is what CUTIE needs to start.

### Number of animals

`detection.max_animals` — the shipped configs use **6**. This is not a two-animal
pipeline; contacts are computed for every pair.

---

## 1.3 Why this design — what was tried before

Every pipeline below is still in the repository. This is the honest ordering of
what was attempted, in roughly chronological order.

| Pipeline | Mask / identity source | Why it was left behind |
|----------|------------------------|------------------------|
| `sam2_yolo` | YOLO every frame + BoT-SORT | BoT-SORT re-identifies by appearance; identical animals swap on every close interaction and the slot state stays corrupted afterwards. |
| `reference` | YOLO every frame + Hungarian matching + SEPARATE/MERGED state machine | Overlapping YOLO boxes made SAM2 segment both animals as one blob; the MERGED timeout mis-assigned identity on separation; intermittent misses dropped an animal for 10–50 frames. |
| `sam2_video` | SAM2 temporal memory | Slow; not pursued. |
| `centroid` | SAM2 prompted with the previous centroid (+ the other animal's centroid as a negative point) | **Was the production default from March 2026.** Solved the swap problem for two animals, but SAM2 is re-prompted from scratch every frame with two points — it has no object memory — and pose still ran on the full frame. |
| `samurai`, `samurai_sleap` | SAM2 video predictor (SAMURAI); SLEAP for pose | Exploratory; no contact classification implemented. |
| `isolated_composite` | SAMURAI + the "erase the other animal" composite | Introduced the composite idea that `cutie_composite` kept. |
| `sam3_composite`, `sam3_reset` | SAM3 video + composites, with reset strategies on failure | SAM3 alone needed explicit failure detection and resets to stay locked on. |
| **`cutie_composite`** | **SAM3 to bootstrap → CUTIE to propagate, + composites** | **Selected.** CUTIE's object memory removes the per-frame re-prompting and the reset machinery; identity comes from a single labelled mask propagated forward. |

The through-line: **identity moved from "match detections each frame" to "propagate
a labelled mask"**, and pose moved from "run on the whole frame and untangle" to
"run on a frame containing one animal".

The `safety.py` module in `cutie_composite/` implements checkpointing, mask
protection and fusion, but the shipped `run.py` is explicitly the *simplified*
two-phase version — its own docstring says "no safety nets". The safety block in
the config is therefore inert for this entry point.

---

## 1.4 Models and configuration

| | Value |
|---|---|
| Bootstrap segmentation | SAM3, text prompt `"mouse"`, score threshold 0.5, first 125 frames |
| Propagation | CUTIE — `mem_every=5`, `max_mem_frames=5`, long-term memory on, `max_internal_size` 480 local / 720 HPC |
| Pose | YOLO `models/yolo/yolo26_v11.pt`, confidence 0.25 |
| Keypoints | **5**: `nose, left_ear, right_ear, mid_body, tail_base` |
| Animals | 6 |
| Composite | erase dilate 15 px, feather 5 px, mask dilate for pick 7 px, mask carry ≤ 5 frames |
| Background | temporal median over 30 sampled frames, cached |
| Configs | `configs/local_cutie_composite.yaml`, `configs/hpc_cutie_composite.yaml` |

Note the keypoint count: earlier pipelines used a **7**-point model
(`tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear`). The
current model has 5 and drops `tail_tip` and `tail_start`. **This has a direct
consequence for contact classification — see §2.6.**

---

## 1.5 Scale and known constraints

- **The whole video is loaded into memory.** `extract_frames_to_memory` holds
  every frame as an array before Phase 1 begins. Memory grows linearly with
  duration and resolution; the local config caps the run at 900 frames.
- **No chunking.** Unlike `centroid`, this entry point takes `--start-frame` and
  `--end-frame` but has no `--chunk-id`, so `scripts/run_parallel.sh` and
  `merge_chunks.py` do not apply to it. A long session is one process.
- **A fixed handoff frame.** CUTIE is seeded once, at frame 124, from whatever
  SAM3 produced. If SAM3 has not locked onto every animal by then, the missing
  ones are never recovered — the run log reports the handoff count
  (`Phase 1 complete. Handoff: n/N rats`), and it is worth checking.
- **No identity re-check after handoff.** With `safety` disabled there is no
  checkpoint or fusion, so a CUTIE tracking failure persists to the end of the
  video.
