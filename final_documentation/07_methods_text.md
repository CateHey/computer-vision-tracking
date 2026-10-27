# 07 — Methods text (draft for the paper)

A paper-ready description of the pipeline as it stands on `develop` (2026-10-28), written so
that every number has a source: either a reference tool/paper or a measurement on this data.
Values in **[brackets]** must be filled from the final runs. Each paragraph can be pasted into
the Methods section and edited for tone. The parameter table at the end is for a supplement.

## Animals, arena and video

**[Lab to complete: strain, sex, age, housing, n animals per session, ethics approval.]**
Sessions were recorded from above at 960 × 960 px and 30 frames s⁻¹ in a **[50 × 50 cm]** square
arena. The arena floor spans 822 px in the image, giving **[0.061 cm px⁻¹]**; the median body
length measured by the tracker (nose to tail base, 80th percentile over the session) was 164 px,
i.e. **[10.0 cm]**, consistent with adult C57BL/6J (JAX Mouse Phenome Database, 9.7–10.1 cm).

## Animal tracking and identity

Animals were segmented and tracked with a two-stage procedure. Segment Anything 3 (SAM3,
`facebook/sam3`, Meta) detected every animal from the text prompt "mouse" and tracked them over
the first 125 frames (4.2 s). If fewer than *N* animals were found, the initialisation was
repeated with lower detection thresholds (0.4, 0.3) and from later start frames (frames 30 and
60); a session in which an animal was still missing at the end of this phase was discarded and
re-run rather than analysed with a missing animal. The masks of the last SAM3 frame seeded
CUTIE (Cheng et al., CVPR 2024), a video object segmentation model with object memory that
propagated one labelled mask per animal for the rest of the session. Identity therefore comes
from mask propagation; masks are disjoint by construction, so two animals never share a pixel
and no per-frame identity re-assignment is needed. Frames after the SAM3 phase are streamed
from the video, so memory use does not depend on session length.

## Pose

Seven keypoints per animal (nose, left ear, right ear, mid-body, tail start = body–tail
junction, tail base, tail tip) were predicted with a YOLO26 pose model (`yolo26_v11.pt`,
trained on **[n]** labelled frames of this arena; mAP **[value]**). To avoid cross-animal
keypoint confusion in close contact, pose was estimated on a per-animal composite in which
every other animal was replaced by a background plate (per-pixel median of **[50]** frames),
and keypoints were assigned to the animal by their mask. Keypoints with confidence < 0.3 were
ignored; when an animal was not detected in a frame, its keypoints were carried over from the
previous frame shifted by the mask centroid displacement and the frame was flagged
(`stale_keypoints`); a nose keypoint falling outside its own mask was flagged
(`nose_off_mask`). Both flags are reported as session quality metrics.

## Social contact classification

Contacts were scored for every pair of animals independently, following the convention of
Live Mouse Tracker (de Chaumont et al., 2019). Distances were normalised by the pair's mean
body length (BL) and converted to cm with the arena scale. For each frame and pair, five
behaviours received a continuous score in [0, 1] built from smooth ramps over the geometric
quantities, and the behaviour with the highest score above its activation threshold (0.50;
0.35 for following) was assigned to the frame, with the fixed priority nose-to-nose,
nose-to-anogenital, following, side-by-side, nose-to-body. Distance criteria had hysteresis
(enter / leave thresholds) to avoid frame-to-frame flicker.

- **Nose-to-nose (N2N).** Nose points within 0.30 BL (**[3.0 cm]**; leave at 0.45 BL), weighted
  by how directly the two heads face each other. Live Mouse Tracker's criterion is 26 mm
  between nose points.
- **Nose-to-anogenital (N2AG).** The nose of the investigating animal within 0.30 BL of the
  other's tail start, the investigator oriented towards it, and the nose projecting onto the
  rear 20 % of the other's body axis (nose → mid-body → tail start), which approximates the
  genital region used by MARS (midway between hips and tail base; Segalin et al., 2021). The
  investigator is the animal whose nose is at the other's rear.
- **Nose-to-body (N2B).** The nose within 6 px (**[0.4 cm]**) of the other animal's mask
  (masks touching, as in Live Mouse Tracker's contact definition), outside the face and
  anogenital regions, with the investigator oriented towards the other animal.
- **Side-by-side (SBS).** Masks sharing a border (≥ 0.35 BL of shared boundary to enter,
  0.20 BL to leave), centroids within 1 BL, body axes parallel (|cos| ≥ 0.7, either direction)
  and both animals slow (< 0.75 BL s⁻¹); score = geometric mean of the four factors. Live Mouse
  Tracker's side-by-side uses 52 mm between flanks and the same orientation.
- **Following (FOL).** A non-contact behaviour: the follower's nose within 0.4 BL (leave at
  0.6 BL) of the path of the followed animal's rear over the previous 12 frames, the follower
  oriented towards the followed animal and moving at ≥ 0.15 BL s⁻¹, and no body contact
  (all nose–body distances > 0.5 BL, no shared mask border). Live Mouse Tracker: both animals
  moving, heading difference < 45°, centre distance < 2 BL.

Each frame also records the runner-up behaviour (score ≥ 0.40) and a distance dynamic
(closing, stable, separating; |Δd| ≥ 0.10 BL s⁻¹).

## From frames to events

Per pair, frame labels were smoothed by a 5-frame majority vote, gaps of ≤ 3 frames between
runs of the same label were bridged, and runs shorter than 0.3 s (0.4 s for following,
matching the tracker's own bout minimum and the 400 ms default of A-SOiD) were discarded.
Each remaining run is one event with its start, end, duration, peak score and investigator
(majority of the frame-level roles; none for the symmetric N2N and SBS). Events are counted
per pair; per-animal summaries count the events an animal takes part in and its contact time
as a union over its pairs (a frame counts once even when the animal touches two others), and
the time an animal spends in contact with two or more others is reported separately, as Live
Mouse Tracker's group2/group3 configurations. A pair-by-pair contact matrix summarises the
dyadic structure of each session.

## Validation

For each session, **[15]** events per behaviour were sampled, stratified by duration tertile,
and rendered as slow-motion clips with the two animals highlighted; events overlapping in time
were shown in one clip with every pair marked. **[Two]** raters blind to the automatic label
scored each event as real, flicker, wrong type (with the correct type), tracking error or
unsure. Precision per behaviour is real / (real + flicker + wrong type + tracking error) with a
Wilson 95 % interval; inter-rater agreement is Cohen's κ **[value]**. To estimate misses,
**[n]** no-contact windows of 3 s per session, biased towards moments when animals were within
1 BL, were reviewed for unlabelled contacts (miss rate **[value]**). Session quality was
reported as the fraction of frames with detected pose, carried-over keypoints and
nose-off-mask flags, and the SAM3 → CUTIE handoff count. The sensitivity of event counts to the
activation threshold (0.40–0.70) and to the minimum event duration (0.2–1.0 s) was computed
offline from the per-frame scores **[and is reported in Supplementary Table S?]**.

## Software and reproducibility

All code is open source **[repository URL]**, commit **[hash]**. Each run writes the exact
configuration, the versions of PyTorch **[2.14]**, Transformers **[5.20]**, Ultralytics
**[8.4]**, the CUTIE commit and the MD5 of the pose weights (`provenance.json`). Runs were
executed on the UQ Bunya HPC (one NVIDIA A100 per session).

## Supplementary parameter table

| Parameter | Value | Unit / meaning | Source or justification |
|-----------|-------|----------------|--------------------------|
| SAM3 phase | 125 frames (4.2 s) | initial detection + tracking | enough for identities to settle; retried at 0.4/0.3 and from frames 30/60 |
| Keypoint confidence | 0.3 (contacts), 0.5 (body length) | YOLO confidence | code defaults; sensitivity not assessed |
| Body length | 80th percentile of nose–tail start | px | robust to crouching; LMT normalises by mean body length |
| Contact enter / leave | 0.30 / 0.45 BL (3.0 / 4.5 cm) | nose–nose, nose–tail start | LMT: 26 mm nose–nose and nose–tail base |
| Proximity | 1.0 BL | centroid distance | zone label only |
| Anogenital region | rear 20 % of the body axis (split 0.8, ramp 0.7–0.9) | | MARS genital boundary midway hips–tail base |
| N2B touch radius | 6 px (0.4 cm) | nose to other mask | LMT "masks touching"; keypoint error scale |
| SBS border enter / leave | 0.35 / 0.20 BL | shared mask boundary | replaces mask IoU (always 0 with disjoint masks) |
| SBS speed, alignment | < 0.75 BL s⁻¹, |cos| ≥ 0.7 | | LMT: orientation only (same way / opposite) |
| FOL radius, speed, alignment | 0.4–0.6 BL, ≥ 0.15 BL s⁻¹, cos ≥ 0.6, path 12 frames | | LMT: 2.5 cm corridor, ≥ 10 cm s⁻¹, 45° |
| Activation / rare / secondary | 0.50 / 0.35 / 0.40 | score thresholds | sensitivity 0.40–0.70 reported |
| Smoothing, gap, minimum | 5 frames, 3 frames, 0.3 s (FOL 0.4 s) | | A-SOiD 200–400 ms; LMT none / 7 frames for follow; MARS HMM |
| Tracker bout minimum | 6 frames (FOL 12), gap 3 | diagnostic bouts only | |
| Dynamics | |Δd| ≥ 0.10 BL s⁻¹, mover ≥ 60 % | | code defaults |
