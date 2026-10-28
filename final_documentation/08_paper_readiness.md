# 08 — Paper readiness checklist

What a methods reviewer will ask for, where it is covered, and what still has to be done.
Tick items off here. Last update: 2026-10-28.

## Covered by the pipeline and its documents

| Requirement | Where |
|-------------|-------|
| Exact definition of every behaviour, with thresholds and code references | [05](05_contact_computation.md) §3–§5, [07](07_methods_text.md) |
| Precedent for the thresholds (Live Mouse Tracker, MARS, A-SOiD, SimBA) | [06](06_open_issues_research.md) |
| Calibration to cm and body-length normalisation | 06 §A1, `Parameters` sheet (`calibration` rows), `event_log.txt` header |
| How frames become events; per-pair, per-animal and group counts | 05 §6–§8, `Events`, `PerAnimal`, `PairMatrix` sheets |
| Precision per behaviour with confidence intervals, blinded review clips | `validate_contacts.py sample` / `score`, 05 §9 |
| Recall proxy (no-contact windows), inter-rater κ | `validate_contacts.py --nc-per-run`, `score --compare` (05 §9) |
| Sensitivity of counts to thresholds | `scripts/sensitivity_analysis.py` (03, last section) |
| Session quality metrics (handoff, detection, carry-over, nose-off-mask) | `logs/run.log`, `Parameters` sheet (`quality_flags`) |
| Reproducibility: versions, commits, weights hash per run | `provenance.json` in every run |
| Fail-fast on incomplete tracking | `sam3.require_all_animals` (B6 in OPEN_ISSUES) |
| Memory independent of session length | frame streaming (D4) |

## Still to do before submission

- [ ] **Measure the arena** (physical side length) and set `contacts.cm_per_px`; confirm the
      50 cm assumption behind the 0.061 cm px⁻¹ figure.
- [ ] **Decide the wording for contact distance** (keep 0.30 BL ≈ 3.0 cm, or 0.26 BL to match
      LMT's 26 mm) and for multi-animal counting (per pair, per animal, group configurations) —
      OPEN_ISSUES A1, A2.
- [ ] **Validation run of the new defaults** on the 6-mouse clip (N2B radius 6 px, anogenital
      split 0.8, FOL 0.4 s) against the baseline run `2026-10-23_191936`; review the N2B / N2AG
      clips; keep or revert (OPEN_ISSUES A4).
      → Run done 2026-10-28 (`2026-10-28_091052`): tracking identical, N2B appears (2 events),
      N2AG fragments (7 → 10 events, 7.6 → 7.0 s). Count investigation bouts (06 §A4); clip
      review of the 21 events still pending.
- [ ] **Confirm on Bunya** that a streamed run equals a preloaded run on one short clip
      (`scan.preload_all_frames=true` vs `false`; identical `Events` sheet expected — the
      frame-level equality was verified locally on both video codecs).
- [ ] **Full-length sessions** on Bunya (10:50 and 20:00 videos), `--time` raised.
- [ ] **Review protocol**: two raters, ≥ 15 events per behaviour per session plus the
      no-contact windows; report precision ± CI, miss rate, κ.
- [ ] **Sensitivity tables** from the final runs (one command per run).
- [ ] **Provenance**: record the repository commit hash and weights MD5 in the paper.
- [ ] **YOLO training details** for the pose paragraph: number of labelled frames, validation
      mAP, keypoint OKS (from `roboflow/TRAINING_GUIDE.md` / the training run).
- [ ] **Figures**: pair contact matrix (heat map), per-animal rates per minute, precision per
      behaviour with CIs, sensitivity curves.
- [ ] **Operational**: rotate the Hugging Face token (OPEN_ISSUES D1); delete the stray copies
      with `git clean -f -d` (D2); decide the "rat" → "mouse" renaming (D5).

## Known limitations to state in the paper

- Pose is least reliable at the moment of contact (composite erasure runs through the contact
  point); hence the blinded clip review is the ground truth for the reported precision.
- Pairs are scored independently; an animal in two simultaneous contacts contributes to two
  dyadic events (reported explicitly as 2+ contact time), as in Live Mouse Tracker.
- SAM3 initialisation is not deterministic; sessions are retried and discarded if an animal is
  missing, which is reported as a session count.
- The first few frames use a provisional body length until three observations exist.
- Side-by-side events are sustained by the hysteresis latch (scores ≈ 0.5): they disappear if the
  activation threshold is raised to 0.55 (sensitivity analysis, 06). State it or adjust the latch.
