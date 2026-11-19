"""cutie_composite area guard, SAM3 recovery and tracks file (pure logic; CUTIE and SAM3 are fakes)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.cutie_composite.area_guard import AreaGuard, GuardConfig  # noqa: E402
from src.pipelines.cutie_composite.recovery import (  # noqa: E402
    MaskRecovery, RecoveryConfig, compose_index_mask, iou, mask_centroid,
)
from src.pipelines.cutie_composite.tracks import (  # noqa: E402
    TracksWriter, plot_area, read_tracks, rle_decode, rle_encode,
)

H = W = 480
FPS = 30.0
FLOOR, RAT = 200, 60


def rat(cx, cy, angle=0.0):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (60, 22), angle, 0, 360, 1, -1)
    return m.astype(bool)


def scene(t):
    """Ground truth at frame t: A and B walk side by side, C stays far away."""
    a = rat(160 + 0.5 * t, 200)
    b = rat(160 + 0.5 * t, 247)
    c = rat(380, 400, 90)
    return [a, b, c]


def frame_rgb(masks):
    img = np.full((H, W, 3), FLOOR, np.uint8)
    for m in masks:
        img[m] = RAT
    return img


BACKGROUND = np.full((H, W, 3), FLOOR, np.uint8)


class FakeSam3:
    """Locates the crop inside the current frame and returns the true masks in it."""
    def __init__(self):
        self.truth = None
        self.frame = None
        self.mode = "truth"      # truth / merged / none / same
        self.same = None
        self.calls = 0

    def __call__(self, crop):
        self.calls += 1
        if self.mode == "none":
            return [], []
        res = cv2.matchTemplate(self.frame, crop, cv2.TM_SQDIFF)
        _, _, (x0, y0), _ = cv2.minMaxLoc(res)
        h, w = crop.shape[:2]
        if self.mode == "same":
            masks = [self.same]
        elif self.mode == "merged":
            masks = [self.truth[0] | self.truth[1]]
        else:
            masks = list(reversed(self.truth))     # reversed: order must not matter
        out = [m[y0:y0 + h, x0:x0 + w] for m in masks]
        out = [m for m in out if m.sum() > 0]
        return out, [0.9] * len(out)


def make(cfg_guard=None, cfg_rec=None):
    guard = AreaGuard(3, cfg_guard or GuardConfig(), FPS)
    sam = FakeSam3()
    rec = MaskRecovery(3, guard, cfg_rec or RecoveryConfig(), sam, BACKGROUND)
    return guard, sam, rec


def run_frames(rec, sam, start, stop, cutie_fn, mode="truth"):
    out = []
    for t in range(start, stop):
        truth = scene(t)
        img = frame_rgb(truth)
        sam.truth, sam.frame, sam.mode = truth, img, mode
        masks = cutie_fn(t, truth)
        cents = [mask_centroid(m) for m in masks]
        out.append((t, truth, rec.step(t, img, masks, cents)))
    return out


def correct(t, truth):
    return list(truth)


def fused(t, truth):            # CUTIE gives B's body to A, B is gone
    return [truth[0] | truth[1], None, truth[2]]


# ---------------------------------------------------------------------- guard
def test_guard_needs_sustained_flag_and_marks_onset():
    g = AreaGuard(1, GuardConfig(min_baseline_samples=5, sustain_frames=3), FPS)
    m = rat(200, 200)
    small = rat(200, 200) & (np.arange(W)[None, :] < 170)      # ~25 % of the area
    for t in range(10):
        assert not g.update(t, [m], [mask_centroid(m)]).triggered
    assert not g.update(10, [small], [None]).triggered          # 1 frame
    assert not g.update(11, [m], [None]).triggered              # back to normal resets
    for t in (12, 13):
        assert not g.update(t, [small], [None]).triggered
    st = g.update(14, [small], [None])
    assert st.triggered == {0: "low"} and st.onsets == {0: 12}


def test_guard_retry_schedule_and_cooldown():
    cfg = GuardConfig(min_baseline_samples=5, retry_fast_every=3, retry_fast_seconds=1.0,
                      retry_slow_every=10, verified_cooldown_frames=30)
    g = AreaGuard(1, cfg, FPS)
    m = rat(200, 200)
    for t in range(10):
        g.update(t, [m], [None])
    attempts = []
    for t in range(10, 90):
        st = g.update(t, [None], [None])
        if g.should_attempt(st):
            attempts.append(t)
            g.record_attempt(t, "failed")
    # flag at frame 12 (3rd missing frame); every 3 frames for 1 s (30 frames), then every 10
    assert attempts[:4] == [12, 15, 18, 21]
    assert all(b - a == 10 for a, b in zip(attempts, attempts[1:]) if a >= 42)
    g.record_attempt(90, "verified")
    assert g.episodes[-1].outcome == "verified"
    st = g.update(91, [None], [None])
    st = g.update(92, [None], [None])
    st = g.update(93, [None], [None])
    assert not st.triggered                                     # paused for 30 frames


# ---------------------------------------------------------------------- recovery
def test_fusion_is_split_with_identities_kept():
    guard, sam, rec = make()
    run_frames(rec, sam, 0, 30, correct)
    out = run_frames(rec, sam, 30, 40, fused)
    recovered = [(t, truth, r) for t, truth, r in out if r.reinject_index_mask is not None]
    assert recovered, "no recovery"
    t, truth, r = recovered[0]
    assert t == 32                                              # third fused frame
    assert iou(r.slot_masks[0], truth[0]) > 0.95                # A stays R1
    assert iou(r.slot_masks[1], truth[1]) > 0.95                # B comes back as R2 (no swap)
    assert r.sources[0] == "recovered" and r.sources[1] == "recovered"
    idx = r.reinject_index_mask
    assert (idx[truth[1]] == 2).mean() > 0.95 and (idx[truth[0]] == 1).mean() > 0.95
    assert guard.episodes[-1].outcome == "recovered"


def test_merged_sam3_is_retried_every_3_frames():
    guard, sam, rec = make()
    run_frames(rec, sam, 0, 30, correct)
    out = run_frames(rec, sam, 30, 38, fused, mode="merged")
    tries = [t for t, _, r in out if any(e["type"] == "attempt" for e in r.events)]
    assert tries == [32, 35] and not any(r.reinject_index_mask is not None for _, _, r in out)
    out = run_frames(rec, sam, 38, 42, fused, mode="truth")
    done = [t for t, _, r in out if r.reinject_index_mask is not None]
    # the fake CUTIE ignores the re-injection and stays fused, so the guard fires again 3 frames later
    assert done == [38, 41]


def test_rearing_is_verified_not_changed():
    guard, sam, rec = make()
    run_frames(rec, sam, 0, 30, correct)

    def rearing(t, truth):
        small = truth[0] & (np.arange(W)[None, :] < truth[0].nonzero()[1].min() + 30)
        sam.same = small
        return [small, truth[1], truth[2]]
    out = run_frames(rec, sam, 30, 34, rearing, mode="same")
    att = [e for _, _, r in out for e in r.events if e["type"] == "attempt"]
    assert att and att[0]["outcome"] == "verified"
    assert not any(r.reinject_index_mask is not None for _, _, r in out)


def test_lost_animal_found_in_background_residual():
    guard, sam, rec = make()
    run_frames(rec, sam, 0, 30, correct)

    def lost(t, truth):
        return [truth[0], None, truth[2]]
    out = run_frames(rec, sam, 30, 34, lost, mode="none")      # SAM3 finds nothing
    hits = [(t, truth, r) for t, truth, r in out if r.reinject_index_mask is not None]
    assert hits, "residual did not find the animal"
    t, truth, r = hits[0]
    assert iou(r.slot_masks[1], truth[1]) > 0.9 and iou(r.slot_masks[0], truth[0]) > 0.99


def test_compose_index_mask_priority():
    a = np.zeros((4, 4), bool); a[:2] = True
    b = np.zeros((4, 4), bool); b[1:3] = True
    assert compose_index_mask([a, b], (4, 4))[1, 0] == 2
    assert compose_index_mask([a, b], (4, 4), priority=[0])[1, 0] == 1


# ---------------------------------------------------------------------- tracks
def test_rle_matches_coco_and_round_trips():
    m = np.zeros((6, 5), np.uint8); m[1:4, 1:3] = 1; m[5, 4] = 1
    assert rle_encode(m) == "7330:N"                            # pycocotools.mask.encode output
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.random((int(rng.integers(1, 40)), int(rng.integers(1, 40)))) > 0.6
        assert (rle_decode(rle_encode(x), x.shape) == x).all()


class _Kp:
    def __init__(self, x, y, c):
        self.x, self.y, self.conf = x, y, c


class _Det:
    def __init__(self, n):
        self.keypoints = [_Kp(10 + i, 20 + i, 0.9) for i in range(n)]


def test_tracks_file_round_trip_and_plot(tmp_path):
    truth = scene(0)
    split = truth[0] | rat(420, 60)                             # R1 in two pieces
    w = TracksWriter(tmp_path / "tracks" / "tracks.jsonl.gz",
                     {"fps": FPS, "width": W, "height": H, "num_slots": 3,
                      "keypoint_names": [f"k{i}" for i in range(7)]}, flush_every=1)
    w.write_frame(0, FPS, "sam3", [split, truth[1], None], [(1, 2), (3, 4), None],
                  ["sam3", "sam3", None], [_Det(7), None, None], [False, False, False],
                  extras=[(truth[2], 0.4)], events=[{"type": "flag"}])
    w.close()
    lines = list(read_tracks(tmp_path / "tracks" / "tracks.jsonl.gz"))
    meta, f0 = lines[0]["meta"], lines[1]
    assert meta["num_slots"] == 3 and f0["phase"] == "sam3"
    r1, r2, r3 = f0["rats"]
    assert r1["n_parts"] == 2 and r2["n_parts"] == 1 and r3["rle"] is None and r3["area"] == 0
    assert (rle_decode(r1["rle"], (H, W)) == split).all()
    assert len(r1["kp"]) == 7 and r2["kp"] is None
    assert (rle_decode(f0["extra"][0]["rle"], (H, W)) == truth[2]).all()
    assert f0["events"] == [{"type": "flag"}]

    # a killed run leaves a truncated gzip: reading stops without raising
    raw = (tmp_path / "tracks" / "tracks.jsonl.gz").read_bytes()
    (tmp_path / "cut.jsonl.gz").write_bytes(raw[: len(raw) - 7])
    assert len(list(read_tracks(tmp_path / "cut.jsonl.gz"))) >= 1

    png = plot_area(list(range(100)), [[1.0, 1.0, 1.0]] * 50 + [[1.9, 0.1, 1.0]] * 50, FPS,
                    [{"start": 52, "end": 60, "outcome": "recovered", "slots": [1, 2], "onsets": {"1": 50}}],
                    [(0, 0, 255), (0, 255, 0), (255, 0, 0)], 0.4, 1.6, tmp_path / "area.png")
    assert png.exists() and png.stat().st_size > 1000
