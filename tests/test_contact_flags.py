"""Quality flags of the contact tracker."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.contacts_v2 import ContactClassifier  # noqa: E402
from src.common.utils import Detection, Keypoint  # noqa: E402


def _mouse(x, y, carried=False):
    kps = [Keypoint(x + 40, y, 0.9, "nose"), Keypoint(x + 30, y - 6, 0.9, "left_ear"),
           Keypoint(x + 30, y + 6, 0.9, "right_ear"), Keypoint(x, y, 0.9, "mid_body"),
           Keypoint(x - 40, y, 0.9, "tail_start"), Keypoint(x - 60, y, 0.9, "tail_base"),
           Keypoint(x - 90, y, 0.9, "tail_tip")]
    return Detection(x - 45, y - 15, x + 45, y + 15, 0.9, "mouse", kps, carried=carried)


def _classify(det_i, det_j):
    clf = ContactClassifier("0_1", 0, 1, {})
    return clf.classify(det_i, det_j, None, None, (100.0, 100.0), (400.0, 100.0),
                        (0.0, 0.0), (0.0, 0.0), 80.0, 80.0, frame_idx=0, time_sec=0.0)


def test_body_length_warm_start_uses_first_observations():
    from src.common.contacts_v2 import BodyLengthEstimator
    est = BodyLengthEstimator(slot_idx=0, fallback_px=120.0)
    assert est.current() == 120.0                      # nothing seen yet
    for _ in range(3):
        est.observe(_mouse(100, 100))                   # nose-tail_start = 80 px
    assert est.current() == 80.0                       # median of the first observations, not 120


def test_n2b_tolerance_radius():
    import numpy as np
    mask_j = np.zeros((300, 300), dtype=bool)
    mask_j[90:110, 143:200] = True                     # body 3 px to the right of i's nose (140, 100)
    det_i, det_j = _mouse(100, 100), _mouse(400, 100)
    strict = ContactClassifier("0_1", 0, 1, {})
    loose = ContactClassifier("0_1", 0, 1, {"n2b_nose_mask_radius_px": 4})
    args = (det_i, det_j, None, mask_j, (100.0, 100.0), (170.0, 100.0), (0.0, 0.0), (0.0, 0.0), 80.0, 80.0)
    assert strict.classify(*args, frame_idx=0, time_sec=0.0).scores.n2b == 0.0
    assert loose.classify(*args, frame_idx=0, time_sec=0.0).scores.n2b > 0.9


def test_simultaneous_contacts_are_counted_per_animal(tmp_path):
    from src.common.contacts_v2 import ContactTrackerV2
    # mouse 0 faces +x with its nose at (140,100); mouse 1 faces -x with its nose 10 px away;
    # mouse 2 faces -y with its nose 10 px below mouse 0's nose -> mouse 0 is in two N2N contacts
    m0 = _mouse(100, 100)
    m1 = Detection(150, 85, 240, 115, 0.9, "mouse", [
        Keypoint(150, 100, 0.9, "nose"), Keypoint(160, 94, 0.9, "left_ear"), Keypoint(160, 106, 0.9, "right_ear"),
        Keypoint(190, 100, 0.9, "mid_body"), Keypoint(230, 100, 0.9, "tail_start")])
    # (nose 21 px from mouse 0's nose = 0.27 BL -> N2N; 29 px from mouse 1's nose = 0.37 BL -> no contact)
    m2 = Detection(110, 115, 140, 205, 0.9, "mouse", [
        Keypoint(125, 115, 0.9, "nose"), Keypoint(119, 125, 0.9, "left_ear"), Keypoint(131, 125, 0.9, "right_ear"),
        Keypoint(125, 155, 0.9, "mid_body"), Keypoint(125, 195, 0.9, "tail_start")])
    # body length 80 px (the geometry of _mouse); the estimator has no observations yet on frame 0
    tr = ContactTrackerV2(tmp_path, fps=30.0, num_slots=3, video_path="x", config={"fallback_body_length_px": 80})
    tr.update([m0, m1, m2], [None, None, None], [m0.center(), m1.center(), m2.center()], frame_idx=0)
    ev = {e.pair_key: e for e in tr._all_events}
    assert ev["0_1"].contact_type.value == "N2N" and ev["0_2"].contact_type.value == "N2N"
    assert (ev["0_1"].n_contacts_i, ev["0_1"].n_contacts_j) == (2, 1)
    assert (ev["0_2"].n_contacts_i, ev["0_2"].n_contacts_j) == (2, 1)
    assert (ev["1_2"].n_contacts_i, ev["1_2"].n_contacts_j) == (1, 1)


def test_carried_keypoints_are_flagged_as_stale():
    assert _classify(_mouse(100, 100), _mouse(400, 100)).stale_keypoints is False
    assert _classify(_mouse(100, 100, carried=True), _mouse(400, 100)).stale_keypoints is True
    assert _classify(_mouse(100, 100), _mouse(400, 100, carried=True)).stale_keypoints is True


def _classify_masks(mask_i, mask_j=None):
    clf = ContactClassifier("0_1", 0, 1, {})
    return clf.classify(_mouse(100, 100), _mouse(400, 100), mask_i, mask_j, (100.0, 100.0), (400.0, 100.0),
                        (0.0, 0.0), (0.0, 0.0), 80.0, 80.0, frame_idx=0, time_sec=0.0)


def test_nose_off_mask_flag():
    # _mouse(100, 100): nose at (140, 100)
    on_nose = np.zeros((300, 300), dtype=bool)
    on_nose[85:115, 50:145] = True                       # covers the nose
    far = np.zeros((300, 300), dtype=bool)
    far[90:110, 100:120] = True                          # nearest pixel 21 px from the nose
    ev = _classify_masks(on_nose)
    assert ev.nose_off_mask_i is False and ev.nose_off_mask_j is False
    assert _classify_masks(far).nose_off_mask_i is True
    assert _classify_masks(None).nose_off_mask_i is False   # no mask: unknown, not flagged
    # j is judged against its own mask: i's mask over i's nose does not help j
    ev = _classify_masks(on_nose, far)                   # far does not cover j's nose (440, 100)
    assert ev.nose_off_mask_i is False and ev.nose_off_mask_j is True
    row = ev.to_csv_row()
    assert (row["nose_off_mask_i"], row["nose_off_mask_j"], row["nose_off_mask"]) == (0, 1, 1)
    assert _classify_masks(on_nose).to_csv_row()["nose_off_mask"] == 0


def test_nose_off_mask_tolerance_is_three_pixels():
    near = np.zeros((300, 300), dtype=bool)
    near[90:110, 100:138] = True                         # last mask column (137) is 3 px left of the nose
    nearer_miss = np.zeros((300, 300), dtype=bool)
    nearer_miss[90:110, 100:137] = True                  # last column (136) is 4 px away
    assert _classify_masks(near).nose_off_mask_i is False
    assert _classify_masks(nearer_miss).nose_off_mask_i is True


def test_nose_off_mask_in_session_quality_flags(tmp_path):
    from src.common.contacts_v2 import ContactTrackerV2
    m0, m1 = _mouse(100, 100), _mouse(400, 100)
    on0 = np.zeros((300, 600), dtype=bool)
    on0[85:115, 50:145] = True                           # covers m0's nose, not m1's
    tr = ContactTrackerV2(tmp_path, fps=30, num_slots=2, video_path="x", config={})
    tr.update([m0, m1], [on0, on0], [m0.center(), m1.center()], frame_idx=0)
    summary = tr.finalize()
    assert summary["quality_flags"]["nose_off_mask"] == 1          # m1's nose is off its mask


def test_summary_reports_thresholds_in_cm(tmp_path):
    from src.common.contacts_v2 import ContactTrackerV2
    m0, m1 = _mouse(100, 100), _mouse(400, 100)
    tr = ContactTrackerV2(tmp_path, fps=30, num_slots=2, video_path="x",
                          config={"fallback_body_length_px": 100, "reference_body_length_cm": 10.0})
    tr.update([m0, m1], [None, None], [m0.center(), m1.center()], frame_idx=0)
    summary = tr.finalize()
    meta = summary["metadata"]
    assert meta["body_length_px_median"] == pytest.approx(100.0)
    assert meta["cm_per_px"] == pytest.approx(0.1)
    assert meta["cm_per_px_source"] == "estimated_from_body_length"
    assert meta["thresholds_cm"]["contact_enter"] == pytest.approx(3.0)      # 0.30 BL x 100 px x 0.1
    assert meta["thresholds_cm"]["contact_exit"] == pytest.approx(4.5)       # 0.45 BL
    assert meta["thresholds_cm"]["proximity"] == pytest.approx(10.0)         # 1.0 BL


def test_measured_cm_per_px_overrides_the_estimate(tmp_path):
    from src.common.contacts_v2 import ContactTrackerV2
    m0, m1 = _mouse(100, 100), _mouse(400, 100)
    tr = ContactTrackerV2(tmp_path, fps=30, num_slots=2, video_path="x",
                          config={"fallback_body_length_px": 100, "cm_per_px": 0.05,
                                  "contact_zone_bl_enter": 0.2})
    tr.update([m0, m1], [None, None], [m0.center(), m1.center()], frame_idx=0)
    meta = tr.finalize()["metadata"]
    assert meta["cm_per_px"] == pytest.approx(0.05) and meta["cm_per_px_source"] == "config"
    assert meta["thresholds_cm"]["contact_enter"] == pytest.approx(1.0)      # 0.2 BL x 100 px x 0.05
