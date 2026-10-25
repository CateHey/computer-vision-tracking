"""Quality flags of the contact tracker."""
import sys
from pathlib import Path

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
