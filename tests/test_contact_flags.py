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


def test_carried_keypoints_are_flagged_as_stale():
    assert _classify(_mouse(100, 100), _mouse(400, 100)).stale_keypoints is False
    assert _classify(_mouse(100, 100, carried=True), _mouse(400, 100)).stale_keypoints is True
    assert _classify(_mouse(100, 100), _mouse(400, 100, carried=True)).stale_keypoints is True
