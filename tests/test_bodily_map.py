"""Synthetic-pose tests for the bodily-map projection (no file I/O)."""
import numpy as np

from src.common.bodily_map.locate import flank_facing, project_to_midline


def mouse_along_x():
    """Mouse lying along +x, nose at origin, tail toward +x (image coords, y down).

    The mouse faces -x, so its right side is image -y: right_ear sits at
    y = -10 and must give s > 0.
    """
    k = {"nose": (0, 0), "mid_body": (50, 0), "tail_start": (100, 0), "tail_base": (130, 0),
         "tail_tip": (180, 0), "right_ear": (10, -10), "left_ear": (10, 10)}
    return {n: np.array(p, float) for n, p in k.items()}


def test_nose_query():
    r = project_to_midline(mouse_along_x(), (0, 0), 100)
    assert r["t"] < 0.02 and r["region"] == "nose"


def test_tail_start_is_anogenital():
    r = project_to_midline(mouse_along_x(), (100, 0), 100)
    assert r["region"] == "anogenital"


def test_beyond_tail_start_is_tail():
    r = project_to_midline(mouse_along_x(), (160, 0), 100)
    assert r["region"] == "tail" and r["t"] > 0.7


def test_ear_sign_convention():
    k = mouse_along_x()
    assert project_to_midline(k, k["right_ear"], 100)["s"] > 0
    assert project_to_midline(k, k["left_ear"], 100)["s"] < 0


def test_flanks_by_side():
    k = mouse_along_x()
    right = project_to_midline(k, (50, -20), 100)
    left = project_to_midline(k, (50, 20), 100)
    assert right["s"] > 0 and right["region"] == "flank_R"
    assert left["s"] < 0 and left["region"] == "flank_L"
    assert project_to_midline(k, (50, 2), 100)["region"] == "mid_back"


def test_missing_keypoints_unresolved():
    k = mouse_along_x()
    del k["tail_start"]
    assert project_to_midline(k, (0, 0), 100) is None


def test_sbs_picks_facing_flank():
    k = mouse_along_x()
    assert flank_facing(k, (50, -80), 100)["region"] == "flank_R"
    assert flank_facing(k, (50, 80), 100)["region"] == "flank_L"
