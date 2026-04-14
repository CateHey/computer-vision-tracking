"""Shared constants across all pipelines."""

# Default keypoint names for the 7-point rat pose model.
# Order matches the training annotation order in Roboflow.
# Override via config: detection.keypoint_names
DEFAULT_KEYPOINT_NAMES = [
    "tail_tip",
    "tail_base",
    "tail_start",
    "mid_body",
    "nose",
    "right_ear",
    "left_ear",
]
