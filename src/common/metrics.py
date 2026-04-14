"""
Geometry and distance metrics for detection and segmentation evaluation.

Provides IoU computation (bounding-box and mask-based), centroid calculation,
and closeness evaluation for multi-animal interaction analysis.
"""

from __future__ import annotations

import math
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .utils import Detection


def bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two bounding boxes.

    Args:
        a: (x1, y1, x2, y2) for box A.
        b: (x1, y1, x2, y2) for box B.

    Returns:
        IoU value in [0.0, 1.0].
    """
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])

    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two boolean masks.

    Args:
        mask1: Boolean numpy array.
        mask2: Boolean numpy array of the same shape.

    Returns:
        IoU value in [0.0, 1.0].
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_centroid(mask: np.ndarray) -> Tuple[float, float] | None:
    """Compute the (x, y) centroid of a boolean mask.

    Args:
        mask: 2D boolean numpy array.

    Returns:
        (x, y) centroid tuple, or None if mask is empty.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def evaluate_closeness(
    detections: List[Detection],
    frame_width: int,
    frame_height: int,
    distance_threshold_norm: float,
    iou_threshold: float,
) -> Tuple[bool, float, float]:
    """Determine if detected animals are "close" in a frame.

    Distance between detection centroids is normalized by the frame diagonal.
    A frame is classified as "close" if there are at least 2 detections and
    either the minimum normalized distance is below the threshold or the
    maximum bounding-box IoU exceeds the threshold.

    Args:
        detections: List of Detection objects.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        distance_threshold_norm: Normalized distance threshold (0.0-1.0).
        iou_threshold: IoU threshold for bounding-box overlap.

    Returns:
        (is_close, min_distance_norm, max_iou) tuple.
    """
    n = len(detections)
    if n < 2:
        return False, 1.0, 0.0

    diag = math.sqrt(frame_width ** 2 + frame_height ** 2)
    if diag <= 0:
        return False, 1.0, 0.0

    min_dist_norm = 1.0
    max_iou = 0.0

    for i in range(n):
        ci = detections[i].center()
        box_i = detections[i].as_tuple()
        for j in range(i + 1, n):
            cj = detections[j].center()
            dist = math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
            dist_norm = dist / diag
            if dist_norm < min_dist_norm:
                min_dist_norm = dist_norm

            iou = bbox_iou(box_i, detections[j].as_tuple())
            if iou > max_iou:
                max_iou = iou

    is_close = (min_dist_norm < distance_threshold_norm) or (max_iou > iou_threshold)
    return is_close, min_dist_norm, max_iou
