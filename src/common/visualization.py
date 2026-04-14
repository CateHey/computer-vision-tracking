"""
Rendering utilities for overlaying masks, centroids, and annotations on frames.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

# Default RGBA colors for tracked animals
DEFAULT_COLORS = [
    (0, 255, 0, 150),    # Green
    (255, 0, 0, 150),    # Red
    (0, 0, 255, 150),    # Blue
    (255, 255, 0, 150),  # Yellow
]


def apply_masks_overlay(
    frame_rgb: np.ndarray,
    masks: List[np.ndarray],
    colors: Optional[List[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    """Overlay colored semi-transparent masks on an RGB frame.

    Args:
        frame_rgb: Input frame in RGB format (H, W, 3).
        masks: List of boolean masks, each (H, W).
        colors: RGBA color tuples per mask. Defaults to DEFAULT_COLORS.

    Returns:
        Frame with masks overlaid (RGB, uint8).
    """
    if colors is None:
        colors = DEFAULT_COLORS

    h, w, _ = frame_rgb.shape
    out = frame_rgb.astype(np.float32).copy()

    for i, m in enumerate(masks):
        r, g, b, a = colors[i % len(colors)]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[:, :, :3] = (r, g, b)
        overlay[:, :, 3] = m.astype(np.uint8) * a

        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        out = out * (1 - alpha) + overlay[:, :, :3].astype(np.float32) * alpha

    return out.astype(np.uint8)


def draw_centroids(
    frame_rgb: np.ndarray,
    centroids: List[Tuple[float, float] | None],
    colors: Optional[List[Tuple[int, int, int, int]]] = None,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw circles and text labels at centroid positions.

    Args:
        frame_rgb: Input frame in RGB format.
        centroids: List of (x, y) tuples (or None for missing).
        colors: RGBA color tuples per centroid. Defaults to DEFAULT_COLORS.
        labels: Text labels per centroid. Defaults to "R1", "R2", etc.

    Returns:
        Annotated frame (RGB, uint8).
    """
    if colors is None:
        colors = DEFAULT_COLORS

    out = frame_rgb.copy()
    for i, c in enumerate(centroids):
        if c is None:
            continue
        cx, cy = int(c[0]), int(c[1])
        r, g, b, _ = colors[i % len(colors)]
        label = labels[i] if labels else f"R{i + 1}"
        cv2.circle(out, (cx, cy), 6, (r, g, b), -1)
        cv2.putText(
            out, label, (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    return out


def draw_detections(
    frame_rgb: np.ndarray,
    detections: list,
    colors: Optional[List[Tuple[int, int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes with class labels and confidence scores.

    Args:
        frame_rgb: Input frame in RGB format.
        detections: List of Detection objects (from src.common.utils).
        colors: RGBA color tuples per detection. Defaults to DEFAULT_COLORS.
        thickness: Box line thickness.
        font_scale: Label font scale.

    Returns:
        Annotated frame with boxes and labels.
    """
    if colors is None:
        colors = DEFAULT_COLORS

    out = frame_rgb.copy()
    for i, det in enumerate(detections):
        r, g, b, _ = colors[i % len(colors)]
        color = (r, g, b)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        label = det.class_name or "obj"
        label = f"{label} {det.conf:.2f} id={getattr(det, 'track_id', 'NA')}"

        # Draw label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
        )

    return out


def draw_keypoints(
    frame_rgb: np.ndarray,
    detections: list,
    colors: Optional[List[Tuple[int, int, int, int]]] = None,
    min_conf: float = 0.3,
    radius: int = 4,
    font_scale: float = 0.35,
) -> np.ndarray:
    """Draw keypoints from pose detections.

    Each detection's keypoints are drawn as colored dots with name labels.
    Only keypoints with confidence above min_conf are drawn.

    Args:
        frame_rgb: Input frame in RGB format.
        detections: List of Detection objects that have .keypoints.
        colors: RGBA color tuples per detection. Defaults to DEFAULT_COLORS.
        min_conf: Minimum keypoint confidence to draw.
        radius: Circle radius for keypoint dots.
        font_scale: Font scale for keypoint name labels.

    Returns:
        Annotated frame with keypoints.
    """
    if colors is None:
        colors = DEFAULT_COLORS

    out = frame_rgb.copy()
    for i, det in enumerate(detections):
        if det.keypoints is None:
            continue
        r, g, b, _ = colors[i % len(colors)]
        color = (r, g, b)

        for kp in det.keypoints:
            if kp.conf < min_conf:
                continue
            cx, cy = int(kp.x), int(kp.y)
            # Filled circle
            cv2.circle(out, (cx, cy), radius, color, -1)
            # White outline for visibility
            cv2.circle(out, (cx, cy), radius, (255, 255, 255), 1)
            # Label
            if kp.name:
                cv2.putText(
                    out, kp.name, (cx + radius + 2, cy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                )

    return out


def draw_text(
    frame_rgb: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw text on a frame.

    Args:
        frame_rgb: Input frame in RGB format.
        text: Text string to render.
        position: (x, y) position for the text baseline.
        scale: Font scale.
        color: RGB color tuple.
        thickness: Text thickness.

    Returns:
        Annotated frame.
    """
    out = frame_rgb.copy()
    cv2.putText(out, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return out
