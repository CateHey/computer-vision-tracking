"""
Composition helpers for the isolated_composite pipeline.

Central idea:
  - Compute an empty-scene background from the video itself via temporal median.
  - For each rat, build a YOLO input frame by ERASING only the OTHER rat
    (replacing its pixels with background), keeping the arena context and
    the rat of interest untouched. Keypoints come out already in original
    frame coordinates — no inverse mapping needed.
  - For visual inspection, also build per-rat videos where only one rat is
    visible on top of the computed background.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.common.utils import Detection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background computation
# ---------------------------------------------------------------------------

def compute_background_median(
    video_path: str | Path,
    n_samples: int = 200,
    seed: int = 42,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> np.ndarray:
    """Compute an empty-scene background via per-pixel temporal median.

    Samples n_samples frames uniformly spaced across the video (or within the
    [start_frame, end_frame) range) and returns the per-pixel median in BGR.
    Moving rats cancel out as long as any given pixel is un-occluded in the
    majority of samples.

    The sampling indices are deterministic given (n_samples, seed, frame range),
    so parallel workers can compute the same background independently without
    synchronization.

    Returns:
        BGR uint8 background image (H, W, 3).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for background: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lo = max(0, start_frame)
    hi = total if end_frame is None else min(total, end_frame)
    span = max(1, hi - lo)

    n_samples = max(5, min(n_samples, span))

    rng = np.random.default_rng(seed)
    idxs = np.linspace(lo, hi - 1, num=n_samples, dtype=np.int64)
    jitter = rng.integers(-2, 3, size=n_samples)
    idxs = np.clip(idxs + jitter, lo, hi - 1)
    idxs = np.unique(idxs)

    logger.info(
        "Computing background median: %d samples across frames [%d, %d), video=%dx%d",
        len(idxs), lo, hi, width, height,
    )

    samples: List[np.ndarray] = []
    for i in tqdm(idxs, desc="Computing background", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        samples.append(frame)

    cap.release()

    if len(samples) < 3:
        raise RuntimeError(
            f"Background computation failed: only {len(samples)} frames readable"
        )

    stack = np.stack(samples, axis=0).astype(np.uint8)
    background = np.median(stack, axis=0).astype(np.uint8)
    logger.info("Background median computed from %d frames", len(samples))
    return background


def _video_fingerprint(video_path: Path) -> str:
    """Short hash of the video file (size + mtime + first 1 MB) for cache keys."""
    h = hashlib.sha1()
    stat = video_path.stat()
    h.update(str(stat.st_size).encode())
    h.update(str(int(stat.st_mtime)).encode())
    with video_path.open("rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def load_or_compute_background(
    config: Dict[str, Any],
    video_path: str | Path,
    run_dir: Path,
) -> np.ndarray:
    """Resolve the background for the current run.

    Precedence:
      1. If config.background.cache_dir contains a cached background whose
         fingerprint matches the source video, load it.
      2. Otherwise, compute via compute_background_median and save it to
         run_dir/background.png plus the cache dir (if configured).

    The cache directory is shared across runs so multi-GPU workers can reuse
    the same computation; each worker with a different chunk range still
    resolves the same fingerprint because it is based on the full video file.
    """
    video_path = Path(video_path).resolve()
    bg_cfg = config.get("background", {}) or {}
    n_samples = int(bg_cfg.get("n_samples", 200))
    seed = int(bg_cfg.get("sampling_seed", 42))
    use_cache = bool(bg_cfg.get("cache", True))

    run_dir = Path(run_dir)
    run_bg_path = run_dir / "background.png"

    cache_dir_cfg = bg_cfg.get("cache_dir")
    if cache_dir_cfg:
        cache_dir = Path(cache_dir_cfg)
        if not cache_dir.is_absolute():
            cache_dir = Path(config["_meta"]["project_root"]) / cache_dir
    else:
        cache_dir = run_dir.parent / "_background_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = _video_fingerprint(video_path)
    cache_key = f"{video_path.stem}_{fingerprint}_n{n_samples}_s{seed}.png"
    cache_path = cache_dir / cache_key

    if use_cache and cache_path.exists():
        logger.info("Loading cached background: %s", cache_path)
        background = cv2.imread(str(cache_path))
        if background is not None:
            cv2.imwrite(str(run_bg_path), background)
            return background
        logger.warning("Cached background unreadable, recomputing")

    background = compute_background_median(
        video_path, n_samples=n_samples, seed=seed,
    )

    cv2.imwrite(str(run_bg_path), background)
    if use_cache:
        cv2.imwrite(str(cache_path), background)
        logger.info("Background cached to %s", cache_path)

    return background


# ---------------------------------------------------------------------------
# Erase / compose helpers
# ---------------------------------------------------------------------------

def _build_erase_alpha(
    mask_other: np.ndarray,
    mask_self: np.ndarray,
    dilate_px: int,
    feather_px: int,
) -> np.ndarray:
    """Build a float alpha map where 1.0 = replace with background, 0.0 = keep frame.

    The alpha is active on mask_other (dilated) but suppressed where mask_self
    is True — we never erase pixels that belong to the rat of interest, even
    if SAM2 claimed both masks own them.
    """
    other = mask_other.astype(np.uint8)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        other = cv2.dilate(other, kernel, iterations=1)

    other = other.astype(bool) & ~mask_self.astype(bool)

    alpha = other.astype(np.float32)
    if feather_px > 0:
        k = 2 * feather_px + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

    return alpha


def erase_other_rat(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray,
    mask_self: np.ndarray,
    mask_other: np.ndarray,
    dilate_px: int = 15,
    feather_px: int = 5,
) -> np.ndarray:
    """Return a copy of frame_bgr with mask_other's pixels replaced by background.

    Pixels shared by both masks are preserved (they belong to the rat of interest).
    A feathered border blends the erase edge to hide lighting mismatch.
    """
    if background_bgr.shape != frame_bgr.shape:
        raise ValueError(
            f"Background shape {background_bgr.shape} does not match frame {frame_bgr.shape}"
        )

    alpha = _build_erase_alpha(mask_other, mask_self, dilate_px, feather_px)
    alpha3 = alpha[:, :, None]

    frame_f = frame_bgr.astype(np.float32)
    bg_f = background_bgr.astype(np.float32)
    out = frame_f * (1.0 - alpha3) + bg_f * alpha3
    return out.astype(np.uint8)


def compose_isolated_video_frame(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray,
    mask_self: np.ndarray,
    feather_px: int = 3,
) -> np.ndarray:
    """Build a single-rat visualization frame: only mask_self over background.

    Used for the per-rat output videos (rat_1.avi, rat_2.avi). NOT used as a
    YOLO input — it has more domain shift than erase_other_rat.
    """
    if background_bgr.shape != frame_bgr.shape:
        raise ValueError(
            f"Background shape {background_bgr.shape} does not match frame {frame_bgr.shape}"
        )

    alpha = mask_self.astype(np.float32)
    if feather_px > 0:
        k = 2 * feather_px + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)
    alpha3 = alpha[:, :, None]

    frame_f = frame_bgr.astype(np.float32)
    bg_f = background_bgr.astype(np.float32)
    out = bg_f * (1.0 - alpha3) + frame_f * alpha3
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-slot detection selection
# ---------------------------------------------------------------------------

def pick_detection_for_slot(
    detections: List[Detection],
    mask_self: np.ndarray,
    anchor_keypoint: str = "mid_body",
    min_anchor_conf: float = 0.1,
) -> Optional[Detection]:
    """Pick the best YOLO detection for a single-rat composite frame.

    Since the composite contains only one rat (the other was erased), most
    of the time there will be exactly one detection. When there are multiple
    (noise, partial echoes), prefer:
      1. Detections whose anchor keypoint (mid_body) falls inside mask_self.
      2. Among those, the one with highest confidence.
      3. If none contains the anchor, fall back to the max-conf detection
         whose bbox center is inside mask_self.
      4. Otherwise None.
    """
    if not detections:
        return None

    h, w = mask_self.shape
    contained: List[Detection] = []
    bbox_contained: List[Detection] = []

    for det in detections:
        anchor = None
        if det.keypoints:
            for kp in det.keypoints:
                if kp.name == anchor_keypoint:
                    anchor = kp
                    break

        if anchor is not None and anchor.conf >= min_anchor_conf:
            ix, iy = int(round(anchor.x)), int(round(anchor.y))
            if 0 <= iy < h and 0 <= ix < w and mask_self[iy, ix]:
                contained.append(det)
                continue

        cx, cy = det.center()
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= iy < h and 0 <= ix < w and mask_self[iy, ix]:
            bbox_contained.append(det)

    if contained:
        return max(contained, key=lambda d: d.conf)
    if bbox_contained:
        return max(bbox_contained, key=lambda d: d.conf)
    return None
