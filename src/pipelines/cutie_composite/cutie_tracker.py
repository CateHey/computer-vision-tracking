"""
Cutie tracker wrapper.

Thin, clean wrapper around the Cutie VOS model (hkchengrex/Cutie) for continuous
multi-object tracking. Based on the official repo API:

    from cutie.utils.get_default_model import get_default_model
    from cutie.inference.inference_core import InferenceCore

Responsibilities:
  - Load Cutie model once
  - Keep a persistent InferenceCore (continuous tracking, no chunk resets)
  - init_with_masks(): first frame, seed masks from SAM3
  - track(): propagate a frame (no mask) -> get current masks
  - reinject_masks(): correction WITHOUT resetting memory (force_permanent=True)

Key facts from the API analysis:
  - Cutie expects RGB images in [0, 1] (no ImageNet normalization from caller;
    the network normalizes internally)
  - step(image, mask, objects, idx_mask, force_permanent) is the only per-frame call
  - mask can be passed as index mask (idx_mask=True) or one-hot (idx_mask=False)
  - output is prob tensor (num_objects+1, H, W); channel 0 = background
  - output_prob_to_mask() remaps to an ID mask (H, W)
  - max_internal_size controls internal resolution (mask detail vs speed/VRAM)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Cutie tensor helpers (copied from Cutie's gui/interactive_utils.py to avoid
# depending on the repo's internal 'gui' module path).
# ============================================================================

def _image_to_torch(frame: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """RGB (H,W,3) uint8 numpy -> torch (3,H,W) float in [0,1]."""
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
    return frame


def _torch_prob_to_numpy_mask(prob: torch.Tensor) -> np.ndarray:
    """Prob tensor (num_objs+1, H, W) -> ID mask (H,W) uint8 (argmax over channels)."""
    mask = torch.max(prob, dim=0).indices
    return mask.cpu().numpy().astype(np.uint8)


def _index_numpy_to_one_hot_torch(mask: np.ndarray, num_classes: int) -> torch.Tensor:
    """Index mask (H,W) -> one-hot torch (num_classes, H, W) float."""
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


class CutieTracker:
    """Continuous Cutie tracker with mask re-injection support."""

    def __init__(
        self,
        device: str = "cuda",
        max_internal_size: int = 720,
        mem_every: int = 5,
        max_mem_frames: int = 5,
        use_long_term: bool = True,
        weights_path: Optional[str] = None,
    ):
        """
        Args:
            device: cuda | cpu
            max_internal_size: internal resolution (shorter side). Higher = more
                mask detail (better tail) but more VRAM/slower. 480/720/960.
            mem_every: write to memory every N frames
            max_mem_frames: working memory size (frames)
            use_long_term: enable long-term memory (needed for long videos)
            weights_path: optional explicit path to cutie-base-mega.pth
        """
        self.device = device
        self.max_internal_size = max_internal_size

        # Import Cutie core (from the installed repo)
        from cutie.utils.get_default_model import get_default_model
        from cutie.inference.inference_core import InferenceCore

        self._InferenceCore = InferenceCore

        logger.info("Loading Cutie model...")
        self.cutie = get_default_model()

        # Apply config overrides
        try:
            from omegaconf import open_dict
            with open_dict(self.cutie.cfg):
                self.cutie.cfg.mem_every = mem_every
                self.cutie.cfg.max_mem_frames = max_mem_frames
                self.cutie.cfg.use_long_term = use_long_term
        except Exception as e:
            logger.warning("Could not apply Cutie config overrides: %s", e)

        # Optional explicit weights
        if weights_path:
            try:
                loaded = torch.load(weights_path, map_location=device)
                weights = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
                self.cutie.load_weights(weights)
                logger.info("Loaded Cutie weights from %s", weights_path)
            except Exception as e:
                logger.warning("Could not load custom weights, using default: %s", e)

        self.cutie = self.cutie.to(device).eval()
        self.processor = self._InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = max_internal_size

        self._num_objects = 0
        logger.info("CutieTracker ready (max_internal_size=%d, device=%s)",
                    max_internal_size, device)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Full reset: new InferenceCore (wipes all memory). Rarely needed."""
        self.processor = self._InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = self.max_internal_size
        self._num_objects = 0

    def _frame_to_torch(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """Convert an RGB numpy frame to Cutie's expected torch tensor."""
        return _image_to_torch(frame_rgb, device=self.device)

    def init_with_masks(
        self,
        frame_rgb: np.ndarray,
        index_mask: np.ndarray,
        num_objects: int,
    ) -> np.ndarray:
        """Seed the tracker on the first frame with an index mask.

        Args:
            frame_rgb: (H, W, 3) RGB frame
            index_mask: (H, W) int mask, 0=background, 1..N=object IDs
            num_objects: number of objects (N)

        Returns:
            (H, W) predicted ID mask for this frame
        """
        self._num_objects = num_objects
        frame_torch = self._frame_to_torch(frame_rgb)

        # Convert index mask to one-hot, drop background channel [0]
        one_hot = _index_numpy_to_one_hot_torch(
            index_mask, num_objects + 1
        ).to(self.device)

        with torch.inference_mode():
            with torch.amp.autocast(self.device, enabled=(self.device == "cuda")):
                prob = self.processor.step(
                    frame_torch,
                    one_hot[1:],           # exclude background channel
                    idx_mask=False,        # we pass one-hot object masks
                    force_permanent=True,  # commit as permanent memory
                )
        return _torch_prob_to_numpy_mask(prob)

    def track(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Propagate one frame (no mask). Returns (H, W) ID mask."""
        frame_torch = self._frame_to_torch(frame_rgb)
        with torch.inference_mode():
            with torch.amp.autocast(self.device, enabled=(self.device == "cuda")):
                prob = self.processor.step(frame_torch)
        return _torch_prob_to_numpy_mask(prob)

    def reinject_masks(
        self,
        frame_rgb: np.ndarray,
        index_mask: np.ndarray,
        num_objects: int,
    ) -> np.ndarray:
        """Correct tracking mid-video WITHOUT resetting memory.

        Passes a fresh mask with force_permanent=True. Cutie fuses the new mask
        with its existing memory (mutual exclusion) and keeps tracking. This is
        the recovery mechanism (Option B) — the good rats' memory is preserved.

        Args:
            frame_rgb: (H, W, 3) RGB frame
            index_mask: (H, W) int mask with the corrected identities
            num_objects: number of objects

        Returns:
            (H, W) predicted ID mask for this frame
        """
        self._num_objects = max(self._num_objects, num_objects)
        frame_torch = self._frame_to_torch(frame_rgb)

        one_hot = _index_numpy_to_one_hot_torch(
            index_mask, num_objects + 1
        ).to(self.device)

        with torch.inference_mode():
            with torch.amp.autocast(self.device, enabled=(self.device == "cuda")):
                prob = self.processor.step(
                    frame_torch,
                    one_hot[1:],
                    idx_mask=False,
                    force_permanent=True,
                )
        return _torch_prob_to_numpy_mask(prob)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def id_mask_to_slot_masks(
        self,
        id_mask: np.ndarray,
        num_slots: int,
    ) -> List[Optional[np.ndarray]]:
        """Split an ID mask (H, W) into per-slot boolean masks.

        Slot i corresponds to object ID (i+1). Returns None for missing IDs.
        """
        slot_masks: List[Optional[np.ndarray]] = [None] * num_slots
        for slot_idx in range(num_slots):
            obj_id = slot_idx + 1
            m = (id_mask == obj_id)
            if m.any():
                slot_masks[slot_idx] = m
        return slot_masks

    def cleanup(self) -> None:
        """Release GPU memory."""
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "cutie"):
            del self.cutie
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def slot_masks_to_index_mask(
    slot_masks: List[Optional[np.ndarray]],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, int]:
    """Combine per-slot boolean masks into a single index mask.

    Slot i -> object ID (i+1). Later slots overwrite earlier ones on overlap.

    Returns: (index_mask (H, W) int, num_objects)
    """
    index_mask = np.zeros(shape, dtype=np.int32)
    num_objects = 0
    for slot_idx, m in enumerate(slot_masks):
        if m is None:
            continue
        obj_id = slot_idx + 1
        index_mask[m] = obj_id
        num_objects = max(num_objects, obj_id)
    return index_mask, num_objects