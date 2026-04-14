"""Shared model loading utilities used by all pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from src.common.config_loader import get_device

logger = logging.getLogger(__name__)


def load_yolo(model_path: str | Path, device: str) -> YOLO:
    """Load a YOLO model. Auto-detects architecture (v8, 11, 26).

    Args:
        model_path: Path to the .pt weights file.
        device: Device string ("cuda" or "cpu").

    Returns:
        Loaded YOLO model on the specified device.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    logger.info("Loading YOLO model from %s", model_path)
    model = YOLO(str(model_path))
    model.to(device)
    return model


def load_sam2_predictor(
    checkpoint_path: str | Path,
    config_name: str,
    device: str,
) -> SAM2ImagePredictor:
    """Load a SAM2 image predictor.

    Args:
        checkpoint_path: Path to the SAM2 checkpoint .pt file.
        config_name: Hydra config name inside the sam2 package.
        device: Device string ("cuda" or "cpu").

    Returns:
        SAM2ImagePredictor ready for inference.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
    logger.info("Loading SAM2 from %s (config: %s)", checkpoint_path, config_name)
    sam2_model = build_sam2(config_name, str(checkpoint_path), device=device)
    return SAM2ImagePredictor(sam2_model)


def load_models(config: Dict[str, Any]) -> Tuple[YOLO, SAM2ImagePredictor]:
    """Load both YOLO and SAM2 models from a config dictionary.

    Args:
        config: Full pipeline config (must contain 'models' section).

    Returns:
        (yolo_model, sam2_predictor) tuple.
    """
    device = get_device(config)
    models_cfg = config["models"]
    logger.info("Using device: %s", device)

    yolo = load_yolo(models_cfg["yolo_path"], device)
    sam = load_sam2_predictor(
        models_cfg["sam2_checkpoint"],
        models_cfg["sam2_config"],
        device,
    )
    return yolo, sam
