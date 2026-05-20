"""
Reset strategies for SAM3 reset pipeline.

When a failure is detected, we try to recover by re-segmenting the failed frame
using SAM3 image predictor (without context). Then we initialize a new video
session from that frame onward.

3 strategies in cascade (fallback order):
  1. Masks as prompt — best for separated rats (preferred)
  2. Intelligent points — better for closely packed rats
  3. Bboxes — last resort

Each strategy attempts to bootstrap the new video session. If a strategy fails,
the next one is tried automatically.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# UTILITIES: convertir máscara a otras representaciones
# ============================================================================

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convierte una máscara binaria al bbox que la envuelve.

    Returns: (x1, y1, x2, y2) o None si la máscara está vacía.
    """
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def mask_to_smart_points(mask: np.ndarray, num_points: int = 7) -> List[Tuple[int, int]]:
    """Toma puntos estratégicos de una máscara para usar como prompt.

    Estrategia: centroide + extremos del eje principal + puntos a lo largo del
    esqueleto medial. Esto garantiza que se cubre toda la rata (incluida la cola).

    Args:
        mask: máscara binaria 2D
        num_points: cantidad aproximada de puntos a generar

    Returns:
        Lista de (x, y) en coordenadas de imagen.
    """
    if not mask.any():
        return []

    ys, xs = np.where(mask)
    points = []

    # 1. Centroide
    cx = float(xs.mean())
    cy = float(ys.mean())
    points.append((int(cx), int(cy)))

    # 2. Extremos: punto más arriba, abajo, izquierda, derecha
    # Esto captura cabeza/cola/lados
    if len(points) < num_points:
        idx_top = ys.argmin()
        points.append((int(xs[idx_top]), int(ys[idx_top])))
    if len(points) < num_points:
        idx_bottom = ys.argmax()
        points.append((int(xs[idx_bottom]), int(ys[idx_bottom])))
    if len(points) < num_points:
        idx_left = xs.argmin()
        points.append((int(xs[idx_left]), int(ys[idx_left])))
    if len(points) < num_points:
        idx_right = xs.argmax()
        points.append((int(xs[idx_right]), int(ys[idx_right])))

    # 3. Si faltan puntos, agregar muestreo uniforme del eje principal
    # Calcular eje principal con SVD simple
    while len(points) < num_points and len(xs) > 1:
        # Tomar puntos a 1/4, 1/2, 3/4 de la distancia entre extremos
        for fraction in [0.25, 0.5, 0.75]:
            if len(points) >= num_points:
                break
            # Punto entre centroide y un extremo aleatorio
            idx_sample = int(len(xs) * fraction)
            if 0 <= idx_sample < len(xs):
                points.append((int(xs[idx_sample]), int(ys[idx_sample])))
        break  # evitar bucle infinito

    return points[:num_points]


# ============================================================================
# RESET PRINCIPAL: SAM3 image predictor sin contexto
# ============================================================================

def sam3_image_segment(
    model,
    processor,
    frame_rgb: np.ndarray,
    text_prompt: str,
    device: str,
    score_threshold: float = 0.5,
) -> Tuple[np.ndarray, List[float]]:
    """Ejecuta SAM3 IMAGE (sin contexto de video) sobre un solo frame.

    Args:
        model: Sam3Model (no video) o el video model usado como image
        processor: Sam3Processor / Sam3VideoProcessor
        frame_rgb: frame RGB (H, W, 3) numpy array
        text_prompt: prompt de texto (ej. "mouse")
        device: cuda o cpu
        score_threshold: filtrar detecciones de score bajo

    Returns:
        (masks: np.ndarray (N, H, W) bool, scores: List[float])
    """
    from PIL import Image
    pil_frame = Image.fromarray(frame_rgb)

    # Usamos el video processor con un solo frame
    # Esto crea una sesión "mini" que solo segmenta ese frame
    session = processor.init_video_session(
        video=[pil_frame],
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    session = processor.add_text_prompt(
        inference_session=session,
        text=text_prompt,
    )

    masks_out = None
    scores_out = None
    for model_out in model.propagate_in_video_iterator(inference_session=session):
        processed = processor.postprocess_outputs(session, model_out)
        masks_out = processed['masks'].cpu().numpy()
        scores_out = processed['scores'].cpu().tolist()
        break  # solo necesitamos el primer (y único) frame

    del session

    if masks_out is None or len(masks_out) == 0:
        return np.zeros((0, frame_rgb.shape[0], frame_rgb.shape[1]), dtype=bool), []

    # Filtrar por score
    keep = [i for i, s in enumerate(scores_out) if s >= score_threshold]
    if not keep:
        return np.zeros((0, frame_rgb.shape[0], frame_rgb.shape[1]), dtype=bool), []

    masks_filtered = masks_out[keep] > 0.5
    scores_filtered = [scores_out[i] for i in keep]
    return masks_filtered, scores_filtered


# ============================================================================
# REASIGNACIÓN DE IDENTIDADES
# ============================================================================

def reassign_identities(
    new_masks: np.ndarray,
    avg_centroids: List[Optional[Tuple[float, float]]],
    num_slots: int,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[Tuple[float, float]]]]:
    """Asigna las máscaras nuevas a los slots existentes comparando con centroides promedio.

    Greedy: empareja el slot con la máscara más cercana, marca ambos como usados,
    repite.

    Args:
        new_masks: array (N_new, H, W) de máscaras frescas
        avg_centroids: centroide promedio de cada slot (de últimos K frames)
        num_slots: cantidad de slots a llenar

    Returns:
        (slot_masks, slot_centroids)
    """
    n_new = len(new_masks)
    slot_masks: List[Optional[np.ndarray]] = [None] * num_slots
    slot_centroids: List[Optional[Tuple[float, float]]] = [None] * num_slots

    if n_new == 0:
        return slot_masks, slot_centroids

    # Calcular centroides de las máscaras nuevas
    new_centroids = []
    for m in new_masks:
        ys, xs = np.where(m)
        if len(xs) == 0:
            new_centroids.append(None)
        else:
            new_centroids.append((float(xs.mean()), float(ys.mean())))

    # Matriz de distancias [n_new x num_slots]
    dist = np.full((n_new, num_slots), float("inf"))
    for i in range(n_new):
        if new_centroids[i] is None:
            continue
        for s in range(num_slots):
            if avg_centroids[s] is None:
                continue
            dx = new_centroids[i][0] - avg_centroids[s][0]
            dy = new_centroids[i][1] - avg_centroids[s][1]
            dist[i, s] = np.sqrt(dx * dx + dy * dy)

    # Greedy assignment
    used_new = set()
    used_slots = set()
    while len(used_slots) < num_slots and len(used_new) < n_new:
        masked = dist.copy()
        for i in used_new:
            masked[i, :] = float("inf")
        for s in used_slots:
            masked[:, s] = float("inf")

        if not np.isfinite(masked).any():
            break

        i, s = np.unravel_index(np.argmin(masked), masked.shape)
        if not np.isfinite(masked[i, s]):
            break

        slot_masks[s] = new_masks[i]
        slot_centroids[s] = new_centroids[i]
        used_new.add(i)
        used_slots.add(s)

    return slot_masks, slot_centroids


# ============================================================================
# STRATEGIES — cómo inicializar la nueva video session
# ============================================================================

def try_init_with_masks(
    processor,
    session,
    slot_masks: List[Optional[np.ndarray]],
    frame_idx: int = 0,
) -> bool:
    """Estrategia 1: inicializar la nueva sesión con máscaras frescas.

    Returns:
        True si se logró, False si el API no lo soporta o falla.
    """
    try:
        # SAM3 video session supports add_input_mask
        for slot_idx, mask in enumerate(slot_masks):
            if mask is None:
                continue
            # Convert to torch tensor (H, W) float
            mask_tensor = torch.from_numpy(mask.astype(np.float32))
            session = processor.add_mask_input(
                inference_session=session,
                frame_idx=frame_idx,
                obj_id=slot_idx,
                mask=mask_tensor,
            )
        return True
    except (AttributeError, TypeError) as e:
        logger.debug("Mask init not supported by SAM3 API: %s", e)
        return False
    except Exception as e:
        logger.warning("Mask init failed: %s", e)
        return False


def try_init_with_points(
    processor,
    session,
    slot_masks: List[Optional[np.ndarray]],
    frame_idx: int = 0,
    points_per_mask: int = 7,
) -> bool:
    """Estrategia 2: inicializar con puntos inteligentes extraídos de las máscaras.

    Returns:
        True si se logró, False si falla.
    """
    try:
        for slot_idx, mask in enumerate(slot_masks):
            if mask is None:
                continue
            points = mask_to_smart_points(mask, num_points=points_per_mask)
            if not points:
                continue
            # Labels: 1 = foreground (positive prompt)
            point_coords = np.array(points)
            point_labels = np.ones(len(points), dtype=np.int64)
            session = processor.add_point_input(
                inference_session=session,
                frame_idx=frame_idx,
                obj_id=slot_idx,
                input_points=point_coords,
                input_labels=point_labels,
            )
        return True
    except (AttributeError, TypeError) as e:
        logger.debug("Points init not supported by SAM3 API: %s", e)
        return False
    except Exception as e:
        logger.warning("Points init failed: %s", e)
        return False


def try_init_with_bboxes(
    processor,
    session,
    slot_masks: List[Optional[np.ndarray]],
    frame_idx: int = 0,
) -> bool:
    """Estrategia 3 (fallback final): inicializar con bboxes derivados de las máscaras.

    Returns:
        True si se logró, False si falla.
    """
    try:
        for slot_idx, mask in enumerate(slot_masks):
            if mask is None:
                continue
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            input_boxes = np.array([list(bbox)], dtype=np.float32)
            session = processor.add_box_input(
                inference_session=session,
                frame_idx=frame_idx,
                obj_id=slot_idx,
                input_boxes=input_boxes,
            )
        return True
    except Exception as e:
        logger.warning("Bbox init failed: %s", e)
        return False


def init_new_session_cascade(
    processor,
    session,
    slot_masks: List[Optional[np.ndarray]],
    frame_idx: int = 0,
    try_masks: bool = True,
    try_points: bool = True,
    try_bboxes: bool = True,
    points_per_mask: int = 7,
) -> Tuple[bool, str]:
    """Intenta inicializar la sesión con las 3 estrategias en cascada.

    Returns:
        (success: bool, strategy_used: str)
    """
    if try_masks:
        if try_init_with_masks(processor, session, slot_masks, frame_idx):
            return True, "masks"

    if try_points:
        if try_init_with_points(processor, session, slot_masks, frame_idx, points_per_mask):
            return True, "points"

    if try_bboxes:
        if try_init_with_bboxes(processor, session, slot_masks, frame_idx):
            return True, "bboxes"

    return False, "none"