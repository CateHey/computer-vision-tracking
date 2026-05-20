"""
Failure detection for SAM3 reset pipeline.

Detects 3 types of tracking failures:
  1. Area growth >250% — mask "exploded" (likely captured background)
  2. Area shrink <10% — mask "collapsed" (likely lost the rat)
  3. Overlap >90% between any pair of masks — identity swap or merged tracking

When any condition is met, the pipeline triggers a reset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASS PARA REPORTAR FALLOS
# ============================================================================

@dataclass
class FailureReport:
    """Reporte de un fallo detectado en un frame."""
    is_failed: bool
    reason: str                              # "area_growth", "area_shrink", "overlap", "none"
    affected_slots: List[int]                # qué ratas están involucradas
    area_before: Optional[float] = None
    area_after: Optional[float] = None
    overlap_value: Optional[float] = None
    overlap_pair: Optional[Tuple[int, int]] = None

    def __str__(self) -> str:
        if not self.is_failed:
            return "OK"
        if self.reason in ("area_growth", "area_shrink"):
            return (f"FAIL {self.reason} slots={self.affected_slots} "
                    f"area: {self.area_before:.0f}→{self.area_after:.0f}")
        if self.reason == "overlap":
            return (f"FAIL overlap pair={self.overlap_pair} "
                    f"value={self.overlap_value:.2f}")
        return f"FAIL {self.reason}"


# ============================================================================
# DETECCIÓN DE FALLOS POR ÁREA
# ============================================================================

def detect_area_anomaly(
    prev_mask: Optional[np.ndarray],
    curr_mask: Optional[np.ndarray],
    slot_idx: int,
    growth_thr: float = 2.5,
    shrink_thr: float = 0.10,
    min_area: int = 100,
) -> FailureReport:
    """Detecta crecimiento o reducción anómala de área entre 2 frames.

    Args:
        prev_mask: máscara del frame anterior (puede ser None si no había)
        curr_mask: máscara del frame actual
        slot_idx: índice del slot (para el reporte)
        growth_thr: factor de crecimiento que considera fallo (default 2.5 = 250%)
        shrink_thr: factor de reducción que considera fallo (default 0.10 = 10%)
        min_area: área mínima para considerar la comparación válida

    Returns:
        FailureReport con is_failed=True si hay anomalía.
    """
    if prev_mask is None or curr_mask is None:
        return FailureReport(is_failed=False, reason="none", affected_slots=[])

    area_prev = int(prev_mask.sum())
    area_curr = int(curr_mask.sum())

    # Si alguna área es muy chica, ignorar (probablemente sin detección)
    if area_prev < min_area or area_curr < min_area:
        return FailureReport(is_failed=False, reason="none", affected_slots=[])

    ratio = area_curr / area_prev

    if ratio > growth_thr:
        return FailureReport(
            is_failed=True,
            reason="area_growth",
            affected_slots=[slot_idx],
            area_before=float(area_prev),
            area_after=float(area_curr),
        )
    if ratio < shrink_thr:
        return FailureReport(
            is_failed=True,
            reason="area_shrink",
            affected_slots=[slot_idx],
            area_before=float(area_prev),
            area_after=float(area_curr),
        )

    return FailureReport(is_failed=False, reason="none", affected_slots=[])


# ============================================================================
# DETECCIÓN DE OVERLAP ENTRE MÁSCARAS
# ============================================================================

def detect_overlap(
    masks: List[Optional[np.ndarray]],
    overlap_thr: float = 0.90,
) -> FailureReport:
    """Detecta si dos máscaras tienen overlap >threshold (relativo a la más chica).

    Para N ratas, chequea todos los pares (N*(N-1)/2 pares).
    Para cada par, mide overlap en ambas direcciones y toma el máximo.

    Args:
        masks: lista de máscaras (puede contener None)
        overlap_thr: umbral de overlap (0.90 = 90%)

    Returns:
        FailureReport con la PRIMERA pareja conflictiva encontrada (si hay).
    """
    n = len(masks)
    for i in range(n):
        for j in range(i + 1, n):
            mi = masks[i]
            mj = masks[j]
            if mi is None or mj is None:
                continue

            inter = np.logical_and(mi, mj).sum()
            if inter == 0:
                continue

            area_i = int(mi.sum())
            area_j = int(mj.sum())
            if area_i == 0 or area_j == 0:
                continue

            # Overlap relativo: ¿qué porcentaje de mi está dentro de mj?
            # Tomamos el máximo de las 2 direcciones
            overlap_i_in_j = inter / area_i
            overlap_j_in_i = inter / area_j
            max_overlap = max(overlap_i_in_j, overlap_j_in_i)

            if max_overlap >= overlap_thr:
                return FailureReport(
                    is_failed=True,
                    reason="overlap",
                    affected_slots=[i, j],
                    overlap_value=float(max_overlap),
                    overlap_pair=(i, j),
                )

    return FailureReport(is_failed=False, reason="none", affected_slots=[])


# ============================================================================
# DETECTOR PRINCIPAL — combina todas las reglas
# ============================================================================

def detect_frame_failure(
    prev_masks: List[Optional[np.ndarray]],
    curr_masks: List[Optional[np.ndarray]],
    growth_thr: float = 2.5,
    shrink_thr: float = 0.10,
    overlap_thr: float = 0.90,
    min_area: int = 100,
) -> FailureReport:
    """Combina las 3 reglas de detección. Si alguna se cumple → reporta fallo.

    Args:
        prev_masks: máscaras del frame anterior por slot
        curr_masks: máscaras del frame actual por slot
        growth_thr, shrink_thr, overlap_thr, min_area: parámetros configurables

    Returns:
        FailureReport. is_failed=True si cualquier regla se activó.
        La primera regla que falla determina reason.
    """
    # Regla 1 y 2: área anómala (por cada slot)
    n = max(len(prev_masks), len(curr_masks))
    for slot_idx in range(n):
        prev = prev_masks[slot_idx] if slot_idx < len(prev_masks) else None
        curr = curr_masks[slot_idx] if slot_idx < len(curr_masks) else None

        report = detect_area_anomaly(
            prev, curr, slot_idx,
            growth_thr=growth_thr,
            shrink_thr=shrink_thr,
            min_area=min_area,
        )
        if report.is_failed:
            return report

    # Regla 3: overlap entre pares
    report = detect_overlap(curr_masks, overlap_thr=overlap_thr)
    if report.is_failed:
        return report

    return FailureReport(is_failed=False, reason="none", affected_slots=[])


# ============================================================================
# HISTORIA DE FRAMES SANOS (para promedio de centroides)
# ============================================================================

class HealthyFrameHistory:
    """Buffer circular de los últimos K frames SANOS (sin fallo).

    Se usa para calcular el centroide promedio de cada rata como referencia
    para reasignar identidades después de un reset.
    """

    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        # Lista de listas de centroides (uno por frame)
        # cada elemento es: list of Optional[(x,y)] de longitud num_slots
        self._buffer: List[List[Optional[Tuple[float, float]]]] = []

    def push(self, centroids: List[Optional[Tuple[float, float]]]) -> None:
        """Agrega un frame sano al buffer."""
        self._buffer.append(list(centroids))
        if len(self._buffer) > self.lookback:
            self._buffer.pop(0)

    def get_average_centroids(self, num_slots: int) -> List[Optional[Tuple[float, float]]]:
        """Calcula centroide promedio de cada slot en los últimos K frames sanos.

        Si un slot no tiene datos válidos, retorna None para ese slot.
        """
        if not self._buffer:
            return [None] * num_slots

        result: List[Optional[Tuple[float, float]]] = []
        for slot in range(num_slots):
            xs, ys = [], []
            for frame_centroids in self._buffer:
                if slot < len(frame_centroids) and frame_centroids[slot] is not None:
                    xs.append(frame_centroids[slot][0])
                    ys.append(frame_centroids[slot][1])
            if xs:
                result.append((sum(xs) / len(xs), sum(ys) / len(ys)))
            else:
                result.append(None)
        return result

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()