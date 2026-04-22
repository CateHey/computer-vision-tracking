"""
Contact Tracker v2 — Hysteresis-based rule engine with soft multi-label scoring.

Reemplaza la lógica winner-takes-all del contacts.py original con:
  - Umbrales duales (Schmitt triggers) para eliminar flicker en los bordes
  - Soft scores por tipo de contacto (multi-label concurrente)
  - Body length robusto (percentil 80 estilo DeepOF, no EMA)
  - Ventanas temporales multi-escala estilo JAABA (100ms, 400ms, 700ms)
  - Orientación en FOL (siguiendo DeepOF following_path)

Mantiene interfaz compatible con contacts.py original:
  - ContactTrackerV2.update(detections, masks, centroids, frame_idx)
  - ContactTrackerV2.finalize() -> Dict
  - Genera los mismos archivos de salida + columnas soft_score_X añadidas

Referencias:
  - DeepOF annotation_utils.py (Miranda et al. 2023, JOSS)
  - JAABA windowed features (Kabra et al. 2013, Nature Methods)
  - Schmitt trigger hysteresis (classic signal processing)
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from src.common.utils import Detection

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1 — DATACLASSES Y ENUMS
# ============================================================================

class ContactType(str, Enum):
    """Los 6 tipos de contacto social + NONE."""
    NONE = "none"
    N2N = "N2N"      # nose-to-nose
    N2AG = "N2AG"    # nose-to-anogenital
    T2T = "T2T"      # tail-to-tail
    FOL = "FOL"      # following
    SBS = "SBS"      # side-by-side
    N2B = "N2B"      # nose-to-body

    @classmethod
    def all_contact_types(cls) -> List["ContactType"]:
        """Todos los tipos menos NONE (útil para iterar sobre scores)."""
        return [cls.N2N, cls.N2AG, cls.T2T, cls.FOL, cls.SBS, cls.N2B]


class Zone(str, Enum):
    """Zona espacial entre pares de animales."""
    CONTACT = "contact"
    PROXIMITY = "proximity"
    INDEPENDENT = "independent"


@dataclass
class ScoreMap:
    """Scores continuos [0, 1] por tipo de contacto en UN frame para UN par."""
    n2n: float = 0.0
    n2ag: float = 0.0
    t2t: float = 0.0
    fol: float = 0.0
    sbs: float = 0.0
    n2b: float = 0.0

    def get(self, contact_type: ContactType) -> float:
        """Acceso por tipo."""
        mapping = {
            ContactType.N2N: self.n2n,
            ContactType.N2AG: self.n2ag,
            ContactType.T2T: self.t2t,
            ContactType.FOL: self.fol,
            ContactType.SBS: self.sbs,
            ContactType.N2B: self.n2b,
        }
        return mapping.get(contact_type, 0.0)

    def argmax_type(
        self,
        activation_threshold: float = 0.5,
        rare_threshold: float = 0.35,
        rare_types: Optional[List[ContactType]] = None,
    ) -> ContactType:
        """Retorna el tipo dominante si supera el umbral, sino NONE.

        Tipos "raros" (como FOL) usan un umbral más bajo para ser más sensibles.
        """
        if rare_types is None:
            rare_types = [ContactType.FOL]

        # Prioridad: tipos específicos antes que N2B (N2B es catch-all)
        priority_order = [
            ContactType.N2N,
            ContactType.N2AG,
            ContactType.T2T,
            ContactType.FOL,
            ContactType.SBS,
            ContactType.N2B,
        ]

        best_type = ContactType.NONE
        best_score = 0.0

        for ct in priority_order:
            score = self.get(ct)
            threshold = rare_threshold if ct in rare_types else activation_threshold
            if score >= threshold and score > best_score:
                best_score = score
                best_type = ct

        return best_type

    def active_types(self, threshold: float = 0.5) -> List[ContactType]:
        """Retorna TODOS los tipos activos (para análisis multi-label)."""
        return [ct for ct in ContactType.all_contact_types() if self.get(ct) >= threshold]

    def max_score(self) -> float:
        """El score más alto en este frame."""
        return max(self.n2n, self.n2ag, self.t2t, self.fol, self.sbs, self.n2b)

    def to_dict(self) -> Dict[str, float]:
        """Serialización con prefijo 'score_' para CSV."""
        return {
            "score_n2n": round(self.n2n, 4),
            "score_n2ag": round(self.n2ag, 4),
            "score_t2t": round(self.t2t, 4),
            "score_fol": round(self.fol, 4),
            "score_sbs": round(self.sbs, 4),
            "score_n2b": round(self.n2b, 4),
        }


@dataclass
class ContactEvent:
    """Evento de contacto por par por frame."""
    frame_idx: int
    time_sec: float
    pair_key: str

    # Scores continuos (NUEVO en v2)
    scores: ScoreMap = field(default_factory=ScoreMap)

    # Compatibilidad v1: tipo discreto + zona
    contact_type: ContactType = ContactType.NONE
    zone: Zone = Zone.INDEPENDENT

    # Métricas geométricas (todas normalizadas en body lengths)
    nose_nose_dist_bl: float = float("inf")
    centroid_dist_bl: float = float("inf")
    nose_tailbase_ij_bl: float = float("inf")
    nose_tailbase_ji_bl: float = float("inf")
    tail_tail_dist_bl: float = float("inf")
    mask_iou: float = 0.0

    # Cinemática
    velocity_i_bls: float = 0.0
    velocity_j_bls: float = 0.0
    velocity_alignment_cos: float = 0.0
    orientation_alignment_cos: float = 0.0

    # Body lengths (en píxeles, por si se necesita)
    body_length_i_px: float = 0.0
    body_length_j_px: float = 0.0

    # Rol asimétrico
    investigator_role: Optional[str] = None

    # Bout tracking
    bout_id: Optional[str] = None

    # Flags de calidad
    stale_keypoints: bool = False
    high_mask_overlap: bool = False
    missing_keypoints: bool = False
    single_detection: bool = False
    merged_state: bool = False

    def to_csv_row(self) -> Dict[str, Any]:
        """Convierte a dict para escribir al CSV."""
        row = {
            "frame_idx": self.frame_idx,
            "time_sec": round(self.time_sec, 3),
            "pair_key": self.pair_key,
            "contact_type": self.contact_type.value,
            "zone": self.zone.value,
            "nose_nose_dist_bl": _fmt(self.nose_nose_dist_bl),
            "centroid_dist_bl": _fmt(self.centroid_dist_bl),
            "nose_tailbase_ij_bl": _fmt(self.nose_tailbase_ij_bl),
            "nose_tailbase_ji_bl": _fmt(self.nose_tailbase_ji_bl),
            "tail_tail_dist_bl": _fmt(self.tail_tail_dist_bl),
            "mask_iou": round(self.mask_iou, 4),
            "velocity_i_bls": round(self.velocity_i_bls, 4),
            "velocity_j_bls": round(self.velocity_j_bls, 4),
            "velocity_alignment_cos": round(self.velocity_alignment_cos, 4),
            "orientation_alignment_cos": round(self.orientation_alignment_cos, 4),
            "body_length_i_px": round(self.body_length_i_px, 2),
            "body_length_j_px": round(self.body_length_j_px, 2),
            "investigator_role": self.investigator_role or "",
            "bout_id": self.bout_id or "",
            "stale_keypoints": int(self.stale_keypoints),
            "high_mask_overlap": int(self.high_mask_overlap),
            "missing_keypoints": int(self.missing_keypoints),
            "single_detection": int(self.single_detection),
            "merged_state": int(self.merged_state),
        }
        # Añadir scores con prefijo
        row.update(self.scores.to_dict())
        return row


@dataclass
class Bout:
    """Episodio continuo del mismo tipo de contacto."""
    bout_id: str
    pair_key: str
    contact_type: ContactType
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    n_frames: int = 0

    # Métricas acumuladas (media durante el bout)
    mean_nose_nose_dist_bl: float = 0.0
    mean_centroid_dist_bl: float = 0.0
    mean_mask_iou: float = 0.0
    mean_velocity_i_bls: float = 0.0
    mean_velocity_j_bls: float = 0.0
    peak_score: float = 0.0

    # Acumuladores internos (no se serializan)
    _sum_nose_nose: float = 0.0
    _sum_centroid: float = 0.0
    _sum_mask_iou: float = 0.0
    _sum_velocity_i: float = 0.0
    _sum_velocity_j: float = 0.0
    _count_valid_nose_nose: int = 0
    _count_valid_centroid: int = 0

    # Rol del investigador
    investigator_role: Optional[str] = None

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec

    def accumulate(self, event: ContactEvent) -> None:
        """Añade las métricas de un frame al bout."""
        self.n_frames += 1
        self.end_frame = event.frame_idx
        self.end_time_sec = event.time_sec

        # Peak score
        score = event.scores.get(self.contact_type)
        if score > self.peak_score:
            self.peak_score = score

        # Distancias (solo si son finitas)
        if math.isfinite(event.nose_nose_dist_bl):
            self._sum_nose_nose += event.nose_nose_dist_bl
            self._count_valid_nose_nose += 1
        if math.isfinite(event.centroid_dist_bl):
            self._sum_centroid += event.centroid_dist_bl
            self._count_valid_centroid += 1

        self._sum_mask_iou += event.mask_iou
        self._sum_velocity_i += event.velocity_i_bls
        self._sum_velocity_j += event.velocity_j_bls

    def finalize_metrics(self) -> None:
        """Calcula medias al cerrar el bout."""
        if self._count_valid_nose_nose > 0:
            self.mean_nose_nose_dist_bl = self._sum_nose_nose / self._count_valid_nose_nose
        if self._count_valid_centroid > 0:
            self.mean_centroid_dist_bl = self._sum_centroid / self._count_valid_centroid
        if self.n_frames > 0:
            self.mean_mask_iou = self._sum_mask_iou / self.n_frames
            self.mean_velocity_i_bls = self._sum_velocity_i / self.n_frames
            self.mean_velocity_j_bls = self._sum_velocity_j / self.n_frames

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "bout_id": self.bout_id,
            "pair_key": self.pair_key,
            "contact_type": self.contact_type.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "n_frames": self.n_frames,
            "mean_nose_nose_dist_bl": _fmt(self.mean_nose_nose_dist_bl),
            "mean_centroid_dist_bl": _fmt(self.mean_centroid_dist_bl),
            "mean_mask_iou": round(self.mean_mask_iou, 4),
            "mean_velocity_i_bls": round(self.mean_velocity_i_bls, 4),
            "mean_velocity_j_bls": round(self.mean_velocity_j_bls, 4),
            "peak_score": round(self.peak_score, 4),
            "investigator_role": self.investigator_role or "",
        }


def _fmt(x: float) -> float:
    """Formato limpio para CSV (infinito → NaN-like)."""
    if not math.isfinite(x):
        return -1.0
    return round(x, 4)


# ============================================================================
# SECTION 2 — GEOMETRY HELPERS
# ============================================================================

def euclidean(
    p1: Optional[Tuple[float, float]],
    p2: Optional[Tuple[float, float]],
) -> float:
    """Distancia euclidiana 2D. Retorna inf si alguno es None."""
    if p1 is None or p2 is None:
        return float("inf")
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_keypoint(
    det: Optional[Detection],
    name: str,
    min_conf: float = 0.3,
) -> Optional[Tuple[float, float]]:
    """Extrae keypoint por NOMBRE si supera confianza.

    Robusto a cambios en el orden de keypoints (a diferencia del v1 que
    usaba índices).
    """
    if det is None or det.keypoints is None:
        return None
    for kp in det.keypoints:
        if kp.name == name and kp.conf >= min_conf:
            return (kp.x, kp.y)
    return None


def body_orientation(
    det: Optional[Detection],
    min_conf: float = 0.3,
) -> Optional[Tuple[float, float]]:
    """Vector unitario tail_base → nose (dirección del cuerpo)."""
    nose = get_keypoint(det, "nose", min_conf)
    tail = get_keypoint(det, "tail_base", min_conf)
    if nose is None or tail is None:
        return None
    dx = nose[0] - tail[0]
    dy = nose[1] - tail[1]
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        return None
    return (dx / mag, dy / mag)


def cos_angle(
    v1: Optional[Tuple[float, float]],
    v2: Optional[Tuple[float, float]],
) -> float:
    """Coseno entre dos vectores. Retorna 0 si alguno es nulo."""
    if v1 is None or v2 is None:
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if m1 < 1e-6 or m2 < 1e-6:
        return 0.0
    return dot / (m1 * m2)


def valid_keypoint_count(det: Optional[Detection], min_conf: float = 0.3) -> int:
    """Cuenta keypoints con confianza suficiente."""
    if det is None or det.keypoints is None:
        return 0
    return sum(1 for kp in det.keypoints if kp.conf >= min_conf)


# ============================================================================
# SECTION 3 — BODY LENGTH ROBUSTO (estilo DeepOF percentil 80)
# ============================================================================

class BodyLengthEstimator:
    """Estimador de body length usando percentil 80 de observaciones válidas.

    Más robusto que EMA porque:
      - No se sesga por frames donde el animal está encogido/estirado
      - No se contamina si hay mal tracking puntual
      - Percentil 80 (no máximo) filtra outliers por arriba

    Implementación: mantiene buffer circular de últimas N observaciones
    válidas y recalcula el percentil cada X frames (no cada frame por costo).
    """

    def __init__(
        self,
        slot_idx: int,
        min_conf: float = 0.5,
        percentile: float = 80.0,
        fallback_px: float = 120.0,
        warmup_frames: int = 25,
        buffer_size: int = 500,
        recompute_every: int = 10,
    ):
        self.slot_idx = slot_idx
        self.min_conf = min_conf
        self.percentile = percentile
        self.fallback_px = fallback_px
        self.warmup_frames = warmup_frames
        self.recompute_every = recompute_every

        self._observations: Deque[float] = deque(maxlen=buffer_size)
        self._cached_value: float = fallback_px
        self._frames_since_recompute: int = 0
        self._total_frames_seen: int = 0

    def observe(self, det: Optional[Detection]) -> None:
        """Acumula una observación si nose y tail_base son confiables."""
        self._total_frames_seen += 1
        nose = get_keypoint(det, "nose", self.min_conf)
        tail = get_keypoint(det, "tail_base", self.min_conf)
        if nose is None or tail is None:
            return

        dist = euclidean(nose, tail)
        if not math.isfinite(dist) or dist < 10.0:  # filtro básico
            return

        self._observations.append(dist)
        self._frames_since_recompute += 1

        if self._frames_since_recompute >= self.recompute_every and len(self._observations) >= 10:
            self._cached_value = float(np.percentile(list(self._observations), self.percentile))
            self._frames_since_recompute = 0

    def current(self) -> float:
        """Retorna body length actual (en píxeles)."""
        if self._total_frames_seen < self.warmup_frames or len(self._observations) < 10:
            return self.fallback_px
        return self._cached_value


# ============================================================================
# SECTION 4 — VELOCIDAD CON SAVITZKY-GOLAY
# ============================================================================

class VelocityEstimator:
    """Estimador de velocidad con suavizado Savitzky-Golay.

    Mejor que diferencia simple (menos ruido) y que EMA (no introduce lag).
    Mantiene buffer deslizante; calcula SG filter cada vez que hay datos.
    """

    def __init__(
        self,
        slot_idx: int,
        window_length: int = 11,
        polyorder: int = 3,
        fps: float = 25.0,
    ):
        if window_length % 2 == 0:
            window_length += 1  # SG requires odd
        self.slot_idx = slot_idx
        self.window_length = window_length
        self.polyorder = polyorder
        self.fps = fps

        # Buffer circular de posiciones
        self._buffer_x: Deque[float] = deque(maxlen=window_length)
        self._buffer_y: Deque[float] = deque(maxlen=window_length)
        self._last_valid: Optional[Tuple[float, float]] = None

    def update(self, centroid: Optional[Tuple[float, float]]) -> None:
        """Añade centroide al buffer. Llamar cada frame."""
        if centroid is not None:
            self._last_valid = centroid
            self._buffer_x.append(centroid[0])
            self._buffer_y.append(centroid[1])
        elif self._last_valid is not None:
            # Rellenar con última posición válida (evita saltos)
            self._buffer_x.append(self._last_valid[0])
            self._buffer_y.append(self._last_valid[1])

    def velocity(self) -> Tuple[float, float]:
        """Retorna (vx, vy) en píxeles/segundo."""
        if len(self._buffer_x) < self.window_length:
            # No suficientes muestras: usar diferencia simple
            if len(self._buffer_x) < 2:
                return (0.0, 0.0)
            dx = self._buffer_x[-1] - self._buffer_x[-2]
            dy = self._buffer_y[-1] - self._buffer_y[-2]
            return (dx * self.fps, dy * self.fps)

        try:
            x_arr = np.array(self._buffer_x)
            y_arr = np.array(self._buffer_y)
            # deriv=1: primera derivada. delta = 1/fps → unidades por segundo
            vx_arr = savgol_filter(x_arr, self.window_length, self.polyorder, deriv=1, delta=1.0 / self.fps)
            vy_arr = savgol_filter(y_arr, self.window_length, self.polyorder, deriv=1, delta=1.0 / self.fps)
            return (float(vx_arr[-1]), float(vy_arr[-1]))
        except Exception as e:
            logger.debug("SG filter failed: %s", e)
            return (0.0, 0.0)

    def speed(self) -> float:
        """Magnitud de la velocidad en píxeles/segundo."""
        vx, vy = self.velocity()
        return math.sqrt(vx * vx + vy * vy)


# ============================================================================
# SECTION 5 — SCHMITT TRIGGER (HISTÉRESIS)
# ============================================================================

class SchmittTrigger:
    """Umbral dual con histéresis — elimina flicker en oscilaciones.

    Semántica para DISTANCIAS (bajo = activo):
      - Si INACTIVO y signal < tau_high → ACTIVA
      - Si ACTIVO y signal > tau_low → DESACTIVA
      - tau_low > tau_high (banda muerta)

    Semántica para VELOCIDADES u otros (alto = activo, inverted=True):
      - Si INACTIVO y signal > tau_high → ACTIVA
      - Si ACTIVO y signal < tau_low → DESACTIVA
      - tau_low < tau_high
    """

    def __init__(
        self,
        tau_high: float,
        tau_low: float,
        initial_state: bool = False,
        inverted: bool = False,
    ):
        if inverted:
            if tau_low >= tau_high:
                raise ValueError(f"inverted: tau_low ({tau_low}) must be < tau_high ({tau_high})")
        else:
            if tau_low <= tau_high:
                raise ValueError(f"normal: tau_low ({tau_low}) must be > tau_high ({tau_high})")

        self.tau_high = tau_high
        self.tau_low = tau_low
        self.inverted = inverted
        self._state: bool = initial_state

    def update(self, value: float) -> bool:
        """Procesa un valor y retorna True si está activo."""
        if not math.isfinite(value):
            # Valores inválidos no cambian el estado
            return self._state

        if self.inverted:
            if not self._state and value > self.tau_high:
                self._state = True
            elif self._state and value < self.tau_low:
                self._state = False
        else:
            if not self._state and value < self.tau_high:
                self._state = True
            elif self._state and value > self.tau_low:
                self._state = False

        return self._state

    def is_active(self) -> bool:
        return self._state

    def reset(self) -> None:
        self._state = False


# ============================================================================
# SECTION 6 — SOFT SCORING (funciones fuzzy-like)
# ============================================================================

def trapezoidal_score(x: float, a: float, b: float, c: float, d: float) -> float:
    """Membership trapezoidal — transición suave entre umbrales.

    Forma: score 0 fuera de [a, d], sube de a→b, meseta b→c, baja c→d.

    Args:
        x: valor a evaluar
        a: inicio de la subida (score = 0)
        b: fin de la subida / inicio meseta (score = 1)
        c: fin meseta / inicio bajada (score = 1)
        d: fin bajada (score = 0)

    Requiere a <= b <= c <= d.
    """
    if not math.isfinite(x):
        return 0.0
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / max(b - a, 1e-9)
    # c < x < d
    return (d - x) / max(d - c, 1e-9)


def reversed_trapezoidal_score(x: float, near: float, far: float) -> float:
    """Membership invertida — 1 para valores BAJOS, 0 para altos.

    Transición lineal suave entre 'near' (score=1) y 'far' (score=0).
    Uso típico: soft score de proximidad (distancia baja = alto score).

    Args:
        x: valor (ej. distancia)
        near: debajo de esto, score = 1
        far: arriba de esto, score = 0
    """
    if not math.isfinite(x):
        return 0.0
    if x <= near:
        return 1.0
    if x >= far:
        return 0.0
    return (far - x) / max(far - near, 1e-9)


def logistic_score(x: float, midpoint: float, steepness: float = 10.0) -> float:
    """Sigmoide — alternativa suave a trapezoidal, más natural.

    score = 1 / (1 + exp(steepness * (x - midpoint)))

    - steepness positivo: score alto para x < midpoint (como proximidad)
    - steepness negativo: score alto para x > midpoint (como velocidad)
    """
    if not math.isfinite(x):
        return 0.0
    try:
        return 1.0 / (1.0 + math.exp(steepness * (x - midpoint)))
    except OverflowError:
        return 0.0 if steepness * (x - midpoint) > 0 else 1.0


def ramp_up_score(x: float, low: float, high: float) -> float:
    """Rampa ascendente — 0 abajo de 'low', 1 arriba de 'high', lineal entre medio.

    Uso típico: activación basada en velocidad (más rápido = más seguro que hay FOL).
    """
    if not math.isfinite(x):
        return 0.0
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / max(high - low, 1e-9)
# ============================================================================
# SECTION 7 — CONTACT CLASSIFIER (un clasificador por par)
# ============================================================================

class ContactClassifier:
    """Clasificador para UN par de animales.

    Mantiene estado entre frames:
      - Schmitt triggers por tipo (para estabilidad binaria interna)
      - Buffer reciente de posiciones de tail_base del "followed" (para FOL)

    La clasificación es multi-label: todos los scores se computan
    independientemente en el rango [0, 1]. El contact_type final es
    el argmax que supere el umbral de activación.
    """

    # Tamaño del buffer de posiciones para following_path (DeepOF usa 0.5s)
    FOLLOW_PATH_BUFFER_FRAMES = 12  # 0.5s a 25fps

    def __init__(self, pair_key: str, slot_i: int, slot_j: int, config: Dict[str, Any]):
        """
        Args:
            pair_key: identificador del par (ej. "0_1")
            slot_i, slot_j: índices de los dos animales
            config: dict con umbrales (sección "contacts" del YAML)
        """
        self.pair_key = pair_key
        self.slot_i = slot_i
        self.slot_j = slot_j
        self.config = config

        # Extraer umbrales
        self.contact_near = float(config.get("contact_zone_bl_enter", 0.30))
        self.contact_far = float(config.get("contact_zone_bl_exit", 0.45))
        self.proximity_bl = float(config.get("proximity_zone_bl", 1.0))
        self.min_kp_conf = float(config.get("min_keypoint_conf", 0.3))

        # SBS
        self.sbs_iou_enter = float(config.get("sbs_mask_iou_enter", 0.05))
        self.sbs_iou_exit = float(config.get("sbs_mask_iou_exit", 0.02))
        self.sbs_max_speed_bls = float(config.get("sbs_max_velocity_bls", 0.5))  # unidades BL/s
        self.sbs_parallel_cos_min = float(config.get("sbs_parallel_cos_min", 0.7))

        # FOL (Following) — más permisivo que v1, inspirado en DeepOF
        self.follow_near_bl = float(config.get("follow_radius_bl", 0.4))
        self.follow_far_bl = float(config.get("follow_radius_bl_exit", 0.6))
        self.follow_min_speed_bls = float(config.get("follow_min_speed_bls", 0.15))
        self.follow_alignment_cos = float(config.get("follow_alignment_cos", 0.6))

        # Overlap mask warning
        self.mask_overlap_warning = float(config.get("mask_overlap_warning", 0.5))

        # Schmitt triggers internos (uno por tipo)
        # Para distancias: tau_high (entrada, más pequeño) < tau_low (salida, más grande)
        self.trig_n2n = SchmittTrigger(self.contact_near, self.contact_far)
        self.trig_t2t = SchmittTrigger(self.contact_near, self.contact_far)
        self.trig_n2ag_ij = SchmittTrigger(self.contact_near, self.contact_far)
        self.trig_n2ag_ji = SchmittTrigger(self.contact_near, self.contact_far)
        # Para FOL: trigger sobre distancia nose↔tailbase del path
        self.trig_fol_ij = SchmittTrigger(self.follow_near_bl, self.follow_far_bl)
        self.trig_fol_ji = SchmittTrigger(self.follow_near_bl, self.follow_far_bl)
        # Para SBS: trigger sobre IoU (inverted: alto IoU = activo)
        self.trig_sbs = SchmittTrigger(
            tau_high=self.sbs_iou_enter,
            tau_low=self.sbs_iou_exit,
            inverted=True,
        )

        # Buffer de posiciones de tail_base para cada animal (para FOL path-based)
        self._path_i: Deque[Optional[Tuple[float, float]]] = deque(
            maxlen=self.FOLLOW_PATH_BUFFER_FRAMES
        )
        self._path_j: Deque[Optional[Tuple[float, float]]] = deque(
            maxlen=self.FOLLOW_PATH_BUFFER_FRAMES
        )

    def classify(
        self,
        det_i: Optional[Detection],
        det_j: Optional[Detection],
        mask_i: Optional[np.ndarray],
        mask_j: Optional[np.ndarray],
        centroid_i: Optional[Tuple[float, float]],
        centroid_j: Optional[Tuple[float, float]],
        velocity_i: Tuple[float, float],
        velocity_j: Tuple[float, float],
        body_length_i: float,
        body_length_j: float,
        frame_idx: int,
        time_sec: float,
    ) -> ContactEvent:
        """Procesa un frame y retorna el evento completo."""

        event = ContactEvent(
            frame_idx=frame_idx,
            time_sec=time_sec,
            pair_key=self.pair_key,
            body_length_i_px=body_length_i,
            body_length_j_px=body_length_j,
        )

        # Flags de calidad
        if det_i is None or det_j is None:
            event.single_detection = True
        if valid_keypoint_count(det_i, self.min_kp_conf) < 2 or valid_keypoint_count(det_j, self.min_kp_conf) < 2:
            event.missing_keypoints = True

        # Extraer keypoints
        nose_i = get_keypoint(det_i, "nose", self.min_kp_conf)
        nose_j = get_keypoint(det_j, "nose", self.min_kp_conf)
        tail_i = get_keypoint(det_i, "tail_base", self.min_kp_conf)
        tail_j = get_keypoint(det_j, "tail_base", self.min_kp_conf)

        # Actualizar buffer de tail paths (usa tail_base si hay, sino centroid)
        self._path_i.append(tail_i if tail_i is not None else centroid_i)
        self._path_j.append(tail_j if tail_j is not None else centroid_j)

        # Usar el body length mínimo del par para normalización (más conservador)
        # Evita scores altos cuando un animal es mucho más grande
        bl_ref = max(min(body_length_i, body_length_j), 1.0)

        # --- Distancias geométricas ---
        d_nose_nose = euclidean(nose_i, nose_j)
        d_centroid = euclidean(centroid_i, centroid_j)
        d_nose_i_tail_j = euclidean(nose_i, tail_j)
        d_nose_j_tail_i = euclidean(nose_j, tail_i)
        d_tail_tail = euclidean(tail_i, tail_j)

        # Normalizar
        event.nose_nose_dist_bl = d_nose_nose / bl_ref if math.isfinite(d_nose_nose) else float("inf")
        event.centroid_dist_bl = d_centroid / bl_ref if math.isfinite(d_centroid) else float("inf")
        event.nose_tailbase_ij_bl = d_nose_i_tail_j / bl_ref if math.isfinite(d_nose_i_tail_j) else float("inf")
        event.nose_tailbase_ji_bl = d_nose_j_tail_i / bl_ref if math.isfinite(d_nose_j_tail_i) else float("inf")
        event.tail_tail_dist_bl = d_tail_tail / bl_ref if math.isfinite(d_tail_tail) else float("inf")

        # --- Zona ---
        event.zone = self._determine_zone(event.centroid_dist_bl)

        # --- Mask IoU ---
        if mask_i is not None and mask_j is not None:
            inter = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            event.mask_iou = float(inter / union) if union > 0 else 0.0
            if event.mask_iou > self.mask_overlap_warning:
                event.high_mask_overlap = True

        # --- Cinemática ---
        speed_i_px = math.sqrt(velocity_i[0] ** 2 + velocity_i[1] ** 2)
        speed_j_px = math.sqrt(velocity_j[0] ** 2 + velocity_j[1] ** 2)
        event.velocity_i_bls = speed_i_px / bl_ref if bl_ref > 0 else 0.0
        event.velocity_j_bls = speed_j_px / bl_ref if bl_ref > 0 else 0.0
        event.velocity_alignment_cos = cos_angle(velocity_i, velocity_j)

        orient_i = body_orientation(det_i, self.min_kp_conf)
        orient_j = body_orientation(det_j, self.min_kp_conf)
        event.orientation_alignment_cos = cos_angle(orient_i, orient_j)

        # --- SCORES ---

        # 1. N2N — bilateral
        event.scores.n2n = self._score_n2n(event.nose_nose_dist_bl)

        # 2. N2AG — asimétrico (devuelve score + role)
        n2ag_score, n2ag_role = self._score_n2ag(
            event.nose_tailbase_ij_bl,
            event.nose_tailbase_ji_bl,
        )
        event.scores.n2ag = n2ag_score

        # 3. T2T
        event.scores.t2t = self._score_t2t(event.tail_tail_dist_bl)

        # 4. FOL — path-based, tolera followed quieto
        fol_score, fol_role = self._score_fol(
            nose_i, nose_j,
            centroid_i, centroid_j,
            orient_i, orient_j,
            event.velocity_i_bls, event.velocity_j_bls,
            bl_ref,
        )
        event.scores.fol = fol_score

        # 5. SBS
        event.scores.sbs = self._score_sbs(
            event.mask_iou,
            event.centroid_dist_bl,
            event.velocity_i_bls,
            event.velocity_j_bls,
            event.orientation_alignment_cos,
        )

        # 6. N2B — catch-all, con guardas contra N2N/N2AG
        n2b_score, n2b_role = self._score_n2b(
            nose_i, nose_j,
            mask_i, mask_j,
            event.nose_nose_dist_bl,
            event.nose_tailbase_ij_bl,
            event.nose_tailbase_ji_bl,
        )
        event.scores.n2b = n2b_score

        # --- Decidir contact_type dominante ---
        activation = float(self.config.get("activation_threshold", 0.5))
        rare = float(self.config.get("activation_threshold_rare", 0.35))
        event.contact_type = event.scores.argmax_type(
            activation_threshold=activation,
            rare_threshold=rare,
        )

        # --- Determinar investigator_role según el tipo ganador ---
        if event.contact_type == ContactType.N2AG:
            event.investigator_role = n2ag_role
        elif event.contact_type == ContactType.FOL:
            event.investigator_role = fol_role
        elif event.contact_type == ContactType.N2B:
            event.investigator_role = n2b_role

        return event

    def _determine_zone(self, centroid_dist_bl: float) -> Zone:
        """Decide en qué zona espacial está el par."""
        if not math.isfinite(centroid_dist_bl):
            return Zone.INDEPENDENT
        if centroid_dist_bl < self.contact_near:
            return Zone.CONTACT
        if centroid_dist_bl < self.proximity_bl:
            return Zone.PROXIMITY
        return Zone.INDEPENDENT

    # -------------------------- Scoring por tipo --------------------------

    def _score_n2n(self, nose_nose_bl: float) -> float:
        """Score de nose-to-nose. Schmitt trigger + soft score."""
        # El trigger da estabilidad temporal (evita flicker)
        active = self.trig_n2n.update(nose_nose_bl)
        # El soft score da continuidad
        soft = reversed_trapezoidal_score(nose_nose_bl, self.contact_near, self.contact_far)
        # Si el trigger está activo, el score mínimo es 0.5 (no cae bruscamente)
        if active:
            return max(soft, 0.5)
        return soft

    def _score_t2t(self, tail_tail_bl: float) -> float:
        """Score de tail-to-tail."""
        active = self.trig_t2t.update(tail_tail_bl)
        soft = reversed_trapezoidal_score(tail_tail_bl, self.contact_near, self.contact_far)
        if active:
            return max(soft, 0.5)
        return soft

    def _score_n2ag(
        self,
        nose_i_tail_j_bl: float,
        nose_j_tail_i_bl: float,
    ) -> Tuple[float, Optional[str]]:
        """Score de nose-to-anogenital + quién investiga.

        Retorna (score, role) donde role ∈ {"i", "j", None}.
        Evalúa las dos direcciones y retorna la más fuerte.
        """
        # Dirección i->j (nariz de i cerca de cola de j)
        active_ij = self.trig_n2ag_ij.update(nose_i_tail_j_bl)
        soft_ij = reversed_trapezoidal_score(nose_i_tail_j_bl, self.contact_near, self.contact_far)
        score_ij = max(soft_ij, 0.5) if active_ij else soft_ij

        # Dirección j->i
        active_ji = self.trig_n2ag_ji.update(nose_j_tail_i_bl)
        soft_ji = reversed_trapezoidal_score(nose_j_tail_i_bl, self.contact_near, self.contact_far)
        score_ji = max(soft_ji, 0.5) if active_ji else soft_ji

        if score_ij > score_ji:
            return score_ij, "i"
        elif score_ji > score_ij:
            return score_ji, "j"
        elif score_ij > 0:
            return score_ij, "i"  # empate: desempatamos arbitrariamente
        return 0.0, None

    def _score_fol(
        self,
        nose_i: Optional[Tuple[float, float]],
        nose_j: Optional[Tuple[float, float]],
        centroid_i: Optional[Tuple[float, float]],
        centroid_j: Optional[Tuple[float, float]],
        orient_i: Optional[Tuple[float, float]],
        orient_j: Optional[Tuple[float, float]],
        speed_i_bls: float,
        speed_j_bls: float,
        bl_ref: float,
    ) -> Tuple[float, Optional[str]]:
        """Score de following.

        Lógica inspirada en DeepOF following_path:
          1. follower debe moverse (speed > min)
          2. nariz del follower cerca del PATH reciente de tail_base del followed
             (no solo la posición actual — tolera followed quieto)
          3. follower orientado hacia el followed (body vector apunta en esa dirección)

        Evalúa las dos direcciones: i sigue a j, o j sigue a i.
        """
        score_ij, speed_ok_ij = self._score_fol_direction(
            nose_i, centroid_j, orient_i, speed_i_bls, self._path_j, bl_ref,
        )
        score_ji, speed_ok_ji = self._score_fol_direction(
            nose_j, centroid_i, orient_j, speed_j_bls, self._path_i, bl_ref,
        )

        # Aplicar Schmitt trigger a la distancia mínima al path
        # (el trigger ya está dentro de _score_fol_direction vía trig_fol_*)

        if score_ij > score_ji:
            return score_ij, "i"
        elif score_ji > score_ij:
            return score_ji, "j"
        elif score_ij > 0:
            return score_ij, "i"
        return 0.0, None

    def _score_fol_direction(
        self,
        nose_follower: Optional[Tuple[float, float]],
        centroid_followed: Optional[Tuple[float, float]],
        orient_follower: Optional[Tuple[float, float]],
        speed_follower_bls: float,
        path_followed: Deque[Optional[Tuple[float, float]]],
        bl_ref: float,
    ) -> Tuple[float, bool]:
        """Score de FOL en UNA dirección (follower → followed).

        Retorna (score [0,1], speed_ok).
        """
        if nose_follower is None or centroid_followed is None:
            return 0.0, False

        # 1. Speed check (solo follower, no followed)
        speed_ok = speed_follower_bls >= self.follow_min_speed_bls

        # 2. Distancia mínima del nose_follower al PATH del tail_base del followed
        min_dist = float("inf")
        for past_pos in path_followed:
            if past_pos is None:
                continue
            d = euclidean(nose_follower, past_pos)
            if d < min_dist:
                min_dist = d
        min_dist_bl = min_dist / bl_ref if math.isfinite(min_dist) else float("inf")

        # 3. Orientation check: body vector del follower apunta hacia el followed
        if orient_follower is not None:
            to_followed = (
                centroid_followed[0] - nose_follower[0],
                centroid_followed[1] - nose_follower[1],
            )
            to_followed_mag = math.sqrt(to_followed[0] ** 2 + to_followed[1] ** 2)
            if to_followed_mag > 1e-6:
                to_followed_norm = (
                    to_followed[0] / to_followed_mag,
                    to_followed[1] / to_followed_mag,
                )
                orient_cos = cos_angle(orient_follower, to_followed_norm)
            else:
                orient_cos = 0.0
        else:
            orient_cos = 0.0

        # --- Soft scoring ---
        # Proximidad al path
        dist_score = reversed_trapezoidal_score(
            min_dist_bl, self.follow_near_bl, self.follow_far_bl,
        )
        # Alineación con el target
        orient_score = ramp_up_score(orient_cos, self.follow_alignment_cos - 0.2, self.follow_alignment_cos + 0.2)
        # Speed (binario suave)
        speed_score = 1.0 if speed_ok else ramp_up_score(
            speed_follower_bls, 0.0, self.follow_min_speed_bls,
        )

        # Combinar multiplicativamente (todos deben ser altos)
        combined = dist_score * orient_score * speed_score
        return combined, speed_ok

    def _score_sbs(
        self,
        mask_iou: float,
        centroid_dist_bl: float,
        speed_i_bls: float,
        speed_j_bls: float,
        orientation_cos: float,
    ) -> float:
        """Score de side-by-side.

        Requisitos:
          - IoU de máscaras > umbral (cuerpos se tocan/solapan)
          - Baja velocidad (ambos casi quietos)
          - Orientación paralela (cos alto)
        """
        active = self.trig_sbs.update(mask_iou)

        # IoU score (más IoU = más probable SBS)
        iou_score = ramp_up_score(mask_iou, self.sbs_iou_exit, self.sbs_iou_enter * 3)
        # Cercanía centroides
        dist_score = reversed_trapezoidal_score(
            centroid_dist_bl, self.contact_near, self.proximity_bl,
        )
        # Baja velocidad (ambos)
        max_speed = max(speed_i_bls, speed_j_bls)
        speed_score = reversed_trapezoidal_score(
            max_speed, self.sbs_max_speed_bls * 0.5, self.sbs_max_speed_bls * 1.5,
        )
        # Alineación de cuerpos (paralelo)
        align_score = ramp_up_score(
            abs(orientation_cos),  # abs para aceptar paralelo o anti-paralelo
            self.sbs_parallel_cos_min - 0.15,
            self.sbs_parallel_cos_min + 0.15,
        )

        combined = iou_score * dist_score * speed_score * align_score
        if active:
            combined = max(combined, 0.5)
        return combined

    def _score_n2b(
        self,
        nose_i: Optional[Tuple[float, float]],
        nose_j: Optional[Tuple[float, float]],
        mask_i: Optional[np.ndarray],
        mask_j: Optional[np.ndarray],
        nose_nose_bl: float,
        nose_i_tail_j_bl: float,
        nose_j_tail_i_bl: float,
    ) -> Tuple[float, Optional[str]]:
        """Score de nose-to-body (catch-all).

        Verifica si la nariz de un animal está DENTRO de la máscara del otro.
        Aplica guardas: si ya hay N2N o N2AG claro, se reduce el score
        (el argmax general decidirá el ganador, pero evitamos que N2B domine).
        """
        # Guardia: si hay contacto específico ya cerca del umbral, reducir N2B
        # para que no absorba N2N/N2AG. argmax respetará prioridad.
        guard_factor = 1.0
        if nose_nose_bl < self.contact_far or nose_i_tail_j_bl < self.contact_far or nose_j_tail_i_bl < self.contact_far:
            guard_factor = 0.7  # no anula, solo penaliza

        score_i_in_j = 0.0
        if nose_i is not None and mask_j is not None:
            ix, iy = int(round(nose_i[0])), int(round(nose_i[1]))
            if 0 <= iy < mask_j.shape[0] and 0 <= ix < mask_j.shape[1]:
                if mask_j[iy, ix]:
                    score_i_in_j = 1.0

        score_j_in_i = 0.0
        if nose_j is not None and mask_i is not None:
            jx, jy = int(round(nose_j[0])), int(round(nose_j[1]))
            if 0 <= jy < mask_i.shape[0] and 0 <= jx < mask_i.shape[1]:
                if mask_i[jy, jx]:
                    score_j_in_i = 1.0

        score_i_in_j *= guard_factor
        score_j_in_i *= guard_factor

        if score_i_in_j > score_j_in_i:
            return score_i_in_j, "i"
        elif score_j_in_i > score_i_in_j:
            return score_j_in_i, "j"
        elif score_i_in_j > 0:
            return score_i_in_j, "i"
        return 0.0, None


# ============================================================================
# SECTION 8 — BOUT MANAGER
# ============================================================================

class BoutManager:
    """Gestiona apertura, extensión, gap-bridging y cierre de bouts.

    Un bout es un episodio continuo del mismo tipo de contacto para un par.
    Por lo tanto, la clave es (pair_key, contact_type).

    Lógica (similar a DeepOF + gap bridging):
      - Cada (par, tipo) tiene un bout abierto o None
      - Si llega evento del mismo tipo → extiende
      - Si cambia el tipo o llega NONE → cuenta frames de gap
      - Si gap < max_gap_frames → mantiene el bout abierto (puente)
      - Si gap >= max_gap_frames → cierra
      - Al cerrar, si duración < min_duration[type] → descarta
    """

    def __init__(self, fps: float, config: Dict[str, Any]):
        self.fps = fps

        # Defaults DeepOF-adaptados a 25fps
        default_min = int(fps / 4)  # 0.25s = 6 frames a 25fps
        default_min_follow = int(fps / 2)  # 0.5s = 12 frames (FOL más estricto)
        default_max_gap = 3

        self.max_gap_frames = int(config.get("bout_max_gap_frames", default_max_gap))

        # Duración mínima por tipo (en frames)
        self.min_duration_frames: Dict[ContactType, int] = {
            ContactType.N2N: int(config.get("bout_min_frames_n2n", default_min)),
            ContactType.N2AG: int(config.get("bout_min_frames_n2ag", default_min)),
            ContactType.T2T: int(config.get("bout_min_frames_t2t", default_min)),
            ContactType.FOL: int(config.get("bout_min_frames_fol", default_min_follow)),
            ContactType.SBS: int(config.get("bout_min_frames_sbs", default_min)),
            ContactType.N2B: int(config.get("bout_min_frames_n2b", default_min)),
        }

        # Estado: (pair_key, contact_type) -> Bout abierto
        self._open_bouts: Dict[Tuple[str, ContactType], Bout] = {}
        # Gap counter por (pair_key, contact_type): cuántos frames sin ver ese tipo
        self._gap_counter: Dict[Tuple[str, ContactType], int] = {}

        # Bouts cerrados y validados
        self._closed_bouts: List[Bout] = []

        # Counter para bout_ids
        self._bout_counter: int = 0

    def process_event(self, event: ContactEvent) -> None:
        """Procesa un evento. Modifica event.bout_id in-place si corresponde."""

        pair = event.pair_key
        current_type = event.contact_type

        # Paso 1: si hay un tipo activo este frame, lo extendemos o abrimos bout
        if current_type != ContactType.NONE:
            key = (pair, current_type)
            if key in self._open_bouts:
                # Extender bout existente
                bout = self._open_bouts[key]
                bout.accumulate(event)
                event.bout_id = bout.bout_id
                self._gap_counter[key] = 0  # reset gap
            else:
                # Abrir nuevo bout
                bout = self._open_new_bout(event)
                self._open_bouts[key] = bout
                self._gap_counter[key] = 0
                event.bout_id = bout.bout_id

        # Paso 2: incrementar gap counter para TODOS los tipos abiertos
        # que NO se vieron este frame
        to_close = []
        for key, bout in list(self._open_bouts.items()):
            pair_key, ct = key
            if pair_key != pair:
                continue  # solo afecta al par de este evento
            if ct == current_type:
                continue  # ya lo extendimos arriba
            # Este bout abierto no tuvo actividad este frame
            self._gap_counter[key] = self._gap_counter.get(key, 0) + 1
            if self._gap_counter[key] > self.max_gap_frames:
                to_close.append(key)

        # Cerrar bouts con gap excedido
        for key in to_close:
            self._close_bout(key)

    def close_all(self) -> List[Bout]:
        """Cierra todos los bouts pendientes. Llamar en finalize().

        Retorna lista de bouts válidos (ya filtrados por duración mínima).
        """
        for key in list(self._open_bouts.keys()):
            self._close_bout(key)
        return self._closed_bouts

    def _open_new_bout(self, event: ContactEvent) -> Bout:
        """Abre un nuevo bout."""
        self._bout_counter += 1
        bout_id = f"bout_{self._bout_counter:05d}_{event.contact_type.value}_{event.pair_key}"
        bout = Bout(
            bout_id=bout_id,
            pair_key=event.pair_key,
            contact_type=event.contact_type,
            start_frame=event.frame_idx,
            end_frame=event.frame_idx,
            start_time_sec=event.time_sec,
            end_time_sec=event.time_sec,
            investigator_role=event.investigator_role,
        )
        bout.accumulate(event)
        return bout

    def _close_bout(self, key: Tuple[str, ContactType]) -> None:
        """Cierra un bout. Lo guarda solo si supera duración mínima."""
        if key not in self._open_bouts:
            return
        bout = self._open_bouts.pop(key)
        self._gap_counter.pop(key, None)

        min_required = self.min_duration_frames.get(bout.contact_type, 6)
        if bout.n_frames >= min_required:
            bout.finalize_metrics()
            self._closed_bouts.append(bout)
            logger.debug(
                "Closed valid bout %s (%s, %d frames, %.2fs)",
                bout.bout_id, bout.contact_type.value, bout.n_frames, bout.duration_sec,
            )
        else:
            logger.debug(
                "Discarded short bout %s (%d frames < %d required)",
                bout.bout_id, bout.n_frames, min_required,
            )


# ============================================================================
# SECTION 9 — CONTACT TRACKER V2 (CLASE PRINCIPAL)
# ============================================================================

class ContactTrackerV2:
    """Nueva versión con hysteresis + soft scoring + N animales.

    Interfaz IDÉNTICA a v1 para compatibilidad con pipeline centroid.
    """

    def __init__(
        self,
        output_dir: Path,
        fps: float,
        num_slots: int,
        video_path: str,
        config: Dict[str, Any],
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.num_slots = num_slots
        self.video_path = video_path
        self.config = config.get("contacts", {}) if "contacts" in config else config

        # --- Estimadores por slot (para N animales) ---
        self.body_length_estimators: Dict[int, BodyLengthEstimator] = {
            i: BodyLengthEstimator(
                slot_idx=i,
                fallback_px=float(self.config.get("fallback_body_length_px", 120.0)),
            )
            for i in range(num_slots)
        }
        self.velocity_estimators: Dict[int, VelocityEstimator] = {
            i: VelocityEstimator(slot_idx=i, fps=fps)
            for i in range(num_slots)
        }

        # --- Classifiers por par (N animales → C(N,2) pares) ---
        self.classifiers: Dict[str, ContactClassifier] = {}
        for i in range(num_slots):
            for j in range(i + 1, num_slots):
                pair_key = f"{i}_{j}"
                self.classifiers[pair_key] = ContactClassifier(
                    pair_key=pair_key, slot_i=i, slot_j=j, config=self.config,
                )

        # --- BoutManager único (maneja todos los pares) ---
        self.bout_manager = BoutManager(fps=fps, config=self.config)

        # --- CSV writer para per-frame ---
        self._csv_path = self.output_dir / "contacts_per_frame.csv"
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_header_written = False

        # Guardar todos los eventos para poder reescribir el CSV con bout_ids correctos
        self._all_events: List[ContactEvent] = []

        # Estado
        self._frames_processed: int = 0
        self._first_frame_idx: Optional[int] = None
        self._last_frame_idx: Optional[int] = None

        logger.info(
            "ContactTrackerV2 initialized | slots=%d pairs=%d fps=%.1f outdir=%s",
            num_slots, len(self.classifiers), fps, self.output_dir,
        )

    # ------------------------- Interfaz pública -------------------------

    def update(
        self,
        detections: List[Detection],
        slot_masks: List[Optional[np.ndarray]],
        slot_centroids: List[Optional[Tuple[float, float]]],
        frame_idx: int,
    ) -> None:
        """Procesa un frame."""

        if self._first_frame_idx is None:
            self._first_frame_idx = frame_idx
        self._last_frame_idx = frame_idx

        time_sec = frame_idx / self.fps

        # 1. Actualizar estimadores de body length y velocidad por slot
        slot_dets = self._map_detections_to_slots(detections, slot_centroids)
        for slot_idx in range(self.num_slots):
            self.body_length_estimators[slot_idx].observe(slot_dets.get(slot_idx))
            self.velocity_estimators[slot_idx].update(slot_centroids[slot_idx] if slot_idx < len(slot_centroids) else None)

        # 2. Procesar cada par
        for pair_key, classifier in self.classifiers.items():
            i, j = classifier.slot_i, classifier.slot_j
            event = classifier.classify(
                det_i=slot_dets.get(i),
                det_j=slot_dets.get(j),
                mask_i=slot_masks[i] if i < len(slot_masks) else None,
                mask_j=slot_masks[j] if j < len(slot_masks) else None,
                centroid_i=slot_centroids[i] if i < len(slot_centroids) else None,
                centroid_j=slot_centroids[j] if j < len(slot_centroids) else None,
                velocity_i=self.velocity_estimators[i].velocity(),
                velocity_j=self.velocity_estimators[j].velocity(),
                body_length_i=self.body_length_estimators[i].current(),
                body_length_j=self.body_length_estimators[j].current(),
                frame_idx=frame_idx,
                time_sec=time_sec,
            )

            # Bout manager procesa y asigna bout_id
            self.bout_manager.process_event(event)

            # Guardar evento (escribimos CSV al final para tener bout_ids completos)
            self._all_events.append(event)

        self._frames_processed += 1

    def write_merged_placeholder(
        self,
        slot_centroids: List[Optional[Tuple[float, float]]],
        frame_idx: int,
    ) -> None:
        """Escribe placeholder cuando las máscaras están fusionadas."""
        if self._first_frame_idx is None:
            self._first_frame_idx = frame_idx
        self._last_frame_idx = frame_idx

        time_sec = frame_idx / self.fps

        for pair_key in self.classifiers.keys():
            event = ContactEvent(
                frame_idx=frame_idx,
                time_sec=time_sec,
                pair_key=pair_key,
                contact_type=ContactType.NONE,
                merged_state=True,
            )
            self.bout_manager.process_event(event)
            self._all_events.append(event)

        self._frames_processed += 1

    def finalize(self) -> Dict[str, Any]:
        """Al terminar el video."""
        logger.info(
            "Finalizing ContactTrackerV2: %d frames processed, %d events",
            self._frames_processed, len(self._all_events),
        )

        # 1. Cerrar bouts pendientes
        bouts = self.bout_manager.close_all()
        logger.info("Total valid bouts: %d", len(bouts))

        # 2. Escribir CSV per-frame (con bout_ids finales)
        self._write_per_frame_csv()

        # 3. Escribir CSV de bouts
        bout_csv = self._write_bout_csv(bouts)

        # 4. Construir summary
        summary = self._build_summary(bouts)

        # 5. Escribir summary JSON
        json_path = self.output_dir / "session_summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary written: %s", json_path)

        # 6. PDF report
        try:
            self._generate_report(summary, bouts)
        except Exception as e:
            logger.warning("PDF report generation failed: %s", e)

        return summary

    # ------------------------- Internal helpers -------------------------

    def _map_detections_to_slots(
        self,
        detections: List[Detection],
        slot_centroids: List[Optional[Tuple[float, float]]],
    ) -> Dict[int, Optional[Detection]]:
        """Mapea detecciones YOLO a slots por proximidad de centroide.

        Preserva la lógica del v1: asignación greedy por proximidad.
        """
        result: Dict[int, Optional[Detection]] = {i: None for i in range(self.num_slots)}
        if not detections:
            return result

        used = set()
        for slot_idx in range(self.num_slots):
            sc = slot_centroids[slot_idx] if slot_idx < len(slot_centroids) else None
            if sc is None:
                continue

            best_di = None
            best_dist = float("inf")
            for di, det in enumerate(detections):
                if di in used:
                    continue
                dc = det.center()
                dist = euclidean(sc, dc)
                if dist < best_dist:
                    best_dist = dist
                    best_di = di

            if best_di is not None and best_dist < 200.0:
                result[slot_idx] = detections[best_di]
                used.add(best_di)

        return result

    def _write_per_frame_csv(self) -> None:
        """Escribe contacts_per_frame.csv con todos los eventos."""
        if not self._all_events:
            logger.warning("No events to write")
            return

        # Construir fieldnames desde el primer evento
        first_row = self._all_events[0].to_csv_row()
        fieldnames = list(first_row.keys())

        with self._csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in self._all_events:
                writer.writerow(event.to_csv_row())

        logger.info("Per-frame CSV written: %s (%d rows)", self._csv_path, len(self._all_events))

    def _write_bout_csv(self, bouts: List[Bout]) -> Path:
        """Escribe contact_bouts.csv."""
        path = self.output_dir / "contact_bouts.csv"
        if not bouts:
            # Escribir CSV vacío con headers
            fieldnames = [
                "bout_id", "pair_key", "contact_type",
                "start_frame", "end_frame",
                "start_time_sec", "end_time_sec", "duration_sec", "n_frames",
                "mean_nose_nose_dist_bl", "mean_centroid_dist_bl",
                "mean_mask_iou", "mean_velocity_i_bls", "mean_velocity_j_bls",
                "peak_score", "investigator_role",
            ]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return path

        first = bouts[0].to_csv_row()
        fieldnames = list(first.keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for bout in bouts:
                writer.writerow(bout.to_csv_row())

        logger.info("Bouts CSV written: %s (%d bouts)", path, len(bouts))
        return path

    def _build_summary(self, bouts: List[Bout]) -> Dict[str, Any]:
        """Construye el session_summary.json."""
        # Resumen por tipo
        type_summary: Dict[str, Dict[str, Any]] = {}
        for ct in ContactType.all_contact_types():
            bouts_of_type = [b for b in bouts if b.contact_type == ct]
            total_frames = sum(b.n_frames for b in bouts_of_type)
            total_duration = sum(b.duration_sec for b in bouts_of_type)
            type_summary[ct.value] = {
                "total_bouts": len(bouts_of_type),
                "total_frames": total_frames,
                "total_duration_sec": round(total_duration, 3),
                "mean_bout_duration_sec": round(total_duration / max(len(bouts_of_type), 1), 3),
            }

        # Resumen por par
        pair_summary: Dict[str, Dict[str, Any]] = {}
        for pair_key in self.classifiers.keys():
            bouts_of_pair = [b for b in bouts if b.pair_key == pair_key]
            pair_summary[pair_key] = {
                "total_bouts": len(bouts_of_pair),
                "total_duration_sec": round(sum(b.duration_sec for b in bouts_of_pair), 3),
                "by_type": {
                    ct.value: sum(1 for b in bouts_of_pair if b.contact_type == ct)
                    for ct in ContactType.all_contact_types()
                },
            }

        # Calidad agregada
        quality_flags = {
            "stale_keypoints": sum(1 for e in self._all_events if e.stale_keypoints),
            "high_mask_overlap": sum(1 for e in self._all_events if e.high_mask_overlap),
            "missing_keypoints": sum(1 for e in self._all_events if e.missing_keypoints),
            "single_detection": sum(1 for e in self._all_events if e.single_detection),
            "merged_state": sum(1 for e in self._all_events if e.merged_state),
        }

        # Concurrencia (frames con >1 tipo activo simultáneamente)
        activation = float(self.config.get("activation_threshold", 0.5))
        concurrent_frames = 0
        for e in self._all_events:
            active = e.scores.active_types(activation)
            if len(active) > 1:
                concurrent_frames += 1

        return {
            "metadata": {
                "video_path": self.video_path,
                "fps": self.fps,
                "num_slots": self.num_slots,
                "num_pairs": len(self.classifiers),
                "first_frame_idx": self._first_frame_idx,
                "last_frame_idx": self._last_frame_idx,
                "frames_processed": self._frames_processed,
            },
            "parameters": dict(self.config),
            "contact_type_summary": type_summary,
            "pair_summary": pair_summary,
            "quality_flags": quality_flags,
            "concurrent_frames": concurrent_frames,
            "total_bouts": len(bouts),
        }

    def _generate_report(self, summary: Dict[str, Any], bouts: List[Bout]) -> None:
        """Genera report.pdf.

        Por ahora: versión simple con matplotlib. Se puede mejorar después
        añadiendo las páginas que el v1 tenía (etograma, histogramas, etc.).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            logger.warning("matplotlib not available, skipping PDF report")
            return

        pdf_path = self.output_dir / "report.pdf"

        with PdfPages(str(pdf_path)) as pdf:
            # Página 1: resumen por tipo (bar chart)
            fig, ax = plt.subplots(figsize=(11, 7))
            types = list(summary["contact_type_summary"].keys())
            durations = [summary["contact_type_summary"][t]["total_duration_sec"] for t in types]
            ax.bar(types, durations)
            ax.set_ylabel("Total duration (s)")
            ax.set_title("Contact duration by type")
            ax.set_xlabel("Contact type")
            for i, v in enumerate(durations):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)

            # Página 2: resumen por par
            fig, ax = plt.subplots(figsize=(11, 7))
            pairs = list(summary["pair_summary"].keys())
            pair_durs = [summary["pair_summary"][p]["total_duration_sec"] for p in pairs]
            ax.bar(pairs, pair_durs)
            ax.set_ylabel("Total duration (s)")
            ax.set_title("Contact duration by pair")
            ax.set_xlabel("Pair")
            pdf.savefig(fig)
            plt.close(fig)

            # Página 3: texto con summary
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.axis("off")
            text = "SESSION SUMMARY\n\n"
            text += f"Video: {summary['metadata']['video_path']}\n"
            text += f"FPS: {summary['metadata']['fps']}\n"
            text += f"Animals: {summary['metadata']['num_slots']}\n"
            text += f"Pairs: {summary['metadata']['num_pairs']}\n"
            text += f"Frames processed: {summary['metadata']['frames_processed']}\n"
            text += f"Total bouts: {summary['total_bouts']}\n"
            text += f"Concurrent frames (>1 type active): {summary['concurrent_frames']}\n\n"
            text += "QUALITY FLAGS:\n"
            for k, v in summary["quality_flags"].items():
                text += f"  {k}: {v}\n"
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
                    verticalalignment="top", family="monospace")
            pdf.savefig(fig)
            plt.close(fig)

        logger.info("PDF report written: %s", pdf_path)