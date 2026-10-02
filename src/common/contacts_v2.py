"""
Contact Tracker v2 — Fase 1.

Cambios respecto a la versión anterior (mismos inputs/outputs, lógica preservada):

  A. FAMILIAS: cada contacto lleva una columna `familia`
     (investigative / affiliative / non_contact / none). N2N y N2B se mantienen
     como tipos separados, solo se agrupan bajo la misma familia.

  B. SOLAPAMIENTOS:
     - FOL (following) ahora es estrictamente NO-CONTACTO: si hay contacto con el
       cuerpo del otro, no puede ser following (deja de competir con N2AG).
     - N2B no se dispara cuando el frame es claramente side-by-side.

  C. T2T (tail-to-tail) ELIMINADO del repertorio (era ruido, sin respaldo en
     etogramas estándar). Se conserva el cálculo de tail_tail_dist_bl como métrica
     geométrica, pero ya no produce un tipo de contacto.

  D. APPROACH / AVOID (nuevo): capa de dinámica de distancia entre centroides.
     `dynamics` ∈ {closing, stable, separating} y `mover` = qué animal genera el
     cambio. Aplica SIEMPRE (haya o no contacto). Cuando hay contacto, se deriva
     además `acepta_repele` (reciprocidad del receptor). Cuando NO hay contacto,
     el approach/avoid puro se escribe a un CSV separado.

  E. INITIATOR por bout: el rol del iniciador se decide por VOTO MAYORITARIO sobre
     todos los frames del bout (antes se fijaba con el primer frame).

  F. FALLBACK FOL: cuando el path de following usa el centroide porque tail_start
     no es confiable, se marca la bandera de calidad `fol_used_centroid`.

  + MAPEO DE KEYPOINTS corregido al pose real de 7 puntos:
       nose, left_ear, right_ear, mid_body, tail_start, tail_base, tail_tip
    donde el "trasero / zona anogenital" es tail_start (NO tail_base, que está a
    media cola). El eje del cuerpo se define como nose -> mid_body -> tail_start.

Outputs:
  - contacts_per_frame.csv   (Hoja A: contactos, con familia/dynamics/initiator/acepta_repele)
  - dynamics_no_contact.csv  (Hoja B: approach/avoid SIN contacto)
  - contact_bouts.csv
  - session_summary.json
  - report.pdf               (igual que antes; individual_metrics se deja intacto)
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import deque, Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter

from src.common.utils import Detection

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1 — DATACLASSES Y ENUMS
# ============================================================================

class ContactType(str, Enum):
    """Tipos de contacto social + NONE. (T2T eliminado en fase 1.)"""
    NONE = "none"
    N2N = "N2N"      # nose-to-nose
    N2AG = "N2AG"    # nose-to-anogenital
    FOL = "FOL"      # following (NO-contacto)
    SBS = "SBS"      # side-by-side
    N2B = "N2B"      # nose-to-body

    @classmethod
    def all_contact_types(cls) -> List["ContactType"]:
        """Todos los tipos menos NONE."""
        return [cls.N2N, cls.N2AG, cls.FOL, cls.SBS, cls.N2B]


class Family(str, Enum):
    """Familia conductual a la que pertenece un tipo de contacto (cambio A)."""
    NONE = "none"
    INVESTIGATIVE = "investigative"   # N2N, N2AG, N2B
    AFFILIATIVE = "affiliative"       # SBS
    NON_CONTACT = "non_contact"       # FOL


# Mapeo tipo -> familia (cambio A)
CONTACT_FAMILY: Dict[ContactType, Family] = {
    ContactType.NONE: Family.NONE,
    ContactType.N2N: Family.INVESTIGATIVE,
    ContactType.N2AG: Family.INVESTIGATIVE,
    ContactType.N2B: Family.INVESTIGATIVE,
    ContactType.SBS: Family.AFFILIATIVE,
    ContactType.FOL: Family.NON_CONTACT,
}


class Zone(str, Enum):
    """Zona espacial entre pares de animales."""
    CONTACT = "contact"
    PROXIMITY = "proximity"
    INDEPENDENT = "independent"


class Dynamics(str, Enum):
    """Dinámica de distancia entre los dos animales (cambio D)."""
    NONE = "none"            # sin movimiento relativo claro / datos insuficientes
    CLOSING = "closing"      # se están acercando
    STABLE = "stable"        # distancia estable
    SEPARATING = "separating"  # se están alejando


@dataclass
class ScoreMap:
    """Scores continuos [0, 1] por tipo de contacto en UN frame para UN par.

    (Se elimina t2t respecto a la versión anterior.)
    """
    n2n: float = 0.0
    n2ag: float = 0.0
    fol: float = 0.0
    sbs: float = 0.0
    n2b: float = 0.0

    def get(self, contact_type: ContactType) -> float:
        mapping = {
            ContactType.N2N: self.n2n,
            ContactType.N2AG: self.n2ag,
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

        Los tipos "raros" (FOL) usan un umbral más bajo para ser más sensibles.
        Prioridad: tipos específicos antes que N2B (catch-all).
        """
        if rare_types is None:
            rare_types = [ContactType.FOL]

        priority_order = [
            ContactType.N2N,
            ContactType.N2AG,
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

    def _threshold_for(
        self,
        ct: ContactType,
        activation_threshold: float,
        rare_threshold: float,
        rare_types: List[ContactType],
    ) -> float:
        return rare_threshold if ct in rare_types else activation_threshold

    def active_types(
        self,
        activation_threshold: float = 0.5,
        rare_threshold: float = 0.35,
        rare_types: Optional[List[ContactType]] = None,
    ) -> List[ContactType]:
        """Todos los tipos activos, respetando el umbral raro por tipo (fix INC-1)."""
        if rare_types is None:
            rare_types = [ContactType.FOL]
        out = []
        for ct in ContactType.all_contact_types():
            thr = self._threshold_for(ct, activation_threshold, rare_threshold, rare_types)
            if self.get(ct) >= thr:
                out.append(ct)
        return out

    def secondary_type(
        self,
        primary: ContactType,
        threshold: float = 0.4,
        rare_threshold: float = 0.35,
        rare_types: Optional[List[ContactType]] = None,
    ) -> Tuple[ContactType, float]:
        """Segundo tipo activo más fuerte (distinto al primary).

        Respeta umbral raro por tipo (fix INC-2).
        """
        if rare_types is None:
            rare_types = [ContactType.FOL]
        best_type = ContactType.NONE
        best_score = 0.0
        for ct in ContactType.all_contact_types():
            if ct == primary:
                continue
            thr = rare_threshold if ct in rare_types else threshold
            score = self.get(ct)
            if score >= thr and score > best_score:
                best_score = score
                best_type = ct
        return best_type, best_score

    def max_score(self) -> float:
        return max(self.n2n, self.n2ag, self.fol, self.sbs, self.n2b)

    def to_dict(self) -> Dict[str, float]:
        """Serialización con prefijo 'score_' para CSV."""
        return {
            "score_n2n": round(self.n2n, 4),
            "score_n2ag": round(self.n2ag, 4),
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

    # Scores continuos
    scores: ScoreMap = field(default_factory=ScoreMap)

    # Clasificación
    contact_type: ContactType = ContactType.NONE
    family: Family = Family.NONE          # cambio A
    zone: Zone = Zone.INDEPENDENT

    # Tipo secundario concurrente
    secondary_type: ContactType = ContactType.NONE
    secondary_score: float = 0.0

    # --- Dinámica de distancia (cambio D) ---
    dynamics: Dynamics = Dynamics.NONE
    mover: Optional[str] = None           # "i", "j" o None (quién genera el cambio)
    dist_delta_bls: float = 0.0           # cambio de distancia centroide (BL/s); <0 = se acercan
    # Reciprocidad del receptor cuando hay contacto: "accepts" / "rejects" / None
    reciprocity: Optional[str] = None

    # Métricas geométricas (normalizadas en body lengths)
    nose_nose_dist_bl: float = float("inf")
    centroid_dist_bl: float = float("inf")
    nose_tailbase_ij_bl: float = float("inf")   # nariz_i -> trasero_j (tail_start_j)
    nose_tailbase_ji_bl: float = float("inf")   # nariz_j -> trasero_i (tail_start_i)
    tail_tail_dist_bl: float = float("inf")     # se conserva como métrica (T2T ya no es tipo)
    mask_iou: float = 0.0
    mask_contact_bl: float = 0.0   # longitud del borde compartido entre máscaras / bl_ref

    # Cinemática
    velocity_i_bls: float = 0.0
    velocity_j_bls: float = 0.0
    velocity_alignment_cos: float = 0.0
    orientation_alignment_cos: float = 0.0

    # Body lengths (px)
    body_length_i_px: float = 0.0
    body_length_j_px: float = 0.0

    # Rol asimétrico (quién investiga este frame; el del bout se decide aparte)
    investigator_role: Optional[str] = None

    # Bout tracking
    bout_id: Optional[str] = None
    # Initiator del bout (se rellena al finalizar por voto mayoritario, cambio E)
    bout_initiator: Optional[str] = None

    # Flags de calidad
    stale_keypoints: bool = False
    high_mask_overlap: bool = False
    missing_keypoints: bool = False
    single_detection: bool = False
    merged_state: bool = False
    fol_used_centroid: bool = False       # cambio F

    def to_csv_row(self) -> Dict[str, Any]:
        """Fila para contacts_per_frame.csv (Hoja A).

        Orden de columnas:
          1. Identificación
          2. Familia + clasificación principal + secundaria   (familia = cambio A)
          3. Dinámica: dynamics, mover, dist_delta, acepta_repele   (cambio D)
          4. Zona
          5. Distancias geométricas (body lengths)
          6. Cinemática
          7. Body lengths (px)
          8. Rol investigador (frame) + initiator (bout)
          9. Bout id
          10. Soft scores
          11. Flags de calidad
        """
        row = {
            # 1
            "frame_idx": self.frame_idx,
            "time_str": _fmt_time(self.time_sec),
            "time_sec": round(self.time_sec, 3),
            "pair_key": self.pair_key,
            # 2
            "family": self.family.value,
            "contact_type": self.contact_type.value,
            "name_contact": CONTACT_TYPE_NAMES.get(self.contact_type.value, ""),
            "secondary_type": self.secondary_type.value if self.secondary_type != ContactType.NONE else "",
            "secondary_name": CONTACT_TYPE_NAMES.get(self.secondary_type.value, "") if self.secondary_type != ContactType.NONE else "",
            "secondary_score": round(self.secondary_score, 4),
            # 3
            "dynamics": self.dynamics.value,
            "mover": self.mover or "",
            "dist_delta_bls": round(self.dist_delta_bls, 4),
            "reciprocity": self.reciprocity or "",
            # 4
            "zone": self.zone.value,
            # 5
            "nose_nose_dist_bl": _fmt(self.nose_nose_dist_bl),
            "centroid_dist_bl": _fmt(self.centroid_dist_bl),
            "nose_tailbase_ij_bl": _fmt(self.nose_tailbase_ij_bl),
            "nose_tailbase_ji_bl": _fmt(self.nose_tailbase_ji_bl),
            "tail_tail_dist_bl": _fmt(self.tail_tail_dist_bl),
            "mask_iou": round(self.mask_iou, 4),
            "mask_contact_bl": round(self.mask_contact_bl, 4),
            # 6
            "velocity_i_bls": round(self.velocity_i_bls, 4),
            "velocity_j_bls": round(self.velocity_j_bls, 4),
            "velocity_alignment_cos": round(self.velocity_alignment_cos, 4),
            "orientation_alignment_cos": round(self.orientation_alignment_cos, 4),
            # 7
            "body_length_i_px": round(self.body_length_i_px, 2),
            "body_length_j_px": round(self.body_length_j_px, 2),
            # 8
            "investigator_role": self.investigator_role or "",
            "initiator": self.bout_initiator or "",
            # 9
            "bout_id": self.bout_id or "",
            # 11
            "stale_keypoints": int(self.stale_keypoints),
            "high_mask_overlap": int(self.high_mask_overlap),
            "missing_keypoints": int(self.missing_keypoints),
            "single_detection": int(self.single_detection),
            "merged_state": int(self.merged_state),
            "fol_used_centroid": int(self.fol_used_centroid),
        }
        # 10
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
    mean_mask_contact_bl: float = 0.0
    mean_velocity_i_bls: float = 0.0
    mean_velocity_j_bls: float = 0.0
    peak_score: float = 0.0

    # Acumuladores internos
    _sum_nose_nose: float = 0.0
    _sum_centroid: float = 0.0
    _sum_mask_iou: float = 0.0
    _sum_mask_contact_bl: float = 0.0
    _sum_velocity_i: float = 0.0
    _sum_velocity_j: float = 0.0
    _count_valid_nose_nose: int = 0
    _count_valid_centroid: int = 0

    # Voto mayoritario de iniciador (cambio E)
    _role_votes: Counter = field(default_factory=Counter)
    investigator_role: Optional[str] = None   # resultado del voto

    @property
    def family(self) -> Family:
        return CONTACT_FAMILY.get(self.contact_type, Family.NONE)

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec

    def accumulate(self, event: ContactEvent) -> None:
        """Añade las métricas de un frame al bout."""
        self.n_frames += 1
        self.end_frame = event.frame_idx
        self.end_time_sec = event.time_sec

        score = event.scores.get(self.contact_type)
        if score > self.peak_score:
            self.peak_score = score

        if math.isfinite(event.nose_nose_dist_bl):
            self._sum_nose_nose += event.nose_nose_dist_bl
            self._count_valid_nose_nose += 1
        if math.isfinite(event.centroid_dist_bl):
            self._sum_centroid += event.centroid_dist_bl
            self._count_valid_centroid += 1

        self._sum_mask_iou += event.mask_iou
        self._sum_mask_contact_bl += event.mask_contact_bl
        self._sum_velocity_i += event.velocity_i_bls
        self._sum_velocity_j += event.velocity_j_bls

        # Voto de iniciador (cambio E): solo cuenta frames con rol definido
        if event.investigator_role in ("i", "j"):
            self._role_votes[event.investigator_role] += 1

    def finalize_metrics(self) -> None:
        """Calcula medias y decide el iniciador por voto mayoritario."""
        if self._count_valid_nose_nose > 0:
            self.mean_nose_nose_dist_bl = self._sum_nose_nose / self._count_valid_nose_nose
        if self._count_valid_centroid > 0:
            self.mean_centroid_dist_bl = self._sum_centroid / self._count_valid_centroid
        if self.n_frames > 0:
            self.mean_mask_iou = self._sum_mask_iou / self.n_frames
            self.mean_mask_contact_bl = self._sum_mask_contact_bl / self.n_frames
            self.mean_velocity_i_bls = self._sum_velocity_i / self.n_frames
            self.mean_velocity_j_bls = self._sum_velocity_j / self.n_frames

        # Cambio E: iniciador = rol más votado a lo largo del bout
        if self._role_votes:
            self.investigator_role = self._role_votes.most_common(1)[0][0]
        else:
            self.investigator_role = None

    def to_csv_row(self) -> Dict[str, Any]:
        """Fila para contact_bouts.csv. Incluye familia (cambio A) e initiator (cambio E)."""
        return {
            "id": self.bout_id,
            "pair_key": self.pair_key,
            "family": self.family.value,
            "contact_type": self.contact_type.value,
            "name_contact": CONTACT_TYPE_NAMES.get(self.contact_type.value, ""),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": _fmt_time(self.start_time_sec),
            "end_time": _fmt_time(self.end_time_sec),
            "duration_sec": round(self.duration_sec, 3),
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
            "n_frames": self.n_frames,
            "mean_nose_nose_dist_bl": _fmt(self.mean_nose_nose_dist_bl),
            "mean_centroid_dist_bl": _fmt(self.mean_centroid_dist_bl),
            "mean_mask_iou": round(self.mean_mask_iou, 4),
            "mean_mask_contact_bl": round(self.mean_mask_contact_bl, 4),
            "mean_velocity_i_bls": round(self.mean_velocity_i_bls, 4),
            "mean_velocity_j_bls": round(self.mean_velocity_j_bls, 4),
            "peak_score": round(self.peak_score, 4),
            "initiator": self.investigator_role or "",
        }


@dataclass
class DynamicsEvent:
    """Evento de approach/avoid SIN contacto (Hoja B, cambio D).

    Episodio continuo en que los animales se acercan o se alejan sin estar en
    ningún contacto. `mover` es el animal que genera el cambio de distancia.
    """
    event_id: str
    pair_key: str
    kind: str                # "approach" o "avoidance"
    mover: Optional[str]
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    n_frames: int = 0
    start_dist_bl: float = float("inf")
    end_dist_bl: float = float("inf")
    _sum_delta: float = 0.0

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec

    @property
    def mean_delta_bls(self) -> float:
        return self._sum_delta / self.n_frames if self.n_frames > 0 else 0.0

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "id": self.event_id,
            "pair_key": self.pair_key,
            "kind": self.kind,
            "mover": self.mover or "",
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": _fmt_time(self.start_time_sec),
            "end_time": _fmt_time(self.end_time_sec),
            "duration_sec": round(self.duration_sec, 3),
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
            "n_frames": self.n_frames,
            "start_dist_bl": _fmt(self.start_dist_bl),
            "end_dist_bl": _fmt(self.end_dist_bl),
            "mean_delta_bls": round(self.mean_delta_bls, 4),
        }


def _fmt(x: float) -> float:
    """Formato limpio para CSV (infinito -> -1.0)."""
    if not math.isfinite(x):
        return -1.0
    return round(x, 4)


def _fmt_time(seconds: float) -> str:
    """Formato mm:ss.ms para columnas de tiempo legibles."""
    if not math.isfinite(seconds) or seconds < 0:
        return "00:00.000"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:06.3f}"


# Mapping sigla -> nombre completo (T2T eliminado)
CONTACT_TYPE_NAMES: Dict[str, str] = {
    "none": "None",
    "N2N": "Nose-to-Nose",
    "N2AG": "Nose-to-Anogenital",
    "FOL": "Following",
    "SBS": "Side-by-Side",
    "N2B": "Nose-to-Body",
}


# ============================================================================
# SECTION 2 — GEOMETRY HELPERS
# ============================================================================

# --- Nombres de keypoints del pose real de 7 puntos ---
# Orden anatómico (cuerpo -> afuera): nose, ears, mid_body, tail_start, tail_base, tail_tip
# IMPORTANTE: el "trasero / zona anogenital" es tail_start, NO tail_base.
KP_NOSE = "nose"
KP_LEFT_EAR = "left_ear"
KP_RIGHT_EAR = "right_ear"
KP_MID_BODY = "mid_body"
KP_TAIL_START = "tail_start"   # unión cuerpo-cola = trasero / anogenital
KP_TAIL_BASE = "tail_base"     # punto a media cola (NO es el trasero)
KP_TAIL_TIP = "tail_tip"


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
    """Extrae keypoint por NOMBRE si supera confianza."""
    if det is None or det.keypoints is None:
        return None
    for kp in det.keypoints:
        if kp.name == name and kp.conf >= min_conf:
            return (kp.x, kp.y)
    return None


def head_center(
    det: Optional[Detection],
    min_conf: float = 0.3,
) -> Optional[Tuple[float, float]]:
    """Centro de la cabeza = punto medio entre orejas.

    Si solo hay una oreja confiable, usa esa. Si ninguna, None.
    """
    le = get_keypoint(det, KP_LEFT_EAR, min_conf)
    re = get_keypoint(det, KP_RIGHT_EAR, min_conf)
    if le is not None and re is not None:
        return ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
    return le if le is not None else re


def head_orientation(
    det: Optional[Detection],
    min_conf: float = 0.3,
    min_ear_sep: float = 4.0,
) -> Optional[Tuple[float, float]]:
    """Vector de orientación de la CABEZA usando las orejas.

    El vector perpendicular a (left_ear -> right_ear) que apunta hacia la nariz
    indica hacia dónde mira la cabeza. Si las orejas están demasiado juntas
    (poco confiable, < min_ear_sep px), retorna None para caer a la orientación
    del cuerpo.
    """
    le = get_keypoint(det, KP_LEFT_EAR, min_conf)
    re = get_keypoint(det, KP_RIGHT_EAR, min_conf)
    nose = get_keypoint(det, KP_NOSE, min_conf)
    if le is None or re is None or nose is None:
        return None
    if euclidean(le, re) < min_ear_sep:
        return None  # orejas casi colapsadas: vector poco fiable
    ear_mid = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
    dx = nose[0] - ear_mid[0]
    dy = nose[1] - ear_mid[1]
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        return None
    return (dx / mag, dy / mag)


def body_orientation(
    det: Optional[Detection],
    min_conf: float = 0.3,
) -> Optional[Tuple[float, float]]:
    """Vector unitario del cuerpo (cola->nariz), apuntando hacia adelante.

    Mapeo corregido: usa el eje nose <- mid_body (preferido) o nose <- tail_start.
    NO usa tail_base (que está a media cola y torcería el vector).
    """
    nose = get_keypoint(det, KP_NOSE, min_conf)
    # Preferir mid_body como ancla trasera del eje del torso
    back = get_keypoint(det, KP_MID_BODY, min_conf)
    if back is None:
        back = get_keypoint(det, KP_TAIL_START, min_conf)
    if nose is None or back is None:
        return None
    dx = nose[0] - back[0]
    dy = nose[1] - back[1]
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        return None
    return (dx / mag, dy / mag)


def best_orientation(
    det: Optional[Detection],
    min_conf: float = 0.3,
) -> Optional[Tuple[float, float]]:
    """Mejor estimación de orientación de avance: cabeza (orejas) si es fiable,
    sino cuerpo (nose<-mid_body)."""
    h = head_orientation(det, min_conf)
    if h is not None:
        return h
    return body_orientation(det, min_conf)


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


def project_param_on_axis(
    point: Optional[Tuple[float, float]],
    axis_start: Optional[Tuple[float, float]],
    axis_end: Optional[Tuple[float, float]],
) -> Optional[float]:
    """Proyecta `point` sobre el segmento axis_start->axis_end y retorna t in [0,1].

    t=0 en axis_start, t=1 en axis_end. Sirve para ubicar la nariz del investigador
    a lo largo del eje del cuerpo del receptor (línea media) y así separar
    olfateo social (mitad delantera) de anogenital (mitad trasera).

    Retorna None si faltan puntos o el eje es degenerado.
    """
    if point is None or axis_start is None or axis_end is None:
        return None
    ax = axis_end[0] - axis_start[0]
    ay = axis_end[1] - axis_start[1]
    denom = ax * ax + ay * ay
    if denom < 1e-9:
        return None
    t = ((point[0] - axis_start[0]) * ax + (point[1] - axis_start[1]) * ay) / denom
    # clamp suave a [0,1] para clasificación de zona
    return max(0.0, min(1.0, t))


def valid_keypoint_count(det: Optional[Detection], min_conf: float = 0.3) -> int:
    """Cuenta keypoints con confianza suficiente."""
    if det is None or det.keypoints is None:
        return 0
    return sum(1 for kp in det.keypoints if kp.conf >= min_conf)


# ============================================================================
# SECTION 3 — BODY LENGTH ROBUSTO (percentil 80 estilo DeepOF)
# ============================================================================

class BodyLengthEstimator:
    """Estimador de body length usando percentil 80 de observaciones válidas.

    Mapeo corregido: el largo del cuerpo se mide nose <-> tail_start (trasero),
    NO nose <-> tail_base, para no incluir media cola en el "largo del cuerpo".
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
        """Acumula una observación si nose y tail_start son confiables."""
        self._total_frames_seen += 1
        nose = get_keypoint(det, KP_NOSE, self.min_conf)
        tail_start = get_keypoint(det, KP_TAIL_START, self.min_conf)
        if nose is None or tail_start is None:
            return

        dist = euclidean(nose, tail_start)
        if not math.isfinite(dist) or dist < 10.0:
            return

        self._observations.append(dist)
        self._frames_since_recompute += 1

        if self._frames_since_recompute >= self.recompute_every and len(self._observations) >= 10:
            self._cached_value = float(np.percentile(list(self._observations), self.percentile))
            self._frames_since_recompute = 0

    def current(self) -> float:
        if self._total_frames_seen < self.warmup_frames or len(self._observations) < 10:
            return self.fallback_px
        return self._cached_value


# ============================================================================
# SECTION 4 — VELOCIDAD CON SAVITZKY-GOLAY
# ============================================================================

class VelocityEstimator:
    """Estimador de velocidad con suavizado Savitzky-Golay (idéntico a la versión previa)."""

    def __init__(
        self,
        slot_idx: int,
        window_length: int = 11,
        polyorder: int = 3,
        fps: float = 25.0,
    ):
        if window_length % 2 == 0:
            window_length += 1
        self.slot_idx = slot_idx
        self.window_length = window_length
        self.polyorder = polyorder
        self.fps = fps

        self._buffer_x: Deque[float] = deque(maxlen=window_length)
        self._buffer_y: Deque[float] = deque(maxlen=window_length)
        self._last_valid: Optional[Tuple[float, float]] = None

    def update(self, centroid: Optional[Tuple[float, float]]) -> None:
        if centroid is not None:
            self._last_valid = centroid
            self._buffer_x.append(centroid[0])
            self._buffer_y.append(centroid[1])
        elif self._last_valid is not None:
            self._buffer_x.append(self._last_valid[0])
            self._buffer_y.append(self._last_valid[1])

    def velocity(self) -> Tuple[float, float]:
        if len(self._buffer_x) < self.window_length:
            if len(self._buffer_x) < 2:
                return (0.0, 0.0)
            dx = self._buffer_x[-1] - self._buffer_x[-2]
            dy = self._buffer_y[-1] - self._buffer_y[-2]
            return (dx * self.fps, dy * self.fps)

        try:
            x_arr = np.array(self._buffer_x)
            y_arr = np.array(self._buffer_y)
            vx_arr = savgol_filter(x_arr, self.window_length, self.polyorder, deriv=1, delta=1.0 / self.fps)
            vy_arr = savgol_filter(y_arr, self.window_length, self.polyorder, deriv=1, delta=1.0 / self.fps)
            return (float(vx_arr[-1]), float(vy_arr[-1]))
        except Exception as e:
            logger.debug("SG filter failed: %s", e)
            return (0.0, 0.0)

    def speed(self) -> float:
        vx, vy = self.velocity()
        return math.sqrt(vx * vx + vy * vy)


# ============================================================================
# SECTION 5 — SCHMITT TRIGGER (HISTÉRESIS)
# ============================================================================

class SchmittTrigger:
    """Umbral dual con histéresis (idéntico a la versión previa)."""

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
        if not math.isfinite(value):
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
# SECTION 6 — SOFT SCORING (funciones fuzzy-like) — idénticas a la versión previa
# ============================================================================

def trapezoidal_score(x: float, a: float, b: float, c: float, d: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / max(b - a, 1e-9)
    return (d - x) / max(d - c, 1e-9)


def reversed_trapezoidal_score(x: float, near: float, far: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x <= near:
        return 1.0
    if x >= far:
        return 0.0
    return (far - x) / max(far - near, 1e-9)


def logistic_score(x: float, midpoint: float, steepness: float = 10.0) -> float:
    if not math.isfinite(x):
        return 0.0
    try:
        return 1.0 / (1.0 + math.exp(steepness * (x - midpoint)))
    except OverflowError:
        return 0.0 if steepness * (x - midpoint) > 0 else 1.0


def mask_contact_px(mask_i, mask_j, dilate_px: int) -> float:
    """Longitud aproximada (px) del borde compartido entre dos máscaras.

    Cuenta los píxeles de borde de cada máscara que caen dentro de la otra
    dilatada dilate_px, y promedia ambas direcciones. Recorta al bbox de la
    unión (+dilate_px+2) por velocidad. Devuelve 0.0 si alguna máscara es
    None o está vacía."""
    if mask_i is None or mask_j is None:
        return 0.0
    mi = np.asarray(mask_i) > 0
    mj = np.asarray(mask_j) > 0
    if not mi.any() or not mj.any():
        return 0.0
    ys, xs = np.nonzero(mi | mj)
    pad = int(dilate_px) + 2
    y0, y1 = max(int(ys.min()) - pad, 0), int(ys.max()) + pad + 1
    x0, x1 = max(int(xs.min()) - pad, 0), int(xs.max()) + pad + 1
    a = mi[y0:y1, x0:x1].astype(np.uint8)
    b = mj[y0:y1, x0:x1].astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    kd = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1))
    bnd_a = a & (1 - cv2.erode(a, k3))
    bnd_b = b & (1 - cv2.erode(b, k3))
    c_ij = int(np.count_nonzero(bnd_a & cv2.dilate(b, kd)))
    c_ji = int(np.count_nonzero(bnd_b & cv2.dilate(a, kd)))
    return (c_ij + c_ji) / 2.0


def ramp_up_score(x: float, low: float, high: float) -> float:
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

    Cambios de fase 1:
      - Mapeo de keypoints corregido (tail_start = trasero, eje nose-mid_body-tail_start).
      - FOL estrictamente no-contacto (cambio B).
      - N2B no se dispara si el frame es claramente SBS (cambio B).
      - Bandera fol_used_centroid cuando el path usa centroide (cambio F).
      - Dinámica de distancia closing/stable/separating + mover (cambio D).
    """

    FOLLOW_PATH_BUFFER_FRAMES = 12  # ~0.5s a 25fps

    def __init__(self, pair_key: str, slot_i: int, slot_j: int, config: Dict[str, Any]):
        self.pair_key = pair_key
        self.slot_i = slot_i
        self.slot_j = slot_j
        self.config = config

        # Umbrales de zona de contacto
        self.contact_near = float(config.get("contact_zone_bl_enter", 0.30))
        self.contact_far = float(config.get("contact_zone_bl_exit", 0.45))
        self.proximity_bl = float(config.get("proximity_zone_bl", 1.0))
        self.min_kp_conf = float(config.get("min_keypoint_conf", 0.3))

        # SBS
        # Contacto de máscaras = longitud del borde compartido (en BL). El IoU es
        # estructuralmente 0 con máscaras disjuntas (CUTIE), por eso no se usa aquí.
        self.mask_contact_dilate_px = int(config.get("mask_contact_dilate_px", 4))
        self.sbs_contact_enter = float(config.get("sbs_contact_bl_enter", 0.35))
        self.sbs_contact_exit = float(config.get("sbs_contact_bl_exit", 0.20))
        self.sbs_latch_min_dist_score = float(config.get("sbs_latch_min_dist_score", 0.5))
        self.sbs_max_speed_bls = float(config.get("sbs_max_velocity_bls", 0.5))
        self.sbs_parallel_cos_min = float(config.get("sbs_parallel_cos_min", 0.7))

        # FOL
        self.follow_near_bl = float(config.get("follow_radius_bl", 0.4))
        self.follow_far_bl = float(config.get("follow_radius_bl_exit", 0.6))
        self.follow_min_speed_bls = float(config.get("follow_min_speed_bls", 0.15))
        self.follow_alignment_cos = float(config.get("follow_alignment_cos", 0.6))
        # Cambio B: distancia mínima al cuerpo del otro por encima de la cual FOL
        # se considera "sin contacto". Si los cuerpos se tocan, no es following.
        self.follow_no_contact_bl = float(config.get("follow_no_contact_bl", 0.5))

        # Overlap mask warning
        self.mask_overlap_warning = float(config.get("mask_overlap_warning", 0.5))

        # Línea media (cambio: separa social vs anogenital por proyección sobre el eje)
        # t in [0,1] sobre nose(0) -> mid_body -> tail_start(1).
        # t <= social_split  => mitad delantera (olfateo social)
        # t >  social_split  => mitad trasera   (olfateo anogenital)
        self.social_split = float(config.get("social_anogenital_split", 0.55))

        # Dinámica (cambio D): umbral de |delta distancia| (BL/s) para considerar
        # que hay acercamiento/alejamiento real (filtra ruido).
        self.dyn_delta_thresh_bls = float(config.get("dynamics_delta_thresh_bls", 0.10))
        # Fracción del movimiento que un animal debe aportar para ser el "mover".
        self.dyn_mover_frac = float(config.get("dynamics_mover_frac", 0.60))

        # Schmitt triggers internos (uno por tipo). Para distancias: tau_high < tau_low.
        self.trig_n2n = SchmittTrigger(self.contact_near, self.contact_far)
        self.trig_n2ag_ij = SchmittTrigger(self.contact_near, self.contact_far)
        self.trig_n2ag_ji = SchmittTrigger(self.contact_near, self.contact_far)
        self.trig_fol_ij = SchmittTrigger(self.follow_near_bl, self.follow_far_bl)
        self.trig_fol_ji = SchmittTrigger(self.follow_near_bl, self.follow_far_bl)
        self.trig_sbs = SchmittTrigger(
            tau_high=self.sbs_contact_enter,
            tau_low=self.sbs_contact_exit,
            inverted=True,
        )

        # Buffers de path para FOL (tail_start del followed; cae a centroide -> flag)
        self._path_i: Deque[Optional[Tuple[float, float]]] = deque(maxlen=self.FOLLOW_PATH_BUFFER_FRAMES)
        self._path_j: Deque[Optional[Tuple[float, float]]] = deque(maxlen=self.FOLLOW_PATH_BUFFER_FRAMES)
        # Marca si en el último append se usó centroide (cambio F)
        self._path_i_used_centroid: bool = False
        self._path_j_used_centroid: bool = False

        # Distancia centroide previa para la dinámica (cambio D)
        self._prev_centroid_dist: Optional[float] = None
        self._prev_time_sec: Optional[float] = None

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

        # Keypoints (mapeo corregido: tail_start = trasero/anogenital)
        nose_i = get_keypoint(det_i, KP_NOSE, self.min_kp_conf)
        nose_j = get_keypoint(det_j, KP_NOSE, self.min_kp_conf)
        rear_i = get_keypoint(det_i, KP_TAIL_START, self.min_kp_conf)   # trasero i
        rear_j = get_keypoint(det_j, KP_TAIL_START, self.min_kp_conf)   # trasero j
        mid_i = get_keypoint(det_i, KP_MID_BODY, self.min_kp_conf)
        mid_j = get_keypoint(det_j, KP_MID_BODY, self.min_kp_conf)
        # tail_base sigue disponible solo para la métrica tail_tail (no para tipos)
        tb_i = get_keypoint(det_i, KP_TAIL_BASE, self.min_kp_conf)
        tb_j = get_keypoint(det_j, KP_TAIL_BASE, self.min_kp_conf)

        # Buffer de paths para following (usa trasero; si no, centroide + flag F)
        if rear_i is not None:
            self._path_i.append(rear_i); self._path_i_used_centroid = False
        else:
            self._path_i.append(centroid_i); self._path_i_used_centroid = centroid_i is not None
        if rear_j is not None:
            self._path_j.append(rear_j); self._path_j_used_centroid = False
        else:
            self._path_j.append(centroid_j); self._path_j_used_centroid = centroid_j is not None

        # Normalización: PROMEDIO de los dos body lengths (fix BUG-2; antes era min)
        bl_ref = max((body_length_i + body_length_j) / 2.0, 1.0)

        # --- Distancias geométricas ---
        d_nose_nose = euclidean(nose_i, nose_j)
        d_centroid = euclidean(centroid_i, centroid_j)
        d_nose_i_rear_j = euclidean(nose_i, rear_j)   # nariz_i -> trasero_j
        d_nose_j_rear_i = euclidean(nose_j, rear_i)   # nariz_j -> trasero_i
        d_tail_tail = euclidean(tb_i, tb_j)

        event.nose_nose_dist_bl = d_nose_nose / bl_ref if math.isfinite(d_nose_nose) else float("inf")
        event.centroid_dist_bl = d_centroid / bl_ref if math.isfinite(d_centroid) else float("inf")
        event.nose_tailbase_ij_bl = d_nose_i_rear_j / bl_ref if math.isfinite(d_nose_i_rear_j) else float("inf")
        event.nose_tailbase_ji_bl = d_nose_j_rear_i / bl_ref if math.isfinite(d_nose_j_rear_i) else float("inf")
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

        # --- Contacto de máscaras (borde compartido en BL); solo si están cerca ---
        if (mask_i is not None and mask_j is not None
                and math.isfinite(event.centroid_dist_bl)
                and event.centroid_dist_bl < 1.5 * self.proximity_bl):
            event.mask_contact_bl = mask_contact_px(
                mask_i, mask_j, self.mask_contact_dilate_px) / bl_ref

        # --- Cinemática ---
        speed_i_px = math.sqrt(velocity_i[0] ** 2 + velocity_i[1] ** 2)
        speed_j_px = math.sqrt(velocity_j[0] ** 2 + velocity_j[1] ** 2)
        event.velocity_i_bls = speed_i_px / bl_ref if bl_ref > 0 else 0.0
        event.velocity_j_bls = speed_j_px / bl_ref if bl_ref > 0 else 0.0
        event.velocity_alignment_cos = cos_angle(velocity_i, velocity_j)

        orient_i = best_orientation(det_i, self.min_kp_conf)
        orient_j = best_orientation(det_j, self.min_kp_conf)
        event.orientation_alignment_cos = cos_angle(orient_i, orient_j)

        # --- SCORES ---

        # Determinar la "zona del cuerpo" que la nariz del investigador toca,
        # proyectando sobre el eje nose->mid_body->tail_start del receptor.
        # Usamos esto para repartir el contacto entre olfateo social y anogenital.
        # t_i_on_j: dónde cae la nariz de i sobre el cuerpo de j.
        t_i_on_j = self._nose_axis_param(nose_i, nose_j, mid_j, rear_j)
        t_j_on_i = self._nose_axis_param(nose_j, nose_i, mid_i, rear_i)

        # 1. N2N — nose-to-nose (orientaciones opuestas)
        event.scores.n2n = self._score_n2n(event.nose_nose_dist_bl, orient_i, orient_j)

        # 2. N2AG — anogenital: nariz cerca del trasero (tail_start) Y proyección
        #    en la mitad trasera del cuerpo del otro.
        n2ag_score, n2ag_role = self._score_n2ag(
            event.nose_tailbase_ij_bl,
            event.nose_tailbase_ji_bl,
            t_i_on_j=t_i_on_j,
            t_j_on_i=t_j_on_i,
            nose_i=nose_i, nose_j=nose_j,
            rear_i=rear_i, rear_j=rear_j,
            orient_i=orient_i, orient_j=orient_j,
        )
        event.scores.n2ag = n2ag_score

        # 3. FOL — following ESTRICTAMENTE no-contacto (cambio B)
        fol_score, fol_role = self._score_fol(
            nose_i, nose_j,
            centroid_i, centroid_j,
            orient_i, orient_j,
            event.velocity_i_bls, event.velocity_j_bls,
            bl_ref,
            min_body_dist_bl=min(event.nose_nose_dist_bl,
                                 event.nose_tailbase_ij_bl,
                                 event.nose_tailbase_ji_bl),
            mask_contact_bl=event.mask_contact_bl,
        )
        event.scores.fol = fol_score
        # Cambio F: propagar bandera de uso de centroide en el path del followed
        if fol_role == "i" and self._path_j_used_centroid:
            event.fol_used_centroid = True
        elif fol_role == "j" and self._path_i_used_centroid:
            event.fol_used_centroid = True

        # 4. SBS — side-by-side
        event.scores.sbs = self._score_sbs(
            event.mask_contact_bl,
            event.centroid_dist_bl,
            event.velocity_i_bls,
            event.velocity_j_bls,
            event.orientation_alignment_cos,
        )

        # 5. N2B — nose-to-body (catch-all). No se dispara si es claramente SBS (cambio B).
        n2b_score, n2b_role = self._score_n2b(
            nose_i, nose_j,
            mask_i, mask_j,
            event.nose_nose_dist_bl,
            event.nose_tailbase_ij_bl,
            event.nose_tailbase_ji_bl,
            centroid_i=centroid_i, centroid_j=centroid_j,
            orient_i=orient_i, orient_j=orient_j,
            sbs_score=event.scores.sbs,
        )
        event.scores.n2b = n2b_score

        # --- Decidir contact_type dominante ---
        activation = float(self.config.get("activation_threshold", 0.5))
        rare = float(self.config.get("activation_threshold_rare", 0.35))
        event.contact_type = event.scores.argmax_type(
            activation_threshold=activation,
            rare_threshold=rare,
        )
        event.family = CONTACT_FAMILY.get(event.contact_type, Family.NONE)

        # --- secondary_type ---
        secondary_threshold = float(self.config.get("secondary_threshold", 0.4))
        sec_type, sec_score = event.scores.secondary_type(
            primary=event.contact_type,
            threshold=secondary_threshold,
            rare_threshold=rare,
        )
        event.secondary_type = sec_type
        event.secondary_score = sec_score

        # --- investigator_role del frame (según el tipo ganador) ---
        if event.contact_type == ContactType.N2AG:
            event.investigator_role = n2ag_role
        elif event.contact_type == ContactType.FOL:
            event.investigator_role = fol_role
        elif event.contact_type == ContactType.N2B:
            event.investigator_role = n2b_role
        elif event.contact_type == ContactType.N2N:
            # En N2N ambos investigan; dejamos rol None salvo asimetría futura
            event.investigator_role = None

        # --- Dinámica de distancia (cambio D) ---
        self._compute_dynamics(event, centroid_i, centroid_j,
                               velocity_i, velocity_j, bl_ref, time_sec)

        return event

    # -------------------------- helpers de zona/eje --------------------------

    def _determine_zone(self, centroid_dist_bl: float) -> Zone:
        if not math.isfinite(centroid_dist_bl):
            return Zone.INDEPENDENT
        if centroid_dist_bl < self.contact_near:
            return Zone.CONTACT
        if centroid_dist_bl < self.proximity_bl:
            return Zone.PROXIMITY
        return Zone.INDEPENDENT

    def _nose_axis_param(
        self,
        nose_investigator: Optional[Tuple[float, float]],
        nose_target: Optional[Tuple[float, float]],
        mid_target: Optional[Tuple[float, float]],
        rear_target: Optional[Tuple[float, float]],
    ) -> Optional[float]:
        """t in [0,1] de dónde cae la nariz del investigador sobre el eje del
        cuerpo del target: nose_target(0) -> mid_target -> rear_target(1).

        Usa dos tramos (nose->mid y mid->rear) y retorna el parámetro global
        aproximado en [0,1]. Si falta mid, usa nose->rear directo.
        """
        if nose_investigator is None or nose_target is None:
            return None
        if rear_target is None:
            return None
        if mid_target is None:
            return project_param_on_axis(nose_investigator, nose_target, rear_target)
        # Tramo delantero nose->mid (mapea a [0, 0.5]) y trasero mid->rear ([0.5,1])
        t_front = project_param_on_axis(nose_investigator, nose_target, mid_target)
        t_back = project_param_on_axis(nose_investigator, mid_target, rear_target)
        # Elegimos el tramo cuyo punto proyectado está más cerca de la nariz
        # (criterio simple: cuál segmento "posee" la proyección)
        if t_front is None and t_back is None:
            return None
        # Distancia de la nariz al punto proyectado en cada tramo
        def _proj_point(a, b, t):
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        cand = []
        if t_front is not None:
            p = _proj_point(nose_target, mid_target, t_front)
            cand.append((euclidean(nose_investigator, p), 0.5 * t_front))
        if t_back is not None:
            p = _proj_point(mid_target, rear_target, t_back)
            cand.append((euclidean(nose_investigator, p), 0.5 + 0.5 * t_back))
        cand.sort(key=lambda x: x[0])
        return cand[0][1]

    def _gaze_alignment(
        self,
        orient_investigator: Optional[Tuple[float, float]],
        nose_investigator: Optional[Tuple[float, float]],
        target_point: Optional[Tuple[float, float]],
    ) -> float:
        """Factor [0.4, 1.0] que indica si el investigador mira al target.
        Si falta dato, 1.0 (fallback seguro)."""
        if orient_investigator is None or nose_investigator is None or target_point is None:
            return 1.0
        dx = target_point[0] - nose_investigator[0]
        dy = target_point[1] - nose_investigator[1]
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            return 1.0
        to_target = (dx / mag, dy / mag)
        cos_align = cos_angle(orient_investigator, to_target)
        return 0.4 + 0.6 * (cos_align + 1.0) / 2.0

    # -------------------------- Scoring por tipo --------------------------

    def _score_n2n(
        self,
        nose_nose_bl: float,
        orient_i: Optional[Tuple[float, float]] = None,
        orient_j: Optional[Tuple[float, float]] = None,
    ) -> float:
        """nose-to-nose: Schmitt + soft score + factor de orientación opuesta."""
        active = self.trig_n2n.update(nose_nose_bl)
        soft = reversed_trapezoidal_score(nose_nose_bl, self.contact_near, self.contact_far)
        if active:
            soft = max(soft, 0.5)
        if orient_i is not None and orient_j is not None:
            opposite_j = (-orient_j[0], -orient_j[1])
            cos_face = cos_angle(orient_i, opposite_j)
            face_factor = 0.3 + 0.7 * (cos_face + 1.0) / 2.0
        else:
            face_factor = 1.0
        return soft * face_factor

    def _score_n2ag(
        self,
        nose_i_rear_j_bl: float,
        nose_j_rear_i_bl: float,
        t_i_on_j: Optional[float] = None,
        t_j_on_i: Optional[float] = None,
        nose_i: Optional[Tuple[float, float]] = None,
        nose_j: Optional[Tuple[float, float]] = None,
        rear_i: Optional[Tuple[float, float]] = None,
        rear_j: Optional[Tuple[float, float]] = None,
        orient_i: Optional[Tuple[float, float]] = None,
        orient_j: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, Optional[str]]:
        """nose-to-anogenital + quién investiga.

        Combina: distancia nariz->trasero (tail_start), factor de mirada, y
        factor de línea media (la nariz debe caer en la mitad TRASERA del cuerpo
        del receptor; t > social_split).
        """
        # i investiga a j
        active_ij = self.trig_n2ag_ij.update(nose_i_rear_j_bl)
        soft_ij = reversed_trapezoidal_score(nose_i_rear_j_bl, self.contact_near, self.contact_far)
        score_ij = max(soft_ij, 0.5) if active_ij else soft_ij
        # j investiga a i
        active_ji = self.trig_n2ag_ji.update(nose_j_rear_i_bl)
        soft_ji = reversed_trapezoidal_score(nose_j_rear_i_bl, self.contact_near, self.contact_far)
        score_ji = max(soft_ji, 0.5) if active_ji else soft_ji

        # Factor de mirada
        score_ij *= self._gaze_alignment(orient_i, nose_i, rear_j)
        score_ji *= self._gaze_alignment(orient_j, nose_j, rear_i)

        # Factor de línea media: la nariz debe estar en la mitad trasera (t alto)
        score_ij *= self._rear_zone_factor(t_i_on_j)
        score_ji *= self._rear_zone_factor(t_j_on_i)

        if score_ij > score_ji:
            return score_ij, "i"
        elif score_ji > score_ij:
            return score_ji, "j"
        elif score_ij > 0:
            return score_ij, "i"
        return 0.0, None

    def _rear_zone_factor(self, t: Optional[float]) -> float:
        """Factor [0,1]: 1 si la nariz cae en la mitad trasera (t -> 1),
        baja a 0 hacia la mitad delantera. Si t es None, 1.0 (sin penalizar)."""
        if t is None:
            return 1.0
        # rampa: por debajo de social_split factor 0; por encima sube a 1
        return ramp_up_score(t, self.social_split - 0.1, self.social_split + 0.1)

    def _front_zone_factor(self, t: Optional[float]) -> float:
        """Complemento: 1 en la mitad delantera (t -> 0), baja hacia atrás."""
        if t is None:
            return 1.0
        return reversed_trapezoidal_score(t, self.social_split - 0.1, self.social_split + 0.1)

    def _score_fol(
        self,
        nose_i, nose_j,
        centroid_i, centroid_j,
        orient_i, orient_j,
        speed_i_bls, speed_j_bls,
        bl_ref,
        min_body_dist_bl: float,
        mask_contact_bl: float,
    ) -> Tuple[float, Optional[str]]:
        """following — ESTRICTAMENTE no-contacto (cambio B).

        Si los cuerpos están en contacto (distancia mínima nariz-cuerpo por debajo
        de follow_no_contact_bl, o borde compartido de máscaras apreciable), el score de FOL se
        anula: en ese caso es un olfateo, no un following.
        """
        # Guardia no-contacto: si hay contacto cercano, no es following.
        if math.isfinite(min_body_dist_bl) and min_body_dist_bl < self.follow_no_contact_bl:
            return 0.0, None
        if mask_contact_bl > self.sbs_contact_enter:
            return 0.0, None

        score_ij, _ = self._score_fol_direction(
            nose_i, centroid_j, orient_i, speed_i_bls, self._path_j, bl_ref,
        )
        score_ji, _ = self._score_fol_direction(
            nose_j, centroid_i, orient_j, speed_j_bls, self._path_i, bl_ref,
        )
        if score_ij > score_ji:
            return score_ij, "i"
        elif score_ji > score_ij:
            return score_ji, "j"
        elif score_ij > 0:
            return score_ij, "i"
        return 0.0, None

    def _score_fol_direction(
        self,
        nose_follower, centroid_followed, orient_follower,
        speed_follower_bls, path_followed, bl_ref,
    ) -> Tuple[float, bool]:
        """Score de FOL en UNA dirección (follower -> followed)."""
        if nose_follower is None or centroid_followed is None:
            return 0.0, False

        speed_ok = speed_follower_bls >= self.follow_min_speed_bls

        # Distancia mínima del nose_follower al PATH del followed
        min_dist = float("inf")
        for past_pos in path_followed:
            if past_pos is None:
                continue
            d = euclidean(nose_follower, past_pos)
            if d < min_dist:
                min_dist = d
        min_dist_bl = min_dist / bl_ref if math.isfinite(min_dist) else float("inf")

        # Orientación hacia el followed
        if orient_follower is not None:
            to_followed = (centroid_followed[0] - nose_follower[0],
                           centroid_followed[1] - nose_follower[1])
            mag = math.sqrt(to_followed[0] ** 2 + to_followed[1] ** 2)
            if mag > 1e-6:
                orient_cos = cos_angle(orient_follower, (to_followed[0] / mag, to_followed[1] / mag))
            else:
                orient_cos = 0.0
        else:
            orient_cos = 0.0

        dist_score = reversed_trapezoidal_score(min_dist_bl, self.follow_near_bl, self.follow_far_bl)
        orient_score = ramp_up_score(orient_cos, self.follow_alignment_cos - 0.2, self.follow_alignment_cos + 0.2)
        speed_score = 1.0 if speed_ok else ramp_up_score(speed_follower_bls, 0.0, self.follow_min_speed_bls)

        combined = dist_score * orient_score * speed_score
        return combined, speed_ok

    def _score_sbs(
        self, mask_contact_bl, centroid_dist_bl, speed_i_bls, speed_j_bls, orientation_cos,
    ) -> float:
        """side-by-side. Media geométrica de factores (suavizada, no colapsa a 0).

        Fix BUG-3 (causa raíz): antes se usaba mask_iou, pero las máscaras de CUTIE
        son disjuntas por construcción (una por rata, sin solape), así que el IoU
        era siempre 0 y SBS nunca disparaba. Ahora se usa mask_contact_bl: longitud
        del borde compartido entre máscaras (dilatación) normalizada por BL."""
        dist_score = reversed_trapezoidal_score(centroid_dist_bl, self.contact_near, self.proximity_bl)
        max_speed = max(speed_i_bls, speed_j_bls)
        speed_score = reversed_trapezoidal_score(max_speed, self.sbs_max_speed_bls * 0.5, self.sbs_max_speed_bls * 1.5)
        align_score = ramp_up_score(abs(orientation_cos),
                                    self.sbs_parallel_cos_min - 0.15,
                                    self.sbs_parallel_cos_min + 0.15)

        # El Schmitt trigger solo se alimenta si pasan los factores "duros" (distancia y
        # alineación); si no, el latch (piso 0.5) haría SBS de cualquier borde en
        # contacto (p. ej. cola-con-cola o perpendicular).
        hard_ok = dist_score > 0.0 and align_score > 0.0
        # El trigger además exige cercanía real (dist_score >= sbs_latch_min_dist_score):
        # bordes en contacto a ~1 BL de centroide (cola con cola) no son lado a lado.
        latch_ok = hard_ok and dist_score >= self.sbs_latch_min_dist_score
        active = self.trig_sbs.update(mask_contact_bl if latch_ok else 0.0)
        iou_score = ramp_up_score(mask_contact_bl, self.sbs_contact_exit, self.sbs_contact_enter * 2)

        # Fix BUG-3: media geométrica (más robusta que producto crudo a un factor bajo)
        factors = [max(iou_score, 1e-6), max(dist_score, 1e-6),
                   max(speed_score, 1e-6), max(align_score, 1e-6)]
        combined = math.exp(sum(math.log(f) for f in factors) / len(factors))
        # pero si algún factor "duro" (dist o align) es 0 real, no hay SBS
        if not hard_ok:
            combined = 0.0

        if active:
            combined = max(combined, 0.5)
        return combined

    def _score_n2b(
        self,
        nose_i, nose_j,
        mask_i, mask_j,
        nose_nose_bl, nose_i_rear_j_bl, nose_j_rear_i_bl,
        centroid_i=None, centroid_j=None,
        orient_i=None, orient_j=None,
        sbs_score: float = 0.0,
    ) -> Tuple[float, Optional[str]]:
        """nose-to-body (catch-all). Nariz dentro de la máscara del otro.

        Cambio B: si el frame es claramente SBS (sbs_score alto), N2B se anula
        para no pisar al side-by-side.
        """
        # Guardia SBS (cambio B)
        sbs_strong = float(self.config.get("n2b_sbs_suppress", 0.5))
        if sbs_score >= sbs_strong:
            return 0.0, None

        # Guardia: si hay contacto específico ya cerca del umbral, reducir N2B
        guard_factor = 1.0
        if nose_nose_bl < self.contact_far or nose_i_rear_j_bl < self.contact_far or nose_j_rear_i_bl < self.contact_far:
            guard_factor = 0.7

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

        align_i = self._gaze_alignment(orient_i, nose_i, centroid_j)
        align_j = self._gaze_alignment(orient_j, nose_j, centroid_i)
        score_i_in_j *= guard_factor * align_i
        score_j_in_i *= guard_factor * align_j

        if score_i_in_j > score_j_in_i:
            return score_i_in_j, "i"
        elif score_j_in_i > score_i_in_j:
            return score_j_in_i, "j"
        elif score_i_in_j > 0:
            return score_i_in_j, "i"
        return 0.0, None

    # -------------------------- Dinámica (cambio D) --------------------------

    def _compute_dynamics(
        self,
        event: ContactEvent,
        centroid_i, centroid_j,
        velocity_i, velocity_j,
        bl_ref, time_sec,
    ) -> None:
        """Calcula closing/stable/separating + mover, y acepta_repele si hay contacto.

        - delta = d(t) - d(t-1) de la distancia centroide (en BL/s).
          delta < 0 -> se acercan (closing); > 0 -> se alejan (separating).
        - mover: el animal cuya velocidad proyectada sobre la línea que une los
          centroides explica la mayor parte del cambio.
        - acepta_repele (solo si hay contacto): si se separan y el mover es el
          receptor del contacto -> "repele"; si estable o se acercan -> "acepta".
        """
        d_now = event.centroid_dist_bl
        if not math.isfinite(d_now) or self._prev_centroid_dist is None or self._prev_time_sec is None:
            event.dynamics = Dynamics.NONE
            self._prev_centroid_dist = d_now if math.isfinite(d_now) else None
            self._prev_time_sec = time_sec
            return

        dt = time_sec - self._prev_time_sec
        if dt <= 1e-6:
            dt = 1.0 / 25.0
        delta = (d_now - self._prev_centroid_dist) / dt  # BL/s
        event.dist_delta_bls = delta

        if abs(delta) < self.dyn_delta_thresh_bls:
            event.dynamics = Dynamics.STABLE
            event.mover = None
        elif delta < 0:
            event.dynamics = Dynamics.CLOSING
            event.mover = self._dynamics_mover(centroid_i, centroid_j, velocity_i, velocity_j, closing=True)
        else:
            event.dynamics = Dynamics.SEPARATING
            event.mover = self._dynamics_mover(centroid_i, centroid_j, velocity_i, velocity_j, closing=False)

        # Reciprocidad si hay contacto este frame
        if event.contact_type != ContactType.NONE:
            event.reciprocity = self._reciprocity(event)

        self._prev_centroid_dist = d_now
        self._prev_time_sec = time_sec

    def _dynamics_mover(
        self, centroid_i, centroid_j, velocity_i, velocity_j, closing: bool,
    ) -> Optional[str]:
        """Decide qué animal genera el cambio de distancia, proyectando la
        velocidad de cada uno sobre el eje que une los centroides."""
        if centroid_i is None or centroid_j is None:
            return None
        ax = centroid_j[0] - centroid_i[0]
        ay = centroid_j[1] - centroid_i[1]
        mag = math.sqrt(ax * ax + ay * ay)
        if mag < 1e-6:
            return None
        ux, uy = ax / mag, ay / mag  # unitario i->j

        # Velocidad de i proyectada sobre i->j: positivo = i se acerca a j
        proj_i = velocity_i[0] * ux + velocity_i[1] * uy
        # Velocidad de j proyectada sobre j->i (= -u): positivo = j se acerca a i
        proj_j = -(velocity_j[0] * ux + velocity_j[1] * uy)

        # Contribución al ACERCAMIENTO: cuanto cada uno reduce la distancia
        contrib_i = proj_i
        contrib_j = proj_j
        if not closing:
            # Para separación, invertimos: quién más aleja
            contrib_i = -proj_i
            contrib_j = -proj_j

        total = contrib_i + contrib_j
        if total <= 1e-6:
            return None
        frac_i = contrib_i / total
        if frac_i >= self.dyn_mover_frac:
            return "i"
        if (1.0 - frac_i) >= self.dyn_mover_frac:
            return "j"
        return None  # ambos contribuyen parecido -> sin mover claro

    def _reciprocity(self, event: ContactEvent) -> Optional[str]:
        """Lectura de reciprocidad del receptor cuando hay contacto.

        Receptor = el animal que NO es el investigador del frame. Si no hay rol
        claro (p.ej. N2N), usamos el mover como aproximación.
        - separating + el receptor es el mover -> 'rejects'
        - closing o stable -> 'accepts'
        """
        if event.dynamics in (Dynamics.CLOSING, Dynamics.STABLE):
            return "accepts"
        # separating
        investigator = event.investigator_role  # "i"/"j"/None
        receptor = None
        if investigator == "i":
            receptor = "j"
        elif investigator == "j":
            receptor = "i"
        # Si el que se aleja (mover) es el receptor -> rejects
        if event.mover is not None and receptor is not None:
            return "rejects" if event.mover == receptor else "accepts"
        # Sin rol claro: si alguien se aleja durante el contacto, lo tratamos como rejects
        if event.mover is not None:
            return "rejects"
        return "accepts"


# ============================================================================
# SECTION 8 — BOUT MANAGER
# ============================================================================

class BoutManager:
    """Gestiona apertura, extensión, gap-bridging y cierre de bouts.

    Sin cambios de lógica respecto a la versión previa, salvo que el iniciador
    se decide en Bout.finalize_metrics() por voto mayoritario (cambio E).
    T2T ya no existe como tipo, por lo que no aparece aquí.
    """

    def __init__(self, fps: float, config: Dict[str, Any]):
        self.fps = fps
        default_min = int(fps / 4)
        default_min_follow = int(fps / 2)
        default_max_gap = 3

        self.max_gap_frames = int(config.get("bout_max_gap_frames", default_max_gap))
        self.min_duration_frames: Dict[ContactType, int] = {
            ContactType.N2N: int(config.get("bout_min_frames_n2n", default_min)),
            ContactType.N2AG: int(config.get("bout_min_frames_n2ag", default_min)),
            ContactType.FOL: int(config.get("bout_min_frames_fol", default_min_follow)),
            ContactType.SBS: int(config.get("bout_min_frames_sbs", default_min)),
            ContactType.N2B: int(config.get("bout_min_frames_n2b", default_min)),
        }

        self._open_bouts: Dict[Tuple[str, ContactType], Bout] = {}
        self._gap_counter: Dict[Tuple[str, ContactType], int] = {}
        self._closed_bouts: List[Bout] = []
        self._bout_counter: int = 0

    def process_event(self, event: ContactEvent) -> None:
        pair = event.pair_key
        current_type = event.contact_type

        if current_type != ContactType.NONE:
            key = (pair, current_type)
            if key in self._open_bouts:
                bout = self._open_bouts[key]
                bout.accumulate(event)
                event.bout_id = bout.bout_id
                self._gap_counter[key] = 0
            else:
                bout = self._open_new_bout(event)
                self._open_bouts[key] = bout
                self._gap_counter[key] = 0
                event.bout_id = bout.bout_id

        to_close = []
        for key, bout in list(self._open_bouts.items()):
            pair_key, ct = key
            if pair_key != pair:
                continue
            if ct == current_type:
                continue
            self._gap_counter[key] = self._gap_counter.get(key, 0) + 1
            if self._gap_counter[key] > self.max_gap_frames:
                to_close.append(key)
        for key in to_close:
            self._close_bout(key)

    def close_all(self) -> List[Bout]:
        for key in list(self._open_bouts.keys()):
            self._close_bout(key)
        return self._closed_bouts

    def _open_new_bout(self, event: ContactEvent) -> Bout:
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
        )
        bout.accumulate(event)
        return bout

    def _close_bout(self, key: Tuple[str, ContactType]) -> None:
        if key not in self._open_bouts:
            return
        bout = self._open_bouts.pop(key)
        self._gap_counter.pop(key, None)
        min_required = self.min_duration_frames.get(bout.contact_type, 6)
        if bout.n_frames >= min_required:
            bout.finalize_metrics()   # cambio E: aquí se decide initiator por voto
            self._closed_bouts.append(bout)
            logger.debug("Closed valid bout %s (%s, %d frames, %.2fs)",
                         bout.bout_id, bout.contact_type.value, bout.n_frames, bout.duration_sec)
        else:
            logger.debug("Discarded short bout %s (%d frames < %d required)",
                         bout.bout_id, bout.n_frames, min_required)


# ============================================================================
# SECTION 8b — DYNAMICS TRACKER (approach/avoid SIN contacto, cambio D)
# ============================================================================

class DynamicsTracker:
    """Agrupa frames de approach/avoid SIN contacto en eventos continuos (Hoja B).

    Un evento se abre cuando, sin contacto, el par viene closing (approach) o
    separating (avoidance) de forma sostenida. Usa el mismo gap-bridging que los
    bouts para no fragmentar por ruido.
    """

    def __init__(self, fps: float, config: Dict[str, Any]):
        self.fps = fps
        self.max_gap_frames = int(config.get("dynamics_max_gap_frames", 3))
        self.min_frames = int(config.get("dynamics_min_frames", max(2, int(fps / 5))))
        self._open: Dict[Tuple[str, str], DynamicsEvent] = {}  # (pair, kind) -> event
        self._gap: Dict[Tuple[str, str], int] = {}
        self._closed: List[DynamicsEvent] = []
        self._counter: int = 0

    def process_event(self, event: ContactEvent) -> None:
        pair = event.pair_key
        # Solo nos interesa approach/avoid SIN contacto
        in_contact = event.contact_type != ContactType.NONE
        kind = None
        if not in_contact:
            if event.dynamics == Dynamics.CLOSING:
                kind = "approach"
            elif event.dynamics == Dynamics.SEPARATING:
                kind = "avoidance"

        if kind is not None:
            key = (pair, kind)
            if key in self._open:
                ev = self._open[key]
                ev.n_frames += 1
                ev.end_frame = event.frame_idx
                ev.end_time_sec = event.time_sec
                ev.end_dist_bl = event.centroid_dist_bl
                ev._sum_delta += event.dist_delta_bls
                self._gap[key] = 0
            else:
                self._counter += 1
                ev = DynamicsEvent(
                    event_id=f"dyn_{self._counter:05d}_{kind}_{pair}",
                    pair_key=pair,
                    kind=kind,
                    mover=event.mover,
                    start_frame=event.frame_idx,
                    end_frame=event.frame_idx,
                    start_time_sec=event.time_sec,
                    end_time_sec=event.time_sec,
                    start_dist_bl=event.centroid_dist_bl,
                    end_dist_bl=event.centroid_dist_bl,
                )
                ev.n_frames = 1
                ev._sum_delta = event.dist_delta_bls
                self._open[key] = ev
                self._gap[key] = 0

        # Gap para los eventos abiertos del par que no se vieron este frame
        to_close = []
        for key in list(self._open.keys()):
            p, k = key
            if p != pair:
                continue
            if kind is not None and k == kind:
                continue
            self._gap[key] = self._gap.get(key, 0) + 1
            if self._gap[key] > self.max_gap_frames:
                to_close.append(key)
        for key in to_close:
            self._close(key)

    def close_all(self) -> List[DynamicsEvent]:
        for key in list(self._open.keys()):
            self._close(key)
        return self._closed

    def _close(self, key: Tuple[str, str]) -> None:
        if key not in self._open:
            return
        ev = self._open.pop(key)
        self._gap.pop(key, None)
        if ev.n_frames >= self.min_frames:
            self._closed.append(ev)


# ============================================================================
# SECTION 9 — CONTACT TRACKER V2 (CLASE PRINCIPAL)
# ============================================================================

class ContactTrackerV2:
    """Tracker principal. Interfaz IDÉNTICA a la versión previa (update/finalize)."""

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

        self.body_length_estimators: Dict[int, BodyLengthEstimator] = {
            i: BodyLengthEstimator(
                slot_idx=i,
                fallback_px=float(self.config.get("fallback_body_length_px", 120.0)),
            ) for i in range(num_slots)
        }
        self.velocity_estimators: Dict[int, VelocityEstimator] = {
            i: VelocityEstimator(slot_idx=i, fps=fps) for i in range(num_slots)
        }

        self.classifiers: Dict[str, ContactClassifier] = {}
        for i in range(num_slots):
            for j in range(i + 1, num_slots):
                pair_key = f"{i}_{j}"
                self.classifiers[pair_key] = ContactClassifier(
                    pair_key=pair_key, slot_i=i, slot_j=j, config=self.config,
                )

        self.bout_manager = BoutManager(fps=fps, config=self.config)
        self.dynamics_tracker = DynamicsTracker(fps=fps, config=self.config)  # cambio D

        self._csv_path = self.output_dir / "contacts_per_frame.csv"
        self._dynamics_csv_path = self.output_dir / "dynamics_no_contact.csv"  # Hoja B

        self._all_events: List[ContactEvent] = []
        self._frames_processed: int = 0
        self._first_frame_idx: Optional[int] = None
        self._last_frame_idx: Optional[int] = None

        logger.info(
            "ContactTrackerV2 (fase 1) initialized | slots=%d pairs=%d fps=%.1f outdir=%s",
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
        if self._first_frame_idx is None:
            self._first_frame_idx = frame_idx
        self._last_frame_idx = frame_idx
        time_sec = frame_idx / self.fps

        slot_dets = self._map_detections_to_slots(detections, slot_centroids)
        for slot_idx in range(self.num_slots):
            self.body_length_estimators[slot_idx].observe(slot_dets.get(slot_idx))
            self.velocity_estimators[slot_idx].update(
                slot_centroids[slot_idx] if slot_idx < len(slot_centroids) else None
            )

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
            self.bout_manager.process_event(event)
            self.dynamics_tracker.process_event(event)   # cambio D
            self._all_events.append(event)

        self._frames_processed += 1

    def write_merged_placeholder(
        self,
        slot_centroids: List[Optional[Tuple[float, float]]],
        frame_idx: int,
    ) -> None:
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
            self.dynamics_tracker.process_event(event)
            self._all_events.append(event)
        self._frames_processed += 1

    def finalize(self) -> Dict[str, Any]:
        logger.info("Finalizing ContactTrackerV2 (fase 1): %d frames, %d events",
                    self._frames_processed, len(self._all_events))

        bouts = self.bout_manager.close_all()
        dynamics_events = self.dynamics_tracker.close_all()
        logger.info("Total valid bouts: %d | dynamics events (no-contact): %d",
                    len(bouts), len(dynamics_events))

        # Cambio E: propagar initiator del bout a cada evento del bout
        self._propagate_bout_initiator(bouts)

        # Hoja A
        self._write_per_frame_csv()
        # Hoja B
        self._write_dynamics_csv(dynamics_events)
        # Bouts
        self._write_bout_csv(bouts)

        summary = self._build_summary(bouts, dynamics_events)
        json_path = self.output_dir / "session_summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary written: %s", json_path)

        # Individual metrics: SE DEJA INTACTO (envuelto en try/except)
        calc = None
        try:
            from src.common.individual_metrics import IndividualMetricsCalculator
            calc = IndividualMetricsCalculator(num_slots=self.num_slots)
            calc.process_bouts(bouts, self._all_events)
            calc.write_json(self.output_dir / "individual_summary.json")
            logger.info("Individual metrics written")
        except Exception as e:
            logger.warning("Individual metrics generation failed (esperado si el módulo no está): %s", e)
            calc = None

        try:
            self._generate_report(summary, bouts, calc=calc)
        except Exception as e:
            logger.warning("PDF report generation failed: %s", e)

        return summary

    # ------------------------- Internal helpers -------------------------

    def _propagate_bout_initiator(self, bouts: List[Bout]) -> None:
        """Asigna a cada evento el initiator final de su bout (cambio E)."""
        initiator_by_bout = {b.bout_id: b.investigator_role for b in bouts}
        for ev in self._all_events:
            if ev.bout_id and ev.bout_id in initiator_by_bout:
                ev.bout_initiator = initiator_by_bout[ev.bout_id]

    def _map_detections_to_slots(
        self,
        detections: List[Detection],
        slot_centroids: List[Optional[Tuple[float, float]]],
    ) -> Dict[int, Optional[Detection]]:
        """Asignación greedy por proximidad de centroide (preservado)."""
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
                if di in used or det is None:
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
        if not self._all_events:
            logger.warning("No events to write")
            return
        first_row = self._all_events[0].to_csv_row()
        fieldnames = list(first_row.keys())
        with self._csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in self._all_events:
                writer.writerow(event.to_csv_row())
        logger.info("Per-frame CSV (Hoja A): %s (%d rows)", self._csv_path, len(self._all_events))

    def _write_dynamics_csv(self, dynamics_events: List[DynamicsEvent]) -> None:
        """Hoja B: approach/avoid SIN contacto (cambio D)."""
        fieldnames = [
            "id", "pair_key", "kind", "mover",
            "start_frame", "end_frame",
            "start_time", "end_time", "duration_sec",
            "start_time_sec", "end_time_sec", "n_frames",
            "start_dist_bl", "end_dist_bl", "mean_delta_bls",
        ]
        with self._dynamics_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ev in dynamics_events:
                writer.writerow(ev.to_csv_row())
        logger.info("Dynamics CSV (Hoja B): %s (%d events)",
                    self._dynamics_csv_path, len(dynamics_events))

    def _write_bout_csv(self, bouts: List[Bout]) -> Path:
        path = self.output_dir / "contact_bouts.csv"
        fieldnames = [
            "id", "pair_key", "family", "contact_type", "name_contact",
            "start_frame", "end_frame",
            "start_time", "end_time", "duration_sec",
            "start_time_sec", "end_time_sec", "n_frames",
            "mean_nose_nose_dist_bl", "mean_centroid_dist_bl",
            "mean_mask_iou", "mean_mask_contact_bl", "mean_velocity_i_bls", "mean_velocity_j_bls",
            "peak_score", "initiator",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for bout in bouts:
                writer.writerow(bout.to_csv_row())
        logger.info("Bouts CSV: %s (%d bouts)", path, len(bouts))
        return path

    def _build_summary(self, bouts: List[Bout], dynamics_events: List[DynamicsEvent]) -> Dict[str, Any]:
        type_summary: Dict[str, Dict[str, Any]] = {}
        for ct in ContactType.all_contact_types():
            bouts_of_type = [b for b in bouts if b.contact_type == ct]
            total_frames = sum(b.n_frames for b in bouts_of_type)
            total_duration = sum(b.duration_sec for b in bouts_of_type)
            type_summary[ct.value] = {
                "family": CONTACT_FAMILY.get(ct, Family.NONE).value,
                "total_bouts": len(bouts_of_type),
                "total_frames": total_frames,
                "total_duration_sec": round(total_duration, 3),
                "mean_bout_duration_sec": round(total_duration / max(len(bouts_of_type), 1), 3),
            }

        # Resumen por familia (cambio A)
        family_summary: Dict[str, Dict[str, Any]] = {}
        for fam in [Family.INVESTIGATIVE, Family.AFFILIATIVE, Family.NON_CONTACT]:
            fam_bouts = [b for b in bouts if b.family == fam]
            family_summary[fam.value] = {
                "total_bouts": len(fam_bouts),
                "total_duration_sec": round(sum(b.duration_sec for b in fam_bouts), 3),
            }

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

        quality_flags = {
            "stale_keypoints": sum(1 for e in self._all_events if e.stale_keypoints),
            "high_mask_overlap": sum(1 for e in self._all_events if e.high_mask_overlap),
            "missing_keypoints": sum(1 for e in self._all_events if e.missing_keypoints),
            "single_detection": sum(1 for e in self._all_events if e.single_detection),
            "merged_state": sum(1 for e in self._all_events if e.merged_state),
            "fol_used_centroid": sum(1 for e in self._all_events if e.fol_used_centroid),
        }

        activation = float(self.config.get("activation_threshold", 0.5))
        rare = float(self.config.get("activation_threshold_rare", 0.35))
        concurrent_frames = 0
        for e in self._all_events:
            if len(e.scores.active_types(activation, rare)) > 1:
                concurrent_frames += 1

        concurrent_pairs: Dict[str, int] = {}
        for e in self._all_events:
            if e.contact_type != ContactType.NONE and e.secondary_type != ContactType.NONE:
                key = f"{e.contact_type.value}+{e.secondary_type.value}"
                concurrent_pairs[key] = concurrent_pairs.get(key, 0) + 1

        # Resumen de dinámica (cambio D)
        dyn_summary = {
            "approach_events": sum(1 for d in dynamics_events if d.kind == "approach"),
            "avoidance_events": sum(1 for d in dynamics_events if d.kind == "avoidance"),
            "total_dynamics_events": len(dynamics_events),
        }

        return {
            "metadata": {
                "video_path": self.video_path,
                "fps": self.fps,
                "num_slots": self.num_slots,
                "num_pairs": len(self.classifiers),
                "first_frame_idx": self._first_frame_idx,
                "last_frame_idx": self._last_frame_idx,
                "frames_processed": self._frames_processed,
                "phase": "1",
            },
            "parameters": dict(self.config),
            "contact_type_summary": type_summary,
            "family_summary": family_summary,
            "pair_summary": pair_summary,
            "dynamics_summary": dyn_summary,
            "quality_flags": quality_flags,
            "concurrent_frames": concurrent_frames,
            "concurrent_pairs": concurrent_pairs,
            "total_bouts": len(bouts),
        }

    def _generate_report(self, summary: Dict[str, Any], bouts: List[Bout], calc: Any = None) -> None:
        """Genera report.pdf (igual que antes; individual pages intactas)."""
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
            # Página 1: duración por tipo
            fig, ax = plt.subplots(figsize=(11, 7))
            types = list(summary["contact_type_summary"].keys())
            durations = [summary["contact_type_summary"][t]["total_duration_sec"] for t in types]
            ax.bar(types, durations)
            ax.set_ylabel("Total duration (s)"); ax.set_title("Contact duration by type")
            ax.set_xlabel("Contact type")
            for i, v in enumerate(durations):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
            pdf.savefig(fig); plt.close(fig)

            # Página 2: duración por par
            fig, ax = plt.subplots(figsize=(11, 7))
            pairs = list(summary["pair_summary"].keys())
            pair_durs = [summary["pair_summary"][p]["total_duration_sec"] for p in pairs]
            ax.bar(pairs, pair_durs)
            ax.set_ylabel("Total duration (s)"); ax.set_title("Contact duration by pair")
            ax.set_xlabel("Pair")
            pdf.savefig(fig); plt.close(fig)

            # Página 3: texto resumen
            fig, ax = plt.subplots(figsize=(11, 7)); ax.axis("off")
            text = "SESSION SUMMARY (fase 1)\n\n"
            text += f"Video: {summary['metadata']['video_path']}\n"
            text += f"FPS: {summary['metadata']['fps']}\n"
            text += f"Animals: {summary['metadata']['num_slots']}\n"
            text += f"Pairs: {summary['metadata']['num_pairs']}\n"
            text += f"Frames processed: {summary['metadata']['frames_processed']}\n"
            text += f"Total bouts: {summary['total_bouts']}\n"
            text += f"Approach events: {summary['dynamics_summary']['approach_events']}\n"
            text += f"Avoidance events: {summary['dynamics_summary']['avoidance_events']}\n"
            text += f"Concurrent frames (>1 type): {summary['concurrent_frames']}\n\n"
            text += "QUALITY FLAGS:\n"
            for k, v in summary["quality_flags"].items():
                text += f"  {k}: {v}\n"
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
                    verticalalignment="top", family="monospace")
            pdf.savefig(fig); plt.close(fig)

            # Páginas individuales (intactas)
            if calc is not None:
                try:
                    from src.common.individual_metrics import add_individual_pages_to_pdf
                    add_individual_pages_to_pdf(pdf, calc.build_summary(), bouts)
                    logger.info("Individual pages added to PDF report")
                except Exception as e:
                    logger.warning("Individual pages generation failed: %s", e)

        logger.info("PDF report written: %s", pdf_path)