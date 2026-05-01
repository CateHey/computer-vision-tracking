"""
Individual metrics — Análisis por rata individual + pares direccionados.

Calcula métricas de comportamiento social a nivel individual:
  - Quién inicia cada bout (incluyendo bilaterales con Opción A: movimiento previo)
  - Distribución de tipos de contacto preferidos por rata
  - Tiempo activo (investigando) vs pasivo (siendo investigado)
  - Pares direccionados (R1→R2 vs R2→R1)

Se integra con ContactTrackerV2: lee los Bouts ya generados y produce
métricas extra, sin re-procesar el video.

Genera 2 outputs:
  - individual_summary.json (datos estructurados para análisis posterior)
  - Páginas extras en el report.pdf (visualización)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.contacts_v2 import (
    Bout,
    ContactEvent,
    ContactType,
    CONTACT_TYPE_NAMES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1 — DATACLASS DE MÉTRICAS POR INDIVIDUO
# ============================================================================

@dataclass
class IndividualMetrics:
    """Métricas para UN animal específico (ej. R1).

    Se llena progresivamente al iterar sobre los bouts.
    """
    slot_idx: int
    label: str

    # Eventos iniciados
    total_events_initiated: int = 0

    # Distribución por tipo (cuántos N2N inició, cuántos N2AG, etc.)
    contact_distribution: Dict[str, int] = field(default_factory=dict)

    # Tipo preferido (calculado al final con argmax)
    preferred_contact_type: str = "none"
    preferred_contact_name: str = "None"

    # Tiempo activo: total de segundos que esta rata fue iniciadora
    active_time_sec: float = 0.0

    # Tiempo pasivo: total de segundos donde la otra fue iniciadora
    passive_time_sec: float = 0.0

    # Activity ratio: active / passive (>1 = más activa)
    activity_ratio: float = 0.0

    # Total tiempo en cualquier contacto (activo o pasivo)
    total_contact_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialización para JSON."""
        return {
            "slot_idx": self.slot_idx,
            "label": self.label,
            "total_events_initiated": self.total_events_initiated,
            "contact_distribution": dict(self.contact_distribution),
            "preferred_contact_type": self.preferred_contact_type,
            "preferred_contact_name": self.preferred_contact_name,
            "active_time_sec": round(self.active_time_sec, 3),
            "passive_time_sec": round(self.passive_time_sec, 3),
            "activity_ratio": round(self.activity_ratio, 3),
            "total_contact_time_sec": round(self.total_contact_time_sec, 3),
        }


# ============================================================================
# SECTION 2 — DATACLASS DE PAR DIRECCIONADO
# ============================================================================

@dataclass
class DirectedPairMetrics:
    """Métricas para una dirección específica de un par (ej. R1→R2)."""
    initiator_slot: int
    target_slot: int
    label: str

    events_count: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    total_duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initiator_slot": self.initiator_slot,
            "target_slot": self.target_slot,
            "label": self.label,
            "events_count": self.events_count,
            "by_type": dict(self.by_type),
            "total_duration_sec": round(self.total_duration_sec, 3),
        }


# ============================================================================
# SECTION 3 — DETECTOR DE INICIADOR PARA BOUTS BILATERALES (Opción A)
# ============================================================================

# Tipos asimétricos: ya tienen investigator_role, no hace falta calcular
ASYMMETRIC_TYPES = {ContactType.N2AG, ContactType.FOL, ContactType.N2B}

# Tipos bilaterales: hay que calcular el iniciador con velocidad/aproximación
BILATERAL_TYPES = {ContactType.N2N, ContactType.T2T, ContactType.SBS}


def determine_initiator_for_bilateral(
    bout: Bout,
    all_events: List[ContactEvent],
    pair_key: str,
    lookback_frames: int = 10,
) -> Optional[int]:
    """Decide quién inició un bout BILATERAL (N2N, T2T, SBS).

    Opción A: la rata que se acercó MÁS a la otra en los frames previos
    al inicio del bout es la iniciadora.

    Lógica:
      1. Tomar los eventos de los `lookback_frames` antes del start_frame
      2. Para cada rata, calcular cuánto se redujo su distancia a la otra
         (delta_centroid_dist)
      3. La rata que más se aproximó es la iniciadora
      4. Si ambas tienen velocidades bajas o similares → None

    Args:
        bout: bout a analizar (debe tener pair_key tipo "0_1")
        all_events: lista completa de events del CSV per-frame
        pair_key: clave del par (ej. "0_1") para filtrar events
        lookback_frames: cuántos frames mirar antes del inicio

    Returns:
        slot_idx del iniciador (0 o 1), o None si no se puede decidir.
    """
    # Parsear pair_key para obtener slot_i, slot_j
    try:
        parts = pair_key.split("_")
        slot_i = int(parts[0])
        slot_j = int(parts[1])
    except (ValueError, IndexError):
        return None

    # Filtrar events del par y ordenar por frame
    pair_events = [
        e for e in all_events
        if e.pair_key == pair_key and e.frame_idx < bout.start_frame
    ]
    if not pair_events:
        return None

    # Tomar los últimos N events antes del start
    pair_events.sort(key=lambda e: e.frame_idx)
    lookback = pair_events[-lookback_frames:]
    if len(lookback) < 2:
        return None

    # Velocidad media de cada rata en el lookback
    avg_vel_i = sum(e.velocity_i_bls for e in lookback) / len(lookback)
    avg_vel_j = sum(e.velocity_j_bls for e in lookback) / len(lookback)

    # Cambio de distancia: si la distancia bajó → ratas se acercaron
    first_dist = lookback[0].centroid_dist_bl
    last_dist = lookback[-1].centroid_dist_bl

    if not math.isfinite(first_dist) or not math.isfinite(last_dist):
        return None

    # Si ambas casi quietas → no se puede decidir
    min_speed_threshold = 0.05  # BL/s
    if avg_vel_i < min_speed_threshold and avg_vel_j < min_speed_threshold:
        return None

    # La que se movió más rápido es la iniciadora
    # (asumiendo que ambas se acercaron, la más rápida fue la activa)
    # Esto funciona porque si una está casi quieta y la otra se acercó,
    # el delta de velocidad es claro
    if avg_vel_i > avg_vel_j * 1.3:
        return slot_i  # i se movió notablemente más
    elif avg_vel_j > avg_vel_i * 1.3:
        return slot_j  # j se movió notablemente más
    else:
        # Velocidades similares → no se puede determinar con seguridad
        return None


# ============================================================================
# SECTION 4 — CALCULADOR PRINCIPAL
# ============================================================================

class IndividualMetricsCalculator:
    """Calcula métricas individuales a partir de bouts + eventos.

    Uso:
        calc = IndividualMetricsCalculator(num_slots=2)
        calc.process_bouts(bouts, all_events)
        summary = calc.build_summary()
        calc.write_json(output_path)
    """

    def __init__(
        self,
        num_slots: int,
        slot_labels: Optional[List[str]] = None,
    ):
        self.num_slots = num_slots
        self.slot_labels = slot_labels or [f"R{i+1}" for i in range(num_slots)]

        # Inicializar métricas por individuo
        self.individuals: Dict[int, IndividualMetrics] = {
            i: IndividualMetrics(slot_idx=i, label=self.slot_labels[i])
            for i in range(num_slots)
        }

        # Pares direccionados: para N animales, N*(N-1) pares
        # (R1→R2 y R2→R1 son distintos)
        self.directed_pairs: Dict[Tuple[int, int], DirectedPairMetrics] = {}
        for i in range(num_slots):
            for j in range(num_slots):
                if i != j:
                    label = f"{self.slot_labels[i]}_to_{self.slot_labels[j]}"
                    self.directed_pairs[(i, j)] = DirectedPairMetrics(
                        initiator_slot=i,
                        target_slot=j,
                        label=label,
                    )

        # Estadísticas globales
        self._total_bouts_processed: int = 0
        self._bouts_with_unknown_initiator: int = 0

    def process_bouts(
        self,
        bouts: List[Bout],
        all_events: List[ContactEvent],
    ) -> None:
        """Procesa todos los bouts y eventos, llenando métricas."""
        for bout in bouts:
            self._total_bouts_processed += 1

            initiator, target = self._determine_initiator(bout, all_events)
            if initiator is None or target is None:
                self._bouts_with_unknown_initiator += 1
                continue

            self._record_bout(bout, initiator, target)

        # Cálculos derivados al final
        self._finalize_individuals()

    def _determine_initiator(
        self,
        bout: Bout,
        all_events: List[ContactEvent],
    ) -> Tuple[Optional[int], Optional[int]]:
        """Decide iniciador y target para un bout.

        Returns:
            (initiator_slot, target_slot) o (None, None)
        """
        # Parsear el pair_key (ej. "0_1" → slot_i=0, slot_j=1)
        try:
            parts = bout.pair_key.split("_")
            slot_i = int(parts[0])
            slot_j = int(parts[1])
        except (ValueError, IndexError):
            return None, None

        # Caso 1: bouts asimétricos — usar investigator_role
        if bout.contact_type in ASYMMETRIC_TYPES:
            if bout.investigator_role == "i":
                return slot_i, slot_j
            elif bout.investigator_role == "j":
                return slot_j, slot_i
            else:
                return None, None

        # Caso 2: bouts bilaterales — usar Opción A
        if bout.contact_type in BILATERAL_TYPES:
            initiator = determine_initiator_for_bilateral(
                bout, all_events, bout.pair_key,
            )
            if initiator is None:
                return None, None
            target = slot_j if initiator == slot_i else slot_i
            return initiator, target

        return None, None

    def _record_bout(
        self,
        bout: Bout,
        initiator: int,
        target: int,
    ) -> None:
        """Registra un bout en las dataclasses correspondientes."""
        ct = bout.contact_type.value
        duration = bout.duration_sec

        # Iniciador: aumenta total_events_initiated, distribution, active_time
        ind_init = self.individuals[initiator]
        ind_init.total_events_initiated += 1
        ind_init.contact_distribution[ct] = ind_init.contact_distribution.get(ct, 0) + 1
        ind_init.active_time_sec += duration
        ind_init.total_contact_time_sec += duration

        # Target: aumenta passive_time
        ind_target = self.individuals[target]
        ind_target.passive_time_sec += duration
        ind_target.total_contact_time_sec += duration

        # Par direccionado
        pair_metrics = self.directed_pairs.get((initiator, target))
        if pair_metrics is not None:
            pair_metrics.events_count += 1
            pair_metrics.by_type[ct] = pair_metrics.by_type.get(ct, 0) + 1
            pair_metrics.total_duration_sec += duration

    def _finalize_individuals(self) -> None:
        """Calcula métricas derivadas: preferred_type y activity_ratio."""
        for ind in self.individuals.values():
            # Preferred type: argmax de contact_distribution
            if ind.contact_distribution:
                preferred = max(
                    ind.contact_distribution.items(),
                    key=lambda kv: kv[1],
                )[0]
                ind.preferred_contact_type = preferred
                ind.preferred_contact_name = CONTACT_TYPE_NAMES.get(preferred, preferred)
            else:
                ind.preferred_contact_type = "none"
                ind.preferred_contact_name = "None"

            # Activity ratio: active / passive
            if ind.passive_time_sec > 1e-6:
                ind.activity_ratio = ind.active_time_sec / ind.passive_time_sec
            elif ind.active_time_sec > 1e-6:
                ind.activity_ratio = float("inf")  # solo activa, nunca pasiva
            else:
                ind.activity_ratio = 0.0

    def build_summary(self) -> Dict[str, Any]:
        """Construye el dict final que va al individual_summary.json."""
        return {
            "metadata": {
                "num_slots": self.num_slots,
                "slot_labels": self.slot_labels,
                "total_bouts_processed": self._total_bouts_processed,
                "bouts_with_unknown_initiator": self._bouts_with_unknown_initiator,
            },
            "individual_metrics": {
                self.slot_labels[i]: self.individuals[i].to_dict()
                for i in range(self.num_slots)
            },
            "directed_pairs": {
                pair.label: pair.to_dict()
                for pair in self.directed_pairs.values()
            },
        }

    def write_json(self, output_path: Path) -> None:
        """Escribe individual_summary.json al disco."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Manejar inf en activity_ratio para JSON válido
        summary = self.build_summary()
        for ind in summary["individual_metrics"].values():
            if not math.isfinite(ind["activity_ratio"]):
                ind["activity_ratio"] = "infinity"  # string en JSON

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)


# ============================================================================
# SECTION 5 — GENERADOR DE PÁGINAS PARA EL PDF
# ============================================================================

def add_individual_pages_to_pdf(
    pdf,
    individual_summary: Dict[str, Any],
    bouts: List[Bout],
) -> None:
    """Añade páginas al report.pdf con análisis individual.

    Páginas añadidas:
      1. Tabla resumen por rata
      2. Bar chart de iniciación por tipo de contacto
      3. Pie charts de distribución por rata
      4. Comparación activo vs pasivo
      5. Texto descriptivo automático
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping individual pages")
        return

    _add_individual_summary_table(pdf, individual_summary, plt)
    _add_initiation_bar_chart(pdf, individual_summary, plt)
    _add_distribution_pie_charts(pdf, individual_summary, plt)
    _add_active_passive_comparison(pdf, individual_summary, plt)
    _add_descriptive_text(pdf, individual_summary, plt)


def _add_individual_summary_table(pdf, summary: Dict, plt) -> None:
    """Página: tabla con todas las métricas de cada rata."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.set_title("Individual Metrics Summary", fontsize=14, fontweight="bold", pad=20)

    individual_metrics = summary.get("individual_metrics", {})
    if not individual_metrics:
        ax.text(0.5, 0.5, "No individual metrics available",
                ha="center", va="center", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Construir tabla
    headers = ["Rat", "Events Init.", "Preferred", "Active (s)", "Passive (s)", "Activity Ratio"]
    rows = []
    for label, data in individual_metrics.items():
        ratio = data.get("activity_ratio", 0)
        if isinstance(ratio, str):
            ratio_str = ratio
        else:
            ratio_str = f"{ratio:.2f}"

        rows.append([
            label,
            str(data.get("total_events_initiated", 0)),
            data.get("preferred_contact_name", "None"),
            f"{data.get('active_time_sec', 0):.2f}",
            f"{data.get('passive_time_sec', 0):.2f}",
            ratio_str,
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#4a4a4a"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # Color text in header to white
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_text_props(color="white", fontweight="bold")

    pdf.savefig(fig)
    plt.close(fig)


def _add_initiation_bar_chart(pdf, summary: Dict, plt) -> None:
    """Página: bar chart de iniciaciones por tipo, agrupado por rata."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 7))
    individual_metrics = summary.get("individual_metrics", {})
    if not individual_metrics:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Tipos de contacto en orden fijo
    types = ["N2N", "N2AG", "T2T", "FOL", "SBS", "N2B"]
    labels = list(individual_metrics.keys())

    # Matriz: filas = ratas, columnas = tipos
    data_matrix = []
    for label in labels:
        dist = individual_metrics[label].get("contact_distribution", {})
        data_matrix.append([dist.get(t, 0) for t in types])

    x = np.arange(len(types))
    width = 0.8 / max(len(labels), 1)

    for i, label in enumerate(labels):
        offset = (i - len(labels) / 2 + 0.5) * width
        ax.bar(x + offset, data_matrix[i], width, label=label)

    ax.set_xlabel("Contact Type")
    ax.set_ylabel("Number of Events Initiated")
    ax.set_title("Initiation by Contact Type (per rat)")
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    pdf.savefig(fig)
    plt.close(fig)


def _add_distribution_pie_charts(pdf, summary: Dict, plt) -> None:
    """Página: pie chart por rata mostrando % de cada tipo iniciado."""
    individual_metrics = summary.get("individual_metrics", {})
    if not individual_metrics:
        return

    n = len(individual_metrics)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, individual_metrics.items()):
        dist = data.get("contact_distribution", {})
        if not dist:
            ax.text(0.5, 0.5, f"{label}\n(no events)", ha="center", va="center")
            ax.axis("off")
            continue

        # Filtrar tipos con conteo > 0
        sizes = [v for v in dist.values() if v > 0]
        labels_pie = [k for k, v in dist.items() if v > 0]

        if sizes:
            ax.pie(sizes, labels=labels_pie, autopct="%1.0f%%", startangle=90)
        ax.set_title(f"{label} — Initiated Contacts")

    fig.suptitle("Distribution of Contact Types per Rat", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_active_passive_comparison(pdf, summary: Dict, plt) -> None:
    """Página: barras horizontales activo vs pasivo por rata."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 7))
    individual_metrics = summary.get("individual_metrics", {})
    if not individual_metrics:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)
        return

    labels = list(individual_metrics.keys())
    active = [individual_metrics[l].get("active_time_sec", 0) for l in labels]
    passive = [individual_metrics[l].get("passive_time_sec", 0) for l in labels]

    y_pos = np.arange(len(labels))
    bar_height = 0.35

    ax.barh(y_pos - bar_height/2, active, bar_height, label="Active (initiator)", color="#2ecc71")
    ax.barh(y_pos + bar_height/2, passive, bar_height, label="Passive (target)", color="#e74c3c")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Active vs Passive Contact Time per Rat")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # Annotar valores
    for i, (a, p) in enumerate(zip(active, passive)):
        ax.text(a + 0.1, i - bar_height/2, f"{a:.1f}s", va="center", fontsize=9)
        ax.text(p + 0.1, i + bar_height/2, f"{p:.1f}s", va="center", fontsize=9)

    pdf.savefig(fig)
    plt.close(fig)


def _add_descriptive_text(pdf, summary: Dict, plt) -> None:
    """Página: texto generado automáticamente con conclusiones legibles."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.set_title("Behavioral Analysis", fontsize=14, fontweight="bold", pad=20)

    individual_metrics = summary.get("individual_metrics", {})
    directed_pairs = summary.get("directed_pairs", {})

    # Total events para % cálculos
    total_events = sum(
        m.get("total_events_initiated", 0)
        for m in individual_metrics.values()
    )

    text_lines = []
    text_lines.append("INDIVIDUAL BEHAVIOR ANALYSIS\n")
    text_lines.append("=" * 60)

    for label, data in individual_metrics.items():
        events = data.get("total_events_initiated", 0)
        pct = (100.0 * events / total_events) if total_events > 0 else 0
        pref = data.get("preferred_contact_name", "None")
        active = data.get("active_time_sec", 0)
        passive = data.get("passive_time_sec", 0)
        ratio = data.get("activity_ratio", 0)

        # Interpretar ratio
        if isinstance(ratio, str):
            activity_label = "exclusively active"
        elif ratio > 1.5:
            activity_label = f"more active ({ratio:.2f}x more than passive)"
        elif ratio < 0.67 and ratio > 0:
            activity_label = f"more passive ({1/ratio:.2f}x more than active)"
        elif ratio == 0:
            activity_label = "no contact activity"
        else:
            activity_label = "balanced active/passive"

        text_lines.append(f"\n{label}:")
        text_lines.append(f"  - Initiated {events} contacts ({pct:.1f}% of total)")
        text_lines.append(f"  - Preferred contact type: {pref}")
        text_lines.append(f"  - Active time: {active:.2f}s | Passive time: {passive:.2f}s")
        text_lines.append(f"  - Behavior: {activity_label}")

    # Análisis de pares direccionados
    text_lines.append("\n" + "=" * 60)
    text_lines.append("DIRECTED PAIR ANALYSIS\n")

    sorted_pairs = sorted(
        directed_pairs.items(),
        key=lambda kv: kv[1].get("events_count", 0),
        reverse=True,
    )

    for pair_label, pair_data in sorted_pairs:
        count = pair_data.get("events_count", 0)
        if count == 0:
            continue
        duration = pair_data.get("total_duration_sec", 0)
        by_type = pair_data.get("by_type", {})
        if by_type:
            top_type = max(by_type.items(), key=lambda kv: kv[1])[0]
            top_name = CONTACT_TYPE_NAMES.get(top_type, top_type)
        else:
            top_name = "N/A"

        text_lines.append(f"\n  {pair_label}:")
        text_lines.append(f"    - {count} events | {duration:.2f}s total")
        text_lines.append(f"    - Most common: {top_name}")

    final_text = "\n".join(text_lines)

    ax.text(
        0.05, 0.95, final_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
    )

    pdf.savefig(fig)
    plt.close(fig)