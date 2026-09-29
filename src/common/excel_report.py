"""
Consolidate the contact analysis outputs into a single Excel workbook.

Replaces the scattered per-run CSV/JSON files with one results.xlsx holding
every table as a separate sheet. This is the file the HTML report viewer reads
and the file to archive for a session.

Handles both contact modules: `contacts.py` (v1, 6 types incl. T2T) and
`contacts_v2.py` (v2, 5 types, families, dynamics, individual metrics). The
contact types are read from the data rather than assumed, and the v2-only
sheets are added when their source files are present.

Sheets (in order):
    Summary     — headline session numbers, one row per contact type
    Events      — cleaned behavioural events (the reportable result)
    PerFrame    — full per-frame geometry with raw + cleaned labels
    Bouts       — inline bouts (diagnostic)
    Dynamics    — approach / avoid episodes without contact   (v2 only)
    Individual  — per-animal locomotion metrics               (v2 only)
    ByType      — raw vs cleaned comparison per contact type
    Global      — raw vs cleaned comparison, session totals
    Parameters  — every parameter actually applied, flattened

Usage:
    from src.common.excel_report import build_excel_report
    build_excel_report(run_dir / "contacts", replace_sources=True)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

WORKBOOK_NAME = "results.xlsx"

# Excel hard limit is 1,048,576 rows. Stay clear of it.
MAX_SHEET_ROWS = 1_000_000

# v1 (contacts.py) has six types; v2 (contacts_v2.py) dropped T2T. The set used
# for a given workbook is resolved from the data, so both are supported.
CONTACT_TYPES_V1 = ["N2N", "N2AG", "N2B", "T2T", "FOL", "SBS"]
CONTACT_TYPES_V2 = ["N2N", "N2AG", "N2B", "FOL", "SBS"]
CONTACT_TYPES = CONTACT_TYPES_V1

TYPE_LABELS = {
    "N2N": "Nose-to-nose",
    "N2AG": "Nose-to-anogenital",
    "N2B": "Nose-to-body",
    "T2T": "Tail-to-tail",
    "FOL": "Following",
    "SBS": "Side-by-side",
    "NC": "No contact",
}

# Source files absorbed into the workbook. Removed when replace_sources=True.
SOURCE_FILES = [
    "contacts_per_frame.csv",
    "contacts_real_per_frame.csv",
    "contacts_real_events.csv",
    "contact_bouts.csv",
    "dynamics_no_contact.csv",       # v2 only
    "session_summary.json",
    "session_summary_real.json",
    "individual_summary.json",       # v2 only
]

SOURCE_FILES_IN_REPORTS = [
    "comparison_by_type.csv",
    "comparison_global.csv",
]


# ── helpers ────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV if it exists, returning None on any failure."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else df
    except Exception as e:
        logger.warning("Could not read %s: %s", path.name, e)
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file if it exists, returning {} on any failure."""
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read %s: %s", path.name, e)
        return {}


def _flatten(obj: Any, prefix: str = "") -> List[Dict[str, Any]]:
    """Flatten a nested dict into [{section, parameter, value}] rows."""
    rows: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            rows.extend(_flatten(val, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        # Lists of scalars become a single comma-joined value; lists of dicts
        # are indexed so nothing is silently dropped.
        if all(not isinstance(v, (dict, list)) for v in obj):
            rows.append({"key": prefix, "value": ", ".join(str(v) for v in obj)})
        else:
            for i, val in enumerate(obj):
                rows.extend(_flatten(val, f"{prefix}[{i}]"))
    else:
        rows.append({"key": prefix, "value": obj})
    return rows


# Sections describing the run itself. They come from session_summary.json, which
# a previous consolidation has already deleted, so on a re-run they are carried
# forward from the existing workbook rather than silently lost.
CARRIED_SECTIONS = ("video", "detection_thresholds", "quality_flags")


def _params_sheet(
    summary: Dict[str, Any],
    summary_real: Dict[str, Any],
    carried: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the Parameters sheet from both session summaries.

    Args:
        summary: session_summary.json contents (may be empty on a re-run).
        summary_real: session_summary_real.json contents.
        carried: Parameters sheet of an existing workbook, used to preserve the
            run-provenance sections when their source JSON is gone.
    """
    rows: List[Dict[str, Any]] = []

    def _add(section: str, source: Dict[str, Any]) -> None:
        for row in _flatten(source):
            rows.append({
                "section": section,
                "parameter": row["key"],
                "value": row["value"],
            })

    _add("video", summary.get("metadata", {}))
    _add("detection_thresholds", summary.get("parameters", {}))
    _add("quality_flags", summary.get("quality", {}))
    _add("postprocessing", summary_real.get("parameters", {}))
    _add("postprocessing_metadata", summary_real.get("metadata", {}))
    _add("filtering_impact", summary_real.get("filtering_impact", {}))

    present = {r["section"] for r in rows}
    if carried is not None and not carried.empty and "section" in carried.columns:
        missing = [s for s in CARRIED_SECTIONS if s not in present]
        if missing:
            kept = carried[carried["section"].isin(missing)]
            if not kept.empty:
                logger.info(
                    "Carrying forward %d parameter rows (%s) from the existing workbook",
                    len(kept), ", ".join(sorted(set(kept["section"]))),
                )
                rows = kept.to_dict("records") + rows

    if not rows:
        return pd.DataFrame(columns=["section", "parameter", "value"])
    return pd.DataFrame(rows)[["section", "parameter", "value"]]


def _individual_sheet(data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Flatten individual_summary.json (contacts_v2) into one row per animal.

    The file's shape is not fixed across versions, so accept either a mapping of
    animal id to metrics or a list of per-animal records, and fall back to a
    flat key/value table for anything else.
    """
    if not data:
        return None

    per_animal = data.get("per_animal", data.get("animals", data))

    rows: List[Dict[str, Any]] = []
    if isinstance(per_animal, dict) and per_animal:
        for key, metrics in per_animal.items():
            if isinstance(metrics, dict):
                row: Dict[str, Any] = {"animal": key}
                for flat in _flatten(metrics):
                    row[flat["key"]] = flat["value"]
                rows.append(row)
    elif isinstance(per_animal, list):
        for i, metrics in enumerate(per_animal):
            if isinstance(metrics, dict):
                row = {"animal": metrics.get("slot", metrics.get("id", i))}
                for flat in _flatten(metrics):
                    row.setdefault(flat["key"], flat["value"])
                rows.append(row)

    if rows:
        return pd.DataFrame(rows)

    # Unrecognised shape — keep the numbers rather than dropping the file.
    flat_rows = [{"metric": r["key"], "value": r["value"]} for r in _flatten(data)]
    return pd.DataFrame(flat_rows) if flat_rows else None


def _resolve_types(
    per_frame: Optional[pd.DataFrame],
    events: Optional[pd.DataFrame],
) -> List[str]:
    """Decide which contact taxonomy this run used.

    contacts_v2 dropped T2T, so assuming the v1 six would add a permanently
    empty row. Detect it from the data: a v2 run carries a `family` column, and
    a T2T label anywhere means v1.
    """
    if per_frame is not None and "family" in per_frame.columns:
        return list(CONTACT_TYPES_V2)

    seen = set()
    for df in (per_frame, events):
        if df is None:
            continue
        for col in ("contact_type", "real_type"):
            if col in df.columns:
                seen |= {str(v).strip().upper() for v in df[col].dropna().unique()}
    if "T2T" in seen:
        return list(CONTACT_TYPES_V1)
    return list(CONTACT_TYPES_V2)


def _summary_sheet(
    summary: Dict[str, Any],
    summary_real: Dict[str, Any],
    events: Optional[pd.DataFrame],
    types: List[str],
) -> pd.DataFrame:
    """Build the headline Summary sheet: one row per contact type.

    Numbers come from the post-processed (cleaned) results, which are the
    reportable ones. Falls back to the events table if the summary JSON is
    missing.
    """
    events_by_type = summary_real.get("real_summary", {}).get("events_by_type", {})
    total_sec = summary_real.get("metadata", {}).get("total_duration_sec")
    if total_sec is None:
        total_sec = summary.get("metadata", {}).get("video_duration_sec", 0.0)

    rows: List[Dict[str, Any]] = []
    for ct in types:
        entry = events_by_type.get(ct)

        if entry is None and events is not None and "contact_type" in events.columns:
            # Derive from the events table when the summary JSON is unavailable.
            subset = events[events["contact_type"] == ct]
            count = len(subset)
            total = float(subset["duration_sec"].sum()) if count else 0.0
            entry = {
                "count": count,
                "total_sec": round(total, 2),
                "mean_sec": round(total / count, 2) if count else 0.0,
                "pct_of_session": round(total / total_sec * 100, 2) if total_sec else 0.0,
            }
        entry = entry or {}

        rows.append({
            "contact_type": ct,
            "contact_label": TYPE_LABELS.get(ct, ct),
            "events": entry.get("count", 0),
            "total_sec": entry.get("total_sec", 0.0),
            "mean_duration_sec": entry.get("mean_sec", 0.0),
            "pct_of_session": entry.get("pct_of_session", 0.0),
        })

    # Total row across contact types only (NC excluded — it is the absence of contact).
    rows.append({
        "contact_type": "TOTAL",
        "contact_label": "All contact types",
        "events": sum(r["events"] for r in rows),
        "total_sec": round(sum(float(r["total_sec"] or 0) for r in rows), 2),
        "mean_duration_sec": "",
        "pct_of_session": round(sum(float(r["pct_of_session"] or 0) for r in rows), 2),
    })

    return pd.DataFrame(rows)


def _truncate(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Guard against exceeding the Excel row limit on very long sessions."""
    if len(df) > MAX_SHEET_ROWS:
        logger.warning(
            "Sheet %s has %d rows, truncating to %d (Excel limit)",
            sheet, len(df), MAX_SHEET_ROWS,
        )
        return df.iloc[:MAX_SHEET_ROWS]
    return df


# ── main entry point ───────────────────────────────────────────────────────

def build_excel_report(
    contacts_dir: Path,
    replace_sources: bool = True,
) -> Optional[Path]:
    """Consolidate a contacts directory into a single results.xlsx.

    Args:
        contacts_dir: Directory holding the contact analysis outputs.
        replace_sources: Delete the source CSV/JSON files once the workbook has
            been written successfully. Set False to keep them alongside.

    Returns:
        Path to the workbook, or None if it could not be built.
    """
    contacts_dir = Path(contacts_dir)
    reports_dir = contacts_dir / "reports"

    if not contacts_dir.exists():
        logger.warning("No contacts directory at %s — skipping Excel report", contacts_dir)
        return None

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        logger.warning(
            "openpyxl is not installed — skipping Excel report. "
            "Install it with: pip install openpyxl"
        )
        return None

    # ── Load every source table ──
    per_frame = _read_csv(contacts_dir / "contacts_real_per_frame.csv")
    if per_frame is None:
        # The cleaned table is the raw table plus three columns; fall back to raw.
        per_frame = _read_csv(contacts_dir / "contacts_per_frame.csv")

    events = _read_csv(contacts_dir / "contacts_real_events.csv")
    bouts = _read_csv(contacts_dir / "contact_bouts.csv")
    dynamics = _read_csv(contacts_dir / "dynamics_no_contact.csv")   # v2 only
    by_type = _read_csv(reports_dir / "comparison_by_type.csv")
    global_cmp = _read_csv(reports_dir / "comparison_global.csv")

    summary = _read_json(contacts_dir / "session_summary.json")
    summary_real = _read_json(contacts_dir / "session_summary_real.json")
    individual = _individual_sheet(_read_json(contacts_dir / "individual_summary.json"))

    # A previous consolidation removed session_summary.json, so recover the
    # run-provenance rows from the workbook it wrote.
    carried: Optional[pd.DataFrame] = None
    xlsx_existing = contacts_dir / WORKBOOK_NAME
    if not summary and xlsx_existing.exists():
        try:
            carried = pd.read_excel(xlsx_existing, sheet_name="Parameters")
        except Exception as e:
            logger.warning("Could not read existing Parameters sheet: %s", e)

    if per_frame is None and events is None:
        logger.warning("No contact tables found in %s — skipping Excel report", contacts_dir)
        return None

    types = _resolve_types(per_frame, events)
    logger.info("Contact taxonomy: %s (%d types)", ", ".join(types), len(types))

    sheets: List[tuple] = [
        ("Summary", _summary_sheet(summary, summary_real, events, types)),
        ("Events", events),
        ("PerFrame", per_frame),
        ("Bouts", bouts),
        ("Dynamics", dynamics),
        ("Individual", individual),
        ("ByType", by_type),
        ("Global", global_cmp),
        ("Parameters", _params_sheet(summary, summary_real, carried)),
    ]

    xlsx_path = contacts_dir / WORKBOOK_NAME
    written: List[str] = []

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, df in sheets:
                if df is None:
                    continue
                df = _truncate(df, name)
                df.to_excel(writer, sheet_name=name, index=False)
                _autosize(writer.sheets[name], df)
                written.append(f"{name} ({len(df)})")
    except Exception as e:
        logger.warning("Could not write %s: %s", WORKBOOK_NAME, e)
        return None

    logger.info("Excel report: %s — sheets: %s", xlsx_path, ", ".join(written))

    if replace_sources:
        _remove_sources(contacts_dir, reports_dir)

    return xlsx_path


def _autosize(worksheet, df: pd.DataFrame, max_width: int = 42) -> None:
    """Set a readable column width from the header and first rows."""
    try:
        from openpyxl.utils import get_column_letter
    except ImportError:
        return

    sample = df.head(200)
    for idx, col in enumerate(df.columns, start=1):
        try:
            widest = max(
                [len(str(col))] + [len(str(v)) for v in sample[col].tolist()]
            )
        except Exception:
            widest = len(str(col))
        worksheet.column_dimensions[get_column_letter(idx)].width = min(
            max(10, widest + 2), max_width
        )
    worksheet.freeze_panes = "A2"


def _remove_sources(contacts_dir: Path, reports_dir: Path) -> None:
    """Delete the source files now consolidated into the workbook."""
    removed = 0
    for name in SOURCE_FILES:
        path = contacts_dir / name
        if path.exists():
            try:
                path.unlink()
                removed += 1
            except OSError as e:
                logger.warning("Could not remove %s: %s", name, e)

    for name in SOURCE_FILES_IN_REPORTS:
        path = reports_dir / name
        if path.exists():
            try:
                path.unlink()
                removed += 1
            except OSError as e:
                logger.warning("Could not remove %s: %s", name, e)

    logger.info("Consolidated into %s — removed %d source files", WORKBOOK_NAME, removed)
