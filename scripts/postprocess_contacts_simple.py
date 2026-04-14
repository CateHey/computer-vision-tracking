#!/usr/bin/env python3
"""
Post-process raw contact labels into clean behavioral events.

Applies 3 temporal filtering rules (majority-vote smoothing, gap bridging,
minimum bout duration) to contacts_per_frame.csv and produces a clean
event table with human-readable timestamps for video validation.

Usage:
    python scripts/postprocess_contacts_simple.py path/to/contacts/
    python scripts/postprocess_contacts_simple.py path/to/contacts/ --fps 30 --make_reports
    python scripts/postprocess_contacts_simple.py path/to/contacts/ \\
        --config configs/contacts_postprocess_simple.yaml smoothing.window=7
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

CONTACT_TYPES = ["N2N", "N2AG", "N2B", "T2T", "FOL", "SBS"]
ALL_EVENT_TYPES = CONTACT_TYPES + ["NC"]
PRIORITY_ORDER = {ct: i for i, ct in enumerate(CONTACT_TYPES)}

CT_COLORS = {
    "N2N": "#1f77b4",
    "N2AG": "#ff7f0e",
    "N2B": "#2ca02c",
    "T2T": "#8c564b",
    "FOL": "#d62728",
    "SBS": "#9467bd",
    "NC": "#cccccc",
}

TYPE_LABELS = {
    "N2N": "Nose-to-nose",
    "N2AG": "Nose-to-anogenital",
    "N2B": "Nose-to-body",
    "T2T": "Tail-to-tail",
    "FOL": "Following",
    "SBS": "Side-by-side",
    "NC": "No contact",
}

REQUIRED_COLUMNS = {"frame_idx", "time_sec", "zone", "contact_type"}

# Default config (used when no --config provided)
DEFAULT_CONFIG = {
    "smoothing": {"window": 5},
    "gap_bridging": {"max_gap": 3},
    "min_bout": {"duration_sec": 0.3},
    "fps_fallback": 30.0,
}


# ── Time formatting helpers ────────────────────────────────────────────────

def format_time(sec: float) -> str:
    """Convert seconds to human-readable MM:SS.d or H:MM:SS.d."""
    if sec < 0:
        return "0:00.0"
    hours = int(sec // 3600)
    remainder = sec - hours * 3600
    minutes = int(remainder // 60)
    seconds = remainder - minutes * 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:04.1f}"
    return f"{minutes}:{seconds:04.1f}"


def format_duration(sec: float) -> str:
    """Convert duration to human-readable string like '1.2s'."""
    return f"{sec:.1f}s"


# ── Config loading ─────────────────────────────────────────────────────────

def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a nested dict value using dotted key notation."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _parse_value(s: str) -> Any:
    """Parse a CLI override value to its natural type."""
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s.lower() in ("null", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load_config(config_path: Optional[str], overrides: List[str]) -> dict:
    """Load YAML config or use defaults, then apply CLI overrides."""
    if config_path:
        try:
            from src.common.config_loader import load_config as _load
            return _load(config_path, overrides)
        except Exception as e:
            logger.warning("Could not load config via config_loader: %s", e)
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
    else:
        import copy
        config = copy.deepcopy(DEFAULT_CONFIG)

    # Apply CLI overrides
    for override in overrides or []:
        if "=" not in override:
            logger.warning("Ignoring invalid override (no '='): %s", override)
            continue
        key, value = override.split("=", 1)
        _set_nested(config, key.strip(), _parse_value(value.strip()))

    return config


# ── FPS resolution ─────────────────────────────────────────────────────────

def resolve_fps(
    args: argparse.Namespace, input_dir: Path, config: dict
) -> Tuple[float, str]:
    """Resolve FPS using a 4-level priority cascade."""
    # Priority 1: --fps CLI argument
    if args.fps is not None:
        return args.fps, "cli_argument"

    # Priority 2: session_summary.json
    summary_path = input_dir / "session_summary.json"
    if summary_path.exists():
        try:
            with summary_path.open() as f:
                summary = json.load(f)
            fps = summary.get("metadata", {}).get("fps")
            if fps is not None and float(fps) > 0:
                return float(fps), f"session_summary.json"
        except Exception as e:
            logger.warning("Could not read FPS from session_summary.json: %s", e)

    # Priority 3: --video_path → OpenCV
    if args.video_path is not None:
        try:
            from src.common.io_video import open_video_reader, get_video_properties
            cap = open_video_reader(args.video_path)
            props = get_video_properties(cap)
            cap.release()
            if props["fps"] > 0:
                return props["fps"], f"video_file ({args.video_path})"
        except Exception as e:
            logger.warning("Could not read FPS from video: %s", e)

    # Priority 4: config fallback with warning
    fallback = config.get("fps_fallback", 30.0)
    logger.warning(
        "No FPS source found. Using fallback %.1f fps. "
        "Use --fps or --video_path for accuracy.",
        fallback,
    )
    return fallback, "config_fallback"


# ── CSV loading & validation ───────────────────────────────────────────────

def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Read contacts_per_frame.csv and validate required columns."""
    df = pd.read_csv(csv_path)

    if df.empty:
        logger.warning("CSV is empty: %s", csv_path)
        return df

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing)}. "
            f"Expected at least: {sorted(REQUIRED_COLUMNS)}"
        )

    # Normalize contact_type: NaN → ""
    df["contact_type"] = df["contact_type"].fillna("").astype(str)
    # Normalize zone: NaN → "independent"
    df["zone"] = df["zone"].fillna("independent").astype(str)

    # Warn about unexpected contact_type values
    raw_types = set(df["contact_type"].unique()) - {""}
    unexpected = raw_types - set(CONTACT_TYPES)
    if unexpected:
        logger.warning("Unexpected contact_type values: %s", unexpected)

    return df


# ── Rule 1: Majority-vote smoothing ───────────────────────────────────────

def apply_majority_vote(types: np.ndarray, window: int) -> np.ndarray:
    """Slide a window over contact_type values. Each frame adopts the most
    frequent label in its window. Empty string participates in voting.

    Tie-break: keep current value; secondary: prefer non-empty.
    """
    if window < 3:
        return types.copy()

    # Ensure window is odd
    if window % 2 == 0:
        window += 1
        logger.warning("Smoothing window adjusted to %d (must be odd)", window)

    half = window // 2
    n = len(types)
    smoothed = types.copy()

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_vals = types[start:end]

        counts = Counter(window_vals)
        max_count = counts.most_common(1)[0][1]

        # All labels tied at max_count
        tied = [label for label, c in counts.items() if c == max_count]

        if len(tied) == 1:
            smoothed[i] = tied[0]
        elif types[i] in tied:
            # Keep current value if it's among the winners
            smoothed[i] = types[i]
        else:
            # Prefer non-empty labels, then highest priority contact type
            non_empty = [t for t in tied if t != ""]
            if non_empty:
                # Pick by priority order
                smoothed[i] = min(non_empty, key=lambda t: PRIORITY_ORDER.get(t, 99))
            else:
                smoothed[i] = ""

    return smoothed


# ── Rule 2: Gap bridging ──────────────────────────────────────────────────

def apply_gap_bridging(types: np.ndarray, max_gap: int) -> np.ndarray:
    """Bridge short gaps of empty frames between runs of the same type."""
    if max_gap < 1:
        return types.copy()

    result = types.copy()
    n = len(result)
    i = 0

    while i < n:
        if result[i] != "":
            current_type = result[i]
            # Find end of this contact run
            run_end = i
            while run_end < n and result[run_end] == current_type:
                run_end += 1

            # Count the gap (empty frames after this run)
            gap_start = run_end
            gap_end = gap_start
            while gap_end < n and result[gap_end] == "":
                gap_end += 1

            gap_length = gap_end - gap_start

            # If next non-empty run is same type and gap is small, fill it
            if 0 < gap_length <= max_gap and gap_end < n:
                if result[gap_end] == current_type:
                    for j in range(gap_start, gap_end):
                        result[j] = current_type

            i = run_end
        else:
            i += 1

    return result


# ── Rule 3: Minimum bout duration ─────────────────────────────────────────

def apply_min_bout_filter(types: np.ndarray, min_frames: int) -> np.ndarray:
    """Remove contiguous runs shorter than min_frames."""
    if min_frames < 2:
        return types.copy()

    result = types.copy()
    n = len(result)
    i = 0

    while i < n:
        if result[i] != "":
            current_type = result[i]
            run_start = i
            while i < n and result[i] == current_type:
                i += 1
            run_length = i - run_start

            if run_length < min_frames:
                for j in range(run_start, i):
                    result[j] = ""
        else:
            i += 1

    return result


# ── Event extraction ───────────────────────────────────────────────────────

def _safe_mean(series: pd.Series) -> Optional[float]:
    """Compute mean of a series, robust to missing/empty values."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) > 0:
        return round(float(numeric.mean()), 2)
    return None


def extract_events(
    df: pd.DataFrame, real_types: np.ndarray, fps: float
) -> pd.DataFrame:
    """Convert cleaned per-frame labels into an event table."""
    events = []
    event_id = 0
    n = len(real_types)
    i = 0

    while i < n:
        if real_types[i] != "":
            ct = real_types[i]
            start_idx = i
            while i < n and real_types[i] == ct:
                i += 1
            end_idx = i - 1  # inclusive

            event_slice = df.iloc[start_idx:i]

            start_frame = int(event_slice.iloc[0]["frame_idx"])
            end_frame = int(event_slice.iloc[-1]["frame_idx"])
            start_sec = start_frame / fps
            end_sec = (end_frame + 1) / fps
            dur_sec = end_sec - start_sec

            # Investigator: majority vote
            inv_slot = None
            if "investigator_slot" in df.columns:
                inv_vals = pd.to_numeric(
                    event_slice["investigator_slot"], errors="coerce"
                ).dropna()
                if len(inv_vals) > 0:
                    inv_slot = int(Counter(inv_vals.astype(int)).most_common(1)[0][0])

            events.append({
                "event_id": event_id,
                "start_time_sec": round(start_sec, 4),
                "end_time_sec": round(end_sec, 4),
                "duration_sec": round(dur_sec, 4),
                "start_time": format_time(start_sec),
                "end_time": format_time(end_sec),
                "duration": format_duration(dur_sec),
                "contact_type": ct,
                "contact_label": TYPE_LABELS.get(ct, ct),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": i - start_idx,
                "investigator_slot": inv_slot if inv_slot is not None else "",
                "mean_nose_nose_dist_px": _safe_mean(event_slice.get("nose_nose_dist_px", pd.Series())),
                "mean_centroid_dist_px": _safe_mean(event_slice.get("centroid_dist_px", pd.Series())),
                "mean_mask_iou": _safe_mean(event_slice.get("mask_iou", pd.Series())),
            })
            event_id += 1
        else:
            i += 1

    return pd.DataFrame(events)


def _count_raw_bouts(types: np.ndarray) -> Dict[str, int]:
    """Count contiguous runs per contact type in a label array."""
    counts = {ct: 0 for ct in CONTACT_TYPES}
    n = len(types)
    i = 0
    while i < n:
        if types[i] != "" and types[i] in counts:
            ct = types[i]
            counts[ct] += 1
            while i < n and types[i] == ct:
                i += 1
        else:
            i += 1
    return counts


def _count_raw_duration_frames(types: np.ndarray) -> Dict[str, int]:
    """Count total frames per contact type."""
    counts = {ct: 0 for ct in CONTACT_TYPES}
    for t in types:
        if t in counts:
            counts[t] += 1
    return counts


# ── Output writers ─────────────────────────────────────────────────────────

def write_real_per_frame(
    df: pd.DataFrame,
    real_types: np.ndarray,
    output_dir: Path,
) -> Path:
    """Write contacts_real_per_frame.csv with appended columns."""
    out = df.copy()

    # real_type
    out["real_type"] = real_types

    # real_zone: "contact" if real_type is a contact type, else original zone
    out["real_zone"] = out.apply(
        lambda row: "contact" if row["real_type"] not in ("", "NC") else row["zone"],
        axis=1,
    )

    # real_event_id: assign from contiguous runs (-1 = no event)
    # Includes NC events so IDs match contacts_real_events.csv
    event_ids = np.full(len(real_types), -1, dtype=int)
    eid = 0
    i = 0
    n = len(real_types)
    while i < n:
        if real_types[i] != "":
            ct = real_types[i]
            while i < n and real_types[i] == ct:
                event_ids[i] = eid
                i += 1
            eid += 1
        else:
            i += 1
    out["real_event_id"] = event_ids

    path = output_dir / "contacts_real_per_frame.csv"
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(out))
    return path


def write_real_events(events_df: pd.DataFrame, output_dir: Path) -> Path:
    """Write contacts_real_events.csv."""
    path = output_dir / "contacts_real_events.csv"
    events_df.to_csv(path, index=False)
    logger.info("Wrote %s (%d events)", path, len(events_df))
    return path


def write_event_log(
    events_df: pd.DataFrame,
    raw_types: np.ndarray,
    real_types: np.ndarray,
    fps: float,
    total_frames: int,
    output_dir: Path,
) -> Path:
    """Write event_log.txt — human-readable event log for validation."""
    total_sec = total_frames / fps
    raw_bouts = _count_raw_bouts(raw_types)
    raw_bouts_total = sum(raw_bouts.values())
    # Exclude NC events from comparison (raw data has no NC)
    real_contact_events = len(events_df[events_df["contact_type"] != "NC"]) if len(events_df) > 0 else 0
    real_events_total = len(events_df)

    real_contact_frames = int(np.sum((real_types != "") & (real_types != "NC")))
    raw_contact_frames = int(np.sum(raw_types != ""))
    real_contact_sec = real_contact_frames / fps

    lines = []
    lines.append("=" * 60)
    lines.append("  Contact Events (cleaned)")
    lines.append("=" * 60)
    lines.append(
        f"Video duration: {format_time(total_sec)} "
        f"({total_frames} frames @ {fps:.0f}fps)"
    )
    contact_pct = real_contact_sec / total_sec * 100 if total_sec > 0 else 0.0
    lines.append(
        f"Total events: {real_events_total} | "
        f"Total contact time: {format_duration(real_contact_sec)} "
        f"({contact_pct:.1f}% of session)"
    )
    lines.append("")

    # Contact type legend
    lines.append("Contact types:")
    for ct in CONTACT_TYPES:
        lines.append(f"  {ct:<4} = {TYPE_LABELS[ct]}")
    lines.append("")

    # Event table (sorted chronologically — first to last contact)
    if len(events_df) > 0:
        # Header
        lines.append(
            f"{'#':>3}  {'Start(s)':>9}  {'End(s)':>9}  {'Dur':>6}  "
            f"{'Type':<5}  {'Contact':<22}  {'Investigator'}"
        )
        lines.append("-" * 80)

        for _, row in events_df.iterrows():
            eid = row["event_id"]
            ct = row["contact_type"]
            label = TYPE_LABELS.get(ct, ct)
            start_sec = row["start_time_sec"]
            end_sec = row["end_time_sec"]
            dur = row["duration"]
            inv = row.get("investigator_slot", "")

            if inv != "" and pd.notna(inv):
                inv_int = int(inv)
                if ct == "FOL":
                    inv_str = f"Rat {inv_int} follows"
                else:
                    inv_str = f"Rat {inv_int}"
            else:
                inv_str = "--"

            lines.append(
                f"{eid:>3}  {start_sec:>9.2f}  {end_sec:>9.2f}  {dur:>6}  "
                f"{ct:<5}  {label:<22}  {inv_str}"
            )
    else:
        lines.append("  (no contact events detected)")

    # Summary by type
    lines.append("")
    lines.append("--- Summary by type ---")
    for ct in ALL_EVENT_TYPES:
        ct_events = events_df[events_df["contact_type"] == ct] if len(events_df) > 0 else pd.DataFrame()
        n_events = len(ct_events)
        label = TYPE_LABELS.get(ct, ct)
        if n_events == 0:
            lines.append(f"  {ct:<4} ({label}): {n_events:>2} events")
            continue

        total_dur = ct_events["duration_sec"].sum()
        mean_dur = total_dur / n_events
        pct = total_dur / total_sec * 100 if total_sec > 0 else 0

        line = (
            f"  {ct:<4} ({label}): {n_events:>2} events, "
            f"{format_duration(total_dur):>6} total ({pct:.1f}%), "
            f"mean {format_duration(mean_dur)}"
        )

        # Investigator breakdown for asymmetric types
        if ct in ("N2AG", "N2B", "FOL") and "investigator_slot" in ct_events.columns:
            inv_vals = pd.to_numeric(
                ct_events["investigator_slot"], errors="coerce"
            ).dropna()
            if len(inv_vals) > 0:
                inv_counts = Counter(inv_vals.astype(int))
                parts = [f"Rat {k}: {v}" for k, v in sorted(inv_counts.items())]
                line += ", " + " | ".join(parts)

        lines.append(line)

    # Filtering impact (compare contact events only — NC is not in raw data)
    lines.append("")
    lines.append("--- Filtering impact ---")
    bouts_removed = raw_bouts_total - real_contact_events
    lines.append(
        f"Raw bouts: {raw_bouts_total} -> Real contact events: {real_contact_events} "
        f"({bouts_removed} removed"
        f"{f', {bouts_removed / raw_bouts_total * 100:.1f}% noise' if raw_bouts_total > 0 else ''})"
    )
    raw_sec = raw_contact_frames / fps
    lines.append(
        f"Raw contact time: {format_duration(raw_sec)} -> "
        f"Real: {format_duration(real_contact_sec)} "
        f"({format_duration(raw_sec - real_contact_sec)} removed)"
    )

    # Validation guidelines
    lines.append("")
    lines.append("--- Validation guidelines ---")
    lines.append("  N2N  (Nose-to-nose):       Both noses within contact zone (<0.3 body lengths)")
    lines.append("  N2AG (Nose-to-anogenital):  Nose near tail base of the other rat")
    lines.append("  N2B  (Nose-to-body):        Nose near body — most common, check for false positives")
    lines.append("  T2T  (Tail-to-tail):        Both tail bases close together (rear-to-rear)")
    lines.append("  FOL  (Following):           Sustained following (speed + alignment + min frames)")
    lines.append("  SBS  (Side-by-side):        Parallel movement with mask overlap")
    lines.append("")
    lines.append("  False positive indicators:")
    lines.append("    - Events near the minimum duration (0.3s) may be noise")
    lines.append("    - N2B during proximity zone: may be incidental body proximity")
    lines.append("    - Merged-state frames are excluded from classification (identity ambiguous)")
    lines.append("    - High flicker rate in raw data suggests noisy detections")

    path = output_dir / "event_log.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def write_real_summary(
    df: pd.DataFrame,
    raw_types: np.ndarray,
    real_types: np.ndarray,
    events_df: pd.DataFrame,
    fps: float,
    fps_source: str,
    config: dict,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Write session_summary_real.json."""
    total_frames = len(df)
    total_sec = total_frames / fps
    min_bout_sec = config.get("min_bout", {}).get("duration_sec", 0.3)
    min_bout_frames = max(1, round(min_bout_sec * fps))

    raw_bouts = _count_raw_bouts(raw_types)
    raw_dur_frames = _count_raw_duration_frames(raw_types)
    raw_contact_frames = int(np.sum(raw_types != ""))
    real_contact_frames = int(np.sum((real_types != "") & (real_types != "NC")))

    # Per-type real summary
    events_by_type = {}
    for ct in ALL_EVENT_TYPES:
        ct_events = events_df[events_df["contact_type"] == ct] if len(events_df) > 0 else pd.DataFrame()
        n = len(ct_events)
        total_dur = float(ct_events["duration_sec"].sum()) if n > 0 else 0.0
        events_by_type[ct] = {
            "count": n,
            "total_sec": round(total_dur, 2),
            "mean_sec": round(total_dur / n, 2) if n > 0 else 0.0,
            "pct_of_session": round(total_dur / total_sec * 100, 2) if total_sec > 0 else 0.0,
        }

    summary = {
        "metadata": {
            "source_csv": str(input_dir / "contacts_per_frame.csv"),
            "postprocess_date": datetime.now().isoformat(timespec="seconds"),
            "fps": fps,
            "fps_source": fps_source,
            "total_frames": total_frames,
            "total_duration_sec": round(total_sec, 2),
            "total_duration_human": format_time(total_sec),
        },
        "parameters": {
            "smoothing_window": config.get("smoothing", {}).get("window", 5),
            "gap_bridging_max_gap": config.get("gap_bridging", {}).get("max_gap", 3),
            "min_bout_duration_sec": min_bout_sec,
            "min_bout_duration_frames": min_bout_frames,
        },
        "raw_summary": {
            "contact_frames": raw_contact_frames,
            "contact_pct": round(raw_contact_frames / total_frames * 100, 2) if total_frames > 0 else 0.0,
            "bouts_by_type": raw_bouts,
            "duration_frames_by_type": raw_dur_frames,
        },
        "real_summary": {
            "contact_frames": real_contact_frames,
            "contact_pct": round(real_contact_frames / total_frames * 100, 2) if total_frames > 0 else 0.0,
            "events_by_type": events_by_type,
            "total_contact_sec": round(real_contact_frames / fps, 2),
            "total_events": len(events_df),
        },
        "filtering_impact": {
            "frames_removed": raw_contact_frames - real_contact_frames,
            "flicker_rate_pct": round(
                (raw_contact_frames - real_contact_frames) / raw_contact_frames * 100, 2
            ) if raw_contact_frames > 0 else 0.0,
            # Compare contact events only (NC is not in raw data)
            "bouts_removed": sum(raw_bouts.values()) - (
                len(events_df[events_df["contact_type"] != "NC"]) if len(events_df) > 0 else 0
            ),
            "bout_removal_rate_pct": round(
                (sum(raw_bouts.values()) - (
                    len(events_df[events_df["contact_type"] != "NC"]) if len(events_df) > 0 else 0
                )) / sum(raw_bouts.values()) * 100, 2
            ) if sum(raw_bouts.values()) > 0 else 0.0,
        },
    }

    path = output_dir / "session_summary_real.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", path)
    return path


# ── Report generation ──────────────────────────────────────────────────────

def _format_time_ticks(ax, axis: str = "x") -> None:
    """Convert seconds tick labels to MM:SS format."""
    from matplotlib.ticker import FuncFormatter

    def _fmt(val, pos):
        return format_time(val)

    if axis == "x":
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt))


def generate_reports(
    df: pd.DataFrame,
    raw_types: np.ndarray,
    real_types: np.ndarray,
    events_df: pd.DataFrame,
    fps: float,
    reports_dir: Path,
) -> None:
    """Generate 4 PNG charts + 2 CSV comparison tables."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available — skipping reports")
        return

    reports_dir.mkdir(parents=True, exist_ok=True)
    total_frames = len(df)
    total_sec = total_frames / fps

    raw_bouts = _count_raw_bouts(raw_types)
    raw_dur_frames = _count_raw_duration_frames(raw_types)

    # ── Chart 1: Timeline comparison ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    fig.suptitle("Contact Timeline: Raw vs Cleaned", fontsize=13)

    for ax, types_arr, label, type_list in [
        (axes[0], raw_types, "Raw contacts", CONTACT_TYPES),
        (axes[1], real_types, "Real contacts (cleaned)", ALL_EVENT_TYPES),
    ]:
        for ct in type_list:
            # Find runs of this type
            in_run = False
            run_start = 0
            for j in range(len(types_arr)):
                if types_arr[j] == ct and not in_run:
                    in_run = True
                    run_start = j
                elif types_arr[j] != ct and in_run:
                    in_run = False
                    start_sec = df.iloc[run_start]["frame_idx"] / fps
                    end_sec = df.iloc[j - 1]["frame_idx"] / fps
                    ax.barh(0, end_sec - start_sec, left=start_sec, height=0.6,
                            color=CT_COLORS[ct], linewidth=0)
            if in_run:
                start_sec = df.iloc[run_start]["frame_idx"] / fps
                end_sec = df.iloc[-1]["frame_idx"] / fps
                ax.barh(0, end_sec - start_sec, left=start_sec, height=0.6,
                        color=CT_COLORS[ct], linewidth=0)

        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=10)
        ax.set_xlim(0, total_sec)
        _format_time_ticks(ax, "x")

    legend_patches = [
        mpatches.Patch(color=CT_COLORS[ct], label=f"{ct} ({TYPE_LABELS[ct]})")
        for ct in ALL_EVENT_TYPES
    ]
    axes[1].legend(handles=legend_patches, loc="upper center",
                   bbox_to_anchor=(0.5, -0.4), ncol=4, fontsize=8)
    axes[1].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(reports_dir / "timeline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote timeline_comparison.png")

    # ── Chart 2: Duration by type ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CONTACT_TYPES))
    width = 0.35

    raw_durations = [raw_dur_frames.get(ct, 0) / fps for ct in CONTACT_TYPES]
    real_durations = []
    for ct in CONTACT_TYPES:
        ct_ev = events_df[events_df["contact_type"] == ct] if len(events_df) > 0 else pd.DataFrame()
        real_durations.append(float(ct_ev["duration_sec"].sum()) if len(ct_ev) > 0 else 0.0)

    bars1 = ax.bar(x - width / 2, raw_durations, width, label="Raw", color="#aaaaaa", edgecolor="black")
    bars2 = ax.bar(x + width / 2, real_durations, width,
                   color=[CT_COLORS[ct] for ct in CONTACT_TYPES], edgecolor="black", label="Real")

    ax.bar_label(bars1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ct}\n{TYPE_LABELS[ct]}" for ct in CONTACT_TYPES], fontsize=8)
    ax.set_ylabel("Total duration (seconds)")
    ax.set_title("Total Contact Duration by Type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "duration_by_type.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote duration_by_type.png")

    # ── Chart 3: Events by type ──
    fig, ax = plt.subplots(figsize=(10, 5))

    raw_counts = [raw_bouts.get(ct, 0) for ct in CONTACT_TYPES]
    real_counts = []
    for ct in CONTACT_TYPES:
        ct_ev = events_df[events_df["contact_type"] == ct] if len(events_df) > 0 else pd.DataFrame()
        real_counts.append(len(ct_ev))

    bars1 = ax.bar(x - width / 2, raw_counts, width, label="Raw bouts", color="#aaaaaa", edgecolor="black")
    bars2 = ax.bar(x + width / 2, real_counts, width,
                   color=[CT_COLORS[ct] for ct in CONTACT_TYPES], edgecolor="black", label="Real events")

    ax.bar_label(bars1, fontsize=9, padding=2)
    ax.bar_label(bars2, fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ct}\n{TYPE_LABELS[ct]}" for ct in CONTACT_TYPES], fontsize=8)
    ax.set_ylabel("Number of events")
    ax.set_title("Event Count by Type: Raw vs Cleaned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "events_by_type.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote events_by_type.png")

    # ── Chart 4: Event duration distribution ──
    n_types = len(ALL_EVENT_TYPES)
    n_cols = 3
    n_rows = math.ceil((n_types + 1) / n_cols)  # +1 for "All types"
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)
    fig.suptitle("Real Event Duration Distribution", fontsize=13)

    for idx, ct in enumerate(ALL_EVENT_TYPES):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ct_ev = events_df[events_df["contact_type"] == ct] if len(events_df) > 0 else pd.DataFrame()
        durations = ct_ev["duration_sec"].values if len(ct_ev) > 0 else []

        if len(durations) > 0:
            n_bins = min(20, max(5, len(durations)))
            ax.hist(durations, bins=n_bins, color=CT_COLORS[ct], edgecolor="black", alpha=0.8)
        ax.set_title(f"{ct} - {TYPE_LABELS[ct]} ({len(durations)} events)", fontsize=10)
        ax.set_xlabel("Duration (s)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)

    # "All types" in next available cell
    all_row, all_col = divmod(n_types, n_cols)
    ax = axes[all_row][all_col]
    all_durations = events_df["duration_sec"].values if len(events_df) > 0 else []
    if len(all_durations) > 0:
        n_bins = min(20, max(5, len(all_durations)))
        ax.hist(all_durations, bins=n_bins, color="#555555", edgecolor="black", alpha=0.8)
    ax.set_title(f"All types ({len(all_durations)} events)", fontsize=10)
    ax.set_xlabel("Duration (s)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)

    # Hide unused subplots
    for idx in range(n_types + 1, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(reports_dir / "event_duration_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote event_duration_distribution.png")

    # ── Table: comparison_by_type.csv ──
    rows = []
    for ct in ALL_EVENT_TYPES:
        ct_ev = events_df[events_df["contact_type"] == ct] if len(events_df) > 0 else pd.DataFrame()
        n_real = len(ct_ev)
        n_raw = raw_bouts.get(ct, 0)
        raw_sec = raw_dur_frames.get(ct, 0) / fps
        real_sec = float(ct_ev["duration_sec"].sum()) if n_real > 0 else 0.0
        change_pct = ((real_sec - raw_sec) / raw_sec * 100) if raw_sec > 0 else 0.0

        rows.append({
            "contact_type": ct,
            "contact_label": TYPE_LABELS.get(ct, ct),
            "raw_bouts": n_raw,
            "real_events": n_real,
            "removed": n_raw - n_real,
            "raw_total_sec": round(raw_sec, 2),
            "real_total_sec": round(real_sec, 2),
            "change_pct": round(change_pct, 1),
            "mean_duration_sec": round(real_sec / n_real, 2) if n_real > 0 else "",
            "median_duration_sec": round(
                float(median(ct_ev["duration_sec"])), 2
            ) if n_real > 0 else "",
        })

    comp_type_path = reports_dir / "comparison_by_type.csv"
    pd.DataFrame(rows).to_csv(comp_type_path, index=False)
    logger.info("Wrote comparison_by_type.csv")

    # ── Table: comparison_global.csv ──
    raw_contact_frames = int(np.sum(raw_types != ""))
    real_contact_frames = int(np.sum((real_types != "") & (real_types != "NC")))

    # Flicker rate: type transitions per 1000 frames
    def _flicker_rate(arr):
        if len(arr) < 2:
            return 0.0
        transitions = sum(1 for k in range(1, len(arr)) if arr[k] != arr[k - 1])
        return round(transitions / len(arr) * 1000, 1)

    global_data = {
        "metric": [
            "total_frames", "total_duration_sec", "fps",
            "raw_contact_frames", "real_contact_frames",
            "raw_contact_pct", "real_contact_pct",
            "raw_bouts_total", "real_events_total",
            "flicker_rate_raw_per_1000", "flicker_rate_real_per_1000",
        ],
        "value": [
            total_frames, round(total_sec, 2), fps,
            raw_contact_frames, real_contact_frames,
            round(raw_contact_frames / total_frames * 100, 2) if total_frames > 0 else 0,
            round(real_contact_frames / total_frames * 100, 2) if total_frames > 0 else 0,
            sum(raw_bouts.values()), len(events_df),
            _flicker_rate(raw_types), _flicker_rate(real_types),
        ],
    }

    comp_global_path = reports_dir / "comparison_global.csv"
    pd.DataFrame(global_data).to_csv(comp_global_path, index=False)
    logger.info("Wrote comparison_global.csv")


# ── Programmatic API (called from pipelines) ──────────────────────────────

def run_postprocess(
    contacts_dir: Path,
    fps: float,
    make_reports: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """Run bout post-processing on a contacts directory.

    Called automatically at the end of each pipeline when contacts are enabled.
    Can also be called standalone.

    Args:
        contacts_dir: Directory containing contacts_per_frame.csv.
        fps: Video FPS.
        make_reports: Whether to generate PNG charts and CSV tables.
        config_overrides: Override default config values (e.g. {"smoothing": {"window": 7}}).

    Returns:
        Path to the output directory (same as contacts_dir).
    """
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_overrides:
        for key, val in config_overrides.items():
            if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                config[key].update(val)
            else:
                config[key] = val

    csv_path = contacts_dir / "contacts_per_frame.csv"
    if not csv_path.exists():
        logger.warning("Skipping post-processing: contacts_per_frame.csv not found in %s", contacts_dir)
        return contacts_dir

    df = load_and_validate(csv_path)
    if df.empty:
        logger.warning("Skipping post-processing: CSV is empty")
        return contacts_dir

    smooth_window = config.get("smoothing", {}).get("window", 5)
    gap_max = config.get("gap_bridging", {}).get("max_gap", 3)
    min_bout_sec = config.get("min_bout", {}).get("duration_sec", 0.3)
    min_bout_frames = max(1, round(min_bout_sec * fps))

    logger.info(
        "Post-processing contacts: smooth=%d, gap=%d, min_bout=%.3fs (%d frames)",
        smooth_window, gap_max, min_bout_sec, min_bout_frames,
    )

    raw_types = df["contact_type"].fillna("").astype(str).values.copy()

    types = raw_types.copy()
    types = apply_majority_vote(types, smooth_window)
    types = apply_gap_bridging(types, gap_max)
    types = apply_min_bout_filter(types, min_bout_frames)
    real_types = np.where(types == "", "NC", types)

    events_df = extract_events(df, real_types, fps)

    raw_bout_count = sum(_count_raw_bouts(raw_types).values())
    logger.info("Post-processing: %d raw bouts -> %d real events", raw_bout_count, len(events_df))

    write_real_per_frame(df, real_types, contacts_dir)
    write_real_events(events_df, contacts_dir)
    write_event_log(events_df, raw_types, real_types, fps, len(df), contacts_dir)
    write_real_summary(
        df, raw_types, real_types, events_df,
        fps, "pipeline", config, contacts_dir, contacts_dir,
    )

    if make_reports:
        reports_dir = contacts_dir / "reports"
        generate_reports(df, raw_types, real_types, events_df, fps, reports_dir)

    return contacts_dir


# ── Main (CLI) ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-process raw contact labels into clean behavioral events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s outputs/runs/2026-03-04_ref/contacts/
              %(prog)s path/to/contacts/ --fps 30 --make_reports
              %(prog)s path/to/contacts/ --config configs/contacts_postprocess_simple.yaml
              %(prog)s path/to/contacts/ smoothing.window=7 min_bout.duration_sec=0.5
        """),
    )
    parser.add_argument(
        "input_dir", type=str,
        help="Directory containing contacts_per_frame.csv",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file (defaults built-in if omitted)",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Video FPS (overrides all other FPS sources)",
    )
    parser.add_argument(
        "--video_path", type=str, default=None,
        help="Video file to auto-detect FPS",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (defaults to input_dir)",
    )
    parser.add_argument(
        "--make_reports", action="store_true",
        help="Generate PNG charts and CSV tables in reports/ subdirectory",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides as key.subkey=value",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = Path(args.input_dir)
    csv_path = input_dir / "contacts_per_frame.csv"
    if not csv_path.exists():
        logger.error("contacts_per_frame.csv not found in %s", input_dir)
        sys.exit(1)

    # Load config
    config = load_config(args.config, args.overrides)

    # Resolve FPS
    fps, fps_source = resolve_fps(args, input_dir, config)
    logger.info("FPS: %.2f (source: %s)", fps, fps_source)

    # Load and validate
    df = load_and_validate(csv_path)
    if df.empty:
        logger.warning("No data in CSV. Nothing to process.")
        return

    logger.info("Loaded %d rows from %s", len(df), csv_path)

    # Extract config parameters
    smooth_window = config.get("smoothing", {}).get("window", 5)
    gap_max = config.get("gap_bridging", {}).get("max_gap", 3)
    min_bout_sec = config.get("min_bout", {}).get("duration_sec", 0.3)
    min_bout_frames = max(1, round(min_bout_sec * fps))

    logger.info(
        "Parameters: smooth=%d, gap=%d, min_bout=%.3fs (%d frames)",
        smooth_window, gap_max, min_bout_sec, min_bout_frames,
    )

    # Get raw types before processing
    raw_types = df["contact_type"].fillna("").astype(str).values.copy()

    # Apply 3 rules in order
    types = raw_types.copy()

    logger.info("Rule 1: Majority-vote smoothing (window=%d)...", smooth_window)
    types = apply_majority_vote(types, smooth_window)

    logger.info("Rule 2: Gap bridging (max_gap=%d frames)...", gap_max)
    types = apply_gap_bridging(types, gap_max)

    logger.info(
        "Rule 3: Minimum bout filter (min=%d frames / %.3fs)...",
        min_bout_frames, min_bout_sec,
    )
    types = apply_min_bout_filter(types, min_bout_frames)

    real_types = np.where(types == "", "NC", types)

    # Extract events
    events_df = extract_events(df, real_types, fps)
    logger.info("Extracted %d clean events from %d raw bouts",
                len(events_df), sum(_count_raw_bouts(raw_types).values()))

    # Write outputs
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_real_per_frame(df, real_types, output_dir)
    write_real_events(events_df, output_dir)
    write_event_log(events_df, raw_types, real_types, fps, len(df), output_dir)
    write_real_summary(
        df, raw_types, real_types, events_df,
        fps, fps_source, config, input_dir, output_dir,
    )

    # Optional reports
    if args.make_reports:
        reports_dir = output_dir / "reports"
        generate_reports(df, raw_types, real_types, events_df, fps, reports_dir)

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
