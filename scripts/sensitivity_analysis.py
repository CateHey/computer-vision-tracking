#!/usr/bin/env python3
"""Offline sensitivity analysis of the contact counts to the labelling and cleaning thresholds.

Reviewers ask how much the reported counts depend on the thresholds. Most of them can be
re-evaluated without a GPU from the per-frame table of a finished run: it holds, per frame
and pair, the five continuous scores (``score_n2n, score_n2ag, score_fol, score_sbs,
score_n2b``), and everything after the scores is a deterministic function of them:

    scores --ScoreMap.argmax_type(activation, rare)--> raw label
           --compute_labels(smoothing, gap bridging, min bout)--> cleaned label
           --extract_events--> events

One parameter is varied at a time (all others stay at their baseline) over the grid in
``configs/sensitivity.yaml``; for each value the events are rebuilt and counted per contact
type. Thresholds fixed at tracking time (Schmitt enter/exit distances, score internals) cannot
be varied here; the markdown report says so.

Usage:
    python scripts/sensitivity_analysis.py outputs/runs/<run>
    python scripts/sensitivity_analysis.py outputs/runs/<run> --out some/dir
    python scripts/sensitivity_analysis.py outputs/runs/<run> min_bout_sec=[0.3,0.5] smoothing_window=7

Outputs (default ``<run>/sensitivity/``):
    sensitivity_long.csv       parameter, value, contact_type, n_events, total_sec, mean_sec, ...
    sensitivity_summary.md     one table per parameter (cells: n_events (total_sec)), notes
    sensitivity_<parameter>.png  n_events vs value, one line per type, baseline dashed
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.postprocess_contacts_simple import (  # noqa: E402
    CT_COLORS,
    DEFAULT_CONFIG as POSTPROCESS_DEFAULTS,
    compute_labels,
    extract_events,
    normalize_labels,
)
from scripts.validate_contacts import _parse_value, find_fps, load_per_frame  # noqa: E402
from src.common.contacts_v2 import ScoreMap  # noqa: E402

logger = logging.getLogger("sensitivity")

# Priority order of ScoreMap.argmax_type == column order of the score matrix below.
PRIORITY: Tuple[str, ...] = ("N2N", "N2AG", "FOL", "SBS", "N2B")
SCORE_COLUMNS: Tuple[str, ...] = tuple(f"score_{t.lower()}" for t in PRIORITY)
RARE_TYPES = ("FOL",)
# Display order of the types in tables and plots (same as the Excel report).
TYPE_ORDER: Tuple[str, ...] = ("N2N", "N2AG", "N2B", "SBS", "FOL")
MARKERS = {"N2N": "o", "N2AG": "s", "N2B": "^", "SBS": "D", "FOL": "v"}

PARAMETERS: Tuple[str, ...] = (
    "activation_threshold",
    "activation_threshold_rare",
    "smoothing_window",
    "gap_bridging_max_gap",
    "min_bout_sec",
)
PARAM_TITLES = {
    "activation_threshold": "Activation threshold (N2N, N2AG, SBS, N2B)",
    "activation_threshold_rare": "Activation threshold for the rare type (FOL)",
    "smoothing_window": "Majority-vote window (frames)",
    "gap_bridging_max_gap": "Gap bridging, maximum gap (frames)",
    "min_bout_sec": "Global minimum event duration (s)",
}
INT_PARAMS = ("smoothing_window", "gap_bridging_max_gap")

DEFAULT_GRID: Dict[str, List[float]] = {
    "activation_threshold": [0.40, 0.45, 0.50, 0.55, 0.60, 0.70],
    "activation_threshold_rare": [0.25, 0.35, 0.45],
    "smoothing_window": [3, 5, 7],
    "gap_bridging_max_gap": [1, 3, 5],
    "min_bout_sec": [0.2, 0.3, 0.4, 0.5, 1.0],
}
# Baselines that are not read from configs/contacts_postprocess_simple.yaml.
BASELINE_ACTIVATION = 0.5
BASELINE_ACTIVATION_RARE = 0.35

LONG_COLUMNS = ["parameter", "value", "contact_type", "n_events", "total_sec", "mean_sec",
                "n_pairs", "n_pair_frames", "is_baseline"]


# --------------------------------------------------------------------------- config
def load_grid(overrides: Sequence[str]) -> Tuple[Dict[str, List[float]], Optional[float]]:
    """Grid from configs/sensitivity.yaml (else the built-in defaults) plus key=value overrides.

    Returns (grid, fps_override). A scalar override such as ``smoothing_window=7`` is a
    one-value grid.
    """
    cfg: Dict[str, Any] = {k: list(v) for k, v in DEFAULT_GRID.items()}
    cfg["fps"] = None
    path = _PROJECT_ROOT / "configs" / "sensitivity.yaml"
    if path.exists():
        import yaml
        cfg.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    for ov in overrides:
        if "=" not in ov:
            logger.warning("Ignoring invalid override (no '='): %s", ov)
            continue
        key, val = ov.split("=", 1)
        key = key.strip()
        if key not in cfg:
            raise SystemExit(f"Unknown override key '{key}'. Valid: {sorted(cfg)}")
        cfg[key] = _parse_value(val.strip())
    fps = cfg.pop("fps", None)
    grid: Dict[str, List[float]] = {}
    for param in PARAMETERS:
        raw = cfg.get(param)
        if raw is None:
            continue
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        grid[param] = [int(v) if param in INT_PARAMS else float(v) for v in values]
    return grid, (float(fps) if fps is not None else None)


def load_baseline(postprocess_yaml: Optional[Path] = None) -> Dict[str, Any]:
    """Baseline parameter values: activation 0.5 / 0.35 and the post-processing YAML values."""
    post = {k: dict(v) if isinstance(v, dict) else v for k, v in POSTPROCESS_DEFAULTS.items()}
    path = postprocess_yaml or (_PROJECT_ROOT / "configs" / "contacts_postprocess_simple.yaml")
    if path.exists():
        import yaml
        for key, val in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items():
            if isinstance(val, dict) and isinstance(post.get(key), dict):
                post[key].update(val)
            else:
                post[key] = val
    else:
        logger.warning("%s not found; using the built-in post-processing defaults", path)
    min_bout = post.get("min_bout", {})
    return {
        "activation_threshold": BASELINE_ACTIVATION,
        "activation_threshold_rare": BASELINE_ACTIVATION_RARE,
        "smoothing_window": int(post.get("smoothing", {}).get("window", 5)),
        "gap_bridging_max_gap": int(post.get("gap_bridging", {}).get("max_gap", 3)),
        "min_bout_sec": float(min_bout.get("duration_sec", 0.3)),
        "min_bout_by_type": {str(k).upper(): float(v)
                             for k, v in (min_bout.get("by_type") or {}).items()},
    }


def complete_grid(grid: Dict[str, List[float]], baseline: Dict[str, Any]) -> Dict[str, List[float]]:
    """Sorted unique values per parameter, with the baseline value always present."""
    out: Dict[str, List[float]] = {}
    for param in PARAMETERS:
        if param not in grid:
            continue
        vals = list(grid[param])
        if not any(_same(v, baseline[param]) for v in vals):
            vals.append(baseline[param])
        uniq: List[float] = []
        for v in sorted(vals):
            if not uniq or not _same(uniq[-1], v):
                uniq.append(v)
        out[param] = uniq
    return out


def _same(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- fps
def resolve_fps(run: Path, override: Optional[float]) -> float:
    """fps: explicit override, else the Parameters sheet / session_summary / config_used."""
    if override:
        return float(override)
    for xlsx in (run / "contacts" / "results.xlsx", run / "results.xlsx"):
        if xlsx.exists():
            try:
                par = pd.read_excel(xlsx, sheet_name="Parameters")
                hit = par[(par["parameter"] == "fps") & (par["section"] == "video")]["value"]
                if len(hit) and float(hit.iloc[0]) > 0:
                    return float(hit.iloc[0])
            except Exception:  # missing sheet / columns: fall through
                pass
    return find_fps(run, None)


# --------------------------------------------------------------------------- labels
@dataclass
class FrameTable:
    """The per-frame table reduced to what the re-evaluation needs."""
    scores: np.ndarray            # (n, 5) in PRIORITY order, missing -> 0
    frames: pd.DataFrame          # frame_idx (+ pair_key when present)
    stored_raw: Optional[np.ndarray]  # stored contact_type, normalised ("" = none)

    @property
    def n_rows(self) -> int:
        return len(self.frames)


def prepare_table(pf: pd.DataFrame) -> FrameTable:
    """Extract the score matrix (missing columns / values count as 0) and the keys."""
    if "frame_idx" not in pf.columns:
        raise ValueError("The per-frame table has no frame_idx column")
    present = [c for c in SCORE_COLUMNS if c in pf.columns]
    if not present:
        raise ValueError(f"None of the score columns {SCORE_COLUMNS} is in the per-frame table; "
                         "this run predates the continuous scores (contacts_v2).")
    missing = [c for c in SCORE_COLUMNS if c not in pf.columns]
    if missing:
        logger.warning("Score columns missing from the table, treated as 0: %s", missing)
    scores = np.zeros((len(pf), len(SCORE_COLUMNS)))
    for k, col in enumerate(SCORE_COLUMNS):
        if col in pf.columns:
            scores[:, k] = pd.to_numeric(pf[col], errors="coerce").fillna(0.0).to_numpy()
    frames = pd.DataFrame({"frame_idx": pd.to_numeric(pf["frame_idx"], errors="coerce").to_numpy()})
    if "pair_key" in pf.columns:
        frames["pair_key"] = pf["pair_key"].fillna("").astype(str).to_numpy()
    stored = normalize_labels(pf["contact_type"]) if "contact_type" in pf.columns else None
    return FrameTable(scores=scores, frames=frames, stored_raw=stored)


def raw_labels(scores: np.ndarray, activation: float, rare: float) -> np.ndarray:
    """Vectorised ``ScoreMap.argmax_type`` for a (n, 5) matrix in PRIORITY order.

    Same rule: walk the types in priority order and keep the highest score that reaches its
    threshold (``rare`` for FOL, ``activation`` for the others); ties go to the earlier type;
    no type reaching its threshold gives "none". ``reference_labels`` is the slow ScoreMap
    version, checked against this one in the tests and on a sample at run time.
    """
    n = scores.shape[0]
    best = np.zeros(n)
    chosen = np.full(n, -1)
    for k, ct in enumerate(PRIORITY):
        thr = rare if ct in RARE_TYPES else activation
        s = scores[:, k]
        take = (s >= thr) & (s > best)
        best = np.where(take, s, best)
        chosen = np.where(take, k, chosen)
    names = np.array(list(PRIORITY) + ["none"], dtype=object)
    return names[chosen]  # chosen == -1 indexes the last entry, "none"


def reference_labels(scores: np.ndarray, activation: float, rare: float) -> np.ndarray:
    """The same labels, one ``ScoreMap`` per row (slow; reference implementation)."""
    out = np.empty(scores.shape[0], dtype=object)
    for i, row in enumerate(scores):
        sm = ScoreMap(n2n=row[0], n2ag=row[1], fol=row[2], sbs=row[3], n2b=row[4])
        out[i] = sm.argmax_type(activation_threshold=activation, rare_threshold=rare).value
    return out


def check_against_scoremap(table: FrameTable, activation: float, rare: float,
                           sample: int = 2000, seed: int = 0) -> None:
    """Fail loudly if the vectorised labels ever disagree with ScoreMap on a row sample."""
    n = table.n_rows
    idx = np.arange(n) if n <= sample else np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
    fast = raw_labels(table.scores[idx], activation, rare)
    slow = reference_labels(table.scores[idx], activation, rare)
    bad = int((fast != slow).sum())
    if bad:
        raise AssertionError(f"Vectorised labels differ from ScoreMap.argmax_type on {bad}/{len(idx)} rows")


# --------------------------------------------------------------------------- evaluation
def postprocess_config(params: Dict[str, Any]) -> dict:
    """Parameter dict -> the config layout expected by ``compute_labels``."""
    return {
        "smoothing": {"window": int(params["smoothing_window"])},
        "gap_bridging": {"max_gap": int(params["gap_bridging_max_gap"])},
        "min_bout": {"duration_sec": float(params["min_bout_sec"]),
                     "by_type": dict(params.get("min_bout_by_type") or {})},
    }


def evaluate(table: FrameTable, params: Dict[str, Any], fps: float) -> Dict[str, Dict[str, float]]:
    """Rebuild raw labels, cleaned labels and events for one parameter set; count per type."""
    labels = raw_labels(table.scores, params["activation_threshold"], params["activation_threshold_rare"])
    df = table.frames.copy()
    df["contact_type"] = labels
    _, real_types = compute_labels(df, postprocess_config(params), fps)
    events = extract_events(df, real_types, fps)
    return summarise(events, real_types)


def summarise(events: pd.DataFrame, real_types: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per contact type: events, total seconds, mean seconds, pairs with an event, pair-frames."""
    stats: Dict[str, Dict[str, float]] = {}
    for ct in TYPE_ORDER:
        if len(events) and "contact_type" in events.columns:
            sel = events[events["contact_type"] == ct]
        else:
            sel = events.iloc[0:0]
        n = int(len(sel))
        total = float(sel["duration_sec"].sum()) if n else 0.0
        if n and "pair_key" in sel.columns:
            n_pairs = int(sel["pair_key"].nunique())
        else:
            n_pairs = 1 if n else 0
        stats[ct] = {
            "n_events": n,
            "total_sec": round(total, 4),
            "mean_sec": round(total / n, 4) if n else float("nan"),
            "n_pairs": n_pairs,
            "n_pair_frames": int((real_types == ct).sum()),
        }
    return stats


def run_sensitivity(table: FrameTable, fps: float, grid: Dict[str, List[float]],
                    baseline: Dict[str, Any]) -> pd.DataFrame:
    """Vary one parameter at a time over ``grid``; return the long table."""
    cache: Dict[Tuple, Dict[str, Dict[str, float]]] = {}

    def stats_for(params: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        key = tuple(round(float(params[p]), 9) for p in PARAMETERS)
        if key not in cache:
            cache[key] = evaluate(table, params, fps)
        return cache[key]

    rows: List[Dict[str, Any]] = []
    total = sum(len(v) for v in grid.values())
    done = 0
    for param, values in grid.items():
        for value in values:
            params = dict(baseline)
            params[param] = value
            logger.info("[%d/%d] %s = %g", done + 1, total, param, value)
            done += 1
            stats = stats_for(params)
            is_base = _same(value, baseline[param])
            for ct in TYPE_ORDER:
                st = stats[ct]
                rows.append({
                    "parameter": param, "value": value, "contact_type": ct,
                    "n_events": st["n_events"], "total_sec": round(st["total_sec"], 3),
                    "mean_sec": (round(st["mean_sec"], 3) if st["n_events"] else float("nan")),
                    "n_pairs": st["n_pairs"], "n_pair_frames": st["n_pair_frames"],
                    "is_baseline": is_base,
                })
    out = pd.DataFrame(rows, columns=LONG_COLUMNS)
    out["value"] = pd.Series([r["value"] for r in rows], dtype=object)  # keep 3 as 3, not 3.0
    return out


# --------------------------------------------------------------------------- reporting
def _fmt(v: float) -> str:
    return f"{float(v):g}"


def robustness_statement(long_df: pd.DataFrame) -> str:
    """One paragraph: per type, min-max event count over the activation and min-bout grids."""

    def span(param: str, ct: str) -> Optional[Tuple[int, int, int]]:
        sub = long_df[(long_df["parameter"] == param) & (long_df["contact_type"] == ct)]
        if sub.empty:
            return None
        base = sub[sub["is_baseline"].astype(bool)]["n_events"]
        return int(sub["n_events"].min()), int(sub["n_events"].max()), int(base.iloc[0]) if len(base) else -1

    def grid_range(param: str) -> str:
        v = [float(x) for x in long_df[long_df["parameter"] == param]["value"].unique()]
        return f"{_fmt(min(v))}-{_fmt(max(v))}"

    def clause(param: str, label: str) -> Optional[str]:
        if param not in set(long_df["parameter"]):
            return None
        parts = []
        for ct in TYPE_ORDER:
            s = span(param, ct)
            if s is None:
                continue
            lo, hi, base = s
            if param == "activation_threshold" and ct == "FOL":
                parts.append(f"FOL {lo} (unaffected: it uses the rare threshold)")
            else:
                parts.append(f"{ct} {lo}-{hi} (baseline {base})" if lo != hi else f"{ct} {lo} (baseline {base})")
        return f"Over the {label} grid ({grid_range(param)}) the number of events is " + ", ".join(parts)

    clauses = [c for c in (clause("activation_threshold", "activation-threshold"),
                           clause("min_bout_sec", "minimum-duration (s)")) if c]
    text = "Robustness. " + ". ".join(clauses) + "." if clauses else "Robustness. No grid to summarise."
    s = span("activation_threshold_rare", "FOL")
    if s is not None:
        verb = f"ranges {s[0]}-{s[1]}" if s[0] != s[1] else f"stays at {s[0]}"
        text += (f" Following, which uses its own threshold, {verb} events over the "
                 f"activation_threshold_rare grid ({grid_range('activation_threshold_rare')}, baseline {s[2]}).")
    return text


NOT_REEVALUABLE = """\
## Thresholds that cannot be re-evaluated offline

The analysis starts from the five stored `score_*` columns, so anything that shaped those
scores while the tracker ran is fixed in this table. Varying these needs a new tracker run
(GPU, whole pipeline):

- **Schmitt enter/exit distances**: `contact_zone_bl_enter/_exit` (N2N, N2AG), `sbs_contact_bl_enter/_exit`
  (SBS), `follow_radius_bl` / `follow_radius_bl_exit` (FOL). The latches carry state from frame to frame
  and are already folded into the scores.
- **Score internals**: the per-type ramps and weights, the orientation and alignment factors,
  `follow_min_speed_bls`, `follow_alignment_cos`, `sbs_latch_min_dist_score`, `mask_contact_dilate_px`.
- **The tracker's own bout filter** (`bout_max_gap_frames`, `bout_min_frames_*`): it only feeds the
  `Bouts` sheet, a diagnostic view; reported events come from the post-processing rules varied here.
- **Upstream of the table**: keypoint model, masks and identity tracking.
- `secondary_threshold` does not change any count (it only annotates ambiguous frames).
"""


def render_markdown(long_df: pd.DataFrame, info: Dict[str, Any], robustness: str) -> str:
    base = info["baseline"]
    lines: List[str] = [
        "# Sensitivity analysis",
        "",
        f"Run: `{info['run']}`  ",
        f"Per-frame table: {info['n_rows']} rows, {info['n_pairs']} pair(s), {info['n_frames']} frames, "
        f"fps {_fmt(info['fps'])}.",
        "",
        "One parameter is varied at a time; the others stay at the baseline: "
        f"activation {_fmt(base['activation_threshold'])}, rare (FOL) {_fmt(base['activation_threshold_rare'])}, "
        f"smoothing window {base['smoothing_window']} frames, gap bridging {base['gap_bridging_max_gap']} frames, "
        f"minimum duration {_fmt(base['min_bout_sec'])} s"
        + (f", per-type minimums {base['min_bout_by_type']} s" if base["min_bout_by_type"] else "")
        + " (post-processing values read from `configs/contacts_postprocess_simple.yaml`). "
        "Cells are `n_events (total contact seconds)`; the baseline row is in bold. "
        "NC (no contact) is not counted.",
        "",
    ]
    if info.get("agreement") is not None:
        lines += [
            f"Check: the raw labels rebuilt at the baseline thresholds match the stored `contact_type` in "
            f"{info['agreement'] * 100:.2f} % of the rows"
            + ("." if info["agreement"] >= 0.9999 else
               " (the difference is frames where the stored label was set by something other than the plain "
               "argmax of the stored scores)."),
            "",
        ]
    lines += ["## Robustness", "", robustness.replace("Robustness. ", "", 1), ""]

    for param in PARAMETERS:
        sub = long_df[long_df["parameter"] == param]
        if sub.empty:
            continue
        lines += [f"## `{param}`: {PARAM_TITLES[param]}", "",
                  "| value | " + " | ".join(TYPE_ORDER) + " |",
                  "|---|" + "---|" * len(TYPE_ORDER)]
        for value in sub["value"].unique():
            rows = sub[sub["value"] == value].set_index("contact_type")
            is_base = bool(rows["is_baseline"].iloc[0])
            cells = []
            for ct in TYPE_ORDER:
                cell = f"{int(rows.loc[ct, 'n_events'])} ({rows.loc[ct, 'total_sec']:.1f} s)"
                cells.append(f"**{cell}**" if is_base else cell)
            label = f"**{_fmt(value)} (baseline)**" if is_base else _fmt(value)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        if param == "min_bout_sec" and base["min_bout_by_type"]:
            lines += [f"The per-type minimums {base['min_bout_by_type']} s stay in force while this value varies, "
                      "so they only bind below their own value.", ""]
        if param == "activation_threshold":
            lines += ["FOL uses `activation_threshold_rare`, so its row is constant here; the other types are "
                      "unaffected by the next table.", ""]

    lines += [NOT_REEVALUABLE, "## Method notes", "",
              "- Raw labels are rebuilt with the rule of `ScoreMap.argmax_type` (priority N2N, N2AG, FOL, SBS, N2B), "
              "vectorised; a sample of rows is checked against `ScoreMap` on every run. Missing scores count as 0.",
              "- Cleaned labels and events come from `compute_labels` and `extract_events` of "
              "`scripts/postprocess_contacts_simple.py`, applied per pair.",
              "- `sensitivity_long.csv` also holds `n_pairs` (pairs with at least one event of the type) and "
              "`n_pair_frames` (pair-frames carrying the type after cleaning); `mean_sec` is empty when there is "
              "no event.",
              ""]
    return "\n".join(lines)


def plot_parameter(long_df: pd.DataFrame, param: str, baseline_value: float, path: Path) -> None:
    """n_events vs value, one line per type, baseline value as a vertical dashed line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = long_df[long_df["parameter"] == param]
    xs = sorted(float(v) for v in sub["value"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for ct in TYPE_ORDER:
        s = sub[sub["contact_type"] == ct].assign(x=lambda d: d["value"].astype(float)).sort_values("x")
        ax.plot(s["x"], s["n_events"], marker=MARKERS[ct], markersize=6, linewidth=1.8,
                color=CT_COLORS.get(ct, "#444444"), label=ct)
    ax.axvline(float(baseline_value), color="#555555", linestyle="--", linewidth=1.2, zorder=0)
    ax.annotate("baseline", xy=(float(baseline_value), 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=8, color="#555555")
    ax.set_xticks(xs)
    ax.set_xticklabels([_fmt(x) for x in xs])
    ax.set_xlabel(PARAM_TITLES[param])
    ax.set_ylabel("Events (n)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(TYPE_ORDER), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(long_df: pd.DataFrame, info: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Write the CSV, the markdown report and one PNG per parameter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    csv_path = out_dir / "sensitivity_long.csv"
    long_df.to_csv(csv_path, index=False)
    written.append(csv_path)
    md_path = out_dir / "sensitivity_summary.md"
    md_path.write_text(render_markdown(long_df, info, robustness_statement(long_df)), encoding="utf-8")
    written.append(md_path)
    for param in PARAMETERS:
        if (long_df["parameter"] == param).any():
            png = out_dir / f"sensitivity_{param}.png"
            plot_parameter(long_df, param, info["baseline"][param], png)
            written.append(png)
    return written


# --------------------------------------------------------------------------- main
def analyse_run(run: Path, out_dir: Path, overrides: Sequence[str]) -> Tuple[pd.DataFrame, str, List[Path]]:
    pf = load_per_frame(run)
    if pf is None or pf.empty:
        raise SystemExit(f"No per-frame table (contacts_per_frame.csv or results.xlsx PerFrame sheet) under {run}")
    grid_raw, fps_override = load_grid(overrides)
    baseline = load_baseline()
    grid = complete_grid(grid_raw, baseline)
    fps = resolve_fps(run, fps_override)
    table = prepare_table(pf)
    check_against_scoremap(table, baseline["activation_threshold"], baseline["activation_threshold_rare"])

    agreement = None
    if table.stored_raw is not None:
        base_labels = normalize_labels(pd.Series(
            raw_labels(table.scores, baseline["activation_threshold"], baseline["activation_threshold_rare"])))
        agreement = float((base_labels == table.stored_raw).mean())
        logger.info("Baseline raw labels match the stored contact_type in %.2f %% of rows", agreement * 100)

    long_df = run_sensitivity(table, fps, grid, baseline)
    n_pairs = int(table.frames["pair_key"].nunique()) if "pair_key" in table.frames.columns else 1
    info = {"run": str(run), "fps": fps, "n_rows": table.n_rows, "n_pairs": n_pairs,
            "n_frames": int(table.frames["frame_idx"].nunique()), "baseline": baseline,
            "agreement": agreement}
    written = write_outputs(long_df, info, out_dir)
    return long_df, robustness_statement(long_df), written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline sensitivity of contact counts to the labelling and post-processing thresholds.")
    ap.add_argument("run_dir", type=Path, help="Run directory (contains contacts/results.xlsx or contacts_per_frame.csv)")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: <run_dir>/sensitivity)")
    ap.add_argument("overrides", nargs="*",
                    help="key=value grid overrides, e.g. min_bout_sec=[0.3,0.5] smoothing_window=7 fps=25")
    args = ap.parse_intermixed_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = args.out or (args.run_dir / "sensitivity")
    _, robustness, written = analyse_run(args.run_dir, out_dir, args.overrides)
    print()
    print(robustness)
    print()
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
