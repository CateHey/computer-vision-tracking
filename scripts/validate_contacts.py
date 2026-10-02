#!/usr/bin/env python
"""Validation phase for detected social contacts.

Subcommands
-----------
sample : stratified sample of events -> evidence clips, contact sheets,
         review.xlsx, index.html, manifest.json
score  : read a filled review.xlsx -> validation_summary.{md,json,png}

Usage
-----
    python scripts/validate_contacts.py sample <run_dir> [--per-type 15] [key=value ...]
    python scripts/validate_contacts.py score <validation_dir>/review.xlsx
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "per_type": 15, "pad_sec": 1.0, "speed": 0.5, "seed": 42, "crf": 23,
    "source": None, "types": None,
    "duration_bins": [0.3, 0.5, 1.0, 2.0],
    "score_grid": [0.0, 0.5, 0.6, 0.7, 0.8, 0.9],
    "duration_grid": [0.0, 0.2, 0.3, 0.5, 1.0, 2.0],
}
TYPE_NAMES = {
    "N2N": "Nose-to-Nose", "N2AG": "Nose-to-Anogenital", "N2B": "Nose-to-Body",
    "SBS": "Side-by-Side", "FOL": "Following",
}
VERDICTS = ["real", "flicker", "wrong_type", "tracking_error", "unsure"]
CORRECT_TYPES = ["N2N", "N2AG", "N2B", "SBS", "FOL", "NC"]
QUALITY_FLAGS = ["stale_keypoints", "missing_keypoints", "single_detection",
                 "high_mask_overlap", "fol_used_centroid"]
TERTILES = ["short", "medium", "long"]
NO_CONTACT = {"", "none", "nc", "nan", "non_contact", "null"}


# --------------------------------------------------------------------------- config
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
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s.startswith("["):
        try:
            return json.loads(s)
        except ValueError:
            pass
    return s


def load_config(overrides: List[str]) -> dict:
    """Load configs/validation.yaml (or defaults) and apply key=value overrides."""
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    path = _PROJECT_ROOT / "configs" / "validation.yaml"
    if path.exists():
        import yaml
        cfg.update(yaml.safe_load(path.read_text()) or {})
    for ov in overrides:
        if "=" not in ov:
            logger.warning("Ignoring invalid override (no '='): %s", ov)
            continue
        k, v = ov.split("=", 1)
        _set_nested(cfg, k.strip(), _parse_value(v.strip()))
    return cfg


# --------------------------------------------------------------------------- helpers
def pair_label(pair_key: str) -> str:
    """'0_1' -> 'R1-R2'."""
    try:
        i, j = str(pair_key).split("_")
        return f"R{int(i) + 1}-R{int(j) + 1}"
    except ValueError:
        return str(pair_key)


def fmt_time(sec: float) -> str:
    """Seconds -> mm:ss.s."""
    m, s = divmod(float(sec), 60)
    return f"{int(m):02d}:{s:04.1f}"


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion. (nan, nan) if n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------- loading
def _read_xlsx_sheet(path: Path, sheet: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:  # missing file / sheet
        return None


def _find_xlsx(run: Path) -> Optional[Path]:
    for p in (run / "contacts" / "results.xlsx", run / "results.xlsx"):
        if p.exists():
            return p
    return None


def load_events(run: Path, source: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Load events (preferred, if they have pair_key) or bouts. Returns (df, source_name)."""
    xlsx = _find_xlsx(run)
    cdir = run / "contacts"
    cands: List[Tuple[str, Optional[pd.DataFrame]]] = []
    if source in (None, "events"):
        ev = None
        if (cdir / "contacts_real_events.csv").exists():
            ev = pd.read_csv(cdir / "contacts_real_events.csv")
        if (ev is None or "pair_key" not in ev.columns) and xlsx:
            ev = _read_xlsx_sheet(xlsx, "Events")
        cands.append(("events", ev))
    if source in (None, "bouts"):
        bt = pd.read_csv(cdir / "contact_bouts.csv") if (cdir / "contact_bouts.csv").exists() else None
        if bt is None and xlsx:
            bt = _read_xlsx_sheet(xlsx, "Bouts")
        cands.append(("bouts", bt))
    for name, df in cands:
        if df is not None and len(df) and "pair_key" in df.columns:
            return _normalise_events(df), name
    raise FileNotFoundError(f"No events/bouts table with pair_key found under {run}")


def _normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["contact_type"] = df["contact_type"].astype(str).str.strip().str.upper()
    df = df[~df["contact_type"].str.lower().isin(NO_CONTACT)].reset_index(drop=True)
    df["pair_key"] = df["pair_key"].astype(str)
    df["start_frame"] = df["start_frame"].astype(int)
    df["end_frame"] = df["end_frame"].astype(int)
    if "n_frames" not in df.columns:
        df["n_frames"] = df["end_frame"] - df["start_frame"] + 1
    if "duration_sec" not in df.columns:
        df["duration_sec"] = np.nan
    if "peak_score" not in df.columns:
        df["peak_score"] = np.nan
    return df


def load_per_frame(run: Path) -> Optional[pd.DataFrame]:
    """Per-frame pair table (csv, else PerFrame sheet); None if unavailable."""
    p = run / "contacts" / "contacts_per_frame.csv"
    if p.exists():
        return pd.read_csv(p)
    xlsx = _find_xlsx(run)
    return _read_xlsx_sheet(xlsx, "PerFrame") if xlsx else None


def find_fps(run: Path, video: Optional[Path]) -> float:
    """fps from session_summary.json, config_used.yaml, or the video."""
    try:
        s = json.loads((run / "contacts" / "session_summary.json").read_text())
        if s.get("metadata", {}).get("fps"):
            return float(s["metadata"]["fps"])
    except Exception:
        pass
    try:
        import yaml
        c = yaml.safe_load((run / "config_used.yaml").read_text()) or {}
        for k in ("fps", "video_fps"):
            if isinstance(c.get(k), (int, float)):
                return float(c[k])
        for sect in c.values():
            if isinstance(sect, dict) and isinstance(sect.get("fps"), (int, float)):
                return float(sect["fps"])
    except Exception:
        pass
    if video is not None:
        import cv2
        cap = cv2.VideoCapture(str(video))
        f = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if f > 0:
            return float(f)
    return 30.0


def find_overlay(run: Path) -> Path:
    """Locate the main overlay video (not the per-animal unitary ones)."""
    ov = run / "overlays"
    for pat in ("cutie_composite_*.avi", "*composite*.avi", "*.avi", "*.mp4"):
        for p in sorted(ov.glob(pat)):
            if "unitary" not in p.name:
                return p
    raise FileNotFoundError(f"No overlay video in {ov}")


# --------------------------------------------------------------------------- sampling
def assign_tertiles(durations: pd.Series) -> pd.Series:
    """Label each event short/medium/long by duration rank (ties broken by order)."""
    n = len(durations)
    ranks = durations.rank(method="first").astype(int) - 1
    idx = np.minimum((ranks * 3) // max(n, 1), 2)
    return idx.map(lambda i: TERTILES[int(i)])


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample up to n events evenly across duration tertiles (seeded).

    Returns rows with a 'duration_tertile' column. Takes all if len(df) <= n;
    when a tertile has too few events, its quota is redistributed.
    """
    df = df.copy()
    df["duration_tertile"] = assign_tertiles(df["duration_sec"]) if len(df) else []
    if len(df) <= n:
        return df.sort_values("start_frame")
    rng = np.random.default_rng(seed)
    groups = {t: df[df["duration_tertile"] == t] for t in TERTILES}
    quota = {t: n // 3 + (1 if i < n % 3 else 0) for i, t in enumerate(TERTILES)}
    take = {t: min(quota[t], len(groups[t])) for t in TERTILES}
    spare = n - sum(take.values())
    while spare > 0:
        grew = False
        for t in TERTILES:
            if spare > 0 and take[t] < len(groups[t]):
                take[t] += 1
                spare -= 1
                grew = True
        if not grew:
            break
    parts = []
    for t in TERTILES:
        g = groups[t]
        if take[t]:
            pick = rng.choice(len(g), size=take[t], replace=False)
            parts.append(g.iloc[np.sort(pick)])
    return pd.concat(parts).sort_values("start_frame")


# --------------------------------------------------------------------------- clips
def _score_col(ctype: str) -> str:
    return f"score_{ctype.lower()}"


def event_stats(ev: pd.Series, pf: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Per-event aggregates from the per-frame table."""
    out: Dict[str, Any] = {}
    if pf is None:
        return out
    sub = pf[(pf["pair_key"].astype(str) == ev["pair_key"])
             & (pf["frame_idx"] >= ev["start_frame"]) & (pf["frame_idx"] <= ev["end_frame"])]
    if sub.empty:
        return out
    sc = _score_col(ev["contact_type"])
    if sc in sub:
        out["mean_score"] = float(sub[sc].mean())
    if "secondary_type" in sub:
        sec = sub["secondary_type"].dropna().astype(str)
        sec = sec[~sec.str.lower().isin(NO_CONTACT)]
        if len(sec):
            out["secondary_type"] = sec.mode().iloc[0]
            if "secondary_score" in sub:
                out["secondary_score"] = float(sub.loc[sec.index, "secondary_score"].mean())
    for f in QUALITY_FLAGS:
        if f in sub:
            out[f"frac_{f}"] = float((sub[f].fillna(0) > 0).mean())
    for c in ("nose_nose_dist_bl", "mask_contact_bl"):
        if c in sub:
            out[f"mean_{c}"] = float(sub[c].mean())
    return out


def _put(img, text, org, scale=0.55, color=(255, 255, 255), th=1):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def annotate(frame: np.ndarray, sid: str, ev: pd.Series, fps: float, fidx: int,
             row: Optional[pd.Series]) -> np.ndarray:
    """Burn banner, state border and live values into a frame."""
    import cv2
    h, w = frame.shape[:2]
    inside = ev["start_frame"] <= fidx <= ev["end_frame"]
    col = (0, 0, 220) if inside else (140, 140, 140)
    ctype = ev["contact_type"]
    ban = frame.copy()
    cv2.rectangle(ban, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.rectangle(ban, (0, h - 30), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(ban, 0.6, frame, 0.4, 0)
    pl = pair_label(ev["pair_key"])
    _put(frame, f"{sid} | {ctype} {TYPE_NAMES.get(ctype, '')} | look at {pl}", (8, 22), 0.6)
    _put(frame, f"event {fmt_time(ev['start_frame'] / fps)}-{fmt_time((ev['end_frame'] + 1) / fps)}  "
                f"frames {ev['start_frame']}-{ev['end_frame']}  dur {ev['n_frames'] / fps:.2f}s  "
                f"[{'LABEL ACTIVE' if inside else 'context'}] f={fidx}", (8, 48), 0.5,
         (200, 200, 255) if inside else (200, 200, 200))
    if row is not None:
        parts = [f"raw={row.get('contact_type', '?')}"]
        sc = _score_col(ctype)
        if sc in row and pd.notna(row[sc]):
            parts.append(f"{sc[6:]}={row[sc]:.2f}")
        for c, lab in (("nose_nose_dist_bl", "nn"), ("centroid_dist_bl", "cd"), ("mask_contact_bl", "mc")):
            if c in row and pd.notna(row[c]):
                parts.append(f"{lab}={row[c]:.2f}BL")
        _put(frame, "  ".join(parts), (8, h - 9), 0.5)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 8 if inside else 4)
    return frame


class _Writer:
    """H.264 writer via ffmpeg pipe, cv2 mp4v fallback."""

    def __init__(self, path: Path, w: int, h: int, fps: float, crf: int):
        self.proc, self.cv = None, None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
        else:
            import cv2
            logger.warning("ffmpeg not found; falling back to cv2 mp4v")
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, f: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(f).tobytes())
        else:
            self.cv.write(f)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def cut_clip(cap, sid: str, ev: pd.Series, pf_pair: Optional[pd.DataFrame], fps: float,
             pad: int, nframes: int, out: Path, sheet: Path, speed: float, crf: int) -> int:
    """Write the annotated clip and 3-thumbnail contact sheet. Returns frames written."""
    import cv2
    a = max(0, int(ev["start_frame"]) - pad)
    b = min(nframes - 1, int(ev["end_frame"]) + pad)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    wr = None
    keep: Dict[int, np.ndarray] = {}
    want = {int(ev["start_frame"]), (int(ev["start_frame"]) + int(ev["end_frame"])) // 2, int(ev["end_frame"])}
    n = 0
    for fi in range(a, b + 1):
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        w2, h2 = w - w % 2, h - h % 2
        if wr is None:
            wr = _Writer(out, w2, h2, fps * speed, crf)
        row = pf_pair.loc[fi] if pf_pair is not None and fi in pf_pair.index else None
        fr = annotate(fr, sid, ev, fps, fi, row)[:h2, :w2]
        wr.write(fr)
        if fi in want:
            keep[fi] = fr
        n += 1
    if wr:
        wr.close()
    if keep:
        thumbs = [cv2.resize(keep[k], None, fx=0.5, fy=0.5) for k in sorted(keep)]
        cv2.imwrite(str(sheet), np.hstack(thumbs))
    return n


# --------------------------------------------------------------------------- outputs
REVIEWER_COLS = ["verdict", "correct_type", "reviewer", "notes"]


def write_review_xlsx(rows: List[Dict[str, Any]], path: Path) -> None:
    """review.xlsx with dropdowns + Spanish Instructions sheet."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    cols = ["sample_id", "clip", "thumbnail", "pair_label", "contact_type", "contact_label",
            "start_time", "end_time", "duration_sec", "duration_frames", "duration_tertile",
            "peak_score", "mean_score", "secondary_type", "secondary_score"] \
        + [f"frac_{f}" for f in QUALITY_FLAGS] + ["mean_nose_nose_dist_bl", "mean_mask_contact_bl"] + REVIEWER_COLS
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
    for r in rows:
        ws.append([r.get(c) for c in cols])
        cell = ws.cell(row=ws.max_row, column=cols.index("clip") + 1)
        cell.hyperlink = r["clip"]
        cell.font = Font(color="0563C1", underline="single")
    ws.freeze_panes = "B2"
    for i, c in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(i)].width = 38 if c in ("clip", "thumbnail", "notes") else (
            22 if c == "contact_label" else 14)
    last = max(len(rows) + 1, 2)
    for col, opts in (("verdict", VERDICTS), ("correct_type", CORRECT_TYPES)):
        dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        L = get_column_letter(cols.index(col) + 1)
        dv.add(f"{L}2:{L}{last}")
    ins = wb.create_sheet("Instructions")
    text = [
        "Instrucciones de revision",
        "Para cada fila abra el clip (borde ROJO = etiqueta activa, GRIS = contexto) y complete 'verdict'.",
        "Fijese en las ratas indicadas en pair_label. Los clips van a camara lenta.",
        "",
        "real: el contacto fisico/social del tipo indicado ocurre visiblemente durante la mayor parte del intervalo de borde rojo.",
        "flicker (destello): etiqueta activa pero sin contacto real, un roce momentaneo (< ~0.3 s) o jitter de keypoints.",
        "wrong_type: hay una interaccion real pero de otro tipo; complete 'correct_type' (N2N, N2AG, N2B, SBS, FOL o NC).",
        "tracking_error: intercambio de identidad, mascara fusionada o keypoints sobre el animal equivocado.",
        "unsure: no se puede decidir con el clip; se excluye de la precision.",
        "",
        "Tambien complete 'reviewer' (sus iniciales) y 'notes' (opcional). No modifique las demas columnas.",
    ]
    for t in text:
        ins.append([t])
    ins["A1"].font = Font(bold=True, size=13)
    ins.column_dimensions["A"].width = 130
    wb.save(path)


def write_index_html(rows: List[Dict[str, Any]], path: Path, title: str) -> None:
    """Self-contained gallery grouped by type with filter buttons."""
    types = sorted({r["contact_type"] for r in rows})
    esc = html.escape
    cards = []
    for r in rows:
        peak = "n/a" if r["peak_score"] is None else f'{r["peak_score"]:.2f}'
        cards.append(
            f'<div class="card" data-type="{esc(r["contact_type"])}"><h3>{esc(r["sample_id"])} '
            f'<span class="t">{esc(r["contact_type"])}</span></h3>'
            f'<video controls loop muted preload="metadata" src="{esc(r["clip"])}"></video>'
            f'<p>{esc(str(r["contact_label"]))} &middot; <b>{esc(r["pair_label"])}</b><br>'
            f'{esc(r["start_time"])} - {esc(r["end_time"])} &middot; {r["duration_sec"]:.2f} s '
            f'({r["duration_frames"]} f, {esc(r["duration_tertile"])})<br>'
            f'peak score {peak}</p></div>')
    btns = '<button data-f="all" class="on">all</button>' + "".join(
        f'<button data-f="{esc(t)}">{esc(t)}</button>' for t in types)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{{--bg:#fff;--fg:#1a1a1a;--card:#f3f4f6;--acc:#305496}}
@media(prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--card:#242424;--acc:#8fb0ee}}}}
body{{background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif;margin:16px}}
#g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}
.card{{background:var(--card);border-radius:8px;padding:10px}}.card video{{width:100%}}
.card h3{{margin:0 0 6px}}.t{{color:var(--acc)}}p{{margin:6px 0 0;font-size:13px}}
button{{margin:0 6px 10px 0;padding:6px 12px;border:1px solid var(--acc);background:none;color:var(--fg);border-radius:6px;cursor:pointer}}
button.on{{background:var(--acc);color:var(--bg)}}</style></head><body>
<h1>{esc(title)}</h1><p>{len(rows)} samples. Red border = label active, grey = context.</p>
<div>{btns}</div><div id="g">{''.join(cards)}</div><script>
document.querySelectorAll('button').forEach(b=>b.onclick=()=>{{
document.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('.card').forEach(c=>c.style.display=(b.dataset.f==='all'||c.dataset.type===b.dataset.f)?'':'none');}});
</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


def cmd_sample(args: argparse.Namespace, overrides: List[str]) -> int:
    """Run the `sample` subcommand."""
    import cv2
    cfg = load_config(overrides)
    for k in ("per_type", "pad_sec", "speed", "seed", "source", "types"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    run = Path(args.run_dir).resolve()
    out = Path(args.out) if args.out else run / "validation"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "thumbs").mkdir(exist_ok=True)

    events, src = load_events(run, cfg["source"])
    pf = load_per_frame(run)
    if pf is None:
        logger.warning("No per-frame table; clips will lack live values")
    video = find_overlay(run)
    fps = find_fps(run, video)
    logger.info("Source=%s, %d events, video=%s, fps=%.2f", src, len(events), video.name, fps)
    if cfg["types"]:
        keep = {t.strip().upper() for t in str(cfg["types"]).split(",")}
        events = events[events["contact_type"].isin(keep)]

    parts, avail, sampled = [], {}, {}
    for t, g in events.groupby("contact_type"):
        s = stratified_sample(g.reset_index(drop=True), int(cfg["per_type"]), int(cfg["seed"]))
        avail[t], sampled[t] = len(g), len(s)
        parts.append(s)
    if not parts:
        logger.error("No events to sample")
        return 1
    sample = pd.concat(parts).sort_values(["contact_type", "start_frame"]).reset_index(drop=True)
    sample["sample_id"] = [f"V{i + 1:03d}" for i in range(len(sample))]

    cap = cv2.VideoCapture(str(video))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pad = int(round(float(cfg["pad_sec"]) * fps))
    pf_pairs: Dict[str, pd.DataFrame] = {}
    if pf is not None:
        for k, g in pf.groupby(pf["pair_key"].astype(str)):
            pf_pairs[k] = g.drop_duplicates("frame_idx").set_index("frame_idx")

    rows = []
    for _, ev in sample.iterrows():
        sid = ev["sample_id"]
        clip, sheet = f"clips/{sid}_{ev['contact_type']}_{ev['pair_key']}.mp4", f"thumbs/{sid}.png"
        n = cut_clip(cap, sid, ev, pf_pairs.get(ev["pair_key"]), fps, pad, nframes,
                     out / clip, out / sheet, float(cfg["speed"]), int(cfg["crf"]))
        st = event_stats(ev, pf)
        dur = ev["duration_sec"] if pd.notna(ev["duration_sec"]) else ev["n_frames"] / fps
        peak = None if pd.isna(ev["peak_score"]) else float(ev["peak_score"])
        rows.append({
            "sample_id": sid, "clip": clip, "thumbnail": sheet, "pair_label": pair_label(ev["pair_key"]),
            "contact_type": ev["contact_type"], "contact_label": TYPE_NAMES.get(ev["contact_type"], ev["contact_type"]),
            "start_time": fmt_time(ev["start_frame"] / fps), "end_time": fmt_time((ev["end_frame"] + 1) / fps),
            "duration_sec": round(float(dur), 3), "duration_frames": int(ev["n_frames"]),
            "duration_tertile": ev["duration_tertile"], "peak_score": peak,
            "mean_score": st.get("mean_score"), "secondary_type": st.get("secondary_type"),
            "secondary_score": st.get("secondary_score"),
            **{f"frac_{f}": st.get(f"frac_{f}") for f in QUALITY_FLAGS},
            "mean_nose_nose_dist_bl": st.get("mean_nose_nose_dist_bl"),
            "mean_mask_contact_bl": st.get("mean_mask_contact_bl"),
            "frames_written": n,
        })
        logger.info("%s %s %s -> %d frames", sid, ev["contact_type"], pair_label(ev["pair_key"]), n)
    cap.release()

    write_review_xlsx(rows, out / "review.xlsx")
    write_index_html(rows, out / "index.html", f"Contact validation - {run.name}")
    (out / "manifest.json").write_text(json.dumps({
        "source": src, "run_dir": str(run), "video": str(video), "fps": fps, "params": cfg,
        "seed": cfg["seed"], "available_per_type": avail, "sampled_per_type": sampled,
        "n_samples": len(rows)}, indent=2, default=str))
    logger.info("Wrote %d clips to %s", len(rows), out)
    return 0


# --------------------------------------------------------------------------- score
def _prec(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    n_real = int((df["verdict"] == "real").sum())
    denom = int((df["verdict"] != "unsure").sum())
    lo, hi = wilson_ci(n_real, denom)
    return {"n_reviewed": n, "n_real": n_real, "n_unsure": n - denom, "n_decided": denom,
            "precision": n_real / denom if denom else None,
            "ci_low": None if math.isnan(lo) else lo, "ci_high": None if math.isnan(hi) else hi,
            "verdicts": {v: int((df["verdict"] == v).sum()) for v in VERDICTS}}


def compute_summary(df: pd.DataFrame, cfg: dict) -> Dict[str, Any]:
    """Aggregate a filled review table into the summary dict."""
    df = df.copy()
    df["verdict"] = df["verdict"].astype(str).str.strip().str.lower()
    df = df[df["verdict"].isin(VERDICTS)]
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    df["peak_score"] = pd.to_numeric(df.get("peak_score"), errors="coerce")
    s: Dict[str, Any] = {"n_reviewed": len(df), "overall": _prec(df) if len(df) else {}}
    s["per_type"] = {t: _prec(g) for t, g in df.groupby("contact_type")}

    def flick(g: pd.DataFrame) -> Dict[str, Any]:
        return {"n": len(g), "n_flicker": int((g["verdict"] == "flicker").sum()),
                "flicker_rate": float((g["verdict"] == "flicker").mean()) if len(g) else None}
    s["flicker_by_tertile"] = {t: flick(g) for t, g in df.groupby("duration_tertile")} \
        if "duration_tertile" in df else {}
    edges = [0.0] + list(cfg["duration_bins"]) + [float("inf")]
    labels = [f"<{edges[1]:g}s"] + [f"{edges[i]:g}-{edges[i + 1]:g}s" for i in range(1, len(edges) - 2)] \
        + [f">{edges[-2]:g}s"]
    bins = pd.cut(df["duration_sec"], edges, labels=labels, right=False)
    s["flicker_by_duration_bin"] = {str(lab): flick(df[bins == lab]) for lab in labels}
    conf: Dict[str, Dict[str, int]] = {}
    for _, r in df[df["verdict"] == "wrong_type"].iterrows():
        ct = r.get("correct_type")
        c = "UNSPECIFIED" if pd.isna(ct) else str(ct).upper()
        conf.setdefault(r["contact_type"], {}).setdefault(c, 0)
        conf[r["contact_type"]][c] += 1
    s["wrong_type_confusion"] = conf
    s["tracking_error_rate"] = float((df["verdict"] == "tracking_error").mean()) if len(df) else None
    fl = df[df["verdict"] == "flicker"][["sample_id", "contact_type", "duration_sec", "peak_score"]]
    s["flicker_samples"] = [{k: (None if pd.isna(v) else v) for k, v in r.items()}
                            for r in fl.to_dict("records")]
    dec = df[df["verdict"] != "unsure"]
    n_real_all = max(int((dec["verdict"] == "real").sum()), 1)

    def grid(col: str, values: List[float]) -> List[Dict[str, Any]]:
        out = []
        for v in values:
            k = dec[dec[col] >= v]
            n_real = int((k["verdict"] == "real").sum())
            lo, hi = wilson_ci(n_real, len(k))
            out.append({"threshold": v, "n_kept": len(k),
                        "precision": n_real / len(k) if len(k) else None,
                        "ci_low": None if math.isnan(lo) else lo, "ci_high": None if math.isnan(hi) else hi,
                        "recall_of_reviewed_real": n_real / n_real_all})
        return out
    s["min_duration_grid"] = grid("duration_sec", list(cfg["duration_grid"]))
    s["min_peak_score_grid"] = grid("peak_score", list(cfg["score_grid"]))
    return s


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def render_md(s: Dict[str, Any]) -> str:
    """Markdown rendering of the summary."""
    L = ["# Validation summary", "",
         f"Reviewed samples: **{s['n_reviewed']}**. Precision = real / (reviewed - unsure), "
         "Wilson 95% CI. Small n means wide intervals - always read n.", "",
         "## Precision per type", "",
         "| type | n | real | unsure | precision | 95% CI | verdicts |", "|---|---|---|---|---|---|---|"]
    for t, p in s["per_type"].items():
        ci = f"{_pct(p['ci_low'])}-{_pct(p['ci_high'])}" if p["ci_low"] is not None else "n/a"
        L.append(f"| {t} | {p['n_reviewed']} | {p['n_real']} | {p['n_unsure']} | {_pct(p['precision'])} | {ci} | "
                 + ", ".join(f"{k}={v}" for k, v in p["verdicts"].items() if v) + " |")
    L += ["", "## Flicker rate by duration tertile", "", "| tertile | n | flicker | rate |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['n_flicker']} | {_pct(v['flicker_rate'])} |"
          for k, v in s["flicker_by_tertile"].items()]
    L += ["", "## Flicker rate by duration bin", "", "| bin | n | flicker | rate |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['n_flicker']} | {_pct(v['flicker_rate'])} |"
          for k, v in s["flicker_by_duration_bin"].items()]
    L += ["", "## Wrong-type confusion (label -> correct_type)", ""]
    if s["wrong_type_confusion"]:
        L += [f"- {t}: " + ", ".join(f"{c} x{n}" for c, n in d.items())
              for t, d in s["wrong_type_confusion"].items()]
    else:
        L.append("None.")
    L += ["", f"Tracking-error rate: {_pct(s['tracking_error_rate'])}", "",
          "## Precision if only events with duration >= d were kept", "",
          "| d (s) | n kept | precision | 95% CI |", "|---|---|---|---|"]
    for g in s["min_duration_grid"]:
        L.append(f"| {g['threshold']:g} | {g['n_kept']} | {_pct(g['precision'])} | "
                 f"{_pct(g['ci_low'])}-{_pct(g['ci_high'])} |")
    L += ["", "## Precision if only events with peak_score >= s were kept", "",
          "| s | n kept | precision | 95% CI |", "|---|---|---|---|"]
    for g in s["min_peak_score_grid"]:
        L.append(f"| {g['threshold']:g} | {g['n_kept']} | {_pct(g['precision'])} | "
                 f"{_pct(g['ci_low'])}-{_pct(g['ci_high'])} |")
    L += ["", "## Samples marked flicker", "", "| sample | type | duration (s) | peak_score |", "|---|---|---|---|"]
    L += [f"| {r['sample_id']} | {r['contact_type']} | {r['duration_sec']} | {r['peak_score']} |"
          for r in s["flicker_samples"]]
    return "\n".join(L) + "\n"


def plot_precision(s: Dict[str, Any], path: Path) -> None:
    """Precision per type with Wilson CI error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = [(t, p) for t, p in s["per_type"].items() if p["precision"] is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    if items:
        x = np.arange(len(items))
        pr = np.array([p["precision"] for _, p in items])
        lo = pr - np.array([p["ci_low"] for _, p in items])
        hi = np.array([p["ci_high"] for _, p in items]) - pr
        ax.bar(x, pr, yerr=[lo, hi], capsize=4, color="#305496")
        ax.set_xticks(x, [f"{t}\n(n={p['n_decided']})" for t, p in items])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("precision (real / decided)")
    ax.set_title("Contact precision by type (Wilson 95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def cmd_score(args: argparse.Namespace, overrides: List[str]) -> int:
    """Run the `score` subcommand."""
    cfg = load_config(overrides)
    path = Path(args.review)
    df = pd.read_excel(path, sheet_name="review")
    df = df[df["verdict"].notna()]
    if df.empty:
        logger.error("No verdicts filled in %s", path)
        return 1
    s = compute_summary(df, cfg)
    (path.parent / "validation_summary.json").write_text(json.dumps(s, indent=2, default=str))
    (path.parent / "validation_summary.md").write_text(render_md(s), encoding="utf-8")
    plot_precision(s, path.parent / "validation_precision.png")
    logger.info("Scored %d reviewed samples -> %s", s["n_reviewed"], path.parent)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample", help="sample events and build evidence clips")
    sp.add_argument("run_dir")
    sp.add_argument("--per-type", dest="per_type", type=int)
    sp.add_argument("--pad-sec", dest="pad_sec", type=float)
    sp.add_argument("--speed", type=float)
    sp.add_argument("--seed", type=int)
    sp.add_argument("--types")
    sp.add_argument("--source", choices=["events", "bouts"])
    sp.add_argument("--out")
    sc = sub.add_parser("score", help="score a filled review.xlsx")
    sc.add_argument("review")
    args, extra = ap.parse_known_args(argv)
    overrides = [e for e in extra if "=" in e]
    return cmd_sample(args, overrides) if args.cmd == "sample" else cmd_score(args, overrides)


if __name__ == "__main__":
    sys.exit(main())
