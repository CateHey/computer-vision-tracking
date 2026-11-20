"""README_WHAT_TO_REVIEW.md for a run folder: what every file is and what to review first.

`write_run_readme(run_dir, n_animals, fps, session_sec, extras)` is called at the end of a
`cutie_composite` run. `update_section(run_dir, title, markdown)` lets the validation tool
refresh its own section (clips, identity check) once it has run; it does nothing when the
README does not exist. Everything is best effort: a missing file or an unreadable log leaves a
blank, never an exception.

Logs of runs made before 2026-11-20 say "Rat N" and "N/N rats"; both spellings are read.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

README_NAME = "README_WHAT_TO_REVIEW.md"

RESULTS_SHEETS = [
    ("Summary", "headline numbers per contact type, a TOTAL row and the ANY row (time with any contact)"),
    ("Events", "the cleaned behavioural events: the reportable result (includes the no-contact rows)"),
    ("PerAnimal", "events and contact time per mouse, against any partner"),
    ("PairMatrix", "events and contact time for every pair of mice, including empty pairs"),
    ("Investigation", "nose-to-nose, nose-to-anogenital and nose-to-body merged into investigation bouts"),
    ("PerFrame", "complete per-frame geometry with raw and cleaned labels (raw data, not a claim)"),
    ("ByType", "raw versus cleaned counts per contact type"),
    ("Global", "raw versus cleaned session totals"),
    ("Parameters", "every parameter applied, plus the calibration"),
]
DIAGNOSTICS_SHEETS = [
    ("README", "one cell saying these tables are not validated"),
    ("Brief", "interactions too short to become events (under 0.3 s; following under 0.4 s)"),
    ("Following", "following episodes (FOL plus walking N2AG of the same pair, merged)"),
    ("Groups", "totals per group of look-alike contact types"),
    ("Similarity", "how alike the contact types are in this video"),
    ("Bouts", "raw tracker bouts with the voted initiator"),
    ("Dynamics", "approach and avoidance episodes without contact"),
    ("Individual", "per-mouse initiator statistics (a different basis from PerAnimal)"),
]

# top-level entry -> one-line meaning
FILE_MEANING = {
    "README_WHAT_TO_REVIEW.md": "this file",
    "config_used.yaml": "the exact parameters of this run",
    "provenance.json": "what produced the run: git commit, package versions, GPU, MD5 of the weights and config",
    "background.png": "the empty-arena image computed from the video, used to erase the other mice",
    "logs": "`run.log`: the full log of the run (read the tail first)",
    "overlays": "videos for visual checks: `cutie_composite_<date>.avi` (masks, keypoints, R1...RN labels) "
                "and one `cutie_mouseN_<date>.avi` per mouse (that mouse alone on the background; a second "
                "mouse visible means the erase step failed)",
    "tracks": "`tracks.jsonl.gz`: every mask, keypoint and id of every frame (JSON key `rats` = one entry per "
              "mouse); `area_per_mouse.png`: mask area against the mouse's own median, recovery episodes shaded",
    "contacts": "the contact analysis (see the lines below)",
    "validation": "blind clip review of the contacts: `clips/`, `thumbs/`, `index.html` (open this), "
                  "`review.xlsx` (fill this), `manifest.json` (what was sampled and how), `identity/` "
                  "(checkpoint images with each mouse labelled)",
    "sensitivity": "contact counts re-computed under changed thresholds (does the result depend on them?)",
}
CONTACT_MEANING = {
    "results.xlsx": "the reported tables, one workbook: the file to keep. Every sheet is covered by the clip review",
    "diagnostics.xlsx": "tracker diagnostics: NOT validated, NOT for reporting",
    "event_log.txt": "the events as readable text, with the per-mouse summary and the validation guidelines",
    "report.pdf": "the tracker's own diagnostic PDF (not validated)",
    "reports": "`*.png` figures: timeline comparison, duration by type, events by type, event duration "
               "distribution (raw versus cleaned)",
}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _find_key(obj: Any, key: str) -> Any:
    """First value of `key` anywhere in a nested dict, or None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _read_config(run_dir: Path) -> Dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001  yaml missing or file unreadable
        return {}


def _onoff(v: Any) -> str:
    return "n/a" if v is None else ("on" if v else "off")


def _fmt_duration(sec: Optional[float]) -> str:
    if not sec:
        return "unknown duration"
    m, s = divmod(int(round(sec)), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def _log_tail(log: Path, n: int = 400) -> List[str]:
    try:
        return log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _log_all(log: Path) -> List[str]:
    try:
        return log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def parse_log(lines: List[str], fps: float) -> Dict[str, Any]:
    """Pull the numbers the review starts from out of run.log (old and new wording)."""
    out: Dict[str, Any] = {"init_attempts": [], "pose": {}, "episodes": [], "recovery_line": None,
                           "handoff": None, "kp_carry": None, "mask_carry": None}
    for ln in lines:
        m = re.search(r"SAM3 init attempt (\d+):.*-> (\d+)/(\d+) animals", ln)
        if m:
            out["init_attempts"].append(f"attempt {m.group(1)}: {m.group(2)}/{m.group(3)}")
        m = re.search(r"Handoff: (\d+)/(\d+) (?:rats|mice)", ln)
        if m:
            out["handoff"] = (int(m.group(1)), int(m.group(2)))
        m = re.search(r"(?:Rat|Mouse) (\d+): pose detected in (\d+)/(\d+) frames \(([\d.]+)%\)", ln)
        if m:
            out["pose"][int(m.group(1))] = float(m.group(4))
        m = re.search(r"Frames with keypoint carry-over: .*\(([\d.]+)%\)", ln)
        if m:
            out["kp_carry"] = float(m.group(1))
        m = re.search(r"Frames with mask carry-over: .*\(([\d.]+)%\)", ln)
        if m:
            out["mask_carry"] = float(m.group(1))
        if "Recovery:" in ln and "episode" in ln:
            out["recovery_line"] = ln.split("] ", 1)[-1].split(": ", 1)[-1].strip()
        m = re.search(r"Area guard: episode (\d+)-(\d+) (\w+) after", ln)
        if m:
            s, e = int(m.group(1)), int(m.group(2))
            out["episodes"].append({"start": s, "end": e, "outcome": m.group(3),
                                    "sec": (e - s + 1) / fps if fps else None})
    return out


def _short_event_flags(run_dir: Path) -> List[str]:
    """Types whose events are mostly 0.3-0.4 s long (read from results.xlsx / Events)."""
    xlsx = run_dir / "contacts" / "results.xlsx"
    if not xlsx.exists():
        return []
    try:
        import pandas as pd
        ev = pd.read_excel(xlsx, sheet_name="Events")
        ev = ev[ev["contact_type"].astype(str).str.upper() != "NC"]
        flags = []
        for ct, g in ev.groupby("contact_type"):
            d = pd.to_numeric(g["duration_sec"], errors="coerce").dropna()
            if len(d) >= 5 and ((d >= 0.3) & (d <= 0.4)).mean() > 0.5:
                flags.append(f"{ct}: {int(((d >= 0.3) & (d <= 0.4)).sum())} of {len(d)} events last 0.3-0.4 s "
                             "(close to the minimum duration; check them for flicker)")
        return flags
    except Exception:  # noqa: BLE001  a README must never fail a run
        logger.warning("Could not read the event durations for the README", exc_info=True)
        return []


def red_flags(run_dir: Path, n_animals: int, fps: float) -> Tuple[List[str], bool]:
    """(flags found automatically, whether run.log could be read)."""
    lines = _log_all(run_dir / "logs" / "run.log")
    info = parse_log(lines, fps)
    flags: List[str] = []
    if info["handoff"] and info["handoff"][0] < info["handoff"][1]:
        flags.append(f"Handoff {info['handoff'][0]}/{info['handoff'][1]}: a mouse never reached CUTIE.")
    for mouse, pct in sorted(info["pose"].items()):
        if pct < 90:
            flags.append(f"Pose detected for mouse {mouse} in only {pct:.1f} % of frames.")
    for ep in info["episodes"]:
        if ep["outcome"] == "unresolved":
            flags.append(f"Recovery episode frames {ep['start']}-{ep['end']} is unresolved.")
        elif ep["sec"] and ep["sec"] > 2:
            flags.append(f"Recovery episode frames {ep['start']}-{ep['end']} lasted {ep['sec']:.1f} s "
                         f"({ep['outcome']}).")
    ident = run_dir / "validation" / "identity" / "identity.xlsx"
    if ident.exists():
        try:
            import pandas as pd
            df = pd.read_excel(ident)
            col = next((c for c in df.columns if str(c).lower() == "identity_ok"), None)
            if col is not None and (df[col].astype(str).str.strip().str.lower() == "no").any():
                flags.append("identity_ok = no in validation/identity/identity.xlsx: labels swapped on a mouse.")
        except Exception:  # noqa: BLE001
            pass
    flags += _short_event_flags(run_dir)
    return flags, bool(lines)


def _ordered(d: Path, order: List[str]) -> List[Path]:
    """Entries of d: the names in `order` first (as listed), then the rest alphabetically."""
    rank = {n: i for i, n in enumerate(order)}
    return sorted(d.iterdir(), key=lambda q: (rank.get(q.name, len(order)), q.name.lower()))


TOP_ORDER = ["README_WHAT_TO_REVIEW.md", "logs", "validation", "tracks", "contacts", "overlays",
             "sensitivity", "background.png", "config_used.yaml", "provenance.json"]
CONTACT_ORDER = ["results.xlsx", "diagnostics.xlsx", "event_log.txt", "report.pdf", "reports"]


def _files_table(run_dir: Path) -> List[str]:
    rows: List[Tuple[str, str]] = []
    for p in _ordered(run_dir, TOP_ORDER):
        name = p.name + ("/" if p.is_dir() else "")
        meaning = FILE_MEANING.get(p.name, "")
        if p.name == "contacts" and p.is_dir():
            rows.append((name, meaning))
            for c in _ordered(p, CONTACT_ORDER):
                rows.append((f"contacts/{c.name}" + ("/" if c.is_dir() else ""),
                             CONTACT_MEANING.get(c.name, "")))
                if c.name == "results.xlsx":
                    rows += [(f"&nbsp;&nbsp;sheet `{n}`", d) for n, d in RESULTS_SHEETS]
                elif c.name == "diagnostics.xlsx":
                    rows += [(f"&nbsp;&nbsp;sheet `{n}`", d) for n, d in DIAGNOSTICS_SHEETS]
            continue
        if p.name == "overlays" and p.is_dir():
            n = len(list(p.glob("*.avi")) + list(p.glob("*.mp4")))
            meaning = f"{meaning} ({n} video(s) here)"
        rows.append((name, meaning))
    out = ["| File | What it is |", "|---|---|"]
    out += [f"| {n if n.startswith('&') else '`' + n + '`'} | {m} |" for n, m in rows]
    return out


def _run_line(run_dir: Path, n_animals: int, fps: float, session_sec: float,
              extras: Dict[str, Any]) -> str:
    prov = _read_json(run_dir / "provenance.json")
    cfg = _read_config(run_dir)
    video = extras.get("video_path") or cfg.get("video_path") or "unknown video"
    v3 = extras.get("contact_rules_v3")
    if v3 is None:
        v3 = _find_key(cfg, "contact_rules_v3")
    rec = extras.get("recovery_enabled")
    if rec is None:
        r = cfg.get("recovery") if isinstance(cfg, dict) else None
        rec = r.get("enabled") if isinstance(r, dict) else None
    commit = (prov.get("git_commit") or "unknown")[:8]
    dirty = " (uncommitted changes)" if prov.get("git_dirty") else ""
    date = (prov.get("timestamp_utc") or datetime.now().isoformat())[:10]
    return (f"**What this is:** a `cutie_composite` run on `{Path(str(video)).name}`: {n_animals} mice, "
            f"{_fmt_duration(session_sec)}, {fps:g} fps, run on {date}, code commit `{commit}`{dirty}; "
            f"contact_rules_v3 {_onoff(v3)}, area-guard recovery {_onoff(rec)}.")


REVIEW_STEPS = [
    "**`logs/run.log`, the last lines.** SAM3 init attempts, `Handoff: N/N mice`, pose detected per mouse "
    "(%), keypoint and mask carry-over (%), and the `Recovery:` line (episodes by outcome).",
    "**`validation/identity/identity.html`.** Does each label (R1...RN) stay on the same mouse for the "
    "whole session? Any change means every per-mouse number is wrong from that point.",
    "**`tracks/area_per_mouse.png`** (when present). A mouse whose area jumps, collapses or stays off the "
    "median band for long was merged with a neighbour or lost.",
    "**`contacts/results.xlsx`.** `Summary` (including the `ANY` row), `Investigation`, `PerAnimal`, "
    "`PairMatrix`.",
    "**`validation/index.html` and `validation/review.xlsx`: the blind clip review.** Clips play at 0.5x. "
    "Fill only the rows with `in_sample = TRUE`: `verdict`, `correct_type`, `investigator_ok`, "
    "`boundaries_ok`, `subtype_ok`. For no-contact (NC) rows: `nc_correct` or `missed_contact`.",
    "**Score the review:** `python scripts/validate_contacts.py score <run>/validation/review.xlsx`.",
]

FLAG_RULES = [
    "Handoff below N/N (a mouse never reached CUTIE).",
    "Pose detected for a mouse in under 90 % of frames.",
    "A recovery episode that is `unresolved` or lasts more than 2 s.",
    "Any `identity_ok = no` in the identity check.",
    "A contact type whose events are mostly 0.3-0.4 s long (just above the minimum duration: flicker).",
]


def _build(run_dir: Path, n_animals: int, fps: float, session_sec: float,
           extras: Dict[str, Any]) -> str:
    L = ["# What to review in this run", "", _run_line(run_dir, n_animals, fps, session_sec, extras), "",
         "## Review in this order", ""]
    L += [f"{i}. {s}" for i, s in enumerate(REVIEW_STEPS, 1)]
    L += ["", "## Files in this folder", ""] + _files_table(run_dir)
    L += ["", "## Red flags", "", "A run is suspect when one of these holds:", ""]
    L += [f"- {r}" for r in FLAG_RULES]
    found, had_log = red_flags(run_dir, n_animals, fps)
    L += ["", "**Found automatically in this run:**", ""]
    if found:
        L += [f"- {f}" for f in found]
    else:
        L += ["- none" if had_log else "- nothing could be checked (no `logs/run.log`)"]
    return "\n".join(L) + "\n"


def write_run_readme(run_dir, n_animals: int, fps: float, session_sec: float,
                     extras: Optional[Dict[str, Any]] = None) -> Path:
    """Write `<run_dir>/README_WHAT_TO_REVIEW.md` and return its path.

    extras (all optional): video_path, contact_rules_v3, recovery_enabled; each is read from
    config_used.yaml when absent.
    """
    run_dir = Path(run_dir)
    path = run_dir / README_NAME
    path.write_text(_build(run_dir, n_animals, fps, session_sec, dict(extras or {})), encoding="utf-8")
    return path


def update_section(run_dir, title: str, markdown: str) -> bool:
    """Replace (or append) the `## <title>` section of the README; False when there is no README."""
    path = Path(run_dir) / README_NAME
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    block = f"## {title}\n\n{markdown.strip()}\n"
    pat = re.compile(rf"^## {re.escape(title)}[ \t]*\n.*?(?=^## |\Z)", re.S | re.M)
    if pat.search(text):
        text = pat.sub(lambda _m: block + "\n", text, count=1)
    else:
        text = text.rstrip("\n") + "\n\n" + block
    path.write_text(text, encoding="utf-8")
    return True
