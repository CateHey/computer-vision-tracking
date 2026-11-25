"""Interactive, time-animated bodily map: data export + inline CSS/JS (offline, no libraries).

``build_payload`` turns the located contact records into a compact columnar JSON object that
``gallery.py`` embeds inline as ``const BM_DATA = {...}``. The silhouette comes from
``template.py`` (same outline, extents and aspect as the static PNGs), so the browser maps
(t, s) -> pixels exactly like matplotlib does. ``BM_JS`` defines the reusable ``BMPlayer`` widget.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import template as T
from .locate import slot_color_rgb

TYPE_CODES = ["N2N", "N2AG", "N2B", "SBS"]
MUTUAL_TYPES = ("N2N", "SBS")      # no active/passive direction
DIRECTIONAL_TYPES = ("N2AG", "N2B")
_DURATION_KEYS = ("duration_sec", "video_duration_sec", "session_duration_sec", "t_end")


def _slot(name: str) -> int:
    return int(str(name)[1:]) - 1


def mouse_hex(name: str) -> str:
    r, g, b = slot_color_rgb(_slot(name))
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def _pts(a, nd: int = 3) -> list:
    return [[round(float(x), nd), round(float(y), nd)] for x, y in np.asarray(a)]


def template_geometry() -> dict:
    """Silhouette + extents of template.py, as plain lists (t, s coordinates)."""
    regions = {n: [_pts(v) for v in T.REGION_POLYS[n]] for n in
               ("nose", "head", "flank_R", "mid_back", "flank_L", "rump", "anogenital")}
    return {
        "body": _pts(T.body_outline()), "tail": _pts(T.tail_outline()),
        "ears": [_pts(T.REGION_POLYS["ear_R"][0]), _pts(T.REGION_POLYS["ear_L"][0])],
        "regions": regions,
        "eyes": [[round(T.U2T * 0.075, 3), 0.026], [round(T.U2T * 0.075, 3), -0.026]],
        "tmin": T.XLIM[0], "tmax": T.XLIM[1], "smin": T.YLIM[0], "smax": T.YLIM[1], "aspect": T.ASPECT,
    }


def ts_to_px(t: float, s: float, width: float) -> tuple:
    """(t, s) -> (x, y) pixels on a plot ``width`` wide; mirrors the JS mapping (y grows downward)."""
    sx = width / (T.XLIM[1] - T.XLIM[0])
    height = width * T.ASPECT * (T.YLIM[1] - T.YLIM[0]) / (T.XLIM[1] - T.XLIM[0])
    return (t - T.XLIM[0]) * sx, (T.YLIM[1] - s) / (T.YLIM[1] - T.YLIM[0]) * height


def session_end(run_dir, records: Sequence[dict]) -> float:
    """Session length in seconds: provenance.json if it has a duration, else the last contact."""
    last = max((r["time_sec"] + r.get("dur_sec", 0.0) for r in records), default=0.0)
    if run_dir:
        try:
            prov = json.loads((Path(run_dir) / "provenance.json").read_text(encoding="utf-8"))
            for k in _DURATION_KEYS:
                if isinstance(prov.get(k), (int, float)) and prov[k] > 0:
                    return round(max(float(prov[k]), last), 3)
        except (OSError, ValueError):
            pass
    return round(last, 3)


def build_payload(records: Sequence[dict], mice: Sequence[str], run_dir=None) -> dict:
    """Compact columnar payload; empty arrays when there are no records."""
    mice = list(mice)
    idx = {m: i for i, m in enumerate(mice)}
    rows = sorted((r for r in records if r["contact_type"] in TYPE_CODES
                   and r["focal_mouse"] in idx and r["other_mouse"] in idx),
                  key=lambda r: (r["time_sec"], r["frame_idx"]))
    reg_names = list(T.REGION_ORDER)
    return {
        "t_end": session_end(run_dir, rows),
        "mice": mice, "colors": [mouse_hex(m) for m in mice],
        "types": TYPE_CODES, "mutual": [TYPE_CODES.index(x) for x in MUTUAL_TYPES], "regions": reg_names,
        "records": {
            "time": [round(r["time_sec"], 3) for r in rows],
            "focal": [idx[r["focal_mouse"]] for r in rows],
            "other": [idx[r["other_mouse"]] for r in rows],
            "type": [TYPE_CODES.index(r["contact_type"]) for r in rows],
            "region": [reg_names.index(r["region"]) if r["region"] in reg_names else -1 for r in rows],
            "tt": [round(r["t"], 3) for r in rows],
            "ss": [round(r["s"], 3) for r in rows],
            "dur": [round(r["dur_sec"], 3) for r in rows],
        },
        "template": template_geometry(),
    }


def payload_script(payload: dict) -> str:
    """The inline ``<script>`` that defines BM_DATA (safe against a stray ``</script>``)."""
    txt = json.dumps(payload, separators=(",", ":"))
    return "<script>const BM_DATA=" + txt.replace("</", "<\\/") + ";</script>"


def widget(focal: str = "", view: str = "all") -> str:
    """Placeholder the JS fills in (lazily, when its tab is first shown)."""
    return (f'<div class="bmp" tabindex="0" data-focal="{focal}" data-view="{view}" '
            'aria-label="Interactive bodily map with timeline"><p class="empty">Loading interactive map...</p></div>')


BM_CSS = r"""
.bmp{outline:none}.bmp:focus-visible{box-shadow:0 0 0 2px var(--accent);border-radius:10px}
.bmp-bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;margin-bottom:10px}
.bmp-bar label{display:inline-flex;align-items:center;gap:6px;font-size:var(--step--1);color:var(--ink-2);font-weight:600}
.bmp select{font:inherit;font-size:var(--step--1);padding:5px 8px;border-radius:7px;border:1px solid var(--border-firm);background:var(--surface);color:var(--ink)}
.bmp .btn.play{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);min-width:84px}
.bmp-time{font-family:var(--mono);font-size:var(--step-1);font-weight:600;color:var(--accent);min-width:5.5em}
.bmp-slider{width:100%;accent-color:var(--accent);margin:0 0 10px}
.bmp-plot{background:#fff;border:1px solid var(--border);border-radius:10px;padding:6px;position:relative}
.bmp-plot canvas{display:block;width:100%}
.bmp-tl{margin-top:10px}
.bmp-tip{position:absolute;pointer-events:none;background:#14152A;color:#fff;font-size:12px;padding:4px 8px;border-radius:6px;white-space:nowrap;display:none;z-index:5}
.bmp-foot{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;margin-top:10px}
.bmp-foot .legend{margin:0}
@media print{.bmp-bar,.bmp-slider,.bmp-tip{display:none!important}.bmp:not(.ready){display:none}}
"""

BM_JS = (Path(__file__).with_name("bm_player.js")).read_text(encoding="utf-8")
