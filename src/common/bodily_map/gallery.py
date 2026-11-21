"""Self-contained HTML gallery (inline CSS, no external dependencies)."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Sequence

from .locate import slot_color_rgb
from .render import (MARKER_NAMES, MARKER_SHAPES, mouse_slot, region_table, select_active,
                     select_active_combined, select_passive)
from .template import REGION_ORDER

FEW = 20  # fewer contact frames than this: flagged as low-evidence

CSS = """
:root{--ink:#1f2933;--mute:#5f6b76;--line:#dde2e7;--bg:#f6f7f9;--card:#fff;--acc:#4a2a6a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header{background:var(--acc);color:#fff;padding:28px 6vw}
header h1{margin:0 0 6px;font-size:26px;font-weight:600}
header p{margin:0;opacity:.85}
main{max-width:1500px;margin:0 auto;padding:24px 4vw 60px}
nav{position:sticky;top:0;background:rgba(246,247,249,.95);padding:10px 0;border-bottom:1px solid var(--line);z-index:5}
nav a{display:inline-block;margin-right:14px;color:var(--acc);text-decoration:none;font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 22px;margin:20px 0}
.card h2{margin:0 0 4px;font-size:21px}
.card h3{margin:22px 0 8px;font-size:16px;color:var(--acc)}
.sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px;border:1px solid #222}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(220px,1fr);gap:18px;align-items:start}
.grid img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--mute);font-weight:600}
.note{color:var(--mute);font-size:13px}
.warn{color:#9a4d00;font-size:13px}
.legend span{display:inline-block;margin-right:18px;white-space:nowrap}
.legend svg{vertical-align:middle;margin-right:4px}
.empty{color:var(--mute);font-style:italic;padding:8px 0}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
@media print{nav{display:none}.card{break-inside:avoid}}
"""


def _rgb(name: str) -> str:
    r, g, b = slot_color_rgb(mouse_slot(name))
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def _shape_svg(name: str) -> str:
    c, sh = _rgb(name), MARKER_SHAPES[mouse_slot(name) % 6]
    st = f'fill="{c}" stroke="#222" stroke-width="1"'
    body = {
        "o": f'<circle cx="9" cy="9" r="5.5" {st}/>',
        "s": f'<rect x="3.5" y="3.5" width="11" height="11" {st}/>',
        "^": f'<polygon points="9,3 15,14 3,14" {st}/>',
        "D": f'<polygon points="9,2.5 15.5,9 9,15.5 2.5,9" {st}/>',
        "*": f'<polygon points="9,1.5 11,7 16.5,7 12,10.5 13.8,16 9,12.6 4.2,16 6,10.5 1.5,7 7,7" {st}/>',
        "P": f'<polygon points="6.5,2.5 11.5,2.5 11.5,6.5 15.5,6.5 15.5,11.5 11.5,11.5 11.5,15.5 6.5,15.5 6.5,11.5 2.5,11.5 2.5,6.5 6.5,6.5" {st}/>',
    }[sh]
    return f'<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">{body}</svg>'


def _table(rows) -> str:
    tab = region_table(rows)
    tot = sum(v["dur_sec"] for v in tab.values()) or 1.0
    out = ['<table><tr><th>Region</th><th>Frames</th><th>Duration (s)</th><th>Share</th></tr>']
    for reg in REGION_ORDER:
        v = tab.get(reg)
        if v and v["count"]:
            out.append(f"<tr><td>{reg}</td><td>{v['count']}</td><td>{v['dur_sec']:.2f}</td>"
                       f"<td>{100 * v['dur_sec'] / tot:.0f}%</td></tr>")
    n = sum(v["count"] for v in tab.values())
    out.append(f"<tr><th>Total</th><th>{n}</th><th>{tot if n else 0:.2f}</th><th></th></tr></table>")
    return "".join(out)


def _figure(out: Path, fname: str, rows, caption: str) -> str:
    if not rows or not (out / fname).exists():
        return f'<h3>{html.escape(caption)}</h3><div class="empty">No contacts of this kind in the run.</div>'
    warn = f'<p class="warn">Only {len(rows)} contact frames: read with caution.</p>' if len(rows) < FEW else ""
    return (f'<h3>{html.escape(caption)}</h3><div class="grid"><a href="{fname}"><img src="{fname}" loading="lazy" '
            f'alt="{html.escape(caption)}"></a><div>{_table(rows)}{warn}</div></div>')


def _overview(records, mice) -> str:
    """Matrix of total duration of A (rows, toucher) touching B (columns, touched)."""
    head = "".join(f"<th>{b}</th>" for b in mice)
    body = []
    for a in mice:
        cells = []
        for b in mice:
            if a == b:
                cells.append("<td>–</td>")
            else:
                sel = select_active(records, a, b)
                cells.append(f"<td>{sum(r['dur_sec'] for r in sel):.1f}</td>")
        body.append(f"<tr><td>{a}</td>{''.join(cells)}</tr>")
    return (f"<table><tr><th>toucher ↓ / touched →</th>{head}</tr>{''.join(body)}</table>")


def build_gallery(run_dir, records: Sequence[dict], out_dir, mice: Sequence[str]) -> Path:
    """Write ``index.html`` in ``out_dir`` referencing the sibling PNGs. Returns its path."""
    out = Path(out_dir)
    run_name = Path(run_dir).name
    mice = sorted(mice, key=mouse_slot)
    total_dur = sum(r["dur_sec"] for r in records)
    legend = "".join(f'<span>{_shape_svg(m)}{m} ({MARKER_NAMES[mouse_slot(m) % 6]})</span>' for m in mice)
    nav = "".join(f'<a href="#{m}">{m}</a>' for m in mice)
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bodily maps of social contact — {html.escape(run_name)}</title><style>{CSS}</style></head><body>
<header><h1>Bodily maps of social contact — {html.escape(run_name)}</h1>
<p>{len(records)} located contact frames, {total_dur:.1f} s of contact, {len(mice)} mice ({", ".join(mice)})</p></header>
<main>
<div class="card"><h2>How to read these maps</h2>
<p>A bodily map shows <em>where on a mouse's body</em> social contacts land. Each contact is placed on the body of the
<strong>touched</strong> mouse, using its own pose (nose to tail tip), so contacts from different frames and positions can be pooled
on one template mouse. The template is drawn from above with the <strong>nose on the left and the tail on the right</strong>;
the mouse's right side is the top and its left side the bottom.</p>
<ul>
<li><strong>Passive map of A</strong>: where the other mice touch A.</li>
<li><strong>Active map of A (all partners)</strong>: where A touches others, pooled on one generic mouse.</li>
<li><strong>Active map A → B</strong>: where A's touches land on B's body.</li>
</ul>
<p>The colour scale is <strong>contact duration in seconds</strong> (smoothed, square-root colour scale), so a long sniff counts
more than a brief touch. Markers show individual contact frames: shape and colour identify the other mouse, and the shape
carries the identity for colour-blind readers. Each figure has a combined panel plus one panel per contact type
(nose-to-nose, nose-to-anogenital, nose-to-body, side-by-side), each with its own colour scale. Following-type contacts are excluded.
Nose-to-nose contacts dominate the total, so the per-type panels are the place to look for contacts elsewhere on the body.</p>
<div class="legend"><strong>Mice:</strong> {legend}</div></div>
<div class="card"><h2>Overview: duration of contact (s)</h2>{_overview(records, mice)}
<p class="note">Row mouse touches column mouse. Detail below.</p></div>
<nav>{nav}</nav>"""]
    for m in mice:
        pas = select_passive(records, m)
        comb = select_active_combined(records, m)
        sec = [f'<div class="card" id="{m}"><h2><span class="sw" style="background:{_rgb(m)}"></span>Mouse {m}</h2>'
               f'<p class="note">Touched {len(pas)} frames, touching {len(comb)} frames.</p>']
        sec.append(_figure(out, f"passive_{m}.png", pas, f"Passive map of {m}: where others touch {m}"))
        sec.append(_figure(out, f"active_{m}_combined.png", comb,
                           f"Active map of {m}: where {m} touches others (all partners pooled)"))
        for b in mice:
            if b != m:
                sec.append(_figure(out, f"active_{m}_to_{b}.png", select_active(records, m, b),
                                   f"Active map {m} → {b}: where {m} touches {b}"))
        sec.append("</div>")
        parts.append("".join(sec))
    parts.append("</main></body></html>")
    path = out / "index.html"
    path.write_text("".join(parts), encoding="utf-8")
    return path
