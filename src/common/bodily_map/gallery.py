"""Self-contained HTML gallery (inline CSS, no external dependencies)."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Sequence

from .locate import slot_color_rgb
from .render import (MARKER_NAMES, MARKER_SHAPES, mouse_slot, region_table, select_active,
                     select_active_combined, select_passive)
from .summary import PROVISIONAL, SUMMARY_FILES, active_passive, mice_of
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
nav{position:sticky;top:0;background:rgba(246,247,249,.97);padding:10px 4vw;border-bottom:1px solid var(--line);z-index:5;display:flex;flex-wrap:wrap;gap:8px}
nav button{border:1px solid var(--line);background:#fff;color:var(--acc);font:inherit;font-weight:600;padding:6px 16px;border-radius:18px;cursor:pointer}
nav button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.tab{display:none}.tab.on{display:block}
.sumimg{width:100%;max-width:1100px;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff;display:block;margin:8px 0}
.explain{border-left:4px solid #c27c0e;background:#fff8ec}
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
@media print{nav{display:none}.tab{display:block!important}.card{break-inside:avoid}}
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


EXPLAINER = (
    "For a contact where one mouse's nose investigates another's body, the investigator is the ACTIVE mouse and the one "
    "investigated is PASSIVE. The active mouse is decided by geometry: its nose or head is on the other's body or rear, it is "
    "facing the other, and there is evidence of real touch. This applies to nose-to-anogenital and nose-to-body contacts (and to "
    "following, which is not a bodily touch and is left out of the maps). Nose-to-nose and side-by-side are MUTUAL: both mice "
    "investigate each other, so no active/passive direction is assigned and they are reported separately.")
CAVEATS = (
    "The direction is inferred per frame from pose, which is least reliable at the moment of contact, then decided per contact "
    "by majority vote; when both mice sniff at once the higher score wins, which can be close. The active/passive accuracy has "
    "not yet been scored by a blinded human rater, so the active-direction numbers are provisional.")

JS = """
function show(id){document.querySelectorAll('.tab').forEach(function(e){e.classList.toggle('on',e.id===id)});
document.querySelectorAll('nav button').forEach(function(b){b.classList.toggle('on',b.dataset.tab===id)});
try{history.replaceState(null,'','#'+id)}catch(e){}window.scrollTo(0,0)}
document.querySelectorAll('nav button').forEach(function(b){b.addEventListener('click',function(){show(b.dataset.tab)})});
var h=location.hash.replace('#','');show(document.getElementById(h)&&document.getElementById(h).classList.contains('tab')?h:'summary');
"""


def _ap_table(records, mice) -> str:
    ap = active_passive(records, mice)
    out = ['<table><tr><th>Mouse</th><th>Active (s)</th><th>Passive (s)</th><th>% initiated (provisional)</th>'
           '<th>Mutual N2N+SBS (s)</th></tr>']
    for m in mice:
        v = ap[m]
        p = "n/a" if v["pct_initiated"] is None else f"{v['pct_initiated']:.0f}%"
        out.append(f"<tr><td>{m}</td><td>{v['active']:.1f}</td><td>{v['passive']:.1f}</td><td>{p}</td>"
                   f"<td>{v['mutual']:.1f}</td></tr>")
    out.append("</table>")
    return "".join(out)


HOWTO = """<div class="card"><h2>How to read these maps</h2>
<p>A bodily map shows <em>where on a mouse's body</em> social contacts land. Each contact is placed on the body of the
<strong>touched</strong> mouse, using its own pose (nose to tail tip), so contacts from different frames can be pooled on one
template mouse, drawn from above with the <strong>nose on the left and the tail on the right</strong> (the mouse's right side is the top).</p>
<ul>
<li><strong>Passive map of A</strong>: where the other mice touch A.</li>
<li><strong>Active map of A (all partners)</strong>: where A touches others, pooled on one generic mouse.</li>
<li><strong>Active map A &rarr; B</strong>: where A's touches land on B's body.</li>
</ul>
<p>The colour scale is <strong>contact duration in seconds</strong> (smoothed, square-root scale), so a long sniff counts more than a
brief touch. The large combined panel is a clean heatmap with no markers. The four per-type panels (nose-to-nose, nose-to-anogenital,
nose-to-body, side-by-side) also show individual contact frames as markers: shape and colour identify the other mouse. Each panel has
its own colour scale. Following is excluded. Nose-to-nose dominates the total, so the per-type panels are the place to look for contacts
elsewhere on the body.</p>
<p>Region names are not printed on the maps; see the template key (dashed lines mark the same regions).</p>"""

CAPS = {
    "summary_contacts_per_mouse.png": "Who is most social: total contact duration and frame count per mouse, by contact type.",
    "summary_active_passive.png": "Active vs passive duration per mouse, directional contacts only. % initiated is "
                                  + PROVISIONAL + ".",
    "summary_region_preference.png": "Where each mouse is touched: share of its touched duration per body region.",
    "summary_contact_matrix.png": "Who touches whom: duration (s), row = toucher, column = touched.",
}


def _summary_tab(out: Path, records, mice, legend) -> str:
    key = ('<img src="template_key.png" alt="Template mouse key" style="max-width:700px;width:100%">'
           if (out / "template_key.png").exists() else "")
    p = ['<div class="tab" id="summary">', HOWTO, key, f'<div class="legend"><strong>Mice:</strong> {legend}</div></div>',
         f'<div class="card explain"><h2>How we tell active from passive, and when we cannot</h2><p>{html.escape(EXPLAINER)}</p>'
         f'<p><strong>Caveats.</strong> {html.escape(CAVEATS)}</p></div>',
         '<div class="card"><h2>Active vs passive numbers (directional contacts only)</h2>' + _ap_table(records, mice)
         + f'<p class="note">Active/passive uses nose-to-anogenital and nose-to-body only. % initiated = active / (active + passive); '
           f'{html.escape(PROVISIONAL)}. Mutual contacts (N2N + SBS) are never split into active or passive.</p></div>']
    for f in SUMMARY_FILES:
        if (out / f).exists():
            p.append(f'<div class="card"><h2>{html.escape(f[8:-4].replace("_", " ").capitalize())}</h2>'
                     f'<a href="{f}"><img class="sumimg" src="{f}" alt="{html.escape(CAPS[f])}"></a>'
                     f'<p class="note">{html.escape(CAPS[f])}</p></div>')
    p.append("</div>")
    return "".join(p)


def build_gallery(run_dir, records: Sequence[dict], out_dir, mice: Sequence[str] = None) -> Path:
    """Write a tabbed ``index.html`` in ``out_dir`` referencing the sibling PNGs. Returns its path."""
    out = Path(out_dir)
    run_name = Path(run_dir).name
    mice = sorted(mice or mice_of(records), key=mouse_slot)
    total_dur = sum(r["dur_sec"] for r in records)
    legend = "".join(f'<span>{_shape_svg(m)}{m} ({MARKER_NAMES[mouse_slot(m) % 6]})</span>' for m in mice)
    nav = '<button data-tab="summary" class="on">Summary</button>' + "".join(
        f'<button data-tab="{m}">{m}</button>' for m in mice)
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bodily maps of social contact — {html.escape(run_name)}</title><style>{CSS}</style></head><body>
<header><h1>Bodily maps of social contact — {html.escape(run_name)}</h1>
<p>{len(records)} located contact frames, {total_dur:.1f} s of contact, {len(mice)} mice ({", ".join(mice)})</p></header>
<nav>{nav}</nav><main>""", _summary_tab(out, records, mice, legend)]
    for m in mice:
        pas = select_passive(records, m)
        comb = select_active_combined(records, m)
        sec = [f'<div class="tab card" id="{m}"><h2><span class="sw" style="background:{_rgb(m)}"></span>Mouse {m}</h2>'
               f'<p class="note">Touched in {len(pas)} frames, touching in {len(comb)} frames.</p>']
        sec.append(_figure(out, f"passive_{m}.png", pas, f"Passive map of {m}: where others touch {m}"))
        sec.append(_figure(out, f"active_{m}_combined.png", comb,
                           f"Active map of {m}: where {m} touches others (all partners pooled)"))
        for b in mice:
            if b != m:
                sec.append(_figure(out, f"active_{m}_to_{b}.png", select_active(records, m, b),
                                   f"Active map {m} → {b}: where {m} touches {b}"))
        sec.append("</div>")
        parts.append("".join(sec))
    parts.append(f"</main><script>{JS}</script></body></html>")
    path = out / "index.html"
    path.write_text("".join(parts), encoding="utf-8")
    return path
