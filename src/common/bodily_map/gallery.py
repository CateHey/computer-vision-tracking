"""Self-contained, offline HTML gallery (inline CSS/JS, no external dependencies).

The visual language (tokens, type scale, cards, tables, pills) is taken from
``scripts/report_viewer.html`` so the two pages read as siblings. Web-font links are
intentionally NOT copied: the font stacks fall back to system fonts.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Sequence

from .findings import key_findings, mouse_stats
from .interactive import BM_CSS, BM_JS, build_payload, payload_script, widget
from .locate import slot_color_rgb
from .render import (MARKER_NAMES, MARKER_SHAPES, mouse_slot, region_table, select_active,
                     select_active_combined, select_passive)
from .summary import PROVISIONAL, SUMMARY_FILES, active_passive, mice_of
from .template import REGION_ORDER

FEW = 20  # fewer contact frames than this: flagged as low-evidence

_DARK = """--ground:#0A0C17;--surface:#141829;--surface-2:#1C2138;--border:#272C48;--border-firm:#3A4068;
--ink:#ECEEF8;--ink-2:#ADB3D4;--ink-3:#7C83AA;--accent:#9B93FF;--accent-ink:#0A0C17;--accent-soft:#201E42;
--accent-2:#34E3E3;--accent-3:#FF74D4;--warn:#E0B062;--warn-soft:#2C2417;
--shadow:0 1px 2px rgba(0,0,0,.5),0 12px 34px -16px rgba(0,0,0,.75);--grad:linear-gradient(90deg,#9B93FF,#B79BFF,#FF74D4);"""

CSS = """
/* Vivid neuroscience palette (light, OS-dark guarded by data-theme, explicit dark). */
:root{color-scheme:light;--ground:#F4F5FB;--surface:#FFFFFF;--surface-2:#EDEEF8;--border:#E0E2F0;--border-firm:#C7CBE4;
--ink:#14152A;--ink-2:#40456A;--ink-3:#686F96;--accent:#5B4BF0;--accent-ink:#FFFFFF;--accent-soft:#EBE9FE;
--accent-2:#0FB6C4;--accent-3:#E0489B;--warn:#9A5B00;--warn-soft:#FDF1DD;
--shadow:0 1px 2px rgba(20,21,42,.06),0 10px 30px -14px rgba(60,40,200,.22);--grad:linear-gradient(90deg,#5B4BF0,#8B5CF6,#E0489B);
--step--1:.75rem;--step-0:.875rem;--step-1:1rem;--step-2:1.375rem;--step-3:1.875rem;--step-4:2.5rem;
--sans:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;%DARK%}}
:root[data-theme="dark"]{color-scheme:dark;%DARK%}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:var(--step-0);line-height:1.5;-webkit-font-smoothing:antialiased}
h1,h2,h3{margin:0;text-wrap:balance;letter-spacing:-.015em;font-weight:600}
h1{font-size:var(--step-2)}h2{font-size:var(--step-1)}h3{font-size:var(--step-0)}
p{margin:0}
a{color:var(--accent)}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.eyebrow{font-size:var(--step--1);font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--accent)}
:where(button,[tabindex],summary):focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
.topbar{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--ground) 88%,transparent);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-bottom:1px solid var(--border)}
.topin{max-width:1240px;margin:0 auto;padding:12px 20px 0;display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px}
.mark{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.01em;font-size:var(--step-1)}
.mark .ttl{color:var(--ink);background:var(--grad);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent;font-weight:700}
.spacer{flex:1 1 auto}
.filechip{display:inline-flex;align-items:center;gap:8px;padding:5px 10px;background:var(--surface);border:1px solid var(--border);border-radius:999px;font-family:var(--mono);font-size:var(--step--1);color:var(--ink-2)}
.pill{display:inline-flex;align-items:center;padding:4px 11px;border-radius:999px;background:var(--warn-soft);border:1px solid var(--warn);color:var(--warn);font-size:var(--step--1);font-weight:600}
.btn{font:inherit;font-size:var(--step--1);font-weight:600;padding:6px 12px;border-radius:7px;border:1px solid var(--border-firm);background:var(--surface);color:var(--ink);cursor:pointer;transition:background .12s ease,border-color .12s ease}
.btn:hover{background:var(--surface-2);border-color:var(--ink-3)}
nav{max-width:1240px;margin:0 auto;padding:6px 12px 0;display:flex;gap:2px;overflow-x:auto;scrollbar-width:none}
nav::-webkit-scrollbar{display:none}
nav button{flex:none;border:0;border-bottom:3px solid transparent;background:none;color:var(--ink-3);font:inherit;font-weight:500;padding:9px 14px;cursor:pointer;display:flex;align-items:center;gap:8px;transition:color .12s ease,background .12s ease,border-color .12s ease;border-radius:6px 6px 0 0}
nav button:hover{color:var(--ink);background:var(--surface-2)}
nav button[aria-selected=true]{color:var(--ink);font-weight:600;border-bottom-color:var(--accent);background:var(--accent-soft)}
.wrap{max-width:1240px;margin:0 auto;padding:28px 20px 24px}
.tab{display:none}.tab.on{display:block}
#summary{position:relative;isolation:isolate}
#summary::before{content:"";position:absolute;z-index:-1;pointer-events:none;top:-28px;left:-20px;right:-20px;height:340px;background:radial-gradient(60% 100% at 18% 0%,color-mix(in srgb,var(--accent) 12%,transparent),transparent 70%),radial-gradient(45% 80% at 88% 0%,color-mix(in srgb,var(--accent-3) 8%,transparent),transparent 70%)}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;flex:none;box-shadow:0 0 0 1px rgba(0,0,0,.5),0 0 0 2px rgba(255,255,255,.4)}
.dot.lg{width:20px;height:20px}
.lead{color:var(--ink-2);font-size:var(--step-1);max-width:72ch;margin-bottom:8px}
.sec-head{display:flex;align-items:baseline;gap:12px;margin:36px 0 14px;flex-wrap:wrap}
.sec-head .note,.note{font-size:var(--step--1);color:var(--ink-3)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow);padding:20px;margin-bottom:20px}
.card h3{margin-bottom:10px}
.card p+p{margin-top:8px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.stat{padding:16px 18px;display:grid;gap:4px;align-content:start;margin:0}
.stat .k{display:flex;align-items:center;gap:8px;font-size:var(--step--1);color:var(--ink-3);font-weight:600}
.stat .who{margin-left:auto;color:var(--ink-2);font-family:var(--mono)}
.stat .v{color:var(--accent);font-family:var(--mono);font-size:var(--step-3);font-weight:600;letter-spacing:-.02em;line-height:1.2}
.stats .stat{border-top:3px solid var(--accent)}
.stat .sub{font-size:var(--step--1);color:var(--ink-2)}
.callout{background:var(--warn-soft);border:1px solid var(--warn);border-radius:12px;padding:18px 20px;margin:28px 0 20px;color:var(--ink)}
.callout h2{color:var(--warn);margin-bottom:8px}
.callout p+p,.callout details{margin-top:10px}
.callout summary{cursor:pointer;font-weight:600;color:var(--warn)}
.callout details p{margin-top:8px}
.fig{background:#fff;border:1px solid var(--border);border-radius:10px;padding:10px;display:block}
.fig img{display:block;width:100%;height:auto;cursor:zoom-in}
.cap{color:var(--ink-3);font-size:var(--step--1);margin-top:10px}
.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
.cols>.card{margin:0}
.howto{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:28px;align-items:start}
.howto ul{margin:8px 0;padding-left:20px}
.howto h2{margin-bottom:8px}
.legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px}
.lg{display:inline-flex;align-items:center;gap:5px;padding:3px 10px 3px 6px;border:1px solid var(--border);border-radius:999px;background:var(--surface);font-size:var(--step--1);font-weight:500;color:var(--ink-2)}
.mhead{display:flex;align-items:center;gap:14px;margin-bottom:6px}
.mhead h1{font-size:var(--step-4);letter-spacing:-.03em}
.chips{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0 24px}
.chip{background:var(--surface);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow);padding:12px 16px}
.chip .k{font-size:var(--step--1);color:var(--ink-3);font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.chip .v{font-family:var(--mono);font-size:var(--step-2);font-weight:600;display:flex;align-items:center;gap:8px;margin-top:2px}
.chip .s{font-size:var(--step--1);color:var(--ink-2)}
.fgrid{display:grid;grid-template-columns:minmax(0,3fr) minmax(240px,1fr);gap:20px;align-items:start}
.small{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
.small .card{margin:0}
.small .fgrid{grid-template-columns:minmax(0,1fr)}
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:var(--step--1)}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--border);font-family:var(--mono);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;font-family:var(--sans)}
th{font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-3);font-family:var(--sans)}
tbody tr:nth-child(even) td{background:var(--surface-2)}
tbody tr:last-child td{border-bottom:none}
tfoot th{color:var(--ink);border-top:1px solid var(--border-firm);border-bottom:none}
.warn{color:var(--warn);font-size:var(--step--1);margin-top:8px}
.empty{color:var(--ink-3);font-style:italic;padding:8px 0}
footer{max-width:1240px;margin:0 auto;padding:8px 20px 48px;color:var(--ink-3);font-size:var(--step--1)}
#lb{position:fixed;inset:0;z-index:100;background:rgba(8,12,11,.88);display:none;align-items:center;justify-content:center;padding:3vmin;cursor:zoom-out}
#lb.on{display:flex}
#lb img{max-width:100%;max-height:100%;background:#fff;border-radius:10px;padding:10px;box-shadow:0 10px 40px rgba(0,0,0,.5)}
@media(max-width:900px){.cols,.howto,.fgrid,.small{grid-template-columns:minmax(0,1fr)}.chips{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:520px){.wrap,.topin,footer{padding-left:16px;padding-right:16px}.mhead h1{font-size:var(--step-3)}.stats{grid-template-columns:minmax(0,1fr)}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
%BM%
@media print{.topbar,#lb{display:none!important}.tab{display:block!important}.tab+.tab{break-before:page}
.card,.chip,.callout{break-inside:avoid;box-shadow:none}body{background:#fff}details>*{display:block}}
""".replace("%DARK%", _DARK).replace("%BM%", BM_CSS)


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


def _dot(name: str, cls: str = "") -> str:
    return f'<span class="dot {cls}" style="background:{_rgb(name)}" title="{name}"></span>'


def _table(rows) -> str:
    tab = region_table(rows)
    tot = sum(v["dur_sec"] for v in tab.values()) or 1.0
    out = ['<div class="tw"><table><thead><tr><th>Region</th><th>Frames</th><th>Duration (s)</th><th>Share</th></tr></thead><tbody>']
    for reg in REGION_ORDER:
        v = tab.get(reg)
        if v and v["count"]:
            out.append(f"<tr><td>{reg}</td><td>{v['count']}</td><td>{v['dur_sec']:.2f}</td>"
                       f"<td>{100 * v['dur_sec'] / tot:.0f}%</td></tr>")
    n = sum(v["count"] for v in tab.values())
    out.append(f"</tbody><tfoot><tr><th>Total</th><th>{n}</th><th>{tot if n else 0:.2f}</th><th></th></tr></tfoot></table></div>")
    return "".join(out)


def _figure(out: Path, fname: str, rows, caption: str) -> str:
    cap = html.escape(caption)
    if not rows or not (out / fname).exists():
        return f'<div class="card"><h3>{cap}</h3><div class="empty">No contacts of this kind in the run.</div></div>'
    warn = f'<p class="warn">Only {len(rows)} contact frames: read with caution.</p>' if len(rows) < FEW else ""
    return (f'<div class="card"><h3>{cap}</h3><div class="fgrid"><span class="fig"><img src="{fname}" loading="lazy" '
            f'alt="{cap}"></span><div>{_table(rows)}{warn}</div></div></div>')


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
(function(){
var root=document.documentElement,tb=document.getElementById('theme');
function theme(t){if(t)root.setAttribute('data-theme',t);else root.removeAttribute('data-theme');
 if(tb)tb.textContent=t==='dark'?'Light mode':t==='light'?'Dark mode':'Theme: auto'}
var st=null;try{st=localStorage.getItem('bm-theme')}catch(e){}theme(st==='dark'||st==='light'?st:null);
if(tb)tb.addEventListener('click',function(){var c=root.getAttribute('data-theme');
 var dm=window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches;
 var n=c?(c==='dark'?'light':'dark'):(dm?'light':'dark');theme(n);try{localStorage.setItem('bm-theme',n)}catch(e){}});
var tabs=[].slice.call(document.querySelectorAll('nav [role=tab]'));
var panels=[].slice.call(document.querySelectorAll('.tab'));
function show(id,focus){
  if(panels.every(function(p){return p.id!==id}))id='summary';
  panels.forEach(function(p){p.classList.toggle('on',p.id===id)});
  if(window.bmInit)bmInit(id);
  tabs.forEach(function(b){var on=b.dataset.tab===id;b.setAttribute('aria-selected',on);b.tabIndex=on?0:-1;
    if(on){if(focus)b.focus();b.scrollIntoView&&b.scrollIntoView({block:'nearest',inline:'nearest'})}});
  window.scrollTo(0,0)}
function setHash(id){try{history.replaceState(null,'','#'+id)}catch(e){}}
tabs.forEach(function(b,i){
  b.addEventListener('click',function(){setHash(b.dataset.tab);show(b.dataset.tab)});
  b.addEventListener('keydown',function(e){var d=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;if(!d)return;e.preventDefault();
    var n=tabs[(i+d+tabs.length)%tabs.length];setHash(n.dataset.tab);show(n.dataset.tab,true)})});
window.addEventListener('hashchange',function(){show(location.hash.slice(1))});
show(location.hash.slice(1));
var lb=document.getElementById('lb'),li=lb.querySelector('img');
document.addEventListener('click',function(e){var t=e.target;
  if(t.tagName==='IMG'&&t.closest('.fig')){li.src=t.currentSrc||t.src;li.alt=t.alt;lb.classList.add('on')}
  else if(lb.classList.contains('on'))lb.classList.remove('on')});
document.addEventListener('keydown',function(e){if(e.key==='Escape')lb.classList.remove('on')});
})();
"""


def _ap_table(records, mice) -> str:
    ap = active_passive(records, mice)
    out = ['<div class="tw"><table><thead><tr><th>Mouse</th><th>Active (s)</th><th>Passive (s)</th><th>% initiated (provisional)</th>'
           '<th>Mutual N2N+SBS (s)</th></tr></thead><tbody>']
    for m in mice:
        v = ap[m]
        p = "n/a" if v["pct_initiated"] is None else f"{v['pct_initiated']:.0f}%"
        out.append(f"<tr><td>{_dot(m)} {m}</td><td>{v['active']:.1f}</td><td>{v['passive']:.1f}</td><td>{p}</td>"
                   f"<td>{v['mutual']:.1f}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


HOWTO = """<h2>How to read these maps</h2>
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


def _findings_html(records, mice) -> str:
    cards = []
    for f in key_findings(records, mice):
        dots = "".join(_dot(m) for m in f["mouse"].split("+"))
        who = f["mouse"].replace("+", " + ")
        cards.append(f'<div class="card stat"><div class="k">{dots}<span>{html.escape(f["label"])}</span>'
                     f'<span class="who">{who}</span></div><div class="v">{html.escape(f["number"])}</div>'
                     f'<div class="sub">{html.escape(f["text"])}</div></div>')
    return '<div class="stats">' + "".join(cards) + "</div>"


def _live_card(title: str, sub: str, focal: str, view: str) -> str:
    return (f'<div class="sec-head"><h2>{html.escape(title)}</h2><span class="note">{html.escape(sub)}</span></div>'
            f'<div class="card">{widget(focal, view)}</div>')


def _summary_tab(out: Path, records, mice, legend) -> str:
    key = ('<span class="fig"><img src="template_key.png" alt="Template mouse key"></span>'
           if (out / "template_key.png").exists() else "")
    p = ['<section class="tab" role="tabpanel" id="summary">',
         '<p class="lead">Where on the body mice touch each other, who initiates, and who is investigated. '
         'All figures are duration weighted; following is excluded.</p>',
         '<div class="sec-head"><h2>Key findings</h2><span class="note">computed from the located contacts of this run</span></div>',
         _findings_html(records, mice),
         _live_card('Contacts over time', 'Press play: contacts appear on the template mouse as they happen, and the timeline below '
                    'shows when. Drag the slider or click the timeline to jump to any moment.', '', 'all'),
         '<div class="callout"><h2>How we tell active from passive, and when we cannot</h2>'
         f'<p>{html.escape(EXPLAINER)}</p><details><summary>Caveats (provisional)</summary>'
         f'<p>{html.escape(CAVEATS)}</p></details></div>',
         f'<div class="card howto"><div>{HOWTO}</div><div>{key}<div class="legend">{legend}</div></div></div>',
         '<div class="card"><h3>Active vs passive numbers (directional contacts only)</h3>' + _ap_table(records, mice)
         + f'<p class="note" style="margin-top:10px">Active/passive uses nose-to-anogenital and nose-to-body only. % initiated = '
           f'active / (active + passive); {html.escape(PROVISIONAL)}. Mutual contacts (N2N + SBS) are never split into active or passive.</p></div>',
         '<div class="sec-head"><h2>Summary figures</h2></div><div class="cols" style="grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr))">']
    for f in SUMMARY_FILES:
        if (out / f).exists():
            p.append(f'<div class="card"><h3>{html.escape(f[8:-4].replace("_", " ").capitalize())}</h3>'
                     f'<span class="fig"><img src="{f}" loading="lazy" alt="{html.escape(CAPS[f])}"></span>'
                     f'<p class="cap">{html.escape(CAPS[f])}</p></div>')
    p.append("</div></section>")
    return "".join(p)


def _mouse_tab(out: Path, records, mice, m: str, st) -> str:
    pas, comb = select_passive(records, m), select_active_combined(records, m)
    s = st[m]
    pct = "n/a" if s["pct_initiated"] is None else f'{s["pct_initiated"]:.0f}%'
    region = html.escape(s["top_region"].replace("_", " ")) if s["top_region"] else "n/a"
    partner = f'{_dot(s["top_partner"])}{s["top_partner"]}' if s["top_partner"] else "n/a"
    chips = [("Total contact", f'{s["total"]:.0f} s', "touching + touched"),
             ("% initiated", pct, "provisional"),
             ("Most-touched region", region, f'{s["top_region_share"]:.0f}% of time touched' if s["top_region"] else ""),
             ("Top partner", partner, f'{s["top_partner_sec"]:.0f} s together' if s["top_partner"] else "")]
    chips_html = "".join(f'<div class="chip"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{sub}</div></div>'
                         for k, v, sub in chips)
    sec = [f'<section class="tab" role="tabpanel" id="{m}"><div class="mhead">{_dot(m, "lg")}<h1>{m}</h1></div>'
           f'<p class="note">Touched in {len(pas)} frames, touching in {len(comb)} frames.</p>'
           f'<div class="chips">{chips_html}</div>',
           _live_card(f'{m} over time', f'Where {m} is touched (or touches) as the session unfolds.', m, 'all'),
           _figure(out, f"passive_{m}.png", pas, f"Passive map of {m}: where others touch {m}"),
           _figure(out, f"active_{m}_combined.png", comb,
                   f"Active map of {m}: where {m} touches others (all partners pooled)"),
           '<div class="small">']
    for b in mice:
        if b != m:
            sec.append(_figure(out, f"active_{m}_to_{b}.png", select_active(records, m, b),
                               f"Active map {m} → {b}: where {m} touches {b}"))
    sec.append("</div></section>")
    return "".join(sec)


def build_gallery(run_dir, records: Sequence[dict], out_dir, mice: Sequence[str] = None) -> Path:
    """Write a tabbed, offline ``index.html`` in ``out_dir`` referencing the sibling PNGs. Returns its path."""
    out = Path(out_dir)
    run_name = Path(run_dir).name
    mice = sorted(mice or mice_of(records), key=mouse_slot)
    m_ = re.match(r"(\d{4}-\d{2}-\d{2})", run_name)
    span = max((r["time_sec"] for r in records), default=0) / 60.0
    meta = " · ".join([f"{len(mice)} mice"] + ([f"session {m_.group(1)}"] if m_ else [])
                          + ([f"{span:.0f} min observed"] if span >= 1 else []))
    legend = "".join(f'<span class="lg">{_shape_svg(m)}{m} ({MARKER_NAMES[mouse_slot(m) % 6]})</span>' for m in mice)
    st = mouse_stats(records, mice)

    def tab(id_, dot, label):
        return (f'<button role="tab" id="t-{id_}" data-tab="{id_}" aria-controls="{id_}" aria-selected="false" tabindex="-1">'
                f'{dot}{label}</button>')
    nav = tab("summary", "", "Summary") + "".join(tab(m, _dot(m), m) for m in mice)
    mark = ('<svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">'
            '<circle cx="6" cy="10" r="4.5" fill="#0f6e63"/><circle cx="14" cy="10" r="4.5" fill="#4fd1bd" fill-opacity=".8"/></svg>')
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bodily Maps of Social Contact</title><style>{CSS}</style></head><body>
<div class="topbar"><div class="topin"><span class="mark">{mark}<span class="ttl">Bodily Maps of Social Contact</span></span>
<span class="spacer"></span><span class="filechip">{html.escape(meta)}</span><span class="pill">provisional</span>
<button id="theme" class="btn" type="button">Theme: auto</button></div>
<nav role="tablist" aria-label="Summary and mice">{nav}</nav></div><div class="wrap">""", _summary_tab(out, records, mice, legend)]
    parts += [_mouse_tab(out, records, mice, m, st) for m in mice]
    parts.append(f"""</div><footer>Generated by scripts/bodily_map.py · provisional research output · {html.escape(run_name)}</footer>
<div id="lb" aria-hidden="true"><img alt=""></div>{payload_script(build_payload(records, mice, run_dir))}<script>{BM_JS}</script><script>{JS}</script></body></html>""")
    path = out / "index.html"
    path.write_text("".join(parts), encoding="utf-8")
    return path
