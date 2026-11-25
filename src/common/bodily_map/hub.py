"""Unified, offline, per-run report hub (``<run>/index.html``).

One self-contained page (inline CSS/JS, no network) with four tabs: Overview, Bodily Map (the
existing gallery in an iframe), Session Report (rendered natively from ``contacts/results.xlsx``)
and Validation (summary plus links to the clip viewers). Only READS existing run outputs.
Dependencies: stdlib + openpyxl (numpy / matplotlib are not needed here).
"""
from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .findings import key_findings
from .gallery import CSS as GALLERY_CSS  # exact blue + purple tokens, shared with the gallery

THEME_KEY = "bm-theme"  # same localStorage key as gallery.py
TYPE_COLORS = {"N2N": "#1f77b4", "N2AG": "#ff7f0e", "N2B": "#2ca02c", "FOL": "#d62728",
               "SBS": "#9467bd", "NC": "#b6bdba"}  # categorical, same as scripts/report_viewer.html
FIGURES = [("events_by_type.png", "Events by contact type", "Number of cleaned events per contact type."),
           ("duration_by_type.png", "Time by contact type", "Total and mean duration per contact type."),
           ("event_duration_distribution.png", "Event duration distribution",
            "How long events last; very short events are the most likely to be flicker."),
           ("timeline_comparison.png", "Timeline comparison", "Raw labels against cleaned events over the session.")]
EVENT_COLS = [("event_id", "#"), ("pair_label", "Pair"), ("contact_type", "Type"), ("subtype", "Subtype"),
              ("start_time", "Start"), ("end_time", "End"), ("duration_sec", "Duration (s)"),
              ("duration_frames", "Frames"), ("investigator_slot", "Investigator"),
              ("peak_score", "Peak score"), ("mean_nose_nose_dist_bl", "Nose-nose (BL)")]
VERDICTS = ["real", "flicker", "wrong_type", "tracking_error", "unsure", "nc_correct", "missed_contact"]


# ---------------------------------------------------------------- readers
def _clean(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)


def read_sheet(xlsx: Optional[Path], name: str) -> List[dict]:
    """Rows of one sheet as dicts (header row = keys). Missing file or sheet gives ``[]``."""
    if not xlsx or not Path(xlsx).exists():
        return []
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        if name not in wb.sheetnames:
            return []
        it = wb[name].iter_rows(values_only=True)
        head = next(it, None)
        if not head:
            return []
        keys = [str(h) if h is not None else f"col{i}" for i, h in enumerate(head)]
        return [{k: _clean(v) for k, v in zip(keys, row)} for row in it if any(c is not None for c in row)]
    finally:
        wb.close()


def read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_config_flat(path: Path, keys=("video_path", "device")) -> dict:
    """Tiny YAML reader (no PyYAML dependency): top-level scalars plus selected ``a.b`` scalars."""
    out: dict = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    stack: List[tuple] = []
    for ln in lines:
        m = re.match(r"^(\s*)([A-Za-z0-9_]+):\s*(.*)$", ln)
        if not m:
            continue
        ind, k, v = len(m.group(1)), m.group(2), m.group(3).strip()
        while stack and stack[-1][0] >= ind:
            stack.pop()
        full = ".".join([s[1] for s in stack] + [k])
        if v:
            out[full] = v.strip("'\"")
        else:
            stack.append((ind, k))
    return out


def read_validation(run_dir: Path) -> dict:
    """Summary of ``validation/``: manifest, review verdict counts and which viewers exist."""
    vd = Path(run_dir) / "validation"
    res = dict(present=vd.is_dir(), manifest={}, verdicts={}, n_rows=0, n_reviewed=0,
               has_clips=(vd / "index.html").exists(), has_identity=(vd / "identity" / "identity.html").exists())
    if not vd.is_dir():
        return res
    res["manifest"] = read_json(vd / "manifest.json")
    rows = read_sheet(vd / "review.xlsx", "review")
    res["n_rows"] = len(rows)
    c = Counter(str(r.get("verdict")).strip() for r in rows if r.get("verdict") not in (None, ""))
    res["verdicts"] = dict(c)
    res["n_reviewed"] = sum(c.values())
    return res


def _load_records(run_dir: Path) -> List[dict]:
    """Located-contact records for the key findings: the gallery's CSV if present, else recomputed."""
    csvs = sorted(Path(run_dir).rglob("contacts_located.csv"))
    if csvs:
        out = []
        with open(csvs[0], newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for k in ("frame_idx",):
                    r[k] = int(float(r[k]))
                for k in ("time_sec", "t", "s", "dur_sec"):
                    r[k] = float(r[k])
                out.append(r)
        return out
    try:
        from .locate import find_run_files, load_perframe, load_tracks, locate_contacts
        tracks_p, xlsx_p = find_run_files(run_dir)
        meta, frames = load_tracks(tracks_p)
        return locate_contacts(meta, frames, load_perframe(xlsx_p))[0]
    except Exception:  # noqa: BLE001 - findings are optional
        return []


# ---------------------------------------------------------------- html bits
def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _fmt_sec(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _e(v)
    return f"{int(v // 60)} min {v % 60:.0f} s" if v >= 60 else f"{v:.1f} s"


def _stat(k, v, sub="", color=None) -> str:
    st = f' style="border-top-color:{color}"' if color else ""
    return f'<div class="card stat"{st}><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div><div class="sub">{_e(sub)}</div></div>'


def _pairs_table(rows: List[dict], cols) -> str:
    h = "".join(f"<th>{_e(c)}</th>" for c in cols)
    b = "".join("<tr>" + "".join(f"<td>{_e(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def _summary_stats(summary: List[dict], glob: Dict[str, object]) -> str:
    cards = []
    dur = glob.get("total_duration_sec")
    if dur is not None:
        cards.append(_stat("Session length", _fmt_sec(dur), f"{glob.get('total_frames', '?')} frames at {glob.get('fps', '?')} fps"))
    for r in summary:
        t = r.get("contact_type")
        if t == "TOTAL":
            cards.append(_stat("Total events", r.get("events"), f"{r.get('total_sec')} s in contact"))
        elif t == "ANY":
            cards.append(_stat("Any contact", f"{r.get('pct_of_session')}%", f"{r.get('total_sec')} s, union over pairs"))
        elif t:
            cards.append(_stat(r.get("contact_label") or t, r.get("events"),
                               f"{r.get('total_sec')} s, mean {r.get('mean_duration_sec')} s, {r.get('pct_of_session')}% of session",
                               TYPE_COLORS.get(str(t))))
    if glob.get("n_pairs") is not None:
        cards.append(_stat("Pairs", glob.get("n_pairs"), "animal pairs tracked"))
    if glob.get("flicker_rate_real_per_1000") is not None:
        cards.append(_stat("Flicker rate", glob.get("flicker_rate_real_per_1000"),
                           f"per 1000 frames after cleaning (raw {glob.get('flicker_rate_raw_per_1000')})"))
    return f'<div class="stats">{"".join(cards)}</div>' if cards else '<p class="empty">No Summary / Global sheet in results.xlsx.</p>'


EXTRA_CSS = """
.hub-note{margin-top:6px}
.jump{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:8px 0 20px}
.jump button{all:unset;box-sizing:border-box;cursor:pointer;display:block;background:var(--surface);border:1px solid var(--border);
border-radius:12px;box-shadow:var(--shadow);padding:18px;border-top:3px solid var(--accent-2);transition:border-color .12s,transform .12s}
.jump button:hover{border-color:var(--accent);transform:translateY(-2px)}
.jump button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.jump b{display:block;font-size:var(--step-1);margin-bottom:4px;background:var(--grad);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.jump span{color:var(--ink-2);font-size:var(--step--1)}
.frame{display:block;width:100%;height:calc(100vh - 128px);min-height:560px;border:0;border-radius:12px;background:var(--ground)}
.wrap.bleed{max-width:none;padding:0}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;font-size:var(--step--1)}
.kv dt{color:var(--ink-3);font-weight:600}.kv dd{margin:0;font-family:var(--mono);overflow-wrap:anywhere}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.figs figure{margin:0}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:10px}
.search{font:inherit;padding:7px 12px;border-radius:8px;border:1px solid var(--border-firm);background:var(--surface);color:var(--ink);min-width:260px}
.toggle{font-size:var(--step--1);color:var(--ink-2);display:flex;gap:6px;align-items:center}
.tbl{max-height:520px;overflow:auto}
.tbl th{position:sticky;top:0;background:var(--surface);cursor:pointer;white-space:nowrap}
.tbl th.s-asc::after{content:" \\25B2";color:var(--accent)}.tbl th.s-desc::after{content:" \\25BC";color:var(--accent)}
.tbl td,.tbl th{text-align:left}
.tbl td.n,.tbl th.n{text-align:right}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:baseline}
.bar{display:grid;grid-template-columns:150px 1fr 70px;gap:10px;align-items:center;margin:6px 0;font-size:var(--step--1)}
.bar i{display:block;height:12px;border-radius:6px}
.btns{display:flex;flex-wrap:wrap;gap:14px;margin:20px 0}
a.big{display:inline-block;text-decoration:none;font-weight:600;padding:14px 22px;border-radius:10px;color:#fff;background:var(--grad);box-shadow:var(--shadow)}
a.big.off{background:var(--surface-2);color:var(--ink-3);pointer-events:none;box-shadow:none}
details.params summary{cursor:pointer;font-weight:600}
@media(max-width:520px){.figs{grid-template-columns:minmax(0,1fr)}.bar{grid-template-columns:100px 1fr 56px}}
"""

JS = r"""
(function(){
var KEY='bm-theme',root=document.documentElement,tb=document.getElementById('theme');
function theme(t){if(t)root.setAttribute('data-theme',t);else root.removeAttribute('data-theme');
 if(tb)tb.textContent=t==='dark'?'Light mode':t==='light'?'Dark mode':'Theme: auto';
 var f=document.getElementById('bmframe');
 try{var d=f&&f.contentDocument&&f.contentDocument.documentElement;if(d){if(t)d.setAttribute('data-theme',t);else d.removeAttribute('data-theme')}}catch(e){}}
var st=null;try{st=localStorage.getItem(KEY)}catch(e){}theme(st==='dark'||st==='light'?st:null);
tb.addEventListener('click',function(){var c=root.getAttribute('data-theme');
 var dm=window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches;
 var n=c?(c==='dark'?'light':'dark'):(dm?'light':'dark');theme(n);try{localStorage.setItem(KEY,n)}catch(e){}});
var fr=document.getElementById('bmframe');
if(fr)fr.addEventListener('load',function(){var c=root.getAttribute('data-theme');if(c)theme(c)});
var tabs=[].slice.call(document.querySelectorAll('nav [role=tab]')),panels=[].slice.call(document.querySelectorAll('.tab'));
function show(id){
 if(panels.every(function(p){return p.id!==id}))id='overview';
 panels.forEach(function(p){p.classList.toggle('on',p.id===id)});
 document.getElementById('wrap').classList.toggle('bleed',id==='bodymap');
 tabs.forEach(function(b){var on=b.dataset.tab===id;b.setAttribute('aria-selected',on);b.tabIndex=on?0:-1});
 window.scrollTo(0,0)}
function go(id){try{history.replaceState(null,'','#'+id)}catch(e){}show(id)}
tabs.forEach(function(b,i){b.addEventListener('click',function(){go(b.dataset.tab)});
 b.addEventListener('keydown',function(e){var d=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;if(!d)return;e.preventDefault();
  var n=tabs[(i+d+tabs.length)%tabs.length];go(n.dataset.tab);n.focus()})});
[].forEach.call(document.querySelectorAll('[data-jump]'),function(b){b.addEventListener('click',function(){go(b.dataset.jump)})});
window.addEventListener('hashchange',function(){show(location.hash.slice(1))});
show(location.hash.slice(1));
var lb=document.getElementById('lb'),li=lb.querySelector('img');
document.addEventListener('click',function(e){var t=e.target;
 if(t.tagName==='IMG'&&t.closest('.fig')){li.src=t.currentSrc||t.src;lb.classList.add('on')}else if(lb.classList.contains('on'))lb.classList.remove('on')});
document.addEventListener('keydown',function(e){if(e.key==='Escape')lb.classList.remove('on')});

var D={};try{D=JSON.parse(document.getElementById('hub-data').textContent)}catch(e){}
var COLORS=D.colors||{};
function td(tr,txt,cls){var c=document.createElement('td');c.textContent=txt==null?'':txt;if(cls)c.className=cls;tr.appendChild(c);return c}
// Events table
var ev=D.events||[],cols=D.event_cols||[],sortK=null,sortDir=1;
var body=document.querySelector('#events tbody'),head=document.querySelector('#events thead tr');
if(body){
 cols.forEach(function(c){var th=document.createElement('th');th.textContent=c[1];th.dataset.k=c[0];
  if(/duration|frames|score|dist|event_id/.test(c[0]))th.className='n';
  th.addEventListener('click',function(){sortDir=sortK===c[0]?-sortDir:1;sortK=c[0];render()});head.appendChild(th)});
 var q=document.getElementById('search'),nc=document.getElementById('show-nc');
 function render(){
  var s=(q.value||'').toLowerCase(),rows=ev.filter(function(r){
   if(!nc.checked&&r.contact_type==='NC')return false;
   if(!s)return true;return cols.some(function(c){return String(r[c[0]]==null?'':r[c[0]]).toLowerCase().indexOf(s)>=0})});
  if(sortK)rows=rows.slice().sort(function(a,b){var x=a[sortK],y=b[sortK];
   if(x==null)return 1;if(y==null)return -1;return (typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y),undefined,{numeric:true}))*sortDir});
  body.textContent='';
  rows.forEach(function(r){var tr=document.createElement('tr');
   cols.forEach(function(c){var v=r[c[0]],cell=td(tr,typeof v==='number'&&!Number.isInteger(v)?v.toFixed(c[0]==='duration_sec'?2:3):v,/duration|frames|score|dist|event_id/.test(c[0])?'n':'');
    if(c[0]==='contact_type'&&v){var sw=document.createElement('span');sw.className='sw';sw.style.background=COLORS[v]||'#999';cell.insertBefore(sw,cell.firstChild)}});
   body.appendChild(tr)});
  [].forEach.call(head.children,function(th){th.classList.remove('s-asc','s-desc');if(th.dataset.k===sortK)th.classList.add(sortDir>0?'s-asc':'s-desc')});
  document.getElementById('ev-count').textContent=rows.length+' of '+ev.length+' rows'}
 q.addEventListener('input',render);nc.addEventListener('change',render);render();
}
// Parameters table
var pb=document.querySelector('#params tbody');
if(pb)(D.params||[]).forEach(function(p){var tr=document.createElement('tr');td(tr,p.section);td(tr,p.parameter);td(tr,p.value);pb.appendChild(tr)});
})();
"""


def _overview(meta: dict, findings, summary, glob, val, has_gallery) -> str:
    rows = [("Video", meta.get("video")), ("Frames processed", glob.get("total_frames")),
            ("Duration", _fmt_sec(glob["total_duration_sec"]) if glob.get("total_duration_sec") is not None else None),
            ("Mice", meta.get("n_mice")), ("Device / GPU", meta.get("gpu") or meta.get("device")),
            ("Run started (UTC)", meta.get("timestamp")), ("Git commit", (meta.get("git_commit") or "")[:10] or None),
            ("Host", meta.get("hostname"))]
    kv = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in rows if v not in (None, ""))
    fd = "".join(f'<div class="card stat"><div class="k">{_e(f["label"])}<span class="who">{_e(f["mouse"])}</span></div>'
                 f'<div class="v">{_e(f["number"])}</div><div class="sub">{_e(f["text"])}</div></div>' for f in findings)
    band = (f'<div class="sec-head"><h2>Key findings</h2><span class="note">provisional, computed from the located contacts</span></div>'
            f'<div class="stats">{fd}</div>') if fd else ""
    tot = next((r for r in summary if r.get("contact_type") == "TOTAL"), {})
    head = _summary_stats([r for r in summary if r.get("contact_type") not in ("ANY",)], {}) if summary else ""
    nrev = (f"{val['n_reviewed']} of {val['n_rows']} rows reviewed" if val["present"] else "no validation folder")
    jumps = [("bodymap", "Bodily Map", "Where each mouse is touched, per mouse and per partner, with a time-animated map."),
             ("session", "Session Report", "Statistics, figures, the full event list and every parameter."),
             ("validation", "Validation", f"Clip and identity review. {nrev}."),
             ("overview", "Run details", "Provenance and configuration for this run (above).")]
    jc = "".join(f'<button type="button" data-jump="{i}"><b>{_e(t)}</b><span>{_e(d)}</span></button>' for i, t, d in jumps[:3])
    jc += ('<button type="button" data-jump="bodymap"><b>Next step</b><span>Open the interactive map to scrub through the session.</span></button>'
           if has_gallery else '<button type="button" data-jump="session"><b>Next step</b><span>Run scripts/bodily_map.py --all to add the bodily maps.</span></button>')
    return f"""<section class="tab" id="overview" role="tabpanel" aria-labelledby="t-overview">
<p class="eyebrow">Run overview</p><h1>{_e(meta.get('run_id'))}</h1>
<p class="lead">{_e(tot.get('events', '?'))} cleaned contact events across {_e(meta.get('n_mice', '?'))} mice.
Everything below is read from this run's own outputs and works offline.</p>
{band}
<div class="sec-head"><h2>Contact totals</h2></div>{head}
<div class="sec-head"><h2>Jump to</h2></div><div class="jump">{jc}</div>
<div class="sec-head"><h2>Run metadata</h2></div><div class="card"><dl class="kv">{kv or '<dd>No provenance.json found.</dd>'}</dl></div>
</section>"""


def _session(summary, glob, bytype, params, run_dir: Path, has_events: bool) -> str:
    figs = "".join(
        f'<figure class="card fig-card"><div class="fig"><img src="contacts/reports/{n}" alt="{_e(t)}" loading="lazy"></div>'
        f'<figcaption class="cap"><b>{_e(t)}.</b> {_e(c)}</figcaption></figure>'
        for n, t, c in FIGURES if (run_dir / "contacts" / "reports" / n).exists())
    bt = ""
    if summary:
        mx = max([float(r.get("total_sec") or 0) for r in summary if r.get("contact_type") in TYPE_COLORS] or [1]) or 1
        bt = "".join(
            f'<div class="bar"><span><span class="sw" style="background:{TYPE_COLORS[r["contact_type"]]}"></span>{_e(r.get("contact_label"))}</span>'
            f'<i style="width:{100 * float(r.get("total_sec") or 0) / mx:.1f}%;background:{TYPE_COLORS[r["contact_type"]]}"></i>'
            f'<span class="num">{_e(r.get("total_sec"))} s</span></div>'
            for r in summary if r.get("contact_type") in TYPE_COLORS)
    impact = _pairs_table(bytype, ["contact_label", "raw_bouts", "real_events", "removed", "raw_total_sec", "real_total_sec", "change_pct"]) if bytype else ""
    events = ('<div class="controls"><input id="search" class="search" type="search" placeholder="Search type, pair or timecode" aria-label="Search events">'
              '<label class="toggle"><input id="show-nc" type="checkbox"> Show no-contact intervals</label></div>'
              '<div class="card"><div class="tbl"><table id="events"><thead><tr></tr></thead><tbody></tbody></table></div></div>'
              if has_events else '<p class="empty">The Events sheet is not in results.xlsx.</p>')
    prm = ('<details class="params card"><summary>Parameters used for this run</summary><div class="tbl" style="margin-top:12px"><table id="params">'
           '<thead><tr><th>Section</th><th>Parameter</th><th>Value</th></tr></thead><tbody></tbody></table></div></details>'
           if params else '<p class="empty">The Parameters sheet is not in results.xlsx.</p>')
    return f"""<section class="tab" id="session" role="tabpanel" aria-labelledby="t-session">
<p class="eyebrow">Session report</p><h1>Contact statistics</h1>
<div class="sec-head"><h2>Session</h2><span class="note">from contacts/results.xlsx</span></div>{_summary_stats(summary, glob)}
<div class="sec-head"><h2>Figures</h2></div>{'<div class="figs">' + figs + '</div>' if figs else '<p class="empty">No figures in contacts/reports/.</p>'}
<div class="cols" style="margin-top:8px"><div><div class="sec-head"><h2>Time by type</h2></div><div class="card">{bt or '<p class="empty">No Summary sheet.</p>'}</div></div>
<div><div class="sec-head"><h2>Filtering impact</h2><span class="note">raw labels vs cleaned events</span></div><div class="card">{impact or '<p class="empty">No ByType sheet.</p>'}</div></div></div>
<div class="sec-head"><h2>Events</h2><span class="note" id="ev-count"></span></div>{events}
<div class="sec-head"><h2>Parameters</h2></div>{prm}
</section>"""


def _validation(val: dict) -> str:
    if not val["present"]:
        return ('<section class="tab" id="validation" role="tabpanel" aria-labelledby="t-validation"><p class="eyebrow">Validation</p>'
                '<h1>Validation</h1><p class="empty">No validation/ folder in this run. Create one with '
                '<code>python scripts/validate_contacts.py sample &lt;run&gt;</code>.</p></section>')
    m = val["manifest"]
    p = m.get("params", {})
    cards = [_stat("Review rows", m.get("n_samples", val["n_rows"]), "events and no-contact windows sampled"),
             _stat("Clips", m.get("n_clips", "?"), f"{p.get('group_by', 'scene')} grouping, {p.get('pad_sec', '?')} s padding"),
             _stat("Seed", m.get("seed", "?"), f"{p.get('per_type', '?')} events per type")]
    for t, n in (m.get("sampled_per_type") or {}).items():
        cards.append(_stat(t, n, f"sampled of {(m.get('available_per_type') or {}).get(t, '?')} available", TYPE_COLORS.get(t)))
    cards.append(_stat("No-contact windows", m.get("nc_windows_sampled", "?"), f"requested {m.get('nc_windows_requested', '?')}"))
    if val["n_reviewed"]:
        rev = "".join(_stat(k, val["verdicts"].get(k, 0), "verdicts") for k in VERDICTS if val["verdicts"].get(k))
        other = {k: v for k, v in val["verdicts"].items() if k not in VERDICTS}
        rev += "".join(_stat(k, v, "verdicts") for k, v in other.items())
        status = (f'<div class="sec-head"><h2>Review progress</h2><span class="note">{val["n_reviewed"]} of {val["n_rows"]} rows have a verdict</span></div>'
                  f'<div class="stats">{rev}</div>')
    else:
        status = ('<div class="callout"><h2>Not yet reviewed</h2><p>No verdicts are filled in <code>validation/review.xlsx</code>. '
                  'Review the clips, fill the <code>verdict</code> column, then run <code>scripts/validate_contacts.py score</code>.</p></div>')

    def btn(href, label, ok):
        return (f'<a class="big" href="{href}" target="_blank" rel="noopener">{label}</a>' if ok
                else f'<a class="big off" href="#" aria-disabled="true">{label} (not found)</a>')
    return f"""<section class="tab" id="validation" role="tabpanel" aria-labelledby="t-validation">
<p class="eyebrow">Validation</p><h1>Human validation sample</h1>
<p class="lead">Clips for a reviewer to confirm that detected contacts are real. The viewers are large, so they open in a new tab.</p>
<div class="btns">{btn('validation/index.html', 'Open event clip viewer', val['has_clips'])}
{btn('validation/identity/identity.html', 'Open identity checkpoints', val['has_identity'])}</div>
<div class="sec-head"><h2>Sample</h2></div><div class="stats">{''.join(cards)}</div>{status}</section>"""


# ---------------------------------------------------------------- build
def build_hub(run_dir, out_path=None) -> Path:
    """Write the unified hub to ``out_path`` (default ``<run_dir>/index.html``) and return it."""
    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "index.html"
    xlsx = run_dir / "contacts" / "results.xlsx"
    summary, glob_rows = read_sheet(xlsx, "Summary"), read_sheet(xlsx, "Global")
    glob = {r.get("metric"): r.get("value") for r in glob_rows}
    bytype, params = read_sheet(xlsx, "ByType"), read_sheet(xlsx, "Parameters")
    events = read_sheet(xlsx, "Events")
    peranimal = read_sheet(xlsx, "PerAnimal")
    prov = read_json(run_dir / "provenance.json")
    cfg = read_config_flat(run_dir / "config_used.yaml")
    val = read_validation(run_dir)

    run_id = run_dir.name
    ts = prov.get("timestamp_utc") or ""
    n_mice = len(peranimal) or int(cfg.get("detection.max_animals", 0) or 0) or None
    video = prov.get("video_path") or cfg.get("video_path") or ""
    meta = dict(run_id=run_id, timestamp=ts, n_mice=n_mice, video=video, git_commit=prov.get("git_commit"),
                hostname=prov.get("hostname"), gpu=prov.get("gpu"), device=cfg.get("device"))
    date = ts[:10] or (re.match(r"\d{4}-\d{2}-\d{2}", run_id).group(0) if re.match(r"\d{4}-\d{2}-\d{2}", run_id) else "")
    chip = " · ".join(x for x in [f"{n_mice} mice" if n_mice else "", date] if x)

    records = _load_records(run_dir)
    mice = sorted({r[k] for r in records for k in ("focal_mouse", "other_mouse")})
    findings = key_findings(records, mice) if records else []
    has_gallery = (run_dir / "bodily_map" / "index.html").exists()

    data = dict(events=[{k: e.get(k) for k, _ in EVENT_COLS} | {"contact_type": e.get("contact_type")} for e in events],
                event_cols=[list(c) for c in EVENT_COLS], params=params, colors=TYPE_COLORS)
    nav = "".join(f'<button role="tab" id="t-{i}" data-tab="{i}" aria-controls="{i}" aria-selected="false" tabindex="-1">{l}</button>'
                  for i, l in [("overview", "Overview"), ("bodymap", "Bodily Map"), ("session", "Session Report"), ("validation", "Validation")])
    mark = ('<svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">'
            '<circle cx="6" cy="10" r="4.5" fill="#2563EB"/><circle cx="14" cy="10" r="4.5" fill="#8B5CF6" fill-opacity=".85"/></svg>')
    if has_gallery:
        bodymap = ('<section class="tab" id="bodymap" role="tabpanel" aria-labelledby="t-bodymap">'
                   '<iframe id="bmframe" class="frame" src="bodily_map/index.html?embed=1" '
                   'title="Bodily maps of social contact"></iframe></section>')
    else:
        bodymap = ('<section class="tab" id="bodymap" role="tabpanel" aria-labelledby="t-bodymap"><div class="wrap"><p class="empty">'
                   'No bodily_map/ folder yet. Run <code>python scripts/bodily_map.py &lt;run&gt; --all</code>. '
                   'It will be embedded here from <code>bodily_map/index.html?embed=1</code>.</p></div></section>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run Report Hub</title><style>{GALLERY_CSS}{EXTRA_CSS}</style></head><body>
<div class="topbar"><div class="topin"><span class="mark">{mark}<span class="ttl">Run Report Hub</span></span>
<span class="spacer"></span><span class="filechip">{_e(run_id)}</span>{f'<span class="filechip">{_e(chip)}</span>' if chip else ''}
<button id="theme" class="btn" type="button">Theme: auto</button></div>
<nav role="tablist" aria-label="Report sections">{nav}</nav></div>
<div class="wrap" id="wrap">{_overview(meta, findings, summary, glob, val, has_gallery)}{bodymap}
{_session(summary, glob, bytype, params, run_dir, bool(events))}{_validation(val)}</div>
<footer>Generated by scripts/report_hub.py · read-only view of this run's outputs · provisional research output</footer>
<div id="lb" aria-hidden="true"><img alt=""></div>
<script type="application/json" id="hub-data">{_json(data)}</script><script>{JS}</script></body></html>"""
    out_path.write_text(page, encoding="utf-8")
    return out_path
