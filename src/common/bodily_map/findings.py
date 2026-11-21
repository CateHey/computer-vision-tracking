"""Key findings computed from located-contact records (no values invented)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from .summary import active_passive


def mouse_stats(records: Sequence[dict], mice: Sequence[str]) -> Dict[str, dict]:
    """Per-mouse totals used by findings and the per-mouse stat chips."""
    ap = active_passive(records, mice)
    out = {}
    for m in mice:
        touched = [r for r in records if r["focal_mouse"] == m]
        touching = [r for r in records if r["other_mouse"] == m]
        reg, partner = defaultdict(float), defaultdict(float)
        for r in touched:
            reg[r["region"]] += r["dur_sec"]
        for r in touched + touching:
            p = r["other_mouse"] if r["focal_mouse"] == m else r["focal_mouse"]
            partner[p] += r["dur_sec"]
        tot_passive = sum(reg.values())
        out[m] = {
            "total": sum(r["dur_sec"] for r in touched) + sum(r["dur_sec"] for r in touching),
            "pct_initiated": ap[m]["pct_initiated"],
            "top_region": max(reg, key=reg.get) if reg else None,
            "top_region_share": (100 * max(reg.values()) / tot_passive) if reg else None,
            "top_partner": max(partner, key=partner.get) if partner else None,
            "top_partner_sec": max(partner.values()) if partner else None,
            "nose_share": 100 * reg.get("nose", 0.0) / tot_passive if tot_passive else None,
            "ag_share": 100 * reg.get("anogenital", 0.0) / tot_passive if tot_passive else None,
        }
    return out


def _best(stats, key, mice, low=False):
    cand = [(stats[m][key], m) for m in mice if stats[m][key] is not None]
    if not cand:
        return None
    return (min if low else max)(cand, key=lambda x: x[0])


def key_findings(records: Sequence[dict], mice: Sequence[str]) -> List[dict]:
    """List of {mouse, label, number, text}; ``mouse`` may be a 'R1' or 'R1+R2' string."""
    st = mouse_stats(records, mice)
    out: List[dict] = []
    b = _best(st, "pct_initiated", mice)
    if b:
        out.append(dict(mouse=b[1], label="Most active initiator", number=f"{b[0]:.0f}%",
                        text=f"{b[1]} initiated the largest share of its directional contacts "
                             f"(nose-to-anogenital and nose-to-body). % initiated is provisional."))
    b = _best(st, "pct_initiated", mice, low=True)
    if b:
        out.append(dict(mouse=b[1], label="Most investigated", number=f"{b[0]:.0f}%",
                        text=f"{b[1]} initiated the smallest share of its directional contacts, "
                             f"so it was mostly the one being investigated. % initiated is provisional."))
    b = _best(st, "total", mice)
    if b:
        out.append(dict(mouse=b[1], label="Most social overall", number=f"{b[0]:.0f} s",
                        text=f"{b[1]} had the most total contact time, touching and touched, "
                             f"all contact types (following excluded)."))
    b = _best(st, "nose_share", mice)
    if b:
        out.append(dict(mouse=b[1], label="Most nose-greeted", number=f"{b[0]:.0f}%",
                        text=f"{b[0]:.0f}% of the time {b[1]} was touched, the contact landed on its nose."))
    b = _best(st, "ag_share", mice)
    if b:
        out.append(dict(mouse=b[1], label="Most rear-sniffed", number=f"{b[0]:.0f}%",
                        text=f"{b[0]:.0f}% of the time {b[1]} was touched, the contact landed on its anogenital region."))
    pair = defaultdict(float)
    for r in records:
        pair[tuple(sorted((r["focal_mouse"], r["other_mouse"])))] += r["dur_sec"]
    if pair:
        (a, c), d = max(pair.items(), key=lambda kv: kv[1])
        out.append(dict(mouse=f"{a}+{c}", label="Busiest dyad", number=f"{d:.0f} s",
                        text=f"{a} and {c} spent the most time in contact with each other, in either direction."))
    return out
