"""Summary graphs comparing all mice (duration based, honest active/passive framing).

Active/passive is defined ONLY for the directional types N2AG and N2B, where
``other_mouse`` is the active investigator and ``focal_mouse`` the passive receiver.
N2N and SBS are mutual and are never counted as active or passive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from .locate import slot_color_rgb
from .render import CONTACT_TYPES, DPI, TYPE_NAMES, mouse_slot

DIRECTIONAL = ("N2AG", "N2B")
MUTUAL = ("N2N", "SBS")
SUMMARY_REGIONS = ["nose", "head", "flank", "mid_back", "rump", "anogenital", "tail"]
PROVISIONAL = "provisional, pending human (two-rater) validation of the investigator label"
TYPE_HATCH = {"N2N": "", "N2AG": "//", "N2B": "..", "SBS": "xx"}


def _region_group(region: str) -> str:
    return "flank" if region.startswith("flank") else region


def mice_of(records: Sequence[dict]) -> List[str]:
    return sorted({r[k] for r in records for k in ("focal_mouse", "other_mouse")}, key=mouse_slot)


def contacts_per_mouse(records, mice) -> Dict[str, Dict[str, dict]]:
    """Per mouse, per type: duration and frame count of contacts the mouse takes part in (either role)."""
    out = {m: {t: {"dur": 0.0, "count": 0} for t in CONTACT_TYPES} for m in mice}
    for r in records:
        if r["contact_type"] not in CONTACT_TYPES:
            continue
        for m in {r["focal_mouse"], r["other_mouse"]}:
            if m in out:
                out[m][r["contact_type"]]["dur"] += r["dur_sec"]
                out[m][r["contact_type"]]["count"] += 1
    return out


def active_passive(records, mice) -> Dict[str, dict]:
    """Directional-only (N2AG+N2B) active/passive durations; N2N+SBS reported as mutual."""
    out = {m: {"active": 0.0, "passive": 0.0, "mutual": 0.0} for m in mice}
    for r in records:
        ct, d = r["contact_type"], r["dur_sec"]
        if ct in DIRECTIONAL:
            if r["other_mouse"] in out:
                out[r["other_mouse"]]["active"] += d
            if r["focal_mouse"] in out:
                out[r["focal_mouse"]]["passive"] += d
        elif ct in MUTUAL:
            for m in {r["focal_mouse"], r["other_mouse"]}:
                if m in out:
                    out[m]["mutual"] += d
    for v in out.values():
        tot = v["active"] + v["passive"]
        v["pct_initiated"] = 100.0 * v["active"] / tot if tot > 0 else None
    return out


def region_preference(records, mice) -> Dict[str, Dict[str, float]]:
    """Share of passive (touched) duration per body region, normalized to sum 1 per mouse
    (all zeros if the mouse was never touched)."""
    raw = {m: {g: 0.0 for g in SUMMARY_REGIONS} for m in mice}
    for r in records:
        g = _region_group(r["region"])
        if r["focal_mouse"] in raw and g in raw[r["focal_mouse"]]:
            raw[r["focal_mouse"]][g] += r["dur_sec"]
    out = {}
    for m, v in raw.items():
        tot = sum(v.values())
        out[m] = {g: (x / tot if tot > 0 else 0.0) for g, x in v.items()}
    return out


def contact_matrix(records, mice) -> Dict[str, Dict[str, float]]:
    """Duration with rows = toucher (other_mouse), cols = touched (focal_mouse); all types."""
    mat = {a: {b: 0.0 for b in mice} for a in mice}
    for r in records:
        a, b = r["other_mouse"], r["focal_mouse"]
        if a in mat and b in mat[a] and a != b:
            mat[a][b] += r["dur_sec"]
    return mat


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _col(m):
    return slot_color_rgb(mouse_slot(m))


def render_contacts_per_mouse(records, mice, path: Path) -> Path:
    plt = _plt()
    data = contacts_per_mouse(records, mice)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, lab in ((axes[0], "dur", "contact duration (s)"), (axes[1], "count", "contact frames")):
        bottom = [0.0] * len(mice)
        for t in CONTACT_TYPES:
            vals = [data[m][t][key] for m in mice]
            ax.bar(mice, vals, bottom=bottom, color=[_col(m) for m in mice], edgecolor="black",
                   linewidth=0.8, hatch=TYPE_HATCH[t], label=f"{t} ({TYPE_NAMES[t]})")
            bottom = [b + v for b, v in zip(bottom, vals)]
        for i, b in enumerate(bottom):
            ax.text(i, b, f"{b:.0f}" if key == "count" else f"{b:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(lab)
        ax.set_xlabel("mouse")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Total contact duration per mouse")
    axes[1].set_title("Number of contact frames per mouse")
    axes[0].legend(fontsize=8, title="contact type (hatching)", loc="upper right")
    fig.suptitle("Who is most social? Contacts a mouse takes part in (either role), by type", fontsize=13)
    fig.text(0.5, 0.005, "A contact between two mice is credited to both. Following is excluded. "
             "Bar colour = mouse; hatching = contact type.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def render_active_passive(records, mice, path: Path) -> Path:
    plt = _plt()
    ap = active_passive(records, mice)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    n, w = len(mice), 0.26
    xs = list(range(n))
    cols = [_col(m) for m in mice]
    ax.bar([x - w for x in xs], [ap[m]["active"] for m in mice], w, color=cols, edgecolor="black",
           label="active (investigates)")
    ax.bar(xs, [ap[m]["passive"] for m in mice], w, color=cols, edgecolor="black", hatch="//",
           alpha=0.7, label="passive (investigated)")
    ax.bar([x + w for x in xs], [ap[m]["mutual"] for m in mice], w, color="white", edgecolor="gray",
           hatch="..", label="mutual (N2N + SBS), no direction")
    top = max([max(v["active"], v["passive"], v["mutual"]) for v in ap.values()] + [1.0])
    for x, m in zip(xs, mice):
        v = ap[m]
        p = v["pct_initiated"]
        ax.text(x, top * 1.08, "n/a" if p is None else f"{p:.0f}% initiated", ha="center",
                fontsize=10, fontweight="bold")
        for dx, key in ((-w, "active"), (0, "passive"), (w, "mutual")):
            ax.text(x + dx, v[key], f"{v[key]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(mice)
    ax.set_ylim(0, top * 1.32)
    ax.set_xlabel("mouse")
    ax.set_ylabel("duration (s)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)
    ax.set_title("Active vs passive (directional contacts only) - % initiated is PROVISIONAL,\n"
                 "pending human (two-rater) validation of the investigator label", fontsize=12)
    fig.text(0.5, 0.005, "Active/passive uses nose-to-anogenital (N2AG) and nose-to-body (N2B) only. "
             "Nose-to-nose and side-by-side are mutual: shown as the separate outlined bar, never split.\n"
             "% initiated = active / (active + passive) over directional contacts.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def render_region_preference(records, mice, path: Path) -> Path:
    import numpy as np
    plt = _plt()
    pref = region_preference(records, mice)
    mat = np.array([[pref[m][g] for g in SUMMARY_REGIONS] for m in mice])
    fig, ax = plt.subplots(figsize=(9.5, 1.0 + 0.7 * len(mice) + 1.5))
    im = ax.imshow(mat, cmap="magma", vmin=0, vmax=max(mat.max(), 1e-9), aspect="auto")
    ax.set_xticks(range(len(SUMMARY_REGIONS)))
    ax.set_xticklabels(SUMMARY_REGIONS)
    ax.set_yticks(range(len(mice)))
    ax.set_yticklabels(mice)
    for i in range(len(mice)):
        for j in range(len(SUMMARY_REGIONS)):
            v = mat[i, j]
            ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if v < 0.5 * mat.max() else "black")
    fig.colorbar(im, ax=ax, label="share of the touched duration")
    ax.set_xlabel("body region of the touched mouse (nose to tail)")
    ax.set_ylabel("touched mouse")
    ax.set_title("Where is each mouse touched? (passive; each row sums to 100%)")
    fig.text(0.5, 0.005, "All contact types except following; duration weighted.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def render_contact_matrix(records, mice, path: Path) -> Path:
    import numpy as np
    plt = _plt()
    mat = contact_matrix(records, mice)
    arr = np.array([[mat[a][b] for b in mice] for a in mice])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(arr, cmap="magma")
    ax.set_xticks(range(len(mice)))
    ax.set_xticklabels(mice)
    ax.set_yticks(range(len(mice)))
    ax.set_yticklabels(mice)
    for i in range(len(mice)):
        for j in range(len(mice)):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", color="gray")
            else:
                ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if arr[i, j] < 0.5 * arr.max() else "black")
    fig.colorbar(im, ax=ax, label="contact duration (s)")
    ax.set_xlabel("touched mouse")
    ax.set_ylabel("toucher mouse")
    ax.set_title("Who touches whom (duration, s)")
    fig.text(0.5, 0.005, "Row = other mouse in the record (investigator for N2AG/N2B). "
             "Includes mutual N2N/SBS, where direction is arbitrary.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


SUMMARY_FILES = ["summary_contacts_per_mouse.png", "summary_active_passive.png",
                 "summary_region_preference.png", "summary_contact_matrix.png"]


def render_all_summaries(records, out_dir) -> List[Path]:
    out = Path(out_dir)
    mice = mice_of(records)
    return [render_contacts_per_mouse(records, mice, out / SUMMARY_FILES[0]),
            render_active_passive(records, mice, out / SUMMARY_FILES[1]),
            render_region_preference(records, mice, out / SUMMARY_FILES[2]),
            render_contact_matrix(records, mice, out / SUMMARY_FILES[3])]
