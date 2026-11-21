"""Renderers for the bodily map of social contact (matplotlib, Agg).

Definitions (all in terms of located-contact records, see ``locate.py``):

* passive map of A   : records with ``focal_mouse == A`` (others touching A)
* active map A -> B  : ``other_mouse == A`` and ``focal_mouse == B``
* active combined A  : ``other_mouse == A`` (A touching anyone), pooled on one template

Every map accumulates count AND duration; the heatmap shows duration (s).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .locate import slot_color_rgb
from .template import ASPECT, REGION_ORDER, XLIM, YLIM, draw_template

CONTACT_TYPES = ["N2N", "N2AG", "N2B", "SBS"]
TYPE_NAMES = {"N2N": "nose-to-nose", "N2AG": "nose-to-anogenital", "N2B": "nose-to-body", "SBS": "side-by-side"}
# One shape per mouse slot (R1..R6): circle, square, triangle, diamond, star, plus.
MARKER_SHAPES = ["o", "s", "^", "D", "*", "P"]
MARKER_NAMES = ["circle", "square", "triangle", "diamond", "star", "plus"]
CMAP = "magma"
DPI = 150
GRID_T, GRID_S = 150, 90
SIGMA_T, SIGMA_S = 0.016, 0.016   # Gaussian smoothing in t and s units


# ------------------------------------------------------------ selection
def select_passive(records: Sequence[dict], focal: str) -> List[dict]:
    return [r for r in records if r["focal_mouse"] == focal]


def select_active(records: Sequence[dict], focal: str, target: str) -> List[dict]:
    return [r for r in records if r["other_mouse"] == focal and r["focal_mouse"] == target]


def select_active_combined(records: Sequence[dict], focal: str) -> List[dict]:
    return [r for r in records if r["other_mouse"] == focal]


# ------------------------------------------------------------ helpers
def mouse_slot(name: str) -> int:
    """'R3' -> 2 (0-based slot)."""
    return int(str(name)[1:]) - 1


def marker_for(name: str) -> str:
    return MARKER_SHAPES[mouse_slot(name) % len(MARKER_SHAPES)]


def color_for(name: str):
    return slot_color_rgb(mouse_slot(name))


def region_table(rows: Sequence[dict]) -> Dict[str, dict]:
    """Per-region count and total duration (all template regions, in order)."""
    tab = {reg: {"count": 0, "dur_sec": 0.0} for reg in REGION_ORDER}
    for r in rows:
        a = tab.setdefault(r["region"], {"count": 0, "dur_sec": 0.0})
        a["count"] += 1
        a["dur_sec"] += r["dur_sec"]
    return tab


def _gauss_kernel(sigma_cells: float) -> np.ndarray:
    rad = max(1, int(round(3 * sigma_cells)))
    x = np.arange(-rad, rad + 1)
    k = np.exp(-0.5 * (x / sigma_cells) ** 2)
    return k / k.sum()


def duration_heatmap(rows: Sequence[dict]):
    """Smoothed duration (s) per cell over the template view. Returns (grid, extent, n_off_view)."""
    t0, t1 = XLIM
    s0, s1 = YLIM
    t = np.array([r["t"] for r in rows], float)
    s = np.array([r["s"] for r in rows], float)
    w = np.array([r["dur_sec"] for r in rows], float)
    ok = (t >= t0) & (t <= t1) & (s >= s0) & (s <= s1)
    h, _, _ = np.histogram2d(s[ok], t[ok], bins=[GRID_S, GRID_T], range=[[s0, s1], [t0, t1]], weights=w[ok])
    kt = _gauss_kernel(SIGMA_T / ((t1 - t0) / GRID_T))
    ks = _gauss_kernel(SIGMA_S / ((s1 - s0) / GRID_S))
    h = np.apply_along_axis(lambda v: np.convolve(v, kt, mode="same"), 1, h)
    h = np.apply_along_axis(lambda v: np.convolve(v, ks, mode="same"), 0, h)
    return h, [t0, t1, s0, s1], int((~ok).sum()) if len(rows) else 0


def _draw_heat(ax, fig, rows, cbar_label=None, cbar_orient="vertical", small=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    draw_template(ax, labels=not small, fontsize=6.5, lw=1.0 if small else 1.2)
    if not rows:
        ax.text(0.5, 0.0, "no contacts", ha="center", va="center", color="#777777", fontsize=9, zorder=9)
        return 0
    h, ext, off = duration_heatmap(rows)
    vmax = float(h.max()) if h.max() > 0 else 1.0
    masked = np.ma.masked_less(h, vmax * 0.02)
    cm = plt.get_cmap(CMAP).copy()
    im = ax.imshow(masked, origin="lower", extent=ext, aspect=ax.get_aspect(), cmap=cm,
                   norm=PowerNorm(0.5, vmin=0, vmax=vmax), alpha=0.85, zorder=3, interpolation="bilinear")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect(ASPECT, adjustable="box")
    if cbar_orient == "vertical":
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
    else:
        cb = fig.colorbar(im, cax=ax.inset_axes([0.1, -0.36, 0.8, 0.06]), orientation="horizontal")
    from matplotlib.ticker import MaxNLocator
    cb.locator = MaxNLocator(4)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=7)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)
    return off


def _scatter(ax, rows, key, size=16, alpha=0.55):
    for name in sorted({r[key] for r in rows}, key=mouse_slot):
        sel = [r for r in rows if r[key] == name]
        ax.scatter([r["t"] for r in sel], [r["s"] for r in sel], s=size * (1.9 if marker_for(name) == "*" else 1.0),
                   marker=marker_for(name), c=[color_for(name)], edgecolors="#222222", linewidths=0.35,
                   alpha=alpha, zorder=4, clip_on=True)


def _style_axes(ax, small=False):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.tick_params(labelsize=6 if small else 7)
    if not small:
        ax.set_xlabel("t along body (0 = nose, 1 = tail tip)", fontsize=8)
        ax.set_ylabel("s (body lengths; + right, - left)", fontsize=8)
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb")


def _legend_panel(ax, keys, key_label, rows, title, subtitle):
    from matplotlib.lines import Line2D
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=12, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.0, 0.90, subtitle, fontsize=8, va="top", color="#444444", transform=ax.transAxes, linespacing=1.4)
    handles = [Line2D([], [], marker=marker_for(k), ls="", mfc=color_for(k), mec="#222222", mew=0.5,
                      ms=11 if marker_for(k) == "*" else 8, label=f"{k}  ({MARKER_NAMES[mouse_slot(k) % 6]})")
               for k in keys]
    if handles:
        ax.legend(handles=handles, title=key_label, loc="upper left", bbox_to_anchor=(0.0, 0.68),
                  frameon=False, fontsize=8, title_fontsize=8, alignment="left")
    tab = region_table(rows)
    lines = ["region        n   dur (s)"] + [
        f"{reg:<11}{v['count']:>5} {v['dur_sec']:>8.2f}" for reg, v in tab.items() if v["count"]]
    ax.text(0.0, 0.0, "\n".join(lines), fontsize=7, family="monospace", va="bottom", transform=ax.transAxes)


def _render_map(rows, key, key_label, title, subtitle, path: Path) -> Dict[str, dict]:
    """Common layout: big all-types panel + legend/table, and one panel per contact type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 7.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.8], hspace=0.55, wspace=0.18,
                          left=0.045, right=0.985, top=0.95, bottom=0.05)
    ax = fig.add_subplot(gs[0, 0:3])
    off = _draw_heat(ax, fig, rows, cbar_label="contact duration (s)")
    _scatter(ax, rows, key)
    _style_axes(ax)
    ax.set_title("All contact types combined", fontsize=10, loc="left")
    keys = sorted({r[key] for r in rows}, key=mouse_slot)
    n_off = f"\n{off} off-plot point(s): table only." if off else ""
    tot = sum(r["dur_sec"] for r in rows)
    _legend_panel(fig.add_subplot(gs[0, 3]), keys, key_label, rows, title,
                  f"{subtitle}\n{len(rows)} contact frames, {tot:.1f} s total{n_off}")
    for i, ct in enumerate(CONTACT_TYPES):
        a = fig.add_subplot(gs[1, i])
        sub = [r for r in rows if r["contact_type"] == ct]
        _draw_heat(a, fig, sub, cbar_orient="horizontal", small=True)
        _scatter(a, sub, key, size=8, alpha=0.5)
        _style_axes(a, small=True)
        a.set_title(f"{ct} {TYPE_NAMES[ct]}\nn={len(sub)}, {sum(r['dur_sec'] for r in sub):.1f} s (own colour scale)",
                    fontsize=8, loc="left")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return region_table(rows)


# ------------------------------------------------------------ public API
def render_passive(records, focal: str, out_dir) -> Dict[str, dict]:
    """passive_<focal>.png: others touching ``focal``, drawn on focal's body."""
    rows = select_passive(records, focal)
    return _render_map(rows, "other_mouse", "touching mouse", f"Passive map of {focal}",
                       f"Where other mice touch {focal}.\nMarker = who touched.", Path(out_dir) / f"passive_{focal}.png")


def render_active(records, focal: str, target: str, out_dir) -> Dict[str, dict]:
    """active_<focal>_to_<target>.png: where ``focal`` touches ``target`` (on target's body)."""
    rows = select_active(records, focal, target)
    return _render_map(rows, "other_mouse", "touching mouse", f"Active map {focal} → {target}",
                       f"Where {focal} touches {target},\ndrawn on {target}'s body.",
                       Path(out_dir) / f"active_{focal}_to_{target}.png")


def render_active_combined(records, focal: str, out_dir) -> Dict[str, dict]:
    """active_<focal>_combined.png: all of ``focal``'s touches pooled on a generic mouse."""
    rows = select_active_combined(records, focal)
    return _render_map(rows, "focal_mouse", "touched mouse", f"Active map of {focal}, all partners pooled",
                       f"Where {focal} touches others,\non a generic template.\nMarker = touched mouse.",
                       Path(out_dir) / f"active_{focal}_combined.png")
