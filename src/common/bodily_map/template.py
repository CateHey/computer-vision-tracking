"""Canonical template mouse in the (t, s) coordinate frame of the bodily map.

Orientation: nose on the LEFT, tail on the RIGHT, viewed from above. With the
nose pointing left, the mouse's right side is the TOP of the plot (s > 0) and
its left side the bottom (s < 0), which matches ``locate.py``.

Coordinates
-----------
* ``t``  along the nose -> tail_tip polyline (nose = 0, tail tip = 1).
* ``s``  signed lateral offset in units of the nose -> tail_start axis.
* ``u``  region axis of ``locate.py`` (nose -> tail_start = 0..1, tail > 1).
  In this data the tail_start sits at t ~ 0.58, so ``t = U2T * u``.

Because t and s use different units, the axes are drawn with a fixed aspect
(``ASPECT``) so the silhouette keeps natural proportions.  The head and nose
are real 2D areas (a pointed head ~0.1 wide), so nose-to-nose contacts spread
over an area instead of piling onto one x value.  Matplotlib is only used
inside ``draw_template``.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

# Region cuts shared with locate.py (kept literal to avoid importing cv deps).
U2T = 0.58            # t of tail_start (u = 1)
U_NOSE, U_HEAD, U_TORSO, U_ANOGENITAL, U_TAIL = 0.06, 0.20, 0.75, 0.90, 1.15
S_FLANK = 0.08
U_TAIL_TIP = 1.0 / U2T      # u at which t = 1
ASPECT = U2T                # y-unit length / x-unit length for natural shape
XLIM = (-0.06, 1.04)
YLIM = (-0.42, 0.42)

REGION_ORDER = ["nose", "head", "flank_R", "mid_back", "flank_L", "rump", "anogenital", "tail"]
DRAW_ONLY = ["ear_L", "ear_R"]           # drawn, but never receive records
ALL_REGIONS = REGION_ORDER + DRAW_ONLY
REGION_LABELS = {"nose": "nose", "head": "head", "flank_R": "flank R", "flank_L": "flank L",
                 "mid_back": "mid back", "rump": "rump", "anogenital": "anogenital",
                 "tail": "tail", "ear_L": "ear L", "ear_R": "ear R"}

_U = np.array([0, .005, .02, .06, .12, .2, .35, .55, .75, .9, 1.0, 1.1, 1.15])
_W = np.array([0, .012, .030, .052, .078, .098, .135, .160, .150, .120, .080, .042, .028])


def _smooth_halfwidth(n: int = 400) -> tuple:
    u = np.linspace(0, U_TAIL, n)
    w = np.interp(u, _U, _W)
    k = np.exp(-0.5 * (np.arange(-12, 13) / 5.0) ** 2)
    k /= k.sum()
    wp = np.pad(w, 12, mode="edge")
    ws = np.convolve(wp, k, mode="valid")
    ws[0] = 0.0
    return u, np.maximum(ws, 0.0)


_UB, _WB = _smooth_halfwidth()


def half_width(u):
    """Body half-width (in s units) at region-axis position ``u`` (0..1.15)."""
    return np.interp(u, _UB, _WB)


def slice_poly(u0: float, u1: float, smin: float, smax: float, n: int = 40) -> np.ndarray:
    """Vertices (t, s) of the part of the body with u0<=u<=u1 and smin<=s<=smax."""
    u = np.linspace(u0, u1, n)
    w = half_width(u)
    top = np.clip(smax, -w, w)
    bot = np.clip(smin, -w, w)
    pts = np.vstack([np.column_stack([U2T * u, top]), np.column_stack([U2T * u[::-1], bot[::-1]])])
    keep = np.ones(len(pts), bool)
    keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)
    return pts[keep]


def _ellipse(t0: float, s0: float, rt: float, rs: float, angle_deg: float = 0.0, n: int = 40) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rt * np.cos(a), rs * np.sin(a)
    c, s_ = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return np.column_stack([t0 + c * x - s_ * y, s0 + s_ * x + c * y])


def body_outline() -> np.ndarray:
    """Closed outline of the head + body (u = 0..1.15)."""
    return slice_poly(0.0, U_TAIL, -1, 1, n=120)


def tail_outline() -> np.ndarray:
    u = np.linspace(U_TAIL, U_TAIL_TIP, 30)
    w = np.interp(u, [U_TAIL, U_TAIL_TIP], [0.026, 0.004])
    return np.vstack([np.column_stack([U2T * u, w]), np.column_stack([U2T * u[::-1], -w[::-1]])])


def _build_regions() -> Dict[str, List[np.ndarray]]:
    return {
        "nose": [slice_poly(0, U_NOSE, -1, 1)],
        "head": [slice_poly(U_NOSE, U_HEAD, -1, 1)],
        "flank_R": [slice_poly(U_HEAD, U_TORSO, S_FLANK, 1)],
        "mid_back": [slice_poly(U_HEAD, U_TORSO, -S_FLANK, S_FLANK)],
        "flank_L": [slice_poly(U_HEAD, U_TORSO, -1, -S_FLANK)],
        "rump": [slice_poly(U_TORSO, U_ANOGENITAL, -1, 1)],
        "anogenital": [slice_poly(U_ANOGENITAL, U_TAIL, -1, 1)],
        "tail": [tail_outline()],
        # Ears sit just behind the head, sticking out sideways and a little back.
        "ear_R": [_ellipse(U2T * 0.175, 0.112, 0.028, 0.052, 25)],
        "ear_L": [_ellipse(U2T * 0.175, -0.112, 0.028, 0.052, -25)],
    }


REGION_POLYS: Dict[str, List[np.ndarray]] = _build_regions()

# (t, s) anchor for each small label.
_LABEL_POS = {
    "nose": (0.0, -0.15), "head": (0.045, 0.0),
    "flank_R": (0.29, 0.105), "flank_L": (0.29, -0.105),
    "mid_back": (0.29, 0.0), "rump": (0.48, 0.0),
    "anogenital": (0.585, -0.17), "tail": (0.83, 0.075),
    "ear_R": (0.10, 0.2), "ear_L": (0.10, -0.2),
}


def region_of_point(t: float, s: float) -> str:
    """Template region of a (t, s) point (same cuts as locate.region_from)."""
    u = t / U2T
    if u >= U_TAIL:
        return "tail"
    if u >= U_ANOGENITAL:
        return "anogenital"
    if u < U_NOSE:
        return "nose"
    if u < U_HEAD:
        return "head"
    if u < U_TORSO:
        return "mid_back" if abs(s) < S_FLANK else ("flank_R" if s > 0 else "flank_L")
    return "rump"


def draw_template(ax, labels: bool = True, fill: str = "#f5f1e8", edge: str = "#4d4d4d",
                  divider: bool = True, fontsize: float = 6.5, lw: float = 1.2):
    """Draw the silhouette on ``ax`` in (t, s) coordinates and fix the view."""
    from matplotlib.patches import Polygon

    def poly(v, **kw):
        return Polygon(v, closed=True, joinstyle="round", **kw)

    shadow = body_outline() + np.array([0.004, -0.006])
    ax.add_patch(poly(shadow, fc=(0, 0, 0, 0.07), ec="none", zorder=0.5))
    for name in ("ear_R", "ear_L"):
        ax.add_patch(poly(REGION_POLYS[name][0], fc="#ecc9c0", ec=edge, lw=lw * 0.8, zorder=1))
    ax.add_patch(poly(tail_outline(), fc="#e9dfd0", ec=edge, lw=lw * 0.8, zorder=1))
    ax.add_patch(poly(body_outline(), fc=fill, ec="none", zorder=1.2))
    ax.add_patch(poly(body_outline(), fc="none", ec=edge, lw=lw, zorder=6))
    if divider:
        for name in ("nose", "head", "flank_R", "mid_back", "flank_L", "rump", "anogenital"):
            for v in REGION_POLYS[name]:
                ax.add_patch(poly(v, fc="none", ec="#8a8a8a", lw=0.5, ls=(0, (2, 2)), zorder=5))
    # Eyes give the head a direction cue.
    ax.plot([U2T * 0.075] * 2, [0.026, -0.026], "o", color="#333333", ms=2.2, zorder=7)
    if labels:
        for name, (x, y) in _LABEL_POS.items():
            ax.text(x, y, REGION_LABELS[name], ha="center", va="center", fontsize=fontsize,
                    color="#555555", zorder=8, style="italic")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect(ASPECT, adjustable="box")
    return ax
