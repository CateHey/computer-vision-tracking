"""Bodily map of social contact, increment 1: located-contacts CSV + passive heatmap.

Usage: python scripts/bodily_map.py <run_dir> [--out <dir>] [--focal R2] [--all]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.bodily_map.locate import (  # noqa: E402
    aggregate_by_region, find_run_files, load_perframe, load_tracks, locate_contacts, slot_color_rgb)

CSV_COLUMNS = ["frame_idx", "time_sec", "focal_mouse", "other_mouse", "direction",
               "contact_type", "region", "t", "s", "dur_sec"]


def write_csv(records, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in records:
            w.writerow([f"{r[c]:.4f}" if isinstance(r[c], float) else r[c] for c in CSV_COLUMNS])


def render_passive(records, focal: str, path: Path) -> int:
    """First-draft passive heatmap: duration-weighted 2D histogram over (t, s)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    rows = [r for r in records if r["focal_mouse"] == focal]
    t = np.array([r["t"] for r in rows])
    s = np.array([r["s"] for r in rows])
    w = np.array([r["dur_sec"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(rows):
        h, _, _ = np.histogram2d(t, s, bins=[50, 36], range=[[0, 1], [-0.45, 0.45]], weights=w)
        im = ax.imshow(h.T, origin="lower", extent=[0, 1, -0.45, 0.45], aspect="auto",
                       cmap="magma", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="contact duration (s)")
    # Crude mouse sketch: body ellipse, nose dot, tail line.
    ax.add_patch(Ellipse((0.32, 0), 0.62, 0.16, fill=False, ec="white", lw=1.5))
    ax.plot([0.63, 1.0], [0, 0], color="white", lw=1.5)
    ax.plot([0.0], [0], "o", color="white", ms=6)
    ax.axhline(0, color="white", lw=0.5, ls=":")
    ax.set_xlabel("t along body (0 = nose, 1 = tail tip)")
    ax.set_ylabel("s (body lengths; + right, - left)")
    color = slot_color_rgb(int(focal[1:]) - 1) if focal[1:].isdigit() else "k"
    ax.set_title(f"Passive contact map, touched mouse {focal} (first draft)")
    ax.plot([0.97], [0.40], "s", color=color, ms=9, clip_on=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return len(rows)


def run_all(records, run_dir, out: Path) -> Path:
    """Generate every passive / active / combined map for all mice, plus index.html."""
    from src.common.bodily_map import render as R
    from src.common.bodily_map.gallery import build_gallery
    mice = sorted({r[k] for r in records for k in ("focal_mouse", "other_mouse")}, key=R.mouse_slot)
    print("mice:", ", ".join(mice))
    written = []
    R.render_template_key(out)
    for a in mice:
        for name, rows, fn in (
                (f"passive_{a}", R.select_passive(records, a), lambda a=a: R.render_passive(records, a, out)),
                (f"active_{a}_combined", R.select_active_combined(records, a),
                 lambda a=a: R.render_active_combined(records, a, out))):
            if rows:
                fn()
                written.append(name)
            else:
                print(f"  skip {name}: no contacts")
        for b in mice:
            if a == b:
                continue
            rows = R.select_active(records, a, b)
            if rows:
                R.render_active(records, a, b, out)
                written.append(f"active_{a}_to_{b}")
                if len(rows) < 20:
                    print(f"  note: {a}->{b} has only {len(rows)} contact frames")
            else:
                print(f"  skip active_{a}_to_{b}: no contacts")
    print(f"wrote {len(written)} PNG maps")
    from src.common.bodily_map.summary import render_all_summaries
    print(f"wrote {len(render_all_summaries(records, out))} summary PNGs")
    return build_gallery(run_dir, records, out, mice)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--focal", default="R2")
    ap.add_argument("--all", action="store_true", help="all maps for all mice + index.html gallery")
    a = ap.parse_args()
    tracks_p, xlsx_p = find_run_files(a.run_dir)
    out = Path(a.out) if a.out else xlsx_p.parent.parent / "bodily_map"
    out.mkdir(parents=True, exist_ok=True)
    meta, frames = load_tracks(tracks_p)
    pf = load_perframe(xlsx_p)
    records, stats = locate_contacts(meta, frames, pf)
    write_csv(records, out / "contacts_located.csv")
    if a.all:
        print("stats:", stats)
        print("gallery written to", run_all(records, a.run_dir, out))
        return
    n = render_passive(records, a.focal, out / f"passive_{a.focal}.png")
    print("stats:", stats)
    print(f"\n{a.focal} passive: {n} rows\n{'region':<12}{'count':>8}{'dur_sec':>10}")
    for reg, v in sorted(aggregate_by_region(records, a.focal).items(), key=lambda kv: -kv[1]["dur_sec"]):
        print(f"{reg:<12}{v['count']:>8}{v['dur_sec']:>10.2f}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
