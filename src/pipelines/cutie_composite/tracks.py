"""Run history for cutie_composite: tracks.jsonl.gz and the area plot (no torch).

tracks.jsonl.gz holds one JSON line per frame, after a first `meta` line:

    {"f": 1830, "t": 61.0, "phase": "cutie",
     "rats": [{"id": 1, "area": 4120, "n_parts": 1, "c": [512.3, 401.7], "src": "cutie",
               "rle": "<COCO compressed counts>", "kp": [[x, y, conf], ...], "kp_carry": false}, ...],
     "extra": [{"rle": "...", "area": 310, "score": 0.62}],
     "events": [...]}

- `rle` is a COCO run-length encoding (column-major, compressed counts; the size is
  [height, width] from the meta line). `rle_decode` here or pycocotools read it.
- `n_parts` counts the separate pieces of an animal's mask (2+ = split mask).
- `src`: sam3 (SAM3 phase), cutie, recovered (SAM3 recovery replaced CUTIE's mask),
  carry (mask carried over from the previous frame), null (no mask).
- `extra`: SAM3 masks that were not given to any animal (the extra detections of the
  SAM3 phase and of each recovery attempt).
- `kp`: the 7 keypoints in the order of meta.keypoint_names; kp_carry = carried over.

The file is gzip-flushed every few hundred frames, so a run that dies keeps what it wrote.
"""
from __future__ import annotations

import gzip
import json
import logging
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "cutie_composite tracks v1"


# ---------------------------------------------------------------------- COCO RLE
def rle_counts(mask: np.ndarray) -> List[int]:
    """Column-major run lengths, starting with a run of zeros (COCO convention)."""
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    if flat.size == 0:
        return [0]
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return runs


def _counts_to_string(counts: Sequence[int]) -> str:
    out = []
    for i, c in enumerate(counts):
        x = int(c)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            ch = x & 0x1F
            x >>= 5
            more = (x != -1) if (ch & 0x10) else (x != 0)
            if more:
                ch |= 0x20
            out.append(chr(ch + 48))
    return "".join(out)


def _string_to_counts(s: str) -> List[int]:
    counts: List[int] = []
    p = 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def rle_encode(mask: np.ndarray) -> str:
    return _counts_to_string(rle_counts(mask))


def rle_decode(rle: str, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in _string_to_counts(rle):
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape((h, w), order="F")


def count_parts(mask: np.ndarray, min_part_px: int) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_part_px))


# ---------------------------------------------------------------------- writer
class TracksWriter:
    def __init__(self, path: Path, meta: dict, min_part_px: int = 20, flush_every: int = 300):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_part_px = int(min_part_px)
        self.flush_every = int(flush_every)
        self.shape = (int(meta["height"]), int(meta["width"]))
        self._fh = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=6)
        self._fh.write(json.dumps({"meta": {"format": FORMAT, **meta}}) + "\n")
        self.frames = 0

    def write_frame(self, frame: int, fps: float, phase: str,
                    slot_masks: Sequence[Optional[np.ndarray]],
                    slot_centroids: Sequence[Optional[Tuple[float, float]]],
                    sources: Sequence[Optional[str]],
                    slot_detections: Sequence,
                    kp_carried: Sequence[bool],
                    extras: Sequence[Tuple[np.ndarray, Optional[float]]] = (),
                    events: Sequence[dict] = ()) -> None:
        rats = []
        for s, m in enumerate(slot_masks):
            has = m is not None and bool(np.any(m))
            c = slot_centroids[s] if s < len(slot_centroids) else None
            det = slot_detections[s] if s < len(slot_detections) else None
            kp = None
            if det is not None and getattr(det, "keypoints", None):
                kp = [[round(float(k.x), 1), round(float(k.y), 1), round(float(k.conf), 3)] for k in det.keypoints]
            rats.append({
                "id": s + 1,
                "area": int(m.sum()) if has else 0,
                "n_parts": count_parts(m, self.min_part_px) if has else 0,
                "c": [round(float(c[0]), 1), round(float(c[1]), 1)] if (has and c is not None) else None,
                "src": (sources[s] if s < len(sources) else None) if has else None,
                "rle": rle_encode(m) if has else None,
                "kp": kp,
                "kp_carry": bool(kp_carried[s]) if (kp is not None and s < len(kp_carried)) else False,
            })
        line = {"f": int(frame), "t": round(frame / fps, 3) if fps else None, "phase": phase, "rats": rats}
        if extras:
            line["extra"] = [{"rle": rle_encode(m), "area": int(m.sum()),
                              "score": None if sc is None else round(float(sc), 3)} for m, sc in extras]
        if events:
            line["events"] = list(events)
        self._fh.write(json.dumps(line, separators=(",", ":")) + "\n")
        self.frames += 1
        if self.flush_every and self.frames % self.flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            size_mb = self.path.stat().st_size / 1e6
            logger.info("Tracks written: %s (%d frames, %.1f MB)", self.path, self.frames, size_mb)


def read_tracks(path: Path) -> Iterator[dict]:
    """Yield the meta line, then one dict per frame. A truncated file (killed run) stops cleanly."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        try:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        except (EOFError, zlib.error, json.JSONDecodeError):
            logger.warning("Tracks file %s ends early (run interrupted?)", path)


# ---------------------------------------------------------------------- area plot
_OUTCOME_STYLE = {  # fill colour, label
    "recovered": ("#8fd19e", "recovered"),
    "verified": ("#c9c9c9", "verified (CUTIE was right)"),
    "cleared": ("#a9c4e8", "cleared by itself"),
    "unresolved": ("#f2a3a3", "unresolved"),
}


def plot_area(frames: Sequence[int], ratios: Sequence[Sequence[float]], fps: float,
              episodes: Sequence[dict], colors_bgr: Sequence[Tuple[int, int, int]],
              low: float, high: float, out_png: Path, title: str = "") -> Optional[Path]:
    """One panel per animal: mask area / that animal's median, with the flag band and episodes."""
    if not frames:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    r = np.asarray(ratios, dtype=float)
    t = np.asarray(frames, dtype=float) / (fps or 30.0) / 60.0
    n = r.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.35 * n + 1.0), sharex=True, squeeze=False)
    ymax = max(2.2, min(4.0, np.nanmax(r) + 0.2 if np.isfinite(r).any() else 2.2))
    for s in range(n):
        ax = axes[s][0]
        b, g, rr = colors_bgr[s % len(colors_bgr)]
        line_color = (0.78 * rr / 255, 0.78 * g / 255, 0.78 * b / 255)  # darkened identity colour
        for ep in episodes:
            if s + 1 not in ep.get("slots", []):
                continue
            fill = _OUTCOME_STYLE.get(ep.get("outcome"), ("#dddddd", ""))[0]
            onset = ep.get("onsets", {}).get(str(s + 1), ep["start"])
            end = ep["end"] if ep.get("end") is not None else frames[-1]
            ax.axvspan(onset / fps / 60.0, end / fps / 60.0, color=fill, alpha=0.6, lw=0)
            ax.axvline(ep["start"] / fps / 60.0, color="#555555", lw=0.8, ls=":")
        ax.axhline(1.0, color="#bbbbbb", lw=0.8)
        ax.axhline(low, color="#888888", lw=0.8, ls="--")
        ax.axhline(high, color="#888888", lw=0.8, ls="--")
        ax.plot(t, r[:, s], color=line_color, lw=1.0)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"R{s + 1}", rotation=0, ha="right", va="center", fontsize=10)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#eeeeee", lw=0.6)
    axes[-1][0].set_xlabel("time (min)")
    handles = [Patch(color=c, alpha=0.6, label=lab) for k, (c, lab) in _OUTCOME_STYLE.items()
               if any(ep.get("outcome") == k for ep in episodes)]
    head = (title + "\n" if title else "") + \
        f"Mask area / own median (dashed: flag at {low:.2f} and {high:.2f}; shaded: onset to end of an episode, dotted: flag raised)"
    fig.suptitle(head, fontsize=9, x=0.01, ha="left")
    if handles:
        fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False, ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
