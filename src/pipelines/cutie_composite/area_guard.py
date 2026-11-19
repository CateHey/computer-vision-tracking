"""Area guard for cutie_composite (pure logic, no torch).

CUTIE can let one animal's mask swallow a neighbour when two identical animals touch:
one mask grows towards two bodies while the other shrinks to a tail or disappears.
The guard watches each animal's mask area against that animal's own recent median
and raises a single flag when the area leaves the normal range:

    area < low_ratio  x median   (being eaten, or lost)
    area > high_ratio x median   (eating a neighbour, or leaking into the background)
    no mask at all               (lost)

The flag must hold for `sustain_frames` frames in a row. Small partial absorptions that
come back are ignored on purpose: they do not change identities.

Episodes: the first triggered frame opens an episode and asks for a recovery attempt;
a failed attempt is retried every `retry_fast_every` frames during `retry_fast_seconds`,
then every `retry_slow_every` frames, for as long as the flag holds. An attempt that
finds CUTIE was right ("verified", e.g. an animal rearing) pauses those animals for
`verified_cooldown_frames`.

The onset of each episode (the frame where the drift started, found backwards as the
frame after the last healthy one) is kept for the log and the area plot, and each
animal's last healthy mask is kept as the reference for re-assigning identities.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

KIND_LOW = "low"
KIND_HIGH = "high"
KIND_MISSING = "missing"


@dataclass
class GuardConfig:
    low_ratio: float = 0.40
    high_ratio: float = 1.60
    sustain_frames: int = 3
    baseline_seconds: float = 3.0
    min_baseline_samples: int = 15
    healthy_band: Tuple[float, float] = (0.80, 1.25)
    retry_fast_every: int = 3
    retry_fast_seconds: float = 5.0
    retry_slow_every: int = 10
    verified_cooldown_frames: int = 30

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "GuardConfig":
        cfg = cfg or {}
        d = cls()
        band = cfg.get("healthy_band", list(d.healthy_band))
        return cls(
            low_ratio=float(cfg.get("low_ratio", d.low_ratio)),
            high_ratio=float(cfg.get("high_ratio", d.high_ratio)),
            sustain_frames=int(cfg.get("sustain_frames", d.sustain_frames)),
            baseline_seconds=float(cfg.get("baseline_seconds", d.baseline_seconds)),
            min_baseline_samples=int(cfg.get("min_baseline_samples", d.min_baseline_samples)),
            healthy_band=(float(band[0]), float(band[1])),
            retry_fast_every=int(cfg.get("retry_fast_every", d.retry_fast_every)),
            retry_fast_seconds=float(cfg.get("retry_fast_seconds", d.retry_fast_seconds)),
            retry_slow_every=int(cfg.get("retry_slow_every", d.retry_slow_every)),
            verified_cooldown_frames=int(cfg.get("verified_cooldown_frames", d.verified_cooldown_frames)),
        )


@dataclass
class HealthyRef:
    """Last frame where an animal's mask area was inside the healthy band."""
    frame: int
    mask: np.ndarray
    centroid: Optional[Tuple[float, float]]
    velocity: Tuple[float, float] = (0.0, 0.0)   # px / frame, from the two last healthy frames


@dataclass
class GuardStatus:
    frame: int
    areas: List[int]
    ratios: List[float]                  # area / median (nan while the median is unknown)
    kinds: Dict[int, str]                # slot -> low / high / missing, this frame
    triggered: Dict[int, str]            # slot -> kind, held for sustain_frames and not paused
    onsets: Dict[int, int]               # triggered slot -> first frame after its last healthy frame


@dataclass
class Episode:
    start: int
    onsets: Dict[int, int]
    kinds: Dict[int, str]
    attempts: int = 0
    next_attempt: int = 0
    end: Optional[int] = None
    outcome: Optional[str] = None        # recovered / verified / cleared / unresolved
    details: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "outcome": self.outcome,
                "attempts": self.attempts, "slots": sorted(int(s) + 1 for s in self.kinds),
                "kinds": {str(int(s) + 1): k for s, k in sorted(self.kinds.items())},
                "onsets": {str(int(s) + 1): int(f) for s, f in sorted(self.onsets.items())}}


class AreaGuard:
    def __init__(self, num_slots: int, cfg: GuardConfig, fps: float):
        self.n = int(num_slots)
        self.cfg = cfg
        self.fps = float(fps) if fps else 30.0
        maxlen = max(int(round(cfg.baseline_seconds * self.fps)), cfg.min_baseline_samples)
        self._hist = [deque(maxlen=maxlen) for _ in range(self.n)]
        self._consec = [0] * self.n
        self._cooldown_until = [-1] * self.n
        self._ever_seen = [False] * self.n
        self.healthy: List[Optional[HealthyRef]] = [None] * self.n
        self.episode: Optional[Episode] = None
        self.episodes: List[Episode] = []
        self.ratio_series: List[List[float]] = []   # one row per observed frame
        self.frame_series: List[int] = []

    # ------------------------------------------------------------------ baseline
    def baseline(self, slot: int) -> Optional[float]:
        h = self._hist[slot]
        if len(h) < self.cfg.min_baseline_samples:
            return None
        return float(np.median(h))

    def baselines(self) -> List[Optional[float]]:
        return [self.baseline(s) for s in range(self.n)]

    # ------------------------------------------------------------------ per frame
    def update(self, frame: int, slot_masks: Sequence[Optional[np.ndarray]],
               slot_centroids: Sequence[Optional[Tuple[float, float]]], act: bool = True) -> GuardStatus:
        """Observe one frame of masks. act=False only feeds the medians (SAM3 phase)."""
        areas, ratios, kinds = [], [], {}
        lo_h, hi_h = self.cfg.healthy_band
        for s in range(self.n):
            m = slot_masks[s] if s < len(slot_masks) else None
            area = int(m.sum()) if m is not None else 0
            areas.append(area)
            base = self.baseline(s)
            ratio = (area / base) if base else float("nan")
            ratios.append(ratio)

            kind = None
            if area == 0:
                if self._ever_seen[s]:
                    kind = KIND_MISSING
            elif base is not None:
                if ratio < self.cfg.low_ratio:
                    kind = KIND_LOW
                elif ratio > self.cfg.high_ratio:
                    kind = KIND_HIGH
            if area > 0:
                self._ever_seen[s] = True
            if kind is not None:
                kinds[s] = kind

            healthy = area > 0 and kind is None and (base is None or lo_h <= ratio <= hi_h)
            if healthy:
                self._hist[s].append(area)
                prev = self.healthy[s]
                c = slot_centroids[s] if s < len(slot_centroids) else None
                vel = (0.0, 0.0)
                if prev is not None and prev.centroid is not None and c is not None and frame > prev.frame:
                    dt = frame - prev.frame
                    vel = ((c[0] - prev.centroid[0]) / dt, (c[1] - prev.centroid[1]) / dt)
                self.healthy[s] = HealthyRef(frame=frame, mask=m.copy(), centroid=c, velocity=vel)

            if act:
                self._consec[s] = self._consec[s] + 1 if kind is not None else 0

        self.ratio_series.append(ratios)
        self.frame_series.append(frame)

        triggered, onsets = {}, {}
        if act:
            for s, k in kinds.items():
                if self._consec[s] >= self.cfg.sustain_frames and frame >= self._cooldown_until[s]:
                    triggered[s] = k
                    ref = self.healthy[s]
                    onsets[s] = (ref.frame + 1) if ref is not None else frame
        return GuardStatus(frame=frame, areas=areas, ratios=ratios, kinds=kinds,
                           triggered=triggered, onsets=onsets)

    # ------------------------------------------------------------------ episodes
    def should_attempt(self, status: GuardStatus) -> bool:
        """True when a recovery attempt should run on this frame."""
        frame = status.frame
        if not status.triggered:
            if self.episode is not None:
                self._close(frame, "cleared", "flag dropped without a recovery")
            return False
        if self.episode is None:
            self.episode = Episode(start=frame, onsets=dict(status.onsets),
                                   kinds=dict(status.triggered), next_attempt=frame)
            logger.info("Area guard: frame %d, %s (onset %s)", frame,
                        ", ".join(f"R{s + 1} {k}" for s, k in sorted(status.triggered.items())),
                        ", ".join(f"R{s + 1}@{f}" for s, f in sorted(status.onsets.items())))
            return True
        for s, k in status.triggered.items():          # an animal can join a running episode
            self.episode.kinds.setdefault(s, k)
            self.episode.onsets.setdefault(s, status.onsets.get(s, frame))
        return frame >= self.episode.next_attempt

    def record_attempt(self, frame: int, outcome: str, detail: str = "",
                       slots: Optional[Sequence[int]] = None) -> None:
        """outcome: recovered / verified / failed."""
        ep = self.episode
        if ep is None:
            return
        ep.attempts += 1
        if detail:
            ep.details.append(f"{frame}: {outcome} {detail}")
        involved = list(slots) if slots is not None else list(ep.kinds)
        if outcome in ("recovered", "verified"):
            for s in involved:
                self._consec[s] = 0
                if outcome == "verified":
                    self._cooldown_until[s] = frame + self.cfg.verified_cooldown_frames
            self._close(frame, outcome, detail)
            return
        fast_window = int(round(self.cfg.retry_fast_seconds * self.fps))
        step = self.cfg.retry_fast_every if (frame - ep.start) < fast_window else self.cfg.retry_slow_every
        ep.next_attempt = frame + max(1, step)

    def finalize(self, frame: int) -> None:
        if self.episode is not None:
            self._close(frame, "unresolved", "video ended with the flag up")

    def _close(self, frame: int, outcome: str, detail: str) -> None:
        ep = self.episode
        ep.end = frame
        ep.outcome = outcome
        self.episodes.append(ep)
        self.episode = None
        logger.info("Area guard: episode %d-%d %s after %d attempt(s) (%s)",
                    ep.start, frame, outcome, ep.attempts,
                    ", ".join(f"R{s + 1}" for s in sorted(ep.kinds)))

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ep in self.episodes:
            out[ep.outcome] = out.get(ep.outcome, 0) + 1
        return out
