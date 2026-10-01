"""Beat-grid rhythm for audio sonification.

MIDI export has always snapped notes to a meter. This module brings the same
idea to audio: instead of giving every scan an equal slice of time, scans are
regrouped onto the steps of a musical grid (e.g. sixteenth notes in 3/4 at
96 BPM) and the total length snaps to whole bars. Each pitch then plays as
notes that follow its signal: a note starts on the grid step where the
pitch's intensity appears and lasts for as many steps as the signal does,
like MIDI notes whose length comes from the peak width. Downbeat emphasis
(``accent``), feel (swing or humanized timing) and a short gap before the
next note (``gate``) shape the notes.

Use it through ``MSSonifier.sonify(rhythm=...)``; the helpers here are
independent of the synthesis method, so they work for gradient and ADSR,
with or without scale quantization.

Tempo is in quarter-note beats per minute, matching the MIDI generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .musical_quantization import (
    MusicMeter,
    QuantizationMode,
    _coerce_meter,
    _coerce_quant_mode,
)

_AGGREGATES = ("max", "mean", "sum")
_SUBDIVISIONS = (4, 8, 16, 32)

# Envelope edge lengths (seconds); both shrink for very short steps.
_ATTACK_S = 0.004
_RELEASE_S = 0.012
# Humanized onsets move by up to this fraction of a step.
_HUMANIZE_FRACTION = 0.08


@dataclass
class RhythmConfig:
    """How to lay a sonification onto a musical grid.

    Args:
        meter: Time signature, as a ``MusicMeter`` or a string like ``"3/4"``.
        tempo: Quarter-note beats per minute.
        subdivision: Grid resolution as a note value (4 = quarters,
            8 = eighths, 16 = sixteenths, 32). Must be at least the meter's
            note value (e.g. 8 or finer for 6/8).
        mode: Timing feel: ``STRICT_GRID`` (straight), ``SWING`` or
            ``HUMANIZED``. ``ADAPTIVE`` is accepted and treated as straight.
        swing_ratio: Share of each quarter note taken by its first half when
            swinging (0.5 = straight, 0.67 = triplet swing).
        accent: Extra gain for notes starting on the first step of a bar;
            notes starting on other beats get half of it. 0 disables accents.
        gate: Fraction of a note's last step that sounds (1.0 = legato into
            the next note; smaller values leave a gap). It never cuts notes
            into steps.
        note_threshold: A pitch sounds while its intensity is at least this
            fraction of its own peak; when it drops below, the note ends.
            Higher values give shorter, more separated notes.
        aggregate: How several scans falling into one step combine:
            ``"max"``, ``"mean"`` or ``"sum"`` of intensities per m/z.
        seed: Random seed for humanized timing (None = random each time).
    """

    meter: Union[MusicMeter, str] = MusicMeter.FOUR_FOUR
    tempo: float = 120.0
    subdivision: int = 16
    mode: Union[QuantizationMode, str] = QuantizationMode.STRICT_GRID
    swing_ratio: float = 0.67
    accent: float = 0.4
    gate: float = 0.9
    note_threshold: float = 0.05
    aggregate: str = "max"
    seed: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.meter, str):
            self.meter = _coerce_meter(self.meter)
        if isinstance(self.mode, str):
            self.mode = _coerce_quant_mode(self.mode)
        self.tempo = float(self.tempo)
        self.subdivision = int(self.subdivision)
        if self.tempo <= 0:
            raise ValueError(f"tempo must be positive, got {self.tempo}.")
        if self.subdivision not in _SUBDIVISIONS:
            raise ValueError(
                f"subdivision must be one of {_SUBDIVISIONS}, "
                f"got {self.subdivision}."
            )
        if self.subdivision < self.meter.note_value:
            raise ValueError(
                f"subdivision {self.subdivision} is coarser than the "
                f"meter's note value ({self.meter.beats_per_measure}/"
                f"{self.meter.note_value}); use {self.meter.note_value} "
                f"or finer."
            )
        if not 0.0 < self.gate <= 1.0:
            raise ValueError(f"gate must be in (0, 1], got {self.gate}.")
        if not 0.0 <= self.note_threshold < 1.0:
            raise ValueError(
                f"note_threshold must be in [0, 1), got {self.note_threshold}."
            )
        if self.accent < 0:
            raise ValueError(f"accent must be >= 0, got {self.accent}.")
        if not 0.5 <= self.swing_ratio < 1.0:
            raise ValueError(
                f"swing_ratio must be in [0.5, 1), got {self.swing_ratio}."
            )
        if self.aggregate not in _AGGREGATES:
            raise ValueError(
                f"aggregate must be one of {_AGGREGATES}, "
                f"got {self.aggregate!r}."
            )

    @classmethod
    def coerce(cls, value) -> Optional["RhythmConfig"]:
        """None -> None, dict -> RhythmConfig(**dict), config -> itself."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            f"rhythm must be a RhythmConfig, a dict or None, "
            f"got {type(value).__name__}."
        )

    @property
    def quarter_seconds(self) -> float:
        return 60.0 / self.tempo

    @property
    def bar_seconds(self) -> float:
        m = self.meter
        return (
            m.beats_per_measure * (4.0 / m.note_value) * self.quarter_seconds
        )

    @property
    def steps_per_bar(self) -> int:
        m = self.meter
        return m.beats_per_measure * self.subdivision // m.note_value

    def describe(self) -> str:
        m = self.meter
        return f"{m.beats_per_measure}/{m.note_value} · " f"{self.tempo:g} BPM"


@dataclass
class RhythmGrid:
    """A concrete grid: ``n_bars`` bars of ``config.steps_per_bar`` steps.

    ``onsets`` are the (possibly swung/humanized) start times of each step in
    seconds; step ``k`` lasts until ``onsets[k + 1]`` (or ``total_seconds``).
    """

    config: RhythmConfig
    n_bars: int
    onsets: np.ndarray
    accents: np.ndarray
    total_seconds: float
    bar_times: np.ndarray = field(repr=False)

    @property
    def n_steps(self) -> int:
        return len(self.onsets)

    @property
    def step_ends(self) -> np.ndarray:
        return np.append(self.onsets[1:], self.total_seconds)

    def segment_lengths(self, sample_rate: int) -> np.ndarray:
        """Samples per step. They sum to the grid's total sample count."""
        bounds = np.round(
            np.append(self.onsets, self.total_seconds) * sample_rate
        ).astype(np.int64)
        return np.diff(bounds)

    def describe(self) -> str:
        return f"{self.config.describe()} · {self.n_bars} bars"


def build_grid(duration_seconds: float, config: RhythmConfig) -> RhythmGrid:
    """Snap ``duration_seconds`` to whole bars and lay out the step grid."""
    cfg = config
    n_bars = max(1, int(round(duration_seconds / cfg.bar_seconds)))
    n_steps = n_bars * cfg.steps_per_bar
    step_s = cfg.bar_seconds / cfg.steps_per_bar
    total = n_bars * cfg.bar_seconds
    straight = np.arange(n_steps) * step_s

    onsets = straight
    if cfg.mode == QuantizationMode.SWING and cfg.subdivision >= 8:
        # Time-warp inside each quarter note: its first half is stretched to
        # swing_ratio of the quarter, the second half squeezed into the rest.
        q = cfg.quarter_seconds
        base = np.floor(straight / q + 1e-9) * q
        p = (straight - base) / q
        r = cfg.swing_ratio
        warped = np.where(p < 0.5, p * 2 * r, r + (p - 0.5) * 2 * (1 - r))
        onsets = base + warped * q
    elif cfg.mode == QuantizationMode.HUMANIZED:
        rng = np.random.default_rng(cfg.seed)
        jitter = rng.uniform(-1, 1, n_steps) * _HUMANIZE_FRACTION * step_s
        jitter[0] = 0.0  # keep the very first onset at t=0
        onsets = straight + jitter
        # Keep onsets ordered and inside the grid.
        onsets = np.clip(onsets, 0.0, total - 1e-6)
        onsets = np.maximum.accumulate(onsets)

    # Accents: bar downbeat > other beat starts (pulse groups of three note
    # values in compound meters like 6/8) > everything else.
    steps_per_note_value = cfg.subdivision // cfg.meter.note_value
    pulse = steps_per_note_value * (3 if cfg.meter.is_compound else 1)
    pos = np.arange(n_steps) % cfg.steps_per_bar
    accents = np.ones(n_steps, dtype=np.float32)
    accents[pos % pulse == 0] = 1.0 + cfg.accent / 2
    accents[pos == 0] = 1.0 + cfg.accent

    return RhythmGrid(
        config=cfg,
        n_bars=n_bars,
        onsets=onsets,
        accents=accents,
        total_seconds=total,
        bar_times=np.arange(n_bars) * cfg.bar_seconds,
    )


def resample_spectra(
    spectra_dfs: List[pd.DataFrame],
    n_steps: int,
    aggregate: str = "max",
    return_members: bool = False,
):
    """Regroup per-scan spectra into ``n_steps`` consecutive steps.

    Returns ``(step_dfs, step_to_scan)``, plus ``members`` (the scan indices
    of each step) when ``return_members`` is set. With more scans than
    steps, the scans inside each step are combined per m/z with
    ``aggregate``; with fewer, each step holds the nearest scan.
    ``step_to_scan[k]`` is the scan that represents step ``k``. A
    ``mobility`` column is combined as the intensity-weighted mean.
    """
    n_scans = len(spectra_dfs)
    if n_scans == 0 or n_steps <= 0:
        return ([], [], []) if return_members else ([], [])

    # Scan i belongs to step floor(i * n_steps / n_scans); owners are sorted,
    # so each step's scans are one contiguous range.
    owner = (np.arange(n_scans) * n_steps) // n_scans
    starts = np.searchsorted(owner, np.arange(n_steps), side="left")
    ends = np.searchsorted(owner, np.arange(n_steps), side="right")
    step_dfs: List[pd.DataFrame] = []
    step_to_scan: List[int] = []
    members_out: List[np.ndarray] = []
    for k in range(n_steps):
        members = np.arange(starts[k], ends[k])
        if members.size == 0:
            # Fewer scans than steps: hold the scan whose span covers k.
            nearest = min(int((k + 0.5) * n_scans / n_steps), n_scans - 1)
            step_dfs.append(spectra_dfs[nearest])
            step_to_scan.append(nearest)
            members_out.append(np.array([nearest]))
            continue
        step_to_scan.append(int(members[members.size // 2]))
        members_out.append(members)
        if members.size == 1:
            step_dfs.append(spectra_dfs[members[0]])
            continue
        parts = [spectra_dfs[i] for i in members if not spectra_dfs[i].empty]
        if not parts:
            step_dfs.append(spectra_dfs[members[0]])
            continue
        step_dfs.append(_combine(parts, aggregate))
    if return_members:
        return step_dfs, step_to_scan, members_out
    return step_dfs, step_to_scan


def _combine(parts: List[pd.DataFrame], aggregate: str) -> pd.DataFrame:
    combined = pd.concat(parts)
    grouped = combined["intensities"].groupby(level=0)
    out = getattr(grouped, aggregate)().to_frame("intensities")
    if "mobility" in combined.columns:
        weight = combined["intensities"].to_numpy()
        weighted = (
            pd.Series(
                combined["mobility"].to_numpy() * weight, index=combined.index
            )
            .groupby(level=0)
            .sum()
        )
        total = pd.Series(weight, index=combined.index).groupby(level=0).sum()
        out["mobility"] = (weighted / total.where(total > 0)).reindex(
            out.index
        )
    return out


def find_notes(
    levels: np.ndarray, threshold: float = 0.05
) -> List[Tuple[int, int]]:
    """Split one pitch's per-step levels into notes.

    A note is a run of consecutive steps whose level is above
    ``threshold`` times the pitch's peak level. Returns ``[(start, end)]``
    step ranges with ``end`` exclusive.
    """
    levels = np.asarray(levels, dtype=float)
    peak = levels.max() if levels.size else 0.0
    if peak <= 0:
        return []
    on = levels > threshold * peak
    edges = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def note_span(grid: RhythmGrid, start: int, end: int) -> Tuple[float, float]:
    """Start and end time (s) of a note covering steps ``start..end-1``;
    only its last step is shortened by ``gate``."""
    last = end - 1
    t0 = float(grid.onsets[start])
    t1 = float(
        grid.onsets[last]
        + grid.config.gate * (grid.step_ends[last] - grid.onsets[last])
    )
    return t0, t1


def note_gain(
    grid: RhythmGrid, levels: np.ndarray, start: int, end: int, t: np.ndarray
) -> np.ndarray:
    """Gain curve over sample times ``t`` for one gradient-style note.

    The level glides between step centres (no steps, no re-attacks), with a
    short attack and release only at the note's ends. The whole note is
    scaled by the accent of the step it starts on.
    """
    t0, t1 = note_span(grid, start, end)
    length = max(t1 - t0, 1e-6)
    attack = min(_ATTACK_S, length * 0.25)
    release = min(_RELEASE_S, length * 0.25)
    accent = float(grid.accents[start])
    steps = np.arange(start, end)
    mids = (grid.onsets[steps] + grid.step_ends[steps]) / 2
    vals = np.asarray(levels, dtype=float)[steps] * accent
    inside = (mids > t0 + attack) & (mids < t1 - release)
    xs = np.concatenate(([t0, t0 + attack], mids[inside], [t1 - release, t1]))
    ys = np.concatenate(([0.0, vals[0]], vals[inside], [vals[-1], 0.0]))
    return np.interp(t, xs, ys, left=0.0, right=0.0)
