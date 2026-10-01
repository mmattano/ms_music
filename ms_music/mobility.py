"""Ion mobility as a sound dimension.

Every tone in a sonification can carry a normalized mobility position
``b`` (0 = most compact ions, 1 = most extended, over the data's 2nd–98th
percentile). :class:`MobilityConfig` decides what that position does:

- ``brightness``: overtones grow with mobility (timbre);
- ``pan``: stereo position, compact ions left and extended ions right;
- ``effects``: :class:`MappedEffect` sweeps one parameter of an audio effect
  from a low to a high value across mobility (e.g. reverb wet/dry, so
  extended ions sound farther away).

Use it through ``MSSonifier.sonify(mobility=...)``.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import effects as audio_effects
from .effect_guide import parameter_guide


@dataclass
class MappedEffect:
    """An effect whose parameter ``param`` goes from ``low`` (compact ions)
    to ``high`` (extended ions). ``params`` holds its other, fixed
    settings."""

    effect: str
    param: str
    low: float
    high: float
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        name = (
            self.effect[len("apply_"):]
            if self.effect.startswith("apply_")
            else self.effect
        )
        func = getattr(audio_effects, f"apply_{name}", None)
        if func is None:
            raise ValueError(f"Unknown effect: {self.effect!r}.")
        self.effect = name
        signature = inspect.signature(func)
        if self.param not in signature.parameters:
            mappable = mappable_parameters(name)
            raise ValueError(
                f"Effect {name!r} has no parameter {self.param!r}. Mappable "
                f"settings: {', '.join(mappable) or 'none'}."
            )
        unknown = set(self.params) - set(signature.parameters)
        if unknown:
            raise ValueError(
                f"Effect {name!r} has no parameter(s) {sorted(unknown)}."
            )
        self.low, self.high = float(self.low), float(self.high)
        guide = parameter_guide(name, self.param)
        if guide is not None:
            if not guide.mappable:
                raise ValueError(
                    f"{name} {self.param} can't be mapped to mobility: "
                    f"{guide.why_not}"
                )
            guide.check(self.low, f"{name} {self.param} (low)")
            guide.check(self.high, f"{name} {self.param} (high)")

    @classmethod
    def coerce(cls, value) -> "MappedEffect":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        if isinstance(value, (tuple, list)) and len(value) in (4, 5):
            return cls(*value)
        raise TypeError(
            "A mapped effect is a MappedEffect, a dict or a tuple "
            "(effect, param, low, high[, params])."
        )

    def value_at(self, position: float) -> float:
        return self.low + (self.high - self.low) * float(position)

    def apply(
        self, audio: np.ndarray, sample_rate: int, position: float
    ) -> np.ndarray:
        """Apply the effect (mono input) with the value for ``position``."""
        func = getattr(audio_effects, f"apply_{self.effect}")
        params = dict(self.params)
        value = self.value_at(position)
        spec = inspect.signature(func).parameters[self.param]
        is_int = (
            isinstance(spec.default, int)
            and not isinstance(spec.default, bool)
        ) or spec.annotation in (int, "int")
        if is_int:
            value = int(round(value))
        params[self.param] = value
        return func(audio, sample_rate, **params)


def numeric_parameters(effect: str) -> List[str]:
    """Parameters of an effect whose defaults are numbers (mappable)."""
    func = getattr(audio_effects, f"apply_{effect}")
    params = list(inspect.signature(func).parameters.items())[2:]

    def numeric(p):
        if isinstance(p.default, bool):
            return False
        if isinstance(p.default, (int, float)):
            return True
        # Required parameters count when annotated as a number.
        return p.default is inspect._empty and p.annotation in (
            float,
            int,
            "float",
            "int",
        )

    return [name for name, p in params if numeric(p)]


def mappable_parameters(effect: str) -> List[str]:
    """Numeric settings of an effect that can vary with mobility (those
    changing the audio length or the effect's structure can't)."""
    out = []
    for name in numeric_parameters(effect):
        guide = parameter_guide(effect, name)
        if guide is None or guide.mappable:
            out.append(name)
    return out


@dataclass
class MobilityConfig:
    """What ion mobility does to the sound.

    Args:
        brightness: Overtone strength 0-1 for extended ions (0 = off).
        pan: Stereo width 0-1. Above 0 the output is stereo, with compact
            ions (low mobility) left and extended ions right.
        invert: Swap the direction (high mobility left / pure).
        effects: Mapped effects, applied per mobility band (see
            :class:`MappedEffect`); dicts or tuples are accepted.
        bands: Number of mobility bands the mapped effects run on. Tones are
            crossfaded between the two nearest bands, so there are no steps
            in the sound, only in the effect settings.
        range: (low, high) mobility mapped to 0-1. Default: the data's
            2nd-98th percentile.
    """

    brightness: float = 0.0
    pan: float = 0.0
    invert: bool = False
    effects: List[MappedEffect] = field(default_factory=list)
    bands: int = 6
    range: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        self.brightness = float(np.clip(self.brightness, 0.0, 1.0))
        self.pan = float(self.pan)
        if not 0.0 <= self.pan <= 1.0:
            raise ValueError(f"pan must be in [0, 1], got {self.pan}.")
        self.effects = [MappedEffect.coerce(e) for e in (self.effects or [])]
        self.bands = int(self.bands)
        if self.effects and self.bands < 2:
            raise ValueError("bands must be at least 2 for mapped effects.")
        if self.range is not None:
            lo, hi = map(float, self.range)
            if hi <= lo:
                raise ValueError("range must be (low, high) with high > low.")
            self.range = (lo, hi)

    @classmethod
    def coerce(cls, value) -> Optional["MobilityConfig"]:
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            f"mobility must be a MobilityConfig, a dict or None, "
            f"got {type(value).__name__}."
        )

    @property
    def active(self) -> bool:
        return bool(self.brightness or self.pan or self.effects)

    @property
    def stereo(self) -> bool:
        return self.pan > 0

    def pan_gains(self, b):
        """Equal-power (left, right) gains for mobility position(s) ``b``."""
        b = np.asarray(b, dtype=np.float32)
        if self.invert:
            b = 1.0 - b
        theta = math.pi / 4 + (b - 0.5) * self.pan * (math.pi / 2)
        return np.cos(theta), np.sin(theta)

    def pan_position(self, b):
        """-1 (left) .. +1 (right) for plots."""
        left, right = self.pan_gains(b)
        return (right**2 - left**2) / (right**2 + left**2)


class ToneBus:
    """Where synthesized tones go: mono, stereo, and/or mobility bands.

    ``add(signal, start, b)`` mixes ``signal`` (1-D) in at sample ``start``
    for mobility position ``b`` (a scalar or one value per sample; None =
    centre). ``render()`` applies the mapped effects per band and returns a
    mono ``(n,)`` or stereo ``(2, n)`` float32 array.
    """

    def __init__(self, n_samples: int, config: MobilityConfig):
        self.config = config
        self.n = int(n_samples)
        self.channels = 2 if config.stereo else 1
        self.n_bands = config.bands if config.effects else 1
        self.buf = np.zeros(
            (self.n_bands, self.channels, self.n), dtype=np.float32
        )

    def local(self, n_samples: int) -> "ToneBus":
        """A same-shaped bus for building one block (e.g. an ADSR note)."""
        return ToneBus(n_samples, self.config)

    def add(self, signal, start: int, b=None):
        signal = np.asarray(signal, dtype=np.float32)
        end = start + signal.size
        if b is None:
            b = 0.5
        b = np.clip(
            np.nan_to_num(np.asarray(b, dtype=np.float32), nan=0.5), 0.0, 1.0
        )
        if self.channels == 2:
            left, right = self.config.pan_gains(b)
            channel_signals = (signal * left, signal * right)
        else:
            channel_signals = (signal,)
        if self.n_bands == 1:
            for ch, sig in enumerate(channel_signals):
                self.buf[0, ch, start:end] += sig
            return
        pos = b * (self.n_bands - 1)
        lo_band = int(np.floor(np.min(pos)))
        hi_band = min(int(np.ceil(np.max(pos))), self.n_bands - 1)
        for k in range(lo_band, hi_band + 1):
            weight = np.clip(1.0 - np.abs(pos - k), 0.0, 1.0)
            if not np.any(weight):
                continue
            for ch, sig in enumerate(channel_signals):
                self.buf[k, ch, start:end] += sig * weight

    def add_block(self, block: "ToneBus", start: int, gain=1.0):
        """Mix another bus's buffer in at ``start`` (times ``gain``)."""
        end = start + block.n
        self.buf[:, :, start:end] += block.buf * gain

    def mixdown_peak(self) -> float:
        return (
            float(np.max(np.abs(self.buf.sum(axis=(0, 1))))) if self.n else 0.0
        )

    def render(self, sample_rate: int) -> np.ndarray:
        out = np.zeros((self.channels, self.n), dtype=np.float32)
        for k in range(self.n_bands):
            stem = self.buf[k]
            if not np.any(stem):
                continue
            if self.config.effects:
                position = k / (self.n_bands - 1)
                for effect in self.config.effects:
                    stem = np.stack(
                        [
                            _fit(
                                effect.apply(ch, sample_rate, position), self.n
                            )
                            for ch in stem
                        ]
                    )
            out += stem
        return out[0] if self.channels == 1 else out


def _fit(audio: np.ndarray, n: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size >= n:
        return audio[:n]
    return np.pad(audio, (0, n - audio.size))


PRESETS = {
    "Depth: reverb wet 0.05 → 0.6": MappedEffect(
        "reverb", "dry_wet_mix", 0.05, 0.6, {"reverb_time_s": 1.5}
    ),
    "Filter sweep: lowpass 8 kHz → 1.5 kHz": MappedEffect(
        "lowpass_filter", "cutoff_freq", 8000, 1500
    ),
    "Echo: delay feedback 0.1 → 0.6": MappedEffect(
        "delay", "feedback", 0.1, 0.6, {"delay_ms": 250.0, "dry_wet_mix": 0.4}
    ),
}
