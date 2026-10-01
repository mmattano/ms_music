"""What each numeric effect setting means, and which values make sense.

Used for mobility-mapped effects (:class:`ms_music.mobility.MappedEffect`)
and the GUI: every setting has a plain-language meaning, a unit, a valid
range (enforced), and a suggested starting pair for compact → extended ions.
Settings that change the audio's length or the effect's structure can't be
mapped per mobility band and are marked ``mappable=False``.

Look one up with :func:`parameter_guide` (``parameter_guide("reverb",
"dry_wet_mix").describe()``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ParamGuide:
    """Meaning and sensible values of one effect setting."""

    minimum: float
    maximum: float
    unit: str
    meaning: str
    suggest: Tuple[float, float]  # compact ions -> extended ions
    integer: bool = False
    mappable: bool = True
    why_not: str = ""  # reason when not mappable

    def range_text(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return f"{_fmt(self.minimum)} – {_fmt(self.maximum)}{unit}"

    def describe(self, default=None) -> str:
        if not self.mappable:
            return f"{self.meaning} Not available for mapping: {self.why_not}"
        unit = f" {self.unit}" if self.unit else ""
        parts = [self.meaning, f"Allowed: {self.range_text()}."]
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            parts.append(f"Effect default: {_fmt(default)}{unit}.")
        lo, hi = self.suggest
        parts.append(f"Try {_fmt(lo)} → {_fmt(hi)}{unit}.")
        return " ".join(parts)

    def check(self, value: float, name: str) -> None:
        if not self.minimum <= value <= self.maximum:
            raise ValueError(
                f"{name}: {_fmt(value)} is outside the allowed range "
                f"{self.range_text()}."
            )


def _fmt(value) -> str:
    value = float(value)
    return f"{value:g}"


def G(minimum, maximum, unit, meaning, suggest, integer=False):
    return ParamGuide(minimum, maximum, unit, meaning, suggest, integer)


def FIXED(meaning, why_not):
    return ParamGuide(
        0, 0, "", meaning, (0, 0), mappable=False, why_not=why_not
    )


_STRUCTURE = (
    "it sets up the effect itself, so it has to be the same for all ions."
)
_LENGTH = "it changes the length of the audio, which can't differ per ion."

MIX = G(
    0,
    1,
    "",
    "Mix of original and effect: 0 = only the original sound, "
    "1 = only the effect.",
    (0.05, 0.6),
)
FREQ = G(20, 20000, "Hz", "A frequency.", (500, 3000))
DB_GAIN = G(-24, 24, "dB", "Boost (+) or cut (−) in decibels.", (-6, 6))

# Settings with the same meaning wherever they appear.
GENERIC: Dict[str, ParamGuide] = {
    "dry_wet_mix": MIX,
    "attack_ms": G(
        0.1,
        200,
        "ms",
        "How fast the effect reacts when a sound " "starts.",
        (5, 50),
    ),
    "release_ms": G(
        5,
        2000,
        "ms",
        "How fast the effect lets go after a sound " "stops.",
        (50, 400),
    ),
    "hold_ms": G(
        0,
        1000,
        "ms",
        "How long the gate stays open after the sound " "drops.",
        (10, 200),
    ),
    "threshold_db": G(
        -80,
        0,
        "dB",
        "Level (in decibels below full scale) "
        "where the effect starts to act.",
        (-40, -15),
    ),
    "ratio": G(
        1,
        20,
        ":1",
        "How strongly levels beyond the threshold are "
        "changed (4 means 4:1).",
        (2, 8),
    ),
    "knee_width_db": G(
        0,
        24,
        "dB",
        "How gently the effect starts around the "
        "threshold (larger = softer).",
        (2, 12),
    ),
    "makeup_gain_db": G(
        0, 24, "dB", "Volume added back after compression.", (0, 6)
    ),
    "lookahead_ms": G(
        0,
        20,
        "ms",
        "How far ahead the limiter looks to catch " "peaks.",
        (1, 10),
    ),
    "feedback": G(
        0,
        0.95,
        "",
        "How much of the output is fed back in: more "
        "gives longer, stronger repeats or resonance.",
        (0.1, 0.6),
    ),
    "rate_hz": G(
        0.05,
        20,
        "Hz",
        "Speed of the movement, in cycles per " "second.",
        (0.3, 3),
    ),
    "depth": G(
        0,
        1,
        "",
        "How strong the movement is (0 = none, 1 = " "full).",
        (0.2, 0.9),
    ),
    "depth_ms": G(
        0.5,
        10,
        "ms",
        "How far the delay time sweeps back and " "forth.",
        (1, 5),
    ),
    "cutoff_freq": G(
        20,
        20000,
        "Hz",
        "Frequency where the filter starts " "cutting.",
        (500, 3000),
    ),
    "low_freq": G(
        20, 20000, "Hz", "Lower edge of the filter band.", (200, 800)
    ),
    "high_freq": G(
        20, 20000, "Hz", "Upper edge of the filter band.", (4000, 1500)
    ),
    "output_level": G(
        0, 1, "", "Output volume after the distortion.", (0.3, 0.7)
    ),
    "drive": G(
        1, 20, "", "How hard the sound is pushed into distortion.", (2, 10)
    ),
    "order": FIXED(
        "Steepness of the filter (higher = sharper cut).", _STRUCTURE
    ),
    "n_fft": FIXED("Analysis window size.", _STRUCTURE),
}

# Effect-specific settings, or shared names that mean something else here.
SPECIFIC: Dict[Tuple[str, str], ParamGuide] = {
    # Filters
    ("lowpass_filter", "cutoff_freq"): G(
        20,
        20000,
        "Hz",
        "Frequencies above this are removed: lower sounds "
        "darker and more muffled.",
        (8000, 1500),
    ),
    ("highpass_filter", "cutoff_freq"): G(
        20,
        20000,
        "Hz",
        "Frequencies below this are removed: higher sounds "
        "thinner and brighter.",
        (80, 800),
    ),
    ("butterworth_filter", "cutoff_freq"): G(
        20,
        20000,
        "Hz",
        "Frequency where the filter starts cutting.",
        (4000, 1000),
    ),
    ("chebyshev1_filter", "cutoff_freq"): G(
        20,
        20000,
        "Hz",
        "Frequency where the filter starts cutting.",
        (4000, 1000),
    ),
    ("chebyshev1_filter", "ripple_db"): G(
        0.1,
        5,
        "dB",
        "Unevenness allowed in the kept range, in exchange for "
        "a steeper cut.",
        (0.5, 2),
    ),
    ("bandstop_filter", "low_freq"): G(
        20, 20000, "Hz", "Lower edge of the removed band.", (400, 800)
    ),
    ("bandstop_filter", "high_freq"): G(
        20, 20000, "Hz", "Upper edge of the removed band.", (1200, 3000)
    ),
    ("notch_filter", "notch_freq"): G(
        20, 20000, "Hz", "The single frequency that is cut out.", (500, 3000)
    ),
    ("notch_filter", "quality_factor"): G(
        1, 100, "", "How narrow the notch is (higher = narrower).", (5, 30)
    ),
    # Reverbs
    ("reverb", "reverb_time_s"): G(
        0.1,
        10,
        "s",
        "How long the reverb rings after the sound (T60).",
        (0.5, 3),
    ),
    ("reverb", "decay_factor"): G(
        1,
        15,
        "",
        "How steeply the reverb tail fades (higher = shorter " "tail).",
        (8, 4),
    ),
    ("reverb", "room_size"): G(
        0, 1, "", "Size of the simulated room.", (0.2, 0.9)
    ),
    ("reverb", "damping"): G(
        0,
        1,
        "",
        "How fast high frequencies die away in the reverb "
        "(1 = dull, soft room).",
        (0.2, 0.8),
    ),
    # Delays and modulation
    ("delay", "delay_ms"): G(
        10, 2000, "ms", "Time until each repeat.", (120, 450)
    ),
    ("delay", "num_taps"): FIXED(
        "Number of separate delay lines.", _STRUCTURE
    ),
    ("echo", "echo_delay_ms"): G(
        20, 2000, "ms", "Time between echoes.", (150, 600)
    ),
    ("echo", "echo_gain"): G(
        0,
        0.95,
        "",
        "Loudness of each echo compared with the one before.",
        (0.2, 0.6),
    ),
    ("echo", "num_echoes"): G(
        1, 10, "", "Number of echoes.", (1, 5), integer=True
    ),
    ("chorus", "delay_ms"): G(
        5, 50, "ms", "Base delay of the chorus voices.", (10, 30)
    ),
    ("chorus", "feedback"): G(
        0,
        0.9,
        "",
        "How much of the output is fed back: more gives a "
        "stronger, ringing chorus.",
        (0.1, 0.5),
    ),
    ("chorus", "num_voices"): G(
        1,
        8,
        "",
        "Number of chorus voices (more = thicker).",
        (1, 5),
        integer=True,
    ),
    ("flanger", "delay_ms"): G(
        0.5, 15, "ms", "Base delay of the flanger.", (2, 8)
    ),
    ("flanger", "feedback"): G(
        -0.95,
        0.95,
        "",
        "Resonance of the sweep; negative values sound " "hollow.",
        (0.1, 0.7),
    ),
    ("phaser", "feedback"): G(
        -0.95,
        0.95,
        "",
        "Resonance of the sweep; negative values sound " "hollow.",
        (0.1, 0.7),
    ),
    ("phaser", "stages"): FIXED("Number of phase-shift stages.", _STRUCTURE),
    ("tremolo", "rate_hz"): G(
        0.5, 20, "Hz", "How many times per second the volume wobbles.", (2, 8)
    ),
    ("tremolo", "depth"): G(
        0, 1, "", "How strongly the volume wobbles.", (0.1, 0.8)
    ),
    ("vibrato", "rate_hz"): G(
        0.5, 15, "Hz", "How many times per second the pitch wobbles.", (2, 7)
    ),
    ("vibrato", "depth_cents"): G(
        0,
        200,
        "cents",
        "How far the pitch wobbles (100 cents = one " "semitone).",
        (10, 80),
    ),
    ("ring_modulation", "frequency_hz"): G(
        1,
        2000,
        "Hz",
        "Frequency of the modulator: low sounds like a "
        "tremolo, high like bells or metal.",
        (30, 600),
    ),
    ("ring_modulation", "depth"): G(
        0,
        1,
        "",
        "How much of the metallic ring-modulated sound is heard.",
        (0.2, 0.9),
    ),
    ("auto_wah", "sensitivity"): G(
        0.1, 10, "", "How strongly loudness opens the wah.", (0.5, 3)
    ),
    ("auto_wah", "resonance"): G(
        0.5, 10, "", "Sharpness of the wah peak.", (1, 5)
    ),
    # Distortion
    ("overdrive", "drive"): G(
        1, 20, "", "How hard the sound is pushed into distortion.", (2, 10)
    ),
    ("overdrive", "tone"): G(
        0,
        1,
        "",
        "Brightness of the distortion (0 = dark, 1 = bright).",
        (0.7, 0.3),
    ),
    ("fuzz", "fuzz_amount"): G(
        1, 50, "", "Strength of the fuzz distortion.", (3, 20)
    ),
    ("fuzz", "gate_threshold"): G(
        0,
        1,
        "",
        "Quiet parts below this are silenced before the fuzz.",
        (0.05, 0.2),
    ),
    ("waveshaper", "drive"): G(
        1, 10, "", "How hard the sound is pushed through the shaper.", (2, 6)
    ),
    ("waveshaper", "symmetry"): G(
        -1,
        1,
        "",
        "Asymmetry of the distortion (adds even harmonics, a "
        "warmer sound).",
        (0, 0.6),
    ),
    ("bitcrusher", "bit_depth"): G(
        2,
        16,
        "bits",
        "Fewer bits sound grittier and more lo-fi.",
        (12, 4),
        integer=True,
    ),
    ("bitcrusher", "downsample_factor"): G(
        1,
        32,
        "",
        "Keeps only every Nth sample (higher = more crunch).",
        (1, 8),
        integer=True,
    ),
    # Dynamics
    ("compressor", "ratio"): G(
        1,
        20,
        ":1",
        "How strongly loud parts are turned down (4 means " "4:1).",
        (2, 8),
    ),
    ("expander", "ratio"): G(
        1,
        10,
        ":1",
        "How strongly quiet parts are turned further down.",
        (1.5, 4),
    ),
    ("gate", "ratio"): G(
        1,
        100,
        ":1",
        "How strongly sound below the threshold is muted.",
        (5, 30),
    ),
    ("limiter", "threshold_db"): G(
        -20, 0, "dB", "Peak level the limiter keeps the sound under.", (-6, -1)
    ),
    ("multiband_compressor", "num_bands"): FIXED(
        "Number of frequency bands.", _STRUCTURE
    ),
    ("normalize", "target_db"): G(
        -40, 0, "dB", "Peak level the sound is scaled to.", (-12, -3)
    ),
    ("fade_in", "fade_duration_ms"): G(
        0, 10000, "ms", "Length of the fade-in.", (200, 2000)
    ),
    ("fade_out", "fade_duration_ms"): G(
        0, 10000, "ms", "Length of the fade-out.", (200, 2000)
    ),
    # EQ
    ("parametric_eq", "frequency_hz"): G(
        20, 20000, "Hz", "Centre of the boosted or cut band.", (300, 3000)
    ),
    ("parametric_eq", "gain_db"): DB_GAIN,
    ("parametric_eq", "q_factor"): G(
        0.1, 20, "", "Width of the band: low = broad, high = narrow.", (0.7, 4)
    ),
    ("shelving_eq", "low_shelf_freq"): G(
        20, 1000, "Hz", "Frequency below which the low shelf acts.", (100, 300)
    ),
    ("shelving_eq", "low_shelf_gain_db"): G(
        -24, 24, "dB", "Boost (+) or cut (−) of the lows.", (0, 6)
    ),
    ("shelving_eq", "high_shelf_freq"): G(
        1000,
        20000,
        "Hz",
        "Frequency above which the high shelf acts.",
        (5000, 10000),
    ),
    ("shelving_eq", "high_shelf_gain_db"): G(
        -24, 24, "dB", "Boost (+) or cut (−) of the highs.", (3, -9)
    ),
    # Pitch and time
    ("pitch_shift", "n_steps"): G(
        -24, 24, "semitones", "Transposes the sound (12 = one octave).", (0, 7)
    ),
    ("formant_shift", "shift_semitones"): G(
        -12,
        12,
        "semitones",
        "Moves the vowel-like colour up or down "
        "without changing the pitch.",
        (0, -4),
    ),
    ("formant_shift", "formant_correction"): G(
        0, 1, "", "How much of the original colour is kept.", (1, 0.5)
    ),
    ("time_stretch", "rate"): FIXED("Playback speed.", _LENGTH),
    ("granular_synthesis", "grain_size_ms"): G(
        5,
        500,
        "ms",
        "Length of each sound grain (short = buzzy, long = " "smeared).",
        (30, 150),
    ),
    ("granular_synthesis", "grain_density"): G(
        0.1, 10, "", "How many grains overlap (more = smoother cloud).", (1, 4)
    ),
    ("granular_synthesis", "pitch_variation_semitones"): G(
        0, 12, "semitones", "Random pitch spread of the grains.", (0, 3)
    ),
    ("granular_synthesis", "time_stretch_ratio"): FIXED(
        "Stretches the sound in time.", _LENGTH
    ),
    ("granular_synthesis", "position_randomness"): G(
        0, 1, "", "How scattered the grains are in time.", (0, 0.5)
    ),
    ("hpss", "margin"): FIXED("Separation strictness.", _STRUCTURE),
    # Adaptive / tracking filters
    ("adaptive_filter", "filter_length"): FIXED(
        "Length of the adaptive filter.", _STRUCTURE
    ),
    ("adaptive_filter", "mu"): G(
        0.0001,
        0.1,
        "",
        "Learning speed of the adaptive filter.",
        (0.005, 0.05),
    ),
    ("pll_filter", "center_freq"): G(
        50, 5000, "Hz", "Frequency the tracker starts from.", (300, 1500)
    ),
    ("pll_filter", "damping_factor"): G(
        0.1,
        2,
        "",
        "Stability of the tracking (0.707 is balanced).",
        (0.5, 1.0),
    ),
    ("pll_filter", "loop_bandwidth_hz"): G(
        1,
        200,
        "Hz",
        "How quickly the tracker follows pitch changes.",
        (10, 60),
    ),
    ("pll_filter", "output_filter_freq_offset_hz"): G(
        -1000,
        1000,
        "Hz",
        "Offset of the output filter from the tracked " "pitch.",
        (0, 200),
    ),
    ("pll_filter", "output_filter_min_freq_hz"): G(
        20,
        2000,
        "Hz",
        "Lowest frequency the output filter may go to.",
        (20, 200),
    ),
    ("spectral_compressor", "n_fft"): FIXED(
        "Analysis window size.", _STRUCTURE
    ),
}


def parameter_guide(effect: str, param: str) -> Optional[ParamGuide]:
    """Guide for ``param`` of ``effect`` (with or without ``apply_``)."""
    if effect.startswith("apply_"):
        effect = effect[len("apply_"):]
    return SPECIFIC.get((effect, param)) or GENERIC.get(param)
