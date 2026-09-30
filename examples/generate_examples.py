"""
generate_examples.py
====================
Generates a complete set of MS_Music output examples covering every package
feature: sonification methods, tuning systems, effects, MIDI, and
visualizations.

Usage
-----
    python examples/generate_examples.py <ms1_mzml_file> [<ms2_mzml_file>]

    ms1_mzml_file  – path to an mzML file with MS1 data (required)
    ms2_mzml_file  – path to an mzML file with MS2 data (optional).
                     When omitted, MS2 examples are skipped entirely.
                     For files that contain both MS1 and MS2, pass the
                     same path twice.

Outputs
-------
    examples/output/          WAV and MIDI files
    examples/output/plots/    PNG figures
    examples/output/videos/   MP4 animations
"""

import sys
import os
import traceback

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print(
        "Usage: python examples/generate_examples.py "
        "<ms1_mzml_file> [<ms2_mzml_file>]"
    )
    sys.exit(1)

MS1_FILE = sys.argv[1]
MS2_FILE = sys.argv[2] if len(sys.argv) > 2 else None
HAS_MS2 = MS2_FILE is not None

# Next to this script, wherever it is run from.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
for _d in (OUTPUT_DIR, PLOT_DIR, VIDEO_DIR):
    os.makedirs(_d, exist_ok=True)


def out(name):
    return os.path.join(OUTPUT_DIR, name)


def plot(name):
    return os.path.join(PLOT_DIR, name)


def vout(name):
    return os.path.join(VIDEO_DIR, name)


# ── package imports ───────────────────────────────────────────────────────────
from ms_music import (
    MSSonifier,
    MSSonifierMidi,
    MidiConfig,
    MusicMeter,
    QuantizationMode,
    load_mzml_data,
    preprocess_spectra,
    effects,
    visualizations,
)
from ms_music.musical_quantization import MusicalNoteQuantizer
from ms_music.io import normalize_audio_to_16bit, save_wav


# ── helpers ───────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def safe(label, fn):
    try:
        fn()
        print(f"  [ok]   {label}")
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()


def savefig(fig, name):
    if fig is not None:
        fig.savefig(plot(name), dpi=100, bbox_inches="tight")
        plt.close(fig)


def fresh_sonifier(filepath, ms_level=1):
    """Return a loaded MSSonifier for the given file."""
    s = MSSonifier(
        filepath, ms_level=ms_level, total_duration_minutes=DURATION_MIN
    )
    s.load_and_preprocess_data(scan_ratio_range=SCAN_RANGE)
    return s


def _ms2_center_scan_range(filepath, window_minutes=0.5):
    """Return (start_ratio, end_ratio) for the middle window_minutes of the file."""
    raw = load_mzml_data(filepath, ms_level=2)
    if not raw:
        return (0.35, 0.65)
    rts = [
        s["retention_time"] for s in raw if s.get("retention_time") is not None
    ]
    if len(rts) < 2:
        return (0.35, 0.65)
    rt_min, rt_max = min(rts), max(rts)
    mid = (rt_min + rt_max) / 2
    half = window_minutes / 2
    start = max(rt_min, mid - half)
    end = min(rt_max, mid + half)
    span = rt_max - rt_min
    return (start - rt_min) / span, (end - rt_min) / span


def fresh_ms2_sonifier(filepath):
    """Return MS2 MSSonifier for the middle 30-second window."""
    s = MSSonifier(
        filepath, ms_level=2, total_duration_minutes=MS2_DURATION_MIN
    )
    s.load_and_preprocess_data(scan_ratio_range=MS2_SCAN_RANGE)
    return s


# ── constants ─────────────────────────────────────────────────────────────────
DURATION_MIN = 0.5  # keep MS1 examples short
MS2_DURATION_MIN = 0.5  # 30-second songs from the middle window
SCAN_RANGE = (0.0, 1.0)
MS2_SCAN_RANGE = _ms2_center_scan_range(MS2_FILE) if HAS_MS2 else (0.35, 0.65)
FREQ_RANGE = (200.0, 4000.0)
MIDI_DURATION = 120.0  # seconds
MS2_MAX_PEAKS = 30  # top-N fragment ions per scan


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RAW IO
# ══════════════════════════════════════════════════════════════════════════════
section("1. Direct IO — load_mzml_data / preprocess_spectra")

raw_ms1 = load_mzml_data(MS1_FILE, ms_level=1)
assert raw_ms1, "No MS1 spectra found in the provided file."

processed_dfs, max_int, min_mz, max_mz = preprocess_spectra(raw_ms1)
print(
    f"  {len(processed_dfs)} MS1 spectra | m/z {min_mz:.0f}–{max_mz:.0f} | "
    f"max intensity {max_int:.2e}"
)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BASE MS1 SONIFIER  (shared across sections 3–12)
# ══════════════════════════════════════════════════════════════════════════════
section("2. Base MS1 sonifier")

sonifier = fresh_sonifier(MS1_FILE, ms_level=1)
sonifier.sonify(
    "gradient", frequency_mapping="inverse_log", freq_range=FREQ_RANGE
)
BASE_AUDIO = sonifier.get_current_audio()
print(
    f"  Base audio: {len(BASE_AUDIO) / sonifier.sample_rate:.1f}s "
    f"@ {sonifier.sample_rate} Hz"
)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GRADIENT METHOD — all frequency mappings
# ══════════════════════════════════════════════════════════════════════════════
section("3. Gradient sonification — frequency mappings")

for mapping in ("inverse_log", "power_law", "musical_octaves", "chromatic"):
    safe(
        f"gradient/{mapping}",
        lambda m=mapping: (
            sonifier.sonify(
                "gradient", frequency_mapping=m, freq_range=FREQ_RANGE
            ),
            sonifier.save_audio(out(f"audio_gradient_{m}.wav")),
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ADSR METHOD
# ══════════════════════════════════════════════════════════════════════════════
section("4. ADSR sonification")

safe(
    "adsr/randomized",
    lambda: (
        sonifier.sonify(
            "adsr",
            frequency_mapping="inverse_log",
            freq_range=FREQ_RANGE,
            adsr_settings={"randomize": True},
        ),
        sonifier.save_audio(out("audio_adsr.wav")),
    ),
)

safe(
    "adsr/fixed",
    lambda: (
        sonifier.sonify(
            "adsr",
            frequency_mapping="power_law",
            freq_range=FREQ_RANGE,
            adsr_settings={
                "randomize": False,
                "attack_time_pc": 0.05,
                "decay_time_pc": 0.10,
                "sustain_level_pc": 0.70,
                "release_time_pc": 0.15,
            },
        ),
        sonifier.save_audio(out("audio_adsr_fixed.wav")),
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  QUANTIZED SONIFICATION — 12-TET scales
# ══════════════════════════════════════════════════════════════════════════════
section("5. Quantized sonification — 12-TET scales")

SCALES = [
    ("chromatic", "C"),
    ("major", "C"),
    ("minor", "A"),
    ("pentatonic_major", "G"),
    ("pentatonic_minor", "A"),
    ("blues", "E"),
    ("dorian", "D"),
    ("whole_tone", "C"),
]

for scale, root in SCALES:

    def _quantized(sc=scale, rt=root):
        sonifier.sonify(
            frequency_mapping="inverse_log",
            freq_range=(200, 3000),
            scale=sc,
            root_note=rt,
        )
        sonifier.save_audio(out(f"audio_quantized_{sc}_{rt}.wav"))

    safe(f"quantized/{scale}/{root}", _quantized)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EDO TUNING SYSTEMS
# ══════════════════════════════════════════════════════════════════════════════
section("6. EDO tuning systems")

# Each entry: (scale_name, root_note, edo_divisions, output_tag)
EDO_CONFIGS = [
    ("19_edo_diatonic", "C", 19, "19_edo_diatonic"),
    ("24_edo", "C", 24, "24_edo"),
    ("31_edo_major", "C", 31, "31_edo_major"),
    ("arabic_maqam_hijaz", "C", 24, "arabic_maqam_hijaz"),
]

for sc, rt, div, tag in EDO_CONFIGS:

    def _edo(scale=sc, root=rt, edo=div, out_tag=tag):
        sonifier.sonify(
            frequency_mapping="inverse_log",
            freq_range=(200, 3000),
            scale=scale,
            root_note=root,
            edo_divisions=edo,
        )
        sonifier.save_audio(out(f"audio_edo_{out_tag}.wav"))

    safe(f"edo/{tag}", _edo)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  JUST INTONATION
# ══════════════════════════════════════════════════════════════════════════════
section("7. Just intonation")

JUST_CONFIGS = [
    ("just_major", "C"),
    ("just_minor", "A"),
    ("pythagorean_major", "C"),
    ("quarter_comma_meantone", "C"),
]

for sc, rt in JUST_CONFIGS:

    def _just(scale=sc, root=rt):
        sonifier.sonify(
            frequency_mapping="inverse_log",
            freq_range=(200, 3000),
            scale=scale,
            root_note=root,
            use_just_intonation=True,
        )
        sonifier.save_audio(out(f"audio_just_{scale}_{root}.wav"))

    safe(f"just_intonation/{sc}", _just)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MS2 SONIFICATION — DDA and DIA modes
# ══════════════════════════════════════════════════════════════════════════════
section("8. MS2 sonification — DDA and DIA modes")

if not HAS_MS2:
    print("  Skipped — no MS2 file provided.")
    print(
        "  Pass a second file path to enable: "
        "python generate_examples.py ms1.mzML ms2.mzML"
    )
else:

    def _ms2_fragments_only():
        s2 = fresh_ms2_sonifier(MS2_FILE)
        n = len(s2.processed_spectra_dfs or [])
        print(f"    {n} MS2 scans | " f"{os.path.basename(MS2_FILE)}")
        s2.sonify(
            "adsr",
            frequency_mapping="inverse_log",
            freq_range=FREQ_RANGE,
            adsr_settings={"randomize": True},
            max_peaks_per_scan=MS2_MAX_PEAKS,
        )
        s2.save_audio(out("audio_ms2_fragments_only.wav"))

    safe("ms2/fragments_only", _ms2_fragments_only)

    def _ms2_dda():
        s2 = fresh_ms2_sonifier(MS2_FILE)
        n_pre = sum(1 for p in (s2.precursor_mz_list or []) if p is not None)
        print(
            f"    {n_pre}/{len(s2.precursor_mz_list or [])} "
            f"scans have a selected precursor m/z (DDA)"
        )

        # ADSR + precursor tone
        s2.sonify(
            "adsr",
            frequency_mapping="inverse_log",
            freq_range=FREQ_RANGE,
            adsr_settings={"randomize": True},
            ms2_mode="dda",
            max_peaks_per_scan=MS2_MAX_PEAKS,
        )
        s2.save_audio(out("audio_ms2_dda_adsr.wav"))

        # Quantized pentatonic A minor + precursor tone
        s2.sonify(
            frequency_mapping="inverse_log",
            freq_range=(200, 3000),
            scale="pentatonic_minor",
            root_note="A",
            ms2_mode="dda",
            max_peaks_per_scan=MS2_MAX_PEAKS,
        )
        s2.save_audio(out("audio_ms2_dda_quantized.wav"))

    safe("ms2/dda", _ms2_dda)

    def _ms2_dia():
        s2 = fresh_ms2_sonifier(MS2_FILE)
        s2.load_ms1_reference(MS1_FILE)

        n_win = sum(
            1 for w in (s2.isolation_window_list or []) if w is not None
        )
        n_ref = len(s2.ms1_reference_spectra or [])
        print(
            f"    {n_win}/{len(s2.isolation_window_list or [])} "
            f"scans have an isolation window (DIA)"
        )
        print(f"    {n_ref} MS1 reference spectra for lookup")

        # ADSR DIA
        s2.sonify(
            "adsr",
            frequency_mapping="inverse_log",
            freq_range=FREQ_RANGE,
            adsr_settings={"randomize": True},
            ms2_mode="dia",
            max_peaks_per_scan=MS2_MAX_PEAKS,
        )
        s2.save_audio(out("audio_ms2_dia_adsr.wav"))

        # Quantized pentatonic A minor DIA
        s2.sonify(
            frequency_mapping="inverse_log",
            freq_range=(200, 3000),
            scale="pentatonic_minor",
            root_note="A",
            ms2_mode="dia",
            max_peaks_per_scan=MS2_MAX_PEAKS,
        )
        s2.save_audio(out("audio_ms2_dia_quantized.wav"))

    safe("ms2/dia", _ms2_dia)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  EFFECTS  (applied to BASE_AUDIO)
# ══════════════════════════════════════════════════════════════════════════════
section("9. Effects")


def apply_and_save(effect_name, params, filename):
    sonifier.current_audio_data = BASE_AUDIO.copy()
    sonifier.apply_effect(effect_name, params)
    sonifier.save_audio(out(filename))


EFFECTS = [
    # Filters
    ("lowpass_filter", {"cutoff_freq": 2000}, "effect_lowpass_filter.wav"),
    ("highpass_filter", {"cutoff_freq": 500}, "effect_highpass_filter.wav"),
    (
        "bandpass_filter",
        {"low_freq": 400, "high_freq": 3000},
        "effect_bandpass_filter.wav",
    ),
    (
        "notch_filter",
        {"notch_freq": 1000, "quality_factor": 10},
        "effect_notch_filter.wav",
    ),
    # Time-based
    (
        "reverb",
        {"reverb_time_s": 0.8, "dry_wet_mix": 0.4},
        "effect_reverb.wav",
    ),
    (
        "delay",
        {"delay_ms": 250, "feedback": 0.4, "num_taps": 3, "dry_wet_mix": 0.4},
        "effect_delay.wav",
    ),
    (
        "chorus",
        {"delay_ms": 20, "depth_ms": 2, "rate_hz": 0.7, "dry_wet_mix": 0.5},
        "effect_chorus.wav",
    ),
    (
        "flanger",
        {"delay_ms": 5, "depth_ms": 3, "rate_hz": 0.3, "feedback": 0.4},
        "effect_flanger.wav",
    ),
    (
        "phaser",
        {"rate_hz": 0.5, "stages": 4, "feedback": 0.3, "dry_wet_mix": 0.5},
        "effect_phaser.wav",
    ),
    # Dynamics
    (
        "compressor",
        {"threshold_db": -20, "ratio": 4.0, "makeup_gain_db": 3.0},
        "effect_compressor.wav",
    ),
    ("gate", {"threshold_db": -40, "ratio": 10.0}, "effect_gate.wav"),
    ("limiter", {"threshold_db": -3.0}, "effect_limiter.wav"),
    ("expander", {"threshold_db": -30, "ratio": 2.0}, "effect_expander.wav"),
    # Distortion
    ("overdrive", {"drive": 5.0, "tone": 0.6}, "effect_overdrive.wav"),
    ("fuzz", {"fuzz_amount": 8.0}, "effect_fuzz.wav"),
    (
        "bitcrusher",
        {"bit_depth": 8, "downsample_factor": 2},
        "effect_bitcrusher.wav",
    ),
    (
        "waveshaper",
        {"curve_type": "tanh", "drive": 3.0},
        "effect_waveshaper.wav",
    ),
    # Modulation
    ("tremolo", {"rate_hz": 5.0, "depth": 0.6}, "effect_tremolo.wav"),
    ("vibrato", {"rate_hz": 4.0, "depth_cents": 50.0}, "effect_vibrato.wav"),
    (
        "ring_modulation",
        {"frequency_hz": 440.0, "depth": 0.8},
        "effect_ring_modulation.wav",
    ),
    (
        "auto_wah",
        {"sensitivity": 1.0, "range_hz": (200, 2000), "resonance": 2.0},
        "effect_auto_wah.wav",
    ),
    # Spectral
    (
        "hpss",
        {"harmonic": True, "percussive": False},
        "effect_hpss_harmonic.wav",
    ),
    ("spectral_gate", {"threshold_db": -30.0}, "effect_spectral_gate.wav"),
    (
        "spectral_compressor",
        {"threshold_db": -20, "ratio": 4.0},
        "effect_spectral_compressor.wav",
    ),
    ("pitch_shift", {"n_steps": 3}, "effect_pitch_shift.wav"),
    ("time_stretch", {"rate": 0.8}, "effect_time_stretch.wav"),
    ("formant_shift", {"shift_semitones": 2.0}, "effect_formant_shift.wav"),
    # Other
    (
        "pll_filter",
        {"center_freq": 500.0, "loop_bandwidth_hz": 20.0},
        "effect_pll_filter.wav",
    ),
    (
        "adaptive_filter",
        {"filter_length": 32, "mu": 0.01, "target_type": "noise_reduction"},
        "effect_adaptive_filter.wav",
    ),
    (
        "granular_synthesis",
        {
            "grain_size_ms": 50.0,
            "grain_density": 1.0,
            "time_stretch_ratio": 1.2,
        },
        "effect_granular_synthesis.wav",
    ),
    (
        "convolution_reverb",
        {"reverb_type": "hall", "dry_wet_mix": 0.3},
        "effect_convolution_reverb.wav",
    ),
    (
        "multiband_compressor",
        {"num_bands": 4},
        "effect_multiband_compressor.wav",
    ),
    # EQ
    (
        "parametric_eq",
        {
            "frequency_hz": 1000,
            "gain_db": 6.0,
            "q_factor": 1.4,
            "filter_type": "peak",
        },
        "effect_parametric_eq.wav",
    ),
    (
        "graphic_eq",
        {"gains_db": [3, -3, 6, -2, 0, 2, -4, 3, 1, -2]},
        "effect_graphic_eq.wav",
    ),
    (
        "shelving_eq",
        {
            "low_shelf_freq": 200,
            "low_shelf_gain_db": 4.0,
            "high_shelf_freq": 6000,
            "high_shelf_gain_db": -3.0,
        },
        "effect_shelving_eq.wav",
    ),
    # Utility
    ("normalize", {"target_db": -6.0, "mode": "rms"}, "effect_normalize.wav"),
    ("fade_in", {"fade_duration_ms": 500}, "effect_fade_in.wav"),
    ("fade_out", {"fade_duration_ms": 500}, "effect_fade_out.wav"),
]

for effect_name, params, filename in EFFECTS:
    safe(
        f"effects/{effect_name}",
        lambda n=effect_name, p=params, f=filename: apply_and_save(n, p, f),
    )


# Effect chain
def _chain():
    sonifier.current_audio_data = BASE_AUDIO.copy()
    sonifier.apply_effect("highpass_filter", {"cutoff_freq": 200})
    sonifier.apply_effect("compressor", {"threshold_db": -18, "ratio": 3.0})
    sonifier.apply_effect("reverb", {"reverb_time_s": 0.6, "dry_wet_mix": 0.3})
    sonifier.apply_effect(
        "delay",
        {"delay_ms": 200, "feedback": 0.3, "num_taps": 2, "dry_wet_mix": 0.25},
    )
    sonifier.apply_effect("limiter", {"threshold_db": -1.0})
    sonifier.save_audio(out("effect_chain.wav"))


safe("effects/chain", _chain)

# Direct calls to effects module functions (bypass apply_effect wrapper)
section("9b. Effects module — direct function calls")

audio_f32 = BASE_AUDIO.astype(np.float32)
SR = sonifier.sample_rate


def _direct(fn, outfile):
    save_wav(out(outfile), normalize_audio_to_16bit(fn()), SR)


safe(
    "effects_direct/crossfade",
    lambda: _direct(
        lambda: effects.apply_crossfade(
            audio_f32, audio_f32[::-1], SR, crossfade_duration_ms=500
        ),
        "effect_crossfade.wav",
    ),
)

safe(
    "effects_direct/echo",
    lambda: _direct(
        lambda: effects.apply_echo(
            audio_f32, SR, echo_delay_ms=300, echo_gain=0.5, num_echoes=3
        ),
        "effect_echo.wav",
    ),
)

safe(
    "effects_direct/hpss_percussive",
    lambda: _direct(
        lambda: effects.apply_hpss(
            audio_f32, SR, harmonic=False, percussive=True
        ),
        "effect_hpss_percussive.wav",
    ),
)

safe(
    "effects_direct/butterworth",
    lambda: _direct(
        lambda: effects.apply_butterworth_filter(
            audio_f32, SR, cutoff_freq=3000, btype="low", order=6
        ),
        "effect_butterworth.wav",
    ),
)

safe(
    "effects_direct/chebyshev1",
    lambda: _direct(
        lambda: effects.apply_chebyshev1_filter(
            audio_f32, SR, cutoff_freq=2000, ripple_db=1.0, btype="low"
        ),
        "effect_chebyshev1.wav",
    ),
)

safe(
    "effects_direct/bandstop",
    lambda: _direct(
        lambda: effects.apply_bandstop_filter(
            audio_f32, SR, low_freq=900, high_freq=1100
        ),
        "effect_bandstop.wav",
    ),
)

safe(
    "effects_direct/waveshaper_arctan",
    lambda: _direct(
        lambda: effects.apply_waveshaper(
            audio_f32, SR, curve_type="arctan", drive=4.0
        ),
        "effect_waveshaper_arctan.wav",
    ),
)

safe(
    "effects_direct/tremolo_square",
    lambda: _direct(
        lambda: effects.apply_tremolo(
            audio_f32, SR, rate_hz=6.0, depth=0.7, waveform="square"
        ),
        "effect_tremolo_square.wav",
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# 10. SCAN-RANGE SUBSETS
# ══════════════════════════════════════════════════════════════════════════════
section("10. Scan-range subsets")

for start, end, label in (
    (0.0, 0.33, "early"),
    (0.33, 0.66, "middle"),
    (0.66, 1.0, "late"),
):

    def _range(s=start, e=end, lbl=label):
        sr = MSSonifier(
            MS1_FILE, ms_level=1, total_duration_minutes=DURATION_MIN
        )
        sr.load_and_preprocess_data(scan_ratio_range=(s, e))
        sr.sonify(
            "gradient", frequency_mapping="inverse_log", freq_range=FREQ_RANGE
        )
        sr.save_audio(out(f"audio_range_{lbl}.wav"))

    safe(f"scan_range/{label}", _range)


# ══════════════════════════════════════════════════════════════════════════════
# 11. MIDI GENERATION
# ══════════════════════════════════════════════════════════════════════════════
section("11. MIDI generation")

MIDI_CONFIGS = [
    # (label, scale, root, tempo, meter, quantization_mode, clustering)
    (
        "classical",
        "major",
        "C",
        120,
        MusicMeter.FOUR_FOUR,
        QuantizationMode.STRICT_GRID,
        True,
    ),
    (
        "jazz",
        "dorian",
        "D",
        140,
        MusicMeter.FOUR_FOUR,
        QuantizationMode.SWING,
        True,
    ),
    (
        "waltz",
        "minor",
        "A",
        90,
        MusicMeter.THREE_FOUR,
        QuantizationMode.HUMANIZED,
        True,
    ),
    (
        "experimental",
        "arabic_maqam_hijaz",
        "C",
        100,
        MusicMeter.FIVE_FOUR,
        QuantizationMode.ADAPTIVE,
        True,
    ),
    (
        "pentatonic",
        "pentatonic_major",
        "G",
        110,
        MusicMeter.FOUR_FOUR,
        QuantizationMode.STRICT_GRID,
        True,
    ),
    (
        "blues_12bar",
        "blues",
        "E",
        80,
        MusicMeter.TWELVE_EIGHT,
        QuantizationMode.SWING,
        True,
    ),
    (
        "microtonal_19",
        "19_edo_diatonic",
        "C",
        120,
        MusicMeter.FOUR_FOUR,
        QuantizationMode.STRICT_GRID,
        True,
    ),
]

for lbl, sc, rt, tp, mt, qm, cl in MIDI_CONFIGS:

    def _midi(
        label=lbl, scale=sc, root=rt, tempo=tp, meter=mt, qmode=qm, cluster=cl
    ):
        cfg = MidiConfig(
            scale=scale,
            root_note=root,
            tempo=tempo,
            meter=meter,
            quantization_mode=qmode,
        )
        midi = MSSonifierMidi(
            MS1_FILE,
            config=cfg,
            enable_mz_clustering=cluster,
            mz_tolerance_ppm=30.0,
        )
        midi.load_and_analyze_data(total_duration_seconds=MIDI_DURATION)
        midi.setup_musical_system(
            scale=scale,
            root_note=root,
            tempo=tempo,
            meter=meter,
            freq_range=(130.81, 2093.0),
            quantization_mode=qmode,
        )
        midi.detect_and_quantize_peaks(
            intensity_threshold_percentile=90.0,
            max_peaks_per_scan=1000,
            frequency_mapping="inverse_log",
            apply_mz_clustering=cluster,
        )
        midi.generate_midi_file(
            out(f"midi_{label}.mid"),
            track_name=f"MS Sonification - {label}",
            instrument=1,
            max_simultaneous_notes=4,
            export_note_data=True,
        )
        r = midi.get_analysis_report()
        print(
            f"    {label}: {r['note_count']} notes, "
            f"{r['unique_pitches']} pitches, "
            f"{r['total_duration_seconds']:.1f}s"
        )

    safe(f"midi/{lbl}", _midi)


# Multi-voice
def _multivoice():
    cfg = MidiConfig(
        scale="major",
        root_note="C",
        tempo=120,
        quantization_mode=QuantizationMode.STRICT_GRID,
    )
    midi = MSSonifierMidi(
        MS1_FILE, config=cfg, enable_mz_clustering=True, mz_tolerance_ppm=30.0
    )
    midi.load_and_analyze_data(total_duration_seconds=MIDI_DURATION)
    midi.setup_musical_system(freq_range=(130.81, 2093.0))
    midi.detect_and_quantize_peaks(
        intensity_threshold_percentile=90.0, max_peaks_per_scan=1000
    )
    midi.generate_midi_file(
        out("midi_multivoice_4.mid"),
        num_voices=4,
        instrument=1,
        max_simultaneous_notes=4,
        export_note_data=True,
    )


safe("midi/multi_voice_4", _multivoice)


# Separate files per voice
def _separate_voices():
    cfg = MidiConfig(scale="pentatonic_minor", root_note="A", tempo=100)
    midi = MSSonifierMidi(
        MS1_FILE, config=cfg, enable_mz_clustering=True, mz_tolerance_ppm=30.0
    )
    midi.load_and_analyze_data(total_duration_seconds=MIDI_DURATION)
    midi.setup_musical_system(freq_range=(130.81, 2093.0))
    midi.detect_and_quantize_peaks(
        intensity_threshold_percentile=90.0, max_peaks_per_scan=1000
    )
    midi.generate_midi_file(
        out("midi_voices_separate.mid"),
        num_voices=3,
        max_simultaneous_notes=4,
        separate_files=True,
    )


safe("midi/separate_voice_files", _separate_voices)


# Frequency deduplication
def _freq_dedup():
    cfg = MidiConfig(scale="chromatic", root_note="C", tempo=120)
    midi = MSSonifierMidi(
        MS1_FILE, config=cfg, enable_mz_clustering=True, mz_tolerance_ppm=30.0
    )
    midi.load_and_analyze_data(total_duration_seconds=MIDI_DURATION)
    midi.setup_musical_system(freq_range=(130.81, 2093.0))
    midi.detect_and_quantize_peaks(
        intensity_threshold_percentile=90.0,
        max_peaks_per_scan=1000,
        apply_frequency_deduplication=True,
        frequency_tolerance_hz=5.0,
    )
    midi.generate_midi_file(
        out("midi_deduped.mid"),
        max_simultaneous_notes=4,
        export_note_data=True,
    )


safe("midi/frequency_deduplication", _freq_dedup)


# ══════════════════════════════════════════════════════════════════════════════
# 12. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("12. Visualizations")

# Build a small dict of named audio versions for comparison plots.
AUDIO_DICT = {}
for _m in ("inverse_log", "power_law", "chromatic"):
    sonifier.sonify("gradient", frequency_mapping=_m, freq_range=FREQ_RANGE)
    AUDIO_DICT[_m] = sonifier.get_current_audio()

SR = sonifier.sample_rate
a1 = AUDIO_DICT["inverse_log"]
a2 = AUDIO_DICT["power_law"]

safe(
    "viz/waveforms",
    lambda: savefig(
        visualizations.plot_waveform_comparison(AUDIO_DICT, SR),
        "waveforms.png",
    ),
)
safe(
    "viz/spectrograms",
    lambda: savefig(
        visualizations.plot_spectrogram_comparison(AUDIO_DICT, SR),
        "spectrograms.png",
    ),
)
safe(
    "viz/freq_spectrum",
    lambda: savefig(
        visualizations.plot_frequency_spectrum_comparison(AUDIO_DICT, SR),
        "frequency_spectrum.png",
    ),
)
safe(
    "viz/diff_spectrogram",
    lambda: savefig(
        visualizations.plot_difference_spectrogram(a1, a2, SR),
        "difference_spectrogram.png",
    ),
)


def _features():
    feature_dict = {
        name: visualizations.extract_audio_features(audio, SR)
        for name, audio in AUDIO_DICT.items()
    }
    savefig(
        visualizations.plot_feature_comparison(feature_dict), "features.png"
    )


safe("viz/features", _features)
safe(
    "viz/chromagram",
    lambda: savefig(
        visualizations.plot_chromagram_comparison(AUDIO_DICT, SR),
        "chromagram.png",
    ),
)
safe(
    "viz/mfcc",
    lambda: savefig(
        visualizations.plot_mfcc_evolution(a1, SR), "mfcc_evolution.png"
    ),
)
safe(
    "viz/3d_spectrogram",
    lambda: savefig(
        visualizations.plot_3d_spectrogram(a1, SR), "3d_spectrogram.png"
    ),
)
safe(
    "viz/3d_waterfall",
    lambda: savefig(
        visualizations.plot_3d_spectrogram_waterfall(a1, SR),
        "3d_waterfall.png",
    ),
)
safe(
    "viz/mz_freq_mappings",
    lambda: savefig(
        visualizations.plot_mz_to_frequency_mapping(sonifier),
        "frequency_mappings.png",
    ),
)
safe(
    "viz/scan_progression",
    lambda: savefig(
        visualizations.plot_scan_progression(sonifier), "scan_progression.png"
    ),
)
safe(
    "viz/audio_envelope",
    lambda: savefig(
        visualizations.plot_audio_envelope_comparison(AUDIO_DICT, SR),
        "audio_envelope.png",
    ),
)


def _similarity():
    methods, corr_matrix, spec_matrix = (
        visualizations.compute_audio_similarity_matrix(AUDIO_DICT, SR)
    )
    savefig(
        visualizations.plot_similarity_matrices(
            methods, corr_matrix, spec_matrix
        ),
        "similarity.png",
    )


safe("viz/similarity_matrix", _similarity)

safe(
    "viz/summary_grid",
    lambda: savefig(
        visualizations.create_summary_grid(sonifier, AUDIO_DICT, SR),
        "summary.png",
    ),
)
safe(
    "viz/spectrogram",
    lambda: savefig(
        visualizations.plot_spectrogram(a1, SR), "spectrogram.png"
    ),
)
safe(
    "viz/audio_comparison",
    lambda: savefig(
        visualizations.plot_audio_comparison(AUDIO_DICT, SR),
        "audio_comparison.png",
    ),
)


def _analysis():
    report = visualizations.analyze_audio(a1, SR)
    print(
        f"    tempo: {report.get('tempo', 'n/a')}, "
        f"rms: {report.get('rms_energy', 0):.4f}"
    )


safe("viz/analyze_audio", _analysis)


# ══════════════════════════════════════════════════════════════════════════════
# 13. VIDEO / ANIMATION  (require ffmpeg)
# ══════════════════════════════════════════════════════════════════════════════
section("13. Videos and animations")

# All videos use pentatonic A minor quantized audio.
sonifier.sonify(
    frequency_mapping="inverse_log",
    freq_range=FREQ_RANGE,
    scale="pentatonic_minor",
    root_note="A",
)
VIDEO_AUDIO = sonifier.get_current_audio()
sonifier.current_audio_data = VIDEO_AUDIO.copy()

safe(
    "video/spectrogram",
    lambda: sonifier.create_video(
        vout("video_spectrogram.mp4"),
        fps=15,
        dpi=80,
        show_spectrogram=True,
        show_waveform=True,
    ),
)

safe(
    "video/3d_scan_window",
    lambda: sonifier.create_3d_scan_video(
        vout("video_3d_scans.mp4"), window_size=30, duration_seconds=30, fps=12
    ),
)

safe(
    "video/3d_heatmap_rotate",
    lambda: sonifier.create_3d_heatmap_video(
        vout("video_3d_heatmap.mp4"), duration_seconds=30, fps=12
    ),
)

safe(
    "video/3d_waterfall",
    lambda: sonifier.create_3d_waterfall_video(
        vout("video_waterfall.mp4"), duration_seconds=30, fps=12
    ),
)

safe(
    "video/3d_spectrogram_buildup",
    lambda: sonifier.create_3d_spectrogram_buildup_video(
        vout("video_spectrogram_buildup.mp4"), duration_seconds=30, fps=12
    ),
)


# Comparison video: quantized variants all rooted in pentatonic A minor base.
def _comparison_video():
    vdict = {}
    for _sc, _rt, _label in [
        ("pentatonic_minor", "A", "pentatonic_minor_A"),
        ("major", "C", "major_C"),
        ("blues", "E", "blues_E"),
    ]:
        sonifier.sonify(
            frequency_mapping="inverse_log",
            freq_range=FREQ_RANGE,
            scale=_sc,
            root_note=_rt,
        )
        vdict[_label] = sonifier.get_current_audio()
    sonifier.current_audio_data = VIDEO_AUDIO.copy()
    sonifier.create_comparison_video(
        vdict, vout("video_comparison.mp4"), fps=12
    )


safe("video/comparison", _comparison_video)

safe(
    "video/animate_spectrogram",
    lambda: sonifier.animate_visualization(
        vout("video_animate_spectrogram.mp4"),
        viz_type="spectrogram",
        fps=12,
        duration_seconds=30,
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# 14. MUSICALNOTEQUANTIZER — standalone usage
# ══════════════════════════════════════════════════════════════════════════════
section("14. MusicalNoteQuantizer — standalone")


def _quantizer():
    q = MusicalNoteQuantizer(
        scale="major", root_note="C", freq_range=(200, 4000)
    )
    result = q.quantize_frequency_log(440.0)
    print(
        f"    440 Hz → {result['note']}{result['octave']} "
        f"({result['frequency']:.2f} Hz) [{q.tuning_system}]"
    )
    MusicalNoteQuantizer.list_available_scales()


safe("quantizer/standalone", _quantizer)


# ══════════════════════════════════════════════════════════════════════════════
# 15.  MUSICAL STYLE PRESETS
# ══════════════════════════════════════════════════════════════════════════════
section("15. Musical style presets")

STYLES_DIR = os.path.join(OUTPUT_DIR, "styles")
os.makedirs(STYLES_DIR, exist_ok=True)


def sout(name):
    return os.path.join(STYLES_DIR, name)


# Register custom scales that are style-specific and not in the core quantizer.
MusicalNoteQuantizer.SCALES.setdefault("minor_natural", [0, 2, 3, 5, 7, 8, 10])
MusicalNoteQuantizer.SCALES.setdefault("minor_pentatonic", [0, 3, 5, 7, 10])
MusicalNoteQuantizer.SCALES.setdefault(
    "phrygian_dominant", [0, 1, 4, 5, 7, 8, 10]
)
MusicalNoteQuantizer.SCALES.setdefault("slendro", [0, 2, 5, 7, 10])
MusicalNoteQuantizer.SCALES.setdefault("raga_bhairav", [0, 1, 4, 5, 7, 8, 11])
MusicalNoteQuantizer.SCALES.setdefault("raga_todi", [0, 1, 3, 6, 7, 8, 11])
MusicalNoteQuantizer.EDO_SCALES.setdefault(
    "maqam_rast", [0, 4, 7, 10, 14, 18, 21]
)


def _q_style(scale, root, freq_range, effects, edo=12):
    """Sonify snapped to a scale, then apply an effects chain."""
    sonifier.sonify(
        frequency_mapping="inverse_log",
        freq_range=freq_range,
        scale=scale,
        root_note=root,
        edo_divisions=edo,
    )
    for fx, params in effects:
        sonifier.apply_effect(fx, params)


def _g_style(freq_range, effects, frequency_mapping="chromatic"):
    """Sonify with continuous pitch, then apply an effects chain."""
    sonifier.sonify(
        method="gradient",
        frequency_mapping=frequency_mapping,
        freq_range=freq_range,
    )
    for fx, params in effects:
        sonifier.apply_effect(fx, params)


# ── Botanica ──────────────────────────────────────────────────────────────────
def _botanica():
    _q_style(
        "slendro",
        "C",
        (200, 1500),
        [
            ("lowpass_filter", {"cutoff_freq": 2000, "order": 2}),
            (
                "reverb",
                {"reverb_time_s": 2.0, "room_size": 0.8, "dry_wet_mix": 0.4},
            ),
            ("chorus", {"delay_ms": 30, "depth_ms": 5, "rate_hz": 0.3}),
            ("compressor", {"threshold_db": -18, "ratio": 3, "attack_ms": 50}),
        ],
    )
    sonifier.save_audio(sout("botanica.wav"))


safe("style/botanica", _botanica)


# ── Dark Ambient ──────────────────────────────────────────────────────────────
def _dark_ambient():
    _q_style(
        "phrygian_dominant",
        "C",
        (50, 1000),
        [
            (
                "reverb",
                {"reverb_time_s": 4.0, "room_size": 0.98, "dry_wet_mix": 0.7},
            ),
            ("lowpass_filter", {"cutoff_freq": 1500, "order": 4}),
        ],
    )
    sonifier.save_audio(sout("dark_ambient.wav"))


safe("style/dark_ambient", _dark_ambient)


# ── Delta Blues ───────────────────────────────────────────────────────────────
def _delta_blues():
    _q_style(
        "blues",
        "E",
        (150, 2200),
        [
            (
                "parametric_eq",
                {"frequency_hz": 1200, "gain_db": 2, "q_factor": 1.0},
            ),
            (
                "reverb",
                {"reverb_time_s": 0.6, "room_size": 0.3, "dry_wet_mix": 0.15},
            ),
        ],
    )
    sonifier.save_audio(sout("delta_blues.wav"))


safe("style/delta_blues", _delta_blues)


# ── Drone Doom ────────────────────────────────────────────────────────────────
def _drone_doom():
    _q_style(
        "phrygian",
        "C",
        (50, 600),
        [
            ("overdrive", {"drive": 10, "tone": 0.2, "output_level": 0.7}),
            ("lowpass_filter", {"cutoff_freq": 800, "order": 6}),
            (
                "reverb",
                {"reverb_time_s": 3.0, "room_size": 0.95, "dry_wet_mix": 0.7},
            ),
            ("chorus", {"delay_ms": 50, "depth_ms": 10, "rate_hz": 0.1}),
        ],
    )
    sonifier.save_audio(sout("drone_doom.wav"))


safe("style/drone_doom", _drone_doom)


# ── Hard Rock ─────────────────────────────────────────────────────────────────
def _hard_rock():
    _q_style(
        "minor_pentatonic",
        "E",
        (80, 2500),
        [
            ("fuzz", {"fuzz_amount": 12, "output_level": 0.6}),
            (
                "parametric_eq",
                {"frequency_hz": 1200, "gain_db": 4, "q_factor": 1.5},
            ),
            ("compressor", {"threshold_db": -10, "ratio": 6, "attack_ms": 3}),
            ("delay", {"delay_ms": 400, "feedback": 0.35, "num_taps": 2}),
            (
                "reverb",
                {"reverb_time_s": 1.2, "room_size": 0.6, "dry_wet_mix": 0.25},
            ),
        ],
    )
    sonifier.save_audio(sout("hard_rock.wav"))


safe("style/hard_rock", _hard_rock)


# ── Harsh Noise Wall ──────────────────────────────────────────────────────────
def _harsh_noise_wall():
    _g_style(
        (100, 8000),
        [
            ("fuzz", {"fuzz_amount": 15, "output_level": 0.8}),
            ("bitcrusher", {"bit_depth": 4, "downsample_factor": 8}),
            ("highpass_filter", {"cutoff_freq": 150, "order": 1}),
            ("limiter", {"threshold_db": -3, "release_ms": 10}),
        ],
        frequency_mapping="chromatic",
    )
    sonifier.save_audio(sout("harsh_noise_wall.wav"))


safe("style/harsh_noise_wall", _harsh_noise_wall)


# ── Hindustani Dhrupad ────────────────────────────────────────────────────────
def _hindustani_dhrupad():
    _q_style(
        "raga_todi",
        "C",
        (100, 1200),
        [
            (
                "reverb",
                {"reverb_time_s": 2.0, "room_size": 0.75, "dry_wet_mix": 0.4},
            ),
            ("chorus", {"delay_ms": 25, "depth_ms": 3, "rate_hz": 0.2}),
        ],
    )
    sonifier.save_audio(sout("hindustani_dhrupad.wav"))


safe("style/hindustani_dhrupad", _hindustani_dhrupad)


# ── Kosmische Musik ───────────────────────────────────────────────────────────
def _kosmische_musik():
    _q_style(
        "dorian",
        "D",
        (100, 2500),
        [
            ("phaser", {"rate_hz": 0.15, "depth": 1.0, "stages": 6}),
            ("delay", {"delay_ms": 750, "feedback": 0.5, "num_taps": 3}),
            (
                "reverb",
                {"reverb_time_s": 2.2, "room_size": 0.8, "dry_wet_mix": 0.35},
            ),
            ("chorus", {"delay_ms": 35, "depth_ms": 5, "rate_hz": 0.25}),
        ],
    )
    sonifier.save_audio(sout("kosmische_musik.wav"))


safe("style/kosmische_musik", _kosmische_musik)


# ── Lo-Fi Hip Hop ─────────────────────────────────────────────────────────────
def _lofi_hip_hop():
    _q_style(
        "dorian",
        "C",
        (150, 2500),
        [
            ("bitcrusher", {"bit_depth": 12, "downsample_factor": 2}),
            ("lowpass_filter", {"cutoff_freq": 4000, "order": 3}),
            (
                "reverb",
                {"reverb_time_s": 1.5, "room_size": 0.6, "dry_wet_mix": 0.35},
            ),
            (
                "compressor",
                {"threshold_db": -18, "ratio": 2.5, "attack_ms": 25},
            ),
        ],
    )
    sonifier.save_audio(sout("lofi_hip_hop.wav"))


safe("style/lofi_hip_hop", _lofi_hip_hop)


# ── Maqam (24-EDO) ────────────────────────────────────────────────────────────
def _maqam():
    # Maqam Rast uses 24-EDO (quarter-tone) tuning.
    sonifier.sonify(
        frequency_mapping="inverse_log",
        freq_range=(200, 2000),
        scale="maqam_rast",
        root_note="C",
        edo_divisions=24,
    )
    sonifier.apply_effect(
        "reverb",
        {"reverb_time_s": 1.8, "room_size": 0.65, "dry_wet_mix": 0.35},
    )
    sonifier.apply_effect(
        "delay", {"delay_ms": 375, "feedback": 0.3, "num_taps": 2}
    )
    sonifier.apply_effect(
        "parametric_eq", {"frequency_hz": 1200, "gain_db": 2, "q_factor": 1.2}
    )
    sonifier.save_audio(sout("maqam.wav"))


safe("style/maqam", _maqam)


# ── New Age ───────────────────────────────────────────────────────────────────
def _new_age():
    _q_style(
        "lydian",
        "C",
        (150, 2000),
        [
            ("chorus", {"delay_ms": 40, "depth_ms": 8, "rate_hz": 0.3}),
            (
                "reverb",
                {"reverb_time_s": 3.0, "room_size": 0.85, "dry_wet_mix": 0.5},
            ),
            ("lowpass_filter", {"cutoff_freq": 5000, "order": 2}),
        ],
    )
    sonifier.save_audio(sout("new_age.wav"))


safe("style/new_age", _new_age)


# ── Post-Rock ─────────────────────────────────────────────────────────────────
def _post_rock():
    _q_style(
        "minor_natural",
        "E",
        (100, 3500),
        [
            ("fuzz", {"fuzz_amount": 6, "output_level": 0.5}),
            ("delay", {"delay_ms": 600, "feedback": 0.5, "num_taps": 4}),
            (
                "reverb",
                {"reverb_time_s": 2.5, "room_size": 0.85, "dry_wet_mix": 0.5},
            ),
            ("chorus", {"delay_ms": 30, "depth_ms": 6, "rate_hz": 0.3}),
            ("compressor", {"threshold_db": -14, "ratio": 3, "attack_ms": 20}),
        ],
    )
    sonifier.save_audio(sout("post_rock.wav"))


safe("style/post_rock", _post_rock)


# ── Psychedelic Rock ──────────────────────────────────────────────────────────
def _psychedelic_rock():
    _q_style(
        "mixolydian",
        "A",
        (100, 4000),
        [
            ("phaser", {"rate_hz": 0.4, "depth": 1.2, "stages": 8}),
            ("overdrive", {"drive": 6, "tone": 0.5, "output_level": 0.6}),
            ("delay", {"delay_ms": 650, "feedback": 0.55, "num_taps": 4}),
            (
                "reverb",
                {"reverb_time_s": 2.5, "room_size": 0.85, "dry_wet_mix": 0.45},
            ),
        ],
    )
    sonifier.save_audio(sout("psychedelic_rock.wav"))


safe("style/psychedelic_rock", _psychedelic_rock)


# ── Raga (Indian classical) ───────────────────────────────────────────────────
def _raga():
    _q_style(
        "raga_bhairav",
        "C",
        (150, 1500),
        [
            (
                "reverb",
                {"reverb_time_s": 1.8, "room_size": 0.7, "dry_wet_mix": 0.3},
            ),
            ("delay", {"delay_ms": 428, "feedback": 0.25, "num_taps": 2}),
        ],
    )
    sonifier.save_audio(sout("raga.wav"))


safe("style/raga", _raga)


# ── Shoegaze ──────────────────────────────────────────────────────────────────
def _shoegaze():
    _q_style(
        "major",
        "A",
        (150, 4000),
        [
            ("fuzz", {"fuzz_amount": 8, "output_level": 0.5}),
            ("chorus", {"delay_ms": 45, "depth_ms": 12, "rate_hz": 0.6}),
            (
                "reverb",
                {"reverb_time_s": 3.5, "room_size": 0.9, "dry_wet_mix": 0.65},
            ),
            ("delay", {"delay_ms": 550, "feedback": 0.6, "num_taps": 5}),
            ("lowpass_filter", {"cutoff_freq": 7000, "order": 2}),
            ("compressor", {"threshold_db": -18, "ratio": 3, "attack_ms": 30}),
        ],
    )
    sonifier.save_audio(sout("shoegaze.wav"))


safe("style/shoegaze", _shoegaze)


# ── Space Rock ────────────────────────────────────────────────────────────────
def _space_rock():
    _q_style(
        "lydian",
        "C",
        (80, 4000),
        [
            ("phaser", {"rate_hz": 0.15, "depth": 1.0, "stages": 8}),
            (
                "reverb",
                {"reverb_time_s": 3.5, "room_size": 0.95, "dry_wet_mix": 0.6},
            ),
            ("delay", {"delay_ms": 800, "feedback": 0.6, "num_taps": 5}),
            ("chorus", {"delay_ms": 45, "depth_ms": 10, "rate_hz": 0.25}),
        ],
    )
    sonifier.save_audio(sout("space_rock.wav"))


safe("style/space_rock", _space_rock)


# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  All examples written to  {os.path.abspath(OUTPUT_DIR)}")
print(f"  Styles:  {os.path.abspath(STYLES_DIR)}")
print(f"  Plots:   {os.path.abspath(PLOT_DIR)}")
print(f"  Videos:  {os.path.abspath(VIDEO_DIR)}")
print("=" * 60)
