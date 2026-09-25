"""Input-file-free smoke tests for ms_music.

These exercise the public surface without needing an mzML/FID file, and
guard against regressions that previously broke a plain ``import ms_music``
or a quantization call.
"""

import importlib

import numpy as np
import pytest


def _synthetic_audio(sample_rate=22050, seconds=0.25, freq=440.0):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_public_api_names_all_resolve():
    """Every name advertised in ``__all__`` must actually exist.

    Regression guard: ``create_microtonal_config`` used to be exported but
    never defined, so ``from ms_music import *`` raised AttributeError.
    """
    import ms_music

    assert ms_music.__version__
    for name in ms_music.__all__:
        assert hasattr(ms_music, name), f"{name} is exported but missing"


@pytest.mark.parametrize(
    "effect_name, params",
    [
        ("normalize", {}),
        ("fade_in", {}),
        ("fade_out", {}),
        ("lowpass_filter", {"cutoff_freq": 2000}),
        ("highpass_filter", {"cutoff_freq": 200}),
        ("reverb", {"reverb_time_s": 0.3, "dry_wet_mix": 0.3}),
        ("delay", {"delay_ms": 50, "feedback": 0.2}),
        ("tremolo", {"rate_hz": 5, "depth": 0.5}),
        ("bitcrusher", {"bit_depth": 8, "downsample_factor": 4}),
        ("overdrive", {"drive": 4}),
        ("flanger", {"depth_ms": 3.0, "feedback": 0.4}),
    ],
)
def test_effects_run_and_stay_finite(effect_name, params):
    from ms_music import effects

    audio = _synthetic_audio()
    sr = 22050
    func = getattr(effects, f"apply_{effect_name}")
    out = func(audio, sr, **params)
    out = np.asarray(out)
    assert out.ndim == 1
    assert out.size > 0
    assert np.all(np.isfinite(out))


def test_effects_are_enumerable():
    """The GUI and MSSonifier.apply_effect both discover effects by dir()."""
    from ms_music import effects

    names = [n for n in dir(effects) if n.startswith("apply_")]
    assert "apply_reverb" in names
    assert len(names) > 20


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale="major", root_note="C"),                       # 12-TET
        dict(scale="19_edo_diatonic", root_note="C", edo_divisions=19),  # EDO
        dict(scale="just_major", root_note="C", use_just_intonation=True),  # JI
    ],
)
def test_quantizer_both_distance_metrics(kwargs):
    """Both quantize_frequency and quantize_frequency_log must work.

    Regression guard: quantize_frequency (linear distance) was called by
    sonify(scale=..., use_log_distance=False) but never implemented.
    """
    from ms_music.musical_quantization import MusicalNoteQuantizer

    q = MusicalNoteQuantizer(freq_range=(200, 2000), **kwargs)
    for method in (q.quantize_frequency, q.quantize_frequency_log):
        info = method(440.0)
        assert info["frequency"] > 0
        assert "note" in info


def test_midi_config_coerces_strings_to_enums():
    """MidiConfig accepts member names and '3/4' notation, and rejects junk."""
    from ms_music import MidiConfig, MusicMeter, QuantizationMode

    # enum member-name form
    cfg = MidiConfig(meter="THREE_FOUR", quantization_mode="swing")
    assert cfg.meter is MusicMeter.THREE_FOUR
    assert cfg.quantization_mode is QuantizationMode.SWING

    # musical "N/M" notation
    assert MidiConfig(meter="3/4").meter is MusicMeter.THREE_FOUR
    assert MidiConfig(meter="7/8").meter is MusicMeter.SEVEN_EIGHT

    # unknown values raise instead of silently defaulting
    with pytest.raises(ValueError):
        MidiConfig(meter="bogus")
    with pytest.raises(ValueError):
        MidiConfig(meter="5/16")  # valid shape, not a defined meter
    with pytest.raises(ValueError):
        MidiConfig(quantization_mode="bogus")


def _flanger_scalar_reference(audio_data, sample_rate, delay_ms=5.0,
                              depth_ms=3.0, rate_hz=0.3, feedback=0.4,
                              dry_wet_mix=0.5):
    """Pre-vectorization scalar flanger, used to pin apply_flanger's output."""
    from ms_music import effects as fx

    audio = fx._validate_audio_input(audio_data, "Flanger")
    base = int(delay_ms * sample_rate / 1000.0)
    depth = int(depth_ms * sample_rate / 1000.0)
    feedback = np.clip(feedback, -0.95, 0.95)
    t = np.arange(len(audio)) / sample_rate
    lfo = np.sin(fx.TAU * rate_hz * t)
    buffer = np.copy(audio)
    wet = np.zeros_like(audio)
    for i in range(len(audio)):
        current_delay = base + depth * lfo[i]
        delay_idx = int(current_delay)
        frac = current_delay - delay_idx
        past, past2 = i - delay_idx, i - delay_idx - 1
        if past >= 0 and past2 >= 0:
            d = (1 - frac) * buffer[past] + frac * buffer[past2]
        elif past >= 0:
            d = buffer[past]
        else:
            d = 0.0
        wet[i] = d
        buffer[i] = audio[i] + d * feedback
    return fx._apply_dry_wet_mix(audio, wet, dry_wet_mix)


@pytest.mark.parametrize(
    "delay_ms, depth_ms, feedback",
    [
        (5.0, 3.0, 0.4),   # d_min > 1 -> block path
        (5.0, 3.0, -0.6),  # negative feedback
        (5.0, 0.0, 0.5),   # zero depth (constant delay)
        (1.0, 5.0, 0.4),   # depth >= base -> scalar fallback path
    ],
)
def test_flanger_matches_scalar_reference(delay_ms, depth_ms, feedback):
    """Block-vectorized apply_flanger is bit-for-bit the old scalar version."""
    from ms_music import effects

    rng = np.random.default_rng(0)
    sr = 8000
    audio = rng.standard_normal(4000).astype(np.float32)
    kw = dict(delay_ms=delay_ms, depth_ms=depth_ms, feedback=feedback)
    got = effects.apply_flanger(audio, sr, **kw)
    ref = _flanger_scalar_reference(audio, sr, **kw)
    assert np.allclose(got, ref, atol=1e-6, rtol=1e-5)


def test_overdrive_asymmetric_clipping_is_vectorized_correctly():
    """apply_overdrive's np.where clipping matches the old per-sample loop.

    With tone == 0.5 both EQ branches are skipped, so the transform is just
    validate -> scale -> asymmetric clip -> output level, which we replicate
    with the original scalar loop.
    """
    from ms_music import effects

    rng = np.random.default_rng(1)
    audio = rng.standard_normal(3000).astype(np.float32)
    drive, output_level = 4.0, 0.5

    got = effects.apply_overdrive(audio, 22050, drive=drive, tone=0.5,
                                  output_level=output_level)

    driven = effects._validate_audio_input(audio, "Overdrive") * drive
    ref = np.zeros_like(driven)
    for i in range(len(driven)):
        x = driven[i]
        ref[i] = np.tanh(x * 0.7) if x > 0 else np.tanh(x * 0.9)
    ref = ref * output_level

    assert np.allclose(got, ref, atol=1e-6)


def test_map_mz_to_frequency_rejects_unknown_mapping():
    """Regression guard: an unknown mapping used to silently return None."""
    from ms_music import MSSonifierMidi

    midi = MSSonifierMidi(filepath="")
    midi.setup_musical_system()  # creates the note_quantizer
    with pytest.raises(ValueError):
        midi._map_mz_to_frequency(500.0, "not_a_real_mapping", 100.0, 1000.0)


def test_gui_module_importable():
    """The GUI package imports without a display (construction is separate)."""
    mod = importlib.import_module("ms_music.gui.app")
    assert hasattr(mod, "main")
