"""Tests for mobility-driven sound: stereo pan, mapped effects, stereo I/O."""

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from ms_music import MSSonifier
from ms_music.mobility import MappedEffect, MobilityConfig, ToneBus

SR = 16000


def _sonifier(mobility=(0.6, 1.6), mz=(150.0, 900.0), n_scans=8):
    s = MSSonifier(filepath="", total_duration_minutes=1 / 60, sample_rate=SR)
    s.processed_spectra_dfs = [
        pd.DataFrame(
            {"intensities": [1.0] * len(mz), "mobility": list(mobility)},
            index=pd.Index(list(mz)),
        )
        for _ in range(n_scans)
    ]
    s.max_intensity_overall, s.min_mz_overall, s.max_mz_overall = (
        1.0,
        100.0,
        1000.0,
    )
    s.ion_mobility_data = {"unit": "1/K0", "range": (0.6, 1.6)}
    return s


def _band_energy(signal, f, sr=SR, width=15.0):
    spec = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(signal.size, 1 / sr)
    return spec[np.abs(freqs - f) < width].sum()


# ------------------------------------------------------------------ config
def test_config_validation_and_coercion():
    cfg = MobilityConfig.coerce(
        {
            "pan": 0.5,
            "effects": [
                ("reverb", "dry_wet_mix", 0.1, 0.6),
                {
                    "effect": "apply_lowpass_filter",
                    "param": "cutoff_freq",
                    "low": 8000,
                    "high": 1500,
                },
            ],
        }
    )
    assert cfg.stereo and cfg.active
    assert [e.effect for e in cfg.effects] == ["reverb", "lowpass_filter"]
    assert MobilityConfig.coerce(None) is None
    assert not MobilityConfig().active
    for bad in (
        dict(pan=2),
        dict(range=(1.0, 0.5)),
        dict(effects=[("reverb", "dry_wet_mix", 0, 1)], bands=1),
    ):
        with pytest.raises(ValueError):
            MobilityConfig(**bad)
    with pytest.raises(ValueError, match="no parameter"):
        MappedEffect("reverb", "not_a_param", 0, 1)
    with pytest.raises(ValueError, match="Unknown effect"):
        MappedEffect("nope", "x", 0, 1)


def test_equal_power_pan():
    cfg = MobilityConfig(pan=1.0)
    left, right = cfg.pan_gains(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(left**2 + right**2, 1.0, rtol=1e-6)
    assert left[0] == pytest.approx(1.0) and right[0] == pytest.approx(
        0.0, abs=1e-6
    )
    assert right[2] == pytest.approx(1.0)
    inv = MobilityConfig(pan=1.0, invert=True)
    assert inv.pan_gains(0.0)[1] == pytest.approx(1.0)


def test_band_crossfade_is_continuous():
    """Sweeping the position moves energy smoothly between bands; the total
    per sample is preserved (no steps or gaps)."""
    cfg = MobilityConfig(effects=[("reverb", "dry_wet_mix", 0, 1)], bands=4)
    bus = ToneBus(1000, cfg)
    bus.add(np.ones(1000), 0, np.linspace(0, 1, 1000))
    np.testing.assert_allclose(bus.buf.sum(axis=(0, 1)), 1.0, rtol=1e-5)
    per_band = bus.buf[:, 0, :]
    assert np.all(np.abs(np.diff(per_band, axis=1)) < 0.01)


# ---------------------------------------------------------------- pan
@pytest.mark.parametrize("method", ["gradient", "adsr"])
@pytest.mark.parametrize("metered", [False, True])
def test_pan_places_low_mobility_left_high_right(method, metered):
    s = _sonifier()
    kwargs = dict(
        method=method,
        frequency_mapping="linear",
        freq_range=(200.0, 1000.0),
        mobility={"pan": 1.0},
        adsr_settings={"randomize": False},
    )
    if metered:
        kwargs["rhythm"] = {"tempo": 120, "accent": 0}
    s.sonify(**kwargs)
    audio = s.current_audio_data
    assert audio.ndim == 2 and audio.shape[0] == 2
    f_low = s._mz_to_frequency(150.0, "linear", (200.0, 1000.0))  # 1/K0 0.6
    f_high = s._mz_to_frequency(900.0, "linear", (200.0, 1000.0))  # 1/K0 1.6
    left, right = audio
    assert _band_energy(left, f_low) > 20 * _band_energy(right, f_low)
    assert _band_energy(right, f_high) > 20 * _band_energy(left, f_high)


def test_mono_mobility_options_stay_mono_and_default_unchanged():
    s = _sonifier()
    s.sonify(frequency_mapping="linear")
    plain = s.current_audio_data.copy()
    s.sonify(frequency_mapping="linear", mobility={"brightness": 0.0})
    np.testing.assert_array_equal(plain, s.current_audio_data)
    s.sonify(frequency_mapping="linear", mobility={"brightness": 0.5})
    assert s.current_audio_data.ndim == 1


# ------------------------------------------------------- mapped effects
def test_mapped_reverb_gives_high_mobility_more_tail():
    """With reverb wet 0 -> 1 across mobility, only the extended ion
    keeps ringing after the sound stops."""
    s = _sonifier(n_scans=4)
    reverb = ("reverb", "dry_wet_mix", 0.0, 1.0, {"reverb_time_s": 1.5})
    s.sonify(
        frequency_mapping="linear",
        freq_range=(200.0, 1000.0),
        mobility={"effects": [reverb], "bands": 3},
    )
    audio = s.current_audio_data
    assert audio.ndim == 1
    f_low = s._mz_to_frequency(150.0, "linear", (200.0, 1000.0))
    f_high = s._mz_to_frequency(900.0, "linear", (200.0, 1000.0))
    # The reverb spreads each tone; compare how much sits away from the
    # pure tone (wet energy) for each pitch.
    spec_low = _band_energy(audio, f_low, width=3) / _band_energy(
        audio, f_low, width=60
    )
    spec_high = _band_energy(audio, f_high, width=3) / _band_energy(
        audio, f_high, width=60
    )
    assert spec_high < spec_low


def test_mapped_effect_value_interpolates_and_rounds_ints():
    e = MappedEffect("echo", "num_echoes", 1, 5)
    assert e.value_at(0.5) == 3
    e = MappedEffect("lowpass_filter", "cutoff_freq", 8000, 2000)
    assert e.value_at(0.25) == pytest.approx(6500)


# --------------------------------------------------------- stereo I/O
def test_stereo_effects_undo_reset_and_wav(tmp_path):
    from ms_music.gui.controller import SonifierController

    c = SonifierController()
    c.sonifier, c.sample_rate, c.source_kind = _sonifier(), SR, "mzml"
    c.sonify(frequency_mapping="linear", mobility={"pan": 1.0})
    base = c.current_audio().copy()
    assert base.shape[0] == 2 and c.is_stereo()
    c.apply_effect("reverb", {"dry_wet_mix": 0.4})
    assert c.current_audio().shape[0] == 2
    assert not np.allclose(c.current_audio()[0], c.current_audio()[1])
    c.undo_effect()
    np.testing.assert_array_equal(c.current_audio(), base)
    c.apply_effect("fade_in", {})
    c.reset_effects()
    np.testing.assert_array_equal(c.current_audio(), base)

    path = tmp_path / "stereo.wav"
    c.sonifier.save_audio(str(path))
    sr, data = wavfile.read(path)
    assert sr == SR and data.ndim == 2 and data.shape[1] == 2
    wav = c.current_wav_bytes()
    assert wav[22:24] == (2).to_bytes(2, "little")  # channel count
    times, lows, highs = c.waveform_preview(100)
    assert len(times) == 100
    assert c.duration_seconds() == pytest.approx(base.shape[1] / SR)


# ------------------------------------------------------ parameter guide
def test_every_numeric_effect_setting_has_a_guide():
    import inspect

    from ms_music import effects
    from ms_music.effect_guide import parameter_guide
    from ms_music.gui.controller import SonifierController
    from ms_music.mobility import numeric_parameters

    for effect in SonifierController.list_effects():
        signature = inspect.signature(getattr(effects, f"apply_{effect}"))
        for param in numeric_parameters(effect):
            guide = parameter_guide(effect, param)
            assert guide is not None, f"no guide for {effect}.{param}"
            if not guide.mappable:
                assert guide.why_not
                continue
            for value in guide.suggest:
                assert guide.minimum <= value <= guide.maximum
            default = signature.parameters[param].default
            if isinstance(default, (int, float)) and not isinstance(
                default, bool
            ):
                assert guide.minimum <= default <= guide.maximum, (
                    effect,
                    param,
                )
            assert "Allowed:" in guide.describe(default)


def test_guide_ranges_are_enforced_and_structure_settings_blocked():
    from ms_music.mobility import mappable_parameters

    with pytest.raises(ValueError, match="outside the allowed range"):
        MappedEffect("delay", "feedback", 0.1, 3.0)
    with pytest.raises(ValueError, match="can't be mapped"):
        MappedEffect("lowpass_filter", "order", 2, 8)
    assert "order" not in mappable_parameters("lowpass_filter")
    assert "cutoff_freq" in mappable_parameters("lowpass_filter")
