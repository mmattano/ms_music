"""Regression tests for bugs fixed before the 0.3.0 release."""

import time

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from ms_music import MSSonifier, effects
from ms_music.gui.controller import SonifierController

SR = 8000


def _tone(seconds=0.5, amplitude=0.25, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _sonifier_with(dfs):
    s = MSSonifier(filepath="", total_duration_minutes=2 / 60, sample_rate=SR)
    s.processed_spectra_dfs = dfs
    s.max_intensity_overall = max(d["intensities"].max() for d in dfs)
    mz = np.concatenate([d.index.to_numpy() for d in dfs])
    s.min_mz_overall, s.max_mz_overall = float(mz.min()), float(mz.max())
    return s


# ------------------------------------------------------------ saving audio
def test_save_audio_without_normalize_keeps_amplitude(tmp_path):
    s = MSSonifier(filepath="", sample_rate=SR)
    s.current_audio_data = _tone(amplitude=0.25)
    path = tmp_path / "quiet.wav"
    s.save_audio(str(path), normalize=False)
    _, data = wavfile.read(path)
    assert np.abs(data).max() / 32767 == pytest.approx(0.25, abs=1e-3)

    s.save_audio(str(path), normalize=True)
    _, data = wavfile.read(path)
    assert np.abs(data).max() == 32767


def test_save_and_apply_without_audio_raise(tmp_path):
    s = MSSonifier(filepath="", sample_rate=SR)
    with pytest.raises(RuntimeError):
        s.save_audio(str(tmp_path / "none.wav"))
    with pytest.raises(RuntimeError):
        s.apply_effect("reverb")


def test_unknown_effect_raises_with_the_valid_names():
    s = MSSonifier(filepath="", sample_rate=SR)
    s.current_audio_data = _tone()
    with pytest.raises(ValueError, match="reverb"):
        s.apply_effect("no_such_effect")


# --------------------------------------------------------- one m/z bin
@pytest.mark.parametrize("method", ["gradient", "adsr"])
@pytest.mark.parametrize("mapping", MSSonifier.FREQUENCY_MAPPINGS)
def test_single_mz_bin_gives_finite_audio(method, mapping):
    dfs = [
        pd.DataFrame({"intensities": [5.0]}, index=pd.Index([150.0], name="mz"))
        for _ in range(5)
    ]
    s = _sonifier_with(dfs)
    s.sonify(method=method, frequency_mapping=mapping)
    audio = s.current_audio_data
    assert audio is not None and audio.size
    assert np.all(np.isfinite(audio))


# ------------------------------------------------------------- effects
def _reference_limiter(x, sr, threshold_db=-1.0, release_ms=50.0,
                       lookahead_ms=5.0):
    """The original sample-by-sample limiter, in float64."""
    thr = 10 ** (threshold_db / 20.0)
    rc = np.exp(-1.0 / (release_ms * sr / 1000.0))
    la = int(lookahead_ms * sr / 1000.0)
    p = np.pad(x.astype(np.float64), (la, 0))
    out = np.zeros_like(p)
    g = 1.0
    for i in range(len(p)):
        peak = np.abs(p[i:i + max(la, 1)]).max()
        if peak * g > thr:
            g = min(g, thr / peak)
        else:
            g = min(1.0, g + (1 - rc))
        out[i] = p[i] * g
    return out[la:]


@pytest.mark.parametrize(
    "params",
    [{}, {"threshold_db": -6, "lookahead_ms": 2},
     {"threshold_db": -12, "release_ms": 10, "lookahead_ms": 10}],
)
def test_limiter_matches_reference(params):
    x = (np.random.default_rng(0).standard_normal(4000) * 0.6).astype(
        np.float32
    )
    out = effects.apply_limiter(x, 22050, **params)
    ref = _reference_limiter(x, 22050, **params)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_limiter_is_fast():
    x = (np.random.default_rng(0).standard_normal(44100 * 30) * 0.6).astype(
        np.float32
    )
    start = time.perf_counter()
    effects.apply_limiter(x, 44100)
    assert time.perf_counter() - start < 5.0


@pytest.mark.parametrize(
    "target_type", ["noise_reduction", "echo_cancellation", "equalization"]
)
def test_adaptive_filter_stays_finite_on_loud_input(target_type):
    loud = _tone(seconds=0.5, amplitude=5.0)
    out = effects.apply_adaptive_filter(loud, SR, target_type=target_type)
    assert np.all(np.isfinite(out))


# ------------------------------------------------------------------ GUI
@pytest.fixture
def controller_with_audio():
    # 22.05 kHz: some effects use fixed filters up to 8 kHz.
    c = SonifierController()
    c.sonifier = MSSonifier(filepath="", sample_rate=22050)
    c.sample_rate = 22050
    c.sonifier.current_audio_data = _tone(sr=22050)
    c.base_audio = c.sonifier.current_audio_data.copy()
    return c


def test_every_effect_applies_with_its_form_defaults(controller_with_audio):
    c = controller_with_audio
    for name in c.list_effects():
        spec = c.effect_param_spec(name)
        c.apply_effect(name, dict(spec))
        c.reset_effects()


def test_missing_required_effect_setting_is_reported(controller_with_audio):
    c = controller_with_audio
    with pytest.raises(ValueError, match="cutoff_freq"):
        c.apply_effect("lowpass_filter", {"cutoff_freq": None})
    assert c.effect_chain == []


def test_failed_load_keeps_the_current_data(controller_with_audio, tmp_path):
    c = controller_with_audio
    before = c.sonifier
    with pytest.raises(Exception):
        c.load_mzml(str(tmp_path / "missing.mzML"), 1, 0.1, SR)
    assert c.sonifier is before
    assert c.current_audio() is not None


def test_pasted_paths_are_cleaned():
    from ms_music.gui.pages import clean_path

    assert clean_path('  "C:\\data\\run 1.mzML" ') == "C:\\data\\run 1.mzML"
    assert clean_path("'/tmp/a.mzML'") == "/tmp/a.mzML"
    assert clean_path("") == ""
