"""Tests for the unified MSSonifier.sonify() and its rhythm/meter option.

All tests run on small synthetic spectra, so no mzML file is needed.
"""

import numpy as np
import pandas as pd
import pytest

from ms_music import MSSonifier, MusicMeter, QuantizationMode, RhythmConfig
from ms_music.rhythm import build_grid, find_notes, note_gain, resample_spectra

SR = 8000


def _spectra(n_scans=40, n_peaks=12, seed=0):
    rng = np.random.default_rng(seed)
    pool = np.round(rng.uniform(100, 900, 30), 3)
    dfs = []
    for _ in range(n_scans):
        mz = np.sort(rng.choice(pool, n_peaks, replace=False))
        dfs.append(
            pd.DataFrame(
                {"intensities": rng.uniform(1, 100, n_peaks)},
                index=pd.Index(mz, name="mz"),
            )
        )
    return dfs, pool


@pytest.fixture
def sonifier():
    s = MSSonifier(filepath="", total_duration_minutes=4 / 60, sample_rate=SR)
    dfs, pool = _spectra()
    s.processed_spectra_dfs = dfs
    s.max_intensity_overall = max(d["intensities"].max() for d in dfs)
    s.min_mz_overall, s.max_mz_overall = float(pool.min()), float(pool.max())
    s.precursor_mz_list = [float(pool[i % len(pool)]) for i in range(len(dfs))]
    return s


# ------------------------------------------------------------ unified API
def test_sonify_quantized_is_gone():
    assert not hasattr(MSSonifier, "sonify_quantized")
    assert not hasattr(MSSonifier, "setup_musical_quantization")


@pytest.mark.parametrize("method", ["gradient", "adsr"])
@pytest.mark.parametrize("mapping", MSSonifier.FREQUENCY_MAPPINGS)
def test_every_method_and_mapping_works_with_a_scale(
    sonifier, method, mapping
):
    """Scale snapping combines with both methods and all mappings (linear +
    scale used to crash; ADSR + scale used to be impossible)."""
    sonifier.sonify(
        method=method,
        frequency_mapping=mapping,
        freq_range=(200, 2000),
        scale="pentatonic_major",
        root_note="G",
        adsr_settings={"randomize": False},
    )
    audio = sonifier.current_audio_data
    assert audio is not None and audio.size > 0
    assert np.isfinite(audio).all()
    assert sonifier.note_quantizer is not None


def test_scale_restricts_pitches_to_the_scale(sonifier):
    """Every synthesized pitch is a scale note: check via the frequency
    function sonify() builds."""
    sonifier.sonify(scale="major", root_note="C", freq_range=(200, 2000))
    q = sonifier.note_quantizer
    fn = sonifier._frequency_function("inverse_log", (200.0, 2000.0), q)
    for mz in np.linspace(
        sonifier.min_mz_overall, sonifier.max_mz_overall, 25
    ):
        f = fn(mz)
        # Quantizing again is a no-op only for pitches already on the scale.
        assert f == pytest.approx(q.quantize_frequency_log(f)["frequency"])


def test_no_scale_clears_quantizer(sonifier):
    sonifier.sonify(scale="major")
    sonifier.sonify()
    assert sonifier.note_quantizer is None


@pytest.mark.parametrize(
    "bad",
    [
        dict(method="nope"),
        dict(frequency_mapping="nope"),
        dict(ms2_mode="nope"),
    ],
)
def test_invalid_options_raise_early(sonifier, bad):
    with pytest.raises(ValueError):
        sonifier.sonify(**bad)


def test_dda_adds_precursor_tones(sonifier):
    sonifier.sonify()
    plain = sonifier.current_audio_data.copy()
    sonifier.sonify(ms2_mode="dda")
    assert not np.allclose(plain, sonifier.current_audio_data)


# ------------------------------------------------------------------ grid
def test_grid_three_four_sixteenths():
    cfg = RhythmConfig(meter="3/4", tempo=120, subdivision=16)
    assert cfg.bar_seconds == pytest.approx(1.5)
    assert cfg.steps_per_bar == 12
    grid = build_grid(7.0, cfg)  # 4.67 bars -> 5
    assert grid.n_bars == 5
    assert grid.n_steps == 60
    assert grid.total_seconds == pytest.approx(7.5)
    np.testing.assert_allclose(np.diff(grid.onsets), 0.125)


def test_grid_snaps_to_at_least_one_bar():
    grid = build_grid(0.1, RhythmConfig(meter="4/4", tempo=60))
    assert grid.n_bars == 1 and grid.total_seconds == pytest.approx(4.0)


def test_accents_simple_and_compound():
    g = build_grid(
        2.0, RhythmConfig(meter="4/4", tempo=120, subdivision=8, accent=0.4)
    )
    # 4/4 in eighths: downbeat, then every quarter (2 steps) gets half.
    np.testing.assert_allclose(g.accents[:8], [1.4, 1, 1.2, 1, 1.2, 1, 1.2, 1])
    g = build_grid(
        1.5, RhythmConfig(meter="6/8", tempo=120, subdivision=8, accent=0.4)
    )
    # 6/8: two dotted-quarter pulses per bar -> accents on steps 0 and 3.
    np.testing.assert_allclose(g.accents[:6], [1.4, 1, 1, 1.2, 1, 1])


def test_swing_delays_offbeat_eighths():
    straight = build_grid(2.0, RhythmConfig(tempo=120, subdivision=8))
    swung = build_grid(
        2.0,
        RhythmConfig(tempo=120, subdivision=8, mode="swing", swing_ratio=0.67),
    )
    q = 0.5
    np.testing.assert_allclose(swung.onsets[::2], straight.onsets[::2])
    np.testing.assert_allclose(
        swung.onsets[1::2] - swung.onsets[::2], 0.67 * q
    )


def test_humanized_is_seeded_and_ordered():
    cfg = dict(tempo=120, mode=QuantizationMode.HUMANIZED, seed=3)
    a = build_grid(4.0, RhythmConfig(**cfg)).onsets
    b = build_grid(4.0, RhythmConfig(**cfg)).onsets
    np.testing.assert_array_equal(a, b)
    assert a[0] == 0 and np.all(np.diff(a) >= 0)


@pytest.mark.parametrize(
    "bad",
    [
        dict(meter="6/8", subdivision=4),
        dict(tempo=0),
        dict(gate=0),
        dict(subdivision=12),
        dict(aggregate="median"),
        dict(meter="bogus"),
        dict(note_threshold=1.0),
    ],
)
def test_config_validation(bad):
    with pytest.raises(ValueError):
        RhythmConfig(**bad)


def test_config_coerce():
    assert RhythmConfig.coerce(None) is None
    cfg = RhythmConfig.coerce({"meter": "7/8", "subdivision": 8})
    assert cfg.meter is MusicMeter.SEVEN_EIGHT
    assert RhythmConfig.coerce(cfg) is cfg


def test_segment_lengths_cover_total():
    grid = build_grid(3.0, RhythmConfig(tempo=97, mode="swing"))
    lengths = grid.segment_lengths(SR)
    assert lengths.sum() == round(grid.total_seconds * SR)
    assert (lengths > 0).all()


# ------------------------------------------------------------- resampling
def test_resample_aggregates_when_more_scans_than_steps():
    dfs = [
        pd.DataFrame({"intensities": [v]}, index=[100.0]) for v in (1, 5, 3, 2)
    ]
    out, owner = resample_spectra(dfs, 2, "max")
    assert [df.loc[100.0, "intensities"] for df in out] == [5, 3]
    out, _ = resample_spectra(dfs, 2, "sum")
    assert [df.loc[100.0, "intensities"] for df in out] == [6, 5]
    assert owner == [1, 3]


def test_resample_holds_when_fewer_scans_than_steps():
    dfs = [pd.DataFrame({"intensities": [v]}, index=[100.0]) for v in (1, 2)]
    out, owner = resample_spectra(dfs, 6, "max")
    assert len(out) == 6
    assert [df.loc[100.0, "intensities"] for df in out] == [1, 1, 1, 2, 2, 2]
    assert owner == [0, 0, 0, 1, 1, 1]


# ------------------------------------------------------------------ notes
def test_find_notes_follows_signal_runs():
    levels = np.array([0, 5, 6, 7, 0, 0, 3, 0.1, 4, 4])
    # threshold 0.05 * 7 = 0.35: the 0.1 dip ends a note.
    assert find_notes(levels, 0.05) == [(1, 4), (6, 7), (8, 10)]
    assert find_notes(np.zeros(5), 0.05) == []
    assert find_notes(np.ones(6), 0.05) == [(0, 6)]


def test_sustained_signal_is_one_unbroken_note():
    """Regression: metering used to re-articulate every grid step, so a
    steady signal came out as a chain of short notes with gaps."""
    cfg = RhythmConfig(
        meter="4/4", tempo=120, subdivision=16, gate=0.5, accent=0.0
    )
    grid = build_grid(2.0, cfg)  # 16 sixteenth steps
    levels = np.ones(grid.n_steps)
    t = np.arange(int(grid.total_seconds * SR)) / SR
    gain = note_gain(grid, levels, 0, grid.n_steps, t)
    inner = gain[int(0.02 * SR) : int(1.8 * SR)]
    assert inner.min() == pytest.approx(1.0)  # no dips at step boundaries


def test_note_accent_depends_on_onset_position():
    cfg = RhythmConfig(meter="4/4", tempo=120, subdivision=8, accent=1.0)
    grid = build_grid(2.0, cfg)
    levels = np.ones(grid.n_steps)
    t = np.arange(int(grid.total_seconds * SR)) / SR
    on_downbeat = note_gain(grid, levels, 0, 2, t).max()
    on_offbeat = note_gain(grid, levels, 1, 2, t).max()
    on_beat_two = note_gain(grid, levels, 2, 4, t).max()
    assert on_downbeat == pytest.approx(2.0)
    assert on_offbeat == pytest.approx(1.0)
    assert on_beat_two == pytest.approx(1.5)


def test_gate_only_shortens_the_last_step():
    cfg = RhythmConfig(tempo=120, subdivision=8, gate=0.5, accent=0.0)
    grid = build_grid(2.0, cfg)  # eighth steps of 0.25 s
    t = np.arange(int(grid.total_seconds * SR)) / SR
    gain = note_gain(grid, np.ones(grid.n_steps), 0, 3, t)
    assert gain[int(0.60 * SR)] == pytest.approx(1.0)  # inside step 2
    assert gain[int(0.70 * SR)] == 0  # gated tail of step 2


# ---------------------------------------------------------- sonify+rhythm
@pytest.mark.parametrize("method", ["gradient", "adsr"])
def test_sonify_with_rhythm_snaps_length(sonifier, method):
    sonifier.sonify(
        method=method,
        scale="major",
        rhythm={"meter": "3/4", "tempo": 90, "mode": "swing"},
        adsr_settings={"randomize": False},
    )
    grid = sonifier.rhythm_grid
    assert grid is not None and grid.n_bars >= 1
    assert sonifier.current_audio_data.size == round(grid.total_seconds * SR)
    assert np.isfinite(sonifier.current_audio_data).all()


def test_rhythm_works_with_ms2(sonifier):
    sonifier.sonify(ms2_mode="dda", rhythm={"meter": "4/4", "tempo": 100})
    assert np.isfinite(sonifier.current_audio_data).all()


def test_no_rhythm_is_deterministic_and_clears_grid(sonifier):
    sonifier.sonify(rhythm={"tempo": 100})
    sonifier.sonify()
    first = sonifier.current_audio_data.copy()
    assert sonifier.rhythm_grid is None
    sonifier.sonify()
    np.testing.assert_array_equal(first, sonifier.current_audio_data)
    # Unmetered length: one equal slice per scan.
    n = len(sonifier.processed_spectra_dfs)
    assert first.size == n * round(SR * sonifier.total_duration_seconds / n)


@pytest.mark.parametrize("method", ["gradient", "adsr"])
def test_metered_steady_signal_is_not_choppy(method):
    """A pitch present in every scan plays as one sustained note."""
    s = MSSonifier(filepath="", total_duration_minutes=2 / 60, sample_rate=SR)
    s.processed_spectra_dfs = [
        pd.DataFrame({"intensities": [10.0]}, index=pd.Index([400.0]))
        for _ in range(40)
    ]
    s.max_intensity_overall, s.min_mz_overall, s.max_mz_overall = (
        10.0,
        100.0,
        900.0,
    )
    s.sonify(
        method=method,
        adsr_settings={
            "randomize": False,
            "attack_time_pc": 0.01,
            "release_time_pc": 0.01,
        },
        rhythm={"tempo": 120, "subdivision": 16, "accent": 0},
    )
    audio = s.current_audio_data
    frame = int(0.01 * SR)
    n = audio.size // frame
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    body = rms[2:-8]  # skip the attack and the final release/gate
    assert body.min() > 0.5 * body.max()  # no silent gaps between steps
