"""Tests for streaming loading, ion mobility and MS2 handling."""

import os

import numpy as np
import pandas as pd
import pytest

from ms_music import MSSonifier, io
from ms_music.rhythm import resample_spectra

from mzml_factory import spectrum, write_mzml


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MS_MUSIC_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def tims_like(tmp_path):
    """MS1 frames with a mobility array, interleaved PASEF-like MS2 scans
    whose file order is NOT their time order."""
    rng = np.random.default_rng(0)
    spectra = []
    for f in range(6):
        mz = np.sort(rng.uniform(100, 1000, 400))
        spectra.append(
            spectrum(
                mz,
                rng.uniform(1, 100, 400),
                rt_seconds=f * 10,
                mobility=0.6 + (mz - 100) / 900,
            )
        )
        for j in (2, 0, 1):  # out of time order within the frame
            spectra.append(
                spectrum(
                    np.array([150.2, 300.7, 450.1]),
                    np.array([5.0, 50.0, 20.0]),
                    rt_seconds=f * 10 + 1 + j,
                    ms_level=2,
                    precursor=500 + 10 * j + f,
                    charge=2,
                    window=(500 + 10 * j + f, 1.5),
                    scan_mobility=0.9 + 0.1 * j,
                )
            )
    return write_mzml(tmp_path / "tims.mzML", spectra)


def test_binning_matches_preprocess_spectra():
    rng = np.random.default_rng(1)
    raw = [
        {"mz": rng.uniform(100, 300, 200), "intensity": rng.uniform(0, 9, 200)}
        for _ in range(4)
    ]
    dfs, max_i, lo, hi = io.preprocess_spectra(raw)
    for spec, df in zip(raw, dfs):
        bins, sums, _ = io.bin_spectrum(spec["mz"], spec["intensity"])
        np.testing.assert_array_equal(df.index, bins)
        expected = (
            pd.Series(spec["intensity"])
            .groupby(np.round(spec["mz"]).astype(int))
            .sum()
        )
        np.testing.assert_allclose(
            df["intensities"].to_numpy(), expected.to_numpy()
        )
    assert max_i == max(r["intensity"].max() for r in raw)


def test_bin_spectrum_weights_mobility_by_intensity():
    bins, sums, mob = io.bin_spectrum(
        [100.1, 99.9, 200.0], [1.0, 3.0, 2.0], [0.8, 1.2, 1.0]
    )
    assert list(bins) == [100, 200]
    np.testing.assert_allclose(sums, [4.0, 2.0])
    np.testing.assert_allclose(mob, [(0.8 + 3.6) / 4, 1.0])


def test_load_ms1_with_mobility(tims_like):
    run = io.load_spectra(tims_like, ms_level=1)
    assert len(run.spectra) == 6
    assert run.has_mobility and run.mobility_unit.startswith("1/K0")
    assert "mobility" in run.spectra[0].columns
    lo, hi = run.mobility_range
    assert 0.6 <= lo < hi <= 1.6
    assert run.precursor_mz == [None] * 6


def test_ms2_sorted_by_time_with_metadata(tims_like):
    run = io.load_spectra(tims_like, ms_level=2)
    assert len(run.spectra) == 18
    assert run.retention_times == sorted(run.retention_times)
    assert all(c == 2 for c in run.charges)
    first = run.isolation_windows[0]
    assert first[1] - first[0] == pytest.approx(3.0)
    # Per-spectrum mobility (no array) becomes the bins' mobility.
    assert run.has_mobility
    assert run.spectrum_mobility[0] == pytest.approx(0.9)


def test_rt_and_mobility_filters(tims_like):
    run = io.load_spectra(tims_like, ms_level=1, rt_range=(10 / 60, 30 / 60))
    assert len(run.spectra) == 3
    run = io.load_spectra(tims_like, ms_level=1, mobility_range=(0.6, 0.8))
    assert run.spectra[0].index.max() <= 100 + 0.2 * 900 + 1


def test_cache_round_trip_and_invalidation(tims_like):
    first = io.load_spectra(tims_like, ms_level=2)
    again = io.load_spectra(tims_like, ms_level=2)
    assert again.from_cache and not first.from_cache
    for a, b in zip(first.spectra, again.spectra):
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    assert again.isolation_windows == first.isolation_windows
    assert again.charges == first.charges
    os.utime(tims_like, (1, 1))  # modification time changes -> new key
    assert not io.load_spectra(tims_like, ms_level=2).from_cache
    assert not io.load_spectra(tims_like, ms_level=2, cache=False).from_cache


def test_sonifier_loads_mobility(tims_like):
    s = MSSonifier(
        filepath=tims_like, total_duration_minutes=0.1, sample_rate=8000
    )
    s.load_and_preprocess_data()
    assert s.has_ion_mobility
    assert s.ion_mobility_data["unit"].startswith("1/K0")


def test_resample_members_and_mobility():
    dfs = [
        pd.DataFrame(
            {"intensities": [1.0, 3.0], "mobility": [1.0, 2.0]},
            index=[100, 101],
        )
        for _ in range(4)
    ]
    dfs[1] = pd.DataFrame(
        {"intensities": [3.0], "mobility": [2.0]}, index=[100]
    )
    out, rep, members = resample_spectra(dfs, 2, "max", return_members=True)
    assert [list(m) for m in members] == [[0, 1], [2, 3]]
    # m/z 100 in step 0: max intensity 3, mobility weighted (1*1 + 3*2)/4.
    assert out[0].loc[100, "intensities"] == 3.0
    assert out[0].loc[100, "mobility"] == pytest.approx(7 / 4)


def test_dense_ms2_is_binned_and_keeps_all_precursors(tmp_path):
    spectra = [
        spectrum(
            [200.0, 300.0],
            [10.0, 5.0],
            rt_seconds=i * 0.01,
            ms_level=2,
            precursor=400.0 + (i % 40),
            charge=2,
        )
        for i in range(2000)
    ]
    path = write_mzml(tmp_path / "dense.mzML", spectra)
    s = MSSonifier(
        filepath=path,
        ms_level=2,
        total_duration_minutes=2 / 60,
        sample_rate=8000,
    )
    s.load_and_preprocess_data()
    s.sonify(ms2_mode="dda")
    audio = s.current_audio_data
    assert audio.size == 2 * 8000 and np.isfinite(audio).all()
    # 2 s / 5 ms = 400 bins of 5 scans; each bin carries 5 precursors.
    # (Widen the m/z range so the precursors don't all clip to one pitch.)
    s.min_mz_overall, s.max_mz_overall = 100.0, 1000.0
    members = [[i for i in range(k * 5, k * 5 + 5)] for k in range(400)]
    pitches = s._segment_precursors(
        members,
        s.processed_spectra_dfs,
        s._frequency_function("inverse_log", (200.0, 4000.0)),
        "dda",
    )
    assert all(len(p) == 5 for p in pitches)


def _pitch_track_sonifier(mobility):
    # Four 250 ms scans: long enough that ADSR note edges don't smear
    # energy across the spectrum and mask the overtone measurement.
    s = MSSonifier(
        filepath="", total_duration_minutes=1 / 60, sample_rate=16000
    )
    s.processed_spectra_dfs = [
        pd.DataFrame(
            {"intensities": [1.0, 1.0], "mobility": mobility},
            index=pd.Index([150.0, 900.0]),
        )
        for _ in range(4)
    ]
    s.max_intensity_overall, s.min_mz_overall, s.max_mz_overall = (
        1.0,
        100.0,
        1000.0,
    )
    s.ion_mobility_data = {"unit": "1/K0", "range": (0.6, 1.6)}
    return s


def _harmonic_ratio(audio, sr, f0):
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1 / sr)
    at = lambda f: spec[np.argmin(np.abs(freqs - f))]  # noqa: E731
    return at(2 * f0) / at(f0)


@pytest.mark.parametrize("method", ["gradient", "adsr"])
@pytest.mark.parametrize("metered", [False, True])
def test_brightness_follows_mobility(method, metered):
    """The high-mobility pitch gets overtones, the low-mobility one none."""
    s = _pitch_track_sonifier([0.6, 1.6])
    kwargs = dict(
        method=method,
        frequency_mapping="linear",
        freq_range=(200.0, 1000.0),
        mobility={"brightness": 1.0},
        adsr_settings={"randomize": False},
    )
    if metered:
        kwargs["rhythm"] = {"tempo": 120, "accent": 0}
    s.sonify(**kwargs)
    audio = s.current_audio_data
    f_low_mob = s._mz_to_frequency(150.0, "linear", (200.0, 1000.0))
    f_high_mob = s._mz_to_frequency(900.0, "linear", (200.0, 1000.0))
    assert _harmonic_ratio(audio, 16000, f_high_mob) > 0.2
    assert _harmonic_ratio(audio, 16000, f_low_mob) < 0.02


def test_brightness_off_is_unchanged_and_requires_mobility():
    s = _pitch_track_sonifier([1.0, 1.0])
    s.sonify(frequency_mapping="linear")
    plain = s.current_audio_data.copy()
    s.sonify(frequency_mapping="linear", mobility=None)
    np.testing.assert_array_equal(plain, s.current_audio_data)
    for df in s.processed_spectra_dfs:
        del df["mobility"]
    with pytest.raises(ValueError, match="ion mobility"):
        s.sonify(mobility={"brightness": 0.5})
    with pytest.raises(TypeError):
        s.sonify(mobility="pan")


def test_dia_reference_and_tones(tims_like):
    s = MSSonifier(
        filepath=tims_like,
        ms_level=2,
        total_duration_minutes=0.05,
        sample_rate=8000,
    )
    s.load_and_preprocess_data(rt_range=(0.0, 25 / 60))
    s.sonify()
    plain = s.current_audio_data.copy()
    s.load_ms1_reference()
    ref_rts = [r["retention_time"] for r in s.ms1_reference_spectra]
    # Default window: the MS2 span (0-~23 s) plus 0.5 min on each side.
    assert ref_rts and max(ref_rts) <= max(s.retention_time_list) + 0.5
    s.sonify(ms2_mode="dia")
    assert not np.allclose(plain, s.current_audio_data)


def test_raw_mobility_map_keeps_separate_bands(tmp_path):
    """Two ion populations at the same m/z but different mobility stay
    separate in the map (the binned mean would merge them)."""
    spectra = [
        spectrum(
            np.array([500.1, 500.2]),
            np.array([10.0, 10.0]),
            rt_seconds=i,
            mobility=np.array([0.8, 1.2]),
        )
        for i in range(3)
    ]
    path = write_mzml(tmp_path / "bands.mzML", spectra)
    run = io.load_spectra(path, ms_level=1)
    assert run.spectra[0].loc[500, "mobility"] == pytest.approx(1.0)
    row = run.mobility_map[500]
    assert (row > 0).sum() == 2  # two mobility bins, not one
    assert run.mobility_profiles.shape == (3, io.MAP_MOBILITY_BINS)
    again = io.load_spectra(path, ms_level=1)
    assert again.from_cache
    np.testing.assert_allclose(again.mobility_map, run.mobility_map)


def test_cache_list_remove_clear(tims_like):
    assert io.list_cache() == []
    io.load_spectra(tims_like, ms_level=1)
    io.load_spectra(tims_like, ms_level=2, rt_range=(0.0, 0.5))
    entries = io.list_cache()
    assert len(entries) == 2
    ms2 = next(e for e in entries if e["ms_level"] == 2)
    assert ms2["file"] == "tims.mzML" and ms2["rt_range"] == [0.0, 0.5]
    assert ms2["spectra"] > 0 and ms2["bytes"] > 0
    assert io.remove_cache(ms2["key"]) is True
    assert io.remove_cache(ms2["key"]) is False
    assert [e["ms_level"] for e in io.list_cache()] == [1]
    # The removed load is read from the file again (not the cache).
    assert not io.load_spectra(
        tims_like, ms_level=2, rt_range=(0.0, 0.5)
    ).from_cache
    assert io.clear_cache() == 2
    assert io.list_cache() == []
    with pytest.raises(ValueError):
        io.remove_cache("../etc/passwd")


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None
    and __import__("importlib").util.find_spec("imageio_ffmpeg") is None,
    reason="no ffmpeg",
)
@pytest.mark.parametrize("level", [1, 2])
def test_raw_data_video_all_panels(tims_like, tmp_path, level):
    """Loaded MS1 or MS2: the other level is read from the file, and every
    panel renders in sync with the audio."""
    from ms_music import visualizations as viz

    s = MSSonifier(
        filepath=tims_like,
        ms_level=level,
        total_duration_minutes=1 / 60,
        sample_rate=8000,
    )
    s.load_and_preprocess_data()
    s.sonify(ms2_mode="dda" if level == 2 else None)
    frames = []
    out = viz.create_raw_data_video(
        s,
        str(tmp_path / "raw.mp4"),
        fps=4,
        dpi=40,
        progress_callback=lambda i, n: frames.append((i, n)),
    )
    assert out and os.path.getsize(out) > 0
    audio_seconds = s.current_audio_data.shape[-1] / 8000
    assert frames[-1][1] == max(1, int(audio_seconds * 4))
    with pytest.raises(ValueError):
        viz.create_raw_data_video(
            s,
            str(tmp_path / "x.mp4"),
            show_chromatogram=False,
            show_ms1=False,
            show_ms2=False,
            show_mobility=False,
        )
