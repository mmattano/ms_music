"""Tests for the GUI's engine layer and a launch smoke test of the web app.

The controller is UI-agnostic, so it is tested directly with a synthetic
WAV file. The launch test starts the real server in a subprocess and checks
the page is served.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from ms_music.gui.controller import SonifierController, parse_value
from ms_music.io import save_wav


@pytest.fixture
def loaded_controller(tmp_path):
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    path = tmp_path / "tone.wav"
    save_wav(
        str(path), (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr
    )
    c = SonifierController()
    c.load_wav(str(path), sr)
    return c


def test_parse_value():
    assert parse_value("0.5", None) == 0.5
    assert parse_value("(1, 2)", None) == (1, 2)
    assert parse_value("hann", None) == "hann"
    assert parse_value("  ", 3) == 3


def test_wav_bytes_and_preview(loaded_controller):
    c = loaded_controller
    wav = c.current_wav_bytes()
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    times, lows, highs = c.waveform_preview(max_points=100)
    assert len(times) == len(lows) == len(highs) == 100
    assert np.all(lows <= highs)
    assert c.duration_seconds() == pytest.approx(0.5, abs=0.01)


def test_empty_controller_has_no_audio():
    c = SonifierController()
    assert c.current_wav_bytes() is None
    assert c.waveform_preview() is None


def test_effect_chain_apply_undo_reset(loaded_controller):
    c = loaded_controller
    base = c.current_audio().copy()
    c.apply_effect("fade_in", {})
    c.apply_effect("normalize", {})
    assert c.chain_labels() == ["fade_in", "normalize"]
    c.undo_effect()
    assert c.chain_labels() == ["fade_in"]
    c.reset_effects()
    assert c.chain_labels() == []
    np.testing.assert_array_equal(c.current_audio(), base)


def test_every_effect_has_a_param_spec():
    for name in SonifierController.list_effects():
        spec = SonifierController.effect_param_spec(name)
        assert all(isinstance(p, str) for p, _ in spec)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_web_app_serves_page():
    port = _free_port()
    # NiceGUI switches to its own test mode when it sees pytest's env vars.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ms_music.gui",
            "--no-browser",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        deadline = time.time() + 90
        html = None
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=2
                ) as r:
                    html = r.read().decode()
                break
            except OSError:
                time.sleep(0.5)
        assert html is not None, (
            proc.stdout.read().decode() if proc.poll() else "timeout"
        )
        assert "ms_music" in html
        # No audio has been generated yet, so the audio route has nothing.
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/ms_music/audio/0.wav", timeout=5
            )
        assert err.value.code == 404
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ------------------------------------------------ snapshots, plots, videos
import shutil  # noqa: E402

import pandas as pd  # noqa: E402

from ms_music.gui import catalog  # noqa: E402
from ms_music.gui.controller import (  # noqa: E402
    CURRENT,
    RenderCancelled,
    function_param_spec,
)


@pytest.fixture
def ms_controller():
    """A controller with small synthetic spectra and metered audio."""
    from ms_music import MSSonifier

    rng = np.random.default_rng(1)
    pool = np.round(rng.uniform(100, 800, 20), 3)
    dfs = []
    for _ in range(30):
        mz = np.sort(rng.choice(pool, 8, replace=False))
        dfs.append(
            pd.DataFrame(
                {
                    "intensities": rng.uniform(1, 50, 8),
                    "mobility": 0.6 + mz / 1000,
                },
                index=pd.Index(mz, name="mz"),
            )
        )
    s = MSSonifier(
        filepath="", total_duration_minutes=2 / 60, sample_rate=8000
    )
    s.processed_spectra_dfs = dfs
    # MS2-style metadata and ion mobility, so every catalog plot applies.
    s.retention_time_list = [i * 0.1 for i in range(30)]
    s.precursor_mz_list = [float(pool[i % 20]) for i in range(30)]
    s.charge_list = [2 + i % 2 for i in range(30)]
    s.ion_mobility_data = {
        "unit": "1/K0",
        "range": (0.7, 1.4),
        "per_spectrum": [1.0] * 30,
    }
    s.max_intensity_overall = max(d["intensities"].max() for d in dfs)
    s.min_mz_overall, s.max_mz_overall = float(pool.min()), float(pool.max())
    c = SonifierController()
    c.sonifier, c.sample_rate, c.source_kind = s, 8000, "mzml"
    c.sonify(scale="major", rhythm={"meter": "3/4", "tempo": 120})
    return c


def test_sonify_labels_and_rhythm(ms_controller):
    c = ms_controller
    assert c.rhythm_grid() is not None
    assert "major in C" in c.last_label and "3/4" in c.last_label


def test_snapshots_unique_labels_and_versions(ms_controller):
    c = ms_controller
    a = c.keep_snapshot("first")
    b = c.keep_snapshot("first")
    assert (a, b) == ("first", "first (2)")
    assert c.keep_snapshot("") == c.default_snapshot_label()
    versions = c.audio_versions([CURRENT, "first", "missing"])
    assert list(versions) == ["current", "first"]
    c.remove_snapshot("first")
    assert "first" not in c.snapshot_labels()


def test_snapshots_dropped_when_sample_rate_changes(ms_controller):
    c = ms_controller
    c.keep_snapshot("x")
    c._set_sample_rate(8000)
    assert c.snapshot_labels() == ["x"]
    c._set_sample_rate(44100)
    assert c.snapshot_labels() == []


def test_function_param_spec_skip_and_overrides():
    def f(audio, sr, n=3, *args, cmap="viridis", flag=False, **kw):
        pass

    spec = function_param_spec(f, skip={"audio", "sr"}, overrides={"n": 7})
    assert spec == [("n", 7), ("cmap", "viridis"), ("flag", False)]


@pytest.mark.parametrize(
    "entry", catalog.PLOTS + catalog.VIDEOS, ids=lambda e: e.label
)
def test_catalog_forms_build(entry):
    spec = function_param_spec(
        entry.func, skip=catalog.skip_names(), overrides=entry.defaults
    )
    names = [n for n, _ in spec]
    assert not set(names) & catalog.skip_names()
    for key in entry.choices:
        assert key in names, f"{entry.label}: choice for unknown {key}"


@pytest.mark.parametrize("entry", catalog.PLOTS, ids=lambda e: e.label)
def test_every_plot_renders(ms_controller, entry):
    import matplotlib.pyplot as plt

    from ms_music.gui.components import render_figure_png

    c = ms_controller
    if catalog.STEREO in entry.needs:
        c.sonify(mobility={"pan": 1.0})
    c.keep_snapshot("other")
    labels = [CURRENT, "other"]
    params = dict(
        function_param_spec(
            entry.func, skip=catalog.skip_names(), overrides=entry.defaults
        )
    )
    png = render_figure_png(c.render_plot(entry, params, labels), dpi=40)
    assert png[:4] == b"\x89PNG"
    plt.close("all")


def test_unmet_needs():
    plot = catalog.find(catalog.PLOTS, "Difference spectrogram")
    assert catalog.unmet_need(
        plot, has_audio=True, has_ms_data=True, n_versions=3
    )
    assert (
        catalog.unmet_need(
            plot, has_audio=True, has_ms_data=True, n_versions=2
        )
        is None
    )
    mapping = catalog.find(catalog.PLOTS, "m/z → frequency mapping")
    assert "mzML" in catalog.unmet_need(
        mapping, has_audio=True, has_ms_data=False, n_versions=0
    )


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


@needs_ffmpeg
def test_video_render_reports_progress(ms_controller, tmp_path):
    c = ms_controller
    c.sonifier.current_audio_data = c.current_audio()[:8000]  # 1 s
    entry = catalog.find(catalog.VIDEOS, "Spectrogram & waveform playback")
    calls = []
    out = c.render_video(
        entry,
        {"fps": 5, "dpi": 40},
        str(tmp_path / "v.mp4"),
        progress=lambda i, n: calls.append((i, n)),
    )
    assert os.path.getsize(out) > 0
    assert calls and calls[-1][1] == 5


@needs_ffmpeg
def test_video_render_can_be_cancelled(ms_controller, tmp_path):
    c = ms_controller
    entry = catalog.find(catalog.VIDEOS, "3D spectrum waterfall")
    with pytest.raises(RenderCancelled):
        c.render_video(
            entry,
            {"fps": 5, "dpi": 40, "duration_seconds": 4},
            str(tmp_path / "v.mp4"),
            cancelled=lambda: True,
        )


def test_ms2_capabilities_and_mobility_flags(ms_controller):
    c = ms_controller
    assert c.has_mobility()
    caps = c.ms2_capabilities()
    assert not caps["dda"] and "MS level 2" in caps["dda_reason"]
    c.sonifier.ms_level = 2
    c.sonifier.isolation_window_list = [(499.0, 502.0)] * 30
    caps = c.ms2_capabilities()
    assert caps["dda"] and not caps["dia"]
    assert "MS1 reference" in caps["dia_reason"]


def test_brightness_through_controller(ms_controller):
    c = ms_controller
    msg = c.sonify(mobility={"brightness": 0.8, "pan": 1.0})
    assert "Generated" in msg and np.isfinite(c.current_audio()).all()


def test_stereo_field_rejects_mono_and_needs_gate(ms_controller):
    from ms_music import visualizations as viz

    with pytest.raises(ValueError, match="stereo"):
        viz.plot_stereo_field(np.zeros(1000), 8000)
    entry = catalog.find(catalog.PLOTS, "Stereo field")
    assert "stereo" in catalog.unmet_need(
        entry, has_audio=True, has_ms_data=True, n_versions=1
    )


@needs_ffmpeg
def test_spatial_stage_video(ms_controller, tmp_path):
    c = ms_controller
    c.sonify(
        mobility={
            "pan": 1.0,
            "effects": [("reverb", "dry_wet_mix", 0.05, 0.6)],
        }
    )
    c.sonifier.current_audio_data = c.current_audio()[:, :8000]  # 1 s
    entry = catalog.find(catalog.VIDEOS, "Spatial stage")
    calls = []
    out = c.render_video(
        entry,
        {"fps": 5, "dpi": 40},
        str(tmp_path / "s.mp4"),
        progress=lambda i, n: calls.append((i, n)),
    )
    assert os.path.getsize(out) > 0 and calls[-1][1] == 5
