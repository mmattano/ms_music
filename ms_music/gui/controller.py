"""Engine-facing state for the GUI.

This module knows nothing about the UI toolkit. It wraps :class:`MSSonifier`
and :class:`MSSonifierMidi`, tracks the current audio buffer and the applied
effect chain, and serializes audio for the browser player. The GUI calls it
from worker threads with plain Python values (see :mod:`ms_music.gui.app`).
"""

from __future__ import annotations

import ast
import glob
import inspect
import io
import os

import numpy as np

from .. import effects as audio_effects
from ..midi_generator import MidiConfig, MSSonifierMidi
from ..musical_quantization import MusicalNoteQuantizer
from ..sonifier import MSSonifier


FREQUENCY_MAPPINGS = [
    "inverse_log",
    "power_law",
    "musical_octaves",
    "chromatic",
    "linear",
]

# Label for "the current audio" when picking versions to compare.
CURRENT = "Current audio"

ROOT_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Effects whose signature is not a single-buffer transform, so they cannot be
# driven through MSSonifier.apply_effect / the Effects tab.
_EFFECTS_EXCLUDED = {"apply_crossfade"}


def parse_value(text, default):
    """Parse a form field back into a Python value.

    Tries a literal eval (numbers, tuples, lists, booleans, None); on failure
    keeps the raw string, which is what most string-valued effect params want.
    """
    text = str(text).strip()
    if text == "":
        return default
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def function_param_spec(func, skip=(), skip_first=0, overrides=None):
    """``[(name, default), ...]`` for ``func``'s keyword parameters.

    Drops the first ``skip_first`` parameters, anything named in ``skip``,
    and *args/**kwargs. ``overrides`` replaces defaults (e.g. faster GUI
    settings for videos)."""
    overrides = overrides or {}
    spec = []
    params = list(inspect.signature(func).parameters.items())[skip_first:]
    for pname, param in params:
        if pname in skip:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        default = None if param.default is inspect._empty else param.default
        spec.append((pname, overrides.get(pname, default)))
    return spec


class RenderCancelled(Exception):
    """Raised from a video progress callback to stop rendering."""


class SonifierController:
    def __init__(self):
        self.sonifier: MSSonifier | None = None
        self.sample_rate: int = 44100
        self.source_path: str | None = None
        self.source_kind: str | None = None  # "mzml" | "fid" | "wav"

        # base_audio is the freshly generated buffer *before* any effects, so
        # the Effects tab can reset/undo without regenerating.
        self.base_audio: np.ndarray | None = None
        self.effect_chain: list[tuple[str, dict]] = []
        self.last_label = "audio"

        # Named copies of earlier audio, for comparison plots and videos.
        self._snapshots: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ data
    def load_mzml(self, filepath, ms_level, duration_minutes, sample_rate,
                  scan_ratio_range=None):
        self.sonifier = MSSonifier(
            filepath=filepath,
            ms_level=int(ms_level),
            total_duration_minutes=float(duration_minutes),
            sample_rate=int(sample_rate),
        )
        self.sonifier.load_and_preprocess_data(scan_ratio_range=scan_ratio_range)
        if not self.sonifier.processed_spectra_dfs:
            raise RuntimeError(
                "No spectra were loaded. Check the file and MS level."
            )
        self._set_sample_rate(int(sample_rate))
        self.source_path = filepath
        self.source_kind = "mzml"
        self._reset_audio_state()
        n = len(self.sonifier.processed_spectra_dfs)
        return f"Loaded {n} MS{int(ms_level)} scans from {os.path.basename(filepath)}."

    def load_fid(self, filepath, duration_minutes, sample_rate,
                 original_sample_rate, conversion_factor):
        self.sonifier = MSSonifier(
            filepath="",
            total_duration_minutes=float(duration_minutes),
            sample_rate=int(sample_rate),
        )
        self.sonifier.load_fid_data(
            filepath,
            original_sample_rate=float(original_sample_rate),
            conversion_factor=float(conversion_factor),
        )
        self._set_sample_rate(int(sample_rate))
        self.source_path = filepath
        self.source_kind = "fid"
        self.base_audio = self._current_or_none()
        self.effect_chain = []
        self.last_label = "FID"
        secs = 0 if self.base_audio is None else len(self.base_audio) / self.sample_rate
        return f"Loaded FID {os.path.basename(filepath)} ({secs:.1f}s audio)."

    def load_wav(self, filepath, sample_rate):
        import librosa

        audio, sr = librosa.load(filepath, sr=int(sample_rate), mono=True)
        self.sonifier = MSSonifier(
            filepath="", sample_rate=int(sr),
        )
        self.sonifier.current_audio_data = audio.astype(np.float32)
        self._set_sample_rate(int(sr))
        self.source_path = filepath
        self.source_kind = "wav"
        self.base_audio = audio.astype(np.float32)
        self.effect_chain = []
        self.last_label = os.path.splitext(os.path.basename(filepath))[0]
        return (f"Loaded audio {os.path.basename(filepath)} "
                f"({len(audio) / sr:.1f}s at {sr} Hz).")

    # -------------------------------------------------------------- synthesis
    def sonify(self, method="gradient", frequency_mapping="inverse_log",
               freq_range=(200.0, 4000.0), scale=None, rhythm=None,
               **options):
        """Run :meth:`MSSonifier.sonify` with the GUI's options.

        ``options`` are passed straight through (root_note, tuning_freq,
        edo_divisions, use_just_intonation, use_log_distance,
        adsr_settings, ms2_mode, max_peaks_per_scan)."""
        self._require_ms_data()
        self.sonifier.sonify(
            method=method,
            frequency_mapping=frequency_mapping,
            freq_range=tuple(freq_range),
            scale=scale,
            rhythm=rhythm,
            **options,
        )
        self._capture_base("Sonification produced no audio.")
        grid = self.sonifier.rhythm_grid
        parts = [method]
        if scale:
            parts.append(f"{scale} in {options.get('root_note', 'C')}")
        if grid is not None:
            parts.append(grid.config.describe())
        self.last_label = " · ".join(parts)
        return f"Generated {self.last_label} ({self._duration_str()})."

    def rhythm_grid(self):
        """The grid of the current (un-effected) audio, if it was metered."""
        if self.sonifier is None or self.base_audio is None:
            return None
        return getattr(self.sonifier, "rhythm_grid", None)

    # ---------------------------------------------------------------- effects
    @staticmethod
    def list_effects():
        names = []
        for name in dir(audio_effects):
            if name.startswith("apply_") and name not in _EFFECTS_EXCLUDED:
                names.append(name[len("apply_"):])
        return sorted(names)

    @staticmethod
    def effect_param_spec(effect_name):
        """Return [(param_name, default), ...] for an effect, skipping the
        audio_data / sample_rate positional args."""
        func = getattr(audio_effects, f"apply_{effect_name}")
        return function_param_spec(func, skip_first=2)

    def apply_effect(self, effect_name, params):
        self._require_audio()
        self.sonifier.apply_effect(effect_name, params)
        self.effect_chain.append((effect_name, params))
        return f"Applied {effect_name}. Chain: {self._chain_str()}"

    def reset_effects(self):
        self._require_base()
        self.sonifier.current_audio_data = self.base_audio.copy()
        self.effect_chain = []
        return "Reset to the un-effected audio."

    def undo_effect(self):
        self._require_base()
        if not self.effect_chain:
            return "No effects to undo."
        self.effect_chain.pop()
        self.sonifier.current_audio_data = self.base_audio.copy()
        for name, params in self.effect_chain:
            self.sonifier.apply_effect(name, params)
        return f"Undone. Chain: {self._chain_str()}"

    def chain_labels(self):
        return [name for name, _ in self.effect_chain]

    # -------------------------------------------------------------- snapshots
    def default_snapshot_label(self):
        label = self.last_label
        if self.effect_chain:
            label += " + " + " + ".join(self.chain_labels())
        return label

    def keep_snapshot(self, label=None):
        """Store a copy of the current audio under a unique label."""
        self._require_audio()
        base = (label or "").strip() or self.default_snapshot_label()
        label, n = base, 2
        while label in self._snapshots or label == CURRENT:
            label, n = f"{base} ({n})", n + 1
        self._snapshots[label] = self._current_or_none().copy()
        return label

    def remove_snapshot(self, label):
        self._snapshots.pop(label, None)

    def snapshot_labels(self):
        return list(self._snapshots)

    def audio_versions(self, labels):
        """``{label: audio}`` for the chosen snapshot labels; ``CURRENT``
        selects the current audio."""
        out = {}
        for label in labels:
            if label == CURRENT:
                audio = self._current_or_none()
                if audio is not None and audio.size:
                    out["current"] = audio
            elif label in self._snapshots:
                out[label] = self._snapshots[label]
        return out

    def _set_sample_rate(self, sample_rate):
        """Adopt a new sample rate; snapshots at another rate can't be
        compared with the new audio, so they are dropped."""
        if self._snapshots and int(sample_rate) != self.sample_rate:
            self._snapshots.clear()
        self.sample_rate = int(sample_rate)

    # ---------------------------------------------------- plots and videos
    def has_ms_data(self):
        return bool(self.sonifier is not None
                    and self.sonifier.processed_spectra_dfs)

    def render_plot(self, entry, params, version_labels=()):
        """Build the figure for a catalog entry (runs on a worker thread)."""
        fig = entry.call(self._context(version_labels), params)
        if fig is None:
            raise RuntimeError(f"{entry.label}: nothing to plot for this data.")
        return fig

    def render_video(self, entry, params, output_path, version_labels=(),
                     progress=None, cancelled=lambda: False):
        """Render a catalog video to ``output_path`` (worker thread).

        ``progress(frame, total)`` is called per frame; when ``cancelled()``
        turns true the render stops with :class:`RenderCancelled`."""
        import shutil

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "Video export needs ffmpeg on your PATH "
                "(macOS: brew install ffmpeg; Linux: apt install ffmpeg).")

        def on_frame(frame, total):
            if cancelled():
                raise RenderCancelled()
            if progress is not None:
                progress(frame, total)

        result = entry.call(
            self._context(version_labels),
            dict(params, output_path=output_path, progress_callback=on_frame),
        )
        if cancelled():
            raise RenderCancelled()
        if result is None or not os.path.exists(output_path):
            raise RuntimeError(
                f"{entry.label} failed; see the log for the reason.")
        return output_path

    def _context(self, version_labels):
        from .catalog import Context

        return Context(
            sonifier=self.sonifier,
            audio=self._current_or_none(),
            sample_rate=self.sample_rate,
            versions=self.audio_versions(version_labels),
        )

    # ------------------------------------------------------------------- midi
    def generate_midi(self, output_path, scale, root_note, tempo, meter,
                      quantization_mode, edo_divisions, use_just_intonation,
                      duration_seconds, enable_mz_clustering, mz_tolerance_ppm,
                      intensity_threshold_percentile, max_peaks_per_scan,
                      frequency_mapping, instrument, max_simultaneous_notes,
                      export_note_data, num_voices=1, separate_files=False):
        if self.source_kind != "mzml" or not self.source_path:
            raise RuntimeError("MIDI export needs an mzML file (Data tab).")

        config = MidiConfig(
            scale=scale,
            root_note=root_note,
            tempo=int(tempo),
            meter=meter,
            quantization_mode=quantization_mode,
            edo_divisions=int(edo_divisions),
            use_just_intonation=bool(use_just_intonation),
        )
        midi = MSSonifierMidi(
            filepath=self.source_path,
            config=config,
            sample_rate=self.sample_rate,
            enable_mz_clustering=bool(enable_mz_clustering),
            mz_tolerance_ppm=float(mz_tolerance_ppm),
            use_ppm_tolerance=True,
        )
        midi.load_and_analyze_data(total_duration_seconds=float(duration_seconds))
        midi.setup_musical_system(
            scale=scale,
            root_note=root_note,
            tempo=int(tempo),
            meter=meter,
            quantization_mode=quantization_mode,
        )
        midi.detect_and_quantize_peaks(
            intensity_threshold_percentile=float(intensity_threshold_percentile),
            max_peaks_per_scan=int(max_peaks_per_scan),
            frequency_mapping=frequency_mapping,
        )
        num_voices = max(1, int(num_voices))
        separate_files = bool(separate_files) and num_voices > 1
        midi.generate_midi_file(
            output_path=output_path,
            track_name="MS Sonification",
            instrument=int(instrument),
            max_simultaneous_notes=int(max_simultaneous_notes),
            export_note_data=bool(export_note_data),
            num_voices=num_voices,
            separate_files=separate_files,
        )
        report = midi.get_analysis_report()
        # Collect everything the generator wrote (voice files, note CSVs).
        base = output_path.rsplit(".mid", 1)[0]
        written = sorted(set(glob.glob(base + "*.mid") + glob.glob(base + "*.csv")))
        message = (
            f"Generated MIDI. Voices: {num_voices}. "
            f"Notes: {report.get('note_count')}, "
            f"unique pitches: {report.get('unique_pitches')}, "
            f"duration: {report.get('total_duration_seconds')}s."
        )
        return message, written

    # ------------------------------------------------------------ save / play
    def save_audio(self, filepath, normalize=True):
        self._require_audio()
        self.sonifier.save_audio(filepath, normalize=normalize)
        return f"Saved audio to {filepath}."

    def current_audio(self):
        return self._current_or_none()

    def current_wav_bytes(self):
        """The current buffer as a peak-normalized 16-bit WAV file in memory."""
        audio = self._current_or_none()
        if audio is None or audio.size == 0:
            return None
        peak = float(np.max(np.abs(audio))) or 1.0
        norm = (audio / peak).astype(np.float32)

        from ..io import save_wav

        buf = io.BytesIO()
        save_wav(buf, norm, self.sample_rate)
        return buf.getvalue()

    def waveform_preview(self, max_points=4000):
        """Min/max envelope of the current buffer: (times, lows, highs).

        Per-bucket min/max keeps transients visible, unlike plain striding.
        """
        audio = self._current_or_none()
        if audio is None or audio.size == 0:
            return None
        n_buckets = max(1, min(max_points, audio.size))
        usable = audio[: audio.size - audio.size % n_buckets]
        if usable.size == 0:
            usable = audio
            n_buckets = audio.size
        buckets = usable.reshape(n_buckets, -1)
        times = (np.arange(n_buckets) * buckets.shape[1]) / self.sample_rate
        return times, buckets.min(axis=1), buckets.max(axis=1)

    def duration_seconds(self):
        audio = self._current_or_none()
        return 0.0 if audio is None else len(audio) / self.sample_rate

    @staticmethod
    def all_scales():
        """Every selectable scale: 12-TET, EDO, and just-intonation names."""
        names = list(MusicalNoteQuantizer.SCALES.keys())
        names += [s for s in MusicalNoteQuantizer.EDO_SCALES
                  if s not in names]
        names += [s for s in MusicalNoteQuantizer.JUST_INTONATION_RATIOS
                  if s not in names]
        return names

    # ----------------------------------------------------------------- helpers
    def _current_or_none(self):
        if self.sonifier is None:
            return None
        return getattr(self.sonifier, "current_audio_data", None)

    def _capture_base(self, empty_msg):
        audio = self._current_or_none()
        if audio is None or audio.size == 0:
            raise RuntimeError(empty_msg)
        self.base_audio = audio.copy()
        self.effect_chain = []

    def _reset_audio_state(self):
        self.base_audio = None
        self.effect_chain = []
        if self.sonifier is not None:
            self.sonifier.current_audio_data = None

    def _duration_str(self):
        audio = self._current_or_none()
        if audio is None:
            return "0.0s"
        return f"{len(audio) / self.sample_rate:.1f}s"

    def _chain_str(self):
        return " -> ".join(self.chain_labels()) or "(empty)"

    def _require_ms_data(self):
        if self.sonifier is None or not self.sonifier.processed_spectra_dfs:
            raise RuntimeError("Load an mzML file first (Data tab).")

    def _require_audio(self):
        if self._current_or_none() is None:
            raise RuntimeError("No audio yet. Generate or load audio first.")

    def _require_base(self):
        if self.base_audio is None:
            raise RuntimeError("No base audio to modify. Generate audio first.")
