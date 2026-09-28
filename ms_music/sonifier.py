import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
from scipy import signal
import random
import librosa

from . import io
from . import effects as audio_effects
from .mobility import MobilityConfig, ToneBus
from .musical_quantization import MusicalNoteQuantizer
from .rhythm import (
    RhythmConfig,
    build_grid,
    find_notes,
    note_gain,
    note_span,
    resample_spectra,
)


class MSSonifier:
    """
    Main class for orchestrating (hehe) the sonification of
    mass spectrometry data.
    """

    def __init__(
        self,
        filepath: str,
        ms_level: int = 1,
        total_duration_minutes: float = 10,
        sample_rate: int = 44100,
    ):
        self.filepath = filepath
        self.ms_level = ms_level
        self.total_duration_seconds = total_duration_minutes * 60
        self.sample_rate = sample_rate

        self.raw_spectra = None
        self.processed_spectra_dfs = None
        self.max_intensity_overall = 0
        self.min_mz_overall = 0
        self.max_mz_overall = 0

        self.current_audio_data = None
        self.note_quantizer = None
        self.rhythm_grid = None

        self.ion_mobility_data = None  # {"unit", "range", "per_spectrum"}
        self.loaded_run = None
        self.charge_list = None
        self.mobility_config = None  # MobilityConfig of last sonify
        self.mobility_norm = None  # (low, high) mobility -> 0..1
        self.segment_starts = None  # audio time (s) of each segment
        self.segment_spectra = None  # spectra sounding per segment
        self.segment_members = None  # source scan indices per segment
        self.frequency_fn = None  # m/z -> Hz of the last sonify
        self.precursor_mz_list = None  # DDA: precursor m/z per scan
        self.isolation_window_list = None  # DIA: (lo_mz, hi_mz) per scan
        self.retention_time_list = None  # Retention time per scan (min)
        self.ms1_reference_spectra = None  # MS1 spectra for DIA lookup

        print(f"MSSonifier initialized for file: {os.path.basename(filepath)}")
        print(
            f"Target audio duration: {total_duration_minutes} minutes, "
            f"Sample rate: {sample_rate} Hz"
        )

    def load_and_preprocess_data(
        self,
        scan_ratio_range=None,
        rt_range=None,
        mobility_range=None,
        cache=True,
        progress=None,
    ):
        """Load and bin the spectra of ``self.ms_level``.

        Streams the file and bins each spectrum to integer m/z as it is
        read, so large runs (e.g. timsTOF with ~300k peaks per frame) fit in
        memory. Ion mobility, when present, is kept per bin (column
        ``mobility``) and summarized in ``self.ion_mobility_data``.

        Args:
            scan_ratio_range: Optional (start, end) fractions (0-1) of the
                loaded spectra to keep.
            rt_range: Optional (start, end) retention time window, minutes.
            mobility_range: Optional (low, high) ion mobility window.
            cache: Reuse a binned copy from an earlier load of the same file
                and settings (see :func:`ms_music.io.load_spectra`).
            progress: Called as ``progress(spectra_read, total)``.
        """
        if scan_ratio_range:
            if not (
                isinstance(scan_ratio_range, tuple)
                and len(scan_ratio_range) == 2
                and 0.0 <= scan_ratio_range[0] < scan_ratio_range[1] <= 1.0
            ):
                print(
                    f"Warning: Invalid scan_ratio_range {scan_ratio_range}. "
                    f"Processing all scans."
                )
                scan_ratio_range = None

        run = io.load_spectra(
            self.filepath,
            ms_level=self.ms_level,
            rt_range=rt_range,
            mobility_range=mobility_range,
            cache=cache,
            progress=progress,
        )
        if run is None:
            print("No spectra found during loading.")
            return

        n = len(run.spectra)
        lo, hi = 0, n
        if scan_ratio_range:
            lo, hi = int(n * scan_ratio_range[0]), int(n * scan_ratio_range[1])
            print(
                f"Selected {hi - lo} scans based on scan_ratio_range "
                f"{scan_ratio_range}."
            )
            if hi <= lo:
                print("No spectra selected after applying scan_ratio_range.")
                return

        self.loaded_run = run
        self.processed_spectra_dfs = run.spectra[lo:hi]
        self.precursor_mz_list = run.precursor_mz[lo:hi]
        self.isolation_window_list = run.isolation_windows[lo:hi]
        self.retention_time_list = run.retention_times[lo:hi]
        self.charge_list = run.charges[lo:hi]
        if scan_ratio_range:
            self._recompute_overall_ranges(lo, hi)
        else:
            self.max_intensity_overall = run.max_intensity
            self.min_mz_overall = run.min_mz
            self.max_mz_overall = run.max_mz

        self.ion_mobility_data = None
        if run.has_mobility:
            self.ion_mobility_data = {
                "unit": run.mobility_unit,
                "range": run.mobility_range,
                "per_spectrum": run.spectrum_mobility[lo:hi],
                # Raw-resolution histograms (per-peak mobility data only):
                # the (m/z, mobility) map covers the whole loaded run.
                "map": run.mobility_map,
                "profiles": (
                    None
                    if run.mobility_profiles is None
                    else run.mobility_profiles[lo:hi]
                ),
                "edges": run.mobility_edges,
            }
        print(
            f"Using {len(self.processed_spectra_dfs)} MS{self.ms_level} "
            f"spectra"
            + (
                f" with ion mobility ({run.mobility_unit})"
                if run.has_mobility
                else ""
            )
        )

    @property
    def has_ion_mobility(self):
        return (
            bool(self.ion_mobility_data)
            and bool(self.processed_spectra_dfs)
            and any(
                "mobility" in df.columns for df in self.processed_spectra_dfs
            )
        )

    def _recompute_overall_ranges(self, lo, hi):
        """Intensity and m/z ranges of the kept slice ``lo:hi``, as if only
        those spectra had been loaded."""
        dfs = [df for df in self.processed_spectra_dfs if not df.empty]
        if not dfs:
            return
        self.max_intensity_overall = float(
            max(self.loaded_run.spectrum_max_intensity[lo:hi])
        )
        self.min_mz_overall = float(min(df.index.min() for df in dfs))
        self.max_mz_overall = float(max(df.index.max() for df in dfs))

    def load_ms1_reference(
        self,
        filepath: str = None,
        top_peaks: int = 2000,
        progress=None,
        rt_range=None,
    ):
        """
        Load MS1 spectra as a reference for DIA precursor tone generation.

        In DIA mode each MS2 scan has an isolation window rather than a
        single selected precursor. This loads the MS1 peaks (m/z, intensity,
        retention time) so that, when rendering a DIA MS2 scan, all MS1
        peaks inside the isolation window can play as precursor tones.

        Args:
            filepath: mzML file with the MS1 data. Defaults to the file
                already loaded into this sonifier.
            top_peaks: Keep only the N most intense peaks of each MS1
                spectrum (at full m/z precision), bounding memory for large
                runs. None keeps all.
            progress: Called as ``progress(spectra_read, total)``.
            rt_range: Optional (start, end) retention time window in
                minutes; by default the loaded MS2 data's span plus half a
                minute on each side, so a windowed load stays quick.
        """
        if filepath is None:
            filepath = self.filepath
        if rt_range is None:
            rts = [
                r for r in (self.retention_time_list or []) if r is not None
            ]
            if rts:
                rt_range = (max(0.0, min(rts) - 0.5), max(rts) + 0.5)

        ref = io.load_reference_peaks(
            filepath,
            ms_level=1,
            top_peaks=top_peaks,
            progress=progress,
            rt_range=rt_range,
        )
        if not ref:
            print("Warning: no MS1 spectra found for DIA reference.")
            self.ms1_reference_spectra = None
            return
        self.ms1_reference_spectra = ref
        print(
            f"Loaded {len(ref)} MS1 reference spectra for DIA mode "
            f"(RT {ref[0]['retention_time']:.2f}–"
            f"{ref[-1]['retention_time']:.2f} min)"
        )

    def _dia_precursor_peaks(self, scan_idx):
        """
        Return a list of (mz, amplitude) pairs for all MS1 peaks within the
        isolation window of the given MS2 scan index.

        The amplitude for each peak is its intensity normalised by the global
        maximum intensity of the loaded MS2 data so that precursor tones sit
        in the same dynamic range as the fragment tones.
        """
        if not self.ms1_reference_spectra:
            return []

        window = (
            self.isolation_window_list[scan_idx]
            if self.isolation_window_list
            and scan_idx < len(self.isolation_window_list)
            else None
        )
        if window is None:
            return []

        win_lo, win_hi = window

        # Find the closest MS1 spectrum by retention time
        rt = (
            self.retention_time_list[scan_idx]
            if self.retention_time_list
            and scan_idx < len(self.retention_time_list)
            else None
        )
        if rt is not None:
            rts = [s["retention_time"] for s in self.ms1_reference_spectra]
            # bisect to nearest RT
            import bisect

            idx = bisect.bisect_left(rts, rt)
            if idx == 0:
                ms1 = self.ms1_reference_spectra[0]
            elif idx >= len(rts):
                ms1 = self.ms1_reference_spectra[-1]
            else:
                # pick closer of the two neighbours
                before = self.ms1_reference_spectra[idx - 1]
                after = self.ms1_reference_spectra[idx]
                ms1 = (
                    before
                    if abs(before["retention_time"] - rt)
                    <= abs(after["retention_time"] - rt)
                    else after
                )
        else:
            ms1 = self.ms1_reference_spectra[0]

        mz_arr = ms1["mz"]
        int_arr = ms1["intensity"]
        if mz_arr is None or len(mz_arr) == 0:
            return []

        mask = (mz_arr >= win_lo) & (mz_arr <= win_hi)
        if not np.any(mask):
            return []

        peaks = []
        for mz_val, intensity_val in zip(mz_arr[mask], int_arr[mask]):
            if mz_val <= 0 or intensity_val <= 0:
                continue
            amplitude = float(intensity_val) / self.max_intensity_overall
            peaks.append((float(mz_val), amplitude))

        return peaks

    def apply_effect(self, effect_name: str, effect_params: dict = None):
        if (
            self.current_audio_data is None
            or self.current_audio_data.size == 0
        ):
            print("No audio data to apply effects to. Please sonify first.")
            return

        if effect_params is None:
            effect_params = {}

        # Construct function name string and get the function object
        effect_function_name = f"apply_{effect_name}"

        # Check if the function exists
        if not hasattr(audio_effects, effect_function_name):
            print(
                f"Error: Effect function '{effect_function_name}' not found "
                f"in effects module."
            )
            effect_names = [
                name
                for name in dir(audio_effects)
                if name.startswith("apply_")
            ]
            effects_list = ", ".join(effect_names)
            print(f"Available effects: {effects_list}")
            return

        effect_function = getattr(audio_effects, effect_function_name)

        print(f"Applying effect: {effect_name} with params: {effect_params}")

        # Apply the effect (per channel for stereo audio; the effects
        # themselves are mono).
        audio = self.current_audio_data
        if audio.ndim == 1:
            self.current_audio_data = effect_function(
                audio, self.sample_rate, **effect_params
            )
        else:
            channels = [
                effect_function(ch, self.sample_rate, **effect_params)
                for ch in audio
            ]
            n = max(len(ch) for ch in channels)
            self.current_audio_data = np.stack(
                [
                    np.pad(np.asarray(ch, dtype=np.float32), (0, n - len(ch)))
                    for ch in channels
                ]
            )
        print(f"Effect '{effect_name}' applied successfully.")

    def save_audio(self, output_filepath: str, normalize: bool = True):
        if (
            self.current_audio_data is None
            or self.current_audio_data.size == 0
        ):
            print("No audio data to save. Please sonify first.")
            return

        audio_to_save = self.current_audio_data
        if normalize:
            audio_to_save = io.normalize_audio_to_16bit(audio_to_save)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        io.save_wav(output_filepath, audio_to_save, self.sample_rate)

    def get_current_audio(self, copy=True):
        """
        Returns the current audio data.

        Args:
            copy (bool, optional): If True (default), returns a copy of the
                                   audio data
                                   to prevent external modifications. If
                                   False, returns
                                   a direct reference.
        Returns:
            np.ndarray or None: The current audio data, or None if not
            generated.
        """
        if self.current_audio_data is None:
            return None
        return (
            np.copy(self.current_audio_data)
            if copy
            else self.current_audio_data
        )

    def _filter_top_peaks(self, spectra_dfs, max_peaks_per_scan):
        """Return a copy of spectra_dfs with each scan trimmed to the
        top-N most intense peaks. Useful for MS2 data where hundreds of
        fragment ions produce white-noise-like audio."""
        filtered = []
        for df in spectra_dfs:
            if df.empty or len(df) <= max_peaks_per_scan:
                filtered.append(df)
            else:
                # Same rows and order as df.nlargest(n, keep="first"), but
                # much faster for hundreds of thousands of small spectra.
                order = np.argsort(
                    -df["intensities"].to_numpy(), kind="stable"
                )[:max_peaks_per_scan]
                filtered.append(df.iloc[order])
        return filtered

    def sonify(
        self,
        method: str = "gradient",
        frequency_mapping: str = "inverse_log",
        freq_range: tuple = (200.0, 4000.0),
        scale: str = None,
        root_note: str = "C",
        tuning_freq: float = 440.0,
        edo_divisions: int = None,
        use_just_intonation: bool = False,
        use_log_distance: bool = True,
        adsr_settings: dict = None,
        rhythm=None,
        ms2_mode: str = None,
        max_peaks_per_scan: int = None,
        mobility=None,
    ):
        """
        Turn the loaded spectra into audio (stored in ``current_audio_data``).

        Every option combines with every other: pitch quantization, rhythm
        and MS2 tones work with both synthesis methods.

        Args:
            method: Synthesis method.
                - 'gradient': continuous tones whose loudness follows each
                  m/z's intensity from scan to scan.
                - 'adsr': each scan becomes a note with an ADSR envelope.
            frequency_mapping: How m/z maps to pitch: 'inverse_log',
                'power_law', 'musical_octaves', 'chromatic' or 'linear'.
            freq_range: (min_freq, max_freq) in Hz for the mapping.
            scale: Snap pitches to this scale (e.g. 'major',
                'pentatonic_minor', '19_edo_diatonic', 'just_major').
                None (default) keeps continuous pitch.
            root_note: Root of the scale ('C', 'C#', ... 'B').
            tuning_freq: Frequency of A4 in Hz.
            edo_divisions: Divisions of the octave for EDO scales
                (default 12).
            use_just_intonation: Use just-intonation ratios (for 'just_*'
                scales).
            use_log_distance: Pick the nearest scale note in log-frequency
                (perceptual) rather than linear Hz.
            adsr_settings: ADSR options for method='adsr': 'randomize'
                (default True), 'attack_time_pc', 'decay_time_pc',
                'sustain_level_pc', 'release_time_pc'.
            rhythm: Lay the sound onto a musical grid: a
                :class:`ms_music.rhythm.RhythmConfig` or a dict of its
                fields, e.g. ``{"meter": "3/4", "tempo": 96}``. The length
                snaps to whole bars. None (default) gives every scan an
                equal slice of time.
            ms2_mode: How to handle MS2 precursor information.
                - None (default): no precursor tone added.
                - 'dda': sustain a tone at the single selected precursor m/z
                  for each MS2 scan (Data-Dependent Acquisition).
                - 'dia': play all MS1 peaks within the isolation window of
                  each MS2 scan simultaneously as precursor tones
                  (Data-Independent Acquisition). Requires calling
                  load_ms1_reference() first.
            max_peaks_per_scan: If set, only the top-N most intense peaks
                per scan are used. Strongly recommended for MS2 data to
                avoid white-noise-like output from hundreds of simultaneous
                fragment ion frequencies.
            mobility: Use ion mobility in the sound (needs data loaded with
                ion mobility): a :class:`ms_music.mobility.MobilityConfig`
                or a dict of its fields, e.g.
                ``{"brightness": 0.6, "pan": 0.8, "effects": [...]}``.
                Brightness adds overtones to extended ions, pan places
                compact ions left and extended ions right (stereo output),
                and mapped effects sweep an effect parameter across
                mobility. None (default) ignores mobility.

        Spectra denser than one per 5 ms of audio (e.g. ~250k PASEF MS2
        spectra) are merged into consecutive time bins first; MS2 precursor
        tones then include every precursor in a bin.
        """
        if method not in ("gradient", "adsr"):
            raise ValueError(
                f"Unknown method: {method}. Choose 'gradient' or 'adsr'."
            )
        if frequency_mapping not in self.FREQUENCY_MAPPINGS:
            raise ValueError(
                f"Unknown mapping type: {frequency_mapping}. Choose one of "
                f"{', '.join(self.FREQUENCY_MAPPINGS)}."
            )
        if ms2_mode not in (None, "dda", "dia"):
            raise ValueError(
                f"Unknown ms2_mode: {ms2_mode}. Choose None, 'dda' or 'dia'."
            )
        mobility = MobilityConfig.coerce(mobility)
        rhythm = RhythmConfig.coerce(rhythm)

        if (
            self.processed_spectra_dfs is None
            or not self.processed_spectra_dfs
        ):
            print(
                "Data not loaded or preprocessed. Please call "
                "load_and_preprocess_data() first."
            )
            return

        freq_range = (float(freq_range[0]), float(freq_range[1]))
        if scale is not None:
            self.note_quantizer = MusicalNoteQuantizer(
                scale=scale,
                root_note=root_note,
                tuning_freq=tuning_freq,
                freq_range=freq_range,
                edo_divisions=edo_divisions or 12,
                use_just_intonation=use_just_intonation,
            )
        else:
            self.note_quantizer = None
        freq_fn = self._frequency_function(
            frequency_mapping,
            freq_range,
            self.note_quantizer,
            use_log_distance,
        )
        self.frequency_fn = freq_fn  # m/z -> Hz used (for synced visuals)

        print(f"Starting sonification using '{method}' method...")
        print(
            f"Frequency mapping: {frequency_mapping}, "
            f"Range: {freq_range[0]}-{freq_range[1]} Hz"
            + (f", scale: {scale} in {root_note}" if scale else "")
        )

        source = (
            self._filter_top_peaks(
                self.processed_spectra_dfs, max_peaks_per_scan
            )
            if max_peaks_per_scan is not None
            else self.processed_spectra_dfs
        )
        timbre = None
        self.mobility_config = None
        if mobility is not None and mobility.active:
            timbre = self._brightness_timbre(source, mobility)
            self.mobility_config = mobility
            self.mobility_norm = (timbre["lo"], timbre["hi"])

        # Time layout: equal slices per scan, or steps of a musical grid.
        # Each segment remembers which source scans it holds (for MS2
        # precursor tones).
        self.rhythm_grid = None
        spectra = source
        members = [[i] for i in range(len(source))]
        duration = self.total_duration_seconds
        segment_lengths = None
        if rhythm is not None:
            grid = build_grid(self.total_duration_seconds, rhythm)
            spectra, _, members = resample_spectra(
                source, grid.n_steps, rhythm.aggregate, return_members=True
            )
            duration = grid.total_seconds
            segment_lengths = grid.segment_lengths(self.sample_rate)
            self.rhythm_grid = grid
            print(f"Rhythm: {grid.describe()} ({grid.n_steps} steps)")
        elif duration / len(source) < self.MIN_SLICE_SECONDS:
            n_bins = max(1, int(duration / self.MIN_SLICE_SECONDS))
            spectra, _, members = resample_spectra(
                source, n_bins, "max", return_members=True
            )
            print(
                f"Merged {len(source):,} spectra into {n_bins:,} time bins "
                f"of ≥{self.MIN_SLICE_SECONDS * 1000:.0f} ms"
            )
        precursors = self._segment_precursors(
            members, source, freq_fn, ms2_mode
        )
        # Remember what sounds when (for synced visuals such as the
        # spatial-stage video).
        if self.rhythm_grid is not None:
            self.segment_starts = np.asarray(self.rhythm_grid.onsets, float)
        else:
            per = round(self.sample_rate * duration / len(spectra))
            self.segment_starts = np.arange(len(spectra)) * (
                per / self.sample_rate
            )
        self.segment_spectra = spectra
        # Source scans (indices into processed_spectra_dfs) per segment.
        self.segment_members = [np.asarray(m, dtype=int) for m in members]

        if self.rhythm_grid is not None:
            audio = self._synthesize_metered(
                spectra,
                freq_fn,
                self.rhythm_grid,
                method,
                precursors=precursors,
                adsr_settings=adsr_settings,
                timbre=timbre,
            )
        else:
            synthesize = (
                self._synthesize_gradient
                if method == "gradient"
                else self._synthesize_adsr
            )
            kwargs = {}
            if method == "adsr":
                kwargs["adsr_settings"] = adsr_settings
            audio = synthesize(
                spectra,
                freq_fn,
                duration,
                segment_lengths=segment_lengths,
                precursors=precursors,
                timbre=timbre,
                **kwargs,
            )

        self.current_audio_data = audio
        if (
            self.current_audio_data is None
            or self.current_audio_data.size == 0
        ):
            print("Sonification failed to produce audio data.")
            self.current_audio_data = None

    def _frequency_function(
        self,
        frequency_mapping,
        freq_range,
        quantizer=None,
        use_log_distance=True,
    ):
        """Return a memoized ``m/z -> Hz`` function for one sonify() call:
        the chosen mapping, optionally snapped to the quantizer's scale."""
        cache = {}

        def freq_fn(mz_value):
            if mz_value in cache:
                return cache[mz_value]
            freq = self._mz_to_frequency(
                mz_value, frequency_mapping, freq_range
            )
            if quantizer is not None and freq > 0:
                note = (
                    quantizer.quantize_frequency_log(freq)
                    if use_log_distance
                    else quantizer.quantize_frequency(freq)
                )
                freq = note["frequency"]
            cache[mz_value] = freq
            return freq

        return freq_fn

    def _segment_layout(self, num_segments, duration, segment_lengths):
        """Sample counts per segment and the total buffer length.

        Without an explicit layout every scan gets
        ``round(sr * duration / num_segments)`` samples, as before."""
        if segment_lengths is None:
            per = round(self.sample_rate * duration / num_segments)
            segment_lengths = np.full(num_segments, per, dtype=np.int64)
        segment_lengths = np.asarray(segment_lengths, dtype=np.int64)
        starts = np.concatenate(([0], np.cumsum(segment_lengths)[:-1]))
        return segment_lengths, starts, int(segment_lengths.sum())

    def _precursor_pitches(self, scan_idx, spectra_df, freq_fn, ms2_mode):
        """``[(freq, amplitude)]`` precursor tones for one scan/segment.

        ``scan_idx`` indexes the per-scan MS2 metadata."""
        if ms2_mode == "dda":
            if spectra_df.empty:
                return []
            amplitude = float(
                spectra_df["intensities"].max() / self.max_intensity_overall
            )
            if amplitude <= 0:
                return []
            pmz = (
                self.precursor_mz_list[scan_idx]
                if self.precursor_mz_list
                and scan_idx < len(self.precursor_mz_list)
                else None
            )
            if pmz is None or pmz <= 0:
                return []
            freq = freq_fn(pmz)
            return [(freq, amplitude)] if freq > 0 else []
        if ms2_mode == "dia":
            pitches = []
            for pmz, amp in self._dia_precursor_peaks(scan_idx):
                freq = freq_fn(pmz)
                if freq > 0:
                    pitches.append((freq, amp))
            return pitches
        return []

    def _segment_precursors(self, members, source, freq_fn, ms2_mode):
        """Per segment, ``[(freq, amplitude)]`` for every precursor of the
        segment's member scans (None when MS2 tones are off). A precursor
        repeated within a segment keeps its loudest amplitude."""
        if ms2_mode not in ("dda", "dia"):
            return None
        out = []
        for scan_ids in members:
            loudest = {}
            for i in scan_ids:
                i = int(i)
                for freq, amp in self._precursor_pitches(
                    i, source[i], freq_fn, ms2_mode
                ):
                    if amp > loudest.get(freq, 0.0):
                        loudest[freq] = amp
            out.append(list(loudest.items()))
        return out

    @staticmethod
    def _precursor_tones(pitches, t):
        """Sum of precursor sines on time vector ``t``."""
        tones = np.zeros(t.size, dtype=np.float32)
        for freq, amp in pitches:
            tones += np.sin(freq * 2 * math.pi * t) * amp
        return tones

    # Spectra shorter than this in the audio are merged into time bins.
    MIN_SLICE_SECONDS = 0.005
    # Overtones 2..N added by the 'brightness' mobility mapping.
    BRIGHTNESS_PARTIALS = 4

    def _brightness_timbre(self, spectra, config):
        """Mobility normalization for a :class:`MobilityConfig`: its
        ``range``, or the 2nd-98th percentile of the data's per-bin
        mobility, maps to 0-1."""
        values = [
            df["mobility"].to_numpy()
            for df in spectra
            if "mobility" in df.columns and not df.empty
        ]
        values = np.concatenate(values) if values else np.array([])
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                "mobility options need ion mobility data, but the loaded "
                "spectra have none."
            )
        if config.range is not None:
            lo, hi = config.range
        else:
            lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            hi = lo + 1e-9
        print(f"Ion mobility {lo:.3f}–{hi:.3f} mapped to sound")
        return {
            "depth": config.brightness,
            "lo": float(lo),
            "hi": float(hi),
            "config": config,
            "routed": config.stereo or bool(config.effects),
        }

    @staticmethod
    def _mobility_position(timbre, mobility):
        """Map mobility values to 0-1 positions for panning and mapped
        effects (NaN -> 0.5, i.e. centre)."""
        b = (np.asarray(mobility, dtype=float) - timbre["lo"]) / (
            timbre["hi"] - timbre["lo"]
        )
        return np.nan_to_num(np.clip(b, 0.0, 1.0), nan=0.5)

    @staticmethod
    def _mobility_brightness(timbre, mobility):
        """Map mobility values to 0-1 brightness (NaN -> 0)."""
        b = (np.asarray(mobility, dtype=float) - timbre["lo"]) / (
            timbre["hi"] - timbre["lo"]
        )
        return np.nan_to_num(np.clip(b, 0.0, 1.0))

    def _add_partials(self, out, sin_theta, theta, frequency, gain):
        """Add overtones 2..N of a tone to ``out`` with amplitude
        ``gain / k``, skipping those near or above Nyquist. Uses the
        Chebyshev recurrence sin(kθ) = 2cosθ·sin((k-1)θ) - sin((k-2)θ)."""
        two_cos = 2.0 * np.cos(theta)
        prev2, prev = np.zeros_like(sin_theta), sin_theta
        for k in range(2, self.BRIGHTNESS_PARTIALS + 1):
            cur = two_cos * prev - prev2
            if k * frequency >= 0.45 * self.sample_rate:
                break
            out += (cur * (gain / k)).astype(out.dtype, copy=False)
            prev2, prev = prev, cur

    def _synthesize_metered(
        self,
        step_dfs,
        freq_fn,
        grid,
        method,
        precursors=None,
        adsr_settings=None,
        timbre=None,
    ):
        """Render grid steps as notes that follow each pitch's signal.

        Every distinct pitch gets a per-step level track (summed normalized
        intensities, plus MS2 precursor tones). Each run of steps where the
        pitch is present becomes one note, so a long chromatographic peak is
        one long note rather than a string of grid-step notes. Gradient
        notes glide with the intensity inside the note; ADSR notes get one
        envelope over their whole length.
        """
        n_steps = len(step_dfs)
        total_samples = int(grid.segment_lengths(self.sample_rate).sum())
        time_vector = np.linspace(
            0,
            total_samples / self.sample_rate,
            total_samples,
            endpoint=False,
            dtype=np.float32,
        )
        song = np.zeros(total_samples, dtype=np.float32)
        routed = timbre is not None and timbre["routed"]
        bus = ToneBus(total_samples, timbre["config"]) if routed else None

        tracks = {}  # freq -> per-step level
        bright_tracks = {}  # freq -> per-step level-weighted brightness
        pos_tracks = {}  # freq -> per-step level-weighted mobility position

        def add(freq, step, level, bright=0.0, pos=0.5):
            track = tracks.get(freq)
            if track is None:
                track = tracks[freq] = np.zeros(n_steps, dtype=np.float64)
                bright_tracks[freq] = np.zeros(n_steps, dtype=np.float64)
                pos_tracks[freq] = np.zeros(n_steps, dtype=np.float64)
            track[step] += level
            bright_tracks[freq][step] += level * bright
            pos_tracks[freq][step] += level * pos

        for step, df in enumerate(step_dfs):
            if not df.empty:
                mzs = df.index.to_numpy(dtype=float)
                levels = (
                    df["intensities"].to_numpy(dtype=float)
                    / self.max_intensity_overall
                )
                has_mob = timbre is not None and "mobility" in df.columns
                brights = (
                    self._mobility_brightness(
                        timbre, df["mobility"].to_numpy()
                    )
                    if has_mob
                    else np.zeros(len(mzs))
                )
                positions = (
                    self._mobility_position(timbre, df["mobility"].to_numpy())
                    if has_mob
                    else np.full(len(mzs), 0.5)
                )
                for mz, level, b, pos in zip(mzs, levels, brights, positions):
                    if mz > 0 and level > 0:
                        freq = freq_fn(mz)
                        if freq > 0:
                            add(freq, step, level, b, pos)
            if precursors is not None:
                for freq, amp in precursors[step]:
                    add(freq, step, amp)

        settings = adsr_settings or {}
        randomize = settings.get("randomize", True)
        n_notes = 0
        for frequency, levels in tqdm(
            tracks.items(), desc="Synthesizing notes"
        ):
            for start, end in find_notes(levels, grid.config.note_threshold):
                t0, t1 = note_span(grid, start, end)
                i0 = int(round(t0 * self.sample_rate))
                i1 = min(int(round(t1 * self.sample_rate)), total_samples)
                if i1 <= i0:
                    continue
                t = time_vector[i0:i1]
                if method == "gradient":
                    gain = note_gain(grid, levels, start, end, t)
                else:
                    if randomize:
                        a, d = random.uniform(0.01, 0.10), random.uniform(
                            0.02, 0.15
                        )
                        sus = random.uniform(0.4, 0.8)
                        r = random.uniform(0.05, max(0.06, (1 - a - d) * 0.8))
                    else:
                        a = settings.get("attack_time_pc", 0.05)
                        d = settings.get("decay_time_pc", 0.1)
                        sus = settings.get("sustain_level_pc", 0.7)
                        r = settings.get("release_time_pc", 0.15)
                    peak = float(levels[start:end].max()) * float(
                        grid.accents[start]
                    )
                    gain = peak * self._adsr_envelope(i1 - i0, a, d, sus, r)
                theta = frequency * 2 * math.pi * t
                tone = np.sin(theta)
                voice = tone * gain
                lv = levels[start:end].sum()
                if timbre is not None and timbre["depth"] > 0:
                    # One timbre per note: its level-weighted brightness.
                    b = (
                        bright_tracks[frequency][start:end].sum() / lv
                        if lv
                        else 0
                    )
                    if b > 0:
                        part = np.zeros_like(tone)
                        self._add_partials(
                            part,
                            tone,
                            theta,
                            frequency,
                            gain * timbre["depth"] * b,
                        )
                        voice = voice + part
                if routed:
                    # One place per note: its level-weighted position.
                    pos = (
                        pos_tracks[frequency][start:end].sum() / lv
                        if lv
                        else 0.5
                    )
                    bus.add(voice, i0, pos)
                else:
                    song[i0:i1] += voice
                n_notes += 1

        print(f"Rendered {n_notes} notes from {len(tracks)} pitches")
        return bus.render(self.sample_rate) if routed else song

    FREQUENCY_MAPPINGS = (
        "inverse_log",
        "power_law",
        "musical_octaves",
        "chromatic",
        "linear",
    )

    def _mz_to_frequency(self, mz_value, mapping_type, freq_range):
        """Helper method for frequency mapping."""
        freq_min, freq_max = freq_range

        if mapping_type == "inverse_log":
            if mz_value <= 0 or self.min_mz_overall <= 0:
                return freq_min
            mz_normalized = (mz_value - self.min_mz_overall) / (
                self.max_mz_overall - self.min_mz_overall
            )
            mz_normalized = np.clip(mz_normalized, 0, 1)
            log_ratio = math.log(freq_max / freq_min)
            return freq_max * math.exp(-mz_normalized * log_ratio)

        elif mapping_type == "power_law":
            if mz_value <= 0:
                return freq_min
            mz_normalized = (mz_value - self.min_mz_overall) / (
                self.max_mz_overall - self.min_mz_overall
            )
            mz_normalized = np.clip(mz_normalized, 0, 1)
            freq_normalized = (1 - mz_normalized) ** 1.5  # Power law exponent
            log_freq_min = math.log(freq_min)
            log_freq_max = math.log(freq_max)
            log_frequency = log_freq_min + freq_normalized * (
                log_freq_max - log_freq_min
            )
            return math.exp(log_frequency)

        elif mapping_type == "musical_octaves":
            if mz_value <= 0:
                return freq_min
            mz_normalized = (mz_value - self.min_mz_overall) / (
                self.max_mz_overall - self.min_mz_overall
            )
            mz_normalized = np.clip(mz_normalized, 0, 1)
            mz_inverted = 1 - mz_normalized
            num_octaves = math.log2(
                freq_max / freq_min
            )  # Calculate octaves from freq range
            octave_position = mz_inverted * num_octaves
            return freq_min * (2**octave_position)

        elif mapping_type == "chromatic":
            if mz_value <= 0:
                return freq_min
            mz_normalized = (mz_value - self.min_mz_overall) / (
                self.max_mz_overall - self.min_mz_overall
            )
            mz_normalized = np.clip(mz_normalized, 0, 1)
            mz_inverted = 1 - mz_normalized
            num_semitones = 12 * math.log2(
                freq_max / freq_min
            )  # Semitones in the range
            semitone_position = mz_inverted * num_semitones
            return freq_min * (2 ** (semitone_position / 12))

        elif mapping_type == "linear":
            if mz_value <= 0:
                return freq_min
            mz_normalized = (mz_value - self.min_mz_overall) / (
                self.max_mz_overall - self.min_mz_overall
            )
            mz_normalized = np.clip(mz_normalized, 0, 1)
            return freq_min + mz_normalized * (freq_max - freq_min)

        else:
            raise ValueError(f"Unknown mapping type: {mapping_type}")

    def _synthesize_gradient(
        self,
        spectra_dfs,
        freq_fn,
        duration,
        segment_lengths=None,
        precursors=None,
        timbre=None,
    ):
        """Continuous tones: one sine per distinct pitch, its loudness
        stepping with the summed normalized intensity of the m/z values
        mapped to that pitch in each segment."""
        num_segments = len(spectra_dfs)
        lengths, starts, total_samples = self._segment_layout(
            num_segments, duration, segment_lengths
        )

        time_vector = np.linspace(
            0,
            total_samples / self.sample_rate,
            total_samples,
            endpoint=False,
            dtype=np.float32,
        )
        song = np.zeros(total_samples, dtype=np.float32)
        routed = timbre is not None and timbre["routed"]
        bus = ToneBus(total_samples, timbre["config"]) if routed else None

        # Long table of (segment, m/z, intensity), then per-pitch rows of
        # summed intensities across segments.
        frames = []
        for seg, df in enumerate(spectra_dfs):
            if df.empty:
                continue
            frame = {
                "segment": seg,
                "mz": df.index.to_numpy(dtype=float),
                "intensity": df["intensities"].to_numpy(dtype=float),
            }
            if timbre is not None:
                frame["mobility"] = (
                    df["mobility"].to_numpy(dtype=float)
                    if "mobility" in df.columns
                    else np.nan
                )
            frames.append(pd.DataFrame(frame))
        if frames:
            peaks = pd.concat(frames, ignore_index=True)
            peaks = peaks[peaks["mz"] > 0]
            unique_mz = peaks["mz"].unique()
            freq_of = {mz: freq_fn(mz) for mz in unique_mz}
            peaks["freq"] = peaks["mz"].map(freq_of)
            peaks = peaks[peaks["freq"] > 0]
            peaks["intensity"] /= self.max_intensity_overall
            per_pitch = peaks.groupby(["freq", "segment"])["intensity"].sum()
            if timbre is not None:
                # Intensity-weighted brightness per pitch and segment.
                peaks["bright"] = (
                    self._mobility_brightness(timbre, peaks["mobility"])
                    * peaks["intensity"]
                )
                bright_sum = peaks.groupby(["freq", "segment"])["bright"].sum()
                if routed:
                    peaks["pos"] = (
                        self._mobility_position(timbre, peaks["mobility"])
                        * peaks["intensity"]
                    )
                    pos_sum = peaks.groupby(["freq", "segment"])["pos"].sum()
            # Synthesize with the exact frequency objects the mapping
            # returned (Python float vs numpy float64 decides float32 vs
            # float64 sine math under NumPy 2 promotion rules).
            pitch_obj = {f: f for f in freq_of.values()}
            print(
                f"Mapped {len(unique_mz)} m/z values to "
                f"{per_pitch.index.get_level_values(0).nunique()} "
                f"frequencies"
            )

            for frequency, rows in tqdm(
                per_pitch.groupby(level=0), desc="Synthesizing pitches"
            ):
                intensities = np.zeros(num_segments, dtype=np.float32)
                intensities[rows.index.get_level_values(1)] = rows.to_numpy()
                gain = np.repeat(intensities, lengths)
                pitch = frequency
                frequency = pitch_obj[frequency]
                theta = frequency * 2 * math.pi * time_vector
                tone = np.sin(theta)
                # Mobility routing mixes each pitch into the bus as one
                # voice; otherwise tones go straight into the mono song.
                voice = tone * gain if routed else None
                target = voice if routed else song
                if not routed:
                    song += tone * gain
                if timbre is not None and timbre["depth"] > 0:
                    bright = np.zeros(num_segments, dtype=np.float32)
                    b_rows = bright_sum.loc[pitch]
                    bright[b_rows.index] = b_rows.to_numpy()
                    self._add_partials(
                        target,
                        tone,
                        theta,
                        frequency,
                        np.repeat(bright, lengths) * timbre["depth"],
                    )
                if routed:
                    pos = np.full(num_segments, 0.5, dtype=np.float32)
                    p_rows = pos_sum.loc[pitch]
                    seg_idx = p_rows.index.to_numpy()
                    with np.errstate(invalid="ignore", divide="ignore"):
                        pos[seg_idx] = np.where(
                            intensities[seg_idx] > 0,
                            p_rows.to_numpy() / intensities[seg_idx],
                            0.5,
                        )
                    bus.add(voice, 0, np.repeat(pos, lengths))

        if precursors is not None:
            for seg in range(num_segments):
                start, end = starts[seg], starts[seg] + lengths[seg]
                tones = self._precursor_tones(
                    precursors[seg], time_vector[start:end]
                )
                if routed:
                    bus.add(tones, start)
                else:
                    song[start:end] += tones

        return bus.render(self.sample_rate) if routed else song

    def _adsr_envelope(
        self,
        length_samples: int,
        attack_time_pc: float,
        decay_time_pc: float,
        sustain_level_pc: float,
        release_time_pc: float,
    ):
        """Generates an ADSR envelope. Times are percentages
        of total length."""
        # Ensure sum of A, D, R percentages is <= 1.0 for sustain
        # to be non-negative.
        # Prioritize A, D, R, and calculate Sustain time.
        attack_samples = max(0, int(attack_time_pc * length_samples))
        decay_samples = max(0, int(decay_time_pc * length_samples))
        release_samples = max(0, int(release_time_pc * length_samples))

        # Adjust if sum of A,D,R > length_samples
        # (should not happen if pc <= 1)
        if attack_samples + decay_samples + release_samples > length_samples:
            # Scale them down proportionally if sum is too large
            total_adr_samples = (
                attack_samples + decay_samples + release_samples
            )
            scale_factor = length_samples / total_adr_samples
            attack_samples = int(attack_samples * scale_factor)
            decay_samples = int(decay_samples * scale_factor)
            release_samples = int(release_samples * scale_factor)
            # Recalculate to ensure integer sum matches
            release_samples = length_samples - (attack_samples + decay_samples)

        sustain_samples = max(
            0,
            length_samples
            - (attack_samples + decay_samples + release_samples),
        )

        envelope = np.zeros(length_samples, dtype=np.float32)
        current_pos = 0

        # Attack
        if attack_samples > 0:
            envelope[current_pos: current_pos + attack_samples] = np.linspace(
                0, 1, attack_samples, endpoint=False
            )
        current_pos += attack_samples

        # Decay
        if decay_samples > 0 and current_pos < length_samples:
            # Start decay from 1 or sustain_level if no attack
            env_start_val = 1.0 if attack_samples > 0 else sustain_level_pc
            decay_end = min(current_pos + decay_samples, length_samples)
            envelope[current_pos:decay_end] = np.linspace(
                env_start_val,
                sustain_level_pc,
                decay_end - current_pos,
                endpoint=False,
            )
        current_pos += decay_samples

        # Sustain
        if sustain_samples > 0 and current_pos < length_samples:
            sustain_end = min(current_pos + sustain_samples, length_samples)
            envelope[current_pos:sustain_end] = sustain_level_pc
        current_pos += sustain_samples

        # Release
        if release_samples > 0 and current_pos < length_samples:
            release_end = min(current_pos + release_samples, length_samples)
            # Ensure release actually fits
            actual_release_samples = release_end - current_pos
            if actual_release_samples > 0:
                envelope[current_pos:release_end] = np.linspace(
                    sustain_level_pc, 0, actual_release_samples, endpoint=False
                )

        # Ensure last point is 0 if there's a release phase that reaches
        # the end
        if release_samples > 0 and (
            current_pos + release_samples >= length_samples
        ):
            if length_samples > 0:
                envelope[-1] = 0.0

        return envelope

    def _synthesize_adsr(
        self,
        spectra_dfs,
        freq_fn,
        duration,
        segment_lengths=None,
        precursors=None,
        adsr_settings=None,
        timbre=None,
    ):
        """One note per segment: all its peaks summed, normalized, and
        shaped by an ADSR envelope (fixed or randomized per note)."""
        target_samples = (
            int(duration * self.sample_rate)
            if segment_lengths is None
            else int(np.sum(segment_lengths))
        )
        if not spectra_dfs:
            print(
                "Warning (ADSR): No processed spectra. "
                "Returning silent audio."
            )
            return np.zeros(target_samples, dtype=np.float32)
        if self.max_intensity_overall <= 0:
            print(
                "Warning (ADSR): Max overall intensity is non-positive. "
                "Sonification will be silent."
            )
            return np.zeros(target_samples, dtype=np.float32)

        num_segments = len(spectra_dfs)
        lengths, starts, total = self._segment_layout(
            num_segments, duration, segment_lengths
        )
        routed = timbre is not None and timbre["routed"]
        bus = (
            ToneBus(max(total, target_samples), timbre["config"])
            if routed
            else None
        )

        if adsr_settings is None:
            adsr_settings = {}
        use_random_adsr = adsr_settings.get("randomize", True)
        fixed_attack_pc = adsr_settings.get("attack_time_pc", 0.05)
        fixed_decay_pc = adsr_settings.get("decay_time_pc", 0.1)
        fixed_sustain_level = adsr_settings.get("sustain_level_pc", 0.7)
        fixed_release_pc = adsr_settings.get("release_time_pc", 0.15)

        song_parts = []
        print("Generating audio with ADSR method...")
        for seg, scan_df in enumerate(
            tqdm(spectra_dfs, desc="Processing Scans (ADSR)", unit="scan")
        ):
            n = int(lengths[seg])
            # Each note starts at phase zero on its own local time axis.
            t = np.linspace(
                0, n / self.sample_rate, n, endpoint=False, dtype=np.float32
            )
            segment = np.zeros(n, dtype=np.float32)
            seg_bus = bus.local(n) if routed else None

            if not scan_df.empty:
                mzs = scan_df.index.to_numpy()
                intensities = scan_df["intensities"].to_numpy()
                has_mob = timbre is not None and "mobility" in scan_df.columns
                brights = (
                    self._mobility_brightness(
                        timbre, scan_df["mobility"].to_numpy()
                    )
                    if has_mob
                    else np.zeros(len(mzs))
                )
                positions = (
                    self._mobility_position(
                        timbre, scan_df["mobility"].to_numpy()
                    )
                    if routed and has_mob
                    else np.full(len(mzs), 0.5)
                )
                for mz_value, intensity, b, pos in zip(
                    mzs, intensities, brights, positions
                ):
                    if mz_value <= 0 or intensity <= 0:
                        continue
                    frequency = freq_fn(mz_value)
                    if frequency <= 0:
                        continue
                    level = intensity / self.max_intensity_overall
                    theta = frequency * 2 * math.pi * t
                    tone = np.sin(theta)
                    voice = tone * level if routed else None
                    target = voice if routed else segment
                    if not routed:
                        segment += tone * level
                    if timbre is not None and timbre["depth"] > 0 and b > 0:
                        self._add_partials(
                            target,
                            tone,
                            theta,
                            frequency,
                            level * timbre["depth"] * b,
                        )
                    if routed:
                        seg_bus.add(voice, 0, pos)

            if precursors is not None:
                if routed:
                    seg_bus.add(self._precursor_tones(precursors[seg], t), 0)
                else:
                    segment += self._precursor_tones(precursors[seg], t)

            if routed:
                max_abs_segment = seg_bus.mixdown_peak()
                if max_abs_segment > 0:
                    seg_bus.buf /= max_abs_segment
            else:
                max_abs_segment = np.max(np.abs(segment)) if n else 0
                if max_abs_segment > 0:
                    segment /= max_abs_segment

            if use_random_adsr:
                attack_t_pc = random.uniform(0.01, 0.10)
                decay_t_pc = random.uniform(0.02, 0.15)
                sustain_l_pc = random.uniform(0.4, 0.8)
                max_r_pc = 1.0 - attack_t_pc - decay_t_pc
                release_t_pc = random.uniform(0.05, max(0.06, max_r_pc * 0.8))
                release_t_pc = max(0.01, min(release_t_pc, max_r_pc))
            else:
                attack_t_pc = fixed_attack_pc
                decay_t_pc = fixed_decay_pc
                sustain_l_pc = fixed_sustain_level
                release_t_pc = fixed_release_pc

            envelope = self._adsr_envelope(
                n, attack_t_pc, decay_t_pc, sustain_l_pc, release_t_pc
            )
            if routed:
                bus.add_block(seg_bus, int(starts[seg]), envelope)
            else:
                song_parts.append(segment * envelope)

        if routed:
            return bus.render(self.sample_rate)[..., :target_samples]
        song = np.concatenate(song_parts)
        if len(song) > target_samples:
            song = song[:target_samples]
        elif len(song) < target_samples:
            song = np.pad(song, (0, target_samples - len(song)), "constant")
        return song

    def load_fid_data(
        self,
        fid_filepath: str,
        original_sample_rate: float = 10e6,  # 10 MHz default
        conversion_factor: float = 2**12,  # 4096 default
    ):
        """
        Load FID data directly as audio, making it compatible with all
        sonifier methods.

        Args:
            fid_filepath: Path to FID binary file
            original_sample_rate: Original FID acquisition rate in Hz
                (default: 10 MHz)
            conversion_factor: Factor to downsample the original rate
                (default: 2^12 = 4096)

        Returns:
            np.ndarray: Processed audio data
        """
        with open(fid_filepath, "rb") as f:
            data = np.fromfile(f, dtype="<i4")

        fid_data = data.astype(np.float32)

        # Calculate effective sample rate after conversion
        effective_sample_rate = original_sample_rate / conversion_factor

        # Resample from effective rate to target rate if needed
        if effective_sample_rate != self.sample_rate:
            new_length = int(
                len(fid_data) * self.sample_rate / effective_sample_rate
            )
            resampled_data = signal.resample(fid_data, new_length)
        else:
            resampled_data = fid_data.copy()

        # Time-stretch to exact target duration while preserving pitch
        current_duration = len(resampled_data) / self.sample_rate
        time_stretch_rate = current_duration / self.total_duration_seconds

        # Use librosa's pitch-preserving time stretch
        self.current_audio_data = librosa.effects.time_stretch(
            resampled_data, rate=time_stretch_rate
        )

        # Normalize
        max_val = np.max(np.abs(self.current_audio_data))
        if max_val > 0:
            self.current_audio_data = self.current_audio_data / max_val

        return self.current_audio_data

    # Video helpers: thin wrappers over ms_music.visualizations. All accept
    # progress_callback(frame, total), which may raise to abort rendering.
    def create_3d_scan_video(
        self,
        output_path,
        window_size=50,
        duration_seconds=15.0,
        fps=15,
        dpi=100,
        azimuth_rotation=True,
        progress_callback=None,
    ):
        """
        Create animated 3D plot showing a moving window of scans.
        See visualizations.create_3d_scan_video() for details.
        """
        from . import visualizations as viz

        return viz.create_3d_scan_video(
            self,
            output_path,
            window_size,
            duration_seconds,
            fps,
            dpi,
            azimuth_rotation,
            progress_callback=progress_callback,
        )

    def create_3d_heatmap_video(
        self,
        output_path,
        duration_seconds=10.0,
        fps=15,
        dpi=100,
        rotate_speed=1.0,
        progress_callback=None,
    ):
        """
        Create rotating 3D heatmap view of all data.
        See visualizations.create_3d_heatmap_video() for details.
        """
        from . import visualizations as viz

        return viz.create_3d_heatmap_video(
            self,
            output_path,
            duration_seconds,
            fps,
            dpi,
            rotate_speed,
            progress_callback=progress_callback,
        )

    def create_comparison_video(
        self, audio_dict, output_path, fps=15, dpi=80, progress_callback=None
    ):
        """
        Create side-by-side comparison video of multiple sonifications.
        See visualizations.create_comparison_video() for details.
        """
        from . import visualizations as viz

        return viz.create_comparison_video(
            self,
            audio_dict,
            output_path,
            fps,
            dpi,
            progress_callback=progress_callback,
        )

    def create_3d_waterfall_video(
        self,
        output_path,
        duration_seconds,
        fps=15,
        dpi=80,
        max_freq=5000,
        rotate=True,
        progress_callback=None,
    ):
        """
        Create an animated 3D waterfall of the current audio's spectrum.
        See visualizations.create_3d_waterfall_video() for details.
        """
        from . import visualizations as viz

        return viz.create_3d_waterfall_video(
            self,
            output_path,
            duration_seconds,
            fps,
            dpi,
            max_freq,
            rotate,
            progress_callback=progress_callback,
        )

    def create_3d_spectrogram_buildup_video(
        self,
        output_path,
        duration_seconds,
        fps=15,
        dpi=80,
        max_freq=5000,
        colormap="plasma",
        style="bars",
        progress_callback=None,
    ):
        """
        Create an animated 3D spectrogram of the current audio that builds
        up over time.
        See visualizations.create_3d_spectrogram_buildup_video() for details.
        """
        from . import visualizations as viz

        return viz.create_3d_spectrogram_buildup_video(
            self,
            output_path,
            duration_seconds,
            fps,
            dpi,
            max_freq,
            colormap,
            style,
            progress_callback=progress_callback,
        )

    def create_video(
        self,
        output_path: str,
        fps: int = 30,
        dpi: int = 100,
        show_spectrogram: bool = True,
        show_waveform: bool = True,
        show_mz_distribution: bool = False,
        progress_callback=None,
    ):
        """
        Create video visualization of the sonification.
        See visualizations.create_video() for details.
        """
        from . import visualizations as viz

        return viz.create_video(
            self,
            output_path,
            fps,
            dpi,
            show_spectrogram,
            show_waveform,
            show_mz_distribution,
            progress_callback=progress_callback,
        )

    def animate_visualization(
        self,
        output_path: str,
        viz_type: str = "spectrogram",
        fps: int = 30,
        dpi: int = 100,
        duration_seconds=None,
        progress_callback=None,
    ):
        """
        Animate any visualization type and save as video.
        See visualizations.animate_visualization() for details.
        """
        from . import visualizations as viz

        return viz.animate_visualization(
            self,
            output_path,
            viz_type,
            fps,
            dpi,
            duration_seconds,
            progress_callback=progress_callback,
        )
