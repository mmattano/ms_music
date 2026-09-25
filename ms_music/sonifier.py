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
from .musical_quantization import MusicalNoteQuantizer
from .rhythm import (
    RhythmConfig, build_grid, find_notes, note_gain, note_span,
    resample_spectra)


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

        self.ion_mobility_data = None
        self.precursor_mz_list = None      # DDA: precursor m/z per scan
        self.isolation_window_list = None  # DIA: (lo_mz, hi_mz) per scan
        self.retention_time_list = None    # Retention time per scan (min)
        self.ms1_reference_spectra = None  # MS1 spectra for DIA lookup

        print(f"MSSonifier initialized for file: {os.path.basename(filepath)}")
        print(
            f"Target audio duration: {total_duration_minutes} minutes, "
            f"Sample rate: {sample_rate} Hz"
        )

    def load_and_preprocess_data(self, scan_ratio_range=None):
        self.raw_spectra = io.load_mzml_data(
            self.filepath,
            ms_level=self.ms_level,
        )

        if not self.raw_spectra:
            print("No spectra found during loading.")
            return

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
                scan_ratio_range = None  # Process all

            if scan_ratio_range:  # Re-check after potential reset
                num_spectra = len(self.raw_spectra)
                start_index = int(num_spectra * scan_ratio_range[0])
                end_index = int(num_spectra * scan_ratio_range[1])
                self.raw_spectra = self.raw_spectra[start_index:end_index]
                print(
                    f"Selected {len(self.raw_spectra)} scans based on "
                    f"scan_ratio_range {scan_ratio_range}."
                )

        if not self.raw_spectra:  # Check if slicing resulted in empty list
            print("No spectra selected after applying scan_ratio_range.")
            return

        print(f"Loaded {len(self.raw_spectra)} spectra for preprocessing.")

        (
            self.processed_spectra_dfs,
            self.max_intensity_overall,
            self.min_mz_overall,
            self.max_mz_overall,
        ) = io.preprocess_spectra(self.raw_spectra)

        if not self.processed_spectra_dfs:
            print(
                "Preprocessing failed or resulted in no processable spectra."
            )
            return

        # Store per-scan metadata for MS2 features
        self.precursor_mz_list = [
            s.get('precursor_mz') for s in self.raw_spectra
        ]
        self.isolation_window_list = [
            s.get('isolation_window') for s in self.raw_spectra
        ]
        self.retention_time_list = [
            s.get('retention_time') for s in self.raw_spectra
        ]

    def load_ms1_reference(self, filepath: str = None):
        """
        Load MS1 spectra as a reference for DIA precursor tone generation.

        In DIA mode each MS2 scan has an isolation window rather than a
        single selected precursor.  This method loads the raw MS1 peaks
        (m/z, intensity, retention time) so that, when rendering a DIA MS2
        scan, all MS1 peaks that fall inside the isolation window can be
        played simultaneously as precursor tones.

        Args:
            filepath: Path to the mzML file containing MS1 data.  Defaults
                      to the file already loaded into this sonifier instance.
        """
        if filepath is None:
            filepath = self.filepath

        raw = io.load_mzml_data(filepath, ms_level=1)
        if not raw:
            print("Warning: no MS1 spectra found for DIA reference.")
            return

        # Keep only what DIA lookup needs: retention time + peaks
        ref = []
        for s in raw:
            rt = s.get('retention_time')
            if rt is None:
                continue
            ref.append({
                'retention_time': float(rt),
                'mz': s['mz'],
                'intensity': s['intensity'],
            })

        # Sort ascending by retention time for binary-search lookup
        ref.sort(key=lambda x: x['retention_time'])
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
            rts = [s['retention_time'] for s in self.ms1_reference_spectra]
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
                ms1 = (before if abs(before['retention_time'] - rt)
                       <= abs(after['retention_time'] - rt) else after)
        else:
            ms1 = self.ms1_reference_spectra[0]

        mz_arr = ms1['mz']
        int_arr = ms1['intensity']
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
            effect_names = [name for name in dir(
                audio_effects) if name.startswith('apply_')]
            effects_list = ', '.join(effect_names)
            print(f"Available effects: {effects_list}")
            return

        effect_function = getattr(audio_effects, effect_function_name)

        print(f"Applying effect: {effect_name} with params: {effect_params}")

        # Apply the effect
        self.current_audio_data = effect_function(
            self.current_audio_data, self.sample_rate, **effect_params
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
                filtered.append(
                    df.nlargest(max_peaks_per_scan, "intensities")
                )
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
            frequency_mapping, freq_range, self.note_quantizer,
            use_log_distance,
        )

        print(f"Starting sonification using '{method}' method...")
        print(
            f"Frequency mapping: {frequency_mapping}, "
            f"Range: {freq_range[0]}-{freq_range[1]} Hz"
            + (f", scale: {scale} in {root_note}" if scale else "")
        )

        spectra = (
            self._filter_top_peaks(self.processed_spectra_dfs,
                                   max_peaks_per_scan)
            if max_peaks_per_scan is not None
            else self.processed_spectra_dfs
        )

        # Time layout: equal slices per scan, or steps of a musical grid.
        self.rhythm_grid = None
        scan_index = list(range(len(spectra)))
        duration = self.total_duration_seconds
        segment_lengths = None
        if rhythm is not None:
            grid = build_grid(self.total_duration_seconds, rhythm)
            spectra, scan_index = resample_spectra(
                spectra, grid.n_steps, rhythm.aggregate)
            duration = grid.total_seconds
            segment_lengths = grid.segment_lengths(self.sample_rate)
            self.rhythm_grid = grid
            print(f"Rhythm: {grid.describe()} ({grid.n_steps} steps)")

        if self.rhythm_grid is not None:
            audio = self._synthesize_metered(
                spectra, freq_fn, self.rhythm_grid, method,
                scan_index=scan_index, ms2_mode=ms2_mode,
                adsr_settings=adsr_settings,
            )
        else:
            synthesize = (
                self._synthesize_gradient if method == "gradient"
                else self._synthesize_adsr
            )
            kwargs = {}
            if method == "adsr":
                kwargs["adsr_settings"] = adsr_settings
            audio = synthesize(
                spectra, freq_fn, duration,
                segment_lengths=segment_lengths,
                scan_index=scan_index,
                ms2_mode=ms2_mode,
                **kwargs,
            )

        self.current_audio_data = audio
        if (
            self.current_audio_data is None
            or self.current_audio_data.size == 0
        ):
            print("Sonification failed to produce audio data.")
            self.current_audio_data = None

    def _frequency_function(self, frequency_mapping, freq_range,
                            quantizer=None, use_log_distance=True):
        """Return a memoized ``m/z -> Hz`` function for one sonify() call:
        the chosen mapping, optionally snapped to the quantizer's scale."""
        cache = {}

        def freq_fn(mz_value):
            if mz_value in cache:
                return cache[mz_value]
            freq = self._mz_to_frequency(
                mz_value, frequency_mapping, freq_range)
            if quantizer is not None and freq > 0:
                note = (quantizer.quantize_frequency_log(freq)
                        if use_log_distance
                        else quantizer.quantize_frequency(freq))
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
                spectra_df["intensities"].max() / self.max_intensity_overall)
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

    def _precursor_tones(self, scan_idx, spectra_df, t, freq_fn, ms2_mode):
        """Precursor tones for one segment, evaluated on time vector ``t``
        (zeros if none apply)."""
        tones = np.zeros(t.size, dtype=np.float32)
        for freq, amp in self._precursor_pitches(
                scan_idx, spectra_df, freq_fn, ms2_mode):
            tones += np.sin(freq * 2 * math.pi * t) * amp
        return tones

    def _synthesize_metered(self, step_dfs, freq_fn, grid, method,
                            scan_index, ms2_mode=None, adsr_settings=None):
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
            0, total_samples / self.sample_rate, total_samples,
            endpoint=False, dtype=np.float32)
        song = np.zeros(total_samples, dtype=np.float32)

        tracks = {}  # freq -> per-step level

        def add(freq, step, level):
            track = tracks.get(freq)
            if track is None:
                track = tracks[freq] = np.zeros(n_steps, dtype=np.float64)
            track[step] += level

        for step, df in enumerate(step_dfs):
            if not df.empty:
                mzs = df.index.to_numpy(dtype=float)
                levels = (df["intensities"].to_numpy(dtype=float)
                          / self.max_intensity_overall)
                for mz, level in zip(mzs, levels):
                    if mz > 0 and level > 0:
                        freq = freq_fn(mz)
                        if freq > 0:
                            add(freq, step, level)
            for freq, amp in self._precursor_pitches(
                    scan_index[step], df, freq_fn, ms2_mode):
                add(freq, step, amp)

        settings = adsr_settings or {}
        randomize = settings.get("randomize", True)
        n_notes = 0
        for frequency, levels in tqdm(tracks.items(),
                                      desc="Synthesizing notes"):
            for start, end in find_notes(levels,
                                         grid.config.note_threshold):
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
                        a, d = random.uniform(0.01, 0.10), random.uniform(0.02, 0.15)
                        sus = random.uniform(0.4, 0.8)
                        r = random.uniform(0.05, max(0.06, (1 - a - d) * 0.8))
                    else:
                        a = settings.get("attack_time_pc", 0.05)
                        d = settings.get("decay_time_pc", 0.1)
                        sus = settings.get("sustain_level_pc", 0.7)
                        r = settings.get("release_time_pc", 0.15)
                    peak = float(levels[start:end].max()) * float(
                        grid.accents[start])
                    gain = peak * self._adsr_envelope(i1 - i0, a, d, sus, r)
                song[i0:i1] += np.sin(frequency * 2 * math.pi * t) * gain
                n_notes += 1

        print(f"Rendered {n_notes} notes from {len(tracks)} pitches")
        return song

    FREQUENCY_MAPPINGS = (
        "inverse_log", "power_law", "musical_octaves", "chromatic", "linear",
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

    def _synthesize_gradient(self, spectra_dfs, freq_fn, duration,
                             segment_lengths=None, scan_index=None,
                             ms2_mode=None):
        """Continuous tones: one sine per distinct pitch, its loudness
        stepping with the summed normalized intensity of the m/z values
        mapped to that pitch in each segment."""
        num_segments = len(spectra_dfs)
        lengths, starts, total_samples = self._segment_layout(
            num_segments, duration, segment_lengths)
        if scan_index is None:
            scan_index = list(range(num_segments))

        time_vector = np.linspace(
            0,
            total_samples / self.sample_rate,
            total_samples,
            endpoint=False,
            dtype=np.float32,
        )
        song = np.zeros(total_samples, dtype=np.float32)

        # Long table of (segment, m/z, intensity), then per-pitch rows of
        # summed intensities across segments.
        frames = []
        for seg, df in enumerate(spectra_dfs):
            if df.empty:
                continue
            frames.append(pd.DataFrame({
                "segment": seg,
                "mz": df.index.to_numpy(dtype=float),
                "intensity": df["intensities"].to_numpy(dtype=float),
            }))
        if frames:
            peaks = pd.concat(frames, ignore_index=True)
            peaks = peaks[peaks["mz"] > 0]
            unique_mz = peaks["mz"].unique()
            freq_of = {mz: freq_fn(mz) for mz in unique_mz}
            peaks["freq"] = peaks["mz"].map(freq_of)
            peaks = peaks[peaks["freq"] > 0]
            peaks["intensity"] /= self.max_intensity_overall
            per_pitch = peaks.groupby(["freq", "segment"])["intensity"].sum()
            # Synthesize with the exact frequency objects the mapping
            # returned (Python float vs numpy float64 decides float32 vs
            # float64 sine math under NumPy 2 promotion rules).
            pitch_obj = {f: f for f in freq_of.values()}
            print(f"Mapped {len(unique_mz)} m/z values to "
                  f"{per_pitch.index.get_level_values(0).nunique()} "
                  f"frequencies")

            for frequency, rows in tqdm(
                per_pitch.groupby(level=0), desc="Synthesizing pitches"
            ):
                intensities = np.zeros(num_segments, dtype=np.float32)
                intensities[rows.index.get_level_values(1)] = rows.to_numpy()
                gain = np.repeat(intensities, lengths)
                frequency = pitch_obj[frequency]
                song += np.sin(frequency * 2 * math.pi * time_vector) * gain

        if ms2_mode in ("dda", "dia"):
            for seg, df in enumerate(spectra_dfs):
                start, end = starts[seg], starts[seg] + lengths[seg]
                song[start:end] += self._precursor_tones(
                    scan_index[seg], df, time_vector[start:end], freq_fn,
                    ms2_mode)

        return song

    def _adsr_envelope(self, length_samples: int, attack_time_pc: float,
                       decay_time_pc: float, sustain_level_pc: float,
                       release_time_pc: float):
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
            total_adr_samples = attack_samples + decay_samples + \
                                release_samples
            scale_factor = length_samples / total_adr_samples
            attack_samples = int(attack_samples * scale_factor)
            decay_samples = int(decay_samples * scale_factor)
            release_samples = int(release_samples * scale_factor)
            # Recalculate to ensure integer sum matches
            release_samples = length_samples - (attack_samples + decay_samples)

        sustain_samples = max(
            0, length_samples - (
                attack_samples + decay_samples + release_samples))

        envelope = np.zeros(length_samples, dtype=np.float32)
        current_pos = 0

        # Attack
        if attack_samples > 0:
            envelope[
                current_pos: current_pos + attack_samples] = np.linspace(
                    0, 1, attack_samples, endpoint=False)
        current_pos += attack_samples

        # Decay
        if decay_samples > 0 and current_pos < length_samples:
            # Start decay from 1 or sustain_level if no attack
            env_start_val = 1.0 if attack_samples > 0 else sustain_level_pc
            decay_end = min(current_pos + decay_samples, length_samples)
            envelope[
                current_pos: decay_end] = np.linspace(
                    env_start_val, sustain_level_pc, decay_end - current_pos,
                    endpoint=False)
        current_pos += decay_samples

        # Sustain
        if sustain_samples > 0 and current_pos < length_samples:
            sustain_end = min(current_pos + sustain_samples, length_samples)
            envelope[current_pos: sustain_end] = sustain_level_pc
        current_pos += sustain_samples

        # Release
        if release_samples > 0 and current_pos < length_samples:
            release_end = min(current_pos + release_samples, length_samples)
            # Ensure release actually fits
            actual_release_samples = release_end - current_pos
            if actual_release_samples > 0:
                envelope[
                    current_pos: release_end] = np.linspace(
                        sustain_level_pc, 0, actual_release_samples,
                        endpoint=False)

        # Ensure last point is 0 if there's a release phase that reaches
        # the end
        if release_samples > 0 and (
                current_pos + release_samples >= length_samples):
            if length_samples > 0:
                envelope[-1] = 0.0

        return envelope

    def _synthesize_adsr(self, spectra_dfs, freq_fn, duration,
                         segment_lengths=None, scan_index=None,
                         ms2_mode=None, adsr_settings=None):
        """One note per segment: all its peaks summed, normalized, and
        shaped by an ADSR envelope (fixed or randomized per note)."""
        target_samples = (
            int(duration * self.sample_rate) if segment_lengths is None
            else int(np.sum(segment_lengths)))
        if not spectra_dfs:
            print("Warning (ADSR): No processed spectra. "
                  "Returning silent audio.")
            return np.zeros(target_samples, dtype=np.float32)
        if self.max_intensity_overall <= 0:
            print("Warning (ADSR): Max overall intensity is non-positive. "
                  "Sonification will be silent.")
            return np.zeros(target_samples, dtype=np.float32)

        num_segments = len(spectra_dfs)
        lengths, _starts, _total = self._segment_layout(
            num_segments, duration, segment_lengths)
        if scan_index is None:
            scan_index = list(range(num_segments))

        if adsr_settings is None:
            adsr_settings = {}
        use_random_adsr = adsr_settings.get('randomize', True)
        fixed_attack_pc = adsr_settings.get('attack_time_pc', 0.05)
        fixed_decay_pc = adsr_settings.get('decay_time_pc', 0.1)
        fixed_sustain_level = adsr_settings.get('sustain_level_pc', 0.7)
        fixed_release_pc = adsr_settings.get('release_time_pc', 0.15)

        song_parts = []
        print("Generating audio with ADSR method...")
        for seg, scan_df in enumerate(
                tqdm(spectra_dfs, desc="Processing Scans (ADSR)",
                     unit="scan")):
            n = int(lengths[seg])
            # Each note starts at phase zero on its own local time axis.
            t = np.linspace(0, n / self.sample_rate, n, endpoint=False,
                            dtype=np.float32)
            segment = np.zeros(n, dtype=np.float32)

            if not scan_df.empty:
                mzs = scan_df.index.to_numpy()
                intensities = scan_df["intensities"].to_numpy()
                for mz_value, intensity in zip(mzs, intensities):
                    if mz_value <= 0 or intensity <= 0:
                        continue
                    frequency = freq_fn(mz_value)
                    if frequency <= 0:
                        continue
                    segment += (np.sin(frequency * 2 * math.pi * t)
                                * (intensity / self.max_intensity_overall))

            if ms2_mode in ("dda", "dia"):
                segment += self._precursor_tones(
                    scan_index[seg], scan_df, t, freq_fn, ms2_mode)

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
                n, attack_t_pc, decay_t_pc, sustain_l_pc, release_t_pc)
            song_parts.append(segment * envelope)

        song = np.concatenate(song_parts)
        if len(song) > target_samples:
            song = song[:target_samples]
        elif len(song) < target_samples:
            song = np.pad(song, (0, target_samples - len(song)), 'constant')
        return song

    def load_fid_data(
        self,
        fid_filepath: str,
        original_sample_rate: float = 10e6,  # 10 MHz default
        conversion_factor: float = 2**12,    # 4096 default
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
                len(fid_data) * self.sample_rate / effective_sample_rate)
            resampled_data = signal.resample(fid_data, new_length)
        else:
            resampled_data = fid_data.copy()

        # Time-stretch to exact target duration while preserving pitch
        current_duration = len(resampled_data) / self.sample_rate
        time_stretch_rate = current_duration / self.total_duration_seconds

        # Use librosa's pitch-preserving time stretch
        self.current_audio_data = librosa.effects.time_stretch(
            resampled_data, rate=time_stretch_rate)

        # Normalize
        max_val = np.max(np.abs(self.current_audio_data))
        if max_val > 0:
            self.current_audio_data = self.current_audio_data / max_val

        return self.current_audio_data

    # Video helpers: thin wrappers over ms_music.visualizations. All accept
    # progress_callback(frame, total), which may raise to abort rendering.
    def create_3d_scan_video(self, output_path, window_size=50,
                             duration_seconds=15.0, fps=15, dpi=100,
                             azimuth_rotation=True, progress_callback=None):
        """
        Create animated 3D plot showing a moving window of scans.
        See visualizations.create_3d_scan_video() for details.
        """
        from . import visualizations as viz
        return viz.create_3d_scan_video(
            self, output_path, window_size, duration_seconds,
            fps, dpi, azimuth_rotation, progress_callback=progress_callback
        )

    def create_3d_heatmap_video(self, output_path, duration_seconds=10.0,
                                fps=15, dpi=100, rotate_speed=1.0,
                                progress_callback=None):
        """
        Create rotating 3D heatmap view of all data.
        See visualizations.create_3d_heatmap_video() for details.
        """
        from . import visualizations as viz
        return viz.create_3d_heatmap_video(
            self, output_path, duration_seconds, fps, dpi, rotate_speed,
            progress_callback=progress_callback
        )

    def create_comparison_video(self, audio_dict, output_path, fps=15, dpi=80,
                                progress_callback=None):
        """
        Create side-by-side comparison video of multiple sonifications.
        See visualizations.create_comparison_video() for details.
        """
        from . import visualizations as viz
        return viz.create_comparison_video(
            self, audio_dict, output_path, fps, dpi,
            progress_callback=progress_callback
        )

    def create_3d_waterfall_video(self, output_path, duration_seconds,
                                  fps=15, dpi=80, max_freq=5000,
                                  rotate=True, progress_callback=None):
        """
        Create an animated 3D waterfall of the current audio's spectrum.
        See visualizations.create_3d_waterfall_video() for details.
        """
        from . import visualizations as viz
        return viz.create_3d_waterfall_video(
            self, output_path, duration_seconds, fps, dpi, max_freq, rotate,
            progress_callback=progress_callback
        )

    def create_3d_spectrogram_buildup_video(self, output_path,
                                            duration_seconds, fps=15,
                                            dpi=80, max_freq=5000,
                                            colormap='plasma',
                                            style='bars',
                                            progress_callback=None):
        """
        Create an animated 3D spectrogram of the current audio that builds
        up over time.
        See visualizations.create_3d_spectrogram_buildup_video() for details.
        """
        from . import visualizations as viz
        return viz.create_3d_spectrogram_buildup_video(
            self, output_path, duration_seconds, fps, dpi,
            max_freq, colormap, style, progress_callback=progress_callback
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
            self, output_path, fps, dpi,
            show_spectrogram, show_waveform, show_mz_distribution,
            progress_callback=progress_callback
        )

    def animate_visualization(
        self,
        output_path: str,
        viz_type: str = 'spectrogram',
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
            self, output_path, viz_type, fps, dpi, duration_seconds,
            progress_callback=progress_callback
        )
