import numpy as np
from tqdm import tqdm
import os
import math
import matplotlib.pyplot as plt
from scipy import signal
import random

from . import io
from . import effects as audio_effects
from .musical_quantization import MusicalNoteQuantizer


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
        self.metadata_harmonization = False

        self.raw_spectra = None
        self.processed_spectra_dfs = None
        self.max_intensity_overall = 0
        self.min_mz_overall = 0
        self.max_mz_overall = 0

        self.current_audio_data = None
        self.note_quantizer = None

        print(f"MSSonifier initialized for file: {os.path.basename(filepath)}")
        print(
            f"Target audio duration: {total_duration_minutes} minutes, "
            f"Sample rate: {sample_rate} Hz"
        )

    def load_and_preprocess_data(self, scan_ratio_range=None):
        self.raw_spectra = io.load_mzml_data(
            self.filepath,
            ms_level=self.ms_level,
            metadata_harmonization=self.metadata_harmonization,
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

    def sonify(self, method: str = "gradient", method_params: dict = None):
        """
        Sonification with better frequency mapping options.

        Args:
            method: 'gradient' or 'adsr'
            method_params: Dictionary with parameters including:
                - frequency_mapping: 'inverse_log', 'power_law',
                  'musical_octaves', 'chromatic', 'linear'
                - freq_range: (min_freq, max_freq) tuple in Hz
                - overlap_percentage: for gradient method
                - adsr_settings: for ADSR method
        """
        if (
            self.processed_spectra_dfs is None
            or not self.processed_spectra_dfs
        ):
            print(
                "Data not loaded or preprocessed. Please call "
                "load_and_preprocess_data() first."
            )
            return

        if method_params is None:
            method_params = {}

        # Default parameters
        frequency_mapping = method_params.get(
            "frequency_mapping", "inverse_log"
        )
        freq_range = method_params.get("freq_range", (200.0, 4000.0))

        print(f"Starting sonification using '{method}' method...")
        print(
            f"Frequency mapping: {frequency_mapping}, "
            f"Range: {freq_range[0]}-{freq_range[1]} Hz"
        )

        if method == "gradient":
            overlap_percentage = method_params.get("overlap_percentage", 0.05)
            self.current_audio_data = self._generate_audio_gradient(
                frequency_mapping=frequency_mapping,
                freq_range=freq_range,
                overlap_percentage=overlap_percentage,
            )
        elif method == "adsr":
            adsr_settings = method_params.get("adsr_settings", None)
            self.current_audio_data = self._generate_audio_adsr(
                frequency_mapping=frequency_mapping,
                freq_range=freq_range,
                adsr_settings=adsr_settings,
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. Choose 'gradient' or 'adsr'."
            )

        if (
            self.current_audio_data is None
            or self.current_audio_data.size == 0
        ):
            print("Sonification failed to produce audio data.")
            self.current_audio_data = None

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
            return self._mz_to_frequency_linear(mz_value, freq_range)

        else:
            raise ValueError(f"Unknown mapping type: {mapping_type}")

    def _generate_audio_gradient(
        self, frequency_mapping, freq_range, overlap_percentage
    ):
        """Gradient method with frequency mapping."""

        num_scans = len(self.processed_spectra_dfs)
        samples_per_scan = round(
            self.sample_rate * self.total_duration_seconds / num_scans
        )
        total_samples = int(samples_per_scan * num_scans)

        time_vector = np.linspace(
            0,
            total_samples / self.sample_rate,
            total_samples,
            endpoint=False,
            dtype=np.float32,
        )

        # Get all unique m/z values
        all_mz_values = set()
        for scan_df in self.processed_spectra_dfs:
            if not scan_df.empty:
                all_mz_values.update(scan_df.index)

        song = np.zeros(total_samples, dtype=np.float32)
        # overlap_samples = round(overlap_percentage * samples_per_scan)
        # I somehow never brought this back, need to check if it makes sense
        # to keep it

        for mz_value in tqdm(all_mz_values, desc="Processing m/z values"):
            if mz_value <= 0:
                continue

            # Convert m/z to frequency using selected mapping
            frequency = self._mz_to_frequency(
                mz_value, frequency_mapping, freq_range
            )
            if frequency <= 0:
                continue

            # Generate sine wave at this frequency
            mz_sine_wave = np.sin(frequency * 2 * math.pi * time_vector)
            modulated_mz_wave_component = np.zeros_like(mz_sine_wave)

            # Get normalized intensities for this m/z across all scans
            normalized_intensities_for_mz = np.zeros(
                num_scans, dtype=np.float32
            )
            for i, scan_df in enumerate(self.processed_spectra_dfs):
                if mz_value in scan_df.index:
                    normalized_intensities_for_mz[i] = (
                        scan_df.loc[mz_value, "intensities"]
                        / self.max_intensity_overall
                    )

            # Apply intensity modulation (apply overlap)
            for scan_idx in range(num_scans):
                start_sample_idx = scan_idx * samples_per_scan
                end_sample_idx = start_sample_idx + samples_per_scan

                current_segment_sine = mz_sine_wave[
                    start_sample_idx:end_sample_idx
                ]
                if current_segment_sine.size == 0:
                    continue

                current_intensity = normalized_intensities_for_mz[scan_idx]
                modulated_mz_wave_component[
                    start_sample_idx:end_sample_idx
                ] = (current_segment_sine * current_intensity)

            song += modulated_mz_wave_component

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

    def _generate_audio_adsr(
            self,
            frequency_mapping, freq_range, adsr_settings=None):
        """Generate audio with ADSR envelopes and proper frequency mapping."""
        if not self.processed_spectra_dfs:
            print("Warning (ADSR): No processed spectra. "
                  "Returning silent audio.")
            return np.zeros(
                int(
                    self.total_duration_seconds * self.sample_rate
                    ), dtype=np.float32)

        if self.max_intensity_overall <= 0:
            print("Warning (ADSR): Max overall intensity is non-positive. "
                  "Sonification will be silent.")
            return np.zeros(
                int(
                    self.total_duration_seconds * self.sample_rate
                    ), dtype=np.float32)

        num_scans = len(self.processed_spectra_dfs)
        samples_per_scan = round(
            self.sample_rate * self.total_duration_seconds / num_scans)
        total_samples_target = int(
            self.total_duration_seconds * self.sample_rate)

        # Default ADSR settings
        if adsr_settings is None:
            adsr_settings = {}

        use_random_adsr = adsr_settings.get('randomize', True)
        fixed_attack_pc = adsr_settings.get('attack_time_pc', 0.05)
        fixed_decay_pc = adsr_settings.get('decay_time_pc', 0.1)
        fixed_sustain_level = adsr_settings.get('sustain_level_pc', 0.7)
        fixed_release_pc = adsr_settings.get('release_time_pc', 0.15)

        song_parts = []
        time_per_scan_segment = np.linspace(
            0, samples_per_scan / self.sample_rate,
            samples_per_scan, endpoint=False, dtype=np.float32)

        print("Generating audio with ADSR method...")
        for scan_df in tqdm(
                            self.processed_spectra_dfs,
                            desc="Processing Scans (ADSR)",
                            unit="scan"):
            scan_audio_segment = np.zeros(samples_per_scan, dtype=np.float32)

            if not scan_df.empty:
                for mz_value, row_series in scan_df.iterrows():
                    intensity = row_series['intensities']
                    if mz_value <= 0 or intensity <= 0:
                        continue

                    # Use proper frequency mapping
                    frequency = self._mz_to_frequency(
                        mz_value, frequency_mapping, freq_range)
                    if frequency <= 0:
                        continue

                    normalized_intensity = (intensity /
                                            self.max_intensity_overall)
                    mz_sine_wave_segment = np.sin(
                        frequency * 2 * math.pi * time_per_scan_segment)
                    scan_audio_segment += (mz_sine_wave_segment *
                                           normalized_intensity)

            # Normalize segment
            max_abs_segment = np.max(np.abs(scan_audio_segment))
            if max_abs_segment > 0:
                scan_audio_segment /= max_abs_segment

            # Generate ADSR envelope
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
                samples_per_scan, attack_t_pc, decay_t_pc,
                sustain_l_pc, release_t_pc)
            song_parts.append(scan_audio_segment * envelope)

        if not song_parts:
            return np.zeros(total_samples_target, dtype=np.float32)

        song = np.concatenate(song_parts)

        # Ensure correct length
        if len(song) > total_samples_target:
            song = song[:total_samples_target]
        elif len(song) < total_samples_target:
            song = np.pad(
                song,
                (0, total_samples_target - len(song)),
                'constant',
                constant_values=0
            )

        return song

    def setup_musical_quantization(
        self,
        scale="chromatic",
        root_note="C",
        tuning_freq=440.0,
        freq_range=None,
    ):
        """
        Set up musical note quantization for sonification.

        Args:
            scale: Musical scale ('chromatic', 'major', 'minor',
                   'pentatonic_major',
                   'pentatonic_minor', 'blues', 'dorian', 'mixolydian',
                   'whole_tone')
            root_note: Root note of the scale ('C', 'D', 'E', 'F', 'G',
                   'A', 'B')
            tuning_freq: Frequency of A4 in Hz (standard is 440.0)
            freq_range: (min_freq, max_freq) or None to use default (200, 4000)
        """
        if freq_range is None:
            freq_range = (200, 4000)

        self.note_quantizer = MusicalNoteQuantizer(
            scale=scale,
            root_note=root_note,
            tuning_freq=tuning_freq,
            freq_range=freq_range,
        )

        print(f"Musical quantization set up: {scale} scale in {root_note}")
        return self.note_quantizer

    def sonify_quantized(self, base_mapping="inverse_log", method_params=None):
        """
        Sonify using quantized musical notes.

        Args:
            base_mapping: Base frequency mapping before quantization
                        ('inverse_log', 'power_law', 'musical_octaves',
                        'chromatic', 'linear')
            method_params: Dictionary with parameters:
                - freq_range: (min_freq, max_freq) for base mapping
                - overlap_percentage: for intensity transitions
                - use_log_distance: Use logarithmic distance for note selection
                - scale: Musical scale to use
                - root_note: Root note of the scale
                - tuning_freq: A4 frequency
        """
        if (
            self.processed_spectra_dfs is None
            or not self.processed_spectra_dfs
        ):
            print(
                "Data not loaded or preprocessed. Please call "
                "load_and_preprocess_data() first."
            )
            return

        if method_params is None:
            method_params = {}

        # Set up quantizer if not already done or if parameters changed
        scale = method_params.get("scale", "major")
        root_note = method_params.get("root_note", "C")
        tuning_freq = method_params.get("tuning_freq", 440.0)
        freq_range = method_params.get("freq_range", (200, 3000))

        if (
            self.note_quantizer is None
            or self.note_quantizer.scale != scale
            or self.note_quantizer.root_note != root_note
            or self.note_quantizer.tuning_freq != tuning_freq
        ):

            self.setup_musical_quantization(
                scale, root_note, tuning_freq, freq_range
            )

        overlap_percentage = method_params.get("overlap_percentage", 0.05)
        use_log_distance = method_params.get("use_log_distance", True)

        self.current_audio_data = self._generate_quantized_audio(
            base_mapping=base_mapping,
            freq_range=freq_range,
            overlap_percentage=overlap_percentage,
            use_log_distance=use_log_distance,
        )

        if (
            self.current_audio_data is not None
            and self.current_audio_data.size > 0
        ):
            None
        else:
            print("Quantized sonification failed to produce audio data.")
            self.current_audio_data = None

    def _generate_quantized_audio(
        self, base_mapping, freq_range, overlap_percentage, use_log_distance
    ):
        """Generate audio with quantized frequencies."""

        num_scans = len(self.processed_spectra_dfs)
        samples_per_scan = round(
            self.sample_rate * self.total_duration_seconds / num_scans
        )
        total_samples = int(samples_per_scan * num_scans)

        time_vector = np.linspace(
            0,
            total_samples / self.sample_rate,
            total_samples,
            endpoint=False,
            dtype=np.float32,
        )

        # Get all unique m/z values
        all_mz_values = set()
        for scan_df in self.processed_spectra_dfs:
            if not scan_df.empty:
                all_mz_values.update(scan_df.index)

        if not all_mz_values:
            return np.zeros(total_samples, dtype=np.float32)

        song = np.zeros(total_samples, dtype=np.float32)
        # overlap_samples = round(overlap_percentage * samples_per_scan)

        # Create mapping from m/z to quantized frequencies
        mz_to_freq_map = {}
        unique_frequencies = set()

        for mz_value in tqdm(
            sorted(all_mz_values), desc="Quantizing frequencies"
        ):
            if mz_value <= 0:
                continue

            # Get continuous frequency using base mapping
            if base_mapping == "inverse_log":
                continuous_freq = self._mz_to_frequency_inverse_log(
                    mz_value, freq_range
                )
            elif base_mapping == "power_law":
                continuous_freq = self._mz_to_frequency_power_law(
                    mz_value, freq_range
                )
            elif base_mapping == "musical_octaves":
                continuous_freq = self._mz_to_frequency_musical_octaves(
                    mz_value,
                )
            elif base_mapping == "chromatic":
                continuous_freq = self._mz_to_frequency_chromatic(
                    mz_value,
                )
            elif base_mapping == "linear":
                continuous_freq = self._mz_to_frequency_linear(
                    mz_value,
                )
            else:
                raise ValueError(f"Unknown mapping type: {base_mapping}")

            # Quantize to nearest musical note
            if use_log_distance:
                note_info = self.note_quantizer.quantize_frequency_log(
                    continuous_freq
                )
            else:
                note_info = self.note_quantizer.quantize_frequency(
                    continuous_freq
                )

            quantized_freq = note_info["frequency"]
            mz_to_freq_map[mz_value] = quantized_freq
            unique_frequencies.add(quantized_freq)

        print(
            f"Mapped {len(all_mz_values)} m/z values "
            f"to {len(unique_frequencies)} unique frequencies"
        )

        # Group m/z values by their quantized frequency
        freq_to_mz_groups = {}
        for mz_value, freq in mz_to_freq_map.items():
            if freq not in freq_to_mz_groups:
                freq_to_mz_groups[freq] = []
            freq_to_mz_groups[freq].append(mz_value)

        for frequency, mz_group in tqdm(
            freq_to_mz_groups.items(), desc="Processing frequencies"
        ):
            if frequency <= 0:
                continue

            # Generate sine wave at this quantized frequency
            mz_sine_wave = np.sin(frequency * 2 * math.pi * time_vector)
            modulated_mz_wave_component = np.zeros_like(mz_sine_wave)

            # Combine intensities from all m/z values that map to this
            # frequency
            combined_intensities = np.zeros(num_scans, dtype=np.float32)
            for mz_value in mz_group:
                for i, scan_df in enumerate(self.processed_spectra_dfs):
                    if mz_value in scan_df.index:
                        combined_intensities[i] += (
                            scan_df.loc[mz_value, "intensities"]
                            / self.max_intensity_overall
                        )

            # Apply intensity modulation with simple transitions
            for scan_idx in range(num_scans):
                start_sample_idx = scan_idx * samples_per_scan
                end_sample_idx = start_sample_idx + samples_per_scan

                current_segment_sine = mz_sine_wave[
                    start_sample_idx:end_sample_idx
                ]
                if current_segment_sine.size == 0:
                    continue

                current_intensity = combined_intensities[scan_idx]

                # Simple intensity modulation
                modulated_mz_wave_component[
                    start_sample_idx:end_sample_idx
                ] = (current_segment_sine * current_intensity)

            song += modulated_mz_wave_component

        return song

    def _mz_to_frequency_inverse_log(self, mz_value, freq_range):
        """Maps m/z values to frequencies using inverse logarithmic scaling."""
        if mz_value <= 0 or self.min_mz_overall <= 0:
            return freq_range[0]

        mz_normalized = (mz_value - self.min_mz_overall) / (
            self.max_mz_overall - self.min_mz_overall
        )
        mz_normalized = np.clip(mz_normalized, 0, 1)

        log_ratio = math.log(freq_range[1] / freq_range[0])
        frequency = freq_range[1] * math.exp(-mz_normalized * log_ratio)

        return frequency

    def _mz_to_frequency_power_law(self, mz_value, freq_range, exponent=1.5):
        """Maps m/z values using a power law relationship."""
        if mz_value <= 0:
            return freq_range[0]

        mz_normalized = (mz_value - self.min_mz_overall) / (
            self.max_mz_overall - self.min_mz_overall
        )
        mz_normalized = np.clip(mz_normalized, 0, 1)

        freq_normalized = (1 - mz_normalized) ** exponent

        log_freq_min = math.log(freq_range[0])
        log_freq_max = math.log(freq_range[1])
        log_frequency = log_freq_min + freq_normalized * (
            log_freq_max - log_freq_min
        )

        return math.exp(log_frequency)

    def _mz_to_frequency_musical_octaves(
        self, mz_value, base_freq=440.0, num_octaves=4
    ):
        """Maps m/z to frequencies using musical octaves."""
        if mz_value <= 0:
            return base_freq

        mz_normalized = (mz_value - self.min_mz_overall) / (
            self.max_mz_overall - self.min_mz_overall
        )
        mz_normalized = np.clip(mz_normalized, 0, 1)

        mz_inverted = 1 - mz_normalized
        octave_position = mz_inverted * num_octaves
        frequency = base_freq * (2**octave_position)

        return frequency

    def _mz_to_frequency_chromatic(
        self, mz_value, base_freq=261.63, num_semitones=48
    ):
        """Maps m/z to frequencies using chromatic scale."""
        if mz_value <= 0:
            return base_freq

        mz_normalized = (mz_value - self.min_mz_overall) / (
            self.max_mz_overall - self.min_mz_overall
        )
        mz_normalized = np.clip(mz_normalized, 0, 1)
        mz_inverted = 1 - mz_normalized

        semitone_position = mz_inverted * num_semitones
        frequency = base_freq * (2 ** (semitone_position / 12))

        return frequency

    def _mz_to_frequency_linear(self, mz_value, freq_range):
        """Maps m/z values to frequencies using linear scaling."""
        if mz_value <= 0:
            return freq_range[0]

        mz_normalized = (mz_value - self.min_mz_overall) / (
            self.max_mz_overall - self.min_mz_overall
        )
        mz_normalized = np.clip(mz_normalized, 0, 1)

        return freq_range[0] + mz_normalized * (freq_range[1] - freq_range[0])


# So this really should be rewritten and integrated into the MSSonifier, I was
# just lazy and didn't do it
class FIDProcessor:
    """Process FID data from binary files, convert to audio"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.fid_data = None
        self.audio_data = None

    def read_fid(self, filepath: str, max_points: int = 2**20) -> np.ndarray:
        """Read FID binary file"""
        with open(filepath, "rb") as f:
            # Read as little-endian 32-bit integers
            data = np.fromfile(f, dtype="<i4", count=max_points)
        self.fid_data = data.astype(np.float32)
        return self.fid_data

    def plot_fid(self, title: str = "FID Data"):
        """Plot FID data"""
        if self.fid_data is None:
            return
        plt.figure(figsize=(12, 4))
        plt.plot(self.fid_data, linewidth=0.5)
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.show()

    def to_audio(self, original_rate: float = 4092.0) -> np.ndarray:
        """Convert FID to audio format."""
        if self.fid_data is None:
            return None

        # Resample to target rate
        if original_rate != self.sample_rate:
            new_length = int(
                len(self.fid_data) * self.sample_rate / original_rate
            )
            self.audio_data = signal.resample(self.fid_data, new_length)
        else:
            self.audio_data = self.fid_data.copy()

        # Normalize
        if np.max(np.abs(self.audio_data)) > 0:
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data))

        return self.audio_data


# Add to existing sonifier
def add_fid_to_sonifier(sonifier_instance):
    """Add FID processing to existing MSSonifier."""

    def sonify_fid(fid_path: str, quantize: bool = False):
        processor = FIDProcessor(sample_rate=sonifier_instance.sample_rate)
        processor.read_fid(fid_path)
        processor.to_audio()
        sonifier_instance.current_audio_data = processor.audio_data

        return sonifier_instance.current_audio_data

    # Add method to sonifier
    sonifier_instance.sonify_fid = sonify_fid
    return sonifier_instance
