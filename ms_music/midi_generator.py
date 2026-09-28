import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from scipy import signal as scipy_signal
from tqdm import tqdm
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from .musical_quantization import (
    MusicalNoteQuantizer,
    MetricalQuantizer,
    MusicMeter,
    QuantizationMode,
    _coerce_meter,
    _coerce_quant_mode,
    mz_to_frequency_inverse_log,
    mz_to_frequency_power_law,
    mz_to_frequency_musical_octaves,
    mz_to_frequency_chromatic,
    mz_to_frequency_linear,
)
from . import io


@dataclass
class MidiNote:
    """Represents a MIDI note with timing and musical information."""

    start_time: int  # MIDI ticks
    duration: int  # MIDI ticks
    midi_number: int  # MIDI note number (0-127)
    velocity: int  # MIDI velocity (0-127)
    channel: int  # MIDI channel (0-15)
    original_mz: float
    original_intensity: float
    retention_time: float
    note_name: str = ""
    frequency: float = 0.0


@dataclass
class MidiConfig:
    """Configuration for MIDI generation."""

    scale: str = "major"
    root_note: str = "C"
    tempo: int = 120
    meter: MusicMeter = MusicMeter.FOUR_FOUR
    quantization_mode: QuantizationMode = QuantizationMode.STRICT_GRID
    edo_divisions: int = 12
    use_just_intonation: bool = False

    def __post_init__(self):
        if isinstance(self.meter, str):
            self.meter = _coerce_meter(self.meter)
        if isinstance(self.quantization_mode, str):
            self.quantization_mode = _coerce_quant_mode(self.quantization_mode)


class MSPeakDetector:
    """Peak detector with m/z clustering and deduplication."""

    def __init__(
        self,
        intensity_threshold_percentile: float = 90.0,
        min_peak_width_scans: int = 3,
        max_peaks_per_scan: int = 1000,
        mz_tolerance_ppm: float = 20.0,
        mz_tolerance_da: float = 0.01,
        use_ppm_tolerance: bool = True,
        consolidation_method: str = "weighted_average",
    ):
        """
        Initialize peak detector with clustering.

        Args:
            intensity_threshold_percentile: Minimum intensity percentile for
            peak detection
            min_peak_width_scans: Minimum width in number of scans
            max_peaks_per_scan: Maximum peaks to extract per scan
            mz_tolerance_ppm: m/z tolerance in parts per million for clustering
            mz_tolerance_da: m/z tolerance in Daltons for clustering
            use_ppm_tolerance: Whether to use ppm (True) or Da (False)
            tolerance
            consolidation_method: How to consolidate clustered peaks
            ("weighted_average", "max_intensity", "median")
        """
        self.intensity_threshold_percentile = intensity_threshold_percentile
        self.min_peak_width_scans = min_peak_width_scans
        self.max_peaks_per_scan = max_peaks_per_scan
        self.mz_tolerance_ppm = mz_tolerance_ppm
        self.mz_tolerance_da = mz_tolerance_da
        self.use_ppm_tolerance = use_ppm_tolerance
        self.consolidation_method = consolidation_method

    def _detect_peaks(
        self,
        processed_spectra_dfs,
        max_intensity_overall,
        apply_clustering=True,
    ):
        """
        Detect peaks with optional m/z clustering.

        Args:
            processed_spectra_dfs: List of processed spectra DataFrames
            max_intensity_overall: Maximum intensity for normalization
            apply_clustering: Whether to apply m/z clustering to remove
            overlaps

        Returns:
            List of peak dictionaries
        """
        # Always start with raw peak detection
        raw_peaks = self._detect_raw_peaks(
            processed_spectra_dfs, max_intensity_overall
        )

        if not apply_clustering or not raw_peaks:
            return raw_peaks

        # Apply clustering to reduce overlapping peaks
        clustered_peaks = self._cluster_and_consolidate_peaks(raw_peaks)

        return clustered_peaks

    def _detect_raw_peaks(self, processed_spectra_dfs, max_intensity_overall):
        peaks = []

        # Global intensity threshold
        all_intensities = []
        for scan_df in processed_spectra_dfs:
            if not scan_df.empty:
                all_intensities.extend(scan_df["intensities"].values)

        if not all_intensities:
            return peaks

        intensity_threshold = np.percentile(
            all_intensities, self.intensity_threshold_percentile
        )
        n_scans = len(processed_spectra_dfs)

        for scan_idx, scan_df in enumerate(processed_spectra_dfs):
            if scan_df.empty:
                continue

            retention_time = scan_idx / n_scans if n_scans > 1 else 0.0

            mz_arr = scan_df.index.values.astype(int)
            int_arr = scan_df["intensities"].values.astype(np.float64)

            # Build a dense intensity array over this scan's m/z span so
            # scipy.signal.find_peaks can detect true local maxima.
            mz_min, mz_max = int(mz_arr.min()), int(mz_arr.max())
            span = mz_max - mz_min + 1

            if span > 1 and span <= 100_000:
                dense = np.zeros(span, dtype=np.float64)
                dense[mz_arr - mz_min] = int_arr

                # Require each peak to be a local maximum with enough
                # prominence to stand above local baseline noise.
                min_prominence = max(
                    intensity_threshold * 0.05,
                    (
                        np.median(int_arr[int_arr > 0]) * 0.1
                        if np.any(int_arr > 0)
                        else 0.0
                    ),
                )
                peak_indices, _ = scipy_signal.find_peaks(
                    dense,
                    height=intensity_threshold,
                    prominence=min_prominence,
                )
                peak_mz = (peak_indices + mz_min).astype(float)
                peak_int = dense[peak_indices]
            else:
                # Fallback for single-point scans or unexpectedly large spans
                mask = int_arr >= intensity_threshold
                peak_mz = mz_arr[mask].astype(float)
                peak_int = int_arr[mask]

            if len(peak_mz) == 0:
                continue

            # Keep only the top-N by intensity
            order = np.argsort(peak_int)[::-1][: self.max_peaks_per_scan]

            for pi in order:
                mz_value = float(peak_mz[pi])
                intensity = float(peak_int[pi])
                peak = {
                    "mz": mz_value,
                    "intensity": intensity,
                    "normalized_intensity": intensity / max_intensity_overall,
                    "retention_time": retention_time,
                    "scan_index": scan_idx,
                    "peak_width_scans": self._estimate_peak_width(
                        int(mz_value), scan_idx, processed_spectra_dfs
                    ),
                }
                peaks.append(peak)

        return peaks

    def _cluster_and_consolidate_peaks(self, raw_peaks):
        if not raw_peaks:
            return raw_peaks

        # Extract m/z values for clustering
        mz_values = np.array([peak["mz"] for peak in raw_peaks]).reshape(-1, 1)

        # Calculate clustering tolerance
        if self.use_ppm_tolerance:
            avg_mz = np.mean(mz_values.flatten())
            epsilon = avg_mz * self.mz_tolerance_ppm / 1e6
        else:
            epsilon = self.mz_tolerance_da

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=epsilon, min_samples=1)
        cluster_labels = clustering.fit_predict(mz_values)

        # Group peaks by cluster
        peak_clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in peak_clusters:
                peak_clusters[label] = []
            peak_clusters[label].append(raw_peaks[i])

        # Consolidate each cluster into a single representative peak
        consolidated_peaks = []
        for cluster_peaks in peak_clusters.values():
            consolidated_peak = self._consolidate_peak_cluster(cluster_peaks)
            consolidated_peaks.append(consolidated_peak)

        return consolidated_peaks

    def _consolidate_peak_cluster(self, cluster_peaks):
        if len(cluster_peaks) == 1:
            return cluster_peaks[0]

        # Extract properties from all peaks in cluster
        mz_values = [p["mz"] for p in cluster_peaks]
        intensities = [p["intensity"] for p in cluster_peaks]
        retention_times = [p["retention_time"] for p in cluster_peaks]
        scan_indices = [p["scan_index"] for p in cluster_peaks]
        peak_widths = [p.get("peak_width_scans", 1) for p in cluster_peaks]

        # Consolidate based on method
        if self.consolidation_method == "weighted_average":
            # Weight by intensity
            weights = np.array(intensities)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(intensities)) / len(intensities)

            consolidated_mz = np.average(mz_values, weights=weights)
            consolidated_intensity = np.sum(intensities)  # Sum all intensities

        elif self.consolidation_method == "max_intensity":
            # Use peak with maximum intensity as base
            max_idx = np.argmax(intensities)
            consolidated_mz = mz_values[max_idx]
            consolidated_intensity = np.sum(intensities)

        elif self.consolidation_method == "median":
            # Use median values
            consolidated_mz = np.median(mz_values)
            consolidated_intensity = np.sum(intensities)

        else:
            # Default to weighted average
            weights = np.array(intensities)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(intensities)) / len(intensities)
            consolidated_mz = np.average(mz_values, weights=weights)
            consolidated_intensity = np.sum(intensities)

        # Use the earliest scan as start time and span as peak width so the
        # MIDI note begins when the chromatographic peak first appears and
        # lasts for its actual elution extent.
        min_scan = int(min(scan_indices))
        max_scan = int(max(scan_indices))
        start_retention_time = float(min(retention_times))
        chrom_width_scans = max(
            max_scan - min_scan + 1, int(np.mean(peak_widths))
        )

        # Create consolidated peak
        consolidated_peak = {
            "mz": float(consolidated_mz),
            "intensity": float(consolidated_intensity),
            "normalized_intensity": float(
                consolidated_intensity / max(intensities)
            ),
            "retention_time": start_retention_time,
            "scan_index": min_scan,
            "peak_width_scans": chrom_width_scans,
            "cluster_size": len(cluster_peaks),
            "mz_range": (float(min(mz_values)), float(max(mz_values))),
            "intensity_range": (
                float(min(intensities)),
                float(max(intensities)),
            ),
        }

        return consolidated_peak

    def _estimate_peak_width(
        self, mz_value, center_scan, processed_spectra_dfs
    ):
        width = 1

        # Look backward
        for i in range(center_scan - 1, max(0, center_scan - 10), -1):
            if (
                i < len(processed_spectra_dfs)
                and not processed_spectra_dfs[i].empty
                and mz_value in processed_spectra_dfs[i].index
            ):
                width += 1
            else:
                break

        # Look forward
        for i in range(
            center_scan + 1, min(len(processed_spectra_dfs), center_scan + 10)
        ):
            if (
                i < len(processed_spectra_dfs)
                and not processed_spectra_dfs[i].empty
                and mz_value in processed_spectra_dfs[i].index
            ):
                width += 1
            else:
                break

        return max(width, self.min_peak_width_scans)


class MSSonifierMidi:
    """
    MIDI sonifier with musical quantization and temporal alignment.
    """

    def __init__(
        self,
        filepath: str,
        config=None,
        sample_rate: int = 44100,
        enable_mz_clustering: bool = False,
        mz_tolerance_ppm: float = 20.0,
        mz_tolerance_da: float = 0.01,
        use_ppm_tolerance: bool = True,
        mz_consolidation_method: str = "weighted_average",
    ):
        """
        Initialize MIDI sonifier with optional peak clustering.

        Args:
            filepath: Path to mzML file
            config: MidiConfig object for configuration
            sample_rate: Audio sample rate for timing calculations
            enable_mz_clustering: Whether to enable m/z peak clustering by
            default
            mz_tolerance_ppm: Parts per million tolerance for peak clustering
            mz_tolerance_da: Absolute tolerance in Daltons for peak clustering
            use_ppm_tolerance: Use ppm (True) or absolute (False) tolerance
            mz_consolidation_method: How to consolidate clustered peaks
        """
        self.filepath = filepath
        self.config = config or MidiConfig()
        self.sample_rate = sample_rate

        # Initialize enhanced peak detector with clustering parameters
        self.peak_detector = MSPeakDetector(
            mz_tolerance_ppm=mz_tolerance_ppm,
            mz_tolerance_da=mz_tolerance_da,
            use_ppm_tolerance=use_ppm_tolerance,
            consolidation_method=mz_consolidation_method,
        )
        self.enable_mz_clustering = enable_mz_clustering

        self.note_quantizer = None
        self.metrical_quantizer = None

        # Data storage
        self.processed_spectra_dfs = None
        self.max_intensity_overall = 0.0
        self.min_mz_overall = 0.0
        self.max_mz_overall = 0.0
        self.detected_peaks = []
        self.quantized_notes = []

        print(f"MSSonifierMidi initialized for: {filepath}")
        if enable_mz_clustering:
            tolerance_str = (
                f"{mz_tolerance_ppm} ppm"
                if use_ppm_tolerance
                else f"{mz_tolerance_da} Da"
            )
            print(f"m/z peak clustering enabled: {tolerance_str} tolerance")

    def load_and_analyze_data(
        self,
        total_duration_seconds: float = 60.0,
        ms_level: int = 1,
        rt_range=None,
        mobility_range=None,
        cache: bool = True,
    ):
        """
        Load MS data and analyze for MIDI generation.

        Streams and bins the file (see :func:`ms_music.io.load_spectra`), so
        large runs fit in memory.

        Args:
            total_duration_seconds: Target duration for the MIDI file
            ms_level: MS level to use (1 or 2)
            rt_range: Optional (start, end) retention time window, minutes
            mobility_range: Optional (low, high) ion mobility window
            cache: Reuse a binned copy from an earlier load
        """
        run = io.load_spectra(
            self.filepath,
            ms_level=ms_level,
            rt_range=rt_range,
            mobility_range=mobility_range,
            cache=cache,
        )
        if run is None:
            raise ValueError("No spectra found in the mzML file")
        self.load_processed(
            run.spectra,
            run.max_intensity,
            run.min_mz,
            run.max_mz,
            total_duration_seconds,
        )

    def load_processed(
        self,
        processed_spectra_dfs,
        max_intensity_overall,
        min_mz_overall,
        max_mz_overall,
        total_duration_seconds: float = 60.0,
    ):
        """
        Use spectra that are already loaded and binned (e.g. an
        ``MSSonifier``'s ``processed_spectra_dfs``) instead of reading the
        file again.
        """
        if not processed_spectra_dfs:
            raise ValueError("No processable spectra found")
        self.processed_spectra_dfs = processed_spectra_dfs
        self.max_intensity_overall = float(max_intensity_overall)
        self.min_mz_overall = float(min_mz_overall)
        self.max_mz_overall = float(max_mz_overall)
        self.total_duration_seconds = total_duration_seconds

    def setup_musical_system(
        self,
        scale: Optional[str] = None,
        root_note: Optional[str] = None,
        tempo: Optional[int] = None,
        meter: Optional[MusicMeter] = None,
        freq_range: Tuple[float, float] = (261.63, 2093.00),
        edo_divisions: Optional[int] = None,
        use_just_intonation: Optional[bool] = None,
        quantization_mode: Optional[QuantizationMode] = None,
    ):
        """
        Set up the musical quantization system.

        Args:
            scale: Musical scale to use (overrides config)
            root_note: Root note of the scale (overrides config)
            tempo: Tempo in BPM (overrides config)
            meter: Musical meter/time signature (overrides config)
            freq_range: Frequency range for note mapping
            edo_divisions: Equal divisions of octave (overrides config)
            use_just_intonation: Use just intonation (overrides config)
            quantization_mode: How strictly to quantize timing
                    (overrides config)
        """
        # Use provided parameters or fall back to config
        scale = scale or self.config.scale
        root_note = root_note or self.config.root_note
        tempo = tempo or self.config.tempo
        meter = meter or self.config.meter
        edo_divisions = edo_divisions or self.config.edo_divisions
        use_just_intonation = (
            use_just_intonation
            if use_just_intonation is not None
            else self.config.use_just_intonation
        )
        quantization_mode = quantization_mode or self.config.quantization_mode

        # Initialize metrical quantizer
        self.metrical_quantizer = MetricalQuantizer(
            meter=meter,
            tempo=tempo,
            ticks_per_beat=480,  # Standard MIDI resolution
            quantization_mode=quantization_mode,
        )

        # Initialize note quantizer
        self.note_quantizer = MusicalNoteQuantizer(
            scale=scale,
            root_note=root_note,
            freq_range=freq_range,
            metrical_quantizer=self.metrical_quantizer,
            edo_divisions=edo_divisions,
            use_just_intonation=use_just_intonation,
        )

    def detect_and_quantize_peaks(
        self,
        intensity_threshold_percentile: float = 90.0,
        max_peaks_per_scan: int = 1000,
        frequency_mapping: str = "inverse_log",
        apply_mz_clustering: bool = None,
        mz_tolerance_ppm: float = None,
        mz_tolerance_da: float = None,
        use_ppm_tolerance: bool = None,
        mz_consolidation_method: str = None,
        apply_frequency_deduplication: bool = False,
        frequency_tolerance_hz: float = 5.0,
    ):
        """
        Enhanced peak detection and quantization with optional m/z clustering.

        Args:
            intensity_threshold_percentile: Minimum intensity percentile for
            peaks
            max_peaks_per_scan: Maximum peaks per scan to process
            frequency_mapping: Method for mapping m/z to frequency
            apply_mz_clustering: Override instance setting for m/z clustering
            mz_tolerance_ppm: Override ppm tolerance
            mz_tolerance_da: Override absolute tolerance
            use_ppm_tolerance: Override tolerance type
            mz_consolidation_method: Override consolidation method
            apply_frequency_deduplication: Apply frequency-based deduplication
            after quantization
            frequency_tolerance_hz: Frequency tolerance for post-quantization
            deduplication
        """
        if not self.processed_spectra_dfs:
            raise ValueError("Must load data first")

        if not self.note_quantizer or not self.metrical_quantizer:
            raise ValueError("Must setup musical system first")

        # Configure peak detector with current or override parameters
        self.peak_detector.intensity_threshold_percentile = (
            intensity_threshold_percentile
        )
        self.peak_detector.max_peaks_per_scan = max_peaks_per_scan

        # Use override parameters or instance defaults
        if mz_tolerance_ppm is not None:
            self.peak_detector.mz_tolerance_ppm = mz_tolerance_ppm
        if mz_tolerance_da is not None:
            self.peak_detector.mz_tolerance_da = mz_tolerance_da
        if use_ppm_tolerance is not None:
            self.peak_detector.use_ppm_tolerance = use_ppm_tolerance
        if mz_consolidation_method is not None:
            self.peak_detector.consolidation_method = mz_consolidation_method

        # Determine whether to apply clustering
        should_cluster = (
            apply_mz_clustering
            if apply_mz_clustering is not None
            else self.enable_mz_clustering
        )

        # Detect peaks with optional clustering
        self.detected_peaks = self.peak_detector._detect_peaks(
            self.processed_spectra_dfs,
            self.max_intensity_overall,
            apply_clustering=should_cluster,
        )

        # Convert peaks to quantized musical notes
        self.quantized_notes = []
        total_ticks = int(
            self.total_duration_seconds
            * (self.metrical_quantizer.tempo / 60.0)
            * self.metrical_quantizer.ticks_per_beat
        )

        for peak in tqdm(self.detected_peaks, desc="Quantizing notes"):
            # Map m/z to continuous frequency
            continuous_freq = self._map_mz_to_frequency(
                peak["mz"],
                frequency_mapping,
                self.min_mz_overall,
                self.max_mz_overall,
            )

            # Convert retention time to MIDI ticks
            time_ticks = int(peak["retention_time"] * total_ticks)

            # Estimate duration based on peak width and intensity
            base_duration_ticks = int(
                peak["peak_width_scans"]
                / len(self.processed_spectra_dfs)
                * total_ticks
            )

            # Quantize frequency, timing, and duration
            note_info = self.note_quantizer.quantize_with_meter(
                frequency=continuous_freq,
                time_ticks=time_ticks,
                duration_ticks=base_duration_ticks,
                intensity=peak["normalized_intensity"],
                peak_width_seconds=peak["peak_width_scans"]
                * self.total_duration_seconds
                / len(self.processed_spectra_dfs),
            )

            # Convert to MIDI note
            midi_note = MidiNote(
                start_time=note_info["quantized_time"],
                duration=note_info["quantized_duration"],
                midi_number=int(
                    np.clip(note_info.get("midi_note", 60), 0, 127)
                ),
                velocity=int(
                    np.clip(20 + peak["normalized_intensity"] * 107, 20, 127)
                ),
                channel=0,
                original_mz=peak["mz"],
                original_intensity=peak["intensity"],
                retention_time=peak["retention_time"],
                note_name=note_info.get("note", "C"),
                frequency=note_info.get("frequency", continuous_freq),
            )

            self.quantized_notes.append(midi_note)

        # Apply frequency-based deduplication if requested
        if apply_frequency_deduplication and self.quantized_notes:
            original_count = len(self.quantized_notes)
            self.quantized_notes = self._deduplicate_by_frequency(
                self.quantized_notes, frequency_tolerance_hz
            )
            print(
                f"Frequency deduplication: {original_count} "
                f"-> {len(self.quantized_notes)} notes"
            )

        # Sort notes by start time
        self.quantized_notes.sort(key=lambda n: n.start_time)

    def _deduplicate_by_frequency(self, notes, frequency_tolerance_hz=5.0):
        """
        Remove notes with very similar frequencies after quantization.

        Args:
            notes: List of MidiNote objects
            frequency_tolerance_hz: Frequency tolerance in Hz

        Returns:
            Deduplicated list of notes
        """
        if not notes:
            return notes

        # Sort by frequency
        sorted_notes = sorted(notes, key=lambda n: n.frequency)

        deduplicated_notes = []
        current_group = [sorted_notes[0]]

        for note in sorted_notes[1:]:
            current_freq = note.frequency
            group_freq = current_group[0].frequency

            if abs(current_freq - group_freq) <= frequency_tolerance_hz:
                # Add to current group
                current_group.append(note)
            else:
                # Process current group and start new group
                consolidated_note = self._consolidate_note_group(current_group)
                deduplicated_notes.append(consolidated_note)
                current_group = [note]

        # Don't forget the last group
        if current_group:
            consolidated_note = self._consolidate_note_group(current_group)
            deduplicated_notes.append(consolidated_note)

        return deduplicated_notes

    def _consolidate_note_group(self, note_group):
        """Consolidate a group of similar frequency notes into one."""
        if len(note_group) == 1:
            return note_group[0]

        # Use note with highest velocity as base
        base_note = max(note_group, key=lambda n: n.velocity)

        # Combine properties from all notes in group
        total_velocity = min(127, sum(n.velocity for n in note_group))
        avg_duration = int(np.mean([n.duration for n in note_group]))
        total_intensity = sum(n.original_intensity for n in note_group)

        # Update the base note
        base_note.velocity = total_velocity
        base_note.duration = avg_duration
        base_note.original_intensity = total_intensity

        return base_note

    def _map_mz_to_frequency(
        self,
        mz_value: float,
        mapping_method: str,
        min_mz_overall: float,
        max_mz_overall: float,
    ) -> float:
        """Map m/z value to frequency using specified method."""
        if mapping_method == "inverse_log":
            return mz_to_frequency_inverse_log(
                mz_value,
                self.note_quantizer.freq_range,
                min_mz_overall,
                max_mz_overall,
            )
        elif mapping_method == "power_law":
            return mz_to_frequency_power_law(
                mz_value,
                self.note_quantizer.freq_range,
                min_mz_overall,
                max_mz_overall,
            )
        elif mapping_method == "musical_octaves":
            return mz_to_frequency_musical_octaves(
                mz_value,
                min_mz_overall,
                max_mz_overall,
                base_freq=440.0,
                num_octaves=4,
            )
        elif mapping_method == "chromatic":
            return mz_to_frequency_chromatic(
                mz_value,
                min_mz_overall,
                max_mz_overall,
                base_freq=261.63,
                num_semitones=48,
            )
        elif mapping_method == "linear":
            return mz_to_frequency_linear(
                mz_value,
                self.note_quantizer.freq_range,
                min_mz_overall,
                max_mz_overall,
            )
        else:
            raise ValueError(
                f"Unknown frequency_mapping '{mapping_method}'. Expected one "
                "of: inverse_log, power_law, musical_octaves, "
                "chromatic, linear."
            )

    def generate_midi_file(
        self,
        output_path: str,
        track_name: str = "MS Sonification",
        instrument: int = 1,  # Acoustic Piano
        max_simultaneous_notes: int = 16,
        export_note_data: bool = False,
        num_voices: int = 1,
        separate_files: bool = False,
    ) -> bool:
        """
        Generate the final MIDI file.

        Args:
            output_path: Path for output MIDI file
            track_name: Name for the MIDI track
            instrument: MIDI program number (1-128)
            max_simultaneous_notes: Maximum simultaneous
                    notes to prevent overload
            export_note_data: Whether to export note data to CSV
            num_voices: Number of voices to separate into (1-4)
            separate_files: If True, save each voice as separate MIDI file

        Returns:
            True if successful, False otherwise
        """
        if not self.quantized_notes:
            print(
                "Error: No quantized notes available. Run"
                " detect_and_quantize_peaks() first."
            )
            return False

        # Separate notes into voices if requested
        if num_voices > 1:
            voice_notes = self._separate_voices(num_voices)
        else:
            voice_notes = [self.quantized_notes]

        # Save as separate files if requested
        if separate_files and num_voices > 1:
            base_path = output_path.rsplit(".mid", 1)[0]

            for voice_idx, notes in enumerate(voice_notes):
                if not notes:  # Skip empty voices
                    continue

                voice_path = f"{base_path}_voice{voice_idx + 1}.mid"
                mid = MidiFile(
                    ticks_per_beat=self.metrical_quantizer.ticks_per_beat
                )
                track = MidiTrack()
                mid.tracks.append(track)

                voice_name = f"{track_name} V{voice_idx + 1}"
                track.append(
                    MetaMessage("track_name", name=voice_name, time=0)
                )
                track.append(
                    MetaMessage(
                        "set_tempo",
                        tempo=mido.bpm2tempo(self.metrical_quantizer.tempo),
                        time=0,
                    )
                )
                track.append(
                    Message(
                        "program_change",
                        program=instrument - 1,
                        channel=0,
                        time=0,
                    )
                )

                self._write_track_notes(
                    track, notes, max_simultaneous_notes, 0
                )
                mid.save(voice_path)

            print(f"Saved {len(voice_notes)} voice files")
            return True

        # Save as single file with multiple tracks
        mid = MidiFile(ticks_per_beat=self.metrical_quantizer.ticks_per_beat)

        # Create track for each voice
        for voice_idx, notes in enumerate(voice_notes):
            track = MidiTrack()
            mid.tracks.append(track)

            # Add track metadata
            voice_name = (
                f"{track_name} V{voice_idx + 1}"
                if num_voices > 1
                else track_name
            )
            track.append(MetaMessage("track_name", name=voice_name, time=0))
            track.append(
                MetaMessage(
                    "set_tempo",
                    tempo=mido.bpm2tempo(self.metrical_quantizer.tempo),
                    time=0,
                )
            )

            # Add program change
            track.append(
                Message(
                    "program_change",
                    program=instrument - 1,
                    channel=voice_idx,
                    time=0,
                )
            )

            self._write_track_notes(
                track, notes, max_simultaneous_notes, voice_idx
            )

        # Save MIDI file
        mid.save(output_path)

        # Export note data if requested
        if export_note_data:
            csv_path = output_path.replace(".mid", "_notes.csv")
            self.export_note_data(csv_path)

        return True

    def _separate_voices(self, num_voices: int):
        """Separate notes into voices by pitch range."""
        if not self.quantized_notes:
            return []

        # Get pitch range
        pitches = [n.midi_number for n in self.quantized_notes]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        pitch_range = max_pitch - min_pitch

        # Assign notes to voices by pitch
        voices = [[] for _ in range(num_voices)]
        for note in self.quantized_notes:
            # Determine voice based on pitch
            voice_idx = min(
                int((note.midi_number - min_pitch) / pitch_range * num_voices),
                num_voices - 1,
            )
            voices[voice_idx].append(note)

        return voices

    def _write_track_notes(
        self, track, notes, max_simultaneous_notes, channel
    ):
        """Write notes to a MIDI track."""
        # Group notes by start time and limit simultaneous notes
        notes_by_time = {}
        for note in notes:
            if note.start_time not in notes_by_time:
                notes_by_time[note.start_time] = []
            notes_by_time[note.start_time].append(note)

        # Limit simultaneous notes
        for start_time in notes_by_time:
            notes_at_time = notes_by_time[start_time]
            if len(notes_at_time) > max_simultaneous_notes:
                notes_at_time.sort(key=lambda n: n.velocity, reverse=True)
                notes_by_time[start_time] = notes_at_time[
                    :max_simultaneous_notes
                ]

        # Create event list
        events = []
        for start_time, note_list in notes_by_time.items():
            for note in note_list:
                events.append(
                    {
                        "time": note.start_time,
                        "type": "note_on",
                        "note": note.midi_number,
                        "velocity": note.velocity,
                        "channel": channel,
                    }
                )
                events.append(
                    {
                        "time": note.start_time + note.duration,
                        "type": "note_off",
                        "note": note.midi_number,
                        "velocity": 0,
                        "channel": channel,
                    }
                )

        # Sort events by time
        events.sort(key=lambda e: (e["time"], e["type"] == "note_off"))

        # Convert to MIDI messages
        current_time = 0
        for event in events:
            delta_time = max(0, event["time"] - current_time)

            if event["type"] == "note_on":
                message = Message(
                    "note_on",
                    channel=event["channel"],
                    note=event["note"],
                    velocity=event["velocity"],
                    time=delta_time,
                )
            else:
                message = Message(
                    "note_off",
                    channel=event["channel"],
                    note=event["note"],
                    velocity=0,
                    time=delta_time,
                )

            track.append(message)
            current_time = event["time"]

        # Add end of track
        track.append(MetaMessage("end_of_track", time=0))

    def get_analysis_report(self) -> Dict[str, Any]:
        """Get detailed analysis report of the MIDI generation process."""
        if not self.quantized_notes:
            return {"error": "No notes generated"}

        notes = self.quantized_notes

        # Basic statistics
        note_count = len(notes)
        unique_pitches = len(set(note.midi_number for note in notes))
        velocity_range = (
            min(note.velocity for note in notes),
            max(note.velocity for note in notes),
        )
        duration_range = (
            min(note.duration for note in notes),
            max(note.duration for note in notes),
        )

        # Timing statistics
        total_duration_ticks = max(
            note.start_time + note.duration for note in notes
        )
        total_duration_seconds = (
            total_duration_ticks
            / self.metrical_quantizer.ticks_per_beat
            * 60
            / self.metrical_quantizer.tempo
        )

        # Pitch distribution
        pitch_counts = {}
        for note in notes:
            pitch = note.midi_number
            pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1

        most_common_pitch = (
            max(pitch_counts.items(), key=lambda x: x[1])
            if pitch_counts
            else (60, 0)
        )

        # Intensity correlation
        original_intensities = [note.original_intensity for note in notes]
        velocities = [note.velocity for note in notes]

        if len(original_intensities) > 1:
            intensity_velocity_corr = np.corrcoef(
                original_intensities, velocities
            )[0, 1]
        else:
            intensity_velocity_corr = 0.0

        return {
            "note_count": note_count,
            "unique_pitches": unique_pitches,
            "velocity_range": velocity_range,
            "duration_range_ticks": duration_range,
            "total_duration_seconds": total_duration_seconds,
            "most_common_pitch": most_common_pitch,
            "intensity_velocity_correlation": intensity_velocity_corr,
            "scale_info": (
                self.note_quantizer.get_scale_info()
                if self.note_quantizer
                else None
            ),
            "meter": (
                self.metrical_quantizer.meter.name
                if self.metrical_quantizer
                else None
            ),
            "tempo": (
                self.metrical_quantizer.tempo
                if self.metrical_quantizer
                else None
            ),
        }

    def export_note_data(self, output_path: str):
        """Export note data to CSV for analysis."""
        if not self.quantized_notes:
            print("No note data to export")
            return

        note_data = []
        for note in self.quantized_notes:
            note_data.append(
                {
                    "start_time_ticks": note.start_time,
                    "duration_ticks": note.duration,
                    "midi_number": note.midi_number,
                    "velocity": note.velocity,
                    "note_name": note.note_name,
                    "frequency_hz": note.frequency,
                    "original_mz": note.original_mz,
                    "original_intensity": note.original_intensity,
                    "retention_time": note.retention_time,
                    "start_time_seconds": note.start_time
                    / self.metrical_quantizer.ticks_per_beat
                    * 60
                    / self.metrical_quantizer.tempo,
                    "duration_seconds": note.duration
                    / self.metrical_quantizer.ticks_per_beat
                    * 60
                    / self.metrical_quantizer.tempo,
                }
            )

        df = pd.DataFrame(note_data)
        df.to_csv(output_path, index=False)
        print(f"Note data exported to: {output_path}")
