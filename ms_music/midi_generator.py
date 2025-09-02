import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from .musical_quantization import (
    MusicalNoteQuantizer,
    MetricalQuantizer,
    MusicMeter,
    QuantizationMode,
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
        # Ensure meter is a MusicMeter enum
        if isinstance(self.meter, str):
            self.meter = getattr(
                MusicMeter,
                self.meter.upper().replace("/", "_"),
                MusicMeter.FOUR_FOUR,
            )
        if isinstance(self.quantization_mode, str):
            self.quantization_mode = getattr(
                QuantizationMode,
                self.quantization_mode.upper(),
                QuantizationMode.STRICT_GRID,
            )


class MSPeakDetector:
    """Detect and extract peaks from MS data with timing information."""

    def __init__(
        self,
        intensity_threshold_percentile: float = 90.0,
        min_peak_width_scans: int = 3,
        max_peaks_per_scan: int = 1000,
    ):
        """
        Initialize peak detector.

        Args:
            intensity_threshold_percentile: Minimum intensity percentile
                    for peak detection
            min_peak_width_scans: Minimum width in number of scans
            max_peaks_per_scan: Maximum peaks to extract per scan
        """
        self.intensity_threshold_percentile = intensity_threshold_percentile
        self.min_peak_width_scans = min_peak_width_scans
        self.max_peaks_per_scan = max_peaks_per_scan

    def detect_peaks(
        self,
        processed_spectra_dfs: List[pd.DataFrame],
        max_intensity_overall: float,
    ) -> List[Dict[str, Any]]:
        """
        Detect peaks across all scans with retention time information.

        Args:
            processed_spectra_dfs: List of processed spectra DataFrames
            max_intensity_overall: Maximum intensity for normalization

        Returns:
            List of peak dictionaries with timing and intensity info
        """
        peaks = []

        # Calculate intensity threshold
        all_intensities = []
        for scan_df in processed_spectra_dfs:
            if not scan_df.empty:
                all_intensities.extend(scan_df["intensities"].values)

        if not all_intensities:
            return peaks

        intensity_threshold = np.percentile(
            all_intensities, self.intensity_threshold_percentile
        )

        # Process each scan
        for scan_idx, scan_df in enumerate(
            tqdm(processed_spectra_dfs, desc="Detecting peaks")
        ):
            if scan_df.empty:
                continue

            # Calculate retention time (normalize across scans)
            retention_time = (
                scan_idx / len(processed_spectra_dfs)
                if len(processed_spectra_dfs) > 1
                else 0.0
            )

            # Get peaks above threshold
            above_threshold = scan_df[
                scan_df["intensities"] >= intensity_threshold
            ]

            # Sort by intensity and take top peaks
            top_peaks = above_threshold.nlargest(
                self.max_peaks_per_scan, "intensities"
            )

            # Convert to peak objects
            for mz_value, row in top_peaks.iterrows():
                intensity = row["intensities"]
                normalized_intensity = intensity / max_intensity_overall

                peak = {
                    "mz": float(mz_value),
                    "intensity": float(intensity),
                    "normalized_intensity": float(normalized_intensity),
                    "retention_time": retention_time,
                    "scan_index": scan_idx,
                    "peak_width_scans": self._estimate_peak_width(
                        mz_value, scan_idx, processed_spectra_dfs
                    ),
                }
                peaks.append(peak)
        return peaks

    def _estimate_peak_width(
        self,
        mz_value: float,
        center_scan: int,
        processed_spectra_dfs: List[pd.DataFrame],
    ) -> int:
        """Estimate peak width in number of scans."""
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
        config: Optional[MidiConfig] = None,
        sample_rate: int = 44100,
    ):
        """
        Initialize MIDI sonifier.

        Args:
            filepath: Path to mzML file
            config: MidiConfig object for configuration
            sample_rate: Audio sample rate for timing calculations
        """
        self.filepath = filepath
        self.config = config or MidiConfig()
        self.sample_rate = sample_rate

        # Initialize components
        self.peak_detector = MSPeakDetector()
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

    def load_and_analyze_data(self, total_duration_seconds: float = 60.0):
        """
        Load MS data and analyze for MIDI generation.

        Args:
            total_duration_seconds: Target duration for the MIDI file
        """
        # Load raw spectra
        raw_spectra = io.load_mzml_data(self.filepath, ms_level=1)
        if not raw_spectra:
            raise ValueError("No spectra found in the mzML file")

        # Preprocess spectra
        (
            self.processed_spectra_dfs,
            self.max_intensity_overall,
            self.min_mz_overall,
            self.max_mz_overall,
        ) = io.preprocess_spectra(raw_spectra)

        if not self.processed_spectra_dfs:
            raise ValueError("No processable spectra found")

        # Store timing information
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
    ):
        """
        Detect peaks and quantize them to musical notes with timing.

        Args:
            intensity_threshold_percentile: Minimum intensity
                    percentile for peaks
            max_peaks_per_scan: Maximum peaks per scan to process
            frequency_mapping: Method for mapping m/z to frequency
        """
        if not self.processed_spectra_dfs:
            raise ValueError("Must load data first")

        if not self.note_quantizer or not self.metrical_quantizer:
            raise ValueError("Must setup musical system first")

        # Configure peak detector
        self.peak_detector = MSPeakDetector(
            intensity_threshold_percentile=intensity_threshold_percentile,
            max_peaks_per_scan=max_peaks_per_scan,
        )

        # Detect peaks
        self.detected_peaks = self.peak_detector.detect_peaks(
            self.processed_spectra_dfs, self.max_intensity_overall
        )

        if not self.detected_peaks:
            print("Warning: No peaks detected for MIDI generation")
            return

        # Convert peaks to quantized musical notes
        self.quantized_notes = []

        # Calculate total duration in MIDI ticks
        total_ticks = int(
            self.total_duration_seconds
            * (self.metrical_quantizer.tempo / 60.0)
            * self.metrical_quantizer.ticks_per_beat
        )

        for peak in tqdm(self.detected_peaks, desc="Quantizing notes"):
            # Map m/z to continuous frequency
            continuous_freq = self._map_mz_to_frequency(
                peak["mz"], frequency_mapping,
                self.min_mz_overall, self.max_mz_overall,
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
                    np.clip(peak["normalized_intensity"] * 100 + 20, 20, 127)
                ),
                channel=0,
                original_mz=peak["mz"],
                original_intensity=peak["intensity"],
                retention_time=peak["retention_time"],
                note_name=note_info.get("note", "C"),
                frequency=note_info.get("frequency", continuous_freq),
            )

            self.quantized_notes.append(midi_note)

        # Sort notes by start time
        self.quantized_notes.sort(key=lambda n: n.start_time)

    def _map_mz_to_frequency(
        self, mz_value: float, mapping_method: str,
        min_mz_overall: float, max_mz_overall: float
    ) -> float:
        """Map m/z value to frequency using specified method."""
        freq_min, freq_max = self.note_quantizer.freq_range

        if mapping_method == "inverse_log":
            return mz_to_frequency_inverse_log(
                mz_value,
                self.note_quantizer.freq_range,
                min_mz_overall,
                max_mz_overall
            )
        elif mapping_method == "power_law":
            return mz_to_frequency_power_law(
                mz_value,
                self.note_quantizer.freq_range,
                min_mz_overall,
                max_mz_overall
            )
        elif mapping_method == "musical_octaves":
            return mz_to_frequency_musical_octaves(
                mz_value, min_mz_overall,
                max_mz_overall,
                base_freq=440.0, num_octaves=4
            )
        elif mapping_method == "chromatic":
            return mz_to_frequency_chromatic(
                mz_value,
                min_mz_overall,
                max_mz_overall,
                base_freq=261.63, num_semitones=48
            )
        elif mapping_method == "linear":
            return mz_to_frequency_linear(
                mz_value, self.note_quantizer.freq_range,
                min_mz_overall,
                max_mz_overall
            )

    def generate_midi_file(
        self,
        output_path: str,
        track_name: str = "MS Sonification",
        instrument: int = 1,  # Acoustic Piano
        max_simultaneous_notes: int = 16,
        export_note_data: bool = False,
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

        Returns:
            True if successful, False otherwise
        """
        if not self.quantized_notes:
            print(
                "Error: No quantized notes available. Run"
                " detect_and_quantize_peaks() first."
            )
            return False

        # Create MIDI file
        mid = MidiFile(ticks_per_beat=self.metrical_quantizer.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        # Add track metadata
        track.append(MetaMessage("track_name", name=track_name, time=0))
        track.append(
            MetaMessage(
                "set_tempo",
                tempo=mido.bpm2tempo(self.metrical_quantizer.tempo),
                time=0,
            )
        )

        # Add program change
        track.append(Message("program_change", program=instrument - 1, time=0))

        # Group notes by start time and limit simultaneous notes
        notes_by_time = {}
        for note in self.quantized_notes:
            if note.start_time not in notes_by_time:
                notes_by_time[note.start_time] = []
            notes_by_time[note.start_time].append(note)

        # Limit simultaneous notes
        for start_time in notes_by_time:
            notes_at_time = notes_by_time[start_time]
            if len(notes_at_time) > max_simultaneous_notes:
                # Keep the highest velocity notes
                notes_at_time.sort(key=lambda n: n.velocity, reverse=True)
                notes_by_time[start_time] = notes_at_time[
                    :max_simultaneous_notes
                ]

        # Create event list (start and end events)
        events = []

        for start_time, notes in notes_by_time.items():
            for note in notes:
                events.append(
                    {
                        "time": note.start_time,
                        "type": "note_on",
                        "note": note.midi_number,
                        "velocity": note.velocity,
                        "channel": note.channel,
                    }
                )
                events.append(
                    {
                        "time": note.start_time + note.duration,
                        "type": "note_off",
                        "note": note.midi_number,
                        "velocity": 0,
                        "channel": note.channel,
                    }
                )

        # Sort events by time
        events.sort(
            key=lambda e: (e["time"], e["type"] == "note_off")
        )  # note_off after note_on at same time

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
            else:  # note_off
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

        # Save MIDI file
        mid.save(output_path)

        # Export note data if requested
        if export_note_data:
            csv_path = output_path.replace(".mid", "_notes.csv")
            self.export_note_data(csv_path)

        return True

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
