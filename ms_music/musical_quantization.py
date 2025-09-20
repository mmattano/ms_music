import numpy as np
import math
from enum import Enum
from dataclasses import dataclass
from typing import List


class MusicMeter(Enum):
    """Musical time signatures/meters."""

    FOUR_FOUR = (4, 4)  # 4/4 time
    THREE_FOUR = (3, 4)  # 3/4 time (waltz)
    TWO_FOUR = (2, 4)  # 2/4 time (march)
    SIX_EIGHT = (6, 8)  # 6/8 time (compound duple)
    NINE_EIGHT = (9, 8)  # 9/8 time (compound triple)
    TWELVE_EIGHT = (12, 8)  # 12/8 time (compound quadruple)
    FIVE_FOUR = (5, 4)  # 5/4 time (irregular)
    SEVEN_EIGHT = (7, 8)  # 7/8 time (irregular)

    def __init__(self, beats_per_measure, note_value):
        self.beats_per_measure = beats_per_measure
        self.note_value = note_value  # 4 = quarter note, 8 = eighth note

    @property
    def is_compound(self):
        """Check if this is a compound meter (divisible by 3)."""
        return self.beats_per_measure % 3 == 0 and self.beats_per_measure > 3


class QuantizationMode(Enum):
    """Different quantization approaches for timing."""

    STRICT_GRID = "strict_grid"  # Force everything to grid
    SWING = "swing"  # Add swing feel (uneven eighths)
    HUMANIZED = "humanized"  # Small random variations
    ADAPTIVE = "adaptive"  # Adapt grid to data characteristics


@dataclass
class NoteLength:
    """Represents musical note lengths in terms of beats."""

    name: str
    beats: float
    ticks_per_beat: int = 480

    @property
    def ticks(self) -> int:
        return int(self.beats * self.ticks_per_beat)

    @classmethod
    def whole_note(cls, ticks_per_beat=480):
        return cls("whole", 4.0, ticks_per_beat)

    @classmethod
    def half_note(cls, ticks_per_beat=480):
        return cls("half", 2.0, ticks_per_beat)

    @classmethod
    def quarter_note(cls, ticks_per_beat=480):
        return cls("quarter", 1.0, ticks_per_beat)

    @classmethod
    def eighth_note(cls, ticks_per_beat=480):
        return cls("eighth", 0.5, ticks_per_beat)

    @classmethod
    def sixteenth_note(cls, ticks_per_beat=480):
        return cls("sixteenth", 0.25, ticks_per_beat)

    @classmethod
    def dotted_quarter(cls, ticks_per_beat=480):
        return cls("dotted_quarter", 1.5, ticks_per_beat)

    @classmethod
    def dotted_eighth(cls, ticks_per_beat=480):
        return cls("dotted_eighth", 0.75, ticks_per_beat)


class MetricalQuantizer:
    """
    Handles quantization of timing and note durations to musical meters.
    """

    def __init__(
        self,
        meter: MusicMeter = MusicMeter.FOUR_FOUR,
        tempo: int = 120,
        ticks_per_beat: int = 480,
        quantization_mode: QuantizationMode = QuantizationMode.STRICT_GRID,
        swing_ratio: float = 0.67,
    ):
        """
        Initialize metrical quantizer.

        Args:
            meter: Musical meter/time signature
            tempo: Beats per minute
            ticks_per_beat: MIDI resolution
            quantization_mode: How to quantize timing
            swing_ratio: For swing feel (0.5 = straight, 0.67 = swing)
        """
        self.meter = meter
        self.tempo = tempo
        self.ticks_per_beat = ticks_per_beat
        self.quantization_mode = quantization_mode
        self.swing_ratio = swing_ratio

        # Calculate measure length in ticks
        self.ticks_per_measure = self.ticks_per_beat * meter.beats_per_measure

        # Define available note lengths
        self.note_lengths = [
            NoteLength.whole_note(ticks_per_beat),
            NoteLength.half_note(ticks_per_beat),
            NoteLength.dotted_quarter(ticks_per_beat),
            NoteLength.quarter_note(ticks_per_beat),
            NoteLength.dotted_eighth(ticks_per_beat),
            NoteLength.eighth_note(ticks_per_beat),
            NoteLength.sixteenth_note(ticks_per_beat),
        ]

        # Create quantization grid
        self.grid_positions = self._create_quantization_grid()

    def _create_quantization_grid(self) -> List[int]:
        """Create quantization grid positions within a measure."""
        grid_positions = []

        if self.meter.is_compound:
            # Compound meters: subdivide into dotted quarters, then eighths
            dotted_quarter_ticks = int(self.ticks_per_beat * 1.5)
            for beat in range(self.meter.beats_per_measure // 3):
                beat_start = beat * dotted_quarter_ticks
                # Add dotted quarter and eighth subdivisions
                grid_positions.extend(
                    [
                        beat_start,
                        beat_start + self.ticks_per_beat // 2,  # First eighth
                        beat_start + self.ticks_per_beat,  # Second eighth
                    ]
                )
        else:
            # Simple meters: subdivide into quarters, then eighths, sixteenths
            for beat in range(self.meter.beats_per_measure):
                beat_start = beat * self.ticks_per_beat
                grid_positions.extend(
                    [
                        beat_start,  # Beat
                        beat_start
                        + self.ticks_per_beat // 4,  # First sixteenth
                        beat_start + self.ticks_per_beat // 2,  # Eighth
                        beat_start
                        + 3 * self.ticks_per_beat // 4,  # Third sixteenth
                    ]
                )

        return sorted(list(set(grid_positions)))

    def quantize_onset_time(
            self,
            time_ticks: int,
            measure_offset: int = 0) -> int:
        """
        Quantize an onset time to the nearest grid position.

        Args:
            time_ticks: Input time in MIDI ticks
            measure_offset: Offset within the current measure (usually 0)

        Returns:
            Quantized time in ticks
        """
        # Calculate which measure this time falls into
        measure_number = time_ticks // self.ticks_per_measure
        measure_start_ticks = measure_number * self.ticks_per_measure

        # Find position within the current measure
        measure_time = time_ticks - measure_start_ticks

        if self.quantization_mode == QuantizationMode.STRICT_GRID:
            # Find closest grid position within the measure
            closest_grid = min(
                self.grid_positions, key=lambda x: abs(x - measure_time)
            )
            quantized_time = measure_start_ticks + closest_grid

        elif self.quantization_mode == QuantizationMode.SWING:
            # Apply swing to eighth note positions
            closest_grid = min(
                self.grid_positions, key=lambda x: abs(x - measure_time)
            )
            quantized_measure_time = self._apply_swing(closest_grid)
            quantized_time = measure_start_ticks + quantized_measure_time

        elif self.quantization_mode == QuantizationMode.HUMANIZED:
            # Add small random variations
            closest_grid = min(
                self.grid_positions, key=lambda x: abs(x - measure_time)
            )
            humanization = np.random.randint(
                -self.ticks_per_beat // 32, self.ticks_per_beat // 32
            )
            quantized_time = measure_start_ticks + closest_grid + humanization

        else:  # ADAPTIVE
            # Use looser quantization for better musical flow
            closest_grid = min(
                self.grid_positions, key=lambda x: abs(x - measure_time)
            )
            if abs(measure_time - closest_grid) < self.ticks_per_beat // 8:
                quantized_time = measure_start_ticks + closest_grid
            else:
                # Allow some deviation if it's not close to a strong beat
                quantized_time = time_ticks

        return max(0, int(round(quantized_time)))

    def quantize_duration(
        self, duration_ticks: int, intensity: float = 0.5,
        peak_width_seconds: float = None
    ) -> int:
        """
        Quantize note duration based primarily on peak width.

        Args:
            duration_ticks: Original duration in ticks
            intensity: Normalized intensity (0-1)
            peak_width_seconds: Width of the original peak in seconds

        Returns:
            Quantized duration in ticks
        """
        if peak_width_seconds is not None and peak_width_seconds > 0:
            # Convert peak width directly to beats
            seconds_per_beat = 60.0 / self.tempo
            peak_beats = peak_width_seconds / seconds_per_beat

            # Find closest musical note length
            closest_length = min(
                self.note_lengths, key=lambda x: abs(x.beats - peak_beats)
            )

            return max(1, int(round(closest_length.ticks)))

        # Fallback: use original duration_ticks if no peak width
        if duration_ticks > 0:
            duration_beats = duration_ticks / self.ticks_per_beat
            closest_length = min(
                self.note_lengths, key=lambda x: abs(x.beats - duration_beats)
            )
            return max(1, int(round(closest_length.ticks)))

        # Last resort: use sixteenth note
        return max(1, int(self.ticks_per_beat * 0.25))

    def _apply_swing(self, time_ticks: int) -> int:
        """Apply swing feel to eighth note positions."""
        measure_pos = time_ticks % self.ticks_per_measure
        beat_pos = measure_pos % self.ticks_per_beat

        eighth_note_ticks = self.ticks_per_beat // 2

        if beat_pos == eighth_note_ticks:  # This is an eighth note position
            # Move the second eighth note later for swing feel
            swing_offset = int((self.swing_ratio - 0.5) * eighth_note_ticks)
            return time_ticks + swing_offset

        return time_ticks


class MusicalNoteQuantizer:
    """
    Note quantizer with support for musical scales, EDO systems, and
    experimental tunings.
    """

    # Note names for reference (12-EDO)
    CHROMATIC_NOTES = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]

    # Traditional scales (semitone intervals from root)
    SCALES = {
        "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "pentatonic_major": [0, 2, 4, 7, 9],
        "pentatonic_minor": [0, 3, 5, 7, 10],
        "blues": [0, 3, 5, 6, 7, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "whole_tone": [0, 2, 4, 6, 8, 10],
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "locrian": [0, 1, 3, 5, 6, 8, 10],
    }

    # EDO (Equal Division of Octave) scales - define as ratios
    # within the octave
    EDO_SCALES = {
        # Popular microtonal EDO systems
        "19_edo": list(range(19)),  # 19 equal divisions
        "24_edo": list(range(24)),  # Quarter-tone system
        "31_edo": list(range(31)),  # Popular for just intonation approximation
        "53_edo": list(range(53)),  # Excellent just intonation approximation
        # Subset scales from EDO systems
        "19_edo_diatonic": [
            0,
            3,
            6,
            8,
            11,
            14,
            17,
        ],  # Major-like scale in 19-EDO
        "24_edo_chromatic": [
            0,
            2,
            4,
            5,
            7,
            9,
            11,
            12,
            14,
            16,
            17,
            19,
            21,
            23,
        ],  # Extended chromatic
        "31_edo_major": [0, 5, 10, 13, 18, 23, 28],  # Just major in 31-EDO
        # International music scales (approximated in 24-EDO) would be
        # amazing to spend some time on this and expand it
        "arabic_maqam_hijaz": [0, 1, 4, 5, 7, 8, 11],  # Quarter-tone positions
        "turkish_makam": [0, 1, 4, 5, 7, 9, 10, 11],
        "indian_raga_bhairav": [0, 1, 4, 5, 7, 8, 11],
        # Experimental/avant-garde scales
        "17_edo": list(range(17)),
        "22_edo": list(range(22)),
        "41_edo": list(range(41)),
        # Just intonation ratios (converted to cents then EDO approximations)
        "just_major": [0, 4, 7, 10, 14, 17, 21],  # Approximation in 24-EDO
        "just_minor": [0, 4, 6, 10, 14, 16, 21],
    }

    # Frequency ratios for pure just intonation scales
    JUST_INTONATION_RATIOS = {
        "just_major": [
            1 / 1,
            9 / 8,
            5 / 4,
            4 / 3,
            3 / 2,
            5 / 3,
            15 / 8,
        ],  # C D E F G A B
        "just_minor": [
            1 / 1,
            9 / 8,
            6 / 5,
            4 / 3,
            3 / 2,
            8 / 5,
            9 / 5,
        ],  # Natural minor
        "pythagorean_major": [
            1 / 1,
            9 / 8,
            81 / 64,
            4 / 3,
            3 / 2,
            27 / 16,
            243 / 128,
        ],
        "quarter_comma_meantone": [
            1 / 1,
            (5 / 4) ** 0.5,
            5 / 4,
            4 / 3,
            3 / 2,
            (5 / 4) ** 1.5,
            (5 / 4) ** 1.75,
        ],
    }

    def __init__(
        self,
        scale="major",
        root_note="C",
        tuning_freq=440.0,
        freq_range=(80, 4000),
        octave_range=None,
        metrical_quantizer: MetricalQuantizer = None,
        edo_divisions=12,
        use_just_intonation=False,
    ):
        """
        Initialize the enhanced note quantizer with EDO and
        experimental tuning support.

        Args:
            scale: Scale to use (traditional or EDO scale name)
            root_note: Root note of the scale
            tuning_freq: Frequency of A4 in Hz
            freq_range: (min_freq, max_freq) range
            octave_range: (min_octave, max_octave) or None
            metrical_quantizer: Optional meter quantizer
            edo_divisions: Number of equal divisions per octave
                    (12 = standard, 19, 24, 31, etc.)
            use_just_intonation: Use just intonation ratios
                    instead of equal temperament
        """
        self.scale = scale
        self.root_note = root_note
        self.tuning_freq = tuning_freq
        self.freq_range = freq_range
        self.metrical_quantizer = metrical_quantizer
        self.edo_divisions = edo_divisions
        self.use_just_intonation = use_just_intonation

        # Calculate the frequency of C4 based on A4 tuning
        self.c4_freq = tuning_freq / (2 ** (9 / 12))

        # Determine tuning system
        self.tuning_system = self._determine_tuning_system()

        # Generate all available note frequencies
        self.note_frequencies = self._generate_note_frequencies(octave_range)

    def _determine_tuning_system(self):
        """Determine which tuning system to use."""
        if (
            self.use_just_intonation
            and self.scale in self.JUST_INTONATION_RATIOS
        ):
            return "Just Intonation"
        elif self.scale in self.EDO_SCALES or self.edo_divisions != 12:
            return "EDO"
        else:
            return "12-TET"

    def _generate_note_frequencies(self, octave_range=None):
        """Generate all note frequencies within the specified range
        using the selected tuning system."""
        if octave_range is None:
            # Auto-calculate octave range based on frequency range
            min_octave = max(
                0, int(math.log2(self.freq_range[0] / self.c4_freq)) + 4 - 1
            )
            max_octave = min(
                10, int(math.log2(self.freq_range[1] / self.c4_freq)) + 4 + 1
            )
        else:
            min_octave, max_octave = octave_range

        note_frequencies = []

        if self.tuning_system == "Just Intonation":
            note_frequencies = self._generate_just_intonation_frequencies(
                min_octave, max_octave
            )
        elif self.tuning_system == "EDO":
            note_frequencies = self._generate_edo_frequencies(
                min_octave, max_octave
            )
        else:  # 12-TET
            note_frequencies = self._generate_12tet_frequencies(
                min_octave, max_octave
            )

        # Sort by frequency
        note_frequencies.sort(key=lambda x: x["frequency"])
        return note_frequencies

    def _generate_12tet_frequencies(self, min_octave, max_octave):
        """Generate frequencies for 12-tone equal temperament."""
        note_frequencies = []
        scale_intervals = self.SCALES.get(self.scale, self.SCALES["major"])
        root_semitone = self.CHROMATIC_NOTES.index(self.root_note)

        for octave in range(min_octave, max_octave + 1):
            for interval in scale_intervals:
                # Calculate semitone position relative to C4
                semitone_from_c4 = (octave - 4) * 12 + root_semitone + interval

                # Calculate frequency
                frequency = self.c4_freq * (2 ** (semitone_from_c4 / 12))

                # Only include if within frequency range
                if self.freq_range[0] <= frequency <= self.freq_range[1]:
                    note_name = self.CHROMATIC_NOTES[
                        (root_semitone + interval) % 12
                    ]
                    note_frequencies.append(
                        {
                            "frequency": frequency,
                            "note": note_name,
                            "octave": octave,
                            "semitone_from_c4": semitone_from_c4,
                            "midi_note": 60 + semitone_from_c4,
                            "cents_from_c4": semitone_from_c4 * 100,
                            "tuning_system": "12-TET",
                        }
                    )

        return note_frequencies

    def _generate_edo_frequencies(self, min_octave, max_octave):
        """Generate frequencies for EDO (Equal Division of Octave) systems."""
        note_frequencies = []

        # Get EDO intervals
        if self.scale in self.EDO_SCALES:
            edo_intervals = self.EDO_SCALES[self.scale]
            # Determine EDO from scale name or use specified divisions
            if "_edo" in self.scale:
                edo_divisions = (
                    int(self.scale.split("_")[0])
                    if self.scale.split("_")[0].isdigit()
                    else self.edo_divisions
                )
            else:
                edo_divisions = self.edo_divisions
        else:
            # Use all divisions for the specified EDO
            edo_intervals = list(range(self.edo_divisions))
            edo_divisions = self.edo_divisions

        # Calculate step size in cents (1200 cents = 1 octave)
        cents_per_step = 1200 / edo_divisions

        # Find root note offset in EDO divisions
        if edo_divisions == 12:
            root_offset = self.CHROMATIC_NOTES.index(self.root_note)
        else:
            # For non-12-EDO, map root note to closest EDO division
            root_semitone = self.CHROMATIC_NOTES.index(self.root_note)
            root_offset = round((root_semitone * edo_divisions) / 12)

        for octave in range(min_octave, max_octave + 1):
            for interval in edo_intervals:
                # Calculate position in EDO system
                edo_position = (
                    (octave - 4) * edo_divisions + root_offset + interval
                )

                # Calculate frequency using EDO division
                cents_from_c4 = edo_position * cents_per_step
                frequency = self.c4_freq * (2 ** (cents_from_c4 / 1200))

                # Only include if within frequency range
                if self.freq_range[0] <= frequency <= self.freq_range[1]:
                    # Generate note name for EDO system
                    note_name = self._generate_edo_note_name(
                        interval, edo_divisions, root_offset
                    )

                    note_frequencies.append(
                        {
                            "frequency": frequency,
                            "note": note_name,
                            "octave": octave,
                            "edo_division": interval,
                            "edo_system": f"{edo_divisions}-EDO",
                            "cents_from_c4": cents_from_c4,
                            "midi_note": 60
                            + (cents_from_c4 / 100),  # Microtonal MIDI
                            "tuning_system": f"{edo_divisions}-EDO",
                        }
                    )

        return note_frequencies

    def _generate_just_intonation_frequencies(self, min_octave, max_octave):
        """Generate frequencies for just intonation scales."""
        note_frequencies = []

        if self.scale not in self.JUST_INTONATION_RATIOS:
            print(
                f"Warning: {self.scale} not available in just "
                f"intonation. Using 12-TET."
            )
            return self._generate_12tet_frequencies(min_octave, max_octave)

        ratios = self.JUST_INTONATION_RATIOS[self.scale]
        note_names = ["C", "D", "E", "F", "G", "A", "B"][: len(ratios)]

        # Find root frequency (assume C4 as reference)
        root_index = self.CHROMATIC_NOTES.index(self.root_note)
        root_frequency_c4 = self.c4_freq * (2 ** (root_index / 12))

        for octave in range(min_octave, max_octave + 1):
            octave_multiplier = 2 ** (octave - 4)

            for i, ratio in enumerate(ratios):
                frequency = root_frequency_c4 * ratio * octave_multiplier

                # Only include if within frequency range
                if self.freq_range[0] <= frequency <= self.freq_range[1]:
                    note_name = (
                        note_names[i] if i < len(note_names) else f"N{i}"
                    )

                    # Calculate cents deviation from 12-TET
                    tet_frequency = (
                        root_frequency_c4
                        * (2 ** (i * 2 / 12))
                        * octave_multiplier
                    )  # Approximate
                    cents_deviation = (
                        1200 * math.log2(frequency / tet_frequency)
                        if tet_frequency > 0
                        else 0
                    )

                    note_frequencies.append(
                        {
                            "frequency": frequency,
                            "note": note_name,
                            "octave": octave,
                            "ratio": ratio,
                            "cents_deviation_from_12tet": cents_deviation,
                            "midi_note": 60
                            + math.log2(frequency / self.c4_freq) * 12,
                            "tuning_system": "Just Intonation",
                        }
                    )

        return note_frequencies

    def _generate_edo_note_name(self, division, edo_divisions, root_offset):
        """Generate a note name for EDO systems."""
        if edo_divisions == 12:
            # Use traditional note names for 12-EDO
            note_index = (division + root_offset) % 12
            return self.CHROMATIC_NOTES[note_index]
        elif edo_divisions == 24:
            # Quarter-tone system
            base_names = [
                "C",
                "C+",
                "C#",
                "D♭+",
                "D",
                "D+",
                "D#",
                "E♭+",
                "E",
                "E+",
                "F",
                "F+",
                "F#",
                "G♭+",
                "G",
                "G+",
                "G#",
                "A♭+",
                "A",
                "A+",
                "A#",
                "B♭+",
                "B",
                "B+",
            ]
            return base_names[division % 24]
        else:
            # Generic EDO notation
            return f"{division}/{edo_divisions}"

    def get_scale_info(self):
        """Get detailed information about the current
        scale and tuning system."""
        info = {
            "scale_name": self.scale,
            "root_note": self.root_note,
            "tuning_system": self.tuning_system,
            "num_notes": len(self.note_frequencies),
            "frequency_range": (
                (
                    self.note_frequencies[0]["frequency"]
                    if self.note_frequencies
                    else 0
                ),
                (
                    self.note_frequencies[-1]["frequency"]
                    if self.note_frequencies
                    else 0
                ),
            ),
        }

        if self.tuning_system == "EDO":
            info["edo_divisions"] = self.edo_divisions
        elif self.tuning_system == "Just Intonation":
            info["ratios"] = self.JUST_INTONATION_RATIOS.get(self.scale, [])

        return info

    @classmethod
    def list_available_scales(cls):
        """List all available scales and tuning systems."""
        print("Available Scales:")
        print("\nTraditional 12-TET scales:")
        for scale_name in sorted(cls.SCALES.keys()):
            intervals = cls.SCALES[scale_name]
            print(f"  {scale_name}: {intervals}")

        print("\nEDO and Microtonal scales:")
        for scale_name in sorted(cls.EDO_SCALES.keys()):
            intervals = cls.EDO_SCALES[scale_name]
            if len(intervals) > 12:
                print(f"  {scale_name}: {len(intervals)} divisions")
            else:
                print(f"  {scale_name}: {intervals}")

        print("\nJust Intonation scales:")
        for scale_name in sorted(cls.JUST_INTONATION_RATIOS.keys()):
            ratios = cls.JUST_INTONATION_RATIOS[scale_name]
            print(f"  {scale_name}: {[f'{r:.3f}' for r in ratios]}")

    @classmethod
    def create_custom_edo(cls, divisions, intervals=None, name=None):
        """
        Create a custom EDO scale.

        Args:
            divisions: Number of equal divisions per octave
            intervals: List of intervals to use (None for all divisions)
            name: Name for the custom scale

        Returns:
            Scale intervals list
        """
        if intervals is None:
            intervals = list(range(divisions))

        if name:
            cls.EDO_SCALES[name] = intervals
            print(
                f"Created custom EDO scale '{name}': {divisions} "
                f"divisions, intervals {intervals}"
            )

        return intervals

    def quantize_frequency_log(self, frequency):
        """Find the closest musical note using logarithmic distance."""
        if not self.note_frequencies:
            return {
                "frequency": frequency,
                "note": "N/A",
                "octave": 0,
                "midi_note": 60,
            }

        if frequency <= 0:
            return self.note_frequencies[0]

        log_freq = math.log(frequency)
        min_distance = float("inf")
        closest_note = None

        for note_info in self.note_frequencies:
            log_note_freq = math.log(note_info["frequency"])
            distance = abs(log_note_freq - log_freq)
            if distance < min_distance:
                min_distance = distance
                closest_note = note_info.copy()
                closest_note["original_frequency"] = frequency
                closest_note["log_distance"] = distance

        return closest_note

    def quantize_with_meter(
        self,
        frequency,
        time_ticks,
        duration_ticks,
        intensity=0.5,
        peak_width_seconds=None,
    ):
        """
        Quantize frequency, timing, and duration using meter information.

        Args:
            frequency: Input frequency
            time_ticks: Input timing in MIDI ticks
            duration_ticks: Input duration in MIDI ticks
            intensity: Normalized intensity (0-1)
            peak_width_seconds: Original peak width in seconds

        Returns:
            dict: Quantized note information
        """
        # Quantize frequency to scale
        note_info = self.quantize_frequency_log(frequency)

        # Quantize timing and duration if meter quantizer is available
        if self.metrical_quantizer:
            quantized_time = self.metrical_quantizer.quantize_onset_time(
                time_ticks
            )
            quantized_duration = self.metrical_quantizer.quantize_duration(
                duration_ticks, intensity, peak_width_seconds
            )
        else:
            quantized_time = time_ticks
            quantized_duration = duration_ticks

        note_info.update(
            {
                "quantized_time": quantized_time,
                "quantized_duration": quantized_duration,
                "original_time": time_ticks,
                "original_duration": duration_ticks,
                "intensity": intensity,
            }
        )

        return note_info


def mz_to_frequency_inverse_log(mz_value, freq_range, min_mz_overall,
                                max_mz_overall):
    """Maps m/z values to frequencies using inverse logarithmic scaling."""
    if mz_value <= 0 or min_mz_overall <= 0:
        return freq_range[0]

    mz_normalized = (mz_value - min_mz_overall) / (
        max_mz_overall - min_mz_overall
    )
    mz_normalized = np.clip(mz_normalized, 0, 1)

    log_ratio = math.log(freq_range[1] / freq_range[0])
    frequency = freq_range[1] * math.exp(-mz_normalized * log_ratio)

    return frequency


def mz_to_frequency_power_law(mz_value, freq_range, min_mz_overall,
                              max_mz_overall, exponent=1.5):
    """Maps m/z values using a power law relationship."""
    if mz_value <= 0:
        return freq_range[0]

    mz_normalized = (mz_value - min_mz_overall) / (
        max_mz_overall - min_mz_overall
    )
    mz_normalized = np.clip(mz_normalized, 0, 1)

    freq_normalized = (1 - mz_normalized) ** exponent

    log_freq_min = math.log(freq_range[0])
    log_freq_max = math.log(freq_range[1])
    log_frequency = log_freq_min + freq_normalized * (
        log_freq_max - log_freq_min
    )

    return math.exp(log_frequency)


def mz_to_frequency_musical_octaves(
    mz_value, min_mz_overall, max_mz_overall, base_freq=440.0, num_octaves=4
):
    """Maps m/z to frequencies using musical octaves."""
    if mz_value <= 0:
        return base_freq

    mz_normalized = (mz_value - min_mz_overall) / (
        max_mz_overall - min_mz_overall
    )
    mz_normalized = np.clip(mz_normalized, 0, 1)

    mz_inverted = 1 - mz_normalized
    octave_position = mz_inverted * num_octaves
    frequency = base_freq * (2**octave_position)

    return frequency


def mz_to_frequency_chromatic(
    mz_value, min_mz_overall, max_mz_overall, base_freq=261.63,
    num_semitones=48
):
    """Maps m/z to frequencies using chromatic scale."""
    if mz_value <= 0:
        return base_freq

    mz_normalized = (mz_value - min_mz_overall) / (
        max_mz_overall - min_mz_overall
    )
    mz_normalized = np.clip(mz_normalized, 0, 1)
    mz_inverted = 1 - mz_normalized

    semitone_position = mz_inverted * num_semitones
    frequency = base_freq * (2 ** (semitone_position / 12))

    return frequency


def mz_to_frequency_linear(mz_value, freq_range, min_mz_overall,
                           max_mz_overall):
    """Maps m/z values to frequencies using linear scaling."""
    if mz_value <= 0:
        return freq_range[0]

    mz_normalized = (mz_value - min_mz_overall) / (
        max_mz_overall - min_mz_overall
    )
    mz_normalized = np.clip(mz_normalized, 0, 1)

    return freq_range[0] + mz_normalized * (freq_range[1] - freq_range[0])
