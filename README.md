# ms_music: Mass Spectrometry Data Sonification

**Version: 0.2.0**

`ms_music` is a  Python package for transforming mass spectrometry data into "music". This sonification toolkit goes beyond simple data-to-sound conversion, offering musical quantization, extensive effects processing, MIDI generation, and visualization.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Graphical interface](#graphical-interface)
5. [Detailed Usage](#detailed-usage)
   * [Basic Sonification](#basic-sonification)
   * [Musical Quantization](#musical-quantization)
   * [Rhythm & Meter](#rhythm--meter)
   * [Frequency Mappings](#frequency-mappings)
   * [Audio Effects Processing](#audio-effects-processing)
   * [MIDI Generation](#midi-generation)
   * [Visualization](#visualization)
   * [FID Data Processing](#fid-data-processing)
6. [Examples](#examples)
7. [Contributing](#contributing)
8. [License](#license)

## Features

### Core Sonification
* **mzML File Support**: Load and process standard `.mzML` files with MS1/MS2 level selection
* **Sonification Methods**:
  - **Gradient Method**: Continuous sine wave synthesis with smooth intensity transitions
  - **ADSR Method**: Event-based synthesis with customizable envelope shaping
* **Multiple Frequency Mappings**: `inverse_log`, `power_law`, `musical_octaves`, `chromatic`, `linear`
* **One `sonify()` for everything**: every method, mapping, scale, tuning and rhythm option combines freely

### Musical Methods
* **Musical Scale Quantization**: Transform raw frequencies into musical scales
  - Traditional Western scales: major, minor, pentatonic, blues, dorian, mixolydian, whole-tone, and more
  - Microtonal systems: 19-EDO, 24-EDO, 31-EDO, 53-EDO, and custom divisions
  - Just intonation with pure frequency ratios
  - Traditional non-Western scales: Arabic maqam, Turkish makam, Indian raga approximations
* **Advanced Tuning Systems**: Support for equal temperament, just intonation, and custom tuning
* **Rhythm & Meter**: Play audio in time: regroup scans onto a beat grid in any meter (4/4, 3/4, 6/8, 5/4, 7/8, …) with tempo, swing, accents and articulation; MIDI export uses the same meters

### Audio Effects
* **Filters**: Lowpass, highpass, bandpass, notch, parametric EQ, graphic EQ
* **Time-based Effects**: Reverb, delay, echo, chorus, flanger, phaser
* **Dynamics**: Compressor, gate, limiter, expander, multiband compressor
* **Distortion**: Overdrive, fuzz, bitcrusher, waveshaper with multiple curve types
* **Modulation**: Tremolo, vibrato, ring modulation, auto-wah
* **Spectral Processing**: HPSS separation, pitch shifting, time stretching, formant shifting
* **Creative Effects**: Granular synthesis, convolution reverb, spectral filtering

### MIDI Generation
* **Musical MIDI Export**: Generate MIDI files with proper musical timing and scales
* **Peak Detection**: Peak detection with retention time mapping
* **Configurable Parameters**: Tempo, time signatures, instruments, quantization modes
* **Analysis Tools**: Reporting and note data export

### Visualization Suite
* **Spectrograms**: 2D and 3D spectrogram visualizations
* **Comparative Analysis**: Waveform, frequency spectrum, and feature comparisons
* **Musical Analysis**: Chromagrams, MFCC evolution, pitch class distributions
* **Data Insights**: m/z mapping visualizations, scan progression, similarity matrices

### Additional Capabilities
* **FID Data Processing**: Support for FID data processing
* **Extensive Customization**: Fine-tune every aspect of the sonification process
* **Professional Output**: High-quality WAV export with normalization options

## Installation

### Prerequisites
* Python 3.8 or higher
* `pip` package manager

### Installation Steps

1. **Install from source:**
   ```bash
   git clone https://github.com/mmattano/ms_music.git
   cd ms_music
   pip install .
   ```

2. **Dependencies** (automatically installed):
   - Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`
   - MS data: `pymzml>=2.5.0`
   - Audio: `librosa>=0.9.0`
   - MIDI: `mido>=1.2.10`, `scikit-learn>=1.0.0` (peak clustering)
   - Visualization: `seaborn>=0.11.0`
   - GUI: `nicegui>=2.0`

   > **External tools:** video/animation export requires [`ffmpeg`](https://ffmpeg.org/)
   > to be installed and available on your `PATH`. Audio, MIDI, and static plots do
   > not need it.

## Quick Start

```python
from ms_music import MSSonifier
import os

# Setup
mzml_file = "path/to/your/data.mzML"  # Replace with your file
output_dir = "ms_music_output"
os.makedirs(output_dir, exist_ok=True)

# Initialize sonifier
sonifier = MSSonifier(
    filepath=mzml_file,
    ms_level=1,                     # MS1 data
    total_duration_minutes=0.5,     # 30 seconds of audio
    sample_rate=44100
)

# Load and process data
sonifier.load_and_preprocess_data()

# Create musical sonification: pitches snapped to C pentatonic major
sonifier.sonify(
    frequency_mapping='inverse_log',
    freq_range=(200, 2000),
    scale='pentatonic_major',
    root_note='C',
)

# Apply professional effects
sonifier.apply_effect('reverb', {'reverb_time_s': 1.5, 'dry_wet_mix': 0.3})
sonifier.apply_effect('compressor', {'threshold_db': -15, 'ratio': 3})

# Save the result
sonifier.save_audio(os.path.join(output_dir, "ms_music.wav"))
print("Musical sonification complete!")
```

## Graphical interface

Prefer clicking to coding? `ms_music` ships with a GUI that exposes the full
audio/MIDI/visualization workflow without writing any Python. It runs locally
and opens in your web browser (built with [NiceGUI](https://nicegui.io)).

The GUI is included in the standard install — just launch it:

```bash
ms-music-gui
```

(or `python -m ms_music.gui`). Useful options:

* `--native` – open in a desktop window instead of a browser tab (needs `pip install pywebview`).
* `--port 8765` – serve on a specific port; `--no-browser` – don't open a tab automatically.

The app runs entirely on your machine: files are read from disk by path (via
the built-in file browser), so even multi-GB `.mzML` files are never uploaded.

Pages in the sidebar follow the normal workflow:

* **Data** – open an `.mzML`, raw FID, or `.wav` file and set duration / sample rate.
* **Sonify** – one page for every option of `sonify()`: gradient or ADSR synthesis, any
  frequency mapping, optional snapping to a scale (12-TET, EDO, just intonation), optional
  rhythm (meter, tempo, grid, swing, accents), and MS2 precursor tones.
* **Effects** – browse every audio effect, tweak its parameters, and chain them with undo / reset.
* **MIDI** – configure scale, tempo, meter, and peak detection, then download a `.mid` file
  (multi-file voice exports come as a `.zip`).
* **Visualize** – audio plots (spectrograms, 3D views, MFCCs, chromagrams, envelopes), MS-data
  plots (m/z → frequency mapping, scan progression, summary grid) and comparisons between kept
  versions (spectrograms, spectra, difference spectrogram, similarity matrices, feature
  heatmap). Every plot's options are editable; download as PNG.
* **Video** – render any of the video types (playback, animated views, 3D waterfalls and
  build-ups, 3D scan/heatmap views, side-by-side comparisons) with a progress bar, time
  estimate and Cancel; preview in the page and download as MP4. Needs `ffmpeg`.

A player bar at the bottom always shows the current audio's waveform (with bar lines when it
is metered), lets you play and seek it in the browser, lists the applied effect chain, and
downloads the result as WAV. Its bookmark button **keeps the current audio as a named
version** for the comparison plots and videos. Long-running steps run in the background, so
the interface stays responsive; a spinner in the header shows when work is in progress, and
the log (header button) records every step.

## Detailed Use

### Basic Sonification

```python
from ms_music import MSSonifier

# Initialize
sonifier = MSSonifier(
    filepath="data.mzML",
    ms_level=1,
    total_duration_minutes=1.0,
    sample_rate=44100
)

# Load data
sonifier.load_and_preprocess_data()

# Basic sonification (continuous tones following each m/z's intensity)
sonifier.sonify(
    method='gradient',
    frequency_mapping='inverse_log',  # or 'power_law', 'musical_octaves', 'chromatic', 'linear'
    freq_range=(200, 4000),
)

# ADSR method: each scan becomes a note.
# Envelope stages are expressed as percentages of each note's duration
# (attack/decay/release) and a sustain level (0-1). Set randomize=False
# to use these fixed values instead of randomized envelopes.
sonifier.sonify(
    method='adsr',
    frequency_mapping='musical_octaves',
    adsr_settings={
        'randomize': False,
        'attack_time_pc': 0.1,
        'decay_time_pc': 0.1,
        'sustain_level_pc': 0.7,
        'release_time_pc': 0.2,
    },
)
```

`sonify()` is the single entry point: the synthesis method, frequency
mapping, scale/tuning and rhythm options below all combine with each other.

### Musical Quantization

Snap pitches to a musical scale by passing `scale` to `sonify()`. This
works with both methods and every frequency mapping:

```python
sonifier.sonify(
    frequency_mapping='inverse_log',
    freq_range=(200, 3000),
    scale='dorian',          # or 'major', 'minor', 'pentatonic_major', 'blues', ...
    root_note='D',
    tuning_freq=440.0,       # A4 frequency
    use_log_distance=True,   # nearest note in perceptual (log) pitch
)

# Explore microtonal scales
from ms_music.musical_quantization import MusicalNoteQuantizer

# List available scales
MusicalNoteQuantizer.list_available_scales()

# 19-tone equal temperament
sonifier.sonify(
    frequency_mapping='power_law',
    scale='19_edo_diatonic',
    root_note='C',
    edo_divisions=19,
)

# Just intonation, with ADSR notes
sonifier.sonify(
    method='adsr',
    scale='just_major',
    root_note='D',
    use_just_intonation=True,
)
```

### Rhythm & Meter

By default every scan gets an equal slice of time. Pass `rhythm` to play the
data in time instead: scans are regrouped onto the steps of a beat grid and
the total length snaps to whole bars. Each pitch then plays as notes that
follow its signal, like the MIDI export: a note starts on the grid step where
the pitch's intensity appears and lasts for as many steps as the signal stays
above `note_threshold` (a fraction of that pitch's peak). A long
chromatographic peak is one long note. Notes starting on the downbeat are
accented. Tempo is in quarter-note BPM, as in MIDI export.

```python
from ms_music import RhythmConfig

sonifier.sonify(
    method='adsr',
    scale='pentatonic_minor', root_note='A',
    rhythm={
        'meter': '3/4',        # any MusicMeter: '4/4', '6/8', '5/4', '7/8', ...
        'tempo': 96,           # quarter-note BPM
        'subdivision': 16,     # grid step: 4, 8, 16 or 32
        'mode': 'swing',       # 'strict_grid', 'swing' or 'humanized'
        'swing_ratio': 0.67,
        'note_threshold': 0.05,  # a note lasts while intensity > 5% of its peak
        'accent': 0.4,         # extra gain for notes starting on beat 1
        'gate': 0.9,           # a note's last step sounds 90% (gap before next)
        'aggregate': 'max',    # how scans inside one step combine
    },
)
print(sonifier.rhythm_grid.describe())   # e.g. "3/4 · 96 BPM · 16 bars"

# A RhythmConfig works too and is validated up front
sonifier.sonify(rhythm=RhythmConfig(meter='7/8', tempo=140, subdivision=8))
```

### Frequency Mappings

```python
# Different mapping approaches for varied sonic results
mappings = ['inverse_log', 'power_law', 'musical_octaves', 'chromatic', 'linear']

for mapping in mappings:
    sonifier.sonify(frequency_mapping=mapping, freq_range=(200, 4000))
    sonifier.save_audio(f"ms_sound_{mapping}.wav")
```

### Audio Effects Processing

Apply audio effects:

```python
# Time-based effects
sonifier.apply_effect('reverb', {
    'reverb_time_s': 2.0,
    'room_size': 0.8,
    'dry_wet_mix': 0.4
})

sonifier.apply_effect('chorus', {
    'delay_ms': 20,
    'depth_ms': 3,
    'rate_hz': 0.5,
    'num_voices': 3
})

# Dynamics processing
sonifier.apply_effect('compressor', {
    'threshold_db': -20,
    'ratio': 4,
    'attack_ms': 10,
    'release_ms': 100
})

# Spectral effects
sonifier.apply_effect('pitch_shift', {'n_steps': 2})  # Up 2 semitones
sonifier.apply_effect('hpss', {'harmonic': True})     # Extract harmonics

# Creative effects
sonifier.apply_effect('granular_synthesis', {
    'grain_size_ms': 50,
    'grain_density': 1.5,
    'pitch_variation_semitones': 2.0
})

# EQ and filtering
sonifier.apply_effect('parametric_eq', {
    'frequency_hz': 1000,
    'gain_db': 6,
    'q_factor': 2
})

# Chain multiple effects
effects_chain = [
    ('highpass_filter', {'cutoff_freq': 80}),
    ('compressor', {'threshold_db': -15, 'ratio': 3}),
    ('chorus', {'delay_ms': 25}),
    ('reverb', {'reverb_time_s': 1.5}),
    ('limiter', {'threshold_db': -1})
]

for effect_name, params in effects_chain:
    sonifier.apply_effect(effect_name, params)
```

### MIDI Generation

Generate MIDI files with proper timing and scales:

```python
from ms_music import MSSonifierMidi, MidiConfig, MusicMeter, QuantizationMode

# Configure MIDI generation
config = MidiConfig(
    scale="major",
    root_note="C",
    tempo=120,
    meter=MusicMeter.FOUR_FOUR,
    quantization_mode=QuantizationMode.STRICT_GRID
)

# Initialize MIDI sonifier
midi_sonifier = MSSonifierMidi(
    filepath="data.mzML",
    config=config
)

# Load and analyze data
midi_sonifier.load_and_analyze_data(total_duration_seconds=60.0)

# Setup musical system
midi_sonifier.setup_musical_system(
    scale="pentatonic_major",
    root_note="G",
    tempo=140,
    meter=MusicMeter.FOUR_FOUR
)

# Detect peaks and generate MIDI
midi_sonifier.detect_and_quantize_peaks(
    intensity_threshold_percentile=85.0,
    frequency_mapping='inverse_log'
)

# Export MIDI file
midi_sonifier.generate_midi_file(
    output_path="ms_music.mid",
    track_name="Mass Spec Sonification",
    instrument=1,  # Acoustic Piano
    export_note_data=True
)

# Get analysis report
report = midi_sonifier.get_analysis_report()
print(f"Generated {report['note_count']} notes across {report['unique_pitches']} pitches")
```

### Visualization

Create comprehensive visualizations:

```python
import ms_music.visualizations as viz

# Generate different sonifications for comparison
audio_results = {}
methods = ['inverse_log', 'power_law', 'pentatonic_major']

for method in methods:
    if 'pentatonic' in method:
        sonifier.sonify(scale=method, root_note='G')
    else:
        sonifier.sonify(frequency_mapping=method)
    audio_results[method] = sonifier.get_current_audio(copy=True)

# Create visualizations
sample_rate = sonifier.sample_rate

# Waveform comparison
fig = viz.plot_waveform_comparison(audio_results, sample_rate)

# Spectrogram analysis
fig = viz.plot_spectrogram_comparison(audio_results, sample_rate)

# 3D spectrogram
fig = viz.plot_3d_spectrogram(
    audio_results['pentatonic_major'], 
    sample_rate,
    title="3D Spectrogram - Pentatonic Scale"
)

# Frequency mapping visualization
fig = viz.plot_mz_to_frequency_mapping(
    sonifier, 
    mapping_types='all',
    freq_range=(200, 4000)
)

# Comprehensive summary
fig = viz.create_summary_grid(
    sonifier, 
    audio_results, 
    sample_rate,
    title="MS Music Analysis Summary"
)
```

### FID Data Processing

An `MSSonifier` can sonify a raw FID (free induction decay) file directly,
bypassing the mzML pipeline. The FID is read as little-endian `int32`,
resampled to the target sample rate, and time-stretched to the requested
duration; you can then apply effects and save it like any other audio
buffer. (Scale and rhythm options apply to spectra, so they are not used
for FID input.)

```python
from ms_music import MSSonifier

# filepath is unused for FID input, so pass an empty string
fid_sonifier = MSSonifier(filepath="", total_duration_minutes=0.5, sample_rate=44100)

# Read and convert the raw FID
fid_sonifier.load_fid_data(
    "path/to/fid/file",
    original_sample_rate=10e6,   # acquisition rate of the instrument (Hz)
    conversion_factor=2 ** 12,   # scaling applied to the raw int32 samples
)

# Apply effects like any other audio buffer
fid_sonifier.apply_effect("reverb", {"reverb_time_s": 2.0})

fid_sonifier.save_audio("fid_processed.wav")
```

## Examples

A Jupyter notebook (`examples.ipynb`) demonstrates:

- Loading and preprocessing MS data
- All sonification methods and frequency mappings
- Musical scale quantization including microtonal systems
- Complete effects processing examples
- MIDI generation workflows
- Visualization techniques
- FID data processing

Run the notebook to explore the full capabilities!

## Contributing

We welcome contributions! Here's how to get involved:

1. **Fork the repository** on GitHub
2. **Create a feature branch** (`git checkout -b feature/new-feature`)
3. **Make your changes** with appropriate tests
4. **Commit your changes** (`git commit -m 'Add new feature'`)
5. **Push to the branch** (`git push origin feature/new-feature`)
6. **Open a Pull Request**

### Areas for Contribution

- New musical scales and tuning systems
- Additional audio effects
- Alternative sonification algorithms
- Improved visualization techniques
- Performance optimizations
- Documentation improvements

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
