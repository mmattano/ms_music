# ms_music: Mass Spectrometry Data Sonification

`ms_music` is a  Python package for transforming mass spectrometry data into "music". This sonification toolkit goes beyond simple data-to-sound conversion, offering musical quantization, extensive effects processing, MIDI generation, and visualization.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
   * [Install (no coding needed)](#install-no-coding-needed)
   * [Install for Python users](#install-for-python-users)
3. [Quick Start](#quick-start)
4. [Graphical interface](#graphical-interface)
5. [Detailed Usage](#detailed-usage)
   * [Basic Sonification](#basic-sonification)
   * [Musical Quantization](#musical-quantization)
   * [Rhythm & Meter](#rhythm--meter)
   * [Large Files, Ion Mobility and MS2](#large-files-ion-mobility-and-ms2)
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
* **Large runs**: spectra are binned while streaming, so multi-GB files (e.g. timsTOF runs with
  ~300k peaks per frame) load in bounded memory; a cache makes repeat loads take seconds
* **Ion mobility**: 1/K0, drift time or FAIMS CV are read automatically, can filter the data,
  shape the sound (brightness, stereo position, effects that change with mobility), and have
  their own plots and a spatial video
* **MS2**: DDA precursor tones, DIA isolation-window tones, and dense PASEF data merged into time bins
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

### Install (no coding needed)

The installers set up everything ms_music needs (including its own copy of
Python) in your user account: no admin rights, no terminal, nothing
installed system-wide. The first install downloads about 500 MB and takes a
few minutes.

**macOS**

1. From the [latest release](https://github.com/mmattano/MS_Music/releases/latest),
   download **ms_music-macOS-installer.zip** and double-click it to unzip.
2. In the unzipped folder, **right-click** (or Control-click)
   **Install ms_music.command** and choose **Open**, then **Open** again.
   macOS asks this once because the file comes from the internet and isn't
   from the App Store.
3. A window shows the progress. When it's done, ms_music opens in your
   browser. From then on, start it from the **ms_music** icon on your
   Desktop or in the Applications folder of your home folder.

**Windows**

1. From the [latest release](https://github.com/mmattano/MS_Music/releases/latest),
   download **Install ms_music.bat** and double-click it.
2. If Windows shows **"Windows protected your PC"**, click **More info** →
   **Run anyway** (it does this for downloaded programs it doesn't know yet).
3. A window shows the progress. When it's done, ms_music opens in your
   browser. From then on, start it from the **ms_music** shortcut on your
   Desktop or in the Start menu.

**Using it:** ms_music runs on your own computer and opens in your web
browser; your data never leaves your machine. Close the browser tab (or
click the power button at the top right) to stop it. Starting takes a few
seconds.

**Updating:** run the installer again. **Removing:** run the
**Uninstall ms_music** file from the same download (it asks before deleting
cached data).

<details>
<summary>Prefer a one-line command?</summary>

macOS (Terminal):

```bash
curl -LsSf https://raw.githubusercontent.com/mmattano/MS_Music/main/installer/install_macos.sh | bash
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/mmattano/MS_Music/main/installer/install_windows.ps1 | iex"
```
</details>

### Install for Python users

Requires Python 3.10 or newer.

```bash
pip install ms_music        # or: uv tool install ms_music
ms-music-gui                # start the GUI
```

From source:

```bash
git clone https://github.com/mmattano/MS_Music.git
cd MS_Music
pip install .
```

Dependencies are installed automatically:

- Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`
- MS data: `pymzml>=2.5.0`
- Audio: `librosa>=0.9.0`
- MIDI: `mido>=1.2.10`, `scikit-learn>=1.0.0` (peak clustering)
- Visualization: `seaborn>=0.11.0`
- GUI: `nicegui>=2.0`
- Video: `imageio-ffmpeg` (a bundled ffmpeg; a system `ffmpeg` on your `PATH`
  is used instead when present)

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

* **Data** – open an `.mzML`, raw FID, or `.wav` file; choose MS1 or MS2, an RT window, a mobility
  window and the cache; a progress bar shows loading, and a summary card lists what was loaded
  (scans, RT, ion mobility, MS2 precursors/windows). For DIA data, load the MS1 reference here.
* **Sonify** – one page for every option of `sonify()`: gradient or ADSR synthesis, any
  frequency mapping, optional snapping to a scale (12-TET, EDO, just intonation), optional
  rhythm (meter, tempo, grid, swing, accents), MS2 precursor tones (only the modes the loaded
  data supports are offered, with the reason for the others), and ion mobility: brightness,
  stereo width, and effects whose settings change with mobility (with presets).
* **Effects** – browse every audio effect, tweak its parameters, and chain them with undo / reset.
* **MIDI** – configure scale, tempo, meter, and peak detection, then download a `.mid` file
  (multi-file voice exports come as a `.zip`).
* **Visualize** – audio plots (spectrograms, 3D views, MFCCs, chromagrams, envelopes), MS-data
  plots (m/z → frequency mapping, scan progression, summary grid, MS2 precursor map), ion mobility
  plots (mobility map, mobility over time) and comparisons between kept
  versions (spectrograms, spectra, difference spectrogram, similarity matrices, feature
  heatmap). Every plot's options are editable; download as PNG.
* **Video** – render any of the video types (playback, animated views, 3D waterfalls and
  build-ups, 3D scan/heatmap views, side-by-side comparisons) with a progress bar, time
  estimate and Cancel; preview in the page and download as MP4 (ffmpeg is bundled).

A player bar at the bottom always shows the current audio's waveform (with bar lines when it
is metered), lets you play and seek it in the browser, lists the applied effect chain, and
downloads the result as WAV. Its bookmark button **keeps the current audio as a named
version** for the comparison plots and videos. Long-running steps run in the background, so
the interface stays responsive; a spinner in the header shows when work is in progress, and
the log (header button) records every step.

## Detailed Usage

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

### Large Files, Ion Mobility and MS2

`load_and_preprocess_data()` streams the file and bins each spectrum to
integer m/z as it is read, so memory follows the binned data rather than the
raw peaks. A 22 GB timsTOF DDA-PASEF run (5,500 MS1 frames with ~300k peaks
each) loads in under 2 GB of RAM. The binned result is cached
(`~/.cache/ms_music`, or `$MS_MUSIC_CACHE_DIR`), keyed by the file's path,
size and modification time, so loading it again takes about a second.
Manage it with `io.list_cache()`, `io.remove_cache(key)` and
`io.clear_cache()`, or in the GUI's Data page ("Cache" section).

```python
sonifier = MSSonifier("run.mzML", ms_level=1, total_duration_minutes=1)
sonifier.load_and_preprocess_data(
    rt_range=(20, 40),          # minutes; reading stops after the window
    mobility_range=(0.8, 1.3),  # keep ions in this 1/K0 window
    cache=True,
)
print(sonifier.ion_mobility_data)  # unit and range, or None
```

**Ion mobility** is read automatically when present, either per peak (e.g.
timsTOF's `mean inverse reduced ion mobility array`, drift-time arrays) or
per spectrum (1/K0 of PASEF scans, FAIMS CV). Each m/z bin keeps its
intensity-weighted mean mobility. Use it to shape the sound with
`sonify(mobility=...)`: each ion gets a position from compact (low 1/K0) to
extended (high 1/K0), which can drive

- **brightness**: extended ions get more overtones;
- **pan**: stereo width; compact ions left, extended ions right (the output
  becomes stereo; effects then run per channel);
- **effects**: any effect setting that changes across mobility, e.g. reverb
  wet/dry so extended ions sound farther away. Each mapped effect runs on a
  few mobility bands, and tones crossfade between neighbouring bands.
  Every setting has a meaning, an allowed range and a suggested starting pair:
  `from ms_music.effect_guide import parameter_guide;
  print(parameter_guide("delay", "feedback").describe())`. Out-of-range
  values are rejected, and settings that change the audio's length or the
  effect's structure (e.g. filter order) can't be mapped. The GUI shows the
  same explanations under each mapped effect.

```python
from ms_music.mobility import MappedEffect

sonifier.sonify(mobility={
    "brightness": 0.4,
    "pan": 0.8,
    "effects": [
        MappedEffect("reverb", "dry_wet_mix", 0.05, 0.6, {"reverb_time_s": 1.5}),
        ("lowpass_filter", "cutoff_freq", 8000, 1500),   # tuples work too
    ],
    "bands": 6,
})
```

To look at the data and the mapping:

- `viz.plot_mobility_map(sonifier)`: m/z × mobility; charge states form
  separate bands.
- `viz.plot_mobility_over_time(sonifier)`.
- `viz.plot_ion_mobilogram(sonifier, mz_ranges=[(445, 446)])`: intensity
  over mobility for m/z ranges.
- `viz.plot_mobility_mapping(sonifier)`: how mobility turns into pan,
  brightness and each effect setting, over where the ions sit.
- `viz.plot_stereo_field(audio, sr)`: where a stereo sound sits between left
  and right over time.
- `viz.create_spatial_stage_video(sonifier, "stage.mp4")`: the ions as dots
  on a stage (left/right = stereo position, up/down = pitch, faint = far),
  synced to the audio.
- `viz.create_raw_data_video(sonifier, "raw.mp4")`: the raw data behind the
  sound, synced to the audio: chromatogram with a playhead, the MS1 spectrum
  sounding now (with the current MS2 precursor marked), the nearest MS2
  spectrum, and the ions' mobility. Each panel can be switched off
  (`show_chromatogram`, `show_ms1`, `show_ms2`, `show_mobility`); the MS level
  you didn't load is read from the same file for the same RT window.

**MS2** spectra are ordered by retention time. Very dense MS2 data (e.g.
~250k PASEF spectra) is merged into short time bins (≥5 ms of audio each)
when sonified; precursor tones then include every precursor in a bin.
Keeping only the top peaks per scan helps MS2 sound less like noise:

```python
import ms_music.visualizations as viz

ms2 = MSSonifier("run.mzML", ms_level=2, total_duration_minutes=1)
ms2.load_and_preprocess_data()
ms2.sonify(ms2_mode="dda", max_peaks_per_scan=50)   # selected precursors

ms2.load_ms1_reference()                             # MS1 peaks for DIA
ms2.sonify(ms2_mode="dia", max_peaks_per_scan=50)   # isolation windows

viz.plot_precursor_map(ms2, color_by="charge")      # RT × precursor m/z
```

MIDI export can reuse data that is already loaded, instead of reading the file again:

```python
from ms_music import MSSonifierMidi

midi = MSSonifierMidi("run.mzML")
midi.load_processed(ms2.processed_spectra_dfs, ms2.max_intensity_overall,
                    ms2.min_mz_overall, ms2.max_mz_overall,
                    total_duration_seconds=60)
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

The Jupyter notebook [`examples/examples.ipynb`](https://github.com/mmattano/MS_Music/blob/main/examples/examples.ipynb) demonstrates:

- Loading and preprocessing MS data
- All sonification methods and frequency mappings
- Musical scale quantization including microtonal systems
- Complete effects processing examples
- MIDI generation workflows
- Visualization techniques
- FID data processing

Run the notebook to explore the full capabilities!

[`examples/generate_examples.py`](https://github.com/mmattano/MS_Music/blob/main/examples/generate_examples.py) renders an example of every
feature (sonification methods, tunings, effects, MIDI, plots and videos) from your own data:

```bash
python examples/generate_examples.py path/to/ms1.mzML [path/to/ms2.mzML]
```

The results go to `examples/output/`.

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
