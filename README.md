# ms_music: Mass Spectrometry Data Sonification

**Version: 0.2.0**

`ms_music` is a  Python package for transforming mass spectrometry data into "music". This sonification toolkit goes beyond simple data-to-sound conversion, offering musical quantization, extensive effects processing, MIDI generation, and visualization.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Usage](#detailed-usage)
   * [Basic Sonification](#basic-sonification)
   * [Musical Quantization](#musical-quantization)
   * [Frequency Mappings](#frequency-mappings)
   * [Audio Effects Processing](#audio-effects-processing)
   * [MIDI Generation](#midi-generation)
   * [Visualization](#visualization)
   * [FID Data Processing](#fid-data-processing)
5. [Examples](#examples)
6. [Contributing](#contributing)
7. [License](#license)

## Features

### Core Sonification
* **mzML File Support**: Load and process standard `.mzML` files with MS1/MS2 level selection
* **Sonification Methods**:
  - **Gradient Method**: Continuous sine wave synthesis with smooth intensity transitions
  - **ADSR Method**: Event-based synthesis with customizable envelope shaping
* **Multiple Frequency Mappings**: `inverse_log`, `power_law`, `musical_octaves`, `chromatic`, `linear`

### Musical Methods
* **Musical Scale Quantization**: Transform raw frequencies into musical scales
  - Traditional Western scales: major, minor, pentatonic, blues, dorian, mixolydian, whole-tone, and more
  - Microtonal systems: 19-EDO, 24-EDO, 31-EDO, 53-EDO, and custom divisions
  - Just intonation with pure frequency ratios
  - Traditional non-Western scales: Arabic maqam, Turkish makam, Indian raga approximations
* **Advanced Tuning Systems**: Support for equal temperament, just intonation, and custom tuning
* **Metrical Quantization**: Align timing to musical meters (4/4, 3/4, 6/8, 5/4, 7/8, etc.)

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
   - MS data: `matchms>=0.25.0`
   - Audio: `librosa>=0.9.0`
   - MIDI: `mido>=1.2.10`
   - Visualization: `seaborn>=0.11.0`

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

# Create musical sonification
sonifier.sonify_quantized(
    base_mapping='inverse_log',
    method_params={
        'scale': 'pentatonic_major',
        'root_note': 'C',
        'freq_range': (200, 2000)
    }
)

# Apply professional effects
sonifier.apply_effect('reverb', {'reverb_time_s': 1.5, 'dry_wet_mix': 0.3})
sonifier.apply_effect('compressor', {'threshold_db': -15, 'ratio': 3})

# Save the result
sonifier.save_audio(os.path.join(output_dir, "ms_music.wav"))
print("Musical sonification complete!")
```

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

# Basic sonification with different frequency mappings
sonifier.sonify(
    method='gradient',
    method_params={
        'frequency_mapping': 'inverse_log',  # or 'power_law', 'musical_octaves', 'chromatic', 'linear'
        'freq_range': (200, 4000),
        'overlap_percentage': 0.05
    }
)

# ADSR method for more rhythmic results
sonifier.sonify(
    method='adsr',
    method_params={
        'frequency_mapping': 'musical_octaves',
        'adsr_settings': {
            'attack': 0.01,
            'decay': 0.1,
            'sustain': 0.7,
            'release': 0.2
        }
    }
)
```

### Musical Quantization

Transform raw frequencies into proper musical scales:

```python
# Setup musical quantization
sonifier.setup_musical_quantization(
    scale="major",          # or "minor", "pentatonic_major", "blues", etc.
    root_note="C",          # Root note of the scale
    tuning_freq=440.0,      # A4 frequency
    freq_range=(200, 3000)
)

# Sonify with quantization
sonifier.sonify_quantized(
    base_mapping='inverse_log',
    method_params={
        'scale': 'dorian',
        'root_note': 'D',
        'use_log_distance': True
    }
)

# Explore microtonal scales
from ms_music.musical_quantization import MusicalNoteQuantizer

# List available scales
MusicalNoteQuantizer.list_available_scales()

# Use 19-tone equal temperament
sonifier.sonify_quantized(
    base_mapping='power_law',
    method_params={
        'scale': '19_edo_diatonic',
        'root_note': 'C'
    }
)
```

### Frequency Mappings

```python
# Different mapping approaches for varied sonic results
mappings = ['inverse_log', 'power_law', 'musical_octaves', 'chromatic', 'linear']

for mapping in mappings:
    sonifier.sonify(
        method='gradient',
        method_params={
            'frequency_mapping': mapping,
            'freq_range': (200, 4000)
        }
    )
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
        sonifier.sonify_quantized(
            base_mapping='inverse_log',
            method_params={'scale': method, 'root_note': 'G'}
        )
    else:
        sonifier.sonify(
            method='gradient',
            method_params={'frequency_mapping': method}
        )
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

Process FID data:

```python
from ms_music.sonifier import FIDProcessor, add_fid_to_sonifier

# Direct FID processing
processor = FIDProcessor(sample_rate=44100)
processor.read_fid("path/to/fid/file")
processor.to_audio()
processor.plot_fid()

# Integrate with sonifier
sonifier = MSSonifier("")
sonifier.setup_musical_quantization()
fid_sonifier = add_fid_to_sonifier(sonifier)
audio = fid_sonifier.sonify_fid("path/to/fid/file")
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
