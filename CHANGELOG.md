# Changelog

## [0.3.0]

First release on PyPI.

### New
- **Graphical interface** (`ms-music-gui`): runs locally in the browser and
  covers loading data, sonifying, effects, MIDI, plots and videos. You can
  keep versions of the audio to compare them.
- **One-click installers** for macOS and Windows: no Python or admin
  rights needed. Running the installer again updates ms_music.
- **One `sonify()` for everything**: gradient and ADSR synthesis, every
  frequency mapping, scales and tunings (12-TET, EDO, just intonation), and
  rhythm/meter (`rhythm=`) combine freely. `sonify_quantized()` and
  `setup_musical_quantization()` were removed.
- **Large files**: spectra are binned while streaming, and the result is
  cached in `~/.cache/ms_music` (or `$MS_MUSIC_CACHE_DIR`).
- **Ion mobility**: 1/K0, drift time and FAIMS CV are read automatically and
  can filter the data. They can also shape the sound: brightness, stereo
  width, and effects whose settings change with mobility. Mobility has its
  own plots and a spatial video.
- **MS2**: DDA precursor tones, DIA isolation-window tones, and dense PASEF
  data merged into time bins.
- Video export uses a bundled ffmpeg (`imageio-ffmpeg`) when none is
  installed.

### Changed
- `MSSonifier.apply_effect()` raises `ValueError` for an unknown effect and
  `RuntimeError` when there is no audio yet. `save_audio()` raises
  `RuntimeError` when there is no audio. Both used to print a message and
  return.
- `save_audio(normalize=False)` now writes the audio as is: float audio is
  full scale at ±1 and clipped beyond that. Before, it was always
  normalized.
- The limiter is about 100× faster.
- Requires Python 3.10+ and NiceGUI 3.

### Fixed
- ADSR sonification no longer produces silent (NaN) audio when all data
  falls in a single m/z bin.
- The adaptive filter no longer diverges to NaN on loud input.
- GUI: effects with required settings (filters, pitch shift, time stretch)
  start with usable values instead of failing with a `TypeError`.
- GUI: a failed load keeps the data and audio that were already loaded.
- GUI: pasted paths may be wrapped in quotes (Windows' "Copy as path"), and
  the file browser can switch drives and volumes.
- The installers stop a running ms_music before updating or removing it.
