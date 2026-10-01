"""ms_music: turn mass spectrometry data into sound, MIDI and visuals.

Public names are loaded on first use (PEP 562), so ``import ms_music`` and
the GUI launcher start quickly; ``from ms_music import MSSonifier`` works as
usual.
"""

import importlib

__version__ = "0.3.0"

# name -> (submodule, attribute or None for the module itself)
_EXPORTS = {
    # Core classes
    "MSSonifier": ("sonifier", "MSSonifier"),
    "MSSonifierMidi": ("midi_generator", "MSSonifierMidi"),
    "MidiConfig": ("midi_generator", "MidiConfig"),
    "RhythmConfig": ("rhythm", "RhythmConfig"),
    "MobilityConfig": ("mobility", "MobilityConfig"),
    # Enums and constants
    "MusicMeter": ("midi_generator", "MusicMeter"),
    "QuantizationMode": ("midi_generator", "QuantizationMode"),
    # Main functions
    "load_mzml_data": ("io", "load_mzml_data"),
    "preprocess_spectra": ("io", "preprocess_spectra"),
    # Modules
    "effects": ("effects", None),
    "visualizations": ("visualizations", None),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module 'ms_music' has no attribute {name!r}"
        ) from None
    module = importlib.import_module(f".{module_name}", __name__)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
