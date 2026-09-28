"""What the Visualize and Video pages offer, and how to call each item.

Each entry names the ``ms_music.visualizations`` function it wraps. The
GUI builds a parameter form from that function's signature (minus the
arguments the GUI supplies itself), so new keyword arguments show up
without touching the UI code. ``call`` adapts the form values plus the
current context into the actual function call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from .. import visualizations as viz
from ..sonifier import MSSonifier

# Arguments the GUI always supplies (never shown in forms).
_ALWAYS_SKIP = {
    "audio",
    "audio_data",
    "sr",
    "sample_rate",
    "sonifier",
    "audio_dict",
    "feature_dict",
    "audio1",
    "audio2",
    "name1",
    "name2",
    "methods",
    "corr_matrix",
    "spec_matrix",
    "output_path",
    "figsize",
    "title",
    "custom_mappings",
    "progress_callback",
    "config",
}

CMAPS = [
    "viridis",
    "magma",
    "plasma",
    "inferno",
    "cividis",
    "coolwarm",
    "twilight",
    "Spectral",
    "gray",
]

# Needs: what must exist before an entry can run.
AUDIO = "audio"  # current audio buffer
MS_DATA = "ms_data"  # loaded mzML spectra
COMPARE = "compare"  # >= 2 selected audio versions
PAIR = "pair"  # exactly 2 selected audio versions
MOBILITY = "mobility"  # data loaded with ion mobility
PRECURSORS = "precursors"  # MS2 data with precursor information
STEREO = "stereo"  # current audio is stereo


@dataclass
class Context:
    """Everything an entry may draw on, captured on the event loop."""

    sonifier: Any
    audio: Any
    sample_rate: int
    versions: Dict[str, Any] = field(default_factory=dict)
    stereo_audio: Any = None  # (2, n) when the current audio is stereo


@dataclass
class Entry:
    label: str
    group: str
    func: Callable  # function whose signature is the form
    call: Callable[[Context, dict], Any]
    needs: Sequence[str] = (AUDIO,)
    choices: Dict[str, List[Any]] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)  # GUI overrides
    help: str = ""
    # Videos: measured render cost per frame at dpi 80 (seconds).
    seconds_per_frame: float = 0.0


def unmet_need(
    entry: Entry,
    *,
    has_audio: bool,
    has_ms_data: bool,
    n_versions: int,
    has_mobility: bool = False,
    has_precursors: bool = False,
    is_stereo: bool = False,
) -> str | None:
    """Why ``entry`` can't run right now (None if it can)."""
    if AUDIO in entry.needs and not has_audio:
        return "Generate or load audio first."
    if MS_DATA in entry.needs and not has_ms_data:
        return "Needs a loaded mzML file (Data page)."
    if COMPARE in entry.needs and n_versions < 2:
        return "Select at least two audio versions to compare."
    if PAIR in entry.needs and n_versions != 2:
        return "Select exactly two audio versions."
    if MOBILITY in entry.needs and not has_mobility:
        return "Needs data with ion mobility (e.g. timsTOF)."
    if PRECURSORS in entry.needs and not has_precursors:
        return "Needs MS2 data with precursor information (MS level 2)."
    if STEREO in entry.needs and not is_stereo:
        return (
            "Needs stereo audio: sonify with stereo width above 0 "
            "(Sonify → Ion mobility)."
        )
    return None


def _one(ctx):
    return {"current": ctx.audio}


# ------------------------------------------------------------------- plots
PLOTS: List[Entry] = [
    # Single audio
    Entry(
        "Waveform",
        "Audio",
        viz.plot_waveform_comparison,
        lambda c, p: viz.plot_waveform_comparison(_one(c), c.sample_rate, **p),
    ),
    Entry(
        "Spectrogram",
        "Audio",
        viz.plot_spectrogram,
        lambda c, p: viz.plot_spectrogram(c.audio, c.sample_rate, **p),
        choices={"y_axis": ["log", "linear", "mel"], "cmap": CMAPS},
    ),
    Entry(
        "3D spectrogram",
        "Audio",
        viz.plot_3d_spectrogram,
        lambda c, p: viz.plot_3d_spectrogram(c.audio, c.sample_rate, **p),
        choices={"cmap": CMAPS},
        help="downsample is (time, frequency) decimation; larger is faster.",
    ),
    Entry(
        "3D waterfall",
        "Audio",
        viz.plot_3d_spectrogram_waterfall,
        lambda c, p: viz.plot_3d_spectrogram_waterfall(
            c.audio, c.sample_rate, **p
        ),
        choices={"cmap": CMAPS},
    ),
    Entry(
        "MFCC evolution",
        "Audio",
        viz.plot_mfcc_evolution,
        lambda c, p: viz.plot_mfcc_evolution(c.audio, c.sample_rate, **p),
        choices={"cmap": CMAPS},
    ),
    Entry(
        "Chromagram",
        "Audio",
        viz.plot_chromagram_comparison,
        lambda c, p: viz.plot_chromagram_comparison(
            _one(c), c.sample_rate, **p
        ),
    ),
    Entry(
        "Frequency spectrum",
        "Audio",
        viz.plot_frequency_spectrum_comparison,
        lambda c, p: viz.plot_frequency_spectrum_comparison(
            _one(c), c.sample_rate, **p
        ),
    ),
    Entry(
        "Amplitude envelope",
        "Audio",
        viz.plot_audio_envelope_comparison,
        lambda c, p: viz.plot_audio_envelope_comparison(
            _one(c), c.sample_rate, **p
        ),
    ),
    Entry(
        "Stereo field",
        "Audio",
        viz.plot_stereo_field,
        lambda c, p: viz.plot_stereo_field(c.stereo_audio, c.sample_rate, **p),
        needs=(AUDIO, STEREO),
        choices={"cmap": CMAPS},
        help="Where the sound sits between left and right over time.",
    ),
    # MS data
    Entry(
        "m/z → frequency mapping",
        "MS data",
        viz.plot_mz_to_frequency_mapping,
        lambda c, p: viz.plot_mz_to_frequency_mapping(c.sonifier, **p),
        needs=(MS_DATA,),
        choices={"mapping_types": ["all", *MSSonifier.FREQUENCY_MAPPINGS]},
        help="Left: how each mapping turns m/z into pitch. "
        "Right: where the data's peaks sit.",
    ),
    Entry(
        "Scan progression",
        "MS data",
        viz.plot_scan_progression,
        lambda c, p: viz.plot_scan_progression(c.sonifier, **p),
        needs=(MS_DATA,),
    ),
    Entry(
        "Summary grid",
        "MS data",
        viz.create_summary_grid,
        lambda c, p: viz.create_summary_grid(
            c.sonifier, c.versions or _one(c), c.sample_rate, **p
        ),
        needs=(MS_DATA, AUDIO),
        help="Uses the selected versions (up to three), or the current "
        "audio if none are selected.",
    ),
    Entry(
        "Precursor map",
        "MS data",
        viz.plot_precursor_map,
        lambda c, p: viz.plot_precursor_map(c.sonifier, **p),
        needs=(MS_DATA, PRECURSORS),
        choices={
            "color_by": ["charge", "intensity", "mobility"],
            "cmap": CMAPS,
        },
        help="Retention time vs. precursor m/z of every MS2 scan.",
    ),
    # Ion mobility
    Entry(
        "Mobility map",
        "Ion mobility",
        viz.plot_mobility_map,
        lambda c, p: viz.plot_mobility_map(c.sonifier, **p),
        needs=(MS_DATA, MOBILITY),
        choices={"cmap": CMAPS},
        help="Intensity over m/z and ion mobility; charge states form "
        "separate diagonal bands.",
    ),
    Entry(
        "Mobility over time",
        "Ion mobility",
        viz.plot_mobility_over_time,
        lambda c, p: viz.plot_mobility_over_time(c.sonifier, **p),
        needs=(MS_DATA, MOBILITY),
        choices={"cmap": CMAPS},
    ),
    Entry(
        "Ion mobilogram",
        "Ion mobility",
        viz.plot_ion_mobilogram,
        lambda c, p: viz.plot_ion_mobilogram(c.sonifier, **p),
        needs=(MS_DATA, MOBILITY),
        help="mz ranges: e.g. [(445, 446), (536, 537)]; empty = the three "
        "most intense m/z.",
    ),
    Entry(
        "Mobility → sound mapping",
        "Ion mobility",
        viz.plot_mobility_mapping,
        lambda c, p: viz.plot_mobility_mapping(c.sonifier, **p),
        needs=(MS_DATA, MOBILITY),
        help="Uses the last sonification's mobility settings (or a "
        "preview with pan and brightness).",
    ),
    # Comparisons across saved versions
    Entry(
        "Waveforms",
        "Compare",
        viz.plot_waveform_comparison,
        lambda c, p: viz.plot_waveform_comparison(
            c.versions, c.sample_rate, **p
        ),
        needs=(COMPARE,),
    ),
    Entry(
        "Spectrograms",
        "Compare",
        viz.plot_spectrogram_comparison,
        lambda c, p: viz.plot_spectrogram_comparison(
            c.versions, c.sample_rate, **p
        ),
        needs=(COMPARE,),
        choices={"cmap": CMAPS},
    ),
    Entry(
        "Frequency spectra",
        "Compare",
        viz.plot_frequency_spectrum_comparison,
        lambda c, p: viz.plot_frequency_spectrum_comparison(
            c.versions, c.sample_rate, **p
        ),
        needs=(COMPARE,),
    ),
    Entry(
        "Chromagrams",
        "Compare",
        viz.plot_chromagram_comparison,
        lambda c, p: viz.plot_chromagram_comparison(
            c.versions, c.sample_rate, **p
        ),
        needs=(COMPARE,),
    ),
    Entry(
        "Amplitude envelopes",
        "Compare",
        viz.plot_audio_envelope_comparison,
        lambda c, p: viz.plot_audio_envelope_comparison(
            c.versions, c.sample_rate, **p
        ),
        needs=(COMPARE,),
    ),
    Entry(
        "Difference spectrogram",
        "Compare",
        viz.plot_difference_spectrogram,
        lambda c, p: viz.plot_difference_spectrogram(
            *list(c.versions.values())[:2],
            c.sample_rate,
            *list(c.versions)[:2],
            **p,
        ),
        needs=(PAIR,),
    ),
    Entry(
        "Similarity matrices",
        "Compare",
        viz.plot_similarity_matrices,
        lambda c, p: viz.plot_similarity_matrices(
            *viz.compute_audio_similarity_matrix(c.versions, c.sample_rate),
            **p,
        ),
        needs=(COMPARE,),
        help="Waveform correlation and spectral similarity between "
        "every pair of versions.",
    ),
    Entry(
        "Feature heatmap",
        "Compare",
        viz.plot_feature_comparison,
        lambda c, p: viz.plot_feature_comparison(
            {
                k: viz.extract_audio_features(v, c.sample_rate)
                for k, v in c.versions.items()
            },
            **p,
        ),
        needs=(COMPARE,),
    ),
]

PLOT_GROUPS = ["Audio", "MS data", "Ion mobility", "Compare"]


# ------------------------------------------------------------------ videos
# Video functions take (sonifier, output_path, ...) and read the sonifier's
# current audio; ``call`` receives the output path in params["output_path"].
_FAST = {"fps": 15, "dpi": 80}

VIDEOS: List[Entry] = [
    Entry(
        "Spectrogram & waveform playback",
        "Video",
        viz.create_video,
        lambda c, p: viz.create_video(c.sonifier, **p),
        defaults=dict(_FAST),
        seconds_per_frame=0.5,
        help="A playhead sweeps across static panels; length = audio "
        "length.",
    ),
    Entry(
        "Animated view",
        "Video",
        viz.animate_visualization,
        lambda c, p: viz.animate_visualization(c.sonifier, **p),
        choices={
            "viz_type": ["spectrogram", "waveform", "mfcc", "chromagram"]
        },
        defaults=dict(_FAST),
        seconds_per_frame=0.3,
    ),
    Entry(
        "3D spectrum waterfall",
        "Video",
        viz.create_3d_waterfall_video,
        lambda c, p: viz.create_3d_waterfall_video(c.sonifier, **p),
        defaults=dict(_FAST, duration_seconds=15.0),
        seconds_per_frame=0.16,
    ),
    Entry(
        "3D spectrogram build-up",
        "Video",
        viz.create_3d_spectrogram_buildup_video,
        lambda c, p: viz.create_3d_spectrogram_buildup_video(c.sonifier, **p),
        choices={"style": ["bars", "lines"], "colormap": CMAPS},
        defaults=dict(_FAST, duration_seconds=15.0),
        seconds_per_frame=1.1,
    ),
    Entry(
        "3D scan window",
        "Video",
        viz.create_3d_scan_video,
        lambda c, p: viz.create_3d_scan_video(c.sonifier, **p),
        needs=(MS_DATA,),
        defaults=dict(_FAST),
        seconds_per_frame=1.2,
    ),
    Entry(
        "3D rotating heatmap",
        "Video",
        viz.create_3d_heatmap_video,
        lambda c, p: viz.create_3d_heatmap_video(c.sonifier, **p),
        needs=(MS_DATA,),
        defaults=dict(_FAST),
        seconds_per_frame=0.25,
    ),
    Entry(
        "Raw data playback",
        "Video",
        viz.create_raw_data_video,
        lambda c, p: viz.create_raw_data_video(c.sonifier, **p),
        needs=(MS_DATA, AUDIO),
        defaults=dict(_FAST),
        seconds_per_frame=0.2,
        help="The data behind the sound, in sync with the audio: "
        "chromatogram with playhead, the MS1 spectrum sounding now, "
        "the nearest MS2 spectrum and the ion mobility of the ions. "
        "Switch panels on or off. If you loaded only MS1 (or only "
        "MS2), the other level is read from the file first; that "
        "can take a while for big files, once.",
    ),
    Entry(
        "Spatial stage",
        "Video",
        viz.create_spatial_stage_video,
        lambda c, p: viz.create_spatial_stage_video(c.sonifier, **p),
        needs=(MS_DATA, MOBILITY, AUDIO),
        defaults=dict(_FAST),
        seconds_per_frame=0.15,
        help="Ions as dots: left/right = stereo position, up/down = "
        "pitch, size = intensity, faint = far (reverb). Follows the "
        "audio exactly.",
    ),
    Entry(
        "Comparison",
        "Video",
        viz.create_comparison_video,
        lambda c, p: viz.create_comparison_video(c.sonifier, c.versions, **p),
        needs=(COMPARE,),
        defaults=dict(_FAST),
        seconds_per_frame=0.37,
        help="Side-by-side spectrograms of the selected versions, with "
        "their audio mixed.",
    ),
]


def skip_names() -> set:
    return set(_ALWAYS_SKIP)


def find(entries: List[Entry], label: str) -> Entry:
    return next(e for e in entries if e.label == label)


def estimated_frames(entry: Entry, params: dict, audio_seconds: float) -> int:
    """Rough frame count, for the UI's time estimate."""
    fps = float(params.get("fps") or 15)
    seconds = params.get("duration_seconds") or audio_seconds
    return max(1, int(float(seconds) * fps))


def estimated_seconds(
    entry: Entry, params: dict, audio_seconds: float
) -> float:
    """Rough render time: measured cost per frame, scaled by pixel count."""
    dpi = float(params.get("dpi") or 80)
    per_frame = entry.seconds_per_frame * (dpi / 80.0) ** 2
    return estimated_frames(entry, params, audio_seconds) * per_frame
