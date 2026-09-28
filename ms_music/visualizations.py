import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Any
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import Callable, Optional
import subprocess
import tempfile
import os
from matplotlib.colors import ListedColormap

from .utils import configure_matplotlib_ffmpeg, ffmpeg_path

# Video export uses the system ffmpeg or the bundled imageio-ffmpeg binary.
configure_matplotlib_ffmpeg()


def extract_audio_features(audio_data, sr):
    """
    Extract various audio features for comparison.

    Args:
        audio_data: Audio signal array
        sr: Sample rate

    Returns:
        dict: Dictionary of extracted features
    """
    features = {}

    # Basic statistics
    features["rms_energy"] = np.sqrt(np.mean(audio_data**2))
    features["max_amplitude"] = np.max(np.abs(audio_data))
    features["zero_crossing_rate"] = np.mean(
        librosa.zero_crossings(audio_data)
    )

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(
        y=audio_data, sr=sr
    )[0]
    features["spectral_centroid_mean"] = np.mean(spectral_centroids)
    features["spectral_centroid_std"] = np.std(spectral_centroids)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data, sr=sr
    )[0]
    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    features["mfcc_mean"] = np.mean(mfccs, axis=1)

    return features


def plot_waveform_comparison(
    audio_dict, sr, title="Waveform Comparison", figsize=None
):
    """
    Plot multiple waveforms for comparison.

    Args:
        audio_dict: Dictionary of {name: audio_array}
        sr: Sample rate
        title: Plot title
        figsize: Figure size tuple (width, height)

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_plots = len(audio_dict)
    if figsize is None:
        figsize = (14, 3 * n_plots)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    for idx, (name, audio) in enumerate(audio_dict.items()):
        time = np.linspace(0, len(audio) / sr, len(audio))
        axes[idx].plot(time, audio, alpha=0.8)
        axes[idx].set_ylabel("Amplitude")
        axes[idx].set_title(name)
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_spectrogram_comparison(
    audio_dict,
    sr,
    title="Spectrogram Comparison",
    cmap="viridis",
    figsize=None,
):
    """
    Plot multiple spectrograms side by side.

    Args:
        audio_dict: Dictionary of {name: audio_array}
        sr: Sample rate
        title: Plot title
        cmap: Colormap name
        figsize: Figure size tuple

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_plots = len(audio_dict)
    if figsize is None:
        figsize = (5 * n_plots, 4)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    if n_plots == 1:
        axes = [axes]

    for idx, (name, audio) in enumerate(audio_dict.items()):
        D = librosa.stft(audio)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            D_db, sr=sr, x_axis="time", y_axis="log", ax=axes[idx], cmap=cmap
        )
        axes[idx].set_title(name)
        if idx == 0:
            axes[idx].set_ylabel("Frequency (Hz)")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    cbar = plt.colorbar(img, ax=axes, format="%+2.0f dB")
    cbar.set_label("Magnitude (dB)")
    return fig


def plot_frequency_spectrum_comparison(
    audio_dict,
    sr,
    title="Frequency Spectrum Comparison",
    figsize=(12, 6),
    log_scale=True,
):
    """
    Plot frequency spectra for comparison.

    Args:
        audio_dict: Dictionary of {name: audio_array}
        sr: Sample rate
        title: Plot title
        figsize: Figure size tuple
        log_scale: Whether to include logarithmic x-axis

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, audio in audio_dict.items():
        # Compute FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Plot in dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        ax.plot(freqs, magnitude_db, label=name, alpha=0.3, linewidth=2)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.set_xlim(0, sr / 2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add logarithmic x-axis option
    if log_scale:
        ax2 = ax.twiny()
        ax2.set_xscale("log")
        ax2.set_xlim(20, sr / 2)
        ax2.set_xlabel("Frequency (Hz) - Log Scale")

    return fig


def plot_difference_spectrogram(
    audio1, audio2, sr, name1="Audio 1", name2="Audio 2", figsize=(15, 4)
):
    """
    Plot the difference between two spectrograms.

    Args:
        audio1: First audio array
        audio2: Second audio array
        sr: Sample rate
        name1: Name for first audio
        name2: Name for second audio
        figsize: Figure size tuple

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    # Compute spectrograms
    D1 = librosa.stft(audio1)
    D2 = librosa.stft(audio2)

    # Ensure same shape
    min_frames = min(D1.shape[1], D2.shape[1])
    D1 = D1[:, :min_frames]
    D2 = D2[:, :min_frames]

    # Convert to dB
    D1_db = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    D2_db = librosa.amplitude_to_db(np.abs(D2), ref=np.max)

    # Plot spectrograms
    librosa.display.specshow(
        D1_db, sr=sr, x_axis="time", y_axis="log", ax=axes[0], cmap="viridis"
    )
    axes[0].set_title(name1)

    librosa.display.specshow(
        D2_db, sr=sr, x_axis="time", y_axis="log", ax=axes[1], cmap="viridis"
    )
    axes[1].set_title(name2)

    # Plot difference
    diff = D2_db - D1_db
    im = librosa.display.specshow(
        diff, sr=sr, x_axis="time", y_axis="log", ax=axes[2], cmap="RdBu_r"
    )
    axes[2].set_title(f"Difference ({name2} - {name1})")

    plt.colorbar(im, ax=axes[2], format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_feature_comparison(
    feature_dict,
    title="Audio Feature Comparison",
    figsize=(12, 8),
    normalize=True,
):
    """
    Plot comparison of extracted audio features as a heatmap.

    Args:
        feature_dict: Dictionary of {method_name: features_dict}
        title: Plot title
        figsize: Figure size tuple
        normalize: Whether to normalize features

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Prepare data for plotting
    methods = list(feature_dict.keys())
    feature_names = list(next(iter(feature_dict.values())).keys())

    # Filter out MFCC (it's an array)
    scalar_features = [f for f in feature_names if f != "mfcc_mean"]

    # Create feature matrix
    feature_matrix = np.zeros((len(methods), len(scalar_features)))
    for i, method in enumerate(methods):
        for j, feature in enumerate(scalar_features):
            feature_matrix[i, j] = feature_dict[method][feature]

    # Normalize features for comparison if requested
    if normalize:
        feature_matrix_norm = (
            feature_matrix - feature_matrix.mean(axis=0)
        ) / (feature_matrix.std(axis=0) + 1e-8)
        plot_matrix = feature_matrix_norm
        cbar_label = "Normalized Value"
    else:
        plot_matrix = feature_matrix
        cbar_label = "Raw Value"

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_matrix.T,
        xticklabels=methods,
        yticklabels=scalar_features,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": cbar_label},
    )

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_chromagram_comparison(
    audio_dict, sr, title="Chromagram Comparison", figsize=None, hop_length=512
):
    """
    Plot chromagrams to visualize pitch content.

    Args:
        audio_dict: Dictionary of {name: audio_array}
        sr: Sample rate
        title: Plot title
        figsize: Figure size tuple
        hop_length: Hop length for chromagram

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_plots = len(audio_dict)
    if figsize is None:
        figsize = (12, 3 * n_plots)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    for idx, (name, audio) in enumerate(audio_dict.items()):
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, hop_length=hop_length
        )
        librosa.display.specshow(
            chroma,
            y_axis="chroma",
            x_axis="time",
            ax=axes[idx],
            hop_length=hop_length,
        )
        axes[idx].set_title(f"{name} - Chromagram")
        axes[idx].set_ylabel("Pitch Class")

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_mfcc_evolution(
    audio,
    sr,
    title="MFCC Evolution",
    n_mfcc=13,
    figsize=(12, 6),
    cmap="coolwarm",
):
    """
    Plot MFCC coefficients over time.

    Args:
        audio: Audio array
        sr: Sample rate
        title: Plot title
        n_mfcc: Number of MFCC coefficients
        figsize: Figure size tuple
        cmap: Colormap name

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(mfccs, x_axis="time", ax=ax, cmap=cmap)
    ax.set_ylabel("MFCC Coefficient")
    ax.set_title(title)
    plt.colorbar(img, ax=ax)
    plt.tight_layout()
    return fig


def plot_3d_spectrogram(
    audio,
    sr,
    title="3D Spectrogram",
    n_fft=512,
    hop_length=256,
    figsize=(12, 8),
    downsample=(4, 4),
    max_freq=None,
    cmap="viridis",
):
    """
    Create a 3D surface plot of the spectrogram.

    Args:
        audio: Audio array
        sr: Sample rate
        title: Plot title
        n_fft: FFT window size
        hop_length: Hop length
        figsize: Figure size tuple
        downsample: (freq_step, time_step) for downsampling
        max_freq: Maximum frequency to display (Hz), None for Nyquist
        cmap: Colormap name

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Validate input
    if len(audio) == 0:
        raise ValueError("Audio array is empty")

    # Compute spectrogram
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Create frequency and time arrays
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(D.shape[1]), sr=sr, hop_length=hop_length
    )

    # Apply frequency limit if specified
    if max_freq is not None:
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        D_db = D_db[freq_mask, :]

    # Downsample for performance
    step_freq, step_time = downsample
    # Ensure we don't skip too many samples
    step_freq = min(step_freq, len(freqs))
    step_time = min(step_time, len(times))

    # Downsample the data
    freqs_down = freqs[::step_freq]
    times_down = times[::step_time]
    D_db_down = D_db[::step_freq, ::step_time]

    # Create meshgrid
    # IMPORTANT: For plot_surface(X, Y, Z):
    # - X should be a 2D array where values change along axis 1 (columns)
    # - Y should be a 2D array where values change along axis 0 (rows)
    # - Z should match the shape of X and Y

    # Method 1: Using indexing='ij' for intuitive ordering
    freqs_mesh, times_mesh = np.meshgrid(freqs_down, times_down, indexing="ij")

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    # X=times, Y=freqs, Z=magnitude
    surf = ax.plot_surface(
        times_mesh,
        freqs_mesh,
        D_db_down,
        cmap=cmap,
        linewidth=0,
        antialiased=False,
        alpha=0.9,
        rcount=min(50, D_db_down.shape[0]),
        ccount=min(50, D_db_down.shape[1]),
    )

    # Set labels and formatting
    ax.set_xlabel("Time (s)", labelpad=10)
    ax.set_ylabel("Frequency (Hz)", labelpad=10)
    ax.set_zlabel("Magnitude (dB)", labelpad=10)
    ax.set_title(title, pad=20)

    # Set frequency axis to log scale for better visualization
    if freqs_down[0] > 0:  # Avoid log(0)
        ax.set_yscale("log")

    # Improve viewing angle
    ax.view_init(elev=20, azim=-60)

    # Set axis limits
    ax.set_xlim(times_down[0], times_down[-1])
    ax.set_ylim(freqs_down[0], freqs_down[-1])

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.15)
    cbar.set_label("Magnitude (dB)", rotation=270, labelpad=20)

    plt.tight_layout()
    return fig


def plot_3d_spectrogram_waterfall(
    audio,
    sr,
    title="3D Waterfall Spectrogram",
    n_fft=512,
    hop_length=256,
    figsize=(12, 8),
    num_slices=50,
    max_freq=None,
    cmap="viridis",
):
    """
    Create a 3D waterfall plot of the spectrogram.
    This is an alternative visualization that might be clearer
    than surface plot.

    Args:
        audio: Audio array
        sr: Sample rate
        title: Plot title
        n_fft: FFT window size
        hop_length: Hop length
        figsize: Figure size tuple
        num_slices: Number of time slices to show
        max_freq: Maximum frequency to display (Hz)
        cmap: Colormap name

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Compute spectrogram
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Create frequency and time arrays
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(D.shape[1]), sr=sr, hop_length=hop_length
    )

    # Apply frequency limit
    if max_freq is not None:
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        D_db = D_db[freq_mask, :]

    # Select time slices
    time_indices = np.linspace(0, len(times) - 1, num_slices, dtype=int)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Create colormap
    cmap_func = plt.get_cmap(cmap)

    # Plot each time slice
    for i, t_idx in enumerate(time_indices):
        # Get spectrum at this time
        spectrum = D_db[:, t_idx]
        time = times[t_idx]

        # Create x (frequency) and y (time) arrays
        x = freqs
        y = np.full_like(x, time)
        z = spectrum

        # Color based on time position
        color = cmap_func(i / len(time_indices))

        ax.plot(x, y, z, color=color, alpha=0.8, linewidth=1.5)

    # Set labels
    ax.set_xlabel("Frequency (Hz)", labelpad=10)
    ax.set_ylabel("Time (s)", labelpad=10)
    ax.set_zlabel("Magnitude (dB)", labelpad=10)
    ax.set_title(title, pad=20)

    # Set frequency axis to log scale if possible
    if freqs[0] > 0:
        ax.set_xscale("log")

    # Improve viewing angle
    ax.view_init(elev=20, azim=-60)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        ax=ax,
        orientation="vertical",
        pad=0.1,
        aspect=10,
    )
    cbar.set_label("Time Slice", rotation=270, labelpad=20)

    plt.tight_layout()
    return fig


def plot_mz_to_frequency_mapping(
    sonifier,
    mapping_types="all",
    freq_range=(200, 4000),
    num_scans=10,
    figsize=(14, 5),
    custom_mappings=None,
):
    """
    Visualize different m/z to frequency mapping functions.

    Args:
        sonifier: MSSonifier instance with loaded data
        mapping_types: List of mapping types to plot, or 'all' for
                        all available mappings
        freq_range: Frequency range tuple for non-linear mappings
        figsize: Figure size tuple
        custom_mappings: Dict of {'name': callable} for custom
                        mapping functions
                        Each callable should accept (mz_value, mz_min,
                        mz_max, freq_min, freq_max)

    Returns:
        matplotlib.figure.Figure: The created figure (or None if no data)
    """
    if not sonifier or not sonifier.processed_spectra_dfs:
        return None

    # Built-in mappings go through the same dispatcher sonify() uses, so the
    # curves match the audio exactly.
    def _builtin(name):
        return lambda mz: sonifier._mz_to_frequency(mz, name, freq_range)

    all_mappings = {
        name: _builtin(name) for name in sonifier.FREQUENCY_MAPPINGS
    }

    # Custom callables take (mz_value, mz_min, mz_max, freq_min, freq_max).
    if custom_mappings:
        for name, func in custom_mappings.items():
            all_mappings[name] = lambda mz, f=func: f(
                mz,
                sonifier.min_mz_overall,
                sonifier.max_mz_overall,
                freq_range[0],
                freq_range[1],
            )

    # Determine which mappings to plot
    if mapping_types == "all":
        mappings_to_plot = list(all_mappings.keys())
    elif isinstance(mapping_types, (list, tuple)):
        mappings_to_plot = [m for m in mapping_types if m in all_mappings]
        if not mappings_to_plot:
            print(f"Warning: No valid mapping types found in {mapping_types}")
            return None
    else:
        print(
            "Warning: mapping_types should be 'all' or a list of mapping names"
        )
        return None

    # Create m/z values for plotting
    mz_values = np.linspace(
        sonifier.min_mz_overall, sonifier.max_mz_overall, 1000
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Mapping functions
    for mapping_name in mappings_to_plot:
        mapping_func = all_mappings[mapping_name]
        frequencies = []

        for mz in mz_values:
            try:
                freq = mapping_func(mz)
            except Exception as e:
                print(f"Error with {mapping_name} mapping: {e}")
                freq = freq_range[0]  # Default to min frequency on error

            frequencies.append(freq)

        # Use different line styles for clarity when many mappings
        line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
        style_idx = mappings_to_plot.index(mapping_name) % len(line_styles)

        axes[0].plot(
            mz_values,
            frequencies,
            label=mapping_name.replace("_", " ").title(),
            linewidth=2,
            linestyle=line_styles[style_idx],
            alpha=0.4,
        )

    axes[0].set_xlabel("m/z Value")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("m/z to Frequency Mapping Functions")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, freq_range[1] * 1.1)

    # Plot 2: Actual data distribution
    all_mz = []
    all_intensities = []
    if num_scans is None:
        num_scans = len(sonifier.processed_spectra_dfs)
    for scan_df in sonifier.processed_spectra_dfs[:num_scans]:
        if not scan_df.empty:
            all_mz.extend(scan_df.index.tolist())
            all_intensities.extend(scan_df["intensities"].tolist())

    # Create 2D histogram
    if all_mz and all_intensities:
        hist, xedges, yedges = np.histogram2d(all_mz, all_intensities, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im = axes[1].imshow(
            hist.T, origin="lower", extent=extent, aspect="auto", cmap="YlOrRd"
        )
        axes[1].set_xlabel("m/z Value")
        axes[1].set_ylabel("Intensity")
        if num_scans == len(sonifier.processed_spectra_dfs):
            axes[1].set_title("m/z vs Intensity Distribution (All Scans)")
        else:
            axes[1].set_title(
                f"m/z vs Intensity Distribution (First {num_scans} Scans)"
            )
        plt.colorbar(im, ax=axes[1], label="Count")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No data to display",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
        axes[1].set_title("m/z vs Intensity Distribution")

    plt.tight_layout()
    return fig


def plot_scan_progression(
    sonifier, num_scans=20, figsize=(12, 8), top_n_mz=10
):
    """
    Visualize how intensity changes across scans for top m/z values.

    Args:
        sonifier: MSSonifier instance with loaded data
        num_scans: Number of scans to display
        figsize: Figure size tuple
        top_n_mz: Number of top m/z values to show

    Returns:
        matplotlib.figure.Figure: The created figure (or None if no data)
    """
    if not sonifier or not sonifier.processed_spectra_dfs:
        return None

    # Find top m/z values by total intensity
    mz_total_intensity = {}
    for scan_df in sonifier.processed_spectra_dfs:
        for mz, intensity in scan_df["intensities"].items():
            if mz not in mz_total_intensity:
                mz_total_intensity[mz] = 0
            mz_total_intensity[mz] += intensity

    # Get top m/z values
    top_mz = sorted(
        mz_total_intensity.items(), key=lambda x: x[1], reverse=True
    )[:top_n_mz]
    top_mz_values = [mz for mz, _ in top_mz]

    # Create intensity matrix
    if num_scans is None:
        num_scans_to_plot = len(sonifier.processed_spectra_dfs)
    else:
        num_scans_to_plot = min(num_scans, len(sonifier.processed_spectra_dfs))
    intensity_matrix = np.zeros((len(top_mz_values), num_scans_to_plot))

    for scan_idx in range(num_scans_to_plot):
        scan_df = sonifier.processed_spectra_dfs[scan_idx]
        for mz_idx, mz in enumerate(top_mz_values):
            if mz in scan_df.index:
                intensity_matrix[mz_idx, scan_idx] = scan_df.loc[
                    mz, "intensities"
                ]

    # Normalize each m/z row
    for i in range(len(top_mz_values)):
        max_val = np.max(intensity_matrix[i, :])
        if max_val > 0:
            intensity_matrix[i, :] /= max_val

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(intensity_matrix, aspect="auto", cmap="viridis")

    ax.set_xlabel("Scan Number")
    ax.set_ylabel("m/z Value")
    if num_scans is None:
        ax.set_title(
            f"Intensity Progression Across Scans (Top {top_n_mz} "
            f"m/z values, all scans)"
        )
    else:
        ax.set_title(
            f"Intensity Progression Across Scans (Top {top_n_mz} "
            f"m/z values, first {num_scans_to_plot} scans)"
        )

    # Set y-tick labels to m/z values
    ax.set_yticks(range(len(top_mz_values)))
    ax.set_yticklabels([f"{int(mz)}" for mz in top_mz_values])

    plt.colorbar(im, ax=ax, label="Normalized Intensity")
    plt.tight_layout()
    return fig


def _mono(audio):
    """Mono mixdown of (channels, samples) audio for analysis and drawing."""
    audio = np.asarray(audio)
    return audio.mean(axis=0) if audio.ndim > 1 else audio


def _mobility_table(sonifier):
    """(scan index, m/z, mobility, intensity) of every binned peak that has
    a mobility value, or None when the data has no ion mobility."""
    dfs = getattr(sonifier, "processed_spectra_dfs", None) or []
    parts = []
    for i, df in enumerate(dfs):
        if df.empty or "mobility" not in df.columns:
            continue
        parts.append(
            np.column_stack(
                [
                    np.full(len(df), i),
                    df.index.to_numpy(dtype=float),
                    df["mobility"].to_numpy(dtype=float),
                    df["intensities"].to_numpy(dtype=float),
                ]
            )
        )
    if not parts:
        return None
    table = np.concatenate(parts)
    return table[np.isfinite(table[:, 2])]


def _mobility_label(sonifier):
    info = getattr(sonifier, "ion_mobility_data", None) or {}
    return info.get("unit") or "Ion mobility"


def _rebin(grid, n_rows, n_cols):
    """Sum a 2D grid into at most ``n_rows`` x ``n_cols`` blocks."""
    r_idx = np.unique(np.linspace(0, grid.shape[0], n_rows + 1).astype(int))[
        :-1
    ]
    c_idx = np.unique(np.linspace(0, grid.shape[1], n_cols + 1).astype(int))[
        :-1
    ]
    return np.add.reduceat(np.add.reduceat(grid, r_idx, axis=0), c_idx, axis=1)


def _heatmap(ax, x, y, weights, bins, log_intensity, cmap):
    hist, xe, ye = np.histogram2d(x, y, bins=bins, weights=weights)
    data = np.log1p(hist.T) if log_intensity else hist.T
    return ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
    )


def plot_mobility_map(
    sonifier,
    mz_bins: int = 300,
    mobility_bins: int = 150,
    log_intensity: bool = True,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 7),
):
    """
    Heatmap of intensity over m/z and ion mobility (e.g. 1/K0).

    With per-peak mobility (e.g. timsTOF), ions of different charge states
    fall on separate diagonal bands. Otherwise the per-bin mean mobility is
    shown.

    Args:
        sonifier: MSSonifier with data loaded with ion mobility
        mz_bins: Number of m/z bins
        mobility_bins: Number of mobility bins
        log_intensity: Show log(1 + summed intensity)
        cmap: Matplotlib colormap name

    Returns:
        matplotlib.figure.Figure, or None without ion mobility data
    """
    info = getattr(sonifier, "ion_mobility_data", None) or {}
    grid = info.get("map")
    fig, ax = plt.subplots(figsize=figsize)
    if grid is not None and grid.any():
        # Raw-resolution map: every ion keeps its own mobility, so charge
        # states show as separate bands.
        rows = np.nonzero(grid.any(axis=1))[0]
        grid = grid[rows[0]: rows[-1] + 1]
        edges = info["edges"]
        data = _rebin(grid, mz_bins, mobility_bins).T
        data = np.log1p(data) if log_intensity else data
        img = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[rows[0], rows[-1] + 1, edges[0], edges[-1]],
        )
    else:
        table = _mobility_table(sonifier)
        if table is None or not len(table):
            plt.close(fig)
            return None
        img = _heatmap(
            ax,
            table[:, 1],
            table[:, 2],
            table[:, 3],
            (mz_bins, mobility_bins),
            log_intensity,
            cmap,
        )
    ax.set_xlabel("m/z")
    ax.set_ylabel(_mobility_label(sonifier))
    ax.set_title("Ion mobility map")
    fig.colorbar(
        img,
        ax=ax,
        label="log(1 + intensity)" if log_intensity else "Intensity",
    )
    fig.tight_layout()
    return fig


def plot_mobility_over_time(
    sonifier,
    time_bins: int = 300,
    mobility_bins: int = 100,
    log_intensity: bool = True,
    cmap: str = "magma",
    figsize: Tuple[int, int] = (12, 7),
):
    """
    Heatmap of intensity over retention time and ion mobility.

    Args:
        sonifier: MSSonifier with data loaded with ion mobility
        time_bins: Number of time bins
        mobility_bins: Number of mobility bins
        log_intensity: Show log(1 + summed intensity)
        cmap: Matplotlib colormap name

    Returns:
        matplotlib.figure.Figure, or None without ion mobility data
    """
    info = getattr(sonifier, "ion_mobility_data", None) or {}
    rts = getattr(sonifier, "retention_time_list", None)
    has_rt = bool(rts) and all(r is not None for r in rts)
    xlabel = "Retention time (min)" if has_rt else "Scan"
    profiles = info.get("profiles")
    fig, ax = plt.subplots(figsize=figsize)
    if profiles is not None and len(profiles):
        # Raw-resolution mobility profile of every spectrum.
        x = (
            np.asarray(rts, dtype=float)
            if has_rt
            else np.arange(len(profiles))
        )
        edges = info["edges"]
        centers = (edges[:-1] + edges[1:]) / 2
        xx = np.repeat(x, profiles.shape[1])
        yy = np.tile(centers, len(profiles))
        # No more time bins than spectra, or empty bins show as stripes.
        img = _heatmap(
            ax,
            xx,
            yy,
            profiles.ravel(),
            (
                min(time_bins, len(profiles)),
                min(mobility_bins, profiles.shape[1]),
            ),
            log_intensity,
            cmap,
        )
    else:
        table = _mobility_table(sonifier)
        if table is None or not len(table):
            plt.close(fig)
            return None
        scan = table[:, 0].astype(int)
        x = (
            np.asarray(rts, dtype=float)[scan]
            if has_rt
            else scan.astype(float)
        )
        img = _heatmap(
            ax,
            x,
            table[:, 2],
            table[:, 3],
            (min(time_bins, len(np.unique(scan))), mobility_bins),
            log_intensity,
            cmap,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_mobility_label(sonifier))
    ax.set_title("Ion mobility over time")
    fig.colorbar(
        img,
        ax=ax,
        label="log(1 + intensity)" if log_intensity else "Intensity",
    )
    fig.tight_layout()
    return fig


def plot_ion_mobilogram(
    sonifier,
    mz_ranges=None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Extracted ion mobilogram: intensity over ion mobility for m/z ranges.

    Args:
        sonifier: MSSonifier with data loaded with ion mobility
        mz_ranges: List of (low, high) m/z ranges, one line each. Default:
            the three most intense m/z values (±0.5).
        normalize: Scale each line to its own maximum

    Returns:
        matplotlib.figure.Figure, or None without ion mobility data
    """
    info = getattr(sonifier, "ion_mobility_data", None) or {}
    grid, edges = info.get("map"), info.get("edges")
    table = None
    if grid is None:
        table = _mobility_table(sonifier)
        if table is None or not len(table):
            return None
    if not mz_ranges:
        if grid is not None:
            totals = grid.sum(axis=1)
        else:
            totals = np.bincount(
                np.round(table[:, 1]).astype(int), weights=table[:, 3]
            )
        top = np.argsort(totals)[::-1][:3]
        mz_ranges = [(m - 0.5, m + 0.5) for m in sorted(top)]
    if isinstance(mz_ranges[0], (int, float)):
        mz_ranges = [tuple(mz_ranges)]

    fig, ax = plt.subplots(figsize=figsize)
    for lo, hi in mz_ranges:
        if grid is not None:
            rows = slice(
                max(int(np.floor(lo)), 0),
                min(int(np.ceil(hi)) + 1, grid.shape[0]),
            )
            profile = grid[rows].sum(axis=0)
            x = (edges[:-1] + edges[1:]) / 2
        else:
            sel = (table[:, 1] >= lo) & (table[:, 1] <= hi)
            profile, bin_edges = np.histogram(
                table[sel, 2], bins=100, weights=table[sel, 3]
            )
            x = (bin_edges[:-1] + bin_edges[1:]) / 2
        if normalize and profile.max() > 0:
            profile = profile / profile.max()
        ax.plot(x, profile, linewidth=1.5, label=f"m/z {lo:g}–{hi:g}")
        ax.fill_between(x, profile, alpha=0.15)
    ax.set_xlabel(_mobility_label(sonifier))
    ax.set_ylabel("Relative intensity" if normalize else "Intensity")
    ax.set_title("Extracted ion mobilogram")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mobility_mapping(
    sonifier,
    config=None,
    figsize: Tuple[int, int] = (12, 7),
):
    """
    How ion mobility turns into sound: stereo position, brightness and each
    mapped effect parameter across the mobility axis, over a histogram of
    where the data's ions sit.

    Args:
        sonifier: MSSonifier with data loaded with ion mobility
        config: A MobilityConfig (or dict) to preview. Default: the one used
            by the last ``sonify()``, or a preview with pan and brightness.

    Returns:
        matplotlib.figure.Figure, or None without ion mobility data
    """
    from .mobility import MobilityConfig

    table = _mobility_table(sonifier)
    if table is None or not len(table):
        return None
    config = (
        MobilityConfig.coerce(config)
        or getattr(sonifier, "mobility_config", None)
        or MobilityConfig(brightness=0.6, pan=1.0)
    )
    if config.range is not None:
        lo, hi = config.range
    elif getattr(sonifier, "mobility_norm", None):
        lo, hi = sonifier.mobility_norm
    else:
        lo, hi = np.percentile(table[:, 2], [2, 98])
    pad = 0.1 * (hi - lo)
    x = np.linspace(lo - pad, hi + pad, 400)
    b = np.clip((x - lo) / (hi - lo), 0, 1)

    rows = 1 + len(config.effects)
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(figsize[0], figsize[1] * (0.6 + 0.4 * rows) / 1.4),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    def backdrop(ax):
        twin = ax.twinx()
        twin.hist(
            table[:, 2],
            bins=120,
            weights=table[:, 3],
            color="grey",
            alpha=0.25,
        )
        twin.set_yticks([])
        ax.set_zorder(twin.get_zorder() + 1)
        ax.patch.set_visible(False)
        for edge in (lo, hi):
            ax.axvline(edge, color="grey", linestyle=":", linewidth=1)

    ax = axes[0]
    backdrop(ax)
    if config.pan > 0:
        ax.plot(
            x,
            config.pan_position(b),
            label="Stereo position " "(-1 left … +1 right)",
            linewidth=2,
        )
    if config.brightness > 0:
        ax.plot(
            x,
            config.brightness * b,
            label="Brightness (overtones)",
            linewidth=2,
        )
    if not (config.pan > 0 or config.brightness > 0):
        ax.text(
            0.5,
            0.5,
            "No pan or brightness set",
            ha="center",
            transform=ax.transAxes,
            color="grey",
        )
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left")
    ax.set_title("Ion mobility → sound (grey: where the data's ions are)")

    for ax, effect in zip(axes[1:], config.effects):
        backdrop(ax)
        # Each band applies its own setting; tones crossfade between the
        # two nearest bands.
        centres = np.linspace(0, 1, config.bands)
        values = [effect.value_at(c) for c in centres]
        band_x = lo + centres * (hi - lo)
        from .effect_guide import parameter_guide

        guide = parameter_guide(effect.effect, effect.param)
        unit = f" ({guide.unit})" if guide is not None and guide.unit else ""
        ax.step(
            band_x,
            values,
            where="mid",
            linewidth=2,
            label=f"{effect.effect}: {effect.param}{unit}",
        )
        ax.plot(band_x, values, "o")
        ax.set_ylabel(f"{effect.param.replace('_', ' ')}{unit}")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel(_mobility_label(sonifier))
    fig.tight_layout()
    return fig


def plot_stereo_field(
    audio_data: np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    position_bins: int = 41,
    log_intensity: bool = True,
    cmap: str = "magma",
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Where the sound sits between left and right over time ("panogram").

    Each time-frequency cell's energy is placed at its stereo position
    (R² − L²)/(R² + L²), from -1 (left) to +1 (right).

    Args:
        audio_data: Stereo audio, shape (2, samples)
        sample_rate: Audio sample rate
        n_fft, hop_length: STFT settings
        position_bins: Number of left-right bins
        log_intensity: Show log(1 + energy)

    Returns:
        matplotlib.figure.Figure
    """
    audio = np.asarray(audio_data)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("plot_stereo_field needs stereo audio (2, samples).")
    left = (
        np.abs(librosa.stft(audio[0], n_fft=n_fft, hop_length=hop_length)) ** 2
    )
    right = (
        np.abs(librosa.stft(audio[1], n_fft=n_fft, hop_length=hop_length)) ** 2
    )
    energy = left + right
    with np.errstate(invalid="ignore", divide="ignore"):
        position = np.where(energy > 0, (right - left) / energy, 0.0)
    n_frames = energy.shape[1]
    edges = np.linspace(-1, 1, position_bins + 1)
    idx = np.clip(np.digitize(position, edges) - 1, 0, position_bins - 1)
    field_map = np.zeros((position_bins, n_frames))
    for frame in range(n_frames):
        field_map[:, frame] = np.bincount(
            idx[:, frame], weights=energy[:, frame], minlength=position_bins
        )
    data = np.log1p(field_map) if log_intensity else field_map
    duration = audio.shape[1] / sample_rate
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[0, duration, -1, 1],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stereo position")
    ax.set_yticks([-1, 0, 1], ["Left", "Centre", "Right"])
    ax.set_title("Stereo field")
    fig.colorbar(
        img, ax=ax, label="log(1 + energy)" if log_intensity else "Energy"
    )
    fig.tight_layout()
    return fig


def create_spatial_stage_video(
    sonifier,
    output_path: str,
    fps: int = 15,
    dpi: int = 80,
    trail: float = 0.4,
    max_points: int = 400,
    point_scale: float = 1.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Animated stage of the ions as they sound.

    Left/right is each ion's stereo position (from its mobility), up/down is
    the pitch it plays, dot size is its intensity, and ions placed "far"
    (the first mapped wet/dry-type effect parameter, e.g. reverb wet) are
    drawn smaller and fainter. Use after ``sonify(mobility=...)`` on data
    with ion mobility; the video follows the audio exactly and carries it
    (stereo if the audio is stereo).

    Args:
        sonifier: MSSonifier after sonify()
        output_path: Output .mp4 path
        fps: Frames per second
        dpi: Resolution
        trail: Seconds of fading trail behind each moment
        max_points: Most intense ions drawn per moment
        point_scale: Dot size multiplier
        progress_callback: Called as ``progress_callback(frame, total)``
            while frames render; raise from it to abort.

    Returns:
        str: Path to the created video, or None on failure
    """
    from .mobility import MobilityConfig

    audio = getattr(sonifier, "current_audio_data", None)
    spectra = getattr(sonifier, "segment_spectra", None)
    starts = getattr(sonifier, "segment_starts", None)
    freq_fn = getattr(sonifier, "frequency_fn", None)
    if audio is None or not spectra or starts is None or freq_fn is None:
        print("Run sonify() first.")
        return None
    config = getattr(sonifier, "mobility_config", None) or MobilityConfig(
        pan=1.0
    )
    norm = getattr(sonifier, "mobility_norm", None)
    if norm is None:
        table = _mobility_table(sonifier)
        if table is None or not len(table):
            print("The data has no ion mobility.")
            return None
        norm = tuple(np.percentile(table[:, 2], [2, 98]))
    lo, hi = norm
    depth_effect = next(
        (e for e in config.effects if "wet" in e.param or "mix" in e.param),
        None,
    )

    frames_pos = []  # per segment: (x, log2 pitch, size, alpha)
    max_i = (
        max(float(df["intensities"].max()) for df in spectra if not df.empty)
        or 1.0
    )
    for df in spectra:
        if df.empty or "mobility" not in df.columns:
            frames_pos.append(None)
            continue
        df = df.nlargest(max_points, "intensities")
        b = np.nan_to_num(
            np.clip((df["mobility"].to_numpy() - lo) / (hi - lo), 0, 1),
            nan=0.5,
        )
        x = config.pan_position(b) if config.pan > 0 else 2 * b - 1
        freqs = np.array(
            [freq_fn(mz) for mz in df.index.to_numpy(dtype=float)]
        )
        ok = freqs > 0
        if depth_effect is not None and depth_effect.high != depth_effect.low:
            far = np.array(
                [
                    (depth_effect.value_at(v) - depth_effect.low)
                    / (depth_effect.high - depth_effect.low)
                    for v in b
                ]
            )
        else:
            far = np.zeros(len(b))
        size = (
            20 + 400 * point_scale * df["intensities"].to_numpy() / max_i
        ) * (1 - 0.6 * far)
        alpha = 0.95 - 0.7 * far
        frames_pos.append(
            (np.asarray(x)[ok], np.log2(freqs[ok]), size[ok], alpha[ok])
        )

    pitches = np.concatenate(
        [f[1] for f in frames_pos if f is not None]
        or [np.log2([200.0, 4000.0])]
    )
    duration = _mono(audio).shape[-1] / sonifier.sample_rate
    n_frames = max(1, int(duration * fps))
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0b0b14")
    ax.set_facecolor("#0b0b14")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(pitches.min() - 0.2, pitches.max() + 0.2)
    ax.set_xticks([-1, 0, 1], ["Left", "Centre", "Right"], color="#bbb")
    octaves = np.arange(np.ceil(pitches.min()), np.floor(pitches.max()) + 1)
    ax.set_yticks(octaves, [f"{2 ** o:.0f} Hz" for o in octaves], color="#bbb")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.set_ylabel("Pitch", color="#bbb")
    subtitle = "compact ions left, extended ions right" + (
        "; far = smaller and fainter" if depth_effect else ""
    )
    ax.set_title(f"Ion mobility stage: {subtitle}", color="#ddd")
    colour = np.array([0.99, 0.75, 0.3])
    trail_scatter = ax.scatter([], [], s=[], linewidths=0)
    scatter = ax.scatter([], [], s=[], linewidths=0)
    clock = ax.text(
        0.99, 0.01, "", transform=ax.transAxes, ha="right", color="#888"
    )

    def segment_at(t):
        return int(
            np.clip(
                np.searchsorted(starts, t, side="right") - 1,
                0,
                len(frames_pos) - 1,
            )
        )

    def rgba(alpha):
        out = np.tile(np.append(colour, 1.0), (len(alpha), 1))
        out[:, 3] = alpha
        return out

    def animate(frame):
        t = frame / fps
        now = frames_pos[segment_at(t)]
        if now is None or not len(now[0]):
            scatter.set_offsets(np.empty((0, 2)))
        else:
            scatter.set_offsets(np.column_stack(now[:2]))
            scatter.set_sizes(now[2])
            scatter.set_facecolors(rgba(now[3]))
        past = [
            frames_pos[i]
            for i in range(segment_at(max(0, t - trail)), segment_at(t))
            if frames_pos[i] is not None and len(frames_pos[i][0])
        ]
        if past:
            trail_scatter.set_offsets(
                np.column_stack(
                    (
                        np.concatenate([p[0] for p in past]),
                        np.concatenate([p[1] for p in past]),
                    )
                )
            )
            trail_scatter.set_sizes(np.concatenate([p[2] for p in past]) * 0.5)
            trail_scatter.set_facecolors(
                rgba(np.concatenate([p[3] for p in past]) * 0.25)
            )
        else:
            trail_scatter.set_offsets(np.empty((0, 2)))
        clock.set_text(f"{t:5.1f} s")
        return scatter, trail_scatter, clock

    anim = FuncAnimation(
        fig, animate, frames=n_frames, interval=1000 / fps, blit=False
    )
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(
            output_path,
            writer=writer,
            dpi=dpi,
            savefig_kwargs={"facecolor": fig.get_facecolor()},
            progress_callback=progress_callback,
        )
        plt.close(fig)
        _add_audio_to_video(
            output_path, audio, sonifier.sample_rate, output_path
        )
        print(f"Video saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating video: {e}")
        plt.close(fig)
        return None


def _companion_run(sonifier, ms_level):
    """Spectra of another MS level from the sonifier's file, for the same
    retention time window (cached like any load). None if unavailable."""
    from . import io as ms_io

    path = getattr(sonifier, "filepath", "")
    rts = [
        r
        for r in (getattr(sonifier, "retention_time_list", None) or [])
        if r is not None
    ]
    if not path or not os.path.exists(path) or not rts:
        return None
    try:
        return ms_io.load_spectra(
            path,
            ms_level=ms_level,
            rt_range=(min(rts) - 0.01, max(rts) + 0.01),
        )
    except Exception as exc:
        print(f"Could not load MS{ms_level} spectra: {exc}")
        return None


def create_raw_data_video(
    sonifier,
    output_path: str,
    show_chromatogram: bool = True,
    show_ms1: bool = True,
    show_ms2: bool = True,
    show_mobility: bool = True,
    fps: int = 15,
    dpi: int = 80,
    spectrum_peaks: int = 300,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Video of the raw data behind the sound, in sync with the audio.

    Panels (each optional):

    - **Chromatogram**: total ion current over retention time, with a
      playhead at the spectra currently sounding.
    - **MS1 spectrum**: the spectrum currently sounding (relative
      intensity); the precursor of the MS2 panel's scan is marked.
    - **MS2 spectrum**: the fragment spectrum nearest the playhead, with its
      precursor m/z and charge.
    - **Ion mobility**: the current spectrum's ions as m/z x mobility dots
      (size = intensity) over the whole run's mobility map.

    If only one MS level was loaded, the other is read from the same file
    for the same retention time window (cached after the first time).

    Args:
        sonifier: MSSonifier after sonify() on mzML data
        output_path: Output .mp4 path
        show_chromatogram, show_ms1, show_ms2, show_mobility: Panels to show
        fps: Frames per second
        dpi: Resolution
        spectrum_peaks: Most intense peaks drawn per spectrum
        progress_callback: Called as ``progress_callback(frame, total)``
            while frames render; raise from it to abort.

    Returns:
        str: Path to the created video, or None on failure
    """
    from matplotlib.collections import LineCollection

    audio = getattr(sonifier, "current_audio_data", None)
    starts = getattr(sonifier, "segment_starts", None)
    members = getattr(sonifier, "segment_members", None)
    spectra = getattr(sonifier, "processed_spectra_dfs", None)
    rts = getattr(sonifier, "retention_time_list", None)
    if audio is None or starts is None or members is None or not spectra:
        print("Run sonify() on mzML data first.")
        return None
    if not rts or any(r is None for r in rts):
        print("The data has no retention times (raw data video needs mzML).")
        return None
    if not (show_chromatogram or show_ms1 or show_ms2 or show_mobility):
        raise ValueError("Choose at least one panel to show.")

    level = getattr(sonifier, "ms_level", 1)
    rts = np.asarray(rts, dtype=float)

    # Scans for each MS level: the loaded ones, the other from the file.
    def as_level(run):
        if run is None:
            return None
        return {
            "dfs": run.spectra,
            "rt": np.asarray(run.retention_times, float),
            "prec": run.precursor_mz,
            "charge": run.charges,
        }

    loaded = {
        "dfs": spectra,
        "rt": rts,
        "prec": getattr(sonifier, "precursor_mz_list", None)
        or [None] * len(spectra),
        "charge": getattr(sonifier, "charge_list", None)
        or [None] * len(spectra),
    }
    ms1 = (
        loaded
        if level == 1
        else (
            as_level(_companion_run(sonifier, 1))
            if (show_ms1 or show_chromatogram or show_mobility)
            else None
        )
    )
    ms2 = (
        loaded
        if level >= 2
        else (as_level(_companion_run(sonifier, 2)) if show_ms2 else None)
    )

    # Audio segment -> retention time of the scans sounding in it.
    seg_rt = np.array([rts[m].mean() if len(m) else np.nan for m in members])
    seg_rt = pd.Series(seg_rt).ffill().bfill().to_numpy()
    duration = _mono(audio).shape[-1] / sonifier.sample_rate
    n_frames = max(1, int(duration * fps))

    def segment_at(t):
        return int(
            np.clip(
                np.searchsorted(starts, t, side="right") - 1,
                0,
                len(members) - 1,
            )
        )

    def nearest(run, rt, before=False):
        if run is None or not len(run["rt"]):
            return None
        i = int(np.searchsorted(run["rt"], rt))
        if before:
            return max(i - 1, 0)
        if i >= len(run["rt"]):
            return len(run["rt"]) - 1
        if i > 0 and rt - run["rt"][i - 1] < run["rt"][i] - rt:
            return i - 1
        return i

    def top_peaks(df):
        if df is None or df.empty:
            return np.array([]), np.array([]), None
        df = df.nlargest(spectrum_peaks, "intensities")
        mob = df["mobility"].to_numpy() if "mobility" in df.columns else None
        return (
            df.index.to_numpy(dtype=float),
            df["intensities"].to_numpy(dtype=float),
            mob,
        )

    # ---------------------------------------------------------- layout
    panels = [
        name
        for name, on in (
            ("ms1", show_ms1),
            ("ms2", show_ms2),
            ("mobility", show_mobility),
        )
        if on
    ]
    rows = (1 if show_chromatogram else 0) + (len(panels) + 1) // 2
    fig = plt.figure(figsize=(14, 3.2 * rows + 0.6))
    grid = GridSpec(rows, 2, figure=fig, hspace=0.55, wspace=0.18)
    axes = {}
    row = 0
    if show_chromatogram:
        axes["chrom"] = fig.add_subplot(grid[0, :])
        row = 1
    for i, name in enumerate(panels):
        r, c = row + i // 2, i % 2
        span = (
            grid[r, :]
            if (i == len(panels) - 1 and len(panels) % 2)
            else grid[r, c]
        )
        axes[name] = fig.add_subplot(span)
    mz_lo, mz_hi = sonifier.min_mz_overall, sonifier.max_mz_overall
    title = fig.suptitle("", fontsize=12)

    # Chromatogram (TIC of MS1 when available)
    if "chrom" in axes:
        ax = axes["chrom"]
        src = ms1 or loaded
        tic = np.array(
            [
                df["intensities"].sum() if not df.empty else 0
                for df in src["dfs"]
            ]
        )
        ax.plot(src["rt"], tic, color="steelblue", linewidth=1)
        ax.fill_between(src["rt"], tic, color="steelblue", alpha=0.2)
        ax.set_xlim(src["rt"].min(), src["rt"].max())
        ax.set_xlabel("Retention time (min)")
        ax.set_ylabel("Total ion current")
        ax.set_title(f"Chromatogram (MS{'1' if src is ms1 else level} TIC)")
        playhead = ax.axvline(src["rt"][0], color="crimson", linewidth=1.5)
    else:
        playhead = None

    def stem_axes(ax, label):
        ax.set_xlim(mz_lo, mz_hi)
        ax.set_ylim(0, 105)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative intensity (%)")
        ax.set_title(label)
        lines = LineCollection([], linewidths=0.8)
        ax.add_collection(lines)
        return lines

    ms1_lines = ms2_lines = prec_marker = None
    ms2_text = None
    if "ms1" in axes:
        ms1_lines = stem_axes(axes["ms1"], "MS1 spectrum (sounding now)")
        ms1_lines.set_color("#1f77b4")
        prec_marker = axes["ms1"].axvline(
            np.nan,
            color="crimson",
            linestyle="--",
            linewidth=1,
            label="MS2 precursor",
        )
        if ms1 is None:
            axes["ms1"].text(
                0.5,
                0.5,
                "No MS1 spectra in this file",
                transform=axes["ms1"].transAxes,
                ha="center",
            )
    if "ms2" in axes:
        ms2_lines = stem_axes(axes["ms2"], "MS2 spectrum")
        ms2_lines.set_color("#d62728")
        ms2_text = axes["ms2"].text(
            0.99,
            0.95,
            "",
            transform=axes["ms2"].transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        if ms2 is None:
            axes["ms2"].text(
                0.5,
                0.5,
                "No MS2 spectra in this file",
                transform=axes["ms2"].transAxes,
                ha="center",
            )

    mob_scatter = None
    if "mobility" in axes:
        ax = axes["mobility"]
        info = getattr(sonifier, "ion_mobility_data", None) or {}
        grid_map, edges = info.get("map"), info.get("edges")
        if grid_map is not None and grid_map.any():
            ax.imshow(
                np.log1p(grid_map.T),
                origin="lower",
                aspect="auto",
                cmap="Greys",
                alpha=0.5,
                extent=[0, grid_map.shape[0], edges[0], edges[-1]],
            )
            ax.set_ylim(edges[0], edges[-1])
        elif info.get("range"):
            ax.set_ylim(*info["range"])
        ax.set_xlim(mz_lo, mz_hi)
        ax.set_xlabel("m/z")
        ax.set_ylabel(_mobility_label(sonifier))
        ax.set_title("Ion mobility (sounding now)")
        mob_scatter = ax.scatter(
            [], [], s=[], c="darkorange", alpha=0.8, linewidths=0
        )
        if not info:
            ax.text(
                0.5,
                0.5,
                "No ion mobility in this file",
                transform=ax.transAxes,
                ha="center",
            )

    def stems(mz, inten):
        if not len(mz):
            return []
        rel = 100 * inten / inten.max()
        return [[(m, 0), (m, r)] for m, r in zip(mz, rel)]

    def animate(frame):
        t = frame / fps
        k = segment_at(t)
        rt_now = seg_rt[k]
        title.set_text(f"{t:5.1f} s of audio  ·  RT {rt_now:.2f} min")
        if playhead is not None:
            playhead.set_xdata([rt_now, rt_now])

        # MS1: what is sounding (loaded MS1), else the nearest MS1 scan.
        if level == 1:
            ms1_df = sonifier.segment_spectra[k]
        else:
            i = nearest(ms1, rt_now)
            ms1_df = ms1["dfs"][i] if i is not None else None
        mz1, in1, mob1 = top_peaks(ms1_df)
        if ms1_lines is not None:
            ms1_lines.set_segments(stems(mz1, in1))

        # MS2: the most intense scan sounding (loaded MS2), else the last
        # MS2 scan acquired up to now.
        j = None
        if ms2 is not None:
            if level >= 2 and len(members[k]):
                j = int(
                    members[k][
                        np.argmax(
                            [
                                (
                                    ms2["dfs"][m]["intensities"].sum()
                                    if not ms2["dfs"][m].empty
                                    else 0
                                )
                                for m in members[k]
                            ]
                        )
                    ]
                )
            else:
                j = nearest(ms2, rt_now, before=True)
        if ms2_lines is not None and j is not None:
            mz2, in2, _ = top_peaks(ms2["dfs"][j])
            ms2_lines.set_segments(stems(mz2, in2))
            prec, charge = ms2["prec"][j], ms2["charge"][j]
            label = f"RT {ms2['rt'][j]:.2f} min"
            if prec is not None:
                label += f"  ·  precursor m/z {prec:.2f}"
            if charge:
                label += f"  ·  z = {int(charge)}"
            ms2_text.set_text(label)
            if prec_marker is not None:
                prec_marker.set_xdata(
                    [prec, prec] if prec is not None else [np.nan, np.nan]
                )

        if mob_scatter is not None:
            if mob1 is not None and len(mz1):
                ok = np.isfinite(mob1)
                mob_scatter.set_offsets(np.column_stack((mz1[ok], mob1[ok])))
                mob_scatter.set_sizes(8 + 200 * in1[ok] / in1.max())
            else:
                mob_scatter.set_offsets(np.empty((0, 2)))
        return ()

    anim = FuncAnimation(
        fig, animate, frames=n_frames, interval=1000 / fps, blit=False
    )
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2500)
        anim.save(
            output_path,
            writer=writer,
            dpi=dpi,
            progress_callback=progress_callback,
        )
        plt.close(fig)
        _add_audio_to_video(
            output_path, audio, sonifier.sample_rate, output_path
        )
        print(f"Video saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating video: {e}")
        plt.close(fig)
        return None


def plot_precursor_map(
    sonifier,
    color_by: str = "charge",
    point_size: float = 4.0,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 7),
):
    """
    Where MS2 precursors were picked: retention time against precursor m/z.

    Args:
        sonifier: MSSonifier with MS2 data loaded
        color_by: 'charge', 'intensity' (log of the scan's summed
            intensity) or 'mobility' (the scan's ion mobility)
        point_size: Marker size
        cmap: Matplotlib colormap for 'intensity' and 'mobility'

    Returns:
        matplotlib.figure.Figure, or None without precursor information
    """
    precursors = getattr(sonifier, "precursor_mz_list", None) or []
    rts = getattr(sonifier, "retention_time_list", None) or []
    rows = [
        (i, rts[i] if i < len(rts) and rts[i] is not None else i, p)
        for i, p in enumerate(precursors)
        if p is not None
    ]
    if not rows:
        return None
    idx = np.array([r[0] for r in rows])
    x = np.array([r[1] for r in rows], dtype=float)
    y = np.array([r[2] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    if color_by == "charge":
        charges = getattr(sonifier, "charge_list", None) or []
        c = np.array(
            [charges[i] if i < len(charges) and charges[i] else 0 for i in idx]
        )
        for z in sorted(set(c)):
            sel = c == z
            ax.scatter(
                x[sel],
                y[sel],
                s=point_size,
                alpha=0.6,
                label=f"z = {z}" if z else "unknown charge",
            )
        ax.legend(markerscale=3, loc="upper left")
    else:
        if color_by == "mobility":
            info = getattr(sonifier, "ion_mobility_data", None) or {}
            per = info.get("per_spectrum") or []
            values = np.array(
                [
                    per[i] if i < len(per) and per[i] is not None else np.nan
                    for i in idx
                ],
                dtype=float,
            )
            label = _mobility_label(sonifier)
        else:
            dfs = sonifier.processed_spectra_dfs
            values = np.log1p([dfs[i]["intensities"].sum() for i in idx])
            label = "log(1 + scan intensity)"
        sc = ax.scatter(x, y, c=values, s=point_size, cmap=cmap, alpha=0.7)
        fig.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("Retention time (min)" if rts else "Scan")
    ax.set_ylabel("Precursor m/z")
    ax.set_title(f"Precursor map ({len(rows):,} MS2 scans)")
    fig.tight_layout()
    return fig


def compute_audio_similarity_matrix(audio_dict, sr):
    """
    Compute similarity matrices between different audio samples.

    Args:
        audio_dict: Dictionary of {name: audio_array}
        sr: Sample rate

    Returns:
        tuple: (methods_list, correlation_matrix, spectral_similarity_matrix)
    """
    methods = list(audio_dict.keys())
    n_methods = len(methods)

    # Initialize similarity matrices
    correlation_matrix = np.zeros((n_methods, n_methods))
    spectral_similarity_matrix = np.zeros((n_methods, n_methods))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            audio1 = audio_dict[method1]
            audio2 = audio_dict[method2]

            # Ensure same length
            min_len = min(len(audio1), len(audio2))
            audio1_trim = audio1[:min_len]
            audio2_trim = audio2[:min_len]

            # Time-domain correlation
            if np.std(audio1_trim) > 0 and np.std(audio2_trim) > 0:
                corr, _ = pearsonr(audio1_trim, audio2_trim)
                correlation_matrix[i, j] = corr
            else:
                correlation_matrix[i, j] = 0

            # Spectral similarity (cosine similarity of magnitude spectra)
            fft1 = np.abs(np.fft.rfft(audio1_trim))
            fft2 = np.abs(np.fft.rfft(audio2_trim))

            if np.linalg.norm(fft1) > 0 and np.linalg.norm(fft2) > 0:
                spectral_similarity = np.dot(fft1, fft2) / (
                    np.linalg.norm(fft1) * np.linalg.norm(fft2)
                )
                spectral_similarity_matrix[i, j] = spectral_similarity
            else:
                spectral_similarity_matrix[i, j] = 0

    return methods, correlation_matrix, spectral_similarity_matrix


def plot_similarity_matrices(
    methods, corr_matrix, spec_matrix, figsize=(16, 6)
):
    """
    Plot similarity matrices as heatmaps.

    Args:
        methods: List of method names
        corr_matrix: Correlation matrix
        spec_matrix: Spectral similarity matrix
        figsize: Figure size tuple

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Time-domain correlation
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        xticklabels=methods,
        yticklabels=methods,
        ax=axes[0],
        vmin=-1,
        vmax=1,
        square=True,
    )
    axes[0].set_title("Time-Domain Correlation")

    # Spectral similarity
    sns.heatmap(
        spec_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=methods,
        yticklabels=methods,
        ax=axes[1],
        vmin=0,
        vmax=1,
        square=True,
    )
    axes[1].set_title("Spectral Similarity (Cosine)")

    plt.tight_layout()
    return fig


def plot_audio_envelope_comparison(
    audio_dict,
    sr,
    title="Amplitude Envelope Comparison",
    figsize=(12, 6),
    frame_length=2048,
    hop_length=512,
):
    """
    Plot amplitude envelopes of multiple audio signals.

    Args:
        audio_dict: Dictionary of {name: audio_array}
        sr: Sample rate
        title: Plot title
        figsize: Figure size tuple
        frame_length: Frame length for RMS calculation
        hop_length: Hop length for RMS calculation

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, audio in audio_dict.items():
        # Calculate RMS envelope
        rms = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )

        ax.plot(times, rms, label=name, alpha=0.3, linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_summary_grid(
    sonifier, audio_dict, sr, title="MS Music Summary", figsize=(16, 12)
):
    """
    Create a comprehensive summary grid visualization.

    Args:
        sonifier: MSSonifier instance
        audio_dict: Dictionary of audio samples to compare
        sr: Sample rate
        title: Overall title
        figsize: Figure size tuple

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Data statistics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if sonifier and sonifier.processed_spectra_dfs:
        stats_text = (
            f"MS Level: {sonifier.ms_level}\n"
            f"Scans: {len(sonifier.processed_spectra_dfs)}\n"
            f"m/z range: {sonifier.min_mz_overall:.1f}-"
            f"{sonifier.max_mz_overall:.1f}\n"
            f"Max intensity: {sonifier.max_intensity_overall:.2e}"
        )
        ax1.text(
            0.1,
            0.5,
            stats_text,
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="center",
        )
    ax1.set_title("Data Statistics")
    ax1.axis("off")

    # 2. Waveform comparison (top middle and right)
    ax2 = fig.add_subplot(gs[0, 1:])
    # Select first 3 methods for clarity
    selected_audio = dict(list(audio_dict.items())[:3])
    for name, audio in selected_audio.items():
        time = np.linspace(0, len(audio) / sr, len(audio))
        ax2.plot(time, audio, label=name, alpha=0.3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Waveform Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Spectrograms (middle row)
    for idx, (name, audio) in enumerate(list(audio_dict.items())[:3]):
        ax = fig.add_subplot(gs[1, idx])
        D = librosa.stft(audio)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(
            D_db, sr=sr, x_axis="time", y_axis="log", ax=ax
        )
        ax.set_title(name)
        if idx == 0:
            ax.set_ylabel("Frequency (Hz)")

    # 4. Feature comparison (bottom row)
    ax4 = fig.add_subplot(gs[2, :])
    feature_dict = {}
    for name, audio in audio_dict.items():
        if audio is not None and audio.size > 0:
            features = extract_audio_features(audio, sr)
            # Select subset of features for visualization
            feature_dict[name] = {
                "RMS Energy": features["rms_energy"],
                "Spectral Centroid": features["spectral_centroid_mean"],
                "Zero Crossing Rate": features["zero_crossing_rate"],
            }

    df = pd.DataFrame(feature_dict).T
    df_normalized = (df - df.mean()) / df.std()

    df_normalized.plot(kind="bar", ax=ax4)
    ax4.set_xlabel("Method")
    ax4.set_ylabel("Normalized Value")
    ax4.set_title("Feature Comparison")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def analyze_audio(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Analyze audio data and extract various features.

    Args:
        audio_data: Audio signal array

    Returns:
        Dictionary of audio features
    """
    if audio_data.size == 0:
        return {}

    analysis = {}

    # Basic statistics
    analysis["duration"] = len(audio_data) / sample_rate
    analysis["max_amplitude"] = np.max(np.abs(audio_data))
    analysis["rms_energy"] = np.sqrt(np.mean(audio_data**2))
    analysis["dynamic_range"] = np.max(audio_data) - np.min(audio_data)

    # Zero crossing rate
    analysis["zero_crossing_rate"] = np.mean(
        librosa.zero_crossings(audio_data)
    )

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(
        y=audio_data, sr=sample_rate
    )[0]
    analysis["spectral_centroid_mean"] = np.mean(spectral_centroids)
    analysis["spectral_centroid_std"] = np.std(spectral_centroids)

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio_data, sr=sample_rate
    )[0]
    analysis["spectral_rolloff_mean"] = np.mean(spectral_rolloff)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data, sr=sample_rate
    )[0]
    analysis["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    analysis["mfcc_mean"] = np.mean(mfccs, axis=1)
    analysis["mfcc_std"] = np.std(mfccs, axis=1)

    return analysis


def plot_audio_comparison(
    audio_dict: Dict[str, np.ndarray],
    sample_rate: int = 44100,
    title: str = "Audio Comparison",
    figsize: Tuple[int, int] = (12, 16),
):
    """
    Plot multiple audio waveforms for comparison.

    Args:
        audio_dict: Dictionary of {name: audio_array}
        sample_rate: Audio sample rate
        title: Plot title
    """
    n_plots = len(audio_dict)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    if n_plots == 1:
        axes = [axes]

    for idx, (name, audio) in enumerate(audio_dict.items()):
        time = np.linspace(0, len(audio) / sample_rate, len(audio))
        axes[idx].plot(time, audio, alpha=0.8)
        axes[idx].set_ylabel("Amplitude")
        axes[idx].set_title(name)
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig


def plot_spectrogram(
    audio_data: np.ndarray,
    sample_rate: int = 44100,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (12, 8),
    n_fft: int = 2048,
    hop_length: int = 512,
    y_axis: str = "log",
    cmap: str = "magma",
):
    """
    Plot spectrogram of audio data.

    Args:
        audio_data: Audio signal array
        sample_rate: Audio sample rate
        title: Plot title
        n_fft: FFT window size (frequency resolution)
        hop_length: Samples between frames (time resolution)
        y_axis: Frequency axis scale: 'log', 'linear' or 'mel'
        cmap: Matplotlib colormap name
    """
    fig, ax = plt.subplots(figsize=figsize)

    D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    img = librosa.display.specshow(
        D_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis=y_axis,
        ax=ax,
        cmap=cmap,
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.show()

    return fig


def create_video(
    sonifier,
    output_path: str,
    fps: int = 30,
    dpi: int = 100,
    show_spectrogram: bool = True,
    show_waveform: bool = True,
    show_mz_distribution: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Create video visualization (original higher quality version).

    Args:
        sonifier: MSSonifier object
        output_path: Path for output video file (.mp4)
        fps: Frames per second
        dpi: Resolution
        show_spectrogram: Show frequency spectrogram
        show_waveform: Show audio waveform
        show_mz_distribution: Show m/z distribution over time
        progress_callback: Called as ``progress_callback(frame, total)``
            while frames render; raise from it to abort.

    Returns:
        str: Path to created video file
    """
    if sonifier.current_audio_data is None:
        print("No audio data. Run sonification first.")
        return None

    audio = sonifier.current_audio_data
    if audio.ndim > 1:
        audio = _mono(audio)

    sample_rate = sonifier.sample_rate
    duration = len(audio) / sample_rate

    # Compute spectrogram
    D = librosa.stft(audio, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Setup figure
    num_panels = sum([show_spectrogram, show_waveform, show_mz_distribution])
    if num_panels == 0:
        print("No panels selected for video")
        return None

    fig = plt.figure(figsize=(14, 4 * num_panels))
    gs = GridSpec(num_panels, 1, hspace=0.3)

    axes = []
    panel_idx = 0

    # Spectrogram
    if show_spectrogram:
        ax_spec = fig.add_subplot(gs[panel_idx])
        axes.append(ax_spec)
        panel_idx += 1

        img = librosa.display.specshow(
            S_db,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="log",
            ax=ax_spec,
            cmap="magma",
        )
        ax_spec.set_title("Spectrogram", fontsize=14, fontweight="bold")
        ax_spec.set_ylabel("Frequency (Hz)", fontsize=11)
        fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")
        line_spec = ax_spec.axvline(x=0, color="cyan", linewidth=2, alpha=0.8)

    # Waveform
    if show_waveform:
        ax_wave = fig.add_subplot(gs[panel_idx])
        axes.append(ax_wave)
        panel_idx += 1

        times_wave = np.arange(len(audio)) / sample_rate
        ax_wave.plot(
            times_wave, audio, color="steelblue", linewidth=0.5, alpha=0.7
        )
        ax_wave.set_xlabel("Time (s)", fontsize=11)
        ax_wave.set_ylabel("Amplitude", fontsize=11)
        ax_wave.set_xlim(0, duration)
        ax_wave.set_ylim(audio.min(), audio.max())
        ax_wave.set_title("Waveform", fontsize=14, fontweight="bold")
        ax_wave.grid(alpha=0.3)
        line_wave = ax_wave.axvline(x=0, color="cyan", linewidth=2, alpha=0.8)

    # m/z distribution over time
    if show_mz_distribution:
        ax_mz = fig.add_subplot(gs[panel_idx])
        axes.append(ax_mz)

        # Create simplified m/z timeline
        if sonifier.processed_spectra_dfs:
            num_spectra = len(sonifier.processed_spectra_dfs)
            times_mz = np.linspace(0, duration, num_spectra)

            # Get dominant m/z at each time point
            dominant_mzs = []
            for df in sonifier.processed_spectra_dfs:
                if not df.empty:
                    max_idx = df["intensities"].idxmax()
                    dominant_mzs.append(max_idx)
                else:
                    dominant_mzs.append(0)

            ax_mz.scatter(
                times_mz,
                dominant_mzs,
                c=dominant_mzs,
                cmap="viridis",
                s=10,
                alpha=0.6,
            )
            ax_mz.set_xlabel("Time (s)", fontsize=11)
            ax_mz.set_ylabel("Dominant m/z", fontsize=11)
            ax_mz.set_xlim(0, duration)
            ax_mz.set_title(
                "m/z Distribution Over Time", fontsize=14, fontweight="bold"
            )
            ax_mz.grid(alpha=0.3)
            line_mz = ax_mz.axvline(x=0, color="cyan", linewidth=2, alpha=0.8)

    plt.tight_layout()

    # Animation function
    def animate(frame):
        current_time = frame / fps

        artists = []
        if show_spectrogram:
            line_spec.set_xdata([current_time, current_time])
            artists.append(line_spec)

        if show_waveform:
            line_wave.set_xdata([current_time, current_time])
            artists.append(line_wave)

        if show_mz_distribution:
            line_mz.set_xdata([current_time, current_time])
            artists.append(line_mz)

        return tuple(artists)

    # Create animation
    total_frames = int(duration * fps)
    anim = FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True
    )

    # Save video
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(
            output_path,
            writer=writer,
            dpi=dpi,
            progress_callback=progress_callback,
        )
        plt.close(fig)
        # Add audio to video
        _add_audio_to_video(
            output_path, sonifier.current_audio_data, sample_rate, output_path
        )
        print(f"Video saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating video: {e}")
        print(
            "Make sure ffmpeg is installed: brew install ffmpeg (macOS)"
            " or apt-get install ffmpeg (Linux)"
        )
        plt.close(fig)
        return None


def animate_visualization(
    sonifier,
    output_path: str,
    viz_type: str = "spectrogram",
    fps: int = 30,
    dpi: int = 100,
    duration_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Animate any visualization type and save as video.

    Args:
        sonifier: MSSonifier object
        output_path: Path for output video file (.mp4)
        viz_type: Type of visualization
            ('spectrogram', 'waveform', 'mfcc', 'chromagram')
        fps: Frames per second
        dpi: Resolution
        duration_seconds: Duration of video (None = full audio length)
        progress_callback: Called as ``progress_callback(frame, total)``
            while frames render; raise from it to abort.

    Returns:
        str: Path to created video file
    """
    if sonifier.current_audio_data is None:
        print("No audio data. Run sonification first.")
        return None

    audio = sonifier.current_audio_data
    if audio.ndim > 1:
        audio = _mono(audio)

    sample_rate = sonifier.sample_rate
    full_duration = len(audio) / sample_rate

    if duration_seconds is None:
        duration_seconds = full_duration
    else:
        duration_seconds = min(duration_seconds, full_duration)

    # Create visualization based on type
    fig, ax = plt.subplots(figsize=(14, 6))

    if viz_type == "spectrogram":
        D = librosa.stft(audio, n_fft=2048, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            S_db,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="log",
            ax=ax,
            cmap="magma",
        )
        ax.set_title("Spectrogram", fontsize=16, fontweight="bold")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")

    elif viz_type == "waveform":
        times = np.arange(len(audio)) / sample_rate
        ax.plot(times, audio, linewidth=0.5)
        ax.set_xlim(0, full_duration)
        ax.set_ylim(audio.min(), audio.max())
        ax.set_title("Waveform", fontsize=16, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

    elif viz_type == "mfcc":
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        img = librosa.display.specshow(
            mfccs, sr=sample_rate, x_axis="time", ax=ax, cmap="coolwarm"
        )
        ax.set_title("MFCC", fontsize=16, fontweight="bold")
        plt.colorbar(img, ax=ax)

    elif viz_type == "chromagram":
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        img = librosa.display.specshow(
            chroma,
            sr=sample_rate,
            x_axis="time",
            y_axis="chroma",
            ax=ax,
            cmap="viridis",
        )
        ax.set_title("Chromagram", fontsize=16, fontweight="bold")
        plt.colorbar(img, ax=ax)

    else:
        print(f"Unknown visualization type: {viz_type}")
        plt.close(fig)
        return None

    # Add playhead line
    line = ax.axvline(x=0, color="cyan", linewidth=2, alpha=0.8)
    plt.tight_layout()

    # Animation function
    def animate(frame):
        current_time = frame / fps
        line.set_xdata([current_time, current_time])
        return (line,)

    # Create animation
    total_frames = int(duration_seconds * fps)
    anim = FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True
    )

    # Save video
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(
            output_path,
            writer=writer,
            dpi=dpi,
            progress_callback=progress_callback,
        )
        plt.close(fig)
        # Add audio to video
        _add_audio_to_video(
            output_path, sonifier.current_audio_data, sample_rate, output_path
        )
        print(f"Video saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating video: {e}")
        plt.close(fig)
        return None


def create_3d_scan_video(
    sonifier,
    output_path: str,
    window_size: int = 50,
    duration_seconds: float = 15.0,
    fps: int = 15,
    dpi: int = 100,
    azimuth_rotation: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Create animated 3D plot showing a moving window of scans.

    Args:
        sonifier: MSSonifier object with processed_spectra_dfs
        output_path: Path for output video file (.mp4)
        window_size: Number of scans to show at once
        duration_seconds: Video duration
        fps: Frames per second
        dpi: Resolution
        azimuth_rotation: Whether to rotate the 3D view
        progress_callback: Called as ``progress_callback(frame, total)``
            while frames render; raise from it to abort.

    Returns:
        str: Path to created video file
    """
    if not sonifier.processed_spectra_dfs:
        print("No spectra data available")
        return None

    print(
        f"Creating 3D scan animation with "
        f"{window_size} scans visible at once..."
    )

    all_scans = []
    for i, df in enumerate(sonifier.processed_spectra_dfs):
        if not df.empty:
            mzs = df.index.values
            intensities = df["intensities"].values
            scan_times = np.full_like(mzs, i, dtype=float)
            all_scans.append((scan_times, mzs, intensities))

    if not all_scans:
        print("No valid scan data")
        return None

    total_scans = len(all_scans)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(0, total_scans)
    ax.set_ylim(sonifier.min_mz_overall, sonifier.max_mz_overall)
    ax.set_zlim(0, sonifier.max_intensity_overall * 1.1)

    ax.set_xlabel("Scan Number", fontsize=10)
    ax.set_ylabel("m/z", fontsize=10)
    ax.set_zlabel("Intensity", fontsize=10)
    ax.set_title(
        "3D Mass Spectrum - Moving Window", fontsize=14, fontweight="bold"
    )

    scatter = ax.scatter([], [], [], c=[], cmap="plasma", s=2, alpha=0.6)

    total_frames = int(duration_seconds * fps)
    positions = np.linspace(0, total_scans - window_size, total_frames)

    def animate(frame):
        start_scan = int(positions[frame])
        end_scan = start_scan + window_size

        window_scans = []
        window_mzs = []
        window_intensities = []

        for i in range(start_scan, min(end_scan, total_scans)):
            scan_times, mzs, intensities = all_scans[i]
            window_scans.extend(scan_times)
            window_mzs.extend(mzs)
            window_intensities.extend(intensities)

        if window_scans:
            scatter._offsets3d = (window_scans, window_mzs, window_intensities)
            log_intensities = np.log10(np.array(window_intensities) + 1)
            scatter.set_array(log_intensities)

        if azimuth_rotation:
            ax.view_init(elev=20, azim=30 + frame * 0.5)

        progress = (frame / total_frames) * 100
        ax.set_title(
            f"3D Mass Spectrum - Scans "
            f"{start_scan}-{end_scan} ({progress:.0f}%)",
            fontsize=14,
            fontweight="bold",
        )

        return (scatter,)

    print(f"Rendering {total_frames} frames...")

    anim = FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=False
    )

    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(
            output_path,
            writer=writer,
            dpi=dpi,
            progress_callback=progress_callback,
        )
        plt.close(fig)

        if sonifier.current_audio_data is not None:
            audio = sonifier.current_audio_data
            if audio.ndim > 1:
                audio = _mono(audio)
            _add_audio_to_video(
                output_path,
                sonifier.current_audio_data,
                sonifier.sample_rate,
                output_path,
            )

        print(f"3D video saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating 3D video: {e}")
        plt.close(fig)
        return None


def create_3d_heatmap_video(
    sonifier,
    output_path: str,
    duration_seconds: float = 10.0,
    fps: int = 15,
    dpi: int = 100,
    rotate_speed: float = 1.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """Create rotating 3D heatmap with better intensity visualization.

    Args:
        progress_callback: Called as ``progress_callback(frame, total)``
            while frames render; raise from it to abort.
    """
    if not sonifier.processed_spectra_dfs:
        return None

    print("Creating rotating 3D heatmap...")

    # Extract all data
    max_points = 5000
    all_scans = []
    all_mzs = []
    all_intensities = []

    for i, df in enumerate(sonifier.processed_spectra_dfs):
        if not df.empty:
            all_scans.extend([i] * len(df))
            all_mzs.extend(df.index.values)
            all_intensities.extend(df["intensities"].values)

    all_scans = np.array(all_scans)
    all_mzs = np.array(all_mzs)
    all_intensities = np.array(all_intensities)

    # Filter out low intensity noise (keep top 70%)
    intensity_threshold = np.percentile(all_intensities, 30)
    mask = all_intensities >= intensity_threshold
    all_scans = all_scans[mask]
    all_mzs = all_mzs[mask]
    all_intensities = all_intensities[mask]

    print(f"Filtered to {len(all_scans)} points (removed bottom 30%)")

    # Apply log scaling to intensities for better visualization
    log_intensities = np.log10(all_intensities + 1)

    # Subsample if still too many points
    if len(all_scans) > max_points:
        idx = np.random.choice(len(all_scans), max_points, replace=False)
        all_scans = all_scans[idx]
        all_mzs = all_mzs[idx]
        log_intensities = log_intensities[idx]
        print(f"Subsampled to {max_points} points for performance")

    # Setup figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create scatter plot with log-scaled intensities
    scatter = ax.scatter(
        all_scans,
        all_mzs,
        log_intensities,
        c=log_intensities,
        cmap="hot",  # Better for intensity visualization
        s=5,  # Larger points
        alpha=0.8,  # More opaque
        vmin=np.percentile(log_intensities, 5),  # Adjust color range
        vmax=np.percentile(log_intensities, 95),
    )

    ax.set_xlabel("Scan Number", fontsize=11, labelpad=10)
    ax.set_ylabel("m/z", fontsize=11, labelpad=10)
    ax.set_zlabel("Log10(Intensity)", fontsize=11, labelpad=10)
    ax.set_title(
        "3D Mass Spectrum Heatmap (Log Scale)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set better viewing angle
    ax.view_init(elev=25, azim=45)

    # Animation function
    total_frames = int(duration_seconds * fps)

    def animate(frame):
        azim = 45 + frame * rotate_speed
        ax.view_init(elev=25, azim=azim)
        return (scatter,)

    print(f"Rendering {total_frames} frames...")

    anim = FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=False
    )

    # Save
    writer = FFMpegWriter(fps=fps, bitrate=1500)
    anim.save(
        output_path,
        writer=writer,
        dpi=dpi,
        progress_callback=progress_callback,
    )
    plt.close(fig)

    if sonifier.current_audio_data is not None:
        audio = sonifier.current_audio_data
        if audio.ndim > 1:
            audio = _mono(audio)
        _add_audio_to_video(
            output_path,
            sonifier.current_audio_data,
            sonifier.sample_rate,
            output_path,
        )

    print(f"Rotating 3D heatmap saved to: {output_path}")
    return output_path


def create_comparison_video(
    sonifier,
    audio_dict: dict,
    output_path: str,
    fps: int = 15,
    dpi: int = 80,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Create side-by-side comparison video of multiple sonifications.

    Args:
        sonifier: MSSonifier object (for sample_rate)
        audio_dict: Dictionary mapping labels to audio arrays
        output_path: Path for output video file (.mp4)
        fps: Frames per second
        dpi: Resolution
        progress_callback: Called as ``progress_callback(frame, total)``
            while frames render; raise from it to abort.

    Returns:
        str: Path to created video file
    """
    # Filter out None values
    audio_dict = {k: v for k, v in audio_dict.items() if v is not None}

    if len(audio_dict) < 2:
        print("Need at least 2 audio tracks for comparison")
        return None

    num_audio = len(audio_dict)
    sample_rate = sonifier.sample_rate

    # Find longest duration and normalize all to same length
    max_duration = 0
    max_length = 0
    for audio in audio_dict.values():
        if audio.ndim > 1:
            audio = _mono(audio)
        duration = len(audio) / sample_rate
        max_duration = max(max_duration, duration)
        max_length = max(max_length, len(audio))

    # Mix audio tracks together (equal weighting)
    mixed_audio = np.zeros(max_length, dtype=np.float32)
    for audio in audio_dict.values():
        if audio.ndim > 1:
            audio = _mono(audio)
        # Pad shorter tracks with zeros
        padded = np.zeros(max_length, dtype=np.float32)
        padded[: len(audio)] = audio
        mixed_audio += padded

    # Normalize mixed audio
    mixed_audio /= len(audio_dict)
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio /= max_val

    # Compute spectrograms for all
    spectrograms = {}
    for label, audio in audio_dict.items():
        if audio.ndim > 1:
            audio = _mono(audio)
        D = librosa.stft(audio, n_fft=1024, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        spectrograms[label] = S_db

    # Setup figure
    fig, axes = plt.subplots(num_audio, 1, figsize=(14, 4 * num_audio))
    if num_audio == 1:
        axes = [axes]

    lines = []
    for idx, (label, S_db) in enumerate(spectrograms.items()):
        ax = axes[idx]
        img = librosa.display.specshow(
            S_db,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="log",
            ax=ax,
            cmap="magma",
        )
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_ylabel("Frequency (Hz)", fontsize=11)
        if idx == num_audio - 1:
            ax.set_xlabel("Time (s)", fontsize=11)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        line = ax.axvline(x=0, color="cyan", linewidth=2, alpha=0.8)
        lines.append(line)

    plt.tight_layout()

    # Animation function
    def animate(frame):
        current_time = frame / fps
        for line in lines:
            line.set_xdata([current_time, current_time])
        return tuple(lines)

    # Create animation
    total_frames = int(max_duration * fps)
    anim = FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True
    )

    # Save video
    try:
        writer = FFMpegWriter(fps=fps, bitrate=1500)
        anim.save(
            output_path,
            writer=writer,
            dpi=dpi,
            progress_callback=progress_callback,
        )
        plt.close(fig)

        # Add mixed audio
        _add_audio_to_video(output_path, mixed_audio, sample_rate, output_path)

        print(f"Comparison video with mixed audio saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating comparison video: {e}")
        plt.close(fig)
        return None


def create_3d_waterfall_video(
    sonifier,
    output_path: str,
    duration_seconds: float = 20.0,
    fps: int = 30,
    dpi: int = 100,
    max_freq: float = 5000,
    rotate: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """Create animated 3D waterfall that builds up scan by scan.

    Args:
        progress_callback: Called as ``progress_callback(frame, total)``
            while frames render; raise from it to abort.
    """
    if sonifier.current_audio_data is None:
        return None

    audio = sonifier.current_audio_data
    if audio.ndim > 1:
        audio = _mono(audio)

    sample_rate = sonifier.sample_rate

    print("Creating 3D waterfall animation...")

    n_fft = 2048
    hop_length = 512
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(D.shape[1]), sr=sample_rate, hop_length=hop_length
    )

    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]
    D_db = D_db[freq_mask, :]

    # Downsampling for smoother surface
    step_freq = 3
    step_time = 8
    freqs = freqs[::step_freq]
    D_db = D_db[::step_freq, ::step_time]
    times = times[::step_time]

    # Apply Gaussian smoothing to reduce spikiness
    D_db = gaussian_filter(D_db, sigma=(1.5, 1.5))

    total_scans = D_db.shape[1]

    print(
        f"Waterfall dimensions: {len(freqs)} frequencies "
        f"× {total_scans} time steps"
    )

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(0, times[-1])
    ax.set_ylim(freqs[0], freqs[-1])
    ax.set_zlim(D_db.min(), D_db.max())

    ax.set_xlabel("Time (s)", fontsize=11, labelpad=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=11, labelpad=10)
    ax.set_zlabel("Magnitude (dB)", fontsize=11, labelpad=10)

    ax.view_init(elev=30, azim=45)

    times_mesh, freqs_mesh = np.meshgrid(times, freqs, indexing="ij")

    total_frames = int(duration_seconds * fps)

    def animate(frame):
        scan_progress = int((frame / total_frames) * total_scans)
        scan_progress = max(1, scan_progress)

        while len(ax.collections) > 0:
            ax.collections[0].remove()

        terrain = plt.get_cmap("terrain").resampled(256)
        newcolors = terrain(np.linspace(0.2, 1, 256))
        newcolors[:1, :] = np.array([39 / 256, 30 / 256, 100 / 256, 1])
        newcmp = ListedColormap(newcolors)

        surf = ax.plot_surface(
            times_mesh[:scan_progress, :],
            freqs_mesh[:scan_progress, :],
            D_db.T[:scan_progress, :],
            cmap=newcmp,
            linewidth=0,  # Remove mesh lines
            antialiased=True,
            alpha=0.9,
            shade=True,  # Enable smooth shading
            vmin=D_db.min(),
            vmax=D_db.max(),
        )

        if rotate:
            azim = 45 + (frame / total_frames) * 90
            ax.view_init(elev=30, azim=azim)

        progress = (scan_progress / total_scans) * 100
        ax.set_title(
            f"3D Spectrogram Waterfall - {progress:.0f}% Complete",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        return (surf,)

    print(f"Rendering {total_frames} frames...")

    anim = FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=False
    )

    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(
        output_path,
        writer=writer,
        dpi=dpi,
        progress_callback=progress_callback,
    )
    plt.close(fig)

    _add_audio_to_video(
        output_path, sonifier.current_audio_data, sample_rate, output_path
    )

    print(f"3D waterfall video saved to: {output_path}")
    return output_path


def create_3d_spectrogram_buildup_video(
    sonifier,
    output_path: str,
    duration_seconds: float = 20.0,
    fps: int = 30,
    dpi: int = 100,
    max_freq: float = 5000,
    colormap: str = "plasma",
    style: str = "bars",  # 'bars' or 'lines'
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """Create animated 3D spectrogram that builds up scan by scan with bars.

    Args:
        progress_callback: Called as ``progress_callback(frame, total)``
            while frames render; raise from it to abort.
    """
    if sonifier.current_audio_data is None:
        return None

    audio = sonifier.current_audio_data
    if audio.ndim > 1:
        audio = _mono(audio)

    sample_rate = sonifier.sample_rate

    print("Creating 3D spectrogram buildup animation...")

    # Compute spectrogram
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Get frequency and time arrays
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(D.shape[1]), sr=sample_rate, hop_length=hop_length
    )

    # Apply frequency limit
    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]
    D_db = D_db[freq_mask, :]

    # Downsample for performance
    step_freq = 8
    step_time = 3
    freqs = freqs[::step_freq]
    D_db = D_db[::step_freq, ::step_time]
    times = times[::step_time]

    total_scans = D_db.shape[1]

    print(
        f"Spectrogram dimensions: {len(freqs)} frequencies "
        f"× {total_scans} time steps"
    )

    # Setup figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Set fixed limits
    ax.set_xlim(0, times[-1])
    ax.set_ylim(freqs[0], freqs[-1])
    ax.set_zlim(D_db.min(), D_db.max())

    ax.set_xlabel("Time (s)", fontsize=11, labelpad=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=11, labelpad=10)
    ax.set_zlabel("Magnitude (dB)", fontsize=11, labelpad=10)

    ax.view_init(elev=25, azim=45)

    # Prepare bar positions
    time_positions = times
    freq_positions = freqs

    total_frames = int(duration_seconds * fps)

    # Pre-compute constants for bars style outside the animation loop.
    dt = (times[1] - times[0]) if len(times) > 1 else 0.1
    df = (freqs[1] - freqs[0]) if len(freqs) > 1 else 10.0
    db_min = D_db.min()
    db_range = D_db.max() - db_min
    cmap_fn = plt.get_cmap(colormap)

    # Track which time slices have already been drawn so each frame only
    # adds NEW slices. This makes the total work O(total_scans) instead of
    # O(frames × avg_scan_progress), which was the cause of the hang.
    prev_progress = [0]

    def animate(frame):
        scan_progress = max(1, int((frame / total_frames) * total_scans))
        new_start = prev_progress[0]
        prev_progress[0] = scan_progress

        if style == "lines":
            for t_idx in range(new_start, scan_progress):
                time_val = time_positions[t_idx]
                magnitudes = D_db[:, t_idx]
                color = cmap_fn(t_idx / total_scans)
                ax.plot(
                    [time_val] * len(freq_positions),
                    freq_positions,
                    magnitudes,
                    color=color,
                    linewidth=1.5,
                    alpha=0.7,
                )

        else:  # bars
            for t_idx in range(new_start, scan_progress):
                time_val = time_positions[t_idx]
                magnitudes = D_db[:, t_idx]
                mask = magnitudes > db_min + 5
                if not np.any(mask):
                    continue
                mags_sel = magnitudes[mask]
                freqs_sel = freq_positions[mask]
                colors = cmap_fn((mags_sel - db_min) / (db_range + 1e-8))
                ax.bar3d(
                    np.full(mask.sum(), time_val),
                    freqs_sel,
                    np.full(mask.sum(), db_min),
                    dt,
                    df,
                    mags_sel - db_min,
                    color=colors,
                    alpha=0.8,
                    shade=True,
                )

        progress = (scan_progress / total_scans) * 100
        style_name = "Lines" if style == "lines" else "Bars"
        ax.set_title(
            f"3D Spectrogram ({style_name}) - {progress:.0f}% Complete",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        azim = 45 + (frame / total_frames) * 60
        ax.view_init(elev=25, azim=azim)

        return (ax,)

    print(f"Rendering {total_frames} frames...")

    anim = FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=False
    )

    # Save
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(
        output_path,
        writer=writer,
        dpi=dpi,
        progress_callback=progress_callback,
    )
    plt.close(fig)

    # Add audio
    _add_audio_to_video(
        output_path, sonifier.current_audio_data, sample_rate, output_path
    )

    print(f"3D spectrogram buildup video saved to: {output_path}")
    return output_path


def _add_audio_to_video(video_path, audio_data, sample_rate, output_path):
    """Combine video and audio using FFmpeg."""
    from . import io

    # Create temp files
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_audio.close()
    temp_output.close()

    # Save audio
    audio_normalized = io.normalize_audio_to_16bit(audio_data)
    io.save_wav(temp_audio.name, audio_normalized, sample_rate)

    # Combine with FFmpeg
    cmd = [
        ffmpeg_path() or "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        temp_audio.name,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        temp_output.name,
    ]

    subprocess.run(cmd)

    # Replace original with merged version (shutil.move works across devices)
    import shutil

    shutil.move(temp_output.name, output_path)

    # Cleanup
    os.remove(temp_audio.name)
