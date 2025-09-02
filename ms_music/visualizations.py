import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import pearsonr
from typing import Tuple, Dict, Any


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

    # Define all available mappings
    all_mappings = {
        "linear": sonifier._mz_to_frequency_linear,
        "inverse_log": sonifier._mz_to_frequency_inverse_log,
        "power_law": sonifier._mz_to_frequency_power_law,
        "musical_octaves": sonifier._mz_to_frequency_musical_octaves,
        "chromatic": sonifier._mz_to_frequency_chromatic,
    }

    # Add custom mappings if provided
    if custom_mappings:
        all_mappings.update(custom_mappings)

    # Determine which mappings to plot
    if mapping_types == "all":
        mappings_to_plot = list(all_mappings.keys())
    elif isinstance(mapping_types, list):
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
                if mapping_name == "linear":
                    freq = mapping_func(
                        mz_value=mz, freq_range=freq_range,
                    )
                elif mapping_name == "power_law":
                    # Power law has an extra exponent parameter
                    freq = mapping_func(
                        mz_value=mz, freq_range=freq_range,
                    )
                elif mapping_name == "musical_octaves":
                    # Musical octaves uses base_freq and num_octaves
                    freq = mapping_func(
                        mz_value=mz,
                    )
                elif mapping_name == "chromatic":
                    # Chromatic scale uses base_freq and num_semitones
                    freq = mapping_func(
                        mz_value=mz,
                    )
                elif mapping_name == "inverse_log":
                    freq = mapping_func(
                        mz_value=mz, freq_range=freq_range,
                    )
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
    figsize: Tuple[int, int] = (12, 16)
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
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot spectrogram of audio data.

    Args:
        audio_data: Audio signal array
        sample_rate: Audio sample rate
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)

    D = librosa.stft(audio_data)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    img = librosa.display.specshow(
        D_db, sr=sample_rate, x_axis="time", y_axis="log", ax=ax
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.show()

    return fig
