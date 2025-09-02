import pandas as pd
from matchms.importing import load_from_mzml
from tqdm import tqdm
import numpy as np
import os
from scipy.io import wavfile
import librosa
from typing import Tuple


def load_mzml_data(
    filepath: str, ms_level: int = 1, metadata_harmonization: bool = False
):
    """
    Loads spectra from an mzML file.
    For MS2, it sorts by scan number if available.
    """
    spectra_to_process = []
    try:
        if ms_level == 1:
            spectra_to_process = list(
                load_from_mzml(
                    filepath,
                    ms_level=1,
                    metadata_harmonization=metadata_harmonization,
                )
            )
        elif ms_level == 2:
            ms2_spectra = list(
                load_from_mzml(
                    filepath,
                    ms_level=2,
                    metadata_harmonization=metadata_harmonization,
                )
            )
            if not ms2_spectra:
                print(
                    f"No MS2 spectra found in {filepath}. Consider loading "
                    f"MS1 only or checking the file content."
                )
                return None

            scan_info_list = []
            for i, spectrum in enumerate(ms2_spectra):
                scan_num = None
                title = spectrum.metadata.get("title", "")
                if 'NativeID:"scan=' in title:  # Specific to some formats
                    scan_str = title.split('NativeID:"scan=')[-1].split('"')[0]
                    if scan_str.isdigit():
                        scan_num = int(scan_str)
                elif "scan=" in title:  # More generic
                    scan_str = (
                        title.split("scan=")[-1].split(" ")[0].split('"')[0]
                    )
                    if scan_str.isdigit():
                        scan_num = int(scan_str)

                if scan_num is None:  # Fallback if title parsing fails
                    scan_num = spectrum.metadata.get(
                        "scan_start_time", float(i)
                    )  # Use time or index

                scan_info_list.append(
                    {"original_index": i, "scan_identifier": scan_num}
                )

            # Sort by the extracted scan identifier
            sorted_scans_info = sorted(
                scan_info_list, key=lambda x: x["scan_identifier"]
            )
            spectra_to_process = [
                ms2_spectra[info["original_index"]]
                for info in sorted_scans_info
            ]
        else:
            raise ValueError("ms_level must be 1 or 2.")

        if not spectra_to_process:
            print(
                f"No spectra of MS level {ms_level} were successfully "
                f"loaded or remained after MS2 sorting "
                f"from {os.path.basename(filepath)}."
            )
            return None
        return spectra_to_process

    except Exception as e:
        print(f"Error loading mzML file {os.path.basename(filepath)}: {e}")
        return None


def preprocess_spectra(spectra_list: list):
    if not spectra_list:
        print("Warning: Empty spectra list provided to preprocess_spectra.")
        return [], 0.0, 0.0, 0.0  # Return float zeros

    processed_spectra_dfs = []
    max_intensity_overall = 0.0
    all_min_mzs = []
    all_max_mzs = []

    for spectrum in tqdm(
        spectra_list, desc="Preprocessing Spectra", unit="spectrum"
    ):
        if spectrum.peaks.mz.size == 0:
            # Represent empty scan with an empty DataFrame
            empty_df = pd.DataFrame(columns=["intensities"]).set_index(
                pd.Index([], name="rounded_mzs", dtype=int)
            )
            processed_spectra_dfs.append(empty_df)
            continue

        peaks_df = pd.DataFrame(
            {
                "mz": spectrum.peaks.mz,
                "intensities": spectrum.peaks.intensities,
            }
        )
        peaks_df["rounded_mzs"] = peaks_df["mz"].round().astype(int)

        grouped_peaks = (
            peaks_df.groupby("rounded_mzs")["intensities"].sum().reset_index()
        )
        grouped_peaks = grouped_peaks.set_index(
            "rounded_mzs"
        )  # Set rounded_mzs as index

        if not grouped_peaks.empty:
            current_max_intensity = grouped_peaks["intensities"].max()
            if (
                pd.notna(current_max_intensity)
                and current_max_intensity > max_intensity_overall
            ):
                max_intensity_overall = float(current_max_intensity)

            all_min_mzs.append(float(grouped_peaks.index.min()))
            all_max_mzs.append(float(grouped_peaks.index.max()))

        processed_spectra_dfs.append(grouped_peaks)

    min_mz_overall = float(min(all_min_mzs)) if all_min_mzs else 0.0
    max_mz_overall = float(max(all_max_mzs)) if all_max_mzs else 0.0

    if max_intensity_overall <= 0 and any(
        not df.empty and df["intensities"].max() > 0
        for df in processed_spectra_dfs
    ):
        # This case should ideally not be hit if max_intensity_overall is
        # updated correctly with floats.
        # If it is, it means all actual intensities were <= 0.
        print(
            "Warning: Max overall intensity calculated as non-positive, "
            "though peaks exist. This might occur if all intensities are "
            "zero or negative. Sonification may be silent."
        )
    elif (
        max_intensity_overall == 0
    ):  # Default case for truly empty/zero-intensity data
        print(
            "Warning: Max overall intensity is 0. Sonification will "
            "likely result in silence."
        )

    return (
        processed_spectra_dfs,
        max_intensity_overall,
        min_mz_overall,
        max_mz_overall,
    )


def normalize_audio_to_16bit(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalize audio data to 16-bit range.

    Args:
        audio_data: Audio signal array

    Returns:
        Normalized audio array as int16
    """
    if audio_data.size == 0:
        return audio_data.astype(np.int16)

    # Normalize to [-1, 1] range
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        normalized = audio_data / max_val
    else:
        normalized = audio_data

    # Convert to 16-bit integer
    audio_int16 = (normalized * 32767).astype(np.int16)
    return audio_int16


def save_wav(filepath: str, audio_data: np.ndarray, sample_rate: int = 44100):
    """
    Save audio data as a WAV file.

    Args:
        filepath: Output file path
        audio_data: Audio signal array
        sample_rate: Sample rate in Hz
    """
    # Ensure audio is in the correct format
    if audio_data.dtype != np.int16:
        audio_data = normalize_audio_to_16bit(audio_data)

    # Save the file
    wavfile.write(filepath, sample_rate, audio_data)


def load_audio(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.

    Args:
        filepath: Path to audio file

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio_data, sample_rate = librosa.load(filepath, sr=None)
    return audio_data, sample_rate
