import pandas as pd
import pymzml
from tqdm import tqdm
import numpy as np
import os
from typing import Tuple
from scipy.io import wavfile
import librosa
import time


def load_mzml_data(
    filepath: str, ms_level: int = 1, metadata_harmonization: bool = False
):
    """
    Loads spectra from an mzML file using pymzml with progress tracking.
    For MS2, it sorts by scan number if available.
    """
    try:
        print(f"Loading MS{ms_level} spectra from {os.path.basename(filepath)}...")
        
        run = pymzml.run.Reader(filepath)
        spectra_data = []
        
        # Single-pass with dynamic progress bar
        pbar = tqdm(desc=f"Loading MS{ms_level} spectra", unit="spectrum")
        
        for spectrum in run:
            # Filter by MS level
            if spectrum.ms_level != ms_level:
                continue
            
            # Extract m/z and intensity arrays
            mz_array = spectrum.mz
            intensity_array = spectrum.i
            
            if len(mz_array) == 0:
                pbar.update(1)
                continue
            
            # Get scan number for sorting (MS2 only)
            if ms_level == 2:
                scan_num = spectrum.ID
                if isinstance(scan_num, str):
                    import re
                    match = re.search(r'scan=(\d+)', scan_num)
                    scan_num = int(match.group(1)) if match else spectrum.scan_time_in_minutes()
            else:
                scan_num = None

            # Capture retention time for all spectra
            retention_time = None
            try:
                retention_time = spectrum.scan_time_in_minutes()
            except Exception:
                pass

            # Capture precursor m/z and isolation window for MS2 spectra
            precursor_mz = None
            isolation_window = None
            if ms_level == 2:
                try:
                    precursors = spectrum.selected_precursors
                    if precursors:
                        precursor_mz = precursors[0].get('mz', None)
                except Exception:
                    pass

                # Isolation window bounds (needed for DIA mode)
                try:
                    iw_target = spectrum.get(
                        'isolation window target m/z', None)
                    iw_lower = spectrum.get(
                        'isolation window lower offset', None)
                    iw_upper = spectrum.get(
                        'isolation window upper offset', None)
                    if iw_target is not None:
                        iw_target = float(iw_target)
                        iw_lower = float(iw_lower) if iw_lower is not None else 0.0
                        iw_upper = float(iw_upper) if iw_upper is not None else 0.0
                        isolation_window = (
                            iw_target - iw_lower,
                            iw_target + iw_upper,
                        )
                except Exception:
                    pass

            spectra_data.append({
                'mz': mz_array,
                'intensity': intensity_array,
                'scan_num': scan_num,
                'precursor_mz': precursor_mz,
                'isolation_window': isolation_window,
                'retention_time': retention_time,
            })
            
            pbar.update(1)
        
        pbar.close()
        
        if not spectra_data:
            print(f"No MS{ms_level} spectra found in {filepath}.")
            return None
        
        # Sort MS2 by scan number. scan_num may be int, float, or None
        # (depending on how it was resolved), so guard against mixed-type
        # comparisons by pushing None entries to the end.
        if ms_level == 2:
            spectra_data.sort(
                key=lambda x: (x['scan_num'] is None, x['scan_num'] or 0)
            )
        
        print(f"✓ Loaded {len(spectra_data)} spectra")
        return spectra_data

    except Exception as e:
        print(f"Error loading mzML file {os.path.basename(filepath)}: {e}")
        return None


def preprocess_spectra(spectra_list: list):
    """
    Vectorized preprocessing of spectra data.
    """
    if not spectra_list:
        print("Warning: Empty spectra list provided to preprocess_spectra.")
        return [], 0.0, 0.0, 0.0

    print(f"Preprocessing {len(spectra_list)} spectra...")
    
    # Extract all peaks at once - no progress bar needed, it's fast
    all_mzs = []
    all_intensities = []
    spectrum_indices = []
    
    for idx, spectrum_dict in enumerate(spectra_list):
        mz_array = spectrum_dict['mz']
        intensity_array = spectrum_dict['intensity']
        
        if len(mz_array) > 0:
            all_mzs.append(mz_array)
            all_intensities.append(intensity_array)
            spectrum_indices.append(np.full(len(mz_array), idx))
    
    if not all_mzs:
        empty_df = pd.DataFrame(columns=["intensities"]).set_index(
            pd.Index([], name="rounded_mzs", dtype=int))
        return [empty_df] * len(spectra_list), 0.0, 0.0, 0.0
    
    # Vectorized operations - extremely fast, no progress bar needed
    all_mzs = np.concatenate(all_mzs)
    all_intensities = np.concatenate(all_intensities)
    spectrum_indices = np.concatenate(spectrum_indices)
    
    rounded_mzs = np.round(all_mzs).astype(int)
    
    df = pd.DataFrame({
        'spectrum_idx': spectrum_indices,
        'rounded_mzs': rounded_mzs,
        'intensities': all_intensities
    })
    
    grouped = df.groupby(['spectrum_idx', 'rounded_mzs'])['intensities'].sum()
    
    # Split back into per-spectrum DataFrames
    processed_spectra_dfs = []
    for idx in range(len(spectra_list)):
        if idx in grouped.index.get_level_values(0):
            spec_data = grouped.loc[idx].to_frame()
            processed_spectra_dfs.append(spec_data)
        else:
            processed_spectra_dfs.append(pd.DataFrame(columns=["intensities"]).set_index(
                pd.Index([], name="rounded_mzs", dtype=int)))
    
    max_intensity_overall = float(df['intensities'].max())
    min_mz_overall = float(rounded_mzs.min())
    max_mz_overall = float(rounded_mzs.max())
    
    print(f"✓ Preprocessed {len(all_mzs):,} peaks | m/z: {min_mz_overall:.1f}-{max_mz_overall:.1f} | max intensity: {max_intensity_overall:.2e}")
    
    return processed_spectra_dfs, max_intensity_overall, min_mz_overall, max_mz_overall


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
