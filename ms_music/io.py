import pandas as pd
import pymzml
from tqdm import tqdm
import numpy as np
import os
import hashlib
import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from scipy.io import wavfile
import librosa
import time


# ---------------------------------------------------------------- loading
# Ion mobility can come as a per-peak array (timsTOF, some Waters/Agilent
# exports) or as one value per spectrum (e.g. PASEF MS2 scans, FAIMS CV).
MOBILITY_ARRAYS = (
    ("mean inverse reduced ion mobility array", "1/K0 (V·s/cm²)"),
    ("inverse reduced ion mobility array", "1/K0 (V·s/cm²)"),
    ("raw ion mobility array", "ion mobility"),
    ("mean drift time array", "drift time (ms)"),
    ("raw ion mobility drift time array", "drift time (ms)"),
)
MOBILITY_SCALARS = (
    ("inverse reduced ion mobility", "1/K0 (V·s/cm²)"),
    ("ion mobility drift time", "drift time (ms)"),
    ("FAIMS compensation voltage", "FAIMS CV (V)"),
)

# Bump when the binned format or its semantics change (invalidates caches).
_LOADER_VERSION = 2

# Raw-resolution ion mobility histograms kept alongside the binned data
# (binning to integer m/z keeps only a mean mobility per bin, which would
# merge charge states): 1-Th m/z bins up to MAP_MZ_MAX, MAP_MOBILITY_BINS
# mobility bins.
MAP_MZ_MAX = 5000
MAP_MOBILITY_BINS = 200


def load_mzml_data(
    filepath: str, ms_level: int = 1, metadata_harmonization: bool = False
):
    """
    Load raw spectra (full m/z and intensity arrays) from an mzML file.

    Keeps every array in memory, so it suits small files. For large runs
    (e.g. timsTOF), use :func:`load_spectra`, which bins while reading.
    Spectra are returned in acquisition (retention time) order.
    """
    try:
        print(
            f"Loading MS{ms_level} spectra from "
            f"{os.path.basename(filepath)}..."
        )

        run = pymzml.run.Reader(filepath)
        spectra_data = []
        pbar = tqdm(desc=f"Loading MS{ms_level} spectra", unit="spectrum")

        for spectrum in run:
            if spectrum.ms_level != ms_level:
                continue
            mz_array = spectrum.mz
            intensity_array = spectrum.i
            if len(mz_array) == 0:
                pbar.update(1)
                continue
            meta = _spectrum_metadata(spectrum, ms_level)
            spectra_data.append(
                {
                    "mz": mz_array,
                    "intensity": intensity_array,
                    "scan_num": None,
                    "precursor_mz": meta["precursor_mz"],
                    "isolation_window": meta["isolation_window"],
                    "retention_time": meta["retention_time"],
                }
            )
            pbar.update(1)

        pbar.close()

        if not spectra_data:
            print(f"No MS{ms_level} spectra found in {filepath}.")
            return None

        _sort_by_retention_time(spectra_data, lambda s: s["retention_time"])
        print(f"✓ Loaded {len(spectra_data)} spectra")
        return spectra_data

    except Exception as e:
        print(f"Error loading mzML file {os.path.basename(filepath)}: {e}")
        return None


def _sort_by_retention_time(items, key):
    """Stable sort by retention time; spectra without one keep file order at
    the end. (pymzml's spectrum ID is not a time: for timsTOF it is the ion
    mobility scan number.)"""
    if any(key(item) is not None for item in items):
        items.sort(key=lambda it: (key(it) is None, key(it) or 0.0))


def _spectrum_metadata(spectrum, ms_level):
    """Retention time (min) plus, for MSn, precursor and isolation window."""
    meta = {
        "retention_time": None,
        "precursor_mz": None,
        "charge": None,
        "isolation_window": None,
    }
    try:
        meta["retention_time"] = float(spectrum.scan_time_in_minutes())
    except Exception:
        pass
    if ms_level < 2:
        return meta
    try:
        precursors = spectrum.selected_precursors
        if precursors:
            meta["precursor_mz"] = precursors[0].get("mz", None)
            meta["charge"] = precursors[0].get("charge", None)
    except Exception:
        pass
    try:
        target = spectrum.get("isolation window target m/z", None)
        if target is not None:
            lower = spectrum.get("isolation window lower offset", None)
            upper = spectrum.get("isolation window upper offset", None)
            target = float(target)
            lower = float(lower) if lower is not None else 0.0
            upper = float(upper) if upper is not None else 0.0
            meta["isolation_window"] = (target - lower, target + upper)
    except Exception:
        pass
    return meta


def bin_spectrum(mz, intensity, mobility=None):
    """Bin one spectrum to integer m/z.

    Returns ``(bins, summed_intensity, mobility)`` where ``mobility`` is the
    intensity-weighted mean mobility per bin (None without mobility).
    This is the same rounding and summing ``preprocess_spectra`` applies.
    """
    rounded = np.round(np.asarray(mz)).astype(int)
    bins, inverse = np.unique(rounded, return_inverse=True)
    intensity = np.asarray(intensity)
    sums = np.bincount(inverse, weights=intensity).astype(
        np.result_type(intensity.dtype, np.float32), copy=False
    )
    mob = None
    if mobility is not None:
        weighted = np.bincount(
            inverse, weights=intensity * np.asarray(mobility)
        )
        total = np.bincount(inverse, weights=intensity)
        mob = np.divide(
            weighted, total, out=np.full(total.shape, np.nan), where=total > 0
        )
    return bins, sums, mob


def _binned_frame(bins, sums, mob=None):
    data = {"intensities": sums}
    if mob is not None:
        data["mobility"] = mob
    return pd.DataFrame(data, index=pd.Index(bins, name="rounded_mzs"))


def _empty_frame():
    return pd.DataFrame(columns=["intensities"]).set_index(
        pd.Index([], name="rounded_mzs", dtype=int)
    )


@dataclass
class LoadedRun:
    """Binned spectra of one MS level plus their per-spectrum metadata.

    ``spectra`` holds one DataFrame per spectrum, indexed by integer m/z
    (``rounded_mzs``) with ``intensities`` and, when the file has ion
    mobility, ``mobility`` (intensity-weighted mean per bin).
    """

    ms_level: int
    spectra: List[pd.DataFrame]
    max_intensity: float
    min_mz: float
    max_mz: float
    retention_times: List[Optional[float]]
    precursor_mz: List[Optional[float]]
    charges: List[Optional[int]]
    isolation_windows: List[Optional[Tuple[float, float]]]
    spectrum_mobility: List[Optional[float]]
    spectrum_max_intensity: List[float]
    mobility_unit: Optional[str] = None
    mobility_range: Optional[Tuple[float, float]] = None
    from_cache: bool = False
    # From per-peak mobility arrays only (None otherwise): summed intensity
    # on an (m/z, mobility) grid and per spectrum over mobility, with the
    # mobility bin edges.
    mobility_map: Optional[np.ndarray] = None
    mobility_profiles: Optional[np.ndarray] = None
    mobility_edges: Optional[np.ndarray] = None

    @property
    def has_mobility(self) -> bool:
        return self.mobility_unit is not None


def _find_mobility_array(spectrum):
    names = set(spectrum.get_all_arrays_in_spec())
    for name, unit in MOBILITY_ARRAYS:
        if name in names:
            return name, unit
    return None, None


def _spectrum_scalar_mobility(spectrum):
    for name, unit in MOBILITY_SCALARS:
        try:
            value = spectrum.get(name, None)
        except Exception:
            value = None
        if value is not None:
            try:
                return float(value), unit
            except (TypeError, ValueError):
                continue
    return None, None


def load_spectra(
    filepath: str,
    ms_level: int = 1,
    rt_range: Optional[Tuple[float, float]] = None,
    mobility_range: Optional[Tuple[float, float]] = None,
    cache: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Optional[LoadedRun]:
    """Stream spectra from an mzML file, binning each one as it is read.

    Memory stays proportional to the *binned* data (one value per integer
    m/z per spectrum), so multi-GB runs with hundreds of thousands of peaks
    per spectrum load without holding the raw arrays.

    Args:
        filepath: mzML file.
        ms_level: MS level to keep (1 or 2).
        rt_range: Optional (start, end) retention time window in minutes.
        mobility_range: Optional (low, high) ion mobility window; peaks (or,
            with only a per-spectrum value, whole spectra) outside it are
            dropped before binning.
        cache: Reuse/save the binned result in the cache directory
            (``$MS_MUSIC_CACHE_DIR`` or ``~/.cache/ms_music``), keyed by the
            file's path, size and modification time and these settings.
        progress: Called as ``progress(spectra_read, total_spectra)``.

    Returns:
        A :class:`LoadedRun`, or None if no spectra matched.
    """
    key = _cache_key(filepath, ms_level, rt_range, mobility_range)
    if cache:
        cached = _read_cache(key)
        if cached is not None:
            print(
                f"✓ Loaded {len(cached.spectra)} MS{ms_level} spectra "
                f"from cache"
            )
            return cached

    print(f"Loading MS{ms_level} spectra from {os.path.basename(filepath)}...")
    run = pymzml.run.Reader(filepath)
    try:
        total = int(run.get_spectrum_count())
    except Exception:
        total = 0

    records = []
    profiles = []
    mob_map = mob_edges = None
    max_intensity = 0.0
    mob_array_name = mob_unit = None
    checked_arrays = False
    mob_lo, mob_hi = np.inf, -np.inf
    lo_rt, hi_rt = rt_range if rt_range else (None, None)

    pbar = tqdm(
        total=total or None,
        desc=f"Reading (keeping MS{ms_level})",
        unit="spectrum",
    )
    for n_read, spectrum in enumerate(run, start=1):
        if progress is not None and n_read % 200 == 0:
            progress(n_read, total)
        pbar.update(1)
        if rt_range:
            try:
                rt_any = float(spectrum.scan_time_in_minutes())
            except Exception:
                rt_any = None
            # mzML stores spectra in acquisition order: past the window's
            # end nothing more can match, so stop reading.
            if rt_any is not None and rt_any > hi_rt:
                break
        if spectrum.ms_level != ms_level:
            continue
        meta = _spectrum_metadata(spectrum, ms_level)
        rt = meta["retention_time"]
        if rt_range and rt is not None and not (lo_rt <= rt <= hi_rt):
            continue

        mz = spectrum.mz
        if len(mz) == 0:
            continue
        intensity = spectrum.i

        if not checked_arrays:
            mob_array_name, unit = _find_mobility_array(spectrum)
            if mob_array_name:
                mob_unit = unit
            checked_arrays = True
        mobility = None
        if mob_array_name:
            mobility = spectrum.get_array(mob_array_name)
        scalar_mob, scalar_unit = _spectrum_scalar_mobility(spectrum)
        if mobility is None and scalar_mob is not None:
            mobility = np.full(len(mz), scalar_mob)
            mob_unit = mob_unit or scalar_unit

        if mobility_range is not None and mobility is not None:
            keep = (mobility >= mobility_range[0]) & (
                mobility <= mobility_range[1]
            )
            if not keep.any():
                continue
            mz, intensity, mobility = mz[keep], intensity[keep], mobility[keep]

        spec_max = float(np.max(intensity))
        max_intensity = max(max_intensity, spec_max)
        if mobility is not None and len(mobility):
            mob_lo = min(mob_lo, float(np.min(mobility)))
            mob_hi = max(mob_hi, float(np.max(mobility)))
        if mob_array_name and mobility is not None and len(mobility):
            if mob_edges is None:
                # Every timsTOF frame spans the whole mobility ramp, so the
                # first frame's range (with a margin) fits the whole run.
                lo_m, hi_m = float(np.min(mobility)), float(np.max(mobility))
                pad = 0.05 * (hi_m - lo_m or 1.0)
                mob_edges = np.linspace(
                    lo_m - pad, hi_m + pad, MAP_MOBILITY_BINS + 1
                )
                mob_map = np.zeros((MAP_MZ_MAX, MAP_MOBILITY_BINS))
            m_idx = np.clip(
                np.searchsorted(mob_edges, mobility) - 1,
                0,
                MAP_MOBILITY_BINS - 1,
            )
            z_idx = np.clip(np.asarray(mz).astype(int), 0, MAP_MZ_MAX - 1)
            _accumulate(mob_map, z_idx, m_idx, intensity)
            profiles.append(
                np.bincount(
                    m_idx, weights=intensity, minlength=MAP_MOBILITY_BINS
                )
            )
        bins, sums, mob = bin_spectrum(mz, intensity, mobility)
        records.append(
            (
                rt,
                bins,
                sums,
                mob,
                meta,
                scalar_mob,
                spec_max,
                profiles[-1] if (mob_array_name and profiles) else None,
            )
        )
    pbar.close()
    if progress is not None:
        progress(total or 1, total or 1)

    if not records:
        print(f"No MS{ms_level} spectra found in {filepath}.")
        return None

    _sort_by_retention_time(records, lambda r: r[0])
    loaded = _assemble(
        ms_level,
        records,
        max_intensity,
        mob_unit if np.isfinite(mob_lo) else None,
        (mob_lo, mob_hi) if np.isfinite(mob_lo) else None,
    )
    if mob_map is not None and all(r[7] is not None for r in records):
        used = np.nonzero(mob_map.any(axis=1))[0]
        top = int(used.max()) + 1 if used.size else 1
        loaded.mobility_map = mob_map[:top]
        loaded.mobility_profiles = np.vstack([r[7] for r in records])
        loaded.mobility_edges = mob_edges
    n_peaks = sum(len(r[1]) for r in records)
    print(
        f"✓ Loaded {len(records)} MS{ms_level} spectra "
        f"({n_peaks:,} binned peaks)"
    )
    if cache:
        _write_cache(
            key,
            loaded,
            source={
                "path": os.path.abspath(filepath),
                "file": os.path.basename(filepath),
                "ms_level": int(ms_level),
                "rt_range": list(rt_range) if rt_range else None,
                "mobility_range": (
                    list(mobility_range) if mobility_range else None
                ),
                "spectra": len(loaded.spectra),
                "created": time.strftime("%Y-%m-%d %H:%M"),
            },
        )
    return loaded


def _accumulate(grid, rows, cols, weights):
    """``grid[rows, cols] += weights`` with repeated indices summed."""
    if len(rows) < 5000:
        np.add.at(grid, (rows, cols), weights)
        return
    # Large spectra: one bincount over the touched rows only.
    lo, hi = int(rows.min()), int(rows.max()) + 1
    flat = (rows - lo) * grid.shape[1] + cols
    grid[lo:hi] += np.bincount(
        flat, weights=weights, minlength=(hi - lo) * grid.shape[1]
    ).reshape(hi - lo, grid.shape[1])


def _assemble(ms_level, records, max_intensity, mob_unit, mob_range):
    spectra = [_binned_frame(r[1], r[2], r[3]) for r in records]
    all_min = min(int(r[1][0]) for r in records)
    all_max = max(int(r[1][-1]) for r in records)
    return LoadedRun(
        ms_level=ms_level,
        spectra=spectra,
        max_intensity=float(max_intensity),
        min_mz=float(all_min),
        max_mz=float(all_max),
        retention_times=[r[0] for r in records],
        precursor_mz=[r[4]["precursor_mz"] for r in records],
        charges=[r[4]["charge"] for r in records],
        isolation_windows=[r[4]["isolation_window"] for r in records],
        spectrum_mobility=[r[5] for r in records],
        spectrum_max_intensity=[r[6] for r in records],
        mobility_unit=mob_unit,
        mobility_range=mob_range,
    )


def load_reference_peaks(
    filepath: str,
    ms_level: int = 1,
    top_peaks: int = 2000,
    progress: Optional[Callable[[int, int], None]] = None,
    rt_range: Optional[Tuple[float, float]] = None,
) -> List[dict]:
    """Retention time plus the ``top_peaks`` most intense peaks (full m/z
    precision) of every spectrum, sorted by retention time. ``rt_range``
    (minutes) limits the spectra read; reading stops after its end.

    Used as the MS1 reference for DIA precursor tones; bounded memory even
    for very large runs.
    """
    run = pymzml.run.Reader(filepath)
    try:
        total = int(run.get_spectrum_count())
    except Exception:
        total = 0
    ref = []
    for n_read, spectrum in enumerate(
        tqdm(
            run,
            total=total or None,
            desc="Reading MS1 reference",
            unit="spectrum",
        ),
        start=1,
    ):
        if progress is not None and n_read % 200 == 0:
            progress(n_read, total)
        try:
            rt = float(spectrum.scan_time_in_minutes())
        except Exception:
            continue
        if rt_range and rt > rt_range[1]:
            break
        if spectrum.ms_level != ms_level:
            continue
        if rt_range and rt < rt_range[0]:
            continue
        mz, intensity = spectrum.mz, spectrum.i
        if len(mz) == 0:
            continue
        if top_peaks and len(mz) > top_peaks:
            keep = np.argpartition(intensity, -top_peaks)[-top_peaks:]
            keep.sort()
            mz, intensity = mz[keep], intensity[keep]
        ref.append(
            {
                "retention_time": rt,
                "mz": np.asarray(mz),
                "intensity": np.asarray(intensity),
            }
        )
    if progress is not None:
        progress(total or 1, total or 1)
    ref.sort(key=lambda x: x["retention_time"])
    return ref


# ------------------------------------------------------------------ cache
def cache_dir() -> str:
    return os.environ.get(
        "MS_MUSIC_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "ms_music"),
    )


def _cache_key(filepath, ms_level, rt_range, mobility_range):
    path = os.path.abspath(filepath)
    try:
        st = os.stat(path)
        stamp = (st.st_size, st.st_mtime_ns)
    except OSError:
        stamp = (None, None)
    payload = json.dumps(
        [
            _LOADER_VERSION,
            path,
            stamp,
            int(ms_level),
            rt_range and list(map(float, rt_range)),
            mobility_range and list(map(float, mobility_range)),
        ]
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def _cache_path(key):
    return os.path.join(cache_dir(), f"{key}.npz")


def _opt_array(values, dtype=float):
    return np.array([np.nan if v is None else v for v in values], dtype=dtype)


def _write_cache(key, run: LoadedRun, source: Optional[dict] = None):
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        lengths = np.array([len(df) for df in run.spectra], dtype=np.int64)
        has_mob = run.has_mobility and all(
            "mobility" in df.columns for df in run.spectra
        )
        iso = [w or (np.nan, np.nan) for w in run.isolation_windows]
        arrays = dict(
            lengths=lengths,
            bins=np.concatenate([df.index.to_numpy() for df in run.spectra]),
            intensities=np.concatenate(
                [df["intensities"].to_numpy() for df in run.spectra]
            ),
            rt=_opt_array(run.retention_times),
            precursor=_opt_array(run.precursor_mz),
            charge=_opt_array(run.charges),
            iso=np.array(iso, dtype=float).reshape(-1, 2),
            spec_mob=_opt_array(run.spectrum_mobility),
            spec_max=np.asarray(run.spectrum_max_intensity, dtype=float),
            meta=np.array(
                json.dumps(
                    {
                        "ms_level": run.ms_level,
                        "max_intensity": run.max_intensity,
                        "min_mz": run.min_mz,
                        "max_mz": run.max_mz,
                        "mobility_unit": run.mobility_unit,
                        "mobility_range": run.mobility_range,
                        "has_mob": has_mob,
                        "source": source,
                    }
                )
            ),
        )
        if has_mob:
            arrays["mobility"] = np.concatenate(
                [df["mobility"].to_numpy() for df in run.spectra]
            )
        if run.mobility_map is not None:
            arrays["mob_map"] = run.mobility_map
            arrays["mob_profiles"] = run.mobility_profiles
            arrays["mob_edges"] = run.mobility_edges
        tmp = _cache_path(key) + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, _cache_path(key))
    except Exception as exc:  # caching is best-effort
        print(f"Warning: could not write cache: {exc}")


def list_cache() -> List[dict]:
    """Cached binned loads, newest first.

    Each entry has ``key``, ``bytes`` and, for entries written by this
    version, ``file``, ``path``, ``ms_level``, ``rt_range``,
    ``mobility_range``, ``spectra`` and ``created``. Older entries only
    have ``key`` and ``bytes`` (and ``file`` is None).
    """
    folder = cache_dir()
    if not os.path.isdir(folder):
        return []
    entries = []
    for name in os.listdir(folder):
        if not name.endswith(".npz") or name.endswith(".tmp.npz"):
            continue
        path = os.path.join(folder, name)
        entry = {
            "key": name[:-4],
            "bytes": os.path.getsize(path),
            "file": None,
            "mtime": os.path.getmtime(path),
        }
        try:
            with np.load(path, allow_pickle=False) as z:  # reads meta only
                source = json.loads(str(z["meta"])).get("source") or {}
            entry.update(source)
        except Exception:
            pass
        entries.append(entry)
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


def remove_cache(key: str) -> bool:
    """Delete one cache entry. Returns whether it existed."""
    if (
        not key
        or os.sep in key
        or not all(c in "0123456789abcdef" for c in key)
    ):
        raise ValueError(f"Not a cache key: {key!r}")
    path = _cache_path(key)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def clear_cache() -> int:
    """Delete every cache entry. Returns how many were removed."""
    removed = 0
    for entry in list_cache():
        removed += remove_cache(entry["key"])
    return removed


def _read_cache(key) -> Optional[LoadedRun]:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            bounds = np.concatenate(([0], np.cumsum(z["lengths"])))
            bins, sums = z["bins"], z["intensities"]
            mob = z["mobility"] if meta["has_mob"] else None
            spectra = [
                _binned_frame(
                    bins[a:b], sums[a:b], None if mob is None else mob[a:b]
                )
                for a, b in zip(bounds[:-1], bounds[1:])
            ]

            def opt(arr, cast=float):
                return [None if np.isnan(v) else cast(v) for v in arr]

            iso = [
                None if np.isnan(lo) else (float(lo), float(hi))
                for lo, hi in z["iso"]
            ]
            return LoadedRun(
                ms_level=meta["ms_level"],
                spectra=spectra,
                max_intensity=meta["max_intensity"],
                min_mz=meta["min_mz"],
                max_mz=meta["max_mz"],
                retention_times=opt(z["rt"]),
                precursor_mz=opt(z["precursor"]),
                charges=opt(z["charge"], int),
                isolation_windows=iso,
                spectrum_mobility=opt(z["spec_mob"]),
                spectrum_max_intensity=z["spec_max"].tolist(),
                mobility_unit=meta["mobility_unit"],
                mobility_range=(
                    tuple(meta["mobility_range"])
                    if meta["mobility_range"]
                    else None
                ),
                from_cache=True,
                mobility_map=z["mob_map"] if "mob_map" in z.files else None,
                mobility_profiles=(
                    z["mob_profiles"] if "mob_profiles" in z.files else None
                ),
                mobility_edges=(
                    z["mob_edges"] if "mob_edges" in z.files else None
                ),
            )
    except Exception as exc:
        print(f"Warning: ignoring unreadable cache {path}: {exc}")
        return None


def preprocess_spectra(spectra_list: list):
    """
    Bin raw spectra (from :func:`load_mzml_data`) to integer m/z.

    Returns ``(processed_spectra_dfs, max_intensity, min_mz, max_mz)``.
    Spectra dicts may carry a ``mobility`` array, which becomes an
    intensity-weighted ``mobility`` column.
    """
    if not spectra_list:
        print("Warning: Empty spectra list provided to preprocess_spectra.")
        return [], 0.0, 0.0, 0.0

    print(f"Preprocessing {len(spectra_list)} spectra...")
    processed, n_peaks = [], 0
    max_intensity = 0.0
    min_mz, max_mz = np.inf, -np.inf
    for spectrum in spectra_list:
        mz, intensity = spectrum["mz"], spectrum["intensity"]
        if len(mz) == 0:
            processed.append(_empty_frame())
            continue
        bins, sums, mob = bin_spectrum(mz, intensity, spectrum.get("mobility"))
        processed.append(_binned_frame(bins, sums, mob))
        n_peaks += len(mz)
        max_intensity = max(max_intensity, float(np.max(intensity)))
        min_mz, max_mz = min(min_mz, bins[0]), max(max_mz, bins[-1])

    if n_peaks == 0:
        return [_empty_frame() for _ in spectra_list], 0.0, 0.0, 0.0

    print(
        f"✓ Preprocessed {n_peaks:,} peaks | m/z: {min_mz:.1f}-{max_mz:.1f} "
        f"| max intensity: {max_intensity:.2e}"
    )
    return processed, float(max_intensity), float(min_mz), float(max_mz)


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
    # Float audio is taken as full scale at +/-1 (clipped, not normalized:
    # normalize first with normalize_audio_to_16bit if that is wanted).
    if audio_data.dtype != np.int16:
        audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
    # Stereo arrays are (channels, samples); WAV writers want (samples, ch).
    if audio_data.ndim == 2 and audio_data.shape[0] <= 8 < audio_data.shape[1]:
        audio_data = audio_data.T

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
