import numpy as np
import librosa
from scipy import signal
import warnings

TAU = 2 * np.pi

# Export all effect functions
__all__ = [
    # Basic filters
    "apply_hpss",
    "apply_notch_filter",
    "apply_butterworth_filter",
    "apply_chebyshev1_filter",
    "apply_highpass_filter",
    "apply_lowpass_filter",
    "apply_bandpass_filter",
    "apply_bandstop_filter",
    # Time-based effects
    "apply_reverb",
    "apply_delay",
    "apply_echo",
    "apply_chorus",
    "apply_flanger",
    "apply_phaser",
    # Dynamics processing
    "apply_compressor",
    "apply_gate",
    "apply_limiter",
    "apply_expander",
    # Distortion effects
    "apply_overdrive",
    "apply_fuzz",
    "apply_bitcrusher",
    "apply_waveshaper",
    # Modulation effects
    "apply_tremolo",
    "apply_vibrato",
    "apply_ring_modulation",
    "apply_auto_wah",
    # Spectral processing
    "apply_spectral_gate",
    "apply_spectral_compressor",
    "apply_pitch_shift",
    "apply_time_stretch",
    "apply_formant_shift",
    # Other effects
    "apply_pll_filter",
    "apply_adaptive_filter",
    "apply_granular_synthesis",
    "apply_convolution_reverb",
    "apply_multiband_compressor",
    # Equalizers
    "apply_parametric_eq",
    "apply_graphic_eq",
    "apply_shelving_eq",
    # Utility effects
    "apply_normalize",
    "apply_fade_in",
    "apply_fade_out",
    "apply_crossfade",
]


def _validate_audio_input(
    audio_data: np.ndarray, effect_name: str
) -> np.ndarray:
    """Validate and prepare audio input for processing."""
    if audio_data.size == 0:
        warnings.warn(f"{effect_name}: Input audio is empty")
        return np.array([], dtype=np.float32)

    audio_float = audio_data.astype(np.float32)
    if audio_float.ndim > 1:
        audio_float = librosa.to_mono(audio_float)

    return audio_float


def _apply_dry_wet_mix(
    dry_signal: np.ndarray, wet_signal: np.ndarray, mix: float
) -> np.ndarray:
    """Apply dry/wet mixing with proper level matching."""
    mix = np.clip(mix, 0.0, 1.0)

    # Level match wet signal to dry signal
    dry_rms = np.sqrt(np.mean(dry_signal**2))
    wet_rms = np.sqrt(np.mean(wet_signal**2))

    if wet_rms > 1e-10 and dry_rms > 1e-10:
        level_correction = dry_rms / wet_rms
        wet_signal = wet_signal * level_correction

    return (1.0 - mix) * dry_signal + mix * wet_signal


# Basic filters
def apply_lowpass_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    cutoff_freq: float,
    order: int = 4,
    filter_type: str = "butterworth",
) -> np.ndarray:
    """
    Apply low-pass filter to remove high frequencies.

    Args:
        audio_data: Input audio signal
        sample_rate: Sample rate in Hz
        cutoff_freq: Cutoff frequency in Hz
        order: Filter order (higher = steeper rolloff)
        filter_type: 'butterworth', 'chebyshev1', or 'elliptic'

    Returns:
        Filtered audio signal
    """
    audio_float = _validate_audio_input(audio_data, "Low-pass Filter")
    if audio_float.size == 0:
        return audio_float

    nyquist = sample_rate / 2
    if cutoff_freq >= nyquist:
        warnings.warn(
            "Cutoff frequency at or above Nyquist, returning original audio"
        )
        return audio_float

    normalized_cutoff = cutoff_freq / nyquist

    try:
        if filter_type == "butterworth":
            b, a = signal.butter(order, normalized_cutoff, btype="low")
        elif filter_type == "chebyshev1":
            b, a = signal.cheby1(order, 1, normalized_cutoff, btype="low")
        elif filter_type == "elliptic":
            b, a = signal.ellip(order, 1, 40, normalized_cutoff, btype="low")
        else:
            b, a = signal.butter(order, normalized_cutoff, btype="low")

        return signal.filtfilt(b, a, audio_float)
    except Exception as e:
        print(f"Filter error: {e}")
        return audio_float


def apply_highpass_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    cutoff_freq: float,
    order: int = 4,
    filter_type: str = "butterworth",
) -> np.ndarray:
    """Apply high-pass filter to remove low frequencies."""
    audio_float = _validate_audio_input(audio_data, "High-pass Filter")
    if audio_float.size == 0:
        return audio_float

    nyquist = sample_rate / 2
    if cutoff_freq >= nyquist:
        return np.zeros_like(audio_float)

    normalized_cutoff = cutoff_freq / nyquist

    try:
        if filter_type == "butterworth":
            b, a = signal.butter(order, normalized_cutoff, btype="high")
        elif filter_type == "chebyshev1":
            b, a = signal.cheby1(order, 1, normalized_cutoff, btype="high")
        elif filter_type == "elliptic":
            b, a = signal.ellip(order, 1, 40, normalized_cutoff, btype="high")
        else:
            b, a = signal.butter(order, normalized_cutoff, btype="high")

        return signal.filtfilt(b, a, audio_float)
    except Exception as e:
        print(f"Filter error: {e}")
        return audio_float


def apply_bandpass_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    order: int = 4,
) -> np.ndarray:
    """Apply band-pass filter to keep only frequencies in specified range."""
    audio_float = _validate_audio_input(audio_data, "Band-pass Filter")
    if audio_float.size == 0:
        return audio_float

    nyquist = sample_rate / 2
    low_freq = max(1, min(low_freq, nyquist - 1))
    high_freq = max(low_freq + 1, min(high_freq, nyquist - 1))

    normalized_freqs = [low_freq / nyquist, high_freq / nyquist]

    try:
        b, a = signal.butter(order, normalized_freqs, btype="band")
        return signal.filtfilt(b, a, audio_float)
    except Exception as e:
        print(f"Filter error: {e}")
        return audio_float


def apply_bandstop_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    order: int = 4,
) -> np.ndarray:
    """Apply band-stop (notch) filter to
    remove frequencies in specified range."""
    audio_float = _validate_audio_input(audio_data, "Band-stop Filter")
    if audio_float.size == 0:
        return audio_float

    nyquist = sample_rate / 2
    low_freq = max(1, min(low_freq, nyquist - 1))
    high_freq = max(low_freq + 1, min(high_freq, nyquist - 1))

    normalized_freqs = [low_freq / nyquist, high_freq / nyquist]

    try:
        b, a = signal.butter(order, normalized_freqs, btype="bandstop")
        return signal.filtfilt(b, a, audio_float)
    except Exception as e:
        print(f"Filter error: {e}")
        return audio_float


def apply_notch_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    notch_freq: float,
    quality_factor: float = 10.0,
) -> np.ndarray:
    """Apply notch filter to remove a specific frequency."""
    if audio_data.size == 0:
        return np.copy(audio_data)
    if not (0 < notch_freq < sample_rate / 2):
        print(
            f"Warning (Notch Filter): "
            f"Invalid notch_freq {notch_freq} Hz. Skipping."
        )
        return np.copy(audio_data)

    b, a = signal.iirnotch(notch_freq, quality_factor, fs=sample_rate)
    return signal.filtfilt(b, a, audio_data)


def apply_butterworth_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    cutoff_freq,
    btype: str = "low",
    order: int = 4,
    gustafson_method: bool = False,
) -> np.ndarray:
    """Apply Butterworth filter."""
    if audio_data.size == 0:
        return np.copy(audio_data)
    nyquist = 0.5 * sample_rate
    normalized_cutoff = np.array(cutoff_freq) / nyquist

    if np.any(normalized_cutoff <= 0) or np.any(normalized_cutoff >= 1):
        print(
            f"Warning (Butterworth): "
            f"Invalid cutoff_freq {cutoff_freq} Hz. Skipping."
        )
        return np.copy(audio_data)

    b, a = signal.butter(order, normalized_cutoff, btype=btype, analog=False)

    filtfilt_method = "gust" if gustafson_method else "pad"
    try:
        return signal.filtfilt(b, a, audio_data, method=filtfilt_method)
    except ValueError:
        print(
            f"Warning (Butterworth): filtfilt "
            f"method='{filtfilt_method}' failed. Using default 'pad'."
        )
        return signal.filtfilt(b, a, audio_data, method="pad")


def apply_chebyshev1_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    cutoff_freq,
    ripple_db: float,
    btype: str = "low",
    order: int = 4,
) -> np.ndarray:
    """Apply Chebyshev Type I filter."""
    if audio_data.size == 0:
        return np.copy(audio_data)
    nyquist = 0.5 * sample_rate
    normalized_cutoff = np.array(cutoff_freq) / nyquist

    if np.any(normalized_cutoff <= 0) or np.any(normalized_cutoff >= 1):
        print(
            f"Warning (Chebyshev1): Invalid "
            f"cutoff_freq {cutoff_freq} Hz. Skipping."
        )
        return np.copy(audio_data)
    if ripple_db <= 0:
        print(
            f"Warning (Chebyshev1): ripple_db must be positive. "
            f"Given: {ripple_db}. Skipping."
        )
        return np.copy(audio_data)

    b, a = signal.cheby1(
        order, ripple_db, normalized_cutoff, btype=btype, analog=False
    )
    return signal.filtfilt(b, a, audio_data)


# Time-based effects
def apply_reverb(
    audio_data: np.ndarray,
    sample_rate: int,
    reverb_time_s: float = 0.7,
    decay_factor: float = 6.908,
    dry_wet_mix: float = 0.3,
    room_size: float = 0.5,
    damping: float = 0.5,
    early_reflections: bool = True,
) -> np.ndarray:
    """
    Reverb with early reflections and frequency-dependent decay.

    Args:
        reverb_time_s: T60 reverb time in seconds
        decay_factor: Exponential decay factor
        dry_wet_mix: Wet signal ratio (0.0 = dry, 1.0 = fully wet)
        room_size: Room size simulation (0.0-1.0)
        damping: High-frequency damping (0.0-1.0)
        early_reflections: Include early reflection simulation
    """
    audio_float = _validate_audio_input(audio_data, "Reverb")
    if audio_float.size == 0:
        return audio_float

    if reverb_time_s <= 0:
        return audio_float

    # Generate impulse response
    impulse_len_samples = int(reverb_time_s * sample_rate * (0.5 + room_size))
    if impulse_len_samples == 0:
        return audio_float

    # Create noise-based impulse response
    impulse_response = np.random.randn(impulse_len_samples).astype(np.float32)
    time_points = np.arange(impulse_len_samples) / sample_rate

    # Apply exponential decay with frequency-dependent damping
    if reverb_time_s < 1e-6:
        reverb_time_s = 1e-6

    # Basic exponential decay
    decay_envelope = np.exp(-decay_factor * time_points / reverb_time_s)

    # Apply frequency-dependent damping
    if damping > 0:
        # High-frequency rolloff for more realistic reverb
        cutoff_freq = 8000 * (1 - damping)  # Higher damping = lower cutoff
        nyquist = sample_rate / 2
        if cutoff_freq < nyquist:
            b, a = signal.butter(2, cutoff_freq / nyquist, btype="low")
            impulse_response = signal.filtfilt(b, a, impulse_response)

    impulse_response *= decay_envelope

    # Add early reflections
    if early_reflections:
        early_delay_samples = int(
            0.02 * sample_rate * room_size
        )  # 0-20ms delay
        if early_delay_samples > 0 and early_delay_samples < len(
            impulse_response
        ):
            # Add a delayed and attenuated copy for early reflections
            early_reflection = np.zeros_like(impulse_response)
            early_reflection[early_delay_samples:] = (
                impulse_response[:-early_delay_samples] * 0.3
            )
            impulse_response += early_reflection

    # Normalize impulse response
    max_ir = np.max(np.abs(impulse_response))
    if max_ir > 1e-9:
        impulse_response /= max_ir

    # Apply convolution
    wet_signal = signal.convolve(audio_float, impulse_response, mode="same")

    return _apply_dry_wet_mix(audio_float, wet_signal, dry_wet_mix)


def apply_delay(
    audio_data: np.ndarray,
    sample_rate: int,
    delay_ms: float = 250.0,
    feedback: float = 0.3,
    dry_wet_mix: float = 0.3,
    num_taps: int = 1,
) -> np.ndarray:
    """
    Multi-tap delay with feedback control.

    Args:
        delay_ms: Delay time in milliseconds
        feedback: Feedback amount (0.0-0.95)
        dry_wet_mix: Wet signal ratio
        num_taps: Number of delay taps
    """
    audio_float = _validate_audio_input(audio_data, "Delay")
    if audio_float.size == 0:
        return audio_float

    delay_samples = int(delay_ms * sample_rate / 1000.0)
    if delay_samples <= 0:
        return audio_float

    feedback = np.clip(feedback, 0.0, 0.95)  # Prevent instability

    # Create delay line
    delay_line = np.zeros(len(audio_float) + delay_samples * num_taps)
    delay_line[: len(audio_float)] = audio_float

    wet_signal = np.zeros_like(audio_float)

    # Process each tap
    for tap in range(num_taps):
        tap_delay = delay_samples * (tap + 1)
        tap_gain = 0.7**tap  # Decrease gain for each tap

        for i in range(len(audio_float)):
            if i + tap_delay < len(delay_line):
                # Add delayed signal with feedback
                delayed_sample = delay_line[i + tap_delay]
                delay_line[i + tap_delay] += (
                    audio_float[i] * feedback * tap_gain
                )
                wet_signal[i] += delayed_sample * tap_gain

    return _apply_dry_wet_mix(audio_float, wet_signal, dry_wet_mix)


def apply_echo(
    audio_data: np.ndarray,
    sample_rate: int,
    echo_delay_ms: float = 500.0,
    echo_gain: float = 0.4,
    num_echoes: int = 3,
) -> np.ndarray:
    """
    Multi-echo effect with decreasing amplitude.
    """
    audio_float = _validate_audio_input(audio_data, "Echo")
    if audio_float.size == 0:
        return audio_float

    echo_samples = int(echo_delay_ms * sample_rate / 1000.0)
    if echo_samples <= 0:
        return audio_float

    # Create output buffer with space for echoes
    total_length = len(audio_float) + echo_samples * num_echoes
    output = np.zeros(total_length)
    output[: len(audio_float)] = audio_float

    # Add each echo
    for echo_num in range(1, num_echoes + 1):
        echo_start = echo_samples * echo_num
        echo_end = echo_start + len(audio_float)
        echo_amplitude = echo_gain**echo_num

        if echo_end <= total_length:
            output[echo_start:echo_end] += audio_float * echo_amplitude

    # Trim back to original length and normalize
    output = output[: len(audio_float)]
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output /= max_val

    return output.astype(np.float32)


def apply_chorus(
    audio_data: np.ndarray,
    sample_rate: int,
    delay_ms: float = 20.0,
    depth_ms: float = 2.0,
    rate_hz: float = 0.7,
    dry_wet_mix: float = 0.5,
    feedback: float = 0.2,
    num_voices: int = 2,
) -> np.ndarray:
    """
    Chorus effect with multiple voices and stereo width.
    """
    audio_float = _validate_audio_input(audio_data, "Chorus")
    if audio_float.size == 0:
        return audio_float

    delay_samples_base = int(delay_ms * sample_rate / 1000.0)
    depth_samples = int(depth_ms * sample_rate / 1000.0)
    feedback = np.clip(feedback, 0.0, 0.9)

    wet_signal = np.zeros_like(audio_float)

    for voice in range(num_voices):
        # Different LFO phase and rate for each voice
        phase_offset = (TAU * voice) / num_voices
        voice_rate = rate_hz * (
            0.8 + 0.4 * voice / num_voices
        )  # Slight rate variations

        t = np.arange(len(audio_float)) / sample_rate
        lfo = depth_samples * np.sin(TAU * voice_rate * t + phase_offset)

        voice_output = np.zeros_like(audio_float)

        # Simple delay line implementation
        for i in range(len(audio_float)):
            delay_amount = delay_samples_base + lfo[i]
            delay_idx = i - int(delay_amount)

            if delay_idx >= 0:
                # Linear interpolation for fractional delays
                frac = delay_amount - int(delay_amount)
                if delay_idx + 1 < len(audio_float):
                    sample = (1 - frac) * audio_float[
                        delay_idx
                    ] + frac * audio_float[delay_idx + 1]
                else:
                    sample = audio_float[delay_idx]
                voice_output[i] = sample

        wet_signal += voice_output / num_voices

    return _apply_dry_wet_mix(audio_float, wet_signal, dry_wet_mix)


def apply_flanger(
    audio_data: np.ndarray,
    sample_rate: int,
    delay_ms: float = 5.0,
    depth_ms: float = 3.0,
    rate_hz: float = 0.3,
    feedback: float = 0.4,
    dry_wet_mix: float = 0.5,
) -> np.ndarray:
    """
    Flanger effect with variable delay modulation.
    """
    audio_float = _validate_audio_input(audio_data, "Flanger")
    if audio_float.size == 0:
        return audio_float

    base_delay_samples = int(delay_ms * sample_rate / 1000.0)
    depth_samples = int(depth_ms * sample_rate / 1000.0)
    feedback = np.clip(feedback, -0.95, 0.95)

    # Create LFO for delay modulation
    t = np.arange(len(audio_float)) / sample_rate
    lfo = np.sin(TAU * rate_hz * t)

    # Create delay line with feedback
    max_delay = base_delay_samples + depth_samples + 1
    delay_line = np.zeros(len(audio_float) + max_delay)
    delay_line[: len(audio_float)] = audio_float

    wet_signal = np.zeros_like(audio_float)

    for i in range(len(audio_float)):
        # Calculate variable delay
        current_delay = base_delay_samples + depth_samples * lfo[i]
        delay_idx = int(current_delay)
        frac = current_delay - delay_idx

        # Get delayed sample with interpolation
        if i - delay_idx >= 0 and i - delay_idx - 1 >= 0:
            delayed_sample = (1 - frac) * delay_line[
                i - delay_idx
            ] + frac * delay_line[i - delay_idx - 1]
        else:
            delayed_sample = 0.0

        # Add feedback
        if i + delay_idx < len(delay_line):
            delay_line[i + delay_idx] += delayed_sample * feedback

        wet_signal[i] = delayed_sample

    return _apply_dry_wet_mix(audio_float, wet_signal, dry_wet_mix)


def apply_phaser(
    audio_data: np.ndarray,
    sample_rate: int,
    rate_hz: float = 0.5,
    depth: float = 1.0,
    stages: int = 4,
    feedback: float = 0.3,
    dry_wet_mix: float = 0.5,
) -> np.ndarray:
    """Phaser effect using all-pass filters."""
    audio_float = _validate_audio_input(audio_data, "Phaser")
    if audio_float.size == 0:
        return audio_float

    feedback = np.clip(feedback, -0.95, 0.95)

    # Create LFO for modulation
    t = np.arange(len(audio_float)) / sample_rate
    lfo = np.sin(TAU * rate_hz * t)

    # All-pass filter frequency range
    min_freq = 200.0
    max_freq = 2000.0

    wet_signal = np.copy(audio_float)

    # Create all-pass filters for each stage
    for stage in range(stages):
        stage_output = np.zeros_like(audio_float)
        z1 = 0.0  # State variable for all-pass

        for i in range(len(audio_float)):
            # Modulate all-pass frequency
            mod_freq = min_freq + (
                max_freq - min_freq
                ) * (0.5 + 0.5 * depth * lfo[i])

            # All-pass coefficient from frequency
            # a = (tan(pi * fc / fs) - 1) / (tan(pi * fc / fs) + 1)
            tan_val = np.tan(np.pi * mod_freq / sample_rate)
            a = (tan_val - 1) / (tan_val + 1)

            # First-order all-pass: y[n] = a*x[n] + x[n-1] - a*y[n-1]
            # Simplified: y[n] = a*x[n] + z1
            # where z1 = x[n-1] - a*y[n-1]
            stage_output[i] = a * wet_signal[i] + z1
            z1 = wet_signal[i] - a * stage_output[i]

        wet_signal = stage_output

    # Add feedback
    wet_signal = audio_float + wet_signal * feedback

    return _apply_dry_wet_mix(audio_float, wet_signal, dry_wet_mix)


# Dynamics processing
def apply_compressor(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    makeup_gain_db: float = 0.0,
    knee_width_db: float = 2.0,
) -> np.ndarray:
    """
    Compressor with soft knee and automatic gain compensation.

    Args:
        threshold_db: Compression threshold in dB
        ratio: Compression ratio (1.0 = no compression, inf = limiter)
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        makeup_gain_db: Post-compression gain boost in dB
        knee_width_db: Soft knee width in dB
    """
    audio_float = _validate_audio_input(audio_data, "Compressor")
    if audio_float.size == 0:
        return audio_float

    # Convert parameters
    threshold_linear = 10 ** (threshold_db / 20.0)
    makeup_gain_linear = 10 ** (makeup_gain_db / 20.0)
    attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
    release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))

    # Initialize envelope follower
    envelope = 0.0
    output = np.zeros_like(audio_float)

    for i in range(len(audio_float)):
        input_level = abs(audio_float[i])

        # Envelope follower (peak detection)
        if input_level > envelope:
            envelope = envelope * attack_coeff + input_level * (
                1 - attack_coeff
            )
        else:
            envelope = envelope * release_coeff + input_level * (
                1 - release_coeff
            )

        # Calculate gain reduction
        if envelope > threshold_linear:
            # Soft knee implementation
            over_threshold_db = 20 * np.log10(envelope / threshold_linear)

            if over_threshold_db < knee_width_db:
                # Soft knee region
                knee_ratio = over_threshold_db / knee_width_db
                soft_ratio = 1.0 + (ratio - 1.0) * knee_ratio * knee_ratio
                gain_reduction_db = over_threshold_db * (
                    1.0 - 1.0 / soft_ratio
                )
            else:
                # Above knee
                gain_reduction_db = over_threshold_db * (1.0 - 1.0 / ratio)

            gain_reduction_linear = 10 ** (-gain_reduction_db / 20.0)
        else:
            gain_reduction_linear = 1.0

        # Apply compression and makeup gain
        output[i] = audio_float[i] * gain_reduction_linear * makeup_gain_linear

    return output


def apply_gate(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    ratio: float = 10.0,
    attack_ms: float = 1.0,
    hold_ms: float = 10.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    """
    Noise gate to reduce background noise.
    """
    audio_float = _validate_audio_input(audio_data, "Gate")
    if audio_float.size == 0:
        return audio_float

    threshold_linear = 10 ** (threshold_db / 20.0)
    attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
    release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))
    hold_samples = int(hold_ms * sample_rate / 1000.0)

    envelope = 0.0
    gate_state = 0.0
    hold_counter = 0
    output = np.zeros_like(audio_float)

    for i in range(len(audio_float)):
        input_level = abs(audio_float[i])

        # Envelope follower
        if input_level > envelope:
            envelope = envelope * attack_coeff + input_level * (
                1 - attack_coeff
            )
        else:
            envelope = envelope * release_coeff + input_level * (
                1 - release_coeff
            )

        # Gate logic with hold
        if envelope > threshold_linear:
            # Signal above threshold - open gate
            gate_state = min(1.0, gate_state + (1 - attack_coeff))
            hold_counter = hold_samples
        elif hold_counter > 0:
            # Hold period - keep gate open
            hold_counter -= 1
        else:
            # Close gate
            reduction = 1.0 - 1.0 / ratio
            gate_state = max(0.0, gate_state - (1 - release_coeff))
            gate_state = max(gate_state, 1.0 - reduction)

        output[i] = audio_float[i] * gate_state

    return output


def apply_limiter(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_db: float = -1.0,
    release_ms: float = 50.0,
    lookahead_ms: float = 5.0,
) -> np.ndarray:
    """
    Brick-wall limiter with lookahead.
    """
    audio_float = _validate_audio_input(audio_data, "Limiter")
    if audio_float.size == 0:
        return audio_float

    threshold_linear = 10 ** (threshold_db / 20.0)
    release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))
    lookahead_samples = int(lookahead_ms * sample_rate / 1000.0)

    # Pad audio for lookahead
    padded_audio = np.pad(audio_float, (lookahead_samples, 0), mode="constant")
    output = np.zeros_like(padded_audio)

    gain_reduction = 1.0

    for i in range(len(padded_audio)):
        # Look ahead for peaks
        future_peak = 0.0
        for j in range(min(lookahead_samples, len(padded_audio) - i)):
            future_peak = max(future_peak, abs(padded_audio[i + j]))

        # Calculate required gain reduction
        if future_peak * gain_reduction > threshold_linear:
            required_gain = threshold_linear / future_peak
            gain_reduction = min(gain_reduction, required_gain)
        else:
            # Release
            gain_reduction = min(1.0, gain_reduction + (1 - release_coeff))

        output[i] = padded_audio[i] * gain_reduction

    return output[lookahead_samples:]


def apply_expander(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_db: float = -30.0,
    ratio: float = 2.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    """
    Expander to increase dynamic range.
    """
    audio_float = _validate_audio_input(audio_data, "Expander")
    if audio_float.size == 0:
        return audio_float

    threshold_linear = 10 ** (threshold_db / 20.0)
    attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
    release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))

    envelope = 0.0
    output = np.zeros_like(audio_float)

    for i in range(len(audio_float)):
        input_level = abs(audio_float[i])

        # Envelope follower
        if input_level > envelope:
            envelope = envelope * attack_coeff + input_level * (
                1 - attack_coeff
            )
        else:
            envelope = envelope * release_coeff + input_level * (
                1 - release_coeff
            )

        # Expansion
        if envelope < threshold_linear and envelope > 1e-10:
            under_threshold_db = 20 * np.log10(envelope / threshold_linear)
            gain_change_db = under_threshold_db * (ratio - 1.0)
            gain_linear = 10 ** (gain_change_db / 20.0)
        else:
            gain_linear = 1.0

        output[i] = audio_float[i] * gain_linear

    return output


# Distortion effects
def apply_overdrive(
    audio_data: np.ndarray,
    sample_rate: int,
    drive: float = 5.0,
    tone: float = 0.5,
    output_level: float = 0.5,
) -> np.ndarray:
    """
    Tube-style overdrive with tone control.
    """
    audio_float = _validate_audio_input(audio_data, "Overdrive")
    if audio_float.size == 0:
        return audio_float

    # Pre-emphasis for more natural distortion
    if tone > 0.5:
        # Boost high frequencies
        emphasis_freq = 1000 + (tone - 0.5) * 4000
        b, a = signal.butter(
            1, emphasis_freq / (sample_rate / 2), btype="high"
        )
        audio_float = audio_float + 0.3 * signal.filtfilt(b, a, audio_float)

    # Apply overdrive using asymmetric clipping
    driven_signal = audio_float * drive

    # Asymmetric soft clipping (tube-like)
    output = np.zeros_like(driven_signal)
    for i in range(len(driven_signal)):
        x = driven_signal[i]
        if x > 0:
            # Positive half - softer clipping
            output[i] = np.tanh(x * 0.7)
        else:
            # Negative half - harder clipping
            output[i] = np.tanh(x * 0.9)

    # Post-EQ based on tone control
    if tone < 0.5:
        # Roll off high frequencies
        cutoff_freq = 2000 + tone * 4000
        b, a = signal.butter(2, cutoff_freq / (sample_rate / 2), btype="low")
        output = signal.filtfilt(b, a, output)

    # Output level control
    return output * output_level


def apply_fuzz(
    audio_data: np.ndarray,
    sample_rate: int,
    fuzz_amount: float = 8.0,
    gate_threshold: float = 0.1,
    output_level: float = 0.3,
) -> np.ndarray:
    """
    Classic fuzz effect with gating.
    """
    audio_float = _validate_audio_input(audio_data, "Fuzz")
    if audio_float.size == 0:
        return audio_float

    # Apply extreme gain
    fuzzed = audio_float * fuzz_amount

    # Hard clipping with asymmetry
    fuzzed = np.clip(fuzzed, -0.8, 1.0)

    # Additional harmonic generation using polynomial
    fuzzed = fuzzed - 0.3 * fuzzed**3 + 0.1 * fuzzed**5

    # Gate very quiet signals
    gate_mask = np.abs(fuzzed) > gate_threshold
    fuzzed = fuzzed * gate_mask

    # Low-pass filter to remove harsh high frequencies
    b, a = signal.butter(2, 3000 / (sample_rate / 2), btype="low")
    fuzzed = signal.filtfilt(b, a, fuzzed)

    return fuzzed * output_level


def apply_bitcrusher(
    audio_data: np.ndarray,
    sample_rate: int,
    bit_depth: int = 8,
    downsample_factor: int = 4,
    dry_wet_mix: float = 1.0,
) -> np.ndarray:
    """
    Bitcrusher for lo-fi digital distortion.
    """
    audio_float = _validate_audio_input(audio_data, "Bitcrusher")
    if audio_float.size == 0:
        return audio_float

    # Bit depth reduction
    if bit_depth < 16:
        levels = 2**bit_depth
        quantized = np.round(audio_float * levels) / levels
    else:
        quantized = audio_float

    # Sample rate reduction
    if downsample_factor > 1:
        # Downsample
        downsampled_indices = np.arange(0, len(quantized), downsample_factor)
        downsampled_values = quantized[downsampled_indices]

        # Upsample with zero-order hold
        crushed = np.zeros_like(quantized)
        for i, idx in enumerate(downsampled_indices):
            end_idx = min(idx + downsample_factor, len(crushed))
            crushed[idx:end_idx] = downsampled_values[i]
    else:
        crushed = quantized

    return _apply_dry_wet_mix(audio_float, crushed, dry_wet_mix)


def apply_waveshaper(
    audio_data: np.ndarray,
    sample_rate: int,
    curve_type: str = "tanh",
    drive: float = 3.0,
    symmetry: float = 0.0,
) -> np.ndarray:
    """
    Waveshaper with various curve types.

    Args:
        curve_type: 'tanh', 'arctan', 'cubic', 'sine', 'hard_clip'
        drive: Drive amount
        symmetry: Asymmetry amount (-1 to 1)
    """
    audio_float = _validate_audio_input(audio_data, "Waveshaper")
    if audio_float.size == 0:
        return audio_float

    driven = audio_float * drive

    # Apply asymmetry
    if symmetry != 0.0:
        driven = driven + symmetry * driven**2

    # Apply waveshaping curve
    if curve_type == "tanh":
        shaped = np.tanh(driven)
    elif curve_type == "arctan":
        shaped = np.arctan(driven) / (np.pi / 2)
    elif curve_type == "cubic":
        shaped = driven - driven**3 / 3.0
        shaped = np.clip(shaped, -1.0, 1.0)
    elif curve_type == "sine":
        shaped = np.sin(driven * np.pi / 2)
    elif curve_type == "hard_clip":
        shaped = np.clip(driven, -1.0, 1.0)
    else:
        shaped = np.tanh(driven)  # Default

    return shaped * 0.7  # Scale down to prevent clipping


# Modulation effects
def apply_tremolo(
    audio_data: np.ndarray,
    sample_rate: int,
    rate_hz: float = 5.0,
    depth: float = 0.5,
    waveform: str = "sine",
) -> np.ndarray:
    """
    Tremolo effect (amplitude modulation).

    Args:
        rate_hz: Modulation rate in Hz
        depth: Modulation depth (0.0-1.0)
        waveform: 'sine', 'triangle', 'square', 'saw'
    """
    audio_float = _validate_audio_input(audio_data, "Tremolo")
    if audio_float.size == 0:
        return audio_float

    t = np.arange(len(audio_float)) / sample_rate

    # Generate modulation waveform
    if waveform == "sine":
        lfo = np.sin(TAU * rate_hz * t)
    elif waveform == "triangle":
        lfo = 2 * np.abs(2 * (rate_hz * t - np.floor(rate_hz * t + 0.5))) - 1
    elif waveform == "square":
        lfo = np.sign(np.sin(TAU * rate_hz * t))
    elif waveform == "saw":
        lfo = 2 * (rate_hz * t - np.floor(rate_hz * t + 0.5))
    else:
        lfo = np.sin(TAU * rate_hz * t)

    # Apply tremolo
    modulation = 1.0 - depth + depth * (1.0 + lfo) / 2.0
    return audio_float * modulation


def apply_vibrato(
    audio_data: np.ndarray,
    sample_rate: int,
    rate_hz: float = 4.0,
    depth_cents: float = 50.0,
) -> np.ndarray:
    """
    Vibrato effect (pitch modulation).
    """
    audio_float = _validate_audio_input(audio_data, "Vibrato")
    if audio_float.size == 0:
        return audio_float

    t = np.arange(len(audio_float)) / sample_rate

    # Convert cents to frequency ratio
    depth_ratio = 2 ** (depth_cents / 1200.0) - 1.0

    # Generate modulation
    lfo = np.sin(TAU * rate_hz * t)
    pitch_mod = 1.0 + depth_ratio * lfo

    # Apply pitch modulation using time-varying delay
    output = np.zeros_like(audio_float)
    phase = 0.0

    for i in range(len(audio_float)):
        # Variable sample rate
        phase_increment = pitch_mod[i]
        phase += phase_increment

        # Interpolate
        if phase < len(audio_float) - 1:
            idx = int(phase)
            frac = phase - idx
            output[i] = (1 - frac) * audio_float[idx] + frac * audio_float[
                idx + 1
            ]

    return output


def apply_ring_modulation(
    audio_data: np.ndarray,
    sample_rate: int,
    frequency_hz: float = 440.0,
    depth: float = 1.0,
    waveform: str = "sine",
) -> np.ndarray:
    """
    Ring modulation effect.
    """
    audio_float = _validate_audio_input(audio_data, "Ring Modulation")
    if audio_float.size == 0:
        return audio_float

    t = np.arange(len(audio_float)) / sample_rate

    # Generate carrier wave
    if waveform == "sine":
        carrier = np.sin(TAU * frequency_hz * t)
    elif waveform == "square":
        carrier = np.sign(np.sin(TAU * frequency_hz * t))
    elif waveform == "triangle":
        carrier = (
            2
            * np.abs(2 * (frequency_hz * t - np.floor(frequency_hz * t + 0.5)))
            - 1
        )
    elif waveform == "saw":
        carrier = 2 * (frequency_hz * t - np.floor(frequency_hz * t + 0.5))
    else:
        carrier = np.sin(TAU * frequency_hz * t)

    # Apply ring modulation
    modulated = audio_float * (1.0 - depth + depth * carrier)
    return modulated


def apply_auto_wah(
    audio_data: np.ndarray,
    sample_rate: int,
    sensitivity: float = 1.0,
    range_hz: tuple = (200, 2000),
    resonance: float = 2.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    """
    Auto-wah effect that follows input dynamics.
    """
    audio_float = _validate_audio_input(audio_data, "Auto-Wah")
    if audio_float.size == 0:
        return audio_float

    min_freq, max_freq = range_hz
    attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
    release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))

    envelope = 0.0
    output = np.zeros_like(audio_float)

    # Pre-calculate filter coefficients for different frequencies
    freq_steps = 50
    frequencies = np.linspace(min_freq, max_freq, freq_steps)
    filter_bank = []

    for freq in frequencies:
        w = freq / (sample_rate / 2)
        if w < 0.99:  # Prevent instability
            b, a = signal.iirpeak(freq, resonance, fs=sample_rate)
            filter_bank.append((b, a))
        else:
            filter_bank.append((np.array([1.0]), np.array([1.0])))

    # Process with time-varying filter
    for i in range(len(audio_float)):
        input_level = abs(audio_float[i])

        # Envelope follower
        if input_level > envelope:
            envelope = envelope * attack_coeff + input_level * (
                1 - attack_coeff
            )
        else:
            envelope = envelope * release_coeff + input_level * (
                1 - release_coeff
            )

        # Map envelope to frequency
        env_scaled = min(1.0, envelope * sensitivity)
        freq_idx = int(env_scaled * (freq_steps - 1))

        # Simple one-sample filter application (approximation)
        b, a = filter_bank[freq_idx]
        if len(b) > 1 and len(a) > 1:
            # Simple IIR filter approximation
            output[i] = b[0] * audio_float[i]
            if i > 0:
                output[i] += b[1] * audio_float[i - 1] - a[1] * output[i - 1]
        else:
            output[i] = audio_float[i]

    return output


# Spectral processing
def apply_hpss(
    audio_data: np.ndarray,
    sample_rate: int,
    margin: float = 16.0,
    harmonic: bool = True,
    percussive: bool = False,
) -> np.ndarray:
    """Harmonic-Percussive Source Separation."""
    if not harmonic and not percussive:
        print(
            "Warning (HPSS): Both harmonic and percussive are False. "
            "Returning original audio."
        )
        return np.copy(audio_data)
    if audio_data.size == 0:
        return np.copy(audio_data)

    audio_float = audio_data.astype(np.float32)
    D = librosa.stft(audio_float)
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=margin)

    output_audio = np.zeros_like(audio_float)
    if harmonic:
        output_audio += librosa.istft(D_harmonic, length=len(audio_float))
    if percussive:
        output_audio += librosa.istft(D_percussive, length=len(audio_float))
    return output_audio


def apply_spectral_gate(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_db: float = -30.0,
    n_fft: int = 2048
) -> np.ndarray:
    """
    Spectral gating - remove spectral components below threshold.
    """
    audio_float = _validate_audio_input(audio_data, "Spectral Gate")
    if audio_float.size == 0:
        return audio_float

    # STFT
    D = librosa.stft(audio_float, n_fft=n_fft)
    magnitude = np.abs(D)
    phase = np.angle(D)

    # Convert threshold to linear
    threshold_linear = 10 ** (threshold_db / 20.0)
    max_magnitude = np.max(magnitude)
    threshold_absolute = threshold_linear * max_magnitude

    # Apply gate
    gate_mask = magnitude > threshold_absolute
    gated_magnitude = magnitude * gate_mask

    # Reconstruct
    gated_stft = gated_magnitude * np.exp(1j * phase)
    return librosa.istft(gated_stft, length=len(audio_float))


def apply_spectral_compressor(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Spectral compressor - compress each frequency band independently.
    """
    audio_float = _validate_audio_input(audio_data, "Spectral Compressor")
    if audio_float.size == 0:
        return audio_float

    # STFT
    D = librosa.stft(audio_float, n_fft=n_fft)
    magnitude = np.abs(D)
    phase = np.angle(D)

    # Convert threshold
    threshold_linear = 10 ** (threshold_db / 20.0)

    # Compress each frequency bin
    compressed_magnitude = np.zeros_like(magnitude)

    for freq_bin in range(magnitude.shape[0]):
        bin_magnitude = magnitude[freq_bin, :]
        max_bin = np.max(bin_magnitude)
        threshold_absolute = threshold_linear * max_bin

        # Apply compression
        over_threshold = bin_magnitude > threshold_absolute
        compressed_magnitude[freq_bin, over_threshold] = (
            threshold_absolute
            + (bin_magnitude[over_threshold] - threshold_absolute) / ratio
        )
        compressed_magnitude[freq_bin, ~over_threshold] = bin_magnitude[
            ~over_threshold
        ]

    # Reconstruct
    compressed_stft = compressed_magnitude * np.exp(1j * phase)
    return librosa.istft(compressed_stft, length=len(audio_float))


def apply_pitch_shift(
    audio_data: np.ndarray, sample_rate: int, n_steps: float
) -> np.ndarray:
    """Pitch shifting using phase vocoder."""
    audio_float = _validate_audio_input(audio_data, "Pitch Shift")
    if audio_float.size == 0:
        return audio_float

    try:
        return librosa.effects.pitch_shift(
            y=audio_float, sr=sample_rate, n_steps=n_steps, res_type="soxr_hq"
        )
    except Exception as e:
        print(f"Pitch shift error: {e}")
        return audio_float


def apply_time_stretch(
        audio_data: np.ndarray,
        sample_rate: int,
        rate: float
        ) -> np.ndarray:
    """Time stretching using phase vocoder."""
    audio_float = _validate_audio_input(audio_data, "Time Stretch")
    if audio_float.size == 0:
        return audio_float

    if rate <= 0:
        print("Warning: Rate must be positive")
        return audio_float

    try:
        return librosa.effects.time_stretch(y=audio_float, rate=rate)
    except Exception as e:
        print(f"Time stretch error: {e}")
        return audio_float


def apply_formant_shift(
    audio_data: np.ndarray,
    sample_rate: int,
    shift_semitones: float = 0.0,
    formant_correction: float = 1.0,
) -> np.ndarray:
    """
    Formant shifting using spectral envelope manipulation.
    """
    audio_float = _validate_audio_input(audio_data, "Formant Shift")
    if audio_float.size == 0:
        return audio_float

    if abs(shift_semitones) < 0.01:
        return audio_float

    # Use librosa's pitch shift with formant preservation
    try:
        return librosa.effects.pitch_shift(
            y=audio_float,
            sr=sample_rate,
            n_steps=shift_semitones,
            bins_per_octave=12 * formant_correction,
        )
    except Exception as e:
        print(f"Formant shift error: {e}")
        return audio_float


# Other effects
def apply_pll_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    center_freq: float = 500.0,
    loop_bandwidth_hz: float = 20.0,
    damping_factor: float = 0.707,
    output_filter_freq_offset_hz: float = 0.0,
    output_filter_min_freq_hz: float = 20.0,
    output_filter_max_freq_hz: float = None,
) -> np.ndarray:
    """
    Phase-Locked Loop based adaptive filter.

    A PLL estimates the instantaneous frequency of the input signal and uses
    this to control a time-varying low-pass filter.
    """
    audio_float = _validate_audio_input(audio_data, "PLL Filter")
    if audio_float.size == 0:
        return audio_float

    if output_filter_max_freq_hz is None:
        output_filter_max_freq_hz = sample_rate / 2.2
    output_filter_min_freq_hz = max(1.0, output_filter_min_freq_hz)

    # PLL parameters
    omega_n_norm = loop_bandwidth_hz * TAU / sample_rate
    kp = 2 * damping_factor * omega_n_norm
    ki = omega_n_norm**2

    nco_phase = 0.0
    nco_freq_rad_per_sample_center = center_freq * TAU / sample_rate
    integrator_state = 0.0
    output_filter_state = 0.0

    output_audio = np.zeros_like(audio_float, dtype=np.float32)

    for i in range(len(audio_float)):
        input_sample = audio_float[i]

        # Phase detector
        nco_ref_quad = np.cos(nco_phase)
        pd_error = input_sample * nco_ref_quad

        # Loop filter (PI controller)
        integrator_state += ki * pd_error
        loop_filter_output = kp * pd_error + integrator_state

        # NCO frequency update
        current_nco_freq_rad_per_sample = (
            nco_freq_rad_per_sample_center + loop_filter_output
        )

        # Clamp frequency
        min_nco_rad_samp = (
            (output_filter_min_freq_hz * 0.1) * TAU / sample_rate
        )
        max_nco_rad_samp = (
            (output_filter_max_freq_hz * 2.0) * TAU / sample_rate
        )
        min_nco_rad_samp = max(min_nco_rad_samp, 0.001 * TAU / sample_rate)
        max_nco_rad_samp = min(
            max_nco_rad_samp, (sample_rate / 2.05) * TAU / sample_rate
        )

        current_nco_freq_rad_per_sample = np.clip(
            current_nco_freq_rad_per_sample, min_nco_rad_samp, max_nco_rad_samp
        )

        # Update NCO phase
        nco_phase += current_nco_freq_rad_per_sample
        nco_phase %= TAU

        # Calculate filter cutoff
        nco_instantaneous_freq_hz = (
            current_nco_freq_rad_per_sample * sample_rate / TAU
        )
        output_lpf_cutoff_hz = (
            nco_instantaneous_freq_hz + output_filter_freq_offset_hz
        )
        output_lpf_cutoff_hz = np.clip(
            output_lpf_cutoff_hz,
            output_filter_min_freq_hz,
            output_filter_max_freq_hz,
        )

        # 1st order LPF
        if output_lpf_cutoff_hz <= 1e-3:
            a_coeff = 0.0
        else:
            a_coeff = 1.0 - np.exp(-TAU * output_lpf_cutoff_hz / sample_rate)

        a_coeff = np.clip(a_coeff, 0.0, 1.0)
        output_filter_state = (
            1.0 - a_coeff
        ) * output_filter_state + a_coeff * input_sample
        output_audio[i] = output_filter_state

    return output_audio


def apply_adaptive_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    filter_length: int = 32,
    mu: float = 0.01,
    target_type: str = "noise_reduction",
) -> np.ndarray:
    """
    Adaptive filter using LMS algorithm.

    Args:
        filter_length: Length of adaptive filter
        mu: Learning rate
        target_type: 'noise_reduction', 'echo_cancellation', 'equalization'
    """
    audio_float = _validate_audio_input(audio_data, "Adaptive Filter")
    if audio_float.size == 0:
        return audio_float

    # Initialize filter weights
    w = np.zeros(filter_length, dtype=np.float32)
    output = np.zeros_like(audio_float)

    # Create reference signal based on target type
    if target_type == "noise_reduction":
        # Use delayed version as reference
        delay = filter_length // 2
        reference = np.roll(audio_float, delay)
    elif target_type == "echo_cancellation":
        # Use heavily delayed version
        delay = sample_rate // 10  # 100ms delay
        reference = np.roll(audio_float, delay) * 0.3
    else:  # equalization
        # Use low-passed version as reference
        b, a = signal.butter(2, 1000 / (sample_rate / 2), btype="low")
        reference = signal.filtfilt(b, a, audio_float)

    # LMS algorithm
    for i in range(filter_length, len(audio_float)):
        # Input vector
        x = audio_float[i - filter_length: i][::-1]  # Reverse for convolution

        # Filter output
        y = np.dot(w, x)
        output[i] = y

        # Error signal
        if target_type == "noise_reduction":
            desired = reference[i] if i < len(reference) else 0
        else:
            desired = audio_float[i]

        error = desired - y

        # Update weights
        w = w + mu * error * x

    return output


def apply_granular_synthesis(
    audio_data: np.ndarray,
    sample_rate: int,
    grain_size_ms: float = 50.0,
    grain_density: float = 1.0,
    pitch_variation_semitones: float = 0.0,
    time_stretch_ratio: float = 1.0,
    position_randomness: float = 0.0,
) -> np.ndarray:
    """
    Advanced granular synthesis.

    Args:
        grain_size_ms: Grain duration in milliseconds
        grain_density: Grains per second relative to normal
        pitch_variation_semitones: Random pitch variation range
        time_stretch_ratio: Time stretching factor
        position_randomness: Randomness in grain positioning (0-1)
    """
    audio_float = _validate_audio_input(audio_data, "Granular Synthesis")
    if audio_float.size == 0:
        return audio_float

    grain_size_samples = int(grain_size_ms * sample_rate / 1000.0)
    if grain_size_samples <= 1:
        return audio_float

    output_length = int(len(audio_float) * time_stretch_ratio)
    output = np.zeros(output_length, dtype=np.float32)

    # Grain window (Hann)
    grain_window = np.hanning(grain_size_samples)

    # Calculate number of grains
    grain_hop = int(grain_size_samples / grain_density)
    if grain_hop <= 0:
        grain_hop = 1

    source_pos = 0.0

    for output_pos in range(0, output_length - grain_size_samples, grain_hop):
        # Random position variation
        if position_randomness > 0:
            pos_variation = int(
                position_randomness
                * grain_size_samples
                * (np.random.random() - 0.5)
            )
            actual_source_pos = max(
                0,
                min(
                    len(audio_float) - grain_size_samples,
                    int(source_pos) + pos_variation,
                ),
            )
        else:
            actual_source_pos = max(
                0, min(len(audio_float) - grain_size_samples, int(source_pos))
            )

        # Extract grain
        grain = audio_float[
            actual_source_pos: actual_source_pos + grain_size_samples
        ]

        # Apply pitch variation if requested
        if abs(pitch_variation_semitones) > 0.01:
            pitch_shift = np.random.uniform(
                -pitch_variation_semitones, pitch_variation_semitones
            )
            grain = librosa.effects.pitch_shift(
                grain, sr=sample_rate, n_steps=pitch_shift
                )

        # Apply window and add to output
        windowed_grain = grain * grain_window
        output[output_pos: output_pos + grain_size_samples] += windowed_grain

        # Advance source position
        source_pos += grain_size_samples / time_stretch_ratio

    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 1e-10:
        output /= max_val

    return output


def apply_convolution_reverb(
    audio_data: np.ndarray,
    sample_rate: int,
    impulse_response: np.ndarray = None,
    reverb_type: str = "hall",
    dry_wet_mix: float = 0.3,
) -> np.ndarray:
    """
    Convolution reverb using impulse responses.

    Args:
        impulse_response: Custom impulse response (if None,
                    generates synthetic)
        reverb_type: 'hall', 'room', 'cathedral', 'plate'
        dry_wet_mix: Wet signal ratio
    """
    audio_float = _validate_audio_input(audio_data, "Convolution Reverb")
    if audio_float.size == 0:
        return audio_float

    if impulse_response is None:
        # Generate synthetic impulse response based on type
        if reverb_type == "hall":
            ir_length = int(2.5 * sample_rate)  # 2.5 seconds
            decay_time = 2.0
        elif reverb_type == "room":
            ir_length = int(0.8 * sample_rate)
            decay_time = 0.6
        elif reverb_type == "cathedral":
            ir_length = int(4.0 * sample_rate)
            decay_time = 3.5
        elif reverb_type == "plate":
            ir_length = int(1.2 * sample_rate)
            decay_time = 1.0
        else:
            ir_length = int(1.5 * sample_rate)
            decay_time = 1.2

        # Generate synthetic IR
        impulse_response = np.random.randn(ir_length).astype(np.float32)

        # Apply exponential decay
        t = np.arange(ir_length) / sample_rate
        decay_envelope = np.exp(-6.91 * t / decay_time)  # -60dB decay
        impulse_response *= decay_envelope

        # Add some early reflections
        if reverb_type in ["hall", "cathedral"]:
            early_reflections = np.zeros_like(impulse_response)
            reflection_times = [0.02, 0.035, 0.055, 0.08, 0.12]
            reflection_gains = [0.6, 0.4, 0.3, 0.25, 0.2]

            for time, gain in zip(reflection_times, reflection_gains):
                idx = int(time * sample_rate)
                if idx < len(early_reflections):
                    early_reflections[idx] += gain * np.random.randn()

            impulse_response += early_reflections

        # Frequency-dependent damping
        if reverb_type != "plate":  # Plates have less high-frequency damping
            b, a = signal.butter(2, 8000 / (sample_rate / 2), btype="low")
            impulse_response = signal.filtfilt(b, a, impulse_response)

        # Normalize
        impulse_response /= np.max(np.abs(impulse_response))

    # Apply convolution
    wet_signal = signal.convolve(audio_float, impulse_response, mode="same")

    return _apply_dry_wet_mix(audio_float, wet_signal, dry_wet_mix)


def apply_multiband_compressor(
    audio_data: np.ndarray,
    sample_rate: int,
    num_bands: int = 4,
    crossover_freqs: list = None,
    thresholds_db: list = None,
    ratios: list = None,
    attack_ms: list = None,
    release_ms: list = None,
) -> np.ndarray:
    """
    Multiband compressor with configurable frequency bands.
    """
    audio_float = _validate_audio_input(audio_data, "Multiband Compressor")
    if audio_float.size == 0:
        return audio_float

    # Default parameters
    if crossover_freqs is None:
        crossover_freqs = [250, 1000, 4000][: num_bands - 1]
    if thresholds_db is None:
        thresholds_db = [-20.0] * num_bands
    if ratios is None:
        ratios = [4.0] * num_bands
    if attack_ms is None:
        attack_ms = [10.0] * num_bands
    if release_ms is None:
        release_ms = [100.0] * num_bands

    # Split into bands using Butterworth filters
    bands = []
    for i in range(num_bands):
        if i == 0:
            # Low band
            if len(crossover_freqs) > 0:
                b, a = signal.butter(
                    4, crossover_freqs[0] / (sample_rate / 2), btype="low"
                )
                band = signal.filtfilt(b, a, audio_float)
            else:
                band = audio_float
        elif i == num_bands - 1:
            # High band
            b, a = signal.butter(
                4, crossover_freqs[i - 1] / (sample_rate / 2), btype="high"
            )
            band = signal.filtfilt(b, a, audio_float)
        else:
            # Mid bands
            low_freq = crossover_freqs[i - 1]
            high_freq = crossover_freqs[i]
            b, a = signal.butter(
                4,
                [low_freq / (sample_rate / 2), high_freq / (sample_rate / 2)],
                btype="band",
            )
            band = signal.filtfilt(b, a, audio_float)

        bands.append(band)

    # Compress each band
    compressed_bands = []
    for i, band in enumerate(bands):
        compressed_band = apply_compressor(
            band,
            sample_rate,
            threshold_db=thresholds_db[i],
            ratio=ratios[i],
            attack_ms=attack_ms[i],
            release_ms=release_ms[i],
        )
        compressed_bands.append(compressed_band)

    # Sum all bands
    output = np.sum(compressed_bands, axis=0)

    return output


# Equalizers
def apply_parametric_eq(
    audio_data: np.ndarray,
    sample_rate: int,
    frequency_hz: float = 1000.0,
    gain_db: float = 0.0,
    q_factor: float = 1.0,
    filter_type: str = "peak",
) -> np.ndarray:
    """
    Parametric EQ with configurable filter type.

    Args:
        frequency_hz: Center frequency
        gain_db: Gain in dB (boost/cut)
        q_factor: Quality factor (bandwidth)
        filter_type: 'peak', 'lowshelf', 'highshelf', 'notch'
    """
    audio_float = _validate_audio_input(audio_data, "Parametric EQ")
    if audio_float.size == 0:
        return audio_float

    if abs(gain_db) < 0.1:
        return audio_float  # No significant change

    nyquist = sample_rate / 2
    if frequency_hz >= nyquist:
        return audio_float

    try:
        if filter_type == "peak":
            # Peaking filter
            w0 = TAU * frequency_hz / sample_rate
            A = 10 ** (gain_db / 40.0)  # sqrt of gain
            alpha = np.sin(w0) / (2 * q_factor)

            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A

        elif filter_type == "lowshelf":
            # Low shelf
            w0 = TAU * frequency_hz / sample_rate
            A = 10 ** (gain_db / 40.0)
            S = 1  # Shelf slope
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)

            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        elif filter_type == "highshelf":
            # High shelf
            w0 = TAU * frequency_hz / sample_rate
            A = 10 ** (gain_db / 40.0)
            S = 1
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)

            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        else:  # notch
            b, a = signal.iirnotch(frequency_hz, q_factor, fs=sample_rate)
            return signal.filtfilt(b, a, audio_float)

        # Normalize coefficients
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])

        return signal.filtfilt(b, a, audio_float)

    except Exception as e:
        print(f"EQ filter error: {e}")
        return audio_float


def apply_graphic_eq(
    audio_data: np.ndarray,
    sample_rate: int,
    gains_db: list = None,
    frequencies_hz: list = None,
) -> np.ndarray:
    """
    Graphic EQ with multiple frequency bands.

    Args:
        gains_db: List of gain values in dB for each band
        frequencies_hz: List of center frequencies for each band
    """
    audio_float = _validate_audio_input(audio_data, "Graphic EQ")
    if audio_float.size == 0:
        return audio_float

    # Default 10-band graphic EQ frequencies
    if frequencies_hz is None:
        frequencies_hz = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

    if gains_db is None:
        gains_db = [0.0] * len(frequencies_hz)

    # Make sure lists are same length
    min_len = min(len(gains_db), len(frequencies_hz))
    gains_db = gains_db[:min_len]
    frequencies_hz = frequencies_hz[:min_len]

    output = np.copy(audio_float)

    # Apply each band
    for freq, gain in zip(frequencies_hz, gains_db):
        if abs(gain) > 0.1:  # Only apply if significant gain change
            output = apply_parametric_eq(
                output,
                sample_rate,
                frequency_hz=freq,
                gain_db=gain,
                q_factor=1.414,  # Standard Q for graphic EQ
                filter_type="peak",
            )

    return output


def apply_shelving_eq(
    audio_data: np.ndarray,
    sample_rate: int,
    low_shelf_freq: float = 100.0,
    low_shelf_gain_db: float = 0.0,
    high_shelf_freq: float = 8000.0,
    high_shelf_gain_db: float = 0.0,
) -> np.ndarray:
    """
    Two-band shelving EQ (low and high shelves).
    """
    audio_float = _validate_audio_input(audio_data, "Shelving EQ")
    if audio_float.size == 0:
        return audio_float

    output = audio_float

    # Apply low shelf
    if abs(low_shelf_gain_db) > 0.1:
        output = apply_parametric_eq(
            output,
            sample_rate,
            frequency_hz=low_shelf_freq,
            gain_db=low_shelf_gain_db,
            q_factor=0.707,
            filter_type="lowshelf",
        )

    # Apply high shelf
    if abs(high_shelf_gain_db) > 0.1:
        output = apply_parametric_eq(
            output,
            sample_rate,
            frequency_hz=high_shelf_freq,
            gain_db=high_shelf_gain_db,
            q_factor=0.707,
            filter_type="highshelf",
        )

    return output


# Utility effects
def apply_normalize(
    audio_data: np.ndarray, sample_rate: int, target_db: float = -3.0,
    mode: str = "peak"
) -> np.ndarray:
    """
    Normalize audio to target level.

    Args:
        target_db: Target level in dB
        mode: 'peak', 'rms', or 'lufs'
    """
    print(f"Applying Normalization: {target_db}dB ({mode})")
    audio_float = _validate_audio_input(audio_data, "Normalize")
    if audio_float.size == 0:
        return audio_float

    target_linear = 10 ** (target_db / 20.0)

    if mode == "peak":
        current_peak = np.max(np.abs(audio_float))
        if current_peak > 1e-10:
            gain = target_linear / current_peak
        else:
            gain = 1.0
    elif mode == "rms":
        current_rms = np.sqrt(np.mean(audio_float**2))
        if current_rms > 1e-10:
            gain = target_linear / current_rms
        else:
            gain = 1.0
    else:  # lufs - simplified
        # This is a very simplified LUFS calculation
        # Real LUFS requires K-weighting filter
        current_rms = np.sqrt(np.mean(audio_float**2))
        if current_rms > 1e-10:
            gain = target_linear / current_rms
        else:
            gain = 1.0

    return audio_float * gain


def apply_fade_in(
    audio_data: np.ndarray,
    sample_rate: int,
    fade_duration_ms: float = 1000.0,
    curve: str = "linear",
) -> np.ndarray:
    """
    Apply fade-in to audio.

    Args:
        fade_duration_ms: Fade duration in milliseconds
        curve: 'linear', 'exponential', 'logarithmic', 'sine'
    """
    audio_float = _validate_audio_input(audio_data, "Fade In")
    if audio_float.size == 0:
        return audio_float

    fade_samples = int(fade_duration_ms * sample_rate / 1000.0)
    fade_samples = min(fade_samples, len(audio_float))

    if fade_samples <= 1:
        return audio_float

    # Create fade curve
    t = np.linspace(0, 1, fade_samples)

    if curve == "linear":
        fade_curve = t
    elif curve == "exponential":
        fade_curve = t**2
    elif curve == "logarithmic":
        fade_curve = np.log10(1 + 9 * t)
    elif curve == "sine":
        fade_curve = np.sin(t * np.pi / 2)
    else:
        fade_curve = t

    output = np.copy(audio_float)
    output[:fade_samples] *= fade_curve

    return output


def apply_fade_out(
    audio_data: np.ndarray,
    sample_rate: int,
    fade_duration_ms: float = 1000.0,
    curve: str = "linear",
) -> np.ndarray:
    """Apply fade-out to audio."""
    audio_float = _validate_audio_input(audio_data, "Fade Out")
    if audio_float.size == 0:
        return audio_float

    fade_samples = int(fade_duration_ms * sample_rate / 1000.0)
    fade_samples = min(fade_samples, len(audio_float))

    if fade_samples <= 1:
        return audio_float

    # Create fade curve (reversed)
    t = np.linspace(1, 0, fade_samples)

    if curve == "linear":
        fade_curve = t
    elif curve == "exponential":
        fade_curve = t**2
    elif curve == "logarithmic":
        fade_curve = np.log10(1 + 9 * t)
    elif curve == "sine":
        fade_curve = np.sin(t * np.pi / 2)
    else:
        fade_curve = t

    output = np.copy(audio_float)
    output[-fade_samples:] *= fade_curve

    return output


def apply_crossfade(
    audio_data1: np.ndarray,
    audio_data2: np.ndarray,
    sample_rate: int,
    crossfade_duration_ms: float = 1000.0,
    curve: str = "equal_power",
) -> np.ndarray:
    """
    Crossfade between two audio signals.

    Args:
        audio_data1: First audio signal
        audio_data2: Second audio signal
        crossfade_duration_ms: Crossfade duration in milliseconds
        curve: 'linear', 'equal_power', 'exponential'
    """
    audio1 = _validate_audio_input(audio_data1, "Crossfade")
    audio2 = _validate_audio_input(audio_data2, "Crossfade")

    if audio1.size == 0:
        return audio2
    if audio2.size == 0:
        return audio1

    crossfade_samples = int(crossfade_duration_ms * sample_rate / 1000.0)

    # Make both signals the same length
    max_length = max(len(audio1), len(audio2))
    if len(audio1) < max_length:
        audio1 = np.pad(audio1, (0, max_length - len(audio1)))
    if len(audio2) < max_length:
        audio2 = np.pad(audio2, (0, max_length - len(audio2)))

    crossfade_samples = min(crossfade_samples, max_length)

    if crossfade_samples <= 1:
        return audio2  # No crossfade

    # Create crossfade curves
    t = np.linspace(0, 1, crossfade_samples)

    if curve == "linear":
        fade_out_curve = 1 - t
        fade_in_curve = t
    elif curve == "equal_power":
        fade_out_curve = np.cos(t * np.pi / 2)
        fade_in_curve = np.sin(t * np.pi / 2)
    elif curve == "exponential":
        fade_out_curve = (1 - t) ** 2
        fade_in_curve = t**2
    else:
        fade_out_curve = 1 - t
        fade_in_curve = t

    # Apply crossfade
    output = np.copy(audio2)
    crossfade_start = max_length - crossfade_samples

    # Crossfade region
    output[crossfade_start:] = (
        audio1[crossfade_start:] * fade_out_curve
        + audio2[crossfade_start:] * fade_in_curve
    )

    # Before crossfade region (use audio1)
    output[:crossfade_start] = audio1[:crossfade_start]

    return output
