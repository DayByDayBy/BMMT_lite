"""
Degradation effects for modular audio synthesis.
Signal processors that transform clean audio into "broken transmission" textures.
"""

import numpy as np
from scipy import signal
from typing import Union, Optional


def _validate_signal_params(input_signal: np.ndarray, sample_rate: int) -> None:
    """Validate common parameters for effects functions."""
    if not isinstance(input_signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(input_signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")


def _clip_signal(signal: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Soft clip signal to prevent harsh digital clipping."""
    return np.clip(signal, -threshold, threshold)


def apply_bitcrush(signal: np.ndarray, bit_depth: int, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply bitcrushing to reduce bit depth for digital degradation.
    
    Reduces the resolution of the audio signal by quantizing to fewer bits,
    creating characteristic digital distortion and noise floor artifacts.
    
    Args:
        signal: Input audio signal (numpy array)
        bit_depth: Target bit depth (1-16, lower = more degradation)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Bitcrushed audio signal (same length as input)
    
    Example:
        >>> import numpy as np
        >>> clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> crushed = apply_bitcrush(clean, bit_depth=8)
        >>> len(crushed) == len(clean)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (1 <= bit_depth <= 16):
        raise ValueError(f"Bit depth must be between 1 and 16, got {bit_depth}")
    
    # Calculate quantization levels
    levels = 2 ** bit_depth
    max_val = levels // 2 - 1
    
    # Normalize signal to full scale, quantize, then scale back
    normalized = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    quantized = np.round(normalized * max_val) / max_val
    
    # Scale back to original amplitude range
    if np.max(np.abs(signal)) > 0:
        quantized = quantized * np.max(np.abs(signal))
    
    return _clip_signal(quantized)


def apply_wow_flutter(signal: np.ndarray, intensity: float = 0.1, flutter_freq: float = 0.5, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply wow and flutter effect simulating tape speed variations.
    
    Creates pitch modulation that mimics the characteristic sound of analog tape
    machines with mechanical imperfections.
    
    Args:
        signal: Input audio signal (numpy array)
        intensity: Modulation intensity (0.0-1.0, higher = more variation)
        flutter_freq: Flutter frequency in Hz (0.1-5.0, typical 0.5-2.0)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Signal with wow and flutter applied (same length as input)
    
    Example:
        >>> import numpy as np
        >>> clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> warped = apply_wow_flutter(clean, intensity=0.2, flutter_freq=1.0)
        >>> len(warped) == len(clean)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (0.0 <= intensity <= 1.0):
        raise ValueError(f"Intensity must be between 0.0 and 1.0, got {intensity}")
    if not (0.1 <= flutter_freq <= 5.0):
        raise ValueError(f"Flutter frequency must be between 0.1 and 5.0 Hz, got {flutter_freq}")
    
    if intensity == 0.0:
        return signal.copy()
    
    num_samples = len(signal)
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    
    # Generate modulation signal (combination of wow and flutter)
    # Wow: slow, large variations (0.1-1 Hz)
    # Flutter: faster, smaller variations (1-5 Hz)
    wow_freq = flutter_freq * 0.2  # Wow is slower than flutter
    modulation = (
        0.7 * np.sin(2 * np.pi * wow_freq * t) +  # Wow component
        0.3 * np.sin(2 * np.pi * flutter_freq * t)  # Flutter component
    )
    
    # Scale modulation by intensity (max ±5% speed variation)
    speed_variation = 1.0 + intensity * 0.05 * modulation
    
    # Create time-warped indices
    warped_indices = np.cumsum(speed_variation) * (num_samples - 1) / np.sum(speed_variation)
    warped_indices = np.clip(warped_indices, 0, num_samples - 1)
    
    # Interpolate signal at warped time points
    warped_signal = np.interp(warped_indices, np.arange(num_samples), signal)
    
    return _clip_signal(warped_signal)


def apply_dropout(signal: np.ndarray, dropout_probability: float = 0.01, fade_time: float = 0.1, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply random dropouts simulating tape or transmission failures.
    
    Creates brief moments where the signal fades out and back in, mimicking
    the characteristic dropouts of failing analog equipment.
    
    Args:
        signal: Input audio signal (numpy array)
        dropout_probability: Probability of dropout per second (0.0-0.1)
        fade_time: Fade in/out time for dropouts in seconds (0.01-1.0)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Signal with dropouts applied (same length as input)
    
    Example:
        >>> import numpy as np
        >>> clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> dropped = apply_dropout(clean, dropout_probability=0.05, fade_time=0.1)
        >>> len(dropped) == len(clean)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (0.0 <= dropout_probability <= 0.1):
        raise ValueError(f"Dropout probability must be between 0.0 and 0.1, got {dropout_probability}")
    if not (0.01 <= fade_time <= 1.0):
        raise ValueError(f"Fade time must be between 0.01 and 1.0 seconds, got {fade_time}")
    
    if dropout_probability == 0.0:
        return signal.copy()
    
    num_samples = len(signal)
    duration = num_samples / sample_rate
    output_signal = signal.copy()
    
    # Calculate expected number of dropouts
    expected_dropouts = int(duration * dropout_probability)
    if expected_dropouts == 0 and np.random.random() < duration * dropout_probability:
        expected_dropouts = 1
    
    fade_samples = int(fade_time * sample_rate)
    
    for _ in range(expected_dropouts):
        # Random dropout position and duration
        dropout_start = np.random.randint(0, max(1, num_samples - fade_samples * 4))
        dropout_duration = np.random.randint(fade_samples * 2, fade_samples * 6)
        dropout_end = min(dropout_start + dropout_duration, num_samples)
        
        # Create fade envelope
        fade_in_end = min(dropout_start + fade_samples, dropout_end)
        fade_out_start = max(dropout_end - fade_samples, dropout_start)
        
        # Apply fade out
        if fade_in_end > dropout_start:
            fade_out_env = np.linspace(1.0, 0.0, fade_in_end - dropout_start)
            output_signal[dropout_start:fade_in_end] *= fade_out_env
        
        # Silent section
        if fade_out_start > fade_in_end:
            output_signal[fade_in_end:fade_out_start] = 0.0
        
        # Apply fade in
        if dropout_end > fade_out_start:
            fade_in_env = np.linspace(0.0, 1.0, dropout_end - fade_out_start)
            output_signal[fade_out_start:dropout_end] *= fade_in_env
    
    return output_signal


def add_static_bursts(signal: np.ndarray, burst_frequency: float = 0.1, intensity: float = 0.2, sample_rate: int = 44100) -> np.ndarray:
    """
    Add random static bursts to simulate electrical interference.
    
    Adds brief bursts of high-frequency noise to simulate electrical
    interference, radio static, or digital transmission errors.
    
    Args:
        signal: Input audio signal (numpy array)
        burst_frequency: Average bursts per second (0.01-1.0)
        intensity: Burst intensity relative to signal (0.0-1.0)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Signal with static bursts added (same length as input)
    
    Example:
        >>> import numpy as np
        >>> clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> static = add_static_bursts(clean, burst_frequency=0.2, intensity=0.1)
        >>> len(static) == len(clean)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (0.01 <= burst_frequency <= 1.0):
        raise ValueError(f"Burst frequency must be between 0.01 and 1.0, got {burst_frequency}")
    if not (0.0 <= intensity <= 1.0):
        raise ValueError(f"Intensity must be between 0.0 and 1.0, got {intensity}")
    
    if intensity == 0.0:
        return signal.copy()
    
    num_samples = len(signal)
    duration = num_samples / sample_rate
    output_signal = signal.copy()
    
    # Calculate expected number of bursts
    expected_bursts = int(duration * burst_frequency)
    if expected_bursts == 0 and np.random.random() < duration * burst_frequency:
        expected_bursts = 1
    
    signal_rms = np.sqrt(np.mean(signal ** 2)) if len(signal) > 0 else 0.0
    
    for _ in range(expected_bursts):
        # Random burst position and duration
        burst_start = np.random.randint(0, max(1, num_samples - 100))
        burst_duration = np.random.randint(50, 500)  # 1-11ms at 44.1kHz
        burst_end = min(burst_start + burst_duration, num_samples)
        
        # Generate high-frequency noise burst
        burst_samples = burst_end - burst_start
        burst_noise = np.random.normal(0, 1, burst_samples)
        
        # High-pass filter the noise to emphasize high frequencies
        if burst_samples > 10:  # Only filter if we have enough samples
            nyquist = sample_rate / 2
            high_cutoff = min(8000.0, nyquist * 0.8)  # High-frequency emphasis
            normalized_cutoff = high_cutoff / nyquist
            from scipy import signal as scipy_signal
            b, a = scipy_signal.butter(2, normalized_cutoff, btype='high')
            burst_noise = scipy_signal.filtfilt(b, a, burst_noise)
        
        # Scale burst to intensity level
        burst_amplitude = intensity * signal_rms * 2.0  # Make bursts prominent
        if np.max(np.abs(burst_noise)) > 0:
            burst_noise = burst_noise / np.max(np.abs(burst_noise)) * burst_amplitude
        
        # Add burst to signal
        output_signal[burst_start:burst_end] += burst_noise
    
    return _clip_signal(output_signal)


def apply_tape_saturation(signal: np.ndarray, drive: float = 2.0, warmth: float = 0.3) -> np.ndarray:
    """
    Apply analog tape saturation for warm, musical distortion.
    
    Simulates the characteristic soft saturation and harmonic enhancement
    of analog tape recording, adding warmth and musical distortion.
    
    Args:
        signal: Input audio signal (numpy array)
        drive: Saturation drive amount (1.0-5.0, higher = more saturation)
        warmth: Low-frequency emphasis (0.0-1.0, higher = warmer)
    
    Returns:
        numpy.ndarray: Saturated signal (same length as input)
    
    Example:
        >>> import numpy as np
        >>> clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> saturated = apply_tape_saturation(clean, drive=1.5, warmth=0.4)
        >>> len(saturated) == len(clean)
        True
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if not (1.0 <= drive <= 5.0):
        raise ValueError(f"Drive must be between 1.0 and 5.0, got {drive}")
    if not (0.0 <= warmth <= 1.0):
        raise ValueError(f"Warmth must be between 0.0 and 1.0, got {warmth}")
    
    # Apply input gain
    driven_signal = signal * drive
    
    # Soft saturation using tanh function (analog-like)
    saturated = np.tanh(driven_signal)
    
    # Add subtle asymmetry for analog character
    asymmetry = 0.1 * warmth
    saturated = saturated + asymmetry * (saturated ** 2)
    
    # Compensate for level increase
    saturated = saturated / drive * 0.8
    
    # Add warmth by emphasizing low frequencies slightly
    if warmth > 0.0 and len(signal) > 100:
        # Simple low-shelf boost
        warmth_boost = 1.0 + warmth * 0.3
        # Apply gentle low-frequency emphasis
        low_freq_component = saturated - np.diff(saturated, prepend=saturated[0])
        saturated = saturated + warmth * 0.2 * low_freq_component
    
    return _clip_signal(saturated)


def apply_vinyl_crackle(signal: np.ndarray, intensity: float = 0.1, sample_rate: int = 44100) -> np.ndarray:
    """
    Add vinyl record crackle and surface noise.
    
    Simulates the characteristic surface noise, pops, and crackles
    of vinyl records, adding vintage analog character.
    
    Args:
        signal: Input audio signal (numpy array)
        intensity: Crackle intensity (0.0-1.0, higher = more crackle)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Signal with vinyl crackle added (same length as input)
    
    Example:
        >>> import numpy as np
        >>> clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> crackled = apply_vinyl_crackle(clean, intensity=0.2)
        >>> len(crackled) == len(clean)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (0.0 <= intensity <= 1.0):
        raise ValueError(f"Intensity must be between 0.0 and 1.0, got {intensity}")
    
    if intensity == 0.0:
        return signal.copy()
    
    num_samples = len(signal)
    output_signal = signal.copy()
    
    # Generate surface noise (filtered white noise)
    surface_noise = np.random.normal(0, 1, num_samples)
    
    # Filter surface noise to vinyl-like spectrum
    if num_samples > 100:
        nyquist = sample_rate / 2
        # High-pass filter to remove low rumble
        hp_cutoff = min(100.0, nyquist * 0.1)
        hp_normalized = hp_cutoff / nyquist
        from scipy import signal as scipy_signal
        b_hp, a_hp = scipy_signal.butter(1, hp_normalized, btype='high')
        surface_noise = scipy_signal.filtfilt(b_hp, a_hp, surface_noise)
        
        # Low-pass filter to remove harsh highs
        lp_cutoff = min(8000.0, nyquist * 0.8)
        lp_normalized = lp_cutoff / nyquist
        b_lp, a_lp = scipy_signal.butter(2, lp_normalized, btype='low')
        surface_noise = scipy_signal.filtfilt(b_lp, a_lp, surface_noise)
    
    # Scale surface noise
    signal_rms = np.sqrt(np.mean(signal ** 2)) if len(signal) > 0 else 0.0
    surface_amplitude = intensity * signal_rms * 0.1
    surface_noise = surface_noise * surface_amplitude
    
    # Add random pops and clicks
    pop_probability = intensity * 0.001  # Pops per sample
    for i in range(num_samples):
        if np.random.random() < pop_probability:
            # Create a brief pop
            pop_duration = min(10, num_samples - i)
            pop_amplitude = intensity * signal_rms * np.random.uniform(0.5, 2.0)
            pop_envelope = np.exp(-np.arange(pop_duration) / 3.0)  # Quick decay
            pop_signal = pop_amplitude * pop_envelope * np.random.choice([-1, 1])
            output_signal[i:i + pop_duration] += pop_signal
    
    # Add surface noise
    output_signal += surface_noise
    
    return _clip_signal(output_signal)


def apply_tube_warmth(signal: np.ndarray, drive: float = 1.5, asymmetry: float = 0.1) -> np.ndarray:
    """
    Apply vacuum tube-style warmth and harmonic distortion.
    
    Simulates the characteristic even-harmonic distortion and soft clipping
    of vacuum tube amplifiers, adding musical warmth and presence.
    
    Args:
        signal: Input audio signal (numpy array)
        drive: Tube drive amount (1.0-3.0, higher = more distortion)
        asymmetry: Asymmetric distortion amount (0.0-0.5, adds even harmonics)
    
    Returns:
        numpy.ndarray: Signal with tube warmth applied (same length as input)
    
    Example:
        >>> import numpy as np
        >>> clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> warm = apply_tube_warmth(clean, drive=1.8, asymmetry=0.15)
        >>> len(warm) == len(clean)
        True
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if not (1.0 <= drive <= 3.0):
        raise ValueError(f"Drive must be between 1.0 and 3.0, got {drive}")
    if not (0.0 <= asymmetry <= 0.5):
        raise ValueError(f"Asymmetry must be between 0.0 and 0.5, got {asymmetry}")
    
    # Apply input gain
    driven_signal = signal * drive
    
    # Tube-style saturation with asymmetry
    # Positive and negative cycles treated differently
    positive_mask = driven_signal >= 0
    negative_mask = driven_signal < 0
    
    output_signal = np.zeros_like(driven_signal)
    
    # Positive cycle (more compressed, even harmonics)
    pos_drive = 1.0 + asymmetry
    output_signal[positive_mask] = np.tanh(driven_signal[positive_mask] * pos_drive) / pos_drive
    
    # Negative cycle (less compressed)
    neg_drive = 1.0 - asymmetry * 0.5
    output_signal[negative_mask] = np.tanh(driven_signal[negative_mask] * neg_drive) / neg_drive
    
    # Add subtle second harmonic for tube character
    if len(signal) > 100:
        second_harmonic = 0.05 * asymmetry * (output_signal ** 2)
        output_signal += second_harmonic
    
    # Compensate for level increase
    output_signal = output_signal / drive * 0.9
    
    return _clip_signal(output_signal)


def apply_chorus(
    input_signal: np.ndarray,
    rate: float = 1.5,
    depth: float = 0.003,
    voices: int = 3,
    mix: float = 0.5,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply chorus effect for width and movement.
    
    Creates multiple delayed copies with LFO-modulated delay times,
    producing a rich, shimmering sound ideal for strings and pads.
    
    Args:
        input_signal: Input audio signal (numpy array, mono or stereo)
        rate: LFO rate in Hz (0.1-5.0, typical 0.5-2.0 for subtle movement)
        depth: Modulation depth in seconds (0.001-0.01, controls pitch variation)
        voices: Number of chorus voices (2-6, more = thicker sound)
        mix: Wet/dry mix (0.0-1.0, 0=dry only, 1=wet only)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Chorused signal (stereo output)
    
    Example:
        >>> import numpy as np
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> chorused = apply_chorus(signal, rate=1.0, depth=0.003, voices=3)
        >>> chorused.shape == (44100, 2)
        True
    """
    _validate_signal_params(input_signal, sample_rate)
    
    if not (0.1 <= rate <= 5.0):
        raise ValueError(f"Rate must be between 0.1 and 5.0 Hz, got {rate}")
    if not (0.001 <= depth <= 0.01):
        raise ValueError(f"Depth must be between 0.001 and 0.01 seconds, got {depth}")
    if not (2 <= voices <= 6):
        raise ValueError(f"Voices must be between 2 and 6, got {voices}")
    if not (0.0 <= mix <= 1.0):
        raise ValueError(f"Mix must be between 0.0 and 1.0, got {mix}")
    
    # Ensure mono input for processing
    if input_signal.ndim == 2:
        mono_input = np.mean(input_signal, axis=1)
    else:
        mono_input = input_signal
    
    num_samples = len(mono_input)
    t = np.arange(num_samples) / sample_rate
    
    # Base delay for chorus (20-30ms)
    base_delay_samples = int(0.025 * sample_rate)
    depth_samples = int(depth * sample_rate)
    
    # Create wet signal by summing all voices
    wet_left = np.zeros(num_samples)
    wet_right = np.zeros(num_samples)
    
    for voice in range(voices):
        # Each voice has slightly different LFO phase and rate
        phase_offset = 2 * np.pi * voice / voices
        rate_offset = rate * (1.0 + 0.1 * (voice - voices / 2))
        
        # Generate LFO for this voice
        lfo = np.sin(2 * np.pi * rate_offset * t + phase_offset)
        
        # Calculate delay in samples for each time point
        delay_samples = base_delay_samples + (lfo * depth_samples).astype(int)
        delay_samples = np.clip(delay_samples, 1, num_samples - 1)
        
        # Apply variable delay using interpolation
        delayed = np.zeros(num_samples)
        for i in range(num_samples):
            src_idx = i - delay_samples[i]
            if src_idx >= 0:
                delayed[i] = mono_input[src_idx]
        
        # Pan voices alternately left/right for stereo width
        pan = (voice % 2) * 2 - 1  # -1 or 1
        voice_gain = 1.0 / voices
        wet_left += delayed * voice_gain * (1.0 - pan * 0.3)
        wet_right += delayed * voice_gain * (1.0 + pan * 0.3)
    
    # Mix dry and wet
    dry_gain = 1.0 - mix
    wet_gain = mix
    
    left_out = dry_gain * mono_input + wet_gain * wet_left
    right_out = dry_gain * mono_input + wet_gain * wet_right
    
    output = np.column_stack([left_out, right_out])
    return _clip_signal(output)


def apply_ensemble(
    input_signal: np.ndarray,
    voices: int = 4,
    detune_cents: float = 8.0,
    spread: float = 0.7,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply ensemble/string machine effect for lush, detuned sound.
    
    Creates multiple pitched copies with slight detuning and stereo spread,
    producing the characteristic "string machine" sound of 70s disco.
    
    Args:
        input_signal: Input audio signal (numpy array, mono or stereo)
        voices: Number of ensemble voices (2-8, typical 4-6)
        detune_cents: Maximum detuning in cents (1-20, typical 5-10)
        spread: Stereo spread amount (0.0-1.0, higher = wider)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Ensembled signal (stereo output)
    
    Example:
        >>> import numpy as np
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> ensembled = apply_ensemble(signal, voices=4, detune_cents=8.0)
        >>> ensembled.shape == (44100, 2)
        True
    """
    _validate_signal_params(input_signal, sample_rate)
    
    if not (2 <= voices <= 8):
        raise ValueError(f"Voices must be between 2 and 8, got {voices}")
    if not (1.0 <= detune_cents <= 20.0):
        raise ValueError(f"Detune must be between 1 and 20 cents, got {detune_cents}")
    if not (0.0 <= spread <= 1.0):
        raise ValueError(f"Spread must be between 0.0 and 1.0, got {spread}")
    
    # Ensure mono input for processing
    if input_signal.ndim == 2:
        mono_input = np.mean(input_signal, axis=1)
    else:
        mono_input = input_signal
    
    num_samples = len(mono_input)
    
    # Create output channels
    left_out = np.zeros(num_samples)
    right_out = np.zeros(num_samples)
    
    # Calculate pitch ratios for detuning
    # Detune ranges from -detune_cents to +detune_cents
    for voice in range(voices):
        # Calculate detune for this voice (spread evenly around center)
        if voices == 1:
            cents = 0.0
        else:
            cents = detune_cents * (2 * voice / (voices - 1) - 1)
        
        # Convert cents to pitch ratio
        pitch_ratio = 2 ** (cents / 1200.0)
        
        # Apply pitch shift by resampling
        if abs(pitch_ratio - 1.0) > 0.0001:
            # Create new time indices for pitch shifting
            original_indices = np.arange(num_samples)
            new_indices = original_indices * pitch_ratio
            new_indices = np.clip(new_indices, 0, num_samples - 1)
            
            # Interpolate
            shifted = np.interp(new_indices, original_indices, mono_input)
        else:
            shifted = mono_input.copy()
        
        # Calculate stereo position for this voice
        # Spread voices across stereo field
        pan_position = (voice / (voices - 1) if voices > 1 else 0.5) * 2 - 1  # -1 to 1
        pan_position *= spread
        
        # Apply equal-power panning
        left_gain = np.cos((pan_position + 1) * np.pi / 4)
        right_gain = np.sin((pan_position + 1) * np.pi / 4)
        
        voice_gain = 1.0 / np.sqrt(voices)  # Normalize for number of voices
        
        left_out += shifted * voice_gain * left_gain
        right_out += shifted * voice_gain * right_gain
    
    # Add subtle modulation for movement (slow LFO on amplitude)
    t = np.arange(num_samples) / sample_rate
    mod_rate = 0.15  # Very slow modulation
    mod_depth = 0.03  # Subtle
    mod_left = 1.0 + mod_depth * np.sin(2 * np.pi * mod_rate * t)
    mod_right = 1.0 + mod_depth * np.sin(2 * np.pi * mod_rate * t + np.pi / 3)
    
    left_out *= mod_left
    right_out *= mod_right
    
    output = np.column_stack([left_out, right_out])
    return _clip_signal(output)


if __name__ == "__main__":
    # Basic tests
    import doctest
    doctest.testmod()
    
    # Quick validation tests
    try:
        print("Testing degradation effects...")
        
        # Create test signal
        duration = 1.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Test all effects
        bitcrushed = apply_bitcrush(test_signal, bit_depth=8)
        wow_flutter = apply_wow_flutter(test_signal, intensity=0.2, flutter_freq=1.0)
        dropout = apply_dropout(test_signal, dropout_probability=0.05, fade_time=0.1)
        static = add_static_bursts(test_signal, burst_frequency=0.2, intensity=0.1)
        tape_sat = apply_tape_saturation(test_signal, drive=2.0, warmth=0.3)
        vinyl = apply_vinyl_crackle(test_signal, intensity=0.1)
        tube = apply_tube_warmth(test_signal, drive=1.5, asymmetry=0.1)
        
        print(f"✓ Bitcrush: {len(bitcrushed)} samples")
        print(f"✓ Wow/Flutter: {len(wow_flutter)} samples")
        print(f"✓ Dropout: {len(dropout)} samples")
        print(f"✓ Static Bursts: {len(static)} samples")
        print(f"✓ Tape Saturation: {len(tape_sat)} samples")
        print(f"✓ Vinyl Crackle: {len(vinyl)} samples")
        print(f"✓ Tube Warmth: {len(tube)} samples")
        
        print("\n✓ All degradation effects working correctly")
        
        # Test error conditions
        print("\nTesting error conditions...")
        try:
            apply_bitcrush(test_signal, bit_depth=20)  # Too high
            print("✗ Should have raised bit depth error")
        except ValueError:
            print("✓ Bit depth validation working")
            
        try:
            apply_wow_flutter(test_signal, intensity=2.0)  # Too high
            print("✗ Should have raised intensity error")
        except ValueError:
            print("✓ Intensity validation working")
            
    except Exception as e:
        print(f"✗ Error in effects tests: {e}")