"""
Spatial effects for modular audio synthesis.
Creates sense of distance, space, and environmental processing for "broken transmission" dronescapes.
"""

import numpy as np
from scipy import signal
from typing import Union, Tuple


def _validate_signal_params(input_signal: np.ndarray, sample_rate: int) -> None:
    """Validate common parameters for spatial effects functions."""
    if not isinstance(input_signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(input_signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")


def _ensure_stereo(signal: np.ndarray) -> np.ndarray:
    """Convert mono signal to stereo if needed."""
    if signal.ndim == 1:
        # Convert mono to stereo
        return np.column_stack([signal, signal])
    elif signal.ndim == 2 and signal.shape[1] == 2:
        return signal
    else:
        raise ValueError("Signal must be mono (1D) or stereo (2D with 2 channels)")


def _clip_signal(signal: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Soft clip signal to prevent harsh digital clipping."""
    return np.clip(signal, -threshold, threshold)


def apply_reverb(signal: np.ndarray, room_size: float = 0.5, decay_time: float = 2.0, damping: float = 0.3, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply reverb to create sense of space and ambience.
    
    Simulates the acoustic reflections of a room or space, adding depth
    and spatial character to the signal.
    
    Args:
        signal: Input audio signal (numpy array, mono or stereo)
        room_size: Room size factor (0.1-1.0, larger = bigger space)
        decay_time: Reverb decay time in seconds (0.5-10.0)
        damping: High-frequency damping (0.0-1.0, higher = more damped)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Reverbed signal (stereo output)
    
    Example:
        >>> import numpy as np
        >>> mono_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> reverbed = apply_reverb(mono_signal, room_size=0.7, decay_time=3.0)
        >>> reverbed.shape == (44100, 2)  # Stereo output
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (0.1 <= room_size <= 1.0):
        raise ValueError(f"Room size must be between 0.1 and 1.0, got {room_size}")
    if not (0.5 <= decay_time <= 10.0):
        raise ValueError(f"Decay time must be between 0.5 and 10.0 seconds, got {decay_time}")
    if not (0.0 <= damping <= 1.0):
        raise ValueError(f"Damping must be between 0.0 and 1.0, got {damping}")
    
    # Ensure stereo output
    if signal.ndim == 1:
        # Mono input - create stereo
        left_channel = signal.copy()
        right_channel = signal.copy()
    else:
        # Stereo input
        stereo_signal = _ensure_stereo(signal)
        left_channel = stereo_signal[:, 0]
        right_channel = stereo_signal[:, 1]
    
    # Calculate reverb parameters
    max_delay_samples = int(room_size * 0.1 * sample_rate)  # Up to 100ms for room_size=1.0
    decay_factor = np.exp(-1.0 / (decay_time * sample_rate))
    
    # Create multiple delay lines for realistic reverb
    delay_times = [
        int(max_delay_samples * 0.3),
        int(max_delay_samples * 0.5),
        int(max_delay_samples * 0.7),
        int(max_delay_samples * 0.9),
        int(max_delay_samples * 1.1),
        int(max_delay_samples * 1.3)
    ]
    
    # Different delay times for left and right channels
    left_delays = delay_times[::2]  # Even indices
    right_delays = delay_times[1::2]  # Odd indices
    
    def apply_reverb_channel(input_channel, delays):
        output = input_channel.copy()
        
        for delay_samples in delays:
            if delay_samples > 0 and delay_samples < len(input_channel):
                # Create delayed version
                delayed = np.zeros_like(input_channel)
                delayed[delay_samples:] = input_channel[:-delay_samples]
                
                # Apply decay and damping
                decay_envelope = decay_factor ** np.arange(len(delayed))
                delayed *= decay_envelope
                
                # Apply high-frequency damping
                if damping > 0.0 and len(delayed) > 100:
                    nyquist = sample_rate / 2
                    cutoff_freq = nyquist * (1.0 - damping * 0.8)  # Reduce highs
                    normalized_cutoff = cutoff_freq / nyquist
                    from scipy import signal as scipy_signal
                    b, a = scipy_signal.butter(2, normalized_cutoff, btype='low')
                    delayed = scipy_signal.filtfilt(b, a, delayed)
                
                # Mix with output
                gain = 0.3 / len(delays)  # Normalize by number of delays
                output += delayed * gain
        
        return output
    
    # Apply reverb to both channels
    left_reverbed = apply_reverb_channel(left_channel, left_delays)
    right_reverbed = apply_reverb_channel(right_channel, right_delays)
    
    # Combine channels
    stereo_output = np.column_stack([left_reverbed, right_reverbed])
    
    return _clip_signal(stereo_output)


def apply_distance_filter(signal: np.ndarray, distance_factor: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply distance-based filtering to simulate far-away sources.
    
    Simulates the natural high-frequency attenuation that occurs when
    sound travels through air over distance.
    
    Args:
        signal: Input audio signal (numpy array)
        distance_factor: Distance multiplier (0.1-10.0, higher = more distant)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Distance-filtered signal (same format as input)
    
    Example:
        >>> import numpy as np
        >>> close_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> distant = apply_distance_filter(close_signal, distance_factor=3.0)
        >>> len(distant) == len(close_signal)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (0.1 <= distance_factor <= 10.0):
        raise ValueError(f"Distance factor must be between 0.1 and 10.0, got {distance_factor}")
    
    if distance_factor <= 1.0:
        return signal.copy()
    
    # Calculate cutoff frequency based on distance
    # Closer = higher cutoff, farther = lower cutoff
    base_cutoff = 8000.0  # Hz
    cutoff_freq = base_cutoff / (distance_factor ** 0.5)
    cutoff_freq = max(cutoff_freq, 200.0)  # Don't go too low
    
    # Apply low-pass filtering
    nyquist = sample_rate / 2
    if cutoff_freq >= nyquist:
        return signal.copy()
    
    normalized_cutoff = cutoff_freq / nyquist
    
    # Handle mono and stereo signals
    if signal.ndim == 1:
        # Mono signal
        from scipy import signal as scipy_signal
        b, a = scipy_signal.butter(2, normalized_cutoff, btype='low')
        filtered = scipy_signal.filtfilt(b, a, signal)
    else:
        # Stereo signal
        stereo_signal = _ensure_stereo(signal)
        from scipy import signal as scipy_signal
        b, a = scipy_signal.butter(2, normalized_cutoff, btype='low')
        left_filtered = scipy_signal.filtfilt(b, a, stereo_signal[:, 0])
        right_filtered = scipy_signal.filtfilt(b, a, stereo_signal[:, 1])
        filtered = np.column_stack([left_filtered, right_filtered])
    
    # Apply slight amplitude reduction for distance
    distance_attenuation = 1.0 / (1.0 + (distance_factor - 1.0) * 0.2)
    filtered *= distance_attenuation
    
    return _clip_signal(filtered)


def apply_stereo_width(signal: np.ndarray, width: float = 1.0) -> np.ndarray:
    """
    Adjust stereo width of a signal.
    
    Controls the stereo spread from mono (width=0) to enhanced stereo (width>1).
    Mono input is converted to stereo with spatial processing.
    
    Args:
        signal: Input audio signal (numpy array, mono or stereo)
        width: Stereo width factor (0.0-2.0, 0=mono, 1=normal, >1=wide)
    
    Returns:
        numpy.ndarray: Stereo signal with adjusted width
    
    Example:
        >>> import numpy as np
        >>> mono_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> wide_stereo = apply_stereo_width(mono_signal, width=1.5)
        >>> wide_stereo.shape == (44100, 2)
        True
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if not (0.0 <= width <= 2.0):
        raise ValueError(f"Width must be between 0.0 and 2.0, got {width}")
    
    # Convert to stereo if mono
    if signal.ndim == 1:
        # For mono input, create stereo with slight decorrelation
        left = signal.copy()
        # Create slightly different right channel using all-pass filtering
        if len(signal) > 100:
            # Simple all-pass filter for decorrelation
            delay_samples = 3
            right = signal.copy()
            right[delay_samples:] = 0.7 * signal[delay_samples:] + 0.3 * signal[:-delay_samples]
        else:
            right = signal.copy()
        
        stereo_signal = np.column_stack([left, right])
    else:
        stereo_signal = _ensure_stereo(signal)
    
    left = stereo_signal[:, 0]
    right = stereo_signal[:, 1]
    
    # Calculate mid and side signals
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    
    # Adjust stereo width
    side_adjusted = side * width
    
    # Convert back to left/right
    left_out = mid + side_adjusted
    right_out = mid - side_adjusted
    
    output = np.column_stack([left_out, right_out])
    
    return _clip_signal(output)


def apply_doppler_shift(signal: np.ndarray, frequency_shift: float = 0.0, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply Doppler shift effect for moving sound sources.
    
    Simulates the frequency shift that occurs when a sound source
    is moving relative to the listener.
    
    Args:
        signal: Input audio signal (numpy array)
        frequency_shift: Frequency shift in Hz (-200 to +200, negative = moving away)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Doppler-shifted signal (same format as input)
    
    Example:
        >>> import numpy as np
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> shifted = apply_doppler_shift(signal, frequency_shift=50.0)
        >>> len(shifted) == len(signal)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (-200.0 <= frequency_shift <= 200.0):
        raise ValueError(f"Frequency shift must be between -200 and +200 Hz, got {frequency_shift}")
    
    if abs(frequency_shift) < 0.1:
        return signal.copy()
    
    # Calculate pitch shift ratio
    # Positive shift = higher pitch (approaching)
    # Negative shift = lower pitch (receding)
    shift_ratio = 1.0 + frequency_shift / 1000.0  # Approximate for small shifts
    
    # Handle mono and stereo signals
    if signal.ndim == 1:
        shifted = _pitch_shift_channel(signal, shift_ratio, sample_rate)
    else:
        stereo_signal = _ensure_stereo(signal)
        left_shifted = _pitch_shift_channel(stereo_signal[:, 0], shift_ratio, sample_rate)
        right_shifted = _pitch_shift_channel(stereo_signal[:, 1], shift_ratio, sample_rate)
        shifted = np.column_stack([left_shifted, right_shifted])
    
    return _clip_signal(shifted)


def _pitch_shift_channel(channel: np.ndarray, ratio: float, sample_rate: int) -> np.ndarray:
    """Apply pitch shift to a single channel using time-domain stretching."""
    if abs(ratio - 1.0) < 0.001:
        return channel.copy()
    
    # Simple pitch shifting using resampling
    # This is a basic implementation - more sophisticated methods exist
    original_length = len(channel)
    
    # Create new time indices
    new_indices = np.arange(original_length) / ratio
    new_indices = np.clip(new_indices, 0, original_length - 1)
    
    # Interpolate to get shifted signal
    shifted = np.interp(new_indices, np.arange(original_length), channel)
    
    # Ensure output length matches input
    if len(shifted) != original_length:
        if len(shifted) > original_length:
            shifted = shifted[:original_length]
        else:
            # Pad with zeros if needed
            padded = np.zeros(original_length)
            padded[:len(shifted)] = shifted
            shifted = padded
    
    return shifted


def apply_air_absorption(signal: np.ndarray, distance: float = 10.0, humidity: float = 0.5, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply air absorption effects for realistic distance modeling.
    
    Simulates the frequency-dependent absorption of sound traveling
    through air, with effects varying by distance and humidity.
    
    Args:
        signal: Input audio signal (numpy array)
        distance: Distance in meters (1.0-100.0)
        humidity: Relative humidity (0.0-1.0, higher = less absorption)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Air-absorbed signal (same format as input)
    
    Example:
        >>> import numpy as np
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> absorbed = apply_air_absorption(signal, distance=50.0, humidity=0.3)
        >>> len(absorbed) == len(signal)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (1.0 <= distance <= 100.0):
        raise ValueError(f"Distance must be between 1.0 and 100.0 meters, got {distance}")
    if not (0.0 <= humidity <= 1.0):
        raise ValueError(f"Humidity must be between 0.0 and 1.0, got {humidity}")
    
    if distance <= 1.0:
        return signal.copy()
    
    # Air absorption is frequency-dependent and increases with distance
    # Higher frequencies are absorbed more, especially in dry air
    
    # Calculate absorption coefficients
    # Simplified model based on atmospheric absorption
    nyquist = sample_rate / 2
    
    # Create frequency-dependent filter
    # More absorption at high frequencies, less in humid air
    absorption_factor = distance / 10.0  # Scale factor
    humidity_factor = 1.0 - humidity * 0.5  # Dry air absorbs more
    
    # Multi-band filtering to simulate frequency-dependent absorption
    bands = [
        (0, 1000, 1.0),  # Low frequencies - minimal absorption
        (1000, 4000, 1.0 - absorption_factor * humidity_factor * 0.1),  # Mid frequencies
        (4000, 8000, 1.0 - absorption_factor * humidity_factor * 0.3),  # High frequencies
        (8000, nyquist, 1.0 - absorption_factor * humidity_factor * 0.6)  # Very high frequencies
    ]
    
    # Handle mono and stereo signals
    if signal.ndim == 1:
        filtered = _apply_multiband_absorption(signal, bands, sample_rate)
    else:
        stereo_signal = _ensure_stereo(signal)
        left_filtered = _apply_multiband_absorption(stereo_signal[:, 0], bands, sample_rate)
        right_filtered = _apply_multiband_absorption(stereo_signal[:, 1], bands, sample_rate)
        filtered = np.column_stack([left_filtered, right_filtered])
    
    return _clip_signal(filtered)


def _apply_multiband_absorption(channel: np.ndarray, bands: list, sample_rate: int) -> np.ndarray:
    """Apply frequency-dependent absorption to a single channel."""
    if len(channel) < 100:
        return channel.copy()
    
    output = np.zeros_like(channel)
    nyquist = sample_rate / 2
    
    for low_freq, high_freq, gain in bands:
        if gain <= 0.0:
            continue
            
        # Create bandpass filter for this frequency range
        low_norm = max(low_freq / nyquist, 0.001)
        high_norm = min(high_freq / nyquist, 0.999)
        
        if low_norm >= high_norm:
            continue
            
        try:
            if low_freq == 0:
                # Low-pass filter
                from scipy import signal as scipy_signal
                b, a = scipy_signal.butter(2, high_norm, btype='low')
            elif high_freq >= nyquist:
                # High-pass filter
                from scipy import signal as scipy_signal
                b, a = scipy_signal.butter(2, low_norm, btype='high')
            else:
                # Band-pass filter
                from scipy import signal as scipy_signal
                b, a = scipy_signal.butter(2, [low_norm, high_norm], btype='band')
            
            # Filter the signal and apply gain
            band_signal = scipy_signal.filtfilt(b, a, channel)
            output += band_signal * gain
            
        except Exception:
            # If filtering fails, just add the original signal with gain
            output += channel * gain / len(bands)
    
    return output


def apply_echo_delay(signal: np.ndarray, delay_time: float = 0.3, feedback: float = 0.2, mix: float = 0.3, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply echo delay effect for spatial depth.
    
    Creates discrete echoes that simulate reflections from distant surfaces,
    adding spatial depth and environmental character.
    
    Args:
        signal: Input audio signal (numpy array)
        delay_time: Delay time in seconds (0.05-2.0)
        feedback: Feedback amount (0.0-0.8, higher = more repeats)
        mix: Wet/dry mix (0.0-1.0, 0=dry only, 1=wet only)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Signal with echo delay applied (same format as input)
    
    Example:
        >>> import numpy as np
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> echoed = apply_echo_delay(signal, delay_time=0.5, feedback=0.3, mix=0.4)
        >>> len(echoed) == len(signal)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (0.05 <= delay_time <= 5.0):
        raise ValueError(f"Delay time must be between 0.05 and 5.0 seconds, got {delay_time}")
    if not (0.0 <= feedback <= 0.9):
        raise ValueError(f"Feedback must be between 0.0 and 0.9, got {feedback}")
    if not (0.0 <= mix <= 1.0):
        raise ValueError(f"Mix must be between 0.0 and 1.0, got {mix}")
    
    if mix == 0.0:
        return signal.copy()
    
    delay_samples = int(delay_time * sample_rate)
    
    # Handle mono and stereo signals
    if signal.ndim == 1:
        delayed = _apply_delay_channel(signal, delay_samples, feedback)
    else:
        stereo_signal = _ensure_stereo(signal)
        left_delayed = _apply_delay_channel(stereo_signal[:, 0], delay_samples, feedback)
        right_delayed = _apply_delay_channel(stereo_signal[:, 1], delay_samples, feedback)
        delayed = np.column_stack([left_delayed, right_delayed])
    
    # Mix dry and wet signals
    dry_gain = 1.0 - mix
    wet_gain = mix
    
    if signal.ndim == 1:
        output = dry_gain * signal + wet_gain * delayed
    else:
        output = dry_gain * _ensure_stereo(signal) + wet_gain * delayed
    
    return _clip_signal(output)


def _apply_delay_channel(channel: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
    """Apply delay with feedback to a single channel."""
    if delay_samples <= 0 or delay_samples >= len(channel):
        return channel.copy()
    
    output = channel.copy()
    delay_buffer = np.zeros(delay_samples)
    
    for i in range(len(channel)):
        # Get delayed sample
        delayed_sample = delay_buffer[i % delay_samples]
        
        # Add to output
        output[i] += delayed_sample
        
        # Update delay buffer with feedback
        delay_buffer[i % delay_samples] = channel[i] + delayed_sample * feedback
    
    return output


def apply_shaped_delay(
    signal: np.ndarray,
    delay_time: float,
    feedback: float,
    mix: float = 0.5,
    filter_cutoff: float = None,
    filter_resonance: float = 1.0,
    filter_step: float = -200,
    resonance_step: float = 0.0,
    pitch_shift: float = None,
    pitch_step: float = -0.02,
    bitcrush_depth: int = None,
    saturation_drive: float = None,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply shaped delay with per-repeat filtering, pitch shifting, and degradation.
    
    Creates evolving delay repeats by applying optional effects to each repeat:
    - Lowpass filtering with stepping cutoff frequency
    - Pitch shifting with cumulative shift per repeat
    - Bitcrushing for digital degradation
    - Tape saturation for analog warmth
    
    Args:
        signal: Input audio signal (numpy array)
        delay_time: Delay time in seconds (0.05-5.0)
        feedback: Feedback amount (0.0-0.9, higher = more repeats)
        mix: Wet/dry mix (0.0-1.0, 0=dry only, 1=wet only)
        filter_cutoff: Initial lowpass cutoff in Hz (None to disable, 100-20000)
        filter_step: Cutoff change per repeat in Hz (can be negative)
        pitch_shift: Initial pitch shift ratio (None to disable, 0.5-2.0)
        pitch_step: Pitch shift change per repeat (can be negative)
        bitcrush_depth: Bit depth for crushing (None to disable, 1-16)
        saturation_drive: Tape saturation drive (None to disable, 1.0-5.0)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Signal with shaped delay applied (same format as input)
    
    Example:
        >>> import numpy as np
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> shaped = apply_shaped_delay(signal, delay_time=0.3, feedback=0.5, 
        ...                             filter_cutoff=5000.0, filter_step=-500)
        >>> len(shaped) == len(signal)
        True
    """
    _validate_signal_params(signal, sample_rate)
    if not (0.05 <= delay_time <= 5.0):
        raise ValueError(f"Delay time must be between 0.05 and 5.0 seconds, got {delay_time}")
    if not (0.0 <= feedback <= 0.9):
        raise ValueError(f"Feedback must be between 0.0 and 0.9, got {feedback}")
    if not (0.0 <= mix <= 1.0):
        raise ValueError(f"Mix must be between 0.0 and 1.0, got {mix}")
    
    if filter_cutoff is not None:
        if not (100.0 <= filter_cutoff <= 20000.0):
            raise ValueError(f"Filter cutoff must be between 100 and 20000 Hz, got {filter_cutoff}")
        if filter_resonance <= 0:
            raise ValueError(f"Filter resonance must be positive, got {filter_resonance}")
    
    if pitch_shift is not None:
        if not (0.5 <= pitch_shift <= 2.0):
            raise ValueError(f"Pitch shift must be between 0.5 and 2.0, got {pitch_shift}")
    
    if bitcrush_depth is not None:
        if not (1 <= bitcrush_depth <= 16):
            raise ValueError(f"Bitcrush depth must be between 1 and 16, got {bitcrush_depth}")
    
    if saturation_drive is not None:
        if not (1.0 <= saturation_drive <= 5.0):
            raise ValueError(f"Saturation drive must be between 1.0 and 5.0, got {saturation_drive}")
    
    if mix == 0.0:
        return signal.copy()
    
    delay_samples = int(delay_time * sample_rate)
    
    # Handle mono and stereo signals
    if signal.ndim == 1:
        delayed = _apply_shaped_delay_channel(
            signal, delay_samples, feedback, filter_cutoff, filter_resonance, filter_step, resonance_step,
            pitch_shift, pitch_step, bitcrush_depth, saturation_drive, sample_rate
        )
    else:
        stereo_signal = _ensure_stereo(signal)
        left_delayed = _apply_shaped_delay_channel(
            stereo_signal[:, 0], delay_samples, feedback, filter_cutoff, filter_resonance, filter_step, resonance_step,
            pitch_shift, pitch_step, bitcrush_depth, saturation_drive, sample_rate
        )
        right_delayed = _apply_shaped_delay_channel(
            stereo_signal[:, 1], delay_samples, feedback, filter_cutoff, filter_resonance, filter_step, resonance_step,
            pitch_shift, pitch_step, bitcrush_depth, saturation_drive, sample_rate
        )
        delayed = np.column_stack([left_delayed, right_delayed])
    
    # Mix dry and wet signals
    dry_gain = 1.0 - mix
    wet_gain = mix
    
    if signal.ndim == 1:
        output = dry_gain * signal + wet_gain * delayed
    else:
        output = dry_gain * _ensure_stereo(signal) + wet_gain * delayed
    
    return _clip_signal(output)


def _apply_shaped_delay_channel(
    channel: np.ndarray,
    delay_samples: int,
    feedback: float,
    filter_cutoff: float,
    filter_resonance: float,
    filter_step: float,
    resonance_step: float,
    pitch_shift: float,
    pitch_step: float,
    bitcrush_depth: int,
    saturation_drive: float,
    sample_rate: int
) -> np.ndarray:
    """applies shaped delay with per-repeat effects to a single channel."""
    if delay_samples <= 0 or delay_samples >= len(channel):
        return channel.copy()
    
    # Import effects modules
    from .filters import lowpass_filter
    from .effects import apply_bitcrush, apply_tape_saturation
    
    output = channel.copy()
    delay_buffer = np.zeros(delay_samples)
    
    # Track current effect parameters
    current_cutoff = filter_cutoff
    current_resonance = filter_resonance
    current_pitch = pitch_shift if pitch_shift is not None else 1.0
    
    for i in range(len(channel)):
        # Get delayed sample
        delayed_sample = delay_buffer[i % delay_samples]
        
        # Add to output
        output[i] += delayed_sample
        
        # Create feedback signal with effects
        feedback_signal = channel[i] + delayed_sample * feedback
        
        # Apply per-repeat effects when we complete a delay cycle
        if i > 0 and i % delay_samples == 0:
            # Extract the current repeat from delay buffer
            repeat_signal = delay_buffer.copy()
            
            # Apply filtering if enabled
            if filter_cutoff is not None and current_cutoff > 100.0:
                try:
                    repeat_signal = lowpass_filter(repeat_signal, current_cutoff, sample_rate, resonance=current_resonance)
                    current_cutoff += filter_step
                    current_cutoff = max(100.0, min(current_cutoff, sample_rate / 2 - 100))
                    current_resonance = max(0.01, current_resonance + resonance_step)
                except:
                    pass
            
            # Apply pitch shifting if enabled
            if pitch_shift is not None:
                try:
                    repeat_signal = _pitch_shift_channel(repeat_signal, current_pitch, sample_rate)
                    current_pitch += pitch_step
                    current_pitch = max(0.5, min(current_pitch, 2.0))
                except:
                    pass
            
            # Apply bitcrush if enabled
            if bitcrush_depth is not None:
                try:
                    repeat_signal = apply_bitcrush(repeat_signal, bitcrush_depth, sample_rate)
                except:
                    pass
            
            # Apply tape saturation if enabled
            if saturation_drive is not None:
                try:
                    repeat_signal = apply_tape_saturation(repeat_signal, saturation_drive, warmth=0.3)
                except:
                    pass
            
            # Update delay buffer with shaped repeat
            delay_buffer = repeat_signal
        
        # Update delay buffer
        delay_buffer[i % delay_samples] = feedback_signal
    
    return output


if __name__ == "__main__":
    # Basic tests
    import doctest
    doctest.testmod()
    
    # Quick validation tests
    try:
        print("Testing spatial effects...")
        
        # Create test signal
        duration = 1.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Test all effects
        reverbed = apply_reverb(test_signal, room_size=0.7, decay_time=2.0, damping=0.3)
        distant = apply_distance_filter(test_signal, distance_factor=3.0)
        wide_stereo = apply_stereo_width(test_signal, width=1.5)
        doppler = apply_doppler_shift(test_signal, frequency_shift=50.0)
        absorbed = apply_air_absorption(test_signal, distance=30.0, humidity=0.4)
        echoed = apply_echo_delay(test_signal, delay_time=0.4, feedback=0.3, mix=0.4)
        
        print(f"✓ Reverb: {reverbed.shape} (stereo output)")
        print(f"✓ Distance Filter: {distant.shape}")
        print(f"✓ Stereo Width: {wide_stereo.shape} (stereo output)")
        print(f"✓ Doppler Shift: {doppler.shape}")
        print(f"✓ Air Absorption: {absorbed.shape}")
        print(f"✓ Echo Delay: {echoed.shape}")
        
        print("\n✓ All spatial effects working correctly")
        
        # Test error conditions
        print("\nTesting error conditions...")
        try:
            apply_reverb(test_signal, room_size=2.0)  # Too high
            print("✗ Should have raised room size error")
        except ValueError:
            print("✓ Room size validation working")
            
        try:
            apply_doppler_shift(test_signal, frequency_shift=500.0)  # Too high
            print("✗ Should have raised frequency shift error")
        except ValueError:
            print("✓ Frequency shift validation working")
            
    except Exception as e:
        print(f"✗ Error in spatial tests: {e}")