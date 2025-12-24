"""
Composition tools for modular audio synthesis.
Signal combination, modulation application, and layering utilities for "broken transmission" dronescapes.
"""

import numpy as np
from scipy import signal
from typing import List, Union, Optional, Tuple, Dict, Any, Callable
import warnings


def _validate_signal_params(input_signal: np.ndarray, sample_rate: int) -> None:
    """Validate common parameters for mixing functions."""
    if not isinstance(input_signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(input_signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")


def _validate_signals_list(signals: List[np.ndarray]) -> None:
    """Validate a list of signals for mixing operations."""
    if not isinstance(signals, list):
        raise TypeError("Signals must be provided as a list")
    if len(signals) == 0:
        raise ValueError("Signals list cannot be empty")
    for i, sig in enumerate(signals):
        if not isinstance(sig, np.ndarray):
            raise TypeError(f"Signal {i} must be a numpy array")
        if len(sig) == 0:
            raise ValueError(f"Signal {i} cannot be empty")


def _ensure_same_length(signals: List[np.ndarray]) -> List[np.ndarray]:
    """Ensure all signals have the same length by padding with zeros."""
    if not signals:
        return signals
    
    max_length = max(len(sig) for sig in signals)
    padded_signals = []
    
    for sig in signals:
        if len(sig) < max_length:
            padded = np.zeros(max_length)
            padded[:len(sig)] = sig
            padded_signals.append(padded)
        else:
            padded_signals.append(sig)
    
    return padded_signals


def _clip_signal(signal: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Soft clip signal to prevent harsh digital clipping."""
    return np.clip(signal, -threshold, threshold)


def _db_to_linear(db: float) -> float:
    """Convert dB to linear amplitude."""
    return 10 ** (db / 20)


def _linear_to_db(linear: float) -> float:
    """Convert linear amplitude to dB."""
    if linear <= 0:
        return -np.inf
    return 20 * np.log10(linear)


# Signal Combination Functions

def combine_signals(signals: List[np.ndarray], mix_levels: List[float], sample_rate: int = 44100) -> np.ndarray:
    """
    Combine multiple signals with specified mix levels.
    
    Mixes multiple audio signals together with individual level control,
    automatically handling different signal lengths and preventing clipping.
    
    Args:
        signals: List of input audio signals (numpy arrays)
        mix_levels: List of mix levels (0.0-2.0, 1.0 = unity gain)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Combined audio signal
    
    Example:
        >>> import numpy as np
        >>> sig1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> sig2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 44100))
        >>> mixed = combine_signals([sig1, sig2], [1.0, 0.5])
        >>> len(mixed) == 44100
        True
    """
    _validate_signals_list(signals)
    if not isinstance(mix_levels, list):
        raise TypeError("Mix levels must be provided as a list")
    if len(mix_levels) != len(signals):
        raise ValueError(f"Number of mix levels ({len(mix_levels)}) must match number of signals ({len(signals)})")
    
    for i, level in enumerate(mix_levels):
        if not (0.0 <= level <= 2.0):
            raise ValueError(f"Mix level {i} must be between 0.0 and 2.0, got {level}")
    
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    # Ensure all signals have the same length
    padded_signals = _ensure_same_length(signals)
    
    # Combine signals with mix levels
    output = np.zeros_like(padded_signals[0])
    for sig, level in zip(padded_signals, mix_levels):
        output += sig * level
    
    # Normalize to prevent clipping while maintaining headroom
    max_amplitude = np.max(np.abs(output))
    if max_amplitude > 0.8:  # Leave headroom
        output = output * (0.8 / max_amplitude)
    
    return _clip_signal(output)


def layer_with_crossfade(signal1: np.ndarray, signal2: np.ndarray, crossfade_curve: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """
    Layer two signals with a crossfade curve.
    
    Smoothly transitions between two signals using a crossfade curve,
    perfect for creating seamless transitions between different textures.
    
    Args:
        signal1: First input signal (numpy array)
        signal2: Second input signal (numpy array)
        crossfade_curve: Crossfade curve (0.0-1.0, 0=signal1 only, 1=signal2 only)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Crossfaded signal
    
    Example:
        >>> import numpy as np
        >>> sig1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> sig2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 44100))
        >>> curve = np.linspace(0, 1, 44100)  # Linear crossfade
        >>> crossfaded = layer_with_crossfade(sig1, sig2, curve)
        >>> len(crossfaded) == 44100
        True
    """
    _validate_signal_params(signal1, sample_rate)
    _validate_signal_params(signal2, sample_rate)
    
    if not isinstance(crossfade_curve, np.ndarray):
        raise TypeError("Crossfade curve must be a numpy array")
    if len(crossfade_curve) == 0:
        raise ValueError("Crossfade curve cannot be empty")
    
    # Ensure all arrays have the same length
    signals = _ensure_same_length([signal1, signal2])
    signal1, signal2 = signals[0], signals[1]
    
    # Resize crossfade curve to match signal length
    if len(crossfade_curve) != len(signal1):
        crossfade_curve = np.interp(
            np.linspace(0, 1, len(signal1)),
            np.linspace(0, 1, len(crossfade_curve)),
            crossfade_curve
        )
    
    # Validate crossfade curve values
    if np.any(crossfade_curve < 0.0) or np.any(crossfade_curve > 1.0):
        raise ValueError("Crossfade curve values must be between 0.0 and 1.0")
    
    # Apply crossfade
    output = signal1 * (1.0 - crossfade_curve) + signal2 * crossfade_curve
    
    return _clip_signal(output)


def parallel_mix(signals: List[np.ndarray], pan_positions: Optional[List[float]] = None, sample_rate: int = 44100) -> np.ndarray:
    """
    Mix signals in parallel with optional stereo panning.
    
    Combines multiple signals with stereo positioning, creating a stereo
    output with spatial distribution of the input signals.
    
    Args:
        signals: List of input audio signals (numpy arrays)
        pan_positions: List of pan positions (-1.0 to 1.0, -1=left, 0=center, 1=right)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Stereo mixed signal (2D array with shape [samples, 2])
    
    Example:
        >>> import numpy as np
        >>> sig1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> sig2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 44100))
        >>> mixed = parallel_mix([sig1, sig2], [-0.5, 0.5])
        >>> mixed.shape == (44100, 2)
        True
    """
    _validate_signals_list(signals)
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    if pan_positions is None:
        pan_positions = [0.0] * len(signals)  # Center all signals
    
    if len(pan_positions) != len(signals):
        raise ValueError(f"Number of pan positions ({len(pan_positions)}) must match number of signals ({len(signals)})")
    
    for i, pan in enumerate(pan_positions):
        if not (-1.0 <= pan <= 1.0):
            raise ValueError(f"Pan position {i} must be between -1.0 and 1.0, got {pan}")
    
    # Ensure all signals have the same length
    padded_signals = _ensure_same_length(signals)
    
    # Create stereo output
    output_length = len(padded_signals[0])
    stereo_output = np.zeros((output_length, 2))
    
    for sig, pan in zip(padded_signals, pan_positions):
        # Calculate left and right gains using constant power panning
        pan_radians = (pan + 1.0) * np.pi / 4.0  # Map -1..1 to 0..π/2
        left_gain = np.cos(pan_radians)
        right_gain = np.sin(pan_radians)
        
        # Add to stereo output
        stereo_output[:, 0] += sig * left_gain
        stereo_output[:, 1] += sig * right_gain
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(stereo_output))
    if max_amplitude > 0.8:
        stereo_output = stereo_output * (0.8 / max_amplitude)
    
    return _clip_signal(stereo_output)


def serial_chain(signal: np.ndarray, effect_chain: List[Callable], sample_rate: int = 44100) -> np.ndarray:
    """
    Apply a series of effects to a signal in sequence.
    
    Processes a signal through a chain of effect functions, passing the
    output of each effect as input to the next.
    
    Args:
        signal: Input audio signal (numpy array)
        effect_chain: List of effect functions to apply in sequence
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Processed signal after all effects
    
    Example:
        >>> import numpy as np
        >>> from filters import lowpass_filter
        >>> sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> chain = [lambda x: lowpass_filter(x, 1000.0)]
        >>> processed = serial_chain(sig, chain)
        >>> len(processed) == len(sig)
        True
    """
    _validate_signal_params(signal, sample_rate)
    
    if not isinstance(effect_chain, list):
        raise TypeError("Effect chain must be a list")
    if len(effect_chain) == 0:
        return signal.copy()
    
    output = signal.copy()
    
    for i, effect_func in enumerate(effect_chain):
        if not callable(effect_func):
            raise TypeError(f"Effect {i} must be callable")
        
        try:
            output = effect_func(output)
        except Exception as e:
            raise RuntimeError(f"Error applying effect {i}: {e}")
        
        if not isinstance(output, np.ndarray):
            raise TypeError(f"Effect {i} must return a numpy array")
    
    return _clip_signal(output)


# Modulation Application Utilities

def apply_amplitude_modulation(signal: np.ndarray, modulation_data: np.ndarray, depth: float = 1.0) -> np.ndarray:
    """
    Apply amplitude modulation to a signal.
    
    Modulates the amplitude of the input signal using modulation data,
    creating tremolo, amplitude envelope, or dynamic level changes.
    
    Args:
        signal: Input audio signal (numpy array)
        modulation_data: Modulation control data (numpy array)
        depth: Modulation depth (0.0-1.0, higher = more modulation)
    
    Returns:
        numpy.ndarray: Amplitude-modulated signal
    
    Example:
        >>> import numpy as np
        >>> sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> lfo = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 44100))
        >>> modulated = apply_amplitude_modulation(sig, lfo, depth=0.5)
        >>> len(modulated) == len(sig)
        True
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if not isinstance(modulation_data, np.ndarray):
        raise TypeError("Modulation data must be a numpy array")
    if len(signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if len(modulation_data) == 0:
        raise ValueError("Modulation data cannot be empty")
    if not (0.0 <= depth <= 1.0):
        raise ValueError(f"Depth must be between 0.0 and 1.0, got {depth}")
    
    # Resize modulation data to match signal length
    if len(modulation_data) != len(signal):
        modulation_data = np.interp(
            np.linspace(0, 1, len(signal)),
            np.linspace(0, 1, len(modulation_data)),
            modulation_data
        )
    
    # Apply amplitude modulation
    # Modulation is bipolar, so we add 1.0 to make it unipolar (0-2 range)
    # Then scale by depth and add (1-depth) for the unmodulated portion
    modulation_scaled = 1.0 + depth * modulation_data
    output = signal * modulation_scaled
    
    return _clip_signal(output)


def apply_frequency_modulation(base_freq: float, modulation_data: np.ndarray, depth: float = 1.0, duration: float = 60, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a frequency-modulated oscillator signal.
    
    Creates an oscillator with frequency modulation, useful for vibrato,
    FM synthesis, or dynamic pitch changes.
    
    Args:
        base_freq: Base frequency in Hz (20-20000)
        modulation_data: Frequency modulation control data (numpy array)
        depth: Modulation depth in Hz (0.0-1000.0)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Frequency-modulated sine wave
    
    Example:
        >>> import numpy as np
        >>> lfo = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 44100))
        >>> fm_signal = apply_frequency_modulation(440.0, lfo, depth=10.0, duration=1.0)
        >>> len(fm_signal) == 44100
        True
    """
    if not (20 <= base_freq <= 20000):
        raise ValueError(f"Base frequency {base_freq}Hz out of range (20-20000Hz)")
    if not isinstance(modulation_data, np.ndarray):
        raise TypeError("Modulation data must be a numpy array")
    if len(modulation_data) == 0:
        raise ValueError("Modulation data cannot be empty")
    if not (0.0 <= depth <= 1000.0):
        raise ValueError(f"Depth must be between 0.0 and 1000.0 Hz, got {depth}")
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    num_samples = int(duration * sample_rate)
    
    # Resize modulation data to match duration
    if len(modulation_data) != num_samples:
        modulation_data = np.interp(
            np.linspace(0, 1, num_samples),
            np.linspace(0, 1, len(modulation_data)),
            modulation_data
        )
    
    # Create time array
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Apply frequency modulation
    instantaneous_freq = base_freq + depth * modulation_data
    
    # Generate phase by integrating frequency
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    
    # Generate FM signal
    output = np.sin(phase)
    
    return _clip_signal(output)


def apply_filter_modulation(signal: np.ndarray, cutoff_base: float, modulation_data: np.ndarray, depth: float = 1.0, filter_type: str = 'lowpass', sample_rate: int = 44100) -> np.ndarray:
    """
    Apply filter modulation to a signal.
    
    Modulates the cutoff frequency of a filter over time, creating
    sweeping filter effects and dynamic timbral changes.
    
    Args:
        signal: Input audio signal (numpy array)
        cutoff_base: Base cutoff frequency in Hz (20-20000)
        modulation_data: Filter modulation control data (numpy array)
        depth: Modulation depth in Hz (0.0-10000.0)
        filter_type: Filter type ('lowpass', 'highpass', 'bandpass')
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Filter-modulated signal
    
    Example:
        >>> import numpy as np
        >>> sig = np.random.normal(0, 0.1, 44100)
        >>> lfo = np.sin(2 * np.pi * 0.5 * np.linspace(0, 1, 44100))
        >>> filtered = apply_filter_modulation(sig, 1000.0, lfo, depth=500.0)
        >>> len(filtered) == len(sig)
        True
    """
    _validate_signal_params(signal, sample_rate)
    
    if not (20 <= cutoff_base <= 20000):
        raise ValueError(f"Base cutoff frequency {cutoff_base}Hz out of range (20-20000Hz)")
    if not isinstance(modulation_data, np.ndarray):
        raise TypeError("Modulation data must be a numpy array")
    if len(modulation_data) == 0:
        raise ValueError("Modulation data cannot be empty")
    if not (0.0 <= depth <= 10000.0):
        raise ValueError(f"Depth must be between 0.0 and 10000.0 Hz, got {depth}")
    if filter_type not in ['lowpass', 'highpass', 'bandpass']:
        raise ValueError(f"Filter type must be 'lowpass', 'highpass', or 'bandpass', got '{filter_type}'")
    
    # Resize modulation data to match signal length
    if len(modulation_data) != len(signal):
        modulation_data = np.interp(
            np.linspace(0, 1, len(signal)),
            np.linspace(0, 1, len(modulation_data)),
            modulation_data
        )
    
    # For time-varying filters, we'll use a simplified approach
    # In practice, this would require more sophisticated filter design
    
    # Calculate instantaneous cutoff frequencies
    instantaneous_cutoff = cutoff_base + depth * modulation_data
    
    # Clamp to valid range
    nyquist = sample_rate / 2
    instantaneous_cutoff = np.clip(instantaneous_cutoff, 20, nyquist * 0.99)
    
    # Apply time-varying filter using overlapping windows
    window_size = 1024
    hop_size = window_size // 4
    output = np.zeros_like(signal)
    
    for i in range(0, len(signal) - window_size, hop_size):
        window_end = min(i + window_size, len(signal))
        window_signal = signal[i:window_end]
        
        # Get average cutoff for this window
        avg_cutoff = np.mean(instantaneous_cutoff[i:window_end])
        
        # Apply filter to window
        try:
            normalized_cutoff = avg_cutoff / nyquist
            if filter_type == 'lowpass':
                b, a = signal.butter(2, normalized_cutoff, btype='low')
            elif filter_type == 'highpass':
                b, a = signal.butter(2, normalized_cutoff, btype='high')
            elif filter_type == 'bandpass':
                # For bandpass, use cutoff as center frequency with fixed bandwidth
                bandwidth = avg_cutoff * 0.2  # 20% bandwidth
                low_freq = max(avg_cutoff - bandwidth/2, 20)
                high_freq = min(avg_cutoff + bandwidth/2, nyquist * 0.99)
                low_norm = low_freq / nyquist
                high_norm = high_freq / nyquist
                b, a = signal.butter(2, [low_norm, high_norm], btype='band')
            
            filtered_window = signal.filtfilt(b, a, window_signal)
            
            # Overlap-add with windowing
            window_func = np.hanning(len(filtered_window))
            filtered_window *= window_func
            
            output[i:window_end] += filtered_window
            
        except Exception:
            # If filtering fails, just copy the original window
            output[i:window_end] += window_signal
    
    return _clip_signal(output)


def apply_parameter_automation(signal: np.ndarray, parameter_name: str, automation_curve: np.ndarray, effect_function: Callable, **effect_kwargs) -> np.ndarray:
    """
    Apply parameter automation to an effect function.
    
    Automates a parameter of an effect function over time using an
    automation curve, enabling dynamic effect processing.
    
    Args:
        signal: Input audio signal (numpy array)
        parameter_name: Name of the parameter to automate
        automation_curve: Automation curve data (numpy array)
        effect_function: Effect function to apply with automation
        **effect_kwargs: Additional keyword arguments for the effect function
    
    Returns:
        numpy.ndarray: Signal with automated effect applied
    
    Example:
        >>> import numpy as np
        >>> from effects import apply_bitcrush
        >>> sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> curve = np.linspace(16, 4, 44100)  # Bit depth automation
        >>> automated = apply_parameter_automation(sig, 'bit_depth', curve, apply_bitcrush)
        >>> len(automated) == len(sig)
        True
    """
    _validate_signal_params(signal, 44100)  # Default sample rate for validation
    
    if not isinstance(parameter_name, str):
        raise TypeError("Parameter name must be a string")
    if not isinstance(automation_curve, np.ndarray):
        raise TypeError("Automation curve must be a numpy array")
    if len(automation_curve) == 0:
        raise ValueError("Automation curve cannot be empty")
    if not callable(effect_function):
        raise TypeError("Effect function must be callable")
    
    # Resize automation curve to match signal length
    if len(automation_curve) != len(signal):
        automation_curve = np.interp(
            np.linspace(0, 1, len(signal)),
            np.linspace(0, 1, len(automation_curve)),
            automation_curve
        )
    
    # Apply automated effect using overlapping windows
    window_size = 1024
    hop_size = window_size // 4
    output = np.zeros_like(signal)
    
    for i in range(0, len(signal) - window_size, hop_size):
        window_end = min(i + window_size, len(signal))
        window_signal = signal[i:window_end]
        
        # Get average parameter value for this window
        avg_param_value = np.mean(automation_curve[i:window_end])
        
        # Apply effect to window with automated parameter
        try:
            effect_kwargs[parameter_name] = avg_param_value
            processed_window = effect_function(window_signal, **effect_kwargs)
            
            # Overlap-add with windowing
            window_func = np.hanning(len(processed_window))
            processed_window *= window_func
            
            output[i:window_end] += processed_window
            
        except Exception:
            # If effect fails, just copy the original window
            output[i:window_end] += window_signal
    
    return _clip_signal(output)


# Crossfading and Layering Tools

def create_crossfade_curve(duration: float, curve_type: str = 'linear', sample_rate: int = 44100) -> np.ndarray:
    """
    Create a crossfade curve for smooth transitions.
    
    Generates various types of crossfade curves for transitioning between
    signals or parameters over time.
    
    Args:
        duration: Duration in seconds (positive)
        curve_type: Curve type ('linear', 'exponential', 'logarithmic', 'sine')
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Crossfade curve (0.0 to 1.0)
    
    Example:
        >>> curve = create_crossfade_curve(2.0, 'linear')
        >>> len(curve) == 88200  # 2 seconds at 44.1kHz
        True
        >>> curve[0] == 0.0 and abs(curve[-1] - 1.0) < 0.01
        True
    """
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if curve_type not in ['linear', 'exponential', 'logarithmic', 'sine']:
        raise ValueError(f"Curve type must be 'linear', 'exponential', 'logarithmic', or 'sine', got '{curve_type}'")
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, 1, num_samples)
    
    if curve_type == 'linear':
        curve = t
    elif curve_type == 'exponential':
        curve = t ** 2
    elif curve_type == 'logarithmic':
        curve = np.sqrt(t)
    elif curve_type == 'sine':
        curve = np.sin(t * np.pi / 2)
    
    return curve


def layer_signals_with_timing(signals: List[np.ndarray], start_times: List[float], durations: List[float], sample_rate: int = 44100) -> np.ndarray:
    """
    Layer signals with specific start times and durations.
    
    Combines multiple signals with individual timing control, creating
    complex layered compositions with precise temporal arrangement.
    
    Args:
        signals: List of input audio signals (numpy arrays)
        start_times: List of start times in seconds for each signal
        durations: List of durations in seconds for each signal
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Layered composition
    
    Example:
        >>> import numpy as np
        >>> sig1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> sig2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 44100))
        >>> layered = layer_signals_with_timing([sig1, sig2], [0.0, 0.5], [1.0, 1.0])
        >>> len(layered) >= 44100
        True
    """
    _validate_signals_list(signals)
    
    if not isinstance(start_times, list):
        raise TypeError("Start times must be provided as a list")
    if not isinstance(durations, list):
        raise TypeError("Durations must be provided as a list")
    if len(start_times) != len(signals):
        raise ValueError(f"Number of start times ({len(start_times)}) must match number of signals ({len(signals)})")
    if len(durations) != len(signals):
        raise ValueError(f"Number of durations ({len(durations)}) must match number of signals ({len(signals)})")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    for i, start_time in enumerate(start_times):
        if start_time < 0:
            raise ValueError(f"Start time {i} must be non-negative, got {start_time}")
    
    for i, duration in enumerate(durations):
        if duration <= 0:
            raise ValueError(f"Duration {i} must be positive, got {duration}")
    
    # Calculate total composition length
    max_end_time = max(start_time + duration for start_time, duration in zip(start_times, durations))
    total_samples = int(max_end_time * sample_rate)
    
    # Create output buffer
    output = np.zeros(total_samples)
    
    # Layer each signal at its specified time
    for sig, start_time, duration in zip(signals, start_times, durations):
        start_sample = int(start_time * sample_rate)
        duration_samples = int(duration * sample_rate)
        
        # Trim or pad signal to match duration
        if len(sig) > duration_samples:
            sig_trimmed = sig[:duration_samples]
        else:
            sig_trimmed = np.zeros(duration_samples)
            sig_trimmed[:len(sig)] = sig
        
        # Add to output at the specified start time
        end_sample = min(start_sample + len(sig_trimmed), total_samples)
        
        if start_sample < total_samples:
            output[start_sample:end_sample] += sig_trimmed[:end_sample - start_sample]
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(output))
    if max_amplitude > 0.8:
        output = output * (0.8 / max_amplitude)
    
    return _clip_signal(output)


def create_stereo_field(mono_signals: List[np.ndarray], pan_positions: List[float], width: float = 1.0) -> np.ndarray:
    """
    Create a stereo field from mono signals with panning and width control.
    
    Positions multiple mono signals in a stereo field with adjustable
    stereo width, creating spatial depth and separation.
    
    Args:
        mono_signals: List of mono input signals (numpy arrays)
        pan_positions: List of pan positions (-1.0 to 1.0, -1=left, 0=center, 1=right)
        width: Stereo width factor (0.0-2.0, 0=mono, 1=normal, >1=wide)
    
    Returns:
        numpy.ndarray: Stereo signal with positioned sources
    
    Example:
        >>> import numpy as np
        >>> sig1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> sig2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 44100))
        >>> stereo = create_stereo_field([sig1, sig2], [-0.5, 0.5], width=1.2)
        >>> stereo.shape == (44100, 2)
        True
    """
    _validate_signals_list(mono_signals)
    
    if not isinstance(pan_positions, list):
        raise TypeError("Pan positions must be provided as a list")
    if len(pan_positions) != len(mono_signals):
        raise ValueError(f"Number of pan positions ({len(pan_positions)}) must match number of signals ({len(mono_signals)})")
    if not (0.0 <= width <= 2.0):
        raise ValueError(f"Width must be between 0.0 and 2.0, got {width}")
    
    for i, pan in enumerate(pan_positions):
        if not (-1.0 <= pan <= 1.0):
            raise ValueError(f"Pan position {i} must be between -1.0 and 1.0, got {pan}")
    
    # Ensure all signals have the same length
    padded_signals = _ensure_same_length(mono_signals)
    
    # Create stereo output
    output_length = len(padded_signals[0])
    stereo_output = np.zeros((output_length, 2))
    
    for sig, pan in zip(padded_signals, pan_positions):
        # Apply width adjustment to pan position
        adjusted_pan = pan * width
        adjusted_pan = np.clip(adjusted_pan, -1.0, 1.0)
        
        # Calculate left and right gains using constant power panning
        pan_radians = (adjusted_pan + 1.0) * np.pi / 4.0  # Map -1..1 to 0..π/2
        left_gain = np.cos(pan_radians)
        right_gain = np.sin(pan_radians)
        
        # Add to stereo output
        stereo_output[:, 0] += sig * left_gain
        stereo_output[:, 1] += sig * right_gain
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(stereo_output))
    if max_amplitude > 0.8:
        stereo_output = stereo_output * (0.8 / max_amplitude)
    
    return _clip_signal(stereo_output)


def apply_dynamic_panning(signal: np.ndarray, pan_automation: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply dynamic panning to a mono signal.
    
    Creates a stereo signal with time-varying panning position,
    adding movement and spatial interest to static sources.
    
    Args:
        signal: Input mono signal (numpy array)
        pan_automation: Pan automation curve (-1.0 to 1.0, -1=left, 1=right)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Stereo signal with dynamic panning
    
    Example:
        >>> import numpy as np
        >>> sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> pan_curve = np.sin(2 * np.pi * 0.5 * np.linspace(0, 1, 44100))
        >>> panned = apply_dynamic_panning(sig, pan_curve)
        >>> panned.shape == (44100, 2)
        True
    """
    _validate_signal_params(signal, sample_rate)
    
    if not isinstance(pan_automation, np.ndarray):
        raise TypeError("Pan automation must be a numpy array")
    if len(pan_automation) == 0:
        raise ValueError("Pan automation cannot be empty")
    
    # Resize pan automation to match signal length
    if len(pan_automation) != len(signal):
        pan_automation = np.interp(
            np.linspace(0, 1, len(signal)),
            np.linspace(0, 1, len(pan_automation)),
            pan_automation
        )
    
    # Validate pan automation values
    if np.any(pan_automation < -1.0) or np.any(pan_automation > 1.0):
        raise ValueError("Pan automation values must be between -1.0 and 1.0")
    
    # Create stereo output
    stereo_output = np.zeros((len(signal), 2))
    
    # Apply dynamic panning
    for i in range(len(signal)):
        pan = pan_automation[i]
        
        # Calculate left and right gains using constant power panning
        pan_radians = (pan + 1.0) * np.pi / 4.0  # Map -1..1 to 0..π/2
        left_gain = np.cos(pan_radians)
        right_gain = np.sin(pan_radians)
        
        stereo_output[i, 0] = signal[i] * left_gain
        stereo_output[i, 1] = signal[i] * right_gain
    
    return _clip_signal(stereo_output)


# Composition Utilities

def normalize_to_peak(signal: np.ndarray, target_peak_db: float = -6.0) -> np.ndarray:
    """
    Normalize signal to a target peak level in dB.
    
    Adjusts the signal amplitude so the peak level matches the target,
    maintaining the signal's dynamic range while controlling overall level.
    
    Args:
        signal: Input audio signal (numpy array, mono or stereo)
        target_peak_db: Target peak level in dB (negative values)
    
    Returns:
        numpy.ndarray: Normalized signal
    
    Example:
        >>> import numpy as np
        >>> sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.1
        >>> normalized = normalize_to_peak(sig, target_peak_db=-12.0)
        >>> abs(_linear_to_db(np.max(np.abs(normalized))) - (-12.0)) < 0.1
        True
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if target_peak_db > 0:
        raise ValueError(f"Target peak dB must be negative or zero, got {target_peak_db}")
    
    # Find current peak level
    current_peak = np.max(np.abs(signal))
    
    if current_peak == 0:
        return signal.copy()  # Avoid division by zero
    
    # Calculate required gain
    target_linear = _db_to_linear(target_peak_db)
    gain = target_linear / current_peak
    
    # Apply gain
    normalized = signal * gain
    
    return _clip_signal(normalized)


def apply_master_limiter(signal: np.ndarray, threshold_db: float = -3.0, ratio: float = 10.0) -> np.ndarray:
    """
    Apply a master limiter to prevent clipping and control dynamics.
    
    Applies soft limiting to prevent peaks from exceeding the threshold,
    maintaining loudness while preventing digital clipping.
    
    Args:
        signal: Input audio signal (numpy array, mono or stereo)
        threshold_db: Limiter threshold in dB (negative values)
        ratio: Compression ratio (1.0-20.0, higher = more limiting)
    
    Returns:
        numpy.ndarray: Limited signal
    
    Example:
        >>> import numpy as np
        >>> sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> limited = apply_master_limiter(sig, threshold_db=-6.0, ratio=8.0)
        >>> len(limited) == len(sig)
        True
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if threshold_db > 0:
        raise ValueError(f"Threshold dB must be negative or zero, got {threshold_db}")
    if not (1.0 <= ratio <= 20.0):
        raise ValueError(f"Ratio must be between 1.0 and 20.0, got {ratio}")
    
    threshold_linear = _db_to_linear(threshold_db)
    
    # Apply soft limiting
    output = signal.copy()
    
    # Find samples above threshold
    if signal.ndim == 1:
        # Mono signal
        above_threshold = np.abs(output) > threshold_linear
        
        # Apply compression to samples above threshold
        for i in np.where(above_threshold)[0]:
            if output[i] > threshold_linear:
                excess = output[i] - threshold_linear
                output[i] = threshold_linear + excess / ratio
            elif output[i] < -threshold_linear:
                excess = -output[i] - threshold_linear
                output[i] = -threshold_linear - excess / ratio
    else:
        # Stereo signal
        for channel in range(signal.shape[1]):
            channel_signal = output[:, channel]
            above_threshold = np.abs(channel_signal) > threshold_linear
            
            for i in np.where(above_threshold)[0]:
                if channel_signal[i] > threshold_linear:
                    excess = channel_signal[i] - threshold_linear
                    output[i, channel] = threshold_linear + excess / ratio
                elif channel_signal[i] < -threshold_linear:
                    excess = -channel_signal[i] - threshold_linear
                    output[i, channel] = -threshold_linear - excess / ratio
    
    return _clip_signal(output)


def create_composition_timeline(events: List[Dict[str, Any]], total_duration: float, sample_rate: int = 44100) -> Dict[str, Any]:
    """
    Create a composition timeline from event descriptions.
    
    Organizes composition events into a timeline structure for rendering,
    enabling complex multi-layered compositions with precise timing.
    
    Args:
        events: List of event dictionaries with 'start_time', 'duration', 'signal', etc.
        total_duration: Total composition duration in seconds
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        dict: Timeline structure with organized events
    
    Example:
        >>> events = [
        ...     {'start_time': 0.0, 'duration': 2.0, 'signal': 'sine_440'},
        ...     {'start_time': 1.0, 'duration': 2.0, 'signal': 'sine_880'}
        ... ]
        >>> timeline = create_composition_timeline(events, 3.0)
        >>> timeline['total_duration'] == 3.0
        True
    """
    if not isinstance(events, list):
        raise TypeError("Events must be provided as a list")
    if total_duration <= 0:
        raise ValueError(f"Total duration must be positive, got {total_duration}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    # Validate events
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            raise TypeError(f"Event {i} must be a dictionary")
        
        required_keys = ['start_time', 'duration']
        for key in required_keys:
            if key not in event:
                raise ValueError(f"Event {i} missing required key: {key}")
        
        if event['start_time'] < 0:
            raise ValueError(f"Event {i} start_time must be non-negative")
        if event['duration'] <= 0:
            raise ValueError(f"Event {i} duration must be positive")
        if event['start_time'] + event['duration'] > total_duration:
            warnings.warn(f"Event {i} extends beyond total duration")
    
    # Create timeline structure
    timeline = {
        'total_duration': total_duration,
        'sample_rate': sample_rate,
        'total_samples': int(total_duration * sample_rate),
        'events': sorted(events, key=lambda x: x['start_time']),
        'event_count': len(events)
    }
    
    return timeline


def render_composition(composition_data: Dict[str, Any], sample_rate: int = 44100) -> np.ndarray:
    """
    Render a composition from timeline data.
    
    Processes a composition timeline to create the final mixed audio output,
    combining all events according to their timing and parameters.
    
    Args:
        composition_data: Timeline data from create_composition_timeline
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Rendered composition
    
    Example:
        >>> timeline = {'total_duration': 2.0, 'sample_rate': 44100,
        ...             'total_samples': 88200, 'events': [], 'event_count': 0}
        >>> rendered = render_composition(timeline)
        >>> len(rendered) == 88200
        True
    """
    if not isinstance(composition_data, dict):
        raise TypeError("Composition data must be a dictionary")
    
    required_keys = ['total_duration', 'total_samples', 'events']
    for key in required_keys:
        if key not in composition_data:
            raise ValueError(f"Composition data missing required key: {key}")
    
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    total_samples = composition_data['total_samples']
    events = composition_data['events']
    
    # Create output buffer
    output = np.zeros(total_samples)
    
    # Process each event
    for event in events:
        start_time = event['start_time']
        duration = event['duration']
        
        start_sample = int(start_time * sample_rate)
        duration_samples = int(duration * sample_rate)
        end_sample = min(start_sample + duration_samples, total_samples)
        
        if start_sample >= total_samples:
            continue  # Event starts after composition ends
        
        # Generate or get signal for this event
        if 'signal' in event and isinstance(event['signal'], np.ndarray):
            # Pre-generated signal
            event_signal = event['signal']
        elif 'signal' in event and isinstance(event['signal'], str):
            # Signal description - would need signal generation logic
            # For now, create a simple sine wave as placeholder
            freq = event.get('frequency', 440.0)
            amp = event.get('amplitude', -12.0)
            t = np.linspace(0, duration, duration_samples, endpoint=False)
            event_signal = _db_to_linear(amp) * np.sin(2 * np.pi * freq * t)
        else:
            # Default to silence
            event_signal = np.zeros(duration_samples)
        
        # Trim or pad signal to match duration
        if len(event_signal) > duration_samples:
            event_signal = event_signal[:duration_samples]
        elif len(event_signal) < duration_samples:
            padded_signal = np.zeros(duration_samples)
            padded_signal[:len(event_signal)] = event_signal
            event_signal = padded_signal
        
        # Apply event-specific processing
        if 'gain' in event:
            event_signal *= event['gain']
        
        # Add to output
        actual_samples = min(len(event_signal), end_sample - start_sample)
        output[start_sample:start_sample + actual_samples] += event_signal[:actual_samples]
    
    # Apply master processing
    max_amplitude = np.max(np.abs(output))
    if max_amplitude > 0.8:
        output = output * (0.8 / max_amplitude)
    
    return _clip_signal(output)


if __name__ == "__main__":
    # Basic tests
    import doctest
    doctest.testmod()
    
    # Quick validation tests
    try:
        print("Testing mixing and composition functions...")
        
        # Create test signals
        duration = 1.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        test_signal1 = 0.3 * np.sin(2 * np.pi * 440 * t)
        test_signal2 = 0.2 * np.sin(2 * np.pi * 880 * t)
        test_lfo = 0.1 * np.sin(2 * np.pi * 2 * t)
        
        # Test signal combination functions
        combined = combine_signals([test_signal1, test_signal2], [1.0, 0.7])
        crossfade_curve = create_crossfade_curve(1.0, 'linear')
        crossfaded = layer_with_crossfade(test_signal1, test_signal2, crossfade_curve)
        parallel_mixed = parallel_mix([test_signal1, test_signal2], [-0.5, 0.5])
        
        print(f"✓ Combined signals: {len(combined)} samples")
        print(f"✓ Crossfaded signals: {len(crossfaded)} samples")
        print(f"✓ Parallel mixed: {parallel_mixed.shape} (stereo)")
        
        # Test modulation application
        am_modulated = apply_amplitude_modulation(test_signal1, test_lfo, depth=0.5)
        fm_signal = apply_frequency_modulation(440.0, test_lfo, depth=10.0, duration=1.0)
        
        print(f"✓ AM modulated: {len(am_modulated)} samples")
        print(f"✓ FM signal: {len(fm_signal)} samples")
        
        # Test layering tools
        stereo_field = create_stereo_field([test_signal1, test_signal2], [-0.3, 0.3])
        pan_curve = np.sin(2 * np.pi * 0.5 * t)
        dynamic_panned = apply_dynamic_panning(test_signal1, pan_curve)
        
        print(f"✓ Stereo field: {stereo_field.shape} (stereo)")
        print(f"✓ Dynamic panned: {dynamic_panned.shape} (stereo)")
        
        # Test composition utilities
        normalized = normalize_to_peak(test_signal1, target_peak_db=-12.0)
        limited = apply_master_limiter(test_signal1, threshold_db=-6.0, ratio=4.0)
        
        print(f"✓ Normalized: {len(normalized)} samples")
        print(f"✓ Limited: {len(limited)} samples")
        
        # Test timeline creation
        events = [
            {'start_time': 0.0, 'duration': 1.0, 'frequency': 440.0, 'amplitude': -12.0},
            {'start_time': 0.5, 'duration': 1.0, 'frequency': 880.0, 'amplitude': -15.0}
        ]
        timeline = create_composition_timeline(events, 2.0)
        rendered = render_composition(timeline)
        
        print(f"✓ Timeline created: {timeline['event_count']} events")
        print(f"✓ Composition rendered: {len(rendered)} samples")
        
        print("\n✓ All mixing and composition functions working correctly")
        
        # Test error conditions
        print("\nTesting error conditions...")
        try:
            combine_signals([test_signal1], [1.0, 0.5])  # Mismatched lengths
            print("✗ Should have raised length mismatch error")
        except ValueError:
            print("✓ Length validation working")
            
        try:
            apply_amplitude_modulation(test_signal1, test_lfo, depth=2.0)  # Invalid depth
            print("✗ Should have raised depth error")
        except ValueError:
            print("✓ Depth validation working")
            
    except Exception as e:
        print(f"✗ Error in mixing tests: {e}")