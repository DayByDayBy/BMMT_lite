"""
Filter functions for modular audio synthesis.
Provides frequency shaping for broken transmission dronescapes.
"""

import numpy as np
from scipy import signal
from typing import Union


def _validate_filter_params(input_signal: np.ndarray, cutoff_freq: float, sample_rate: int) -> None:
    """Validate common parameters for filter functions."""
    if not isinstance(input_signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(input_signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if cutoff_freq <= 0:
        raise ValueError(f"Cutoff frequency must be positive, got {cutoff_freq}")
    if cutoff_freq >= sample_rate / 2:
        raise ValueError(f"Cutoff frequency {cutoff_freq}Hz must be less than Nyquist frequency {sample_rate/2}Hz")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")


def lowpass_filter(input_signal: np.ndarray, cutoff_freq: float, sample_rate: int = 44100, resonance: float = 1.0) -> np.ndarray:
    """
    Apply a lowpass filter to remove high frequencies.
    
    Useful for creating warm, muffled tones and removing harsh high-frequency content
    from oscillators and noise sources.
    
    Args:
        input_signal: Input audio signal (numpy array)
        cutoff_freq: Cutoff frequency in Hz (positive, < sample_rate/2)
        sample_rate: Sample rate in Hz (default 44100)
        resonance: Filter resonance/Q factor (default 1.0, higher = more resonant)
    
    Returns:
        numpy.ndarray: Filtered audio signal (same length as input)
    
    Example:
        >>> import numpy as np
        >>> signal = np.random.normal(0, 0.1, 1000)
        >>> filtered = lowpass_filter(signal, 1000.0)
        >>> len(filtered) == len(signal)
        True
    """
    _validate_filter_params(input_signal, cutoff_freq, sample_rate)
    
    if resonance <= 0:
        raise ValueError(f"Resonance must be positive, got {resonance}")
    
    # Calculate normalized frequency (0 to 1, where 1 is Nyquist)
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Design Butterworth lowpass filter
    # Order 2 provides good balance between rolloff and phase response
    order = 2
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Apply resonance by adjusting the filter coefficients
    if resonance > 1.0:
        # Increase resonance by modifying the denominator coefficients
        # This is a simplified approach - more complex resonance would require different filter designs
        a[1] = a[1] / resonance
    
    # Apply the filter
    filtered_signal = signal.filtfilt(b, a, input_signal)
    
    return filtered_signal


def highpass_filter(input_signal: np.ndarray, cutoff_freq: float, sample_rate: int = 44100, resonance: float = 1.0) -> np.ndarray:
    """
    Apply a highpass filter to remove low frequencies.
    
    Useful for removing DC offset, reducing rumble, or creating thinner tones
    by emphasizing higher frequency content.
    
    Args:
        input_signal: Input audio signal (numpy array)
        cutoff_freq: Cutoff frequency in Hz (positive, < sample_rate/2)
        sample_rate: Sample rate in Hz (default 44100)
        resonance: Filter resonance/Q factor (default 1.0, higher = more resonant)
    
    Returns:
        numpy.ndarray: Filtered audio signal (same length as input)
    
    Example:
        >>> import numpy as np
        >>> signal = np.random.normal(0, 0.1, 1000)
        >>> filtered = highpass_filter(signal, 100.0)
        >>> len(filtered) == len(signal)
        True
    """
    _validate_filter_params(input_signal, cutoff_freq, sample_rate)
    
    if resonance <= 0:
        raise ValueError(f"Resonance must be positive, got {resonance}")
    
    # Calculate normalized frequency
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Design Butterworth highpass filter
    order = 2
    b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
    
    # Apply resonance
    if resonance > 1.0:
        a[1] = a[1] / resonance
    
    # Apply the filter
    filtered_signal = signal.filtfilt(b, a, input_signal)
    
    return filtered_signal


def bandpass_filter(input_signal: np.ndarray, center_freq: float, bandwidth: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply a bandpass filter to isolate a frequency range.
    
    Perfect for creating focused frequency bands, isolating specific harmonics,
    or creating telephone/radio-like filtering effects.
    
    Args:
        input_signal: Input audio signal (numpy array)
        center_freq: Center frequency in Hz (positive, < sample_rate/2)
        bandwidth: Bandwidth in Hz (positive, determines filter width)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Filtered audio signal (same length as input)
    
    Example:
        >>> import numpy as np
        >>> signal = np.random.normal(0, 0.1, 1000)
        >>> filtered = bandpass_filter(signal, 1000.0, 200.0)
        >>> len(filtered) == len(signal)
        True
    """
    if not isinstance(input_signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(input_signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if center_freq <= 0:
        raise ValueError(f"Center frequency must be positive, got {center_freq}")
    if bandwidth <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    # Calculate low and high cutoff frequencies
    low_freq = center_freq - bandwidth / 2
    high_freq = center_freq + bandwidth / 2
    
    # Ensure frequencies are within valid range
    nyquist = sample_rate / 2
    if low_freq <= 0:
        low_freq = 1.0  # Minimum frequency
    if high_freq >= nyquist:
        high_freq = nyquist * 0.99  # Just below Nyquist
    
    # Calculate normalized frequencies
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist
    
    # Design Butterworth bandpass filter
    order = 2
    b, a = signal.butter(order, [low_normalized, high_normalized], btype='band', analog=False)
    
    # Apply the filter
    filtered_signal = signal.filtfilt(b, a, input_signal)
    
    return filtered_signal


if __name__ == "__main__":
    # Basic tests
    import doctest
    doctest.testmod()
    
    # Quick validation tests
    try:
        # Create test signal
        test_signal = np.random.normal(0, 0.1, 44100)  # 1 second of noise
        
        # Test all filters
        lowpass = lowpass_filter(test_signal, 1000.0)
        highpass = highpass_filter(test_signal, 100.0)
        bandpass = bandpass_filter(test_signal, 500.0, 200.0)
        
        print("✓ All filter functions working correctly")
        print(f"✓ Lowpass filter: {len(lowpass)} samples")
        print(f"✓ Highpass filter: {len(highpass)} samples")
        print(f"✓ Bandpass filter: {len(bandpass)} samples")
        
        # Test with resonance
        resonant_lowpass = lowpass_filter(test_signal, 1000.0, resonance=2.0)
        resonant_highpass = highpass_filter(test_signal, 100.0, resonance=1.5)
        
        print(f"✓ Resonant lowpass: {len(resonant_lowpass)} samples")
        print(f"✓ Resonant highpass: {len(resonant_highpass)} samples")
        
        # Test error conditions
        try:
            lowpass_filter(test_signal, 25000.0)  # Freq too high
            print("✗ Should have raised frequency error")
        except ValueError:
            print("✓ Frequency validation working")
            
        try:
            bandpass_filter(test_signal, 1000.0, -100.0)  # Negative bandwidth
            print("✗ Should have raised bandwidth error")
        except ValueError:
            print("✓ Bandwidth validation working")
            
        try:
            highpass_filter(np.array([]), 100.0)  # Empty signal
            print("✗ Should have raised empty signal error")
        except ValueError:
            print("✓ Empty signal validation working")
            
    except Exception as e:
        print(f"✗ Error in filter tests: {e}")