"""
Core oscillator functions for modular audio synthesis.
Generates basic waveforms for broken transmission dronescapes.
"""

import numpy as np
from typing import Union


def _validate_params(freq: float, amp: float, duration: float, sample_rate: int) -> None:
    """Validate common parameters for oscillator functions."""
    if not (20 <= freq <= 20000):
        raise ValueError(f"Frequency {freq}Hz out of range (20-20000Hz)")
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    if amp > 0:
        raise ValueError(f"Amplitude should be negative dB value, got {amp}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")


def _db_to_linear(db: float) -> float:
    """Convert dB to linear amplitude."""
    return 10 ** (db / 20)


def generate_sine(freq: float, amp: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a sine wave.
    
    Args:
        freq: Frequency in Hz (20-20000)
        amp: Amplitude in dB (negative values, e.g., -12)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Mono audio signal (float64)
    
    Example:
        >>> signal = generate_sine(440.0, -12.0, 1.0)
        >>> len(signal) == 44100  # 1 second at 44.1kHz
        True
    """
    _validate_params(freq, amp, duration, sample_rate)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    linear_amp = _db_to_linear(amp)
    
    return linear_amp * np.sin(2 * np.pi * freq * t)


def generate_triangle(freq: float, amp: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a triangle wave.
    
    Args:
        freq: Frequency in Hz (20-20000)
        amp: Amplitude in dB (negative values, e.g., -12)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Mono audio signal (float64)
    
    Example:
        >>> signal = generate_triangle(220.0, -18.0, 0.5)
        >>> len(signal) == 22050  # 0.5 seconds at 44.1kHz
        True
    """
    _validate_params(freq, amp, duration, sample_rate)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    linear_amp = _db_to_linear(amp)
    
    # Triangle wave using sawtooth and absolute value
    phase = 2 * np.pi * freq * t
    triangle = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
    
    return linear_amp * triangle


def generate_sawtooth(freq: float, amp: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a sawtooth wave.
    
    Args:
        freq: Frequency in Hz (20-20000)
        amp: Amplitude in dB (negative values, e.g., -12)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Mono audio signal (float64)
    
    Example:
        >>> signal = generate_sawtooth(110.0, -6.0, 2.0)
        >>> len(signal) == 88200  # 2 seconds at 44.1kHz
        True
    """
    _validate_params(freq, amp, duration, sample_rate)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    linear_amp = _db_to_linear(amp)
    
    # Sawtooth wave: ramp from -1 to 1
    phase = (freq * t) % 1.0
    sawtooth = 2 * phase - 1
    
    return linear_amp * sawtooth


def generate_square(freq: float, amp: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a square wave.
    
    Args:
        freq: Frequency in Hz (20-20000)
        amp: Amplitude in dB (negative values, e.g., -12)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Mono audio signal (float64)
    
    Example:
        >>> signal = generate_square(55.0, -24.0, 0.1)
        >>> len(signal) == 4410  # 0.1 seconds at 44.1kHz
        True
    """
    _validate_params(freq, amp, duration, sample_rate)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    linear_amp = _db_to_linear(amp)
    
    # Square wave: sign of sine wave
    square = np.sign(np.sin(2 * np.pi * freq * t))
    
    return linear_amp * square


if __name__ == "__main__":
    # Basic tests
    import doctest
    doctest.testmod()
    
    # Quick validation tests
    try:
        # Test valid parameters
        sine = generate_sine(440.0, -12.0, 1.0)
        triangle = generate_triangle(220.0, -18.0, 0.5)
        sawtooth = generate_sawtooth(110.0, -6.0, 2.0)
        square = generate_square(55.0, -24.0, 0.1)
        
        print("✓ All oscillator functions working correctly")
        print(f"✓ Sine wave: {len(sine)} samples")
        print(f"✓ Triangle wave: {len(triangle)} samples")
        print(f"✓ Sawtooth wave: {len(sawtooth)} samples")
        print(f"✓ Square wave: {len(square)} samples")
        
        # Test error conditions
        try:
            generate_sine(10.0, -12.0, 1.0)  # Freq too low
            print("✗ Should have raised frequency error")
        except ValueError:
            print("✓ Frequency validation working")
            
        try:
            generate_sine(440.0, 12.0, 1.0)  # Positive amplitude
            print("✗ Should have raised amplitude error")
        except ValueError:
            print("✓ Amplitude validation working")
            
    except Exception as e:
        print(f"✗ Error in oscillator tests: {e}")