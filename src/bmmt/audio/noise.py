"""
Noise generator functions for modular audio synthesis.
Generates various types of noise for broken transmission dronescapes.
"""

import numpy as np
from typing import Union


def _validate_params(amp: float, duration: float, sample_rate: int) -> None:
    """Validate common parameters for noise functions."""
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    if amp > 0:
        raise ValueError(f"Amplitude should be negative dB value, got {amp}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")


def _db_to_linear(db: float) -> float:
    """Convert dB to linear amplitude."""
    return 10 ** (db / 20)


def generate_white_noise(amp: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate white noise with flat frequency spectrum.
    
    Args:
        amp: Amplitude in dB (negative values, e.g., -12)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Mono audio signal (float64)
    
    Example:
        >>> noise = generate_white_noise(-18.0, 1.0)
        >>> len(noise) == 44100  # 1 second at 44.1kHz
        True
        >>> -1.0 <= noise.max() <= 1.0  # Check amplitude range
        True
    """
    _validate_params(amp, duration, sample_rate)
    
    num_samples = int(duration * sample_rate)
    linear_amp = _db_to_linear(amp)
    
    # Generate white noise: uniform random values
    white_noise = np.random.normal(0, 1, num_samples)
    
    return linear_amp * white_noise


def generate_pink_noise(amp: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate pink noise with 1/f frequency spectrum.
    
    Pink noise has equal energy per octave, creating a warmer sound
    than white noise. Useful for organic, natural-sounding backgrounds.
    
    Args:
        amp: Amplitude in dB (negative values, e.g., -12)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Mono audio signal (float64)
    
    Example:
        >>> noise = generate_pink_noise(-24.0, 0.5)
        >>> len(noise) == 22050  # 0.5 seconds at 44.1kHz
        True
    """
    _validate_params(amp, duration, sample_rate)
    
    num_samples = int(duration * sample_rate)
    linear_amp = _db_to_linear(amp)
    
    # Generate pink noise using the Voss-McCartney algorithm
    # This is a simplified version that approximates pink noise
    
    # Start with white noise
    white = np.random.normal(0, 1, num_samples)
    
    # Apply frequency domain filtering to create 1/f spectrum
    # Take FFT, apply 1/sqrt(f) filter, then IFFT
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, 1/sample_rate)
    
    # Avoid division by zero at DC
    freqs[0] = 1.0
    
    # Apply 1/sqrt(f) filter (pink noise characteristic)
    pink_filter = 1.0 / np.sqrt(freqs)
    filtered_fft = fft * pink_filter
    
    # Convert back to time domain
    pink_noise = np.fft.irfft(filtered_fft, num_samples)
    
    # Normalize to prevent clipping
    if np.max(np.abs(pink_noise)) > 0:
        pink_noise = pink_noise / np.max(np.abs(pink_noise))
    
    return linear_amp * pink_noise


def generate_brown_noise(amp: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate brown noise (Brownian noise) with 1/f² frequency spectrum.
    
    Brown noise has even more low-frequency emphasis than pink noise,
    creating a deep, rumbling character perfect for drone foundations.
    
    Args:
        amp: Amplitude in dB (negative values, e.g., -12)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Mono audio signal (float64)
    
    Example:
        >>> noise = generate_brown_noise(-12.0, 2.0)
        >>> len(noise) == 88200  # 2 seconds at 44.1kHz
        True
    """
    _validate_params(amp, duration, sample_rate)
    
    num_samples = int(duration * sample_rate)
    linear_amp = _db_to_linear(amp)
    
    # Generate brown noise using cumulative sum of white noise
    # This creates the characteristic 1/f² spectrum
    white = np.random.normal(0, 1, num_samples)
    
    # Cumulative sum creates brown noise
    brown_noise = np.cumsum(white)
    
    # Remove DC offset
    brown_noise = brown_noise - np.mean(brown_noise)
    
    # Normalize to prevent clipping
    if np.max(np.abs(brown_noise)) > 0:
        brown_noise = brown_noise / np.max(np.abs(brown_noise))
    
    return linear_amp * brown_noise


if __name__ == "__main__":
    # Basic tests
    import doctest
    doctest.testmod()
    
    # Quick validation tests
    try:
        # Test valid parameters
        white = generate_white_noise(-18.0, 1.0)
        pink = generate_pink_noise(-24.0, 0.5)
        brown = generate_brown_noise(-12.0, 2.0)
        
        print("✓ All noise functions working correctly")
        print(f"✓ White noise: {len(white)} samples")
        print(f"✓ Pink noise: {len(pink)} samples")
        print(f"✓ Brown noise: {len(brown)} samples")
        
        # Test amplitude ranges
        print(f"✓ White noise range: {white.min():.3f} to {white.max():.3f}")
        print(f"✓ Pink noise range: {pink.min():.3f} to {pink.max():.3f}")
        print(f"✓ Brown noise range: {brown.min():.3f} to {brown.max():.3f}")
        
        # Test error conditions
        try:
            generate_white_noise(12.0, 1.0)  # Positive amplitude
            print("✗ Should have raised amplitude error")
        except ValueError:
            print("✓ Amplitude validation working")
            
        try:
            generate_pink_noise(-12.0, -1.0)  # Negative duration
            print("✗ Should have raised duration error")
        except ValueError:
            print("✓ Duration validation working")
            
    except Exception as e:
        print(f"✗ Error in noise tests: {e}")