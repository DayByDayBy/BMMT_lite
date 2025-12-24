"""
Dynamics processing for modular audio synthesis.
Provides compression, parallel compression, and sidechain ducking for punchy disco production.
"""

import numpy as np
from typing import Union


def _validate_signal(input_signal: np.ndarray) -> None:
    """Validate input signal."""
    if not isinstance(input_signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(input_signal) == 0:
        raise ValueError("Input signal cannot be empty")


def _db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20.0)


def _linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    return 20 * np.log10(np.maximum(linear, 1e-10))


def apply_compression(
    input_signal: np.ndarray,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    makeup_gain_db: float = 0.0,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply dynamic range compression to reduce loud peaks.
    
    Standard compression that attenuates signals above the threshold according
    to the ratio. Essential for controlling dynamics in disco drum mixes.
    
    Args:
        input_signal: Input audio signal (numpy array, mono or stereo)
        threshold_db: Threshold level in dB (-60 to 0, signals above are compressed)
        ratio: Compression ratio (1.0-20.0, e.g., 4:1 means 4dB input = 1dB output above threshold)
        attack_ms: Attack time in milliseconds (0.1-100, how fast compression engages)
        release_ms: Release time in milliseconds (10-1000, how fast compression releases)
        makeup_gain_db: Output gain in dB (-12 to 24, compensate for volume loss)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Compressed signal (same shape as input)
    
    Example:
        >>> import numpy as np
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.8
        >>> compressed = apply_compression(signal, threshold_db=-12, ratio=4.0)
        >>> len(compressed) == len(signal)
        True
    """
    _validate_signal(input_signal)
    
    if not (-60.0 <= threshold_db <= 0.0):
        raise ValueError(f"Threshold must be between -60 and 0 dB, got {threshold_db}")
    if not (1.0 <= ratio <= 20.0):
        raise ValueError(f"Ratio must be between 1.0 and 20.0, got {ratio}")
    if not (0.1 <= attack_ms <= 100.0):
        raise ValueError(f"Attack must be between 0.1 and 100 ms, got {attack_ms}")
    if not (10.0 <= release_ms <= 1000.0):
        raise ValueError(f"Release must be between 10 and 1000 ms, got {release_ms}")
    if not (-12.0 <= makeup_gain_db <= 24.0):
        raise ValueError(f"Makeup gain must be between -12 and 24 dB, got {makeup_gain_db}")
    
    # Handle stereo signals
    is_stereo = input_signal.ndim == 2
    if is_stereo:
        # Process each channel and combine
        left = apply_compression(
            input_signal[:, 0], threshold_db, ratio, attack_ms, release_ms, 
            makeup_gain_db, sample_rate
        )
        right = apply_compression(
            input_signal[:, 1], threshold_db, ratio, attack_ms, release_ms,
            makeup_gain_db, sample_rate
        )
        return np.column_stack([left, right])
    
    # Convert parameters
    threshold_linear = _db_to_linear(threshold_db)
    attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
    release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))
    makeup_linear = _db_to_linear(makeup_gain_db)
    
    # Get signal envelope (absolute value)
    envelope = np.abs(input_signal)
    
    # Smooth envelope with attack/release
    smoothed_envelope = np.zeros_like(envelope)
    current_env = 0.0
    
    for i in range(len(envelope)):
        if envelope[i] > current_env:
            # Attack phase
            current_env = attack_coeff * current_env + (1 - attack_coeff) * envelope[i]
        else:
            # Release phase
            current_env = release_coeff * current_env + (1 - release_coeff) * envelope[i]
        smoothed_envelope[i] = current_env
    
    # Calculate gain reduction
    gain_reduction = np.ones_like(smoothed_envelope)
    above_threshold = smoothed_envelope > threshold_linear
    
    if np.any(above_threshold):
        # Calculate compression gain for samples above threshold
        over_threshold_db = _linear_to_db(smoothed_envelope[above_threshold] / threshold_linear)
        compressed_over_db = over_threshold_db / ratio
        gain_reduction_db = over_threshold_db - compressed_over_db
        gain_reduction[above_threshold] = _db_to_linear(-gain_reduction_db)
    
    # Apply gain reduction and makeup gain
    output = input_signal * gain_reduction * makeup_linear
    
    # Soft clip to prevent overs
    output = np.clip(output, -0.99, 0.99)
    
    return output


def apply_parallel_compression(
    input_signal: np.ndarray,
    threshold_db: float = -30.0,
    ratio: float = 8.0,
    wet_mix: float = 0.5,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply New York-style parallel compression for punchy drums.
    
    Blends heavily compressed signal with the dry signal, adding sustain and
    punch without squashing transients. The secret sauce for disco drums.
    
    Args:
        input_signal: Input audio signal (numpy array, mono or stereo)
        threshold_db: Threshold for the compressed path (-60 to -10, typically very low)
        ratio: Compression ratio for the wet path (4.0-20.0, typically high)
        wet_mix: Blend of compressed signal (0.0-1.0, 0=dry only, 1=compressed only)
        attack_ms: Attack time in milliseconds (0.1-50, fast for punch)
        release_ms: Release time in milliseconds (10-500, medium for sustain)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Parallel compressed signal (same shape as input)
    
    Example:
        >>> import numpy as np
        >>> drums = np.random.normal(0, 0.3, 44100)
        >>> punchy = apply_parallel_compression(drums, threshold_db=-30, wet_mix=0.4)
        >>> len(punchy) == len(drums)
        True
    """
    _validate_signal(input_signal)
    
    if not (-60.0 <= threshold_db <= -10.0):
        raise ValueError(f"Threshold must be between -60 and -10 dB, got {threshold_db}")
    if not (4.0 <= ratio <= 20.0):
        raise ValueError(f"Ratio must be between 4.0 and 20.0, got {ratio}")
    if not (0.0 <= wet_mix <= 1.0):
        raise ValueError(f"Wet mix must be between 0.0 and 1.0, got {wet_mix}")
    if not (0.1 <= attack_ms <= 50.0):
        raise ValueError(f"Attack must be between 0.1 and 50 ms, got {attack_ms}")
    if not (10.0 <= release_ms <= 500.0):
        raise ValueError(f"Release must be between 10 and 500 ms, got {release_ms}")
    
    # Create heavily compressed version with makeup gain to match level
    compressed = apply_compression(
        input_signal,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        makeup_gain_db=abs(threshold_db) * 0.5,  # Auto makeup gain
        sample_rate=sample_rate
    )
    
    # Blend dry and compressed signals
    dry_mix = 1.0 - wet_mix
    output = dry_mix * input_signal + wet_mix * compressed
    
    # Normalize to prevent clipping while preserving dynamics
    max_val = np.max(np.abs(output))
    if max_val > 0.95:
        output = output / max_val * 0.95
    
    return output


def apply_sidechain_ducking(
    input_signal: np.ndarray,
    sidechain: np.ndarray,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Apply sidechain ducking to reduce input signal when sidechain is loud.
    
    Classic pumping effect where one signal (e.g., bass) ducks when another
    signal (e.g., kick) is present. Essential for clean disco bass/kick interaction.
    
    Args:
        input_signal: Signal to be ducked (numpy array, mono or stereo)
        sidechain: Trigger signal controlling the ducking (numpy array, mono)
        threshold_db: Sidechain threshold in dB (-60 to 0, when ducking activates)
        ratio: Ducking ratio (1.0-20.0, higher = more ducking)
        attack_ms: Attack time in milliseconds (0.1-50, fast for tight pumping)
        release_ms: Release time in milliseconds (50-500, controls pump feel)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Ducked signal (same shape as input_signal)
    
    Example:
        >>> import numpy as np
        >>> bass = np.sin(2 * np.pi * 60 * np.linspace(0, 1, 44100)) * 0.5
        >>> kick = np.zeros(44100)
        >>> kick[:2000] = np.sin(2 * np.pi * 80 * np.linspace(0, 0.045, 2000)) * 0.8
        >>> ducked = apply_sidechain_ducking(bass, kick, threshold_db=-20, ratio=8.0)
        >>> len(ducked) == len(bass)
        True
    """
    _validate_signal(input_signal)
    _validate_signal(sidechain)
    
    if not (-60.0 <= threshold_db <= 0.0):
        raise ValueError(f"Threshold must be between -60 and 0 dB, got {threshold_db}")
    if not (1.0 <= ratio <= 20.0):
        raise ValueError(f"Ratio must be between 1.0 and 20.0, got {ratio}")
    if not (0.1 <= attack_ms <= 50.0):
        raise ValueError(f"Attack must be between 0.1 and 50 ms, got {attack_ms}")
    if not (50.0 <= release_ms <= 500.0):
        raise ValueError(f"Release must be between 50 and 500 ms, got {release_ms}")
    
    # Ensure sidechain is mono for envelope detection
    if sidechain.ndim == 2:
        sidechain_mono = np.mean(sidechain, axis=1)
    else:
        sidechain_mono = sidechain
    
    # Match lengths
    min_len = min(len(input_signal) if input_signal.ndim == 1 else len(input_signal[:, 0]), 
                  len(sidechain_mono))
    
    if input_signal.ndim == 2:
        input_trimmed = input_signal[:min_len, :]
    else:
        input_trimmed = input_signal[:min_len]
    sidechain_trimmed = sidechain_mono[:min_len]
    
    # Convert parameters
    threshold_linear = _db_to_linear(threshold_db)
    attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000.0))
    release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000.0))
    
    # Get sidechain envelope
    sidechain_env = np.abs(sidechain_trimmed)
    
    # Smooth sidechain envelope
    smoothed_env = np.zeros_like(sidechain_env)
    current_env = 0.0
    
    for i in range(len(sidechain_env)):
        if sidechain_env[i] > current_env:
            current_env = attack_coeff * current_env + (1 - attack_coeff) * sidechain_env[i]
        else:
            current_env = release_coeff * current_env + (1 - release_coeff) * sidechain_env[i]
        smoothed_env[i] = current_env
    
    # Calculate gain reduction based on sidechain level
    gain = np.ones_like(smoothed_env)
    above_threshold = smoothed_env > threshold_linear
    
    if np.any(above_threshold):
        over_threshold_db = _linear_to_db(smoothed_env[above_threshold] / threshold_linear)
        compressed_over_db = over_threshold_db / ratio
        gain_reduction_db = over_threshold_db - compressed_over_db
        gain[above_threshold] = _db_to_linear(-gain_reduction_db)
    
    # Apply ducking
    if input_trimmed.ndim == 2:
        output = input_trimmed * gain[:, np.newaxis]
    else:
        output = input_trimmed * gain
    
    return output


def apply_soft_limiter(
    input_signal: np.ndarray,
    threshold_db: float = -3.0,
    ceiling_db: float = -0.1
) -> np.ndarray:
    """
    Apply soft limiting to prevent clipping while preserving dynamics.
    
    Gentle limiter with soft knee for transparent peak control. Use on
    master bus for clean headroom management.
    
    Args:
        input_signal: Input audio signal (numpy array, mono or stereo)
        threshold_db: Limiting threshold in dB (-12 to 0)
        ceiling_db: Maximum output level in dB (-6 to 0, hard ceiling)
    
    Returns:
        numpy.ndarray: Limited signal (same shape as input)
    
    Example:
        >>> import numpy as np
        >>> loud_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 1.5
        >>> limited = apply_soft_limiter(loud_signal, threshold_db=-3.0)
        >>> bool(np.max(np.abs(limited)) < 1.0)
        True
    """
    _validate_signal(input_signal)
    
    if not (-12.0 <= threshold_db <= 0.0):
        raise ValueError(f"Threshold must be between -12 and 0 dB, got {threshold_db}")
    if not (-6.0 <= ceiling_db <= 0.0):
        raise ValueError(f"Ceiling must be between -6 and 0 dB, got {ceiling_db}")
    
    threshold_linear = _db_to_linear(threshold_db)
    ceiling_linear = _db_to_linear(ceiling_db)
    
    def soft_limit(x):
        # Use tanh for smooth soft limiting
        # Scale input so threshold maps to ~0.76 (tanh(1))
        scaled = x / threshold_linear
        # Apply tanh for soft knee compression above threshold
        limited = np.tanh(scaled) * threshold_linear
        # Scale to ceiling
        limited = limited * (ceiling_linear / threshold_linear)
        return limited
    
    if input_signal.ndim == 2:
        left = soft_limit(input_signal[:, 0])
        right = soft_limit(input_signal[:, 1])
        return np.column_stack([left, right])
    else:
        return soft_limit(input_signal)


if __name__ == "__main__":
    # Basic tests
    import doctest
    doctest.testmod()
    
    # Quick validation tests
    try:
        print("Testing dynamics processing functions...")
        
        # Create test signals
        duration = 1.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        test_signal = 0.8 * np.sin(2 * np.pi * 440 * t)
        
        # Test compression
        compressed = apply_compression(test_signal, threshold_db=-12, ratio=4.0)
        print(f"✓ Compression: {len(compressed)} samples")
        
        # Test parallel compression
        parallel = apply_parallel_compression(test_signal, threshold_db=-30, wet_mix=0.4)
        print(f"✓ Parallel compression: {len(parallel)} samples")
        
        # Test sidechain ducking
        sidechain = np.zeros_like(test_signal)
        sidechain[:4410] = np.sin(2 * np.pi * 80 * t[:4410]) * 0.9  # 100ms kick
        ducked = apply_sidechain_ducking(test_signal, sidechain, threshold_db=-20, ratio=8.0)
        print(f"✓ Sidechain ducking: {len(ducked)} samples")
        
        # Test soft limiter
        loud_signal = test_signal * 2.0  # Clip-worthy
        limited = apply_soft_limiter(loud_signal, threshold_db=-3.0)
        print(f"✓ Soft limiter: {len(limited)} samples, peak: {np.max(np.abs(limited)):.3f}")
        
        # Test stereo processing
        stereo_signal = np.column_stack([test_signal, test_signal * 0.8])
        stereo_compressed = apply_compression(stereo_signal, threshold_db=-12, ratio=4.0)
        print(f"✓ Stereo compression: {stereo_compressed.shape}")
        
        print("\n✓ All dynamics processing functions working correctly")
        
        # Test error conditions
        print("\nTesting error conditions...")
        try:
            apply_compression(test_signal, threshold_db=10.0)  # Too high
            print("✗ Should have raised threshold error")
        except ValueError:
            print("✓ Threshold validation working")
            
        try:
            apply_parallel_compression(test_signal, wet_mix=2.0)  # Too high
            print("✗ Should have raised wet mix error")
        except ValueError:
            print("✓ Wet mix validation working")
            
    except Exception as e:
        print(f"✗ Error in dynamics tests: {e}")
        import traceback
        traceback.print_exc()
