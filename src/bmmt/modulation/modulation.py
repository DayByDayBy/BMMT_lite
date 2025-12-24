"""
Modulation system for modular audio synthesis.
Generates control data for parameter automation in broken transmission dronescapes.
"""

import numpy as np
from typing import Union, Optional


def _validate_modulation_params(duration: float, sample_rate: int) -> None:
    """Validate common parameters for modulation functions."""
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")


def _validate_lfo_params(freq: float, depth: float, duration: float, sample_rate: int, offset: float) -> None:
    """Validate parameters specific to LFO functions."""
    _validate_modulation_params(duration, sample_rate)
    if not (0.001 <= freq <= 20.0):
        raise ValueError(f"LFO frequency {freq}Hz out of range (0.001-20Hz)")
    if depth < 0:
        raise ValueError(f"Depth must be non-negative, got {depth}")


# LFO Generators - Low Frequency Oscillators for parameter automation

def generate_lfo_sine(freq: float, depth: float, duration: float, sample_rate: int = 44100, offset: float = 0.0) -> np.ndarray:
    """
    Generate a sine wave LFO for smooth parameter modulation.
    
    Args:
        freq: LFO frequency in Hz (0.001-20, typically 0.01-2 for drones)
        depth: Modulation depth (amplitude of the LFO)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
        offset: DC offset added to the LFO (default 0.0)
    
    Returns:
        numpy.ndarray: Control data array (float64)
    
    Example:
        >>> lfo = generate_lfo_sine(0.1, 5.0, 10.0)
        >>> len(lfo) == 441000  # 10 seconds at 44.1kHz
        True
        >>> np.abs(np.max(lfo) - 5.0) < 0.01  # Peak should be near depth
        True
    """
    _validate_lfo_params(freq, depth, duration, sample_rate, offset)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    return offset + depth * np.sin(2 * np.pi * freq * t)


def generate_lfo_triangle(freq: float, depth: float, duration: float, sample_rate: int = 44100, offset: float = 0.0) -> np.ndarray:
    """
    Generate a triangle wave LFO for linear parameter sweeps.
    
    Args:
        freq: LFO frequency in Hz (0.001-20, typically 0.01-2 for drones)
        depth: Modulation depth (amplitude of the LFO)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
        offset: DC offset added to the LFO (default 0.0)
    
    Returns:
        numpy.ndarray: Control data array (float64)
    
    Example:
        >>> lfo = generate_lfo_triangle(0.5, 2.0, 4.0)
        >>> len(lfo) == 176400  # 4 seconds at 44.1kHz
        True
    """
    _validate_lfo_params(freq, depth, duration, sample_rate, offset)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Triangle wave using sawtooth and absolute value
    phase = 2 * np.pi * freq * t
    triangle = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
    
    return offset + depth * triangle


def generate_lfo_sawtooth(freq: float, depth: float, duration: float, sample_rate: int = 44100, offset: float = 0.0) -> np.ndarray:
    """
    Generate a sawtooth wave LFO for ramping parameter changes.
    
    Args:
        freq: LFO frequency in Hz (0.001-20, typically 0.01-2 for drones)
        depth: Modulation depth (amplitude of the LFO)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
        offset: DC offset added to the LFO (default 0.0)
    
    Returns:
        numpy.ndarray: Control data array (float64)
    
    Example:
        >>> lfo = generate_lfo_sawtooth(0.25, 10.0, 8.0)
        >>> len(lfo) == 352800  # 8 seconds at 44.1kHz
        True
    """
    _validate_lfo_params(freq, depth, duration, sample_rate, offset)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Sawtooth wave: ramp from -1 to 1
    phase = (freq * t) % 1.0
    sawtooth = 2 * phase - 1
    
    return offset + depth * sawtooth


def generate_lfo_square(freq: float, depth: float, duration: float, sample_rate: int = 44100, offset: float = 0.0) -> np.ndarray:
    """
    Generate a square wave LFO for stepped parameter changes.
    
    Args:
        freq: LFO frequency in Hz (0.001-20, typically 0.01-2 for drones)
        depth: Modulation depth (amplitude of the LFO)
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
        offset: DC offset added to the LFO (default 0.0)
    
    Returns:
        numpy.ndarray: Control data array (float64)
    
    Example:
        >>> lfo = generate_lfo_square(1.0, 3.0, 2.0)
        >>> len(lfo) == 88200  # 2 seconds at 44.1kHz
        True
    """
    _validate_lfo_params(freq, depth, duration, sample_rate, offset)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Square wave: sign of sine wave
    square = np.sign(np.sin(2 * np.pi * freq * t))
    
    return offset + depth * square


# Random Walk Functions - For slow, organic drift

def generate_random_walk(duration: float, step_size: float = 0.01, sample_rate: int = 44100, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random walk for organic parameter drift.
    
    Args:
        duration: Duration in seconds (positive)
        step_size: Maximum step size per sample (default 0.01)
        sample_rate: Sample rate in Hz (default 44100)
        seed: Random seed for reproducible results (optional)
    
    Returns:
        numpy.ndarray: Control data array (float64)
    
    Example:
        >>> walk = generate_random_walk(5.0, step_size=0.1, seed=42)
        >>> len(walk) == 220500  # 5 seconds at 44.1kHz
        True
        >>> isinstance(walk[0], (float, np.floating))
        True
    """
    _validate_modulation_params(duration, sample_rate)
    if step_size <= 0:
        raise ValueError(f"Step size must be positive, got {step_size}")
    
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    
    # Generate random steps
    steps = np.random.uniform(-step_size, step_size, num_samples)
    
    # Cumulative sum to create random walk
    walk = np.cumsum(steps)
    
    return walk


def generate_perlin_drift(duration: float, frequency: float = 0.1, amplitude: float = 1.0, sample_rate: int = 44100, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate Perlin-like noise for smooth, organic parameter drift.
    Uses multiple octaves of sine waves with random phases for natural movement.
    
    Args:
        duration: Duration in seconds (positive)
        frequency: Base frequency for the drift (default 0.1Hz)
        amplitude: Amplitude of the drift (default 1.0)
        sample_rate: Sample rate in Hz (default 44100)
        seed: Random seed for reproducible results (optional)
    
    Returns:
        numpy.ndarray: Control data array (float64)
    
    Example:
        >>> drift = generate_perlin_drift(10.0, frequency=0.05, amplitude=2.0, seed=123)
        >>> len(drift) == 441000  # 10 seconds at 44.1kHz
        True
    """
    _validate_modulation_params(duration, sample_rate)
    if frequency <= 0:
        raise ValueError(f"Frequency must be positive, got {frequency}")
    if amplitude < 0:
        raise ValueError(f"Amplitude must be non-negative, got {amplitude}")
    
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Generate multiple octaves with random phases for organic feel
    drift = np.zeros(num_samples)
    octaves = 4
    
    for octave in range(octaves):
        octave_freq = frequency * (2 ** octave)
        octave_amp = amplitude / (2 ** octave)  # Decrease amplitude with frequency
        phase_offset = np.random.uniform(0, 2 * np.pi)
        
        drift += octave_amp * np.sin(2 * np.pi * octave_freq * t + phase_offset)
    
    return drift


# Envelope Generators - For shaping parameters over time

def generate_linear_envelope(start_val: float, end_val: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a linear envelope from start to end value.
    
    Args:
        start_val: Starting value
        end_val: Ending value
        duration: Duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Control data array (float64)
    
    Example:
        >>> env = generate_linear_envelope(-60.0, -12.0, 5.0)
        >>> len(env) == 220500  # 5 seconds at 44.1kHz
        True
        >>> abs(env[0] - (-60.0)) < 0.01  # First sample should be start_val
        True
        >>> abs(env[-1] - (-12.0)) < 0.01  # Last sample should be end_val
        True
    """
    _validate_modulation_params(duration, sample_rate)
    
    num_samples = int(duration * sample_rate)
    
    return np.linspace(start_val, end_val, num_samples)


def generate_exponential_envelope(start_val: float, end_val: float, duration: float, curve: float = 2.0, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate an exponential envelope from start to end value.
    
    Args:
        start_val: Starting value
        end_val: Ending value
        duration: Duration in seconds (positive)
        curve: Exponential curve factor (>1 for slow start, <1 for fast start)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Control data array (float64)
    
    Example:
        >>> env = generate_exponential_envelope(0.0, 1.0, 3.0, curve=3.0)
        >>> len(env) == 132300  # 3 seconds at 44.1kHz
        True
        >>> abs(env[0] - 0.0) < 0.01  # First sample should be start_val
        True
    """
    _validate_modulation_params(duration, sample_rate)
    if curve <= 0:
        raise ValueError(f"Curve must be positive, got {curve}")
    
    num_samples = int(duration * sample_rate)
    
    # Create exponential curve
    t = np.linspace(0, 1, num_samples)
    exp_curve = (np.power(curve, t) - 1) / (curve - 1) if curve != 1.0 else t
    
    # Scale to start and end values
    return start_val + (end_val - start_val) * exp_curve


def generate_adsr_envelope(attack: float, decay: float, sustain_level: float, release: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate an ADSR (Attack, Decay, Sustain, Release) envelope.
    
    Args:
        attack: Attack time in seconds (>= 0)
        decay: Decay time in seconds (>= 0)
        sustain_level: Sustain level (0.0 to 1.0)
        release: Release time in seconds (>= 0)
        duration: Total duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Control data array (float64)
    
    Example:
        >>> env = generate_adsr_envelope(0.1, 0.2, 0.7, 0.5, 2.0)
        >>> len(env) == 88200  # 2 seconds at 44.1kHz
        True
        >>> np.max(env) <= 1.0  # Peak should not exceed 1.0
        True
    """
    _validate_modulation_params(duration, sample_rate)
    if attack < 0 or decay < 0 or release < 0:
        raise ValueError("Attack, decay, and release times must be non-negative")
    if not (0.0 <= sustain_level <= 1.0):
        raise ValueError(f"Sustain level must be between 0.0 and 1.0, got {sustain_level}")
    if attack + decay + release > duration:
        raise ValueError("Sum of attack, decay, and release times exceeds total duration")
    
    num_samples = int(duration * sample_rate)
    envelope = np.zeros(num_samples)
    
    # Calculate sample indices for each phase
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = num_samples - attack_samples - decay_samples - release_samples
    
    current_idx = 0
    
    # Attack phase: 0 to 1
    if attack_samples > 0:
        envelope[current_idx:current_idx + attack_samples] = np.linspace(0, 1, attack_samples)
        current_idx += attack_samples
    
    # Decay phase: 1 to sustain_level
    if decay_samples > 0:
        envelope[current_idx:current_idx + decay_samples] = np.linspace(1, sustain_level, decay_samples)
        current_idx += decay_samples
    
    # Sustain phase: constant sustain_level
    if sustain_samples > 0:
        envelope[current_idx:current_idx + sustain_samples] = sustain_level
        current_idx += sustain_samples
    
    # Release phase: sustain_level to 0
    if release_samples > 0:
        envelope[current_idx:current_idx + release_samples] = np.linspace(sustain_level, 0, release_samples)
    
    return envelope


def generate_percussive_envelope(attack_ms: float, decay_ms: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a fast AD envelope optimized for percussive sounds.
    
    Args:
        attack_ms: Attack time in milliseconds (>= 0, typically 0-10 for drums)
        decay_ms: Decay time in milliseconds (>= 0, typically 50-500 for drums)
        duration: Total duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Amplitude envelope array (float64, 0.0 to 1.0)
    
    Example:
        >>> env = generate_percussive_envelope(5.0, 100.0, 0.2)
        >>> len(env) == 8820  # 0.2 seconds at 44.1kHz
        True
        >>> np.max(env) <= 1.0  # Peak should not exceed 1.0
        True
    """
    _validate_modulation_params(duration, sample_rate)
    if attack_ms < 0 or decay_ms < 0:
        raise ValueError("Attack and decay times must be non-negative")
    
    # Convert milliseconds to seconds
    attack = attack_ms / 1000.0
    decay = decay_ms / 1000.0
    
    if attack + decay > duration:
        raise ValueError("Sum of attack and decay times exceeds total duration")
    
    num_samples = int(duration * sample_rate)
    envelope = np.zeros(num_samples)
    
    # Calculate sample indices
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    
    # Attack phase: 0 to 1
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay phase: 1 to 0
    if decay_samples > 0:
        start_idx = attack_samples
        end_idx = start_idx + decay_samples
        envelope[start_idx:end_idx] = np.linspace(1, 0, decay_samples)
    
    # Set peak at the transition point
    if attack_samples > 0 and attack_samples < num_samples:
        envelope[attack_samples] = 1.0
    
    return envelope


def generate_pitch_envelope(start_freq: float, end_freq: float, time_ms: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a pitch sweep envelope for drum synthesis (e.g., kick drum pitch drops).
    
    Args:
        start_freq: Starting frequency in Hz (>= 20)
        end_freq: Ending frequency in Hz (>= 20)
        time_ms: Time to complete the sweep in milliseconds (>= 0)
        duration: Total duration in seconds (positive)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Frequency curve array (float64, in Hz)
    
    Example:
        >>> env = generate_pitch_envelope(150.0, 60.0, 10.0, 0.2)
        >>> len(env) == 8820  # 0.2 seconds at 44.1kHz
        True
        >>> env[0] >= env[-1]  # Should sweep from high to low
        True
    """
    _validate_modulation_params(duration, sample_rate)
    if not (20 <= start_freq <= 20000):
        raise ValueError(f"Start frequency {start_freq}Hz out of range (20-20000Hz)")
    if not (20 <= end_freq <= 20000):
        raise ValueError(f"End frequency {end_freq}Hz out of range (20-20000Hz)")
    if time_ms < 0:
        raise ValueError("Sweep time must be non-negative")
    
    # Convert milliseconds to seconds
    sweep_time = time_ms / 1000.0
    
    if sweep_time > duration:
        raise ValueError("Sweep time exceeds total duration")
    
    num_samples = int(duration * sample_rate)
    envelope = np.ones(num_samples) * end_freq  # Default to end frequency
    
    # Calculate sample indices
    sweep_samples = int(sweep_time * sample_rate)
    
    if sweep_samples > 0:
        # Create exponential sweep from start to end frequency
        # Using exponential curve for more natural pitch decay
        sweep_curve = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), sweep_samples))
        envelope[:sweep_samples] = sweep_curve
    
    return envelope


if __name__ == "__main__":
    # Basic tests
    import doctest
    doctest.testmod()
    
    # Quick validation tests
    try:
        print("Testing LFO generators...")
        lfo_sine = generate_lfo_sine(0.1, 5.0, 1.0)
        lfo_triangle = generate_lfo_triangle(0.2, 3.0, 1.0)
        lfo_sawtooth = generate_lfo_sawtooth(0.5, 2.0, 1.0)
        lfo_square = generate_lfo_square(1.0, 1.0, 1.0)
        print(f"✓ LFO Sine: {len(lfo_sine)} samples")
        print(f"✓ LFO Triangle: {len(lfo_triangle)} samples")
        print(f"✓ LFO Sawtooth: {len(lfo_sawtooth)} samples")
        print(f"✓ LFO Square: {len(lfo_square)} samples")
        
        print("\nTesting random walk functions...")
        random_walk = generate_random_walk(1.0, step_size=0.01, seed=42)
        perlin_drift = generate_perlin_drift(1.0, frequency=0.1, amplitude=1.0, seed=123)
        print(f"✓ Random Walk: {len(random_walk)} samples")
        print(f"✓ Perlin Drift: {len(perlin_drift)} samples")
        
        print("\nTesting envelope generators...")
        linear_env = generate_linear_envelope(-60.0, -12.0, 1.0)
        exp_env = generate_exponential_envelope(0.0, 1.0, 1.0, curve=2.0)
        adsr_env = generate_adsr_envelope(0.1, 0.2, 0.7, 0.3, 1.0)
        print(f"✓ Linear Envelope: {len(linear_env)} samples")
        print(f"✓ Exponential Envelope: {len(exp_env)} samples")
        print(f"✓ ADSR Envelope: {len(adsr_env)} samples")
        
        print("\nTesting percussive envelope functions...")
        perc_env = generate_percussive_envelope(5.0, 100.0, 0.2, 44100)
        pitch_env = generate_pitch_envelope(150.0, 60.0, 10.0, 0.2, 44100)
        print(f"✓ Percussive Envelope: {len(perc_env)} samples")
        print(f"✓ Pitch Envelope: {len(pitch_env)} samples")
        
        print("\n✓ All modulation functions working correctly")
        
        # Test error conditions
        print("\nTesting error conditions...")
        try:
            generate_lfo_sine(0.0001, 1.0, 1.0)  # Freq too low
            print("✗ Should have raised frequency error")
        except ValueError:
            print("✓ LFO frequency validation working")
            
        try:
            generate_adsr_envelope(0.5, 0.5, 0.5, 0.5, 1.0)  # Times exceed duration
            print("✗ Should have raised ADSR duration error")
        except ValueError:
            print("✓ ADSR duration validation working")
            
    except Exception as e:
        print(f"✗ Error in modulation tests: {e}")