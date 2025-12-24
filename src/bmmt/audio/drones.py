"""
Style-specific presets for modular audio synthesis.
High-level functions that combine all previous phases to create complete "broken transmission" compositions.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

# Import all modular components from previous phases
from .oscillators import generate_sine, generate_triangle, generate_sawtooth, generate_square
from .noise import generate_white_noise, generate_pink_noise, generate_brown_noise
from ..modulation.modulation import (
    generate_lfo_sine, generate_lfo_triangle, generate_lfo_square, generate_random_walk, 
    generate_perlin_drift, generate_linear_envelope, generate_exponential_envelope
)
from ..processing.effects import (
    apply_bitcrush, apply_wow_flutter, apply_dropout, add_static_bursts,
    apply_tape_saturation, apply_vinyl_crackle, apply_tube_warmth
)
from ..processing.spatial import (
    apply_reverb, apply_distance_filter, apply_stereo_width, 
    apply_air_absorption, apply_echo_delay
)
from ..processing.filters import lowpass_filter, highpass_filter, bandpass_filter
from ..composition.mixing import (
    combine_signals, layer_with_crossfade, parallel_mix, apply_amplitude_modulation,
    apply_frequency_modulation, create_crossfade_curve, normalize_to_peak,
    apply_master_limiter, create_stereo_field
)


def _validate_preset_params(duration: float, sample_rate: int, seed: Optional[int] = None) -> None:
    """Validate common preset parameters."""
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if seed is not None and not isinstance(seed, int):
        raise TypeError("Seed must be an integer or None")


def _set_random_seed(seed: Optional[int]) -> None:
    """Set random seed for reproducible results."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def _db_to_linear(db: float) -> float:
    """Convert dB to linear amplitude."""
    return 10 ** (db / 20)


def _get_complexity_params(complexity: str) -> Dict[str, float]:
    """Get parameter multipliers based on complexity level."""
    complexity_map = {
        'simple': {
            'layer_count': 0.5,
            'modulation_depth': 0.3,
            'effect_intensity': 0.2,
            'frequency_spread': 0.2
        },
        'medium': {
            'layer_count': 1.0,
            'modulation_depth': 0.5,
            'effect_intensity': 0.4,
            'frequency_spread': 0.3
        },
        'complex': {
            'layer_count': 1.5,
            'modulation_depth': 0.8,
            'effect_intensity': 0.6,
            'frequency_spread': 0.5
        }
    }
    
    if complexity not in complexity_map:
        raise ValueError(f"Complexity must be 'simple', 'medium', or 'complex', got '{complexity}'")
    
    return complexity_map[complexity]


# Core Preset Functions

def generate_broken_transmission(duration: float = 60, complexity: str = 'medium', 
                               seed: Optional[int] = None, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a complete broken transmission dronescape.
    
    Creates a complex, evolving drone that captures the essence of failing
    communication equipment with organic degradation and spatial depth.
    
    Args:
        duration: Duration in seconds (default 60)
        complexity: Complexity level ('simple', 'medium', 'complex')
        seed: Random seed for reproducible results (optional)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Stereo broken transmission composition
    """
    _validate_preset_params(duration, sample_rate, seed)
    _set_random_seed(seed)
    
    params = _get_complexity_params(complexity)
    
    # Base frequencies in the 40-200Hz range for broken transmission aesthetic
    base_freqs = [
        random.uniform(40, 80),   # Deep fundamental
        random.uniform(60, 120),  # Mid-low harmonic
        random.uniform(80, 160),  # Upper harmonic
        random.uniform(100, 200)  # Brightness layer
    ]
    
    layers = []
    layer_count = int(2 + params['layer_count'] * 3)  # 2-5 layers based on complexity
    
    for i in range(layer_count):
        freq = base_freqs[i % len(base_freqs)]
        
        # Create base oscillator with slight detuning
        detune = random.uniform(-0.5, 0.5) * params['frequency_spread']
        actual_freq = freq + detune
        
        # Choose oscillator type with bias toward warmer sounds
        osc_type = random.choices(
            ['sine', 'triangle', 'sawtooth'],
            weights=[0.5, 0.3, 0.2]
        )[0]
        
        amp = -18 - i * 3  # Each layer gets quieter
        
        if osc_type == 'sine':
            layer = generate_sine(actual_freq, amp, duration, sample_rate)
        elif osc_type == 'triangle':
            layer = generate_triangle(actual_freq, amp, duration, sample_rate)
        else:
            layer = generate_sawtooth(actual_freq, amp, duration, sample_rate)
        
        # Add organic modulation
        if random.random() < 0.7:  # 70% chance of modulation
            mod_freq = random.uniform(0.01, 0.1)
            mod_depth = params['modulation_depth'] * random.uniform(0.5, 2.0)
            
            if random.random() < 0.6:
                # Smooth LFO modulation
                lfo = generate_lfo_sine(mod_freq, mod_depth, duration, sample_rate)
            else:
                # Organic drift
                lfo = generate_perlin_drift(duration, mod_freq, mod_depth, sample_rate, seed)
            
            layer = apply_amplitude_modulation(layer, lfo, depth=0.3)
        
        # Apply transmission-style filtering
        if random.random() < 0.8:
            cutoff = random.uniform(200, 2000)
            layer = lowpass_filter(layer, cutoff, sample_rate)
        
        # Add degradation effects
        effect_intensity = params['effect_intensity']
        
        # Tape saturation for warmth
        if random.random() < 0.6:
            drive = 1.0 + effect_intensity * random.uniform(0.5, 2.0)
            warmth = effect_intensity * random.uniform(0.2, 0.5)
            layer = apply_tape_saturation(layer, drive, warmth)
        
        # Wow and flutter for analog character
        if random.random() < 0.5:
            intensity = effect_intensity * random.uniform(0.1, 0.3)
            flutter_freq = random.uniform(0.3, 1.5)
            layer = apply_wow_flutter(layer, intensity, flutter_freq, sample_rate)
        
        # Occasional dropouts
        if random.random() < 0.3:
            dropout_prob = effect_intensity * random.uniform(0.005, 0.02)
            layer = apply_dropout(layer, dropout_prob, 0.2, sample_rate)
        
        layers.append(layer)
    
    # Add noise layer for texture
    noise_amp = -30 - params['effect_intensity'] * 10
    if random.random() < 0.7:
        noise_type = random.choice(['pink', 'brown'])
        if noise_type == 'pink':
            noise = generate_pink_noise(noise_amp, duration, sample_rate)
        else:
            noise = generate_brown_noise(noise_amp, duration, sample_rate)
        
        # Filter noise to transmission band
        noise = bandpass_filter(noise, random.uniform(100, 800), random.uniform(200, 600), sample_rate)
        
        # Add static bursts
        if random.random() < 0.4:
            burst_freq = effect_intensity * random.uniform(0.05, 0.2)
            burst_intensity = effect_intensity * random.uniform(0.1, 0.3)
            noise = add_static_bursts(noise, burst_freq, burst_intensity, sample_rate)
        
        layers.append(noise)
    
    # Mix all layers
    mix_levels = [1.0] * len(layers)
    combined = combine_signals(layers, mix_levels, sample_rate)
    
    # Create stereo field with spatial processing
    stereo = create_stereo_field([combined], [0.0], width=1.2)
    
    # Add spatial depth
    if random.random() < 0.8:
        # Distance filtering
        distance = 1.0 + params['effect_intensity'] * random.uniform(1.0, 3.0)
        stereo = apply_distance_filter(stereo, distance, sample_rate)
    
    if random.random() < 0.6:
        # Subtle reverb
        room_size = 0.3 + params['effect_intensity'] * 0.4
        decay_time = 1.0 + params['effect_intensity'] * 2.0
        damping = 0.4 + params['effect_intensity'] * 0.3
        stereo = apply_reverb(stereo, room_size, decay_time, damping, sample_rate)
    
    # Master processing
    stereo = normalize_to_peak(stereo, -6.0)
    stereo = apply_master_limiter(stereo, -3.0, 4.0)
    
    return stereo


def generate_distant_radio(duration: float = 60, frequency_range: Tuple[float, float] = (40, 200),
                         interference_level: float = 0.3, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a distant radio transmission with interference.
    
    Creates the sound of a weak radio signal struggling through static
    and atmospheric interference.
    
    Args:
        duration: Duration in seconds (default 60)
        frequency_range: Frequency range tuple (low, high) in Hz
        interference_level: Interference intensity (0.0-1.0)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Stereo distant radio composition
    """
    _validate_preset_params(duration, sample_rate)
    
    if not (0.0 <= interference_level <= 1.0):
        raise ValueError(f"Interference level must be between 0.0 and 1.0, got {interference_level}")
    
    low_freq, high_freq = frequency_range
    if low_freq >= high_freq or low_freq <= 0:
        raise ValueError("Invalid frequency range")
    
    # Create carrier signal - the "radio station"
    carrier_freq = random.uniform(low_freq, high_freq)
    carrier = generate_sine(carrier_freq, -15, duration, sample_rate)
    
    # Add subtle frequency modulation for instability
    fm_lfo = generate_lfo_sine(random.uniform(0.05, 0.2), random.uniform(0.5, 2.0), duration, sample_rate)
    carrier = apply_amplitude_modulation(carrier, fm_lfo, depth=0.2)
    
    # Add harmonic content
    harmonic_freq = carrier_freq * random.uniform(1.5, 2.5)
    if harmonic_freq < sample_rate / 2:
        harmonic = generate_sine(harmonic_freq, -25, duration, sample_rate)
        carrier = combine_signals([carrier, harmonic], [1.0, 0.3], sample_rate)
    
    # Apply radio-style bandpass filtering
    center_freq = (low_freq + high_freq) / 2
    bandwidth = (high_freq - low_freq) * 0.8
    carrier = bandpass_filter(carrier, center_freq, bandwidth, sample_rate)
    
    # Add interference based on level
    interference_layers = []
    
    if interference_level > 0.1:
        # Static noise
        static_amp = -25 - (1.0 - interference_level) * 15
        static = generate_white_noise(static_amp, duration, sample_rate)
        static = highpass_filter(static, 1000, sample_rate)  # High-frequency static
        interference_layers.append(static)
    
    if interference_level > 0.3:
        # Atmospheric noise (pink noise)
        atmo_amp = -30 - (1.0 - interference_level) * 10
        atmospheric = generate_pink_noise(atmo_amp, duration, sample_rate)
        atmospheric = bandpass_filter(atmospheric, 200, 1000, sample_rate)
        interference_layers.append(atmospheric)
    
    if interference_level > 0.5:
        # Random dropouts
        dropout_prob = interference_level * 0.03
        carrier = apply_dropout(carrier, dropout_prob, 0.3, sample_rate)
        
        # Static bursts
        burst_freq = interference_level * 0.15
        burst_intensity = interference_level * 0.4
        carrier = add_static_bursts(carrier, burst_freq, burst_intensity, sample_rate)
    
    # Combine carrier with interference
    all_signals = [carrier] + interference_layers
    mix_levels = [1.0] + [0.5] * len(interference_layers)
    combined = combine_signals(all_signals, mix_levels, sample_rate)
    
    # Apply distance effects
    distance_factor = 2.0 + interference_level * 2.0
    combined = apply_distance_filter(combined, distance_factor, sample_rate)
    
    # Create stereo field
    stereo = create_stereo_field([combined], [0.0], width=0.8)
    
    # Add subtle echo for distance
    if interference_level > 0.2:
        delay_time = 0.1 + interference_level * 0.3
        feedback = interference_level * 0.2
        mix_level = interference_level * 0.3
        stereo = apply_echo_delay(stereo, delay_time, feedback, mix_level, sample_rate)
    
    # Master processing
    stereo = normalize_to_peak(stereo, -6.0)
    stereo = apply_master_limiter(stereo, -3.0, 6.0)
    
    return stereo


def generate_failing_technology(duration: float = 60, degradation_curve: str = 'exponential',
                              sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a composition that represents failing technology.
    
    Creates a drone that starts relatively clean and progressively
    degrades over time, simulating equipment failure.
    
    Args:
        duration: Duration in seconds (default 60)
        degradation_curve: Degradation type ('linear', 'exponential', 'sudden')
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Stereo failing technology composition
    """
    _validate_preset_params(duration, sample_rate)
    
    if degradation_curve not in ['linear', 'exponential', 'sudden']:
        raise ValueError(f"Degradation curve must be 'linear', 'exponential', or 'sudden', got '{degradation_curve}'")
    
    # Create base signal - starts clean
    base_freq = random.uniform(60, 120)
    base_signal = generate_sine(base_freq, -12, duration, sample_rate)
    
    # Add harmonic for richness
    harmonic = generate_sine(base_freq * 1.5, -18, duration, sample_rate)
    clean_signal = combine_signals([base_signal, harmonic], [1.0, 0.4], sample_rate)
    
    # Create degradation envelope
    if degradation_curve == 'linear':
        degradation_env = generate_linear_envelope(0.0, 1.0, duration, sample_rate)
    elif degradation_curve == 'exponential':
        degradation_env = generate_exponential_envelope(0.0, 1.0, duration, curve=3.0, sample_rate=sample_rate)
    else:  # sudden
        # Sudden failure at 70% through
        failure_point = int(0.7 * duration * sample_rate)
        degradation_env = np.zeros(int(duration * sample_rate))
        degradation_env[failure_point:] = 1.0
        # Add some ramp-up before failure
        ramp_samples = int(0.1 * duration * sample_rate)
        if failure_point > ramp_samples:
            ramp_start = failure_point - ramp_samples
            degradation_env[ramp_start:failure_point] = np.linspace(0, 1, ramp_samples)
    
    # Apply progressive effects based on degradation envelope
    degraded_signal = clean_signal.copy()
    
    # Progressive bitcrushing
    bit_depth_curve = 16 - degradation_env * 12  # 16-bit to 4-bit
    for i in range(0, len(degraded_signal), 1024):
        end_idx = min(i + 1024, len(degraded_signal))
        window = degraded_signal[i:end_idx]
        avg_bit_depth = int(np.mean(bit_depth_curve[i:end_idx]))
        avg_bit_depth = max(1, min(16, avg_bit_depth))
        degraded_signal[i:end_idx] = apply_bitcrush(window, avg_bit_depth, sample_rate)
    
    # Progressive wow and flutter
    flutter_intensity = degradation_env * 0.4
    for i in range(0, len(degraded_signal), 2048):
        end_idx = min(i + 2048, len(degraded_signal))
        window = degraded_signal[i:end_idx]
        avg_intensity = np.mean(flutter_intensity[i:end_idx])
        if avg_intensity > 0.01:
            flutter_freq = 0.5 + avg_intensity * 2.0
            degraded_signal[i:end_idx] = apply_wow_flutter(window, avg_intensity, flutter_freq, sample_rate)
    
    # Add progressive noise
    noise_level = -40 + degradation_env * 20  # -40dB to -20dB
    noise = generate_white_noise(-30, duration, sample_rate)
    noise = apply_amplitude_modulation(noise, degradation_env, depth=1.0)
    
    # Add dropouts that increase over time
    dropout_prob = degradation_env * 0.05
    for i in range(0, len(degraded_signal), 4410):  # 0.1 second windows
        end_idx = min(i + 4410, len(degraded_signal))
        window = degraded_signal[i:end_idx]
        avg_dropout = np.mean(dropout_prob[i:end_idx])
        if avg_dropout > 0.001:
            degraded_signal[i:end_idx] = apply_dropout(window, avg_dropout, 0.1, sample_rate)
    
    # Combine with noise
    combined = combine_signals([degraded_signal, noise], [1.0, 0.3], sample_rate)
    
    # Progressive filtering - signal gets more muffled
    cutoff_curve = 8000 - degradation_env * 6000  # 8kHz to 2kHz
    for i in range(0, len(combined), 2048):
        end_idx = min(i + 2048, len(combined))
        window = combined[i:end_idx]
        avg_cutoff = np.mean(cutoff_curve[i:end_idx])
        avg_cutoff = max(200, min(8000, avg_cutoff))
        combined[i:end_idx] = lowpass_filter(window, avg_cutoff, sample_rate)
    
    # Create stereo field
    stereo = create_stereo_field([combined], [0.0], width=1.0)
    
    # Add distance effect that increases over time
    distance_curve = 1.0 + degradation_env * 3.0
    for i in range(0, stereo.shape[0], 4410):
        end_idx = min(i + 4410, stereo.shape[0])
        window = stereo[i:end_idx]
        avg_distance = np.mean(distance_curve[i:end_idx])
        stereo[i:end_idx] = apply_distance_filter(window, avg_distance, sample_rate)
    
    # Master processing
    stereo = normalize_to_peak(stereo, -6.0)
    stereo = apply_master_limiter(stereo, -3.0, 4.0)
    
    return stereo


def generate_meditation_drone(duration: float = 60, base_freq: float = 60,
                            harmonic_complexity: float = 0.5, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a meditative drone with harmonic richness.
    
    Creates a stable, evolving drone perfect for meditation or ambient
    listening, with subtle harmonic movement and organic modulation.
    
    Args:
        duration: Duration in seconds (default 60)
        base_freq: Base frequency in Hz (20-200)
        harmonic_complexity: Harmonic richness (0.0-1.0)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Stereo meditation drone composition
    """
    _validate_preset_params(duration, sample_rate)
    
    if not (20 <= base_freq <= 200):
        raise ValueError(f"Base frequency must be between 20 and 200 Hz, got {base_freq}")
    if not (0.0 <= harmonic_complexity <= 1.0):
        raise ValueError(f"Harmonic complexity must be between 0.0 and 1.0, got {harmonic_complexity}")
    
    # Create harmonic series
    harmonics = []
    harmonic_count = int(2 + harmonic_complexity * 6)  # 2-8 harmonics
    
    for i in range(harmonic_count):
        harmonic_ratio = i + 1
        harmonic_freq = base_freq * harmonic_ratio
        
        if harmonic_freq >= sample_rate / 2:
            break
        
        # Amplitude decreases with harmonic number
        harmonic_amp = -12 - i * 6 - random.uniform(0, 3)
        
        # Slight detuning for organic feel
        detune = random.uniform(-0.1, 0.1) * harmonic_complexity
        actual_freq = harmonic_freq + detune
        
        # Use sine waves for pure, meditative tones
        harmonic_signal = generate_sine(actual_freq, harmonic_amp, duration, sample_rate)
        
        # Add subtle modulation to some harmonics
        if random.random() < harmonic_complexity * 0.7:
            mod_freq = random.uniform(0.01, 0.05)  # Very slow modulation
            mod_depth = harmonic_complexity * random.uniform(0.1, 0.3)
            
            if random.random() < 0.8:
                # Smooth sine LFO
                lfo = generate_lfo_sine(mod_freq, mod_depth, duration, sample_rate)
            else:
                # Organic drift
                lfo = generate_perlin_drift(duration, mod_freq, mod_depth, sample_rate)
            
            harmonic_signal = apply_amplitude_modulation(harmonic_signal, lfo, depth=0.2)
        
        harmonics.append(harmonic_signal)
    
    # Add subtle noise layer for texture
    if harmonic_complexity > 0.3:
        noise_amp = -35 - (1.0 - harmonic_complexity) * 10
        texture_noise = generate_pink_noise(noise_amp, duration, sample_rate)
        
        # Filter to complement the harmonic content
        cutoff = base_freq * 4
        texture_noise = lowpass_filter(texture_noise, cutoff, sample_rate)
        
        # Very subtle modulation
        noise_lfo = generate_perlin_drift(duration, 0.02, 0.1, sample_rate)
        texture_noise = apply_amplitude_modulation(texture_noise, noise_lfo, depth=0.1)
        
        harmonics.append(texture_noise)
    
    # Mix harmonics
    mix_levels = [1.0] * len(harmonics)
    combined = combine_signals(harmonics, mix_levels, sample_rate)
    
    # Apply gentle tube warmth for analog character
    combined = apply_tube_warmth(combined, drive=1.2, asymmetry=0.05)
    
    # Create wide stereo field
    stereo = create_stereo_field([combined], [0.0], width=1.3)
    
    # Add subtle spatial processing
    if harmonic_complexity > 0.4:
        # Very subtle reverb
        room_size = 0.2 + harmonic_complexity * 0.3
        decay_time = 2.0 + harmonic_complexity * 2.0
        damping = 0.6
        stereo = apply_reverb(stereo, room_size, decay_time, damping, sample_rate)
    
    # Master processing with gentle limiting
    stereo = normalize_to_peak(stereo, -8.0)
    stereo = apply_master_limiter(stereo, -6.0, 2.0)
    
    return stereo


# Specialized Transmission Types

def generate_shortwave_static(duration: float = 60, signal_strength: float = 0.4,
                            atmospheric_noise: float = 0.6, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate shortwave radio static with atmospheric interference.
    
    Simulates the characteristic sound of shortwave radio with varying
    signal strength and atmospheric noise conditions.
    
    Args:
        duration: Duration in seconds (default 60)
        signal_strength: Signal strength (0.0-1.0)
        atmospheric_noise: Atmospheric noise level (0.0-1.0)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Stereo shortwave static composition
    """
    _validate_preset_params(duration, sample_rate)
    
    if not (0.0 <= signal_strength <= 1.0):
        raise ValueError(f"Signal strength must be between 0.0 and 1.0, got {signal_strength}")
    if not (0.0 <= atmospheric_noise <= 1.0):
        raise ValueError(f"Atmospheric noise must be between 0.0 and 1.0, got {atmospheric_noise}")
    
    # Create weak carrier signal
    carrier_freq = random.uniform(80, 150)
    signal_amp = -20 - (1.0 - signal_strength) * 15
    carrier = generate_sine(carrier_freq, signal_amp, duration, sample_rate)
    
    # Add signal instability
    instability_lfo = generate_random_walk(duration, step_size=0.02, sample_rate=sample_rate)
    carrier = apply_amplitude_modulation(carrier, instability_lfo, depth=0.4)
    
    # Shortwave-style bandpass filtering
    center_freq = carrier_freq
    bandwidth = 100 + signal_strength * 200
    carrier = bandpass_filter(carrier, center_freq, bandwidth, sample_rate)
    
    # Create atmospheric noise layers
    noise_layers = []
    
    # High-frequency static
    static_amp = -25 - (1.0 - atmospheric_noise) * 15
    static = generate_white_noise(static_amp, duration, sample_rate)
    static = highpass_filter(static, 2000, sample_rate)
    
    # Add crackling
    crackle_freq = atmospheric_noise * 0.3
    crackle_intensity = atmospheric_noise * 0.5
    static = add_static_bursts(static, crackle_freq, crackle_intensity, sample_rate)
    noise_layers.append(static)
    
    # Low-frequency atmospheric rumble
    if atmospheric_noise > 0.3:
        rumble_amp = -35 - (1.0 - atmospheric_noise) * 10
        rumble = generate_brown_noise(rumble_amp, duration, sample_rate)
        rumble = lowpass_filter(rumble, 200, sample_rate)
        
        # Slow modulation for atmospheric movement
        atmo_lfo = generate_perlin_drift(duration, 0.03, 0.3, sample_rate)
        rumble = apply_amplitude_modulation(rumble, atmo_lfo, depth=0.5)
        noise_layers.append(rumble)
    
    # Mid-frequency interference
    if atmospheric_noise > 0.5:
        interference_amp = -30 - (1.0 - atmospheric_noise) * 10
        interference = generate_pink_noise(interference_amp, duration, sample_rate)
        interference = bandpass_filter(interference, 500, 800, sample_rate)
        
        # Random dropouts in interference
        dropout_prob = atmospheric_noise * 0.02
        interference = apply_dropout(interference, dropout_prob, 0.2, sample_rate)
        noise_layers.append(interference)
    
    # Combine all elements
    all_signals = [carrier] + noise_layers
    mix_levels = [1.0] + [0.6] * len(noise_layers)
    combined = combine_signals(all_signals, mix_levels, sample_rate)
    
    # Apply shortwave characteristics
    combined = apply_wow_flutter(combined, 0.1, 0.8, sample_rate)
    
    # Create stereo field with movement
    pan_lfo = generate_lfo_sine(0.05, 0.3, duration, sample_rate)
    stereo = create_stereo_field([combined], [0.0], width=1.1)
    
    # Add distance and air absorption
    distance = 2.0 + (1.0 - signal_strength) * 3.0
    stereo = apply_distance_filter(stereo, distance, sample_rate)
    stereo = apply_air_absorption(stereo, distance * 10, 0.4, sample_rate)
    
    # Master processing
    stereo = normalize_to_peak(stereo, -6.0)
    stereo = apply_master_limiter(stereo, -3.0, 8.0)
    
    return stereo


def generate_analog_hum(duration: float = 60, power_frequency: float = 50,
                       harmonic_content: float = 0.3, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate analog electrical hum with harmonics.
    
    Creates the characteristic hum of analog equipment, including
    power line interference and harmonic distortion.
    
    Args:
        duration: Duration in seconds (default 60)
        power_frequency: Power line frequency in Hz (50 or 60)
        harmonic_content: Harmonic richness (0.0-1.0)
        sample_rate: Sample rate in Hz (default 44100)
Returns:
        numpy.ndarray: Stereo analog hum composition
    """
    _validate_preset_params(duration, sample_rate)
    
    if power_frequency not in [50, 60]:
        raise ValueError(f"Power frequency must be 50 or 60 Hz, got {power_frequency}")
    if not (0.0 <= harmonic_content <= 1.0):
        raise ValueError(f"Harmonic content must be between 0.0 and 1.0, got {harmonic_content}")
    
    # Create fundamental hum
    fundamental = generate_sine(power_frequency, -15, duration, sample_rate)
    
    # Add harmonics
    harmonics = [fundamental]
    harmonic_count = int(1 + harmonic_content * 8)  # Up to 8 harmonics
    
    for i in range(2, harmonic_count + 1):
        harmonic_freq = power_frequency * i
        if harmonic_freq >= sample_rate / 2:
            break
        
        # Odd harmonics are stronger in power line hum
        if i % 2 == 1:
            harmonic_amp = -18 - i * 2
        else:
            harmonic_amp = -25 - i * 3
        
        harmonic_signal = generate_sine(harmonic_freq, harmonic_amp, duration, sample_rate)
        
        # Add slight instability to harmonics
        if harmonic_content > 0.3:
            instability = generate_lfo_sine(random.uniform(0.1, 0.5), 0.1, duration, sample_rate)
            harmonic_signal = apply_amplitude_modulation(harmonic_signal, instability, depth=0.1)
        
        harmonics.append(harmonic_signal)
    
    # Mix harmonics
    mix_levels = [1.0] * len(harmonics)
    combined = combine_signals(harmonics, mix_levels, sample_rate)
    
    # Add analog character
    combined = apply_tube_warmth(combined, drive=1.3, asymmetry=0.1)
    
    # Add subtle noise for realism
    if harmonic_content > 0.2:
        noise_amp = -40 - (1.0 - harmonic_content) * 10
        hum_noise = generate_pink_noise(noise_amp, duration, sample_rate)
        hum_noise = lowpass_filter(hum_noise, power_frequency * 10, sample_rate)
        combined = combine_signals([combined, hum_noise], [1.0, 0.2], sample_rate)
    
    # Create stereo field
    stereo = create_stereo_field([combined], [0.0], width=0.8)
    
    # Master processing
    stereo = normalize_to_peak(stereo, -8.0)
    stereo = apply_master_limiter(stereo, -6.0, 3.0)
    
    return stereo


def generate_deep_space_signal(duration: float = 60, carrier_freq: float = 80,
                             modulation_depth: float = 0.2, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a deep space communication signal.
    
    Creates an otherworldly signal that suggests communication from
    deep space, with mysterious modulation and cosmic interference.
    
    Args:
        duration: Duration in seconds (default 60)
        carrier_freq: Carrier frequency in Hz (40-200)
        modulation_depth: Modulation intensity (0.0-1.0)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Stereo deep space signal composition
    """
    _validate_preset_params(duration, sample_rate)
    
    if not (40 <= carrier_freq <= 200):
        raise ValueError(f"Carrier frequency must be between 40 and 200 Hz, got {carrier_freq}")
    if not (0.0 <= modulation_depth <= 1.0):
        raise ValueError(f"Modulation depth must be between 0.0 and 1.0, got {modulation_depth}")
    
    # Create carrier signal
    carrier = generate_sine(carrier_freq, -12, duration, sample_rate)
    
    # Add complex modulation patterns
    modulation_layers = []
    
    # Slow, mysterious frequency modulation
    fm_freq = random.uniform(0.01, 0.05)
    fm_depth = modulation_depth * random.uniform(2.0, 8.0)
    fm_lfo = generate_perlin_drift(duration, fm_freq, fm_depth, sample_rate)
    carrier = apply_frequency_modulation(carrier_freq, fm_lfo, fm_depth, duration, sample_rate)
    
    # Amplitude modulation with complex patterns
    if modulation_depth > 0.2:
        am_freq1 = random.uniform(0.02, 0.08)
        am_depth1 = modulation_depth * 0.4
        am_lfo1 = generate_lfo_sine(am_freq1, am_depth1, duration, sample_rate)
        
        am_freq2 = random.uniform(0.05, 0.15)
        am_depth2 = modulation_depth * 0.3
        am_lfo2 = generate_lfo_triangle(am_freq2, am_depth2, duration, sample_rate)
        
        # Combine modulation sources
        combined_am = combine_signals([am_lfo1, am_lfo2], [1.0, 0.7], sample_rate)
        carrier = apply_amplitude_modulation(carrier, combined_am, depth=0.6)
    
    # Add harmonic sidebands
    if modulation_depth > 0.4:
        sideband_freq = carrier_freq * random.uniform(1.3, 1.7)
        if sideband_freq < sample_rate / 2:
            sideband = generate_sine(sideband_freq, -20, duration, sample_rate)
            
            # Modulate sideband differently
            sideband_lfo = generate_random_walk(duration, step_size=0.05, sample_rate=sample_rate)
            sideband = apply_amplitude_modulation(sideband, sideband_lfo, depth=0.8)
            
            carrier = combine_signals([carrier, sideband], [1.0, 0.3], sample_rate)
    
    # Add cosmic noise
    cosmic_noise_amp = -35 - (1.0 - modulation_depth) * 10
    cosmic_noise = generate_pink_noise(cosmic_noise_amp, duration, sample_rate)
    
    # Filter cosmic noise to complement the signal
    cosmic_noise = bandpass_filter(cosmic_noise, carrier_freq * 0.5, carrier_freq * 2, sample_rate)
    
    # Add mysterious dropouts
    if modulation_depth > 0.3:
        dropout_prob = modulation_depth * 0.01
        cosmic_noise = apply_dropout(cosmic_noise, dropout_prob, 0.5, sample_rate)
    
    # Combine signal with cosmic noise
    combined = combine_signals([carrier, cosmic_noise], [1.0, 0.4], sample_rate)
    
    # Apply space-like processing
    combined = apply_distance_filter(combined, 5.0, sample_rate)
    combined = apply_air_absorption(combined, 100.0, 0.1, sample_rate)  # Very dry space
    
    # Create wide stereo field with movement
    pan_lfo = generate_lfo_sine(0.03, 0.5, duration, sample_rate)
    stereo = create_stereo_field([combined], [0.0], width=1.5)
    
    # Add deep space reverb
    if modulation_depth > 0.5:
        room_size = 0.8
        decay_time = 5.0 + modulation_depth * 5.0
        damping = 0.2  # Less damping for space-like reverb
        stereo = apply_reverb(stereo, room_size, decay_time, damping, sample_rate)
    
    # Master processing
    stereo = normalize_to_peak(stereo, -6.0)
    stereo = apply_master_limiter(stereo, -3.0, 4.0)
    
    return stereo


def generate_underground_transmission(duration: float = 60, earth_filtering: float = 0.7,
                                    echo_depth: float = 0.5, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate an underground transmission signal.
    
    Creates a signal that sounds like it's coming from deep underground,
    with earth filtering and cavernous echoes.
    
    Args:
        duration: Duration in seconds (default 60)
        earth_filtering: Earth filtering intensity (0.0-1.0)
        echo_depth: Echo/reverb depth (0.0-1.0)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        numpy.ndarray: Stereo underground transmission composition
    """
    _validate_preset_params(duration, sample_rate)
    
    if not (0.0 <= earth_filtering <= 1.0):
        raise ValueError(f"Earth filtering must be between 0.0 and 1.0, got {earth_filtering}")
    if not (0.0 <= echo_depth <= 1.0):
        raise ValueError(f"Echo depth must be between 0.0 and 1.0, got {echo_depth}")
    
    # Create base signal - lower frequencies travel better through earth
    base_freq = random.uniform(40, 80)
    base_signal = generate_sine(base_freq, -10, duration, sample_rate)
    
    # Add low-frequency harmonics
    harmonic_freq = base_freq * random.uniform(1.5, 2.0)
    if harmonic_freq < 200:  # Keep it low
        harmonic = generate_sine(harmonic_freq, -18, duration, sample_rate)
        base_signal = combine_signals([base_signal, harmonic], [1.0, 0.4], sample_rate)
    
    # Add subtle modulation for instability
    instability_lfo = generate_random_walk(duration, step_size=0.03, sample_rate=sample_rate)
    base_signal = apply_amplitude_modulation(base_signal, instability_lfo, depth=0.3)
    
    # Apply earth filtering - progressive low-pass filtering
    filtered_signal = base_signal.copy()
    
    # Multiple stages of filtering to simulate earth absorption
    cutoff_freq = 2000 - earth_filtering * 1500  # 2kHz down to 500Hz
    filtered_signal = lowpass_filter(filtered_signal, cutoff_freq, sample_rate)
    
    if earth_filtering > 0.3:
        # Additional filtering stage
        cutoff_freq2 = 1000 - earth_filtering * 600
        filtered_signal = lowpass_filter(filtered_signal, cutoff_freq2, sample_rate)
    
    if earth_filtering > 0.6:
        # Heavy earth filtering
        cutoff_freq3 = 400 - earth_filtering * 200
        filtered_signal = lowpass_filter(filtered_signal, cutoff_freq3, sample_rate)
    
    # Add underground rumble
    rumble_amp = -25 - (1.0 - earth_filtering) * 10
    rumble = generate_brown_noise(rumble_amp, duration, sample_rate)
    rumble = lowpass_filter(rumble, 100, sample_rate)  # Very low rumble
    
    # Slow modulation of rumble
    rumble_lfo = generate_perlin_drift(duration, 0.02, 0.2, sample_rate)
    rumble = apply_amplitude_modulation(rumble, rumble_lfo, depth=0.4)
    
    # Combine signal with rumble
    combined = combine_signals([filtered_signal, rumble], [1.0, 0.3], sample_rate)
    
    # Create stereo field
    stereo = create_stereo_field([combined], [0.0], width=1.0)
    
    # Add cavernous echoes
    if echo_depth > 0.2:
        # Multiple echo delays for cave-like effect
        delay1_time = 0.2 + echo_depth * 0.3
        delay1_feedback = echo_depth * 0.3
        delay1_mix = echo_depth * 0.4
        stereo = apply_echo_delay(stereo, delay1_time, delay1_feedback, delay1_mix, sample_rate)
        
        if echo_depth > 0.5:
            # Second echo for deeper caves
            delay2_time = 0.5 + echo_depth * 0.5
            delay2_feedback = echo_depth * 0.2
            delay2_mix = echo_depth * 0.3
            stereo = apply_echo_delay(stereo, delay2_time, delay2_feedback, delay2_mix, sample_rate)
    
    # Add underground reverb
    if echo_depth > 0.3:
        room_size = 0.6 + echo_depth * 0.4
        decay_time = 3.0 + echo_depth * 4.0
        damping = 0.7 + earth_filtering * 0.2  # More damping with earth filtering
        stereo = apply_reverb(stereo, room_size, decay_time, damping, sample_rate)
    
    # Apply distance effects
    distance = 3.0 + earth_filtering * 2.0
    stereo = apply_distance_filter(stereo, distance, sample_rate)
    
    # Master processing
    stereo = normalize_to_peak(stereo, -6.0)
    stereo = apply_master_limiter(stereo, -3.0, 4.0)
    
    return stereo


# Utility Functions

def create_preset_variations(preset_function: callable, num_variations: int = 5,
                           variation_amount: float = 0.2, **preset_kwargs) -> List[np.ndarray]:
    """
    Create variations of a preset function.
    
    Generates multiple variations of a preset by randomly modifying
    its parameters within specified bounds.
    
    Args:
        preset_function: The preset function to create variations of
        num_variations: Number of variations to create (1-10)
        variation_amount: Amount of variation (0.0-1.0)
        **preset_kwargs: Keyword arguments for the preset function
    
    Returns:
        List[np.ndarray]: List of preset variations
    """
    if not callable(preset_function):
        raise TypeError("Preset function must be callable")
    if not (1 <= num_variations <= 10):
        raise ValueError(f"Number of variations must be between 1 and 10, got {num_variations}")
    if not (0.0 <= variation_amount <= 1.0):
        raise ValueError(f"Variation amount must be between 0.0 and 1.0, got {variation_amount}")
    
    variations = []
    
    for i in range(num_variations):
        # Create modified parameters
        modified_kwargs = preset_kwargs.copy()
        
        # Apply variations to numeric parameters
        for key, value in modified_kwargs.items():
            if isinstance(value, (int, float)) and key != 'sample_rate':
                if key == 'duration':
                    # Don't vary duration too much
                    variation_factor = 1.0 + random.uniform(-0.1, 0.1) * variation_amount
                elif key in ['base_freq', 'carrier_freq', 'power_frequency']:
                    # Frequency variations
                    variation_factor = 1.0 + random.uniform(-0.2, 0.2) * variation_amount
                elif key.endswith('_level') or key.endswith('_strength') or key.endswith('_depth'):
                    # Level/strength/depth parameters (0-1 range)
                    variation = random.uniform(-0.3, 0.3) * variation_amount
                    modified_kwargs[key] = max(0.0, min(1.0, value + variation))
                    continue
                else:
                    # General numeric parameters
                    variation_factor = 1.0 + random.uniform(-0.3, 0.3) * variation_amount
                
                modified_kwargs[key] = value * variation_factor
        
        # Generate variation with modified parameters
        try:
            variation = preset_function(**modified_kwargs)
            variations.append(variation)
        except Exception as e:
            # If variation fails, use original parameters
            warnings.warn(f"Variation {i} failed: {e}. Using original parameters.")
            variation = preset_function(**preset_kwargs)
            variations.append(variation)
    
    return variations


def export_composition(signal: np.ndarray, filename: str, format: str = 'wav',
                      sample_rate: int = 44100, bit_depth: int = 24) -> None:
    """
    Export composition to audio file.
    
    Saves the generated composition to an audio file with specified
    format and quality settings.
    
    Args:
        signal: Audio signal to export (mono or stereo)
        filename: Output filename (without extension)
        format: Audio format ('wav', 'flac', 'aiff')
        sample_rate: Sample rate in Hz (default 44100)
        bit_depth: Bit depth (16, 24, 32)
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array")
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")
    if format not in ['wav', 'flac', 'aiff']:
        raise ValueError(f"Format must be 'wav', 'flac', or 'aiff', got '{format}'")
    if bit_depth not in [16, 24, 32]:
        raise ValueError(f"Bit depth must be 16, 24, or 32, got {bit_depth}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile library required for audio export. Install with: pip install soundfile")
    
    # Ensure signal is in correct format
    if signal.ndim == 1:
        # Mono signal
        export_signal = signal
    elif signal.ndim == 2:
        # Stereo signal
        export_signal = signal
    else:
        raise ValueError("Signal must be mono (1D) or stereo (2D)")
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(export_signal))
    if max_val > 0.95:
        export_signal = export_signal * (0.95 / max_val)
    
    # Add file extension
    full_filename = f"{filename}.{format}"
    
    # Determine subtype based on bit depth
    if format == 'wav':
        if bit_depth == 16:
            subtype = 'PCM_16'
        elif bit_depth == 24:
            subtype = 'PCM_24'
        else:  # 32
            subtype = 'PCM_32'
    elif format == 'flac':
        if bit_depth == 16:
            subtype = 'PCM_16'
        elif bit_depth == 24:
            subtype = 'PCM_24'
        else:  # 32
            subtype = 'PCM_24'  # FLAC doesn't support 32-bit
    else:  # aiff
        if bit_depth == 16:
            subtype = 'PCM_16'
        elif bit_depth == 24:
            subtype = 'PCM_24'
        else:  # 32
            subtype = 'PCM_32'
    
    # Write file
    sf.write(full_filename, export_signal, sample_rate, subtype=subtype)
    print(f"Exported composition to {full_filename}")


def analyze_composition_spectrum(signal: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
    """
    Analyze the frequency spectrum of a composition.
    
    Provides spectral analysis information about the composition,
    useful for understanding its frequency content and characteristics.
    
    Args:
        signal: Audio signal to analyze (mono or stereo)
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        dict: Analysis results including peak frequency, spectral centroid, etc.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array")
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    # Convert to mono if stereo
    if signal.ndim == 2:
        mono_signal = np.mean(signal, axis=1)
    else:
        mono_signal = signal
    
    # Compute FFT
    fft = np.fft.rfft(mono_signal)
    magnitude = np.abs(fft)
    frequencies = np.fft.rfftfreq(len(mono_signal), 1/sample_rate)
    
    # Find peak frequency
    peak_idx = np.argmax(magnitude)
    peak_frequency = frequencies[peak_idx]
    
    # Calculate spectral centroid
    spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
    
    # Calculate RMS energy
    rms_energy = np.sqrt(np.mean(mono_signal ** 2))
    
    # Calculate dynamic range
    peak_amplitude = np.max(np.abs(mono_signal))
    dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
    
    # Find dominant frequency bands
    low_band = np.sum(magnitude[(frequencies >= 20) & (frequencies < 200)])
    mid_band = np.sum(magnitude[(frequencies >= 200) & (frequencies < 2000)])
    high_band = np.sum(magnitude[(frequencies >= 2000) & (frequencies < 20000)])
    
    total_energy = low_band + mid_band + high_band
    if total_energy > 0:
        low_ratio = low_band / total_energy
        mid_ratio = mid_band / total_energy
        high_ratio = high_band / total_energy
    else:
        low_ratio = mid_ratio = high_ratio = 0.0
    
    return {
        'peak_frequency': float(peak_frequency),
        'spectral_centroid': float(spectral_centroid),
        'rms_energy': float(rms_energy),
        'peak_amplitude': float(peak_amplitude),
        'dynamic_range_db': float(dynamic_range),
        'frequency_bands': {
            'low_20_200hz': float(low_ratio),
            'mid_200_2000hz': float(mid_ratio),
            'high_2000_20000hz': float(high_ratio)
        },
        'duration_seconds': len(mono_signal) / sample_rate,
        'sample_rate': sample_rate
    }


def generate_composition_metadata(preset_name: str, parameters: Dict[str, Any],
                                duration: float, sample_rate: int = 44100) -> Dict[str, Any]:
    """
    Generate metadata for a composition.
    
    Creates comprehensive metadata about a generated composition,
    useful for cataloging and organizing generated pieces.
    
    Args:
        preset_name: Name of the preset function used
        parameters: Parameters used to generate the composition
        duration: Duration in seconds
        sample_rate: Sample rate in Hz (default 44100)
    
    Returns:
        dict: Composition metadata
    """
    if not isinstance(preset_name, str):
        raise TypeError("Preset name must be a string")
    if not isinstance(parameters, dict):
        raise TypeError("Parameters must be a dictionary")
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    import datetime
    
    # Generate unique ID based on timestamp and parameters
    timestamp = datetime.datetime.now().isoformat()
    param_hash = hash(str(sorted(parameters.items())))
    composition_id = f"{preset_name}_{abs(param_hash)}_{int(datetime.datetime.now().timestamp())}"
    
    # Categorize preset type
    if 'transmission' in preset_name or 'radio' in preset_name:
        category = 'transmission'
    elif 'meditation' in preset_name or 'ambient' in preset_name:
        category = 'ambient'
    elif 'space' in preset_name or 'underground' in preset_name:
        category = 'environmental'
    elif 'layered' in preset_name or 'evolving' in preset_name:
        category = 'compositional'
    else:
        category = 'experimental'
    
    # Estimate complexity based on parameters
    complexity_indicators = ['num_layers', 'harmonic_complexity', 'texture_density', 'modulation_depth']
    complexity_score = 0.0
    complexity_count = 0
    
    for indicator in complexity_indicators:
        if indicator in parameters:
            complexity_score += float(parameters[indicator])
            complexity_count += 1
    
    if complexity_count > 0:
        avg_complexity = complexity_score / complexity_count
        if avg_complexity < 0.3:
            complexity_level = 'simple'
        elif avg_complexity < 0.7:
            complexity_level = 'medium'
        else:
            complexity_level = 'complex'
    else:
        complexity_level = 'unknown'
    
    return {
        'composition_id': composition_id,
        'preset_name': preset_name,
        'category': category,
        'complexity_level': complexity_level,
        'parameters': parameters.copy(),
        'duration_seconds': duration,
        'sample_rate': sample_rate,
        'generated_at': timestamp,
        'estimated_file_size_mb': (duration * sample_rate * 2 * 3) / (1024 * 1024),  # Stereo, 24-bit
        'tags': [
            'drone',
            'ambient',
            'broken_transmission',
            category,
            complexity_level
        ]
    }


if __name__ == "__main__":
    # Basic tests and examples
    print("Testing drone preset functions...")
    
    try:
        # Test core presets
        print("Testing core presets...")
        broken_tx = generate_broken_transmission(duration=5, complexity='medium', seed=42)
        print(f"✓ Broken transmission: {broken_tx.shape}")
        
        distant_radio = generate_distant_radio(duration=5, interference_level=0.4)
        print(f"✓ Distant radio: {distant_radio.shape}")
        
        failing_tech = generate_failing_technology(duration=5, degradation_curve='exponential')
        print(f"✓ Failing technology: {failing_tech.shape}")
        
        meditation = generate_meditation_drone(duration=5, base_freq=60, harmonic_complexity=0.5)
        print(f"✓ Meditation drone: {meditation.shape}")
        
        # Test specialized transmission types
        print("\nTesting specialized transmission types...")
        shortwave = generate_shortwave_static(duration=5, signal_strength=0.4, atmospheric_noise=0.6)
        print(f"✓ Shortwave static: {shortwave.shape}")
        
        analog_hum = generate_analog_hum(duration=5, power_frequency=50, harmonic_content=0.3)
        print(f"✓ Analog hum: {analog_hum.shape}")
        
        deep_space = generate_deep_space_signal(duration=5, carrier_freq=80, modulation_depth=0.2)
        print(f"✓ Deep space signal: {deep_space.shape}")
        
        underground = generate_underground_transmission(duration=5, earth_filtering=0.7, echo_depth=0.5)
        print(f"✓ Underground transmission: {underground.shape}")
        
        # Test utility functions
        print("\nTesting utility functions...")
        variations = create_preset_variations(generate_meditation_drone, num_variations=3, 
                                            variation_amount=0.2, duration=3, base_freq=80)
        print(f"✓ Created {len(variations)} variations")
        
        analysis = analyze_composition_spectrum(meditation)
        print(f"✓ Spectrum analysis: peak at {analysis['peak_frequency']:.1f}Hz")
        
        metadata = generate_composition_metadata('meditation_drone', 
                                               {'base_freq': 60, 'harmonic_complexity': 0.5}, 
                                               5.0)
        print(f"✓ Generated metadata: {metadata['composition_id']}")
        
        print("\n✓ All drone preset functions working correctly!")
        print("Phase 5 implementation complete.")
        
    except Exception as e:
        print(f"✗ Error in drone preset tests: {e}")
        import traceback
        traceback.print_exc()