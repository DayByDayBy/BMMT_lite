"""
kleprami Ambient Drone Generator

Creates ambient drone compositions in the style of 'kleprami'.
Each run produces a distinctly different track while maintaining core sonic aesthetic.

Core kleprami principles:
- Extremely slow motion (0.005-0.03Hz modulation)
- Soft attacks (5-30 second fade-ins)
- Low-to-mid frequency focus with sub-bass foundation
- Gentle detuning for slow beating patterns
- Organic modulation using Perlin drift
- Vast spatial depth with long reverb tails
- Granular/textured quality with noise layers
- No hard transients or fast changes
- Non-teleological structure (feels eternal)
"""

import numpy as np
import random
import json
import os
import datetime
from typing import Dict, Any, Optional, Tuple
import soundfile as sf

# BMMT imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from bmmt.audio import generate_sine, generate_triangle, generate_white_noise, generate_pink_noise
from bmmt.modulation.modulation import generate_perlin_drift, generate_exponential_envelope
from bmmt.processing import (
    apply_reverb, apply_distance_filter, apply_air_absorption, apply_stereo_width,
    apply_tube_warmth, apply_tape_saturation, apply_soft_limiter,
    lowpass_filter, highpass_filter
)
from bmmt.processing.reverbs import apply_hall_reverb, apply_plate_reverb


# ============================================================================
# PARAMETER SYSTEM WITH RELATIONSHIP CONSTRAINTS
# ============================================================================

KLEPRAMI_RANGES = {
    'base_freq': (30, 90),  # Hz - deep but not subsonic
    'num_layers': (3, 6),   # Number of drone layers
    'fade_in_duration': (5, 30),  # seconds - soft attacks
    'modulation_speed': (0.005, 0.03),  # Hz - extremely slow
    'detuning_amount': (0.1, 0.8),  # Hz - subtle beating
    'reverb_decay': (3.0, 8.0),  # seconds - vast spaces
    'distance_factor': (2.0, 5.0),  # spatial depth
    'noise_floor': (-45, -35),  # dB - subtle texture
    'track_duration': (15*60, 45*60),  # 15-45 minutes
    'shimmer_presence': (0.3, 0.8),  # probability of shimmer layer
    'tube_warmth': (0.2, 0.6),  # analog warmth amount
    'tape_saturation': (0.1, 0.4),  # tape saturation amount
}


def apply_parameter_constraints(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply intelligent parameter relationships to ensure musical coherence.
    Prevents nonsensical parameter combinations.
    """
    # More layers → shorter reverb (avoid mud)
    if params['num_layers'] >= 5:
        params['reverb_decay'] = min(params['reverb_decay'], 5.0)
    elif params['num_layers'] <= 3:
        params['reverb_decay'] = max(params['reverb_decay'], 4.0)
    
    # Lower base freqs → longer fade-ins (avoid muddiness)
    if params['base_freq'] <= 40:
        params['fade_in_duration'] = max(params['fade_in_duration'], 15.0)
    elif params['base_freq'] >= 70:
        params['fade_in_duration'] = min(params['fade_in_duration'], 12.0)
    
    # Higher detuning → fewer layers (avoid chaos)
    if params['detuning_amount'] >= 0.6:
        params['num_layers'] = min(params['num_layers'], 4)
    elif params['detuning_amount'] <= 0.2:
        params['num_layers'] = max(params['num_layers'], 4)
    
    # Adjust distance factor based on reverb decay
    if params['reverb_decay'] >= 6.0:
        params['distance_factor'] = max(params['distance_factor'], 3.0)
    
    return params


def generate_kleprami_parameters(seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate parameters for kleprami track with intelligent constraints.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of parameters for track generation
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate base parameters within ranges
    params = {}
    for key, (min_val, max_val) in KLEPRAMI_RANGES.items():
        if isinstance(min_val, float) or isinstance(max_val, float):
            params[key] = random.uniform(min_val, max_val)
        else:
            params[key] = random.randint(min_val, max_val)
    
    # Apply intelligent constraints
    params = apply_parameter_constraints(params)
    
    # Add derived parameters
    params['sample_rate'] = 44100
    params['duration'] = params['track_duration']
    
    # Generate harmonic ratios for layers
    params['harmonic_ratios'] = []
    for i in range(params['num_layers']):
        if i == 0:
            # Foundation layer - fundamental
            params['harmonic_ratios'].append(1.0)
        else:
            # Harmonic layers - use musical ratios
            ratio_choices = [2.0, 3.0, 5.0, 7.0, 1.5, 2.5, 3.5]
            params['harmonic_ratios'].append(random.choice(ratio_choices))
    
    # Generate oscillator types (bias toward sine/triangle)
    oscillator_choices = ['sine', 'sine', 'sine', 'triangle', 'triangle', 'sawtooth']
    params['oscillator_types'] = []
    for i in range(params['num_layers']):
        params['oscillator_types'].append(random.choice(oscillator_choices))
    
    # Generate layer amplitudes (negative dB, following BMMT convention)
    params['layer_amplitudes'] = []
    for i in range(params['num_layers']):
        if i == 0:
            # Foundation layer - strongest
            amp = random.uniform(-12, -9)
        elif i < params['num_layers'] // 2:
            # Mid layers
            amp = random.uniform(-18, -12)
        else:
            # Upper layers
            amp = random.uniform(-24, -18)
        params['layer_amplitudes'].append(amp)
    
    return params


# ============================================================================
# LAYER BUILDER CLASS FOR HARMONIC COHERENCE
# ============================================================================

class DroneLayerBuilder:
    """
    Builds harmonically coherent drone layers for kleprami tracks.
    Ensures all layers relate to each other musically.
    """
    
    def __init__(self, params: Dict[str, Any], duration: float, sample_rate: int = 44100):
        self.params = params
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        
        # Generate shared modulation patterns for coherence
        # Use multiple modulation speeds for evolution over time
        slow_mod = generate_perlin_drift(
            duration, 
            frequency=params['modulation_speed'], 
            amplitude=params['detuning_amount'],
            seed=params.get('seed', None)
        )
        
        medium_mod = generate_perlin_drift(
            duration,
            frequency=params['modulation_speed'] * 3.0,  # 3x faster
            amplitude=params['detuning_amount'] * 0.5,  # Half amplitude
            seed=params.get('seed', 11111) if params.get('seed') else None
        )
        
        # Crossfade between slow and medium modulation over time
        crossfade_duration = min(duration * 0.3, 300)  # 30% of track or 5 minutes
        crossfade_samples = int(crossfade_duration * self.sample_rate)
        
        self.freq_modulation = np.zeros(self.num_samples)
        if crossfade_samples < self.num_samples:
            # Start with slow modulation
            self.freq_modulation[:crossfade_samples] = slow_mod[:crossfade_samples]
            # Crossfade to medium modulation
            fade = np.linspace(0, 1, crossfade_samples)
            crossfade_section = slow_mod[crossfade_samples:2*crossfade_samples] * (1 - fade) + \
                              medium_mod[crossfade_samples:2*crossfade_samples] * fade
            self.freq_modulation[crossfade_samples:2*crossfade_samples] = crossfade_section
            # Continue with medium modulation
            self.freq_modulation[2*crossfade_samples:] = medium_mod[2*crossfade_samples:]
        else:
            self.freq_modulation = slow_mod
        
        # Amplitude modulation with similar evolution
        slow_amp = generate_perlin_drift(
            duration,
            frequency=params['modulation_speed'] * 0.7,
            amplitude=0.1,
            seed=params.get('seed', 12345) if params.get('seed') else None
        )
        
        medium_amp = generate_perlin_drift(
            duration,
            frequency=params['modulation_speed'] * 2.0,
            amplitude=0.15,
            seed=params.get('seed', 54321) if params.get('seed') else None
        )
        
        self.amp_modulation = np.zeros(self.num_samples)
        if crossfade_samples < self.num_samples:
            self.amp_modulation[:crossfade_samples] = slow_amp[:crossfade_samples]
            fade = np.linspace(0, 1, crossfade_samples)
            crossfade_section = slow_amp[crossfade_samples:2*crossfade_samples] * (1 - fade) + \
                              medium_amp[crossfade_samples:2*crossfade_samples] * fade
            self.amp_modulation[crossfade_samples:2*crossfade_samples] = crossfade_section
            self.amp_modulation[2*crossfade_samples:] = medium_amp[2*crossfade_samples:]
        else:
            self.amp_modulation = slow_amp
        
        # Generate fade envelope
        self.fade_envelope = generate_exponential_envelope(
            -60.0,  # Start from silence
            0.0,    # End at full level
            params['fade_in_duration'],
            self.sample_rate
        )
        
        # Extend fade envelope to full duration
        if len(self.fade_envelope) < self.num_samples:
            fade_extension = np.ones(self.num_samples - len(self.fade_envelope))
            self.fade_envelope = np.concatenate([self.fade_envelope, fade_extension])
        else:
            self.fade_envelope = self.fade_envelope[:self.num_samples]
    
    def build_foundation_layer(self, freq: float, detune: float) -> np.ndarray:
        """
        Build sub-bass and low-mid foundation layer.
        Uses base frequency with small detuning for slow beating.
        """
        # Apply frequency modulation
        modulated_freq = freq + self.freq_modulation * detune
        
        # Generate oscillator
        if self.params['oscillator_types'][0] == 'sine':
            layer = generate_sine(freq, self.params['layer_amplitudes'][0], self.duration, self.sample_rate)
        elif self.params['oscillator_types'][0] == 'triangle':
            layer = generate_triangle(freq, self.params['layer_amplitudes'][0], self.duration, self.sample_rate)
        else:
            layer = generate_sine(freq, self.params['layer_amplitudes'][0], self.duration, self.sample_rate)
        
        # Apply amplitude modulation (very subtle)
        layer = layer * (1.0 + self.amp_modulation * 0.05)
        
        # Apply fade envelope
        layer = layer * self.fade_envelope
        
        return layer
    
    def build_harmonic_layer(self, base_freq: float, harmonic_ratio: float, layer_idx: int) -> np.ndarray:
        """
        Build mid-range harmonic layer using integer ratios.
        Creates harmonic relationships to foundation.
        """
        freq = base_freq * harmonic_ratio
        
        # Apply frequency modulation (scaled by layer position)
        detune_scale = 1.0 / (layer_idx + 1)  # Higher layers have less detuning
        modulated_freq = freq + self.freq_modulation * self.params['detuning_amount'] * detune_scale
        
        # Generate oscillator
        if self.params['oscillator_types'][layer_idx] == 'sine':
            layer = generate_sine(freq, self.params['layer_amplitudes'][layer_idx], self.duration, self.sample_rate)
        elif self.params['oscillator_types'][layer_idx] == 'triangle':
            layer = generate_triangle(freq, self.params['layer_amplitudes'][layer_idx], self.duration, self.sample_rate)
        else:
            layer = generate_sine(freq, self.params['layer_amplitudes'][layer_idx], self.duration, self.sample_rate)
        
        # Apply amplitude modulation
        layer = layer * (1.0 + self.amp_modulation * 0.03)
        
        # Apply fade envelope
        layer = layer * self.fade_envelope
        
        # Gentle high-pass for higher harmonics to avoid mud
        if harmonic_ratio > 3.0:
            layer = highpass_filter(layer, 80, self.sample_rate)
        
        return layer
    
    def build_shimmer_layer(self, base_freq: float) -> Optional[np.ndarray]:
        """
        Build high-frequency texture layer.
        Uses high harmonics with heavy filtering and evolving modulation.
        """
        if random.random() > self.params['shimmer_presence']:
            return None
        
        # Use very high harmonic (16th-32nd)
        shimmer_freq = base_freq * random.choice([16.0, 24.0, 32.0])
        
        # Generate with sine for pure tone
        shimmer = generate_sine(shimmer_freq, -30, self.duration, self.sample_rate)
        
        # Heavy low-pass to create shimmer, not harshness
        shimmer = lowpass_filter(shimmer, 2000, self.sample_rate)
        
        # Apply evolving amplitude modulation with multiple phases
        # Phase 1: Subtle shimmer (first third)
        phase1_samples = self.num_samples // 3
        shimmer_mod1 = generate_perlin_drift(
            self.duration / 3,
            frequency=self.params['modulation_speed'] * 4.0,
            amplitude=0.2,
            seed=self.params.get('seed', 98765) if self.params.get('seed') else None
        )
        
        # Phase 2: More active shimmer (middle third)
        phase2_samples = self.num_samples // 3
        shimmer_mod2 = generate_perlin_drift(
            self.duration / 3,
            frequency=self.params['modulation_speed'] * 6.0,
            amplitude=0.3,
            seed=self.params.get('seed', 43210) if self.params.get('seed') else None
        )
        
        # Phase 3: Evolving shimmer (final third)
        phase3_samples = self.num_samples - phase1_samples - phase2_samples
        shimmer_mod3 = generate_perlin_drift(
            self.duration / 3,
            frequency=self.params['modulation_speed'] * 8.0,
            amplitude=0.25,
            seed=self.params.get('seed', 13579) if self.params.get('seed') else None
        )
        
        # Combine phases with smooth transitions
        shimmer_mod = np.concatenate([shimmer_mod1, shimmer_mod2, shimmer_mod3])
        shimmer_mod = shimmer_mod[:self.num_samples]  # Ensure correct length
        
        shimmer = shimmer * (1.0 + shimmer_mod)
        
        # Apply fade envelope
        shimmer = shimmer * self.fade_envelope
        
        return shimmer
    
    def render_all(self) -> np.ndarray:
        """
        Generate and mix all layers at once.
        Returns stereo numpy array.
        """
        layers = []
        
        # Build foundation layer
        foundation = self.build_foundation_layer(
            self.params['base_freq'], 
            self.params['detuning_amount']
        )
        layers.append(foundation)
        
        # Build harmonic layers
        for i in range(1, self.params['num_layers']):
            harmonic_layer = self.build_harmonic_layer(
                self.params['base_freq'],
                self.params['harmonic_ratios'][i],
                i
            )
            layers.append(harmonic_layer)
        
        # Build optional shimmer layer
        shimmer = self.build_shimmer_layer(self.params['base_freq'])
        if shimmer is not None:
            layers.append(shimmer)
        
        # Mix all layers
        mix = np.zeros(self.num_samples)
        for layer in layers:
            mix += layer
        
        # Convert to stereo
        stereo_mix = np.column_stack([mix, mix])
        
        return stereo_mix


# ============================================================================
# AUTOMATED AESTHETIC VALIDATION
# ============================================================================

def validate_kleprami_aesthetic(audio: np.ndarray, params: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Validate that generated audio complies with kleprami aesthetic constraints.
    Provides automated checks alongside subjective listening.
    
    Args:
        audio: Stereo audio signal (numpy array)
        params: Generation parameters
        
    Returns:
        Tuple of (is_valid, warnings_list)
    """
    warnings = []
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    elif audio.shape[1] != 2:
        warnings.append("Audio should be stereo (2 channels)")
    
    # Check 1: No fast modulation (spectral envelope analysis)
    try:
        # Analyze amplitude envelope for fast changes
        left_channel = audio[:, 0]
        window_size = int(params['sample_rate'] * 1.0)  # 1-second windows
        
        # Calculate RMS in windows
        rms_values = []
        for i in range(0, len(left_channel) - window_size, window_size // 4):
            window = left_channel[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        rms_values = np.array(rms_values)
        
        # Check for fast changes (>0.5dB/second)
        if len(rms_values) > 1:
            rms_db = 20 * np.log10(rms_values + 1e-10)
            rms_diff = np.abs(np.diff(rms_db))
            fast_changes = np.sum(rms_diff > 0.5)
            
            if fast_changes > len(rms_diff) * 0.1:  # More than 10% fast changes
                warnings.append(f"Fast amplitude modulation detected: {fast_changes} rapid changes")
    except Exception as e:
        warnings.append(f"Could not analyze modulation speed: {e}")
    
    # Check 2: Frequency distribution (bass/mid heavy)
    try:
        # FFT analysis
        fft = np.fft.fft(left_channel)
        freqs = np.fft.fftfreq(len(left_channel), 1/params['sample_rate'])
        magnitude = np.abs(fft)
        
        # Only consider positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        magnitude = magnitude[pos_mask]
        
        # Calculate energy in different bands
        bass_energy = np.sum(magnitude[(freqs >= 20) & (freqs < 200)])
        mid_energy = np.sum(magnitude[(freqs >= 200) & (freqs < 2000)])
        high_energy = np.sum(magnitude[(freqs >= 2000) & (freqs < 8000)])
        ultra_high_energy = np.sum(magnitude[freqs >= 8000])
        
        total_energy = bass_energy + mid_energy + high_energy + ultra_high_energy
        
        if total_energy > 0:
            bass_ratio = bass_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
            ultra_high_ratio = ultra_high_energy / total_energy
            
            # kleprami should be bass/mid heavy
            if bass_ratio + mid_ratio < 0.7:
                warnings.append(f"Insufficient bass/mid content: {bass_ratio + mid_ratio:.2f} (should be >0.7)")
            
            if ultra_high_ratio > 0.1:
                warnings.append(f"Excessive high frequency content: {ultra_high_ratio:.2f} (should be <0.1)")
            
            if high_ratio > 0.2:
                warnings.append(f"Excessive high-mid content: {high_ratio:.2f} (should be <0.2)")
    except Exception as e:
        warnings.append(f"Could not analyze frequency distribution: {e}")
    
    # Check 3: Dynamic range (gentle, no spikes)
    try:
        # Analyze dynamic range
        rms_level = 20 * np.log10(np.sqrt(np.mean(left_channel ** 2)) + 1e-10)
        peak_level = 20 * np.log10(np.max(np.abs(left_channel)) + 1e-10)
        
        # Check for excessive dynamic range
        dynamic_range = peak_level - rms_level
        if dynamic_range > 12:  # More than 12dB dynamic range
            warnings.append(f"Excessive dynamic range: {dynamic_range:.1f}dB (should be <12dB)")
        
        # Check for transient peaks
        # Look for samples that are much louder than local average
        window_size = int(params['sample_rate'] * 0.1)  # 100ms windows
        local_rms = []
        for i in range(0, len(left_channel) - window_size, window_size // 10):
            window = left_channel[i:i + window_size]
            local_rms.append(np.sqrt(np.mean(window ** 2)))
        
        local_rms = np.array(local_rms)
        local_rms_db = 20 * np.log10(local_rms + 1e-10)
        
        # Find peaks in RMS
        peak_threshold = np.mean(local_rms_db) + 6  # 6dB above average
        peaks = local_rms_db > peak_threshold
        
        if np.sum(peaks) > len(peaks) * 0.05:  # More than 5% peaks
            warnings.append(f"Too many transient peaks: {np.sum(peaks)} peaks detected")
    except Exception as e:
        warnings.append(f"Could not analyze dynamic range: {e}")
    
    # Check 4: Stereo correlation (wide but coherent)
    try:
        left_channel = audio[:, 0]
        right_channel = audio[:, 1]
        
        # Calculate correlation
        correlation = np.corrcoef(left_channel, right_channel)[0, 1]
        
        if np.isnan(correlation):
            warnings.append("Could not calculate stereo correlation")
        elif correlation > 0.95:
            warnings.append(f"Stereo channels too similar: {correlation:.3f} (should be <0.95)")
        elif correlation < 0.7:
            warnings.append(f"Stereo channels too different: {correlation:.3f} (should be >0.7)")
        
        # Check stereo balance
        left_rms = np.sqrt(np.mean(left_channel ** 2))
        right_rms = np.sqrt(np.mean(right_channel ** 2))
        balance_diff = 20 * np.log10(left_rms / right_rms)
        
        if abs(balance_diff) > 3:  # More than 3dB imbalance
            warnings.append(f"Stereo imbalance: {balance_diff:.1f}dB (should be <3dB)")
    except Exception as e:
        warnings.append(f"Could not analyze stereo correlation: {e}")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_kleprami_track(seed: Optional[int] = None, output_dir: Optional[str] = None) -> str:
    """
    Generate a complete kleprami track using direct audio generation.
    
    Args:
        seed: Random seed for reproducibility
        output_dir: Directory to save output (defaults to ./kleprami/output)
        
    Returns:
        Path to generated WAV file
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Store seed in params for reproducibility
    params = generate_kleprami_parameters(seed)
    params['seed'] = seed
    
    print(f"Generating kleprami track with seed: {seed}")
    print(f"Parameters: {json.dumps(params, indent=2, default=str)}")
    
    # Generate layers
    builder = DroneLayerBuilder(params, params['duration'], params['sample_rate'])
    layers = builder.render_all()
    
    # TODO: Apply spatial processing and mastering
    # spatial = apply_kleprami_spatial_processing(layers, params)
    # mastered = apply_master_processing(spatial, params)
    
    # For now, just use layers as output
    final_audio = layers
    
    # Validate aesthetic
    is_valid, warnings = validate_kleprami_aesthetic(final_audio, params)
    
    if not is_valid:
        print("⚠️  Aesthetic validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ Aesthetic validation passed")
    
    # Save validation results in metadata
    params['validation_results'] = {
        'is_valid': is_valid,
        'warnings': warnings
    }
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    track_no = len([f for f in os.listdir(output_dir) if f.endswith('.wav')]) + 1
    filename = f"kleprami_{track_no:03d}_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    
    # Save as 24-bit WAV
    sf.write(filepath, final_audio, params['sample_rate'], subtype='PCM_24')
    
    # Save metadata
    metadata_path = filepath.replace('.wav', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(params, f, indent=2, default=str)
    
    print(f"Track saved to: {filepath}")
    print(f"Metadata saved to: {metadata_path}")
    
    return filepath


if __name__ == "__main__":
    # Test generation
    generate_kleprami_track(seed=42)
