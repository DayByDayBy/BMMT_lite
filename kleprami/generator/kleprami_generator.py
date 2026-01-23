"""
kleprami Ambient Drone Generator - Version 2

Fixed version that properly implements frequency modulation and evolving drones.

Key fixes:
- Frequency modulation applied DURING synthesis (not after)
- Multiple evolving modulation sources
- Proper spatial processing implementation
- Texture layers with evolving characteristics
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

from bmmt.audio.oscillators import generate_sine, generate_triangle
from bmmt.audio.noise import generate_pink_noise, generate_brown_noise
from bmmt.modulation.modulation import (
    generate_perlin_drift, generate_exponential_envelope, 
    generate_lfo_sine, generate_random_walk
)
from bmmt.processing.filters import lowpass_filter, highpass_filter, bandpass_filter
from bmmt.processing.effects import apply_tube_warmth, apply_tape_saturation
from bmmt.processing.spatial import (
    apply_reverb, apply_distance_filter, apply_stereo_width,
    apply_air_absorption
)
from bmmt.composition.mixing import (
    combine_signals, normalize_to_peak, apply_master_limiter,
    create_stereo_field, apply_amplitude_modulation
)

# ============================================================================
# PARAMETER SYSTEM
# ============================================================================

KLEPRAMI_RANGES = {
    'base_freq': (35, 85),
    'num_layers': (3, 5),
    'fade_in_duration': (8, 25),
    'modulation_speed': (0.008, 0.025),
    'detuning_amount': (0.15, 0.6),
    'reverb_decay': (4.0, 7.0),
    'distance_factor': (2.5, 4.5),
    'noise_floor': (-42, -36),
    'track_duration_minutes': (3, 5),  # Reduced for testing
    'shimmer_probability': (0.4, 0.7),
    'warmth_amount': (0.3, 0.55),
}


def generate_kleprami_parameters(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate parameters with intelligent constraints."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    params = {
        'seed': seed,
        'sample_rate': 44100,
    }
    
    # Generate base parameters
    for key, (min_val, max_val) in KLEPRAMI_RANGES.items():
        if isinstance(min_val, float):
            params[key] = random.uniform(min_val, max_val)
        else:
            params[key] = random.randint(min_val, max_val)
    
    # Convert duration to seconds
    params['duration'] = params['track_duration_minutes'] * 60
    
    # Apply constraints
    if params['num_layers'] >= 5:
        params['reverb_decay'] = min(params['reverb_decay'], 5.5)
    
    if params['base_freq'] <= 45:
        params['fade_in_duration'] = max(params['fade_in_duration'], 15.0)
    
    # Generate harmonic structure
    params['harmonic_ratios'] = [1.0]  # Fundamental
    harmonic_choices = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for i in range(1, params['num_layers']):
        params['harmonic_ratios'].append(random.choice(harmonic_choices))
    
    # Layer amplitudes (dB)
    params['layer_amps'] = []
    for i in range(params['num_layers']):
        if i == 0:
            amp = random.uniform(-15, -12)
        elif i < 3:
            amp = random.uniform(-20, -16)
        else:
            amp = random.uniform(-26, -22)
        params['layer_amps'].append(amp)
    
    return params


# ============================================================================
# CORE SYNTHESIS FUNCTIONS
# ============================================================================

def synthesize_modulated_drone(
    base_freq: float,
    amplitude_db: float,
    duration: float,
    freq_modulation: np.ndarray,
    amp_modulation: np.ndarray,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Synthesize a drone with continuous frequency and amplitude modulation.
    
    This is the KEY fix - we generate the waveform WITH modulation, not after.
    """
    num_samples = int(duration * sample_rate)
    
    # Ensure modulation arrays are correct length
    if len(freq_modulation) != num_samples:
        freq_modulation = np.interp(
            np.linspace(0, 1, num_samples),
            np.linspace(0, 1, len(freq_modulation)),
            freq_modulation
        )
    
    if len(amp_modulation) != num_samples:
        amp_modulation = np.interp(
            np.linspace(0, 1, num_samples),
            np.linspace(0, 1, len(amp_modulation)),
            amp_modulation
        )
    
    # Create time-varying frequency
    instantaneous_freq = base_freq + freq_modulation
    
    # Generate phase from frequency (this is the key to FM)
    phase = np.cumsum(2 * np.pi * instantaneous_freq / sample_rate)
    
    # Generate waveform (sine for smoothness)
    waveform = np.sin(phase)
    
    # Convert amplitude from dB to linear
    amplitude_linear = 10 ** (amplitude_db / 20)
    
    # Apply amplitude modulation
    waveform = waveform * amplitude_linear * (1.0 + amp_modulation)
    
    return waveform


def generate_evolving_drone_layer(
    base_freq: float,
    harmonic_ratio: float,
    amplitude_db: float,
    duration: float,
    modulation_speed: float,
    detuning_amount: float,
    layer_index: int,
    seed: Optional[int] = None,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Generate a single evolving drone layer.
    Uses proper frequency modulation synthesis.
    """
    # Generate frequency modulation (evolving over time)
    # Use multiple perlin noise sources at different speeds for complexity
    slow_drift = generate_perlin_drift(
        duration,
        frequency=modulation_speed,
        amplitude=detuning_amount * 0.7,
        sample_rate=sample_rate,
        seed=seed + layer_index * 1000 if seed else None
    )
    
    medium_drift = generate_perlin_drift(
        duration,
        frequency=modulation_speed * 2.5,
        amplitude=detuning_amount * 0.3,
        sample_rate=sample_rate,
        seed=seed + layer_index * 1000 + 100 if seed else None
    )
    
    # Combine drifts (slow becomes medium over time)
    num_samples = int(duration * sample_rate)
    crossfade = np.linspace(0, 1, num_samples) ** 0.5  # Gradual transition
    freq_mod = slow_drift * (1 - crossfade) + medium_drift * crossfade
    
    # Generate amplitude modulation (very subtle)
    amp_drift = generate_perlin_drift(
        duration,
        frequency=modulation_speed * 0.5,
        amplitude=0.08,
        sample_rate=sample_rate,
        seed=seed + layer_index * 1000 + 200 if seed else None
    )
    
    # Synthesize with modulation
    layer_freq = base_freq * harmonic_ratio
    layer = synthesize_modulated_drone(
        layer_freq,
        amplitude_db,
        duration,
        freq_mod,
        amp_drift,
        sample_rate
    )
    
    return layer


def generate_shimmer_layer(
    base_freq: float,
    duration: float,
    modulation_speed: float,
    seed: Optional[int] = None,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Generate high-frequency shimmer texture.
    Uses filtered noise with evolving characteristics.
    """
    # Generate pink noise as base
    noise = generate_pink_noise(-32, duration, sample_rate)
    
    # Filter to high frequencies (shimmer range)
    shimmer_freq = base_freq * random.uniform(12, 20)
    bandwidth = random.uniform(800, 1500)
    noise = bandpass_filter(noise, shimmer_freq, bandwidth, sample_rate)
    
    # Evolving amplitude modulation
    num_samples = int(duration * sample_rate)
    
    # Multiple modulation sources at different speeds
    mod1 = generate_perlin_drift(
        duration,
        frequency=modulation_speed * 3.0,
        amplitude=0.4,
        sample_rate=sample_rate,
        seed=seed + 9000 if seed else None
    )
    
    mod2 = generate_perlin_drift(
        duration,
        frequency=modulation_speed * 7.0,
        amplitude=0.3,
        sample_rate=sample_rate,
        seed=seed + 9100 if seed else None
    )
    
    # Combine with evolving balance
    crossfade = np.linspace(0, 1, num_samples) ** 2
    combined_mod = mod1 * (1 - crossfade) + mod2 * crossfade
    
    # Apply modulation
    noise = noise * (0.3 + combined_mod * 0.7)
    
    return noise


def generate_texture_layer(
    base_freq: float,
    duration: float,
    noise_floor_db: float,
    seed: Optional[int] = None,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Generate subtle texture layer using filtered noise.
    """
    # Use brown noise for warm, organic texture
    noise = generate_brown_noise(noise_floor_db, duration, sample_rate)
    
    # Filter to complement drone frequencies
    cutoff = base_freq * random.uniform(4, 8)
    noise = lowpass_filter(noise, cutoff, sample_rate)
    
    # Very slow amplitude variation
    variation = generate_random_walk(
        duration,
        step_size=0.02,
        sample_rate=sample_rate,
        seed=seed + 8000 if seed else None
    )
    
    noise = noise * (1.0 + variation * 0.15)
    
    return noise


# ============================================================================
# SPATIAL PROCESSING
# ============================================================================

def apply_kleprami_spatial_processing(
    audio: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Apply spatial processing to create vast, ancient spaces.
    """
    sample_rate = params['sample_rate']
    
    # Create stereo field (already stereo from mixing)
    if audio.ndim == 1:
        audio = create_stereo_field([audio], [0.0], width=1.2)
    
    # # Apply distance filtering (muffles high frequencies)
    # audio = apply_distance_filter(audio, params['distance_factor'], sample_rate)
    
    # Apply air absorption (simulates sound traveling through air)
    distance_meters = max(1.0, min(100.0, params['distance_factor'] * 50))
    audio = apply_air_absorption(audio, distance_meters, 0.3, sample_rate)
    
    # Apply reverb (vast spaces)
    room_size = 0.7 + (params['reverb_decay'] - 4.0) / 4.0 * 0.3
    audio = apply_reverb(
        audio,
        room_size=room_size,
        decay_time=params['reverb_decay'],
        damping=0.5,
        sample_rate=sample_rate
    )
    
    # Widen stereo field
    audio = apply_stereo_width(audio, width=1.3)
    
    return audio


def apply_master_processing(
    audio: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Apply final mastering chain.
    """
    sample_rate = params['sample_rate']
    
    # Remove subsonic content
    audio_left = highpass_filter(audio[:, 0], 25, sample_rate)
    audio_right = highpass_filter(audio[:, 1], 25, sample_rate)
    audio = np.column_stack([audio_left, audio_right])
    
    # Gentle tube warmth
    audio = apply_tube_warmth(audio, drive=1.0 + params['warmth_amount'], asymmetry=0.08)
    
    # Subtle tape saturation
    audio = apply_tape_saturation(audio, drive=1.2, warmth=0.3)
    
    # Normalize to -12dB (leaving headroom)
    audio = normalize_to_peak(audio, -12.0)
    
    # Soft limiting (prevent any clipping)
    audio = apply_master_limiter(audio, threshold_db=-6.0, ratio=3.0)
    
    return audio


# ============================================================================
# FADE ENVELOPE
# ============================================================================

def generate_kleprami_envelope(
    duration: float,
    fade_in_duration: float,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Generate fade envelope for entire track.
    Very slow fade in, sustained, very slow fade out.
    """
    num_samples = int(duration * sample_rate)
    
    # Fade in (exponential for naturalness)
    fade_in = generate_exponential_envelope(
        -60.0, 0.0, fade_in_duration, curve=2.0, sample_rate=sample_rate,
    )
    
    # Fade out (last 10% of track, or 30 seconds, whichever is longer)
    fade_out_duration = max(duration * 0.1, 30.0)
    fade_out = generate_exponential_envelope(
        0.0, -60.0, fade_out_duration, curve=2.0, sample_rate=sample_rate,
    )
    
    # Sustained section
    sustain_duration = duration - fade_in_duration - fade_out_duration
    if sustain_duration < 0:
        # Track too short for this envelope structure
        sustain_samples = 0
        fade_in_duration = duration * 0.3
        fade_out_duration = duration * 0.3
        fade_in = generate_exponential_envelope(
            0.0, 1.0, fade_in_duration, curve=3.0, sample_rate=sample_rate
        )
        fade_out = generate_exponential_envelope(
            1.0, 0.0, fade_out_duration, curve=3.0, sample_rate=sample_rate
        )
    else:
        sustain_samples = int(sustain_duration * sample_rate)
    
    sustain = np.ones(sustain_samples)
    
    # Combine
    envelope = np.concatenate([fade_in, sustain, fade_out])
    
    # Ensure correct length
    if len(envelope) > num_samples:
        envelope = envelope[:num_samples]
    elif len(envelope) < num_samples:
        # Pad with ones
        pad_length = num_samples - len(envelope)
        envelope = np.concatenate([envelope, np.ones(pad_length)])
    
    return envelope


# ============================================================================
# MAIN GENERATION
# ============================================================================

def generate_kleprami_track(
    seed: Optional[int] = None,
    output_dir: Optional[str] = None
) -> str:
    """
    Generate complete kleprami track.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate parameters
    params = generate_kleprami_parameters(seed)
    
    print(f"\n{'='*60}")
    print(f"Generating kleprami track")
    print(f"{'='*60}")
    print(f"Seed: {seed}")
    print(f"Duration: {params['duration']/60:.1f} minutes")
    print(f"Base frequency: {params['base_freq']:.1f} Hz")
    print(f"Layers: {params['num_layers']}")
    print(f"{'='*60}\n")
    
    duration = params['duration']
    sample_rate = params['sample_rate']
    
    # Generate all layers
    print("Generating drone layers...")
    layers = []
    
    for i in range(params['num_layers']):
        print(f"  Layer {i+1}/{params['num_layers']}: "
              f"{params['base_freq'] * params['harmonic_ratios'][i]:.1f} Hz")
        
        layer = generate_evolving_drone_layer(
            base_freq=params['base_freq'],
            harmonic_ratio=params['harmonic_ratios'][i],
            amplitude_db=params['layer_amps'][i],
            duration=duration,
            modulation_speed=params['modulation_speed'],
            detuning_amount=params['detuning_amount'],
            layer_index=i,
            seed=params['seed'],
            sample_rate=sample_rate
        )
        layers.append(layer)
    
    # Add shimmer layer (probabilistic)
    if random.random() < params['shimmer_probability']:
        print("  Adding shimmer layer...")
        shimmer = generate_shimmer_layer(
            base_freq=params['base_freq'],
            duration=duration,
            modulation_speed=params['modulation_speed'],
            seed=params['seed'],
            sample_rate=sample_rate
        )
        layers.append(shimmer)
    
    # Add texture layer
    print("  Adding texture layer...")
    texture = generate_texture_layer(
        base_freq=params['base_freq'],
        duration=duration,
        noise_floor_db=params['noise_floor'],
        seed=params['seed'],
        sample_rate=sample_rate
    )
    layers.append(texture)
    
    # Mix layers
    print("\nMixing layers...")
    mix_levels = [1.0] * len(layers)
    mixed = combine_signals(layers, mix_levels, sample_rate)
    
    # Apply master envelope
    print("Applying master envelope...")
    envelope = generate_kleprami_envelope(
        duration,
        params['fade_in_duration'],
        sample_rate
    )
    mixed = mixed * envelope
    
    # Convert to stereo
    stereo = create_stereo_field([mixed], [0.0], width=1.0)
    
    # Apply spatial processing
    print("Applying spatial processing...")
    spatial = apply_kleprami_spatial_processing(stereo, params)
    
    # Master processing
    print("Applying master processing...")
    final = apply_master_processing(spatial, params)
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    existing_tracks = [f for f in os.listdir(output_dir) if f.startswith('kleprami_')]
    track_no = len(existing_tracks) + 1
    filename = f"kleprami_{track_no:03d}_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    
    # Save audio
    print(f"\nSaving to {filename}...")
    sf.write(filepath, final, sample_rate, subtype='PCM_24')
    
    # Save metadata
    metadata_path = filepath.replace('.wav', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(params, f, indent=2, default=str)
    
    # Generate human-readable notes
    notes_path = filepath.replace('.wav', '_notes.txt')
    with open(notes_path, 'w') as f:
        f.write(f"kleprami Track {track_no:03d}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Duration: {params['duration']/60:.1f} minutes\n\n")
        f.write(f"Sonic Character:\n")
        f.write(f"  Base frequency: {params['base_freq']:.1f} Hz ")
        f.write(f"({'sub-bass focus' if params['base_freq'] < 50 else 'low-mid focus'})\n")
        f.write(f"  Layers: {params['num_layers']}\n")
        f.write(f"  Harmonics: {params['harmonic_ratios']}\n")
        f.write(f"  Detuning: {params['detuning_amount']:.2f} Hz ")
        f.write(f"({'subtle' if params['detuning_amount'] < 0.3 else 'prominent'} beating)\n")
        f.write(f"  Modulation speed: {params['modulation_speed']:.3f} Hz ")
        f.write(f"({1/params['modulation_speed']:.0f}s period)\n\n")
        f.write(f"Spatial Character:\n")
        f.write(f"  Reverb decay: {params['reverb_decay']:.1f}s ")
        f.write(f"({'intimate' if params['reverb_decay'] < 5 else 'vast'} space)\n")
        f.write(f"  Distance: {params['distance_factor']:.1f} ")
        f.write(f"({'close' if params['distance_factor'] < 3 else 'distant'})\n")
        f.write(f"  Shimmer layer: {'present' if 'shimmer' in str(layers) else 'absent'}\n")
    
    print(f"\n{'='*60}")
    print(f"✅ Generation complete!")
    print(f"{'='*60}")
    print(f"Audio: {filepath}")
    print(f"Metadata: {metadata_path}")
    print(f"Notes: {notes_path}")
    print(f"{'='*60}\n")
    
    return filepath


if __name__ == "__main__":
    # Quick test with short duration
    generate_kleprami_track()