#!/usr/bin/env python3
"""
Generate sample audio files demonstrating each reverb type in BMMT.
Uses other modules to create interesting dry sounds before applying reverb.
"""

import numpy as np
import soundfile as sf
import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')

from bmmt.processing import (
    apply_spring_reverb,
    apply_plate_reverb,
    apply_hall_reverb,
    apply_shimmer_reverb,
    apply_gated_reverb,
)

from bmmt.audio.oscillators import generate_sine, generate_sawtooth, generate_square
from bmmt.audio.drums import synthesize_kick, synthesize_snare, synthesize_hihat
from bmmt.modulation.modulation import generate_adsr_envelope
from bmmt.composition.mixing import combine_signals


def create_test_pad(duration=2.0, sample_rate=44100):
    """Create a warm pad using saw oscillators with envelope."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Create chord (C major)
    freqs = [261.63, 329.63, 392.00]  # C4, E4, G4
    
    pad = np.zeros(int(duration * sample_rate))
    for freq in freqs:
        osc = generate_sawtooth(freq, -6.0, duration, sample_rate) * 0.3
        pad += osc
    
    # Apply slow envelope
    envelope = generate_adsr_envelope(
        0.5, 0.3, 0.7, 0.5, duration, sample_rate
    )
    return pad * envelope * 0.5


def create_test_pluck(duration=1.5, sample_rate=44100):
    """Create a plucky sound using square wave with fast envelope."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Create arpeggio pattern
    freqs = [440.0, 554.37, 659.25, 880.0]  # A4, C#5, E5, A5
    
    pluck = np.zeros(int(duration * sample_rate))
    note_duration = duration / len(freqs)
    
    for i, freq in enumerate(freqs):
        start_idx = int(i * note_duration * sample_rate)
        end_idx = int((i + 1) * note_duration * sample_rate)
        note_len = end_idx - start_idx
        
        osc = generate_square(freq, -9.0, note_duration, sample_rate) * 0.4
        
        # Fast envelope for pluck
        env = generate_adsr_envelope(
            0.01, 0.1, 0.0, 0.05, note_duration, sample_rate
        )
        pluck[start_idx:end_idx] += osc * env
    
    return pluck * 0.5


def create_test_drum_loop(duration=2.0, sample_rate=44100):
    """Create a simple drum loop."""
    loop = np.zeros(int(duration * sample_rate))
    
    # Basic 4/4 pattern: kick on 1, snare on 3, hihats on 1/8ths
    beat_duration = 0.5  # 120 BPM
    eighth_duration = beat_duration / 2
    
    for beat in range(int(duration / beat_duration)):
        beat_start = int(beat * beat_duration * sample_rate)
        
        # Kick on beats 1 and 3
        if beat % 2 == 0:
            kick = synthesize_kick(fundamental=60, pitch_start=150, decay_ms=120, sample_rate=sample_rate) * 0.8
            kick_end = min(len(loop), beat_start + len(kick))
            loop[beat_start:kick_end] += kick[:kick_end - beat_start]
        
        # Snare on beats 2 and 4
        else:
            snare = synthesize_snare(tone_freq=200, decay_ms=80, sample_rate=sample_rate) * 0.6
            snare_end = min(len(loop), beat_start + len(snare))
            loop[beat_start:snare_end] += snare[:snare_end - beat_start]
        
        # Hihats on eighth notes
        for eighth in range(4):
            hihat_start = beat_start + int(eighth * eighth_duration * sample_rate)
            hihat = synthesize_hihat(decay_ms=40, sample_rate=sample_rate) * 0.3
            hihat_end = min(len(loop), hihat_start + len(hihat))
            loop[hihat_start:hihat_end] += hihat[:hihat_end - hihat_start]
    
    return loop * 0.6


def create_test_bass(duration=2.0, sample_rate=44100):
    """Create a simple bassline."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Simple bass pattern (C minor)
    freqs = [65.41, 65.41, 78.39, 87.31]  # C2, C2, G#2, A2
    
    bass = np.zeros(int(duration * sample_rate))
    note_duration = duration / len(freqs)
    
    for i, freq in enumerate(freqs):
        start_idx = int(i * note_duration * sample_rate)
        end_idx = int((i + 1) * note_duration * sample_rate)
        note_len = end_idx - start_idx
        
        osc = generate_sine(freq, -3.0, note_duration, sample_rate) * 0.7
        
        # Subtle envelope
        env = generate_adsr_envelope(
            0.02, 0.1, 0.3, 0.1, note_duration, sample_rate
        )
        bass[start_idx:end_idx] += osc * env
    
    return bass * 0.5


def generate_reverb_samples():
    """Generate and save samples for each reverb type."""
    sample_rate = 44100
    duration = 2.0
    
    # Create output directory
    output_dir = "reverb_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating reverb samples in '{output_dir}/'...")
    
    # Use plucky sound for all reverbs for easy comparison
    dry_pluck = create_test_pluck(duration, sample_rate)
    
    # 1. Spring Reverb
    print("\n1. Spring Reverb (plucky arpeggio)...")
    wet_spring = apply_spring_reverb(
        dry_pluck,
        mix=0.4,
        springs=3,
        tension=0.6,
        shake=0.15,
        bass=0.5,
        treble=0.8,
        sample_rate=sample_rate
    )
    sf.write(f"{output_dir}/spring_reverb.wav", wet_spring, sample_rate)
    
    # 2. Plate Reverb
    print("2. Plate Reverb (plucky arpeggio)...")
    wet_plate = apply_plate_reverb(
        dry_pluck,
        decay_time=2.5,
        pre_delay=0.015,
        bass_damping=0.4,
        treble_damping=0.7,
        mix=0.35,
        size=0.7,
        diffusion=0.85,
        sample_rate=sample_rate
    )
    sf.write(f"{output_dir}/plate_reverb.wav", wet_plate, sample_rate)
    
    # 3. Hall Reverb
    print("3. Hall Reverb (plucky arpeggio)...")
    wet_hall = apply_hall_reverb(
        dry_pluck,
        size=0.9,
        decay_time=3.5,
        damping=0.3,
        diffusion=0.8,
        pre_delay=0.02,
        mix=0.4,
        sample_rate=sample_rate
    )
    sf.write(f"{output_dir}/hall_reverb.wav", wet_hall, sample_rate)
    
    # 4. Shimmer Reverb
    print("4. Shimmer Reverb (plucky arpeggio)...")
    wet_shimmer = apply_shimmer_reverb(
        dry_pluck,
        size=0.85,
        diffusion=0.75,
        decay=6.0,
        feedback=0.5,
        mix=0.35,
        pitch_a_semitones=12.0,  # Octave up
        pitch_b_semitones=0.0,
        pitch_blend=0.8,
        modulation=0.4,
        tone=0.7,
        sample_rate=sample_rate
    )
    sf.write(f"{output_dir}/shimmer_reverb.wav", wet_shimmer, sample_rate)
    
    # 5. Gated Reverb
    print("5. Gated Reverb (plucky arpeggio)...")
    wet_gated = apply_gated_reverb(
        dry_pluck,
        threshold=0.15,
        attack=0.003,
        hold=0.08,
        release=0.12,
        reverb_mix=0.5,
        reverb_decay=1.8,
        size=0.75,
        diffusion=0.7,
        sample_rate=sample_rate
    )
    sf.write(f"{output_dir}/gated_reverb.wav", wet_gated, sample_rate)
    
    # Save dry reference for comparison
    print("\nSaving dry reference signal...")
    sf.write(f"{output_dir}/dry_pluck.wav", dry_pluck, sample_rate)
    
    print(f"\n✓ All samples saved to '{output_dir}/'")
    print("\nGenerated files (all using plucky arpeggio):")
    print("- spring_reverb.wav")
    print("- plate_reverb.wav")
    print("- hall_reverb.wav")
    print("- shimmer_reverb.wav")
    print("- gated_reverb.wav")
    print("\nDry reference file:")
    print("- dry_pluck.wav")


if __name__ == "__main__":
    try:
        generate_reverb_samples()
        print("\n🎛️ Reverb samples generated successfully!")
        print("Listen to compare different reverb characteristics.")
    except Exception as e:
        print(f"❌ Error generating samples: {e}")
        import traceback
        traceback.print_exc()
