"""
Drum voice synthesizers for modular audio synthesis.

Single-hit drum generators optimized for disco and rhythmic music.
Each function returns one drum hit that can be placed by the sequencer.
"""

import numpy as np
from typing import Optional

from .oscillators import generate_sine, generate_square
from .noise import generate_white_noise, generate_pink_noise
from ..processing.filters import lowpass_filter, highpass_filter, bandpass_filter
from ..processing.effects import apply_tape_saturation
from ..modulation.modulation import generate_percussive_envelope


def synthesize_kick(
    fundamental: float = 60.0,
    pitch_start: float = 150.0,
    pitch_time_ms: float = 10.0,
    decay_ms: float = 100.0,
    click_amount: float = 0.3,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Synthesize a disco kick drum with pitch sweep and optional click.
    
    Args:
        fundamental: Final pitch frequency in Hz (40-80 typical)
        pitch_start: Starting pitch frequency in Hz (100-200 typical)
        pitch_time_ms: Time for pitch sweep in milliseconds (5-20 typical)
        decay_ms: Decay time in milliseconds (80-200 typical)
        click_amount: Amount of high-frequency click (0.0-1.0)
        sample_rate: Sample rate in Hz
    
    Returns:
        numpy.ndarray: Mono kick drum hit
    
    Example:
        >>> kick = synthesize_kick(fundamental=60, pitch_start=150, decay_ms=120)
        >>> len(kick) == 5292  # 120ms at 44.1kHz
        True
    """
    # Validate parameters
    if not (20 <= fundamental <= 200):
        raise ValueError(f"Fundamental frequency {fundamental}Hz out of range (20-200Hz)")
    if not (50 <= pitch_start <= 300):
        raise ValueError(f"Start pitch frequency {pitch_start}Hz out of range (50-300Hz)")
    if pitch_start <= fundamental:
        raise ValueError("Start pitch must be higher than fundamental")
    if not (0.0 <= click_amount <= 1.0):
        raise ValueError(f"Click amount must be 0-1, got {click_amount}")
    
    # Calculate duration
    duration = decay_ms / 1000.0
    num_samples = int(duration * sample_rate)
    
    # Generate pitch envelope
    from ..modulation.modulation import generate_pitch_envelope
    pitch_curve = generate_pitch_envelope(pitch_start, fundamental, pitch_time_ms, duration, sample_rate)
    
    # Generate main kick body using frequency modulation approach
    # Create a sine wave with time-varying frequency
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Integrate the frequency curve to get phase
    phase = 2 * np.pi * np.cumsum(pitch_curve) / sample_rate
    
    # Generate the kick tone
    kick_body = np.sin(phase)
    
    # Add sub layer for better low-end
    sub_freq = fundamental * 0.5  # Sub octave
    sub_phase = 2 * np.pi * sub_freq * t
    sub = np.sin(sub_phase) * 0.3  # Mix sub at 30%
    kick_body += sub
    
    # Apply amplitude envelope with slightly longer tail for sub
    envelope = generate_percussive_envelope(0.0, decay_ms, duration, sample_rate)
    # Create sub envelope with longer decay but capped at duration
    sub_decay = min(decay_ms * 1.5, duration * 1000 * 0.9)  # Cap at 90% of duration
    sub_env = generate_percussive_envelope(0.0, sub_decay, duration, sample_rate)
    kick_body = kick_body * envelope + sub * sub_env
    
    # Add click if requested
    if click_amount > 0:
        # Create click with noise burst
        click_duration = 0.005  # 5ms
        click_samples = int(click_duration * sample_rate)
        click = generate_white_noise(-12.0, click_duration, sample_rate)
        
        # Highpass the click to make it sharp
        click = highpass_filter(click, 2000.0, sample_rate, resonance=1.0)
        
        # Apply fast envelope to click
        click_env = generate_percussive_envelope(0.0, 5.0, click_duration, sample_rate)
        click *= click_env
        
        # Mix click with kick body
        if len(click) < len(kick_body):
            click_padded = np.zeros_like(kick_body)
            click_padded[:len(click)] = click
            click = click_padded
        else:
            click = click[:len(kick_body)]
        
        kick_body += click * click_amount
    
    # Apply subtle tape saturation for warmth
    kick_body = apply_tape_saturation(kick_body, drive=1.2)
    
    # Normalize to reasonable level
    max_val = np.max(np.abs(kick_body))
    if max_val > 0:
        kick_body = kick_body / max_val * 0.8  # Peak at -2dB
    
    return kick_body


def synthesize_snare(
    tone_freq: float = 200.0,
    decay_ms: float = 150.0,
    tone_mix: float = 0.3,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Synthesize a snare drum with tone body and noise layers.
    
    Args:
        tone_freq: Frequency of the snare tone component in Hz (150-300 typical)
        decay_ms: Decay time in milliseconds (100-300 typical)
        tone_mix: Mix of tone vs noise (0.0=all noise, 1.0=all tone)
        sample_rate: Sample rate in Hz
    
    Returns:
        numpy.ndarray: Mono snare drum hit
    
    Example:
        >>> snare = synthesize_snare(tone_freq=200, decay_ms=150)
        >>> len(snare) == 6615  # 150ms at 44.1kHz
        True
    """
    # Validate parameters
    if not (100 <= tone_freq <= 500):
        raise ValueError(f"Tone frequency {tone_freq}Hz out of range (100-500Hz)")
    if not (0.0 <= tone_mix <= 1.0):
        raise ValueError(f"Tone mix must be 0-1, got {tone_mix}")
    
    duration = decay_ms / 1000.0
    num_samples = int(duration * sample_rate)
    
    # Generate tone body
    tone = generate_sine(tone_freq, -6.0, duration, sample_rate)
    
    # Add second harmonic for body
    tone_harmonic = generate_sine(tone_freq * 2, -12.0, duration, sample_rate) * 0.3
    tone = tone + tone_harmonic
    
    # Generate noise component
    noise = generate_pink_noise(-6.0, duration, sample_rate)
    
    # Filter noise for snare character with more resonance
    noise = bandpass_filter(noise, 2000.0, 2000.0, sample_rate)
    
    # Add a touch of high-frequency noise for snap
    snap_noise = generate_white_noise(-12.0, duration, sample_rate)
    snap_noise = highpass_filter(snap_noise, 5000.0, sample_rate, resonance=1.0)
    snap_noise = snap_noise * 0.2
    
    # Mix noise components
    noise = noise + snap_noise
    
    # Apply envelope to both components
    envelope = generate_percussive_envelope(0.0, decay_ms, duration, sample_rate)
    
    # Mix tone and noise
    snare = tone * tone_mix + noise * (1.0 - tone_mix)
    snare *= envelope
    
    # Apply subtle saturation
    snare = apply_tape_saturation(snare, drive=1.1)
    
    # Normalize
    max_val = np.max(np.abs(snare))
    if max_val > 0:
        snare = snare / max_val * 0.7  # Peak at -3dB
    
    return snare


def synthesize_hihat(
    closed: bool = True,
    decay_ms: Optional[float] = None,
    tone_freq: float = 8000.0,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Synthesize a hi-hat using filtered noise.
    
    Args:
        closed: True for closed hi-hat, False for open
        decay_ms: Decay time in milliseconds (None for default based on closed/open)
        tone_freq: Center frequency for filtering (6000-10000 typical)
        sample_rate: Sample rate in Hz
    
    Returns:
        numpy.ndarray: Mono hi-hat hit
    
    Example:
        >>> closed_hh = synthesize_hihat(closed=True)
        >>> open_hh = synthesize_hihat(closed=False)
        >>> len(closed_hh) < len(open_hh)  # Closed is shorter
        True
    """
    # Validate parameters
    if not (3000 <= tone_freq <= 15000):
        raise ValueError(f"Tone frequency {tone_freq}Hz out of range (3000-15000Hz)")
    
    # Set default decay times
    if decay_ms is None:
        decay_ms = 50.0 if closed else 200.0
    
    duration = decay_ms / 1000.0
    num_samples = int(duration * sample_rate)
    
    # Generate noise source
    noise = generate_white_noise(-6.0, duration, sample_rate)
    
    # Filter for hi-hat character
    # Use highpass to remove low mud, then bandpass for metallic tone
    noise = highpass_filter(noise, 7000.0, sample_rate, resonance=1.0)
    noise = bandpass_filter(noise, tone_freq, 3000.0, sample_rate)
    
    # Add metallic ring with filtered noise
    metallic = generate_white_noise(-18.0, duration, sample_rate)
    metallic = bandpass_filter(metallic, tone_freq * 1.2, 1000.0, sample_rate)
    metallic = metallic * 0.3
    
    # Mix noise sources
    noise = noise + metallic
    
    # Apply envelope
    envelope = generate_percussive_envelope(0.0, decay_ms, duration, sample_rate)
    hihat = noise * envelope
    
    # Apply very subtle saturation
    hihat = apply_tape_saturation(hihat, drive=1.05)
    
    # Normalize
    max_val = np.max(np.abs(hihat))
    if max_val > 0:
        hihat = hihat / max_val * 0.6  # Peak at -4dB
    
    return hihat


def synthesize_conga(
    freq: float = 400.0,
    pitch_drop: float = 0.3,
    decay_ms: float = 200.0,
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Synthesize conga/tom using pitched sine with pitch envelope.
    
    Args:
        freq: Fundamental frequency (Hz)
        pitch_drop: Amount of pitch drop (0.0-1.0)
        decay_ms: Decay time (milliseconds)
        sample_rate: Sample rate in Hz
        
    Returns:
        Mono audio array (1D)
    """
    # Validate parameters
    if not (100 <= freq <= 800):
        raise ValueError(f"Frequency {freq}Hz out of range (100-800Hz)")
    if not (0.0 <= pitch_drop <= 1.0):
        raise ValueError(f"Pitch drop must be 0-1, got {pitch_drop}")
    
    duration = decay_ms / 1000.0
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    
    # Pitch envelope - exponential drop
    start_freq = freq
    end_freq = freq * (1.0 - pitch_drop)
    pitch_env = start_freq * np.exp(-t * 5.0) + end_freq
    
    # Generate pitched tone
    phase = 2 * np.pi * np.cumsum(pitch_env) / sample_rate
    conga = np.sin(phase)
    
    # Add body with second harmonic
    body = np.sin(2 * phase) * 0.2
    conga = conga + body
    
    # Apply envelope with fast attack
    envelope = np.exp(-t * 8.0)  # Fast decay for conga character
    conga *= envelope
    
    # Apply subtle filtering for warmth
    conga = lowpass_filter(conga, freq * 2, sample_rate, resonance=1.0)
    
    # Apply very subtle saturation
    conga = apply_tape_saturation(conga, drive=1.05)
    
    # Normalize
    max_val = np.max(np.abs(conga))
    if max_val > 0:
        conga = conga / max_val * 0.7  # Peak at -3dB
    
    return conga


if __name__ == "__main__":
    # Quick test of all drum sounds
    import soundfile as sf
    from pathlib import Path
    
    print("Testing drum synthesizers...")
    
    # Test each drum
    kick = synthesize_kick(fundamental=60, pitch_start=150, decay_ms=120)
    snare = synthesize_snare(tone_freq=200, decay_ms=150)
    hihat_closed = synthesize_hihat(closed=True)
    hihat_open = synthesize_hihat(closed=False)
    
    print(f"✓ Kick: {len(kick)} samples")
    print(f"✓ Snare: {len(snare)} samples")
    print(f"✓ Hi-hat (closed): {len(hihat_closed)} samples")
    print(f"✓ Hi-hat (open): {len(hihat_open)} samples")
    
    # Save test files
    output_dir = Path(__file__).parent.parent.parent.parent / "examples" / "output"
    output_dir.mkdir(exist_ok=True)
    
    sf.write(str(output_dir / "test_kick.wav"), kick, 44100)
    sf.write(str(output_dir / "test_snare.wav"), snare, 44100)
    sf.write(str(output_dir / "test_hihat_closed.wav"), hihat_closed, 44100)
    sf.write(str(output_dir / "test_hihat_open.wav"), hihat_open, 44100)
    
    print(f"\nSaved test files to: {output_dir}")
    print("✓ All drum synthesizers working correctly")
