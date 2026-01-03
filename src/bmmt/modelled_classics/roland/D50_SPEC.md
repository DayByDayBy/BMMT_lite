# D-50 Linear Arithmetic Synthesizer Specification

Technical specification for a Roland D-50-esque synthesizer module. This document defines architecture, signal flow, data structures, math, and public interfaces sufficient for implementation.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Hierarchy](#architecture-hierarchy)
3. [Partial](#partial)
4. [Tone](#tone)
5. [Patch](#patch)
6. [Envelopes](#envelopes)
7. [PCM Handling](#pcm-handling)
8. [Waveform Generation](#waveform-generation)
9. [Filters](#filters)
10. [Effects](#effects)
11. [Public API](#public-api)
12. [Validation and Error Handling](#validation-and-error-handling)
13. [Implementation Notes](#implementation-notes)

---

## Overview

### Synthesis Method

Linear Arithmetic (LA) synthesis combines:
- Short PCM attack transients (one-shot samples)
- Synthesized sustain waveforms (saw, square, pulse, noise)

The PCM portion provides realistic attack characteristics. The synthesized portion provides sustaining body. This produces the characteristic bright, glassy, digital sound of the D-50.

### Design Principles

- Fixed signal flow (no arbitrary modulation routing)
- Offline clip-based rendering (not real-time streaming)
- Deterministic output for identical inputs
- Naive DSP where appropriate (no oversampling, no anti-aliasing)
- numpy-based audio-rate processing

### Signal Flow Summary

```
Partial 1 ─┐
           ├─► Tone A ─┐
Partial 2 ─┘           │
                       ├─► Patch ─► Chorus ─► Reverb ─► Output
Partial 3 ─┐           │
           ├─► Tone B ─┘
Partial 4 ─┘
```

---

## Architecture Hierarchy

### Hierarchy Levels

| Level   | Contains              | Count |
|---------|-----------------------|-------|
| Patch   | 2 Tones               | 1     |
| Tone    | 2 Partials            | 2     |
| Partial | 1 Sound Source        | 4     |

### Rendering Order

1. Render all 4 Partials independently
2. Mix Partials 1+2 into Tone A
3. Mix Partials 3+4 into Tone B
4. Combine Tones A+B according to Patch structure
5. Apply Chorus
6. Apply Reverb
7. Output stereo signal

---

## Partial

A Partial is the fundamental sound generator. Each Partial produces a mono signal.

### Partial Configuration

```python
@dataclass
class PartialConfig:
    # Source selection
    source_type: Literal["pcm", "synth"]
    
    # PCM source (if source_type == "pcm")
    pcm_data: Optional[np.ndarray] = None  # Mono float32, normalized [-1, 1]
    pcm_sample_rate: Optional[int] = None  # Original sample rate of PCM
    
    # Synth source (if source_type == "synth")
    waveform: Literal["saw", "square", "pulse", "noise"] = "saw"
    pulse_width: float = 0.5  # 0.0-1.0, only for pulse waveform
    
    # Pitch
    coarse_tune: int = 0       # Semitones, -24 to +24
    fine_tune: float = 0.0     # Cents, -50.0 to +50.0
    
    # Envelopes
    pitch_envelope: PitchEnvelopeConfig
    tva_envelope: TVAEnvelopeConfig
    tvf_envelope: Optional[TVFEnvelopeConfig] = None  # None = filter bypassed
    
    # Filter
    tvf_cutoff: float = 1.0    # Normalized 0.0-1.0 (maps to 100Hz-20kHz)
    tvf_resonance: float = 0.0 # Normalized 0.0-1.0
    tvf_enabled: bool = False
    
    # Key scaling
    key_follow_pitch: float = 1.0   # 0.0 = fixed pitch, 1.0 = normal tracking
    key_follow_tvf: float = 0.0     # Cutoff tracking with pitch
    key_follow_tva: float = 0.0     # Amplitude scaling with pitch
    
    # Output
    level: float = 0.0         # dB, -inf to 0.0
    pan: float = 0.0           # -1.0 (left) to +1.0 (right)
```

### Partial Signal Flow

```
[PCM or Synth Source]
        │
        ▼
  [Pitch Envelope] ──► Pitch modulation
        │
        ▼
  [TVF (Low-pass)] ──► Cutoff from TVF Envelope
        │
        ▼
  [TVA (Amplitude)] ──► Level from TVA Envelope
        │
        ▼
  [Pan + Level]
        │
        ▼
    Output (mono)
```

### Partial Rendering

```python
def render_partial(
    config: PartialConfig,
    note: int,
    velocity: float,
    duration_samples: int,
    sample_rate: int
) -> np.ndarray:
    """
    Render a single Partial.
    
    Args:
        config: Partial configuration
        note: MIDI note number (0-127)
        velocity: Normalized velocity (0.0-1.0)
        duration_samples: Output length in samples
        sample_rate: Audio sample rate
        
    Returns:
        Mono audio signal, shape (duration_samples,)
    """
```

---

## Tone

A Tone combines exactly two Partials with a defined structure.

### Tone Structure Types

| Structure     | Partial 1      | Partial 2      | Description                    |
|---------------|----------------|----------------|--------------------------------|
| `pcm_synth`   | PCM attack     | Synth sustain  | Classic LA sound               |
| `synth_synth` | Synth          | Synth          | Dual synthesized               |
| `pcm_pcm`     | PCM            | PCM            | Dual sample (short sounds)     |
| `synth_only`  | Synth          | Muted          | Single synth partial           |
| `pcm_only`    | PCM            | Muted          | Single PCM partial             |

### Tone Configuration

```python
@dataclass
class ToneConfig:
    structure: Literal["pcm_synth", "synth_synth", "pcm_pcm", "synth_only", "pcm_only"]
    
    partial_1: PartialConfig
    partial_2: PartialConfig
    
    # Mixing
    balance: float = 0.0       # -1.0 = P1 only, 0.0 = equal, +1.0 = P2 only
    
    # Shared modulation
    lfo_rate: float = 5.0      # Hz, 0.1-20.0
    lfo_depth_pitch: float = 0.0   # Semitones peak deviation
    lfo_depth_tvf: float = 0.0     # Normalized cutoff modulation
    lfo_depth_tva: float = 0.0     # Amplitude modulation depth (0.0-1.0)
    lfo_waveform: Literal["triangle", "sine", "square", "random"] = "triangle"
    lfo_delay: float = 0.0     # Seconds before LFO reaches full depth
```

### Tone Mixing Rules

Balance controls the mix between Partial 1 and Partial 2:

```python
def compute_partial_gains(balance: float) -> tuple[float, float]:
    """
    Compute gain multipliers for each partial.
    
    balance = -1.0: gain_1 = 1.0, gain_2 = 0.0
    balance =  0.0: gain_1 = 0.707, gain_2 = 0.707 (equal power)
    balance = +1.0: gain_1 = 0.0, gain_2 = 1.0
    """
    # Linear panning law (simple, matches D-50 character)
    gain_2 = (balance + 1.0) / 2.0
    gain_1 = 1.0 - gain_2
    return gain_1, gain_2
```

### LFO Generation

```python
def generate_lfo(
    waveform: str,
    rate: float,
    delay: float,
    duration_samples: int,
    sample_rate: int
) -> np.ndarray:
    """
    Generate LFO signal with fade-in delay.
    
    Returns:
        LFO signal, shape (duration_samples,), range [-1.0, +1.0]
    """
    t = np.arange(duration_samples) / sample_rate
    phase = 2 * np.pi * rate * t
    
    if waveform == "triangle":
        lfo = 2.0 * np.abs(2.0 * (t * rate - np.floor(t * rate + 0.5))) - 1.0
    elif waveform == "sine":
        lfo = np.sin(phase)
    elif waveform == "square":
        lfo = np.sign(np.sin(phase))
    elif waveform == "random":
        # Sample-and-hold random
        samples_per_cycle = int(sample_rate / rate)
        num_cycles = duration_samples // samples_per_cycle + 1
        values = np.random.uniform(-1.0, 1.0, num_cycles)
        lfo = np.repeat(values, samples_per_cycle)[:duration_samples]
    
    # Apply delay envelope (linear fade-in)
    if delay > 0:
        delay_samples = int(delay * sample_rate)
        fade = np.minimum(np.arange(duration_samples) / delay_samples, 1.0)
        lfo = lfo * fade
    
    return lfo
```

---

## Patch

A Patch combines exactly two Tones with layer/split behavior.

### Patch Configuration

```python
@dataclass
class PatchConfig:
    tone_a: ToneConfig
    tone_b: ToneConfig
    
    # Layer/split mode
    mode: Literal["layer", "split"] = "layer"
    split_point: int = 60      # MIDI note for split (only if mode == "split")
    
    # Tone levels
    tone_a_level: float = 0.0  # dB
    tone_b_level: float = 0.0  # dB
    
    # Global output
    output_level: float = 0.0  # dB
    
    # Effects
    chorus_enabled: bool = True
    chorus_config: ChorusConfig = field(default_factory=ChorusConfig)
    
    reverb_enabled: bool = True
    reverb_config: ReverbConfig = field(default_factory=ReverbConfig)
```

### Layer Mode

Both Tones sound simultaneously for all notes:

```python
output = (tone_a * db_to_linear(tone_a_level) + 
          tone_b * db_to_linear(tone_b_level))
```

### Split Mode

Tone A sounds for notes below split_point, Tone B for notes at or above:

```python
if note < split_point:
    output = tone_a * db_to_linear(tone_a_level)
else:
    output = tone_b * db_to_linear(tone_b_level)
```

---

## Envelopes

All D-50 envelopes are multi-stage with exponential curves. They are NOT standard ADSR.

### TVA Envelope (Time-Variant Amplifier)

5-stage envelope controlling amplitude.

```python
@dataclass
class TVAEnvelopeConfig:
    # Times in seconds (0.0 = instant, max ~30.0)
    t1: float = 0.001   # Time to L1
    t2: float = 0.1     # Time to L2
    t3: float = 0.2     # Time to L3
    t4: float = 0.5     # Time to L4 (sustain)
    t5: float = 0.3     # Release time (to zero)
    
    # Levels (0.0 = silence, 1.0 = full)
    l1: float = 1.0     # Peak after attack
    l2: float = 0.8     # Second stage
    l3: float = 0.7     # Third stage
    l4: float = 0.6     # Sustain level
    
    # Velocity sensitivity (0.0 = none, 1.0 = full)
    velocity_sensitivity: float = 0.5
```

#### TVA Envelope Stages

```
Level
  │
L1├────●
  │   ╱ ╲
L2├──╱   ●
  │ ╱     ╲
L3├╱       ●
  │         ╲
L4├──────────●──────────┐
  │                      ╲
  │                       ●───► 0
  └───┬──┬──┬──┬─────────┬──────► Time
      t1 t2 t3 t4      note-off
                         t5
```

#### TVA Envelope Math

Each stage uses exponential interpolation:

```python
def exponential_interp(
    start_level: float,
    end_level: float,
    duration_samples: int,
    curve: float = -4.0  # Negative = fast initial, slow final
) -> np.ndarray:
    """
    Compute exponential envelope segment.
    
    Args:
        start_level: Starting amplitude
        end_level: Target amplitude
        duration_samples: Segment length
        curve: Curvature (-10 to +10, 0 = linear)
        
    Returns:
        Envelope segment, shape (duration_samples,)
    """
    if duration_samples <= 0:
        return np.array([end_level])
    
    t = np.linspace(0, 1, duration_samples)
    
    if abs(curve) < 0.01:
        # Linear
        return start_level + (end_level - start_level) * t
    else:
        # Exponential
        if curve < 0:
            # Fast attack (concave down for increasing, up for decreasing)
            shaped = 1.0 - np.exp(curve * t)
            shaped = shaped / (1.0 - np.exp(curve))
        else:
            # Slow attack (concave up for increasing, down for decreasing)
            shaped = np.exp(curve * (t - 1.0))
            shaped = (shaped - np.exp(-curve)) / (1.0 - np.exp(-curve))
        
        return start_level + (end_level - start_level) * shaped
```

#### Full TVA Envelope Generation

```python
def generate_tva_envelope(
    config: TVAEnvelopeConfig,
    velocity: float,
    note_on_samples: int,
    sample_rate: int
) -> np.ndarray:
    """
    Generate complete TVA envelope.
    
    Args:
        config: Envelope configuration
        velocity: Normalized velocity (0.0-1.0)
        note_on_samples: Samples until note-off
        sample_rate: Audio sample rate
        
    Returns:
        Envelope signal, shape (note_on_samples + release_samples,)
    """
    # Apply velocity scaling
    vel_scale = 1.0 - config.velocity_sensitivity * (1.0 - velocity)
    l1 = config.l1 * vel_scale
    l2 = config.l2 * vel_scale
    l3 = config.l3 * vel_scale
    l4 = config.l4 * vel_scale
    
    # Convert times to samples
    s1 = int(config.t1 * sample_rate)
    s2 = int(config.t2 * sample_rate)
    s3 = int(config.t3 * sample_rate)
    s4 = int(config.t4 * sample_rate)
    s5 = int(config.t5 * sample_rate)
    
    segments = []
    
    # Stage 1: 0 -> L1
    segments.append(exponential_interp(0.0, l1, s1, curve=-4.0))
    
    # Stage 2: L1 -> L2
    segments.append(exponential_interp(l1, l2, s2, curve=-2.0))
    
    # Stage 3: L2 -> L3
    segments.append(exponential_interp(l2, l3, s3, curve=-2.0))
    
    # Stage 4: L3 -> L4
    segments.append(exponential_interp(l3, l4, s4, curve=-1.0))
    
    # Concatenate attack/decay portion
    attack_decay = np.concatenate(segments)
    
    # Sustain: hold at L4 until note_on_samples
    if len(attack_decay) < note_on_samples:
        sustain_samples = note_on_samples - len(attack_decay)
        sustain = np.full(sustain_samples, l4)
        envelope = np.concatenate([attack_decay, sustain])
    else:
        envelope = attack_decay[:note_on_samples]
        l4 = envelope[-1]  # Use current level for release
    
    # Stage 5: Release (L4 -> 0)
    release = exponential_interp(l4, 0.0, s5, curve=-4.0)
    
    return np.concatenate([envelope, release])
```

### TVF Envelope (Time-Variant Filter)

4-stage envelope controlling filter cutoff.

```python
@dataclass
class TVFEnvelopeConfig:
    # Times in seconds
    t1: float = 0.01    # Time to L1
    t2: float = 0.2     # Time to L2
    t3: float = 0.3     # Time to L3 (sustain)
    t4: float = 0.2     # Release time
    
    # Levels (0.0-1.0, multiplied with base cutoff)
    l1: float = 1.0     # Peak cutoff
    l2: float = 0.7     # Decay target
    l3: float = 0.5     # Sustain level
    
    # Envelope depth (0.0-1.0)
    depth: float = 0.5
    
    # Velocity sensitivity
    velocity_sensitivity: float = 0.3
```

#### TVF Envelope Application

```python
cutoff_hz = base_cutoff_hz * (1.0 + tvf_envelope * depth)
```

### Pitch Envelope

3-stage envelope for pitch modulation, critical for PCM attack character.

```python
@dataclass
class PitchEnvelopeConfig:
    # Times in seconds
    t1: float = 0.0     # Time to L1 (initial pitch)
    t2: float = 0.05    # Time to L2 (usually 0 = no deviation)
    t3: float = 0.1     # Time to L3 (release pitch deviation)
    
    # Levels in semitones (can be negative)
    l0: float = 0.0     # Starting pitch deviation
    l1: float = 0.0     # After t1
    l2: float = 0.0     # After t2 (sustain)
    l3: float = 0.0     # Release target
    
    # Depth multiplier
    depth: float = 1.0
```

#### Pitch Envelope Application

```python
frequency = base_frequency * (2.0 ** (pitch_envelope_semitones / 12.0))
```

---

## PCM Handling

### PCM Data Requirements

- Format: numpy array, dtype float32 or float64
- Range: normalized to [-1.0, +1.0]
- Channels: mono only
- Length: typically 0.1 to 1.0 seconds (attack transients)

### PCM Playback

PCM samples are one-shot (no looping). Playback uses naive linear interpolation for pitch shifting.

```python
def render_pcm(
    pcm_data: np.ndarray,
    pcm_sample_rate: int,
    target_frequency: float,
    base_frequency: float,
    output_samples: int,
    output_sample_rate: int
) -> np.ndarray:
    """
    Render PCM sample with pitch shifting.
    
    Args:
        pcm_data: Source PCM samples
        pcm_sample_rate: Original sample rate of PCM
        target_frequency: Desired playback frequency (Hz)
        base_frequency: Original recorded frequency of PCM (Hz)
        output_samples: Desired output length
        output_sample_rate: Output sample rate
        
    Returns:
        Pitch-shifted PCM, shape (output_samples,), zero-padded if PCM ends
    """
    # Compute playback rate
    pitch_ratio = target_frequency / base_frequency
    rate_ratio = pcm_sample_rate / output_sample_rate
    playback_rate = pitch_ratio * rate_ratio
    
    # Generate sample indices (naive, no anti-aliasing)
    output_indices = np.arange(output_samples)
    source_indices = output_indices * playback_rate
    
    # Linear interpolation
    output = np.zeros(output_samples)
    valid_mask = source_indices < len(pcm_data) - 1
    
    idx_floor = source_indices[valid_mask].astype(int)
    frac = source_indices[valid_mask] - idx_floor
    
    output[valid_mask] = (
        pcm_data[idx_floor] * (1.0 - frac) +
        pcm_data[idx_floor + 1] * frac
    )
    
    return output
```

### PCM + Synth Combination

For `pcm_synth` structure, the PCM and synth signals are combined:

```python
def combine_pcm_synth(
    pcm_signal: np.ndarray,
    synth_signal: np.ndarray,
    crossfade_samples: int
) -> np.ndarray:
    """
    Combine PCM attack with synth sustain.
    
    The PCM naturally fades out (one-shot). The synth is present throughout.
    A short crossfade smooths the transition.
    """
    output = synth_signal.copy()
    
    # PCM length determines blend region
    pcm_length = len(pcm_signal.nonzero()[0])  # Find last non-zero sample
    if pcm_length == 0:
        return output
    
    # Crossfade: PCM fades out, synth fades in over crossfade region
    fade_start = max(0, pcm_length - crossfade_samples)
    fade_end = pcm_length
    
    if fade_end > fade_start:
        fade_len = fade_end - fade_start
        fade_out = np.linspace(1.0, 0.0, fade_len)
        fade_in = 1.0 - fade_out
        
        output[fade_start:fade_end] = (
            pcm_signal[fade_start:fade_end] * fade_out +
            synth_signal[fade_start:fade_end] * fade_in
        )
        output[:fade_start] = pcm_signal[:fade_start]
    else:
        output[:pcm_length] = pcm_signal[:pcm_length]
    
    return output
```

---

## Waveform Generation

### Sawtooth

```python
def generate_sawtooth(
    frequency: float,
    duration_samples: int,
    sample_rate: int,
    phase_offset: float = 0.0
) -> np.ndarray:
    """
    Generate naive (aliased) sawtooth wave.
    """
    t = np.arange(duration_samples) / sample_rate
    phase = (frequency * t + phase_offset) % 1.0
    return 2.0 * phase - 1.0
```

### Square

```python
def generate_square(
    frequency: float,
    duration_samples: int,
    sample_rate: int,
    phase_offset: float = 0.0
) -> np.ndarray:
    """
    Generate naive (aliased) square wave.
    """
    t = np.arange(duration_samples) / sample_rate
    phase = (frequency * t + phase_offset) % 1.0
    return np.where(phase < 0.5, 1.0, -1.0)
```

### Pulse

```python
def generate_pulse(
    frequency: float,
    pulse_width: float,
    duration_samples: int,
    sample_rate: int,
    phase_offset: float = 0.0
) -> np.ndarray:
    """
    Generate naive (aliased) pulse wave with variable width.
    
    Args:
        pulse_width: 0.0-1.0, duty cycle (0.5 = square)
    """
    t = np.arange(duration_samples) / sample_rate
    phase = (frequency * t + phase_offset) % 1.0
    return np.where(phase < pulse_width, 1.0, -1.0)
```

### Noise

```python
def generate_noise(
    duration_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate white noise.
    
    Args:
        seed: Random seed for deterministic output
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, duration_samples)
```

---

## Filters

### TVF (Time-Variant Filter)

2-pole resonant low-pass filter (12dB/octave). State-variable filter implementation.

```python
@dataclass
class TVFState:
    low: float = 0.0
    band: float = 0.0

def apply_tvf(
    signal: np.ndarray,
    cutoff_envelope: np.ndarray,
    resonance: float,
    sample_rate: int
) -> np.ndarray:
    """
    Apply time-variant low-pass filter.
    
    Args:
        signal: Input audio
        cutoff_envelope: Cutoff frequency per sample (Hz)
        resonance: 0.0-1.0 (maps to Q 0.5-20)
        sample_rate: Audio sample rate
        
    Returns:
        Filtered signal
    """
    output = np.zeros_like(signal)
    state = TVFState()
    
    # Map resonance to Q
    q = 0.5 + resonance * 19.5
    
    for i in range(len(signal)):
        # Compute coefficients
        cutoff = np.clip(cutoff_envelope[i], 20.0, sample_rate * 0.45)
        f = 2.0 * np.sin(np.pi * cutoff / sample_rate)
        fb = q + q / (1.0 - f)
        
        # State-variable filter
        state.low += f * state.band
        high = signal[i] - state.low - state.band / fb
        state.band += f * high
        
        output[i] = state.low
    
    return output
```

### Cutoff Frequency Mapping

```python
def normalized_to_hz(normalized: float) -> float:
    """
    Map normalized cutoff (0.0-1.0) to frequency (100Hz-20kHz).
    Logarithmic mapping.
    """
    return 100.0 * (200.0 ** normalized)
```

---

## Effects

### Chorus

Stereo chorus using modulated delay lines.

```python
@dataclass
class ChorusConfig:
    rate: float = 0.5          # LFO rate in Hz (0.1-5.0)
    depth: float = 0.5         # Modulation depth (0.0-1.0)
    delay_ms: float = 7.0      # Base delay in milliseconds (1.0-20.0)
    mix: float = 0.5           # Wet/dry mix (0.0-1.0)
    voices: int = 2            # Number of chorus voices (1-4)
```

#### Chorus Implementation

```python
def apply_chorus(
    signal: np.ndarray,
    config: ChorusConfig,
    sample_rate: int
) -> np.ndarray:
    """
    Apply stereo chorus effect.
    
    Args:
        signal: Mono input signal
        config: Chorus configuration
        sample_rate: Audio sample rate
        
    Returns:
        Stereo output, shape (len(signal), 2)
    """
    num_samples = len(signal)
    output = np.zeros((num_samples, 2))
    
    # Base delay in samples
    base_delay = int(config.delay_ms * sample_rate / 1000.0)
    max_mod = int(config.depth * base_delay * 0.5)
    
    # Pre-pad signal for delay line
    padded = np.concatenate([np.zeros(base_delay + max_mod), signal])
    
    for voice in range(config.voices):
        # LFO for this voice (phase-offset)
        phase_offset = voice / config.voices
        t = np.arange(num_samples) / sample_rate
        lfo = np.sin(2 * np.pi * config.rate * t + phase_offset * 2 * np.pi)
        
        # Modulated delay
        delay_mod = base_delay + (lfo * max_mod).astype(int)
        
        # Read from delay line
        indices = np.arange(num_samples) + (base_delay + max_mod) - delay_mod
        delayed = padded[indices]
        
        # Pan voices across stereo field
        pan = (voice / max(1, config.voices - 1)) * 2.0 - 1.0 if config.voices > 1 else 0.0
        left_gain = np.sqrt(0.5 * (1.0 - pan))
        right_gain = np.sqrt(0.5 * (1.0 + pan))
        
        output[:, 0] += delayed * left_gain / config.voices
        output[:, 1] += delayed * right_gain / config.voices
    
    # Mix wet/dry
    dry_stereo = np.column_stack([signal, signal])
    return config.mix * output + (1.0 - config.mix) * dry_stereo
```

### Reverb

Simple algorithmic reverb using parallel comb filters and series allpass filters.

```python
@dataclass
class ReverbConfig:
    room_size: float = 0.8     # 0.0-1.0 (affects comb delays)
    damping: float = 0.5       # 0.0-1.0 (high frequency absorption)
    wet: float = 0.3           # Wet level (0.0-1.0)
    dry: float = 1.0           # Dry level (0.0-1.0)
    width: float = 1.0         # Stereo width (0.0-1.0)
```

#### Reverb Implementation

```python
# Comb filter delay times (samples at 44100Hz, scale for other rates)
COMB_DELAYS = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
ALLPASS_DELAYS = [225, 556, 441, 341]

def apply_reverb(
    signal: np.ndarray,
    config: ReverbConfig,
    sample_rate: int
) -> np.ndarray:
    """
    Apply algorithmic reverb.
    
    Args:
        signal: Stereo input, shape (samples, 2)
        config: Reverb configuration
        sample_rate: Audio sample rate
        
    Returns:
        Stereo output, shape (samples, 2)
    """
    # Scale delays for sample rate
    rate_scale = sample_rate / 44100.0
    comb_delays = [int(d * rate_scale * config.room_size) for d in COMB_DELAYS]
    allpass_delays = [int(d * rate_scale) for d in ALLPASS_DELAYS]
    
    # Mono sum for reverb input
    mono = (signal[:, 0] + signal[:, 1]) * 0.5
    
    # Parallel comb filters
    comb_out = np.zeros(len(mono))
    for delay in comb_delays:
        comb_out += comb_filter(mono, delay, config.damping)
    comb_out /= len(comb_delays)
    
    # Series allpass filters
    reverb = comb_out
    for delay in allpass_delays:
        reverb = allpass_filter(reverb, delay, 0.5)
    
    # Stereo spread
    left = reverb
    right = np.roll(reverb, int(23 * rate_scale))  # Small delay for width
    
    # Mix with width control
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * config.width
    wet_left = mid + side
    wet_right = mid - side
    
    # Final mix
    output = np.zeros_like(signal)
    output[:, 0] = signal[:, 0] * config.dry + wet_left * config.wet
    output[:, 1] = signal[:, 1] * config.dry + wet_right * config.wet
    
    return output


def comb_filter(
    signal: np.ndarray,
    delay: int,
    damping: float
) -> np.ndarray:
    """Feedback comb filter with damping."""
    output = np.zeros(len(signal))
    buffer = np.zeros(delay)
    feedback = 0.84  # Fixed feedback for long tail
    damp1 = damping
    damp2 = 1.0 - damping
    filterstore = 0.0
    
    for i in range(len(signal)):
        buf_idx = i % delay
        output[i] = buffer[buf_idx]
        
        # Lowpass in feedback path
        filterstore = output[i] * damp2 + filterstore * damp1
        buffer[buf_idx] = signal[i] + filterstore * feedback
    
    return output


def allpass_filter(
    signal: np.ndarray,
    delay: int,
    gain: float
) -> np.ndarray:
    """Allpass filter for diffusion."""
    output = np.zeros(len(signal))
    buffer = np.zeros(delay)
    
    for i in range(len(signal)):
        buf_idx = i % delay
        buffered = buffer[buf_idx]
        output[i] = -signal[i] + buffered
        buffer[buf_idx] = signal[i] + buffered * gain
    
    return output
```

---

## Public API

### Primary Interface

```python
class D50:
    """
    Roland D-50-esque Linear Arithmetic Synthesizer.
    
    Attributes:
        sample_rate: Audio sample rate (immutable after construction)
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize synthesizer.
        
        Args:
            sample_rate: Audio sample rate in Hz (default 44100)
        """
        self.sample_rate = sample_rate
    
    def render_note(
        self,
        patch: PatchConfig,
        note: int,
        velocity: float,
        duration: float
    ) -> np.ndarray:
        """
        Render a single note.
        
        Args:
            patch: Complete patch configuration
            note: MIDI note number (0-127, 60 = middle C)
            velocity: Normalized velocity (0.0-1.0)
            duration: Note duration in seconds (note-on time)
            
        Returns:
            Stereo audio signal, shape (total_samples, 2), dtype float64
            Total samples = duration + max release time
            
        Raises:
            ValueError: If parameters are out of valid range
            ValueError: If patch configuration is invalid
        """
        pass
    
    def render_note_samples(
        self,
        patch: PatchConfig,
        note: int,
        velocity: float,
        duration_samples: int
    ) -> np.ndarray:
        """
        Render a single note with sample-accurate duration.
        
        Args:
            patch: Complete patch configuration
            note: MIDI note number (0-127)
            velocity: Normalized velocity (0.0-1.0)
            duration_samples: Note-on duration in samples
            
        Returns:
            Stereo audio signal, shape (total_samples, 2), dtype float64
        """
        pass
```

### Helper Functions

```python
def midi_to_hz(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    if linear <= 0:
        return -120.0  # Floor
    return 20.0 * np.log10(linear)
```

### Factory Functions

```python
def create_default_patch() -> PatchConfig:
    """Create a basic LA patch (PCM+Synth structure)."""
    pass


def create_synth_only_patch(
    waveform: str = "saw",
    tvf_cutoff: float = 0.7
) -> PatchConfig:
    """Create a simple single-oscillator synth patch."""
    pass
```

---

## Validation and Error Handling

### Parameter Ranges

| Parameter         | Valid Range       | Invalid Behavior     |
|-------------------|-------------------|----------------------|
| note              | 0-127             | Raise ValueError     |
| velocity          | 0.0-1.0           | Clamp to range       |
| duration          | > 0               | Raise ValueError     |
| sample_rate       | 8000-192000       | Raise ValueError     |
| level (dB)        | -120.0 to 0.0     | Clamp to range       |
| pan               | -1.0 to 1.0       | Clamp to range       |
| envelope times    | >= 0              | Clamp to 0           |
| envelope levels   | 0.0-1.0           | Clamp to range       |
| cutoff            | 0.0-1.0           | Clamp to range       |
| resonance         | 0.0-1.0           | Clamp to range       |

### Configuration Validation

```python
def validate_partial_config(config: PartialConfig) -> list[str]:
    """
    Validate partial configuration.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    if config.source_type == "pcm":
        if config.pcm_data is None:
            errors.append("PCM source requires pcm_data")
        elif config.pcm_data.ndim != 1:
            errors.append("pcm_data must be mono (1D array)")
        if config.pcm_sample_rate is None:
            errors.append("PCM source requires pcm_sample_rate")
    
    if config.source_type == "synth":
        if config.waveform not in ["saw", "square", "pulse", "noise"]:
            errors.append(f"Unknown waveform: {config.waveform}")
        if config.waveform == "pulse":
            if not 0.0 <= config.pulse_width <= 1.0:
                errors.append("pulse_width must be 0.0-1.0")
    
    if not -24 <= config.coarse_tune <= 24:
        errors.append("coarse_tune must be -24 to +24")
    
    if not -50.0 <= config.fine_tune <= 50.0:
        errors.append("fine_tune must be -50.0 to +50.0")
    
    return errors


def validate_tone_config(config: ToneConfig) -> list[str]:
    """Validate tone configuration."""
    errors = []
    
    # Validate structure matches partial types
    p1_type = config.partial_1.source_type
    p2_type = config.partial_2.source_type
    
    structure_rules = {
        "pcm_synth": ("pcm", "synth"),
        "synth_synth": ("synth", "synth"),
        "pcm_pcm": ("pcm", "pcm"),
        "synth_only": ("synth", None),
        "pcm_only": ("pcm", None),
    }
    
    expected = structure_rules.get(config.structure)
    if expected is None:
        errors.append(f"Unknown structure: {config.structure}")
    else:
        if expected[0] != p1_type:
            errors.append(f"Partial 1 type mismatch: expected {expected[0]}, got {p1_type}")
        if expected[1] is not None and expected[1] != p2_type:
            errors.append(f"Partial 2 type mismatch: expected {expected[1]}, got {p2_type}")
    
    # Validate partials
    errors.extend(validate_partial_config(config.partial_1))
    if expected and expected[1] is not None:
        errors.extend(validate_partial_config(config.partial_2))
    
    return errors


def validate_patch_config(config: PatchConfig) -> list[str]:
    """Validate complete patch configuration."""
    errors = []
    errors.extend(validate_tone_config(config.tone_a))
    errors.extend(validate_tone_config(config.tone_b))
    
    if config.mode == "split":
        if not 0 <= config.split_point <= 127:
            errors.append("split_point must be 0-127")
    
    return errors
```

### Error Handling Policy

- **Critical errors** (invalid structure, missing PCM data): Raise `ValueError`
- **Range errors** (out-of-bounds numeric parameters): Clamp to valid range, log warning
- **Performance errors** (NaN, Inf in output): Replace with zeros, log error

---

## Implementation Notes

### Determinism

For identical inputs, `render_note` must produce bit-identical output. This requires:
- Deterministic noise generation (seeded RNG)
- No reliance on global state
- Consistent floating-point operations

Noise seed derivation:
```python
noise_seed = hash((note, int(velocity * 1000), int(duration * 1000)))
```

### Memory Layout

- Mono signals: shape `(samples,)`, dtype `float64`
- Stereo signals: shape `(samples, 2)`, dtype `float64`
- Envelopes: shape `(samples,)`, dtype `float64`

### Sample Rate Independence

All timing parameters are in seconds or Hz. Internal sample conversions:
```python
samples = int(time_seconds * sample_rate)
```

### Polyphony

Polyphony is managed externally. Each `render_note` call is independent. The caller (sequencer) is responsible for:
- Voice allocation
- Note overlap/mixing
- Global output limiting

### File Structure (Suggested)

```
src/bmmt/modelled_classics/roland/
├── __init__.py
├── d50.py              # Main synthesizer class
├── partials.py         # Partial rendering
├── envelopes.py        # Envelope generators
├── waveforms.py        # Oscillators
├── filters.py          # TVF implementation
├── effects.py          # Chorus, reverb
├── config.py           # Dataclass definitions
└── D50_SPEC.md         # This specification
```

---

## Appendix: Default Configurations

### Default TVA Envelope (Piano-like)

```python
TVAEnvelopeConfig(
    t1=0.005, t2=0.1, t3=0.2, t4=0.5, t5=0.4,
    l1=1.0, l2=0.7, l3=0.5, l4=0.4,
    velocity_sensitivity=0.6
)
```

### Default TVF Envelope (Bright attack)

```python
TVFEnvelopeConfig(
    t1=0.01, t2=0.15, t3=0.3, t4=0.2,
    l1=1.0, l2=0.6, l3=0.4,
    depth=0.6,
    velocity_sensitivity=0.4
)
```

### Default Pitch Envelope (Subtle attack bend)

```python
PitchEnvelopeConfig(
    t1=0.0, t2=0.03, t3=0.1,
    l0=0.5, l1=0.5, l2=0.0, l3=0.0,
    depth=1.0
)
```

### Default Chorus

```python
ChorusConfig(
    rate=0.8,
    depth=0.4,
    delay_ms=8.0,
    mix=0.4,
    voices=2
)
```

### Default Reverb

```python
ReverbConfig(
    room_size=0.75,
    damping=0.4,
    wet=0.25,
    dry=1.0,
    width=0.9
)
```
