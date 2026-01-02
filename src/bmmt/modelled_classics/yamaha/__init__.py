"""
Yamaha DX7II-esque Phase Modulation Synthesizer.

A DX7II-inspired 6-operator PM synthesizer for offline audio rendering.
Produces the crystalline FM tones characteristic of late-80s/early-90s music.

This is DX7-esque, not a bit-accurate Yamaha emulation.

Example:
    >>> from bmmt.modelled_classics.yamaha import DX7II, render_note
    >>> synth = DX7II(sample_rate=44100)
    >>> audio = synth.render_note(freq=440.0, duration=1.0, velocity=0.8)
    
    # Or use convenience function with preset:
    >>> audio = render_note(440.0, 1.0, velocity=0.8, preset="e_piano_1")

Key features:
- 6 sine operators with phase modulation
- 4-stage rate/level envelopes (not ADSR)
- Algorithm-based operator routing (DAGs)
- Operator 6 self-feedback
- Logarithmic amplitude scaling
- Intentional aliasing for authentic DX character

Available presets:
- init: Basic patch for sound design
- e_piano_1: Classic FM electric piano
- brass_1: Punchy FM brass
- bass_1: Solid FM bass
- strings_pad: Lush string pad
- bell: Tubular bell
- organ: Drawbar-style organ
- marimba: Woody mallet
- glass_pad: Ethereal glass texture
"""

import numpy as np
from typing import Union

from .dx7 import (
    DX7II,
    DX7IIParams,
    OperatorParams,
    DXEnvelope,
    Operator,
    ALGORITHMS,
    level_map,
    feedback_map,
    key_scale,
)
from .presets import PRESETS, get_preset, list_presets


def render_note(
    freq: float,
    duration: float,
    velocity: float = 0.8,
    preset: Union[str, DX7IIParams] = "init",
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Render a single note with the DX7II synthesizer.
    
    Convenience function for quick note rendering without
    manually instantiating DX7II.
    
    Args:
        freq: Note frequency in Hz
        duration: Note duration in seconds
        velocity: Note velocity (0..1)
        preset: Preset name or DX7IIParams instance
        sample_rate: Audio sample rate
        
    Returns:
        Mono audio as numpy array (float64)
        
    Example:
        >>> audio = render_note(440.0, 1.0, preset="e_piano_1")
        >>> audio = render_note(261.63, 0.5, velocity=0.6, preset="bass_1")
    """
    synth = DX7II(sample_rate=sample_rate)
    
    if isinstance(preset, str):
        params = get_preset(preset)
    else:
        params = preset
    
    synth.set_params(params)
    return synth.render_note(freq=freq, duration=duration, velocity=velocity)


def render_midi_note(
    midi_note: int,
    duration: float,
    velocity: float = 0.8,
    preset: Union[str, DX7IIParams] = "init",
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Render a note from MIDI note number.
    
    Args:
        midi_note: MIDI note number (0-127, 60=middle C)
        duration: Note duration in seconds
        velocity: Note velocity (0..1)
        preset: Preset name or DX7IIParams instance
        sample_rate: Audio sample rate
        
    Returns:
        Mono audio as numpy array (float64)
        
    Example:
        >>> audio = render_midi_note(60, 1.0, preset="e_piano_1")  # Middle C
        >>> audio = render_midi_note(69, 0.5, preset="brass_1")   # A4 (440Hz)
    """
    freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
    return render_note(freq, duration, velocity, preset, sample_rate)


__all__ = [
    # Main class
    "DX7II",
    # Parameter classes
    "DX7IIParams",
    "OperatorParams", 
    "DXEnvelope",
    "Operator",
    # Convenience functions
    "render_note",
    "render_midi_note",
    # Presets
    "PRESETS",
    "get_preset",
    "list_presets",
    # Algorithms
    "ALGORITHMS",
    # Helper functions
    "level_map",
    "feedback_map",
    "key_scale",
]
