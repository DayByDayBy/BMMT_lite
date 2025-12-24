"""Audio generation module for oscillators, noise, and drums."""

from .oscillators import (
    generate_sine,
    generate_triangle,
    generate_sawtooth,
    generate_square
)

from .noise import (
    generate_white_noise,
    generate_pink_noise,
    generate_brown_noise
)

from .drums import (
    synthesize_kick,
    synthesize_snare,
    synthesize_hihat,
    synthesize_conga
)

__all__ = [
    # Oscillators
    "generate_sine",
    "generate_triangle",
    "generate_sawtooth",
    "generate_square",
    
    # Noise generators
    "generate_white_noise",
    "generate_pink_noise",
    "generate_brown_noise",
    
    # Drum synthesizers
    "synthesize_kick",
    "synthesize_snare",
    "synthesize_hihat",
    "synthesize_conga"
]