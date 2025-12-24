"""
Processing module for BMMT modular audio synthesis.
Provides filters, effects, dynamics, and spatial processing.
"""

from .filters import (
    lowpass_filter,
    highpass_filter,
    bandpass_filter,
)

from .effects import (
    apply_bitcrush,
    apply_wow_flutter,
    apply_dropout,
    add_static_bursts,
    apply_tape_saturation,
    apply_vinyl_crackle,
    apply_tube_warmth,
    apply_chorus,
    apply_ensemble,
)

from .dynamics import (
    apply_compression,
    apply_parallel_compression,
    apply_sidechain_ducking,
    apply_soft_limiter,
)

from .spatial import (
    apply_reverb,
    apply_distance_filter,
    apply_stereo_width,
    apply_doppler_shift,
    apply_air_absorption,
    apply_echo_delay,
    apply_shaped_delay,
)

__all__ = [
    # Filters
    "lowpass_filter",
    "highpass_filter",
    "bandpass_filter",
    # Effects
    "apply_bitcrush",
    "apply_wow_flutter",
    "apply_dropout",
    "add_static_bursts",
    "apply_tape_saturation",
    "apply_vinyl_crackle",
    "apply_tube_warmth",
    "apply_chorus",
    "apply_ensemble",
    # Dynamics
    "apply_compression",
    "apply_parallel_compression",
    "apply_sidechain_ducking",
    "apply_soft_limiter",
    # Spatial
    "apply_reverb",
    "apply_distance_filter",
    "apply_stereo_width",
    "apply_doppler_shift",
    "apply_air_absorption",
    "apply_echo_delay",
    "apply_shaped_delay",
]
