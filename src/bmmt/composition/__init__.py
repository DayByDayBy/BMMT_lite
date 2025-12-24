"""Composition module for audio mixing and sequencing."""

from .sequencer import Track, Sequencer, create_trigger_pattern, create_euclidean_rhythm, beats_to_triggers
from .mixing import (
    combine_signals,
    layer_signals_with_timing,
    apply_frequency_modulation,
    apply_filter_modulation,
    normalize_to_peak,
    apply_master_limiter,
    create_stereo_field,
    apply_dynamic_panning,
    apply_parameter_automation,
    layer_with_crossfade,
    parallel_mix,
    serial_chain
)

__all__ = [
    # Sequencer classes
    'Track',
    'Sequencer',
    
    # Step sequencer functions
    'create_trigger_pattern',
    'create_euclidean_rhythm',
    'beats_to_triggers',
    
    # Mixing functions
    'combine_signals',
    'layer_signals_with_timing',
    'apply_frequency_modulation',
    'apply_filter_modulation',
    'normalize_to_peak',
    'apply_master_limiter',
    'create_stereo_field',
    'apply_dynamic_panning',
    'apply_parameter_automation',
    'layer_with_crossfade',
    'parallel_mix',
    'serial_chain'
]