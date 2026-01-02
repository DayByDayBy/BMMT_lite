"""
DX7II Factory Presets.

Classic DX7-inspired sounds for immediate use.
These are starting points, not exact recreations.
"""

from .dx7 import DX7IIParams, OperatorParams


def _make_params(
    algorithm: int,
    op_configs: list[dict]
) -> DX7IIParams:
    """Helper to build DX7IIParams from simplified config."""
    operators = []
    for cfg in op_configs:
        operators.append(OperatorParams(
            freq_ratio=cfg.get("ratio", 1.0),
            detune_cents=cfg.get("detune", 0.0),
            level=cfg.get("level", 0.8),
            rates=cfg.get("rates", [0.3, 0.05, 0.02, 0.1]),
            levels=cfg.get("levels", [1.0, 0.7, 0.5, 0.0]),
            feedback=cfg.get("feedback", 0.0),
            velocity_sens=cfg.get("vel_sens", 1.0),
        ))
    return DX7IIParams(algorithm=algorithm, operators=operators)


# =============================================================================
# Factory Presets
# =============================================================================

PRESETS: dict[str, DX7IIParams] = {}

# -----------------------------------------------------------------------------
# INIT - Basic patch for sound design
# -----------------------------------------------------------------------------
PRESETS["init"] = _make_params(
    algorithm=32,  # All carriers
    op_configs=[
        {"ratio": 1.0, "level": 0.7, "rates": [0.5, 0.1, 0.05, 0.2], "levels": [1.0, 0.8, 0.6, 0.0]},
        {"ratio": 1.0, "level": 0.0},  # Off
        {"ratio": 1.0, "level": 0.0},
        {"ratio": 1.0, "level": 0.0},
        {"ratio": 1.0, "level": 0.0},
        {"ratio": 1.0, "level": 0.0},
    ]
)

# -----------------------------------------------------------------------------
# E_PIANO_1 - Classic FM electric piano with bell-like attack
# -----------------------------------------------------------------------------
PRESETS["e_piano_1"] = _make_params(
    algorithm=5,  # Two parallel 3-op stacks
    op_configs=[
        # Stack 1: Carrier
        {"ratio": 1.0, "level": 0.85, "rates": [0.8, 0.03, 0.01, 0.15], "levels": [1.0, 0.5, 0.3, 0.0]},
        # Stack 1: Modulator
        {"ratio": 1.0, "level": 0.75, "rates": [0.9, 0.1, 0.02, 0.2], "levels": [1.0, 0.3, 0.1, 0.0]},
        # Stack 1: Top modulator (bell component)
        {"ratio": 14.0, "level": 0.5, "rates": [0.95, 0.2, 0.05, 0.3], "levels": [1.0, 0.1, 0.0, 0.0]},
        # Stack 2: Carrier (warmth)
        {"ratio": 1.0, "level": 0.6, "detune": 7.0, "rates": [0.7, 0.02, 0.01, 0.1], "levels": [1.0, 0.6, 0.4, 0.0]},
        # Stack 2: Modulator
        {"ratio": 1.0, "level": 0.65, "rates": [0.85, 0.08, 0.02, 0.15], "levels": [1.0, 0.4, 0.2, 0.0]},
        # Stack 2: Top (adds shimmer)
        {"ratio": 3.0, "level": 0.4, "rates": [0.9, 0.15, 0.03, 0.25], "levels": [1.0, 0.2, 0.05, 0.0]},
    ]
)

# -----------------------------------------------------------------------------
# BRASS_1 - Punchy FM brass with feedback
# -----------------------------------------------------------------------------
PRESETS["brass_1"] = _make_params(
    algorithm=1,  # Serial chain
    op_configs=[
        # Carrier
        {"ratio": 1.0, "level": 0.9, "rates": [0.4, 0.02, 0.015, 0.08], "levels": [1.0, 0.85, 0.7, 0.0]},
        # Modulators with increasing ratios
        {"ratio": 1.0, "level": 0.7, "rates": [0.5, 0.03, 0.02, 0.1], "levels": [1.0, 0.75, 0.6, 0.0]},
        {"ratio": 1.0, "level": 0.6, "rates": [0.6, 0.04, 0.02, 0.12], "levels": [1.0, 0.65, 0.5, 0.0]},
        {"ratio": 1.0, "level": 0.5, "rates": [0.7, 0.05, 0.025, 0.15], "levels": [1.0, 0.55, 0.4, 0.0]},
        {"ratio": 1.0, "level": 0.4, "rates": [0.8, 0.06, 0.03, 0.18], "levels": [1.0, 0.45, 0.3, 0.0]},
        # Top with feedback for growl
        {"ratio": 1.0, "level": 0.35, "feedback": 0.6, "rates": [0.85, 0.07, 0.035, 0.2], "levels": [1.0, 0.4, 0.25, 0.0]},
    ]
)

# -----------------------------------------------------------------------------
# BASS_1 - Solid FM bass
# -----------------------------------------------------------------------------
PRESETS["bass_1"] = _make_params(
    algorithm=13,  # Stacked modulators
    op_configs=[
        # Carrier - fundamental
        {"ratio": 0.5, "level": 0.95, "rates": [0.7, 0.02, 0.01, 0.15], "levels": [1.0, 0.9, 0.85, 0.0]},
        # Modulator adding harmonic content
        {"ratio": 1.0, "level": 0.7, "rates": [0.8, 0.05, 0.02, 0.2], "levels": [1.0, 0.5, 0.3, 0.0]},
        {"ratio": 1.0, "level": 0.55, "rates": [0.85, 0.06, 0.025, 0.22], "levels": [1.0, 0.45, 0.25, 0.0]},
        {"ratio": 2.0, "level": 0.45, "rates": [0.9, 0.08, 0.03, 0.25], "levels": [1.0, 0.35, 0.15, 0.0]},
        {"ratio": 1.0, "level": 0.5, "rates": [0.8, 0.05, 0.02, 0.2], "levels": [1.0, 0.4, 0.2, 0.0]},
        {"ratio": 3.0, "level": 0.3, "rates": [0.95, 0.1, 0.04, 0.3], "levels": [1.0, 0.2, 0.05, 0.0]},
    ]
)

# -----------------------------------------------------------------------------
# STRINGS_PAD - Lush string pad with slow attack
# -----------------------------------------------------------------------------
PRESETS["strings_pad"] = _make_params(
    algorithm=21,  # Branching
    op_configs=[
        # Carrier - main body
        {"ratio": 1.0, "level": 0.8, "rates": [0.01, 0.005, 0.003, 0.02], "levels": [1.0, 0.9, 0.85, 0.0]},
        # Modulator for shimmer
        {"ratio": 2.0, "level": 0.5, "rates": [0.015, 0.008, 0.004, 0.025], "levels": [1.0, 0.7, 0.6, 0.0]},
        {"ratio": 3.0, "level": 0.4, "rates": [0.02, 0.01, 0.005, 0.03], "levels": [1.0, 0.6, 0.5, 0.0]},
        # Second branch
        {"ratio": 1.0, "level": 0.6, "detune": 5.0, "rates": [0.012, 0.006, 0.003, 0.022], "levels": [1.0, 0.85, 0.8, 0.0]},
        {"ratio": 2.0, "level": 0.45, "rates": [0.018, 0.009, 0.0045, 0.027], "levels": [1.0, 0.65, 0.55, 0.0]},
        # Shared modulator
        {"ratio": 1.0, "level": 0.5, "detune": -5.0, "rates": [0.008, 0.004, 0.002, 0.015], "levels": [1.0, 0.8, 0.75, 0.0]},
    ]
)

# -----------------------------------------------------------------------------
# BELL - Tubular bell with inharmonic ratios
# -----------------------------------------------------------------------------
PRESETS["bell"] = _make_params(
    algorithm=7,  # Three mods into carrier
    op_configs=[
        # Carrier
        {"ratio": 1.0, "level": 0.85, "rates": [0.95, 0.01, 0.005, 0.03], "levels": [1.0, 0.6, 0.4, 0.0]},
        # Additional carrier for body
        {"ratio": 2.0, "level": 0.5, "rates": [0.9, 0.02, 0.008, 0.04], "levels": [1.0, 0.4, 0.25, 0.0]},
        {"ratio": 3.5, "level": 0.4, "rates": [0.92, 0.025, 0.01, 0.05], "levels": [1.0, 0.35, 0.2, 0.0]},
        # Inharmonic modulators for bell character
        {"ratio": 5.5, "level": 0.55, "rates": [0.98, 0.08, 0.03, 0.15], "levels": [1.0, 0.2, 0.05, 0.0]},
        {"ratio": 7.0, "level": 0.45, "rates": [0.99, 0.1, 0.04, 0.2], "levels": [1.0, 0.15, 0.03, 0.0]},
        {"ratio": 11.0, "level": 0.35, "rates": [0.995, 0.15, 0.05, 0.25], "levels": [1.0, 0.1, 0.01, 0.0]},
    ]
)

# -----------------------------------------------------------------------------
# ORGAN - Drawbar-style with all carriers
# -----------------------------------------------------------------------------
PRESETS["organ"] = _make_params(
    algorithm=32,  # All carriers
    op_configs=[
        # Fundamental
        {"ratio": 0.5, "level": 0.85, "rates": [0.9, 0.01, 0.005, 0.1], "levels": [1.0, 1.0, 0.95, 0.0]},
        # 2nd harmonic
        {"ratio": 1.0, "level": 0.8, "rates": [0.9, 0.01, 0.005, 0.1], "levels": [1.0, 1.0, 0.95, 0.0]},
        # 3rd harmonic
        {"ratio": 1.5, "level": 0.6, "rates": [0.9, 0.01, 0.005, 0.1], "levels": [1.0, 1.0, 0.95, 0.0]},
        # 4th harmonic
        {"ratio": 2.0, "level": 0.7, "rates": [0.9, 0.01, 0.005, 0.1], "levels": [1.0, 1.0, 0.95, 0.0]},
        # 6th harmonic
        {"ratio": 3.0, "level": 0.4, "rates": [0.9, 0.01, 0.005, 0.1], "levels": [1.0, 1.0, 0.95, 0.0]},
        # 8th harmonic
        {"ratio": 4.0, "level": 0.3, "rates": [0.9, 0.01, 0.005, 0.1], "levels": [1.0, 1.0, 0.95, 0.0]},
    ]
)

# -----------------------------------------------------------------------------
# MARIMBA - Woody mallet with fast decay
# -----------------------------------------------------------------------------
PRESETS["marimba"] = _make_params(
    algorithm=5,  # Two stacks
    op_configs=[
        # Carrier - main tone
        {"ratio": 1.0, "level": 0.9, "rates": [0.95, 0.15, 0.05, 0.3], "levels": [1.0, 0.3, 0.1, 0.0]},
        # Modulator
        {"ratio": 4.0, "level": 0.6, "rates": [0.98, 0.25, 0.1, 0.4], "levels": [1.0, 0.15, 0.02, 0.0]},
        # Attack transient
        {"ratio": 10.0, "level": 0.4, "rates": [0.999, 0.5, 0.2, 0.6], "levels": [1.0, 0.02, 0.0, 0.0]},
        # Second carrier for resonance
        {"ratio": 1.0, "level": 0.5, "detune": 3.0, "rates": [0.9, 0.12, 0.04, 0.25], "levels": [1.0, 0.35, 0.15, 0.0]},
        {"ratio": 3.0, "level": 0.35, "rates": [0.96, 0.2, 0.08, 0.35], "levels": [1.0, 0.1, 0.01, 0.0]},
        {"ratio": 5.0, "level": 0.25, "rates": [0.98, 0.3, 0.12, 0.45], "levels": [1.0, 0.05, 0.0, 0.0]},
    ]
)

# -----------------------------------------------------------------------------
# GLASS_PAD - Ethereal glass texture
# -----------------------------------------------------------------------------
PRESETS["glass_pad"] = _make_params(
    algorithm=21,  # Branching
    op_configs=[
        # Carrier
        {"ratio": 1.0, "level": 0.75, "rates": [0.02, 0.008, 0.004, 0.03], "levels": [1.0, 0.85, 0.8, 0.0]},
        # High harmonic modulators for glass
        {"ratio": 7.0, "level": 0.4, "rates": [0.03, 0.015, 0.008, 0.05], "levels": [1.0, 0.5, 0.3, 0.0]},
        {"ratio": 11.0, "level": 0.3, "rates": [0.04, 0.02, 0.01, 0.06], "levels": [1.0, 0.4, 0.2, 0.0]},
        # Second voice
        {"ratio": 2.0, "level": 0.55, "detune": 8.0, "rates": [0.025, 0.01, 0.005, 0.035], "levels": [1.0, 0.8, 0.75, 0.0]},
        {"ratio": 5.0, "level": 0.35, "rates": [0.035, 0.018, 0.009, 0.055], "levels": [1.0, 0.45, 0.25, 0.0]},
        # Shared mod
        {"ratio": 3.0, "level": 0.45, "detune": -8.0, "rates": [0.015, 0.007, 0.0035, 0.025], "levels": [1.0, 0.7, 0.6, 0.0]},
    ]
)


def get_preset(name: str) -> DX7IIParams:
    """
    Get a preset by name.
    
    Args:
        name: Preset name (e.g., 'e_piano_1', 'brass_1')
        
    Returns:
        DX7IIParams instance
        
    Raises:
        KeyError: If preset not found
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> list[str]:
    """Return list of available preset names."""
    return sorted(PRESETS.keys())
