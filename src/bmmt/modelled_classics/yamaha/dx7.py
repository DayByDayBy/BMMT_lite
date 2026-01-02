"""
DX7II-esque Phase Modulation Synthesizer.

Core implementation of a 6-operator PM synthesizer inspired by the Yamaha DX7II.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Helper Functions
# =============================================================================

def level_map(level: float) -> float:
    """
    Convert normalized level (0..1) to logarithmic amplitude.
    
    FM spectra explode linearly; log scaling keeps timbres usable.
    
    Args:
        level: Normalized level in [0, 1]
        
    Returns:
        Logarithmic amplitude multiplier
    """
    return 10 ** ((level - 1.0) * 3)


def feedback_map(feedback: float) -> float:
    """
    Convert normalized feedback (0..1) to bounded gain.
    
    Nonlinear curve prevents instability at high values.
    
    Args:
        feedback: Normalized feedback in [0, 1]
        
    Returns:
        Bounded feedback gain (max 0.99)
    """
    raw_gain = 0.5 * (2 ** (4 * feedback) - 1)
    return min(0.99, raw_gain)


def key_scale(freq: float) -> float:
    """
    Keyboard scaling factor for modulators.
    
    Higher notes get brighter (more modulation).
    
    Args:
        freq: Note frequency in Hz
        
    Returns:
        Scaling factor in [0.5, 2.0]
    """
    return np.clip((freq / 440.0) ** 0.5, 0.5, 2.0)


def cents_to_ratio(cents: float) -> float:
    """
    Convert detune in cents to frequency ratio.
    
    Args:
        cents: Detune amount in cents
        
    Returns:
        Frequency multiplier
    """
    return 2 ** (cents / 1200.0)


# =============================================================================
# DXEnvelope
# =============================================================================

@dataclass
class DXEnvelope:
    """
    DX7-style 4-stage rate/level envelope.
    
    Unlike ADSR, this envelope uses 4 rates (R1-R4) and 4 levels (L1-L4).
    Each stage transitions from current value toward target level at a given rate.
    
    Attributes:
        rates: List of 4 rates [R1, R2, R3, R4] in (0, 1]
        levels: List of 4 target levels [L1, L2, L3, L4] in [0, 1]
        sample_rate: Audio sample rate
    """
    rates: list[float]
    levels: list[float]
    sample_rate: int
    
    stage: int = field(default=0, init=False)
    value: float = field(default=0.0, init=False)
    released: bool = field(default=False, init=False)
    
    def __post_init__(self):
        if len(self.rates) != 4 or len(self.levels) != 4:
            raise ValueError("DXEnvelope requires exactly 4 rates and 4 levels")
        self.value = 0.0
        self.stage = 0
        self.released = False
    
    def reset(self) -> None:
        """Reset envelope to initial state for new note."""
        self.stage = 0
        self.value = 0.0
        self.released = False
    
    def note_off(self) -> None:
        """Trigger release stage (stage 3)."""
        if not self.released:
            self.released = True
            self.stage = 3
    
    def step(self) -> float:
        """
        Advance envelope by one sample and return current value.
        
        Uses first-order exponential smoothing for stage transitions.
        
        Returns:
            Current envelope value in [0, 1]
        """
        if self.stage >= 4:
            return self.value
        
        target = self.levels[self.stage]
        rate = self.rates[self.stage]
        
        # First-order exponential smoothing
        # Rate controls how quickly we approach target
        # Higher rate = faster approach
        delta = target - self.value
        self.value += delta * rate
        
        # Advance stage when close enough to target
        if abs(delta) < 1e-5:
            self.value = target
            self.stage += 1
        
        return self.value
    
    def is_finished(self) -> bool:
        """Check if envelope has completed all stages."""
        return self.stage >= 4


# =============================================================================
# Operator
# =============================================================================

@dataclass
class Operator:
    """
    Single FM operator: sine oscillator with envelope and modulation.
    
    Attributes:
        freq_ratio: Frequency multiplier relative to base frequency
        detune_cents: Fine tuning in cents
        level: Output level (0..1, will be log-mapped)
        envelope: DXEnvelope instance
        feedback: Self-feedback amount (0..1), only meaningful for op6
        velocity_sens: How much velocity affects output (0..1)
    """
    freq_ratio: float
    detune_cents: float
    level: float
    envelope: DXEnvelope
    feedback: float = 0.0
    velocity_sens: float = 1.0
    
    phase: float = field(default=0.0, init=False)
    last_output: float = field(default=0.0, init=False)
    
    def reset(self) -> None:
        """Reset operator state for new note."""
        self.phase = 0.0
        self.last_output = 0.0
        self.envelope.reset()
    
    def note_off(self) -> None:
        """Trigger envelope release."""
        self.envelope.note_off()
    
    def process(
        self,
        base_freq: float,
        phase_mod: float,
        sample_rate: int,
        velocity: float = 1.0,
        is_modulator: bool = False,
        key_scale_factor: float = 1.0
    ) -> float:
        """
        Process one sample.
        
        Args:
            base_freq: Note base frequency in Hz
            phase_mod: Summed phase modulation from upstream operators
            sample_rate: Audio sample rate
            velocity: Note velocity (0..1)
            is_modulator: True if this operator modulates others
            key_scale_factor: Keyboard scaling multiplier
            
        Returns:
            Operator output sample
        """
        # Get envelope value
        env = self.envelope.step()
        
        # Calculate operator frequency with detune
        detune_factor = cents_to_ratio(self.detune_cents)
        op_freq = base_freq * self.freq_ratio * detune_factor
        
        # Phase increment
        phase_inc = 2 * np.pi * op_freq / sample_rate
        self.phase += phase_inc
        
        # Wrap phase to prevent float overflow
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        
        # Feedback (applied to phase)
        fb_term = 0.0
        if self.feedback > 0:
            fb_gain = feedback_map(self.feedback)
            fb_term = self.last_output * fb_gain
        
        # Compute output
        mapped_level = level_map(self.level)
        
        # Apply velocity sensitivity
        vel_scale = 1.0 - self.velocity_sens * (1.0 - velocity)
        
        # Apply keyboard scaling for modulators
        if is_modulator:
            output = env * mapped_level * vel_scale * key_scale_factor * np.sin(
                self.phase + phase_mod + fb_term
            )
        else:
            # Carriers: envelope controls amplitude, velocity affects amplitude
            output = env * mapped_level * vel_scale * np.sin(
                self.phase + phase_mod + fb_term
            )
        
        self.last_output = output
        return output


# =============================================================================
# Algorithms
# =============================================================================

# Algorithm representation: dict mapping operator -> list of its modulators
# Operator numbers are 1-6 (DX7 convention)
# Carriers are operators with no downstream connections (not in any modulator list)

ALGORITHMS: dict[int, dict[int, list[int]]] = {
    # Algorithm 1: Full serial chain 6→5→4→3→2→1
    1: {
        6: [],
        5: [6],
        4: [5],
        3: [4],
        2: [3],
        1: [2],
    },
    
    # Algorithm 5: Two parallel 3-op stacks (6→5→4) and (3→2→1)
    5: {
        6: [],
        5: [6],
        4: [5],
        3: [],
        2: [3],
        1: [2],
    },
    
    # Algorithm 7: Three modulators (6,5,4) into carrier 1, separate 3→2
    7: {
        6: [],
        5: [],
        4: [],
        3: [],
        2: [3],
        1: [4, 5, 6],
    },
    
    # Algorithm 13: Stacked modulators - 6→5, 4→3, both into 2→1
    13: {
        6: [],
        5: [6],
        4: [],
        3: [4],
        2: [3, 5],
        1: [2],
    },
    
    # Algorithm 21: Branching - 6→5→4, 6→3→2, all into 1
    21: {
        6: [],
        5: [6],
        4: [5],
        3: [6],
        2: [3],
        1: [2, 4],
    },
    
    # Algorithm 31: Two modulators, four carriers
    31: {
        6: [],
        5: [6],
        4: [5],
        3: [5],
        2: [5],
        1: [5],
    },
    
    # Algorithm 32: All carriers (no modulation) - organ-like
    32: {
        6: [],
        5: [],
        4: [],
        3: [],
        2: [],
        1: [],
    },
    
    # Algorithm 11: Y-shape - 6→5, 4→3, both into 2, 2→1
    11: {
        6: [],
        5: [6],
        4: [],
        3: [4],
        2: [3, 5],
        1: [2],
    },
}


def validate_algorithm(algo: dict[int, list[int]]) -> None:
    """
    Validate algorithm structure.
    
    Raises:
        ValueError: If algorithm is invalid
    """
    # Check all operators 1-6 present
    if set(algo.keys()) != {1, 2, 3, 4, 5, 6}:
        raise ValueError("Algorithm must contain exactly operators 1-6")
    
    # Check for valid modulator references
    for op, modulators in algo.items():
        for mod in modulators:
            if mod not in algo:
                raise ValueError(f"Invalid modulator {mod} for operator {op}")
    
    # Check for cycles using DFS
    def has_cycle(start: int, visited: set, path: set) -> bool:
        visited.add(start)
        path.add(start)
        for mod in algo[start]:
            if mod in path:
                return True
            if mod not in visited:
                if has_cycle(mod, visited, path):
                    return True
        path.remove(start)
        return False
    
    visited: set[int] = set()
    for op in algo:
        if op not in visited:
            if has_cycle(op, visited, set()):
                raise ValueError("Algorithm contains a cycle")


def topological_sort(algo: dict[int, list[int]]) -> list[int]:
    """
    Return operators in processing order (modulators before carriers).
    
    Args:
        algo: Algorithm dictionary
        
    Returns:
        List of operator numbers in processing order
    """
    # Build reverse graph (who modulates whom)
    in_degree = {op: 0 for op in algo}
    dependents: dict[int, list[int]] = {op: [] for op in algo}
    
    for op, modulators in algo.items():
        in_degree[op] = len(modulators)
        for mod in modulators:
            dependents[mod].append(op)
    
    # Kahn's algorithm
    queue = [op for op in algo if in_degree[op] == 0]
    result = []
    
    while queue:
        # Sort for deterministic order
        queue.sort(reverse=True)
        current = queue.pop()
        result.append(current)
        
        for dep in dependents[current]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
    
    return result


def get_carriers(algo: dict[int, list[int]]) -> list[int]:
    """
    Identify carrier operators (those that contribute to final output).
    
    Carriers are operators that don't modulate any other operator.
    
    Args:
        algo: Algorithm dictionary
        
    Returns:
        List of carrier operator numbers
    """
    # Build set of all modulators
    all_modulators: set[int] = set()
    for modulators in algo.values():
        all_modulators.update(modulators)
    
    # Carriers are operators NOT in the modulator set
    return [op for op in algo if op not in all_modulators]


# =============================================================================
# DX7II Synthesizer
# =============================================================================

@dataclass
class OperatorParams:
    """Parameters for a single operator."""
    freq_ratio: float = 1.0
    detune_cents: float = 0.0
    level: float = 0.8
    rates: list[float] = field(default_factory=lambda: [0.5, 0.1, 0.05, 0.1])
    levels: list[float] = field(default_factory=lambda: [1.0, 0.7, 0.5, 0.0])
    feedback: float = 0.0
    velocity_sens: float = 1.0


@dataclass
class DX7IIParams:
    """Complete patch parameters for DX7II."""
    algorithm: int = 1
    operators: list[OperatorParams] = field(
        default_factory=lambda: [OperatorParams() for _ in range(6)]
    )
    
    def __post_init__(self):
        if len(self.operators) != 6:
            raise ValueError("DX7IIParams requires exactly 6 operators")


class DX7II:
    """
    DX7II-esque Phase Modulation Synthesizer.
    
    A 6-operator PM synthesizer for offline audio rendering.
    
    Args:
        sample_rate: Audio sample rate (immutable after construction)
        
    Example:
        >>> synth = DX7II(sample_rate=44100)
        >>> audio = synth.render_note(freq=440.0, duration=1.0, velocity=0.8)
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._params = DX7IIParams()
        self._operators: list[Operator] = []
        self._algo: dict[int, list[int]] = {}
        self._topo_order: list[int] = []
        self._carriers: list[int] = []
        
        # Initialize with default params
        self._build_operators()
        self._set_algorithm(self._params.algorithm)
    
    def _build_operators(self) -> None:
        """Build operator instances from current params."""
        self._operators = []
        for i, op_params in enumerate(self._params.operators):
            env = DXEnvelope(
                rates=op_params.rates.copy(),
                levels=op_params.levels.copy(),
                sample_rate=self.sample_rate
            )
            op = Operator(
                freq_ratio=op_params.freq_ratio,
                detune_cents=op_params.detune_cents,
                level=op_params.level,
                envelope=env,
                feedback=op_params.feedback if i == 5 else 0.0,  # Only op6 has feedback
                velocity_sens=op_params.velocity_sens
            )
            self._operators.append(op)
    
    def _set_algorithm(self, algo_num: int) -> None:
        """Set and validate algorithm."""
        if algo_num not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {algo_num}. Available: {list(ALGORITHMS.keys())}")
        
        self._algo = ALGORITHMS[algo_num]
        validate_algorithm(self._algo)
        self._topo_order = topological_sort(self._algo)
        self._carriers = get_carriers(self._algo)
    
    def set_params(self, params: DX7IIParams) -> None:
        """
        Set complete patch parameters.
        
        Args:
            params: DX7IIParams instance
        """
        self._params = params
        self._build_operators()
        self._set_algorithm(params.algorithm)
    
    def set_algorithm(self, algo_num: int) -> None:
        """
        Set algorithm by number.
        
        Args:
            algo_num: Algorithm number (1, 5, 7, 11, 13, 21, 31, 32)
        """
        self._params.algorithm = algo_num
        self._set_algorithm(algo_num)
    
    def set_operator(self, op_num: int, params: OperatorParams) -> None:
        """
        Set parameters for a single operator.
        
        Args:
            op_num: Operator number (1-6)
            params: OperatorParams instance
        """
        if not 1 <= op_num <= 6:
            raise ValueError(f"Operator number must be 1-6, got {op_num}")
        
        idx = op_num - 1
        self._params.operators[idx] = params
        
        # Rebuild just this operator
        env = DXEnvelope(
            rates=params.rates.copy(),
            levels=params.levels.copy(),
            sample_rate=self.sample_rate
        )
        self._operators[idx] = Operator(
            freq_ratio=params.freq_ratio,
            detune_cents=params.detune_cents,
            level=params.level,
            envelope=env,
            feedback=params.feedback if idx == 5 else 0.0,
            velocity_sens=params.velocity_sens
        )
    
    def render_note(
        self,
        freq: float,
        duration: float,
        velocity: float = 0.8
    ) -> np.ndarray:
        """
        Render a single note.
        
        Args:
            freq: Note frequency in Hz
            duration: Note duration in seconds
            velocity: Note velocity (0..1)
            
        Returns:
            Mono audio as numpy array (float64)
        """
        num_samples = int(duration * self.sample_rate)
        output = np.zeros(num_samples, dtype=np.float64)
        
        # Reset all operators
        for op in self._operators:
            op.reset()
        
        # Precompute keyboard scaling
        ks = key_scale(freq)
        
        # Identify which operators are modulators
        all_modulators: set[int] = set()
        for mods in self._algo.values():
            all_modulators.update(mods)
        
        # Render sample by sample
        for i in range(num_samples):
            # Trigger release at 80% of duration (simple gate)
            if i == int(num_samples * 0.8):
                for op in self._operators:
                    op.note_off()
            
            # Process operators in topological order
            op_outputs: dict[int, float] = {}
            
            for op_num in self._topo_order:
                op_idx = op_num - 1
                op = self._operators[op_idx]
                
                # Sum modulation from upstream operators
                modulators = self._algo[op_num]
                phase_mod = sum(op_outputs.get(m, 0.0) for m in modulators)
                
                # Process this operator
                is_mod = op_num in all_modulators
                out = op.process(
                    base_freq=freq,
                    phase_mod=phase_mod,
                    sample_rate=self.sample_rate,
                    velocity=velocity,
                    is_modulator=is_mod,
                    key_scale_factor=ks if is_mod else 1.0
                )
                op_outputs[op_num] = out
            
            # Sum carrier outputs
            output[i] = sum(op_outputs[c] for c in self._carriers)
        
        return output


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("DX7II Module Self-Test")
    print("=" * 50)
    
    errors = []
    
    # Test 1: DXEnvelope
    print("\n[Test 1] DXEnvelope...")
    try:
        env = DXEnvelope(
            rates=[0.01, 0.005, 0.002, 0.01],  # Slower rates for longer envelope
            levels=[1.0, 0.7, 0.5, 0.0],
            sample_rate=44100
        )
        # Step through envelope
        values = [env.step() for _ in range(1000)]
        assert env.value >= 0, "Envelope value should be non-negative"
        assert not env.is_finished(), "Envelope should not finish in 1000 samples with slow rates"
        
        # Test note_off
        env.note_off()
        assert env.stage == 3, "note_off should jump to stage 3"
        print("  ✓ DXEnvelope basic behavior OK")
    except Exception as e:
        errors.append(f"DXEnvelope: {e}")
        print(f"  ✗ {e}")
    
    # Test 2: Operator
    print("\n[Test 2] Operator...")
    try:
        env = DXEnvelope([0.5, 0.1, 0.05, 0.1], [1.0, 0.7, 0.5, 0.0], 44100)
        op = Operator(
            freq_ratio=1.0,
            detune_cents=0.0,
            level=0.8,
            envelope=env
        )
        # Process some samples
        samples = [op.process(440.0, 0.0, 44100) for _ in range(1000)]
        assert all(np.isfinite(s) for s in samples), "Output should be finite"
        assert max(abs(s) for s in samples) > 0, "Output should be non-zero"
        print("  ✓ Operator basic behavior OK")
    except Exception as e:
        errors.append(f"Operator: {e}")
        print(f"  ✗ {e}")
    
    # Test 3: Algorithm validation
    print("\n[Test 3] Algorithm validation...")
    try:
        # Valid algorithm
        validate_algorithm(ALGORITHMS[1])
        print("  ✓ Valid algorithm accepted")
        
        # Invalid: cycle
        try:
            invalid = {1: [2], 2: [1], 3: [], 4: [], 5: [], 6: []}
            validate_algorithm(invalid)
            errors.append("Algorithm: Should reject cycle")
            print("  ✗ Should have rejected cycle")
        except ValueError:
            print("  ✓ Cycle correctly rejected")
            
    except Exception as e:
        errors.append(f"Algorithm: {e}")
        print(f"  ✗ {e}")
    
    # Test 4: Topological sort
    print("\n[Test 4] Topological sort...")
    try:
        order = topological_sort(ALGORITHMS[1])
        # In algo 1 (serial chain), 6 should come first, 1 last
        assert order[0] == 6, "Operator 6 should be processed first in algo 1"
        assert order[-1] == 1, "Operator 1 should be processed last in algo 1"
        print(f"  ✓ Topo order for algo 1: {order}")
    except Exception as e:
        errors.append(f"Topological sort: {e}")
        print(f"  ✗ {e}")
    
    # Test 5: Carrier detection
    print("\n[Test 5] Carrier detection...")
    try:
        carriers_1 = get_carriers(ALGORITHMS[1])
        assert carriers_1 == [1], f"Algo 1 carrier should be [1], got {carriers_1}"
        print(f"  ✓ Algo 1 carriers: {carriers_1}")
        
        carriers_32 = get_carriers(ALGORITHMS[32])
        assert set(carriers_32) == {1, 2, 3, 4, 5, 6}, "Algo 32 should have all carriers"
        print(f"  ✓ Algo 32 carriers: {sorted(carriers_32)}")
    except Exception as e:
        errors.append(f"Carrier detection: {e}")
        print(f"  ✗ {e}")
    
    # Test 6: DX7II render
    print("\n[Test 6] DX7II render_note...")
    try:
        synth = DX7II(sample_rate=44100)
        audio = synth.render_note(freq=440.0, duration=0.5, velocity=0.8)
        
        assert len(audio) == 22050, f"Expected 22050 samples, got {len(audio)}"
        assert audio.dtype == np.float64, "Output should be float64"
        assert np.all(np.isfinite(audio)), "Output should be finite"
        assert np.max(np.abs(audio)) > 0, "Output should be non-silent"
        print(f"  ✓ Rendered {len(audio)} samples, peak={np.max(np.abs(audio)):.4f}")
    except Exception as e:
        errors.append(f"DX7II render: {e}")
        print(f"  ✗ {e}")
    
    # Test 7: Different algorithms
    print("\n[Test 7] Algorithm variations...")
    try:
        synth = DX7II(sample_rate=44100)
        peaks = {}
        for algo in [1, 5, 32]:
            synth.set_algorithm(algo)
            audio = synth.render_note(freq=440.0, duration=0.2, velocity=0.8)
            peaks[algo] = np.max(np.abs(audio))
        print(f"  ✓ Algo peaks: {peaks}")
    except Exception as e:
        errors.append(f"Algorithm variations: {e}")
        print(f"  ✗ {e}")
    
    # Test 8: Feedback stability
    print("\n[Test 8] Feedback stability...")
    try:
        synth = DX7II(sample_rate=44100)
        op6 = OperatorParams(
            freq_ratio=1.0, level=0.9, feedback=1.0,
            rates=[0.9, 0.1, 0.05, 0.1], levels=[1.0, 0.8, 0.6, 0.0]
        )
        synth.set_operator(6, op6)
        audio = synth.render_note(freq=440.0, duration=0.5, velocity=1.0)
        
        assert np.all(np.isfinite(audio)), "Feedback should not cause NaN/Inf"
        print(f"  ✓ Max feedback stable, peak={np.max(np.abs(audio)):.4f}")
    except Exception as e:
        errors.append(f"Feedback stability: {e}")
        print(f"  ✗ {e}")
    
    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
