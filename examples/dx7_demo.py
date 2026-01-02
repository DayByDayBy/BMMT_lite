"""
DX7II Synthesizer Demo

Demonstrates the DX7II FM synthesizer module with various presets
and sequencer integration.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# Add src to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bmmt.modelled_classics.yamaha import (
    DX7II,
    render_note,
    render_midi_note,
    list_presets,
    get_preset,
)


def demo_all_presets(output_dir: Path, sample_rate: int = 44100):
    """Render a short sample of each preset."""
    print("Rendering all presets...")
    
    for preset_name in list_presets():
        audio = render_note(
            freq=440.0,  # A4
            duration=1.5,
            velocity=0.8,
            preset=preset_name,
            sample_rate=sample_rate,
        )
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.8
        
        output_path = output_dir / f"dx7_{preset_name}.wav"
        sf.write(str(output_path), audio, sample_rate)
        print(f"  ✓ {preset_name}: {output_path}")


def demo_melody(output_dir: Path, sample_rate: int = 44100):
    """Render a simple melody using the e_piano preset."""
    print("\nRendering E-Piano melody...")
    
    # Simple melody: C-E-G-C (arpeggio)
    melody = [
        (60, 0.4),  # C4
        (64, 0.4),  # E4
        (67, 0.4),  # G4
        (72, 0.8),  # C5
    ]
    
    clips = []
    for midi_note, duration in melody:
        audio = render_midi_note(
            midi_note=midi_note,
            duration=duration,
            velocity=0.75,
            preset="e_piano_1",
            sample_rate=sample_rate,
        )
        clips.append(audio)
    
    # Concatenate with slight overlap for legato feel
    output = np.concatenate(clips)
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.8
    
    output_path = output_dir / "dx7_melody.wav"
    sf.write(str(output_path), output, sample_rate)
    print(f"  ✓ Melody: {output_path}")


def demo_chord(output_dir: Path, sample_rate: int = 44100):
    """Render a chord by layering multiple notes."""
    print("\nRendering Strings chord...")
    
    # C major chord
    chord_notes = [60, 64, 67, 72]  # C4, E4, G4, C5
    duration = 3.0
    
    # Render each note
    voices = []
    for midi_note in chord_notes:
        audio = render_midi_note(
            midi_note=midi_note,
            duration=duration,
            velocity=0.6,
            preset="strings_pad",
            sample_rate=sample_rate,
        )
        voices.append(audio)
    
    # Mix voices
    output = np.sum(voices, axis=0)
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.8
    
    output_path = output_dir / "dx7_chord.wav"
    sf.write(str(output_path), output, sample_rate)
    print(f"  ✓ Chord: {output_path}")


def demo_bass_line(output_dir: Path, sample_rate: int = 44100):
    """Render a simple bass line."""
    print("\nRendering Bass line...")
    
    # Simple bass pattern
    bass_notes = [
        (36, 0.3),  # C2
        (36, 0.3),
        (38, 0.3),  # D2
        (40, 0.6),  # E2
    ]
    
    clips = []
    for midi_note, duration in bass_notes:
        audio = render_midi_note(
            midi_note=midi_note,
            duration=duration,
            velocity=0.9,
            preset="bass_1",
            sample_rate=sample_rate,
        )
        clips.append(audio)
    
    output = np.concatenate(clips)
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.8
    
    output_path = output_dir / "dx7_bass.wav"
    sf.write(str(output_path), output, sample_rate)
    print(f"  ✓ Bass: {output_path}")


def demo_with_synth_instance():
    """Demonstrate direct DX7II class usage."""
    print("\nDirect DX7II usage example:")
    
    synth = DX7II(sample_rate=44100)
    
    # Load a preset
    synth.set_params(get_preset("bell"))
    
    # Render a note
    audio = synth.render_note(freq=880.0, duration=2.0, velocity=0.7)
    
    print(f"  ✓ Rendered {len(audio)} samples")
    print(f"  ✓ Peak amplitude: {np.max(np.abs(audio)):.4f}")
    print(f"  ✓ Duration: {len(audio) / 44100:.2f}s")
    
    return audio


def main():
    """Run all demos."""
    print("=" * 50)
    print("DX7II Synthesizer Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    sample_rate = 44100
    
    # Run demos
    demo_all_presets(output_dir, sample_rate)
    demo_melody(output_dir, sample_rate)
    demo_chord(output_dir, sample_rate)
    demo_bass_line(output_dir, sample_rate)
    demo_with_synth_instance()
    
    print("\n" + "=" * 50)
    print(f"All demos complete! Output in: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
