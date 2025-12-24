"""
Audio sequencer for timing, layering, and mixing.

Provides Track and Sequencer classes for offline audio composition.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any


class Track:
    """
    A single audio layer (e.g., kick drum, synth pad, bass).
    
    Manages:
    - Audio clips with timing and velocity
    - Per-track volume automation
    - Mixing clips into a single buffer
    
    Example:
        >>> track = Track("kick", sample_rate=44100)
        >>> track.add_clip(kick_audio, start_time=0.0, velocity=1.0)
        >>> track.add_clip(kick_audio, start_time=0.5, velocity=0.8)
        >>> rendered = track.render(duration=2.0)
    """
    
    def __init__(self, name: str, sample_rate: int = 44100):
        self.name = name
        self.sample_rate = sample_rate
        self._clips: List[Dict[str, Any]] = []
        self._volume_automation: Optional[Tuple[np.ndarray, float, float]] = None
        self._effect_chain: List[Tuple[Callable, Dict[str, Any]]] = []
    
    def add_clip(self, audio: np.ndarray, start_time: float, velocity: float = 1.0):
        """
        Add an audio clip at a specific time.
        
        Args:
            audio: Mono (1D) or stereo (2D, shape [N, 2]) audio
            start_time: Start time in seconds
            velocity: Amplitude multiplier (0.0-1.0+)
        """
        if audio.ndim not in [1, 2]:
            raise ValueError(f"Audio must be 1D (mono) or 2D (stereo), got shape {audio.shape}")
        if audio.ndim == 2 and audio.shape[1] != 2:
            raise ValueError(f"Stereo audio must have shape [N, 2], got {audio.shape}")
        if start_time < 0:
            raise ValueError(f"start_time must be >= 0, got {start_time}")
        if velocity < 0:
            raise ValueError(f"velocity must be >= 0, got {velocity}")
        
        self._clips.append({
            'audio': audio,
            'start_time': start_time,
            'velocity': velocity,
            'is_stereo': audio.ndim == 2
        })
    
    def add_effect(self, effect_func: Callable, **effect_params):
        """
        Add an effect to be applied during rendering.
        
        Args:
            effect_func: Function that takes (audio, sample_rate, **params) -> audio
            **effect_params: Parameters to pass to the effect function
            
        Example:
            >>> from bmmt.processing.spatial import apply_reverb
            >>> track.add_effect(apply_reverb, room_size=0.5, decay_time=2.0)
        """
        self._effect_chain.append((effect_func, effect_params))
    
    def add_volume_automation(self, envelope: np.ndarray, start_time: float, end_time: float):
        """
        Add volume automation (fade in/out, etc).
        
        Args:
            envelope: Array of volume multipliers (will be interpolated to fit)
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        if start_time >= end_time:
            raise ValueError(f"start_time must be < end_time")
        self._volume_automation = (envelope, start_time, end_time)
    
    def render(self, duration: float) -> np.ndarray:
        """
        Render track to stereo audio buffer.
        
        Args:
            duration: Length in seconds
            
        Returns:
            Stereo audio [N, 2]
        """
        num_samples = int(duration * self.sample_rate)
        output = np.zeros((num_samples, 2), dtype=np.float64)
        
        # Place all clips
        for clip in self._clips:
            start_sample = int(clip['start_time'] * self.sample_rate)
            if start_sample >= num_samples:
                continue  # Clip starts after track ends
            
            audio = clip['audio'] * clip['velocity']
            
            # Convert mono to stereo
            if not clip['is_stereo']:
                audio = np.column_stack([audio, audio])
            
            # Calculate how much of the clip fits
            clip_length = len(audio)
            available_space = num_samples - start_sample
            samples_to_copy = min(clip_length, available_space)
            
            # Mix into output
            output[start_sample:start_sample + samples_to_copy] += audio[:samples_to_copy]
        
        # Apply volume automation
        if self._volume_automation is not None:
            envelope, start_time, end_time = self._volume_automation
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            if start_sample < num_samples and end_sample > 0:
                # Clip to valid range
                start_sample = max(0, start_sample)
                end_sample = min(num_samples, end_sample)
                
                # Interpolate envelope to fit the automation range
                automation_length = end_sample - start_sample
                interpolated = np.interp(
                    np.linspace(0, 1, automation_length),
                    np.linspace(0, 1, len(envelope)),
                    envelope
                )
                
                # Apply to both channels
                output[start_sample:end_sample] *= interpolated[:, np.newaxis]
        
        # Apply effect chain
        for effect_func, effect_params in self._effect_chain:
            # Add sample_rate if the effect needs it
            if 'sample_rate' in effect_func.__code__.co_varnames:
                effect_params = {**effect_params, 'sample_rate': self.sample_rate}
            output = effect_func(output, **effect_params)
        
        return output


class Sequencer:
    """
    Manages multiple tracks and renders final mix.
    
    Example:
        >>> seq = Sequencer(bpm=120, sample_rate=44100)
        >>> kick = seq.add_track("kick")
        >>> kick.add_clip(kick_audio, start_time=0.0)
        >>> audio = seq.render(duration=4.0)
    """
    
    def __init__(self, bpm: float, sample_rate: int = 44100):
        self.bpm = bpm
        self.sample_rate = sample_rate
        self._tracks: List[Track] = []
        self._master_volume: float = 1.0
        self._master_automation: Optional[Tuple[np.ndarray, float, float]] = None
    
    def add_track(self, name: str) -> Track:
        """Create and return a new track."""
        track = Track(name, self.sample_rate)
        self._tracks.append(track)
        return track
    
    def set_master_volume(self, volume: float):
        """Set static master volume (multiplier)."""
        if volume < 0:
            raise ValueError(f"Master volume must be >= 0, got {volume}")
        self._master_volume = volume
    
    def add_master_automation(self, envelope: np.ndarray, start_time: float, end_time: float):
        """Add master volume automation."""
        if start_time >= end_time:
            raise ValueError(f"start_time must be < end_time")
        self._master_automation = (envelope, start_time, end_time)
    
    def beats_to_seconds(self, beats: float) -> float:
        """Convert beats to seconds based on BPM."""
        return beats * (60.0 / self.bpm)
    
    def seconds_to_samples(self, seconds: float) -> int:
        """Convert seconds to sample count."""
        return int(seconds * self.sample_rate)
    
    def render(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Render all tracks to final stereo mix.
        
        Args:
            duration: Length in seconds (if None, auto-calculate from clips)
            
        Returns:
            Stereo audio [N, 2]
        """
        # Auto-calculate duration if not provided
        if duration is None:
            max_end_time = 0.0
            for track in self._tracks:
                for clip in track._clips:
                    clip_end = clip['start_time'] + (len(clip['audio']) / self.sample_rate)
                    max_end_time = max(max_end_time, clip_end)
            duration = max_end_time
            
            if duration == 0.0:
                raise ValueError("Cannot auto-calculate duration: no clips added")
        
        # Render each track
        num_samples = int(duration * self.sample_rate)
        mix = np.zeros((num_samples, 2), dtype=np.float64)
        
        for track in self._tracks:
            track_audio = track.render(duration)
            mix += track_audio
        
        # Apply master volume
        mix *= self._master_volume
        
        # Apply master automation
        if self._master_automation is not None:
            envelope, start_time, end_time = self._master_automation
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            if start_sample < num_samples and end_sample > 0:
                start_sample = max(0, start_sample)
                end_sample = min(num_samples, end_sample)
                
                automation_length = end_sample - start_sample
                interpolated = np.interp(
                    np.linspace(0, 1, automation_length),
                    np.linspace(0, 1, len(envelope)),
                    envelope
                )
                
                mix[start_sample:end_sample] *= interpolated[:, np.newaxis]
        
        # Soft clip to prevent hard clipping
        mix = np.clip(mix, -1.0, 1.0)
        
        return mix

# Step sequencer utility functions

def create_trigger_pattern(
    pattern: List[bool],
    step_duration: float,
    duration: float,
    sample_rate: int = 44100
) -> List[float]:
    """
    Generate trigger times from boolean pattern.
    
    Args:
        pattern: List of booleans (True = trigger, False = rest)
        step_duration: Duration of each step in seconds
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        List of trigger times in seconds
        
    Example:
        >>> pattern = [True, False, True, False]
        >>> triggers = create_trigger_pattern(pattern, 0.25, 1.0)
        >>> triggers
        [0.0, 0.5]
    """
    triggers = []
    num_repeats = int(np.ceil(duration / (step_duration * len(pattern))))
    
    for repeat in range(num_repeats):
        for i, triggered in enumerate(pattern):
            if triggered:
                time = (repeat * len(pattern) + i) * step_duration
                if time < duration:
                    triggers.append(time)
    
    return triggers


def create_euclidean_rhythm(
    hits: int,
    steps: int,
    duration: float,
    sample_rate: int = 44100
) -> List[float]:
    """
    Generate Euclidean rhythm (evenly distributed hits).
    
    Uses Bjorklund algorithm for even distribution.
    
    Args:
        hits: Number of hits to distribute
        steps: Total number of steps
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        List of trigger times in seconds
        
    Example:
        >>> triggers = create_euclidean_rhythm(3, 8, 2.0)
        >>> # Distributes 3 hits across 8 steps: [1,0,0,1,0,0,1,0]
    """
    if hits > steps:
        raise ValueError(f"hits ({hits}) cannot exceed steps ({steps})")
    if hits == 0:
        return []
    
    # Bjorklund algorithm (simple version)
    pattern = [True] * hits + [False] * (steps - hits)
    
    # Distribute evenly
    if hits > 0 and steps > 0:
        # Create groups
        groups = [[x] for x in pattern]
        
        # Distribute
        while len(set(map(len, groups))) > 1:
            # Find groups of different lengths
            long = [g for g in groups if len(g) == max(map(len, groups))]
            short = [g for g in groups if len(g) == min(map(len, groups))]
            
            if not long or not short:
                break
            
            # Append short to long
            num_to_append = min(len(long), len(short))
            for i in range(num_to_append):
                long[i].extend(short.pop(0))
            
            groups = long + short
        
        # Flatten
        pattern = [item for group in groups for item in group]
    
    # Convert to trigger times
    step_duration = duration / steps
    triggers = [i * step_duration for i, triggered in enumerate(pattern) if triggered]
    
    return triggers


def beats_to_triggers(
    beat_positions: List[float],
    bpm: float,
    duration: float,
    sample_rate: int = 44100
) -> List[float]:
    """
    Convert beat positions to trigger times.
    
    Args:
        beat_positions: List of beat positions (e.g., [0, 0.5, 1.0, 1.5])
        bpm: Beats per minute
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        List of trigger times in seconds
    """
    beat_duration = 60.0 / bpm
    triggers = [beat * beat_duration for beat in beat_positions if beat * beat_duration < duration]
    return triggers
