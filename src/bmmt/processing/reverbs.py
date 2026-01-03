import numpy as np
from scipy import signal


def _validate_signal_params(input_signal: np.ndarray, sample_rate: int) -> None:
    if not isinstance(input_signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    if len(input_signal) == 0:
        raise ValueError("Input signal cannot be empty")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")


def _ensure_stereo(input_signal: np.ndarray) -> np.ndarray:
    if input_signal.ndim == 1:
        return np.column_stack([input_signal, input_signal])
    if input_signal.ndim == 2 and input_signal.shape[1] == 2:
        return input_signal
    raise ValueError("Signal must be mono (1D) or stereo (2D with 2 channels)")


def _clip_signal(input_signal: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    return np.clip(input_signal, -threshold, threshold)


def _pitch_shift_channel(channel: np.ndarray, ratio: float) -> np.ndarray:
    if abs(ratio - 1.0) < 0.001:
        return channel.copy()

    original_length = len(channel)
    new_indices = np.arange(original_length) / ratio
    new_indices = np.clip(new_indices, 0, original_length - 1)
    shifted = np.interp(new_indices, np.arange(original_length), channel)

    if len(shifted) != original_length:
        if len(shifted) > original_length:
            shifted = shifted[:original_length]
        else:
            padded = np.zeros(original_length)
            padded[: len(shifted)] = shifted
            shifted = padded

    return shifted


def _allpass_filter(channel: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    if delay_samples <= 0 or delay_samples >= len(channel):
        return channel.copy()

    output = np.zeros_like(channel)
    buffer = np.zeros(delay_samples)

    for i in range(len(channel)):
        buf_out = buffer[i % delay_samples]
        y = -gain * channel[i] + buf_out
        buffer[i % delay_samples] = channel[i] + gain * y
        output[i] = y

    return output


def _comb_filter(channel: np.ndarray, delay_samples: int, feedback: float, damping: float) -> np.ndarray:
    if delay_samples <= 0 or delay_samples >= len(channel):
        return channel.copy()

    output = np.zeros_like(channel)
    buffer = np.zeros(delay_samples)
    filter_store = 0.0

    for i in range(len(channel)):
        buf_out = buffer[i % delay_samples]
        filter_store = (1.0 - damping) * buf_out + damping * filter_store
        y = channel[i] + filter_store * feedback
        buffer[i % delay_samples] = y
        output[i] = y

    return output


def _decay_to_feedback(delay_samples: int, decay_time: float, sample_rate: int) -> float:
    if decay_time <= 0.0:
        return 0.0
    return float(10.0 ** (-3.0 * delay_samples / (decay_time * sample_rate)))


def _schroeder_reverb_mono(
    channel: np.ndarray,
    sample_rate: int,
    size: float,
    decay_time: float,
    damping: float,
    diffusion: float,
    pre_delay: float,
    comb_delays_s: list[float],
    allpass_delays_s: list[float],
    allpass_gain: float,
) -> np.ndarray:
    if len(channel) < 10:
        return channel.copy()

    if pre_delay <= 0.0:
        predelayed = channel
    else:
        pd = int(pre_delay * sample_rate)
        if pd <= 0:
            predelayed = channel
        else:
            predelayed = np.zeros_like(channel)
            predelayed[pd:] = channel[:-pd]

    comb_sum = np.zeros_like(channel)
    for d_s in comb_delays_s:
        d = int(max(1, round(d_s * sample_rate * size)))
        fb = _decay_to_feedback(d, decay_time, sample_rate)
        comb_sum += _comb_filter(predelayed, d, fb, damping)

    comb_sum *= 1.0 / max(1, len(comb_delays_s))

    ap_gain = float(np.clip(allpass_gain * (0.3 + 0.7 * diffusion), 0.05, 0.85))
    out = comb_sum
    for d_s in allpass_delays_s:
        d = int(max(1, round(d_s * sample_rate * size)))
        out = _allpass_filter(out, d, ap_gain)

    return out


def _tone_filter_mono(channel: np.ndarray, bass: float, treble: float, sample_rate: int) -> np.ndarray:
    if len(channel) < 100:
        return channel.copy()

    bass = float(np.clip(bass, 0.0, 1.0))
    treble = float(np.clip(treble, 0.0, 1.0))

    hp_cutoff = 40.0 + (1.0 - bass) * 360.0
    lp_cutoff = 2000.0 + treble * 16000.0

    nyquist = sample_rate / 2

    out = channel

    hp_norm = hp_cutoff / nyquist
    if 0.001 < hp_norm < 0.999:
        b, a = signal.butter(2, hp_norm, btype="high")
        out = signal.filtfilt(b, a, out)

    lp_norm = lp_cutoff / nyquist
    if 0.001 < lp_norm < 0.999:
        b, a = signal.butter(2, lp_norm, btype="low")
        out = signal.filtfilt(b, a, out)

    return out


def apply_hall_reverb(
    input_signal: np.ndarray,
    size: float = 0.8,
    decay_time: float = 3.0,
    damping: float = 0.35,
    diffusion: float = 0.7,
    pre_delay: float = 0.02,
    mix: float = 0.35,
    sample_rate: int = 44100,
) -> np.ndarray:
    _validate_signal_params(input_signal, sample_rate)

    size = float(np.clip(size, 0.1, 1.0))
    decay_time = float(np.clip(decay_time, 0.2, 20.0))
    damping = float(np.clip(damping, 0.0, 0.95))
    diffusion = float(np.clip(diffusion, 0.0, 1.0))
    pre_delay = float(np.clip(pre_delay, 0.0, 0.2))
    mix = float(np.clip(mix, 0.0, 1.0))

    stereo = _ensure_stereo(input_signal)
    dry = stereo

    if mix == 0.0:
        return _clip_signal(dry)

    comb_delays_s_l = [0.0297, 0.0371, 0.0411, 0.0437, 0.005, 0.011, 0.017]
    comb_delays_s_r = [0.0301, 0.0367, 0.0403, 0.0441, 0.006, 0.012, 0.018]
    allpass_delays_s_l = [0.005, 0.0017, 0.0006]
    allpass_delays_s_r = [0.0047, 0.0019, 0.0007]

    wet_l = _schroeder_reverb_mono(
        stereo[:, 0],
        sample_rate,
        size,
        decay_time,
        damping,
        diffusion,
        pre_delay,
        comb_delays_s_l,
        allpass_delays_s_l,
        allpass_gain=0.7,
    )
    wet_r = _schroeder_reverb_mono(
        stereo[:, 1],
        sample_rate,
        size,
        decay_time,
        damping,
        diffusion,
        pre_delay,
        comb_delays_s_r,
        allpass_delays_s_r,
        allpass_gain=0.7,
    )

    wet = np.column_stack([wet_l, wet_r])
    out = dry * (1.0 - mix) + wet * mix
    return _clip_signal(out)


def apply_plate_reverb(
    input_signal: np.ndarray,
    decay_time: float = 2.2,
    pre_delay: float = 0.01,
    bass_damping: float = 0.6,
    treble_damping: float = 0.6,
    mix: float = 0.3,
    size: float = 0.65,
    diffusion: float = 0.85,
    sample_rate: int = 44100,
) -> np.ndarray:
    _validate_signal_params(input_signal, sample_rate)

    decay_time = float(np.clip(decay_time, 0.2, 15.0))
    pre_delay = float(np.clip(pre_delay, 0.0, 0.1))
    bass_damping = float(np.clip(bass_damping, 0.0, 1.0))
    treble_damping = float(np.clip(treble_damping, 0.0, 1.0))
    mix = float(np.clip(mix, 0.0, 1.0))
    size = float(np.clip(size, 0.1, 1.0))
    diffusion = float(np.clip(diffusion, 0.0, 1.0))

    stereo = _ensure_stereo(input_signal)
    dry = stereo

    if mix == 0.0:
        return _clip_signal(dry)

    damping = float(np.clip(0.15 + 0.7 * treble_damping, 0.0, 0.95))

    comb_delays_s_l = [0.0123, 0.0151, 0.0179, 0.0197, 0.0213, 0.0229]
    comb_delays_s_r = [0.0129, 0.0147, 0.0183, 0.0203, 0.0219, 0.0235]
    allpass_delays_s_l = [0.0031, 0.0011, 0.0006, 0.0003]
    allpass_delays_s_r = [0.0029, 0.0012, 0.0007, 0.0004]

    wet_l = _schroeder_reverb_mono(
        stereo[:, 0],
        sample_rate,
        size,
        decay_time,
        damping,
        diffusion,
        pre_delay,
        comb_delays_s_l,
        allpass_delays_s_l,
        allpass_gain=0.75,
    )
    wet_r = _schroeder_reverb_mono(
        stereo[:, 1],
        sample_rate,
        size,
        decay_time,
        damping,
        diffusion,
        pre_delay,
        comb_delays_s_r,
        allpass_delays_s_r,
        allpass_gain=0.75,
    )

    wet_l = _tone_filter_mono(wet_l, bass=bass_damping, treble=1.0, sample_rate=sample_rate)
    wet_r = _tone_filter_mono(wet_r, bass=bass_damping, treble=1.0, sample_rate=sample_rate)

    wet = np.column_stack([wet_l, wet_r])
    out = dry * (1.0 - mix) + wet * mix
    return _clip_signal(out)


def apply_spring_reverb(
    input_signal: np.ndarray,
    mix: float = 0.35,
    springs: int = 3,
    tension: float = 0.5,
    shake: float = 0.0,
    bass: float = 0.6,
    treble: float = 0.7,
    sample_rate: int = 44100,
) -> np.ndarray:
    _validate_signal_params(input_signal, sample_rate)

    mix = float(np.clip(mix, 0.0, 1.0))
    springs = int(np.clip(springs, 1, 8))
    tension = float(np.clip(tension, 0.0, 1.0))
    shake = float(np.clip(shake, 0.0, 1.0))
    bass = float(np.clip(bass, 0.0, 1.0))
    treble = float(np.clip(treble, 0.0, 1.0))

    stereo = _ensure_stereo(input_signal)
    dry = stereo

    if mix == 0.0:
        return _clip_signal(dry)

    base_decay = 0.8 + (1.0 - tension) * 3.0
    damping = float(np.clip(0.65 - 0.45 * tension, 0.05, 0.95))

    def spring_channel(channel: np.ndarray, offset: int) -> np.ndarray:
        wet = np.zeros_like(channel)
        for s in range(springs):
            delay_s = 0.010 + 0.0037 * s + 0.0003 * offset
            delay_samples = int(max(1, round(delay_s * sample_rate)))
            fb = float(np.clip(_decay_to_feedback(delay_samples, base_decay, sample_rate), 0.0, 0.98))
            wet += _comb_filter(channel, delay_samples, fb, damping)

        wet *= 1.0 / springs

        ap1 = int(max(1, round((0.0029 + 0.0002 * offset) * sample_rate)))
        ap2 = int(max(1, round((0.0011 + 0.0001 * offset) * sample_rate)))
        wet = _allpass_filter(wet, ap1, 0.65)
        wet = _allpass_filter(wet, ap2, 0.6)

        wet = _tone_filter_mono(wet, bass=bass, treble=treble, sample_rate=sample_rate)
        return wet

    wet_l = spring_channel(stereo[:, 0], 0)
    wet_r = spring_channel(stereo[:, 1], 1)

    if shake > 0.0 and len(stereo) > 100:
        rng = np.random.default_rng(12345)
        noise = rng.normal(0.0, 1.0, len(stereo)).astype(float)
        burst_env = np.zeros(len(stereo))
        burst_len = max(64, int(0.006 * sample_rate))
        num_bursts = max(1, int(shake * 6))
        positions = rng.integers(0, max(1, len(stereo) - burst_len), size=num_bursts)
        window = np.hanning(burst_len)
        for p in positions:
            burst_env[p : p + burst_len] += window
        shake_sig = noise * burst_env * (0.05 * shake)
        wet_l += spring_channel(shake_sig, 2) * 0.6
        wet_r += spring_channel(shake_sig, 3) * 0.6

    wet = np.column_stack([wet_l, wet_r])
    out = dry * (1.0 - mix) + wet * mix
    return _clip_signal(out)


def apply_shimmer_reverb(
    input_signal: np.ndarray,
    size: float = 0.85,
    diffusion: float = 0.75,
    decay: float = 6.0,
    feedback: float = 0.45,
    mix: float = 0.35,
    pitch_a_semitones: float = 12.0,
    pitch_b_semitones: float = 0.0,
    pitch_blend: float = 0.7,
    modulation: float = 0.35,
    tone: float = 0.7,
    sample_rate: int = 44100,
) -> np.ndarray:
    _validate_signal_params(input_signal, sample_rate)

    size = float(np.clip(size, 0.1, 1.0))
    diffusion = float(np.clip(diffusion, 0.0, 1.0))
    decay = float(np.clip(decay, 0.2, 30.0))
    feedback = float(np.clip(feedback, 0.0, 0.95))
    mix = float(np.clip(mix, 0.0, 1.0))
    pitch_blend = float(np.clip(pitch_blend, 0.0, 1.0))
    modulation = float(np.clip(modulation, 0.0, 1.0))
    tone = float(np.clip(tone, 0.0, 1.0))

    stereo = _ensure_stereo(input_signal)
    dry = stereo

    if mix == 0.0:
        return _clip_signal(dry)

    base_wet = apply_hall_reverb(
        stereo,
        size=size,
        decay_time=max(0.6, decay * 0.6),
        damping=0.35,
        diffusion=diffusion,
        pre_delay=0.02,
        mix=1.0,
        sample_rate=sample_rate,
    )

    base_mono = np.mean(base_wet, axis=1)

    ratio_a = 2 ** (float(pitch_a_semitones) / 12.0)
    ratio_b = 2 ** (float(pitch_b_semitones) / 12.0)

    pitched_a = _pitch_shift_channel(base_mono, ratio_a)
    pitched_b = _pitch_shift_channel(base_mono, ratio_b)
    pitched = pitched_a * pitch_blend + pitched_b * (1.0 - pitch_blend)

    nyquist = sample_rate / 2
    lp_cutoff = 2000.0 + tone * 16000.0
    lp_norm = lp_cutoff / nyquist
    if len(pitched) > 100 and 0.001 < lp_norm < 0.999:
        b, a = signal.butter(2, lp_norm, btype="low")
        pitched = signal.filtfilt(b, a, pitched)

    pitched_stereo = np.column_stack([pitched, pitched])

    if modulation > 0.0:
        from .effects import apply_chorus

        chorus_mix = min(1.0, 0.15 + 0.65 * modulation)
        pitched_stereo = apply_chorus(
            pitched_stereo,
            rate=0.25 + 0.75 * modulation,
            depth=0.001 + 0.004 * modulation,
            voices=3,
            mix=chorus_mix,
            sample_rate=sample_rate,
        )

    fb_input = pitched_stereo * feedback

    fb_passes = int(np.clip(1 + int(feedback * 3), 1, 4))
    fb_wet = base_wet.copy()
    for _ in range(fb_passes):
        fb_wet = fb_wet + apply_hall_reverb(
            fb_input,
            size=size,
            decay_time=decay,
            damping=0.4,
            diffusion=diffusion,
            pre_delay=0.0,
            mix=1.0,
            sample_rate=sample_rate,
        ) * 0.5
        fb_mono = np.mean(fb_wet, axis=1)
        fb_pitched = _pitch_shift_channel(fb_mono, ratio_a)
        fb_input = np.column_stack([fb_pitched, fb_pitched]) * feedback

    wet = fb_wet
    out = dry * (1.0 - mix) + wet * mix
    return _clip_signal(out)


def apply_gated_reverb(
    input_signal: np.ndarray,
    threshold: float = 0.12,
    attack: float = 0.003,
    hold: float = 0.08,
    release: float = 0.12,
    reverb_mix: float = 0.45,
    reverb_decay: float = 1.8,
    size: float = 0.75,
    diffusion: float = 0.7,
    sample_rate: int = 44100,
) -> np.ndarray:
    _validate_signal_params(input_signal, sample_rate)

    threshold = float(np.clip(threshold, 0.0, 1.0))
    attack = float(np.clip(attack, 0.0001, 1.0))
    hold = float(np.clip(hold, 0.0, 2.0))
    release = float(np.clip(release, 0.0001, 2.0))
    reverb_mix = float(np.clip(reverb_mix, 0.0, 1.0))
    reverb_decay = float(np.clip(reverb_decay, 0.2, 10.0))
    size = float(np.clip(size, 0.1, 1.0))
    diffusion = float(np.clip(diffusion, 0.0, 1.0))

    stereo = _ensure_stereo(input_signal)
    dry = stereo

    wet = apply_plate_reverb(
        stereo,
        decay_time=reverb_decay,
        pre_delay=0.0,
        bass_damping=0.6,
        treble_damping=0.75,
        mix=1.0,
        size=size,
        diffusion=diffusion,
        sample_rate=sample_rate,
    )

    ctrl = np.mean(np.abs(stereo), axis=1)

    attack_samps = max(1, int(round(attack * sample_rate)))
    hold_samps = int(round(hold * sample_rate))
    release_samps = max(1, int(round(release * sample_rate)))

    gate = np.zeros(len(ctrl))

    state_open = False
    hold_counter = 0
    gain = 0.0
    attack_step = 1.0 / attack_samps
    release_step = 1.0 / release_samps

    for i in range(len(ctrl)):
        if ctrl[i] >= threshold:
            state_open = True
            hold_counter = hold_samps
        else:
            if hold_counter > 0:
                hold_counter -= 1
            else:
                state_open = False

        if state_open:
            gain = min(1.0, gain + attack_step)
        else:
            gain = max(0.0, gain - release_step)

        gate[i] = gain

    gated_wet = wet * gate[:, None]
    out = dry * (1.0 - reverb_mix) + gated_wet * reverb_mix
    return _clip_signal(out)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    try:
        duration = 1.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        spring = apply_spring_reverb(test_signal, mix=0.4, springs=3, tension=0.6, shake=0.2)
        plate = apply_plate_reverb(test_signal, decay_time=2.5, pre_delay=0.015, mix=0.35)
        hall = apply_hall_reverb(test_signal, size=0.9, decay_time=3.5, mix=0.35)
        shimmer = apply_shimmer_reverb(test_signal, feedback=0.5, pitch_a_semitones=12.0, mix=0.35)
        gated = apply_gated_reverb(test_signal, threshold=0.1, attack=0.003, hold=0.08, release=0.15)

        print("✓ reverbs module basic tests")
        print(f"✓ Spring: {spring.shape}")
        print(f"✓ Plate: {plate.shape}")
        print(f"✓ Hall: {hall.shape}")
        print(f"✓ Shimmer: {shimmer.shape}")
        print(f"✓ Gated: {gated.shape}")

    except Exception as e:
        print(f"✗ Error in reverbs tests: {e}")
