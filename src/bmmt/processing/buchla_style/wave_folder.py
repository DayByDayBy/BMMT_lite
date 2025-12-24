import numpy as np

# ============================================================
# Physical Component Models
# ============================================================

VT = 0.02585   # Thermal voltage (~300K)
IS = 1e-12     # Diode saturation current
N  = 1.75      # Ideal factor
BJT_VT = 0.026 # For differential pair

def diode_current(v):
    """Shockley diode equation."""
    return IS * (np.exp(v / (N * VT)) - 1.0)

def diode_pair(v):
    """Symmetrical diode pair used in feedback."""
    return diode_current(v) - diode_current(-v)

def diff_pair(v, gain=1.0):
    """Differential pair soft-limiting transfer."""
    return np.tanh(v * gain / (2 * BJT_VT))

def opamp_soft_clip(v, limit=3.0):
    """Softly saturating op-amp stage."""
    return limit * np.tanh(v / limit)

# ============================================================
# Newton–Raphson Solver
# ============================================================

def newton_solve(func, dfunc, x0, max_iter=12, tol=1e-8):
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        dfx = dfunc(x)
        if abs(dfx) < 1e-12:
            break
        step = fx / dfx
        x -= step
        if abs(step) < tol:
            break
    return x

# ============================================================
# Single Buchla Fold Stage
# ============================================================

def buchla_fold_sample(x, fold_amount=1.0, dp_gain=1.0, bias=0.0):
    """Single analog-accurate fold stage with ZDF."""

    dp = diff_pair(x + bias, dp_gain)

    # f(y) = y + fold_amount * diode_pair(y) - dp
    def f(y):
        return y + fold_amount * diode_pair(y) - dp

    # approximate derivative for NR
    def df(y):
        return 1.0 + fold_amount * (1.0 / (N * VT)) * np.cosh(y / (N * VT))

    y0 = dp
    y = newton_solve(f, df, y0)

    out = opamp_soft_clip(y, 3.0)
    return out

# ============================================================
# Oversampling (8x)
# ============================================================

def oversample_8x(x, process_fn):
    up = np.repeat(x, 8)
    y_up = np.zeros_like(up)
    for i in range(len(up)):
        y_up[i] = process_fn(up[i])
    return y_up[::8]

# ============================================================
# Tape-Hysteresis Model
# ============================================================

class WavefolderState:
    """Stores tape-memory state for hysteresis."""

    def __init__(self):
        self.m = 0.0  # magnetization / hysteresis memory

def apply_tape_hysteresis(x, state, tape_amount=0.0, alpha=0.01):
    """Update tape memory and return effective input."""
    state.m = (1 - alpha) * state.m + alpha * x
    return x + tape_amount * state.m

# ============================================================
# Multi-Stage Buchla Wavefolder
# ============================================================

def buchla_wavefolder(
        x,
        drive=2.0,
        bias=0.0,
        fold_amount=1.5,
        dp_gain=2.0,
        num_base_stages=3,
        extra_stage_amount=0.0,
        tape_amount=0.0,
        tape_tau=0.01,
        auto_gain_compensate=True,
        sr=48000):
    """
    Analog-accurate Buchla-style wavefolder with:
    - Hybrid multi-stage chain
    - Tape-like hysteresis (continuous)
    - 8x oversampling
    - Optional automatic gain compensation
    """

    x = np.asarray(x, dtype=float)

    # Pre-drive
    x_d = x * drive

    # Alpha for tape memory based on tau
    alpha = 1.0 - np.exp(-1.0 / (sr * tape_tau))

    state = WavefolderState()

    # Processing function per sample
    def process_sample(s):
        # Tape hysteresis
        s_eff = apply_tape_hysteresis(s, state, tape_amount, alpha)

        # Multi-stage folding
        v = s_eff
        for _ in range(num_base_stages):
            v = buchla_fold_sample(v, fold_amount=fold_amount, dp_gain=dp_gain, bias=bias)

        # Optional extra hybrid stage
        if extra_stage_amount > 0.0:
            v_extra = buchla_fold_sample(v, fold_amount=fold_amount, dp_gain=dp_gain, bias=bias)
            v = (1 - extra_stage_amount) * v + extra_stage_amount * v_extra

        return v

    # Oversampled processing
    y = oversample_8x(x_d, process_sample)
    
    # Automatic gain compensation
    if auto_gain_compensate:
        # Compensate for drive gain and typical folding amplification
        # The factor accounts for both input drive and typical output expansion
        compensation = 1.0 / (drive * 0.65)
        y = y * compensation
    
    return y


def apply_wavefolder(
        signal,
        drive=2.0,
        bias=0.0,
        fold_amount=1.5,
        dp_gain=2.0,
        num_base_stages=3,
        extra_stage_amount=0.0,
        tape_amount=0.0,
        tape_tau=0.01,
        auto_gain_compensate=True,
        sample_rate=44100):
    return buchla_wavefolder(
        x=signal,
        drive=drive,
        bias=bias,
        fold_amount=fold_amount,
        dp_gain=dp_gain,
        num_base_stages=num_base_stages,
        extra_stage_amount=extra_stage_amount,
        tape_amount=tape_amount,
        tape_tau=tape_tau,
        auto_gain_compensate=auto_gain_compensate,
        sr=sample_rate,
    )

# ============================================================
# example Usage
# ============================================================

if __name__ == "__main__":
    import soundfile as sf

    # Load audio buffer
    x, sr = sf.read("input.wav")

    # Process with hybrid Buchla wavefolder
    y = buchla_wavefolder(
        x,
        drive=2.5,
        fold_amount=1.2,
        dp_gain=2.0,
        bias=0.05,
        num_base_stages=3,
        extra_stage_amount=0.3,
        tape_amount=0.5,
        tape_tau=0.01,
        auto_gain_compensate=True,
        sr=sr
    )

    # Save output
    sf.write("folded.wav", y, sr)





# Parameter Recommendations for Tape-Like Warmth:

# Parameter         Recommended Range
# drive             2–3
# fold_amount       1–1.5
# dp_gain           1.5–3
# bias              -0.1 – 0.1
# num_base_stages   3
# extra_stage_amount  0–0.5
# tape_amount       0.3–0.7
# tape_tau          0.005–0.02 s



# Higher drive → more harmonic richness
# tape_amount > 0.5 → strong hysteresis, subtle lag
# extra_stage_amount → increases “overdrive” and additional folds
# auto_gain_compensate → maintains consistent perceived loudness





# ...








# --------------------
# optional plotting utility


# import numpy as np
# import matplotlib.pyplot as plt

# def plot_wavefolder_curve(
#         wavefolder_fn,
#         x_range=(-3, 3),
#         num_points=5000,
#         drive=2.0,
#         fold_amount=1.5,
#         dp_gain=2.0,
#         bias=0.0,
#         num_base_stages=3,
#         extra_stage_amount=0.3,
#         tape_amount=0.5,
#         tape_tau=0.01,
#         sr=48000):
#     """
#     Plot input vs output transfer curve of the wavefolder.
#     Also simulates tape-hysteresis by doing a forward and backward sweep.
#     """

#     # Create forward and backward sweeps
#     x_forward = np.linspace(x_range[0], x_range[1], num_points)
#     x_backward = np.linspace(x_range[1], x_range[0], num_points)
    
#     # Process forward sweep
#     y_forward = wavefolder_fn(
#         x_forward,
#         drive=drive,
#         fold_amount=fold_amount,
#         dp_gain=dp_gain,
#         bias=bias,
#         num_base_stages=num_base_stages,
#         extra_stage_amount=extra_stage_amount,
#         tape_amount=tape_amount,
#         tape_tau=tape_tau,
#         sr=sr
#     )

#     # Reset internal tape memory
#     state = None  # Assuming your module uses a fresh WavefolderState per call

#     # Process backward sweep
#     y_backward = wavefolder_fn(
#         x_backward,
#         drive=drive,
#         fold_amount=fold_amount,
#         dp_gain=dp_gain,
#         bias=bias,
#         num_base_stages=num_base_stages,
#         extra_stage_amount=extra_stage_amount,
#         tape_amount=tape_amount,
#         tape_tau=tape_tau,
#         sr=sr
#     )

#     # Plot curves
#     plt.figure(figsize=(8,6))
#     plt.plot(x_forward, y_forward, label="Forward sweep", color='blue')
#     plt.plot(x_backward, y_backward, label="Backward sweep", color='orange')
#     plt.title("Buchla-Style Wavefolder Transfer Curve with Tape Hysteresis")
#     plt.xlabel("Input amplitude")
#     plt.ylabel("Output amplitude")
#     plt.grid(True)
#     plt.legend()
#     plt.show()


# # =========================
# # Example Usage
# # =========================
# if __name__ == "__main__":
#     # Assuming buchla_wavefolder is already imported from previous module
#     plot_wavefolder_curve(
#         wavefolder_fn=buchla_wavefolder,
#         drive=2.5,
#         fold_amount=1.2,
#         dp_gain=2.0,
#         bias=0.05,
#         num_base_stages=3,
#         extra_stage_amount=0.3,
#         tape_amount=0.5,
#         tape_tau=0.01
#     )
