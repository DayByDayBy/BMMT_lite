# Unit Tests & Invariants

This document defines numerical, structural, and spectral properties that must hold true.

The goal is not exhaustive DSP correctness, but early detection of broken math.

---

## Operator Tests

### Phase Accumulation

Invariant:
- Phase increases monotonically
- Wraps at 2π

Test:
- Render N samples
- Assert no phase jumps > π

---

### Frequency Accuracy

Invariant:
- FFT peak within ±1 bin of expected frequency

Test:
- Render 1 second of A440
- FFT and locate max bin

---

## Envelope Tests

### Stage Progression

Invariant:
- Envelope reaches each target level
- Stage index increases monotonically

Test:
- Simulate envelope until completion

---

### Release Trigger

Invariant:
- note_off jumps to release stage

Test:
- Trigger note_off mid-attack
- Assert release begins

---

## Phase Modulation Tests

### Sideband Presence

Invariant:
- Modulated signal contains frequencies at f_c ± n·f_m

Test:
- Compare FFT of carrier-only vs modulated

---

## Algorithm Tests

### Validation

Invariant:
- All operators 1–6 present
- No cycles

Test:
- Valid algorithm passes
- Cyclic algorithm raises error

---

### Topological Order

Invariant:
- Modulators processed before carriers

Test:
- Verify ordering against graph

---

## Feedback Tests

### Stability

Invariant:
- Output remains finite
- No NaNs or Infs

Test:
- Render with feedback=1.0

---

## Integration Tests

### render_note Contract

Invariant:
- Output length == duration * sample_rate
- Mono signal

Test:
- Render multiple notes, assert shapes

---

## Spectral Regression (Optional)

Store:
- FFT magnitude snapshots for key patches

Invariant:
- Future changes stay within tolerance

Use only if stability becomes an issue.

