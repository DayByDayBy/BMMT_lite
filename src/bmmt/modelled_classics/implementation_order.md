# implementation order

## Phase 0: Agent Assumptions

Before starting, the agent should assume the following defaults unless explicitly overridden:
- `sample_rate` is provided at DX7II construction and is immutable.
- Envelope step progression uses a first-order exponential smoothing approximation.
- Keyboard scaling affects modulator amplitude; carrier amplitude remains unscaled unless specified.
- Feedback curve uses the default mapping: `feedback_map(x) = min(0.99, 0.5 * (2 ** (4 * x) - 1))`.
- Rendered audio for each note is mono and of length `duration * sample_rate`.
- The agent should maintain a `dev_diary.md` throughout execution to record assumptions, decisions, and any issues encountered.

## Phase 1: Single Operator
- Implement sine oscillator and phase accumulator.
- Implement DXEnvelope class and 4-stage envelope.
- Implement level_map() for logarithmic amplitude.
- Validate envelope progression numerically.

## Phase 2: Phase Modulation
- Add single modulator -> carrier path.
- Verify sideband frequencies in FFT.
- Test velocity scaling of modulator depth.

## Phase 3: Multi-Operator Algorithm
- Implement graph-based algorithm routing.
- Implement topological sort.
- Validate operator order against algorithm.
- Ensure invalid algorithms raise errors.

## Phase 4: Full Voice Rendering
- Implement render_note() interface.
- Combine multiple operators per algorithm.
- Verify mono output and note length.

## Phase 5: Feedback and Fine Tuning
- Add feedback path to operator 6.
- Apply feedback_map() curve and bounds.
- Validate stability (no NaNs/Infs).
- Test extreme velocities and envelope edge cases.

## Phase 6: Sequencer Integration
- Hook DX7II module to Track-based sequencer.
- Convert sequencer triggers to render_note() calls.
- Apply track effects and automation.
- Render final mix and compare against REFERENCE_SOUND.md.

## Phase 7: Polyphony & Optimization
- Implement note-stealing and voice management if required.
- Optional: oversampling for reduced aliasing.
- Validate integration with multiple tracks.

---

**Exit Criteria per Phase:**
- Phase 1–5: Audio matches numerical + spectral invariants in UNIT_TESTS.md.
- Phase 6: Clips correctly aligned with sequencer triggers; final mix audible and matches reference character.
- Phase 7: Polyphony works without instability or clipping; performance within acceptable limits.