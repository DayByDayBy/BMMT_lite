# Reference Sound Guide

This document defines what "working" should sound like at each development stage.

These are qualitative and semi-quantitative targets, not strict emulation goals.

---

## 1. Single Operator (Carrier Only)

Sound:
- Pure sine wave
- No beating, no drift

Expectations:
- A440 sounds identical to a test oscillator
- FFT shows single peak at fundamental

Failure modes:
- Phase reset clicks
- Pitch wobble

---

## 2. Envelope Applied

Sound:
- Smooth attack
- Audible decay and release

Expectations:
- No zipper noise
- Release tails fade naturally

Failure modes:
- Stepped amplitude
- Envelope never reaching target

---

## 3. Simple FM (2 Operators)

Sound:
- Metallic or bell-like
- Clear change in brightness with mod depth

Expectations:
- Sidebands symmetric around carrier

Reference:
- Classic DX electric piano partials

Failure modes:
- Only amplitude change (means AM, not PM)

---

## 4. Algorithm Variations

Sound:
- Different algorithms clearly distinguishable
- Some dull, some aggressive

Expectations:
- Parallel carriers sound fuller
- Long modulation chains sound complex

Failure modes:
- Algorithms sound identical

---

## 5. Feedback

Sound:
- Growl, edge, instability at high settings

Expectations:
- Feedback increases high-frequency content
- Still bounded

Reference:
- DX7 basses, distorted EPs

Failure modes:
- Exploding NaNs
- Silence at high feedback

---

## 6. Velocity & Keyboard Scaling

Sound:
- Harder velocity = brighter sound
- Higher notes = slightly thinner modulators

Expectations:
- Musical, not dramatic

Failure modes:
- Volume-only velocity response

---

## Golden Reference Patches

You do NOT need to match these exactly, but they are useful sanity checks:

- DX7 EPiano 1
- Tubular Bells
- Solid Bass
- Glass Pad

If your output evokes the same *class* of sound, you are on track.

