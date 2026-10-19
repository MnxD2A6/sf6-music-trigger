# Background-independent SF6 meter detection

## Problem

The current meter classifier compares an entire normalized grayscale crop to full and spent reference images. SF6's meter background is translucent, so a change of stage, character, camera pose, or lighting changes the crop even when the meter state has not changed. This prevents a full meter from arming the trigger.

## Scope

Replace the per-pixel grayscale comparison with a meter-only feature that measures the coloured segmented gauge. The change covers feature extraction, calibration persistence, panel diagnostics, and tests. It does not alter capture, audio playback, the old prototype `main.py`, or game input.

## Detection model

For each calibrated P1/P2 ROI, compare the stable CA/full and zero/spent batches at canonical resolution. Their persistent difference learns an immutable active-gauge mask and the full-state HSV colour signature. CA labels, fixed UI ornaments, stage backgrounds, and character pixels that do not change consistently between the two states are excluded from that mask. At runtime the classifier evaluates only masked pixels against the learned colour signature, divides the mask into fixed horizontal segment bands, and derives a filled-segment score. Classification uses calibrated full and spent score ranges with a fail-closed ambiguous zone. If the crop lacks enough matching masked pixels, the observation is hidden/ambiguous rather than full or spent.

The classifier must be invariant to unrelated scene pixels outside the learned active gauge mask. It must not rely on character art, stage background, title text, CA lettering, or a static reference screenshot.

## Calibration

The existing HUD workflow stays responsible for selecting the battle and P1/P2 meter ROIs. For each enabled side it prompts for two stable states: CA/full, then zero/spent. It collects five fresh frames per state, extracts meter-only features, rejects unstable or insufficiently separable samples, and atomically saves a new profile/reference pair only after every enabled side completes.

The UI exposes the current calibration phase and a safe reason when a phase cannot collect a valid meter feature. Existing profiles remain loadable through the legacy reference path; new profiles identify the meter-feature classifier explicitly.

## Failure handling

All calibration and classification failures are fail-closed: no arm and no audio trigger. A stage/character change after calibration must not itself invalidate a visible meter feature. Unsupported feature schema, hidden HUD, too few colour pixels, unstable samples, and insufficient full-versus-spent separation leave the side ambiguous and report a safe diagnostic category without scene paths or raw capture content.

## Tests and acceptance

Tests will prove:

- identical gauge segments over materially different synthetic backgrounds classify identically;
- full versus zero scores remain separated, while three-bar/partial states are ambiguous;
- background-only changes cannot arm or confirm an event;
- active-mask construction, HSV signature matching, segment position, empty/low-saturation ROI, and range-boundary mutations are rejected or killed by tests;
- calibration consumes fresh frames, rejects unstable batches, persists atomically, and preserves the prior profile on failure;
- existing reference profiles remain readable; and
- the complete offline suite passes.

Real acceptance is a single SF6-only run: with P1 enabled, CA/full must arm on both the training grid and an alternate stage, and P1 zero must cause the configured P1 audio to play. No MK1 or other application is operated. This spec does not claim that real acceptance has occurred.

## Offline evidence (2026-08-12)

- The background-independent model, persistence, observation, tracker, and calibration-panel suites ran together: **356 passed**.
- The complete offline suite then ran: **1415 passed**.
- The feature derives its active mask from the five fresh full and five fresh spent crops; schema-2 references persist that mask and its HSV signature, while schema-1 references remain legacy grayscale-only.
- These are offline results only. No SF6 game session, MK1 session, Computer Use action, or real audio-trigger acceptance is claimed here.
