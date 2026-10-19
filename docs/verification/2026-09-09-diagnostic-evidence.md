# 2026-09-09 diagnostic evidence follow-up

## Scope

This follow-up is limited to the local diagnostic contract. It does not change
meter recognition, candidate timing, audio policy, capture, game input, or the
live-acceptance boundary.

## Reproduced gap

The explicit `WHIFF` routing regression used the closed evidence code
`contact_window_elapsed`, but the diagnostic evidence allowlist did not accept
that code. Audio routing still produced the expected fade, while construction
of the corresponding `OUTCOME` diagnostic record failed closed and latched a
diagnostic warning. The event result was therefore not observable in the
JSONL lifecycle log.

## Minimal change

`contact_window_elapsed` is now a permitted closed evidence code. The existing
privacy-safe identifier and evidence validation remain unchanged. A regression
asserts that a confirmed event followed by an explicit WHIFF emits both the
`CONFIRMED` and `OUTCOME` records with the same event identity, the WHIFF label,
and no diagnostic warning.

## Verification

- Focused engine/diagnostic regression: **118 passed in 1.40s**.
- Full offline suite after the change: **1907 passed in 102.33s**.

This is offline evidence only. Automatic WHIFF recognition remains disabled;
no SF6 launch, input, network access, or audible/live acceptance was performed.
