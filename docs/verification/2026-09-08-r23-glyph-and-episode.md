# R23: glyph integrity and candidate episode fixes

## Backup and scope

Baseline: R22, commit `83de74fa5d09a6c1f7c5a6c7b3d8e4ce58b7dc20`.
Before editing, source bundle/ZIP, R22 release ZIP, local config and diagnostics
were saved under the sibling `backups/r23-before-20260908` directory. R22 itself
is not overwritten. No game launch/input, network, capture-backend, HUD-profile,
music-file, running-config or unrelated-project changes were made.

## Reproduced causes

1. P2's unchanged visible 0 was read as 3, 2 or 1. The digit's violet-blue glow
   differs from the cyan fractional meter. Per-pixel colour masking also clipped
   intact white digits into other shapes. This fabricated downward transitions
   and caused baseline loss/missed activations. R22 samples at 662.0, 662.2 and
   749.0 seconds demonstrate this without relying on tracker output for labels.
2. In the post-KO scene around 303.767 seconds, bright background pixels joined
   P1's 2 and satisfied the old wide-1 heuristic, manufacturing a 2-to-1 spend.
3. Cancelling a candidate discarded its identity/episode; the unchanged older
   baseline could issue another candidate and replay. The live log has four
   P2 candidates around monotonic 32844.739-32848.088, with three timeouts.
   A separate reproducible race consumed an authority transition on the very
   frame a candidate was returned, losing confirmation and causing timeout.
4. R22 emitted `contact_not_established`, but the diagnostic evidence allowlist
   did not accept it. This dropped those records and latched the warning visible
   in the video. It was not necessarily a filesystem/logger-worker failure.

## Minimal changes

- Preserve connected white glyphs when a real local colour seed supports them;
  reject edge-connected flashes and unrelated/opposite-colour components.
- Recognize standard P2 violet-blue glyph outlines separately from cyan gauge
  fill. Calibration and fractional fill extraction remain unchanged. Retain
  the prior calibrated extraction if violet effects connect to the ROI edge.
- Require the vertical stem before accepting a widened 1. Contaminated shapes
  abstain instead of claiming another meter level.
- Keep a retired candidate identity until recovery or a new established meter
  baseline. Late authority retains that identity, so the real audio router does
  not restart it. State remains independent per side and resets with capture.
- Positive matching lower-level evidence can renew the 0.75 s soft deadline,
  bounded to 3 s total. This does not lengthen initial playback waiting.
  Confirmation wins when candidate/timeout and authority resolve together.
- Accept the missing closed diagnostic code; arbitrary/private text is still
  rejected. R22 UNKNOWN-does-not-fade and old-event audio ownership are retained.

## Verification

Regressions were observed failing before the corresponding changes: partial
colour outlines, P2 violet zero, post-KO false 1, timeout/retry, lost same-frame
confirmation, and the missing diagnostic code. The tests use real pixel reading,
trackers and scheduling/routing. A recording audio port performs no playback.

Final full regression command:

`../.r21-runtime/Scripts/python.exe -m pytest -q -p no:cacheprovider --tb=short`

Result: **1907 passed in 111.03 seconds**. This run required the normal local
test environment; the sandbox account cannot read the pre-existing pytest temp
directory/Tcl runtime. An escalated read-only Tcl probe succeeded. No ACL or
global runtime configuration was changed.

### R22 recording replay

Input: 851.933 seconds, 2528 x 1438, 30 fps. SF6 client crop: (480,242),
1920 x 1080. Sample two of each three decoded frames (20 fps), use the unchanged
calibrated profile and real production reader/tracker. This is offline replay,
not a recreation of capture cadence or a live audio-latency measurement.

| Production version | Candidates | Confirmations | Cancellations |
|---|---:|---:|---:|
| R22 baseline | 13 | 20 | 2 |
| R23 final | 11 | 15 | 0 |

These are transition counts, not a claimed all-video precision/recall score.
The final replay removes the verified false spend around 303 s and the false
P2 spend around 662 s. It detects the visually checked late P2 SA1 animations
around 702, 733 and 769 s. Four confirmations (567.600, 702.200, 733.300,
769.367) have no early candidate; residual startup delay is still an acceptance
item. Earlier P1 CA/SA3 detection times are preserved at 114.167, 226.267,
363.467, 625.167 and 788.667 seconds.

Local replay artifacts are in sibling `.r20-video-analysis`: `r23_trace.py`,
`r22-baseline.json`, `r23-release-replay.json`, original/final observation JSONL,
and glyph/scene inspection sheets. Intermediate experimental files are not
release evidence. Both sides' small HUD-only regression fixtures are committed
under `tests/fixtures`; they contain no desktop or audio paths.

SHA256:

- R22 video: `b70661d5b6c655f905217baf73d020f985a8b365c4a719d60e0e5326a905844f`
- R22 scalar observations: `0dfb67d621634faf446e0f9375cc246f3837d9d000173a061d87fba0793f64b4`
- Final R23 scalar observations: `585b6d97958e987fa15dfd294f937c7b52514b9476af288fbf56b3116f086dea`

## Remaining acceptance boundary

No new live SF6 or audible end-to-end acceptance. No claim of all-character
1v1 completion, zero false triggers, or instant playback. Four late-only cases
remain. Current automatic whiff recognition stays deliberately disabled:
UNKNOWN preserves the music; the UI options and sound preview remain, but
missing hit evidence is not treated as WHIFF. Hit/block sensing is still the
existing conservative observer. Further capture/performance/audio-latency and
hit/block/whiff testing must be reported separately.

R23 is a separate test candidate, not a mandatory upgrade or overwrite.
