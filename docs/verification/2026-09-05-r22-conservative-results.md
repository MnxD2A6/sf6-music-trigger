# R22 conservative result handling

## Scope and backup

Based on R21 source commit 64822786632d1e5e3b7e1d700c2750fd58dc195b.
Before changes, source bundle/archive, R21 release ZIP, local configuration and
diagnostics were copied to the sibling `backups/r22-before-20260905` directory.
The user's R21 installation and audio files are unchanged.

## Evidence and changes

R21 analysis found repeated candidate cancellation/restart around the single
Jamie cinematic near video 199-202 seconds, a visibly contacting Juri SA1 near
181-186 seconds classified WHIFF, and buff SA2 activations treated as missing
contact. These approximate visual times are not sample-accurate audio alignment.

Regression tests reproduced the following code paths before modification:

1. UNKNOWN directly emitted FADE. It now leaves playback unchanged.
2. Unchanged contact features at the checkpoint emitted WHIFF with confidence
   0.8. Absence of contact evidence now emits UNKNOWN with confidence zero.
   The current observer therefore does not automatically issue WHIFF. The UI
   settings/preview and explicit WHIFF audio handling remain for a future
   evidence-backed detector. This is a deliberately limited mitigation, not a
   completed whiff detector.
3. A single recovered meter sample cancelled a candidate. Recovery now needs
   three consecutive samples, like other level conflicts. The existing deadline
   and confirmation thresholds are unchanged; this does not solve all delays or
   repeated triggers.
4. An old event outcome could fade a newer candidate track on the same side.
   Playback ownership now prevents a mismatched old outcome from issuing audio
   actions. A new confirmed event also replaces the old pending result window.

Tests exercise real tracker/scheduler/observer logic and engine routing with a
recording audio sink; they do not play sound or send game input.

Final offline verification: `python -m pytest -q -p no:cacheprovider --tb=short`
completed with **1881 passed in 102.60 seconds**. `git diff --check` passed.
Nine new regression cases were observed failing before the corresponding fixes.

## Acceptance boundary

No SF6 launch, game input, capture backend changes or live acceptance in this
change. Existing private-video regressions are historical recordings, not proof
that R21.mp4 or all characters are fixed. Full R21 frame/audio alignment, genuine
health-bar/contact extraction, buff classification, delayed confirmation and
all-character 1v1 acceptance remain outstanding. Reset and low-tier filtering
can still fade by design. Do not market this candidate as a complete fix.
