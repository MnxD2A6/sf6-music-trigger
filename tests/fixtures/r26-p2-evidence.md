# R26 P2 bounded glyph veto — handoff, 2026-09-10

Production source frozen: `src/sf6_music_trigger/meter_level_features.py` only.
No edits to `gauge_tracker.py`; its pre-existing R23 color-flicker change remains.
Other agents own engine/audio/outcome. No release/config/music changes.

## Root cause and change

On R25 119.367–120s, the permissive saturation <=65 white mask includes tinted
stage highlights. Morphological closing joins the open `2` into a loop, and
the fixed-alphabet classifier assigns 3 or 0 with confidence .9. At the later
windows, background appended to `1` similarly produces 3 or 0. Temporal
agreement cannot distinguish these repeated wrong glyphs from an activation.

The original extraction is retained. A P2 result of 0 or 3 is checked against
the same extractor with saturation <=20. A *readable conflicting* digit vetoes
the original result to unknown/confidence zero. The strict reading never
replaces the original digit, and an unreadable strict core does not invent one.
P1 is untouched. There are no new tracker timers or disabled sides.

A strict-only extraction trial was rejected because it broke the existing
wide-glowing-one and partial-CA controls. The veto preserves those controls.
This is a bounded contamination check, not complete OCR or all-character safety.
Contamination on which both masks agree can remain undetected.

## Red / green evidence

Runtime: `../.r21-runtime/Scripts/python.exe -B`; pytest `-p no:cacheprovider`.

- Initial 12 raw glyph tests: **7 failed / 5 passed** against unchanged source.
  Failures were actual 2 -> 3/0 at 119s and actual 1 -> 3 at 361.9s.
- Original-reader control on initial 15 raw sequences: **1 failed / 14 passed**.
  R25 119.600 candidate 2->0 and 119.667 confirmation 2->0 reappeared.
- Independent 30fps comparison: original reader emits a false candidate at
  360.3667 and confirmation at 360.400 (3->1); the veto emits neither.
- Final new tests: **60 passed in 25.91s** (18 glyph samples, 21 sequences at
  both 20fps and 30fps). Negative windows assert no candidate AND no confirmation.
- Existing glyph checks plus initial new glyph tests: **130 passed / 4 xfailed**.
- `git diff --check` for the modified source passed.

## Controls and limitations

R25 confirmed SA1 timestamps at 20fps are unchanged:
42.500, 89.000, 251.967, 309.567, 347.467.
P1 CA candidate/confirmation are unchanged at 430.167/430.200.
R23 confirmed positives covered: 36.300, 90.367, 117.100, 167.400, 219.600,
245.600, 289.100, 291.700, 370.400, 384.800, 424.867.
The R23 255–260s KO window has no candidate or confirmation at either rate.

One candidate latency difference: R23 167.167 -> 167.367 (+200ms), while
confirmation remains 167.400. Other selected 20fps positive event timestamps
are unchanged. This comes from rejected contaminated frames, not a timer change.

R25 329–335 and 358–364 windows emit no candidate/confirmation after the fix
at 20/30fps. The exact recorded runtime events at 333.345 and 362.277 were not
reproduced in the original 20fps recording replay. Raw regressions include
331.4/331.467/332.533 false 3 and 362.0/362.167 false 0; all now abstain.
Do not equate these meter tests with an end-to-end no-audio pass. Parent owns
the full R25 engine/audio replay and final acceptance.

All four existing strict XFAIL cases remain XFAIL, with **zero XPASS**:
`test_r23_p2_real_glyphs_do_not_turn_two_into_zero[2-2]`, `[3-2]`, `[4-2]`,
and `[10-None]`. Existing tests were not edited.

## Shared-worktree full-suite snapshot

**2153 passed / 12 failed / 4 xfailed in 183.71s.** This was a moving shared
worktree; concurrent engine/outcome work was in progress. No P2 tests failed.

- 6 `test_block_fade.py` failures (four owner-side variants, two delayed-block).
- 3 `test_r26_ca_safety.py` failures: `[2-0.75]`, latest changed baseline,
  preconfirmation attacker damage.
- 1 private `test_replay_integration.py` outcome mismatch (HIT vs UNKNOWN).
- 2 repository write-permission failures: wheel hygiene test attempted a build
  and was blocked copying `build/lib/.../audio.py`; ignored-screenshot-directory
  test could not create its directory. No package was produced by this task.

The first full-suite attempt hit inaccessible shared `pytest-of-<user>`; a
fresh uniquely named OS temporary directory allowed the above run. The extra
whole-video meter comparison was stopped during R23 after the parent took
ownership of full replay; it is not claimed as a completed full-video audit.

## Fixture provenance / reproduction

`r26-p2-glyphs.npz` contains 18 lossless P2 HUD crops from R25, with timestamps
and manually assigned visible levels. `r26-r23-*.npz` / `r26-r25-*.npz` contain
21 continuous HUD-only windows at original 30fps, frame indices, dimensions,
crop bounds, and the R25 calibrated meter references. No audio, full frames,
account identity, or absolute recording paths are stored. Total NPZ size:
191,253,598 bytes. Read with `allow_pickle=False`.

R25 source is 1902x1104; R23 is 1916x1092. Both remove exactly 26 titlebar pixels.
Tests reconstruct only the meter+glyph ROI on the original-size black canvas
and invoke production `observe_gauge` and `GaugeTracker`; cached JSON scalars
are not used as truth or as replay input.

Run the two new test modules with pytest. Their `__main__` paths document local
fixture extraction; the sequence extractor skips existing fixtures unless
explicitly passed `--refresh`. Do not refresh while tests are reading fixtures.

Owned files: the single source file, `tests/test_r26_p2_glyphs.py`,
`tests/test_r26_p2_sequences.py`, this note, `r26-p2-glyphs.npz`, and the 21
`r26-r23-*`/`r26-r25-*` NPZs. Other r26 fixtures/tests belong to parallel work.
