# SF6 Meter-Level Super Detection Design

## Goal

Replace fractional colour-fill changes as the primary Super activation signal
with a stable reading of SF6's visible meter level (`0`, `1`, `2`, `3`, or
`CA`). A level decrease confirms a Super; a level increase is ordinary meter
gain and must never start music.

The change remains screen-only, local, SF6-only, and side-independent. It does
not read game memory, inject into the game, send inputs, use network access,
operate MK1, modify the old prototype `main.py`, or change the audio playback
boundary.

This design supersedes only the activation and tier-resolution sections of
`2026-08-16-immediate-super-playback-resolution-design.md`. Its post-activation
hit/block/whiff rules and audio mutual-exclusion rules remain in force.

## Evidence and Root Cause

The 468.6-second recording
`实际测试，误触，没触发，出发错误音频.mp4` was replayed through the exact
20 FPS `_r5` observation and tracker path. It emitted ten events:

| Time | Side | Video result |
| --- | --- | --- |
| 56.467 s | P1 | False: ordinary hit raised meter `1 -> 2` |
| 137.767 s | P1 | True: CA spent `3 -> 0` |
| 154.800 s | P2 | False: round-end `WIN` transition |
| 188.000 s | P1 | False: ordinary hit raised meter `0 -> 1` |
| 234.467 s | P1 | True: CA spent `3 -> 0` |
| 297.067 s | P2 | False: round-end `WIN` transition |
| 322.867 s | P2 | False: meter `0 -> 1` during a strong visual effect |
| 387.800 s | P1 | True: CA spent `3 -> 0` |
| 399.067 s | P2 | True: SA3/CA spent to zero |
| 448.000 s | P1 | False: ordinary hit raised meter `1 -> 2` |

The colour model measures only the fractional bar beside the integer level.
Natural gain across an integer boundary wraps that fractional bar from full to
empty, which is visually identical to the old spend signal. Combat effects
also disrupt the calibrated crop, so the `_r5` visual-disruption guard cannot
separate natural gain from a Super. Conversely, spending one or two whole
levels may leave the fractional fill nearly unchanged, so SA1/SA2 can be
missed. Audio routing follows the detector side correctly; an incorrect P1 or
P2 detection therefore produces the corresponding incorrect track.

## Approaches Considered

### 1. Tighten colour and disruption thresholds

Rejected. The long replay proves that a natural level wrap plus a normal hit
effect can satisfy the same colour and structure conditions as Super startup.
Threshold tuning can trade false positives for missed activations but cannot
recover the missing direction of the integer-level change.

### 2. Recognize Super cinematics or character-specific effects

Rejected. This would require per-character and per-move assets, would be
sensitive to skins, stages, camera cuts, patches, and localization, and would
conflict with the all-character 1v1 requirement.

### 3. Read the meter level and detect its direction

Selected. The `0/1/2/3/CA` marker is part of the stable HUD, is independent of
character and stage, directly distinguishes gain from spend, and also provides
the spent tier without waiting for an entire cinematic to finish.

## Level Feature

For each side, derive a narrow level-marker ROI around the outer endpoint of
the existing meter-bar ROI. Evidence from the real-match replay requires one
quarter of the calibrated meter width outward plus one fifth inward:

- P1 extends outward to the left and slightly into the bar on the right;
- P2 mirrors that geometry;
- the vertical bounds remain those of the meter ROI; and
- all bounds are clamped to the captured client frame.

The current screenshot profile therefore gains the missing digit area without
requiring users to repeat calibration. The HUD confirmation overlay will show
both the existing meter bar and the derived level-marker band, but calibration
will not add another manual prompt.

The crop is resized to a fixed height while preserving aspect ratio. At that
canonical resolution, the extractor finds the white level glyph only when
it is adjacent to the calibrated side-colour outline. This rejects unrelated
white stage pixels. It classifies the small fixed alphabet using geometric
features rather than captured game templates:

- `0`: one enclosed hole;
- `1`: a single narrow glyph;
- `2`: lower-left occupancy and no hole;
- `3`: right-side lower occupancy and no hole; and
- `CA`: two full-height, vertically aligned letter components, normalized to
  numeric level three.

Single components must remain in the side-specific digit anchor. Partial CA
letters, stretched effect shapes, and small extra components are rejected or
reduced to the one trustworthy full-height component instead of guessed.

The extractor returns an immutable `MeterLevelReading` containing the side,
numeric level, confidence, and visibility. It does not retain or save the
source crop. Low confidence, clipped glyphs, competing components, strong
occlusion, and unsupported shapes return `UNKNOWN` rather than guessing.

## Stable Level Tracking

Each side owns an independent level history. A downward candidate becomes
stable after three consecutive matching trustworthy readings at the
configured 20 FPS, giving a nominal decrease-confirmation latency of
0.10-0.15 seconds. A level must first remain trustworthy for ten samples
(0.5 seconds) before it may serve as a spend baseline; this prevents short
round-intro and effect glyphs from arming the detector.

The state rules are:

1. the first stable level establishes a provisional baseline and emits
   nothing; only ten trustworthy samples make it spend-ready;
2. a stable increase updates the baseline and emits nothing, and the new
   higher baseline must satisfy the same ten-sample dwell;
3. an unchanged stable level emits nothing;
4. a direct stable decrease of one, two, or three levels confirms one Super;
5. `CA` is treated as level three, so `CA -> 0` confirms a three-level spend;
6. `UNKNOWN` pauses comparison for at most nine consecutive samples (under
   0.5 seconds at 20 FPS) but cannot itself emit; the tenth consecutive
   unknown sample clears the baseline;
7. a long HUD loss, round-end transition, capture reset, geometry change, or
   profile change clears the baseline so levels are never compared across
   rounds or capture sessions; and
8. after a confirmed decrease, the post-spend level becomes the new baseline,
   preventing duplicate events.

The fractional colour feature remains responsible for HUD/structure
trustworthiness and as auxiliary evidence, but it can no longer confirm a
Super without a stable downward level transition. A reduced numeric level
that conflicts with a still-high trustworthy colour fill is treated as
`UNKNOWN`: zero is rejected above 0.45 normalized fill, and one/two are
rejected above 0.78. This cross-signal gate prevents animated or unrelated
same-colour shapes from becoming false level decreases while accepting the
recorded real HUD recovery states.
A strong visual effect, clean fractional wrap, or `WIN` transition therefore
cannot start music.

## Domain and Engine Data Flow

`GaugeObservation` will carry an optional trustworthy numeric meter level in
addition to the existing colour evidence. `GaugeTracker` will retain the last
stable level and attach `meter_before`, `meter_after`, and `levels_spent` to a
confirmed event. Existing event identity, side, frame IDs, and profile ID are
preserved.

The engine and audio port continue to receive side-bound events and actions.
The audio port is not changed. P1 and P2 remain separate channels, and an event
from one side cannot select or stop the other side's track.

Playback policy becomes deterministic at confirmation time:

- `ALL_SUPERS`: a confirmed one-, two-, or three-level decrease starts that
  side's main track once;
- `SA3_CA_ONLY`: only a confirmed three-level decrease starts the main track;
  one- and two-level decreases remain diagnostic events but do not produce a
  provisional audible start; and
- no increase, same-level reading, unknown reading, or round reset produces an
  audio action.

This intentionally removes the old need to start SA1/SA2 provisionally in
`SA3_CA_ONLY`: the reliable level delta is already known within the activation
confirmation window. Existing post-activation hit/block/whiff behavior remains
unchanged for events that pass the selected scope.

## Compatibility and Calibration

- Existing schema-2 colour profiles remain loadable. Their level ROI is
  derived at runtime from the calibrated meter ROI.
- The existing screenshot profile requires no manual migration.
- Legacy grayscale-only profiles retain their existing fail-closed path and do
  not claim reliable SA1/SA2 support until recalibrated to a colour profile.
- Calibration continues to collect full and spent meter samples. Its ROI
  preview adds the derived level band so a user can correct an unusually tight
  meter selection before saving.
- If the derived band is outside the frame or repeatedly unreadable, the panel
  reports `LEVEL_UNKNOWN`/`气槽等级不可读` and leaves that side disarmed.

## Failure and Privacy Handling

- Ambiguous glyphs fail closed; they cannot be converted into a decrease.
- A bounded short obstruction preserves the last baseline but cannot emit
  until a stable post-obstruction level is visible.
- A long obstruction or round transition clears the baseline rather than
  comparing the next round's initial meter with the prior round.
- Desktop, task view, stale frames, backward timestamps, and capture errors
  cannot advance level stability.
- Diagnostics expose only side, state, numeric level when trustworthy, and a
  privacy-safe reason. They never expose audio paths or captured pixels.
- No level crops or templates are written into the release package.

## Test Strategy

Implementation is test-driven:

1. pure level-feature tests for `0/1/2/3/CA`, both sides, different scales,
   unrelated bright backgrounds, partial clipping, flashes, and unknown input;
2. pure stability tests for matching samples, gains, one/two/three-level
   spends, duplicates, short occlusion, long reset, and round boundaries;
3. tracker tests proving colour-only full-to-empty wraps and visual disruption
   cannot emit without a downward level transition;
4. engine/audio tests proving `ALL_SUPERS`, `SA3_CA_ONLY`, side isolation, and
   unchanged hit/block/whiff behavior;
5. offline replay of the new 468.6-second recording, requiring exactly the four
   verified Super events above and zero events at the six documented false
   timestamps;
6. replay of `杰米测试3.mp4`, `终于基本成功了，谢谢你S02.mp4`, and the prior
   training/real-match fixtures to prevent regression;
7. the complete offline suite and portable EXE self-check; and
8. only after offline completion, an SF6-only real acceptance covering
   natural `0 -> 1` and `1 -> 2` gains, SA1, SA2, SA3/CA, hit, block, whiff,
   round end, and both P1/P2 tracks. MK1 and other applications remain out of
   scope.

## Acceptance Criteria

- The long replay emits exactly four verified Super activations and none of
  the six documented false activations.
- Natural meter gain never starts music, including during ordinary hit and
  Drive Impact effects.
- Round-end `WIN` transitions never start music.
- `ALL_SUPERS` detects a stable one-, two-, or three-level decrease for either
  side with no more than 0.25 seconds nominal confirmation latency.
- `SA3_CA_ONLY` produces audible playback only for a three-level decrease.
- Correct P1/P2 events select only the configured track for that screen side.
- Hit/block/whiff resolution and mutual exclusion between whiff sound and fade
  remain unchanged.
- Existing profiles load without manual migration, `main.py` remains
  untouched, and the previous `_r5` source is recoverable from
  `backup/pre-meter-level-detector-20260822`.

## Backup and Real-World Status

The pre-change source is commit
`31774fd3b2070d9055749e83dee402fd6aec2566`, protected by local tag
`backup/pre-meter-level-detector-20260822`.

This document records a design approved for planning. It does not claim that
the implementation, portable build, or SF6 real acceptance has completed.
