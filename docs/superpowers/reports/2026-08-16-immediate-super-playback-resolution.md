# Immediate Super Playback Resolution — Offline Verification

Date: 2026-08-16

Branch: `background-independent-meter`

Verified parent: `4ed0f5e3f5b6`

Stable backup tag: `backup/stable-ca-audio-20260816` (`ffc6a51`)

## Scope

This checkpoint implements and verifies the approved behavior:

- Every detected super starts the configured music immediately.
- In `ALL_SUPERS`, later meter classification does not cancel playback.
- In `SA3_CA_ONLY`, a stable remaining meter tier of 1 or 2 resolves to
  `FILTERED_TIER` and fades the just-started track.
- A zero-meter event proceeds to result resolution. Hit or block keeps playback;
  whiff fades the track, or plays the mutually exclusive whiff sound when that
  option is enabled.
- Missing HUD recovery evidence resolves conservatively as `UNKNOWN` and fades.
- Event detection uses the pre-spend frame as the outcome baseline and suppresses
  super-effect frames before judging the recovered screen.

No changes were made to `main.py`, no runtime user configuration was committed,
and no unrelated files are part of this checkpoint.

## Reproducible offline evidence

All commands were run from `<workspace>` with an isolated temporary directory.
The project virtual environment was used for the recorded pytest runs.

### Synthetic replay and focused regression

The replay integration suite contains deterministic controls for immediate
playback, filtered SA1/SA2 fade, and zero-meter whiff fade. The focused regression
over capture, configuration, controller, engine, gauge, meter, outcome, panel,
audio, and replay behavior produced:

```text
675 passed, 5 failed in 96.76s
```

The five failures were all host-specific Windows `WinError 448` failures while
pytest attempted to create directory links inside its temporary directory. They
were not product assertions.

### Full regression

```text
1532 passed, 6 failed in 109.06s
```

The same five directory-link cases plus one equivalent HUD-profile directory-link
case failed with `WinError 448`. Re-running the complete suite while deselecting
only those six host-incompatible link-creation cases produced:

```text
1532 passed, 6 deselected in 119.26s
```

All Python modules in `src/sf6_music_trigger`, `tools/replay.py`, and
`tests/test_replay_integration.py` also passed `py_compile` using the bundled
Python runtime.

## Private recording replay

Private recordings were read locally, silently, and in memory. No audio device,
game process, network, Computer Use, upload, or external service was used.
Paths are intentionally omitted.

| Recording label | Detected events | Resolutions | Audio behavior |
| --- | ---: | --- | --- |
| pre-start control | 0 | none | none |
| CA zero-meter control | 1 | `HIT` | immediate stop/play; no fade |
| stable successful run | 3 | `HIT`, `HIT`, `HIT` | immediate stop/play for all three; no fade |
| mixed Jamie run | 3 | `UNKNOWN`, `FILTERED_TIER`, `WHIFF` | immediate stop/play for each, then one fade per non-keep result |

The mixed run resolved the second event as remaining meter tier 1, demonstrating
the `SA3_CA_ONLY` filter, and resolved the last event as whiff after the contact
checkpoint.

## Known environment issue

After the recorded regression runs, the historical desktop virtual-environment
launcher began pointing at a Python installation that is no longer present. This
is an environment/launcher issue rather than a source-code failure; static
compilation was repeated successfully with the bundled Python runtime. The new
candidate launcher must therefore use a currently valid interpreter or clearly
report that prerequisite.

## Acceptance boundary

SF6 real acceptance not yet performed.

The offline evidence supports a candidate build, but hit, block, whiff, and
scope-filter behavior must still be observed in a real SF6 session before this
checkpoint can be called hardware-accepted. MK1 is outside scope and must never be
started or operated.
