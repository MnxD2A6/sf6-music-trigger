# Whiff Outcome Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a confirmed SF6 CA spend, conservatively distinguish visible contact from a whiff and either fade the side's main music or replace it with a configured whiff sound.

**Architecture:** A small stateful `OutcomeObserver` owns one bounded post-spend window per side. It samples only fresh, geometry-compatible frames while the HUD is visible, compares the opponent-health and optional training-HUD result regions with the confirmation-frame baseline, suppresses whiff on stable contact evidence, and emits `WHIFF` only when the window expires without contact. The existing engine routes decisions through the side's `AudioScheduler`; the audio port gains a fixed-channel fade action. Configuration stores only `play_whiff_sound`, so default fade and sound playback cannot be simultaneously enabled.

**Tech Stack:** Python 3.12, NumPy, pygame mixer channels, Tkinter, pytest.

## Global Constraints

- Screen pixels remain the only SF6 signal; no input, memory, injection, network, or game-file access.
- Outcome observation starts only from an existing `ConfirmedSuperEvent`.
- Hidden, stale, lost, or geometrically incompatible frames never prove contact or whiff.
- Low-confidence contact suppresses whiff instead of guessing.
- Default behavior is a 1500 ms fade. Enabling whiff sound replaces the side channel without issuing fade.
- P1 and P2 keep independent outcome state, configuration, audio channels, and diagnostics.
- No change to legacy `main.py`, no MK1 operation, no network access, and no committed audio/game media.

---

### Task 1: Conservative post-spend outcome state

**Files:**
- Create: `src/sf6_music_trigger/outcome.py`
- Test: `tests/test_outcome.py`

**Interfaces:**
- Consumes: `ConfirmedSuperEvent`, `CapturedFrame`, `HudProfile`, and per-frame HUD visibility.
- Produces: `OutcomeLabel`, `OutcomeDecision`, and `OutcomeObserver.begin()/update()/reset()`.

- [ ] **Step 1: Write failing state-machine tests**

Cover P1/P2 isolation, no decision before a confirmed event, stable health/training-HUD change cancelling whiff, single-frame noise not proving contact, exact deadline whiff, hidden frames pausing classification, and reset clearing pending windows.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_outcome.py -v`

Expected: collection succeeds and tests fail because `sf6_music_trigger.outcome` does not exist.

- [ ] **Step 3: Implement the minimal observer**

Use owned baseline crops, two consecutive visible contact frames, a 4-second deadline, exact side ownership, and immutable decisions. Evaluate contact before the deadline branch on the first visible frame after a hidden interval.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m pytest tests/test_outcome.py -v`

Expected: all outcome tests pass.

### Task 2: Mutually exclusive whiff policy and fade action

**Files:**
- Modify: `src/sf6_music_trigger/config.py`
- Modify: `src/sf6_music_trigger/audio.py`
- Test: `tests/test_domain_config.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `OutcomeDecision` and extended `SideAudioConfig`.
- Produces: `AudioKind.FADE`, strict `AudioAction.fade_ms`, `AudioScheduler.on_outcome()`, and `PygameAudioPort` fixed-channel fade execution.

- [ ] **Step 1: Write failing configuration and audio tests**

Cover schema defaults, JSON round-trip, rejected invalid whiff path/volume/fade values, diagnostic redaction, HIT/BLOCK no-op, default WHIFF fade, whiff-sound-only PLAY, disabled side no-op, pygame `Channel.fadeout(1500)`, and failure privacy.

- [ ] **Step 2: Run exact tests and verify RED**

Run the new exact nodes in `tests/test_domain_config.py` and `tests/test_audio.py`.

Expected: assertions fail because whiff fields, `FADE`, and outcome scheduling are absent.

- [ ] **Step 3: Implement the minimal policy**

Persist `whiff_track`, `whiff_volume`, `play_whiff_sound`, and `whiff_fade_ms`; derive `fade_on_whiff = not play_whiff_sound`. A WHIFF returns exactly one action: PLAY when sound is enabled, otherwise FADE. Add no queue, cache, retry, or global mixer stop.

- [ ] **Step 4: Run audio/config tests and verify GREEN**

Run: `python -m pytest tests/test_domain_config.py tests/test_audio.py -v`

Expected: all selected tests pass.

### Task 3: Engine and panel wiring

**Files:**
- Modify: `src/sf6_music_trigger/engine.py`
- Modify: `src/sf6_music_trigger/panel.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_panel_view_model.py`

**Interfaces:**
- Consumes: `OutcomeObserver` decisions and extended side config.
- Produces: one outcome update per pending side and visible panel controls for whiff track/sound/fade.

- [ ] **Step 1: Write failing engine and view-model tests**

Cover begin-only-on-event, later frame contact/whiff routing, audio failure containment, runtime reset clearing pending state, config/view-model round-trip, track validation only when whiff sound is enabled, and sound/fade mutual exclusion.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_engine.py tests/test_panel_view_model.py -k "outcome or whiff" -v`

Expected: new tests fail on missing wiring/fields.

- [ ] **Step 3: Implement minimal engine and UI wiring**

Start observation after main-track scheduling, update pending outcomes once per accepted frame, route decisions in P1-then-P2 order, and contain `AudioPlaybackError`. Add one whiff row per side. Checking sound clears fade; checking fade clears sound. Save only the sound flag, so sound off always reloads as fade on.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the focused command from Step 2 and both full files.

### Task 4: Offline verification and scope gate

**Files:**
- Verify only the files listed above plus this plan.

- [ ] **Step 1: Run focused outcome/audio/engine/panel tests**

- [ ] **Step 2: Run the complete offline test suite**

Run: `python -m pytest -q`

- [ ] **Step 3: Verify source scope and hygiene**

Run `git diff --check`, `git status --short`, and confirm no `main.py`, media, config, personal path, network, MK1, or runtime SF6 modification appears.

- [ ] **Step 4: Report exact startup path and live-test boundary**

Report the launcher path. Do not claim real hit/block/whiff acceptance until an SF6-only live matrix is performed.
