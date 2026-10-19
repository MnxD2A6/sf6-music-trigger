# Meter-Level Super Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect SF6 Super activations from stable downward `0/1/2/3/CA` meter-level transitions so natural meter gain, combat effects, and round-end HUD loss cannot start music.

**Architecture:** Add a pure meter-level glyph extractor beside the existing fractional colour feature, then pass readings through a separate stability boundary before `GaugeTracker` emits. Colour profiles use direction as the exclusive activation signal; the existing side-isolated audio and hit/block/whiff pipeline remains downstream.

**Tech Stack:** Python 3.12, NumPy, OpenCV, dataclasses, pytest, Tkinter, Pygame, DXcam, PyInstaller.

**Spec:** `docs/superpowers/specs/2026-08-22-meter-level-super-detection-design.md`

## Global Constraints

- Stay on local `background-independent-meter`; do not push or create a PR.
- Preserve baseline `31774fd3b2070d9055749e83dee402fd6aec2566` through `backup/pre-meter-level-detector-20260822`.
- Do not modify `main.py`, runtime configuration, audio files, recordings, MK1, or other applications.
- Add no network, OCR, model-runtime, game-memory, or input dependencies.
- Store no captured frames, glyph crops, private paths, or game-pixel templates.
- Existing schema-2 colour profiles load without manual migration; derive level ROIs at runtime.
- Three matching readings confirm a decrease at 20 FPS; a baseline must dwell for ten trustworthy samples, and the tenth consecutive unknown clears it.
- Colour evidence alone cannot emit. `ALL_SUPERS` accepts spends 1/2/3; `SA3_CA_ONLY` plays only spend 3.
- Hit/block/whiff behavior and whiff-sound-versus-fade mutual exclusion remain unchanged.
- Resolve `$workspaceParent = Split-Path -Parent (Get-Location)` and `$sharedPython = Join-Path $workspaceParent 'SF6_Music_Trigger_Full_SharedVenv\Scripts\python.exe'`; test with `$sharedPython`, `-p no:cacheprovider`, and a fresh workspace-local `--basetemp`.

## File Map

- Create `meter_level_features.py`: derived ROI and geometric `0/1/2/3/CA` reader.
- Create `meter_level_tracker.py`: stability, direction, reset, immutable transition.
- Modify `gauge_features.py`, `gauge_tracker.py`, `domain.py`: production integration and event metadata.
- Modify `audio.py`, `engine.py`: deterministic tier gate and existing side routing.
- Modify `controller.py`, `panel.py`: level snapshot, derived ROI overlay, safe diagnostics.
- Create `tools/replay_meter_levels.py`: production-path offline replay without frame export.
- Add focused tests and replay/release evidence under existing test/docs structure.

---

### Task 1: Pure meter-level feature extraction

**Files:**
- Create: `src/sf6_music_trigger/meter_level_features.py`
- Create: `tests/test_meter_level_features.py`

**Interfaces:**
- Consumes: `Side`, `NormalizedRoi`, `MeterColorModel`, full BGR frame.
- Produces: `MeterLevelReading`, `derive_meter_level_roi(...)`, `read_meter_level(...)`.

- [ ] **Step 1: Write failing ROI and value tests**

```python
def test_level_roi_extends_one_quarter_meter_width_outward():
    p1 = derive_meter_level_roi(NormalizedRoi(.065, .885, .225, .975), Side.P1)
    p2 = derive_meter_level_roi(NormalizedRoi(.775, .885, .945, .975), Side.P2)
    assert p1 == NormalizedRoi(.025, .885, .065, .975)
    assert p2 == NormalizedRoi(.945, .885, .9875, .975)

@pytest.mark.parametrize("level", [0, 1, 2, 3, None])
def test_reading_accepts_only_fixed_alphabet(level):
    item = MeterLevelReading(Side.P1, level, .9 if level is not None else 0., level is not None)
    assert item.level is level
```

Also reject wrong side/type, bool levels, integers outside 0..3, nonfinite confidence, confidence outside 0..1, and invisible readings carrying a level.

- [ ] **Step 2: Run RED**

Run `& $python -m pytest tests/test_meter_level_features.py -q -p no:cacheprovider --basetemp ..\pytest-level-feature-red`.

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement strict contracts and derived ROI**

```python
@dataclass(frozen=True)
class MeterLevelReading:
    side: Side
    level: int | None
    confidence: float
    visible: bool

def derive_meter_level_roi(meter_roi: NormalizedRoi, side: Side) -> NormalizedRoi:
    margin = (meter_roi.x1 - meter_roi.x0) / 4.0
    if side is Side.P1:
        return NormalizedRoi(max(0., meter_roi.x0-margin), meter_roi.y0, meter_roi.x0, meter_roi.y1)
    return NormalizedRoi(meter_roi.x1, meter_roi.y0, min(1., meter_roi.x1+margin), meter_roi.y1)
```

- [ ] **Step 4: Write failing synthetic glyph tests**

Use generated masks rather than game pixels. Render a saturated side-colour outline and white core over changed bright/noisy backgrounds.

```python
@pytest.mark.parametrize((glyph, level), [("0",0), ("1",1), ("2",2), ("3",3), ("CA",3)])
@pytest.mark.parametrize("side", list(Side))
def test_fixed_hud_alphabet_over_unrelated_background(glyph, level, side):
    frame, roi, model = synthetic_level_frame(side, glyph, background_seed=91)
    item = read_meter_level(frame, side=side, meter_roi=roi, color_model=model)
    assert item.level == level
    assert item.confidence >= .75
```

Require unknown for clipping, unsupported white pixels, competing components, flash, low contrast, wrong-side outline, and invalid arrays.

- [ ] **Step 5: Run RED, then implement the minimal classifier**

Canonicalize the derived ROI. Build a side-hue outline mask from `MeterColorModel`; keep bright low-saturation white pixels adjacent to that outline; clean connected components. Classify two wide components as `CA/3`, one hole as `0`, narrow width as `1`, and lower-left versus lower-right occupancy as `2` versus `3`. Return unknown when confidence bounds overlap.

- [ ] **Step 6: Run GREEN and commit**

Run the complete Task 1 suite twice, then commit only its two files as `feat: read SF6 meter levels from HUD glyphs`.

---

### Task 2: Stable level direction and reset semantics

**Files:**
- Create: `src/sf6_music_trigger/meter_level_tracker.py`
- Create: `tests/test_meter_level_tracker.py`

**Interfaces:**
- Consumes: `MeterLevelReading`, increasing frame IDs.
- Produces: `MeterLevelTransition` and `StableMeterLevelTracker.update(...)`.

- [ ] **Step 1: Write failing direction tests**

```python
def test_three_lower_readings_emit_once():
    tracker = StableMeterLevelTracker(Side.P1, stable_samples=3, unknown_reset_samples=10)
    feed(tracker, [2,2,2], first_frame=1)
    events = feed(tracker, [1,1,1,1], first_frame=4)
    assert [(e.before, e.after, e.levels_spent) for e in events] == [(2,1,1)]

def test_gain_and_round_gap_never_emit():
    tracker = StableMeterLevelTracker(Side.P1, stable_samples=3, unknown_reset_samples=10)
    assert feed(tracker, [0,0,0,1,1,1]) == ()
    feed(tracker, [None] * 10)
    assert feed(tracker, [0,0,0]) == ()
```

Cover spends 1/2/3, duplicates, oscillation, nine unknown preserving baseline, tenth clearing it, reset, side mismatch, and backward/repeated frame IDs.

- [ ] **Step 2: Run RED**

Expected: module import failure.

- [ ] **Step 3: Implement immutable transition and tracker**

```python
@dataclass(frozen=True)
class MeterLevelTransition:
    side: Side
    before: int
    after: int
    pre_frame_id: int
    confirmed_frame_id: int

    @property
    def levels_spent(self) -> int:
        return self.before - self.after
```

The tracker stores candidate/count, stable baseline/frame, unknown count, and last frame. A newly stable lower value emits and becomes baseline; gain only replaces baseline; same values cannot duplicate.

- [ ] **Step 4: Run GREEN and commit**

Commit only Task 2 files as `feat: track stable SF6 meter-level spends`.

---

### Task 3: Observation, tracker, and event integration

**Files:**
- Modify: `src/sf6_music_trigger/gauge_features.py`
- Modify: `src/sf6_music_trigger/gauge_tracker.py`
- Modify: `src/sf6_music_trigger/domain.py`
- Modify: `src/sf6_music_trigger/engine.py`
- Modify: `tests/test_gauge_features.py`
- Modify: `tests/test_gauge_tracker.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: colour observations with level support/value/confidence; events with before/after/spent metadata.

- [ ] **Step 1: Write failing backward-compatible dataclass tests**

Append defaulted fields to existing contracts:

```python
GaugeObservation(..., meter_level_supported=True, meter_level=2, meter_level_confidence=.91)
ConfirmedSuperEvent(..., meter_before=3, meter_after=0, levels_spent=3)
```

Existing constructors remain valid. Level metadata is either all absent or internally consistent; bool/out-of-range/inconsistent values fail.

- [ ] **Step 2: Run RED, implement validation, run existing domain tests**

Expected RED: unexpected keyword arguments. Minimal GREEN adds fields and validation only.

- [ ] **Step 3: Write failing `observe_gauge` tests**

Monkeypatch `read_meter_level`. Prove colour observations always declare support; trustworthy reading carries value/confidence; unreadable stays supported with `None/0`; grayscale remains unsupported; a glyph alone cannot make a genuinely hidden HUD visible.

- [ ] **Step 4: Wire Task 1 into colour observation**

Call the reader once, preserve existing fill/structure calculations, and copy only derived scalar data into the immutable observation. Never retain/write the crop.

- [ ] **Step 5: Write failing colour-tracker tests**

```python
def test_gain_wrap_with_disruption_is_rejected_but_downward_level_emits():
    feed_levels(tracker, [1,1,1,2,2,2], disrupted=True)
    assert events == []
    feed_levels(tracker, [1,1,1])
    assert [(e.meter_before,e.meter_after,e.levels_spent) for e in events] == [(2,1,1)]
```

Also prove `3->0`, two-level spend, colour full-to-empty without decrease, unknown/reset, and P1/P2 isolation. Keep legacy grayscale full/spent tests.

- [ ] **Step 6: Integrate Task 2 exclusively for colour profiles**

When `meter_level_supported=True`, bypass confidence classification as an activation boundary. Map one transition to one `ConfirmedSuperEvent` and expose its current stable numeric level through a read-only `GaugeTracker.stable_meter_level` property for diagnostics. Unsupported legacy observations retain the old path. Reset both paths together.

- [ ] **Step 7: Write failing engine tests for bounded UNKNOWN handling**

Prove one to nine consecutive colour observations with an unreadable level do not immediately reset the stable pre-effect baseline, a restored lower level can still confirm after three matching readings, and the tenth unreadable sample clears the baseline without emitting. Preserve immediate reset for stale capture frames, geometry/profile changes, explicit session resets, and unsupported legacy observations that fail the existing trust boundary.

- [ ] **Step 8: Let supported colour observations reach the level tracker**

Replace the engine's immediate `neither HUD meter is trustworthy` return only when the observations declare meter-level support. Feed an unknown level into the bounded tracker so it owns the nine-versus-ten decision. Do not let an unknown reading become a level, emit an event, or bypass freshness/profile validation.

- [ ] **Step 9: Run domain/gauge/engine suites and commit**

Commit Task 3 files as `feat: confirm supers from downward meter levels`.

---

### Task 4: Deterministic Super-scope gate and unchanged outcome audio

**Files:**
- Modify: `src/sf6_music_trigger/audio.py`
- Modify: `src/sf6_music_trigger/engine.py`
- Modify: `tests/test_audio.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- Consumes: `ConfirmedSuperEvent.levels_spent`, `SideAudioConfig.super_scope`.
- Produces: an explicit accepted/filtered decision before outcome tracking or audio execution.

- [ ] **Step 1: Write failing scheduler scope tests**

Prove `ALL_SUPERS` returns the existing side-bound `STOP`/`PLAY` pair for one-, two-, and three-level spends. Prove `SA3_CA_ONLY` accepts exactly a three-level spend and returns no actions for one or two. Keep a compatibility case in which legacy events without level metadata follow the pre-change scheduler behavior.

```python
@pytest.mark.parametrize("levels_spent", [1, 2])
def test_sa3_only_scheduler_filters_confirmed_lower_tiers(levels_spent):
    event = confirmed_event(Side.P1, levels_spent=levels_spent)
    config = enabled_config(super_scope=SuperScope.SA3_CA_ONLY)
    assert AudioScheduler(Side.P1).on_confirmed_spend(event, config) == ()
```

- [ ] **Step 2: Run RED, then add the minimal scheduler guard**

After the existing side/config validation and before producing actions:

```python
if (
    config.super_scope is SuperScope.SA3_CA_ONLY
    and event.levels_spent is not None
    and event.levels_spent < 3
):
    return ()
```

Do not change `AudioPort`, `PygameAudioPort`, channel ownership, volume, or preview behavior.

- [ ] **Step 3: Write failing engine boundary tests**

For a confirmed one-level event under `SA3_CA_ONLY`, assert all of the following: the event remains in `EngineUpdate.events` and the event log; the scheduler is called; the audio port is not called; `OutcomeObserver.begin` is not called; no pending hit/block/whiff observation is created; the side status becomes the privacy-safe label `低等级已过滤`. For a three-level event and every `ALL_SUPERS` event, assert the existing immediate play and outcome observation sequence. Also cover P1/P2 isolation and a disabled config so an empty action tuple is not mislabeled as a tier rejection.

- [ ] **Step 4: Implement an explicit engine scope predicate**

Add a small typed helper that returns filtered only when the event carries `levels_spent in {1, 2}` and the selected scope is `SA3_CA_ONLY`. Call the scheduler for validation/diagnostic consistency, validate its tuple exactly as today, then skip `OutcomeObserver.begin` only for that explicit filtered case. Preserve the legacy missing-metadata path and all accepted-event ordering.

- [ ] **Step 5: Run audio/engine/outcome suites and commit**

Run `tests/test_audio.py`, `tests/test_engine.py`, and `tests/test_outcome.py`. Commit only Task 4 files as `feat: gate playback by confirmed super tier`.

---

### Task 5: Level diagnostics and calibration preview

**Files:**
- Modify: `src/sf6_music_trigger/controller.py`
- Modify: `src/sf6_music_trigger/panel.py`
- Modify: `tests/test_controller.py`
- Modify: `tests/test_panel_view_model.py`

**Interfaces:**
- Consumes: current stable level snapshots exposed by each production tracker.
- Produces: typed `(P1, P2)` level diagnostics and an extra derived-ROI overlay, never pixels or paths.

- [ ] **Step 1: Write failing controller snapshot tests**

Add an immutable `TrackerDiagnostics(states, levels)` contract and a controller property `tracker_levels: tuple[int | None, int | None] | None`. Snapshot levels under the same controller lock as tracker states, clear them on `start`, and validate that each item is `None` or an exact non-bool integer in `0..3`. Test normal recording, unknown values, invalid provider output, restart clearing, and concurrent reads returning complete tuples only.

- [ ] **Step 2: Implement the minimal typed snapshot**

Expose the stable numeric level through the existing diagnostics provider boundary rather than letting the panel inspect tracker internals. Update the production provider in `panel.py` to return `TrackerDiagnostics((p1.state, p2.state), (p1.stable_meter_level, p2.stable_meter_level))`. Keep old diagnostics-provider implementations source-compatible by accepting the existing two-state tuple and normalizing it to levels unknown.

- [ ] **Step 3: Write failing view-model diagnostic tests**

Extend `PanelViewModel.update_runtime_diagnostics(...)` with the optional level tuple. Assert labels such as `P1=LEVEL_2`, `P2=LEVEL_UNKNOWN`, and `气槽等级不可读` are emitted without audio paths, profile paths, filenames, pixel values, or captured images. Verify stopped/reset state removes stale values.

- [ ] **Step 4: Write failing calibration-overlay tests**

In the existing ROI confirmation path, assert each visible colour meter ROI is accompanied by its `derive_meter_level_roi(...)` band, with a distinct safe label such as `P1 LEVEL`. Grayscale-only profiles keep their current overlay. There must be no extra prompt, no new sample step, and cancel/save semantics remain unchanged.

- [ ] **Step 5: Wire diagnostics and overlay, run GREEN, and commit**

Run controller, panel view-model, and calibration tests. Commit only Task 5 files as `feat: show meter-level detection diagnostics`.

---

### Task 6: Production-path offline replay regression

**Files:**
- Create: `tools/replay_meter_levels.py`
- Modify: `tests/test_replay_integration.py`
- Create: `docs/verification/2026-08-22-meter-level-replay.md`

**Interfaces:**
- Consumes: a local SF6 recording and an existing profile through production observation/tracker code.
- Produces: timestamped scalar events only: `(time, side, before, after, levels_spent)`.

- [x] **Step 1: Write failing synthetic replay tests**

Build generated observation streams for: `0->1` and `1->2` gains with and without disruption; `1->0`, `2->0`, `3/CA->0` spends; short and long unknown gaps; `WIN`/round reset; simultaneous sides; duplicate frames; stale/backward frame IDs; and side isolation. Assert gains and resets produce zero events and stable decreases produce exactly one event.

- [x] **Step 2: Implement a bounded replay CLI**

Accept `--video`, `--profile-root`, `--profile`, and optional `--csv`. Sample at the configured production FPS and call the production observer/tracker boundary; do not reproduce detector logic in the tool. Print/write only scalar diagnostics. Do not export frames, crops, templates, audio paths, or screenshots. Return nonzero on unreadable video/profile or violated expected counts.

- [x] **Step 3: Replay the 468.6-second real-match recording**

Resolve `$recordingRoot = Join-Path ([Environment]::GetFolderPath('MyVideos')) '屏幕录制\SF6'` and run against `Join-Path $recordingRoot '实际测试，误触，没触发，出发错误音频.mp4'`. Require exactly four replay confirmations at `137.733 P1`, `234.400 P1`, `387.800 P1`, and `399.533 P2`, with a tolerance no wider than the three-sample confirmation window. The last confirmation follows the `399.067` action landmark because the numeric HUD is occluded until it restores. Require no events near `56.467`, `154.800`, `188.000`, `297.067`, `322.867`, or `448.000`.

- [x] **Step 4: Replay prior positive recordings and fixtures**

Run the same production path over `杰米测试3.mp4`, `终于基本成功了，谢谢你S02.mp4`, and existing training/real-match replay fixtures. Record detected side, level delta, count, tool command, profile identity, and SHA-256 of each input without copying recordings into the repository.

- [x] **Step 5: Record evidence and commit**

Write the commands, hashes, event table, and pass/fail result to the verification document. State explicitly that replay is offline evidence and is not SF6 real acceptance. Commit the tool, integration tests, and evidence as `test: replay meter-level supers against real matches`.

---

### Task 7: Full verification, backup, and portable `_r6` package

**Files:**
- Modify/Create: `docs/verification/2026-08-22-meter-level-release.md`
- Generated outside Git: `SF6_Music_Trigger_GroupTest_0.1.0a1_20260822_r6/`
- Generated outside Git: `SF6_Music_Trigger_GroupTest_0.1.0a1_20260822_r6.zip`

- [x] **Step 1: Run focused suites from a fresh temp root**

Run feature, tracker, domain, audio, engine, controller, panel, calibration, outcome, capture, and replay tests with `-p no:cacheprovider` and a new workspace-local `--basetemp`. A repeated focused run must produce the same result.

- [x] **Step 2: Run the complete offline suite**

```powershell
$workspaceParent = Split-Path -Parent (Get-Location)
$sharedPython = Join-Path $workspaceParent 'SF6_Music_Trigger_Full_SharedVenv\Scripts\python.exe'
& $sharedPython -m pytest -q -p no:cacheprovider --basetemp '..\pytest-meter-level-full'
```

Record the exact pass count, duration, interpreter version, and commit. Do not claim completion from focused tests alone.

- [x] **Step 3: Audit scope and create the rollback point**

Use `git status --short`, `git diff --check`, and `git diff backup/pre-meter-level-detector-20260822...HEAD --name-only`. Confirm no `main.py`, runtime config, audio, video, profile, private-path artifact, or unrelated file changed. Create local annotated tag `backup/meter-level-detector-20260822` only after the audit and full suite pass.

- [x] **Step 4: Build the portable `_r6` directory and ZIP**

Use the repository's `tools/portable_release.py`. If the build interpreter needs the already-installed shared PyInstaller, resolve `$desktop = [Environment]::GetFolderPath('Desktop')`, run `Join-Path $desktop 'SF6_Music_Trigger_Full\.venv\Scripts\python.exe'`, and set `PYTHONPATH` to `Join-Path $workspaceParent 'SF6_Music_Trigger_Full_SharedVenv\Lib\site-packages'`. Do not download dependencies or include user profiles/audio.

- [x] **Step 5: Verify the portable artifact**

Run the EXE's noninteractive self-check; verify bundled Tcl/Tk, DXcam/OpenCV/Pygame imports, version/commit metadata, sanitized default config, absence of recordings/private paths, directory/ZIP SHA-256, and extraction into a fresh path. Re-run `git status --short` afterward and require a clean tracked worktree.

- [x] **Step 6: Write release evidence and handoff**

Record all commands, pass counts, hashes, artifact sizes, rollback tags, and known limitations. State `SF6 实机验收尚未执行` until a real SF6-only test covers natural gains, SA1, SA2, SA3/CA, hit, block, whiff, round end, and both sides. Provide the exact absolute ZIP and EXE paths; do not push, create a PR, start SF6, or use Computer Use during offline packaging.
