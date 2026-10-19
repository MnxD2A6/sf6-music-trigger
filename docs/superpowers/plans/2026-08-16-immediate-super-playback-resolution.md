# Immediate Super Playback Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep immediate audio playback for every plausible SF6 super activation, then resolve SA1/SA2 filtering, hit/block continuation, whiff fade/sound, and unknown-safe fade from bounded post-event screen evidence.

**Architecture:** Extend the existing screen-only pipeline without changing the stable gauge-trigger boundary. `GaugeObservation` carries a calibrated normalized meter level; `OutcomeObserver` owns a bounded per-side feature ring and pending resolution state; `VisualTriggerEngine` remembers pre-spend features, starts audio immediately, and routes exactly one later corrective action. Configuration and panel wiring expose an independent per-side scope selector.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Tkinter, pytest, existing `GaugeTracker`, `AudioScheduler`, and local replay tooling.

## Global Constraints

- Work from commit `15c8a91` on linked worktree branch `background-independent-meter`.
- Preserve backup tag `backup/stable-ca-audio-20260816` and snapshot `<workspace>\SF6_Music_Trigger_Snapshots\2026-08-16-ffc6a51-stable-ca-audio`.
- Do not modify `main.py`.
- Do not weaken the current desktop/task-view false-trigger or CA-animation protections.
- Do not store full screenshots in the pending result state or write captured pixels to disk.
- Keep P1/P2 state and audio channels independent; deterministic routing order is P1 then P2.
- `SA3_CA_ONLY` is the missing-field default; unknown scope values fail closed.
- A suspected super always starts the configured main track immediately in both scopes.
- `FILTERED_TIER`, `UNKNOWN`, and `RESET` fade only; they never play the whiff sound.
- A confirmed `WHIFF` plays STOP+PLAY whiff when enabled, otherwise FADE.
- No network, no MK1, no SF6 launch, and no Computer Use during implementation or offline regression.
- Final real validation is SF6-only and remains a separate user-visible phase after offline approval.

---

### Task 1: Persist the per-side super scope and expose it in the panel

**Files:**
- Modify: `src/sf6_music_trigger/config.py`
- Modify: `src/sf6_music_trigger/panel.py`
- Modify: `tests/test_domain_config.py`
- Modify: `tests/test_panel_view_model.py`

**Interfaces:**
- Produces: `SuperScope(str, Enum)` with `ALL_SUPERS` and `SA3_CA_ONLY`.
- Produces: `SideAudioConfig.super_scope: SuperScope = SuperScope.SA3_CA_ONLY`.
- Produces: `PanelViewModel.p1_super_scope` and `p2_super_scope`, both `SuperScope`.
- Consumed later by: `VisualTriggerEngine` and `OutcomeObserver.begin(...)`.

- [ ] **Step 1: Add failing configuration tests**

Add tests that independently prove the default, strict enum, unknown-value rejection, and JSON round trip:

```python
def test_super_scope_defaults_to_sa3_ca_only_and_round_trips(tmp_path: Path) -> None:
    from sf6_music_trigger.config import SuperScope

    assert SideAudioConfig().super_scope is SuperScope.SA3_CA_ONLY
    config = AppConfig(
        p1=SideAudioConfig(super_scope=SuperScope.ALL_SUPERS),
        p2=SideAudioConfig(super_scope=SuperScope.SA3_CA_ONLY),
    )
    target = tmp_path / "config.json"
    save_config(target, config)
    assert load_config(target) == config


def test_unknown_super_scope_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    target.write_text(
        '{"p1":{"super_scope":"EVERYTHING"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="super_scope"):
        load_config(target)
```

- [ ] **Step 2: Run the configuration RED tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_domain_config.py -k "super_scope" -v
```

Expected: tests collect and FAIL because `SuperScope` and `super_scope` do not exist.

- [ ] **Step 3: Implement the strict configuration model**

Use a string enum and explicit parser so direct construction and JSON loading share the same strict type:

```python
class SuperScope(str, Enum):
    ALL_SUPERS = "ALL_SUPERS"
    SA3_CA_ONLY = "SA3_CA_ONLY"


@dataclass(frozen=True)
class SideAudioConfig:
    # existing fields stay in their current order
    super_scope: SuperScope = SuperScope.SA3_CA_ONLY

    def __post_init__(self) -> None:
        # existing validation first
        if not isinstance(self.super_scope, SuperScope):
            raise TypeError("super_scope must be SuperScope")


def _parse_side(value: object, side_name: str) -> SideAudioConfig:
    if not isinstance(value, dict):
        raise TypeError(f"{side_name} must be an object")
    _reject_unknown_keys(value, _SIDE_KEYS, side_name)
    values = dict(value)
    if "super_scope" in values:
        try:
            values["super_scope"] = SuperScope(values["super_scope"])
        except (TypeError, ValueError):
            raise ValueError(f"{side_name}.super_scope is invalid") from None
    return SideAudioConfig(**values)
```

Serialize the enum as `.value`, not the Python enum representation.

- [ ] **Step 4: Run configuration tests GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_domain_config.py -v
```

Expected: all configuration tests PASS.

- [ ] **Step 5: Add failing panel view-model tests**

```python
def test_panel_round_trips_independent_super_scopes(tmp_path: Path) -> None:
    from sf6_music_trigger.config import SuperScope
    from sf6_music_trigger.panel import PanelViewModel

    vm = PanelViewModel(user_dir=tmp_path)
    vm.p1_super_scope = SuperScope.ALL_SUPERS
    vm.p2_super_scope = SuperScope.SA3_CA_ONLY
    config = vm.to_config()
    assert config.p1.super_scope is SuperScope.ALL_SUPERS
    assert config.p2.super_scope is SuperScope.SA3_CA_ONLY
```

Also extend the application-load test to prove saved scopes populate the two view-model fields.

- [ ] **Step 6: Run the panel RED tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_panel_view_model.py -k "super_scope" -v
```

Expected: FAIL because the view-model fields and selector wiring are absent.

- [ ] **Step 7: Implement panel state and compact selectors**

Add one `ttk.Combobox(state="readonly")` per side using an explicit label mapping:

```python
_SUPER_SCOPE_LABELS = {
    SuperScope.ALL_SUPERS: "全部大招（SA1/SA2/SA3/CA）",
    SuperScope.SA3_CA_ONLY: "仅 SA3/CA",
}
_SUPER_SCOPE_FROM_LABEL = {label: scope for scope, label in _SUPER_SCOPE_LABELS.items()}
```

Initialize from config, sync back before save/start, and leave whiff sound/fade controls unchanged.

- [ ] **Step 8: Run panel and configuration tests GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_domain_config.py tests/test_panel_view_model.py -v
```

Expected: both files PASS.

- [ ] **Step 9: Commit Task 1**

```powershell
git add src/sf6_music_trigger/config.py src/sf6_music_trigger/panel.py tests/test_domain_config.py tests/test_panel_view_model.py
git commit -m "feat: configure super playback scope"
```

---

### Task 2: Carry calibrated remaining-meter evidence and stabilize bar buckets

**Files:**
- Modify: `src/sf6_music_trigger/gauge_features.py`
- Modify: `src/sf6_music_trigger/meter_color_features.py`
- Modify: `tests/test_gauge_features.py`
- Modify: `tests/test_meter_color_features.py`
- Modify: `tests/test_outcome.py`

**Interfaces:**
- Produces: `GaugeObservation.normalized_meter: float | None` where spent endpoint is `0.0` and full endpoint is `1.0`.
- Produces: `remaining_meter_bucket(normalized_meter: Real) -> int | None` returning `0`, `1`, `2`, `3`, or `None` at ambiguous boundaries/out-of-band input.
- Consumed later by: `OutcomeObserver.update(...)`.

- [ ] **Step 1: Add failing normalized-meter tests**

```python
@pytest.mark.parametrize(
    ("fill", "spent", "full", "want"),
    [(0.20, 0.20, 0.80, 0.0), (0.40, 0.20, 0.80, 1 / 3),
     (0.60, 0.20, 0.80, 2 / 3), (0.80, 0.20, 0.80, 1.0)],
)
def test_color_observation_normalizes_fill_between_calibrated_endpoints(
    fill: float, spent: float, full: float, want: float
) -> None:
    observation = _observe_color_fill(fill=fill, spent=spent, full=full)
    assert observation.normalized_meter == pytest.approx(want)
```

Add a grayscale-reference test proving `normalized_meter is None`, and hidden/unverified construction tests proving invalid combinations fail closed.

- [ ] **Step 2: Run normalized-meter tests RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gauge_features.py tests/test_meter_color_features.py -k "normalized_meter or calibrated_endpoints" -v
```

Expected: FAIL because `GaugeObservation` does not expose normalized meter evidence.

- [ ] **Step 3: Implement normalized meter evidence**

In the colour-reference path only, compute:

```python
normalized_meter = (fill - model.spent_fill) / (model.full_fill - model.spent_fill)
normalized_meter = min(1.0, max(0.0, float(normalized_meter)))
```

Use `None` for hidden HUD and legacy grayscale references. Validate that hidden observations and unverified observations cannot invent normalized tier evidence.

- [ ] **Step 4: Run gauge feature tests GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gauge_features.py tests/test_meter_color_features.py -v
```

Expected: both files PASS.

- [ ] **Step 5: Add failing bucket and five-sample stability tests**

Define literal bucket centers at `0`, `1/3`, `2/3`, and `1`. Require a bounded tolerance of `0.12` around a center; exact midpoint/unstable values remain unresolved.

```python
@pytest.mark.parametrize(
    ("value", "want"),
    [(0.02, 0), (0.34, 1), (0.66, 2), (0.98, 3), (0.50, None)],
)
def test_remaining_meter_bucket_is_conservative(value: float, want: int | None) -> None:
    assert remaining_meter_bucket(value) == want


def test_tier_requires_five_consecutive_trustworthy_matching_samples() -> None:
    observer = _observer_with_pre_spend_baseline(scope=SuperScope.SA3_CA_ONLY)
    for frame_id in range(10, 14):
        assert observer.update(_frame(frame_id), observations=_meters(p1=1)) == ()
    assert observer.update(_frame(14), observations=_meters(p1=1)) == (
        _decision(OutcomeLabel.FILTERED_TIER, Side.P1),
    )
```

- [ ] **Step 6: Run bucket/stability tests RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_outcome.py -k "remaining_meter or tier_requires" -v
```

Expected: FAIL because the bucket classifier and tier streak do not exist.

- [ ] **Step 7: Implement the pure bucket helper and pending streak state**

```python
def remaining_meter_bucket(value: Real) -> int | None:
    normalized = _require_probability("normalized_meter", value)
    centers = (0.0, 1 / 3, 2 / 3, 1.0)
    nearest = min(range(4), key=lambda index: abs(normalized - centers[index]))
    return nearest if abs(normalized - centers[nearest]) <= 0.12 else None
```

Store `tier_bucket` and `tier_streak` per pending side. Reset only that streak when evidence is hidden, unverified, ambiguous, or changes bucket; do not reset the pending event.

- [ ] **Step 8: Run outcome meter tests GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_outcome.py -k "remaining_meter or tier_requires" -v
```

Expected: selected tests PASS.

- [ ] **Step 9: Commit Task 2**

```powershell
git add src/sf6_music_trigger/gauge_features.py src/sf6_music_trigger/meter_color_features.py src/sf6_music_trigger/outcome.py tests/test_gauge_features.py tests/test_meter_color_features.py tests/test_outcome.py
git commit -m "feat: quantify restored super meter"
```

---

### Task 3: Replace confirmation-frame outcomes with pre-spend deferred resolution

**Files:**
- Modify: `src/sf6_music_trigger/outcome.py`
- Modify: `tests/test_outcome.py`

**Interfaces:**
- Produces: `OutcomeLabel.FILTERED_TIER` in addition to existing labels.
- Produces: `OutcomeObserver.remember(frame: CapturedFrame) -> None`.
- Produces: `OutcomeObserver.begin(event: ConfirmedSuperEvent, scope: SuperScope) -> None`.
- Produces: `OutcomeObserver.update(frame, *, observations: tuple[GaugeObservation, GaugeObservation]) -> tuple[OutcomeDecision, ...]`.
- Produces: `OutcomeObserver.reset(reason) -> tuple[OutcomeDecision, ...]`, returning one RESET decision per pending side so the engine can fade exactly once.

- [ ] **Step 1: Add failing pre-spend baseline tests**

```python
def test_begin_uses_features_keyed_by_pre_spend_frame_id() -> None:
    observer = OutcomeObserver(
        contact_delta=0.10,
        contact_samples=2,
        feature_extractor=_features_encoded_in_image,
    )
    observer.remember(_feature_frame(4, time=0.20, p1=(0.80, 0.20)))
    observer.remember(_feature_frame(9, time=0.45, p1=(0.60, 0.20)))
    observer.begin(_event(pre_spend_frame_id=4, confirmed_frame_id=9), SuperScope.ALL_SUPERS)
    assert observer.update(
        _feature_frame(10, time=1.45, p1=(0.60, 0.20)),
        observations=_meters(p1=0),
    ) == ()
    decisions = observer.update(
        _feature_frame(11, time=1.50, p1=(0.60, 0.20)),
        observations=_meters(p1=0),
    )
    assert tuple(decision.label for decision in decisions) == (OutcomeLabel.HIT,)
```

`_features_encoded_in_image`, `_feature_frame`, and `_meters` are test-file
helpers only. The production API must not expose a test-only baseline getter;
the assertion above differs observably depending on whether frame 4 or frame 9
was used.

Also prove missing/evicted baseline reaches UNKNOWN at the hard deadline and never WHIFF.

- [ ] **Step 2: Run baseline tests RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_outcome.py -k "pre_spend or missing_baseline" -v
```

Expected: FAIL because the observer still requires the confirmation frame as baseline.

- [ ] **Step 3: Implement the bounded derived-feature ring**

Use an insertion-ordered `deque`/dict bounded to 240 frame IDs (12 seconds at 20 FPS). Each entry stores only:

```python
@dataclass(frozen=True)
class OutcomeFeatures:
    frame_id: int
    sampled_at: Real
    p1: tuple[float, float]
    p2: tuple[float, float]
```

`remember` extracts both sides from a valid accepted frame. `begin` looks up `event.pre_spend_frame_id`; absence sets `baseline=None` and remains fail-closed.

- [ ] **Step 4: Add failing timing and contact tests**

Cover these independent observable branches:

Create these exact nodes with literal frame sequences. The `_run_resolution_case`
test helper constructs the baseline at frame 4/time 0.2, activates at time 1.0,
feeds the listed `(time, hud_visible, meter_bucket, health, result)` rows, and
returns the real observer decisions:

```python
RESOLUTION_CASES = {
    "test_contact_cannot_finalize_during_first_four_tenths": {
        "rows": [(1.20, True, 0, 0.60, 0.20), (1.35, True, 0, 0.60, 0.20)],
        "want": (),
    },
    "test_two_stable_health_changes_after_suppression_resolve_hit": {
        "rows": [(1.45, True, 0, 0.60, 0.20), (1.50, True, 0, 0.60, 0.20)],
        "want": (OutcomeLabel.HIT,),
    },
    "test_two_stable_training_ui_changes_resolve_block": {
        "rows": [(1.45, True, 0, 0.80, 0.40), (1.50, True, 0, 0.80, 0.40)],
        "want": (OutcomeLabel.BLOCK,),
    },
    "test_hidden_hud_at_three_seconds_preserves_pending_until_restore": {
        "rows": [(4.00, False, None, 0.80, 0.20)],
        "want": (),
        "pending": (Side.P1,),
    },
    "test_restored_hud_without_contact_at_checkpoint_resolves_whiff": {
        "rows": [(4.00, True, 0, 0.80, 0.20)],
        "want": (OutcomeLabel.WHIFF,),
    },
    "test_hidden_hud_at_eight_seconds_resolves_unknown": {
        "rows": [(9.00, False, None, 0.80, 0.20)],
        "want": (OutcomeLabel.UNKNOWN,),
    },
    "test_three_remaining_bars_after_event_resolves_unknown_not_whiff": {
        "rows": [(8.80, True, 3, 0.80, 0.20), (9.00, True, 3, 0.80, 0.20)],
        "want": (OutcomeLabel.UNKNOWN,),
    },
}
```

Each expectation uses literal times: activation `1.0`, suppression boundary `1.4`, checkpoint `4.0`, hard deadline `9.0`.

- [ ] **Step 5: Run timing/contact tests RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_outcome.py -k "four_tenths or stable_health or stable_training or three_seconds or eight_seconds or three_remaining" -v
```

Expected: FAIL against the old single six-second deadline implementation.

- [ ] **Step 6: Implement pending resolution state and decision ordering**

Use these exact defaults:

```python
suppression_seconds = 0.4
checkpoint_seconds = 3.0
hard_deadline_seconds = 8.0
contact_samples = 2
tier_samples = 5
```

Per update, apply: capture/HUD trust → tier streak/policy → contact streak → trustworthy no-contact checkpoint → hard deadline. In `ALL_SUPERS`, stable tier is diagnostic and does not filter. In `SA3_CA_ONLY`, buckets 1/2 return `FILTERED_TIER`, bucket 0 permits later contact/whiff resolution, and bucket 3 stays unresolved until UNKNOWN.

- [ ] **Step 7: Add failing reset, duplicate, and side-isolation tests**

Use this literal oracle table; each named node calls the real observer and
asserts all listed values rather than private fields:

```python
STATE_ORACLES = {
    "test_reset_returns_exactly_one_reset_decision_per_pending_side": {
        "labels": (OutcomeLabel.RESET, OutcomeLabel.RESET),
        "sides": (Side.P1, Side.P2),
        "pending_after": (),
    },
    "test_duplicate_event_id_does_not_replace_pending_or_restart_resolution": {
        "pending_after_second_begin": (Side.P1,),
        "decision_event_ids": ("event-p1-1",),
    },
    "test_p1_and_p2_pending_resolve_independently_in_side_order": {
        "labels": (OutcomeLabel.HIT, OutcomeLabel.FILTERED_TIER),
        "sides": (Side.P1, Side.P2),
    },
    "test_backward_or_duplicate_frames_cannot_advance_streaks": {
        "accepted_contact_samples": 1,
        "decisions": (),
    },
}
```

- [ ] **Step 8: Run reset/isolation tests RED, implement, then run all outcome tests GREEN**

Run RED selection, implement exact frame ordering guards and per-side cleanup, then run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_outcome.py -v
```

Expected: all outcome tests PASS.

- [ ] **Step 9: Commit Task 3**

```powershell
git add src/sf6_music_trigger/outcome.py tests/test_outcome.py
git commit -m "feat: resolve supers from pre-spend evidence"
```

---

### Task 4: Map every terminal result to one atomic audio correction

**Files:**
- Modify: `src/sf6_music_trigger/audio.py`
- Modify: `tests/test_audio.py`

**Interfaces:**
- Consumes: `OutcomeLabel.HIT`, `BLOCK`, `WHIFF`, `FILTERED_TIER`, `UNKNOWN`, `RESET`.
- Produces: `AudioScheduler.on_outcome(...)` action tuples matching the design table.

- [ ] **Step 1: Add failing scheduler mapping tests**

```python
@pytest.mark.parametrize("label", [OutcomeLabel.FILTERED_TIER, OutcomeLabel.UNKNOWN, OutcomeLabel.RESET])
def test_safe_terminal_results_fade_and_never_play_whiff(label: OutcomeLabel) -> None:
    scheduler = AudioScheduler(Side.P1)
    config = SideAudioConfig(
        enabled=True,
        play_whiff_sound=True,
        whiff_track="audio/whiff.wav",
        whiff_fade_ms=1700,
    )
    assert scheduler.on_outcome(_decision(label), config) == (
        AudioAction(AudioKind.FADE, Side.P1, fade_ms=1700),
    )


@pytest.mark.parametrize("label", [OutcomeLabel.HIT, OutcomeLabel.BLOCK])
def test_contact_results_leave_immediate_main_playback_untouched(label: OutcomeLabel) -> None:
    assert AudioScheduler(Side.P1).on_outcome(_decision(label), SideAudioConfig()) == ()
```

Retain the existing two whiff tests and make them assert exactly one FADE or exactly one STOP+PLAY tuple.

- [ ] **Step 2: Run scheduler mapping tests RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -k "terminal_results or contact_results or whiff" -v
```

Expected: `FILTERED_TIER` is absent or safe terminal results incorrectly return no action.

- [ ] **Step 3: Implement explicit result mapping**

```python
if not config.enabled or decision.label in {OutcomeLabel.HIT, OutcomeLabel.BLOCK}:
    return ()
if decision.label in {OutcomeLabel.FILTERED_TIER, OutcomeLabel.UNKNOWN, OutcomeLabel.RESET}:
    return (AudioAction(AudioKind.FADE, decision.side, fade_ms=config.whiff_fade_ms),)
if decision.label is OutcomeLabel.WHIFF:
    # retain current mutually exclusive fade or STOP+PLAY behavior
```

- [ ] **Step 4: Run audio tests GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v
```

Expected: all audio tests PASS.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/sf6_music_trigger/audio.py tests/test_audio.py
git commit -m "feat: route resolved super audio actions"
```

---

### Task 5: Integrate immediate playback, deferred correction, and reset fade in the engine

**Files:**
- Modify: `src/sf6_music_trigger/engine.py`
- Modify: `tests/test_engine.py`
- Modify: `src/sf6_music_trigger/controller.py`
- Modify: `tests/test_controller.py`

**Interfaces:**
- Consumes: `OutcomeObserver.remember`, `begin`, `update`, and reset decisions.
- Preserves: existing tracker event and immediate `AudioScheduler.on_confirmed_spend` behavior.
- Produces: exactly one later corrective batch per event, or no corrective batch for contact.

- [ ] **Step 1: Add failing immediate-play and pre-spend wiring tests**

Use the existing `_engine` and recording-audio fixtures with these exact
observable oracles:

```python
IMMEDIATE_ENGINE_ORACLES = {
    "test_both_scopes_play_immediately_at_the_existing_confirmation_boundary": {
        "scopes": (SuperScope.ALL_SUPERS, SuperScope.SA3_CA_ONLY),
        "batch_kinds": ((AudioKind.STOP, AudioKind.PLAY),) * 2,
    },
    "test_engine_remembers_frame_before_tracker_can_emit_event": {
        "tracker_pre_spend_frame_id": 5,
        "observer_baseline_source_frame_id": 5,
    },
    "test_engine_begins_resolution_with_event_scope_and_pre_spend_lookup": {
        "event_side": Side.P2,
        "scope": SuperScope.SA3_CA_ONLY,
        "pre_spend_frame_id": 7,
    },
    "test_duplicate_event_id_cannot_execute_a_second_immediate_batch": {
        "event_log_ids": ("stable-event",),
        "audio_batch_count": 1,
    },
}
```

Use fake trackers/schedulers only at the external collaborator boundary; assert real `VisualTriggerEngine` event log and audio batches.

- [ ] **Step 2: Run engine integration tests RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_engine.py -k "both_scopes or remembers_frame or event_scope or duplicate_event" -v
```

Expected: FAIL because the engine passes the confirmation frame to `begin` and has no scope/feature-ring wiring.

- [ ] **Step 3: Implement the new engine order**

For each accepted frame:

```python
self._outcome_observer.remember(frame)
outcome_decisions = self._outcome_observer.update(
    frame,
    observations=(p1_observation, p2_observation),
)
self._route_outcomes(outcome_decisions)
# tracker updates and immediate confirmed-spend audio remain after this
self._outcome_observer.begin(event, config.super_scope)
```

Do not reset the pending observer merely because expected cinematic meter evidence is hidden/unverified. Runtime capture/profile/freshness resets must route returned RESET decisions before clearing trackers.

- [ ] **Step 4: Add failing result-routing and reset tests**

Build one real-engine test per row below. The expected action kinds are
hand-derived literals and must be asserted on the recording audio port:

```python
ROUTING_ORACLES = {
    "test_filtered_sa1_sa2_fades_once_without_whiff_sound": {
        "decision": OutcomeLabel.FILTERED_TIER,
        "batches": ((AudioKind.FADE,),),
    },
    "test_zero_meter_contact_keeps_immediate_playback_running": {
        "decision": OutcomeLabel.HIT,
        "corrective_batches": (),
    },
    "test_whiff_sound_mode_routes_one_stop_play_batch": {
        "decision": OutcomeLabel.WHIFF,
        "batches": ((AudioKind.STOP, AudioKind.PLAY),),
    },
    "test_whiff_fade_mode_routes_one_fade_batch": {
        "decision": OutcomeLabel.WHIFF,
        "batches": ((AudioKind.FADE,),),
    },
    "test_capture_reset_fades_each_pending_side_once": {
        "decision": OutcomeLabel.RESET,
        "batches": ((AudioKind.FADE,),),
    },
    "test_simultaneous_p1_p2_corrections_are_p1_then_p2": {
        "sides": (Side.P1, Side.P2),
        "batch_kinds": ((AudioKind.FADE,), (AudioKind.FADE,)),
    },
    "test_outcome_audio_failure_is_diagnostic_without_state_rollback": {
        "diagnostic_count": 1,
        "pending_after": (),
        "repeat_batch_count": 0,
    },
}
```

- [ ] **Step 5: Run routing/reset tests RED, implement, and run engine GREEN**

Run the selection first to observe RED, then implement routing and run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_engine.py -v
```

Expected: all engine tests PASS.

- [ ] **Step 6: Update controller diagnostics without paths or pixels**

Expose only privacy-safe labels: `等待大招结果`, `后续动画`, `命中`, `格挡`, `打空`, `低等级已淡出`, `结果不确定`. Add controller tests proving audio paths and captured images never enter status strings.

- [ ] **Step 7: Run controller tests GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_controller.py -v
```

Expected: all controller tests PASS.

- [ ] **Step 8: Commit Task 5**

```powershell
git add src/sf6_music_trigger/engine.py src/sf6_music_trigger/controller.py tests/test_engine.py tests/test_controller.py
git commit -m "feat: correct immediate super playback outcomes"
```

---

### Task 6: Prove the feature against recordings and the complete offline suite

**Files:**
- Modify: `tests/test_replay_integration.py`
- Modify only if required by existing CLI contract: `tools/replay.py`
- Create: `docs/superpowers/reports/2026-08-16-immediate-super-playback-resolution.md`

**Interfaces:**
- Consumes the complete application pipeline with a silent recording audio port.
- Produces reproducible event timestamps, outcome labels, and action tuples without playing real audio.

- [ ] **Step 1: Add deterministic replay assertions**

Use these absolute local fixtures when present, and skip with an explicit local-fixture reason when absent:

```text
<private-video-root>\不说了.mp4
<private-video-root>\CA0气测试.mp4
<private-video-root>\终于基本成功了，谢谢你S02.mp4
<private-video-root>\杰米测试3.mp4
```

The replay test must assert:

- desktop/task-view recording: zero confirmed events and zero audio batches;
- stable CA recordings: same confirmed event counts as the `ffc6a51` baseline;
- `杰米测试3.mp4`: immediate batches remain at the three established event times; stable 1/2-bar restoration produces fade under `SA3_CA_ONLY`; zero-bar contact never produces the low-tier fade;
- silent fake audio only; no SDL/device access.

- [ ] **Step 2: Run replay tests and fix only reproducible pipeline defects**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_replay_integration.py -v
```

Expected: all available fixture cases PASS; absent private fixtures are explicitly reported as skipped, not silently ignored.

- [ ] **Step 3: Run focused feature regression**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_domain_config.py tests/test_panel_view_model.py tests/test_gauge_features.py tests/test_meter_color_features.py tests/test_outcome.py tests/test_audio.py tests/test_engine.py tests/test_controller.py tests/test_replay_integration.py -v
```

Expected: zero failures.

- [ ] **Step 4: Run full offline regression with isolated temp directories**

Run:

```powershell
$env:TEMP = '<workspace>\SF6_Result_Temp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pytest -v --basetemp '<workspace>\SF6_Result_Pytest'
```

Expected: zero product regressions. If the known Windows directory-link host limitation recurs, report its exact node IDs separately and rerun the remaining suite without hiding it.

- [ ] **Step 5: Verify source hygiene and scope**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile src/sf6_music_trigger/config.py src/sf6_music_trigger/panel.py src/sf6_music_trigger/gauge_features.py src/sf6_music_trigger/meter_color_features.py src/sf6_music_trigger/outcome.py src/sf6_music_trigger/audio.py src/sf6_music_trigger/engine.py src/sf6_music_trigger/controller.py
git diff --check
git status --short
git diff --name-only 15c8a91
git diff --name-only 15c8a91 -- main.py
```

Expected: compile and diff check exit 0; only planned Task 6 feature/test/report files differ; the final `main.py` command prints nothing.

- [ ] **Step 6: Write the reproducible report**

Record exact commands, exit codes, pass/fail/skip counts, replay event times, outcome labels, audio action tuples, current branch/commit, backup tag, and the statement: `SF6 real acceptance not yet performed`.

- [ ] **Step 7: Commit the verified implementation**

```powershell
git add src/sf6_music_trigger tests/test_domain_config.py tests/test_panel_view_model.py tests/test_gauge_features.py tests/test_meter_color_features.py tests/test_outcome.py tests/test_audio.py tests/test_engine.py tests/test_controller.py tests/test_replay_integration.py docs/superpowers/reports/2026-08-16-immediate-super-playback-resolution.md
git commit -m "feat: resolve immediate super playback"
```

- [ ] **Step 8: Prepare, but do not launch, the isolated candidate**

Create a new timestamped candidate directory under `<workspace>` rather than overwriting the stable candidate. Its launcher must use the exact tested source commit and preserve the current user configuration. Report the full launcher path. Do not start SF6 or use CU in this step.

---

## Self-Review Traceability

- Immediate playback in both scopes: Task 5.
- Stable restored 0/1/2/3 meter resolution: Tasks 2–3.
- Pre-spend baseline and cinematic suppression/extension: Task 3.
- Hit/block/whiff/filtered/unknown ordering: Tasks 3–4.
- Mutually exclusive whiff sound/fade: Task 4 and existing panel controls in Task 1.
- P1/P2 isolation, duplicate suppression, reset fade: Tasks 3 and 5.
- Desktop/task-view and stable CA regression: Task 6.
- No `main.py`, no MK1, no CU, no premature real-validation claim: Global Constraints and Task 6.
