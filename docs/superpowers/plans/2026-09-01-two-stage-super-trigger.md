# Two-Stage Super Trigger Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start configured SF6 music from a stable fast Super candidate, then confirm or cancel it without weakening the authoritative meter baseline or the existing hit/block/whiff policy.

**Architecture:** `GaugeTracker` remains the sole owner of meter and scene trust and emits typed candidate, confirmation, or cancellation transitions in `GaugeTrackerUpdate`. `VisualTriggerEngine` starts audio on a candidate, applies tier policy on confirmation, fades on cancellation, and begins outcome observation from the candidate timestamp. A bounded asynchronous JSONL writer records privacy-safe transition and audio metadata outside the capture hot path.

**Tech Stack:** Python 3.12, dataclasses/enums, NumPy/OpenCV observations, pytest 8, pygame audio boundary, Tkinter panel, PyInstaller portable packaging.

**Spec:** `docs/superpowers/specs/2026-09-01-two-stage-super-trigger-design.md`

## Global Constraints

- Keep the project Windows-only, screen-only, local, SF6-only, and independent for P1 and P2.
- Do not read game memory, inject input, inspect traffic, upload frames, operate MK1, or modify `main.py`.
- Preserve current schema-2 profiles, configuration fields, portable releases, and all backup tags.
- Candidate playback may last at most 0.75 seconds before an unconfirmed candidate begins fading.
- Candidate emission must not mutate the authoritative stable meter baseline.
- Diagnostic output must contain no screenshots, audio paths, window titles, usernames, or arbitrary exception text.
- Every production behavior change follows RED → GREEN → focused regression → commit.
- Do not package or claim real-match acceptance until the full offline suite and portable self-check pass.

---

### Task 1: Define Typed Candidate Lifecycle Events

**Files:**
- Modify: `src/sf6_music_trigger/domain.py:143-180`
- Modify: `tests/test_domain_config.py:70-230`

**Interfaces:**
- Consumes: existing `Side`, strict validation helpers, and `ConfirmedSuperEvent`.
- Produces: `CandidateCancelReason`, `SuperCandidateEvent`, `CancelledSuperCandidate`, and `GaugeTrackerUpdate`; adds `candidate_id: str | None = None` to `ConfirmedSuperEvent`.

- [ ] **Step 1: Write failing lifecycle contract tests**

Add imports and these representative tests to `tests/test_domain_config.py`:

```python
from sf6_music_trigger.domain import (
    CancelledSuperCandidate,
    CandidateCancelReason,
    GaugeTrackerUpdate,
    SuperCandidateEvent,
)


def _candidate(side: Side = Side.P1) -> SuperCandidateEvent:
    return SuperCandidateEvent(
        candidate_id="cand-p1-1",
        side=side,
        suspected_at=10.15,
        pre_spend_frame_id=100,
        candidate_frame_id=103,
        profile_id="profile",
        meter_before=3,
        meter_after=0,
        evidence=("stable_downward_level",),
    )


def test_candidate_lifecycle_values_are_frozen_and_side_bound() -> None:
    candidate = _candidate()
    assert (candidate.meter_before, candidate.meter_after) == (3, 0)
    assert candidate.levels_suspected == 3
    with pytest.raises(FrozenInstanceError):
        candidate.meter_after = 1

    cancelled = CancelledSuperCandidate(
        candidate.candidate_id,
        Side.P1,
        10.6,
        CandidateCancelReason.LEVEL_RECOVERED,
    )
    assert GaugeTrackerUpdate(cancelled).transition is cancelled


def test_confirmed_event_accepts_matching_candidate_identity() -> None:
    event = ConfirmedSuperEvent(
        "evt-1", Side.P1, 10.6, 100, 112, "profile",
        meter_before=3, meter_after=0, levels_spent=3,
        candidate_id="cand-p1-1",
    )
    assert event.candidate_id == "cand-p1-1"


@pytest.mark.parametrize("value", ["", "   ", 7, object()])
def test_candidate_identity_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SuperCandidateEvent(
            value, Side.P1, 1.0, 1, 3, "profile", 3, 0,
            ("stable_downward_level",),
        )
```

The production mutation these tests catch is accepting an untyped, mutable,
cross-side, or partially identified transition that the engine cannot safely
deduplicate.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_domain_config.py -k "candidate_lifecycle or matching_candidate or candidate_identity"
```

Expected: collection fails because the four new domain symbols and
`candidate_id` do not exist.

- [ ] **Step 3: Implement strict immutable domain values**

Add this public shape to `domain.py`, reusing the existing validation helpers:

```python
class CandidateCancelReason(str, Enum):
    LEVEL_RECOVERED = "LEVEL_RECOVERED"
    LEVEL_CONFLICT = "LEVEL_CONFLICT"
    CONFIRMATION_TIMEOUT = "CONFIRMATION_TIMEOUT"
    CAPTURE_RESET = "CAPTURE_RESET"
    TRACKER_RESET = "TRACKER_RESET"


@dataclass(frozen=True)
class SuperCandidateEvent:
    candidate_id: str
    side: Side
    suspected_at: Real
    pre_spend_frame_id: int
    candidate_frame_id: int
    profile_id: str
    meter_before: int
    meter_after: int
    evidence: tuple[str, ...]

    @property
    def levels_suspected(self) -> int:
        return self.meter_before - self.meter_after


@dataclass(frozen=True)
class CancelledSuperCandidate:
    candidate_id: str
    side: Side
    decided_at: Real
    reason: CandidateCancelReason


TrackerTransition = SuperCandidateEvent | ConfirmedSuperEvent | CancelledSuperCandidate


@dataclass(frozen=True)
class GaugeTrackerUpdate:
    transition: TrackerTransition | None = None
```

Validate exact integers, ordered frame IDs, meter range `0..3`, a strict
downward candidate, nonblank evidence strings, finite nonnegative times, and
actual enum/side instances. Add `candidate_id: str | None = None` after existing
optional meter metadata on `ConfirmedSuperEvent`; validate it when present.

- [ ] **Step 4: Run domain regression and verify GREEN**

Run:

```powershell
python -m pytest -q tests/test_domain_config.py
```

Expected: all domain/config tests pass.

- [ ] **Step 5: Commit the domain contract**

```powershell
git add src/sf6_music_trigger/domain.py tests/test_domain_config.py
git commit -m "feat: define two-stage super lifecycle"
```

---

### Task 2: Emit Candidate, Confirmation, and Cancellation from One Tracker

**Files:**
- Modify: `src/sf6_music_trigger/meter_level_tracker.py:70-184`
- Modify: `src/sf6_music_trigger/gauge_tracker.py:1-509`
- Modify: `tests/test_meter_level_tracker.py`
- Modify: `tests/test_gauge_tracker.py:1-1760`

**Interfaces:**
- Consumes: Task 1 lifecycle values and existing `GaugeObservation`/
  `StableMeterLevelTracker` authority.
- Produces: `GaugeTracker.update(observation) -> GaugeTrackerUpdate` and
  `GaugeTracker.reset(reason, decided_at=None) -> GaugeTrackerUpdate`.

- [ ] **Step 1: Expose read-only authority state with failing tests**

Add to `tests/test_meter_level_tracker.py`:

```python
def test_tracker_exposes_ready_baseline_without_allowing_candidate_mutation() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, baseline_samples=10, unknown_reset_samples=10
    )
    for frame_id in range(10):
        assert tracker.update(_reading(Side.P1, 3), frame_id) is None
    assert tracker.stable_level == 3
    assert tracker.stable_frame_id == 9
    assert tracker.baseline_ready is True
```

Run:

```powershell
python -m pytest -q tests/test_meter_level_tracker.py -k exposes_ready_baseline
```

Expected: FAIL because `stable_frame_id` and `baseline_ready` are absent.

- [ ] **Step 2: Add read-only authority properties and verify GREEN**

Implement:

```python
@property
def stable_frame_id(self) -> int | None:
    return self._stable_frame_id

@property
def baseline_ready(self) -> bool:
    return (
        self._stable_level is not None
        and self._stable_count >= self._baseline_samples
    )
```

Run:

```powershell
python -m pytest -q tests/test_meter_level_tracker.py
```

Expected: all stable-level tests pass.

- [ ] **Step 3: Write failing fast-candidate and baseline-isolation tests**

Update test helpers to unwrap lifecycle updates explicitly:

```python
def transition(update: GaugeTrackerUpdate):
    assert isinstance(update, GaugeTrackerUpdate)
    return update.transition


def feed_updates(tracker: GaugeTracker, observations: list[GaugeObservation]):
    return tuple(
        item
        for observation in observations
        if (item := transition(tracker.update(observation))) is not None
    )
```

Add focused tests using explicit 20 FPS timestamps:

```python
def test_three_lower_samples_emit_candidate_without_mutating_baseline() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(level_obs(frame_id, 3))) is None

    updates = [
        tracker.update(obs(
            10 + offset, 0.50 + offset * 0.05,
            full=1.0, spent=0.0, meter=0.0,
            meter_level_supported=True, meter_level=0,
            meter_level_confidence=0.9,
        ))
        for offset in range(3)
    ]

    candidate = transition(updates[-1])
    assert isinstance(candidate, SuperCandidateEvent)
    assert (candidate.meter_before, candidate.meter_after) == (3, 0)
    assert tracker.stable_meter_level == 3


def test_candidate_confirms_once_without_second_candidate() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    transitions = _feed_clean_level_until(tracker, level=0, through=1.20)
    assert [type(item) for item in transitions] == [ConfirmedSuperEvent]
    assert transitions[0].candidate_id == candidate.candidate_id
    assert tracker.stable_meter_level == 0
```

The mutation caught is routing audio only after authority confirmation or
lowering the trusted baseline at candidate time.

- [ ] **Step 4: Run candidate tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_gauge_tracker.py -k "three_lower_samples or candidate_confirms_once"
```

Expected: FAIL because tracker updates still return tuples and have no candidate
state.

- [ ] **Step 5: Implement the minimal candidate state inside `GaugeTracker`**

Add fields for one side-local candidate and one three-sample lower-level streak:

```python
self._active_candidate: SuperCandidateEvent | None = None
self._candidate_level: int | None = None
self._candidate_level_count = 0
self._candidate_deadline: Real | None = None
```

Before passing the authority reading to `StableMeterLevelTracker`, compare the
raw trustworthy level with `stable_level`. When `baseline_ready` and three
consecutive fresh raw readings agree below the baseline, create one
`SuperCandidateEvent`, set deadline to `suspected_at + Fraction(3, 4)`, retain
the authoritative baseline, and return `GaugeTrackerUpdate(candidate)`.

Refactor the rest of `_update_meter_level` to call a private
`_authority_transition(observation)`. If authority emits a downward transition,
create `ConfirmedSuperEvent(candidate_id=active.candidate_id)` and return it.
Clear the candidate only after creating its terminal transition.

- [ ] **Step 6: Write failing cancellation tests**

Add:

```python
def test_level_recovery_cancels_candidate_immediately() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    cancelled = transition(tracker.update(_level_at(20, 0.70, 3)))
    assert isinstance(cancelled, CancelledSuperCandidate)
    assert cancelled.candidate_id == candidate.candidate_id
    assert cancelled.reason is CandidateCancelReason.LEVEL_RECOVERED
    assert tracker.stable_meter_level == 3


def test_unconfirmed_candidate_cancels_at_fixed_deadline() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    before = transition(tracker.update(_unknown_level_at(20, 0.89)))
    at_deadline = transition(tracker.update(_unknown_level_at(21, 0.90)))
    assert before is None
    assert isinstance(at_deadline, CancelledSuperCandidate)
    assert at_deadline.reason is CandidateCancelReason.CONFIRMATION_TIMEOUT


def test_reset_returns_one_cancellation_before_clearing_candidate() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    update = tracker.reset("capture session changed", decided_at=0.5)
    assert transition(update).candidate_id == candidate.candidate_id
    assert transition(update).reason is CandidateCancelReason.CAPTURE_RESET
    assert tracker.reset("already clean") == GaugeTrackerUpdate()
```

Run the three tests and confirm they fail because cancellation is absent.

- [ ] **Step 7: Implement cancellation ordering and deadline handling**

Create `_cancel_candidate(reason, decided_at) -> GaugeTrackerUpdate`, invoke it
before clearing detection state, and map capture/window/geometry/session reset
reasons to `CAPTURE_RESET`; other explicit tracker resets use `TRACKER_RESET`.
Check recovery/conflict before timeout, and timeout before returning an empty
update. Ensure confirmation and cancellation cannot happen in one call.

- [ ] **Step 8: Migrate existing tracker tests to the typed update contract**

Use these exact assertion forms throughout `tests/test_gauge_tracker.py`:

```python
assert transition(tracker.update(observation)) is None
confirmed = transition(tracker.update(observation))
assert isinstance(confirmed, ConfirmedSuperEvent)
```

Keep `feed_levels` returning only confirmed events for historical regression
assertions, and add `feed_updates` for candidate/cancellation tests. Do not
delete old gain, glitch, wrap, round-transition, fractional-colour, ordering,
or atomicity cases.

- [ ] **Step 9: Run tracker regression and commit**

Run:

```powershell
python -m pytest -q tests/test_meter_level_tracker.py tests/test_gauge_tracker.py
```

Expected: all tests pass, including no candidates from gains, wraps, unknowns,
or a single-frame level glitch.

```powershell
git add src/sf6_music_trigger/meter_level_tracker.py src/sf6_music_trigger/gauge_tracker.py tests/test_meter_level_tracker.py tests/test_gauge_tracker.py
git commit -m "feat: emit fast super candidate lifecycle"
```

---

### Task 3: Split Candidate, Confirmation, Cancellation, and Outcome Audio Policy

**Files:**
- Modify: `src/sf6_music_trigger/audio.py:151-230`
- Modify: `tests/test_audio.py:190-340`

**Interfaces:**
- Consumes: Task 1 lifecycle values and existing `SideAudioConfig`.
- Produces: `on_candidate`, `on_confirmation`, `on_cancellation`, and unchanged
  `on_outcome` methods on `AudioScheduler`.

- [ ] **Step 1: Write failing candidate and terminal-policy tests**

Add candidate/cancellation factories and tests:

```python
def test_every_enabled_candidate_starts_main_track_for_both_scopes() -> None:
    scheduler = AudioScheduler(Side.P1)
    for scope in (SuperScope.ALL_SUPERS, SuperScope.SA3_CA_ONLY):
        actions = scheduler.on_candidate(
            _candidate(),
            SideAudioConfig(
                enabled=True, main_track="audio/main.mp3", volume=0.18,
                super_scope=scope,
            ),
        )
        assert [item.kind for item in actions] == [AudioKind.STOP, AudioKind.PLAY]


def test_started_low_tier_confirmation_fades_without_whiff_sound() -> None:
    actions = AudioScheduler(Side.P1).on_confirmation(
        _event(levels_spent=1),
        SideAudioConfig(
            enabled=True, main_track="audio/main.mp3",
            play_whiff_sound=True, whiff_track="audio/whiff.mp3",
            super_scope=SuperScope.SA3_CA_ONLY,
        ),
        playback_started=True,
    )
    assert len(actions) == 1
    assert actions[0].kind is AudioKind.FADE


def test_candidate_cancellation_always_fades_and_never_plays_whiff() -> None:
    actions = AudioScheduler(Side.P1).on_cancellation(
        _cancellation(),
        SideAudioConfig(
            enabled=True, main_track="audio/main.mp3",
            play_whiff_sound=True, whiff_track="audio/whiff.mp3",
        ),
    )
    assert [item.kind for item in actions] == [AudioKind.FADE]
```

- [ ] **Step 2: Run focused audio tests and verify RED**

```powershell
python -m pytest -q tests/test_audio.py -k "enabled_candidate or low_tier_confirmation or candidate_cancellation"
```

Expected: FAIL because the three scheduler methods do not exist.

- [ ] **Step 3: Implement stateless policy methods**

Use one private `_main_play(config, side)` helper. Implement confirmation rules
exactly:

```python
if not config.enabled:
    return ()
filtered = (
    config.super_scope is SuperScope.SA3_CA_ONLY
    and event.levels_spent in {1, 2}
)
if filtered:
    return (AudioAction(AudioKind.FADE, self.side,
                        fade_ms=config.whiff_fade_ms),) if playback_started else ()
return () if playback_started else self._main_play(config, self.side)
```

Remove `on_confirmed_spend` only after all call sites and tests use the new
methods. Preserve validation before disabled-config short circuits so side/type
errors remain deterministic.

- [ ] **Step 4: Run audio regression and commit**

```powershell
python -m pytest -q tests/test_audio.py
git add src/sf6_music_trigger/audio.py tests/test_audio.py
git commit -m "feat: split provisional and terminal audio policy"
```

---

### Task 4: Anchor Outcome Timing to Candidate Playback

**Files:**
- Modify: `src/sf6_music_trigger/outcome.py:220-240`
- Modify: `tests/test_outcome.py:90-280`

**Interfaces:**
- Consumes: `ConfirmedSuperEvent`, `SuperScope`, and a candidate activation
  timestamp supplied by the engine.
- Produces: `OutcomeObserver.begin(event, scope, *, activation_at)`.

- [ ] **Step 1: Write failing activation-time tests**

```python
def test_begin_anchors_checkpoint_to_candidate_activation() -> None:
    features = _Features({})
    observer = OutcomeObserver(
        checkpoint_seconds=3.0,
        hard_deadline_seconds=8.0,
        feature_extractor=features,
    )
    _remember_baseline(observer, features)
    event = _event(confirmed_at=1.6)
    observer.begin(event, SuperScope.ALL_SUPERS, activation_at=1.0)

    assert observer.update(
        _frame(10, 3.99), observations=_trusted_observations(10, 3.99)
    ) == ()
    decisions = observer.update(
        _frame(11, 4.0), observations=_trusted_observations(11, 4.0)
    )
    assert decisions[0].label is OutcomeLabel.WHIFF


def test_begin_rejects_activation_after_confirmation_atomically() -> None:
    observer = OutcomeObserver()
    with pytest.raises(ValueError, match="activation_at"):
        observer.begin(
            _event(confirmed_at=1.0), SuperScope.ALL_SUPERS,
            activation_at=1.01,
        )
    assert observer.pending_sides == ()
```

- [ ] **Step 2: Run focused outcome tests and verify RED**

```powershell
python -m pytest -q tests/test_outcome.py -k "anchors_checkpoint or activation_after"
```

Expected: FAIL because `begin` does not accept `activation_at`.

- [ ] **Step 3: Implement activation-time validation and deadlines**

Validate `activation_at` with `_require_nonnegative_real`, reject values greater
than `event.confirmed_at`, and derive suppression/checkpoint/hard deadline from
the validated activation time. Keep baseline lookup keyed by
`pre_spend_frame_id` and duplicate handling unchanged.

- [ ] **Step 4: Update all callers in outcome tests and commit**

Legacy tests pass `activation_at=event.confirmed_at`. Run:

```powershell
python -m pytest -q tests/test_outcome.py
git add src/sf6_music_trigger/outcome.py tests/test_outcome.py
git commit -m "feat: anchor outcomes to candidate playback"
```

---

### Task 5: Route the Lifecycle Exactly Once in the Engine

**Files:**
- Modify: `src/sf6_music_trigger/engine.py:20-387`
- Modify: `tests/test_engine.py:350-1320`
- Modify: `tests/test_replay_integration.py`

**Interfaces:**
- Consumes: Tasks 1–4 tracker updates, scheduler methods, and outcome
  `activation_at`.
- Produces: engine candidate/cancellation logs, exact audio routing, and
  `EngineUpdate(events=..., candidates=..., cancellations=...)`.

- [ ] **Step 1: Upgrade engine spies and write failing candidate-routing test**

Change `_TrackerSpy.update` to return `GaugeTrackerUpdate()` by default and
`reset` to return `GaugeTrackerUpdate()`. Replace `_SchedulerSpy` calls with
separate candidate, confirmation, cancellation, and outcome lists. Add:

```python
def test_candidate_starts_once_and_confirmation_does_not_restart() -> None:
    candidate = _candidate(side=Side.P1, candidate_id="cand-1")
    confirmed = _event(
        side=Side.P1, event_id="evt-1", candidate_id="cand-1",
        levels_spent=3,
    )
    p1_tracker = _TrackerSpy([
        GaugeTrackerUpdate(candidate),
        GaugeTrackerUpdate(confirmed),
    ])
    harness = _build_harness(p1_tracker=p1_tracker)

    _drive(harness, [(1.0, "level_3", "level_unknown"),
                     (1.6, "level_0", "level_unknown")])

    assert len(harness.audio.batches) == 1
    assert [item.kind for item in harness.audio.batches[0]] == [
        AudioKind.STOP, AudioKind.PLAY,
    ]
    assert harness.outcome_observer.begin_calls == [
        (confirmed, harness.p1_config.super_scope, candidate.suspected_at)
    ]
```

- [ ] **Step 2: Run the engine test and verify RED**

```powershell
python -m pytest -q tests/test_engine.py -k candidate_starts_once
```

Expected: FAIL because engine validation still expects tuples of confirmed
events and audio still starts at confirmation.

- [ ] **Step 3: Implement lifecycle routing and pending candidate state**

Add:

```python
self._pending_candidates_by_side: dict[Side, SuperCandidateEvent] = {}
self._terminal_candidate_ids: set[str] = set()
```

Replace `_validated_tracker_events` with `_validated_tracker_update`. Route in
three private methods `_route_candidate`, `_route_confirmation`, and
`_route_cancellation`. Validate side and matching ID before state mutation.
Append only confirmations to existing `event_log`; expose separate immutable
`candidate_log` and `cancellation_log` properties. Add candidate and
cancellation tuples to `EngineUpdate` without changing the meaning of
`events`.

- [ ] **Step 4: Write failing tier, cancellation, reset, duplicate, and recovery tests**

Add separate behavioral cases proving:

```python
# started SA1 under SA3_CA_ONLY -> one play batch, then one fade batch
assert [[a.kind for a in batch] for batch in harness.audio.batches] == [
    [AudioKind.STOP, AudioKind.PLAY], [AudioKind.FADE]
]

# cancellation -> fade, no outcome begin
assert harness.outcome_observer.begin_calls == []

# confirmation without candidate -> accepted tier starts once
assert len(harness.audio.batches) == 1

# duplicate candidate/terminal IDs -> no additional audio
assert len(harness.audio.batches) == expected_batch_count
```

For reset, configure tracker spies to return one cancellation from `reset` and
assert the cancellation fade is routed before outcome reset actions.

- [ ] **Step 5: Implement terminal routing, legacy recovery, and reset ordering**

On confirmation, pop the matching candidate, call
`on_confirmation(..., playback_started=True)`, and begin outcome only if scope
accepts the tier. Without a pending candidate call confirmation with
`playback_started=False` and use `event.confirmed_at` as activation time.
On cancellation, pop only a matching live candidate and call
`on_cancellation`. During `_runtime_reset` and explicit `reset`, route tracker
reset updates before `OutcomeObserver.reset`.

- [ ] **Step 6: Preserve outcome and replay behavior**

Update replay fakes to accept `GaugeTrackerUpdate`. Add a deterministic replay
sequence containing baseline 3, three provisional 0 samples, confirmation,
and hit evidence. Assert playback action counts occur on the candidate frame,
not the confirmation frame. Add a round-transition candidate sequence that
ends in one cancellation fade and zero confirmed events.

- [ ] **Step 7: Run engine/replay regression and commit**

```powershell
python -m pytest -q tests/test_engine.py tests/test_replay_integration.py
git add src/sf6_music_trigger/engine.py tests/test_engine.py tests/test_replay_integration.py
git commit -m "feat: route provisional super audio lifecycle"
```

---

### Task 6: Add Bounded Privacy-Safe Diagnostic JSONL

**Files:**
- Create: `src/sf6_music_trigger/diagnostic_log.py`
- Create: `tests/test_diagnostic_log.py`
- Modify: `src/sf6_music_trigger/engine.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- Consumes: typed lifecycle transitions, outcome decisions, scope enum, and
  audio action kinds.
- Produces: `DiagnosticRecord`,
  `DiagnosticEventLog.start/record/close/has_warning`, plus an engine
  `diagnostic_sink(record)` boundary.

- [ ] **Step 1: Write failing real-filesystem log tests**

```python
def test_log_writes_fixed_privacy_safe_schema_on_close(tmp_path: Path) -> None:
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=8)
    log.start()
    log.record(DiagnosticRecord(
        monotonic_time=1.25,
        side=Side.P1,
        identity="cand-1",
        transition="CANDIDATE",
        evidence=("stable_downward_level",),
        meter_before=3,
        meter_after=0,
        levels_spent=None,
        scope=SuperScope.SA3_CA_ONLY,
        audio_actions=(AudioKind.STOP, AudioKind.PLAY),
        outcome=None,
    ))
    log.close()

    payload = json.loads(next((tmp_path / "diagnostics").glob("*.jsonl")).read_text())
    assert payload["schema_version"] == 1
    assert payload["transition"] == "CANDIDATE"
    assert payload["audio_actions"] == ["stop", "play"]
    assert "track" not in payload
    assert "path" not in payload


def test_unstarted_bounded_queue_drops_without_blocking(tmp_path: Path) -> None:
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=1)
    log.record(_record(identity="one"))
    log.record(_record(identity="two"))
    assert log.dropped_count == 1
    log.close()
```

Add rotation tests with `max_bytes=200` and `retained_files=2`, then assert only
the two newest files inside the exact diagnostics directory remain.

- [ ] **Step 2: Run log tests and verify RED**

```powershell
python -m pytest -q tests/test_diagnostic_log.py
```

Expected: import failure because the module is absent.

- [ ] **Step 3: Implement the bounded writer**

Use default limits `max_bytes=2 * 1024 * 1024`, `retained_files=5`, and
`queue_capacity=1024`. Use `Queue(maxsize=queue_capacity)` and one non-daemon
worker thread. `record`
uses `put_nowait`; `Full` increments `dropped_count` and returns immediately.
`start` creates only `<root>/diagnostics`, opens a session JSONL file, and
starts the worker. `close` sends a private sentinel, joins the worker, flushes
and closes the file, and is idempotent. Serialize a hand-built dictionary of
the fixed `DiagnosticRecord` fields; never serialize `repr(record)`, exceptions,
configuration objects, or audio actions carrying tracks.

Expose `has_warning: bool` under the writer lock. It becomes true after queue
overflow, directory/open/write/rotation failure, or incomplete worker close;
reading it has no side effect and exposes no exception text.

- [ ] **Step 4: Add engine diagnostic boundary with failing tests**

Pass `diagnostic_sink: Callable[[DiagnosticRecord], None] = _ignore_record` to
`VisualTriggerEngine`. In engine tests use a real list's `append` and assert one
record per candidate, confirmation/cancellation, outcome, and audio batch with
only action kinds.

- [ ] **Step 5: Implement engine records and run focused regression**

Build records only after transition validation and audio policy calculation.
If the sink raises, set an engine `diagnostic_warning` boolean and continue
gameplay routing; do not retry the failed record on the capture thread. The
engine exposes this boolean together with the writer's `has_warning` through
the controller diagnostics boundary.

Run:

```powershell
python -m pytest -q tests/test_diagnostic_log.py tests/test_engine.py
git add src/sf6_music_trigger/diagnostic_log.py src/sf6_music_trigger/engine.py tests/test_diagnostic_log.py tests/test_engine.py
git commit -m "feat: record privacy-safe trigger decisions"
```

---

### Task 7: Wire Logging and Candidate Status into Listener Lifecycle

**Files:**
- Modify: `src/sf6_music_trigger/controller.py:20-280`
- Modify: `src/sf6_music_trigger/panel.py:790-860, 1220-1590`
- Modify: `tests/test_controller.py`
- Modify: `tests/test_panel_view_model.py`

**Interfaces:**
- Consumes: `DiagnosticEventLog`, engine status strings, and tracker update/reset
  behavior.
- Produces: listener-owned log start/close lifecycle,
  `ControllerStatus(kind="diagnostic_error")`, and safe panel states
  `疑似大招，正在确认` and `疑似大招已撤销`.

- [ ] **Step 1: Write failing controller lifecycle tests**

Create a `_DiagnosticLogSpy` with validated `start`/`close` call counters and
pass it through a new optional `diagnostic_log` constructor argument. Assert:

```python
controller.start()
controller.join(timeout=2)
assert diagnostic_log.start_calls == 1
assert diagnostic_log.close_calls == 1
```

Add a start-failure case proving a worker error status and no source start, and
a close-failure case proving one worker error without preventing audio close.

- [ ] **Step 2: Run controller tests and verify RED**

```powershell
python -m pytest -q tests/test_controller.py -k diagnostic_log
```

Expected: FAIL because `ListenerController` rejects the constructor argument.

- [ ] **Step 3: Implement listener ownership and safe statuses**

Extend `ControllerStatusKind` with `diagnostic_error`; it carries no operation,
side, issue, path, or exception text. Validate `start`, `close`, and
`has_warning` when a log is supplied. Start the log
before the capture source and close it in the worker's `finally` after engine
reset but before posting `stopped`. After each engine update, publish one
latched `diagnostic_error` status if either engine or writer warning is true.
Extend `_SAFE_OUTCOME_STATUSES` and panel mapping only with the two approved
Chinese labels.

- [ ] **Step 4: Wire the portable data root and one-shot warning**

In `_build_listener_controller`, construct:

```python
diagnostic_log = DiagnosticEventLog(application.user_data_root)
```

Pass `diagnostic_log.record` to the engine and the log itself to the
controller. Translate `ControllerStatus(kind="diagnostic_error")` to one
privacy-safe panel message `诊断日志不可用，检测仍在运行`; never include its
path or raw exception.

- [ ] **Step 5: Run panel/controller regression and commit**

```powershell
python -m pytest -q tests/test_controller.py tests/test_panel_view_model.py
git add src/sf6_music_trigger/controller.py src/sf6_music_trigger/panel.py tests/test_controller.py tests/test_panel_view_model.py
git commit -m "feat: expose provisional trigger diagnostics"
```

---

### Task 8: Replay Gates, Full Verification, and New Portable Candidate

**Files:**
- Modify: `tools/replay_meter_levels.py`
- Modify: `tests/test_replay_integration.py`
- Modify: `tests/test_portable_release.py`
- Modify: `tools/portable_release.py` only if the diagnostics directory needs
  an explicit public-payload rule
- Create: `docs/verification/2026-09-01-two-stage-trigger-release.md`
- Create: sibling release directory
  `SF6_Music_Trigger_GroupTest_0.1.0a1_20260901_r16`
- Create: sibling archive
  `SF6_Music_Trigger_GroupTest_0.1.0a1_20260901_r16.zip`

**Interfaces:**
- Consumes: complete two-stage implementation and existing local replay/video
  evidence.
- Produces: deterministic offline evidence and a non-overwriting r16 group-test
  package only if every gate passes.

- [ ] **Step 1: Add failing replay timing and false-trigger assertions**

Extend replay output with privacy-safe transition rows:

```json
{
  "frame_id": 103,
  "side": "P1",
  "transition": "CANDIDATE",
  "meter_before": 3,
  "meter_after": 0,
  "audio_actions": ["stop", "play"]
}
```

Tests assert candidate frame, terminal frame, confirmed-event count, and action
kinds from literal synthetic sequences. The round-end sequence must have zero
confirmed events and at most one candidate followed by one cancellation fade.

- [ ] **Step 2: Run replay tests and verify RED, then implement output**

```powershell
python -m pytest -q tests/test_replay_integration.py
```

Expected initial failure: replay result lacks lifecycle transition rows.
Implement serialization from engine transition logs without IDs, paths, or
captured pixels, then rerun until GREEN.

- [ ] **Step 3: Run focused product regression**

```powershell
python -m pytest -q tests/test_domain_config.py tests/test_meter_level_tracker.py tests/test_gauge_tracker.py tests/test_audio.py tests/test_outcome.py tests/test_engine.py tests/test_diagnostic_log.py tests/test_controller.py tests/test_panel_view_model.py tests/test_replay_integration.py
```

Expected: all focused tests pass with no warnings or errors.

- [ ] **Step 4: Run the complete offline suite**

```powershell
python -m pytest -q
```

Expected: every test passes. Record the exact count and duration in the
verification document; do not reuse the earlier 1,745 count.

- [ ] **Step 5: Run deterministic private-video replay locally**

Use `tools/replay_meter_levels.py` against the existing local evidence without
copying, uploading, or committing frames. Record only aggregate candidate,
confirmation, cancellation, outcome, and audio-action counts. At minimum check:

- the earlier stable CA recording;
- the long real-match false-trigger recording;
- `中，空，挡.mp4`;
- `22分钟实机.mp4`; and
- `08.mp4`.

If video decoding cannot run inside the current environment, record the exact
offline blocker and do not claim the replay gate passed.

- [ ] **Step 6: Build r16 without overwriting r15 or earlier releases**

Run the existing `tools/portable_release.py` pipeline with the sibling release
root `SF6_Music_Trigger_GroupTest_0.1.0a1_20260901_r16` beside the source
repository.

The builder must refuse if that folder or ZIP already exists. Run
`SF6 Music Trigger.exe --portable-self-check`, verify `SHA256SUMS.txt`, ensure
the ZIP has one top-level folder, and confirm public payload hygiene.

- [ ] **Step 7: Write verification evidence and commit source/tests/docs**

The verification document records commit, exact commands, pass counts, replay
aggregates, self-check exit code, release absolute paths, hashes, and the clear
statement `SF6 real acceptance not yet performed` unless an explicitly
authorized live session is completed.

```powershell
git add tools/replay_meter_levels.py tests/test_replay_integration.py tests/test_portable_release.py tools/portable_release.py docs/verification/2026-09-01-two-stage-trigger-release.md
git commit -m "test: verify two-stage super trigger candidate"
```

- [ ] **Step 8: Inspect final scope**

```powershell
git status --short
git diff cbaaabe..HEAD --stat
git log --oneline -10
```

Expected: source repository is clean; only the intended domain, tracker, audio,
outcome, engine, diagnostic, controller/panel, replay/release tests, plan/spec,
and verification document appear in the change history. Runtime configuration,
private videos, selected music, and `main.py` are absent.
