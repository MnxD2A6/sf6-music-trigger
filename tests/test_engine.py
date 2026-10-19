from __future__ import annotations

from collections import deque
from dataclasses import dataclass, FrozenInstanceError, fields
from fractions import Fraction
import inspect
import time
from typing import Any, Callable

import numpy as np
import pytest

from sf6_music_trigger.audio import (
    AudioAction,
    AudioIssue,
    AudioKind,
    AudioPlaybackError,
    AudioScheduler,
)
from sf6_music_trigger.capture import FreshFrameGate
from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.domain import (
    CancelledSuperCandidate,
    CandidateCancelReason,
    CaptureIssue,
    CaptureSample,
    CaptureStatus,
    CapturedFrame,
    ClientGeometry,
    ConfirmedSuperEvent,
    GaugeTrackerUpdate,
    Side,
    SuperCandidateEvent,
)
from sf6_music_trigger.diagnostic_log import DiagnosticRecord
from sf6_music_trigger.gauge_features import GaugeObservation
from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig
from sf6_music_trigger.hud_profile import (
    CalibrationContext,
    HudProfile,
    NormalizedRoi,
    SideHudProfile,
)
from sf6_music_trigger.meter_references import MeterReferences
from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel


def _profile(
    *,
    aspect_ratio: float = 2.0,
    p1_roi: NormalizedRoi | None = None,
    p2_roi: NormalizedRoi | None = None,
) -> HudProfile:
    width = 200
    height = 100
    assert width / height == aspect_ratio
    return HudProfile(
        profile_id="task8-profile",
        calibration=CalibrationContext(
            client_width=width,
            client_height=height,
            client_aspect_ratio=aspect_ratio,
            hud_scale_setting="default",
            language="English",
            brightness_setting="default",
            hdr_enabled=False,
        ),
        battle_roi=NormalizedRoi(0.0, 0.0, 1.0, 1.0),
        training_hud_roi=None,
        p1=SideHudProfile(
            meter_roi=p1_roi or NormalizedRoi(0.05, 0.05, 0.35, 0.25),
            references_path="references/p1.npz",
        ),
        p2=SideHudProfile(
            meter_roi=p2_roi or NormalizedRoi(0.65, 0.05, 0.95, 0.25),
            references_path="references/p2.npz",
        ),
    )


def _event(
    side: Side = Side.P1,
    *,
    event_id: str | None = None,
    candidate_id: str | None = None,
    confirmed_at: float = 1.0,
    levels_spent: int | None = None,
) -> ConfirmedSuperEvent:
    meter_before = levels_spent if levels_spent is not None else None
    meter_after = 0 if levels_spent is not None else None
    return ConfirmedSuperEvent(
        event_id=event_id or f"event-{side.value}",
        side=side,
        confirmed_at=confirmed_at,
        pre_spend_frame_id=1,
        confirmed_frame_id=2,
        profile_id="task8-profile",
        candidate_id=candidate_id,
        meter_before=meter_before,
        meter_after=meter_after,
        levels_spent=levels_spent,
    )


def _candidate(
    side: Side = Side.P1,
    *,
    candidate_id: str | None = None,
    suspected_at: float = 1.0,
    levels_suspected: int = 3,
) -> SuperCandidateEvent:
    return SuperCandidateEvent(
        candidate_id=candidate_id or f"candidate-{side.value}",
        side=side,
        suspected_at=suspected_at,
        pre_spend_frame_id=1,
        candidate_frame_id=2,
        profile_id="task8-profile",
        meter_before=levels_suspected,
        meter_after=0,
        evidence=("stable_downward_level",),
    )


def _cancellation(
    side: Side = Side.P1,
    *,
    candidate_id: str | None = None,
    decided_at: float = 1.5,
    reason: CandidateCancelReason = CandidateCancelReason.CONFIRMATION_TIMEOUT,
) -> CancelledSuperCandidate:
    return CancelledSuperCandidate(
        candidate_id=candidate_id or f"candidate-{side.value}",
        side=side,
        decided_at=decided_at,
        reason=reason,
    )


def test_engine_update_exposes_immutable_lifecycle_tuples() -> None:
    from sf6_music_trigger.engine import EngineUpdate

    update = EngineUpdate()
    assert update.events == ()
    assert update.candidates == ()
    assert update.cancellations == ()
    assert [field.name for field in fields(EngineUpdate)] == [
        "events",
        "candidates",
        "cancellations",
    ]
    assert vars(update) == {
        "events": (),
        "candidates": (),
        "cancellations": (),
    }
    with pytest.raises(TypeError):
        EngineUpdate(events=[])
    with pytest.raises(TypeError):
        EngineUpdate(events=(object(),))
    with pytest.raises(TypeError):
        EngineUpdate(candidates=[])
    with pytest.raises(TypeError):
        EngineUpdate(candidates=(object(),))
    with pytest.raises(TypeError):
        EngineUpdate(cancellations=[])
    with pytest.raises(TypeError):
        EngineUpdate(cancellations=(object(),))
    with pytest.raises(FrozenInstanceError):
        update.events = ()


def test_engine_update_preserves_the_exact_event_tuple() -> None:
    from sf6_music_trigger.engine import EngineUpdate

    events = (_event(Side.P1), _event(Side.P2))
    candidates = (_candidate(Side.P1), _candidate(Side.P2))
    cancellations = (_cancellation(Side.P1), _cancellation(Side.P2))
    update = EngineUpdate(
        events=events,
        candidates=candidates,
        cancellations=cancellations,
    )
    assert update.events is events
    assert update.candidates is candidates
    assert update.cancellations is cancellations


class _TupleSubclass(tuple):
    pass


def test_engine_update_rejects_tuple_subclasses() -> None:
    from sf6_music_trigger.engine import EngineUpdate

    with pytest.raises(TypeError):
        EngineUpdate(events=_TupleSubclass((_event(),)))
    with pytest.raises(TypeError):
        EngineUpdate(candidates=_TupleSubclass((_candidate(),)))
    with pytest.raises(TypeError):
        EngineUpdate(cancellations=_TupleSubclass((_cancellation(),)))


@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        (ClientGeometry(10, -20, 400, 200), True),
        (ClientGeometry(0, 0, 20_199, 10_000), True),
        (ClientGeometry(0, 0, 20_200, 10_000), True),
        (ClientGeometry(0, 0, 20_201, 10_000), False),
    ],
    ids=("same-aspect-other-resolution", "below-one-percent", "exact-one-percent", "above-one-percent"),
)
def test_profile_applies_to_uses_inclusive_relative_aspect_boundary(
    geometry: ClientGeometry,
    expected: bool,
) -> None:
    from sf6_music_trigger.engine import profile_applies_to

    assert profile_applies_to(_profile(), geometry) is expected


def test_profile_applies_to_uses_absolute_boundary_tolerance() -> None:
    from sf6_music_trigger.engine import profile_applies_to

    profile = _profile()
    exact_boundary = ClientGeometry(0, 0, 20_200, 10_000)
    assert profile_applies_to(
        profile,
        exact_boundary,
        max_relative_aspect_error=0.01 - 5e-13,
    ) is True
    assert profile_applies_to(
        profile,
        exact_boundary,
        max_relative_aspect_error=0.01 - 2e-12,
    ) is False


def test_profile_applies_to_accepts_huge_finite_rational_tolerance_without_float_coercion() -> None:
    from sf6_music_trigger.engine import profile_applies_to

    tolerance = Fraction(10**10_000, 1)
    assert profile_applies_to(
        _profile(),
        ClientGeometry(0, 0, 1, 1),
        max_relative_aspect_error=tolerance,
    ) is True


@pytest.mark.parametrize("bad_profile", [None, object(), "profile"])
def test_profile_applies_to_rejects_non_profiles(bad_profile: object) -> None:
    from sf6_music_trigger.engine import profile_applies_to

    with pytest.raises(TypeError):
        profile_applies_to(bad_profile, ClientGeometry(0, 0, 200, 100))


@pytest.mark.parametrize("bad_geometry", [None, object(), (200, 100)])
def test_profile_applies_to_rejects_non_geometries(bad_geometry: object) -> None:
    from sf6_music_trigger.engine import profile_applies_to

    with pytest.raises(TypeError):
        profile_applies_to(_profile(), bad_geometry)


@pytest.mark.parametrize("bad_tolerance", [True, False, None, "0.01", object()])
def test_profile_applies_to_rejects_non_real_or_boolean_tolerance(
    bad_tolerance: object,
) -> None:
    from sf6_music_trigger.engine import profile_applies_to

    with pytest.raises(TypeError):
        profile_applies_to(
            _profile(),
            ClientGeometry(0, 0, 200, 100),
            max_relative_aspect_error=bad_tolerance,
        )


@pytest.mark.parametrize("bad_tolerance", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_profile_applies_to_rejects_negative_or_nonfinite_tolerance(
    bad_tolerance: float,
) -> None:
    from sf6_music_trigger.engine import profile_applies_to

    with pytest.raises(ValueError):
        profile_applies_to(
            _profile(),
            ClientGeometry(0, 0, 200, 100),
            max_relative_aspect_error=bad_tolerance,
        )


def _references(side: Side) -> MeterReferences:
    return MeterReferences(
        side=side,
        full_gray=np.zeros((32, 64), dtype=np.uint8),
        spent_gray=np.full((32, 64), 255, dtype=np.uint8),
    )


def _frame(
    frame_id: int,
    sampled_at: float,
    *,
    geometry: ClientGeometry | None = None,
) -> CapturedFrame:
    actual_geometry = geometry or ClientGeometry(0, 0, 200, 100)
    image = np.zeros(
        (actual_geometry.height, actual_geometry.width, 3),
        dtype=np.uint8,
        order="C",
    )
    assert image.flags.owndata and image.flags.writeable and image.flags.c_contiguous
    return CapturedFrame(
        frame_id=frame_id,
        image=image,
        monotonic_time=sampled_at,
        geometry=actual_geometry,
    )


def _running(frame: CapturedFrame) -> CaptureSample:
    return CaptureSample(status=CaptureStatus.RUNNING, frame=frame)


def _observation(
    side: Side,
    frame_id: int,
    sampled_at: float,
    kind: str,
) -> GaugeObservation:
    if kind == "level_unknown":
        return GaugeObservation(
            side,
            frame_id,
            sampled_at,
            False,
            0.0,
            0.0,
            1.0,
            1.0,
            meter_level_supported=True,
        )
    if kind.startswith(("level_", "empty_level_")):
        empty_fill = kind.startswith("empty_level_")
        level = int(kind.removeprefix("empty_").removeprefix("level_"))
        return GaugeObservation(
            side,
            frame_id,
            sampled_at,
            True,
            1.0,
            0.0,
            0.0,
            1.0,
            normalized_meter=0.0 if empty_fill else level / 3,
            meter_level_supported=True,
            meter_level=level,
            meter_level_confidence=0.9,
        )
    if kind == "hidden":
        return GaugeObservation(side, frame_id, sampled_at, False, 0.0, 0.0, 1.0, 1.0)
    if kind == "full":
        return GaugeObservation(side, frame_id, sampled_at, True, 1.0, 0.0, 0.0, 1.0)
    if kind == "spent":
        return GaugeObservation(side, frame_id, sampled_at, True, 0.0, 1.0, 1.0, 0.0)
    if kind == "unverified_spent":
        return GaugeObservation(
            side,
            frame_id,
            sampled_at,
            True,
            0.0,
            1.0,
            1.0,
            0.0,
            spent_structure_verified=False,
        )
    if kind == "ambiguous":
        return GaugeObservation(side, frame_id, sampled_at, True, 0.5, 0.5, 1.0, 1.0)
    raise AssertionError(f"unsupported scripted observation: {kind}")


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.calls: list[float] = []

    def __call__(self) -> float:
        self.calls.append(self.value)
        return self.value


class _Observer:
    def __init__(self) -> None:
        self.kinds: dict[tuple[int, Side], str] = {}
        self.calls: list[dict[str, Any]] = []
        self.frame_args: list[np.ndarray] = []

    def __call__(
        self,
        frame_bgr: np.ndarray,
        *,
        side: Side,
        frame_id: int,
        roi: NormalizedRoi,
        references: MeterReferences,
        sampled_at: float,
    ) -> GaugeObservation:
        self.frame_args.append(frame_bgr)
        self.calls.append(
            {
                "side": side,
                "frame_id": frame_id,
                "roi": roi,
                "references": references,
                "sampled_at": sampled_at,
            }
        )
        return _observation(
            side,
            frame_id,
            sampled_at,
            self.kinds.get((frame_id, side), "hidden"),
        )


class _TrackerSpy:
    def __init__(
        self,
        returns: list[object] | None = None,
        *,
        reset_returns: list[object] | None = None,
    ) -> None:
        self._returns = deque(returns or [])
        self._reset_returns = deque(reset_returns or [])
        self.observations: list[GaugeObservation] = []
        self.reset_reasons: list[str] = []

    def update(self, observation: GaugeObservation) -> object:
        self.observations.append(observation)
        return self._returns.popleft() if self._returns else GaugeTrackerUpdate()

    def reset(self, reason: str) -> object:
        self.reset_reasons.append(reason)
        return (
            self._reset_returns.popleft()
            if self._reset_returns
            else GaugeTrackerUpdate()
        )


class _SchedulerSpy:
    def __init__(self, side: Side, result: object) -> None:
        self.side = side
        self.result = result
        self.candidate_calls: list[tuple[SuperCandidateEvent, SideAudioConfig]] = []
        self.confirmation_calls: list[
            tuple[ConfirmedSuperEvent, SideAudioConfig, bool]
        ] = []
        self.cancellation_calls: list[
            tuple[CancelledSuperCandidate, SideAudioConfig]
        ] = []
        self.outcome_calls: list[tuple[OutcomeDecision, SideAudioConfig]] = []

    def on_candidate(
        self,
        candidate: SuperCandidateEvent,
        config: SideAudioConfig,
    ) -> object:
        self.candidate_calls.append((candidate, config))
        return self.result

    def on_confirmation(
        self,
        event: ConfirmedSuperEvent,
        config: SideAudioConfig,
        playback_started: bool,
    ) -> object:
        self.confirmation_calls.append((event, config, playback_started))
        return self.result

    def on_cancellation(
        self,
        cancellation: CancelledSuperCandidate,
        config: SideAudioConfig,
    ) -> object:
        self.cancellation_calls.append((cancellation, config))
        return self.result

    def on_outcome(
        self,
        decision: OutcomeDecision,
        config: SideAudioConfig,
    ) -> object:
        self.outcome_calls.append((decision, config))
        return self.result


class _OutcomeObserverSpy:
    def __init__(self) -> None:
        self.remember_calls: list[CapturedFrame] = []
        self.begin_calls: list[tuple[ConfirmedSuperEvent, SuperScope, float]] = []
        self.update_calls: list[
            tuple[CapturedFrame, tuple[GaugeObservation, GaugeObservation]]
        ] = []
        self.reset_reasons: list[str] = []
        self.results: dict[int, tuple[OutcomeDecision, ...]] = {}
        self.reset_results: tuple[OutcomeDecision, ...] = ()

    def remember(self, frame: CapturedFrame) -> None:
        self.remember_calls.append(frame)

    def begin(
        self,
        event: ConfirmedSuperEvent,
        scope: SuperScope,
        *,
        activation_at: float,
    ) -> None:
        self.begin_calls.append((event, scope, activation_at))

    def update(
        self,
        frame: CapturedFrame,
        *,
        observations: tuple[GaugeObservation, GaugeObservation],
    ) -> tuple[OutcomeDecision, ...]:
        self.update_calls.append((frame, observations))
        return self.results.get(frame.frame_id, ())

    def reset(self, reason: str) -> tuple[OutcomeDecision, ...]:
        self.reset_reasons.append(reason)
        result = self.reset_results
        self.reset_results = ()
        return result


class _RecordingAudioPort:
    def __init__(
        self,
        failures: dict[Side, AudioIssue] | None = None,
    ) -> None:
        self.failures = failures or {}
        self.batches: list[tuple[AudioAction, ...]] = []
        self.engine: object | None = None
        self.log_snapshots: list[tuple[ConfirmedSuperEvent, ...]] = []

    def execute(self, actions: tuple[AudioAction, ...]) -> None:
        assert type(actions) is tuple and actions
        assert all(isinstance(action, AudioAction) for action in actions)
        sides = {action.side for action in actions}
        assert len(sides) == 1
        side = next(iter(sides))
        self.batches.append(actions)
        if self.engine is not None:
            self.log_snapshots.append(self.engine.event_log)
        issue = self.failures.get(side)
        if issue is not None:
            raise AudioPlaybackError("execute", side, issue)


@dataclass
class _Harness:
    engine: object
    profile: HudProfile
    p1_references: MeterReferences
    p2_references: MeterReferences
    p1_tracker: object
    p2_tracker: object
    p1_scheduler: object
    p2_scheduler: object
    p1_config: SideAudioConfig
    p2_config: SideAudioConfig
    gate: object
    clock: _Clock
    observer: _Observer
    outcome_observer: _OutcomeObserverSpy
    audio: _RecordingAudioPort
    diagnostics: list[AudioPlaybackError]
    diagnostic_records: list[DiagnosticRecord]


def _build_harness(
    *,
    profile: HudProfile | None = None,
    p1_references: MeterReferences | None = None,
    p2_references: MeterReferences | None = None,
    p1_tracker: object | None = None,
    p2_tracker: object | None = None,
    p1_scheduler: object | None = None,
    p2_scheduler: object | None = None,
    p1_config: SideAudioConfig | None = None,
    p2_config: SideAudioConfig | None = None,
    gate: object | None = None,
    clock: _Clock | None = None,
    observer: _Observer | None = None,
    outcome_observer: _OutcomeObserverSpy | None = None,
    audio: _RecordingAudioPort | None = None,
    diagnostic_sink: Callable[[DiagnosticRecord], None] | None = None,
    lifecycle_history_capacity: int = 1024,
    use_default_outcome_observer: bool = False,
    viewport_resolver: object | None = None,
) -> _Harness:
    from sf6_music_trigger.engine import VisualTriggerEngine

    actual_profile = profile or _profile()
    actual_p1_references = p1_references or _references(Side.P1)
    actual_p2_references = p2_references or _references(Side.P2)
    actual_p1_tracker = p1_tracker or GaugeTracker(
        Side.P1,
        actual_profile.profile_id,
        GaugeTrackerConfig(),
    )
    actual_p2_tracker = p2_tracker or GaugeTracker(
        Side.P2,
        actual_profile.profile_id,
        GaugeTrackerConfig(),
    )
    actual_p1_scheduler = p1_scheduler or AudioScheduler(Side.P1)
    actual_p2_scheduler = p2_scheduler or AudioScheduler(Side.P2)
    actual_p1_config = p1_config or SideAudioConfig(
        enabled=True,
        main_track="C:\\music\\p1.wav",
        volume=0.2,
    )
    actual_p2_config = p2_config or SideAudioConfig(
        enabled=True,
        main_track="C:\\music\\p2.wav",
        volume=0.3,
    )
    actual_gate = gate or FreshFrameGate()
    actual_clock = clock or _Clock()
    actual_observer = observer or _Observer()
    actual_outcome_observer = (
        None if use_default_outcome_observer
        else outcome_observer or _OutcomeObserverSpy()
    )
    actual_audio = audio or _RecordingAudioPort()
    diagnostics: list[AudioPlaybackError] = []
    diagnostic_records: list[DiagnosticRecord] = []
    actual_diagnostic_sink = diagnostic_sink or diagnostic_records.append
    engine = VisualTriggerEngine(
        profile=actual_profile,
        p1_references=actual_p1_references,
        p2_references=actual_p2_references,
        p1_tracker=actual_p1_tracker,
        p2_tracker=actual_p2_tracker,
        freshness_gate=actual_gate,
        p1_scheduler=actual_p1_scheduler,
        p2_scheduler=actual_p2_scheduler,
        p1_audio_config=actual_p1_config,
        p2_audio_config=actual_p2_config,
        audio_port=actual_audio,
        audio_diagnostic=diagnostics.append,
        diagnostic_sink=actual_diagnostic_sink,
        clock=actual_clock,
        observer=actual_observer,
        outcome_observer=actual_outcome_observer,
        lifecycle_history_capacity=lifecycle_history_capacity,
        viewport_resolver=viewport_resolver,
    )
    actual_audio.engine = engine
    actual_outcome_observer = engine._outcome_observer
    return _Harness(
        engine,
        actual_profile,
        actual_p1_references,
        actual_p2_references,
        actual_p1_tracker,
        actual_p2_tracker,
        actual_p1_scheduler,
        actual_p2_scheduler,
        actual_p1_config,
        actual_p2_config,
        actual_gate,
        actual_clock,
        actual_observer,
        actual_outcome_observer,
        actual_audio,
        diagnostics,
        diagnostic_records,
    )


def _drive(
    harness: _Harness,
    rows: list[tuple[float, str, str]],
    *,
    first_frame_id: int = 0,
) -> list[object]:
    updates: list[object] = []
    for offset, (sampled_at, p1_kind, p2_kind) in enumerate(rows):
        frame_id = first_frame_id + offset
        harness.observer.kinds[(frame_id, Side.P1)] = p1_kind
        harness.observer.kinds[(frame_id, Side.P2)] = p2_kind
        harness.clock.value = sampled_at
        updates.append(harness.engine.process(_running(_frame(frame_id, sampled_at))))
    return updates


def test_candidate_starts_once_and_confirmation_does_not_restart() -> None:
    candidate = _candidate(side=Side.P1, candidate_id="cand-1")
    confirmed = _event(
        side=Side.P1,
        event_id="evt-1",
        candidate_id="cand-1",
        confirmed_at=1.6,
        levels_spent=3,
    )
    p1_tracker = _TrackerSpy([
        GaugeTrackerUpdate(candidate),
        GaugeTrackerUpdate(confirmed),
    ])
    harness = _build_harness(p1_tracker=p1_tracker)

    _drive(harness, [
        (1.0, "level_3", "level_unknown"),
        (1.6, "level_0", "level_unknown"),
    ])

    assert len(harness.audio.batches) == 1
    assert [item.kind for item in harness.audio.batches[0]] == [
        AudioKind.STOP,
        AudioKind.PLAY,
    ]
    assert harness.outcome_observer.begin_calls == [
        (confirmed, harness.p1_config.super_scope, candidate.suspected_at)
    ]


@pytest.mark.parametrize("label,evidence,expected_actions", [
    (OutcomeLabel.HIT, "opponent_health_changed", ()),
    (OutcomeLabel.UNKNOWN, "contact_not_established", ()),
])
def test_engine_emits_one_closed_record_for_candidate_confirmation_and_outcome(
    label: OutcomeLabel, evidence: str, expected_actions: tuple[AudioKind, ...],
) -> None:
    candidate = _candidate(candidate_id="cand-trace")
    confirmed = _event(
        event_id="evt-trace",
        candidate_id=candidate.candidate_id,
        confirmed_at=1.5,
        levels_spent=3,
    )
    outcome = _OutcomeObserverSpy()
    outcome.results[2] = (
        OutcomeDecision(
            event_id=confirmed.event_id,
            side=Side.P1,
            label=label,
            confidence=0.9,
            decided_at=2.0,
            evidence=(evidence,),
        ),
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy(
            [GaugeTrackerUpdate(candidate), GaugeTrackerUpdate(confirmed)]
        ),
        outcome_observer=outcome,
    )

    _drive(
        harness,
        [
            (1.0, "level_3", "level_unknown"),
            (1.5, "level_0", "level_unknown"),
            (2.0, "level_0", "level_unknown"),
        ],
    )

    assert [
        (
            record.transition,
            record.side,
            record.identity,
            record.audio_actions,
            record.outcome,
        )
        for record in harness.diagnostic_records
    ] == [
        ("CANDIDATE", Side.P1, "cand-trace", (AudioKind.STOP, AudioKind.PLAY), None),
        ("CONFIRMED", Side.P1, "cand-trace", (), None),
        ("OUTCOME", Side.P1, "cand-trace", expected_actions, label),
    ]
    assert harness.diagnostic_records[1].levels_spent == 3
    assert harness.diagnostic_records[2].evidence == (evidence,)
    assert harness.engine.diagnostic_warning is False


@pytest.mark.parametrize("side", list(Side))
def test_timeout_then_late_authority_cannot_restart_the_same_audio(side: Side) -> None:
    harness = _build_harness()
    tracker = GaugeTracker(side, "late-authority", GaugeTrackerConfig())
    # Real tracker -> real engine routing/scheduler -> recording audio port.
    from test_gauge_tracker import _level_at, _unknown_level_at, _candidate_conflict_at
    from dataclasses import replace
    observations = [_level_at(i, 0.0, 3) for i in range(10)]
    observations += [_candidate_conflict_at(10+i, .05+i*.05, level=0) for i in range(3)]
    observations += [_unknown_level_at(13, .91)]
    observations += [_candidate_conflict_at(14+i, .96+i*.05, level=0) for i in range(24)]
    for observation in observations:
        event = tracker.update(replace(observation, side=side)).transition
        if isinstance(event, SuperCandidateEvent):
            harness.engine._route_candidate(event)
        elif isinstance(event, CancelledSuperCandidate):
            harness.engine._route_cancellation(event)
        elif isinstance(event, ConfirmedSuperEvent):
            harness.engine._route_confirmation(event)
    kinds = [action.kind for batch in harness.audio.batches for action in batch]
    assert kinds.count(AudioKind.PLAY) == 1
    assert kinds.count(AudioKind.FADE) == 1
    assert len(harness.engine.candidate_log) == 1
    assert len(harness.engine.cancellation_log) == 1


def test_engine_emits_one_closed_record_for_candidate_cancellation() -> None:
    candidate = _candidate(candidate_id="cand-cancel-trace")
    cancellation = _cancellation(candidate_id=candidate.candidate_id)
    harness = _build_harness(
        p1_tracker=_TrackerSpy(
            [GaugeTrackerUpdate(candidate), GaugeTrackerUpdate(cancellation)]
        )
    )

    _drive(
        harness,
        [
            (1.0, "level_3", "level_unknown"),
            (1.5, "level_3", "level_unknown"),
        ],
    )

    assert [
        (
            record.transition,
            record.side,
            record.identity,
            record.evidence,
            record.audio_actions,
        )
        for record in harness.diagnostic_records
    ] == [
        (
            "CANDIDATE",
            Side.P1,
            "cand-cancel-trace",
            ("stable_downward_level",),
            (AudioKind.STOP, AudioKind.PLAY),
        ),
        (
            "CANCELLED",
            Side.P1,
            "cand-cancel-trace",
            ("CONFIRMATION_TIMEOUT",),
            (AudioKind.FADE,),
        ),
    ]


def test_diagnostic_sink_failure_latches_warning_without_changing_gameplay() -> None:
    candidate = _candidate(candidate_id="cand-sink-failure")

    def fail_sink(record: DiagnosticRecord) -> None:
        del record
        raise RuntimeError(r"C:\private\audio.wav")

    harness = _build_harness(
        p1_tracker=_TrackerSpy([GaugeTrackerUpdate(candidate)]),
        diagnostic_sink=fail_sink,
    )

    update = _drive(harness, [(1.0, "level_3", "level_unknown")])[0]

    assert update.candidates == (candidate,)
    assert harness.engine.candidate_log == (candidate,)
    assert [[action.kind for action in batch] for batch in harness.audio.batches] == [
        [AudioKind.STOP, AudioKind.PLAY]
    ]
    assert harness.engine.diagnostic_warning is True


def test_sa1_candidate_stays_silent_on_filtered_confirmation_without_outcome() -> None:
    candidate = _candidate(candidate_id="cand-sa1", levels_suspected=1)
    confirmed = _event(
        event_id="evt-sa1",
        candidate_id=candidate.candidate_id,
        confirmed_at=1.4,
        levels_spent=1,
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy([
            GaugeTrackerUpdate(candidate),
            GaugeTrackerUpdate(confirmed),
        ]),
        p1_config=SideAudioConfig(
            enabled=True,
            main_track="C:\\music\\p1.wav",
            super_scope=SuperScope.SA3_CA_ONLY,
        ),
    )

    updates = _drive(harness, [
        (1.0, "level_1", "level_unknown"),
        (1.4, "level_0", "level_unknown"),
    ])

    assert harness.audio.batches == []
    assert updates[0].candidates == (candidate,)
    assert updates[1].events == (confirmed,)
    assert harness.outcome_observer.begin_calls == []


def test_candidate_cancellation_fades_without_beginning_outcome() -> None:
    candidate = _candidate(candidate_id="cand-cancel")
    cancellation = _cancellation(candidate_id=candidate.candidate_id)
    harness = _build_harness(
        p1_tracker=_TrackerSpy([
            GaugeTrackerUpdate(candidate),
            GaugeTrackerUpdate(cancellation),
        ])
    )

    updates = _drive(harness, [
        (1.0, "level_3", "level_unknown"),
        (1.5, "level_3", "level_unknown"),
    ])

    assert [[action.kind for action in batch] for batch in harness.audio.batches] == [
        [AudioKind.STOP, AudioKind.PLAY],
        [AudioKind.FADE],
    ]
    assert updates[1].cancellations == (cancellation,)
    assert harness.engine.cancellation_log == (cancellation,)
    assert harness.engine.event_log == ()
    assert harness.outcome_observer.begin_calls == []


def test_confirmation_without_candidate_starts_accepted_audio_once() -> None:
    confirmed = _event(
        event_id="evt-recovery",
        candidate_id="cand-missing",
        confirmed_at=1.6,
        levels_spent=3,
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy([
            GaugeTrackerUpdate(confirmed),
            GaugeTrackerUpdate(confirmed),
        ])
    )

    updates = _drive(harness, [
        (1.6, "level_0", "level_unknown"),
        (1.7, "level_0", "level_unknown"),
    ])

    assert len(harness.audio.batches) == 1
    assert [action.kind for action in harness.audio.batches[0]] == [
        AudioKind.STOP,
        AudioKind.PLAY,
    ]
    assert updates[0].events == (confirmed,)
    assert updates[1].events == ()
    assert harness.outcome_observer.begin_calls == [
        (confirmed, harness.p1_config.super_scope, confirmed.confirmed_at)
    ]


def test_duplicate_candidate_and_terminal_ids_do_not_repeat_audio() -> None:
    candidate = _candidate(candidate_id="cand-duplicate")
    confirmed = _event(
        event_id="evt-duplicate",
        candidate_id=candidate.candidate_id,
        confirmed_at=1.6,
        levels_spent=3,
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy([
            GaugeTrackerUpdate(candidate),
            GaugeTrackerUpdate(candidate),
            GaugeTrackerUpdate(confirmed),
            GaugeTrackerUpdate(confirmed),
        ])
    )

    _drive(harness, [
        (1.0, "level_3", "level_unknown"),
        (1.1, "level_3", "level_unknown"),
        (1.6, "level_0", "level_unknown"),
        (1.7, "level_0", "level_unknown"),
    ])

    assert len(harness.audio.batches) == 1
    assert harness.engine.candidate_log == (candidate,)
    assert harness.engine.event_log == (confirmed,)


def test_duplicate_cancellation_id_does_not_repeat_terminal_fade() -> None:
    candidate = _candidate(candidate_id="cand-cancel-duplicate")
    cancellation = _cancellation(candidate_id=candidate.candidate_id)
    harness = _build_harness(
        p1_tracker=_TrackerSpy([
            GaugeTrackerUpdate(candidate),
            GaugeTrackerUpdate(cancellation),
            GaugeTrackerUpdate(cancellation),
        ])
    )

    updates = _drive(harness, [
        (1.0, "level_3", "level_unknown"),
        (1.5, "level_3", "level_unknown"),
        (1.6, "level_3", "level_unknown"),
    ])

    assert len(harness.audio.batches) == 2
    assert updates[1].cancellations == (cancellation,)
    assert updates[2].cancellations == ()
    assert harness.engine.cancellation_log == (cancellation,)


def test_simultaneous_side_candidates_and_confirmations_remain_independent() -> None:
    p1_candidate = _candidate(Side.P1, candidate_id="cand-shared")
    p2_candidate = _candidate(Side.P2, candidate_id="cand-shared")
    p1_event = _event(
        Side.P1,
        event_id="evt-p1",
        candidate_id=p1_candidate.candidate_id,
        confirmed_at=1.6,
        levels_spent=3,
    )
    p2_event = _event(
        Side.P2,
        event_id="evt-p2",
        candidate_id=p2_candidate.candidate_id,
        confirmed_at=1.6,
        levels_spent=3,
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy([
            GaugeTrackerUpdate(p1_candidate),
            GaugeTrackerUpdate(p1_event),
        ]),
        p2_tracker=_TrackerSpy([
            GaugeTrackerUpdate(p2_candidate),
            GaugeTrackerUpdate(p2_event),
        ]),
    )

    updates = _drive(harness, [
        (1.0, "level_3", "level_3"),
        (1.6, "level_0", "level_0"),
    ])

    assert updates[0].candidates == (p1_candidate, p2_candidate)
    assert updates[1].events == (p1_event, p2_event)
    assert [batch[0].side for batch in harness.audio.batches] == [Side.P1, Side.P2]
    assert harness.outcome_observer.begin_calls == [
        (p1_event, harness.p1_config.super_scope, p1_candidate.suspected_at),
        (p2_event, harness.p2_config.super_scope, p2_candidate.suspected_at),
    ]


def test_same_frame_same_event_id_remains_side_local_through_outcome_terminal() -> None:
    shared_id = "shared-event"
    p1_event = _event(Side.P1, event_id=shared_id, levels_spent=3)
    p2_event = _event(Side.P2, event_id=shared_id, levels_spent=3)
    outcome = _OutcomeObserverSpy()
    outcome.results[1] = (
        OutcomeDecision(
            shared_id,
            Side.P1,
            OutcomeLabel.HIT,
            0.9,
            2.0,
            ("opponent_health_changed",),
        ),
        OutcomeDecision(
            shared_id,
            Side.P2,
            OutcomeLabel.BLOCK,
            0.9,
            2.0,
            ("training_result_changed",),
        ),
    )
    p1_scheduler = _SchedulerSpy(
        Side.P1,
        (
            AudioAction(AudioKind.STOP, Side.P1),
            AudioAction(AudioKind.PLAY, Side.P1, "C:\\music\\p1.wav", 0.2),
        ),
    )
    p2_scheduler = _SchedulerSpy(
        Side.P2,
        (
            AudioAction(AudioKind.STOP, Side.P2),
            AudioAction(AudioKind.PLAY, Side.P2, "C:\\music\\p2.wav", 0.3),
        ),
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy([GaugeTrackerUpdate(p1_event)]),
        p2_tracker=_TrackerSpy([GaugeTrackerUpdate(p2_event)]),
        p1_scheduler=p1_scheduler,
        p2_scheduler=p2_scheduler,
        outcome_observer=outcome,
    )

    updates = _drive(harness, [
        (1.0, "level_0", "level_0"),
        (2.0, "level_0", "level_0"),
    ])

    assert updates[0].events == (p1_event, p2_event)
    assert [batch[0].side for batch in harness.audio.batches[:2]] == [
        Side.P1,
        Side.P2,
    ]
    assert harness.outcome_observer.begin_calls == [
        (p1_event, harness.p1_config.super_scope, p1_event.confirmed_at),
        (p2_event, harness.p2_config.super_scope, p2_event.confirmed_at),
    ]
    assert [call[0].side for call in harness.p1_scheduler.outcome_calls] == [Side.P1]
    assert [call[0].side for call in harness.p2_scheduler.outcome_calls] == [Side.P2]
    assert harness.engine._event_context_by_key == {}


def test_engine_lifecycle_histories_are_bounded_and_terminal_keeps_dedupe() -> None:
    harness = _build_harness(lifecycle_history_capacity=3)
    last_event: ConfirmedSuperEvent | None = None
    for index in range(6):
        event = _event(
            Side.P1,
            event_id=f"event-{index}",
            confirmed_at=1.0 + index,
            levels_spent=3,
        )
        assert harness.engine._route_confirmation(event) is True
        assert (Side.P1, event.event_id) in harness.engine._event_context_by_key
        harness.engine._route_outcomes((OutcomeDecision(
            event.event_id,
            Side.P1,
            OutcomeLabel.HIT,
            0.9,
            event.confirmed_at + 0.1,
            ("opponent_health_changed",),
        ),))
        assert (Side.P1, event.event_id) not in harness.engine._event_context_by_key
        last_event = event

    assert last_event is not None
    assert len(harness.engine._executed_event_keys) == 3
    assert (Side.P1, "event-0") not in harness.engine._executed_event_keys
    assert harness.engine._route_confirmation(last_event) is False

    for index in range(6):
        candidate = _candidate(Side.P1, candidate_id=f"candidate-{index}")
        assert harness.engine._route_candidate(candidate) is True
        assert harness.engine._route_cancellation(_cancellation(
            Side.P1,
            candidate_id=candidate.candidate_id,
        )) is True
    assert len(harness.engine._terminal_candidate_keys) == 3
    assert len(harness.engine._started_candidate_keys) <= 3


def test_reset_routes_candidate_cancellation_before_outcome_reset() -> None:
    candidate = _candidate(candidate_id="cand-reset")
    cancellation = _cancellation(
        candidate_id=candidate.candidate_id,
        reason=CandidateCancelReason.CAPTURE_RESET,
    )
    tracker = _TrackerSpy(
        [GaugeTrackerUpdate(candidate)],
        reset_returns=[GaugeTrackerUpdate(cancellation)],
    )
    outcome = _OutcomeObserverSpy()
    outcome.reset_results = (
        OutcomeDecision(
            "evt-other-side",
            Side.P2,
            OutcomeLabel.RESET,
            0.0,
            1.5,
            ("reset",),
        ),
    )
    harness = _build_harness(
        p1_tracker=tracker,
        p2_tracker=_TrackerSpy(),
        outcome_observer=outcome,
    )
    _drive(harness, [(1.0, "level_3", "level_unknown")])

    harness.engine.reset("capture session ended")

    assert [[action.kind for action in batch] for batch in harness.audio.batches] == [
        [AudioKind.STOP, AudioKind.PLAY],
        [AudioKind.FADE],
    ]
    assert [batch[0].side for batch in harness.audio.batches] == [
        Side.P1,
        Side.P1,
    ]
    assert harness.engine.cancellation_log == (cancellation,)
    assert outcome.reset_reasons == ["capture session ended"]


def test_runtime_reset_returns_and_routes_tracker_cancellation() -> None:
    candidate = _candidate(candidate_id="cand-runtime-reset")
    cancellation = _cancellation(
        candidate_id=candidate.candidate_id,
        reason=CandidateCancelReason.CAPTURE_RESET,
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy(
            [GaugeTrackerUpdate(candidate)],
            reset_returns=[GaugeTrackerUpdate(cancellation)],
        ),
        p2_tracker=_TrackerSpy(),
    )
    _drive(harness, [(1.0, "level_3", "level_unknown")])

    update = harness.engine.process(CaptureSample(status=CaptureStatus.OCCLUDED))

    assert update.cancellations == (cancellation,)
    assert [[action.kind for action in batch] for batch in harness.audio.batches] == [
        [AudioKind.STOP, AudioKind.PLAY],
        [AudioKind.FADE],
    ]
    assert harness.outcome_observer.reset_reasons == [
        "capture sample is not running"
    ]


def test_engine_preserves_numeric_baseline_through_nine_hidden_unknowns() -> None:
    harness = _build_harness()
    rows = [
        (index / 60, "level_2", "level_unknown")
        for index in range(10)
    ]
    rows.extend(
        ((index + 10) / 60, "level_unknown", "level_unknown")
        for index in range(9)
    )
    rows.extend(
        ((index + 19) / 60, "empty_level_1", "level_unknown")
        for index in range(40)
    )

    updates = _drive(harness, rows)
    events = tuple(event for update in updates for event in update.events)

    assert len(events) == 1
    assert (
        events[0].meter_before,
        events[0].meter_after,
        events[0].levels_spent,
    ) == (2, 1, 1)


def test_engine_tenth_hidden_unknown_clears_numeric_baseline_without_event() -> None:
    harness = _build_harness()
    rows = [
        (index / 60, "level_2", "level_unknown")
        for index in range(3)
    ]
    rows.extend(
        ((index + 10) / 60, "level_unknown", "level_unknown")
        for index in range(10)
    )
    rows.extend(
        ((index + 20) / 60, "level_1", "level_unknown")
        for index in range(3)
    )

    updates = _drive(harness, rows)

    assert tuple(event for update in updates for event in update.events) == ()
    assert harness.engine.event_log == ()


def test_engine_logs_lower_tier_but_skips_audio_and_outcome_tracking() -> None:
    event = _event(Side.P1, levels_spent=1)
    p1_tracker = _TrackerSpy(returns=[GaugeTrackerUpdate(event)])
    p1_scheduler = _SchedulerSpy(Side.P1, ())
    harness = _build_harness(
        p1_tracker=p1_tracker,
        p1_scheduler=p1_scheduler,
        p1_config=SideAudioConfig(
            enabled=True,
            main_track="C:\\music\\p1.wav",
            super_scope=SuperScope.SA3_CA_ONLY,
        ),
    )

    update = _drive(harness, [(0.0, "full", "full")])[0]

    assert update.events == (event,)
    assert harness.engine.event_log == (event,)
    assert p1_scheduler.confirmation_calls == [
        (event, harness.p1_config, False)
    ]
    assert harness.audio.batches == []
    assert harness.outcome_observer.begin_calls == []
    assert harness.engine.outcome_statuses == ("低等级已过滤", None)


@pytest.mark.parametrize(
    ("levels_spent", "scope"),
    [(3, SuperScope.SA3_CA_ONLY), (1, SuperScope.ALL_SUPERS)],
)
def test_engine_keeps_immediate_play_and_outcome_for_accepted_tier(
    levels_spent: int,
    scope: SuperScope,
) -> None:
    event = _event(Side.P1, levels_spent=levels_spent)
    harness = _build_harness(
        p1_tracker=_TrackerSpy(returns=[GaugeTrackerUpdate(event)]),
        p1_config=SideAudioConfig(
            enabled=True,
            main_track="C:\\music\\p1.wav",
            super_scope=scope,
        ),
    )

    update = _drive(harness, [(0.0, "full", "full")])[0]

    assert update.events == (event,)
    assert len(harness.audio.batches) == 1
    if levels_spent == 3:
        assert harness.outcome_observer.begin_calls == [
            (event, scope, event.confirmed_at)
        ]
        assert harness.engine.outcome_statuses == ("等待大招结果", None)
    else:
        assert harness.outcome_observer.begin_calls == []
        assert harness.engine.outcome_statuses == ("等待淡出", None)


def test_disabled_lower_tier_is_still_filtered_from_outcome_observation() -> None:
    event = _event(Side.P1, levels_spent=1)
    harness = _build_harness(
        p1_tracker=_TrackerSpy(returns=[GaugeTrackerUpdate(event)]),
        p1_config=SideAudioConfig(
            enabled=False,
            main_track="",
            super_scope=SuperScope.SA3_CA_ONLY,
        ),
    )

    _drive(harness, [(0.0, "full", "full")])

    assert harness.engine.outcome_statuses == ("低等级已过滤", None)
    assert harness.outcome_observer.begin_calls == []


def _spend_rows(
    side: Side,
    *,
    start: float = 0.0,
) -> list[tuple[float, str, str]]:
    rows: list[tuple[float, str, str]] = []
    for index in range(5):
        kinds = ("full", "hidden") if side is Side.P1 else ("hidden", "full")
        rows.append((start + index * 0.05, *kinds))
    for index in range(3):
        kinds = ("spent", "hidden") if side is Side.P1 else ("hidden", "spent")
        rows.append((start + 0.30 + index * 0.06, *kinds))
    return rows


def test_constructor_is_lazy_and_rejects_unsafe_side_wiring() -> None:
    clock = _Clock()
    harness = _build_harness(clock=clock)
    assert clock.calls == []

    with pytest.raises(ValueError):
        _build_harness(
            p1_references=harness.p1_references,
            p2_references=harness.p1_references,
        )
    with pytest.raises(ValueError):
        _build_harness(p2_references=_references(Side.P1))
    with pytest.raises(ValueError):
        _build_harness(p1_references=_references(Side.P2))
    with pytest.raises(ValueError):
        _build_harness(p1_scheduler=AudioScheduler(Side.P2))

    shared_roi = NormalizedRoi(0.1, 0.1, 0.4, 0.3)
    with pytest.raises(ValueError):
        _build_harness(profile=_profile(p1_roi=shared_roi, p2_roi=shared_roi))


def test_engine_default_clock_matches_dxcam_perf_counter_timestamps() -> None:
    """DXcam timestamps use perf_counter(), so freshness must share that epoch."""
    from sf6_music_trigger.engine import VisualTriggerEngine

    parameter = inspect.signature(VisualTriggerEngine).parameters["clock"]
    assert parameter.default is time.perf_counter


def test_each_accepted_frame_wires_distinct_side_roi_reference_and_image() -> None:
    p1_tracker = _TrackerSpy()
    p2_tracker = _TrackerSpy()
    harness = _build_harness(p1_tracker=p1_tracker, p2_tracker=p2_tracker)
    frame = _frame(7, 1.25)
    harness.clock.value = 1.25
    harness.observer.kinds[(7, Side.P1)] = "full"
    harness.observer.kinds[(7, Side.P2)] = "full"

    update = harness.engine.process(_running(frame))

    assert update.events == ()
    assert harness.clock.calls == [1.25]
    assert harness.observer.frame_args == [frame.image, frame.image]
    assert harness.observer.calls == [
        {
            "side": Side.P1,
            "frame_id": 7,
            "roi": harness.profile.p1.meter_roi,
            "references": harness.p1_references,
            "sampled_at": 1.25,
        },
        {
            "side": Side.P2,
            "frame_id": 7,
            "roi": harness.profile.p2.meter_roi,
            "references": harness.p2_references,
            "sampled_at": 1.25,
        },
    ]
    assert p1_tracker.observations == [_observation(Side.P1, 7, 1.25, "full")]
    assert p2_tracker.observations == [_observation(Side.P2, 7, 1.25, "full")]


@pytest.mark.parametrize("side", [Side.P1, Side.P2])
def test_real_tracker_full_to_spent_emits_one_event_and_one_stop_play_batch(
    side: Side,
) -> None:
    harness = _build_harness()
    updates = _drive(harness, _spend_rows(side))

    assert all(type(update.events) is tuple for update in updates)
    assert all(update.events == () for update in updates[:-1])
    assert [event.side for event in updates[-1].events] == [side]
    assert [event.side for event in harness.engine.event_log] == [side]
    assert len(harness.audio.batches) == 1
    assert [action.kind for action in harness.audio.batches[0]] == [
        AudioKind.STOP,
        AudioKind.PLAY,
    ]
    assert {action.side for action in harness.audio.batches[0]} == {side}


def test_p2_different_roi_ignores_p1_full_distractor() -> None:
    harness = _build_harness()
    rows = [
        (sampled_at, "full", p2_kind)
        for sampled_at, _, p2_kind in _spend_rows(Side.P2)
    ]
    update = _drive(harness, rows)[-1]
    assert [event.side for event in update.events] == [Side.P2]
    assert [{action.side for action in batch} for batch in harness.audio.batches] == [
        {Side.P2}
    ]


def test_simultaneous_events_and_audio_batches_are_p1_then_p2() -> None:
    harness = _build_harness()
    rows = [(index * 0.05, "full", "full") for index in range(5)]
    rows += [(0.30 + index * 0.06, "spent", "spent") for index in range(3)]

    update = _drive(harness, rows)[-1]

    assert [event.side for event in update.events] == [Side.P1, Side.P2]
    assert [event.side for event in harness.engine.event_log] == [Side.P1, Side.P2]
    assert [batch[0].side for batch in harness.audio.batches] == [Side.P1, Side.P2]
    assert all(type(batch) is tuple and len(batch) == 2 for batch in harness.audio.batches)


def test_one_unverified_spent_meter_can_confirm_while_other_meter_is_verified() -> None:
    harness = _build_harness()
    rows = [(index * 0.05, "full", "full") for index in range(5)]
    rows += [
        (0.30 + index * 0.06, "unverified_spent", "full")
        for index in range(3)
    ]

    update = _drive(harness, rows)[-1]

    assert [event.side for event in update.events] == [Side.P1]
    assert [event.side for event in harness.engine.event_log] == [Side.P1]
    assert len(harness.audio.batches) == 1


@pytest.mark.parametrize(
    ("p1_kind", "p2_kind"),
    [
        ("unverified_spent", "unverified_spent"),
        ("unverified_spent", "hidden"),
        ("hidden", "unverified_spent"),
    ],
)
def test_no_trustworthy_meter_resets_before_tracker_updates(
    p1_kind: str,
    p2_kind: str,
) -> None:
    p1_tracker = _TrackerSpy()
    p2_tracker = _TrackerSpy()
    harness = _build_harness(p1_tracker=p1_tracker, p2_tracker=p2_tracker)
    harness.observer.kinds[(0, Side.P1)] = p1_kind
    harness.observer.kinds[(0, Side.P2)] = p2_kind

    update = harness.engine.process(_running(_frame(0, 0.0)))

    assert update.events == ()
    assert p1_tracker.observations == p2_tracker.observations == []
    assert len(p1_tracker.reset_reasons) == len(p2_tracker.reset_reasons) == 1


@pytest.mark.parametrize(
    "issue",
    [AudioIssue.BACKEND_FAILURE, AudioIssue.TRACK_REJECTED],
)
def test_p1_audio_failure_is_diagnostic_and_does_not_rollback_or_block_p2(
    issue: AudioIssue,
) -> None:
    audio = _RecordingAudioPort({Side.P1: issue})
    harness = _build_harness(audio=audio)
    rows = [(index * 0.05, "full", "full") for index in range(5)]
    rows += [(0.30 + index * 0.06, "spent", "spent") for index in range(3)]

    update = _drive(harness, rows)[-1]

    assert [event.side for event in update.events] == [Side.P1, Side.P2]
    assert [event.side for event in harness.engine.event_log] == [Side.P1, Side.P2]
    assert [batch[0].side for batch in audio.batches] == [Side.P1, Side.P2]
    assert [error.side for error in harness.diagnostics] == [Side.P1]
    assert harness.diagnostics[0].issue is issue
    assert [event.side for event in audio.log_snapshots[0]] == [Side.P1]
    assert [event.side for event in audio.log_snapshots[1]] == [Side.P1, Side.P2]


def test_disabled_event_is_logged_but_empty_batch_skips_audio_port() -> None:
    event = _event(Side.P1)
    p1_tracker = _TrackerSpy(returns=[GaugeTrackerUpdate(event)])
    scheduler = _SchedulerSpy(Side.P1, ())
    harness = _build_harness(
        p1_tracker=p1_tracker,
        p2_tracker=_TrackerSpy(),
        p1_scheduler=scheduler,
    )
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"

    update = harness.engine.process(_running(_frame(0, 0.0)))

    assert update.events == (event,)
    assert harness.engine.event_log == (event,)
    assert scheduler.confirmation_calls == [(event, harness.p1_config, False)]
    assert harness.audio.batches == []


def test_engine_remembers_frame_before_tracker_can_emit_event_and_uses_scope() -> None:
    event = _event(Side.P1)
    harness = _build_harness(
        p1_tracker=_TrackerSpy(returns=[GaugeTrackerUpdate(event)]),
        p2_tracker=_TrackerSpy(),
    )
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"
    frame = _frame(0, 0.0)

    harness.engine.process(_running(frame))

    assert harness.outcome_observer.remember_calls == [frame]
    assert harness.outcome_observer.begin_calls == [
        (event, harness.p1_config.super_scope, event.confirmed_at)
    ]
    assert harness.outcome_observer.update_calls == [
        (
            frame,
            (
                _observation(Side.P1, 0, 0.0, "full"),
                _observation(Side.P2, 0, 0.0, "full"),
            ),
        )
    ]


@pytest.mark.parametrize("scope", [SuperScope.ALL_SUPERS, SuperScope.SA3_CA_ONLY])
def test_both_scopes_play_immediately_at_existing_confirmation_boundary(
    scope: SuperScope,
) -> None:
    event = _event(Side.P1)
    harness = _build_harness(
        p1_tracker=_TrackerSpy(returns=[GaugeTrackerUpdate(event)]),
        p2_tracker=_TrackerSpy(),
        p1_config=SideAudioConfig(
            enabled=True,
            main_track="C:\\music\\p1.wav",
            super_scope=scope,
        ),
    )
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"

    update = harness.engine.process(_running(_frame(0, 0.0)))

    assert update.events == (event,)
    assert [[item.kind for item in batch] for batch in harness.audio.batches] == [
        [AudioKind.STOP, AudioKind.PLAY]
    ]
    assert harness.outcome_observer.begin_calls == [
        (event, scope, event.confirmed_at)
    ]
    assert harness.engine.outcome_statuses == ("等待大招结果", None)


def test_duplicate_event_id_cannot_execute_second_immediate_batch() -> None:
    event = _event(Side.P1)
    harness = _build_harness(
        p1_tracker=_TrackerSpy(
            returns=[GaugeTrackerUpdate(event), GaugeTrackerUpdate(event)]
        ),
        p2_tracker=_TrackerSpy(),
    )
    for frame_id in (0, 1):
        harness.observer.kinds[(frame_id, Side.P1)] = "full"
        harness.observer.kinds[(frame_id, Side.P2)] = "full"
        harness.clock.value = frame_id * 0.1
        harness.engine.process(_running(_frame(frame_id, frame_id * 0.1)))

    assert harness.engine.event_log == (event,)
    assert len(harness.audio.batches) == 1


def test_whiff_decision_after_confirmed_event_routes_default_fade_once() -> None:
    event = _event(Side.P1)
    decision = OutcomeDecision(
        event.event_id,
        Side.P1,
        OutcomeLabel.WHIFF,
        0.8,
        6.0,
        ("contact_window_elapsed",),
    )
    outcome = _OutcomeObserverSpy()
    outcome.results[1] = (decision,)
    harness = _build_harness(
        p1_tracker=_TrackerSpy(
            returns=[GaugeTrackerUpdate(event), GaugeTrackerUpdate()]
        ),
        p2_tracker=_TrackerSpy(),
        outcome_observer=outcome,
    )
    for frame_id in (0, 1):
        harness.observer.kinds[(frame_id, Side.P1)] = "full"
        harness.observer.kinds[(frame_id, Side.P2)] = "full"

    harness.clock.value = 0.0
    harness.engine.process(_running(_frame(0, 0.0)))
    harness.clock.value = 6.0
    harness.engine.process(_running(_frame(1, 6.0)))

    assert [[item.kind for item in batch] for batch in harness.audio.batches] == [
        [AudioKind.STOP, AudioKind.PLAY],
        [AudioKind.FADE],
    ]
    assert harness.audio.batches[1][0].fade_ms == 1500
    assert [
        (record.transition, record.identity, record.outcome, record.evidence)
        for record in harness.diagnostic_records
    ] == [
        ("CONFIRMED", "event-P1", None, ()),
        ("OUTCOME", "event-P1", OutcomeLabel.WHIFF, ("contact_window_elapsed",)),
    ]
    assert harness.engine.diagnostic_warning is False


def test_filtered_sa1_sa2_fades_once_without_whiff_sound() -> None:
    outcome = _OutcomeObserverSpy()
    outcome.results[0] = (
        OutcomeDecision(
            "event-P1", Side.P1, OutcomeLabel.FILTERED_TIER, 1.0, 2.0,
            ("remaining_meter_1",),
        ),
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy(),
        p2_tracker=_TrackerSpy(),
        outcome_observer=outcome,
        p1_config=SideAudioConfig(
            enabled=True,
            main_track="C:\\music\\p1.wav",
            play_whiff_sound=True,
            whiff_track="C:\\music\\whiff.wav",
        ),
    )
    harness.engine._route_confirmation(_event(Side.P1, levels_spent=3))
    harness.audio.batches.clear()
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"
    harness.clock.value = 2.0

    harness.engine.process(_running(_frame(0, 2.0)))

    assert [[item.kind for item in batch] for batch in harness.audio.batches] == [
        [AudioKind.FADE]
    ]
    assert harness.engine.outcome_statuses == ("低等级已淡出", None)


@pytest.mark.parametrize(
    ("label", "text"),
    [
        (OutcomeLabel.HIT, "命中"),
        (OutcomeLabel.BLOCK, "格挡"),
        (OutcomeLabel.WHIFF, "打空"),
        (OutcomeLabel.NON_HIT, "格挡/未命中"),
        (OutcomeLabel.UNKNOWN, "结果不确定"),
        (OutcomeLabel.RESET, "结果不确定"),
    ],
)
def test_outcome_status_uses_only_privacy_safe_labels(
    label: OutcomeLabel, text: str,
) -> None:
    outcome = _OutcomeObserverSpy()
    outcome.results[0] = (
        OutcomeDecision("private-path-event", Side.P1, label, 0.0, 2.0, ("safe",)),
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy(), p2_tracker=_TrackerSpy(), outcome_observer=outcome
    )
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"
    harness.clock.value = 2.0

    harness.engine.process(_running(_frame(0, 2.0)))

    assert harness.engine.outcome_statuses == (text, None)
    assert "private" not in repr(harness.engine.outcome_statuses).lower()


def test_simultaneous_p1_p2_corrections_are_routed_in_side_order() -> None:
    outcome = _OutcomeObserverSpy()
    outcome.results[0] = (
        OutcomeDecision("p1", Side.P1, OutcomeLabel.RESET, 0.0, 2.0, ("reset",)),
        OutcomeDecision("p2", Side.P2, OutcomeLabel.RESET, 0.0, 2.0, ("reset",)),
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy(), p2_tracker=_TrackerSpy(), outcome_observer=outcome
    )
    harness.engine._route_confirmation(_event(Side.P1, event_id="p1", levels_spent=3))
    harness.engine._route_confirmation(_event(Side.P2, event_id="p2", levels_spent=3))
    harness.audio.batches.clear()
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"
    harness.clock.value = 2.0

    harness.engine.process(_running(_frame(0, 2.0)))

    assert [batch[0].side for batch in harness.audio.batches] == [Side.P2]
    assert [[item.kind for item in batch] for batch in harness.audio.batches] == [
        [AudioKind.FADE]
    ]


def test_outcome_audio_failure_is_diagnostic_without_retry() -> None:
    outcome = _OutcomeObserverSpy()
    outcome.results[0] = (
        OutcomeDecision("p1", Side.P1, OutcomeLabel.RESET, 0.0, 2.0, ("reset",)),
    )
    audio = _RecordingAudioPort()
    harness = _build_harness(
        p1_tracker=_TrackerSpy(), p2_tracker=_TrackerSpy(),
        outcome_observer=outcome, audio=audio,
    )
    harness.engine._route_confirmation(_event(Side.P1, event_id="p1", levels_spent=3))
    audio.batches.clear()
    audio.failures[Side.P1] = AudioIssue.BACKEND_FAILURE
    for frame_id in (0, 1):
        harness.observer.kinds[(frame_id, Side.P1)] = "full"
        harness.observer.kinds[(frame_id, Side.P2)] = "full"
        harness.clock.value = 2.0 + frame_id * 0.1
        harness.engine.process(_running(_frame(frame_id, 2.0 + frame_id * 0.1)))

    assert len(audio.batches) == 1
    assert len(harness.diagnostics) == 1


@pytest.mark.parametrize("label", [OutcomeLabel.HIT, OutcomeLabel.BLOCK])
def test_contact_outcome_never_changes_immediate_audio(label: OutcomeLabel) -> None:
    decision = OutcomeDecision("event-P1", Side.P1, label, 0.8, 6.0, ("safe",))
    outcome = _OutcomeObserverSpy()
    outcome.results[0] = (decision,)
    harness = _build_harness(
        p1_tracker=_TrackerSpy(),
        p2_tracker=_TrackerSpy(),
        outcome_observer=outcome,
    )
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"

    harness.engine.process(_running(_frame(0, 6.0)))

    assert harness.audio.batches == []


def test_capture_reset_fades_pending_outcome_once() -> None:
    outcome = _OutcomeObserverSpy()
    outcome.reset_results = (
        OutcomeDecision(
            "event-P1", Side.P1, OutcomeLabel.RESET, 0.0, 1.0, ("reset",)
        ),
    )
    harness = _build_harness(outcome_observer=outcome)
    harness.engine._route_confirmation(_event(Side.P1, levels_spent=3))
    harness.audio.batches.clear()
    harness.engine.process(CaptureSample(status=CaptureStatus.OCCLUDED))

    assert outcome.reset_reasons == ["capture sample is not running"]
    assert [[item.kind for item in batch] for batch in harness.audio.batches] == [
        [AudioKind.FADE]
    ]


def test_rearmed_repeated_event_produces_a_distinct_second_batch() -> None:
    harness = _build_harness()
    first = _spend_rows(Side.P1)
    rearm = [(0.60 + index * 0.05, "full", "hidden") for index in range(5)]
    second_spend = [
        (0.90 + index * 0.06, "spent", "hidden") for index in range(3)
    ]

    _drive(harness, first)
    updates = _drive(harness, rearm + second_spend, first_frame_id=len(first))

    assert len(harness.engine.event_log) == 2
    assert len(harness.audio.batches) == 2
    assert harness.audio.batches[0] is not harness.audio.batches[1]
    assert [event.side for event in updates[-1].events] == [Side.P1]


def _nonrunning_samples() -> list[object]:
    samples: list[object] = [
        pytest.param(CaptureSample(status=CaptureStatus.OCCLUDED), id="occluded"),
        pytest.param(CaptureSample(status=CaptureStatus.MINIMIZED), id="minimized"),
        pytest.param(CaptureSample(status=CaptureStatus.STOPPED), id="stopped"),
    ]
    samples.extend(
        pytest.param(
            CaptureSample(status=CaptureStatus.LOST, issue=issue),
            id=f"lost-{issue.value}",
        )
        for issue in CaptureIssue
    )
    return samples


@pytest.mark.parametrize("sample", _nonrunning_samples())
def test_every_nonrunning_sample_resets_both_without_clock_observer_or_audio(
    sample: CaptureSample,
) -> None:
    p1_tracker = _TrackerSpy()
    p2_tracker = _TrackerSpy()
    harness = _build_harness(p1_tracker=p1_tracker, p2_tracker=p2_tracker)

    update = harness.engine.process(sample)

    assert update.events == ()
    assert len(p1_tracker.reset_reasons) == len(p2_tracker.reset_reasons) == 1
    assert p1_tracker.reset_reasons == p2_tracker.reset_reasons
    assert harness.clock.calls == []
    assert harness.observer.calls == []
    assert harness.audio.batches == []


@pytest.mark.parametrize(
    ("sampled_at", "now"),
    [(0.0, 0.251), (0.051, 0.0)],
    ids=("stale", "future"),
)
def test_stale_or_future_frame_resets_trackers_without_observation(
    sampled_at: float,
    now: float,
) -> None:
    p1_tracker = _TrackerSpy()
    p2_tracker = _TrackerSpy()
    clock = _Clock(now)
    harness = _build_harness(
        p1_tracker=p1_tracker,
        p2_tracker=p2_tracker,
        clock=clock,
    )

    update = harness.engine.process(_running(_frame(0, sampled_at)))

    assert update.events == ()
    assert clock.calls == [now]
    assert harness.observer.calls == []
    assert len(p1_tracker.reset_reasons) == len(p2_tracker.reset_reasons) == 1


def test_runtime_resets_preserve_freshness_until_explicit_reset_starts_new_epoch() -> None:
    p1_tracker = _TrackerSpy()
    p2_tracker = _TrackerSpy()
    harness = _build_harness(p1_tracker=p1_tracker, p2_tracker=p2_tracker)
    frame = _frame(7, 7.0)
    harness.clock.value = 7.0
    harness.observer.kinds[(7, Side.P1)] = "full"
    harness.observer.kinds[(7, Side.P2)] = "full"

    harness.engine.process(_running(frame))
    assert len(harness.observer.calls) == 2
    harness.engine.process(CaptureSample(status=CaptureStatus.OCCLUDED))
    harness.engine.process(_running(frame))
    assert len(harness.observer.calls) == 2

    harness.engine.reset("new capture session")
    harness.engine.process(_running(frame))
    assert len(harness.observer.calls) == 4
    assert p1_tracker.reset_reasons == p2_tracker.reset_reasons
    assert p1_tracker.reset_reasons[-1] == "new capture session"


def test_geometry_mismatch_resets_both_and_still_consumes_freshness() -> None:
    p1_tracker = _TrackerSpy()
    p2_tracker = _TrackerSpy()
    harness = _build_harness(p1_tracker=p1_tracker, p2_tracker=p2_tracker)
    mismatched = _frame(3, 3.0, geometry=ClientGeometry(0, 0, 300, 100))
    harness.clock.value = 3.0

    harness.engine.process(_running(mismatched))
    harness.engine.process(_running(mismatched))

    assert harness.observer.calls == []
    assert len(p1_tracker.reset_reasons) == len(p2_tracker.reset_reasons) == 2


def test_viewport_resolver_normalizes_frame_before_all_visual_observers() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver

    outcome = _OutcomeObserverSpy()
    harness = _build_harness(
        p1_tracker=_TrackerSpy(),
        p2_tracker=_TrackerSpy(),
        outcome_observer=outcome,
        viewport_resolver=CenteredViewportResolver(
            2.0,
            confirmation_frames=1,
            minimum_content_height=1,
        ),
    )
    frame = _frame(10, 10.0, geometry=ClientGeometry(25, 50, 300, 100))
    harness.clock.value = 10.0

    harness.engine.process(_running(frame))

    assert harness.engine.viewport_status == "normalized"
    assert [image.shape for image in harness.observer.frame_args] == [
        (100, 200, 3),
        (100, 200, 3),
    ]
    assert outcome.remember_calls[0].geometry == ClientGeometry(75, 50, 200, 100)
    assert outcome.update_calls[0][0].geometry == ClientGeometry(75, 50, 200, 100)


def test_viewport_confirmation_fails_closed_before_gauge_observation() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver

    p1_tracker = _TrackerSpy()
    p2_tracker = _TrackerSpy()
    harness = _build_harness(
        p1_tracker=p1_tracker,
        p2_tracker=p2_tracker,
        viewport_resolver=CenteredViewportResolver(
            2.0,
            confirmation_frames=3,
            minimum_content_height=1,
        ),
    )
    frame = _frame(11, 11.0, geometry=ClientGeometry(0, 0, 300, 100))
    harness.clock.value = 11.0

    harness.engine.process(_running(frame))

    assert harness.engine.viewport_status == "confirming"
    assert harness.observer.calls == []
    assert p1_tracker.reset_reasons == p2_tracker.reset_reasons == [
        "capture viewport is not ready"
    ]


def test_explicit_engine_reset_discards_viewport_confirmation() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver

    resolver = CenteredViewportResolver(
        2.0,
        confirmation_frames=2,
        minimum_content_height=1,
    )
    harness = _build_harness(
        p1_tracker=_TrackerSpy(),
        p2_tracker=_TrackerSpy(),
        viewport_resolver=resolver,
    )
    first = _frame(12, 12.0, geometry=ClientGeometry(0, 0, 300, 100))
    harness.clock.value = 12.0
    harness.engine.process(_running(first))
    harness.engine.reset("new capture session")

    second = _frame(13, 13.0, geometry=ClientGeometry(0, 0, 300, 100))
    harness.clock.value = 13.0
    harness.engine.process(_running(second))

    assert harness.engine.viewport_status == "confirming"
    assert harness.observer.calls == []


def test_both_hidden_reset_trackers_without_reusing_or_updating_observations() -> None:
    p1_tracker = _TrackerSpy()
    p2_tracker = _TrackerSpy()
    harness = _build_harness(p1_tracker=p1_tracker, p2_tracker=p2_tracker)

    update = harness.engine.process(_running(_frame(0, 0.0)))

    assert update.events == ()
    assert len(harness.observer.calls) == 2
    assert p1_tracker.observations == p2_tracker.observations == []
    assert len(p1_tracker.reset_reasons) == len(p2_tracker.reset_reasons) == 1


@pytest.mark.parametrize("kind", ["hidden", "ambiguous"])
def test_terminal_outcome_is_routed_before_hidden_or_untrusted_early_return(kind) -> None:
    outcome = _OutcomeObserverSpy()
    event = _event(Side.P1, event_id="terminal-before-return", levels_spent=3)
    outcome.results[0] = (
        OutcomeDecision(event.event_id, Side.P1, OutcomeLabel.NON_HIT, .9, 10., ()),
    )
    config = SideAudioConfig(
        main_track="audio/main.wav",
        super_scope=SuperScope.SA3_CA_ONLY,
    )
    harness = _build_harness(
        outcome_observer=outcome,
        p1_config=config,
        p2_config=config,
    )
    harness.engine._route_confirmation(event)
    harness.audio.batches.clear()
    harness.clock.value = 10.
    harness.observer.kinds[(0, Side.P1)] = kind
    harness.observer.kinds[(0, Side.P2)] = kind

    harness.engine.process(_running(_frame(0, 10.)))

    assert harness.audio.batches == [
        (AudioAction(AudioKind.FADE, Side.P1, fade_ms=1500),),
    ]
    assert harness.engine.outcome_statuses == ("格挡/未命中", None)
    assert harness.diagnostic_records[-1].transition == "OUTCOME"
    assert harness.diagnostic_records[-1].outcome is OutcomeLabel.NON_HIT
    assert harness.engine._event_context_by_key == {}


@pytest.mark.parametrize("kind", ["hidden", "ambiguous"])
def test_due_held_fade_flushes_on_hidden_or_untrusted_early_return(kind) -> None:
    config = SideAudioConfig(
        main_track="audio/main.wav",
        super_scope=SuperScope.ALL_SUPERS,
        fade_non_three=True,
    )
    harness = _build_harness(p1_config=config)
    event = _event(Side.P1, event_id="held-before-return", levels_spent=1)
    harness.clock.value = 1.
    harness.engine._route_confirmation(event)
    assert harness.engine.outcome_statuses == ("等待淡出", None)

    harness.clock.value = 5.
    harness.observer.kinds[(0, Side.P1)] = kind
    harness.observer.kinds[(0, Side.P2)] = kind
    harness.engine.process(_running(_frame(0, 5.)))

    assert harness.audio.batches[-1] == (
        AudioAction(AudioKind.FADE, Side.P1, fade_ms=1500),
    )
    assert harness.engine.outcome_statuses == ("淡出中", None)
    assert harness.diagnostic_records[-1].transition == "FADE"
    assert harness.diagnostic_records[-1].evidence == ("playback_hold_elapsed",)
    assert harness.engine._held_fades == {}


@pytest.mark.parametrize(
    ("label", "text"),
    [
        (OutcomeLabel.HIT, "命中"),
        (OutcomeLabel.NON_HIT, "格挡/未命中"),
        (OutcomeLabel.UNKNOWN, "结果不确定"),
    ],
)
def test_disabled_audio_current_event_still_reaches_terminal_status(label, text) -> None:
    config = SideAudioConfig(enabled=False, super_scope=SuperScope.SA3_CA_ONLY)
    harness = _build_harness(p1_config=config)
    event = _event(Side.P1, event_id="disabled-current", levels_spent=3)
    harness.engine._route_confirmation(event)

    harness.engine._route_outcomes((
        OutcomeDecision(event.event_id, Side.P1, label, .9, 2., ()),
    ))

    assert harness.audio.batches == []
    assert harness.engine.outcome_statuses == (text, None)
    assert harness.diagnostic_records[-1].outcome is label


def test_disabled_audio_reset_and_superseded_result_use_current_status_event() -> None:
    outcome = _OutcomeObserverSpy()
    config = SideAudioConfig(enabled=False, super_scope=SuperScope.SA3_CA_ONLY)
    harness = _build_harness(p1_config=config, outcome_observer=outcome)
    old = _event(Side.P1, event_id="disabled-old", levels_spent=3)
    new = _event(Side.P1, event_id="disabled-new", levels_spent=3)
    harness.engine._route_confirmation(old)
    harness.engine._route_confirmation(new)

    harness.engine._route_outcomes((
        OutcomeDecision(old.event_id, Side.P1, OutcomeLabel.HIT, .9, 2., ()),
    ))
    assert harness.engine.outcome_statuses == ("等待大招结果", None)

    outcome.reset_results = (
        OutcomeDecision(new.event_id, Side.P1, OutcomeLabel.RESET, 0., 3., ("reset",)),
    )
    harness.engine.reset("test reset")
    assert harness.engine.outcome_statuses == ("结果不确定", None)


def test_non_tuple_tracker_result_fails_before_event_log_or_audio() -> None:
    p1_tracker = _TrackerSpy(returns=[[_event(Side.P1)]])
    harness = _build_harness(p1_tracker=p1_tracker, p2_tracker=_TrackerSpy())
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"

    with pytest.raises(TypeError):
        harness.engine.process(_running(_frame(0, 0.0)))
    assert harness.engine.event_log == ()
    assert harness.audio.batches == []


def test_wrong_side_tracker_event_fails_before_event_log_or_audio() -> None:
    p1_tracker = _TrackerSpy(
        returns=[GaugeTrackerUpdate(_event(Side.P2))]
    )
    harness = _build_harness(p1_tracker=p1_tracker, p2_tracker=_TrackerSpy())
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"

    with pytest.raises(ValueError):
        harness.engine.process(_running(_frame(0, 0.0)))
    assert harness.engine.event_log == ()
    assert harness.audio.batches == []


def test_non_tuple_scheduler_result_is_programmer_error_after_event_is_logged() -> None:
    event = _event(Side.P1)
    harness = _build_harness(
        p1_tracker=_TrackerSpy(returns=[GaugeTrackerUpdate(event)]),
        p2_tracker=_TrackerSpy(),
        p1_scheduler=_SchedulerSpy(Side.P1, [AudioAction(AudioKind.STOP, Side.P1)]),
    )
    harness.observer.kinds[(0, Side.P1)] = "full"
    harness.observer.kinds[(0, Side.P2)] = "full"

    with pytest.raises(TypeError):
        harness.engine.process(_running(_frame(0, 0.0)))
    assert harness.engine.event_log == (event,)
    assert harness.audio.batches == []


def test_process_requires_actual_capture_sample() -> None:
    harness = _build_harness()
    for bad_sample in (None, object(), "RUNNING"):
        with pytest.raises(TypeError):
            harness.engine.process(bad_sample)


def test_invalid_explicit_reset_reason_is_atomic() -> None:
    p1_tracker = _TrackerSpy()
    p2_tracker = _TrackerSpy()
    harness = _build_harness(p1_tracker=p1_tracker, p2_tracker=p2_tracker)
    for bad_reason in (None, "", "   "):
        with pytest.raises((TypeError, ValueError)):
            harness.engine.reset(bad_reason)
    assert p1_tracker.reset_reasons == p2_tracker.reset_reasons == []
