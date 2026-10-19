"""Frame-to-audio orchestration for the screen-only trigger pipeline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from numbers import Rational, Real
import time
from typing import Callable

from .audio import AudioAction, AudioKind, AudioPlaybackError
from .capture import FreshFrameGate
from .config import SideAudioConfig, SuperScope
from .contact_hud import read_contact_hud
from .diagnostic_log import DiagnosticEventLog, DiagnosticRecord, LatencyRecord
from .domain import (
    CancelledSuperCandidate,
    CaptureSample,
    CaptureStatus,
    ClientGeometry,
    ConfirmedSuperEvent,
    GaugeTrackerUpdate,
    Side,
    SuperCandidateEvent,
)
from .gauge_features import GaugeObservation, observe_gauge
from .hud_profile import HudProfile
from .meter_references import MeterReferences
from .outcome import OutcomeDecision, OutcomeLabel, OutcomeObserver
from .viewport import CenteredViewportResolver, ViewportStatus


@dataclass(frozen=True)
class EngineUpdate:
    events: tuple[ConfirmedSuperEvent, ...] = ()
    candidates: tuple[SuperCandidateEvent, ...] = ()
    cancellations: tuple[CancelledSuperCandidate, ...] = ()

    def __post_init__(self) -> None:
        if type(self.events) is not tuple:
            raise TypeError("events must be a tuple")
        if not all(isinstance(event, ConfirmedSuperEvent) for event in self.events):
            raise TypeError("events must contain only ConfirmedSuperEvent values")
        if type(self.candidates) is not tuple:
            raise TypeError("candidates must be a tuple")
        if not all(
            isinstance(candidate, SuperCandidateEvent)
            for candidate in self.candidates
        ):
            raise TypeError(
                "candidates must contain only SuperCandidateEvent values"
            )
        if type(self.cancellations) is not tuple:
            raise TypeError("cancellations must be a tuple")
        if not all(
            isinstance(cancellation, CancelledSuperCandidate)
            for cancellation in self.cancellations
        ):
            raise TypeError(
                "cancellations must contain only CancelledSuperCandidate values"
            )


@dataclass(frozen=True)
class _HeldFade:
    due: Real
    event: ConfirmedSuperEvent
    action: AudioAction
    scope: SuperScope
    outcome: OutcomeLabel | None


def _require_aspect_tolerance(value: object) -> Real:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("max_relative_aspect_error must be a real number")
    if isinstance(value, Rational):
        if value < 0:
            raise ValueError("max_relative_aspect_error must be nonnegative")
        return value
    if not math.isfinite(value) or value < 0:
        raise ValueError("max_relative_aspect_error must be finite and nonnegative")
    return value


def profile_applies_to(
    profile: HudProfile,
    geometry: ClientGeometry,
    *,
    max_relative_aspect_error: Real = 0.01,
) -> bool:
    """Return whether a captured client geometry matches the calibrated aspect."""

    if not isinstance(profile, HudProfile):
        raise TypeError("profile must be HudProfile")
    if not isinstance(geometry, ClientGeometry):
        raise TypeError("geometry must be ClientGeometry")
    tolerance = _require_aspect_tolerance(max_relative_aspect_error)

    expected = profile.calibration.client_aspect_ratio
    actual = geometry.width / geometry.height
    relative_error = abs(actual - expected) / expected
    return relative_error <= tolerance or math.isclose(
        relative_error,
        tolerance,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _scope_filters_event(
    event: ConfirmedSuperEvent, config: SideAudioConfig
) -> bool:
    return (
        config.super_scope is SuperScope.SA3_CA_ONLY
        and event.levels_spent in {1, 2}
    )


def _ignore_record(record: DiagnosticRecord) -> None:
    del record


def _remember_bounded_key(
    history: OrderedDict[tuple[Side, str], None],
    key: tuple[Side, str],
    capacity: int,
) -> None:
    history.pop(key, None)
    history[key] = None
    while len(history) > capacity:
        history.popitem(last=False)


def _remember_bounded_context(
    history: OrderedDict[tuple[Side, str], ConfirmedSuperEvent],
    key: tuple[Side, str],
    event: ConfirmedSuperEvent,
    capacity: int,
) -> None:
    history.pop(key, None)
    history[key] = event
    while len(history) > capacity:
        history.popitem(last=False)


class VisualTriggerEngine:
    """Coordinate fresh captured frames into independent P1/P2 audio batches."""

    def __init__(
        self,
        *,
        profile: HudProfile,
        p1_references: MeterReferences,
        p2_references: MeterReferences,
        p1_tracker: object,
        p2_tracker: object,
        freshness_gate: FreshFrameGate,
        p1_scheduler: object,
        p2_scheduler: object,
        p1_audio_config: SideAudioConfig,
        p2_audio_config: SideAudioConfig,
        audio_port: object,
        audio_diagnostic: Callable[[AudioPlaybackError], None],
        diagnostic_sink: Callable[[DiagnosticRecord], None] = _ignore_record,
        outcome_observer: object | None = None,
        clock: Callable[[], Real] = time.perf_counter,
        observer: Callable[..., GaugeObservation] = observe_gauge,
        lifecycle_history_capacity: int = 1024,
        viewport_resolver: CenteredViewportResolver | None = None,
    ) -> None:
        if not isinstance(profile, HudProfile):
            raise TypeError("profile must be HudProfile")
        if not isinstance(p1_references, MeterReferences):
            raise TypeError("p1_references must be MeterReferences")
        if not isinstance(p2_references, MeterReferences):
            raise TypeError("p2_references must be MeterReferences")
        if p1_references is p2_references:
            raise ValueError("P1 and P2 references must be distinct objects")
        if p1_references.side is not Side.P1:
            raise ValueError("p1_references must contain P1 references")
        if p2_references.side is not Side.P2:
            raise ValueError("p2_references must contain P2 references")
        if profile.p1.meter_roi == profile.p2.meter_roi:
            raise ValueError("P1 and P2 meter ROIs must be distinct")
        if not isinstance(freshness_gate, FreshFrameGate):
            raise TypeError("freshness_gate must be FreshFrameGate")
        if not isinstance(p1_audio_config, SideAudioConfig):
            raise TypeError("p1_audio_config must be SideAudioConfig")
        if not isinstance(p2_audio_config, SideAudioConfig):
            raise TypeError("p2_audio_config must be SideAudioConfig")
        self._require_tracker(p1_tracker, "p1_tracker")
        self._require_tracker(p2_tracker, "p2_tracker")
        self._require_scheduler(p1_scheduler, Side.P1, "p1_scheduler")
        self._require_scheduler(p2_scheduler, Side.P2, "p2_scheduler")
        if not callable(getattr(audio_port, "execute", None)):
            raise TypeError("audio_port must provide execute")
        if not callable(audio_diagnostic):
            raise TypeError("audio_diagnostic must be callable")
        if not callable(diagnostic_sink):
            raise TypeError("diagnostic_sink must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(observer):
            raise TypeError("observer must be callable")
        if type(lifecycle_history_capacity) is not int:
            raise TypeError("lifecycle_history_capacity must be an integer")
        if lifecycle_history_capacity < 2:
            raise ValueError("lifecycle_history_capacity must be at least two")
        if viewport_resolver is not None and not isinstance(
            viewport_resolver, CenteredViewportResolver
        ):
            raise TypeError(
                "viewport_resolver must be CenteredViewportResolver or None"
            )
        actual_outcome_observer = outcome_observer or OutcomeObserver(
            contact_reader=read_contact_hud,
        )
        self._require_outcome_observer(actual_outcome_observer)

        self._profile = profile
        self._p1_references = p1_references
        self._p2_references = p2_references
        self._p1_tracker = p1_tracker
        self._p2_tracker = p2_tracker
        self._freshness_gate = freshness_gate
        self._p1_scheduler = p1_scheduler
        self._p2_scheduler = p2_scheduler
        self._p1_audio_config = p1_audio_config
        self._p2_audio_config = p2_audio_config
        self._audio_port = audio_port
        self._audio_diagnostic = audio_diagnostic
        self._diagnostic_sink = diagnostic_sink
        log_owner = getattr(diagnostic_sink, "__self__", None)
        self._latency_log = (
            log_owner.latency_log() if isinstance(log_owner, DiagnosticEventLog) else None
        )
        self._latency_cycle = 0
        self._latency_context: dict[str, object] = {}
        self._latency_samples: dict[object, tuple[float, object]] = {}
        self._latency_audio_call = 0
        self._latency_capture_selected = False
        self._clock = clock
        self._observer = observer
        self._viewport_resolver = viewport_resolver
        self._viewport_status = ViewportStatus.NATIVE.value
        self._outcome_observer = actual_outcome_observer
        self._lifecycle_history_capacity = lifecycle_history_capacity
        self._event_log: list[ConfirmedSuperEvent] = []
        self._candidate_log: list[SuperCandidateEvent] = []
        self._cancellation_log: list[CancelledSuperCandidate] = []
        self._pending_candidates_by_side: dict[Side, SuperCandidateEvent] = {}
        self._started_candidate_keys: OrderedDict[tuple[Side, str], None] = (
            OrderedDict()
        )
        self._terminal_candidate_keys: OrderedDict[tuple[Side, str], None] = (
            OrderedDict()
        )
        self._executed_event_keys: OrderedDict[tuple[Side, str], None] = (
            OrderedDict()
        )
        self._event_context_by_key: OrderedDict[
            tuple[Side, str], ConfirmedSuperEvent
        ] = OrderedDict()
        self._status_event_by_side: dict[Side, str] = {}
        self._audio_owner_by_side: dict[Side, tuple[str, str]] = {}
        self._superseded_candidate_keys: set[tuple[Side, str]] = set()
        self._playback_started_by_side: dict[Side, Real] = {}
        self._held_fades: dict[Side, _HeldFade] = {}
        self._diagnostic_warning = False
        self._latency_reason = "waiting"
        self._outcome_status_by_side: dict[Side, str | None] = {
            Side.P1: None,
            Side.P2: None,
        }

    @property
    def event_log(self) -> tuple[ConfirmedSuperEvent, ...]:
        return tuple(self._event_log)

    @property
    def candidate_log(self) -> tuple[SuperCandidateEvent, ...]:
        return tuple(self._candidate_log)

    @property
    def cancellation_log(self) -> tuple[CancelledSuperCandidate, ...]:
        return tuple(self._cancellation_log)

    @property
    def outcome_statuses(self) -> tuple[str | None, str | None]:
        return (
            self._outcome_status_by_side[Side.P1],
            self._outcome_status_by_side[Side.P2],
        )

    @property
    def diagnostic_warning(self) -> bool:
        return self._diagnostic_warning

    @property
    def viewport_status(self) -> str:
        return self._viewport_status

    @property
    def processing_reason(self) -> str:
        return self._latency_reason

    def _latency_due(self, key: object, signature: object = None) -> bool:
        if self._latency_log is None:
            return False
        now = time.perf_counter()
        previous = self._latency_samples.get(key)
        if previous is not None:
            interval = 0.1 if signature != previous[1] else 0.5
            if now - previous[0] < interval:
                return False
        self._latency_samples[key] = (now, signature)
        return True

    def _latency_emit(self, phase: str, **values: object) -> None:
        if self._latency_log is None:
            return
        try:
            self._latency_log.record(LatencyRecord(
                phase=phase, host_time=time.perf_counter(),
                **(self._latency_context | values),
            ))
        except Exception:
            self._diagnostic_warning = True

    def capture_begin(self) -> None:
        """Host read boundary, called only by the listener, not a device clock."""
        if self._latency_log is None:
            return
        self._latency_cycle += 1
        self._latency_context = {"cycle_id": self._latency_cycle,
                                 "capture_begin": time.perf_counter()}
        self._latency_capture_selected = self._latency_due("capture")
        if self._latency_capture_selected:
            self._latency_emit("CAPTURE_BEGIN")

    def capture_end(self) -> None:
        if self._latency_log is None:
            return
        self._latency_context["capture_end"] = time.perf_counter()
        if self._latency_capture_selected:
            self._latency_emit("CAPTURE_END")

    def process(self, sample: CaptureSample) -> EngineUpdate:
        if self._latency_log is None:
            return self._process(sample)
        if "capture_begin" not in self._latency_context:
            self._latency_cycle += 1
            self._latency_context = {"cycle_id": self._latency_cycle}
        self._latency_context["process_begin"] = time.perf_counter()
        if isinstance(sample, CaptureSample) and sample.frame is not None:
            self._latency_context.update(frame_id=sample.frame.frame_id,
                                         source_time=float(sample.frame.monotonic_time))
        selected = self._latency_due("process")
        if selected:
            self._latency_emit("PROCESS_BEGIN")
        self._latency_reason = "failed"
        try:
            result = self._process(sample)
            return result
        except BaseException:
            self._latency_reason = "failed"
            raise
        finally:
            if selected or self._latency_due("process_reason", self._latency_reason):
                self._latency_emit("PROCESS_END", reason=self._latency_reason)
            # No frame object or stale frame attribution survives processing.
            self._latency_context = {}

    def _latency_meter(self, observation: GaugeObservation, tracker: object,
                       baseline_before: int | None) -> None:
        if self._latency_log is None:
            return
        try:
            baseline_after = getattr(tracker, "stable_meter_level", None)
            state = getattr(tracker, "state", None)
            signature = (observation.meter_level, observation.hud_visible,
                         observation.meter_level_supported, observation.spent_structure_verified,
                         round(observation.meter_level_confidence, 1),
                         round(observation.spent_confidence, 1),
                         baseline_before, baseline_after, state,
                         None if observation.normalized_meter is None else round(observation.normalized_meter, 1))
            if self._latency_due(("meter", observation.side), signature):
                self._latency_emit(
                    "METER", side=observation.side, meter_level=observation.meter_level,
                    meter_level_confidence=observation.meter_level_confidence,
                    normalized_meter=observation.normalized_meter,
                    spent_confidence=float(observation.spent_confidence),
                    hud_visible=observation.hud_visible,
                    level_supported=observation.meter_level_supported,
                    structure_verified=observation.spent_structure_verified,
                    baseline_before=baseline_before, baseline_after=baseline_after, state=state,
                )
        except Exception:
            self._diagnostic_warning = True

    def _process(self, sample: CaptureSample) -> EngineUpdate:
        if type(sample) is not CaptureSample:
            raise TypeError("sample must be CaptureSample")
        if sample.status is not CaptureStatus.RUNNING:
            self._latency_reason = "not_running"
            cancellations = self._runtime_reset("capture sample is not running")
            return EngineUpdate(cancellations=cancellations)

        frame = sample.frame
        if frame is None:
            raise RuntimeError("RUNNING sample did not contain a frame")
        now = self._clock()
        if not self._freshness_gate.accept(frame, now=now):
            self._latency_reason = "stale"
            cancellations = self._runtime_reset(
                "capture frame was rejected as stale"
            )
            return EngineUpdate(cancellations=cancellations)
        if self._viewport_resolver is not None:
            viewport = self._viewport_resolver.resolve(frame)
            self._viewport_status = viewport.status.value
            if viewport.frame is None:
                self._latency_reason = "viewport"
                cancellations = self._runtime_reset(
                    "capture viewport is not ready"
                )
                return EngineUpdate(cancellations=cancellations)
            frame = viewport.frame
        else:
            self._viewport_status = ViewportStatus.NATIVE.value
        if not profile_applies_to(self._profile, frame.geometry):
            self._latency_reason = "geometry"
            cancellations = self._runtime_reset(
                "capture geometry does not match profile"
            )
            return EngineUpdate(cancellations=cancellations)

        for side in (Side.P1, Side.P2):
            if self._outcome_status_by_side[side] == "等待大招结果":
                self._outcome_status_by_side[side] = "后续动画"
        self._outcome_observer.remember(frame)
        p1_observation = self._observer(
            frame.image,
            side=Side.P1,
            frame_id=frame.frame_id,
            roi=self._profile.p1.meter_roi,
            references=self._p1_references,
            sampled_at=frame.monotonic_time,
        )
        p2_observation = self._observer(
            frame.image,
            side=Side.P2,
            frame_id=frame.frame_id,
            roi=self._profile.p2.meter_roi,
            references=self._p2_references,
            sampled_at=frame.monotonic_time,
        )
        if not isinstance(p1_observation, GaugeObservation):
            raise TypeError("P1 observer result must be GaugeObservation")
        if not isinstance(p2_observation, GaugeObservation):
            raise TypeError("P2 observer result must be GaugeObservation")
        if p1_observation.side is not Side.P1:
            raise ValueError("P1 observer result must belong to P1")
        if p2_observation.side is not Side.P2:
            raise ValueError("P2 observer result must belong to P2")
        self._latency_reason = "ok"
        outcome_decisions = self._outcome_observer.update(
            frame,
            observations=(p1_observation, p2_observation),
        )

        level_tracking_supported = (
            p1_observation.meter_level_supported
            or p2_observation.meter_level_supported
        )
        if (
            not p1_observation.hud_visible
            and not p2_observation.hud_visible
            and not level_tracking_supported
        ):
            self._latency_reason = "hud_hidden"
            self._latency_meter(p1_observation, self._p1_tracker, None)
            self._latency_meter(p2_observation, self._p2_tracker, None)
            cancellations = self._reset_trackers("both HUD meters are hidden")
            self._route_outcomes(outcome_decisions)
            self._flush_held_fades()
            return EngineUpdate(cancellations=cancellations)
        p1_trustworthy = (
            p1_observation.hud_visible
            and p1_observation.spent_structure_verified
        )
        p2_trustworthy = (
            p2_observation.hud_visible
            and p2_observation.spent_structure_verified
        )
        if (
            not p1_trustworthy
            and not p2_trustworthy
            and not level_tracking_supported
        ):
            self._latency_reason = "hud_untrusted"
            self._latency_meter(p1_observation, self._p1_tracker, None)
            self._latency_meter(p2_observation, self._p2_tracker, None)
            cancellations = self._reset_trackers(
                "neither HUD meter is trustworthy"
            )
            self._route_outcomes(outcome_decisions)
            self._flush_held_fades()
            return EngineUpdate(cancellations=cancellations)

        baselines = (None, None)
        if self._latency_log is not None:
            try:
                baselines = (getattr(self._p1_tracker, "stable_meter_level", None),
                             getattr(self._p2_tracker, "stable_meter_level", None))
            except Exception:
                self._diagnostic_warning = True
        p1_update = self._validated_tracker_update(
            self._p1_tracker.update(p1_observation),
            Side.P1,
        )
        p2_update = self._validated_tracker_update(
            self._p2_tracker.update(p2_observation),
            Side.P2,
        )
        self._latency_meter(p1_observation, self._p1_tracker, baselines[0])
        self._latency_meter(p2_observation, self._p2_tracker, baselines[1])
        events: list[ConfirmedSuperEvent] = []
        candidates: list[SuperCandidateEvent] = []
        cancellations: list[CancelledSuperCandidate] = []
        for update in (p1_update, p2_update):
            transition = update.transition
            if transition is None:
                continue
            if isinstance(transition, SuperCandidateEvent):
                if self._route_candidate(transition):
                    candidates.append(transition)
                continue
            if isinstance(transition, ConfirmedSuperEvent):
                if self._route_confirmation(transition):
                    events.append(transition)
                continue
            if self._route_cancellation(transition):
                cancellations.append(transition)

        # Commit this frame's newest confirmations before routing older result
        # decisions. This prevents a superseded BLOCK/NON_HIT sound from
        # producing a brief audible blip immediately before the new main track.
        self._route_outcomes(outcome_decisions)
        # Flush after this frame's transitions: a newer play always wins over
        # an old fade becoming due on the same frame. No timer thread races.
        self._flush_held_fades()
        return EngineUpdate(
            events=tuple(events),
            candidates=tuple(candidates),
            cancellations=tuple(cancellations),
        )

    def reset(self, reason: str) -> None:
        if type(reason) is not str:
            raise TypeError("reason must be a string")
        if not reason.strip():
            raise ValueError("reason must be nonblank")
        self._flush_held_fades(force=True)
        self._reset_trackers(reason)
        reset_decisions = self._outcome_observer.reset(reason)
        self._route_outcomes(reset_decisions)
        self._freshness_gate.reset()
        if self._viewport_resolver is not None:
            self._viewport_resolver.reset()
        self._viewport_status = ViewportStatus.NATIVE.value
        self._latency_reason = "waiting"
        self._pending_candidates_by_side.clear()
        self._started_candidate_keys.clear()
        self._terminal_candidate_keys.clear()
        self._executed_event_keys.clear()
        self._event_context_by_key.clear()
        self._status_event_by_side.clear()
        self._audio_owner_by_side.clear()
        self._superseded_candidate_keys.clear()
        self._playback_started_by_side.clear()
        self._held_fades.clear()

    def _runtime_reset(
        self,
        reason: str,
    ) -> tuple[CancelledSuperCandidate, ...]:
        self._flush_held_fades(force=True)
        cancellations = self._reset_trackers(reason)
        reset_decisions = self._outcome_observer.reset(reason)
        self._route_outcomes(reset_decisions)
        self._pending_candidates_by_side.clear()
        self._started_candidate_keys.clear()
        self._terminal_candidate_keys.clear()
        self._executed_event_keys.clear()
        self._event_context_by_key.clear()
        self._status_event_by_side.clear()
        self._audio_owner_by_side.clear()
        self._superseded_candidate_keys.clear()
        self._playback_started_by_side.clear()
        self._held_fades.clear()
        return cancellations

    def _reset_trackers(
        self,
        reason: str,
    ) -> tuple[CancelledSuperCandidate, ...]:
        p1_update = self._validated_tracker_update(
            self._p1_tracker.reset(reason),
            Side.P1,
        )
        p2_update = self._validated_tracker_update(
            self._p2_tracker.reset(reason),
            Side.P2,
        )
        cancellations: list[CancelledSuperCandidate] = []
        for update in (p1_update, p2_update):
            transition = update.transition
            if transition is None:
                continue
            if not isinstance(transition, CancelledSuperCandidate):
                raise TypeError("tracker reset may return only a cancellation")
            if self._route_cancellation(transition):
                cancellations.append(transition)
        return tuple(cancellations)

    @staticmethod
    def _require_tracker(value: object, name: str) -> None:
        if not callable(getattr(value, "update", None)):
            raise TypeError(f"{name} must provide update")
        if not callable(getattr(value, "reset", None)):
            raise TypeError(f"{name} must provide reset")

    @staticmethod
    def _require_scheduler(value: object, side: Side, name: str) -> None:
        if getattr(value, "side", None) is not side:
            raise ValueError(f"{name} side must match {side.value}")
        if not callable(getattr(value, "on_candidate", None)):
            raise TypeError(f"{name} must provide on_candidate")
        if not callable(getattr(value, "on_confirmation", None)):
            raise TypeError(f"{name} must provide on_confirmation")
        if not callable(getattr(value, "on_cancellation", None)):
            raise TypeError(f"{name} must provide on_cancellation")
        if not callable(getattr(value, "on_outcome", None)):
            raise TypeError(f"{name} must provide on_outcome")

    @staticmethod
    def _require_outcome_observer(value: object) -> None:
        for method in ("remember", "begin", "update", "reset"):
            if not callable(getattr(value, method, None)):
                raise TypeError(f"outcome_observer must provide {method}")

    def _route_outcomes(self, value: object) -> None:
        if type(value) is not tuple:
            raise TypeError("outcome observer result must be a tuple")
        for decision in value:
            if not isinstance(decision, OutcomeDecision):
                raise TypeError(
                    "outcome observer result must contain only OutcomeDecision values"
                )
            if decision.side is Side.P1:
                scheduler = self._p1_scheduler
                config = self._p1_audio_config
            else:
                scheduler = self._p2_scheduler
                config = self._p2_audio_config
            actions = scheduler.on_outcome(decision, config)
            if type(actions) is not tuple:
                raise TypeError("outcome scheduler result must be a tuple")
            if not all(isinstance(action, AudioAction) for action in actions):
                raise TypeError(
                    "outcome scheduler result must contain only AudioAction values"
                )
            if any(action.side is not decision.side for action in actions):
                raise ValueError("outcome audio action side must match decision side")
            event_key = (decision.side, decision.event_id)
            event = self._event_context_by_key.get(event_key)
            owner = self._audio_owner_by_side.get(decision.side)
            if owner != ("event", decision.event_id):
                # A delayed result belongs to its event, never the newer track.
                actions = ()
            status_event_id = self._status_event_by_side.get(decision.side)
            if event is None or status_event_id == decision.event_id:
                # Observer-spy/external decisions have no retained event
                # context, but remain safe to display. Status authority is
                # independent of audio ownership, so disabled audio still
                # reaches a terminal result while superseded events cannot.
                self._outcome_status_by_side[decision.side] = {
                    "HIT": "命中",
                    "BLOCK": "格挡",
                    "WHIFF": "打空",
                    "NON_HIT": "格挡/未命中",
                    "FILTERED_TIER": "低等级已淡出",
                    "UNKNOWN": "结果不确定",
                    "RESET": "结果不确定",
                }[decision.label.value]
            if decision.label is OutcomeLabel.HIT:
                held = self._held_fades.get(decision.side)
                if held is not None and held.event.event_id == decision.event_id:
                    self._held_fades.pop(decision.side, None)
            if event is not None and decision.label in {
                OutcomeLabel.UNKNOWN, OutcomeLabel.BLOCK, OutcomeLabel.WHIFF,
                OutcomeLabel.NON_HIT,
            }:
                actions = self._hold_result_fade(actions, event, config, decision.label)
            if actions:
                self._execute_audio(actions, owner=("event", decision.event_id),
                                    diagnostic_identity=(decision.event_id if event is None
                                                         else event.candidate_id or event.event_id))
            self._emit_diagnostic(
                monotonic_time=decision.decided_at,
                side=decision.side,
                identity=(
                    decision.event_id
                    if event is None
                    else event.candidate_id or event.event_id
                ),
                transition="OUTCOME",
                evidence=decision.evidence,
                meter_before=None if event is None else event.meter_before,
                meter_after=None if event is None else event.meter_after,
                levels_spent=None if event is None else event.levels_spent,
                scope=config.super_scope,
                audio_actions=self._audio_kinds(actions),
                outcome=decision.label,
            )
            self._event_context_by_key.pop(event_key, None)

    def _scheduler_for(
        self,
        side: Side,
    ) -> tuple[object, SideAudioConfig]:
        if side is Side.P1:
            return self._p1_scheduler, self._p1_audio_config
        return self._p2_scheduler, self._p2_audio_config

    def _route_candidate(self, candidate: SuperCandidateEvent) -> bool:
        pending = self._pending_candidates_by_side.get(candidate.side)
        candidate_key = (candidate.side, candidate.candidate_id)
        if candidate_key in self._terminal_candidate_keys:
            return False
        if pending is not None:
            if pending.candidate_id == candidate.candidate_id:
                return False
            raise ValueError("candidate ID must match the pending side candidate")

        scheduler, config = self._scheduler_for(candidate.side)
        self._pending_candidates_by_side[candidate.side] = candidate
        self._candidate_log.append(candidate)
        actions = scheduler.on_candidate(candidate, config)
        self._validate_audio_actions(actions, candidate.side)
        if actions and self._execute_audio(actions, owner=("candidate", candidate.candidate_id)):
            _remember_bounded_key(
                self._started_candidate_keys,
                candidate_key,
                self._lifecycle_history_capacity,
            )
        self._emit_diagnostic(
            monotonic_time=candidate.suspected_at,
            side=candidate.side,
            identity=candidate.candidate_id,
            transition="CANDIDATE",
            evidence=candidate.evidence,
            meter_before=candidate.meter_before,
            meter_after=candidate.meter_after,
            levels_spent=None,
            scope=config.super_scope,
            audio_actions=self._audio_kinds(actions),
            outcome=None,
        )
        return True

    def _route_confirmation(self, event: ConfirmedSuperEvent) -> bool:
        event_key = (event.side, event.event_id)
        if event_key in self._executed_event_keys:
            return False
        pending = self._pending_candidates_by_side.get(event.side)
        event_candidate_key = (
            (event.side, event.candidate_id)
            if event.candidate_id is not None
            else None
        )
        if pending is not None and event.candidate_id != pending.candidate_id:
            raise ValueError("confirmation ID must match the pending candidate")
        if (
            pending is None
            and event_candidate_key is not None
            and event_candidate_key in self._terminal_candidate_keys
        ):
            return False

        playback_started = False
        superseded = event_candidate_key in self._superseded_candidate_keys
        activation_at = event.confirmed_at
        if pending is not None:
            del self._pending_candidates_by_side[event.side]
            pending_key = (pending.side, pending.candidate_id)
            playback_started = (
                pending_key in self._started_candidate_keys
                and self._audio_owner_by_side.get(event.side) == ("candidate", pending.candidate_id)
            )
            self._started_candidate_keys.pop(pending_key, None)
            _remember_bounded_key(
                self._terminal_candidate_keys,
                pending_key,
                self._lifecycle_history_capacity,
            )
            activation_at = pending.suspected_at
        elif event_candidate_key is not None:
            _remember_bounded_key(
                self._terminal_candidate_keys,
                event_candidate_key,
                self._lifecycle_history_capacity,
            )

        _remember_bounded_key(
            self._executed_event_keys,
            event_key,
            self._lifecycle_history_capacity,
        )
        self._event_log.append(event)
        self._superseded_candidate_keys.discard(event_candidate_key)
        scheduler, config = self._scheduler_for(event.side)
        actions = scheduler.on_confirmation(event, config, playback_started)
        self._validate_audio_actions(actions, event.side)
        if superseded:
            actions = ()
        if playback_started:
            self._audio_owner_by_side[event.side] = ("event", event.event_id)
        if config.super_scope is SuperScope.ALL_SUPERS and event.levels_spent in {1, 2}:
            # PLAY must succeed before starting the four-second clock. A
            # candidate that already played retains its original start time.
            initial = tuple(a for a in actions if a.kind is not AudioKind.FADE)
            fades = tuple(a for a in actions if a.kind is AudioKind.FADE)
            if initial:
                if not self._execute_audio(initial, owner=("event", event.event_id),
                                           diagnostic_identity=event.candidate_id or event.event_id):
                    fades = ()
            fades = self._hold_result_fade(fades, event, config, None)
            if fades:
                self._execute_audio(fades, diagnostic_identity=event.candidate_id or event.event_id)
            actions = initial + fades
        elif actions:
            self._execute_audio(actions, owner=("event", event.event_id),
                                diagnostic_identity=event.candidate_id or event.event_id)
        # begin() replaces the side's pending result window. Commit it only
        # after audio execution: a preparation failure can leave the previous
        # event audible and still dependent on its original result/deadline.
        owner = self._audio_owner_by_side.get(event.side)
        if owner is None or owner == ("event", event.event_id):
            self._status_event_by_side[event.side] = event.event_id
            if _scope_filters_event(event, config):
                self._outcome_status_by_side[event.side] = "低等级已过滤"
            elif config.super_scope is SuperScope.ALL_SUPERS and event.levels_spent in {1, 2}:
                # This confirmed spend owns its fade; contact cannot override it.
                self._outcome_status_by_side[event.side] = "等待淡出"
            else:
                _remember_bounded_context(
                    self._event_context_by_key,
                    event_key,
                    event,
                    self._lifecycle_history_capacity,
                )
                self._outcome_observer.begin(
                    event,
                    config.super_scope,
                    activation_at=activation_at,
                )
                self._outcome_status_by_side[event.side] = "等待大招结果"
        self._emit_diagnostic(
            monotonic_time=event.confirmed_at,
            side=event.side,
            identity=event.candidate_id or event.event_id,
            transition="CONFIRMED",
            evidence=(),
            meter_before=event.meter_before,
            meter_after=event.meter_after,
            levels_spent=event.levels_spent,
            scope=config.super_scope,
            audio_actions=self._audio_kinds(actions),
            outcome=None,
        )
        return True

    def _route_cancellation(
        self,
        cancellation: CancelledSuperCandidate,
    ) -> bool:
        candidate_key = (cancellation.side, cancellation.candidate_id)
        if candidate_key in self._terminal_candidate_keys:
            return False
        pending = self._pending_candidates_by_side.get(cancellation.side)
        if pending is None:
            return False
        if cancellation.candidate_id != pending.candidate_id:
            raise ValueError("cancellation ID must match the pending candidate")

        del self._pending_candidates_by_side[cancellation.side]
        playback_started = (
            candidate_key in self._started_candidate_keys
            and self._audio_owner_by_side.get(cancellation.side) == ("candidate", cancellation.candidate_id)
        )
        self._superseded_candidate_keys.discard(candidate_key)
        self._started_candidate_keys.pop(candidate_key, None)
        _remember_bounded_key(
            self._terminal_candidate_keys,
            candidate_key,
            self._lifecycle_history_capacity,
        )
        self._cancellation_log.append(cancellation)
        scheduler, config = self._scheduler_for(cancellation.side)
        # A silent candidate does not own the current channel's older music.
        actions = scheduler.on_cancellation(cancellation, config) if playback_started else ()
        self._validate_audio_actions(actions, cancellation.side)
        if actions:
            self._execute_audio(actions, diagnostic_identity=cancellation.candidate_id)
        self._emit_diagnostic(
            monotonic_time=cancellation.decided_at,
            side=cancellation.side,
            identity=cancellation.candidate_id,
            transition="CANCELLED",
            evidence=(cancellation.reason.value,),
            meter_before=None,
            meter_after=None,
            levels_spent=None,
            scope=config.super_scope,
            audio_actions=self._audio_kinds(actions),
            outcome=None,
        )
        return True

    @staticmethod
    def _validate_audio_actions(value: object, side: Side) -> None:
        if type(value) is not tuple:
            raise TypeError("scheduler result must be a tuple")
        if not all(isinstance(action, AudioAction) for action in value):
            raise TypeError("scheduler result must contain only AudioAction values")
        if any(action.side is not side for action in value):
            raise ValueError("audio action side must match lifecycle transition side")

    def _execute_audio(
        self, actions: tuple[AudioAction, ...], *, owner: tuple[str, str] | None = None,
        diagnostic_identity: str | None = None,
    ) -> bool:
        self._latency_audio_call += 1
        audio_values = {"audio_call_id": self._latency_audio_call,
                        "side": actions[0].side if actions else None,
                        "audio_actions": self._audio_kinds(actions),
                        # Lifecycle correlation is independent of the channel's
                        # candidate->event ownership promotion.
                        "identity": (diagnostic_identity if diagnostic_identity is not None
                                     else None if owner is None else owner[1])}
        # Event-driven records are never sampled: two plays or a stop/fade can
        # execute on the same frame. Only the bounded writer queue may drop them.
        selected = self._latency_log is not None
        if selected:
            self._latency_emit("AUDIO_BEGIN", **audio_values)
        try:
            try:
                self._audio_port.execute(actions)
            except BaseException:
                if selected:
                    self._latency_emit("AUDIO_END", reason="failed", **audio_values)
                raise
            else:
                if selected:
                    self._latency_emit("AUDIO_END", **audio_values)
        except AudioPlaybackError as error:
            for side in error.invalidated_sides:
                self._audio_owner_by_side.pop(side, None)
                self._playback_started_by_side.pop(side, None)
                self._held_fades.pop(side, None)
            self._audio_diagnostic(error)
            return False
        else:
            for action in actions:
                self._held_fades.pop(action.side, None)
                if action.kind is AudioKind.STOP:
                    self._playback_started_by_side.pop(action.side, None)
                if action.kind is AudioKind.PLAY:
                    # A successful PLAY is the single global ownership commit.
                    # Keep detector histories, but silence their stale followups.
                    self._audio_owner_by_side.clear()
                    self._playback_started_by_side.clear()
                    self._held_fades.clear()
                    for side, candidate in self._pending_candidates_by_side.items():
                        if side != action.side or owner != ("candidate", candidate.candidate_id):
                            self._superseded_candidate_keys.add((side, candidate.candidate_id))
                    if owner is not None:
                        self._audio_owner_by_side[action.side] = owner
                    self._playback_started_by_side[action.side] = self._clock()
            return True

    def _hold_result_fade(
        self, actions: tuple[AudioAction, ...], event: ConfirmedSuperEvent,
        config: SideAudioConfig, outcome: OutcomeLabel | None,
    ) -> tuple[AudioAction, ...]:
        if len(actions) != 1 or actions[0].kind is not AudioKind.FADE:
            return actions
        started = self._playback_started_by_side.get(event.side)
        if started is None:
            return actions
        due = started + 4.0
        if self._clock() >= due:
            return actions
        self._held_fades[event.side] = _HeldFade(due, event, actions[0], config.super_scope, outcome)
        self._outcome_status_by_side[event.side] = "等待淡出"
        return ()

    def _flush_held_fades(self, *, force: bool = False) -> None:
        if not self._held_fades:
            return
        now = self._clock()
        for side, held in tuple(self._held_fades.items()):
            if not force and now < held.due:
                continue
            self._held_fades.pop(side, None)
            event = held.event
            if self._audio_owner_by_side.get(side) != ("event", event.event_id):
                continue
            succeeded = self._execute_audio((held.action,),
                                            diagnostic_identity=event.candidate_id or event.event_id)
            self._outcome_status_by_side[side] = "淡出中" if succeeded else "淡出失败"
            self._emit_diagnostic(
                monotonic_time=now, side=side,
                identity=event.candidate_id or event.event_id, transition="FADE",
                evidence=(("reset" if force else "playback_hold_elapsed"),) if succeeded else ("fade_failed",), meter_before=event.meter_before,
                meter_after=event.meter_after, levels_spent=event.levels_spent,
                scope=held.scope, audio_actions=(AudioKind.FADE,) if succeeded else (), outcome=held.outcome,
            )

    @staticmethod
    def _audio_kinds(actions: tuple[AudioAction, ...]) -> tuple[AudioKind, ...]:
        return tuple(action.kind for action in actions)

    def _emit_diagnostic(
        self,
        *,
        monotonic_time: Real,
        side: Side,
        identity: str,
        transition: str,
        evidence: tuple[str, ...],
        meter_before: int | None,
        meter_after: int | None,
        levels_spent: int | None,
        scope: SuperScope,
        audio_actions: tuple[AudioKind, ...],
        outcome: OutcomeLabel | None,
    ) -> None:
        self._latency_emit("TRANSITION", side=side, identity=identity, transition=transition)
        try:
            self._diagnostic_sink(
                DiagnosticRecord(
                    monotonic_time=monotonic_time,
                    side=side,
                    identity=identity,
                    transition=transition,
                    evidence=evidence,
                    meter_before=meter_before,
                    meter_after=meter_after,
                    levels_spent=levels_spent,
                    scope=scope,
                    audio_actions=audio_actions,
                    outcome=outcome,
                )
            )
        except Exception:
            self._diagnostic_warning = True

    @staticmethod
    def _validated_tracker_update(
        value: object,
        side: Side,
    ) -> GaugeTrackerUpdate:
        if type(value) is not GaugeTrackerUpdate:
            raise TypeError("tracker result must be GaugeTrackerUpdate")
        transition = value.transition
        if transition is not None and transition.side is not side:
            raise ValueError("tracker transition side must match its tracker")
        return value
