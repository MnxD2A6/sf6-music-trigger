from __future__ import annotations

from collections import deque
from dataclasses import FrozenInstanceError
import threading
import time
from typing import Callable

import numpy as np
import pytest

from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError
from sf6_music_trigger.capture import CaptureStartError
from sf6_music_trigger.domain import (
    CancelledSuperCandidate,
    CandidateCancelReason,
    CaptureIssue,
    CaptureSample,
    CaptureStatus,
    CapturedFrame,
    ClientGeometry,
    ConfirmedSuperEvent,
    Side,
    SuperCandidateEvent,
    TrackerState,
)
from sf6_music_trigger.engine import EngineUpdate


def _running_frame_zero(sampled_at: float = 1.0) -> CaptureSample:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    assert image.flags.owndata and image.flags.writeable and image.flags.c_contiguous
    return CaptureSample(
        status=CaptureStatus.RUNNING,
        frame=CapturedFrame(
            frame_id=0,
            image=image,
            monotonic_time=sampled_at,
            geometry=ClientGeometry(0, 0, 1, 1),
        ),
    )


class _Source:
    def __init__(
        self,
        session_samples: Callable[[], list[CaptureSample]] | None = None,
        *,
        calls: list[str] | None = None,
        start_error: Exception | None = None,
        start_errors: list[Exception] | None = None,
    ) -> None:
        self._session_samples = session_samples or (lambda: [])
        self._samples: deque[CaptureSample] = deque()
        self._calls = calls
        self._start_error = start_error
        self._start_errors = deque(start_errors or [])
        self._wake = threading.Event()
        self.read_entered = threading.Event()
        self.start_calls = 0
        self.read_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self._calls is not None:
            self._calls.append("source.start")
        if self._start_errors:
            raise self._start_errors.popleft()
        if self._start_error is not None:
            raise self._start_error
        self._samples = deque(self._session_samples())
        self._wake.clear()
        self.read_entered.clear()

    def read(self) -> CaptureSample:
        self.read_calls += 1
        if self._calls is not None:
            self._calls.append("source.read")
        self.read_entered.set()
        if self._samples:
            return self._samples.popleft()
        if not self._wake.wait(timeout=2.0):
            raise AssertionError("test source was not stopped")
        return CaptureSample(status=CaptureStatus.STOPPED)

    def stop(self) -> None:
        self.stop_calls += 1
        if self._calls is not None:
            self._calls.append("source.stop")
        self._wake.set()


class _Engine:
    def __init__(
        self,
        *,
        calls: list[str] | None = None,
        reset_error: Exception | None = None,
        process_error: Exception | None = None,
        updates: list[EngineUpdate] | None = None,
        outcome_status_updates: list[tuple[str | None, str | None]] | None = None,
    ) -> None:
        self._calls = calls
        self._reset_error = reset_error
        self._process_error = process_error
        self._updates = deque(updates or [])
        self._outcome_status_updates = deque(outcome_status_updates or [])
        self._condition = threading.Condition()
        self.reset_reasons: list[str] = []
        self.samples: list[CaptureSample] = []
        self.outcome_statuses: tuple[str | None, str | None] = (None, None)
        self.diagnostic_warning = False

    def reset(self, reason: str) -> None:
        if self._calls is not None:
            self._calls.append(f"engine.reset:{reason}")
        self.reset_reasons.append(reason)
        if self._reset_error is not None:
            raise self._reset_error

    def process(self, sample: CaptureSample) -> EngineUpdate:
        if self._calls is not None:
            self._calls.append("engine.process")
        with self._condition:
            self.samples.append(sample)
            self._condition.notify_all()
        if self._process_error is not None:
            raise self._process_error
        if self._outcome_status_updates:
            self.outcome_statuses = self._outcome_status_updates.popleft()
        return self._updates.popleft() if self._updates else EngineUpdate()

    def wait_for_samples(self, count: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: len(self.samples) >= count,
                timeout=2.0,
            )


class _Audio:
    def __init__(
        self,
        close_errors: list[Exception] | None = None,
        *,
        calls: list[str] | None = None,
    ) -> None:
        self._close_errors = deque(close_errors or [])
        self._calls = calls
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self._calls is not None:
            self._calls.append("audio.close")
        if self._close_errors:
            raise self._close_errors.popleft()


class _DiagnosticLogSpy:
    def __init__(
        self,
        *,
        calls: list[str] | None = None,
        warnings: list[object] | None = None,
        start_error: Exception | None = None,
        close_error: Exception | None = None,
        warning_error: Exception | None = None,
    ) -> None:
        self._calls = calls
        self._warnings = deque(warnings or [False])
        self._start_error = start_error
        self._close_error = close_error
        self._warning_error = warning_error
        self.start_calls = 0
        self.close_calls = 0
        self.warning_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self._calls is not None:
            self._calls.append("diagnostic.start")
        if self._start_error is not None:
            raise self._start_error

    @property
    def has_warning(self) -> object:
        self.warning_calls += 1
        if self._calls is not None:
            self._calls.append("diagnostic.has_warning")
        if self._warning_error is not None:
            raise self._warning_error
        if len(self._warnings) > 1:
            return self._warnings.popleft()
        return self._warnings[0]

    def close(self) -> None:
        self.close_calls += 1
        if self._calls is not None:
            self._calls.append("diagnostic.close")
        if self._close_error is not None:
            raise self._close_error


def _controller(
    source: _Source,
    engine: _Engine,
    audio: _Audio,
    *,
    diagnostic_log: object | None = None,
    start_retry_seconds: float = 1.0,
):
    from sf6_music_trigger.controller import ListenerController

    return ListenerController(
        source=source,
        engine=engine,
        audio_port=audio,
        diagnostic_log=diagnostic_log,
        start_retry_seconds=start_retry_seconds,
    )


def _diagnostic_statuses(controller: object) -> list[object]:
    return [
        status
        for status in controller.drain_statuses()
        if status.kind == "diagnostic_error"
    ]


def _wait_for_outcome_statuses(
    controller: object,
    expected: tuple[str | None, str | None],
) -> None:
    deadline = time.monotonic() + 2.0
    while controller.outcome_statuses != expected:
        if time.monotonic() >= deadline:
            pytest.fail(
                f"controller outcomes did not reach {expected!r}: "
                f"{controller.outcome_statuses!r}"
            )
        threading.Event().wait(0.001)


def _candidate(side: Side, candidate_id: str) -> SuperCandidateEvent:
    return SuperCandidateEvent(
        candidate_id=candidate_id,
        side=side,
        suspected_at=1.0,
        pre_spend_frame_id=1,
        candidate_frame_id=2,
        profile_id="controller-profile",
        meter_before=3,
        meter_after=0,
        evidence=("stable_downward_level",),
    )


def _confirmation(side: Side, candidate_id: str) -> ConfirmedSuperEvent:
    return ConfirmedSuperEvent(
        event_id=f"event-{candidate_id}",
        side=side,
        confirmed_at=1.5,
        pre_spend_frame_id=1,
        confirmed_frame_id=3,
        profile_id="controller-profile",
        meter_before=3,
        meter_after=0,
        levels_spent=3,
        candidate_id=candidate_id,
    )


def _cancellation(side: Side, candidate_id: str) -> CancelledSuperCandidate:
    return CancelledSuperCandidate(
        candidate_id=candidate_id,
        side=side,
        decided_at=1.5,
        reason=CandidateCancelReason.CONFIRMATION_TIMEOUT,
    )


def test_controller_status_is_frozen_and_path_safe() -> None:
    from sf6_music_trigger.controller import ControllerStatus

    status = ControllerStatus(
        kind="audio_error",
        operation="close_stop",
        side=Side.P1,
        issue=AudioIssue.BACKEND_FAILURE,
    )
    assert vars(status) == {
        "kind": "audio_error",
        "operation": "close_stop",
        "side": Side.P1,
        "issue": AudioIssue.BACKEND_FAILURE,
        "capture_status": None,
        "capture_issue": None,
    }
    assert "private" not in str(status).lower()
    with pytest.raises(FrozenInstanceError):
        status.kind = "stopped"


def test_tracker_diagnostics_is_frozen_and_strictly_validated() -> None:
    from sf6_music_trigger.controller import TrackerDiagnostics

    diagnostics = TrackerDiagnostics(
        (TrackerState.ARMED, TrackerState.COOLDOWN),
        (2, None),
        processing_reason="hud_untrusted",
        viewport_status="native",
    )

    assert diagnostics.levels == (2, None)
    assert diagnostics.processing_reason == "hud_untrusted"
    assert diagnostics.viewport_status == "native"
    with pytest.raises(FrozenInstanceError):
        diagnostics.levels = (1, 0)
    with pytest.raises(TypeError):
        TrackerDiagnostics((TrackerState.ARMED, TrackerState.COOLDOWN), (True, 0))
    with pytest.raises(ValueError):
        TrackerDiagnostics((TrackerState.ARMED, TrackerState.COOLDOWN), (4, 0))
    with pytest.raises(ValueError):
        TrackerDiagnostics(
            (TrackerState.ARMED, TrackerState.COOLDOWN),
            (2, 0),
            processing_reason=r"private\path",
        )


def test_start_is_single_non_daemon_worker_and_stop_is_idempotent() -> None:
    source = _Source()
    controller = _controller(source, _Engine(), _Audio())

    assert controller.start() is True
    assert source.read_entered.wait(timeout=2.0)
    assert controller.worker_thread is not None
    assert controller.worker_thread.daemon is False
    assert controller.start() is False
    assert controller.request_stop() is True
    assert controller.request_stop() is True
    controller.join(timeout=2.0)

    assert controller.is_running is False
    assert source.stop_calls >= 2


def test_worker_starts_source_then_resets_engine_before_first_read() -> None:
    calls: list[str] = []
    source = _Source(lambda: [_running_frame_zero()], calls=calls)
    engine = _Engine(calls=calls)
    controller = _controller(source, engine, _Audio())

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    assert calls[:4] == [
        "source.start",
        "engine.reset:capture session started",
        "source.read",
        "engine.process",
    ]
    assert engine.samples[0].frame.frame_id == 0


def test_diagnostic_log_lifecycle_surrounds_capture_and_final_engine_reset() -> None:
    calls: list[str] = []
    source = _Source(lambda: [_running_frame_zero()], calls=calls)
    engine = _Engine(calls=calls)
    audio = _Audio(calls=calls)
    diagnostic_log = _DiagnosticLogSpy(calls=calls)
    controller = _controller(
        source,
        engine,
        audio,
        diagnostic_log=diagnostic_log,
    )

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    assert diagnostic_log.start_calls == 1
    assert diagnostic_log.close_calls == 1
    assert calls.index("diagnostic.start") < calls.index("source.start")
    assert calls.index("engine.reset:listener stopped") < calls.index(
        "diagnostic.close"
    )
    assert calls.index("audio.close") < calls.index("diagnostic.close")


def test_diagnostic_start_failure_reports_once_and_capture_still_runs() -> None:
    sample = _running_frame_zero()
    source = _Source(lambda: [sample])
    engine = _Engine()
    diagnostic_log = _DiagnosticLogSpy(
        start_error=RuntimeError("private-start-path")
    )
    controller = _controller(
        source,
        engine,
        _Audio(),
        diagnostic_log=diagnostic_log,
    )

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    statuses = _diagnostic_statuses(controller)
    assert source.start_calls == 1
    assert engine.samples == [sample]
    assert len(statuses) == 1
    assert vars(statuses[0]) == {
        "kind": "diagnostic_error",
        "operation": None,
        "side": None,
        "issue": None,
        "capture_status": None,
        "capture_issue": None,
    }
    assert "private-start-path" not in repr(statuses)


def test_diagnostic_warning_is_latched_once_after_engine_updates() -> None:
    source = _Source(lambda: [_running_frame_zero(), _running_frame_zero(2.0)])
    engine = _Engine()
    diagnostic_log = _DiagnosticLogSpy(warnings=[False, True, True])
    controller = _controller(
        source,
        engine,
        _Audio(),
        diagnostic_log=diagnostic_log,
    )

    assert controller.start()
    engine.wait_for_samples(2)
    controller.request_stop()
    controller.join(timeout=2.0)

    assert len(_diagnostic_statuses(controller)) == 1
    assert diagnostic_log.warning_calls >= 3


def test_engine_diagnostic_warning_is_latched_after_an_update() -> None:
    source = _Source(lambda: [_running_frame_zero()])
    engine = _Engine()
    engine.diagnostic_warning = True
    controller = _controller(source, engine, _Audio())

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    assert source.start_calls == 1
    assert len(_diagnostic_statuses(controller)) == 1


def test_diagnostic_warning_probe_failure_reports_once_without_stopping_detection() -> None:
    sample = _running_frame_zero()
    source = _Source(lambda: [sample])
    engine = _Engine()
    diagnostic_log = _DiagnosticLogSpy(
        warning_error=RuntimeError("private-warning-path")
    )
    controller = _controller(
        source,
        engine,
        _Audio(),
        diagnostic_log=diagnostic_log,
    )

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    statuses = _diagnostic_statuses(controller)
    assert engine.samples == [sample]
    assert len(statuses) == 1
    assert "private-warning-path" not in repr(statuses)


def test_diagnostic_close_failure_reports_once_after_audio_cleanup() -> None:
    calls: list[str] = []
    source = _Source(calls=calls)
    audio = _Audio(calls=calls)
    diagnostic_log = _DiagnosticLogSpy(
        calls=calls,
        close_error=RuntimeError("private-close-path"),
    )
    controller = _controller(
        source,
        _Engine(calls=calls),
        audio,
        diagnostic_log=diagnostic_log,
    )

    assert controller.start()
    assert source.read_entered.wait(timeout=2.0)
    controller.request_stop()
    controller.join(timeout=2.0)

    statuses = _diagnostic_statuses(controller)
    assert audio.close_calls == 1
    assert calls.index("audio.close") < calls.index("diagnostic.close")
    assert len(statuses) == 1
    assert "private-close-path" not in repr(statuses)


def test_stop_calls_source_stop_to_wake_blocked_read() -> None:
    source = _Source()
    controller = _controller(source, _Engine(), _Audio())
    assert controller.start()
    assert source.read_entered.wait(timeout=2.0)

    assert controller.request_stop()
    controller.join(timeout=2.0)

    assert source.stop_calls >= 1
    assert controller.worker_thread is not None
    assert not controller.worker_thread.is_alive()


def test_worker_routes_each_typed_sample_directly_to_engine() -> None:
    sample = _running_frame_zero()
    source = _Source(lambda: [sample])
    engine = _Engine()
    controller = _controller(source, engine, _Audio())

    controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    assert engine.samples == [sample]
    assert engine.samples[0] is sample


@pytest.mark.parametrize("failure", [RuntimeError("source-start-secret")])
def test_source_start_failure_does_not_reset_or_read_and_still_cleans_up(
    failure: Exception,
) -> None:
    source = _Source(start_error=failure)
    engine = _Engine()
    audio = _Audio()
    controller = _controller(source, engine, audio)

    controller.start()
    controller.join(timeout=2.0)

    assert engine.reset_reasons == []
    assert source.read_calls == 0
    assert source.stop_calls == 1
    assert audio.close_calls == 1
    assert "source-start-secret" not in repr(controller.drain_statuses())


def test_typed_capture_start_failure_waits_then_recovers_without_new_user_start() -> None:
    source = _Source(
        lambda: [_running_frame_zero()],
        start_errors=[
            CaptureStartError(
                CaptureStatus.LOST,
                issue=CaptureIssue.WINDOW_CHANGED,
            )
        ],
    )
    engine = _Engine()
    controller = _controller(
        source,
        engine,
        _Audio(),
        start_retry_seconds=0.01,
    )

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    statuses = controller.drain_statuses()
    waiting = [status for status in statuses if status.kind == "capture_waiting"]
    assert len(waiting) == 1
    assert waiting[0].capture_status is CaptureStatus.LOST
    assert waiting[0].capture_issue is CaptureIssue.WINDOW_CHANGED
    assert source.start_calls == 2
    assert engine.reset_reasons == ["capture session started", "listener stopped"]
    assert not any(status.kind == "worker_error" for status in statuses)


def test_stop_interrupts_persistent_capture_start_retry_without_worker_error() -> None:
    source = _Source(
        start_error=CaptureStartError(
            CaptureStatus.LOST,
            issue=CaptureIssue.BACKEND_ERROR,
        )
    )
    controller = _controller(
        source,
        _Engine(),
        _Audio(),
        start_retry_seconds=30.0,
    )

    assert controller.start()
    deadline = time.monotonic() + 2.0
    while source.start_calls < 1 and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    assert source.start_calls == 1
    controller.request_stop()
    controller.join(timeout=2.0)

    statuses = controller.drain_statuses()
    assert not controller.is_running
    assert not any(status.kind == "worker_error" for status in statuses)


def test_reset_failure_does_not_read_and_still_cleans_up() -> None:
    source = _Source()
    engine = _Engine(reset_error=RuntimeError("reset-secret"))
    audio = _Audio()
    controller = _controller(source, engine, audio)

    controller.start()
    controller.join(timeout=2.0)

    assert engine.reset_reasons == ["capture session started"]
    assert source.read_calls == 0
    assert source.stop_calls == 1
    assert audio.close_calls == 1
    assert "reset-secret" not in repr(controller.drain_statuses())


def test_worker_failure_stops_capture_and_audio() -> None:
    source = _Source(lambda: [_running_frame_zero()])
    engine = _Engine(process_error=RuntimeError("worker-private-secret"))
    audio = _Audio()
    controller = _controller(source, engine, audio)

    controller.start()
    controller.join(timeout=2.0)

    assert source.stop_calls == 1
    assert audio.close_calls == 1
    statuses = controller.drain_statuses()
    assert any(status.kind == "worker_error" for status in statuses)
    assert "worker-private-secret" not in repr(statuses)


def test_stop_restart_accepts_new_source_frame_zero_with_same_engine_and_audio() -> None:
    source = _Source(lambda: [_running_frame_zero()])
    engine = _Engine()
    audio = _Audio()
    controller = _controller(source, engine, audio)

    for expected_count in (1, 2):
        assert controller.start()
        engine.wait_for_samples(expected_count)
        controller.request_stop()
        controller.join(timeout=2.0)

    assert [sample.frame.frame_id for sample in engine.samples] == [0, 0]
    assert engine.reset_reasons == [
        "capture session started",
        "listener stopped",
        "capture session started",
        "listener stopped",
    ]
    assert audio.close_calls == 2


def test_controller_reuses_real_diagnostic_log_for_two_sessions(tmp_path) -> None:
    from sf6_music_trigger.diagnostic_log import DiagnosticEventLog

    sessions = iter(([_running_frame_zero()], [_running_frame_zero()]))
    source = _Source(lambda: next(sessions))
    engine = _Engine()
    diagnostic_log = DiagnosticEventLog(
        tmp_path,
        session_id="controller-restart",
        queue_capacity=2,
    )
    controller = _controller(
        source,
        engine,
        _Audio(),
        diagnostic_log=diagnostic_log,
    )

    for expected_count in (1, 2):
        assert controller.start()
        engine.wait_for_samples(expected_count)
        controller.request_stop()
        controller.join(timeout=2.0)

    assert len(list((tmp_path / "diagnostics").glob("*.jsonl"))) == 2


def test_controller_resets_diagnostic_warning_latch_each_start() -> None:
    sessions = iter(([_running_frame_zero()], [_running_frame_zero()]))
    source = _Source(lambda: next(sessions))
    engine = _Engine()
    diagnostic_log = _DiagnosticLogSpy(warnings=[True])
    controller = _controller(
        source,
        engine,
        _Audio(),
        diagnostic_log=diagnostic_log,
    )

    for expected_count in (1, 2):
        assert controller.start()
        engine.wait_for_samples(expected_count)
        controller.request_stop()
        controller.join(timeout=2.0)
        assert len(_diagnostic_statuses(controller)) == 1


def test_audio_cleanup_error_enqueues_only_safe_fields_and_final_close_retries() -> None:
    cleanup_error = AudioPlaybackError(
        "close_stop",
        Side.P2,
        AudioIssue.BACKEND_FAILURE,
    )
    source = _Source()
    audio = _Audio([cleanup_error])
    controller = _controller(source, _Engine(), audio)

    controller.start()
    assert source.read_entered.wait(timeout=2.0)
    controller.close(timeout=2.0)

    assert audio.close_calls == 2
    statuses = controller.drain_statuses()
    audio_statuses = [status for status in statuses if status.kind == "audio_error"]
    assert len(audio_statuses) == 1
    assert vars(audio_statuses[0]) == {
        "kind": "audio_error",
        "operation": "close_stop",
        "side": Side.P2,
        "issue": AudioIssue.BACKEND_FAILURE,
        "capture_status": None,
        "capture_issue": None,
    }


def test_controller_exposes_typed_runtime_diagnostics_without_raw_dependencies() -> None:
    from sf6_music_trigger.controller import ListenerController, TrackerDiagnostics

    sample = _running_frame_zero()
    source = _Source(lambda: [sample])
    engine = _Engine()
    engine.outcome_statuses = ("命中", "低等级已过滤")
    audio = _Audio()
    controller = ListenerController(
        source=source,
        engine=engine,
        audio_port=audio,
        diagnostics_provider=lambda: TrackerDiagnostics(
            (TrackerState.ARMED, TrackerState.COOLDOWN),
            (2, None),
            processing_reason="ok",
            viewport_status="normalized",
        ),
    )

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    assert controller.capture_status is CaptureStatus.RUNNING
    assert controller.current_geometry == sample.frame.geometry
    assert controller.tracker_states == (TrackerState.ARMED, TrackerState.COOLDOWN)
    assert controller.tracker_levels == (2, None)
    assert controller.processing_reason == "ok"
    assert controller.viewport_status == "normalized"
    assert controller.outcome_statuses == ("命中", "低等级已过滤")
    assert "private" not in repr(controller.outcome_statuses).lower()


@pytest.mark.parametrize(
    ("side", "plain_status"),
    [
        (Side.P1, "疑似大招，正在确认"),
        (Side.P2, "疑似大招已撤销"),
    ],
)
def test_controller_rejects_provisional_labels_from_plain_engine_outcomes(
    side: Side,
    plain_status: str,
) -> None:
    source = _Source(lambda: [_running_frame_zero()])
    engine = _Engine()
    engine.outcome_statuses = (
        (plain_status, None) if side is Side.P1 else (None, plain_status)
    )
    controller = _controller(source, engine, _Audio())

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    assert controller.outcome_statuses == (None, None)
    assert any(
        status.kind == "worker_error" for status in controller.drain_statuses()
    )


def test_controller_latches_real_candidate_update_across_an_empty_update() -> None:
    candidate = _candidate(Side.P1, "candidate-p1")
    source = _Source(lambda: [_running_frame_zero(), _running_frame_zero(2.0)])
    engine = _Engine(
        updates=[EngineUpdate(candidates=(candidate,)), EngineUpdate()],
    )
    controller = _controller(source, engine, _Audio())

    assert controller.start()
    engine.wait_for_samples(2)

    _wait_for_outcome_statuses(controller, ("疑似大招，正在确认", None))
    controller.request_stop()
    controller.join(timeout=2.0)
    assert controller.outcome_statuses == (None, None)


def test_controller_latches_matching_real_cancellation_across_empty_update() -> None:
    candidate = _candidate(Side.P2, "candidate-p2")
    cancellation = _cancellation(Side.P2, candidate.candidate_id)
    source = _Source(
        lambda: [
            _running_frame_zero(),
            _running_frame_zero(2.0),
            _running_frame_zero(3.0),
        ]
    )
    engine = _Engine(
        updates=[
            EngineUpdate(candidates=(candidate,)),
            EngineUpdate(cancellations=(cancellation,)),
            EngineUpdate(),
        ],
    )
    controller = _controller(source, engine, _Audio())

    assert controller.start()
    engine.wait_for_samples(3)

    _wait_for_outcome_statuses(controller, (None, "疑似大招已撤销"))
    controller.request_stop()
    controller.join(timeout=2.0)


def test_matching_real_confirmation_restores_engine_outcome_status() -> None:
    candidate = _candidate(Side.P1, "candidate-confirmed")
    confirmation = _confirmation(Side.P1, candidate.candidate_id)
    source = _Source(lambda: [_running_frame_zero(), _running_frame_zero(2.0)])
    engine = _Engine(
        updates=[
            EngineUpdate(candidates=(candidate,)),
            EngineUpdate(events=(confirmation,)),
        ],
        outcome_status_updates=[
            (None, None),
            ("等待大招结果", None),
        ],
    )
    controller = _controller(source, engine, _Audio())

    assert controller.start()
    engine.wait_for_samples(2)

    _wait_for_outcome_statuses(controller, ("等待大招结果", None))
    controller.request_stop()
    controller.join(timeout=2.0)


def test_active_candidate_ignores_legacy_confirmation_without_candidate_id() -> None:
    candidate = _candidate(Side.P1, "candidate-active")
    legacy_confirmation = ConfirmedSuperEvent(
        event_id="legacy-confirmation",
        side=Side.P1,
        confirmed_at=1.5,
        pre_spend_frame_id=1,
        confirmed_frame_id=3,
        profile_id="controller-profile",
        meter_before=3,
        meter_after=0,
        levels_spent=3,
        candidate_id=None,
    )
    source = _Source(lambda: [_running_frame_zero(), _running_frame_zero(2.0)])
    engine = _Engine(
        updates=[
            EngineUpdate(candidates=(candidate,)),
            EngineUpdate(events=(legacy_confirmation,)),
        ],
        outcome_status_updates=[
            (None, None),
            ("等待大招结果", None),
        ],
    )
    controller = _controller(source, engine, _Audio())

    assert controller.start()
    engine.wait_for_samples(2)

    _wait_for_outcome_statuses(controller, ("疑似大招，正在确认", None))
    controller.request_stop()
    controller.join(timeout=2.0)


def test_active_candidate_ignores_mismatched_candidate_confirmation() -> None:
    candidate = _candidate(Side.P2, "candidate-active-p2")
    mismatch = _confirmation(Side.P2, "different-candidate")
    source = _Source(lambda: [_running_frame_zero(), _running_frame_zero(2.0)])
    engine = _Engine(
        updates=[
            EngineUpdate(candidates=(candidate,)),
            EngineUpdate(events=(mismatch,)),
        ],
        outcome_status_updates=[
            (None, None),
            (None, "等待大招结果"),
        ],
    )
    controller = _controller(source, engine, _Audio())

    assert controller.start()
    engine.wait_for_samples(2)

    _wait_for_outcome_statuses(controller, (None, "疑似大招，正在确认"))
    controller.request_stop()
    controller.join(timeout=2.0)


def test_legacy_confirmation_without_active_candidate_keeps_compatibility() -> None:
    candidate = _candidate(Side.P1, "candidate-cancelled-before-legacy")
    cancellation = _cancellation(Side.P1, candidate.candidate_id)
    legacy_confirmation = ConfirmedSuperEvent(
        event_id="legacy-without-active",
        side=Side.P1,
        confirmed_at=1.5,
        pre_spend_frame_id=1,
        confirmed_frame_id=3,
        profile_id="controller-profile",
        candidate_id=None,
    )
    source = _Source(
        lambda: [
            _running_frame_zero(),
            _running_frame_zero(2.0),
            _running_frame_zero(3.0),
        ]
    )
    engine = _Engine(
        updates=[
            EngineUpdate(candidates=(candidate,)),
            EngineUpdate(cancellations=(cancellation,)),
            EngineUpdate(events=(legacy_confirmation,)),
        ],
        outcome_status_updates=[
            (None, None),
            (None, None),
            ("等待大招结果", None),
        ],
    )
    controller = _controller(source, engine, _Audio())

    assert controller.start()
    engine.wait_for_samples(3)

    _wait_for_outcome_statuses(controller, ("等待大招结果", None))
    controller.request_stop()
    controller.join(timeout=2.0)


def test_controller_normalizes_legacy_state_only_diagnostics_to_unknown_levels() -> None:
    from sf6_music_trigger.controller import ListenerController

    source = _Source(lambda: [_running_frame_zero()])
    engine = _Engine()
    controller = ListenerController(
        source=source,
        engine=engine,
        audio_port=_Audio(),
        diagnostics_provider=lambda: (TrackerState.ARMED, TrackerState.DISARMED),
    )

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)

    assert controller.tracker_states == (TrackerState.ARMED, TrackerState.DISARMED)
    assert controller.tracker_levels == (None, None)


def test_controller_restart_clears_stale_tracker_levels_before_next_sample() -> None:
    from sf6_music_trigger.controller import ListenerController, TrackerDiagnostics

    sessions = iter(([_running_frame_zero()], []))
    source = _Source(lambda: next(sessions))
    engine = _Engine()
    controller = ListenerController(
        source=source,
        engine=engine,
        audio_port=_Audio(),
        diagnostics_provider=lambda: TrackerDiagnostics(
            (TrackerState.ARMED, TrackerState.DISARMED),
            (3, 0),
        ),
    )

    assert controller.start()
    engine.wait_for_samples(1)
    controller.request_stop()
    controller.join(timeout=2.0)
    assert controller.tracker_levels == (3, 0)

    assert controller.start()
    assert source.read_entered.wait(timeout=2.0)
    assert controller.tracker_states is None
    assert controller.tracker_levels is None
    controller.request_stop()
    controller.join(timeout=2.0)
