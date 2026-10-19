"""Cooperative listener lifecycle without UI or backend initialization side effects."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from queue import Empty, Queue
import threading
from typing import Callable, Literal

from .audio import AudioIssue, AudioPlaybackError
from .capture import CaptureStartError
from .domain import (
    CaptureIssue,
    CaptureSample,
    CaptureStatus,
    ClientGeometry,
    Side,
    TrackerState,
)
from .engine import EngineUpdate, VisualTriggerEngine


ControllerStatusKind = Literal[
    "started",
    "stopped",
    "worker_error",
    "audio_error",
    "diagnostic_error",
    "capture_waiting",
]
_SAFE_OUTCOME_STATUSES = frozenset(
    {
        "等待大招结果",
        "后续动画",
        "命中",
        "格挡",
        "打空",
        "格挡/未命中",
        "低等级已淡出",
        "低等级已过滤",
        "结果不确定",
        "等待淡出",
        "淡出中",
        "淡出失败",
    }
)
_SAFE_PROCESSING_REASONS = frozenset(
    {
        "waiting",
        "ok",
        "failed",
        "not_running",
        "stale",
        "viewport",
        "geometry",
        "hud_hidden",
        "hud_untrusted",
    }
)
_SAFE_VIEWPORT_STATUSES = frozenset(
    {"native", "confirming", "normalized", "unsupported", "too_small"}
)


@dataclass(frozen=True)
class ControllerStatus:
    kind: ControllerStatusKind
    operation: str | None = None
    side: Side | None = None
    issue: AudioIssue | None = None
    capture_status: CaptureStatus | None = None
    capture_issue: CaptureIssue | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not str:
            raise TypeError("kind must be a string")
        if self.kind not in {
            "started",
            "stopped",
            "worker_error",
            "audio_error",
            "diagnostic_error",
            "capture_waiting",
        }:
            raise ValueError("unsupported controller status kind")
        if self.kind == "audio_error":
            if type(self.operation) is not str or not self.operation:
                raise TypeError("audio error status requires an operation")
            if self.side is not None and not isinstance(self.side, Side):
                raise TypeError("side must be Side or None")
            if not isinstance(self.issue, AudioIssue):
                raise TypeError("audio error status requires AudioIssue")
            if self.capture_status is not None or self.capture_issue is not None:
                raise ValueError("audio error status cannot carry capture fields")
            return
        if self.kind == "capture_waiting":
            if self.operation is not None or self.side is not None or self.issue is not None:
                raise ValueError("capture waiting status cannot carry audio fields")
            if self.capture_status in {CaptureStatus.MINIMIZED, CaptureStatus.OCCLUDED}:
                if self.capture_issue is not None:
                    raise ValueError("window-state waits cannot carry a capture issue")
                return
            if self.capture_status is CaptureStatus.LOST:
                if not isinstance(self.capture_issue, CaptureIssue):
                    raise TypeError("lost capture waits require CaptureIssue")
                return
            raise ValueError("capture waiting status requires a recoverable status")
        if (
            self.operation is not None
            or self.side is not None
            or self.issue is not None
            or self.capture_status is not None
            or self.capture_issue is not None
        ):
            raise ValueError("non-audio status must not carry audio fields")


@dataclass(frozen=True)
class TrackerDiagnostics:
    """Privacy-safe tracker state and numeric meter levels for the panel."""

    states: tuple[TrackerState, TrackerState]
    levels: tuple[int | None, int | None]
    processing_reason: str = "waiting"
    viewport_status: str = "native"

    def __post_init__(self) -> None:
        if (
            type(self.states) is not tuple
            or len(self.states) != 2
            or not all(isinstance(state, TrackerState) for state in self.states)
        ):
            raise TypeError("states must contain two TrackerState values")
        if type(self.levels) is not tuple or len(self.levels) != 2:
            raise TypeError("levels must contain two meter levels")
        for level in self.levels:
            if level is None:
                continue
            if type(level) is not int:
                raise TypeError("meter level must be an integer or None")
            if not 0 <= level <= 3:
                raise ValueError("meter level must be between zero and three")
        if self.processing_reason not in _SAFE_PROCESSING_REASONS:
            raise ValueError("processing_reason must be a safe diagnostic code")
        if self.viewport_status not in _SAFE_VIEWPORT_STATUSES:
            raise ValueError("viewport_status must be a safe diagnostic code")


class ListenerController:
    """Own one restartable non-daemon listener worker and its cooperative stop."""

    def __init__(
        self,
        *,
        source: object,
        engine: object,
        audio_port: object,
        diagnostics_provider: Callable[
            [], TrackerDiagnostics | tuple[TrackerState, TrackerState]
        ]
        | None = None,
        diagnostic_log: object | None = None,
        start_retry_seconds: Real = 1.0,
        runtime_message_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._require_method(source, "start", "source")
        self._require_method(source, "read", "source")
        self._require_method(source, "stop", "source")
        self._require_method(engine, "reset", "engine")
        self._require_method(engine, "process", "engine")
        self._require_method(audio_port, "close", "audio_port")
        if diagnostics_provider is not None and not callable(diagnostics_provider):
            raise TypeError("diagnostics_provider must be callable")
        if runtime_message_provider is not None and not callable(runtime_message_provider):
            raise TypeError("runtime_message_provider must be callable")
        if diagnostic_log is not None:
            self._require_method(diagnostic_log, "start", "diagnostic_log")
            self._require_method(diagnostic_log, "close", "diagnostic_log")
        if isinstance(start_retry_seconds, bool) or not isinstance(
            start_retry_seconds, Real
        ):
            raise TypeError("start_retry_seconds must be a real number")
        retry_seconds = float(start_retry_seconds)
        if not math.isfinite(retry_seconds) or retry_seconds <= 0:
            raise ValueError("start_retry_seconds must be finite and positive")

        self._source = source
        self._engine = engine
        self._audio_port = audio_port
        self._diagnostics_provider = diagnostics_provider
        self._runtime_message_provider = runtime_message_provider
        self._runtime_message: str | None = None
        self._diagnostic_log = diagnostic_log
        self._start_retry_seconds = retry_seconds
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._status_queue: Queue[ControllerStatus] = Queue()
        self._thread: threading.Thread | None = None
        self._capture_status: CaptureStatus | None = None
        self._current_geometry: ClientGeometry | None = None
        self._tracker_states: tuple[TrackerState, TrackerState] | None = None
        self._tracker_levels: tuple[int | None, int | None] | None = None
        self._processing_reason: str | None = None
        self._viewport_status: str | None = None
        self._outcome_statuses: tuple[str | None, str | None] = (None, None)
        self._active_candidate_ids: dict[Side, str | None] = {
            Side.P1: None,
            Side.P2: None,
        }
        self._provisional_outcome_statuses: dict[Side, str | None] = {
            Side.P1: None,
            Side.P2: None,
        }
        self._diagnostic_error_reported = False

    @property
    def worker_thread(self) -> threading.Thread | None:
        with self._lock:
            return self._thread

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def capture_status(self) -> CaptureStatus | None:
        with self._lock:
            return self._capture_status

    @property
    def current_geometry(self) -> ClientGeometry | None:
        with self._lock:
            return self._current_geometry

    @property
    def tracker_states(self) -> tuple[TrackerState, TrackerState] | None:
        with self._lock:
            return self._tracker_states

    @property
    def tracker_levels(self) -> tuple[int | None, int | None] | None:
        with self._lock:
            return self._tracker_levels

    @property
    def processing_reason(self) -> str | None:
        with self._lock:
            return self._processing_reason

    @property
    def viewport_status(self) -> str | None:
        with self._lock:
            return self._viewport_status

    @property
    def outcome_statuses(self) -> tuple[str | None, str | None]:
        with self._lock:
            return self._outcome_statuses

    @property
    def engine(self) -> object:
        return self._engine

    @property
    def runtime_message(self) -> str | None:
        """Backend-specific status text (REFramework backend only)."""

        with self._lock:
            return self._runtime_message

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._capture_status = None
            self._current_geometry = None
            self._tracker_states = None
            self._tracker_levels = None
            self._processing_reason = None
            self._viewport_status = None
            self._outcome_statuses = (None, None)
            self._runtime_message = None
            self._diagnostic_error_reported = False
            self._clear_provisional_outcomes_locked()
            worker = threading.Thread(
                target=self._worker,
                name="sf6-listener",
                daemon=False,
            )
            self._thread = worker
            worker.start()
            return True

    def request_stop(self) -> bool:
        self._stop_event.set()
        try:
            self._source.stop()
        except Exception:
            self._status_queue.put(ControllerStatus(kind="worker_error"))
        return True

    def join(self, timeout: float | None = None) -> None:
        with self._lock:
            worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout)

    def close(self, timeout: float | None = None) -> None:
        self.request_stop()
        self.join(timeout=timeout)
        self._close_audio()

    def drain_statuses(self) -> tuple[ControllerStatus, ...]:
        statuses: list[ControllerStatus] = []
        while True:
            try:
                statuses.append(self._status_queue.get_nowait())
            except Empty:
                return tuple(statuses)

    def _worker(self) -> None:
        session_initialized = False
        try:
            self._start_diagnostic_log()
            if not self._start_source_until_ready():
                return
            self._engine.reset("capture session started")
            session_initialized = True
            self._status_queue.put(ControllerStatus(kind="started"))
            while not self._stop_event.is_set():
                self._capture_latency("capture_begin")
                try:
                    sample = self._source.read()
                finally:
                    self._capture_latency("capture_end")
                if self._stop_event.is_set():
                    break
                self._record_capture_sample(sample)
                engine_update = self._engine.process(sample)
                self._record_diagnostic_warning()
                self._record_tracker_states()
                self._record_outcome_statuses(engine_update)
                self._record_runtime_message()
        except Exception:
            if not self._stop_event.is_set():
                self._status_queue.put(ControllerStatus(kind="worker_error"))
        finally:
            try:
                self._source.stop()
            except Exception:
                self._status_queue.put(ControllerStatus(kind="worker_error"))
            if session_initialized:
                try:
                    self._engine.reset("listener stopped")
                except Exception:
                    self._status_queue.put(ControllerStatus(kind="worker_error"))
                finally:
                    self._clear_provisional_outcomes()
            self._close_audio()
            self._close_diagnostic_log()
            self._status_queue.put(ControllerStatus(kind="stopped"))

    def _start_source_until_ready(self) -> bool:
        while not self._stop_event.is_set():
            try:
                self._source.start()
                return True
            except CaptureStartError as error:
                with self._lock:
                    self._capture_status = error.status
                self._status_queue.put(
                    ControllerStatus(
                        kind="capture_waiting",
                        capture_status=error.status,
                        capture_issue=error.issue,
                    )
                )
                if self._stop_event.wait(self._start_retry_seconds):
                    return False
        return False

    def _capture_latency(self, method: str) -> None:
        if isinstance(self._engine, VisualTriggerEngine):
            try:
                getattr(self._engine, method)()
            except Exception:
                self._report_diagnostic_error_once()

    def _record_capture_sample(self, sample: object) -> None:
        if not isinstance(sample, CaptureSample):
            return
        with self._lock:
            self._capture_status = sample.status
            if sample.status is CaptureStatus.RUNNING and sample.frame is not None:
                self._current_geometry = sample.frame.geometry

    def _record_tracker_states(self) -> None:
        provider = self._diagnostics_provider
        if provider is None:
            return
        provided = provider()
        if isinstance(provided, TrackerDiagnostics):
            diagnostics = provided
        else:
            diagnostics = TrackerDiagnostics(provided, (None, None))
        with self._lock:
            self._tracker_states = diagnostics.states
            self._tracker_levels = diagnostics.levels
            self._processing_reason = diagnostics.processing_reason
            self._viewport_status = diagnostics.viewport_status

    def _record_outcome_statuses(self, update: object) -> None:
        if not isinstance(update, EngineUpdate):
            raise TypeError("engine process must return EngineUpdate")
        statuses = getattr(self._engine, "outcome_statuses", (None, None))
        if type(statuses) is not tuple or len(statuses) != 2:
            raise TypeError("engine outcome_statuses must be a two-item tuple")
        if not all(
            item is None or (type(item) is str and item in _SAFE_OUTCOME_STATUSES)
            for item in statuses
        ):
            raise ValueError("engine outcome_statuses must contain only safe labels")
        with self._lock:
            for candidate in update.candidates:
                self._active_candidate_ids[candidate.side] = candidate.candidate_id
                self._provisional_outcome_statuses[candidate.side] = (
                    "疑似大招，正在确认"
                )
            for event in update.events:
                active_id = self._active_candidate_ids[event.side]
                if active_id is not None and event.candidate_id != active_id:
                    continue
                if active_id is None and event.candidate_id is not None:
                    continue
                self._active_candidate_ids[event.side] = None
                self._provisional_outcome_statuses[event.side] = None
            for cancellation in update.cancellations:
                if (
                    cancellation.candidate_id
                    != self._active_candidate_ids[cancellation.side]
                ):
                    continue
                self._active_candidate_ids[cancellation.side] = None
                self._provisional_outcome_statuses[cancellation.side] = (
                    "疑似大招已撤销"
                )
            self._outcome_statuses = tuple(
                self._provisional_outcome_statuses[side] or statuses[index]
                for index, side in enumerate((Side.P1, Side.P2))
            )

    def _record_runtime_message(self) -> None:
        provider = self._runtime_message_provider
        if provider is None:
            return
        message = provider()
        if message is not None and type(message) is not str:
            raise TypeError("runtime message must be text or None")
        with self._lock:
            self._runtime_message = message

    def _clear_provisional_outcomes(self) -> None:
        with self._lock:
            self._clear_provisional_outcomes_locked()

    def _clear_provisional_outcomes_locked(self) -> None:
        outcomes = list(self._outcome_statuses)
        for index, side in enumerate((Side.P1, Side.P2)):
            if self._provisional_outcome_statuses[side] is not None:
                outcomes[index] = None
            self._active_candidate_ids[side] = None
            self._provisional_outcome_statuses[side] = None
        self._outcome_statuses = tuple(outcomes)

    def _close_audio(self) -> None:
        try:
            self._audio_port.close()
        except AudioPlaybackError as error:
            self._status_queue.put(
                ControllerStatus(
                    kind="audio_error",
                    operation=error.operation,
                    side=error.side,
                    issue=error.issue,
                )
            )
        except Exception:
            self._status_queue.put(ControllerStatus(kind="worker_error"))

    def _start_diagnostic_log(self) -> None:
        diagnostic_log = self._diagnostic_log
        if diagnostic_log is None:
            return
        try:
            diagnostic_log.start()
        except Exception:
            self._report_diagnostic_error_once()
            return
        self._record_writer_warning()

    def _close_diagnostic_log(self) -> None:
        diagnostic_log = self._diagnostic_log
        if diagnostic_log is None:
            return
        try:
            diagnostic_log.close()
        except Exception:
            self._report_diagnostic_error_once()
            return
        self._record_writer_warning()

    def _record_diagnostic_warning(self) -> None:
        try:
            engine_warning = getattr(self._engine, "diagnostic_warning", False)
            if type(engine_warning) is not bool:
                raise TypeError("engine diagnostic_warning must be a boolean")
            if engine_warning:
                self._report_diagnostic_error_once()
            self._record_writer_warning()
        except Exception:
            self._report_diagnostic_error_once()

    def _record_writer_warning(self) -> None:
        diagnostic_log = self._diagnostic_log
        if diagnostic_log is None:
            return
        try:
            warning = diagnostic_log.has_warning
            if type(warning) is not bool:
                raise TypeError("diagnostic_log has_warning must be a boolean")
        except Exception:
            self._report_diagnostic_error_once()
            return
        if warning:
            self._report_diagnostic_error_once()

    def _report_diagnostic_error_once(self) -> None:
        with self._lock:
            if self._diagnostic_error_reported:
                return
            self._diagnostic_error_reported = True
        self._status_queue.put(ControllerStatus(kind="diagnostic_error"))

    @staticmethod
    def _require_method(value: object, method: str, name: str) -> None:
        if not callable(getattr(value, method, None)):
            raise TypeError(f"{name} must provide {method}")
