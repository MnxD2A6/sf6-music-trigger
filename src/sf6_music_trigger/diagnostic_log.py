"""Bounded, privacy-safe JSONL diagnostics for trigger decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Real
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import secrets
import threading
import time
from typing import TextIO

from .audio import AudioKind
from .config import SuperScope
from .domain import CandidateCancelReason, Side, TrackerState
from .outcome import OutcomeLabel


_SCHEMA_VERSION = 1
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_TRANSITIONS = frozenset({"CANDIDATE", "CONFIRMED", "CANCELLED", "OUTCOME", "FADE"})
_EVIDENCE_CODES = frozenset(
    {
        "stable_downward_level",
        "playback_hold_elapsed",
        "fade_failed",
        "opponent_health_changed",
        "opponent_health_unchanged",
        "opponent_drive_changed",
        "opponent_drive_unchanged",
        "timer_resumed",
        "hud_resumed",
        "training_result_changed",
        "contact_checkpoint_elapsed",
        "contact_window_elapsed",
        "contact_not_established",
        "hud_unavailable_at_deadline",
        "resolution_deadline_elapsed",
        "remaining_meter_0",
        "remaining_meter_1",
        "remaining_meter_2",
        "remaining_meter_3",
        "reset",
        "unrecognized",
        *(reason.value for reason in CandidateCancelReason),
    }
)
_SENTINEL = object()
_CLOSE_TIMEOUT_SECONDS = 1.0
_SENTINEL_RETRY_SECONDS = 0.01


def _safe_identifier(name: str, value: object) -> str:
    if type(value) is not str or _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a privacy-safe identifier")
    return value


def _meter_level(name: str, value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer or None")
    if not 0 <= value <= 3:
        raise ValueError(f"{name} must be between zero and three")
    return value


@dataclass(frozen=True)
class DiagnosticRecord:
    """Closed metadata emitted after a trigger routing decision is complete."""

    monotonic_time: Real
    side: Side
    identity: str
    transition: str
    evidence: tuple[str, ...]
    meter_before: int | None
    meter_after: int | None
    levels_spent: int | None
    scope: SuperScope
    audio_actions: tuple[AudioKind, ...]
    outcome: OutcomeLabel | None

    def __post_init__(self) -> None:
        if isinstance(self.monotonic_time, bool) or not isinstance(
            self.monotonic_time, Real
        ):
            raise TypeError("monotonic_time must be a real number")
        if not math.isfinite(self.monotonic_time) or self.monotonic_time < 0:
            raise ValueError("monotonic_time must be finite and nonnegative")
        if not isinstance(self.side, Side):
            raise TypeError("side must be Side")
        _safe_identifier("identity", self.identity)
        if self.transition not in _TRANSITIONS:
            raise ValueError("transition must be a closed lifecycle code")
        if type(self.evidence) is not tuple:
            raise TypeError("evidence must be a tuple")
        if any(type(code) is not str or code not in _EVIDENCE_CODES for code in self.evidence):
            raise ValueError("evidence must contain only closed evidence codes")

        meter_before = _meter_level("meter_before", self.meter_before)
        meter_after = _meter_level("meter_after", self.meter_after)
        levels_spent = _meter_level("levels_spent", self.levels_spent)
        if levels_spent is not None:
            if meter_before is None or meter_after is None:
                raise ValueError("levels_spent requires before and after meter levels")
            if levels_spent != meter_before - meter_after:
                raise ValueError("levels_spent must equal the meter delta")
        if not isinstance(self.scope, SuperScope):
            raise TypeError("scope must be SuperScope")
        if type(self.audio_actions) is not tuple or not all(
            isinstance(kind, AudioKind) for kind in self.audio_actions
        ):
            raise TypeError("audio_actions must contain only AudioKind values")
        if self.outcome is not None and not isinstance(self.outcome, OutcomeLabel):
            raise TypeError("outcome must be OutcomeLabel or None")


@dataclass(frozen=True)
class LatencyRecord:
    """Separate schema: host perf_counter times and capture source time, no media.

    Missing values mean unavailable, never a zero reading. Identity is the same
    bounded anonymous lifecycle ID used by the legacy event log.
    """

    phase: str
    host_time: float
    cycle_id: int = 0
    frame_id: int | None = None
    source_time: float | None = None
    capture_begin: float | None = None
    capture_end: float | None = None
    process_begin: float | None = None
    side: Side | None = None
    meter_level: int | None = None
    meter_level_confidence: float | None = None
    normalized_meter: float | None = None
    spent_confidence: float | None = None
    hud_visible: bool | None = None
    level_supported: bool | None = None
    structure_verified: bool | None = None
    baseline_before: int | None = None
    baseline_after: int | None = None
    state: TrackerState | None = None
    audio_call_id: int | None = None
    audio_actions: tuple[AudioKind, ...] = ()
    identity: str | None = None
    transition: str | None = None
    reason: str = "ok"

    def __post_init__(self) -> None:
        if self.phase not in {"CAPTURE_BEGIN", "CAPTURE_END", "PROCESS_BEGIN",
                              "PROCESS_END", "METER", "AUDIO_BEGIN", "AUDIO_END", "TRANSITION"}:
            raise ValueError("unsupported latency phase")
        if self.reason not in {"ok", "failed", "not_running", "stale", "geometry",
                               "viewport", "hud_hidden", "hud_untrusted"}:
            raise ValueError("unsupported latency reason")
        for name in ("host_time", "source_time", "capture_begin", "capture_end", "process_begin",
                     "meter_level_confidence", "normalized_meter", "spent_confidence"):
            value = getattr(self, name)
            if value is None and name != "host_time":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("latency scalar must be finite and nonnegative")
            if name in {"meter_level_confidence", "normalized_meter", "spent_confidence"} and value > 1:
                raise ValueError("confidence and fill must be normalized")
        for name in ("cycle_id", "frame_id", "audio_call_id"):
            value = getattr(self, name)
            if value is None and name != "cycle_id":
                continue
            if type(value) is not int or not 0 <= value <= 2**63 - 1:
                raise ValueError("latency identifier must be a bounded integer")
        for name in ("meter_level", "baseline_before", "baseline_after"):
            _meter_level(name, getattr(self, name))
        for name in ("hud_visible", "level_supported", "structure_verified"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError("latency flag must be boolean")
        if self.side is not None and not isinstance(self.side, Side):
            raise TypeError("side must be Side")
        if self.state is not None and not isinstance(self.state, TrackerState):
            raise TypeError("state must be TrackerState")
        if self.identity is not None:
            _safe_identifier("identity", self.identity)
        if self.transition is not None and self.transition not in _TRANSITIONS:
            raise ValueError("unsupported transition")
        if type(self.audio_actions) is not tuple or len(self.audio_actions) > 8 or not all(
            isinstance(kind, AudioKind) for kind in self.audio_actions
        ):
            raise ValueError("audio actions must be bounded enum values")


class DiagnosticEventLog:
    """Write fixed-shape records asynchronously without blocking capture."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        session_id: str | None = None,
        max_bytes: int = 2 * 1024 * 1024,
        retained_files: int = 5,
        queue_capacity: int = 1024,
    ) -> None:
        root_text = os.fspath(root)
        if type(root_text) is not str or not root_text:
            raise TypeError("root must resolve to nonempty text")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        if type(retained_files) is not int or retained_files < 1:
            raise ValueError("retained_files must be a positive integer")
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")

        from . import BUILD_ID
        actual_session_id = session_id or f"{BUILD_ID}-{secrets.token_hex(8)}"
        _safe_identifier("session_id", actual_session_id)
        self._directory = Path(root_text) / "diagnostics"
        self._session_id = actual_session_id
        self._max_bytes = max_bytes
        self._retained_files = retained_files
        self._queue_capacity = queue_capacity
        self._queue: Queue[DiagnosticRecord | object] = Queue(
            maxsize=queue_capacity
        )
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._warning = False
        self._dropped_count = 0
        self._worker: threading.Thread | None = None
        self._stream: TextIO | None = None
        self._stream_bytes = 0
        self._file_sequence = 0
        self._latency: LatencyEventLog | None = None

    def latency_log(self) -> LatencyEventLog:
        """Attach before start; independent retention preserves legacy consumers."""
        if self._latency is None:
            self._latency = LatencyEventLog(
                self._directory.parent, session_id=self._session_id,
                max_bytes=self._max_bytes, retained_files=self._retained_files,
                queue_capacity=min(self._queue_capacity, 256),
            )
        return self._latency

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def has_warning(self) -> bool:
        with self._lock:
            return self._warning or (self._latency is not None and self._latency.has_warning)

    def start(self) -> None:
        """Open a session file and start one writer thread for this run."""

        if self._latency is not None:
            self._latency.start()
        with self._lock:
            if self._started:
                return
            if self._closed:
                self._queue = Queue(maxsize=self._queue_capacity)
                self._closed = False
            self._started = True
            self._prepare_stream()
            worker = threading.Thread(
                target=self._run_worker,
                name="sf6-diagnostic-log",
                daemon=True,
            )
            self._worker = worker
            try:
                worker.start()
            except Exception:
                self._warning = True
                self._worker = None
                self._started = False
                self._closed = True
                self._close_stream()

    def record(self, record: DiagnosticRecord) -> None:
        """Enqueue a record immediately, or count a deterministic drop."""

        if not isinstance(record, DiagnosticRecord):
            raise TypeError("record must be DiagnosticRecord")
        with self._lock:
            if self._closed:
                self._drop_locked()
                return
            try:
                self._queue.put_nowait(record)
            except Full:
                self._drop_locked()

    def close(self) -> None:
        """Drain the queue and close the writer; repeated calls do nothing."""

        self._close_with_deadline(time.monotonic() + _CLOSE_TIMEOUT_SECONDS)

    def _close_with_deadline(self, deadline: float) -> None:
        """Both writers share the existing one-second listener close budget."""
        if self._latency is not None:
            self._latency._close_with_deadline(deadline)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
            worker = self._worker
        if not started:
            self._discard_unstarted_records()
            return
        if worker is None:
            self._mark_warning()
            self._close_stream()
            return

        sentinel_sent = False
        while not sentinel_sent:
            try:
                self._queue.put_nowait(_SENTINEL)
                sentinel_sent = True
            except Full:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._mark_warning()
                    break
                time.sleep(min(_SENTINEL_RETRY_SECONDS, remaining))
            except Exception:
                self._mark_warning()
                break

        try:
            worker.join(max(0.0, deadline - time.monotonic()))
            worker_alive = worker.is_alive()
        except Exception:
            self._mark_warning()
            return
        if not sentinel_sent or worker_alive:
            self._mark_warning()
        if not worker_alive:
            with self._lock:
                if self._worker is worker:
                    self._worker = None
                    self._started = False

    def _drop_locked(self) -> None:
        self._dropped_count += 1
        self._warning = True

    def _mark_warning(self) -> None:
        with self._lock:
            self._warning = True

    def _discard_unstarted_records(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def _prepare_stream(self) -> None:
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._open_next_stream()
            self._prune_old_files()
        except Exception:
            self._warning = True
            self._close_stream()

    def _run_worker(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    return
                if isinstance(item, (DiagnosticRecord, LatencyRecord)):
                    self._write_record(item)
                else:
                    self._mark_warning()
        finally:
            self._close_stream()

    def _write_record(self, record: DiagnosticRecord) -> None:
        stream = self._stream
        if stream is None:
            return
        line = json.dumps(
            ({"schema_version": 1, "record_type": "latency", **asdict(record)}
             if isinstance(record, LatencyRecord) else {
                "schema_version": _SCHEMA_VERSION,
                "monotonic_time": float(record.monotonic_time),
                "side": record.side.value,
                "identity": record.identity,
                "transition": record.transition,
                "evidence": list(record.evidence),
                "meter_before": record.meter_before,
                "meter_after": record.meter_after,
                "levels_spent": record.levels_spent,
                "scope": record.scope.value,
                "audio_actions": [kind.value for kind in record.audio_actions],
                "outcome": None if record.outcome is None else record.outcome.value,
            }),
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if self._stream_bytes and self._stream_bytes + line_bytes > self._max_bytes:
            if not self._rotate_stream():
                return

        try:
            stream = self._stream
            if stream is None:
                return
            stream.write(line)
            stream.flush()
            self._stream_bytes += line_bytes
        except Exception:
            self._mark_warning()
            self._close_stream()

    def _rotate_stream(self) -> bool:
        self._close_stream()
        try:
            self._open_next_stream()
            self._prune_old_files()
        except Exception:
            self._mark_warning()
            self._close_stream()
            return False
        return True

    def _open_next_stream(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        while True:
            path = self._directory / (
                f"{stamp}-{self._session_id}-{self._file_sequence:04d}.jsonl"
            )
            self._file_sequence += 1
            try:
                self._stream = path.open("x", encoding="utf-8", newline="\n")
            except FileExistsError:
                continue
            self._stream_bytes = 0
            return

    def _prune_old_files(self) -> None:
        files = sorted(
            (
                path
                for path in self._directory.iterdir()
                if path.is_file() and path.suffix == ".jsonl"
            ),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        for path in files[self._retained_files :]:
            path.unlink()

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        self._stream_bytes = 0
        if stream is None:
            return
        try:
            stream.flush()
        except Exception:
            self._mark_warning()
        try:
            stream.close()
        except Exception:
            self._mark_warning()


class LatencyEventLog(DiagnosticEventLog):
    """Dedicated bounded queue/files; a producer never waits for a writer lock."""

    def __init__(self, root: str | os.PathLike[str], **kwargs: object) -> None:
        super().__init__(root, **kwargs)
        self._directory = self._directory / "latency"

    def record(self, record: LatencyRecord) -> None:
        if type(record) is not LatencyRecord:
            raise TypeError("record must be LatencyRecord")
        if not self._lock.acquire(blocking=False):
            # Single listener producer; best-effort counters must not wait on IO.
            self._dropped_count += 1
            self._warning = True
            return
        try:
            if self._closed:
                self._drop_locked()
                return
            try:
                self._queue.put_nowait(record)
            except Full:
                self._drop_locked()
        finally:
            self._lock.release()
