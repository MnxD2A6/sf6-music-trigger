from __future__ import annotations

import json
import inspect
from pathlib import Path
import subprocess
import sys
import threading
import textwrap
import time

import pytest

import sf6_music_trigger.diagnostic_log as diagnostic_log
from sf6_music_trigger.audio import AudioKind
from sf6_music_trigger.config import SuperScope
from sf6_music_trigger.diagnostic_log import DiagnosticEventLog, DiagnosticRecord
from sf6_music_trigger.domain import Side
from sf6_music_trigger.outcome import OutcomeLabel


def _record(
    *,
    identity: str = "cand-1",
    evidence: tuple[str, ...] = ("stable_downward_level",),
) -> DiagnosticRecord:
    return DiagnosticRecord(
        monotonic_time=1.25,
        side=Side.P1,
        identity=identity,
        transition="CANDIDATE",
        evidence=evidence,
        meter_before=3,
        meter_after=0,
        levels_spent=None,
        scope=SuperScope.SA3_CA_ONLY,
        audio_actions=(AudioKind.STOP, AudioKind.PLAY),
        outcome=None,
    )


def test_log_writes_fixed_privacy_safe_schema_on_close(tmp_path: Path) -> None:
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=8)
    log.start()
    log.record(_record())
    log.close()

    payload = json.loads(next((tmp_path / "diagnostics").glob("*.jsonl")).read_text())
    assert payload == {
        "schema_version": 1,
        "monotonic_time": 1.25,
        "side": "P1",
        "identity": "cand-1",
        "transition": "CANDIDATE",
        "evidence": ["stable_downward_level"],
        "meter_before": 3,
        "meter_after": 0,
        "levels_spent": None,
        "scope": "SA3_CA_ONLY",
        "audio_actions": ["stop", "play"],
        "outcome": None,
    }
    assert "track" not in payload
    assert "path" not in payload


def test_unstarted_bounded_queue_drops_without_blocking(tmp_path: Path) -> None:
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=1)
    log.record(_record(identity="one"))
    log.record(_record(identity="two"))

    assert log.dropped_count == 1
    assert log.has_warning is True
    log.close()


def test_no_contact_checkpoint_is_a_closed_diagnostic_evidence_code() -> None:
    record = _record(evidence=("contact_not_established",))
    assert record.evidence == ("contact_not_established",)


def test_training_hud_return_is_closed_non_hit_evidence() -> None:
    evidence = (
        "opponent_health_unchanged",
        "opponent_drive_unchanged",
        "hud_resumed",
    )
    assert _record(evidence=evidence).evidence == evidence


def test_rotation_retains_only_the_two_newest_files(tmp_path: Path) -> None:
    log = DiagnosticEventLog(
        tmp_path,
        session_id="abc123",
        max_bytes=200,
        retained_files=2,
        queue_capacity=8,
    )
    log.start()
    log.record(_record(identity="one"))
    log.record(_record(identity="two"))
    log.record(_record(identity="three"))
    log.close()

    files = sorted((tmp_path / "diagnostics").glob("*.jsonl"))
    assert len(files) == 2
    assert {json.loads(path.read_text())["identity"] for path in files} == {
        "two",
        "three",
    }
    assert not list(tmp_path.glob("*.jsonl"))


def test_record_serializes_closed_outcome_and_audio_metadata_only(
    tmp_path: Path,
) -> None:
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=8)
    log.start()
    log.record(
        DiagnosticRecord(
            monotonic_time=3.0,
            side=Side.P2,
            identity="P2-candidate-9",
            transition="OUTCOME",
            evidence=("opponent_health_changed",),
            meter_before=3,
            meter_after=0,
            levels_spent=3,
            scope=SuperScope.ALL_SUPERS,
            audio_actions=(AudioKind.FADE,),
            outcome=OutcomeLabel.HIT,
        )
    )
    log.close()

    serialized = next((tmp_path / "diagnostics").glob("*.jsonl")).read_text()
    payload = json.loads(serialized)
    assert payload["side"] == "P2"
    assert payload["identity"] == "P2-candidate-9"
    assert payload["audio_actions"] == ["fade"]
    assert payload["outcome"] == "HIT"
    assert "private-song" not in serialized
    assert "C:\\Users" not in serialized


@pytest.mark.parametrize(
    ("identity", "evidence"),
    [
        (r"C:\\Users\\name", ("stable_downward_level",)),
        ("cand-1", (r"C:\\captures\\frame.png",)),
        ("cand-1", ("private-song.mp3",)),
        ("cand-1", ("window title",)),
    ],
)
def test_record_rejects_open_text_that_could_leak_private_data(
    identity: str,
    evidence: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        _record(identity=identity, evidence=evidence)


def test_close_is_idempotent_and_record_after_close_fails_safe(tmp_path: Path) -> None:
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=2)
    log.start()
    log.record(_record(identity="before"))
    log.close()
    log.close()

    log.record(_record(identity="after"))

    assert log.dropped_count == 1
    assert log.has_warning is True
    payloads = [
        json.loads(line)
        for path in (tmp_path / "diagnostics").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [payload["identity"] for payload in payloads] == ["before"]


def test_log_can_start_close_and_write_two_real_sessions(tmp_path: Path) -> None:
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=2)

    for identity in ("round-one", "round-two"):
        log.start()
        worker = log._worker
        assert worker is not None and worker.daemon is True
        log.record(_record(identity=identity))
        log.close()

    payloads = [
        json.loads(line)
        for path in sorted((tmp_path / "diagnostics").glob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    assert [payload["identity"] for payload in payloads] == [
        "round-one",
        "round-two",
    ]


def test_permanently_blocked_write_cannot_keep_subprocess_alive(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        from pathlib import Path
        import sys
        import threading
        import time

        import sf6_music_trigger.diagnostic_log as module
        from sf6_music_trigger.diagnostic_log import DiagnosticEventLog, DiagnosticRecord
        from sf6_music_trigger.audio import AudioKind
        from sf6_music_trigger.config import SuperScope
        from sf6_music_trigger.domain import Side

        entered = threading.Event()
        class BlockingStream:
            def write(self, value):
                entered.set()
                threading.Event().wait()
            def flush(self):
                return None
            def close(self):
                return None

        Path.open = lambda *args, **kwargs: BlockingStream()
        module._CLOSE_TIMEOUT_SECONDS = 0.05
        log = DiagnosticEventLog(sys.argv[1], session_id="blocked", queue_capacity=2)
        log.start()
        log.record(DiagnosticRecord(
            1.0, Side.P1, "candidate", "CANDIDATE",
            ("stable_downward_level",), 3, 0, None,
            SuperScope.ALL_SUPERS, (AudioKind.PLAY,), None,
        ))
        assert entered.wait(0.5)
        close_started = time.monotonic()
        log.close()
        assert time.monotonic() - close_started < 0.5
        assert log.has_warning is True
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        # Interpreter startup and Windows temporary-directory metadata can take
        # several seconds. Bound close() itself above; a non-daemon stuck
        # writer must still fail this outer process-exit deadline.
        timeout=10.0,
        check=False,
    )
    assert result.returncode == 0, result.stderr


class _StuckWorker:
    def join(self, timeout: float | None = None) -> None:
        time.sleep(0.5 if timeout is None else min(timeout, 0.05))

    def is_alive(self) -> bool:
        return True


def test_full_queue_and_stuck_worker_close_returns_bounded_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostic_log,
        "_CLOSE_TIMEOUT_SECONDS",
        0.05,
        raising=False,
    )
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=1)
    worker = _StuckWorker()
    log._started = True
    log._worker = worker
    log.record(_record(identity="fills-queue"))

    closer = threading.Thread(target=log.close, daemon=True)
    started_at = time.monotonic()
    closer.start()
    closer.join(0.25)
    returned_bounded = not closer.is_alive()
    elapsed = time.monotonic() - started_at
    if closer.is_alive():
        log._queue.get_nowait()
        closer.join(0.75)

    assert returned_bounded is True
    assert elapsed < 0.25
    assert log.has_warning is True


class _FlushFailureStream:
    def __init__(self) -> None:
        self.close_calls = 0

    def flush(self) -> None:
        raise OSError(r"C:\private\flush.txt")

    def close(self) -> None:
        self.close_calls += 1


def test_flush_failure_still_attempts_close_and_only_latches_warning(
    tmp_path: Path,
) -> None:
    log = DiagnosticEventLog(tmp_path, session_id="abc123")
    stream = _FlushFailureStream()
    log._stream = stream

    log._close_stream()

    assert stream.close_calls == 1
    assert log.has_warning is True


def test_directory_failure_latches_warning_without_raising(tmp_path: Path) -> None:
    blocked_root = tmp_path / "not-a-directory"
    blocked_root.write_text("occupied")
    log = DiagnosticEventLog(blocked_root, session_id="abc123")

    log.start()
    log.close()

    assert log.has_warning is True


def test_open_failure_latches_warning_without_exception_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError(r"C:\Users\private\song.wav")

    monkeypatch.setattr(Path, "open", fail_open)
    log = DiagnosticEventLog(tmp_path, session_id="abc123")

    log.start()
    log.close()

    assert log.has_warning is True
    assert "private" not in repr(log).lower()


class _WriteFailureStream:
    def __init__(self) -> None:
        self.close_calls = 0

    def write(self, value: str) -> int:
        del value
        raise OSError(r"C:\captures\frame.png")

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1


def test_write_failure_latches_warning_and_closes_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _WriteFailureStream()
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: stream)
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=2)
    log.start()

    log.record(_record())
    log.close()

    assert log.has_warning is True
    assert stream.close_calls == 1


class _MemoryStream:
    def __init__(self) -> None:
        self.closed = False

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_rotation_failure_latches_warning_and_closes_current_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_stream = _MemoryStream()
    open_calls = 0

    def open_then_fail(*args: object, **kwargs: object) -> _MemoryStream:
        nonlocal open_calls
        del args, kwargs
        open_calls += 1
        if open_calls == 1:
            return first_stream
        raise OSError(r"C:\private\rotation.jsonl")

    monkeypatch.setattr(Path, "open", open_then_fail)
    log = DiagnosticEventLog(
        tmp_path,
        session_id="abc123",
        max_bytes=1,
        queue_capacity=3,
    )
    log.start()

    log.record(_record(identity="one"))
    log.record(_record(identity="two"))
    log.close()

    assert first_stream.closed is True
    assert log.has_warning is True


class _JoinFailureWorker:
    def join(self, timeout: float | None = None) -> None:
        del timeout
        raise RuntimeError(r"C:\private\thread.txt")

    def is_alive(self) -> bool:
        return True


def test_worker_close_failure_latches_warning_without_raising(tmp_path: Path) -> None:
    log = DiagnosticEventLog(tmp_path, session_id="abc123", queue_capacity=1)
    log._started = True
    log._worker = _JoinFailureWorker()

    log.close()

    assert log.has_warning is True


def test_writer_defaults_match_bounded_retention_contract() -> None:
    parameters = inspect.signature(DiagnosticEventLog).parameters

    assert parameters["max_bytes"].default == 2 * 1024 * 1024
    assert parameters["retained_files"].default == 5
    assert parameters["queue_capacity"].default == 1024
