"""Anonymous latency evidence through real engine, listener and JSONL writer."""
import json
import threading
import time
from dataclasses import replace

import pytest

from sf6_music_trigger import diagnostic_log as logs
from sf6_music_trigger.audio import AudioAction, AudioIssue, AudioKind
from sf6_music_trigger.controller import ListenerController
from sf6_music_trigger.domain import (
    CaptureSample,
    CaptureStatus,
    ClientGeometry,
    GaugeTrackerUpdate,
    Side,
)
from test_engine import (
    _build_harness, _frame, _running, _observation, _event,
    _RecordingAudioPort, _TrackerSpy,
)


def _rows(root):
    return [json.loads(line) for path in sorted(root.glob('*.jsonl'))
            for line in path.read_text().splitlines()]


def test_engine_serializes_raw_readings_and_correlated_audio(tmp_path):
    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record, p2_tracker=_TrackerSpy([
        GaugeTrackerUpdate(_event(Side.P2, levels_spent=3))]))
    h.engine._observer = lambda image, **kw: replace(
        _observation(kw['side'], kw['frame_id'], kw['sampled_at'], 'full'),
        normalized_meter=0.77, meter_level_supported=True,
        meter_level=1, meter_level_confidence=0.9,
    )
    log.start()
    try:
        h.clock.value = 1.0
        h.engine.process(_running(_frame(7, 1.0)))
    finally:
        log.close()
    rows = _rows(tmp_path / 'diagnostics' / 'latency')
    assert rows, 'actual event-log sink must enable separate latency serialization'
    meters = [r for r in rows if r['phase'] == 'METER']
    assert {r['side'] for r in meters} == {'P1', 'P2'}
    assert all(r['meter_level'] == 1 and r['normalized_meter'] == 0.77 for r in meters)
    assert all(r['frame_id'] == 7 and r['source_time'] == 1.0 for r in meters)
    assert all('baseline_before' in r and 'baseline_after' in r for r in meters)
    audio = [r for r in rows if r['phase'].startswith('AUDIO_')]
    assert [r['phase'] for r in audio] == ['AUDIO_BEGIN', 'AUDIO_END']
    assert audio[0]['audio_call_id'] == audio[1]['audio_call_id']
    assert all(r['frame_id'] == 7 for r in audio)
    transition = next(r for r in rows if r['phase'] == 'TRANSITION')
    assert transition['frame_id'] == 7
    assert transition['identity'] == audio[0]['identity'] == 'event-P2'
    assert audio[0]['host_time'] <= audio[1]['host_time']
    assert audio[1]['reason'] == 'ok'
    assert 'C:\\music' not in json.dumps(rows)
    legacy = _rows(tmp_path / 'diagnostics')
    assert legacy[-1]['transition'] == 'CONFIRMED'
    assert 'phase' not in legacy[-1]


@pytest.mark.parametrize('side', list(Side))
@pytest.mark.parametrize('case', ['silent_candidate', 'held_fade'])
def test_audio_lifecycle_identity_matches_confirmation_and_held_fade(tmp_path, side, case):
    from test_engine import _candidate
    from sf6_music_trigger.config import SideAudioConfig, SuperScope
    log = logs.DiagnosticEventLog(tmp_path)
    cfg = SideAudioConfig(main_track='test.wav', super_scope=SuperScope.ALL_SUPERS,
                          fade_non_three=True)
    h = _build_harness(diagnostic_sink=log.record)
    if case == 'held_fade':
        setattr(h.engine, '_p1_audio_config' if side is Side.P1 else '_p2_audio_config', cfg)
    log.start()
    try:
        h.clock.value = 1.
        if case == 'silent_candidate':
            h.engine._route_candidate(_candidate(side, levels_suspected=1))
            h.engine._route_confirmation(_event(side, candidate_id='candidate-'+side.value, levels_spent=3))
        else:
            h.engine._route_confirmation(_event(side, levels_spent=1))
            h.clock.value = 5.
            h.engine._flush_held_fades()
    finally:
        log.close()
    rows = _rows(tmp_path / 'diagnostics' / 'latency')
    transition = next(r for r in rows if r['phase'] == 'TRANSITION'
                      and r['transition'] == ('CONFIRMED' if case == 'silent_candidate' else 'FADE'))
    audio = [r for r in rows if r['phase'].startswith('AUDIO_')]
    assert len(audio) >= 2
    assert all(r['identity'] == transition['identity'] for r in audio[-2:])


def test_latency_rejects_private_or_unbounded_payloads():
    record_type = getattr(logs, 'LatencyRecord', None)
    assert record_type is not None, 'closed latency record is missing'
    for changes in ({'reason': 'C:/private/file.wav'}, {'host_time': float('nan')},
                    {'identity': 'window title'}, {'audio_actions': ('secret',)},
                    {'meter_level': 99}, {'normalized_meter': float('inf')}):
        with pytest.raises((ValueError, TypeError)):
            record_type(phase='METER', **({'host_time': 1.0} | changes))


def test_viewport_rejection_emits_process_end_reason(tmp_path):
    from sf6_music_trigger.viewport import CenteredViewportResolver

    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record)
    h.engine._viewport_resolver = CenteredViewportResolver(16 / 9)
    log.start()
    try:
        h.clock.value = 1.0
        h.engine.process(_running(_frame(
            7, 1.0, geometry=ClientGeometry(0, 0, 1280, 800)
        )))
    finally:
        log.close()

    rows = _rows(tmp_path / 'diagnostics' / 'latency')
    assert any(row['phase'] == 'PROCESS_END' and row['reason'] == 'viewport'
               for row in rows)


def test_stalled_writer_and_full_queue_do_not_block_producer(tmp_path):
    log = logs.DiagnosticEventLog(tmp_path, queue_capacity=1)
    h = _build_harness(diagnostic_sink=log.record)
    child = getattr(log, '_latency', None)
    assert child is not None, 'engine must wire bounded latency writer'
    entered, release = threading.Event(), threading.Event()
    original = child._write_record
    def stalled(record):
        entered.set()
        assert release.wait(2)
        original(record)
    child._write_record = stalled
    log.start()
    try:
        record = logs.LatencyRecord(phase='PROCESS_BEGIN', host_time=1.0)
        child.record(record)
        assert entered.wait(1)
        started = time.perf_counter()
        for _ in range(500):
            child.record(record)
        assert time.perf_counter() - started < 0.25
        assert child.dropped_count >= 499
    finally:
        release.set()
        log.close()


def test_failed_latency_sink_does_not_change_audio_policy(tmp_path):
    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record)
    child = getattr(log, '_latency', None)
    assert child is not None
    def fail(record):
        raise OSError('private disk path')
    child.record = fail
    h.engine._route_confirmation(_event(Side.P2, levels_spent=3))
    assert len(h.audio.batches) == 1
    assert h.engine.event_log == (_event(Side.P2, levels_spent=3),)
    assert h.engine.diagnostic_warning
    log.close()


def test_audio_failure_emits_paired_end_without_private_error(tmp_path):
    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record,
                       audio=_RecordingAudioPort({Side.P2: AudioIssue.BACKEND_FAILURE}))
    log.start()
    try:
        h.engine._route_confirmation(_event(Side.P2, levels_spent=3))

    finally:
        log.close()
    rows = _rows(tmp_path / 'diagnostics' / 'latency')
    audio = [r for r in rows if r['phase'].startswith('AUDIO_')]
    assert [r['phase'] for r in audio] == ['AUDIO_BEGIN', 'AUDIO_END']
    assert audio[-1]['reason'] == 'failed'


def test_meter_sampling_is_bounded_but_keeps_changes_and_heartbeat(tmp_path, monkeypatch):
    import sf6_music_trigger.engine as module
    log = logs.DiagnosticEventLog(tmp_path, queue_capacity=1024)
    h = _build_harness(diagnostic_sink=log.record)
    host = [10.0]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: host[0])
    h.engine._observer = lambda image, **kw: replace(
        _observation(kw['side'], kw['frame_id'], kw['sampled_at'], 'full'),
        normalized_meter=0.7, meter_level_supported=True,
        meter_level=1 if kw['frame_id'] < 20 else 0, meter_level_confidence=0.9)
    log.start()
    try:
        for i in range(100):
            host[0] = 10 + i / 100
            h.clock.value = 1 + i / 100
            h.engine.process(_running(_frame(i, h.clock.value)))
    finally:
        log.close()
    meters = [r for r in _rows(tmp_path / 'diagnostics' / 'latency') if r['phase'] == 'METER']
    assert meters, 'sampled meter records missing'
    for side in ('P1', 'P2'):
        selected = [r for r in meters if r['side'] == side]
        assert 3 <= len(selected) <= 10
        assert {r['meter_level'] for r in selected} == {0, 1}
        assert any(r['frame_id'] >= 70 for r in selected)


def test_actual_panel_factory_listener_writes_with_default_ui_toggle_off(tmp_path, monkeypatch):
    from sf6_music_trigger.config import AppConfig, DetectionBackend
    from sf6_music_trigger.panel import create_panel_application, _build_listener_controller
    from test_panel_view_model import _write_valid_profile, PROFILE_PATH
    import sf6_music_trigger.capture as capture

    _write_valid_profile(tmp_path)
    local_app_data = tmp_path / "local-app-data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    (tmp_path / 'track.wav').write_bytes(b'fixture-only-never-loaded')
    processed, release = threading.Event(), threading.Event()
    class Audio:
        def execute(self, actions):
            raise AssertionError('blank frame must never play')
        def close(self):
            pass
    class Source:
        def __init__(self, **kwargs):
            self.count = 0
        def start(self):
            pass
        def read(self):
            self.count += 1
            if self.count == 1:
                return _running(
                    _frame(
                        0,
                        time.perf_counter(),
                        geometry=ClientGeometry(0, 0, 1600, 800),
                    )
                )
            processed.set()
            assert release.wait(2)
            return CaptureSample(CaptureStatus.STOPPED)
        def stop(self):
            release.set()
    monkeypatch.setattr(capture, 'DxcamFrameSource', Source)
    app = create_panel_application(config_path=tmp_path / 'config.json',
                                   audio_port_factory=lambda root: Audio(),
                                   config_loader=lambda path: AppConfig(detection_backend=DetectionBackend.SCREEN))
    app.view_model.profile_path = PROFILE_PATH
    app.view_model.p1_main_track = 'track.wav'
    app.view_model.p2_enabled = False
    # The factory is exercised without constructing/enabling the Tk diagnostic view.
    controller = _build_listener_controller(app, audio_diagnostic=lambda error: None)
    assert isinstance(controller, ListenerController)
    controller.start()
    try:
        assert processed.wait(2)
    finally:
        controller.request_stop()
        controller.join(2)
    assert not controller.is_running
    assert not any(s.kind in {'worker_error', 'diagnostic_error'} for s in controller.drain_statuses())
    rows = _rows(
        local_app_data
        / "SF6 Music Trigger"
        / "V2"
        / "diagnostics"
        / "latency"
    )
    by_phase = {r['phase']: r for r in rows}
    assert {'CAPTURE_BEGIN', 'CAPTURE_END', 'PROCESS_BEGIN', 'PROCESS_END', 'METER'} <= by_phase.keys()
    begin, end = by_phase['PROCESS_BEGIN'], by_phase['PROCESS_END']
    assert begin['cycle_id'] == end['cycle_id'] == by_phase['CAPTURE_BEGIN']['cycle_id']
    assert end['capture_begin'] <= end['source_time'] <= end['capture_end']
    assert end['capture_end'] <= end['process_begin'] <= end['host_time']
    assert 'fixture-only' not in json.dumps(rows) and str(tmp_path) not in json.dumps(rows)


def test_processing_exception_has_failed_end_and_preserves_exception(tmp_path):
    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record)
    def fail(frame, **kwargs):
        raise RuntimeError('private frame owner')
    h.outcome_observer.update = fail
    h.clock.value = 1.0
    log.start()
    try:
        with pytest.raises(RuntimeError, match='private frame owner'):
            h.engine.process(_running(_frame(0, 1.0)))
    finally:
        log.close()
    rows = _rows(tmp_path / 'diagnostics' / 'latency')
    assert rows[-1]['phase'] == 'PROCESS_END'
    assert rows[-1]['reason'] == 'failed'
    assert 'private frame owner' not in json.dumps(rows)


def test_contended_latency_lock_is_a_drop_not_a_wait(tmp_path):
    log = logs.LatencyEventLog(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def hold():
        with log._lock:
            entered.set()
            assert release.wait(2)
    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert entered.wait(1)
        started = time.perf_counter()
        log.record(logs.LatencyRecord(phase='METER', host_time=1))
        assert time.perf_counter() - started < 0.25
    finally:
        release.set()
        worker.join(2)
    assert log.dropped_count == 1
    log.close()


def test_latency_rotation_and_restart_preserve_legacy_file(tmp_path):
    log = logs.DiagnosticEventLog(tmp_path, max_bytes=1500, retained_files=2)
    child = log.latency_log()
    log.start()
    for i in range(8):
        child.record(logs.LatencyRecord(phase='METER', host_time=i))
    log.close()
    assert len(list((tmp_path / 'diagnostics' / 'latency').glob('*.jsonl'))) == 2
    retained = _rows(tmp_path / 'diagnostics' / 'latency')
    assert retained[-1]['host_time'] == 7
    assert all(r['host_time'] > 0 for r in retained)
    assert all(p.stat().st_size <= 1500 for p in (tmp_path / 'diagnostics' / 'latency').glob('*.jsonl'))
    assert len(list((tmp_path / 'diagnostics').glob('*.jsonl'))) == 1
    log.start()
    child.record(logs.LatencyRecord(phase='METER', host_time=9))
    log.close()
    assert _rows(tmp_path / 'diagnostics' / 'latency')[-1]['host_time'] == 9


def test_failed_latency_storage_sets_warning_but_engine_keeps_processing(tmp_path):
    (tmp_path / 'diagnostics').mkdir()
    (tmp_path / 'diagnostics' / 'latency').write_text('blocks directory')
    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record)
    log.start()
    try:
        h.clock.value = 1
        result = h.engine.process(_running(_frame(0, 1)))
        assert result.events == ()
        assert log.has_warning
    finally:
        log.close()


def test_audio_begin_reaches_disk_while_execute_is_still_blocked(tmp_path):
    entered, release, written = threading.Event(), threading.Event(), threading.Event()
    class Audio(_RecordingAudioPort):
        def execute(self, actions):
            entered.set()
            assert release.wait(2)
            super().execute(actions)
    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record, audio=Audio())
    original = log._latency._write_record
    def observe_write(record):
        original(record)
        if record.phase == 'AUDIO_BEGIN':
            written.set()
    log._latency._write_record = observe_write
    log.start()
    worker = threading.Thread(target=lambda: h.engine._route_confirmation(_event(Side.P2, levels_spent=3)))
    worker.start()
    try:
        assert entered.wait(1) and written.wait(1)
        phases = [r['phase'] for r in _rows(tmp_path / 'diagnostics' / 'latency')]
        assert phases == ['AUDIO_BEGIN']
    finally:
        release.set()
        worker.join(2)
        log.close()
    phases = [r['phase'] for r in _rows(tmp_path / 'diagnostics' / 'latency')]
    assert phases == ['AUDIO_BEGIN', 'AUDIO_END', 'TRANSITION']


def test_continuously_changing_meters_are_hard_rate_limited(tmp_path, monkeypatch):
    import sf6_music_trigger.engine as module
    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record)
    host = [10.0]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: host[0])
    log.start()
    try:
        for i in range(1000):
            host[0] = 10 + i / 1000
            observation = replace(_observation(Side.P2, i, 1.0, 'full'),
                                  normalized_meter=(i % 10) / 10,
                                  meter_level_supported=True,
                                  meter_level=i % 4, meter_level_confidence=0.9)
            h.engine._latency_meter(observation, h.p2_tracker, None)
    finally:
        log.close()
    rows = _rows(tmp_path / 'diagnostics' / 'latency')
    assert 1 <= sum(r['phase'] == 'METER' for r in rows) <= 10


def test_same_frame_two_plays_and_immediate_stop_keep_every_audio_pair(tmp_path, monkeypatch):
    import sf6_music_trigger.engine as module
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 10.0)
    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record,
                       p1_tracker=_TrackerSpy([GaugeTrackerUpdate(_event(Side.P1, levels_spent=3))]),
                       p2_tracker=_TrackerSpy([GaugeTrackerUpdate(_event(Side.P2, levels_spent=3))]))
    h.engine._observer = lambda image, **kw: _observation(
        kw['side'], kw['frame_id'], kw['sampled_at'], 'full')
    h.clock.value = 1.0
    log.start()
    try:
        h.engine.process(_running(_frame(7, 1)))
        h.engine._execute_audio((AudioAction(AudioKind.STOP, Side.P2),))
        h.engine._route_confirmation(_event(Side.P2, event_id='second-P2', levels_spent=3))
    finally:
        log.close()
    rows = _rows(tmp_path / 'diagnostics' / 'latency')
    begin = [r for r in rows if r['phase'] == 'AUDIO_BEGIN']
    end = [r for r in rows if r['phase'] == 'AUDIO_END']
    assert len(begin) == len(end) == len(h.audio.batches) == 4
    assert [r['audio_call_id'] for r in begin] == [r['audio_call_id'] for r in end]
    assert [r['side'] for r in begin[:2]] == ['P1', 'P2']
    assert all(r['frame_id'] == 7 for r in begin[:2] + end[:2])
    assert [r['identity'] for r in rows if r['phase'] == 'TRANSITION'] == ['event-P1', 'event-P2', 'second-P2']


def test_two_stalled_writers_share_original_one_second_close_budget(tmp_path):
    from test_diagnostic_log import _record
    log = logs.DiagnosticEventLog(tmp_path)
    child = log.latency_log()
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    def stall(index):
        def write(record):
            entered[index].set()
            assert release.wait(4)
        return write
    log._write_record = stall(0)
    child._write_record = stall(1)
    log.start()
    log.record(_record())
    child.record(logs.LatencyRecord(phase='METER', host_time=1))
    workers = [log._worker, child._worker]
    try:
        assert all(event.wait(1) for event in entered)
        started = time.perf_counter()
        log.close()
        elapsed = time.perf_counter() - started
        assert elapsed < 1.5, f'combined shutdown took {elapsed:.3f}s'
        assert log.has_warning
    finally:
        release.set()
        for worker in workers:
            worker.join(2)


def test_public_baseline_is_recorded_and_stale_frame_does_not_reuse_meter(tmp_path, monkeypatch):
    import sf6_music_trigger.engine as module
    log = logs.DiagnosticEventLog(tmp_path)
    h = _build_harness(diagnostic_sink=log.record)
    h.engine._observer = lambda image, **kw: replace(
        _observation(kw['side'], kw['frame_id'], kw['sampled_at'], 'full'),
        normalized_meter=0.7, meter_level_supported=True,
        meter_level=1, meter_level_confidence=0.9)
    host = [10.0]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: host[0])
    log.start()
    try:
        for i in range(20):
            host[0] = 10 + i * 0.1
            h.clock.value = 1 + i * 0.1
            h.engine.process(_running(_frame(i, h.clock.value)))
        assert h.p1_tracker.stable_meter_level == 1
        assert h.p2_tracker.stable_meter_level == 1
        host[0] = 13.0
        h.clock.value = 10
        h.engine.process(_running(_frame(100, 1)))
    finally:
        log.close()
    rows = _rows(tmp_path / 'diagnostics' / 'latency')
    meters = [r for r in rows if r['phase'] == 'METER']
    for side in ('P1', 'P2'):
        assert any(r['side'] == side and r['baseline_before'] == r['baseline_after'] == 1 for r in meters)
    assert not any(r['frame_id'] == 100 for r in meters)
    assert rows[-1]['phase'] == 'PROCESS_END' and rows[-1]['reason'] == 'stale'
