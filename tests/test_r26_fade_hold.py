"""Four seconds of successful playback precede an ordinary result fade."""
import pytest
from sf6_music_trigger.audio import AudioKind
from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.domain import Side
from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel
from test_engine import _build_harness, _candidate, _event, _drive


def kinds(h):
    return [a.kind for b in h.audio.batches for a in b]


@pytest.mark.parametrize('side', list(Side))
@pytest.mark.parametrize('preview', [False, True])
def test_non_three_waits_four_seconds_from_play_not_confirmation(side, preview):
    cfg = SideAudioConfig(main_track='music.wav', super_scope=SuperScope.ALL_SUPERS, fade_non_three=True)
    h = _build_harness(p1_config=cfg, p2_config=cfg)
    h.clock.value = 10.
    candidate = _candidate(side, suspected_at=9., levels_suspected=1)
    if preview:
        h.engine._route_candidate(candidate)
    h.clock.value = 11.
    h.engine._route_confirmation(_event(side, confirmed_at=10.5, levels_spent=1,
        candidate_id=candidate.candidate_id if preview else None))
    assert kinds(h) == [AudioKind.STOP, AudioKind.PLAY]
    deadline = 14. if preview else 15.
    _drive(h, [(deadline-.01, 'level_1', 'level_1'), (deadline, 'level_1', 'level_1')])
    assert kinds(h) == [AudioKind.STOP, AudioKind.PLAY, AudioKind.FADE]
    assert h.audio.batches[-1][0].fade_ms == 1500
    assert h.diagnostic_records[-1].transition == 'FADE'
    assert h.diagnostic_records[-1].monotonic_time == deadline
    assert not h.engine.diagnostic_warning


@pytest.mark.parametrize('label', [
    OutcomeLabel.UNKNOWN, OutcomeLabel.BLOCK, OutcomeLabel.WHIFF,
    OutcomeLabel.NON_HIT,
])
def test_early_result_waits_remaining_playback_time(label):
    h = _build_harness()
    h.clock.value = 1.
    h.engine._route_confirmation(_event(event_id='held', levels_spent=3))
    h.clock.value = 4.
    h.engine._route_outcomes((OutcomeDecision('held', Side.P1, label, .9, 4., ()),))
    assert kinds(h) == [AudioKind.STOP, AudioKind.PLAY]
    _drive(h, [(4.99, 'level_1', 'level_1'), (5., 'level_1', 'level_1')])
    assert kinds(h)[-1] is AudioKind.FADE


def test_new_music_and_reset_invalidate_queued_fade():
    cfg = SideAudioConfig(main_track='music.wav', super_scope=SuperScope.ALL_SUPERS, fade_non_three=True)
    h = _build_harness(p1_config=cfg)
    h.clock.value = 1.
    h.engine._route_confirmation(_event(event_id='old', levels_spent=1))
    h.clock.value = 4.
    h.engine._route_confirmation(_event(event_id='new', levels_spent=3))
    _drive(h, [(5., 'level_1', 'level_1')])
    assert AudioKind.FADE not in kinds(h)
    h.clock.value = 6.
    h.engine._route_confirmation(_event(event_id='reset-pending', levels_spent=1))
    h.engine.reset('test stop')
    before = len(h.audio.batches)
    _drive(h, [(12., 'level_1', 'level_1')])
    assert len(h.audio.batches) == before


def test_latest_side_retains_its_own_hold_deadline():
    cfg = SideAudioConfig(main_track='music.wav', super_scope=SuperScope.ALL_SUPERS, fade_non_three=True)
    h = _build_harness(p1_config=cfg, p2_config=cfg)
    for side, at in [(Side.P1, 1.), (Side.P2, 2.)]:
        h.clock.value = at
        h.engine._route_confirmation(_event(side, levels_spent=2))
    _drive(h, [(5., 'level_1', 'level_1'), (6., 'level_1', 'level_1')])
    fades = [a for b in h.audio.batches for a in b if a.kind is AudioKind.FADE]
    assert [a.side for a in fades] == [Side.P2]
    assert [(r.side, r.monotonic_time) for r in h.diagnostic_records if r.transition == 'FADE'] == [
        (Side.P2, 6.)]


def test_failed_play_does_not_schedule_or_retry_a_fade():
    from sf6_music_trigger.audio import AudioIssue
    from test_engine import _RecordingAudioPort
    cfg = SideAudioConfig(main_track='music.wav', super_scope=SuperScope.ALL_SUPERS, fade_non_three=True)
    audio = _RecordingAudioPort(failures={Side.P1: AudioIssue.BACKEND_FAILURE})
    h = _build_harness(p1_config=cfg, audio=audio)
    h.engine._route_confirmation(_event(levels_spent=1))
    _drive(h, [(5., 'level_1', 'level_1')])
    assert len(h.diagnostics) == 1
    assert kinds(h) == [AudioKind.STOP, AudioKind.PLAY]


def test_loading_time_is_not_counted_as_audible_playback():
    from test_engine import _Clock, _RecordingAudioPort
    clock = _Clock(1.)
    class SlowPort(_RecordingAudioPort):
        def execute(self, actions):
            super().execute(actions)
            if any(a.kind is AudioKind.PLAY for a in actions):
                clock.value += 2.
    cfg = SideAudioConfig(main_track='music.wav', super_scope=SuperScope.ALL_SUPERS, fade_non_three=True)
    h = _build_harness(p1_config=cfg, audio=SlowPort(), clock=clock)
    h.engine._route_confirmation(_event(levels_spent=1))
    _drive(h, [(5., 'level_1', 'level_1'), (6.99, 'level_1', 'level_1'), (7., 'level_1', 'level_1')])
    assert [(r.transition, r.monotonic_time) for r in h.diagnostic_records if r.transition == 'FADE'] == [('FADE', 7.)]


def test_late_confirmation_does_not_add_another_four_seconds():
    cfg = SideAudioConfig(main_track='music.wav', super_scope=SuperScope.ALL_SUPERS, fade_non_three=True)
    h = _build_harness(p1_config=cfg)
    h.clock.value = 1.
    c = _candidate(levels_suspected=1)
    h.engine._route_candidate(c)
    h.clock.value = 6.
    h.engine._route_confirmation(_event(levels_spent=1, candidate_id=c.candidate_id, confirmed_at=6.))
    assert kinds(h) == [AudioKind.STOP, AudioKind.PLAY, AudioKind.FADE]


def test_cancelled_invalid_candidate_is_not_forced_to_play_four_seconds():
    from test_engine import _cancellation
    h = _build_harness()
    h.clock.value = 1.
    h.engine._route_candidate(_candidate())
    h.clock.value = 1.5
    h.engine._route_cancellation(_cancellation())
    assert kinds(h) == [AudioKind.STOP, AudioKind.PLAY, AudioKind.FADE]


def test_new_play_on_exact_due_frame_supersedes_old_fade():
    from sf6_music_trigger.domain import GaugeTrackerUpdate
    from test_engine import _TrackerSpy
    cfg = SideAudioConfig(main_track='music.wav', super_scope=SuperScope.ALL_SUPERS, fade_non_three=True)
    h = _build_harness(p1_config=cfg, p1_tracker=_TrackerSpy([
        GaugeTrackerUpdate(_event(event_id='next', confirmed_at=5., levels_spent=3))]))
    h.clock.value = 1.
    h.engine._route_confirmation(_event(event_id='old', levels_spent=1))
    _drive(h, [(5., 'level_1', 'level_1')])
    assert kinds(h) == [AudioKind.STOP, AudioKind.PLAY, AudioKind.STOP, AudioKind.PLAY]


@pytest.mark.parametrize('runtime', [False, True])
def test_reset_immediately_cleans_up_held_playback(runtime):
    cfg = SideAudioConfig(main_track='music.wav', super_scope=SuperScope.ALL_SUPERS, fade_non_three=True)
    h = _build_harness(p1_config=cfg)
    h.clock.value = 1.
    h.engine._route_confirmation(_event(levels_spent=1))
    h.clock.value = 2.
    if runtime:
        h.engine._runtime_reset('lost capture')
    else:
        h.engine.reset('stopped')
    assert kinds(h) == [AudioKind.STOP, AudioKind.PLAY, AudioKind.FADE]
    _drive(h, [(6., 'level_1', 'level_1')])
    assert kinds(h).count(AudioKind.FADE) == 1


def test_delayed_fade_failure_is_not_reported_as_success():
    from sf6_music_trigger.audio import AudioIssue
    cfg = SideAudioConfig(main_track='music.wav', super_scope=SuperScope.ALL_SUPERS, fade_non_three=True)
    h = _build_harness(p1_config=cfg)
    h.engine._route_confirmation(_event(levels_spent=1))
    h.audio.failures[Side.P1] = AudioIssue.BACKEND_FAILURE
    _drive(h, [(4., 'level_1', 'level_1'), (5., 'level_1', 'level_1')])
    assert len(h.diagnostics) == 1
    assert h.engine.outcome_statuses[0] == '淡出失败'
    assert h.diagnostic_records[-1].audio_actions == ()
    assert h.diagnostic_records[-1].evidence == ('fade_failed',)
