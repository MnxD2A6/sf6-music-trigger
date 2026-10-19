"""R27: real engine/scheduler/port, fake mixer only; never opens real audio."""
import pytest

from sf6_music_trigger.audio import AudioAction, AudioKind, AudioPlaybackError
from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.domain import Side, GaugeTrackerUpdate
from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel
from test_audio import fake_audio
from test_engine import (
    _build_harness, _candidate, _event, _cancellation, _drive,
    _OutcomeObserverSpy, _TrackerSpy,
)


def play(side, track='main.wav'):
    return (AudioAction(AudioKind.STOP, side), AudioAction(AudioKind.PLAY, side, track, .2))


def harness(probe, **kwargs):
    cfg = SideAudioConfig(main_track='main.wav', whiff_track='whiff.wav',
                          play_whiff_sound=True, fade_non_three=True,
                          super_scope=SuperScope.ALL_SUPERS)
    return _build_harness(audio=probe.port, p1_config=cfg, p2_config=cfg, **kwargs)


def playing(probe):
    return [i for i, c in probe.mixer.channels.items() if c.playing]


def outcome(h, side, identity, label=OutcomeLabel.WHIFF):
    h.engine._route_outcomes((OutcomeDecision(identity, side, label, .9, h.clock.value, ()),))


@pytest.mark.parametrize('first,second', [(Side.P1, Side.P2), (Side.P2, Side.P1)])
def test_r27_port_hard_stops_opposite_before_play(fake_audio, first, second):
    fake_audio.port.execute(play(first))
    fake_audio.reset_calls()
    fake_audio.port.execute(play(second))
    assert playing(fake_audio) == [0 if second is Side.P1 else 1]
    ops = [x[:2] for x in fake_audio.timeline if x[0].startswith('channel.')]
    assert ops.index(('channel.stop', 0 if first is Side.P1 else 1)) < ops.index(('channel.play', 0 if second is Side.P1 else 1))


@pytest.mark.parametrize('second', list(Side))
@pytest.mark.parametrize('stage', ['sound', 'set_volume'])
def test_r27_preparation_failure_preserves_old_play_and_deadline(fake_audio, second, stage):
    h = harness(fake_audio)
    h.clock.value = 1.
    h.engine._route_confirmation(_event(Side.P1, event_id='old', levels_spent=1))
    fake_audio.fail_next(stage)
    h.clock.value = 2.
    h.engine._route_confirmation(_event(second, event_id='failed', levels_spent=3))
    assert playing(fake_audio) == [0]
    _drive(h, [(5., 'level_1', 'level_1')])
    assert ('fadeout', 1500) in fake_audio.mixer.channels[0].calls
    assert len(h.diagnostics) == 1


def test_r27_opposite_stop_failure_never_starts_new_track(fake_audio):
    fake_audio.port.execute(play(Side.P1))
    fake_audio.mixer.channels[0].stop_failures_remaining = 1
    with pytest.raises(AudioPlaybackError):
        fake_audio.port.execute(play(Side.P2))
    assert playing(fake_audio) == [0]
    assert not any(c[0] == 'play' for c in fake_audio.mixer.channels[1].calls)


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_r27_partial_play_failure_cleans_up_and_blocks_overlap(fake_audio, cleanup_fails):
    fake_audio.port.execute(play(Side.P1))
    target = fake_audio.mixer.channels[1]
    original = target.play
    def partial(sound):
        target.playing = True
        if cleanup_fails:
            target.stop_failures_remaining = 2
        raise RuntimeError('partial device start')
    target.play = partial
    with pytest.raises(AudioPlaybackError):
        fake_audio.port.execute(play(Side.P2))
    assert playing(fake_audio) == ([1] if cleanup_fails else [])
    target.play = original
    if cleanup_fails:
        with pytest.raises(AudioPlaybackError):
            fake_audio.port.execute(play(Side.P1))
        assert playing(fake_audio) == [1]
    fake_audio.port.execute(play(Side.P1))
    assert playing(fake_audio) == [0]


@pytest.mark.parametrize('first,second', [(Side.P1, Side.P2), (Side.P2, Side.P1)])
def test_r27_old_whiff_cannot_revive_superseded_music(fake_audio, first, second):
    h = harness(fake_audio)
    h.engine._route_confirmation(_event(first, event_id='old', levels_spent=3))
    h.engine._route_confirmation(_event(second, event_id='new', levels_spent=3))
    fake_audio.reset_calls()
    outcome(h, first, 'old')
    assert fake_audio.timeline == []
    assert len(h.engine.event_log) == 2


def test_same_frame_new_confirmation_suppresses_old_non_hit_sound(fake_audio):
    old = _event(Side.P1, event_id="old", levels_spent=3)
    new = _event(Side.P2, event_id="new", levels_spent=3)
    outcomes = _OutcomeObserverSpy()
    outcomes.results[0] = (
        OutcomeDecision("old", Side.P1, OutcomeLabel.NON_HIT, .9, 2., ()),
    )
    h = harness(
        fake_audio,
        outcome_observer=outcomes,
        p2_tracker=_TrackerSpy([GaugeTrackerUpdate(new)]),
    )
    h.engine._route_confirmation(old)
    loaded_before = len(fake_audio.mixer.sound_paths)
    fake_audio.reset_calls()

    _drive(h, [(2., "level_1", "level_1")])

    assert playing(fake_audio) == [1]
    loaded = fake_audio.mixer.sound_paths[loaded_before:]
    assert len(loaded) == 1
    assert loaded[0].endswith("main.wav")
    assert h.engine.outcome_statuses == ("格挡/未命中", "等待大招结果")


@pytest.mark.parametrize('terminal', ['confirm', 'cancel'])
@pytest.mark.parametrize('silent', [False, True])
def test_r27_superseded_candidate_followup_is_silent(fake_audio, terminal, silent):
    h = harness(fake_audio)
    if silent:
        fake_audio.fail_next('sound')
    h.engine._route_candidate(_candidate(Side.P1))
    h.engine._route_confirmation(_event(Side.P2, event_id='new', levels_spent=3))
    fake_audio.reset_calls()
    if terminal == 'confirm':
        event = _event(Side.P1, candidate_id='candidate-P1', levels_spent=3)
        assert h.engine._route_confirmation(event)
        assert not h.engine._route_confirmation(event)
        outcome(h, Side.P1, event.event_id)
    else:
        assert h.engine._route_cancellation(_cancellation(Side.P1))
    assert fake_audio.timeline == []


def test_r27_p1_p2_p1_identity_protects_new_p1(fake_audio):
    h = harness(fake_audio)
    for side, identity in [(Side.P1, 'a'), (Side.P2, 'b'), (Side.P1, 'c')]:
        h.engine._route_confirmation(_event(side, event_id=identity, levels_spent=3))
    fake_audio.reset_calls()
    outcome(h, Side.P1, 'a', OutcomeLabel.UNKNOWN)
    outcome(h, Side.P2, 'b')
    assert fake_audio.timeline == []
    assert playing(fake_audio) == [0]
    outcome(h, Side.P1, 'c')
    assert fake_audio.mixer.sound_paths[-1].endswith('whiff.wav')


def test_r27_same_frame_old_fade_and_two_new_plays_leave_p2(fake_audio):
    h = harness(fake_audio,
        p1_tracker=_TrackerSpy([GaugeTrackerUpdate(_event(Side.P1, event_id='next1', levels_spent=3))]),
        p2_tracker=_TrackerSpy([GaugeTrackerUpdate(_event(Side.P2, event_id='next2', levels_spent=1))]))
    h.clock.value = 1.
    h.engine._route_confirmation(_event(Side.P1, event_id='old', levels_spent=1))
    _drive(h, [(5., 'level_1', 'level_1')])
    assert playing(fake_audio) == [1]
    assert not any(x[0] == 'channel.fadeout' for x in fake_audio.timeline)
    _drive(h, [(8.99, 'level_1', 'level_1')], first_frame_id=1)
    assert playing(fake_audio) == [1]
    _drive(h, [(9., 'level_1', 'level_1')], first_frame_id=2)
    assert playing(fake_audio) == []
    assert fake_audio.mixer.channels[1].calls[-1] == ('fadeout', 1500)


@pytest.mark.parametrize('runtime', [False, True])
def test_r27_reset_drops_superseded_fade_and_rejects_old_outcome(fake_audio, runtime):
    h = harness(fake_audio)
    for side in Side:
        h.engine._route_confirmation(_event(side, levels_spent=1))
    (h.engine._runtime_reset if runtime else h.engine.reset)('reset')
    fake_audio.reset_calls()
    outcome(h, Side.P1, 'event-P1')
    outcome(h, Side.P2, 'event-P2')
    _drive(h, [(10., 'level_1', 'level_1')])
    assert fake_audio.timeline == []


@pytest.mark.parametrize('first,second', [(Side.P1, Side.P2), (Side.P2, Side.P1)])
def test_r27_opposite_play_on_exact_old_deadline_discards_old_fade(fake_audio, first, second):
    tracker = _TrackerSpy([GaugeTrackerUpdate(_event(second, event_id='new', levels_spent=3))])
    h = harness(fake_audio, **{'p1_tracker' if second is Side.P1 else 'p2_tracker': tracker})
    h.clock.value = 1.
    h.engine._route_confirmation(_event(first, event_id='old', levels_spent=1))
    _drive(h, [(5., 'level_1', 'level_1'), (10., 'level_1', 'level_1')])
    assert playing(fake_audio) == [0 if second is Side.P1 else 1]
    assert not any(x[0] == 'channel.fadeout' for x in fake_audio.timeline)


@pytest.mark.parametrize('stage', ['stop', 'play'])
def test_r27_engine_failure_clears_invalidated_old_hold_and_outcomes(fake_audio, stage):
    h = harness(fake_audio)
    h.clock.value = 1.
    h.engine._route_confirmation(_event(Side.P1, event_id='old', levels_spent=1))
    if stage == 'stop':
        fake_audio.mixer.channels[0].stop_failures_remaining = 1
    else:
        fake_audio.mixer.channels[1].play_failures_remaining = 1
    h.engine._route_confirmation(_event(Side.P2, event_id='failed', levels_spent=1))
    assert len(h.diagnostics) == 1
    fake_audio.reset_calls()
    outcome(h, Side.P1, 'old')
    outcome(h, Side.P2, 'failed')
    _drive(h, [(5., 'level_1', 'level_1')])
    assert fake_audio.timeline == []


def test_r27_failed_candidate_can_confirm_once_with_fresh_hold_clock(fake_audio):
    h = harness(fake_audio)
    fake_audio.fail_next('sound')
    h.clock.value = 1.
    h.engine._route_candidate(_candidate(Side.P1, levels_suspected=1))
    h.clock.value = 3.
    event = _event(Side.P1, candidate_id='candidate-P1', levels_spent=1)
    assert h.engine._route_confirmation(event)
    assert not h.engine._route_confirmation(event)
    _drive(h, [(6.99, 'level_1', 'level_1')])
    assert playing(fake_audio) == [0]
    _drive(h, [(7., 'level_1', 'level_1')], first_frame_id=1)
    assert playing(fake_audio) == []
    assert fake_audio.mixer.channels[0].calls[-1] == ('fadeout', 1500)


@pytest.mark.parametrize('side', list(Side))
@pytest.mark.parametrize('stage', ['sound', 'set_volume', None])
def test_r27_real_observer_retains_audible_owner_on_preparation_failure(fake_audio, side, stage):
    from sf6_music_trigger.outcome import OutcomeObserver
    h = harness(fake_audio, outcome_observer=OutcomeObserver())
    h.clock.value = 1.
    a = _event(side, event_id='unresolved-A', confirmed_at=1., levels_spent=3)
    h.engine._route_confirmation(a)
    if stage is not None:
        fake_audio.fail_next(stage)
    h.clock.value = 2.
    b = _event(side, event_id='replacement-B', confirmed_at=2., levels_spent=3)
    h.engine._route_confirmation(b)
    assert h.engine.event_log == (a, b)
    assert not h.engine._route_confirmation(b)
    assert len(h.diagnostics) == (0 if stage is None else 1)
    assert playing(fake_audio) == [0 if side is Side.P1 else 1]
    # No pre-spend baseline: retain A's t=9 deadline on failure, or use
    # successful B's t=10 deadline. ALL_SUPERS now intentionally keeps UNKNOWN
    # playback, including after a replacement preparation failure.
    _drive(h, [(8.99, 'level_1', 'level_1'), (9., 'level_1', 'level_1'),
               (10., 'level_1', 'level_1'), (12., 'level_1', 'level_1')])
    results = [(r.identity, r.monotonic_time, r.outcome, r.audio_actions)
               for r in h.diagnostic_records if r.transition == 'OUTCOME']
    assert results == [('replacement-B' if stage is None else 'unresolved-A',
                        10. if stage is None else 9., OutcomeLabel.UNKNOWN,
                        ())]
    assert playing(fake_audio) == [0 if side is Side.P1 else 1]
    channel = fake_audio.mixer.channels[0 if side is Side.P1 else 1]
    assert [c for c in channel.calls if c[0] == 'fadeout'] == []
