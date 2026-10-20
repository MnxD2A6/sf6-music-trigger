import pytest

from sf6_music_trigger.audio import AudioIssue, AudioKind, AudioPlaybackError
from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.domain import Side
from sf6_music_trigger.ref_engine import RefTriggerEngine
from sf6_music_trigger.reframework_events import Connection, RefEventBatch, SuperEnd, SuperStart

P1, P2 = Side.P1, Side.P2


class Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


class Port:
    def __init__(self) -> None:
        self.calls = []
        self.fail_next = None

    def execute(self, actions):
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        self.calls.append(tuple((a.kind.value, a.side.value, a.track, a.fade_ms) for a in actions))


def side_config(scope=SuperScope.ALL_SUPERS, *, whiff_sound=False, enabled=True, track="main.wav"):
    return SideAudioConfig(enabled, track, 0.5, "whiff.wav" if whiff_sound else "", 0.4,
                           whiff_sound, 1500, scope, False)


def make(p1=None, p2=None):
    clock, port, errors = Clock(), Port(), []
    engine = RefTriggerEngine(
        p1_audio_config=p1 or side_config(),
        p2_audio_config=p2 or side_config(track="p2.wav"),
        audio_port=port,
        audio_diagnostic=errors.append,
        clock=clock,
    )
    return engine, port, clock, errors


def feed(engine, clock, *events, connection=Connection.BATTLE, session_changed=False):
    return engine.process(RefEventBatch(clock.now, tuple(events), connection, session_changed))


PLAY_P1 = (("stop", "P1", None, None), ("play", "P1", "main.wav", None))
PLAY_P2 = (("stop", "P2", None, None), ("play", "P2", "p2.wav", None))
FADE_P1 = (("fade", "P1", None, 1500),)


def test_sa3_hit_keeps_music():
    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 3, 1, 21, 1210))
    assert port.calls == [PLAY_P1]
    assert engine.outcome_statuses == ("等待大招结果", None)
    clock.now += 2
    feed(engine, clock, SuperEnd(P1, 3, 1, "hit", "recovered"))
    clock.now += 10
    feed(engine, clock)
    assert port.calls == [PLAY_P1]
    assert engine.outcome_statuses == ("命中", None)


@pytest.mark.parametrize("result,label", [("block", "格挡"), ("whiff", "打空")])
def test_sa3_block_or_whiff_fades_after_four_second_hold(result, label):
    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 3, 1, None, None))
    clock.now += 1.5
    feed(engine, clock, SuperEnd(P1, 3, 1, result, "recovered"))
    assert port.calls == [PLAY_P1] and engine.outcome_statuses[0] == "等待淡出"
    clock.now += 2.4
    feed(engine, clock)
    assert port.calls == [PLAY_P1]
    clock.now += 0.2
    feed(engine, clock)
    assert port.calls == [PLAY_P1, FADE_P1]
    assert engine.outcome_statuses[0] == "淡出中"


def test_late_result_fades_immediately():
    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 4, 1, None, None))
    clock.now += 6
    feed(engine, clock, SuperEnd(P1, 4, 1, "whiff", "recovered"))
    assert port.calls == [PLAY_P1, FADE_P1]


def test_whiff_sound_mode_plays_whiff_track_immediately():
    engine, port, clock, _ = make(p1=side_config(whiff_sound=True))
    feed(engine, clock, SuperStart(P1, 3, 1, None, None), SuperEnd(P1, 3, 1, "block", "recovered"))
    assert port.calls == [PLAY_P1, (("stop", "P1", None, None), ("play", "P1", "whiff.wav", None))]


def test_unknown_keeps_music_in_all_supers_but_fades_in_sa3_ca_only():
    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 3, 1, None, None), SuperEnd(P1, 3, 1, "unknown", "interrupted"))
    clock.now += 5
    feed(engine, clock)
    assert port.calls == [PLAY_P1] and engine.outcome_statuses[0] == "结果不确定"

    engine, port, clock, _ = make(p1=side_config(SuperScope.SA3_CA_ONLY))
    feed(engine, clock, SuperStart(P1, 3, 1, None, None), SuperEnd(P1, 3, 1, "unknown", "interrupted"))
    clock.now += 5
    feed(engine, clock)
    assert port.calls == [PLAY_P1, FADE_P1]


def test_all_supers_low_tier_always_fades_after_hold_even_on_hit():
    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 1, 1, None, None))
    assert engine.outcome_statuses[0] == "等待淡出"
    feed(engine, clock, SuperEnd(P1, 1, 1, "hit", "recovered"))
    clock.now += 4.0
    feed(engine, clock)
    assert port.calls == [PLAY_P1, FADE_P1]


def test_sa3_ca_only_filters_low_tiers():
    engine, port, clock, _ = make(p1=side_config(SuperScope.SA3_CA_ONLY))
    feed(engine, clock, SuperStart(P1, 2, 1, None, None), SuperEnd(P1, 2, 1, "hit", "recovered"))
    assert port.calls == [] and engine.outcome_statuses[0] == "低等级已过滤"


def test_disabled_side_is_silent():
    engine, port, clock, _ = make(p2=side_config(enabled=False, track="p2.wav"))
    feed(engine, clock, SuperStart(P2, 3, 1, None, None), SuperEnd(P2, 3, 1, "whiff", "recovered"))
    assert port.calls == [] and engine.outcome_statuses == (None, None)


def test_later_side_takes_over_and_old_results_do_not_touch_new_music():
    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 3, 1, None, None))
    clock.now += 1
    feed(engine, clock, SuperStart(P2, 3, 1, None, None))
    clock.now += 1
    feed(engine, clock, SuperEnd(P1, 3, 1, "whiff", "recovered"))
    clock.now += 10
    feed(engine, clock)
    assert port.calls == [PLAY_P1, PLAY_P2]


def test_new_super_on_same_side_cancels_pending_fade():
    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 1, 1, None, None))
    clock.now += 2
    feed(engine, clock, SuperEnd(P1, 1, 1, "hit", "recovered"), SuperStart(P1, 3, 2, None, None))
    clock.now += 3
    feed(engine, clock)
    assert port.calls == [PLAY_P1, PLAY_P1]


def test_stale_end_for_unknown_episode_is_ignored():
    engine, port, clock, _ = make()
    feed(engine, clock, SuperEnd(P1, 3, 9, "whiff", "recovered"))
    feed(engine, clock, SuperStart(P1, 3, 1, None, None))
    feed(engine, clock, SuperEnd(P1, 3, 2, "whiff", "recovered"))
    clock.now += 5
    feed(engine, clock)
    assert port.calls == [PLAY_P1]


def test_session_change_forgets_open_episodes():
    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 3, 1, None, None))
    feed(engine, clock, SuperEnd(P1, 3, 1, "whiff", "recovered"), session_changed=True)
    clock.now += 5
    feed(engine, clock)
    assert port.calls == [PLAY_P1]


def test_reset_forces_pending_fade_and_clears_status():
    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 1, 1, None, None))
    engine.reset("listener stopped")
    assert port.calls == [PLAY_P1, FADE_P1]
    assert engine.outcome_statuses == (None, None)


def test_audio_failure_is_reported_and_does_not_take_ownership():
    engine, port, clock, errors = make()
    error = AudioPlaybackError("play", P1, AudioIssue.BACKEND_FAILURE)
    error.invalidated_sides = frozenset({P1})
    port.fail_next = error
    feed(engine, clock, SuperStart(P1, 1, 1, None, None))
    clock.now += 5
    feed(engine, clock)
    assert errors == [error] and port.calls == []


def test_connection_and_chara_are_exposed():
    engine, _, clock, _ = make()
    engine.process(RefEventBatch(clock.now, (), Connection.MENU, False, (21, 6), 2))
    assert engine.connection is Connection.MENU and engine.chara == (21, 6) and engine.rejected_lines == 2
    with pytest.raises(TypeError):
        engine.process(object())


def test_early_result_acts_immediately_and_end_is_not_applied_twice():
    from sf6_music_trigger.reframework_events import SuperResult

    engine, port, clock, _ = make(p1=side_config(whiff_sound=True))
    feed(engine, clock, SuperStart(P1, 3, 1, None, None))
    clock.now += 2.0
    feed(engine, clock, SuperResult(P1, 3, 1, "whiff", "attack_frames_over"))
    whiff = (("stop", "P1", None, None), ("play", "P1", "whiff.wav", None))
    assert port.calls == [PLAY_P1, whiff]
    clock.now += 2.0
    feed(engine, clock, SuperEnd(P1, 3, 1, "whiff", "recovered"))
    clock.now += 5
    feed(engine, clock)
    assert port.calls == [PLAY_P1, whiff]


def test_early_block_fade_still_respects_the_hold():
    from sf6_music_trigger.reframework_events import SuperResult

    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 4, 1, None, None))
    clock.now += 1.6
    feed(engine, clock, SuperResult(P1, 4, 1, "block", "contact"))
    assert port.calls == [PLAY_P1] and engine.outcome_statuses[0] == "等待淡出"
    clock.now += 2.5
    feed(engine, clock)
    assert port.calls == [PLAY_P1, FADE_P1]


def test_first_result_wins_over_a_different_final_result():
    from sf6_music_trigger.reframework_events import SuperResult

    engine, port, clock, _ = make()
    feed(engine, clock, SuperStart(P1, 3, 1, None, None), SuperResult(P1, 3, 1, "hit", "contact"))
    feed(engine, clock, SuperEnd(P1, 3, 1, "whiff", "recovered"))
    clock.now += 6
    feed(engine, clock)
    assert port.calls == [PLAY_P1] and engine.outcome_statuses[0] == "命中"


def test_reset_result_fades_without_whiff_sound_in_any_scope():
    from sf6_music_trigger.reframework_events import SuperResult

    for scope in (SuperScope.ALL_SUPERS, SuperScope.SA3_CA_ONLY):
        engine, port, clock, _ = make(p1=side_config(scope, whiff_sound=True))
        feed(engine, clock, SuperStart(P1, 3, 1, None, None))
        clock.now += 0.7
        feed(engine, clock, SuperResult(P1, 3, 1, "unknown", "reset"))
        assert port.calls == [PLAY_P1, FADE_P1]
