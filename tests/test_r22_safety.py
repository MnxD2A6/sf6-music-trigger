import pytest
from dataclasses import replace

from sf6_music_trigger.audio import AudioAction, AudioScheduler
from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.domain import Side, ConfirmedSuperEvent
from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel, OutcomeObserver
from test_outcome import _Features, _remember_baseline, _frame, _meters, _event
from test_gauge_tracker import _tracker_with_candidate, _level_at, transition
from sf6_music_trigger.domain import GaugeTrackerUpdate
from sf6_music_trigger.audio import AudioKind
from test_engine import _build_harness, _drive, _TrackerSpy, _OutcomeObserverSpy
from test_engine import _event as engine_event, _candidate


@pytest.mark.parametrize('side', list(Side))
@pytest.mark.parametrize('sound', [False, True])
def test_missing_result_fades_without_playing_whiff_under_r25_policy(side, sound):
    decision = OutcomeDecision('event', side, OutcomeLabel.UNKNOWN, 0.0, 9.0, ('missing',))
    config = SideAudioConfig(play_whiff_sound=sound, whiff_track='whiff.wav')
    assert AudioScheduler(side).on_outcome(decision, config) == (
        AudioAction(AudioKind.FADE, side, fade_ms=config.whiff_fade_ms),
    )
    assert decision.label is OutcomeLabel.UNKNOWN


def test_unchanged_picture_is_not_positive_whiff_evidence():
    features = _Features({})
    observer = OutcomeObserver(feature_extractor=features)
    _remember_baseline(observer, features)
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    features.values[(20, Side.P1)] = (0.8, 0.2)
    decisions = observer.update(_frame(20, 4.0, 20), observations=_meters(20, 4.0))
    assert decisions[0].label is OutcomeLabel.UNKNOWN
    assert decisions[0].confidence == 0.0


def test_transient_recovered_meter_does_not_cancel_and_replay_same_super():
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    assert transition(tracker.update(_level_at(13, 0.16, 3))) is None
    assert transition(tracker.update(_level_at(14, 0.17, 3))) is None
    updates = [transition(tracker.update(_level_at(i, 0.18 + (i - 15) * 0.05, 0)))
               for i in range(15, 30)]
    events = [item for item in updates if item is not None]
    assert len(events) == 1
    assert isinstance(events[0], ConfirmedSuperEvent)
    assert events[0].candidate_id == candidate.candidate_id


@pytest.mark.parametrize('side', list(Side))
def test_old_result_cannot_fade_newer_candidate_music(side):
    old = engine_event(side, event_id='old', levels_spent=3, confirmed_at=0.5)
    candidate = _candidate(side)
    outcome = _OutcomeObserverSpy()
    outcome.results[2] = (OutcomeDecision('old', side, OutcomeLabel.WHIFF, 0.8, 1.5, ('old',)),)
    tracker = _TrackerSpy([GaugeTrackerUpdate(old), GaugeTrackerUpdate(candidate), GaugeTrackerUpdate()])
    harness = _build_harness(outcome_observer=outcome, **{f'{side.value.lower()}_tracker': tracker})
    _drive(harness, [(0.5, 'level_3', 'level_3'), (1.0, 'level_0', 'level_0'), (1.5, 'level_0', 'level_0')])
    assert [[a.kind for a in batch] for batch in harness.audio.batches] == [
        [AudioKind.STOP, AudioKind.PLAY], [AudioKind.STOP, AudioKind.PLAY]]


def test_new_confirmed_event_replaces_old_pending_result_window():
    features = _Features({})
    observer = OutcomeObserver(feature_extractor=features)
    _remember_baseline(observer, features)
    first = _event(Side.P1)
    second = replace(first, event_id='new', confirmed_at=2.0)
    observer.begin(first, SuperScope.ALL_SUPERS, activation_at=1.0)
    observer.begin(second, SuperScope.ALL_SUPERS, activation_at=2.0)
    for frame_id, at in [(20, 4.0), (21, 5.0)]:
        features.values[(frame_id, Side.P1)] = (0.8, 0.2)
        result = observer.update(_frame(frame_id, at, frame_id), observations=_meters(frame_id, at))
        if at == 4.0:
            assert result == ()
        else:
            assert [item.event_id for item in result] == ['new']
