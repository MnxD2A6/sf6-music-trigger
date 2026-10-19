"""Three agreeing P2 zero glyphs must survive one marginal color sample."""
import pytest
import json
from pathlib import Path
from sf6_music_trigger.domain import Side, SuperCandidateEvent
from sf6_music_trigger.gauge_features import GaugeObservation
from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig
from test_gauge_tracker import obs


def sample(i, side, level, spent, *, structure=False, confidence=.9):
    return obs(i, i / 20, side=side, spent=spent, structure=structure,
               meter_level_supported=True, meter_level=level,
               meter_level_confidence=confidence if level is not None else 0)


@pytest.mark.parametrize('before', [1, 2, 3])
def test_p2_two_strong_color_samples_then_one_marginal_keeps_candidate(before):
    tracker=GaugeTracker(Side.P2,'p',GaugeTrackerConfig())
    for i in range(12):
        tracker.update(sample(i,Side.P2,before,.9))
    assert tracker.update(sample(12,Side.P2,0,.74)).transition is None
    assert tracker.update(sample(13,Side.P2,0,.74)).transition is None
    event=tracker.update(sample(14,Side.P2,0,.69)).transition
    assert isinstance(event,SuperCandidateEvent)
    assert (event.meter_before,event.meter_after)==(before,0)


@pytest.mark.parametrize('side,level,spent,confidence',[
    (Side.P1,0,.69,.9), (Side.P2,0,.59,.9),
    (Side.P2,1,.69,.9), (Side.P2,None,.69,0), (Side.P2,0,.69,.8),
])
def test_flicker_exception_does_not_relax_other_evidence(side,level,spent,confidence):
    tracker=GaugeTracker(side,'p',GaugeTrackerConfig())
    for i in range(12):tracker.update(sample(i,side,2,.9))
    tracker.update(sample(12,side,0,.74))
    tracker.update(sample(13,side,0,.74))
    assert not isinstance(tracker.update(sample(14,side,level,spent,confidence=confidence)).transition,SuperCandidateEvent)


def test_marginal_samples_cannot_build_a_candidate_without_two_strong_samples():
    tracker=GaugeTracker(Side.P2,'p',GaugeTrackerConfig())
    for i in range(12):tracker.update(sample(i,Side.P2,1,.9))
    events=[tracker.update(sample(i,Side.P2,0,.69)).transition for i in range(12,15)]
    assert not any(isinstance(event,SuperCandidateEvent) for event in events)


def test_real_167_second_segment_previews_before_authoritative_confirmation():
    rows=json.loads((Path(__file__).parent/'fixtures/r23-p2-167-observations.json').read_text())
    tracker=GaugeTracker(Side.P2,'sf6-screenshot-color-20260816',GaugeTrackerConfig())
    events=[]
    for row in rows:
        row['side']=Side(row['side'])
        event=tracker.update(GaugeObservation(**row)).transition
        if event is not None:events.append(event)
    assert len(events)==2
    assert isinstance(events[0],SuperCandidateEvent)
    assert events[0].suspected_at==pytest.approx(167.1666666667)
    assert events[1].confirmed_at==pytest.approx(167.4)
    assert events[1].candidate_id==events[0].candidate_id
