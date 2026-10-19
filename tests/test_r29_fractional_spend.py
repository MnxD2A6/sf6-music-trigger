"""R28 reviewed SA1 spends retain a fractional bar; zero is not empty."""
import json
from pathlib import Path

import pytest

from sf6_music_trigger.domain import Side, ConfirmedSuperEvent, SuperCandidateEvent
from sf6_music_trigger.gauge_features import GaugeObservation
from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig


def recorded(side, fps, filename='r29-fractional-observations.json'):
    data = json.loads((Path(__file__).parent / 'fixtures' / filename).read_text())
    for values in data['sequences'][side.value]:
        fields = dict(zip(data['fields'], values, strict=True))
        frame_id = round(fields['sampled_at'] * 30)
        if fps == 20 and frame_id % 3 == 1:
            continue
        yield GaugeObservation(side=side, frame_id=frame_id, hud_visible=True,
                               full_distance=1-fields['full_confidence'],
                               spent_distance=1-fields['spent_confidence'],
                               meter_level_supported=True, **fields)


@pytest.mark.parametrize('fps', [20, 30])
@pytest.mark.parametrize('side,start,deadline', [(Side.P1, 1027.4, 1028.0),
                                              (Side.P2, 437.7667, 438.35)])
def test_recorded_sa1_is_prompt_and_no_followup_false_three(side, start, deadline, fps):
    tracker = GaugeTracker(side, 'r28-reviewed-fraction', GaugeTrackerConfig())
    events = [e for o in recorded(side, fps) if (e := tracker.update(o).transition)]
    confirmed = [e for e in events if isinstance(e, ConfirmedSuperEvent)]
    assert [(e.meter_before, e.meter_after) for e in confirmed] == [(1, 0)]
    assert start - .04 <= confirmed[0].confirmed_at <= deadline
    candidates = [e for e in events if isinstance(e, SuperCandidateEvent)]
    assert len(candidates) <= 1
    assert all(e.meter_before == 1 and e.meter_after == 0 for e in candidates)


def reading(i, level, fill=.77, *, side=Side.P1, at=None):
    return GaugeObservation(side=side, frame_id=i, sampled_at=i/20 if at is None else at,
                            hud_visible=True, full_confidence=fill, spent_confidence=1-fill,
                            full_distance=1-fill, spent_distance=fill,
                            spent_structure_verified=True, normalized_meter=fill,
                            meter_level_supported=True, meter_level=level,
                            meter_level_confidence=.9 if level is not None else 0.)


@pytest.mark.parametrize('fps', [20, 30])
def test_severe_native_lag_reference_stays_timely_offline_without_late_replay(fps):
    # The native sound was at772.208s, but the spend is already visible764.4s.
    # This contrast is evidence for telemetry, NOT a native latency fix claim.
    tracker = GaugeTracker(Side.P2, 'r28-native-lag-reference', GaugeTrackerConfig())
    events = [e for o in recorded(Side.P2, fps, 'r29-severe-latency-observations.json')
              if (e := tracker.update(o).transition)]
    confirmed = [e for e in events if isinstance(e, ConfirmedSuperEvent)]
    assert [(e.meter_before, e.meter_after) for e in confirmed] == [(2, 1)]
    assert 764.2 <= confirmed[0].confirmed_at <= 764.8
    assert all(getattr(e, 'suspected_at', getattr(e, 'confirmed_at', 0)) < 765 for e in events)


@pytest.mark.parametrize('side', list(Side))
@pytest.mark.parametrize('failure', ['brief_zero', 'unrelated_fill', 'stale', 'reset', 'three_baseline'])
def test_fractional_exception_needs_fresh_matching_one_bar_history(side, failure):
    tracker = GaugeTracker(side, 'r29-guards', GaugeTrackerConfig())
    baseline = 3 if failure == 'three_baseline' else 1
    for i in range(20):
        assert tracker.update(reading(i, baseline, side=side)).transition is None
    if failure == 'reset':
        tracker.reset('capture reset')
    events = []
    for i in range(20, 45):
        level = 1 if failure == 'brief_zero' and i >= 23 else 0
        fill = .99 if failure == 'unrelated_fill' else .77
        at = i/20 + 5 if failure == 'stale' else i/20
        event = tracker.update(reading(i, level, fill, side=side, at=at)).transition
        if event:
            events.append(event)
    assert events == []


def test_partial_zero_evidence_cannot_bridge_a_capture_gap():
    tracker = GaugeTracker(Side.P1, 'gap', GaugeTrackerConfig())
    for i in range(20):
        tracker.update(reading(i, 1))
    for i in range(20, 23):
        assert tracker.update(reading(i, 0)).transition is None
    events = [e for i in range(23, 35)
              if (e := tracker.update(reading(i, 0, at=i/20+.3)).transition)]
    assert events == []


@pytest.mark.parametrize('side', list(Side))
@pytest.mark.parametrize('invalid', ['gap', 'fill'])
def test_fractional_approval_is_revoked_before_confirmation_when_evidence_breaks(side, invalid):
    tracker = GaugeTracker(side, 'revoked', GaugeTrackerConfig())
    for i in range(20):
        tracker.update(reading(i, 1, side=side))
    for i, at in enumerate([1., 1.05, 1.10, 1.15, 1.21], 20):
        assert tracker.update(reading(i, 0, at=at, side=side)).transition is None
    events = []
    for i in range(25, 32):
        at = (1.55 if invalid == 'gap' else 1.26) + (i-25)*.05
        fill = .91 if invalid == 'fill' else .77
        event = tracker.update(reading(i, 0, fill, at=at, side=side)).transition
        if event:
            events.append(event)
    assert events == []
