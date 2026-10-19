"""Unknown samples must not repeatedly discard already stable valid evidence."""
import pytest
import json
from pathlib import Path
from sf6_music_trigger.gauge_features import GaugeObservation
from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig
from sf6_music_trigger.domain import Side
from sf6_music_trigger.meter_level_features import MeterLevelReading
from sf6_music_trigger.meter_level_tracker import StableMeterLevelTracker

@pytest.mark.parametrize('side',list(Side))
def test_ten_valid_baseline_samples_survive_short_glyph_abstentions(side):
    t=StableMeterLevelTracker(side,stable_samples=3,baseline_samples=10,unknown_reset_samples=5)
    levels=[3,3,3]+[None,3]*7+[1,1,1]
    out=[]
    for i,v in enumerate(levels):
        r=t.update(MeterLevelReading(side,v,.9 if v is not None else 0.,v is not None),i)
        if r:out.append(r)
    assert [(r.before,r.after) for r in out]==[(3,1)]

@pytest.mark.parametrize('side',list(Side))
def test_insufficient_or_lost_baseline_does_not_emit_spend(side):
    for levels in ([3]*9+[1]*3,[3]*10+[None]*5+[1]*3):
        t=StableMeterLevelTracker(side,stable_samples=3,baseline_samples=10,unknown_reset_samples=5)
        assert all(t.update(MeterLevelReading(side,v,.9 if v is not None else 0.,v is not None),i) is None
                   for i,v in enumerate(levels))

@pytest.mark.parametrize('fps',[20,30])
def test_recorded_r29_p1_sa2_is_not_consumed_without_event(fps):
    from sf6_music_trigger.domain import ConfirmedSuperEvent
    t=GaugeTracker(Side.P1,'r29-p1-sa2',GaugeTrackerConfig())
    rows=json.loads((Path(__file__).parent/'fixtures/r30-p1-sa2.json').read_text())
    events=[]
    for d in rows:
        if fps==20 and d['frame_id']%3==1:continue
        d['side']=Side.P1
        e=t.update(GaugeObservation(**d)).transition
        if isinstance(e,ConfirmedSuperEvent):events.append(e)
    assert [(e.meter_before,e.meter_after) for e in events]==[(3,1)]
    assert 277.4<=events[0].confirmed_at<278.


@pytest.mark.parametrize('side',list(Side))
def test_substituted_baseline_values_do_not_manufacture_spend_authority(side):
    """A preserved old level is not a newly observed baseline sample."""
    from sf6_music_trigger.gauge_features import GaugeObservation
    from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig
    from sf6_music_trigger.domain import ConfirmedSuperEvent

    def observation(i,level,fill,structure=True):
        full,spent=(.8,.2) if structure else (.2,.8)
        return GaugeObservation(side=side,frame_id=i,sampled_at=i/20,
            hud_visible=True,full_confidence=full,spent_confidence=spent,
            full_distance=spent,spent_distance=full,spent_structure_verified=structure,
            normalized_meter=fill,meter_level_supported=True,meter_level=level,
            meter_level_confidence=.9 if level is not None else 0.)

    rows=[observation(i,3,.95) for i in range(3)]
    for _ in range(7):
        rows.append(observation(len(rows),None,.95))
        rows.append(observation(len(rows),2,.95))
    rows.extend(observation(len(rows),0,None,False) for _ in range(4))
    tracker=GaugeTracker(side,'synthetic-baseline-guard',GaugeTrackerConfig())
    events=[e for row in rows if isinstance((e:=tracker.update(row).transition),ConfirmedSuperEvent)]
    assert events==[]


def _r30_level_observation(i, level, fill, *, structure=True):
    full, spent = (
        (.8, .2)
        if level in (2, 3) or (fill is not None and fill >= .72)
        else (.02, .98)
    )
    return GaugeObservation(
        side=Side.P1,
        frame_id=i,
        sampled_at=i / 20,
        hud_visible=True,
        full_confidence=full,
        spent_confidence=spent,
        full_distance=spent,
        spent_distance=full,
        spent_structure_verified=structure,
        normalized_meter=fill,
        meter_level_supported=True,
        meter_level=level,
        meter_level_confidence=.9 if level is not None else 0.,
    )


def test_round_intro_colour_conflict_cannot_seed_a_p1_sa3_candidate():
    """R30 52s: intro reads ten numeric 3s over an empty colour meter."""
    tracker = GaugeTracker(Side.P1, 'r30-round-two', GaugeTrackerConfig())
    rows = [
        *(_r30_level_observation(i, 3, 0.) for i in range(10)),
        *(
            _r30_level_observation(i, 0, None, structure=False)
            for i in range(10, 13)
        ),
    ]

    assert [tracker.update(row).transition for row in rows] == [None] * len(rows)


def test_round_intro_short_new_three_baseline_cannot_emit_p1_sa3():
    """R30 3s: a newly promoted 3 collapses to 0 during the intro effect."""
    tracker = GaugeTracker(Side.P1, 'r30-round-one', GaugeTrackerConfig())
    rows = [
        *(_r30_level_observation(i, 2, .55) for i in range(10)),
        *(_r30_level_observation(i, 3, .82) for i in range(10, 15)),
        *(
            _r30_level_observation(i, 0, None, structure=False)
            for i in range(15, 18)
        ),
    ]

    assert [tracker.update(row).transition for row in rows] == [None] * len(rows)


def test_colour_supported_three_baseline_still_emits_p1_sa3():
    tracker = GaugeTracker(Side.P1, 'r30-real-sa3', GaugeTrackerConfig())
    rows = [
        *(_r30_level_observation(i, 3, .9) for i in range(10)),
        *(
            _r30_level_observation(i, 0, None, structure=False)
            for i in range(10, 13)
        ),
    ]

    transitions = [
        item for row in rows if (item := tracker.update(row).transition) is not None
    ]
    assert len(transitions) == 1
    assert transitions[0].meter_before == 3
    assert transitions[0].meter_after == 0


def test_sustained_full_colour_can_seed_unreadable_p1_three_baseline():
    """R31: the visible P1 3 joins the bright floor, but its bar stays full."""
    tracker = GaugeTracker(Side.P1, 'r31-full-colour-fallback', GaugeTrackerConfig())
    rows = [
        *(_r30_level_observation(i, None, .94) for i in range(20)),
        *(
            _r30_level_observation(i, 0, None, structure=False)
            for i in range(20, 23)
        ),
    ]

    transitions = [
        item for row in rows if (item := tracker.update(row).transition) is not None
    ]
    assert len(transitions) == 1
    assert transitions[0].meter_before == 3
    assert transitions[0].meter_after == 0


def test_brief_or_interrupted_full_colour_cannot_seed_p1_three_baseline():
    tracker = GaugeTracker(Side.P1, 'r31-intro-colour-guard', GaugeTrackerConfig())
    rows = [
        _r30_level_observation(0, None, .94),
        _r30_level_observation(1, None, None, structure=False),
        _r30_level_observation(2, None, .97),
        *(
            _r30_level_observation(i, 0, None, structure=False)
            for i in range(3, 6)
        ),
    ]

    assert [tracker.update(row).transition for row in rows] == [None] * len(rows)
