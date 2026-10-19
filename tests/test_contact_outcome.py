"""Experimental HUD decisions must never use scene luma as contact evidence."""
import numpy as np
import pytest

from sf6_music_trigger.contact_hud import ContactHudReading, read_contact_hud
from sf6_music_trigger.config import SuperScope
from sf6_music_trigger.domain import Side
from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
from test_outcome import _frame, _event, _meters


def test_production_default_uses_conservative_contact_hud_reader():
    from sf6_music_trigger.engine import VisualTriggerEngine
    from test_engine import _build_harness
    observer=_build_harness(use_default_outcome_observer=True).engine._outcome_observer
    assert isinstance(observer,OutcomeObserver)
    assert observer._contact_reader is read_contact_hud


def test_contact_evidence_survives_privacy_allowlist():
    from test_diagnostic_log import _record
    codes = ('opponent_health_unchanged', 'opponent_drive_changed', 'timer_resumed')
    assert _record(evidence=codes).evidence == codes


def reading(health=.8, drive=.8, tick=False):
    mask = np.zeros((40, 48), dtype=bool)
    if tick:
        mask[:10] = True
    return ContactHudReading(health, health, drive, drive, mask)


def untimed_reading(health=.8, drive=.8):
    return ContactHudReading(health, health, drive, drive, None)


def setup(side=Side.P2):
    values = {4: reading()}
    observer = OutcomeObserver(contact_reader=lambda image: values[int(image[0, 0, 0])])
    observer.remember(_frame(4, .2, 4))
    observer.begin(_event(side), SuperScope.ALL_SUPERS, activation_at=1.)
    return observer, values


def step(observer, values, index, time, value):
    values[index] = value
    return observer.update(_frame(index, time, index), observations=_meters(index, time))


@pytest.mark.parametrize('side', [Side.P1, Side.P2])
def test_block_requires_drive_loss_health_unchanged_and_resumed_timer(side):
    observer, values = setup(side)
    for index in range(10, 14):
        assert not step(observer, values, index, 1.5 + index / 100, reading(drive=.5))
    results = []
    for index in range(14, 19):
        results.extend(step(observer, values, index, 4 + index / 100, reading(drive=.5, tick=True)))
    assert [item.label for item in results] == [OutcomeLabel.BLOCK]
    assert 'opponent_drive_changed' in results[0].evidence


@pytest.mark.parametrize('side',[Side.P1,Side.P2])
def test_whiff_requires_unchanged_health_drive_and_resumed_timer(side):
    observer,values=setup(side)
    results=[]
    for index in range(14,20):
        results.extend(step(observer,values,index,4+index/100,reading(tick=True)))
        if results:
            break
    assert [item.label for item in results]==[OutcomeLabel.WHIFF]
    assert results[0].evidence==(
        'opponent_health_unchanged','opponent_drive_unchanged','timer_resumed',
    )


@pytest.mark.parametrize('side',[Side.P1,Side.P2])
def test_training_mode_whiff_requires_hud_obscuration_then_stable_return(side):
    values={4:untimed_reading()}
    observer=OutcomeObserver(contact_reader=lambda image:values[int(image[0,0,0])])
    observer.remember(_frame(4,.2,4))
    observer.begin(_event(side),SuperScope.ALL_SUPERS,activation_at=1.)
    for index,time in ((10,1.5),(11,1.6)):
        assert not step(observer,values,index,time,untimed_reading(None,None))
    results=[]
    for index in range(12,19):
        results.extend(step(observer,values,index,4+(index-12)/5,untimed_reading()))
        if results:
            break
    assert [item.label for item in results]==[OutcomeLabel.NON_HIT]
    assert 'hud_resumed' in results[0].evidence


def test_training_mode_unchanged_hud_without_obscuration_stays_unknown():
    values={4:untimed_reading()}
    observer=OutcomeObserver(contact_reader=lambda image:values[int(image[0,0,0])])
    observer.remember(_frame(4,.2,4))
    observer.begin(_event(Side.P1),SuperScope.ALL_SUPERS,activation_at=1.)
    for index in range(10,15):
        assert not step(observer,values,index,4+index/100,untimed_reading())
    result=step(observer,values,15,9.,untimed_reading())
    assert result[0].label is OutcomeLabel.UNKNOWN


@pytest.mark.parametrize(
    "partial", [untimed_reading(None, .8), untimed_reading(.8, None)],
    ids=["health-only-missing", "drive-only-missing"],
)
def test_training_mode_partial_hud_loss_cannot_establish_result_window(partial):
    values={4:untimed_reading()}
    observer=OutcomeObserver(contact_reader=lambda image:values[int(image[0,0,0])])
    observer.remember(_frame(4,.2,4))
    observer.begin(_event(Side.P1),SuperScope.ALL_SUPERS,activation_at=1.)
    for index,time in ((10,1.5),(11,1.6)):
        assert not step(observer,values,index,time,partial)
    for index in range(12,17):
        assert not step(observer,values,index,4+index/100,untimed_reading())
    result=step(observer,values,17,9.,untimed_reading())
    assert result[0].label is OutcomeLabel.UNKNOWN


def test_true_hit_wins_over_drive_loss_and_does_not_wait_for_timer():
    observer, values = setup()
    assert not step(observer, values, 10, 1.5, reading(.7, .5))
    result = step(observer, values, 11, 1.6, reading(.7, .5))
    assert result[0].label is OutcomeLabel.HIT


@pytest.mark.parametrize('scope',[SuperScope.ALL_SUPERS,SuperScope.SA3_CA_ONLY])
def test_brief_mid_cinematic_hud_return_cannot_finalize_block_before_late_hit(scope):
    """R32 Juri CA: an early HUD return precedes the cinematic's real damage."""
    values={4:untimed_reading()}
    observer=OutcomeObserver(contact_reader=lambda image:values[int(image[0,0,0])])
    observer.remember(_frame(4,.2,4))
    observer.begin(_event(Side.P1),scope,activation_at=1.)

    for index,time in ((10,1.5),(11,1.6)):
        assert not step(observer,values,index,time,untimed_reading(None,None))
    for index,time in ((12,1.7),(13,1.8),(14,1.9),(15,2.0)):
        assert not step(observer,values,index,time,untimed_reading(.8,.5))
    for index,time in ((16,2.1),(17,2.2)):
        assert not step(observer,values,index,time,untimed_reading(None,None))

    assert not step(observer,values,18,2.3,untimed_reading(.6,.5))
    result=step(observer,values,19,2.4,untimed_reading(.6,.5))
    assert [item.label for item in result]==[OutcomeLabel.HIT]


@pytest.mark.parametrize('value', [reading(None, None, True), reading(.8, None, True)])
def test_missing_contact_is_unknown_never_timeout_whiff(value):
    observer, values = setup()
    for index in range(10, 15):
        assert not step(observer, values, index, 4 + index / 100, value)
    result = step(observer, values, 15, 9., value)
    assert result[0].label is OutcomeLabel.UNKNOWN


def test_one_health_drop_cannot_later_be_reinterpreted_as_block():
    observer, values = setup()
    assert not step(observer, values, 10, 1.5, reading(.7, .5))
    for index in range(11, 20):
        assert not step(observer, values, index, 4 + index / 100, reading(.8, .5, True))
    assert step(observer, values, 20, 9., reading(.8, .5, True))[0].label is OutcomeLabel.UNKNOWN


def test_missing_baseline_and_scene_flash_do_not_create_block():
    observer = OutcomeObserver(contact_reader=read_contact_hud)
    observer.remember(_frame(4, .2, 4))
    observer.begin(_event(Side.P2), SuperScope.ALL_SUPERS, activation_at=1.)
    assert observer.update(_frame(10, 4., 255), observations=_meters(10, 4.)) == ()
    result = observer.update(_frame(11, 9., 255), observations=_meters(11, 9.))
    assert result[0].label is OutcomeLabel.UNKNOWN
