"""User-approved unresolved-result behavior differs by selected scope."""
import pytest
from sf6_music_trigger.audio import AudioScheduler, AudioAction, AudioKind
from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.domain import Side
from sf6_music_trigger.outcome import OutcomeObserver, OutcomeDecision, OutcomeLabel
from test_outcome import _frame, _event, _meters


@pytest.mark.parametrize('side',list(Side))
@pytest.mark.parametrize('duration',[0,725,1500])
@pytest.mark.parametrize('whiff_sound',[False,True])
@pytest.mark.parametrize('scope',[SuperScope.SA3_CA_ONLY,SuperScope.ALL_SUPERS])
def test_unknown_follows_scope_without_claiming_whiff_or_playing_secondary(side,duration,whiff_sound,scope):
    config=SideAudioConfig(whiff_fade_ms=duration,play_whiff_sound=whiff_sound,super_scope=scope)
    decision=OutcomeDecision('unresolved',side,OutcomeLabel.UNKNOWN,0.,8.,('contact_not_established',))
    expected=((AudioAction(AudioKind.FADE,side,fade_ms=duration),)
              if scope is SuperScope.SA3_CA_ONLY else ())
    assert AudioScheduler(side).on_outcome(decision,config)==expected
    assert decision.label is OutcomeLabel.UNKNOWN


def test_scene_center_motion_is_not_an_early_block_decision():
    observer=OutcomeObserver()
    observer.remember(_frame(4,.2,0))
    observer.begin(_event(Side.P2),SuperScope.ALL_SUPERS,activation_at=1.)
    for index,time in [(10,1.5),(11,1.6),(12,4.)]:
        frame=_frame(index,time,0)
        frame.image[3:6,14:26]=255
        decisions=observer.update(frame,observations=_meters(index,time))
        if time<4.:
            assert decisions==()
    assert len(decisions)==1 and decisions[0].label is OutcomeLabel.UNKNOWN
    assert AudioScheduler(Side.P2).on_outcome(decisions[0],SideAudioConfig())[0].kind is AudioKind.FADE


@pytest.mark.parametrize('side',list(Side))
def test_unknown_can_only_fade_its_own_current_event(side):
    from test_engine import _build_harness, _event as engine_event
    harness=_build_harness()
    for name in ('old','new'):
        harness.engine._route_confirmation(engine_event(side,event_id=name,levels_spent=3))
    before=len(harness.audio.batches)
    harness.clock.value=10.
    def decision(name):
        return OutcomeDecision(name,side,OutcomeLabel.UNKNOWN,0.,10.,('contact_not_established',))
    harness.engine._route_outcomes((decision('old'),))
    assert len(harness.audio.batches)==before
    harness.engine._route_outcomes((decision('new'),))
    assert harness.audio.batches[before:]==[(AudioAction(AudioKind.FADE,side,fade_ms=1500),)]
    assert harness.diagnostic_records[-1].outcome is OutcomeLabel.UNKNOWN


@pytest.mark.parametrize('side',list(Side))
def test_disabled_side_does_not_execute_fallback_fade(side):
    decision=OutcomeDecision('disabled',side,OutcomeLabel.UNKNOWN,0.,8.,())
    assert AudioScheduler(side).on_outcome(decision,SideAudioConfig(enabled=False))==()
