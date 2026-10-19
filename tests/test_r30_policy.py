"""Approved R30 mode/result matrix, through real schedulers and engine."""
import pytest
from sf6_music_trigger.audio import AudioScheduler, AudioKind
from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.domain import Side
from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel
from test_engine import _build_harness, _event, _drive

@pytest.mark.parametrize('side', list(Side))
@pytest.mark.parametrize('scope', list(SuperScope))
@pytest.mark.parametrize('sound', [False, True])
@pytest.mark.parametrize('label', [
    OutcomeLabel.HIT, OutcomeLabel.BLOCK, OutcomeLabel.WHIFF,
    OutcomeLabel.NON_HIT, OutcomeLabel.UNKNOWN,
])
def test_high_tier_matrix(side, scope, sound, label):
    cfg=SideAudioConfig(main_track='main.wav',whiff_track='miss.wav',play_whiff_sound=sound,super_scope=scope)
    h=_build_harness(p1_config=cfg,p2_config=cfg)
    h.clock.value=1.
    h.engine._route_confirmation(_event(side,event_id='matrix',levels_spent=3))
    before=len(h.audio.batches)
    h.clock.value=10.
    h.engine._route_outcomes((OutcomeDecision('matrix',side,label,.9,10.,()),))
    actual=[a for b in h.audio.batches[before:] for a in b]
    if label is OutcomeLabel.HIT or (label is OutcomeLabel.UNKNOWN and scope is SuperScope.ALL_SUPERS):
        assert actual==[]
    elif label in (OutcomeLabel.BLOCK,OutcomeLabel.WHIFF,OutcomeLabel.NON_HIT) and sound:
        assert [a.kind for a in actual]==[AudioKind.STOP,AudioKind.PLAY]
        assert actual[-1].track=='miss.wav'
    else:
        assert [a.kind for a in actual]==[AudioKind.FADE]

@pytest.mark.parametrize('side',list(Side))
@pytest.mark.parametrize('rank',[1,2])
@pytest.mark.parametrize('sound',[False,True])
def test_low_tier_always_holds_four_then_fades_even_old_option_off(side,rank,sound):
    cfg=SideAudioConfig(main_track='main.wav',whiff_track='miss.wav',play_whiff_sound=sound,
                        super_scope=SuperScope.ALL_SUPERS,fade_non_three=False)
    h=_build_harness(p1_config=cfg,p2_config=cfg)
    h.clock.value=1.
    h.engine._route_confirmation(_event(side,event_id='low',levels_spent=rank))
    assert [a.kind for b in h.audio.batches for a in b]==[AudioKind.STOP,AudioKind.PLAY]
    _drive(h,[(4.99,'level_1','level_1')])
    assert not any(a.kind is AudioKind.FADE for b in h.audio.batches for a in b)
    _drive(h,[(5.,'level_1','level_1')])
    assert [a.kind for b in h.audio.batches for a in b]==[AudioKind.STOP,AudioKind.PLAY,AudioKind.FADE]
    assert h.audio.batches[-1][0].fade_ms==1500
