"""Non-three-bar policy: config compatibility and real engine routing."""
from dataclasses import replace
import json
import pytest
from sf6_music_trigger.audio import AudioKind
from sf6_music_trigger.config import AppConfig, SideAudioConfig, SuperScope, load_config, save_config
from sf6_music_trigger.domain import Side
from test_engine import _build_harness, _candidate, _event, _drive

def test_old_config_defaults_off_and_new_policy_round_trips(tmp_path):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'p1':{},'p2':{}}))
    old=load_config(path)
    assert old.p1.fade_non_three is False and old.p2.fade_non_three is False
    new=replace(old,p2=replace(old.p2,fade_non_three=True))
    save_config(path,new)
    assert load_config(path)==new

@pytest.mark.parametrize('bad',[0,1,'true',None])
def test_policy_requires_boolean(bad):
    with pytest.raises(TypeError): SideAudioConfig(fade_non_three=bad)

@pytest.mark.parametrize('side',list(Side))
@pytest.mark.parametrize('spent',[1,2,3])
@pytest.mark.parametrize('preview',[False,True])
def test_confirmed_lower_tier_fades_without_waiting_for_contact(side,spent,preview):
    config=SideAudioConfig(main_track='music.wav',super_scope=SuperScope.ALL_SUPERS,fade_non_three=True)
    h=_build_harness(p1_config=config,p2_config=config)
    candidate=_candidate(side,levels_suspected=spent)
    if preview: h.engine._route_candidate(candidate)
    event=_event(side,levels_spent=spent,candidate_id=candidate.candidate_id if preview else None)
    h.engine._route_confirmation(event)
    h.engine._route_confirmation(event)
    assert [a.kind for b in h.audio.batches for a in b] == [AudioKind.STOP, AudioKind.PLAY]
    _drive(h, [(4., 'level_1', 'level_1')])
    kinds=[a.kind for batch in h.audio.batches for a in batch]
    assert kinds==([AudioKind.STOP,AudioKind.PLAY,AudioKind.FADE] if spent<3 else [AudioKind.STOP,AudioKind.PLAY])
    assert bool(h.outcome_observer.begin_calls)==(spent==3)
    assert h.engine.diagnostic_warning is False
    if spent<3:
        assert h.audio.batches[-1][-1].fade_ms==1500
        assert h.diagnostic_records[-1].audio_actions[-1] is AudioKind.FADE

def test_restricted_scope_silent_lower_tier_cannot_fade_old_music():
    cfg=SideAudioConfig(main_track='music.wav',fade_non_three=True)
    h=_build_harness(p1_config=cfg)
    h.engine._route_confirmation(_event(event_id='old',levels_spent=3))
    h.engine._route_confirmation(_event(event_id='low',levels_spent=1))
    assert len(h.audio.batches)==1

def test_legacy_disabled_option_cannot_disable_mandatory_low_tier_fade():
    cfg=SideAudioConfig(main_track='music.wav',super_scope=SuperScope.ALL_SUPERS)
    h=_build_harness(p1_config=cfg)
    h.engine._route_confirmation(_event(levels_spent=1))
    assert [a.kind for b in h.audio.batches for a in b]==[AudioKind.STOP,AudioKind.PLAY]
    assert h.outcome_observer.begin_calls==[]
    assert h.engine.outcome_statuses[0]=='等待淡出'


@pytest.mark.parametrize('p1,p2',[(False,True),(True,False),(True,True)])
def test_panel_saves_and_restores_independent_policy(tmp_path,p1,p2):
    from sf6_music_trigger.panel import create_panel_application
    from test_panel_view_model import _Audio

    path=tmp_path/'config.json'
    app=create_panel_application(config_path=path,audio_port_factory=lambda root:_Audio())
    app.view_model.p1_fade_non_three=p1
    app.view_model.p2_fade_non_three=p2
    save_config(path,app.view_model.to_config())
    restored=create_panel_application(config_path=path,audio_port_factory=lambda root:_Audio())
    assert restored.view_model.p1_fade_non_three is p1
    assert restored.view_model.p2_fade_non_three is p2


@pytest.mark.parametrize('enabled,track',[(False,'music.wav'),(True,'')])
def test_silent_confirmation_cannot_fade_unowned_music(enabled,track):
    from sf6_music_trigger.audio import AudioScheduler

    cfg=SideAudioConfig(enabled=enabled,main_track=track,super_scope=SuperScope.ALL_SUPERS,fade_non_three=True)
    scheduler=AudioScheduler(Side.P1)
    if enabled and not track:
        with pytest.raises(ValueError,match='requires a main track'):
            scheduler.on_confirmation(_event(levels_spent=1),cfg,False)
    else:
        assert scheduler.on_confirmation(_event(levels_spent=1),cfg,False)==()
