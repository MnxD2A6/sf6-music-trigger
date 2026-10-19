"""Production contact reader on previously labelled R26 SA3/CA clips."""
import pytest
from sf6_music_trigger.config import SuperScope
from sf6_music_trigger.contact_hud import read_contact_hud
from sf6_music_trigger.domain import ConfirmedSuperEvent,Side
from sf6_music_trigger.outcome import OutcomeLabel,OutcomeObserver
from test_outcome import _meters
from test_r27_outcome_video import r27_frames

@pytest.mark.parametrize('sample_rate',[20,30])
@pytest.mark.parametrize('start,pre,activation,confirm,end,want',[
    (236.,7118/30,237.56666666666666,237.66666666666666,246.,OutcomeLabel.NON_HIT),
    (438.,439.,439.1,439.1666666666667,452.,OutcomeLabel.HIT),
])
def test_contact_hud_distinguishes_blue_parry_and_landed_ca(
    sample_rate,start,pre,activation,confirm,end,want,
):
    observer=OutcomeObserver(contact_reader=read_contact_hud)
    results=[]
    begun=False
    for frame in r27_frames(round(start*30),round(end*30)):
        if sample_rate==20 and frame.frame_id%3==1:continue
        observer.remember(frame)
        if not begun and frame.monotonic_time>=confirm:
            observer.begin(ConfirmedSuperEvent(
                'r30-contact',Side.P1,confirm,round(pre*30),frame.frame_id,
                'golden',3,0,3),SuperScope.ALL_SUPERS,activation_at=activation)
            begun=True
        results.extend(observer.update(frame,observations=_meters(
            frame.frame_id,frame.monotonic_time,visible=True)))
    assert [result.label for result in results]==[want]
