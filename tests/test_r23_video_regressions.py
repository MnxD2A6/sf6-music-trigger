from pathlib import Path
from dataclasses import replace
import math
import numpy as np
import pytest
from sf6_music_trigger.domain import Side
from sf6_music_trigger.hud_profile import NormalizedRoi
from sf6_music_trigger.meter_level_features import derive_meter_level_roi,read_meter_level
from test_meter_level_features import _model

@pytest.mark.parametrize('index,expected',[(0,1),(1,2),(2,2),(3,2),(4,2),(5,2),(6,0),(7,0),(8,0),(9,0),(10,None)])
def test_r23_p2_real_glyphs_do_not_turn_two_into_zero(index,expected,request):
    if index in {2, 3, 4, 10}:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason="Known R23 glyph contamination; experimental fix regressed KO at 257.6s and was withdrawn"))
    roi=NormalizedRoi(.775,.885,.945,.975)
    band=derive_meter_level_roi(roi,Side.P2)
    x0,x1=math.floor(band.x0*1916),math.ceil(band.x1*1916)
    y0,y1=math.floor(band.y0*1066),math.ceil(band.y1*1066)
    model=replace(_model(Side.P2),full_hue=100,hue_tolerance=20,saturation_floor=33,value_floor=147)
    with np.load(Path(__file__).parent/'fixtures/r23-p2-glyphs.npz',allow_pickle=False) as data:
        frame=np.zeros((1066,1916,3),np.uint8)
        frame[y0:y1,x0:x1]=data['crops'][index]
        reading=read_meter_level(frame,side=Side.P2,meter_roi=roi,color_model=model)
        assert reading.level==expected, data['times'][index]
