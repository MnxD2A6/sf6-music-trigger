"""Raw recording regressions: background must not invent P2 meter spending."""
from pathlib import Path
from dataclasses import replace
import math

import cv2
import numpy as np
import pytest

from sf6_music_trigger.domain import Side
from sf6_music_trigger.hud_profile import NormalizedRoi
from sf6_music_trigger.meter_level_features import derive_meter_level_roi, read_meter_level
from test_meter_level_features import _model


FIXTURE = Path(__file__).parent / 'fixtures/r26-p2-glyphs.npz'
ROI = NormalizedRoi(.775, .885, .945, .975)
MODEL = replace(_model(Side.P2), full_hue=100, hue_tolerance=20,
                saturation_floor=33, value_floor=147)


def frame_from_crop(crop):
    frame = np.zeros((1078, 1902, 3), np.uint8)
    band = derive_meter_level_roi(ROI, Side.P2)
    frame[math.floor(band.y0*1078):math.ceil(band.y1*1078),
          math.floor(band.x0*1902):math.ceil(band.x1*1902)] = crop
    return frame


@pytest.mark.parametrize('index', range(18))
def test_background_cannot_change_visible_p2_digit(index):
    # Removing contamination rejection must fail on the actual unchanged 2/1,
    # before any tracker or audio policy can hide an invented transition.
    with np.load(FIXTURE, allow_pickle=False) as data:
        reading = read_meter_level(frame_from_crop(data['crops'][index]),
                                  side=Side.P2, meter_roi=ROI, color_model=MODEL)
        assert reading.level in (None, int(data['expected'][index])), data['times'][index]


if __name__ == '__main__':
    # Reproducible lossless HUD-only fixture extraction; no audio or full frames.
    from sf6_music_trigger.hud_profile import crop_normalized
    capture = cv2.VideoCapture(str(Path.home() / 'Videos/屏幕录制/SF6/R25.mp4'))
    times = [119.3667,119.4667,119.5,119.7,119.9,120.0,
             332.8,333.0,333.2,361.9,362.1,362.3,
             331.4,331.4667,332.5333,362.0,362.1667,360.2]
    crops = []
    for timestamp in times:
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp*30))
        ok, frame = capture.read()
        assert ok
        crops.append(crop_normalized(frame[26:], derive_meter_level_roi(ROI, Side.P2)))
    capture.release()
    np.savez_compressed(FIXTURE, crops=np.stack(crops), times=times,
                        expected=[2]*6+[1]*12)
