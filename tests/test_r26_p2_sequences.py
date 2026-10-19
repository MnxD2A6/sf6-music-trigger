"""Lossless HUD sequences exercise the real reader and tracker, without audio."""
from dataclasses import asdict
from pathlib import Path
import math
import sys

import cv2
import numpy as np
import pytest

from sf6_music_trigger.domain import Side, ConfirmedSuperEvent, SuperCandidateEvent
from sf6_music_trigger.gauge_features import observe_gauge
from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig
from sf6_music_trigger.hud_profile import NormalizedRoi
from sf6_music_trigger.meter_color_features import MeterColorModel
from sf6_music_trigger.meter_references import MeterReferences


ROOT = Path(__file__).parent / 'fixtures'
# Labels come from the reviewed recording, not the production event stream.
# before/after are literal gameplay spends; None means ordinary play/KO.
CASES = [
    ('r25', 117, 121, 'P2', None),
    ('r25', 329, 335, 'P2', None),
    ('r25', 358, 364, 'P2', None),
    ('r25', 40, 45, 'P2', (1, 0)),
    ('r25', 86, 91, 'P2', (1, 0)),
    ('r25', 249, 255, 'P2', (1, 0)),
    ('r25', 307, 313, 'P2', (1, 0)),
    ('r25', 345, 351, 'P2', (1, 0)),
    ('r25', 428, 434, 'P1', (3, 0)),
    ('r23', 34, 39, 'P2', (1, 0)),
    ('r23', 88, 93, 'P2', (2, 0)),
    ('r23', 165, 170, 'P2', (1, 0)),
    ('r23', 217, 223, 'P2', (1, 0)),
    ('r23', 255, 260, 'P2', None),
    ('r23', 382, 388, 'P2', (1, 0)),
    ('r23', 115, 120, 'P1', (3, 0)),
    ('r23', 243, 248, 'P1', (3, 1)),
    ('r23', 287, 291, 'P2', (3, 1)),
    ('r23', 290, 295, 'P2', (1, 0)),
    ('r23', 368, 373, 'P1', (1, 0)),
    ('r23', 423, 428, 'P1', (3, 0)),
]


def fixture_path(case):
    video, start, _, side, _ = case
    return ROOT / f'r26-{video}-{side.lower()}-{start}.npz'


def replay(case, sample_fps=20):
    with np.load(fixture_path(case), allow_pickle=False) as data:
        side = Side(case[3])
        model = MeterColorModel(side=side, **{
            key: data['model_' + key] if key == 'active_mask' else data['model_' + key].item()
            for key in ('active_mask', 'full_hue', 'hue_tolerance', 'saturation_floor',
                        'value_floor', 'full_fill', 'spent_fill')
        })
        references = MeterReferences(side, data['full_gray'], data['spent_gray'], color_model=model)
        roi = NormalizedRoi(*data['roi'])
        tracker = GaugeTracker(side, 'r26-raw-hud', GaugeTrackerConfig())
        frame = np.zeros(tuple(data['shape']), np.uint8)
        x0, y0, x1, y1 = data['bounds']
        events = []
        for index, crop in zip(data['indices'], data['crops'], strict=True):
            if sample_fps == 20 and index % 3 == 1:
                continue
            frame[y0:y1, x0:x1] = crop
            observation = observe_gauge(frame, side=side, frame_id=int(index),
                                        sampled_at=int(index)/30, roi=roi, references=references)
            transition = tracker.update(observation).transition
            if isinstance(transition, (SuperCandidateEvent, ConfirmedSuperEvent)):
                events.append(transition)
        return events


@pytest.mark.parametrize('case', CASES, ids=lambda c: f'{c[0]}-{c[3]}-{c[1]}')
@pytest.mark.parametrize('sample_fps', [20, 30])
def test_raw_hud_does_not_invent_spending_or_suppress_real_super(case, sample_fps):
    events = replay(case, sample_fps)
    if case[4] is None:
        assert events == [], events  # Both false candidates and confirmations.
    else:
        confirmed = [e for e in events if isinstance(e, ConfirmedSuperEvent)]
        assert [(e.meter_before, e.meter_after) for e in confirmed] == [case[4]], events


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.replay_meter_levels import _load_inputs
    base = Path(__file__).resolve().parents[2]
    release = base / 'SF6_Music_Trigger_GroupTest_0.1.0a1_20260910_r25'
    profile, p1, p2, _ = _load_inputs(release, 'profiles/screenshot-sf6-color-20260816.json')
    for case in CASES:
        if fixture_path(case).exists() and '--refresh' not in sys.argv:
            continue
        video, start, end, side_text, _ = case
        side = Side(side_text)
        refs = p1 if side is Side.P1 else p2
        roi = profile.p1.meter_roi if side is Side.P1 else profile.p2.meter_roi
        cap = cv2.VideoCapture(str(Path.home() / f'Videos/屏幕录制/SF6/{video.upper()}.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start*30)
        crops, indices = [], []
        for index in range(start*30, end*30):
            ok, frame = cap.read()
            assert ok
            frame = frame[26:]
            height, width = frame.shape[:2]
            x0 = math.floor((.025 if side is Side.P1 else .775)*width)
            x1 = math.ceil((.225 if side is Side.P1 else .9875)*width)
            y0, y1 = math.floor(.885*height), math.ceil(.975*height)
            crops.append(frame[y0:y1, x0:x1].copy())
            indices.append(index)
        cap.release()
        np.savez_compressed(fixture_path(case), crops=np.stack(crops), indices=indices,
                            shape=frame.shape, bounds=[x0,y0,x1,y1],
                            roi=list(asdict(roi).values()),
                            full_gray=refs.full_gray, spent_gray=refs.spent_gray,
                            **{'model_'+k:v for k,v in asdict(refs.color_model).items() if k != 'side'})
        print(fixture_path(case).name, flush=True)
