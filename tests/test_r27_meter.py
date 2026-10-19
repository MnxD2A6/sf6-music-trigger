"""R27 raw HUD checks; private originals are decoded locally, never captured."""
from pathlib import Path
import sys
from dataclasses import asdict
import math

import cv2
import numpy as np
import pytest

from sf6_music_trigger.domain import Side, SuperCandidateEvent, ConfirmedSuperEvent
from sf6_music_trigger.gauge_features import observe_gauge
from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig
from sf6_music_trigger.hud_profile import NormalizedRoi
from sf6_music_trigger.meter_color_features import MeterColorModel
from sf6_music_trigger.meter_references import MeterReferences
from sf6_music_trigger.meter_level_features import read_meter_level


FIXTURES = Path(__file__).parent / 'fixtures'


def fixture_inputs(data):
    model = MeterColorModel(side=Side.P2, **{
        key: data['model_' + key] if key == 'active_mask' else data['model_' + key].item()
        for key in ('active_mask', 'full_hue', 'hue_tolerance', 'saturation_floor',
                    'value_floor', 'full_fill', 'spent_fill')
    })
    return NormalizedRoi(*data['roi']), MeterReferences(
        Side.P2, data['full_gray'], data['spent_gray'], color_model=model)


def fixture_frames(data):
    frame = np.zeros(tuple(data['shape']), np.uint8)
    x0, y0, x1, y1 = data['bounds']
    for index, crop in zip(data['indices'], data['crops'], strict=True):
        frame[y0:y1, x0:x1] = crop
        yield int(index), frame


def original_frames(start, end):
    path = Path.home() / 'Videos/屏幕录制/SF6/R26.mp4'
    if not path.exists():
        pytest.skip('Private R26 original unavailable')
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), 'Present private original must decode'
    cap.set(cv2.CAP_PROP_POS_FRAMES, start * 30)
    try:
        for index in range(start * 30, end * 30):
            ok, frame = cap.read()
            assert ok, index
            yield index, frame[38:1118, 14:1934].copy()
    finally:
        cap.release()


def inputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.replay_meter_levels import _load_inputs
    root = Path(__file__).resolve().parents[2] / 'SF6_Music_Trigger_GroupTest_0.1.0a1_20260910_r25'
    if not root.exists():
        pytest.skip('Private calibrated release unavailable')
    profile, _, refs, _ = _load_inputs(root, 'profiles/screenshot-sf6-color-20260816.json')
    return profile.p2.meter_roi, refs


def replay(frames, sample_fps=20, trace=False, calibration=None):
    roi, refs = inputs() if calibration is None else calibration
    tracker = GaugeTracker(Side.P2, 'r27-raw-hud', GaugeTrackerConfig())
    events = []
    previous = None
    for index, frame in frames:
        if sample_fps == 20 and index % 3 == 1:
            continue
        observation = observe_gauge(frame, side=Side.P2, frame_id=index,
                                    sampled_at=index / 30, roi=roi, references=refs)
        transition = tracker.update(observation).transition
        state = (observation.meter_level, tracker.stable_meter_level)
        if trace and (state != previous or transition is not None):
            print(index, index / 30, state, observation.normalized_meter, transition, flush=True)
        previous = state
        if isinstance(transition, (SuperCandidateEvent, ConfirmedSuperEvent)):
            events.append(transition)
    return events


@pytest.mark.parametrize('sample_fps', [20, 30])
def test_r27_original_zero_meter_cannot_spend(sample_fps):
    # A zero glyph must never seed a positive baseline and a false spend.
    assert replay(original_frames(158, 168), sample_fps) == []


@pytest.mark.parametrize('sample_fps', [20, 30])
def test_r27_original_real_one_bar_super_retains_rank(sample_fps):
    events = replay(original_frames(249, 257), sample_fps)
    confirmed = [e for e in events if isinstance(e, ConfirmedSuperEvent)]
    # Reviewed frames 7585/7587 show 1 then 0, independent of reader/log rank.
    assert [(e.meter_before, e.meter_after, e.levels_spent) for e in confirmed] == [(1, 0, 1)]


@pytest.mark.parametrize('start,expected', [(158, []), (249, [(1, 0, 1)])])
@pytest.mark.parametrize('sample_fps', [20, 30])
def test_r27_lossless_hud_episode(start, expected, sample_fps):
    with np.load(FIXTURES / f'test_r27_p2_{start}.npz', allow_pickle=False) as data:
        events = replay(fixture_frames(data), sample_fps, calibration=fixture_inputs(data))
    if not expected:
        assert events == []  # Candidates as well as confirmations must be absent.
    else:
        confirmed = [e for e in events if isinstance(e, ConfirmedSuperEvent)]
        assert [(e.meter_before, e.meter_after, e.levels_spent) for e in confirmed] == expected


@pytest.mark.parametrize('start,index,expected', [
    (158, 4841, 0), (158, 4845, 0), (158, 4850, 0), (158, 4854, 0),
    (249, 7478, 1), (249, 7487, 1), (249, 7491, 1), (249, 7572, 1),
])
def test_r27_tinted_background_cannot_invent_two(start, index, expected):
    with np.load(FIXTURES / f'test_r27_p2_{start}.npz', allow_pickle=False) as data:
        roi, refs = fixture_inputs(data)
        frame = next(frame.copy() for i, frame in fixture_frames(data) if i == index)
    reading = read_meter_level(frame, side=Side.P2, meter_roi=roi, color_model=refs.color_model)
    assert reading.level in (None, expected)


# Freeze observed R26 event frames for reviewed R23/R25 positive spends.
# Wrong suppression, rank, extra event, or candidate/confirmation latency fails.
GOLDENS = [
    ('r25', 40, [('confirmed', 1275)]),
    ('r25', 86, [('confirmed', 2670)]),
    ('r25', 249, [('confirmed', 7559)]),
    ('r25', 307, [('confirmed', 9287)]),
    ('r25', 345, [('confirmed', 10424)]),
    ('r25', 428, [('candidate', 12905), ('confirmed', 12906)]),
    ('r23', 34, [('candidate', 1088), ('confirmed', 1089)]),
    ('r23', 88, [('candidate', 2709), ('confirmed', 2711)]),
    ('r23', 165, [('candidate', 5021), ('confirmed', 5022)]),
    ('r23', 217, [('confirmed', 6588)]),
    ('r23', 382, [('confirmed', 11544)]),
    ('r23', 115, [('confirmed', 3513)]),
    ('r23', 243, [('candidate', 7367), ('confirmed', 7368)]),
    ('r23', 287, [('candidate', 8672), ('confirmed', 8673)]),
    ('r23', 290, [('candidate', 8750), ('confirmed', 8751)]),
    ('r23', 368, [('confirmed', 11112)]),
    ('r23', 423, [('candidate', 12744), ('confirmed', 12746)]),
]


@pytest.mark.parametrize('video,start,expected', GOLDENS)
def test_r27_preserves_golden_positive_frames(video, start, expected):
    from test_r26_p2_sequences import CASES, replay as golden_replay
    case = next(c for c in CASES if c[0] == video and c[1] == start)
    events = golden_replay(case)
    actual = [('candidate', e.candidate_frame_id) if isinstance(e, SuperCandidateEvent)
              else ('confirmed', e.confirmed_frame_id) for e in events]
    if (video, start) in {('r25', 40), ('r25', 86)}:
        # R29 deliberately removes the delayed fractional-bar confirmation.
        # Retain event count/rank and the old frame as a no-later-than bound.
        assert len(actual) == 1 and actual[0][0] == 'confirmed'
        assert start * 30 <= actual[0][1] <= expected[0][1]
    else:
        assert actual == expected
    assert [(e.meter_before, e.meter_after) for e in events] == [case[4]] * len(expected)


if __name__ == '__main__':
    if sys.argv[1] == '--extract':
        roi, refs = inputs()
        for start, end in [(158, 168), (249, 257)]:
            path = FIXTURES / f'test_r27_p2_{start}.npz'
            if path.exists():
                raise FileExistsError(path)
            crops, indices = [], []
            # Full meter and glyph, native client size, lossless pixels only.
            bounds = [math.floor(.775*1920), math.floor(.885*1080),
                      math.ceil(.9875*1920), math.ceil(.975*1080)]
            x0, y0, x1, y1 = bounds
            for index, frame in original_frames(start, end):
                crops.append(frame[y0:y1, x0:x1].copy())
                indices.append(index)
            np.savez_compressed(path, crops=np.stack(crops), indices=indices,
                                shape=frame.shape, bounds=bounds,
                                roi=list(asdict(roi).values()),
                                full_gray=refs.full_gray, spent_gray=refs.spent_gray,
                                **{'model_'+k:v for k,v in asdict(refs.color_model).items() if k != 'side'})
            print(path.name, path.stat().st_size)
    else:
        replay(original_frames(int(sys.argv[1]), int(sys.argv[2])), trace=True)
