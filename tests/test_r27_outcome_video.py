"""R27 outcome goldens decode private originals; no outcome/HUD reader mocks.

The gauge visibility input is controlled to challenge both outcome paths;
whole-engine meter detection belongs to the parent integration replay.
"""
from pathlib import Path

import cv2
import pytest

from sf6_music_trigger.config import SuperScope
from sf6_music_trigger.contact_hud import read_health_pair
from sf6_music_trigger.domain import CapturedFrame, ClientGeometry, ConfirmedSuperEvent, Side
from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
from test_outcome import _meters


def r27_frames(start, end):
    path = Path.home() / 'Videos' / '屏幕录制' / 'SF6' / 'R26.mp4'
    if not path.is_file():
        pytest.skip(f'Private original absent: {path.name}')
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30)
        assert capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, end + 1):
            ok, image = capture.read()
            assert ok and image.shape == (1138, 1948, 3)
            client = image[38:1118, 14:1934].copy()
            yield CapturedFrame(index, client, index / 30,
                                ClientGeometry(0, 0, 1920, 1080))
    finally:
        capture.release()


@pytest.mark.parametrize('sample_rate', [20, 30])
@pytest.mark.parametrize('visible', [False, True])
@pytest.mark.parametrize('start,pre,activation,confirm,end,want', [
    (236., 7118 / 30, 237.56666666666666, 237.66666666666666, 246., OutcomeLabel.UNKNOWN),
    (438., 439., 439.1, 439.1666666666667, 452., OutcomeLabel.HIT),
])
def test_r27_parry_abstains_and_landed_ca_survives_fallback(
    sample_rate, visible, start, pre, activation, confirm, end, want,
):
    # Catches brightness-only HIT and false attacker-loss invalidation during CA.
    observer = OutcomeObserver()
    results = []
    begun = False
    for frame in r27_frames(round(start * 30), round(end * 30)):
        if sample_rate == 20 and frame.frame_id % 3 == 1:
            continue
        observer.remember(frame)
        if not begun and frame.monotonic_time >= confirm:
            observer.begin(ConfirmedSuperEvent(
                'r27-ca', Side.P1, confirm, round(pre * 30), frame.frame_id,
                'golden', 3, 0, 3), SuperScope.ALL_SUPERS, activation_at=activation)
            begun = True
        results.extend(observer.update(frame, observations=_meters(
            frame.frame_id, frame.monotonic_time, visible=visible)))
    assert [result.label for result in results] == [want]
    if want is OutcomeLabel.HIT:
        # Independently visible target damage begins at 441; no startup HIT.
        assert 441. <= results[0].decided_at < 442.1
    else:
        assert activation + 3 <= results[0].decided_at <= activation + 8.1
    assert observer.pending_sides == ()


@pytest.mark.parametrize('index', [13197, 13198, 13199, 13200])
def test_r27_curved_foreground_edge_is_not_attacker_damage(index):
    # The cinematic foreground covers the outer endpoint, while attacker
    # health stays near 0.15. A curved white edge is not the straight HUD rim.
    frame = next(r27_frames(index, index))
    health = read_health_pair(frame.image)[0]
    assert health is None or health >= .15


def test_r27_final_ca_visible_target_loss_remains_readable():
    before = next(r27_frames(13169, 13169))
    after = next(r27_frames(13230, 13230))
    assert read_health_pair(before.image) == pytest.approx((.159, .121), abs=.009)
    assert read_health_pair(after.image) == pytest.approx((.152, .042), abs=.009)
