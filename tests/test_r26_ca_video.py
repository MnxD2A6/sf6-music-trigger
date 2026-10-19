"""Real video goldens: default observer must resolve landed CA before fallback.

R25 native diagnostics omit frame IDs. Its 429.5 pre-spend time comes from
the existing offline replay, while activation/confirmation use audio-aligned
native times (430.067 / 430.680). This is an observer replay, not capture QA.
Only absent originals skip on other machines; locally require zero skips.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from sf6_music_trigger.config import SuperScope
from sf6_music_trigger.contact_hud import read_health_pair
from sf6_music_trigger.domain import CapturedFrame, ClientGeometry, ConfirmedSuperEvent, Side
from sf6_music_trigger.outcome import OutcomeObserver, OutcomeLabel
from test_outcome import _meters


def frames(name, start, end):
    path = Path.home() / 'Videos' / '屏幕录制' / 'SF6' / f'{name}.mp4'
    if not path.is_file():
        pytest.skip(f'Local golden video absent: {path.name}')
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert capture.set(cv2.CAP_PROP_POS_FRAMES, round(start * 30))
        for index in range(round(start * 30), round(end * 30) + 1):
            ok, image = capture.read()
            assert ok
            assert image.shape == ((1104, 1902, 3) if name == 'R25' else (1092, 1916, 3))
            client = image[26:].copy()
            yield CapturedFrame(index, client, index / 30,
                                ClientGeometry(0, 0, client.shape[1], client.shape[0]))
    finally:
        capture.release()


@pytest.mark.parametrize('sample_rate', [20, 30])
@pytest.mark.parametrize('name,start,pre,activation,confirmed,end', [
    ('R25', 428.5, 429.5, 430.067, 430.680, 440.),
    ('R23', 116.2, 116.9, 117.413, 117.7, 127.),
    ('R23', 423.4, 424., 424.3, 425.3, 434.),
])
def test_landed_ca_resolves_hit_before_unknown_fade(name, start, pre, activation, confirmed, end, sample_rate):
    observer = OutcomeObserver()
    results = []
    begun = False
    for frame in frames(name, start, end):
        if sample_rate == 20 and frame.frame_id % 3 == 1:
            continue
        observer.remember(frame)
        if not begun and frame.monotonic_time >= confirmed:
            observer.begin(ConfirmedSuperEvent(
                'golden-ca', Side.P1, confirmed, round(pre * 30), frame.frame_id,
                'golden', 3, 0, 3), SuperScope.ALL_SUPERS, activation_at=activation)
            begun = True
        results.extend(observer.update(frame, observations=_meters(
            frame.frame_id, frame.monotonic_time, visible=False)))
    assert [result.label for result in results] == [OutcomeLabel.HIT]
    assert results[0].decided_at < activation + 8
    assert observer.pending_sides == ()


def test_cinematic_occlusion_is_not_a_shortened_attacker_health_bar():
    # Source3541: cinematic foreground clips the live fill; only8/11 rows
    # carry an apparent endpoint, not a sufficiently supported HUD boundary.
    item = next(frames('R23', 3541 / 30, 3541 / 30))
    assert read_health_pair(item.image)[0] is None


def test_historical_jamie_ca_hit_follows_visible_damage_not_startup_flash():
    """3.20s: combo damage 2090. 3.30s: 2513 and landed cinematic.

    Frame source 100 (3.333s) is labelled 3.35s by the historical 20Hz
    replay clock. Thus HIT at 3.35 is after damage, not an early phantom.
    See fixtures/r26-ca-jamie-evidence.jpg; full client, no titlebar.
    """
    path = Path.home() / 'Videos' / '屏幕录制' / 'SF6' / '杰米测试3.mp4'
    if not path.is_file():
        pytest.skip(f'Local golden video absent: {path.name}')
    capture = cv2.VideoCapture(str(path))
    observer = OutcomeObserver()
    results = []
    try:
        for index in [30, 31, 38, 39, 42, 64, 65, 66, 67, 198]:
            assert capture.set(cv2.CAP_PROP_POS_FRAMES, round(index * 30 / 20))
            ok, image = capture.read()
            assert ok and image.shape == (1440, 2560, 3)
            item = CapturedFrame(index, image, index / 20, ClientGeometry(0, 0, 2560, 1440))
            observer.remember(item)
            if index == 39:
                observer.begin(ConfirmedSuperEvent('historical-ca', Side.P1, 1.95, 31, 39,
                                                   'golden', 3, 0, 3),
                               SuperScope.SA3_CA_ONLY, activation_at=1.9)
            result = observer.update(item, observations=_meters(index, index / 20, visible=False))
            if index < 67:
                assert not result, 'No HIT before independently visible sustained damage'
            results.extend(result)
    finally:
        capture.release()
    assert [(result.label, result.decided_at) for result in results] == [(OutcomeLabel.HIT, 3.35)]


@pytest.mark.parametrize('start,pre,activation', [(332.6, 332.8, 333.),
                                               (361.7, 361.9, 362.),
                                               (118.8, 119., 119.2)])
def test_r25_non_super_controls_cannot_supply_supplemental_hit(start, pre, activation):
    # Deliberately supply a three-bar event to challenge the outcome gate even
    # though visual review found no P2 super. Glyph correction belongs upstream.
    observer = OutcomeObserver()
    begun = False
    results = []
    for item in frames('R25', start, activation + .8):
        observer.remember(item)
        if not begun and item.monotonic_time >= activation:
            observer.begin(ConfirmedSuperEvent('non-super-control', Side.P2, item.monotonic_time,
                                               round(pre * 30), item.frame_id, 'golden', 3, 0, 3),
                           SuperScope.ALL_SUPERS, activation_at=activation)
            begun = True
        results.extend(observer.update(item, observations=_meters(item.frame_id, item.monotonic_time,
                                                                  visible=False)))
    # No later unrelated fight damage is attributed to this labelled window.
    missing = CapturedFrame(item.frame_id + 1, np.zeros_like(item.image), activation + 8,
                            item.geometry)
    results.extend(observer.update(missing, observations=_meters(missing.frame_id,
                                                                 missing.monotonic_time, visible=False)))
    assert [result.label for result in results] == [OutcomeLabel.UNKNOWN]
