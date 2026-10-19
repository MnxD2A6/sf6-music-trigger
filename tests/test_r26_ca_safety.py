"""CA evidence gates exercised with real R23 HUD pixels, never mocked readers."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sf6_music_trigger.config import SuperScope
from sf6_music_trigger.contact_hud import read_health_pair
from sf6_music_trigger.domain import CapturedFrame, ClientGeometry, ConfirmedSuperEvent, Side
from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
from test_outcome import _meters


@pytest.fixture(scope='module')
def hud_images():
    with np.load(Path(__file__).parent / 'fixtures/r23-contact-hud.npz') as data:
        result = {}
        for time, strip in zip(data['times'], data['bgr_top']):
            image = np.zeros(tuple(data['client_shape']), dtype=np.uint8)
            image[:len(strip)] = strip
            result[float(time)] = image
        return result


def frame(index, time, image):
    return CapturedFrame(index, image, time, ClientGeometry(0, 0, image.shape[1], image.shape[0]))


def event(**changes):
    return replace(ConfirmedSuperEvent('ca', Side.P1, 1.5, 2, 10, 'golden', 3, 0, 3), **changes)


def begin(images, *, before=116.9, count=2, activation=1., **changes):
    observer = OutcomeObserver()
    for index in range(1, count + 1):
        observer.remember(frame(index, .7 + index / 10, images[before]))
    observer.begin(event(confirmed_at=activation + .5, **changes), SuperScope.ALL_SUPERS, activation_at=activation)
    return observer


def step(observer, index, time, image):
    item = frame(index, time, image)
    observer.remember(item)
    return observer.update(item, observations=_meters(index, time, visible=False))


def finish(observer, image):
    return step(observer, 100, 9., image)


def test_independent_health_hit_is_single_terminal_decision(hud_images):
    observer = begin(hud_images)
    assert not step(observer, 10, 1.5, hud_images[120.])
    assert step(observer, 11, 1.6, hud_images[120.])[0].label is OutcomeLabel.HIT
    assert not finish(observer, hud_images[120.])


@pytest.mark.parametrize('changes', [
    {'levels_spent': 1, 'meter_before': 1},
    {'levels_spent': 2, 'meter_before': 2},
    {'levels_spent': None, 'meter_before': None, 'meter_after': None},
    {'pre_spend_frame_id': 3},
])
def test_without_explicit_three_bar_spend_or_cached_anchor_fallback_survives(hud_images, changes):
    observer = begin(hud_images, **changes)
    assert not step(observer, 10, 1.5, hud_images[120.])
    assert not step(observer, 11, 1.6, hud_images[120.])
    assert finish(observer, hud_images[120.])[0].label is OutcomeLabel.UNKNOWN


@pytest.mark.parametrize('count,activation', [(1, 1.), (2, 2.5)])
def test_unconfirmed_or_old_baseline_cannot_protect_ca(hud_images, count, activation):
    observer = begin(hud_images, count=count, activation=activation)
    assert not step(observer, 10, 3., hud_images[120.])
    assert not step(observer, 11, 3.1, hud_images[120.])
    assert step(observer, 100, 10.5, hud_images[120.])[0].label is OutcomeLabel.UNKNOWN


@pytest.mark.parametrize('mode', ['missing', 'gap', 'recovery', 'shape', 'single', 'duplicate'])
def test_discontinuous_damage_does_not_become_hit(hud_images, mode):
    observer = begin(hud_images)
    after = hud_images[120.]
    assert not step(observer, 10, 1.5, after)
    if mode == 'missing':
        assert not step(observer, 11, 1.55, np.zeros_like(after))
    elif mode == 'recovery':
        assert not step(observer, 11, 1.55, hud_images[116.9])
    elif mode == 'shape':
        assert not step(observer, 11, 1.55, after[:-20])
    if mode not in ('single', 'duplicate'):
        assert not step(observer, 12, 2. if mode == 'gap' else 1.6, after)
    if mode == 'duplicate':
        assert not step(observer, 10, 1.5, after)
    assert finish(observer, np.zeros_like(after))[0].label is OutcomeLabel.UNKNOWN


def test_damage_after_deadline_cannot_retroactively_protect_ca(hud_images):
    observer = begin(hud_images)
    assert not step(observer, 10, 8.99, hud_images[120.])
    assert step(observer, 11, 9.01, hud_images[120.])[0].label is OutcomeLabel.UNKNOWN


def test_latest_changed_baseline_does_not_fall_back_to_older_higher_health(hud_images):
    observer = OutcomeObserver()
    for index, time in [(1, .7), (2, .8)]:
        observer.remember(frame(index, time, hud_images[116.9]))
    observer.remember(frame(3, .9, hud_images[120.]))
    observer.begin(event(pre_spend_frame_id=3), SuperScope.ALL_SUPERS, activation_at=1.)
    assert not step(observer, 10, 1.5, hud_images[120.])
    assert not step(observer, 11, 1.6, hud_images[120.])
    assert finish(observer, hud_images[120.])[0].label is OutcomeLabel.UNKNOWN


def test_preconfirmation_attacker_damage_invalidates_attribution(hud_images):
    observer = OutcomeObserver()
    for index in (1, 2):
        observer.remember(frame(index, .7 + index / 10, hud_images[116.9]))
    damaged = hud_images[120.].copy()
    damaged[:, :900] = hud_images[424.][:, :900]
    observer.remember(frame(9, 1.5, damaged))
    observer.begin(event(confirmed_at=1.6), SuperScope.ALL_SUPERS, activation_at=1.)
    hidden = hud_images[120.].copy()
    hidden[:, :900] = 0
    assert not step(observer, 10, 1.6, hidden)
    assert finish(observer, hidden)[0].label is OutcomeLabel.UNKNOWN


def test_target_occlusion_cannot_hide_observed_attacker_damage(hud_images):
    observer = begin(hud_images)
    damaged = hud_images[424.].copy()
    damaged[:, 900:] = 0
    baseline = read_health_pair(hud_images[116.9])
    reading = read_health_pair(damaged)
    assert reading[1] is None
    assert baseline[0] - reading[0] > .012
    assert not step(observer, 10, 1.5, damaged)
    hidden = hud_images[120.].copy()
    hidden[:, :900] = 0
    assert read_health_pair(hidden)[0] is None
    assert not step(observer, 11, 1.6, hidden)
    assert not step(observer, 12, 1.7, hidden)
    assert finish(observer, hidden)[0].label is OutcomeLabel.UNKNOWN


@pytest.mark.parametrize('before,after,side', [(35.9, 38., Side.P2), (35.9, 39., Side.P2),
                                             (116.9, 116.9, Side.P1)])
def test_real_block_and_unchanged_health_keep_unknown_fallback(hud_images, before, after, side):
    observer = begin(hud_images, before=before, side=side)
    assert not step(observer, 10, 1.5, hud_images[after])
    assert not step(observer, 11, 1.6, hud_images[after])
    assert finish(observer, hud_images[after])[0].label is OutcomeLabel.UNKNOWN


@pytest.mark.parametrize('kind', ['black', 'white', 'yellow', 'dim', 'flash'])
def test_offset_health_reader_rejects_scene_or_occlusion_as_damage(hud_images, kind):
    image = hud_images[116.9].copy()
    if kind in ('dim', 'flash'):
        image = np.clip(image.astype(float) * (.5 if kind == 'dim' else 1) +
                        (100 if kind == 'flash' else 0), 0, 255).astype(np.uint8)
    else:
        image[55:105, 1080:1130] = {'black': (0, 0, 0), 'white': (255, 255, 255),
                                  'yellow': (120, 255, 255)}[kind]
    reading = read_health_pair(image)[1]
    assert reading is None or reading >= .29
    observer = begin(hud_images)
    assert not step(observer, 10, 1.5, image)
    assert not step(observer, 11, 1.6, image)
    assert finish(observer, image)[0].label is OutcomeLabel.UNKNOWN
