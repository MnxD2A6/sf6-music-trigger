"""Candidate default-HUD observer, using real local R23 crops.

Fixture: R23.mp4, 1916x1092 at 30 fps; remove the 26-pixel titlebar,
then retain only the top 144 client rows. Times and source frame indices
are embedded in r23-contact-hud.npz. Fractions below are visually measured
bar extents, not contact labels or expectations computed by the observer.
The 425-second regression additionally needs the original local video.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="module")
def crops():
    with np.load(Path(__file__).parent / "fixtures" / "r23-contact-hud.npz") as data:
        return {
            float(time): (strip.copy(), tuple(data["client_shape"]))
            for time, strip in zip(data["times"], data["bgr_top"], strict=True)
        }


def frame(crops, time):
    strip, shape = crops[time]
    image = np.zeros(shape, dtype=np.uint8)
    image[:len(strip)] = strip
    return image


def read(image):
    from sf6_music_trigger.contact_hud import read_contact_hud

    return read_contact_hud(image)


@pytest.mark.parametrize("time,p1,p2", [
    (35.9, .71, .69), (38., .71, .69), (39., .71, .69),
    (89.9, 1., .80), (92.5, .94, .80),
    (166.9, .87, .63), (167.8, .87, .63), (168.8, .79, .63),
])
def test_real_health_extents_exclude_yellow_damage_overlay(crops, time, p1, p2):
    result = read(frame(crops, time))
    assert result.p1_health == pytest.approx(p1, abs=.025)
    assert result.p2_health == pytest.approx(p2, abs=.025)


def test_real_cinematic_model_occlusion_abstains_instead_of_reporting_damage(crops):
    result = read(frame(crops, 91.5))
    assert result.p1_health is None and result.p2_health is None
    assert result.p1_drive is None and result.p2_drive is None
    assert result.timer_mask is None


@pytest.mark.parametrize("time", [169.5, 170.])
def test_real_partial_model_occlusion_does_not_corrupt_other_side(crops, time):
    result = read(frame(crops, time))
    assert result.p1_health is None
    assert result.p2_health == pytest.approx(.63, abs=.025)


@pytest.mark.parametrize("time,p1,p2", [
    (116.9, .18, .31), (120., .18, .24), (125., .18, .11),
    (424., .16, .12), (429., .16, None), (433., .16, None),
])
def test_short_critical_yellow_health_excludes_dark_yellow_trail(crops, time, p1, p2):
    result = read(frame(crops, time))
    assert result.p1_health == pytest.approx(p1, abs=.02)
    if p2 is None:
        assert result.p2_health is None  # No bright anchor is not proof of zero.
    else:
        assert result.p2_health == pytest.approx(p2, abs=.02)


def test_first_p1_sa3_target_health_drop_is_visible_in_critical_yellow(crops):
    before = read(frame(crops, 116.9))
    after = read(frame(crops, 125.))
    assert before.p2_health == pytest.approx(.31, abs=.02)
    assert after.p2_health == pytest.approx(.11, abs=.02)
    assert before.p2_health - after.p2_health > .17


def test_second_p1_sa3_target_health_drop_is_visible_in_critical_yellow(crops):
    video = Path.home() / "Videos" / "屏幕录制" / "SF6" / "R23.mp4"
    if not video.is_file():
        pytest.skip("Original local R23.mp4 is absent; 425-second frame required")
    capture = cv2.VideoCapture(str(video))
    try:
        assert capture.isOpened(), "Existing R23.mp4 must be decodable"
        assert capture.set(cv2.CAP_PROP_POS_MSEC, 425_000)
        ok, image = capture.read()
        assert ok, "Cannot decode the real R23 425-second frame"
        assert image.shape == (1092, 1916, 3)
    finally:
        capture.release()
    before = read(frame(crops, 424.))
    after = read(image[26:])  # Full 1066-row client, without the titlebar.
    assert before.p2_health == pytest.approx(.12, abs=.02)
    assert after.p2_health == pytest.approx(.057, abs=.015)
    assert before.p2_health - after.p2_health > .04


@pytest.mark.parametrize("occluder", [(0, 0, 0), (255, 255, 255), (70, 140, 210)])
def test_critical_yellow_occlusion_is_unknown_not_damage(crops, occluder):
    image = frame(crops, 424.)
    image[57:103, 1080:1100] = occluder
    assert read(image).p2_health is None


def test_yellow_scene_patch_without_outline_is_not_a_critical_health_bar(crops):
    image = frame(crops, 424.)
    image[57:103, 1050:1125] = (120, 255, 255)
    assert read(image).p2_health is None


def test_yellow_anchor_overlay_cannot_hide_normal_pink_live_pixels(crops):
    image = frame(crops, 89.9)
    image[65:93, 790:870] = (120, 255, 255)
    assert read(image).p1_health is None


def test_bright_yellow_bar_above_critical_length_is_not_accepted(crops):
    image = frame(crops, 89.9)
    hsv = cv2.cvtColor(image[:144], cv2.COLOR_BGR2HSV)
    pink = (hsv[..., 0] >= 158) & (hsv[..., 0] <= 177) & (hsv[..., 1] >= 100)
    pink[:, 880:] = False
    hsv[pink] = (30, 135, 255)
    image[:144] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert read(image).p1_health is None


@pytest.mark.parametrize("time,p1,p2", [
    (35.9, 1/6, 1.), (38., 1/6, 1.),
    (89.9, 5/6, 5/6), (166.9, 2/6, 4/6),
])
def test_real_segmented_drive_fractions(crops, time, p1, p2):
    result = read(frame(crops, time))
    assert result.p1_drive == pytest.approx(p1, abs=.045)
    assert result.p2_drive == pytest.approx(p2, abs=.045)


@pytest.mark.parametrize("value", [0, 128, 255])
def test_blank_or_flash_frame_has_no_reliable_hud(value):
    result = read(np.full((1066, 1916, 3), value, dtype=np.uint8))
    assert (result.p1_health, result.p2_health, result.p1_drive, result.p2_drive) == (
        None, None, None, None,
    )
    assert result.timer_mask is None


@pytest.mark.parametrize("occluder", [(0, 0, 0), (255, 255, 255), (70, 140, 210)])
@pytest.mark.parametrize("x0,x1", [(590, 650), (315, 520)])
def test_missing_health_pixels_cannot_be_counted_as_loss(crops, occluder, x0, x1):
    image = frame(crops, 89.9)
    image[57:103, x0:x1] = occluder
    result = read(image)
    assert result.p1_health is None
    assert result.p2_health == pytest.approx(.80, abs=.025)


def test_flat_matching_color_patch_without_bar_edges_is_unreliable(crops):
    image = frame(crops, 89.9)
    image[57:103, 315:875] = cv2.cvtColor(
        np.uint8([[[167, 220, 180]]]), cv2.COLOR_HSV2BGR,
    )[0, 0]
    assert read(image).p1_health is None


def test_yellow_overlay_never_counts_as_live_health(crops):
    image = frame(crops, 89.9)
    hsv = cv2.cvtColor(image[:144], cv2.COLOR_BGR2HSV)
    live = (hsv[:, :, 0] >= 158) & (hsv[:, :, 0] <= 177) & (hsv[:, :, 1] >= 100)
    live[:, 880:] = False
    hsv[:, :, 0][live] = 30
    image[:144] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert read(image).p1_health is None


def test_drive_occlusion_and_solid_green_glow_abstain(crops):
    image = frame(crops, 35.9)
    image[109:135, 1040:1100] = 0
    assert read(image).p2_drive is None
    image[109:135, 1030:1375] = (0, 230, 70)
    assert read(image).p2_drive is None


def test_timer_mask_represents_fixed_numerals_not_background_motion(crops):
    masks = [read(frame(crops, time)).timer_mask for time in (35.9, 38., 39.)]
    for mask in masks:
        assert mask is not None
        assert mask.shape == (40, 48) and mask.dtype == np.bool_
        assert not mask.flags.writeable
    assert np.mean(masks[0] != masks[1]) < .04
    assert np.mean(masks[0] != masks[2]) < .04


def test_timer_changed_digit_produces_material_mask_difference(crops):
    before = read(frame(crops, 89.9)).timer_mask  # visually 84
    after = read(frame(crops, 92.5)).timer_mask  # visually 83
    assert before is not None and after is not None
    assert np.mean(before != after) > .05


def test_long_cinematic_keeps_same_timer_numeral_despite_lighting_change(crops):
    before = read(frame(crops, 120.)).timer_mask  # visually 59
    after = read(frame(crops, 125.)).timer_mask  # still 59
    assert before is not None and after is not None
    assert np.mean(before != after) < .04


@pytest.mark.parametrize("gain,offset", [
    (.6, 0), (.8, 0), (1.1, 0), (1.3, 0), (1., 20), (1., 60), (1., 100),
])
def test_exposure_change_does_not_create_valid_timer_progress(crops, gain, offset):
    image = frame(crops, 35.9)
    before = read(image).timer_mask
    changed = np.clip(image.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    after = read(changed).timer_mask
    assert before is not None
    assert after is None or np.mean(before != after) < .05


@pytest.mark.parametrize("x,sigma,strength", [
    (934, 20, 70), (982, 20, 70), (958, 35, 100), (934, 12, 160),
])
def test_local_violet_glow_is_not_valid_timer_progress(crops, x, sigma, strength):
    image = frame(crops, 35.9)
    before = read(image).timer_mask
    yy, xx = np.mgrid[:144, :image.shape[1]]
    glow = np.exp(-((xx - x) ** 2 + (yy - 85) ** 2) / (2 * sigma ** 2)) * strength
    image[:144] = np.clip(
        image[:144].astype(float) + glow[..., None] * np.array([1., .25, 1.]),
        0, 255,
    ).astype(np.uint8)
    after = read(image).timer_mask
    assert before is not None
    assert after is None or np.mean(before != after) < .05


def test_dark_outlined_drive_slot_is_not_assumed_empty_or_block(crops):
    before = read(frame(crops, 38.))
    after = read(frame(crops, 39.))
    assert after.p1_health == pytest.approx(before.p1_health, abs=.01)
    assert before.p1_drive == pytest.approx(1 / 6, abs=.02)
    assert after.p1_drive is None  # Real BLOCK remains unknown to this helper.


def test_timer_ignores_scene_and_glow_outside_numeral_boxes(crops):
    image = frame(crops, 35.9)
    before = read(image).timer_mask
    changed = np.full_like(image, (255, 0, 255))
    changed[46:127, 909:1007] = image[46:127, 909:1007]
    after = read(changed).timer_mask
    assert before is not None and after is not None
    np.testing.assert_array_equal(before, after)


def test_partial_timer_occlusion_abstains_even_when_other_digit_is_visible(crops):
    image = frame(crops, 35.9)
    image[46:88, 909:957] = 0
    assert read(image).timer_mask is None


def test_scale_and_mild_brightness_variation_preserve_readings(crops):
    image = frame(crops, 35.9)
    baseline = read(image)
    resized = read(cv2.resize(image, (1280, 712), interpolation=cv2.INTER_AREA))
    dimmed = read((image.astype(np.float32) * .9).astype(np.uint8))
    for result in (resized, dimmed):
        assert result.p1_health == pytest.approx(baseline.p1_health, abs=.025)
        assert result.p2_health == pytest.approx(baseline.p2_health, abs=.025)
        assert result.timer_mask is not None
        assert np.mean(result.timer_mask != baseline.timer_mask) < .05


def test_reading_is_frozen_and_input_is_not_mutated(crops):
    image = frame(crops, 35.9)
    original = image.copy()
    result = read(image)
    np.testing.assert_array_equal(image, original)
    with pytest.raises(FrozenInstanceError):
        result.p1_health = .1


@pytest.mark.parametrize("image,error", [
    (None, TypeError), (np.zeros((720, 1280), dtype=np.uint8), ValueError),
    (np.zeros((720, 1280, 4), dtype=np.uint8), ValueError),
    (np.zeros((720, 1280, 3), dtype=np.float32), TypeError),
    (np.zeros((0, 1280, 3), dtype=np.uint8), ValueError),
])
def test_bad_input_rejected_explicitly(image, error):
    with pytest.raises(error):
        read(image)


def test_insufficient_resolution_abstains():
    result = read(np.zeros((180, 320, 3), dtype=np.uint8))
    assert result.p1_health is None and result.timer_mask is None
