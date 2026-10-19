from __future__ import annotations

import math
from dataclasses import replace
import os
from pathlib import Path
import warnings

import cv2
import numpy as np
import pytest

from sf6_music_trigger.domain import Side
from sf6_music_trigger.hud_profile import NormalizedRoi
from sf6_music_trigger.hud_profile import load_profile
from sf6_music_trigger.meter_level_features import (
    MeterLevelReading,
    _classify_single_component,
    derive_meter_level_roi,
    read_meter_level,
)
from sf6_music_trigger.meter_color_features import METER_FEATURE_SHAPE, MeterColorModel
from sf6_music_trigger.meter_references import load_references


def _side_bgr(side: Side) -> tuple[int, int, int]:
    hsv = np.array([[[0 if side is Side.P1 else 110, 230, 245]]], dtype=np.uint8)
    pixel = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in pixel)


def _model(side: Side) -> MeterColorModel:
    return MeterColorModel(
        side=side,
        active_mask=np.ones(METER_FEATURE_SHAPE, dtype=np.bool_),
        full_hue=0 if side is Side.P1 else 110,
        hue_tolerance=10,
        saturation_floor=120,
        value_floor=100,
        full_fill=1.0,
        spent_fill=0.0,
    )


def _synthetic_level_frame(
    side: Side,
    glyph: str,
    *,
    background_seed: int,
    scale: float = 1.0,
    outline_side: Side | None = None,
    x_shift: int = 0,
) -> tuple[np.ndarray, NormalizedRoi, MeterColorModel]:
    height, width = 240, 480
    rng = np.random.default_rng(background_seed)
    frame = rng.integers(18, 115, size=(height, width, 3), dtype=np.uint8)
    yy = np.linspace(0, 40, height, dtype=np.uint8)[:, None, None]
    frame = np.clip(frame.astype(np.uint16) + yy, 0, 255).astype(np.uint8)
    meter_roi = (
        NormalizedRoi(0.25, 0.65, 0.65, 0.85)
        if side is Side.P1
        else NormalizedRoi(0.35, 0.65, 0.75, 0.85)
    )
    band = derive_meter_level_roi(meter_roi, side)
    x0, y0 = round(band.x0 * width), round(band.y0 * height)
    x1, y1 = round(band.x1 * width), round(band.y1 * height)
    crop = frame[y0:y1, x0:x1]
    outline_thickness = max(5, round(7 * scale))
    core_thickness = max(2, round(3 * scale))
    center_x, center_y = crop.shape[1] // 2 + x_shift, crop.shape[0] // 2

    def point(dx: float, dy: float) -> tuple[int, int]:
        return round(center_x + dx * scale), round(center_y + dy * scale)

    segments: list[tuple[tuple[int, int], tuple[int, int]]]
    if glyph == "0":
        segments = [
            (point(-10, -14), point(10, -14)),
            (point(-10, -14), point(-10, 14)),
            (point(10, -14), point(10, 14)),
            (point(-10, 14), point(10, 14)),
        ]
    elif glyph == "1":
        segments = [
            (point(-3, -10), point(4, -14)),
            (point(4, -14), point(4, 14)),
        ]
    elif glyph in {"2", "3"}:
        segments = [
            (point(-11, -14), point(11, -14)),
            (point(11, -14), point(11, 0)),
            (point(-11, 0), point(11, 0)),
            (
                (point(-11, 0) if glyph == "2" else point(11, 0)),
                (point(-11, 14) if glyph == "2" else point(11, 14)),
            ),
            (point(-11, 14), point(11, 14)),
        ]
    elif glyph == "CA":
        segments = [
            (point(-17, -12), point(-7, -12)),
            (point(-17, -12), point(-17, 12)),
            (point(-17, 12), point(-7, 12)),
            (point(-1, 12), point(8, -12)),
            (point(8, -12), point(17, 12)),
            (point(3, 3), point(13, 3)),
        ]
    else:
        raise ValueError(f"unsupported synthetic glyph: {glyph}")

    for colour, thickness in (
        (_side_bgr(outline_side or side), outline_thickness),
        ((245, 245, 245), core_thickness),
    ):
        for start, end in segments:
            cv2.line(crop, start, end, colour, thickness, cv2.LINE_AA)
    return frame, meter_roi, _model(side)


def test_level_roi_covers_real_glyph_with_outward_span_and_small_meter_overlap() -> None:
    p1 = derive_meter_level_roi(NormalizedRoi(0.065, 0.885, 0.225, 0.975), Side.P1)
    p2 = derive_meter_level_roi(NormalizedRoi(0.775, 0.885, 0.945, 0.975), Side.P2)

    assert p1 == NormalizedRoi(0.025, 0.885, 0.097, 0.975)
    assert p2 == NormalizedRoi(0.911, 0.885, 0.9875, 0.975)


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("glyph, expected", [("0", 0), ("1", 1), ("2", 2), ("3", 3)])
def test_partial_coloured_outline_does_not_clip_white_digit(
    side: Side, glyph: str, expected: int,
) -> None:
    frame, roi, model = _synthetic_level_frame(side, glyph, background_seed=23)
    band = derive_meter_level_roi(roi, side)
    crop = frame[round(band.y0 * 240):round(band.y1 * 240),
                 round(band.x0 * 480):round(band.x1 * 480)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Keep the white glyph intact, but only its upper third has the calibrated
    # outline colour. Live blue/purple glow changes must not cut a 0 into a 3/1.
    coloured = (hsv[..., 1] > 120) & (hsv[..., 2] > 180)
    coloured[:crop.shape[0] // 3] = False
    crop[coloured] = (45, 45, 45)

    reading = read_meter_level(frame, side=side, meter_roi=roi, color_model=model)

    assert reading.level == expected


@pytest.mark.parametrize("hue", [125, 130])
@pytest.mark.parametrize("glyph, expected", [("0", 0), ("1", 1), ("2", 2), ("3", 3)])
def test_p2_violet_digit_glow_is_not_the_cyan_meter_fill(hue, glyph, expected) -> None:
    frame, roi, model = _synthetic_level_frame(Side.P2, glyph, background_seed=23)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    outline = (hsv[..., 0] >= 100) & (hsv[..., 0] <= 120) & (hsv[..., 1] > 120) & (hsv[..., 2] > 180)
    hsv[..., 0][outline] = hue
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    model = replace(model, full_hue=100, hue_tolerance=20, saturation_floor=33, value_floor=147)

    reading = read_meter_level(frame, side=Side.P2, meter_roi=roi, color_model=model)

    assert reading.level == expected


def test_r22_p2_recorded_glyphs_do_not_invent_meter_from_zero() -> None:
    roi = NormalizedRoi(0.775, 0.885, 0.945, 0.975)
    band = derive_meter_level_roi(roi, Side.P2)
    x0, x1 = math.floor(band.x0*1920), math.ceil(band.x1*1920)
    y0, y1 = math.floor(band.y0*1080), math.ceil(band.y1*1080)
    model = replace(_model(Side.P2), full_hue=100, hue_tolerance=20,
                    saturation_floor=33, value_floor=147)
    with np.load(Path(__file__).parent / "fixtures/r22-p2-glyphs.npz", allow_pickle=False) as data:
        for crop, timestamp, expected in zip(data["crops"], data["times"], data["expected"], strict=True):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[y0:y1, x0:x1] = crop
            reading = read_meter_level(frame, side=Side.P2, meter_roi=roi, color_model=model)
            assert reading.level == (None if expected == -1 else expected), timestamp


def test_r22_post_ko_two_with_background_cannot_become_one() -> None:
    roi = NormalizedRoi(0.065, 0.885, 0.225, 0.975)
    band = derive_meter_level_roi(roi, Side.P1)
    x0, x1 = math.floor(band.x0*1920), math.ceil(band.x1*1920)
    y0, y1 = math.floor(band.y0*1080), math.ceil(band.y1*1080)
    model = replace(_model(Side.P1), full_hue=165, hue_tolerance=17,
                    saturation_floor=1, value_floor=122)
    with np.load(Path(__file__).parent / "fixtures/r22-p1-two-with-background.npz", allow_pickle=False) as data:
        for crop, timestamp in zip(data["crops"], data["times"], strict=True):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[y0:y1, x0:x1] = crop
            reading = read_meter_level(frame, side=Side.P1, meter_roi=roi, color_model=model)
            # Background-contaminated 2 may abstain, never claim another level.
            assert reading.level in (None, 2), timestamp


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("level", [0, 1, 2, 3, None])
def test_reading_accepts_only_fixed_alphabet(side: Side, level: int | None) -> None:
    reading = MeterLevelReading(
        side,
        level,
        0.9 if level is not None else 0.0,
        level is not None,
    )

    assert reading.level is level


@pytest.mark.parametrize("side", ["P1", 1, None])
def test_reading_rejects_non_enum_side(side: object) -> None:
    with pytest.raises(TypeError, match="side must be Side"):
        MeterLevelReading(side, 1, 0.9, True)  # type: ignore[arg-type]


@pytest.mark.parametrize("level", [True, False, -1, 4, 1.0, np.int64(1), "1"])
def test_reading_rejects_invalid_level(level: object) -> None:
    expected = TypeError if level in (True, False, 1.0, "1") or isinstance(level, np.integer) else ValueError
    with pytest.raises(expected):
        MeterLevelReading(Side.P1, level, 0.9, True)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence", [math.nan, math.inf, -math.inf, -0.01, 1.01])
def test_reading_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError):
        MeterLevelReading(Side.P1, 1, confidence, True)


@pytest.mark.parametrize("confidence", [True, False, "0.9", None])
def test_reading_rejects_non_real_confidence(confidence: object) -> None:
    with pytest.raises(TypeError, match="confidence must be a real number"):
        MeterLevelReading(Side.P1, 1, confidence, True)  # type: ignore[arg-type]


@pytest.mark.parametrize("visible", [0, 1, None, "yes"])
def test_reading_rejects_non_boolean_visibility(visible: object) -> None:
    with pytest.raises(TypeError, match="visible must be a boolean"):
        MeterLevelReading(Side.P1, 1, 0.9, visible)  # type: ignore[arg-type]


def test_invisible_reading_cannot_carry_level_or_confidence() -> None:
    with pytest.raises(ValueError, match="invisible reading cannot carry a level"):
        MeterLevelReading(Side.P1, 1, 0.9, False)
    with pytest.raises(ValueError, match="invisible reading must have zero confidence"):
        MeterLevelReading(Side.P1, None, 0.1, False)


@pytest.mark.parametrize("side", ["P1", 1, None])
def test_level_roi_rejects_non_enum_side(side: object) -> None:
    with pytest.raises(TypeError, match="side must be Side"):
        derive_meter_level_roi(
            NormalizedRoi(0.1, 0.8, 0.3, 0.9), side  # type: ignore[arg-type]
        )


def test_level_roi_rejects_wrong_roi_type() -> None:
    with pytest.raises(TypeError, match="meter_roi must be NormalizedRoi"):
        derive_meter_level_roi((0.1, 0.8, 0.3, 0.9), Side.P1)  # type: ignore[arg-type]


def test_level_roi_clamps_safely_at_frame_edges() -> None:
    assert derive_meter_level_roi(
        NormalizedRoi(0.0, 0.8, 0.2, 0.9), Side.P1
    ) == NormalizedRoi(0.0, 0.8, 0.04, 0.9)
    assert derive_meter_level_roi(
        NormalizedRoi(0.8, 0.8, 1.0, 0.9), Side.P2
    ) == NormalizedRoi(0.96, 0.8, 1.0, 0.9)


def test_single_pixel_component_returns_none_without_empty_slice_warning() -> None:
    component = np.ones((24, 1), dtype=np.bool_)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _classify_single_component(component) is None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("codex-clipboard-c3c17663-add8-4e3c-b42d-376cd4fc397e.png", 3),
        ("codex-clipboard-7c02fe7b-741d-4239-a358-11fa699daea3.png", 0),
        ("codex-clipboard-572a8515-b9ea-4c41-90b1-dfb54c449f00.png", 3),
    ],
)
def test_private_user_labeled_sf6_screenshots_read_both_sides(
    filename: str,
    expected: int,
) -> None:
    screenshot = Path(os.environ.get("TEMP", "")) / filename
    fixture_root = (
        Path(__file__).resolve().parents[2]
        / "SF6_Music_Trigger_ColorCandidate_20260816"
    )
    profile_path = "profiles/screenshot-sf6-color-20260816.json"
    if not screenshot.is_file():
        pytest.skip("local user-labeled SF6 screenshot fixture is absent")
    if not (fixture_root / profile_path).is_file():
        pytest.skip("local calibrated colour profile fixture is absent")
    profile = load_profile(fixture_root, profile_path)
    references = {
        Side.P1: load_references(
            fixture_root,
            profile_path,
            profile.p1.references_path,
            expected_side=Side.P1,
        ),
        Side.P2: load_references(
            fixture_root,
            profile_path,
            profile.p2.references_path,
            expected_side=Side.P2,
        ),
    }
    frame = cv2.imread(str(screenshot), cv2.IMREAD_COLOR)
    assert frame is not None

    for side, side_profile in ((Side.P1, profile.p1), (Side.P2, profile.p2)):
        reading = read_meter_level(
            frame,
            side=side,
            meter_roi=side_profile.meter_roi,
            color_model=references[side].color_model,
        )
        assert reading.level == expected
        assert reading.visible is True


@pytest.mark.parametrize(
    ("timestamp_seconds", "expected"),
    [
        pytest.param(17.866, (0, 0), id="round-hud-zero-zero"),
        pytest.param(40.533, (0, 1), id="wide-glowing-one"),
        pytest.param(60.733, (None, 2), id="effect-stretched-one"),
        pytest.param(102.400, (3, 3), id="effect-holed-three"),
        pytest.param(137.733, (0, 3), id="cinematic-zero-three"),
        pytest.param(396.800, (None, None), id="partial-ca-must-not-be-two"),
    ],
)
def test_private_real_match_frames_read_visible_meter_levels(
    timestamp_seconds: float,
    expected: tuple[int, int],
) -> None:
    video = Path.home() / "Videos" / "屏幕录制" / "SF6" / "实际测试，误触，没触发，出发错误音频.mp4"
    fixture_root = (
        Path(__file__).resolve().parents[2]
        / "SF6_Music_Trigger_ColorCandidate_20260816"
    )
    profile_path = "profiles/screenshot-sf6-color-20260816.json"
    if not video.is_file():
        pytest.skip("local private SF6 replay fixture is absent")
    if not (fixture_root / profile_path).is_file():
        pytest.skip("local calibrated colour profile fixture is absent")

    profile = load_profile(fixture_root, profile_path)
    references = {
        Side.P1: load_references(
            fixture_root,
            profile_path,
            profile.p1.references_path,
            expected_side=Side.P1,
        ),
        Side.P2: load_references(
            fixture_root,
            profile_path,
            profile.p2.references_path,
            expected_side=Side.P2,
        ),
    }
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok and frame is not None

    actual = []
    for side, side_profile in ((Side.P1, profile.p1), (Side.P2, profile.p2)):
        actual.append(
            read_meter_level(
                frame,
                side=side,
                meter_roi=side_profile.meter_roi,
                color_model=references[side].color_model,
            ).level
        )
    assert tuple(actual) == expected


@pytest.mark.parametrize(("glyph", "level"), [("0", 0), ("1", 1), ("2", 2), ("3", 3), ("CA", 3)])
@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize(("seed", "scale"), [(91, 0.85), (177, 1.0), (903, 1.1)])
def test_fixed_hud_alphabet_over_unrelated_background(
    glyph: str, level: int, side: Side, seed: int, scale: float
) -> None:
    frame, roi, model = _synthetic_level_frame(
        side, glyph, background_seed=seed, scale=scale
    )

    reading = read_meter_level(frame, side=side, meter_roi=roi, color_model=model)

    assert reading.side is side
    assert reading.visible is True
    assert reading.level == level
    assert reading.confidence >= 0.75


@pytest.mark.parametrize("side", list(Side))
def test_wrong_side_outline_is_unknown(side: Side) -> None:
    other = Side.P2 if side is Side.P1 else Side.P1
    frame, roi, model = _synthetic_level_frame(
        side, "3", background_seed=23, outline_side=other
    )

    reading = read_meter_level(frame, side=side, meter_roi=roi, color_model=model)

    assert reading == MeterLevelReading(side, None, 0.0, False)


@pytest.mark.parametrize("side", list(Side))
def test_clipped_glyph_is_unknown(side: Side) -> None:
    shift = -100 if side is Side.P1 else 100
    frame, roi, model = _synthetic_level_frame(
        side, "2", background_seed=55, x_shift=shift
    )

    reading = read_meter_level(frame, side=side, meter_roi=roi, color_model=model)

    assert reading.level is None


@pytest.mark.parametrize("side", list(Side))
def test_unrelated_white_pixels_without_side_outline_are_unknown(side: Side) -> None:
    frame, roi, model = _synthetic_level_frame(side, "1", background_seed=83)
    band = derive_meter_level_roi(roi, side)
    height, width = frame.shape[:2]
    x0, y0 = round(band.x0 * width), round(band.y0 * height)
    x1, y1 = round(band.x1 * width), round(band.y1 * height)
    frame[y0:y1, x0:x1] = 30
    frame[y0 + 8 : y1 - 8, x0 + 10 : x1 - 10] = 245

    reading = read_meter_level(frame, side=side, meter_roi=roi, color_model=model)

    assert reading == MeterLevelReading(side, None, 0.0, False)


@pytest.mark.parametrize("side", list(Side))
def test_full_white_flash_is_unknown(side: Side) -> None:
    frame, roi, model = _synthetic_level_frame(side, "3", background_seed=71)
    frame[:] = 255

    reading = read_meter_level(frame, side=side, meter_roi=roi, color_model=model)

    assert reading == MeterLevelReading(side, None, 0.0, False)


@pytest.mark.parametrize("bad_frame", [None, [], np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float32)])
def test_reader_rejects_invalid_frame_arrays(bad_frame: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        read_meter_level(
            bad_frame,  # type: ignore[arg-type]
            side=Side.P1,
            meter_roi=NormalizedRoi(0.25, 0.65, 0.65, 0.85),
            color_model=_model(Side.P1),
        )


def test_reader_rejects_model_for_other_side() -> None:
    frame, roi, _ = _synthetic_level_frame(Side.P1, "3", background_seed=3)
    with pytest.raises(ValueError, match="color_model side must match"):
        read_meter_level(
            frame,
            side=Side.P1,
            meter_roi=roi,
            color_model=_model(Side.P2),
        )
