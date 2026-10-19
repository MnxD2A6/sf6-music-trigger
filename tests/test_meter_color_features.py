from __future__ import annotations

import numpy as np
import pytest

from sf6_music_trigger.domain import Side


def _crop(
    filled_segments: int,
    background: tuple[int, int, int],
    *,
    include_static_decoy: bool = True,
) -> np.ndarray:
    """Synthetic translucent scene with a fixed red decoy and a six-part gauge."""

    image = np.full((48, 180, 3), background, dtype=np.uint8)
    if include_static_decoy:
        image[2:12, 3:38] = (0, 0, 255)
    for index in range(6):
        x0 = 48 + index * 20
        x1 = x0 + 15
        image[29:43, x0:x1] = (0, 0, 255) if index < filled_segments else (55, 55, 55)
    return image


def _gray_crop() -> np.ndarray:
    return np.full((48, 180, 3), 120, dtype=np.uint8)


def _model():
    from sf6_music_trigger.meter_color_features import build_meter_color_model

    return build_meter_color_model(
        Side.P1,
        [_crop(6, (5, 5, 5))] * 5,
        [_crop(0, (5, 5, 5))] * 5,
    )


def test_calibration_excludes_static_coloured_decoy_from_active_gauge_mask() -> None:
    model = _model()

    assert int(np.count_nonzero(model.active_mask)) > 0
    assert bool(model.active_mask[4, 4]) is False


def test_full_and_zero_ignore_material_background_changes_outside_active_mask() -> None:
    from sf6_music_trigger.meter_color_features import extract_meter_fill

    model = _model()

    assert extract_meter_fill(_crop(6, (200, 30, 160)), model) == pytest.approx(
        model.full_fill
    )
    assert extract_meter_fill(_crop(0, (200, 30, 160)), model) == pytest.approx(
        model.spent_fill
    )


def test_three_segments_stay_between_full_and_spent() -> None:
    from sf6_music_trigger.meter_color_features import extract_meter_fill

    model = _model()
    partial = extract_meter_fill(_crop(3, (90, 20, 30)), model)

    assert partial is not None
    assert model.spent_fill < partial < model.full_fill


def test_uniform_low_detail_crop_has_no_feature() -> None:
    from sf6_music_trigger.meter_color_features import extract_meter_fill

    assert extract_meter_fill(_gray_crop(), _model()) is None


def test_vivid_ca_overlay_does_not_erase_spent_colour_candidate() -> None:
    from sf6_music_trigger.meter_color_features import extract_meter_fill

    model = _model()
    spent_meter = _crop(0, (90, 90, 90)).astype(np.float32)
    yy, xx = np.indices(spent_meter.shape[:2])
    purple_overlay = np.empty_like(spent_meter)
    purple_overlay[((xx // 12) + (yy // 12)) % 2 == 0] = (180, 0, 135)
    purple_overlay[((xx // 12) + (yy // 12)) % 2 == 1] = (45, 0, 65)
    ca_obscured = np.rint(spent_meter * 0.25 + purple_overlay * 0.75).astype(
        np.uint8
    )

    fill = extract_meter_fill(ca_obscured, model)

    assert fill is not None
    assert fill == pytest.approx(model.spent_fill, abs=0.08)
    assert extract_meter_fill(_crop(0, (90, 90, 90)), model) == pytest.approx(
        model.spent_fill
    )
