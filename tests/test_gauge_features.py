from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

import sf6_music_trigger.gauge_features as gauge_features_module
from sf6_music_trigger.domain import Side
from sf6_music_trigger.gauge_features import (
    GaugeObservation,
    _score_distances,
    observe_gauge,
)
from sf6_music_trigger.hud_profile import NormalizedRoi
from sf6_music_trigger.meter_references import (
    METER_OUTPUT_SHAPE,
    MeterReferences,
    PreprocessedMeter,
    preprocess_meter,
)
from sf6_music_trigger.meter_color_features import MeterColorModel
from sf6_music_trigger.meter_level_features import MeterLevelReading


FULL_FRAME_ROI = NormalizedRoi(0.0, 0.0, 1.0, 1.0)


def _stable_parameter_id(value: object) -> str | None:
    if type(value) is int and value.bit_length() > 1000:
        return "huge-int"
    return None


def _bgr(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def _meter_crops(height: int = 40, width: int = 80) -> tuple[np.ndarray, np.ndarray]:
    row = np.linspace(0, 255, width, dtype=np.uint8)
    full_gray = np.tile(row, (height, 1))
    spent_gray = np.flip(full_gray, axis=1).copy()
    return _bgr(full_gray), _bgr(spent_gray)


def _references(side: Side) -> MeterReferences:
    full_crop, spent_crop = _meter_crops()
    full = preprocess_meter(full_crop)
    spent = preprocess_meter(spent_crop)
    assert full.hud_visible and spent.hud_visible
    return MeterReferences(side, full.normalized_gray, spent.normalized_gray)


def _colour_references(side: Side) -> MeterReferences:
    mask = np.zeros(METER_OUTPUT_SHAPE, dtype=np.bool_)
    mask[8:24, 8:56] = True
    model = MeterColorModel(side, mask, 0, 5, 150, 150, 1.0, 0.0)
    return MeterReferences(
        side,
        np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8),
        np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8),
        color_model=model,
    )


def _colour_references_with_endpoints(
    side: Side,
    *,
    spent_fill: float,
    full_fill: float,
) -> MeterReferences:
    mask = np.zeros(METER_OUTPUT_SHAPE, dtype=np.bool_)
    mask[8:24, 8:56] = True
    model = MeterColorModel(
        side,
        mask,
        0,
        5,
        150,
        150,
        full_fill,
        spent_fill,
    )
    return MeterReferences(
        side,
        np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8),
        np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8),
        color_model=model,
    )


def _colour_meter(filled: bool, background: tuple[int, int, int]) -> np.ndarray:
    crop = np.full((*METER_OUTPUT_SHAPE, 3), background, dtype=np.uint8)
    crop[8:24, 8:56] = (0, 0, 255) if filled else (50, 50, 50)
    return crop


class FeatureFixture:
    def __init__(self) -> None:
        self.full_crop, self.spent_crop = _meter_crops()
        self.p1_roi = NormalizedRoi(0.0, 0.0, 0.5, 1.0)
        self.p2_roi = NormalizedRoi(0.5, 0.0, 1.0, 1.0)
        self.p1_refs = _references(Side.P1)
        self.p2_refs = _references(Side.P2)

    def frame_for(self, side: Side, crop: np.ndarray) -> np.ndarray:
        other = np.full_like(crop, 127)
        if side is Side.P1:
            return np.concatenate((crop, other), axis=1)
        return np.concatenate((other, crop), axis=1)

    def frame_with_p1_full_and_outside_spent_distractor(self) -> np.ndarray:
        return np.concatenate((self.full_crop, self.spent_crop), axis=1)

    def partial_crop(self) -> np.ndarray:
        midpoint = self.full_crop.shape[0] // 2
        return np.concatenate(
            (self.full_crop[:midpoint], self.spent_crop[midpoint:]), axis=0
        )


@pytest.fixture
def feature_fixture() -> FeatureFixture:
    return FeatureFixture()


@pytest.fixture
def p1_roi(feature_fixture: FeatureFixture) -> NormalizedRoi:
    return feature_fixture.p1_roi


@pytest.fixture
def p1_refs(feature_fixture: FeatureFixture) -> MeterReferences:
    return feature_fixture.p1_refs


@pytest.fixture
def hidden_frame(feature_fixture: FeatureFixture) -> np.ndarray:
    height, width = feature_fixture.full_crop.shape[:2]
    return np.zeros((height, width * 2, 3), dtype=np.uint8)


def _valid_observation(**changes: object) -> GaugeObservation:
    values: dict[str, object] = {
        "side": Side.P1,
        "frame_id": 0,
        "sampled_at": 0.0,
        "hud_visible": True,
        "full_confidence": 1.0,
        "spent_confidence": 0.0,
        "full_distance": 0.0,
        "spent_distance": 1.0,
    }
    values.update(changes)
    return GaugeObservation(**values)


def _noncontiguous_clone(value: np.ndarray) -> np.ndarray:
    storage = np.empty(
        (value.shape[0], value.shape[1] * 2, value.shape[2]), dtype=value.dtype
    )
    storage[:, ::2] = value
    storage[:, 1::2] = value
    result = storage[:, ::2]
    assert not result.flags.owndata and not result.flags.c_contiguous
    return result


def _array_state(value: np.ndarray) -> tuple[object, ...]:
    return (
        value.copy(),
        value.shape,
        value.strides,
        value.flags.c_contiguous,
        value.flags.f_contiguous,
        value.flags.writeable,
        value.flags.owndata,
        id(value.base),
    )


def _assert_array_state(value: np.ndarray, before: tuple[object, ...]) -> None:
    (
        contents,
        shape,
        strides,
        c_contiguous,
        f_contiguous,
        writeable,
        own_data,
        base_id,
    ) = before
    assert np.array_equal(value, contents)
    assert value.shape == shape
    assert value.strides == strides
    assert value.flags.c_contiguous is c_contiguous
    assert value.flags.f_contiguous is f_contiguous
    assert value.flags.writeable is writeable
    assert value.flags.owndata is own_data
    assert id(value.base) == base_id


def _reference_state(references: MeterReferences) -> tuple[tuple[object, ...], ...]:
    return (_array_state(references.full_gray), _array_state(references.spent_gray))


def _assert_reference_state(
    references: MeterReferences, before: tuple[tuple[object, ...], ...]
) -> None:
    _assert_array_state(references.full_gray, before[0])
    _assert_array_state(references.spent_gray, before[1])


def test_constructor_is_typed_frozen_and_preserves_huge_time():
    huge = 10**10000
    item = GaugeObservation(
        side=Side.P1,
        frame_id=7,
        sampled_at=huge,
        hud_visible=True,
        full_confidence=0.25,
        spent_confidence=0.75,
        full_distance=0.75,
        spent_distance=0.25,
    )
    assert item.sampled_at is huge
    with pytest.raises(FrozenInstanceError):
        item.frame_id = 8


def test_constructor_cleanly_accepts_and_preserves_huge_fraction_time():
    huge_fraction = Fraction(10**10000, 1)
    item = _valid_observation(sampled_at=huge_fraction)
    assert item.sampled_at is huge_fraction


@pytest.mark.parametrize("side", [Side.P1, Side.P2])
def test_constructor_accepts_each_typed_side(side: Side):
    item = _valid_observation(side=side)
    assert item.side is side


@pytest.mark.parametrize("side", ["P1", "P2", 1, None])
def test_constructor_rejects_untyped_side(side: object):
    with pytest.raises(TypeError):
        _valid_observation(side=side)


@pytest.mark.parametrize("frame_id", [0, 1, 10**100])
def test_constructor_accepts_actual_nonnegative_int_frame_id(frame_id: int):
    item = _valid_observation(frame_id=frame_id)
    assert item.frame_id is frame_id


@pytest.mark.parametrize("frame_id", [True, False, -1, 0.0, np.int64(0), "0", None])
def test_constructor_rejects_non_actual_or_negative_frame_id(frame_id: object):
    with pytest.raises((TypeError, ValueError)):
        _valid_observation(frame_id=frame_id)


@pytest.mark.parametrize(
    "sampled_at",
    [0, 10**10000, Fraction(3, 2), np.float32(1.25)],
    ids=["zero-int", "huge-int", "fraction", "numpy-real"],
)
def test_constructor_accepts_and_preserves_nonnegative_finite_real_time(
    sampled_at: object,
):
    item = _valid_observation(sampled_at=sampled_at)
    assert item.sampled_at is sampled_at


@pytest.mark.parametrize(
    "sampled_at",
    [True, False, -1, Fraction(-1, 2), float("nan"), float("inf"), float("-inf"), "1", None],
)
def test_constructor_rejects_invalid_sampled_time(sampled_at: object):
    with pytest.raises((TypeError, ValueError)):
        _valid_observation(sampled_at=sampled_at)


@pytest.mark.parametrize("hud_visible", [0, 1, "true", None, np.bool_(True)])
def test_constructor_requires_actual_bool_visibility(hud_visible: object):
    with pytest.raises(TypeError):
        _valid_observation(hud_visible=hud_visible)


@pytest.mark.parametrize(
    "spent_structure_verified",
    [0, 1, "false", None, np.bool_(False)],
)
def test_constructor_requires_actual_bool_spent_structure_verification(
    spent_structure_verified: object,
):
    with pytest.raises(TypeError):
        _valid_observation(
            spent_structure_verified=spent_structure_verified,
        )


def test_unverified_structure_is_only_valid_for_visible_spent_evidence():
    item = _valid_observation(
        full_confidence=0.0,
        spent_confidence=1.0,
        full_distance=1.0,
        spent_distance=0.0,
        spent_structure_verified=False,
    )
    assert item.spent_structure_verified is False

    with pytest.raises(ValueError):
        _valid_observation(
            hud_visible=False,
            full_confidence=0.0,
            spent_confidence=0.0,
            full_distance=1.0,
            spent_distance=1.0,
            spent_structure_verified=False,
        )
    with pytest.raises(ValueError):
        _valid_observation(spent_structure_verified=False)


def test_constructor_preserves_every_accepted_metric_scalar():
    full_distance = Fraction(3, 4)
    spent_distance = Fraction(1, 4)
    full_confidence = Fraction(1, 4)
    spent_confidence = Fraction(3, 4)
    item = _valid_observation(
        full_confidence=full_confidence,
        spent_confidence=spent_confidence,
        full_distance=full_distance,
        spent_distance=spent_distance,
    )
    assert item.full_confidence is full_confidence
    assert item.spent_confidence is spent_confidence
    assert item.full_distance is full_distance
    assert item.spent_distance is spent_distance


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        (field_name, invalid)
        for field_name in (
            "full_confidence",
            "spent_confidence",
            "full_distance",
            "spent_distance",
        )
        for invalid in (
            True,
            False,
            "0.5",
            None,
            -0.01,
            1.01,
            float("nan"),
            float("inf"),
            float("-inf"),
            10**10000,
        )
    ],
    ids=_stable_parameter_id,
)
def test_constructor_rejects_invalid_metric_with_clean_type_or_value_error(
    field_name: str, invalid: object
):
    with pytest.raises((TypeError, ValueError)):
        _valid_observation(**{field_name: invalid})


def test_hidden_constructor_requires_exact_fail_closed_tuple():
    item = _valid_observation(
        hud_visible=False,
        full_confidence=0.0,
        spent_confidence=0.0,
        full_distance=1.0,
        spent_distance=1.0,
    )
    assert (
        item.full_confidence,
        item.spent_confidence,
        item.full_distance,
        item.spent_distance,
    ) == (0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"full_confidence": 0.01},
        {"spent_confidence": 0.01},
        {"full_distance": 0.99},
        {"spent_distance": 0.99},
    ],
)
def test_hidden_constructor_rejects_every_non_sentinel_combination(
    changes: dict[str, object],
):
    values: dict[str, object] = {
        "hud_visible": False,
        "full_confidence": 0.0,
        "spent_confidence": 0.0,
        "full_distance": 1.0,
        "spent_distance": 1.0,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        _valid_observation(**values)


def test_visible_constructor_requires_cross_distance_confidences():
    item = _valid_observation(
        full_confidence=0.25,
        spent_confidence=0.75,
        full_distance=0.75,
        spent_distance=0.25,
    )
    assert item.full_confidence == item.spent_distance
    assert item.spent_confidence == item.full_distance


def test_visible_constructor_accepts_ambiguous_zero_total():
    item = _valid_observation(
        full_confidence=0.5,
        spent_confidence=0.5,
        full_distance=0.0,
        spent_distance=0.0,
    )
    assert (item.full_confidence, item.spent_confidence) == (0.5, 0.5)


def test_visible_constructor_accepts_only_the_documented_absolute_tolerance():
    accepted = _valid_observation(
        full_confidence=0.25 + 5e-13,
        spent_confidence=0.75 - 5e-13,
        full_distance=0.75,
        spent_distance=0.25,
    )
    assert accepted.full_confidence == 0.25 + 5e-13
    with pytest.raises(ValueError):
        _valid_observation(
            full_confidence=0.25 + 2e-12,
            spent_confidence=0.75,
            full_distance=0.75,
            spent_distance=0.25,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"full_confidence": 0.0},
        {"spent_confidence": 1.0},
        {"full_confidence": 0.4, "spent_confidence": 0.6},
    ],
)
def test_visible_constructor_rejects_inconsistent_confidence_combinations(
    changes: dict[str, object],
):
    with pytest.raises(ValueError):
        _valid_observation(**changes)


def test_frozen_output_is_scalar_only_and_retains_no_ndarray():
    item = _valid_observation()
    assert not any(isinstance(value, np.ndarray) for value in vars(item).values())
    assert tuple(vars(item)) == (
        "side",
        "frame_id",
        "sampled_at",
        "hud_visible",
        "full_confidence",
        "spent_confidence",
        "full_distance",
        "spent_distance",
        "spent_structure_verified",
        "normalized_meter",
        "meter_level_supported",
        "meter_level",
        "meter_level_confidence",
    )


def test_hidden_is_exact_fail_closed_sentinel(hidden_frame, p1_roi, p1_refs):
    item = observe_gauge(
        hidden_frame,
        side=Side.P1,
        frame_id=3,
        roi=p1_roi,
        references=p1_refs,
        sampled_at=1.2,
    )
    assert not item.hud_visible
    assert (
        item.full_confidence,
        item.spent_confidence,
        item.full_distance,
        item.spent_distance,
    ) == (0.0, 0.0, 1.0, 1.0)


def test_observe_cleanly_accepts_and_preserves_huge_fraction_time(feature_fixture):
    huge_fraction = Fraction(10**10000, 1)
    item = observe_gauge(
        feature_fixture.frame_for(Side.P1, feature_fixture.full_crop),
        side=Side.P1,
        frame_id=3,
        roi=feature_fixture.p1_roi,
        references=feature_fixture.p1_refs,
        sampled_at=huge_fraction,
    )
    assert item.sampled_at is huge_fraction


def test_non_full_roi_ignores_outside_distractor(feature_fixture):
    item = observe_gauge(
        feature_fixture.frame_with_p1_full_and_outside_spent_distractor(),
        side=Side.P1,
        frame_id=4,
        roi=feature_fixture.p1_roi,
        references=feature_fixture.p1_refs,
        sampled_at=1.3,
    )
    assert item.hud_visible
    assert item.full_confidence >= 0.78
    assert item.spent_confidence < 0.72


def test_zero_total_scoring_is_ambiguous():
    assert _score_distances(0.0, 0.0) == (0.5, 0.5)


@pytest.mark.parametrize(
    ("full_distance", "spent_distance", "expected"),
    [
        (0.0, 1.0, (1.0, 0.0)),
        (1.0, 0.0, (0.0, 1.0)),
        (Fraction(3, 4), Fraction(1, 4), (Fraction(1, 4), Fraction(3, 4))),
        (
            Fraction(1, 5),
            Fraction(3, 10),
            (Fraction(3, 5), Fraction(2, 5)),
        ),
        (np.float32(0.2), np.float32(0.8), (np.float32(0.8), np.float32(0.2))),
    ],
)
def test_score_distances_accepts_finite_non_bool_unit_interval_reals(
    full_distance: object, spent_distance: object, expected: tuple[object, object]
):
    assert _score_distances(full_distance, spent_distance) == expected


@pytest.mark.parametrize(
    ("full_distance", "spent_distance"),
    [
        (True, 0.5),
        (0.5, False),
        ("0.5", 0.5),
        (0.5, None),
        (-0.01, 0.5),
        (0.5, -0.01),
        (1.01, 0.5),
        (0.5, 1.01),
        (float("nan"), 0.5),
        (0.5, float("inf")),
        (float("-inf"), 0.5),
        (10**10000, 0.5),
    ],
    ids=_stable_parameter_id,
)
def test_score_distances_rejects_invalid_inputs_cleanly(
    full_distance: object, spent_distance: object
):
    with pytest.raises((TypeError, ValueError)):
        _score_distances(full_distance, spent_distance)


@pytest.mark.parametrize(
    ("side", "state"),
    [
        (Side.P1, "full"),
        (Side.P1, "spent"),
        (Side.P2, "full"),
        (Side.P2, "spent"),
    ],
)
def test_exact_full_and_spent_are_routed_for_both_sides(
    feature_fixture: FeatureFixture, side: Side, state: str
):
    crop = (
        feature_fixture.full_crop if state == "full" else feature_fixture.spent_crop
    )
    roi = feature_fixture.p1_roi if side is Side.P1 else feature_fixture.p2_roi
    references = (
        feature_fixture.p1_refs if side is Side.P1 else feature_fixture.p2_refs
    )
    item = observe_gauge(
        feature_fixture.frame_for(side, crop),
        side=side,
        frame_id=10,
        roi=roi,
        references=references,
        sampled_at=2.0,
    )
    assert item.side is side and item.hud_visible
    if state == "full":
        assert (item.full_confidence, item.spent_confidence, item.full_distance) == (
            1.0,
            0.0,
            0.0,
        )
        assert item.spent_distance > 0.0
    else:
        assert (item.full_confidence, item.spent_confidence, item.spent_distance) == (
            0.0,
            1.0,
            0.0,
        )
        assert item.full_distance > 0.0


def test_wrong_side_references_reject_before_crop_or_preprocess(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def unexpected_crop(*args: object, **kwargs: object) -> np.ndarray:
        calls.append("crop")
        raise AssertionError("crop_normalized must not be called")

    def unexpected_preprocess(*args: object, **kwargs: object) -> PreprocessedMeter:
        calls.append("preprocess")
        raise AssertionError("preprocess_meter must not be called")

    monkeypatch.setattr(gauge_features_module, "crop_normalized", unexpected_crop)
    monkeypatch.setattr(gauge_features_module, "preprocess_meter", unexpected_preprocess)
    with pytest.raises(ValueError):
        observe_gauge(
            np.zeros((4, 4, 3), dtype=np.uint8),
            side=Side.P1,
            frame_id=0,
            roi=FULL_FRAME_ROI,
            references=_references(Side.P2),
            sampled_at=0,
        )
    assert calls == []


def test_hsv_route_recognizes_full_meter_over_changed_backgrounds(
    monkeypatch: pytest.MonkeyPatch,
):
    references = _colour_references(Side.P1)

    def unexpected_preprocess(*args: object, **kwargs: object) -> PreprocessedMeter:
        raise AssertionError("preprocess_meter must not be called for HSV references")

    monkeypatch.setattr(gauge_features_module, "preprocess_meter", unexpected_preprocess)
    item = observe_gauge(
        _colour_meter(True, (210, 40, 160)),
        side=Side.P1,
        frame_id=1,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=1,
    )

    assert item.hud_visible
    assert item.full_confidence == 1.0
    assert item.spent_confidence == 0.0


def test_hsv_route_returns_hidden_for_low_colour_crop():
    item = observe_gauge(
        np.full((*METER_OUTPUT_SHAPE, 3), 80, dtype=np.uint8),
        side=Side.P1,
        frame_id=1,
        roi=FULL_FRAME_ROI,
        references=_colour_references(Side.P1),
        sampled_at=1,
    )

    assert item.hud_visible is False
    assert (item.full_confidence, item.spent_confidence) == (0.0, 0.0)


def test_hsv_route_marks_detailed_desktop_spent_candidate_unverified():
    yy, xx = np.indices(METER_OUTPUT_SHAPE)
    gray = np.where(((xx // 6) + (yy // 6)) % 2 == 0, 235, 40).astype(
        np.uint8
    )
    desktop_like_crop = _bgr(gray)

    item = observe_gauge(
        desktop_like_crop,
        side=Side.P1,
        frame_id=2,
        roi=FULL_FRAME_ROI,
        references=_colour_references(Side.P1),
        sampled_at=2,
    )

    assert item.hud_visible
    assert item.full_confidence == 0.0
    assert item.spent_confidence == 1.0
    assert item.spent_structure_verified is False


def test_hsv_route_accepts_colour_spent_when_gray_structure_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    references = _colour_references(Side.P1)
    calls: list[np.ndarray] = []

    def matching_spent_structure(crop: np.ndarray) -> PreprocessedMeter:
        calls.append(crop)
        return PreprocessedMeter(references.spent_gray, 120.0, 8.0, True)

    monkeypatch.setattr(
        gauge_features_module,
        "preprocess_meter",
        matching_spent_structure,
    )
    item = observe_gauge(
        _colour_meter(False, (90, 90, 90)),
        side=Side.P1,
        frame_id=3,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=3,
    )

    assert len(calls) == 1
    assert item.hud_visible
    assert item.full_confidence == 0.0
    assert item.spent_confidence == 1.0
    assert item.spent_structure_verified is True


@pytest.mark.parametrize(
    ("fill", "want"),
    [
        (0.20, 0.0),
        (0.40, 1 / 3),
        (0.60, 2 / 3),
        (0.80, 1.0),
    ],
)
def test_color_observation_normalizes_fill_between_calibrated_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    fill: float,
    want: float,
) -> None:
    references = _colour_references_with_endpoints(
        Side.P1,
        spent_fill=0.20,
        full_fill=0.80,
    )
    monkeypatch.setattr(gauge_features_module, "extract_meter_fill", lambda *_: fill)
    monkeypatch.setattr(
        gauge_features_module,
        "preprocess_meter",
        lambda _crop: PreprocessedMeter(
            references.spent_gray,
            120.0,
            8.0,
            True,
        ),
    )

    observation = observe_gauge(
        np.full((*METER_OUTPUT_SHAPE, 3), 90, dtype=np.uint8),
        side=Side.P1,
        frame_id=20,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=2.0,
    )

    assert observation.normalized_meter == pytest.approx(want)


def test_grayscale_hidden_and_unverified_observations_have_no_normalized_meter(
    feature_fixture: FeatureFixture,
    hidden_frame: np.ndarray,
) -> None:
    grayscale = observe_gauge(
        feature_fixture.frame_for(Side.P1, feature_fixture.full_crop),
        side=Side.P1,
        frame_id=21,
        roi=feature_fixture.p1_roi,
        references=feature_fixture.p1_refs,
        sampled_at=2.1,
    )
    hidden = observe_gauge(
        hidden_frame,
        side=Side.P1,
        frame_id=22,
        roi=feature_fixture.p1_roi,
        references=feature_fixture.p1_refs,
        sampled_at=2.2,
    )
    unverified = _valid_observation(
        full_confidence=0.0,
        spent_confidence=1.0,
        full_distance=1.0,
        spent_distance=0.0,
        spent_structure_verified=False,
    )

    assert grayscale.normalized_meter is None
    assert hidden.normalized_meter is None
    assert unverified.normalized_meter is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "normalized_meter must be a real number"),
        ("0.5", "normalized_meter must be a real number"),
        (-0.01, "normalized_meter must be between zero and one"),
        (1.01, "normalized_meter must be between zero and one"),
        (float("nan"), "normalized_meter must be between zero and one"),
    ],
)
def test_constructor_rejects_invalid_normalized_meter(
    value: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=f"^{message}$"):
        _valid_observation(normalized_meter=value)


def test_hidden_or_unverified_observation_rejects_invented_meter_evidence() -> None:
    with pytest.raises(ValueError, match="hidden"):
        _valid_observation(
            hud_visible=False,
            full_confidence=0.0,
            spent_confidence=0.0,
            full_distance=1.0,
            spent_distance=1.0,
            normalized_meter=0.0,
        )
    with pytest.raises(ValueError, match="unverified"):
        _valid_observation(
            full_confidence=0.0,
            spent_confidence=1.0,
            full_distance=1.0,
            spent_distance=0.0,
            spent_structure_verified=False,
            normalized_meter=0.0,
        )


def test_observation_accepts_backward_compatible_meter_level_metadata() -> None:
    legacy = _valid_observation()
    supported = _valid_observation(
        meter_level_supported=True,
        meter_level=2,
        meter_level_confidence=0.91,
    )
    unreadable = _valid_observation(
        meter_level_supported=True,
        meter_level=None,
        meter_level_confidence=0.0,
    )

    assert (
        legacy.meter_level_supported,
        legacy.meter_level,
        legacy.meter_level_confidence,
    ) == (False, None, 0.0)
    assert (
        supported.meter_level_supported,
        supported.meter_level,
        supported.meter_level_confidence,
    ) == (True, 2, 0.91)
    assert unreadable.meter_level is None


@pytest.mark.parametrize(
    ("supported", "level", "confidence"),
    [
        (1, 2, 0.9),
        (False, 2, 0.9),
        (False, None, 0.1),
        (True, True, 0.9),
        (True, -1, 0.9),
        (True, 4, 0.9),
        (True, 2.0, 0.9),
        (True, None, 0.1),
        (True, 2, 0.0),
        (True, 2, float("nan")),
        (True, 2, 1.01),
    ],
)
def test_observation_rejects_inconsistent_meter_level_metadata(
    supported: object,
    level: object,
    confidence: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _valid_observation(
            meter_level_supported=supported,
            meter_level=level,
            meter_level_confidence=confidence,
        )


def test_colour_observation_carries_one_trustworthy_meter_level_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _colour_references(Side.P1)
    frame = np.full((*METER_OUTPUT_SHAPE, 3), 90, dtype=np.uint8)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(gauge_features_module, "extract_meter_fill", lambda *_: 1.0)

    def read_spy(
        frame_arg: np.ndarray,
        *,
        side: Side,
        meter_roi: NormalizedRoi,
        color_model: MeterColorModel,
    ) -> MeterLevelReading:
        calls.append((frame_arg, side, meter_roi, color_model))
        return MeterLevelReading(side, 2, 0.91, True)

    monkeypatch.setattr(gauge_features_module, "read_meter_level", read_spy)

    observation = observe_gauge(
        frame,
        side=Side.P1,
        frame_id=30,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=3.0,
    )

    assert calls == [(frame, Side.P1, FULL_FRAME_ROI, references.color_model)]
    assert (
        observation.meter_level_supported,
        observation.meter_level,
        observation.meter_level_confidence,
    ) == (True, 2, 0.91)


def test_colour_observation_keeps_support_when_level_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _colour_references(Side.P1)
    monkeypatch.setattr(gauge_features_module, "extract_meter_fill", lambda *_: 1.0)
    monkeypatch.setattr(
        gauge_features_module,
        "read_meter_level",
        lambda *_args, **_kwargs: MeterLevelReading(Side.P1, None, 0.0, False),
    )

    observation = observe_gauge(
        np.full((*METER_OUTPUT_SHAPE, 3), 90, dtype=np.uint8),
        side=Side.P1,
        frame_id=31,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=3.1,
    )

    assert (
        observation.meter_level_supported,
        observation.meter_level,
        observation.meter_level_confidence,
    ) == (True, None, 0.0)


def test_colour_fill_does_not_discard_a_readable_level_before_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _colour_references(Side.P2)
    monkeypatch.setattr(gauge_features_module, "extract_meter_fill", lambda *_: 0.48)
    monkeypatch.setattr(
        gauge_features_module,
        "read_meter_level",
        lambda *_args, **_kwargs: MeterLevelReading(Side.P2, 0, 0.9, True),
    )
    monkeypatch.setattr(
        gauge_features_module,
        "preprocess_meter",
        lambda _crop: PreprocessedMeter(
            np.full(METER_OUTPUT_SHAPE, 127, dtype=np.uint8),
            120.0,
            8.0,
            False,
        ),
    )

    observation = observe_gauge(
        np.full((*METER_OUTPUT_SHAPE, 3), 90, dtype=np.uint8),
        side=Side.P2,
        frame_id=32,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=3.2,
    )

    assert observation.spent_structure_verified is False
    assert (
        observation.meter_level_supported,
        observation.meter_level,
        observation.meter_level_confidence,
    ) == (True, 0, 0.9)


def test_level_glyph_alone_cannot_make_hidden_colour_meter_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = _colour_references(Side.P1)
    monkeypatch.setattr(gauge_features_module, "extract_meter_fill", lambda *_: None)
    monkeypatch.setattr(
        gauge_features_module,
        "read_meter_level",
        lambda *_args, **_kwargs: MeterLevelReading(Side.P1, 3, 0.95, True),
    )

    observation = observe_gauge(
        np.zeros((*METER_OUTPUT_SHAPE, 3), dtype=np.uint8),
        side=Side.P1,
        frame_id=32,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=3.2,
    )

    assert observation.hud_visible is False
    assert (
        observation.meter_level_supported,
        observation.meter_level,
        observation.meter_level_confidence,
    ) == (True, None, 0.0)


def test_grayscale_observation_does_not_call_meter_level_reader(
    monkeypatch: pytest.MonkeyPatch,
    feature_fixture: FeatureFixture,
) -> None:
    monkeypatch.setattr(
        gauge_features_module,
        "read_meter_level",
        lambda *_args, **_kwargs: pytest.fail("legacy grayscale must not read a level"),
    )

    observation = observe_gauge(
        feature_fixture.frame_for(Side.P1, feature_fixture.full_crop),
        side=Side.P1,
        frame_id=33,
        roi=feature_fixture.p1_roi,
        references=feature_fixture.p1_refs,
        sampled_at=3.3,
    )

    assert observation.meter_level_supported is False


@pytest.mark.parametrize(
    ("argument", "invalid"),
    [
        ("side", "P1"),
        ("frame_id", True),
        ("frame_id", -1),
        ("sampled_at", False),
        ("sampled_at", -1),
        ("roi", (0.0, 0.0, 1.0, 1.0)),
        ("references", object()),
    ],
)
def test_observe_rejects_invalid_routing_values(
    argument: str, invalid: object
):
    values: dict[str, object] = {
        "side": Side.P1,
        "frame_id": 0,
        "roi": FULL_FRAME_ROI,
        "references": _references(Side.P1),
        "sampled_at": 0,
    }
    values[argument] = invalid
    with pytest.raises((TypeError, ValueError)):
        observe_gauge(np.zeros((4, 4, 3), dtype=np.uint8), **values)


def test_crop_then_preprocess_are_called_once_in_order_with_identity_pass_through(
    monkeypatch: pytest.MonkeyPatch,
):
    frame = np.zeros((5, 7, 3), dtype=np.uint8)
    crop_token = np.ones((3, 4, 3), dtype=np.uint8)
    observed = np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8)
    references = MeterReferences(
        Side.P1,
        np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8),
        np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8),
    )
    calls: list[tuple[str, object, object | None]] = []

    def crop_spy(frame_arg: np.ndarray, roi_arg: NormalizedRoi) -> np.ndarray:
        calls.append(("crop", frame_arg, roi_arg))
        assert frame_arg is frame and roi_arg is FULL_FRAME_ROI
        return crop_token

    def preprocess_spy(crop_arg: np.ndarray) -> PreprocessedMeter:
        calls.append(("preprocess", crop_arg, None))
        assert crop_arg is crop_token
        return PreprocessedMeter(observed, 120.0, 8.0, True)

    monkeypatch.setattr(gauge_features_module, "crop_normalized", crop_spy)
    monkeypatch.setattr(gauge_features_module, "preprocess_meter", preprocess_spy)
    item = observe_gauge(
        frame,
        side=Side.P1,
        frame_id=1,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=1,
    )
    assert [call[0] for call in calls] == ["crop", "preprocess"]
    assert calls[0][1] is frame and calls[1][1] is crop_token
    assert item.hud_visible and item.full_confidence == 1.0


def test_literal_arrays_lock_distance_dtype_normalization_and_cross_confidence(
    monkeypatch: pytest.MonkeyPatch,
):
    full_reference = np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8)
    spent_reference = np.full(METER_OUTPUT_SHAPE, 200, dtype=np.uint8)
    observed = np.tile(
        np.array([0, 64, 128, 255], dtype=np.uint8),
        (METER_OUTPUT_SHAPE[0], METER_OUTPUT_SHAPE[1] // 4),
    )
    references = MeterReferences(Side.P1, full_reference, spent_reference)
    real_mean = np.mean
    mean_calls: list[tuple[np.dtype, object]] = []

    def checked_mean(values: np.ndarray, *, dtype: object = None) -> object:
        mean_calls.append((values.dtype, dtype))
        assert values.dtype == np.dtype(np.float32)
        assert dtype is np.float64
        return real_mean(values, dtype=dtype)

    monkeypatch.setattr(
        gauge_features_module,
        "preprocess_meter",
        lambda crop: PreprocessedMeter(observed, 120.0, 8.0, True),
    )
    monkeypatch.setattr(gauge_features_module.np, "mean", checked_mean)
    item = observe_gauge(
        np.zeros((2, 3, 3), dtype=np.uint8),
        side=Side.P1,
        frame_id=2,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=2,
    )
    expected_full_distance = 111.75 / 255.0
    expected_spent_distance = 115.75 / 255.0
    total_distance = expected_full_distance + expected_spent_distance
    assert mean_calls == [
        (np.dtype(np.float32), np.float64),
        (np.dtype(np.float32), np.float64),
    ]
    assert type(item.full_distance) is np.float64
    assert type(item.spent_distance) is np.float64
    assert item.full_distance == expected_full_distance
    assert item.spent_distance == expected_spent_distance
    assert item.full_confidence == expected_spent_distance / total_distance
    assert item.spent_confidence == expected_full_distance / total_distance


def test_bgr_unequal_noncanonical_crop_is_resized_and_matches_reference():
    yy, xx = np.indices((19, 37))
    crop = np.stack(
        (
            (xx * 7 + yy * 3) % 256,
            (xx * 2 + yy * 11 + 31) % 256,
            (xx * 13 + yy * 5 + 79) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    spent_crop = np.flip(crop, axis=1).copy()
    full = preprocess_meter(crop)
    spent = preprocess_meter(spent_crop)
    references = MeterReferences(
        Side.P1, full.normalized_gray, spent.normalized_gray
    )
    assert crop.shape[:2] != METER_OUTPUT_SHAPE
    assert not np.array_equal(crop[:, :, 0], crop[:, :, 1])
    item = observe_gauge(
        crop,
        side=Side.P1,
        frame_id=3,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=3,
    )
    assert item.hud_visible
    assert (item.full_confidence, item.spent_confidence, item.full_distance) == (
        1.0,
        0.0,
        0.0,
    )


def test_deliberately_partial_visible_crop_remains_visible_without_guess_suppression(
    feature_fixture: FeatureFixture,
):
    item = observe_gauge(
        feature_fixture.frame_for(Side.P1, feature_fixture.partial_crop()),
        side=Side.P1,
        frame_id=4,
        roi=feature_fixture.p1_roi,
        references=feature_fixture.p1_refs,
        sampled_at=4,
    )
    assert item.hud_visible
    assert item.full_confidence < 0.78
    assert item.spent_confidence < 0.72


@pytest.mark.parametrize(
    "crop",
    [
        np.zeros((40, 80, 3), dtype=np.uint8),
        np.full((40, 80, 3), 255, dtype=np.uint8),
        np.full((40, 80, 3), 127, dtype=np.uint8),
        _bgr(
            (
                120
                + (np.indices((40, 80))[0] + np.indices((40, 80))[1]) % 2
            ).astype(np.uint8)
        ),
    ],
    ids=["black", "white", "flat-mid-gray", "low-contrast"],
)
def test_hidden_and_low_information_inputs_return_exact_sentinel(crop: np.ndarray):
    item = observe_gauge(
        crop,
        side=Side.P1,
        frame_id=5,
        roi=FULL_FRAME_ROI,
        references=_references(Side.P1),
        sampled_at=5,
    )
    assert item.hud_visible is False
    assert (
        item.full_confidence,
        item.spent_confidence,
        item.full_distance,
        item.spent_distance,
    ) == (0.0, 0.0, 1.0, 1.0)


def test_visible_relative_result_is_not_rejected_by_an_absolute_ood_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    references = MeterReferences(
        Side.P1,
        np.zeros(METER_OUTPUT_SHAPE, dtype=np.uint8),
        np.full(METER_OUTPUT_SHAPE, 255, dtype=np.uint8),
    )
    observed = np.full(METER_OUTPUT_SHAPE, 64, dtype=np.uint8)
    monkeypatch.setattr(
        gauge_features_module,
        "preprocess_meter",
        lambda crop: PreprocessedMeter(observed, 120.0, 8.0, True),
    )
    item = observe_gauge(
        np.zeros((2, 2, 3), dtype=np.uint8),
        side=Side.P1,
        frame_id=6,
        roi=FULL_FRAME_ROI,
        references=references,
        sampled_at=6,
    )
    assert item.hud_visible
    assert item.full_distance == 64 / 255.0
    assert item.spent_distance == 191 / 255.0
    assert item.full_confidence == 191 / 255.0


def test_success_preserves_frame_values_layout_flags_and_ownership(
    feature_fixture: FeatureFixture,
):
    contiguous = feature_fixture.frame_for(Side.P1, feature_fixture.full_crop)
    frame = _noncontiguous_clone(contiguous)
    before = _array_state(frame)
    references_before = _reference_state(feature_fixture.p1_refs)
    item = observe_gauge(
        frame,
        side=Side.P1,
        frame_id=7,
        roi=feature_fixture.p1_roi,
        references=feature_fixture.p1_refs,
        sampled_at=7,
    )
    assert item.hud_visible
    _assert_array_state(frame, before)
    _assert_reference_state(feature_fixture.p1_refs, references_before)


def test_hidden_preserves_frame_values_layout_flags_and_ownership(
    feature_fixture: FeatureFixture,
):
    frame = _noncontiguous_clone(
        np.zeros_like(feature_fixture.frame_for(Side.P1, feature_fixture.full_crop))
    )
    before = _array_state(frame)
    references_before = _reference_state(feature_fixture.p1_refs)
    item = observe_gauge(
        frame,
        side=Side.P1,
        frame_id=8,
        roi=feature_fixture.p1_roi,
        references=feature_fixture.p1_refs,
        sampled_at=8,
    )
    assert item.hud_visible is False
    _assert_array_state(frame, before)
    _assert_reference_state(feature_fixture.p1_refs, references_before)


def test_invalid_input_preserves_frame_and_reference_arrays(
    feature_fixture: FeatureFixture,
):
    frame = _noncontiguous_clone(
        feature_fixture.frame_for(Side.P1, feature_fixture.full_crop)
    )
    frame_before = _array_state(frame)
    references_before = _reference_state(feature_fixture.p1_refs)
    with pytest.raises(TypeError):
        observe_gauge(
            frame,
            side=Side.P1,
            frame_id=9,
            roi=(0.0, 0.0, 1.0, 1.0),
            references=feature_fixture.p1_refs,
            sampled_at=9,
        )
    _assert_array_state(frame, frame_before)
    _assert_reference_state(feature_fixture.p1_refs, references_before)


def test_dependency_error_preserves_frame_and_reference_arrays(
    feature_fixture: FeatureFixture, monkeypatch: pytest.MonkeyPatch
):
    frame = _noncontiguous_clone(
        feature_fixture.frame_for(Side.P1, feature_fixture.full_crop)
    )
    frame_before = _array_state(frame)
    references_before = _reference_state(feature_fixture.p1_refs)

    def fail_preprocess(crop: np.ndarray) -> PreprocessedMeter:
        raise RuntimeError("injected preprocessing failure")

    monkeypatch.setattr(gauge_features_module, "preprocess_meter", fail_preprocess)
    with pytest.raises(RuntimeError, match="injected"):
        observe_gauge(
            frame,
            side=Side.P1,
            frame_id=10,
            roi=feature_fixture.p1_roi,
            references=feature_fixture.p1_refs,
            sampled_at=10,
        )
    _assert_array_state(frame, frame_before)
    _assert_reference_state(feature_fixture.p1_refs, references_before)
