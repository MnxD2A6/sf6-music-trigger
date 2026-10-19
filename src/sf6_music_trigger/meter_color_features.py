"""Calibration-derived, background-independent SF6 meter features."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real

import cv2
import numpy as np

from .domain import Side


METER_FEATURE_SHAPE = (32, 64)
_SAMPLE_COUNT = 5
_SEGMENT_COUNT = 6
_MIN_ACTIVE_PIXELS = 36
_MIN_ACTIVE_DELTA = 24
_MIN_ENDPOINT_SEPARATION = 0.10
_MIN_VISIBLE_STDDEV = 6.0


def _require_bgr(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("crop_bgr must be a NumPy array")
    if value.dtype != np.dtype(np.uint8):
        raise TypeError("crop_bgr must have uint8 dtype")
    if value.ndim != 3 or value.shape[0] == 0 or value.shape[1] == 0 or value.shape[2] != 3:
        raise ValueError("crop_bgr must be a nonempty H x W x 3 BGR array")
    return value


def _require_actual_int(name: str, value: object, *, minimum: int, maximum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _require_fill(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, int):
        candidate = value
    else:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        candidate = value
    if not 0 <= candidate <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return float(candidate)


def _readonly_mask(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("active_mask must be a NumPy array")
    if value.dtype != np.dtype(np.bool_):
        raise TypeError("active_mask must have bool dtype")
    if value.shape != METER_FEATURE_SHAPE:
        raise ValueError(f"active_mask must have shape {METER_FEATURE_SHAPE}")
    copied = np.array(value, dtype=np.bool_, order="C", copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, eq=False)
class MeterColorModel:
    """Colour signature and calibrated active gauge positions for one side."""

    side: Side
    active_mask: np.ndarray
    full_hue: int
    hue_tolerance: int
    saturation_floor: int
    value_floor: int
    full_fill: float
    spent_fill: float

    def __post_init__(self) -> None:
        if not isinstance(self.side, Side):
            raise TypeError("side must be Side")
        mask = _readonly_mask(self.active_mask)
        if int(np.count_nonzero(mask)) < _MIN_ACTIVE_PIXELS:
            raise ValueError("active_mask must contain enough pixels")
        _require_actual_int("full_hue", self.full_hue, minimum=0, maximum=179)
        _require_actual_int("hue_tolerance", self.hue_tolerance, minimum=1, maximum=89)
        _require_actual_int("saturation_floor", self.saturation_floor, minimum=1, maximum=255)
        _require_actual_int("value_floor", self.value_floor, minimum=1, maximum=255)
        full = _require_fill("full_fill", self.full_fill)
        spent = _require_fill("spent_fill", self.spent_fill)
        if full - spent < _MIN_ENDPOINT_SEPARATION:
            raise ValueError("full/spent colour fills must be sufficiently separated")
        object.__setattr__(self, "active_mask", mask)
        object.__setattr__(self, "full_fill", full)
        object.__setattr__(self, "spent_fill", spent)


def _require_five_crops(name: str, value: object) -> Sequence[np.ndarray]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(value) != _SAMPLE_COUNT:
        raise ValueError(f"{name} must contain exactly five samples")
    return value


def _canonical_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    crop = _require_bgr(crop_bgr)
    return cv2.resize(
        crop,
        (METER_FEATURE_SHAPE[1], METER_FEATURE_SHAPE[0]),
        interpolation=cv2.INTER_AREA,
    )


def _hue_distance(hues: np.ndarray, expected: int) -> np.ndarray:
    direct = np.abs(hues.astype(np.int16, copy=False) - expected)
    return np.minimum(direct, 180 - direct)


def _segment_occupancy(mask: np.ndarray, active_mask: np.ndarray) -> float:
    active_columns = np.flatnonzero(np.any(active_mask, axis=0))
    if active_columns.size == 0:
        raise ValueError("active_mask must contain horizontal gauge pixels")
    first = int(active_columns.min())
    last = int(active_columns.max()) + 1
    widths = np.array_split(np.arange(first, last), _SEGMENT_COUNT)
    values: list[float] = []
    for columns in widths:
        active = active_mask[:, columns]
        active_count = int(np.count_nonzero(active))
        if active_count == 0:
            raise ValueError("active_mask must cover every meter segment")
        values.append(float(np.count_nonzero(mask[:, columns] & active) / active_count))
    return float(np.mean(values, dtype=np.float64))


def _visible(canonical_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY)
    return int(np.ptp(gray)) >= _MIN_VISIBLE_STDDEV


def _matching_mask(canonical_bgr: np.ndarray, model: MeterColorModel) -> np.ndarray:
    hsv = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2HSV)
    return (
        (_hue_distance(hsv[..., 0], model.full_hue) <= model.hue_tolerance)
        & (hsv[..., 1] >= model.saturation_floor)
        & (hsv[..., 2] >= model.value_floor)
    )


def extract_meter_fill(crop_bgr: np.ndarray, model: MeterColorModel) -> float | None:
    """Return masked coloured-gauge occupancy, or ``None`` for a hidden HUD."""

    if not isinstance(model, MeterColorModel):
        raise TypeError("model must be MeterColorModel")
    canonical = _canonical_bgr(crop_bgr)
    if not _visible(canonical):
        return None
    matching = _matching_mask(canonical, model)
    return _segment_occupancy(matching, model.active_mask)


def _median_bgr(crops: Sequence[np.ndarray]) -> np.ndarray:
    values = np.stack([_canonical_bgr(crop) for crop in crops], axis=0)
    return np.rint(np.median(values, axis=0)).astype(np.uint8)


def _active_mask(full_bgr: np.ndarray, spent_bgr: np.ndarray) -> np.ndarray:
    delta = np.mean(
        np.abs(full_bgr.astype(np.int16) - spent_bgr.astype(np.int16)),
        axis=2,
        dtype=np.float64,
    )
    changed = (delta >= _MIN_ACTIVE_DELTA).astype(np.uint8)
    # Only the interior of a coloured segment is independent of the
    # translucent scene behind its antialiased edge.
    interior = cv2.erode(changed, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return interior.astype(np.bool_)


def _full_signature(full_bgr: np.ndarray, active_mask: np.ndarray) -> tuple[int, int, int, int]:
    hsv = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2HSV)
    hue_values = hsv[..., 0][active_mask]
    saturation_values = hsv[..., 1][active_mask]
    value_values = hsv[..., 2][active_mask]
    if hue_values.size < _MIN_ACTIVE_PIXELS:
        raise ValueError("active gauge mask is too sparse")
    hue = int(np.rint(np.median(hue_values)))
    deviations = _hue_distance(hue_values, hue)
    tolerance = max(3, int(np.ceil(np.percentile(deviations, 95.0, method="linear"))) + 2)
    # Resizing a HUD crop blends the coloured segment edge with its translucent
    # surroundings. Keep a bounded margin for those edge pixels without ever
    # accepting arbitrary low-saturation scene background.
    saturation_floor = max(
        1,
        int(np.floor(np.percentile(saturation_values, 10.0, method="linear"))) - 32,
    )
    value_floor = max(
        1,
        int(np.floor(np.percentile(value_values, 10.0, method="linear"))) - 16,
    )
    return hue, min(89, tolerance), saturation_floor, value_floor


def _occupancy(crop: np.ndarray, model: MeterColorModel) -> float:
    return _segment_occupancy(_matching_mask(crop, model), model.active_mask)


def build_meter_color_model(
    side: Side,
    full_crops: Sequence[np.ndarray],
    spent_crops: Sequence[np.ndarray],
) -> MeterColorModel:
    """Learn a mask and colour signature from five full and five spent frames."""

    if not isinstance(side, Side):
        raise TypeError("side must be Side")
    full_values = _require_five_crops("full_crops", full_crops)
    spent_values = _require_five_crops("spent_crops", spent_crops)
    full_median = _median_bgr(full_values)
    spent_median = _median_bgr(spent_values)
    active_mask = _active_mask(full_median, spent_median)
    hue, tolerance, saturation_floor, value_floor = _full_signature(full_median, active_mask)
    provisional = MeterColorModel(
        side, active_mask, hue, tolerance, saturation_floor, value_floor, 1.0, 0.0
    )
    full_fills = np.asarray([_occupancy(_canonical_bgr(crop), provisional) for crop in full_values])
    spent_fills = np.asarray([_occupancy(_canonical_bgr(crop), provisional) for crop in spent_values])
    if float(np.ptp(full_fills)) > 0.08 or float(np.ptp(spent_fills)) > 0.08:
        raise ValueError("calibration colour fills are unstable")
    return MeterColorModel(
        side,
        active_mask,
        hue,
        tolerance,
        saturation_floor,
        value_floor,
        float(np.median(full_fills)),
        float(np.median(spent_fills)),
    )
