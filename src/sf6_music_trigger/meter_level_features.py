"""Pure, calibration-derived SF6 super-meter level features."""

from dataclasses import dataclass
import math
from numbers import Real

import cv2
import numpy as np

from .domain import Side
from .hud_profile import NormalizedRoi, crop_normalized
from .meter_color_features import MeterColorModel


_LEVEL_FEATURE_HEIGHT = 64
_MIN_LEVEL_FEATURE_WIDTH = 32
_MAX_LEVEL_FEATURE_WIDTH = 320
_MIN_SIDE_PIXELS = 24
_MIN_CORE_PIXELS = 12


def _require_side(value: object) -> Side:
    if not isinstance(value, Side):
        raise TypeError("side must be Side")
    return value


def _require_level(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("level must be an integer or None")
    if not 0 <= value <= 3:
        raise ValueError("level must be between zero and three")
    return value


def _require_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("confidence must be a real number")
    if not math.isfinite(value):
        raise ValueError("confidence must be finite")
    if not 0 <= value <= 1:
        raise ValueError("confidence must be between zero and one")
    return float(value)


@dataclass(frozen=True)
class MeterLevelReading:
    """A single side's visible fixed-alphabet meter-level reading."""

    side: Side
    level: int | None
    confidence: float
    visible: bool

    def __post_init__(self) -> None:
        side = _require_side(self.side)
        level = _require_level(self.level)
        confidence = _require_confidence(self.confidence)
        if type(self.visible) is not bool:
            raise TypeError("visible must be a boolean")
        if not self.visible and level is not None:
            raise ValueError("invisible reading cannot carry a level")
        if not self.visible and confidence != 0:
            raise ValueError("invisible reading must have zero confidence")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "confidence", confidence)


def derive_meter_level_roi(meter_roi: NormalizedRoi, side: Side) -> NormalizedRoi:
    """Derive a level-glyph band outside and slightly overlapping a meter ROI."""

    if not isinstance(meter_roi, NormalizedRoi):
        raise TypeError("meter_roi must be NormalizedRoi")
    checked_side = _require_side(side)
    meter_width = meter_roi.x1 - meter_roi.x0
    outward = meter_width / 4.0
    inward = meter_width / 5.0
    if checked_side is Side.P1:
        x0 = max(0.0, round(meter_roi.x0 - outward, 12))
        x1 = min(1.0, round(meter_roi.x0 + inward, 12))
    else:
        x0 = max(0.0, round(meter_roi.x1 - inward, 12))
        x1 = min(1.0, round(meter_roi.x1 + outward, 12))
    if x0 >= x1:
        raise ValueError("no room for meter level ROI outside meter")
    return NormalizedRoi(x0, meter_roi.y0, x1, meter_roi.y1)


def _require_bgr_frame(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("frame_bgr must be a NumPy array")
    if value.dtype != np.dtype(np.uint8):
        raise TypeError("frame_bgr must have uint8 dtype")
    if value.ndim != 3 or value.shape[0] == 0 or value.shape[1] == 0 or value.shape[2] != 3:
        raise ValueError("frame_bgr must be a nonempty H x W x 3 BGR array")
    return value


def _hue_distance(hues: np.ndarray, expected: int) -> np.ndarray:
    direct = np.abs(hues.astype(np.int16, copy=False) - expected)
    return np.minimum(direct, 180 - direct)


def _unknown(side: Side, *, visible: bool = False) -> MeterLevelReading:
    return MeterLevelReading(side, None, 0.0, visible)


def _component_masks(
    glyph_mask: np.ndarray, core_mask: np.ndarray
) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        glyph_mask.astype(np.uint8), connectivity=8
    )
    components: list[np.ndarray] = []
    for label in range(1, count):
        x, y, width, height, area = (int(item) for item in stats[label])
        component = labels == label
        core_pixels = int(np.count_nonzero(component & core_mask))
        if area < 30 or core_pixels < _MIN_CORE_PIXELS:
            continue
        if x <= 1 or y <= 1 or x + width >= glyph_mask.shape[1] - 1 or y + height >= glyph_mask.shape[0] - 1:
            continue
        components.append(component)
    return components


def _hole_count(component: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return 0
    return sum(1 for item in hierarchy[0] if int(item[3]) >= 0)


def _classify_single_component(component: np.ndarray) -> int | None:
    ys, xs = np.nonzero(component)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width = x1 - x0
    height = y1 - y0
    if height < 18 or width < 2:
        return None
    aspect = width / height
    local = component[y0:y1, x0:x1]
    fill = float(np.mean(local, dtype=np.float64))
    row_slices = np.array_split(local, 3, axis=0)
    row_maxima = tuple(
        float(np.max(np.mean(row, axis=0, dtype=np.float64)))
        for row in row_slices
    )
    lower = local[height // 2 :, :]
    half = max(1, width // 2)
    left = float(np.mean(lower[:, :half], dtype=np.float64))
    right = float(np.mean(lower[:, half:], dtype=np.float64))
    difference = left - right

    holes = _hole_count(component)
    if aspect >= 1.10:
        return 3
    if holes == 1:
        if difference > -0.10:
            return 0
        if aspect >= 0.58:
            return 3
        if min(row_maxima) >= 0.90:
            return 0
        if fill <= 0.62 and difference <= -0.10:
            return 3
        return None
    if holes > 1:
        return None
    one_like = (
        aspect <= 0.40
        or (row_maxima[0] <= 0.60 and row_maxima[1] <= 0.55 and row_maxima[2] >= 0.80)
        or (fill <= 0.52 and -0.10 < difference < 0.18 and row_maxima[2] >= 0.90)
    )
    if one_like:
        # A foot/background joined to a 2 can satisfy the old wide-1 shape
        # heuristics. A 1 must also retain its vertical stem above the foot.
        stem_coverage = float(np.max(np.mean(local[:2 * height // 3], axis=0)))
        return 1 if stem_coverage >= 0.85 else None
    if difference >= 0.18:
        return 2
    if difference <= -0.10:
        return 3
    return None


def read_meter_level(
    frame_bgr: np.ndarray,
    *,
    side: Side,
    meter_roi: NormalizedRoi,
    color_model: MeterColorModel,
) -> MeterLevelReading:
    """Read a glyph, rejecting P2 shapes introduced by tinted background."""
    reading = _read_meter_level(frame_bgr, side=side, meter_roi=meter_roi,
                                color_model=color_model)
    if side is Side.P2 and reading.level in (0, 2, 3):
        # Warm highlights can close a 2 into 0, add a bowl to 1 (3), or
        # join/clip a 0 or 1 into a false 2 that poisons the spend baseline.
        # A second, achromatic core must not identify a different digit.
        # It is a veto, never a replacement: clipping antialiasing can itself
        # change topology, so it cannot invent a new level or baseline.
        core = _read_meter_level(frame_bgr, side=side, meter_roi=meter_roi,
                                 color_model=color_model, white_saturation=20)
        if core.level is not None and core.level != reading.level:
            return _unknown(side, visible=True)
    return reading


def _read_meter_level(
    frame_bgr: np.ndarray,
    *,
    side: Side,
    meter_roi: NormalizedRoi,
    color_model: MeterColorModel,
    white_saturation: int = 65,
) -> MeterLevelReading:
    """Read SF6's fixed 0/1/2/3/CA level glyph beside one calibrated meter."""

    frame = _require_bgr_frame(frame_bgr)
    checked_side = _require_side(side)
    if not isinstance(meter_roi, NormalizedRoi):
        raise TypeError("meter_roi must be NormalizedRoi")
    if not isinstance(color_model, MeterColorModel):
        raise TypeError("color_model must be MeterColorModel")
    if color_model.side is not checked_side:
        raise ValueError("color_model side must match requested side")

    level_roi = derive_meter_level_roi(meter_roi, checked_side)
    crop = crop_normalized(frame, level_roi)
    canonical_width = min(
        _MAX_LEVEL_FEATURE_WIDTH,
        max(
            _MIN_LEVEL_FEATURE_WIDTH,
            round(_LEVEL_FEATURE_HEIGHT * crop.shape[1] / crop.shape[0]),
        ),
    )
    canonical = cv2.resize(
        crop,
        (canonical_width, _LEVEL_FEATURE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    hsv = cv2.cvtColor(canonical, cv2.COLOR_BGR2HSV)
    glyph_hue_match = (
        _hue_distance(hsv[..., 0], color_model.full_hue) <= color_model.hue_tolerance
    )
    if checked_side is Side.P2 and 90 <= color_model.full_hue <= 115:
        # SF6's standard P2 digits glow violet-blue although the calibrated
        # fractional bar is cyan. This is glyph-only; gauge fill calibration,
        # opposite-side rejection and custom non-blue profiles stay unchanged.
        glyph_hue_match |= _hue_distance(hsv[..., 0], 125) <= 10
    side_mask = (
        glyph_hue_match
        & (hsv[..., 1] >= color_model.saturation_floor)
        & (hsv[..., 2] >= color_model.value_floor)
    )
    if int(np.count_nonzero(side_mask)) < _MIN_SIDE_PIXELS:
        return _unknown(checked_side)

    white_mask = (hsv[..., 1] <= white_saturation) & (hsv[..., 2] >= 180)
    adjacent = cv2.dilate(side_mask.astype(np.uint8), np.ones((11, 11), np.uint8)) != 0
    core_mask = white_mask & adjacent
    if int(np.count_nonzero(core_mask)) < _MIN_CORE_PIXELS:
        return _unknown(checked_side)

    # A changing blue/purple glow must not cut an intact 0 into a 3, 2 or 1.
    # Recover severely clipped, bounded white components from a nearby colour
    # seed. Do not grow into edge-connected white flashes, unrelated letters,
    # or small background appendages on an otherwise complete glyph.
    glyph_mask = core_mask.copy()
    closed_white = cv2.morphologyEx(
        white_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ) != 0
    for component in _component_masks(closed_white, core_mask):
        coverage = np.count_nonzero(component & core_mask) / np.count_nonzero(component)
        ring = (cv2.dilate(component.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0) & ~component
        if coverage < 0.85 and np.count_nonzero(ring & side_mask) >= _MIN_CORE_PIXELS:
            glyph_mask |= component
    glyph_mask = cv2.morphologyEx(
        glyph_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ) != 0
    components = _component_masks(glyph_mask, core_mask)
    if not components and checked_side is Side.P2:
        # Violet effects can join the glyph to a bright screen-edge component.
        # In that case retain the original calibrated-colour extraction, which
        # can still isolate the digit. Never prefer it over a complete glyph.
        calibrated_side = (
            (_hue_distance(hsv[..., 0], color_model.full_hue) <= color_model.hue_tolerance)
            & (hsv[..., 1] >= color_model.saturation_floor)
            & (hsv[..., 2] >= color_model.value_floor)
        )
        calibrated_core = white_mask & (
            cv2.dilate(calibrated_side.astype(np.uint8), np.ones((11, 11), np.uint8)) != 0
        )
        calibrated_glyph = cv2.morphologyEx(
            calibrated_core.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        ) != 0
        components = _component_masks(calibrated_glyph, calibrated_core)
    if len(components) == 2:
        bounds = []
        for component in components:
            ys, xs = np.nonzero(component)
            bounds.append(
                (
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                    float((ys.min() + ys.max()) / 2.0),
                )
            )
        widths = [item[0] for item in bounds]
        heights = [item[1] for item in bounds]
        centers = [item[2] for item in bounds]
        if (
            min(widths) >= 8
            and min(heights) >= 28
            and abs(centers[0] - centers[1]) <= 6.0
        ):
            return MeterLevelReading(checked_side, 3, 0.92, True)
        tall = [
            component
            for component, (_, height, _) in zip(components, bounds)
            if height >= 28
        ]
        if len(tall) == 1:
            level = _classify_single_component(tall[0])
            if level is not None:
                return MeterLevelReading(checked_side, level, 0.9, True)
        return _unknown(checked_side, visible=True)
    if len(components) != 1:
        return _unknown(checked_side, visible=bool(components))

    component = components[0]
    _, xs = np.nonzero(component)
    center_x = float((xs.min() + xs.max()) / 2.0) / canonical.shape[1]
    anchored = (
        center_x <= 0.72
        if checked_side is Side.P1
        else center_x >= 0.28
    )
    if not anchored:
        return _unknown(checked_side, visible=True)
    level = _classify_single_component(component)
    if level is None:
        return _unknown(checked_side, visible=True)
    return MeterLevelReading(checked_side, level, 0.9, True)
