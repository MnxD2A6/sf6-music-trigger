"""Deterministic gauge observations from calibrated meter references."""

from dataclasses import dataclass
import math
from numbers import Rational, Real

import numpy as np

from .domain import Side
from .hud_profile import NormalizedRoi, crop_normalized
from .meter_color_features import extract_meter_fill
from .meter_level_features import read_meter_level
from .meter_references import MeterReferences, preprocess_meter


_MAX_COLOR_SPENT_STRUCTURE_DISTANCE = 0.25


def _require_side(side: object) -> None:
    if not isinstance(side, Side):
        raise TypeError("side must be Side")


def _require_frame_id(frame_id: object) -> None:
    if type(frame_id) is not int:
        raise TypeError("frame_id must be an integer")
    if frame_id < 0:
        raise ValueError("frame_id must be nonnegative")


def _require_sampled_at(sampled_at: object) -> None:
    if isinstance(sampled_at, bool) or not isinstance(sampled_at, Real):
        raise TypeError("sampled_at must be a real number")
    if isinstance(sampled_at, Rational):
        if sampled_at < 0:
            raise ValueError("sampled_at must be finite and nonnegative")
        return
    if not math.isfinite(sampled_at) or sampled_at < 0:
        raise ValueError("sampled_at must be finite and nonnegative")


def _require_metric(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")


def _score_distances(
    full_distance: Real, spent_distance: Real
) -> tuple[Real, Real]:
    """Return full/spent confidence from the opposite reference distances."""

    _require_metric("full_distance", full_distance)
    _require_metric("spent_distance", spent_distance)
    total_distance = full_distance + spent_distance
    if total_distance == 0:
        return (0.5, 0.5)
    return (
        spent_distance / total_distance,
        full_distance / total_distance,
    )


@dataclass(frozen=True)
class GaugeObservation:
    side: Side
    frame_id: int
    sampled_at: Real
    hud_visible: bool
    full_confidence: Real
    spent_confidence: Real
    full_distance: Real
    spent_distance: Real
    spent_structure_verified: bool = True
    normalized_meter: float | None = None
    meter_level_supported: bool = False
    meter_level: int | None = None
    meter_level_confidence: float = 0.0

    def __post_init__(self) -> None:
        _require_side(self.side)
        _require_frame_id(self.frame_id)
        _require_sampled_at(self.sampled_at)
        if type(self.hud_visible) is not bool:
            raise TypeError("hud_visible must be a boolean")
        if type(self.spent_structure_verified) is not bool:
            raise TypeError("spent_structure_verified must be a boolean")
        if type(self.meter_level_supported) is not bool:
            raise TypeError("meter_level_supported must be a boolean")
        for name, value in (
            ("full_confidence", self.full_confidence),
            ("spent_confidence", self.spent_confidence),
            ("full_distance", self.full_distance),
            ("spent_distance", self.spent_distance),
        ):
            _require_metric(name, value)
        if self.normalized_meter is not None:
            _require_metric("normalized_meter", self.normalized_meter)
            object.__setattr__(self, "normalized_meter", float(self.normalized_meter))
        if self.meter_level is not None:
            if type(self.meter_level) is not int:
                raise TypeError("meter_level must be an integer or None")
            if not 0 <= self.meter_level <= 3:
                raise ValueError("meter_level must be between zero and three")
        _require_metric("meter_level_confidence", self.meter_level_confidence)
        object.__setattr__(
            self, "meter_level_confidence", float(self.meter_level_confidence)
        )
        if not self.meter_level_supported:
            if self.meter_level is not None or self.meter_level_confidence != 0:
                raise ValueError("unsupported meter level metadata must be empty")
        elif self.meter_level is None:
            if self.meter_level_confidence != 0:
                raise ValueError("unreadable meter level must have zero confidence")
        elif self.meter_level_confidence <= 0:
            raise ValueError("readable meter level must have positive confidence")

        if not self.hud_visible:
            if self.normalized_meter is not None:
                raise ValueError("hidden observations cannot carry normalized meter")
            if not self.spent_structure_verified:
                raise ValueError("hidden observations cannot carry spent evidence")
            if (
                self.full_confidence,
                self.spent_confidence,
                self.full_distance,
                self.spent_distance,
            ) != (0.0, 0.0, 1.0, 1.0):
                raise ValueError("hidden observations must use the fail-closed sentinel")
            return

        expected_full, expected_spent = _score_distances(
            self.full_distance, self.spent_distance
        )
        if not (
            self.full_confidence == expected_full
            or math.isclose(
                self.full_confidence,
                expected_full,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("full_confidence is inconsistent with distances")
        if not (
            self.spent_confidence == expected_spent
            or math.isclose(
                self.spent_confidence,
                expected_spent,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("spent_confidence is inconsistent with distances")
        if (
            not self.spent_structure_verified
            and self.spent_confidence <= self.full_confidence
        ):
            raise ValueError("unverified structure must belong to spent evidence")
        if not self.spent_structure_verified and self.normalized_meter is not None:
            raise ValueError("unverified observations cannot carry normalized meter")


def observe_gauge(
    frame_bgr: np.ndarray,
    *,
    side: Side,
    frame_id: int,
    roi: NormalizedRoi,
    references: MeterReferences,
    sampled_at: Real,
) -> GaugeObservation:
    """Extract one immutable observation through the Task 3 preprocessing path."""

    _require_side(side)
    _require_frame_id(frame_id)
    _require_sampled_at(sampled_at)
    if not isinstance(roi, NormalizedRoi):
        raise TypeError("roi must be NormalizedRoi")
    if not isinstance(references, MeterReferences):
        raise TypeError("references must be MeterReferences")
    if references.side is not side:
        raise ValueError("references side must match observation side")

    crop = crop_normalized(frame_bgr, roi)
    if references.color_model is not None:
        try:
            level_reading = read_meter_level(
                frame_bgr,
                side=side,
                meter_roi=roi,
                color_model=references.color_model,
            )
            meter_level = level_reading.level
            meter_level_confidence = level_reading.confidence
        except ValueError:
            # Legacy/synthetic profiles may leave no outward glyph band.
            meter_level = None
            meter_level_confidence = 0.0
        fill = extract_meter_fill(crop, references.color_model)
        if fill is None:
            return GaugeObservation(
                side=side,
                frame_id=frame_id,
                sampled_at=sampled_at,
                hud_visible=False,
                full_confidence=0.0,
                spent_confidence=0.0,
                full_distance=1.0,
                spent_distance=1.0,
                meter_level_supported=True,
            )
        full_distance = abs(fill - references.color_model.full_fill)
        spent_distance = abs(fill - references.color_model.spent_fill)
        normalized_meter = min(
            1.0,
            max(
                0.0,
                float(
                    (fill - references.color_model.spent_fill)
                    / (
                        references.color_model.full_fill
                        - references.color_model.spent_fill
                    )
                ),
            ),
        )
        spent_structure_verified = True
        if spent_distance < full_distance:
            spent_structure = preprocess_meter(crop)
            spent_structure_distance = np.mean(
                np.abs(
                    spent_structure.normalized_gray.astype(np.float32)
                    - references.spent_gray.astype(np.float32)
                ),
                dtype=np.float64,
            ) / 255.0
            if (
                not spent_structure.hud_visible
                or spent_structure_distance > _MAX_COLOR_SPENT_STRUCTURE_DISTANCE
            ):
                spent_structure_verified = False
        full_confidence, spent_confidence = _score_distances(
            full_distance, spent_distance
        )
        return GaugeObservation(
            side=side,
            frame_id=frame_id,
            sampled_at=sampled_at,
            hud_visible=True,
            full_confidence=full_confidence,
            spent_confidence=spent_confidence,
            full_distance=full_distance,
            spent_distance=spent_distance,
            spent_structure_verified=spent_structure_verified,
            normalized_meter=(
                normalized_meter if spent_structure_verified else None
            ),
            meter_level_supported=True,
            meter_level=meter_level,
            meter_level_confidence=meter_level_confidence,
        )
    preprocessed = preprocess_meter(crop)
    if not preprocessed.hud_visible:
        return GaugeObservation(
            side=side,
            frame_id=frame_id,
            sampled_at=sampled_at,
            hud_visible=False,
            full_confidence=0.0,
            spent_confidence=0.0,
            full_distance=1.0,
            spent_distance=1.0,
        )

    observed = preprocessed.normalized_gray
    full_distance = np.mean(
        np.abs(
            observed.astype(np.float32)
            - references.full_gray.astype(np.float32)
        ),
        dtype=np.float64,
    ) / 255.0
    spent_distance = np.mean(
        np.abs(
            observed.astype(np.float32)
            - references.spent_gray.astype(np.float32)
        ),
        dtype=np.float64,
    ) / 255.0
    full_confidence, spent_confidence = _score_distances(
        full_distance, spent_distance
    )
    return GaugeObservation(
        side=side,
        frame_id=frame_id,
        sampled_at=sampled_at,
        hud_visible=True,
        full_confidence=full_confidence,
        spent_confidence=spent_confidence,
        full_distance=full_distance,
        spent_distance=spent_distance,
    )
