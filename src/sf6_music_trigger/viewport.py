"""Fail-closed normalization of centered SF6 game content viewports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

import numpy as np

from .domain import CapturedFrame, ClientGeometry


@dataclass(frozen=True)
class ViewportRect:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name, value in (
            ("left", self.left),
            ("top", self.top),
            ("width", self.width),
            ("height", self.height),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.left < 0 or self.top < 0:
            raise ValueError("viewport offsets must be nonnegative")
        if self.width < 1 or self.height < 1:
            raise ValueError("viewport dimensions must be positive")


def _positive_dimension(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_aspect(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("target_aspect must be a real number")
    aspect = float(value)
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError("target_aspect must be finite and positive")
    return aspect


def centered_viewport(
    width: int,
    height: int,
    target_aspect: Real,
) -> ViewportRect:
    """Return the largest centered integer rectangle at ``target_aspect``."""

    client_width = _positive_dimension("width", width)
    client_height = _positive_dimension("height", height)
    aspect = _positive_aspect(target_aspect)
    actual = client_width / client_height
    if math.isclose(actual, aspect, rel_tol=0.0, abs_tol=1e-12):
        return ViewportRect(0, 0, client_width, client_height)
    if actual > aspect:
        viewport_width = max(1, min(client_width, round(client_height * aspect)))
        return ViewportRect(
            (client_width - viewport_width) // 2,
            0,
            viewport_width,
            client_height,
        )
    viewport_height = max(1, min(client_height, round(client_width / aspect)))
    return ViewportRect(
        0,
        (client_height - viewport_height) // 2,
        client_width,
        viewport_height,
    )


def _is_dark(region: np.ndarray) -> bool:
    if region.size == 0:
        return False
    brightest_channel = np.max(region, axis=2)
    # Window decorations and compressed captures can make a black bar dark gray.
    return float(np.mean(brightest_channel <= 48)) >= 0.98


def _is_light_taskbar(region: np.ndarray) -> bool:
    if region.size == 0:
        return False
    darkest_channel = np.min(region, axis=2)
    brightest_channel = np.max(region, axis=2)
    neutral_light = (darkest_channel >= 160) & (
        brightest_channel.astype(np.int16) - darkest_channel.astype(np.int16) <= 30
    )
    return float(np.mean(neutral_light)) >= 0.75


def borders_confirm_viewport(image: np.ndarray, rect: ViewportRect) -> bool:
    """Confirm dark bars, or a dark top bar paired with a light taskbar below."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 array")
    if not isinstance(rect, ViewportRect):
        raise TypeError("rect must be ViewportRect")
    height, width = image.shape[:2]
    right = rect.left + rect.width
    bottom = rect.top + rect.height
    if right > width or bottom > height:
        raise ValueError("viewport must be contained in image")
    if rect.left == 0 and rect.top == 0 and right == width and bottom == height:
        return True
    if rect.left > 0:
        return _is_dark(image[:, : rect.left]) and _is_dark(image[:, right:])
    if rect.top > 0:
        top_bar = image[: rect.top, :]
        bottom_bar = image[bottom:, :]
        return _is_dark(top_bar) and (
            _is_dark(bottom_bar) or _is_light_taskbar(bottom_bar)
        )
    return False


class ViewportStatus(str, Enum):
    NATIVE = "native"
    CONFIRMING = "confirming"
    NORMALIZED = "normalized"
    UNSUPPORTED = "unsupported"
    TOO_SMALL = "too_small"


@dataclass(frozen=True)
class ViewportResult:
    frame: CapturedFrame | None
    status: ViewportStatus

    def __post_init__(self) -> None:
        if self.frame is not None and not isinstance(self.frame, CapturedFrame):
            raise TypeError("frame must be CapturedFrame or None")
        if not isinstance(self.status, ViewportStatus):
            raise TypeError("status must be ViewportStatus")
        if self.frame is None and self.status in {
            ViewportStatus.NATIVE,
            ViewportStatus.NORMALIZED,
        }:
            raise ValueError("successful viewport status requires a frame")
        if self.frame is not None and self.status not in {
            ViewportStatus.NATIVE,
            ViewportStatus.NORMALIZED,
        }:
            raise ValueError("unsuccessful viewport status cannot carry a frame")


class CenteredViewportResolver:
    """Lock a centered, dark-bordered viewport for one stable client geometry."""

    def __init__(
        self,
        target_aspect: Real,
        *,
        confirmation_frames: int = 3,
        minimum_content_height: int = 720,
        native_aspect_tolerance: Real = 0.01,
    ) -> None:
        self.target_aspect = _positive_aspect(target_aspect)
        self.confirmation_frames = _positive_dimension(
            "confirmation_frames", confirmation_frames
        )
        self.minimum_content_height = _positive_dimension(
            "minimum_content_height", minimum_content_height
        )
        if isinstance(native_aspect_tolerance, bool) or not isinstance(
            native_aspect_tolerance, Real
        ):
            raise TypeError("native_aspect_tolerance must be a real number")
        tolerance = float(native_aspect_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("native_aspect_tolerance must be finite and nonnegative")
        self.native_aspect_tolerance = tolerance
        self.reset()

    def reset(self) -> None:
        self._geometry: ClientGeometry | None = None
        self._candidate_rect: ViewportRect | None = None
        self._locked_rect: ViewportRect | None = None
        self._confirmations = 0

    def _change_geometry(self, geometry: ClientGeometry) -> None:
        if geometry != self._geometry:
            self._geometry = geometry
            self._candidate_rect = None
            self._locked_rect = None
            self._confirmations = 0

    @staticmethod
    def _valid_image(frame: CapturedFrame) -> bool:
        image = frame.image
        return (
            image.ndim == 3
            and image.shape == (
                frame.geometry.height,
                frame.geometry.width,
                3,
            )
            and image.dtype == np.uint8
        )

    def _crop(self, frame: CapturedFrame, rect: ViewportRect) -> CapturedFrame:
        image = frame.image[
            rect.top : rect.top + rect.height,
            rect.left : rect.left + rect.width,
        ].copy(order="C")
        geometry = ClientGeometry(
            frame.geometry.left + rect.left,
            frame.geometry.top + rect.top,
            rect.width,
            rect.height,
        )
        return CapturedFrame(
            frame.frame_id,
            image,
            frame.monotonic_time,
            geometry,
        )

    def resolve(self, frame: CapturedFrame) -> ViewportResult:
        if not isinstance(frame, CapturedFrame):
            raise TypeError("frame must be CapturedFrame")
        self._change_geometry(frame.geometry)
        if not self._valid_image(frame):
            self._confirmations = 0
            return ViewportResult(None, ViewportStatus.UNSUPPORTED)

        actual_aspect = frame.geometry.width / frame.geometry.height
        relative_error = abs(actual_aspect - self.target_aspect) / self.target_aspect
        if relative_error <= self.native_aspect_tolerance:
            self._candidate_rect = None
            self._locked_rect = None
            self._confirmations = 0
            if frame.geometry.height < self.minimum_content_height:
                return ViewportResult(None, ViewportStatus.TOO_SMALL)
            return ViewportResult(frame, ViewportStatus.NATIVE)

        rect = centered_viewport(
            frame.geometry.width,
            frame.geometry.height,
            self.target_aspect,
        )
        if rect.height < self.minimum_content_height:
            self._confirmations = 0
            return ViewportResult(None, ViewportStatus.TOO_SMALL)
        if self._locked_rect is not None:
            return ViewportResult(
                self._crop(frame, self._locked_rect),
                ViewportStatus.NORMALIZED,
            )
        if not borders_confirm_viewport(frame.image, rect):
            self._candidate_rect = None
            self._confirmations = 0
            return ViewportResult(None, ViewportStatus.UNSUPPORTED)
        if rect != self._candidate_rect:
            self._candidate_rect = rect
            self._confirmations = 0
        self._confirmations += 1
        if self._confirmations < self.confirmation_frames:
            return ViewportResult(None, ViewportStatus.CONFIRMING)
        self._locked_rect = rect
        return ViewportResult(self._crop(frame, rect), ViewportStatus.NORMALIZED)
