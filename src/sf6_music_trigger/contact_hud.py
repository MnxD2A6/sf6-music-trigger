"""Conservative, frame-local observations of the default SF6 contact HUD.

Input is the full BGR client image, WITHOUT a titlebar. Geometry is normalized
from the default pink-P1/blue-P2 R23 HUD (1916x1066 client). This is neither a
contact classifier nor a timer/OCR state machine. Alternate skins, cinematic
grading, missing borders, and obscured/empty meters deliberately abstain.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ContactHudReading:
    """Fractions of full meters; None means unknown, never implicit zero.

    timer_mask is a read-only (40, 48) bool array of the two numeral outlines.
    Compare only valid masks over multiple frames; exact array inequality is
    NOT timer advancement. A changed numeral alone cannot establish contact.
    """

    p1_health: float | None
    p2_health: float | None
    p1_drive: float | None
    p2_drive: float | None
    timer_mask: np.ndarray | None


_REFERENCE_WIDTH = 1916.0
_REFERENCE_HEIGHT = 1066.0


def _sample(
    image: np.ndarray, xs: np.ndarray, ys: np.ndarray, *, mirror: bool = False,
    pivot_y: float = 80.0,
    y_offset: float = 0.0,
) -> np.ndarray:
    """Rectify the HUD parallelograms, in center-to-outer-edge order."""
    xmap = xs[None, :] + .7 * (ys[:, None] - pivot_y)
    if mirror:
        xmap = _REFERENCE_WIDTH - xmap
    xmap = (xmap * image.shape[1] / _REFERENCE_WIDTH).astype(np.float32)
    ymap = np.broadcast_to(
        (ys[:, None] + y_offset) * image.shape[0] / _REFERENCE_HEIGHT, xmap.shape,
    ).astype(np.float32)
    bgr = cv2.remap(image, xmap, ymap, cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _white(hsv: np.ndarray, *, max_saturation: int = 160) -> np.ndarray:
    return (hsv[..., 1] <= max_saturation) & (hsv[..., 2] >= 150)


def _health(image: np.ndarray, *, mirror: bool, y_offset: float = 0.0) -> float | None:
    xs = np.linspace(854, 328, 264)
    body = _sample(image, xs, np.linspace(70, 90, 11), mirror=mirror, y_offset=y_offset)
    lo, hi = (95, 116) if mirror else (158, 177)
    normal = (
        (body[..., 0] >= lo) & (body[..., 0] <= hi)
        & (body[..., 1] >= 100) & (body[..., 2] >= 65)
    )
    # R23 critical health is bright yellow on BOTH sides; the damage trail
    # shares its hue but is dark (roughly V=130, versus live V=245..255).
    # Select one palette, never union yellow overlays with normal live pixels.
    critical = not (normal.mean(axis=0)[:4] >= .75).all()
    if critical:
        if (normal.mean(axis=0) >= .75).sum() > 6:
            return None
        live = (
            (body[..., 0] >= 25) & (body[..., 0] <= 35)
            & (body[..., 1] >= 80) & (body[..., 1] <= 190)
            & (body[..., 2] >= 220)
        )
    else:
        live = normal
    # Close at most a few compression pixels, not a character-sized gap.
    live = cv2.morphologyEx(
        live.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8),
    ).astype(bool)
    supported = live.mean(axis=0) >= .75
    if not supported[:4].all():
        return None
    missing = np.flatnonzero(~supported)
    end = int(missing[0]) if len(missing) else len(xs)
    if end < 8:
        return None
    if critical and end / len(xs) > .35:
        return None
    # Interior holes or multiple colored islands indicate occlusion/background.
    if live[:, :end].mean() < .94 or supported[end + 4:].sum() > 6:
        return None
    row_ends = np.array([
        int(np.flatnonzero(~row)[0]) if not row.all() else len(xs)
        for row in live
    ])
    if np.quantile(row_ends, .9) - np.quantile(row_ends, .1) > 6:
        return None
    # A genuine moving health endpoint has an outlined edge. Cutting off colored
    # pixels with a black/skin patch cannot manufacture that edge.
    # Critical fill itself fits the normal white tolerance; require a paler
    # outline so a flat bright-yellow scene patch cannot supply its own border.
    outline_saturation = 100 if critical else 160
    if end < len(xs) - 3:
        edge = _white(
            body[:, max(0, end - 2):min(len(xs), end + 5)],
            max_saturation=outline_saturation,
        )
        # Require at least nine of the eleven sampled rows. A cinematic
        # foreground edge in R23 supplies eight and mimics lost health.
        if edge.any(axis=1).mean() < .8:
            return None
        # After rectification a HUD endpoint is straight. A curved foreground
        # can offer white pixels in nine rows but at unrelated columns (R26
        # CA startup); require those rows within one narrow endpoint band.
        if not any(edge[:, column:column + 3].any(axis=1).mean() >= .8
                   for column in range(max(1, edge.shape[1] - 2))):
            return None
        if end + 12 < len(xs) and _white(body[:, end + 5:end + 12]).mean() > .4:
            return None  # A wide white flash/model is not a thin outlined edge.
    for ys in (np.arange(61, 67), np.arange(94, 99)):
        border = _white(
            _sample(image, xs, ys, mirror=mirror, y_offset=y_offset),
            max_saturation=outline_saturation,
        )
        if border[:, 3:max(4, end - 3)].any(axis=0).mean() < .72:
            return None
    return min(1.0, end / len(xs))


def read_health_pair(image: np.ndarray) -> tuple[float | None, float | None]:
    """Health only; no experimental drive/timer/contact classification.

    Permit the four reference-pixel vertical HUD offset seen in R25. Each
    hypothesis still requires the live anchor, endpoint and both outlines.
    Conflicting valid extents abstain; absent live pixels never mean zero.
    """
    if image.shape[0] < 360 or image.shape[1] < 640:
        return None, None
    values = []
    for mirror in (False, True):
        candidates = [value for offset in (0., -4.)
                      if (value := _health(image, mirror=mirror, y_offset=offset)) is not None]
        values.append(None if not candidates or max(candidates) - min(candidates) > .012
                      else max(candidates))
    return tuple(values)


def _drive(image: np.ndarray, *, mirror: bool) -> float | None:
    # Six slots, each with a parallelogram interior and a bright upper/lower rim.
    # Remove the fixed inter-slot gaps before measuring the remaining fill.
    xs = np.concatenate([
        np.linspace(868 - 55 * slot, 834 - 55 * slot, 25)
        for slot in range(6)
    ])
    body = _sample(
        image, xs, np.linspace(119, 126, 8), mirror=mirror, pivot_y=122,
    )
    active = (
        (body[..., 0] >= 20) & (body[..., 0] <= 85)
        & (body[..., 1] >= 100) & (body[..., 2] >= 95)
    )
    active = cv2.morphologyEx(
        active.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8),
    ).astype(bool)
    supported = active.mean(axis=0) >= .75
    if not supported[:3].all():
        return None
    missing = np.flatnonzero(~supported)
    end = int(missing[0]) if len(missing) else len(xs)
    if end < 4 or active[:, :end].mean() < .94:
        return None
    if supported[end + 3:].sum() > 3:
        return None
    # The unfilled part should be the dark slot tray, not a bright model/effect.
    if end + 5 < len(xs) and np.mean(body[:, end + 5:, 2] < 150) < .85:
        return None
    for ys, outside_ys in (
        (np.array([115., 116.]), np.array([110., 111.])),
        (np.array([128., 129.]), np.array([133., 134.])),
    ):
        rim = _sample(image, xs, ys, mirror=mirror, pivot_y=122)
        rim_value = rim[..., 2].max(axis=0).astype(float)
        outside_value = _sample(
            image, xs, outside_ys, mirror=mirror, pivot_y=122,
        )[..., 2].max(axis=0).astype(float)
        if np.mean(rim_value[:end] >= outside_value[:end] + 15) < .75:
            return None
    return min(1.0, end / len(xs))


def _timer(image: np.ndarray) -> np.ndarray | None:
    # Fixed tight boxes exclude the surrounding violet glow and moving scene.
    xmap, ymap = np.meshgrid(np.arange(909, 1007), np.arange(46, 127))
    bgr = cv2.remap(
        image,
        (xmap * image.shape[1] / _REFERENCE_WIDTH).astype(np.float32),
        (ymap * image.shape[0] / _REFERENCE_HEIGHT).astype(np.float32),
        cv2.INTER_LINEAR,
    )
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    digits = []
    for digit in (hsv[:, :49], hsv[:, 49:]):
        threshold = max(135., float(np.quantile(digit[..., 2], .99)) * .73)
        bright = (
            (digit[..., 2] >= threshold) & (digit[..., 1] <= 145)
            & (((digit[..., 0] >= 125) & (digit[..., 0] <= 175))
               | (digit[..., 1] <= 45))
        ).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(bright)
        candidates = [
            (index, x, y, w, h, area)
            for index, (x, y, w, h, area) in enumerate(stats[1:], 1)
            if 1 <= x and x + w <= 48 and y <= 8 and y + h >= 75
            and 18 <= w <= 46 and 66 <= h <= 80 and 180 <= area <= 1500
        ]
        if len(candidates) != 1:
            return None
        index, x, y, w, h, area = candidates[0]
        if not .06 <= area / (w * h) <= .45:
            return None
        # Include disconnected internal numeral outlines, but no exterior glow.
        clean = np.zeros_like(bright)
        clean[y:y + h, x:x + w] = bright[y:y + h, x:x + w]
        if clean.mean() > .38:
            return None
        digits.append(cv2.resize(
            clean.astype(np.float32), (24, 40), interpolation=cv2.INTER_AREA,
        ) >= .25)
    result = np.concatenate(digits, axis=1)
    result.setflags(write=False)
    return result


def read_contact_hud(image: np.ndarray) -> ContactHudReading:
    """Observe default HUD geometry independently in one uint8 BGR client frame.

    No files, network, state, OCR, or input mutation. Fractions have pixel-level
    uncertainty; callers must require stable multi-frame changes. Complete lack
    of live-color support, including burnout/KO, returns None instead of zero.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("image must be a nonempty HxWx3 BGR client frame")
    if image.shape[0] < 360 or image.shape[1] < 640:
        return ContactHudReading(None, None, None, None, None)
    return ContactHudReading(
        p1_health=_health(image, mirror=False),
        p2_health=_health(image, mirror=True),
        p1_drive=_drive(image, mirror=False),
        p2_drive=_drive(image, mirror=True),
        timer_mask=_timer(image),
    )
