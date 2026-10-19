from __future__ import annotations

import numpy as np
import pytest

from sf6_music_trigger.domain import CapturedFrame, ClientGeometry


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1280, 720), (0, 0, 1280, 720)),
        ((1920, 1080), (0, 0, 1920, 1080)),
        ((2560, 1440), (0, 0, 2560, 1440)),
        ((3840, 2160), (0, 0, 3840, 2160)),
        ((1680, 1050), (0, 52, 1680, 945)),
        ((1920, 1200), (0, 60, 1920, 1080)),
        ((2560, 1080), (320, 0, 1920, 1080)),
        ((3440, 1440), (440, 0, 2560, 1440)),
        ((3840, 1600), (498, 0, 2844, 1600)),
    ],
)
def test_centered_viewport_maps_common_client_sizes(
    size: tuple[int, int], expected: tuple[int, int, int, int]
) -> None:
    from sf6_music_trigger.viewport import centered_viewport

    rect = centered_viewport(*size, 16 / 9)

    assert (rect.left, rect.top, rect.width, rect.height) == expected


def test_dark_pillarbox_confirms_centered_viewport() -> None:
    from sf6_music_trigger.viewport import borders_confirm_viewport, centered_viewport

    image = np.full((720, 1280, 3), 96, dtype=np.uint8)
    image = np.pad(image, ((0, 0), (213, 213), (0, 0)), constant_values=6)
    rect = centered_viewport(image.shape[1], image.shape[0], 16 / 9)

    assert borders_confirm_viewport(image, rect) is True


def test_dark_letterbox_confirms_centered_viewport() -> None:
    from sf6_music_trigger.viewport import borders_confirm_viewport, centered_viewport

    image = np.full((900, 1600, 3), 96, dtype=np.uint8)
    image = np.pad(image, ((50, 50), (0, 0), (0, 0)), constant_values=0)
    rect = centered_viewport(image.shape[1], image.shape[0], 16 / 9)

    assert borders_confirm_viewport(image, rect) is True


def test_dark_gray_top_and_light_taskbar_confirm_16_10_viewport() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver, ViewportStatus

    image = np.full((800, 1280, 3), 96, dtype=np.uint8)
    image[:40, :] = 32
    image[760:, :] = 220
    image[760:, :200] = 48  # taskbar icons and dark desktop elements
    resolver = CenteredViewportResolver(16 / 9)

    assert resolver.resolve(_frame(1, image)).status is ViewportStatus.CONFIRMING
    assert resolver.resolve(_frame(2, image)).status is ViewportStatus.CONFIRMING
    result = resolver.resolve(_frame(3, image))

    assert result.status is ViewportStatus.NORMALIZED
    assert result.frame is not None
    assert result.frame.image.shape == (720, 1280, 3)
    assert result.frame.geometry == ClientGeometry(10, 60, 1280, 720)


def test_dark_top_and_decorated_bottom_do_not_confirm_viewport() -> None:
    from sf6_music_trigger.viewport import borders_confirm_viewport, centered_viewport

    image = np.full((800, 1280, 3), 96, dtype=np.uint8)
    image[:40, :] = 32
    image[760:, :] = 100
    image[760:, :600] = 220
    rect = centered_viewport(1280, 800, 16 / 9)

    assert borders_confirm_viewport(image, rect) is False


def test_bright_or_decorated_outer_area_fails_closed() -> None:
    from sf6_music_trigger.viewport import borders_confirm_viewport, centered_viewport

    image = np.full((720, 1706, 3), 6, dtype=np.uint8)
    image[:, 0:30] = 80
    rect = centered_viewport(image.shape[1], image.shape[0], 16 / 9)

    assert borders_confirm_viewport(image, rect) is False


@pytest.mark.parametrize(
    ("width", "height", "aspect"),
    [(0, 720, 16 / 9), (1280, 0, 16 / 9), (1280, 720, 0), (True, 720, 16 / 9)],
)
def test_centered_viewport_rejects_invalid_geometry(
    width: object, height: object, aspect: object
) -> None:
    from sf6_music_trigger.viewport import centered_viewport

    with pytest.raises((TypeError, ValueError)):
        centered_viewport(width, height, aspect)


def _frame(
    frame_id: int,
    image: np.ndarray,
    *,
    left: int = 10,
    top: int = 20,
) -> CapturedFrame:
    height, width = image.shape[:2]
    return CapturedFrame(
        frame_id,
        image,
        frame_id / 20,
        ClientGeometry(left, top, width, height),
    )


def _pillarboxed_frame(frame_id: int, *, bar_value: int = 0) -> CapturedFrame:
    content = np.full((720, 1280, 3), 96, dtype=np.uint8)
    image = np.pad(
        content,
        ((0, 0), (213, 213), (0, 0)),
        constant_values=bar_value,
    )
    return _frame(frame_id, image)


def test_native_aspect_passes_through_without_confirmation() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver, ViewportStatus

    resolver = CenteredViewportResolver(16 / 9)
    original = _frame(1, np.full((720, 1280, 3), 96, dtype=np.uint8))

    result = resolver.resolve(original)

    assert result.status is ViewportStatus.NATIVE
    assert result.frame is original


def test_non_native_viewport_requires_three_frames_then_locks_crop() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver, ViewportStatus

    resolver = CenteredViewportResolver(16 / 9)

    first = resolver.resolve(_pillarboxed_frame(1))
    second = resolver.resolve(_pillarboxed_frame(2))
    third = resolver.resolve(_pillarboxed_frame(3))

    assert first.frame is None and first.status is ViewportStatus.CONFIRMING
    assert second.frame is None and second.status is ViewportStatus.CONFIRMING
    assert third.status is ViewportStatus.NORMALIZED
    assert third.frame is not None
    assert third.frame.image.shape == (720, 1280, 3)
    assert third.frame.geometry == ClientGeometry(223, 20, 1280, 720)
    assert third.frame.frame_id == 3
    assert third.frame.monotonic_time == 0.15
    assert third.frame.image.flags.c_contiguous
    assert third.frame.image.flags.owndata

    bright_after_lock = _pillarboxed_frame(4, bar_value=90)
    locked = resolver.resolve(bright_after_lock)
    assert locked.status is ViewportStatus.NORMALIZED
    assert locked.frame is not None
    assert locked.frame.image.shape == (720, 1280, 3)


def test_bright_borders_fail_closed_and_break_pending_confirmation() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver, ViewportStatus

    resolver = CenteredViewportResolver(16 / 9)
    assert resolver.resolve(_pillarboxed_frame(1)).status is ViewportStatus.CONFIRMING

    bright = resolver.resolve(_pillarboxed_frame(2, bar_value=80))
    restarted = resolver.resolve(_pillarboxed_frame(3))

    assert bright.frame is None and bright.status is ViewportStatus.UNSUPPORTED
    assert restarted.frame is None and restarted.status is ViewportStatus.CONFIRMING


def test_geometry_change_discards_locked_viewport() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver, ViewportStatus

    resolver = CenteredViewportResolver(16 / 9)
    for frame_id in range(1, 4):
        resolver.resolve(_pillarboxed_frame(frame_id))

    moved = _pillarboxed_frame(4, bar_value=80)
    moved = CapturedFrame(
        moved.frame_id,
        moved.image,
        moved.monotonic_time,
        ClientGeometry(11, 20, moved.geometry.width, moved.geometry.height),
    )
    result = resolver.resolve(moved)

    assert result.frame is None
    assert result.status is ViewportStatus.UNSUPPORTED


def test_resolver_reset_discards_locked_viewport() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver, ViewportStatus

    resolver = CenteredViewportResolver(16 / 9)
    for frame_id in range(1, 4):
        resolver.resolve(_pillarboxed_frame(frame_id))
    resolver.reset()

    result = resolver.resolve(_pillarboxed_frame(4, bar_value=80))

    assert result.frame is None
    assert result.status is ViewportStatus.UNSUPPORTED


def test_effective_content_below_720p_fails_closed() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver, ViewportStatus

    resolver = CenteredViewportResolver(16 / 9)
    small = _frame(1, np.zeros((768, 1024, 3), dtype=np.uint8))

    result = resolver.resolve(small)

    assert result.frame is None
    assert result.status is ViewportStatus.TOO_SMALL


def test_image_geometry_mismatch_fails_closed() -> None:
    from sf6_music_trigger.viewport import CenteredViewportResolver, ViewportStatus

    resolver = CenteredViewportResolver(16 / 9)
    malformed = CapturedFrame(
        1,
        np.zeros((720, 1279, 3), dtype=np.uint8),
        0.05,
        ClientGeometry(0, 0, 1280, 720),
    )

    result = resolver.resolve(malformed)

    assert result.frame is None
    assert result.status is ViewportStatus.UNSUPPORTED
