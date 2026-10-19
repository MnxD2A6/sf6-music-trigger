from contextlib import contextmanager
import ctypes
from dataclasses import FrozenInstanceError
from fractions import Fraction
from numbers import Real
from pathlib import Path
import subprocess
import sys
import threading
import tomllib
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest

import sf6_music_trigger.capture as capture_module
import sf6_music_trigger.window as window_module
from sf6_music_trigger.capture import (
    CaptureStartError,
    DxcamCamera,
    DxcamFactory,
    DxcamFrameSource,
    DxcamOutput,
    FrameSource,
    FreshFrameGate,
    load_dxcam_factory,
)
from sf6_music_trigger.domain import (
    CapturedFrame,
    CaptureIssue,
    CaptureSample,
    CaptureStatus,
    ClientGeometry,
)
from sf6_music_trigger.window import (
    GameWindowAmbiguous,
    GameWindowError,
    GameWindowGeometryChanged,
    GameWindowIdentityChanged,
    GameWindowMinimized,
    GameWindowNotFound,
    GameWindowOccluded,
    MonitorDescriptor,
    Sf6WindowTarget,
    WindowIdentity,
    _CtypesWin32Facade,
    find_sf6_client_geometry,
    find_sf6_window_target,
)


class _IntSubclass(int):
    pass


def test_pyproject_pins_dxcam_winrt_to_03_series():
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert pyproject["project"]["dependencies"].count(
        "dxcam[winrt]>=0.3,<0.4"
    ) == 1
    assert not any(
        dependency.startswith("dxcam")
        and dependency != "dxcam[winrt]>=0.3,<0.4"
        for dependency in pyproject["project"]["dependencies"]
    )


def test_importing_capture_does_not_import_dxcam():
    script = """
import builtins
import importlib
import sys

sys.path.insert(0, 'src')
original_import = builtins.__import__
original_import_module = importlib.import_module

def guarded_import(name, *args, **kwargs):
    if name == 'dxcam' or name.startswith('dxcam.'):
        raise AssertionError('capture imported dxcam')
    return original_import(name, *args, **kwargs)

def guarded_import_module(name, package=None):
    if name == 'dxcam' or name.startswith('dxcam.'):
        raise AssertionError('capture imported dxcam')
    return original_import_module(name, package)

builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
import sf6_music_trigger.capture
assert not any(name == 'dxcam' or name.startswith('dxcam.') for name in sys.modules)
print('import-safe')
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "import-safe"


def test_constructing_source_does_not_import_dxcam_or_call_window_or_factory(
    monkeypatch,
):
    import builtins

    calls = []
    original_import = builtins.__import__
    original_import_module = capture_module.importlib.import_module

    def guarded_import(name, *args, **kwargs):
        if name == "dxcam" or name.startswith("dxcam."):
            raise AssertionError("constructor imported dxcam")
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, package=None):
        if name == "dxcam" or name.startswith("dxcam."):
            raise AssertionError("constructor imported dxcam")
        return original_import_module(name, package)

    def window_finder():
        calls.append("window")
        raise AssertionError("constructor called the window finder")

    def factory_loader():
        calls.append("factory")
        raise AssertionError("constructor called the DXcam loader")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(capture_module.importlib, "import_module", guarded_import_module)
    source = DxcamFrameSource(
        window_finder=window_finder,
        dxcam_factory_loader=factory_loader,
        target_fps=20,
    )

    assert type(source) is DxcamFrameSource
    assert calls == []


def test_dxcam_output_is_frozen_and_source_constructor_is_strictly_typed():
    geometry = ClientGeometry(-1920, -100, 1920, 1080)
    output = DxcamOutput(71, " DISPLAY-LEFT ", 2, 3, geometry, 270)

    assert output.monitor_handle == 71
    assert output.device_name == " DISPLAY-LEFT "
    assert output.device_index == 2
    assert output.output_index == 3
    assert output.desktop_geometry is geometry
    assert output.rotation_angle == 270
    assert output != DxcamOutput(71, " DISPLAY-LEFT ", 2, 3, geometry, 0)
    with pytest.raises(FrozenInstanceError):
        output.rotation_angle = 0

    valid = dict(
        monitor_handle=71,
        device_name="DISPLAY-LEFT",
        device_index=2,
        output_index=3,
        desktop_geometry=geometry,
        rotation_angle=0,
    )
    invalid_fields = {
        "monitor_handle": [True, False, 0, -1, 1.0, "71", _IntSubclass(71)],
        "device_name": ["", " \t", 71, None],
        "device_index": [True, False, -1, 1.0, "2", _IntSubclass(2)],
        "output_index": [True, False, -1, 1.0, "3", _IntSubclass(3)],
        "desktop_geometry": [object(), (-1920, -100, 1920, 1080)],
        "rotation_angle": [
            True,
            False,
            -90,
            45,
            360,
            90.0,
            "90",
            _IntSubclass(90),
        ],
    }
    for field, values in invalid_fields.items():
        for value in values:
            kwargs = valid | {field: value}
            with pytest.raises((TypeError, ValueError), match=field):
                DxcamOutput(**kwargs)

    def window_finder():
        raise AssertionError("not called")

    def factory_loader():
        raise AssertionError("not called")

    for bad_finder in (None, 1, "finder"):
        with pytest.raises(TypeError, match="window_finder"):
            DxcamFrameSource(
                window_finder=bad_finder,
                dxcam_factory_loader=factory_loader,
            )
    for bad_loader in (None, 1, "loader"):
        with pytest.raises(TypeError, match="dxcam_factory_loader"):
            DxcamFrameSource(
                window_finder=window_finder,
                dxcam_factory_loader=bad_loader,
            )
    for bad_fps in (True, False, 9, 31, 20.0, "20", _IntSubclass(20)):
        with pytest.raises((TypeError, ValueError), match="target_fps"):
            DxcamFrameSource(
                window_finder=window_finder,
                dxcam_factory_loader=factory_loader,
                target_fps=bad_fps,
            )

    assert DxcamCamera._is_protocol is True
    assert DxcamFactory._is_protocol is True
    assert FrameSource._is_protocol is True


class _NoFloatFraction(Fraction):
    def __float__(self):
        raise AssertionError("Rational values must not be converted to float")


class _ExplodingOnceFraction(Fraction):
    def __new__(cls, numerator=0, denominator=None):
        if denominator is None:
            instance = super().__new__(cls, numerator)
        else:
            instance = super().__new__(cls, numerator, denominator)
        instance.should_explode = True
        return instance

    def __rsub__(self, other):
        if self.should_explode:
            self.should_explode = False
            raise RuntimeError("comparison probe")
        return super().__rsub__(other)


def test_capture_issue_has_exact_public_members_without_aliases():
    assert [(item.name, item.value) for item in CaptureIssue] == [
        ("BLACK_FRAME", "BLACK_FRAME"),
        ("NO_NEW_FRAME", "NO_NEW_FRAME"),
        ("WINDOW_CHANGED", "WINDOW_CHANGED"),
        ("GEOMETRY_CHANGED", "GEOMETRY_CHANGED"),
        ("BACKEND_ERROR", "BACKEND_ERROR"),
        ("UNSUPPORTED_OUTPUT", "UNSUPPORTED_OUTPUT"),
    ]
    assert len(CaptureIssue.__members__) == 6


def _captured_frame(
    image: np.ndarray | None = None,
    *,
    frame_id: int = 0,
    monotonic_time: Real = 1,
    geometry: ClientGeometry | None = None,
) -> CapturedFrame:
    if image is None:
        image = np.zeros((2, 3, 3), dtype=np.uint8)
    if geometry is None:
        geometry = ClientGeometry(-3, -2, 3, 2)
    return CapturedFrame(frame_id, image, monotonic_time, geometry)


def test_capture_sample_requires_running_exactly_when_frame_present():
    frame = _captured_frame()

    class FrameLike:
        frame_id = frame.frame_id
        image = frame.image
        monotonic_time = frame.monotonic_time
        geometry = frame.geometry

    class StatusLike:
        value = "RUNNING"

    assert CaptureSample(CaptureStatus.RUNNING, frame=frame).frame is frame
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(CaptureStatus.RUNNING)
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(CaptureStatus.OCCLUDED, frame=frame)
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(
            CaptureStatus.RUNNING,
            frame=frame,
            issue=CaptureIssue.BACKEND_ERROR,
        )
    with pytest.raises(TypeError):
        CaptureSample("RUNNING", frame=frame)
    with pytest.raises(TypeError):
        CaptureSample(CaptureStatus.RUNNING, frame=object())
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(CaptureStatus.RUNNING, frame=frame, issue=CaptureIssue.BACKEND_ERROR)
    with pytest.raises(TypeError):
        CaptureSample("RUNNING", frame=frame)
    with pytest.raises(TypeError):
        CaptureSample(StatusLike(), frame=frame)
    with pytest.raises(TypeError):
        CaptureSample(CaptureStatus.RUNNING, frame=FrameLike())


def test_capture_sample_lost_requires_exact_capture_issue():
    assert list(CaptureIssue.__members__) == [
        "BLACK_FRAME",
        "NO_NEW_FRAME",
        "WINDOW_CHANGED",
        "GEOMETRY_CHANGED",
        "BACKEND_ERROR",
        "UNSUPPORTED_OUTPUT",
    ]
    assert [issue.value for issue in CaptureIssue] == [
        "BLACK_FRAME",
        "NO_NEW_FRAME",
        "WINDOW_CHANGED",
        "GEOMETRY_CHANGED",
        "BACKEND_ERROR",
        "UNSUPPORTED_OUTPUT",
    ]
    sample = CaptureSample(CaptureStatus.LOST, issue=CaptureIssue.BACKEND_ERROR)

    assert sample.issue is CaptureIssue.BACKEND_ERROR
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(CaptureStatus.LOST)
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(CaptureStatus.LOST, issue="BACKEND_ERROR")
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(
            CaptureStatus.LOST,
            frame=_captured_frame(),
            issue=CaptureIssue.BACKEND_ERROR,
        )
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(
            CaptureStatus.LOST,
            frame=_captured_frame(),
            issue=CaptureIssue.BACKEND_ERROR,
        )


@pytest.mark.parametrize(
    "status",
    [CaptureStatus.OCCLUDED, CaptureStatus.MINIMIZED, CaptureStatus.STOPPED],
)
def test_capture_sample_nonrunning_statuses_forbid_frame_and_issue(status: CaptureStatus):
    frame = _captured_frame()

    assert CaptureSample(status).status is status
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(status, frame=frame)
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(status, issue=CaptureIssue.NO_NEW_FRAME)
    with pytest.raises((TypeError, ValueError)):
        CaptureSample(status, frame=frame, issue=CaptureIssue.NO_NEW_FRAME)


def test_capture_sample_requires_owned_writable_c_contiguous_bgr():
    wrong_dtype = np.zeros((2, 3, 3), dtype=np.float32)
    wrong_channels = np.zeros((2, 3, 4), dtype=np.uint8)
    non_contiguous = np.asfortranarray(np.zeros((2, 3, 3), dtype=np.uint8))
    non_owned = np.zeros((2, 3, 3), dtype=np.uint8).view()
    read_only = np.zeros((2, 3, 3), dtype=np.uint8)
    read_only.setflags(write=False)

    for image in (wrong_dtype, wrong_channels, non_contiguous, non_owned, read_only):
        with pytest.raises((TypeError, ValueError)):
            CaptureSample(CaptureStatus.RUNNING, frame=_captured_frame(image))


def test_capture_sample_requires_image_shape_to_match_geometry():
    geometry = ClientGeometry(10, 20, 3, 2)

    valid = _captured_frame(np.zeros((2, 3, 3), dtype=np.uint8), geometry=geometry)
    assert CaptureSample(CaptureStatus.RUNNING, frame=valid).frame is valid

    for shape in ((1, 3, 3), (2, 4, 3)):
        frame = _captured_frame(np.zeros(shape, dtype=np.uint8), geometry=geometry)
        with pytest.raises(ValueError):
            CaptureSample(CaptureStatus.RUNNING, frame=frame)


def test_capture_sample_is_frozen_and_does_not_copy_or_change_flags():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    frame = _captured_frame(image)
    sample = CaptureSample(CaptureStatus.RUNNING, frame=frame)

    assert sample.frame is frame
    assert sample.frame.image is image
    assert image.flags.owndata
    assert image.flags.c_contiguous
    assert image.flags.writeable
    with pytest.raises(FrozenInstanceError):
        sample.status = CaptureStatus.STOPPED


def test_captured_frame_time_annotation_is_real_and_fraction_identity_is_preserved():
    timestamp = Fraction(10**100, 3)
    frame = _captured_frame(monotonic_time=timestamp)
    sample = CaptureSample(CaptureStatus.RUNNING, frame=frame)

    assert get_type_hints(CapturedFrame)["monotonic_time"] is Real
    assert sample.frame is frame
    assert sample.frame.monotonic_time is timestamp


def test_fresh_gate_constructor_retains_valid_real_limits_without_coercion():
    age = _NoFloatFraction(1, 4)
    future = _NoFloatFraction(1, 20)
    gate = FreshFrameGate(age, future)

    assert gate.max_age_seconds is age
    assert gate.max_future_seconds is future


@pytest.mark.parametrize("value", [True, False, "0.25", None, object()])
@pytest.mark.parametrize("field", ["max_age_seconds", "max_future_seconds"])
def test_fresh_gate_constructor_rejects_bool_and_nonreal_limits(field, value):
    kwargs = {field: value}
    with pytest.raises(TypeError):
        FreshFrameGate(**kwargs)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["max_age_seconds", "max_future_seconds"])
def test_fresh_gate_constructor_rejects_negative_and_nonfinite_limits(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        FreshFrameGate(**kwargs)


def test_fresh_gate_accepts_exact_age_and_future_boundaries():
    age_gate = FreshFrameGate(Fraction(1, 4), Fraction(1, 20))
    future_gate = FreshFrameGate(Fraction(1, 4), Fraction(1, 20))

    assert age_gate.accept(
        _captured_frame(frame_id=1, monotonic_time=Fraction(39, 4)),
        now=Fraction(10),
    )
    assert future_gate.accept(
        _captured_frame(frame_id=1, monotonic_time=Fraction(201, 20)),
        now=Fraction(10),
    )


def test_fresh_gate_rejects_just_beyond_age_and_future_boundaries():
    epsilon = Fraction(1, 10**100)
    age_gate = FreshFrameGate(Fraction(1, 4), Fraction(1, 20))
    future_gate = FreshFrameGate(Fraction(1, 4), Fraction(1, 20))

    assert not age_gate.accept(
        _captured_frame(
            frame_id=1,
            monotonic_time=Fraction(10) - Fraction(1, 4) - epsilon,
        ),
        now=Fraction(10),
    )
    assert not future_gate.accept(
        _captured_frame(
            frame_id=1,
            monotonic_time=Fraction(10) + Fraction(1, 20) + epsilon,
        ),
        now=Fraction(10),
    )


def test_fresh_gate_requires_strictly_increasing_id_and_timestamp():
    gate = FreshFrameGate(100, 100)
    assert gate.accept(_captured_frame(frame_id=1, monotonic_time=10), now=10)

    assert not gate.accept(_captured_frame(frame_id=1, monotonic_time=11), now=11)
    assert not gate.accept(_captured_frame(frame_id=0, monotonic_time=11), now=11)
    assert not gate.accept(_captured_frame(frame_id=2, monotonic_time=10), now=10)
    assert not gate.accept(_captured_frame(frame_id=2, monotonic_time=9), now=9)


def test_fresh_gate_equal_pixels_with_new_ordering_remain_fresh():
    image = np.full((2, 3, 3), 17, dtype=np.uint8)
    gate = FreshFrameGate(1, 1)

    assert gate.accept(_captured_frame(image, frame_id=1, monotonic_time=10), now=10)
    assert gate.accept(_captured_frame(image, frame_id=2, monotonic_time=11), now=11)


def test_fresh_gate_rejection_does_not_advance_either_watermark():
    gate = FreshFrameGate(100, 100)
    assert gate.accept(_captured_frame(frame_id=1, monotonic_time=10), now=10)
    assert not gate.accept(_captured_frame(frame_id=99, monotonic_time=9), now=10)
    assert gate.accept(_captured_frame(frame_id=2, monotonic_time=Fraction(101, 10)), now=Fraction(101, 10))

    other = FreshFrameGate(100, 100)
    assert other.accept(_captured_frame(frame_id=1, monotonic_time=10), now=10)
    assert not other.accept(_captured_frame(frame_id=1, monotonic_time=20), now=20)
    assert other.accept(
        _captured_frame(frame_id=2, monotonic_time=Fraction(101, 10)),
        now=Fraction(101, 10),
    )


@pytest.mark.parametrize("now", [True, "10", None, object()])
def test_fresh_gate_invalid_now_type_raises_without_advancing_watermarks(now):
    gate = FreshFrameGate(1, 1)
    candidate = _captured_frame(frame_id=1, monotonic_time=10)

    with pytest.raises(TypeError):
        gate.accept(candidate, now=now)
    assert gate.accept(candidate, now=10)


@pytest.mark.parametrize("now", [-1, float("nan"), float("inf"), float("-inf")])
def test_fresh_gate_invalid_now_value_raises_without_advancing_watermarks(now):
    gate = FreshFrameGate(1, 1)
    candidate = _captured_frame(frame_id=1, monotonic_time=0)

    with pytest.raises(ValueError):
        gate.accept(candidate, now=now)
    assert gate.accept(candidate, now=0)


def test_fresh_gate_comparison_error_does_not_advance_either_watermark():
    gate = FreshFrameGate(1, 1)
    timestamp = _ExplodingOnceFraction(1, 2)
    candidate = _captured_frame(frame_id=1, monotonic_time=timestamp)

    with pytest.raises(RuntimeError, match="comparison probe"):
        gate.accept(candidate, now=1)
    assert gate.accept(candidate, now=1)


def test_fresh_gate_requires_actual_captured_frame_before_now_validation():
    gate = FreshFrameGate()
    with pytest.raises(TypeError, match="frame"):
        gate.accept(object(), now="also invalid")


def test_fresh_gate_reset_clears_both_watermarks():
    gate = FreshFrameGate(100, 100)
    assert gate.accept(_captured_frame(frame_id=10, monotonic_time=10), now=10)

    gate.reset()

    assert gate.accept(_captured_frame(frame_id=0, monotonic_time=0), now=0)


def test_fresh_gate_accepts_fraction_and_huge_rational_without_coercion():
    limit = _NoFloatFraction(10**1000, 3)
    timestamp = _NoFloatFraction(10**999, 7)
    now = timestamp + _NoFloatFraction(1, 10)
    gate = FreshFrameGate(limit, limit)

    assert gate.accept(_captured_frame(frame_id=1, monotonic_time=timestamp), now=now)
    assert gate.max_age_seconds is limit
    assert gate.max_future_seconds is limit


def test_fresh_gate_mixed_rational_and_float_comparisons_never_narrow_rational():
    huge = _NoFloatFraction(10**1000, 1)

    future_gate = FreshFrameGate(0.0, 0.0)
    assert not future_gate.accept(
        _captured_frame(frame_id=1, monotonic_time=huge),
        now=0.0,
    )

    age_gate = FreshFrameGate(0.0, 0.0)
    assert not age_gate.accept(
        _captured_frame(frame_id=1, monotonic_time=0.0),
        now=huge,
    )


_FIRST_IDENTITY = WindowIdentity(101, 2001, (7 << 32) | 11)
_SECOND_IDENTITY = WindowIdentity(101, 2001, (8 << 32) | 13)
_CLIENT = ClientGeometry(-80, 40, 640, 360)
_MOVED_CLIENT = ClientGeometry(-70, 45, 640, 360)
_DESKTOP = ClientGeometry(-1920, 0, 2560, 1080)
_MONITOR = MonitorDescriptor(91, r"\\.\DISPLAY2", _DESKTOP)


class _FakeWin32:
    def __init__(
        self,
        *,
        windows=(101,),
        identities=None,
        minimized=None,
        visible=None,
        cloaked=None,
        geometries=None,
        monitors=None,
        fail_enum=None,
    ):
        self.windows = tuple(windows)
        self.identities = identities or {
            101: [
                (_FIRST_IDENTITY, r"C:\Games\StreetFighter6.exe"),
                (_FIRST_IDENTITY, r"C:\Games\StreetFighter6.exe"),
            ]
        }
        self.minimized = minimized or {101: [False, False]}
        self.visible = visible or {101: [True, True]}
        self.cloaked = cloaked or {101: [False, False]}
        self.geometries = geometries or {101: [_CLIENT, _CLIENT]}
        self.monitors = monitors or {101: [_MONITOR, _MONITOR]}
        self.fail_enum = fail_enum
        self.calls = []
        self.monitor_geometry_args = []

    @staticmethod
    def _next(values, hwnd):
        sequence = values[hwnd]
        value = sequence.pop(0) if len(sequence) > 1 else sequence[0]
        if isinstance(value, BaseException):
            raise value
        return value

    @contextmanager
    def per_monitor_v2(self):
        self.calls.append("dpi.enter")
        try:
            yield
        finally:
            self.calls.append("dpi.exit")

    def enum_top_level_windows(self):
        self.calls.append("enum")
        if self.fail_enum is not None:
            raise self.fail_enum
        return self.windows

    def is_window_visible(self, hwnd):
        self.calls.append(f"visible:{hwnd}")
        return self._next(self.visible, hwnd)

    def is_window_minimized(self, hwnd):
        self.calls.append(f"minimized:{hwnd}")
        return self._next(self.minimized, hwnd)

    def is_window_cloaked(self, hwnd):
        self.calls.append(f"cloaked:{hwnd}")
        return self._next(self.cloaked, hwnd)

    def read_process_identity(self, hwnd):
        self.calls.append(f"identity:{hwnd}")
        return self._next(self.identities, hwnd)

    def client_geometry(self, hwnd):
        self.calls.append(f"geometry:{hwnd}")
        return self._next(self.geometries, hwnd)

    def monitor_descriptor(self, hwnd, geometry):
        self.calls.append(f"monitor:{hwnd}")
        self.monitor_geometry_args.append((hwnd, geometry))
        return self._next(self.monitors, hwnd)

    def window_title(self, hwnd):
        raise AssertionError("window titles are outside the finder contract")


def test_window_identity_monitor_and_target_are_frozen_and_strictly_typed():
    identity = WindowIdentity(101, 2001, 0)
    monitor = MonitorDescriptor(91, r"\\.\DISPLAY2", _DESKTOP)
    target = Sf6WindowTarget(identity, _CLIENT, monitor)

    with pytest.raises(FrozenInstanceError):
        identity.pid = 3
    with pytest.raises(FrozenInstanceError):
        monitor.handle = 3
    with pytest.raises(FrozenInstanceError):
        target.geometry = _MOVED_CLIENT

    for args in (
        (True, 1, 0),
        (0, 1, 0),
        (-1, 1, 0),
        (1.0, 1, 0),
        (1, False, 0),
        (1, 0, 0),
        (1, -1, 0),
        (1, 1.0, 0),
        (1, 1, True),
        (1, 1, 0.0),
    ):
        with pytest.raises((TypeError, ValueError)):
            WindowIdentity(*args)
    with pytest.raises(ValueError):
        WindowIdentity(1, 1, -1)

    for args in (
        (True, "DISPLAY", _DESKTOP),
        (0, "DISPLAY", _DESKTOP),
        (-1, "DISPLAY", _DESKTOP),
        (1.0, "DISPLAY", _DESKTOP),
        (1, object(), _DESKTOP),
        (1, " ", _DESKTOP),
        (1, "DISPLAY", object()),
    ):
        with pytest.raises((TypeError, ValueError)):
            MonitorDescriptor(*args)

    with pytest.raises(TypeError):
        Sf6WindowTarget(object(), _CLIENT, monitor)
    with pytest.raises(TypeError):
        Sf6WindowTarget(identity, object(), monitor)
    with pytest.raises(TypeError):
        Sf6WindowTarget(identity, _CLIENT, object())
    with pytest.raises(ValueError):
        Sf6WindowTarget(
            identity,
            ClientGeometry(500, 40, 640, 360),
            monitor,
        )


@pytest.mark.parametrize(
    "outside",
    [
        ClientGeometry(-11, -20, 1, 1),
        ClientGeometry(-10, -21, 1, 1),
        ClientGeometry(89, -20, 2, 1),
        ClientGeometry(-10, 59, 1, 2),
    ],
)
def test_target_requires_complete_half_open_monitor_containment(outside):
    identity = WindowIdentity(1, 2, 3)
    desktop = ClientGeometry(-10, -20, 100, 80)
    monitor = MonitorDescriptor(4, "DISPLAY", desktop)

    exact_edge = Sf6WindowTarget(identity, desktop, monitor)
    assert exact_edge.geometry is desktop

    with pytest.raises(ValueError):
        Sf6WindowTarget(identity, outside, monitor)


def test_finder_uses_executable_basename_not_window_title():
    other = WindowIdentity(11, 111, 1)
    game = WindowIdentity(22, 222, 2)
    fake = _FakeWin32(
        windows=(11, 22),
        identities={
            11: [(other, r"C:\Games\NotStreetFighter6.exe")],
            22: [
                (game, r"D:\Steam\STREETFIGHTER6.EXE"),
                (game, r"D:\Steam\STREETFIGHTER6.EXE"),
            ],
        },
        minimized={22: [False, False]},
        visible={22: [True, True]},
        cloaked={22: [False, False]},
        geometries={22: [_CLIENT, _CLIENT]},
        monitors={22: [_MONITOR, _MONITOR]},
    )

    target = find_sf6_window_target(win32=fake)

    assert target.identity is game
    assert "identity:11" in fake.calls
    assert "identity:22" in fake.calls


def test_finder_skips_unreadable_unrelated_windows_before_sf6():
    game = WindowIdentity(22, 222, 2)
    fake = _FakeWin32(
        windows=(11, 22),
        identities={
            11: [GameWindowError("unrelated process access denied")],
            22: [
                (game, r"D:\Steam\StreetFighter6.exe"),
                (game, r"D:\Steam\StreetFighter6.exe"),
            ],
        },
        minimized={22: [False, False]},
        visible={22: [True, True]},
        cloaked={22: [False, False]},
        geometries={22: [_CLIENT, _CLIENT]},
        monitors={22: [_MONITOR, _MONITOR]},
    )

    target = find_sf6_window_target(win32=fake)

    assert target.identity is game
    assert fake.calls.index("identity:11") < fake.calls.index("identity:22")


def test_finder_ignores_hidden_auxiliary_windows_from_the_sf6_process():
    game = WindowIdentity(101, 2001, 1)
    hidden_auxiliary = WindowIdentity(202, 2001, 1)
    fake = _FakeWin32(
        windows=(101, 202),
        identities={
            101: [
                (game, "StreetFighter6.exe"),
                (game, "StreetFighter6.exe"),
            ],
            202: [(hidden_auxiliary, "StreetFighter6.exe")],
        },
        minimized={101: [False, False], 202: [False]},
        visible={101: [True, True], 202: [False]},
        cloaked={101: [False, False]},
        geometries={101: [_CLIENT, _CLIENT]},
        monitors={101: [_MONITOR, _MONITOR]},
    )

    target = find_sf6_window_target(win32=fake)

    assert target.identity is game
    assert "visible:202" in fake.calls
    assert "geometry:202" not in fake.calls


def test_finder_fails_closed_for_zero_or_multiple_sf6_windows():
    with pytest.raises(GameWindowNotFound):
        find_sf6_window_target(win32=_FakeWin32(windows=()))

    not_game = WindowIdentity(11, 111, 1)
    with pytest.raises(GameWindowNotFound):
        find_sf6_window_target(
            win32=_FakeWin32(
                windows=(11,),
                identities={11: [(not_game, "Other.exe")]},
            )
        )

    second = WindowIdentity(202, 2002, 2)
    with pytest.raises(GameWindowAmbiguous):
        find_sf6_window_target(
            win32=_FakeWin32(
                windows=(101, 202),
                identities={
                    101: [(_FIRST_IDENTITY, "StreetFighter6.exe")],
                    202: [(second, "streetfighter6.EXE")],
                },
                minimized={101: [False], 202: [False]},
                visible={101: [True], 202: [True]},
            )
        )


def test_finder_queries_identity_before_minimized_before_hidden_cloaked_before_geometry():
    fake = _FakeWin32()

    find_sf6_window_target(win32=fake)

    assert fake.calls == [
        "dpi.enter",
        "enum",
        "identity:101",
        "minimized:101",
        "visible:101",
        "cloaked:101",
        "geometry:101",
        "monitor:101",
        "identity:101",
        "minimized:101",
        "visible:101",
        "cloaked:101",
        "geometry:101",
        "monitor:101",
        "dpi.exit",
    ]


def test_finder_defines_occluded_as_hidden_or_dwm_cloaked_only():
    hidden = _FakeWin32(visible={101: [False]})
    with pytest.raises(GameWindowOccluded):
        find_sf6_window_target(win32=hidden)
    assert "cloaked:101" not in hidden.calls
    assert "geometry:101" not in hidden.calls

    cloaked = _FakeWin32(cloaked={101: [True]})
    with pytest.raises(GameWindowOccluded):
        find_sf6_window_target(win32=cloaked)
    assert "geometry:101" not in cloaked.calls

    minimized = _FakeWin32(minimized={101: [True]})
    with pytest.raises(GameWindowMinimized):
        find_sf6_window_target(win32=minimized)
    assert "visible:101" not in minimized.calls


def test_finder_never_claims_to_detect_ordinary_window_overlap():
    class CoverageTrapWin32(_FakeWin32):
        def is_window_covered(self, hwnd):
            self.calls.append(f"covered:{hwnd}")
            return True

    fake = CoverageTrapWin32()

    target = find_sf6_window_target(win32=fake)

    assert target.identity == _FIRST_IDENTITY
    assert not any(call.startswith("covered:") for call in fake.calls)


def test_finder_rechecks_identity_geometry_and_monitor_before_return():
    first_identity = WindowIdentity(
        _FIRST_IDENTITY.hwnd,
        _FIRST_IDENTITY.pid,
        _FIRST_IDENTITY.process_created_at_ticks,
    )
    second_identity = WindowIdentity(
        _FIRST_IDENTITY.hwnd,
        _FIRST_IDENTITY.pid,
        _FIRST_IDENTITY.process_created_at_ticks,
    )
    first_geometry = ClientGeometry(
        _CLIENT.left,
        _CLIENT.top,
        _CLIENT.width,
        _CLIENT.height,
    )
    second_geometry = ClientGeometry(
        _CLIENT.left,
        _CLIENT.top,
        _CLIENT.width,
        _CLIENT.height,
    )
    first_monitor = MonitorDescriptor(
        _MONITOR.handle,
        _MONITOR.device_name,
        ClientGeometry(
            _DESKTOP.left,
            _DESKTOP.top,
            _DESKTOP.width,
            _DESKTOP.height,
        ),
    )
    second_monitor = MonitorDescriptor(
        _MONITOR.handle,
        _MONITOR.device_name,
        ClientGeometry(
            _DESKTOP.left,
            _DESKTOP.top,
            _DESKTOP.width,
            _DESKTOP.height,
        ),
    )
    fake = _FakeWin32(
        identities={
            101: [
                (first_identity, "StreetFighter6.exe"),
                (second_identity, "StreetFighter6.exe"),
            ]
        },
        geometries={101: [first_geometry, second_geometry]},
        monitors={101: [first_monitor, second_monitor]},
    )

    target = find_sf6_window_target(win32=fake)

    assert target.identity is second_identity
    assert target.geometry is second_geometry
    assert target.monitor is second_monitor
    assert fake.calls.count("identity:101") == 2
    assert fake.calls.count("geometry:101") == 2
    assert fake.calls.count("monitor:101") == 2
    assert find_sf6_client_geometry(win32=_FakeWin32()) == _CLIENT


def test_finder_detects_same_hwnd_pid_with_changed_creation_ticks():
    fake = _FakeWin32(
        identities={
            101: [
                (_FIRST_IDENTITY, "StreetFighter6.exe"),
                (_SECOND_IDENTITY, "StreetFighter6.exe"),
            ]
        }
    )

    with pytest.raises(GameWindowIdentityChanged) as caught:
        find_sf6_window_target(win32=fake)

    assert caught.value.before.identity is _FIRST_IDENTITY
    assert caught.value.after is not None
    assert caught.value.after.identity is _SECOND_IDENTITY


def test_finder_second_query_disappearance_is_typed_identity_change():
    fake = _FakeWin32(
        identities={
            101: [
                (_FIRST_IDENTITY, "StreetFighter6.exe"),
                GameWindowError("window disappeared"),
            ]
        }
    )

    with pytest.raises(GameWindowIdentityChanged) as caught:
        find_sf6_window_target(win32=fake)

    assert caught.value.before == Sf6WindowTarget(_FIRST_IDENTITY, _CLIENT, _MONITOR)
    assert caught.value.after is None


def test_finder_identity_change_carries_typed_before_and_optional_after():
    before = Sf6WindowTarget(_FIRST_IDENTITY, _CLIENT, _MONITOR)
    after = Sf6WindowTarget(_SECOND_IDENTITY, _MOVED_CLIENT, _MONITOR)
    changed = GameWindowIdentityChanged(before=before, after=after)
    disappeared = GameWindowIdentityChanged(before=before, after=None)

    assert changed.before is before
    assert changed.after is after
    assert disappeared.before is before
    assert disappeared.after is None
    with pytest.raises(AttributeError):
        changed.before = after
    with pytest.raises(AttributeError):
        changed.after = None
    with pytest.raises(TypeError):
        GameWindowIdentityChanged(before=object(), after=None)
    with pytest.raises(TypeError):
        GameWindowIdentityChanged(before=before, after=object())
    with pytest.raises(ValueError):
        GameWindowIdentityChanged(before=before, after=before)
    with pytest.raises(ValueError):
        GameWindowIdentityChanged(
            before=before,
            after=Sf6WindowTarget(_FIRST_IDENTITY, _MOVED_CLIENT, _MONITOR),
        )


def test_finder_incomplete_changed_identity_diagnostic_keeps_typed_precedence():
    fake = _FakeWin32(
        identities={
            101: [
                (_FIRST_IDENTITY, "StreetFighter6.exe"),
                (_SECOND_IDENTITY, "StreetFighter6.exe"),
            ]
        },
        geometries={101: [_CLIENT, GameWindowError("second geometry vanished")]},
    )

    with pytest.raises(GameWindowIdentityChanged) as caught:
        find_sf6_window_target(win32=fake)

    assert caught.value.before == Sf6WindowTarget(_FIRST_IDENTITY, _CLIENT, _MONITOR)
    assert caught.value.after is None
    assert fake.calls.count("minimized:101") == 1
    assert fake.calls.count("visible:101") == 1
    assert fake.calls.count("cloaked:101") == 1


@pytest.mark.parametrize(
    "second_minimized,second_cloaked",
    [(True, False), (False, True)],
)
def test_finder_second_identity_change_precedes_new_window_minimized_or_cloaked_state(
    second_minimized,
    second_cloaked,
):
    fake = _FakeWin32(
        identities={
            101: [
                (_FIRST_IDENTITY, "StreetFighter6.exe"),
                (_SECOND_IDENTITY, "StreetFighter6.exe"),
            ]
        },
        minimized={101: [False, second_minimized]},
        cloaked={101: [False, second_cloaked]},
    )

    with pytest.raises(GameWindowIdentityChanged):
        find_sf6_window_target(win32=fake)

    assert fake.calls.count("minimized:101") == 1
    assert fake.calls.count("cloaked:101") == 1


def test_finder_same_identity_geometry_change_carries_two_typed_targets():
    fake = _FakeWin32(geometries={101: [_CLIENT, _MOVED_CLIENT]})

    with pytest.raises(GameWindowGeometryChanged) as caught:
        find_sf6_window_target(win32=fake)

    assert caught.value.before == Sf6WindowTarget(_FIRST_IDENTITY, _CLIENT, _MONITOR)
    assert caught.value.after == Sf6WindowTarget(
        _FIRST_IDENTITY,
        _MOVED_CLIENT,
        _MONITOR,
    )
    with pytest.raises(AttributeError):
        caught.value.after = caught.value.before
    with pytest.raises(AttributeError):
        caught.value.before = caught.value.after
    with pytest.raises(TypeError):
        GameWindowGeometryChanged(before=object(), after=caught.value.after)
    with pytest.raises(ValueError):
        GameWindowGeometryChanged(before=caught.value.before, after=caught.value.before)
    with pytest.raises(ValueError):
        GameWindowGeometryChanged(
            before=caught.value.before,
            after=Sf6WindowTarget(_SECOND_IDENTITY, _MOVED_CLIENT, _MONITOR),
        )


@pytest.mark.parametrize(
    "changed_monitor",
    [
        MonitorDescriptor(92, r"\\.\DISPLAY2", _DESKTOP),
        MonitorDescriptor(91, r"\\.\DISPLAY3", _DESKTOP),
        MonitorDescriptor(91, r"\\.\DISPLAY2", ClientGeometry(-1921, 0, 2561, 1080)),
    ],
)
def test_finder_same_identity_monitor_change_is_typed_geometry_change(
    changed_monitor,
):
    fake = _FakeWin32(monitors={101: [_MONITOR, changed_monitor]})

    with pytest.raises(GameWindowGeometryChanged) as caught:
        find_sf6_window_target(win32=fake)

    assert caught.value.before == Sf6WindowTarget(_FIRST_IDENTITY, _CLIENT, _MONITOR)
    assert caught.value.after == Sf6WindowTarget(
        _FIRST_IDENTITY,
        _CLIENT,
        changed_monitor,
    )


def test_finder_uses_both_client_to_screen_corners():
    first_geometry = ClientGeometry(17, 29, 701, 413)
    second_geometry = ClientGeometry(17, 29, 701, 413)
    monitor = MonitorDescriptor(3, "DISPLAY", ClientGeometry(0, 0, 1000, 700))
    fake = _FakeWin32(
        geometries={101: [first_geometry, second_geometry]},
        monitors={101: [monitor, monitor]},
    )

    assert find_sf6_window_target(win32=fake).geometry is second_geometry
    assert fake.calls.count("geometry:101") == 2
    assert fake.monitor_geometry_args == [
        (101, first_geometry),
        (101, second_geometry),
    ]
    assert fake.monitor_geometry_args[0][1] is first_geometry
    assert fake.monitor_geometry_args[1][1] is second_geometry


def test_finder_accepts_negative_desktop_origin():
    target = find_sf6_window_target(win32=_FakeWin32())

    assert target.geometry.left == -80
    assert target.monitor.desktop_geometry.left == -1920


def test_finder_rejects_client_crossing_monitor_boundary():
    crossing = ClientGeometry(-100, 40, 900, 360)
    monitor = MonitorDescriptor(91, "DISPLAY", ClientGeometry(-800, 0, 1000, 1080))
    fake = _FakeWin32(
        geometries={101: [crossing]},
        monitors={101: [monitor]},
    )

    with pytest.raises(GameWindowError):
        find_sf6_window_target(win32=fake)


def test_dpi_context_restores_previous_context_on_success_and_failure():
    success = _FakeWin32()
    find_sf6_window_target(win32=success)
    assert success.calls[0] == "dpi.enter"
    assert success.calls[-1] == "dpi.exit"

    failure = _FakeWin32(fail_enum=RuntimeError("enumeration failed"))
    with pytest.raises(GameWindowError):
        find_sf6_window_target(win32=failure)
    assert failure.calls == ["dpi.enter", "enum", "dpi.exit"]


class _FakeRawWin32Api:
    def __init__(
        self,
        *,
        fail_at=None,
        windows=(101,),
        pid=2001,
        process_path=r"Q:\private\account-name\StreetFighter6.exe",
        created_low=0x89ABCDEF,
        created_high=0x12345678,
        visible=True,
        minimized=False,
        cloaked=0,
        client_rect=(7, 11, 647, 371),
        screen_points=((-2000, -500), (-1359, -139)),
        monitor_rect=(-2560, -1440, 0, 0),
        monitor_name=r"\\.\DISPLAY2",
    ):
        self.fail_at = fail_at
        self.windows = tuple(windows)
        self.process_path = process_path
        self.created_low = created_low
        self.created_high = created_high
        self.visible = visible
        self.minimized = minimized
        self.cloaked = cloaked
        self.client_rect = client_rect
        self.screen_points = tuple(screen_points)
        self.monitor_rect = monitor_rect
        self.monitor_name = monitor_name
        self.pid = pid
        self.process_handle = 0x12345678ABCDEF01
        self.monitor_handle = 0x123456780000005B
        self.previous_context = ctypes.c_void_p(0x123456789ABC)
        self.calls = []
        self.client_to_screen_inputs = []
        self.dwm_output_type = None
        self.monitor_info_size = None
        self._dpi_calls = 0

    def SetThreadDpiAwarenessContext(self, context):
        self._dpi_calls += 1
        self.calls.append(("SetThreadDpiAwarenessContext", context))
        if self._dpi_calls == 1:
            if self.fail_at == "dpi_enter_return":
                return None
            if self.fail_at == "dpi_enter_raise":
                raise RuntimeError("dpi enter failed")
            return self.previous_context
        if self.fail_at == "dpi_restore_return":
            return None
        if self.fail_at == "dpi_restore_raise":
            raise RuntimeError("dpi restore failed")
        return ctypes.c_void_p(1)

    def EnumWindows(self, callback, lparam):
        self.calls.append(("EnumWindows", lparam))
        if self.fail_at == "enum_return":
            return 0
        if self.fail_at == "enum_raise":
            raise RuntimeError("enum failed")
        for hwnd in self.windows:
            assert callback(hwnd, lparam)
        return 1

    def IsWindowVisible(self, hwnd):
        self.calls.append(("IsWindowVisible", hwnd))
        return int(self.visible)

    def IsIconic(self, hwnd):
        self.calls.append(("IsIconic", hwnd))
        return int(self.minimized)

    def GetWindowThreadProcessId(self, hwnd, pid_out):
        self.calls.append(("GetWindowThreadProcessId", hwnd))
        if self.fail_at == "pid_return":
            return 0
        if self.fail_at == "pid_raise":
            raise RuntimeError("pid failed")
        pid_out._obj.value = 0 if self.fail_at == "pid_zero" else self.pid
        return 77

    def OpenProcess(self, access, inherit_handle, pid):
        self.calls.append(("OpenProcess", access, inherit_handle, pid))
        if self.fail_at == "open_return":
            return None
        if self.fail_at == "open_raise":
            raise RuntimeError("open failed")
        return self.process_handle

    def QueryFullProcessImageNameW(self, handle, flags, buffer, size_out):
        self.calls.append(("QueryFullProcessImageNameW", handle, flags))
        buffer.value = self.process_path
        size_out._obj.value = len(self.process_path)
        if self.fail_at == "path_return":
            return 0
        if self.fail_at == "path_raise":
            raise RuntimeError("path failed")
        return 1

    def GetProcessTimes(
        self,
        handle,
        creation_out,
        exit_out,
        kernel_out,
        user_out,
    ):
        self.calls.append(("GetProcessTimes", handle))
        if self.fail_at == "times_return":
            return 0
        if self.fail_at == "times_raise":
            raise RuntimeError("times failed")
        creation_out._obj.dwLowDateTime = self.created_low
        creation_out._obj.dwHighDateTime = self.created_high
        return 1

    def CloseHandle(self, handle):
        self.calls.append(("CloseHandle", handle))
        if self.fail_at == "close_return":
            return 0
        if self.fail_at == "close_raise":
            raise RuntimeError("close failed")
        return 1

    def DwmGetWindowAttribute(self, hwnd, attribute, output, output_size):
        self.calls.append(
            ("DwmGetWindowAttribute", hwnd, attribute, output_size)
        )
        self.dwm_output_type = type(output._obj)
        if self.fail_at == "dwm_hresult":
            return 1
        if self.fail_at == "dwm_raise":
            raise RuntimeError("dwm failed")
        output._obj.value = self.cloaked
        return 0

    def GetClientRect(self, hwnd, rect_out):
        self.calls.append(("GetClientRect", hwnd))
        if self.fail_at == "client_rect_return":
            return 0
        if self.fail_at == "client_rect_raise":
            raise RuntimeError("client rect failed")
        rect = rect_out._obj
        rect.left, rect.top, rect.right, rect.bottom = self.client_rect
        return 1

    def ClientToScreen(self, hwnd, point_in_out):
        point = point_in_out._obj
        call_index = len(self.client_to_screen_inputs)
        self.client_to_screen_inputs.append((point.x, point.y))
        self.calls.append(("ClientToScreen", hwnd, point.x, point.y))
        corner_index = call_index % 2
        if self.fail_at == f"client_point_{corner_index + 1}_return":
            return 0
        if self.fail_at == f"client_point_{corner_index + 1}_raise":
            raise RuntimeError("client point failed")
        point.x, point.y = self.screen_points[corner_index]
        return 1

    def MonitorFromWindow(self, hwnd, flags):
        self.calls.append(("MonitorFromWindow", hwnd, flags))
        if self.fail_at == "monitor_return":
            return None
        if self.fail_at == "monitor_raise":
            raise RuntimeError("monitor failed")
        return self.monitor_handle

    def GetMonitorInfoW(self, monitor_handle, info_out):
        self.calls.append(("GetMonitorInfoW", monitor_handle))
        if self.fail_at == "monitor_info_return":
            return 0
        if self.fail_at == "monitor_info_raise":
            raise RuntimeError("monitor info failed")
        info = info_out._obj
        self.monitor_info_size = info.cbSize
        (
            info.rcMonitor.left,
            info.rcMonitor.top,
            info.rcMonitor.right,
            info.rcMonitor.bottom,
        ) = self.monitor_rect
        info.rcWork.left = 100
        info.rcWork.top = 200
        info.rcWork.right = 300
        info.rcWork.bottom = 400
        info.szDevice = self.monitor_name
        return 1


def _raw_calls(raw, name):
    return [call for call in raw.calls if call[0] == name]


class _NativeFunctionStub:
    def __init__(self, name):
        self.name = name
        self.argtypes = None
        self.restype = None

    def __call__(self, *args, **kwargs):
        raise AssertionError(f"native function {self.name} must not be called")


class _NativeDllStub:
    def __init__(self):
        self.requested = []
        self.functions = {}

    def __getattr__(self, name):
        self.requested.append(name)
        function = _NativeFunctionStub(name)
        self.functions[name] = function
        return function


def test_raw_api_binds_only_the_reviewed_functions_with_pointer_width_signatures():
    user32 = _NativeDllStub()
    kernel32 = _NativeDllStub()
    dwmapi = _NativeDllStub()

    raw = window_module._RawWin32Api(user32, kernel32, dwmapi)

    assert user32.requested == [
        "SetThreadDpiAwarenessContext",
        "EnumWindows",
        "IsWindowVisible",
        "IsIconic",
        "GetWindowThreadProcessId",
        "GetClientRect",
        "ClientToScreen",
        "MonitorFromWindow",
        "GetMonitorInfoW",
    ]
    assert kernel32.requested == [
        "OpenProcess",
        "QueryFullProcessImageNameW",
        "GetProcessTimes",
        "CloseHandle",
    ]
    assert dwmapi.requested == ["DwmGetWindowAttribute"]
    assert raw.SetThreadDpiAwarenessContext.argtypes == [ctypes.c_void_p]
    assert raw.SetThreadDpiAwarenessContext.restype is ctypes.c_void_p
    assert raw.window_enum_callback_type._argtypes_ == (
        ctypes.c_void_p,
        ctypes.c_ssize_t,
    )
    assert raw.window_enum_callback_type._restype_ is ctypes.c_int32
    assert raw.EnumWindows.argtypes == [
        raw.window_enum_callback_type,
        ctypes.c_ssize_t,
    ]
    assert raw.IsWindowVisible.argtypes == [ctypes.c_void_p]
    assert raw.IsIconic.argtypes == [ctypes.c_void_p]
    assert raw.GetWindowThreadProcessId.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    assert raw.GetClientRect.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(window_module._RECT),
    ]
    assert raw.ClientToScreen.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(window_module._POINT),
    ]
    assert raw.MonitorFromWindow.argtypes == [
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    assert raw.GetMonitorInfoW.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(window_module._MONITORINFOEXW),
    ]
    assert raw.OpenProcess.argtypes == [
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_uint32,
    ]
    assert raw.OpenProcess.restype is ctypes.c_void_p
    assert raw.QueryFullProcessImageNameW.argtypes == [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    assert raw.GetProcessTimes.argtypes == [
        ctypes.c_void_p,
        ctypes.POINTER(window_module._FILETIME),
        ctypes.POINTER(window_module._FILETIME),
        ctypes.POINTER(window_module._FILETIME),
        ctypes.POINTER(window_module._FILETIME),
    ]
    assert raw.CloseHandle.argtypes == [ctypes.c_void_p]
    assert raw.MonitorFromWindow.restype is ctypes.c_void_p
    assert raw.DwmGetWindowAttribute.argtypes == [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    assert raw.DwmGetWindowAttribute.restype is ctypes.c_int32
    assert ctypes.sizeof(window_module._FILETIME) == 8
    assert ctypes.sizeof(window_module._POINT) == 8
    assert ctypes.sizeof(window_module._RECT) == 16


def test_process_query_uses_exactly_process_query_limited_information():
    raw = _FakeRawWin32Api()
    facade = _CtypesWin32Facade(raw)

    identity, executable = facade.read_process_identity(101)

    assert identity.pid == raw.pid
    assert executable == "StreetFighter6.exe"
    assert _raw_calls(raw, "OpenProcess") == [
        ("OpenProcess", 0x1000, False, raw.pid)
    ]
    assert _raw_calls(raw, "QueryFullProcessImageNameW") == [
        ("QueryFullProcessImageNameW", raw.process_handle, 0)
    ]
    assert _raw_calls(raw, "CloseHandle") == [
        ("CloseHandle", raw.process_handle)
    ]
    assert [call[0] for call in raw.calls] == [
        "GetWindowThreadProcessId",
        "OpenProcess",
        "QueryFullProcessImageNameW",
        "GetProcessTimes",
        "CloseHandle",
    ]


@pytest.mark.parametrize(
    "stage",
    ["path_return", "path_raise", "times_return", "times_raise"],
)
def test_process_query_always_closes_handle_after_every_failure_stage(stage):
    raw = _FakeRawWin32Api(fail_at=stage)
    facade = _CtypesWin32Facade(raw)

    with pytest.raises(GameWindowError):
        facade.read_process_identity(101)

    assert _raw_calls(raw, "CloseHandle") == [
        ("CloseHandle", raw.process_handle)
    ]


@pytest.mark.parametrize("stage", ["close_return", "close_raise"])
def test_process_query_calls_close_exactly_once_when_close_itself_fails(stage):
    raw = _FakeRawWin32Api(fail_at=stage)
    facade = _CtypesWin32Facade(raw)

    with pytest.raises(GameWindowError):
        facade.read_process_identity(101)

    assert _raw_calls(raw, "CloseHandle") == [
        ("CloseHandle", raw.process_handle)
    ]


@pytest.mark.parametrize(
    "stage",
    ["pid_return", "pid_raise", "pid_zero", "open_return", "open_raise"],
)
def test_process_query_does_not_close_a_handle_that_was_never_opened(stage):
    raw = _FakeRawWin32Api(fail_at=stage)
    facade = _CtypesWin32Facade(raw)

    with pytest.raises(GameWindowError):
        facade.read_process_identity(101)

    assert _raw_calls(raw, "CloseHandle") == []


def test_process_creation_filetime_combines_high_and_low_words_into_64_bit_ticks():
    raw = _FakeRawWin32Api(created_high=0xFEDCBA98, created_low=0x76543210)

    identity, _ = _CtypesWin32Facade(raw).read_process_identity(101)

    assert identity.process_created_at_ticks == (0xFEDCBA98 << 32) | 0x76543210
    assert isinstance(identity.process_created_at_ticks, int)


def test_full_process_path_is_reduced_to_basename_and_not_retained():
    secret_path = r"Z:\Users\private-user\hidden\StreetFighter6.exe"
    raw = _FakeRawWin32Api(process_path=secret_path)
    facade = _CtypesWin32Facade(raw)

    _, executable = facade.read_process_identity(101)

    assert executable == "StreetFighter6.exe"
    assert secret_path not in executable
    facade_owned_state = {
        name: value for name, value in vars(facade).items() if name != "_api"
    }
    assert secret_path not in repr(facade_owned_state)

    failing_raw = _FakeRawWin32Api(process_path=secret_path, fail_at="times_return")
    with pytest.raises(GameWindowError) as caught:
        _CtypesWin32Facade(failing_raw).read_process_identity(101)
    assert secret_path not in str(caught.value)


@pytest.mark.parametrize(
    "stage",
    [
        "path_return",
        "path_raise",
        "times_return",
        "times_raise",
        "close_return",
        "close_raise",
    ],
)
def test_full_process_path_is_scrubbed_before_later_failure_traceback(stage):
    secret_path = r"Z:\Users\private-user\hidden\StreetFighter6.exe"
    raw = _FakeRawWin32Api(process_path=secret_path, fail_at=stage)

    with pytest.raises(GameWindowError) as caught:
        _CtypesWin32Facade(raw).read_process_identity(101)

    fragment = "private-user"
    pending = [caught.value]
    seen = set()
    saw_identity_frame = False
    boundary_file = Path(window_module.__file__).resolve()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        traceback = error.__traceback__
        while traceback is not None:
            frame_locals = traceback.tb_frame.f_locals
            frame_file = Path(traceback.tb_frame.f_code.co_filename).resolve()
            if frame_file != boundary_file:
                traceback = traceback.tb_next
                continue
            if traceback.tb_frame.f_code.co_name == "read_process_identity":
                saw_identity_frame = True
            assert "queried_path" not in frame_locals
            assert "buffer" not in frame_locals
            assert fragment not in repr(frame_locals)
            for value in frame_locals.values():
                if isinstance(value, ctypes.Array) and getattr(
                    value,
                    "_type_",
                    None,
                ) is ctypes.c_wchar:
                    assert fragment not in value.value
            traceback = traceback.tb_next
        if error.__cause__ is not None:
            pending.append(error.__cause__)
        if error.__context__ is not None:
            pending.append(error.__context__)
    assert saw_identity_frame
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("invalid_path", ["", "C:\\Games\\"])
def test_invalid_process_path_still_closes_the_open_handle(invalid_path):
    raw = _FakeRawWin32Api(process_path=invalid_path)

    with pytest.raises(GameWindowError):
        _CtypesWin32Facade(raw).read_process_identity(101)

    assert _raw_calls(raw, "CloseHandle") == [
        ("CloseHandle", raw.process_handle)
    ]


def test_window_module_contains_no_vm_read_or_read_process_memory_path():
    source = Path(window_module.__file__).read_text(encoding="utf-8")

    assert "PROCESS_VM_READ" not in source
    assert "ReadProcessMemory" not in source
    assert "GetWindowText" not in source


def test_thread_dpi_context_calls_set_thread_with_pointer_sized_negative_four():
    raw = _FakeRawWin32Api()
    facade = _CtypesWin32Facade(raw)

    with facade.per_monitor_v2():
        pass

    calls = _raw_calls(raw, "SetThreadDpiAwarenessContext")
    requested = calls[0][1]
    assert type(requested) is ctypes.c_void_p
    assert ctypes.c_ssize_t(requested.value).value == -4
    assert ctypes.sizeof(requested) == ctypes.sizeof(ctypes.c_void_p)


def test_thread_dpi_context_restores_exact_returned_pointer_in_finally():
    raw = _FakeRawWin32Api()
    facade = _CtypesWin32Facade(raw)

    with pytest.raises(RuntimeError, match="body failed"):
        with facade.per_monitor_v2():
            raise RuntimeError("body failed")

    calls = _raw_calls(raw, "SetThreadDpiAwarenessContext")
    assert len(calls) == 2
    assert calls[1][1] is raw.previous_context


@pytest.mark.parametrize("stage", ["dpi_enter_return", "dpi_enter_raise"])
def test_thread_dpi_context_zero_or_exception_on_entry_fails_closed(stage):
    raw = _FakeRawWin32Api(fail_at=stage)

    with pytest.raises(GameWindowError):
        with _CtypesWin32Facade(raw).per_monitor_v2():
            raise AssertionError("must not enter")

    assert len(_raw_calls(raw, "SetThreadDpiAwarenessContext")) == 1


@pytest.mark.parametrize("stage", ["dpi_restore_return", "dpi_restore_raise"])
def test_thread_dpi_context_zero_or_exception_on_restore_fails_closed(stage):
    raw = _FakeRawWin32Api(fail_at=stage)

    with pytest.raises(GameWindowError):
        with _CtypesWin32Facade(raw).per_monitor_v2():
            pass

    assert len(_raw_calls(raw, "SetThreadDpiAwarenessContext")) == 2


def test_window_module_uses_no_process_wide_or_per_monitor_v1_dpi_path():
    source = Path(window_module.__file__).read_text(encoding="utf-8")

    assert "SetProcessDpiAwarenessContext" not in source
    assert "PER_MONITOR_AWARE = -3" not in source
    assert "c_void_p(-3)" not in source


def test_dpi_context_restores_previous_value_when_client_query_raises():
    raw = _FakeRawWin32Api(fail_at="client_rect_raise")
    facade = _CtypesWin32Facade(raw)

    with pytest.raises(GameWindowError):
        with facade.per_monitor_v2():
            facade.client_geometry(101)

    calls = _raw_calls(raw, "SetThreadDpiAwarenessContext")
    assert len(calls) == 2
    assert calls[1][1] is raw.previous_context


@pytest.mark.parametrize("cloaked, expected", [(0, False), (7, True)])
def test_dwm_cloaked_query_uses_typed_dwm_attribute_and_owned_dword(
    cloaked,
    expected,
):
    raw = _FakeRawWin32Api(cloaked=cloaked)

    assert _CtypesWin32Facade(raw).is_window_cloaked(101) is expected
    assert _raw_calls(raw, "DwmGetWindowAttribute") == [
        ("DwmGetWindowAttribute", 101, 14, ctypes.sizeof(ctypes.c_uint32))
    ]
    assert raw.dwm_output_type is ctypes.c_uint32


@pytest.mark.parametrize("stage", ["dwm_hresult", "dwm_negative_hresult", "dwm_raise"])
def test_dwm_cloaked_query_nonzero_hresult_or_exception_fails_closed(stage):
    raw = _FakeRawWin32Api(fail_at=stage)
    if stage == "dwm_negative_hresult":
        raw.fail_at = None

        def negative_hresult(hwnd, attribute, output, output_size):
            raw.calls.append(
                ("DwmGetWindowAttribute", hwnd, attribute, output_size)
            )
            return -1

        raw.DwmGetWindowAttribute = negative_hresult

    with pytest.raises(GameWindowError):
        _CtypesWin32Facade(raw).is_window_cloaked(101)


def test_raw_client_geometry_converts_both_corners_and_accepts_negative_origins():
    raw = _FakeRawWin32Api()
    facade = _CtypesWin32Facade(raw)

    geometry = facade.client_geometry(101)

    assert raw.client_to_screen_inputs == [(7, 11), (647, 371)]
    assert geometry == ClientGeometry(-2000, -500, 641, 361)
    assert _raw_calls(raw, "ClientToScreen") == [
        ("ClientToScreen", 101, 7, 11),
        ("ClientToScreen", 101, 647, 371),
    ]


@pytest.mark.parametrize(
    "stage",
    [
        "client_rect_return",
        "client_rect_raise",
        "client_point_1_return",
        "client_point_1_raise",
        "client_point_2_return",
        "client_point_2_raise",
    ],
)
def test_raw_client_geometry_fails_closed_at_each_native_stage(stage):
    raw = _FakeRawWin32Api(fail_at=stage)

    with pytest.raises(GameWindowError):
        _CtypesWin32Facade(raw).client_geometry(101)


def test_raw_monitor_descriptor_uses_monitor_info_and_complete_containment():
    raw = _FakeRawWin32Api()
    facade = _CtypesWin32Facade(raw)
    geometry = facade.client_geometry(101)

    monitor = facade.monitor_descriptor(101, geometry)
    target = Sf6WindowTarget(
        WindowIdentity(101, raw.pid, (raw.created_high << 32) | raw.created_low),
        geometry,
        monitor,
    )

    assert target.geometry == ClientGeometry(-2000, -500, 641, 361)
    assert monitor == MonitorDescriptor(
        raw.monitor_handle,
        raw.monitor_name,
        ClientGeometry(-2560, -1440, 2560, 1440),
    )
    assert _raw_calls(raw, "MonitorFromWindow") == [
        ("MonitorFromWindow", 101, 0)
    ]
    assert raw.monitor_info_size == ctypes.sizeof(window_module._MONITORINFOEXW)


@pytest.mark.parametrize(
    "stage",
    ["monitor_return", "monitor_raise", "monitor_info_return", "monitor_info_raise"],
)
def test_raw_monitor_descriptor_fails_closed_at_each_native_stage(stage):
    raw = _FakeRawWin32Api(fail_at=stage)
    facade = _CtypesWin32Facade(raw)

    with pytest.raises(GameWindowError):
        facade.monitor_descriptor(101, ClientGeometry(-2000, -500, 641, 361))


def test_raw_facade_enumerates_top_level_windows_and_maps_visibility_and_iconic():
    raw = _FakeRawWin32Api(windows=(101, 202), visible=False, minimized=True)
    facade = _CtypesWin32Facade(raw)

    assert facade.enum_top_level_windows() == (101, 202)
    assert facade.is_window_visible(101) is False
    assert facade.is_window_minimized(101) is True
    assert _raw_calls(raw, "EnumWindows") == [("EnumWindows", 0)]


def test_raw_facade_preserves_pointer_width_hwnd_and_unsigned_dword_pid():
    huge_hwnd = 0x1234567800000065
    high_pid = 0xFEDCBA98
    raw = _FakeRawWin32Api(windows=(huge_hwnd,), pid=high_pid)
    facade = _CtypesWin32Facade(raw)

    assert facade.enum_top_level_windows() == (huge_hwnd,)
    identity, _ = facade.read_process_identity(huge_hwnd)

    assert identity.hwnd == huge_hwnd
    assert identity.pid == high_pid
    assert _raw_calls(raw, "GetWindowThreadProcessId") == [
        ("GetWindowThreadProcessId", huge_hwnd)
    ]
    assert _raw_calls(raw, "OpenProcess") == [
        ("OpenProcess", 0x1000, False, high_pid)
    ]


@pytest.mark.parametrize("stage", ["enum_return", "enum_raise"])
def test_raw_window_enumeration_failure_is_typed(stage):
    with pytest.raises(GameWindowError):
        _CtypesWin32Facade(_FakeRawWin32Api(fail_at=stage)).enum_top_level_windows()


def test_raw_finder_rejects_client_crossing_monitor_boundary():
    raw = _FakeRawWin32Api(
        screen_points=((-2700, -500), (-1359, -139)),
    )

    with pytest.raises(GameWindowError):
        find_sf6_window_target(win32=_CtypesWin32Facade(raw))


def test_default_facade_loads_raw_api_only_when_uninjected_finder_is_called(
    monkeypatch,
):
    raw = _FakeRawWin32Api()
    calls = []

    def load_raw():
        calls.append("load")
        return raw

    monkeypatch.setattr(window_module, "_load_raw_win32_api", load_raw)
    assert calls == []

    target = window_module.find_sf6_window_target()

    assert target.identity == WindowIdentity(
        101,
        raw.pid,
        (raw.created_high << 32) | raw.created_low,
    )
    assert target.geometry == ClientGeometry(-2000, -500, 641, 361)
    assert calls == ["load"]


def test_importing_window_does_not_load_any_native_dll():
    script = """
import ctypes
import sys

sys.path.insert(0, 'src')

def explode(*args, **kwargs):
    raise AssertionError('native loading or callback construction happened during import')

ctypes.WinDLL = explode
ctypes.WINFUNCTYPE = explode
import sf6_music_trigger.window
print('import-safe')
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "import-safe"


class _FakeDxcamRawOutput:
    def __init__(
        self,
        *,
        monitor_handle=71,
        device_name="DISPLAY-LEFT",
        rectangle=(-1920, -100, 0, 980),
        rotation_angle=0,
        attached_to_desktop=True,
        update_error=None,
    ):
        self.hmonitor = SimpleNamespace(value=monitor_handle)
        self.devicename = device_name
        left, top, right, bottom = rectangle
        self.desc = SimpleNamespace(
            DesktopCoordinates=SimpleNamespace(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )
        )
        self.rotation_angle = rotation_angle
        self.attached_to_desktop = attached_to_desktop
        self.update_error = update_error
        self.update_calls = 0

    def update_desc(self):
        self.update_calls += 1
        if self.update_error is not None:
            raise self.update_error


class _FakeDxcamRawFactory:
    def __init__(self, outputs):
        self.outputs = outputs
        self._camera_instances = {}


class _FakeDxcamModule:
    def __init__(self, outputs):
        self.factory = _FakeDxcamRawFactory(outputs)
        self.factory_calls = 0
        self.create_calls = []
        self.output_info_calls = 0

    def DXFactory(self):
        self.factory_calls += 1
        return self.factory

    def output_info(self):
        self.output_info_calls += 1
        raise AssertionError("adapter parsed output_info")

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        raise AssertionError("create is not configured")


def _dxcam_adapter(module):
    return capture_module._Dxcam03FactoryAdapter(module)


def test_loader_wraps_fake_dxcam_module_in_dedicated_03_adapter(monkeypatch):
    module = _FakeDxcamModule([[_FakeDxcamRawOutput()]])
    calls = []

    def import_module(name):
        calls.append(("import", name))
        return module

    def version(name):
        calls.append(("version", name))
        return "0.3.0"

    monkeypatch.setattr(capture_module.importlib, "import_module", import_module)
    monkeypatch.setattr(capture_module.metadata, "version", version)

    adapter = load_dxcam_factory()

    assert type(adapter).__name__ == "_Dxcam03FactoryAdapter"
    assert adapter is not module
    assert adapter.outputs() == (
        DxcamOutput(
            71,
            "DISPLAY-LEFT",
            0,
            0,
            ClientGeometry(-1920, -100, 1920, 1080),
            0,
        ),
    )
    assert calls == [("version", "dxcam"), ("import", "dxcam")]


@pytest.mark.parametrize(
    "version",
    ["0.2.9", "0.4.0", "0.30", "0.31", "0.3foo", "1", "not-a-version"],
)
def test_loader_rejects_versions_outside_pinned_03_adapter(monkeypatch, version):
    module = _FakeDxcamModule([[_FakeDxcamRawOutput()]])
    monkeypatch.setattr(capture_module.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(capture_module.metadata, "version", lambda name: version)

    with pytest.raises(capture_module._DxcamAdapterError):
        load_dxcam_factory()


def test_03_adapter_reads_only_structured_dxfactory_outputs():
    first = _FakeDxcamRawOutput(monitor_handle=71)
    second = _FakeDxcamRawOutput(
        monitor_handle=72,
        device_name="DISPLAY-RIGHT",
        rectangle=(0, -100, 2560, 1340),
        rotation_angle=90,
    )
    module = _FakeDxcamModule([[first], [second]])
    adapter = _dxcam_adapter(module)

    first_snapshot = adapter.outputs()
    second_snapshot = adapter.outputs()

    assert first_snapshot == second_snapshot
    assert [item.device_index for item in first_snapshot] == [0, 1]
    assert [item.output_index for item in first_snapshot] == [0, 0]
    assert first.update_calls == 2
    assert second.update_calls == 2
    # dxcam 0.3 exposes DXFactory as a process singleton.  Reconstructing it
    # on each frame refresh emits its singleton warning continuously.
    assert module.factory_calls == 1
    assert module.output_info_calls == 0


@pytest.mark.parametrize(
    "outputs",
    [
        ((_FakeDxcamRawOutput(),),),
        [(_FakeDxcamRawOutput(),)],
    ],
)
def test_03_adapter_rejects_tuple_output_collections_as_03_api_drift(outputs):
    with pytest.raises(capture_module._DxcamAdapterError):
        _dxcam_adapter(_FakeDxcamModule(outputs)).outputs()


def test_03_adapter_updates_each_output_and_maps_hmonitor_name_rectangle_rotation_and_indices():
    outputs = [
        [
            _FakeDxcamRawOutput(
                monitor_handle=80,
                device_name="DISPLAY-A",
                rectangle=(-3000, -400, -1000, 800),
                rotation_angle=270,
            ),
            _FakeDxcamRawOutput(
                monitor_handle=81,
                device_name="DISPLAY-B",
                rectangle=(-1000, -400, 1000, 800),
                rotation_angle=180,
            ),
        ],
        [
            _FakeDxcamRawOutput(
                monitor_handle=82,
                device_name="DISPLAY-C",
                rectangle=(1000, -400, 3000, 800),
                rotation_angle=90,
            )
        ],
    ]

    snapshot = _dxcam_adapter(_FakeDxcamModule(outputs)).outputs()

    assert snapshot == (
        DxcamOutput(80, "DISPLAY-A", 0, 0, ClientGeometry(-3000, -400, 2000, 1200), 270),
        DxcamOutput(81, "DISPLAY-B", 0, 1, ClientGeometry(-1000, -400, 2000, 1200), 180),
        DxcamOutput(82, "DISPLAY-C", 1, 0, ClientGeometry(1000, -400, 2000, 1200), 90),
    )
    assert [raw.update_calls for group in outputs for raw in group] == [1, 1, 1]


def test_03_adapter_refreshes_descriptor_before_reading_any_output_field():
    class RefreshRequiredOutput:
        def __init__(self):
            self.ready = False
            self.update_calls = 0

        def update_desc(self):
            self.update_calls += 1
            self.ready = True

        def _require_refresh(self):
            if not self.ready:
                raise AssertionError("descriptor field read before update_desc")

        @property
        def hmonitor(self):
            self._require_refresh()
            return SimpleNamespace(value=71)

        @property
        def devicename(self):
            self._require_refresh()
            return "DISPLAY-LEFT"

        @property
        def desc(self):
            self._require_refresh()
            return SimpleNamespace(
                DesktopCoordinates=SimpleNamespace(
                    left=-1920,
                    top=-100,
                    right=0,
                    bottom=980,
                )
            )

        @property
        def rotation_angle(self):
            self._require_refresh()
            return 0

        @property
        def attached_to_desktop(self):
            self._require_refresh()
            return True

    raw = RefreshRequiredOutput()

    assert _dxcam_adapter(_FakeDxcamModule([[raw]])).outputs() == (
        DxcamOutput(
            71,
            "DISPLAY-LEFT",
            0,
            0,
            ClientGeometry(-1920, -100, 1920, 1080),
            0,
        ),
    )
    assert raw.update_calls == 1


def test_03_adapter_skips_only_attached_to_desktop_false_outputs():
    attached = _FakeDxcamRawOutput(monitor_handle=71, attached_to_desktop=True)
    detached = _FakeDxcamRawOutput(
        monitor_handle=72,
        device_name="DISPLAY-DISCONNECTED",
        rectangle=(0, 0, 640, 480),
        attached_to_desktop=False,
    )
    module = _FakeDxcamModule([[detached, attached]])

    assert _dxcam_adapter(module).outputs() == (
        DxcamOutput(
            71,
            "DISPLAY-LEFT",
            0,
            1,
            ClientGeometry(-1920, -100, 1920, 1080),
            0,
        ),
    )
    assert detached.update_calls == attached.update_calls == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("monitor_handle", True),
        ("monitor_handle", _IntSubclass(71)),
        ("device_name", " "),
        ("rectangle", (0, 0, 0, 100)),
        ("rectangle", (0, False, 100, 100)),
        ("rectangle", (0, 0, _IntSubclass(100), 100)),
        ("rotation_angle", True),
        ("rotation_angle", _IntSubclass(90)),
        ("rotation_angle", 45),
        ("attached_to_desktop", 1),
        ("attached_to_desktop", 0),
    ],
)
@pytest.mark.parametrize("attached", [True, False])
def test_03_adapter_rejects_truthy_and_malformed_output_fields_before_detached_skip(
    field,
    value,
    attached,
):
    kwargs = {"attached_to_desktop": attached}
    if field == "rectangle":
        kwargs[field] = value
    elif field == "monitor_handle":
        raw = _FakeDxcamRawOutput(**kwargs)
        raw.hmonitor = SimpleNamespace(value=value)
        module = _FakeDxcamModule([[raw]])
        with pytest.raises(capture_module._DxcamAdapterError):
            _dxcam_adapter(module).outputs()
        return
    else:
        kwargs[field] = value
    module = _FakeDxcamModule([[_FakeDxcamRawOutput(**kwargs)]])

    with pytest.raises(capture_module._DxcamAdapterError):
        _dxcam_adapter(module).outputs()


def test_03_adapter_api_drift_and_update_failures_raise_private_adapter_error():
    malformed = _FakeDxcamRawOutput()
    del malformed.desc
    broken_update = _FakeDxcamRawOutput(update_error=RuntimeError("update failed"))

    for raw in (malformed, broken_update):
        with pytest.raises(capture_module._DxcamAdapterError):
            _dxcam_adapter(_FakeDxcamModule([[raw]])).outputs()


@pytest.mark.parametrize(
    "attribute_path",
    [
        "hmonitor",
        "devicename",
        "desc",
        "rotation_angle",
        "attached_to_desktop",
        "desc.DesktopCoordinates",
        "desc.DesktopCoordinates.left",
        "desc.DesktopCoordinates.top",
        "desc.DesktopCoordinates.right",
        "desc.DesktopCoordinates.bottom",
    ],
)
def test_03_adapter_rejects_each_missing_structured_output_field(attribute_path):
    raw = _FakeDxcamRawOutput()
    owner = raw
    components = attribute_path.split(".")
    for component in components[:-1]:
        owner = getattr(owner, component)
    delattr(owner, components[-1])

    with pytest.raises(capture_module._DxcamAdapterError):
        _dxcam_adapter(_FakeDxcamModule([[raw]])).outputs()


def test_03_adapter_never_calls_or_parses_output_info():
    class NoDisplayStringOutput(_FakeDxcamRawOutput):
        def __str__(self):
            raise AssertionError("adapter parsed an output display string")

        def __repr__(self):
            raise AssertionError("adapter parsed an output display representation")

    module = _FakeDxcamModule([[NoDisplayStringOutput()]])

    assert len(_dxcam_adapter(module).outputs()) == 1
    assert module.output_info_calls == 0


class _FakeDxcamCamera:
    def __init__(
        self,
        *,
        is_released=False,
        is_capturing=False,
        start_error=None,
        stop_error=None,
        release_error=None,
        stop_callback=None,
        events=None,
        name="camera",
    ):
        self.is_released = is_released
        self.is_capturing = is_capturing
        self.start_error = start_error
        self.stop_error = stop_error
        self.release_error = release_error
        self.stop_callback = stop_callback
        self.events = events
        self.name = name
        self.start_calls = []
        self.stop_calls = 0
        self.release_calls = 0
        self.read_calls = []

    def start(self, **kwargs):
        if self.events is not None:
            self.events.append((f"{self.name}.start", kwargs))
        self.start_calls.append(kwargs)
        if self.start_error is not None:
            raise self.start_error
        self.is_capturing = True

    def get_latest_frame(self, **kwargs):
        self.read_calls.append(kwargs)
        return None

    def stop(self):
        if self.events is not None:
            self.events.append((f"{self.name}.stop",))
        self.stop_calls += 1
        if self.stop_callback is not None:
            self.stop_callback()
        if self.stop_error is not None:
            raise self.stop_error
        self.is_capturing = False

    def release(self):
        if self.events is not None:
            self.events.append((f"{self.name}.release",))
        self.release_calls += 1
        self.is_released = True
        if self.release_error is not None:
            raise self.release_error


class _CreatingDxcamModule(_FakeDxcamModule):
    def __init__(self, outputs, effects):
        super().__init__(outputs)
        self.effects = list(effects)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if not self.effects:
            raise AssertionError("unexpected create")
        effect = self.effects.pop(0)
        key = (
            kwargs["device_idx"],
            kwargs["output_idx"],
            kwargs["backend"].casefold(),
        )
        if isinstance(effect, BaseException):
            raise effect
        if isinstance(effect, tuple) and effect[0] == "raise_registered":
            camera, error = effect[1], effect[2]
            self.factory._camera_instances[key] = camera
            raise error
        if isinstance(effect, tuple) and effect[0] == "mismatch":
            returned, registered = effect[1], effect[2]
            self.factory._camera_instances[key] = registered
            return returned
        camera = effect
        self.factory._camera_instances[key] = camera
        return camera


def _create_adapter_camera(adapter, *, backend="dxgi"):
    return adapter.create(
        device_idx=2,
        output_idx=3,
        region=(10, 20, 650, 380),
        output_color="BGR",
        max_buffer_len=2,
        backend=backend,
        processor_backend="cv2",
    )


def test_03_adapter_create_delegates_exact_keywords_to_module_create():
    camera = _FakeDxcamCamera()
    module = _CreatingDxcamModule([[_FakeDxcamRawOutput()]], [camera])
    adapter = _dxcam_adapter(module)

    assert _create_adapter_camera(adapter, backend="DXGI") is camera
    assert module.create_calls == [
        {
            "device_idx": 2,
            "output_idx": 3,
            "region": (10, 20, 650, 380),
            "output_color": "BGR",
            "max_buffer_len": 2,
            "backend": "dxgi",
            "processor_backend": "cv2",
        }
    ]
    assert module.factory._camera_instances[(2, 3, "dxgi")] is camera
    assert adapter._cleanup_owned_camera(camera) is True


def test_03_adapter_reuses_one_dxfactory_for_output_refresh_and_camera_create():
    camera = _FakeDxcamCamera()
    module = _CreatingDxcamModule([[_FakeDxcamRawOutput()]], [camera])
    adapter = _dxcam_adapter(module)

    assert len(adapter.outputs()) == 1
    assert _create_adapter_camera(adapter) is camera
    assert module.factory_calls == 1
    assert adapter._cleanup_owned_camera(camera) is True


def test_03_adapter_rejects_preexisting_unreleased_exact_key_without_create_or_cleanup():
    foreign = _FakeDxcamCamera(is_released=False)
    module = _CreatingDxcamModule([[_FakeDxcamRawOutput()]], [])
    module.factory._camera_instances[(2, 3, "dxgi")] = foreign

    with pytest.raises(capture_module._DxcamAdapterError):
        _create_adapter_camera(_dxcam_adapter(module))

    assert module.create_calls == []
    assert foreign.stop_calls == foreign.release_calls == 0


def test_failure_before_module_create_never_cleans_registry_replacement_as_foreign():
    foreign = _FakeDxcamCamera()
    module = _CreatingDxcamModule([[_FakeDxcamRawOutput()]], [])
    key = (2, 3, "dxgi")

    class ReplacingProbe(_FakeDxcamCamera):
        @property
        def is_released(self):
            module.factory._camera_instances[key] = foreign
            raise RuntimeError("release probe failed")

        @is_released.setter
        def is_released(self, value):
            self._stored_release = value

    probe = ReplacingProbe()
    module.factory._camera_instances[key] = probe

    with pytest.raises(capture_module._DxcamAdapterError):
        _create_adapter_camera(_dxcam_adapter(module))

    assert module.create_calls == []
    assert probe.stop_calls == probe.release_calls == 0
    assert foreign.stop_calls == foreign.release_calls == 0
    assert module.factory._camera_instances[key] is foreign


def test_03_adapter_allows_released_exact_key_replacement_and_registers_returned_identity():
    released = _FakeDxcamCamera(is_released=True)
    replacement = _FakeDxcamCamera()
    module = _CreatingDxcamModule([[_FakeDxcamRawOutput()]], [replacement])
    module.factory._camera_instances[(2, 3, "dxgi")] = released
    adapter = _dxcam_adapter(module)

    assert _create_adapter_camera(adapter) is replacement
    assert module.factory._camera_instances[(2, 3, "dxgi")] is replacement
    assert released.stop_calls == released.release_calls == 0
    assert adapter._cleanup_owned_camera(replacement) is True


def test_two_adapters_concurrently_reserve_same_key_and_only_one_calls_module_create(
    monkeypatch,
):
    first_camera = _FakeDxcamCamera()
    second_camera = _FakeDxcamCamera()

    class BlockingCoordinator(dict):
        def __init__(self):
            super().__init__()
            self.reservation_attempts = 0
            self.first_reservation_entered = threading.Event()
            self.allow_first_reservation = threading.Event()
            self.second_reservation_entered = threading.Event()

        def __setitem__(self, key, value):
            if isinstance(value, capture_module._DxcamReservation):
                self.reservation_attempts += 1
                if self.reservation_attempts == 1:
                    self.first_reservation_entered.set()
                    if not self.allow_first_reservation.wait(timeout=2):
                        raise AssertionError("first reservation was not released")
                else:
                    self.second_reservation_entered.set()
            super().__setitem__(key, value)

    coordinator = BlockingCoordinator()
    monkeypatch.setattr(capture_module, "_DXCAM_COORDINATOR", coordinator)

    class BlockingCreateModule(_FakeDxcamModule):
        def __init__(self):
            super().__init__([[_FakeDxcamRawOutput()]])
            self.effects = [first_camera, second_camera]
            self.first_create_entered = threading.Event()
            self.allow_first_create = threading.Event()
            self.second_create_entered = threading.Event()

        def create(self, **kwargs):
            self.create_calls.append(kwargs)
            ordinal = len(self.create_calls)
            if ordinal == 1:
                self.first_create_entered.set()
                if not self.allow_first_create.wait(timeout=2):
                    raise AssertionError("first create was not released")
            else:
                self.second_create_entered.set()
            camera = self.effects.pop(0)
            key = (
                kwargs["device_idx"],
                kwargs["output_idx"],
                kwargs["backend"].casefold(),
            )
            self.factory._camera_instances[key] = camera
            return camera

    module = BlockingCreateModule()
    adapters = [_dxcam_adapter(module), _dxcam_adapter(module)]
    second_attempting = threading.Event()
    results = []
    errors = []

    def worker(adapter, attempting=None):
        if attempting is not None:
            attempting.set()
        try:
            results.append((adapter, _create_adapter_camera(adapter)))
        except Exception as exc:
            errors.append(exc)

    first_thread = threading.Thread(target=worker, args=(adapters[0],), daemon=True)
    second_thread = threading.Thread(
        target=worker,
        args=(adapters[1], second_attempting),
        daemon=True,
    )
    first_thread.start()
    assert coordinator.first_reservation_entered.wait(timeout=2)
    second_thread.start()
    assert second_attempting.wait(timeout=2)
    second_reserved_while_first_was_blocked = (
        coordinator.second_reservation_entered.wait(timeout=0.5)
    )
    coordinator.allow_first_reservation.set()
    assert module.first_create_entered.wait(timeout=2)

    second_entered_while_first_was_blocked = module.second_create_entered.wait(
        timeout=0.5
    )
    module.allow_first_create.set()
    threads = [first_thread, second_thread]
    for thread in threads:
        thread.join(timeout=2)

    cleanup_results = [
        adapter._cleanup_owned_camera(camera) for adapter, camera in results
    ]
    assert all(not thread.is_alive() for thread in threads)
    assert second_reserved_while_first_was_blocked is False
    assert second_entered_while_first_was_blocked is False
    assert len(module.create_calls) == 1
    assert len(results) == 1
    assert len(errors) == 1
    assert type(errors[0]) is capture_module._DxcamAdapterError
    _, winner_camera = results[0]
    assert winner_camera in (first_camera, second_camera)
    assert cleanup_results == [True]


def test_failed_create_releases_reservation_and_second_adapter_can_retry():
    camera = _FakeDxcamCamera()
    module = _CreatingDxcamModule(
        [[_FakeDxcamRawOutput()]],
        [RuntimeError("first create failed"), camera],
    )

    with pytest.raises(capture_module._DxcamAdapterError):
        _create_adapter_camera(_dxcam_adapter(module))

    second = _dxcam_adapter(module)
    assert _create_adapter_camera(second) is camera
    assert len(module.create_calls) == 2
    assert second._cleanup_owned_camera(camera) is True


def test_dxgi_and_winrt_reservation_keys_are_independent_and_backend_is_casefolded():
    dxgi_camera = _FakeDxcamCamera()
    winrt_camera = _FakeDxcamCamera()
    module = _CreatingDxcamModule(
        [[_FakeDxcamRawOutput()]],
        [dxgi_camera, winrt_camera],
    )
    dxgi_adapter = _dxcam_adapter(module)
    winrt_adapter = _dxcam_adapter(module)

    assert _create_adapter_camera(dxgi_adapter, backend="DXGI") is dxgi_camera
    assert _create_adapter_camera(winrt_adapter, backend="winrt") is winrt_camera
    with pytest.raises(capture_module._DxcamAdapterError):
        _create_adapter_camera(_dxcam_adapter(module), backend="dxgi")

    assert [call["backend"] for call in module.create_calls] == ["dxgi", "winrt"]
    assert dxgi_adapter._cleanup_owned_camera(dxgi_camera) is True
    assert winrt_adapter._cleanup_owned_camera(winrt_camera) is True


def test_device_and_output_indices_are_independent_coordinator_key_components():
    cameras = [_FakeDxcamCamera() for _ in range(3)]
    module = _CreatingDxcamModule([[_FakeDxcamRawOutput()]], cameras)
    requests = [(2, 3), (4, 3), (2, 4)]
    owned = []

    for (device_idx, output_idx), camera in zip(requests, cameras, strict=True):
        adapter = _dxcam_adapter(module)
        created = adapter.create(
            device_idx=device_idx,
            output_idx=output_idx,
            region=(10, 20, 650, 380),
            output_color="BGR",
            max_buffer_len=2,
            backend="dxgi",
            processor_backend="cv2",
        )
        assert created is camera
        owned.append((adapter, camera))

    assert [
        (call["device_idx"], call["output_idx"])
        for call in module.create_calls
    ] == requests
    assert [adapter._cleanup_owned_camera(camera) for adapter, camera in owned] == [
        True,
        True,
        True,
    ]


def test_coordinator_uses_fake_module_identity_as_part_of_the_key():
    first_camera = _FakeDxcamCamera()
    second_camera = _FakeDxcamCamera()
    first_module = _CreatingDxcamModule([[_FakeDxcamRawOutput()]], [first_camera])
    second_module = _CreatingDxcamModule([[_FakeDxcamRawOutput()]], [second_camera])
    first = _dxcam_adapter(first_module)
    second = _dxcam_adapter(second_module)

    assert _create_adapter_camera(first) is first_camera
    assert _create_adapter_camera(second) is second_camera
    assert first._cleanup_owned_camera(first_camera) is True
    assert second._cleanup_owned_camera(second_camera) is True


def test_03_adapter_cleanup_touches_only_exact_adapter_owned_camera_and_allows_rebuild():
    first_camera = _FakeDxcamCamera()
    replacement = _FakeDxcamCamera()
    foreign = _FakeDxcamCamera()
    module = _CreatingDxcamModule(
        [[_FakeDxcamRawOutput()]],
        [first_camera, replacement],
    )
    owner = _dxcam_adapter(module)
    other = _dxcam_adapter(module)
    assert _create_adapter_camera(owner) is first_camera

    assert other._cleanup_owned_camera(first_camera) is False
    assert owner._cleanup_owned_camera(foreign) is False
    assert foreign.stop_calls == foreign.release_calls == 0
    assert owner._cleanup_owned_camera(first_camera) is True
    assert owner._cleanup_owned_camera(first_camera) is False
    assert first_camera.stop_calls == first_camera.release_calls == 1

    assert _create_adapter_camera(other) is replacement
    assert other._cleanup_owned_camera(replacement) is True


def test_03_adapter_rejects_registry_return_identity_mismatch_without_cleaning_either_camera():
    returned = _FakeDxcamCamera()
    registered = _FakeDxcamCamera()
    module = _CreatingDxcamModule(
        [[_FakeDxcamRawOutput()]],
        [("mismatch", returned, registered)],
    )

    with pytest.raises(capture_module._DxcamAdapterError):
        _create_adapter_camera(_dxcam_adapter(module))

    assert returned.stop_calls == returned.release_calls == 0
    assert registered.stop_calls == registered.release_calls == 0
    assert module.factory._camera_instances[(2, 3, "dxgi")] is registered


def test_create_exception_that_registers_new_camera_cleans_only_that_new_identity():
    registered = _FakeDxcamCamera()
    replacement = _FakeDxcamCamera()
    module = _CreatingDxcamModule(
        [[_FakeDxcamRawOutput()]],
        [
            ("raise_registered", registered, RuntimeError("late failure")),
            replacement,
        ],
    )

    with pytest.raises(capture_module._DxcamAdapterError):
        _create_adapter_camera(_dxcam_adapter(module))

    assert registered.stop_calls == registered.release_calls == 1
    second = _dxcam_adapter(module)
    assert _create_adapter_camera(second) is replacement
    assert second._cleanup_owned_camera(replacement) is True


def test_protocol_validation_stop_getter_failure_still_releases_and_allows_retry():
    class StopGetterFailureCamera(_FakeDxcamCamera):
        @property
        def stop(self):
            raise RuntimeError("stop getter failed")

    broken = StopGetterFailureCamera()
    replacement = _FakeDxcamCamera()
    module = _CreatingDxcamModule(
        [[_FakeDxcamRawOutput()]],
        [broken, replacement],
    )

    with pytest.raises(capture_module._DxcamAdapterError):
        _create_adapter_camera(_dxcam_adapter(module))

    assert broken.release_calls == 1
    second = _dxcam_adapter(module)
    assert _create_adapter_camera(second) is replacement
    assert second._cleanup_owned_camera(replacement) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_released", 0),
        ("is_released", 1),
        ("is_capturing", 0),
        ("is_capturing", 1),
    ],
)
def test_03_adapter_rejects_truthy_integer_camera_state_and_cleans_new_registered_camera(
    field,
    value,
):
    camera = _FakeDxcamCamera()
    setattr(camera, field, value)
    module = _CreatingDxcamModule([[_FakeDxcamRawOutput()]], [camera])

    with pytest.raises(capture_module._DxcamAdapterError):
        _create_adapter_camera(_dxcam_adapter(module))

    assert camera.stop_calls == camera.release_calls == 1


def test_cleanup_runs_outside_coordinator_lock_and_allows_reentry():
    first_camera = _FakeDxcamCamera()
    second_camera = _FakeDxcamCamera()
    module = _CreatingDxcamModule(
        [[_FakeDxcamRawOutput()]],
        [first_camera, second_camera],
    )
    adapter = _dxcam_adapter(module)
    assert _create_adapter_camera(adapter, backend="dxgi") is first_camera
    assert _create_adapter_camera(adapter, backend="winrt") is second_camera
    first_camera.stop_callback = lambda: adapter._cleanup_owned_camera(second_camera)
    outcome = []

    worker = threading.Thread(
        target=lambda: outcome.append(adapter._cleanup_owned_camera(first_camera)),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert outcome == [True]
    assert first_camera.stop_calls == first_camera.release_calls == 1
    assert second_camera.stop_calls == second_camera.release_calls == 1


def test_cleanup_releases_ownership_even_when_stop_raises_and_key_is_retryable():
    broken = _FakeDxcamCamera(stop_error=RuntimeError("stop failed"))
    replacement = _FakeDxcamCamera()
    module = _CreatingDxcamModule(
        [[_FakeDxcamRawOutput()]],
        [broken, replacement],
    )
    first = _dxcam_adapter(module)
    assert _create_adapter_camera(first) is broken

    with pytest.raises(RuntimeError, match="stop failed"):
        first._cleanup_owned_camera(broken)

    assert broken.release_calls == 1
    second = _dxcam_adapter(module)
    assert _create_adapter_camera(second) is replacement
    assert second._cleanup_owned_camera(replacement) is True


def _source_target(
    *,
    identity=None,
    geometry=None,
    monitor=None,
):
    if identity is None:
        identity = WindowIdentity(101, 202, 303)
    if monitor is None:
        monitor = MonitorDescriptor(
            71,
            "DISPLAY-LEFT",
            ClientGeometry(-1920, -100, 1920, 1080),
        )
    if geometry is None:
        geometry = ClientGeometry(-1800, -50, 640, 360)
    return Sf6WindowTarget(identity, geometry, monitor)


def _matching_dxcam_output(
    *,
    device_index=2,
    output_index=3,
    rotation_angle=0,
    device_name="display-left",
):
    return DxcamOutput(
        71,
        device_name,
        device_index,
        output_index,
        ClientGeometry(-1920, -100, 1920, 1080),
        rotation_angle,
    )


class _SourceFactory:
    def __init__(self, outputs, effects, *, events=None):
        self.outputs_value = outputs
        self.effects = list(effects)
        self.events = events
        self.outputs_calls = 0
        self.create_calls = []

    def outputs(self):
        self.outputs_calls += 1
        if self.events is not None:
            self.events.append(("outputs",))
        if isinstance(self.outputs_value, BaseException):
            raise self.outputs_value
        return self.outputs_value

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.events is not None:
            self.events.append(("create", kwargs["backend"]))
        if not self.effects:
            raise AssertionError("unexpected source factory create")
        effect = self.effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect


def test_start_error_exposes_legal_status_issue_pair_for_each_frozen_stage():
    annotations = get_type_hints(CaptureStartError)
    assert annotations["status"] is CaptureStatus
    assert annotations["issue"] == CaptureIssue | None

    legal = [
        (CaptureStatus.MINIMIZED, None),
        (CaptureStatus.OCCLUDED, None),
        (CaptureStatus.LOST, CaptureIssue.WINDOW_CHANGED),
        (CaptureStatus.LOST, CaptureIssue.GEOMETRY_CHANGED),
        (CaptureStatus.LOST, CaptureIssue.BACKEND_ERROR),
        (CaptureStatus.LOST, CaptureIssue.UNSUPPORTED_OUTPUT),
    ]
    for status, issue in legal:
        error = CaptureStartError(status, issue=issue)
        assert error.status is status
        assert error.issue is issue
        with pytest.raises(AttributeError):
            error.status = CaptureStatus.LOST
        with pytest.raises(AttributeError):
            error.issue = CaptureIssue.BACKEND_ERROR

    invalid = [
        ("LOST", CaptureIssue.BACKEND_ERROR),
        (CaptureStatus.RUNNING, None),
        (CaptureStatus.STOPPED, None),
        (CaptureStatus.LOST, None),
        (CaptureStatus.LOST, "BACKEND_ERROR"),
        (CaptureStatus.MINIMIZED, CaptureIssue.BACKEND_ERROR),
        (CaptureStatus.OCCLUDED, CaptureIssue.BACKEND_ERROR),
    ]
    for status, issue in invalid:
        with pytest.raises((TypeError, ValueError)):
            CaptureStartError(status, issue=issue)


def test_start_resolves_window_before_loading_dxcam_and_maps_each_window_stage():
    before = _source_target()
    changed_identity = _source_target(identity=WindowIdentity(102, 203, 304))
    moved = _source_target(geometry=ClientGeometry(-1700, -40, 640, 360))
    cases = [
        (GameWindowMinimized("minimized"), CaptureStatus.MINIMIZED, None),
        (GameWindowOccluded("occluded"), CaptureStatus.OCCLUDED, None),
        (
            GameWindowNotFound("not found"),
            CaptureStatus.LOST,
            CaptureIssue.WINDOW_CHANGED,
        ),
        (
            GameWindowAmbiguous("ambiguous"),
            CaptureStatus.LOST,
            CaptureIssue.WINDOW_CHANGED,
        ),
        (
            GameWindowIdentityChanged(before=before, after=changed_identity),
            CaptureStatus.LOST,
            CaptureIssue.WINDOW_CHANGED,
        ),
        (
            GameWindowGeometryChanged(before=before, after=moved),
            CaptureStatus.LOST,
            CaptureIssue.GEOMETRY_CHANGED,
        ),
        (
            GameWindowError("native boundary"),
            CaptureStatus.LOST,
            CaptureIssue.BACKEND_ERROR,
        ),
        (
            RuntimeError("unexpected finder failure"),
            CaptureStatus.LOST,
            CaptureIssue.BACKEND_ERROR,
        ),
    ]

    for window_error, status, issue in cases:
        calls = []

        def finder(error=window_error):
            calls.append("window")
            raise error

        def loader():
            calls.append("loader")
            raise AssertionError("loader called after window failure")

        source = DxcamFrameSource(window_finder=finder, dxcam_factory_loader=loader)
        with pytest.raises(CaptureStartError) as raised:
            source.start()

        assert raised.value.status is status
        assert raised.value.issue is issue
        assert calls == ["window"]


def test_start_maps_negative_desktop_coordinates_to_output_relative_region():
    events = []
    camera = _FakeDxcamCamera(events=events, name="dxgi")
    factory = _SourceFactory(
        (_matching_dxcam_output(rotation_angle=270),),
        [camera],
        events=events,
    )
    calls = []

    def finder():
        calls.append("window")
        return _source_target()

    def loader():
        calls.append("loader")
        return factory

    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=loader,
        target_fps=23,
    )

    source.start()

    assert calls == ["window", "loader"]
    assert factory.outputs_calls == 1
    assert factory.create_calls == [
        {
            "device_idx": 2,
            "output_idx": 3,
            "region": (120, 50, 760, 410),
            "output_color": "BGR",
            "max_buffer_len": 2,
            "backend": "dxgi",
            "processor_backend": "cv2",
        }
    ]
    assert camera.start_calls == [
        {
            "region": (120, 50, 760, 410),
            "target_fps": 23,
            "video_mode": False,
        }
    ]
    source.stop()
    assert camera.stop_calls == camera.release_calls == 1


@pytest.mark.parametrize(
    "outputs",
    [
        (),
        (
            _matching_dxcam_output(output_index=3),
            _matching_dxcam_output(output_index=4),
        ),
        (
            DxcamOutput(
                72,
                "DISPLAY-RIGHT",
                2,
                3,
                ClientGeometry(0, -100, 2560, 1440),
                0,
            ),
        ),
    ],
)
def test_start_requires_one_unique_containing_output(outputs):
    factory = _SourceFactory(outputs, [])
    source = DxcamFrameSource(
        window_finder=_source_target,
        dxcam_factory_loader=lambda: factory,
    )

    with pytest.raises(CaptureStartError) as raised:
        source.start()

    assert raised.value.status is CaptureStatus.LOST
    assert raised.value.issue is CaptureIssue.UNSUPPORTED_OUTPUT
    assert factory.create_calls == []


def test_start_rejects_cross_output_and_mirrored_ambiguous_outputs():
    crossing = DxcamOutput(
        71,
        "DISPLAY-LEFT",
        2,
        3,
        ClientGeometry(-1920, -100, 900, 1080),
        0,
    )
    mirrored = (
        _matching_dxcam_output(device_index=0, output_index=0),
        _matching_dxcam_output(device_index=1, output_index=0),
    )

    for outputs in ((crossing,), mirrored):
        factory = _SourceFactory(outputs, [])
        source = DxcamFrameSource(
            window_finder=_source_target,
            dxcam_factory_loader=lambda factory=factory: factory,
        )
        with pytest.raises(CaptureStartError) as raised:
            source.start()
        assert raised.value.issue is CaptureIssue.UNSUPPORTED_OUTPUT
        assert factory.create_calls == []


@pytest.mark.parametrize(
    "outputs",
    [
        [_matching_dxcam_output()],
        (_matching_dxcam_output(), object()),
        RuntimeError("enumeration failed"),
    ],
)
def test_03_adapter_api_drift_fails_closed_as_backend_error_during_start(outputs):
    factory = _SourceFactory(outputs, [])
    source = DxcamFrameSource(
        window_finder=_source_target,
        dxcam_factory_loader=lambda: factory,
    )

    with pytest.raises(CaptureStartError) as raised:
        source.start()

    assert raised.value.status is CaptureStatus.LOST
    assert raised.value.issue is CaptureIssue.BACKEND_ERROR
    assert factory.create_calls == []


def test_start_cleans_partial_dxgi_camera_before_winrt_fallback():
    events = []
    dxgi = _FakeDxcamCamera(
        start_error=RuntimeError("DXGI start failed"),
        events=events,
        name="dxgi",
    )
    winrt = _FakeDxcamCamera(events=events, name="winrt")
    factory = _SourceFactory(
        (_matching_dxcam_output(),),
        [dxgi, winrt],
        events=events,
    )
    source = DxcamFrameSource(
        window_finder=_source_target,
        dxcam_factory_loader=lambda: factory,
    )

    source.start()

    assert events == [
        ("outputs",),
        ("create", "dxgi"),
        (
            "dxgi.start",
            {
                "region": (120, 50, 760, 410),
                "target_fps": 20,
                "video_mode": False,
            },
        ),
        ("dxgi.stop",),
        ("dxgi.release",),
        ("create", "winrt"),
        (
            "winrt.start",
            {
                "region": (120, 50, 760, 410),
                "target_fps": 20,
                "video_mode": False,
            },
        ),
    ]
    assert [call["backend"] for call in factory.create_calls] == ["dxgi", "winrt"]
    source.stop()
    assert winrt.stop_calls == winrt.release_calls == 1


def test_start_uses_adapter_owned_cleanup_before_fallback_and_stop():
    events = []
    dxgi = _FakeDxcamCamera(
        start_error=RuntimeError("DXGI start failed"),
        events=events,
        name="dxgi",
    )
    winrt = _FakeDxcamCamera(events=events, name="winrt")
    module = _CreatingDxcamModule(
        [[_FakeDxcamRawOutput()]],
        [dxgi, winrt],
    )
    adapter = _dxcam_adapter(module)
    source = DxcamFrameSource(
        window_finder=_source_target,
        dxcam_factory_loader=lambda: adapter,
    )

    source.start()

    assert [call["backend"] for call in module.create_calls] == ["dxgi", "winrt"]
    assert dxgi.stop_calls == dxgi.release_calls == 1
    source.stop()
    assert winrt.stop_calls == winrt.release_calls == 1
    retry = _dxcam_adapter(module)
    replacement = _FakeDxcamCamera()
    module.effects.append(replacement)
    assert retry.create(
        device_idx=0,
        output_idx=0,
        region=(120, 50, 760, 410),
        output_color="BGR",
        max_buffer_len=2,
        backend="winrt",
        processor_backend="cv2",
    ) is replacement
    assert retry._cleanup_owned_camera(replacement) is True


def test_source_does_not_directly_cleanup_when_factory_denies_ownership_claim():
    camera = _FakeDxcamCamera(start_error=RuntimeError("start failed"))

    class DenyingFactory(_SourceFactory):
        def __init__(self):
            super().__init__((_matching_dxcam_output(),), [camera])
            self.cleanup_calls = []

        def _cleanup_owned_camera(self, candidate):
            self.cleanup_calls.append(candidate)
            return False

    factory = DenyingFactory()
    source = DxcamFrameSource(
        window_finder=_source_target,
        dxcam_factory_loader=lambda: factory,
    )

    with pytest.raises(CaptureStartError) as raised:
        source.start()

    assert raised.value.issue is CaptureIssue.BACKEND_ERROR
    assert factory.cleanup_calls == [camera]
    assert len(factory.create_calls) == 1
    assert camera.stop_calls == camera.release_calls == 0


def test_all_backend_failures_leave_source_stopped_and_restartable():
    recovered = _FakeDxcamCamera()
    factory = _SourceFactory(
        (_matching_dxcam_output(),),
        [
            RuntimeError("DXGI create failed"),
            RuntimeError("WinRT create failed"),
            recovered,
        ],
    )
    calls = []

    def finder():
        calls.append("window")
        return _source_target()

    def loader():
        calls.append("loader")
        return factory

    source = DxcamFrameSource(window_finder=finder, dxcam_factory_loader=loader)

    with pytest.raises(CaptureStartError) as raised:
        source.start()
    assert raised.value.issue is CaptureIssue.BACKEND_ERROR

    source.start()
    assert calls == ["window", "loader", "window", "loader"]
    assert [call["backend"] for call in factory.create_calls] == [
        "dxgi",
        "winrt",
        "dxgi",
    ]
    source.stop()
    assert recovered.stop_calls == recovered.release_calls == 1


class _Step6Camera(_FakeDxcamCamera):
    def __init__(self, read_effects, *, events=None, name="camera"):
        super().__init__(events=events, name=name)
        self.read_effects = list(read_effects)

    def get_latest_frame(self, **kwargs):
        if self.events is not None:
            self.events.append((f"{self.name}.get_latest_frame", kwargs))
        self.read_calls.append(kwargs)
        if not self.read_effects:
            raise AssertionError("unexpected camera read")
        effect = self.read_effects.pop(0)
        if callable(effect):
            effect = effect()
        if isinstance(effect, BaseException):
            raise effect
        return effect

    def grab(self):
        raise AssertionError("read must use the atomic timestamped API")

    @property
    def latest_frame_time(self):
        raise AssertionError("read must not fetch a separate timestamp")


class _ExplodingCaptureStateCamera(_Step6Camera):
    def __init__(self, read_effects, *, events=None, name="camera"):
        self._stored_is_capturing = False
        self.raise_capture_state = False
        super().__init__(read_effects, events=events, name=name)

    @property
    def is_capturing(self):
        if self.raise_capture_state:
            raise RuntimeError("capture state probe failed")
        return self._stored_is_capturing

    @is_capturing.setter
    def is_capturing(self, value):
        self._stored_is_capturing = value


class _ExplodingAnyArray(np.ndarray):
    def __new__(cls, shape):
        instance = super().__new__(cls, shape, dtype=np.uint8)
        instance.fill(1)
        return instance

    def __array_function__(self, function, types, args, kwargs):
        if function is np.any:
            raise RuntimeError("array reduction failed")
        return super().__array_function__(function, types, args, kwargs)


class _Step6Finder:
    def __init__(self, effects, *, events=None):
        self.effects = list(effects)
        self.events = events
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.events is not None:
            self.events.append(("finder", self.calls))
        if not self.effects:
            raise AssertionError("unexpected window finder call")
        effect = self.effects.pop(0) if len(self.effects) > 1 else self.effects[0]
        if callable(effect):
            effect = effect()
        if isinstance(effect, BaseException):
            raise effect
        return effect


class _Step6Factory(_SourceFactory):
    def __init__(self, output_effects, cameras, *, events=None):
        first_output = output_effects[0] if output_effects else ()
        super().__init__(first_output, cameras, events=events)
        self.output_effects = list(output_effects)

    def outputs(self):
        self.outputs_calls += 1
        if self.events is not None:
            self.events.append(("outputs", self.outputs_calls))
        if not self.output_effects:
            raise AssertionError("unexpected output enumeration")
        effect = (
            self.output_effects.pop(0)
            if len(self.output_effects) > 1
            else self.output_effects[0]
        )
        if callable(effect):
            effect = effect()
        if isinstance(effect, BaseException):
            raise effect
        return effect


def _step6_target(*, identity=None, geometry=None, monitor=None):
    return _source_target(
        identity=identity,
        geometry=geometry or ClientGeometry(-1800, -50, 3, 2),
        monitor=monitor,
    )


def _step6_image(value=17, *, geometry=None):
    geometry = geometry or ClientGeometry(-1800, -50, 3, 2)
    return np.full(
        (geometry.height, geometry.width, 3),
        value,
        dtype=np.uint8,
    )


def _start_step6_source(
    camera,
    *,
    finder_effects=None,
    output_effects=None,
    replacement_cameras=(),
    events=None,
):
    target = _step6_target()
    output = _matching_dxcam_output()
    finder = _Step6Finder(finder_effects or [target], events=events)
    factory = _Step6Factory(
        output_effects or [(output,)],
        [camera, *replacement_cameras],
        events=events,
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    return source, finder, factory, target, output


def test_read_uses_get_latest_frame_copy_true_with_timestamp_true():
    image = _step6_image()
    camera = _Step6Camera([(image, Fraction(3, 2))])
    source, _, _, _, _ = _start_step6_source(camera)

    sample = source.read()

    assert sample.status is CaptureStatus.RUNNING
    assert camera.read_calls == [{"copy": True, "with_timestamp": True}]
    source.stop()


def test_none_result_is_no_new_frame_and_never_replays_cached_pixels():
    first_image = _step6_image(23)
    camera = _Step6Camera([(first_image, 1), None])
    source, _, _, _, _ = _start_step6_source(camera)

    first = source.read()
    second = source.read()

    assert first.status is CaptureStatus.RUNNING
    assert first.frame.image is first_image
    assert second == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.NO_NEW_FRAME,
    )
    assert second.frame is None
    assert camera.read_calls == [
        {"copy": True, "with_timestamp": True},
        {"copy": True, "with_timestamp": True},
    ]
    source.stop()


def test_none_from_active_generation_uses_is_capturing_true_for_no_new_frame():
    camera = _Step6Camera([None])
    source, _, _, _, _ = _start_step6_source(camera)

    sample = source.read()

    assert camera.is_capturing is True
    assert sample == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.NO_NEW_FRAME,
    )
    source.stop()


@pytest.mark.parametrize("capture_state", [False, 0, 1, "yes", None])
def test_none_with_is_capturing_false_or_invalid_is_backend_error(capture_state):
    camera = _Step6Camera([None])
    source, _, factory, target, output = _start_step6_source(camera)
    camera.is_capturing = capture_state

    sample = source.read()

    assert sample == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.BACKEND_ERROR,
    )
    assert source._camera is camera
    assert source._factory is factory
    assert source._target == target
    assert source._output == output
    camera.is_capturing = True
    source.stop()


def test_none_with_is_capturing_property_failure_is_backend_error():
    camera = _ExplodingCaptureStateCamera([None])
    source, _, _, _, _ = _start_step6_source(camera)
    camera.raise_capture_state = True

    sample = source.read()

    assert sample == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.BACKEND_ERROR,
    )
    assert camera.stop_calls == camera.release_calls == 0
    camera.raise_capture_state = False
    source.stop()


def test_equal_pixels_with_new_backend_timestamps_emit_two_running_samples():
    first_image = _step6_image(41)
    second_image = first_image.copy()
    camera = _Step6Camera([(first_image, 10), (second_image, 11)])
    source, _, _, _, _ = _start_step6_source(camera)

    first = source.read()
    second = source.read()

    assert first.status is second.status is CaptureStatus.RUNNING
    assert first.frame.frame_id == 0
    assert second.frame.frame_id == 1
    assert np.array_equal(first.frame.image, second.frame.image)
    source.stop()


def test_repeated_or_decreasing_backend_timestamp_is_no_new_frame():
    images = [_step6_image(value) for value in (10, 11, 12, 13)]
    camera = _Step6Camera(
        [
            (images[0], 10),
            (images[1], 10),
            (images[2], 9),
            (images[3], 11),
        ]
    )
    source, _, _, _, _ = _start_step6_source(camera)

    accepted = source.read()
    repeated = source.read()
    decreasing = source.read()
    recovered = source.read()

    assert accepted.frame.frame_id == 0
    assert repeated.issue is CaptureIssue.NO_NEW_FRAME
    assert decreasing.issue is CaptureIssue.NO_NEW_FRAME
    assert recovered.frame.frame_id == 1
    assert recovered.frame.monotonic_time == 11
    source.stop()


def test_timestamp_rejection_does_not_advance_frame_id_or_time():
    camera = _Step6Camera(
        [
            (_step6_image(31), Fraction(10)),
            (_step6_image(32), Fraction(9)),
            (_step6_image(33), Fraction(101, 10)),
        ]
    )
    source, _, _, _, _ = _start_step6_source(camera)

    first = source.read()
    rejected = source.read()
    next_valid = source.read()

    assert first.frame.frame_id == 0
    assert rejected.issue is CaptureIssue.NO_NEW_FRAME
    assert next_valid.frame.frame_id == 1
    assert next_valid.frame.monotonic_time == Fraction(101, 10)
    source.stop()


def test_exact_all_zero_image_is_black_frame():
    valid = _step6_image(19)
    camera = _Step6Camera([(_step6_image(0), 10), (valid, 1)])
    source, _, _, _, _ = _start_step6_source(camera)

    rejected = source.read()
    accepted = source.read()

    assert rejected == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.BLACK_FRAME,
    )
    assert accepted.status is CaptureStatus.RUNNING
    assert accepted.frame.frame_id == 0
    assert accepted.frame.monotonic_time == 1
    assert accepted.frame.image is valid
    source.stop()


def test_array_reduction_failure_is_backend_error_without_advancing_watermarks():
    exploding = _ExplodingAnyArray((2, 3, 3))
    valid = _step6_image(20)
    assert exploding.dtype == np.dtype(np.uint8)
    assert exploding.flags.c_contiguous
    assert exploding.flags.owndata
    assert exploding.flags.writeable

    camera = _Step6Camera([(exploding, 10), (valid, 1)])
    source, _, factory, target, output = _start_step6_source(camera)
    generation = source._camera_generation

    rejected = source.read()
    accepted = source.read()

    assert rejected == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.BACKEND_ERROR,
    )
    assert accepted.status is CaptureStatus.RUNNING
    assert accepted.frame.frame_id == 0
    assert accepted.frame.monotonic_time == 1
    assert accepted.frame.image is valid
    assert source._camera is camera
    assert source._factory is factory
    assert source._target == target
    assert source._output == output
    assert source._camera_generation == generation
    assert camera.stop_calls == camera.release_calls == 0
    source.stop()


def test_near_black_white_and_low_contrast_images_reach_running_sample():
    images = [_step6_image(1), _step6_image(255), _step6_image(127)]
    camera = _Step6Camera(
        [(image, timestamp) for timestamp, image in enumerate(images, start=1)]
    )
    source, _, _, _, _ = _start_step6_source(camera)

    samples = [source.read(), source.read(), source.read()]

    assert [sample.status for sample in samples] == [CaptureStatus.RUNNING] * 3
    assert [sample.frame.frame_id for sample in samples] == [0, 1, 2]
    assert all(
        sample.frame.image is image
        for sample, image in zip(samples, images, strict=True)
    )
    source.stop()


def test_invalid_dtype_shape_layout_or_nonowned_buffer_is_backend_error():
    geometry = ClientGeometry(-1800, -50, 3, 2)
    wrong_dtype = np.ones((2, 3, 3), dtype=np.float32)
    wrong_shape = np.ones((2, 4, 3), dtype=np.uint8)
    non_contiguous = np.asfortranarray(_step6_image(3, geometry=geometry))
    non_owned = _step6_image(4, geometry=geometry).view()
    read_only = _step6_image(5, geometry=geometry)
    read_only.setflags(write=False)
    invalid_images = [
        wrong_dtype,
        wrong_shape,
        non_contiguous,
        non_owned,
        read_only,
    ]
    valid = _step6_image(6, geometry=geometry)
    camera = _Step6Camera(
        [
            *((image, timestamp) for timestamp, image in enumerate(invalid_images, 1)),
            (valid, Fraction(1, 2)),
        ]
    )
    source, _, _, _, _ = _start_step6_source(camera)

    rejected = [source.read() for _ in invalid_images]
    accepted = source.read()

    assert [sample.issue for sample in rejected] == [
        CaptureIssue.BACKEND_ERROR
    ] * len(invalid_images)
    assert accepted.status is CaptureStatus.RUNNING
    assert accepted.frame.frame_id == 0
    assert accepted.frame.image is valid
    source.stop()


@pytest.mark.parametrize(
    "bad_result",
    [
        object(),
        (_step6_image(7),),
        (_step6_image(7), 1, "extra"),
        (object(), 1),
        (_step6_image(7), True),
        (_step6_image(7), -1),
        (_step6_image(7), float("nan")),
        (_step6_image(7), float("inf")),
    ],
)
def test_invalid_backend_result_or_timestamp_is_backend_error_without_advancing(
    bad_result,
):
    valid = _step6_image(8)
    camera = _Step6Camera([bad_result, (valid, 2)])
    source, _, _, _, _ = _start_step6_source(camera)

    rejected = source.read()
    accepted = source.read()

    assert rejected.issue is CaptureIssue.BACKEND_ERROR
    assert accepted.status is CaptureStatus.RUNNING
    assert accepted.frame.frame_id == 0
    assert accepted.frame.image is valid
    source.stop()


def test_running_sample_preserves_backend_timestamp_and_array_identity():
    timestamp = _NoFloatFraction(10**200, 7)
    image = _step6_image(91)
    camera = _Step6Camera([(image, timestamp)])
    source, _, _, target, _ = _start_step6_source(camera)

    sample = source.read()

    assert sample.status is CaptureStatus.RUNNING
    assert sample.frame.image is image
    assert sample.frame.monotonic_time is timestamp
    assert sample.frame.geometry is target.geometry
    assert image.flags.owndata
    assert image.flags.c_contiguous
    assert image.flags.writeable
    source.stop()


def test_active_backend_exception_preserves_attachment_for_retry():
    image = _step6_image(52)
    camera = _Step6Camera([RuntimeError("backend failed"), (image, 4)])
    source, _, factory, target, output = _start_step6_source(camera)
    generation = source._camera_generation

    rejected = source.read()
    recovered = source.read()

    assert rejected.issue is CaptureIssue.BACKEND_ERROR
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 0
    assert source._camera is camera
    assert source._factory is factory
    assert source._target is target
    assert source._output is output
    assert source._camera_generation == generation
    assert camera.stop_calls == camera.release_calls == 0
    assert len(factory.create_calls) == 1
    source.stop()


def test_read_before_start_returns_stopped_without_touching_dependencies():
    def forbidden_window():
        raise AssertionError("stopped read touched the window finder")

    def forbidden_factory():
        raise AssertionError("stopped read touched the DXcam factory")

    source = DxcamFrameSource(
        window_finder=forbidden_window,
        dxcam_factory_loader=forbidden_factory,
    )

    assert source.read() == CaptureSample(CaptureStatus.STOPPED)


def test_read_order_is_pre_target_output_then_atomic_call_then_generation_then_post_target_output():
    events = []
    image = _step6_image(61)
    camera = _Step6Camera([(image, 1)], events=events)
    source, _, _, target, output = _start_step6_source(
        camera,
        finder_effects=[_step6_target()],
        output_effects=[(_matching_dxcam_output(),)],
        events=events,
    )
    events.clear()

    sample = source.read()

    assert sample.status is CaptureStatus.RUNNING
    assert events == [
        ("finder", 2),
        ("outputs", 2),
        (
            "camera.get_latest_frame",
            {"copy": True, "with_timestamp": True},
        ),
        ("finder", 3),
        ("outputs", 3),
    ]
    assert source._target == target
    assert source._output == output
    source.stop()


def test_post_read_always_reenumerates_and_uniquely_maps_output():
    image = _step6_image(62)
    camera = _Step6Camera([(image, 1)])
    output = _matching_dxcam_output()
    source, _, factory, _, _ = _start_step6_source(
        camera,
        output_effects=[(output,), (output,), ()],
    )

    sample = source.read()

    assert sample == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.UNSUPPORTED_OUTPUT,
    )
    assert sample.frame is None
    assert factory.outputs_calls == 3
    source.stop()


@pytest.mark.parametrize(
    "finder_error,expected_status",
    [
        (GameWindowMinimized("minimized"), CaptureStatus.MINIMIZED),
        (GameWindowOccluded("occluded"), CaptureStatus.OCCLUDED),
    ],
)
def test_active_read_maps_minimized_and_occluded_before_backend_call(
    finder_error,
    expected_status,
):
    camera = _Step6Camera([AssertionError("camera must not be read")])
    source, _, _, _, _ = _start_step6_source(
        camera,
        finder_effects=[_step6_target(), finder_error],
    )

    sample = source.read()

    assert sample == CaptureSample(expected_status)
    assert camera.read_calls == []
    source.stop()


def test_generic_window_or_output_backend_error_preserves_active_attachment():
    image = _step6_image(63)
    camera = _Step6Camera([(image, 2)])
    target = _step6_target()
    output = _matching_dxcam_output()
    finder = _Step6Finder(
        [target, GameWindowError("DWM query failed"), target, target]
    )
    factory = _Step6Factory([(output,)], [camera])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    generation = source._camera_generation

    rejected = source.read()
    recovered = source.read()

    assert rejected.issue is CaptureIssue.BACKEND_ERROR
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 0
    assert source._camera is camera
    assert source._camera_generation == generation
    assert camera.stop_calls == camera.release_calls == 0
    source.stop()


def test_generic_output_enumeration_error_preserves_active_attachment():
    target = _step6_target()
    output = _matching_dxcam_output()
    image = _step6_image(73)
    camera = _Step6Camera([(image, 2)])
    finder = _Step6Finder([target])
    factory = _Step6Factory(
        [
            (output,),
            RuntimeError("output enumeration failed"),
            (output,),
        ],
        [camera],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    generation = source._camera_generation

    rejected = source.read()
    recovered = source.read()

    assert rejected.issue is CaptureIssue.BACKEND_ERROR
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 0
    assert source._camera is camera
    assert source._camera_generation == generation
    assert source._state == "running"
    assert source._pending_geometry_notice is False
    assert camera.stop_calls == camera.release_calls == 0
    assert len(factory.create_calls) == 1
    source.stop()


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_changed_target_then_output_error_suspends_and_preserves_epoch_until_notice(
    phase,
):
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1796, -50, 3, 2))
    output = _matching_dxcam_output()
    old_effects = [(_step6_image(74), 10)]
    finder_effects = [attached, attached, attached]
    output_effects = [(output,), (output,), (output,)]
    if phase == "pre":
        finder_effects.extend([moved, moved])
        output_effects.extend(
            [RuntimeError("changed-target output refresh failed"), (output,)]
        )
    else:
        old_effects.append((_step6_image(75), 100))
        finder_effects.extend([attached, moved, moved])
        output_effects.extend(
            [
                (output,),
                RuntimeError("changed-target output refresh failed"),
                (output,),
            ]
        )
    old = _Step6Camera(old_effects, name="old")
    recovered_image = _step6_image(76, geometry=moved.geometry)
    recovered = _Step6Camera([(recovered_image, 11)], name="recovered")
    finder = _Step6Finder(finder_effects)
    factory = _Step6Factory(output_effects, [old, recovered])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    first = source.read()

    failed = source.read()

    assert first.status is CaptureStatus.RUNNING
    assert first.frame.frame_id == 0
    assert first.frame.monotonic_time == 10
    assert failed == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.BACKEND_ERROR,
    )
    assert source._state == "suspended"
    assert source._camera is None
    assert source._pending_geometry_notice is True
    assert source._next_frame_id == 1
    assert source._last_backend_timestamp == 10
    assert old.stop_calls == old.release_calls == 1

    notice = source.read()
    assert notice == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.GEOMETRY_CHANGED,
    )
    assert recovered.read_calls == []

    accepted = source.read()
    assert accepted.status is CaptureStatus.RUNNING
    assert accepted.frame.frame_id == 1
    assert accepted.frame.monotonic_time == 11
    assert accepted.frame.image is recovered_image
    assert source._pending_geometry_notice is False
    assert old.stop_calls == old.release_calls == 1
    source.stop()


def test_geometry_exception_then_output_error_suspends_and_retries_with_notice():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1795, -50, 3, 2))
    geometry_error = GameWindowGeometryChanged(before=attached, after=moved)
    output = _matching_dxcam_output()
    old = _Step6Camera([], name="old")
    recovered_image = _step6_image(77, geometry=moved.geometry)
    recovered = _Step6Camera([(recovered_image, 1)], name="recovered")
    finder = _Step6Finder([attached, geometry_error, moved])
    factory = _Step6Factory(
        [
            (output,),
            RuntimeError("geometry-exception output refresh failed"),
            (output,),
        ],
        [old, recovered],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    failed = source.read()

    assert failed.issue is CaptureIssue.BACKEND_ERROR
    assert source._state == "suspended"
    assert source._camera is None
    assert source._pending_geometry_notice is True
    assert old.stop_calls == old.release_calls == 1

    notice = source.read()
    assert notice.issue is CaptureIssue.GEOMETRY_CHANGED
    assert recovered.read_calls == []
    accepted = source.read()
    assert accepted.status is CaptureStatus.RUNNING
    assert accepted.frame.frame_id == 0
    assert accepted.frame.image is recovered_image
    source.stop()


def test_stale_snapshot_cannot_set_pending_geometry_on_a_new_session():
    old = _Step6Camera([], name="old")
    current = _Step6Camera([], name="current")
    source, _, _, _, _ = _start_step6_source(
        old,
        replacement_cameras=[current],
    )
    stale_snapshot = source._active_snapshot_locked()
    source.stop()
    source.start()

    rejected = source._suspend_attachment(
        stale_snapshot,
        CaptureSample(CaptureStatus.LOST, issue=CaptureIssue.BACKEND_ERROR),
        pending_geometry_notice=True,
    )

    assert rejected == CaptureSample(CaptureStatus.STOPPED)
    assert source._state == "running"
    assert source._camera is current
    assert source._pending_geometry_notice is False
    assert current.stop_calls == current.release_calls == 0
    source.stop()


@pytest.mark.parametrize("post_change", ["identity", "geometry", "output"])
def test_post_window_and_output_changes_precede_backend_exception(post_change):
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1796, -50, 3, 2))
    foreign = _step6_target(identity=WindowIdentity(130, 230, 330))
    old_output = _matching_dxcam_output()
    changed_output = _matching_dxcam_output(rotation_angle=180)
    old = _Step6Camera([RuntimeError("atomic capture failed")], name="old")
    replacement = _Step6Camera([], name="replacement")

    if post_change == "identity":
        post_target = foreign
        output_effects = [(old_output,)]
        cameras = [old]
        expected_issue = CaptureIssue.WINDOW_CHANGED
    elif post_change == "geometry":
        post_target = moved
        output_effects = [(old_output,)]
        cameras = [old, replacement]
        expected_issue = CaptureIssue.GEOMETRY_CHANGED
    else:
        post_target = attached
        output_effects = [
            (old_output,),
            (old_output,),
            (changed_output,),
        ]
        cameras = [old, replacement]
        expected_issue = CaptureIssue.GEOMETRY_CHANGED

    finder = _Step6Finder([attached, attached, post_target])
    factory = _Step6Factory(output_effects, cameras)
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    sample = source.read()

    assert sample == CaptureSample(CaptureStatus.LOST, issue=expected_issue)
    assert old.read_calls == [{"copy": True, "with_timestamp": True}]
    assert old.stop_calls == old.release_calls == 1
    if post_change == "identity":
        assert source._camera is None
        assert len(factory.create_calls) == 1
    else:
        assert source._camera is replacement
        assert len(factory.create_calls) == 2
    source.stop()


def test_identity_change_during_block_discards_payload_and_latches_window_changed():
    attached = _step6_target()
    foreign = _step6_target(identity=WindowIdentity(102, 203, 304))
    output = _matching_dxcam_output()
    camera = _Step6Camera([(_step6_image(64), 1)])
    finder = _Step6Finder([attached, attached, foreign])
    factory = _Step6Factory([(output,)], [camera])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    changed = source.read()
    finder_calls_after_change = finder.calls
    output_calls_after_change = factory.outputs_calls
    latched = source.read()

    expected = CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.WINDOW_CHANGED,
    )
    assert changed == expected
    assert changed.frame is None
    assert latched == expected
    assert finder.calls == finder_calls_after_change
    assert factory.outputs_calls == output_calls_after_change
    assert camera.read_calls == [{"copy": True, "with_timestamp": True}]
    assert camera.stop_calls == camera.release_calls == 1
    source.stop()
    assert camera.stop_calls == camera.release_calls == 1
    assert camera.stop_calls == camera.release_calls == 1


def test_geometry_change_during_block_discards_payload_and_rebuilds_camera():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1799, -50, 3, 2))
    output = _matching_dxcam_output()
    old = _Step6Camera([(_step6_image(65), 1)], name="old")
    rebuilt_image = _step6_image(66, geometry=moved.geometry)
    replacement = _Step6Camera([(rebuilt_image, 2)], name="replacement")
    finder = _Step6Finder([attached, attached, moved, moved])
    factory = _Step6Factory([(output,)], [old, replacement])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    changed = source.read()
    recovered = source.read()

    assert changed == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.GEOMETRY_CHANGED,
    )
    assert changed.frame is None
    assert old.stop_calls == old.release_calls == 1
    assert replacement.start_calls == [
        {
            "region": (121, 50, 124, 52),
            "target_fps": 20,
            "video_mode": False,
        }
    ]
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 0
    assert recovered.frame.image is rebuilt_image
    source.stop()


def _assert_post_output_change_rebuilds(old_output, changed_output):
    target = _step6_target()
    old = _Step6Camera([(_step6_image(67), 1)], name="old")
    replacement = _Step6Camera([(_step6_image(68), 2)], name="replacement")
    finder = _Step6Finder([target])
    factory = _Step6Factory(
        [(old_output,), (old_output,), (changed_output,)],
        [old, replacement],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    changed = source.read()

    assert changed == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.GEOMETRY_CHANGED,
    )
    assert changed.frame is None
    assert old.stop_calls == old.release_calls == 1
    assert replacement.start_calls == [
        {
            "region": (120, 50, 123, 52),
            "target_fps": 20,
            "video_mode": False,
        }
    ]
    assert source._camera is replacement
    assert source._output is changed_output
    assert factory.create_calls[-1]["device_idx"] == changed_output.device_index
    assert factory.create_calls[-1]["output_idx"] == changed_output.output_index
    source.stop()


@pytest.mark.parametrize(
    "changed_output",
    [
        _matching_dxcam_output(device_index=4, output_index=3),
        _matching_dxcam_output(device_index=2, output_index=5),
    ],
    ids=["device-index-only", "output-index-only"],
)
def test_output_change_during_block_discards_payload_and_rebuilds_camera(
    changed_output,
):
    old_output = _matching_dxcam_output(device_index=2, output_index=3)

    _assert_post_output_change_rebuilds(old_output, changed_output)


def test_rotation_only_change_during_block_discards_payload_and_rebuilds_camera():
    old_output = _matching_dxcam_output(rotation_angle=0)
    changed_output = _matching_dxcam_output(rotation_angle=90)

    _assert_post_output_change_rebuilds(old_output, changed_output)


@pytest.mark.parametrize("phase", ["pre", "post"])
@pytest.mark.parametrize("change_kind", ["identity", "geometry"])
def test_mid_finder_identity_and_geometry_exceptions_map_to_distinct_adapter_paths(
    phase,
    change_kind,
):
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1798, -50, 3, 2))
    foreign = _step6_target(identity=WindowIdentity(110, 210, 310))
    output = _matching_dxcam_output()
    old_effects = [] if phase == "pre" else [(_step6_image(69), 1)]
    old = _Step6Camera(old_effects, name="old")
    replacement = _Step6Camera([], name="replacement")
    if change_kind == "identity":
        finder_error = GameWindowIdentityChanged(
            before=attached,
            after=foreign,
        )
        expected_issue = CaptureIssue.WINDOW_CHANGED
        cameras = [old]
    else:
        finder_error = GameWindowGeometryChanged(before=attached, after=moved)
        expected_issue = CaptureIssue.GEOMETRY_CHANGED
        cameras = [old, replacement]
    finder_effects = (
        [attached, finder_error]
        if phase == "pre"
        else [attached, attached, finder_error]
    )
    finder = _Step6Finder(finder_effects)
    factory = _Step6Factory([(output,)], cameras)
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    sample = source.read()

    assert sample == CaptureSample(CaptureStatus.LOST, issue=expected_issue)
    assert len(old.read_calls) == (0 if phase == "pre" else 1)
    assert old.stop_calls == old.release_calls == 1
    if change_kind == "identity":
        assert len(factory.create_calls) == 1
    else:
        assert len(factory.create_calls) == 2
        assert source._camera is replacement
    source.stop()


def test_geometry_changed_with_foreign_before_after_latches_window_changed():
    attached = _step6_target()
    foreign_identity = WindowIdentity(120, 220, 320)
    foreign_before = _step6_target(identity=foreign_identity)
    foreign_after = _step6_target(
        identity=foreign_identity,
        geometry=ClientGeometry(-1797, -50, 3, 2),
    )
    nominal_geometry_error = GameWindowGeometryChanged(
        before=foreign_before,
        after=foreign_after,
    )
    camera = _Step6Camera([AssertionError("foreign target must skip camera")])
    source, _, factory, _, _ = _start_step6_source(
        camera,
        finder_effects=[attached, nominal_geometry_error],
        output_effects=[
            (_matching_dxcam_output(),),
            RuntimeError("foreign target output refresh failed"),
        ],
    )

    sample = source.read()

    assert sample == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.WINDOW_CHANGED,
    )
    assert camera.read_calls == []
    assert camera.stop_calls == camera.release_calls == 1
    assert len(factory.create_calls) == 1
    assert factory.outputs_calls == 2
    assert source.read().issue is CaptureIssue.WINDOW_CHANGED
    source.stop()


@pytest.mark.parametrize(
    "identity_error",
    [
        GameWindowNotFound("missing"),
        GameWindowAmbiguous("ambiguous"),
    ],
)
def test_missing_or_ambiguous_window_latches_identity_change(identity_error):
    camera = _Step6Camera([AssertionError("missing window must skip camera")])
    source, finder, _, target, _ = _start_step6_source(
        camera,
        finder_effects=[_step6_target(), identity_error],
    )

    first = source.read()
    second = source.read()

    assert first.issue is CaptureIssue.WINDOW_CHANGED
    assert second.issue is CaptureIssue.WINDOW_CHANGED
    assert finder.calls == 2
    assert camera.read_calls == []
    assert camera.stop_calls == camera.release_calls == 1
    assert source._target is None
    assert target.identity != WindowIdentity(999, 999, 999)
    source.stop()


def _threaded_read(source, outcomes, errors):
    try:
        outcomes.append(source.read())
    except Exception as exc:
        errors.append(exc)


def test_stop_inside_atomic_get_invalidates_generation_and_skips_post_checks():
    target = _step6_target()
    output = _matching_dxcam_output()
    source = None

    def stop_inside_get():
        source.stop()
        return _step6_image(74), 1

    camera = _Step6Camera([stop_inside_get])
    finder = _Step6Finder([target])
    factory = _Step6Factory([(output,)], [camera])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    sample = source.read()

    assert sample == CaptureSample(CaptureStatus.STOPPED)
    assert finder.calls == 2
    assert factory.outputs_calls == 2
    assert camera.read_calls == [{"copy": True, "with_timestamp": True}]
    assert camera.stop_calls == camera.release_calls == 1


def test_stop_during_blocked_pre_finder_invalidates_generation_before_camera_call():
    target = _step6_target()
    output = _matching_dxcam_output()
    finder_entered = threading.Event()
    allow_finder = threading.Event()

    def blocked_pre_finder():
        finder_entered.set()
        if not allow_finder.wait(timeout=2):
            raise AssertionError("pre finder was not released")
        return target

    camera = _Step6Camera([AssertionError("stale camera must not be read")])
    finder = _Step6Finder([target, blocked_pre_finder])
    factory = _Step6Factory([(output,)], [camera])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    outcomes = []
    errors = []
    reader = threading.Thread(
        target=_threaded_read,
        args=(source, outcomes, errors),
    )
    reader.start()
    assert finder_entered.wait(timeout=2)
    try:
        source.stop()
    finally:
        allow_finder.set()
    reader.join(timeout=2)

    assert not reader.is_alive()
    assert errors == []
    assert outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    assert camera.read_calls == []
    assert camera.stop_calls == camera.release_calls == 1


def test_stop_during_blocked_post_finder_discards_payload_before_interpretation():
    target = _step6_target()
    output = _matching_dxcam_output()
    finder_entered = threading.Event()
    allow_finder = threading.Event()

    def blocked_post_finder():
        finder_entered.set()
        if not allow_finder.wait(timeout=2):
            raise AssertionError("post finder was not released")
        return target

    camera = _Step6Camera([(_step6_image(70), 1)])
    finder = _Step6Finder([target, target, blocked_post_finder])
    factory = _Step6Factory([(output,)], [camera])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    outcomes = []
    errors = []
    reader = threading.Thread(
        target=_threaded_read,
        args=(source, outcomes, errors),
    )
    reader.start()
    assert finder_entered.wait(timeout=2)
    try:
        source.stop()
    finally:
        allow_finder.set()
    reader.join(timeout=2)

    assert not reader.is_alive()
    assert errors == []
    assert outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    assert camera.read_calls == [{"copy": True, "with_timestamp": True}]
    assert camera.stop_calls == camera.release_calls == 1


def test_same_identity_geometry_change_releases_and_rebuilds_before_running():
    events = []
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1795, -50, 3, 2))
    output = _matching_dxcam_output()
    old = _Step6Camera([], events=events, name="old")
    image = _step6_image(81, geometry=moved.geometry)
    replacement = _Step6Camera(
        [(image, 2)],
        events=events,
        name="replacement",
    )
    finder = _Step6Finder([attached, moved], events=events)
    factory = _Step6Factory([(output,)], [old, replacement], events=events)
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    changed = source.read()
    recovered = source.read()

    assert changed == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.GEOMETRY_CHANGED,
    )
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 0
    assert recovered.frame.image is image
    event_names = [event[0] for event in events]
    assert event_names.index("old.stop") < event_names.index("replacement.start")
    assert event_names.index("old.release") < event_names.index("replacement.start")
    assert old.stop_calls == old.release_calls == 1
    source.stop()


def test_identity_change_latches_window_changed_until_restart():
    attached = _step6_target()
    foreign = _step6_target(identity=WindowIdentity(501, 502, 503))
    output = _matching_dxcam_output()
    old = _Step6Camera([])
    restarted_image = _step6_image(82)
    restarted = _Step6Camera([(restarted_image, 1)])
    finder = _Step6Finder([attached, foreign, attached])
    factory = _Step6Factory([(output,)], [old, restarted])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    changed = source.read()
    calls_after_change = (finder.calls, factory.outputs_calls)
    still_latched = source.read()

    assert changed.issue is CaptureIssue.WINDOW_CHANGED
    assert still_latched.issue is CaptureIssue.WINDOW_CHANGED
    assert (finder.calls, factory.outputs_calls) == calls_after_change
    source.stop()
    source.start()
    recovered = source.read()
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 0
    assert recovered.frame.image is restarted_image
    source.stop()


@pytest.mark.parametrize(
    "window_error,expected_status",
    [
        (GameWindowMinimized("minimized"), CaptureStatus.MINIMIZED),
        (GameWindowOccluded("occluded"), CaptureStatus.OCCLUDED),
    ],
)
def test_minimize_and_occlusion_never_return_cached_frame(
    window_error,
    expected_status,
):
    target = _step6_target()
    image = _step6_image(83)
    camera = _Step6Camera([(image, 1)])
    source, _, _, _, _ = _start_step6_source(
        camera,
        finder_effects=[target, target, target, window_error],
    )

    accepted = source.read()
    hidden = source.read()

    assert accepted.status is CaptureStatus.RUNNING
    assert accepted.frame.image is image
    assert hidden == CaptureSample(expected_status)
    assert hidden.frame is None
    assert camera.read_calls == [{"copy": True, "with_timestamp": True}]
    assert camera.stop_calls == camera.release_calls == 1
    source.stop()
    assert camera.stop_calls == camera.release_calls == 1


@pytest.mark.parametrize("condition", ["minimized", "occluded", "unsupported"])
def test_minimize_occlusion_and_unsupported_output_detach_into_suspended_state(
    condition,
):
    target = _step6_target()
    output = _matching_dxcam_output()
    camera = _Step6Camera([])
    if condition == "minimized":
        finder_effects = [target, GameWindowMinimized("minimized")]
        output_effects = [(output,)]
        expected = CaptureSample(CaptureStatus.MINIMIZED)
    elif condition == "occluded":
        finder_effects = [target, GameWindowOccluded("occluded")]
        output_effects = [(output,)]
        expected = CaptureSample(CaptureStatus.OCCLUDED)
    else:
        finder_effects = [target]
        output_effects = [(output,), (), ()]
        expected = CaptureSample(
            CaptureStatus.LOST,
            issue=CaptureIssue.UNSUPPORTED_OUTPUT,
        )
    finder = _Step6Finder(finder_effects)
    factory = _Step6Factory(output_effects, [camera])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    first = source.read()
    repeated = source.read()

    assert first == expected
    assert repeated == expected
    assert source._state == "suspended"
    assert source._camera is None
    assert source._factory is factory
    assert source._target == target
    assert source._output == output
    assert len(factory.create_calls) == 1
    assert finder.calls == 3
    assert factory.outputs_calls == (3 if condition == "unsupported" else 1)
    assert camera.read_calls == []
    assert camera.stop_calls == camera.release_calls == 1
    source.stop()


def test_suspended_restore_rebuilds_before_emitting_fresh_frame():
    events = []
    target = _step6_target()
    output = _matching_dxcam_output()
    first_image = _step6_image(84)
    restored_image = _step6_image(85)
    old = _Step6Camera([(first_image, 10)], events=events, name="old")
    restored = _Step6Camera(
        [(restored_image, 11)],
        events=events,
        name="restored",
    )
    finder = _Step6Finder(
        [
            target,
            target,
            target,
            GameWindowMinimized("minimized"),
            target,
        ],
        events=events,
    )
    factory = _Step6Factory([(output,)], [old, restored], events=events)
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    first = source.read()
    minimized = source.read()
    recovered = source.read()

    assert first.status is CaptureStatus.RUNNING
    assert first.frame.frame_id == 0
    assert minimized == CaptureSample(CaptureStatus.MINIMIZED)
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 1
    assert recovered.frame.image is restored_image
    names = [event[0] for event in events]
    assert names.index("restored.start") < names.index("restored.get_latest_frame")
    assert old.stop_calls == old.release_calls == 1
    source.stop()


def test_suspended_restore_preserves_id_and_rejects_time_at_or_below_last_accepted():
    target = _step6_target()
    output = _matching_dxcam_output()
    old = _Step6Camera([(_step6_image(86), 10)])
    restored_image = _step6_image(87)
    restored = _Step6Camera(
        [
            (_step6_image(88), 10),
            (_step6_image(93), 9),
            (restored_image, 11),
        ]
    )
    finder = _Step6Finder(
        [
            target,
            target,
            target,
            GameWindowOccluded("occluded"),
            target,
        ]
    )
    factory = _Step6Factory([(output,)], [old, restored])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    first = source.read()
    occluded = source.read()
    equal_time = source.read()
    lower_time = source.read()
    recovered = source.read()

    assert first.frame.frame_id == 0
    assert occluded == CaptureSample(CaptureStatus.OCCLUDED)
    assert equal_time == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.NO_NEW_FRAME,
    )
    assert lower_time == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.NO_NEW_FRAME,
    )
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 1
    assert recovered.frame.monotonic_time == 11
    assert recovered.frame.image is restored_image
    source.stop()


def test_in_session_rebuild_failure_cleans_partial_stays_suspended_and_retries():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1794, -50, 3, 2))
    output = _matching_dxcam_output()
    old = _Step6Camera([])
    broken_dxgi = _FakeDxcamCamera(start_error=RuntimeError("dxgi failed"))
    broken_winrt = _FakeDxcamCamera(start_error=RuntimeError("winrt failed"))
    recovered_image = _step6_image(89, geometry=moved.geometry)
    recovered_camera = _Step6Camera([(recovered_image, 1)])
    finder = _Step6Finder([attached, moved])
    factory = _Step6Factory(
        [(output,)],
        [old, broken_dxgi, broken_winrt, recovered_camera],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    failed = source.read()

    assert failed == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.BACKEND_ERROR,
    )
    assert source._state == "suspended"
    assert source._camera is None
    assert source._target == attached
    assert source._output == output
    assert source._pending_geometry_notice is True
    assert old.stop_calls == old.release_calls == 1
    assert broken_dxgi.stop_calls == broken_dxgi.release_calls == 1
    assert broken_winrt.stop_calls == broken_winrt.release_calls == 1

    restored_notice = source.read()
    assert restored_notice == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.GEOMETRY_CHANGED,
    )
    assert recovered_camera.read_calls == []
    recovered = source.read()
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 0
    assert recovered.frame.image is recovered_image
    assert source._target == moved
    assert source._pending_geometry_notice is False
    source.stop()


def test_start_vs_active_pre_post_and_restore_failures_use_distinct_error_envelopes():
    target = _step6_target()
    output = _matching_dxcam_output()

    failed_start_factory = _Step6Factory(
        [(output,)],
        [RuntimeError("dxgi failed"), RuntimeError("winrt failed")],
    )
    failed_start = DxcamFrameSource(
        window_finder=lambda: target,
        dxcam_factory_loader=lambda: failed_start_factory,
    )
    with pytest.raises(CaptureStartError) as raised:
        failed_start.start()
    assert raised.value.status is CaptureStatus.LOST
    assert raised.value.issue is CaptureIssue.BACKEND_ERROR
    assert failed_start.read() == CaptureSample(CaptureStatus.STOPPED)

    active = _Step6Camera([RuntimeError("atomic backend failure")])
    active_finder = _Step6Finder(
        [target, GameWindowError("pre query failed"), target, target]
    )
    active_factory = _Step6Factory([(output,)], [active])
    active_source = DxcamFrameSource(
        window_finder=active_finder,
        dxcam_factory_loader=lambda: active_factory,
    )
    active_source.start()
    pre_failed = active_source.read()
    post_or_backend_failed = active_source.read()
    assert pre_failed.issue is CaptureIssue.BACKEND_ERROR
    assert post_or_backend_failed.issue is CaptureIssue.BACKEND_ERROR
    assert active_source._camera is active
    assert active.stop_calls == active.release_calls == 0
    active_source.stop()

    post_payload = _step6_image(94)
    post_camera = _Step6Camera([(post_payload, 1)])
    post_finder = _Step6Finder(
        [target, target, GameWindowError("post query failed")]
    )
    post_factory = _Step6Factory([(output,)], [post_camera])
    post_source = DxcamFrameSource(
        window_finder=post_finder,
        dxcam_factory_loader=lambda: post_factory,
    )
    post_source.start()
    post_generation = post_source._camera_generation
    post_failed = post_source.read()
    assert post_failed.issue is CaptureIssue.BACKEND_ERROR
    assert post_failed.frame is None
    assert post_source._camera is post_camera
    assert post_source._camera_generation == post_generation
    assert post_camera.stop_calls == post_camera.release_calls == 0
    post_source.stop()

    old = _Step6Camera([])
    broken_dxgi = _FakeDxcamCamera(start_error=RuntimeError("dxgi restore failed"))
    broken_winrt = _FakeDxcamCamera(start_error=RuntimeError("winrt restore failed"))
    restore_image = _step6_image(95)
    restored_camera = _Step6Camera([(restore_image, 1)])
    restore_finder = _Step6Finder(
        [target, GameWindowMinimized("minimized"), target]
    )
    restore_factory = _Step6Factory(
        [(output,)],
        [old, broken_dxgi, broken_winrt, restored_camera],
    )
    restore_source = DxcamFrameSource(
        window_finder=restore_finder,
        dxcam_factory_loader=lambda: restore_factory,
    )
    restore_source.start()
    assert restore_source.read().status is CaptureStatus.MINIMIZED
    restore_failed = restore_source.read()
    assert restore_failed == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.BACKEND_ERROR,
    )
    assert restore_source._state == "suspended"
    assert restore_source._camera is None
    restored_sample = restore_source.read()
    assert restored_sample.status is CaptureStatus.RUNNING
    assert restored_sample.frame.frame_id == 0
    assert restored_sample.frame.image is restore_image
    restore_source.stop()


def test_geometry_rebuild_preserves_next_monotonic_source_frame_id_and_time_watermark():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1793, -50, 3, 2))
    output = _matching_dxcam_output()
    old = _Step6Camera([(_step6_image(90), 10)])
    valid = _step6_image(92, geometry=moved.geometry)
    replacement = _Step6Camera(
        [
            (_step6_image(91, geometry=moved.geometry), 10),
            (valid, 11),
        ]
    )
    finder = _Step6Finder([attached, attached, attached, moved])
    factory = _Step6Factory([(output,)], [old, replacement])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    first = source.read()
    changed = source.read()
    repeated = source.read()
    recovered = source.read()

    assert first.frame.frame_id == 0
    assert changed.issue is CaptureIssue.GEOMETRY_CHANGED
    assert repeated.issue is CaptureIssue.NO_NEW_FRAME
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 1
    assert recovered.frame.monotonic_time == 11
    assert recovered.frame.image is valid
    source.stop()


class _ConditionWaitProbe:
    def __init__(self, condition):
        self._condition = condition
        self.wait_entered = threading.Event()
        self.predicates = []

    def __enter__(self):
        return self._condition.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self._condition.__exit__(exc_type, exc_value, traceback)

    def wait_for(self, predicate):
        self.predicates.append(predicate)
        self.wait_entered.set()
        return self._condition.wait_for(predicate)

    def notify_all(self):
        return self._condition.notify_all()


class _BlockingCleanupCamera(_Step6Camera):
    def __init__(self, *, stop_error=None, release_error=None, name="camera"):
        super().__init__([], name=name)
        self.stop_entered = threading.Event()
        self.allow_stop = threading.Event()
        self.stop_error = stop_error
        self.release_error = release_error

    def stop(self):
        self.stop_entered.set()
        if not self.allow_stop.wait(timeout=2):
            raise AssertionError("blocked camera cleanup was not released")
        return super().stop()


class _BlockingReadCamera(_Step6Camera):
    def __init__(self, effect, *, name="camera"):
        self._stored_is_capturing = False
        self.capture_state_reads = 0
        self.forbid_capture_state = False
        super().__init__([], name=name)
        self.effect = effect
        self.read_entered = threading.Event()
        self.allow_read = threading.Event()

    @property
    def is_capturing(self):
        self.capture_state_reads += 1
        if self.forbid_capture_state:
            raise AssertionError("stale read inspected is_capturing")
        return self._stored_is_capturing

    @is_capturing.setter
    def is_capturing(self, value):
        self._stored_is_capturing = value

    def get_latest_frame(self, **kwargs):
        self.read_calls.append(kwargs)
        self.read_entered.set()
        if not self.allow_read.wait(timeout=2):
            raise AssertionError("blocked camera read was not released")
        if isinstance(self.effect, BaseException):
            raise self.effect
        return self.effect


class _ExplodingEqualityCamera(_Step6Camera):
    def __eq__(self, other):
        raise AssertionError("camera identity must use 'is', not equality")


class _ReusingCameraFactory(_Step6Factory):
    def __init__(self, output, camera):
        super().__init__([(output,)], [], events=None)
        self.camera = camera

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        self.camera.is_released = False
        self.camera.is_capturing = False
        return self.camera


def _thread_call(callable_, outcomes, errors, done=None):
    try:
        outcomes.append(callable_())
    except Exception as exc:
        errors.append(exc)
    finally:
        if done is not None:
            done.set()


@pytest.mark.parametrize("failure_stage", ["stop", "release"])
def test_stop_is_idempotent_and_releases_in_finally(failure_stage):
    cleanup_error = RuntimeError(f"{failure_stage} cleanup failed")
    old = _BlockingCleanupCamera(
        stop_error=cleanup_error if failure_stage == "stop" else None,
        release_error=cleanup_error if failure_stage == "release" else None,
        name="old",
    )
    replacement = _Step6Camera([])
    target = _step6_target()
    output = _matching_dxcam_output()
    factory = _Step6Factory([(output,)], [old, replacement])
    source = DxcamFrameSource(
        window_finder=lambda: target,
        dxcam_factory_loader=lambda: factory,
    )
    condition_probe = _ConditionWaitProbe(source._lifecycle_condition)
    source._lifecycle_condition = condition_probe
    source.start()
    owner_outcomes = []
    owner_errors = []
    owner_done = threading.Event()
    owner = threading.Thread(
        target=_thread_call,
        args=(source.stop, owner_outcomes, owner_errors, owner_done),
    )
    owner.start()
    assert old.stop_entered.wait(timeout=2)

    waiter_outcomes = []
    waiter_errors = []
    waiter_done = threading.Event()
    waiter_started = threading.Event()

    def second_stop():
        waiter_started.set()
        return source.stop()

    waiter = threading.Thread(
        target=_thread_call,
        args=(second_stop, waiter_outcomes, waiter_errors, waiter_done),
    )
    waiter.start()
    assert waiter_started.wait(timeout=2)
    assert condition_probe.wait_entered.wait(timeout=2)
    state_during_cleanup = source._state
    waiter_returned_early = waiter_done.is_set()
    try:
        assert state_during_cleanup == "stopping"
        assert source.read() == CaptureSample(CaptureStatus.STOPPED)
        with pytest.raises(RuntimeError):
            source.start()
        assert waiter_returned_early is False
    finally:
        old.allow_stop.set()
        owner.join(timeout=2)
        waiter.join(timeout=2)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert owner_outcomes == []
    assert owner_errors == [cleanup_error]
    assert waiter_outcomes == [None]
    assert waiter_errors == []
    assert old.stop_calls == old.release_calls == 1
    assert source._state == "stopped"
    source.stop()
    assert old.stop_calls == old.release_calls == 1
    source.start()
    source.stop()
    assert replacement.stop_calls == replacement.release_calls == 1


@pytest.mark.parametrize(
    "effect",
    [
        (_step6_image(96), 20),
        None,
        RuntimeError("atomic read failed"),
    ],
    ids=["payload", "none", "exception"],
)
def test_stop_race_returns_stopped_before_none_backend_or_window_mapping(effect):
    target = _step6_target()
    output = _matching_dxcam_output()
    camera = _BlockingReadCamera(effect)
    finder = _Step6Finder([target])
    factory = _Step6Factory([(output,)], [camera])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    outcomes = []
    errors = []
    reader = threading.Thread(target=_threaded_read, args=(source, outcomes, errors))
    reader.start()
    assert camera.read_entered.wait(timeout=2)

    source.stop()
    reads_after_stop = camera.capture_state_reads
    camera.forbid_capture_state = True
    try:
        assert reader.is_alive()
        assert finder.calls == 2
        assert factory.outputs_calls == 2
        assert camera.stop_calls == camera.release_calls == 1
    finally:
        camera.allow_read.set()
        reader.join(timeout=2)

    assert not reader.is_alive()
    assert errors == []
    assert outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    assert finder.calls == 2
    assert factory.outputs_calls == 2
    assert camera.capture_state_reads == reads_after_stop
    assert camera.stop_calls == camera.release_calls == 1


def test_stop_racing_blocked_read_discards_inflight_frame():
    camera = _BlockingReadCamera((_step6_image(97), 30))
    source, _, _, _, _ = _start_step6_source(camera)
    outcomes = []
    errors = []
    reader = threading.Thread(target=_threaded_read, args=(source, outcomes, errors))
    reader.start()
    assert camera.read_entered.wait(timeout=2)

    source.stop()
    try:
        assert reader.is_alive()
        assert camera.stop_calls == camera.release_calls == 1
    finally:
        camera.allow_read.set()
        reader.join(timeout=2)

    assert not reader.is_alive()
    assert errors == []
    assert outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    assert camera.stop_calls == camera.release_calls == 1


def test_stale_read_discards_frame_without_cleanup_or_post_detach_camera_call():
    target = _step6_target()
    output = _matching_dxcam_output()
    old = _Step6Camera([(_step6_image(98), 100)])
    blocked_entered = threading.Event()
    allow_blocked = threading.Event()

    def stale_payload():
        blocked_entered.set()
        if not allow_blocked.wait(timeout=2):
            raise AssertionError("stale read was not released")
        return _step6_image(99), 101

    old.read_effects.append(stale_payload)
    fresh_image = _step6_image(100)
    fresh = _Step6Camera([(fresh_image, 1)])
    finder = _Step6Finder([target])
    factory = _Step6Factory([(output,)], [old, fresh])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    assert source.read().frame.monotonic_time == 100
    outcomes = []
    errors = []
    reader = threading.Thread(target=_threaded_read, args=(source, outcomes, errors))
    reader.start()
    assert blocked_entered.wait(timeout=2)

    source.stop()
    source.start()
    calls_before_rejected_read = (finder.calls, factory.outputs_calls, len(fresh.read_calls))
    with pytest.raises(RuntimeError):
        source.read()
    assert (finder.calls, factory.outputs_calls, len(fresh.read_calls)) == calls_before_rejected_read
    try:
        assert reader.is_alive()
        assert old.stop_calls == old.release_calls == 1
    finally:
        allow_blocked.set()
        reader.join(timeout=2)

    assert errors == []
    assert outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    assert old.stop_calls == old.release_calls == 1
    recovered = source.read()
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 0
    assert recovered.frame.monotonic_time == 1
    assert recovered.frame.image is fresh_image
    source.stop()


def test_detach_transfers_exactly_one_cleanup_owner():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1792, -50, 3, 2))
    output = _matching_dxcam_output()
    finder_entered = threading.Event()
    allow_finder = threading.Event()

    def blocked_geometry():
        finder_entered.set()
        if not allow_finder.wait(timeout=2):
            raise AssertionError("geometry finder was not released")
        return moved

    old = _Step6Camera([])
    finder = _Step6Finder([attached, blocked_geometry])
    factory = _Step6Factory([(output,)], [old])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    outcomes = []
    errors = []
    reader = threading.Thread(target=_threaded_read, args=(source, outcomes, errors))
    reader.start()
    assert finder_entered.wait(timeout=2)

    source.stop()
    try:
        assert old.stop_calls == old.release_calls == 1
    finally:
        allow_finder.set()
        reader.join(timeout=2)

    assert errors == []
    assert outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    assert old.stop_calls == old.release_calls == 1
    assert len(factory.create_calls) == 1
    source.stop()
    assert old.stop_calls == old.release_calls == 1


def test_camera_identity_uses_is_and_generation_blocks_same_object_aba():
    target = _step6_target()
    output = _matching_dxcam_output()
    entered = threading.Event()
    allow = threading.Event()

    def stale_payload():
        entered.set()
        if not allow.wait(timeout=2):
            raise AssertionError("ABA read was not released")
        return _step6_image(102), 101

    camera = _ExplodingEqualityCamera(
        [
            (_step6_image(101), 100),
            stale_payload,
            (_step6_image(103), 1),
        ]
    )
    finder = _Step6Finder([target])
    factory = _ReusingCameraFactory(output, camera)
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    assert source.read().frame.frame_id == 0
    outcomes = []
    errors = []
    reader = threading.Thread(target=_threaded_read, args=(source, outcomes, errors))
    reader.start()
    assert entered.wait(timeout=2)

    source.stop()
    source.start()
    try:
        assert reader.is_alive()
    finally:
        allow.set()
        reader.join(timeout=2)

    assert errors == []
    assert outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    recovered = source.read()
    assert recovered.status is CaptureStatus.RUNNING
    assert recovered.frame.frame_id == 0
    assert recovered.frame.monotonic_time == 1
    source.stop()
    assert camera.stop_calls == camera.release_calls == 2


def test_second_concurrent_read_is_rejected_without_state_change():
    camera = _BlockingReadCamera((_step6_image(104), 1))
    source, finder, factory, _, _ = _start_step6_source(camera)
    outcomes = []
    errors = []
    reader = threading.Thread(target=_threaded_read, args=(source, outcomes, errors))
    reader.start()
    assert camera.read_entered.wait(timeout=2)
    before = (
        source._state,
        source._camera_generation,
        finder.calls,
        factory.outputs_calls,
        len(camera.read_calls),
    )

    with pytest.raises(RuntimeError):
        source.read()

    assert (
        source._state,
        source._camera_generation,
        finder.calls,
        factory.outputs_calls,
        len(camera.read_calls),
    ) == before
    camera.allow_read.set()
    reader.join(timeout=2)
    assert not reader.is_alive()
    assert errors == []
    assert outcomes[0].status is CaptureStatus.RUNNING
    source.stop()


def test_start_while_active_is_rejected_without_second_camera():
    camera = _Step6Camera([])
    source, finder, factory, _, _ = _start_step6_source(camera)
    before = (finder.calls, factory.outputs_calls, len(factory.create_calls))

    with pytest.raises(RuntimeError):
        source.start()

    assert (finder.calls, factory.outputs_calls, len(factory.create_calls)) == before
    assert source._camera is camera
    source.stop()


@pytest.mark.parametrize("failure_stage", ["stop", "release"])
def test_cleanup_failure_still_clears_state_and_allows_restart(failure_stage):
    cleanup_error = RuntimeError(f"{failure_stage} failed")
    broken = _FakeDxcamCamera(
        stop_error=cleanup_error if failure_stage == "stop" else None,
        release_error=cleanup_error if failure_stage == "release" else None,
    )
    replacement = _Step6Camera([])
    source, _, _, _, _ = _start_step6_source(
        broken,
        replacement_cameras=[replacement],
    )

    with pytest.raises(RuntimeError) as raised:
        source.stop()
    assert raised.value is cleanup_error
    assert source.read() == CaptureSample(CaptureStatus.STOPPED)
    assert broken.stop_calls == broken.release_calls == 1
    source.stop()
    assert broken.stop_calls == broken.release_calls == 1
    source.start()
    source.stop()
    assert replacement.stop_calls == replacement.release_calls == 1


def test_stop_start_resets_source_frame_id_and_backend_time_epoch():
    first = _Step6Camera([(_step6_image(105), 100)])
    second_image = _step6_image(106)
    second = _Step6Camera([(second_image, 1)])
    source, _, _, _, _ = _start_step6_source(
        first,
        replacement_cameras=[second],
    )

    old_sample = source.read()
    source.stop()
    source.start()
    new_sample = source.read()

    assert old_sample.frame.frame_id == 0
    assert old_sample.frame.monotonic_time == 100
    assert new_sample.status is CaptureStatus.RUNNING
    assert new_sample.frame.frame_id == 0
    assert new_sample.frame.monotonic_time == 1
    assert new_sample.frame.image is second_image
    source.stop()


def test_persistent_suspension_rechecks_window_and_output_without_recreating():
    target = _step6_target()
    output = _matching_dxcam_output()
    old = _Step6Camera([])
    finder = _Step6Finder([target])
    factory = _Step6Factory([(output,), (), ()], [old])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    first = source.read()
    second = source.read()

    expected = CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.UNSUPPORTED_OUTPUT,
    )
    assert first == second == expected
    assert finder.calls == 3
    assert factory.outputs_calls == 3
    assert len(factory.create_calls) == 1
    assert old.stop_calls == old.release_calls == 1
    source.stop()
    assert old.stop_calls == old.release_calls == 1


@pytest.mark.parametrize(
    "foreign_case",
    ["target", "not-found", "ambiguous", "geometry-error"],
)
def test_suspended_foreign_identity_precedes_output_error_and_latches(
    foreign_case,
):
    attached = _step6_target()
    foreign_identity = WindowIdentity(601, 602, 603)
    foreign_before = _step6_target(identity=foreign_identity)
    foreign_after = _step6_target(
        identity=foreign_identity,
        geometry=ClientGeometry(-1791, -50, 3, 2),
    )
    if foreign_case == "target":
        foreign_effect = foreign_before
    elif foreign_case == "not-found":
        foreign_effect = GameWindowNotFound("missing")
    elif foreign_case == "ambiguous":
        foreign_effect = GameWindowAmbiguous("ambiguous")
    else:
        foreign_effect = GameWindowGeometryChanged(
            before=foreign_before,
            after=foreign_after,
        )
    output = _matching_dxcam_output()
    old = _Step6Camera([])
    finder = _Step6Finder(
        [attached, GameWindowMinimized("minimized"), foreign_effect]
    )
    factory = _Step6Factory(
        [(output,), RuntimeError("foreign output refresh failed")],
        [old],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    assert source.read() == CaptureSample(CaptureStatus.MINIMIZED)

    changed = source.read()
    calls_after_latch = (finder.calls, factory.outputs_calls, len(factory.create_calls))
    latched = source.read()

    expected = CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.WINDOW_CHANGED,
    )
    assert changed == latched == expected
    assert (finder.calls, factory.outputs_calls, len(factory.create_calls)) == calls_after_latch
    assert len(factory.create_calls) == 1
    assert source._camera is None
    assert old.stop_calls == old.release_calls == 1
    source.stop()


def test_rebuild_retry_preserves_epoch_and_emits_one_notice_before_new_pixels():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1790, -50, 3, 2))
    output = _matching_dxcam_output()
    old = _Step6Camera([(_step6_image(107), 10)])
    broken_dxgi = _FakeDxcamCamera(start_error=RuntimeError("dxgi failed"))
    broken_winrt = _FakeDxcamCamera(start_error=RuntimeError("winrt failed"))
    accepted_image = _step6_image(110, geometry=moved.geometry)
    recovered = _Step6Camera(
        [
            (_step6_image(108, geometry=moved.geometry), 10),
            (_step6_image(109, geometry=moved.geometry), 9),
            (accepted_image, 11),
        ]
    )
    finder = _Step6Finder([attached, attached, attached, moved])
    factory = _Step6Factory(
        [(output,)],
        [old, broken_dxgi, broken_winrt, recovered],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    first = source.read()
    failed = source.read()
    notice = source.read()
    assert recovered.read_calls == []
    equal_time = source.read()
    lower_time = source.read()
    accepted = source.read()

    assert first.frame.frame_id == 0
    assert first.frame.monotonic_time == 10
    assert failed.issue is CaptureIssue.BACKEND_ERROR
    assert notice.issue is CaptureIssue.GEOMETRY_CHANGED
    assert equal_time.issue is CaptureIssue.NO_NEW_FRAME
    assert lower_time.issue is CaptureIssue.NO_NEW_FRAME
    assert accepted.status is CaptureStatus.RUNNING
    assert accepted.frame.frame_id == 1
    assert accepted.frame.monotonic_time == 11
    assert accepted.frame.image is accepted_image
    source.stop()


def test_suspended_restore_runs_full_ordered_capture_path_after_fresh_start():
    events = []
    target = _step6_target()
    output = _matching_dxcam_output()
    old = _Step6Camera([], events=events, name="old")
    image = _step6_image(111)
    restored = _Step6Camera([(image, 1)], events=events, name="restored")
    finder = _Step6Finder(
        [target, GameWindowOccluded("occluded"), target],
        events=events,
    )
    factory = _Step6Factory(
        [(output,)],
        [old, restored],
        events=events,
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    assert source.read() == CaptureSample(CaptureStatus.OCCLUDED)
    events.clear()

    sample = source.read()

    assert sample.status is CaptureStatus.RUNNING
    assert sample.frame.image is image
    assert [event[0] for event in events] == [
        "finder",
        "outputs",
        "create",
        "restored.start",
        "finder",
        "outputs",
        "restored.get_latest_frame",
        "finder",
        "outputs",
    ]
    source.stop()


@pytest.mark.parametrize("change_kind", ["geometry", "monitor", "output"])
def test_suspended_same_identity_change_emits_notice_before_camera_read(change_kind):
    attached = _step6_target()
    old_output = _matching_dxcam_output()
    if change_kind == "geometry":
        restored_target = _step6_target(
            geometry=ClientGeometry(-1789, -50, 3, 2)
        )
        restored_output = old_output
    elif change_kind == "monitor":
        changed_monitor = MonitorDescriptor(
            72,
            "DISPLAY-SECOND",
            ClientGeometry(-1920, -100, 1920, 1080),
        )
        restored_target = _step6_target(monitor=changed_monitor)
        restored_output = DxcamOutput(
            72,
            "display-second",
            4,
            5,
            changed_monitor.desktop_geometry,
            0,
        )
    else:
        restored_target = attached
        restored_output = _matching_dxcam_output(rotation_angle=90)
    old = _Step6Camera([])
    restored = _Step6Camera(
        [AssertionError("geometry notice must precede camera pixels")]
    )
    finder = _Step6Finder(
        [attached, GameWindowMinimized("minimized"), restored_target]
    )
    factory = _Step6Factory(
        [(old_output,), (restored_output,)],
        [old, restored],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    assert source.read() == CaptureSample(CaptureStatus.MINIMIZED)

    notice = source.read()

    assert notice == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.GEOMETRY_CHANGED,
    )
    assert restored.read_calls == []
    assert source._target == restored_target
    assert source._output == restored_output
    source.stop()


def test_failed_suspended_geometry_restore_persists_notice_across_baseline_return():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1788, -50, 3, 2))
    output = _matching_dxcam_output()
    old = _Step6Camera([])
    broken_dxgi = _FakeDxcamCamera(start_error=RuntimeError("dxgi failed"))
    broken_winrt = _FakeDxcamCamera(start_error=RuntimeError("winrt failed"))
    recovered = _Step6Camera(
        [AssertionError("persisted geometry notice must precede pixels")]
    )
    finder = _Step6Finder(
        [attached, GameWindowMinimized("minimized"), moved, attached]
    )
    factory = _Step6Factory(
        [(output,)],
        [old, broken_dxgi, broken_winrt, recovered],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    assert source.read() == CaptureSample(CaptureStatus.MINIMIZED)

    failed = source.read()

    assert failed == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.BACKEND_ERROR,
    )
    assert source._state == "suspended"
    assert source._target == attached
    assert source._output == output
    assert source._pending_geometry_notice is True

    notice = source.read()

    assert notice == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.GEOMETRY_CHANGED,
    )
    assert recovered.read_calls == []
    source.stop()


def test_suspended_changed_target_output_error_persists_notice_across_baseline_return():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1786, -50, 3, 2))
    output = _matching_dxcam_output()
    old = _Step6Camera([])
    recovered_image = _step6_image(118)
    recovered = _Step6Camera([(recovered_image, 1)])
    finder = _Step6Finder(
        [attached, GameWindowMinimized("minimized"), moved, attached]
    )
    factory = _Step6Factory(
        [
            (output,),
            RuntimeError("transient changed-target output refresh failed"),
            (output,),
        ],
        [old, recovered],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()
    assert source.read() == CaptureSample(CaptureStatus.MINIMIZED)

    failed = source.read()

    assert failed == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.BACKEND_ERROR,
    )
    assert source._state == "suspended"
    assert source._camera is None
    assert source._target == attached
    assert source._output == output
    assert source._pending_geometry_notice is True
    assert old.stop_calls == old.release_calls == 1

    notice = source.read()
    assert notice == CaptureSample(
        CaptureStatus.LOST,
        issue=CaptureIssue.GEOMETRY_CHANGED,
    )
    assert recovered.read_calls == []

    accepted = source.read()
    assert accepted.status is CaptureStatus.RUNNING
    assert accepted.frame.frame_id == 0
    assert accepted.frame.image is recovered_image
    assert source._pending_geometry_notice is False
    source.stop()


def test_stopping_suspended_source_never_cleans_detached_camera_twice():
    old = _Step6Camera([])
    target = _step6_target()
    source, _, _, _, _ = _start_step6_source(
        old,
        finder_effects=[target, GameWindowMinimized("minimized")],
    )
    assert source.read() == CaptureSample(CaptureStatus.MINIMIZED)
    assert old.stop_calls == old.release_calls == 1

    source.stop()
    source.stop()

    assert old.stop_calls == old.release_calls == 1


def test_suspended_output_drift_preserves_baseline_pending_notice_and_retries():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1787, -50, 3, 2))
    output = _matching_dxcam_output()
    old = _Step6Camera([])
    broken_dxgi = _FakeDxcamCamera(start_error=RuntimeError("dxgi failed"))
    broken_winrt = _FakeDxcamCamera(start_error=RuntimeError("winrt failed"))
    recovered = _Step6Camera(
        [AssertionError("pending notice must precede recovered pixels")]
    )
    finder = _Step6Finder([attached, moved])
    factory = _Step6Factory(
        [
            (output,),
            (output,),
            RuntimeError("output refresh failed"),
            (output,),
        ],
        [old, broken_dxgi, broken_winrt, recovered],
    )
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    rebuild_failed = source.read()
    output_failed = source.read()

    assert rebuild_failed.issue is CaptureIssue.BACKEND_ERROR
    assert output_failed.issue is CaptureIssue.BACKEND_ERROR
    assert source._state == "suspended"
    assert source._target == attached
    assert source._output == output
    assert source._pending_geometry_notice is True

    notice = source.read()

    assert notice.issue is CaptureIssue.GEOMETRY_CHANGED
    assert recovered.read_calls == []
    source.stop()


class _BlockingStartCamera(_Step6Camera):
    def __init__(self, *, start_error=None, name="candidate"):
        super().__init__([], name=name)
        self.start_entered = threading.Event()
        self.allow_start = threading.Event()
        self.blocked_start_error = start_error

    def start(self, **kwargs):
        if self.events is not None:
            self.events.append((f"{self.name}.start", kwargs))
        self.start_calls.append(kwargs)
        self.start_entered.set()
        if not self.allow_start.wait(timeout=2):
            raise AssertionError("blocked camera start was not released")
        if self.blocked_start_error is not None:
            raise self.blocked_start_error
        self.is_capturing = True


class _CallbackStartCamera(_Step6Camera):
    def __init__(self, callback, *, name="candidate"):
        super().__init__([], name=name)
        self.callback = callback

    def start(self, **kwargs):
        self.start_calls.append(kwargs)
        self.callback()
        self.is_capturing = True


def test_stop_cancels_blocked_initial_start_and_waits_exact_transition_owner():
    target = _step6_target()
    output = _matching_dxcam_output()
    candidate = _BlockingStartCamera(start_error=RuntimeError("dxgi start failed"))
    replacement = _Step6Camera([])
    factory = _Step6Factory([(output,)], [candidate, replacement])
    source = DxcamFrameSource(
        window_finder=lambda: target,
        dxcam_factory_loader=lambda: factory,
    )
    condition_probe = _ConditionWaitProbe(source._lifecycle_condition)
    source._lifecycle_condition = condition_probe
    initial_lifecycle_token = getattr(source, "_lifecycle_token", None)
    initial_generation = source._camera_generation
    start_outcomes = []
    start_errors = []
    starter = threading.Thread(
        target=_thread_call,
        args=(source.start, start_outcomes, start_errors),
    )
    starter.start()
    assert candidate.start_entered.wait(timeout=2)
    assert type(source._lifecycle_token) is int
    assert source._lifecycle_token > initial_lifecycle_token
    assert source._camera_generation == initial_generation
    with pytest.raises(RuntimeError):
        source.read()
    with pytest.raises(RuntimeError):
        source.start()

    stop_outcomes = []
    stop_errors = []
    stopper = threading.Thread(
        target=_thread_call,
        args=(source.stop, stop_outcomes, stop_errors),
    )
    stopper.start()
    wait_entered = condition_probe.wait_entered.wait(timeout=2)
    state_while_cancelled = source._state
    cleanup_before_owner_resumed = (candidate.stop_calls, candidate.release_calls)
    try:
        assert wait_entered is True
        with source._lifecycle_lock:
            observed_owner = source._transition_owner_token
            replacement_owner = object()
            source._transition_owner_token = replacement_owner
            try:
                assert condition_probe.predicates[-1]() is True
            finally:
                source._transition_owner_token = observed_owner
        assert state_while_cancelled == "stopping"
        assert cleanup_before_owner_resumed == (0, 0)
        assert stopper.is_alive()
    finally:
        candidate.allow_start.set()
        starter.join(timeout=2)
        stopper.join(timeout=2)

    assert not starter.is_alive()
    assert not stopper.is_alive()
    assert start_outcomes == []
    assert len(start_errors) == 1
    assert type(start_errors[0]) is RuntimeError
    assert not isinstance(start_errors[0], CaptureStartError)
    assert "cancel" in str(start_errors[0]).casefold()
    assert stop_outcomes == [None]
    assert stop_errors == []
    assert candidate.stop_calls == candidate.release_calls == 1
    assert len(factory.create_calls) == 1
    assert source._state == "stopped"
    assert source._camera_generation > initial_generation
    assert type(source._lifecycle_token) is int
    if initial_lifecycle_token is not None:
        assert source._lifecycle_token > initial_lifecycle_token

    token_after_cancel = source._lifecycle_token
    generation_after_cancel = source._camera_generation
    source.start()
    assert source._lifecycle_token > token_after_cancel
    assert source._camera_generation > generation_after_cancel
    token_after_restart = source._lifecycle_token
    generation_after_restart = source._camera_generation
    source.stop()
    assert source._lifecycle_token > token_after_restart
    assert source._camera_generation > generation_after_restart
    assert replacement.stop_calls == replacement.release_calls == 1


def test_stop_cancels_blocked_reconfigure_candidate_with_unique_cleanup():
    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1786, -50, 3, 2))
    output = _matching_dxcam_output()
    old = _Step6Camera([])
    candidate = _BlockingStartCamera()
    replacement = _Step6Camera([])
    finder = _Step6Finder([attached, moved])
    factory = _Step6Factory([(output,)], [old, candidate, replacement])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    condition_probe = _ConditionWaitProbe(source._lifecycle_condition)
    source._lifecycle_condition = condition_probe
    source.start()
    read_outcomes = []
    read_errors = []
    reader = threading.Thread(
        target=_threaded_read,
        args=(source, read_outcomes, read_errors),
    )
    reader.start()
    assert candidate.start_entered.wait(timeout=2)
    assert source._state == "reconfiguring"
    assert old.stop_calls == old.release_calls == 1
    assert candidate.stop_calls == candidate.release_calls == 0
    with pytest.raises(RuntimeError):
        source.read()

    stop_outcomes = []
    stop_errors = []
    stopper = threading.Thread(
        target=_thread_call,
        args=(source.stop, stop_outcomes, stop_errors),
    )
    stopper.start()
    wait_entered = condition_probe.wait_entered.wait(timeout=2)
    try:
        assert wait_entered is True
        assert source._state == "stopping"
        assert stopper.is_alive()
        assert candidate.stop_calls == candidate.release_calls == 0
    finally:
        candidate.allow_start.set()
        reader.join(timeout=2)
        stopper.join(timeout=2)

    assert not reader.is_alive()
    assert not stopper.is_alive()
    assert read_errors == []
    assert read_outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    assert stop_errors == []
    assert stop_outcomes == [None]
    assert old.stop_calls == old.release_calls == 1
    assert candidate.stop_calls == candidate.release_calls == 1
    assert source._camera is None
    assert source._state == "stopped"
    assert len(factory.create_calls) == 2

    source.start()
    source.stop()
    assert replacement.stop_calls == replacement.release_calls == 1


def test_stop_cancels_blocked_suspended_restore_candidate_before_restart():
    target = _step6_target()
    output = _matching_dxcam_output()
    old = _Step6Camera([])
    candidate = _BlockingStartCamera()
    replacement = _Step6Camera([])
    finder = _Step6Finder(
        [target, GameWindowMinimized("minimized"), target]
    )
    factory = _Step6Factory([(output,)], [old, candidate, replacement])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    condition_probe = _ConditionWaitProbe(source._lifecycle_condition)
    source._lifecycle_condition = condition_probe
    source.start()
    assert source.read() == CaptureSample(CaptureStatus.MINIMIZED)
    assert old.stop_calls == old.release_calls == 1

    read_outcomes = []
    read_errors = []
    reader = threading.Thread(
        target=_threaded_read,
        args=(source, read_outcomes, read_errors),
    )
    reader.start()
    assert candidate.start_entered.wait(timeout=2)
    assert source._state == "reconfiguring"

    stop_outcomes = []
    stop_errors = []
    stopper = threading.Thread(
        target=_thread_call,
        args=(source.stop, stop_outcomes, stop_errors),
    )
    stopper.start()
    wait_entered = condition_probe.wait_entered.wait(timeout=2)
    try:
        assert wait_entered is True
        assert source._state == "stopping"
        assert candidate.stop_calls == candidate.release_calls == 0
    finally:
        candidate.allow_start.set()
        reader.join(timeout=2)
        stopper.join(timeout=2)

    assert read_errors == []
    assert read_outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    assert stop_errors == []
    assert stop_outcomes == [None]
    assert candidate.stop_calls == candidate.release_calls == 1
    assert source._state == "stopped"
    source.start()
    source.stop()
    assert replacement.stop_calls == replacement.release_calls == 1


def test_transition_owner_callbacks_reenter_stop_without_self_deadlock():
    target = _step6_target()
    output = _matching_dxcam_output()
    source = None
    callback_events = []

    def start_callback():
        source.stop()
        callback_events.append("start-stop-returned")
        with pytest.raises(RuntimeError):
            source.start()
        callback_events.append("nested-start-rejected")

    start_candidate = _CallbackStartCamera(start_callback)
    start_factory = _Step6Factory([(output,)], [start_candidate])
    source = DxcamFrameSource(
        window_finder=lambda: target,
        dxcam_factory_loader=lambda: start_factory,
    )

    with pytest.raises(RuntimeError, match="cancel"):
        source.start()

    assert callback_events == ["start-stop-returned", "nested-start-rejected"]
    assert start_candidate.stop_calls == start_candidate.release_calls == 1
    assert source._state == "stopped"

    attached = _step6_target()
    moved = _step6_target(geometry=ClientGeometry(-1785, -50, 3, 2))
    reconfig_events = []
    source = None

    def old_stop_callback():
        source.stop()
        reconfig_events.append("reconfig-stop-returned")
        with pytest.raises(RuntimeError):
            source.start()
        reconfig_events.append("reconfig-start-rejected")

    old = _Step6Camera([])
    old.stop_callback = old_stop_callback
    unused_candidate = _Step6Camera([])
    finder = _Step6Finder([attached, moved])
    factory = _Step6Factory([(output,)], [old, unused_candidate])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    source.start()

    sample = source.read()

    assert sample == CaptureSample(CaptureStatus.STOPPED)
    assert reconfig_events == [
        "reconfig-stop-returned",
        "reconfig-start-rejected",
    ]
    assert old.stop_calls == old.release_calls == 1
    assert len(factory.create_calls) == 1
    assert source._state == "stopped"

    source = None
    active = _Step6Camera([])

    def active_stop_callback():
        source.stop()
        callback_events.append("cleanup-stop-returned")

    active.stop_callback = active_stop_callback
    active_factory = _Step6Factory([(output,)], [active])
    source = DxcamFrameSource(
        window_finder=lambda: target,
        dxcam_factory_loader=lambda: active_factory,
    )
    source.start()
    source.stop()
    assert callback_events[-1] == "cleanup-stop-returned"
    assert active.stop_calls == active.release_calls == 1

    source = None

    class ReleaseCallbackCamera(_Step6Camera):
        def release(self):
            source.stop()
            callback_events.append("release-stop-returned")
            return super().release()

    release_camera = ReleaseCallbackCamera([])
    release_factory = _Step6Factory([(output,)], [release_camera])
    source = DxcamFrameSource(
        window_finder=lambda: target,
        dxcam_factory_loader=lambda: release_factory,
    )
    source.start()
    source.stop()
    assert callback_events[-1] == "release-stop-returned"
    assert release_camera.stop_calls == release_camera.release_calls == 1


def test_finder_callback_cancellation_skips_loader_and_camera_creation():
    target = _step6_target()
    output = _matching_dxcam_output()
    camera = _Step6Camera([])
    factory = _Step6Factory([(output,)], [camera])
    loader_calls = []
    source = None

    def finder():
        source.stop()
        return target

    def loader():
        loader_calls.append("loader")
        return factory

    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=loader,
    )

    with pytest.raises(RuntimeError, match="cancel"):
        source.start()

    assert loader_calls == []
    assert factory.outputs_calls == 0
    assert factory.create_calls == []
    assert source._state == "stopped"


@pytest.mark.parametrize("detach_kind", ["minimized", "identity"])
def test_stop_waits_reader_owned_detach_cleanup_without_double_release(detach_kind):
    attached = _step6_target()
    foreign = _step6_target(identity=WindowIdentity(701, 702, 703))
    output = _matching_dxcam_output()
    old = _BlockingCleanupCamera(name="old")
    replacement = _Step6Camera([])
    detach_effect = (
        GameWindowMinimized("minimized")
        if detach_kind == "minimized"
        else foreign
    )
    finder = _Step6Finder([attached, detach_effect, attached])
    factory = _Step6Factory([(output,)], [old, replacement])
    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=lambda: factory,
    )
    condition_probe = _ConditionWaitProbe(source._lifecycle_condition)
    source._lifecycle_condition = condition_probe
    source.start()
    read_outcomes = []
    read_errors = []
    reader = threading.Thread(
        target=_threaded_read,
        args=(source, read_outcomes, read_errors),
    )
    reader.start()
    assert old.stop_entered.wait(timeout=2)
    assert source._state == (
        "suspended" if detach_kind == "minimized" else "window_changed"
    )
    observed_cleanup_owner = source._cleanup_owner_token
    assert observed_cleanup_owner is not None

    stop_outcomes = []
    stop_errors = []
    stopper = threading.Thread(
        target=_thread_call,
        args=(source.stop, stop_outcomes, stop_errors),
    )
    stopper.start()
    wait_entered = condition_probe.wait_entered.wait(timeout=2)
    try:
        assert wait_entered is True
        with source._lifecycle_lock:
            replacement_owner = object()
            source._cleanup_owner_token = replacement_owner
            try:
                assert condition_probe.predicates[-1]() is True
            finally:
                source._cleanup_owner_token = observed_cleanup_owner
        assert source._state == "stopping"
        assert stopper.is_alive()
        assert old.stop_calls == 0
        assert old.release_calls == 0
    finally:
        old.allow_stop.set()
        reader.join(timeout=2)
        stopper.join(timeout=2)

    assert not reader.is_alive()
    assert not stopper.is_alive()
    assert read_errors == []
    assert read_outcomes == [CaptureSample(CaptureStatus.STOPPED)]
    assert stop_errors == []
    assert stop_outcomes == [None]
    assert old.stop_calls == old.release_calls == 1
    assert source._state == "stopped"
    source.stop()
    assert old.stop_calls == old.release_calls == 1
    source.start()
    source.stop()
    assert replacement.stop_calls == replacement.release_calls == 1


def test_all_capture_callbacks_run_without_lifecycle_lock():
    target = _step6_target()
    output = _matching_dxcam_output()
    source = None
    callback_names = []

    def assert_unlocked(name):
        acquired = source._lifecycle_lock.acquire(blocking=False)
        assert acquired, f"{name} ran while lifecycle lock was held"
        source._lifecycle_lock.release()
        callback_names.append(name)

    class LockOracleCamera:
        def __init__(self):
            self._capturing = False
            self._released = False

        @property
        def is_capturing(self):
            assert_unlocked("camera.is_capturing")
            return self._capturing

        @property
        def is_released(self):
            assert_unlocked("camera.is_released")
            return self._released

        def start(self, **kwargs):
            assert_unlocked("camera.start")
            self._capturing = True

        def get_latest_frame(self, **kwargs):
            assert_unlocked("camera.get_latest_frame")
            return _step6_image(112), 1

        def stop(self):
            assert_unlocked("camera.stop")
            self._capturing = False

        def release(self):
            assert_unlocked("camera.release")
            self._released = True

    camera = LockOracleCamera()

    class LockOracleFactory:
        def outputs(self):
            assert_unlocked("factory.outputs")
            return (output,)

        def create(self, **kwargs):
            assert_unlocked("factory.create")
            return camera

    factory = LockOracleFactory()

    def finder():
        assert_unlocked("window_finder")
        return target

    def loader():
        assert_unlocked("factory_loader")
        return factory

    source = DxcamFrameSource(
        window_finder=finder,
        dxcam_factory_loader=loader,
    )
    source.start()
    assert source.read().status is CaptureStatus.RUNNING
    source.stop()

    for required in (
        "window_finder",
        "factory_loader",
        "factory.outputs",
        "factory.create",
        "camera.is_released",
        "camera.is_capturing",
        "camera.start",
        "camera.get_latest_frame",
        "camera.stop",
        "camera.release",
    ):
        assert required in callback_names
