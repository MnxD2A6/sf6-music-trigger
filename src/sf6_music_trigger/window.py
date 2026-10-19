"""Typed, fail-closed SF6 window discovery over an injected Win32 facade."""

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import ntpath
from typing import ContextManager, Iterator, Protocol

from .domain import ClientGeometry


_SF6_EXECUTABLE = "StreetFighter6.exe"
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PER_MONITOR_AWARE_V2 = -4
_DWMWA_CLOAKED = 14
_MONITOR_DEFAULTTONULL = 0
_MAX_EXECUTABLE_PATH_CHARS = 32_768

_DWORD = ctypes.c_uint32
_LONG = ctypes.c_int32


class _FILETIME(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", _DWORD),
        ("dwHighDateTime", _DWORD),
    ]


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", _LONG),
        ("top", _LONG),
        ("right", _LONG),
        ("bottom", _LONG),
    ]


class _POINT(ctypes.Structure):
    _fields_ = [("x", _LONG), ("y", _LONG)]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", _DWORD),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", _DWORD),
        ("szDevice", ctypes.c_wchar * 32),
    ]


def _require_int(name: str, value: object, *, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_nonblank_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def _contains(outer: ClientGeometry, inner: ClientGeometry) -> bool:
    return (
        outer.left <= inner.left
        and outer.top <= inner.top
        and inner.left + inner.width <= outer.left + outer.width
        and inner.top + inner.height <= outer.top + outer.height
    )


@dataclass(frozen=True)
class WindowIdentity:
    hwnd: int
    pid: int
    process_created_at_ticks: int

    def __post_init__(self) -> None:
        _require_int("hwnd", self.hwnd, minimum=1)
        _require_int("pid", self.pid, minimum=1)
        _require_int(
            "process_created_at_ticks",
            self.process_created_at_ticks,
            minimum=0,
        )


@dataclass(frozen=True)
class MonitorDescriptor:
    handle: int
    device_name: str
    desktop_geometry: ClientGeometry

    def __post_init__(self) -> None:
        _require_int("handle", self.handle, minimum=1)
        _require_nonblank_string("device_name", self.device_name)
        if not isinstance(self.desktop_geometry, ClientGeometry):
            raise TypeError("desktop_geometry must be ClientGeometry")


@dataclass(frozen=True)
class Sf6WindowTarget:
    identity: WindowIdentity
    geometry: ClientGeometry
    monitor: MonitorDescriptor

    def __post_init__(self) -> None:
        if not isinstance(self.identity, WindowIdentity):
            raise TypeError("identity must be WindowIdentity")
        if not isinstance(self.geometry, ClientGeometry):
            raise TypeError("geometry must be ClientGeometry")
        if not isinstance(self.monitor, MonitorDescriptor):
            raise TypeError("monitor must be MonitorDescriptor")
        if not _contains(self.monitor.desktop_geometry, self.geometry):
            raise ValueError("client geometry must be contained by its monitor")


class Win32Facade(Protocol):
    def per_monitor_v2(self) -> ContextManager[None]: ...

    def enum_top_level_windows(self) -> tuple[int, ...]: ...

    def is_window_visible(self, hwnd: int) -> bool: ...

    def is_window_minimized(self, hwnd: int) -> bool: ...

    def is_window_cloaked(self, hwnd: int) -> bool: ...

    def read_process_identity(self, hwnd: int) -> tuple[WindowIdentity, str]: ...

    def client_geometry(self, hwnd: int) -> ClientGeometry: ...

    def monitor_descriptor(
        self,
        hwnd: int,
        geometry: ClientGeometry,
    ) -> MonitorDescriptor: ...


class GameWindowError(RuntimeError):
    """Base failure for fail-closed window discovery."""


class GameWindowNotFound(GameWindowError):
    """No top-level SF6 executable window was found."""


class GameWindowAmbiguous(GameWindowError):
    """More than one top-level SF6 executable window was found."""


class GameWindowMinimized(GameWindowError):
    """The unique SF6 window is minimized."""


class GameWindowOccluded(GameWindowError):
    """The unique SF6 window is hidden or DWM-cloaked."""


class GameWindowIdentityChanged(GameWindowError):
    before: Sf6WindowTarget
    after: Sf6WindowTarget | None

    def __init__(
        self,
        *,
        before: Sf6WindowTarget,
        after: Sf6WindowTarget | None,
    ) -> None:
        if not isinstance(before, Sf6WindowTarget):
            raise TypeError("before must be Sf6WindowTarget")
        if after is not None and not isinstance(after, Sf6WindowTarget):
            raise TypeError("after must be Sf6WindowTarget or None")
        if after is not None and after.identity == before.identity:
            raise ValueError("identity-change targets must have different identities")
        super().__init__("SF6 window identity changed during discovery")
        self._before = before
        self._after = after

    @property
    def before(self) -> Sf6WindowTarget:
        return self._before

    @property
    def after(self) -> Sf6WindowTarget | None:
        return self._after


class GameWindowGeometryChanged(GameWindowError):
    before: Sf6WindowTarget
    after: Sf6WindowTarget

    def __init__(
        self,
        *,
        before: Sf6WindowTarget,
        after: Sf6WindowTarget,
    ) -> None:
        if not isinstance(before, Sf6WindowTarget):
            raise TypeError("before must be Sf6WindowTarget")
        if not isinstance(after, Sf6WindowTarget):
            raise TypeError("after must be Sf6WindowTarget")
        if after.identity != before.identity:
            raise ValueError("geometry-change targets must have the same identity")
        if after == before:
            raise ValueError("geometry-change targets must differ")
        super().__init__("SF6 window geometry or monitor changed during discovery")
        self._before = before
        self._after = after

    @property
    def before(self) -> Sf6WindowTarget:
        return self._before

    @property
    def after(self) -> Sf6WindowTarget:
        return self._after


def _configure_function(function, *, argtypes, restype):
    function.argtypes = argtypes
    function.restype = restype
    return function


class _RawWin32Api:
    """Lazily loaded native functions with fixed pointer-width signatures."""

    def __init__(self, user32, kernel32, dwmapi) -> None:
        callback_factory = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
        self.window_enum_callback_type = callback_factory(
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_ssize_t,
        )

        self.SetThreadDpiAwarenessContext = _configure_function(
            user32.SetThreadDpiAwarenessContext,
            argtypes=[ctypes.c_void_p],
            restype=ctypes.c_void_p,
        )
        self.EnumWindows = _configure_function(
            user32.EnumWindows,
            argtypes=[self.window_enum_callback_type, ctypes.c_ssize_t],
            restype=ctypes.c_int32,
        )
        self.IsWindowVisible = _configure_function(
            user32.IsWindowVisible,
            argtypes=[ctypes.c_void_p],
            restype=ctypes.c_int32,
        )
        self.IsIconic = _configure_function(
            user32.IsIconic,
            argtypes=[ctypes.c_void_p],
            restype=ctypes.c_int32,
        )
        self.GetWindowThreadProcessId = _configure_function(
            user32.GetWindowThreadProcessId,
            argtypes=[ctypes.c_void_p, ctypes.POINTER(_DWORD)],
            restype=_DWORD,
        )
        self.GetClientRect = _configure_function(
            user32.GetClientRect,
            argtypes=[ctypes.c_void_p, ctypes.POINTER(_RECT)],
            restype=ctypes.c_int32,
        )
        self.ClientToScreen = _configure_function(
            user32.ClientToScreen,
            argtypes=[ctypes.c_void_p, ctypes.POINTER(_POINT)],
            restype=ctypes.c_int32,
        )
        self.MonitorFromWindow = _configure_function(
            user32.MonitorFromWindow,
            argtypes=[ctypes.c_void_p, _DWORD],
            restype=ctypes.c_void_p,
        )
        self.GetMonitorInfoW = _configure_function(
            user32.GetMonitorInfoW,
            argtypes=[ctypes.c_void_p, ctypes.POINTER(_MONITORINFOEXW)],
            restype=ctypes.c_int32,
        )

        self.OpenProcess = _configure_function(
            kernel32.OpenProcess,
            argtypes=[_DWORD, ctypes.c_int32, _DWORD],
            restype=ctypes.c_void_p,
        )
        self.QueryFullProcessImageNameW = _configure_function(
            kernel32.QueryFullProcessImageNameW,
            argtypes=[
                ctypes.c_void_p,
                _DWORD,
                ctypes.c_wchar_p,
                ctypes.POINTER(_DWORD),
            ],
            restype=ctypes.c_int32,
        )
        self.GetProcessTimes = _configure_function(
            kernel32.GetProcessTimes,
            argtypes=[
                ctypes.c_void_p,
                ctypes.POINTER(_FILETIME),
                ctypes.POINTER(_FILETIME),
                ctypes.POINTER(_FILETIME),
                ctypes.POINTER(_FILETIME),
            ],
            restype=ctypes.c_int32,
        )
        self.CloseHandle = _configure_function(
            kernel32.CloseHandle,
            argtypes=[ctypes.c_void_p],
            restype=ctypes.c_int32,
        )
        self.DwmGetWindowAttribute = _configure_function(
            dwmapi.DwmGetWindowAttribute,
            argtypes=[ctypes.c_void_p, _DWORD, ctypes.POINTER(_DWORD), _DWORD],
            restype=_LONG,
        )


def _load_raw_win32_api() -> _RawWin32Api:
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise GameWindowError("native Win32 APIs are unavailable")
    try:
        user32 = win_dll("user32", use_last_error=True)
        kernel32 = win_dll("kernel32", use_last_error=True)
        dwmapi = win_dll("dwmapi", use_last_error=True)
        return _RawWin32Api(user32, kernel32, dwmapi)
    except Exception as exc:
        raise GameWindowError("native Win32 API loading failed") from exc


def _pointer_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, ctypes.c_void_p):
        return value.value
    if type(value) is int:
        return value
    return None


def _require_positive_native_handle(name: str, value: object) -> int:
    pointer = _pointer_value(value)
    if pointer is None or pointer <= 0:
        raise GameWindowError(f"{name} returned an invalid handle")
    return pointer


def _native_succeeded(value: object) -> bool:
    if type(value) in (bool, int):
        return bool(value)
    scalar_value = getattr(value, "value", None)
    if type(scalar_value) is int:
        return bool(scalar_value)
    raise GameWindowError("native API returned malformed status")


def _try_native_call(function, *args) -> tuple[bool, object | None]:
    try:
        return True, function(*args)
    except Exception:
        return False, None


def _query_executable_basename(api: object, raw_handle: object) -> str:
    buffer = ctypes.create_unicode_buffer(_MAX_EXECUTABLE_PATH_CHARS)
    path_size = _DWORD(_MAX_EXECUTABLE_PATH_CHARS)
    queried_path: str | None = None
    try:
        called, result = _try_native_call(
            api.QueryFullProcessImageNameW,
            raw_handle,
            0,
            buffer,
            ctypes.byref(path_size),
        )
        if not called:
            raise GameWindowError("process executable query failed")
        try:
            succeeded = _native_succeeded(result)
        except GameWindowError:
            succeeded = False
        if not succeeded:
            raise GameWindowError("process executable query failed")
        if not 0 < path_size.value < _MAX_EXECUTABLE_PATH_CHARS:
            raise GameWindowError("process executable query returned invalid data")
        queried_path = buffer.value
        if len(queried_path) != path_size.value:
            raise GameWindowError("process executable query returned invalid data")
        executable_name = ntpath.basename(queried_path)
        if not executable_name.strip():
            raise GameWindowError("process executable query returned invalid data")
        return executable_name
    finally:
        ctypes.memset(ctypes.addressof(buffer), 0, ctypes.sizeof(buffer))
        del queried_path
        del buffer


class _CtypesWin32Facade:
    """Read-only ctypes adapter; its raw API is injectable for offline tests."""

    def __init__(self, api: object) -> None:
        if api is None:
            raise TypeError("api must not be None")
        self._api = api

    @contextmanager
    def per_monitor_v2(self) -> Iterator[None]:
        requested = ctypes.c_void_p(_PER_MONITOR_AWARE_V2)
        try:
            previous = self._api.SetThreadDpiAwarenessContext(requested)
        except Exception as exc:
            raise GameWindowError("thread DPI context setup failed") from exc
        if _pointer_value(previous) in (None, 0):
            raise GameWindowError("thread DPI context setup failed")

        try:
            yield
        finally:
            try:
                restored = self._api.SetThreadDpiAwarenessContext(previous)
            except Exception as exc:
                raise GameWindowError("thread DPI context restoration failed") from exc
            if _pointer_value(restored) in (None, 0):
                raise GameWindowError("thread DPI context restoration failed")

    def enum_top_level_windows(self) -> tuple[int, ...]:
        handles: list[int] = []
        invalid_callback_value = False

        def collect(hwnd, _lparam):
            nonlocal invalid_callback_value
            try:
                handles.append(_require_positive_native_handle("EnumWindows", hwnd))
            except GameWindowError:
                invalid_callback_value = True
                return False
            return True

        callback_type = getattr(self._api, "window_enum_callback_type", None)
        callback = callback_type(collect) if callback_type is not None else collect
        try:
            succeeded = self._api.EnumWindows(callback, 0)
        except Exception as exc:
            raise GameWindowError("top-level window enumeration failed") from exc
        if not _native_succeeded(succeeded) or invalid_callback_value:
            raise GameWindowError("top-level window enumeration failed")
        return tuple(handles)

    def is_window_visible(self, hwnd: int) -> bool:
        _require_int("hwnd", hwnd, minimum=1)
        try:
            return _native_succeeded(self._api.IsWindowVisible(hwnd))
        except GameWindowError:
            raise
        except Exception as exc:
            raise GameWindowError("window visibility query failed") from exc

    def is_window_minimized(self, hwnd: int) -> bool:
        _require_int("hwnd", hwnd, minimum=1)
        try:
            return _native_succeeded(self._api.IsIconic(hwnd))
        except GameWindowError:
            raise
        except Exception as exc:
            raise GameWindowError("window minimized query failed") from exc

    def is_window_cloaked(self, hwnd: int) -> bool:
        _require_int("hwnd", hwnd, minimum=1)
        cloaked = _DWORD()
        try:
            result = self._api.DwmGetWindowAttribute(
                hwnd,
                _DWMWA_CLOAKED,
                ctypes.byref(cloaked),
                ctypes.sizeof(cloaked),
            )
        except Exception as exc:
            raise GameWindowError("DWM cloaked query failed") from exc
        result_value = getattr(result, "value", result)
        if type(result_value) is not int or result_value != 0:
            raise GameWindowError("DWM cloaked query failed")
        return cloaked.value != 0

    def read_process_identity(self, hwnd: int) -> tuple[WindowIdentity, str]:
        _require_int("hwnd", hwnd, minimum=1)
        pid = _DWORD()
        try:
            thread_id = self._api.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        except Exception as exc:
            raise GameWindowError("window process query failed") from exc
        if not _native_succeeded(thread_id) or pid.value == 0:
            raise GameWindowError("window process query failed")

        try:
            raw_handle = self._api.OpenProcess(
                _PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                pid.value,
            )
        except Exception as exc:
            raise GameWindowError("process open failed") from exc
        _require_positive_native_handle("OpenProcess", raw_handle)

        try:
            executable_name = _query_executable_basename(self._api, raw_handle)

            creation = _FILETIME()
            exit_time = _FILETIME()
            kernel_time = _FILETIME()
            user_time = _FILETIME()
            times_called, times_ok = _try_native_call(
                self._api.GetProcessTimes,
                raw_handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            )
            if not times_called or not _native_succeeded(times_ok):
                raise GameWindowError("process creation-time query failed")
            creation_ticks = (
                int(creation.dwHighDateTime) << 32
            ) | int(creation.dwLowDateTime)
            return WindowIdentity(hwnd, int(pid.value), creation_ticks), executable_name
        finally:
            close_called, close_ok = _try_native_call(
                self._api.CloseHandle,
                raw_handle,
            )
            if not close_called or not _native_succeeded(close_ok):
                raise GameWindowError("process handle close failed")

    def client_geometry(self, hwnd: int) -> ClientGeometry:
        _require_int("hwnd", hwnd, minimum=1)
        rect = _RECT()
        try:
            rect_ok = self._api.GetClientRect(hwnd, ctypes.byref(rect))
        except Exception as exc:
            raise GameWindowError("client rectangle query failed") from exc
        if not _native_succeeded(rect_ok):
            raise GameWindowError("client rectangle query failed")

        top_left = _POINT(rect.left, rect.top)
        bottom_right = _POINT(rect.right, rect.bottom)
        for label, point in (("top-left", top_left), ("bottom-right", bottom_right)):
            try:
                converted = self._api.ClientToScreen(hwnd, ctypes.byref(point))
            except Exception as exc:
                raise GameWindowError(
                    f"client {label} desktop conversion failed"
                ) from exc
            if not _native_succeeded(converted):
                raise GameWindowError(f"client {label} desktop conversion failed")
        try:
            return ClientGeometry(
                int(top_left.x),
                int(top_left.y),
                int(bottom_right.x - top_left.x),
                int(bottom_right.y - top_left.y),
            )
        except (TypeError, ValueError) as exc:
            raise GameWindowError("client rectangle is invalid") from exc

    def monitor_descriptor(
        self,
        hwnd: int,
        geometry: ClientGeometry,
    ) -> MonitorDescriptor:
        _require_int("hwnd", hwnd, minimum=1)
        if not isinstance(geometry, ClientGeometry):
            raise GameWindowError("client geometry is invalid")
        try:
            raw_monitor = self._api.MonitorFromWindow(
                hwnd,
                _MONITOR_DEFAULTTONULL,
            )
        except Exception as exc:
            raise GameWindowError("window monitor query failed") from exc
        monitor_handle = _require_positive_native_handle(
            "MonitorFromWindow",
            raw_monitor,
        )

        info = _MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        try:
            monitor_ok = self._api.GetMonitorInfoW(
                raw_monitor,
                ctypes.byref(info),
            )
        except Exception as exc:
            raise GameWindowError("monitor descriptor query failed") from exc
        if not _native_succeeded(monitor_ok):
            raise GameWindowError("monitor descriptor query failed")

        try:
            desktop_geometry = ClientGeometry(
                int(info.rcMonitor.left),
                int(info.rcMonitor.top),
                int(info.rcMonitor.right - info.rcMonitor.left),
                int(info.rcMonitor.bottom - info.rcMonitor.top),
            )
            return MonitorDescriptor(
                monitor_handle,
                str(info.szDevice),
                desktop_geometry,
            )
        except (TypeError, ValueError) as exc:
            raise GameWindowError("monitor descriptor is invalid") from exc


def _make_default_win32_facade() -> Win32Facade:
    return _CtypesWin32Facade(_load_raw_win32_api())


def _require_bool_result(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise GameWindowError(f"{name} returned a non-boolean value")
    return value


def _read_identity(
    win32: Win32Facade,
    hwnd: int,
) -> tuple[WindowIdentity, str]:
    result = win32.read_process_identity(hwnd)
    if not isinstance(result, tuple) or len(result) != 2:
        raise GameWindowError("process identity query returned malformed data")
    identity, executable_path = result
    if not isinstance(identity, WindowIdentity):
        raise GameWindowError("process identity query returned an invalid identity")
    if identity.hwnd != hwnd:
        raise GameWindowError("process identity query returned a different window")
    if not isinstance(executable_path, str) or not executable_path.strip():
        raise GameWindowError("process identity query returned an invalid executable")
    return identity, ntpath.basename(executable_path)


def _target_after_status_checks(
    win32: Win32Facade,
    hwnd: int,
    identity: WindowIdentity,
) -> Sf6WindowTarget:
    if _require_bool_result("is_window_minimized", win32.is_window_minimized(hwnd)):
        raise GameWindowMinimized("SF6 window is minimized")
    if not _require_bool_result("is_window_visible", win32.is_window_visible(hwnd)):
        raise GameWindowOccluded("SF6 window is hidden")
    if _require_bool_result("is_window_cloaked", win32.is_window_cloaked(hwnd)):
        raise GameWindowOccluded("SF6 window is DWM-cloaked")

    geometry = win32.client_geometry(hwnd)
    if not isinstance(geometry, ClientGeometry):
        raise GameWindowError("client geometry query returned invalid data")
    monitor = win32.monitor_descriptor(hwnd, geometry)
    return Sf6WindowTarget(identity, geometry, monitor)


def _diagnostic_target(
    win32: Win32Facade,
    hwnd: int,
    identity: WindowIdentity,
) -> Sf6WindowTarget | None:
    try:
        geometry = win32.client_geometry(hwnd)
        if not isinstance(geometry, ClientGeometry):
            return None
        monitor = win32.monitor_descriptor(hwnd, geometry)
        return Sf6WindowTarget(identity, geometry, monitor)
    except Exception:
        return None


def _find_sf6_window_target(win32: Win32Facade) -> Sf6WindowTarget:
    windows = win32.enum_top_level_windows()
    if not isinstance(windows, tuple):
        raise GameWindowError("window enumeration returned malformed data")

    candidates: list[tuple[int, WindowIdentity]] = []
    for hwnd in windows:
        if type(hwnd) is not int or hwnd <= 0:
            raise GameWindowError("window enumeration returned an invalid handle")
        try:
            identity, executable_name = _read_identity(win32, hwnd)
        except GameWindowError:
            # Unrelated protected/elevated top-level windows are common on a
            # Windows desktop.  They cannot be SF6 candidates until their
            # executable identity has been read, so ignore them here.  The
            # selected SF6 window is still read and validated twice below.
            continue
        if executable_name.casefold() == _SF6_EXECUTABLE.casefold():
            candidates.append((hwnd, identity))

    if not candidates:
        raise GameWindowNotFound("StreetFighter6.exe window was not found")
    if len(candidates) != 1:
        visible_candidates = [
            candidate
            for candidate in candidates
            if win32.is_window_visible(candidate[0])
        ]
        if len(visible_candidates) != 1:
            raise GameWindowAmbiguous("multiple StreetFighter6.exe windows were found")
        candidates = visible_candidates

    hwnd, first_identity = candidates[0]
    first = _target_after_status_checks(win32, hwnd, first_identity)

    try:
        second_identity, second_executable = _read_identity(win32, hwnd)
    except Exception as exc:
        raise GameWindowIdentityChanged(before=first, after=None) from exc

    if second_identity != first.identity:
        after = None
        if second_executable.casefold() == _SF6_EXECUTABLE.casefold():
            after = _diagnostic_target(win32, hwnd, second_identity)
        raise GameWindowIdentityChanged(before=first, after=after)
    if second_executable.casefold() != _SF6_EXECUTABLE.casefold():
        raise GameWindowIdentityChanged(before=first, after=None)

    second = _target_after_status_checks(win32, hwnd, second_identity)
    if second != first:
        raise GameWindowGeometryChanged(before=first, after=second)
    return second


def find_sf6_window_target(
    *,
    win32: Win32Facade | None = None,
) -> Sf6WindowTarget:
    """Return a twice-validated SF6 client target from an injected facade."""

    try:
        if win32 is None:
            win32 = _make_default_win32_facade()
        with win32.per_monitor_v2():
            return _find_sf6_window_target(win32)
    except GameWindowError:
        raise
    except Exception as exc:
        raise GameWindowError("SF6 window discovery failed") from exc


def find_sf6_client_geometry(
    *,
    win32: Win32Facade | None = None,
) -> ClientGeometry:
    return find_sf6_window_target(win32=win32).geometry
