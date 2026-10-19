"""Typed capture boundaries that remain side-effect free until explicit start."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib import metadata
import math
from fractions import Fraction
from numbers import Rational, Real
import re
import threading
from typing import Callable, Literal, Protocol

import numpy as np

from .domain import (
    CaptureIssue,
    CaptureSample,
    CaptureStatus,
    CapturedFrame,
    ClientGeometry,
)
from .window import (
    GameWindowAmbiguous,
    GameWindowError,
    GameWindowGeometryChanged,
    GameWindowIdentityChanged,
    GameWindowMinimized,
    GameWindowNotFound,
    GameWindowOccluded,
    Sf6WindowTarget,
    find_sf6_window_target,
)


@dataclass(frozen=True)
class DxcamOutput:
    monitor_handle: int
    device_name: str
    device_index: int
    output_index: int
    desktop_geometry: ClientGeometry
    rotation_angle: int

    def __post_init__(self) -> None:
        if type(self.monitor_handle) is not int:
            raise TypeError("monitor_handle must be an integer")
        if self.monitor_handle <= 0:
            raise ValueError("monitor_handle must be positive")
        if not isinstance(self.device_name, str):
            raise TypeError("device_name must be a string")
        if not self.device_name.strip():
            raise ValueError("device_name must be nonblank")
        for field_name in ("device_index", "output_index"):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        if not isinstance(self.desktop_geometry, ClientGeometry):
            raise TypeError("desktop_geometry must be ClientGeometry")
        if type(self.rotation_angle) is not int:
            raise TypeError("rotation_angle must be an integer")
        if self.rotation_angle not in (0, 90, 180, 270):
            raise ValueError("rotation_angle must be 0, 90, 180, or 270")


class DxcamCamera(Protocol):
    @property
    def is_capturing(self) -> bool: ...

    @property
    def is_released(self) -> bool: ...

    def start(
        self,
        *,
        region: tuple[int, int, int, int],
        target_fps: int,
        video_mode: bool,
    ) -> None: ...

    def get_latest_frame(
        self,
        *,
        copy: bool,
        with_timestamp: Literal[True],
    ) -> tuple[np.ndarray, Real] | None: ...

    def stop(self) -> None: ...

    def release(self) -> None: ...


class DxcamFactory(Protocol):
    def outputs(self) -> tuple[DxcamOutput, ...]: ...

    def create(
        self,
        *,
        device_idx: int,
        output_idx: int,
        region: tuple[int, int, int, int],
        output_color: str,
        max_buffer_len: int,
        backend: str,
        processor_backend: str,
    ) -> DxcamCamera: ...


class FrameSource(Protocol):
    def start(self) -> None: ...

    def read(self) -> CaptureSample: ...

    def stop(self) -> None: ...


class CaptureStartError(RuntimeError):
    status: CaptureStatus
    issue: CaptureIssue | None

    def __init__(
        self,
        status: CaptureStatus,
        *,
        issue: CaptureIssue | None,
    ) -> None:
        if not isinstance(status, CaptureStatus):
            raise TypeError("status must be CaptureStatus")
        if status in (CaptureStatus.MINIMIZED, CaptureStatus.OCCLUDED):
            if issue is not None:
                raise ValueError("window-state start errors cannot carry an issue")
        elif status is CaptureStatus.LOST:
            if not isinstance(issue, CaptureIssue):
                raise TypeError("LOST start errors require CaptureIssue")
        else:
            raise ValueError("start errors must be MINIMIZED, OCCLUDED, or LOST")
        super().__init__(
            f"capture start failed: {status.value}"
            + (f" ({issue.value})" if issue is not None else "")
        )
        self._status = status
        self._issue = issue

    @property
    def status(self) -> CaptureStatus:
        return self._status

    @property
    def issue(self) -> CaptureIssue | None:
        return self._issue


class _DxcamAdapterError(RuntimeError):
    """DXcam 0.3 boundary failed without exposing dependency internals."""


class _UnsupportedDxcamOutput(RuntimeError):
    """Structured output metadata has no unique physical match."""


@dataclass(frozen=True)
class _DxcamReservation:
    adapter_token: object
    module: object


@dataclass(frozen=True)
class _DxcamOwnership:
    adapter_token: object
    module: object
    key: tuple[int, int, int, str]
    camera: object


_DXCAM_COORDINATOR_LOCK = threading.Lock()
_DXCAM_COORDINATOR: dict[
    tuple[int, int, int, str],
    _DxcamReservation | _DxcamOwnership,
] = {}
_MISSING = object()


def _actual_int(value: object) -> int:
    if type(value) is not int:
        raise _DxcamAdapterError("DXcam integer field is invalid")
    return value


class _Dxcam03FactoryAdapter:
    """Narrow adapter for the pinned structured DXcam 0.3 API."""

    def __init__(self, module: object) -> None:
        self._module = module
        self._token = object()
        self._raw_factory: object = _MISSING
        self._raw_factory_lock = threading.Lock()
        self._owned_cameras: dict[
            int,
            tuple[object, tuple[int, int, int, str]],
        ] = {}

    def outputs(self) -> tuple[DxcamOutput, ...]:
        try:
            raw_factory = self._get_raw_factory()
            device_outputs = raw_factory.outputs
        except Exception as exc:
            raise _DxcamAdapterError("DXcam output enumeration failed") from exc
        if type(device_outputs) is not list:
            raise _DxcamAdapterError("DXcam output collection is invalid")

        snapshots: list[DxcamOutput] = []
        for device_index, outputs in enumerate(device_outputs):
            if type(outputs) is not list:
                raise _DxcamAdapterError("DXcam device output collection is invalid")
            for output_index, raw_output in enumerate(outputs):
                snapshots.extend(
                    self._snapshot_output(raw_output, device_index, output_index)
                )
        return tuple(snapshots)

    def create(
        self,
        *,
        device_idx: int,
        output_idx: int,
        region: tuple[int, int, int, int],
        output_color: str,
        max_buffer_len: int,
        backend: str,
        processor_backend: str,
    ) -> DxcamCamera:
        device_idx = self._validate_index("device_idx", device_idx)
        output_idx = self._validate_index("output_idx", output_idx)
        region = self._validate_region(region)
        if not isinstance(output_color, str) or not output_color:
            raise _DxcamAdapterError("output_color is invalid")
        if type(max_buffer_len) is not int or max_buffer_len <= 0:
            raise _DxcamAdapterError("max_buffer_len is invalid")
        if not isinstance(backend, str):
            raise _DxcamAdapterError("backend is invalid")
        normalized_backend = backend.casefold()
        if normalized_backend not in ("dxgi", "winrt"):
            raise _DxcamAdapterError("backend is invalid")
        if not isinstance(processor_backend, str) or not processor_backend:
            raise _DxcamAdapterError("processor_backend is invalid")

        coordinator_key = (
            id(self._module),
            device_idx,
            output_idx,
            normalized_backend,
        )
        registry_key = (device_idx, output_idx, normalized_backend)
        cleanup_camera: object | None = None
        failure: _DxcamAdapterError | None = None
        camera: object = _MISSING

        with _DXCAM_COORDINATOR_LOCK:
            if coordinator_key in _DXCAM_COORDINATOR:
                raise _DxcamAdapterError("DXcam camera key is already reserved")
            reservation = _DxcamReservation(self._token, self._module)
            _DXCAM_COORDINATOR[coordinator_key] = reservation
            registry: object = _MISSING
            previous: object = _MISSING
            create_called = False
            create_returned = False
            try:
                raw_factory = self._get_raw_factory()
                registry = raw_factory._camera_instances
                if not callable(getattr(registry, "get", None)):
                    raise _DxcamAdapterError("DXcam camera registry is invalid")
                previous = registry.get(registry_key, _MISSING)
                if previous is not _MISSING:
                    released = self._camera_bool(previous, "is_released")
                    if not released:
                        raise _DxcamAdapterError(
                            "DXcam camera key is owned outside this adapter"
                        )

                create_method = self._module.create
                if not callable(create_method):
                    raise _DxcamAdapterError("DXcam create API is invalid")
                create_called = True
                camera = create_method(
                    device_idx=device_idx,
                    output_idx=output_idx,
                    region=region,
                    output_color=output_color,
                    max_buffer_len=max_buffer_len,
                    backend=normalized_backend,
                    processor_backend=processor_backend,
                )
                create_returned = True
                registered = registry.get(registry_key, _MISSING)
                if registered is not camera:
                    raise _DxcamAdapterError(
                        "DXcam return and registry identities differ"
                    )
                if previous is not _MISSING and registered is previous:
                    raise _DxcamAdapterError(
                        "DXcam did not replace a released camera"
                    )
                self._validate_camera(camera)

                ownership = _DxcamOwnership(
                    self._token,
                    self._module,
                    coordinator_key,
                    camera,
                )
                _DXCAM_COORDINATOR[coordinator_key] = ownership
                self._owned_cameras[id(camera)] = (camera, coordinator_key)
            except Exception as exc:
                try:
                    registered = (
                        registry.get(registry_key, _MISSING)
                        if registry is not _MISSING
                        else _MISSING
                    )
                except Exception:
                    registered = _MISSING
                if create_returned:
                    if (
                        registered is camera
                        and camera is not previous
                        and camera is not _MISSING
                    ):
                        cleanup_camera = camera
                elif (
                    create_called
                    and registered is not _MISSING
                    and registered is not previous
                ):
                    cleanup_camera = registered
                if _DXCAM_COORDINATOR.get(coordinator_key) is reservation:
                    del _DXCAM_COORDINATOR[coordinator_key]
                if isinstance(exc, _DxcamAdapterError):
                    failure = exc
                else:
                    failure = _DxcamAdapterError("DXcam camera creation failed")
                    failure.__cause__ = exc

        if failure is not None:
            if cleanup_camera is not None:
                try:
                    _stop_and_release_camera(cleanup_camera)
                except Exception:
                    pass
            raise failure
        if camera is _MISSING:
            raise _DxcamAdapterError("DXcam camera creation failed")
        return camera

    def _get_raw_factory(self) -> object:
        """Return the one DXcam 0.3 factory owned by this adapter."""
        with self._raw_factory_lock:
            if self._raw_factory is _MISSING:
                self._raw_factory = self._module.DXFactory()
            return self._raw_factory

    def _cleanup_owned_camera(self, camera: object) -> bool:
        claimed: object | None = None
        with _DXCAM_COORDINATOR_LOCK:
            local = self._owned_cameras.get(id(camera))
            if local is None or local[0] is not camera:
                return False
            coordinator_key = local[1]
            record = _DXCAM_COORDINATOR.get(coordinator_key)
            if (
                not isinstance(record, _DxcamOwnership)
                or record.adapter_token is not self._token
                or record.camera is not camera
                or record.key != coordinator_key
            ):
                return False
            del _DXCAM_COORDINATOR[coordinator_key]
            del self._owned_cameras[id(camera)]
            claimed = camera
        _stop_and_release_camera(claimed)
        return True

    @staticmethod
    def _validate_index(name: str, value: object) -> int:
        if type(value) is not int:
            raise _DxcamAdapterError(f"{name} is invalid")
        if value < 0:
            raise _DxcamAdapterError(f"{name} is invalid")
        return value

    @staticmethod
    def _validate_region(value: object) -> tuple[int, int, int, int]:
        if not isinstance(value, tuple) or len(value) != 4:
            raise _DxcamAdapterError("region is invalid")
        if any(type(item) is not int for item in value):
            raise _DxcamAdapterError("region is invalid")
        left, top, right, bottom = value
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise _DxcamAdapterError("region is invalid")
        return value

    @staticmethod
    def _camera_bool(camera: object, name: str) -> bool:
        try:
            value = getattr(camera, name)
        except Exception as exc:
            raise _DxcamAdapterError(f"DXcam camera {name} is unavailable") from exc
        if type(value) is not bool:
            raise _DxcamAdapterError(f"DXcam camera {name} is invalid")
        return value

    @classmethod
    def _validate_camera(cls, camera: object) -> None:
        if cls._camera_bool(camera, "is_released"):
            raise _DxcamAdapterError("DXcam returned a released camera")
        cls._camera_bool(camera, "is_capturing")
        for method_name in ("start", "get_latest_frame", "stop", "release"):
            try:
                method = getattr(camera, method_name)
            except Exception as exc:
                raise _DxcamAdapterError(
                    f"DXcam camera {method_name} is unavailable"
                ) from exc
            if not callable(method):
                raise _DxcamAdapterError(
                    f"DXcam camera {method_name} is invalid"
                )

    @staticmethod
    def _snapshot_output(
        raw_output: object,
        device_index: int,
        output_index: int,
    ) -> tuple[DxcamOutput, ...]:
        try:
            update_desc = raw_output.update_desc
            if not callable(update_desc):
                raise TypeError("update_desc is not callable")
            update_desc()

            raw_handle = raw_output.hmonitor
            if isinstance(raw_handle, int) and not isinstance(raw_handle, bool):
                monitor_handle = raw_handle
            else:
                monitor_handle = _actual_int(raw_handle.value)
            if monitor_handle <= 0:
                raise ValueError("monitor handle must be positive")

            device_name = raw_output.devicename
            if not isinstance(device_name, str) or not device_name.strip():
                raise TypeError("device name is invalid")

            rectangle = raw_output.desc.DesktopCoordinates
            left = _actual_int(rectangle.left)
            top = _actual_int(rectangle.top)
            right = _actual_int(rectangle.right)
            bottom = _actual_int(rectangle.bottom)
            if right <= left or bottom <= top:
                raise ValueError("desktop rectangle is invalid")

            rotation_angle = _actual_int(raw_output.rotation_angle)
            if rotation_angle not in (0, 90, 180, 270):
                raise ValueError("rotation is invalid")

            attached = raw_output.attached_to_desktop
            if type(attached) is not bool:
                raise TypeError("attached_to_desktop is invalid")

            snapshot = DxcamOutput(
                monitor_handle=monitor_handle,
                device_name=device_name,
                device_index=device_index,
                output_index=output_index,
                desktop_geometry=ClientGeometry(
                    left,
                    top,
                    right - left,
                    bottom - top,
                ),
                rotation_angle=rotation_angle,
            )
        except _DxcamAdapterError:
            raise
        except Exception as exc:
            raise _DxcamAdapterError("DXcam output descriptor is invalid") from exc
        if not attached:
            return ()
        return (snapshot,)


def _stop_and_release_camera(camera: object) -> None:
    try:
        stop = getattr(camera, "stop", None)
        if callable(stop):
            stop()
    finally:
        release = getattr(camera, "release", None)
        if callable(release):
            release()


def load_dxcam_factory() -> DxcamFactory:
    try:
        version_text = metadata.version("dxcam")
        if not isinstance(version_text, str) or re.fullmatch(
            r"0\.3(?:\.\d+)*(?:(?:a|b|rc|post|dev)\d+)?"
            r"(?:\+[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*)?",
            version_text,
        ) is None:
            raise _DxcamAdapterError("DXcam 0.3 is required")
        module = importlib.import_module("dxcam")
    except _DxcamAdapterError:
        raise
    except Exception as exc:
        raise _DxcamAdapterError("DXcam could not be loaded") from exc
    return _Dxcam03FactoryAdapter(module)


@dataclass(frozen=True)
class _ActiveReadSnapshot:
    camera: object
    generation: int
    target: Sf6WindowTarget
    output: DxcamOutput
    factory: DxcamFactory


@dataclass(frozen=True)
class _ReadObservation:
    target: Sf6WindowTarget | None = None
    output: DxcamOutput | None = None
    window_error: Exception | None = None
    output_error: Exception | None = None


class _CaptureTransitionCancelled(RuntimeError):
    """A lifecycle token invalidated an uncommitted camera transition."""


class DxcamFrameSource:
    def __init__(
        self,
        *,
        window_finder: Callable[[], Sf6WindowTarget] = find_sf6_window_target,
        dxcam_factory_loader: Callable[[], DxcamFactory] = load_dxcam_factory,
        target_fps: int = 20,
    ) -> None:
        if not callable(window_finder):
            raise TypeError("window_finder must be callable")
        if not callable(dxcam_factory_loader):
            raise TypeError("dxcam_factory_loader must be callable")
        if type(target_fps) is not int:
            raise TypeError("target_fps must be an integer")
        if not 10 <= target_fps <= 30:
            raise ValueError("target_fps must be between 10 and 30")
        self._window_finder = window_finder
        self._dxcam_factory_loader = dxcam_factory_loader
        self._target_fps = target_fps
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_condition = threading.Condition(self._lifecycle_lock)
        self._state = "stopped"
        self._camera: object | None = None
        self._factory: DxcamFactory | None = None
        self._target: Sf6WindowTarget | None = None
        self._output: DxcamOutput | None = None
        self._camera_generation = 0
        self._reader_token: object | None = None
        self._next_frame_id = 0
        self._last_backend_timestamp: Real | None = None
        self._pending_geometry_notice = False
        self._cleanup_owner_token: object | None = None
        self._cleanup_owner_thread: threading.Thread | None = None
        self._lifecycle_token = 0
        self._transition_owner_token: object | None = None
        self._transition_owner_thread: threading.Thread | None = None

    def _begin_transition_locked(self, state: str) -> tuple[object, int]:
        if self._transition_owner_token is not None:
            raise RuntimeError("capture source lifecycle transition is already owned")
        self._lifecycle_token += 1
        owner = object()
        self._transition_owner_token = owner
        self._transition_owner_thread = threading.current_thread()
        self._state = state
        return owner, self._lifecycle_token

    def _transition_is_current_locked(self, owner: object, token: int) -> bool:
        return (
            self._transition_owner_token is owner
            and self._lifecycle_token == token
            and self._state in ("starting", "reconfiguring")
        )

    def _transition_is_cancelled_locked(self, owner: object, token: int) -> bool:
        return (
            self._transition_owner_token is owner
            and (
                self._lifecycle_token != token
                or self._state == "stopping"
            )
        )

    def _transition_cancelled(self, owner: object, token: int) -> bool:
        with self._lifecycle_lock:
            return not self._transition_is_current_locked(owner, token)

    def _require_transition_current(self, owner: object, token: int) -> None:
        with self._lifecycle_lock:
            if not self._transition_is_current_locked(owner, token):
                raise _CaptureTransitionCancelled("capture transition was cancelled")

    def _finish_transition_locked(self, owner: object, state: str) -> None:
        if self._transition_owner_token is not owner:
            return
        self._transition_owner_token = None
        self._transition_owner_thread = None
        self._state = state
        self._lifecycle_condition.notify_all()

    def _clear_session_locked(self) -> None:
        self._camera = None
        self._factory = None
        self._target = None
        self._output = None
        self._next_frame_id = 0
        self._last_backend_timestamp = None
        self._pending_geometry_notice = False

    def start(self) -> None:
        with self._lifecycle_condition:
            if self._state != "stopped":
                raise RuntimeError("capture source is already active")
            owner, lifecycle_token = self._begin_transition_locked("starting")

        camera: object | None = None
        factory: DxcamFactory | None = None
        try:
            try:
                target = self._window_finder()
                if not isinstance(target, Sf6WindowTarget):
                    raise GameWindowError("window finder returned an invalid target")
            except Exception as exc:
                raise _map_window_start_error(exc) from exc
            self._require_transition_current(owner, lifecycle_token)

            try:
                factory = self._dxcam_factory_loader()
                output = _resolve_dxcam_output(factory, target)
            except _UnsupportedDxcamOutput as exc:
                raise CaptureStartError(
                    CaptureStatus.LOST,
                    issue=CaptureIssue.UNSUPPORTED_OUTPUT,
                ) from exc
            except Exception as exc:
                raise CaptureStartError(
                    CaptureStatus.LOST,
                    issue=CaptureIssue.BACKEND_ERROR,
                ) from exc
            self._require_transition_current(owner, lifecycle_token)

            try:
                camera = _create_started_camera(
                    factory,
                    target,
                    output,
                    target_fps=self._target_fps,
                    cancelled=lambda: self._transition_cancelled(
                        owner,
                        lifecycle_token,
                    ),
                )
            except _CaptureTransitionCancelled:
                raise
            except Exception as exc:
                raise CaptureStartError(
                    CaptureStatus.LOST,
                    issue=CaptureIssue.BACKEND_ERROR,
                ) from exc

            with self._lifecycle_condition:
                if self._transition_is_current_locked(owner, lifecycle_token):
                    self._camera = camera
                    self._factory = factory
                    self._target = target
                    self._output = output
                    self._camera_generation += 1
                    self._next_frame_id = 0
                    self._last_backend_timestamp = None
                    self._pending_geometry_notice = False
                    self._finish_transition_locked(owner, "running")
                    return
            raise _CaptureTransitionCancelled("capture start was cancelled")
        except Exception as exc:
            with self._lifecycle_lock:
                cancelled = (
                    isinstance(exc, _CaptureTransitionCancelled)
                    or self._transition_is_cancelled_locked(owner, lifecycle_token)
                )
            if camera is not None and factory is not None:
                try:
                    _cleanup_factory_camera(factory, camera)
                except Exception:
                    pass
            with self._lifecycle_condition:
                if self._transition_owner_token is owner:
                    self._clear_session_locked()
                    self._finish_transition_locked(owner, "stopped")
            if cancelled:
                raise RuntimeError("capture source start was cancelled") from exc
            raise

    def read(self) -> CaptureSample:
        reader_token = object()
        with self._lifecycle_lock:
            if self._state in ("stopped", "stopping"):
                return CaptureSample(CaptureStatus.STOPPED)
            if self._state == "window_changed":
                return _lost_sample(CaptureIssue.WINDOW_CHANGED)
            if self._state not in ("running", "suspended"):
                raise RuntimeError("capture source is not readable")
            if self._reader_token is not None:
                raise RuntimeError("capture source already has an active reader")
            self._reader_token = reader_token
            suspended = self._state == "suspended"
            snapshot = None if suspended else self._active_snapshot_locked()

        try:
            if suspended:
                restored = self._restore_suspended()
                if restored is not None:
                    return restored
                with self._lifecycle_lock:
                    if self._state != "running":
                        return CaptureSample(CaptureStatus.STOPPED)
                    snapshot = self._active_snapshot_locked()
            if snapshot is None:
                return CaptureSample(CaptureStatus.STOPPED)
            pre = self._observe_read_boundary(snapshot.factory)
            if not self._snapshot_is_active(snapshot):
                return CaptureSample(CaptureStatus.STOPPED)
            routed = self._route_observation(snapshot, pre)
            if routed is not None:
                return routed

            backend_result: object = None
            backend_error: Exception | None = None
            try:
                backend_result = snapshot.camera.get_latest_frame(
                    copy=True,
                    with_timestamp=True,
                )
            except Exception as exc:
                backend_error = exc

            if not self._snapshot_is_active(snapshot):
                return CaptureSample(CaptureStatus.STOPPED)

            post = self._observe_read_boundary(snapshot.factory)
            if not self._snapshot_is_active(snapshot):
                return CaptureSample(CaptureStatus.STOPPED)
            routed = self._route_observation(snapshot, post)
            if routed is not None:
                return routed

            if (
                pre.target != snapshot.target
                or pre.output != snapshot.output
                or post.target != snapshot.target
                or post.output != snapshot.output
            ):
                raise RuntimeError("read boundary routing did not stabilize attachment")

            return self._interpret_backend_result(
                snapshot,
                backend_result,
                backend_error,
            )
        finally:
            with self._lifecycle_lock:
                if self._reader_token is reader_token:
                    self._reader_token = None

    def _active_snapshot_locked(self) -> _ActiveReadSnapshot:
        if (
            self._state != "running"
            or self._camera is None
            or self._factory is None
            or self._target is None
            or self._output is None
        ):
            raise RuntimeError("capture source active attachment is incomplete")
        return _ActiveReadSnapshot(
            self._camera,
            self._camera_generation,
            self._target,
            self._output,
            self._factory,
        )

    def _restore_suspended(self) -> CaptureSample | None:
        with self._lifecycle_lock:
            if self._state != "suspended":
                return CaptureSample(CaptureStatus.STOPPED)
            factory = self._factory
            baseline_target = self._target
            baseline_output = self._output
            generation = self._camera_generation
            if factory is None or baseline_target is None or baseline_output is None:
                return _lost_sample(CaptureIssue.BACKEND_ERROR)

        observation = self._observe_read_boundary(factory)
        identities = _observation_identities(observation)
        if any(identity != baseline_target.identity for identity in identities):
            return self._latch_suspended_window_changed(generation)
        error = observation.window_error
        if isinstance(
            error,
            (GameWindowNotFound, GameWindowAmbiguous, GameWindowIdentityChanged),
        ):
            return self._latch_suspended_window_changed(generation)
        if isinstance(error, GameWindowMinimized):
            return self._suspended_sample_if_current(
                generation,
                CaptureSample(CaptureStatus.MINIMIZED),
            )
        if isinstance(error, GameWindowOccluded):
            return self._suspended_sample_if_current(
                generation,
                CaptureSample(CaptureStatus.OCCLUDED),
            )
        if error is not None and not isinstance(error, GameWindowGeometryChanged):
            return self._suspended_sample_if_current(
                generation,
                _lost_sample(CaptureIssue.BACKEND_ERROR),
            )
        target = observation.target
        output = observation.output
        geometry_changed = (
            isinstance(error, GameWindowGeometryChanged)
            or (target is not None and target != baseline_target)
            or (output is not None and output != baseline_output)
        )
        if observation.output_error is not None:
            issue = (
                CaptureIssue.UNSUPPORTED_OUTPUT
                if isinstance(observation.output_error, _UnsupportedDxcamOutput)
                else CaptureIssue.BACKEND_ERROR
            )
            return self._suspended_sample_if_current(
                generation,
                _lost_sample(issue),
                pending_geometry_notice=geometry_changed,
            )
        if target is None or output is None:
            return self._suspended_sample_if_current(
                generation,
                _lost_sample(CaptureIssue.BACKEND_ERROR),
                pending_geometry_notice=geometry_changed,
            )

        with self._lifecycle_condition:
            if self._state != "suspended" or self._camera_generation != generation:
                return CaptureSample(CaptureStatus.STOPPED)
            if geometry_changed:
                self._pending_geometry_notice = True
            self._camera_generation += 1
            owner, lifecycle_token = self._begin_transition_locked("reconfiguring")
        try:
            camera = _create_started_camera(
                factory,
                target,
                output,
                target_fps=self._target_fps,
                cancelled=lambda: self._transition_cancelled(
                    owner,
                    lifecycle_token,
                ),
            )
        except Exception as exc:
            with self._lifecycle_condition:
                cancelled = (
                    isinstance(exc, _CaptureTransitionCancelled)
                    or self._transition_is_cancelled_locked(owner, lifecycle_token)
                )
                if self._transition_owner_token is owner:
                    if cancelled:
                        self._clear_session_locked()
                        self._finish_transition_locked(owner, "stopped")
                    else:
                        self._finish_transition_locked(owner, "suspended")
            if cancelled:
                return CaptureSample(CaptureStatus.STOPPED)
            return _lost_sample(CaptureIssue.BACKEND_ERROR)

        with self._lifecycle_condition:
            if self._transition_is_current_locked(owner, lifecycle_token):
                self._camera = camera
                self._factory = factory
                self._target = target
                self._output = output
                self._camera_generation += 1
                pending_notice = self._pending_geometry_notice or geometry_changed
                self._pending_geometry_notice = False
                self._finish_transition_locked(owner, "running")
                cleanup = False
            else:
                cleanup = True
                pending_notice = False
        if cleanup:
            try:
                _cleanup_factory_camera(factory, camera)
            except Exception:
                pass
            with self._lifecycle_condition:
                if self._transition_owner_token is owner:
                    self._clear_session_locked()
                    self._finish_transition_locked(owner, "stopped")
            return CaptureSample(CaptureStatus.STOPPED)
        if pending_notice:
            return _lost_sample(CaptureIssue.GEOMETRY_CHANGED)
        return None

    def _suspended_sample_if_current(
        self,
        generation: int,
        sample: CaptureSample,
        *,
        pending_geometry_notice: bool = False,
    ) -> CaptureSample:
        with self._lifecycle_lock:
            if self._state != "suspended" or self._camera_generation != generation:
                return CaptureSample(CaptureStatus.STOPPED)
            if pending_geometry_notice:
                self._pending_geometry_notice = True
            return sample

    def _latch_suspended_window_changed(self, generation: int) -> CaptureSample:
        with self._lifecycle_lock:
            if self._state != "suspended" or self._camera_generation != generation:
                return CaptureSample(CaptureStatus.STOPPED)
            self._factory = None
            self._target = None
            self._output = None
            self._camera_generation += 1
            self._state = "window_changed"
        return _lost_sample(CaptureIssue.WINDOW_CHANGED)

    def _observe_read_boundary(
        self,
        factory: DxcamFactory,
    ) -> _ReadObservation:
        target: Sf6WindowTarget | None = None
        window_error: Exception | None = None
        try:
            target = self._window_finder()
            if not isinstance(target, Sf6WindowTarget):
                raise GameWindowError("window finder returned an invalid target")
        except GameWindowGeometryChanged as exc:
            target = exc.after
            window_error = exc
        except Exception as exc:
            return _ReadObservation(window_error=exc)

        try:
            output = _resolve_dxcam_output(factory, target)
        except Exception as exc:
            return _ReadObservation(
                target=target,
                window_error=window_error,
                output_error=exc,
            )
        return _ReadObservation(
            target=target,
            output=output,
            window_error=window_error,
        )

    def _snapshot_is_active(self, snapshot: _ActiveReadSnapshot) -> bool:
        with self._lifecycle_lock:
            return self._snapshot_is_active_locked(snapshot)

    def _snapshot_is_active_locked(self, snapshot: _ActiveReadSnapshot) -> bool:
        return (
            self._state == "running"
            and self._camera is snapshot.camera
            and self._camera_generation == snapshot.generation
        )

    def _route_observation(
        self,
        snapshot: _ActiveReadSnapshot,
        observation: _ReadObservation,
    ) -> CaptureSample | None:
        error = observation.window_error
        attached_identity = snapshot.target.identity
        identities = _observation_identities(observation)
        if any(identity != attached_identity for identity in identities):
            return self._latch_window_changed(snapshot)

        if isinstance(
            error,
            (
                GameWindowNotFound,
                GameWindowAmbiguous,
                GameWindowIdentityChanged,
            ),
        ):
            return self._latch_window_changed(snapshot)
        if isinstance(error, GameWindowMinimized):
            return self._suspend_attachment(
                snapshot,
                CaptureSample(CaptureStatus.MINIMIZED),
            )
        if isinstance(error, GameWindowOccluded):
            return self._suspend_attachment(
                snapshot,
                CaptureSample(CaptureStatus.OCCLUDED),
            )
        if error is not None and not isinstance(error, GameWindowGeometryChanged):
            return self._finish_rejected_sample(
                snapshot,
                _lost_sample(CaptureIssue.BACKEND_ERROR),
            )

        geometry_changed = (
            isinstance(error, GameWindowGeometryChanged)
            or (
                observation.target is not None
                and observation.target != snapshot.target
            )
        )
        if observation.output_error is not None:
            issue = (
                CaptureIssue.UNSUPPORTED_OUTPUT
                if isinstance(observation.output_error, _UnsupportedDxcamOutput)
                else CaptureIssue.BACKEND_ERROR
            )
            if geometry_changed:
                return self._suspend_attachment(
                    snapshot,
                    _lost_sample(issue),
                    pending_geometry_notice=True,
                )
            if isinstance(observation.output_error, _UnsupportedDxcamOutput):
                return self._suspend_attachment(
                    snapshot,
                    _lost_sample(CaptureIssue.UNSUPPORTED_OUTPUT),
                )
            return self._finish_rejected_sample(
                snapshot,
                _lost_sample(CaptureIssue.BACKEND_ERROR),
            )

        target = observation.target
        output = observation.output
        if target is None or output is None:
            return self._finish_rejected_sample(
                snapshot,
                _lost_sample(CaptureIssue.BACKEND_ERROR),
            )
        if geometry_changed or output != snapshot.output:
            return self._rebuild_attachment(snapshot, target, output)
        return None

    def _finish_rejected_sample(
        self,
        snapshot: _ActiveReadSnapshot,
        sample: CaptureSample,
    ) -> CaptureSample:
        with self._lifecycle_lock:
            if not self._snapshot_is_active_locked(snapshot):
                return CaptureSample(CaptureStatus.STOPPED)
            return sample

    def _latch_window_changed(
        self,
        snapshot: _ActiveReadSnapshot,
    ) -> CaptureSample:
        return self._detach_for_status(
            snapshot,
            state="window_changed",
            sample=_lost_sample(CaptureIssue.WINDOW_CHANGED),
            retain_baseline=False,
        )

    def _suspend_attachment(
        self,
        snapshot: _ActiveReadSnapshot,
        sample: CaptureSample,
        *,
        pending_geometry_notice: bool = False,
    ) -> CaptureSample:
        return self._detach_for_status(
            snapshot,
            state="suspended",
            sample=sample,
            retain_baseline=True,
            pending_geometry_notice=pending_geometry_notice,
        )

    def _detach_for_status(
        self,
        snapshot: _ActiveReadSnapshot,
        *,
        state: str,
        sample: CaptureSample,
        retain_baseline: bool,
        pending_geometry_notice: bool = False,
    ) -> CaptureSample:
        cleanup_token: object | None = None
        with self._lifecycle_condition:
            if not self._snapshot_is_active_locked(snapshot):
                return CaptureSample(CaptureStatus.STOPPED)
            camera = self._camera
            factory = self._factory
            self._camera = None
            if not retain_baseline:
                self._factory = None
                self._target = None
                self._output = None
            if pending_geometry_notice:
                self._pending_geometry_notice = True
            self._camera_generation += 1
            self._state = state
            if camera is not None and factory is not None:
                cleanup_token = object()
                self._cleanup_owner_token = cleanup_token
                self._cleanup_owner_thread = threading.current_thread()
        if camera is not None and factory is not None:
            try:
                _cleanup_factory_camera(factory, camera)
            except Exception:
                pass
        with self._lifecycle_condition:
            if (
                cleanup_token is not None
                and self._cleanup_owner_token is cleanup_token
            ):
                self._cleanup_owner_token = None
                self._cleanup_owner_thread = None
                if self._state == "stopping":
                    self._state = "stopped"
                self._lifecycle_condition.notify_all()
            if self._state == "stopped":
                return CaptureSample(CaptureStatus.STOPPED)
            return sample

    def _rebuild_attachment(
        self,
        snapshot: _ActiveReadSnapshot,
        target: Sf6WindowTarget,
        output: DxcamOutput,
    ) -> CaptureSample:
        with self._lifecycle_condition:
            if not self._snapshot_is_active_locked(snapshot):
                return CaptureSample(CaptureStatus.STOPPED)
            old_camera = self._camera
            factory = self._factory
            self._camera = None
            self._camera_generation += 1
            self._pending_geometry_notice = True
            owner, lifecycle_token = self._begin_transition_locked("reconfiguring")

        if old_camera is None or factory is None:
            with self._lifecycle_condition:
                if self._transition_owner_token is owner:
                    self._finish_transition_locked(owner, "suspended")
            return _lost_sample(CaptureIssue.BACKEND_ERROR)
        try:
            _cleanup_factory_camera(factory, old_camera)
        except Exception:
            with self._lifecycle_condition:
                cancelled = self._transition_is_cancelled_locked(
                    owner,
                    lifecycle_token,
                )
                if self._transition_owner_token is owner:
                    if cancelled:
                        self._clear_session_locked()
                        self._finish_transition_locked(owner, "stopped")
                    else:
                        self._finish_transition_locked(owner, "suspended")
            return (
                CaptureSample(CaptureStatus.STOPPED)
                if cancelled
                else _lost_sample(CaptureIssue.BACKEND_ERROR)
            )

        with self._lifecycle_condition:
            if not self._transition_is_current_locked(owner, lifecycle_token):
                if self._transition_owner_token is owner:
                    self._clear_session_locked()
                    self._finish_transition_locked(owner, "stopped")
                return CaptureSample(CaptureStatus.STOPPED)

        try:
            new_camera = _create_started_camera(
                factory,
                target,
                output,
                target_fps=self._target_fps,
                cancelled=lambda: self._transition_cancelled(
                    owner,
                    lifecycle_token,
                ),
            )
        except Exception as exc:
            with self._lifecycle_condition:
                cancelled = (
                    isinstance(exc, _CaptureTransitionCancelled)
                    or self._transition_is_cancelled_locked(owner, lifecycle_token)
                )
                if self._transition_owner_token is owner:
                    if cancelled:
                        self._clear_session_locked()
                        self._finish_transition_locked(owner, "stopped")
                    else:
                        self._finish_transition_locked(owner, "suspended")
            return (
                CaptureSample(CaptureStatus.STOPPED)
                if cancelled
                else _lost_sample(CaptureIssue.BACKEND_ERROR)
            )

        with self._lifecycle_condition:
            if self._transition_is_current_locked(owner, lifecycle_token):
                self._camera = new_camera
                self._factory = factory
                self._target = target
                self._output = output
                self._camera_generation += 1
                self._pending_geometry_notice = False
                self._finish_transition_locked(owner, "running")
                cleanup_new = False
            else:
                cleanup_new = True
        if cleanup_new:
            try:
                _cleanup_factory_camera(factory, new_camera)
            except Exception:
                pass
            with self._lifecycle_condition:
                if self._transition_owner_token is owner:
                    self._clear_session_locked()
                    self._finish_transition_locked(owner, "stopped")
            return CaptureSample(CaptureStatus.STOPPED)
        return _lost_sample(CaptureIssue.GEOMETRY_CHANGED)

    def _interpret_backend_result(
        self,
        snapshot: _ActiveReadSnapshot,
        result: object,
        backend_error: Exception | None,
    ) -> CaptureSample:
        if backend_error is not None:
            return self._finish_rejected_sample(
                snapshot,
                _lost_sample(CaptureIssue.BACKEND_ERROR),
            )
        if result is None:
            try:
                capturing = _camera_boolean(snapshot.camera, "is_capturing")
            except Exception:
                return self._finish_rejected_sample(
                    snapshot,
                    _lost_sample(CaptureIssue.BACKEND_ERROR),
                )
            issue = (
                CaptureIssue.NO_NEW_FRAME
                if capturing is True
                else CaptureIssue.BACKEND_ERROR
            )
            return self._finish_rejected_sample(snapshot, _lost_sample(issue))

        try:
            if type(result) is not tuple or len(result) != 2:
                raise TypeError("DXcam result must be an image/timestamp tuple")
            image, timestamp = result
            if not isinstance(image, np.ndarray):
                raise TypeError("DXcam image must be a NumPy array")
            _require_nonnegative_finite_real("backend timestamp", timestamp)
            if image.dtype != np.dtype(np.uint8):
                raise TypeError("DXcam image must have uint8 dtype")
            if image.shape != (
                snapshot.target.geometry.height,
                snapshot.target.geometry.width,
                3,
            ):
                raise ValueError("DXcam image shape does not match client geometry")
            if not image.flags.c_contiguous:
                raise ValueError("DXcam image must be C-contiguous")
            if not image.flags.owndata:
                raise ValueError("DXcam image must own its storage")
            if not image.flags.writeable:
                raise ValueError("DXcam image must be writable")
        except Exception:
            return self._finish_rejected_sample(
                snapshot,
                _lost_sample(CaptureIssue.BACKEND_ERROR),
            )

        try:
            is_black = not np.any(image)
        except Exception:
            return self._finish_rejected_sample(
                snapshot,
                _lost_sample(CaptureIssue.BACKEND_ERROR),
            )
        if is_black:
            return self._finish_rejected_sample(
                snapshot,
                _lost_sample(CaptureIssue.BLACK_FRAME),
            )

        with self._lifecycle_lock:
            if not self._snapshot_is_active_locked(snapshot):
                return CaptureSample(CaptureStatus.STOPPED)
            try:
                if (
                    self._last_backend_timestamp is not None
                    and timestamp <= self._last_backend_timestamp
                ):
                    return _lost_sample(CaptureIssue.NO_NEW_FRAME)
                frame = CapturedFrame(
                    self._next_frame_id,
                    image,
                    timestamp,
                    snapshot.target.geometry,
                )
                sample = CaptureSample(CaptureStatus.RUNNING, frame=frame)
            except Exception:
                return _lost_sample(CaptureIssue.BACKEND_ERROR)
            self._next_frame_id += 1
            self._last_backend_timestamp = timestamp
            return sample

    def stop(self) -> None:
        cleanup_token: object | None = None
        cleanup_error: Exception | None = None
        current_thread = threading.current_thread()
        with self._lifecycle_condition:
            if self._state == "stopped":
                return
            if self._state == "stopping":
                observed_transition = self._transition_owner_token
                observed_cleanup = self._cleanup_owner_token
                if (
                    self._transition_owner_thread is current_thread
                    or self._cleanup_owner_thread is current_thread
                ):
                    return
                if observed_transition is not None:
                    self._lifecycle_condition.wait_for(
                        lambda: (
                            self._transition_owner_token is not observed_transition
                        )
                    )
                elif observed_cleanup is not None:
                    self._lifecycle_condition.wait_for(
                        lambda: self._cleanup_owner_token is not observed_cleanup
                    )
                return
            if self._state in ("starting", "reconfiguring"):
                observed_transition = self._transition_owner_token
                if observed_transition is None:
                    raise RuntimeError("capture transition has no owner")
                self._lifecycle_token += 1
                self._camera_generation += 1
                self._clear_session_locked()
                self._state = "stopping"
                if self._transition_owner_thread is current_thread:
                    return
                self._lifecycle_condition.wait_for(
                    lambda: self._transition_owner_token is not observed_transition
                )
                return
            if self._cleanup_owner_token is not None:
                observed_cleanup = self._cleanup_owner_token
                self._lifecycle_token += 1
                self._camera_generation += 1
                self._clear_session_locked()
                self._state = "stopping"
                if self._cleanup_owner_thread is current_thread:
                    return
                self._lifecycle_condition.wait_for(
                    lambda: self._cleanup_owner_token is not observed_cleanup
                )
                return
            camera = self._camera
            factory = self._factory
            self._lifecycle_token += 1
            self._camera_generation += 1
            self._clear_session_locked()
            if camera is None or factory is None:
                self._state = "stopped"
                self._lifecycle_condition.notify_all()
                return
            cleanup_token = object()
            self._cleanup_owner_token = cleanup_token
            self._cleanup_owner_thread = current_thread
            self._state = "stopping"
        try:
            _cleanup_factory_camera(factory, camera)
        except Exception as exc:
            cleanup_error = exc
        finally:
            with self._lifecycle_condition:
                if self._cleanup_owner_token is cleanup_token:
                    self._cleanup_owner_token = None
                    self._cleanup_owner_thread = None
                    self._state = "stopped"
                    self._lifecycle_condition.notify_all()
        if cleanup_error is not None:
            raise cleanup_error


def _lost_sample(issue: CaptureIssue) -> CaptureSample:
    return CaptureSample(CaptureStatus.LOST, issue=issue)


def _observation_identities(
    observation: _ReadObservation,
) -> tuple[object, ...]:
    identities: list[object] = []
    if observation.target is not None:
        identities.append(observation.target.identity)
    error = observation.window_error
    if isinstance(error, (GameWindowIdentityChanged, GameWindowGeometryChanged)):
        identities.append(error.before.identity)
        if error.after is not None:
            identities.append(error.after.identity)
    return tuple(identities)


def _create_started_camera(
    factory: DxcamFactory,
    target: Sf6WindowTarget,
    output: DxcamOutput,
    *,
    target_fps: int,
    cancelled: Callable[[], bool] | None = None,
) -> object:
    region = _relative_region(target.geometry, output.desktop_geometry)
    camera: object | None = None
    for backend in ("dxgi", "winrt"):
        if cancelled is not None and cancelled():
            raise _CaptureTransitionCancelled("capture transition was cancelled")
        candidate: object | None = None
        cleanup_failed = False
        try:
            candidate = factory.create(
                device_idx=output.device_index,
                output_idx=output.output_index,
                region=region,
                output_color="BGR",
                max_buffer_len=2,
                backend=backend,
                processor_backend="cv2",
            )
            if cancelled is not None and cancelled():
                raise _CaptureTransitionCancelled("capture transition was cancelled")
            _validate_source_camera(candidate)
            candidate.start(
                region=region,
                target_fps=target_fps,
                video_mode=False,
            )
            if cancelled is not None and cancelled():
                raise _CaptureTransitionCancelled("capture transition was cancelled")
            if _camera_boolean(candidate, "is_capturing") is not True:
                raise _DxcamAdapterError("DXcam camera did not start")
        except Exception as exc:
            if candidate is not None:
                try:
                    _cleanup_factory_camera(factory, candidate)
                except Exception:
                    cleanup_failed = True
            if isinstance(exc, _CaptureTransitionCancelled):
                raise
            if cancelled is not None and cancelled():
                raise _CaptureTransitionCancelled(
                    "capture transition was cancelled"
                ) from exc
            if cleanup_failed:
                break
            continue
        camera = candidate
        break

    if camera is None:
        raise _DxcamAdapterError("DXcam camera could not be started")
    return camera


def _map_window_start_error(exc: Exception) -> CaptureStartError:
    if isinstance(exc, GameWindowMinimized):
        return CaptureStartError(CaptureStatus.MINIMIZED, issue=None)
    if isinstance(exc, GameWindowOccluded):
        return CaptureStartError(CaptureStatus.OCCLUDED, issue=None)
    if isinstance(
        exc,
        (
            GameWindowNotFound,
            GameWindowAmbiguous,
            GameWindowIdentityChanged,
        ),
    ):
        return CaptureStartError(
            CaptureStatus.LOST,
            issue=CaptureIssue.WINDOW_CHANGED,
        )
    if isinstance(exc, GameWindowGeometryChanged):
        return CaptureStartError(
            CaptureStatus.LOST,
            issue=CaptureIssue.GEOMETRY_CHANGED,
        )
    return CaptureStartError(
        CaptureStatus.LOST,
        issue=CaptureIssue.BACKEND_ERROR,
    )


def _contains_geometry(outer: ClientGeometry, inner: ClientGeometry) -> bool:
    return (
        outer.left <= inner.left
        and outer.top <= inner.top
        and inner.left + inner.width <= outer.left + outer.width
        and inner.top + inner.height <= outer.top + outer.height
    )


def _resolve_dxcam_output(
    factory: DxcamFactory,
    target: Sf6WindowTarget,
) -> DxcamOutput:
    try:
        outputs_method = factory.outputs
        if not callable(outputs_method):
            raise TypeError("outputs is not callable")
        outputs = outputs_method()
    except Exception as exc:
        raise _DxcamAdapterError("DXcam output enumeration failed") from exc
    if type(outputs) is not tuple or any(type(item) is not DxcamOutput for item in outputs):
        raise _DxcamAdapterError("DXcam output snapshot is invalid")

    monitor = target.monitor
    matches = tuple(
        output
        for output in outputs
        if output.monitor_handle == monitor.handle
        and output.device_name.casefold() == monitor.device_name.casefold()
        and output.desktop_geometry == monitor.desktop_geometry
        and _contains_geometry(output.desktop_geometry, target.geometry)
    )
    if len(matches) != 1:
        raise _UnsupportedDxcamOutput("DXcam output is absent or ambiguous")
    return matches[0]


def _relative_region(
    client: ClientGeometry,
    desktop: ClientGeometry,
) -> tuple[int, int, int, int]:
    return (
        client.left - desktop.left,
        client.top - desktop.top,
        client.left + client.width - desktop.left,
        client.top + client.height - desktop.top,
    )


def _camera_boolean(camera: object, name: str) -> bool:
    try:
        value = getattr(camera, name)
    except Exception as exc:
        raise _DxcamAdapterError(f"DXcam camera {name} is unavailable") from exc
    if type(value) is not bool:
        raise _DxcamAdapterError(f"DXcam camera {name} is invalid")
    return value


def _validate_source_camera(camera: object) -> None:
    if _camera_boolean(camera, "is_released"):
        raise _DxcamAdapterError("DXcam returned a released camera")
    _camera_boolean(camera, "is_capturing")
    for method_name in ("start", "get_latest_frame", "stop", "release"):
        if not callable(getattr(camera, method_name, None)):
            raise _DxcamAdapterError(f"DXcam camera {method_name} is invalid")


def _cleanup_factory_camera(factory: DxcamFactory, camera: object) -> None:
    claim = getattr(factory, "_cleanup_owned_camera", None)
    if callable(claim):
        claimed = claim(camera)
        if claimed is not True:
            raise _DxcamAdapterError("DXcam camera ownership could not be claimed")
        return
    _stop_and_release_camera(camera)


def _require_nonnegative_finite_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, Rational):
        if value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _exact_rational(value: Real) -> Rational:
    if isinstance(value, Rational):
        return value
    ratio_method = getattr(value, "as_integer_ratio", None)
    if ratio_method is None:
        numerator, denominator = float(value).as_integer_ratio()
    else:
        numerator, denominator = ratio_method()
    return Fraction(numerator, denominator)


class FreshFrameGate:
    """Accept only timely frames that advance both source watermarks."""

    def __init__(
        self,
        max_age_seconds: Real = 0.25,
        max_future_seconds: Real = 0.05,
    ) -> None:
        _require_nonnegative_finite_real("max_age_seconds", max_age_seconds)
        _require_nonnegative_finite_real("max_future_seconds", max_future_seconds)
        self.max_age_seconds = max_age_seconds
        self.max_future_seconds = max_future_seconds
        self._last_frame_id: int | None = None
        self._last_timestamp: Real | None = None

    def accept(self, frame: CapturedFrame, *, now: Real) -> bool:
        if not isinstance(frame, CapturedFrame):
            raise TypeError("frame must be CapturedFrame")
        _require_nonnegative_finite_real("now", now)

        frame_id = frame.frame_id
        timestamp = frame.monotonic_time
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            return False
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            return False

        exact_now = _exact_rational(now)
        exact_timestamp = _exact_rational(timestamp)
        age = exact_now - exact_timestamp
        future = exact_timestamp - exact_now
        if (
            age > _exact_rational(self.max_age_seconds)
            or future > _exact_rational(self.max_future_seconds)
        ):
            return False

        self._last_frame_id = frame_id
        self._last_timestamp = timestamp
        return True

    def reset(self) -> None:
        self._last_frame_id = None
        self._last_timestamp = None
