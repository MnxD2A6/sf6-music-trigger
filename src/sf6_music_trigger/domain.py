"""Immutable domain values shared by capture, tracking, and playback."""

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Rational, Real

import numpy as np


class Side(str, Enum):
    P1 = "P1"
    P2 = "P2"


class CaptureStatus(str, Enum):
    RUNNING = "RUNNING"
    OCCLUDED = "OCCLUDED"
    MINIMIZED = "MINIMIZED"
    LOST = "LOST"
    STOPPED = "STOPPED"


class CaptureIssue(str, Enum):
    BLACK_FRAME = "BLACK_FRAME"
    NO_NEW_FRAME = "NO_NEW_FRAME"
    WINDOW_CHANGED = "WINDOW_CHANGED"
    GEOMETRY_CHANGED = "GEOMETRY_CHANGED"
    BACKEND_ERROR = "BACKEND_ERROR"
    UNSUPPORTED_OUTPUT = "UNSUPPORTED_OUTPUT"


class TrackerState(str, Enum):
    DISARMED = "DISARMED"
    ARMED = "ARMED"
    SPEND_CANDIDATE = "SPEND_CANDIDATE"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


def _require_int(name: str, value: object, *, minimum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_nonnegative_finite_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, Rational):
        if value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _require_nonblank_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


@dataclass(frozen=True)
class ClientGeometry:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        _require_int("left", self.left)
        _require_int("top", self.top)
        _require_int("width", self.width, minimum=1)
        _require_int("height", self.height, minimum=1)


@dataclass(frozen=True, eq=False)
class CapturedFrame:
    frame_id: int
    image: np.ndarray
    monotonic_time: Real
    geometry: ClientGeometry

    def __post_init__(self) -> None:
        _require_int("frame_id", self.frame_id, minimum=0)
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        _require_nonnegative_finite_real("monotonic_time", self.monotonic_time)
        if not isinstance(self.geometry, ClientGeometry):
            raise TypeError("geometry must be ClientGeometry")


@dataclass(frozen=True)
class CaptureSample:
    status: CaptureStatus
    frame: CapturedFrame | None = None
    issue: CaptureIssue | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CaptureStatus):
            raise TypeError("status must be CaptureStatus")

        if self.status is CaptureStatus.RUNNING:
            if not isinstance(self.frame, CapturedFrame):
                raise TypeError("RUNNING sample must carry CapturedFrame")
            if self.issue is not None:
                raise ValueError("RUNNING sample must not carry an issue")

            image = self.frame.image
            expected_shape = (
                self.frame.geometry.height,
                self.frame.geometry.width,
                3,
            )
            if image.dtype != np.dtype(np.uint8):
                raise TypeError("RUNNING frame image must have uint8 dtype")
            if image.shape != expected_shape:
                raise ValueError("RUNNING frame image must match its geometry")
            if not image.flags.c_contiguous:
                raise ValueError("RUNNING frame image must be C-contiguous")
            if not image.flags.owndata:
                raise ValueError("RUNNING frame image must own its storage")
            if not image.flags.writeable:
                raise ValueError("RUNNING frame image must be writable")
            return

        if self.frame is not None:
            raise ValueError("non-running sample must not carry a frame")

        if self.status is CaptureStatus.LOST:
            if not isinstance(self.issue, CaptureIssue):
                raise TypeError("LOST sample must carry CaptureIssue")
            return

        if self.issue is not None:
            raise ValueError("only LOST sample may carry an issue")


class CandidateCancelReason(str, Enum):
    LEVEL_RECOVERED = "LEVEL_RECOVERED"
    LEVEL_CONFLICT = "LEVEL_CONFLICT"
    CONFIRMATION_TIMEOUT = "CONFIRMATION_TIMEOUT"
    CAPTURE_RESET = "CAPTURE_RESET"
    TRACKER_RESET = "TRACKER_RESET"


@dataclass(frozen=True)
class ConfirmedSuperEvent:
    event_id: str
    side: Side
    confirmed_at: float
    pre_spend_frame_id: int
    confirmed_frame_id: int
    profile_id: str
    meter_before: int | None = None
    meter_after: int | None = None
    levels_spent: int | None = None
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank_string("event_id", self.event_id)
        if not isinstance(self.side, Side):
            raise TypeError("side must be Side")
        _require_nonnegative_finite_real("confirmed_at", self.confirmed_at)
        _require_int("pre_spend_frame_id", self.pre_spend_frame_id, minimum=0)
        _require_int("confirmed_frame_id", self.confirmed_frame_id, minimum=0)
        if self.pre_spend_frame_id > self.confirmed_frame_id:
            raise ValueError("pre_spend_frame_id must not exceed confirmed_frame_id")
        _require_nonblank_string("profile_id", self.profile_id)
        if self.candidate_id is not None:
            _require_nonblank_string("candidate_id", self.candidate_id)
        metadata = (self.meter_before, self.meter_after, self.levels_spent)
        if metadata == (None, None, None):
            return
        if any(value is None for value in metadata):
            raise ValueError("meter metadata must be entirely present or absent")
        meter_before = self.meter_before
        meter_after = self.meter_after
        levels_spent = self.levels_spent
        _require_int("meter_before", meter_before, minimum=0)
        _require_int("meter_after", meter_after, minimum=0)
        _require_int("levels_spent", levels_spent, minimum=1)
        if meter_before > 3 or meter_after > 3 or levels_spent > 3:
            raise ValueError("meter metadata must stay within levels zero through three")
        if meter_before <= meter_after:
            raise ValueError("meter_before must be greater than meter_after")
        if levels_spent != meter_before - meter_after:
            raise ValueError("levels_spent must equal meter_before minus meter_after")


@dataclass(frozen=True)
class SuperCandidateEvent:
    candidate_id: str
    side: Side
    suspected_at: Real
    pre_spend_frame_id: int
    candidate_frame_id: int
    profile_id: str
    meter_before: int
    meter_after: int
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonblank_string("candidate_id", self.candidate_id)
        if not isinstance(self.side, Side):
            raise TypeError("side must be Side")
        _require_nonnegative_finite_real("suspected_at", self.suspected_at)
        _require_int("pre_spend_frame_id", self.pre_spend_frame_id, minimum=0)
        _require_int("candidate_frame_id", self.candidate_frame_id, minimum=0)
        if self.pre_spend_frame_id > self.candidate_frame_id:
            raise ValueError("pre_spend_frame_id must not exceed candidate_frame_id")
        _require_nonblank_string("profile_id", self.profile_id)
        _require_int("meter_before", self.meter_before, minimum=0)
        _require_int("meter_after", self.meter_after, minimum=0)
        if self.meter_before > 3 or self.meter_after > 3:
            raise ValueError("meter levels must stay within zero through three")
        if self.meter_before <= self.meter_after:
            raise ValueError("meter_before must be greater than meter_after")
        if type(self.evidence) is not tuple:
            raise TypeError("evidence must be a tuple")
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        for evidence_code in self.evidence:
            _require_nonblank_string("evidence", evidence_code)

    @property
    def levels_suspected(self) -> int:
        return self.meter_before - self.meter_after


@dataclass(frozen=True)
class CancelledSuperCandidate:
    candidate_id: str
    side: Side
    decided_at: Real
    reason: CandidateCancelReason

    def __post_init__(self) -> None:
        _require_nonblank_string("candidate_id", self.candidate_id)
        if not isinstance(self.side, Side):
            raise TypeError("side must be Side")
        _require_nonnegative_finite_real("decided_at", self.decided_at)
        if not isinstance(self.reason, CandidateCancelReason):
            raise TypeError("reason must be CandidateCancelReason")


TrackerTransition = SuperCandidateEvent | ConfirmedSuperEvent | CancelledSuperCandidate


@dataclass(frozen=True)
class GaugeTrackerUpdate:
    transition: TrackerTransition | None = None

    def __post_init__(self) -> None:
        if self.transition is not None and not isinstance(
            self.transition,
            (SuperCandidateEvent, ConfirmedSuperEvent, CancelledSuperCandidate),
        ):
            raise TypeError("transition must be a tracker lifecycle event")
