"""Stable direction tracking for fixed SF6 super-meter level readings."""

from dataclasses import dataclass

from .domain import Side
from .meter_level_features import MeterLevelReading


def _require_side(value: object) -> Side:
    if not isinstance(value, Side):
        raise TypeError("side must be Side")
    return value


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return value


@dataclass(frozen=True)
class MeterLevelTransition:
    """One confirmed stable downward meter-level transition."""

    side: Side
    before: int
    after: int
    pre_frame_id: int
    confirmed_frame_id: int

    def __post_init__(self) -> None:
        _require_side(self.side)
        before = _require_int("before", self.before, minimum=0, maximum=3)
        after = _require_int("after", self.after, minimum=0, maximum=3)
        pre_frame_id = _require_int("pre_frame_id", self.pre_frame_id, minimum=0)
        confirmed_frame_id = _require_int(
            "confirmed_frame_id", self.confirmed_frame_id, minimum=0
        )
        if before <= after:
            raise ValueError("before must be greater than after")
        if confirmed_frame_id <= pre_frame_id:
            raise ValueError("confirmed_frame_id must be after pre_frame_id")

    @property
    def levels_spent(self) -> int:
        return self.before - self.after


class StableMeterLevelTracker:
    """Debounce meter glyphs and emit only stable downward transitions."""

    def __init__(
        self,
        side: Side,
        *,
        stable_samples: int,
        baseline_samples: int | None = None,
        unknown_reset_samples: int,
    ) -> None:
        self._side = _require_side(side)
        self._stable_samples = _require_int(
            "stable_samples", stable_samples, minimum=1
        )
        self._baseline_samples = (
            self._stable_samples
            if baseline_samples is None
            else _require_int(
                "baseline_samples",
                baseline_samples,
                minimum=self._stable_samples,
            )
        )
        self._unknown_reset_samples = _require_int(
            "unknown_reset_samples", unknown_reset_samples, minimum=1
        )
        self.reset()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StableMeterLevelTracker):
            return NotImplemented
        return vars(self) == vars(other)

    @property
    def stable_level(self) -> int | None:
        return self._stable_level

    @property
    def stable_frame_id(self) -> int | None:
        return self._stable_frame_id

    @property
    def baseline_ready(self) -> bool:
        return (
            self._stable_level is not None
            and self._stable_count >= self._baseline_samples
        )

    def update(
        self,
        reading: MeterLevelReading,
        frame_id: int,
        *,
        baseline_trusted: bool = False,
        baseline_observed: bool = True,
    ) -> MeterLevelTransition | None:
        if not isinstance(reading, MeterLevelReading):
            raise TypeError("reading must be MeterLevelReading")
        if reading.side is not self._side:
            raise ValueError("reading side does not match tracker side")
        if type(baseline_trusted) is not bool:
            raise TypeError("baseline_trusted must be a boolean")
        if type(baseline_observed) is not bool:
            raise TypeError("baseline_observed must be a boolean")
        checked_frame_id = _require_int("frame_id", frame_id, minimum=0)
        if self._last_frame_id is not None and checked_frame_id <= self._last_frame_id:
            raise ValueError("frame_id must increase strictly")
        self._last_frame_id = checked_frame_id

        level = reading.level
        if level is None:
            self._candidate = None
            self._candidate_count = 0
            self._unknown_count += 1
            if self._unknown_count >= self._unknown_reset_samples:
                self._stable_level = None
                self._stable_frame_id = None
                self._stable_count = 0
            return None

        self._unknown_count = 0
        if level == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = level
            self._candidate_count = 1
        # Once established, every genuinely observed matching sample supports
        # the same baseline. Callers may preserve an old level while rejecting
        # a conflicting glyph; that substitute is not fresh baseline evidence.
        if level == self._stable_level:
            if baseline_observed:
                self._stable_frame_id = checked_frame_id
                self._stable_count += 1
                if baseline_trusted:
                    self._stable_count = max(
                        self._stable_count,
                        self._baseline_samples,
                    )
            return None
        if self._candidate_count < self._stable_samples:
            return None

        if self._stable_level is None:
            self._stable_level = level
            self._stable_frame_id = checked_frame_id
            self._stable_count = (
                self._baseline_samples
                if baseline_trusted
                else self._candidate_count
            )
            return None

        before = self._stable_level
        pre_frame_id = self._stable_frame_id
        baseline_ready = self._stable_count >= self._baseline_samples
        self._stable_level = level
        self._stable_frame_id = checked_frame_id
        self._stable_count = (
            self._baseline_samples
            if baseline_trusted
            else self._candidate_count
        )
        if level > before:
            return None
        if not baseline_ready:
            return None
        if pre_frame_id is None:
            raise RuntimeError("stable meter baseline is missing its frame")
        return MeterLevelTransition(
            side=self._side,
            before=before,
            after=level,
            pre_frame_id=pre_frame_id,
            confirmed_frame_id=checked_frame_id,
        )

    def reset(self) -> None:
        self._candidate: int | None = None
        self._candidate_count = 0
        self._stable_level: int | None = None
        self._stable_frame_id: int | None = None
        self._stable_count = 0
        self._unknown_count = 0
        self._last_frame_id: int | None = None
