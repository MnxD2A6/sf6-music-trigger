"""Conservative, deferred resolution of confirmed SF6 super events."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from enum import Enum
import math
from numbers import Rational, Real
from typing import Callable, TYPE_CHECKING

import numpy as np

from .config import SuperScope
from .contact_hud import read_health_pair
if TYPE_CHECKING:
    from .contact_hud import ContactHudReading
from .domain import CapturedFrame, ConfirmedSuperEvent, Side
from .gauge_features import GaugeObservation
from .hud_profile import NormalizedRoi, crop_normalized


class OutcomeLabel(str, Enum):
    HIT = "HIT"
    BLOCK = "BLOCK"
    WHIFF = "WHIFF"
    NON_HIT = "NON_HIT"
    FILTERED_TIER = "FILTERED_TIER"
    RESET = "RESET"
    UNKNOWN = "UNKNOWN"


def _require_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, Rational):
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be between zero and one")
    elif not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return float(value)


def _require_nonnegative_real(name: str, value: object) -> Real:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, Rational):
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    elif not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _require_positive_real(name: str, value: object) -> Real:
    checked = _require_nonnegative_real(name, value)
    if checked == 0:
        raise ValueError(f"{name} must be positive")
    return checked


def remaining_meter_bucket(normalized_meter: Real) -> int | None:
    """Return a conservative 0/1/2/3-bar bucket for normalized meter evidence."""

    value = _require_probability("normalized_meter", normalized_meter)
    centers = (0.0, 1 / 3, 2 / 3, 1.0)
    nearest = min(range(4), key=lambda index: abs(value - centers[index]))
    return nearest if abs(value - centers[nearest]) <= 0.12 else None


@dataclass(frozen=True)
class OutcomeDecision:
    event_id: str
    side: Side
    label: OutcomeLabel
    confidence: float
    decided_at: Real
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.event_id) is not str or not self.event_id.strip():
            raise ValueError("event_id must be nonblank text")
        if not isinstance(self.side, Side):
            raise TypeError("side must be Side")
        if not isinstance(self.label, OutcomeLabel):
            raise TypeError("label must be OutcomeLabel")
        object.__setattr__(self, "confidence", _require_probability("confidence", self.confidence))
        _require_nonnegative_real("decided_at", self.decided_at)
        if type(self.evidence) is not tuple or not all(
            type(item) is str and item for item in self.evidence
        ):
            raise TypeError("evidence must contain nonempty strings")


_P1_TARGET_HEALTH = NormalizedRoi(0.55, 0.025, 0.94, 0.105)
_P2_TARGET_HEALTH = NormalizedRoi(0.06, 0.025, 0.45, 0.105)
_TRAINING_RESULT = NormalizedRoi(0.37, 0.14, 0.63, 0.27)


def _mean_luma(image: np.ndarray, roi: NormalizedRoi) -> float:
    crop = crop_normalized(image, roi)
    luma = (
        crop[..., 0].astype(np.float32) * 0.114
        + crop[..., 1].astype(np.float32) * 0.587
        + crop[..., 2].astype(np.float32) * 0.299
    )
    return float(np.mean(luma) / 255.0)


def extract_outcome_features(image: np.ndarray, side: Side) -> tuple[float, float]:
    """Return broad ROI luminance signatures, not calibrated contact evidence."""

    if not isinstance(side, Side):
        raise TypeError("side must be Side")
    health_roi = _P1_TARGET_HEALTH if side is Side.P1 else _P2_TARGET_HEALTH
    return (_mean_luma(image, health_roi), _mean_luma(image, _TRAINING_RESULT))


@dataclass(frozen=True)
class OutcomeFeatures:
    frame_id: int
    sampled_at: Real
    p1: tuple[float, float]
    p2: tuple[float, float]
    hud: ContactHudReading | None = None
    health: tuple[float | None, float | None] = (None, None)
    image_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        _require_nonnegative_real("sampled_at", self.sampled_at)
        for name, pair in (("p1", self.p1), ("p2", self.p2)):
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError(f"{name} must be a two-item tuple")
            _require_probability(f"{name} health", pair[0])
            _require_probability(f"{name} result", pair[1])

    def for_side(self, side: Side) -> tuple[float, float]:
        return self.p1 if side is Side.P1 else self.p2


@dataclass
class _PendingOutcome:
    event: ConfirmedSuperEvent
    scope: SuperScope
    baseline: tuple[float, float] | None
    suppression_until: Real
    checkpoint: Real
    hard_deadline: Real
    tier_candidate: int | None = None
    tier_streak: int = 0
    stable_tier: int | None = None
    contact_label: OutcomeLabel | None = None
    contact_streak: int = 0
    hud_baseline: ContactHudReading | None = None
    health_loss_seen: bool = False
    timer_candidate: np.ndarray | None = None
    timer_streak: int = 0
    timer_resumed: bool = False
    hud_gap_streak: int = 0
    hud_obscured: bool = False
    hud_return_streak: int = 0
    hud_resumed: bool = False
    hud_resumed_at: Real | None = None
    ca_baseline: OutcomeFeatures | None = None
    ca_previous: OutcomeFeatures | None = None
    ca_streak: int = 0


class OutcomeObserver:
    """Resolve each side independently after cinematic pixels have settled."""

    def __init__(
        self,
        *,
        suppression_seconds: Real = 0.4,
        checkpoint_seconds: Real = 3.0,
        hard_deadline_seconds: Real = 8.0,
        contact_delta: Real = 0.08,
        contact_samples: int = 2,
        tier_samples: int = 5,
        feature_capacity: int = 240,
        event_history_capacity: int = 1024,
        feature_extractor: Callable[[np.ndarray, Side], tuple[float, float]] = extract_outcome_features,
        contact_reader: Callable[[np.ndarray], ContactHudReading] | None = None,
    ) -> None:
        self._suppression_seconds = _require_nonnegative_real(
            "suppression_seconds", suppression_seconds
        )
        self._checkpoint_seconds = _require_positive_real(
            "checkpoint_seconds", checkpoint_seconds
        )
        self._hard_deadline_seconds = _require_positive_real(
            "hard_deadline_seconds", hard_deadline_seconds
        )
        if self._checkpoint_seconds >= self._hard_deadline_seconds:
            raise ValueError("checkpoint_seconds must precede hard_deadline_seconds")
        if type(contact_samples) is not int:
            raise TypeError("contact_samples must be an integer")
        if contact_samples < 2:
            raise ValueError("contact_samples must be at least two")
        if type(tier_samples) is not int:
            raise TypeError("tier_samples must be an integer")
        if tier_samples < 2:
            raise ValueError("tier_samples must be at least two")
        if type(feature_capacity) is not int:
            raise TypeError("feature_capacity must be an integer")
        if feature_capacity < 1:
            raise ValueError("feature_capacity must be positive")
        if type(event_history_capacity) is not int:
            raise TypeError("event_history_capacity must be an integer")
        if event_history_capacity < 2:
            raise ValueError("event_history_capacity must be at least two")
        if not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")
        if contact_reader is not None and not callable(contact_reader):
            raise TypeError("contact_reader must be callable")
        # The independent reader abstains when HUD geometry is unavailable;
        # scene brightness is never accepted as production contact evidence.
        self._contact_reader = contact_reader
        self._contact_delta = _require_probability("contact_delta", contact_delta)
        self._contact_samples = contact_samples
        self._tier_samples = tier_samples
        self._feature_capacity = feature_capacity
        self._event_history_capacity = event_history_capacity
        self._feature_extractor = feature_extractor
        self._features: OrderedDict[int, OutcomeFeatures] = OrderedDict()
        self._pending: dict[Side, _PendingOutcome] = {}
        self._seen_event_keys: OrderedDict[tuple[Side, str], None] = OrderedDict()
        self._last_frame_id: int | None = None
        self._last_sampled_at: Real | None = None

    @property
    def pending_sides(self) -> tuple[Side, ...]:
        return tuple(side for side in (Side.P1, Side.P2) if side in self._pending)

    def remember(self, frame: CapturedFrame) -> None:
        if type(frame) is not CapturedFrame:
            raise TypeError("frame must be CapturedFrame")
        item = OutcomeFeatures(
            frame_id=frame.frame_id,
            sampled_at=frame.monotonic_time,
            p1=self._extract(frame.image, Side.P1) if self._contact_reader is None else (0., 0.),
            p2=self._extract(frame.image, Side.P2) if self._contact_reader is None else (0., 0.),
            hud=None if self._contact_reader is None else self._contact_reader(frame.image),
            health=read_health_pair(frame.image) if self._contact_reader is None else (None, None),
            image_shape=frame.image.shape,
        )
        self._features.pop(frame.frame_id, None)
        self._features[frame.frame_id] = item
        while len(self._features) > self._feature_capacity:
            self._features.popitem(last=False)

    def begin(
        self,
        event: ConfirmedSuperEvent,
        scope: SuperScope,
        *,
        activation_at: Real,
    ) -> None:
        if not isinstance(event, ConfirmedSuperEvent):
            raise TypeError("event must be ConfirmedSuperEvent")
        if not isinstance(scope, SuperScope):
            raise TypeError("scope must be SuperScope")
        activation_time = _require_nonnegative_real("activation_at", activation_at)
        if activation_time > event.confirmed_at:
            raise ValueError("activation_at must not exceed event.confirmed_at")
        event_key = (event.side, event.event_id)
        if event_key in self._seen_event_keys:
            return
        # A new confirmed spend owns a new result window on this side.
        # Keeping the previous window would discard the new event entirely.
        self._seen_event_keys[event_key] = None
        while len(self._seen_event_keys) > self._event_history_capacity:
            self._seen_event_keys.popitem(last=False)
        baseline_item = self._features.get(event.pre_spend_frame_id)
        self._pending[event.side] = _PendingOutcome(
            event=event,
            scope=scope,
            baseline=None if baseline_item is None else baseline_item.for_side(event.side),
            suppression_until=activation_time + self._suppression_seconds,
            checkpoint=activation_time + self._checkpoint_seconds,
            hard_deadline=activation_time + self._hard_deadline_seconds,
            hud_baseline=self._contact_baseline(
                event, activation_time, baseline_item,
            ),
            ca_baseline=self._ca_baseline(event, activation_time, baseline_item),
        )
        pending = self._pending[event.side]
        # Confirmation can arrive after a short visible contact window. Carry
        # only the immediately preceding sample, never a pre-activation loss.
        for item in reversed(tuple(self._features.values())):
            if item.frame_id < event.confirmed_frame_id:
                if item.sampled_at >= pending.suppression_until and pending.ca_baseline is not None:
                    target = 1 if event.side is Side.P1 else 0
                    attacker = 1 - target
                    if (item.health[attacker] is not None and
                            abs(item.health[attacker] - pending.ca_baseline.health[attacker]) > .012):
                        pending.ca_baseline = None
                        break
                    if (item.image_shape == pending.ca_baseline.image_shape
                            and item.health[target] is not None
                            and pending.ca_baseline.health[target] - item.health[target] >= .025):
                        pending.ca_previous = item
                        pending.ca_streak = 1
                break

    def _contact_baseline(
        self, event: ConfirmedSuperEvent, activation_at: Real,
        anchor: OutcomeFeatures | None,
    ) -> ContactHudReading | None:
        if self._contact_reader is None:
            return None if anchor is None else anchor.hud
        if anchor is not None and anchor.hud is not None:
            health, drive = self._target_hud(anchor.hud, event.side)
            if health is not None and drive is not None:
                return anchor.hud
        previous: ContactHudReading | None = None
        previous_at: Real | None = None
        for item in reversed(tuple(self._features.values())):
            if item.frame_id > event.pre_spend_frame_id:
                continue
            if not (0 <= activation_at - item.sampled_at <= 1.5):
                continue
            reading = item.hud
            if reading is None:
                previous = None
                previous_at = None
                continue
            health,drive = self._target_hud(reading,event.side)
            if health is None or drive is None:
                previous = None
                previous_at = None
                continue
            if previous is not None and previous_at is not None:
                old_health,old_drive = self._target_hud(previous,event.side)
                if (0 < previous_at-item.sampled_at <= .25
                        and old_health is not None and old_drive is not None
                        and abs(old_health-health) <= .012
                        and abs(old_drive-drive) <= .02):
                    return previous
                return None
            previous=reading
            previous_at=item.sampled_at
        return None

    def update(
        self,
        frame: CapturedFrame,
        *,
        observations: tuple[GaugeObservation, GaugeObservation],
    ) -> tuple[OutcomeDecision, ...]:
        if type(frame) is not CapturedFrame:
            raise TypeError("frame must be CapturedFrame")
        self._validate_observations(frame, observations)
        if self._last_frame_id is not None and (
            frame.frame_id <= self._last_frame_id
            or frame.monotonic_time < self._last_sampled_at  # type: ignore[operator]
        ):
            return ()
        self._last_frame_id = frame.frame_id
        self._last_sampled_at = frame.monotonic_time

        decisions: list[OutcomeDecision] = []
        by_side = {item.side: item for item in observations}
        for side in (Side.P1, Side.P2):
            pending = self._pending.get(side)
            if pending is None:
                continue
            if self._contact_reader is not None:
                decision = self._update_contact_hud(frame, side, pending)
                if decision is not None:
                    decisions.append(decision)
                continue
            observation = by_side[side]
            decision = self._update_ca_health(frame, side, pending)
            if decision is not None:
                decisions.append(decision)
                continue
            trusted = observation.hud_visible and observation.spent_structure_verified
            if not trusted:
                self._clear_streaks(pending)
                if frame.monotonic_time >= pending.hard_deadline:
                    decisions.append(self._finish(
                        side, OutcomeLabel.UNKNOWN, frame.monotonic_time, 0.0,
                        ("hud_unavailable_at_deadline",),
                    ))
                continue

            if frame.monotonic_time >= pending.suppression_until:
                self._advance_tier(pending, observation)
                if (
                    pending.scope is SuperScope.SA3_CA_ONLY
                    and pending.stable_tier in (1, 2)
                ):
                    decisions.append(self._finish(
                        side, OutcomeLabel.FILTERED_TIER, frame.monotonic_time, 1.0,
                        (f"remaining_meter_{pending.stable_tier}",),
                    ))
                    continue

                contact_allowed = (
                    pending.scope is SuperScope.ALL_SUPERS
                    or pending.stable_tier == 0
                )
                if contact_allowed and pending.baseline is not None:
                    contact, evidence = self._contact(frame, side, pending)
                    if contact is not None and pending.contact_streak >= self._contact_samples:
                        decisions.append(self._finish(
                            side, contact, frame.monotonic_time, 0.9, evidence,
                        ))
                        continue
                else:
                    pending.contact_label = None
                    pending.contact_streak = 0

                checkpoint_allowed = (
                    pending.scope is SuperScope.ALL_SUPERS
                    or pending.stable_tier == 0
                )
                if (
                    frame.monotonic_time >= pending.checkpoint
                    and checkpoint_allowed
                    and pending.baseline is not None
                ):
                    decisions.append(self._finish(
                        # No contact observed is not proof of a whiff: buffs,
                        # cinematics and missed HUD samples have the same trace.
                        side, OutcomeLabel.UNKNOWN, frame.monotonic_time, 0.0,
                        ("contact_not_established",),
                    ))
                    continue

            if frame.monotonic_time >= pending.hard_deadline:
                decisions.append(self._finish(
                    side, OutcomeLabel.UNKNOWN, frame.monotonic_time, 0.0,
                    ("resolution_deadline_elapsed",),
                ))
        return tuple(decisions)

    def _ca_baseline(
        self, event: ConfirmedSuperEvent, activation_at: Real,
        anchor: OutcomeFeatures | None,
    ) -> OutcomeFeatures | None:
        if event.levels_spent != 3 or anchor is None or self._contact_reader is not None:
            return None
        # The pre-spend frame can be covered by startup VFX. Only a recent,
        # stable pair preceding that exact frame can supply its health baseline.
        previous = None
        for item in reversed(tuple(self._features.values())):
            if item.frame_id > event.pre_spend_frame_id:
                continue
            if not (0 <= activation_at - item.sampled_at <= 1.5
                    and 0 <= anchor.sampled_at - item.sampled_at <= .75):
                continue
            if item.image_shape != anchor.image_shape or None in item.health:
                previous = None
                continue
            if previous is not None and 0 < previous.sampled_at - item.sampled_at <= .25:
                if all(abs(a - b) <= .012 for a, b in zip(item.health, previous.health)):
                    return previous
                return None  # A newer changed reading cannot borrow an older plateau.
            previous = item
        return None

    def _update_ca_health(
        self, frame: CapturedFrame, side: Side, pending: _PendingOutcome,
    ) -> OutcomeDecision | None:
        baseline = pending.ca_baseline
        now = frame.monotonic_time
        if baseline is None or now < pending.suppression_until or now > pending.hard_deadline:
            return None
        current = self._features.get(frame.frame_id)
        if current is None:
            current = OutcomeFeatures(frame.frame_id, now, (0., 0.), (0., 0.),
                                      health=read_health_pair(frame.image), image_shape=frame.image.shape)
        if current.image_shape != baseline.image_shape:
            pending.ca_baseline = None
            return None
        target = 1 if side is Side.P1 else 0
        attacker = 1 - target
        # Visible attacker damage invalidates attribution even if the target
        # health is currently occluded.
        if (current.health[attacker] is not None
                and abs(baseline.health[attacker] - current.health[attacker]) > .012):
            pending.ca_baseline = None
            return None
        if current.health[target] is None:
            pending.ca_previous = None
            pending.ca_streak = 0
            return None
        loss = baseline.health[target] - current.health[target]
        # Health recovery / attacker damage breaks attribution to this spend.
        if loss < -.012:
            pending.ca_baseline = None
            return None
        previous = pending.ca_previous
        continuous = (previous is not None and 0 < now - previous.sampled_at <= .25
                      and abs(current.health[target] - previous.health[target]) <= .012)
        pending.ca_streak = (pending.ca_streak + 1 if continuous else 1) if loss >= .025 else 0
        pending.ca_previous = current
        if pending.ca_streak >= self._contact_samples:
            return self._finish(side, OutcomeLabel.HIT, now, .9, ("opponent_health_changed",))
        return None

    def _update_contact_hud(
        self, frame: CapturedFrame, side: Side, pending: _PendingOutcome,
    ) -> OutcomeDecision | None:
        """Use independently visible HUD evidence; obscured meters stay unknown.

        Timer resumption anchors the normal-match result window. Training mode
        has no changing timer, so a sustained HUD return after a real cinematic
        obscuration can anchor that window instead. Uninterrupted static HUD is
        never treated as a miss. Spent-tier filtering already happens before
        engine.begin().
        """
        now = frame.monotonic_time
        if now < pending.suppression_until:
            return None
        cached = self._features.get(frame.frame_id)
        current = cached.hud if cached is not None else self._contact_reader(frame.image)
        baseline = pending.hud_baseline
        label = None
        evidence: tuple[str, ...] = ()
        if baseline is not None and current is not None:
            before_health, before_drive = self._target_hud(baseline, side)
            health, drive = self._target_hud(current, side)
            before_timer, timer = baseline.timer_mask, current.timer_mask
            hud_available = health is not None and drive is not None
            hud_fully_obscured = health is None and drive is None
            if hud_fully_obscured:
                pending.hud_gap_streak += 1
                pending.hud_return_streak = 0
                pending.hud_resumed = False
                pending.hud_resumed_at = None
                if pending.hud_gap_streak >= 2:
                    pending.hud_obscured = True
            elif hud_available:
                pending.hud_gap_streak = 0
                if pending.hud_obscured and not pending.hud_resumed:
                    pending.hud_return_streak += 1
                    if pending.hud_return_streak >= 3:
                        pending.hud_resumed = True
                        if pending.hud_resumed_at is None:
                            pending.hud_resumed_at = now
            else:
                # One missing meter is an ordinary local abstention, not proof
                # that a cinematic hid the complete target HUD.
                pending.hud_gap_streak = 0
                pending.hud_return_streak = 0
            if before_timer is not None and timer is not None and np.mean(before_timer != timer) > .05:
                if pending.timer_candidate is not None and np.mean(pending.timer_candidate != timer) < .04:
                    pending.timer_streak += 1
                else:
                    pending.timer_candidate = timer.copy()
                    pending.timer_streak = 1
                if pending.timer_streak >= 3:
                    pending.timer_resumed = True
            else:
                pending.timer_candidate = None
                pending.timer_streak = 0
            if before_health is not None and health is not None:
                loss = before_health - health
                stable_training_hud_return = (
                    pending.hud_resumed
                    and pending.hud_resumed_at is not None
                    and now - pending.hud_resumed_at >= Fraction(1, 2)
                )
                result_window_established = (
                    pending.timer_resumed or stable_training_hud_return
                )
                result_window_evidence = (
                    "timer_resumed" if pending.timer_resumed else "hud_resumed"
                )
                if loss >= .025:
                    pending.health_loss_seen = True
                    label = OutcomeLabel.HIT
                    evidence = ("opponent_health_changed",)
                elif (
                    abs(loss) <= .012 and not pending.health_loss_seen
                    and before_drive is not None and drive is not None
                    and before_drive - drive >= .06 and result_window_established
                ):
                    label = OutcomeLabel.BLOCK
                    evidence = (
                        "opponent_health_unchanged",
                        "opponent_drive_changed",
                        result_window_evidence,
                    )
                elif (
                    now >= pending.checkpoint
                    and abs(loss) <= .012 and not pending.health_loss_seen
                    and before_drive is not None and drive is not None
                    and abs(before_drive - drive) <= .02
                    and result_window_established
                ):
                    label = (
                        OutcomeLabel.WHIFF
                        if pending.timer_resumed
                        else OutcomeLabel.NON_HIT
                    )
                    evidence = (
                        "opponent_health_unchanged",
                        "opponent_drive_unchanged",
                        result_window_evidence,
                    )
        if label is None:
            pending.contact_label = None
            pending.contact_streak = 0
        elif label is pending.contact_label:
            pending.contact_streak += 1
        else:
            pending.contact_label = label
            pending.contact_streak = 1
        if label is not None and pending.contact_streak >= self._contact_samples:
            return self._finish(side, label, now, .9, evidence)
        if now >= pending.hard_deadline:
            return self._finish(side, OutcomeLabel.UNKNOWN, now, 0., ("contact_not_established",))
        return None

    @staticmethod
    def _target_hud(reading: ContactHudReading, attacker: Side) -> tuple[float | None, float | None]:
        if attacker is Side.P1:
            return reading.p2_health, reading.p2_drive
        return reading.p1_health, reading.p1_drive

    def reset(self, reason: str) -> tuple[OutcomeDecision, ...]:
        if type(reason) is not str or not reason.strip():
            raise ValueError("reason must be nonblank text")
        decisions = tuple(
            OutcomeDecision(
                self._pending[side].event.event_id,
                side,
                OutcomeLabel.RESET,
                0.0,
                self._pending[side].event.confirmed_at,
                ("reset",),
            )
            for side in (Side.P1, Side.P2)
            if side in self._pending
        )
        self._pending.clear()
        self._features.clear()
        self._seen_event_keys.clear()
        self._last_frame_id = None
        self._last_sampled_at = None
        return decisions

    def _validate_observations(
        self,
        frame: CapturedFrame,
        observations: tuple[GaugeObservation, GaugeObservation],
    ) -> None:
        if type(observations) is not tuple or len(observations) != 2:
            raise TypeError("observations must be a two-item tuple")
        for index, (want_side, item) in enumerate(zip((Side.P1, Side.P2), observations)):
            if type(item) is not GaugeObservation:
                raise TypeError(f"observation {index} must be GaugeObservation")
            if item.side is not want_side:
                raise ValueError("observations must be ordered P1 then P2")
            if item.frame_id != frame.frame_id or item.sampled_at != frame.monotonic_time:
                raise ValueError("observations must belong to the supplied frame")

    def _advance_tier(self, pending: _PendingOutcome, observation: GaugeObservation) -> None:
        bucket = (
            None
            if observation.normalized_meter is None
            else remaining_meter_bucket(observation.normalized_meter)
        )
        if bucket is None:
            pending.tier_candidate = None
            pending.tier_streak = 0
            return
        if bucket == pending.tier_candidate:
            pending.tier_streak += 1
        else:
            pending.tier_candidate = bucket
            pending.tier_streak = 1
        if pending.tier_streak >= self._tier_samples:
            pending.stable_tier = bucket

    def _contact(
        self,
        frame: CapturedFrame,
        side: Side,
        pending: _PendingOutcome,
    ) -> tuple[OutcomeLabel | None, tuple[str, ...]]:
        current = self._extract(frame.image, side)
        health_delta = abs(current[0] - pending.baseline[0])  # type: ignore[index]
        result_delta = abs(current[1] - pending.baseline[1])  # type: ignore[index]
        contact: OutcomeLabel | None = None
        evidence: tuple[str, ...] = ()
        if (
            health_delta >= self._contact_delta
            and self._feature_extractor is not extract_outcome_features
        ):
            # Default brightness changes include parry VFX and cinematics.
            # Only the independent health reader may establish default HIT;
            # explicitly supplied extractors retain their calibrated contract.
            contact = OutcomeLabel.HIT
            evidence = ("opponent_health_changed",)
        elif (
            result_delta >= self._contact_delta
            and self._feature_extractor is not extract_outcome_features
        ):
            # The default ROI is scene pixels, not a calibrated BLOCK indicator.
            # Leave it unresolved until the checkpoint/fallback instead of
            # ending observation early on a cinematic flash or a buff.
            contact = OutcomeLabel.BLOCK
            evidence = ("training_result_changed",)
        if contact is None:
            pending.contact_label = None
            pending.contact_streak = 0
        elif contact is pending.contact_label:
            pending.contact_streak += 1
        else:
            pending.contact_label = contact
            pending.contact_streak = 1
        return contact, evidence

    @staticmethod
    def _clear_streaks(pending: _PendingOutcome) -> None:
        pending.tier_candidate = None
        pending.tier_streak = 0
        pending.contact_label = None
        pending.contact_streak = 0

    def _extract(self, image: np.ndarray, side: Side) -> tuple[float, float]:
        value = self._feature_extractor(image, side)
        if type(value) is not tuple or len(value) != 2:
            raise TypeError("feature extractor must return a two-item tuple")
        return (
            _require_probability("health feature", value[0]),
            _require_probability("result feature", value[1]),
        )

    def _finish(
        self,
        side: Side,
        label: OutcomeLabel,
        decided_at: Real,
        confidence: float,
        evidence: tuple[str, ...],
    ) -> OutcomeDecision:
        pending = self._pending.pop(side)
        return OutcomeDecision(
            pending.event.event_id,
            side,
            label,
            confidence,
            decided_at,
            evidence,
        )
