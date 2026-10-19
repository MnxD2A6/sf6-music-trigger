from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Rational, Real
from typing import Literal
import uuid

from sf6_music_trigger.domain import (
    CandidateCancelReason,
    CancelledSuperCandidate,
    ConfirmedSuperEvent,
    GaugeTrackerUpdate,
    Side,
    SuperCandidateEvent,
    TrackerState,
)
from sf6_music_trigger.gauge_features import GaugeObservation
from sf6_music_trigger.meter_level_features import MeterLevelReading
from sf6_music_trigger.meter_level_tracker import (
    MeterLevelTransition,
    StableMeterLevelTracker,
)


Classification = Literal["full", "spent", "ambiguous"]

_MAX_ZERO_LEVEL_COLOUR_FILL = 0.65
_MAX_REDUCED_LEVEL_COLOUR_FILL = 0.78
_MAX_PARTIAL_CANDIDATE_COLOUR_FILL = 0.10
_MIN_THREE_LEVEL_COLOUR_FILL = 0.72
_MIN_UNTRUSTED_GAIN_SECONDS = Fraction(7, 2)
_MIN_CLEAN_DROP_SECONDS = Fraction(3, 5)
_MIN_WRAP_FILL = Fraction(3, 5)
_MAX_WRAP_EVIDENCE_AGE_SECONDS = Fraction(1, 1)
_MAX_DELAYED_STRUCTURE_ZERO_AGE_SECONDS = Fraction(5, 4)
_CANDIDATE_EVIDENCE_GRACE_SECONDS = Fraction(3, 4)
_MAX_CANDIDATE_SECONDS = Fraction(3, 1)
_THREE_LEVEL_BASELINE_SUPPORT_SAMPLES = 3
_UNREADABLE_FULL_BASELINE_SAMPLES = 10
_DELAYED_STRUCTURE_ZERO_SAMPLES = 2


def _require_threshold(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, Rational):
        if value <= Fraction(1, 2) or value > 1:
            raise ValueError(f"{name} must be greater than 0.5 and at most 1")
        return
    if not math.isfinite(value) or value <= 0.5 or value > 1:
        raise ValueError(f"{name} must be greater than 0.5 and at most 1")


def _require_exact_int(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")


def _require_minimum_spend_seconds(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("minimum_spend_seconds must be a real number")
    floor = Fraction(1, 10)
    if isinstance(value, Rational):
        if value < floor:
            raise ValueError("minimum_spend_seconds must be at least one tenth")
        return
    if not math.isfinite(value) or value < floor:
        raise ValueError("minimum_spend_seconds must be at least one tenth")


@dataclass(frozen=True)
class GaugeTrackerConfig:
    full_threshold: Real = 0.78
    spent_threshold: Real = 0.72
    full_samples: int = 5
    evidence_window: int = 5
    spent_samples: int = 3
    minimum_spend_seconds: Real = 0.10
    ambiguous_disarm_samples: int = 3
    rearm_samples: int = 5

    def __post_init__(self) -> None:
        _require_threshold("full_threshold", self.full_threshold)
        _require_threshold("spent_threshold", self.spent_threshold)
        _require_exact_int("full_samples", self.full_samples, minimum=5)
        _require_exact_int(
            "evidence_window",
            self.evidence_window,
            minimum=5,
            maximum=5,
        )
        _require_exact_int(
            "spent_samples",
            self.spent_samples,
            minimum=3,
            maximum=5,
        )
        _require_minimum_spend_seconds(self.minimum_spend_seconds)
        _require_exact_int(
            "ambiguous_disarm_samples",
            self.ambiguous_disarm_samples,
            minimum=1,
        )
        _require_exact_int("rearm_samples", self.rearm_samples, minimum=5)


def _classify_confidences(
    full_confidence: Real,
    spent_confidence: Real,
    config: GaugeTrackerConfig,
) -> Classification:
    full_hit = full_confidence >= config.full_threshold
    spent_hit = spent_confidence >= config.spent_threshold
    if full_hit == spent_hit:
        return "ambiguous"
    if full_hit:
        return "full"
    return "spent"


def _colour_conflicts_with_level(observation: GaugeObservation) -> bool:
    level = observation.meter_level
    normalized_meter = observation.normalized_meter
    return (
        normalized_meter is not None
        and (
            (level == 0 and normalized_meter > _MAX_ZERO_LEVEL_COLOUR_FILL)
            or (
                level in (1, 2)
                and normalized_meter > _MAX_REDUCED_LEVEL_COLOUR_FILL
            )
            or (
                level == 3
                and normalized_meter < _MIN_THREE_LEVEL_COLOUR_FILL
            )
        )
    )


class GaugeTracker:
    def __init__(
        self,
        side: Side,
        profile_id: str,
        config: GaugeTrackerConfig,
    ) -> None:
        if not isinstance(side, Side):
            raise TypeError("side must be a Side")
        if type(profile_id) is not str:
            raise TypeError("profile_id must be a string")
        if not profile_id.strip():
            raise ValueError("profile_id must be nonblank")
        if not isinstance(config, GaugeTrackerConfig):
            raise TypeError("config must be a GaugeTrackerConfig")

        self._side = side
        self._profile_id = profile_id
        self._config = config
        self._state = TrackerState.DISARMED
        self._evidence: deque[tuple[GaugeObservation, Classification]] = deque(
            maxlen=config.evidence_window
        )
        self._consecutive_full = 0
        self._candidate_ambiguous = 0
        self._candidate_visual_disruption = False
        self._color_meter_mode = False
        self._rearm_streak = 0
        self._pre_spend_frame_id: int | None = None
        self._last_frame_id: int | None = None
        self._last_sampled_at: Real | None = None
        self._level_tracker = StableMeterLevelTracker(
            side,
            stable_samples=3,
            baseline_samples=10,
            unknown_reset_samples=10,
        )
        self._level_meter_mode = False
        self._untrusted_gain_level: int | None = None
        self._untrusted_gain_started_at: Real | None = None
        self._untrusted_drop_level: int | None = None
        self._approved_drop_level: int | None = None
        self._clean_drop_level: int | None = None
        self._clean_drop_started_at: Real | None = None
        self._approved_gain_level: int | None = None
        self._last_trusted_meter_fill: Real | None = None
        self._last_trusted_meter_fill_at: Real | None = None
        self._active_candidate: SuperCandidateEvent | None = None
        self._retired_candidate: SuperCandidateEvent | None = None
        self._retired_recovery_count = 0
        self._candidate_level: int | None = None
        self._candidate_level_count = 0
        self._candidate_deadline: Real | None = None
        self._candidate_conflict_level: int | None = None
        self._candidate_conflict_count = 0
        self._partial_residual_conflict_baseline: int | None = None
        self._stable_three_colour_count = 0
        self._three_level_colour_available = False
        self._three_level_baseline_support_count = 0
        self._three_level_baseline_supported = False
        self._unreadable_full_baseline_count = 0
        self._fractional_history: deque[GaugeObservation] = deque(maxlen=64)
        self._fractional_zero_anchor: tuple[Real, Real] | None = None
        self._fractional_zero_count = 0
        self._fractional_zero_last_at: Real | None = None
        self._fractional_zero_ready = False
        self._fractional_drop_approved = False
        self._delayed_three_zero_times: deque[Real] = deque(maxlen=4)
        self._delayed_three_zero_bridge = False

    @property
    def state(self) -> TrackerState:
        return self._state

    @property
    def stable_meter_level(self) -> int | None:
        if not self._level_meter_mode:
            return None
        return self._level_tracker.stable_level

    def update(
        self,
        observation: GaugeObservation,
    ) -> GaugeTrackerUpdate:
        if not isinstance(observation, GaugeObservation):
            raise TypeError("observation must be a GaugeObservation")
        if observation.side is not self._side:
            raise ValueError("observation side does not match tracker side")
        if (
            self._last_frame_id is not None
            and observation.frame_id <= self._last_frame_id
        ):
            raise ValueError("frame_id must increase strictly")
        if (
            self._last_sampled_at is not None
            and observation.sampled_at < self._last_sampled_at
        ):
            raise ValueError("sampled_at must not move backward")

        self._last_frame_id = observation.frame_id
        self._last_sampled_at = observation.sampled_at

        if observation.meter_level_supported:
            if not self._level_meter_mode:
                self._clear_detection_history()
                self._level_tracker.reset()
                self._level_meter_mode = True
            return self._update_meter_level(observation)

        if self._level_meter_mode:
            cancellation = self._cancel_candidate(
                CandidateCancelReason.CAPTURE_RESET,
                observation.sampled_at,
            )
            self._level_tracker.reset()
            self._level_meter_mode = False
            if cancellation.transition is not None:
                self._clear_detection_history()
                self._state = TrackerState.DISARMED
                return cancellation

        if not observation.hud_visible:
            self._clear_detection_history()
            self._state = TrackerState.DISARMED
            return GaugeTrackerUpdate()

        classification = _classify_confidences(
            observation.full_confidence,
            observation.spent_confidence,
            self._config,
        )
        if observation.normalized_meter is not None:
            self._color_meter_mode = True
        self._evidence.append((observation, classification))

        if self._state is TrackerState.DISARMED:
            if classification == "full":
                self._consecutive_full += 1
                if self._consecutive_full >= self._config.full_samples:
                    self._pre_spend_frame_id = observation.frame_id
                    self._state = TrackerState.ARMED
            else:
                self._consecutive_full = 0
            return GaugeTrackerUpdate()

        if self._state is TrackerState.ARMED:
            if classification == "full":
                self._pre_spend_frame_id = observation.frame_id
                return GaugeTrackerUpdate()
            self._state = TrackerState.SPEND_CANDIDATE
            return self._process_candidate(observation, classification)

        if self._state is TrackerState.SPEND_CANDIDATE:
            return self._process_candidate(observation, classification)

        if self._state is TrackerState.COOLDOWN:
            if classification == "full":
                self._rearm_streak += 1
                if self._rearm_streak >= self._config.rearm_samples:
                    self._rearm_streak = 0
                    self._pre_spend_frame_id = observation.frame_id
                    self._state = TrackerState.ARMED
            else:
                self._rearm_streak = 0
            return GaugeTrackerUpdate()

        return GaugeTrackerUpdate()

    def reset(
        self,
        reason: str,
        decided_at: Real | None = None,
    ) -> GaugeTrackerUpdate:
        if type(reason) is not str:
            raise TypeError("reason must be a string")
        if not reason.strip():
            raise ValueError("reason must be nonblank")
        if decided_at is not None:
            if isinstance(decided_at, bool) or not isinstance(decided_at, Real):
                raise TypeError("decided_at must be a real number or None")
            if isinstance(decided_at, Rational):
                if decided_at < 0:
                    raise ValueError("decided_at must be finite and nonnegative")
            elif not math.isfinite(decided_at) or decided_at < 0:
                raise ValueError("decided_at must be finite and nonnegative")
        cancellation = self._cancel_candidate(
            self._reset_cancel_reason(reason),
            self._last_sampled_at if decided_at is None else decided_at,
        )
        self._clear_detection_history()
        self._last_frame_id = None
        self._last_sampled_at = None
        self._level_tracker.reset()
        self._level_meter_mode = False
        self._state = TrackerState.DISARMED
        return cancellation

    def _update_meter_level(
        self,
        observation: GaugeObservation,
    ) -> GaugeTrackerUpdate:
        self._update_unreadable_full_baseline(observation)
        self._update_fractional_zero_evidence(observation)
        self._update_partial_residual_conflict(observation)
        self._update_delayed_three_zero_evidence(observation)
        candidate_update = self._candidate_transition(observation)
        three_level_colour_available = self._three_level_colour_available
        three_level_baseline_supported = self._three_level_baseline_supported
        authority = self._authority_transition(observation)
        self._update_three_level_baseline_support(observation)
        if authority is None:
            if isinstance(candidate_update.transition, SuperCandidateEvent):
                self._state = TrackerState.SPEND_CANDIDATE
            return candidate_update

        if (
            self._side is Side.P1
            and authority.before == 3
            and three_level_colour_available
            and not three_level_baseline_supported
        ):
            return GaugeTrackerUpdate()

        if authority.after == 0:
            self._partial_residual_conflict_baseline = None

        # Confirmation wins if both gates become ready on this frame. Returning
        # only the candidate used to consume and lose the authority transition.
        active_candidate = self._active_candidate or self._retired_candidate
        event = ConfirmedSuperEvent(
            event_id=uuid.uuid4().hex,
            side=self._side,
            confirmed_at=observation.sampled_at,
            pre_spend_frame_id=authority.pre_frame_id,
            confirmed_frame_id=authority.confirmed_frame_id,
            profile_id=self._profile_id,
            meter_before=authority.before,
            meter_after=authority.after,
            levels_spent=authority.levels_spent,
            candidate_id=(
                active_candidate.candidate_id
                if active_candidate is not None
                else None
            ),
        )
        if active_candidate is not None:
            self._clear_candidate_state()
            self._retired_candidate = None
            self._retired_recovery_count = 0
        return GaugeTrackerUpdate(event)

    def _update_fractional_zero_evidence(self, observation: GaugeObservation) -> None:
        """A 1->0 spend can preserve the unfinished fractional bar.

        Do not generalize this to a three-bar baseline: transient 0/3 glyphs
        remain subject to the existing colour veto. Require recent raw one-bar
        history as well as the debounced baseline, then a sustained matching
        zero. This never seeds a baseline from the current zero itself.
        """
        now = observation.sampled_at
        history = self._fractional_history
        while history and now - history[0].sampled_at > 1:
            history.popleft()
        fill = observation.normalized_meter
        valid = (
            observation.hud_visible
            and observation.meter_level_confidence >= .9
            and observation.spent_structure_verified
            and fill is not None and .30 <= fill <= .90
        )
        self._fractional_zero_ready = False
        if (self._fractional_zero_last_at is not None
                and now - self._fractional_zero_last_at > .15):
            self._fractional_zero_anchor = None
            self._fractional_zero_count = 0
            self._fractional_zero_last_at = None
        if (observation.hud_visible and observation.meter_level is None
                and self._fractional_zero_anchor is not None
                and self._fractional_zero_last_at is not None
                and now - self._fractional_zero_last_at <= .10):
            # A bounded unreadable gap supplies no positive evidence, but does
            # not discard the preceding agreeing zero samples.
            history.append(observation)
            return
        eligible = (
            valid and observation.meter_level == 0
            and self._level_tracker.stable_level == 1
            and self._level_tracker.baseline_ready
        )
        if not eligible:
            self._fractional_zero_anchor = None
            self._fractional_zero_count = 0
            self._fractional_zero_last_at = None
        else:
            anchor = self._fractional_zero_anchor
            if anchor is None:
                ones = [o for o in history if o.meter_level == 1
                        and o.hud_visible and o.meter_level_confidence >= .9
                        and o.spent_structure_verified
                        and o.normalized_meter is not None
                        and abs(o.normalized_meter - fill) <= .12]
                if (len(ones) >= 8
                        and ones[-1].sampled_at - ones[0].sampled_at >= .4
                        and now - ones[-1].sampled_at <= .15):
                    fills = sorted(o.normalized_meter for o in ones[-5:])
                    anchor = (now, fills[len(fills) // 2])
                    self._fractional_zero_anchor = anchor
            if anchor is not None:
                started, before_fill = anchor
                if now - started > .65 or abs(fill - before_fill) > .12:
                    self._fractional_zero_anchor = None
                    self._fractional_zero_count = 0
                    self._fractional_zero_last_at = None
                else:
                    self._fractional_zero_count += 1
                    self._fractional_zero_last_at = now
                    self._fractional_zero_ready = (
                        self._fractional_zero_count >= 4 and now - started >= .2
                    )
        history.append(observation)

    def _update_delayed_three_zero_evidence(
        self,
        observation: GaugeObservation,
    ) -> None:
        """Bridge P1's zero glyph to a slightly later SA3 structure break.

        Some capture timings expose the spent zero digits before the gauge
        animation becomes structurally disrupted. Keep that rejected zero
        evidence briefly, but require the later strong disruption before it
        can participate in candidate or authority tracking.
        """
        now = observation.sampled_at
        zero_times = self._delayed_three_zero_times
        while (
            zero_times
            and now - zero_times[0]
            > _MAX_DELAYED_STRUCTURE_ZERO_AGE_SECONDS
        ):
            zero_times.popleft()

        eligible_baseline = (
            self._side is Side.P1
            and self._level_tracker.stable_level == 3
            and self._level_tracker.baseline_ready
            and self._three_level_baseline_supported
        )
        if not eligible_baseline:
            zero_times.clear()
            self._delayed_three_zero_bridge = False
            return

        lagging_colour_zero = (
            observation.hud_visible
            and observation.meter_level_supported
            and observation.meter_level == 0
            and observation.meter_level_confidence >= 0.9
            and observation.spent_structure_verified
            and _colour_conflicts_with_level(observation)
        )
        if lagging_colour_zero:
            zero_times.append(now)

        self._delayed_three_zero_bridge = (
            len(zero_times) >= _DELAYED_STRUCTURE_ZERO_SAMPLES
            and now - zero_times[-1]
            <= _MAX_DELAYED_STRUCTURE_ZERO_AGE_SECONDS
            and observation.hud_visible
            and observation.meter_level_supported
            and observation.meter_level is None
            and not observation.spent_structure_verified
        )

    def _candidate_transition(
        self,
        observation: GaugeObservation,
    ) -> GaugeTrackerUpdate:
        raw_level = self._raw_trustworthy_level(observation)
        active_candidate = self._active_candidate
        if active_candidate is not None:
            if raw_level is None or raw_level == active_candidate.meter_after:
                self._clear_candidate_conflict_streak()
            else:
                if raw_level == self._candidate_conflict_level:
                    self._candidate_conflict_count += 1
                else:
                    self._candidate_conflict_level = raw_level
                    self._candidate_conflict_count = 1
                if self._candidate_conflict_count >= 3:
                    return self._cancel_candidate(
                        (CandidateCancelReason.LEVEL_RECOVERED
                         if raw_level == active_candidate.meter_before
                         else CandidateCancelReason.LEVEL_CONFLICT),
                        observation.sampled_at,
                    )
            deadline = self._candidate_deadline
            if deadline is None:
                raise RuntimeError("active candidate is missing its deadline")
            hard_deadline = active_candidate.suspected_at + _MAX_CANDIDATE_SECONDS
            if (
                observation.sampled_at <= deadline
                and raw_level == active_candidate.meter_after
                and not _colour_conflicts_with_level(observation)
            ):
                deadline = min(
                    hard_deadline,
                    observation.sampled_at + _CANDIDATE_EVIDENCE_GRACE_SECONDS,
                )
                self._candidate_deadline = deadline
            if observation.sampled_at >= min(deadline, hard_deadline):
                return self._cancel_candidate(
                    CandidateCancelReason.CONFIRMATION_TIMEOUT,
                    observation.sampled_at,
                )
            return GaugeTrackerUpdate()

        stable_level = self._level_tracker.stable_level
        retired = self._retired_candidate
        if retired is not None:
            recovered = (
                raw_level == retired.meter_before
                and not _colour_conflicts_with_level(observation)
            )
            self._retired_recovery_count = (
                self._retired_recovery_count + 1 if recovered else 0
            )
            new_baseline = (
                stable_level is not None
                and stable_level != retired.meter_before
                and self._level_tracker.baseline_ready
            )
            if self._retired_recovery_count >= 3 or new_baseline:
                self._retired_candidate = None
                self._retired_recovery_count = 0
            else:
                # A timeout/conflict retires this spend episode, not merely its
                # UUID. Late authority keeps its ID; it cannot restart audio.
                self._clear_candidate_streak()
                return GaugeTrackerUpdate()
        candidate_classification = _classify_confidences(
            observation.full_confidence,
            observation.spent_confidence,
            self._config,
        )
        # R23 167.067/167.100/167.167s: the P2 zero glyph agrees three times,
        # but the third color score dips from .739 to .691 during the effect.
        # Keep two preceding strong samples; do not rebuild evidence from weak
        # color alone, relax glyph agreement, or change authoritative confirmation.
        p2_color_flicker = (
            self._side is Side.P2
            and raw_level == 0
            and self._candidate_level == 0
            and self._candidate_level_count >= 2
            and observation.meter_level_confidence >= 0.9
            and not observation.spent_structure_verified
            and observation.full_confidence < self._config.full_threshold
            and observation.spent_confidence >= 0.60
        )
        if (
            raw_level is None
            or stable_level is None
            or not self._level_tracker.baseline_ready
            or (
                self._side is Side.P1
                and stable_level == 3
                and self._three_level_colour_available
                and not self._three_level_baseline_supported
            )
            or raw_level >= stable_level
            or (_colour_conflicts_with_level(observation)
                and not self._fractional_zero_ready)
            or (
                self._partial_residual_conflict_baseline is not None
                and raw_level != 0
            )
            or (candidate_classification != "spent" and not p2_color_flicker
                and not self._fractional_zero_ready)
        ):
            self._clear_candidate_streak()
            return GaugeTrackerUpdate()

        if raw_level == self._candidate_level:
            self._candidate_level_count += 1
        else:
            self._candidate_level = raw_level
            self._candidate_level_count = 1
        if self._delayed_three_zero_bridge and raw_level == 0:
            # The preceding zero glyphs were vetoed only because the colour
            # animation lagged. The delayed structure break completes that
            # same candidate instead of starting its evidence from scratch.
            self._candidate_level_count = max(
                self._candidate_level_count,
                self._config.spent_samples,
            )
        if self._candidate_level_count < 3:
            return GaugeTrackerUpdate()

        pre_spend_frame_id = self._level_tracker.stable_frame_id
        if pre_spend_frame_id is None:
            raise RuntimeError("ready meter baseline is missing its frame")
        candidate = SuperCandidateEvent(
            candidate_id=uuid.uuid4().hex,
            side=self._side,
            suspected_at=observation.sampled_at,
            pre_spend_frame_id=pre_spend_frame_id,
            candidate_frame_id=observation.frame_id,
            profile_id=self._profile_id,
            meter_before=stable_level,
            meter_after=raw_level,
            evidence=("stable_downward_level",),
        )
        self._active_candidate = candidate
        self._candidate_deadline = candidate.suspected_at + _CANDIDATE_EVIDENCE_GRACE_SECONDS
        self._clear_candidate_streak()
        return GaugeTrackerUpdate(candidate)

    def _update_three_level_baseline_support(
        self,
        observation: GaugeObservation,
    ) -> None:
        """Require fresh colour-backed evidence before a three-level spend.

        Round-intro animation can briefly manufacture a stable numeric 3 and
        then 0.  Waiting for three direct, structurally valid full-colour
        samples after level 3 becomes stable rejects that transition without
        changing level 1/2 detection.
        """
        if self._level_tracker.stable_level != 3:
            self._three_level_colour_available = False
            self._three_level_baseline_support_count = 0
            self._three_level_baseline_supported = False
            return
        if observation.normalized_meter is not None:
            self._three_level_colour_available = True
        if self._three_level_baseline_supported:
            return
        supported = (
            observation.hud_visible
            and observation.meter_level == 3
            and observation.meter_level_confidence >= 0.9
            and observation.spent_structure_verified
            and observation.normalized_meter is not None
            and observation.normalized_meter >= _MIN_THREE_LEVEL_COLOUR_FILL
        )
        self._three_level_baseline_support_count = (
            self._three_level_baseline_support_count + 1 if supported else 0
        )
        self._three_level_baseline_supported = (
            self._three_level_baseline_support_count
            >= _THREE_LEVEL_BASELINE_SUPPORT_SAMPLES
        )

    def _update_unreadable_full_baseline(
        self,
        observation: GaugeObservation,
    ) -> None:
        """Count sustained full colour when a bright scene hides P1's glyph."""
        eligible = (
            self._side is Side.P1
            and self._level_tracker.stable_level is None
            and observation.hud_visible
            and observation.meter_level is None
            and observation.spent_structure_verified
            and observation.normalized_meter is not None
            and observation.normalized_meter >= _MIN_THREE_LEVEL_COLOUR_FILL
            and _classify_confidences(
                observation.full_confidence,
                observation.spent_confidence,
                self._config,
            ) == "full"
        )
        self._unreadable_full_baseline_count = (
            self._unreadable_full_baseline_count + 1 if eligible else 0
        )

    def _raw_trustworthy_level(
        self,
        observation: GaugeObservation,
    ) -> int | None:
        if not observation.hud_visible or not observation.meter_level_supported:
            return None
        if self._delayed_three_zero_bridge:
            return 0
        return observation.meter_level

    def _cancel_candidate(
        self,
        reason: CandidateCancelReason,
        decided_at: Real | None,
    ) -> GaugeTrackerUpdate:
        candidate = self._active_candidate
        if candidate is None:
            return GaugeTrackerUpdate()
        if decided_at is None:
            raise RuntimeError("active candidate is missing a decision time")
        cancellation = CancelledSuperCandidate(
            candidate_id=candidate.candidate_id,
            side=self._side,
            decided_at=decided_at,
            reason=reason,
        )
        if reason in {
            CandidateCancelReason.CONFIRMATION_TIMEOUT,
            CandidateCancelReason.LEVEL_CONFLICT,
        }:
            self._retired_candidate = candidate
            self._retired_recovery_count = 0
        self._clear_candidate_state()
        return GaugeTrackerUpdate(cancellation)

    @staticmethod
    def _reset_cancel_reason(reason: str) -> CandidateCancelReason:
        normalized_reason = reason.casefold()
        if any(
            token in normalized_reason
            for token in ("capture", "window", "geometry", "session")
        ):
            return CandidateCancelReason.CAPTURE_RESET
        return CandidateCancelReason.TRACKER_RESET

    def _clear_candidate_streak(self) -> None:
        self._candidate_level = None
        self._candidate_level_count = 0

    def _clear_candidate_conflict_streak(self) -> None:
        self._candidate_conflict_level = None
        self._candidate_conflict_count = 0

    def _clear_candidate_state(self) -> None:
        self._active_candidate = None
        self._candidate_deadline = None
        self._clear_candidate_streak()
        self._clear_candidate_conflict_streak()

    def _authority_transition(
        self, observation: GaugeObservation
    ) -> MeterLevelTransition | None:
        level = (
            0
            if self._delayed_three_zero_bridge
            else observation.meter_level
        )
        normalized_meter = observation.normalized_meter
        stable_level = self._level_tracker.stable_level
        baseline_requires_trust = (
            level is not None
            and stable_level is not None
            and level > stable_level
        )
        colour_conflicts_with_level = (
            (_colour_conflicts_with_level(observation)
             and not self._fractional_zero_ready)
            or (
                self._partial_residual_conflict_baseline is not None
                and level != 0
            )
        )
        recent_wrap_evidence = (
            baseline_requires_trust
            and level == stable_level + 1
            and self._last_trusted_meter_fill is not None
            and self._last_trusted_meter_fill_at is not None
            and self._last_trusted_meter_fill >= _MIN_WRAP_FILL
            and observation.sampled_at - self._last_trusted_meter_fill_at
            <= _MAX_WRAP_EVIDENCE_AGE_SECONDS
        )
        gain_approval_just_started = False
        if not baseline_requires_trust:
            self._approved_gain_level = None
        elif level == self._approved_gain_level:
            pass
        elif (
            observation.spent_structure_verified
            and not colour_conflicts_with_level
        ) or recent_wrap_evidence:
            self._approved_gain_level = level
            gain_approval_just_started = True
        else:
            self._approved_gain_level = None
        gain_is_approved = (
            baseline_requires_trust and level == self._approved_gain_level
        )
        untrusted_gain = baseline_requires_trust and not gain_is_approved
        if untrusted_gain:
            if level != self._untrusted_gain_level:
                self._untrusted_gain_level = level
                self._untrusted_gain_started_at = observation.sampled_at
            gain_started_at = self._untrusted_gain_started_at
            if gain_started_at is None:
                raise RuntimeError("untrusted meter gain is missing its start time")
            preserve_stable_baseline = (
                observation.sampled_at - gain_started_at
                < _MIN_UNTRUSTED_GAIN_SECONDS
            )
        else:
            self._untrusted_gain_level = None
            self._untrusted_gain_started_at = None
            preserve_stable_baseline = False
        preserve_stable_baseline = (
            preserve_stable_baseline or gain_approval_just_started
        )
        downward_level = (
            level is not None
            and stable_level is not None
            and level < stable_level
        )
        if self._fractional_drop_approved and not self._fractional_zero_ready:
            # A fractional exception is a live evidence lease, not the sticky
            # approval used by the original disrupted/clean-drop paths.
            self._approved_drop_level = None
            self._fractional_drop_approved = False
            self._clean_drop_level = None
            self._clean_drop_started_at = None
        if not downward_level:
            self._approved_drop_level = None
            self._fractional_drop_approved = False
            self._clean_drop_level = None
            self._clean_drop_started_at = None
        elif level == self._approved_drop_level:
            pass
        elif colour_conflicts_with_level:
            self._approved_drop_level = None
            self._clean_drop_level = None
            self._clean_drop_started_at = None
        elif self._fractional_zero_ready or not observation.spent_structure_verified:
            self._approved_drop_level = level
            self._fractional_drop_approved = self._fractional_zero_ready
            self._clean_drop_level = None
            self._clean_drop_started_at = None
        else:
            if level != self._clean_drop_level:
                self._clean_drop_level = level
                self._clean_drop_started_at = observation.sampled_at
            clean_drop_started_at = self._clean_drop_started_at
            if clean_drop_started_at is None:
                raise RuntimeError("clean meter drop is missing its start time")
            if (
                observation.sampled_at - clean_drop_started_at
                >= _MIN_CLEAN_DROP_SECONDS
            ):
                self._approved_drop_level = level
            else:
                self._approved_drop_level = None
        drop_is_approved = (
            downward_level and level == self._approved_drop_level
        )
        preserve_unapproved_drop = downward_level and not drop_is_approved
        untrusted_drop = (
            downward_level
            and not observation.spent_structure_verified
        )
        preserve_first_untrusted_drop = False
        if untrusted_drop:
            if level != self._untrusted_drop_level:
                self._untrusted_drop_level = level
                preserve_first_untrusted_drop = True
        else:
            self._untrusted_drop_level = None
        stable_three_colour_sample = (
            level is None
            and stable_level == 3
            and observation.hud_visible
            and observation.spent_structure_verified
            and normalized_meter is not None
            and normalized_meter >= _MIN_THREE_LEVEL_COLOUR_FILL
        )
        self._stable_three_colour_count = (
            self._stable_three_colour_count + 1
            if stable_three_colour_sample else 0
        )
        colour_supports_stable_three = self._stable_three_colour_count >= 3
        colour_seeds_unreadable_three = (
            stable_level is None
            and self._unreadable_full_baseline_count
            >= _UNREADABLE_FULL_BASELINE_SAMPLES
        )
        tracked_level = (
            (3 if colour_seeds_unreadable_three else stable_level)
            if (
                preserve_stable_baseline
                or preserve_unapproved_drop
                or preserve_first_untrusted_drop
                or colour_supports_stable_three
                or colour_seeds_unreadable_three
            )
            else level
        )
        visible = (
            observation.hud_visible
            and tracked_level is not None
        )
        reading = MeterLevelReading(
            side=self._side,
            level=tracked_level if visible else None,
            confidence=(observation.meter_level_confidence if visible else 0.0),
            visible=visible,
        )
        transition = self._level_tracker.update(
            reading,
            observation.frame_id,
            baseline_trusted=(gain_is_approved or colour_seeds_unreadable_three),
            baseline_observed=(
                level == tracked_level
                or colour_supports_stable_three
                or colour_seeds_unreadable_three
            ),
        )
        if colour_seeds_unreadable_three:
            self._three_level_colour_available = True
            self._three_level_baseline_supported = True
        if (
            stable_level is not None
            and self._level_tracker.stable_level is None
        ):
            self._partial_residual_conflict_baseline = None
            self._clear_candidate_conflict_streak()
        if normalized_meter is not None:
            self._last_trusted_meter_fill = normalized_meter
            self._last_trusted_meter_fill_at = observation.sampled_at
        self._state = (
            TrackerState.ARMED
            if self._level_tracker.stable_level is not None
            else TrackerState.DISARMED
        )
        if transition is None:
            return None
        return transition

    def _update_partial_residual_conflict(
        self,
        observation: GaugeObservation,
    ) -> None:
        level = observation.meter_level
        normalized_meter = observation.normalized_meter
        conflict_baseline = self._partial_residual_conflict_baseline
        if conflict_baseline is not None:
            if level is None:
                return
            if level >= conflict_baseline:
                self._partial_residual_conflict_baseline = None
                return
            if level == 0:
                return
            if level not in (1, 2) or normalized_meter is None:
                return
            if normalized_meter <= _MAX_PARTIAL_CANDIDATE_COLOUR_FILL:
                self._partial_residual_conflict_baseline = None
            return

        stable_level = self._level_tracker.stable_level
        if (
            level in (1, 2)
            and stable_level is not None
            and level < stable_level
            and normalized_meter is not None
            and normalized_meter > _MAX_PARTIAL_CANDIDATE_COLOUR_FILL
        ):
            self._partial_residual_conflict_baseline = stable_level

    def _process_candidate(
        self,
        observation: GaugeObservation,
        classification: Classification,
    ) -> GaugeTrackerUpdate:
        if classification == "full":
            self._candidate_ambiguous = 0
            self._candidate_visual_disruption = False
            self._pre_spend_frame_id = observation.frame_id
            self._state = TrackerState.ARMED
            return GaugeTrackerUpdate()

        if not observation.spent_structure_verified:
            self._candidate_visual_disruption = True

        if classification == "ambiguous":
            self._candidate_ambiguous += 1
            if (
                self._candidate_ambiguous
                >= self._config.ambiguous_disarm_samples
            ):
                self._clear_detection_history()
                self._state = TrackerState.DISARMED
            return GaugeTrackerUpdate()

        self._candidate_ambiguous = 0
        if not self._confirmation_ready(observation):
            return GaugeTrackerUpdate()
        if self._color_meter_mode and not self._candidate_visual_disruption:
            # In live matches each earned level wraps the fractional colour
            # fill from full to empty.  A real Super also disrupts the meter's
            # calibrated structure during its startup freeze; a clean wrap is
            # therefore ordinary meter gain, not a spend.
            self._clear_detection_history()
            self._state = TrackerState.DISARMED
            return GaugeTrackerUpdate()

        self._state = TrackerState.CONFIRMED
        event = ConfirmedSuperEvent(
            event_id=uuid.uuid4().hex,
            side=self._side,
            confirmed_at=observation.sampled_at,
            pre_spend_frame_id=self._pre_spend_frame_id,
            confirmed_frame_id=observation.frame_id,
            profile_id=self._profile_id,
        )
        self._state = TrackerState.COOLDOWN
        return GaugeTrackerUpdate(event)

    def _confirmation_ready(self, current: GaugeObservation) -> bool:
        if len(self._evidence) != self._config.evidence_window:
            return False
        spent_observations = [
            observation
            for observation, classification in self._evidence
            if classification == "spent"
        ]
        if len(spent_observations) < self._config.spent_samples:
            return False
        earliest_spent = spent_observations[0]
        return (
            current.sampled_at - earliest_spent.sampled_at
            >= self._config.minimum_spend_seconds
        )

    def _clear_detection_history(self) -> None:
        self._fractional_drop_approved = False
        self._fractional_history.clear()
        self._fractional_zero_anchor = None
        self._fractional_zero_count = 0
        self._fractional_zero_last_at = None
        self._fractional_zero_ready = False
        self._delayed_three_zero_times.clear()
        self._delayed_three_zero_bridge = False
        self._evidence.clear()
        self._consecutive_full = 0
        self._candidate_ambiguous = 0
        self._candidate_visual_disruption = False
        self._color_meter_mode = False
        self._rearm_streak = 0
        self._pre_spend_frame_id = None
        self._untrusted_gain_level = None
        self._untrusted_gain_started_at = None
        self._untrusted_drop_level = None
        self._approved_drop_level = None
        self._clean_drop_level = None
        self._clean_drop_started_at = None
        self._approved_gain_level = None
        self._last_trusted_meter_fill = None
        self._last_trusted_meter_fill_at = None
        self._partial_residual_conflict_baseline = None
        self._stable_three_colour_count = 0
        self._three_level_colour_available = False
        self._three_level_baseline_support_count = 0
        self._three_level_baseline_supported = False
        self._unreadable_full_baseline_count = 0
        self._retired_candidate = None
        self._retired_recovery_count = 0
        self._clear_candidate_streak()
        self._clear_candidate_conflict_streak()
