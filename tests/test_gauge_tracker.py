from __future__ import annotations

import copy
from collections import deque
from dataclasses import FrozenInstanceError
from fractions import Fraction
from numbers import Rational, Real
from types import SimpleNamespace

import numpy as np
import pytest

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
import sf6_music_trigger.gauge_tracker as gauge_tracker_module
from sf6_music_trigger.gauge_tracker import (
    GaugeTracker,
    GaugeTrackerConfig,
    _classify_confidences,
)


class IntSubclass(int):
    pass


class StrSubclass(str):
    pass


class FloatNarrowingTrap(Fraction):
    def __float__(self):
        raise AssertionError("Rational validation must not narrow through float")


def obs(
    frame_id,
    t,
    *,
    side=Side.P1,
    full=None,
    spent=None,
    visible=True,
    structure=True,
    meter=None,
    meter_level_supported=False,
    meter_level=None,
    meter_level_confidence=0.0,
):
    if not visible:
        return GaugeObservation(
            side=side,
            frame_id=frame_id,
            sampled_at=t,
            hud_visible=False,
            full_confidence=0.0,
            spent_confidence=0.0,
            full_distance=1.0,
            spent_distance=1.0,
        )
    if full is None and spent is None:
        full = spent = 0.5
    elif full is None:
        full = 1.0 - spent
    elif spent is None:
        spent = 1.0 - full
    return GaugeObservation(
        side=side,
        frame_id=frame_id,
        sampled_at=t,
        hud_visible=True,
        full_confidence=full,
        spent_confidence=spent,
        full_distance=spent,
        spent_distance=full,
        spent_structure_verified=structure,
        normalized_meter=meter,
        meter_level_supported=meter_level_supported,
        meter_level=meter_level,
        meter_level_confidence=meter_level_confidence,
    )


def level_obs(
    frame_id: int,
    level: int | None,
    *,
    side: Side = Side.P1,
    disrupted: bool = False,
) -> GaugeObservation:
    return obs(
        frame_id,
        frame_id / 60,
        side=side,
        full=0.0 if disrupted else 1.0,
        spent=1.0 if disrupted else 0.0,
        meter=None if disrupted or level is None else level / 3,
        structure=not disrupted,
        meter_level_supported=True,
        meter_level=level,
        meter_level_confidence=0.9 if level is not None else 0.0,
    )


def transition(update: GaugeTrackerUpdate):
    assert isinstance(update, GaugeTrackerUpdate)
    return update.transition


def confirmed_events(update: GaugeTrackerUpdate) -> tuple[ConfirmedSuperEvent, ...]:
    item = transition(update)
    return (item,) if isinstance(item, ConfirmedSuperEvent) else ()


def feed_updates(
    tracker: GaugeTracker,
    observations: list[GaugeObservation],
):
    return tuple(
        item
        for observation in observations
        if (item := transition(tracker.update(observation))) is not None
    )


def feed_levels(
    tracker: GaugeTracker,
    levels: list[int | None],
    *,
    first_frame: int = 0,
    side: Side = Side.P1,
    disrupted: bool = False,
) -> tuple[ConfirmedSuperEvent, ...]:
    updates = feed_updates(
        tracker,
        [
            level_obs(
                first_frame + offset,
                level,
                side=side,
                disrupted=disrupted,
            )
            for offset, level in enumerate(levels)
        ],
    )
    return tuple(
        item for item in updates if isinstance(item, ConfirmedSuperEvent)
    )


def _level_at(frame_id: int, sampled_at: float, level: int) -> GaugeObservation:
    return obs(
        frame_id,
        sampled_at,
        full=1.0,
        spent=0.0,
        meter=level / 3,
        meter_level_supported=True,
        meter_level=level,
        meter_level_confidence=0.9,
    )


def _unknown_level_at(frame_id: int, sampled_at: float) -> GaugeObservation:
    return obs(
        frame_id,
        sampled_at,
        full=0.5,
        spent=0.5,
        meter=None,
        meter_level_supported=True,
    )


def _tracker_with_candidate(
    *, before: int, after: int
) -> tuple[GaugeTracker, SuperCandidateEvent]:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, 0.0, before))) is None
    for frame_id, sampled_at in ((10, 0.05), (11, 0.10), (12, 0.15)):
        update = transition(tracker.update(obs(
            frame_id,
            sampled_at,
            full=0.0,
            spent=1.0,
            meter=None if after in (1, 2) else after / 3,
            meter_level_supported=True,
            meter_level=after,
            meter_level_confidence=0.9,
        )))
    assert isinstance(update, SuperCandidateEvent)
    return tracker, update


def _candidate_conflict_at(
    frame_id: int, sampled_at: float, *, level: int = 2
) -> GaugeObservation:
    return obs(
        frame_id,
        sampled_at,
        spent=1.0,
        meter=None,
        meter_level_supported=True,
        meter_level=level,
        meter_level_confidence=0.9,
    )


def _emit_candidate_on_existing_tracker(
    tracker: GaugeTracker,
    *,
    first_frame: int,
    first_sampled_at: float,
    after: int = 1,
) -> SuperCandidateEvent:
    update: object = None
    for offset in range(3):
        update = transition(tracker.update(obs(
            first_frame + offset,
            first_sampled_at + offset * 0.05,
            spent=1.0,
            meter=None,
            meter_level_supported=True,
            meter_level=after,
            meter_level_confidence=0.9,
        )))
    assert isinstance(update, SuperCandidateEvent)
    return update


def _feed_clean_level_until(
    tracker: GaugeTracker,
    *,
    level: int,
    through: float,
) -> tuple[object, ...]:
    return feed_updates(
        tracker,
        [
            _level_at(frame_id, frame_id * 0.05, level)
            for frame_id in range(13, round(through / 0.05) + 1)
        ],
    )


def test_three_lower_samples_emit_candidate_without_mutating_baseline() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(level_obs(frame_id, 3))) is None

    updates = [
        tracker.update(obs(
            10 + offset, 0.50 + offset * 0.05,
            full=0.0, spent=1.0, meter=0.0,
            meter_level_supported=True, meter_level=0,
            meter_level_confidence=0.9,
        ))
        for offset in range(3)
    ]

    candidate = transition(updates[-1])
    assert isinstance(candidate, SuperCandidateEvent)
    assert (candidate.meter_before, candidate.meter_after) == (3, 0)
    assert tracker.stable_meter_level == 3


def test_full_colour_contradiction_cannot_emit_lower_level_candidate() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, frame_id * 0.05, 3))) is None

    transitions = feed_updates(
        tracker,
        [
            obs(
                frame_id,
                frame_id * 0.05,
                full=0.95,
                spent=0.05,
                meter=0.95,
                meter_level_supported=True,
                meter_level=1,
                meter_level_confidence=0.9,
            )
            for frame_id in range(10, 13)
        ],
    )

    assert transitions == ()
    assert tracker.stable_meter_level == 3


def _partial_level_at(
    frame_id: int,
    sampled_at: float,
    level: int,
    *,
    fill: float | None,
    side: Side = Side.P1,
    structure: bool = True,
) -> GaugeObservation:
    return obs(
        frame_id,
        sampled_at,
        side=side,
        spent=0.90,
        structure=structure,
        meter=fill,
        meter_level_supported=True,
        meter_level=level,
        meter_level_confidence=0.9,
    )


def test_ambiguous_colour_cannot_emit_plausible_lower_level_candidate() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, frame_id * 0.05, 3))) is None

    transitions = feed_updates(
        tracker,
        [
            obs(
                frame_id,
                frame_id * 0.05,
                full=0.75,
                spent=0.25,
                meter=0.75,
                meter_level_supported=True,
                meter_level=2,
                meter_level_confidence=0.9,
            )
            for frame_id in range(10, 13)
        ],
    )

    assert transitions == ()
    assert tracker.stable_meter_level == 3


def test_partial_candidate_rejects_observed_residual_fractional_fill() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, frame_id * 0.05, 3))) is None

    transitions = feed_updates(
        tracker,
        [
            obs(
                10,
                0.50,
                spent=0.87,
                meter=0.131,
                meter_level_supported=True,
                meter_level=1,
                meter_level_confidence=0.9,
            ),
            obs(
                11,
                0.55,
                spent=0.86,
                meter=0.137,
                meter_level_supported=True,
                meter_level=1,
                meter_level_confidence=0.9,
            ),
            obs(
                12,
                0.60,
                spent=0.86,
                structure=False,
                meter_level_supported=True,
                meter_level=1,
                meter_level_confidence=0.9,
            ),
            *[
                _partial_level_at(
                    frame_id,
                    0.50 + (frame_id - 10) * 0.05,
                    1,
                    fill=None,
                    structure=False,
                )
                for frame_id in range(13, 28)
            ],
        ],
    )

    assert transitions == ()
    assert tracker.stable_meter_level == 3


@pytest.mark.parametrize(
    ("fills", "structures"),
    [
        pytest.param((None, None, None), (True, True, True), id="all-unreadable"),
        pytest.param((0.0, None, 0.0), (True, False, True), id="observed-empty"),
    ],
)
def test_partial_candidate_allows_unreadable_or_empty_fractional_fill(
    fills: tuple[float | None, ...],
    structures: tuple[bool, ...],
) -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, frame_id * 0.05, 3))) is None

    transitions = feed_updates(
        tracker,
        [
            *[
                _partial_level_at(
                    10 + offset,
                    0.50 + offset * 0.05,
                    1,
                    fill=fills[offset],
                    structure=structures[offset],
                )
                for offset in range(3)
            ],
            *[
                _partial_level_at(
                    frame_id,
                    0.50 + (frame_id - 10) * 0.05,
                    1,
                    fill=0.0,
                )
                for frame_id in range(13, 26)
            ],
        ],
    )

    assert [type(item) for item in transitions] == [
        SuperCandidateEvent,
        ConfirmedSuperEvent,
    ]


def test_partial_residual_episode_blocks_staged_partial_after_transient_zero() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, frame_id * 0.05, 3))) is None

    observations = [
        _partial_level_at(10, 0.50, 1, fill=0.131),
        _partial_level_at(11, 0.55, 0, fill=None, structure=False),
        obs(
            12,
            0.60,
            spent=0.90,
            structure=False,
            meter_level_supported=True,
        ),
        *[
            _partial_level_at(
                frame_id,
                0.65 + (frame_id - 13) * 0.05,
                2,
                fill=None,
                structure=False,
            )
            for frame_id in range(13, 17)
        ],
    ]

    assert feed_updates(tracker, observations) == ()
    assert tracker.stable_meter_level == 3

    zero_updates = feed_updates(
        tracker,
        [
            _partial_level_at(
                frame_id,
                0.85 + (frame_id - 17) * 0.05,
                0,
                fill=None,
                structure=False,
            )
            for frame_id in range(17, 21)
        ],
    )
    assert [type(item) for item in zero_updates] == [
        SuperCandidateEvent,
        ConfirmedSuperEvent,
    ]
    candidate, event = zero_updates
    assert (candidate.meter_before, candidate.meter_after) == (3, 0)
    assert (event.meter_before, event.meter_after) == (3, 0)
    assert event.candidate_id == candidate.candidate_id
    assert tracker.stable_meter_level == 0


def test_trusted_low_fill_zero_candidate_bypasses_partial_residual_episode() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, frame_id * 0.05, 3))) is None

    assert transition(
        tracker.update(_partial_level_at(10, 0.50, 1, fill=0.131))
    ) is None
    updates = feed_updates(
        tracker,
        [
            _partial_level_at(
                11 + offset,
                0.55 + offset * 0.05,
                0,
                fill=0.0904,
                structure=True,
            )
            for offset in range(3)
        ],
    )

    assert [type(item) for item in updates] == [SuperCandidateEvent]
    candidate = updates[0]
    assert (candidate.meter_before, candidate.meter_after) == (3, 0)
    assert tracker.stable_meter_level == 3


def test_trusted_empty_fill_clears_partial_residual_blocker() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, frame_id * 0.05, 3))) is None

    transitions = feed_updates(
        tracker,
        [
            _partial_level_at(10, 0.50, 1, fill=0.131),
            *[
                _partial_level_at(
                    frame_id,
                    0.50 + (frame_id - 10) * 0.05,
                    1,
                    fill=0.0,
                )
                for frame_id in range(11, 29)
            ],
        ],
    )

    assert [type(item) for item in transitions] == [
        SuperCandidateEvent,
        ConfirmedSuperEvent,
    ]


@pytest.mark.parametrize(
    ("first_level", "second_level"),
    [
        pytest.param(1, 2, id="one-to-two"),
        pytest.param(2, 1, id="two-to-one"),
    ],
)
def test_partial_level_oscillation_does_not_clear_residual_blocker(
    first_level: int,
    second_level: int,
) -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, frame_id * 0.05, 3))) is None

    transitions = feed_updates(
        tracker,
        [
            _partial_level_at(10, 0.50, first_level, fill=0.131),
            _partial_level_at(11, 0.55, second_level, fill=None),
            _partial_level_at(12, 0.60, first_level, fill=None),
            *[
                _partial_level_at(
                    frame_id,
                    0.50 + (frame_id - 10) * 0.05,
                    second_level,
                    fill=None,
                )
                for frame_id in range(13, 31)
            ],
        ],
    )

    assert transitions == ()


@pytest.mark.parametrize("baseline", [2, 3])
def test_authoritative_baseline_recovery_clears_partial_residual_blocker(
    baseline: int,
) -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(
            tracker.update(_level_at(frame_id, frame_id * 0.05, baseline))
        ) is None

    assert transition(
        tracker.update(_partial_level_at(10, 0.50, 1, fill=0.131))
    ) is None
    assert transition(tracker.update(_level_at(11, 0.55, baseline))) is None
    transitions = feed_updates(
        tracker,
        [
            _partial_level_at(
                frame_id,
                0.60 + (frame_id - 12) * 0.05,
                1,
                fill=None,
            )
            for frame_id in range(12, 30)
        ],
    )

    assert [type(item) for item in transitions] == [
        SuperCandidateEvent,
        ConfirmedSuperEvent,
    ]


@pytest.mark.parametrize("side", [Side.P1, Side.P2])
def test_unknown_reset_clears_partial_residual_blocker_before_new_baseline(
    side: Side,
) -> None:
    tracker = GaugeTracker(side, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_partial_level_at(
            frame_id,
            frame_id * 0.05,
            3,
            fill=1.0,
            side=side,
        ))) is None

    assert transition(tracker.update(
        _partial_level_at(10, 0.50, 1, fill=0.131, side=side)
    )) is None
    for frame_id in range(11, 21):
        assert transition(tracker.update(obs(
            frame_id,
            frame_id * 0.05,
            side=side,
            meter_level_supported=True,
        ))) is None
    assert tracker.stable_meter_level is None

    for frame_id in range(21, 31):
        assert transition(tracker.update(_partial_level_at(
            frame_id,
            frame_id * 0.05,
            2,
            fill=None,
            side=side,
            structure=False,
        ))) is None
    assert tracker.stable_meter_level == 2

    transitions = feed_updates(
        tracker,
        [
            _partial_level_at(
                frame_id,
                frame_id * 0.05,
                1,
                fill=None,
                side=side,
                structure=False,
            )
            for frame_id in range(31, 35)
        ],
    )
    assert [type(item) for item in transitions] == [
        SuperCandidateEvent,
        ConfirmedSuperEvent,
    ]
    candidate, event = transitions
    assert (candidate.side, candidate.meter_before, candidate.meter_after) == (
        side,
        2,
        1,
    )
    assert event.candidate_id == candidate.candidate_id


def test_reset_clears_partial_residual_blocker() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, frame_id * 0.05, 3))) is None
    assert transition(
        tracker.update(_partial_level_at(10, 0.50, 1, fill=0.131))
    ) is None

    assert tracker.reset("new replay session") == GaugeTrackerUpdate()
    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, 1.0 + frame_id * 0.05, 3))) is None
    transitions = feed_updates(
        tracker,
        [
            _partial_level_at(
                10 + offset,
                1.50 + offset * 0.05,
                1,
                fill=0.0,
            )
            for offset in range(16)
        ],
    )

    assert [type(item) for item in transitions] == [
        SuperCandidateEvent,
        ConfirmedSuperEvent,
    ]


def test_partial_residual_blockers_are_side_independent() -> None:
    trackers = {
        Side.P1: GaugeTracker(Side.P1, "profile", GaugeTrackerConfig()),
        Side.P2: GaugeTracker(Side.P2, "profile", GaugeTrackerConfig()),
    }
    for side, tracker in trackers.items():
        for frame_id in range(10):
            observation = _level_at(frame_id, frame_id * 0.05, 3)
            if side is Side.P2:
                observation = _partial_level_at(
                    frame_id,
                    frame_id * 0.05,
                    3,
                    fill=1.0,
                    side=Side.P2,
                )
            assert transition(tracker.update(observation)) is None

    assert transition(
        trackers[Side.P1].update(_partial_level_at(10, 0.50, 1, fill=0.131))
    ) is None
    p1_updates = feed_updates(
        trackers[Side.P1],
        [
            _partial_level_at(frame_id, frame_id * 0.05, 1, fill=None)
            for frame_id in range(11, 16)
        ],
    )
    p2_updates = feed_updates(
        trackers[Side.P2],
        [
            _partial_level_at(
                frame_id,
                frame_id * 0.05,
                1,
                fill=0.0,
                side=Side.P2,
            )
            for frame_id in range(10, 16)
        ],
    )

    assert p1_updates == ()
    assert isinstance(p2_updates[0], SuperCandidateEvent)


def test_candidate_confirms_once_without_second_candidate() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    transitions = _feed_clean_level_until(tracker, level=0, through=1.20)
    assert [type(item) for item in transitions] == [ConfirmedSuperEvent]
    assert transitions[0].candidate_id == candidate.candidate_id
    assert tracker.stable_meter_level == 0


def test_sustained_level_recovery_cancels_candidate() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    assert transition(tracker.update(_level_at(20, 0.70, 3))) is None
    assert transition(tracker.update(_level_at(21, 0.75, 3))) is None
    cancelled = transition(tracker.update(_level_at(22, 0.80, 3)))
    assert isinstance(cancelled, CancelledSuperCandidate)
    assert cancelled.candidate_id == candidate.candidate_id
    assert cancelled.reason is CandidateCancelReason.LEVEL_RECOVERED
    assert tracker.stable_meter_level == 3


@pytest.mark.parametrize("side", [Side.P1, Side.P2])
def test_active_candidate_requires_three_fresh_conflicts_and_clears_streaks(
    side: Side,
) -> None:
    tracker = GaugeTracker(side, "profile", GaugeTrackerConfig())
    for frame_id in range(10):
        assert transition(tracker.update(obs(
            frame_id,
            0.0,
            side=side,
            full=1.0,
            meter=1.0,
            meter_level_supported=True,
            meter_level=3,
            meter_level_confidence=0.9,
        ))) is None
    candidate: object = None
    for frame_id, sampled_at in ((10, 0.05), (11, 0.10), (12, 0.15)):
        candidate = transition(tracker.update(obs(
            frame_id,
            sampled_at,
            side=side,
            spent=1.0,
            meter=None,
            meter_level_supported=True,
            meter_level=1,
            meter_level_confidence=0.9,
        )))
    assert isinstance(candidate, SuperCandidateEvent)

    def level(frame_id: int, sampled_at: float, value: int) -> GaugeObservation:
        return obs(
            frame_id,
            sampled_at,
            side=side,
            spent=1.0,
            meter=None,
            meter_level_supported=True,
            meter_level=value,
            meter_level_confidence=0.9,
        )

    assert transition(tracker.update(level(13, 0.20, 2))) is None
    assert transition(tracker.update(level(14, 0.25, 2))) is None
    assert transition(tracker.update(obs(
        15,
        0.30,
        side=side,
        full=0.5,
        spent=0.5,
        meter=None,
        meter_level_supported=True,
    ))) is None
    assert transition(tracker.update(level(16, 0.35, 2))) is None
    assert transition(tracker.update(level(17, 0.40, 2))) is None
    assert transition(tracker.update(level(18, 0.45, 1))) is None
    assert transition(tracker.update(level(19, 0.50, 2))) is None
    assert transition(tracker.update(level(20, 0.55, 2))) is None
    cancelled = transition(tracker.update(level(21, 0.60, 2)))

    assert isinstance(cancelled, CancelledSuperCandidate)
    assert cancelled.side is side
    assert cancelled.candidate_id == candidate.candidate_id
    assert cancelled.reason is CandidateCancelReason.LEVEL_CONFLICT


def test_candidate_conflict_streak_does_not_survive_tracker_reset() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=1)
    for frame_id in (13, 14):
        assert transition(tracker.update(
            _candidate_conflict_at(frame_id, 0.20 + (frame_id - 13) * 0.05)
        )) is None

    reset = transition(tracker.reset("new capture session", decided_at=0.75))
    assert isinstance(reset, CancelledSuperCandidate)
    assert reset.candidate_id == candidate.candidate_id
    assert reset.reason is CandidateCancelReason.CAPTURE_RESET

    for frame_id in range(10):
        assert transition(tracker.update(_level_at(frame_id, 1.0, 3))) is None
    _emit_candidate_on_existing_tracker(
        tracker, first_frame=10, first_sampled_at=1.05
    )
    assert transition(tracker.update(_candidate_conflict_at(
        13,
        1.20,
    ))) is None


def test_candidate_conflict_streak_does_not_survive_confirmation() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=1)
    for frame_id in (13, 14):
        assert transition(tracker.update(
            _candidate_conflict_at(frame_id, 0.20 + (frame_id - 13) * 0.05)
        )) is None

    confirmed: ConfirmedSuperEvent | None = None
    candidate_transition = tracker._candidate_transition
    tracker._candidate_transition = (  # type: ignore[method-assign]
        lambda _observation: GaugeTrackerUpdate()
    )
    try:
        for frame_id in range(15, 31):
            item = transition(tracker.update(_candidate_conflict_at(
                frame_id, 0.30 + (frame_id - 15) * 0.05, level=1
            )))
            if isinstance(item, ConfirmedSuperEvent):
                confirmed = item
    finally:
        tracker._candidate_transition = candidate_transition  # type: ignore[method-assign]
    assert confirmed is not None
    assert confirmed.candidate_id == candidate.candidate_id

    for frame_id in range(31, 41):
        assert transition(tracker.update(_level_at(
            frame_id, 1.10 + (frame_id - 31) * 0.05, 3
        ))) is None
    _emit_candidate_on_existing_tracker(
        tracker, first_frame=41, first_sampled_at=1.60
    )
    assert transition(tracker.update(
        _candidate_conflict_at(44, 1.75)
    )) is None


def test_candidate_conflict_streak_does_not_survive_authority_loss() -> None:
    tracker, _ = _tracker_with_candidate(before=3, after=1)
    for frame_id in (13, 14):
        assert transition(tracker.update(
            _candidate_conflict_at(frame_id, 0.20 + (frame_id - 13) * 0.05)
        )) is None

    candidate_transition = tracker._candidate_transition
    tracker._candidate_transition = (  # type: ignore[method-assign]
        lambda _observation: GaugeTrackerUpdate()
    )
    try:
        for frame_id in range(15, 25):
            assert transition(tracker.update(_unknown_level_at(
                frame_id, 0.30 + (frame_id - 15) * 0.03
            ))) is None
    finally:
        tracker._candidate_transition = candidate_transition  # type: ignore[method-assign]
    assert tracker.stable_meter_level is None
    assert transition(tracker.update(
        _candidate_conflict_at(25, 0.61)
    )) is None


def test_candidate_conflict_streak_does_not_survive_non_reset_cancel() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=1)
    for frame_id in (13, 14):
        assert transition(tracker.update(
            _candidate_conflict_at(frame_id, 0.20 + (frame_id - 13) * 0.05)
        )) is None

    assert transition(tracker.update(_level_at(15, 0.30, 3))) is None
    assert transition(tracker.update(_level_at(16, 0.35, 3))) is None
    cancelled = transition(tracker.update(_level_at(17, 0.40, 3)))
    assert isinstance(cancelled, CancelledSuperCandidate)
    assert cancelled.candidate_id == candidate.candidate_id
    assert cancelled.reason is CandidateCancelReason.LEVEL_RECOVERED

    _emit_candidate_on_existing_tracker(
        tracker, first_frame=18, first_sampled_at=0.45
    )
    assert transition(tracker.update(
        _candidate_conflict_at(21, 0.60)
    )) is None


def test_unconfirmed_candidate_cancels_at_fixed_deadline() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    before = transition(tracker.update(_unknown_level_at(20, 0.89)))
    at_deadline = transition(tracker.update(_unknown_level_at(21, 0.90)))
    assert before is None
    assert isinstance(at_deadline, CancelledSuperCandidate)
    assert at_deadline.reason is CandidateCancelReason.CONFIRMATION_TIMEOUT


def test_timeout_consumes_candidate_episode_until_meter_recovers() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    expired = transition(tracker.update(_unknown_level_at(13, 0.90)))
    assert isinstance(expired, CancelledSuperCandidate)
    updates = feed_updates(tracker, [
        _candidate_conflict_at(14 + i, 0.95 + i * 0.05, level=0)
        for i in range(24)
    ])
    # A late confirmation can consume the meter baseline, but must retain the
    # terminal candidate ID so the audio router cannot start it again.
    assert [type(item) for item in updates] == [ConfirmedSuperEvent]
    assert updates[0].candidate_id == candidate.candidate_id


def test_timeout_lock_does_not_block_a_new_spend_after_real_recovery() -> None:
    tracker, original = _tracker_with_candidate(before=3, after=0)
    tracker.update(_unknown_level_at(13, 0.90))
    for i in range(3):
        assert transition(tracker.update(_level_at(14+i, 0.95+i*0.05, 3))) is None
    candidate = _emit_candidate_on_existing_tracker(
        tracker, first_frame=17, first_sampled_at=1.10, after=0,
    )
    assert candidate.candidate_id != original.candidate_id


def test_matching_lower_level_extends_soft_deadline_but_not_forever() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    updates = []
    # An unknown every other frame prevents final confirmation; positive lower
    # samples keep the same candidate alive briefly, not new replay candidates.
    for i in range(1, 67):
        t = round(0.15+i*0.05, 6)
        observation = (_candidate_conflict_at(12+i, t, level=0)
                       if i % 2 == 0 else _unknown_level_at(12+i, t))
        item = transition(tracker.update(observation))
        if item is not None:
            updates.append(item)
    assert len(updates) == 1
    assert isinstance(updates[0], CancelledSuperCandidate)
    assert updates[0].candidate_id == candidate.candidate_id
    assert updates[0].reason is CandidateCancelReason.CONFIRMATION_TIMEOUT
    assert 3.0 <= updates[0].decided_at - candidate.suspected_at <= 3.05


def test_authority_is_not_lost_when_candidate_is_ready_on_same_frame() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for i in range(10):
        tracker.update(_level_at(i, 0.0, 3))
    updates = []
    for i in range(15):
        t = Fraction(10+i, 20)
        observation = (_candidate_conflict_at(10+i, t, level=0)
                       if i >= 12 else _level_at(10+i, t, 0))
        item = transition(tracker.update(observation))
        if item is not None:
            updates.append(item)
    assert [type(item) for item in updates] == [ConfirmedSuperEvent]
    assert (updates[0].meter_before, updates[0].meter_after) == (3, 0)


def test_disabling_level_support_cancels_candidate_and_restarts_color_rearm() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)

    cancelled = transition(tracker.update(obs(13, 0.20, full=0.78)))

    assert isinstance(cancelled, CancelledSuperCandidate)
    assert cancelled.candidate_id == candidate.candidate_id
    assert cancelled.reason is CandidateCancelReason.CAPTURE_RESET
    assert tracker.state is TrackerState.DISARMED

    for frame_id, sampled_at in (
        (14, 0.25),
        (15, 0.30),
        (16, 0.35),
        (17, 0.40),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
        assert tracker.state is TrackerState.DISARMED

    assert transition(tracker.update(obs(18, 0.45, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(19, 0.50, spent=0.72))) is None
    assert transition(tracker.update(obs(20, 0.56, spent=0.72))) is None
    confirmed = transition(tracker.update(obs(21, 0.62, spent=0.72)))
    assert isinstance(confirmed, ConfirmedSuperEvent)
    assert confirmed.pre_spend_frame_id == 18


def test_reset_returns_one_cancellation_before_clearing_candidate() -> None:
    tracker, candidate = _tracker_with_candidate(before=3, after=0)
    update = tracker.reset("capture session changed", decided_at=0.5)
    assert transition(update).candidate_id == candidate.candidate_id
    assert transition(update).reason is CandidateCancelReason.CAPTURE_RESET
    assert tracker.reset("already clean") == GaugeTrackerUpdate()


def arm_default(tracker: GaugeTracker, *, side: Side = Side.P1) -> None:
    for frame_id, sampled_at in (
        (0, 0.00),
        (1, 0.05),
        (2, 0.10),
        (3, 0.15),
        (4, 0.20),
    ):
        assert transition(tracker.update(
            obs(frame_id, sampled_at, side=side, full=0.78)
        )) is None
    assert tracker.state is TrackerState.ARMED


def candidate_tracker() -> GaugeTracker:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    return tracker


def disarmed_tracker() -> GaugeTracker:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert transition(tracker.update(obs(0, 0.00))) is None
    assert transition(tracker.update(obs(1, 0.05))) is None
    assert transition(tracker.update(obs(2, 0.10))) is None
    assert tracker.state is TrackerState.DISARMED
    return tracker


def armed_tracker() -> GaugeTracker:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    return tracker


def cooldown_tracker() -> GaugeTracker:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.36, spent=0.72))) is None
    confirmed = transition(tracker.update(obs(7, 0.42, spent=0.72)))
    assert isinstance(confirmed, ConfirmedSuperEvent)
    assert tracker.state is TrackerState.COOLDOWN
    return tracker


def test_numeric_level_gain_then_downward_spend_emits_one_metadata_event() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [1] * 10) == ()
    assert feed_levels(
        tracker,
        [2] * 3,
        first_frame=10,
        disrupted=True,
    ) == ()
    assert feed_levels(tracker, [2] * 10, first_frame=13) == ()

    events = feed_levels(
        tracker,
        [1, 1, 1, 1],
        first_frame=23,
        disrupted=True,
    )

    assert len(events) == 1
    event = events[0]
    assert (event.meter_before, event.meter_after, event.levels_spent) == (2, 1, 1)
    # Frame23 is already the disrupted lower reading; the last genuine
    # pre-spend baseline is frame22.
    assert (event.pre_spend_frame_id, event.confirmed_frame_id) == (22, 26)
    assert tracker.stable_meter_level == 1


def test_disrupted_upward_glitch_cannot_create_a_spend_baseline() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [2] * 10) == ()
    assert transition(tracker.update(
        obs(
            10,
            10 / 60,
            full=0.08,
            spent=0.92,
            meter=0.08,
            meter_level_supported=True,
            meter_level=2,
            meter_level_confidence=0.9,
        )
    )) is None

    assert feed_levels(
        tracker,
        [3] * 12,
        first_frame=11,
        disrupted=True,
    ) == ()
    events = feed_levels(
        tracker,
        [2] * 3,
        first_frame=23,
        disrupted=True,
    )

    assert events == ()
    assert tracker.stable_meter_level == 2


def test_three_sample_disrupted_downward_glitch_does_not_emit() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [2] * 10) == ()

    assert feed_levels(
        tracker,
        [1] * 3,
        first_frame=10,
        disrupted=True,
    ) == ()
    assert feed_levels(
        tracker,
        [2] * 3,
        first_frame=13,
        disrupted=True,
    ) == ()
    assert tracker.stable_meter_level == 2


def test_clean_downward_glyph_change_without_disruption_does_not_emit() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [3] * 10) == ()

    assert feed_levels(tracker, [1] * 6, first_frame=10) == ()

    assert tracker.stable_meter_level == 3


def test_sustained_clean_downward_level_eventually_emits() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [1] * 10) == ()
    events = []
    for frame_id, sampled_at in enumerate(
        (0.2, 0.4, 0.6, 0.8, 0.9, 1.0),
        start=10,
    ):
        events.extend(
            confirmed_events(tracker.update(
                obs(
                    frame_id,
                    sampled_at,
                    full=0.5,
                    spent=0.5,
                    meter=0.5,
                    meter_level_supported=True,
                    meter_level=0,
                    meter_level_confidence=0.9,
                )
            ))
        )

    assert [(event.meter_before, event.meter_after) for event in events] == [(1, 0)]


def test_clean_downward_level_with_conflicting_colour_never_emits() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [3] * 10) == ()

    events = tuple(
        event
        for frame_id in range(10, 30)
        for event in confirmed_events(tracker.update(
            obs(
                frame_id,
                frame_id / 10,
                meter=0.95,
                meter_level_supported=True,
                meter_level=1,
                meter_level_confidence=0.9,
            )
        ))
    )

    assert events == ()
    assert tracker.stable_meter_level == 3


def test_p1_three_to_zero_can_bridge_delayed_structure_disruption() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [3] * 15) == ()

    observations = [
        obs(
            15,
            0.50,
            full=0.84,
            spent=0.16,
            meter=0.84,
            meter_level_supported=True,
            meter_level=0,
            meter_level_confidence=0.9,
        ),
        obs(
            16,
            0.60,
            full=0.84,
            spent=0.16,
            meter=0.84,
            meter_level_supported=True,
            meter_level=0,
            meter_level_confidence=0.9,
        ),
        obs(
            17,
            0.70,
            full=0.69,
            spent=0.31,
            meter=0.69,
            meter_level_supported=True,
            meter_level=3,
            meter_level_confidence=0.92,
        ),
        obs(
            18,
            0.80,
            full=0.80,
            spent=0.20,
            meter=0.80,
            meter_level_supported=True,
            meter_level=1,
            meter_level_confidence=0.9,
        ),
        *[
            obs(
                frame_id,
                sampled_at,
                full=0.05,
                spent=0.95,
                meter=None,
                structure=False,
                meter_level_supported=True,
                meter_level=None,
                meter_level_confidence=0.0,
            )
            for frame_id, sampled_at in (
                (19, 0.95),
                (20, 1.05),
                (21, 1.15),
                (22, 1.20),
            )
        ],
    ]

    updates = feed_updates(tracker, observations)

    assert [type(item) for item in updates] == [
        SuperCandidateEvent,
        ConfirmedSuperEvent,
    ]
    candidate, confirmed = updates
    assert (confirmed.meter_before, confirmed.meter_after) == (3, 0)
    assert confirmed.candidate_id == candidate.candidate_id


def test_p1_three_to_zero_delayed_structure_bridge_expires() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [3] * 15) == ()

    observations = [
        obs(
            15,
            0.50,
            full=0.84,
            spent=0.16,
            meter=0.84,
            meter_level_supported=True,
            meter_level=0,
            meter_level_confidence=0.9,
        ),
        obs(
            16,
            0.60,
            full=0.84,
            spent=0.16,
            meter=0.84,
            meter_level_supported=True,
            meter_level=0,
            meter_level_confidence=0.9,
        ),
        *[
            obs(
                frame_id,
                sampled_at,
                full=0.05,
                spent=0.95,
                meter=None,
                structure=False,
                meter_level_supported=True,
                meter_level=None,
                meter_level_confidence=0.0,
            )
            for frame_id, sampled_at in (
                (17, 2.00),
                (18, 2.10),
                (19, 2.20),
                (20, 2.30),
            )
        ],
    ]

    assert feed_updates(tracker, observations) == ()
    assert tracker.stable_meter_level == 3


def test_p1_single_conflicting_zero_cannot_bridge_structure_disruption() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [3] * 15) == ()

    observations = [
        obs(
            15,
            0.50,
            full=0.84,
            spent=0.16,
            meter=0.84,
            meter_level_supported=True,
            meter_level=0,
            meter_level_confidence=0.9,
        ),
        *[
            obs(
                frame_id,
                sampled_at,
                full=0.05,
                spent=0.95,
                meter=None,
                structure=False,
                meter_level_supported=True,
                meter_level=None,
                meter_level_confidence=0.0,
            )
            for frame_id, sampled_at in (
                (16, 0.70),
                (17, 0.80),
                (18, 0.90),
                (19, 1.00),
            )
        ],
    ]

    assert feed_updates(tracker, observations) == ()
    assert tracker.stable_meter_level == 3


def test_clean_zero_level_with_moderate_lagging_colour_eventually_emits() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [1] * 10) == ()

    events = tuple(
        event
        for frame_id in range(10, 20)
        for event in confirmed_events(tracker.update(
            obs(
                frame_id,
                frame_id / 10,
                meter=0.55,
                meter_level_supported=True,
                meter_level=0,
                meter_level_confidence=0.9,
            )
        ))
    )

    assert [(event.meter_before, event.meter_after) for event in events] == [(1, 0)]


def test_disrupted_initial_level_can_still_establish_a_baseline() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())

    assert feed_levels(tracker, [3] * 10, disrupted=True) == ()

    assert tracker.stable_meter_level == 3


def test_sustained_disrupted_gain_eventually_becomes_the_new_baseline() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [1] * 10) == ()
    frame_id = 10
    for sampled_at in (0.2, 1.0, 2.0, 3.0, 3.8, 3.9, 4.0):
        observation = obs(
            frame_id,
            sampled_at,
            full=0.0,
            spent=1.0,
            meter=None,
            structure=False,
            meter_level_supported=True,
            meter_level=2,
            meter_level_confidence=0.9,
        )
        assert transition(tracker.update(observation)) is None
        frame_id += 1

    assert tracker.stable_meter_level == 2


def test_one_trusted_gain_sample_does_not_approve_following_disrupted_samples() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [1] * 10) == ()

    trusted_gain = obs(
        10,
        10 / 60,
        full=0.8,
        spent=0.2,
        meter=0.8,
        meter_level_supported=True,
        meter_level=3,
        meter_level_confidence=0.9,
    )
    assert transition(tracker.update(trusted_gain)) is None
    assert feed_levels(
        tracker,
        [3, 3],
        first_frame=11,
        disrupted=True,
    ) == ()

    assert tracker.stable_meter_level == 1


def test_two_trusted_gain_samples_approve_following_disrupted_sample() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [0] * 10) == ()

    for frame_id in (10, 11):
        assert transition(tracker.update(
            obs(
                frame_id,
                frame_id / 60,
                full=0.2,
                spent=0.8,
                meter=0.2,
                meter_level_supported=True,
                meter_level=1,
                meter_level_confidence=0.9,
            )
        )) is None
    assert feed_levels(
        tracker,
        [1, 1],
        first_frame=12,
        disrupted=True,
    ) == ()

    assert tracker.stable_meter_level == 1


def test_recent_full_fill_does_not_approve_a_disrupted_one_level_wrap() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [0] * 10) == ()
    assert transition(tracker.update(
        obs(
            10,
            10 / 60,
            full=0.7,
            spent=0.3,
            meter=0.7,
            meter_level_supported=True,
            meter_level=0,
            meter_level_confidence=0.9,
        )
    )) is None

    assert feed_levels(
        tracker,
        [1, 1, 1],
        first_frame=11,
        disrupted=True,
    ) == ()

    assert tracker.stable_meter_level == 0


def test_recent_full_fill_promotes_a_sustained_disrupted_one_level_wrap() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [0] * 10) == ()
    assert transition(tracker.update(
        obs(
            10,
            10 / 60,
            full=0.7,
            spent=0.3,
            meter=0.7,
            meter_level_supported=True,
            meter_level=0,
            meter_level_confidence=0.9,
        )
    )) is None

    assert feed_levels(
        tracker,
        [1, 1, 1, 1],
        first_frame=11,
        disrupted=True,
    ) == ()

    assert tracker.stable_meter_level == 1


def test_numeric_level_production_tracker_rejects_a_short_false_baseline() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())

    assert feed_levels(tracker, [3] * 3 + [0] * 3, disrupted=True) == ()

    tracker.reset("retry with durable baseline")
    assert feed_levels(tracker, [3] * 10) == ()
    events = feed_levels(tracker, [0] * 4, first_frame=10, disrupted=True)
    assert [(event.meter_before, event.meter_after) for event in events] == [(3, 0)]


def test_lagging_colour_fill_does_not_block_a_disrupted_downward_level() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [3] * 10) == ()

    assert transition(tracker.update(
        obs(
            10,
            10 / 60,
            full=0.9,
            spent=0.1,
            meter=0.9,
            meter_level_supported=True,
            meter_level=0,
            meter_level_confidence=0.9,
        )
    )) is None
    events = feed_levels(
        tracker,
        [0] * 4,
        first_frame=11,
        disrupted=True,
    )

    assert [(event.meter_before, event.meter_after) for event in events] == [(3, 0)]


def test_numeric_three_inconsistent_with_one_bar_colour_fill_is_unknown() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [1] * 10) == ()

    inconsistent = [
        obs(
            frame_id,
            frame_id / 60,
            full=0.33,
            spent=0.67,
            meter=0.33,
            meter_level_supported=True,
            meter_level=3,
            meter_level_confidence=0.9,
        )
        for frame_id in range(10, 22)
    ]

    assert tuple(
        event for observation in inconsistent for event in confirmed_events(tracker.update(observation))
    ) == ()
    events = feed_levels(
        tracker,
        [1] * 3,
        first_frame=22,
        disrupted=True,
    )
    assert events == ()
    assert tracker.stable_meter_level == 1


@pytest.mark.parametrize(("before", "after"), [(3, 0), (3, 1), (3, 2)])
def test_numeric_level_spends_emit_exact_tier(before: int, after: int) -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [before] * 10) == ()

    events = feed_levels(
        tracker,
        [after] * 4,
        first_frame=10,
        disrupted=True,
    )

    assert len(events) == 1
    assert events[0].levels_spent == before - after


def test_colour_fill_change_without_level_decrease_never_emits() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert feed_levels(tracker, [3, 3, 3]) == ()
    observations = [
        obs(
            frame_id,
            frame_id / 60,
            full=0.0,
            spent=1.0,
            structure=False,
            meter_level_supported=True,
            meter_level=3,
            meter_level_confidence=0.9,
        )
        for frame_id in range(3, 8)
    ]

    assert tuple(event for item in observations for event in confirmed_events(tracker.update(item))) == ()
    assert tracker.stable_meter_level == 3


def test_numeric_level_unknown_reset_is_bounded_at_ten_samples() -> None:
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    feed_levels(tracker, [2] * 10)
    assert feed_levels(tracker, [None] * 9, first_frame=10) == ()
    events = feed_levels(
        tracker,
        [1] * 4,
        first_frame=19,
        disrupted=True,
    )
    assert len(events) == 1

    tracker.reset("new round")
    feed_levels(tracker, [2] * 10, first_frame=0)
    assert feed_levels(tracker, [None] * 10, first_frame=10) == ()
    assert feed_levels(
        tracker,
        [1] * 4,
        first_frame=20,
        disrupted=True,
    ) == ()


def test_numeric_level_trackers_remain_side_isolated() -> None:
    p1 = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    p2 = GaugeTracker(Side.P2, "profile", GaugeTrackerConfig())
    feed_levels(p1, [3] * 10, side=Side.P1)
    feed_levels(p2, [2] * 10, side=Side.P2)

    p1_events = feed_levels(
        p1,
        [0, 0, 0, 0],
        first_frame=10,
        side=Side.P1,
        disrupted=True,
    )
    p2_events = feed_levels(
        p2,
        [1] * 4,
        first_frame=10,
        side=Side.P2,
        disrupted=True,
    )

    assert [(event.side, event.levels_spent) for event in p1_events] == [
        (Side.P1, 3)
    ]
    assert [(event.side, event.levels_spent) for event in p2_events] == [
        (Side.P2, 1)
    ]


def test_default_config_literals_are_frozen():
    config = GaugeTrackerConfig()
    assert (
        config.full_threshold,
        config.spent_threshold,
        config.full_samples,
        config.evidence_window,
        config.spent_samples,
        config.minimum_spend_seconds,
        config.ambiguous_disarm_samples,
        config.rearm_samples,
    ) == (0.78, 0.72, 5, 5, 3, 0.10, 3, 5)
    with pytest.raises(FrozenInstanceError):
        config.spent_samples = 2


@pytest.mark.parametrize("field", ["full_threshold", "spent_threshold"])
@pytest.mark.parametrize(
    "value",
    [Fraction(500_001, 1_000_000), 0.500001, Fraction(3, 4), 1],
)
def test_thresholds_accept_exact_open_lower_and_closed_upper_range(field, value):
    config = GaugeTrackerConfig(**{field: value})
    assert getattr(config, field) is value


@pytest.mark.parametrize("field", ["full_threshold", "spent_threshold"])
@pytest.mark.parametrize(
    "value",
    [0.499999, Fraction(499999, 1000000)],
)
def test_thresholds_reject_values_just_below_half_without_rounding(field, value):
    assert value < Fraction(1, 2)
    with pytest.raises(ValueError):
        GaugeTrackerConfig(**{field: value})


@pytest.mark.parametrize("field", ["full_threshold", "spent_threshold"])
def test_threshold_rational_just_above_half_avoids_float_and_math_narrowing(field):
    value = FloatNarrowingTrap(10**400 + 1, 2 * 10**400)
    assert isinstance(value, Rational)
    assert value > Fraction(1, 2)
    config = GaugeTrackerConfig(**{field: value})
    assert getattr(config, field) is value


@pytest.mark.parametrize("field", ["full_threshold", "spent_threshold"])
def test_thresholds_accept_numpy_real_without_coercion(field):
    value = np.float32(0.75)
    assert isinstance(value, Real)
    config = GaugeTrackerConfig(**{field: value})
    assert getattr(config, field) is value


@pytest.mark.parametrize("field", ["full_threshold", "spent_threshold"])
@pytest.mark.parametrize(
    "value",
    [
        Fraction(1, 2),
        0.5,
        Fraction(1_000_001, 1_000_000),
        1.000001,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_thresholds_reject_boundaries_outside_range_and_nonfinite(field, value):
    with pytest.raises(ValueError):
        GaugeTrackerConfig(**{field: value})


@pytest.mark.parametrize("field", ["full_threshold", "spent_threshold"])
@pytest.mark.parametrize("value", [True, False, "0.78", None, object()])
def test_thresholds_reject_booleans_and_non_reals(field, value):
    with pytest.raises(TypeError):
        GaugeTrackerConfig(**{field: value})


def test_integer_config_fields_accept_exact_boundaries():
    assert GaugeTrackerConfig(full_samples=5).full_samples == 5
    assert GaugeTrackerConfig(evidence_window=5).evidence_window == 5
    assert GaugeTrackerConfig(spent_samples=3).spent_samples == 3
    assert GaugeTrackerConfig(spent_samples=5).spent_samples == 5
    assert (
        GaugeTrackerConfig(ambiguous_disarm_samples=1).ambiguous_disarm_samples
        == 1
    )
    assert GaugeTrackerConfig(rearm_samples=5).rearm_samples == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_samples", 4),
        ("evidence_window", 4),
        ("evidence_window", 6),
        ("spent_samples", 2),
        ("spent_samples", 6),
        ("ambiguous_disarm_samples", 0),
        ("ambiguous_disarm_samples", -1),
        ("rearm_samples", 4),
    ],
)
def test_integer_config_fields_reject_outside_boundaries(field, value):
    with pytest.raises(ValueError):
        GaugeTrackerConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_samples", True),
        ("full_samples", 5.0),
        ("full_samples", float("nan")),
        ("full_samples", float("inf")),
        ("full_samples", float("-inf")),
        ("full_samples", IntSubclass(5)),
        ("full_samples", np.int64(5)),
        ("full_samples", "5"),
        ("full_samples", None),
        ("evidence_window", True),
        ("evidence_window", 5.0),
        ("evidence_window", float("nan")),
        ("evidence_window", float("inf")),
        ("evidence_window", float("-inf")),
        ("evidence_window", IntSubclass(5)),
        ("evidence_window", np.int64(5)),
        ("evidence_window", "5"),
        ("evidence_window", None),
        ("spent_samples", True),
        ("spent_samples", 3.0),
        ("spent_samples", float("nan")),
        ("spent_samples", float("inf")),
        ("spent_samples", float("-inf")),
        ("spent_samples", IntSubclass(3)),
        ("spent_samples", np.int64(3)),
        ("spent_samples", "3"),
        ("spent_samples", None),
        ("ambiguous_disarm_samples", True),
        ("ambiguous_disarm_samples", 1.0),
        ("ambiguous_disarm_samples", float("nan")),
        ("ambiguous_disarm_samples", float("inf")),
        ("ambiguous_disarm_samples", float("-inf")),
        ("ambiguous_disarm_samples", IntSubclass(1)),
        ("ambiguous_disarm_samples", np.int64(1)),
        ("ambiguous_disarm_samples", "1"),
        ("ambiguous_disarm_samples", None),
        ("rearm_samples", True),
        ("rearm_samples", 5.0),
        ("rearm_samples", float("nan")),
        ("rearm_samples", float("inf")),
        ("rearm_samples", float("-inf")),
        ("rearm_samples", IntSubclass(5)),
        ("rearm_samples", np.int64(5)),
        ("rearm_samples", "5"),
        ("rearm_samples", None),
    ],
)
def test_integer_config_fields_reject_bool_float_string_and_none(field, value):
    with pytest.raises(TypeError):
        GaugeTrackerConfig(**{field: value})


def test_minimum_spend_floor_is_exactly_one_tenth():
    exact_floor = Fraction(1, 10)
    config = GaugeTrackerConfig(minimum_spend_seconds=exact_floor)
    assert config.minimum_spend_seconds is exact_floor
    with pytest.raises(ValueError):
        GaugeTrackerConfig(
            minimum_spend_seconds=exact_floor - Fraction(1, 10**6)
        )


def test_minimum_spend_accepts_large_fraction_and_retains_real_objects():
    huge = Fraction(10**10000, 3)
    config = GaugeTrackerConfig(minimum_spend_seconds=huge)
    assert config.minimum_spend_seconds is huge
    finite_float = 0.25
    assert (
        GaugeTrackerConfig(minimum_spend_seconds=finite_float)
        .minimum_spend_seconds
        is finite_float
    )


def test_minimum_spend_accepts_numpy_real_without_coercion():
    value = np.float32(0.25)
    assert isinstance(value, Real)
    config = GaugeTrackerConfig(minimum_spend_seconds=value)
    assert config.minimum_spend_seconds is value


@pytest.mark.parametrize(
    "value",
    [Fraction(-1, 2), 0.099999, float("nan"), float("inf"), float("-inf")],
)
def test_minimum_spend_rejects_below_floor_and_nonfinite(value):
    with pytest.raises(ValueError):
        GaugeTrackerConfig(minimum_spend_seconds=value)


@pytest.mark.parametrize("value", [True, False, "0.10", None, object()])
def test_minimum_spend_rejects_booleans_and_non_reals(value):
    with pytest.raises(TypeError):
        GaugeTrackerConfig(minimum_spend_seconds=value)


@pytest.mark.parametrize("side", [Side.P1, Side.P2])
def test_tracker_constructor_accepts_both_sides(side):
    config = GaugeTrackerConfig()
    tracker = GaugeTracker(side, "profile", config)
    assert tracker.state is TrackerState.DISARMED


@pytest.mark.parametrize("side", ["P1", 1, True, None, object()])
def test_tracker_constructor_rejects_wrong_side(side):
    with pytest.raises(TypeError):
        GaugeTracker(side, "profile", GaugeTrackerConfig())


@pytest.mark.parametrize("profile_id", ["", " ", "\t\n"])
def test_tracker_constructor_rejects_blank_profile(profile_id):
    with pytest.raises(ValueError):
        GaugeTracker(Side.P1, profile_id, GaugeTrackerConfig())


@pytest.mark.parametrize(
    "profile_id",
    [None, 1, True, {}, object(), StrSubclass("profile")],
)
def test_tracker_constructor_rejects_non_string_profile(profile_id):
    with pytest.raises(TypeError):
        GaugeTracker(Side.P1, profile_id, GaugeTrackerConfig())


@pytest.mark.parametrize("config", [None, {}, {"full_samples": 5}, object()])
def test_tracker_constructor_rejects_wrong_config(config):
    with pytest.raises(TypeError):
        GaugeTracker(Side.P1, "profile", config)


def test_state_assignment_raises_initially_and_in_candidate():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert tracker.state is TrackerState.DISARMED
    initial_before = copy.deepcopy(vars(tracker))
    with pytest.raises(AttributeError):
        tracker.state = TrackerState.ARMED
    assert tracker.state is TrackerState.DISARMED
    assert vars(tracker) == initial_before
    arm_default(tracker)
    tracker.update(obs(5, 0.30, spent=0.72))
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    before = copy.deepcopy(vars(tracker))
    with pytest.raises(AttributeError):
        tracker.state = TrackerState.DISARMED
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert vars(tracker) == before


def test_reset_accepts_nonblank_reason_and_starts_a_fresh_ordering_epoch():
    tracker = candidate_tracker()
    assert tracker.reset("new round") == GaugeTrackerUpdate()
    assert tracker.state is TrackerState.DISARMED
    assert transition(tracker.update(obs(0, 0.00, full=0.78))) is None
    assert tracker.state is TrackerState.DISARMED
    for frame_id, sampled_at in (
        (1, 0.01),
        (2, 0.02),
        (3, 0.03),
        (4, 0.04),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED


def test_reset_clears_stale_ambiguous_streak_and_global_evidence():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.25))) is None
    assert transition(tracker.update(obs(6, 0.30))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE

    assert tracker.reset("round boundary") == GaugeTrackerUpdate()
    evidence_deques = [
        value for value in vars(tracker).values() if isinstance(value, deque)
    ]
    assert evidence_deques
    assert all(len(evidence) == 0 for evidence in evidence_deques)

    for frame_id, sampled_at in (
        (0, 0.00),
        (1, 0.05),
        (2, 0.10),
        (3, 0.15),
        (4, 0.20),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(5, 0.25))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert transition(tracker.update(obs(6, 0.30))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert transition(tracker.update(obs(7, 0.35))) is None
    assert tracker.state is TrackerState.DISARMED


@pytest.mark.parametrize(
    "reason",
    ["", " ", "\t\n", None, 1, True, {}, StrSubclass("reason")],
)
def test_invalid_reset_reason_rejects_before_any_mutation(reason):
    tracker = candidate_tracker()
    before = copy.deepcopy(vars(tracker))
    expected = ValueError if type(reason) is str else TypeError
    with pytest.raises(expected):
        tracker.reset(reason)
    assert vars(tracker) == before


def test_update_rejects_non_observation_atomically_from_candidate():
    tracker = candidate_tracker()
    before = copy.deepcopy(vars(tracker))
    with pytest.raises(TypeError):
        tracker.update({"frame_id": 6})
    assert vars(tracker) == before


def test_update_rejects_side_mismatch_atomically_from_candidate():
    tracker = candidate_tracker()
    before = copy.deepcopy(vars(tracker))
    with pytest.raises(ValueError):
        tracker.update(obs(6, 0.31, side=Side.P2, spent=0.72))
    assert vars(tracker) == before


@pytest.mark.parametrize(
    ("builder", "last_frame_id", "last_sampled_at"),
    [
        (disarmed_tracker, 2, 0.10),
        (armed_tracker, 4, 0.20),
        (candidate_tracker, 5, 0.30),
        (cooldown_tracker, 7, 0.42),
    ],
    ids=["disarmed", "armed", "candidate", "cooldown"],
)
@pytest.mark.parametrize(
    "violation",
    ["duplicate_frame", "backward_frame", "backward_time"],
)
def test_ordering_violations_are_atomic_in_every_persistent_state(
    builder,
    last_frame_id,
    last_sampled_at,
    violation,
):
    tracker = builder()
    if violation == "duplicate_frame":
        invalid = obs(last_frame_id, last_sampled_at + 0.01)
    elif violation == "backward_frame":
        invalid = obs(last_frame_id - 1, last_sampled_at + 0.01)
    else:
        invalid = obs(last_frame_id + 1, last_sampled_at - 0.01)
    before = copy.deepcopy(vars(tracker))
    with pytest.raises(ValueError):
        tracker.update(invalid)
    assert vars(tracker) == before


@pytest.mark.parametrize(
    ("builder", "next_frame_id", "same_time", "expected_state"),
    [
        (disarmed_tracker, 3, 0.10, TrackerState.DISARMED),
        (armed_tracker, 5, 0.20, TrackerState.SPEND_CANDIDATE),
        (candidate_tracker, 6, 0.30, TrackerState.SPEND_CANDIDATE),
        (cooldown_tracker, 8, 0.42, TrackerState.COOLDOWN),
    ],
    ids=["disarmed", "armed", "candidate", "cooldown"],
)
def test_equal_time_is_accepted_in_every_persistent_state(
    builder,
    next_frame_id,
    same_time,
    expected_state,
):
    tracker = builder()
    assert transition(tracker.update(obs(next_frame_id, same_time))) is None
    assert tracker.state is expected_state


def test_equal_timestamps_with_increasing_frames_are_accepted_but_not_elapsed():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id in (0, 1, 2, 3, 4):
        assert transition(tracker.update(obs(frame_id, 0.00, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(5, 0.00, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.00, spent=0.72))) is None
    assert transition(tracker.update(obs(7, 0.00, spent=0.72))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE


def test_hidden_observation_advances_ordering_then_disarms_and_clears_history():
    tracker = candidate_tracker()
    assert transition(tracker.update(obs(6, 0.31))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert transition(tracker.update(obs(7, 0.32, visible=False))) is None
    assert tracker.state is TrackerState.DISARMED
    evidence_deques = [
        value for value in vars(tracker).values() if isinstance(value, deque)
    ]
    assert evidence_deques
    assert all(len(evidence) == 0 for evidence in evidence_deques)
    for invalid in (
        obs(7, 0.32, full=0.78),
        obs(6, 0.33, full=0.78),
        obs(8, 0.31, full=0.78),
    ):
        before = copy.deepcopy(vars(tracker))
        with pytest.raises(ValueError):
            tracker.update(invalid)
        assert vars(tracker) == before

    for frame_id, sampled_at in (
        (8, 0.35),
        (9, 0.40),
        (10, 0.45),
        (11, 0.50),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
        assert tracker.state is TrackerState.DISARMED
    assert transition(tracker.update(obs(12, 0.55, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(13, 0.60))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert transition(tracker.update(obs(14, 0.65))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert transition(tracker.update(obs(15, 0.70))) is None
    assert tracker.state is TrackerState.DISARMED


def test_hidden_observation_clears_stale_cooldown_rearm_history():
    tracker = cooldown_tracker()
    for frame_id, sampled_at in (
        (8, 0.50),
        (9, 0.55),
        (10, 0.60),
        (11, 0.65),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
        assert tracker.state is TrackerState.COOLDOWN

    assert transition(tracker.update(obs(12, 0.70, visible=False))) is None
    assert tracker.state is TrackerState.DISARMED
    for frame_id, sampled_at in (
        (13, 0.75),
        (14, 0.80),
        (15, 0.85),
        (16, 0.90),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
        assert tracker.state is TrackerState.DISARMED
    assert transition(tracker.update(obs(17, 0.95, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED

    assert transition(tracker.update(obs(18, 1.00, spent=0.72))) is None
    assert transition(tracker.update(obs(19, 1.06, spent=0.72))) is None
    events = confirmed_events(tracker.update(obs(20, 1.12, spent=0.72)))
    assert len(events) == 1
    assert events[0].pre_spend_frame_id == 17
    assert tracker.state is TrackerState.COOLDOWN
    assert transition(tracker.update(obs(21, 1.20, full=0.78))) is None
    assert tracker.state is TrackerState.COOLDOWN


def test_classification_full_threshold_is_inclusive():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id, sampled_at in (
        (0, 0.00),
        (1, 0.05),
        (2, 0.10),
        (3, 0.15),
        (4, 0.20),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED


def test_classification_spent_threshold_is_inclusive():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.36, spent=0.72))) is None
    events = confirmed_events(tracker.update(obs(7, 0.42, spent=0.72)))
    assert len(events) == 1


def test_classification_neither_hit_is_ambiguous():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, full=0.50, spent=0.50))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE


def test_double_hit_classifies_fail_closed_ambiguous():
    assert (
        _classify_confidences(0.90, 0.90, GaugeTrackerConfig())
        == "ambiguous"
    )


def test_spent_without_prior_stable_full_never_confirms():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id, sampled_at in (
        (0, 0.00),
        (1, 0.05),
        (2, 0.10),
        (3, 0.15),
        (4, 0.20),
        (5, 0.25),
        (6, 0.30),
        (7, 0.35),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, spent=0.72))) is None
    assert tracker.state is TrackerState.DISARMED


def test_partial_sa1_sa2_like_values_never_arm_or_confirm():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id, sampled_at, full, spent in (
        (0, 0.00, 0.60, 0.40),
        (1, 0.05, 0.40, 0.60),
        (2, 0.10, 0.60, 0.40),
        (3, 0.15, 0.40, 0.60),
        (4, 0.20, 0.60, 0.40),
        (5, 0.25, 0.40, 0.60),
        (6, 0.30, 0.60, 0.40),
    ):
        assert transition(tracker.update(
            obs(frame_id, sampled_at, full=full, spent=spent)
        )) is None
    assert tracker.state is TrackerState.DISARMED


def test_only_consecutive_full_observations_arm_disarmed_tracker():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id, sampled_at, full in (
        (0, 0.00, 0.78),
        (1, 0.05, 0.78),
        (2, 0.10, 0.50),
        (3, 0.15, 0.78),
        (4, 0.20, 0.78),
        (5, 0.25, 0.78),
        (6, 0.30, 0.78),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=full))) is None
    assert tracker.state is TrackerState.DISARMED
    assert transition(tracker.update(obs(7, 0.35, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED


def test_p1_and_p2_trackers_keep_state_and_events_isolated():
    p1 = GaugeTracker(Side.P1, "p1-profile", GaugeTrackerConfig())
    p2 = GaugeTracker(Side.P2, "p2-profile", GaugeTrackerConfig())
    arm_default(p1, side=Side.P1)
    arm_default(p2, side=Side.P2)
    assert transition(p1.update(obs(5, 0.30, side=Side.P1, spent=0.72))) is None
    assert transition(p1.update(obs(6, 0.36, side=Side.P1, spent=0.72))) is None
    p1_events = confirmed_events(p1.update(obs(7, 0.42, side=Side.P1, spent=0.72)))
    assert len(p1_events) == 1 and p1_events[0].side is Side.P1
    assert p2.state is TrackerState.ARMED
    assert transition(p2.update(obs(5, 0.30, side=Side.P2, spent=0.72))) is None
    assert transition(p2.update(obs(6, 0.36, side=Side.P2, spent=0.72))) is None
    p2_events = confirmed_events(p2.update(obs(7, 0.42, side=Side.P2, spent=0.72)))
    assert len(p2_events) == 1 and p2_events[0].side is Side.P2


def test_five_full_then_only_three_spent_confirms_on_third():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id, t in enumerate((0.00, 0.05, 0.10, 0.15, 0.20)):
        assert transition(tracker.update(obs(frame_id, t, full=0.78))) is None
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.35, spent=0.72))) is None
    events = confirmed_events(tracker.update(obs(7, 0.41, spent=0.72)))
    assert isinstance(events, tuple) and len(events) == 1
    event = events[0]
    assert (
        event.side,
        event.confirmed_at,
        event.pre_spend_frame_id,
        event.confirmed_frame_id,
        event.profile_id,
    ) == (Side.P1, 0.41, 4, 7, "profile")
    assert tracker.state is TrackerState.COOLDOWN
    assert transition(tracker.update(obs(8, 0.45, spent=0.72))) is None


def test_huge_fraction_confirmed_at_preserves_identity():
    huge = Fraction(10**10000, 3)
    tracker = GaugeTracker(Side.P2, "fraction-profile", GaugeTrackerConfig())
    full_times = (
        huge - 1,
        huge - Fraction(4, 5),
        huge - Fraction(3, 5),
        huge - Fraction(2, 5),
        huge - Fraction(3, 10),
    )
    for frame_id, t in enumerate(full_times):
        assert transition(tracker.update(
            obs(frame_id, t, side=Side.P2, full=0.78)
        )) is None
    assert transition(tracker.update(
        obs(5, huge - Fraction(1, 5), side=Side.P2, spent=0.72)
    )) is None
    assert transition(tracker.update(
        obs(6, huge - Fraction(1, 10), side=Side.P2, spent=0.72)
    )) is None
    event = transition(tracker.update(obs(7, huge, side=Side.P2, spent=0.72)))
    assert isinstance(event, ConfirmedSuperEvent)
    assert event.confirmed_at is huge


def test_exact_fraction_duration_boundary_is_inclusive_and_below_does_not_emit():
    config = GaugeTrackerConfig(minimum_spend_seconds=Fraction(1, 10))
    exact = GaugeTracker(Side.P1, "exact", config)
    below = GaugeTracker(Side.P1, "below", config)
    for tracker in (exact, below):
        for frame_id, sampled_at in (
            (0, Fraction(0, 1)),
            (1, Fraction(1, 20)),
            (2, Fraction(1, 10)),
            (3, Fraction(3, 20)),
            (4, Fraction(1, 5)),
        ):
            assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
    assert transition(exact.update(obs(5, Fraction(3, 10), spent=0.72))) is None
    assert transition(exact.update(obs(6, Fraction(7, 20), spent=0.72))) is None
    exact_events = confirmed_events(exact.update(obs(7, Fraction(2, 5), spent=0.72)))
    assert len(exact_events) == 1
    assert transition(below.update(obs(5, Fraction(3, 10), spent=0.72))) is None
    assert transition(below.update(obs(6, Fraction(7, 20), spent=0.72))) is None
    just_below = Fraction(3, 10) + Fraction(1, 10) - Fraction(1, 10**6)
    assert transition(below.update(obs(7, just_below, spent=0.72))) is None
    assert below.state is TrackerState.SPEND_CANDIDATE


def test_duration_comparison_does_not_narrow_ultra_close_fraction():
    config = GaugeTrackerConfig(minimum_spend_seconds=Fraction(1, 10))
    tracker = GaugeTracker(Side.P1, "ultra-close", config)
    for frame_id, sampled_at in (
        (0, Fraction(0, 1)),
        (1, Fraction(1, 20)),
        (2, Fraction(1, 10)),
        (3, Fraction(3, 20)),
        (4, Fraction(1, 5)),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
    first_spent = Fraction(3, 10)
    assert transition(tracker.update(obs(5, first_spent, spent=0.72))) is None
    assert transition(tracker.update(obs(6, Fraction(7, 20), spent=0.72))) is None
    just_below = Fraction(1, 10) - Fraction(1, 10**1000)
    assert transition(tracker.update(obs(7, first_spent + just_below, spent=0.72))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE


def test_armed_full_observations_occupy_global_evidence_slots():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.36, spent=0.72))) is None
    assert transition(tracker.update(obs(7, 0.40, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(8, 0.45, full=0.78))) is None
    assert transition(tracker.update(obs(9, 0.50, full=0.78))) is None
    assert transition(tracker.update(obs(10, 0.60, spent=0.72))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE


def test_two_spent_entries_in_fixed_five_window_do_not_confirm():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    for frame_id, sampled_at, classification in (
        (5, 0.30, "A"),
        (6, 0.35, "A"),
        (7, 0.40, "S"),
        (8, 0.45, "A"),
        (9, 0.51, "S"),
    ):
        current = (
            obs(frame_id, sampled_at, spent=0.72)
            if classification == "S"
            else obs(frame_id, sampled_at)
        )
        assert transition(tracker.update(current)) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE


def test_ambiguous_slots_evict_old_spent_evidence_from_fixed_window():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    for frame_id, sampled_at, classification in (
        (5, 0.30, "S"),
        (6, 0.35, "A"),
        (7, 0.40, "A"),
        (8, 0.45, "S"),
        (9, 0.50, "A"),
        (10, 0.55, "S"),
    ):
        current = (
            obs(frame_id, sampled_at, spent=0.72)
            if classification == "S"
            else obs(frame_id, sampled_at)
        )
        assert transition(tracker.update(current)) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE


def test_candidate_ambiguous_observation_never_routes_to_confirmation():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    for frame_id, sampled_at in ((5, 0.30), (6, 0.34), (7, 0.38)):
        assert transition(tracker.update(obs(frame_id, sampled_at, spent=0.72))) is None
    assert transition(tracker.update(obs(8, 0.45))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE


def test_duration_anchor_is_earliest_spent_in_current_window_only():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    for frame_id, sampled_at, classification in (
        (5, 0.30, "S"),
        (6, 0.35, "A"),
        (7, 0.40, "A"),
        (8, 0.45, "S"),
        (9, 0.46, "A"),
        (10, 0.47, "S"),
        (11, 0.48, "S"),
    ):
        current = (
            obs(frame_id, sampled_at, spent=0.72)
            if classification == "S"
            else obs(frame_id, sampled_at)
        )
        assert transition(tracker.update(current)) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE


def test_candidate_spent_resets_consecutive_ambiguous_disarm_counter():
    tracker = GaugeTracker(
        Side.P1,
        "profile",
        GaugeTrackerConfig(ambiguous_disarm_samples=2),
    )
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30))) is None
    assert transition(tracker.update(obs(6, 0.35, spent=0.72))) is None
    assert transition(tracker.update(obs(7, 0.40))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert transition(tracker.update(obs(8, 0.45))) is None
    assert tracker.state is TrackerState.DISARMED


def test_candidate_ambiguous_streak_resets_when_full_bounces_back_to_armed():
    tracker = GaugeTracker(
        Side.P1,
        "profile",
        GaugeTrackerConfig(ambiguous_disarm_samples=2),
    )
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.25))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert transition(tracker.update(obs(6, 0.30, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(7, 0.35))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert transition(tracker.update(obs(8, 0.40))) is None
    assert tracker.state is TrackerState.DISARMED


def test_three_candidate_ambiguous_samples_disarm_and_clear_detection_history():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.25))) is None
    assert transition(tracker.update(obs(6, 0.30))) is None
    assert transition(tracker.update(obs(7, 0.35))) is None
    assert tracker.state is TrackerState.DISARMED
    evidence_deques = [
        value for value in vars(tracker).values() if isinstance(value, deque)
    ]
    assert evidence_deques
    assert all(len(evidence) == 0 for evidence in evidence_deques)
    for frame_id, sampled_at in (
        (8, 0.40),
        (9, 0.45),
        (10, 0.50),
        (11, 0.55),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
    assert tracker.state is TrackerState.DISARMED
    assert transition(tracker.update(obs(12, 0.60, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(13, 0.65))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE


def test_armed_full_refreshes_pre_spend_frame_before_event():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.25, full=0.78))) is None
    assert transition(tracker.update(obs(6, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(7, 0.36, spent=0.72))) is None
    events = confirmed_events(tracker.update(obs(8, 0.42, spent=0.72)))
    assert len(events) == 1
    assert events[0].pre_spend_frame_id == 5


def test_candidate_to_full_refreshes_pre_spend_frame_before_event():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.25))) is None
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    assert transition(tracker.update(obs(6, 0.30, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(7, 0.35, spent=0.72))) is None
    assert transition(tracker.update(obs(8, 0.41, spent=0.72))) is None
    events = confirmed_events(tracker.update(obs(9, 0.47, spent=0.72)))
    assert len(events) == 1
    assert events[0].pre_spend_frame_id == 6


def test_event_contract_uses_uuid_original_time_latest_full_and_no_duplicates(
    monkeypatch,
):
    monkeypatch.setattr(
        gauge_tracker_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="0123456789abcdef"),
    )
    tracker = GaugeTracker(Side.P2, "exact-profile", GaugeTrackerConfig())
    for frame_id, sampled_at in (
        (0, Fraction(0, 1)),
        (1, Fraction(1, 20)),
        (2, Fraction(1, 10)),
        (3, Fraction(3, 20)),
        (4, Fraction(1, 5)),
    ):
        assert transition(tracker.update(
            obs(frame_id, sampled_at, side=Side.P2, full=0.78)
        )) is None
    assert transition(tracker.update(
        obs(5, Fraction(1, 4), side=Side.P2, full=0.78)
    )) is None
    assert transition(tracker.update(
        obs(6, Fraction(3, 10), side=Side.P2, spent=0.72)
    )) is None
    assert transition(tracker.update(
        obs(7, Fraction(9, 25), side=Side.P2, spent=0.72)
    )) is None
    confirming_time = Fraction(21, 50)
    events = confirmed_events(tracker.update(
        obs(8, confirming_time, side=Side.P2, spent=0.72)
    ))
    assert isinstance(events, tuple) and len(events) == 1
    event = events[0]
    assert event.event_id == "0123456789abcdef"
    assert event.side is Side.P2
    assert event.confirmed_at is confirming_time
    assert event.pre_spend_frame_id == 5
    assert event.confirmed_frame_id == 8
    assert event.profile_id == "exact-profile"
    assert tracker.state is TrackerState.COOLDOWN
    assert transition(tracker.update(
        obs(9, Fraction(43, 100), side=Side.P2, spent=0.72)
    )) is None
    assert transition(tracker.update(obs(10, Fraction(11, 25), side=Side.P2))) is None


def test_event_is_constructed_while_transiently_confirmed_then_returns_cooldown(
    monkeypatch,
):
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    observed_states = []

    def construct_event(**kwargs):
        observed_states.append(tracker.state)
        return ConfirmedSuperEvent(**kwargs)

    monkeypatch.setattr(
        gauge_tracker_module,
        "ConfirmedSuperEvent",
        construct_event,
    )
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.36, spent=0.72))) is None
    events = confirmed_events(tracker.update(obs(7, 0.42, spent=0.72)))
    assert len(events) == 1
    assert observed_states == [TrackerState.CONFIRMED]
    assert tracker.state is TrackerState.COOLDOWN


def test_color_meter_level_wrap_is_ignored_but_disrupted_spend_confirms():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())

    for frame_id, sampled_at in (
        (0, 0.00),
        (1, 0.05),
        (2, 0.10),
        (3, 0.15),
        (4, 0.20),
    ):
        assert transition(tracker.update(
            obs(frame_id, sampled_at, full=0.80, meter=0.80)
        )) is None

    for frame_id, sampled_at in ((5, 0.30), (6, 0.36), (7, 0.42)):
        assert transition(tracker.update(
            obs(frame_id, sampled_at, spent=0.99, meter=0.0)
        )) is None
    assert transition(tracker.update(
        obs(8, 0.48, spent=0.95, structure=False)
    )) is None
    assert tracker.state is TrackerState.DISARMED

    for frame_id, sampled_at in (
        (9, 0.55),
        (10, 0.60),
        (11, 0.65),
        (12, 0.70),
        (13, 0.75),
    ):
        assert transition(tracker.update(
            obs(frame_id, sampled_at, full=0.90, meter=0.90)
        )) is None

    assert transition(tracker.update(
        obs(14, 0.85, spent=0.95, structure=False)
    )) is None
    assert transition(tracker.update(obs(15, 0.91, spent=0.95, meter=0.0))) is None
    events = confirmed_events(tracker.update(obs(16, 0.97, spent=0.95, meter=0.0)))

    assert len(events) == 1
    assert events[0].pre_spend_frame_id == 13


def test_default_rearm_requires_consecutive_full_and_supports_three_event_cycles():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.36, spent=0.72))) is None
    events = list(confirmed_events(tracker.update(obs(7, 0.42, spent=0.72))))
    assert len(events) == 1
    for frame_id, sampled_at, classification in (
        (8, 0.50, "F"),
        (9, 0.51, "A"),
        (10, 0.55, "F"),
        (11, 0.56, "A"),
        (12, 0.60, "F"),
        (13, 0.61, "A"),
        (14, 0.65, "F"),
        (15, 0.66, "A"),
        (16, 0.70, "F"),
        (17, 0.71, "A"),
    ):
        current = (
            obs(frame_id, sampled_at, full=0.78)
            if classification == "F"
            else obs(frame_id, sampled_at)
        )
        assert transition(tracker.update(current)) is None
        assert tracker.state is TrackerState.COOLDOWN
    for frame_id, sampled_at in ((18, 0.80), (19, 0.86), (20, 0.92)):
        assert transition(tracker.update(obs(frame_id, sampled_at, spent=0.72))) is None
    assert len(events) == 1
    assert tracker.state is TrackerState.COOLDOWN
    for frame_id, sampled_at in (
        (21, 0.95),
        (22, 1.00),
        (23, 1.05),
        (24, 1.10),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
        assert tracker.state is TrackerState.COOLDOWN
    assert transition(tracker.update(obs(25, 1.15, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(26, 1.20, spent=0.72))) is None
    assert transition(tracker.update(obs(27, 1.26, spent=0.72))) is None
    second = confirmed_events(tracker.update(obs(28, 1.32, spent=0.72)))
    assert len(second) == 1
    assert second[0].pre_spend_frame_id == 25
    events.extend(second)
    assert len(events) == 2
    for frame_id, sampled_at in (
        (29, 1.40),
        (30, 1.45),
        (31, 1.50),
        (32, 1.55),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
        assert tracker.state is TrackerState.COOLDOWN
    assert transition(tracker.update(obs(33, 1.60, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(34, 1.70, spent=0.72))) is None
    assert transition(tracker.update(obs(35, 1.76, spent=0.72))) is None
    third = confirmed_events(tracker.update(obs(36, 1.82, spent=0.72)))
    assert len(third) == 1
    assert third[0].pre_spend_frame_id == 33
    events.extend(third)
    assert len(events) == 3


def test_nondefault_full_threshold_is_honored():
    tracker = GaugeTracker(
        Side.P1,
        "profile",
        GaugeTrackerConfig(full_threshold=0.90),
    )
    for frame_id, sampled_at in (
        (0, 0.00),
        (1, 0.05),
        (2, 0.10),
        (3, 0.15),
        (4, 0.20),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.89))) is None
    assert tracker.state is TrackerState.DISARMED
    for frame_id, sampled_at in (
        (5, 0.25),
        (6, 0.30),
        (7, 0.35),
        (8, 0.40),
        (9, 0.45),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.90))) is None
    assert tracker.state is TrackerState.ARMED


def test_nondefault_spent_threshold_is_honored():
    tracker = GaugeTracker(
        Side.P1,
        "profile",
        GaugeTrackerConfig(spent_threshold=0.90),
    )
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.90))) is None
    assert transition(tracker.update(obs(6, 0.35, spent=0.89))) is None
    assert transition(tracker.update(obs(7, 0.40, spent=0.89))) is None
    assert transition(tracker.update(obs(8, 0.45, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(9, 0.50, spent=0.90))) is None
    assert transition(tracker.update(obs(10, 0.56, spent=0.90))) is None
    events = confirmed_events(tracker.update(obs(11, 0.62, spent=0.90)))
    assert len(events) == 1


def test_nondefault_full_samples_is_honored():
    tracker = GaugeTracker(
        Side.P1,
        "profile",
        GaugeTrackerConfig(full_samples=6),
    )
    for frame_id, sampled_at in (
        (0, 0.00),
        (1, 0.05),
        (2, 0.10),
        (3, 0.15),
        (4, 0.20),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
    assert tracker.state is TrackerState.DISARMED
    assert transition(tracker.update(obs(5, 0.25, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED


def test_nondefault_spent_samples_is_honored():
    tracker = GaugeTracker(
        Side.P1,
        "profile",
        GaugeTrackerConfig(spent_samples=4),
    )
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.35, spent=0.72))) is None
    assert transition(tracker.update(obs(7, 0.41, spent=0.72))) is None
    events = confirmed_events(tracker.update(obs(8, 0.46, spent=0.72)))
    assert len(events) == 1


def test_nondefault_minimum_spend_seconds_is_honored():
    tracker = GaugeTracker(
        Side.P1,
        "profile",
        GaugeTrackerConfig(minimum_spend_seconds=0.20),
    )
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.35, spent=0.72))) is None
    assert transition(tracker.update(obs(7, 0.40, spent=0.72))) is None
    events = confirmed_events(tracker.update(obs(8, 0.51, spent=0.72)))
    assert len(events) == 1


def test_nondefault_ambiguous_disarm_samples_is_honored():
    tracker = GaugeTracker(
        Side.P1,
        "profile",
        GaugeTrackerConfig(ambiguous_disarm_samples=1),
    )
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30))) is None
    assert tracker.state is TrackerState.DISARMED


def test_nondefault_rearm_samples_is_honored():
    tracker = GaugeTracker(
        Side.P1,
        "profile",
        GaugeTrackerConfig(rearm_samples=6),
    )
    arm_default(tracker)
    assert transition(tracker.update(obs(5, 0.30, spent=0.72))) is None
    assert transition(tracker.update(obs(6, 0.36, spent=0.72))) is None
    confirmed = transition(tracker.update(obs(7, 0.42, spent=0.72)))
    assert isinstance(confirmed, ConfirmedSuperEvent)
    for frame_id, sampled_at in (
        (8, 0.50),
        (9, 0.55),
        (10, 0.60),
        (11, 0.65),
        (12, 0.70),
    ):
        assert transition(tracker.update(obs(frame_id, sampled_at, full=0.78))) is None
        assert tracker.state is TrackerState.COOLDOWN
    assert transition(tracker.update(obs(13, 0.75, full=0.78))) is None
    assert tracker.state is TrackerState.ARMED
    assert transition(tracker.update(obs(14, 0.80, spent=0.72))) is None
    assert transition(tracker.update(obs(15, 0.86, spent=0.72))) is None
    second = confirmed_events(tracker.update(obs(16, 0.92, spent=0.72)))
    assert len(second) == 1
