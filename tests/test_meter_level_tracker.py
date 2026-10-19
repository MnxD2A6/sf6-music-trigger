from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sf6_music_trigger.domain import Side
from sf6_music_trigger.meter_level_features import MeterLevelReading
from sf6_music_trigger.meter_level_tracker import (
    MeterLevelTransition,
    StableMeterLevelTracker,
)


def _reading(side: Side, level: int | None) -> MeterLevelReading:
    return MeterLevelReading(
        side=side,
        level=level,
        confidence=0.9 if level is not None else 0.0,
        visible=level is not None,
    )


def _feed(
    tracker: StableMeterLevelTracker,
    levels: list[int | None],
    *,
    first_frame: int = 0,
    side: Side = Side.P1,
) -> tuple[MeterLevelTransition, ...]:
    events: list[MeterLevelTransition] = []
    for offset, level in enumerate(levels):
        event = tracker.update(_reading(side, level), first_frame + offset)
        if event is not None:
            events.append(event)
    return tuple(events)


def test_three_lower_readings_emit_once() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    assert _feed(tracker, [2, 2, 2], first_frame=1) == ()

    events = _feed(tracker, [1, 1, 1, 1], first_frame=4)

    assert [(event.before, event.after, event.levels_spent) for event in events] == [
        (2, 1, 1)
    ]
    assert (events[0].pre_frame_id, events[0].confirmed_frame_id) == (3, 6)


def test_baseline_must_reach_its_independent_dwell_before_a_spend_can_emit() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1,
        stable_samples=3,
        baseline_samples=10,
        unknown_reset_samples=10,
    )
    assert _feed(tracker, [3] * 3, first_frame=0) == ()
    assert _feed(tracker, [0] * 3, first_frame=3) == ()

    tracker.reset()
    assert _feed(tracker, [3] * 10, first_frame=0) == ()
    events = _feed(tracker, [0] * 3, first_frame=10)

    assert [(event.before, event.after) for event in events] == [(3, 0)]


def test_trusted_gain_is_ready_for_an_immediate_followup_spend() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1,
        stable_samples=3,
        baseline_samples=10,
        unknown_reset_samples=10,
    )
    assert _feed(tracker, [1] * 10, first_frame=0) == ()
    for frame_id in range(10, 13):
        assert tracker.update(
            _reading(Side.P1, 2),
            frame_id,
            baseline_trusted=True,
        ) is None

    events = _feed(tracker, [0] * 3, first_frame=13)

    assert [(event.before, event.after) for event in events] == [(2, 0)]


@pytest.mark.parametrize(("before", "after"), [(3, 2), (3, 1), (3, 0)])
def test_spends_one_two_or_three_levels(before: int, after: int) -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    _feed(tracker, [before] * 3, first_frame=1)

    events = _feed(tracker, [after] * 3, first_frame=4)

    assert len(events) == 1
    assert events[0].levels_spent == before - after


def test_stable_duplicate_does_not_emit_again() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    _feed(tracker, [3, 3, 3], first_frame=1)

    events = _feed(tracker, [2, 2, 2, 2, 2, 2], first_frame=4)

    assert len(events) == 1


def test_gain_and_round_gap_never_emit() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    assert _feed(tracker, [0, 0, 0, 1, 1, 1]) == ()
    assert _feed(tracker, [None] * 10, first_frame=6) == ()
    assert _feed(tracker, [0, 0, 0], first_frame=16) == ()


def test_gain_replaces_baseline_for_later_real_spend() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    assert _feed(tracker, [0, 0, 0, 2, 2, 2]) == ()

    events = _feed(tracker, [1, 1, 1], first_frame=6)

    assert len(events) == 1
    assert (events[0].before, events[0].after) == (2, 1)


def test_oscillation_never_becomes_stable_or_emits() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    _feed(tracker, [3, 3, 3], first_frame=1)

    assert _feed(tracker, [2, 1, 2, 1, 2, 1], first_frame=4) == ()


def test_nine_unknowns_preserve_stable_baseline() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    _feed(tracker, [2, 2, 2], first_frame=1)
    assert _feed(tracker, [None] * 9, first_frame=4) == ()

    events = _feed(tracker, [1, 1, 1], first_frame=13)

    assert len(events) == 1
    assert (events[0].before, events[0].after) == (2, 1)


def test_tenth_unknown_clears_stable_baseline() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    _feed(tracker, [2, 2, 2], first_frame=1)
    assert _feed(tracker, [None] * 10, first_frame=4) == ()

    assert _feed(tracker, [1, 1, 1], first_frame=14) == ()


def test_stable_level_property_is_read_only_and_tracks_reset() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    assert tracker.stable_level is None
    _feed(tracker, [2, 2, 2], first_frame=1)
    assert tracker.stable_level == 2
    tracker.reset()
    assert tracker.stable_level is None
    with pytest.raises(AttributeError):
        tracker.stable_level = 1  # type: ignore[misc]


def test_tracker_exposes_ready_baseline_without_allowing_candidate_mutation() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, baseline_samples=10, unknown_reset_samples=10
    )
    for frame_id in range(10):
        assert tracker.update(_reading(Side.P1, 3), frame_id) is None
    assert tracker.stable_level == 3
    assert tracker.stable_frame_id == 9
    assert tracker.baseline_ready is True


@pytest.mark.parametrize("value", [None, 0, 1, "true", object()])
def test_baseline_observed_requires_an_actual_boolean(value: object) -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, baseline_samples=10, unknown_reset_samples=10
    )
    with pytest.raises(TypeError, match="baseline_observed must be a boolean"):
        tracker.update(_reading(Side.P1, 3), 1, baseline_observed=value)  # type: ignore[arg-type]


def test_explicit_reset_clears_baseline_and_frame_order_history() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    _feed(tracker, [3, 3, 3], first_frame=20)

    tracker.reset()

    assert _feed(tracker, [0, 0, 0], first_frame=0) == ()


def test_side_mismatch_is_rejected_without_advancing_frame() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    with pytest.raises(ValueError, match="reading side does not match"):
        tracker.update(_reading(Side.P2, 3), 1)

    assert tracker.update(_reading(Side.P1, 3), 1) is None


@pytest.mark.parametrize("frame_id", [True, 1.0, "1", None])
def test_frame_id_must_be_actual_integer(frame_id: object) -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    with pytest.raises(TypeError, match="frame_id must be an integer"):
        tracker.update(_reading(Side.P1, 3), frame_id)  # type: ignore[arg-type]


def test_frame_ids_must_increase_strictly() -> None:
    tracker = StableMeterLevelTracker(
        Side.P1, stable_samples=3, unknown_reset_samples=10
    )
    tracker.update(_reading(Side.P1, 3), 5)
    with pytest.raises(ValueError, match="frame_id must increase strictly"):
        tracker.update(_reading(Side.P1, 3), 5)
    with pytest.raises(ValueError, match="frame_id must increase strictly"):
        tracker.update(_reading(Side.P1, 3), 4)


@pytest.mark.parametrize("value", [True, 0, -1, 1.0, "3"])
def test_tracker_rejects_invalid_sample_counts(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        StableMeterLevelTracker(
            Side.P1,
            stable_samples=value,  # type: ignore[arg-type]
            unknown_reset_samples=10,
        )
    with pytest.raises((TypeError, ValueError)):
        StableMeterLevelTracker(
            Side.P1,
            stable_samples=3,
            unknown_reset_samples=value,  # type: ignore[arg-type]
        )


def test_transition_is_strict_frozen_value() -> None:
    transition = MeterLevelTransition(Side.P1, 3, 0, 10, 13)
    assert transition.levels_spent == 3
    with pytest.raises(FrozenInstanceError):
        transition.after = 1  # type: ignore[misc]

    with pytest.raises(ValueError, match="before must be greater than after"):
        MeterLevelTransition(Side.P1, 1, 1, 10, 13)
    with pytest.raises(ValueError, match="confirmed_frame_id must be after"):
        MeterLevelTransition(Side.P1, 3, 0, 10, 10)
