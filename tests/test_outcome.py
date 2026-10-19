from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from sf6_music_trigger.config import SuperScope
from sf6_music_trigger.domain import CapturedFrame, ClientGeometry, ConfirmedSuperEvent, Side
from sf6_music_trigger.gauge_features import GaugeObservation


def _frame(frame_id: int, sampled_at: float, code: int) -> CapturedFrame:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[0, 0, 0] = code
    return CapturedFrame(frame_id, image, sampled_at, ClientGeometry(0, 0, 40, 20))


def _event(side: Side, *, event_id: str | None = None, confirmed_at: float = 1.0,
           pre_spend_frame_id: int = 4, confirmed_frame_id: int = 9) -> ConfirmedSuperEvent:
    return ConfirmedSuperEvent(event_id or f"event-{side.value}", side, confirmed_at,
                               pre_spend_frame_id, confirmed_frame_id, "profile")


class _Features:
    def __init__(self, values: dict[tuple[int, Side], tuple[float, float]]) -> None:
        self.values = values
        self.calls: list[tuple[int, Side]] = []

    def __call__(self, image: np.ndarray, side: Side) -> tuple[float, float]:
        key = (int(image[0, 0, 0]), side)
        self.calls.append(key)
        return self.values[key]


def _observation(side: Side, frame_id: int, sampled_at: float, bucket: int | None,
                 *, visible: bool = True, verified: bool = True) -> GaugeObservation:
    if not visible:
        return GaugeObservation(side, frame_id, sampled_at, False, 0.0, 0.0, 1.0, 1.0)
    normalized = None if bucket is None else bucket / 3
    full_distance = 0.5 if normalized is None else 1.0 - normalized
    spent_distance = 0.5 if normalized is None else normalized
    total = full_distance + spent_distance
    return GaugeObservation(side, frame_id, sampled_at, True, spent_distance / total,
                            full_distance / total, full_distance, spent_distance,
                            verified, normalized if verified else None)


def _meters(frame_id: int, sampled_at: float, *, p1: int | None = 0,
            p2: int | None = 0, visible: bool = True) -> tuple[GaugeObservation, GaugeObservation]:
    return (_observation(Side.P1, frame_id, sampled_at, p1, visible=visible),
            _observation(Side.P2, frame_id, sampled_at, p2, visible=visible))


def _remember_baseline(observer: object, features: _Features) -> None:
    features.values[(4, Side.P1)] = (0.80, 0.20)
    features.values[(4, Side.P2)] = (0.80, 0.20)
    observer.remember(_frame(4, 0.20, 4))  # type: ignore[attr-defined]


@pytest.mark.parametrize(("value", "want"), [
    (0.0, 0), (0.12, 0), (1 / 3, 1), (2 / 3, 2), (0.88, 3),
    (1.0, 3), (0.50, None), (0.13, None),
])
def test_remaining_meter_bucket_is_conservative(value: float, want: int | None) -> None:
    from sf6_music_trigger.outcome import remaining_meter_bucket
    assert remaining_meter_bucket(value) == want


@pytest.mark.parametrize("value", [True, "0.5", -0.01, 1.01, float("nan")])
def test_remaining_meter_bucket_rejects_invalid_values(value: object) -> None:
    from sf6_music_trigger.outcome import remaining_meter_bucket
    with pytest.raises((TypeError, ValueError)):
        remaining_meter_bucket(value)  # type: ignore[arg-type]


def test_outcome_decision_is_frozen_and_validates_fields() -> None:
    from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel
    item = OutcomeDecision("event", Side.P1, OutcomeLabel.HIT, 0.9, 1.5, ("health",))
    with pytest.raises(FrozenInstanceError):
        item.label = OutcomeLabel.BLOCK  # type: ignore[misc]
    with pytest.raises(ValueError):
        OutcomeDecision("", Side.P1, OutcomeLabel.HIT, 0.9, 1.5, ("health",))


def test_outcome_label_includes_filtered_tier() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel
    assert OutcomeLabel.FILTERED_TIER.value == "FILTERED_TIER"


def test_update_with_no_pending_does_not_extract_features() -> None:
    from sf6_music_trigger.outcome import OutcomeObserver
    features = _Features({})
    observer = OutcomeObserver(feature_extractor=features)
    assert observer.update(_frame(1, 0.1, 1), observations=_meters(1, 0.1)) == ()
    assert features.calls == []


def test_begin_uses_features_keyed_by_pre_spend_frame_id() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({
        (4, Side.P1): (0.80, 0.20), (4, Side.P2): (0.80, 0.20),
        (9, Side.P1): (0.60, 0.20), (9, Side.P2): (0.80, 0.20),
        (10, Side.P1): (0.60, 0.20), (11, Side.P1): (0.60, 0.20),
    })
    observer = OutcomeObserver(contact_delta=0.10, contact_samples=2, feature_extractor=features)
    observer.remember(_frame(4, 0.20, 4))
    observer.remember(_frame(9, 0.45, 9))
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    assert observer.update(_frame(10, 1.45, 10), observations=_meters(10, 1.45)) == ()
    decisions = observer.update(_frame(11, 1.50, 11), observations=_meters(11, 1.50))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.HIT,)


def test_begin_anchors_checkpoint_to_candidate_activation() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    observer = OutcomeObserver(
        checkpoint_seconds=3.0,
        hard_deadline_seconds=8.0,
        feature_extractor=features,
    )
    _remember_baseline(observer, features)
    features.values[(10, Side.P1)] = (0.80, 0.20)
    features.values[(11, Side.P1)] = (0.80, 0.20)
    event = _event(Side.P1, confirmed_at=1.6)
    observer.begin(event, SuperScope.ALL_SUPERS, activation_at=1.0)

    assert observer.update(
        _frame(10, 3.99, 10), observations=_meters(10, 3.99)
    ) == ()
    decisions = observer.update(
        _frame(11, 4.0, 11), observations=_meters(11, 4.0)
    )
    assert decisions[0].label is OutcomeLabel.UNKNOWN


def test_begin_rejects_activation_after_confirmation_atomically() -> None:
    from sf6_music_trigger.outcome import OutcomeObserver
    observer = OutcomeObserver()
    with pytest.raises(ValueError, match="activation_at"):
        observer.begin(
            _event(Side.P1, confirmed_at=1.0), SuperScope.ALL_SUPERS,
            activation_at=1.01,
        )
    assert observer.pending_sides == ()


def test_duplicate_event_rejects_invalid_explicit_activation_atomically() -> None:
    from sf6_music_trigger.outcome import OutcomeObserver
    observer = OutcomeObserver()
    event = _event(Side.P1)
    observer.begin(event, SuperScope.ALL_SUPERS, activation_at=event.confirmed_at)
    with pytest.raises(ValueError, match="activation_at"):
        observer.begin(event, SuperScope.ALL_SUPERS, activation_at=-0.01)
    assert observer.pending_sides == (Side.P1,)


def test_missing_or_evicted_baseline_never_resolves_whiff() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    observer = OutcomeObserver(feature_extractor=_Features({}))
    observer.begin(_event(Side.P1, pre_spend_frame_id=3), SuperScope.ALL_SUPERS,
                   activation_at=1.0)
    decisions = observer.update(_frame(20, 9.0, 20), observations=_meters(20, 9.0))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.UNKNOWN,)


def test_feature_ring_evicts_oldest_frame_after_240_entries() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    values: dict[tuple[int, Side], tuple[float, float]] = {}
    features = _Features(values)
    observer = OutcomeObserver(feature_extractor=features)
    for frame_id in range(1, 242):
        code = frame_id % 256
        values[(code, Side.P1)] = (0.8, 0.2)
        values[(code, Side.P2)] = (0.8, 0.2)
        observer.remember(_frame(frame_id, frame_id / 20, code))
    observer.begin(_event(Side.P1, confirmed_at=12.1, pre_spend_frame_id=1,
                          confirmed_frame_id=242), SuperScope.ALL_SUPERS,
                   activation_at=12.1)
    decisions = observer.update(_frame(300, 20.1, 20), observations=_meters(300, 20.1))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.UNKNOWN,)


def test_contact_cannot_finalize_during_first_four_tenths() -> None:
    from sf6_music_trigger.outcome import OutcomeObserver
    features = _Features({(4, Side.P1): (0.8, 0.2), (4, Side.P2): (0.8, 0.2),
                          (10, Side.P1): (0.6, 0.2), (11, Side.P1): (0.6, 0.2)})
    observer = OutcomeObserver(contact_delta=0.1, feature_extractor=features)
    observer.remember(_frame(4, 0.2, 4))
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    assert observer.update(_frame(10, 1.2, 10), observations=_meters(10, 1.2)) == ()
    assert observer.update(_frame(11, 1.35, 11), observations=_meters(11, 1.35)) == ()
    assert observer.pending_sides == (Side.P1,)


@pytest.mark.parametrize(("health", "result", "want"), [
    (0.60, 0.20, "HIT"), (0.80, 0.40, "BLOCK"),
])
def test_two_stable_contact_changes_after_suppression_resolve(
    health: float, result: float, want: str,
) -> None:
    from sf6_music_trigger.outcome import OutcomeObserver
    values = {(4, Side.P1): (0.8, 0.2), (4, Side.P2): (0.8, 0.2),
              (10, Side.P1): (health, result), (11, Side.P1): (health, result)}
    observer = OutcomeObserver(contact_delta=0.1, feature_extractor=_Features(values))
    observer.remember(_frame(4, 0.2, 4))
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    assert observer.update(_frame(10, 1.45, 10), observations=_meters(10, 1.45)) == ()
    decisions = observer.update(_frame(11, 1.5, 11), observations=_meters(11, 1.5))
    assert tuple(item.label.value for item in decisions) == (want,)


def test_hidden_hud_at_checkpoint_preserves_pending_until_restore() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({(4, Side.P1): (0.8, 0.2), (4, Side.P2): (0.8, 0.2),
                          (21, Side.P1): (0.8, 0.2)})
    observer = OutcomeObserver(feature_extractor=features)
    observer.remember(_frame(4, 0.2, 4))
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    assert observer.update(_frame(20, 4.0, 20), observations=_meters(20, 4.0, visible=False)) == ()
    assert observer.pending_sides == (Side.P1,)
    decisions = observer.update(_frame(21, 4.1, 21), observations=_meters(21, 4.1))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.UNKNOWN,)


def test_hidden_hud_at_hard_deadline_resolves_unknown() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    _remember_baseline(observer := OutcomeObserver(feature_extractor=features), features)
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    decisions = observer.update(_frame(20, 9.0, 20), observations=_meters(20, 9.0, visible=False))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.UNKNOWN,)


def test_five_stable_one_or_two_bar_samples_filter_sa1_sa2() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    _remember_baseline(observer := OutcomeObserver(feature_extractor=features), features)
    observer.begin(_event(Side.P1), SuperScope.SA3_CA_ONLY, activation_at=1.0)
    decisions = ()
    for offset in range(5):
        frame_id, time = 10 + offset, 1.45 + offset * 0.05
        features.values[(frame_id, Side.P1)] = (0.8, 0.2)
        decisions = observer.update(_frame(frame_id, time, frame_id),
                                    observations=_meters(frame_id, time, p1=1))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.FILTERED_TIER,)


def test_zero_bar_tier_allows_later_contact_resolution() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    _remember_baseline(observer := OutcomeObserver(contact_delta=0.1, feature_extractor=features), features)
    observer.begin(_event(Side.P1), SuperScope.SA3_CA_ONLY, activation_at=1.0)
    for offset in range(5):
        frame_id, time = 10 + offset, 1.45 + offset * 0.05
        features.values[(frame_id, Side.P1)] = (0.8, 0.2)
        assert observer.update(_frame(frame_id, time, frame_id),
                               observations=_meters(frame_id, time, p1=0)) == ()
    features.values[(20, Side.P1)] = (0.6, 0.2)
    features.values[(21, Side.P1)] = (0.6, 0.2)
    assert observer.update(_frame(20, 1.8, 20), observations=_meters(20, 1.8)) == ()
    decisions = observer.update(_frame(21, 1.85, 21), observations=_meters(21, 1.85))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.HIT,)


def test_all_supers_scope_never_filters_one_bar_event() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    _remember_baseline(observer := OutcomeObserver(feature_extractor=features), features)
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    for offset in range(5):
        frame_id, time = 10 + offset, 1.45 + offset * 0.05
        features.values[(frame_id, Side.P1)] = (0.8, 0.2)
        assert observer.update(_frame(frame_id, time, frame_id),
                               observations=_meters(frame_id, time, p1=1)) == ()
    features.values[(20, Side.P1)] = (0.8, 0.2)
    decisions = observer.update(_frame(20, 4.0, 20), observations=_meters(20, 4.0, p1=1))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.UNKNOWN,)


def test_three_remaining_bars_resolve_unknown_not_whiff() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    _remember_baseline(observer := OutcomeObserver(feature_extractor=features), features)
    observer.begin(_event(Side.P1), SuperScope.SA3_CA_ONLY, activation_at=1.0)
    decisions = ()
    for offset in range(5):
        frame_id, time = 10 + offset, 8.8 + offset * 0.05
        features.values[(frame_id, Side.P1)] = (0.8, 0.2)
        decisions = observer.update(_frame(frame_id, time, frame_id),
                                    observations=_meters(frame_id, time, p1=3))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.UNKNOWN,)


def test_reset_returns_one_decision_per_pending_side_in_side_order() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    _remember_baseline(observer := OutcomeObserver(feature_extractor=features), features)
    observer.begin(_event(Side.P2), SuperScope.ALL_SUPERS, activation_at=1.0)
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    decisions = observer.reset("capture lost")
    assert [(item.side, item.label) for item in decisions] == [
        (Side.P1, OutcomeLabel.RESET), (Side.P2, OutcomeLabel.RESET),
    ]
    assert observer.pending_sides == ()


def test_duplicate_event_id_does_not_replace_pending_or_restart_resolution() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    _remember_baseline(observer := OutcomeObserver(feature_extractor=features), features)
    first = _event(Side.P1, event_id="event-p1-1")
    observer.begin(first, SuperScope.ALL_SUPERS, activation_at=first.confirmed_at)
    observer.begin(first, SuperScope.SA3_CA_ONLY, activation_at=first.confirmed_at)
    features.values[(20, Side.P1)] = (0.8, 0.2)
    decisions = observer.update(_frame(20, 4.0, 20), observations=_meters(20, 4.0, p1=1))
    assert tuple(item.event_id for item in decisions) == ("event-p1-1",)
    assert tuple(item.label for item in decisions) == (OutcomeLabel.UNKNOWN,)


def test_same_event_id_is_independent_per_side_and_seen_history_is_bounded() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver

    observer = OutcomeObserver(event_history_capacity=3)
    p1 = _event(Side.P1, event_id="shared-event")
    p2 = _event(Side.P2, event_id="shared-event")
    observer.begin(p1, SuperScope.ALL_SUPERS, activation_at=1.0)
    observer.begin(p2, SuperScope.ALL_SUPERS, activation_at=1.0)
    assert observer.pending_sides == (Side.P1, Side.P2)
    assert [(item.side, item.event_id) for item in observer.reset("round end")] == [
        (Side.P1, "shared-event"),
        (Side.P2, "shared-event"),
    ]

    for index in range(6):
        event = _event(Side.P1, event_id=f"event-{index}")
        observer.begin(event, SuperScope.ALL_SUPERS, activation_at=1.0)
        assert observer._finish(
            Side.P1,
            OutcomeLabel.UNKNOWN,
            1.1,
            0.0,
            ("resolution_deadline_elapsed",),
        ).event_id == event.event_id
    assert len(observer._seen_event_keys) == 3
    assert (Side.P1, "event-0") not in observer._seen_event_keys


def test_p1_and_p2_pending_resolve_independently_in_side_order() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    _remember_baseline(observer := OutcomeObserver(contact_delta=0.1, feature_extractor=features), features)
    observer.begin(_event(Side.P2), SuperScope.SA3_CA_ONLY, activation_at=1.0)
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    for offset in range(4):
        frame_id, time = 10 + offset, 1.45 + offset * 0.05
        features.values[(frame_id, Side.P1)] = (
            (0.6, 0.2) if offset == 3 else (0.8, 0.2)
        )
        features.values[(frame_id, Side.P2)] = (0.8, 0.2)
        assert observer.update(_frame(frame_id, time, frame_id),
                               observations=_meters(frame_id, time, p2=1)) == ()
    frame_id, time = 14, 1.65
    features.values[(frame_id, Side.P1)] = (0.6, 0.2)
    features.values[(frame_id, Side.P2)] = (0.8, 0.2)
    decisions = observer.update(_frame(frame_id, time, frame_id),
                                observations=_meters(frame_id, time, p2=1))
    assert [(item.side, item.label) for item in decisions] == [
        (Side.P1, OutcomeLabel.HIT), (Side.P2, OutcomeLabel.FILTERED_TIER),
    ]


def test_backward_or_duplicate_frames_cannot_advance_contact_streaks() -> None:
    from sf6_music_trigger.outcome import OutcomeLabel, OutcomeObserver
    features = _Features({})
    _remember_baseline(observer := OutcomeObserver(contact_delta=0.1, feature_extractor=features), features)
    observer.begin(_event(Side.P1), SuperScope.ALL_SUPERS, activation_at=1.0)
    for frame_id, time in ((10, 1.45), (10, 1.45), (9, 1.46)):
        features.values[(frame_id, Side.P1)] = (0.6, 0.2)
        assert observer.update(_frame(frame_id, time, frame_id),
                               observations=_meters(frame_id, time)) == ()
    features.values[(11, Side.P1)] = (0.6, 0.2)
    decisions = observer.update(_frame(11, 1.5, 11), observations=_meters(11, 1.5))
    assert tuple(item.label for item in decisions) == (OutcomeLabel.HIT,)


def test_observation_tuple_must_match_frame_and_side_order() -> None:
    from sf6_music_trigger.outcome import OutcomeObserver
    observer = OutcomeObserver()
    frame = _frame(10, 1.5, 10)
    with pytest.raises(ValueError):
        observer.update(frame, observations=tuple(reversed(_meters(10, 1.5))))
    with pytest.raises(ValueError):
        observer.update(frame, observations=_meters(11, 1.5))


def test_defaults_accept_exact_rational_timing_without_float_narrowing() -> None:
    from sf6_music_trigger.outcome import OutcomeObserver
    observer = OutcomeObserver(suppression_seconds=Fraction(2, 5),
                               checkpoint_seconds=Fraction(3, 1),
                               hard_deadline_seconds=Fraction(8, 1))
    assert observer.pending_sides == ()
