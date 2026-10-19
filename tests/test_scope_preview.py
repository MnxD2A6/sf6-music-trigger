from dataclasses import replace

import pytest

from sf6_music_trigger.audio import AudioKind, AudioScheduler
from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.domain import GaugeTrackerUpdate, Side
from test_engine import _build_harness, _candidate, _cancellation, _drive, _event, _TrackerSpy


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("before,after", [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (3, 0)])
@pytest.mark.parametrize("scope", list(SuperScope))
def test_preview_requires_three_to_zero_in_restricted_scope(side, before, after, scope):
    candidate = replace(_candidate(side), meter_before=before, meter_after=after)
    actions = AudioScheduler(side).on_candidate(
        candidate, SideAudioConfig(main_track="main.wav", super_scope=scope)
    )
    expected = scope is SuperScope.ALL_SUPERS or (before, after) == (3, 0)
    assert [a.kind for a in actions] == ([AudioKind.STOP, AudioKind.PLAY] if expected else [])


@pytest.mark.parametrize("terminal", ["confirm", "cancel", "reset"])
@pytest.mark.parametrize("side", list(Side))
def test_silent_lower_tier_episode_does_not_stop_existing_music(side, terminal):
    accepted = _event(side, event_id="previous-super", levels_spent=3, confirmed_at=0.5)
    candidate = _candidate(side, levels_suspected=1)
    ending = (_event(side, levels_spent=1, candidate_id=candidate.candidate_id, confirmed_at=1.4)
              if terminal == "confirm" else _cancellation(side))
    tracker = _TrackerSpy([GaugeTrackerUpdate(accepted), GaugeTrackerUpdate(candidate),
                           GaugeTrackerUpdate(ending)])
    harness = _build_harness(**{f"{side.value.lower()}_tracker": tracker})
    rows = [(0.5, "level_3", "level_3"), (1.0, "level_1", "level_1")]
    if terminal != "reset":
        rows.append((1.5, "level_0", "level_0"))
    _drive(harness, rows)
    if terminal == "reset":
        harness.engine._reset_trackers("test reset")
    assert [[a.kind for a in batch] for batch in harness.audio.batches] == [
        [AudioKind.STOP, AudioKind.PLAY]
    ]


@pytest.mark.parametrize("side", list(Side))
def test_partial_candidate_confirmed_as_three_starts_once(side):
    candidate = replace(_candidate(side), meter_after=1)
    confirmed = _event(side, levels_spent=3, candidate_id=candidate.candidate_id, confirmed_at=1.4)
    tracker = _TrackerSpy([GaugeTrackerUpdate(candidate), GaugeTrackerUpdate(confirmed),
                           GaugeTrackerUpdate(confirmed)])
    harness = _build_harness(**{f"{side.value.lower()}_tracker": tracker})
    _drive(harness, [(1.0, "level_1", "level_1"), (1.4, "level_0", "level_0"),
                     (1.5, "level_0", "level_0")])
    assert [[a.kind for a in batch] for batch in harness.audio.batches] == [
        [AudioKind.STOP, AudioKind.PLAY]
    ]
    assert harness.diagnostic_records[0].audio_actions == ()
