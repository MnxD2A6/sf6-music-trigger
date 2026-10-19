def test_audio_module_contract_is_absent_then_added():
    import importlib

    module = importlib.import_module("sf6_music_trigger.audio")
    assert module.__name__ == "sf6_music_trigger.audio"


from dataclasses import FrozenInstanceError, is_dataclass
from fractions import Fraction
import inspect
import logging
import math
from numbers import Real
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import threading
from threading import Event, Thread
from typing import get_type_hints

import pytest

from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.domain import (
    CandidateCancelReason,
    CancelledSuperCandidate,
    ConfirmedSuperEvent,
    Side,
    SuperCandidateEvent,
)


def _event(*, side=Side.P1, event_id="evt-1", levels_spent=None):
    meter_before = levels_spent if levels_spent is not None else None
    meter_after = 0 if levels_spent is not None else None
    return ConfirmedSuperEvent(
        event_id=event_id,
        side=side,
        confirmed_at=10.0,
        pre_spend_frame_id=100,
        confirmed_frame_id=101,
        profile_id="default",
        meter_before=meter_before,
        meter_after=meter_after,
        levels_spent=levels_spent,
    )


def _candidate(*, side=Side.P1, candidate_id="candidate-1"):
    return SuperCandidateEvent(
        candidate_id=candidate_id,
        side=side,
        suspected_at=9.0,
        pre_spend_frame_id=100,
        candidate_frame_id=101,
        profile_id="default",
        meter_before=3,
        meter_after=0,
        evidence=("stable_downward_level",),
    )


def _cancellation(*, side=Side.P1, candidate_id="candidate-1"):
    return CancelledSuperCandidate(
        candidate_id=candidate_id,
        side=side,
        decided_at=9.5,
        reason=CandidateCancelReason.CONFIRMATION_TIMEOUT,
    )


def test_kind_and_issue_enums_have_exact_public_values():
    from sf6_music_trigger.audio import AudioIssue, AudioKind

    assert issubclass(AudioKind, str)
    assert issubclass(AudioIssue, str)
    assert all(isinstance(item, str) for item in (*AudioKind, *AudioIssue))
    assert [(item.name, item.value) for item in AudioKind] == [
        ("STOP", "stop"),
        ("PLAY", "play"),
        ("FADE", "fade"),
    ]
    assert [(item.name, item.value) for item in AudioIssue] == [
        ("BACKEND_FAILURE", "backend_failure"),
        ("ACTIVE_PORT_CONFLICT", "active_port_conflict"),
        ("CLOSE_INCOMPLETE", "close_incomplete"),
        ("TRACK_REJECTED", "track_rejected"),
    ]


def test_action_is_frozen_and_as_tuple_omits_typed_side():
    from sf6_music_trigger.audio import AudioAction, AudioKind

    assert is_dataclass(AudioAction)
    assert AudioAction.__dataclass_params__.frozen is True
    action = AudioAction(AudioKind.PLAY, Side.P1, "audio/main.mp3", 0.18)
    assert action.as_tuple() == ("play", "audio/main.mp3", 0.18)
    assert action.side is Side.P1
    with pytest.raises(FrozenInstanceError):
        action.volume = 0.5


@pytest.mark.parametrize("kind", ["stop", 1, None])
def test_action_rejects_non_enum_kind(kind):
    from sf6_music_trigger.audio import AudioAction

    with pytest.raises(TypeError):
        AudioAction(kind, Side.P1)


@pytest.mark.parametrize("side", ["P1", 1, None])
def test_action_rejects_non_side_value(side):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    with pytest.raises(TypeError):
        AudioAction(AudioKind.STOP, side)


@pytest.mark.parametrize(
    ("track", "volume"),
    [("track.bin", None), (None, 0.2), ("track.bin", 0.2)],
)
def test_action_stop_rejects_either_payload(track, volume):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    with pytest.raises(ValueError):
        AudioAction(AudioKind.STOP, Side.P1, track, volume)


@pytest.mark.parametrize("track", [None, 7, ""])
def test_action_play_rejects_non_string_or_empty_track(track):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    expected = TypeError if track is None or isinstance(track, int) else ValueError
    with pytest.raises(expected):
        AudioAction(AudioKind.PLAY, Side.P1, track, 0.2)


@pytest.mark.parametrize("volume", [True, False, "0.2", object(), None])
def test_action_play_rejects_bool_and_non_real_volume(volume):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    with pytest.raises(TypeError):
        AudioAction(AudioKind.PLAY, Side.P1, "track.bin", volume)


@pytest.mark.parametrize("volume", [math.nan, math.inf, -math.inf, -0.01, 5.01])
def test_action_play_rejects_nonfinite_or_out_of_range_volume(volume):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    with pytest.raises(ValueError):
        AudioAction(AudioKind.PLAY, Side.P1, "track.bin", volume)


@pytest.mark.parametrize("volume", [0, 1, 0.0, 1.0, Fraction(1, 4)])
def test_action_play_accepts_closed_volume_boundaries_and_rationals(volume):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    action = AudioAction(AudioKind.PLAY, Side.P1, "track.bin", volume)
    assert action.volume == float(volume)
    assert type(action.volume) is float


def test_action_play_volume_handles_huge_rationals_without_overflow():
    from sf6_music_trigger.audio import AudioAction, AudioKind

    with pytest.raises(ValueError):
        AudioAction(AudioKind.PLAY, Side.P1, "track.bin", 10**10000)
    with pytest.raises(ValueError):
        AudioAction(AudioKind.PLAY, Side.P1, "track.bin", -(10**10000))
    tiny = AudioAction(
        AudioKind.PLAY,
        Side.P1,
        "track.bin",
        Fraction(1, 10**10000),
    )
    assert tiny.volume == float(Fraction(1, 10**10000))


@pytest.mark.parametrize("fade_ms", [0, 1, 1500, 10**100])
def test_action_fade_accepts_actual_nonnegative_integer_milliseconds(fade_ms):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    action = AudioAction(AudioKind.FADE, Side.P1, fade_ms=fade_ms)
    assert action.fade_ms == fade_ms
    assert action.track is None
    assert action.volume is None


@pytest.mark.parametrize("fade_ms", [True, False, 1.5, "1500", None])
def test_action_fade_rejects_non_integer_milliseconds(fade_ms):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    with pytest.raises(TypeError):
        AudioAction(AudioKind.FADE, Side.P1, fade_ms=fade_ms)


def test_action_fade_rejects_negative_or_play_payload():
    from sf6_music_trigger.audio import AudioAction, AudioKind

    with pytest.raises(ValueError):
        AudioAction(AudioKind.FADE, Side.P1, fade_ms=-1)
    with pytest.raises(ValueError):
        AudioAction(AudioKind.FADE, Side.P1, "audio/main.mp3", 0.18, 1500)


def test_action_repr_redacts_track_but_preserves_value_access():
    from sf6_music_trigger.audio import AudioAction, AudioKind

    secret = r"C:\Users\private\selected-track.mp3"
    action = AudioAction(AudioKind.PLAY, Side.P2, secret, 0.4)
    assert action.track == secret
    assert secret not in repr(action)
    assert "selected-track.mp3" not in repr(action)


def test_scheduler_constructor_requires_actual_side():
    from sf6_music_trigger.audio import AudioScheduler

    for invalid in ("P1", 1, None):
        with pytest.raises(TypeError):
            AudioScheduler(invalid)


def test_every_enabled_candidate_starts_main_track_for_both_scopes() -> None:
    from sf6_music_trigger.audio import AudioKind, AudioScheduler

    scheduler = AudioScheduler(Side.P1)
    for scope in (SuperScope.ALL_SUPERS, SuperScope.SA3_CA_ONLY):
        actions = scheduler.on_candidate(
            _candidate(),
            SideAudioConfig(
                enabled=True,
                main_track="audio/main.mp3",
                volume=0.18,
                super_scope=scope,
            ),
        )
        assert [item.kind for item in actions] == [AudioKind.STOP, AudioKind.PLAY]


def test_candidate_returns_exact_typed_stop_then_play_batch() -> None:
    from sf6_music_trigger.audio import AudioScheduler

    actions = AudioScheduler(Side.P1).on_candidate(
        _candidate(), SideAudioConfig(main_track="audio/main.mp3", volume=0.18)
    )
    assert tuple(action.as_tuple() for action in actions) == (
        ("stop", None, None),
        ("play", "audio/main.mp3", 0.18),
    )
    assert tuple(action.side for action in actions) == (Side.P1, Side.P1)


def test_confirmation_keeps_started_main_track_for_all_tiers_and_scopes() -> None:
    from sf6_music_trigger.audio import AudioScheduler

    for scope in (SuperScope.ALL_SUPERS, SuperScope.SA3_CA_ONLY):
        for levels_spent in (1, 2, 3):
            actions = AudioScheduler(Side.P1).on_confirmation(
                _event(levels_spent=levels_spent),
                SideAudioConfig(
                    main_track="audio/main.mp3", super_scope=scope
                ),
                playback_started=True,
            )
            if levels_spent == 3:
                assert actions == ()
            else:
                assert len(actions) == 1


def test_started_low_tier_confirmation_fades_without_whiff_sound() -> None:
    from sf6_music_trigger.audio import AudioKind, AudioScheduler

    actions = AudioScheduler(Side.P1).on_confirmation(
        _event(levels_spent=1),
        SideAudioConfig(
            enabled=True,
            main_track="audio/main.mp3",
            play_whiff_sound=True,
            whiff_track="audio/whiff.mp3",
            super_scope=SuperScope.SA3_CA_ONLY,
        ),
        playback_started=True,
    )
    assert len(actions) == 1
    assert actions[0].kind is AudioKind.FADE


def test_unstarted_accepted_confirmation_recovers_main_track_once() -> None:
    from sf6_music_trigger.audio import AudioKind, AudioScheduler

    scheduler = AudioScheduler(Side.P2)
    config = SideAudioConfig(
        main_track="audio/p2.mp3", super_scope=SuperScope.SA3_CA_ONLY
    )
    actions = scheduler.on_confirmation(
        _event(side=Side.P2, levels_spent=3), config, playback_started=False
    )
    assert [action.kind for action in actions] == [AudioKind.STOP, AudioKind.PLAY]
    assert scheduler.on_confirmation(
        _event(side=Side.P2, levels_spent=3), config, playback_started=True
    ) == ()


def test_unstarted_filtered_confirmation_does_not_start_audio() -> None:
    from sf6_music_trigger.audio import AudioScheduler

    actions = AudioScheduler(Side.P1).on_confirmation(
        _event(levels_spent=2),
        SideAudioConfig(
            main_track="audio/main.mp3", super_scope=SuperScope.SA3_CA_ONLY
        ),
        playback_started=False,
    )
    assert actions == ()


def test_candidate_cancellation_always_fades_and_never_plays_whiff() -> None:
    from sf6_music_trigger.audio import AudioKind, AudioScheduler

    actions = AudioScheduler(Side.P1).on_cancellation(
        _cancellation(),
        SideAudioConfig(
            enabled=True,
            main_track="audio/main.mp3",
            play_whiff_sound=True,
            whiff_track="audio/whiff.mp3",
        ),
    )
    assert [item.kind for item in actions] == [AudioKind.FADE]


def test_scheduler_lifecycle_validation_precedes_disabled_short_circuit() -> None:
    from sf6_music_trigger.audio import AudioScheduler

    scheduler = AudioScheduler(Side.P1)
    disabled = SideAudioConfig(enabled=False, main_track="")
    with pytest.raises(TypeError):
        scheduler.on_candidate(object(), disabled)
    with pytest.raises(TypeError):
        scheduler.on_confirmation(object(), disabled, playback_started=False)
    with pytest.raises(TypeError):
        scheduler.on_cancellation(object(), disabled)
    with pytest.raises(TypeError):
        scheduler.on_candidate(_candidate(), object())
    with pytest.raises(TypeError):
        scheduler.on_confirmation(_event(), object(), playback_started=False)
    with pytest.raises(TypeError):
        scheduler.on_cancellation(_cancellation(), object())
    with pytest.raises(TypeError):
        scheduler.on_confirmation(_event(), disabled, playback_started=1)
    with pytest.raises(ValueError):
        scheduler.on_candidate(_candidate(side=Side.P2), disabled)
    with pytest.raises(ValueError):
        scheduler.on_confirmation(_event(side=Side.P2), disabled, playback_started=False)
    with pytest.raises(ValueError):
        scheduler.on_cancellation(_cancellation(side=Side.P2), disabled)


def test_scheduler_disabled_lifecycle_events_return_empty_tuples() -> None:
    from sf6_music_trigger.audio import AudioScheduler

    scheduler = AudioScheduler(Side.P1)
    config = SideAudioConfig(enabled=False, main_track="")
    assert scheduler.on_candidate(_candidate(), config) == ()
    assert scheduler.on_confirmation(_event(levels_spent=3), config, playback_started=False) == ()
    assert scheduler.on_cancellation(_cancellation(), config) == ()


def test_scheduler_lifecycle_actions_remain_side_isolated_and_stateless() -> None:
    from sf6_music_trigger.audio import AudioScheduler

    scheduler = AudioScheduler(Side.P2)
    candidate = _candidate(side=Side.P2)
    config = SideAudioConfig(main_track="audio/p2.mp3", volume=0.2)
    first = scheduler.on_candidate(candidate, config)
    second = scheduler.on_candidate(candidate, config)
    assert first == second
    assert first is not second
    assert all(left is not right for left, right in zip(first, second, strict=True))
    assert vars(scheduler) == {"side": Side.P2}


def test_whiff_scheduler_defaults_to_one_fade_action() -> None:
    from sf6_music_trigger.audio import AudioKind, AudioScheduler
    from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel

    decision = OutcomeDecision(
        "evt-whiff", Side.P1, OutcomeLabel.WHIFF, 0.8, 16.0,
        ("contact_window_elapsed",),
    )
    actions = AudioScheduler(Side.P1).on_outcome(decision, SideAudioConfig())
    assert len(actions) == 1
    assert actions[0].kind is AudioKind.FADE
    assert actions[0].side is Side.P1
    assert actions[0].fade_ms == 1500


def test_whiff_sound_mode_replaces_fade_with_stop_play() -> None:
    from sf6_music_trigger.audio import AudioKind, AudioScheduler
    from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel

    decision = OutcomeDecision(
        "evt-whiff", Side.P2, OutcomeLabel.WHIFF, 0.8, 16.0,
        ("contact_window_elapsed",),
    )
    config = SideAudioConfig(
        whiff_track="audio/whiff.wav",
        whiff_volume=0.37,
        play_whiff_sound=True,
    )
    actions = AudioScheduler(Side.P2).on_outcome(decision, config)
    assert [action.kind for action in actions] == [AudioKind.STOP, AudioKind.PLAY]
    assert all(action.kind is not AudioKind.FADE for action in actions)
    assert actions[1].track == "audio/whiff.wav"
    assert actions[1].volume == 0.37


@pytest.mark.parametrize("label", ["HIT"])
def test_hit_and_unknown_leave_immediate_main_playback_untouched(label: str) -> None:
    from sf6_music_trigger.audio import AudioScheduler
    from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel

    scheduler = AudioScheduler(Side.P1)
    decision = OutcomeDecision(
        "evt", Side.P1, OutcomeLabel[label], 0.8, 2.0, ("contact",)
    )
    assert scheduler.on_outcome(decision, SideAudioConfig()) == ()


@pytest.mark.parametrize(
    "label", ["UNKNOWN", "BLOCK", "NON_HIT", "FILTERED_TIER", "RESET"],
)
def test_safe_terminal_results_fade_and_never_play_whiff(label: str) -> None:
    from sf6_music_trigger.audio import AudioAction, AudioKind, AudioScheduler
    from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel

    config = SideAudioConfig(
        enabled=True,
        play_whiff_sound=True,
        whiff_track="audio/whiff.wav",
        whiff_fade_ms=1700,
    )
    decision = OutcomeDecision(
        "evt", Side.P1, OutcomeLabel[label], 0.0, 2.0, ("safe_terminal",)
    )
    expected = (
        (
            AudioAction(AudioKind.STOP, Side.P1),
            AudioAction(AudioKind.PLAY, Side.P1, "audio/whiff.wav", 0.18),
        )
        if label in {"BLOCK", "NON_HIT"}
        else (AudioAction(AudioKind.FADE, Side.P1, fade_ms=1700),)
    )
    assert AudioScheduler(Side.P1).on_outcome(decision, config) == expected


@pytest.mark.parametrize(
    "operation",
    [
        "initialize",
        "execute",
        "stop",
        "load",
        "set_volume",
        "play",
        "fade",
        "preview",
        "preview_stop",
        "close_stop",
    ],
)
def test_error_envelope_accepts_each_permitted_operation(operation):
    from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError

    error = AudioPlaybackError(operation, Side.P1, AudioIssue.BACKEND_FAILURE)
    assert (error.operation, error.side, error.issue) == (
        operation,
        Side.P1,
        AudioIssue.BACKEND_FAILURE,
    )


@pytest.mark.parametrize("operation", ["", "decode", 1, None])
def test_error_envelope_rejects_invalid_operation(operation):
    from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError

    expected = TypeError if not isinstance(operation, str) else ValueError
    with pytest.raises(expected):
        AudioPlaybackError(operation, Side.P1, AudioIssue.BACKEND_FAILURE)


def test_error_envelope_rejects_str_subclass_without_exposing_custom_text(
    capsys,
    caplog,
):
    from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError

    secret = r"C:\Users\private\evil-operation-secret.mp3"

    class SecretOperation(str):
        def __str__(self):
            return secret

        def __repr__(self):
            return secret

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TypeError) as caught:
            AudioPlaybackError(
                SecretOperation("load"),
                Side.P1,
                AudioIssue.BACKEND_FAILURE,
            )

    captured = capsys.readouterr()
    surfaces = (
        str(caught.value),
        repr(caught.value),
        repr(caught.value.args),
        repr(caught.value.__cause__),
        repr(caught.value.__context__),
        captured.out,
        captured.err,
        caplog.text,
    )
    assert all(secret not in surface for surface in surfaces)


@pytest.mark.parametrize("side", ["P1", 1, object()])
def test_error_envelope_rejects_invalid_side(side):
    from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError

    with pytest.raises(TypeError):
        AudioPlaybackError("play", side, AudioIssue.BACKEND_FAILURE)


@pytest.mark.parametrize("issue", ["backend_failure", 1, None])
def test_error_envelope_rejects_invalid_issue(issue):
    from sf6_music_trigger.audio import AudioPlaybackError

    with pytest.raises(TypeError):
        AudioPlaybackError("play", Side.P1, issue)


def test_error_envelope_has_only_typed_safe_public_fields():
    from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError

    error = AudioPlaybackError("load", None, AudioIssue.TRACK_REJECTED)
    assert isinstance(error, RuntimeError)
    assert vars(error) == {
        "operation": "load",
        "side": None,
        "issue": AudioIssue.TRACK_REJECTED,
        "invalidated_sides": frozenset(),
    }
    assert not hasattr(error, "track")
    assert "load" in str(error)
    assert "track_rejected" in str(error)
    assert "side=none" in str(error)


def test_error_envelope_message_contains_only_safe_typed_context():
    from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError

    error = AudioPlaybackError("play", Side.P2, AudioIssue.BACKEND_FAILURE)
    assert str(error) == (
        "audio playback failed: operation=play; side=P2; issue=backend_failure"
    )


def test_error_envelope_never_exposes_private_path_or_raw_exception_text():
    from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError

    secret_drive = r"C:\Users\private\selected-track.mp3"
    secret_unc = r"\\private-server\share\selected-track.mp3"
    raw_text = "backend rejected " + secret_drive + " " + secret_unc
    error = AudioPlaybackError("load", Side.P2, AudioIssue.TRACK_REJECTED)
    surfaces = (
        str(error),
        repr(error),
        repr(vars(error)),
        repr(error.args),
        repr(error.__cause__),
        repr(error.__context__),
    )
    for surface in surfaces:
        assert secret_drive not in surface
        assert secret_unc not in surface
        assert raw_text not in surface
    assert error.__cause__ is None
    assert error.__context__ is None


_RAW_BACKEND_SECRET = r"C:\Users\private\backend-selected-track.mp3"
_SELECTED_DRIVE_SECRET = r"Q:\Private Audio\selected-execute-secret.bin"
_SELECTED_UNC_SECRET = (
    r"\\private-audio-server\share\selected-preview-secret.bin"
)


def _assert_safe_audio_error(error, operation, side, issue, *extra_secrets):
    assert (error.operation, error.side, error.issue) == (operation, side, issue)
    surfaces = (
        str(error),
        repr(error),
        repr(vars(error)),
        repr(error.args),
        repr(error.__cause__),
        repr(error.__context__),
    )
    secrets = (_RAW_BACKEND_SECRET, *extra_secrets)
    assert all(secret not in surface for secret in secrets for surface in surfaces)
    assert all("raw backend failure" not in surface for surface in surfaces)
    assert error.__cause__ is None
    assert error.__context__ is None


def _assert_no_direct_track_copy(port, *tracks):
    direct_strings = {
        value for value in vars(port).values() if isinstance(value, str)
    }
    for track in tracks:
        assert track not in direct_strings
        assert PureWindowsPath(track).name not in direct_strings


class _FakeSound:
    def __init__(self, path, timeline):
        self.path = path
        self.volume = None
        self.calls = []
        self.set_volume_failures_remaining = 0
        self._timeline = timeline

    def set_volume(self, volume):
        self.calls.append(("set_volume", volume))
        self._timeline.append(("sound.set_volume", self.path, volume))
        if self.set_volume_failures_remaining:
            self.set_volume_failures_remaining -= 1
            raise RuntimeError("raw backend failure " + _RAW_BACKEND_SECRET)
        self.volume = volume

    def play(self):
        self.calls.append(("play",))
        self._timeline.append(("sound.play", self.path))
        raise AssertionError("automatic Sound.play allocation is forbidden")


class _FakeChannel:
    def __init__(self, channel_id, timeline):
        self.channel_id = channel_id
        self.calls = []
        self.playing = False
        self.stop_failures_remaining = 0
        self.play_failures_remaining = 0
        self.play_entered = None
        self.allow_play = None
        self._timeline = timeline

    @property
    def stop_calls(self):
        return sum(call[0] == "stop" for call in self.calls)

    def stop(self):
        self.calls.append(("stop",))
        self._timeline.append(("channel.stop", self.channel_id))
        if self.stop_failures_remaining:
            self.stop_failures_remaining -= 1
            raise RuntimeError("raw backend failure " + _RAW_BACKEND_SECRET)
        self.playing = False

    def play(self, sound):
        self.calls.append(("play", sound))
        self._timeline.append(("channel.play", self.channel_id, sound.path))
        if self.play_entered is not None:
            self.play_entered.set()
        if self.allow_play is not None and not self.allow_play.wait(1.0):
            raise RuntimeError("raw backend failure " + _RAW_BACKEND_SECRET)
        if self.play_failures_remaining:
            self.play_failures_remaining -= 1
            raise RuntimeError("raw backend failure " + _RAW_BACKEND_SECRET)
        self.playing = True

    def fadeout(self, milliseconds):
        self.calls.append(("fadeout", milliseconds))
        self._timeline.append(("channel.fadeout", self.channel_id, milliseconds))
        self.playing = False


class _FakeMixer:
    def __init__(self, timeline):
        self.calls = []
        self.initialized = True
        self.num_channels = 3
        self.channels = {}
        self.sounds = []
        self.sound_paths = []
        self.keep_channel_count_on_set = False
        self._failures = {}
        self._timeline = timeline

    def fail_next(self, stage):
        self._failures[stage] = self._failures.get(stage, 0) + 1

    def clear_failures(self):
        self._failures.clear()
        self.keep_channel_count_on_set = False

    def _consume_failure(self, stage):
        remaining = self._failures.get(stage, 0)
        if not remaining:
            return
        if remaining == 1:
            self._failures.pop(stage)
        else:
            self._failures[stage] = remaining - 1
        raise RuntimeError("raw backend failure " + _RAW_BACKEND_SECRET)

    def _take_failure(self, stage):
        remaining = self._failures.get(stage, 0)
        if not remaining:
            return False
        if remaining == 1:
            self._failures.pop(stage)
        else:
            self._failures[stage] = remaining - 1
        return True

    def get_init(self):
        self.calls.append(("get_init",))
        self._timeline.append(("mixer.get_init",))
        self._consume_failure("get_init")
        return self.initialized

    def init(self):
        self.calls.append(("init",))
        self._timeline.append(("mixer.init",))
        self._consume_failure("init")
        self.initialized = True

    def get_num_channels(self):
        self.calls.append(("get_num_channels",))
        self._timeline.append(("mixer.get_num_channels",))
        reads = sum(call == ("get_num_channels",) for call in self.calls)
        if reads >= 2:
            self._consume_failure("get_num_channels_after_resize")
        self._consume_failure("get_num_channels")
        return self.num_channels

    def set_num_channels(self, count):
        self.calls.append(("set_num_channels", count))
        self._timeline.append(("mixer.set_num_channels", count))
        self._consume_failure("set_num_channels")
        if not self.keep_channel_count_on_set:
            self.num_channels = count

    def Channel(self, channel_id):
        self.calls.append(("Channel", channel_id))
        self._timeline.append(("mixer.Channel", channel_id))
        self._consume_failure(f"channel_{channel_id}")
        channel = _FakeChannel(channel_id, self._timeline)
        self.channels[channel_id] = channel
        return channel

    def Sound(self, path):
        self.calls.append(("Sound", path))
        self._timeline.append(("mixer.Sound", path))
        self._consume_failure("sound")
        sound = _FakeSound(path, self._timeline)
        if self._take_failure("set_volume"):
            sound.set_volume_failures_remaining = 1
        self.sounds.append(sound)
        self.sound_paths.append(path)
        return sound

    def find_channel(self, *args, **kwargs):
        self.calls.append(("find_channel", args, kwargs))
        self._timeline.append(("mixer.find_channel",))
        raise AssertionError("automatic channel allocation is forbidden")

    def stop(self):
        self.calls.append(("stop",))
        self._timeline.append(("mixer.stop",))
        raise AssertionError("global mixer stop is forbidden")

    def quit(self):
        self.calls.append(("quit",))
        self._timeline.append(("mixer.quit",))
        raise AssertionError("global mixer quit is forbidden")


class _FakePygame:
    def __init__(self, mixer):
        self._mixer = mixer
        self.quit_calls = 0

    @property
    def mixer(self):
        self._mixer._consume_failure("mixer")
        return self._mixer

    def quit(self):
        self.quit_calls += 1
        self._mixer._timeline.append(("pygame.quit",))
        raise AssertionError("global pygame quit is forbidden")


class _FakeLoader:
    def __init__(self, module, mixer):
        self.calls = 0
        self.module = module
        self.entered = None
        self.allow = None
        self._mixer = mixer

    def __call__(self):
        self.calls += 1
        self._mixer._timeline.append(("loader",))
        if self.entered is not None:
            self.entered.set()
        if self.allow is not None and not self.allow.wait(1.0):
            raise RuntimeError("raw backend failure " + _RAW_BACKEND_SECRET)
        self._mixer._consume_failure("loader")
        return self.module


class _FakeAudioProbe:
    def __init__(self, root):
        self.root = root
        self.root.mkdir()
        self.timeline = []
        self.mixer = _FakeMixer(self.timeline)
        self.loader = _FakeLoader(_FakePygame(self.mixer), self.mixer)
        self._port = None
        self.ports = []

    @property
    def port(self):
        if self._port is None:
            self._port = self.new_port()
        return self._port

    def new_port(self, root=None):
        from sf6_music_trigger.audio import PygameAudioPort

        port = PygameAudioPort(
            self.root if root is None else root,
            pygame_loader=self.loader,
        )
        self.ports.append(port)
        return port

    def fail_next(self, stage):
        self.mixer.fail_next(stage)

    def reset_calls(self):
        self.loader.calls = 0
        self.mixer.calls.clear()
        self.timeline.clear()
        for channel in self.mixer.channels.values():
            channel.calls.clear()
        for sound in self.mixer.sounds:
            sound.calls.clear()

    def play_all_three_channels(self):
        from sf6_music_trigger.audio import AudioAction, AudioKind

        self.port.execute(
            (AudioAction(AudioKind.PLAY, Side.P1, _SELECTED_DRIVE_SECRET, 0.2),)
        )
        self.port.execute(
            (AudioAction(AudioKind.PLAY, Side.P2, _SELECTED_UNC_SECRET, 0.3),)
        )
        self.port.preview(_SELECTED_UNC_SECRET, 0.4)
        self.reset_calls()

    def cleanup(self):
        self.mixer.clear_failures()
        self.mixer.initialized = True
        self.mixer.num_channels = 3
        self.loader.allow = None
        for channel in self.mixer.channels.values():
            channel.stop_failures_remaining = 0
            channel.play_failures_remaining = 0
            if channel.allow_play is not None:
                channel.allow_play.set()
        for port in reversed(self.ports):
            for _ in range(4):
                try:
                    port.close()
                    break
                except BaseException:
                    continue


@pytest.fixture
def fake_audio(tmp_path):
    probe = _FakeAudioProbe(tmp_path / "user-data")
    try:
        yield probe
    finally:
        probe.cleanup()


def test_audio_port_protocol_exposes_exact_structural_methods():
    from sf6_music_trigger.audio import AudioAction, AudioPort

    assert AudioPort._is_protocol is True
    assert list(inspect.signature(AudioPort.execute).parameters) == [
        "self",
        "actions",
    ]
    assert list(inspect.signature(AudioPort.preview).parameters) == [
        "self",
        "track",
        "volume",
    ]
    assert list(inspect.signature(AudioPort.close).parameters) == ["self"]
    assert get_type_hints(AudioPort.execute) == {
        "actions": tuple[AudioAction, ...],
        "return": type(None),
    }
    assert get_type_hints(AudioPort.preview) == {
        "track": str,
        "volume": Real,
        "return": type(None),
    }
    assert get_type_hints(AudioPort.close) == {"return": type(None)}


def test_import_safety_construct_and_unused_close_never_import_pygame(tmp_path):
    source_root = Path(__file__).parents[1] / "src"
    script = r'''
import builtins
import importlib
import importlib.abc
from pathlib import Path
import sys

expected_source = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])
sys.path.insert(0, str(expected_source))
real_import = builtins.__import__
real_import_module = importlib.import_module

class BlockPygameFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "pygame" or fullname.startswith("pygame."):
            raise AssertionError("pygame meta-path import attempted")
        return None

def guarded_import(name, *args, **kwargs):
    if name == "pygame" or name.startswith("pygame."):
        raise AssertionError("pygame import attempted")
    return real_import(name, *args, **kwargs)

def guarded_import_module(name, package=None):
    if name == "pygame" or name.startswith("pygame."):
        raise AssertionError("pygame importlib import attempted")
    return real_import_module(name, package)

builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
sys.meta_path.insert(0, BlockPygameFinder())
import sf6_music_trigger.audio as audio
assert Path(audio.__file__).resolve().is_relative_to(expected_source)
assert not any(name == "pygame" or name.startswith("pygame.") for name in sys.modules)

def blocked_loader():
    raise AssertionError("loader invoked")

port = audio.PygameAudioPort(root, pygame_loader=blocked_loader)
port.close()
assert not any(name == "pygame" or name.startswith("pygame.") for name in sys.modules)
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(source_root), str(tmp_path)],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_lazy_construct_and_unused_close_have_zero_loader_calls(fake_audio):
    port = fake_audio.port
    assert fake_audio.loader.calls == 0
    port.close()
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


def test_initialize_resizes_and_reserves_fixed_channels(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.mixer.initialized = False
    fake_audio.mixer.num_channels = 1
    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    assert fake_audio.loader.calls == 1
    assert fake_audio.mixer.calls[:7] == [
        ("get_init",),
        ("init",),
        ("get_num_channels",),
        ("set_num_channels", 3),
        ("get_num_channels",),
        ("Channel", 0),
        ("Channel", 1),
    ]
    assert fake_audio.mixer.calls[7] == ("Channel", 2)
    assert fake_audio.mixer.channels[0].calls == [("stop",)]


@pytest.mark.parametrize("channel_count", [3, 5])
def test_initialize_skips_init_and_resize_when_backend_is_ready(
    fake_audio,
    channel_count,
):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.mixer.initialized = True
    fake_audio.mixer.num_channels = channel_count
    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P2),))
    assert fake_audio.mixer.calls == [
        ("get_init",),
        ("get_num_channels",),
        ("Channel", 0),
        ("Channel", 1),
        ("Channel", 2),
    ]
    assert fake_audio.mixer.channels[1].calls == [("stop",)]


@pytest.mark.parametrize(
    "stage",
    [
        "loader",
        "mixer",
        "get_init",
        "init",
        "get_num_channels",
        "get_num_channels_after_resize",
        "set_num_channels",
        "channel_0",
        "channel_1",
        "channel_2",
        "still_too_small",
    ],
)
def test_initialize_failure_rolls_back_for_same_and_other_port_retry(
    fake_audio,
    stage,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    if stage in {
        "init",
        "set_num_channels",
        "get_num_channels_after_resize",
        "still_too_small",
    }:
        fake_audio.mixer.initialized = False
        fake_audio.mixer.num_channels = 1
    if stage == "still_too_small":
        fake_audio.mixer.keep_channel_count_on_set = True
    else:
        fake_audio.fail_next(stage)

    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    _assert_safe_audio_error(
        caught.value,
        "initialize",
        Side.P1,
        AudioIssue.BACKEND_FAILURE,
    )

    fake_audio.mixer.clear_failures()
    fake_audio.mixer.initialized = True
    fake_audio.mixer.num_channels = 3
    contender = fake_audio.new_port()
    contender.execute((AudioAction(AudioKind.STOP, Side.P2),))
    contender.close()
    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))


def test_initialize_failure_never_publishes_partial_channels(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind, AudioPlaybackError

    fake_audio.fail_next("channel_1")
    with pytest.raises(AudioPlaybackError):
        fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    partial_channel_zero = fake_audio.mixer.channels[0]
    fake_audio.reset_calls()
    fake_audio.port.close()
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []
    assert partial_channel_zero.calls == []

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    assert fake_audio.mixer.channels[0] is not partial_channel_zero


def test_reservation_conflict_happens_before_contender_mixer_access(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    loader_calls = fake_audio.loader.calls
    mixer_calls = list(fake_audio.mixer.calls)
    contender = fake_audio.new_port()
    with pytest.raises(AudioPlaybackError) as caught:
        contender.execute((AudioAction(AudioKind.STOP, Side.P2),))
    _assert_safe_audio_error(
        caught.value,
        "initialize",
        Side.P2,
        AudioIssue.ACTIVE_PORT_CONFLICT,
    )
    assert fake_audio.loader.calls == loader_calls
    assert fake_audio.mixer.calls == mixer_calls


def test_reservation_non_owner_close_cannot_release_active_port(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    contender = fake_audio.new_port()
    contender.close()
    with pytest.raises(AudioPlaybackError) as caught:
        contender.execute((AudioAction(AudioKind.STOP, Side.P2),))
    assert caught.value.issue is AudioIssue.ACTIVE_PORT_CONFLICT


def test_reservation_complete_close_releases_and_same_port_rebuilds(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    old_channels = dict(fake_audio.mixer.channels)
    fake_audio.port.close()
    assert [old_channels[index].stop_calls for index in range(3)] == [2, 1, 1]

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P2),))
    assert all(fake_audio.mixer.channels[index] is not old_channels[index] for index in range(3))
    assert [
        call for call in fake_audio.mixer.calls if call[0] == "Channel"
    ] == [
        ("Channel", 0),
        ("Channel", 1),
        ("Channel", 2),
        ("Channel", 0),
        ("Channel", 1),
        ("Channel", 2),
    ]


def test_initialize_concurrent_same_port_first_calls_publish_one_channel_set(
    fake_audio,
):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.loader.entered = Event()
    fake_audio.loader.allow = Event()
    errors = []
    first_done = Event()
    second_done = Event()
    second_attempted = Event()

    def run(side, done, attempted=None):
        try:
            if attempted is not None:
                attempted.set()
            fake_audio.port.execute((AudioAction(AudioKind.STOP, side),))
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    first = Thread(target=run, args=(Side.P1, first_done), daemon=True)
    second = Thread(
        target=run,
        args=(Side.P2, second_done, second_attempted),
        daemon=True,
    )
    first.start()
    assert fake_audio.loader.entered.wait(1.0)
    second.start()
    try:
        assert second_attempted.wait(1.0)
        assert not second_done.wait(0.05)
        assert fake_audio.loader.calls == 1
    finally:
        fake_audio.loader.allow.set()
        first.join(1.0)
        second.join(1.0)
    assert first_done.is_set() and second_done.is_set()
    assert errors == []
    assert fake_audio.loader.calls == 1
    assert [call for call in fake_audio.mixer.calls if call[0] == "Channel"] == [
        ("Channel", 0),
        ("Channel", 1),
        ("Channel", 2),
    ]


def test_initialize_uses_module_process_rlock_and_blocks_other_port_mid_init(
    fake_audio,
):
    import sf6_music_trigger.audio as audio_module
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    rlock_type = type(threading.RLock())
    assert any(
        isinstance(value, rlock_type) for value in vars(audio_module).values()
    )
    first_port = fake_audio.port
    second_port = fake_audio.new_port()
    fake_audio.loader.entered = Event()
    fake_audio.loader.allow = Event()
    first_done = Event()
    second_attempted = Event()
    second_done = Event()
    first_errors = []
    second_errors = []

    def run_first():
        try:
            first_port.execute((AudioAction(AudioKind.STOP, Side.P1),))
        except BaseException as error:
            first_errors.append(error)
        finally:
            first_done.set()

    def run_second():
        second_attempted.set()
        try:
            second_port.execute((AudioAction(AudioKind.STOP, Side.P2),))
        except BaseException as error:
            second_errors.append(error)
        finally:
            second_done.set()

    first = Thread(target=run_first, daemon=True)
    second = Thread(target=run_second, daemon=True)
    first.start()
    assert fake_audio.loader.entered.wait(1.0)
    second.start()
    try:
        assert second_attempted.wait(1.0)
        assert not second_done.wait(0.05)
    finally:
        fake_audio.loader.allow.set()
        first.join(1.0)
        second.join(1.0)
    assert first_done.is_set() and second_done.is_set()
    assert first_errors == []
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], AudioPlaybackError)
    _assert_safe_audio_error(
        second_errors[0],
        "initialize",
        Side.P2,
        AudioIssue.ACTIVE_PORT_CONFLICT,
    )
    assert fake_audio.loader.calls == 1


def _make_directory_link(link, target):
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError as symlink_error:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0:
            pytest.fail(
                "unable to create temporary directory link: "
                + type(symlink_error).__name__
                + "; junction exit="
                + str(completed.returncode)
            )


def _remove_directory_link(link):
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        os.rmdir(link)


@pytest.mark.parametrize("actions", [[], (), [object()]])
def test_batch_validation_rejects_non_tuple_or_empty_before_backend(
    fake_audio,
    actions,
):
    expected = ValueError if actions == () else TypeError
    with pytest.raises(expected):
        fake_audio.port.execute(actions)
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


def test_batch_validation_rejects_invalid_element_before_backend(fake_audio):
    with pytest.raises(TypeError):
        fake_audio.port.execute((object(),))
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


def test_batch_validation_rejects_mixed_side_before_backend(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    actions = (
        AudioAction(AudioKind.STOP, Side.P1),
        AudioAction(AudioKind.STOP, Side.P2),
    )
    with pytest.raises(ValueError):
        fake_audio.port.execute(actions)
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


def test_batch_validation_mixed_side_precedes_any_play_path_resolution(
    fake_audio,
    monkeypatch,
):
    import sf6_music_trigger.audio as audio_module
    from sf6_music_trigger.audio import AudioAction, AudioKind

    resolve_calls = []

    def record_resolve(root, track):
        resolve_calls.append((root, track))
        return str(root / track)

    monkeypatch.setattr(
        audio_module,
        "_resolve_audio_track",
        record_resolve,
        raising=False,
    )
    actions = (
        AudioAction(AudioKind.PLAY, Side.P1, "audio/p1.bin", 0.2),
        AudioAction(AudioKind.STOP, Side.P2),
    )
    with pytest.raises(ValueError):
        fake_audio.port.execute(actions)
    assert resolve_calls == []
    assert fake_audio.loader.calls == 0


def test_batch_validation_rejects_tampered_audio_action_before_backend(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    malformed = object.__new__(AudioAction)
    object.__setattr__(malformed, "kind", AudioKind.PLAY)
    object.__setattr__(malformed, "side", Side.P1)
    object.__setattr__(malformed, "track", "")
    object.__setattr__(malformed, "volume", 0.2)
    with pytest.raises(ValueError):
        fake_audio.port.execute((malformed,))
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


@pytest.mark.parametrize(
    "track",
    [
        r"D:\Music\selected.bin",
        "D:/Music/selected.bin",
        r"D:\Music\..\selected.bin",
        r"\\audio-server\share\selected.bin",
        "//audio-server/share/selected.bin",
        r"\\audio-server\share\folder\..\selected.bin",
    ],
)
def test_track_path_preserves_external_drive_and_unc_strings_byte_for_byte(
    fake_audio,
    track,
):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P1, track, 0.2),)
    )
    assert fake_audio.mixer.sound_paths == [track]


@pytest.mark.parametrize("track", ["audio/selected.bin", r"audio\selected.bin"])
def test_track_path_resolves_relative_string_canonically_at_call_time(
    fake_audio,
    track,
):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P1, track, 0.2),)
    )
    expected = str((fake_audio.root / "audio" / "selected.bin").resolve())
    assert fake_audio.mixer.sound_paths == [expected]


def test_track_path_allows_internal_normalization_without_existence_filter(
    fake_audio,
):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    track = "audio/missing/../unusual.extension-not-listed"
    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P2, track, 0.3),)
    )
    expected = str(
        (fake_audio.root / "audio" / "unusual.extension-not-listed").resolve()
    )
    assert fake_audio.mixer.sound_paths == [expected]


@pytest.mark.parametrize(
    "track",
    [
        "../private.bin",
        r"..\private.bin",
        "folder/../../private.bin",
        r"folder\..\..\private.bin",
        r"D:music\selected.bin",
        "D:music/selected.bin",
        r"\music\selected.bin",
        "/music/selected.bin",
        r"\\?\C:\private\selected.bin",
        r"\\.\C:\private\selected.bin",
        "//?/C:/private/selected.bin",
        "//./C:/private/selected.bin",
        r"audio\selected.bin:secret",
        r"D:\music\selected.bin:secret",
    ],
)
def test_track_rejected_windows_forms_are_safe_and_backend_free(
    fake_audio,
    track,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.execute(
            (AudioAction(AudioKind.PLAY, Side.P1, track, 0.2),)
        )
    _assert_safe_audio_error(
        caught.value,
        "load",
        Side.P1,
        AudioIssue.TRACK_REJECTED,
        track,
    )
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


def test_track_rejected_sibling_prefix_escape_is_not_containment(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    sibling = fake_audio.root.parent / (fake_audio.root.name + "-sibling")
    sibling.mkdir()
    track = "../" + sibling.name + "/private.bin"
    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.execute(
            (AudioAction(AudioKind.PLAY, Side.P1, track, 0.2),)
        )
    _assert_safe_audio_error(
        caught.value,
        "load",
        Side.P1,
        AudioIssue.TRACK_REJECTED,
        track,
    )
    assert fake_audio.loader.calls == 0


def test_track_rejected_resolved_sibling_prefix_via_directory_link(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    sibling = fake_audio.root.parent / (fake_audio.root.name + "-sibling")
    sibling.mkdir()
    link = fake_audio.root / "prefix-link"
    _make_directory_link(link, sibling)
    try:
        track = "prefix-link/private.bin"
        with pytest.raises(AudioPlaybackError) as caught:
            fake_audio.port.execute(
                (AudioAction(AudioKind.PLAY, Side.P1, track, 0.2),)
            )
        _assert_safe_audio_error(
            caught.value,
            "load",
            Side.P1,
            AudioIssue.TRACK_REJECTED,
            track,
        )
        assert fake_audio.loader.calls == 0
    finally:
        _remove_directory_link(link)


def test_track_rejected_existing_directory_link_escape_before_construction(
    fake_audio,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    outside = fake_audio.root.parent / "outside-audio"
    outside.mkdir()
    link = fake_audio.root / "linked-audio"
    _make_directory_link(link, outside)
    try:
        with pytest.raises(AudioPlaybackError) as caught:
            fake_audio.port.execute(
                (
                    AudioAction(
                        AudioKind.PLAY,
                        Side.P1,
                        "linked-audio/private.bin",
                        0.2,
                    ),
                )
            )
        _assert_safe_audio_error(
            caught.value,
            "load",
            Side.P1,
            AudioIssue.TRACK_REJECTED,
        )
        assert fake_audio.loader.calls == 0
    finally:
        _remove_directory_link(link)


def test_track_rejected_post_construction_link_proves_call_time_revalidation(
    fake_audio,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    port = fake_audio.port
    outside = fake_audio.root.parent / "outside-after-construction"
    outside.mkdir()
    link = fake_audio.root / "late-linked-audio"
    _make_directory_link(link, outside)
    try:
        with pytest.raises(AudioPlaybackError) as caught:
            port.execute(
                (
                    AudioAction(
                        AudioKind.PLAY,
                        Side.P2,
                        "late-linked-audio/private.bin",
                        0.2,
                    ),
                )
            )
        _assert_safe_audio_error(
            caught.value,
            "load",
            Side.P2,
            AudioIssue.TRACK_REJECTED,
        )
        assert fake_audio.loader.calls == 0
    finally:
        _remove_directory_link(link)


@pytest.mark.parametrize("track", [None, 7, ""])
def test_track_path_preview_rejects_structural_track_before_loader(
    fake_audio,
    track,
):
    expected = ValueError if track == "" else TypeError
    with pytest.raises(expected):
        fake_audio.port.preview(track, 0.2)
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


def test_track_rejected_preview_uses_none_side_and_no_backend(fake_audio):
    from sf6_music_trigger.audio import (
        AudioIssue,
        AudioPlaybackError,
    )

    track = "../preview-private.bin"
    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.preview(track, 0.2)
    _assert_safe_audio_error(
        caught.value,
        "load",
        None,
        AudioIssue.TRACK_REJECTED,
        track,
    )
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


@pytest.mark.parametrize(
    "raw_error",
    [
        TypeError("raw path classifier " + _RAW_BACKEND_SECRET),
        ValueError("raw path containment " + _RAW_BACKEND_SECRET),
        OSError("raw path resolver " + _RAW_BACKEND_SECRET),
    ],
)
def test_track_rejected_internal_path_exception_is_privacy_wrapped(
    fake_audio,
    monkeypatch,
    raw_error,
):
    import sf6_music_trigger.audio as audio_module
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    def reject(root, track):
        raise raw_error

    monkeypatch.setattr(
        audio_module,
        "_resolve_audio_track",
        reject,
        raising=False,
    )
    track = r"C:\private\selected.bin"
    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.execute(
            (AudioAction(AudioKind.PLAY, Side.P2, track, 0.2),)
        )
    _assert_safe_audio_error(
        caught.value,
        "load",
        Side.P2,
        AudioIssue.TRACK_REJECTED,
        track,
        str(raw_error),
    )
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []
    contender = fake_audio.new_port()
    contender.execute((AudioAction(AudioKind.STOP, Side.P1),))
    contender.close()


def test_unsafe_play_preflight_has_zero_stop_loader_or_owner_effect(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    actions = (
        AudioAction(AudioKind.STOP, Side.P1),
        AudioAction(AudioKind.PLAY, Side.P1, "../private.mp3", 0.2),
    )
    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.execute(actions)
    _assert_safe_audio_error(
        caught.value,
        "load",
        Side.P1,
        AudioIssue.TRACK_REJECTED,
        "../private.mp3",
    )
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []
    contender = fake_audio.new_port()
    contender.execute((AudioAction(AudioKind.STOP, Side.P2),))


def test_preflight_rejects_later_play_before_loading_any_earlier_play(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioKind,
        AudioPlaybackError,
    )

    actions = (
        AudioAction(AudioKind.PLAY, Side.P1, "audio/first.bin", 0.2),
        AudioAction(AudioKind.PLAY, Side.P1, "../second-private.bin", 0.2),
    )
    with pytest.raises(AudioPlaybackError):
        fake_audio.port.execute(actions)
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


def test_blocked_preflight_linearizes_before_close(fake_audio, monkeypatch):
    from threading import Event, Thread
    import sf6_music_trigger.audio as audio_module
    from sf6_music_trigger.audio import AudioAction, AudioKind

    entered = Event()
    allow = Event()
    close_attempted = Event()
    close_done = Event()
    errors = []

    def blocked_resolve(root, track):
        entered.set()
        assert allow.wait(1.0)
        return str(root / "audio/p1.bin")

    def run_play():
        try:
            fake_audio.port.execute(
                (AudioAction(AudioKind.PLAY, Side.P1, "audio/p1.bin", 0.2),)
            )
        except BaseException as error:
            errors.append(error)

    def run_close():
        close_attempted.set()
        try:
            fake_audio.port.close()
        except BaseException as error:
            errors.append(error)
        finally:
            close_done.set()

    monkeypatch.setattr(
        audio_module,
        "_resolve_audio_track",
        blocked_resolve,
        raising=False,
    )
    play_thread = Thread(target=run_play, daemon=True)
    close_thread = Thread(target=run_close, daemon=True)
    play_thread.start()
    assert entered.wait(1.0)
    close_thread.start()
    try:
        assert close_attempted.wait(1.0)
        assert not close_done.wait(0.05)
    finally:
        allow.set()
        play_thread.join(1.0)
        close_thread.join(1.0)
    assert not play_thread.is_alive() and not close_thread.is_alive()
    assert errors == []
    assert close_done.is_set()
    assert not fake_audio.mixer.channels[0].playing


def test_blocked_preflight_uses_process_lock_before_reservation(
    fake_audio,
    monkeypatch,
):
    import sf6_music_trigger.audio as audio_module
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    first_port = fake_audio.port
    second_port = fake_audio.new_port()
    entered = Event()
    allow = Event()
    first_done = Event()
    second_attempted = Event()
    second_done = Event()
    first_errors = []
    second_errors = []

    def blocked_resolve(root, track):
        entered.set()
        assert allow.wait(1.0)
        return str(root / "audio/p1.bin")

    def run_first():
        try:
            first_port.execute(
                (AudioAction(AudioKind.PLAY, Side.P1, "audio/p1.bin", 0.2),)
            )
        except BaseException as error:
            first_errors.append(error)
        finally:
            first_done.set()

    def run_second():
        second_attempted.set()
        try:
            second_port.execute((AudioAction(AudioKind.STOP, Side.P2),))
        except BaseException as error:
            second_errors.append(error)
        finally:
            second_done.set()

    monkeypatch.setattr(
        audio_module,
        "_resolve_audio_track",
        blocked_resolve,
        raising=False,
    )
    first = Thread(target=run_first, daemon=True)
    second = Thread(target=run_second, daemon=True)
    first.start()
    assert entered.wait(1.0)
    second.start()
    try:
        assert second_attempted.wait(1.0)
        assert not second_done.wait(0.05)
    finally:
        allow.set()
        first.join(1.0)
        second.join(1.0)
    assert first_done.is_set() and second_done.is_set()
    assert first_errors == []
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], AudioPlaybackError)
    _assert_safe_audio_error(
        second_errors[0],
        "initialize",
        Side.P2,
        AudioIssue.ACTIVE_PORT_CONFLICT,
    )


def test_execute_order_stop_then_play_has_no_hidden_second_stop(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    fake_audio.reset_calls()
    track = "audio/replacement.bin"
    resolved = str((fake_audio.root / track).resolve())

    fake_audio.port.execute(
        (
            AudioAction(AudioKind.STOP, Side.P1),
            AudioAction(AudioKind.PLAY, Side.P1, track, 0.2),
        )
    )

    sound = fake_audio.mixer.sounds[-1]
    assert fake_audio.timeline == [
        ("mixer.Sound", resolved),
        ("sound.set_volume", resolved, 0.2),
        ("channel.stop", 0),
        ("channel.play", 0, resolved),
    ]
    assert fake_audio.mixer.channels[0].calls == [
        ("stop",),
        ("play", sound),
    ]
    assert fake_audio.port._sound_refs[Side.P1] is sound


@pytest.mark.parametrize(("side", "channel_id"), [(Side.P1, 0), (Side.P2, 1)])
def test_fade_action_uses_only_the_fixed_side_channel(
    fake_audio,
    side,
    channel_id,
):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute((AudioAction(AudioKind.STOP, side),))
    fake_audio.reset_calls()

    fake_audio.port.execute((AudioAction(AudioKind.FADE, side, fade_ms=1500),))

    assert fake_audio.timeline == [("channel.fadeout", channel_id, 1500)]
    assert fake_audio.mixer.channels[channel_id].calls == [("fadeout", 1500)]
    other = 1 if channel_id == 0 else 0
    assert fake_audio.mixer.channels[other].calls == []


@pytest.mark.parametrize(
    ("stage", "operation"),
    [
        ("stop", "stop"),
        ("sound", "load"),
        ("set_volume", "set_volume"),
        ("play", "play"),
    ],
)
def test_playback_failure_maps_safely_and_preserves_side_references(
    fake_audio,
    capsys,
    caplog,
    stage,
    operation,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    caplog.set_level(logging.DEBUG)
    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P1, _SELECTED_UNC_SECRET, 0.2),)
    )
    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P2, _SELECTED_DRIVE_SECRET, 0.3),)
    )
    p1_reference = fake_audio.port._sound_refs[Side.P1]
    p2_reference = fake_audio.port._sound_refs[Side.P2]
    channel = fake_audio.mixer.channels[1]
    successful_capture = capsys.readouterr()
    assert successful_capture.out == successful_capture.err == ""
    assert caplog.records == []
    fake_audio.reset_calls()
    caplog.clear()

    if stage == "stop":
        channel.stop_failures_remaining = 1
        action = AudioAction(AudioKind.STOP, Side.P2)
    else:
        if stage in {"sound", "set_volume"}:
            fake_audio.fail_next(stage)
        else:
            channel.play_failures_remaining = 1
        action = AudioAction(
            AudioKind.PLAY,
            Side.P2,
            _SELECTED_UNC_SECRET,
            0.4,
        )

    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.execute((action,))

    _assert_safe_audio_error(
        caught.value,
        operation,
        Side.P2,
        AudioIssue.BACKEND_FAILURE,
        _SELECTED_DRIVE_SECRET,
        PureWindowsPath(_SELECTED_DRIVE_SECRET).name,
        _SELECTED_UNC_SECRET,
        PureWindowsPath(_SELECTED_UNC_SECRET).name,
        str(fake_audio.root),
        fake_audio.root.name,
    )
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert caplog.records == []
    assert fake_audio.port._sound_refs[Side.P1] is p1_reference
    assert fake_audio.port._sound_refs[Side.P2] is (None if stage == "play" else p2_reference)
    assert fake_audio.port._channels[1] is channel
    _assert_no_direct_track_copy(
        fake_audio.port,
        _SELECTED_DRIVE_SECRET,
        _SELECTED_UNC_SECRET,
    )


def test_playback_failure_does_not_commit_a_new_reference(
    fake_audio,
    capsys,
    caplog,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    channel = fake_audio.mixer.channels[0]
    channel.play_failures_remaining = 1

    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.execute(
            (
                AudioAction(
                    AudioKind.PLAY,
                    Side.P1,
                    _SELECTED_DRIVE_SECRET,
                    0.2,
                ),
            )
        )

    _assert_safe_audio_error(
        caught.value,
        "play",
        Side.P1,
        AudioIssue.BACKEND_FAILURE,
        _SELECTED_DRIVE_SECRET,
        PureWindowsPath(_SELECTED_DRIVE_SECRET).name,
        str(fake_audio.root),
        fake_audio.root.name,
    )
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert caplog.records == []
    assert fake_audio.port._sound_refs[Side.P1] is None
    assert fake_audio.mixer.sounds[-1] is not fake_audio.port._sound_refs[Side.P1]
    _assert_no_direct_track_copy(fake_audio.port, _SELECTED_DRIVE_SECRET)


def test_execute_order_is_fail_fast_after_first_backend_failure(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P1, "audio/original.bin", 0.2),)
    )
    channel = fake_audio.mixer.channels[0]
    fake_audio.reset_calls()
    fake_audio.fail_next("set_volume")
    actions = (
        AudioAction(AudioKind.STOP, Side.P1),
        AudioAction(AudioKind.PLAY, Side.P1, "audio/failing.bin", 0.3),
        AudioAction(AudioKind.STOP, Side.P1),
    )

    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.execute(actions)

    _assert_safe_audio_error(
        caught.value,
        "set_volume",
        Side.P1,
        AudioIssue.BACKEND_FAILURE,
    )
    assert channel.calls == []
    assert channel.playing
    assert fake_audio.port._sound_refs[Side.P1] is not None


def test_reference_ownership_remains_independent_between_p1_and_p2(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P1, "audio/p1.bin", 0.2),)
    )
    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P2, "audio/p2.bin", 0.3),)
    )
    p1_reference = fake_audio.port._sound_refs[Side.P1]
    p2_reference = fake_audio.port._sound_refs[Side.P2]

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))

    assert fake_audio.port._sound_refs[Side.P1] is None
    assert fake_audio.port._sound_refs[Side.P2] is p2_reference
    assert p1_reference is not p2_reference
    assert not fake_audio.mixer.channels[0].playing
    assert fake_audio.mixer.channels[1].playing


@pytest.mark.parametrize(
    ("volume", "expected"),
    [
        (-(10**10000), 0.0),
        (1, 1.0),
        (Fraction(1, 10**10000), 0.0),
        (Fraction(1, 3), float(Fraction(1, 3))),
        (-0.25, 0.0),
        (1.0, 1.0),
    ],
    ids=[
        "huge-negative",
        "unity-integer",
        "tiny-fraction",
        "fraction",
        "finite-float-low",
        "unity-float",
    ],
)
def test_preview_order_clamps_volume_and_isolates_side_references(
    fake_audio,
    volume,
    expected,
):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P1, "audio/p1.bin", 0.2),)
    )
    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P2, "audio/p2.bin", 0.3),)
    )
    p1_reference = fake_audio.port._sound_refs[Side.P1]
    p2_reference = fake_audio.port._sound_refs[Side.P2]
    fake_audio.reset_calls()
    track = "audio/preview.bin"
    resolved = str((fake_audio.root / track).resolve())

    fake_audio.port.preview(track, volume)

    preview_sound = fake_audio.mixer.sounds[-1]
    assert fake_audio.timeline == [
        ("channel.stop", 2),
        ("mixer.Sound", resolved),
        ("sound.set_volume", resolved, expected),
        ("channel.play", 2, resolved),
    ]
    assert fake_audio.mixer.channels[0].calls == []
    assert fake_audio.mixer.channels[1].calls == []
    assert fake_audio.port._sound_refs[Side.P1] is p1_reference
    assert fake_audio.port._sound_refs[Side.P2] is p2_reference
    assert fake_audio.port._sound_refs[None] is preview_sound


@pytest.mark.parametrize(
    ("volume", "error_type"),
    [
        (True, TypeError),
        (False, TypeError),
        ("0.2", TypeError),
        (None, TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-math.inf, ValueError),
    ],
)
def test_preview_volume_rejected_before_path_or_backend(
    fake_audio,
    monkeypatch,
    volume,
    error_type,
):
    import sf6_music_trigger.audio as audio_module

    resolve_calls = []

    def record_resolve(root, track):
        resolve_calls.append((root, track))
        return str(root / track)

    monkeypatch.setattr(audio_module, "_resolve_audio_track", record_resolve)
    with pytest.raises(error_type):
        fake_audio.port.preview("audio/preview.bin", volume)
    assert resolve_calls == []
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []


@pytest.mark.parametrize(
    ("stage", "operation"),
    [
        ("stop", "preview_stop"),
        ("sound", "load"),
        ("set_volume", "set_volume"),
        ("play", "play"),
    ],
)
def test_preview_playback_failure_maps_safely_and_preserves_retry_state(
    fake_audio,
    capsys,
    caplog,
    stage,
    operation,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    caplog.set_level(logging.DEBUG)
    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P1, _SELECTED_DRIVE_SECRET, 0.2),)
    )
    fake_audio.port.execute(
        (AudioAction(AudioKind.PLAY, Side.P2, _SELECTED_UNC_SECRET, 0.3),)
    )
    fake_audio.port.preview(_SELECTED_UNC_SECRET, 0.4)
    p1_reference = fake_audio.port._sound_refs[Side.P1]
    p2_reference = fake_audio.port._sound_refs[Side.P2]
    preview_reference = fake_audio.port._sound_refs[None]
    preview_channel = fake_audio.mixer.channels[2]
    sound_count = len(fake_audio.mixer.sounds)
    successful_capture = capsys.readouterr()
    assert successful_capture.out == successful_capture.err == ""
    assert caplog.records == []
    fake_audio.reset_calls()
    caplog.clear()

    if stage == "stop":
        preview_channel.stop_failures_remaining = 1
    elif stage in {"sound", "set_volume"}:
        fake_audio.fail_next(stage)
    else:
        preview_channel.play_failures_remaining = 1

    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.preview(_SELECTED_DRIVE_SECRET, 0.5)

    _assert_safe_audio_error(
        caught.value,
        operation,
        None,
        AudioIssue.BACKEND_FAILURE,
        _SELECTED_DRIVE_SECRET,
        PureWindowsPath(_SELECTED_DRIVE_SECRET).name,
        _SELECTED_UNC_SECRET,
        PureWindowsPath(_SELECTED_UNC_SECRET).name,
        str(fake_audio.root),
        fake_audio.root.name,
    )
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert caplog.records == []
    assert fake_audio.port._sound_refs[Side.P1] is p1_reference
    assert fake_audio.port._sound_refs[Side.P2] is p2_reference
    assert fake_audio.port._channels[2] is preview_channel
    assert fake_audio.mixer.channels[0].calls == []
    assert fake_audio.mixer.channels[1].calls == []
    _assert_no_direct_track_copy(
        fake_audio.port,
        _SELECTED_DRIVE_SECRET,
        _SELECTED_UNC_SECRET,
    )
    if stage == "stop":
        assert fake_audio.port._sound_refs[None] is preview_reference
        assert len(fake_audio.mixer.sounds) == sound_count
        assert preview_channel.playing
    else:
        assert fake_audio.port._sound_refs[None] is None
        assert not preview_channel.playing


def test_preview_first_use_initializes_and_reserves_audio_port(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.port.preview("audio/first-preview.bin", 0.2)

    assert fake_audio.loader.calls == 1
    assert [call for call in fake_audio.mixer.calls if call[0] == "Channel"] == [
        ("Channel", 0),
        ("Channel", 1),
        ("Channel", 2),
    ]
    assert fake_audio.port._sound_refs[None] is fake_audio.mixer.sounds[-1]
    contender = fake_audio.new_port()
    with pytest.raises(AudioPlaybackError) as caught:
        contender.execute((AudioAction(AudioKind.STOP, Side.P1),))
    _assert_safe_audio_error(
        caught.value,
        "initialize",
        Side.P1,
        AudioIssue.ACTIVE_PORT_CONFLICT,
    )
    assert fake_audio.loader.calls == 1


def test_preview_initialization_failure_uses_none_side_and_rolls_back(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.fail_next("loader")
    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.preview("audio/preview.bin", 0.2)
    _assert_safe_audio_error(
        caught.value,
        "initialize",
        None,
        AudioIssue.BACKEND_FAILURE,
    )

    contender = fake_audio.new_port()
    contender.execute((AudioAction(AudioKind.STOP, Side.P1),))
    contender.close()


def test_preview_active_port_conflict_uses_none_side_before_backend(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    contender = fake_audio.new_port()
    loader_calls = fake_audio.loader.calls
    mixer_calls = list(fake_audio.mixer.calls)
    with pytest.raises(AudioPlaybackError) as caught:
        contender.preview("audio/preview.bin", 0.2)
    _assert_safe_audio_error(
        caught.value,
        "initialize",
        None,
        AudioIssue.ACTIVE_PORT_CONFLICT,
    )
    assert fake_audio.loader.calls == loader_calls
    assert fake_audio.mixer.calls == mixer_calls


@pytest.mark.parametrize("method", ["execute", "preview"])
def test_preview_and_execute_preflight_rejection_precedes_active_port_conflict(
    fake_audio,
    method,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    owner = fake_audio.port
    owner.execute((AudioAction(AudioKind.STOP, Side.P1),))
    contender = fake_audio.new_port()
    loader_calls = fake_audio.loader.calls
    mixer_calls = list(fake_audio.mixer.calls)
    channel_calls = {
        channel_id: list(channel.calls)
        for channel_id, channel in fake_audio.mixer.channels.items()
    }
    track = "../contender-private-track.bin"

    with pytest.raises(AudioPlaybackError) as caught:
        if method == "execute":
            contender.execute(
                (AudioAction(AudioKind.PLAY, Side.P2, track, 0.2),)
            )
        else:
            contender.preview(track, 0.2)
    _assert_safe_audio_error(
        caught.value,
        "load",
        Side.P2 if method == "execute" else None,
        AudioIssue.TRACK_REJECTED,
        track,
        PureWindowsPath(track).name,
    )
    assert fake_audio.loader.calls == loader_calls
    assert fake_audio.mixer.calls == mixer_calls
    assert {
        channel_id: list(channel.calls)
        for channel_id, channel in fake_audio.mixer.channels.items()
    } == channel_calls

    with pytest.raises(AudioPlaybackError) as conflict:
        contender.execute((AudioAction(AudioKind.STOP, Side.P2),))
    _assert_safe_audio_error(
        conflict.value,
        "initialize",
        Side.P2,
        AudioIssue.ACTIVE_PORT_CONFLICT,
    )


def test_event_batch_linearizes_before_close_and_ends_silent(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    channel = fake_audio.mixer.channels[0]
    channel.play_entered = Event()
    channel.allow_play = Event()
    fake_audio.reset_calls()
    actions = (
        AudioAction(AudioKind.STOP, Side.P1),
        AudioAction(AudioKind.PLAY, Side.P1, "audio/p1.bin", 0.2),
    )
    batch_errors = []
    close_errors = []
    close_attempted = Event()
    close_done = Event()

    def run_batch():
        try:
            fake_audio.port.execute(actions)
        except BaseException as error:
            batch_errors.append(error)

    def run_close():
        close_attempted.set()
        try:
            fake_audio.port.close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_done.set()

    batch_thread = Thread(target=run_batch, daemon=True)
    close_thread = Thread(target=run_close, daemon=True)
    batch_thread.start()
    assert channel.play_entered.wait(1.0)
    close_thread.start()
    try:
        assert close_attempted.wait(1.0)
        assert not close_done.wait(0.05)
    finally:
        channel.allow_play.set()
        batch_thread.join(1.0)
        close_thread.join(1.0)
    assert not batch_thread.is_alive() and not close_thread.is_alive()
    assert batch_errors == close_errors == []
    assert [call[0] for call in channel.calls[-3:]] == ["stop", "play", "stop"]
    assert not channel.playing


def test_preview_linearization_before_close_ends_channel_two_silent(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    channel = fake_audio.mixer.channels[2]
    channel.play_entered = Event()
    channel.allow_play = Event()
    fake_audio.reset_calls()
    preview_errors = []
    close_errors = []
    close_attempted = Event()
    close_done = Event()

    def run_preview():
        try:
            fake_audio.port.preview("audio/preview.bin", 0.2)
        except BaseException as error:
            preview_errors.append(error)

    def run_close():
        close_attempted.set()
        try:
            fake_audio.port.close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_done.set()

    preview_thread = Thread(target=run_preview, daemon=True)
    close_thread = Thread(target=run_close, daemon=True)
    preview_thread.start()
    assert channel.play_entered.wait(1.0)
    close_thread.start()
    try:
        assert close_attempted.wait(1.0)
        assert not close_done.wait(0.05)
    finally:
        channel.allow_play.set()
        preview_thread.join(1.0)
        close_thread.join(1.0)
    assert not preview_thread.is_alive() and not close_thread.is_alive()
    assert preview_errors == close_errors == []
    assert [call[0] for call in channel.calls[-3:]] == ["stop", "play", "stop"]
    assert not channel.playing


def test_preview_linearization_uses_process_lock_across_ports(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    first_port = fake_audio.port
    first_port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    second_port = fake_audio.new_port()
    channel = fake_audio.mixer.channels[2]
    channel.play_entered = Event()
    channel.allow_play = Event()
    preview_done = Event()
    execute_attempted = Event()
    execute_done = Event()
    close_attempted = Event()
    close_done = Event()
    preview_errors = []
    execute_errors = []
    close_errors = []

    def run_preview():
        try:
            first_port.preview("audio/preview.bin", 0.2)
        except BaseException as error:
            preview_errors.append(error)
        finally:
            preview_done.set()

    def run_execute():
        execute_attempted.set()
        try:
            second_port.execute((AudioAction(AudioKind.STOP, Side.P2),))
        except BaseException as error:
            execute_errors.append(error)
        finally:
            execute_done.set()

    def run_close():
        close_attempted.set()
        try:
            second_port.close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_done.set()

    preview_thread = Thread(target=run_preview, daemon=True)
    execute_thread = Thread(target=run_execute, daemon=True)
    close_thread = Thread(target=run_close, daemon=True)
    preview_thread.start()
    assert channel.play_entered.wait(1.0)
    execute_thread.start()
    close_thread.start()
    try:
        assert execute_attempted.wait(1.0)
        assert close_attempted.wait(1.0)
        assert not execute_done.wait(0.05)
        assert not close_done.wait(0.05)
    finally:
        channel.allow_play.set()
        preview_thread.join(1.0)
        execute_thread.join(1.0)
        close_thread.join(1.0)
    assert not any(
        thread.is_alive()
        for thread in (preview_thread, execute_thread, close_thread)
    )
    assert preview_done.is_set() and execute_done.is_set() and close_done.is_set()
    assert preview_errors == close_errors == []
    assert len(execute_errors) == 1
    assert isinstance(execute_errors[0], AudioPlaybackError)
    _assert_safe_audio_error(
        execute_errors[0],
        "initialize",
        Side.P2,
        AudioIssue.ACTIVE_PORT_CONFLICT,
    )


def test_partial_close_retries_only_failed_channel_then_releases(
    fake_audio,
    capsys,
    caplog,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    caplog.set_level(logging.DEBUG)
    fake_audio.play_all_three_channels()
    successful_capture = capsys.readouterr()
    assert successful_capture.out == successful_capture.err == ""
    assert caplog.records == []
    caplog.clear()
    channels = dict(fake_audio.mixer.channels)
    references = dict(fake_audio.port._sound_refs)
    channels[1].stop_failures_remaining = 1

    with pytest.raises(AudioPlaybackError) as first:
        fake_audio.port.close()
    _assert_safe_audio_error(
        first.value,
        "close_stop",
        Side.P2,
        AudioIssue.BACKEND_FAILURE,
        _SELECTED_DRIVE_SECRET,
        PureWindowsPath(_SELECTED_DRIVE_SECRET).name,
        _SELECTED_UNC_SECRET,
        PureWindowsPath(_SELECTED_UNC_SECRET).name,
        str(fake_audio.root),
        fake_audio.root.name,
    )
    failed_capture = capsys.readouterr()
    assert failed_capture.out == failed_capture.err == ""
    assert caplog.records == []
    _assert_no_direct_track_copy(
        fake_audio.port,
        _SELECTED_DRIVE_SECRET,
        _SELECTED_UNC_SECRET,
    )
    assert [channels[index].stop_calls for index in range(3)] == [1, 1, 1]
    assert fake_audio.port._channels == (
        channels[0],
        channels[1],
        channels[2],
    )
    assert fake_audio.port._sound_refs == {
        Side.P1: None,
        Side.P2: references[Side.P2],
        None: None,
    }

    with pytest.raises(AudioPlaybackError) as gated:
        fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    _assert_safe_audio_error(
        gated.value,
        "execute",
        Side.P1,
        AudioIssue.CLOSE_INCOMPLETE,
    )
    assert [channels[index].stop_calls for index in range(3)] == [1, 1, 1]

    fake_audio.port.close()
    assert [channels[index].stop_calls for index in range(3)] == [1, 2, 1]
    assert fake_audio.port._channels is None
    assert all(value is None for value in fake_audio.port._sound_refs.values())
    replacement = fake_audio.new_port()
    replacement.execute((AudioAction(AudioKind.STOP, Side.P1),))


def test_cleanup_best_effort_order_retains_only_failed_refs_and_first_error(
    fake_audio,
):
    from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError

    fake_audio.play_all_three_channels()
    channels = dict(fake_audio.mixer.channels)
    references = dict(fake_audio.port._sound_refs)
    channels[0].stop_failures_remaining = 1
    channels[2].stop_failures_remaining = 2

    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.close()

    _assert_safe_audio_error(
        caught.value,
        "close_stop",
        Side.P1,
        AudioIssue.BACKEND_FAILURE,
    )
    assert fake_audio.timeline == [
        ("mixer.get_init",),
        ("channel.stop", 0),
        ("channel.stop", 1),
        ("channel.stop", 2),
    ]
    assert [channels[index].stop_calls for index in range(3)] == [1, 1, 1]
    assert fake_audio.port._sound_refs == {
        Side.P1: references[Side.P1],
        Side.P2: None,
        None: references[None],
    }
    assert fake_audio.port._channels == (
        channels[0],
        channels[1],
        channels[2],
    )

    with pytest.raises(AudioPlaybackError) as second:
        fake_audio.port.close()
    _assert_safe_audio_error(
        second.value,
        "close_stop",
        None,
        AudioIssue.BACKEND_FAILURE,
    )
    assert [channels[index].stop_calls for index in range(3)] == [2, 1, 2]
    assert fake_audio.port._sound_refs == {
        Side.P1: None,
        Side.P2: None,
        None: references[None],
    }
    assert fake_audio.port._channels == (
        channels[0],
        channels[1],
        channels[2],
    )

    fake_audio.port.close()
    assert [channels[index].stop_calls for index in range(3)] == [2, 1, 3]
    assert fake_audio.port._channels is None


def test_cleanup_initial_get_init_failure_retains_all_state_and_owner(
    fake_audio,
    capsys,
    caplog,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    caplog.set_level(logging.DEBUG)
    fake_audio.play_all_three_channels()
    successful_capture = capsys.readouterr()
    assert successful_capture.out == successful_capture.err == ""
    assert caplog.records == []
    caplog.clear()
    channels = tuple(fake_audio.port._channels)
    references = dict(fake_audio.port._sound_refs)
    fake_audio.fail_next("get_init")

    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.close()

    _assert_safe_audio_error(
        caught.value,
        "close_stop",
        None,
        AudioIssue.BACKEND_FAILURE,
        _SELECTED_DRIVE_SECRET,
        PureWindowsPath(_SELECTED_DRIVE_SECRET).name,
        _SELECTED_UNC_SECRET,
        PureWindowsPath(_SELECTED_UNC_SECRET).name,
        str(fake_audio.root),
        fake_audio.root.name,
    )
    failed_capture = capsys.readouterr()
    assert failed_capture.out == failed_capture.err == ""
    assert caplog.records == []
    assert fake_audio.mixer.calls == [("get_init",)]
    assert [channel.stop_calls for channel in channels] == [0, 0, 0]
    assert fake_audio.port._channels == channels
    assert fake_audio.port._sound_refs == references
    _assert_no_direct_track_copy(
        fake_audio.port,
        _SELECTED_DRIVE_SECRET,
        _SELECTED_UNC_SECRET,
    )
    contender = fake_audio.new_port()
    with pytest.raises(AudioPlaybackError) as conflict:
        contender.execute((AudioAction(AudioKind.STOP, Side.P1),))
    _assert_safe_audio_error(
        conflict.value,
        "initialize",
        Side.P1,
        AudioIssue.ACTIVE_PORT_CONFLICT,
    )

    fake_audio.port.close()
    assert [channel.stop_calls for channel in channels] == [1, 1, 1]
    contender.execute((AudioAction(AudioKind.STOP, Side.P1),))


def test_cleanup_get_init_failure_retains_partial_state_and_owner_for_retry(
    fake_audio,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.play_all_three_channels()
    channels = dict(fake_audio.mixer.channels)
    references = dict(fake_audio.port._sound_refs)
    channels[1].stop_failures_remaining = 1
    with pytest.raises(AudioPlaybackError):
        fake_audio.port.close()
    assert fake_audio.port._sound_refs[Side.P2] is references[Side.P2]
    fake_audio.fail_next("get_init")

    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.close()

    _assert_safe_audio_error(
        caught.value,
        "close_stop",
        None,
        AudioIssue.BACKEND_FAILURE,
    )
    assert [channels[index].stop_calls for index in range(3)] == [1, 1, 1]
    assert fake_audio.port._channels == (
        channels[0],
        channels[1],
        channels[2],
    )
    assert fake_audio.port._sound_refs == {
        Side.P1: None,
        Side.P2: references[Side.P2],
        None: None,
    }
    contender = fake_audio.new_port()
    with pytest.raises(AudioPlaybackError) as conflict:
        contender.execute((AudioAction(AudioKind.STOP, Side.P1),))
    assert conflict.value.issue is AudioIssue.ACTIVE_PORT_CONFLICT

    fake_audio.port.close()
    assert [channels[index].stop_calls for index in range(3)] == [1, 2, 1]
    contender.execute((AudioAction(AudioKind.STOP, Side.P1),))


def test_partial_cleanup_gates_execute_and_preview_without_other_effects(
    fake_audio,
    monkeypatch,
):
    import sf6_music_trigger.audio as audio_module
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.play_all_three_channels()
    channel = fake_audio.mixer.channels[1]
    channel.stop_failures_remaining = 1
    with pytest.raises(AudioPlaybackError):
        fake_audio.port.close()
    references = dict(fake_audio.port._sound_refs)
    channels = tuple(fake_audio.port._channels)
    fake_audio.reset_calls()
    resolve_calls = []

    def record_resolve(root, track):
        resolve_calls.append((root, track))
        return str(root / track)

    monkeypatch.setattr(audio_module, "_resolve_audio_track", record_resolve)

    with pytest.raises(AudioPlaybackError) as execute_error:
        fake_audio.port.execute(
            (
                AudioAction(
                    AudioKind.PLAY,
                    Side.P1,
                    "audio/gated-p1.bin",
                    0.2,
                ),
            )
        )
    _assert_safe_audio_error(
        execute_error.value,
        "execute",
        Side.P1,
        AudioIssue.CLOSE_INCOMPLETE,
    )
    with pytest.raises(AudioPlaybackError) as preview_error:
        fake_audio.port.preview("audio/gated-preview.bin", 0.2)
    _assert_safe_audio_error(
        preview_error.value,
        "preview",
        None,
        AudioIssue.CLOSE_INCOMPLETE,
    )

    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []
    assert fake_audio.timeline == []
    assert resolve_calls == []
    assert fake_audio.port._sound_refs == references
    assert fake_audio.port._channels == channels

    fake_audio.port.close()


def test_partial_cleanup_non_owner_close_cannot_release_or_mutate_owner(
    fake_audio,
):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.play_all_three_channels()
    owner = fake_audio.port
    channels = tuple(owner._channels)
    channels[1].stop_failures_remaining = 1
    with pytest.raises(AudioPlaybackError):
        owner.close()
    references = dict(owner._sound_refs)
    contender = fake_audio.new_port()
    fake_audio.reset_calls()

    contender.close()

    assert owner._channels == channels
    assert owner._sound_refs == references
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []
    assert fake_audio.timeline == []
    with pytest.raises(AudioPlaybackError) as conflict:
        contender.execute((AudioAction(AudioKind.STOP, Side.P2),))
    _assert_safe_audio_error(
        conflict.value,
        "initialize",
        Side.P2,
        AudioIssue.ACTIVE_PORT_CONFLICT,
    )
    with pytest.raises(AudioPlaybackError) as gated:
        owner.execute((AudioAction(AudioKind.STOP, Side.P1),))
    assert (gated.value.operation, gated.value.issue) == (
        "execute",
        AudioIssue.CLOSE_INCOMPLETE,
    )

    owner.close()
    contender.execute((AudioAction(AudioKind.STOP, Side.P2),))


def test_cleanup_known_uninitialized_clears_without_channel_stops(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.play_all_three_channels()
    channels = dict(fake_audio.mixer.channels)
    fake_audio.mixer.initialized = False

    fake_audio.port.close()

    assert fake_audio.mixer.calls == [("get_init",)]
    assert [channels[index].stop_calls for index in range(3)] == [0, 0, 0]
    assert fake_audio.port._channels is None
    assert fake_audio.port._mixer is None
    assert fake_audio.port._pygame is None
    assert all(value is None for value in fake_audio.port._sound_refs.values())
    fake_audio.mixer.initialized = True
    replacement = fake_audio.new_port()
    replacement.execute((AudioAction(AudioKind.STOP, Side.P1),))


def test_cleanup_known_uninitialized_finishes_partial_close_without_retrying(
    fake_audio,
):
    from sf6_music_trigger.audio import AudioAction, AudioKind, AudioPlaybackError

    fake_audio.play_all_three_channels()
    channels = dict(fake_audio.mixer.channels)
    channels[1].stop_failures_remaining = 1
    with pytest.raises(AudioPlaybackError):
        fake_audio.port.close()
    assert [channels[index].stop_calls for index in range(3)] == [1, 1, 1]
    fake_audio.mixer.initialized = False
    fake_audio.reset_calls()

    fake_audio.port.close()

    assert fake_audio.mixer.calls == [("get_init",)]
    assert [channels[index].stop_calls for index in range(3)] == [0, 0, 0]
    assert fake_audio.port._channels is None
    fake_audio.mixer.initialized = True
    replacement = fake_audio.new_port()
    replacement.execute((AudioAction(AudioKind.STOP, Side.P2),))


def test_cleanup_complete_close_is_idempotent_and_same_port_is_reusable(
    fake_audio,
):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.play_all_three_channels()
    old_channels = dict(fake_audio.mixer.channels)

    fake_audio.port.close()

    assert [old_channels[index].stop_calls for index in range(3)] == [1, 1, 1]
    assert fake_audio.port._channels is None
    assert fake_audio.port._mixer is None
    assert fake_audio.port._pygame is None
    assert all(value is None for value in fake_audio.port._sound_refs.values())
    calls_after_first_close = list(fake_audio.mixer.calls)
    fake_audio.port.close()
    assert fake_audio.mixer.calls == calls_after_first_close

    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P2),))
    assert all(
        fake_audio.mixer.channels[index] is not old_channels[index]
        for index in range(3)
    )
    fake_audio.port.close()


def test_port_never_calls_process_global_pygame_or_mixer_shutdown(fake_audio):
    from sf6_music_trigger.audio import AudioPlaybackError

    fake_audio.play_all_three_channels()
    channels = dict(fake_audio.mixer.channels)
    channels[1].stop_failures_remaining = 1

    with pytest.raises(AudioPlaybackError):
        fake_audio.port.close()
    fake_audio.port.close()

    assert ("stop",) not in fake_audio.mixer.calls
    assert ("quit",) not in fake_audio.mixer.calls
    assert fake_audio.loader.module.quit_calls == 0
    assert not any(
        call in {("mixer.stop",), ("mixer.quit",), ("pygame.quit",)}
        for call in fake_audio.timeline
    )


def test_port_never_uses_automatic_channel_or_sound_playback(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.port.execute(
        (
            AudioAction(
                AudioKind.PLAY,
                Side.P1,
                _SELECTED_DRIVE_SECRET,
                0.2,
            ),
        )
    )
    fake_audio.port.preview(_SELECTED_UNC_SECRET, 0.3)

    assert not any(call[0] == "find_channel" for call in fake_audio.mixer.calls)
    assert not any(
        call[0] == "play"
        for sound in fake_audio.mixer.sounds
        for call in sound.calls
    )
    assert not any(
        call[0] in {"mixer.find_channel", "sound.play"}
        for call in fake_audio.timeline
    )


def test_default_loader_constructor_is_exact_and_remains_lazy(
    tmp_path,
    monkeypatch,
):
    import builtins
    import sf6_music_trigger.audio as audio_module

    import_calls = []
    builtin_calls = []
    real_builtin_import = builtins.__import__

    def forbidden_import(name):
        import_calls.append(name)
        raise AssertionError("default loader ran during construction or unused close")

    def guarded_builtin_import(name, *args, **kwargs):
        if name == "pygame" or name.startswith("pygame."):
            builtin_calls.append(name)
            raise AssertionError("pygame was eagerly imported by the default port")
        return real_builtin_import(name, *args, **kwargs)

    monkeypatch.setattr(audio_module.importlib, "import_module", forbidden_import)
    monkeypatch.setattr(builtins, "__import__", guarded_builtin_import)
    port = audio_module.PygameAudioPort(tmp_path)
    port.close()
    assert import_calls == []
    assert builtin_calls == []
    pygame_loader = inspect.signature(
        audio_module.PygameAudioPort
    ).parameters["pygame_loader"]
    assert pygame_loader.default is audio_module._load_pygame


def test_default_loader_calls_importlib_for_pygame_only(monkeypatch):
    import sf6_music_trigger.audio as audio_module

    sentinel = object()
    import_calls = []

    def record_import(name):
        import_calls.append(name)
        return sentinel

    monkeypatch.setattr(audio_module.importlib, "import_module", record_import)
    assert audio_module._load_pygame() is sentinel
    assert import_calls == ["pygame"]


def test_sdl_dummy_default_loader_roundtrip(tmp_path):
    from importlib.metadata import PackageNotFoundError, version
    from packaging.specifiers import SpecifierSet
    import wave

    try:
        installed = version("pygame")
    except PackageNotFoundError:
        pytest.fail("Task 7 environment blocker: pygame is not installed")
    assert installed in SpecifierSet(">=2.5,<3"), (
        "Task 7 environment requires pygame>=2.5,<3; installed=" + installed
    )
    repository_root = Path(__file__).parents[1].resolve()
    source_root = (repository_root / "src").resolve()
    repository_wavs_before = {
        path.resolve() for path in repository_root.rglob("*.wav")
    }
    wav_path = tmp_path / "silence.wav"
    assert not wav_path.resolve().is_relative_to(repository_root)
    with wave.open(str(wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44_100)
        stream.writeframes(b"\x00\x00" * 4_410)
    with wave.open(str(wav_path), "rb") as stream:
        assert (
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.getframerate(),
            stream.getnframes(),
        ) == (1, 2, 44_100, 4_410)

    script = r'''
import os
from pathlib import Path
import sys

assert os.environ["SDL_AUDIODRIVER"] == "dummy"
assert os.environ["SDL_VIDEODRIVER"] == "dummy"
assert os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] == "1"
expected_source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(expected_source))
import sf6_music_trigger.audio as audio
from sf6_music_trigger.domain import Side
assert Path(audio.__file__).resolve().is_relative_to(expected_source)
root, track = sys.argv[2], sys.argv[3]
port = audio.PygameAudioPort(root)
port.execute((
    audio.AudioAction(audio.AudioKind.STOP, Side.P1),
    audio.AudioAction(audio.AudioKind.PLAY, Side.P1, track, 0.18),
))
port.preview(track, 0.25)
port.close()
port.execute((
    audio.AudioAction(audio.AudioKind.STOP, Side.P2),
    audio.AudioAction(audio.AudioKind.PLAY, Side.P2, track, 0.18),
))
port.close()
'''
    env = os.environ.copy()
    env["SDL_AUDIODRIVER"] = "dummy"
    env["SDL_VIDEODRIVER"] = "dummy"
    env["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(source_root)
    if existing_pythonpath:
        env["PYTHONPATH"] += os.pathsep + existing_pythonpath
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(source_root),
            str(tmp_path),
            str(wav_path),
        ],
        cwd=repository_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert wav_path.is_file()
    assert {
        path.resolve() for path in repository_root.rglob("*.wav")
    } == repository_wavs_before
