"""Audio routing of resolved BLOCK decisions; recognition is tested separately."""

import pytest

from sf6_music_trigger.audio import AudioAction, AudioKind, AudioScheduler
from sf6_music_trigger.config import SideAudioConfig
from sf6_music_trigger.domain import Side
from sf6_music_trigger.outcome import OutcomeDecision, OutcomeLabel
from test_engine import _build_harness, _event


def _decision(side, label=OutcomeLabel.BLOCK, event_id="shared-event"):
    evidence = (
        "opponent_health_changed" if label is OutcomeLabel.HIT
        else "opponent_drive_changed"
    )
    return OutcomeDecision(event_id, side, label, 0.9, 12.0, (evidence,))


def test_legacy_scene_luma_block_cannot_fade_real_music():
    decision = OutcomeDecision('old-block', Side.P2, OutcomeLabel.BLOCK, .9, 12.,
                               ('training_result_changed',))
    assert AudioScheduler(Side.P2).on_outcome(decision, SideAudioConfig()) == ()


@pytest.mark.parametrize("side", list(Side))
@pytest.mark.parametrize("fade_ms", [0, 725])
@pytest.mark.parametrize("play_whiff_sound", [False, True])
def test_block_follows_selected_sound_or_fade_mode(
    side, fade_ms, play_whiff_sound,
):
    config = SideAudioConfig(
        whiff_fade_ms=fade_ms,
        play_whiff_sound=play_whiff_sound,
        whiff_track="audio/whiff.wav",
    )

    expected = (
        (
            AudioAction(AudioKind.STOP, side),
            AudioAction(AudioKind.PLAY, side, "audio/whiff.wav", 0.18),
        )
        if play_whiff_sound
        else (AudioAction(AudioKind.FADE, side, fade_ms=fade_ms),)
    )
    assert AudioScheduler(side).on_outcome(_decision(side), config) == expected


@pytest.mark.parametrize("side", list(Side))
def test_disabled_audio_ignores_block(side):
    assert AudioScheduler(side).on_outcome(
        _decision(side), SideAudioConfig(enabled=False),
    ) == ()


@pytest.mark.parametrize("side", list(Side))
def test_block_rejects_other_side_before_routing(side):
    other_side = Side.P2 if side is Side.P1 else Side.P1
    with pytest.raises(ValueError, match="decision side must match"):
        AudioScheduler(side).on_outcome(_decision(other_side), SideAudioConfig())


@pytest.mark.parametrize("blocked_side", list(Side))
@pytest.mark.parametrize("fade_non_three", [False, True])
def test_confirmed_block_fades_only_owner_while_other_side_hit_keeps_music(
    blocked_side, fade_non_three,
):
    config = SideAudioConfig(
        main_track="audio/main.wav",
        whiff_track="audio/whiff.wav",
        play_whiff_sound=True,
        whiff_fade_ms=725,
        fade_non_three=fade_non_three,
    )
    harness = _build_harness(p1_config=config, p2_config=config)
    for side in Side:
        harness.engine._route_confirmation(
            _event(side, event_id="shared-event", levels_spent=3),
        )
    assert len(harness.audio.batches) == 2
    assert all(
        [action.kind for action in batch] == [AudioKind.STOP, AudioKind.PLAY]
        for batch in harness.audio.batches
    )

    harness.clock.value = 12.0
    harness.engine._route_outcomes(tuple(
        _decision(side, OutcomeLabel.BLOCK if side is blocked_side else OutcomeLabel.HIT)
        for side in Side
    ))

    assert harness.audio.batches[2:] == ([
        (
            AudioAction(AudioKind.STOP, blocked_side),
            AudioAction(AudioKind.PLAY, blocked_side, "audio/whiff.wav", 0.18),
        ),
    ] if blocked_side is Side.P2 else [])
    assert [record.outcome for record in harness.diagnostic_records[-2:]] == [
        OutcomeLabel.BLOCK if side is blocked_side else OutcomeLabel.HIT
        for side in Side
    ]
    assert harness.engine.diagnostic_warning is False


@pytest.mark.parametrize("side", list(Side))
def test_delayed_block_cannot_fade_newer_event_music(side):
    harness = _build_harness()
    for event_id in ("older", "newer"):
        harness.engine._route_confirmation(
            _event(side, event_id=event_id, levels_spent=3),
        )
    assert len(harness.audio.batches) == 2

    harness.clock.value = 12.0
    harness.engine._route_outcomes((_decision(side, event_id="older"),))
    assert len(harness.audio.batches) == 2

    harness.engine._route_outcomes((_decision(side, event_id="newer"),))
    assert harness.audio.batches[2:] == [
        (AudioAction(AudioKind.FADE, side, fade_ms=1500),),
    ]
