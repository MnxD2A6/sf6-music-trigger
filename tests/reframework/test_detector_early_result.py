"""Detector v0.2: early super_result from attack frames and first contact."""

from pathlib import Path

import pytest

from fake_reframework import FakeSF6, super_trigger

DETECTOR = (
    Path(__file__).resolve().parents[2]
    / "src" / "sf6_music_trigger" / "game_scripts" / "sf6_music_trigger.lua"
)
EVENTS = "SF6MusicTrigger/events.jsonl"
BAS, GRD, DMG = 1, 100, 200
FRAME = 1 / 60

# Shaped after the real MMDK data: Jamie SA3/CA strike at 93-96 and 109-112,
# Zangief CA throw at 103-105, Ryu SA1 is a projectile (ShotKey).
JAMIE_FAB = {
    1210: {"attack": [(93, 96), (109, 112)]},
    1211: {},                                   # hit cinematic, no attack frames
    1215: {"attack": [(93, 96), (109, 112)]},
    1220: {"attack": [(20, 25)]},               # multi-stage: first stage
    1221: {"attack": [(10, 14)]},               # multi-stage: second stage
    1230: {"projectile": True},
}


@pytest.fixture
def game():
    g = FakeSF6(DETECTOR)
    g.set_chara(1, 21)
    g.set_chara(2, 6)
    g.set_triggers(1, [super_trigger(1210, 3, 30000), super_trigger(1215, 4, 30000),
                       super_trigger(1220, 2, 20000), super_trigger(1230, 1, 10000)])
    g.set_triggers(2, [super_trigger(1222, 4, 30000)])
    g.set_fab(1, JAMIE_FAB)
    g.set_fab(2, {1222: {"attack": [(103, 105)]}, 1223: {}})
    g.load()
    g.frames(5)
    return g


def events(game, kind):
    return [e for e in game.lines(EVENTS) if e["type"] == kind]


def start_super(game, side, act):
    game.set_act(side, act)
    game.frames(1)
    return events(game, "super_start")[-1]["t"]


def test_attack_windows_are_reported_per_character(game):
    windows = {e["side"]: e for e in events(game, "windows")}
    assert windows[1]["chara"] == 21
    timed = {row[0]: (row[1], row[2]) for row in windows[1]["timed"]}
    assert timed[1210] == (112, False)
    assert timed[1230] == (-1, True)
    assert {row[0]: row[1] for row in windows[2]["timed"]}[1222] == 105


def test_whiff_is_decided_right_after_the_attack_frames(game):
    started = start_super(game, 1, 1210)
    game.frames(115)
    assert events(game, "super_result") == []
    game.frames(10)
    result = events(game, "super_result")[-1]
    assert (result["result"], result["reason"], result["label"]) == ("whiff", "attack_frames_over", "SA3")
    assert 1.9 < result["t"] - started < 2.1
    # The episode itself still ends only when the character recovers.
    game.frames(60)
    assert events(game, "super_end") == []
    game.set_act(1, BAS)
    game.frames(1)
    assert events(game, "super_end")[-1]["result"] == "whiff"
    assert len(events(game, "super_result")) == 1


def test_block_is_decided_on_first_guard(game):
    started = start_super(game, 1, 1215)
    game.frames(91)
    game.set_act(2, GRD)
    game.frames(1)
    result = events(game, "super_result")[-1]
    assert (result["result"], result["reason"], result["label"]) == ("block", "contact", "CA")
    assert result["t"] - started < 1.6


def test_contact_during_the_whiff_grace_still_counts_as_hit(game):
    start_super(game, 1, 1210)
    game.frames(116)          # attack frames over, grace running
    game.set_act(2, DMG)      # opponent's reaction lands late
    game.frames(1)
    assert [e["result"] for e in events(game, "super_result")] == ["hit"]


def test_opponent_locked_in_grab_cinematic_is_a_hit(game):
    game.set_act(1, BAS)
    start_super(game, 2, 1222)
    game.frames(103)
    game.set_act(2, 1223)     # attacker branches to the grab cinematic
    game.set_act(1, 1224)     # victim plays an action from the attacker's block
    game.frames(30)
    results = events(game, "super_result")
    assert [(e["side"], e["result"], e["reason"]) for e in results] == [(2, "hit", "contact")]


def test_branch_to_recovery_without_contact_is_a_whiff(game):
    start_super(game, 1, 1220)
    game.frames(10)
    game.set_act(1, 1211)     # recovery action without attack frames, before window end
    game.frames(10)
    assert [e["result"] for e in events(game, "super_result")] == ["whiff"]


def test_multi_stage_super_waits_for_the_later_stage(game):
    start_super(game, 1, 1220)
    game.frames(20)
    game.set_act(1, 1221)     # second attacking stage
    game.frames(10)
    assert events(game, "super_result") == []
    game.set_act(2, DMG)
    game.frames(1)
    assert [e["result"] for e in events(game, "super_result")] == ["hit"]


def test_projectile_super_is_decided_at_the_end(game):
    start_super(game, 1, 1230)
    game.frames(200)
    assert events(game, "super_result") == []
    game.set_act(1, BAS)
    game.frames(1)
    assert [(e["result"], e["reason"]) for e in events(game, "super_result")] == [("whiff", "recovered")]


def test_without_frame_data_the_result_comes_at_the_end(game):
    game.set_fab(1, {})
    game.set_chara(1, 22)     # force a fresh read with no FAB data
    game.frames(70)
    start_super(game, 1, 1210)
    game.frames(200)
    assert events(game, "super_result") == []
    game.set_act(1, BAS)
    game.frames(1)
    assert [e["reason"] for e in events(game, "super_result")] == ["recovered"]


def test_loading_errors_are_rate_limited():
    g = FakeSF6(DETECTOR)
    g.set_engine_error(True)
    g.load()
    g.frames(60 * 12)
    errors = [e for e in g.lines(EVENTS) if e["type"] == "error"]
    assert 2 <= len(errors) <= 4
    assert errors[1]["suppressed"] > 200
