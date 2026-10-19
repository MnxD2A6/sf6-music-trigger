from pathlib import Path

import pytest

from fake_reframework import FakeSF6, normal_trigger, super_trigger

DETECTOR = (
    Path(__file__).resolve().parents[2]
    / "src" / "sf6_music_trigger" / "game_scripts" / "sf6_music_trigger.lua"
)
EVENTS = "SF6MusicTrigger/events.jsonl"

BAS, GRD, DMG, THROWN, OPP_ATTACK = 1, 100, 200, 320, 600
JAMIE, ZANGIEF, MAI = 21, 6, 30


def make_game(*, saved_settings=None):
    g = FakeSF6(DETECTOR)
    if saved_settings is not None:
        g.set_saved_json("SF6MusicTrigger/detector_settings.json", saved_settings)
    g.set_chara(1, JAMIE)
    g.set_chara(2, ZANGIEF)
    g.set_triggers(1, [
        normal_trigger(600),
        super_trigger(1200, 1, 10000),
        super_trigger(1205, 2, 20000),
        super_trigger(1210, 3, 30000),
        super_trigger(1215, 4, 30000, vital_ratio=25),
    ])
    g.set_triggers(2, [super_trigger(1195, 1, 10000), super_trigger(1214, 3, 30000)])
    g.load()
    g.frames(3)
    return g


@pytest.fixture
def game():
    return make_game()


def of_type(game, kind):
    return [e for e in game.lines(EVENTS) if e["type"] == kind]


def ends(game):
    return [(e["side"], e["label"], e["result"], e["reason"]) for e in of_type(game, "super_end")]


def do_super(game, side, act, *, opp_act=None, frames=20, follow_ups=()):
    game.set_act(side, act)
    game.frames(1)
    for follow in follow_ups:
        game.set_act(side, follow)
        game.frames(1)
    if opp_act is not None:
        game.set_act(3 - side, opp_act)
    game.frames(frames)
    game.set_act(side, BAS)
    game.set_act(3 - side, BAS)
    game.frames(2)


def test_every_line_is_versioned_and_sequenced(game):
    game.frames(70)
    lines = game.lines(EVENTS)
    assert lines[0]["type"] == "hello"
    assert {e["v"] for e in lines} == {1}
    assert len({e["session"] for e in lines}) == 1
    assert [e["seq"] for e in lines] == list(range(1, len(lines) + 1))
    beat = of_type(game, "heartbeat")[-1]
    assert beat["battle"] is True and beat["chara"] == [JAMIE, ZANGIEF] and beat["supers"] == [4, 2]
    assert {(e["side"], e["chara"], e["supers"]) for e in of_type(game, "chara")} == {
        (1, JAMIE, 4), (2, ZANGIEF, 2)}


def test_sa3_hit_block_and_whiff(game):
    do_super(game, 1, 1210, opp_act=DMG)
    do_super(game, 1, 1210, opp_act=GRD)
    do_super(game, 1, 1210)
    assert ends(game) == [
        (1, "SA3", "hit", "recovered"),
        (1, "SA3", "block", "recovered"),
        (1, "SA3", "whiff", "recovered"),
    ]
    starts = of_type(game, "super_start")
    assert [(e["level"], e["act"], e["chara"]) for e in starts] == [(3, 1210, JAMIE)] * 3
    assert len({e["episode"] for e in starts}) == 3


def test_combo_ender_hit_is_the_new_reaction_not_the_old_hitstun(game):
    # Real log (Jamie CA, 2026-10-19 ep30): opponent in 232 from the combo,
    # switches to 239 when the CA connects.
    game.set_act(2, 232)
    game.frames(5)
    game.set_act(1, 1215)
    game.frames(1)
    game.frames(90)
    game.set_act(2, 239)
    game.frames(30)
    game.set_act(1, BAS)
    game.set_act(2, BAS)
    game.frames(2)
    assert ends(game) == [(1, "CA", "hit", "recovered")]


def test_dropped_combo_then_blocked_super_is_block(game):
    # Real log ep47/50/51: opponent still in 206 hitstun when the CA starts,
    # recovers and blocks it (163) - this used to be reported as a hit.
    game.set_act(2, 206)
    game.frames(5)
    game.set_act(1, 1215)
    game.frames(1)
    game.frames(92)
    game.set_act(2, 163)
    game.frames(24)
    game.set_act(2, 168)
    game.frames(40)
    game.set_act(2, BAS)
    game.frames(20)
    game.set_act(1, BAS)
    game.frames(2)
    assert ends(game) == [(1, "CA", "block", "recovered")]


def test_dropped_combo_opponent_falling_is_whiff(game):
    # Real log ep46: last combo hit (239) lands just before the CA, the
    # opponent falls (330 knockdown, 340 down) while the CA misses.
    game.set_act(2, 239)
    game.frames(12)
    game.set_act(1, 1215)
    game.frames(1)
    game.frames(89)
    game.set_act(2, 330)
    game.frames(10)
    game.set_act(2, 340)
    game.frames(42)
    game.set_act(2, BAS)
    game.frames(57)
    game.set_act(1, BAS)
    game.frames(2)
    assert ends(game) == [(1, "CA", "whiff", "recovered")]


def test_dropped_combo_opponent_recovering_is_whiff(game):
    game.set_act(2, DMG)
    game.frames(5)
    game.set_act(1, 1210)
    game.frames(30)
    game.set_act(2, BAS)
    game.frames(120)
    game.set_act(1, BAS)
    game.frames(2)
    assert ends(game) == [(1, "SA3", "whiff", "recovered")]


def test_same_reaction_restarting_counts_as_a_new_hit(game):
    game.set_act(2, DMG)
    game.frames(5)
    game.set_act(1, 1210)
    game.frames(60)
    game.fake.state.last_act[1] = None      # same DMG action restarts at frame 0
    game.frames(5)
    game.set_act(1, BAS)
    game.set_act(2, BAS)
    game.frames(2)
    assert ends(game)[-1][2] == "hit"


def test_grab_super_victim_state_is_hit(game):
    do_super(game, 2, 1214, opp_act=THROWN)
    assert ends(game) == [(2, "SA3", "hit", "recovered")]


def test_whiff_while_opponent_attacks_is_unknown_not_whiff(game):
    do_super(game, 1, 1210, opp_act=OPP_ATTACK)
    assert ends(game) == [(1, "SA3", "unknown", "recovered")]


def test_follow_up_actions_and_repeated_trigger_rows_keep_one_episode(game):
    do_super(game, 1, 1205, follow_ups=(1206, 1207, 1205, 1208), opp_act=DMG)
    assert len(of_type(game, "super_start")) == 1
    assert ends(game) == [(1, "SA2", "hit", "recovered")]


def test_zangief_super_below_1200_is_detected(game):
    do_super(game, 2, 1195, follow_ups=(1196,), opp_act=DMG)
    assert ends(game) == [(2, "SA1", "hit", "recovered")]


def test_higher_level_preempts_and_own_damage_interrupts(game):
    game.set_act(1, 1200)
    game.frames(1)
    game.set_act(1, 1210)
    game.frames(1)
    game.set_act(1, DMG)
    game.frames(1)
    assert [e["label"] for e in of_type(game, "super_start")] == ["SA1", "SA3"]
    assert ends(game) == [(1, "SA1", "unknown", "preempted"), (1, "SA3", "unknown", "interrupted")]


def test_episode_times_out(game):
    game.set_act(1, 1210)
    game.frames(1)
    game.set_act(1, 1211)
    game.frames(16 * 60)
    assert ends(game)[-1][3] == "timeout"


def test_character_change_and_leaving_battle_reset_episodes(game):
    game.set_act(1, 1210)
    game.frames(1)
    game.set_chara(1, MAI)
    game.set_triggers(1, [super_trigger(1200, 1, 10000), super_trigger(1220, 3, 30000)])
    game.frames(2)
    assert ends(game) == [(1, "SA3", "unknown", "reset")]
    assert of_type(game, "chara")[-1] == {**of_type(game, "chara")[-1], "side": 1, "chara": MAI, "supers": 2}

    game.set_act(1, 1220)
    game.frames(1)
    game.set_in_battle(False)
    game.frames(70)
    assert ends(game)[-1][:2] == (1, "SA3") and ends(game)[-1][3] == "reset"
    assert of_type(game, "heartbeat")[-1]["battle"] is False


def test_both_sides_are_independent(game):
    game.set_act(1, 1210)
    game.set_act(2, 1214)
    game.frames(1)
    game.set_act(1, BAS)
    game.frames(1)
    game.set_act(2, BAS)
    game.frames(1)
    assert sorted(e["side"] for e in of_type(game, "super_start")) == [1, 2]
    assert [e[0] for e in ends(game)] == [1, 2]


def test_detection_can_be_disabled_from_saved_settings():
    g = make_game(saved_settings={"enabled": False, "overlay": False})
    g.set_act(1, 1210)
    g.frames(70)
    assert of_type(g, "super_start") == []
    assert of_type(g, "heartbeat")[-1]["enabled"] is False


def test_online_match_turns_detection_off(game):
    game.set_act(1, 1210)
    game.frames(1)
    game.set_online(True)
    game.frames(70)
    assert ends(game) == [(1, "SA3", "unknown", "reset")]
    game.set_act(1, BAS)
    game.frames(1)
    game.set_act(1, 1210)
    game.frames(60)
    assert len(of_type(game, "super_start")) == 1          # nothing new while online
    beat = of_type(game, "heartbeat")[-1]
    assert beat["online"] is True and beat["battle"] is False

    game.set_online(False)
    game.set_act(1, BAS)
    game.frames(70)
    game.set_act(1, 1210)
    game.frames(1)
    assert len(of_type(game, "super_start")) == 2
    assert of_type(game, "heartbeat")[-1]["online"] is False
