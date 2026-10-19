from pathlib import Path

import pytest

from fake_reframework import FakeSF6, normal_trigger, super_trigger

PROBE = Path(__file__).resolve().parents[2] / "reframework" / "autorun" / "sf6_music_trigger_probe.lua"

BAS, GRD, DMG = 1, 100, 200
RYU, KEN, TERRY = 1, 10, 27


@pytest.fixture
def game():
    g = FakeSF6(PROBE)
    g.define_act_names({"BAS_STD_Loop": BAS, "5000_GRD_STD_START": GRD, "0010_DMG_HL_ST": DMG})
    g.define_int_fields(["vital_new", "gauge_candidate", "pos_x"])
    g.set_chara(1, RYU)
    g.set_chara(2, KEN)
    g.set_triggers(1, [
        normal_trigger(600),
        super_trigger(1200, 1, 10000),
        super_trigger(1212, 2, 20000),
        super_trigger(1233, 3, 30000),
        super_trigger(1238, 4, 30000, vital_ratio=25),
        super_trigger(1239, 4, 30000, vital_ratio=25),
    ])
    g.set_triggers(2, [super_trigger(1200, 1, 10000), super_trigger(1215, 3, 30000)])
    for side in (1, 2):
        g.set_int(side, "vital_new", 10000)
        g.set_int(side, "gauge_candidate", 30000)
    g.load()
    g.frames(3)
    return g


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]


def test_reads_character_and_super_tables_for_both_sides(game):
    events = game.events()
    assert of_type(events, "session")[0]["game"] == "sf6"
    assert {(e["side"], e["chara"]) for e in of_type(events, "chara")} == {(1, RYU), (2, KEN)}
    tables = {e["side"]: e for e in of_type(events, "super_table")}
    assert tables[1]["count"] == 5
    assert tables[2]["count"] == 2
    levels = {row["act"]: row["level"] for row in tables[1]["rows"]}
    assert levels == {1200: 1, 1212: 2, 1233: 3, 1238: 4, 1239: 4}
    assert any(path.startswith("SF6MusicTrigger/schema-") for path in game.dumped_files())


def test_sa3_blocked_then_recovery_reports_guard_and_gauge_diff(game):
    game.set_act(1, 1233)
    game.set_int(1, "gauge_candidate", 0)
    game.frames(1)
    game.set_act(1, 1235)          # character-specific follow-up action, not a trigger
    game.set_act(2, GRD)
    game.set_int(2, "vital_new", 9800)  # chip damage on block
    game.frames(50)
    game.set_act(1, BAS)
    game.set_act(2, BAS)
    game.frames(2)

    events = game.events()
    start = of_type(events, "super_start")
    assert [(e["side"], e["label"], e["act"]) for e in start] == [(1, "SA3", 1233)]
    reaction = of_type(events, "opp_reaction")
    assert [(e["side"], e["class"]) for e in reaction] == [(1, "guard")]
    diff = of_type(events, "super_diff")[0]
    assert diff["self_diff"]["gauge_candidate"] == [30000, 0]
    assert diff["opp_diff"]["vital_new"] == [10000, 9800]
    end = of_type(events, "super_end")
    assert [(e["side"], e["reason"], e["opp_class"]) for e in end] == [(1, "returned_to_basic", "guard")]


def test_p2_super_hit_and_whiff_are_distinguishable(game):
    game.set_act(2, 1200)
    game.frames(1)
    game.set_act(1, DMG)
    game.frames(5)
    game.set_act(2, BAS)
    game.set_act(1, BAS)
    game.frames(2)
    game.set_act(2, 1215)          # whiffed SA3: opponent never reacts
    game.frames(30)
    game.set_act(2, BAS)
    game.frames(2)

    ends = of_type(game.events(), "super_end")
    assert [(e["side"], e["label"], e["opp_class"]) for e in ends] == [(2, "SA1", "damage"), (2, "SA3", "none")]


def test_same_level_follow_up_does_not_retrigger_and_damage_interrupts(game):
    game.set_act(1, 1238)
    game.frames(1)
    game.set_act(1, 1239)          # second CA trigger row, same episode
    game.frames(1)
    game.set_act(1, DMG)           # traded / interrupted
    game.frames(1)

    events = game.events()
    assert [e["label"] for e in of_type(events, "super_start")] == ["CA"]
    assert [e["act"] for e in of_type(events, "super_continue")] == [1239]
    assert of_type(events, "super_end")[0]["reason"] == "interrupted_by_damage"


def test_higher_level_super_preempts_running_episode(game):
    game.set_act(1, 1200)
    game.frames(1)
    game.set_act(1, 1233)
    game.frames(1)

    events = game.events()
    assert [e["label"] for e in of_type(events, "super_start")] == ["SA1", "SA3"]
    assert of_type(events, "super_end")[0]["reason"] == "preempted_by_higher_level"


def test_character_change_rebuilds_super_table(game):
    game.set_chara(1, TERRY)
    game.set_triggers(1, [super_trigger(1200, 1, 10000), super_trigger(1230, 3, 30000)])
    game.frames(2)
    game.set_act(1, 1230)
    game.frames(1)

    events = game.events()
    tables = [e for e in of_type(events, "super_table") if e["side"] == 1]
    assert [t["chara"] for t in tables] == [RYU, TERRY]
    assert of_type(events, "super_start")[-1]["label"] == "SA3"


def test_heartbeat_is_written(game):
    game.frames(70)
    beats = of_type(game.events(), "heartbeat")
    assert beats and beats[-1]["chara"] == [RYU, KEN]
