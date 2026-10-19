"""Replay real action timelines from the 2026-10-19 live test through the detector.

The fixture holds the action-id changes the probe logged for both sides around
seven Jamie CAs: clean hits, CAs blocked or whiffed after a dropped combo
(previously misreported as hits) and a plain whiff.
"""

import json
from pathlib import Path

import pytest

from fake_reframework import FakeSF6, super_trigger

ROOT = Path(__file__).resolve().parents[2]
DETECTOR = ROOT / "src" / "sf6_music_trigger" / "game_scripts" / "sf6_music_trigger.lua"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "real_dropped_combos.json"
EVENTS = "SF6MusicTrigger/events.jsonl"
EPISODES = json.loads(FIXTURE.read_text(encoding="utf-8"))["episodes"]

# Jamie (21) frame data from MMDK: CA start action strikes at 93-96 and 109-112,
# 1216 is the hit cinematic, SA1/SA3 for completeness.
JAMIE_TRIGGERS = [super_trigger(1200, 1, 10000), super_trigger(1205, 2, 20000),
                  super_trigger(1211, 3, 30000), super_trigger(1215, 4, 30000)]
JAMIE_FAB = {1211: {"attack": [(93, 96), (109, 112)]}, 1215: {"attack": [(93, 96), (109, 112)]},
             1212: {}, 1216: {}}


@pytest.mark.parametrize("episode", EPISODES, ids=lambda e: f"ep{e['episode']}-{e['expected']}")
def test_real_timeline_gives_the_true_result(episode):
    game = FakeSF6(DETECTOR)
    game.set_chara(1, episode["chara"])
    game.set_chara(2, 16)
    game.set_triggers(1, JAMIE_TRIGGERS)
    game.set_fab(1, JAMIE_FAB)
    game.set_act(1, episode["initial"]["1"])
    game.set_act(2, episode["initial"]["2"])
    game.load()
    game.frames(70)                       # super table and frame data are read

    changes = sorted(episode["timeline"], key=lambda c: c["t"])
    total = int(episode["duration"] * 60) + 2
    index = 0
    for frame in range(total):
        while index < len(changes) and changes[index]["t"] * 60 <= frame:
            game.set_act(changes[index]["side"], changes[index]["act"])
            index += 1
        game.frames(1)

    results = [e for e in game.lines(EVENTS) if e["type"] == "super_result"]
    assert results, "the CA was not detected"
    assert results[0]["result"] == episode["expected"]
    if episode["expected"] in {"whiff", "block"}:
        assert results[0]["at"] < 2.3      # decided early, not at recovery
