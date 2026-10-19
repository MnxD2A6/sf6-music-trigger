import json
import sys
from pathlib import Path

from fake_reframework import FakeSF6, super_trigger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_probe_log  # noqa: E402

PROBE = ROOT / "reframework" / "autorun" / "sf6_music_trigger_probe.lua"
BAS, GRD, DMG = 1, 100, 200


def run_session(tmp_path: Path) -> Path:
    g = FakeSF6(PROBE)
    g.define_act_names({"BAS_STD_Loop": BAS, "5000_GRD_STD_START": GRD, "0010_DMG_HL_ST": DMG})
    g.define_int_fields(["vital_new", "gauge_candidate", "pos_x"])
    g.set_chara(1, 1)
    g.set_chara(2, 10)
    g.set_triggers(1, [super_trigger(1200, 1, 10000), super_trigger(1233, 3, 30000)])
    g.set_triggers(2, [super_trigger(1200, 1, 10000)])
    for side in (1, 2):
        g.set_int(side, "vital_new", 10000)
        g.set_int(side, "gauge_candidate", 30000)
    g.load()
    g.frames(3)

    # P1 SA1 on hit: gauge 30000 -> 20000, opponent vital drops.
    g.set_act(1, 1200)
    g.set_int(1, "gauge_candidate", 20000)
    g.set_int(1, "pos_x", 5)
    g.frames(1)
    g.set_act(2, DMG)
    g.set_int(2, "vital_new", 8000)
    g.frames(50)
    g.set_act(1, BAS)
    g.set_act(2, BAS)
    g.frames(2)

    # P1 SA3 blocked: wait, gauge 20000 is not enough in the real game, but the
    # analyzer only needs the drop to equal the trigger cost.
    g.set_int(1, "gauge_candidate", 30000)
    g.frames(2)
    g.set_act(1, 1233)
    g.set_int(1, "gauge_candidate", 0)
    g.frames(1)
    g.set_act(2, GRD)
    g.frames(50)
    g.set_act(1, BAS)
    g.set_act(2, BAS)
    g.frames(2)

    path = tmp_path / "probe.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in g.events()) + "\n", encoding="utf-8")
    return path


def test_summary_finds_gauge_and_vital_fields(tmp_path):
    summary = analyze_probe_log.summarise(analyze_probe_log.load_events(run_session(tmp_path)))

    assert summary["charas"] == {1: [1], 2: [10]}
    assert [(ep["label"], ep["opp_class"], ep["end_reason"]) for ep in summary["episodes"]] == [
        ("SA1", "damage", "returned_to_basic"),
        ("SA3", "guard", "returned_to_basic"),
    ]
    assert summary["gauge_candidates"][0] == ("gauge_candidate", 2)
    assert summary["vital_candidates"][0] == ("vital_new", 1)
    assert summary["errors"] == []
    report = analyze_probe_log.render(summary)
    assert "Super-gauge field candidates" in report


def test_corrupt_lines_are_reported_not_fatal(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"type": "session", "probe": "x", "t": 0, "seq": 1}\n{not json\n', encoding="utf-8")
    summary = analyze_probe_log.summarise(analyze_probe_log.load_events(path))
    assert summary["counts"]["corrupt_line"] == 1
