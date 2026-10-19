"""V2 in-game settings bridge (Python side)."""

import json
from pathlib import Path

import pytest

from sf6_music_trigger.config import SuperScope
from sf6_music_trigger.domain import Side
from sf6_music_trigger.game_ui import (
    GameUiBridge,
    UiCommand,
    apply_ui_command,
    build_app_state,
    collect_tracks,
    parse_ui_command,
)
from sf6_music_trigger.panel import PanelViewModel


def cmd(**fields):
    return {"v": 1, "type": "ui", "session": "s", "seq": 1, **fields}


@pytest.mark.parametrize("record, expected", [
    (cmd(cmd="set", side="p1", key="volume", value=120), UiCommand("set", Side.P1, "volume", 120)),
    (cmd(cmd="set", side="p2", key="scope", value="SA3_CA_ONLY"), UiCommand("set", Side.P2, "scope", "SA3_CA_ONLY")),
    (cmd(cmd="set_option", key="volume_unlimited", value=True), UiCommand("set_option", None, "volume_unlimited", True)),
    (cmd(cmd="reset_volume"), UiCommand("reset_volume")),
    (cmd(cmd="preview", side="p1", value="whiff"), UiCommand("preview", Side.P1, None, "whiff")),
    (cmd(cmd="start"), UiCommand("start")),
])
def test_parse_valid_commands(record, expected):
    assert parse_ui_command(record) == expected


@pytest.mark.parametrize("record", [
    {"v": 2, "type": "ui", "cmd": "start"},
    cmd(cmd="explode"),
    cmd(cmd="set", side="p3", key="volume", value=1),
    cmd(cmd="set", side="p1", key="colour", value=1),
    cmd(cmd="set", side="p1", key="enabled", value="yes"),
    cmd(cmd="set", side="p1", key="volume", value="loud"),
    cmd(cmd="set", side="p1", key="scope", value="EVERYTHING"),
    cmd(cmd="set", side="p1", key="whiff_mode", value="scream"),
    cmd(cmd="set_option", key="volume_unlimited", value=1),
    cmd(cmd="apply_template", value=""),
    cmd(cmd="preview", side="p1", value="both"),
])
def test_parse_rejects_bad_commands(record):
    with pytest.raises(ValueError):
        parse_ui_command(record)


@pytest.fixture
def vm(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    for name in ("a.mp3", "b.wav", "notes.txt"):
        (music / name).write_bytes(b"x")
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.mp3").write_bytes(b"x")
    (other / "c.ogg").write_bytes(b"x")
    model = PanelViewModel(user_dir=tmp_path)
    model.p1_main_track = "other/c.ogg"
    model.p2_main_track = "gone/old.wav"
    return model


def test_collect_tracks_lists_configured_folders_and_music(vm):
    tracks = collect_tracks(vm)
    assert tracks["c.ogg"] == "other/c.ogg"
    assert tracks["b.wav"] == "music/b.wav"
    assert {"a.mp3（other）", "a.mp3（music）"} <= set(tracks)
    assert tracks["old.wav"] == "gone/old.wav"          # missing file keeps a name
    assert "notes.txt" not in tracks


def test_apply_commands_update_the_view_model(vm):
    tracks = collect_tracks(vm)
    apply_ui_command(vm, UiCommand("set", Side.P1, "volume", 250), tracks)
    assert vm.p1_volume_percent == 100                  # capped while limited
    apply_ui_command(vm, UiCommand("set_option", None, "volume_unlimited", True), tracks)
    apply_ui_command(vm, UiCommand("set", Side.P1, "volume", 250), tracks)
    apply_ui_command(vm, UiCommand("set", Side.P2, "whiff_volume", 999), tracks)
    assert vm.p1_volume_percent == 250 and vm.p2_whiff_volume_percent == 500
    apply_ui_command(vm, UiCommand("set", Side.P1, "scope", "SA3_CA_ONLY"), tracks)
    apply_ui_command(vm, UiCommand("set", Side.P1, "whiff_mode", "sound"), tracks)
    apply_ui_command(vm, UiCommand("set", Side.P2, "enabled", False), tracks)
    apply_ui_command(vm, UiCommand("set", Side.P1, "whiff_track", "b.wav"), tracks)
    assert vm.p1_super_scope is SuperScope.SA3_CA_ONLY and vm.p1_play_whiff_sound
    assert vm.p2_enabled is False and vm.p1_whiff_track == "music/b.wav"
    assert "找不到" in apply_ui_command(vm, UiCommand("set", Side.P1, "main_track", "nope.mp3"), tracks)

    apply_ui_command(vm, UiCommand("set_option", None, "volume_unlimited", False), tracks)
    assert vm.p1_volume_percent == 100 and vm.p2_whiff_volume_percent == 100
    vm.p1_volume_percent = 40
    apply_ui_command(vm, UiCommand("reset_volume"), tracks)
    assert [vm.p1_volume_percent, vm.p2_volume_percent, vm.p1_whiff_volume_percent] == [100, 100, 100]


def test_app_state_describes_settings_by_display_name(vm):
    vm.set_volume_unlimited(True)
    vm.p1_volume_percent = 150.4
    state = build_app_state(vm, listening=True, status="监听中", tracks=collect_tracks(vm), session="x")
    assert state["v"] == 1 and state["listening"] is True and state["volume_max"] == 500
    assert state["sides"]["p1"]["main_track"] == "c.ogg" and state["sides"]["p1"]["volume"] == 150
    assert state["sides"]["p1"]["whiff_mode"] == "fade"
    assert "b.wav" in state["tracks"] and state["templates"]
    json.dumps(state, ensure_ascii=False)


def test_bridge_skips_history_dedupes_and_publishes(tmp_path):
    bridge = GameUiBridge(tmp_path)
    commands = bridge.data_dir / "ui_commands.jsonl"
    commands.parent.mkdir(parents=True)
    commands.write_text(json.dumps(cmd(cmd="stop")) + "\n", encoding="utf-8")
    bridge.start()
    assert bridge.poll() == []
    with commands.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(cmd(cmd="start", seq=2)) + "\n")
        stream.write(json.dumps(cmd(cmd="start", seq=2)) + "\n")      # duplicate
        stream.write("garbage\n")
        stream.write(json.dumps(cmd(cmd="reset_volume", seq=3)) + "\n")
    assert [c.cmd for c in bridge.poll()] == ["start", "reset_volume"]
    assert bridge.rejected == 1

    assert bridge.publish({"v": 1, "status": "好"})
    assert json.loads((bridge.data_dir / "app_state.json").read_text(encoding="utf-8"))["status"] == "好"
    assert not (bridge.data_dir / "app_state.json.tmp").exists()
