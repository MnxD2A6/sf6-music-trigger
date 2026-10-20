"""V2: separate music for SA1 / SA2 / SA3 / CA (empty = the side's main track)."""

import json

import pytest

from sf6_music_trigger.audio_templates import AudioTemplateSide, load_audio_templates
from sf6_music_trigger.config import AppConfig, SideAudioConfig, SuperScope, load_config, save_config
from sf6_music_trigger.domain import Side
from sf6_music_trigger.game_ui import UiCommand, apply_ui_command, build_app_state, collect_tracks, parse_ui_command
from sf6_music_trigger.panel import PanelViewModel
from sf6_music_trigger.ref_engine import RefTriggerEngine
from sf6_music_trigger.reframework_events import Connection, RefEventBatch, SuperStart


def test_track_for_level_falls_back_to_main():
    config = SideAudioConfig(main_track="main.mp3", sa1_track="sa1.mp3", ca_track="ca.mp3")
    assert [config.track_for_level(level) for level in (1, 2, 3, 4)] == ["sa1.mp3", "main.mp3", "main.mp3", "ca.mp3"]
    assert SideAudioConfig().track_for_level(3) == ""


def test_level_tracks_round_trip_and_old_configs_still_load(tmp_path):
    path = tmp_path / "config.json"
    config = AppConfig(p1=SideAudioConfig(main_track="m.mp3", sa3_track="music/sa3.mp3"))
    save_config(path, config)
    assert json.loads(path.read_text(encoding="utf-8"))["p1"]["sa3_track"] == "music/sa3.mp3"
    assert load_config(path) == config
    path.write_text(json.dumps({"schema_version": 1, "p1": {"main_track": "m.mp3"}}), encoding="utf-8")
    assert load_config(path).p1.ca_track == ""


class Port:
    def __init__(self):
        self.tracks = []

    def execute(self, actions):
        self.tracks += [a.track for a in actions if a.kind.value == "play"]


def test_engine_plays_the_music_of_each_level():
    port = Port()
    config = SideAudioConfig(True, "main.mp3", 0.5, "", 0.5, False, 1500, SuperScope.ALL_SUPERS, False,
                             sa1_track="sa1.mp3", sa2_track="sa2.mp3", ca_track="ca.mp3")
    engine = RefTriggerEngine(p1_audio_config=config, p2_audio_config=SideAudioConfig(enabled=False),
                              audio_port=port, audio_diagnostic=lambda error: None, clock=lambda: 0.0)
    for episode, level in enumerate((1, 2, 3, 4), start=1):
        engine.process(RefEventBatch(0.0, (SuperStart(Side.P1, level, episode, None, None),), Connection.BATTLE))
    assert port.tracks == ["sa1.mp3", "sa2.mp3", "main.mp3", "ca.mp3"]


def test_level_with_no_music_at_all_stays_silent():
    port = Port()
    config = SideAudioConfig(True, "", 0.5, "", 0.5, False, 1500, SuperScope.ALL_SUPERS, False, ca_track="ca.mp3")
    engine = RefTriggerEngine(p1_audio_config=config, p2_audio_config=SideAudioConfig(enabled=False),
                              audio_port=port, audio_diagnostic=lambda error: None, clock=lambda: 0.0)
    engine.process(RefEventBatch(0.0, (SuperStart(Side.P1, 3, 1, None, None),
                                       SuperStart(Side.P1, 4, 2, None, None)), Connection.BATTLE))
    assert port.tracks == ["ca.mp3"]


def test_templates_keep_level_tracks_and_read_old_files(tmp_path):
    path = tmp_path / "audio_templates.json"
    old_side = {"enabled": True, "main_track": "a.mp3", "volume": 0.5, "whiff_track": "", "whiff_volume": 0.5}
    path.write_text(json.dumps({"schema_version": 1, "templates": {"模板 1": {"p1": old_side, "p2": old_side}}}),
                    encoding="utf-8")
    assert load_audio_templates(path)["模板 1"].p1.sa1_track == ""
    assert AudioTemplateSide(main_track="a.mp3", ca_track="ca.mp3").ca_track == "ca.mp3"

    vm = PanelViewModel(user_dir=tmp_path)
    vm.p1_ca_track = "ca.mp3"
    assert vm.add_audio_template("CA 专用")
    vm.p1_ca_track = ""
    assert vm.apply_audio_template("CA 专用") and vm.p1_ca_track == "ca.mp3"


def _game_vm(tmp_path):
    game = tmp_path / "Street Fighter 6"
    game.mkdir()
    (game / "StreetFighter6.exe").write_bytes(b"")
    (game / "dinput8.dll").write_bytes(b"")
    from sf6_music_trigger.reframework_events import install_detector
    install_detector(game)
    music = tmp_path / "music"
    music.mkdir()
    (music / "ca.mp3").write_bytes(b"x")
    (music / "main.mp3").write_bytes(b"x")
    vm = PanelViewModel(user_dir=tmp_path)
    vm.game_dir = str(game)
    vm.p2_enabled = False
    return vm


def test_can_start_with_only_a_level_track_and_reports_missing_ones(tmp_path):
    vm = _game_vm(tmp_path)
    vm.p1_ca_track = "music/ca.mp3"
    assert vm.can_start
    vm.p1_sa1_track = "music/gone.mp3"
    assert not vm.can_start and "SA1" in vm.validation_message


def test_preview_level_uses_its_own_track_or_main(tmp_path):
    played = []

    class Audio:
        def preview(self, track, volume):
            played.append(track)

    vm = PanelViewModel(user_dir=tmp_path, audio_port=Audio())
    vm.p1_main_track = "main.mp3"
    vm.p1_ca_track = "ca.mp3"
    assert vm.preview_level(Side.P1, 4) and vm.preview_level(Side.P1, 3)
    assert played == ["ca.mp3", "main.mp3"]


def test_in_game_commands_set_and_clear_level_tracks(tmp_path):
    vm = _game_vm(tmp_path)
    vm.p1_main_track = "music/main.mp3"
    tracks = collect_tracks(vm)
    command = parse_ui_command({"v": 1, "type": "ui", "cmd": "set", "side": "p1", "key": "ca_track", "value": "ca.mp3"})
    apply_ui_command(vm, command, tracks)
    assert vm.p1_ca_track == "music/ca.mp3"
    state = build_app_state(vm, listening=False, status="", tracks=tracks, session="s")
    assert state["sides"]["p1"]["ca_track"] == "ca.mp3" and state["sides"]["p1"]["sa1_track"] == ""
    apply_ui_command(vm, UiCommand("set", Side.P1, "ca_track", ""), tracks)
    assert vm.p1_ca_track == ""
