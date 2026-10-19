"""V2: REFramework backend selection in config, panel and controller."""

import json
from pathlib import Path
import threading

import pytest

from sf6_music_trigger.config import AppConfig, DetectionBackend, load_config, save_config
from sf6_music_trigger.controller import ListenerController
from sf6_music_trigger.engine import EngineUpdate
from sf6_music_trigger.panel import PanelViewModel, ref_runtime_message
from sf6_music_trigger.reframework_events import Connection, install_detector


def test_config_defaults_to_reframework_and_round_trips(tmp_path):
    path = tmp_path / "config.json"
    assert load_config(path).detection_backend is DetectionBackend.REFRAMEWORK
    game = tmp_path / "Street Fighter 6"
    config = AppConfig(detection_backend=DetectionBackend.SCREEN, game_dir=str(game))
    save_config(path, config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["detection_backend"] == "screen" and payload["game_dir"] == str(game)
    assert load_config(path) == config
    assert config.diagnostic_view()["game_dir"] == "<redacted>"


def test_t03_config_without_new_keys_still_loads(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1, "capture_fps": 20}), encoding="utf-8")
    config = load_config(path)
    assert config.detection_backend is DetectionBackend.REFRAMEWORK and config.game_dir == ""


@pytest.mark.parametrize("payload", [{"detection_backend": "camera"}, {"game_dir": "relative\\path"}])
def test_invalid_backend_values_are_rejected(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1, **payload}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def _game(tmp_path: Path, *, reframework: bool = True) -> Path:
    game = tmp_path / "Street Fighter 6"
    game.mkdir()
    (game / "StreetFighter6.exe").write_bytes(b"")
    if reframework:
        (game / "dinput8.dll").write_bytes(b"")
    return game


def _view_model(tmp_path: Path, game: Path) -> PanelViewModel:
    track = tmp_path / "audio" / "p1.wav"
    track.parent.mkdir()
    track.write_bytes(b"audio")
    vm = PanelViewModel(user_dir=tmp_path)
    vm.game_dir = str(game)
    vm.p1_main_track = "audio/p1.wav"
    vm.p2_enabled = False
    return vm


def test_reframework_can_start_checks_game_reframework_script_and_tracks(tmp_path):
    vm = _view_model(tmp_path, tmp_path / "missing")
    assert vm.detection_backend is DetectionBackend.REFRAMEWORK
    assert not vm.can_start and "游戏目录" in vm.validation_message

    game = _game(tmp_path, reframework=False)
    vm.game_dir = str(game)
    assert not vm.can_start and "REFramework" in vm.validation_message

    (game / "dinput8.dll").write_bytes(b"")
    assert not vm.can_start and "一键安装" in vm.validation_message

    install_detector(game)
    assert vm.can_start and vm.validation_message == ""
    assert vm.selected_profile is None  # no HUD profile needed

    vm.p1_main_track = "audio/missing.wav"
    assert not vm.can_start


def test_to_config_carries_backend_and_game_dir(tmp_path):
    game = _game(tmp_path)
    vm = _view_model(tmp_path, game)
    config = vm.to_config()
    assert config.detection_backend is DetectionBackend.REFRAMEWORK and config.game_dir == str(game)


def test_ref_runtime_message_covers_every_connection_state():
    args = dict(p1_enabled=True, p2_enabled=False)
    assert "等待游戏内检测脚本" in ref_runtime_message(None, (None, None), (None, None), **args)
    assert "启动游戏" in ref_runtime_message(Connection.NO_FILE, (None, None), (None, None), **args)
    assert "没有响应" in ref_runtime_message(Connection.WAITING, (None, None), (None, None), **args)
    assert "Detect supers" in ref_runtime_message(Connection.DISABLED, (None, None), (None, None), **args)
    assert "等待进入对战" in ref_runtime_message(Connection.MENU, (None, None), (None, None), **args)
    assert "自动停用" in ref_runtime_message(Connection.ONLINE, (None, None), (None, None), **args)
    battle = ref_runtime_message(Connection.BATTLE, (21, 6), ("命中", None), **args)
    assert battle == "监听中（REFramework）：P1（角色 21）：命中；P2（角色 6）：关闭"


def test_panel_builds_a_working_reframework_listener(tmp_path):
    from sf6_music_trigger.panel import _build_listener_controller, create_panel_application
    from sf6_music_trigger.reframework_events import events_path

    class Audio:
        def __init__(self) -> None:
            self.calls = []
            self.played = threading.Event()

        def execute(self, actions):
            self.calls.append(tuple(a.kind.value for a in actions))
            self.played.set()

        def close(self):
            pass

    game = _game(tmp_path)
    install_detector(game)
    audio = Audio()
    application = create_panel_application(
        config_path=tmp_path / "config.json",
        audio_port_factory=lambda _root: audio,
        config_loader=lambda _path: AppConfig(game_dir=str(game)),
    )
    track = tmp_path / "audio" / "p1.wav"
    track.parent.mkdir()
    track.write_bytes(b"audio")
    application.view_model.p1_main_track = "audio/p1.wav"
    application.view_model.p2_enabled = False
    controller = _build_listener_controller(application, audio_diagnostic=lambda error: None)
    events = events_path(game)
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text("", encoding="utf-8")
    assert controller.start()
    try:
        for _ in range(100):
            if controller.runtime_message is not None:
                break
            threading.Event().wait(0.01)
        line = {"v": 1, "session": "s", "seq": 1, "t": 1.0, "type": "super_start",
                "side": 1, "level": 3, "episode": 1, "chara": 21, "act": 1210}
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line) + "\n")
        assert audio.played.wait(3.0)
    finally:
        controller.close(timeout=3)
    assert audio.calls[0] == ("stop", "play")


class _Source:
    def __init__(self) -> None:
        self.reads = 0
        self.stopped = threading.Event()

    def start(self) -> None:
        pass

    def read(self) -> object:
        self.reads += 1
        if self.stopped.wait(0.01):
            return object()
        return object()

    def stop(self) -> None:
        self.stopped.set()


class _Engine:
    outcome_statuses = (None, None)

    def reset(self, reason: str) -> None:
        pass

    def process(self, batch: object) -> EngineUpdate:
        return EngineUpdate()


class _Port:
    def close(self) -> None:
        pass


def test_controller_exposes_runtime_message():
    messages = iter(["first", "second"] + ["steady"] * 1000)
    controller = ListenerController(
        source=_Source(), engine=_Engine(), audio_port=_Port(),
        runtime_message_provider=lambda: next(messages),
    )
    assert controller.runtime_message is None
    controller.start()
    deadline = threading.Event()
    for _ in range(200):
        if controller.runtime_message == "steady":
            break
        deadline.wait(0.01)
    controller.close(timeout=2)
    assert controller.runtime_message == "steady"
    with pytest.raises(TypeError):
        ListenerController(source=_Source(), engine=_Engine(), audio_port=_Port(), runtime_message_provider=1)
