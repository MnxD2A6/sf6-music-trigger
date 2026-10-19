"""Lua detector (fake SF6) -> events.jsonl -> RefEventSource -> RefTriggerEngine -> audio."""

from pathlib import Path

from fake_reframework import FakeSF6, normal_trigger, super_trigger

from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.ref_engine import RefTriggerEngine
from sf6_music_trigger.reframework_events import Connection, RefEventSource, events_path

DETECTOR = (
    Path(__file__).resolve().parents[2]
    / "src" / "sf6_music_trigger" / "game_scripts" / "sf6_music_trigger.lua"
)
LUA_EVENTS = "SF6MusicTrigger/events.jsonl"
BAS, GRD, DMG = 1, 100, 200


class Port:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, actions):
        self.calls.append(tuple((a.kind.value, a.side.value, a.track) for a in actions))


class Rig:
    def __init__(self, tmp_path: Path, p1_scope=SuperScope.ALL_SUPERS, p2_scope=SuperScope.ALL_SUPERS):
        self.game = FakeSF6(DETECTOR)
        self.game.set_chara(1, 21)
        self.game.set_chara(2, 6)
        self.game.set_triggers(1, [normal_trigger(600), super_trigger(1200, 1, 10000),
                                   super_trigger(1210, 3, 30000)])
        self.game.set_triggers(2, [super_trigger(1214, 3, 30000), super_trigger(1220, 4, 30000)])
        self.game.load()
        self.game_dir = tmp_path
        events_path(tmp_path).parent.mkdir(parents=True)
        self.sync()
        clock = lambda: float(self.game.fake.now)
        self.source = RefEventSource(tmp_path, clock=clock, sleep=lambda _s: None)
        self.port = Port()
        self.engine = RefTriggerEngine(
            p1_audio_config=SideAudioConfig(True, "p1.wav", 0.5, "", 0.5, False, 1500, p1_scope, False),
            p2_audio_config=SideAudioConfig(True, "p2.wav", 0.5, "", 0.5, False, 1500, p2_scope, False),
            audio_port=self.port,
            audio_diagnostic=lambda error: (_ for _ in ()).throw(error),
            clock=clock,
        )
        self.source.start()

    def sync(self) -> None:
        lines = self.game.fake.files[LUA_EVENTS]
        text = "".join(lines[i] for i in range(1, len(lines) + 1)) if lines is not None else ""
        events_path(self.game_dir).write_text(text, encoding="utf-8", newline="\n")

    def run(self, frames: int) -> None:
        for _ in range(frames):
            self.game.frames(1)
            self.sync()
            self.engine.process(self.source.read())

    def act(self, side: int, act: int) -> None:
        self.game.set_act(side, act)


def test_full_pipeline_plays_fades_and_keeps_music(tmp_path):
    rig = Rig(tmp_path)
    rig.run(70)
    assert rig.engine.connection is Connection.BATTLE
    assert rig.engine.chara == (21, 6)
    assert rig.port.calls == []

    # P1 SA3 blocked: plays, then fades once 4 s have passed.
    rig.act(1, 1210)
    rig.run(1)
    assert rig.port.calls == [(("stop", "P1", None), ("play", "P1", "p1.wav"))]
    rig.act(2, GRD)
    rig.run(30)
    rig.act(1, BAS)
    rig.act(2, BAS)
    rig.run(10)
    assert rig.engine.outcome_statuses[0] == "等待淡出"
    rig.run(4 * 60)
    assert rig.port.calls[-1] == (("fade", "P1", None),)

    # P2 CA lands as a combo ender: the opponent's hit reaction changes when
    # the CA connects, music keeps playing.
    calls_before = len(rig.port.calls)
    rig.act(1, DMG)
    rig.run(2)
    rig.act(2, 1220)
    rig.run(30)
    rig.act(1, DMG + 7)
    rig.run(30)
    rig.act(2, BAS)
    rig.act(1, BAS)
    rig.run(6 * 60)
    assert rig.port.calls[calls_before:] == [(("stop", "P2", None), ("play", "P2", "p2.wav"))]
    assert rig.engine.outcome_statuses[1] == "命中"


def test_sa3_ca_only_ignores_sa1_and_history_before_start(tmp_path):
    rig = Rig(tmp_path, p1_scope=SuperScope.SA3_CA_ONLY)
    rig.act(1, 1200)
    rig.run(30)
    rig.act(1, BAS)
    rig.run(5)
    assert rig.port.calls == []
    assert rig.engine.outcome_statuses[0] == "低等级已过滤"

    # A later source start must not replay the SA1 already in the file.
    rig.source.start()
    rig.run(5)
    assert rig.port.calls == []


def test_game_restart_truncates_file_and_listener_follows(tmp_path):
    rig = Rig(tmp_path)
    rig.run(70)
    # Simulate the game restarting: a fresh detector session truncates the file.
    rig.game = FakeSF6(DETECTOR)
    rig.game.set_chara(1, 21)
    rig.game.set_triggers(1, [super_trigger(1210, 3, 30000)])
    rig.game.load()
    rig.run(70)
    rig.act(1, 1210)
    rig.run(1)
    assert rig.port.calls == [(("stop", "P1", None), ("play", "P1", "p1.wav"))]


def test_whiff_sound_plays_right_after_the_attack_frames(tmp_path):
    rig = Rig.__new__(Rig)
    rig.game = FakeSF6(DETECTOR)
    rig.game.set_chara(1, 21)
    rig.game.set_chara(2, 6)
    rig.game.set_triggers(1, [super_trigger(1210, 3, 30000)])
    rig.game.set_fab(1, {1210: {"attack": [(93, 96), (109, 112)]}})
    rig.game.load()
    rig.game_dir = tmp_path
    events_path(tmp_path).parent.mkdir(parents=True)
    rig.sync()
    clock = lambda: float(rig.game.fake.now)
    rig.source = RefEventSource(tmp_path, clock=clock, sleep=lambda _s: None)
    rig.port = Port()
    rig.engine = RefTriggerEngine(
        p1_audio_config=SideAudioConfig(True, "p1.wav", 0.5, "whiff.wav", 0.5, True, 1500,
                                        SuperScope.SA3_CA_ONLY, False),
        p2_audio_config=SideAudioConfig(False, "", 0.5, "", 0.5, False, 1500, SuperScope.SA3_CA_ONLY, False),
        audio_port=rig.port,
        audio_diagnostic=lambda error: (_ for _ in ()).throw(error),
        clock=clock,
    )
    rig.source.start()
    rig.run(70)

    started = rig.game.fake.now
    rig.act(1, 1210)
    while len(rig.port.calls) < 2 and rig.game.fake.now - started < 5:
        rig.run(1)
    assert rig.port.calls[1] == (("stop", "P1", None), ("play", "P1", "whiff.wav"))
    assert rig.game.fake.now - started < 2.2
