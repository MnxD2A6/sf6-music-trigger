import json

import pytest

from sf6_music_trigger.domain import Side
from sf6_music_trigger.reframework_events import (
    Connection,
    EventFileTail,
    RefEventSource,
    SuperEnd,
    SuperStart,
    bundled_detector_script,
    detector_installed,
    events_path,
    find_sf6_install,
    install_detector,
    parse_event,
)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def line(seq, kind, session="s1", **fields):
    return json.dumps({"v": 1, "session": session, "seq": seq, "t": seq * 0.1, "type": kind, **fields}) + "\n"


@pytest.fixture
def game_dir(tmp_path):
    (tmp_path / "StreetFighter6.exe").write_bytes(b"")
    events_path(tmp_path).parent.mkdir(parents=True)
    return tmp_path


def append(game_dir, text, mode="a"):
    with events_path(game_dir).open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def make_source(game_dir, clock):
    return RefEventSource(game_dir, clock=clock, sleep=lambda _s: None)


def test_parse_event_validates_protocol_fields():
    assert parse_event({"type": "heartbeat"}) is None
    start = parse_event({"type": "super_start", "side": 2, "level": 4, "episode": 7, "chara": 6, "act": 1215})
    assert start == SuperStart(Side.P2, 4, 7, 6, 1215) and start.label == "CA"
    end = parse_event({"type": "super_end", "side": 1, "level": 3, "episode": 7, "result": "block", "reason": "recovered"})
    assert end == SuperEnd(Side.P1, 3, 7, "block", "recovered")
    with pytest.raises(ValueError):
        parse_event({"type": "super_start", "side": 3, "level": 3, "episode": 1})
    with pytest.raises(ValueError):
        parse_event({"type": "super_end", "side": 1, "level": 3, "episode": 1, "result": "maybe", "reason": "x"})


def test_tail_handles_partial_lines_and_truncation(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"a":1}\n{"b":')
    tail = EventFileTail(path)
    assert tail.read_lines() == [b'{"a":1}']
    with path.open("ab") as stream:
        stream.write(b'2}\n')
    assert tail.read_lines() == [b'{"b":2}']
    path.write_bytes(b'{"c":3}\n')           # truncated by a new game session
    assert tail.read_lines() == [b'{"c":3}']
    path.unlink()
    assert tail.read_lines() == []


def test_source_ignores_history_and_reports_connection(game_dir):
    clock = Clock()
    append(game_dir, line(1, "hello") + line(2, "super_start", side=1, level=3, episode=1))
    source = make_source(game_dir, clock)
    source.start()
    batch = source.read()
    assert batch.events == () and batch.connection is Connection.WAITING

    append(game_dir, line(3, "heartbeat", enabled=True, battle=True, chara=[21, 6], supers=[4, 5]))
    append(game_dir, line(4, "super_start", side=1, level=3, episode=2, chara=21, act=1210))
    batch = source.read()
    assert batch.connection is Connection.BATTLE
    assert batch.chara == (21, 6)
    assert batch.events == (SuperStart(Side.P1, 3, 2, 21, 1210),)

    clock.now += 5
    assert source.read().connection is Connection.WAITING


def test_source_reports_menu_disabled_and_missing_file(game_dir, tmp_path):
    clock = Clock()
    source = make_source(game_dir, clock)
    source.start()
    append(game_dir, line(1, "heartbeat", enabled=True, battle=False, chara=[-1, -1]))
    batch = source.read()
    assert batch.connection is Connection.MENU and batch.chara == (None, None)
    append(game_dir, line(2, "heartbeat", enabled=False, battle=True))
    assert source.read().connection is Connection.DISABLED

    missing = make_source(tmp_path / "nowhere", clock)
    missing.start()
    assert missing.read().connection is Connection.NO_FILE


def test_source_detects_new_session_and_drops_bad_lines(game_dir):
    clock = Clock()
    source = make_source(game_dir, clock)
    source.start()
    append(game_dir, line(1, "hello") + "not json\n" + json.dumps({"v": 99, "type": "hello"}) + "\n")
    append(game_dir, line(2, "super_end", side=1, level=3, episode=1, result="nope", reason="x"))
    batch = source.read()
    assert batch.rejected_lines == 3 and not batch.session_changed

    append(game_dir, line(1, "hello", session="s2") + line(2, "super_start", session="s2", side=2, level=1, episode=1), mode="w")
    batch = source.read()
    assert batch.session_changed
    assert batch.events == (SuperStart(Side.P2, 1, 1, None, None),)


def test_duplicate_sequence_numbers_are_ignored(game_dir):
    source = make_source(game_dir, Clock())
    source.start()
    text = line(5, "super_start", side=1, level=3, episode=1)
    append(game_dir, text + text)
    assert len(source.read().events) == 1


def test_find_sf6_install_reads_steam_library_list(tmp_path):
    steam = tmp_path / "Steam"
    library = tmp_path / "SteamLibrary"
    game = library / "steamapps" / "common" / "Street Fighter 6"
    game.mkdir(parents=True)
    (game / "StreetFighter6.exe").write_bytes(b"")
    (steam / "steamapps").mkdir(parents=True)
    escaped = str(library).replace("\\", "\\\\")
    (steam / "steamapps" / "libraryfolders.vdf").write_text(
        f'"libraryfolders"\n{{\n "0"\n {{\n  "path"  "{escaped}"\n }}\n}}\n', encoding="utf-8"
    )
    assert find_sf6_install([steam]) == game
    assert find_sf6_install([tmp_path / "missing"]) is None


def test_install_detector_requires_game_and_reframework(game_dir, tmp_path):
    with pytest.raises(ValueError):
        install_detector(tmp_path / "not-a-game")
    with pytest.raises(FileNotFoundError):
        install_detector(game_dir)
    (game_dir / "dinput8.dll").write_bytes(b"")
    assert not detector_installed(game_dir)
    target = install_detector(game_dir)
    assert target.read_bytes() == bundled_detector_script().read_bytes()
    assert detector_installed(game_dir)


def test_parse_super_result():
    from sf6_music_trigger.reframework_events import SuperResult

    result = parse_event({"type": "super_result", "side": 2, "level": 3, "episode": 4,
                          "result": "whiff", "reason": "attack_frames_over", "at": 1.9})
    assert result == SuperResult(Side.P2, 3, 4, "whiff", "attack_frames_over")


def test_online_heartbeat_reports_online_connection(game_dir):
    source = make_source(game_dir, Clock())
    source.start()
    append(game_dir, line(1, "heartbeat", enabled=True, battle=False, online=True))
    assert source.read().connection is Connection.ONLINE
