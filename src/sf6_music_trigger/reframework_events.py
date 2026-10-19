"""Read super events written by the in-game REFramework detector.

The Lua detector (game_scripts/sf6_music_trigger.lua) appends one JSON object
per line to ``<SF6>/reframework/data/SF6MusicTrigger/events.jsonl`` and
truncates the file when the game starts or the file grows too large. This
module tails that file without ever touching the game process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable

from .domain import Side

PROTOCOL_VERSION = 1
EVENTS_RELATIVE_PATH = Path("reframework") / "data" / "SF6MusicTrigger" / "events.jsonl"
DETECTOR_RELATIVE_PATH = Path("reframework") / "autorun" / "sf6_music_trigger.lua"
GAME_EXECUTABLE = "StreetFighter6.exe"
HEARTBEAT_TIMEOUT_S = 3.0
_MAX_LINE_BYTES = 64 * 1024

LEVEL_LABELS = {1: "SA1", 2: "SA2", 3: "SA3", 4: "CA"}
RESULTS = frozenset({"hit", "block", "whiff", "unknown"})


class Connection(str, Enum):
    NO_FILE = "no_file"          # detector never ran (not installed / game never started)
    WAITING = "waiting"          # file exists but no recent heartbeat (game closed?)
    DISABLED = "disabled"        # detector running but switched off in the REFramework menu
    ONLINE = "online"            # online match: detection is off (offline use only)
    MENU = "menu"                # game running, not in a battle
    BATTLE = "battle"            # game running, battle state readable


@dataclass(frozen=True)
class SuperStart:
    side: Side
    level: int
    episode: int
    chara: int | None
    act: int | None

    @property
    def label(self) -> str:
        return LEVEL_LABELS[self.level]


@dataclass(frozen=True)
class SuperResult:
    """First decisive result of an episode (detector v0.2+), sent before SuperEnd."""

    side: Side
    level: int
    episode: int
    result: str
    reason: str

    @property
    def label(self) -> str:
        return LEVEL_LABELS[self.level]


@dataclass(frozen=True)
class SuperEnd:
    side: Side
    level: int
    episode: int
    result: str
    reason: str

    @property
    def label(self) -> str:
        return LEVEL_LABELS[self.level]


RefEvent = SuperStart | SuperResult | SuperEnd


@dataclass(frozen=True)
class RefEventBatch:
    """Everything observed since the previous read, stamped on arrival."""

    received_at: float
    events: tuple[RefEvent, ...]
    connection: Connection
    session_changed: bool = False
    chara: tuple[int | None, int | None] = (None, None)
    rejected_lines: int = 0


def events_path(game_dir: str | os.PathLike[str]) -> Path:
    return Path(game_dir) / EVENTS_RELATIVE_PATH


def detector_path(game_dir: str | os.PathLike[str]) -> Path:
    return Path(game_dir) / DETECTOR_RELATIVE_PATH


def bundled_detector_script() -> Path:
    return Path(__file__).resolve().parent / "game_scripts" / "sf6_music_trigger.lua"


def is_game_dir(game_dir: str | os.PathLike[str]) -> bool:
    try:
        return (Path(game_dir) / GAME_EXECUTABLE).is_file()
    except (OSError, TypeError, ValueError):
        return False


GAME_SCRIPTS = ("sf6_music_trigger.lua", "sf6_music_trigger_ui.lua")


def bundled_game_script(name: str) -> Path:
    return Path(__file__).resolve().parent / "game_scripts" / name


def detector_installed(game_dir: str | os.PathLike[str]) -> bool:
    """True when every bundled game script is installed unchanged in the game."""

    autorun = Path(game_dir) / DETECTOR_RELATIVE_PATH.parent
    try:
        return all(
            (autorun / name).is_file()
            and (autorun / name).read_bytes() == bundled_game_script(name).read_bytes()
            for name in GAME_SCRIPTS
        )
    except OSError:
        return False


def install_detector(game_dir: str | os.PathLike[str]) -> Path:
    """Copy the detector and the in-game settings window into ``reframework/autorun``."""

    if not is_game_dir(game_dir):
        raise ValueError("folder does not contain StreetFighter6.exe")
    if not (Path(game_dir) / "dinput8.dll").is_file():
        raise FileNotFoundError("REFramework (dinput8.dll) is not installed in the game folder")
    autorun = Path(game_dir) / DETECTOR_RELATIVE_PATH.parent
    autorun.mkdir(parents=True, exist_ok=True)
    for name in GAME_SCRIPTS:
        target = autorun / name
        temp = target.with_suffix(".lua.tmp")
        temp.write_bytes(bundled_game_script(name).read_bytes())
        os.replace(temp, target)
    return detector_path(game_dir)


def _steam_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            roots.append(Path(winreg.QueryValueEx(key, "SteamPath")[0]))
    except (ImportError, OSError):
        pass
    for env in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env)
        if base:
            roots.append(Path(base) / "Steam")
    return roots


def find_sf6_install(steam_roots: list[Path] | None = None) -> Path | None:
    """Locate the Steam install of Street Fighter 6 from Steam's library list."""

    libraries: list[Path] = []
    for root in steam_roots if steam_roots is not None else _steam_roots():
        libraries.append(root)
        vdf = root / "steamapps" / "libraryfolders.vdf"
        try:
            text = vdf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in re.finditer(r'"path"\s+"([^"]+)"', text):
            libraries.append(Path(match.group(1).replace("\\\\", "\\")))
    for library in libraries:
        candidate = library / "steamapps" / "common" / "Street Fighter 6"
        if is_game_dir(candidate):
            return candidate
    return None


def _parse_side(value: object) -> Side | None:
    if value == 1:
        return Side.P1
    if value == 2:
        return Side.P2
    return None


def _parse_int(value: object) -> int | None:
    return value if type(value) is int else None


def parse_event(record: dict) -> RefEvent | None:
    """Convert one protocol-v1 record into a typed event; ignore anything else."""

    kind = record.get("type")
    if kind not in {"super_start", "super_result", "super_end"}:
        return None
    side = _parse_side(record.get("side"))
    level = _parse_int(record.get("level"))
    episode = _parse_int(record.get("episode"))
    if side is None or level not in LEVEL_LABELS or episode is None:
        raise ValueError("super event is missing side, level or episode")
    if kind == "super_start":
        return SuperStart(side, level, episode, _parse_int(record.get("chara")), _parse_int(record.get("act")))
    result = record.get("result")
    reason = record.get("reason")
    if result not in RESULTS or type(reason) is not str:
        raise ValueError("super result has an invalid result or reason")
    if kind == "super_result":
        return SuperResult(side, level, episode, result, reason)
    return SuperEnd(side, level, episode, result, reason)


class EventFileTail:
    """Incrementally read complete lines from a file that may be truncated."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._offset = 0
        self._pending = b""
        self._identity: tuple[int, int] | None = None

    def seek_to_end(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            self._offset, self._pending, self._identity = 0, b"", None
            return
        self._offset = stat.st_size
        self._pending = b""
        self._identity = (stat.st_dev, stat.st_ino)

    def exists(self) -> bool:
        return self.path.is_file()

    def read_lines(self) -> list[bytes]:
        try:
            stat = self.path.stat()
        except OSError:
            self._offset, self._pending, self._identity = 0, b"", None
            return []
        identity = (stat.st_dev, stat.st_ino)
        if stat.st_size < self._offset or (self._identity is not None and identity != self._identity):
            # Truncated or replaced by a new game session: start over.
            self._offset, self._pending = 0, b""
        self._identity = identity
        if stat.st_size == self._offset:
            return []
        try:
            with self.path.open("rb") as stream:
                stream.seek(self._offset)
                chunk = stream.read(stat.st_size - self._offset)
        except OSError:
            return []
        self._offset += len(chunk)
        data = self._pending + chunk
        lines = data.split(b"\n")
        self._pending = lines.pop()
        if len(self._pending) > _MAX_LINE_BYTES:
            self._pending = b""
        return [line for line in lines if line.strip()]


class RefEventSource:
    """Listener source that turns the detector's event file into batches.

    Provides the ``start``/``read``/``stop`` protocol used by
    :class:`ListenerController`. ``read`` blocks for at most ``poll_s``.
    """

    def __init__(
        self,
        game_dir: str | os.PathLike[str],
        *,
        poll_s: float = 0.01,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        self._tail = EventFileTail(events_path(game_dir))
        self._poll_s = poll_s
        self._clock = clock
        self._stop = threading.Event()
        self._sleep = sleep or self._stop.wait
        self._session: str | None = None
        self._last_seq: int | None = None
        self._last_heartbeat_at: float | None = None
        self._enabled = True
        self._online = False
        self._battle = False
        self._chara: tuple[int | None, int | None] = (None, None)

    def start(self) -> None:
        self._stop.clear()
        # Only events produced after the listener starts may trigger music.
        self._tail.seek_to_end()
        self._session = None
        self._last_seq = None
        self._last_heartbeat_at = None

    def stop(self) -> None:
        self._stop.set()

    def read(self) -> RefEventBatch:
        if not self._stop.is_set():
            self._sleep(self._poll_s)
        lines = self._tail.read_lines()
        now = self._clock()
        events: list[RefEvent] = []
        rejected = 0
        session_changed = False
        for raw in lines:
            try:
                record = json.loads(raw.decode("utf-8"))
                if not isinstance(record, dict) or record.get("v") != PROTOCOL_VERSION:
                    raise ValueError("unsupported protocol")
            except (UnicodeDecodeError, ValueError):
                rejected += 1
                continue
            session = record.get("session")
            seq = record.get("seq")
            if session != self._session:
                if self._session is not None:
                    session_changed = True
                self._session = session
                self._last_seq = None
            if type(seq) is int:
                if self._last_seq is not None and seq <= self._last_seq:
                    continue  # duplicate after a re-read
                self._last_seq = seq
            kind = record.get("type")
            if kind == "heartbeat":
                self._last_heartbeat_at = now
                self._enabled = record.get("enabled", True) is not False
                self._online = record.get("online") is True
                self._battle = record.get("battle") is True
                chara = record.get("chara")
                if isinstance(chara, list) and len(chara) == 2:
                    self._chara = tuple(c if type(c) is int and c >= 0 else None for c in chara)
                continue
            if kind == "hello":
                self._last_heartbeat_at = now
                continue
            try:
                event = parse_event(record)
            except ValueError:
                rejected += 1
                continue
            if event is not None:
                events.append(event)
        return RefEventBatch(
            received_at=now,
            events=tuple(events),
            connection=self._connection(now),
            session_changed=session_changed,
            chara=self._chara,
            rejected_lines=rejected,
        )

    def _connection(self, now: float) -> Connection:
        if not self._tail.exists():
            return Connection.NO_FILE
        if self._last_heartbeat_at is None or now - self._last_heartbeat_at > HEARTBEAT_TIMEOUT_S:
            return Connection.WAITING
        if not self._enabled:
            return Connection.DISABLED
        if self._online:
            return Connection.ONLINE
        return Connection.BATTLE if self._battle else Connection.MENU
