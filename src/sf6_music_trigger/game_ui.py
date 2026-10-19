"""In-game settings: bridge between the REFramework settings window and the panel.

The Lua script ``game_scripts/sf6_music_trigger_ui.lua`` draws a settings
window inside SF6 (shown while the REFramework menu is open). It can only write
inside ``reframework/data``, so the two sides talk through two files there:

* ``ui_commands.jsonl`` - commands from the game (Lua appends, we tail it).
* ``app_state.json``   - current settings, track list and status (we replace it
  atomically, Lua re-reads it twice a second).

The panel stays the single owner of the configuration: commands are applied to
its view model, saved, and pushed to a running listener.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PureWindowsPath
import time
from typing import Callable

from .config import SuperScope
from .domain import Side
from .reframework_events import EventFileTail

PROTOCOL_VERSION = 1
DATA_RELATIVE = Path("reframework") / "data" / "SF6MusicTrigger"
STATE_FILE = "app_state.json"
COMMANDS_FILE = "ui_commands.jsonl"
AUDIO_SUFFIXES = (".wav", ".ogg", ".mp3")
MAX_TRACKS = 300
NORMAL_VOLUME_PERCENT = 100
UNLIMITED_VOLUME_PERCENT = 500

_SIDE_NAMES = {"p1": Side.P1, "p2": Side.P2}
_SCOPES = {"ALL_SUPERS": SuperScope.ALL_SUPERS, "SA3_CA_ONLY": SuperScope.SA3_CA_ONLY}
_SIDE_KEYS = {"enabled", "volume", "whiff_volume", "scope", "whiff_mode", "main_track", "whiff_track"}
_COMMANDS = {"set", "set_option", "reset_volume", "apply_template", "start", "stop", "preview"}


@dataclass(frozen=True)
class UiCommand:
    cmd: str
    side: Side | None = None
    key: str | None = None
    value: object = None


def parse_ui_command(record: object) -> UiCommand:
    """Validate one protocol-v1 command line from the game."""

    if not isinstance(record, dict) or record.get("v") != PROTOCOL_VERSION or record.get("type") != "ui":
        raise ValueError("not a v1 ui command")
    cmd = record.get("cmd")
    if cmd not in _COMMANDS:
        raise ValueError("unknown command")
    side = None
    if "side" in record:
        side = _SIDE_NAMES.get(record["side"])
        if side is None:
            raise ValueError("invalid side")
    key, value = record.get("key"), record.get("value")
    if cmd == "set":
        if side is None or key not in _SIDE_KEYS:
            raise ValueError("set needs a side and a known key")
        if key == "enabled" and type(value) is not bool:
            raise ValueError("enabled must be a boolean")
        if key in {"volume", "whiff_volume"} and (type(value) not in (int, float) or value != value):
            raise ValueError("volume must be a number")
        if key == "scope" and value not in _SCOPES:
            raise ValueError("invalid scope")
        if key == "whiff_mode" and value not in {"sound", "fade"}:
            raise ValueError("invalid whiff mode")
        if key in {"main_track", "whiff_track"} and type(value) is not str:
            raise ValueError("track must be a name")
    elif cmd == "set_option":
        if key not in {"volume_unlimited"} or type(value) is not bool:
            raise ValueError("invalid option")
    elif cmd == "apply_template":
        if type(value) is not str or not value:
            raise ValueError("template needs a name")
    elif cmd == "preview":
        if side is None or value not in {"main", "whiff"}:
            raise ValueError("preview needs a side and main/whiff")
    return UiCommand(cmd, side, key, value)


def _resolve(root: Path, stored: str) -> Path | None:
    if type(stored) is not str or not stored.strip():
        return None
    try:
        if PureWindowsPath(stored).is_absolute() or Path(stored).is_absolute():
            return Path(stored)
        return (root / stored).resolve(strict=False)
    except (OSError, ValueError):
        return None


def portable_path(root: Path, target: Path) -> str:
    """Store tracks below the user-data root relative to it, others absolute."""

    resolved = target.resolve(strict=False)
    try:
        return resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return str(resolved)


def collect_tracks(view_model: object) -> dict[str, str]:
    """Audio files offered in the game: the folders of every configured track.

    Returns display name -> stored track path (the panel's own path format).
    """

    root = Path(view_model.user_dir)
    stored = [
        view_model.p1_main_track, view_model.p2_main_track,
        view_model.p1_whiff_track, view_model.p2_whiff_track,
    ]
    folders: list[Path] = []
    for value in stored:
        path = _resolve(root, value)
        if path is not None and path.parent not in folders:
            folders.append(path.parent)
    music = root / "music"
    if music not in folders:
        folders.append(music)
    files: list[Path] = []
    for folder in folders:
        try:
            entries = sorted(folder.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        for entry in entries:
            if entry.suffix.lower() in AUDIO_SUFFIXES and entry.is_file() and entry not in files:
                files.append(entry)
            if len(files) >= MAX_TRACKS:
                break
    names: dict[str, str] = {}
    counts: dict[str, int] = {}
    for path in files:
        counts[path.name] = counts.get(path.name, 0) + 1
    for path in files:
        name = path.name if counts[path.name] == 1 else f"{path.name}（{path.parent.name}）"
        names[name] = portable_path(root, path)
    # Configured tracks that no longer exist still need a stable name.
    for value in stored:
        if value and value not in names.values():
            names[PureWindowsPath(value.replace("/", "\\")).name or value] = value
    return names


def _display_name(tracks: dict[str, str], stored: str) -> str:
    for name, value in tracks.items():
        if value == stored:
            return name
    return ""


def volume_limit_percent(view_model: object) -> int:
    return UNLIMITED_VOLUME_PERCENT if view_model.volume_unlimited else NORMAL_VOLUME_PERCENT


def build_app_state(
    view_model: object,
    *,
    listening: bool,
    status: str,
    tracks: dict[str, str],
    session: str,
) -> dict:
    sides = {}
    for name, side in _SIDE_NAMES.items():
        prefix = side.value.lower()
        sides[name] = {
            "enabled": bool(getattr(view_model, f"{prefix}_enabled")),
            "main_track": _display_name(tracks, getattr(view_model, f"{prefix}_main_track")),
            "volume": round(float(getattr(view_model, f"{prefix}_volume_percent"))),
            "scope": getattr(view_model, f"{prefix}_super_scope").value,
            "whiff_mode": "sound" if getattr(view_model, f"{prefix}_play_whiff_sound") else "fade",
            "whiff_track": _display_name(tracks, getattr(view_model, f"{prefix}_whiff_track")),
            "whiff_volume": round(float(getattr(view_model, f"{prefix}_whiff_volume_percent"))),
        }
    return {
        "v": PROTOCOL_VERSION,
        "session": session,
        "updated": time.time(),
        "listening": bool(listening),
        "status": status,
        "volume_max": volume_limit_percent(view_model),
        "volume_unlimited": bool(view_model.volume_unlimited),
        "sides": sides,
        "tracks": list(tracks),
        "templates": list(view_model.audio_template_names),
    }


def _clamp_percent(view_model: object, value: float) -> float:
    return float(min(max(value, 0), volume_limit_percent(view_model)))


def apply_ui_command(view_model: object, command: UiCommand, tracks: dict[str, str]) -> str:
    """Apply a settings command to the panel view model; returns a status line.

    ``start``, ``stop`` and ``preview`` need the GUI and are left to the caller.
    """

    if command.cmd == "set":
        prefix = command.side.value.lower()
        key, value = command.key, command.value
        if key == "enabled":
            setattr(view_model, f"{prefix}_enabled", value)
        elif key in {"volume", "whiff_volume"}:
            attr = f"{prefix}_volume_percent" if key == "volume" else f"{prefix}_whiff_volume_percent"
            setattr(view_model, attr, _clamp_percent(view_model, value))
        elif key == "scope":
            setattr(view_model, f"{prefix}_super_scope", _SCOPES[value])
        elif key == "whiff_mode":
            view_model.set_whiff_sound_mode(command.side, value == "sound")
        else:
            stored = tracks.get(value)
            if stored is None:
                return f"游戏内设置：找不到音频 {value}"
            setattr(view_model, f"{prefix}_{key}", stored)
        return f"游戏内设置已更新：{command.side.value}"
    if command.cmd == "set_option":
        view_model.set_volume_unlimited(command.value)
        return "游戏内设置：已解除音量上限" if command.value else "游戏内设置：音量上限恢复为 100%"
    if command.cmd == "reset_volume":
        view_model.reset_volumes()
        return "游戏内设置：音量已重置为 100%"
    if command.cmd == "apply_template":
        view_model.apply_audio_template(command.value)
        return view_model.validation_message
    return ""


class GameUiBridge:
    """File transport for the in-game settings window of one game folder."""

    def __init__(self, game_dir: str | os.PathLike[str], *, clock: Callable[[], float] = time.time) -> None:
        self.data_dir = Path(game_dir) / DATA_RELATIVE
        self._tail = EventFileTail(self.data_dir / COMMANDS_FILE)
        self._session: str | None = None
        self._last_seq: int | None = None
        self._clock = clock
        self._last_published: str | None = None
        self.rejected = 0

    def start(self) -> None:
        """Ignore commands written before the panel opened."""

        self._tail.seek_to_end()

    def poll(self) -> list[UiCommand]:
        commands = []
        for raw in self._tail.read_lines():
            try:
                record = json.loads(raw.decode("utf-8"))
                session, seq = record.get("session"), record.get("seq")
                if session != self._session:
                    self._session, self._last_seq = session, None
                if type(seq) is int:
                    if self._last_seq is not None and seq <= self._last_seq:
                        continue
                    self._last_seq = seq
                commands.append(parse_ui_command(record))
            except (UnicodeDecodeError, ValueError, AttributeError):
                self.rejected += 1
        return commands

    def publish(self, state: dict) -> bool:
        """Atomically replace app_state.json; returns False if the folder is unusable."""

        text = json.dumps(state, ensure_ascii=False)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            temp = self.data_dir / (STATE_FILE + ".tmp")
            temp.write_text(text, encoding="utf-8")
            os.replace(temp, self.data_dir / STATE_FILE)
        except OSError:
            return False
        self._last_published = text
        return True
