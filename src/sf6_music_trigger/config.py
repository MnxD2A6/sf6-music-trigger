"""Validated local JSON configuration with atomic persistence."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from numbers import Real
import os
from pathlib import Path, PureWindowsPath
import ntpath
import tempfile
from typing import Any


_ROOT_KEYS = {
    "schema_version",
    "p1",
    "p2",
    "profile_path",
    "capture_fps",
    "detection_backend",
    "game_dir",
    "volume_unlimited",
    "auto_start",
}
_SIDE_KEYS = {
    "enabled",
    "main_track",
    "volume",
    "whiff_track",
    "whiff_volume",
    "play_whiff_sound",
    "whiff_fade_ms",
    "super_scope",
    "fade_non_three",
}
_DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\")
# Volumes are gains: 1.0 is the track's own loudness. Above 1.0 the audio
# port amplifies the samples (with a soft limiter), up to 500 %.
MAX_VOLUME = 5.0


class SuperScope(str, Enum):
    ALL_SUPERS = "ALL_SUPERS"
    SA3_CA_ONLY = "SA3_CA_ONLY"


class DetectionBackend(str, Enum):
    REFRAMEWORK = "reframework"
    SCREEN = "screen"


def _require_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")


def _require_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")


def _path_properties(value: str) -> tuple[PureWindowsPath, bool]:
    windows_text = value.replace("/", "\\")
    if windows_text.startswith(_DEVICE_PREFIXES):
        raise ValueError("Windows device paths are not allowed")

    windows_path = PureWindowsPath(value)
    drive_colon = len(value) >= 2 and value[1] == ":" and value[0].isalpha()
    colon_search = value[2:] if drive_colon else value
    if ":" in colon_search:
        raise ValueError("NTFS alternate data streams are not allowed")

    absolute = windows_path.is_absolute()
    if not absolute and (windows_path.drive or windows_path.root):
        raise ValueError("drive-relative and root-relative paths are not allowed")
    return windows_path, absolute


def _relative_path_lexically_escapes(value: str) -> bool:
    normalized = ntpath.normpath(value)
    return normalized == ".." or normalized.startswith("..\\")


def _validate_profile_path(value: object, root: Path | None = None) -> str:
    _require_string("profile_path", value)
    if not value:
        raise ValueError("profile_path must not be empty")
    windows_text = value.replace("/", "\\")
    final_component = windows_text.rsplit("\\", 1)[-1]
    if windows_text.endswith("\\") or final_component in {".", ".."}:
        raise ValueError("profile_path must identify a file below the user-data root")
    _, absolute = _path_properties(value)
    if absolute:
        raise ValueError("profile_path must be relative")
    if _relative_path_lexically_escapes(value):
        raise ValueError("profile_path must remain inside the user-data root")
    if root is not None:
        _require_contained(root, value, "profile_path")
    return value


def _validate_audio_path(
    value: object,
    root: Path | None = None,
    name: str = "main_track",
) -> str:
    _require_string(name, value)
    if not value:
        return value
    _, absolute = _path_properties(value)
    if absolute:
        return value
    if _relative_path_lexically_escapes(value):
        raise ValueError(f"relative {name} must remain inside the user-data root")
    if root is not None:
        _require_contained(root, value, name)
    return value


def _require_contained(root: Path, relative_value: str, name: str) -> None:
    candidate = (root / relative_value).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must remain inside the user-data root") from error


def _validate_game_dir(value: object) -> str:
    _require_string("game_dir", value)
    if not value:
        return value
    _, absolute = _path_properties(value)
    if not absolute:
        raise ValueError("game_dir must be an absolute folder path")
    return value


def _validate_volume(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("volume must be a real number")
    if isinstance(value, int):
        if value <= 0:
            return 0.0
        if value >= MAX_VOLUME:
            return MAX_VOLUME
    if not math.isfinite(value):
        raise ValueError("volume must be finite")
    return min(MAX_VOLUME, max(0.0, float(value)))


@dataclass(frozen=True)
class SideAudioConfig:
    enabled: bool = True
    main_track: str = ""
    volume: float = 0.18
    whiff_track: str = ""
    whiff_volume: float = 0.18
    play_whiff_sound: bool = False
    whiff_fade_ms: int = 1500
    super_scope: SuperScope = SuperScope.SA3_CA_ONLY
    fade_non_three: bool = False

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if type(self.play_whiff_sound) is not bool:
            raise TypeError("play_whiff_sound must be a boolean")
        if type(self.fade_non_three) is not bool:
            raise TypeError("fade_non_three must be a boolean")
        _require_int("whiff_fade_ms", self.whiff_fade_ms)
        if self.whiff_fade_ms < 0:
            raise ValueError("whiff_fade_ms must not be negative")
        if not isinstance(self.super_scope, SuperScope):
            raise TypeError("super_scope must be SuperScope")
        _validate_audio_path(self.main_track)
        _validate_audio_path(self.whiff_track, name="whiff_track")
        object.__setattr__(self, "volume", _validate_volume(self.volume))
        object.__setattr__(self, "whiff_volume", _validate_volume(self.whiff_volume))


@dataclass(frozen=True)
class AppConfig:
    schema_version: int = 1
    p1: SideAudioConfig = field(default_factory=SideAudioConfig)
    p2: SideAudioConfig = field(default_factory=SideAudioConfig)
    profile_path: str = "profiles/default.json"
    capture_fps: int = 20
    detection_backend: DetectionBackend = DetectionBackend.REFRAMEWORK
    game_dir: str = ""
    # UI preferences: volume sliders go to 500 % instead of 100 %, and the
    # panel starts listening (minimized) as soon as it opens.
    volume_unlimited: bool = False
    auto_start: bool = False

    def __post_init__(self) -> None:
        if type(self.volume_unlimited) is not bool or type(self.auto_start) is not bool:
            raise TypeError("volume_unlimited and auto_start must be booleans")
        if not isinstance(self.detection_backend, DetectionBackend):
            raise TypeError("detection_backend must be DetectionBackend")
        _validate_game_dir(self.game_dir)
        _require_int("schema_version", self.schema_version)
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not isinstance(self.p1, SideAudioConfig) or not isinstance(self.p2, SideAudioConfig):
            raise TypeError("p1 and p2 must be SideAudioConfig values")
        _validate_profile_path(self.profile_path)
        _require_int("capture_fps", self.capture_fps)
        if not 10 <= self.capture_fps <= 30:
            raise ValueError("capture_fps must be between 10 and 30")

    def diagnostic_view(self) -> dict[str, Any]:
        view = asdict(self)
        view["detection_backend"] = self.detection_backend.value
        if view["game_dir"]:
            view["game_dir"] = "<redacted>"
        for side_name in ("p1", "p2"):
            view[side_name]["super_scope"] = getattr(
                view[side_name]["super_scope"],
                "value",
                view[side_name]["super_scope"],
            )
            if view[side_name]["main_track"]:
                view[side_name]["main_track"] = "<redacted>"
            if view[side_name]["whiff_track"]:
                view[side_name]["whiff_track"] = "<redacted>"
        return view


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_unknown_keys(mapping: dict[str, Any], allowed: set[str], scope: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"unknown {scope} field(s): {', '.join(sorted(unknown))}")


def _parse_side(value: object, side_name: str) -> SideAudioConfig:
    if not isinstance(value, dict):
        raise TypeError(f"{side_name} must be an object")
    _reject_unknown_keys(value, _SIDE_KEYS, side_name)
    values = dict(value)
    if "super_scope" in values:
        raw_scope = values["super_scope"]
        if type(raw_scope) is not str:
            raise ValueError(f"{side_name}.super_scope is invalid")
        try:
            values["super_scope"] = SuperScope(raw_scope)
        except ValueError:
            raise ValueError(f"{side_name}.super_scope is invalid") from None
    return SideAudioConfig(**values)


def _validate_config_for_root(config: object, root: Path) -> AppConfig:
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")

    _require_int("schema_version", config.schema_version)
    if config.schema_version != 1:
        raise ValueError(f"unsupported schema_version: {config.schema_version}")
    _require_int("capture_fps", config.capture_fps)
    if not 10 <= config.capture_fps <= 30:
        raise ValueError("capture_fps must be between 10 and 30")
    _validate_profile_path(config.profile_path, root)
    if not isinstance(config.detection_backend, DetectionBackend):
        raise TypeError("detection_backend must be DetectionBackend")
    _validate_game_dir(config.game_dir)

    for side_name, side in (("p1", config.p1), ("p2", config.p2)):
        if not isinstance(side, SideAudioConfig):
            raise TypeError(f"{side_name} must be a SideAudioConfig")
        if type(side.enabled) is not bool:
            raise TypeError(f"{side_name}.enabled must be a boolean")
        if type(side.play_whiff_sound) is not bool:
            raise TypeError(f"{side_name}.play_whiff_sound must be a boolean")
        if type(side.fade_non_three) is not bool:
            raise TypeError(f"{side_name}.fade_non_three must be a boolean")
        _require_int(f"{side_name}.whiff_fade_ms", side.whiff_fade_ms)
        if side.whiff_fade_ms < 0:
            raise ValueError(f"{side_name}.whiff_fade_ms must not be negative")
        if not isinstance(side.super_scope, SuperScope):
            raise TypeError(f"{side_name}.super_scope must be SuperScope")
        _validate_audio_path(side.main_track, root)
        _validate_audio_path(side.whiff_track, root, "whiff_track")
        _validate_volume(side.volume)
        _validate_volume(side.whiff_volume)
    return config


def load_config(path: str | os.PathLike[str]) -> AppConfig:
    target = Path(path)
    root = target.parent.resolve(strict=False)
    if not target.exists():
        return _validate_config_for_root(AppConfig(), root)

    with target.open("r", encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=_reject_json_constant)
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be an object")
    _reject_unknown_keys(payload, _ROOT_KEYS, "configuration")

    values = dict(payload)
    if "p1" in values:
        values["p1"] = _parse_side(values["p1"], "p1")
    if "p2" in values:
        values["p2"] = _parse_side(values["p2"], "p2")
    if "detection_backend" in values:
        try:
            values["detection_backend"] = DetectionBackend(values["detection_backend"])
        except (TypeError, ValueError):
            raise ValueError("detection_backend is invalid") from None
    return _validate_config_for_root(AppConfig(**values), root)


def _serialize_config(config: AppConfig) -> str:
    p1 = asdict(config.p1)
    p2 = asdict(config.p2)
    p1["super_scope"] = config.p1.super_scope.value
    p2["super_scope"] = config.p2.super_scope.value
    payload = {
        "schema_version": config.schema_version,
        "p1": p1,
        "p2": p2,
        "profile_path": config.profile_path,
        "capture_fps": config.capture_fps,
        "detection_backend": config.detection_backend.value,
        "game_dir": config.game_dir,
        "volume_unlimited": config.volume_unlimited,
        "auto_start": config.auto_start,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def save_config(path: str | os.PathLike[str], config: AppConfig) -> None:
    target = Path(path)
    root = target.parent.resolve(strict=False)
    validated = _validate_config_for_root(config, root)
    serialized = _serialize_config(validated)

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
