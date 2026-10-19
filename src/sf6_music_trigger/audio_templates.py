"""Small, local-only audio template persistence for the control panel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from .config import SideAudioConfig


DEFAULT_AUDIO_TEMPLATE_NAMES = ("模板 1", "模板 2")
_ROOT_KEYS = {"schema_version", "templates"}
_TEMPLATE_KEYS = {"p1", "p2"}
_SIDE_KEYS = {"enabled", "main_track", "volume", "whiff_track", "whiff_volume"}


@dataclass(frozen=True)
class AudioTemplateSide:
    enabled: bool = True
    main_track: str = ""
    volume: float = 0.18
    whiff_track: str = ""
    whiff_volume: float = 0.18

    def __post_init__(self) -> None:
        validated = SideAudioConfig(
            enabled=self.enabled,
            main_track=self.main_track,
            volume=self.volume,
            whiff_track=self.whiff_track,
            whiff_volume=self.whiff_volume,
        )
        object.__setattr__(self, "volume", validated.volume)
        object.__setattr__(self, "whiff_volume", validated.whiff_volume)


@dataclass(frozen=True)
class AudioTemplate:
    p1: AudioTemplateSide
    p2: AudioTemplateSide

    def __post_init__(self) -> None:
        if not isinstance(self.p1, AudioTemplateSide):
            raise TypeError("p1 must be an AudioTemplateSide")
        if not isinstance(self.p2, AudioTemplateSide):
            raise TypeError("p2 must be an AudioTemplateSide")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    if set(value) != keys:
        raise ValueError(f"{label} has invalid fields")
    return value


def _parse_side(value: object, label: str) -> AudioTemplateSide:
    payload = _require_exact_keys(value, _SIDE_KEYS, label)
    return AudioTemplateSide(**payload)


def _validated_templates(
    templates: Mapping[str, AudioTemplate],
) -> dict[str, AudioTemplate]:
    if not isinstance(templates, Mapping):
        raise TypeError("templates must be a mapping")
    validated: dict[str, AudioTemplate] = {}
    for name, template in templates.items():
        if type(name) is not str or not name.strip() or len(name) > 64:
            raise ValueError("template names must be nonblank and at most 64 characters")
        if not isinstance(template, AudioTemplate):
            raise TypeError("template values must be AudioTemplate instances")
        validated[name] = template
    return validated


def _normalized_template_name(name: object) -> str:
    if type(name) is not str:
        raise TypeError("template name must be a string")
    normalized = name.strip()
    if not normalized or len(normalized) > 64:
        raise ValueError("template names must be nonblank and at most 64 characters")
    return normalized


def _find_casefold_name(templates: Mapping[str, AudioTemplate], name: str) -> str | None:
    wanted = name.casefold()
    return next((existing for existing in templates if existing.casefold() == wanted), None)


def load_audio_templates(path: str | os.PathLike[str]) -> dict[str, AudioTemplate]:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as stream:
        root = json.load(stream, parse_constant=_reject_json_constant)
    payload = _require_exact_keys(root, _ROOT_KEYS, "audio template file")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported audio template schema")
    raw_templates = payload["templates"]
    if not isinstance(raw_templates, dict):
        raise TypeError("templates must be an object")
    templates: dict[str, AudioTemplate] = {}
    for name, raw_template in raw_templates.items():
        if type(name) is not str or not name.strip() or len(name) > 64:
            raise ValueError("template names must be nonblank and at most 64 characters")
        template = _require_exact_keys(raw_template, _TEMPLATE_KEYS, name)
        templates[name] = AudioTemplate(
            p1=_parse_side(template["p1"], f"{name}.p1"),
            p2=_parse_side(template["p2"], f"{name}.p2"),
        )
    return templates


def save_audio_templates(
    path: str | os.PathLike[str], templates: Mapping[str, AudioTemplate]
) -> None:
    target = Path(path)
    validated = _validated_templates(templates)
    payload = {
        "schema_version": 1,
        "templates": {
            name: {"p1": asdict(template.p1), "p2": asdict(template.p2)}
            for name, template in validated.items()
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
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


def add_audio_template(
    path: str | os.PathLike[str], name: str, template: AudioTemplate
) -> str:
    normalized = _normalized_template_name(name)
    if not isinstance(template, AudioTemplate):
        raise TypeError("template must be an AudioTemplate")
    templates = load_audio_templates(path)
    if _find_casefold_name(templates, normalized) is not None:
        raise ValueError("audio template already exists")
    templates[normalized] = template
    save_audio_templates(path, templates)
    return normalized


def rename_audio_template(
    path: str | os.PathLike[str], old_name: str, new_name: str
) -> str:
    old_normalized = _normalized_template_name(old_name)
    new_normalized = _normalized_template_name(new_name)
    templates = load_audio_templates(path)
    if old_normalized not in templates:
        raise KeyError(old_normalized)
    duplicate = _find_casefold_name(templates, new_normalized)
    if duplicate is not None and duplicate != old_normalized:
        raise ValueError("audio template already exists")
    renamed = {
        (new_normalized if name == old_normalized else name): template
        for name, template in templates.items()
    }
    save_audio_templates(path, renamed)
    return new_normalized
