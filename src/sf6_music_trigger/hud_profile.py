"""Validated HUD calibration profiles with contained local persistence."""

from dataclasses import dataclass
import json
import math
from numbers import Real
import ntpath
import os
from pathlib import Path, PureWindowsPath
import tempfile
from typing import Any

import numpy as np


MAX_PROFILE_BYTES = 64 * 1024

_PROFILE_KEYS = {
    "schema_version",
    "profile_id",
    "calibration",
    "battle_roi",
    "training_hud_roi",
    "p1",
    "p2",
}
_CALIBRATION_KEYS = {
    "client_width",
    "client_height",
    "client_aspect_ratio",
    "hud_scale_setting",
    "language",
    "brightness_setting",
    "hdr_enabled",
    "game_version_note",
}
_ROI_KEYS = {"x0", "y0", "x1", "y1"}
_SIDE_KEYS = {"meter_roi", "references_path"}
_DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\")


def _require_finite_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, int):
        return
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_actual_int(name: str, value: object, *, minimum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_string(name: str, value: object, *, nonblank: bool) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if nonblank and not value.strip():
        raise ValueError(f"{name} must not be blank")


def _validate_roi_values(roi: object, *, name: str) -> None:
    if not isinstance(roi, NormalizedRoi):
        raise TypeError(f"{name} must be NormalizedRoi")
    for field_name, value in (
        ("x0", roi.x0),
        ("y0", roi.y0),
        ("x1", roi.x1),
        ("y1", roi.y1),
    ):
        _require_finite_real(f"{name}.{field_name}", value)
    if not 0 <= roi.x0 < roi.x1 <= 1:
        raise ValueError(f"{name} x coordinates must satisfy 0 <= x0 < x1 <= 1")
    if not 0 <= roi.y0 < roi.y1 <= 1:
        raise ValueError(f"{name} y coordinates must satisfy 0 <= y0 < y1 <= 1")


@dataclass(frozen=True)
class NormalizedRoi:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        _validate_roi_values(self, name="roi")


def _validate_calibration(value: object) -> None:
    if not isinstance(value, CalibrationContext):
        raise TypeError("calibration must be CalibrationContext")
    _require_actual_int("client_width", value.client_width, minimum=1)
    _require_actual_int("client_height", value.client_height, minimum=1)
    _require_finite_real("client_aspect_ratio", value.client_aspect_ratio)
    if value.client_aspect_ratio <= 0:
        raise ValueError("client_aspect_ratio must be positive")
    expected = value.client_width / value.client_height
    try:
        difference = abs(value.client_aspect_ratio - expected)
    except OverflowError as error:
        raise ValueError("client_aspect_ratio is inconsistent with resolution") from error
    if difference > 1e-6:
        raise ValueError("client_aspect_ratio is inconsistent with resolution")
    _require_string("hud_scale_setting", value.hud_scale_setting, nonblank=True)
    _require_string("language", value.language, nonblank=True)
    _require_string("brightness_setting", value.brightness_setting, nonblank=True)
    if type(value.hdr_enabled) is not bool:
        raise TypeError("hdr_enabled must be a boolean")
    _require_string("game_version_note", value.game_version_note, nonblank=False)


@dataclass(frozen=True)
class CalibrationContext:
    client_width: int
    client_height: int
    client_aspect_ratio: float
    hud_scale_setting: str
    language: str
    brightness_setting: str
    hdr_enabled: bool
    game_version_note: str = ""

    def __post_init__(self) -> None:
        _validate_calibration(self)


def _validate_side(value: object, *, name: str) -> None:
    if not isinstance(value, SideHudProfile):
        raise TypeError(f"{name} must be SideHudProfile")
    _validate_roi_values(value.meter_roi, name=f"{name}.meter_roi")
    _require_string(f"{name}.references_path", value.references_path, nonblank=True)


@dataclass(frozen=True)
class SideHudProfile:
    meter_roi: NormalizedRoi
    references_path: str

    def __post_init__(self) -> None:
        _validate_side(self, name="side")


def _validate_profile(value: object) -> None:
    if not isinstance(value, HudProfile):
        raise TypeError("profile must be HudProfile")
    _require_string("profile_id", value.profile_id, nonblank=True)
    _validate_calibration(value.calibration)
    _validate_roi_values(value.battle_roi, name="battle_roi")
    if value.training_hud_roi is not None:
        _validate_roi_values(value.training_hud_roi, name="training_hud_roi")
    _validate_side(value.p1, name="p1")
    _validate_side(value.p2, name="p2")
    _require_actual_int("schema_version", value.schema_version)
    if value.schema_version != 1:
        raise ValueError(f"unsupported schema_version: {value.schema_version}")


@dataclass(frozen=True)
class HudProfile:
    profile_id: str
    calibration: CalibrationContext
    battle_roi: NormalizedRoi
    training_hud_roi: NormalizedRoi | None
    p1: SideHudProfile
    p2: SideHudProfile
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_profile(self)


def _validate_bgr_array(frame_bgr: object, *, name: str) -> np.ndarray:
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if frame_bgr.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} must have uint8 dtype")
    if (
        frame_bgr.ndim != 3
        or frame_bgr.shape[0] == 0
        or frame_bgr.shape[1] == 0
        or frame_bgr.shape[2] != 3
    ):
        raise ValueError(f"{name} must be a nonempty H x W x 3 BGR array")
    return frame_bgr


def crop_normalized(frame_bgr: np.ndarray, roi: NormalizedRoi) -> np.ndarray:
    """Return an owned pixel crop mapped from a normalized ROI."""

    frame = _validate_bgr_array(frame_bgr, name="frame_bgr")
    _validate_roi_values(roi, name="roi")
    height, width = frame.shape[:2]
    x0 = max(0, min(width, math.floor(roi.x0 * width)))
    y0 = max(0, min(height, math.floor(roi.y0 * height)))
    x1 = max(0, min(width, math.ceil(roi.x1 * width)))
    y1 = max(0, min(height, math.ceil(roi.y1 * height)))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("normalized ROI maps to an empty crop")
    return frame[y0:y1, x0:x1].copy(order="C")


def _path_properties(value: str) -> PureWindowsPath:
    windows_text = value.replace("/", "\\")
    if windows_text.startswith(_DEVICE_PREFIXES):
        raise ValueError("Windows device paths are not allowed")

    windows_path = PureWindowsPath(value)
    drive_colon = len(value) >= 2 and value[0].isalpha() and value[1] == ":"
    colon_search = value[2:] if drive_colon else value
    if ":" in colon_search:
        raise ValueError("NTFS alternate data streams are not allowed")
    if windows_path.is_absolute():
        raise ValueError("absolute and UNC paths are not allowed")
    if windows_path.drive or windows_path.root:
        raise ValueError("drive-relative and root-relative paths are not allowed")
    return windows_path


def _lexically_escapes(value: str) -> bool:
    normalized = ntpath.normpath(value.replace("/", "\\"))
    return normalized == ".." or normalized.startswith("..\\")


def _reject_windows_alias_components(value: str, *, field_name: str) -> None:
    for component in value.replace("/", "\\").split("\\"):
        if component not in {".", ".."} and component.endswith((".", " ")):
            raise ValueError(
                f"{field_name} contains a Windows component ending in dot or space"
            )


def resolve_contained_path(
    root: str | os.PathLike[str],
    relative_path: str,
    *,
    field_name: str,
    required_suffix: str | None = None,
) -> Path:
    """Resolve one Windows-classified relative path below an exact root."""

    _require_string(field_name, relative_path, nonblank=True)
    _reject_windows_alias_components(relative_path, field_name=field_name)
    windows_text = relative_path.replace("/", "\\")
    windows_path = _path_properties(relative_path)
    normalized = ntpath.normpath(windows_text)
    final_component = windows_text.rsplit("\\", 1)[-1]
    if (
        windows_text.endswith("\\")
        or final_component in {".", ".."}
        or normalized in {"", "."}
    ):
        raise ValueError(f"{field_name} must identify a file")
    if _lexically_escapes(relative_path):
        raise ValueError(f"{field_name} must remain inside its root")
    if required_suffix is not None:
        _require_string("required_suffix", required_suffix, nonblank=True)
        if windows_path.suffix.casefold() != required_suffix.casefold():
            raise ValueError(
                f"{field_name} must use the {required_suffix} final suffix"
            )

    resolved_root = Path(root).resolve(strict=False)
    candidate = (resolved_root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{field_name} must remain inside its root") from error
    return candidate


def _resolve_reference_pair(profile: HudProfile, profile_target: Path) -> tuple[Path, Path]:
    _validate_profile(profile)
    p1_target = resolve_contained_path(
        profile_target.parent,
        profile.p1.references_path,
        field_name="p1.references_path",
        required_suffix=".npz",
    )
    p2_target = resolve_contained_path(
        profile_target.parent,
        profile.p2.references_path,
        field_name="p2.references_path",
        required_suffix=".npz",
    )
    if p1_target == p2_target:
        raise ValueError("P1 and P2 references must resolve to different paths")
    if p1_target.exists() and p2_target.exists() and os.path.samefile(p1_target, p2_target):
        raise ValueError("P1 and P2 references must not be hardlink aliases")
    return p1_target, p2_target


def _require_exact_mapping(
    value: object, expected_keys: set[str], *, scope: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{scope} must be an object")
    actual = set(value)
    if actual != expected_keys:
        missing = sorted(expected_keys - actual)
        unknown = sorted(actual - expected_keys)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ValueError(f"invalid {scope} fields: {'; '.join(details)}")
    return value


def _roi_from_json(value: object, *, scope: str) -> NormalizedRoi:
    mapping = _require_exact_mapping(value, _ROI_KEYS, scope=scope)
    return NormalizedRoi(
        x0=mapping["x0"],
        y0=mapping["y0"],
        x1=mapping["x1"],
        y1=mapping["y1"],
    )


def _calibration_from_json(value: object) -> CalibrationContext:
    mapping = _require_exact_mapping(
        value, _CALIBRATION_KEYS, scope="calibration"
    )
    return CalibrationContext(
        client_width=mapping["client_width"],
        client_height=mapping["client_height"],
        client_aspect_ratio=mapping["client_aspect_ratio"],
        hud_scale_setting=mapping["hud_scale_setting"],
        language=mapping["language"],
        brightness_setting=mapping["brightness_setting"],
        hdr_enabled=mapping["hdr_enabled"],
        game_version_note=mapping["game_version_note"],
    )


def _side_from_json(value: object, *, scope: str) -> SideHudProfile:
    mapping = _require_exact_mapping(value, _SIDE_KEYS, scope=scope)
    return SideHudProfile(
        meter_roi=_roi_from_json(mapping["meter_roi"], scope=f"{scope}.meter_roi"),
        references_path=mapping["references_path"],
    )


def _profile_from_json(value: object) -> HudProfile:
    mapping = _require_exact_mapping(value, _PROFILE_KEYS, scope="profile")
    training_value = mapping["training_hud_roi"]
    return HudProfile(
        profile_id=mapping["profile_id"],
        calibration=_calibration_from_json(mapping["calibration"]),
        battle_roi=_roi_from_json(mapping["battle_roi"], scope="battle_roi"),
        training_hud_roi=(
            None
            if training_value is None
            else _roi_from_json(training_value, scope="training_hud_roi")
        ),
        p1=_side_from_json(mapping["p1"], scope="p1"),
        p2=_side_from_json(mapping["p2"], scope="p2"),
        schema_version=mapping["schema_version"],
    )


def _json_number(value: Real) -> int | float:
    """Convert a validated real to a built-in JSON numeric scalar."""

    return value if type(value) is int else float(value)


def _roi_to_json(roi: NormalizedRoi) -> dict[str, int | float]:
    return {
        "x0": _json_number(roi.x0),
        "y0": _json_number(roi.y0),
        "x1": _json_number(roi.x1),
        "y1": _json_number(roi.y1),
    }


def _profile_to_json(profile: HudProfile) -> dict[str, object]:
    calibration = profile.calibration
    return {
        "schema_version": profile.schema_version,
        "profile_id": profile.profile_id,
        "calibration": {
            "client_width": calibration.client_width,
            "client_height": calibration.client_height,
            "client_aspect_ratio": _json_number(calibration.client_aspect_ratio),
            "hud_scale_setting": calibration.hud_scale_setting,
            "language": calibration.language,
            "brightness_setting": calibration.brightness_setting,
            "hdr_enabled": calibration.hdr_enabled,
            "game_version_note": calibration.game_version_note,
        },
        "battle_roi": _roi_to_json(profile.battle_roi),
        "training_hud_roi": (
            None
            if profile.training_hud_roi is None
            else _roi_to_json(profile.training_hud_roi)
        ),
        "p1": {
            "meter_roi": _roi_to_json(profile.p1.meter_roi),
            "references_path": profile.p1.references_path,
        },
        "p2": {
            "meter_roi": _roi_to_json(profile.p2.meter_roi),
            "references_path": profile.p2.references_path,
        },
    }


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _atomic_write(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_profile(
    user_data_root: str | os.PathLike[str], profile_path: str
) -> HudProfile:
    target = resolve_contained_path(
        user_data_root, profile_path, field_name="profile_path"
    )
    with target.open("rb") as stream:
        raw = stream.read(MAX_PROFILE_BYTES + 1)
    if len(raw) > MAX_PROFILE_BYTES:
        raise ValueError("profile exceeds 64 KiB")
    text = raw.decode("utf-8", errors="strict")
    payload = json.loads(text, parse_constant=_reject_nonfinite)
    profile = _profile_from_json(payload)
    _resolve_reference_pair(profile, target)
    return profile


def save_profile(
    user_data_root: str | os.PathLike[str],
    profile_path: str,
    profile: HudProfile,
) -> None:
    target = resolve_contained_path(
        user_data_root, profile_path, field_name="profile_path"
    )
    _resolve_reference_pair(profile, target)
    serialized = (
        json.dumps(
            _profile_to_json(profile),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(serialized) > MAX_PROFILE_BYTES:
        raise ValueError("profile exceeds 64 KiB")
    _atomic_write(target, serialized)
