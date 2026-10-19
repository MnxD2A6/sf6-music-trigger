from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from sf6_music_trigger.hud_profile import (
    MAX_PROFILE_BYTES,
    CalibrationContext,
    HudProfile,
    NormalizedRoi,
    SideHudProfile,
    crop_normalized,
    load_profile,
    resolve_contained_path,
    save_profile,
)


ROI = NormalizedRoi(0.0, 0.0, 1.0, 1.0)


def sample_context() -> CalibrationContext:
    return CalibrationContext(
        client_width=1920,
        client_height=1080,
        client_aspect_ratio=1920 / 1080,
        hud_scale_setting="100%",
        language="zh-CN",
        brightness_setting="default",
        hdr_enabled=False,
        game_version_note="phase-1 fixture",
    )


def sample_profile(
    *, training_hud_roi: NormalizedRoi | None = None
) -> HudProfile:
    return HudProfile(
        profile_id="fixture",
        calibration=sample_context(),
        battle_roi=ROI,
        training_hud_roi=training_hud_roi,
        p1=SideHudProfile(ROI, "refs/p1.npz"),
        p2=SideHudProfile(ROI, "refs/p2.npz"),
    )


def roi_payload(roi: NormalizedRoi = ROI) -> dict[str, float]:
    return {"x0": roi.x0, "y0": roi.y0, "x1": roi.x1, "y1": roi.y1}


def profile_payload(
    *, training_hud_roi: NormalizedRoi | None = None
) -> dict[str, object]:
    context = sample_context()
    return {
        "schema_version": 1,
        "profile_id": "fixture",
        "calibration": {
            "client_width": context.client_width,
            "client_height": context.client_height,
            "client_aspect_ratio": context.client_aspect_ratio,
            "hud_scale_setting": context.hud_scale_setting,
            "language": context.language,
            "brightness_setting": context.brightness_setting,
            "hdr_enabled": context.hdr_enabled,
            "game_version_note": context.game_version_note,
        },
        "battle_roi": roi_payload(),
        "training_hud_roi": (
            None if training_hud_roi is None else roi_payload(training_hud_roi)
        ),
        "p1": {"meter_roi": roi_payload(), "references_path": "refs/p1.npz"},
        "p2": {"meter_roi": roi_payload(), "references_path": "refs/p2.npz"},
    }


def write_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )


def nested_mapping(payload: dict[str, object], path: tuple[str, ...]) -> dict:
    current: object = payload
    for component in path:
        assert isinstance(current, dict)
        current = current[component]
    assert isinstance(current, dict)
    return current


def replace_nested(
    payload: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    parent = nested_mapping(payload, path[:-1])
    parent[path[-1]] = value


def create_directory_link_or_skip(link: Path, target: Path) -> str:
    try:
        link.symlink_to(target, target_is_directory=True)
        return "symlink"
    except (NotImplementedError, OSError) as symlink_error:
        if os.name != "nt":
            pytest.skip(f"host does not permit directory links: {symlink_error}")

        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip(
                "host permits neither directory symlinks nor unprivileged "
                f"junctions: {symlink_error}; {result.stderr or result.stdout}"
            )
        return "junction"


def test_roi_context_side_and_profile_values_are_frozen_and_have_safe_comparison():
    context = sample_context()
    side = SideHudProfile(ROI, "refs/p1.npz")
    profile = sample_profile()

    assert ROI == NormalizedRoi(0.0, 0.0, 1.0, 1.0)
    assert ROI != NormalizedRoi(0.0, 0.0, 0.5, 1.0)
    assert context == sample_context()
    assert side == SideHudProfile(ROI, "refs/p1.npz")
    assert profile == sample_profile()

    for value, field_name, replacement in (
        (ROI, "x0", 0.1),
        (context, "client_width", 1280),
        (side, "references_path", "refs/other.npz"),
        (profile, "profile_id", "other"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field_name, replacement)
        with pytest.raises(TypeError):
            value < value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x0", -0.01),
        ("x0", True),
        ("x0", float("nan")),
        ("x0", float("inf")),
        ("x0", "0"),
        ("x0", object()),
        ("y0", -1),
        ("y0", False),
        ("x1", 1.01),
        ("x1", float("-inf")),
        ("x1", None),
        ("y1", 2),
        ("y1", np.array(1.0)),
    ],
)
def test_normalized_roi_rejects_wrong_types_nonfinite_and_out_of_range(
    field: str, value: object
):
    values = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        NormalizedRoi(**values)


@pytest.mark.parametrize(
    "values",
    [
        (0.5, 0.0, 0.5, 1.0),
        (0.75, 0.0, 0.25, 1.0),
        (0.0, 0.5, 1.0, 0.5),
        (0.0, 0.75, 1.0, 0.25),
    ],
)
def test_normalized_roi_rejects_empty_or_reversed_ranges(values: tuple[object, ...]):
    with pytest.raises(ValueError):
        NormalizedRoi(*values)


def test_normalized_roi_retains_accepted_real_values_without_coercion():
    x0 = np.float64(0.125)
    roi = NormalizedRoi(x0, 0, 1, 1.0)
    assert roi.x0 is x0
    assert type(roi.y0) is int
    assert type(roi.x1) is int


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_width", 0),
        ("client_width", -1),
        ("client_width", True),
        ("client_width", 1920.0),
        ("client_width", np.int64(1920)),
        ("client_width", "1920"),
        ("client_height", 0),
        ("client_height", False),
        ("client_height", 1080.0),
        ("client_aspect_ratio", 0),
        ("client_aspect_ratio", -1.0),
        ("client_aspect_ratio", True),
        ("client_aspect_ratio", float("nan")),
        ("client_aspect_ratio", float("inf")),
        ("client_aspect_ratio", "16:9"),
        ("client_aspect_ratio", 1.7),
        ("hud_scale_setting", ""),
        ("hud_scale_setting", "   "),
        ("hud_scale_setting", 100),
        ("language", ""),
        ("language", None),
        ("brightness_setting", "\t"),
        ("brightness_setting", object()),
        ("hdr_enabled", 0),
        ("hdr_enabled", "false"),
        ("game_version_note", None),
        ("game_version_note", 1),
    ],
)
def test_calibration_context_rejects_invalid_values(field: str, value: object):
    values = {
        "client_width": 1920,
        "client_height": 1080,
        "client_aspect_ratio": 1920 / 1080,
        "hud_scale_setting": "unknown",
        "language": "unknown",
        "brightness_setting": "unknown",
        "hdr_enabled": False,
        "game_version_note": "",
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        CalibrationContext(**values)


def test_calibration_context_aspect_absolute_tolerance_is_exactly_one_e_minus_six():
    expected = 1920 / 1080
    accepted = CalibrationContext(
        1920, 1080, expected + 1e-6, "100%", "en-US", "default", False
    )
    assert accepted.client_aspect_ratio == expected + 1e-6
    with pytest.raises(ValueError):
        CalibrationContext(
            1920,
            1080,
            expected + 1.000001e-6,
            "100%",
            "en-US",
            "default",
            False,
        )


@pytest.mark.parametrize(
    "args",
    [
        (None, "refs/p1.npz"),
        ({"x0": 0, "y0": 0, "x1": 1, "y1": 1}, "refs/p1.npz"),
        (ROI, None),
        (ROI, 1),
        (ROI, ""),
        (ROI, "   "),
    ],
)
def test_side_hud_profile_rejects_wrong_nested_and_string_values(args: tuple):
    with pytest.raises((TypeError, ValueError)):
        SideHudProfile(*args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile_id", ""),
        ("profile_id", "   "),
        ("profile_id", 1),
        ("calibration", {}),
        ("calibration", None),
        ("battle_roi", None),
        ("battle_roi", {}),
        ("training_hud_roi", {}),
        ("training_hud_roi", "none"),
        ("p1", {}),
        ("p1", None),
        ("p2", {}),
        ("p2", None),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", 0),
        ("schema_version", 2),
        ("schema_version", "1"),
    ],
)
def test_hud_profile_constructor_rejects_invalid_nested_and_scalar_values(
    field: str, value: object
):
    values = {
        "profile_id": "fixture",
        "calibration": sample_context(),
        "battle_roi": ROI,
        "training_hud_roi": None,
        "p1": SideHudProfile(ROI, "refs/p1.npz"),
        "p2": SideHudProfile(ROI, "refs/p2.npz"),
        "schema_version": 1,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        HudProfile(**values)


def test_hud_profile_accepts_only_none_or_normalized_training_roi():
    training_roi = NormalizedRoi(0.1, 0.2, 0.9, 0.8)
    assert sample_profile(training_hud_roi=None).training_hud_roi is None
    assert sample_profile(training_hud_roi=training_roi).training_hud_roi is training_roi


def test_crop_normalized_maps_exact_pixels():
    frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    roi = NormalizedRoi(x0=0.25, y0=0.50, x1=0.75, y1=0.90)
    crop = crop_normalized(frame, roi)
    original = crop[0, 0].copy()
    assert crop.shape == (40, 100, 3)
    assert np.array_equal(crop[0, 0], frame[50, 50])
    assert crop.flags.owndata and crop.flags.c_contiguous
    frame[50, 50] = 0
    assert np.array_equal(crop[0, 0], original)


def test_crop_normalized_uses_floor_start_ceil_end_and_boundary_clamp():
    frame = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    roi = NormalizedRoi(x0=0.19, y0=0.29, x1=0.82, y1=1.0)
    crop = crop_normalized(frame, roi)
    assert crop.shape == (5, 8, 3)
    assert np.array_equal(crop[0, 0], frame[2, 2])
    assert np.array_equal(crop[-1, -1], frame[6, 9])


@pytest.mark.parametrize(
    "frame",
    [
        [0],
        np.empty((0, 3, 3), dtype=np.uint8),
        np.empty((3, 0, 3), dtype=np.uint8),
        np.zeros((3, 3), dtype=np.uint8),
        np.zeros((3, 3, 1), dtype=np.uint8),
        np.zeros((3, 3, 4), dtype=np.uint8),
        np.zeros((3, 3, 3), dtype=np.float32),
        np.zeros((3, 3, 3), dtype=np.uint16),
    ],
)
def test_crop_normalized_rejects_malformed_bgr(frame: object):
    with pytest.raises((TypeError, ValueError)):
        crop_normalized(frame, ROI)


def test_crop_normalized_rejects_wrong_roi_object():
    with pytest.raises(TypeError):
        crop_normalized(np.zeros((3, 3, 3), dtype=np.uint8), (0, 0, 1, 1))


@pytest.mark.parametrize(
    "training_roi",
    [None, NormalizedRoi(0.05, 0.1, 0.95, 0.9)],
    ids=["none", "present"],
)
def test_profile_round_trip_records_context_and_optional_rois(
    tmp_path: Path, training_roi: NormalizedRoi | None
):
    profile = sample_profile(training_hud_roi=training_roi)
    save_profile(tmp_path, "profiles/default.json", profile)
    loaded = load_profile(tmp_path, "profiles/default.json")
    assert loaded == profile
    assert loaded.battle_roi == ROI
    assert loaded.training_hud_roi == training_roi
    target = tmp_path / "profiles" / "default.json"
    assert json.loads(target.read_text(encoding="utf-8")) == profile_payload(
        training_hud_roi=training_roi
    )
    assert target.read_bytes().endswith(b"\n")


def test_profile_save_serializes_every_accepted_real_without_mutating_values(
    tmp_path: Path,
):
    x0 = np.float32(0.125)
    y0 = Fraction(1, 4)
    x1 = np.float32(0.875)
    y1 = Fraction(3, 4)
    aspect = Fraction(2, 1)
    roi = NormalizedRoi(x0, y0, x1, y1)
    profile = HudProfile(
        profile_id="real-values",
        calibration=CalibrationContext(
            client_width=2,
            client_height=1,
            client_aspect_ratio=aspect,
            hud_scale_setting="100%",
            language="en",
            brightness_setting="default",
            hdr_enabled=False,
        ),
        battle_roi=roi,
        training_hud_roi=None,
        p1=SideHudProfile(roi, "refs/p1.npz"),
        p2=SideHudProfile(roi, "refs/p2.npz"),
    )

    save_profile(tmp_path, "profiles/reals.json", profile)
    loaded = load_profile(tmp_path, "profiles/reals.json")

    assert profile.battle_roi.x0 is x0
    assert profile.battle_roi.y0 is y0
    assert profile.calibration.client_aspect_ratio is aspect
    assert loaded.battle_roi == NormalizedRoi(0.125, 0.25, 0.875, 0.75)
    assert loaded.calibration.client_aspect_ratio == 2.0


def test_profile_save_uses_the_exact_canonical_utf8_json_encoding(tmp_path: Path):
    profile = replace(
        sample_profile(),
        profile_id="隆",
        calibration=replace(sample_context(), language="日本語"),
    )
    expected = profile_payload()
    expected["profile_id"] = "隆"
    calibration = expected["calibration"]
    assert isinstance(calibration, dict)
    calibration["language"] = "日本語"

    save_profile(tmp_path, "profiles/canonical.json", profile)

    expected_bytes = (
        json.dumps(
            expected,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    actual = (tmp_path / "profiles" / "canonical.json").read_bytes()
    assert actual == expected_bytes
    assert "隆".encode() in actual and "日本語".encode() in actual


ROOT_KEYS = (
    "schema_version",
    "profile_id",
    "calibration",
    "battle_roi",
    "training_hud_roi",
    "p1",
    "p2",
)
CALIBRATION_KEYS = (
    "client_width",
    "client_height",
    "client_aspect_ratio",
    "hud_scale_setting",
    "language",
    "brightness_setting",
    "hdr_enabled",
    "game_version_note",
)
ROI_KEYS = ("x0", "y0", "x1", "y1")
SIDE_KEYS = ("meter_roi", "references_path")


MISSING_KEY_CASES = (
    [((), key) for key in ROOT_KEYS]
    + [(('calibration',), key) for key in CALIBRATION_KEYS]
    + [(('battle_roi',), key) for key in ROI_KEYS]
    + [(('training_hud_roi',), key) for key in ROI_KEYS]
    + [(('p1',), key) for key in SIDE_KEYS]
    + [(('p1', 'meter_roi'), key) for key in ROI_KEYS]
    + [(('p2',), key) for key in SIDE_KEYS]
    + [(('p2', 'meter_roi'), key) for key in ROI_KEYS]
)


@pytest.mark.parametrize(("path", "missing_key"), MISSING_KEY_CASES)
def test_profile_json_rejects_every_missing_key(
    tmp_path: Path, path: tuple[str, ...], missing_key: str
):
    payload = profile_payload(training_hud_roi=NormalizedRoi(0.1, 0.2, 0.9, 0.8))
    nested_mapping(payload, path).pop(missing_key)
    write_payload(tmp_path / "profiles" / "invalid", payload)
    with pytest.raises((TypeError, ValueError)):
        load_profile(tmp_path, "profiles/invalid")


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("calibration",),
        ("battle_roi",),
        ("training_hud_roi",),
        ("p1",),
        ("p1", "meter_roi"),
        ("p2",),
        ("p2", "meter_roi"),
    ],
)
def test_profile_json_rejects_unknown_keys_at_every_level(
    tmp_path: Path, path: tuple[str, ...]
):
    payload = profile_payload(training_hud_roi=NormalizedRoi(0.1, 0.2, 0.9, 0.8))
    nested_mapping(payload, path)["unknown_field"] = True
    write_payload(tmp_path / "profiles" / "invalid", payload)
    with pytest.raises(ValueError):
        load_profile(tmp_path, "profiles/invalid")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ((), None),
        ((), []),
        ((), True),
        ((), 1),
        ((), "profile"),
        (("calibration",), []),
        (("battle_roi",), "roi"),
        (("training_hud_roi",), []),
        (("p1",), None),
        (("p1", "meter_roi"), 1),
        (("p2",), "side"),
        (("p2", "meter_roi"), False),
    ],
)
def test_profile_json_rejects_non_object_root_and_nested_values(
    tmp_path: Path, path: tuple[str, ...], value: object
):
    payload: object = profile_payload(
        training_hud_roi=NormalizedRoi(0.1, 0.2, 0.9, 0.8)
    )
    if path:
        assert isinstance(payload, dict)
        replace_nested(payload, path, value)
    else:
        payload = value
    write_payload(tmp_path / "profiles" / "invalid", payload)
    with pytest.raises((TypeError, ValueError)):
        load_profile(tmp_path, "profiles/invalid")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("schema_version",), 2),
        (("profile_id",), ""),
        (("profile_id",), 42),
        (("calibration", "client_width"), True),
        (("calibration", "client_width"), 1920.0),
        (("calibration", "client_height"), 0),
        (("calibration", "client_aspect_ratio"), "16:9"),
        (("calibration", "client_aspect_ratio"), 1.5),
        (("calibration", "hud_scale_setting"), ""),
        (("calibration", "language"), 1),
        (("calibration", "brightness_setting"), "   "),
        (("calibration", "hdr_enabled"), 0),
        (("calibration", "game_version_note"), None),
        (("battle_roi", "x0"), True),
        (("battle_roi", "x1"), 2),
        (("training_hud_roi", "y0"), "0.2"),
        (("p1", "references_path"), 1),
        (("p1", "references_path"), ""),
        (("p2", "meter_roi", "y1"), -1),
    ],
)
def test_profile_json_rejects_wrong_scalar_types_values_and_inconsistent_aspect(
    tmp_path: Path, path: tuple[str, ...], value: object
):
    payload = profile_payload(training_hud_roi=NormalizedRoi(0.1, 0.2, 0.9, 0.8))
    replace_nested(payload, path, value)
    write_payload(tmp_path / "profiles" / "invalid", payload)
    with pytest.raises((TypeError, ValueError)):
        load_profile(tmp_path, "profiles/invalid")


@pytest.mark.parametrize("constant", [float("nan"), float("inf"), float("-inf")])
def test_profile_json_rejects_nonstandard_nonfinite_constants(
    tmp_path: Path, constant: float
):
    payload = profile_payload()
    replace_nested(payload, ("calibration", "client_aspect_ratio"), constant)
    write_payload(tmp_path / "profiles" / "invalid", payload)
    with pytest.raises(ValueError):
        load_profile(tmp_path, "profiles/invalid")


def test_profile_json_rejects_invalid_utf8(tmp_path: Path):
    path = tmp_path / "profiles" / "invalid"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{\"profile_id\":\"\xff\"}")
    with pytest.raises(UnicodeDecodeError):
        load_profile(tmp_path, "profiles/invalid")


def test_profile_read_accepts_exactly_64_kib_when_json_is_otherwise_valid(
    tmp_path: Path,
):
    raw = json.dumps(
        profile_payload(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    assert len(raw) < MAX_PROFILE_BYTES
    exact = raw + b" " * (MAX_PROFILE_BYTES - len(raw))
    path = tmp_path / "profiles" / "exact"
    path.parent.mkdir(parents=True)
    path.write_bytes(exact)
    assert path.stat().st_size == MAX_PROFILE_BYTES
    assert load_profile(tmp_path, "profiles/exact") == sample_profile()


def test_profile_read_is_bounded_before_json_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "profiles" / "oversized"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff" + b" " * MAX_PROFILE_BYTES)
    original_open = Path.open
    read_sizes: list[int] = []

    class TrackingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.stream.__exit__(exc_type, exc_value, traceback)

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self.stream.read(size)

    def traced_open(path_value: Path, *args, **kwargs):
        return TrackingReader(original_open(path_value, *args, **kwargs))

    monkeypatch.setattr("sf6_music_trigger.hud_profile.Path.open", traced_open)
    with pytest.raises(ValueError, match="64 KiB"):
        load_profile(tmp_path, "profiles/oversized")
    assert read_sizes == [MAX_PROFILE_BYTES + 1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside",
        r"..\outside",
        "../data-private/file",
        r"C:\profiles\profile.json",
        "C:/profiles/profile.json",
        r"\\server\share\profile.json",
        r"C:profiles\profile.json",
        r"\profiles\profile.json",
        "/profiles/profile.json",
        r"\\?\C:\profiles\profile.json",
        r"\\.\C:\profiles\profile.json",
        r"profiles\profile.json:stream",
        "",
        ".",
        "profiles/",
        "profiles\\",
        "profiles/.",
        r"profiles\.",
        "profiles/..",
        r"profiles\..",
    ],
)
def test_resolve_contained_path_rejects_windows_escapes_and_directory_equivalents(
    tmp_path: Path, relative_path: str
):
    root = tmp_path / "data"
    root.mkdir()
    (tmp_path / "data-private").mkdir()
    with pytest.raises((TypeError, ValueError)):
        resolve_contained_path(root, relative_path, field_name="profile_path")


@pytest.mark.parametrize(
    "relative_path",
    [
        ". ",
        "...",
        "profiles./default",
        r"profiles.\default",
        "profiles /default",
        r"profiles \default",
        "profiles/default.",
        r"profiles\default.",
        "profiles/default ",
        r"profiles\default ",
    ],
)
def test_resolve_contained_path_rejects_raw_windows_components_ending_in_dot_or_space(
    tmp_path: Path, relative_path: str
):
    root = tmp_path / "data"
    root.mkdir()

    with pytest.raises(ValueError):
        resolve_contained_path(root, relative_path, field_name="profile_path")


@pytest.mark.parametrize(
    "reference_path",
    [
        "refs./p1.npz",
        r"refs.\p1.npz",
        "refs /p1.npz",
        r"refs \p1.npz",
        "refs/p1.npz.",
        r"refs\p1.npz ",
    ],
)
def test_reference_paths_reject_raw_windows_alias_components(
    tmp_path: Path, reference_path: str
):
    with pytest.raises(ValueError):
        resolve_contained_path(
            tmp_path,
            reference_path,
            field_name="reference_path",
            required_suffix=".npz",
        )


def test_profile_save_rejects_directory_alias_before_reference_root_can_retreat(
    tmp_path: Path,
):
    user_data_root = tmp_path / "data"
    user_data_root.mkdir()

    with pytest.raises(ValueError):
        save_profile(user_data_root, ". ", sample_profile())

    assert list(tmp_path.iterdir()) == [user_data_root]


def test_resolve_contained_path_allows_benign_internal_normalization(
    tmp_path: Path,
):
    root = tmp_path / "data"
    root.mkdir()
    resolved = resolve_contained_path(
        root, "subdir/../profile", field_name="profile_path"
    )
    assert resolved == (root / "profile").resolve(strict=False)


@pytest.mark.parametrize("suffix", [".npz", ".NPZ", ".NpZ"])
def test_reference_suffix_is_case_insensitive(tmp_path: Path, suffix: str):
    resolved = resolve_contained_path(
        tmp_path,
        f"refs/p1{suffix}",
        field_name="reference_path",
        required_suffix=".npz",
    )
    assert resolved.name == f"p1{suffix}"


@pytest.mark.parametrize("name", ["p1.npz.exe", "p1.zip", "p1.npy", "p1"])
def test_reference_suffix_rejects_wrong_final_suffix(tmp_path: Path, name: str):
    with pytest.raises(ValueError):
        resolve_contained_path(
            tmp_path,
            f"refs/{name}",
            field_name="reference_path",
            required_suffix=".npz",
        )


def test_profile_filename_has_no_suffix_policy(tmp_path: Path):
    profile = sample_profile()
    save_profile(tmp_path, "profiles/no-suffix", profile)
    assert load_profile(tmp_path, "profiles/no-suffix") == profile


def test_existing_directory_link_cannot_escape_contained_root(tmp_path: Path):
    root = tmp_path / "data"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape"
    link_kind = create_directory_link_or_skip(link, outside)
    try:
        with pytest.raises(ValueError):
            resolve_contained_path(
                root, "escape/profile", field_name="profile_path"
            )
    finally:
        if link_kind == "junction":
            link.rmdir()
        else:
            link.unlink()


def test_profile_rejects_equal_resolved_p1_and_p2_reference_paths(tmp_path: Path):
    profile = HudProfile(
        profile_id="aliases",
        calibration=sample_context(),
        battle_roi=ROI,
        training_hud_roi=None,
        p1=SideHudProfile(ROI, "refs/p1.npz"),
        p2=SideHudProfile(ROI, "refs/sub/../p1.npz"),
    )
    with pytest.raises(ValueError, match="P1|P2|reference|same|equal"):
        save_profile(tmp_path, "profiles/default", profile)

    payload = profile_payload()
    assert isinstance(payload["p2"], dict)
    payload["p2"]["references_path"] = "refs/sub/../p1.npz"
    write_payload(tmp_path / "profiles" / "load-default", payload)
    with pytest.raises(ValueError, match="P1|P2|reference|same|equal"):
        load_profile(tmp_path, "profiles/load-default")


def test_profile_load_rejects_distinct_existing_hardlink_names(tmp_path: Path):
    refs = tmp_path / "profiles" / "refs"
    refs.mkdir(parents=True)
    first = refs / "p1.npz"
    second = refs / "p2.npz"
    first.write_bytes(b"same inode")
    os.link(first, second)
    assert first.resolve() != second.resolve()
    assert os.path.samefile(first, second)

    write_payload(tmp_path / "profiles" / "default", profile_payload())
    with pytest.raises(ValueError, match="P1|P2|reference|same|hardlink"):
        load_profile(tmp_path, "profiles/default")
    with pytest.raises(ValueError, match="P1|P2|reference|same|hardlink"):
        save_profile(tmp_path, "profiles/save-default", sample_profile())


def test_atomic_profile_save_round_trips_and_leaves_no_temp(tmp_path: Path):
    target = tmp_path / "profiles" / "default"
    save_profile(tmp_path, "profiles/default", sample_profile())
    assert load_profile(tmp_path, "profiles/default") == sample_profile()
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_profile_replace_failure_preserves_old_neighbor_and_cleans_owned_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "profiles" / "default"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old profile bytes")
    neighbor = target.parent / "neighbor.tmp"
    neighbor.write_bytes(b"keep neighbor")
    owned_sources: list[Path] = []

    def fail_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        source_path = Path(source)
        owned_sources.append(source_path)
        assert source_path.parent == target.parent
        assert Path(destination) == target
        raise OSError("injected profile replace failure")

    monkeypatch.setattr("sf6_music_trigger.hud_profile.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        save_profile(tmp_path, "profiles/default", sample_profile())

    assert target.read_bytes() == b"old profile bytes"
    assert neighbor.read_bytes() == b"keep neighbor"
    assert len(owned_sources) == 1 and not owned_sources[0].exists()
    assert sorted(path.name for path in target.parent.iterdir()) == [
        "default",
        "neighbor.tmp",
    ]


def test_atomic_profile_save_uses_unique_temp_names_across_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sources: list[Path] = []

    def fail_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        sources.append(Path(source))
        raise OSError("injected")

    monkeypatch.setattr("sf6_music_trigger.hud_profile.os.replace", fail_replace)
    for _ in range(2):
        with pytest.raises(OSError):
            save_profile(tmp_path, "profiles/default", sample_profile())

    assert len(sources) == 2
    assert len(set(sources)) == 2
    assert all(source.suffix == ".tmp" and not source.exists() for source in sources)
    assert list((tmp_path / "profiles").iterdir()) == []


def test_profile_save_rejects_oversized_serialization_before_touching_target(
    tmp_path: Path,
):
    context = sample_context()
    object.__setattr__(context, "game_version_note", "x" * MAX_PROFILE_BYTES)
    profile = HudProfile(
        "large", context, ROI, None, SideHudProfile(ROI, "p1.npz"), SideHudProfile(ROI, "p2.npz")
    )
    with pytest.raises(ValueError, match="64 KiB"):
        save_profile(tmp_path, "profiles/default", profile)
    assert not (tmp_path / "profiles").exists()
