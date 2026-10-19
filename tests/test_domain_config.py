from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from sf6_music_trigger.config import AppConfig, SideAudioConfig, load_config, save_config
from sf6_music_trigger.domain import (
    CancelledSuperCandidate,
    CapturedFrame,
    CaptureStatus,
    CandidateCancelReason,
    ClientGeometry,
    ConfirmedSuperEvent,
    GaugeTrackerUpdate,
    Side,
    SuperCandidateEvent,
    TrackerState,
)


def write_config(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_domain_enums_are_string_enums_with_stable_values():
    assert Side.P1 == "P1"
    assert Side.P2 == "P2"
    assert [status.value for status in CaptureStatus] == [
        "RUNNING",
        "OCCLUDED",
        "MINIMIZED",
        "LOST",
        "STOPPED",
    ]
    assert [state.value for state in TrackerState] == [
        "DISARMED",
        "ARMED",
        "SPEND_CANDIDATE",
        "CONFIRMED",
        "COOLDOWN",
    ]


def test_client_geometry_accepts_negative_origins_and_is_immutable():
    geometry = ClientGeometry(-1920, -1080, 1920, 1080)
    assert (geometry.left, geometry.top, geometry.width, geometry.height) == (
        -1920,
        -1080,
        1920,
        1080,
    )
    with pytest.raises(FrozenInstanceError):
        geometry.left = 0


@pytest.mark.parametrize(
    ("values", "error_type"),
    [
        ((True, 0, 1, 1), TypeError),
        ((0, False, 1, 1), TypeError),
        ((0.0, 0, 1, 1), TypeError),
        ((0, 0, True, 1), TypeError),
        ((0, 0, 1, 2.0), TypeError),
        ((0, 0, 0, 1), ValueError),
        ((0, 0, 1, -1), ValueError),
    ],
)
def test_client_geometry_rejects_invalid_scalars(values: tuple, error_type: type[Exception]):
    with pytest.raises(error_type):
        ClientGeometry(*values)


def _candidate(side: Side = Side.P1) -> SuperCandidateEvent:
    return SuperCandidateEvent(
        candidate_id="cand-p1-1",
        side=side,
        suspected_at=10.15,
        pre_spend_frame_id=100,
        candidate_frame_id=103,
        profile_id="profile",
        meter_before=3,
        meter_after=0,
        evidence=("stable_downward_level",),
    )


def test_candidate_lifecycle_values_are_frozen_and_side_bound() -> None:
    candidate = _candidate()
    assert (candidate.meter_before, candidate.meter_after) == (3, 0)
    assert candidate.levels_suspected == 3
    with pytest.raises(FrozenInstanceError):
        candidate.meter_after = 1

    cancelled = CancelledSuperCandidate(
        candidate.candidate_id,
        Side.P1,
        10.6,
        CandidateCancelReason.LEVEL_RECOVERED,
    )
    assert GaugeTrackerUpdate(cancelled).transition is cancelled


def test_confirmed_event_accepts_matching_candidate_identity() -> None:
    event = ConfirmedSuperEvent(
        "evt-1", Side.P1, 10.6, 100, 112, "profile",
        meter_before=3, meter_after=0, levels_spent=3,
        candidate_id="cand-p1-1",
    )
    assert event.candidate_id == "cand-p1-1"


@pytest.mark.parametrize("value", ["", "   ", 7, object()])
def test_candidate_identity_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SuperCandidateEvent(
            value, Side.P1, 1.0, 1, 3, "profile", 3, 0,
            ("stable_downward_level",),
        )


def test_confirmed_event_is_immutable_and_side_typed():
    event = ConfirmedSuperEvent("evt-1", Side.P1, 12.5, 42, 45, "default")
    with pytest.raises(FrozenInstanceError):
        event.event_id = "changed"

    with pytest.raises((TypeError, ValueError)):
        ConfirmedSuperEvent("evt-2", "P1", 12.5, 42, 45, "default")


def test_confirmed_event_accepts_consistent_optional_meter_metadata():
    event = ConfirmedSuperEvent(
        "evt-level",
        Side.P1,
        12.5,
        42,
        45,
        "default",
        meter_before=3,
        meter_after=0,
        levels_spent=3,
    )

    assert (event.meter_before, event.meter_after, event.levels_spent) == (3, 0, 3)


@pytest.mark.parametrize(
    ("meter_before", "meter_after", "levels_spent"),
    [
        (3, None, None),
        (None, 0, None),
        (None, None, 3),
        (True, 0, 3),
        (3, False, 3),
        (3, 0, True),
        (4, 0, 4),
        (3, -1, 4),
        (2, 2, 0),
        (3, 1, 1),
    ],
)
def test_confirmed_event_rejects_inconsistent_meter_metadata(
    meter_before: object,
    meter_after: object,
    levels_spent: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ConfirmedSuperEvent(
            "evt-level",
            Side.P1,
            12.5,
            42,
            45,
            "default",
            meter_before=meter_before,  # type: ignore[arg-type]
            meter_after=meter_after,  # type: ignore[arg-type]
            levels_spent=levels_spent,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "args",
    [
        ("", Side.P1, 0.0, 0, 0, "default"),
        ("   ", Side.P1, 0.0, 0, 0, "default"),
        (1, Side.P1, 0.0, 0, 0, "default"),
        ("evt", Side.P1, -0.01, 0, 0, "default"),
        ("evt", Side.P1, float("nan"), 0, 0, "default"),
        ("evt", Side.P1, float("inf"), 0, 0, "default"),
        ("evt", Side.P1, True, 0, 0, "default"),
        ("evt", Side.P1, 0.0, -1, 0, "default"),
        ("evt", Side.P1, 0.0, 1, 0, "default"),
        ("evt", Side.P1, 0.0, True, 1, "default"),
        ("evt", Side.P1, 0.0, 0, 1.0, "default"),
        ("evt", Side.P1, 0.0, 0, 1, ""),
        ("evt", Side.P1, 0.0, 0, 1, "  "),
        ("evt", Side.P1, 0.0, 0, 1, 42),
    ],
)
def test_confirmed_event_rejects_invalid_domain_scalars(args: tuple):
    with pytest.raises((TypeError, ValueError)):
        ConfirmedSuperEvent(*args)


def test_captured_frame_is_a_frozen_identity_envelope_without_copying_pixels():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    geometry = ClientGeometry(-1920, 0, 1920, 1080)
    first = CapturedFrame(1, image, 1.25, geometry)
    second = CapturedFrame(1, image, 1.25, geometry)
    assert first != second
    assert first.image is image
    with pytest.raises(FrozenInstanceError):
        first.frame_id = 2


@pytest.mark.parametrize(
    "args",
    [
        (-1, np.zeros(1), 0.0, ClientGeometry(0, 0, 1, 1)),
        (True, np.zeros(1), 0.0, ClientGeometry(0, 0, 1, 1)),
        (1.0, np.zeros(1), 0.0, ClientGeometry(0, 0, 1, 1)),
        (0, [0], 0.0, ClientGeometry(0, 0, 1, 1)),
        (0, np.zeros(1), -1.0, ClientGeometry(0, 0, 1, 1)),
        (0, np.zeros(1), float("nan"), ClientGeometry(0, 0, 1, 1)),
        (0, np.zeros(1), float("inf"), ClientGeometry(0, 0, 1, 1)),
        (0, np.zeros(1), True, ClientGeometry(0, 0, 1, 1)),
        (0, np.zeros(1), 0.0, (0, 0, 1, 1)),
    ],
)
def test_captured_frame_rejects_invalid_domain_scalars(args: tuple):
    with pytest.raises((TypeError, ValueError)):
        CapturedFrame(*args)


def test_domain_times_accept_arbitrarily_large_nonnegative_python_integers():
    huge_time = 10**1000
    geometry = ClientGeometry(0, 0, 1, 1)

    event = ConfirmedSuperEvent("evt-huge", Side.P2, huge_time, 0, 0, "default")
    frame = CapturedFrame(0, np.zeros(1), huge_time, geometry)

    assert event.confirmed_at == huge_time
    assert frame.monotonic_time == huge_time


def test_domain_times_accept_arbitrarily_large_finite_fractions_without_coercion():
    huge_time = Fraction(10**10000, 1)
    geometry = ClientGeometry(0, 0, 1, 1)

    event = ConfirmedSuperEvent("evt-fraction", Side.P2, huge_time, 0, 0, "default")
    frame = CapturedFrame(0, np.zeros(1), huge_time, geometry)

    assert event.confirmed_at is huge_time
    assert frame.monotonic_time is huge_time


@pytest.mark.parametrize(
    "invalid_time",
    [Fraction(-1, 2), float("nan"), float("inf"), float("-inf")],
)
def test_domain_times_reject_negative_fraction_and_nonfinite_floats(invalid_time: object):
    geometry = ClientGeometry(0, 0, 1, 1)

    with pytest.raises(ValueError):
        ConfirmedSuperEvent("evt-invalid", Side.P1, invalid_time, 0, 0, "default")
    with pytest.raises(ValueError):
        CapturedFrame(0, np.zeros(1), invalid_time, geometry)


def test_missing_config_loads_safe_defaults(tmp_path: Path):
    assert load_config(tmp_path / "missing.json") == AppConfig()


@pytest.mark.parametrize(
    "audio_path",
    [
        "D:/Music/song.mp3",
        r"C:\Music\song.mp3",
        r"\\server\share\song.mp3",
    ],
)
def test_runtime_config_preserves_nonexistent_user_selected_absolute_audio_path(
    tmp_path: Path, audio_path: str
):
    path = tmp_path / "user_config.json"
    write_config(path, {"p1": {"main_track": audio_path}})
    assert load_config(path).p1.main_track == audio_path


def test_partial_config_clamps_high_volume_and_fills_defaults(tmp_path: Path):
    path = tmp_path / "user_config.json"
    write_config(path, {"p1": {"main_track": "audio/p1.mp3", "volume": 9.0}})
    loaded = load_config(path)
    assert loaded.p1.main_track == "audio/p1.mp3"
    assert loaded.p1.volume == 5.0
    assert loaded.p2 == SideAudioConfig()


def test_volume_clamps_low_and_fps_boundaries_are_accepted(tmp_path: Path):
    path = tmp_path / "user_config.json"
    write_config(path, {"capture_fps": 10, "p1": {"volume": -3}})
    loaded = load_config(path)
    assert loaded.capture_fps == 10
    assert loaded.p1.volume == 0.0

    write_config(path, {"capture_fps": 30, "p2": {"volume": 1}})
    loaded = load_config(path)
    assert loaded.capture_fps == 30
    assert loaded.p2.volume == 1.0


def test_arbitrarily_large_finite_integer_volumes_clamp_without_overflow(tmp_path: Path):
    path = tmp_path / "user_config.json"
    huge_volume = 10**1000
    write_config(
        path,
        {
            "p1": {"volume": huge_volume},
            "p2": {"volume": -huge_volume},
        },
    )

    loaded = load_config(path)

    assert loaded.p1.volume == 5.0
    assert loaded.p2.volume == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        {"profile_path": "../outside.json"},
        {"profile_path": r"..\outside.json"},
        {"profile_path": "D:/profiles/outside.json"},
        {"profile_path": r"D:profiles\outside.json"},
        {"profile_path": r"\profiles\outside.json"},
        {"profile_path": "/profiles/outside.json"},
        {"profile_path": r"\\server\share\outside.json"},
        {"profile_path": r"\\?\C:\profiles\outside.json"},
        {"profile_path": r"\\.\C:\profiles\outside.json"},
        {"profile_path": r"profiles\profile.json:stream"},
        {"profile_path": ""},
        {"profile_path": "."},
        {"profile_path": "profiles/"},
        {"profile_path": "profiles\\"},
        {"profile_path": "profiles/."},
        {"profile_path": "profiles\\."},
        {"profile_path": "profiles/.."},
        {"profile_path": "profiles\\.."},
        {"p1": {"main_track": "../private.mp3"}},
        {"p1": {"main_track": r"..\private.mp3"}},
        {"p1": {"main_track": r"D:music\private.mp3"}},
        {"p1": {"main_track": r"\music\private.mp3"}},
        {"p1": {"main_track": "/music/private.mp3"}},
        {"p1": {"main_track": r"\\?\C:\music\private.mp3"}},
        {"p1": {"main_track": r"\\.\C:\music\private.mp3"}},
        {"p1": {"main_track": r"audio\private.mp3:stream"}},
        {"capture_fps": 9},
        {"capture_fps": 31},
        {"capture_fps": True},
        {"capture_fps": 20.0},
        {"p1": {"volume": float("nan")}},
        {"p1": {"volume": float("inf")}},
        {"p1": {"volume": True}},
        {"p1": {"enabled": "true"}},
        {"p1": {"main_track": 7}},
        {"profile_path": 7},
    ],
)
def test_config_rejects_escaping_paths_and_invalid_types(tmp_path: Path, payload: dict):
    path = tmp_path / "user_config.json"
    write_config(path, payload)
    with pytest.raises((TypeError, ValueError)):
        load_config(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_config_rejects_nonstandard_nonfinite_json_constants(tmp_path: Path, constant: str):
    path = tmp_path / "user_config.json"
    path.write_text('{"p1":{"volume":' + constant + "}}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("payload", [None, [], True, 1, "config", {"p1": []}, {"p2": "side"}])
def test_config_rejects_non_object_roots_and_sides(tmp_path: Path, payload: object):
    path = tmp_path / "user_config.json"
    write_config(path, payload)
    with pytest.raises((TypeError, ValueError)):
        load_config(path)


def test_diagnostic_config_redacts_audio_paths_without_exposing_roots_or_basenames():
    config = AppConfig(
        p1=SideAudioConfig(main_track="D:/Music/private-name.mp3"),
        p2=SideAudioConfig(main_track=r"\\server\share\second-private.wav"),
    )
    diagnostic = json.dumps(config.diagnostic_view())
    assert "private-name" not in diagnostic
    assert "second-private" not in diagnostic
    assert "D:/Music" not in diagnostic
    assert "server" not in diagnostic
    assert diagnostic.count("<redacted>") == 2


def test_diagnostic_config_leaves_empty_audio_paths_empty():
    diagnostic = AppConfig().diagnostic_view()
    assert diagnostic["p1"]["main_track"] == ""
    assert diagnostic["p2"]["main_track"] == ""


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 0},
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"unknown_option": True},
        {"p1": {"typo_volume": 0.5}},
        {"p2": {"unknown": False}},
    ],
)
def test_config_schema_and_unknown_fields_fail_closed(tmp_path: Path, payload: dict):
    path = tmp_path / "user_config.json"
    write_config(path, payload)
    with pytest.raises((TypeError, ValueError)):
        load_config(path)


def test_relative_paths_inside_user_data_root_are_preserved_without_existence_checks(
    tmp_path: Path,
):
    path = tmp_path / "user_config.json"
    payload = {
        "profile_path": "profiles/missing.json",
        "p1": {"main_track": "audio/also-missing.mp3"},
    }
    write_config(path, payload)
    loaded = load_config(path)
    assert loaded.profile_path == payload["profile_path"]
    assert loaded.p1.main_track == payload["p1"]["main_track"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"profile_path": "../data-private/profile.json"},
            id="profile",
        ),
        pytest.param(
            {"p1": {"main_track": "../data-private/song.mp3"}},
            id="relative-audio",
        ),
    ],
)
def test_sibling_prefix_escape_is_rejected(tmp_path: Path, payload: dict):
    root = tmp_path / "data"
    root.mkdir()
    sibling = tmp_path / "data-private"
    sibling.mkdir()
    path = root / "user_config.json"
    write_config(path, payload)
    with pytest.raises(ValueError):
        load_config(path)


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
                "host permits neither directory symlinks nor unprivileged junctions: "
                f"{symlink_error}; {result.stderr or result.stdout}"
            )
        return "junction"


@pytest.mark.parametrize("field", ["profile_path", "main_track"])
def test_existing_directory_link_cannot_escape_user_data_root(tmp_path: Path, field: str):
    root = tmp_path / "data"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape"
    link_kind = create_directory_link_or_skip(link, outside)

    path = root / "user_config.json"
    payload = (
        {"profile_path": "escape/profile.json"}
        if field == "profile_path"
        else {"p1": {"main_track": "escape/song.mp3"}}
    )
    write_config(path, payload)
    try:
        with pytest.raises(ValueError):
            load_config(path)
    finally:
        if link_kind == "junction":
            link.rmdir()
        else:
            link.unlink()


def test_side_audio_and_app_config_validate_direct_construction():
    assert SideAudioConfig(volume=3).volume == 3.0
    assert SideAudioConfig(volume=9).volume == 5.0
    assert SideAudioConfig(volume=-2).volume == 0.0
    with pytest.raises(TypeError):
        SideAudioConfig(enabled=1)
    with pytest.raises(TypeError):
        SideAudioConfig(main_track=None)
    with pytest.raises(ValueError):
        SideAudioConfig(volume=math.nan)
    with pytest.raises(TypeError):
        AppConfig(capture_fps=True)
    with pytest.raises(ValueError):
        AppConfig(capture_fps=9)
    with pytest.raises(TypeError):
        AppConfig(p1={})


def test_atomic_save_round_trips_and_leaves_no_temp_file(tmp_path: Path):
    path = tmp_path / "nested" / "user_config.json"
    config = AppConfig(
        p1=SideAudioConfig(main_track="audio/p1.mp3", volume=0.25),
        p2=SideAudioConfig(enabled=False, main_track=r"C:\Music\p2.wav", volume=0.75),
        profile_path="profiles/custom.json",
        capture_fps=30,
    )
    save_config(path, config)
    assert load_config(path) == config
    assert list(path.parent.glob("*.tmp")) == []
    assert json.loads(path.read_text(encoding="utf-8"))["capture_fps"] == 30


def test_whiff_audio_config_defaults_to_fade_and_round_trips_sound_mode(tmp_path: Path):
    path = tmp_path / "user_config.json"
    assert SideAudioConfig().play_whiff_sound is False
    assert SideAudioConfig().whiff_track == ""
    assert SideAudioConfig().whiff_volume == 0.18
    assert SideAudioConfig().whiff_fade_ms == 1500

    config = AppConfig(
        p1=SideAudioConfig(
            main_track="audio/main.wav",
            whiff_track="audio/whiff.wav",
            whiff_volume=0.42,
            play_whiff_sound=True,
            whiff_fade_ms=2400,
        )
    )
    save_config(path, config)
    assert load_config(path) == config


def test_super_scope_defaults_to_sa3_ca_only_and_round_trips(tmp_path: Path) -> None:
    from sf6_music_trigger.config import SuperScope

    assert issubclass(SuperScope, str)
    assert SideAudioConfig().super_scope is SuperScope.SA3_CA_ONLY
    config = AppConfig(
        p1=SideAudioConfig(super_scope=SuperScope.ALL_SUPERS),
        p2=SideAudioConfig(super_scope=SuperScope.SA3_CA_ONLY),
    )
    target = tmp_path / "config.json"

    save_config(target, config)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["p1"]["super_scope"] == "ALL_SUPERS"
    assert payload["p2"]["super_scope"] == "SA3_CA_ONLY"
    assert load_config(target) == config


def test_missing_super_scope_uses_safe_sa3_ca_only_default(tmp_path: Path) -> None:
    from sf6_music_trigger.config import SuperScope

    target = tmp_path / "config.json"
    write_config(target, {"p1": {"enabled": True}})

    assert load_config(target).p1.super_scope is SuperScope.SA3_CA_ONLY


@pytest.mark.parametrize("value", ["EVERYTHING", "all_supers", 1, None])
def test_unknown_super_scope_fails_closed(tmp_path: Path, value: object) -> None:
    target = tmp_path / "config.json"
    write_config(target, {"p1": {"super_scope": value}})

    with pytest.raises(ValueError, match=r"^p1\.super_scope is invalid$"):
        load_config(target)


def test_direct_side_config_rejects_raw_or_subclassed_scope_strings() -> None:
    class ScopeText(str):
        pass

    with pytest.raises(TypeError, match="super_scope"):
        SideAudioConfig(super_scope="ALL_SUPERS")
    with pytest.raises(TypeError, match="super_scope"):
        SideAudioConfig(super_scope=ScopeText("ALL_SUPERS"))


def test_whiff_paths_are_redacted_and_invalid_policy_values_fail_closed() -> None:
    secret = r"C:\private\whiff-name.wav"
    diagnostic = AppConfig(
        p1=SideAudioConfig(main_track="audio/main.wav", whiff_track=secret)
    ).diagnostic_view()
    assert diagnostic["p1"]["whiff_track"] == "<redacted>"
    assert "whiff-name" not in json.dumps(diagnostic)

    with pytest.raises(TypeError):
        SideAudioConfig(play_whiff_sound=1)
    with pytest.raises(TypeError):
        SideAudioConfig(whiff_fade_ms=1.5)
    with pytest.raises(ValueError):
        SideAudioConfig(whiff_fade_ms=-1)
    with pytest.raises(TypeError):
        SideAudioConfig(whiff_track=None)


def test_save_validates_before_creating_parent_or_touching_target(tmp_path: Path):
    path = tmp_path / "not-created" / "user_config.json"
    config = AppConfig()
    object.__setattr__(config.p1, "volume", float("nan"))

    with pytest.raises(ValueError):
        save_config(path, config)

    assert not path.parent.exists()


def test_atomic_save_preserves_old_target_and_cleans_owned_temp_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "user_config.json"
    path.write_text("old-bytes", encoding="utf-8")
    neighbor = tmp_path / "neighbor.tmp"
    neighbor.write_text("keep", encoding="utf-8")

    def fail_replace(source, target):
        assert Path(source).parent == path.parent
        assert Path(target) == path
        raise OSError("injected replace failure")

    monkeypatch.setattr("sf6_music_trigger.config.os.replace", fail_replace)
    with pytest.raises(OSError):
        save_config(path, AppConfig())

    assert path.read_text(encoding="utf-8") == "old-bytes"
    assert neighbor.read_text(encoding="utf-8") == "keep"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["neighbor.tmp", "user_config.json"]


def test_save_uses_a_unique_temp_name_for_each_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "user_config.json"
    sources: list[Path] = []

    def fail_replace(source, target):
        sources.append(Path(source))
        raise OSError("injected")

    monkeypatch.setattr("sf6_music_trigger.config.os.replace", fail_replace)
    for _ in range(2):
        with pytest.raises(OSError):
            save_config(path, AppConfig())

    assert len(set(sources)) == 2
    assert all(source.parent == path.parent and source.suffix == ".tmp" for source in sources)
    assert list(tmp_path.iterdir()) == []
