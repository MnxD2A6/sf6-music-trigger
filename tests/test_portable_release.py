from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import pytest


def test_portable_entry_uses_config_beside_frozen_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools import portable_entry

    executable = tmp_path / "SF6 Music Trigger.exe"
    seen: list[list[str]] = []
    monkeypatch.setattr(portable_entry.sys, "frozen", True, raising=False)
    monkeypatch.setattr(portable_entry.sys, "executable", str(executable))
    monkeypatch.setenv("TCL_LIBRARY", "original-tcl")
    monkeypatch.setenv("TK_LIBRARY", "original-tk")

    result = portable_entry.main(app_main=lambda argv: seen.append(list(argv)) or 0)

    assert result == 0
    assert seen == [["--config", str(tmp_path / "config.json")]]


def test_frozen_entry_points_tcl_and_tk_at_packaged_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools import portable_entry

    executable = tmp_path / "SF6 Music Trigger.exe"
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(portable_entry.sys, "frozen", True, raising=False)
    monkeypatch.setattr(portable_entry.sys, "executable", str(executable))
    monkeypatch.setenv("TCL_LIBRARY", "wrong-tcl")
    monkeypatch.setenv("TK_LIBRARY", "wrong-tk")

    result = portable_entry.main(
        app_main=lambda argv: seen.append(
            (os.environ["TCL_LIBRARY"], os.environ["TK_LIBRARY"])
        )
        or 0
    )

    assert result == 0
    assert seen == [
        (
            str(tmp_path / "_internal" / "_tcl_data"),
            str(tmp_path / "_internal" / "_tk_data"),
        )
    ]


def test_portable_self_check_validates_payload_without_starting_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools import portable_entry

    executable = tmp_path / "SF6 Music Trigger.exe"
    checked: list[Path] = []
    panel_calls: list[list[str]] = []
    exit_codes: list[int] = []
    monkeypatch.setattr(portable_entry.sys, "frozen", True, raising=False)
    monkeypatch.setattr(portable_entry.sys, "executable", str(executable))

    result = portable_entry.main(
        argv=["--portable-self-check"],
        app_main=lambda argv: panel_calls.append(list(argv)) or 0,
        self_checker=lambda root: checked.append(root),
        exit_process=exit_codes.append,
    )

    assert result == 0
    assert checked == [tmp_path]
    assert panel_calls == []
    assert exit_codes == [0]


def test_frozen_self_check_exits_nonzero_without_opening_an_error_dialog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools import portable_entry

    executable = tmp_path / "SF6 Music Trigger.exe"
    exit_codes: list[int] = []
    monkeypatch.setattr(portable_entry.sys, "frozen", True, raising=False)
    monkeypatch.setattr(portable_entry.sys, "executable", str(executable))

    def fail_check(root: Path) -> None:
        raise RuntimeError("broken portable runtime")

    result = portable_entry.main(
        argv=["--portable-self-check"],
        self_checker=fail_check,
        exit_process=exit_codes.append,
    )

    assert result == 1
    assert exit_codes == [1]
    assert "RuntimeError: broken portable runtime" in (
        tmp_path / "PORTABLE_SELF_CHECK_ERROR.txt"
    ).read_text(encoding="utf-8")


def test_portable_self_check_requires_tkinter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools import portable_entry

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sf6_music_trigger.config.load_config",
        lambda path: type("Config", (), {"profile_path": "profiles/default.json"})(),
    )
    monkeypatch.setattr("sf6_music_trigger.hud_profile.load_profile", lambda root, path: None)
    monkeypatch.setattr(
        portable_entry,
        "find_spec",
        lambda name: None if name == "tkinter" else object(),
    )

    with pytest.raises(RuntimeError, match=r"portable dependencies are missing: tkinter"):
        portable_entry.self_check(tmp_path)


def test_portable_self_check_requires_winrt_capture_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools import portable_entry

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sf6_music_trigger.config.load_config",
        lambda path: type("Config", (), {"profile_path": "profiles/default.json"})(),
    )
    monkeypatch.setattr("sf6_music_trigger.hud_profile.load_profile", lambda root, path: None)
    monkeypatch.setattr(
        portable_entry,
        "find_spec",
        lambda name: None if name == "winrt.windows.graphics.capture" else object(),
    )

    with pytest.raises(
        RuntimeError,
        match=r"portable dependencies are missing: winrt\.windows\.graphics\.capture",
    ):
        portable_entry.self_check(tmp_path)


def test_portable_self_check_initializes_tcl_without_opening_a_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools import portable_entry

    calls: list[str] = []

    class FakeInterpreter:
        def eval(self, script: str) -> str:
            calls.append(script)
            return "8.6.13"

    fake_tkinter = ModuleType("tkinter")
    fake_tkinter.Tcl = FakeInterpreter  # type: ignore[attr-defined]
    fake_tkinter.Tk = lambda: pytest.fail("headless self-check must not open a window")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tkinter", fake_tkinter)
    monkeypatch.setattr(portable_entry, "find_spec", lambda name: object())
    monkeypatch.setattr(
        "sf6_music_trigger.config.load_config",
        lambda path: type("Config", (), {"profile_path": "profiles/default.json"})(),
    )
    monkeypatch.setattr("sf6_music_trigger.hud_profile.load_profile", lambda root, path: None)

    portable_entry.self_check(tmp_path)

    assert calls == ["info patchlevel"]


def test_portable_self_check_imports_every_winrt_capture_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools import portable_entry

    imported: list[str] = []

    class FakeInterpreter:
        def eval(self, script: str) -> str:
            return "8.6.12"

    fake_tkinter = ModuleType("tkinter")
    fake_tkinter.Tcl = FakeInterpreter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tkinter", fake_tkinter)
    monkeypatch.setattr(portable_entry, "find_spec", lambda name: object())
    monkeypatch.setattr(
        portable_entry,
        "import_module",
        lambda name: imported.append(name) or ModuleType(name),
    )
    monkeypatch.setattr(
        "sf6_music_trigger.config.load_config",
        lambda path: type("Config", (), {"profile_path": "profiles/default.json"})(),
    )
    monkeypatch.setattr("sf6_music_trigger.hud_profile.load_profile", lambda root, path: None)

    portable_entry.self_check(tmp_path)

    assert imported == [*portable_entry.WINRT_CAPTURE_MODULES, "pygame.sndarray"]


def test_install_tk_runtime_copies_required_assets_without_cache(
    tmp_path: Path,
) -> None:
    from tools.portable_release import install_tk_runtime

    python_root = tmp_path / "python-runtime"
    data_root = tmp_path / "tk-data"
    internal = tmp_path / "release" / "_internal"
    (python_root / "tkinter" / "__pycache__").mkdir(parents=True)
    (python_root / "tkinter" / "__init__.py").write_text("# tkinter", encoding="utf-8")
    (python_root / "tkinter" / "__pycache__" / "cached.pyc").write_bytes(b"cache")
    for name in ("_tkinter.pyd", "tcl86t.dll", "tk86t.dll"):
        (python_root / name).write_bytes(name.encode("ascii"))
    for name in ("_tcl_data", "_tk_data"):
        folder = data_root / name
        folder.mkdir(parents=True)
        (folder / "init.tcl").write_text(name, encoding="utf-8")
    internal.mkdir(parents=True)

    install_tk_runtime(
        release_root=internal.parent,
        python_root=python_root,
        data_root=data_root,
    )

    assert (internal / "tkinter" / "__init__.py").is_file()
    assert not (internal / "tkinter" / "__pycache__").exists()
    assert (internal / "_tkinter.pyd").read_bytes() == b"_tkinter.pyd"
    assert (internal / "_tcl_data" / "init.tcl").read_text(encoding="utf-8") == "_tcl_data"
    assert (internal / "_tk_data" / "init.tcl").read_text(encoding="utf-8") == "_tk_data"


def test_prepare_payload_removes_private_paths_and_copies_profile_references(
    tmp_path: Path,
) -> None:
    from tools.portable_release import prepare_payload

    source_root = tmp_path / "source"
    profile_root = source_root / "profiles"
    refs_root = profile_root / "refs"
    refs_root.mkdir(parents=True)
    profile_path = profile_root / "default.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "group-test-default",
                "battle_roi": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
                "p1": {
                    "meter_roi": {"x0": 0.06, "y0": 0.88, "x1": 0.23, "y1": 0.98},
                    "references_path": "refs/p1.npz",
                },
                "p2": {
                    "meter_roi": {"x0": 0.77, "y0": 0.88, "x1": 0.95, "y1": 0.98},
                    "references_path": "refs/p2.npz",
                },
                "calibration": {
                    "client_width": 1920,
                    "client_height": 1080,
                    "client_aspect_ratio": 16 / 9,
                    "hud_scale_setting": "default",
                    "brightness_setting": "default",
                    "hdr_enabled": False,
                    "language": "Chinese",
                    "game_version_note": "group test",
                },
                "training_hud_roi": None,
            }
        ),
        encoding="utf-8",
    )
    (refs_root / "p1.npz").write_bytes(b"p1-reference")
    (refs_root / "p2.npz").write_bytes(b"p2-reference")
    source_config = source_root / "config.json"
    source_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "p1": {
                    "enabled": True,
                    "main_track": r"C:\Users\private\secret-song.mp3",
                    "volume": 0.31,
                    "whiff_track": r"D:\private\secret-whiff.wav",
                    "whiff_volume": 1.0,
                    "play_whiff_sound": True,
                    "whiff_fade_ms": 1500,
                    "super_scope": "SA3_CA_ONLY",
                    "fade_non_three": True,
                },
                "p2": {
                    "enabled": True,
                    "main_track": "",
                    "volume": 0.26,
                    "whiff_track": "",
                    "whiff_volume": 0.18,
                    "play_whiff_sound": False,
                    "whiff_fade_ms": 1500,
                    "super_scope": "SA3_CA_ONLY",
                },
                "profile_path": "profiles/default.json",
                "capture_fps": 20,
            }
        ),
        encoding="utf-8",
    )
    release_root = tmp_path / "release"

    prepare_payload(
        release_root=release_root,
        source_config=source_config,
        source_profile=profile_path,
        commit="febab79f7412b754ddbdfe8ff3bc79a579bede62",
    )

    config = json.loads((release_root / "config.json").read_text(encoding="utf-8"))
    assert config["profile_path"] == "profiles/default.json"
    assert config["p1"]["main_track"] == ""
    assert config["p1"]["whiff_track"] == ""
    assert config["p1"]["play_whiff_sound"] is False
    assert config["p1"]["fade_non_three"] is True
    assert config["p2"]["fade_non_three"] is False
    assert config["p2"]["main_track"] == ""
    assert config["p2"]["whiff_track"] == ""
    assert (release_root / "profiles" / "refs" / "p1.npz").read_bytes() == b"p1-reference"
    assert (release_root / "profiles" / "refs" / "p2.npz").read_bytes() == b"p2-reference"
    assert "private" not in (release_root / "config.json").read_text(encoding="utf-8")
    assert "V2" in (release_root / "VERSION.txt").read_text(encoding="utf-8")
    assert config["detection_backend"] == "reframework" and config["game_dir"] == ""
    assert config["volume_unlimited"] is False and config["auto_start"] is False
    assert (release_root / "music").is_dir()


def test_prepare_payload_rejects_missing_profile_reference(tmp_path: Path) -> None:
    from tools.portable_release import ReleaseValidationError, prepare_payload

    profile = tmp_path / "profiles" / "default.json"
    profile.parent.mkdir()
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "p1": {"references_path": "refs/missing-p1.npz"},
                "p2": {"references_path": "refs/missing-p2.npz"},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"schema_version": 1, "p1": {}, "p2": {}, "profile_path": "profiles/default.json"}),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseValidationError, match="profile reference is missing"):
        prepare_payload(
            release_root=tmp_path / "release",
            source_config=config,
            source_profile=profile,
            commit="febab79f7412b754ddbdfe8ff3bc79a579bede62",
        )


def test_write_sha256_manifest_covers_payload_without_hashing_itself(
    tmp_path: Path,
) -> None:
    from tools.portable_release import write_sha256_manifest

    (tmp_path / "SF6 Music Trigger.exe").write_bytes(b"portable-exe")
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "default.json").write_text("{}", encoding="utf-8")

    manifest = write_sha256_manifest(tmp_path)

    expected_exe = hashlib.sha256(b"portable-exe").hexdigest()
    expected_profile = hashlib.sha256(b"{}").hexdigest()
    assert manifest.read_text(encoding="utf-8").splitlines() == [
        f"{expected_exe}  SF6 Music Trigger.exe",
        f"{expected_profile}  profiles/default.json",
    ]


def test_validate_public_text_rejects_private_absolute_path(tmp_path: Path) -> None:
    from tools.portable_release import ReleaseValidationError, validate_public_text

    (tmp_path / "config.json").write_text(
        json.dumps({"main_track": r"C:\Users\private\secret.mp3"}),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseValidationError, match="absolute Windows path"):
        validate_public_text(tmp_path)


def test_create_release_zip_has_one_top_level_folder(tmp_path: Path) -> None:
    from tools.portable_release import create_release_zip

    release = tmp_path / "SF6_Music_Trigger_GroupTest_0.1.0a1"
    release.mkdir()
    (release / "SF6 Music Trigger.exe").write_bytes(b"exe")
    (release / "README-群测说明.txt").write_text("test build", encoding="utf-8")

    archive = create_release_zip(release)

    assert archive == tmp_path / "SF6_Music_Trigger_GroupTest_0.1.0a1.zip"
    with ZipFile(archive) as bundle:
        assert bundle.namelist() == [
            "SF6_Music_Trigger_GroupTest_0.1.0a1/README-群测说明.txt",
            "SF6_Music_Trigger_GroupTest_0.1.0a1/SF6 Music Trigger.exe",
        ]


def test_validate_no_forbidden_bytes_rejects_embedded_builder_identity(
    tmp_path: Path,
) -> None:
    from tools.portable_release import ReleaseValidationError, validate_no_forbidden_bytes

    (tmp_path / "SF6 Music Trigger.exe").write_bytes(
        b"binary-prefix C:\\Users\\build-user\\private-source binary-suffix"
    )

    with pytest.raises(ReleaseValidationError, match="forbidden builder marker"):
        validate_no_forbidden_bytes(tmp_path, ("build-user",))


def test_pyinstaller_command_builds_windowed_onedir_from_current_source(tmp_path: Path) -> None:
    from tools.portable_release import pyinstaller_command

    repo_root = tmp_path / "repo"
    build_root = tmp_path / "build"

    command = pyinstaller_command(
        python=Path(sys.executable),
        repo_root=repo_root,
        build_root=build_root,
    )

    assert command[:3] == [sys.executable, "-m", "PyInstaller"]
    assert "--onedir" in command
    assert "--windowed" in command
    assert "--collect-all" not in command
    assert "comtypes" not in command
    assert command[command.index("--copy-metadata") + 1] == "dxcam"
    assert command[command.index("--paths") + 1] == str(repo_root / "src")
    assert command[command.index("--name") + 1] == "SF6 Music Trigger"
    assert command[-1] == str(repo_root / "tools" / "portable_entry.py")
    add_data = command[command.index("--add-data") + 1]
    assert add_data == f"{repo_root / 'src' / 'sf6_music_trigger' / 'game_scripts'};sf6_music_trigger/game_scripts"
    hidden_imports = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--hidden-import"
    ]
    assert {
        "winrt.runtime",
        "winrt.windows.foundation",
        "winrt.windows.foundation.collections",
        "winrt.windows.graphics",
        "winrt.windows.graphics.capture",
        "winrt.windows.graphics.capture.interop",
        "winrt.windows.graphics.directx",
        "winrt.windows.graphics.directx.direct3d11",
        "winrt.windows.graphics.directx.direct3d11.interop",
        "pygame.sndarray",
        "sf6_music_trigger.game_ui",
        "sf6_music_trigger.ref_engine",
    }.issubset(hidden_imports)


def test_group_test_readme_explains_v2_install_use_and_removal() -> None:
    from tools.portable_release import RELEASE_VERSION, group_test_readme, validate_public_text

    text = group_test_readme()

    for phrase in (
        "V2", RELEASE_VERSION, "解压", "一键安装/更新", "从游戏中移除", "REFramework",
        "离线", "联机", "自动停用", "SF6 音乐设置", "Insert", "解除音量上限", "重置音量",
        "music", "events.jsonl", "画面识别", "SHA256SUMS.txt",
    ):
        assert phrase in text, phrase


def test_readme_passes_the_public_text_check(tmp_path: Path) -> None:
    from tools.portable_release import group_test_readme, validate_public_text

    (tmp_path / "README-群测说明.txt").write_text(group_test_readme(), encoding="utf-8")
    validate_public_text(tmp_path)


def test_reframework_bundle_is_copied_with_its_licence(tmp_path: Path) -> None:
    from tools.portable_release import ReleaseValidationError, copy_reframework_bundle

    source = tmp_path / "source"
    source.mkdir()
    (source / "dinput8.dll").write_bytes(b"dll")
    (source / "LICENSE.txt").write_text("MIT License", encoding="utf-8")
    release = tmp_path / "release"
    with pytest.raises(ReleaseValidationError):
        copy_reframework_bundle(release, source)
    (source / "revision.txt").write_text("nightly", encoding="utf-8")
    target = copy_reframework_bundle(release, source)
    assert target == release / "REFramework"
    assert (target / "dinput8.dll").read_bytes() == b"dll"
    assert (target / "LICENSE.txt").read_text(encoding="utf-8") == "MIT License"
    assert "一键安装" in (target / "README.txt").read_text(encoding="utf-8")
