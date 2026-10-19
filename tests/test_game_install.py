"""V2 one-click install and removal in the SF6 game folder."""

from pathlib import Path

import pytest

from sf6_music_trigger.game_install import (
    REFRAMEWORK_DLL,
    install_all,
    inspect_game,
    other_reframework_mods,
    remove_from_game,
)
from sf6_music_trigger.reframework_events import GAME_SCRIPTS, bundled_game_script


@pytest.fixture
def bundle(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    (folder / REFRAMEWORK_DLL).write_bytes(b"bundled-reframework")
    return folder


@pytest.fixture
def game(tmp_path):
    folder = tmp_path / "Street Fighter 6"
    folder.mkdir()
    (folder / "StreetFighter6.exe").write_bytes(b"game")
    (folder / "re_chunk_000.pak").write_bytes(b"game data")
    return folder


def simulate_reframework_run(game: Path) -> None:
    """What REFramework and V2 create once the game has run."""

    (game / "re2_fw_config.txt").write_text("MenuKey=45\n", encoding="utf-8")
    (game / "re2_framework_log.txt").write_text("log\n", encoding="utf-8")
    data = game / "reframework" / "data" / "SF6MusicTrigger"
    data.mkdir(parents=True)
    (data / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (data / "app_state.json").write_text("{}", encoding="utf-8")


def test_fresh_game_gets_reframework_and_scripts(game, bundle):
    assert inspect_game(game, bundle).reframework == "missing"
    report = install_all(game, bundle)
    assert report.reframework == "installed" and report.backup is None
    assert (game / REFRAMEWORK_DLL).read_bytes() == b"bundled-reframework"
    for name in GAME_SCRIPTS:
        installed = game / "reframework" / "autorun" / name
        assert installed.read_bytes() == bundled_game_script(name).read_bytes()
    state = inspect_game(game, bundle)
    assert (state.reframework, state.scripts_current, state.other_mods) == ("bundled", True, ())
    assert install_all(game, bundle).reframework == "unchanged"


def test_foreign_dll_is_kept_by_default_and_backed_up_when_replaced(game, bundle):
    (game / REFRAMEWORK_DLL).write_bytes(b"users-own-reframework")
    assert inspect_game(game, bundle).reframework == "other"
    assert install_all(game, bundle).reframework == "kept"
    assert (game / REFRAMEWORK_DLL).read_bytes() == b"users-own-reframework"

    report = install_all(game, bundle, replace_other_dll=True)
    assert report.reframework == "replaced"
    assert (game / report.backup).read_bytes() == b"users-own-reframework"
    assert (game / REFRAMEWORK_DLL).read_bytes() == b"bundled-reframework"


def test_install_without_bundle_needs_an_existing_reframework(game):
    with pytest.raises(FileNotFoundError):
        install_all(game, None)
    (game / REFRAMEWORK_DLL).write_bytes(b"own")
    assert install_all(game, None).reframework == "kept"


def test_install_rejects_a_folder_without_the_game(tmp_path, bundle):
    with pytest.raises(ValueError):
        install_all(tmp_path, bundle)


def test_remove_v2_only_keeps_reframework(game, bundle):
    install_all(game, bundle)
    simulate_reframework_run(game)
    report = remove_from_game(game)
    assert "reframework/data/SF6MusicTrigger/" in report.removed
    assert not (game / "reframework" / "data" / "SF6MusicTrigger").exists()
    assert not any((game / "reframework" / "autorun" / name).exists() for name in GAME_SCRIPTS)
    assert (game / REFRAMEWORK_DLL).exists() and (game / "re2_fw_config.txt").exists()


def test_remove_everything_restores_the_original_game_folder(game, bundle):
    original = sorted(p.name for p in game.iterdir())
    install_all(game, bundle)
    simulate_reframework_run(game)
    (game / "reframework" / "autorun" / "sf6_music_trigger_probe.lua").write_text("--", encoding="utf-8")
    report = remove_from_game(game, remove_reframework=True)
    assert REFRAMEWORK_DLL in report.removed and report.kept == ()
    assert sorted(p.name for p in game.iterdir()) == original
    assert (game / "re_chunk_000.pak").read_bytes() == b"game data"


def test_other_reframework_mods_protect_reframework(game, bundle):
    install_all(game, bundle)
    other = game / "reframework" / "autorun" / "SomeOtherMod.lua"
    other.write_text("--", encoding="utf-8")
    (game / "reframework" / "plugins").mkdir()
    (game / "reframework" / "plugins" / "native.dll").write_bytes(b"x")
    assert other_reframework_mods(game) == ("reframework/autorun/SomeOtherMod.lua", "reframework/plugins/native.dll")
    assert inspect_game(game, bundle).other_mods
    with pytest.raises(ValueError):
        remove_from_game(game, remove_reframework=True)
    remove_from_game(game)
    assert other.exists() and (game / REFRAMEWORK_DLL).exists()


def test_leftover_files_of_other_tools_are_reported_not_deleted(game, bundle):
    install_all(game, bundle)
    keep = game / "reframework" / "data" / "SomeTool" / "settings.json"
    keep.parent.mkdir(parents=True)
    keep.write_text("{}", encoding="utf-8")
    report = remove_from_game(game, remove_reframework=True)
    assert report.kept == ("reframework/data/SomeTool/settings.json",)
    assert keep.exists() and not (game / REFRAMEWORK_DLL).exists()
