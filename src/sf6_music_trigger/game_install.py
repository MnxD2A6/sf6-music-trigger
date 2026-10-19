"""One-click install and clean removal of V2 (and the bundled REFramework) in SF6.

Everything V2 puts into the game folder:

* ``dinput8.dll`` - REFramework loader (bundled copy, MIT licensed); REFramework
  itself then creates ``reframework/``, ``re2_fw_config.txt`` and
  ``re2_framework_log.txt``.
* ``reframework/autorun/sf6_music_trigger*.lua`` - detector and settings window.
* ``reframework/data/SF6MusicTrigger/`` - event log and settings exchange files.

Nothing of the game's own files is modified. Removal never touches REFramework
while other REFramework mods (scripts or plugins) are present.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import time

from .reframework_events import GAME_SCRIPTS, detector_installed, install_detector, is_game_dir

REFRAMEWORK_DLL = "dinput8.dll"
BUNDLE_DIR_NAME = "REFramework"
REFRAMEWORK_ROOT_FILES = (REFRAMEWORK_DLL, "re2_fw_config.txt", "re2_framework_log.txt")
V2_SCRIPTS = (*GAME_SCRIPTS, "sf6_music_trigger_probe.lua")
V2_DATA_DIR = Path("reframework") / "data" / "SF6MusicTrigger"


def bundled_reframework_dir() -> Path | None:
    """The REFramework copy shipped with V2 (release: next to the exe)."""

    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / BUNDLE_DIR_NAME)
    # Source checkout: third_party/REFramework (not committed, see README).
    candidates.append(Path(__file__).resolve().parents[2] / "third_party" / BUNDLE_DIR_NAME)
    for candidate in candidates:
        if (candidate / REFRAMEWORK_DLL).is_file():
            return candidate
    return None


@dataclass(frozen=True)
class GameInstallState:
    game_found: bool
    reframework: str            # "missing" | "bundled" | "other"
    scripts_current: bool
    other_mods: tuple[str, ...]  # other REFramework scripts/plugins, relative paths


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def other_reframework_mods(game_dir: str | os.PathLike[str]) -> tuple[str, ...]:
    root = Path(game_dir) / "reframework"
    found: list[str] = []
    for folder in (root / "autorun", root / "plugins"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not (path.parent == root / "autorun" and path.name in V2_SCRIPTS):
                found.append(path.relative_to(Path(game_dir)).as_posix())
    return tuple(found)


def inspect_game(game_dir: str | os.PathLike[str], bundle_dir: Path | None) -> GameInstallState:
    if not is_game_dir(game_dir):
        return GameInstallState(False, "missing", False, ())
    dll = Path(game_dir) / REFRAMEWORK_DLL
    if not dll.is_file():
        reframework = "missing"
    elif bundle_dir is not None and _same_file(dll, bundle_dir / REFRAMEWORK_DLL):
        reframework = "bundled"
    else:
        reframework = "other"
    return GameInstallState(True, reframework, detector_installed(game_dir), other_reframework_mods(game_dir))


@dataclass(frozen=True)
class InstallReport:
    reframework: str            # "installed" | "replaced" | "kept" | "unchanged"
    backup: str | None = None   # file name of the backed-up foreign dinput8.dll


def install_all(
    game_dir: str | os.PathLike[str],
    bundle_dir: Path | None,
    *,
    replace_other_dll: bool = False,
) -> InstallReport:
    """Install the bundled REFramework (if needed) and both V2 game scripts.

    A dinput8.dll that differs from the bundled one is kept unless
    ``replace_other_dll``; then it is renamed to ``dinput8.dll.bak-<time>``.
    """

    if not is_game_dir(game_dir):
        raise ValueError("folder does not contain StreetFighter6.exe")
    game = Path(game_dir)
    dll = game / REFRAMEWORK_DLL
    state = inspect_game(game_dir, bundle_dir)
    backup = None
    if state.reframework == "bundled":
        action = "unchanged"
    elif state.reframework == "other" and not replace_other_dll:
        action = "kept"
    else:
        if bundle_dir is None:
            raise FileNotFoundError("no bundled REFramework and none installed in the game")
        if state.reframework == "other":
            backup = f"{REFRAMEWORK_DLL}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
            os.replace(dll, game / backup)
        temp = game / f"{REFRAMEWORK_DLL}.tmp"
        shutil.copyfile(bundle_dir / REFRAMEWORK_DLL, temp)
        os.replace(temp, dll)
        action = "replaced" if backup else "installed"
    install_detector(game_dir)
    return InstallReport(action, backup)


@dataclass(frozen=True)
class RemoveReport:
    removed: tuple[str, ...]
    kept: tuple[str, ...]       # REFramework-related leftovers (other mods' files, their data)


def _prune_empty_dirs(root: Path) -> None:
    if not root.is_dir():
        return
    for folder in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            folder.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def remove_from_game(game_dir: str | os.PathLike[str], *, remove_reframework: bool = False) -> RemoveReport:
    """Remove V2's files from the game; optionally REFramework's own files too.

    Raises PermissionError while the game holds the files open (game running).
    """

    if not is_game_dir(game_dir):
        raise ValueError("folder does not contain StreetFighter6.exe")
    game = Path(game_dir)
    if remove_reframework and other_reframework_mods(game_dir):
        raise ValueError("other REFramework mods are installed; keep REFramework")
    removed: list[str] = []
    autorun = game / "reframework" / "autorun"
    for name in V2_SCRIPTS:
        script = autorun / name
        if script.is_file():
            script.unlink()
            removed.append(script.relative_to(game).as_posix())
    data = game / V2_DATA_DIR
    if data.is_dir():
        shutil.rmtree(data)
        removed.append(V2_DATA_DIR.as_posix() + "/")
    if remove_reframework:
        for name in REFRAMEWORK_ROOT_FILES:
            path = game / name
            if path.is_file():
                path.unlink()
                removed.append(name)
        _prune_empty_dirs(game / "reframework")
    leftovers = ()
    root = game / "reframework"
    if remove_reframework and root.exists():
        leftovers = tuple(p.relative_to(game).as_posix() for p in sorted(root.rglob("*")) if p.is_file())
    return RemoveReport(tuple(removed), leftovers)
