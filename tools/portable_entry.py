"""Frozen executable entry point for the portable group-test build."""

from __future__ import annotations

import os
import sys
import traceback
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Sequence


WINRT_CAPTURE_MODULES = (
    "winrt.runtime",
    "winrt.windows.foundation",
    "winrt.windows.foundation.collections",
    "winrt.windows.graphics",
    "winrt.windows.graphics.capture",
    "winrt.windows.graphics.capture.interop",
    "winrt.windows.graphics.directx",
    "winrt.windows.graphics.directx.direct3d11",
    "winrt.windows.graphics.directx.direct3d11.interop",
)
REQUIRED_PORTABLE_MODULES = (
    "numpy",
    "cv2",
    "pygame",
    "pygame.sndarray",
    "dxcam",
    "tkinter",
    *WINRT_CAPTURE_MODULES,
)


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def configure_frozen_tk(root: Path) -> None:
    if not getattr(sys, "frozen", False):
        return
    internal = root / "_internal"
    os.environ["TCL_LIBRARY"] = str(internal / "_tcl_data")
    os.environ["TK_LIBRARY"] = str(internal / "_tk_data")


def self_check(root: Path) -> None:
    from sf6_music_trigger.config import load_config
    from sf6_music_trigger.hud_profile import load_profile

    config = load_config(root / "config.json")
    load_profile(root, config.profile_path)
    missing = [name for name in REQUIRED_PORTABLE_MODULES if find_spec(name) is None]
    if missing:
        raise RuntimeError(f"portable dependencies are missing: {', '.join(missing)}")
    for module_name in WINRT_CAPTURE_MODULES:
        import_module(module_name)
    import_module("pygame.sndarray")  # volumes above 100 %

    from sf6_music_trigger.reframework_events import GAME_SCRIPTS, bundled_game_script

    missing_scripts = [name for name in GAME_SCRIPTS if not bundled_game_script(name).is_file()]
    if missing_scripts:
        raise RuntimeError(f"bundled game scripts are missing: {', '.join(missing_scripts)}")

    import tkinter

    interpreter = tkinter.Tcl()
    interpreter.eval("info patchlevel")


def main(
    argv: Sequence[str] | None = None,
    *,
    app_main: Callable[[Sequence[str]], int] | None = None,
    self_checker: Callable[[Path], None] | None = None,
    exit_process: Callable[[int], object] | None = None,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = runtime_root()
    configure_frozen_tk(root)
    if arguments == ["--portable-self-check"]:
        try:
            (self_checker or self_check)(root)
        except Exception:
            if getattr(sys, "frozen", False):
                (root / "PORTABLE_SELF_CHECK_ERROR.txt").write_text(
                    traceback.format_exc(),
                    encoding="utf-8",
                )
                (exit_process or os._exit)(1)
                return 1
            raise
        if getattr(sys, "frozen", False):
            (exit_process or os._exit)(0)
        return 0
    if app_main is None:
        from sf6_music_trigger.__main__ import main as app_main

    return app_main(["--config", str(root / "config.json")])


if __name__ == "__main__":
    raise SystemExit(main())
