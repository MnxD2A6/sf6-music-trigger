"""Command-line entry point for the persistent SF6 control panel."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Sequence


def _default_config_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return root / "SF6MusicTrigger" / "config.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sf6-music-trigger",
        description="Persistent screen-only Street Fighter 6 music trigger panel.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="path to the local JSON configuration file",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    panel_runner: Callable[[Path], object] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if panel_runner is None:
        from .panel import run_panel

        panel_runner = run_panel
    panel_runner(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
