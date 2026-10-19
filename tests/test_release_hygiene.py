from email.parser import Parser
from pathlib import Path
import os
import subprocess
import sys
from zipfile import ZipFile

from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).parents[1]


def _git(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _build_current_wheel(tmp_path: Path) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
            str(ROOT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(tmp_path.glob("sf6_music_trigger-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_runtime_data_paths_are_actually_ignored():
    candidates = (
        "music/test.mp3",
        "diagnostic_captures/event/frame.png",
        "logs/trigger.log",
        "screenshots/frame.png",
        "app_config.json",
        "input_config.json",
        "profiles/default.json",
        "models/model.onnx",
        "src/sf6_music_trigger.egg-info/PKG-INFO",
        "private-song.mp3",
        "private-song.wav",
        "private-song.flac",
    )
    result = _git("check-ignore", "-z", "--stdin", input_text="\0".join(candidates) + "\0")
    assert result.returncode == 0, result.stderr
    assert [value for value in result.stdout.split("\0") if value] == list(candidates)


def test_release_candidate_contains_no_private_runtime_assets_or_user_paths():
    listed = _git("ls-files", "-co", "--exclude-standard", "-z")
    assert listed.returncode == 0, listed.stderr
    paths = [Path(value) for value in listed.stdout.split("\0") if value]
    forbidden_roots = {
        "music",
        "diagnostic_captures",
        "logs",
        "screenshots",
        "profiles",
        "models",
    }
    forbidden_names = {"app_config.json", "input_config.json", "user_config.json"}
    forbidden_suffixes = {".mp3", ".wav", ".flac", ".onnx"}
    assert not any(
        path.parts[0] in forbidden_roots
        or path.name in forbidden_names
        or path.suffix.lower() in forbidden_suffixes
        for path in paths
    )

    user = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown-user"
    forbidden = (f"C:\\Users\\{user}", "OneDrive\\Desktop\\音效")
    for relative_path in paths:
        path = ROOT / relative_path
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".toml", ".json", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            assert not any(value in text for value in forbidden), path


def test_fresh_wheel_has_expected_metadata_and_working_console_script(tmp_path: Path):
    with ZipFile(_build_current_wheel(tmp_path)) as wheel:
        names = wheel.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        assert "sf6_music_trigger/__main__.py" in names
        project = Parser().parsestr(wheel.read(metadata_name).decode("utf-8"))
        assert SpecifierSet(project["Requires-Python"]) == SpecifierSet(">=3.12,<3.13")
        assert (project.get("License-Expression") or project.get("License")) == "MIT"
        assert wheel.read(entry_points_name).decode("utf-8") == (
            "[console_scripts]\n"
            "sf6-music-trigger = sf6_music_trigger.__main__:main\n"
        )
        description = project.get_payload()
        assert "choose their own music" in description
        assert "does not bundle commercial songs" in description
        assert "not affiliated with or endorsed by Capcom" in description
