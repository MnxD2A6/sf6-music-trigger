# Foundation and Gauge Trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable open-source release candidate: a persistent Windows panel that detects real P1/P2 SA3/CA spend from calibrated HUD pixels and immediately plays each side’s main track without reading controls or game data.

**Architecture:** The application uses a source-layout Python package. Window capture, HUD profiles, deterministic gauge observations, per-side state machines, audio policy, orchestration, and UI lifecycle are separate modules connected by immutable dataclasses. Live adapters are dependency-injected so all state and scheduling behavior can be replayed without launching SF6.

**Tech Stack:** Python 3.12, NumPy 1.26+, OpenCV 4.9+, DXcam 0.0.5+, pygame 2.5+, Tkinter, pytest 8+, PyInstaller deferred to the packaging phase.

## Global Constraints

- Work only in this new repository; do not modify the sibling legacy `SF6_Music_Trigger` project.
- Support Windows 10 22H2 and Windows 11 23H2 or newer, Steam PC Street Fighter 6 only.
- The canonical trigger is pixels: no keyboard/gamepad mapping is required, and no input edge can create an event.
- Never read game memory, inject code, modify game files, synthesize input, inspect network traffic, or require administrator/kernel components.
- Do not commit commercial music, old templates, screenshots, diagnostics, logs, personal configuration, or machine-specific absolute paths.
- Every production behavior is implemented through a verified RED -> GREEN test cycle.
- P1 and P2 own independent tracker, scheduler, channel, and cooldown state.
- Missing, stale, minimized, or invalid capture cannot generate `ConfirmedSuperEvent`.
- Phase 1 plays the main track after confirmed spend and does not claim hit/block/whiff classification.

---

## File Map

```text
SF6_Music_Trigger_Full/
├─ .gitignore                     repository privacy and build exclusions
├─ LICENSE                        MIT source license
├─ README.md                      install, safety boundary, phase-1 behavior
├─ PRIVACY.md                     local processing and diagnostic policy
├─ THIRD_PARTY.md                 dependency license inventory
├─ pyproject.toml                 package metadata and pinned capture dependency
├─ src/sf6_music_trigger/
│  ├─ __init__.py                 package version
│  ├─ domain.py                   immutable shared event/status/capture dataclasses
│  ├─ config.py                   validated relative-path user configuration
│  ├─ hud_profile.py              normalized ROI profile, context, and safe paths
│  ├─ meter_references.py         canonical preprocessing and local references
│  ├─ gauge_features.py           validated deterministic gauge observations
│  ├─ gauge_tracker.py            ordered per-side five-slot evidence state machine
│  ├─ capture.py                  typed samples, freshness, lazy DXcam adapter
│  ├─ window.py                   stable read-only SF6 window/output identity
│  ├─ audio.py                    stateless policy and lazy fixed-channel pygame port
│  ├─ engine.py                   typed P1/P2 orchestration and contained audio dispatch
│  ├─ controller.py               cooperative worker and reusable audio lifecycle
│  ├─ panel.py                    panel, percent/preview bridge, and calibration flow
│  └─ __main__.py                 process entry point
├─ tools/replay.py                offline observation/event replay CLI
├─ tests/
│  ├─ test_release_hygiene.py
│  ├─ test_domain_config.py
│  ├─ test_hud_profile.py
│  ├─ test_meter_references.py
│  ├─ test_gauge_features.py
│  ├─ test_gauge_tracker.py
│  ├─ test_capture_window.py
│  ├─ test_audio.py
│  ├─ test_engine.py
│  ├─ test_controller.py
│  ├─ test_panel_view_model.py
│  └─ test_replay_integration.py
└─ docs/verification/             generated text evidence, no game frames
```

---

### Task 1: Clean Repository Foundation and Release Hygiene

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `README.md`
- Create: `PRIVACY.md`
- Create: `THIRD_PARTY.md`
- Create: `pyproject.toml`
- Create: `src/sf6_music_trigger/__init__.py`
- Create: `tests/test_release_hygiene.py`

**Interfaces:**
- Produces: installable `sf6-music-trigger` package, Python version floor, pytest configuration, repository exclusion contract.
- Consumes: approved design specification and global constraints only.

- [ ] **Step 0: Create the project virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "pytest>=8,<9"
```

Expected: `.venv\Scripts\python.exe` reports Python 3.12 and pytest imports successfully. If the machine’s launcher is not named `python`, use its installed Python 3.12 executable for this command without writing that machine-specific path into repository files.

- [ ] **Step 1: Write the release-hygiene RED test**

```python
from email.parser import Parser
from pathlib import Path
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

    forbidden = ("C:\\Users\\<user>", "OneDrive\\Desktop\\音效")
    for relative_path in paths:
        path = ROOT / relative_path
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".toml", ".json", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            assert not any(value in text for value in forbidden), path


def test_fresh_wheel_has_expected_metadata_and_no_broken_console_script(tmp_path: Path):
    with ZipFile(_build_current_wheel(tmp_path)) as wheel:
        names = wheel.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        project = Parser().parsestr(wheel.read(metadata_name).decode("utf-8"))
        assert SpecifierSet(project["Requires-Python"]) == SpecifierSet(">=3.12,<3.13")
        assert (project.get("License-Expression") or project.get("License")) == "MIT"
        assert not any(name.endswith(".dist-info/entry_points.txt") for name in names)
        description = project.get_payload()
        assert "choose their own music" in description
        assert "does not bundle commercial songs" in description
        assert "not affiliated with or endorsed by Capcom" in description
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_release_hygiene.py -v
```

Expected: FAIL because Git does not yet ignore the runtime candidates and the `sf6-music-trigger` distribution is not installed. This is a behavior-level RED: the test does not pass merely because certain strings were written into repository files.

- [ ] **Step 3: Add the minimal repository files**

Use this dependency set in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sf6-music-trigger"
version = "0.1.0a1"
description = "Screen-only local music triggers for Street Fighter 6 on Windows"
readme = "README.md"
requires-python = ">=3.12,<3.13"
license = "MIT"
dependencies = [
  "numpy>=1.26,<3",
  "opencv-python>=4.9,<5",
  "dxcam>=0.0.5,<1",
  "pygame>=2.5,<3",
]

[project.optional-dependencies]
dev = [
  "pytest>=8,<9",
  "pytest-cov>=5,<6",
  "setuptools>=75",
  "wheel>=0.45,<1",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-ra"
```

`.gitignore` must contain exactly scoped runtime/build exclusions that make every runtime candidate in the behavior test resolve through `git check-ignore`, including `*.mp3`, `*.wav`, and `*.flac`, plus `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `dist/`, `build/`, `profiles/`, `models/`, and `user_config.json`.

Do not register the `sf6-music-trigger` console script until Task 9 creates and tests `sf6_music_trigger.__main__:main`; Phase 1 Task 1 must not install a broken entry point.

`README.md` must state that Phase 1 confirms visible meter spend and plays main audio; it does not classify hit/block/whiff. It must also say that users select their own music, no commercial songs are bundled, and this is an unofficial community project not affiliated with or endorsed by Capcom. `PRIVACY.md` must state that processing is local and capture saving is opt-in. `THIRD_PARTY.md` must list NumPy BSD-3-Clause, OpenCV Apache-2.0, DXcam MIT, pygame LGPL-2.1, and pytest MIT.

Install the project and development dependencies after creating `pyproject.toml`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- [ ] **Step 4: Run the focused test and full collection**

Run the Step 2 command, then:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected: both commands PASS with no warnings caused by project files.

- [ ] **Step 5: Commit**

```powershell
git add .gitignore LICENSE README.md PRIVACY.md THIRD_PARTY.md pyproject.toml src/sf6_music_trigger/__init__.py tests/test_release_hygiene.py
git commit -m "chore: establish clean open source foundation"
```

---

### Task 2: Domain Types and Validated User Configuration

**Files:**
- Create: `src/sf6_music_trigger/domain.py`
- Create: `src/sf6_music_trigger/config.py`
- Create: `tests/test_domain_config.py`

**Interfaces:**
- Produces: `Side`, `CaptureStatus`, `TrackerState`, `ClientGeometry`, `CapturedFrame`, `ConfirmedSuperEvent`, `SideAudioConfig`, `AppConfig`, `load_config`, `save_config`.
- Consumes: Python standard library and NumPy only.

- [ ] **Step 1: Write RED tests for immutable events and safe configuration**

```python
from dataclasses import FrozenInstanceError
from pathlib import Path
import json

import numpy as np
import pytest

from sf6_music_trigger.config import AppConfig, SideAudioConfig, load_config, save_config
from sf6_music_trigger.domain import CapturedFrame, ClientGeometry, ConfirmedSuperEvent, Side


def test_confirmed_event_is_immutable_and_side_typed():
    event = ConfirmedSuperEvent("evt-1", Side.P1, 12.5, 42, 45, "default")
    with pytest.raises(FrozenInstanceError):
        event.event_id = "changed"

    with pytest.raises((TypeError, ValueError)):
        ConfirmedSuperEvent("evt-2", "P1", 12.5, 42, 45, "default")


def test_captured_frame_is_a_frozen_envelope_with_identity_equality():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    geometry = ClientGeometry(-1920, 0, 1920, 1080)
    first = CapturedFrame(1, image, 1.25, geometry)
    second = CapturedFrame(1, image.copy(), 1.25, geometry)
    assert first != second
    with pytest.raises(FrozenInstanceError):
        first.frame_id = 2


@pytest.mark.parametrize("audio_path", [
    "D:/Music/song.mp3",
    r"C:\Music\song.mp3",
    r"\\server\share\song.mp3",
])
def test_runtime_config_preserves_user_selected_absolute_audio_path(tmp_path: Path, audio_path: str):
    path = tmp_path / "user_config.json"
    path.write_text(json.dumps({"p1": {"main_track": audio_path}}), encoding="utf-8")
    assert load_config(path).p1.main_track == audio_path


def test_partial_config_clamps_volume_and_fills_defaults(tmp_path: Path):
    path = tmp_path / "user_config.json"
    path.write_text(json.dumps({"p1": {"main_track": "audio/p1.mp3", "volume": 4.0}}), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.p1.main_track == "audio/p1.mp3"
    assert loaded.p1.volume == 1.0
    assert loaded.p2 == SideAudioConfig()


@pytest.mark.parametrize("payload", [
    {"profile_path": "../outside.json"},
    {"profile_path": r"..\outside.json"},
    {"profile_path": "D:/profiles/outside.json"},
    {"profile_path": r"D:profiles\outside.json"},
    {"profile_path": r"\profiles\outside.json"},
    {"profile_path": r"\\server\share\outside.json"},
    {"profile_path": r"\\?\C:\profiles\outside.json"},
    {"profile_path": r"\\.\C:\profiles\outside.json"},
    {"profile_path": r"profiles\profile.json:stream"},
    {"profile_path": ""},
    {"profile_path": "."},
    {"profile_path": "profiles/"},
    {"profile_path": "profiles/."},
    {"profile_path": "profiles\\"},
    {"profile_path": "profiles\\."},
    {"p1": {"main_track": "../private.mp3"}},
    {"p1": {"main_track": r"D:music\private.mp3"}},
    {"p1": {"main_track": r"\music\private.mp3"}},
    {"p1": {"main_track": r"\\?\C:\music\private.mp3"}},
    {"p1": {"main_track": r"audio\private.mp3:stream"}},
    {"capture_fps": 9},
    {"capture_fps": 31},
    {"capture_fps": True},
    {"capture_fps": 20.0},
    {"p1": {"volume": float("nan")}},
    {"p1": {"volume": float("inf")}},
])
def test_config_rejects_escaping_paths_and_invalid_types(tmp_path: Path, payload: dict):
    path = tmp_path / "user_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_config(path)


def test_diagnostic_config_redacts_audio_paths():
    config = AppConfig(
        p1=SideAudioConfig(main_track="D:/Music/private-name.mp3"),
        p2=SideAudioConfig(main_track=r"\\server\share\second-private.wav"),
    )
    diagnostic = json.dumps(config.diagnostic_view())
    assert "private-name" not in diagnostic
    assert "second-private" not in diagnostic
    assert diagnostic.count("<redacted>") == 2


def test_config_schema_and_unknown_fields_fail_closed(tmp_path: Path):
    path = tmp_path / "user_config.json"
    for payload in ({"schema_version": 2}, {"unknown_option": True}, {"p1": {"typo_volume": 0.5}}):
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


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
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_domain_config.py -v
```

Expected: collection ERROR because `sf6_music_trigger.config` and `domain` do not exist. Do not add those modules before capturing this RED.

- [ ] **Step 3: Implement minimal immutable types and JSON config**

Use frozen dataclasses and string enums. Required signatures:

```python
class Side(str, Enum):
    P1 = "P1"
    P2 = "P2"


class CaptureStatus(str, Enum):
    RUNNING = "RUNNING"
    OCCLUDED = "OCCLUDED"
    MINIMIZED = "MINIMIZED"
    LOST = "LOST"
    STOPPED = "STOPPED"


class TrackerState(str, Enum):
    DISARMED = "DISARMED"
    ARMED = "ARMED"
    SPEND_CANDIDATE = "SPEND_CANDIDATE"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class ClientGeometry:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True, eq=False)
class CapturedFrame:
    frame_id: int
    image: np.ndarray
    monotonic_time: float
    geometry: ClientGeometry


@dataclass(frozen=True)
class ConfirmedSuperEvent:
    event_id: str
    side: Side
    confirmed_at: float
    pre_spend_frame_id: int
    confirmed_frame_id: int
    profile_id: str


@dataclass(frozen=True)
class SideAudioConfig:
    enabled: bool = True
    main_track: str = ""
    volume: float = 0.18


@dataclass(frozen=True)
class AppConfig:
    schema_version: int = 1
    p1: SideAudioConfig = field(default_factory=SideAudioConfig)
    p2: SideAudioConfig = field(default_factory=SideAudioConfig)
    profile_path: str = "profiles/default.json"
    capture_fps: int = 20
```

Contract details:

- The resolved parent of the configuration file passed to `load_config(path)` / `save_config(path, ...)` is the user-data root. `profile_path` is stored as a relative path and its resolved target must remain inside that root. Task 3 applies the equivalent rule to reference paths relative to the validated profile file.
- `main_track` is the explicit exception: normal drive-rooted and UNC absolute audio paths are preserved exactly because the user selected them. A relative audio path is allowed only when it remains inside the user-data root. Reject Windows device namespaces (`\\?\\`, `\\.\\`), drive-relative forms such as `D:music\\song.mp3`, root-relative forms such as `\\music\\song.mp3`, traversal escapes, and NTFS alternate-data-stream syntax. Do not resolve or normalize an accepted external absolute audio string before storing it.
- A missing configuration file and omitted fields load safe defaults. Configured audio/profile paths are validated for type and containment but not existence; preserve them so the panel/profile loader can report a missing or unreadable selection and keep Start disabled. Missing `schema_version` means version 1 for compatibility; any unsupported schema or unknown mapping key is rejected so misspellings cannot silently disappear.
- Reject an empty/directory-equivalent profile path, including any trailing slash/backslash or raw final component `.` / `..`; also reject drive/root/device-qualified profile paths and relative profile paths containing NTFS alternate-data-stream syntax. On Windows, do not rely on `Path.is_absolute()` alone: any otherwise-relative path carrying a `.drive` or `.root` component is invalid.
- Require actual booleans for `enabled`; strings for paths; finite real numbers other than `bool` for volume. Clamp finite volume to `[0.0, 1.0]`. Non-boolean Python integers are already finite and must not be forced through a float conversion that can raise `OverflowError`. Require `capture_fps` to be a real `int` other than `bool` in the closed range `10..30`; reject rather than clamp anything outside it.
- `ClientGeometry` accepts negative desktop origins but requires integer coordinates and positive integer dimensions. Events require typed `Side`, finite nonnegative time, nonnegative ordered frame IDs, and nonblank IDs. `CapturedFrame` requires a non-boolean integer `frame_id >= 0`, a finite nonnegative `monotonic_time`, an actual NumPy array, and a `ClientGeometry`; it is a frozen, identity-equality envelope. For 20 FPS performance it does not copy the NumPy pixel buffer, and the capture adapter must not mutate emitted frames.
- `diagnostic_view()` replaces every nonempty audio path for both sides with `"<redacted>"` and must never expose a basename or root.
- `save_config` validates and serializes with `allow_nan=False` before touching the target; creates the parent after validation; writes a uniquely named same-directory `.tmp`, flushes and `fsync`s, then calls `os.replace` for only the exact target. On failure it removes only its own temp file and leaves the prior target and neighboring files unchanged.

Add focused cases beyond the compact snippet for JSON `NaN`/`Infinity`, non-object roots/sides, non-string fields, lower volume clamp (including an arbitrarily large finite integer), FPS boundaries `10` and `30`, missing config defaults, atomic successful round-trip, and invalid domain scalars (including arbitrarily large nonnegative integer times). Test profile and audio sibling-prefix escapes in separate payloads so one rejection cannot mask the other. For existing-link containment, try a directory symlink and fall back to an unprivileged Windows junction if symlink privilege is denied; both profile and relative audio assertions must actually execute whenever either mechanism is available. Tests must assert external behavior rather than implementation wording.

- [ ] **Step 4: Run focused and full tests**

Expected: focused tests and all existing tests PASS. Also run `git diff --check` and confirm no config/audio/profile fixture escaped pytest's temp directory.

- [ ] **Step 5: Commit**

```powershell
git add src/sf6_music_trigger/domain.py src/sf6_music_trigger/config.py tests/test_domain_config.py
git commit -m "feat: add typed events and safe local configuration"
```

---

### Task 3: HUD Profiles and Local Meter References

**Files:**
- Create: `src/sf6_music_trigger/hud_profile.py`
- Create: `src/sf6_music_trigger/meter_references.py`
- Create: `tests/test_hud_profile.py`
- Create: `tests/test_meter_references.py`

**Interfaces:**
- Produces from `hud_profile.py`: `MAX_PROFILE_BYTES`, `NormalizedRoi`, `CalibrationContext`, `SideHudProfile`, `HudProfile`, `crop_normalized`, `resolve_contained_path`, `save_profile`, `load_profile`.
- Produces from `meter_references.py`: `REFERENCE_SCHEMA_VERSION`, `PREPROCESSING_VERSION`, `METER_OUTPUT_SHAPE`, `MIN_REFERENCE_DISTANCE`, `MAX_REFERENCE_ARCHIVE_BYTES`, `MAX_NPY_MEMBER_BYTES`, `MAX_NPZ_UNCOMPRESSED_BYTES`, `PreprocessedMeter`, `MeterReferences`, `preprocess_meter`, `build_meter_references`, `save_references`, `load_references`.
- Consumes: Task 2 `Side`, Python standard library, NumPy, and OpenCV. Do not import Task 2 private path helpers or change Task 2's public contract.

- [ ] **Step 1: Write RED tests for profiles, canonical preprocessing, containment, and persistence**

```python
import json

import numpy as np
import pytest

from sf6_music_trigger.domain import Side
from sf6_music_trigger.hud_profile import (
    CalibrationContext,
    HudProfile,
    NormalizedRoi,
    SideHudProfile,
    crop_normalized,
    load_profile,
    save_profile,
)
from sf6_music_trigger.meter_references import (
    METER_OUTPUT_SHAPE,
    build_meter_references,
    load_references,
    preprocess_meter,
    save_references,
)


ROI = NormalizedRoi(0.0, 0.0, 1.0, 1.0)


def bgr(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def calibration_crops() -> tuple[list[np.ndarray], list[np.ndarray]]:
    full = np.tile(np.arange(0, 256, 4, dtype=np.uint8), (32, 1))
    spent = np.flip(full, axis=1).copy()
    return [bgr(full.copy()) for _ in range(5)], [bgr(spent.copy()) for _ in range(5)]


def sample_profile() -> HudProfile:
    context = CalibrationContext(
        client_width=1920,
        client_height=1080,
        client_aspect_ratio=1920 / 1080,
        hud_scale_setting="100%",
        language="zh-CN",
        brightness_setting="default",
        hdr_enabled=False,
        game_version_note="phase-1 fixture",
    )
    return HudProfile(
        profile_id="fixture",
        calibration=context,
        battle_roi=ROI,
        training_hud_roi=None,
        p1=SideHudProfile(ROI, "refs/p1.npz"),
        p2=SideHudProfile(ROI, "refs/p2.npz"),
    )


def test_crop_normalized_maps_exact_pixels():
    frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    roi = NormalizedRoi(x0=0.25, y0=0.50, x1=0.75, y1=0.90)
    crop = crop_normalized(frame, roi)
    assert crop.shape == (40, 100, 3)
    assert np.array_equal(crop[0, 0], frame[50, 50])
    assert crop.flags.owndata
    frame[50, 50] = 0
    assert np.any(crop[0, 0] != 0)


def test_profile_round_trip_records_context_and_optional_rois(tmp_path):
    profile = sample_profile()
    save_profile(tmp_path, "profiles/default.json", profile)
    loaded = load_profile(tmp_path, "profiles/default.json")
    assert loaded == profile
    assert loaded.battle_roi == ROI
    assert loaded.training_hud_roi is None


def test_reference_round_trip_is_canonical_sided_and_owned(tmp_path):
    full_crops, spent_crops = calibration_crops()
    refs = build_meter_references(Side.P1, full_crops, spent_crops)
    save_references(tmp_path, "profiles/default.json", "refs/p1.npz", refs)
    loaded = load_references(
        tmp_path,
        "profiles/default.json",
        "refs/p1.npz",
        expected_side=Side.P1,
    )
    assert loaded.full_gray.shape == METER_OUTPUT_SHAPE
    assert loaded.full_gray.dtype == np.uint8
    assert loaded.full_gray.flags.owndata and not loaded.full_gray.flags.writeable
    assert loaded.spent_gray.flags.owndata and not loaded.spent_gray.flags.writeable
    assert not np.shares_memory(loaded.full_gray, loaded.spent_gray)


def test_percentile_degenerate_crop_is_zero_and_hidden():
    sample = preprocess_meter(np.full((40, 80, 3), 120, np.uint8))
    assert np.array_equal(sample.normalized_gray, np.zeros(METER_OUTPUT_SHAPE, np.uint8))
    assert not sample.hud_visible


def test_profile_schema_and_reference_side_fail_closed(tmp_path):
    profile_path = tmp_path / "profiles" / "default.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(tmp_path, "profiles/default.json")

    full_crops, spent_crops = calibration_crops()
    refs = build_meter_references(Side.P1, full_crops, spent_crops)
    save_references(tmp_path, "profiles/default.json", "refs/p1.npz", refs)
    with pytest.raises(ValueError, match="side"):
        load_references(
            tmp_path,
            "profiles/default.json",
            "refs/p1.npz",
            expected_side=Side.P2,
        )


def test_profile_read_is_bounded_before_json_decode(tmp_path):
    path = tmp_path / "profiles" / "oversized"
    path.parent.mkdir(parents=True)
    path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="64 KiB"):
        load_profile(tmp_path, "profiles/oversized")


def test_npz_duplicate_member_and_oversized_header_fail_preflight(tmp_path, archive_factory):
    duplicate = archive_factory.duplicate_member("full_gray.npy")
    oversized = archive_factory.npy_header("full_gray.npy", shape=(1_000_000_000,))
    for name, payload in (("duplicate.npz", duplicate), ("oversized.npz", oversized)):
        archive_factory.write_reference(tmp_path, name, payload)
        with pytest.raises(ValueError):
            load_references(
                tmp_path,
                "profiles/default",
                f"refs/{name}",
                expected_side=Side.P1,
            )
```

- [ ] **Step 2: Complete the mandatory RED edge-case matrix before production work**

Add focused parametrized tests for every item below now, while both production modules are still absent. These tests are part of the initial RED suite; do not defer any item until after a Task 3 implementation step:

- ROI/context/profile constructors: frozen-object mutation; equality/ordering failures; negative/out-of-range, bool, NaN/Infinity, wrong object/dtype/string values; battle ROI required; training HUD ROI accepted only as `None` or `NormalizedRoi`; and inconsistent stored resolution/aspect. Constructors reject invalid values rather than clamp, stringify, cast, resize, or otherwise coerce them.
- JSON: exact round trip with battle/training ROIs; missing/unknown keys at every level; root/nested non-object; unsupported or missing schema; JSON `NaN`/`Infinity`; invalid UTF-8; exactly `64 KiB` accepted when otherwise valid; `64 KiB + 1` rejected after a bounded `limit + 1` read and before JSON decoding.
- Paths: forward/backslash lexical traversal escapes, sibling-prefix, drive-rooted, UNC, drive-relative, root-relative, device namespace, ADS, empty/dot/directory-equivalent, and existing symlink/junction escape when host privileges permit. A profile filename has no suffix policy. Reference paths accept `.npz`, `.NPZ`, and mixed-case `.NpZ` case-insensitively, while `.npz.exe` and other final suffixes fail. Test equal resolved P1/P2 paths and two distinct existing hardlink names detected by `os.path.samefile`.
- Crop/preprocessing: malformed BGR shape/dtype, floor/ceil/clamp, crop ownership, black/white/low-contrast safety, and the `p95 == p05` zero-output rule. Include two literal golden vectors whose expected `uint8` bytes are stored in the test rather than recomputed with OpenCV: an asymmetric BGR-unequal crop smaller than `(32, 64)` to lock the upscale path and an asymmetric BGR-unequal crop larger than `(32, 64)` to lock the downscale path. Together they must detect BGR-versus-RGB conversion, `INTER_AREA`, operation order, Gaussian border mode, percentile method, float32 normalization, and `np.rint`.
- Reference construction/ownership: not exactly five samples; hidden sample among otherwise valid samples; wrong side type; insufficient full/spent distance; wrong constructor ndarray dtype/shape; caller source mutation after construction; assignment to arrays exposed by `PreprocessedMeter` or `MeterReferences`; and no sharing between full/spent or separately built P1/P2 arrays. Tests must also prove input crops and `CapturedFrame.image` remain writable and never have their flags changed by preprocessing.
- NPZ envelope: exact valid round trip below every bound; outer archive `1 MiB + 1` rejected before ZIP parsing; missing, extra, or duplicate ZIP names; directory entries; encrypted flags; non-`ZIP_DEFLATED` entries; declared per-member uncompressed `16 KiB + 1`; declared total uncompressed `64 KiB + 1`; truncated/bad-CRC members; truncated or appended payload bytes after an otherwise valid `.npy` header; object arrays; malformed `.npy`; oversized declared shape in a small `.npy` header; wrong header version, Fortran flag, scalar shape/dtype/value, reference/preprocessing version, output shape/image dtype, or stored/expected side.
- Atomic JSON/NPZ: successful round trip leaves no temp; injected replace failure preserves old bytes and unrelated neighbor, cleans only the uniquely owned temp, and demonstrates unique temp names across attempts.

- [ ] **Step 3: Run the complete focused suite and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hud_profile.py tests/test_meter_references.py -v
```

Expected: collection ERROR because `hud_profile` and `meter_references` do not exist. Confirm the representative tests and the entire mandatory matrix are present before capturing this RED; do not add either production module first.

- [ ] **Step 4: Implement strict profile and contained-path contracts**

Use these exact public dataclasses and signatures in `hud_profile.py`:

```python
MAX_PROFILE_BYTES = 64 * 1024


@dataclass(frozen=True)
class NormalizedRoi:
    x0: float
    y0: float
    x1: float
    y1: float


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


@dataclass(frozen=True)
class SideHudProfile:
    meter_roi: NormalizedRoi
    references_path: str


@dataclass(frozen=True)
class HudProfile:
    profile_id: str
    calibration: CalibrationContext
    battle_roi: NormalizedRoi
    training_hud_roi: NormalizedRoi | None
    p1: SideHudProfile
    p2: SideHudProfile
    schema_version: int = 1


def crop_normalized(frame_bgr: np.ndarray, roi: NormalizedRoi) -> np.ndarray: ...

def resolve_contained_path(
    root: str | os.PathLike[str],
    relative_path: str,
    *,
    field_name: str,
    required_suffix: str | None = None,
) -> Path: ...

def load_profile(
    user_data_root: str | os.PathLike[str], profile_path: str
) -> HudProfile: ...

def save_profile(
    user_data_root: str | os.PathLike[str],
    profile_path: str,
    profile: HudProfile,
) -> None: ...
```

Validation and JSON schema are exact, not best-effort:

- `NormalizedRoi` accepts finite real non-booleans satisfying `0 <= x0 < x1 <= 1` and `0 <= y0 < y1 <= 1`; it retains accepted numeric values and does not coerce strings, clamp, or silently normalize bad input.
- `CalibrationContext.client_width` and `client_height` are actual positive `int` values; `client_aspect_ratio` is a finite positive real non-boolean equal to `client_width / client_height` within absolute tolerance `1e-6`. `hud_scale_setting`, `language`, and `brightness_setting` are nonblank strings; use the explicit string `"unknown"` when unavailable. `hdr_enabled` is an actual boolean and `game_version_note` is a string that may be empty.
- `HudProfile.profile_id` is nonblank; `battle_roi` is required; `training_hud_roi` is either `None` or a `NormalizedRoi`; both side values have the declared dataclass types; and schema version is exactly `1`. Public dataclass constructors reject wrong nested types rather than converting mappings or arrays for the caller.
- JSON root keys are exactly `schema_version`, `profile_id`, `calibration`, `battle_roi`, `training_hud_roi`, `p1`, and `p2`. Calibration keys are exactly the eight dataclass field names. Each side has exactly `meter_roi` and `references_path`; every non-null ROI has exactly `x0`, `y0`, `x1`, `y1`; `training_hud_roi` may be JSON `null`. Reject missing or unknown keys, non-object mappings, wrong scalar types, unsupported versions, JSON `NaN`/`Infinity`, and inconsistent aspect metadata.
- `crop_normalized` accepts only a nonempty `uint8` `H x W x 3` BGR array. It uses floor for start pixels, ceil for end pixels, clamps to frame bounds, rejects an empty result, and returns a C-contiguous owned copy rather than a view into `CapturedFrame.image`.

`resolve_contained_path` resolves one candidate only and performs Windows path classification plus containment; it never compares P1/P2 aliases or opens the target file. Without importing Task 2 private helpers, reject empty or directory-equivalent strings, absolute/UNC/device-namespace, drive-relative, root-relative, ADS, and lexical traversal escapes. Benign internal normalization such as `subdir/../file` is not itself an escape. Resolve the root and candidate with `strict=False` (which may inspect already-existing path components), then use component-aware `relative_to` so sibling-prefix and already-existing symlink/junction escapes fail. Apply a case-insensitive `required_suffix=".npz"` only to reference paths; profile filenames have no suffix requirement.

`load_profile` and `save_profile` first resolve `profile_path` beneath the user-data root with no suffix rule, then resolve each side reference beneath the resolved profile parent with `required_suffix=".npz"`. They reject P1/P2 when the two resolved paths compare equal; when both targets exist they also call `os.path.samefile` and reject distinct names for the same existing hardlink. This protects links that exist at validation time but is not a claim of adversarial race resistance if another local process swaps a reparse point between validation and open/replace; that threat is outside the Phase 1 local user-data boundary.

- [ ] **Step 5: Implement the one canonical BGR/reference representation**

Use these exact constants, dataclasses, and signatures in `meter_references.py`:

```python
REFERENCE_SCHEMA_VERSION = 1
PREPROCESSING_VERSION = 1
METER_OUTPUT_SHAPE = (32, 64)  # height, width
MIN_REFERENCE_DISTANCE = 0.08
MAX_REFERENCE_ARCHIVE_BYTES = 1 * 1024 * 1024
MAX_NPY_MEMBER_BYTES = 16 * 1024
MAX_NPZ_UNCOMPRESSED_BYTES = 64 * 1024


@dataclass(frozen=True, eq=False)
class PreprocessedMeter:
    normalized_gray: np.ndarray
    mean_luminance: float
    std_luminance: float
    hud_visible: bool


@dataclass(frozen=True, eq=False)
class MeterReferences:
    side: Side
    full_gray: np.ndarray
    spent_gray: np.ndarray
    schema_version: int = REFERENCE_SCHEMA_VERSION
    preprocessing_version: int = PREPROCESSING_VERSION


def preprocess_meter(crop_bgr: np.ndarray) -> PreprocessedMeter: ...

def build_meter_references(
    side: Side,
    full_crops: Sequence[np.ndarray],
    spent_crops: Sequence[np.ndarray],
) -> MeterReferences: ...

def load_references(
    user_data_root: str | os.PathLike[str],
    profile_path: str,
    reference_path: str,
    *,
    expected_side: Side,
) -> MeterReferences: ...

def save_references(
    user_data_root: str | os.PathLike[str],
    profile_path: str,
    reference_path: str,
    references: MeterReferences,
) -> None: ...
```

There is exactly one preprocessing order for calibration and live observations:

1. Require a nonempty actual NumPy `uint8` `H x W x 3` BGR array.
2. Resize to `(width=64, height=32)` with `cv2.INTER_AREA`, including upscales, so interpolation never depends on a caller-selected branch.
3. Convert with `cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)`.
4. On this resized, pre-blur `uint8` gray image, compute `np.mean(gray, dtype=np.float64)` and population standard deviation `np.std(gray, dtype=np.float64, ddof=0)`.
5. Apply `cv2.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REFLECT_101)`.
6. Compute `p05, p95 = np.percentile(blurred, (5.0, 95.0), method="linear")`.
7. If `p95 <= p05`, emit an owned read-only all-zero `uint8` `(32, 64)` image and force `hud_visible=False`.
8. Otherwise convert the blurred image, `p05`, and `p95` to float32 before subtraction/division, calculate `np.clip((gray32 - p05_32) / (p95_32 - p05_32), 0.0, 1.0) * np.float32(255.0)`, apply `np.rint`, and cast to `uint8`.
9. HUD visibility requires nondegenerate percentiles, `std_luminance >= 6.0`, and inclusive `4.0 <= mean_luminance <= 251.0`.

`build_meter_references` requires a typed `Side`, exactly five full crops and five spent crops, applies `preprocess_meter` to every crop, and rejects the complete calibration if any sample is hidden. Take the per-pixel median across each five-image canonical stack and cast that internally produced median to `uint8`. The two canonical arrays must be `(32, 64)` and have normalized mean absolute distance at least `0.08`.

The persisted NPZ schema has exactly these keys and dtypes; load with `allow_pickle=False` and reject missing or extra keys before constructing the result:

```text
schema_version        scalar np.uint16 == 1
preprocessing_version scalar np.uint16 == 1
side                  scalar Unicode <U2, "P1" or "P2"
shape_hw              shape (2,), np.uint16, [32, 64]
full_gray             shape (32, 64), np.uint8
spent_gray            shape (32, 64), np.uint8
```

`load_references(..., expected_side=...)` requires a typed `Side` and rejects a stored-side mismatch. Public `PreprocessedMeter` and `MeterReferences` constructors accept only arrays already having the exact declared ndarray dtype and shape; they reject rather than cast, reshape, resize, or clip caller input. `PreprocessedMeter` also requires finite nonnegative real non-boolean mean/std values and an actual boolean visibility flag. `MeterReferences` requires a typed side and actual-integer non-boolean schema/preprocessing versions equal to the declared constants, plus the minimum full/spent distance. After validation they copy accepted arrays into separate C-contiguous owned buffers and mark only the arrays exposed by those two value objects read-only; full and spent must not share memory. They never mutate input crops, call `setflags` on caller arrays, or alter Task 2's borrowed, writable `CapturedFrame.image`. Load copies arrays before closing the NPZ.

- [ ] **Step 6: Implement bounded reads and atomic JSON/NPZ writes**

`load_profile` opens the resolved profile in binary mode and performs one bounded `MAX_PROFILE_BYTES + 1` read. Reject a longer input before UTF-8 decoding; otherwise decode strictly and call `json.loads(..., parse_constant=reject_nonfinite)` before the exact schema conversion. Never call unbounded `read()`, `read_text()`, or `json.load(file)` on profile input.

`load_references` and `save_references` must first call `resolve_contained_path(user_data_root, profile_path, field_name="profile_path")` with no suffix policy, then call `resolve_contained_path(resolved_profile.parent, reference_path, field_name="reference_path", required_suffix=".npz")`. Every read and atomic write uses only that exact resolved reference target; reference paths are never resolved relative to the broader user-data root.

`load_references` opens that exact resolved archive in binary mode and performs one bounded `MAX_REFERENCE_ARCHIVE_BYTES + 1` read, rejecting an archive over `1 MiB` before ZIP parsing. On an `io.BytesIO` over the accepted bytes, use `zipfile.ZipFile` for a complete preflight before NumPy sees the archive:

1. `infolist()` must contain exactly six entries with six distinct names: `schema_version.npy`, `preprocessing_version.npy`, `side.npy`, `shape_hw.npy`, `full_gray.npy`, and `spent_gray.npy`. Reject duplicates, missing/extra names, directories, or any encryption flag.
2. Every member must use `zipfile.ZIP_DEFLATED`, have `file_size <= MAX_NPY_MEMBER_BYTES`, and the sum of all declared uncompressed sizes must be `<= MAX_NPZ_UNCOMPRESSED_BYTES`.
3. Before `np.load`, open each member and read at most `MAX_NPY_MEMBER_BYTES + 1`; reject an over-limit or length mismatch and read through EOF so `zipfile` verifies decompression and CRC. Retain only these bounded member bytes for header checks.
4. Parse each bounded `.npy` member with `numpy.lib.format.read_magic` plus the matching `read_array_header_1_0` or `read_array_header_2_0`. Reject other header versions, malformed headers, object/structured dtypes, Fortran order, and any dtype/shape that differs from the six-member contract above. Record the exact header end offset and require `len(member_bytes) - header_end == math.prod(shape) * dtype.itemsize`; reject truncated or appended payload bytes. This rejects a tiny compressed member whose header declares an enormous array before allocation and enforces an exact envelope before allocation.
5. Only after every envelope, CRC, and header check passes, call `np.load(io.BytesIO(archive_bytes), allow_pickle=False)`, validate scalar values, copy the six values needed for construction, and close the loader.

Both save functions validate the complete object and secure target, then serialize completely in memory before touching the target: JSON uses `allow_nan=False`, `ensure_ascii=False`, `indent=2`, `sort_keys=True`, and a trailing newline and rejects output over `64 KiB`; NPZ uses `np.savez_compressed` with an `io.BytesIO`, the exact six non-object arrays above, and rejects any internally produced output that violates the same archive/member/total bounds. After serialization, create the parent, write a uniquely named same-directory owned `.tmp`, flush and `fsync`, close, and call `os.replace(temp, exact_target)`. On any failure remove only that call's temp and preserve the previous target and neighboring files. No raw crop, screenshot, or reference image may enter Git.

- [ ] **Step 7: Run focused and full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hud_profile.py tests/test_meter_references.py -v
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
```

Expected: focused and full suites PASS. Conditional link tests may skip only for a documented host privilege denial. Confirm no `.tmp`, profile JSON, NPZ, crop, or screenshot fixture escaped pytest's temporary directory.

- [ ] **Step 8: Commit**

```powershell
git add src/sf6_music_trigger/hud_profile.py src/sf6_music_trigger/meter_references.py tests/test_hud_profile.py tests/test_meter_references.py
git commit -m "feat: add calibrated HUD profiles"
```

---

### Task 4: Deterministic Gauge Feature Extraction

**Files:**
- Create: `src/sf6_music_trigger/gauge_features.py`
- Create: `tests/test_gauge_features.py`

**Interfaces:**
- Produces: `GaugeObservation`, `observe_gauge(frame_bgr, *, side, frame_id, roi, references, sampled_at)`.
- Private pure seam: `_score_distances(full_distance, spent_distance) -> tuple[Real, Real]`, tested directly only for the otherwise unreachable zero-total branch.
- Consumes Task 3 `NormalizedRoi`, `MeterReferences`, `crop_normalized`, `preprocess_meter`; Task 2 `Side`; NumPy. Do not define another crop, preprocessing, visibility, persistence, OOD, or classification path.

- [ ] **Step 1: Write representative RED tests**

```python
from dataclasses import FrozenInstanceError
import pytest
from sf6_music_trigger.domain import Side
from sf6_music_trigger.gauge_features import GaugeObservation, _score_distances, observe_gauge


def test_constructor_is_typed_frozen_and_preserves_huge_time():
    huge = 10**10000
    item = GaugeObservation(
        side=Side.P1, frame_id=7, sampled_at=huge, hud_visible=True,
        full_confidence=0.25, spent_confidence=0.75,
        full_distance=0.75, spent_distance=0.25,
    )
    assert item.sampled_at is huge
    with pytest.raises(FrozenInstanceError):
        item.frame_id = 8


def test_hidden_is_exact_fail_closed_sentinel(hidden_frame, p1_roi, p1_refs):
    item = observe_gauge(
        hidden_frame, side=Side.P1, frame_id=3, roi=p1_roi,
        references=p1_refs, sampled_at=1.2,
    )
    assert not item.hud_visible
    assert (
        item.full_confidence, item.spent_confidence,
        item.full_distance, item.spent_distance,
    ) == (0.0, 0.0, 1.0, 1.0)


def test_non_full_roi_ignores_outside_distractor(feature_fixture):
    item = observe_gauge(
        feature_fixture.frame_with_p1_full_and_outside_spent_distractor(),
        side=Side.P1, frame_id=4, roi=feature_fixture.p1_roi,
        references=feature_fixture.p1_refs, sampled_at=1.3,
    )
    assert item.hud_visible
    assert item.full_confidence >= 0.78
    assert item.spent_confidence < 0.72


def test_zero_total_scoring_is_ambiguous():
    assert _score_distances(0.0, 0.0) == (0.5, 0.5)
```

- [ ] **Step 2: Complete the mandatory RED matrix before production**

Add all cases while `gauge_features.py` is absent; capture Step 3 RED before production:

- Direct frozen constructor: typed P1/P2; reject untyped side; actual non-bool `int frame_id >= 0`; finite nonnegative `Real` non-bool sampled time, including an arbitrarily large Python int without conversion/`OverflowError`; actual bool visibility; four finite `Real` non-bool metrics in inclusive `[0,1]`, with huge metric giving clean `ValueError`; hidden objects require exact 0/0/1/1; visible confidence fields must agree with cross-distance scoring, including 0.5/0.5 at zero total; inconsistent combinations reject; preserve accepted scalar objects; frozen scalar-only output retains no ndarray.
- Routing: explicit P1/P2; wrong-side references reject before dependencies; non-full ROI plus outside distractor; BGR-unequal input requiring resize; spies prove Task 3 crop then preprocess each exactly once and identity pass-through; caller frame values, shape, strides, contiguity, writeability, and ownership unchanged after success, hidden, and errors.
- Scoring: literal canonical arrays lock `np.mean(np.abs(observed.astype(np.float32) - reference.astype(np.float32)), dtype=np.float64) / 255.0`; lock cross-distance confidences; directly lock zero-total `(0.5,0.5)`; helper uses finite non-bool `[0,1]` inputs.
- Behavior: exact full/spent for both sides; deliberately partial visible crop remains visible with `full_confidence < 0.78` and `spent_confidence < 0.72`; black, white, flat mid-gray, low contrast all return exact 0/0/1/1; invalid inputs do not mutate frames/references.

Confidence is relative between two references, not an absolute in-distribution probability. A visible OOD crop can receive high relative confidence. Do not invent an OOD/confidence gate or guess suppression; tracker thresholds are later policy.

- [ ] **Step 3: Run complete focused RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gauge_features.py -v
```

Expected: collection ERROR because `gauge_features` is absent; the full matrix already exists.

- [ ] **Step 4: Implement minimal validated observation**

```python
@dataclass(frozen=True)
class GaugeObservation:
    side: Side
    frame_id: int
    sampled_at: Real
    hud_visible: bool
    full_confidence: Real
    spent_confidence: Real
    full_distance: Real
    spent_distance: Real


def observe_gauge(
    frame_bgr: np.ndarray, *, side: Side, frame_id: int,
    roi: NormalizedRoi, references: MeterReferences, sampled_at: Real,
) -> GaugeObservation: ...
```

Constructor rules are exactly Step 2. For sampled time, reject bool/non-Real/negative; ints are finite without conversion, other reals get a finite check. For metrics, reject bool/non-Real then require `0 <= value <= 1` without conversion, so huge/nonfinite values fail cleanly. When hidden, require the exact 0/0/1/1 tuple. When visible, calculate expected confidences through `_score_distances` and require each supplied confidence to be equal or `math.isclose(..., rel_tol=0.0, abs_tol=1e-12)` to its expected cross-distance value. Store accepted scalars unchanged.

Validate side/frame/time/ROI/references and require `references.side is side` before dependencies. Call `crop_normalized` once and pass its result to `preprocess_meter` once. Hidden returns exact 0/0/1/1. Visible:

```python
full_distance = np.mean(
    np.abs(observed.astype(np.float32) - references.full_gray.astype(np.float32)),
    dtype=np.float64,
) / 255.0
spent_distance = np.mean(
    np.abs(observed.astype(np.float32) - references.spent_gray.astype(np.float32)),
    dtype=np.float64,
) / 255.0
```

Private helper validates both distances and implements:

```python
total_distance = full_distance + spent_distance
if total_distance == 0:
    return (0.5, 0.5)
return (spent_distance / total_distance, full_distance / total_distance)
```

No caller or Task 3 array is mutated.

- [ ] **Step 5: Run focused/full GREEN and hygiene**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gauge_features.py -v
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
```

Expected: PASS; only two Task 4 files changed; no escaped fixtures.

- [ ] **Step 6: Commit**

```powershell
git add src/sf6_music_trigger/gauge_features.py tests/test_gauge_features.py
git commit -m "feat: extract calibrated gauge observations"
```

---



### Task 5: Per-Side GaugeTracker State Machine

**Files:**
- Prerequisite modify: `src/sf6_music_trigger/domain.py`
- Prerequisite test: `tests/test_domain_config.py`
- Create: `src/sf6_music_trigger/gauge_tracker.py`
- Create: `tests/test_gauge_tracker.py`

The two prerequisite files form one compatibility commit before Task 5 RED begins. The Task 5 implementation commit contains only `gauge_tracker.py` and `test_gauge_tracker.py`; never combine the two commits.

In the current lineage, independent commit `cdf1bc0` satisfies this prerequisite. Do not amend or fold it into Task 5. Steps 0.1-0.3 preserve the required RED/GREEN lineage for replay from an earlier base; an implementer starting after `cdf1bc0` verifies that evidence and begins tracker Step 1.

**Interfaces:**
- Produces: frozen `GaugeTrackerConfig`; `GaugeTracker(side: Side, profile_id: str, config: GaugeTrackerConfig)`; read-only `GaugeTracker.state`; `GaugeTracker.update(observation: GaugeObservation) -> tuple[ConfirmedSuperEvent, ...]`; and `GaugeTracker.reset(reason: str) -> None`.
- Private pure seam: `_classify_confidences(full_confidence, spent_confidence, config)`, tested directly only for the otherwise unreachable valid-observation double-hit branch.
- Consumes: Task 4 `GaugeObservation`; Task 2 `Side`, `TrackerState`, and Rational-safe `ConfirmedSuperEvent`. No wrapper update type is produced.
- Preserves accepted `Real` timestamp objects end to end. Task 5 must never call `float`, `int`, `round`, NumPy conversion, or another narrowing operation on `sampled_at`, `confirmed_at`, or elapsed-time operands.

#### Authoritative plan and execution-base gate

The routing, transition, and timing text in this Task 5 section is the authoritative implementation contract where the approved design §5.3 diagram is less detailed; this clarification does not authorize editing the design specification or expanding Phase 1 scope.

This current plan document must pass independent review and be committed as its own plan-only commit after `cdf1bc0` before anyone creates `tests/test_gauge_tracker.py` or `src/sf6_music_trigger/gauge_tracker.py` or begins Task 5 RED. At Task 5 implementation start, run:

```powershell
git status --short
$task5_plan_base = git rev-parse HEAD
$task5_plan_base
git diff --name-only "cdf1bc0..$task5_plan_base"
```

Expected: clean status; record the full HEAD hash as `task5_plan_base`; the range contains only `docs/superpowers/plans/2026-08-08-foundation-gauge-trigger.md`. HEAD remains that plan base until the final Task 5 commit, so pre-commit scope checks compare the working tree with HEAD and do not include this plan diff.

#### Required prerequisite: make Task 2 event time Rational-safe in its own commit

- [ ] **Step 0.1: Add one focused Task 2 compatibility RED**

Append this regression to `tests/test_domain_config.py` before changing `domain.py`:

```python
from fractions import Fraction


def test_domain_times_accept_arbitrarily_large_finite_fractions_without_coercion():
    huge_time = Fraction(10**10000, 1)
    geometry = ClientGeometry(0, 0, 1, 1)
    event = ConfirmedSuperEvent(
        "evt-fraction", Side.P2, huge_time, 0, 0, "default"
    )
    frame = CapturedFrame(0, np.zeros(1), huge_time, geometry)
    assert event.confirmed_at is huge_time
    assert frame.monotonic_time is huge_time
```

Also add this companion:

```python
@pytest.mark.parametrize(
    "invalid_time",
    [Fraction(-1, 2), float("nan"), float("inf"), float("-inf")],
)
def test_domain_times_reject_negative_fraction_and_nonfinite_floats(
    invalid_time: object,
):
    geometry = ClientGeometry(0, 0, 1, 1)
    with pytest.raises(ValueError):
        ConfirmedSuperEvent(
            "evt-invalid", Side.P1, invalid_time, 0, 0, "default"
        )
    with pytest.raises(ValueError):
        CapturedFrame(0, np.zeros(1), invalid_time, geometry)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_domain_config.py::test_domain_times_accept_arbitrarily_large_finite_fractions_without_coercion -v
```

Expected: RED with the existing `math.isfinite` path raising `OverflowError`; the failure proves Task 2, not tracker code, owns the compatibility gap.

- [ ] **Step 0.2: Apply the minimal Task 2 compatibility fix**

In `domain.py`, import `Rational` and change the shared nonnegative finite-real helper only as follows:

```python
def _require_nonnegative_finite_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, Rational):
        if value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
```

Do not cast or otherwise change the accepted object. This prerequisite supersedes only Task 2's `int`-only fast path; preserve the existing public dataclass annotations and every other Task 2 validation rule.

- [ ] **Step 0.3: Verify and commit the prerequisite independently**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_domain_config.py::test_domain_times_accept_arbitrarily_large_finite_fractions_without_coercion -v
.\.venv\Scripts\python.exe -m pytest tests/test_domain_config.py -v
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
git status --short
git add src/sf6_music_trigger/domain.py tests/test_domain_config.py
git commit -m "fix: accept finite rational domain timestamps"
```

Expected: GREEN, with exactly those two paths in this compatibility commit. Do not create `gauge_tracker.py` or `test_gauge_tracker.py` until this commit exists.

- [ ] **Step 1: Write representative tracker RED tests with literal oracles**

```python
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from sf6_music_trigger.domain import Side, TrackerState
from sf6_music_trigger.gauge_features import GaugeObservation
from sf6_music_trigger.gauge_tracker import (
    GaugeTracker,
    GaugeTrackerConfig,
    _classify_confidences,
)


def obs(frame_id, t, *, side=Side.P1, full=None, spent=None, visible=True):
    if not visible:
        return GaugeObservation(
            side=side, frame_id=frame_id, sampled_at=t, hud_visible=False,
            full_confidence=0.0, spent_confidence=0.0,
            full_distance=1.0, spent_distance=1.0,
        )
    if full is None and spent is None:
        full = spent = 0.5
    elif full is None:
        full = 1.0 - spent
    elif spent is None:
        spent = 1.0 - full
    return GaugeObservation(
        side=side, frame_id=frame_id, sampled_at=t, hud_visible=True,
        full_confidence=full, spent_confidence=spent,
        full_distance=spent, spent_distance=full,
    )


def test_default_config_literals_are_frozen():
    config = GaugeTrackerConfig()
    assert (
        config.full_threshold,
        config.spent_threshold,
        config.full_samples,
        config.evidence_window,
        config.spent_samples,
        config.minimum_spend_seconds,
        config.ambiguous_disarm_samples,
        config.rearm_samples,
    ) == (0.78, 0.72, 5, 5, 3, 0.10, 3, 5)
    with pytest.raises(FrozenInstanceError):
        config.spent_samples = 2


def test_minimum_spend_floor_is_exactly_one_tenth():
    exact_floor = Fraction(1, 10)
    config = GaugeTrackerConfig(minimum_spend_seconds=exact_floor)
    assert config.minimum_spend_seconds is exact_floor
    with pytest.raises(ValueError):
        GaugeTrackerConfig(
            minimum_spend_seconds=exact_floor - Fraction(1, 10**6)
        )


def test_five_full_then_only_three_spent_confirms_on_third():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    for frame_id, t in enumerate((0.00, 0.05, 0.10, 0.15, 0.20)):
        assert tracker.update(obs(frame_id, t, full=0.78)) == ()
    assert tracker.update(obs(5, 0.30, spent=0.72)) == ()
    assert tracker.update(obs(6, 0.35, spent=0.72)) == ()
    events = tracker.update(obs(7, 0.41, spent=0.72))
    assert isinstance(events, tuple) and len(events) == 1
    event = events[0]
    assert (
        event.side,
        event.confirmed_at,
        event.pre_spend_frame_id,
        event.confirmed_frame_id,
        event.profile_id,
    ) == (Side.P1, 0.41, 4, 7, "profile")
    assert tracker.state is TrackerState.COOLDOWN
    assert tracker.update(obs(8, 0.45, spent=0.72)) == ()


def test_huge_fraction_confirmed_at_preserves_identity():
    huge = Fraction(10**10000, 3)
    tracker = GaugeTracker(Side.P2, "fraction-profile", GaugeTrackerConfig())
    full_times = (
        huge - 1,
        huge - Fraction(4, 5),
        huge - Fraction(3, 5),
        huge - Fraction(2, 5),
        huge - Fraction(3, 10),
    )
    for frame_id, t in enumerate(full_times):
        assert tracker.update(obs(frame_id, t, side=Side.P2, full=0.78)) == ()
    assert tracker.update(
        obs(5, huge - Fraction(1, 5), side=Side.P2, spent=0.72)
    ) == ()
    assert tracker.update(
        obs(6, huge - Fraction(1, 10), side=Side.P2, spent=0.72)
    ) == ()
    event = tracker.update(obs(7, huge, side=Side.P2, spent=0.72))[0]
    assert event.confirmed_at is huge


def test_double_hit_classifies_fail_closed_ambiguous():
    assert _classify_confidences(0.90, 0.90, GaugeTrackerConfig()) == "ambiguous"


def test_state_assignment_raises_initially_and_in_candidate():
    tracker = GaugeTracker(Side.P1, "profile", GaugeTrackerConfig())
    assert tracker.state is TrackerState.DISARMED
    with pytest.raises(AttributeError):
        tracker.state = TrackerState.ARMED
    assert tracker.state is TrackerState.DISARMED
    for frame_id, t in enumerate((0.00, 0.05, 0.10, 0.15, 0.20)):
        tracker.update(obs(frame_id, t, full=0.78))
    tracker.update(obs(5, 0.30, spent=0.72))
    assert tracker.state is TrackerState.SPEND_CANDIDATE
    with pytest.raises(AttributeError):
        tracker.state = TrackerState.DISARMED
    assert tracker.state is TrackerState.SPEND_CANDIDATE
```

- [ ] **Step 2: Complete the entire mandatory tracker RED matrix before production**

Add every case below while `src/sf6_music_trigger/gauge_tracker.py` is absent. The initial RED must contain the whole matrix, not only the representative snippet.

- Config constructor: default literals remain exactly `(0.78, 0.72, 5, 5, 3, 0.10, 3, 5)` and frozen assignment; both thresholds accept finite `Real` non-booleans only and require `0.5 < value <= 1`; `full_samples` is an actual non-bool `int >= 5`; `evidence_window` is an actual non-bool `int` exactly `5`; `spent_samples` is an actual non-bool `int` in `[3, 5]`; `minimum_spend_seconds` is a finite `Real` non-boolean no smaller than the mathematical value `Fraction(1, 10)`; `ambiguous_disarm_samples` is an actual non-bool `int >= 1`; `rearm_samples` is an actual non-bool `int >= 5`. Retain the field default literal `0.10`, but validate the floor by direct comparison with `Fraction(1, 10)`, so `Fraction(1, 10)` is accepted and `Fraction(1, 10) - Fraction(1, 10**6)` is rejected. Cover each exact boundary, just-outside value, bool, float-for-int, string, NaN, and infinity as applicable. Constructors reject rather than coerce and retain accepted Real objects. Validate Rational values by exact comparison without `float`/`math.isfinite`; apply `math.isfinite` only to non-Rational Real values.
- Tracker constructor/reset: accept P1 and P2 with an actual nonblank `profile_id` string and a `GaugeTrackerConfig`; reject wrong side, blank/non-string profile, and mapping/`None`/wrong config. Initial state is DISARMED, and the `state` property rejects assignment both initially and after a nontrivial transition without changing state or other internals. `reset(reason)` accepts only an actual nonblank string, returns `None`, clears detection history and both ordering watermarks, and starts a new epoch in which lower frame/time values are legal. Invalid reset reasons reject before mutation.
- Update order and error atomicity: reject a non-`GaugeObservation`, a side mismatch, `frame_id <= last_frame_id`, or `sampled_at < last_sampled_at` before changing state, evidence, counters, pre-spend frame, or either watermark. Exercise each rejection from a nontrivial candidate/cooldown state and compare a deep copy of `vars(tracker)` before/after only in tests; do not add a public debug API. Equal timestamps with strictly increasing frame IDs are accepted, but equal-time spent samples cannot satisfy a positive minimum duration. Every ordering rule applies in every state and after hidden observations.
- Hidden/reset watermarks: an accepted hidden observation first advances both watermarks, then returns `()`, moves to DISARMED, and clears all detection history; subsequent duplicate/older frame or earlier time still rejects. A valid public reset clears those watermarks so a new epoch may start again at frame/time zero.
- Classification/routing: P1 and P2 stay isolated. Full means only `full_confidence >= full_threshold`; spent means only `spent_confidence >= spent_threshold`; neither hit is ambiguous; a defensive double hit is also ambiguous and never full or spent. Directly test `_classify_confidences` for the otherwise unreachable valid-observation double-hit branch. Spent without prior stable full never confirms; partial/ambiguous SA1/SA2-like values never confirm. Confirmation evaluation is reachable only from the SPEND_CANDIDATE/spent row; a candidate ambiguous observation must never invoke it.
- Global evidence/state behavior: every accepted visible observation is appended before state-specific logic to one global fixed-five deque, including full/ambiguous/spent observations in every state. Only consecutive full samples arm DISARMED. ARMED full refreshes the pre-spend frame; its first non-full observation enters SPEND_CANDIDATE and is processed there without clearing evidence. Candidate full returns to ARMED and refreshes pre-spend; candidate spent resets the consecutive-ambiguous counter and is the only classification that evaluates confirmation; candidate ambiguous increments that counter without evaluating confirmation, and reaching the configured limit moves to DISARMED and clears detection history. Ambiguous values occupy evidence slots and old evidence is evicted. Confirmation passes through CONFIRMED while constructing exactly one event and is already COOLDOWN when `update` returns. In COOLDOWN, any non-full observation resets the rearm streak; only consecutive full observations reaching `rearm_samples` clear that streak, refresh the pre-spend frame, and return to ARMED. This COOLDOWN-to-ARMED transition is the one canonical transition reset; confirmation/entry to COOLDOWN does not perform a second reset. Tests cover separate ARMED-full and candidate-to-full pre-spend refreshes, plus a black-box third event cycle.
- Event contract: patch `uuid.uuid4` to make `.hex` deterministic and assert exact `event_id`, tracker side, the original confirming observation's `sampled_at` object by identity, last ARMED full `pre_spend_frame_id`, current `confirmed_frame_id`, and exact tracker `profile_id`. The event is returned only as `(event,)`; all other paths return `()`. Additional spent/ambiguous observations in COOLDOWN cannot duplicate it. Spy on `ConfirmedSuperEvent` construction to prove the state is transiently CONFIRMED and final state is COOLDOWN.

Use fixed literal sequences and expected values in behavioral tests. Do not derive every oracle, loop length, threshold, quorum, or duration from the `GaugeTrackerConfig` instance under test. Mandatory named mutation controls must kill all of these single changes:

- `>=` changed to `>` at the literal `0.78` full boundary, literal `0.72` spent boundary, and an exact `Fraction(1, 10)` duration boundary under an explicit `minimum_spend_seconds=Fraction(1, 10)` config;
- changing only `deque(maxlen=5)` to `deque(maxlen=6)`, killed by the normal positive happy path because the exact-five readiness guard otherwise stays false once the sixth item is retained;
- a six-slot/unbounded implementation that also uses `len >= 5` or otherwise retains old-sample semantics, killed by the fixed six-input old-evidence sequence below;
- a two-spent quorum, using a fixed five-observation window containing exactly two spent samples and a current spent sample;
- filtering ambiguous observations instead of giving them slots;
- routing a candidate ambiguous observation through confirmation evaluation, using the fixed route-invariant sequence below;
- anchoring duration to an older spent observation outside the current five-slot window, using the fixed duration-anchor sequence below;
- omitting either ARMED-full or candidate-to-full pre-spend-frame refresh;
- removing either frame/time ordering guard, including duplicate frame, backward frame, backward time, and equal-time acceptance cases;
- accumulating nonconsecutive COOLDOWN full observations, using an alternating full/non-full literal sequence followed by spent inputs that must not produce a second event;
- hard-coding any configurable default instead of honoring each independent nondefault literal case below.

Use these literal post-arm classification/time sequences for the evidence guards; `S` is spent at `0.72` and `A` is visible ambiguous at `0.50/0.50`:

| Named RED | Literal post-arm inputs | Required result |
|---|---|---|
| two-spent quorum | `A@0.30, A@0.35, S@0.40, A@0.45, S@0.51` | No event: the current fixed-five window has exactly two spent entries. |
| ambiguous occupancy and old-evidence eviction | `S@0.30, A@0.35, A@0.40, S@0.45, A@0.50, S@0.55` | No event: `maxlen=5` evicts the first `S`, leaving exactly two spent entries. This negative control kills six-slot/unbounded implementations only when they also use `len >= 5` or otherwise retain old-sample semantics, and it kills filtering out `A`; the separate normal happy path kills the bare `maxlen=6` false negative. |
| candidate-ambiguous route invariant | `S@0.30, S@0.34, S@0.38, A@0.45` | No event: the spent samples themselves span only `0.08` and fail duration; for the later `A`, elapsed time from the earliest `S` is `0.15`, but an ambiguous current observation must never evaluate confirmation. |
| earliest spent in current-window duration anchor | `S@0.30, A@0.35, A@0.40, S@0.45, A@0.46, S@0.47, S@0.48` | No event: the final fixed-five window has three `S`, but its earliest spent is `S@0.45`, so the span is only `0.03`; `S@0.30` is outside the window and cannot anchor duration. |

For the exact duration boundary, use two independent trackers with `minimum_spend_seconds=Fraction(1, 10)`: three spent timestamps whose first/last are exactly `Fraction(1, 10)` apart must confirm, while a final timestamp that makes the span `Fraction(1, 10) - Fraction(1, 10**6)` must not.

Add two separate pre-spend refresh REDs. In the ARMED-full test, arm on frames `0..4`, accept an extra full on frame `5@0.25`, then feed spent frames `6@0.30, 7@0.36, 8@0.42`; confirm only on frame `8` and assert the event's `pre_spend_frame_id` is exactly `5`. In the candidate-bounce test, arm on frames `0..4`, enter candidate on ambiguous frame `5@0.25`, bounce to full on frame `6@0.30`, then feed spent frames `7@0.35, 8@0.41, 9@0.47`; confirm only on frame `9` and assert the event's `pre_spend_frame_id` is exactly `6`. The final event in each test must name the latest full frame, not the original arming frame; both non-boundary positive traces use a first-to-confirming spent span of `0.12`.

Add independent nondefault behavior tests with literal inputs and literal oracles; do not combine them or derive counts/thresholds from the config under test:

| Nondefault field | Mandatory literal behavior |
|---|---|
| `full_threshold=0.90` | Five consecutive `full=0.89` observations do not arm; five consecutive `full=0.90` observations do arm. |
| `spent_threshold=0.90` | After arming, feed `spent=0.90@0.30`, `spent=0.89@0.35`, and `spent=0.89@0.40` with no event; feed `full@0.45`, then `spent=0.90@0.50, 0.56, 0.62`, confirming only on the third spent at `0.62` with span `0.12`. |
| `full_samples=6` | The fifth consecutive full remains DISARMED; the sixth enters ARMED. |
| `spent_samples=4` | After arming, feed spent observations at `0.30, 0.35, 0.41, 0.46`: the first three produce no event and the fourth produces exactly one event. A hard-coded quorum of three would incorrectly confirm at `0.41`, whose span is already `0.11`. |
| `minimum_spend_seconds=0.20` | Spent observations at `0.30, 0.35, 0.40` produce no event; a further spent observation at `0.51` produces exactly one event. |
| `ambiguous_disarm_samples=1` | One ambiguous observation after arming immediately enters DISARMED without an event. |
| `rearm_samples=6` | After event one, the first five consecutive full observations remain COOLDOWN; the sixth enters ARMED, and spent observations at relative offsets `0.00, 0.06, 0.12` produce event two only on the third. |

For the default rearm guard, first confirm once, then feed literal alternating `full, ambiguous` pairs containing five total but never consecutive full observations, followed by spent observations at relative offsets `0.00, 0.06, 0.12`; the event count remains one. Next, five consecutive full observations return to ARMED, and spent observations at relative offsets `0.00, 0.06, 0.12` produce event two only on the third. After event two, assert only public behavior: the first four consecutive full observations leave `state is TrackerState.COOLDOWN`, the fifth leaves `state is TrackerState.ARMED`, and spent observations at fresh relative offsets `0.00, 0.06, 0.12` produce event three only on the third. Do not inspect a private rearm counter or claim an independent mutation control for the transition reset.

- [ ] **Step 3: Run the complete focused tracker suite and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gauge_tracker.py -v
```

Expected: collection ERROR because `gauge_tracker` does not exist. Confirm the representative tests and every Step 2 matrix/mutation-control test are already present before capturing RED.

- [ ] **Step 4: Implement the minimal ordered state machine**

```python
@dataclass(frozen=True)
class GaugeTrackerConfig:
    full_threshold: Real = 0.78
    spent_threshold: Real = 0.72
    full_samples: int = 5
    evidence_window: int = 5
    spent_samples: int = 3
    minimum_spend_seconds: Real = 0.10
    ambiguous_disarm_samples: int = 3
    rearm_samples: int = 5


def _classify_confidences(
    full_confidence: Real,
    spent_confidence: Real,
    config: GaugeTrackerConfig,
) -> Literal["full", "spent", "ambiguous"]: ...


class GaugeTracker:
    def __init__(
        self,
        side: Side,
        profile_id: str,
        config: GaugeTrackerConfig,
    ) -> None: ...

    @property
    def state(self) -> TrackerState: ...

    def update(
        self, observation: GaugeObservation
    ) -> tuple[ConfirmedSuperEvent, ...]: ...

    def reset(self, reason: str) -> None: ...
```

Validate constructors exactly as Step 2; no clamping or coercion. In particular, keep the declared `minimum_spend_seconds` default as `0.10` while comparing every supplied value against the exact floor `Fraction(1, 10)`. Keep separate epoch watermarks `last_frame_id` and `last_sampled_at`. For `update`, perform this order exactly:

1. Require a `GaugeObservation` with exactly the tracker's side.
2. Before any mutation, reject `frame_id <= last_frame_id` and `sampled_at < last_sampled_at`. Equal sampled time is allowed.
3. Store the accepted frame/time objects as the new watermarks without conversion.
4. If hidden, clear detection history to DISARMED but retain the new watermarks and return `()`.
5. Classify the visible observation, append it and its classification to `deque(maxlen=5)`, then apply the state transition.

Classification is fail-closed:

| Raw threshold hits | Classification |
|---|---|
| full only | `"full"` |
| spent only | `"spent"` |
| neither | `"ambiguous"` |
| both | `"ambiguous"` |

Use this exact transition table after the global append:

| Current state | Classification | Action and next state |
|---|---|---|
| DISARMED | full | Increment consecutive-full streak; at `full_samples`, set `pre_spend_frame_id` to current frame and enter ARMED. |
| DISARMED | spent/ambiguous | Reset consecutive-full streak; remain DISARMED. |
| ARMED | full | Refresh `pre_spend_frame_id`; remain ARMED. |
| ARMED | spent/ambiguous | Enter SPEND_CANDIDATE and process this same observation under the matching candidate row. Do not clear global evidence. |
| SPEND_CANDIDATE | full | Clear candidate ambiguous streak, refresh `pre_spend_frame_id`, return to ARMED. |
| SPEND_CANDIDATE | spent | Reset candidate ambiguous streak; this is the only row that evaluates confirmation. Otherwise remain SPEND_CANDIDATE. |
| SPEND_CANDIDATE | ambiguous | Never evaluate confirmation. Increment consecutive-ambiguous streak; at `ambiguous_disarm_samples`, clear detection history and enter DISARMED, otherwise remain candidate. |
| COOLDOWN | full | Increment consecutive rearm streak; at `rearm_samples`, reset the rearm streak to zero, set `pre_spend_frame_id` to current frame, and enter ARMED. |
| COOLDOWN | spent/ambiguous | Reset rearm streak; remain COOLDOWN and never emit. |

Candidate confirmation is called only by the SPEND_CANDIDATE/spent row and requires all three conditions, with inclusive comparisons: the global deque has exactly five entries; at least `spent_samples` entries are classified spent; and `current.sampled_at - earliest_spent_in_current_window.sampled_at >= minimum_spend_seconds`. Perform subtraction/comparison on original Real objects only. On success, set CONFIRMED, construct one `ConfirmedSuperEvent(event_id=uuid.uuid4().hex, side=tracker_side, confirmed_at=current.sampled_at, pre_spend_frame_id=remembered_full_frame, confirmed_frame_id=current.frame_id, profile_id=tracker_profile_id)`, then set COOLDOWN before returning `(event,)`.

`Detection history` means the evidence deque, full/ambiguous/rearm streaks, and remembered pre-spend frame. Hidden input and candidate ambiguous disarm clear that history but preserve ordering watermarks. A valid `reset(reason)` clears both detection history and watermarks, creating a new ordering epoch. Invalid input/reset and all ordering violations leave every field unchanged.

- [ ] **Step 5: Run focused/full GREEN, mutation controls, and scope hygiene**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gauge_tracker.py -v
.\.venv\Scripts\python.exe -m pytest -v
$task5_plan_base = git rev-parse HEAD
$plan_base_paths = @(git diff --name-only "cdf1bc0..$task5_plan_base")
if (
    $plan_base_paths.Count -ne 1 -or
    $plan_base_paths[0] -ne "docs/superpowers/plans/2026-08-08-foundation-gauge-trigger.md"
) {
    throw "HEAD is not the reviewed Task 5 plan-only base after cdf1bc0"
}
git add -N src/sf6_music_trigger/gauge_tracker.py tests/test_gauge_tracker.py
git diff --check
git diff --name-only $task5_plan_base
git status --short
```

Expected: PASS. Recomputing `$task5_plan_base` in this Step 5 shell and validating the one-path `cdf1bc0..$task5_plan_base` range re-proves that the current HEAD is the reviewed plan-only base after `cdf1bc0` without relying on state from an earlier shell process. Then `git diff --name-only $task5_plan_base` lists only `src/sf6_music_trigger/gauge_tracker.py` and `tests/test_gauge_tracker.py`; `git add -N` makes both new paths visible to `git diff --check` and the name-only scope check without staging file contents. The already-committed plan document must not appear in this working-tree diff. One at a time, temporarily make every mutation listed in Step 2, run its named focused control, observe failure, and restore without persisting. In particular, the normal positive happy path must kill a bare `maxlen=6` false negative, while the old-evidence negative must separately kill six-slot/unbounded `len >= 5` or old-retention semantics and ambiguous filtering. Also prove the three inclusive boundaries, three-spent quorum, candidate-ambiguous route exclusion, current-window duration anchor, both pre-spend refresh routes, both ordering guards, consecutive rearm, black-box third-cycle behavior, and every nondefault literal are test-sensitive. Except for the exact Fraction boundary test, every Task 5 positive confirmation trace uses a first-to-confirming spent span of at least `0.11`. Task 8's existing integration trace must remain byte-identical here, including `spent_frames("P1", count=3, step=0.05)`; before Task 8 is executed, re-review that trace to use a safe starting time and first-to-confirming spent span of at least `0.11`, but do not edit Task 8 in this Task 5 plan amendment.

- [ ] **Step 6: Commit Task 5 independently**

```powershell
git add src/sf6_music_trigger/gauge_tracker.py tests/test_gauge_tracker.py
git commit -m "feat: confirm visual meter spend per side"
```

---



### Task 6: Typed Capture Samples, Freshness Gate, and Read-Only SF6 Window Capture

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/sf6_music_trigger/domain.py`
- Create: `src/sf6_music_trigger/window.py`
- Create: `src/sf6_music_trigger/capture.py`
- Create: `tests/test_capture_window.py`

**Interfaces:**
- Produces from `domain.py`: `CaptureIssue`, frozen `CaptureSample`.
- Produces from `window.py`: `WindowIdentity`, `MonitorDescriptor`, `Sf6WindowTarget`, `Win32Facade`, `find_sf6_window_target`, compatibility wrapper `find_sf6_client_geometry`, and typed window failures.
- Produces from `capture.py`: `DxcamOutput`, `DxcamCamera`, `DxcamFactory`, `FrameSource`, `FreshFrameGate`, `DxcamFrameSource`, `CaptureStartError`, `load_dxcam_factory`.
- Consumes: Task 2 `CapturedFrame`, `CaptureStatus`, `ClientGeometry`; Python standard library; NumPy; DXcam only after explicit `DxcamFrameSource.start()`.
- Does not consume or inspect controls, input history, game memory, network traffic, outcome classification, audio state, or GUI state.

#### Public domain contract

Add these exact values to `domain.py`:

```python
class CaptureIssue(str, Enum):
    BLACK_FRAME = "BLACK_FRAME"
    NO_NEW_FRAME = "NO_NEW_FRAME"
    WINDOW_CHANGED = "WINDOW_CHANGED"
    GEOMETRY_CHANGED = "GEOMETRY_CHANGED"
    BACKEND_ERROR = "BACKEND_ERROR"
    UNSUPPORTED_OUTPUT = "UNSUPPORTED_OUTPUT"


@dataclass(frozen=True)
class CaptureSample:
    status: CaptureStatus
    frame: CapturedFrame | None = None
    issue: CaptureIssue | None = None
```

`CaptureSample` rejects rather than coerces. Its valid combinations are exactly:

| status | frame | issue |
|---|---|---|
| `RUNNING` | typed `CapturedFrame` | `None` |
| `LOST` | `None` | typed `CaptureIssue` |
| `OCCLUDED` | `None` | `None` |
| `MINIMIZED` | `None` | `None` |
| `STOPPED` | `None` | `None` |

Therefore both invariants always hold:

```python
(sample.status is CaptureStatus.RUNNING) == (sample.frame is not None)
(sample.status is CaptureStatus.LOST) == (sample.issue is not None)
```

For a `RUNNING` sample, `frame.image` must be an actual NumPy array with exact `uint8` dtype, exact `H x W x 3` BGR shape `(frame.geometry.height, frame.geometry.width, 3)`, C-contiguous layout, caller ownership (`OWNDATA`), and writable storage. `CaptureSample` does not copy the array or alter its flags. Change the existing `CapturedFrame.monotonic_time` annotation from `float` to `Real`; its validation and storage behavior remain unchanged, and the frozen identity envelope retains the original backend timestamp object without float coercion.

#### Public window contract

Add these exact public values and protocol to `window.py`:

```python
@dataclass(frozen=True)
class WindowIdentity:
    hwnd: int
    pid: int
    process_created_at_ticks: int


@dataclass(frozen=True)
class MonitorDescriptor:
    handle: int
    device_name: str
    desktop_geometry: ClientGeometry


@dataclass(frozen=True)
class Sf6WindowTarget:
    identity: WindowIdentity
    geometry: ClientGeometry
    monitor: MonitorDescriptor


class Win32Facade(Protocol):
    def per_monitor_v2(self) -> ContextManager[None]: ...
    def enum_top_level_windows(self) -> tuple[int, ...]: ...
    def is_window_visible(self, hwnd: int) -> bool: ...
    def is_window_minimized(self, hwnd: int) -> bool: ...
    def is_window_cloaked(self, hwnd: int) -> bool: ...
    def read_process_identity(
        self, hwnd: int
    ) -> tuple[WindowIdentity, str]: ...
    def client_geometry(self, hwnd: int) -> ClientGeometry: ...
    def monitor_descriptor(
        self, hwnd: int, geometry: ClientGeometry
    ) -> MonitorDescriptor: ...


def find_sf6_window_target(
    *,
    win32: Win32Facade | None = None,
) -> Sf6WindowTarget: ...


def find_sf6_client_geometry(
    *,
    win32: Win32Facade | None = None,
) -> ClientGeometry:
    return find_sf6_window_target(win32=win32).geometry
```

Expose these typed failures:

```python
class GameWindowError(RuntimeError): ...
class GameWindowNotFound(GameWindowError): ...
class GameWindowAmbiguous(GameWindowError): ...
class GameWindowIdentityChanged(GameWindowError):
    before: Sf6WindowTarget
    after: Sf6WindowTarget | None

class GameWindowGeometryChanged(GameWindowError):
    before: Sf6WindowTarget
    after: Sf6WindowTarget

class GameWindowMinimized(GameWindowError): ...
class GameWindowOccluded(GameWindowError): ...
```

Validation is exact: `hwnd`, `pid`, and monitor handle are actual non-boolean positive integers; `process_created_at_ticks` is an actual non-boolean integer `>= 0`; monitor device name is a nonblank string; every nested value has its declared type; and the complete half-open client rectangle is contained in the monitor rectangle. Change exceptions expose read-only attributes and reject rather than coerce: `before` is always a typed `Sf6WindowTarget`; identity-change `after` is a typed target when a complete diagnostic second target can safely be built, otherwise `None` for disappearance or incomplete changed-process metadata; geometry-change `after` is always a typed target with the same `WindowIdentity` and a changed geometry or monitor value.

`find_sf6_window_target` performs this order:

1. Enter per-monitor-V2 thread DPI awareness.
2. Enumerate top-level windows without using window titles.
3. Identify candidates only by the case-insensitive executable basename `StreetFighter6.exe`, without first filtering hidden or minimized windows.
4. Fail closed for zero or more than one SF6 executable candidate.
5. For that unique HWND, query process identity first; then test minimized; then test `not is_window_visible(hwnd)` or `is_window_cloaked(hwnd)`; only then query client geometry and monitor. Minimized raises `GameWindowMinimized`; hidden or DWM-cloaked raises `GameWindowOccluded`.
6. Build the first typed `Sf6WindowTarget`, then immediately query the same HWND's identity again before any second minimized/visibility/cloaked check.
7. If the second identity query shows disappearance or a different identity, identity change has absolute precedence over minimized/occluded. Raise `GameWindowIdentityChanged(before=first, after=diagnostic_second_or_none)` immediately; a best-effort complete second target may be carried for diagnostics, but its window status cannot mask the identity route.
8. Only when the second identity equals the first, check minimized, then hidden/cloaked, then query geometry/monitor and build the second target.
9. If the equal-identity second target changes client geometry, monitor handle, monitor name, or monitor desktop geometry, raise `GameWindowGeometryChanged(before=first, after=second)`.
10. Return only the second stable target.
11. Restore the previous thread DPI context in `finally`, including every success, typed-status, and failure path.

`OCCLUDED` has this deliberately narrow meaning: `IsWindowVisible` is false or `DwmGetWindowAttribute(DWMWA_CLOAKED)` reports cloaked. The adapter does not claim to detect an otherwise visible window merely covered by another application's window.

The stable identity is `(hwnd, pid, process creation FILETIME ticks)`, not merely HWND or PID. Combine the two unsigned FILETIME words exactly as `(high_dw << 32) | low_dw`; never keep only one word or pass through float. HWND/PID reuse therefore cannot silently attach to another process. The ctypes implementation may use only these process-related calls: `GetWindowThreadProcessId`, `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`, `QueryFullProcessImageNameW`, `GetProcessTimes`, and `CloseHandle`. `PROCESS_QUERY_LIMITED_INFORMATION` is exactly `0x1000`. Do not request `PROCESS_VM_READ`, call `ReadProcessMemory`, retain the full executable path, or log it. Close the process handle in `finally`, including path-query and process-time failures.

`per_monitor_v2()` uses the thread-scoped `SetThreadDpiAwarenessContext` API with pointer-sized signed context value `PER_MONITOR_AWARE_V2 = -4`; in ctypes it passes/returns an opaque pointer value rather than truncating it to a 32-bit integer. A zero return is a failure. It records the exact previous context returned by the first call and, in `finally`, calls the same thread API with that exact pointer to restore it on every path. Do not use `SetProcessDpiAwarenessContext`, a process-global DPI mode, or the `-3` per-monitor-v1 value.

`is_window_cloaked` calls `DwmGetWindowAttribute` with exact attribute `DWMWA_CLOAKED = 14`, an owned `DWORD` output pointer, and its exact byte size. A nonzero HRESULT fails closed through `GameWindowError`; only an actual nonzero returned DWORD means cloaked. It does not inspect pixels or window overlap.

Client geometry uses `GetClientRect`, then calls `ClientToScreen` separately for both client corners; it does not infer the bottom-right desktop coordinate from only the top-left. Negative desktop origins are valid. A window crossing monitor/output boundaries is unsupported and fails closed. No desktop coordinate is passed directly to DXcam.

#### Public capture contract

Add these exact public values and protocols to `capture.py`:

```python
@dataclass(frozen=True)
class DxcamOutput:
    monitor_handle: int
    device_name: str
    device_index: int
    output_index: int
    desktop_geometry: ClientGeometry
    rotation_angle: int


class DxcamCamera(Protocol):
    @property
    def is_capturing(self) -> bool: ...

    @property
    def is_released(self) -> bool: ...

    def start(
        self,
        *,
        region: tuple[int, int, int, int],
        target_fps: int,
        video_mode: bool,
    ) -> None: ...

    def get_latest_frame(
        self,
        *,
        copy: bool,
        with_timestamp: Literal[True],
    ) -> tuple[np.ndarray, Real] | None: ...

    def stop(self) -> None: ...
    def release(self) -> None: ...


class DxcamFactory(Protocol):
    def outputs(self) -> tuple[DxcamOutput, ...]: ...

    def create(
        self,
        *,
        device_idx: int,
        output_idx: int,
        region: tuple[int, int, int, int],
        output_color: str,
        max_buffer_len: int,
        backend: str,
        processor_backend: str,
    ) -> DxcamCamera: ...


class FrameSource(Protocol):
    def start(self) -> None: ...
    def read(self) -> CaptureSample: ...
    def stop(self) -> None: ...


class FreshFrameGate:
    def __init__(
        self,
        max_age_seconds: Real = 0.25,
        max_future_seconds: Real = 0.05,
    ) -> None: ...

    def accept(
        self,
        frame: CapturedFrame,
        *,
        now: Real,
    ) -> bool: ...

    def reset(self) -> None: ...


def load_dxcam_factory() -> DxcamFactory: ...


class DxcamFrameSource:
    def __init__(
        self,
        *,
        window_finder: Callable[[], Sf6WindowTarget] = find_sf6_window_target,
        dxcam_factory_loader: Callable[[], DxcamFactory] = load_dxcam_factory,
        target_fps: int = 20,
    ) -> None: ...

    def start(self) -> None: ...
    def read(self) -> CaptureSample: ...
    def stop(self) -> None: ...


class CaptureStartError(RuntimeError):
    status: CaptureStatus
    issue: CaptureIssue | None
```

`DxcamOutput` is frozen and rejects rather than coerces: monitor handle is an actual non-boolean positive integer; device/output indices are actual non-boolean integers `>= 0`; device name is nonblank; desktop geometry is a typed `ClientGeometry`; and `rotation_angle` is an actual non-boolean integer in `{0, 90, 180, 270}`. Output equality includes rotation, so a rotation-only change cannot validate a payload captured across two output states. `target_fps` is an actual non-boolean integer in `[10, 30]`. `DxcamCamera.is_capturing` and `is_released` must each return an actual boolean.

`CaptureStartError` exposes read-only `status` and `issue` values and validates the same legal non-running pairs as `CaptureSample`: `MINIMIZED/OCCLUDED` carry `issue=None`, while `LOST` carries a typed issue. It never carries `RUNNING` or `STOPPED`. Start-stage mapping and priority are frozen:

| start failure stage | status | issue |
|---|---|---|
| unique SF6 window is minimized | `MINIMIZED` | `None` |
| unique SF6 window is hidden or DWM-cloaked | `OCCLUDED` | `None` |
| window absent/ambiguous, second-query disappearance, or identity change | `LOST` | `WINDOW_CHANGED` |
| same-identity geometry/monitor change during finder | `LOST` | `GEOMETRY_CHANGED` |
| other Win32/DPI/DWM façade failure | `LOST` | `BACKEND_ERROR` |
| structurally valid outputs have no unique containing match | `LOST` | `UNSUPPORTED_OUTPUT` |
| DXcam import/version/API shape/enumeration/create/start failure after both backend attempts | `LOST` | `BACKEND_ERROR` |

Window state/identity is resolved before loading DXcam; output mapping is resolved before backend creation. A higher row reached at its stage is not overwritten by a later generic backend error.

#### FreshFrameGate semantics

`max_age_seconds`, `max_future_seconds`, and `now` accept finite nonnegative `Real` values, reject booleans and non-`Real` values, compare `Rational` values exactly without float conversion, and use `math.isfinite` only for non-`Rational` `Real` values. Accepted constructor values are retained without coercion. `accept` requires an actual `CapturedFrame` before comparing its existing validated frame ID and timestamp.

- Frame IDs and timestamps must each be strictly greater than their last accepted values.
- Age exactly equal to `max_age_seconds` is accepted; any positive excess rejects.
- Future skew exactly equal to `max_future_seconds` is accepted; any positive excess rejects.
- Equal/lower ID and equal/lower timestamp reject independently; pixel equality is irrelevant.
- All validation and predicates complete before either watermark is assigned.
- A rejected frame or raised validation/comparison error advances neither watermark.
- An accepted frame commits both watermarks together.
- `reset()` clears both watermarks.

This literal atomicity trace is mandatory:

```text
accept id=1, t=10.00, now=10.00
reject id=2, t=9.00, now=10.00 as stale
accept id=2, t=10.10, now=10.10
```

The third frame remains acceptable; this kills an implementation that commits the ID watermark before completing timestamp validation.

#### DXcam adapter semantics

Change the runtime dependency to exactly:

```toml
"dxcam[winrt]>=0.3,<0.4",
```

Neither importing `sf6_music_trigger.capture` nor constructing `DxcamFrameSource` may import `dxcam`, enumerate adapters, initialize DXGI/WinRT, or touch a camera. The default factory loader imports DXcam only inside `DxcamFrameSource.start()` and returns a dedicated private `_Dxcam03FactoryAdapter`; no DXcam object escapes that adapter except the camera protocol returned by `create`. DXcam 0.3 itself may attempt process-wide `SetProcessDpiAwareness(2)` during that delayed import; the wrapper cannot prevent this dependency side effect. The project accepts and documents that pinned-library risk, while every project-owned Win32 geometry query still enters/restores thread PMv2 and project code never calls a process-wide DPI API.

The adapter is intentionally locked to the pinned 0.3 API. It obtains structured output metadata from the hidden but accessible singleton `module.DXFactory().outputs` collection. Each `outputs()` call newly traverses that pinned cached collection and, for every device/output index pair, calls `raw_output.update_desc()` before reading and validating `raw_output.hmonitor`, `raw_output.devicename`, `raw_output.desc.DesktopCoordinates`, `raw_output.rotation_angle`, and `raw_output.attached_to_desktop`. The plugin layer never caches a prior match, but DXcam 0.3's singleton collection is not promised to discover a newly hot-plugged topology; missing new topology fails closed and may require a process restart. Only an output with actual `attached_to_desktop is True` is emitted. `DesktopCoordinates` supplies integer left/top/right/bottom desktop coordinates; the adapter requires positive width/height and converts them to `ClientGeometry(left, top, right - left, bottom - top)`. It preserves enumeration indices as `device_index`/`output_index` and produces a tuple of validated `DxcamOutput` values.

All `_Dxcam03FactoryAdapter` instances in the process share one module-scoped coordinator lock and reservation/ownership registry. Its exact coordinator key is `(module identity, device_idx, output_idx, normalized_backend)`; DXGI and WinRT are therefore independent keys. `create(...)` holds that lock continuously while it validates/reserves the coordinator key, obtains `raw_factory = module.DXFactory()`, checks private `_camera_instances[(device_idx, output_idx, normalized_backend)]`, calls `module.create(...)`, validates the returned/registered identity and protocol, and atomically changes the reservation into an adapter-owned `(adapter token, camera identity)` record. No second adapter can interleave a check→create sequence for the same key.

Under that lock, an existing coordinator reservation/ownership record or a private-registry camera whose actual `is_released is False` is an `_DxcamAdapterError` ownership conflict. The adapter never calls `stop()` or `release()` on it. A missing private entry is safe; an entry with actual `is_released is True` may be replaced by pinned 0.3 `module.create`; an absent/invalid release property is API drift. The returned camera must be the exact object at the private registry key, unreleased, and satisfy `DxcamCamera` before ownership registration.

Every failure path releases its coordinator reservation while holding the same lock. If this call demonstrably created and registered a new object before a later validation failure, the lock transfers that exact ownership record to one local cleanup owner and removes the coordinator entry; `camera.stop()`/`release()` then run outside the coordinator lock. Normal fallback, rebuild, and stop follow the same rule: under the lock prove `(adapter token, key, object identity)`, remove/transfer the record to exactly one owner, then clean outside the lock. No path cleans a pre-existing singleton, a registry-replaced object, or another adapter's camera. A failed create leaves the key retryable; another adapter may retry only after the reservation is gone.

The adapter never parses `output_info()` or another display string. Missing/renamed 0.3 attributes, malformed nesting, invalid rectangles/handles/names/rotation/booleans, an exception from `update_desc`, an ownership conflict, or an incompatible camera raises private `_DxcamAdapterError`. `start()` maps it to `CaptureStartError(LOST, BACKEND_ERROR)`; active/read-time creation maps it to `CaptureSample(LOST, issue=BACKEND_ERROR)`. A structurally valid output tuple with zero or multiple physical matches remains `UNSUPPORTED_OUTPUT`, not API drift.

This narrow adapter contains a deliberate private-API risk inside the `<0.4` pin. Dependency upgrades must first update the fake-module contract and adapter tests; do not broaden the parser or silently guess a new layout.

Start performs this exact sequence:

1. Obtain a stable `Sf6WindowTarget`.
2. Load the DXcam 0.3 factory adapter and call `outputs()` to traverse/update its pinned singleton output objects.
3. Find exactly one output matching monitor handle, case-insensitive device name, and physical desktop geometry.
4. Require the complete client rectangle to fit that output.
5. Translate desktop client coordinates to output-relative coordinates:

```python
desktop = output.desktop_geometry
region = (
    client.left - desktop.left,
    client.top - desktop.top,
    client.left + client.width - desktop.left,
    client.top + client.height - desktop.top,
)
```

6. Try `backend="dxgi"` first.
7. Fully stop/release any partially created DXGI camera before trying `backend="winrt"`.
8. Create each attempt with:

```python
factory.create(
    device_idx=output.device_index,
    output_idx=output.output_index,
    region=region,
    output_color="BGR",
    max_buffer_len=2,
    backend=backend,
    processor_backend="cv2",
)
```

9. Start it with:

```python
camera.start(
    region=region,
    target_fps=target_fps,
    video_mode=False,
)
```

Start and every read use one fail-closed output matcher. It calls adapter `outputs()` anew so every cached 0.3 output receives `update_desc`, requires exactly one value matching target monitor handle, case-insensitive device name, full desktop geometry, and rotation, and requires the complete client rectangle to fit. The plugin never reuses its prior match or falls back to an index when current structured mapping is absent or ambiguous; this is descriptor refresh, not a promise of true hot-plug enumeration.

`read()` obtains pixels and backend time atomically:

```python
camera.get_latest_frame(copy=True, with_timestamp=True)
```

Do not use `grab()` plus `latest_frame_time`; those values can describe different backend frames.

Each read follows this exact order, and no branch may commit a frame ID or backend timestamp early:

1. Under the lifecycle lock, require one active generation and acquire the sole-reader slot; snapshot its camera, generation, stable target, matched output, and factory adapter.
2. Outside the lock, obtain a pre-capture stable target through `find_sf6_window_target`, call adapter `outputs()` to traverse/update every pinned output descriptor, and resolve a fresh unique mapping.
3. If pre-capture identity differs from the attached identity, detach/release and latch `WINDOW_CHANGED`. If identity is equal but geometry, monitor, or output differs, transfer unique cleanup ownership, rebuild for the new target/output, and return one `GEOMETRY_CHANGED`. Do not call the old camera for either branch.
4. Only when the pre-capture target/output exactly equals the active attachment, call `camera.get_latest_frame(copy=True, with_timestamp=True)` outside the lifecycle lock. Capture either its owned result or raised exception without mapping it yet.
5. Reacquire the lifecycle lock immediately after the blocking call and compare both camera identity and generation. If stop/reconfiguration detached it, discard any payload or exception and return `STOPPED`; this generation check has priority over `None`, `is_capturing`, backend errors, and window errors. The stale reader performs no cleanup and makes no further call on that camera.
6. For a still-active generation, release the lock and obtain a post-capture stable target, then call adapter `outputs()` again to update/traverse every pinned descriptor and uniquely map the post-capture output. Finder identity exceptions/disappearance latch `WINDOW_CHANGED`; finder geometry exceptions rebuild and return `GEOMETRY_CHANGED`; minimized/hidden/cloaked map to their typed statuses; unsupported output releases the camera and returns `UNSUPPORTED_OUTPUT`. Every such branch discards the backend payload.
7. Reacquire the lock before interpretation/commit and repeat the camera/generation check so a stop during post-validation also wins as `STOPPED`.
8. Require pre target/output, post target/output, and attached target/output to remain exactly equal. A post identity difference latches `WINDOW_CHANGED`; a same-identity geometry/monitor/output difference rebuilds and returns `GEOMETRY_CHANGED`. Discard the payload in both cases.
9. Only now interpret the backend exception/result, validate timestamp and owned BGR payload, assign the next frame ID, and atomically commit frame-ID/backend-time watermarks together with the `RUNNING` sample.

`GameWindowIdentityChanged`, `GameWindowGeometryChanged`, and second-query disappearance can arise inside either pre/post finder call. For every normal returned target and every available `before`/`after` snapshot on either exception, compare its identity to the source's currently attached identity before considering the exception class. If any available identity is foreign, latch `WINDOW_CHANGED`; only when every available identity equals the attached identity may `GameWindowGeometryChanged` or target/output differences select a geometry rebuild. Thus a nominal `GameWindowGeometryChanged(before=B, after=B)` observed while attached to identity A is a window-identity change, not a rebuild. No such branch accepts that read's payload.

Interpret each read exactly:

| condition | `CaptureSample` result |
|---|---|
| source stopped or captured generation detached before either post-block generation check | `STOPPED`, no frame/issue |
| window minimized | `MINIMIZED`, no frame/issue |
| window hidden or DWM-cloaked | `OCCLUDED`, no frame/issue |
| active DXcam returns `None` and `camera.is_capturing is True` | `LOST + NO_NEW_FRAME` |
| active DXcam returns `None` and `camera.is_capturing is False` | `LOST + BACKEND_ERROR` |
| exact all-zero BGR image | `LOST + BLACK_FRAME` |
| repeated/decreasing backend timestamp | `LOST + NO_NEW_FRAME` |
| identity changed/disappeared/ambiguous before, inside, or after the blocking call | `LOST + WINDOW_CHANGED` |
| same identity but geometry/monitor/output changed before, inside, or after the blocking call | `LOST + GEOMETRY_CHANGED` |
| no unique containing DXcam output | `LOST + UNSUPPORTED_OUTPUT` |
| exception or invalid backend payload | `LOST + BACKEND_ERROR` |
| valid owned BGR image with newer timestamp | `RUNNING + CapturedFrame` |

Near-black, all-white, low-contrast, and pixel-identical-to-previous frames remain valid downstream inputs. Only an exact all-zero array is `BLACK_FRAME`. A successful backend result retains the original timestamp object, receives the next nonnegative source frame ID, does not copy an already owned `copy=True` result again, and does not alter writeability. `camera.is_capturing` is consulted only for an active generation whose atomic result is `None`; a non-boolean value or property failure is `BACKEND_ERROR`. Commit frame-ID and backend-time watermarks only after all pre/post window/output checks, payload validation, and both generation checks. Every rejected result advances neither value.

#### Window changes and lifecycle

Every read revalidates the window and uniquely remaps the current structured DXcam output both before and after the blocking camera call.

- Same identity and unchanged geometry/output: continue.
- Same identity with movement, resize, DPI geometry change, monitor change, or any output-descriptor/mapping change: detach and release the old camera, rebuild using the new output-relative region, and return one `LOST + GEOMETRY_CHANGED` sample before accepting a rebuilt-camera frame.
- Changed process identity: release the camera and latch `WINDOW_CHANGED` until an explicit `stop()`/`start()` cycle.
- Minimized, hidden/cloaked, cross-monitor, or ambiguous output detaches/releases the camera and enters an internal `SUSPENDED` state; return the matching status/issue and never replay cached pixels.
- A `read()` in `SUSPENDED` owns no camera. It first performs the stable target and refreshed unique-output checks: if the suspension condition remains, return it again; if a valid target/output returns, create/start a fresh camera, enter a new camera generation, and only then run the normal ordered capture path. Restore never exposes an old buffer.
- Any in-session creation/start failure or adapter drift during pre-read reconfiguration, post-read rebuild, or suspended restore cleans only this adapter's partial owned object, preserves the source frame-ID/backend-time epoch, remains `SUSPENDED` with no camera, and returns `LOST + BACKEND_ERROR` for that read. The next `read()` repeats stable finder/output/create and may recover. The same failure during initial `start()` instead leaves the source stopped/restartable and raises `CaptureStartError(LOST, BACKEND_ERROR)`.
- `WINDOW_CHANGED` is a distinct latched state and never auto-recovers; only explicit `stop()` then `start()` clears it.

`start()` is non-reentrant. Calling it while starting, running, reconfiguring, or stopping raises `RuntimeError` without creating another camera. `read()` permits at most one in-flight reader; a second concurrent call raises `RuntimeError` without touching camera or source state.

`stop()` may be called from another thread to wake an in-flight DXcam read. It is idempotent. While holding the lifecycle lock, stop or reconfiguration detaches the active camera, increments the camera generation, and transfers unique cleanup ownership of that old camera to exactly one local cleanup path. That owner then calls `camera.stop()` outside the lock and calls `camera.release()` exactly once in `finally`; these are the only camera calls permitted after detachment. The blocking `get_latest_frame` call occurs without the lifecycle lock. After it returns, `read()` reacquires the lock and requires the same camera and generation to remain active; if stop or reconfiguration won, it discards the owned result without emitting a frame, cleaning the camera, or making another call on that stale camera. Start, read, and every non-owner stale-generation path make no camera call after detachment.

Only explicit `stop()`/`start()` begins a new source ordering epoch: stop clears target/output, next-frame ID, and accepted backend-time state even if cleanup raises, so the next successful start begins again at frame ID `0`. Every in-session suspension or geometry/output rebuild preserves the next monotonically increasing source frame ID and the last accepted backend timestamp; a rebuilt camera must produce a strictly newer timestamp before emission. This matches Task 8, which preserves its freshness watermarks for non-running/geometry samples and resets them only through the controller's explicit new-session `engine.reset`.

- [ ] **Step 0: Verify the orchestrator-frozen Task 6 plan base**

```powershell
git status --short
$task6IgnorePath = '.superpowers/sdd/.gitignore'
$task6PlanBasePath = '.superpowers/sdd/task6-plan-base.txt'
if (-not (Test-Path -LiteralPath $task6IgnorePath)) {
    throw 'Orchestrator did not pre-create the SDD ignore file'
}
if (-not [regex]::IsMatch(
    [IO.File]::ReadAllText($task6IgnorePath),
    '\A\*\n\z'
)) {
    throw 'SDD ignore file must contain exactly star plus LF'
}
if (-not (Test-Path -LiteralPath $task6PlanBasePath)) {
    throw 'Orchestrator did not pre-create the frozen Task 6 plan-base evidence'
}
$task6PlanBaseRaw = [IO.File]::ReadAllText($task6PlanBasePath)
if (-not [regex]::IsMatch($task6PlanBaseRaw, '\A[0-9a-f]{40}\n\z')) {
    throw 'Frozen Task 6 plan base must be full lowercase hash plus LF'
}
$task6PlanBase = $task6PlanBaseRaw.Substring(0, 40)
$currentHead = (git rev-parse HEAD).Trim()
if ($task6PlanBase -ne $currentHead) {
    throw 'Frozen Task 6 plan base does not match current HEAD'
}
git check-ignore -q $task6PlanBasePath
if ($LASTEXITCODE -ne 0) { throw 'Task 6 plan-base evidence is not ignored' }
.\.venv\Scripts\python.exe -m pytest -v
```

Expected: Task 5 is committed and integrated, status is clean, and the complete suite reports zero failures. After the reviewed Task 6 plan is committed and before implementation Step 0, the orchestrator/brief stage must use `apply_patch`—never shell redirection, `Set-Content`, or `[IO.File]::WriteAllText`—to pre-create ignored `.superpowers/sdd/.gitignore` with exact content `*` and `task6-plan-base.txt` with the full plan-commit hash plus one newline. Step 0 is read-only: it validates existence, exact hash format, ignore status, and equality to HEAD, and never creates or overwrites either file. The frozen hash remains the only scope base until Step 9.

- [ ] **Step 1: RED -> GREEN for `CaptureIssue` and `CaptureSample` only**

Create the domain tests first and run their exact nodes before editing `domain.py`. Initial RED is the missing `CaptureIssue`/`CaptureSample` import. Implement only the public domain contract above, then make this complete matrix GREEN:

```text
test_capture_sample_requires_running_exactly_when_frame_present
test_capture_sample_lost_requires_exact_capture_issue
test_capture_sample_nonrunning_statuses_forbid_frame_and_issue
test_capture_sample_requires_owned_writable_c_contiguous_bgr
test_capture_sample_requires_image_shape_to_match_geometry
test_capture_sample_is_frozen_and_does_not_copy_or_change_flags
test_captured_frame_time_annotation_is_real_and_fraction_identity_is_preserved
```

Do not import future `capture.py` or `window.py` symbols in this RED batch.

- [ ] **Step 2: RED -> GREEN for `FreshFrameGate`**

Add one missing-module test for `FreshFrameGate`, run it to capture RED, add the minimal `capture.py`, then add and satisfy each behavior one at a time:

```text
test_fresh_gate_accepts_exact_age_and_future_boundaries
test_fresh_gate_rejects_just_beyond_age_and_future_boundaries
test_fresh_gate_requires_strictly_increasing_id_and_timestamp
test_fresh_gate_equal_pixels_with_new_ordering_remain_fresh
test_fresh_gate_rejection_does_not_advance_either_watermark
test_fresh_gate_validation_error_does_not_advance_either_watermark
test_fresh_gate_reset_clears_both_watermarks
test_fresh_gate_accepts_fraction_and_huge_rational_without_coercion
test_fresh_gate_rejects_bool_nonreal_negative_nan_and_infinity
```

- [ ] **Step 3: RED -> GREEN for window values and injected finder**

Stage the first `window.py` import separately, then implement value validation and the finder entirely against a fake `Win32Facade`:

```text
test_window_identity_monitor_and_target_are_frozen_and_strictly_typed
test_finder_uses_executable_basename_not_window_title
test_finder_fails_closed_for_zero_or_multiple_sf6_windows
test_finder_queries_identity_before_minimized_before_hidden_cloaked_before_geometry
test_finder_defines_occluded_as_hidden_or_dwm_cloaked_only
test_finder_rechecks_identity_geometry_and_monitor_before_return
test_finder_detects_same_hwnd_pid_with_changed_creation_ticks
test_finder_second_query_disappearance_is_typed_identity_change
test_finder_identity_change_carries_typed_before_and_optional_after
test_finder_second_identity_change_precedes_new_window_minimized_or_cloaked_state
test_finder_same_identity_geometry_change_carries_two_typed_targets
test_finder_uses_both_client_to_screen_corners
test_finder_accepts_negative_desktop_origin
test_finder_rejects_client_crossing_monitor_boundary
test_dpi_context_restores_previous_context_on_success_and_failure
```

- [ ] **Step 4: RED -> GREEN for the ctypes security boundary**

Use an injected fake raw Win32 API; never call the host Win32 API in tests:

```text
test_process_query_uses_exactly_process_query_limited_information
test_process_query_always_closes_handle_after_every_failure_stage
test_process_creation_filetime_combines_high_and_low_words_into_64_bit_ticks
test_full_process_path_is_reduced_to_basename_and_not_retained
test_window_module_contains_no_vm_read_or_read_process_memory_path
test_thread_dpi_context_calls_set_thread_with_pointer_sized_negative_four
test_thread_dpi_context_restores_exact_returned_pointer_in_finally
test_window_module_uses_no_process_wide_or_per_monitor_v1_dpi_path
test_dpi_context_restores_previous_value_when_client_query_raises
test_dwm_cloaked_query_uses_typed_dwm_attribute_and_fails_closed
```

- [ ] **Step 5: RED -> GREEN for lazy DXcam loading and output mapping**

Add the `pyproject.toml` dependency assertion first, then the import-safety subprocess test, then factory/output tests:

```text
test_pyproject_pins_dxcam_winrt_to_03_series
test_importing_capture_does_not_import_dxcam
test_constructing_source_does_not_call_window_or_factory
test_dxcam_output_is_frozen_and_source_constructor_is_strictly_typed
test_loader_wraps_fake_dxcam_module_in_dedicated_03_adapter
test_03_adapter_reads_only_structured_dxfactory_outputs
test_03_adapter_updates_each_output_and_maps_hmonitor_name_rectangle_rotation_and_indices
test_03_adapter_skips_only_attached_to_desktop_false_outputs
test_03_adapter_create_delegates_exact_keywords_to_module_create
test_03_adapter_rejects_preexisting_unreleased_exact_key_without_create_or_cleanup
test_03_adapter_allows_released_exact_key_replacement_and_registers_returned_identity
test_two_adapters_concurrently_reserve_same_key_and_only_one_calls_module_create
test_failed_create_releases_reservation_and_second_adapter_can_retry
test_dxgi_and_winrt_reservation_keys_are_independent
test_03_adapter_cleanup_touches_only_exact_adapter_owned_camera
test_03_adapter_rejects_registry_return_identity_mismatch_without_cleaning_foreign_camera
test_03_adapter_never_calls_or_parses_output_info
test_03_adapter_api_drift_fails_closed_as_backend_error
test_start_maps_negative_desktop_coordinates_to_output_relative_region
test_start_requires_one_unique_containing_output
test_start_rejects_cross_output_and_mirrored_ambiguous_outputs
test_start_error_exposes_legal_status_issue_pair_for_each_frozen_stage
test_start_passes_bgr_buffer_two_cv2_and_video_mode_false
test_start_cleans_partial_dxgi_camera_before_winrt_fallback
test_all_backend_failures_leave_source_stopped_and_restartable
```

- [ ] **Step 6: RED -> GREEN for read semantics and buffer ownership**

Add and satisfy these cases one at a time:

```text
test_read_uses_get_latest_frame_copy_true_with_timestamp_true
test_none_result_is_no_new_frame_and_never_replays_cached_pixels
test_none_from_active_generation_uses_is_capturing_true_for_no_new_frame
test_none_with_is_capturing_false_or_invalid_is_backend_error
test_equal_pixels_with_new_backend_timestamps_emit_two_running_samples
test_repeated_or_decreasing_backend_timestamp_is_no_new_frame
test_timestamp_rejection_does_not_advance_frame_id_or_time
test_exact_all_zero_image_is_black_frame
test_near_black_white_and_low_contrast_images_reach_running_sample
test_invalid_dtype_shape_layout_or_nonowned_buffer_is_backend_error
test_running_sample_preserves_backend_timestamp_and_array_identity
test_read_order_is_pre_target_output_then_atomic_call_then_generation_then_post_target_output
test_post_read_always_reenumerates_and_uniquely_maps_output
test_identity_change_during_block_discards_payload_and_latches_window_changed
test_geometry_change_during_block_discards_payload_and_rebuilds_camera
test_output_change_during_block_discards_payload_and_rebuilds_camera
test_rotation_only_change_during_block_discards_payload_and_rebuilds_camera
test_mid_finder_identity_and_geometry_exceptions_map_to_distinct_adapter_paths
test_geometry_changed_with_foreign_before_after_latches_window_changed
```

- [ ] **Step 7: RED -> GREEN for changes and lifecycle races**

Add and satisfy these cases with fake cameras and synchronization primitives, never timing sleeps:

```text
test_same_identity_geometry_change_releases_and_rebuilds_before_running
test_identity_change_latches_window_changed_until_restart
test_minimize_and_occlusion_never_return_cached_frame
test_minimize_occlusion_and_unsupported_output_detach_into_suspended_state
test_suspended_restore_rebuilds_before_emitting_fresh_frame
test_suspended_restore_preserves_id_and_rejects_time_at_or_below_last_accepted
test_in_session_rebuild_failure_cleans_partial_stays_suspended_and_retries
test_start_vs_active_pre_post_and_restore_failures_use_distinct_error_envelopes
test_stop_is_idempotent_and_releases_in_finally
test_stop_racing_blocked_read_discards_inflight_frame
test_stop_race_returns_stopped_before_none_backend_or_window_mapping
test_detach_transfers_exactly_one_cleanup_owner
test_stale_read_discards_frame_without_cleanup_or_post_detach_camera_call
test_second_concurrent_read_is_rejected_without_state_change
test_start_while_active_is_rejected_without_second_camera
test_cleanup_failure_still_clears_state_and_allows_restart
test_stop_start_resets_source_frame_id_and_backend_time_epoch
test_geometry_rebuild_preserves_next_monotonic_source_frame_id_and_time_watermark
```

No Task 6 test may import real DXcam, initialize DXGI/WinRT, enumerate real windows, open a real process, launch SF6, use Computer Use, or save a captured image.

- [ ] **Step 8: Run focused/full verification and the complete mutation gate**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_capture_window.py -v
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
```

Expected: all commands PASS. Apply each mutation individually, run its named control, observe failure, and restore it immediately:

| mutation | required killing control |
|---|---|
| permit `RUNNING + None` | CaptureSample combination test |
| commit gate watermark before all checks | rejected-then-valid same-ID trace |
| change age `<=` to `<` | exact age boundary |
| change future `<=` to `<` | exact future boundary |
| accept equal ID or timestamp | strict ordering tests |
| reject equal pixel arrays | equal pixels/new timestamp test |
| omit second window snapshot | identity/geometry change-between-queries tests |
| identify by PID only | creation-time reuse test |
| collapse identity and same-identity geometry exceptions | typed finder/adapter route tests |
| map second-query disappearance as geometry | disappearance-is-identity test |
| check second minimized/cloaked before comparing identity | identity-precedence test |
| filter visibility before executable identity | finder-priority test |
| omit DWM cloaked or claim covered-window detection | exact occluded-definition tests |
| pass desktop coordinates directly | negative-origin output-region test |
| permit a spanning or ambiguous output | cross/mirrored output tests |
| request `PROCESS_VM_READ` | exact OpenProcess-flags test |
| omit `CloseHandle` on an exception | handle-finally test |
| use FILETIME low word only | 64-bit FILETIME composition test |
| use process DPI, `-3`, or integer-truncated context | thread-DPI pointer/restore tests |
| import DXcam at module import/construction | subprocess import-safety test |
| return raw DXcam module instead of `_Dxcam03FactoryAdapter` | fake-module loader test |
| parse `output_info()` or other display strings | no-string-parser spy |
| tolerate missing/renamed 0.3 structured fields | API-drift fail-closed tests |
| call `module.create` when exact singleton key is already unreleased | preexisting-key ownership-conflict test |
| perform singleton check and create without process-scoped reservation lock | concurrent-two-adapter test |
| leak reservation after create failure | failed-create retry test |
| conflate DXGI and WinRT reservation keys | backend-key isolation test |
| stop/release a preexisting or registry-replaced camera | foreign-camera no-cleanup tests |
| treat returned camera as owned without exact registry identity | owned-registry identity test |
| omit output rotation from stable snapshot | rotation-only race test |
| use `video_mode=True` | exact start-call test |
| use `grab()` or separate `latest_frame_time` | atomic-call spy |
| use `copy=False` | ownership/call-signature test |
| replay the previous result after `None` | no-new-frame test |
| map every active-generation `None` to `NO_NEW_FRAME` | `is_capturing` true/false tests |
| treat near-black/white/low contrast as black | black-boundary tests |
| omit partial-camera cleanup before fallback | fallback call-order test |
| omit pre/post target or post output validation | frozen read-order and remap spies |
| accept payload after mid-block identity/geometry/output change | synchronized mid-block race tests |
| trust `GeometryChanged` class before comparing snapshots to attached identity | attached-A/geometry-B-to-B test |
| omit camera rebuild after geometry change | geometry-rebuild test |
| reset source frame ID during in-session rebuild | rebuilt-next-monotonic-ID test |
| retain camera/buffer while minimized, cloaked, or unsupported | suspended-detach/restore tests |
| reset ID/time on suspend restore or accept timestamp `<= T` | suspend N/T then restore N+1/newer-time test |
| leave partial camera/ownership or stop retry after rebuild failure | partial-cleanup suspended-retry test |
| return an in-flight frame after stop | generation-race test |
| map stop-race payload/None/error before generation | STOPPED-priority test |
| give stop and stale read duplicate cleanup ownership | unique-cleanup-owner test |
| let stale read clean or call the detached camera | stale-read-no-post-detach-call test |
| omit `release()` when `stop()` raises | cleanup-finally test |

- [ ] **Step 9: Verify exact scope and commit**

```powershell
$task6PlanBasePath = '.superpowers/sdd/task6-plan-base.txt'
if (-not (Test-Path -LiteralPath $task6PlanBasePath)) {
    throw 'Frozen Task 6 plan-base evidence is missing'
}
$task6PlanBaseRaw = [IO.File]::ReadAllText($task6PlanBasePath)
if (-not [regex]::IsMatch($task6PlanBaseRaw, '\A[0-9a-f]{40}\n\z')) {
    throw 'Frozen Task 6 plan base must be full lowercase hash plus LF'
}
$task6PlanBase = $task6PlanBaseRaw.Substring(0, 40)
$currentHead = (git rev-parse HEAD).Trim()
if ($currentHead -ne $task6PlanBase) {
    throw 'HEAD moved after the Task 6 plan base was frozen'
}
git check-ignore -q $task6PlanBasePath
if ($LASTEXITCODE -ne 0) { throw 'Task 6 plan-base evidence is not ignored' }
git diff --check
git diff --name-only $task6PlanBase
git status --short
```

The implementation diff against the independently reviewed plan base contains exactly:

```text
pyproject.toml
src/sf6_music_trigger/domain.py
src/sf6_music_trigger/capture.py
src/sf6_music_trigger/window.py
tests/test_capture_window.py
```

Verify no forbidden APIs, real capture artifacts, absolute user paths, screenshots, logs, audio, or Task 7-9 implementation entered the diff. Then commit:

```powershell
git add pyproject.toml src/sf6_music_trigger/domain.py src/sf6_music_trigger/capture.py src/sf6_music_trigger/window.py tests/test_capture_window.py
git commit -m "feat: add typed fresh window capture boundary"
```

---

### Task 7: Pure Audio Policy and Pygame Adapter

**Files:**
- Create: `src/sf6_music_trigger/audio.py`
- Create: `tests/test_audio.py`

**Interfaces:**
- Produces:
  - `AudioKind(str, Enum)` with exact values `STOP = "stop"` and `PLAY = "play"`.
  - frozen `AudioAction(kind: AudioKind, side: Side, track: str | None = field(default=None, repr=False), volume: float | None = None)`.
  - stateless `AudioScheduler(side: Side)` and `on_confirmed_spend(event: ConfirmedSuperEvent, config: SideAudioConfig) -> tuple[AudioAction, ...]`.
  - `AudioIssue`, `AudioPlaybackError`, `AudioPort` protocol, and reusable lazy `PygameAudioPort`.
- Consumes: Task 2 `Side`, `ConfirmedSuperEvent`, and `SideAudioConfig`; Python path/threading primitives; pygame only through an injected lazy loader.

`AudioAction` is the complete Phase-1 policy value. It requires actual `AudioKind` and `Side` values. `STOP` requires both payload fields to be `None`. `PLAY` requires a string track whose value is not `""` and a finite `numbers.Real` volume other than `bool` in the closed range `[0.0, 1.0]`. For an actual `numbers.Rational` (including `int` and `Fraction`), compare against 0 and 1 exactly before any float conversion; reject an out-of-range value, then convert the accepted value to `float`. For a non-Rational `Real`, require `math.isfinite(value)`, perform the range check, then convert. This makes huge integers fail with the required `ValueError` rather than leaking `OverflowError`. `as_tuple()` returns exactly `(kind.value, track, volume)` and deliberately omits `side` only for compact display compatibility; downstream code uses the retained typed `.side`.

`AudioScheduler` stores only its typed side. `on_confirmed_spend` first requires actual `ConfirmedSuperEvent` and `SideAudioConfig` values and an exact event/scheduler side match. A disabled config returns `()` even if its track is empty. An enabled config with `main_track == ""` raises `ValueError`. Every successful call—including a repeated call with the same event ID—returns a fresh exact pair:

```python
(
    AudioAction(AudioKind.STOP, event.side),
    AudioAction(AudioKind.PLAY, event.side, config.main_track, config.volume),
)
```

There is no `pending_actions`, deduplication set, fade queue, busy-state query, or mutable scheduling state. Phase 1 implements confirmed-spend replacement only; block, whiff, hit, fade, outcome policy, and decoded-sound caching remain outside Task 7.

`AudioPlaybackError` is a `RuntimeError` with only these typed public fields:

```python
operation: Literal[
    "initialize",
    "execute",
    "stop",
    "load",
    "set_volume",
    "play",
    "preview",
    "preview_stop",
    "close_stop",
]
side: Side | None
issue: AudioIssue
```

`AudioIssue(str, Enum)` has exact values `BACKEND_FAILURE = "backend_failure"`, `ACTIVE_PORT_CONFLICT = "active_port_conflict"`, `CLOSE_INCOMPLETE = "close_incomplete"`, and `TRACK_REJECTED = "track_rejected"`. The error's human-readable message contains only safe enum values and optional side, never a track, basename, root, underlying exception text, or other private path material. `repr(error)`, `vars(error)`, `error.args`, `error.__cause__`, and `error.__context__` must also contain no private path/raw backend text; cause and context are `None`. Therefore catch backend/path exceptions, retain only a safe internal failure flag, leave the `except` block, discard the raw exception, and raise the typed privacy envelope outside the active exception context. `AudioAction.track` is excluded from dataclass `repr`. The port itself does not print, log, serialize, or persist paths or errors.

`AudioPort` is a structural protocol with these exact methods:

```python
def execute(self, actions: tuple[AudioAction, ...]) -> None
def preview(self, track: str, volume: Real) -> None
def close(self) -> None
```

`execute` acquires the module process `RLock` immediately on method entry, before structural checks or path resolution, and holds it until the complete batch succeeds or raises. Under that lock it requires an actual nonempty tuple containing only `AudioAction` values for one exact `Side`. Before loader, reservation, mixer, channel, or reference effects, validate the complete tuple and resolve every PLAY path into a local prepared batch. Thus `(STOP, PLAY("../escape"))` rejects with zero backend/ownership change instead of stopping current audio first. Keeping preflight under the same lock ensures close cannot return while an earlier execute is resolving a path and then be followed by that execute reserving/playing. Prepared actions then run in tuple order, so preview, close, and another port cannot interleave. Backend execution is fail-fast and raises the first failing action's `AudioPlaybackError`; a later event/side is still attempted by Task 8.

`PygameAudioPort(user_data_root: str | os.PathLike[str], *, pygame_loader: Callable[[], object] = _load_pygame)` canonicalizes the user-data root at construction but does not invoke the loader. Importing `sf6_music_trigger.audio`, constructing a port, and calling `close()` on a never-used port must neither import pygame nor initialize a mixer.

Because pygame mixer channels are process-global, every port instance shares one module-scoped `threading.RLock` and a single active-port reservation. `execute`, `preview`, and `close` acquire that lock at method entry, before validation/preflight, and hold it through every effect or error exit. The first structurally/path-valid `execute` or `preview` reserves the mixer for that port; the same port may reuse the reservation, but a different live port fails with `AudioPlaybackError("initialize", side=..., issue=ACTIVE_PORT_CONFLICT)` before touching the mixer. If initialization fails before a complete channel set/reference state is published, atomically clear partial local state and roll back that provisional reservation so this or another port can retry. Closing a never-used or non-active port cannot release another port's reservation. After successful initialization, only complete successful `close()` releases ownership.

Before backend initialization, a `PLAY` or preview validates its track with Windows path semantics matching Task 2:

- reject device namespaces (`\\?\` and `\\.\`), NTFS alternate-data-stream colons, drive-relative paths such as `D:music\song.mp3`, root-relative paths such as `\music\song.mp3`, lexical `..` escape, and directory symlink/junction escape;
- preserve an accepted drive-rooted or UNC absolute string byte-for-byte as the argument to `mixer.Sound`;
- resolve an accepted relative path against the canonical user-data root, re-check resolved containment at call time, and pass the resulting canonical local string to `mixer.Sound`;
- do not require existence, normalize an accepted external absolute string, infer safety from `Path.is_absolute()` alone, or allow a sibling-prefix containment false positive;
- do not use an extension allowlist: pygame, not the policy layer, decides whether selected content can be decoded.

Resolve and validate every PLAY/preview path at call time; never cache a previous containment result. A test must construct the port first, then create an escaping junction under the user-data root, then prove the actual playback call rejects it before loader/Sound access.

This inherits Task 3's explicitly non-adversarial local-data TOCTOU boundary: resolved containment protects normal user mistakes and already-present links/junctions, but Task 7 does not claim resistance to a malicious concurrent reparse-point swap between validation and pygame's path-based `Sound` open. Closing that race would require a handle-based decoder/load API that pygame does not expose and is outside Phase 1.

Structural caller errors remain ordinary path-neutral `TypeError`/`ValueError`: a non-string/empty preview track, a non-tuple/empty/mixed-side batch, or a malformed `AudioAction`. Once `execute`/`preview` has a structurally valid track, any call-time Windows classification, containment, or resolution rejection—including `PureWindowsPath`/`Path.resolve`/comparison exceptions—becomes exactly `AudioPlaybackError(operation="load", side=batch_side_or_None, issue=AudioIssue.TRACK_REJECTED)` before loader/reservation/channel/reference effects. Discard the raw path exception outside its active context; message, repr, vars, args, cause, and context expose no input path or raw OS text. This distinction lets Task 8/9 safely report a track that becomes invalid after Start without terminating the worker while preserving programmer-error visibility for malformed protocol values.

`preview` separately requires a finite non-boolean `Real` and clamps it to `[0.0, 1.0]` without unsafe eager conversion. For an actual `Rational`, return `0.0` when `value <= 0` and `1.0` when `value >= 1`, otherwise convert to float. For a non-Rational `Real`, first require `math.isfinite(value)`, then apply the same comparisons and only convert an interior value. Thus `-10**10000`/`10**10000` clamp without `OverflowError`, while NaN/infinity reject. A `PLAY` action's volume is already strict and is never re-clamped. Task 9 alone converts UI percentages from `0..100` to this `0.0..1.0` boundary.

The first valid `execute` or `preview` call invokes `pygame_loader()` and establishes the entire local backend atomically:

1. obtain `pygame.mixer`;
2. call `mixer.init()` only when `mixer.get_init()` reports uninitialized;
3. read `mixer.get_num_channels()`; when it is below three, call `mixer.set_num_channels(3)`, then read and require the count to be at least three;
4. obtain exactly `mixer.Channel(0)`, `mixer.Channel(1)`, and `mixer.Channel(2)` for P1, P2, and preview;
5. publish local pygame/mixer/channel state only after every step succeeds.

Any loader, mixer-init, channel-count, or channel-construction failure clears partial local state, rolls back only that provisional reservation, and raises `AudioPlaybackError("initialize", side=..., issue=BACKEND_FAILURE)` without a raw cause. It never calls `pygame.quit`, `mixer.quit`, or `mixer.stop`; a later call may reacquire and retry against the same process-global mixer.

Execution has no hidden replacement behavior:

- `STOP` calls only its fixed side channel's `stop()`. It clears that side's retained Sound reference only after stop succeeds or the mixer is known uninitialized; on failure it preserves the channel handle and strong reference because audio may still be playing and must remain retryable.
- `PLAY` performs exactly `mixer.Sound(resolved_track)`, `sound.set_volume(action.volume)`, then its fixed side channel's `play(sound)`. It never calls `stop`, `find_channel`, `Sound.play`, or an automatic-channel API.
- Only after `Channel.play` succeeds does `PLAY` commit the new strong Sound reference for that side. It retains no path copy. A failed load/set/play does not replace an earlier successful reference.
- `preview(track, volume)` affects only Channel 2 and performs exact order `channel.stop()`, `mixer.Sound(resolved_track)`, `sound.set_volume(clamped_volume)`, `channel.play(sound)`. Its failure operations are exactly `"preview_stop"`, `"load"`, `"set_volume"`, and `"play"` respectively, always with `side=None` and `issue=BACKEND_FAILURE`. It clears the previous preview reference only after stop succeeds or the mixer is known uninitialized; a stop failure aborts preview and preserves the handle/reference for retry. It commits the new reference only after play succeeds and never touches P1/P2 channels or references.
- Backend stop/load/set/play failures map to the corresponding typed `AudioPlaybackError` with only the operation and side (`None` for preview), a privacy-safe message, and no raw cause. During partial close, `execute` and `preview` fail before other effects with exact operations `"execute"`/`"preview"`, the relevant batch side or `None`, and `AudioIssue.CLOSE_INCOMPLETE`.

`close()` is idempotent after complete success and makes the same port reusable after Task 9 stop/restart. While holding the process lock, an initialized active port first checks mixer initialization. If the mixer is known uninitialized, no owned channel can still play: clear all local handles/references and release the reservation without channel calls. If `mixer.get_init()` itself fails, retain every handle/reference/pending mark and the active owner, discard the raw exception outside its context, and raise exactly `AudioPlaybackError(operation="close_stop", side=None, issue=AudioIssue.BACKEND_FAILURE)`. A later `close()` retries the initialization check and cleanup.

Otherwise close best-effort calls `stop()` on each still-pending owned channel in deterministic P1, P2, preview order even when an earlier channel fails. A successful stop clears that channel's Sound reference and marks it complete, but retains the complete Channel handle map until whole-close success; a retry skips completed channels and never double-stops them. A failed stop retains that channel's handle/reference and pending mark. If any channel remains pending, keep all local pygame/mixer/Channel handles and the active-port reservation, reject `execute`/`preview` before other effects with operations `"execute"`/`"preview"` and `AudioIssue.CLOSE_INCOMPLETE`, and raise the first `AudioPlaybackError("close_stop", side=..., issue=BACKEND_FAILURE)` after attempting all pending channels. A later `close()` retries only failed channels. Only after every owned channel is successfully stopped does close clear backend handles, leave partial-close state, and release its own reservation. It never stops unrelated channels or invokes global quit/stop APIs; a subsequently valid call may reserve again and lazily obtain fresh Channel 0/1/2 handles.

The reservation coordinates only `PygameAudioPort` instances from this module. The standalone application assumes exclusive ownership of pygame Channel 0/1/2; unrelated external pygame code in the process is outside this boundary. pygame loader/init/Sound/channel calls are synchronous. If such a backend call permanently blocks while holding the process audio lock, preview/close waits and Task 7 has no safe forced-interruption or hard-exit-timeout mechanism; global mixer quit would violate ownership and is forbidden. Record this as a backend limitation rather than claiming bounded shutdown.

- [ ] **Step 0: Verify the orchestrator-frozen Task 7 plan base and dependency**

After this Task 7 plan receives independent review and is committed, and before implementation begins, the orchestrator uses `apply_patch`—not shell redirection, `Set-Content`, or `[IO.File]::WriteAllText`—to create ignored `.superpowers/sdd/task7-plan-base.txt` containing the complete lowercase plan-commit hash plus one LF. Existing `.superpowers/sdd/.gitignore` must contain exactly `*` plus LF. The implementation worker runs this read-only gate; it never creates or overwrites either evidence file:

```powershell
$status = @(git status --short)
if ($status.Count -ne 0) {
    throw "Task 7 plan base is not clean: $($status -join '; ')"
}
$ignorePath = '.superpowers/sdd/.gitignore'
$basePath = '.superpowers/sdd/task7-plan-base.txt'
if (-not (Test-Path -LiteralPath $ignorePath)) {
    throw 'Orchestrator did not pre-create the SDD ignore file'
}
if (-not [regex]::IsMatch([IO.File]::ReadAllText($ignorePath), '\A\*\n\z')) {
    throw 'SDD ignore file must contain exactly star plus LF'
}
if (-not (Test-Path -LiteralPath $basePath)) {
    throw 'Orchestrator did not freeze the Task 7 plan base'
}
$baseRaw = [IO.File]::ReadAllText($basePath)
if (-not [regex]::IsMatch($baseRaw, '\A[0-9a-f]{40}\n\z')) {
    throw 'Frozen Task 7 plan base must be a full lowercase hash plus LF'
}
$base = $baseRaw.Substring(0, 40)
$head = (git rev-parse HEAD).Trim()
if ($base -ne $head) {
    throw 'Frozen Task 7 plan base does not match current HEAD'
}
git check-ignore -q $basePath
if ($LASTEXITCODE -ne 0) { throw 'Task 7 plan-base evidence is not ignored' }
.\.venv\Scripts\python.exe -c "from importlib.metadata import version; from packaging.specifiers import SpecifierSet; v = version('pygame'); assert v in SpecifierSet('>=2.5,<3'), v; print(v)"
.\.venv\Scripts\python.exe -m pytest -v
```

Expected: status has no non-ignored changes; the frozen base equals current HEAD; metadata validates the installed pygame version against `>=2.5,<3` without importing pygame (this workstation currently reports `2.6.1` as evidence, not as an interface pin); and the Task 6 merged baseline suite reports zero failures. Missing/out-of-range pygame is an environment dependency blocker to repair before Task 7, not a skip or a passing smoke.

- [ ] **Step 1: RED -> GREEN for module import only**

Create `tests/test_audio.py` with exactly one initial test. Keep the import inside the test body so pytest collection succeeds:

```python
def test_audio_module_contract_is_absent_then_added():
    import importlib

    module = importlib.import_module("sf6_music_trigger.audio")
    assert module.__name__ == "sf6_music_trigger.audio"
```

Run it while `audio.py` is absent:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py::test_audio_module_contract_is_absent_then_added -v
```

Expected RED: the test is collected and reported `FAILED` with `ModuleNotFoundError` raised inside the test body; this is not a collection error. Then create `src/sf6_music_trigger/audio.py` containing only its real module docstring and the standard-library/Task-2 imports already required by the final design—no `pass`, placeholder value, stub method, or `NotImplementedError`—and rerun the same command to record GREEN.

For every later cluster, append all tests for that cluster first, keep newly missing production-symbol imports inside test bodies/helpers so collection remains successful, run the named cluster and record at least one behavior-level `FAILED`, implement only that cluster's contract, then rerun it GREEN. The ignored SDD report records each exact RED and GREEN command/output.

The following is the complete coverage catalogue, distributed across Steps 2–7; every item appears in its named RED cluster before its production behavior:

- action/enum: exact enum strings; frozen mutation failure; `as_tuple` shape; wrong kind/side types; STOP with either payload; PLAY with non-string/empty track; bool/non-`Real`/NaN/infinity/below-zero/above-one volume; exact `0.0`/`1.0` acceptance; `Fraction(1, 10**10000)` safe conversion; and `±10**10000` rejection as `ValueError` without `OverflowError`;
- scheduler: invalid constructor side; invalid event/config types; event-side mismatch even when disabled; disabled config returning `()` with empty or nonempty track; enabled empty track failure; P1/P2 action-side isolation; exact STOP-before-PLAY order; repeated identical event; and proof that the instance retains no queue/dedup/fade state;
- error privacy: exact `AudioIssue` values; each permitted operation and typed side/issue field; invalid field types; no track attribute; `AudioAction` repr redaction; and a secret absolute/UNC track plus path-bearing backend exception absent from `str(error)`, `repr(error)`, `vars(error)`, `error.args`, `error.__cause__`, and `error.__context__`;
- module/laziness: a clean subprocess whose import hook raises on `pygame` can import `sf6_music_trigger.audio`, construct a port, and call unused `close()`; injected loader call count stays zero for construction, unused close, and rejected track/preview-volume inputs;
- backend initialization: uninitialized mixer init; already initialized mixer without duplicate init; existing count `>= 3` without resize; count `< 3` exact resize to three and mandatory re-read; still-too-small re-read; loader/mixer/init/count/set-count/each Channel constructor failure; partial state not published; provisional reservation rollback and same/other-port reacquisition after every initialize failure; exactly Channel IDs `0,1,2`; no `find_channel`; concurrent same-port first calls publishing one complete local channel set; two-port reservation conflict before mixer access; non-owner close isolation; close release; and same-port re-reservation;
- path boundary: exact drive-rooted and UNC string preservation; canonical relative resolution; `..` and sibling-prefix escape; both slash styles; device namespaces; ADS; drive-relative/root-relative forms; non-string/empty structural values; no extension/existence precheck; an existing directory symlink with unprivileged junction fallback; construct-port-first/create-escaping-junction-second proving actual-call revalidation; exact `TRACK_REJECTED/load` mapping with zero loader effects; and injected classification/resolve/containment exceptions whose message/repr/vars/args/cause/context reveal no path or raw OS text;
- batch/order/reference ownership: reject list/empty/mixed-side/invalid-element batches before mixer access; `(STOP, PLAY("../escape"))` has zero loader/reservation/channel/ref effect; one actual STOP→PLAY tuple holds the process lock continuously; a latch-blocked batch proves close cannot enter between STOP and PLAY and waits to leave the final channel silent; a latch-blocked preview proves close waits until preview completes then silences Channel 2; only an explicit execute after completed close may re-reserve; STOP makes only fixed-channel `stop` and preserves ref/handle on failure; PLAY makes exact Sound/set-volume/fixed-channel-play order without a second stop; preview maps stop/load/set/play exactly and preserves ref/handle on stop failure; preview clamps `±10**10000` without overflow, accepts a tiny `Fraction`, and rejects bool/NaN/infinity; P1/P2/preview references remain independent; and a new strong reference appears only after successful play;
- injected operation failures: side STOP; Sound construction; `set_volume`; side `Channel.play`; every preview stage; correct `AudioPlaybackError` operation/side only; batch fail-fast; failure on one side does not mutate the other side's reference; no path/raw cause; and no calls to `pygame.quit`, `mixer.quit`, or `mixer.stop`;
- close/reuse: unused close does not load or reserve; known-uninitialized mixer clears safely; `mixer.get_init()` failure maps to exact safe `close_stop/None/BACKEND_FAILURE` while retaining all state/owner for retry; close tries every pending owned channel in order; successful channels clear and are not double-stopped on retry; failed channels retain handles/refs/owner; execute/preview map partial-close gates exactly; repeated close retries only failed channels; first typed error wins each pass; reservation releases only after all channels succeed; non-owner close cannot release it; complete close is idempotent; and a valid call after complete close re-reserves/rebuilds on the same port without global shutdown;
- SDL dummy: a subprocess sets `SDL_AUDIODRIVER=dummy` and `PYGAME_HIDE_SUPPORT_PROMPT=1` before importing pygame, creates a 100 ms silent WAV only under pytest's temporary directory with the standard `wave` module, executes P1 STOP/PLAY plus preview, closes, reuses the same port for P2 STOP/PLAY, and closes again. It asserts exit code zero and leaves no repository audio file.

The fake pygame graph records every loader/mixer/Sound/channel call and exposes failure injection at each named boundary. It must fail if production uses `find_channel`, `Sound.play`, an automatic channel, a global stop/quit, a hidden second STOP before PLAY, premature reference commit, a non-atomic batch, competing active ports, or a private path/raw cause in an error. The SDL subprocess checks `importlib.metadata.version("pygame") in SpecifierSet(">=2.5,<3")` first; missing/out-of-range dependency fails with an explicit environment-block message rather than skipping.

- [ ] **Step 2: RED -> GREEN for values, stateless scheduler, and safe error envelope**

Append the complete action/enum/scheduler/error-structure catalogue before editing production. Representative tests use body-local imports:

```python
def test_confirmed_spend_returns_exact_typed_stop_then_play(event_factory):
    from sf6_music_trigger.audio import AudioKind, AudioScheduler

    event = event_factory(side=Side.P1, event_id="evt-1")
    config = SideAudioConfig(main_track="audio/main.mp3", volume=0.18)
    actions = AudioScheduler(Side.P1).on_confirmed_spend(event, config)
    assert [action.kind for action in actions] == [AudioKind.STOP, AudioKind.PLAY]
    assert tuple(action.as_tuple() for action in actions) == (
        ("stop", None, None),
        ("play", "audio/main.mp3", 0.18),
    )
    assert tuple(action.side for action in actions) == (Side.P1, Side.P1)


def test_play_volume_handles_huge_rationals_without_overflow():
    from fractions import Fraction
    from sf6_music_trigger.audio import AudioAction, AudioKind

    with pytest.raises(ValueError):
        AudioAction(AudioKind.PLAY, Side.P1, "track.bin", 10**10000)
    with pytest.raises(ValueError):
        AudioAction(AudioKind.PLAY, Side.P1, "track.bin", -(10**10000))
    tiny = AudioAction(
        AudioKind.PLAY,
        Side.P1,
        "track.bin",
        Fraction(1, 10**10000),
    )
    assert tiny.volume == float(Fraction(1, 10**10000))


def test_repeated_event_returns_fresh_equal_actions(event_factory):
    from sf6_music_trigger.audio import AudioScheduler

    scheduler = AudioScheduler(Side.P2)
    event = event_factory(side=Side.P2, event_id="same-event")
    config = SideAudioConfig(main_track="audio/p2.mp3", volume=0.2)
    first = scheduler.on_confirmed_spend(event, config)
    second = scheduler.on_confirmed_spend(event, config)
    assert first == second
    assert first is not second
    assert all(left is not right for left, right in zip(first, second, strict=True))
    assert not hasattr(scheduler, "pending_actions")
```

Include explicit bool/non-Real/NaN/±infinity tests, exact 0/1, STOP payload rules, frozen/repr/as-tuple behavior, all scheduler type/side/disabled/empty/repeated/no-state cases, all `AudioIssue` values, and error field/message/repr/vars/args validation. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v -k "kind or action or scheduler or error_envelope"
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v
```

Expected RED: the targeted command and current complete file both report `FAILED` inside test bodies because the named values/behaviors are absent; no collection error. Implement only `AudioKind`, `AudioIssue`, frozen `AudioAction`, `AudioPlaybackError`, and `AudioScheduler` with the exact rational/real rules. Rerun both commands GREEN before continuing.

- [ ] **Step 3: RED -> GREEN for lazy initialization and process ownership**

Append the fake-pygame foundation and only the import/construct/unused-close plus backend initialization, reservation, and fixed-channel-allocation subset. Fake channels support a STOP-only batch so initialization is exercised through the public protocol. Tests first require `AudioPort`/`PygameAudioPort`, import safety, zero loader calls for construct/unused close, init/count/re-read/channel acquisition, atomic publication, provisional-owner rollback, competing-port rejection, and concurrent first-use linearization. Rejected-track behavior belongs only to Step 4; preview-volume behavior belongs only to Step 5.

Use one shared `fake_audio` fixture with this exact observable graph: `probe.port`; `probe.loader.calls: int` and `probe.loader.module`; `probe.mixer.calls: list[tuple]`, `initialized: bool`, `num_channels: int`, `channels: dict[int, FakeChannel]`, `sounds: list[FakeSound]`, `sound_paths: list[str]`, and `fail_next(stage: str)`; `probe.new_port(root=None)` uses the same fake module/mixer; `probe.reset_calls()` clears call logs only. Each `FakeChannel` exposes `channel_id`, `calls`, `playing`, `stop_failures_remaining`, `play_failures_remaining`, `play_entered: Event | None`, and `allow_play: Event | None`. `stop()` logs first, raises the injected path-bearing error before changing `playing`, otherwise sets it false. `play(sound)` logs, signals/waits on latches, raises before state change when injected, otherwise sets `playing=True`. `FakeSound` exposes `path`, `volume`, `calls`, and `set_volume_failures_remaining`. `FakeMixer` implements only `get_init/init/get_num_channels/set_num_channels/Channel/Sound`; forbidden `find_channel/stop/quit` methods raise `AssertionError`. The fake pygame module's `quit` also raises.

```python
def test_initialize_resizes_and_reserves_fixed_channels(fake_audio):
    from sf6_music_trigger.audio import AudioAction, AudioKind

    fake_audio.mixer.initialized = False
    fake_audio.mixer.num_channels = 1
    fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    assert fake_audio.loader.calls == 1
    assert fake_audio.mixer.calls[:7] == [
        ("get_init",),
        ("init",),
        ("get_num_channels",),
        ("set_num_channels", 3),
        ("get_num_channels",),
        ("Channel", 0),
        ("Channel", 1),
    ]
    assert ("Channel", 2) in fake_audio.mixer.calls
    assert fake_audio.mixer.channels[0].calls == [("stop",)]
```

Exact RED node: `tests/test_audio.py::test_initialize_resizes_and_reserves_fixed_channels`. Before this cluster's implementation it fails inside the test because `PygameAudioPort`/its behavior is absent; after the minimal STOP-path initialization state transition it passes.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py::test_initialize_resizes_and_reserves_fixed_channels -v
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v -k "import_safety or lazy or initialize or reservation or channel_allocation"
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v
```

Expected RED: the exact node, targeted subset, and current complete file report `FAILED` because the lazy port/ownership behavior is absent. Implement the structural batch protocol, process `RLock`, provisional/active reservation, injected-loader initialization, fixed Channel 0/1/2 acquisition, STOP-only execution needed by this cluster, privacy-safe initialization failures, and unused close. Do not add path resolution, PLAY, preview, partial-close, or the real default pygame loader yet. Rerun all three commands GREEN.

- [ ] **Step 4: RED -> GREEN for path classification and whole-batch preflight**

Append every structural-batch and path-boundary test before production: list/empty/mixed-side/invalid elements, exact drive/UNC preservation, relative canonicalization, all Windows rejects, sibling/link/junction escapes, construction-then-junction escape, no extension/existence filter, and injected classification/resolve/containment failures. The critical trace is `(STOP, PLAY("../escape"))` producing one safe `load/<side>/TRACK_REJECTED` and zero loader/reservation/channel/reference effects; structural values still raise ordinary `TypeError`/`ValueError`.

```python
def test_unsafe_play_preflight_has_zero_stop_loader_or_owner_effect(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    actions = (
        AudioAction(AudioKind.STOP, Side.P1),
        AudioAction(AudioKind.PLAY, Side.P1, "../private.mp3", 0.2),
    )
    with pytest.raises(AudioPlaybackError) as caught:
        fake_audio.port.execute(actions)
    error = caught.value
    assert (error.operation, error.side, error.issue) == (
        "load",
        Side.P1,
        AudioIssue.TRACK_REJECTED,
    )
    assert fake_audio.loader.calls == 0
    assert fake_audio.mixer.calls == []
    contender = fake_audio.new_port()
    contender.execute((AudioAction(AudioKind.STOP, Side.P2),))


def test_blocked_preflight_linearizes_before_close(fake_audio, monkeypatch):
    from threading import Event, Thread
    import sf6_music_trigger.audio as audio_module
    from sf6_music_trigger.audio import AudioAction, AudioKind

    entered = Event()
    allow = Event()
    close_done = Event()

    def blocked_resolve(root, track):
        entered.set()
        assert allow.wait(1.0)
        return str(root / "audio/p1.bin")

    monkeypatch.setattr(audio_module, "_resolve_audio_track", blocked_resolve)
    action = AudioAction(AudioKind.PLAY, Side.P1, "audio/p1.bin", 0.2)
    play_thread = Thread(target=lambda: fake_audio.port.execute((action,)))
    close_thread = Thread(
        target=lambda: (fake_audio.port.close(), close_done.set())
    )
    play_thread.start()
    assert entered.wait(1.0)
    close_thread.start()
    assert not close_done.wait(0.05)
    allow.set()
    play_thread.join(1.0)
    close_thread.join(1.0)
    assert close_done.is_set()
    assert not fake_audio.mixer.channels[0].playing
```

Exact RED nodes: `tests/test_audio.py::test_unsafe_play_preflight_has_zero_stop_loader_or_owner_effect` and `tests/test_audio.py::test_blocked_preflight_linearizes_before_close`. They initially fail because unsafe PLAY is not privacy-wrapped/preflighted or the lock begins after resolution. Minimal GREEN acquires the process lock at entry, prepares every PLAY path before effects, emits one safe rejection, and leaves the coordinator unowned on rejection; close cannot overtake an already-entered preflight.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py::test_unsafe_play_preflight_has_zero_stop_loader_or_owner_effect -v
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py::test_blocked_preflight_linearizes_before_close -v
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v -k "batch_validation or track_path or preflight or track_rejected"
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v
```

Expected RED: the exact nodes, targeted subset, and current complete file report `FAILED` because valid/invalid paths and atomic preflight are not implemented. Add call-time Windows classification, canonical-root containment, full-batch preparation before effects, path-exception privacy wrapping, and the minimum PLAY load/set/channel-play path needed to observe accepted resolved strings. Rerun all commands GREEN; no raw path, loader call, stop, or reservation change occurs on rejected preflight.

- [ ] **Step 5: RED -> GREEN for atomic execute and isolated preview**

Append the complete command-order/reference/failure/concurrency catalogue: STOP→PLAY has no hidden second stop; Sound/set/play order; strong refs only after success; batch fail-fast; exact backend error mapping; P1/P2 independence; preview Channel-2 stop/load/set/play mapping; `±10**10000`, tiny `Fraction`, bool/NaN/infinity; and latch tests proving batch/preview linearize against close. All fake failure text includes a secret path and every surfaced envelope asserts safe message/repr/vars/args/cause/context.

```python
def test_event_batch_linearizes_before_close_and_ends_silent(fake_audio):
    from threading import Event, Thread
    from sf6_music_trigger.audio import AudioAction, AudioKind

    channel = fake_audio.mixer.channels[0]
    channel.play_entered = Event()
    channel.allow_play = Event()
    actions = (
        AudioAction(AudioKind.STOP, Side.P1),
        AudioAction(AudioKind.PLAY, Side.P1, "audio/p1.bin", 0.2),
    )
    batch_errors: list[BaseException] = []
    close_errors: list[BaseException] = []
    close_done = Event()

    def run_batch():
        try:
            fake_audio.port.execute(actions)
        except BaseException as error:
            batch_errors.append(error)

    def run_close():
        try:
            fake_audio.port.close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_done.set()

    batch_thread = Thread(target=run_batch)
    close_thread = Thread(target=run_close)
    batch_thread.start()
    assert channel.play_entered.wait(1.0)
    close_thread.start()
    assert not close_done.wait(0.05)
    channel.allow_play.set()
    batch_thread.join(1.0)
    close_thread.join(1.0)
    assert batch_errors == close_errors == []
    assert [call[0] for call in channel.calls[-3:]] == ["stop", "play", "stop"]
    assert not channel.playing
```

Exact RED node: `tests/test_audio.py::test_event_batch_linearizes_before_close_and_ends_silent`. It fails if the lock covers one action rather than the tuple or if close lacks the basic successful cleanup path. Minimal GREEN holds the process lock through the full batch and adds successful initialized close; failure-retaining close state waits for Step 6.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py::test_event_batch_linearizes_before_close_and_ends_silent -v
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v -k "execute_order or playback_failure or preview or linearization or reference"
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v
```

Expected RED: the exact node, targeted subset, and current complete file report one or more newly appended `FAILED` behaviors. Implement exact whole-tuple execution, preview normalization/order, reference commits, process-lock coverage, operation/side/issue translation, and successful close needed by the linearization test; rerun all commands GREEN. Do not add partial-close retry or the real pygame loader in this step.

- [ ] **Step 6: RED -> GREEN for close, partial cleanup, and reuse**

Append every close/reuse case before editing close: unused close; known-uninitialized mixer; exact `get_init` failure envelope/state retention; best-effort three-channel order; completed-channel marks; failed-handle/ref/owner retention; partial-close execute/preview gates; retry only pending channels; first-error rule; non-owner isolation; full-success release; idempotence; same-port re-reservation; and blocked batch/preview versus close ending silent.

```python
def test_partial_close_retries_only_failed_channel_then_releases(fake_audio):
    from sf6_music_trigger.audio import (
        AudioAction,
        AudioIssue,
        AudioKind,
        AudioPlaybackError,
    )

    fake_audio.play_all_three_channels()
    fake_audio.reset_calls()
    fake_audio.mixer.channels[1].stop_failures_remaining = 1
    with pytest.raises(AudioPlaybackError) as first:
        fake_audio.port.close()
    assert (first.value.operation, first.value.side, first.value.issue) == (
        "close_stop",
        Side.P2,
        AudioIssue.BACKEND_FAILURE,
    )
    assert [fake_audio.mixer.channels[i].stop_calls for i in range(3)] == [1, 1, 1]
    with pytest.raises(AudioPlaybackError) as gated:
        fake_audio.port.execute((AudioAction(AudioKind.STOP, Side.P1),))
    assert (gated.value.operation, gated.value.issue) == (
        "execute",
        AudioIssue.CLOSE_INCOMPLETE,
    )
    fake_audio.port.close()
    assert [fake_audio.mixer.channels[i].stop_calls for i in range(3)] == [1, 2, 1]
    replacement = fake_audio.new_port()
    replacement.execute((AudioAction(AudioKind.STOP, Side.P1),))
```

`probe.play_all_three_channels()` uses real port calls to play P1, P2, and preview, then `reset_calls()` zeros only counters—not playing/ref/owner state. `FakeChannel.stop_calls` is a numeric property derived from its log. Exact RED node: `tests/test_audio.py::test_partial_close_retries_only_failed_channel_then_releases`. Before the partial-close state machine, it fails by clearing ownership/handles or double-stopping successful channels. Minimal GREEN retains the failed P2 state/owner, gates playback, retries only P2, then releases after complete success.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py::test_partial_close_retries_only_failed_channel_then_releases -v
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v -k "close or cleanup or reuse"
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v
```

Expected RED: the exact node, targeted subset, and current complete file report newly appended partial-cleanup/retry `FAILED` tests. Implement the exact close state machine without global quit/stop, rerun all commands GREEN, and prove a failed `mixer.get_init()` can be retried without losing ownership/state.

- [ ] **Step 7: RED -> GREEN for the real lazy loader and SDL dummy smoke**

Only now append the constructor-default test and SDL-dummy subprocess test. Import safety was already added and made GREEN in Step 3; do not duplicate or expand it here. Before adding the real default loader, constructing `PygameAudioPort(root)` without the injected fake must be a collected `FAILED` test. The subprocess sets `SDL_AUDIODRIVER=dummy` and `PYGAME_HIDE_SUPPORT_PROMPT=1` before import, creates a 100 ms temporary WAV, executes P1 and P2 batches plus preview, closes/reuses/closes, and checks no repository asset.

```python
def test_sdl_dummy_default_loader_roundtrip(tmp_path):
    from importlib.metadata import version
    import os
    from pathlib import Path
    from packaging.specifiers import SpecifierSet
    import subprocess
    import sys
    import wave

    installed = version("pygame")
    assert installed in SpecifierSet(">=2.5,<3")
    wav_path = tmp_path / "silence.wav"
    with wave.open(str(wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44_100)
        stream.writeframes(b"\x00\x00" * 4_410)
    script = r"""
import sys
from sf6_music_trigger.audio import AudioAction, AudioKind, PygameAudioPort
from sf6_music_trigger.domain import Side
root, track = sys.argv[1], sys.argv[2]
port = PygameAudioPort(root)
port.execute((
    AudioAction(AudioKind.STOP, Side.P1),
    AudioAction(AudioKind.PLAY, Side.P1, track, 0.18),
))
port.preview(track, 0.25)
port.close()
port.execute((
    AudioAction(AudioKind.STOP, Side.P2),
    AudioAction(AudioKind.PLAY, Side.P2, track, 0.18),
))
port.close()
"""
    env = os.environ.copy()
    env["SDL_AUDIODRIVER"] = "dummy"
    env["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(wav_path)],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
```

Exact RED node: `tests/test_audio.py::test_sdl_dummy_default_loader_roundtrip`. Before the default loader/signature is added it fails inside the child at `PygameAudioPort(root)`; missing/out-of-range pygame is reported as an environment block before subprocess execution, never skipped. Minimal GREEN adds only lazy `_load_pygame` and the constructor default, after which the 4,410 mono 16-bit frames are exactly 100 ms at 44.1 kHz.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py::test_sdl_dummy_default_loader_roundtrip -v
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v -k "default_loader or sdl_dummy"
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v
```

Expected RED: the exact node, targeted subset, and current complete file report the absent default-loader contract; no skip. Add only the default loader whose function body calls `importlib.import_module("pygame")` and its lazy constructor default. Rerun all commands GREEN with the installed dependency satisfying `>=2.5,<3` and the dummy driver; never touch a real device.

- [ ] **Step 8: Run focused/full verification including the mandatory dummy driver**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audio.py -v
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
```

Expected: the installed pygame version is programmatically inside `>=2.5,<3` and the SDL-dummy subprocess runs rather than skips; focused and full suites PASS; no real device, SF6, MK1, GUI, network, or Computer Use is started; and the generated WAV exists only in pytest's temporary directory.

- [ ] **Step 9: Execute and restore the Task 7 mutation controls**

Apply one mutation at a time, run the named control, observe failure, and restore immediately:

| Mutation | Required failing control |
|---|---|
| remove `side` from `AudioAction` or return it from `as_tuple` | typed-side / tuple-shape tests |
| accept strings for enum/side, accept bool/non-finite, or float-convert huge Rational before exact bounds | strict/huge/Fraction value tests |
| give STOP a payload or let PLAY accept empty track | action invariant tests |
| deduplicate event IDs or add pending state | repeated-event / no-state tests |
| check config before event-side equality and let disabled mismatch pass | scheduler side-mismatch test |
| return PLAY without preceding STOP or reverse the order | exact policy-order test |
| import pygame at module load/construction/unused close | blocked-import subprocess / loader-count tests |
| use `find_channel`, `Sound.play`, or any channel other than 0/1/2 | fake call-graph tests |
| skip post-`set_num_channels` count verification | still-too-small initialization test |
| publish partial channels, leak provisional reservation on initialize failure, or make retry fail | partial-state / rollback/reacquisition tests |
| normalize an external absolute string | drive/UNC preservation tests |
| treat drive/root/device/ADS paths as ordinary relative paths or surface path Type/Value to runtime callers | Windows classification / TRACK_REJECTED tests |
| omit call-time resolved containment, cache it at construction, or use string-prefix containment | post-construction junction and sibling-prefix tests |
| add an extension/existence filter | unusual-extension/nonexistent pass-through test |
| clamp PLAY instead of rejecting it, or fail to clamp preview | strict-action / preview-boundary tests |
| hide a STOP inside PLAY or reorder Sound/set/play | exact command-order tests |
| commit a sound reference before play succeeds | play-failure reference test |
| mislabel preview stop/load/set/play, skip huge-value clamp, or touch either side channel/reference | preview mapping/volume/isolation tests |
| retain track/raw OS/backend text in error fields, repr, vars, args, cause, or context | backend/path error-privacy tests |
| clear a ref/handle after STOP or preview-stop failure | retryable-stop reference tests |
| stop after the first close failure, clear a failed channel, clear state on get_init failure, double-stop a successful channel on retry, or release ownership early | partial-close/get-init retry tests |
| call a global pygame/mixer stop/quit | forbidden-global-call tests |
| retain handles/refs after fully successful close or make the port one-shot | idempotent-close / same-port-reuse tests |
| accept an empty execute tuple, execute actions one-by-one, use an instance lock, acquire the lock after preflight, or omit active-port reservation | nonempty/blocked-preflight/batch/competing-port tests |

After restoring every mutation, rerun the Step 8 commands. No mutation artifact may remain in `git diff`.

- [ ] **Step 10: Verify the exact implementation scope and commit**

This gate is self-contained and re-reads the frozen hash; it does not depend on a variable from Step 0. Intent-to-add makes the two new files visible to `git diff --check` and the base diff:

```powershell
$basePath = '.superpowers/sdd/task7-plan-base.txt'
if (-not (Test-Path -LiteralPath $basePath)) {
    throw 'Frozen Task 7 plan-base evidence is missing'
}
$baseRaw = [IO.File]::ReadAllText($basePath)
if (-not [regex]::IsMatch($baseRaw, '\A[0-9a-f]{40}\n\z')) {
    throw 'Frozen Task 7 plan base must be a full lowercase hash plus LF'
}
$base = $baseRaw.Substring(0, 40)
$head = (git rev-parse HEAD).Trim()
if ($head -ne $base) {
    throw 'HEAD moved after the Task 7 plan base was frozen'
}
git check-ignore -q $basePath
if ($LASTEXITCODE -ne 0) { throw 'Task 7 plan-base evidence is not ignored' }
git add -N -- src/sf6_music_trigger/audio.py tests/test_audio.py
git diff --check
$tracked = @(git diff --name-only $base)
$untracked = @(git ls-files --others --exclude-standard)
$actual = @($tracked + $untracked | Sort-Object -Unique)
$expected = @(
    'src/sf6_music_trigger/audio.py'
    'tests/test_audio.py'
) | Sort-Object -Unique
if (Compare-Object -ReferenceObject $expected -DifferenceObject $actual) {
    throw 'Task 7 tracked/untracked scope escaped the exact two-file allowlist'
}
git status --short
```

Verify the diff contains no config/domain/engine/panel changes, pygame global shutdown, personal paths, logs, audio files, device access, Task 8/9 implementation, or outcome/block/whiff/fade/cache behavior. Then stage and commit exactly:

```powershell
git add src/sf6_music_trigger/audio.py tests/test_audio.py
git diff --cached --check
$staged = @(git diff --cached --name-only | Sort-Object -Unique)
$expected = @(
    'src/sf6_music_trigger/audio.py'
    'tests/test_audio.py'
) | Sort-Object -Unique
if (Compare-Object -ReferenceObject $expected -DifferenceObject $staged) {
    throw 'Task 7 staged diff escaped the exact two-file allowlist'
}
git commit -m "feat: play confirmed super audio immediately"
```

---

### Task 8: Frame-to-Audio Engine with Independent P1/P2 Trackers

**Files:**
- Create: `src/sf6_music_trigger/engine.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- Produces: frozen `EngineUpdate`, `profile_applies_to(profile, geometry, *, max_relative_aspect_error=0.01)`, `VisualTriggerEngine.process(sample: CaptureSample) -> EngineUpdate`, and `VisualTriggerEngine.reset(reason)`.
- Consumes: Task 6 `CaptureSample` and `FreshFrameGate`; Task 3 profile/side ROIs and separately loaded P1/P2 references; Task 4 `observe_gauge`; Task 5 trackers returning event tuples; distinct P1/P2 `AudioScheduler` values and `SideAudioConfig` values; one Task 7 `AudioPort`; an injected non-throwing `audio_diagnostic: Callable[[AudioPlaybackError], None]`; and an injected monotonic `clock: Callable[[], Real]`.

- [ ] **Step 1: Write RED integration and wiring tests without mocking trackers**

```python
from dataclasses import FrozenInstanceError

import pytest

from sf6_music_trigger.audio import AudioIssue, AudioKind
from sf6_music_trigger.domain import CaptureSample, CaptureStatus, Side
from sf6_music_trigger.engine import EngineUpdate


def running(frame):
    return CaptureSample(status=CaptureStatus.RUNNING, frame=frame)


def test_p1_full_to_spent_replay_emits_one_event_and_stop_play(trace_factory):
    engine, audio = trace_factory(enabled_sides=("P1",))
    for frame in trace_factory.full_frames("P1", count=5):
        engine.process(running(frame))
    for frame in trace_factory.spent_frames("P1", count=3, step=0.06):
        update = engine.process(running(frame))
        assert isinstance(update.events, tuple)
    assert [action.kind for action in audio.actions] == [
        AudioKind.STOP,
        AudioKind.PLAY,
    ]
    assert audio.batch_sides == [Side.P1]
    assert len(engine.event_log) == 1


def test_engine_update_is_minimal_frozen_event_tuple():
    update = EngineUpdate()
    assert update.events == ()
    with pytest.raises(TypeError):
        EngineUpdate(events=[])
    with pytest.raises(TypeError):
        EngineUpdate(events=(object(),))
    with pytest.raises(FrozenInstanceError):
        update.events = ()


def test_each_frame_wires_distinct_side_roi_and_references(engine_fixture, observer_spy):
    engine, frame = engine_fixture.with_observer(observer_spy)
    engine.process(running(frame))
    assert len(observer_spy.frame_args) == 2
    assert observer_spy.frame_args[0] is frame.image
    assert observer_spy.frame_args[1] is frame.image
    assert observer_spy.calls == [
        {
            "side": Side.P1,
            "frame_id": frame.frame_id,
            "roi": engine_fixture.profile.p1.meter_roi,
            "references": engine_fixture.p1_references,
            "sampled_at": frame.monotonic_time,
        },
        {
            "side": Side.P2,
            "frame_id": frame.frame_id,
            "roi": engine_fixture.profile.p2.meter_roi,
            "references": engine_fixture.p2_references,
            "sampled_at": frame.monotonic_time,
        },
    ]
    assert observer_spy.calls[0]["roi"] != observer_spy.calls[1]["roi"]


def test_p2_different_roi_ignores_p1_distractor(trace_factory):
    engine, audio = trace_factory(enabled_sides=("P1", "P2"), different_side_rois=True)
    trace_factory.drive_p2_spend_with_p1_distractor(engine)
    assert {action.side for action in audio.actions} == {Side.P2}


@pytest.mark.parametrize(
    "issue",
    [AudioIssue.BACKEND_FAILURE, AudioIssue.TRACK_REJECTED],
)
def test_audio_failure_reports_diagnostic_without_losing_events_or_p2(
    simultaneous_spend_factory, issue
):
    engine, audio, diagnostics = simultaneous_spend_factory(
        fail_batch_for=Side.P1,
        issue=issue,
    )
    update = simultaneous_spend_factory.drive_confirmation(engine)
    assert [event.side for event in update.events] == [Side.P1, Side.P2]
    assert [event.side for event in engine.event_log] == [Side.P1, Side.P2]
    assert audio.attempted_batch_sides == [Side.P1, Side.P2]
    assert [error.side for error in diagnostics] == [Side.P1]
    assert diagnostics[0].issue is issue


def test_disabled_scheduler_derives_once_and_never_calls_nonempty_port(
    disabled_audio_spend_factory,
):
    engine, scheduler, audio = disabled_audio_spend_factory()
    update = disabled_audio_spend_factory.drive_confirmation(engine)
    assert len(update.events) == 1
    assert scheduler.calls == 1
    assert audio.batches == []
```

Retain RED cases for simultaneous HUD loss, P1 event isolation, P2 event isolation, consecutive events, stale capture, every legal non-running `CaptureSample`, runtime geometry mismatch resetting both trackers, and the just-below/exact/just-above 1% aspect boundary. Every synthetic running sample uses an owned writable C-contiguous BGR array whose shape exactly matches its `ClientGeometry`. Add wiring failures for swapped references, reused ROI, wrong P2 side, missing a side call, passing the `CapturedFrame` envelope instead of `frame.image`, and non-tuple tracker output. Add focused controls proving that the injected clock supplies `now`, a rejected/non-running sample resets both trackers without clearing freshness watermarks, and only explicit `reset(reason)` starts a new freshness epoch. Every `process` path returns a frozen `EngineUpdate` whose `events` is a tuple.

Task 8—not the Task 7 commit—adds audio seam tests for exact event/config routing, exactly one scheduler derivation per confirmed event, no port call when that derived tuple is `()`, exactly one actual tuple call otherwise, P1-before-P2 batch order, repeated events producing distinct repeated batches, `TRACK_REJECTED` playback diagnostics, and a P1 batch failure not blocking P2. The recording port's `execute` must reject/record the same nonempty tuple contract as Task 7 rather than exposing an action-at-a-time shortcut.

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_engine.py -v
```

Expected: collection ERROR because `VisualTriggerEngine`/`EngineUpdate` is missing.

- [ ] **Step 3: Implement minimal typed orchestration**

```python
@dataclass(frozen=True)
class EngineUpdate:
    events: tuple[ConfirmedSuperEvent, ...] = ()


def profile_applies_to(
    profile: HudProfile,
    geometry: ClientGeometry,
    *,
    max_relative_aspect_error: float = 0.01,
) -> bool: ...


class VisualTriggerEngine:
    def process(
        self,
        sample: CaptureSample,
    ) -> EngineUpdate: ...

    def reset(self, reason: str) -> None: ...
```

`EngineUpdate` has no additional Phase 1 field. It validates `events` is an actual tuple containing only `ConfirmedSuperEvent` values and does not coerce lists.

The engine constructor receives a `FreshFrameGate` and a keyword-only monotonic `clock: Callable[[], Real]` whose default is `time.monotonic`. It stores the callable without invoking it during construction. `process` requires an actual `CaptureSample`; only a `RUNNING` sample can carry a frame, as enforced by Task 6.

Keep the approved geometry predicate: typed profile/geometry, finite nonnegative real non-boolean tolerance, relative aspect error with an inclusive 1% boundary using absolute tolerance `1e-12`; same aspect at another resolution applies.

For a `RUNNING` sample, bind `frame = sample.frame`, call `freshness_gate.accept(frame, now=clock())` exactly once, and do not call the clock for a non-running sample. Task 6 guarantees this local frame is not `None`. Only when the gate accepts it, invoke Task 4 exactly once per side in deterministic P1-then-P2 order:

```python
p1_observation = observe_gauge(
    frame.image,
    side=Side.P1,
    frame_id=frame.frame_id,
    roi=profile.p1.meter_roi,
    references=p1_references,
    sampled_at=frame.monotonic_time,
)
p2_observation = observe_gauge(
    frame.image,
    side=Side.P2,
    frame_id=frame.frame_id,
    roi=profile.p2.meter_roi,
    references=p2_references,
    sampled_at=frame.monotonic_time,
)
```

P1 and P2 references must be distinct `MeterReferences` objects with matching embedded sides. Pass each observation only to its matching tracker. Concatenate tracker event tuples in P1-then-P2 order and append those immutable events to the log before playback so audio cannot roll them back. For each event in that order, select only its matching scheduler/config and call `on_confirmed_spend(event, config)` exactly once. When the derived tuple is `()`, do not call the port; otherwise call `audio_port.execute(actions)` exactly once for that whole nonempty event batch. Catch `AudioPlaybackError`—including `TRACK_REJECTED`—around each event batch, pass that safe error to the injected error sink, do not retry or rethrow it from `process`, and continue to the next event/side. The error sink must be non-throwing; Task 9 implements it as a safe status-queue insertion. Return `EngineUpdate(events=events)` regardless of contained audio failures.

`EngineUpdate` gains no audio/action/diagnostic field. Scheduler type/side/config invariant failures are programmer/start-validation errors and are not disguised as playback failures. Task 8 owns this integration and `tests/test_engine.py` coverage; `engine.py` remains outside the Task 7 implementation commit.

A non-`RUNNING` sample, freshness rejection, inapplicable/changing geometry, or both HUDs hidden resets both trackers and returns `EngineUpdate()`; cached observations cannot be reused. These runtime fail-closed resets preserve `FreshFrameGate` watermarks, so an old/repeated frame cannot become fresh merely because capture was briefly lost. Only explicit `VisualTriggerEngine.reset(reason)` resets both trackers and calls `FreshFrameGate.reset()`, creating a new source/session epoch. The engine has no input mapping or outcome classifier.

- [ ] **Step 4: Run focused/full tests and mutation controls**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_engine.py -v
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
```

Expected: PASS. Temporarily swap P2 ROI/reference wiring and verify the dedicated tests fail; change the three-spent trace step from `0.06` to a value whose first-to-confirming span is below `0.10` and verify the integration test fails; clear freshness watermarks on a non-running sample and verify the retained-watermark test fails; mutate `config.spent_samples` or `config.evidence_window` and verify integration noise controls fail. Also mutate event playback into per-action calls, reverse P1/P2 batch order, omit the diagnostic sink, rethrow a contained `AudioPlaybackError`, remove the event from `event_log` on failure, and stop before the second side; each dedicated audio-seam test must fail. Restore every mutation without persisting.

- [ ] **Step 5: Commit**

```powershell
git add src/sf6_music_trigger/engine.py tests/test_engine.py
git commit -m "feat: orchestrate screen-only super events"
```

---



### Task 9: Cooperative Controller, Panel View Model, and Calibration Flow

**Files:**
- Create: `src/sf6_music_trigger/controller.py`
- Create: `src/sf6_music_trigger/panel.py`
- Create: `src/sf6_music_trigger/__main__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_controller.py`
- Create: `tests/test_panel_view_model.py`

**Interfaces:**
- Produces: `ListenerController`, `ControllerStatus`, `PanelViewModel`, `run_panel()`.
- Consumes: Task 6 `FrameSource`/`CaptureSample`/`FreshFrameGate`, capture factory, engine factory, a separate calibration-gate factory and monotonic calibration clock, the same reusable Task 7 `AudioPort` for engine playback and UI preview, config persistence, Task 3 `CalibrationContext`, `HudProfile`, `SideHudProfile`, `build_meter_references`, containment-aware profile/reference persistence, and Task 8 `VisualTriggerEngine.process(sample)`/`reset(reason)`/`profile_applies_to`.

- [ ] **Step 1: Write RED lifecycle and view-model tests**

```python
def test_start_is_single_worker_and_stop_is_idempotent(worker_probe):
    controller = worker_probe.controller()
    assert controller.start()
    assert not controller.start()
    assert controller.request_stop()
    assert controller.request_stop()
    controller.join(timeout=1.0)
    assert worker_probe.maximum_concurrent == 1


def test_exit_stops_capture_and_all_audio_even_after_worker_failure(failing_dependencies):
    controller = failing_dependencies.controller()
    controller.start()
    controller.join(timeout=1.0)
    assert failing_dependencies.capture.stopped
    assert failing_dependencies.audio.closed


def test_worker_routes_each_typed_sample_directly_to_engine(controller_probe):
    controller_probe.run_one_iteration()
    assert controller_probe.engine.samples == [controller_probe.source.sample]


def test_stop_calls_source_stop_to_wake_blocked_read(blocked_read_probe):
    controller = blocked_read_probe.controller()
    assert controller.start()
    blocked_read_probe.wait_until_read_is_blocked()
    assert controller.request_stop()
    controller.join(timeout=1.0)
    assert blocked_read_probe.source.stop_calls >= 1
    assert blocked_read_probe.worker_exited


def test_successful_source_start_resets_engine_before_first_read(controller_probe):
    controller_probe.start_and_run_one_iteration()
    assert controller_probe.calls[:4] == [
        "source.start",
        "engine.reset:capture session started",
        "source.read",
        "engine.process",
    ]


def test_stop_restart_accepts_new_source_frame_zero(restart_probe):
    restart_probe.run_session_with_frame_id_zero()
    restart_probe.stop()
    restart_probe.run_session_with_frame_id_zero()
    assert restart_probe.accepted_frame_ids == [0, 0]


def test_stop_restart_reuses_same_audio_port_after_successful_close(restart_probe):
    first_port = restart_probe.audio
    restart_probe.run_then_stop()
    restart_probe.run_then_stop()
    assert restart_probe.audio is first_port
    assert restart_probe.audio.close_calls == 2
    assert restart_probe.audio.successful_session_batches == [1, 1]


def test_preview_converts_percent_and_never_touches_side_batches(panel_audio_probe):
    panel_audio_probe.vm.p1_main_track = "audio/p1.mp3"
    panel_audio_probe.vm.p1_volume_percent = 35
    panel_audio_probe.vm.preview(Side.P1)
    assert panel_audio_probe.audio.preview_calls == [("audio/p1.mp3", 0.35)]
    assert panel_audio_probe.audio.side_batches == []


def test_config_profile_and_audio_share_one_canonical_user_root(root_probe):
    root_probe.write_config(main_track="audio/p1.wav")
    root_probe.start_and_play_p1()
    expected_root = root_probe.config_path.parent.resolve(strict=False)
    assert root_probe.load_config_root == expected_root
    assert root_probe.save_config_root == expected_root
    assert root_probe.profile_root == expected_root
    assert root_probe.audio_port_root == expected_root
    assert root_probe.sound_paths == [
        str((expected_root / "audio/p1.wav").resolve(strict=False))
    ]


def test_view_model_cannot_start_without_valid_profile_and_main_track(tmp_path):
    vm = PanelViewModel(user_dir=tmp_path)
    vm.profile_path = "profiles/missing.json"
    vm.p1_main_track = ""
    assert not vm.can_start
    assert "HUD" in vm.validation_message


def test_view_model_disables_start_when_current_geometry_needs_recalibration(
    valid_files, geometry_16_by_10
):
    vm = PanelViewModel(user_dir=valid_files.user_dir)
    valid_files.select_on(vm)
    vm.current_geometry = geometry_16_by_10
    assert not vm.can_start
    assert "校准" in vm.validation_message


def test_cancelled_or_hidden_calibration_writes_nothing(calibration_probe):
    calibration_probe.cancel_during_spent_capture()
    assert calibration_probe.persist_calls == []


def test_calibration_session_owns_and_resets_separate_freshness_gate(calibration_probe):
    calibration_probe.begin()
    assert calibration_probe.gate is not calibration_probe.engine_gate
    assert calibration_probe.gate.reset_calls == 1


def test_calibration_accepts_only_running_samples_accepted_with_injected_clock(
    calibration_probe,
):
    calibration_probe.feed_running_frame(frame_id=0, sampled_at=10.0, now=10.0)
    assert calibration_probe.accepted_frame_ids == [0]
    assert calibration_probe.gate.accept_calls == [(0, 10.0)]


def test_calibration_commits_profile_only_after_both_refs_reload(calibration_probe):
    calibration_probe.complete()
    assert calibration_probe.calls == [
        "save-p1-new",
        "save-p2-new",
        "reload-p1-as-p1",
        "reload-p2-as-p2",
        "save-profile-last",
    ]


@pytest.mark.parametrize(
    "failure_stage",
    [
        "save-p1",
        "save-p2",
        "reload-p1",
        "reload-p2",
        "profile-serialize",
        "profile-replace",
    ],
)
def test_calibration_stage_failure_preserves_old_profile_and_cleans_attempt_refs(
    calibration_probe, failure_stage
):
    old_profile_bytes, old_reference_bytes = calibration_probe.snapshot_old_calibration()
    calibration_probe.fail_at(failure_stage)
    with pytest.raises(OSError):
        calibration_probe.complete()
    assert calibration_probe.old_profile_bytes() == old_profile_bytes
    assert calibration_probe.old_reference_bytes() == old_reference_bytes
    assert calibration_probe.attempt_reference_paths() == []
    assert calibration_probe.unrelated_neighbor_untouched
```

- [ ] **Step 2: Run and verify RED**

Expected: imports fail because controller/panel modules do not exist.

- [ ] **Step 3: Implement lifecycle and testable view model**

`ListenerController` owns a non-daemon worker, stop event, lock, and status queue. It holds the lifecycle lock through thread start and supports idempotent stop. Each worker session performs this exact order: `source.start()`; after that call succeeds, `engine.reset("capture session started")`; only then may it call `sample = source.read()` followed by `engine.process(sample)`. A start failure does not reset the engine, and a reset failure performs normal source/audio cleanup without reading. This explicit new-session reset clears Task 8's freshness watermarks, so a stop followed by restart can accept the new source epoch's first `frame_id=0`; no capture-loss status inside one session clears those watermarks. `request_stop()` sets the stop event and calls the thread-safe, idempotent `source.stop()` so an in-flight read wakes.

Final worker cleanup again stops capture and calls `audio.close()` in `finally`, including worker failures. A normal successful close releases Task 7's reservation; restart injects the exact same port instance, whose next event lazily reserves/reinitializes it. The final Exit path joins the worker and calls the same idempotent `audio.close()` again, which also retries a Task 7 partial close. Catch `AudioPlaybackError` from cleanup and enqueue only its safe operation/side/issue; do not interpolate `repr` of dependencies. A permanently blocking pygame backend remains Task 7's documented unbounded backend limitation, so the controller must not claim a hard shutdown deadline or call global pygame quit to force it.

`PanelViewModel` is independent of Tk widgets and exposes validated fields, percentage strings, capture/tracker states, calibration context, current client geometry, and command callbacks. The Tk layer binds Start/Stop/Exit, P1/P2 track pickers, volume sliders, preview buttons, profile picker, “校准 HUD”, and a diagnostics opt-in checkbox. Start is disabled until at least one side has a valid main track and the selected strict profile plus both separately sided reference files load through Task 3 APIs. Missing files or any path/schema/version/shape/dtype/side failure keeps Start disabled rather than falling back to guesses. Task 8 owns the predicate and runtime reset behavior; when current client geometry is known, the panel reuses Task 8 `profile_applies_to` and disables Start with a recalibration message on mismatch rather than duplicating its math.

The UI is the only percentage bridge: validate a finite non-boolean slider value in `0..100`, divide by `100.0` for `SideAudioConfig.volume` and `audio.preview(track, normalized_volume)`, and display the effective integer percentage. A preview button calls only the shared port's `preview` with the selected side's track; it never constructs a STOP/PLAY batch, changes P1/P2 config, or touches engine event state. Task 7's process lock makes preview mutually exclusive with a whole event batch while its dedicated Channel 2 remains independent of P1/P2. Preview errors feed the same path-safe status sink.

Task 9 derives exactly one canonical user-data root as `config_path.parent.resolve(strict=False)`. Pass that same `Path` to config load/save ownership, Task 3 profile/reference resolution, panel validation, and the sole `PygameAudioPort`; no component independently chooses CWD, the repository root, or a second user directory. A relative `main_track` is validated against this root when config loads/saves and Task 7 resolves it against the identical root at playback. The root-probe test above must observe the final `mixer.Sound` string under that root.

Only after `sf6_music_trigger.__main__:main` exists and its import/help behavior is covered, add `[project.scripts] sf6-music-trigger = "sf6_music_trigger.__main__:main"` to `pyproject.toml`. Reinstall the editable package and smoke-test the generated console command so the release-hygiene wheel audit can no longer pass a missing target.

Calibration is a two-step local wizard for both P1 and P2; audio enablement controls playback, not profile completeness or reference ownership. The calibration flow receives an injected `calibration_gate_factory: Callable[[], FreshFrameGate]` and `calibration_clock: Callable[[], Real]`. Every explicit calibration session obtains a new dedicated gate, proves it is not the engine's gate or a prior session's gate, and calls `gate.reset()` exactly once before its first source read. It accepts pixels only from `CaptureSample(status=CaptureStatus.RUNNING, ...)` after exactly one `gate.accept(sample.frame, now=calibration_clock())`; it never calls the clock/gate for a non-running sample. A gate rejection for stale age, future skew, duplicate/backward frame ID, or duplicate/backward timestamp discards the current in-memory batch and does not advance accepted calibration state. Watermarks remain intact for the rest of that session; cancel/restart begins a new calibration session and resets the new gate, allowing the restarted source's `frame_id=0`.

From a freshness-gate-accepted running frame, require the user to confirm the P1/P2 meter ROIs and required battle ROI, optionally confirm a training-HUD ROI, record `client_width`, `client_height`, and their exact aspect ratio, and collect the HUD scale setting, language, brightness setting, HDR boolean, and optional game-version note into `CalibrationContext`.

For each side capture exactly five stable full-meter crops, prompt the user to spend SA3/CA, then capture exactly five stable spent-meter crops. A stable batch means five consecutive freshness-gate-accepted `RUNNING` frames with strictly increasing frame IDs, identical client width/height, visible preprocessed meter samples, and maximum pairwise normalized mean absolute distance `<= 0.03` across the five canonical images. A stale/non-running/hidden sample, geometry change, or distance violation discards that in-memory batch and restarts it; it is never silently skipped to assemble five nonconsecutive samples.

Keep every crop and derived object in memory until both sides and all profile fields validate. Call `build_meter_references(Side.P1, ...)` and `build_meter_references(Side.P2, ...)` separately. Then perform this ordered calibration transaction:

1. Generate a fresh attempt ID and two distinct, previously nonexistent contained paths such as `refs/<profile-id>.<attempt-id>.p1.npz` and `.p2.npz`; never overwrite the files referenced by the currently committed profile.
2. Atomically `save_references` for P1, then P2.
3. Reload each new archive with `load_references(..., expected_side=Side.P1/P2)` and verify its arrays, versions, side, and shape equal the in-memory reference value.
4. Construct `HudProfile` with the required battle ROI, optional training-HUD ROI, full calibration context, side ROIs, and only those verified new relative paths.
5. Atomically `save_profile` last. Its successful `os.replace` is the sole commit point after which the new calibration is active.

Until the profile commit succeeds, any cancellation or exception cleans up only the exact new reference paths created and recorded by this attempt; never glob or remove old references, the old profile, neighboring files, or another attempt's files. If P1 save succeeds and any later stage fails—including P2 save, either reload/validation, serialization, or profile replace—the old profile and its referenced calibration remain byte-for-byte usable. P1/P2 must never share a reference object, ndarray, path, tracker, scheduler, channel, or cooldown. Phase 1 may retain superseded old reference files after a successful commit; unrelated garbage collection is out of scope.

Add view-model/calibration cases for full valid context, required battle ROI and optional training-HUD ROI, missing/invalid/wrong-side references, incomplete P1/P2 calibration, context/ROI display, every measurable stable-batch rejection, zero-write early cancellation, and separate P1/P2 build/save calls. The freshness matrix must independently cover exact-age/future boundaries, just-stale, just-future, duplicate/backward ID, duplicate/backward time, non-running samples without clock calls, rejected-watermark atomicity, isolation from the engine gate, and cancel/restart accepting a new `frame_id=0`. Inject failures at P1 save, P2 save, each expected-side reload, profile serialization, and profile replace; assert the call order, attempt-only cleanup, no temp leak, and preservation of old/neighbor bytes.

Audio seam tests in Task 9 cover the same port object across stop/restart, worker-finally close plus final-Exit retry, privacy-safe cleanup diagnostics, exact `0/100` percentage boundaries, non-real/bool/out-of-range slider rejection, preview path/normalized volume routing, preview not mutating side batches/config/event state, and `TRACK_REJECTED` preview failure becoming a safe status without stopping the worker. Root tests cover relative-main-track config validation and the exact final Sound path under the one canonical config-parent root. Mock ports and persistence functions; do not create Tk roots, initialize DXcam/pygame, launch SF6, or use Computer Use in tests.

- [ ] **Step 4: Run lifecycle/view-model tests and headless import smoke**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_controller.py tests/test_panel_view_model.py -v
.\.venv\Scripts\python.exe -m pytest -v
git diff --check
```

Expected: tests PASS; importing `sf6_music_trigger.panel` does not create a Tk root or initialize DXcam/pygame. Apply each mutation separately and restore immediately: omit `engine.reset("capture session started")` and require stop/restart frame zero to fail; move reset after first read and require the call-order test to fail; replace the reusable port on restart, omit worker-finally/final-Exit close, expose raw cleanup text, pass percentage without dividing by 100, or route preview through side playback and require the dedicated audio seam test to fail; share the engine/calibration gate and require isolation to fail; omit calibration-session reset and require restart frame zero to fail; bypass `gate.accept` or its injected clock and require stale/future/order tests to fail; advance calibration state on a rejected sample and require rejected-watermark/batch tests to fail.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml src/sf6_music_trigger/controller.py src/sf6_music_trigger/panel.py src/sf6_music_trigger/__main__.py tests/test_controller.py tests/test_panel_view_model.py
git commit -m "feat: add persistent control panel and calibration"
```

---

### Task 10: Offline Replay CLI and Phase-1 Verification

**Files:**
- Create: `tools/replay.py`
- Create: `tests/test_replay_integration.py`
- Create: `docs/verification/phase1-offline-checklist.md`
- Modify: `README.md`

**Interfaces:**
- Produces: `python tools/replay.py --profile <path> --frames <folder> --side P1|P2|both --json <result>`.
- Consumes: public engine interfaces and local, user-provided frame folders.

- [ ] **Step 1: Write RED end-to-end replay test**

```python
def test_replay_cli_reports_exactly_one_confirmed_event(tmp_path, synthetic_replay):
    profile, frames = synthetic_replay.write_to(tmp_path)
    result_path = tmp_path / "result.json"
    completed = run_cli(profile, frames, result_path)
    assert completed.returncode == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["events"] == [{"side": "P1", "pre_spend_frame_id": 5, "confirmed_frame_id": 8}]
    assert result["rejected_stale_frames"] == 0
```

Add controls for no-spend, full-to-one-bar, full-to-two-bars, single-frame noise, hidden HUD, repeated frame IDs, P2, and two independent sides.

- [ ] **Step 2: Run and verify RED**

Expected: FAIL because `tools/replay.py` does not exist.

- [ ] **Step 3: Implement replay CLI and documentation**

The CLI loads sequential `.npy` or `.png` frames from a local path, uses filename order as frame ID, applies a caller-supplied FPS timestamp, runs the real engine with a recording audio port, and writes only numeric JSON. It refuses paths inside the source tree’s ignored copyrighted directories and never copies input frames.

The verification checklist must record exact commands for lint-free compilation, full pytest, SDL dummy smoke, synthetic replay, old-project read-only replay, and the pending Computer Use live gate. README must state Phase-1 limitations and safe online boundary.

The two literal Task 3 preprocessing vectors are release compatibility gates for `PREPROCESSING_VERSION = 1`. Record the exact `numpy.__version__` and `cv2.__version__` used by verification. Before publishing a release, lock the verified NumPy/OpenCV versions in the release environment; if a dependency upgrade changes either literal output, increment `PREPROCESSING_VERSION`, require recalibration, and update the golden vectors. Never ship different canonical reference bytes under the same preprocessing version.

- [ ] **Step 4: Run fresh verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tools tests
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -c "import cv2, numpy; print(f'numpy={numpy.__version__} opencv={cv2.__version__}')"
```

Then run the replay CLI against synthetic P1/P2 traces. Expected: compile exit `0`, complete suite PASS, each true spend produces exactly one `stop -> play`, and all no-spend/stale/reset traces produce zero events.

- [ ] **Step 5: Request independent review**

Reviewer must inspect: no input dependency, no game-process access, no copyrighted assets, P1/P2 isolation, fresh-frame enforcement, lifecycle cleanup, configuration privacy, and proof that tests fail under one-sample confirmation mutations.

- [ ] **Step 6: Commit**

```powershell
git add tools/replay.py tests/test_replay_integration.py docs/verification/phase1-offline-checklist.md README.md
git commit -m "test: verify phase one screen-only trigger"
```

---

## Plan Self-Review Checklist

- [x] Every Phase-1 design requirement maps to a task above.
- [x] Every production module has at least one RED-first behavioral test.
- [x] No task requires SF6, MK1, Computer Use, administrator rights, game memory, control input, cloud APIs, commercial audio, or committed game imagery.
- [x] P1/P2 isolation is tested at tracker, audio, engine, and replay levels.
- [x] Capture loss and repeated frames cannot confirm spend.
- [x] The plan does not claim outcome classification or live acceptance.
- [x] All referenced interfaces and default values are defined consistently.
