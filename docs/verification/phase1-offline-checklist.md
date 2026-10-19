# Phase-1 Offline Verification Checklist

Status: **offline candidate**. The Computer Use live gate has not been run and
this document does not claim real-machine acceptance.

Run every command from the repository root in PowerShell with the repository's
Python 3.12 virtual environment. Use separate writable temporary directories if
the host `%TEMP%` has restrictive permissions.

## 1. Compilation and complete regression

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tools tests
.\.venv\Scripts\python.exe -m pytest -v
```

Expected: both commands exit `0`; pytest has no failures or skips.

## 2. Preprocessing compatibility gate

```powershell
.\.venv\Scripts\python.exe -c "import cv2, numpy; print(f'numpy={numpy.__version__} opencv={cv2.__version__}')"
.\.venv\Scripts\python.exe -m pytest `
  tests/test_meter_references.py::test_preprocess_upscale_matches_literal_bgr_golden_vector `
  tests/test_meter_references.py::test_preprocess_downscale_matches_literal_bgr_golden_vector `
  -v
```

Verified offline environment on 2026-08-10:

- NumPy `2.5.1`
- OpenCV `4.14.0`
- `PREPROCESSING_VERSION = 1`

The two literal Task 3 vectors are release compatibility gates. Before a
release, pin the verified NumPy/OpenCV versions in that release environment. If
an upgrade changes either literal output, increment `PREPROCESSING_VERSION`,
require recalibration, and update the vectors. Never publish different
canonical reference bytes under the same preprocessing version.

## 3. Device-free audio smoke

```powershell
$env:SDL_AUDIODRIVER = "dummy"
$env:SDL_VIDEODRIVER = "dummy"
.\.venv\Scripts\python.exe -m pytest `
  tests/test_audio.py::test_sdl_dummy_default_loader_roundtrip `
  -v
```

Expected: exit `0`. This is an SDL dummy-backend check, not evidence that a real
audio device or game was exercised.

## 4. Synthetic replay matrix

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_replay_integration.py -v
```

Expected:

- each true P1/P2 spend produces one event and exactly one `stop -> play` pair;
- no-spend, one-bar, two-bar, single-frame-noise, hidden-HUD reset, and repeated
  frame-ID traces produce zero events;
- repeated frame IDs increment `rejected_stale_frames`;
- `.npy` and `.png` inputs remain byte-for-byte unchanged and are never copied.

For a caller-provided local sequence, filenames must begin with a fixed-width
numeric frame ID and use `.npy` or `.png`:

```powershell
$profile = (Resolve-Path $env:SF6_REPLAY_PROFILE).Path
$frames = (Resolve-Path $env:SF6_REPLAY_FRAMES).Path
$result = Join-Path $env:TEMP "sf6-phase1-replay.json"
.\.venv\Scripts\python.exe tools\replay.py `
  --profile $profile --frames $frames --side both --fps 20 --json $result
Get-Content -Raw $result
```

## 5. Old-project read-only replay

This optional offline comparison may read an explicitly selected old-project
frame export, but it must not execute or modify the old prototype. Set the two
environment variables to a calibrated profile and an exported frame folder,
then run the exact read-only hash guard below:

```powershell
$oldRoot = (Resolve-Path $env:SF6_OLD_PROJECT_ROOT).Path
$oldMain = Join-Path $oldRoot "main.py"
$before = (Get-FileHash -Algorithm SHA256 -LiteralPath $oldMain).Hash
$profile = (Resolve-Path $env:SF6_REPLAY_PROFILE).Path
$frames = (Resolve-Path $env:SF6_OLD_REPLAY_FRAMES).Path
$result = Join-Path $env:TEMP "sf6-old-project-read-only-replay.json"
.\.venv\Scripts\python.exe tools\replay.py `
  --profile $profile --frames $frames --side both --fps 20 --json $result
$after = (Get-FileHash -Algorithm SHA256 -LiteralPath $oldMain).Hash
if ($before -ne $after) { throw "old prototype main.py changed" }
Get-Content -Raw $result
```

Do not point `--json` into the old project. Do not commit or redistribute old
screens, music, captures, or personal configuration.

## 6. Pending Computer Use live gate

Stop after the offline checklist. Before any live action, report to the user:

1. the only target application to be launched is **Street Fighter 6**;
2. the purpose is final screen-capture, trigger, audio, lifecycle, and P1/P2
   acceptance;
3. the smallest planned interaction sequence and estimated duration;
4. Mortal Kombat 1 and every other application will not be launched or used.

Wait for explicit permission granting full Computer Use access. Do not request,
enable, or use Computer Use autonomously. Until the SF6-only live sequence has
actually passed, report the build only as an offline candidate.
