# SF6 Full-Resolution Viewport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize common letterboxed or pillarboxed SF6 client sizes into the selected HUD profile coordinate space while preserving the R34 detector and audio policies.

**Architecture:** Add one stateful viewport resolver before the existing visual observers. Native-aspect frames pass through unchanged; non-native frames require consecutive dark-border confirmation and then use a locked centered crop until client geometry changes.

**Tech Stack:** Python 3.12, NumPy, OpenCV-independent pixel statistics, pytest, PyInstaller.

**Spec:** `docs/superpowers/specs/2026-09-13-full-resolution-viewport-design.md`

## Global Constraints

- Preserve the R34 source and package; all implementation occurs in `SF6_Music_Trigger_FullResolution_R35`.
- Do not alter gauge, meter-level, contact/outcome, fade, audio, or template thresholds.
- Fail closed when viewport placement cannot be confirmed.
- Support common 16:9, 16:10, and 21:9 clients with an effective content height of at least 720 pixels.
- Keep capture screen-only and local; do not save viewport images.
- Do not commit the untracked snapshot into the dirty parent repository.

---

### Task 1: Centered viewport geometry and border evidence

**Files:**
- Create: `src/sf6_music_trigger/viewport.py`
- Create: `tests/test_viewport.py`

**Interfaces:**
- Produces: `ViewportRect(left: int, top: int, width: int, height: int)`.
- Produces: `centered_viewport(width: int, height: int, target_aspect: Real) -> ViewportRect`.
- Produces: `borders_confirm_viewport(image: np.ndarray, rect: ViewportRect) -> bool`.

- [ ] **Step 1: Write failing table-driven geometry tests**

```python
@pytest.mark.parametrize(("size", "expected"), [
    ((1920, 1080), ViewportRect(0, 0, 1920, 1080)),
    ((1920, 1200), ViewportRect(0, 60, 1920, 1080)),
    ((2560, 1080), ViewportRect(320, 0, 1920, 1080)),
    ((3440, 1440), ViewportRect(440, 0, 2560, 1440)),
])
def test_centered_viewport_maps_common_clients(size, expected):
    assert centered_viewport(*size, 16 / 9) == expected
```

- [ ] **Step 2: Run the geometry tests and verify they fail because `viewport.py` is absent**

Run: `python -m pytest -q tests/test_viewport.py -k centered_viewport`

- [ ] **Step 3: Implement validated integer geometry using a centered largest-fit rectangle**

```python
def centered_viewport(width: int, height: int, target_aspect: Real) -> ViewportRect:
    actual = width / height
    if actual > target_aspect:
        viewport_width = round(height * float(target_aspect))
        return ViewportRect((width - viewport_width) // 2, 0, viewport_width, height)
    viewport_height = round(width / float(target_aspect))
    return ViewportRect(0, (height - viewport_height) // 2, width, viewport_height)
```

- [ ] **Step 4: Add failing dark-border and bright-border tests, then implement conservative luminance evidence**

Use literal synthetic frames: bars filled with values `0` or `6` must pass; a side bar containing a value-80 region must fail. Require both outer regions, at least 98% of pixels at luminance 20 or lower, and a non-empty content rectangle.

- [ ] **Step 5: Run `tests/test_viewport.py` and verify all Task 1 tests pass**

### Task 2: Stable fail-closed normalization

**Files:**
- Modify: `src/sf6_music_trigger/viewport.py`
- Modify: `tests/test_viewport.py`

**Interfaces:**
- Produces: `ViewportStatus` with values `native`, `confirming`, `normalized`, `unsupported`, `too_small`.
- Produces: `ViewportResult(frame: CapturedFrame | None, status: ViewportStatus)`.
- Produces: `CenteredViewportResolver(target_aspect: Real, confirmation_frames: int = 3, minimum_content_height: int = 720)` with `resolve(frame)` and `reset()`.

- [ ] **Step 1: Write failing tests for native pass-through, three-frame confirmation, and locked crop**

The first two non-native frames return `frame=None/status=confirming`; the third returns a cropped 16:9 frame/status=normalized; later frames with the same geometry use the locked crop without reclassifying dark scene pixels.

- [ ] **Step 2: Run the resolver tests and verify the missing resolver failure**

- [ ] **Step 3: Implement state keyed by the complete `ClientGeometry`**

Reset confirmation whenever left, top, width, or height changes. Preserve frame id and monotonic timestamp; offset the returned geometry by the viewport rectangle.

- [ ] **Step 4: Write and pass tests for bright borders, sub-720p content, invalid images, and explicit reset**

Bright borders return `unsupported`; sub-720p effective content returns `too_small`; neither returns a frame.

### Task 3: Engine integration without detector changes

**Files:**
- Modify: `src/sf6_music_trigger/engine.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- `VisualTriggerEngine(..., viewport_resolver: object | None = None)` accepts a resolver implementing `resolve(CapturedFrame) -> ViewportResult` and `reset() -> None`.
- Exposes read-only `viewport_status: str`.

- [ ] **Step 1: Write a failing engine test proving no observer call while a resolver is confirming**

Use a real `ViewportResult(None, ViewportStatus.CONFIRMING)` and assert no meter observation or audio action occurs.

- [ ] **Step 2: Write a failing engine test proving both observers and the outcome observer receive the normalized frame**

The resolver returns a distinct cropped frame with the same frame id; assert the observer image dimensions match the crop.

- [ ] **Step 3: Implement resolver invocation after freshness acceptance and before `profile_applies_to`**

On no effective frame, call the existing runtime reset path and return cancellations only. On engine reset, reset the resolver too.

- [ ] **Step 4: Run engine, viewport, R27 exclusive-audio, R30 policy, and R33 outcome tests**

Run: `python -m pytest -q tests/test_viewport.py tests/test_engine.py tests/test_r27_exclusive_audio.py tests/test_r30_policy.py tests/test_outcome.py tests/test_contact_outcome.py`

### Task 4: R35 panel diagnostics and package identity

**Files:**
- Modify: `src/sf6_music_trigger/panel.py`
- Modify: `src/sf6_music_trigger/__init__.py`
- Modify: `tools/portable_release.py`
- Modify: `tests/test_panel_view_model.py`
- Modify: `tests/test_portable_release.py`

**Interfaces:**
- Panel status appends a short viewport label when the controller exposes it.
- `BUILD_ID` is exactly `R35`.

- [ ] **Step 1: Write failing tests for the R35 identity and privacy-safe viewport status text**

- [ ] **Step 2: Implement only status/identity wiring; do not alter panel audio or template controls**

- [ ] **Step 3: Update the group-test readme with the supported matrix and calibration fallback**

- [ ] **Step 4: Run panel and portable-release tests**

### Task 5: Regression, immutable R34 audit, and portable candidate

**Files:**
- Verify: all source and test files
- Create: sibling `SF6_Music_Trigger_R35` package and ZIP

**Interfaces:**
- Produces a portable self-checking R35 candidate with a cleared public audio configuration.

- [ ] **Step 1: Compare R34 and R35 hashes for detection-policy files that are not authorized to change**

`gauge_tracker.py`, `gauge_features.py`, `meter_level_features.py`, `meter_level_tracker.py`, `outcome.py`, `contact_hud.py`, `audio.py`, and `controller.py` must remain byte-identical unless an integration test proves a narrowly required controller status addition.

- [ ] **Step 2: Run the complete offline suite from an ASCII temporary directory**

Run: `python -m pytest -q -p no:cacheprovider --tb=short --basetemp <writable-test-temp>\pytest-r35-full`.

- [ ] **Step 3: Build the R35 portable folder and ZIP without overwriting R34**

- [ ] **Step 4: Run final EXE `--portable-self-check` under ordinary Windows permissions**

- [ ] **Step 5: Verify every manifest hash and ZIP member, then record the ZIP SHA-256**

- [ ] **Step 6: Report automated evidence separately from pending 16:10 and 21:9 live SF6 acceptance**
