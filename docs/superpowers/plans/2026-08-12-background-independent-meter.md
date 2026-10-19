# Background-independent SF6 meter detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect SF6 CA/full and zero/spent meters without relying on the translucent gauge's stage or character background.

**Architecture:** Add a calibration-derived active-gauge mask and HSV signature alongside legacy grayscale references. The mask is the stable full-versus-zero difference, so fixed CA labels and changing stage pixels are excluded. New calibration persists the mask, colour signature, and ranges; `observe_gauge` explicitly routes to them while schema-1 grayscale profiles retain their current path.

**Tech Stack:** Python 3.12, NumPy, OpenCV, pytest, DXcam abstraction.

## Global Constraints

- SF6 only; do not operate MK1 or other games.
- Do not modify the separate legacy prototype `SF6_Music_Trigger/main.py`.
- Do not change capture lifecycle, audio scheduling, or game input.
- Missing, unstable, or low-colour features fail closed and never arm a side.
- Real validation happens only after offline tests and explicit SF6-only permission.

---

### Task 1: HSV segmented-meter model

**Files:**
- Create: `src/sf6_music_trigger/meter_color_features.py`
- Create: `tests/test_meter_color_features.py`

**Interfaces:**
- Produces `MeterColorModel(side: Side, active_mask: np.ndarray, full_hue: int, hue_tolerance: int, saturation_floor: int, value_floor: int, full_fill: float, spent_fill: float)`.
- Produces `extract_meter_fill(crop_bgr: np.ndarray, model: MeterColorModel) -> float | None`.
- Produces `build_meter_color_model(side: Side, full_crops: Sequence[np.ndarray], spent_crops: Sequence[np.ndarray]) -> MeterColorModel`.

- [ ] **Step 1: Write failing synthetic-background tests**

```python
def test_full_and_zero_ignore_material_background_changes_outside_active_mask() -> None:
    model = build_meter_color_model(
        Side.P1, [_crop(6, (5, 5, 5))] * 5, [_crop(0, (5, 5, 5))] * 5
    )
    assert extract_meter_fill(_crop(6, (200, 30, 160)), model) == pytest.approx(model.full_fill)
    assert extract_meter_fill(_crop(0, (200, 30, 160)), model) == pytest.approx(model.spent_fill)

def test_three_segments_stay_between_full_and_spent() -> None:
    model = _model()
    assert model.spent_fill < extract_meter_fill(_crop(3, (90, 20, 30)), model) < model.full_fill

def test_low_saturation_crop_has_no_feature() -> None:
    assert extract_meter_fill(_gray_crop(), _model()) is None
```

- [ ] **Step 2: Verify RED**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_meter_color_features.py -v`

Expected: FAIL because `meter_color_features` does not exist.

- [ ] **Step 3: Implement minimum feature model**

```python
def extract_meter_fill(crop_bgr: np.ndarray, model: MeterColorModel) -> float | None:
    hsv = cv2.cvtColor(_require_bgr(crop_bgr), cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] >= model.saturation_floor) & (hsv[..., 2] >= model.value_floor)
    return None if int(mask.sum()) < _MIN_GAUGE_PIXELS else _median_segment_occupancy(mask)
```

`build_meter_color_model` accepts exactly five BGR crops per state, derives a persistent full-versus-zero active mask before deriving a full-state HSV signature and median occupancy, and rejects masks that are sparse, fragmented, unstable, reversed, or equal at their endpoints.

- [ ] **Step 4: Verify GREEN and validation**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_meter_color_features.py -v`

Expected: PASS, including malformed BGR, non-`Side`, bool/float floor, wrong sample count, unstable batch, and endpoint-separation tests.

- [ ] **Step 5: Commit**

```powershell
git add src/sf6_music_trigger/meter_color_features.py tests/test_meter_color_features.py
git commit -m "feat: add background-independent meter features"
```

### Task 2: Strict persisted model with legacy loading

**Files:**
- Modify: `src/sf6_music_trigger/meter_references.py`
- Modify: `tests/test_meter_references.py`

**Interfaces:**
- `MeterReferences` gains `color_model: MeterColorModel | None = None`.
- Schema-2 NPZ stores `classifier.npy == "masked_hsv_segments"`, an exact canonical boolean mask member, and exact scalar model members.
- Existing schema-1 archives load as `color_model=None`.

- [ ] **Step 1: Write failing persistence tests**

```python
def test_hsv_references_round_trip(tmp_path: Path) -> None:
    original = MeterReferences(Side.P1, _full(), _spent(), color_model=_model(Side.P1))
    save_references(tmp_path, "profiles/hud.json", "refs/p1.npz", original)
    loaded = load_references(tmp_path, "profiles/hud.json", "refs/p1.npz", expected_side=Side.P1)
    assert loaded.color_model == original.color_model

def test_schema_one_archive_stays_legacy(tmp_path: Path) -> None:
    _write_schema_one(tmp_path)
    assert load_references(tmp_path, "profiles/hud.json", "refs/p1.npz", expected_side=Side.P1).color_model is None

def test_unknown_classifier_fails_closed(tmp_path: Path) -> None:
    _write_classifier_archive(tmp_path, "unknown")
    with pytest.raises(ValueError, match="classifier"):
        load_references(tmp_path, "profiles/hud.json", "refs/p1.npz", expected_side=Side.P1)
```

- [ ] **Step 2: Verify RED**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_meter_references.py -k "hsv or schema_one or classifier" -v`

Expected: FAIL because schema-2 persistence is absent.

- [ ] **Step 3: Implement bounded schema dispatch**

Keep ZIP size/member validation. Add a schema-2 expected-member map; validate classifier, canonical boolean mask shape, scalar dtypes/ranges, and side before `MeterColorModel` construction. Write schema 2 only for a colour model; keep legacy arrays schema 1.

- [ ] **Step 4: Verify GREEN**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_meter_references.py tests/test_meter_color_features.py -v`

Expected: PASS, including legacy fixtures and atomic-write failure preservation.

- [ ] **Step 5: Commit**

```powershell
git add src/sf6_music_trigger/meter_references.py tests/test_meter_references.py
git commit -m "feat: persist hsv meter calibration"
```

### Task 3: Observation routing and tracker safety

**Files:**
- Modify: `src/sf6_music_trigger/gauge_features.py`
- Modify: `tests/test_gauge_features.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- `observe_gauge` uses `extract_meter_fill` only when `references.color_model` exists.
- `None` returns the existing hidden sentinel.
- Filled-segment score distances are `abs(fill - full_fill)` and `abs(fill - spent_fill)` after only active-mask pixels are colour-matched.

- [ ] **Step 1: Write failing routing tests**

```python
def test_hsv_route_recognizes_full_meter_over_two_backgrounds() -> None:
    refs = _masked_hsv_references(Side.P1)
    observation = observe_gauge(
        _frame(6, (220, 10, 150)), side=Side.P1, frame_id=1,
        roi=_roi(), references=refs, sampled_at=0
    )
    assert observation.full_confidence > 0.9

def test_hsv_route_returns_hidden_for_low_colour_crop() -> None:
    assert _observe_hsv(_gray_crop()).hud_visible is False

def test_three_segment_hsv_samples_neither_arm_nor_confirm() -> None:
    assert _run_tracker(_three_segment_samples()).events == ()
```

- [ ] **Step 2: Verify RED**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_gauge_features.py tests/test_engine.py -k "hsv_route or three_segment_hsv" -v`

Expected: FAIL because observation only compares grayscale arrays.

- [ ] **Step 3: Implement explicit colour-model branch**

```python
if references.color_model is not None:
    fill = extract_meter_fill(crop, references.color_model)
    if fill is None:
        return _hidden_observation(side, frame_id, sampled_at)
    return _observation_from_distances(
        side, frame_id, sampled_at,
        abs(fill - references.color_model.full_fill),
        abs(fill - references.color_model.spent_fill),
    )
```

Keep the legacy branch behaviour unchanged. Add a spy proving `preprocess_meter` is not called for HSV references.

- [ ] **Step 4: Verify GREEN**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_gauge_features.py tests/test_engine.py -v`

Expected: PASS, including a background-only full-to-zero trigger sequence.

- [ ] **Step 5: Commit**

```powershell
git add src/sf6_music_trigger/gauge_features.py tests/test_gauge_features.py tests/test_engine.py
git commit -m "feat: classify gauges without scene background"
```

### Task 4: Existing wizard integration and offline gate

**Files:**
- Modify: `src/sf6_music_trigger/panel.py`
- Modify: `tests/test_panel_view_model.py`
- Modify: `docs/superpowers/specs/2026-08-12-background-independent-meter-design.md`

**Interfaces:**
- `CalibrationSession.complete()` builds one `MeterColorModel` per P1/P2 batch, then persists it in `MeterReferences` under the existing reservation/rollback transaction.
- Wizard prompts name `P1 满气（CA）`, `P1 零气`, `P2 满气（CA）`, `P2 零气`.
- A colour build error maps to safe `颜色特征无效` diagnostics.

- [ ] **Step 1: Write failing wizard tests**

```python
def test_complete_persists_colour_models_after_four_fresh_batches(tmp_path: Path) -> None:
    session = _configured_session(tmp_path)
    _accept_full_and_zero_batches(session, backgrounds=[(0, 0, 0), (200, 30, 160)])
    session.complete()
    assert _loaded_p1(tmp_path).color_model is not None

def test_colour_build_failure_preserves_prior_files(tmp_path: Path) -> None:
    before = _persisted_bytes(tmp_path)
    with pytest.raises(ValueError, match="colour"):
        _gray_batch_session(tmp_path).complete()
    assert _persisted_bytes(tmp_path) == before

def test_wizard_colour_failure_message_is_safe() -> None:
    assert "颜色特征无效" in _failing_wizard_message()
```

- [ ] **Step 2: Verify RED**

Run: `\.venv\Scripts\python.exe -m pytest tests/test_panel_view_model.py -k "colour_models or colour_build_failure or wizard_colour" -v`

Expected: FAIL because `CalibrationSession` only builds grayscale references.

- [ ] **Step 3: Implement minimal panel wiring**

Build masked-colour models from the already accepted batches in `complete`, pass them into `MeterReferences`, and preserve save/reload/equality/rollback behavior. Change only calibration labels and safe colour-error category; do not touch capture ownership or Tk loop lifetime.

- [ ] **Step 4: Verify all offline tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_meter_color_features.py tests/test_meter_references.py tests/test_gauge_features.py tests/test_engine.py tests/test_panel_view_model.py -v
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: all tests pass and `git diff --check` has no output.

- [ ] **Step 5: Record offline evidence and commit**

Append actual counts to the design spec, explicitly retain “real SF6 acceptance not yet claimed,” then:

```powershell
git add src/sf6_music_trigger/panel.py tests/test_panel_view_model.py docs/superpowers/specs/2026-08-12-background-independent-meter-design.md
git commit -m "feat: calibrate background-independent SF6 meters"
```

## Plan self-review

- Task 1 implements the scene-independent feature and fail-closed bounds.
- Task 2 defines exact persistence and legacy compatibility.
- Task 3 prevents background changes from producing tracker events.
- Task 4 wires fresh calibration batches, atomic failure handling, safe diagnostics, and the full offline test gate.
- All named types and functions are defined before use; no deferred implementation markers remain.
