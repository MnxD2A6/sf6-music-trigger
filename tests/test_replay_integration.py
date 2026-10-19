from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import cv2
import numpy as np
import pytest

from sf6_music_trigger.domain import (
    CaptureSample,
    CaptureStatus,
    CapturedFrame,
    ClientGeometry,
    Side,
)
from sf6_music_trigger.gauge_features import GaugeObservation
from sf6_music_trigger.hud_profile import (
    CalibrationContext,
    HudProfile,
    NormalizedRoi,
    SideHudProfile,
    save_profile,
)
from sf6_music_trigger.meter_references import (
    MeterReferences,
    preprocess_meter,
    save_references,
)
from sf6_music_trigger.meter_color_features import build_meter_color_model


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = REPO_ROOT / "tools" / "replay.py"
METER_REPLAY_SCRIPT = REPO_ROOT / "tools" / "replay_meter_levels.py"


@dataclass(frozen=True)
class ReplayFiles:
    profile: Path
    frames: Path


def _level_observation(side: Side, frame_id: int, level: int | None) -> GaugeObservation:
    if level is None:
        return GaugeObservation(
            side=side,
            frame_id=frame_id,
            sampled_at=frame_id / 20,
            hud_visible=False,
            full_confidence=0.0,
            spent_confidence=0.0,
            full_distance=1.0,
            spent_distance=1.0,
            meter_level_supported=True,
        )
    return GaugeObservation(
        side=side,
        frame_id=frame_id,
        sampled_at=frame_id / 20,
        hud_visible=True,
        full_confidence=0.5,
        spent_confidence=0.5,
        full_distance=0.5,
        spent_distance=0.5,
        normalized_meter=None,
        meter_level_supported=True,
        meter_level=level,
        meter_level_confidence=0.9,
    )


def _level_pairs(
    p1: list[int | None],
    p2: list[int | None] | None = None,
) -> list[tuple[GaugeObservation, GaugeObservation]]:
    second = p2 if p2 is not None else [0] * len(p1)
    assert len(p1) == len(second)
    return [
        (
            _level_observation(Side.P1, frame_id, first_level),
            _level_observation(Side.P2, frame_id, second_level),
        )
        for frame_id, (first_level, second_level) in enumerate(zip(p1, second))
    ]


@pytest.mark.parametrize(
    ("levels", "expected_delta"),
    [
        pytest.param([0] * 10 + [1] * 3, None, id="zero-to-one-gain"),
        pytest.param([1] * 10 + [2] * 3, None, id="one-to-two-gain"),
        pytest.param([1] * 10 + [0] * 16, (1, 0, 1), id="sa1"),
        pytest.param([2] * 10 + [0] * 16, (2, 0, 2), id="sa2"),
        pytest.param([3] * 10 + [0] * 16, (3, 0, 3), id="sa3-ca"),
        pytest.param([1] * 10 + [None] * 9 + [0] * 16, (1, 0, 1), id="short-gap"),
        pytest.param([1] * 10 + [None] * 10 + [0] * 16, None, id="long-gap-reset"),
    ],
)
def test_meter_level_replay_uses_production_tracker_for_gains_spends_and_gaps(
    levels: list[int | None],
    expected_delta: tuple[int, int, int] | None,
) -> None:
    from tools.replay_meter_levels import MeterLevelReplay

    replay = MeterLevelReplay("synthetic-profile")
    for pair in _level_pairs(levels):
        replay.update(pair)

    actual = [
        (event.meter_before, event.meter_after, event.levels_spent)
        for event in replay.events
        if event.side is Side.P1
    ]
    assert actual == ([] if expected_delta is None else [expected_delta])


def test_meter_level_replay_isolates_sides_and_resets_on_round_or_stale_frame() -> None:
    from tools.replay_meter_levels import MeterLevelReplay

    simultaneous = MeterLevelReplay("synthetic-profile")
    for pair in _level_pairs([3] * 10 + [0] * 16, [2] * 10 + [0] * 16):
        simultaneous.update(pair)
    assert [
        (event.side, event.levels_spent) for event in simultaneous.events
    ] == [(Side.P1, 3), (Side.P2, 2)]

    reset = MeterLevelReplay("synthetic-profile")
    baseline = _level_pairs([3] * 10)
    for pair in baseline:
        reset.update(pair)
    reset.reset("round ended")
    for pair in _level_pairs([0] * 15):
        reset.update(pair)
    assert reset.events == ()

    stale = MeterLevelReplay("synthetic-profile")
    for pair in baseline:
        stale.update(pair)
    assert stale.update(baseline[-1]) == ()
    for pair in _level_pairs([0] * 15):
        stale.update(pair)
    assert stale.events == ()


def test_meter_level_replay_cli_fails_closed_for_unreadable_video(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(METER_REPLAY_SCRIPT),
            "--video",
            str(tmp_path / "private-video.mp4"),
            "--profile-root",
            str(tmp_path),
            "--profile",
            "profiles/missing.json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "private-video" not in completed.stderr


def _replay_private_video(
    profile: Path,
    video: Path,
    *,
    selected: frozenset[Side],
    sample_fps: int = 20,
) -> dict[str, object]:
    from tools.replay import _build_engine

    engine, clock, gate, audio, outcomes, _inputs = _build_engine(profile, selected)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise AssertionError("private replay video could not be opened")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise AssertionError("private replay video reported an invalid frame rate")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    geometry = ClientGeometry(0, 0, width, height)
    frame_id = 0
    source_index = 0
    next_source_index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if source_index >= next_source_index:
                timestamp = frame_id / sample_fps
                clock.value = timestamp
                engine.process(
                    CaptureSample(
                        CaptureStatus.RUNNING,
                        frame=CapturedFrame(frame_id, image, timestamp, geometry),
                    )
                )
                frame_id += 1
                next_source_index = round(frame_id * source_fps / sample_fps)
            source_index += 1
    finally:
        capture.release()
    return {
        "processed_frames": frame_id,
        "rejected_stale_frames": gate.rejected_frames,
        "events": [
            {
                "side": event.side.value,
                "pre_spend_frame_id": event.pre_spend_frame_id,
                "confirmed_frame_id": event.confirmed_frame_id,
                "confirmed_at": float(event.confirmed_at),
            }
            for event in engine.event_log
            if event.side in selected
        ],
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "side": candidate.side.value,
                "suspected_at": float(candidate.suspected_at),
            }
            for candidate in engine.candidate_log
            if candidate.side in selected
        ],
        "cancellations": [
            {
                "candidate_id": cancellation.candidate_id,
                "side": cancellation.side.value,
                "decided_at": float(cancellation.decided_at),
                "reason": cancellation.reason.value,
            }
            for cancellation in engine.cancellation_log
            if cancellation.side in selected
        ],
        "outcomes": [
            {
                "side": decision.side.value,
                "label": decision.label.value,
                "decided_at": float(decision.decided_at),
                "evidence": list(decision.evidence),
            }
            for decision in outcomes.decisions
            if decision.side in selected
        ],
        "audio_batches": [
            {
                "at": float(timestamp),
                "actions": [_action(action.kind.value, action.side.value, volume=action.volume, fade_ms=action.fade_ms) for action in actions],
            }
            for timestamp, actions in audio.batches
        ],
    }


def _meter_pattern(kind: str) -> np.ndarray:
    horizontal = np.linspace(80, 48, 64, dtype=np.uint8)
    spent_gray = np.repeat(horizontal[np.newaxis, :], 32, axis=0)
    spent = np.repeat(spent_gray[:, :, np.newaxis], 3, axis=2)
    full = spent.copy()
    full[..., 0] = 12
    full[..., 1] = np.repeat(
        np.linspace(176, 224, 64, dtype=np.uint8)[np.newaxis, :],
        32,
        axis=0,
    )
    full[..., 2] = 24
    if kind == "full":
        return np.ascontiguousarray(full)
    elif kind == "spent":
        return np.ascontiguousarray(spent)
    elif kind == "one_bar":
        image = spent.copy()
        image[:, :21] = full[:, :21]
        return np.ascontiguousarray(image)
    elif kind == "two_bars":
        image = spent.copy()
        image[:, :42] = full[:, :42]
        return np.ascontiguousarray(image)
    elif kind == "disrupted":
        checker = (
            (np.indices((32, 64)).sum(axis=0) // 4 % 2) * 255
        ).astype(np.uint8)
        return np.repeat(checker[:, :, np.newaxis], 3, axis=2)
    elif kind == "hidden":
        return np.zeros((32, 64, 3), dtype=np.uint8)
    else:
        raise AssertionError(f"unsupported synthetic meter kind: {kind}")


def _profile() -> HudProfile:
    return HudProfile(
        profile_id="task10-synthetic",
        calibration=CalibrationContext(
            client_width=480,
            client_height=320,
            client_aspect_ratio=1.5,
            hud_scale_setting="synthetic",
            language="English",
            brightness_setting="synthetic",
            hdr_enabled=False,
        ),
        battle_roi=NormalizedRoi(0.0, 0.0, 1.0, 1.0),
        training_hud_roi=None,
        p1=SideHudProfile(
            meter_roi=NormalizedRoi(0.25, 0.75, 0.45, 0.95),
            references_path="references/p1.npz",
        ),
        p2=SideHudProfile(
            meter_roi=NormalizedRoi(0.55, 0.75, 0.75, 0.95),
            references_path="references/p2.npz",
        ),
    )


def _references(side: Side) -> MeterReferences:
    full = preprocess_meter(_meter_pattern("full"))
    spent = preprocess_meter(_meter_pattern("spent"))
    assert full.hud_visible and spent.hud_visible
    color_model = build_meter_color_model(
        side,
        [_meter_pattern("full") for _ in range(5)],
        [_meter_pattern("spent") for _ in range(5)],
    )
    return MeterReferences(
        side,
        full.normalized_gray,
        spent.normalized_gray,
        color_model=color_model,
    )


def _draw_replay_level(image: np.ndarray, side: Side, glyph: str) -> None:
    center_x = 110 if side is Side.P1 else 366
    top, middle, bottom = 258, 272, 286
    left, right = center_x - 10, center_x + 10
    if glyph == "0":
        segments = [
            ((left, top), (right, top)),
            ((left, top), (left, bottom)),
            ((right, top), (right, bottom)),
            ((left, bottom), (right, bottom)),
        ]
    elif glyph == "1":
        segments = [((center_x - 2, top + 3), (center_x + 2, top)), ((center_x + 2, top), (center_x + 2, bottom))]
    elif glyph in {"2", "3"}:
        lower_start = (left, middle) if glyph == "2" else (right, middle)
        lower_end = (left, bottom) if glyph == "2" else (right, bottom)
        segments = [
            ((left, top), (right, top)),
            ((right, top), (right, middle)),
            ((left, middle), (right, middle)),
            (lower_start, lower_end),
            ((left, bottom), (right, bottom)),
        ]
    else:
        raise AssertionError(f"unsupported replay glyph: {glyph}")
    outline = (12, 220, 24)
    for colour, thickness in ((outline, 7), ((245, 245, 245), 3)):
        for start, end in segments:
            cv2.line(image, start, end, colour, thickness, cv2.LINE_AA)


def _write_replay(
    root: Path,
    *,
    p1: list[str],
    p2: list[str],
    frame_ids: list[int] | None = None,
    suffix: str = ".npy",
    reverse_write_order: bool = False,
    p1_hit_from_frame: int | None = None,
) -> ReplayFiles:
    assert len(p1) == len(p2)
    ids = list(range(len(p1))) if frame_ids is None else frame_ids
    assert len(ids) == len(p1)

    user_root = root / "user-data"
    profile_path = user_root / "profile.json"
    save_profile(user_root, "profile.json", _profile())
    save_references(
        user_root,
        "profile.json",
        "references/p1.npz",
        _references(Side.P1),
    )
    save_references(
        user_root,
        "profile.json",
        "references/p2.npz",
        _references(Side.P2),
    )

    frames = root / "frames"
    frames.mkdir(parents=True)
    duplicate_ordinals: dict[int, int] = {}
    entries: list[tuple[Path, np.ndarray]] = []
    for frame_id, p1_kind, p2_kind in zip(ids, p1, p2, strict=True):
        ordinal = duplicate_ordinals.get(frame_id, 0)
        duplicate_ordinals[frame_id] = ordinal + 1
        path = frames / f"{frame_id:06d}-{ordinal:03d}{suffix}"
        image = np.full((320, 480, 3), 96, dtype=np.uint8)
        image[240:304, 120:216] = cv2.resize(
            _meter_pattern(p1_kind), (96, 64), interpolation=cv2.INTER_AREA
        )
        image[240:304, 264:360] = cv2.resize(
            _meter_pattern(p2_kind), (96, 64), interpolation=cv2.INTER_AREA
        )
        if p1_hit_from_frame is not None and frame_id >= p1_hit_from_frame:
            image[8:34, 264:452] = 0
        level_by_kind = {"full": "3", "spent": "0", "one_bar": "1", "two_bars": "2"}
        for side, kind in (
            (Side.P1, p1_kind),
            (Side.P2, p2_kind),
        ):
            glyph = level_by_kind.get(kind)
            if glyph is None:
                continue
            _draw_replay_level(image, side, glyph)
        entries.append((path, np.ascontiguousarray(image)))

    if reverse_write_order:
        entries.reverse()
    for path, image in entries:
        if suffix == ".npy":
            np.save(path, image, allow_pickle=False)
        elif suffix == ".png":
            assert cv2.imwrite(str(path), image)
        else:
            raise AssertionError(f"unsupported test suffix: {suffix}")
    return ReplayFiles(profile_path, frames)


def _run_cli(
    files: ReplayFiles,
    result_path: Path,
    *,
    side: str,
    fps: int = 20,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not existing else os.pathsep.join((source_path, existing))
    )
    return subprocess.run(
        [
            sys.executable,
            str(REPLAY_SCRIPT),
            "--profile",
            str(files.profile),
            "--frames",
            str(files.frames),
            "--side",
            side,
            "--fps",
            str(fps),
            "--json",
            str(result_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _read_success(completed: subprocess.CompletedProcess[str], result: Path) -> dict:
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    return json.loads(result.read_text(encoding="utf-8"))


def _input_hashes(files: ReplayFiles) -> dict[str, str]:
    inputs = [files.profile]
    inputs.extend(sorted((files.profile.parent / "references").iterdir()))
    inputs.extend(sorted(files.frames.iterdir()))
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }


def _audio_counts(*, p1_stop: int = 0, p1_play: int = 0, p2_stop: int = 0, p2_play: int = 0) -> dict:
    return {
        "P1": {"stop": p1_stop, "play": p1_play, "fade": 0},
        "P2": {"stop": p2_stop, "play": p2_play, "fade": 0},
    }


def _action(kind: str, side: str, *, volume: float | None = None, fade_ms: int | None = None) -> dict:
    return {
        "kind": kind,
        "side": side,
        "volume": volume,
        "fade_ms": fade_ms,
    }


def test_replay_cli_reports_exactly_one_confirmed_event(tmp_path: Path) -> None:
    files = _write_replay(
        tmp_path,
        p1=["full"] * 10 + ["disrupted"] + ["spent"] * 16,
        p2=["full"] * 27,
    )
    result_path = tmp_path / "result.json"

    result = _read_success(_run_cli(files, result_path, side="P1"), result_path)

    assert result == {
        "processed_frames": 27,
        "rejected_stale_frames": 0,
        "events": [
            {
                "side": "P1",
                    "pre_spend_frame_id": 9,
                "confirmed_frame_id": 26,
                "confirmed_at": 1.3,
            }
        ],
        "lifecycle": [
            {
                "frame_id": 13,
                "side": "P1",
                "transition": "CANDIDATE",
                "meter_before": 3,
                "meter_after": 0,
                "audio_actions": ["stop", "play"],
            },
            {
                "frame_id": 26,
                "side": "P1",
                "transition": "CONFIRMED",
                "meter_before": 3,
                "meter_after": 0,
                "audio_actions": [],
            },
        ],
        "audio_actions": _audio_counts(p1_stop=1, p1_play=1),
        "outcomes": [],
        "audio_batches": [
            {
                "at": 0.65,
                "actions": [
                    _action("stop", "P1"),
                    _action("play", "P1", volume=1.0),
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    ("side", "p1", "p2", "expected_events", "expected_audio"),
    [
        pytest.param(
            "P2",
            ["full"] * 15,
            ["full"] * 10 + ["disrupted"] + ["spent"] * 4,
            [{"side": "P2", "pre_spend_frame_id": 9, "confirmed_frame_id": 14, "confirmed_at": 0.7}],
            _audio_counts(p2_stop=1, p2_play=1),
            id="P2",
        ),
        pytest.param(
            "both",
            ["full"] * 10 + ["disrupted"] + ["spent"] * 17,
            ["full"] * 11 + ["disrupted"] + ["spent"] * 16,
            [
                {"side": "P2", "pre_spend_frame_id": 10, "confirmed_frame_id": 15, "confirmed_at": 0.75},
                    {"side": "P1", "pre_spend_frame_id": 9, "confirmed_frame_id": 26, "confirmed_at": 1.3},
            ],
            _audio_counts(p1_stop=1, p1_play=1, p2_stop=1, p2_play=1),
            id="independent-sides",
        ),
    ],
)
def test_replay_cli_preserves_p2_and_two_side_isolation(
    tmp_path: Path,
    side: str,
    p1: list[str],
    p2: list[str],
    expected_events: list[dict],
    expected_audio: dict,
) -> None:
    files = _write_replay(tmp_path, p1=p1, p2=p2)
    result_path = tmp_path / "result.json"

    result = _read_success(_run_cli(files, result_path, side=side), result_path)

    assert result["events"] == expected_events
    assert result["audio_actions"] == expected_audio


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param(["full"] * 3, id="no-spend"),
        pytest.param(["spent", "full", "full"], id="single-frame-noise"),
        pytest.param(["spent", "hidden", "spent", "spent"], id="hidden-hud-reset"),
    ],
)
def test_replay_cli_no_spend_controls_produce_no_events(
    tmp_path: Path,
    tail: list[str],
) -> None:
    p1 = ["full"] * 10 + tail
    files = _write_replay(tmp_path, p1=p1, p2=["full"] * len(p1))
    result_path = tmp_path / "result.json"

    result = _read_success(_run_cli(files, result_path, side="P1"), result_path)

    assert result["events"] == []
    assert result["audio_actions"] == _audio_counts()
    assert result["rejected_stale_frames"] == 0


def test_replay_cli_rejects_repeated_frame_ids_and_resets_detection(
    tmp_path: Path,
) -> None:
    p1 = ["full"] * 10 + ["spent"] * 4
    files = _write_replay(
        tmp_path,
        p1=p1,
        p2=["full"] * len(p1),
        frame_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 12, 13],
    )
    result_path = tmp_path / "result.json"

    result = _read_success(_run_cli(files, result_path, side="P1"), result_path)

    assert result == {
        "processed_frames": 14,
        "rejected_stale_frames": 1,
        "events": [],
        "lifecycle": [],
        "audio_actions": _audio_counts(),
        "outcomes": [],
        "audio_batches": [],
    }


def test_replay_cli_confirms_without_restart_but_brightness_is_not_hit(
    tmp_path: Path,
) -> None:
    p1 = ["full"] * 10 + ["disrupted"] + ["spent"] * 30
    files = _write_replay(
        tmp_path,
        p1=p1,
        p2=["full"] * len(p1),
        p1_hit_from_frame=27,
    )
    result_path = tmp_path / "result.json"

    result = _read_success(_run_cli(files, result_path, side="P1"), result_path)

    assert result["events"] == [
        {
            "side": "P1",
                "pre_spend_frame_id": 9,
            "confirmed_frame_id": 26,
            "confirmed_at": 1.3,
        }
    ]
    # This synthetic clip changes ROI brightness, not a readable health bar.
    # It ends before the UNKNOWN fallback deadline; luma alone is not HIT.
    assert result["outcomes"] == []
    assert result["audio_actions"] == _audio_counts(p1_stop=1, p1_play=1)
    assert [batch["at"] for batch in result["audio_batches"]] == [0.65]
    assert result["audio_batches"][0]["at"] != result["events"][0]["confirmed_at"]
    assert result["lifecycle"] == [
        {
            "frame_id": 13,
            "side": "P1",
            "transition": "CANDIDATE",
            "meter_before": 3,
            "meter_after": 0,
            "audio_actions": ["stop", "play"],
        },
        {
            "frame_id": 26,
            "side": "P1",
            "transition": "CONFIRMED",
            "meter_before": 3,
            "meter_after": 0,
            "audio_actions": [],
        },
    ]


def test_replay_cli_round_end_candidate_cancels_once_without_confirmation(
    tmp_path: Path,
) -> None:
    p1 = ["full"] * 10 + ["spent"] * 3 + ["full"] * 3
    files = _write_replay(tmp_path, p1=p1, p2=["full"] * len(p1))
    result_path = tmp_path / "result.json"

    result = _read_success(_run_cli(files, result_path, side="P1"), result_path)

    assert result["events"] == []
    assert result["lifecycle"] == [
        {
            "frame_id": 12,
            "side": "P1",
            "transition": "CANDIDATE",
            "meter_before": 3,
            "meter_after": 0,
            "audio_actions": ["stop", "play"],
        },
        {
            "frame_id": 15,
            "side": "P1",
            "transition": "CANCELLED",
            "meter_before": 3,
            "meter_after": 0,
            "audio_actions": ["fade"],
        },
    ]


def test_replay_lifecycle_isolates_concurrent_same_identity_by_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sf6_music_trigger.gauge_tracker as gauge_tracker_module
    from tools import replay as replay_module

    files = _write_replay(
        tmp_path,
        p1=["full"] * 10 + ["spent"] * 3 + ["full"] * 4,
        p2=["two_bars"] * 10 + ["spent"] * 3 + ["hidden"] + ["two_bars"] * 3,
    )
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        gauge_tracker_module.uuid,
        "uuid4",
        lambda: type("FixedUuid", (), {"hex": "shared-identity"})(),
    )

    assert replay_module.run([
        "--profile",
        str(files.profile),
        "--frames",
        str(files.frames),
        "--side",
        "both",
        "--fps",
        "20",
        "--json",
        str(result_path),
    ]) == 0

    result = json.loads(result_path.read_text(encoding="utf-8"))
    cancellations = [
        row for row in result["lifecycle"] if row["transition"] == "CANCELLED"
    ]
    assert cancellations == [
        {
            "frame_id": 15,
            "side": "P1",
            "transition": "CANCELLED",
            "meter_before": 3,
            "meter_after": 0,
            "audio_actions": ["fade"],
        },
        {
            "frame_id": 16,
            "side": "P2",
            "transition": "CANCELLED",
            "meter_before": 2,
            "meter_after": 0,
            "audio_actions": [],
        },
    ]


def test_replay_round_transition_cancels_candidate_once_without_confirmation(
    tmp_path: Path,
) -> None:
    from tools.replay import _build_engine

    p1 = ["full"] * 10 + ["spent"] * 3
    files = _write_replay(tmp_path, p1=p1, p2=["full"] * len(p1))
    engine, clock, _gate, audio, _outcomes, _inputs = _build_engine(
        files.profile,
        frozenset({Side.P1}),
    )
    geometry = ClientGeometry(0, 0, 480, 320)
    updates = []
    for frame_id, path in enumerate(sorted(files.frames.iterdir())):
        sampled_at = frame_id / 20
        clock.value = sampled_at
        updates.append(engine.process(CaptureSample(
            CaptureStatus.RUNNING,
            CapturedFrame(
                frame_id,
                np.load(path, allow_pickle=False).copy(),
                sampled_at,
                geometry,
            ),
        )))

    engine.reset("round transition")

    assert sum(len(update.candidates) for update in updates) == 1
    assert engine.event_log == ()
    assert len(engine.cancellation_log) == 1
    assert [[action.kind.value for action in batch] for _, batch in audio.batches] == [
        ["stop", "play"],
        ["fade"],
    ]


def test_replay_cli_loads_png_and_uses_sorted_filename_order(tmp_path: Path) -> None:
    files = _write_replay(
        tmp_path,
        p1=["full"] * 10 + ["disrupted"] + ["spent"] * 16,
        p2=["full"] * 27,
        suffix=".png",
        reverse_write_order=True,
    )
    result_path = tmp_path / "result.json"

    result = _read_success(_run_cli(files, result_path, side="P1"), result_path)

    assert result["events"] == [
        {"side": "P1", "pre_spend_frame_id": 9, "confirmed_frame_id": 26, "confirmed_at": 1.3}
    ]


def test_replay_cli_sa3_scope_fades_stable_one_bar_after_immediate_playback(
    tmp_path: Path,
) -> None:
    p1 = ["full"] * 10 + ["disrupted"] + ["spent"] * 16 + ["one_bar"] * 20
    files = _write_replay(tmp_path, p1=p1, p2=["full"] * len(p1))
    result_path = tmp_path / "result.json"

    result = _read_success(_run_cli(files, result_path, side="P1"), result_path)

    assert result["events"] == [
        {
            "side": "P1",
                "pre_spend_frame_id": 9,
            "confirmed_frame_id": 26,
            "confirmed_at": 1.3,
        }
    ]
    assert result["outcomes"] == [
        {
            "side": "P1",
            "label": "FILTERED_TIER",
            "decided_at": 1.55,
            "evidence": ["remaining_meter_1"],
        }
    ]
    assert result["audio_actions"] == {
        "P1": {"stop": 1, "play": 1, "fade": 1},
        "P2": {"stop": 0, "play": 0, "fade": 0},
    }
    assert result["audio_batches"] == [
        {
            "at": 0.65,
            "actions": [
                _action("stop", "P1"),
                _action("play", "P1", volume=1.0),
            ],
        },
        {
            "at": 1.55,
            "actions": [_action("fade", "P1", fade_ms=1500)],
        },
    ]


def test_replay_cli_zero_bar_without_contact_fades_after_four_seconds_played(
    tmp_path: Path,
) -> None:
    p1 = ["full"] * 10 + ["disrupted"] + ["spent"] * 90
    files = _write_replay(tmp_path, p1=p1, p2=["full"] * len(p1))
    result_path = tmp_path / "result.json"

    result = _read_success(_run_cli(files, result_path, side="P1"), result_path)

    assert result["outcomes"] == [
        {
            "side": "P1",
            "label": "UNKNOWN",
            "decided_at": 3.65,
            "evidence": ["contact_not_established"],
        }
    ]
    assert result["audio_actions"] == {
        "P1": {"stop": 1, "play": 1, "fade": 1},
        "P2": {"stop": 0, "play": 0, "fade": 0},
    }
    assert result["audio_batches"][-1] == {
        "at": 4.65, "actions": [_action("fade", "P1", fade_ms=1500)],
    }


@pytest.mark.parametrize(
    (
        "recording_name",
        "candidate_count",
        "event_times",
        "cancellation_count",
        "outcome_labels",
        "batch_times",
    ),
    [
        pytest.param("不说了.mp4", 0, [], 0, [], [], id="desktop-task-view"),
        pytest.param(
            "CA0气测试.mp4",
            1,
            [9.2],
            0,
            ["HIT"],
            [9.15],
            id="stable-ca-zero",
        ),
        pytest.param(
            "终于基本成功了，谢谢你S02.mp4",
            3,
            [3.8, 16.2, 29.9],
            0,
            ["HIT", "HIT", "HIT"],
            [3.75, 16.15, 29.85],
            id="stable-three-ca-events",
        ),
        pytest.param(
            "杰米测试3.mp4",
            3,
            [1.95, 14.55, 23.05],
            0,
            ["HIT"],
            [1.9],
            id="sa3-filter-and-zero-bar-resolution",
        ),
    ],
)
def test_private_sf6_recordings_replay_without_real_audio(
    recording_name: str,
    candidate_count: int,
    event_times: list[float],
    cancellation_count: int,
    outcome_labels: list[str],
    batch_times: list[float],
) -> None:
    fixture_root = Path.home() / "Videos" / "屏幕录制" / "SF6"
    recording = fixture_root / recording_name
    profile = (
        REPO_ROOT.parent
        / "SF6_Music_Trigger_ColorCandidate_20260816"
        / "profiles"
        / "screenshot-sf6-color-20260816.json"
    )
    if not recording.is_file():
        pytest.skip(f"local SF6 recording fixture is absent: {recording_name}")
    if not profile.is_file():
        pytest.skip("local calibrated colour profile fixture is absent")

    result = _replay_private_video(
        profile,
        recording,
        selected=frozenset({Side.P1, Side.P2}),
    )

    assert result["rejected_stale_frames"] == 0
    assert len(result["candidates"]) == candidate_count
    assert [event["confirmed_at"] for event in result["events"]] == event_times
    assert len(result["cancellations"]) == cancellation_count
    candidate_ids = {
        candidate["candidate_id"] for candidate in result["candidates"]
    }
    assert all(
        cancellation["candidate_id"] in candidate_ids
        for cancellation in result["cancellations"]
    )
    assert [outcome["label"] for outcome in result["outcomes"]] == outcome_labels
    assert [batch["at"] for batch in result["audio_batches"]] == batch_times
    if recording_name == "不说了.mp4":
        assert result["events"] == []
        assert result["audio_batches"] == []
    if recording_name == "杰米测试3.mp4":
        assert [
            [action["kind"] for action in batch["actions"]]
            for batch in result["audio_batches"]
        ] == [
            ["stop", "play"],
        ]
        assert result["outcomes"][0]["evidence"] == [
            "opponent_health_changed"
        ]
        assert result["outcomes"][0]["decided_at"] == 3.35


def test_replay_cli_rejects_variable_width_frame_ids_before_replay(
    tmp_path: Path,
) -> None:
    files = _write_replay(
        tmp_path,
        p1=["full"] * 11,
        p2=["full"] * 11,
    )
    for path in list(files.frames.iterdir()):
        frame_id = int(path.name.split("-", 1)[0])
        path.rename(files.frames / f"{frame_id}-000.npy")
    result_path = tmp_path / "result.json"

    completed = _run_cli(files, result_path, side="P1")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "fixed-width numeric frame ids" in completed.stderr.lower()
    assert not result_path.exists()


def test_replay_cli_reads_inputs_without_modifying_or_copying_them(tmp_path: Path) -> None:
    files = _write_replay(
        tmp_path,
        p1=["full"] * 6 + ["spent"] * 3,
        p2=["full"] * 9,
    )
    result_path = tmp_path / "out" / "result.json"
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files.frames.iterdir()
    }

    _read_success(_run_cli(files, result_path, side="P1"), result_path)

    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files.frames.iterdir()
    }
    assert after == before
    assert sorted(path.name for path in files.frames.iterdir()) == sorted(before)
    assert not any(
        path.suffix.lower() in {".npy", ".png"}
        for path in result_path.parent.iterdir()
    )


@pytest.mark.parametrize(
    "collision",
    ["profile", "reference", "npy-frame", "png-frame", "frames-folder", "hardlink"],
)
def test_replay_cli_refuses_result_paths_that_overlap_any_input(
    tmp_path: Path,
    collision: str,
) -> None:
    suffix = ".png" if collision == "png-frame" else ".npy"
    files = _write_replay(
        tmp_path,
        p1=["full"] * 6 + ["spent"] * 3,
        p2=["full"] * 9,
        suffix=suffix,
    )
    frame_paths = sorted(files.frames.iterdir())
    reference = files.profile.parent / "references" / "p1.npz"
    if collision == "profile":
        result_path = files.profile
    elif collision == "reference":
        result_path = reference
    elif collision in {"npy-frame", "png-frame"}:
        result_path = frame_paths[0]
    elif collision == "frames-folder":
        result_path = files.frames / "result.json"
    else:
        result_path = tmp_path / "profile-hardlink.json"
        os.link(files.profile, result_path)

    before = _input_hashes(files)
    frame_names = sorted(path.name for path in files.frames.iterdir())

    completed = _run_cli(files, result_path, side="P1")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "result path must not overlap replay inputs" in completed.stderr.lower()
    assert _input_hashes(files) == before
    assert sorted(path.name for path in files.frames.iterdir()) == frame_names
    for private_text in (str(result_path), result_path.name):
        assert private_text not in completed.stderr


def test_replay_cli_refuses_frames_below_ignored_source_capture_directory(
    tmp_path: Path,
) -> None:
    files = _write_replay(
        tmp_path,
        p1=["full"],
        p2=["full"],
    )
    capture_root = REPO_ROOT / "screenshots"
    capture_root_preexisted = capture_root.exists()
    ignored = capture_root / f"task10-test-{uuid.uuid4().hex}"
    ignored.mkdir(parents=True, exist_ok=False)
    try:
        frame = np.concatenate(
            (_meter_pattern("full"), _meter_pattern("full")),
            axis=1,
        )
        np.save(ignored / "000000-000.npy", frame, allow_pickle=False)
        result_path = tmp_path / "result.json"

        completed = _run_cli(
            ReplayFiles(files.profile, ignored),
            result_path,
            side="P1",
        )

        assert completed.returncode != 0
        assert not result_path.exists()
        assert "ignored source capture directory" in completed.stderr.lower()
    finally:
        shutil.rmtree(ignored)
        if not capture_root_preexisted:
            capture_root.rmdir()


def test_replay_cli_rejects_empty_or_malformed_frame_folders(tmp_path: Path) -> None:
    files = _write_replay(tmp_path, p1=["full"], p2=["full"])
    for path in files.frames.iterdir():
        path.unlink()
    result_path = tmp_path / "result.json"

    empty = _run_cli(files, result_path, side="P1")
    assert empty.returncode != 0
    assert not result_path.exists()
    assert "no replay frames" in empty.stderr.lower()

    np.save(files.frames / "frame-without-id.npy", _meter_pattern("full"))
    malformed = _run_cli(files, result_path, side="P1")
    assert malformed.returncode != 0
    assert not result_path.exists()
    assert "numeric frame id" in malformed.stderr.lower()


def test_replay_cli_configuration_failure_does_not_expose_local_paths(
    tmp_path: Path,
) -> None:
    files = _write_replay(tmp_path, p1=["full"], p2=["full"])
    missing_reference = files.profile.parent / "references" / "p1.npz"
    missing_reference.unlink()
    result_path = tmp_path / "result.json"

    completed = _run_cli(files, result_path, side="P1")

    assert completed.returncode != 0
    assert not result_path.exists()
    assert completed.stdout == ""
    assert "profile or meter references are invalid" in completed.stderr.lower()
    for private_text in (
        str(files.profile),
        files.profile.name,
        str(missing_reference),
        missing_reference.name,
    ):
        assert private_text not in completed.stderr


def test_replay_cli_truncated_frame_failure_is_path_free(tmp_path: Path) -> None:
    files = _write_replay(tmp_path, p1=["full"], p2=["full"])
    truncated = next(files.frames.iterdir())
    truncated.write_bytes(b"")
    result_path = tmp_path / "result.json"

    completed = _run_cli(files, result_path, side="P1")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "replay frame could not be decoded" in completed.stderr.lower()
    assert "traceback" not in completed.stderr.lower()
    assert str(truncated) not in completed.stderr
    assert truncated.name not in completed.stderr
    assert not result_path.exists()


@pytest.mark.parametrize("failure", ["directory", "parent-is-file"])
def test_replay_cli_output_failure_is_path_free_and_preserves_inputs(
    tmp_path: Path,
    failure: str,
) -> None:
    files = _write_replay(tmp_path, p1=["full"], p2=["full"])
    before = _input_hashes(files)
    if failure == "directory":
        result_path = tmp_path / "private-result-directory"
        result_path.mkdir()
    else:
        blocked_parent = tmp_path / "private-result-parent"
        blocked_parent.write_text("not a directory", encoding="utf-8")
        result_path = blocked_parent / "result.json"

    completed = _run_cli(files, result_path, side="P1")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "result could not be written" in completed.stderr.lower()
    assert "traceback" not in completed.stderr.lower()
    assert str(result_path) not in completed.stderr
    assert result_path.name not in completed.stderr
    assert _input_hashes(files) == before
