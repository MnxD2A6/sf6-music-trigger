"""Offline replay of local synthetic or user-provided SF6 frame sequences."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sf6_music_trigger.audio import AudioAction, AudioKind, AudioScheduler
from sf6_music_trigger.capture import FreshFrameGate
from sf6_music_trigger.config import SideAudioConfig
from sf6_music_trigger.domain import (
    CaptureSample,
    CaptureStatus,
    CapturedFrame,
    ClientGeometry,
    ConfirmedSuperEvent,
    Side,
)
from sf6_music_trigger.diagnostic_log import DiagnosticRecord
from sf6_music_trigger.engine import VisualTriggerEngine
from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig
from sf6_music_trigger.hud_profile import load_profile, resolve_contained_path
from sf6_music_trigger.meter_references import load_references
from sf6_music_trigger.outcome import OutcomeDecision, OutcomeObserver


_FRAME_SUFFIXES = frozenset({".npy", ".png"})
_FRAME_ID_PATTERN = re.compile(r"^(\d+)(?:[-_.]|$)")
_IGNORED_SOURCE_MEDIA_DIRS = (
    "diagnostic_captures",
    "music",
    "screenshots",
)


class ReplayError(RuntimeError):
    """A safe, user-actionable replay input failure."""


class _ReplayClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _CountingFreshFrameGate(FreshFrameGate):
    def __init__(self) -> None:
        super().__init__()
        self.rejected_frames = 0

    def accept(self, frame: CapturedFrame, *, now: float) -> bool:
        accepted = super().accept(frame, now=now)
        if not accepted:
            self.rejected_frames += 1
        return accepted


class _RecordingAudioPort:
    def __init__(self, clock: _ReplayClock) -> None:
        self._clock = clock
        self.counts = {
            Side.P1: {kind: 0 for kind in AudioKind},
            Side.P2: {kind: 0 for kind in AudioKind},
        }
        self.batches: list[tuple[float, tuple[AudioAction, ...]]] = []

    def execute(self, actions: tuple[AudioAction, ...]) -> None:
        self.batches.append((self._clock.value, actions))
        for action in actions:
            self.counts[action.side][action.kind] += 1


class _RecordingOutcomeObserver(OutcomeObserver):
    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[OutcomeDecision] = []

    def update(self, *args: object, **kwargs: object) -> tuple[OutcomeDecision, ...]:
        decisions = super().update(*args, **kwargs)  # type: ignore[arg-type]
        self.decisions.extend(decisions)
        return decisions

    def reset(self, reason: str) -> tuple[OutcomeDecision, ...]:
        decisions = super().reset(reason)
        self.decisions.extend(decisions)
        return decisions


def _positive_fps(value: str) -> float:
    try:
        fps = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("fps must be a number") from error
    if not math.isfinite(fps) or fps <= 0:
        raise argparse.ArgumentTypeError("fps must be finite and positive")
    return fps


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--side", required=True, choices=("P1", "P2", "both"))
    parser.add_argument("--fps", required=True, type=_positive_fps)
    parser.add_argument("--json", required=True, type=Path, dest="result_path")
    return parser.parse_args(argv)


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validated_frames_folder(value: Path) -> Path:
    try:
        frames = value.resolve(strict=True)
    except OSError as error:
        raise ReplayError("frames folder does not exist") from error
    if not frames.is_dir():
        raise ReplayError("frames path must be a folder")

    source_tree = PROJECT_ROOT.resolve(strict=True)
    for directory in _IGNORED_SOURCE_MEDIA_DIRS:
        ignored = (source_tree / directory).resolve(strict=False)
        if _is_below(frames, ignored):
            raise ReplayError(
                "frames path is inside an ignored source capture directory"
            )
    return frames


def _ordered_frame_paths(folder: Path) -> list[tuple[int, Path]]:
    paths = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in _FRAME_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not paths:
        raise ReplayError("no replay frames were found")

    ordered: list[tuple[int, Path]] = []
    prefix_widths: set[int] = set()
    for path in paths:
        match = _FRAME_ID_PATTERN.match(path.stem)
        if match is None:
            raise ReplayError("every replay filename must begin with a numeric frame id")
        frame_id_text = match.group(1)
        prefix_widths.add(len(frame_id_text))
        ordered.append((int(frame_id_text), path))
    if len(prefix_widths) != 1:
        raise ReplayError("replay filenames must use fixed-width numeric frame ids")
    return ordered


def _load_frame(path: Path) -> np.ndarray:
    try:
        if path.suffix.lower() == ".npy":
            loaded = np.load(path, allow_pickle=False)
        else:
            loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except (OSError, EOFError, TypeError, ValueError, cv2.error) as error:
        raise ReplayError("replay frame could not be decoded") from error
    if not isinstance(loaded, np.ndarray):
        raise ReplayError("replay frame could not be decoded")
    if loaded.dtype != np.dtype(np.uint8):
        raise ReplayError("replay frames must use uint8 BGR pixels")
    if (
        loaded.ndim != 3
        or loaded.shape[0] == 0
        or loaded.shape[1] == 0
        or loaded.shape[2] != 3
    ):
        raise ReplayError("replay frames must be nonempty H x W x 3 BGR arrays")
    return np.array(loaded, dtype=np.uint8, order="C", copy=True)


def _selected_sides(value: str) -> frozenset[Side]:
    if value == "P1":
        return frozenset({Side.P1})
    if value == "P2":
        return frozenset({Side.P2})
    return frozenset({Side.P1, Side.P2})


def _audio_config(side: Side, selected: frozenset[Side]) -> SideAudioConfig:
    if side not in selected:
        return SideAudioConfig(enabled=False)
    return SideAudioConfig(
        enabled=True,
        main_track=f"C:\\offline-replay\\{side.value.lower()}.wav",
        volume=1.0,
    )


def _build_engine(
    profile_path: Path,
    selected: frozenset[Side],
    *,
    diagnostic_sink: Callable[[DiagnosticRecord], None] | None = None,
) -> tuple[
    VisualTriggerEngine,
    _ReplayClock,
    _CountingFreshFrameGate,
    _RecordingAudioPort,
    _RecordingOutcomeObserver,
    tuple[Path, ...],
]:
    try:
        resolved_profile = profile_path.resolve(strict=True)
    except OSError as error:
        raise ReplayError("profile does not exist") from error
    if not resolved_profile.is_file():
        raise ReplayError("profile path must identify a file")
    user_root = resolved_profile.parent
    relative_profile = resolved_profile.name
    try:
        profile = load_profile(user_root, relative_profile)
        p1_references = load_references(
            user_root,
            relative_profile,
            profile.p1.references_path,
            expected_side=Side.P1,
        )
        p2_references = load_references(
            user_root,
            relative_profile,
            profile.p2.references_path,
            expected_side=Side.P2,
        )
    except (OSError, TypeError, ValueError):
        raise ReplayError("profile or meter references are invalid") from None
    reference_paths = (
        resolve_contained_path(
            user_root,
            profile.p1.references_path,
            field_name="p1.references_path",
        ).resolve(strict=True),
        resolve_contained_path(
            user_root,
            profile.p2.references_path,
            field_name="p2.references_path",
        ).resolve(strict=True),
    )
    clock = _ReplayClock()
    gate = _CountingFreshFrameGate()
    audio = _RecordingAudioPort(clock)
    outcomes = _RecordingOutcomeObserver()
    engine = VisualTriggerEngine(
        profile=profile,
        p1_references=p1_references,
        p2_references=p2_references,
        p1_tracker=GaugeTracker(Side.P1, profile.profile_id, GaugeTrackerConfig()),
        p2_tracker=GaugeTracker(Side.P2, profile.profile_id, GaugeTrackerConfig()),
        freshness_gate=gate,
        p1_scheduler=AudioScheduler(Side.P1),
        p2_scheduler=AudioScheduler(Side.P2),
        p1_audio_config=_audio_config(Side.P1, selected),
        p2_audio_config=_audio_config(Side.P2, selected),
        audio_port=audio,
        audio_diagnostic=lambda _error: None,
        diagnostic_sink=diagnostic_sink or (lambda _record: None),
        outcome_observer=outcomes,
        clock=clock,
    )
    return engine, clock, gate, audio, outcomes, (resolved_profile, *reference_paths)


def _validated_result_path(
    value: Path,
    frames_folder: Path,
    input_paths: tuple[Path, ...],
) -> Path:
    try:
        result = value.resolve(strict=False)
        if _is_below(result, frames_folder):
            raise ReplayError("result path must not overlap replay inputs")
        for input_path in input_paths:
            if result == input_path or (
                result.exists() and os.path.samefile(result, input_path)
            ):
                raise ReplayError("result path must not overlap replay inputs")
    except ReplayError:
        raise
    except OSError:
        raise ReplayError("result path could not be validated") from None
    return result


def _event_json(event: ConfirmedSuperEvent) -> dict[str, float | int | str]:
    return {
        "side": event.side.value,
        "pre_spend_frame_id": event.pre_spend_frame_id,
        "confirmed_frame_id": event.confirmed_frame_id,
        "confirmed_at": float(event.confirmed_at),
    }


def _audio_json(audio: _RecordingAudioPort) -> dict[str, dict[str, int]]:
    return {
        side.value: {
            AudioKind.STOP.value: audio.counts[side][AudioKind.STOP],
            AudioKind.PLAY.value: audio.counts[side][AudioKind.PLAY],
            AudioKind.FADE.value: audio.counts[side][AudioKind.FADE],
        }
        for side in (Side.P1, Side.P2)
    }


def _outcome_json(decision: OutcomeDecision) -> dict[str, object]:
    return {
        "side": decision.side.value,
        "label": decision.label.value,
        "decided_at": float(decision.decided_at),
        "evidence": list(decision.evidence),
    }


def _action_json(action: AudioAction) -> dict[str, object]:
    return {
        "kind": action.kind.value,
        "side": action.side.value,
        "volume": action.volume,
        "fade_ms": action.fade_ms,
    }


def _batch_json(audio: _RecordingAudioPort) -> list[dict[str, object]]:
    return [
        {
            "at": float(timestamp),
            "actions": [_action_json(action) for action in actions],
        }
        for timestamp, actions in audio.batches
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    parent = path.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    frames_folder = _validated_frames_folder(arguments.frames)
    ordered_frames = _ordered_frame_paths(frames_folder)
    selected = _selected_sides(arguments.side)
    diagnostics: list[DiagnosticRecord] = []
    engine, clock, gate, audio, outcomes, configuration_inputs = _build_engine(
        arguments.profile,
        selected,
        diagnostic_sink=diagnostics.append,
    )
    frame_inputs = tuple(path.resolve(strict=True) for _, path in ordered_frames)
    result_path = _validated_result_path(
        arguments.result_path,
        frames_folder,
        configuration_inputs + frame_inputs,
    )
    selected_events: list[ConfirmedSuperEvent] = []
    lifecycle: list[dict[str, object]] = []
    pending_meter_by_identity: dict[tuple[Side, str], tuple[int, int]] = {}

    for sequence_index, (frame_id, path) in enumerate(ordered_frames):
        image = _load_frame(path)
        timestamp = sequence_index / arguments.fps
        clock.value = timestamp
        geometry = ClientGeometry(0, 0, image.shape[1], image.shape[0])
        diagnostic_start = len(diagnostics)
        update = engine.process(
            CaptureSample(
                CaptureStatus.RUNNING,
                frame=CapturedFrame(frame_id, image, timestamp, geometry),
            )
        )
        selected_events.extend(
            event for event in update.events if event.side in selected
        )
        for record in diagnostics[diagnostic_start:]:
            if record.transition == "OUTCOME":
                continue
            if record.transition == "CANDIDATE":
                assert record.meter_before is not None
                assert record.meter_after is not None
                pending_meter_by_identity[(record.side, record.identity)] = (
                    record.meter_before,
                    record.meter_after,
                )
            meter_before = record.meter_before
            meter_after = record.meter_after
            if record.transition == "CANCELLED":
                meter_before, meter_after = pending_meter_by_identity.get(
                    (record.side, record.identity),
                    (None, None),
                )
            if record.side in selected:
                lifecycle.append(
                    {
                        "frame_id": frame_id,
                        "side": record.side.value,
                        "transition": record.transition,
                        "meter_before": meter_before,
                        "meter_after": meter_after,
                        "audio_actions": [kind.value for kind in record.audio_actions],
                    }
                )
            if record.transition in {"CONFIRMED", "CANCELLED"}:
                pending_meter_by_identity.pop(
                    (record.side, record.identity),
                    None,
                )

    payload: dict[str, object] = {
        "processed_frames": len(ordered_frames),
        "rejected_stale_frames": gate.rejected_frames,
        "events": [_event_json(event) for event in selected_events],
        "lifecycle": lifecycle,
        "audio_actions": _audio_json(audio),
        "outcomes": [
            _outcome_json(decision)
            for decision in outcomes.decisions
            if decision.side in selected
        ],
        "audio_batches": _batch_json(audio),
    }
    try:
        _write_json(result_path, payload)
    except OSError:
        raise ReplayError("result could not be written") from None
    return 0


def main() -> int:
    try:
        return run()
    except ReplayError as error:
        print(f"replay failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
