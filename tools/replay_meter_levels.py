"""Bounded offline replay for SF6 numeric meter-level transitions.

The tool uses the production observation and tracker boundary.  It emits only
scalar event diagnostics and never writes frames, crops, templates, or audio
configuration.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sf6_music_trigger.config import load_config
from sf6_music_trigger.domain import ConfirmedSuperEvent, Side
from sf6_music_trigger.engine import profile_applies_to
from sf6_music_trigger.gauge_features import GaugeObservation, observe_gauge
from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig
from sf6_music_trigger.hud_profile import load_profile
from sf6_music_trigger.meter_references import load_references


class MeterReplayError(RuntimeError):
    """A privacy-safe, user-actionable replay failure."""


class MeterLevelReplay:
    """Replay paired production observations through independent trackers."""

    def __init__(self, profile_id: str) -> None:
        if type(profile_id) is not str or not profile_id.strip():
            raise ValueError("profile_id must be nonblank text")
        config = GaugeTrackerConfig()
        self._trackers = {
            Side.P1: GaugeTracker(Side.P1, profile_id, config),
            Side.P2: GaugeTracker(Side.P2, profile_id, config),
        }
        self._events: list[ConfirmedSuperEvent] = []
        self._last_frame_id: int | None = None
        self._last_sampled_at: float | None = None

    @property
    def events(self) -> tuple[ConfirmedSuperEvent, ...]:
        return tuple(self._events)

    def update(
        self,
        observations: tuple[GaugeObservation, GaugeObservation],
    ) -> tuple[ConfirmedSuperEvent, ...]:
        if type(observations) is not tuple or len(observations) != 2:
            raise TypeError("observations must be a P1/P2 pair")
        p1, p2 = observations
        if not isinstance(p1, GaugeObservation) or not isinstance(p2, GaugeObservation):
            raise TypeError("observation pair must contain GaugeObservation values")
        if p1.side is not Side.P1 or p2.side is not Side.P2:
            raise ValueError("observation pair must be ordered P1 then P2")
        if p1.frame_id != p2.frame_id or p1.sampled_at != p2.sampled_at:
            raise ValueError("paired observations must share frame identity and time")

        sampled_at = float(p1.sampled_at)
        if (
            self._last_frame_id is not None
            and (
                p1.frame_id <= self._last_frame_id
                or sampled_at <= self._last_sampled_at
            )
        ):
            self._reset_trackers("stale replay frame")
            return ()
        self._last_frame_id = p1.frame_id
        self._last_sampled_at = sampled_at

        updates = (
            self._trackers[Side.P1].update(p1),
            self._trackers[Side.P2].update(p2),
        )
        events = tuple(
            update.transition
            for update in updates
            if isinstance(update.transition, ConfirmedSuperEvent)
        )
        self._events.extend(events)
        return events

    def reset(self, reason: str) -> None:
        self._reset_trackers(reason)
        self._last_frame_id = None
        self._last_sampled_at = None

    def _reset_trackers(self, reason: str) -> None:
        for tracker in self._trackers.values():
            tracker.reset(reason)


def replay_observation_pairs(
    observations: Iterable[tuple[GaugeObservation, GaugeObservation]],
    *,
    profile_id: str,
) -> tuple[ConfirmedSuperEvent, ...]:
    replay = MeterLevelReplay(profile_id)
    for pair in observations:
        replay.update(pair)
    return replay.events


def _load_inputs(profile_root: Path, profile_path: str):
    try:
        root = profile_root.resolve(strict=True)
        if not root.is_dir():
            raise OSError
        profile = load_profile(root, profile_path)
        p1 = load_references(
            root,
            profile_path,
            profile.p1.references_path,
            expected_side=Side.P1,
        )
        p2 = load_references(
            root,
            profile_path,
            profile.p2.references_path,
            expected_side=Side.P2,
        )
    except (OSError, TypeError, ValueError):
        raise MeterReplayError("profile or meter references are invalid") from None
    if p1.color_model is None or p2.color_model is None:
        raise MeterReplayError("numeric meter replay requires colour references")
    config_path = root / "config.json"
    sample_fps = 20
    if config_path.is_file():
        try:
            sample_fps = load_config(config_path).capture_fps
        except (OSError, TypeError, ValueError):
            raise MeterReplayError("capture configuration is invalid") from None
    return profile, p1, p2, sample_fps


def _write_csv(path: Path, events: tuple[ConfirmedSuperEvent, ...]) -> None:
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(("time", "side", "before", "after", "levels_spent"))
            for event in events:
                writer.writerow(
                    (
                        f"{float(event.confirmed_at):.6f}",
                        event.side.value,
                        event.meter_before,
                        event.meter_after,
                        event.levels_spent,
                    )
                )
        temporary.replace(target)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise MeterReplayError("CSV output could not be written") from None


def replay_video(
    video_path: Path,
    *,
    profile_root: Path,
    profile_path: str,
) -> tuple[tuple[ConfirmedSuperEvent, ...], int, int]:
    profile, p1_references, p2_references, sample_fps = _load_inputs(
        profile_root, profile_path
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise MeterReplayError("video could not be opened")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(source_fps) or source_fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise MeterReplayError("video metadata is invalid")

    from sf6_music_trigger.domain import ClientGeometry

    geometry = ClientGeometry(0, 0, width, height)
    if not profile_applies_to(profile, geometry):
        capture.release()
        raise MeterReplayError("video aspect ratio does not match the profile")

    replay = MeterLevelReplay(profile.profile_id)
    source_index = 0
    processed = 0
    next_source_index = 0
    try:
        while True:
            ok = capture.grab()
            if not ok:
                break
            if source_index >= next_source_index:
                ok, frame = capture.retrieve()
                if not ok:
                    raise MeterReplayError("video frame could not be decoded")
                sampled_at = source_index / source_fps
                pair = (
                    observe_gauge(
                        frame,
                        side=Side.P1,
                        frame_id=processed,
                        roi=profile.p1.meter_roi,
                        references=p1_references,
                        sampled_at=sampled_at,
                    ),
                    observe_gauge(
                        frame,
                        side=Side.P2,
                        frame_id=processed,
                        roi=profile.p2.meter_roi,
                        references=p2_references,
                        sampled_at=sampled_at,
                    ),
                )
                replay.update(pair)
                processed += 1
                next_source_index = round(processed * source_fps / sample_fps)
            source_index += 1
    finally:
        capture.release()
    if processed == 0:
        raise MeterReplayError("video contained no decodable replay frames")
    return replay.events, processed, sample_fps


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--expect-count", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expect_count is not None and args.expect_count < 0:
        print("meter replay failed: expected count must be nonnegative", file=sys.stderr)
        return 2
    try:
        events, processed, sample_fps = replay_video(
            args.video,
            profile_root=args.profile_root,
            profile_path=args.profile,
        )
        if args.csv is not None:
            _write_csv(args.csv, events)
        if args.expect_count is not None and len(events) != args.expect_count:
            raise MeterReplayError(
                f"detected {len(events)} events; expected {args.expect_count}"
            )
    except MeterReplayError as error:
        print(f"meter replay failed: {error}", file=sys.stderr)
        return 2

    print(f"processed_frames={processed} sample_fps={sample_fps} events={len(events)}")
    for event in events:
        print(
            "event="
            f"{float(event.confirmed_at):.6f},"
            f"{event.side.value},"
            f"{event.meter_before},"
            f"{event.meter_after},"
            f"{event.levels_spent}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
