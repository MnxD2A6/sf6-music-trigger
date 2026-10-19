"""Private real-video regression for R31's unreadable P1 level glyph."""
from pathlib import Path

import cv2
import pytest

from sf6_music_trigger.domain import ConfirmedSuperEvent, Side
from sf6_music_trigger.gauge_features import observe_gauge
from sf6_music_trigger.gauge_tracker import GaugeTracker, GaugeTrackerConfig
from sf6_music_trigger.hud_profile import load_profile
from sf6_music_trigger.meter_references import load_references


def _profile_and_p1_references():
    root = (
        Path(__file__).resolve().parents[2]
        / "SF6_Music_Trigger_ColorCandidate_20260816"
    )
    profile_path = "profiles/screenshot-sf6-color-20260816.json"
    if not (root / profile_path).is_file():
        pytest.skip("local calibrated colour profile fixture is absent")
    profile = load_profile(root, profile_path)
    references = load_references(
        root,
        profile_path,
        profile.p1.references_path,
        expected_side=Side.P1,
    )
    return profile, references


def _replay_p1(path, start, end, crop):
    if not path.is_file():
        pytest.skip(f"local golden video absent: {path.name}")
    profile, references = _profile_and_p1_references()
    tracker = GaugeTracker(Side.P1, profile.profile_id, GaugeTrackerConfig())
    capture = cv2.VideoCapture(str(path))
    transitions = []
    try:
        for frame_id, source_frame in enumerate(range(start * 30, end * 30, 2)):
            assert capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = capture.read()
            assert ok and image is not None
            observation = observe_gauge(
                crop(image),
                side=Side.P1,
                frame_id=frame_id,
                roi=profile.p1.meter_roi,
                references=references,
                sampled_at=source_frame / 30,
            )
            update = tracker.update(observation)
            if update.transition is not None:
                transitions.append(update.transition)
    finally:
        capture.release()
    return transitions


def test_r31_visible_p1_ca_triggers_while_r30_round_intro_stays_silent():
    video_root = Path.home() / "Videos" / "屏幕录制" / "SF6"

    r31 = _replay_p1(
        video_root / "R31.mp4",
        1,
        8,
        lambda image: image[140:1220, 466:2386],
    )
    confirmed = [item for item in r31 if isinstance(item, ConfirmedSuperEvent)]
    assert [(item.meter_before, item.meter_after) for item in confirmed] == [(3, 0)]

    r30_intro = _replay_p1(
        video_root / "R30 01.mp4",
        0,
        8,
        lambda image: image[26:],
    )
    assert not [item for item in r30_intro if isinstance(item, ConfirmedSuperEvent)]
