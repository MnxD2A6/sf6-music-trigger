from __future__ import annotations

from collections import deque
from fractions import Fraction
import inspect
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import tomllib
from typing import Any

import numpy as np
import pytest

from sf6_music_trigger.audio import AudioIssue, AudioPlaybackError
from sf6_music_trigger.config import AppConfig, DetectionBackend, SideAudioConfig, save_config
from sf6_music_trigger.domain import (
    CaptureIssue,
    CaptureSample,
    CaptureStatus,
    CapturedFrame,
    ClientGeometry,
    Side,
    TrackerState,
)
from sf6_music_trigger.hud_profile import (
    CalibrationContext,
    HudProfile,
    NormalizedRoi,
    SideHudProfile,
    save_profile,
)
from sf6_music_trigger.meter_references import (
    MeterReferences,
    save_references,
)
from sf6_music_trigger.meter_color_features import MeterColorModel


PROFILE_PATH = "profiles/default.json"
P1_REFERENCE_PATH = "refs/p1.npz"
P2_REFERENCE_PATH = "refs/p2.npz"


def _context() -> CalibrationContext:
    return CalibrationContext(
        client_width=200,
        client_height=100,
        client_aspect_ratio=2.0,
        hud_scale_setting="default",
        language="English",
        brightness_setting="default",
        hdr_enabled=False,
        game_version_note="offline-test",
    )


def _profile() -> HudProfile:
    return HudProfile(
        profile_id="task9-profile",
        calibration=_context(),
        battle_roi=NormalizedRoi(0.0, 0.0, 1.0, 1.0),
        training_hud_roi=None,
        p1=SideHudProfile(
            NormalizedRoi(0.05, 0.05, 0.35, 0.25),
            P1_REFERENCE_PATH,
        ),
        p2=SideHudProfile(
            NormalizedRoi(0.65, 0.05, 0.95, 0.25),
            P2_REFERENCE_PATH,
        ),
    )


def _references(side: Side) -> MeterReferences:
    return MeterReferences(
        side,
        np.zeros((32, 64), dtype=np.uint8),
        np.full((32, 64), 255, dtype=np.uint8),
    )


def _write_valid_profile(root: Path) -> HudProfile:
    profile = _profile()
    save_references(root, PROFILE_PATH, P1_REFERENCE_PATH, _references(Side.P1))
    save_references(root, PROFILE_PATH, P2_REFERENCE_PATH, _references(Side.P2))
    save_profile(root, PROFILE_PATH, profile)
    return profile


class _Audio:
    def __init__(self, error: AudioPlaybackError | None = None) -> None:
        self.error = error
        self.preview_calls: list[tuple[str, float]] = []
        self.side_batches: list[object] = []

    def preview(self, track: str, volume: float) -> None:
        self.preview_calls.append((track, volume))
        if self.error is not None:
            raise self.error


def test_view_model_requires_valid_profile_references_and_one_existing_track(
    tmp_path: Path,
) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    vm = PanelViewModel(user_dir=tmp_path)
    vm.detection_backend = DetectionBackend.SCREEN
    vm.profile_path = PROFILE_PATH
    vm.p1_main_track = ""
    assert vm.can_start is False
    assert "HUD" in vm.validation_message

    _write_valid_profile(tmp_path)
    vm.p1_main_track = "audio/p1.wav"
    assert vm.can_start is False
    assert "音频" in vm.validation_message

    track = tmp_path / "audio" / "p1.wav"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"offline-audio-placeholder")
    assert vm.can_start is True
    assert vm.validation_message == ""
    assert vm.selected_profile == _profile()
    assert vm.calibration_context == _context()


def test_wrong_side_reference_keeps_start_disabled(tmp_path: Path) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    profile = _profile()
    save_references(tmp_path, PROFILE_PATH, P1_REFERENCE_PATH, _references(Side.P2))
    save_references(tmp_path, PROFILE_PATH, P2_REFERENCE_PATH, _references(Side.P2))
    save_profile(tmp_path, PROFILE_PATH, profile)
    track = tmp_path / "audio" / "p1.wav"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"x")

    vm = PanelViewModel(user_dir=tmp_path)
    vm.detection_backend = DetectionBackend.SCREEN
    vm.profile_path = PROFILE_PATH
    vm.p1_main_track = "audio/p1.wav"
    assert vm.can_start is False
    assert "HUD" in vm.validation_message


def test_current_geometry_mismatch_requires_recalibration(tmp_path: Path) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    _write_valid_profile(tmp_path)
    track = tmp_path / "audio" / "p1.wav"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"x")
    vm = PanelViewModel(user_dir=tmp_path)
    vm.detection_backend = DetectionBackend.SCREEN
    vm.profile_path = PROFILE_PATH
    vm.p1_main_track = "audio/p1.wav"
    vm.current_geometry = ClientGeometry(0, 0, 160, 100)

    assert vm.can_start is False
    assert "重新校准" in vm.validation_message


@pytest.mark.parametrize(
    "geometry",
    [
        ClientGeometry(0, 0, 1920, 1200),
        ClientGeometry(0, 0, 2560, 1080),
    ],
)
def test_large_mismatched_geometry_can_start_for_runtime_viewport_confirmation(
    tmp_path: Path,
    geometry: ClientGeometry,
) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    _write_valid_profile(tmp_path)
    track = tmp_path / "audio" / "p1.wav"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"x")
    vm = PanelViewModel(user_dir=tmp_path)
    vm.detection_backend = DetectionBackend.SCREEN
    vm.profile_path = PROFILE_PATH
    vm.p1_main_track = "audio/p1.wav"
    vm.current_geometry = geometry

    assert vm.can_start is True
    assert vm.validation_message == ""


@pytest.mark.parametrize(
    ("percent", "expected"),
    [(0, 0.0), (35, 0.35), (100, 1.0), (Fraction(25, 2), 0.125)],
)
def test_preview_normalizes_percentage_without_touching_side_batches(
    tmp_path: Path,
    percent: object,
    expected: float,
) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    audio = _Audio()
    vm = PanelViewModel(user_dir=tmp_path, audio_port=audio)
    vm.p1_main_track = "audio/p1.mp3"
    vm.p1_volume_percent = percent

    assert vm.preview(Side.P1) is True
    assert audio.preview_calls == [("audio/p1.mp3", expected)]
    assert audio.side_batches == []


@pytest.mark.parametrize(
    "percent",
    [True, False, None, "35", -0.01, 500.01, float("nan"), float("inf")],
)
def test_preview_rejects_invalid_percentage(tmp_path: Path, percent: object) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    audio = _Audio()
    vm = PanelViewModel(user_dir=tmp_path, audio_port=audio)
    vm.p1_main_track = "audio/p1.mp3"
    vm.p1_volume_percent = percent
    with pytest.raises((TypeError, ValueError)):
        vm.preview(Side.P1)
    assert audio.preview_calls == []


def test_preview_failure_becomes_safe_status_without_side_playback(tmp_path: Path) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    error = AudioPlaybackError("load", None, AudioIssue.TRACK_REJECTED)
    audio = _Audio(error)
    statuses: list[AudioPlaybackError] = []
    vm = PanelViewModel(user_dir=tmp_path, audio_port=audio, status_sink=statuses.append)
    vm.p2_main_track = "audio/private-p2.mp3"
    vm.p2_volume_percent = 50

    assert vm.preview(Side.P2) is False
    assert statuses == [error]
    assert audio.side_batches == []


def test_view_model_builds_typed_config_with_percentage_bridge(tmp_path: Path) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    vm = PanelViewModel(user_dir=tmp_path)
    vm.profile_path = PROFILE_PATH
    vm.p1_enabled = True
    vm.p2_enabled = False
    vm.p1_main_track = "audio/p1.wav"
    vm.p2_main_track = ""
    vm.p1_volume_percent = 35
    vm.p2_volume_percent = 100

    config = vm.to_config()
    assert config == AppConfig(
        p1=SideAudioConfig(True, "audio/p1.wav", 0.35),
        p2=SideAudioConfig(False, "", 1.0),
        profile_path=PROFILE_PATH,
    )


def test_view_model_defaults_both_sides_to_sa3_ca_only(tmp_path: Path) -> None:
    from sf6_music_trigger.config import SuperScope
    from sf6_music_trigger.panel import PanelViewModel

    vm = PanelViewModel(user_dir=tmp_path)

    assert vm.p1_super_scope is SuperScope.SA3_CA_ONLY
    assert vm.p2_super_scope is SuperScope.SA3_CA_ONLY


def test_panel_round_trips_independent_super_scopes(tmp_path: Path) -> None:
    from sf6_music_trigger.config import SuperScope
    from sf6_music_trigger.panel import PanelViewModel

    vm = PanelViewModel(user_dir=tmp_path)
    vm.p1_super_scope = SuperScope.ALL_SUPERS
    vm.p2_super_scope = SuperScope.SA3_CA_ONLY

    config = vm.to_config()

    assert config.p1.super_scope is SuperScope.ALL_SUPERS
    assert config.p2.super_scope is SuperScope.SA3_CA_ONLY


def test_view_model_exposes_commands_and_typed_runtime_diagnostics(tmp_path: Path) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    calls: list[str] = []
    vm = PanelViewModel(user_dir=tmp_path)
    vm.start_command = lambda: calls.append("start")
    vm.stop_command = lambda: calls.append("stop")
    vm.exit_command = lambda: calls.append("exit")
    vm.calibration_command = lambda: calls.append("calibrate")
    geometry = ClientGeometry(10, 20, 200, 100)

    vm.start()
    vm.stop()
    vm.calibrate()
    vm.exit()
    vm.update_runtime_diagnostics(
        capture_status=CaptureStatus.RUNNING,
        current_geometry=geometry,
        tracker_states=(TrackerState.ARMED, TrackerState.COOLDOWN),
        tracker_levels=(2, None),
    )

    assert calls == ["start", "stop", "calibrate", "exit"]
    assert vm.capture_status is CaptureStatus.RUNNING
    assert vm.current_geometry == geometry
    assert vm.p1_tracker_state is TrackerState.ARMED
    assert vm.p2_tracker_state is TrackerState.COOLDOWN
    assert vm.p1_meter_level == 2
    assert vm.p2_meter_level is None

    vm.update_runtime_diagnostics(
        capture_status=CaptureStatus.STOPPED,
        current_geometry=None,
        tracker_states=None,
        tracker_levels=None,
    )
    assert vm.p1_tracker_state is None and vm.p2_tracker_state is None
    assert vm.p1_meter_level is None and vm.p2_meter_level is None


def test_runtime_status_message_distinguishes_idle_connecting_running_and_loss() -> None:
    from sf6_music_trigger.panel import runtime_status_message

    assert "点击“开始”" in runtime_status_message(None, None, None)
    assert "正在连接" in runtime_status_message(CaptureStatus.STOPPED, None, None)
    running = runtime_status_message(
        CaptureStatus.RUNNING,
        TrackerState.ARMED,
        TrackerState.DISARMED,
        p1_level=2,
        p2_level=None,
    )
    assert "监听中" in running
    assert "P1=ARMED" in running
    assert "P2=DISARMED" in running
    assert "保持 CA" in running
    assert "0.25" in running
    assert "P1=LEVEL_2" in running
    assert "P2=LEVEL_UNKNOWN" in running
    assert "气槽等级不可读" not in running
    unreadable = runtime_status_message(
        CaptureStatus.RUNNING,
        TrackerState.DISARMED,
        TrackerState.DISARMED,
        p1_level=None,
        p2_level=None,
    )
    assert "气槽等级不可读" in unreadable
    assert "捕获暂时不可用" in runtime_status_message(
        CaptureStatus.LOST,
        TrackerState.ARMED,
        TrackerState.DISARMED,
    )


def test_calibration_preview_adds_derived_level_bands_only_for_colour_profiles() -> None:
    from sf6_music_trigger.panel import _calibration_preview_rois

    p1 = NormalizedRoi(0.10, 0.80, 0.30, 0.90)
    p2 = NormalizedRoi(0.70, 0.80, 0.90, 0.90)

    colour = _calibration_preview_rois(
        battle_roi=NormalizedRoi(0, 0, 1, 1),
        p1_roi=p1,
        p2_roi=p2,
        training_roi=None,
        supports_meter_levels=True,
    )
    grayscale = _calibration_preview_rois(
        battle_roi=NormalizedRoi(0, 0, 1, 1),
        p1_roi=p1,
        p2_roi=p2,
        training_roi=None,
        supports_meter_levels=False,
    )

    assert tuple(colour) == ("Battle", "P1", "P2", "P1 LEVEL", "P2 LEVEL")
    assert colour["P1 LEVEL"] == NormalizedRoi(0.05, 0.80, 0.14, 0.90)
    assert colour["P2 LEVEL"] == NormalizedRoi(0.86, 0.80, 0.95, 0.90)
    assert tuple(grayscale) == ("Battle", "P1", "P2")


def test_view_model_builds_mutually_exclusive_whiff_sound_or_default_fade(
    tmp_path: Path,
) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    vm = PanelViewModel(user_dir=tmp_path)
    assert vm.p1_play_whiff_sound is False
    assert vm.p1_fade_on_whiff is True

    vm.set_whiff_sound_mode(Side.P1, True)
    vm.p1_whiff_track = "audio/p1-whiff.wav"
    vm.p1_whiff_volume_percent = 42
    config = vm.to_config()

    assert vm.p1_fade_on_whiff is False
    assert config.p1.play_whiff_sound is True
    assert config.p1.whiff_track == "audio/p1-whiff.wav"
    assert config.p1.whiff_volume == 0.42
    assert config.p1.whiff_fade_ms == 1500

    vm.set_whiff_sound_mode(Side.P1, False)
    assert vm.p1_fade_on_whiff is True
    assert vm.to_config().p1.play_whiff_sound is False


def test_whiff_sound_mode_requires_existing_track_and_can_be_previewed(
    tmp_path: Path,
) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    _write_valid_profile(tmp_path)
    main = tmp_path / "audio" / "main.wav"
    main.parent.mkdir(parents=True, exist_ok=True)
    main.write_bytes(b"main")
    audio = _Audio()
    vm = PanelViewModel(user_dir=tmp_path, audio_port=audio)
    vm.detection_backend = DetectionBackend.SCREEN
    vm.profile_path = PROFILE_PATH
    vm.p1_main_track = "audio/main.wav"
    vm.p1_whiff_track = "audio/missing-whiff.wav"
    vm.p1_whiff_volume_percent = 35
    vm.set_whiff_sound_mode(Side.P1, True)

    assert vm.can_start is False

    whiff = tmp_path / "audio" / "missing-whiff.wav"
    whiff.write_bytes(b"whiff")
    assert vm.can_start is True
    assert vm.preview_whiff(Side.P1) is True
    assert audio.preview_calls == [("audio/missing-whiff.wav", 0.35)]


def test_application_restores_whiff_policy_from_config(tmp_path: Path) -> None:
    from sf6_music_trigger.panel import create_panel_application

    target = tmp_path / "app.json"
    config = AppConfig(
        p1=SideAudioConfig(
            main_track="audio/main.wav",
            whiff_track="audio/whiff.wav",
            whiff_volume=0.44,
            play_whiff_sound=True,
            whiff_fade_ms=2300,
        )
    )
    save_config(target, config)

    application = create_panel_application(
        config_path=target,
        audio_port_factory=lambda _root: _Audio(),
    )

    vm = application.view_model
    assert vm.p1_whiff_track == "audio/whiff.wav"
    assert vm.p1_whiff_volume_percent == 44
    assert vm.p1_play_whiff_sound is True
    assert vm.p1_fade_on_whiff is False
    assert vm.p1_whiff_fade_ms == 2300


def test_application_restores_independent_super_scopes_from_config(
    tmp_path: Path,
) -> None:
    from sf6_music_trigger.config import SuperScope
    from sf6_music_trigger.panel import create_panel_application

    target = tmp_path / "app.json"
    save_config(
        target,
        AppConfig(
            p1=SideAudioConfig(super_scope=SuperScope.ALL_SUPERS),
            p2=SideAudioConfig(super_scope=SuperScope.SA3_CA_ONLY),
        ),
    )

    application = create_panel_application(
        config_path=target,
        audio_port_factory=lambda _root: _Audio(),
    )

    assert application.view_model.p1_super_scope is SuperScope.ALL_SUPERS
    assert application.view_model.p2_super_scope is SuperScope.SA3_CA_ONLY


def test_audio_template_restores_audio_without_changing_trigger_policy(
    tmp_path: Path,
) -> None:
    from sf6_music_trigger.config import SuperScope
    from sf6_music_trigger.panel import PanelViewModel

    vm = PanelViewModel(user_dir=tmp_path)
    vm.p1_enabled = False
    vm.p2_enabled = True
    vm.p1_main_track = "audio/template-p1.mp3"
    vm.p2_main_track = "audio/template-p2.ogg"
    vm.p1_volume_percent = 23
    vm.p2_volume_percent = 34
    vm.p1_whiff_track = "audio/template-p1-whiff.wav"
    vm.p2_whiff_track = "audio/template-p2-whiff.wav"
    vm.p1_whiff_volume_percent = 45
    vm.p2_whiff_volume_percent = 56

    save_template = getattr(vm, "save_audio_template", None)
    apply_template = getattr(vm, "apply_audio_template", None)
    assert callable(save_template) and save_template("模板 1") is True

    vm.p1_enabled = True
    vm.p2_enabled = False
    vm.p1_main_track = "audio/current-p1.mp3"
    vm.p2_main_track = "audio/current-p2.mp3"
    vm.p1_volume_percent = 90
    vm.p2_volume_percent = 91
    vm.p1_whiff_track = ""
    vm.p2_whiff_track = ""
    vm.p1_whiff_volume_percent = 92
    vm.p2_whiff_volume_percent = 93
    vm.p1_super_scope = SuperScope.ALL_SUPERS
    vm.p2_super_scope = SuperScope.SA3_CA_ONLY
    vm.set_whiff_sound_mode(Side.P1, True)
    vm.set_whiff_sound_mode(Side.P2, False)
    vm.p1_fade_non_three = True
    vm.p2_fade_non_three = False

    assert callable(apply_template) and apply_template("模板 1") is True
    assert (
        vm.p1_enabled,
        vm.p2_enabled,
        vm.p1_main_track,
        vm.p2_main_track,
        vm.p1_volume_percent,
        vm.p2_volume_percent,
        vm.p1_whiff_track,
        vm.p2_whiff_track,
        vm.p1_whiff_volume_percent,
        vm.p2_whiff_volume_percent,
    ) == (
        False,
        True,
        "audio/template-p1.mp3",
        "audio/template-p2.ogg",
        23,
        34,
        "audio/template-p1-whiff.wav",
        "audio/template-p2-whiff.wav",
        45,
        56,
    )
    assert vm.p1_super_scope is SuperScope.ALL_SUPERS
    assert vm.p2_super_scope is SuperScope.SA3_CA_ONLY
    assert vm.p1_play_whiff_sound is True
    assert vm.p2_play_whiff_sound is False
    assert vm.p1_fade_non_three is True
    assert vm.p2_fade_non_three is False


def test_view_model_adds_and_renames_custom_audio_templates(tmp_path: Path) -> None:
    from sf6_music_trigger.panel import PanelViewModel

    vm = PanelViewModel(user_dir=tmp_path)
    vm.p1_main_track = r"D:\Music\Juri Theme.mp3"
    vm.p2_main_track = "audio/Ken.ogg"

    assert vm.add_audio_template("排位赛") is True
    assert vm.audio_template_names == ("排位赛",)
    assert vm.rename_audio_template("排位赛", "韩蛛莉") is True
    assert vm.audio_template_names == ("韩蛛莉",)
    assert vm.apply_audio_template("韩蛛莉") is True
    assert vm.p1_main_track == r"D:\Music\Juri Theme.mp3"
    assert vm.p2_main_track == "audio/Ken.ogg"


@pytest.mark.parametrize(
    ("track", "expected"),
    [
        (r"D:\Music\Juri Theme.mp3", "Juri Theme.mp3"),
        ("audio/Ken.ogg", "Ken.ogg"),
        ("", "未选择音频"),
    ],
)
def test_audio_track_display_uses_only_the_file_name(track: str, expected: str) -> None:
    from sf6_music_trigger.panel import audio_track_display_name

    assert audio_track_display_name(track) == expected


def test_runtime_status_does_not_request_arming_for_a_disabled_side() -> None:
    from sf6_music_trigger.panel import runtime_status_message

    message = runtime_status_message(
        CaptureStatus.RUNNING,
        TrackerState.DISARMED,
        TrackerState.DISARMED,
        p1_enabled=True,
        p2_enabled=False,
    )

    assert "P1=DISARMED" in message
    assert "P2=disabled" in message
    assert "P1 请保持 CA" in message
    assert "P2 请保持 CA" not in message


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        (("疑似大招，正在确认", None), "P1：疑似大招，正在确认"),
        ((None, "疑似大招已撤销"), "P2：疑似大招已撤销"),
        (
            ("疑似大招，正在确认", "疑似大招已撤销"),
            "P1：疑似大招，正在确认；P2：疑似大招已撤销",
        ),
        (("命中", "private-path"), None),
    ],
)
def test_panel_maps_only_approved_provisional_outcome_labels(
    outcomes: tuple[str | None, str | None],
    expected: str | None,
) -> None:
    from sf6_music_trigger.panel import _provisional_outcome_message

    assert _provisional_outcome_message(outcomes) == expected


def test_panel_maps_diagnostic_error_to_one_fixed_privacy_safe_warning() -> None:
    from sf6_music_trigger.controller import ControllerStatus
    from sf6_music_trigger.panel import _controller_warning_message

    message = _controller_warning_message(ControllerStatus(kind="diagnostic_error"))

    assert message == "诊断日志不可用（不影响检测）"
    assert "path" not in message.lower()


def test_panel_keeps_runtime_state_visible_when_diagnostic_log_is_unavailable() -> None:
    from sf6_music_trigger.controller import ControllerStatus
    from sf6_music_trigger.panel import _PanelStatusLatch

    state = _PanelStatusLatch()
    statuses = (ControllerStatus(kind="diagnostic_error"),)

    warning = state.resolve(
        statuses,
        fallback="监听中：capture=RUNNING；P1=DISARMED；P2=disabled",
    )
    next_refresh = state.resolve(
        (),
        fallback="监听中：capture=RUNNING；P1=ARMED；P2=disabled",
    )

    assert "capture=RUNNING" in warning
    assert "P1=DISARMED" in warning
    assert "诊断日志不可用" in warning
    assert "P1=ARMED" in next_refresh
    assert "诊断日志不可用" in next_refresh
    state.reset()
    assert state.resolve((), fallback="监听中") == "监听中"


def test_t02_diagnostic_root_uses_local_appdata_instead_of_portable_folder() -> None:
    from sf6_music_trigger.panel import diagnostic_data_root

    root = diagnostic_data_root(
        {"LOCALAPPDATA": r"C:\Users\Tester\AppData\Local"}
    )

    assert root == Path(
        r"C:\Users\Tester\AppData\Local\SF6 Music Trigger\V2"
    )


def test_open_diagnostic_folder_creates_and_opens_the_exact_log_directory(
    tmp_path: Path,
) -> None:
    from sf6_music_trigger.panel import open_diagnostic_folder

    opened: list[str] = []

    folder = open_diagnostic_folder(tmp_path, opener=opened.append)

    assert folder == tmp_path / "diagnostics"
    assert folder.is_dir()
    assert opened == [str(folder)]


def test_detailed_runtime_status_exposes_geometry_profile_and_recognition_reason() -> None:
    from sf6_music_trigger.panel import runtime_diagnostic_message

    message = runtime_diagnostic_message(
        CaptureStatus.RUNNING,
        TrackerState.DISARMED,
        TrackerState.DISARMED,
        p1_enabled=True,
        p2_enabled=False,
        p1_level=None,
        p2_level=None,
        current_geometry=ClientGeometry(-1920, 0, 1920, 1080),
        calibration=_context(),
        viewport_status="native",
        processing_reason="hud_untrusted",
    )

    assert "capture=RUNNING" in message
    assert "P1=DISARMED" in message
    assert "窗口=1920x1080@(-1920,0)" in message
    assert "校准=200x100" in message
    assert "viewport=native" in message
    assert "process=hud_untrusted" in message
    assert "C:\\Users" not in message


@pytest.mark.parametrize(
    ("capture_status", "capture_issue", "expected"),
    [
        (CaptureStatus.MINIMIZED, None, "正在等待 SF6：请恢复游戏窗口"),
        (CaptureStatus.OCCLUDED, None, "正在等待 SF6：请保持游戏窗口可见"),
        (
            CaptureStatus.LOST,
            CaptureIssue.WINDOW_CHANGED,
            "正在等待 SF6：未找到稳定的游戏窗口",
        ),
        (
            CaptureStatus.LOST,
            CaptureIssue.GEOMETRY_CHANGED,
            "正在等待 SF6：显示模式正在变化",
        ),
        (
            CaptureStatus.LOST,
            CaptureIssue.UNSUPPORTED_OUTPUT,
            "正在等待 SF6：请将游戏移到单个主显示器",
        ),
        (
            CaptureStatus.LOST,
            CaptureIssue.BACKEND_ERROR,
            "正在等待 SF6：捕获后端暂不可用，正在重试",
        ),
    ],
)
def test_panel_explains_each_recoverable_capture_start_failure(
    capture_status: CaptureStatus,
    capture_issue: CaptureIssue | None,
    expected: str,
) -> None:
    from sf6_music_trigger.controller import ControllerStatus
    from sf6_music_trigger.panel import _PanelStatusLatch

    status = ControllerStatus(
        kind="capture_waiting",
        capture_status=capture_status,
        capture_issue=capture_issue,
    )

    assert _PanelStatusLatch().resolve((status,), fallback="已停止") == expected


def test_panel_keeps_worker_error_visible_when_worker_then_stopped_arrive_together() -> None:
    from sf6_music_trigger.controller import ControllerStatus
    from sf6_music_trigger.panel import _PanelStatusLatch

    statuses = (
        ControllerStatus(kind="worker_error"),
        ControllerStatus(kind="stopped"),
    )

    assert _PanelStatusLatch().resolve(statuses, fallback="监听中") == (
        "监听器错误：请检查 SF6 窗口和本地配置"
    )


def test_user_stop_replaces_an_older_capture_waiting_message() -> None:
    from sf6_music_trigger.controller import ControllerStatus
    from sf6_music_trigger.panel import _PanelStatusLatch

    statuses = (
        ControllerStatus(
            kind="capture_waiting",
            capture_status=CaptureStatus.LOST,
            capture_issue=CaptureIssue.BACKEND_ERROR,
        ),
        ControllerStatus(kind="stopped"),
    )

    assert _PanelStatusLatch().resolve(statuses, fallback="监听中") == "已停止"


def test_listener_builder_shares_one_portable_diagnostic_log_with_engine_and_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sf6_music_trigger.capture as capture_module
    import sf6_music_trigger.controller as controller_module
    import sf6_music_trigger.diagnostic_log as diagnostic_module
    import sf6_music_trigger.engine as engine_module
    from sf6_music_trigger.panel import (
        _build_listener_controller,
        create_panel_application,
    )

    _write_valid_profile(tmp_path)
    track = tmp_path / "audio" / "p1.wav"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"audio")
    audio = _Audio()
    application = create_panel_application(
        config_path=tmp_path / "config.json",
        audio_port_factory=lambda _root: audio,
        config_loader=lambda _path: AppConfig(detection_backend=DetectionBackend.SCREEN),
    )
    application.view_model.profile_path = PROFILE_PATH
    application.view_model.p1_main_track = "audio/p1.wav"
    application.view_model.p2_enabled = False
    assert application.view_model.can_start
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))

    observed: dict[str, object] = {}

    class DiagnosticLogSpy:
        def __init__(self, root: Path) -> None:
            observed["root"] = root

        def start(self) -> None:
            pass

        @property
        def has_warning(self) -> bool:
            return False

        def record(self, _record: object) -> None:
            pass

        def close(self) -> None:
            pass

    class SourceSpy:
        def __init__(self, *, target_fps: int) -> None:
            observed["target_fps"] = target_fps

    class EngineSpy:
        def __init__(self, **kwargs: object) -> None:
            observed["engine_kwargs"] = kwargs
            self.processing_reason = "hud_untrusted"
            self.viewport_status = "normalized"

    class ControllerSpy:
        def __init__(self, **kwargs: object) -> None:
            observed["controller_kwargs"] = kwargs

    monkeypatch.setattr(diagnostic_module, "DiagnosticEventLog", DiagnosticLogSpy)
    monkeypatch.setattr(capture_module, "DxcamFrameSource", SourceSpy)
    monkeypatch.setattr(engine_module, "VisualTriggerEngine", EngineSpy)
    monkeypatch.setattr(controller_module, "ListenerController", ControllerSpy)

    controller = _build_listener_controller(
        application,
        audio_diagnostic=lambda _error: None,
    )

    engine_kwargs = observed["engine_kwargs"]
    controller_kwargs = observed["controller_kwargs"]
    diagnostic_sink = engine_kwargs["diagnostic_sink"]
    diagnostic_log = controller_kwargs["diagnostic_log"]
    from sf6_music_trigger.viewport import CenteredViewportResolver

    viewport_resolver = engine_kwargs["viewport_resolver"]
    assert controller.__class__ is ControllerSpy
    assert observed["root"] == (
        tmp_path / "local-app-data" / "SF6 Music Trigger" / "V2"
    )
    assert diagnostic_sink.__self__ is diagnostic_log
    assert controller_kwargs["engine"].__class__ is EngineSpy
    diagnostics = controller_kwargs["diagnostics_provider"]()
    assert diagnostics.processing_reason == "hud_untrusted"
    assert diagnostics.viewport_status == "normalized"
    assert isinstance(viewport_resolver, CenteredViewportResolver)
    assert viewport_resolver.target_aspect == 2.0
    assert viewport_resolver.confirmation_frames == 3
    assert viewport_resolver.minimum_content_height == 720


def test_application_context_uses_config_parent_as_single_canonical_root(
    tmp_path: Path,
) -> None:
    from sf6_music_trigger.panel import create_panel_application

    config_path = tmp_path / "nested" / "app_config.json"
    config = AppConfig(
        p1=SideAudioConfig(True, "audio/p1.wav", 0.2),
        p2=SideAudioConfig(False, "", 0.3),
        profile_path=PROFILE_PATH,
    )
    save_config(config_path, config)
    observed_roots: list[Path] = []

    def audio_factory(root: Path) -> _Audio:
        observed_roots.append(root)
        return _Audio()

    application = create_panel_application(
        config_path=config_path,
        audio_port_factory=audio_factory,
    )
    expected = config_path.parent.resolve(strict=False)
    assert application.user_data_root == expected
    assert application.view_model.user_dir == expected
    assert observed_roots == [expected]
    assert application.view_model.to_config() == config


class _Gate:
    def __init__(self, results: list[bool] | None = None) -> None:
        self._results = deque(results or [])
        self.reset_calls = 0
        self.accept_calls: list[tuple[int, object]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def accept(self, frame: CapturedFrame, *, now: object) -> bool:
        self.accept_calls.append((frame.frame_id, now))
        return self._results.popleft() if self._results else True


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self._values = deque(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self._values.popleft()


def _sample(
    frame_id: int,
    sampled_at: float,
    *,
    width: int = 80,
    left: int = 0,
    top: int = 0,
) -> CaptureSample:
    image = np.zeros((40, width, 3), dtype=np.uint8)
    return CaptureSample(
        status=CaptureStatus.RUNNING,
        frame=CapturedFrame(
            frame_id,
            image,
            sampled_at,
            ClientGeometry(left, top, width, 40),
        ),
    )


def _meter_crops() -> tuple[np.ndarray, np.ndarray]:
    row = np.linspace(0, 255, 80, dtype=np.uint8)
    full = np.repeat(np.tile(row, (40, 1))[:, :, None], 3, axis=2)
    spent = np.flip(full, axis=1).copy()
    return full, spent


def _calibration_colour_model(side: Side) -> MeterColorModel:
    mask = np.zeros((32, 64), dtype=np.bool_)
    mask[8:24, 8:56] = True
    return MeterColorModel(side, mask, 0, 5, 150, 150, 1.0, 0.0)


def test_tracker_config_keeps_fail_closed_thresholds_for_legacy_references() -> None:
    from sf6_music_trigger.panel import _tracker_config_for_references

    legacy = _references(Side.P1)
    modern = MeterReferences(
        Side.P1,
        legacy.full_gray,
        legacy.spent_gray,
        color_model=_calibration_colour_model(Side.P1),
    )

    legacy_config = _tracker_config_for_references(legacy)
    modern_config = _tracker_config_for_references(modern)

    assert (legacy_config.full_threshold, legacy_config.spent_threshold) == (0.78, 0.72)
    assert (modern_config.full_threshold, modern_config.spent_threshold) == (0.78, 0.72)


def _new_session(
    tmp_path: Path,
    *,
    gate: _Gate,
    clock: _Clock,
    engine_gate: object | None = None,
    **changes: Any,
):
    from sf6_music_trigger.panel import CalibrationSession

    values = {
        "user_data_root": tmp_path,
        "profile_path": PROFILE_PATH,
        "engine_gate": engine_gate or object(),
        "calibration_gate_factory": lambda: gate,
        "calibration_clock": clock,
        "build_color_model_fn": lambda side, _full, _spent: _calibration_colour_model(side),
    }
    values.update(changes)
    return CalibrationSession(**values)


def test_calibration_session_owns_and_resets_a_distinct_gate(tmp_path: Path) -> None:
    engine_gate = object()
    gate = _Gate()
    session = _new_session(
        tmp_path,
        gate=gate,
        clock=_Clock([]),
        engine_gate=engine_gate,
    )
    session.begin()
    assert session.gate is gate
    assert session.gate is not engine_gate
    assert gate.reset_calls == 1
    with pytest.raises(ValueError):
        _new_session(
            tmp_path,
            gate=gate,
            clock=_Clock([]),
            engine_gate=gate,
        ).begin()


def test_calibration_default_clock_matches_dxcam_perf_counter_timestamps() -> None:
    """The interactive sampler must accept timestamps from the live DXcam source."""
    from sf6_music_trigger.panel import _run_calibration_wizard

    parameter = inspect.signature(_run_calibration_wizard).parameters["calibration_clock"]
    assert parameter.default is time.perf_counter


def test_calibration_accepts_only_running_samples_through_gate_and_clock(
    tmp_path: Path,
) -> None:
    gate = _Gate([True])
    clock = _Clock([10.0])
    session = _new_session(tmp_path, gate=gate, clock=clock)
    session.begin()
    full, _ = _meter_crops()

    assert session.accept_crop(Side.P1, "full", _sample(0, 10.0), full) is True
    assert gate.accept_calls == [(0, 10.0)]
    assert clock.calls == 1

    assert session.accept_crop(
        Side.P1,
        "full",
        CaptureSample(status=CaptureStatus.OCCLUDED),
        full,
    ) is False
    assert gate.accept_calls == [(0, 10.0)]
    assert clock.calls == 1
    assert session.batch_size(Side.P1, "full") == 0


def test_calibration_context_frame_is_gate_accepted_without_entering_a_batch(
    tmp_path: Path,
) -> None:
    gate = _Gate([True])
    clock = _Clock([5.0])
    session = _new_session(tmp_path, gate=gate, clock=clock)
    session.begin()
    sample = _sample(0, 5.0)

    assert session.accept_context_frame(sample) is sample.frame
    assert gate.accept_calls == [(0, 5.0)]
    assert clock.calls == 1
    assert all(
        session.batch_size(side, phase) == 0
        for side in (Side.P1, Side.P2)
        for phase in ("full", "spent")
    )


def test_gate_rejection_and_geometry_change_discard_nonconsecutive_batch(
    tmp_path: Path,
) -> None:
    gate = _Gate([True, False, True, True])
    session = _new_session(tmp_path, gate=gate, clock=_Clock([0.0, 0.1, 0.2, 0.3]))
    session.begin()
    full, _ = _meter_crops()
    assert session.accept_crop(Side.P1, "full", _sample(0, 0.0), full)
    assert not session.accept_crop(Side.P1, "full", _sample(1, 0.1), full)
    assert session.batch_size(Side.P1, "full") == 0
    assert session.accept_crop(Side.P1, "full", _sample(2, 0.2), full)
    assert not session.accept_crop(
        Side.P1,
        "full",
        _sample(3, 0.3, width=81),
        np.pad(full, ((0, 0), (0, 1), (0, 0))),
    )
    assert session.batch_size(Side.P1, "full") == 0


def test_stable_batch_requires_five_visible_similar_consecutive_crops(
    tmp_path: Path,
) -> None:
    gate = _Gate()
    session = _new_session(
        tmp_path,
        gate=gate,
        clock=_Clock([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    )
    session.begin()
    full, _ = _meter_crops()
    for frame_id in range(5):
        assert session.accept_crop(
            Side.P1,
            "full",
            _sample(frame_id, frame_id / 10),
            full.copy(),
        )
    assert session.batch_size(Side.P1, "full") == 5

    hidden = np.zeros_like(full)
    assert not session.accept_crop(Side.P2, "spent", _sample(5, 0.5), hidden)
    assert session.batch_size(Side.P2, "spent") == 0


def test_stable_batch_uses_client_size_not_window_position(tmp_path: Path) -> None:
    session = _new_session(
        tmp_path,
        gate=_Gate(),
        clock=_Clock([0.0, 0.1]),
    )
    session.begin()
    full, _ = _meter_crops()

    assert session.accept_crop(
        Side.P1, "full", _sample(0, 0.0, left=0, top=0), full
    )
    assert session.accept_crop(
        Side.P1, "full", _sample(1, 0.1, left=50, top=-20), full
    )
    assert session.batch_size(Side.P1, "full") == 2


class _PersistenceProbe:
    def __init__(self, root: Path, fail_stage: str | None = None) -> None:
        self.root = root
        self.fail_stage = fail_stage
        self.calls: list[str] = []
        self.saved: dict[Side, MeterReferences] = {}

    def build(self, side: Side, full: object, spent: object) -> MeterReferences:
        self.calls.append(f"build-{side.value.lower()}")
        return _references(side)

    def save_refs(
        self,
        root: Path,
        profile_path: str,
        reference_path: str,
        references: MeterReferences,
    ) -> None:
        side_name = references.side.value.lower()
        self.calls.append(f"save-{side_name}")
        profile_target = root / profile_path
        target = profile_target.parent / reference_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"new-{side_name}".encode())
        if self.fail_stage == f"save-{side_name}":
            raise OSError(self.fail_stage)
        self.saved[references.side] = references

    def load_refs(
        self,
        root: Path,
        profile_path: str,
        reference_path: str,
        *,
        expected_side: Side,
    ) -> MeterReferences:
        side_name = expected_side.value.lower()
        self.calls.append(f"reload-{side_name}")
        if self.fail_stage == f"reload-{side_name}":
            raise OSError(self.fail_stage)
        return self.saved[expected_side]

    def save_profile(self, root: Path, profile_path: str, profile: HudProfile) -> None:
        self.calls.append("save-profile")
        if self.fail_stage == "save-profile":
            raise OSError(self.fail_stage)
        target = root / profile_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"new-profile")


def _ready_session(tmp_path: Path, probe: _PersistenceProbe, **changes: Any):
    session = _new_session(
        tmp_path,
        gate=_Gate(),
        clock=_Clock([index / 10 for index in range(20)]),
        build_references_fn=probe.build,
        save_references_fn=probe.save_refs,
        load_references_fn=probe.load_refs,
        save_profile_fn=probe.save_profile,
        attempt_id_factory=lambda: "attempt-1",
        **changes,
    )
    session.begin()
    session.configure(
        profile_id="task9-profile",
        calibration=_context(),
        battle_roi=NormalizedRoi(0.0, 0.0, 1.0, 1.0),
        training_hud_roi=None,
        p1_roi=NormalizedRoi(0.05, 0.05, 0.35, 0.25),
        p2_roi=NormalizedRoi(0.65, 0.05, 0.95, 0.25),
    )
    background = np.tile(np.linspace(45, 105, 80, dtype=np.uint8), (40, 1))
    spent = np.repeat(background[:, :, None], 3, axis=2)
    full = spent.copy()
    full[:, 12:68] = (0, 0, 255)
    frame_id = 0
    for side in (Side.P1, Side.P2):
        for phase, crop in (("full", full), ("spent", spent)):
            for _ in range(5):
                assert session.accept_crop(
                    side,
                    phase,
                    _sample(frame_id, frame_id / 10),
                    crop.copy(),
                )
                frame_id += 1
    return session


def test_calibration_commits_profile_only_after_both_refs_reload(tmp_path: Path) -> None:
    probe = _PersistenceProbe(tmp_path)
    session = _ready_session(tmp_path, probe)

    profile = session.complete()

    assert probe.calls == [
        "build-p1",
        "build-p2",
        "save-p1",
        "save-p2",
        "reload-p1",
        "reload-p2",
        "save-profile",
    ]
    assert profile.p1.references_path.endswith("attempt-1.p1.npz")
    assert profile.p2.references_path.endswith("attempt-1.p2.npz")
    assert (tmp_path / PROFILE_PATH).read_bytes() == b"new-profile"


def test_calibration_persists_colour_models_after_four_fresh_batches(
    tmp_path: Path,
) -> None:
    probe = _PersistenceProbe(tmp_path)
    session = _ready_session(tmp_path, probe)

    session.complete()

    assert probe.saved[Side.P1].color_model is not None
    assert probe.saved[Side.P2].color_model is not None
    assert probe.saved[Side.P1].schema_version == 2
    assert probe.saved[Side.P2].schema_version == 2


def test_colour_build_failure_preserves_prior_files(tmp_path: Path) -> None:
    old_profile = tmp_path / PROFILE_PATH
    old_profile.parent.mkdir(parents=True)
    old_profile.write_bytes(b"old-profile")
    probe = _PersistenceProbe(tmp_path)
    session = _ready_session(
        tmp_path,
        probe,
        build_color_model_fn=lambda *_args: (_ for _ in ()).throw(
            ValueError("colour features are invalid")
        ),
    )

    with pytest.raises(ValueError, match="colour"):
        session.complete()

    assert old_profile.read_bytes() == b"old-profile"
    assert not (tmp_path / "profiles" / "refs").exists()


def test_calibration_reserves_both_attempt_paths_before_first_save(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingProbe(_PersistenceProbe):
        reserved_before_save = False

        def save_refs(
            self,
            root: Path,
            profile_path: str,
            reference_path: str,
            references: MeterReferences,
        ) -> None:
            if references.side is Side.P1:
                profile_target = root / profile_path
                p1_target = profile_target.parent / reference_path
                p2_target = profile_target.parent / reference_path.replace(
                    ".p1.npz", ".p2.npz"
                )
                self.reserved_before_save = p1_target.exists() and p2_target.exists()
                entered.set()
                assert release.wait(timeout=2.0)
            super().save_refs(root, profile_path, reference_path, references)

    first_probe = BlockingProbe(tmp_path)
    first = _ready_session(tmp_path, first_probe)
    first_errors: list[BaseException] = []

    def complete_first() -> None:
        try:
            first.complete()
        except BaseException as error:
            first_errors.append(error)

    worker = threading.Thread(target=complete_first, daemon=True)
    worker.start()
    assert entered.wait(timeout=2.0)
    second = _ready_session(tmp_path, _PersistenceProbe(tmp_path))
    try:
        with pytest.raises((FileExistsError, ValueError)):
            second.complete()
    finally:
        release.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert first_errors == []
    assert first_probe.reserved_before_save is True


@pytest.mark.parametrize(
    "failure_stage",
    ["save-p1", "save-p2", "reload-p1", "reload-p2", "save-profile"],
)
def test_calibration_failure_preserves_old_profile_and_cleans_only_attempt_refs(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    old_profile = tmp_path / PROFILE_PATH
    old_profile.parent.mkdir(parents=True)
    old_profile.write_bytes(b"old-profile")
    neighbor = old_profile.parent / "refs" / "neighbor.npz"
    neighbor.parent.mkdir(parents=True)
    neighbor.write_bytes(b"neighbor")
    probe = _PersistenceProbe(tmp_path, failure_stage)
    session = _ready_session(tmp_path, probe)

    with pytest.raises(OSError):
        session.complete()

    assert old_profile.read_bytes() == b"old-profile"
    assert neighbor.read_bytes() == b"neighbor"
    assert list(neighbor.parent.glob("*attempt-1*.npz")) == []


def test_cancelled_calibration_writes_nothing(tmp_path: Path) -> None:
    probe = _PersistenceProbe(tmp_path)
    session = _new_session(
        tmp_path,
        gate=_Gate(),
        clock=_Clock([]),
        build_references_fn=probe.build,
        save_references_fn=probe.save_refs,
        load_references_fn=probe.load_refs,
        save_profile_fn=probe.save_profile,
    )
    session.begin()
    session.cancel()
    assert probe.calls == []
    assert list(tmp_path.rglob("*")) == []


def test_panel_and_main_imports_do_not_initialize_gui_or_backends() -> None:
    repository = Path(__file__).resolve().parents[1]
    script = """
import sys
for name in ('tkinter', 'pygame', 'dxcam'):
    sys.modules.pop(name, None)
import sf6_music_trigger.panel
import sf6_music_trigger.__main__
loaded = sorted(name for name in ('tkinter', 'pygame', 'dxcam') if name in sys.modules)
if loaded:
    raise SystemExit('unexpected imports: ' + ','.join(loaded))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(repository / 'src')!r});" + script],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_main_help_is_headless_and_returns_success() -> None:
    repository = Path(__file__).resolve().parents[1]
    source = str(repository / "src")
    launcher = (
        f"import runpy, sys; sys.path.insert(0, {source!r}); "
        "sys.argv=['sf6-music-trigger', '--help']; "
        "runpy.run_module('sf6_music_trigger', run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", launcher],
        cwd=repository,
        env={
            **os.environ,
            "PYTHONPATH": str(repository / "src"),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert completed.stderr == ""


def test_main_forwards_exact_config_path_to_panel_runner(tmp_path: Path) -> None:
    from sf6_music_trigger.__main__ import main

    observed: list[Path] = []
    config_path = tmp_path / "settings" / "app.json"
    assert main(["--config", str(config_path)], panel_runner=observed.append) == 0
    assert observed == [config_path]


def test_console_script_targets_existing_main_function() -> None:
    repository = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["scripts"] == {
        "sf6-music-trigger": "sf6_music_trigger.__main__:main"
    }


def test_calibration_wizard_is_reachable_and_commits_both_sides_offline(
    tmp_path: Path,
) -> None:
    from sf6_music_trigger.capture import FreshFrameGate
    from sf6_music_trigger.panel import (
        PanelApplication,
        PanelViewModel,
        _run_calibration_wizard,
    )
    from sf6_music_trigger.hud_profile import load_profile
    from sf6_music_trigger.meter_references import load_references

    background = np.tile(np.linspace(45, 105, 80, dtype=np.uint8), (40, 1))
    spent = np.repeat(background[:, :, None], 3, axis=2)
    full = spent.copy()
    full[:, 12:68] = (0, 0, 255)

    def client_frame(crop: np.ndarray, frame_id: int) -> CaptureSample:
        image = np.concatenate((crop, crop), axis=1).copy(order="C")
        return CaptureSample(
            status=CaptureStatus.RUNNING,
            frame=CapturedFrame(
                frame_id,
                image,
                frame_id / 10,
                ClientGeometry(20, -10, 160, 40),
            ),
        )

    samples = deque(
        [client_frame(full, 0)]
        + [client_frame(full, index) for index in range(1, 6)]
        + [client_frame(spent, index) for index in range(6, 11)]
        + [client_frame(full, index) for index in range(11, 16)]
        + [client_frame(spent, index) for index in range(16, 21)]
    )

    class Source:
        start_calls = 0
        stop_calls = 0

        def start(self) -> None:
            self.start_calls += 1

        def read(self) -> CaptureSample:
            return samples.popleft()

        def stop(self) -> None:
            self.stop_calls += 1

    class Dialogs:
        infos: list[str] = []
        errors: list[str] = []
        confirm_calls: list[tuple[tuple[int, ...], object]] = []

        @staticmethod
        def askstring(_title: str, prompt: str, **_kwargs: object) -> str:
            values = {
                "Profile ID": "task9-profile",
                "Battle ROI": "0,0,1,1",
                "P1 meter ROI": "0,0,0.5,1",
                "P2 meter ROI": "0.5,0,1,1",
                "Training HUD ROI": "",
                "HUD scale": "default",
                "Language": "English",
                "Brightness": "default",
                "Game version note": "offline-test",
            }
            return values[prompt]

        @staticmethod
        def askyesno(_title: str, _prompt: str, **_kwargs: object) -> bool:
            return False

        @staticmethod
        def askokcancel(_title: str, _prompt: str, **_kwargs: object) -> bool:
            return True

        @classmethod
        def confirm_rois(
            cls,
            _parent: object,
            image: np.ndarray,
            rois: object,
        ) -> bool:
            cls.confirm_calls.append((image.shape, rois))
            return True

        @classmethod
        def showinfo(cls, _title: str, message: str, **_kwargs: object) -> None:
            cls.infos.append(message)

        @classmethod
        def showerror(cls, _title: str, message: str, **_kwargs: object) -> None:
            cls.errors.append(message)

    audio = _Audio()
    config_path = tmp_path / "config.json"
    config = AppConfig(profile_path=PROFILE_PATH)
    view_model = PanelViewModel(user_dir=tmp_path, audio_port=audio)
    application = PanelApplication(config_path, tmp_path, config, audio, view_model)
    source = Source()
    statuses: list[str] = []

    assert _run_calibration_wizard(
        parent=object(),
        application=application,
        status_sink=statuses.append,
        source_factory=lambda _fps: source,
        gate_factory=FreshFrameGate,
        calibration_clock=_Clock([index / 10 for index in range(21)]),
        dialogs=Dialogs,
        attempt_id_factory=lambda: "wizard-attempt",
    ) is True

    profile = load_profile(tmp_path, PROFILE_PATH)
    p1 = load_references(
        tmp_path, PROFILE_PATH, profile.p1.references_path, expected_side=Side.P1
    )
    p2 = load_references(
        tmp_path, PROFILE_PATH, profile.p2.references_path, expected_side=Side.P2
    )
    assert p1.side is Side.P1 and p2.side is Side.P2
    assert p1 is not p2
    assert source.start_calls == 1 and source.stop_calls == 1
    assert statuses == ["HUD 校准完成"]
    assert Dialogs.errors == []
    assert len(Dialogs.confirm_calls) == 1
    assert Dialogs.confirm_calls[0][0] == (40, 160, 3)
