"""UI-independent panel state and offline HUD calibration transactions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Rational, Real
import os
from pathlib import Path, PureWindowsPath
import time
from typing import Callable
import uuid

import numpy as np

from .audio import AudioPlaybackError, PygameAudioPort
from .audio_templates import (
    DEFAULT_AUDIO_TEMPLATE_NAMES,
    AudioTemplate,
    AudioTemplateSide,
    add_audio_template,
    load_audio_templates,
    rename_audio_template,
    save_audio_templates,
)
from .config import (
    MAX_VOLUME,
    AppConfig,
    DetectionBackend,
    SideAudioConfig,
    SuperScope,
    load_config,
    save_config,
)
from .domain import (
    CaptureIssue,
    CaptureSample,
    CaptureStatus,
    CapturedFrame,
    ClientGeometry,
    Side,
    TrackerState,
)
from .engine import profile_applies_to
from .viewport import centered_viewport
from .hud_profile import (
    CalibrationContext,
    HudProfile,
    NormalizedRoi,
    SideHudProfile,
    crop_normalized,
    load_profile,
    resolve_contained_path,
    save_profile,
)
from .meter_references import (
    MeterReferences,
    build_meter_references,
    load_references,
    preprocess_meter,
    save_references,
)
from .meter_color_features import MeterColorModel, build_meter_color_model
from .meter_level_features import derive_meter_level_roi
from .reframework_events import (
    Connection,
    detector_installed,
    find_sf6_install,
    is_game_dir,
)


_PHASES = frozenset({"full", "spent"})
_STABLE_SAMPLE_COUNT = 5
_MAX_PAIRWISE_DISTANCE = 0.03
_SUPER_SCOPE_LABELS = {
    SuperScope.ALL_SUPERS: "全部大招（SA1/SA2/SA3/CA）",
    SuperScope.SA3_CA_ONLY: "仅 SA3/CA",
}
_SUPER_SCOPE_FROM_LABEL = {
    label: scope for scope, label in _SUPER_SCOPE_LABELS.items()
}
_BACKEND_LABELS = {
    DetectionBackend.REFRAMEWORK: "REFramework 游戏内检测（推荐）",
    DetectionBackend.SCREEN: "画面识别（旧版 T03）",
}
_BACKEND_FROM_LABEL = {label: backend for backend, label in _BACKEND_LABELS.items()}
NORMAL_VOLUME_PERCENT = 100
MAX_VOLUME_PERCENT = round(MAX_VOLUME * 100)


def _tracker_config_for_references(references: MeterReferences):
    """Use the tracker’s fail-closed thresholds for every reference format."""

    from .gauge_tracker import GaugeTrackerConfig

    if not isinstance(references, MeterReferences):
        raise TypeError("references must be MeterReferences")
    return GaugeTrackerConfig()


def runtime_status_message(
    capture_status: CaptureStatus | None,
    p1_state: TrackerState | None,
    p2_state: TrackerState | None,
    *,
    p1_enabled: bool = True,
    p2_enabled: bool = True,
    p1_level: int | None = None,
    p2_level: int | None = None,
) -> str:
    """Return an unambiguous, privacy-safe listener status for the panel."""

    if capture_status is None:
        return '就绪：选择音频后点击“开始”以连接 SF6'
    if capture_status is CaptureStatus.STOPPED:
        return "正在连接 SF6：等待第一帧 HUD"
    if capture_status is CaptureStatus.LOST:
        return "捕获暂时不可用：请保持 SF6 窗口可见并等待恢复"
    if capture_status is not CaptureStatus.RUNNING:
        return "捕获暂时不可用：请保持 SF6 窗口可见并等待恢复"
    if type(p1_enabled) is not bool or type(p2_enabled) is not bool:
        raise TypeError("side enabled flags must be bool")
    for level in (p1_level, p2_level):
        if level is not None and type(level) is not int:
            raise TypeError("meter levels must be integers or None")
        if level is not None and not 0 <= level <= 3:
            raise ValueError("meter levels must be between zero and three")
    first = (
        "disabled"
        if not p1_enabled
        else "UNKNOWN" if p1_state is None else p1_state.value
    )
    second = (
        "disabled"
        if not p2_enabled
        else "UNKNOWN" if p2_state is None else p2_state.value
    )
    message = f"监听中：capture=RUNNING；P1={first}；P2={second}"
    level_labels = [
        f"{side.value}={'LEVEL_UNKNOWN' if level is None else f'LEVEL_{level}'}"
        for side, level, enabled in (
            (Side.P1, p1_level, p1_enabled),
            (Side.P2, p2_level, p2_enabled),
        )
        if enabled
    ]
    if level_labels:
        message += f"；气槽：{'; '.join(level_labels)}"
    enabled_levels = [
        level
        for level, enabled in ((p1_level, p1_enabled), (p2_level, p2_enabled))
        if enabled
    ]
    if enabled_levels and all(level is None for level in enabled_levels):
        message += "；气槽等级不可读"
    waiting = [
        side.value
        for side, state, enabled in (
            (Side.P1, p1_state, p1_enabled),
            (Side.P2, p2_state, p2_enabled),
        )
        if enabled and state is TrackerState.DISARMED
    ]
    if waiting:
        message += f"；{'/'.join(waiting)} 请保持 CA 至少 0.25 秒以完成武装"
    return message


def ref_runtime_message(
    connection: Connection | None,
    chara: tuple[int | None, int | None],
    outcomes: tuple[str | None, str | None],
    *,
    p1_enabled: bool,
    p2_enabled: bool,
) -> str:
    """Status line for the REFramework backend; never contains private paths."""

    if connection is None:
        return "正在启动监听：等待游戏内检测脚本"
    if connection is Connection.NO_FILE:
        return "等待 SF6：还没有检测脚本的输出，请启动游戏（需要 REFramework 和检测脚本）"
    if connection is Connection.WAITING:
        return "等待 SF6：游戏未运行，或检测脚本没有响应"
    if connection is Connection.DISABLED:
        return "SF6 内检测已关闭：请在 REFramework 菜单里勾选 Detect supers"
    if connection is Connection.ONLINE:
        return "联机对战中：V2 已自动停用（仅支持离线：训练、CPU、本地对战）"
    if connection is Connection.MENU:
        return "已连接 SF6：等待进入对战或训练模式"
    parts = []
    for side, enabled, character, outcome in (
        (Side.P1, p1_enabled, chara[0], outcomes[0]),
        (Side.P2, p2_enabled, chara[1], outcomes[1]),
    ):
        state = "关闭" if not enabled else outcome or "监听中"
        who = "?" if character is None else str(character)
        parts.append(f"{side.value}（角色 {who}）：{state}")
    return "监听中（REFramework）：" + "；".join(parts)


def diagnostic_data_root(environ: Mapping[str, str] | None = None) -> Path:
    """Return a user-writable diagnostic root independent of the EXE folder."""

    source = os.environ if environ is None else environ
    if not isinstance(source, Mapping):
        raise TypeError("environ must be a mapping")
    local_app_data = source.get("LOCALAPPDATA", "")
    if type(local_app_data) is not str:
        raise TypeError("LOCALAPPDATA must be text")
    base = (
        Path(local_app_data.strip())
        if local_app_data.strip()
        else Path.home() / "AppData" / "Local"
    )
    from . import BUILD_ID

    return base / "SF6 Music Trigger" / BUILD_ID


def open_diagnostic_folder(
    root: str | os.PathLike[str] | None = None,
    *,
    opener: Callable[[str], object] | None = None,
) -> Path:
    """Create and open the diagnostic directory for non-technical testers."""

    folder = (
        Path(root) / "diagnostics"
        if root is not None
        else diagnostic_data_root() / "diagnostics"
    )
    folder.mkdir(parents=True, exist_ok=True)
    selected_opener = os.startfile if opener is None else opener
    selected_opener(str(folder))
    return folder


def runtime_diagnostic_message(
    capture_status: CaptureStatus | None,
    p1_state: TrackerState | None,
    p2_state: TrackerState | None,
    *,
    p1_enabled: bool,
    p2_enabled: bool,
    p1_level: int | None,
    p2_level: int | None,
    current_geometry: ClientGeometry | None,
    calibration: CalibrationContext | None,
    viewport_status: str | None,
    processing_reason: str | None,
) -> str:
    """Expose privacy-safe capture and recognition boundaries for remote QA."""

    message = runtime_status_message(
        capture_status,
        p1_state,
        p2_state,
        p1_enabled=p1_enabled,
        p2_enabled=p2_enabled,
        p1_level=p1_level,
        p2_level=p2_level,
    )
    details: list[str] = []
    if current_geometry is not None:
        details.append(
            "窗口="
            f"{current_geometry.width}x{current_geometry.height}"
            f"@({current_geometry.left},{current_geometry.top})"
        )
    else:
        details.append("窗口=UNKNOWN")
    if calibration is not None:
        details.append(
            f"校准={calibration.client_width}x{calibration.client_height}"
        )
        details.append(f"HDR={'on' if calibration.hdr_enabled else 'off'}")
    else:
        details.append("校准=UNKNOWN")
    details.append(f"viewport={viewport_status or 'UNKNOWN'}")
    details.append(f"process={processing_reason or 'UNKNOWN'}")
    return f"{message}；诊断：{'；'.join(details)}"


def _require_percent(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("volume percentage must be a real number")
    if isinstance(value, Rational):
        if value < 0 or value > MAX_VOLUME_PERCENT:
            raise ValueError("volume percentage must be between 0 and MAX_VOLUME_PERCENT")
        return float(Fraction(value, 100))
    if not math.isfinite(value) or value < 0 or value > MAX_VOLUME_PERCENT:
        raise ValueError("volume percentage must be finite and between 0 and MAX_VOLUME_PERCENT")
    return float(value) / 100.0


def _track_exists(root: Path, track: str) -> bool:
    if type(track) is not str or not track.strip():
        return False
    try:
        windows_path = PureWindowsPath(track)
        if windows_path.is_absolute() or Path(track).is_absolute():
            target = Path(track)
        else:
            target = resolve_contained_path(root, track, field_name="main_track")
        return target.is_file()
    except (OSError, TypeError, ValueError):
        return False


def audio_track_display_name(track: str) -> str:
    """Return a compact audio label without exposing its directory path."""

    if type(track) is not str or not track.strip():
        return "未选择音频"
    return PureWindowsPath(track.replace("/", "\\")).name or "未选择音频"


class PanelViewModel:
    """Mutable presentation state with validation independent of a GUI toolkit."""

    def __init__(
        self,
        *,
        user_dir: str | Path,
        audio_port: object | None = None,
        status_sink: Callable[[AudioPlaybackError], object] | None = None,
        profile_loader: Callable[[str | Path, str], HudProfile] = load_profile,
        reference_loader: Callable[..., MeterReferences] = load_references,
    ) -> None:
        if not callable(profile_loader) or not callable(reference_loader):
            raise TypeError("profile and reference loaders must be callable")
        if status_sink is not None and not callable(status_sink):
            raise TypeError("status_sink must be callable")
        self.user_dir = Path(user_dir).resolve(strict=False)
        self.audio_port = audio_port
        self._status_sink = status_sink or (lambda _error: None)
        self._profile_loader = profile_loader
        self._reference_loader = reference_loader

        self.profile_path = "profiles/default.json"
        self.p1_enabled = True
        self.p2_enabled = True
        self.p1_main_track = ""
        self.p2_main_track = ""
        self.p1_volume_percent: Real = 18
        self.p2_volume_percent: Real = 18
        self.p1_whiff_track = ""
        self.p2_whiff_track = ""
        self.p1_whiff_volume_percent: Real = 18
        self.p2_whiff_volume_percent: Real = 18
        self.p1_play_whiff_sound = False
        self.p2_play_whiff_sound = False
        self.p1_whiff_fade_ms = 1500
        self.p2_whiff_fade_ms = 1500
        self.p1_super_scope = SuperScope.SA3_CA_ONLY
        self.p2_super_scope = SuperScope.SA3_CA_ONLY
        self.p1_fade_non_three = False
        self.p2_fade_non_three = False
        self.capture_fps = 20
        self.detection_backend = DetectionBackend.REFRAMEWORK
        self.game_dir = ""
        self.volume_unlimited = False
        self.auto_start = False
        self.current_geometry: ClientGeometry | None = None
        self.capture_status: CaptureStatus | None = None
        self.p1_tracker_state: TrackerState | None = None
        self.p2_tracker_state: TrackerState | None = None
        self.p1_meter_level: int | None = None
        self.p2_meter_level: int | None = None
        self.processing_reason: str | None = None
        self.viewport_status: str | None = None
        self.validation_message = ""
        self.selected_profile: HudProfile | None = None
        self.start_command: Callable[[], object] = lambda: None
        self.stop_command: Callable[[], object] = lambda: None
        self.exit_command: Callable[[], object] = lambda: None
        self.calibration_command: Callable[[], object] = lambda: None

    @property
    def calibration_context(self) -> CalibrationContext | None:
        profile = self.selected_profile
        return None if profile is None else profile.calibration

    @property
    def p1_fade_on_whiff(self) -> bool:
        return not self.p1_play_whiff_sound

    @property
    def p2_fade_on_whiff(self) -> bool:
        return not self.p2_play_whiff_sound

    _VOLUME_ATTRS = (
        "p1_volume_percent",
        "p2_volume_percent",
        "p1_whiff_volume_percent",
        "p2_whiff_volume_percent",
    )

    def set_volume_unlimited(self, enabled: bool) -> None:
        """Allow volumes up to 500 %; switching back caps them at 100 %."""

        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        self.volume_unlimited = enabled
        if not enabled:
            for attr in self._VOLUME_ATTRS:
                if getattr(self, attr) > NORMAL_VOLUME_PERCENT:
                    setattr(self, attr, NORMAL_VOLUME_PERCENT)

    def reset_volumes(self) -> None:
        """Music and whiff sounds back to their own loudness (100 %)."""

        for attr in self._VOLUME_ATTRS:
            setattr(self, attr, NORMAL_VOLUME_PERCENT)

    def set_whiff_sound_mode(self, side: Side, enabled: bool) -> None:
        if not isinstance(side, Side):
            raise TypeError("side must be Side")
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if side is Side.P1:
            self.p1_play_whiff_sound = enabled
        else:
            self.p2_play_whiff_sound = enabled

    @property
    def audio_template_names(self) -> tuple[str, ...]:
        try:
            names = tuple(load_audio_templates(self.audio_template_path))
        except (OSError, TypeError, ValueError):
            names = ()
        return names or DEFAULT_AUDIO_TEMPLATE_NAMES

    @property
    def audio_template_path(self) -> Path:
        return self.user_dir / "audio_templates.json"

    def _current_audio_template(self) -> AudioTemplate:
        return AudioTemplate(
            p1=AudioTemplateSide(
                self.p1_enabled,
                self.p1_main_track,
                _require_percent(self.p1_volume_percent),
                self.p1_whiff_track,
                _require_percent(self.p1_whiff_volume_percent),
            ),
            p2=AudioTemplateSide(
                self.p2_enabled,
                self.p2_main_track,
                _require_percent(self.p2_volume_percent),
                self.p2_whiff_track,
                _require_percent(self.p2_whiff_volume_percent),
            ),
        )

    def save_audio_template(self, name: str) -> bool:
        if name not in self.audio_template_names:
            self.validation_message = "请选择有效的音频模板"
            return False
        try:
            templates = load_audio_templates(self.audio_template_path)
            templates[name] = self._current_audio_template()
            save_audio_templates(self.audio_template_path, templates)
        except (OSError, TypeError, ValueError):
            self.validation_message = "音频模板保存失败"
            return False
        self.validation_message = f"{name} 已保存"
        return True

    def add_audio_template(self, name: str) -> bool:
        try:
            normalized = add_audio_template(
                self.audio_template_path,
                name,
                self._current_audio_template(),
            )
        except (OSError, TypeError, ValueError):
            self.validation_message = "模板名称无效、重复或保存失败"
            return False
        self.validation_message = f"{normalized} 已新增并保存"
        return True

    def rename_audio_template(self, old_name: str, new_name: str) -> bool:
        try:
            normalized = rename_audio_template(
                self.audio_template_path,
                old_name,
                new_name,
            )
        except KeyError:
            self.validation_message = f"{old_name} 尚未保存"
            return False
        except (OSError, TypeError, ValueError):
            self.validation_message = "模板名称无效、重复或重命名失败"
            return False
        self.validation_message = f"模板已重命名为 {normalized}"
        return True

    def apply_audio_template(self, name: str) -> bool:
        if name not in self.audio_template_names:
            self.validation_message = "请选择有效的音频模板"
            return False
        try:
            template = load_audio_templates(self.audio_template_path)[name]
        except KeyError:
            self.validation_message = f"{name} 尚未保存"
            return False
        except (OSError, TypeError, ValueError):
            self.validation_message = "音频模板读取失败"
            return False

        self.p1_enabled = template.p1.enabled
        self.p2_enabled = template.p2.enabled
        self.p1_main_track = template.p1.main_track
        self.p2_main_track = template.p2.main_track
        self.p1_volume_percent = round(template.p1.volume * 100, 10)
        self.p2_volume_percent = round(template.p2.volume * 100, 10)
        self.p1_whiff_track = template.p1.whiff_track
        self.p2_whiff_track = template.p2.whiff_track
        self.p1_whiff_volume_percent = round(template.p1.whiff_volume * 100, 10)
        self.p2_whiff_volume_percent = round(template.p2.whiff_volume * 100, 10)
        self.validation_message = f"{name} 已应用"
        return True

    def _validate_tracks(self) -> bool:
        enabled_tracks = (
            (self.p1_enabled, self.p1_main_track),
            (self.p2_enabled, self.p2_main_track),
        )
        if not any(enabled and _track_exists(self.user_dir, track) for enabled, track in enabled_tracks):
            self.validation_message = "请选择有效音频文件 闊抽"
            return False

        whiff_tracks = (
            (self.p1_enabled, self.p1_play_whiff_sound, self.p1_whiff_track),
            (self.p2_enabled, self.p2_play_whiff_sound, self.p2_whiff_track),
        )
        if any(
            enabled and sound_mode and not _track_exists(self.user_dir, track)
            for enabled, sound_mode, track in whiff_tracks
        ):
            self.validation_message = "Whiff sound file is invalid"
            return False
        return True

    def _can_start_reframework(self) -> bool:
        if not is_game_dir(self.game_dir):
            self.validation_message = "请选择 SF6 游戏目录（包含 StreetFighter6.exe）"
            return False
        if not (Path(self.game_dir) / "dinput8.dll").is_file():
            self.validation_message = "游戏里还没有安装 REFramework：请先点击“一键安装”"
            return False
        if not detector_installed(self.game_dir):
            self.validation_message = "游戏内脚本未安装或不是最新版：请先点击“一键安装”"
            return False
        if not self._validate_tracks():
            return False
        self.validation_message = ""
        return True

    @property
    def can_start(self) -> bool:
        self.selected_profile = None
        if self.detection_backend is DetectionBackend.REFRAMEWORK:
            return self._can_start_reframework()
        try:
            profile = self._profile_loader(self.user_dir, self.profile_path)
            p1 = self._reference_loader(
                self.user_dir,
                self.profile_path,
                profile.p1.references_path,
                expected_side=Side.P1,
            )
            p2 = self._reference_loader(
                self.user_dir,
                self.profile_path,
                profile.p2.references_path,
                expected_side=Side.P2,
            )
            if p1 is p2:
                raise ValueError("P1 and P2 references must be distinct")
        except (OSError, TypeError, ValueError):
            self.validation_message = "HUD 配置或参考图无效"
            return False

        if self.current_geometry is not None:
            try:
                applies = profile_applies_to(profile, self.current_geometry)
                if not applies:
                    viewport = centered_viewport(
                        self.current_geometry.width,
                        self.current_geometry.height,
                        profile.calibration.client_aspect_ratio,
                    )
                    applies = viewport.height >= 720
            except (TypeError, ValueError):
                applies = False
            if not applies:
                self.validation_message = "需要重新校准 閲嶆柊鏍″噯"
                return False

        if not self._validate_tracks():
            return False

        self.selected_profile = profile
        self.validation_message = ""
        return True

    def preview(self, side: Side) -> bool:
        if not isinstance(side, Side):
            raise TypeError("side must be Side")
        if self.audio_port is None or not callable(getattr(self.audio_port, "preview", None)):
            raise RuntimeError("audio preview is unavailable")
        track = self.p1_main_track if side is Side.P1 else self.p2_main_track
        percent = (
            self.p1_volume_percent if side is Side.P1 else self.p2_volume_percent
        )
        volume = _require_percent(percent)
        try:
            self.audio_port.preview(track, volume)
        except AudioPlaybackError as error:
            self._status_sink(error)
            return False
        return True

    def preview_whiff(self, side: Side) -> bool:
        if not isinstance(side, Side):
            raise TypeError("side must be Side")
        if self.audio_port is None or not callable(getattr(self.audio_port, "preview", None)):
            raise RuntimeError("audio preview is unavailable")
        track = self.p1_whiff_track if side is Side.P1 else self.p2_whiff_track
        percent = (
            self.p1_whiff_volume_percent
            if side is Side.P1
            else self.p2_whiff_volume_percent
        )
        try:
            self.audio_port.preview(track, _require_percent(percent))
        except AudioPlaybackError as error:
            self._status_sink(error)
            return False
        return True

    def to_config(self) -> AppConfig:
        return AppConfig(
            p1=SideAudioConfig(
                self.p1_enabled,
                self.p1_main_track,
                _require_percent(self.p1_volume_percent),
                self.p1_whiff_track,
                _require_percent(self.p1_whiff_volume_percent),
                self.p1_play_whiff_sound,
                self.p1_whiff_fade_ms,
                self.p1_super_scope,
                self.p1_fade_non_three,
            ),
            p2=SideAudioConfig(
                self.p2_enabled,
                self.p2_main_track,
                _require_percent(self.p2_volume_percent),
                self.p2_whiff_track,
                _require_percent(self.p2_whiff_volume_percent),
                self.p2_play_whiff_sound,
                self.p2_whiff_fade_ms,
                self.p2_super_scope,
                self.p2_fade_non_three,
            ),
            profile_path=self.profile_path,
            capture_fps=self.capture_fps,
            detection_backend=self.detection_backend,
            game_dir=self.game_dir,
            volume_unlimited=self.volume_unlimited,
            auto_start=self.auto_start,
        )

    def start(self) -> object:
        return self.start_command()

    def stop(self) -> object:
        return self.stop_command()

    def exit(self) -> object:
        return self.exit_command()

    def calibrate(self) -> object:
        return self.calibration_command()

    def update_runtime_diagnostics(
        self,
        *,
        capture_status: CaptureStatus | None,
        current_geometry: ClientGeometry | None,
        tracker_states: tuple[TrackerState, TrackerState] | None,
        tracker_levels: tuple[int | None, int | None] | None = None,
        processing_reason: str | None = None,
        viewport_status: str | None = None,
    ) -> None:
        if capture_status is not None and not isinstance(capture_status, CaptureStatus):
            raise TypeError("capture_status must be CaptureStatus or None")
        if current_geometry is not None and not isinstance(
            current_geometry, ClientGeometry
        ):
            raise TypeError("current_geometry must be ClientGeometry or None")
        if tracker_states is not None and (
            type(tracker_states) is not tuple
            or len(tracker_states) != 2
            or not all(isinstance(state, TrackerState) for state in tracker_states)
        ):
            raise TypeError("tracker_states must contain two TrackerState values")
        if tracker_levels is not None:
            if type(tracker_levels) is not tuple or len(tracker_levels) != 2:
                raise TypeError("tracker_levels must contain two meter levels")
            for level in tracker_levels:
                if level is not None and type(level) is not int:
                    raise TypeError("meter levels must be integers or None")
                if level is not None and not 0 <= level <= 3:
                    raise ValueError("meter levels must be between zero and three")
        for name, value in (
            ("processing_reason", processing_reason),
            ("viewport_status", viewport_status),
        ):
            if value is not None and (type(value) is not str or not value):
                raise TypeError(f"{name} must be nonempty text or None")
        self.capture_status = capture_status
        self.current_geometry = current_geometry
        self.processing_reason = processing_reason
        self.viewport_status = viewport_status
        if tracker_states is None:
            self.p1_tracker_state = None
            self.p2_tracker_state = None
            self.p1_meter_level = None
            self.p2_meter_level = None
        else:
            self.p1_tracker_state, self.p2_tracker_state = tracker_states
            if tracker_levels is None:
                self.p1_meter_level = None
                self.p2_meter_level = None
            else:
                self.p1_meter_level, self.p2_meter_level = tracker_levels


@dataclass(frozen=True)
class PanelApplication:
    config_path: Path
    user_data_root: Path
    config: AppConfig
    audio_port: object
    view_model: PanelViewModel


def create_panel_application(
    *,
    config_path: str | Path,
    audio_port_factory: Callable[[Path], object] = PygameAudioPort,
    config_loader: Callable[[str | Path], AppConfig] = load_config,
    status_sink: Callable[[AudioPlaybackError], object] | None = None,
) -> PanelApplication:
    if not callable(audio_port_factory) or not callable(config_loader):
        raise TypeError("application factories must be callable")
    target = Path(config_path)
    root = target.parent.resolve(strict=False)
    config = config_loader(target)
    if not isinstance(config, AppConfig):
        raise TypeError("config loader must return AppConfig")
    audio_port = audio_port_factory(root)
    view_model = PanelViewModel(
        user_dir=root,
        audio_port=audio_port,
        status_sink=status_sink,
    )
    view_model.profile_path = config.profile_path
    view_model.p1_enabled = config.p1.enabled
    view_model.p2_enabled = config.p2.enabled
    view_model.p1_main_track = config.p1.main_track
    view_model.p2_main_track = config.p2.main_track
    view_model.p1_volume_percent = config.p1.volume * 100
    view_model.p2_volume_percent = config.p2.volume * 100
    view_model.p1_whiff_track = config.p1.whiff_track
    view_model.p2_whiff_track = config.p2.whiff_track
    view_model.p1_whiff_volume_percent = config.p1.whiff_volume * 100
    view_model.p2_whiff_volume_percent = config.p2.whiff_volume * 100
    view_model.p1_play_whiff_sound = config.p1.play_whiff_sound
    view_model.p2_play_whiff_sound = config.p2.play_whiff_sound
    view_model.p1_whiff_fade_ms = config.p1.whiff_fade_ms
    view_model.p2_whiff_fade_ms = config.p2.whiff_fade_ms
    view_model.p1_super_scope = config.p1.super_scope
    view_model.p2_super_scope = config.p2.super_scope
    view_model.p1_fade_non_three = config.p1.fade_non_three
    view_model.p2_fade_non_three = config.p2.fade_non_three
    view_model.capture_fps = config.capture_fps
    view_model.detection_backend = config.detection_backend
    view_model.game_dir = config.game_dir
    view_model.volume_unlimited = config.volume_unlimited
    view_model.auto_start = config.auto_start
    return PanelApplication(target, root, config, audio_port, view_model)


@dataclass(frozen=True)
class _CalibrationEntry:
    client_size: tuple[int, int]
    crop_bgr: np.ndarray
    normalized_gray: np.ndarray


class CalibrationSession:
    """Own one freshness epoch and atomically publish a calibrated profile."""

    def __init__(
        self,
        *,
        user_data_root: str | Path,
        profile_path: str,
        engine_gate: object,
        calibration_gate_factory: Callable[[], object],
        calibration_clock: Callable[[], Real],
        build_references_fn: Callable[..., MeterReferences] = build_meter_references,
        build_color_model_fn: Callable[..., MeterColorModel] = build_meter_color_model,
        save_references_fn: Callable[..., None] = save_references,
        load_references_fn: Callable[..., MeterReferences] = load_references,
        save_profile_fn: Callable[..., None] = save_profile,
        attempt_id_factory: Callable[[], str] | None = None,
    ) -> None:
        callables = (
            calibration_gate_factory,
            calibration_clock,
            build_references_fn,
            build_color_model_fn,
            save_references_fn,
            load_references_fn,
            save_profile_fn,
        )
        if not all(callable(value) for value in callables):
            raise TypeError("calibration dependencies must be callable")
        self.user_data_root = Path(user_data_root).resolve(strict=False)
        self.profile_path = profile_path
        self.engine_gate = engine_gate
        self._gate_factory = calibration_gate_factory
        self._clock = calibration_clock
        self._build_references = build_references_fn
        self._build_color_model = build_color_model_fn
        self._save_references = save_references_fn
        self._load_references = load_references_fn
        self._save_profile = save_profile_fn
        self._attempt_id_factory = attempt_id_factory or (lambda: uuid.uuid4().hex)
        self.gate: object | None = None
        self._used_gates: list[object] = []
        self._batches = {
            (side, phase): [] for side in (Side.P1, Side.P2) for phase in _PHASES
        }
        self._configuration: dict[str, object] | None = None

    def begin(self) -> None:
        gate = self._gate_factory()
        if gate is self.engine_gate or any(gate is old for old in self._used_gates):
            raise ValueError("calibration must own a distinct freshness gate")
        reset = getattr(gate, "reset", None)
        accept = getattr(gate, "accept", None)
        if not callable(reset) or not callable(accept):
            raise TypeError("calibration gate must provide reset and accept")
        reset()
        self._clear_batches()
        self.gate = gate
        self._used_gates.append(gate)

    def configure(
        self,
        *,
        profile_id: str,
        calibration: CalibrationContext,
        battle_roi: NormalizedRoi,
        training_hud_roi: NormalizedRoi | None,
        p1_roi: NormalizedRoi,
        p2_roi: NormalizedRoi,
    ) -> None:
        self._configuration = {
            "profile_id": profile_id,
            "calibration": calibration,
            "battle_roi": battle_roi,
            "training_hud_roi": training_hud_roi,
            "p1_roi": p1_roi,
            "p2_roi": p2_roi,
        }

    def accept_context_frame(self, sample: CaptureSample) -> CapturedFrame | None:
        if not isinstance(sample, CaptureSample):
            raise TypeError("sample must be CaptureSample")
        gate = self.gate
        if gate is None:
            raise RuntimeError("calibration session has not begun")
        if sample.status is not CaptureStatus.RUNNING:
            return None
        frame = sample.frame
        assert frame is not None
        if not gate.accept(frame, now=self._clock()):
            return None
        return frame

    def _clear_batches(self) -> None:
        for batch in self._batches.values():
            batch.clear()

    def _batch(self, side: Side, phase: str) -> list[_CalibrationEntry]:
        if not isinstance(side, Side):
            raise TypeError("side must be Side")
        if type(phase) is not str:
            raise TypeError("phase must be a string")
        if phase not in _PHASES:
            raise ValueError("phase must be full or spent")
        return self._batches[(side, phase)]

    def batch_size(self, side: Side, phase: str) -> int:
        return len(self._batch(side, phase))

    def accept_crop(
        self,
        side: Side,
        phase: str,
        sample: CaptureSample,
        crop_bgr: np.ndarray,
    ) -> bool:
        batch = self._batch(side, phase)
        if not isinstance(sample, CaptureSample):
            raise TypeError("sample must be CaptureSample")
        gate = self.gate
        if gate is None:
            raise RuntimeError("calibration session has not begun")
        if sample.status is not CaptureStatus.RUNNING:
            batch.clear()
            return False
        frame = sample.frame
        assert frame is not None
        now = self._clock()
        if not gate.accept(frame, now=now):
            batch.clear()
            return False

        processed = preprocess_meter(crop_bgr)
        if not processed.hud_visible:
            batch.clear()
            return False
        client_size = (frame.geometry.width, frame.geometry.height)
        if batch and client_size != batch[0].client_size:
            batch.clear()
            return False
        for previous in batch:
            difference = np.abs(
                processed.normalized_gray.astype(np.float32, copy=False)
                - previous.normalized_gray.astype(np.float32, copy=False)
            )
            distance = float(np.mean(difference, dtype=np.float64) / 255.0)
            if distance > _MAX_PAIRWISE_DISTANCE:
                batch.clear()
                return False
        if len(batch) < _STABLE_SAMPLE_COUNT:
            crop = np.array(crop_bgr, dtype=np.uint8, order="C", copy=True)
            normalized = np.array(
                processed.normalized_gray, dtype=np.uint8, order="C", copy=True
            )
            normalized.setflags(write=False)
            batch.append(_CalibrationEntry(client_size, crop, normalized))
        return True

    def complete(self) -> HudProfile:
        configuration = self._configuration
        if configuration is None:
            raise RuntimeError("calibration configuration is incomplete")
        if any(len(batch) != _STABLE_SAMPLE_COUNT for batch in self._batches.values()):
            raise RuntimeError("all calibration batches must contain five samples")

        p1 = self._build_references(
            Side.P1,
            [entry.crop_bgr for entry in self._batches[(Side.P1, "full")]],
            [entry.crop_bgr for entry in self._batches[(Side.P1, "spent")]],
        )
        p2 = self._build_references(
            Side.P2,
            [entry.crop_bgr for entry in self._batches[(Side.P2, "full")]],
            [entry.crop_bgr for entry in self._batches[(Side.P2, "spent")]],
        )
        if not isinstance(p1, MeterReferences) or p1.side is not Side.P1:
            raise ValueError("P1 builder returned invalid references")
        if not isinstance(p2, MeterReferences) or p2.side is not Side.P2:
            raise ValueError("P2 builder returned invalid references")
        if p1 is p2:
            raise ValueError("reference builders must return distinct values")
        p1_color = self._build_color_model(
            Side.P1,
            [entry.crop_bgr for entry in self._batches[(Side.P1, "full")]],
            [entry.crop_bgr for entry in self._batches[(Side.P1, "spent")]],
        )
        p2_color = self._build_color_model(
            Side.P2,
            [entry.crop_bgr for entry in self._batches[(Side.P2, "full")]],
            [entry.crop_bgr for entry in self._batches[(Side.P2, "spent")]],
        )
        if not isinstance(p1_color, MeterColorModel) or p1_color.side is not Side.P1:
            raise ValueError("P1 colour builder returned invalid model")
        if not isinstance(p2_color, MeterColorModel) or p2_color.side is not Side.P2:
            raise ValueError("P2 colour builder returned invalid model")
        p1 = MeterReferences(p1.side, p1.full_gray, p1.spent_gray, color_model=p1_color)
        p2 = MeterReferences(p2.side, p2.full_gray, p2.spent_gray, color_model=p2_color)

        attempt = self._attempt_id_factory()
        if type(attempt) is not str or not attempt.strip():
            raise ValueError("attempt id must be a nonblank string")
        profile_id = configuration["profile_id"]
        if type(profile_id) is not str or not profile_id.strip():
            raise ValueError("profile_id must be a nonblank string")
        p1_path = f"refs/{profile_id}.{attempt}.p1.npz"
        p2_path = f"refs/{profile_id}.{attempt}.p2.npz"
        profile_target = resolve_contained_path(
            self.user_data_root, self.profile_path, field_name="profile_path"
        )
        targets = (
            resolve_contained_path(
                profile_target.parent,
                p1_path,
                field_name="p1.references_path",
                required_suffix=".npz",
            ),
            resolve_contained_path(
                profile_target.parent,
                p2_path,
                field_name="p2.references_path",
                required_suffix=".npz",
            ),
        )
        if targets[0] == targets[1]:
            raise ValueError("calibration attempt paths must be distinct")

        reserved_targets: list[Path] = []
        try:
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb"):
                    pass
                reserved_targets.append(target)
            self._save_references(
                self.user_data_root, self.profile_path, p1_path, p1
            )
            self._save_references(
                self.user_data_root, self.profile_path, p2_path, p2
            )
            loaded_p1 = self._load_references(
                self.user_data_root,
                self.profile_path,
                p1_path,
                expected_side=Side.P1,
            )
            loaded_p2 = self._load_references(
                self.user_data_root,
                self.profile_path,
                p2_path,
                expected_side=Side.P2,
            )
            self._require_same_references(p1, loaded_p1)
            self._require_same_references(p2, loaded_p2)
            profile = HudProfile(
                profile_id=profile_id,
                calibration=configuration["calibration"],
                battle_roi=configuration["battle_roi"],
                training_hud_roi=configuration["training_hud_roi"],
                p1=SideHudProfile(configuration["p1_roi"], p1_path),
                p2=SideHudProfile(configuration["p2_roi"], p2_path),
            )
            self._save_profile(self.user_data_root, self.profile_path, profile)
            return profile
        except Exception as original_error:
            cleanup_error: OSError | None = None
            for target in reserved_targets:
                try:
                    target.unlink(missing_ok=True)
                except OSError as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None:
                raise cleanup_error from original_error
            raise

    @staticmethod
    def _require_same_references(
        expected: MeterReferences, actual: MeterReferences
    ) -> None:
        if not isinstance(actual, MeterReferences) or actual.side is not expected.side:
            raise ValueError("reloaded references have the wrong side")
        if (
            actual.schema_version != expected.schema_version
            or actual.preprocessing_version != expected.preprocessing_version
            or not np.array_equal(actual.full_gray, expected.full_gray)
            or not np.array_equal(actual.spent_gray, expected.spent_gray)
        ):
            raise ValueError("reloaded references do not match the saved references")
        expected_model = expected.color_model
        actual_model = actual.color_model
        if (expected_model is None) != (actual_model is None):
            raise ValueError("reloaded colour model does not match saved references")
        if expected_model is not None and actual_model is not None and (
            actual_model.side is not expected_model.side
            or not np.array_equal(actual_model.active_mask, expected_model.active_mask)
            or (
                actual_model.full_hue,
                actual_model.hue_tolerance,
                actual_model.saturation_floor,
                actual_model.value_floor,
                actual_model.full_fill,
                actual_model.spent_fill,
            )
            != (
                expected_model.full_hue,
                expected_model.hue_tolerance,
                expected_model.saturation_floor,
                expected_model.value_floor,
                expected_model.full_fill,
                expected_model.spent_fill,
            )
        ):
            raise ValueError("reloaded colour model does not match saved references")

    def cancel(self) -> None:
        self._clear_batches()
        self.gate = None


def _build_listener_controller(
    application: PanelApplication,
    *,
    audio_diagnostic: Callable[[AudioPlaybackError], None],
) -> CapturedFrame:
    """Build one fresh runtime session without importing device backends early."""

    if application.view_model.detection_backend is DetectionBackend.REFRAMEWORK:
        return _build_ref_listener_controller(application, audio_diagnostic=audio_diagnostic)

    from .audio import AudioScheduler
    from .capture import DxcamFrameSource, FreshFrameGate
    from .controller import ListenerController, TrackerDiagnostics
    from .diagnostic_log import DiagnosticEventLog
    from .engine import VisualTriggerEngine
    from .gauge_tracker import GaugeTracker, GaugeTrackerConfig
    from .viewport import CenteredViewportResolver

    view_model = application.view_model
    if not view_model.can_start or view_model.selected_profile is None:
        raise ValueError(view_model.validation_message or "panel selections are invalid")
    profile = view_model.selected_profile
    config = view_model.to_config()
    p1_references = load_references(
        application.user_data_root,
        view_model.profile_path,
        profile.p1.references_path,
        expected_side=Side.P1,
    )
    p2_references = load_references(
        application.user_data_root,
        view_model.profile_path,
        profile.p2.references_path,
        expected_side=Side.P2,
    )
    source = DxcamFrameSource(target_fps=config.capture_fps)
    p1_tracker = GaugeTracker(
        Side.P1, profile.profile_id, _tracker_config_for_references(p1_references)
    )
    p2_tracker = GaugeTracker(
        Side.P2, profile.profile_id, _tracker_config_for_references(p2_references)
    )
    diagnostic_log = DiagnosticEventLog(diagnostic_data_root())
    engine = VisualTriggerEngine(
        profile=profile,
        p1_references=p1_references,
        p2_references=p2_references,
        p1_tracker=p1_tracker,
        p2_tracker=p2_tracker,
        freshness_gate=FreshFrameGate(),
        p1_scheduler=AudioScheduler(Side.P1),
        p2_scheduler=AudioScheduler(Side.P2),
        p1_audio_config=config.p1,
        p2_audio_config=config.p2,
        audio_port=application.audio_port,
        audio_diagnostic=audio_diagnostic,
        diagnostic_sink=diagnostic_log.record,
        viewport_resolver=CenteredViewportResolver(
            profile.calibration.client_aspect_ratio,
            confirmation_frames=3,
            minimum_content_height=720,
        ),
    )
    return ListenerController(
        source=source,
        engine=engine,
        audio_port=application.audio_port,
        diagnostic_log=diagnostic_log,
        diagnostics_provider=lambda: TrackerDiagnostics(
            (p1_tracker.state, p2_tracker.state),
            (p1_tracker.stable_meter_level, p2_tracker.stable_meter_level),
            processing_reason=engine.processing_reason,
            viewport_status=engine.viewport_status,
        ),
    )


def _build_ref_listener_controller(
    application: PanelApplication,
    *,
    audio_diagnostic: Callable[[AudioPlaybackError], None],
) -> object:
    """Listener fed by the in-game REFramework detector instead of the screen."""

    from .controller import ListenerController
    from .ref_engine import RefTriggerEngine
    from .reframework_events import RefEventSource

    view_model = application.view_model
    if not view_model.can_start:
        raise ValueError(view_model.validation_message or "panel selections are invalid")
    config = view_model.to_config()
    engine = RefTriggerEngine(
        p1_audio_config=config.p1,
        p2_audio_config=config.p2,
        audio_port=application.audio_port,
        audio_diagnostic=audio_diagnostic,
    )
    return ListenerController(
        source=RefEventSource(config.game_dir),
        engine=engine,
        audio_port=application.audio_port,
        runtime_message_provider=lambda: ref_runtime_message(
            engine.connection,
            engine.chara,
            engine.outcome_statuses,
            p1_enabled=config.p1.enabled,
            p2_enabled=config.p2.enabled,
        ),
    )


_PROVISIONAL_OUTCOME_LABELS = frozenset(
    {"疑似大招，正在确认", "疑似大招已撤销"}
)
_DIAGNOSTIC_LOG_WARNING = "诊断日志不可用（不影响检测）"


def _provisional_outcome_message(
    outcomes: tuple[str | None, str | None],
) -> str | None:
    if type(outcomes) is not tuple or len(outcomes) != 2:
        return None
    messages = [
        f"{side.value}：{outcome}"
        for side, outcome in zip((Side.P1, Side.P2), outcomes, strict=True)
        if type(outcome) is str and outcome in _PROVISIONAL_OUTCOME_LABELS
    ]
    return "；".join(messages) or None


def _controller_warning_message(status: object) -> str | None:
    from .controller import ControllerStatus

    if not isinstance(status, ControllerStatus):
        raise TypeError("status must be ControllerStatus")
    if status.kind == "diagnostic_error":
        return _DIAGNOSTIC_LOG_WARNING
    return None


def _capture_waiting_message(status: object) -> str | None:
    from .controller import ControllerStatus

    if not isinstance(status, ControllerStatus):
        raise TypeError("status must be ControllerStatus")
    if status.kind != "capture_waiting":
        return None
    if status.capture_status is CaptureStatus.MINIMIZED:
        return "正在等待 SF6：请恢复游戏窗口"
    if status.capture_status is CaptureStatus.OCCLUDED:
        return "正在等待 SF6：请保持游戏窗口可见"
    messages = {
        CaptureIssue.WINDOW_CHANGED: "正在等待 SF6：未找到稳定的游戏窗口",
        CaptureIssue.GEOMETRY_CHANGED: "正在等待 SF6：显示模式正在变化",
        CaptureIssue.UNSUPPORTED_OUTPUT: "正在等待 SF6：请将游戏移到单个主显示器",
        CaptureIssue.BACKEND_ERROR: "正在等待 SF6：捕获后端暂不可用，正在重试",
    }
    return messages.get(
        status.capture_issue,
        "正在等待 SF6：捕获暂时不可用，正在重试",
    )


@dataclass
class _PanelStatusLatch:
    diagnostic_warning: bool = False

    def reset(self) -> None:
        self.diagnostic_warning = False

    def resolve(self, statuses: tuple[object, ...], *, fallback: str) -> str:
        from .controller import ControllerStatus

        if type(statuses) is not tuple or not all(
            isinstance(status, ControllerStatus) for status in statuses
        ):
            raise TypeError("statuses must contain ControllerStatus values")
        if type(fallback) is not str:
            raise TypeError("fallback must be a string")
        if any(
            _controller_warning_message(status) is not None
            for status in statuses
        ):
            self.diagnostic_warning = True
        audio_errors = [status for status in statuses if status.kind == "audio_error"]
        if audio_errors:
            status = audio_errors[-1]
            side = "" if status.side is None else f"/{status.side.value}"
            message = f"音频错误: {status.operation}{side}/{status.issue.value}"
        elif any(status.kind == "worker_error" for status in statuses):
            message = "监听器错误：请检查 SF6 窗口和本地配置"
        elif any(status.kind == "stopped" for status in statuses):
            message = "已停止"
        else:
            waiting_messages = [
                message
                for status in statuses
                if (message := _capture_waiting_message(status)) is not None
            ]
            message = waiting_messages[-1] if waiting_messages else fallback
        if self.diagnostic_warning:
            message = f"{message}；{_DIAGNOSTIC_LOG_WARNING}"
        return message


def _portable_selected_path(root: Path, selected: str) -> str:
    target = Path(selected).resolve(strict=False)
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)


def _parse_roi(text: object, *, field_name: str) -> NormalizedRoi:
    if type(text) is not str:
        raise TypeError(f"{field_name} must be text")
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{field_name} must contain x0,y0,x1,y1")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"{field_name} must contain numeric coordinates") from error
    return NormalizedRoi(*values)


def _calibration_preview_rois(
    *,
    battle_roi: NormalizedRoi,
    p1_roi: NormalizedRoi,
    p2_roi: NormalizedRoi,
    training_roi: NormalizedRoi | None,
    supports_meter_levels: bool,
) -> dict[str, NormalizedRoi]:
    """Build a deterministic calibration overlay without adding a sample step."""

    if type(supports_meter_levels) is not bool:
        raise TypeError("supports_meter_levels must be a boolean")
    visible_rois = {"Battle": battle_roi, "P1": p1_roi, "P2": p2_roi}
    if training_roi is not None:
        visible_rois["Training"] = training_roi
    if not supports_meter_levels:
        return visible_rois
    for label, roi, side in (
        ("P1 LEVEL", p1_roi, Side.P1),
        ("P2 LEVEL", p2_roi, Side.P2),
    ):
        try:
            visible_rois[label] = derive_meter_level_roi(roi, side)
        except ValueError:
            # A boundary-touching legacy ROI has no safe outward band.
            continue
    return visible_rois


def _capture_calibration_batch(
    *,
    source: object,
    session: CalibrationSession,
    side: Side,
    phase: str,
    roi: NormalizedRoi,
    max_attempts: int = 400,
) -> ClientGeometry:
    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("calibration source must provide read")
    last_geometry: ClientGeometry | None = None
    empty_crop = np.zeros((1, 1, 3), dtype=np.uint8)
    for _ in range(max_attempts):
        sample = read()
        if not isinstance(sample, CaptureSample):
            raise TypeError("calibration source must return CaptureSample")
        if sample.status is CaptureStatus.RUNNING and sample.frame is not None:
            crop = crop_normalized(sample.frame.image, roi)
        else:
            crop = empty_crop
        accepted = session.accept_crop(side, phase, sample, crop)
        if accepted and sample.frame is not None:
            last_geometry = sample.frame.geometry
        if session.batch_size(side, phase) == _STABLE_SAMPLE_COUNT:
            if last_geometry is None:
                raise RuntimeError("stable calibration batch has no geometry")
            return last_geometry
    raise RuntimeError("unable to collect five stable calibration frames")


def _read_calibration_context_frame(
    *,
    source: object,
    session: CalibrationSession,
    max_attempts: int = 400,
) -> object:
    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("calibration source must provide read")
    for _ in range(max_attempts):
        sample = read()
        if not isinstance(sample, CaptureSample):
            raise TypeError("calibration source must return CaptureSample")
        frame = session.accept_context_frame(sample)
        if frame is not None:
            return frame
    raise RuntimeError("unable to obtain a fresh calibration context frame")


def _run_calibration_wizard(
    *,
    parent: object,
    application: PanelApplication,
    status_sink: Callable[[str], object],
    source_factory: Callable[[int], object] | None = None,
    gate_factory: Callable[[], object] | None = None,
    calibration_clock: Callable[[], Real] = time.perf_counter,
    dialogs: object | None = None,
    attempt_id_factory: Callable[[], str] | None = None,
) -> bool:
    """Collect an explicit four-batch SF6 calibration without hidden defaults."""

    if not isinstance(application, PanelApplication):
        raise TypeError("application must be PanelApplication")
    if not callable(status_sink) or not callable(calibration_clock):
        raise TypeError("calibration callbacks must be callable")
    if dialogs is None:
        import base64
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk

        import cv2

        def confirm_rois(
            dialog_parent: object,
            image_bgr: np.ndarray,
            rois: dict[str, NormalizedRoi],
        ) -> bool:
            success, encoded = cv2.imencode(".png", image_bgr)
            if not success:
                raise RuntimeError("unable to render calibration frame")
            encoded_text = base64.b64encode(encoded.tobytes()).decode("ascii")
            original = tk.PhotoImage(data=encoded_text)
            factor = max(
                1,
                math.ceil(original.width() / 960),
                math.ceil(original.height() / 540),
            )
            shown = original if factor == 1 else original.subsample(factor, factor)
            window = tk.Toplevel(dialog_parent)
            window.title("确认 SF6 HUD ROI")
            canvas = tk.Canvas(
                window,
                width=shown.width(),
                height=shown.height(),
                highlightthickness=0,
            )
            canvas.pack(padx=10, pady=10)
            canvas.create_image(0, 0, image=shown, anchor=tk.NW)
            colors = {
                "Battle": "#00d7ff",
                "P1": "#5cff65",
                "P2": "#ff5ca8",
                "Training": "#ffd65c",
            }
            for label, roi in rois.items():
                color = colors[label]
                x0 = roi.x0 * shown.width()
                y0 = roi.y0 * shown.height()
                x1 = roi.x1 * shown.width()
                y1 = roi.y1 * shown.height()
                canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)
                canvas.create_text(
                    x0 + 4,
                    y0 + 4,
                    text=label,
                    fill=color,
                    anchor=tk.NW,
                )
            accepted = [False]

            def finish(value: bool) -> None:
                accepted[0] = value
                window.destroy()

            buttons = ttk.Frame(window, padding=(10, 0, 10, 10))
            buttons.pack(fill="x")
            ttk.Button(
                buttons,
                text="确认 ROI",
                command=lambda: finish(True),
            ).pack(side="left", expand=True, fill="x", padx=3)
            ttk.Button(
                buttons,
                text="取消",
                command=lambda: finish(False),
            ).pack(side="left", expand=True, fill="x", padx=3)
            window.protocol("WM_DELETE_WINDOW", lambda: finish(False))
            window.transient(dialog_parent)
            window.grab_set()
            window.wait_window()
            return accepted[0]

        class _Dialogs:
            askstring = staticmethod(simpledialog.askstring)
            askyesno = staticmethod(messagebox.askyesno)
            askokcancel = staticmethod(messagebox.askokcancel)
            showinfo = staticmethod(messagebox.showinfo)
            showerror = staticmethod(messagebox.showerror)
            confirm_rois = staticmethod(confirm_rois)

        dialogs = _Dialogs

    askstring = getattr(dialogs, "askstring", None)
    askyesno = getattr(dialogs, "askyesno", None)
    askokcancel = getattr(dialogs, "askokcancel", None)
    showinfo = getattr(dialogs, "showinfo", None)
    showerror = getattr(dialogs, "showerror", None)
    confirm_rois = getattr(dialogs, "confirm_rois", None)
    if not all(
        callable(value)
        for value in (
            askstring,
            askyesno,
            askokcancel,
            showinfo,
            showerror,
            confirm_rois,
        )
    ):
        raise TypeError("dialogs must provide the calibration prompt methods")

    prompts = (
        ("Profile ID", "task9-profile"),
        ("HUD scale", "default"),
        ("Language", "English"),
        ("Brightness", "default"),
        ("Game version note", ""),
    )
    answers: dict[str, str] = {}
    for prompt, initial in prompts:
        answer = askstring(
            "SF6 HUD 校准",
            prompt,
            parent=parent,
            initialvalue=initial,
        )
        if answer is None:
            status_sink("HUD 校准已取消")
            return False
        answers[prompt] = answer
    hdr_enabled = askyesno("SF6 HUD 校准", "HDR enabled", parent=parent)

    if gate_factory is None or source_factory is None:
        from .capture import DxcamFrameSource, FreshFrameGate

        if gate_factory is None:
            gate_factory = FreshFrameGate
        if source_factory is None:
            source_factory = lambda fps: DxcamFrameSource(target_fps=fps)
    if not callable(gate_factory) or not callable(source_factory):
        raise TypeError("calibration factories must be callable")

    source = source_factory(application.view_model.capture_fps)
    start_source = getattr(source, "start", None)
    stop_source = getattr(source, "stop", None)
    if not callable(start_source) or not callable(stop_source):
        raise TypeError("calibration source must provide start and stop")
    session = CalibrationSession(
        user_data_root=application.user_data_root,
        profile_path=application.view_model.profile_path,
        engine_gate=gate_factory(),
        calibration_gate_factory=gate_factory,
        calibration_clock=calibration_clock,
        attempt_id_factory=attempt_id_factory,
    )
    geometries: list[ClientGeometry] = []
    try:
        session.begin()
        start_source()
        context_frame = _read_calibration_context_frame(
            source=source,
            session=session,
        )
        geometries.append(context_frame.geometry)
        roi_answers: dict[str, str] = {}
        for prompt in (
            "Battle ROI",
            "P1 meter ROI",
            "P2 meter ROI",
            "Training HUD ROI",
        ):
            answer = askstring(
                "SF6 HUD 校准",
                prompt,
                parent=parent,
                initialvalue="",
            )
            if answer is None:
                session.cancel()
                status_sink("HUD 校准已取消")
                return False
            roi_answers[prompt] = answer
        battle_roi = _parse_roi(roi_answers["Battle ROI"], field_name="battle_roi")
        p1_roi = _parse_roi(roi_answers["P1 meter ROI"], field_name="p1_roi")
        p2_roi = _parse_roi(roi_answers["P2 meter ROI"], field_name="p2_roi")
        training_text = roi_answers["Training HUD ROI"].strip()
        training_roi = (
            None
            if not training_text
            else _parse_roi(training_text, field_name="training_hud_roi")
        )
        visible_rois = _calibration_preview_rois(
            battle_roi=battle_roi,
            p1_roi=p1_roi,
            p2_roi=p2_roi,
            training_roi=training_roi,
            supports_meter_levels=True,
        )
        if not confirm_rois(parent, context_frame.image, visible_rois):
            session.cancel()
            status_sink("HUD 校准已取消")
            return False
        steps = (
            (Side.P1, "full", p1_roi, "P1 满气状态"),
            (Side.P1, "spent", p1_roi, "P1 消耗 SA3/CA 后状态"),
            (Side.P2, "full", p2_roi, "P2 满气状态"),
            (Side.P2, "spent", p2_roi, "P2 消耗 SA3/CA 后状态"),
        )
        for side, phase, roi, instruction in steps:
            if not askokcancel(
                "SF6 HUD 校准",
                f"请在 SF6 训练模式准备：{instruction}，然后继续。",
                parent=parent,
            ):
                session.cancel()
                status_sink("HUD 校准已取消")
                return False
            geometries.append(
                _capture_calibration_batch(
                    source=source,
                    session=session,
                    side=side,
                    phase=phase,
                    roi=roi,
                )
            )
        sizes = {(geometry.width, geometry.height) for geometry in geometries}
        if len(sizes) != 1:
            raise ValueError("client size changed across calibration phases")
        width, height = sizes.pop()
        session.configure(
            profile_id=answers["Profile ID"],
            calibration=CalibrationContext(
                client_width=width,
                client_height=height,
                client_aspect_ratio=width / height,
                hud_scale_setting=answers["HUD scale"],
                language=answers["Language"],
                brightness_setting=answers["Brightness"],
                hdr_enabled=bool(hdr_enabled),
                game_version_note=answers["Game version note"],
            ),
            battle_roi=battle_roi,
            training_hud_roi=training_roi,
            p1_roi=p1_roi,
            p2_roi=p2_roi,
        )
        profile = session.complete()
        application.view_model.selected_profile = profile
        application.view_model.current_geometry = geometries[-1]
        status_sink("HUD 校准完成")
        showinfo("SF6 HUD 校准", "P1/P2 HUD 校准已保存", parent=parent)
        return True
    except Exception:
        session.cancel()
        status_sink("HUD 校准失败")
        showerror(
            "SF6 HUD 校准",
            "校准未完成；旧配置仍保持不变",
            parent=parent,
        )
        return False
    finally:
        stop_source()


def run_panel(config_path: str | Path) -> None:
    """Run the persistent Tk control panel; imports and devices stay lazy until use."""

    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk

    from .game_ui import GameUiBridge, apply_ui_command, build_app_state, collect_tracks

    status_text = runtime_status_message(None, None, None)
    pending_audio_errors: list[AudioPlaybackError] = []

    def collect_audio_error(error: AudioPlaybackError) -> None:
        pending_audio_errors.append(error)

    application = create_panel_application(
        config_path=config_path,
        status_sink=collect_audio_error,
    )
    view_model = application.view_model
    root = tk.Tk()
    from . import BUILD_ID
    root.title(f"SF6 Music Trigger — {BUILD_ID}")
    root.minsize(760, 560)

    if not view_model.game_dir:
        try:
            found_game_dir = find_sf6_install()
        except OSError:
            found_game_dir = None
        if found_game_dir is not None:
            view_model.game_dir = str(found_game_dir)

    backend_var = tk.StringVar(value=_BACKEND_LABELS[view_model.detection_backend])
    game_dir_var = tk.StringVar(value=view_model.game_dir)
    profile_var = tk.StringVar(value=view_model.profile_path)
    audio_template_var = tk.StringVar(value=view_model.audio_template_names[0])
    unlimited_var = tk.BooleanVar(value=view_model.volume_unlimited)
    auto_start_var = tk.BooleanVar(value=view_model.auto_start)
    diagnostics_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value=status_text)
    panel_status = _PanelStatusLatch()
    controller_holder: list[object] = []

    def side_attr(side: Side, name: str) -> str:
        return f"{side.value.lower()}_{name}"

    side_vars: dict[Side, dict[str, object]] = {}
    for side in (Side.P1, Side.P2):
        side_vars[side] = {
            "enabled": tk.BooleanVar(value=getattr(view_model, side_attr(side, "enabled"))),
            "track": tk.StringVar(value=getattr(view_model, side_attr(side, "main_track"))),
            "track_name": tk.StringVar(),
            "volume": tk.DoubleVar(value=float(getattr(view_model, side_attr(side, "volume_percent")))),
            "scope": tk.StringVar(value=_SUPER_SCOPE_LABELS[getattr(view_model, side_attr(side, "super_scope"))]),
            "whiff_mode": tk.StringVar(
                value="sound" if getattr(view_model, side_attr(side, "play_whiff_sound")) else "fade"
            ),
            "whiff_track": tk.StringVar(value=getattr(view_model, side_attr(side, "whiff_track"))),
            "whiff_name": tk.StringVar(),
            "whiff_volume": tk.DoubleVar(
                value=float(getattr(view_model, side_attr(side, "whiff_volume_percent")))
            ),
        }
    volume_scales: list[object] = []

    def refresh_names() -> None:
        for vars_ in side_vars.values():
            vars_["track_name"].set(audio_track_display_name(vars_["track"].get()))
            vars_["whiff_name"].set(audio_track_display_name(vars_["whiff_track"].get()))

    def volume_limit() -> int:
        return MAX_VOLUME_PERCENT if unlimited_var.get() else NORMAL_VOLUME_PERCENT

    def sync_view_model() -> None:
        view_model.detection_backend = _BACKEND_FROM_LABEL[backend_var.get()]
        view_model.game_dir = game_dir_var.get().strip()
        view_model.profile_path = profile_var.get()
        view_model.volume_unlimited = bool(unlimited_var.get())
        view_model.auto_start = bool(auto_start_var.get())
        for side, vars_ in side_vars.items():
            setattr(view_model, side_attr(side, "enabled"), bool(vars_["enabled"].get()))
            setattr(view_model, side_attr(side, "main_track"), vars_["track"].get())
            setattr(view_model, side_attr(side, "volume_percent"), vars_["volume"].get())
            setattr(view_model, side_attr(side, "whiff_track"), vars_["whiff_track"].get())
            setattr(view_model, side_attr(side, "whiff_volume_percent"), vars_["whiff_volume"].get())
            setattr(view_model, side_attr(side, "super_scope"), _SUPER_SCOPE_FROM_LABEL[vars_["scope"].get()])
            view_model.set_whiff_sound_mode(side, vars_["whiff_mode"].get() == "sound")

    def refresh_vars_from_view_model() -> None:
        unlimited_var.set(view_model.volume_unlimited)
        apply_volume_limit()
        for side, vars_ in side_vars.items():
            vars_["enabled"].set(getattr(view_model, side_attr(side, "enabled")))
            vars_["track"].set(getattr(view_model, side_attr(side, "main_track")))
            vars_["volume"].set(float(getattr(view_model, side_attr(side, "volume_percent"))))
            vars_["scope"].set(_SUPER_SCOPE_LABELS[getattr(view_model, side_attr(side, "super_scope"))])
            vars_["whiff_mode"].set("sound" if getattr(view_model, side_attr(side, "play_whiff_sound")) else "fade")
            vars_["whiff_track"].set(getattr(view_model, side_attr(side, "whiff_track")))
            vars_["whiff_volume"].set(float(getattr(view_model, side_attr(side, "whiff_volume_percent"))))
        refresh_names()

    def apply_volume_limit() -> None:
        limit = volume_limit()
        for scale in volume_scales:
            scale.configure(to=limit)
        for vars_ in side_vars.values():
            for key in ("volume", "whiff_volume"):
                if vars_[key].get() > limit:
                    vars_[key].set(limit)

    def flush_audio_errors() -> None:
        if not pending_audio_errors:
            return
        error = pending_audio_errors[-1]
        pending_audio_errors.clear()
        side = "" if error.side is None else f"/{error.side.value}"
        status_var.set(f"音频错误: {error.operation}{side}/{error.issue.value}")

    def listener_running() -> bool:
        return bool(controller_holder) and controller_holder[0].is_running

    content = ttk.Frame(root, padding=12)
    content.grid(row=0, column=0, sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    content.columnconfigure(0, weight=1)
    content.columnconfigure(1, weight=1)

    # -- 游戏连接 ---------------------------------------------------------------
    connection = ttk.LabelFrame(content, text="游戏连接", padding=8)
    connection.grid(row=0, column=0, columnspan=2, sticky="ew")
    connection.columnconfigure(1, weight=1)
    ttk.Label(connection, text="检测方式").grid(row=0, column=0, sticky="w")
    backend_box = ttk.Combobox(
        connection, textvariable=backend_var, values=tuple(_BACKEND_FROM_LABEL), state="readonly"
    )
    backend_box.grid(row=0, column=1, columnspan=4, sticky="ew", padx=6)
    ttk.Label(connection, text="SF6 游戏目录").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(connection, textvariable=game_dir_var).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))

    def choose_game_dir() -> None:
        selected = filedialog.askdirectory(parent=root, mustexist=True)
        if selected:
            game_dir_var.set(str(Path(selected)))

    def one_click_install() -> None:
        from .game_install import bundled_reframework_dir, inspect_game, install_all

        if listener_running():
            status_var.set("请先停止监听，再安装")
            return
        game_dir = game_dir_var.get().strip()
        bundle = bundled_reframework_dir()
        state = inspect_game(game_dir, bundle)
        if not state.game_found:
            status_var.set("请先选择 SF6 游戏目录（包含 StreetFighter6.exe 的文件夹）")
            return
        replace = False
        if state.reframework == "other" and bundle is not None:
            keep = messagebox.askyesnocancel(
                "一键安装",
                "游戏目录里已经有一个 dinput8.dll（可能是你自己装的 REFramework，"
                "也可能属于别的 Mod）。\n\n"
                "是：保留它，只安装 V2 的游戏内脚本（推荐）\n"
                "否：备份后换成 V2 附带的 REFramework",
                parent=root,
            )
            if keep is None:
                return
            replace = not keep
        try:
            report = install_all(game_dir, bundle, replace_other_dll=replace)
        except FileNotFoundError:
            status_var.set("没有找到附带的 REFramework，请确认发布包完整（REFramework 文件夹）")
            return
        except PermissionError:
            status_var.set("文件被占用：请先关闭游戏再安装")
            return
        except (OSError, ValueError):
            status_var.set("安装失败：请确认已选择 SF6 游戏目录")
            return
        notes = {
            "installed": "已安装 REFramework 和游戏内脚本",
            "replaced": f"已替换为附带的 REFramework（原文件备份为 {report.backup}），并安装游戏内脚本",
            "kept": "保留了现有的 dinput8.dll，已安装游戏内脚本",
            "unchanged": "REFramework 已是附带版本，游戏内脚本已更新",
        }
        status_var.set(notes[report.reframework] + "；游戏开着的话请重开游戏，或在 REFramework 菜单的 ScriptRunner 里点 Reset scripts")

    def remove_from_game_clicked() -> None:
        from .game_install import bundled_reframework_dir, inspect_game, remove_from_game

        if listener_running():
            status_var.set("请先停止监听，再从游戏中移除")
            return
        game_dir = game_dir_var.get().strip()
        state = inspect_game(game_dir, bundled_reframework_dir())
        if not state.game_found:
            status_var.set("请先选择 SF6 游戏目录（包含 StreetFighter6.exe 的文件夹）")
            return
        if not messagebox.askyesno(
            "从游戏中移除",
            "将删除 V2 放进游戏的两个脚本和日志文件夹（reframework\\data\\SF6MusicTrigger）。\n"
            "游戏本身的文件不会被改动。继续吗？",
            parent=root,
        ):
            return
        remove_reframework = False
        if state.reframework != "missing":
            if state.other_mods:
                messagebox.showinfo(
                    "从游戏中移除",
                    "游戏里还有其他 REFramework Mod，所以会保留 REFramework（dinput8.dll）。",
                    parent=root,
                )
            else:
                remove_reframework = messagebox.askyesno(
                    "从游戏中移除",
                    "没有发现其他 REFramework Mod。要同时移除 REFramework 吗？\n\n"
                    "是：删除 dinput8.dll 和它的配置/日志，游戏目录恢复原样\n"
                    "否：保留 REFramework",
                    parent=root,
                )
        try:
            report = remove_from_game(game_dir, remove_reframework=remove_reframework)
        except PermissionError:
            status_var.set("文件被占用：请先关闭游戏再移除")
            return
        except (OSError, ValueError):
            status_var.set("移除失败：请关闭游戏后重试")
            return
        message = "已从游戏中移除 V2" + ("和 REFramework，游戏目录已恢复原样" if remove_reframework else "")
        if report.kept:
            message += f"；保留了其他工具的 {len(report.kept)} 个文件"
        status_var.set(message + "。不再使用时，直接删除 V2 程序文件夹即可。")

    ttk.Button(connection, text="选择目录", command=choose_game_dir).grid(row=1, column=2, pady=(6, 0))
    ttk.Button(connection, text="一键安装/更新", command=one_click_install).grid(
        row=1, column=3, padx=(6, 0), pady=(6, 0)
    )
    ttk.Button(connection, text="从游戏中移除", command=remove_from_game_clicked).grid(
        row=1, column=4, padx=(6, 0), pady=(6, 0)
    )
    ttk.Label(
        connection,
        text="提示：游戏里按 REFramework 菜单键（默认 Insert）打开菜单，会同时出现“SF6 音乐设置”窗口，可直接修改下面的设置。",
        foreground="#555555",
    ).grid(row=2, column=0, columnspan=5, sticky="w", pady=(6, 0))

    # -- 旧版画面识别（仅画面识别模式） -------------------------------------------
    legacy = ttk.LabelFrame(content, text="画面识别设置（旧版 T03）", padding=8)
    legacy.columnconfigure(1, weight=1)
    ttk.Label(legacy, text="HUD 配置").grid(row=0, column=0, sticky="w")
    ttk.Entry(legacy, textvariable=profile_var).grid(row=0, column=1, sticky="ew", padx=6)

    def choose_profile() -> None:
        selected = filedialog.askopenfilename(
            parent=root,
            initialdir=application.user_data_root,
            filetypes=(("JSON", "*.json"),),
        )
        if not selected:
            return
        target = Path(selected).resolve(strict=False)
        try:
            profile_var.set(target.relative_to(application.user_data_root).as_posix())
        except ValueError:
            status_var.set("HUD 配置必须位于当前用户数据目录")

    def calibrate_hud() -> None:
        if listener_running():
            status_var.set("请先停止监听，再开始 HUD 校准")
            return
        sync_view_model()
        if view_model.detection_backend is DetectionBackend.REFRAMEWORK:
            status_var.set("REFramework 游戏内检测不需要校准 HUD")
            return
        if _run_calibration_wizard(
            parent=root,
            application=application,
            status_sink=status_var.set,
        ):
            profile_var.set(view_model.profile_path)

    ttk.Button(legacy, text="选择配置", command=choose_profile).grid(row=0, column=2)
    ttk.Button(legacy, text="校准 HUD", command=lambda: view_model.calibrate()).grid(
        row=0, column=3, padx=(6, 0)
    )

    def update_backend_sections(_event: object = None) -> None:
        if _BACKEND_FROM_LABEL[backend_var.get()] is DetectionBackend.SCREEN:
            legacy.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        else:
            legacy.grid_remove()

    backend_box.bind("<<ComboboxSelected>>", update_backend_sections)

    # -- 音频模板 ---------------------------------------------------------------
    templates = ttk.LabelFrame(content, text="音频模板", padding=8)
    templates.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    templates.columnconfigure(0, weight=1)
    template_box = ttk.Combobox(
        templates, textvariable=audio_template_var, values=view_model.audio_template_names, state="readonly"
    )
    template_box.grid(row=0, column=0, sticky="ew")

    def refresh_audio_template_choices(selected: str | None = None) -> None:
        names = view_model.audio_template_names
        template_box.configure(values=names)
        if selected in names:
            audio_template_var.set(selected)
        elif audio_template_var.get() not in names:
            audio_template_var.set(names[0] if names else "")

    def apply_selected_audio_template() -> None:
        if not view_model.apply_audio_template(audio_template_var.get()):
            status_var.set(view_model.validation_message)
            return
        refresh_vars_from_view_model()
        status_var.set(view_model.validation_message)

    def save_selected_audio_template() -> None:
        sync_view_model()
        view_model.save_audio_template(audio_template_var.get())
        status_var.set(view_model.validation_message)

    def add_custom_audio_template() -> None:
        name = simpledialog.askstring("新增音频模板", "请输入模板名称：", parent=root)
        if name is None:
            return
        sync_view_model()
        if view_model.add_audio_template(name):
            refresh_audio_template_choices(name.strip())
        status_var.set(view_model.validation_message)

    def rename_selected_audio_template() -> None:
        old_name = audio_template_var.get()
        name = simpledialog.askstring(
            "重命名音频模板", "请输入新名称：", initialvalue=old_name, parent=root
        )
        if name is None:
            return
        if view_model.rename_audio_template(old_name, name):
            refresh_audio_template_choices(name.strip())
        status_var.set(view_model.validation_message)

    for column, (label, command) in enumerate(
        (
            ("应用", apply_selected_audio_template),
            ("保存", save_selected_audio_template),
            ("新增", add_custom_audio_template),
            ("改名", rename_selected_audio_template),
        ),
        start=1,
    ):
        ttk.Button(templates, text=label, command=command, width=6).grid(row=0, column=column, padx=(6, 0))

    # -- P1 / P2 ----------------------------------------------------------------
    def choose_audio(var: object, name_var: object) -> None:
        selected = filedialog.askopenfilename(
            parent=root,
            initialdir=application.user_data_root,
            filetypes=(("Audio", "*.wav *.ogg *.mp3"), ("All files", "*.*")),
        )
        if selected:
            stored = _portable_selected_path(application.user_data_root, selected)
            var.set(stored)
            name_var.set(audio_track_display_name(stored))

    def preview(side: Side, whiff: bool) -> None:
        sync_view_model()
        ok = view_model.preview_whiff(side) if whiff else view_model.preview(side)
        if ok:
            status_var.set(f"{side.value} {'打空音效' if whiff else '主音乐'}试听")
        flush_audio_errors()

    def build_side(column: int, side: Side) -> None:
        vars_ = side_vars[side]
        box = ttk.LabelFrame(content, text=side.value, padding=8)
        box.grid(row=3, column=column, sticky="nsew", pady=(8, 0), padx=(0, 4) if column == 0 else (4, 0))
        box.columnconfigure(1, weight=1)
        ttk.Checkbutton(box, text=f"启用 {side.value}", variable=vars_["enabled"]).grid(
            row=0, column=0, columnspan=4, sticky="w"
        )

        def audio_row(row: int, label: str, var: str, name: str, whiff: bool) -> None:
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(box, textvariable=vars_[name], state="readonly").grid(
                row=row, column=1, sticky="ew", padx=6, pady=(6, 0)
            )
            ttk.Button(box, text="选择", width=5, command=lambda: choose_audio(vars_[var], vars_[name])).grid(
                row=row, column=2, pady=(6, 0)
            )
            ttk.Button(box, text="试听", width=5, command=lambda: preview(side, whiff)).grid(
                row=row, column=3, padx=(4, 0), pady=(6, 0)
            )

        def volume_row(row: int, label: str, var: str) -> None:
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w")
            scale = tk.Scale(
                box, from_=0, to=volume_limit(), orient=tk.HORIZONTAL, variable=vars_[var],
                showvalue=True, resolution=1, length=160, highlightthickness=0,
            )
            scale.grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
            ttk.Label(box, text="%").grid(row=row, column=3, sticky="w")
            volume_scales.append(scale)

        audio_row(1, "主音乐", "track", "track_name", False)
        volume_row(2, "音量", "volume")
        ttk.Label(box, text="大招范围").grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Combobox(
            box, textvariable=vars_["scope"], values=tuple(_SUPER_SCOPE_FROM_LABEL), state="readonly"
        ).grid(row=3, column=1, columnspan=3, sticky="ew", padx=6, pady=(4, 0))
        ttk.Label(box, text="全部大招时，SA1/SA2 播满 4 秒后淡出", foreground="#777777").grid(
            row=4, column=1, columnspan=3, sticky="w", padx=6
        )
        ttk.Label(box, text="空大/被防").grid(row=5, column=0, sticky="w", pady=(6, 0))
        modes = ttk.Frame(box)
        modes.grid(row=5, column=1, columnspan=3, sticky="w", padx=6, pady=(6, 0))
        ttk.Radiobutton(modes, text="播放打空音效", value="sound", variable=vars_["whiff_mode"]).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(modes, text="淡出主音乐", value="fade", variable=vars_["whiff_mode"]).grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )
        audio_row(6, "打空音效", "whiff_track", "whiff_name", True)
        volume_row(7, "音效音量", "whiff_volume")

    build_side(0, Side.P1)
    build_side(1, Side.P2)
    content.rowconfigure(3, weight=1)
    refresh_names()

    # -- 选项 -------------------------------------------------------------------
    options = ttk.LabelFrame(content, text="选项", padding=8)
    options.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def toggle_unlimited() -> None:
        view_model.set_volume_unlimited(bool(unlimited_var.get()))
        apply_volume_limit()

    def reset_volumes() -> None:
        view_model.reset_volumes()
        for vars_ in side_vars.values():
            vars_["volume"].set(NORMAL_VOLUME_PERCENT)
            vars_["whiff_volume"].set(NORMAL_VOLUME_PERCENT)
        status_var.set("音量已重置为 100%")

    ttk.Checkbutton(
        options, text="解除音量上限（最高 500%）", variable=unlimited_var, command=toggle_unlimited
    ).grid(row=0, column=0, sticky="w")
    ttk.Button(options, text="重置音量", command=reset_volumes).grid(row=0, column=1, padx=(12, 0))
    ttk.Checkbutton(
        options, text="启动后自动开始监听并最小化", variable=auto_start_var
    ).grid(row=0, column=2, sticky="w", padx=(18, 0))
    ttk.Checkbutton(options, text="显示诊断状态", variable=diagnostics_var).grid(
        row=1, column=0, sticky="w", pady=(6, 0)
    )

    def show_diagnostic_folder() -> None:
        sync_view_model()
        try:
            if view_model.detection_backend is DetectionBackend.REFRAMEWORK and is_game_dir(view_model.game_dir):
                # V2 evidence lives in the game: events.jsonl, ui_commands.jsonl, app_state.json.
                folder = Path(view_model.game_dir) / "reframework" / "data" / "SF6MusicTrigger"
                folder.mkdir(parents=True, exist_ok=True)
                os.startfile(str(folder))
            else:
                open_diagnostic_folder()
        except OSError:
            status_var.set("无法打开诊断文件夹")

    ttk.Button(options, text="打开诊断文件夹", command=show_diagnostic_folder).grid(
        row=1, column=1, padx=(12, 0), pady=(6, 0)
    )
    apply_volume_limit()

    # -- 控制 -------------------------------------------------------------------
    controls = ttk.Frame(content)
    controls.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
    for column in range(3):
        controls.columnconfigure(column, weight=1)

    start_button: object
    stop_button: object

    def start_listener() -> bool:
        nonlocal status_text
        sync_view_model()
        if not view_model.can_start:
            status_var.set(view_model.validation_message)
            return False
        panel_status.reset()
        status_var.set("正在启动监听：正在连接 SF6")
        root.update_idletasks()
        config = view_model.to_config()
        try:
            save_config(application.config_path, config)
            controller = _build_listener_controller(
                application,
                audio_diagnostic=collect_audio_error,
            )
            if not controller.start():
                status_var.set("监听器已在运行")
                return False
        except Exception:
            status_var.set("启动失败：请检查 SF6 游戏目录、HUD 配置和音频设置")
            return False
        controller_holder[:] = [controller]
        if view_model.detection_backend is DetectionBackend.REFRAMEWORK:
            status_text = ref_runtime_message(
                None, (None, None), (None, None),
                p1_enabled=view_model.p1_enabled, p2_enabled=view_model.p2_enabled,
            )
        else:
            status_text = runtime_status_message(
                controller.capture_status, None, None,
                p1_enabled=view_model.p1_enabled, p2_enabled=view_model.p2_enabled,
            )
        status_var.set(status_text)
        start_button.configure(state=tk.DISABLED)
        stop_button.configure(state=tk.NORMAL)
        settings_state["applied"] = config
        return True

    def stop_listener() -> None:
        if controller_holder:
            controller_holder[0].request_stop()
        status_var.set("正在停止")

    view_model.start_command = start_listener
    view_model.stop_command = stop_listener
    view_model.calibration_command = calibrate_hud

    start_button = ttk.Button(controls, text="开始监听", command=view_model.start)
    start_button.grid(row=0, column=0, sticky="ew", padx=3)
    stop_button = ttk.Button(controls, text="停止", command=view_model.stop, state=tk.DISABLED)
    stop_button.grid(row=0, column=1, sticky="ew", padx=3)

    def exit_panel() -> None:
        if controller_holder:
            controller_holder[0].close()
        else:
            try:
                application.audio_port.close()
            except AudioPlaybackError as error:
                collect_audio_error(error)
        root.destroy()

    view_model.exit_command = exit_panel
    ttk.Button(controls, text="退出", command=view_model.exit).grid(row=0, column=2, sticky="ew", padx=3)
    ttk.Label(content, textvariable=status_var, wraplength=720).grid(
        row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0)
    )

    # -- 设置同步：面板修改、游戏内修改 --------------------------------------------
    settings_state: dict[str, object] = {
        "applied": None, "saved": None, "changed_at": 0.0,
        "bridge": None, "bridge_dir": None, "published_at": 0.0, "tracks": {}, "tracks_at": 0.0,
    }

    def apply_settings_live(now: float) -> None:
        """Save changed settings and hand them to a running REFramework listener."""

        sync_view_model()
        try:
            config = view_model.to_config()
        except (TypeError, ValueError):
            return
        if config != settings_state["applied"]:
            settings_state["applied"] = config
            settings_state["changed_at"] = now
            settings_state["published_at"] = 0.0
            engine = getattr(controller_holder[0], "engine", None) if listener_running() else None
            update = getattr(engine, "update_configs", None)
            if callable(update):
                try:
                    update(config.p1, config.p2)
                except (TypeError, ValueError):
                    pass
        if config != settings_state["saved"] and now - settings_state["changed_at"] >= 0.5:
            try:
                save_config(application.config_path, config)
                settings_state["saved"] = config
            except (OSError, TypeError, ValueError):
                pass

    def game_bridge() -> GameUiBridge | None:
        if view_model.detection_backend is not DetectionBackend.REFRAMEWORK or not is_game_dir(view_model.game_dir):
            settings_state["bridge"] = None
            settings_state["bridge_dir"] = None
            return None
        if settings_state["bridge_dir"] != view_model.game_dir:
            bridge = GameUiBridge(view_model.game_dir)
            bridge.start()
            settings_state["bridge"] = bridge
            settings_state["bridge_dir"] = view_model.game_dir
        return settings_state["bridge"]

    def handle_game_commands(now: float) -> None:
        bridge = game_bridge()
        if bridge is None:
            return
        if now - settings_state["tracks_at"] >= 5.0 or not settings_state["tracks"]:
            settings_state["tracks"] = collect_tracks(view_model)
            settings_state["tracks_at"] = now
        for command in bridge.poll():
            if command.cmd == "start":
                if not listener_running():
                    start_listener()
            elif command.cmd == "stop":
                stop_listener()
            elif command.cmd == "preview":
                preview(command.side, command.value == "whiff")
            else:
                message = apply_ui_command(view_model, command, settings_state["tracks"])
                refresh_vars_from_view_model()
                refresh_audio_template_choices()
                if message:
                    status_var.set(message)
            settings_state["published_at"] = 0.0
        if now - settings_state["published_at"] >= 1.0:
            state = build_app_state(
                view_model,
                listening=listener_running(),
                status=status_var.get(),
                tracks=settings_state["tracks"],
                session=str(id(root)),
            )
            if bridge.publish(state):
                settings_state["published_at"] = now

    def poll_controller() -> None:
        now = time.monotonic()
        flush_audio_errors()
        try:
            handle_game_commands(now)
        except Exception:
            status_var.set("游戏内设置同步出错，已跳过本次更新")
        apply_settings_live(now)
        if controller_holder:
            controller = controller_holder[0]
            statuses = controller.drain_statuses()
            view_model.update_runtime_diagnostics(
                capture_status=controller.capture_status,
                current_geometry=controller.current_geometry,
                tracker_states=controller.tracker_states,
                tracker_levels=controller.tracker_levels,
                processing_reason=controller.processing_reason,
                viewport_status=controller.viewport_status,
            )
            for status in statuses:
                if status.kind == "stopped":
                    start_button.configure(state=tk.NORMAL)
                    stop_button.configure(state=tk.DISABLED)
            if not controller.is_running and any(status.kind == "stopped" for status in statuses):
                controller_holder.clear()
            fallback = status_var.get()
            runtime_message = getattr(controller, "runtime_message", None)
            if not statuses and runtime_message is not None:
                fallback = runtime_message
            elif not statuses:
                if diagnostics_var.get():
                    fallback = runtime_diagnostic_message(
                        view_model.capture_status,
                        view_model.p1_tracker_state,
                        view_model.p2_tracker_state,
                        p1_enabled=view_model.p1_enabled,
                        p2_enabled=view_model.p2_enabled,
                        p1_level=view_model.p1_meter_level,
                        p2_level=view_model.p2_meter_level,
                        current_geometry=view_model.current_geometry,
                        calibration=view_model.calibration_context,
                        viewport_status=view_model.viewport_status,
                        processing_reason=view_model.processing_reason,
                    )
                else:
                    provisional_message = _provisional_outcome_message(controller.outcome_statuses)
                    if provisional_message is not None:
                        fallback = provisional_message
                    else:
                        fallback = runtime_status_message(
                            view_model.capture_status,
                            view_model.p1_tracker_state,
                            view_model.p2_tracker_state,
                            p1_enabled=view_model.p1_enabled,
                            p2_enabled=view_model.p2_enabled,
                            p1_level=view_model.p1_meter_level,
                            p2_level=view_model.p2_meter_level,
                        )
            status_var.set(panel_status.resolve(statuses, fallback=fallback))
        root.after(100, poll_controller)

    def auto_start() -> None:
        if start_listener():
            root.iconify()

    update_backend_sections()
    try:
        initial = view_model.to_config()
    except (TypeError, ValueError):
        initial = None
    settings_state["saved"] = settings_state["applied"] = (
        initial if view_model.game_dir == application.config.game_dir else None
    )
    root.protocol("WM_DELETE_WINDOW", view_model.exit)
    root.after(100, poll_controller)
    if view_model.auto_start:
        root.after(800, auto_start)
    root.mainloop()
