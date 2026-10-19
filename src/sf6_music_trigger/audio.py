"""Typed audio policy and a lazy pygame playback boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import importlib
import math
from numbers import Rational, Real
import ntpath
import os
from pathlib import Path, PureWindowsPath
import threading
from typing import Literal, Protocol

from .config import MAX_VOLUME, SideAudioConfig, SuperScope
from .domain import (
    CancelledSuperCandidate,
    ConfirmedSuperEvent,
    Side,
    SuperCandidateEvent,
)
from .outcome import OutcomeDecision, OutcomeLabel


class AudioKind(str, Enum):
    STOP = "stop"
    PLAY = "play"
    FADE = "fade"


class AudioIssue(str, Enum):
    BACKEND_FAILURE = "backend_failure"
    ACTIVE_PORT_CONFLICT = "active_port_conflict"
    CLOSE_INCOMPLETE = "close_incomplete"
    TRACK_REJECTED = "track_rejected"


@dataclass(frozen=True)
class AudioAction:
    kind: AudioKind
    side: Side
    track: str | None = field(default=None, repr=False)
    volume: float | None = None
    fade_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AudioKind):
            raise TypeError("kind must be AudioKind")
        if not isinstance(self.side, Side):
            raise TypeError("side must be Side")

        if self.kind is AudioKind.STOP:
            if self.track is not None or self.volume is not None or self.fade_ms is not None:
                raise ValueError("STOP must not carry a track, volume, or fade duration")
            return

        if self.kind is AudioKind.FADE:
            if self.track is not None or self.volume is not None:
                raise ValueError("FADE must not carry a track or volume")
            if type(self.fade_ms) is not int:
                raise TypeError("FADE duration must be an integer")
            if self.fade_ms < 0:
                raise ValueError("FADE duration must not be negative")
            return

        if self.fade_ms is not None:
            raise ValueError("PLAY must not carry a fade duration")

        if not isinstance(self.track, str):
            raise TypeError("PLAY track must be a string")
        if self.track == "":
            raise ValueError("PLAY track must not be empty")
        if isinstance(self.volume, bool) or not isinstance(self.volume, Real):
            raise TypeError("PLAY volume must be a real number")
        if isinstance(self.volume, Rational):
            if self.volume < 0 or self.volume > MAX_VOLUME:
                raise ValueError("PLAY volume must be between zero and MAX_VOLUME")
        elif not math.isfinite(self.volume) or self.volume < 0 or self.volume > MAX_VOLUME:
            raise ValueError("PLAY volume must be finite and between zero and MAX_VOLUME")
        object.__setattr__(self, "volume", float(self.volume))

    def as_tuple(self) -> tuple[str, str | None, float | None]:
        return (self.kind.value, self.track, self.volume)


_AUDIO_OPERATIONS = frozenset(
    {
        "initialize",
        "execute",
        "stop",
        "load",
        "set_volume",
        "play",
        "fade",
        "preview",
        "preview_stop",
        "close_stop",
    }
)


class AudioPlaybackError(RuntimeError):
    operation: Literal[
        "initialize",
        "execute",
        "stop",
        "load",
        "set_volume",
        "play",
        "fade",
        "preview",
        "preview_stop",
        "close_stop",
    ]
    side: Side | None
    issue: AudioIssue

    def __init__(
        self,
        operation: Literal[
            "initialize",
            "execute",
            "stop",
            "load",
            "set_volume",
            "play",
            "fade",
            "preview",
            "preview_stop",
            "close_stop",
        ],
        side: Side | None,
        issue: AudioIssue,
    ) -> None:
        if type(operation) is not str:
            raise TypeError("operation must be a string")
        if operation not in _AUDIO_OPERATIONS:
            raise ValueError("unsupported audio operation")
        if side is not None and not isinstance(side, Side):
            raise TypeError("side must be Side or None")
        if not isinstance(issue, AudioIssue):
            raise TypeError("issue must be AudioIssue")

        self.operation = operation
        self.side = side
        self.issue = issue
        # Internal port -> engine state repair; no diagnostic schema change.
        self.invalidated_sides: frozenset[Side] = frozenset()
        safe_side = "none" if side is None else side.value
        super().__init__(
            "audio playback failed: "
            f"operation={operation}; side={safe_side}; issue={issue.value}"
        )


class AudioScheduler:
    def __init__(self, side: Side) -> None:
        if not isinstance(side, Side):
            raise TypeError("side must be Side")
        self.side = side

    def on_candidate(
        self,
        candidate: SuperCandidateEvent,
        config: SideAudioConfig,
    ) -> tuple[AudioAction, ...]:
        if not isinstance(candidate, SuperCandidateEvent):
            raise TypeError("candidate must be SuperCandidateEvent")
        if not isinstance(config, SideAudioConfig):
            raise TypeError("config must be SideAudioConfig")
        if candidate.side is not self.side:
            raise ValueError("candidate side must match scheduler side")
        if not config.enabled:
            return ()
        # Only preview an eligible three-bar drop in restricted mode. Partial
        # drops can still be confirmed as SA3 later, without a lower-tier blip.
        if (
            config.super_scope is SuperScope.SA3_CA_ONLY
            and (candidate.meter_before, candidate.meter_after) != (3, 0)
        ):
            return ()
        return self._main_play(config, candidate.side)

    def on_confirmation(
        self,
        event: ConfirmedSuperEvent,
        config: SideAudioConfig,
        playback_started: bool,
    ) -> tuple[AudioAction, ...]:
        if not isinstance(event, ConfirmedSuperEvent):
            raise TypeError("event must be ConfirmedSuperEvent")
        if not isinstance(config, SideAudioConfig):
            raise TypeError("config must be SideAudioConfig")
        if event.side is not self.side:
            raise ValueError("event side must match scheduler side")
        if type(playback_started) is not bool:
            raise TypeError("playback_started must be a boolean")
        if not config.enabled:
            return ()
        if (
            config.super_scope is SuperScope.SA3_CA_ONLY
            and event.levels_spent in {1, 2}
        ):
            return (
                AudioAction(
                    AudioKind.FADE,
                    event.side,
                    fade_ms=config.whiff_fade_ms,
                ),
            ) if playback_started else ()
        actions = () if playback_started else self._main_play(config, event.side)
        if (
            config.super_scope is SuperScope.ALL_SUPERS
            and event.levels_spent in {1, 2}
            and (playback_started or any(action.kind is AudioKind.PLAY for action in actions))
        ):
            actions += (AudioAction(AudioKind.FADE, event.side, fade_ms=config.whiff_fade_ms),)
        return actions

    def on_cancellation(
        self,
        cancellation: CancelledSuperCandidate,
        config: SideAudioConfig,
    ) -> tuple[AudioAction, ...]:
        if not isinstance(cancellation, CancelledSuperCandidate):
            raise TypeError("cancellation must be CancelledSuperCandidate")
        if not isinstance(config, SideAudioConfig):
            raise TypeError("config must be SideAudioConfig")
        if cancellation.side is not self.side:
            raise ValueError("cancellation side must match scheduler side")
        if not config.enabled:
            return ()
        return (
            AudioAction(
                AudioKind.FADE,
                cancellation.side,
                fade_ms=config.whiff_fade_ms,
            ),
        )

    @staticmethod
    def _main_play(
        config: SideAudioConfig,
        side: Side,
    ) -> tuple[AudioAction, ...]:
        if config.main_track == "":
            raise ValueError("enabled audio config requires a main track")
        return (
            AudioAction(AudioKind.STOP, side),
            AudioAction(
                AudioKind.PLAY,
                side,
                config.main_track,
                config.volume,
            ),
        )

    def on_outcome(
        self,
        decision: OutcomeDecision,
        config: SideAudioConfig,
    ) -> tuple[AudioAction, ...]:
        if not isinstance(decision, OutcomeDecision):
            raise TypeError("decision must be OutcomeDecision")
        if not isinstance(config, SideAudioConfig):
            raise TypeError("config must be SideAudioConfig")
        if decision.side is not self.side:
            raise ValueError("decision side must match scheduler side")
        if (
            decision.label is OutcomeLabel.BLOCK
            and "training_result_changed" in decision.evidence
        ):
            # Legacy default extractor measures scene luma, not a calibrated
            # block indicator. R23 buff footage proves it can false-positive.
            return ()
        if not config.enabled or decision.label is OutcomeLabel.HIT:
            return ()
        if decision.label is OutcomeLabel.UNKNOWN and config.super_scope is SuperScope.ALL_SUPERS:
            return ()
        if decision.label in {
            OutcomeLabel.UNKNOWN,
            OutcomeLabel.FILTERED_TIER,
            OutcomeLabel.RESET,
        }:
            return (
                AudioAction(
                    AudioKind.FADE,
                    decision.side,
                    fade_ms=config.whiff_fade_ms,
                ),
            )
        if decision.label not in {
            OutcomeLabel.WHIFF,
            OutcomeLabel.BLOCK,
            OutcomeLabel.NON_HIT,
        }:
            raise AssertionError("unhandled outcome label")
        if not config.play_whiff_sound:
            return (
                AudioAction(
                    AudioKind.FADE,
                    decision.side,
                    fade_ms=config.whiff_fade_ms,
                ),
            )
        if config.whiff_track == "":
            raise ValueError("whiff sound mode requires a whiff track")
        return (
            AudioAction(AudioKind.STOP, decision.side),
            AudioAction(
                AudioKind.PLAY,
                decision.side,
                config.whiff_track,
                config.whiff_volume,
            ),
        )


class AudioPort(Protocol):
    def execute(self, actions: tuple[AudioAction, ...]) -> None: ...

    def preview(self, track: str, volume: Real) -> None: ...

    def close(self) -> None: ...


_PROCESS_AUDIO_LOCK = threading.RLock()
_active_audio_port: object | None = None


def _resolve_audio_track(user_data_root: Path, track: str) -> str:
    windows_text = track.replace("/", "\\")
    if windows_text.startswith(("\\\\?\\", "\\\\.\\")):
        raise ValueError("device paths are not permitted")

    windows_path = PureWindowsPath(windows_text)
    colon_count = windows_text.count(":")
    if colon_count and not (
        colon_count == 1
        and len(windows_text) >= 2
        and windows_text[1] == ":"
    ):
        raise ValueError("alternate data streams are not permitted")

    if windows_path.drive:
        if not windows_path.root:
            raise ValueError("drive-relative paths are not permitted")
        return track
    if windows_path.root:
        raise ValueError("root-relative paths are not permitted")

    normalized = ntpath.normpath(windows_text)
    if normalized == ".." or normalized.startswith("..\\"):
        raise ValueError("track escapes the user data root")

    candidate = (user_data_root / Path(normalized)).resolve(strict=False)
    try:
        candidate.relative_to(user_data_root)
    except ValueError:
        raise ValueError("track escapes the user data root") from None
    return str(candidate)


def _normalize_preview_volume(volume: Real) -> float:
    if isinstance(volume, bool) or not isinstance(volume, Real):
        raise TypeError("preview volume must be a real number")
    if isinstance(volume, Rational):
        if volume <= 0:
            return 0.0
        if volume >= MAX_VOLUME:
            return MAX_VOLUME
        return float(volume)
    if not math.isfinite(volume):
        raise ValueError("preview volume must be finite")
    if volume <= 0:
        return 0.0
    if volume >= MAX_VOLUME:
        return MAX_VOLUME
    return float(volume)


_BOOST_CACHE_SIZE = 6


def boost_samples(samples: object, gain: float) -> object:
    """Amplify PCM samples by ``gain`` (> 1) without harsh digital clipping.

    Below 70 % of full scale the gain is exact; above it a tanh knee bends the
    signal smoothly towards a 98 % ceiling instead of hard-clipping it.
    """

    import numpy as np

    data = np.asarray(samples)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        full = float(max(abs(info.min), info.max))
    else:
        info = None
        full = 1.0
    scaled = data.astype(np.float32) * np.float32(gain / full)
    knee = np.float32(0.7)
    ceiling = np.float32(0.98)
    magnitude = np.abs(scaled)
    over = magnitude > knee
    if np.any(over):
        headroom = ceiling - knee
        bent = knee + headroom * np.tanh((magnitude[over] - knee) / headroom)
        scaled[over] = np.sign(scaled[over]) * bent
    scaled *= np.float32(full)
    if info is not None:
        scaled = np.clip(np.rint(scaled), info.min, info.max)
    return scaled.astype(data.dtype)


def _load_pygame() -> object:
    return importlib.import_module("pygame")


class PygameAudioPort:
    def __init__(
        self,
        user_data_root: str | os.PathLike[str],
        *,
        pygame_loader: Callable[[], object] = _load_pygame,
    ) -> None:
        try:
            root_text = os.fspath(user_data_root)
        except TypeError:
            raise TypeError("user_data_root must be a string or path-like value") from None
        if not isinstance(root_text, str):
            raise TypeError("user_data_root must resolve to a string")
        if not callable(pygame_loader):
            raise TypeError("pygame_loader must be callable")

        self._user_data_root = Path(root_text).resolve(strict=False)
        self._pygame_loader = pygame_loader
        self._pygame: object | None = None
        self._mixer: object | None = None
        self._channels: tuple[object, object, object] | None = None
        self._sound_refs: dict[Side | None, object | None] = {
            Side.P1: None,
            Side.P2: None,
            None: None,
        }
        self._close_pending: set[Side | None] | None = None
        # (resolved track, gain) -> amplified Sound for volumes above 100 %.
        self._boosted: dict[tuple[str, float], object] = {}

    def prepare(self, tracks: tuple[tuple[str, float], ...]) -> None:
        """Pre-build amplified copies so a >100 % super starts without delay."""

        with _PROCESS_AUDIO_LOCK:
            for track, volume in tracks:
                if not track or volume <= 1.0:
                    continue
                try:
                    resolved = _resolve_audio_track(self._user_data_root, track)
                    self._ensure_backend(None)
                    self._boosted_sound(resolved, volume, None)
                except Exception:
                    continue  # playback will report the problem if it recurs

    def _boosted_sound(self, resolved_track: str, gain: float, side: Side | None) -> object:
        key = (resolved_track, round(float(gain), 3))
        cached = self._boosted.pop(key, None)
        if cached is None:
            base = self._load_sound(resolved_track, side)
            try:
                sndarray = importlib.import_module("pygame.sndarray")
                cached = sndarray.make_sound(boost_samples(sndarray.array(base), key[1]))
            except Exception:
                raise AudioPlaybackError("load", side, AudioIssue.BACKEND_FAILURE) from None
        self._boosted[key] = cached
        while len(self._boosted) > _BOOST_CACHE_SIZE:
            self._boosted.pop(next(iter(self._boosted)))
        return cached

    def _sound_for(self, resolved_track: str, volume: float, side: Side | None) -> object:
        if volume > 1.0:
            sound = self._boosted_sound(resolved_track, volume, side)
            self._set_sound_volume(sound, 1.0, side)
            return sound
        sound = self._load_sound(resolved_track, side)
        self._set_sound_volume(sound, volume, side)
        return sound

    def execute(self, actions: tuple[AudioAction, ...]) -> None:
        with _PROCESS_AUDIO_LOCK:
            side = self._validate_actions(actions)
            self._require_close_complete("execute", side)
            prepared = self._preflight_actions(actions, side)
            self._ensure_backend(side)
            # Decode and set volume before touching any existing playback.
            sounds = []
            for action, resolved_track in prepared:
                sound = None
                if action.kind is AudioKind.PLAY:
                    sound = self._sound_for(resolved_track, action.volume, action.side)
                sounds.append(sound)
            invalidated: set[Side] = set()
            try:
                for (action, _), sound in zip(prepared, sounds, strict=True):
                    if action.kind is AudioKind.STOP:
                        invalidated.add(action.side)
                        self._stop_side(action.side)
                    elif action.kind is AudioKind.PLAY:
                        opposite = Side.P2 if action.side is Side.P1 else Side.P1
                        if self._sound_refs[opposite] is not None:
                            invalidated.add(opposite)
                            self._stop_side(opposite)
                        invalidated.add(action.side)
                        self._play_side(action, sound)
                    else:
                        self._fade_side(action)
            except AudioPlaybackError as error:
                error.invalidated_sides = frozenset(invalidated)
                raise

    def preview(self, track: str, volume: Real) -> None:
        with _PROCESS_AUDIO_LOCK:
            if not isinstance(track, str):
                raise TypeError("preview track must be a string")
            if track == "":
                raise ValueError("preview track must not be empty")
            normalized_volume = _normalize_preview_volume(volume)
            self._require_close_complete("preview", None)
            resolved_track = self._resolve_track(track, None)
            self._ensure_backend(None)
            self._stop_preview()
            sound = self._sound_for(resolved_track, normalized_volume, None)
            self._play_sound(self._channels[2], sound, None)
            self._sound_refs[None] = sound

    def close(self) -> None:
        global _active_audio_port

        with _PROCESS_AUDIO_LOCK:
            if self._channels is None:
                return

            check_failed = False
            initialized = False
            try:
                initialized = bool(self._mixer.get_init())
            except Exception:
                check_failed = True
            if check_failed:
                raise AudioPlaybackError(
                    "close_stop",
                    None,
                    AudioIssue.BACKEND_FAILURE,
                ) from None

            if not initialized:
                self._clear_backend()
                if _active_audio_port is self:
                    _active_audio_port = None
                return

            if self._close_pending is None:
                self._close_pending = {Side.P1, Side.P2, None}

            first_failed_side: Side | None = None
            failed = False
            for side, channel in zip(
                (Side.P1, Side.P2, None),
                self._channels,
                strict=True,
            ):
                if side not in self._close_pending:
                    continue
                stop_failed = False
                try:
                    channel.stop()
                except Exception:
                    stop_failed = True
                if stop_failed and not failed:
                    first_failed_side = side
                    failed = True
                if not stop_failed:
                    self._sound_refs[side] = None
                    self._close_pending.remove(side)
            if failed:
                raise AudioPlaybackError(
                    "close_stop",
                    first_failed_side,
                    AudioIssue.BACKEND_FAILURE,
                ) from None

            self._clear_backend()
            if _active_audio_port is self:
                _active_audio_port = None

    def _require_close_complete(
        self,
        operation: Literal["execute", "preview"],
        side: Side | None,
    ) -> None:
        if self._close_pending is not None:
            raise AudioPlaybackError(
                operation,
                side,
                AudioIssue.CLOSE_INCOMPLETE,
            ) from None

    @staticmethod
    def _validate_actions(actions: object) -> Side:
        if type(actions) is not tuple:
            raise TypeError("actions must be a tuple")
        if not actions:
            raise ValueError("actions must not be empty")
        for action in actions:
            if not isinstance(action, AudioAction):
                raise TypeError("actions must contain only AudioAction values")
            AudioAction(
                action.kind,
                action.side,
                action.track,
                action.volume,
                action.fade_ms,
            )
        side = actions[0].side
        if any(action.side is not side for action in actions[1:]):
            raise ValueError("all actions in a batch must use one side")
        return side

    def _preflight_actions(
        self,
        actions: tuple[AudioAction, ...],
        side: Side,
    ) -> tuple[tuple[AudioAction, str | None], ...]:
        prepared = []
        for action in actions:
            resolved_track = None
            if action.kind is AudioKind.PLAY:
                resolved_track = self._resolve_track(action.track, side)
            prepared.append((action, resolved_track))
        return tuple(prepared)

    def _resolve_track(self, track: str, side: Side | None) -> str:
        failed = False
        resolved_track = ""
        try:
            resolved_track = _resolve_audio_track(self._user_data_root, track)
        except Exception:
            failed = True
        if failed:
            raise AudioPlaybackError(
                "load",
                side,
                AudioIssue.TRACK_REJECTED,
            ) from None
        return resolved_track

    def _ensure_backend(self, side: Side | None) -> None:
        global _active_audio_port

        if self._channels is not None:
            return
        if _active_audio_port is not None and _active_audio_port is not self:
            raise AudioPlaybackError(
                "initialize",
                side,
                AudioIssue.ACTIVE_PORT_CONFLICT,
            ) from None

        provisional = _active_audio_port is None
        if provisional:
            _active_audio_port = self

        failed = False
        pygame_module = None
        mixer = None
        channels = None
        try:
            pygame_module = self._pygame_loader()
            mixer = pygame_module.mixer
            if not mixer.get_init():
                mixer.init()
            channel_count = mixer.get_num_channels()
            if channel_count < 3:
                mixer.set_num_channels(3)
                channel_count = mixer.get_num_channels()
                if channel_count < 3:
                    raise RuntimeError("audio backend did not provide three channels")
            channels = (
                mixer.Channel(0),
                mixer.Channel(1),
                mixer.Channel(2),
            )
        except Exception:
            failed = True

        if failed:
            self._clear_backend()
            if provisional and _active_audio_port is self:
                _active_audio_port = None
            raise AudioPlaybackError(
                "initialize",
                side,
                AudioIssue.BACKEND_FAILURE,
            ) from None

        self._pygame = pygame_module
        self._mixer = mixer
        self._channels = channels

    def _stop_side(self, side: Side) -> None:
        if self._channels is None:
            raise AssertionError("audio backend must be initialized")
        channel = self._channels[0 if side is Side.P1 else 1]
        failed = False
        try:
            channel.stop()
        except Exception:
            failed = True
        if failed:
            raise AudioPlaybackError(
                "stop",
                side,
                AudioIssue.BACKEND_FAILURE,
            ) from None
        self._sound_refs[side] = None

    def _play_side(self, action: AudioAction, sound: object) -> None:
        if self._channels is None or self._mixer is None:
            raise AssertionError("audio backend must be initialized")
        channel = self._channels[0 if action.side is Side.P1 else 1]
        # Retain the reference if partial PLAY and cleanup both fail, so the
        # next opposite PLAY must stop this potentially audible channel first.
        self._sound_refs[action.side] = sound
        try:
            self._play_sound(channel, sound, action.side)
        except AudioPlaybackError:
            try:
                self._stop_side(action.side)
            except AudioPlaybackError:
                pass
            raise

    def _fade_side(self, action: AudioAction) -> None:
        if self._channels is None or action.fade_ms is None:
            raise AssertionError("audio backend and FADE duration are required")
        channel = self._channels[0 if action.side is Side.P1 else 1]
        failed = False
        try:
            channel.fadeout(action.fade_ms)
        except Exception:
            failed = True
        if failed:
            raise AudioPlaybackError(
                "fade",
                action.side,
                AudioIssue.BACKEND_FAILURE,
            ) from None

    def _stop_preview(self) -> None:
        if self._channels is None:
            raise AssertionError("audio backend must be initialized")
        failed = False
        try:
            self._channels[2].stop()
        except Exception:
            failed = True
        if failed:
            raise AudioPlaybackError(
                "preview_stop",
                None,
                AudioIssue.BACKEND_FAILURE,
            ) from None
        self._sound_refs[None] = None

    def _load_sound(self, resolved_track: str, side: Side | None) -> object:
        if self._mixer is None:
            raise AssertionError("audio backend must be initialized")
        failed = False
        sound = None
        try:
            sound = self._mixer.Sound(resolved_track)
        except Exception:
            failed = True
        if failed:
            raise AudioPlaybackError(
                "load",
                side,
                AudioIssue.BACKEND_FAILURE,
            ) from None
        return sound

    @staticmethod
    def _set_sound_volume(
        sound: object,
        volume: float,
        side: Side | None,
    ) -> None:
        failed = False
        try:
            sound.set_volume(volume)
        except Exception:
            failed = True
        if failed:
            raise AudioPlaybackError(
                "set_volume",
                side,
                AudioIssue.BACKEND_FAILURE,
            ) from None

    @staticmethod
    def _play_sound(
        channel: object,
        sound: object,
        side: Side | None,
    ) -> None:
        failed = False
        try:
            channel.play(sound)
        except Exception:
            failed = True
        if failed:
            raise AudioPlaybackError(
                "play",
                side,
                AudioIssue.BACKEND_FAILURE,
            ) from None

    def _clear_backend(self) -> None:
        self._boosted.clear()
        self._pygame = None
        self._mixer = None
        self._channels = None
        self._close_pending = None
        self._sound_refs[Side.P1] = None
        self._sound_refs[Side.P2] = None
        self._sound_refs[None] = None
