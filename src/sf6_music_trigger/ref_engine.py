"""Audio policy for super events reported by the in-game REFramework detector.

Keeps the T03 playback rules, now driven by exact game events instead of HUD
pixels:

* ALL_SUPERS: SA1/SA2 play, then fade after the 4-second hold whatever the
  result. SA3/CA play; a block or whiff runs the configured whiff action
  (whiff sound, or a fade held until 4 s after playback started); an unknown
  result keeps the music; a hit keeps the music.
* SA3_CA_ONLY: SA1/SA2 stay silent. SA3/CA play; block, whiff and unknown all
  run the whiff action.
* The side that plays last owns the audio; the port stops the other side and
  stale results or fade timers of older episodes are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from .audio import AudioAction, AudioKind, AudioPlaybackError
from .config import SideAudioConfig, SuperScope
from .domain import Side
from .engine import EngineUpdate
from .reframework_events import Connection, RefEventBatch, SuperEnd, SuperResult, SuperStart

PLAYBACK_HOLD_S = 4.0

_RESULT_STATUS = {"hit": "命中", "block": "格挡", "whiff": "打空", "unknown": "结果不确定"}


@dataclass(frozen=True)
class _Owner:
    side: Side
    episode: int


@dataclass(frozen=True)
class _HeldFade:
    due: float
    owner: _Owner
    action: AudioAction


class RefTriggerEngine:
    """Engine for :class:`ListenerController` fed by :class:`RefEventSource`."""

    def __init__(
        self,
        *,
        p1_audio_config: SideAudioConfig,
        p2_audio_config: SideAudioConfig,
        audio_port: object,
        audio_diagnostic: Callable[[AudioPlaybackError], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(p1_audio_config, SideAudioConfig) or not isinstance(
            p2_audio_config, SideAudioConfig
        ):
            raise TypeError("audio configs must be SideAudioConfig values")
        if not callable(getattr(audio_port, "execute", None)):
            raise TypeError("audio_port must provide execute")
        if not callable(audio_diagnostic) or not callable(clock):
            raise TypeError("audio_diagnostic and clock must be callable")
        self._configs = {Side.P1: p1_audio_config, Side.P2: p2_audio_config}
        self._audio_port = audio_port
        self._audio_diagnostic = audio_diagnostic
        self._clock = clock
        self._owner: _Owner | None = None
        self._playback_started: float | None = None
        self._held_fade: _HeldFade | None = None
        # side -> (episode, level, result already applied)
        self._episodes: dict[Side, tuple[int, int, bool] | None] = {Side.P1: None, Side.P2: None}
        self._status: dict[Side, str | None] = {Side.P1: None, Side.P2: None}
        self.connection: Connection = Connection.NO_FILE
        self.chara: tuple[int | None, int | None] = (None, None)
        self.rejected_lines = 0
        self.event_log: list[SuperStart | SuperResult | SuperEnd] = []

    def update_configs(self, p1_audio_config: SideAudioConfig, p2_audio_config: SideAudioConfig) -> None:
        """Swap in new audio settings while listening (in-game or panel edits).

        Music that is already playing keeps going; the next super uses the new
        settings. Called from the UI thread: a single dict assignment.
        """

        if not isinstance(p1_audio_config, SideAudioConfig) or not isinstance(
            p2_audio_config, SideAudioConfig
        ):
            raise TypeError("audio configs must be SideAudioConfig values")
        self._configs = {Side.P1: p1_audio_config, Side.P2: p2_audio_config}
        self._prepare_audio()

    def _prepare_audio(self) -> None:
        """Let the port pre-build amplified (>100 %) tracks before they are needed."""

        prepare = getattr(self._audio_port, "prepare", None)
        if not callable(prepare):
            return
        tracks = []
        for config in self._configs.values():
            if config.enabled:
                tracks.append((config.main_track, config.volume))
                if config.play_whiff_sound:
                    tracks.append((config.whiff_track, config.whiff_volume))
        try:
            prepare(tuple(tracks))
        except Exception:
            pass  # playback reports real problems when a super fires

    # -- ListenerController engine protocol ----------------------------------
    @property
    def outcome_statuses(self) -> tuple[str | None, str | None]:
        return (self._status[Side.P1], self._status[Side.P2])

    @property
    def diagnostic_warning(self) -> bool:
        return False

    def reset(self, reason: str) -> None:
        if type(reason) is not str or not reason.strip():
            raise ValueError("reason must be nonblank text")
        self._flush_held_fade(force=True)
        self._owner = None
        self._playback_started = None
        self._held_fade = None
        self._episodes = {Side.P1: None, Side.P2: None}
        self._status = {Side.P1: None, Side.P2: None}
        if reason == "capture session started":
            self._prepare_audio()

    def process(self, batch: RefEventBatch) -> EngineUpdate:
        if not isinstance(batch, RefEventBatch):
            raise TypeError("batch must be RefEventBatch")
        self.connection = batch.connection
        self.chara = batch.chara
        self.rejected_lines += batch.rejected_lines
        if batch.session_changed:
            # The game restarted: older episodes can never end now.
            self._episodes = {Side.P1: None, Side.P2: None}
        for event in batch.events:
            self.event_log.append(event)
            if isinstance(event, SuperStart):
                self._on_start(event)
            else:
                self._on_result(event)
        self._flush_held_fade()
        return EngineUpdate()

    # -- policy ----------------------------------------------------------------
    def _on_start(self, event: SuperStart) -> None:
        config = self._configs[event.side]
        self._episodes[event.side] = (event.episode, event.level, False)
        if not config.enabled:
            self._status[event.side] = None
            return
        if config.super_scope is SuperScope.SA3_CA_ONLY and event.level in (1, 2):
            self._status[event.side] = "低等级已过滤"
            return
        owner = _Owner(event.side, event.episode)
        played = self._execute(
            (
                AudioAction(AudioKind.STOP, event.side),
                AudioAction(AudioKind.PLAY, event.side, config.main_track, config.volume),
            ),
            new_owner=owner,
        )
        if not played:
            return
        if event.level in (1, 2):
            # ALL_SUPERS low tiers: fixed fade after the playback hold.
            self._hold_fade(owner, AudioAction(AudioKind.FADE, event.side, fade_ms=config.whiff_fade_ms))
        else:
            self._status[event.side] = "等待大招结果"

    def _on_result(self, event: SuperResult | SuperEnd) -> None:
        """Apply the first result of an episode; SuperEnd only closes it.

        Detector v0.2 sends SuperResult as early as possible (e.g. right after
        a super's attack frames whiffed); v0.1 only sends SuperEnd.
        """

        current = self._episodes.get(event.side)
        if current is None or current[0] != event.episode:
            return
        episode, level, applied = current
        if isinstance(event, SuperEnd):
            self._episodes[event.side] = None
            if applied:
                return
        else:
            if applied:
                return
            self._episodes[event.side] = (episode, level, True)
        config = self._configs[event.side]
        if not config.enabled:
            return
        if event.level in (1, 2):
            return  # filtered, or already owns its fixed fade
        self._status[event.side] = _RESULT_STATUS[event.result]
        owner = _Owner(event.side, event.episode)
        if self._owner != owner:
            return  # a newer super owns the audio now
        if event.result == "hit":
            self._held_fade = None
            return
        if event.result == "unknown" and config.super_scope is SuperScope.ALL_SUPERS:
            return
        if config.play_whiff_sound and config.whiff_track:
            self._execute(
                (
                    AudioAction(AudioKind.STOP, event.side),
                    AudioAction(AudioKind.PLAY, event.side, config.whiff_track, config.whiff_volume),
                ),
                new_owner=None,
            )
            return
        self._hold_fade(owner, AudioAction(AudioKind.FADE, event.side, fade_ms=config.whiff_fade_ms))

    def _hold_fade(self, owner: _Owner, action: AudioAction) -> None:
        started = self._playback_started if self._owner == owner else None
        due = (started if started is not None else self._clock()) + PLAYBACK_HOLD_S
        self._held_fade = _HeldFade(due, owner, action)
        self._status[owner.side] = "等待淡出"
        self._flush_held_fade()

    def _flush_held_fade(self, *, force: bool = False) -> None:
        held = self._held_fade
        if held is None:
            return
        if not force and self._clock() < held.due:
            return
        self._held_fade = None
        if self._owner != held.owner:
            return
        succeeded = self._execute((held.action,), new_owner=held.owner)
        self._status[held.owner.side] = "淡出中" if succeeded else "淡出失败"

    def _execute(self, actions: tuple[AudioAction, ...], *, new_owner: _Owner | None) -> bool:
        try:
            self._audio_port.execute(actions)
        except AudioPlaybackError as error:
            if self._owner is not None and self._owner.side in error.invalidated_sides:
                self._owner = None
                self._playback_started = None
                self._held_fade = None
            self._audio_diagnostic(error)
            return False
        if any(action.kind is AudioKind.PLAY for action in actions):
            # A successful PLAY is the single ownership commit (the port has
            # stopped the other side); older timers must not touch it.
            self._owner = new_owner
            self._playback_started = self._clock()
            self._held_fade = None
        elif any(action.kind is AudioKind.STOP for action in actions):
            self._owner = None
            self._playback_started = None
        return True
