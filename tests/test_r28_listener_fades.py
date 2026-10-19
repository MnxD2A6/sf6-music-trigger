"""Real worker + engine + audio port; only capture/mixer and detector inputs fake."""
import threading
import pytest
from sf6_music_trigger.config import SideAudioConfig, SuperScope
from sf6_music_trigger.controller import ListenerController
from sf6_music_trigger.domain import Side, GaugeTrackerUpdate, CaptureSample, CaptureStatus
from test_audio import fake_audio
from test_engine import _build_harness, _TrackerSpy, _event, _frame, _running


@pytest.mark.parametrize('side', list(Side))
@pytest.mark.parametrize('bars', [1, 2])
@pytest.mark.parametrize('fade_failure', [False, True])
def test_listener_survives_real_non_three_hold_and_fade(fake_audio, side, bars, fade_failure):
    config = SideAudioConfig(main_track='main.wav', super_scope=SuperScope.ALL_SUPERS,
                             fade_non_three=True)
    tracker = _TrackerSpy([GaugeTrackerUpdate(_event(side, levels_spent=bars, confirmed_at=1.))])
    h = _build_harness(audio=fake_audio.port, p1_config=config, p2_config=config,
                       **{'p1_tracker' if side is Side.P1 else 'p2_tracker': tracker})
    times = [1., 4.99, 5., 6.]
    for i in range(len(times)):
        for s in Side:
            h.observer.kinds[i, s] = 'level_1'
    reached_end, release = threading.Event(), threading.Event()
    snapshots = []

    class Source:
        index = 0
        def start(self):
            pass
        def stop(self):
            release.set()
        def read(self):
            if self.index:
                snapshots.append(controller.outcome_statuses)
            if self.index == len(times):
                reached_end.set()
                release.wait(3)
                return CaptureSample(CaptureStatus.STOPPED)
            if self.index == 2 and fade_failure:
                channel = fake_audio.mixer.channels[0 if side is Side.P1 else 1]
                def failed_fade(ms):
                    raise RuntimeError('injected device failure')
                channel.fadeout = failed_fade
            at = times[self.index]
            h.clock.value = at
            sample = _running(_frame(self.index, at))
            self.index += 1
            return sample

    controller = ListenerController(source=Source(), engine=h.engine, audio_port=fake_audio.port)
    try:
        controller.start()
        assert reached_end.wait(2), 'worker terminated before the post-fade frame'
        assert controller.is_running
        assert not any(s.kind in {'worker_error', 'stopped'} for s in controller.drain_statuses())
        index = 0 if side is Side.P1 else 1
        assert [row[index] for row in snapshots] == [
            '等待淡出', '等待淡出',
            '淡出失败' if fade_failure else '淡出中',
            '淡出失败' if fade_failure else '淡出中',
        ]
        fades = [r for r in h.diagnostic_records if r.transition == 'FADE']
        assert len(fades) == 1 and fades[0].monotonic_time == 5.
        assert fades[0].evidence == (('fade_failed',) if fade_failure else ('playback_hold_elapsed',))
        if not fade_failure:
            channel = fake_audio.mixer.channels[index]
            assert [c for c in channel.calls if c[0] == 'fadeout'] == [('fadeout', 1500)]
    finally:
        controller.request_stop()
        controller.worker_thread.join(2)
        assert not controller.is_running
