"""V2: volumes above 100 % via amplified samples with a soft limiter."""

from fractions import Fraction
import math
import os
import wave

import numpy as np
import pytest

from sf6_music_trigger.audio import (
    AudioAction,
    AudioKind,
    PygameAudioPort,
    _normalize_preview_volume,
    boost_samples,
)
from sf6_music_trigger.config import MAX_VOLUME
from sf6_music_trigger.domain import Side


def test_boost_is_exact_below_the_knee_and_soft_limited_above():
    quiet = np.array([0, 1000, -1000, 5000], dtype=np.int16)
    assert boost_samples(quiet, 2.0).tolist() == [0, 2000, -2000, 10000]

    loud = np.array([20000, -20000, 32767, -32768], dtype=np.int16)
    boosted = boost_samples(loud, 4.0)
    assert boosted.dtype == np.int16
    assert np.all(np.abs(boosted.astype(np.int32)) <= 0.98 * 32768 + 1)
    assert boosted[0] > 20000 and boosted[1] < -20000          # still louder
    assert int(boosted[0]) == -int(boosted[1])                  # symmetric
    assert boosted[2] < 32767                                   # no hard clip at full scale

    floats = boost_samples(np.array([0.1, 0.9], dtype=np.float32), 3.0)
    assert floats[0] == pytest.approx(0.3, abs=1e-6) and 0.9 < floats[1] <= 0.98


@pytest.mark.parametrize(("volume", "expected"), [
    (10**10000, MAX_VOLUME), (1.25, 1.25), (Fraction(5, 2), 2.5), (-1, 0.0), (7.5, MAX_VOLUME)],
    ids=["huge", "float", "fraction", "negative", "above-max"])
def test_preview_volume_clamps_to_max(volume, expected):
    assert _normalize_preview_volume(volume) == expected


def test_play_accepts_up_to_max_volume():
    assert AudioAction(AudioKind.PLAY, Side.P1, "a.wav", MAX_VOLUME).volume == MAX_VOLUME
    with pytest.raises(ValueError):
        AudioAction(AudioKind.PLAY, Side.P1, "a.wav", math.nextafter(MAX_VOLUME, 10))


@pytest.fixture
def real_port(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.mixer.quit()
    pygame.mixer.init(frequency=22050, size=-16, channels=1)
    path = tmp_path / "tone.wav"
    samples = (np.sin(np.linspace(0, 40 * np.pi, 2205)) * 6000).astype(np.int16)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(22050)
        stream.writeframes(samples.tobytes())
    port = PygameAudioPort(tmp_path)
    try:
        yield port, pygame, samples
    finally:
        port.close()
        pygame.mixer.quit()


def test_loud_play_uses_amplified_sound_and_reuses_it(real_port):
    port, pygame, samples = real_port
    port.execute((AudioAction(AudioKind.PLAY, Side.P1, "tone.wav", 2.0),))
    played = port._sound_refs[Side.P1]
    assert played.get_volume() == pytest.approx(1.0, abs=0.01)
    data = pygame.sndarray.array(played).reshape(-1)
    assert np.max(np.abs(data.astype(np.int32))) == pytest.approx(12000, abs=2)

    port.execute((AudioAction(AudioKind.PLAY, Side.P1, "tone.wav", 2.0),))
    assert port._sound_refs[Side.P1] is played                  # cached, not rebuilt

    port.execute((AudioAction(AudioKind.PLAY, Side.P1, "tone.wav", 0.5),))
    normal = port._sound_refs[Side.P1]
    assert normal is not played and normal.get_volume() == pytest.approx(0.5, abs=0.01)


def test_prepare_builds_amplified_sounds_ahead_of_time(real_port):
    port, _pygame, _samples = real_port
    port.prepare((("tone.wav", 3.0), ("tone.wav", 0.8), ("", 2.0), ("missing.wav", 2.0)))
    assert list(port._boosted) == [(str(port._user_data_root / "tone.wav"), 3.0)]
    built = next(iter(port._boosted.values()))
    port.execute((AudioAction(AudioKind.PLAY, Side.P2, "tone.wav", 3.0),))
    assert port._sound_refs[Side.P2] is built
