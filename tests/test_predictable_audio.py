import numpy as np
from scripts.extract_predictable_audio import prosody


def test_pitch_clock_and_energy():
    sr = 16000
    t = np.arange(sr) / sr
    wave = .2 * np.sin(2 * np.pi * 200 * t)
    p = prosody(wave, np.array([.1, .3, .7]), np.ones(3, bool))
    np.testing.assert_allclose(np.exp(p[:, 0]), 200, rtol=.01)
    np.testing.assert_allclose(np.exp(p[:, 1]), .2 / np.sqrt(2), rtol=.01)
    assert (p[:, 3] == 1).all()
    np.testing.assert_allclose(p[0], p[2], atol=1e-5)


def test_silence_and_invalid_are_not_voiced():
    p = prosody(np.zeros(16000), np.array([0., .5, 1.]), np.array([True, False, True]))
    assert np.isfinite(p).all()
    assert (p[:, 0] == 0).all() and (p[:, 3] == 0).all()
    assert (p[1] == 0).all()
