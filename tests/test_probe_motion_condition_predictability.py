import numpy as np
import torch

from scripts.probe_motion_condition_predictability import (
    audio_windows, locations, motion_descriptor, native_runs,
    projected_native_audio, weighted_moments,
)


def clip(length=50):
    return {
        'valid': torch.ones(length, dtype=torch.bool),
        'features': torch.arange(length*1540, dtype=torch.float32).reshape(length, 1540) / 1000,
        'context': torch.zeros(5),
        'motion9': torch.zeros(length, 9),
        'motion_mask': torch.ones(length, 9, dtype=torch.bool),
    }


def test_native_windows_are_fixed_nonoverlap_and_do_not_bridge_gaps():
    mask = torch.tensor([1]*50 + [0]*3 + [1]*50, dtype=torch.bool)
    assert native_runs(mask) == [(0, 50), (53, 103)]
    assert locations(mask, 25) == [(0, 25, 0, 50), (25, 50, 0, 50), (53, 78, 53, 103), (78, 103, 53, 103)]


def test_audio_interventions_never_read_motion_and_keep_clock():
    c = clip()
    c['motion9'].fill_(999.)
    stats = {'audio_mean': np.zeros(1540), 'audio_scale': np.ones(1540),
             'projection': np.zeros((1536, 64))}
    projected = projected_native_audio(c, stats)
    real, loc = audio_windows(c, projected, 25, 'real')
    static, loc2 = audio_windows(c, projected, 25, 'static')
    assert loc == loc2 and real.shape == static.shape
    # Static retains the native-run mean; only the three local blocks vanish.
    assert np.allclose(static[:, len(c['context']) + projected.shape[1]:], 0.)
    assert np.isfinite(real).all()


def test_motion_descriptor_has_four_speed_and_four_signed_targets():
    motion = np.zeros((25, 9), dtype=np.float64)
    motion[:, 2:5] = np.arange(25)[:, None]
    value = motion_descriptor(motion)
    assert value.shape == (8,)
    assert np.isfinite(value).all()
    assert value[0] > 0 and value[4] > 0


def test_fit_weighted_moments_equal_rows():
    values = np.array([[0., 2.], [2., 4.]])
    mean, scale = weighted_moments(values, np.ones(2), .01)
    assert np.allclose(mean, [1., 3.])
    assert np.all(scale > 0)
