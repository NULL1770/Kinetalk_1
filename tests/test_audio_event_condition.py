import numpy as np
import torch

from scripts.audio_event_condition import EVENT_DIM, PROSODY, derive_event_condition


def make_features(n=64):
    x = torch.zeros(n, 1540, dtype=torch.float32)
    # log-f0, log-rms, periodicity, voiced
    x[:, 1536] = np.log(130.)
    x[:, 1538] = 0.95
    x[:, 1539] = 1.
    return x


def test_event_condition_shape_mask_and_ignores_nonprosody():
    x = make_features()
    valid = torch.ones(len(x), dtype=torch.bool)
    valid[20:24] = False
    x[:, :1536] = torch.randn_like(x[:, :1536])
    out = derive_event_condition(x, valid)
    assert out.shape == (len(x), EVENT_DIM)
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[~valid], torch.zeros_like(out[~valid]))
    x2 = x.clone(); x2[:, :1536] = torch.randn_like(x2[:, :1536]) * 1000
    torch.testing.assert_close(derive_event_condition(x2, valid), out, rtol=0, atol=0)


def test_energy_rise_and_fall_are_directional():
    x = make_features(80)
    # A smooth rise followed by a smooth fall in log-RMS.
    up = torch.linspace(-1., 1., 30)
    down = torch.linspace(1., -1., 30)
    x[:30, 1537] = up
    x[30:60, 1537] = down
    valid = torch.ones(80, dtype=torch.bool)
    out = derive_event_condition(x, valid)
    rise, fall = out[:, 1], out[:, 2]
    assert float(rise[8:25].mean()) > float(fall[8:25].mean()) + .05
    assert float(fall[35:52].mean()) > float(rise[35:52].mean()) + .05
    assert float(rise.max()) <= 1. and float(fall.max()) <= 1.


def test_voicing_onset_offset_and_gap_are_run_local():
    x = make_features(72)
    x[:18, 1539] = 0.; x[:18, 1538] = 0.
    x[18:36, 1539] = 1.; x[18:36, 1538] = .95
    x[36:54, 1539] = 0.; x[36:54, 1538] = 0.
    valid = torch.ones(72, dtype=torch.bool); valid[40:44] = False
    out = derive_event_condition(x, valid)
    assert float(out[16:23, 7].max()) > .05  # onset
    assert float(out[33:40, 8].max()) > .05  # offset
    torch.testing.assert_close(out[~valid], torch.zeros_like(out[~valid]))
    # The first post-gap unvoiced frames have no onset from the missing interval.
    assert float(out[44:50, 7].max()) < .05
