import torch
from torch.nn import functional as F
from kinetalk_b0.models.relative_audio_timing import RelativeAudioTiming


def example():
    torch.manual_seed(42); torch.set_num_threads(1)
    m = RelativeAudioTiming(torch.zeros(1540), torch.ones(1540),
        torch.randn(1536, 8) / 40, torch.ones(52), torch.ones(4), hidden=12).eval()
    torch.nn.init.normal_(m.head.weight, std=.02)
    x = torch.randn(2, 39, 1540); v = torch.ones(2, 39, dtype=torch.bool)
    v[0, :2] = False; v[1, 31:] = False
    x[..., -1] = 1.
    return m, x, v


def test_offset_padding_and_missing_payload_invariance():
    m, x, v = example(); y = m(x, v)
    dirty = x.clone(); dirty[~v] = float('nan')
    offset = torch.randn(1, 1, 1540); offset[..., -1] = 0.
    moved = m(x + offset, v)
    padded = m(F.pad(dirty, (0, 0, 0, 13), value=float('nan')), F.pad(v, (0, 13)))
    torch.testing.assert_close(moved['state'], y['state'], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(padded['state'][:, :39], y['state'], atol=1e-7, rtol=0)


def test_zero_mean_static_reverse_and_nonupper_support():
    m, x, v = example(); y = m(x, v)
    torch.testing.assert_close(y['state'].sum(1), torch.zeros(2, 4), atol=2e-6, rtol=0)
    assert m(x, v, 'static')['state'].count_nonzero() == 0
    reverse = m(x, v, 'reverse')['state']
    for i in range(2):
        ix = v[i].nonzero().flatten()
        torch.testing.assert_close(reverse[i, ix], y['state'][i, ix.flip(0)], atol=0, rtol=0)
    upper = {41,42,43,44,45,5,6,12,13}
    assert y['delta'][..., [i for i in range(52) if i not in upper]].count_nonzero() == 0


def test_state_can_fit_temporal_signal_without_identity_input():
    m, x, v = example(); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=.01)
    target = torch.sin(torch.arange(39).float()/12)[None, :, None].expand(2, -1, 4).clone()
    target = torch.where(v[..., None], target, 0.)
    target = torch.where(v[..., None], target-target.sum(1, keepdim=True)/v.sum(1)[:, None, None], 0.)
    with torch.no_grad(): initial = float((m(x,v)['state'][v]-target[v]).square().mean())
    for _ in range(40):
        loss = (m(x,v)['state'][v]-target[v]).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    m.eval()
    assert float((m(x,v)['state'][v]-target[v]).square().mean()) < initial*.7
