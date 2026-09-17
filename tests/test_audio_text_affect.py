import pytest
import torch

from kinetalk_b0.models.audio_text_affect import AudioTextAffect


def fixture():
    torch.manual_seed(18)
    model = AudioTextAffect(torch.zeros(9), torch.ones(9), text_dim=10, global_dim=7, hidden=16, local_dim=8)
    x = torch.randn(2, 9, 9)
    valid = torch.tensor([[True] * 7 + [False] * 2, [True] * 9])
    g, tokens = torch.randn(2, 7), torch.randn(2, 5, 10)
    tv = torch.tensor([[True, True, True, False, False], [False] * 5])
    return model, x, valid, g, tokens, tv


def activate(model):
    with torch.no_grad():
        model.fusion[-1].weight.normal_(std=.1)
        model.intensity_head[-1].weight.normal_(std=.1)


def test_zero_start_original_global_and_empty_text_are_safe():
    model, x, v, g, tok, tv = fixture()
    out = model(x, v, g, tok, tv)
    assert torch.equal(out['global'], g) and out['local'].count_nonzero() == 0
    assert torch.isfinite(out['predicted_intensity']).all()
    activate(model)
    text, no_text = model(x, v, g, tok, tv), model(x, v, g, tok, tv, use_text=False)
    torch.testing.assert_close(text['local'][1], no_text['local'][1])
    assert not torch.allclose(text['local'][0], no_text['local'][0])


def test_mask_poison_and_padding_do_not_change_observed_output():
    model, x, v, g, tok, tv = fixture(); activate(model)
    out = model(x, v, g, tok, tv)
    x[~v] = float('nan'); tok[~tv] = float('nan')
    xp = torch.cat((x, torch.full((2, 3, 9), float('nan'))), 1)
    vp = torch.cat((v, torch.zeros(2, 3, dtype=torch.bool)), 1)
    tp = torch.cat((tok, torch.full((2, 2, 10), float('nan'))), 1)
    tvp = torch.cat((tv, torch.zeros(2, 2, dtype=torch.bool)), 1)
    again = model(xp, vp, g, tp, tvp)
    for key in ('local', 'predicted_intensity'):
        torch.testing.assert_close(out[key][v], again[key][:, :9][v], rtol=1e-5, atol=1e-6)


def test_no_text_branch_is_independent_of_all_token_values_and_has_no_text_gradient():
    model, x, v, g, tok, tv = fixture(); activate(model)
    first = model(x, v, g, tok, tv, use_text=False)
    second = model(x, v, g, torch.full_like(tok, float('nan')), tv, use_text=False)
    torch.testing.assert_close(first['local'], second['local'], rtol=0, atol=0)
    first['local'].square().sum().backward()
    assert all(p.grad is None for name, p in model.named_parameters() if name.startswith(('cross_attention', 'text_')))


def test_intensity_control_is_noncentered_and_text_gradient_reaches_fusion():
    model, x, v, g, tok, tv = fixture(); activate(model)
    low = model(x, v, g, tok, tv, intensity_override=torch.ones(2, 9, 1))
    high = model(x, v, g, tok, tv, intensity_override=torch.full((2, 9, 1), 3.))
    assert not torch.allclose(low['local'], high['local'])
    loss = high['local'].square().sum() + high['predicted_intensity'][v].square().mean()
    loss.backward()
    assert model.cross_attention.in_proj_weight.grad.abs().sum() > 0
    assert model.intensity_head[-1].weight.grad.abs().sum() > 0
    assert high['local'][v].mean().abs() > 1e-6


def test_nonfinite_observed_text_rejected():
    model, x, v, g, tok, tv = fixture()
    tok[0, 0] = float('nan')
    with pytest.raises(ValueError, match='text'):
        model(x, v, g, tok, tv)
