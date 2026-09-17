import torch

from kinetalk_b0.models.label_guided_affect import (
    LabelGuidedAffect, automatic_guidance, explicit_guidance,
)


def inputs():
    torch.manual_seed(53)
    x = torch.randn(2, 8, 11)
    valid = torch.tensor([[True] * 6 + [False] * 2, [True] * 8])
    global_code = torch.randn(2, 7)
    ep, lp = explicit_guidance(torch.tensor([0, 5]), torch.tensor([-1, 3]))
    model = LabelGuidedAffect(torch.zeros(11), torch.ones(11), global_dim=7, hidden=12, local_dim=9)
    return model, x, valid, global_code, ep, lp


def test_zero_start_preserves_global_and_unknown_level_is_separate():
    model, x, valid, g, ep, lp = inputs()
    out = model(x, valid, g, ep, lp)
    assert torch.equal(out['global'], g) and out['local'].count_nonzero() == 0
    assert lp[0, -1] == 1 and lp[0, 0] == 0
    assert (out['predicted_intensity'][0, :6] > 0).all()  # neutral not zero-gated
    auto_e, auto_l = automatic_guidance(torch.randn(2, 8), torch.randn(2, 4))
    assert auto_e.shape == (2, 8) and auto_l.shape == (2, 5) and auto_l[:, -1].count_nonzero() == 0


def test_padding_poison_cannot_change_valid_conditions_and_static_is_constant():
    model, x, valid, g, ep, lp = inputs()
    with torch.no_grad():
        model.fusion[-1].weight.normal_(std=.1)
        model.intensity_head.weight.normal_(std=.1)
    full = model(x, valid, g, ep, lp)
    poison = x.clone(); poison[~valid] = float('nan')
    padded = torch.cat((poison, torch.full((2, 3, 11), float('nan'))), 1)
    mask = torch.cat((valid, torch.zeros(2, 3, dtype=torch.bool)), 1)
    again = model(padded, mask, g, ep, lp)
    for key in ('local', 'predicted_intensity'):
        torch.testing.assert_close(full[key][valid], again[key][:, :8][valid], atol=1e-6, rtol=1e-6)
    static = model(x, valid, g, ep, lp, temporal_mode='static')
    for i in range(2):
        for key in ('local', 'predicted_intensity'):
            observed = static[key][i, valid[i]]
            torch.testing.assert_close(observed, observed[:1].expand_as(observed))


def test_condition_uses_labels_and_noncentered_intensity_without_target_motion():
    model, x, valid, g, ep, lp = inputs()
    with torch.no_grad():
        model.fusion[-1].weight.normal_(std=.1)
    low = model(x, valid, g, ep, lp, intensity_override=torch.ones(2, 8, 1))
    high = model(x, valid, g, ep, lp, intensity_override=torch.full((2, 8, 1), 3.))
    other = model(x, valid, g, ep.flip(0), lp)
    assert not torch.allclose(low['local'], high['local'])
    assert not torch.allclose(low['local'], other['local'])
    assert low['local'][valid].mean().abs() > 1e-5


def test_supervised_intensity_and_generated_condition_receive_gradients():
    model, x, valid, g, ep, lp = inputs()
    out = model(x, valid, g, ep, lp)
    loss = (out['predicted_intensity'][valid] - 2).square().mean() + out['local'][valid].sum()
    loss.backward()
    assert model.intensity_head.weight.grad.abs().sum() > 0
    assert model.fusion[-1].weight.grad.abs().sum() > 0


def test_invalid_probabilities_and_observed_nan_are_rejected():
    model, x, valid, g, ep, lp = inputs()
    import pytest
    with pytest.raises(ValueError, match='probabilities'):
        model(x, valid, g, ep * 2, lp)
    x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        model(x, valid, g, ep, lp)
