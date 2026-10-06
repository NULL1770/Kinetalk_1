import copy
import json

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_full_staged import configure_flow_coordinates, flow_vector_mse, parser
from tests.test_full_staged_runner import config
from tests.test_flow_source_noise import inputs


def arguments():
    valid, content, base, identity, affect, noise = inputs()
    return valid, base['h0'], identity, affect, noise


def render(model, x, time, valid, content, identity, affect):
    return model.renderer(x, time, content, affect['global'], affect['intensity_value'],
        identity['code'], valid, temporal_condition=affect['u_a'], condition_dropout=False)


def test_default_disable_is_bit_exact_and_old_state_keys_unchanged():
    torch.manual_seed(45)
    model = NeutralAffectSystem(config()).eval()
    valid, content, identity, affect, x = arguments()
    time = torch.tensor([.2, .8])
    before = render(model, x, time, valid, content, identity, affect)
    keys = set(model.state_dict())
    model.renderer.set_channel_coordinates(torch.ones(52), torch.full((52,), .01))
    model.renderer.set_channel_coordinates()
    assert torch.equal(before, render(model, x, time, valid, content, identity, affect))
    assert keys == set(model.state_dict())
    assert not any('coordinate_' in k for k in keys)
    assert parser().parse_args(['--output', 'unused']).flow_coordinate_system == 'scalar'


def test_moving_center_derivative_and_scale_match_analytic_vector_field():
    torch.manual_seed(46)
    model = NeutralAffectSystem(config()).eval()
    original = copy.deepcopy(model)
    valid, content, identity, affect, y = arguments()
    time = torch.tensor([.15, .85])
    mean = torch.linspace(-.4, .4, 52)
    std = torch.linspace(.1, 1.2, 52)
    model.renderer.set_channel_coordinates(mean, std)
    x = time[:, None, None] * mean + std * y
    want = mean + std * render(original, y, time, valid, content, identity, affect)
    want = torch.where(valid[..., None], want, 0.)
    torch.testing.assert_close(render(model, x, time, valid, content, identity, affect), want,
        rtol=2e-5, atol=2e-6)


def test_shared_head_reset_consumes_no_rng_and_preserves_other_parameters(tmp_path):
    model = NeutralAffectSystem(config()).eval()
    before = copy.deepcopy(model.state_dict())
    support = torch.ones(52, dtype=torch.bool)
    support[-1] = False
    mean = torch.linspace(-.1, .1, 52)
    mean[-1] = 0
    p = tmp_path / 'stats.json'
    p.write_text(json.dumps({'residual_mean': mean.tolist()}))
    rng = torch.get_rng_state()
    cfg = config()
    info = configure_flow_coordinates(model, cfg, p, torch.full((52,), .01), support, 'train-standardized')
    assert torch.equal(rng, torch.get_rng_state())
    assert info['mode'] == 'train-standardized'
    after = model.state_dict()
    for k in before:
        if k in ('renderer.output.weight', 'renderer.output.bias'):
            assert not torch.count_nonzero(after[k])
        else:
            assert torch.equal(before[k], after[k]), k
    restored = NeutralAffectSystem(cfg).eval()
    restored.load_state_dict(after, strict=True)
    assert restored.renderer.channel_coordinates
    assert torch.equal(model.renderer.coordinate_mean, restored.renderer.coordinate_mean)
    assert torch.equal(model.renderer.coordinate_std, restored.renderer.coordinate_std)


def test_zero_head_initial_mean_field_and_physical_rollout_are_shared():
    torch.manual_seed(47)
    model = NeutralAffectSystem(config()).eval()
    valid, content, base, identity, affect, white = inputs()
    mean = torch.linspace(-.2, .2, 52)
    std = torch.linspace(.01, 1.2, 52)
    model.set_flow_source_std(std)
    model.renderer.output.weight.data.zero_()
    model.renderer.output.bias.data.zero_()
    outputs = []
    for units in (torch.ones(52), std):
        model.renderer.set_channel_coordinates(mean, units)
        out = model.generate(content, valid, identity, affect, initial_noise=white, steps=12, base=base)
        want = torch.where(valid[..., None], (white * std + mean) * model.residual_scale, 0.)
        torch.testing.assert_close(out['raw_residual'], want, rtol=2e-6, atol=2e-7)
        outputs.append(out['motion'])
    assert torch.equal(*outputs)


def test_head_gradients_do_not_starve_tiny_channels_in_consistent_units():
    torch.manual_seed(48)
    model = NeutralAffectSystem(config()).eval()
    model.renderer.output.weight.data.zero_()
    model.renderer.output.bias.data.zero_()
    valid, content, identity, affect, x = arguments()
    x.zero_()
    time = torch.tensor([.2, .8])
    grads = []
    for std in (torch.ones(52), torch.logspace(-4, 0, 52)):
        model.renderer.set_channel_coordinates(torch.zeros(52), std)
        pred = render(model, x, time, valid, content, identity, affect)
        target = std.view(1, 1, -1).expand_as(pred)
        loss = flow_vector_mse(pred, target, valid[..., None].expand_as(pred), std)
        grads.append(torch.autograd.grad(loss, model.renderer.output.weight)[0])
    torch.testing.assert_close(*grads, rtol=1e-6, atol=1e-7)
    assert float(grads[1][0].norm()) > 0


@pytest.mark.parametrize('mean,std', [(None, torch.ones(52)), (torch.ones(51), torch.ones(52)),
    (torch.zeros(52), torch.zeros(52)), (torch.full((52,), float('nan')), torch.ones(52))])
def test_invalid_coordinates_are_rejected(mean, std):
    with pytest.raises(ValueError):
        NeutralAffectSystem(config()).renderer.set_channel_coordinates(mean, std)
