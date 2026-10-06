import copy
import json
import sys

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_full_staged import load_flow_source_stats, main, parser
from scripts.diagnose_flow_sampling import decode_with_sampler
from tests.test_full_staged_runner import config


def inputs():
    torch.set_num_threads(1)
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    content = torch.randn(2, 4, 8)
    base = {'h0': torch.randn(2, 4, 8), 'b0': torch.full((2, 4, 52), .2)}
    ident = {'code': torch.randn(2, 8), 'baseline': torch.full((2, 52), .1)}
    affect = {'global': torch.randn(2, 8), 'intensity_value': torch.ones(2, 1), 'u_a': torch.randn(2, 4, 8)}
    noise = torch.randn(2, 4, 52)
    return valid, content, base, ident, affect, noise


def test_default_and_explicit_unit_source_preserve_outputs_and_checkpoint_keys():
    torch.manual_seed(31)
    model = NeutralAffectSystem(config()).eval()
    assert parser().parse_args(['--output', 'unused']).flow_source_noise == 'standard'
    state = copy.deepcopy(model.state_dict())
    assert 'flow_source_std' not in state
    valid, content, base, ident, affect, noise = inputs()
    original = model.generate(content, valid, ident, affect, initial_noise=noise, steps=3, base=base)
    model.set_flow_source_std(torch.ones(52))
    unit = model.generate(content, valid, ident, affect, initial_noise=noise, steps=3, base=base)
    torch.testing.assert_close(original['raw_motion'], unit['raw_motion'], rtol=0, atol=0)
    model.set_flow_source_std(None); torch.manual_seed(101)
    original_random = model.generate(content, valid, ident, affect, steps=3, base=base)
    model.set_flow_source_std(torch.ones(52)); torch.manual_seed(101)
    unit_random = model.generate(content, valid, ident, affect, steps=3, base=base)
    torch.testing.assert_close(original_random['raw_motion'], unit_random['raw_motion'], rtol=0, atol=0)
    assert set(model.state_dict()) == set(state)
    cfg = config(); cfg['model']['flow_source_std'] = torch.linspace(.0001, 1.3, 52).tolist()
    restored = NeutralAffectSystem(cfg).eval(); restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored.flow_source_std, torch.tensor(cfg['model']['flow_source_std']))
    assert restored.flow_source_scaled


def test_scaled_source_has_same_training_and_sampling_algebra_without_output_gain():
    torch.manual_seed(32)
    model = NeutralAffectSystem(config()).eval()
    std = torch.linspace(.0001, 1.3, 52)
    model.set_flow_source_std(std)
    valid, content, base, ident, affect, noise = inputs()
    observed = valid[..., None].expand_as(noise)
    motion = torch.rand_like(noise)
    time = torch.tensor([.2, .8])
    target = torch.where(observed, (motion-base['b0']-ident['baseline'][:, None])/.25, 0.)
    source = torch.where(observed, noise*std, 0.)
    result = model.flow(motion, content, valid, ident, affect, noise=noise, time=time, base=base)
    torch.testing.assert_close(result['velocity_target'], target-source)
    torch.testing.assert_close(result['x_t'], (1-time[:, None, None])*source+time[:, None, None]*target)
    result['prediction'].square().mean().backward()
    grads = [p.grad for p in model.renderer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads) > 0
    replay = decode_with_sampler(model, content, valid, ident, affect, base, noise, 3, 'euler')
    rollout = model.generate(content, valid, ident, affect, initial_noise=noise, steps=3, base=base)
    torch.testing.assert_close(replay, rollout['raw_residual'], rtol=1e-5, atol=1e-6)
    # With a zero vector field the decoded residual is exactly the scaled
    # prior, while the deterministic B0/identity anchor remains unscaled.
    model.renderer.forward = lambda x, *args, **kwargs: torch.zeros_like(x)
    pred = model.generate(content, valid, ident, affect, initial_noise=noise, steps=3, base=base)
    torch.testing.assert_close(pred['raw_residual'], .25*source)
    torch.testing.assert_close(pred['raw_motion'], torch.where(observed, .3+.25*source, 0.))
    changed = noise.clone(); changed[~observed] = float('nan')
    safe = model.generate(content, valid, ident, affect, initial_noise=changed, steps=3, base=base)
    torch.testing.assert_close(safe['raw_motion'], pred['raw_motion'], rtol=0, atol=0)


@pytest.mark.parametrize('std', [torch.zeros(52), torch.ones(51), torch.full((52,), float('inf'))])
def test_source_rejects_degenerate_or_malformed_spread(std):
    with pytest.raises(ValueError):
        NeutralAffectSystem(config()).set_flow_source_std(std)


def test_source_file_binding_and_population_formula(tmp_path):
    support = torch.tensor([True, True, False])
    stats = {'schema': 'frozen_train_residual_source_spread_v1', 'source_fit_split': 'train',
             'test_loaded': False, 'validation_motion_used': False, 'training_performed': False,
             'checkpoint_sha256': 'warm', 'data_manifest_sha256': 'manifest', 'normalization': .25,
             'residual_support': support.tolist(), 'train_clips': 10,
             'nondegeneracy_floor_normalized': 1e-4, 'residual_std': [.025, 0., 0.],
             'count': [50, 50, 0], 'normalized_source_std': [.1, .0001, 1.]}
    path = tmp_path/'stats.json'; path.write_text(json.dumps(stats))
    std, info = load_flow_source_stats(path, checkpoint_sha256='warm', manifest_sha256='manifest',
                                       support=support, residual_scale=.25)
    torch.testing.assert_close(std, torch.tensor([.1, .0001, 1.]))
    assert info['validation_motion_used'] is False
    stats['validation_motion_used'] = True; path.write_text(json.dumps(stats))
    with pytest.raises(ValueError):
        load_flow_source_stats(path, checkpoint_sha256='warm', manifest_sha256='manifest', support=support, residual_scale=.25)


@pytest.mark.parametrize('args', [
    ['--flow-source-noise', 'train-residual-std'],
    ['--flow-source-stats', 'unused.json'],
    ['--flow-source-stats', 'unused.json', '--paper-data', 'unused', '--stage-checkpoint', 'unused.pt',
     '--start-stage', 'audio', '--end-stage', 'audio', '--protect-mouth'],
])
def test_nonisolated_source_trial_fails_before_filesystem_access(monkeypatch, args):
    monkeypatch.setattr(sys, 'argv', ['train_full_staged.py', '--output', 'unused', *args])
    with pytest.raises(ValueError):
        main()
