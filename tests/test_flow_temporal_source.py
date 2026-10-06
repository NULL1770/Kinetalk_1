import copy
import json
import sys

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.diagnose_flow_sampling import decode_with_sampler
from scripts.train_full_staged import load_flow_temporal_stats, main
from tests.test_flow_source_noise import inputs
from tests.test_full_staged_runner import config


def system():
    model = NeutralAffectSystem(config()).eval()
    model.set_flow_source_std(torch.linspace(.05, 1.3, 52))
    model.set_flow_source_rho(torch.linspace(-.85, .97, 52))
    return model


def test_zero_rho_exact_legacy_flow_rollout_and_implicit_zero():
    torch.manual_seed(36)
    model = NeutralAffectSystem(config()).eval()
    model.set_flow_source_std(torch.linspace(.01, 1.2, 52))
    valid, content, base, ident, affect, noise = inputs()
    motion, time = torch.rand_like(noise), torch.tensor([.2, .8])
    legacy = model.flow(motion, content, valid, ident, affect, noise=noise, time=time, base=base)
    rollout = model.generate(content, valid, ident, affect, initial_noise=noise, steps=3, base=base)
    implicit = model.generate(content, valid, ident, affect, steps=3, base=base)
    model.set_flow_source_rho(torch.zeros(52))
    new = model.flow(motion, content, valid, ident, affect, noise=noise, time=time, base=base)
    for key in legacy:
        torch.testing.assert_close(new[key], legacy[key], rtol=0, atol=0)
    torch.testing.assert_close(rollout['motion'], model.generate(content, valid, ident, affect,
                               initial_noise=noise, steps=3, base=base)['motion'], rtol=0, atol=0)
    model.set_flow_source_rho(torch.full((52,), .9))
    torch.testing.assert_close(implicit['motion'], model.generate(content, valid, ident, affect,
                               steps=3, base=base)['motion'], rtol=0, atol=0)


def test_stationary_source_variance_lag_and_nonsingular_innovation():
    torch.manual_seed(37); torch.set_num_threads(1)
    model = system()
    white = torch.randn(15000, 8, 52)
    source = model.source_noise(white)
    standardized = source / model.flow_source_std
    torch.testing.assert_close(standardized.square().mean(0), torch.ones(8, 52), atol=.045, rtol=0)
    cross = (standardized[:, :-1] * standardized[:, 1:]).mean((0, 1))
    energy = (standardized[:, :-1].square().mean((0, 1))*standardized[:, 1:].square().mean((0, 1))).sqrt()
    torch.testing.assert_close(cross / energy,
                               model.flow_source_rho, atol=.025, rtol=0)
    innovation = (standardized[:, 1:] - standardized[:, :-1]*model.flow_source_rho)
    torch.testing.assert_close(innovation.square().mean((0, 1)), 1-model.flow_source_rho.square(), atol=.02, rtol=0)
    assert (innovation.square().mean((0, 1)) > 0).all()


def test_native_gap_reset_padding_and_invalid_value_invariance():
    torch.manual_seed(38); model = system()
    white = torch.randn(2, 7, 52)
    valid = torch.tensor([[1,1,0,1,1,0,0], [1,1,1,0,1,1,1]], dtype=torch.bool)
    result = model.source_noise(white, valid)
    corrupt = white.clone(); corrupt[~valid] = float('nan')
    torch.testing.assert_close(model.source_noise(corrupt, valid), result, rtol=0, atol=0)
    padded = torch.cat([white, torch.full((2, 3, 52), float('nan'))], dim=1)
    padded_mask = torch.cat([valid, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(model.source_noise(padded, padded_mask)[:, :7], result, rtol=0, atol=0)
    torch.testing.assert_close(result[0, 3], white[0, 3]*model.flow_source_std, rtol=0, atol=0)
    torch.testing.assert_close(result[1, 4], white[1, 4]*model.flow_source_std, rtol=0, atol=0)
    assert (result[~valid] == 0).all()


def test_flow_source_independent_of_target_channel_observations_and_matches_rollout():
    torch.manual_seed(39); model = system()
    valid, content, base, ident, affect, noise = inputs()
    valid[0, 1] = False
    observed = valid[..., None].expand_as(noise).clone()
    observed[1, 1, 17] = False  # This channel missing in target must not reset its prior.
    motion = torch.rand_like(noise); time = torch.tensor([0., 0.])
    full = model.flow(motion, content, valid, ident, affect, noise=noise, time=time, base=base)
    partial = model.flow(motion, content, valid, ident, affect, noise=noise, time=time,
                         base=base, observation_mask=observed)
    torch.testing.assert_close(partial['x_t'][observed], full['x_t'][observed], rtol=0, atol=0)
    source = model.source_noise(torch.where(valid[..., None], noise, 0.), valid)
    torch.testing.assert_close(partial['x_t'], torch.where(observed, source, 0.), rtol=0, atol=0)
    partial['prediction'].square().mean().backward()
    gradients = [p.grad for p in model.renderer.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    replay = decode_with_sampler(model, content, valid, ident, affect, base, noise, 3, 'euler')
    rollout = model.generate(content, valid, ident, affect, initial_noise=noise, steps=3, base=base)
    torch.testing.assert_close(replay, rollout['raw_residual'], rtol=1e-5, atol=1e-6)
    model.renderer.forward = lambda x, *args, **kwargs: torch.zeros_like(x)
    zero_field = model.generate(content, valid, ident, affect, initial_noise=noise, steps=3, base=base)
    torch.testing.assert_close(zero_field['raw_residual'], .25*source, rtol=0, atol=0)


def test_config_roundtrip_without_new_parameter_keys():
    model = system(); cfg = copy.deepcopy(config())
    cfg['model'].update(flow_source_std=model.flow_source_std.tolist(), flow_source_rho=model.flow_source_rho.tolist())
    loaded = NeutralAffectSystem(json.loads(json.dumps(cfg))).eval()
    loaded.load_state_dict(model.state_dict(), strict=True)
    assert 'flow_source_rho' not in loaded.state_dict()
    white = torch.randn(2, 8, 52)
    torch.testing.assert_close(loaded.source_noise(white), model.source_noise(white), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Cross-device inference compatibility requires CUDA')
def test_correlated_rollout_accepts_cpu_white_noise_for_cuda_model():
    torch.manual_seed(40);model=system().cuda()
    valid,content,base,ident,affect,noise=inputs()
    valid,content=valid.cuda(),content.cuda()
    base={k:v.cuda() for k,v in base.items()};ident={k:v.cuda() for k,v in ident.items()}
    affect={k:v.cuda() for k,v in affect.items()}
    cpu=model.generate(content,valid,ident,affect,initial_noise=noise,steps=2,base=base)
    gpu=model.generate(content,valid,ident,affect,initial_noise=noise.cuda(),steps=2,base=base)
    torch.testing.assert_close(cpu['raw_motion'],gpu['raw_motion'],rtol=0,atol=0)


@pytest.mark.parametrize('rho', [torch.ones(52), torch.full((52,), -1.), torch.zeros(51), torch.full((52,), float('nan'))])
def test_invalid_rho_rejected(rho):
    with pytest.raises(ValueError):
        system().set_flow_source_rho(rho)


def test_lag_statistics_formula_and_source_binding(tmp_path):
    support = torch.tensor([True, True, False])
    spread_info = {'sha256':'spread', 'train_clips':10}
    stats = {'schema':'frozen_train_demeaned_residual_lag_v1', 'source_fit_split':'train',
             'test_loaded':False,'validation_motion_used':False,'training_performed':False,
             'checkpoint_sha256':'warm','data_manifest_sha256':'manifest','source_stats_sha256':'spread',
             'residual_support':support.tolist(),'train_clips':10,'nondegeneracy_rho_margin':1e-4,
             'results':{'adjacent_count':[50,50,0],'cross_sum':[1.,0.,0.],
                        'left_square_sum':[1.,0.,0.],'right_square_sum':[1.,0.,0.],
                        'source_rho':[.9999,0.,0.]}}
    path = tmp_path/'lag.json'; path.write_text(json.dumps(stats))
    kwargs = dict(checkpoint_sha256='warm',manifest_sha256='manifest',support=support,source_info=spread_info)
    rho, info = load_flow_temporal_stats(path, **kwargs)
    torch.testing.assert_close(rho, torch.tensor([.9999,0.,0.]))
    assert not info['validation_motion_used']
    stats['source_stats_sha256'] = 'another'; path.write_text(json.dumps(stats))
    with pytest.raises(ValueError): load_flow_temporal_stats(path, **kwargs)


@pytest.mark.parametrize('args', [
    ['--flow-source-noise','train-residual-ar','--flow-source-stats','missing'],
    ['--flow-source-temporal-stats','missing'],
])
def test_incomplete_trial_rejected_before_io(monkeypatch, args):
    monkeypatch.setattr(sys,'argv',['train_full_staged.py','--output','unused',*args])
    with pytest.raises(ValueError): main()
