import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow, ContinuousUpperAE
from scripts import audit_prior_sampling_resolution as sampling
from tests.test_audit_event_receiver_bottleneck import fixture


def tiny_inputs():
    valid = torch.tensor([[True, True, False]])
    noise = torch.tensor([[[.5, -.25], [.1, .3], [float('nan'), float('nan')]]])
    context = torch.zeros(1, 3)
    blocks = torch.zeros(1, 3, 2, 1540)
    frames = torch.tensor([[[True, True], [True, False], [False, False]]])
    return valid, context, blocks, noise, frames


def test_euler_wrapper_is_exact_original_solver_and_preserves_noise_and_state():
    torch.manual_seed(32)
    prior = ContinuousLatentFlow(3, latent_dim=2, hidden=8, depth=1, block_size=2).eval()
    valid, context, blocks, noise, frames = tiny_inputs()
    initial_noise = noise.clone()
    before = sampling.common._value_sha(prior.state_dict())
    expected = prior.sample(valid, context, blocks, noise, steps=24, use_audio=False,
                            audio_frame_valid=frames)
    actual = sampling.SamplingWrapper(prior, 'euler').sample(
        valid, context, blocks, noise, steps=24, audio_frame_valid=frames)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:, 2], torch.zeros(1, 2))
    torch.testing.assert_close(noise, initial_noise, equal_nan=True)
    assert before == sampling.common._value_sha(prior.state_dict())
    assert not actual.requires_grad


class AnalyticField(nn.Module):
    latent_dim = 2
    block_size = 2

    def __init__(self):
        super().__init__()
        self.times = []

    def velocity(self, x, time, valid, context, audio_blocks, use_audio, *, audio_frame_valid=None):
        assert use_audio is False
        assert torch.isfinite(x).all()
        assert torch.equal(x[~valid], torch.zeros_like(x[~valid]))
        self.times.append(float(time[0]))
        # dx/dt=x+t; closed solution at 1 is e*x0 + e-2.
        return torch.where(valid[..., None], x+time[:, None, None], 0.)


def test_heun_solves_known_ode_with_exact_nfe_endpoint_and_mask():
    valid, context, blocks, noise, frames = tiny_inputs()
    errors = []
    for steps in (24, 48):
        field = AnalyticField()
        actual = sampling.SamplingWrapper(field, 'heun').sample(
            valid, context, blocks, noise, steps=steps, audio_frame_valid=frames)
        expected = math.e*noise[valid]+math.e-2
        errors.append(float((actual[valid]-expected).abs().max()))
        assert len(field.times) == 2*steps
        assert field.times[0] == 0. and field.times[-1] == 1.
        assert torch.equal(actual[:, 2], torch.zeros(1, 2))
    assert errors[0] < .002
    assert errors[1] < errors[0]/3.5


def test_native_evaluation_reproduces_saved_euler_and_rejects_changed_draws(tmp_path):
    torch.manual_seed(8)
    torch.set_num_threads(1)
    clip, _ = fixture()
    clip.update(features=torch.zeros(23, 1540), context=torch.zeros(3), _audit_block_size=2)
    stats = {'metric_scale': torch.ones(9), 'residual_scale': torch.ones(9),
             'latent_mean': torch.zeros(2), 'latent_scale': torch.ones(2),
             'context_mean': torch.zeros(3), 'context_scale': torch.ones(3),
             'audio_mean': torch.zeros(1540), 'audio_scale': torch.ones(1540)}
    ae = ContinuousUpperAE(block_size=2, latent_dim=2, hidden=8, depth=1).eval()
    prior = ContinuousLatentFlow(3, latent_dim=2, block_size=2, hidden=8, depth=1).eval()
    sampling.native.evaluate_generation(prior, ae, [clip], stats, tmp_path/'reference',
                                        use_audio=False, seeds=sampling.audit.SEEDS, steps=24)
    reference = sampling.common._load(tmp_path/'reference/curves.pt')['clips']
    for method, steps in (('euler', 24), ('heun', 24)):
        destination = tmp_path/method
        sampling.native.evaluate_generation(sampling.SamplingWrapper(prior, method), ae,
            [clip], stats, destination, use_audio=False, seeds=sampling.audit.SEEDS, steps=steps)
        curves = sampling.common._load(destination/'curves.pt')['clips']
        result = sampling.verify_same_draws(curves, reference, [clip], compare_samples=method == 'euler')
        assert result['same_native_runs_seeds_and_noise_keys']
        if method == 'euler':
            assert result['bitwise_equal'] and result['max_absolute_difference'] == 0.
        scored, arkit = sampling.audit.rescore_arm(curves, [clip], stats, destination,
                                                  deterministic=False, scope='synthetic test')
        assert sampling.audit.table_row(method, scored, arkit)['arkit']['arkit_mbe']['status'] == 'computed'
        assert len(sampling.signal_summary(scored)['groups']) == 4
    changed = copy.deepcopy(reference)
    changed['clip']['run_records'][0]['noise_seed'] += 1
    with pytest.raises(ValueError, match='noise keys'):
        sampling.verify_same_draws(changed, reference, [clip])
    changed = copy.deepcopy(reference)
    changed['clip']['samples'][0, 0, 0] += .01
    with pytest.raises(ValueError, match='does not reproduce'):
        sampling.verify_same_draws(changed, reference, [clip], compare_samples=True)
