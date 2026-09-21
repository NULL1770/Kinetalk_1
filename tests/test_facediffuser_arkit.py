import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kinetalk_b0.models.facediffuser_arkit import FaceDiffuserARKit, SOURCE_COMMIT, cosine_betas


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example():
    torch.manual_seed(49)
    model = FaceDiffuserARKit(content_dim=7, latent_dim=12, gru_hidden=16,
                              diffusion_steps=8, num_layers=2).eval()
    valid = torch.ones(2, 13, dtype=torch.bool)
    valid[0, [0, 5, 12]] = False
    valid[1, 10:] = False
    target = torch.randn(2, 13, 52)
    target[..., 51] = float('nan')
    target[~valid] = float('nan')
    content = torch.randn(2, 13, 7)
    content[~valid] = float('nan')
    noise = torch.randn_like(target)
    return model, target, torch.tensor([0, 7]), content, valid, noise


def official_beat_class():
    """Execute the pinned class definition without loading/downloading HuBERT.

    The frozen fake returns already-aligned audio. All official time/LN/GRU/
    output computations remain exactly as supplied in the pinned source.
    """
    source = Path(__file__).resolve().parents[1] / 'third_party/facediffuser/models.py'
    tree = ast.parse(source.read_text())
    body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FaceDiffBeat']
    assert len(body) == 1

    class CachedHubert(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=7))
            self.feature_extractor = SimpleNamespace(_freeze_parameters=lambda: None)

        @classmethod
        def from_pretrained(cls, name):
            assert name == 'facebook/hubert-base-ls960'
            return cls()

        def forward(self, x):
            return SimpleNamespace(last_hidden_state=x)

    namespace = {'torch': torch, 'np': np, 'nn': nn, 'Tensor': torch.Tensor,
                 'HubertModel': CachedHubert,
                 'adjust_input_representation': lambda audio, x, ifps, ofps: (audio, x, len(x[0]))}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['FaceDiffBeat']


def test_official_beat_decoder_core_is_numerically_identical_on_dense_inputs():
    torch.manual_seed(8)
    adapter = FaceDiffuserARKit(content_dim=7, latent_dim=12, gru_hidden=16,
                                num_layers=2, diffusion_steps=8).eval()
    args = SimpleNamespace(input_fps=25, output_fps=25, diff_steps=8, device='cpu')
    official = official_beat_class()(args, 51, latent_dim=12, cond_feature_dim=7,
                                      diffusion_steps=8, gru_latent_dim=16, num_layers=2).eval()
    for name in ('time_mlp', 'norm_cond', 'gru', 'final_layer'):
        getattr(official, name).load_state_dict(getattr(adapter, name).state_dict())
    x, audio = torch.randn(2, 9, 52), torch.randn(2, 9, 7)
    times, valid = torch.tensor([1, 6]), torch.ones(2, 9, dtype=torch.bool)
    expected = official(x[..., :51], times, audio)
    actual = adapter(x, times, audio, valid)
    torch.testing.assert_close(actual[..., :51], expected, rtol=0, atol=0)
    assert actual[..., 51].count_nonzero() == 0
    assert SOURCE_COMMIT == 'e15f3500fdae0eda962f5d018488dfa0a1a9d552'


def test_cosine_schedule_is_the_pinned_official_schedule():
    source = Path(__file__).resolve().parents[1] / 'third_party/facediffuser/diffusion/gaussian_diffusion.py'
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('get_named_beta_schedule', 'betas_for_alpha_bar')]
    import math
    namespace = {'np': np, 'math': math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    for steps in (8, 1000):
        expected = namespace['get_named_beta_schedule']('cosine', steps, 1.)
        np.testing.assert_array_equal(cosine_betas(steps).numpy(), expected)


def test_q_sample_and_fixed_small_posterior_match_ddpm_equations():
    model, target, times, content, valid, noise = example()
    observed = valid[..., None] & model.support
    clean = torch.where(observed, target, 0.)
    expected = model.sqrt_alpha_bar[times].float()[:, None, None] * clean + model.sqrt_one_minus_alpha_bar[times].float()[:, None, None] * torch.where(observed, noise, 0.)
    x_t = model.q_sample(target, times, valid, noise=noise)
    torch.testing.assert_close(x_t, expected, rtol=0, atol=0)
    result = model.p_mean_variance(x_t, times, content, valid)
    expected_mean = model.posterior_mean_coef1[times].float()[:, None, None] * result['pred_xstart'] + model.posterior_mean_coef2[times].float()[:, None, None] * x_t
    torch.testing.assert_close(result['mean'], expected_mean, rtol=0, atol=0)
    draw = model.p_sample(x_t, times, content, valid, noise=noise)
    torch.testing.assert_close(draw['sample'][0], draw['pred_xstart'][0], rtol=0, atol=0)
    assert model.posterior_variance[0] == 0


def test_masked_x0_loss_ignores_unknown_targets_and_uses_equal_clip_weighting():
    model, target, times, content, valid, noise = example()
    channels = torch.ones(2, 52, dtype=torch.bool)
    channels[0, 3] = False
    target[0, :, 3] = float('nan')
    target.requires_grad_()
    result = model.training_losses(target, times, content, valid, noise=noise, channel_mask=channels)
    observed = valid[..., None] & channels[:, None] & model.support
    expected = torch.stack([(result['pred_xstart'][i][observed[i]] - target[i][observed[i]]).square().mean()
                            for i in range(2)])
    torch.testing.assert_close(result['loss'], expected)
    assert result['pred_xstart'][~observed].count_nonzero() == 0
    result['loss'].mean().backward()
    assert torch.isfinite(target.grad).all()
    assert target.grad[~observed].count_nonzero() == 0
    assert model.final_layer.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_padding_and_unknown_values_leave_denoising_unchanged():
    model, target, times, content, valid, noise = example()
    result = model.training_losses(target, times, content, valid, noise=noise)
    pad = lambda x: F.pad(x, (0, 0, 0, 9), value=float('nan'))
    longer = model.training_losses(pad(target), times, pad(content), F.pad(valid, (0, 9), value=False), noise=pad(noise))
    torch.testing.assert_close(longer['loss'], result['loss'], rtol=0, atol=0)
    torch.testing.assert_close(longer['pred_xstart'][:, :13], result['pred_xstart'], rtol=0, atol=0)
    assert longer['pred_xstart'][:, 13:].count_nonzero() == 0


def test_ancestral_sampling_is_seeded_target_free_and_unclipped():
    model, _, _, content, valid, noise = example()
    for parameter in model.parameters():
        parameter.data.zero_()
    model.final_layer.bias.data.fill_(1.5)
    rng = torch.get_rng_state().clone()
    first = model.sample(content, valid, initial_noise=noise, generator=torch.Generator().manual_seed(20))
    second = model.sample(content, valid, initial_noise=noise, generator=torch.Generator().manual_seed(20))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first[valid][..., :51], torch.full_like(first[valid][..., :51], 1.5), rtol=0, atol=0)
    assert first[..., 51].count_nonzero() == 0
    assert torch.equal(rng, torch.get_rng_state())
    model.train()
    with pytest.raises(ValueError, match='eval'):
        model.sample(content, valid, initial_noise=noise, generator=torch.Generator())


def test_clip_condition_is_explicit_and_reload_retains_fixed_support():
    model, target, times, content, valid, noise = example()
    with pytest.raises(ValueError, match='Audio-only'):
        model(target, times, content, valid, clip_condition=torch.zeros(2, 3))
    cfg = model.export_config()
    conditional = FaceDiffuserARKit(**{**cfg, 'clip_condition_dim': 3}).eval()
    with pytest.raises(ValueError, match='Declared'):
        conditional(target, times, content, valid)
    value = conditional(target, times, content, valid, clip_condition=torch.zeros(2, 3))
    changed = conditional(target, times, content, valid, clip_condition=torch.ones(2, 3))
    assert not torch.equal(value, changed)
    condition = torch.ones(2, 3, requires_grad=True)
    conditional(target, times, content, valid, clip_condition=condition).square().sum().backward()
    assert condition.grad is None
    assert conditional.final_layer.weight.grad.abs().sum() > 0
    rebuilt = FaceDiffuserARKit(**cfg).eval()
    rebuilt.load_state_dict(model.state_dict())
    torch.testing.assert_close(rebuilt(target, times, content, valid), model(target, times, content, valid), rtol=0, atol=0)


def test_invalid_observed_targets_or_schedule_are_rejected():
    model, target, times, content, valid, noise = example()
    target[0, 1, 0] = float('nan')
    with pytest.raises(ValueError, match='finite observed'):
        model.training_losses(target, times, content, valid, noise=noise)
    with pytest.raises(ValueError, match='diffusion_steps'):
        cosine_betas(1)
    with pytest.raises(ValueError, match='support'):
        FaceDiffuserARKit(support=[1] * 52)
