import copy

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_full_staged import base_forward, parser, stage1_training_dropout
from tests.test_full_staged_runner import config


def test_default_preserves_legacy_mode_and_probabilities():
    model = NeutralAffectSystem(config()).eval()
    assert parser().parse_args(['--output', 'trial']).stage1_train_dropout is None
    assert stage1_training_dropout(model.stage1, None) is None
    assert not model.stage1.training
    assert all(m.p == 0 for m in model.stage1.modules() if isinstance(m, torch.nn.Dropout))


def test_eval_unchanged_train_stochastic_gradients_and_other_modules_untouched():
    torch.set_num_threads(1)
    torch.manual_seed(20)
    model = NeutralAffectSystem(config()).eval()
    model.stage1.requires_grad_(True)
    saved = copy.deepcopy(model.state_dict())
    x = torch.randn(2, 14, 8)
    mask = torch.ones(2, 14, dtype=torch.bool)
    mask[0, 11:] = False
    before = base_forward(model, x, mask)['b0']
    counts = stage1_training_dropout(model.stage1, .1)
    assert counts == {'dropout_modules': 15, 'attention_modules': 5}
    assert model.stage1.training and not model.renderer.training
    a = base_forward(model, x, mask, gradients=True)['b0']
    b = base_forward(model, x, mask, gradients=True)['b0']
    assert not torch.equal(a, b)
    a.square().mean().backward()
    grads = [p.grad for p in model.stage1.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    model.stage1.eval()
    after = base_forward(model, x, mask)['b0']
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


def test_zero_control_is_deterministic_and_eval_caches_ignore_padding():
    torch.set_num_threads(1)
    model = NeutralAffectSystem(config()).eval()
    stage1_training_dropout(model.stage1, 0.)
    x = torch.randn(2, 13, 8)
    mask = torch.ones(2, 13, dtype=torch.bool)
    a = base_forward(model, x, mask, gradients=True)['b0']
    b = base_forward(model, x, mask, gradients=True)['b0']
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.stage1.eval()
    a = base_forward(model, x, mask)['b0']
    b = base_forward(model, torch.nn.functional.pad(x, (0, 0, 0, 5)),
                     torch.nn.functional.pad(mask, (0, 5)))['b0']
    torch.testing.assert_close(a, b[:, :13], rtol=0, atol=0)


@pytest.mark.parametrize('probability', [True, -.01, .51, float('nan'), float('inf')])
def test_invalid_policy_rejected_before_mode_change(probability):
    model = NeutralAffectSystem(config()).eval()
    with pytest.raises(ValueError):
        stage1_training_dropout(model.stage1, probability)
    assert not model.stage1.training
