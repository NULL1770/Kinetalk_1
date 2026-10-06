"""Regression contracts for training-only global feature regularization."""
import pytest
import torch

from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts.train_full_staged import audio_affect


def fixture():
    torch.manual_seed(9029)
    model = SlowStateAffect(torch.zeros(20), torch.ones(20)).eval()
    with torch.no_grad():
        model.local_head.weight.normal_(0., .05)
    features = torch.randn(2, 11, 20)
    valid = torch.arange(11)[None] < torch.tensor([11, 7])[:, None]
    features[~valid] = float('nan')
    return model, features, valid


def test_global_regularizer_keeps_native_local_and_original_random_stream():
    model, features, valid = fixture()
    baseline = audio_affect(model, features, valid)
    before = torch.get_rng_state().clone()
    gen = torch.Generator().manual_seed(290047)
    output = audio_affect(model, features, valid, global_dropout=.1, dropout_generator=gen)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(baseline['u_a'], output['u_a'])
    assert baseline['u_a'].abs().sum() > 0
    assert not torch.equal(baseline['global'], output['global'])
    assert all(not child.training for child in model.modules())


def test_eval_disables_regularization_and_private_rng_replays_next_call():
    model, features, valid = fixture()
    gen = torch.Generator().manual_seed(290047)
    state = gen.get_state().clone()
    baseline = model(features, valid, include_legacy_state=False)
    inference = model(features, valid, include_legacy_state=False, global_dropout=.1, dropout_generator=gen)
    assert all(torch.equal(value, inference[key]) for key, value in baseline.items())
    assert torch.equal(state, gen.get_state())
    expected = audio_affect(model, features, valid, global_dropout=.1, dropout_generator=gen)
    restored = torch.Generator()
    restored.set_state(state)
    resumed = audio_affect(model, features, valid, global_dropout=.1, dropout_generator=restored)
    assert all(torch.equal(value, resumed[key]) for key, value in expected.items())


def test_invalid_generator_does_not_leave_training_mode_enabled():
    model, features, valid = fixture()
    with pytest.raises(ValueError, match='independent CPU generator'):
        audio_affect(model, features, valid, global_dropout=.1)
    assert all(not child.training for child in model.modules())


def test_regularization_does_not_create_missing_frame_gradients():
    model, features, valid = fixture()
    features.requires_grad_()
    output = audio_affect(model, features, valid, global_dropout=.1,
                          dropout_generator=torch.Generator().manual_seed(290047))
    output['global'].square().mean().backward()
    assert torch.isfinite(features.grad).all()
    assert (features.grad[~valid] == 0).all()
    assert model.global_head.weight.grad.abs().sum() > 0
