"""Synthetic formal-batch integration checks; never open persisted datasets."""
import copy

import pytest
import torch
from torch import nn

from kinetalk_b0.models.history_upper_flow import HistoryUpperFlow
from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
from scripts import train_prefix_formal as formal


CFG = {'model': {'content_dim': 4, 'emotion_dim': 3, 'style_dim': 2,
                 'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.}}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(912)
    clock = torch.arange(96).float()[None, :, None]
    valid = torch.ones(3, 96, dtype=torch.bool)
    valid[0, 11:13] = False
    valid[1, :16] = False
    valid[1, 21:25] = False
    valid[1, 32:48] = False
    valid[1, 90:] = False
    valid[2, :89] = False
    b = {'valid': valid, 'h0': clock.expand(3, -1, 4).clone(),
         'audio_global': torch.randn(3, 3), 'audio_intensity': torch.randn(3, 1),
         'motion': torch.randn(3, 96, 52), 'static_upper': torch.randn(3, 9)}
    local = torch.randn(3, 96, 3)
    identity = {'code': torch.stack((torch.arange(3).float(), torch.ones(3)), -1)}
    scales = torch.linspace(.5, 2., 9)
    b['motion'][~valid] = float('nan')
    b['h0'][~valid] = float('nan')
    local[~valid] = float('nan')
    return b, identity, local, scales


class RecordingFlow(nn.Module):
    """Observable known-token interface with a nonconstant supervised loss."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.calls = []

    def decode_prefix(self, valid, h0, identity, affect, local, noise, *, known, known_mask, steps):
        mean = known.sum(1) / known_mask.sum(1)[:, None].clamp_min(1)
        return torch.where(known_mask[..., None], known,
                           torch.where(valid[..., None], noise + mean[:, None], 0.))

    def flow_loss_prefix(self, target, valid, h0, identity, affect, local, noise, time, *, known, known_mask):
        self.calls.append({'valid': valid.clone(), 'known': known.clone(),
                           'known_mask': known_mask.clone(), 'h0': h0.clone(),
                           'identity': identity.clone(), 'target': target.clone()})
        return self.weight * target[valid & ~known_mask].square().mean()


def execute(model, b, identity, local, scales, probability, use_prefix, seed=827):
    return formal.training_batch(model, b, identity, local, scales,
                                 torch.Generator().manual_seed(seed), 1, probability, use_prefix)


def test_six_chunk_loss_has_one_valid_current_frame_denominator_despite_gaps():
    b, identity, local, scales = fixture()
    model = RecordingFlow()
    loss, counts, _ = execute(model, b, identity, local, scales, 1., True)
    expected = formal.p.h.normalized_target(b, scales)[b['valid']].square().mean()
    torch.testing.assert_close(loss, expected)
    assert len(model.calls) == 6
    assert sum(int((c['valid'] & ~c['known_mask']).sum()) for c in model.calls) == int(b['valid'].sum())
    assert counts == {'teacher_draw_count': 11, 'valid_history_count': 7,
                      'teacher_eligible_count': 7, 'teacher_used_count': 7}
    loss.backward()
    torch.testing.assert_close(model.weight.grad, expected)


@pytest.mark.parametrize('probability', [0., 1.])
def test_teacher_extremes_supply_only_valid_strict_past_from_selected_source(monkeypatch, probability):
    b, identity, local, scales = fixture()
    model = RecordingFlow()
    original = formal.p.rollout
    observed = {}

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed['generated'] = result.clone()
        return result

    monkeypatch.setattr(formal.p, 'rollout', capture)
    _, counts, _ = execute(model, b, identity, local, scales, probability, True)
    source = formal.p.h.normalized_target(b, scales) if probability else observed['generated']
    for start, call in zip(range(0, 96, 16), model.calls):
        ids = call['identity'][:, 0].long()
        assert not call['known_mask'][:, 8:].any()
        assert not call['known'][:, 8:].any()
        for row, sample in enumerate(ids):
            expected_mask = torch.zeros(24, dtype=torch.bool)
            expected = torch.zeros(24, 9)
            if start:
                expected_mask[:8] = b['valid'][sample, start-8:start]
                expected[:8] = torch.where(expected_mask[:8, None], source[sample, start-8:start], 0.)
            assert torch.equal(call['known_mask'][row], expected_mask)
            torch.testing.assert_close(call['known'][row], expected, atol=0, rtol=0)
            assert torch.isfinite(call['known']).all()
    assert counts['teacher_used_count'] == (7 if probability else 0)
    assert counts['teacher_draw_count'] == (11 if probability else 0)


def test_no_prefix_ignores_teacher_selection_and_has_no_observed_past_tokens():
    b, identity, local, scales = fixture()
    zero, one = RecordingFlow(), RecordingFlow()
    loss0, counts0, draws0 = execute(zero, b, identity, local, scales, 0., False)
    loss1, counts1, draws1 = execute(one, b, identity, local, scales, 1., False)
    torch.testing.assert_close(loss0, loss1, atol=0, rtol=0)
    assert counts0['teacher_used_count'] == counts1['teacher_used_count'] == 0
    assert counts1['teacher_eligible_count'] == 7
    for left, right in zip(zero.calls, one.calls):
        assert not right['valid'][:, :8].any()
        assert not right['known_mask'].any() and not right['known'].any()
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
    for left, right in zip(draws0, draws1):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_both_arms_consume_identical_full_random_stream_with_masked_chunks():
    b, identity, local, scales = fixture()
    generators = [torch.Generator().manual_seed(15) for _ in range(2)]
    outputs = []
    for use_prefix, generator in zip((False, True), generators):
        outputs.append(formal.training_batch(RecordingFlow(), b, identity, local, scales,
                                             generator, 1, .4, use_prefix))
    assert [tuple(x.shape) for x in outputs[0][2]] == [(3, 6, 24, 9), (3, 96, 9), (3, 6), (3, 6)]
    for left, right in zip(outputs[0][2], outputs[1][2]):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert torch.equal(generators[0].get_state(), generators[1].get_state())
    for key in ('teacher_draw_count', 'valid_history_count', 'teacher_eligible_count'):
        assert outputs[0][1][key] == outputs[1][1][key]
    assert outputs[0][1]['teacher_used_count'] == 0


@pytest.mark.parametrize('probability', [0., 1.])
@pytest.mark.parametrize('use_prefix', [False, True])
def test_real_prefix_model_has_finite_backward_with_nan_padding(probability, use_prefix):
    b, identity, local, scales = fixture()
    model = PrefixUpperFlow(CFG).eval()
    before = {key: value.clone() for key, value in b.items()}
    loss, _, _ = execute(model, b, identity, local, scales, probability, use_prefix)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert sum(float(grad.abs().sum()) for grad in grads) > 0
    for key, value in b.items():
        torch.testing.assert_close(value, before[key], equal_nan=True, atol=0, rtol=0)


@pytest.mark.parametrize('invalid', ['all_empty', 'wrong_clock', 'negative_probability', 'large_probability'])
def test_invalid_formal_batch_is_rejected_before_random_draws(invalid):
    b, identity, local, scales = fixture()
    probability = .5
    if invalid == 'all_empty':
        b['valid'].fill_(False)
    elif invalid == 'wrong_clock':
        b['valid'] = b['valid'][:, :95]
    else:
        probability = -0.1 if invalid == 'negative_probability' else 1.1
    generator = torch.Generator().manual_seed(12)
    before = generator.get_state()
    with pytest.raises(ValueError, match='Formal batch'):
        formal.training_batch(RecordingFlow(), b, identity, local, scales, generator, 1, probability, True)
    assert torch.equal(generator.get_state(), before)


def test_shared_warmstart_copies_entire_backbone_and_keeps_new_embedding_zero():
    torch.manual_seed(718)
    source = HistoryUpperFlow(CFG, use_history=False, history_hidden=5).state_dict()
    destination = PrefixUpperFlow(CFG)
    removed = formal.load_shared_warmstart(destination, source)
    assert len(removed) == 9
    assert set(removed) == set(source) - set(destination.state_dict())
    assert not destination.known_embedding.weight.any()
    for key, value in destination.state_dict().items():
        if key != 'known_embedding.weight':
            torch.testing.assert_close(value, source[key], atol=0, rtol=0)


@pytest.mark.parametrize('change', ['unknown_key', 'missing_history_key', 'missing_backbone_key', 'existing_embedding'])
def test_warmstart_rejects_unexpected_or_missing_parameters(change):
    source = copy.deepcopy(HistoryUpperFlow(CFG, use_history=False, history_hidden=5).state_dict())
    destination = PrefixUpperFlow(CFG)
    if change == 'unknown_key':
        source['mystery.weight'] = torch.ones(1)
    elif change == 'missing_history_key':
        source.pop('history_gru.bias_ih')
    elif change == 'missing_backbone_key':
        source.pop(next(key for key in destination.state_dict() if key != 'known_embedding.weight'))
    else:
        source['known_embedding.weight'] = torch.zeros_like(destination.known_embedding.weight)
    with pytest.raises(ValueError, match='Warmstart'):
        formal.load_shared_warmstart(destination, source)
