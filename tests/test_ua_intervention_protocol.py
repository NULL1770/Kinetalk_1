"""Interventions preserve frame support, aliases, marginal values and RNG."""
import importlib.util
from pathlib import Path
import pytest
import torch

path = Path(__file__).resolve().parents[1] / '.codex-finalizer/phase30_ua_protocol.py'
spec = importlib.util.spec_from_file_location('phase30_ua_protocol', path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def fixture():
    valid = torch.tensor([[True, False, True, True, False], [False, True, False, False, False]])
    x = torch.arange(30, dtype=torch.float64).reshape(2, 5, 3)
    x[~valid] = float('nan')
    return x, valid


@pytest.mark.parametrize('mode', ['zero', 'static', 'reverse', 'shuffle'])
def test_only_observed_native_frames_and_original_unchanged(mode):
    x, valid = fixture()
    original = x.clone()
    y = protocol.temporal_variant(x, valid, mode, ['clipA', 'clipB'])
    assert not y[~valid].any()
    torch.testing.assert_close(x, original, rtol=0, atol=0, equal_nan=True)
    assert torch.isfinite(y).all()
    if mode in ('reverse', 'shuffle'):
        for i in range(len(x)):
            assert torch.equal(y[i, valid[i]].sort(dim=0).values, x[i, valid[i]].sort(dim=0).values)
    elif mode == 'static':
        assert torch.equal(y[0, valid[0]], x[0, valid[0]].mean(0).expand(3, -1))
    else:
        assert not y.any()


def test_reverse_actual_time_order_and_single_frame():
    x, valid = fixture()
    y = protocol.temporal_variant(x, valid, 'reverse', ['A', 'B'])
    assert torch.equal(y[0, valid[0]], x[0, valid[0]].flip(0))
    assert torch.equal(y[1, valid[1]], x[1, valid[1]])


def test_shuffle_is_clip_bound_batch_order_independent_and_rng_free():
    x = torch.arange(160, dtype=torch.float32).reshape(2, 40, 2)
    valid = torch.ones(2, 40, dtype=torch.bool)
    before = torch.get_rng_state().clone()
    y = protocol.temporal_variant(x, valid, 'shuffle', ['A', 'B'])
    reversed_batch = protocol.temporal_variant(x.flip(0), valid, 'shuffle', ['B', 'A']).flip(0)
    assert torch.equal(y, reversed_batch) and torch.equal(torch.get_rng_state(), before)
    assert not torch.equal(y, x)


def test_static_ignores_padding_and_affect_other_objects_are_identical():
    x, valid = fixture()
    y = protocol.temporal_variant(x, valid, 'static', ['A', 'B'])
    padded = torch.cat([x, torch.full((2, 4, 3), float('nan'))], dim=1)
    expanded = torch.cat([valid, torch.zeros(2, 4, dtype=torch.bool)], dim=1)
    assert torch.equal(y, protocol.temporal_variant(padded, expanded, 'static', ['A', 'B'])[:, :5])
    affect = {'global': torch.ones(2, 64), 'intensity_value': torch.ones(2, 1),
              'u_a': x, 'temporal': x, 'local': x, 'emotion_logits': torch.zeros(2, 8)}
    changed = protocol.replace_temporal(affect, y)
    assert changed['u_a'] is changed['temporal'] is changed['local'] is y
    assert changed['global'] is affect['global'] and changed['intensity_value'] is affect['intensity_value']
    assert changed['emotion_logits'] is affect['emotion_logits'] and affect['u_a'] is x


def test_bad_observed_values_and_empty_frames_rejected():
    x, valid = fixture()
    x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        protocol.temporal_variant(x, valid, 'static', ['A', 'B'])
    with pytest.raises(ValueError):
        protocol.temporal_variant(torch.zeros_like(x), torch.zeros_like(valid), 'zero', ['A', 'B'])


def test_content_static_changes_only_valid_hubert_slice_without_mutation():
    valid = torch.tensor([[True, False, True, True]])
    features = torch.arange(4*1540, dtype=torch.float32).reshape(1, 4, 1540)
    original = features.clone()
    changed = protocol.content_static_features(features, valid)
    assert torch.equal(features, original)
    assert torch.equal(changed[..., 768:], features[..., 768:])
    assert torch.equal(changed[~valid], features[~valid])
    assert torch.equal(changed[0, valid[0], :768], features[0, valid[0], :768].mean(0).expand(3, -1))
