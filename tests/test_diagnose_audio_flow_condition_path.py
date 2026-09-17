import torch
import pytest

from kinetalk_b0.models.dit import ResidualDiT
from scripts.diagnose_audio_flow_condition_path import (
    ConditionCapture, attention_entropy, condition_responses,
    reconstruct_attention, select_fit_metadata, diagnostic_flow_state, run_conditions,
)


def test_explicit_mha_matches_original_and_masks_keys():
    torch.manual_seed(2)
    module = torch.nn.MultiheadAttention(12, 3, dropout=.2, batch_first=True).eval()
    q, k = torch.randn(2, 5, 12), torch.randn(2, 5, 12)
    padding = torch.tensor([[False, False, False, True, True], [False] * 5])
    with torch.no_grad():
        actual, _ = module(q, k, k, key_padding_mask=padding, need_weights=False)
        rebuilt, weights = reconstruct_attention(module, q, k, k, key_padding_mask=padding)
    torch.testing.assert_close(actual, rebuilt, atol=1e-6, rtol=1e-5)
    assert torch.equal(weights[0, :, :, 3:], torch.zeros_like(weights[0, :, :, 3:]))
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))


def test_hooks_preserve_state_and_output_and_clean_up_after_error():
    torch.manual_seed(3)
    model = ResidualDiT(52, 8, 6, 4, 12, 2, 3, .2).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    args = (torch.randn(2, 5, 52), torch.tensor([0., .5]), torch.randn(2, 5, 8),
            torch.randn(2, 6), torch.randn(2, 1), torch.randn(2, 4), torch.ones(2, 5, dtype=torch.bool))
    with torch.no_grad():
        expected = model(*args, condition_dropout=False)
        with ConditionCapture(model) as capture:
            output = model(*args, condition_dropout=False)
            recorded = capture.take()
            assert set(recorded['gates']) == {0, 1}
            assert recorded['attention'].shape == (2, 3, 5, 5)
            assert float(recorded['reconstruction_max_abs_error']) < 1e-6
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    with pytest.raises(RuntimeError, match='intentional'):
        with ConditionCapture(model):
            raise RuntimeError('intentional')
    assert not model.blocks[-1].cross_attention._forward_hooks
    assert all(not block.modulation._forward_hooks for block in model.blocks)


def test_uniform_attention_entropy_and_single_key_case():
    valid = torch.tensor([[True, True, False], [True, False, False]])
    weights = torch.zeros(2, 2, 3, 3)
    weights[0, :, :, :2] = .5
    weights[1, :, :, :1] = 1.
    entropy, norm = attention_entropy(weights, valid)
    torch.testing.assert_close(entropy, torch.tensor([2., 1.]).double().log())
    torch.testing.assert_close(norm, torch.tensor([1., 0.]).double())


def test_fit_selection_depends_on_metadata_not_tensor_values():
    clips = [f'mead_M0_happy_{i:03d}' for i in range(120)]
    query = {'clip_id': clips, 'sentence_id': [str(i % 11) for i in range(120)],
             'speaker_id': torch.zeros(120, dtype=torch.long), 'emotion_id': torch.ones(120, dtype=torch.long),
             'motion': torch.randn(120, 5, 52)}
    before = select_fit_metadata(query, clips)
    query['motion'].fill_(float('nan'))
    after = select_fit_metadata(query, clips)
    assert before == after and len(before) == 96
    assert len({row['index'] for row in before}) == 96
    with pytest.raises(ValueError, match='order'):
        select_fit_metadata(query, clips[::-1])


def test_condition_response_ignores_invalid_queries_keys_and_channels():
    valid = torch.tensor([[True, True, False]])
    channels = torch.ones(1, 52, dtype=torch.bool)
    values = {'attention': torch.zeros(1, 2, 3, 3), 'output': torch.zeros(1, 3, 4),
              'gates': {0: torch.ones(1, 4)}, 'velocity': torch.zeros(1, 3, 52)}
    other = {key: (item.clone() if torch.is_tensor(item) else {0: item[0].clone()}) for key, item in values.items()}
    other['attention'][:, :, 2, :] = 100
    other['attention'][:, :, :, 2] = 100
    other['output'][:, 2] = float('nan')
    other['velocity'][:, 2] = float('nan')
    response = condition_responses(values, other, valid, channels)
    assert all(torch.equal(value, torch.zeros_like(value)) for value in response.values())



def test_diagnostic_state_uses_valid_mask_like_real_flow_not_channel_mask():
    from types import SimpleNamespace
    system = SimpleNamespace(residual_scale=.25)
    motion = torch.ones(1, 3, 52)
    batch = {'q': {'motion': motion, 'valid': torch.tensor([[True, True, False]]),
        'channel_mask': torch.zeros(1, 52, dtype=torch.bool)},
        'base': {'b0': torch.zeros_like(motion)}, 'identity': {'baseline': torch.zeros(1, 52)}}
    noise = torch.zeros_like(motion)
    x, velocity = diagnostic_flow_state(system, batch, noise, torch.tensor([.5]))
    assert (x[:, :2] == 2).all()  # Channel mask must not remove input state.
    assert (velocity[:, :2] == 4).all()
    assert (x[:, 2] == 0).all()


def test_bias_free_local_and_time_centered_metrics_are_distinct():
    from types import SimpleNamespace
    torch.manual_seed(17)
    renderer = ResidualDiT(52, 8, 6, 4, 12, 2, 3, 0.).eval()
    system = SimpleNamespace(renderer=renderer)
    valid = torch.ones(1, 4, dtype=torch.bool)
    batch = {'q': {'valid': valid, 'channel_mask': torch.ones(1, 52, dtype=torch.bool)},
        'base': {'h0': torch.randn(1, 4, 8)}, 'identity': {'code': torch.randn(1, 4)}}
    fixed = {'global': torch.randn(1, 6), 'intensity_value': torch.ones(1, 1),
        'emotion_logits': torch.tensor([[1., 2.]])}
    conditions = {mode: {**fixed, 'local': torch.ones(1, 4, 6) if mode != 'zero' else torch.zeros(1, 4, 6)}
                  for mode in ('full', 'zero', 'reverse', 'oracle')}
    with ConditionCapture(renderer) as capture:
        records, differences, _ = run_conditions(system, batch, conditions, torch.zeros(1, 4, 52), torch.zeros(1), capture)
    assert records['zero']['local_embedding_with_bias_rms'][0] > 0
    assert records['zero']['local_bias_free_contribution_rms'][0] == 0
    assert records['full']['local_bias_free_contribution_rms'][0] > 0
    assert records['full']['local_temporally_centered_rms'][0] == pytest.approx(0, abs=1e-7)
    assert records['zero']['local_temporally_centered_rms'][0] == pytest.approx(0, abs=1e-7)



def test_four_conditions_share_identical_state_and_rollout_matches_decode():
    from types import SimpleNamespace
    torch.manual_seed(19)
    renderer = ResidualDiT(52, 8, 6, 4, 12, 1, 3, 0.).eval()
    system = SimpleNamespace(renderer=renderer)
    valid = torch.tensor([[True, True, True, False]])
    batch = {'q': {'valid': valid, 'channel_mask': torch.ones(1, 52, dtype=torch.bool)},
        'base': {'h0': torch.randn(1, 4, 8)}, 'identity': {'code': torch.randn(1, 4)}}
    fixed = {'global': torch.randn(1, 6), 'intensity_value': torch.ones(1, 1),
        'emotion_logits': torch.tensor([[1., 2.]]), 'local': torch.randn(1, 4, 6)}
    conditions = {mode: fixed for mode in ('full', 'zero', 'reverse', 'oracle')}
    noise = torch.randn(1, 4, 52)
    expected = renderer.decode(batch['base']['h0'], fixed['global'], fixed['intensity_value'],
        batch['identity']['code'], valid, residual_scale=.25, steps=12,
        local_emotion=fixed['local'], initial_noise=noise)
    seen = []
    handle = renderer.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    state = noise.clone()
    times = torch.linspace(0., 1., 13, dtype=noise.dtype)
    try:
        with ConditionCapture(renderer) as capture:
            for step in range(12):
                original = state.clone()
                _, _, velocity = run_conditions(system, batch, conditions, state,
                    torch.full((1,), times[step]), capture)
                assert all(torch.equal(value, original) for value in seen[-4:])
                assert torch.equal(state, original)
                state = (state + (times[step + 1] - times[step]) * velocity) * valid[..., None]
    finally:
        handle.remove()
    torch.testing.assert_close(state * .25, expected, rtol=0, atol=0)
