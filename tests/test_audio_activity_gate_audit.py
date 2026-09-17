"""Contracts for gate auditing; synthetic tests are not quality evidence."""
import numpy as np
import pytest
import torch

from scripts.audit_audio_activity_gate import (
    activity_gate, aggregate_sentence_statistics, evaluate, intervention_controls,
    readout_saved_prediction,
    velocity_clip_statistics,
)
from scripts.train_predictable_renderer import center, reverse_controls
from scripts.audit_projection_readiness import relative_error_audit


def test_gate_obeys_config_order_and_never_backpropagates_classifier():
    logits = torch.tensor([[0., 6., 0.], [0., -6., 0.]], requires_grad=True)
    gate = activity_gate(logits, ['happy', 'neutral', 'sad'])
    expected = 1 - logits.detach().softmax(-1)[:, 1]
    torch.testing.assert_close(gate, expected)
    assert not gate.requires_grad and gate[0] < .01 and gate[1] > .99
    with pytest.raises(ValueError, match='configured'):
        activity_gate(logits, ['neutral', 'happy'])
    with pytest.raises(ValueError, match='uniquely'):
        activity_gate(logits, ['neutral', 'neutral', 'happy'])


def test_oracle_and_reverse_receive_identical_audio_gate_and_keep_padding():
    weight = torch.tensor([[4., 3., 0.], [4., 4., 1.]])
    audio = center(torch.arange(12.).reshape(2, 3, 2), weight)
    teacher = audio * 7
    gate = torch.tensor([.1, .9])
    full = intervention_controls('full', audio, teacher, weight, gate, .5)
    gated = intervention_controls('gated', audio, teacher, weight, gate, .5)
    oracle = intervention_controls('oracle_gated', audio, teacher, weight, gate, .5)
    reverse = intervention_controls('reverse_gated', audio, teacher, weight, gate, .5)
    torch.testing.assert_close(full, audio)
    torch.testing.assert_close(gated, audio * gate[:, None, None])
    torch.testing.assert_close(oracle, teacher * gate[:, None, None])
    torch.testing.assert_close(reverse, reverse_controls(audio, weight) * gate[:, None, None])
    assert gated[weight == 0].count_nonzero() == 0
    assert intervention_controls('zero', audio, teacher, weight, gate, .5).count_nonzero() == 0
    torch.testing.assert_close(intervention_controls('constant', audio, teacher, weight, gate, .5), audio * .5)


def test_sentence_statistics_cluster_clips_without_mixing_populations():
    statistics = np.arange(28, dtype=np.float64).reshape(4, 7)
    out = aggregate_sentence_statistics(statistics, ['a', 'a', 'b', 'a'],
                                        {'neutral': [0, 2], 'nonneutral': [1, 3]})
    np.testing.assert_array_equal(out['neutral']['a'], statistics[0])
    np.testing.assert_array_equal(out['neutral']['b'], statistics[2])
    np.testing.assert_array_equal(out['nonneutral']['a'], statistics[1] + statistics[3])


def test_evaluation_repeats_identical_noise_and_saves_all_interventions(tmp_path):
    class FakeHead:
        def __call__(self, features, weight):
            return features[..., :2]

        def teacher(self, motion, weight):
            return motion[..., :2]

    class FakeSystem:
        def project_affect(self, raw, mask):
            return {**raw, 'local': raw['controls'].repeat_interleave(4, dim=1)}

        def generate(self, content, valid, identity, affect, initial_noise, steps, base):
            assert steps == 12
            return {'motion': base['b0'] + affect['local'].sum(-1, keepdim=True) + initial_noise * .01}

    frames, clips = 8, 2
    motion = torch.ones(clips, frames, 52)
    split = {'q': {'motion': motion, 'content': torch.zeros(clips, frames, 2),
                  'valid': torch.ones(clips, frames, dtype=torch.bool),
                  'channel_mask': torch.ones(clips, 52, dtype=torch.bool)},
             'base': {'b0': torch.zeros_like(motion)}, 'identity': {'baseline': torch.zeros(clips, 52)},
             'affect': {'local': torch.zeros(clips, frames, 2), 'emotion_logits': torch.tensor([[4., 0.], [0., 4.]])}}
    features = torch.tensor([[[1., 2.], [3., 4.]], [[2., 3.], [4., 5.]]])
    bundle = {'features': {'content': features, 'middle': features, 'prosody': features},
              'weight': torch.full((clips, 2), 4.), 'motion_bins': torch.ones(clips, 2, 52)}
    stats, per_seed, rms, groups, modes = evaluate(FakeSystem(), FakeHead(), split, bundle,
        ['neutral', 'happy'], .5, device='cpu', batch_size=1, curves_dir=tmp_path)
    saved = torch.load(tmp_path / 'gate_seed42_curves.pt', weights_only=False)
    assert set(saved['motion']) == set(modes)
    assert {'full', 'zero', 'gated', 'constant', 'original', 'oracle_gated', 'reverse_gated'} == set(modes)
    prediction = saved['motion']
    # Paired noise cancels exactly up to float32 rounding; ungated local is 3 or 7.
    expected = features.sum(-1).repeat_interleave(4, dim=1)[..., None].expand(-1, -1, 52)
    torch.testing.assert_close(prediction['full'] - prediction['zero'], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(prediction['original'], prediction['zero'], atol=0, rtol=0)
    for kind in ('raw_motion', 'centered_motion', 'centered_residual'):
        expected_stats = np.stack([per_seed[str(s)]['gated'][kind]['upper_expression'] for s in (42, 123, 2026)]).mean(0)
        np.testing.assert_allclose(stats[('gated', kind, 'upper_expression')], expected_stats)
    assert np.all(rms['gated'] < rms['full'])


def test_teacher_readout_uses_observed_residual_and_original_labels():
    class FakeSystem:
        def motion_teacher(self, residual, valid):
            # Cached B0=3 and identity=2 must both be removed; invalid channels
            # and frames must be cleared despite having very large values.
            assert torch.equal(residual[..., 1], torch.zeros_like(residual[..., 1]))
            assert torch.equal(residual[:, 2], torch.zeros_like(residual[:, 2]))
            value = (residual[..., 0] * valid).sum(1)
            return {'emotion_logits': torch.stack([-value, value], -1)}

    motion = torch.zeros(2, 3, 2)
    valid = torch.tensor([[True, True, False], [True, True, False]])
    split = {'q': {'motion': motion, 'valid': valid, 'motion_valid': valid,
                   'channel_mask': torch.tensor([[True, False], [True, False]]),
                   'emotion_id': torch.tensor([1, 0])},
             'base': {'b0': torch.full_like(motion, 3)},
             'identity': {'baseline': torch.full((2, 2), 2.)}, 'affect': {}}
    prediction = torch.tensor([[[6., 100.], [7., 100.], [100., 100.]],
                               [[4., 100.], [3., 100.], [100., 100.]]])
    result = readout_saved_prediction(FakeSystem(), split, prediction, device='cpu', batch_size=1)
    assert result['accuracy'] == 1
    assert result['predicted_class_counts'] == [1, 1]
    assert result['clips'] == 2


def test_velocity_statistics_use_real_clock_and_ignore_invalid_adjacent_frames():
    split = {'q': {'motion': torch.zeros(1, 4, 2),
                   'valid': torch.tensor([[True, True, False, True]]),
                   'times': torch.tensor([[0., .5, 1., 1.5]])}}
    prediction = torch.tensor([[[0., 0.], [1., 2.], [1000., 1000.], [0., 0.]]])
    row = velocity_clip_statistics(prediction, split, [0, 1])
    # Only first adjacent pair is observed: squared velocity = 2² + 4².
    np.testing.assert_array_equal(row, [[20., 2.]])


def test_readiness_bootstrap_uses_ratio_of_sse_totals_and_paired_sentences():
    baseline = np.array([[1., 3.], [100., 3.], [100., 3.], [1., 3.]])
    candidate = np.array([[2., 3.], [100., 3.], [101., 3.], [1., 3.]])
    result = relative_error_audit(candidate, baseline, ['a', 'b', 'b', 'c'], [0, 1, 2, 3])
    assert result['relative_mse_increase'] == pytest.approx(204 / 202 - 1)
    assert result['sentences'] == 3
    assert result['one_sided_90_upper'] > result['relative_mse_increase']
    # Exactly proportional paired errors have deterministic relative CIs.
    proportional = baseline.copy(); proportional[:, 0] *= 1.02
    result = relative_error_audit(proportional, baseline, ['a', 'b', 'b', 'c'], [0, 1, 2, 3])
    assert result['one_sided_90_upper'] == pytest.approx(.02)
    np.testing.assert_allclose(result['ci95'], [.02, .02])
