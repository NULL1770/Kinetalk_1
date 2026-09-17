"""Metadata selection and diagnostic-only exports on small synthetic inputs."""
from types import SimpleNamespace

import pytest
import torch

from scripts import train_context_mechanism as runner


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class MetadataOnly(dict):
    def __getitem__(self, key):
        if key not in ('clip_id', 'speaker_id', 'emotion_id'):
            raise AssertionError('Selection attempted to read target or features: '+key)
        return super().__getitem__(key)


def selection_source():
    rows = [(speaker, emotion, index) for speaker in (5, 2, 8, 1)
            for emotion in (3, 0, 2, 1) for index in range(12)]
    # Input ordering deliberately differs from speaker/emotion/clip sorting.
    rows = rows[::2][::-1]+rows[1::2]
    return MetadataOnly(clip_id=[f's{speaker}_e{emotion}_{index:03d}' for speaker, emotion, index in rows],
                        speaker_id=torch.tensor([row[0] for row in rows]),
                        emotion_id=torch.tensor([row[1] for row in rows]))


def test_fit_selection_128_is_unique_metadata_only_round_robin_and_order_stable():
    source = selection_source()
    indices, report = runner.fixed_fit_selection(source)
    assert len(indices) == len(set(indices.tolist())) == 128
    assert report['count'] == 128 and report['selection_uses_motion'] is False
    assert set(report['selected_group_counts'].values()) == {8}
    expected = [f's{s}_e{e}_{depth:03d}' for depth in range(8)
                for s in (1, 2, 5, 8) for e in range(4)]
    assert [row['clip_id'] for row in report['clips']] == expected
    assert [source['clip_id'][int(index)] for index in indices] == expected
    assert [row['original_fit_index'] for row in report['clips']] == indices.tolist()
    perm = torch.randperm(len(source['clip_id']), generator=torch.Generator().manual_seed(731))
    permuted = MetadataOnly(clip_id=[source['clip_id'][int(i)] for i in perm],
                             speaker_id=source['speaker_id'][perm], emotion_id=source['emotion_id'][perm])
    _, again = runner.fixed_fit_selection(permuted)
    assert [row['clip_id'] for row in again['clips']] == expected
    assert again['selected_group_counts'] == report['selected_group_counts']


def test_small_fit_selection_exhausts_uneven_groups_without_duplicates():
    source = MetadataOnly(clip_id=['a2', 'b0', 'a0', 'a1', 'c0'],
                          speaker_id=torch.tensor([0, 1, 0, 0, 3]), emotion_id=torch.zeros(5, dtype=torch.long))
    indices, report = runner.fixed_fit_selection(source)
    assert indices.tolist() == [2, 1, 4, 3, 0]
    assert report['count'] == 5
    assert report['selected_group_counts'] == {'(0, 0)': 3, '(1, 0)': 1, '(3, 0)': 1}


def diagnostic_fixture(monkeypatch):
    torch.manual_seed(812)
    valid = torch.ones(3, 96, dtype=torch.bool)
    valid[0, 15] = False
    valid[1, 33:39] = False
    valid[2, 80:] = False
    q = {'valid': valid, 'motion': torch.rand(3, 96, 52),
         'h0': torch.randn(3, 96, 4), 'prefix_local': torch.randn(3, 96, 3),
         'audio_global': torch.randn(3, 3), 'audio_intensity': torch.randn(3, 1),
         'static_upper': torch.rand(3, 9), 'clip_id': ['synthetic_a', 'synthetic_b', 'synthetic_c'],
         'speaker_id': torch.tensor([1, 2, 0]), 'emotion_id': torch.tensor([0, 1, 2]),
         'channel_mask': torch.ones(3, 52, dtype=torch.bool), 'b0': torch.zeros(3, 96, 52),
         'times': torch.arange(96, dtype=torch.float64)[None].expand(3, -1).clone()*.04}
    identities = {'code': torch.tensor([[0., 1.], [1., 1.], [2., 1.]]), 'baseline': torch.zeros(3, 52)}
    monkeypatch.setattr(runner.p.r, 'batch_identity',
                        lambda cache, b: {key: value[b['speaker_id']] for key, value in cache.items()})
    return q, identities, torch.ones(9), SimpleNamespace(device='cpu', batch_size=2)


class PositionDecoder:
    def __init__(self):
        self.calls = []

    def decode_prefix(self, valid, h0, identity, affect, local, noise, *, known, known_mask, steps):
        self.calls.append({'valid': valid.clone(), 'noise': noise.clone(), 'known_mask': known_mask.clone()})
        index = torch.arange(valid.shape[1], dtype=noise.dtype)[None, :, None]*.01
        value = noise+index
        return torch.where(known_mask[..., None], known, torch.where(valid[..., None], value, 0.))


def test_fit_oracle_is_explicitly_labeled_and_deploy_calls_receive_only_audio_conditions(monkeypatch):
    q, identities, scales, args = diagnostic_fixture(monkeypatch)
    calls = []

    def fake_rollout(upper, b, identity, native, noise, *, steps, mode, arm, oracle_target=None):
        assert set(b) == {'valid', 'h0', 'audio_global', 'audio_intensity'}
        assert arm == 'chunk_teacher'
        is_oracle = mode.startswith('oracle_')
        assert (oracle_target is not None) == is_oracle
        calls.append((mode, noise.clone(), None if oracle_target is None else oracle_target.clone()))
        value = oracle_target if is_oracle else noise
        return torch.where(b['valid'][..., None], value, 0.)

    monkeypatch.setattr(runner, 'decode_context', fake_rollout)
    report, curves = runner.fit_diagnostic(PositionDecoder(), q, identities, scales, args, 1, 'chunk_teacher')
    assert report['role'] == 'fixed_fit_reconstruction_diagnostic'
    assert report['clips'] == 3 and report['seed'] == 42 and report['test_loaded'] is False
    assert report['nonupper_scored'] is False and curves['nonupper_placeholders'] is True
    assert set(curves['predictions']) == {'42/'+mode for mode in runner.MODES}
    for mode, metrics in report['metrics'].items():
        assert metrics['GT_was_input'] == mode.startswith('oracle_')
        assert metrics['grid_is_decoder_boundary'] is True
        assert (metrics['actual_supplied_prefix'] is None) == (mode == 'empty')
    for prediction in curves['predictions'].values():
        assert prediction[..., list(runner.p.r.NOT_UPPER)].count_nonzero() == 0
    expected = torch.randn(3, 96, 9, generator=torch.Generator().manual_seed(42))
    for mode in runner.MODES:
        received = torch.cat([noise for current, noise, _ in calls if current == mode])
        torch.testing.assert_close(received, expected, atol=0, rtol=0)
    observed = q['valid'][..., None].expand(-1, -1, 9)
    for mode in ('oracle_history', 'oracle_reverse_history'):
        pred = curves['predictions']['42/'+mode][..., runner.p.CC]
        torch.testing.assert_close(pred[observed], q['motion'][..., runner.p.CC][observed])
    expected_pairs = ((q['valid'][:, 1:] & q['valid'][:, :-1]) & (torch.arange(1, 96).remainder(16) == 0)[None]).sum()
    assert report['adjacent_prefix_boundary_pairs'] == int(expected_pairs)


def test_position_probe_uses_same_noise_and_reports_native_position_shift_without_training(monkeypatch):
    q, identities, scales, args = diagnostic_fixture(monkeypatch)
    model = PositionDecoder()
    report, curves = runner.fit_diagnostic(model, q, identities, scales, args, 1, 'chunk_empty', position_probe=True)
    assert report['position0_vs8_same_weights_same_noise']['observed_upper_rms_difference'] == pytest.approx(.08, abs=1e-7)
    assert set(curves['predictions']) == {'42/full'}
    padded = [call for call in model.calls if call['valid'].shape[1] == 24]
    direct = [call for call in model.calls if call['valid'].shape[1] == 16]
    assert len(padded) == len(direct) and padded
    for left, right in zip(padded, direct):
        assert not left['valid'][:, :8].any() and not left['known_mask'].any()
        assert torch.equal(left['valid'][:, 8:], right['valid'])
        torch.testing.assert_close(left['noise'][:, 8:], right['noise'], atol=0, rtol=0)
    assert report['metrics']['full']['GT_was_input'] is False
    assert report['metrics']['full']['actual_supplied_prefix'] is None


def test_whole_window_fit_grid_is_not_reported_as_decoder_boundary(monkeypatch):
    q, identities, scales, args = diagnostic_fixture(monkeypatch)
    report, curves = runner.fit_diagnostic(PositionDecoder(), q, identities, scales, args, 1, 'whole')
    assert set(report['metrics']) == {'full'} and set(curves['predictions']) == {'42/full'}
    assert report['metrics']['full']['grid_is_decoder_boundary'] is False
    assert report['metrics']['full']['actual_supplied_prefix'] is None
    assert report['metrics']['full']['GT_was_input'] is False
    assert 'position0_vs8_same_weights_same_noise' not in report
    assert report['nonupper_scored'] is False and curves['nonupper_placeholders'] is True
