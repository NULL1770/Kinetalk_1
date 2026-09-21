import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.diagnose_regional_targets import (
    UPPER, canonical_hash, crop_diagnostics, digest, fit_ridge, load_train_shard,
    ordered_context, ridge_predict, run, split_cells, target_fields, train_inventory,
)


def test_target_semantics_and_crop_dependence_are_distinct():
    upper = torch.linspace(0., 1., 80, dtype=torch.float64)[:, None].expand(-1, 9)
    valid = torch.ones(80, dtype=torch.bool)
    anchor = torch.zeros(9, dtype=torch.float64); scales = torch.ones(9, dtype=torch.float64)
    fields = target_fields(upper, valid, anchor, scales)
    current = fields['clip_centered_rms'][0]
    neutral = fields['neutral_relative_rms'][0]
    assert current[0, 0] > current[40, 0]
    assert neutral[-1, 0] > neutral[0, 0]
    crop = crop_diagnostics(upper.square(), valid, anchor, scales)['targets']
    assert crop['clip_centered_rms']['relative_rmse'] > .1
    assert crop['neutral_relative_rms']['mse'] < 1e-20
    assert crop['velocity_rms_per_second']['mse'] < 1e-20


def test_velocity_and_context_do_not_cross_a_missing_frame():
    valid = torch.tensor([True, True, True, False, True, True, True])
    x = torch.tensor([1., 2., 3., float('nan'), 100., 101., 102.])[:, None]
    context = ordered_context(x, valid)
    assert context[2].max() == 3
    assert context[4].min() == 100
    fields = target_fields(x.expand(-1, 9), valid, torch.zeros(9), torch.ones(9))
    speed, mask = fields['velocity_rms_per_second']
    assert not mask[4]
    torch.testing.assert_close(speed[mask], torch.full_like(speed[mask], 25.))


def test_fold_matches_native_runner_and_reports_mixed_cells():
    from scripts.train_audio_regional_envelope import _fold
    rows = [{'clip_id': f'{s}_{t}', 'speaker': f'S{s}', 'sentence': f'T{t}'}
            for s in range(10) for t in range(8)]
    cells, metadata = split_cells(rows)
    fit, held, old = _fold({'speaker': [r['speaker'] for r in rows], 'sentence_id': [r['sentence'] for r in rows]})
    assert {rows[int(i)]['clip_id'] for i in fit} == {cid for cid, cell in cells.items() if cell == 'fit'}
    assert {rows[int(i)]['clip_id'] for i in held} == {cid for cid, cell in cells.items() if cell == 'new_speaker_new_sentence'}
    assert set(cells.values()) == {'fit', 'seen_speaker_new_sentence', 'new_speaker_seen_sentence', 'new_speaker_new_sentence'}
    assert metadata['held_speakers'] == old['held_speakers']


def test_fixed_ridge_learns_linear_relation_without_target_at_prediction():
    x = torch.linspace(-1., 1., 200)[:, None]
    y = torch.cat((2 * x + 1, -.5 * x + .2), -1)
    model = fit_ridge([(x, y)] * 30)
    prediction = ridge_predict(model, x)
    # Independently derive shrinkage: thirty equal clip weights and alpha10
    # give a 30/(30+10) slope; the intercept remains the observed target mean.
    expected = y.mean(0) + .75 * (y - y.mean(0))
    torch.testing.assert_close(prediction, expected.double(), atol=2e-7, rtol=2e-7)
    assert model['coefficient'].shape == (2, 2)


def prepared_fixture(root):
    root.mkdir()
    roles = {'train': {'query': [], 'enrollment': []},
             'val': {'query': [], 'enrollment': []}, 'test': {'query': [], 'enrollment': []}}
    index_records = []
    generator = torch.Generator().manual_seed(22)
    for speaker in range(6):
        for kind, numbers in [('query', range(8)), ('enrollment', range(2))]:
            for number in numbers:
                clip_id = f'S{speaker}_{kind}_{number}'
                frames = 24
                row = {'clip_id': clip_id, 'speaker': f'S{speaker}', 'sentence': f'T{number}' if kind == 'query' else f'R{number}',
                       'emotion': number % 4 if kind == 'query' else 0, 'source_split': 'train'}
                roles['train'][kind].append(row)
                valid = torch.ones(frames, dtype=torch.bool); valid[-1] = False
                motion = .3 + .05 * torch.randn(frames, 52, generator=generator)
                saved = {'row': row, 'motion': motion, 'valid': valid, 'channel_mask': torch.ones(52, dtype=torch.bool),
                         'times': torch.arange(frames, dtype=torch.float64) / 25,
                         'content': torch.randn(frames, 768, generator=generator),
                         'middle': torch.randn(frames, 768, generator=generator),
                         'prosody': torch.randn(frames, 4, generator=generator)}
                path = root / (clip_id + '.pt'); torch.save(saved, path)
                index_records.append({'clip_id': clip_id, 'role': 'train', 'kind': kind, 'path': path.name,
                                      'sha256': digest(path), 'frames': frames, 'valid_frames': int(valid.sum())})
    # These tensor paths intentionally do not exist. An overbroad loader fails.
    for role in ('val', 'test'):
        row = {'clip_id': role + '_forbidden', 'speaker': role, 'sentence': 'forbidden', 'source_split': role, 'emotion': 0}
        roles[role]['query'].append(row)
        index_records.append({'clip_id': row['clip_id'], 'role': role, 'kind': 'query', 'path': role + '_MUST_NOT_OPEN.pt',
                              'sha256': 'not-a-real-hash', 'frames': 24, 'valid_frames': 24})
    manifest = {'roles': roles, 'status': 'approved_train_val_only', 'sealed_test_targets_loaded': False}
    manifest['manifest_sha256'] = canonical_hash(manifest)
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    (root / 'index.json').write_text(json.dumps({'recipe': {'manifest_sha256': manifest['manifest_sha256']},
                                               'records': index_records, 'test_loaded': False}), encoding='utf8')


def test_end_to_end_reads_train_only_and_writes_comparable_curves(tmp_path, monkeypatch):
    data = tmp_path / 'prepared'; prepared_fixture(data)
    loaded = []
    original = torch.load
    def tracked(path, **kwargs):
        loaded.append(str(path))
        assert 'MUST_NOT_OPEN' not in str(path)
        return original(path, **kwargs)
    monkeypatch.setattr(torch, 'load', tracked)
    output = tmp_path / 'diagnostic'
    run(SimpleNamespace(data=data, output=output, device='cpu', pca_frames=48, example_clips=12, threads=2))
    result = json.loads((output / 'evaluation.json').read_text())
    assert result['protocol']['validation_tensors_loaded'] is False
    assert result['protocol']['test_tensors_loaded'] is False
    assert result['protocol']['ridge_alpha'] == 10
    assert len(result['examples']) == 12
    for source in ('prosody', 'pca_audio'):
        for cell, report in result['ridge'][source].items():
            assert report['conditions']['full']['clips'] > 0
            assert report['conditions']['trained_static']['temporal_std'] < 1e-12
            assert report['conditions']['trained_static']['correlation'] is None
    with np.load(output / result['examples'][0]['path'], allow_pickle=False) as curves:
        assert curves['clip_centered_rms'].shape[-1] == 2
        assert curves['pca_audio_full'].shape[-1] == 2
        assert 'velocity_rms_per_second' in curves
    assert loaded


def test_enrollment_overlap_and_forbidden_shard_are_rejected(tmp_path):
    root = tmp_path / 'prepared'; prepared_fixture(root)
    _, records, _ = train_inventory(root)
    forbidden = {**records[0], 'role': 'val'}
    with pytest.raises(ValueError, match='Only TRAIN'):
        load_train_shard(root, forbidden)
    path = root / 'manifest.json'; manifest = json.loads(path.read_text())
    manifest['roles']['train']['enrollment'][0]['sentence'] = 'T0'
    manifest.pop('manifest_sha256'); manifest['manifest_sha256'] = canonical_hash(manifest)
    path.write_text(json.dumps(manifest))
    index_path = root / 'index.json'; index = json.loads(index_path.read_text())
    index['recipe']['manifest_sha256'] = manifest['manifest_sha256']; index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='overlap'):
        train_inventory(root)
