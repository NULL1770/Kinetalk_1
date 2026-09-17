import copy
import json

import numpy as np
import pytest
import torch

from scripts.evaluate_neutral_affect_pilot import validate_evaluation_data, sha
from scripts.extract_emotion2vec_pilot import align_features, audio_statistics, MODEL_ID, verify_loaded_weights


def locked_raw(tmp_path):
    folder = tmp_path / 'audit'
    folder.mkdir()
    refs = {0: [{'sentence_id': 'enroll', 'clip_id': 'enroll'}]}
    stats = {'mean': torch.zeros(83), 'std': torch.ones(83)}
    train = {'queries': [{'sentence_id': 'train', 'clip_id': 'train'}],
             'identity_references': refs, 'audio_stats': stats}
    torch.save(train, folder / 'train.pt')
    sources = {'train_cache': sha(folder / 'train.pt'), 'development_cache': 'd' * 64}
    coverage = {'populated_cells': 28, 'expected_cells': 28}
    query = {'sentence_id': 'new', 'clip_id': 'new'}
    audit = {'queries': [query], 'identity_references': refs, 'audio_stats': stats,
             'audit_role': 'locked_sentence_audit', 'selection_source_sha256': sources,
             'audit_coverage': coverage, 'missing_cells': [], 'eval_scope': 'synthetic contract test'}
    torch.save(audit, folder / 'heldout.pt')
    (folder / 'selected_manifest.jsonl').write_text('{}\n')
    (folder / 'available.jsonl').write_text('{}\n')
    selection = {'schema': 'neutral_affect_locked_sentence_audit_v1', 'status': 'complete_locked_audit',
                 'source_sha256': sources, 'coverage': coverage, 'missing_cells': [], 'eval_scope': audit['eval_scope'],
                 'selected_clip_ids': ['new'], 'selected_sentence_ids': ['new'], 'query_count': 1,
                 'excluded_clip_ids': ['train', 'enroll', 'dev'], 'excluded_sentence_ids': ['train', 'enroll', 'dev'],
                 'isolation_checks': {key: True for key in (
                     'selected_sentence_disjoint_from_all_development_train_enrollment',
                     'selected_clip_disjoint_from_all_development_train_enrollment',
                     'copied_train_hash_unchanged', 'source_audio_statistics_reused')},
                 'output_sha256': {'train_cache': sources['train_cache'], 'audit_cache': sha(folder / 'heldout.pt'),
                                   'selected_manifest': sha(folder / 'selected_manifest.jsonl'),
                                   'available_manifest': sha(folder / 'available.jsonl')}}
    path = folder / 'selection.json'
    path.write_text(json.dumps(selection))
    checkpoint = {'provenance': {'train_sha256': sources['train_cache'], 'heldout_sha256': 'd' * 64}}
    return folder, path, checkpoint


def test_new_audit_requires_explicit_lock_and_matching_checkpoint_origin(tmp_path):
    folder, path, checkpoint = locked_raw(tmp_path)
    with pytest.raises(ValueError, match='hash differs'):
        validate_evaluation_data(checkpoint, folder)
    assert validate_evaluation_data(checkpoint, folder, path)['coverage']['populated_cells'] == 28
    wrong = copy.deepcopy(checkpoint)
    wrong['provenance']['heldout_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='sources differ'):
        validate_evaluation_data(wrong, folder, path)


def test_audit_rejects_changed_data_and_false_lock(tmp_path):
    folder, path, checkpoint = locked_raw(tmp_path)
    selection = json.loads(path.read_text())
    selection['status'] = 'selection_locked_before_materialization'
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='completed locked audit'):
        validate_evaluation_data(checkpoint, folder, path)
    selection['status'] = 'complete_locked_audit'
    path.write_text(json.dumps(selection))
    with (folder / 'heldout.pt').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        validate_evaluation_data(checkpoint, folder, path)


def test_emotion2vec_clock_interpolates_centers_and_rejects_outside_wave():
    geometry = {'sample_rate': 16000, 'receptive_samples': 400, 'hop_samples': 320}
    times = np.array([.01246875, .03246875, .05246875])
    features = np.arange(3, dtype=np.float32)[:, None].repeat(768, axis=1)
    targets = np.array([0., .02246875, .05246875, .09])
    valid = np.array([True, True, True, False])
    aligned, covered, extended = align_features(features, times, targets, valid, 1200, geometry)
    np.testing.assert_allclose(aligned[:, 0], [0., .5, 2., 0.])
    assert np.array_equal(covered | extended, valid)
    assert extended.tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match='outside'):
        align_features(features, times, targets, np.ones(4, bool), 1200, geometry)


def test_reused_normalization_preserves_train_order_and_tensors(tmp_path):
    keys = ('model_id', 'model_sha256', 'config_sha256', 'model_source_sha256',
            'audio_encoder_source_sha256', 'frontend_source_sha256')
    model = {key: key for key in keys}
    geometry = {'hop': 320}
    queries = [{'audio': torch.randn(3, 768), 'valid': torch.ones(3, dtype=torch.bool),
                'clip_id': clip, 'sentence_id': clip} for clip in ('z', 'a')]
    stats, _ = audio_statistics(queries, [], model, geometry)
    cache = {'split': 'train', 'queries': queries, 'audio_stats': stats,
             'audio_feature_provenance': {'model': model, 'geometry': geometry}}
    path = tmp_path / 'train.pt'
    torch.save(cache, path)
    reused, info = audio_statistics([], [{'clip_id': 'new', 'sentence_id': 'new'}], model, geometry, path)
    assert reused['fit_clip_ids'] == ['z', 'a']
    assert reused['feature_type'] == MODEL_ID
    assert torch.equal(reused['mean'], stats['mean'])
    assert info['current_extraction_used_to_fit'] is False
    with pytest.raises(ValueError, match='overlaps'):
        audio_statistics([], [{'clip_id': 'new', 'sentence_id': 'z'}], model, geometry, path)


def test_pretrained_weight_verification_rejects_any_unloaded_active_tensor(tmp_path):
    module = torch.nn.Linear(3, 2)
    path = tmp_path / 'model.pt'
    torch.save({'model': module.state_dict()}, path)
    assert verify_loaded_weights(module, path) == 2
    torch.save({'model': {'weight': module.weight.detach()}}, path)
    with pytest.raises(ValueError, match='missing'):
        verify_loaded_weights(module, path)
