import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from kinetalk_b0.reference_mouth_calibration import MOUTH
from scripts import reference_decoder_data as loader
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_full_staged import subset


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(tmp_path):
    root = tmp_path/'prepared'; (root/'clips').mkdir(parents=True)
    cfg = {'data': {'motion_dim': 52, 'content_dim': 768, 'audio_dim': 83,
                   'audio_emotion_dim': 768, 'neutral_output_indices': list(MOUTH),
                   'emotion_classes': ['neutral', 'happy'], 'num_intensity_levels': 3},
           'model': {'content_dim': 8, 'emotion_dim': 8, 'style_dim': 8, 'hidden_dim': 8,
                     'affect_hidden_dim': 8, 'affect_rank': 4, 'affect_stride': 4,
                     'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.,
                     'residual_scale': .25, 'identity_bound': .5}}
    (root/'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf8')
    manifest = {'status': 'approved_train_val_only', 'sealed_test_targets_loaded': False,
                'reserved_sentences': [], 'roles': {r: {'query': [], 'enrollment': []} for r in ('train', 'val', 'test')}}
    records = []; generator = torch.Generator().manual_seed(41)
    for role, person in (('train', 'personZ'), ('val', 'personA'), ('test', 'personT')):
        for kind, count in (('query', 1), ('enrollment', 2)):
            for j in range(count):
                clip = f'{role}_{kind}_{j}'; frames = 8 if role == 'train' else 10
                row = {'clip_id': clip, 'speaker': person, 'sentence': f'{kind}_{j}',
                       'source_split': role, 'dataset': 'mead', 'emotion': 1 if kind == 'query' else 0,
                       'intensity': 1 if kind == 'query' else 0, 'artifact_sha256': 'native-hash',
                       'frames': frames, 'valid_frames': frames-1}
                manifest['roles'][role][kind].append(row)
                # Test has metadata only: it must never be prepared/opened.
                if role == 'test':
                    continue
                valid = torch.ones(frames, dtype=torch.bool); valid[-1] = False
                motion = torch.full((frames, 52), .2 + .2*j if kind == 'enrollment' else .9)
                saved = {'schema': loader.PREPARED_SCHEMA, 'row': row, 'valid': valid,
                         'times': torch.arange(frames, dtype=torch.float64)/25,
                         'channel_mask': torch.ones(52, dtype=torch.bool), 'motion': motion,
                         'content': torch.randn(frames, 768, generator=generator)*.01,
                         'audio': torch.zeros(frames, 83)}
                if kind == 'query':
                    saved.update(middle=torch.randn(frames, 768, generator=generator)*.01,
                                 prosody=torch.zeros(frames, 4))
                path = root/'clips'/(clip+'.pt'); torch.save(saved, path)
                records.append({'clip_id': clip, 'role': role, 'kind': kind, 'path': 'clips/'+path.name,
                                'sha256': loader.sha(path), 'frames': frames, 'valid_frames': frames-1})
    manifest['manifest_sha256'] = canonical_hash(manifest)
    (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    recipe = {'schema': loader.PREPARED_SCHEMA, 'manifest_sha256': manifest['manifest_sha256'],
              'config_sha256': loader.sha(root/'config.yaml'), 'test_loaded': False}
    (root/'index.json').write_text(json.dumps({'schema': loader.PREPARED_SCHEMA, 'recipe': recipe,
                                            'records': records, 'test_loaded': False}), encoding='utf8')
    source_cfg = copy.deepcopy(cfg)
    residual = [True]*52
    for c in MOUTH:
        residual[c] = False
    calibration = {'schema': 'reference_mouth_calibration_v1', 'channels': list(MOUTH),
                   'fit_scope': 'train_neutral_queries_and_independent_neutral_enrollment_only',
                   'development_used_for_fit': False, 'test_used_for_fit': False,
                   'observed_channels': [True]*27, 'gain': [1.]*27, 'bias': [0.]*27}
    source_cfg['model'].update(motion_support=[True]*52, residual_support=residual,
                               mouth_reference_calibration=calibration)
    torch.manual_seed(2)
    system = NeutralAffectSystem(source_cfg)
    audio = SlowStateAffect(torch.full((1540,), .7), torch.full((1540,), 1.3), hidden=8,
                           global_dim=8, local_dim=8, num_emotions=2, num_levels=3)
    source = tmp_path/'source.pt'
    torch.save({'system': system.state_dict(), 'audio': audio.state_dict(), 'config': source_cfg,
                'data_manifest_sha256': manifest['manifest_sha256'], 'test_loaded': False, 'stage': 'audio'}, source)
    return root, source


@pytest.mark.parametrize('role,forbidden', [('train', 'val'), ('validation', 'train')])
def test_load_opens_only_selected_role_and_uses_source_statistics(tmp_path, monkeypatch, role, forbidden):
    root, source = fixture(tmp_path)
    for path in (root/'clips').glob(forbidden+'_*.pt'):
        path.unlink()
    actual_load = torch.load; opened = []

    def tracked(path, *args, **kwargs):
        opened.append(Path(path).name)
        assert not Path(path).name.startswith(forbidden+'_')
        assert not Path(path).name.startswith('test_')
        return actual_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', tracked)
    data, system, audio, identities = loader.load_reference_context(root, source, 'cpu', role=role,
        batch_size=1, expected_source_sha256=loader.sha(source))
    assert set(data['splits']) == {role}
    q = data['splits'][role]
    assert q['speaker_id'].tolist() == ([1] if role == 'train' else [0])
    assert q['valid'].shape[1] == 10  # Padding matches legacy metadata convention.
    torch.testing.assert_close(q['anchors'], torch.full((1, 52), .3))
    assert q['anchor_valid'].all()
    for key in ('teacher_logits', 'target_mean', 'target_centered_state'):
        assert key not in q
    assert 'target_scales' not in data
    assert not any(p.requires_grad for module in (system, audio) for p in module.parameters())
    assert not system.training and not audio.training
    assert q['global_code'].shape == (1, 8) and q['frozen_local'].shape == (1, 10, 8)
    assert q['identity_baseline'].shape == (1, 52)
    assert q['b0'].shape == (1, 10, 52) and q['h0'].shape == (1, 10, 8)
    torch.testing.assert_close(data['feature_stats']['mean'], torch.full((1540,), .7))
    torch.testing.assert_close(data['feature_stats']['std'], torch.full((1540,), 1.3))
    assert data['provenance']['source_hash_verified']
    assert not data['provenance']['other_role_shards_opened']
    assert len(opened) == 4  # Three selected shards and one source checkpoint.
    assert subset(q, torch.tensor([0]), 'cpu')['identity_code'].shape == (1, 8)


def _resign_index(root, changed_path):
    index = json.loads((root/'index.json').read_text())
    for rec in index['records']:
        if (root/rec['path']) == changed_path:
            rec['sha256'] = loader.sha(changed_path)
    (root/'index.json').write_text(json.dumps(index))


def test_query_motion_cannot_change_frozen_deployment_conditions(tmp_path):
    root, source = fixture(tmp_path)
    before = loader.load_reference_context(root, source, 'cpu')[0]['splits']['train']
    path = root/'clips/train_query_0.pt'
    shard = torch.load(path, weights_only=False); shard['motion'].fill_(23.)
    torch.save(shard, path); _resign_index(root, path)
    after = loader.load_reference_context(root, source, 'cpu')[0]['splits']['train']
    assert not torch.equal(before['motion'], after['motion'])
    for key in ('anchors', 'anchor_valid', 'global_code', 'intensity_value', 'emotion_logits',
                'intensity_logits', 'identity_code', 'identity_baseline', 'frozen_local', 'b0', 'h0'):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


@pytest.mark.parametrize('failure', ['source_hash', 'source_manifest', 'shard_hash', 'mouth_calibration'])
def test_binding_and_calibration_fail_closed(tmp_path, failure):
    root, source = fixture(tmp_path)
    kwargs = {}
    if failure == 'source_hash':
        kwargs['expected_source_sha256'] = '0'*64
    elif failure in ('source_manifest', 'mouth_calibration'):
        saved = torch.load(source, weights_only=False)
        if failure == 'source_manifest':
            saved['data_manifest_sha256'] = 'wrong'
        else:
            del saved['config']['model']['mouth_reference_calibration']
        torch.save(saved, source)
    else:
        with (root/'clips/train_query_0.pt').open('ab') as f:
            f.write(b'tampered')
    with pytest.raises(ValueError):
        loader.load_reference_context(root, source, 'cpu', **kwargs)


def test_reference_query_sentence_overlap_rejected_before_shards(tmp_path, monkeypatch):
    root, source = fixture(tmp_path)
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['roles']['train']['enrollment'][0]['sentence'] = manifest['roles']['train']['query'][0]['sentence']
    manifest.pop('manifest_sha256'); manifest['manifest_sha256'] = canonical_hash(manifest)
    (root/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('No tensor should be opened'))
    with pytest.raises(ValueError, match='Query/reference'):
        loader.load_reference_context(root, source, 'cpu')


def test_test_role_rejected_before_any_metadata_or_tensors(tmp_path):
    with pytest.raises(ValueError, match='test tensor access'):
        loader.load_reference_context(tmp_path/'absent', tmp_path/'absent.pt', 'cpu', role='test')
