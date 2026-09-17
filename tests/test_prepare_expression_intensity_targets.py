import copy
import json

import numpy as np
import pytest
import torch

from scripts.prepare_expression_intensity_targets import prepare_targets, sha


def fixture(tmp_path):
    native = tmp_path/'native'; native.mkdir()
    rows = []
    for i, (level, length) in enumerate(((.1, 2), (.3, 7))):
        path = native/f'ref{i}.npz'
        motion = np.full((length, 52), level, dtype=np.float32)
        np.savez(path, motion=motion, mask=np.ones(length, dtype=bool), channel_mask=np.ones(52, dtype=bool),
            provenance=np.asarray(json.dumps({'schema': 'native_affect_style_v4.1', 'clock_evidence': 'embedded_video'})))
        rows.append({'clip_id': f'ref{i}', 'speaker': 'speaker_a', 'speaker_id': 7, 'sentence': f'ref_sentence{i}',
            'split': 'train', 'emotion': 0, 'emotion_id': 0, 'artifact': path.name, 'artifact_sha256': sha(path)})
    enrollment = tmp_path/'enrollment.jsonl'
    enrollment.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf8')
    cache = {'schema': 'predictable_renderer_cache_v1', 'provenance': {'marker': 'test'}, 'splits': {}}
    for name, value in (('train', .4), ('validation', .8)):
        motion = torch.full((1, 4, 52), value)
        cache['splits'][name] = {'q': {'motion': motion, 'valid': torch.ones(1, 4, dtype=torch.bool),
            'channel_mask': torch.ones(1, 52, dtype=torch.bool), 'clip_id': [name+'_query'],
            'speaker': ['speaker_a'], 'speaker_id': torch.tensor([7]), 'sentence_id': [name+'_sentence'],
            'emotion_id': torch.tensor([0])}}
    cache_path = tmp_path/'cache.pt'; torch.save(cache, cache_path)
    return cache_path, enrollment, native, cache, rows


def test_reference_median_equal_weights_fit_only_scales_and_bindings(tmp_path):
    cp, ep, native, cache, _ = fixture(tmp_path)
    result = prepare_targets(cp, ep, native, tmp_path/'out')
    torch.testing.assert_close(result['splits']['train']['anchors'], torch.full((1, 52), .2))
    torch.testing.assert_close(result['scales'], torch.full((52,), .2))
    torch.testing.assert_close(result['splits']['train']['intensity'], torch.ones(1, 4, 1))
    torch.testing.assert_close(result['splits']['validation']['intensity'], torch.full((1, 4, 1), 3.))
    assert result['splits']['validation']['valid'].all()
    p=json.loads((tmp_path/'out/provenance.json').read_text(encoding='utf8'))
    assert p['source_sha256']['cache'] == sha(cp) and p['output_sha256']['targets.pt'] == sha(tmp_path/'out/targets.pt')
    cache['splits']['validation']['q']['motion'][:] = 50; torch.save(cache, cp)
    other = prepare_targets(cp, ep, native, tmp_path/'other')
    torch.testing.assert_close(other['scales'], result['scales'], rtol=0, atol=0)
    torch.testing.assert_close(other['splits']['validation']['anchors'], result['splits']['validation']['anchors'], rtol=0, atol=0)


@pytest.mark.parametrize('bad', ['split','emotion','speaker','clip','sentence','duplicate','hash','path'])
def test_reference_scope_disjointness_identity_and_hash_contract(tmp_path, bad):
    cp, ep, native, _, rows = fixture(tmp_path)
    if bad=='split': rows[0]['split']='test'
    if bad=='emotion': rows[0]['emotion_id']=1
    if bad=='speaker': rows[0]['speaker_id']=8
    if bad=='clip': rows[0]['clip_id']='train_query'
    if bad=='sentence': rows[0]['sentence']='validation_sentence'
    if bad=='duplicate': rows[1]['sentence']=rows[0]['sentence']
    if bad=='hash': rows[0]['artifact_sha256']='0'*64
    if bad=='path': rows[0]['artifact']='../elsewhere.npz'
    ep.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
    with pytest.raises(ValueError): prepare_targets(cp,ep,native,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_mask_poison_ignored_observed_query_poison_rejected(tmp_path):
    cp, ep, native, cache, _ = fixture(tmp_path)
    q=cache['splits']['train']['q']; q['valid'][0,-1]=False; q['motion'][0,-1]=float('nan')
    torch.save(cache,cp); result=prepare_targets(cp,ep,native,tmp_path/'out')
    assert result['splits']['train']['intensity'][0,-1]==0
    assert not result['splits']['train']['valid'][0,-1]
    q['motion'][0,0,41]=float('nan');torch.save(cache,cp)
    with pytest.raises(ValueError,match='Nonfinite observed query'): prepare_targets(cp,ep,native,tmp_path/'bad')


def test_invalid_reference_channel_requires_min_refs_and_masks_target(tmp_path):
    cp,ep,native,_,rows=fixture(tmp_path)
    path=native/rows[0]['artifact']
    with np.load(path) as z: values={k:z[k] for k in z.files}
    values['channel_mask'][41]=False;values['motion'][:,41]=np.nan
    np.savez(path,**values);rows[0]['artifact_sha256']=sha(path)
    ep.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
    r=prepare_targets(cp,ep,native,tmp_path/'out')
    assert not r['splits']['train']['anchor_valid'][0,41]
    assert torch.isnan(r['splits']['train']['anchors'][0,41])
    assert r['splits']['train']['valid'].all()
    assert r['scales'][41]==pytest.approx(.02)


def test_reject_extra_cache_splits_and_missing_reference_coverage(tmp_path):
    cp,ep,native,cache,rows=fixture(tmp_path)
    cache['splits']['test']=copy.deepcopy(cache['splits']['validation']);torch.save(cache,cp)
    with pytest.raises(ValueError,match='only train and validation'): prepare_targets(cp,ep,native,tmp_path/'out')
    del cache['splits']['test'];torch.save(cache,cp)
    ep.write_text(json.dumps(rows[0]),encoding='utf8')
    with pytest.raises(ValueError,match='Too few'): prepare_targets(cp,ep,native,tmp_path/'other')
