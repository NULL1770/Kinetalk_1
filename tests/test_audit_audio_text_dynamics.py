import copy
import json

import numpy as np
import pytest
import torch

from scripts import audit_audio_text_dynamics as audit
from scripts.audit_predictable_motion_predictions import clip_statistics, scalar_summary
from scripts.train_predictable_renderer import sha
from scripts.train_projection_schedule_ablation import curve_binding


def reference():
    motion=torch.tensor([.1,.4,.2,.3])[None,:,None].expand(4,4,52).clone()
    return {'q':{'motion':motion,'valid':torch.ones(4,4,dtype=torch.bool),
        'channel_mask':torch.ones(4,52,dtype=torch.bool),'times':torch.arange(4).float()[None].expand(4,-1)/25,
        'clip_id':['c0','c1','c2','c3'],'sentence_id':['s0','s1','s0','s1'],
        'speaker_id':torch.tensor([0,0,1,1]),'emotion_id':torch.tensor([0,1,0,1])},
        'base':{'b0':torch.zeros_like(motion)},'identity':{'baseline':torch.zeros(4,52)}}


def curves(ref,arm,initial=False):
    modes=('full','zero') if initial else audit.runner.MODES
    return {'schema':audit.SCHEMA,'arm':arm,'noise_seeds':audit.SEEDS.copy(),'decode_steps':12,
        'clip_id':ref['q']['clip_id'].copy(),'oracle_is_target_conditioned':True,
        'mismatched_text_indices':audit.runner.mismatch_indices(ref['q']),
        'motion':{str(seed):{mode:ref['q']['motion'].clone()+.01*j
            for mode in (modes if seed==42 else ('full','zero'))} for j,seed in enumerate(audit.SEEDS)},
        'intensity':{mode:{field:torch.ones(4,4,1) for field in ('predicted','driving')} for mode in modes}}


def arms_and_targets():
    ref=reference();arms={arm:{'epoch0':curves(ref,arm,True),'final':curves(ref,arm)} for arm in audit.ARMS}
    truth=torch.tensor([1.,3.,1.,3.])[None,:,None].expand(4,4,1).clone()
    targets={'splits':{role:{'intensity':truth.clone(),'valid':torch.ones_like(truth,dtype=torch.bool)}
        for role in ('train','validation')}}
    return arms,targets,ref


def test_saved_seed_modes_initial_eight_seeds_and_single_seed_interventions():
    ref=reference()
    for initial in (False,True):
        value=curves(ref,'text',initial)
        audit.validate_saved_curves(value,ref,'text',initial=initial)
        del value['motion']['997']
        with pytest.raises(ValueError,match='metadata'):
            audit.validate_saved_curves(value,ref,'text',initial=initial)
    value=curves(ref,'text');value['motion']['123']['oracle_intensity']=ref['q']['motion']
    with pytest.raises(ValueError,match='mode/seed'):
        audit.validate_saved_curves(value,ref,'text')


@pytest.mark.parametrize('bad',['mask_shape','negative_intensity','nan_motion','oracle_flag','mismatch'])
def test_saved_curve_corruptions_rejected(bad):
    ref=reference();value=curves(ref,'text')
    if bad=='mask_shape':value['intensity']['full']['predicted']=torch.zeros(4,4)
    if bad=='negative_intensity':value['intensity']['full']['driving'][0,0,0]=-1
    if bad=='nan_motion':value['motion']['42']['full'][0,0,41]=float('nan')
    if bad=='oracle_flag':value['oracle_is_target_conditioned']=False
    if bad=='mismatch':value['mismatched_text_indices'][0]=0
    with pytest.raises(ValueError):audit.validate_saved_curves(value,ref,'text')


def test_single_seed_statistics_are_not_zero_filled_or_averaged_over_eight():
    ref=reference();value=ref['q']['motion']+1
    stats,_,_=audit._condition_statistics({'42':{'one':value}},ref,[42])
    expected=clip_statistics(value,ref['q']['motion'],ref['q']['valid'].float(),audit.GROUPS['brows'])
    np.testing.assert_array_equal(stats['one','raw_motion','brows'],expected)
    with pytest.raises(ValueError,match='seed evidence'):
        audit._condition_statistics({'42':{'one':value}},ref,audit.SEEDS)


def test_intensity_clip_mean_oracle_fails_dynamics_despite_matching_dc():
    truth=torch.tensor([1.,3.,1.,3.])[None,:,None];valid=torch.ones_like(truth,dtype=torch.bool)
    static=audit.masked_intensity_mean(truth,valid)
    rows=audit.intensity_statistics(static,truth,valid)
    raw,centered=(scalar_summary(rows[k]) for k in ('raw_level','centered_dynamics'))
    assert raw['native_mse']==pytest.approx(1.)
    assert raw['prediction_rms_amplitude_ratio']>0.8
    assert centered['prediction_rms_amplitude_ratio']==0
    assert centered['r2_against_zero']==0
    wrong_dc=audit.intensity_statistics(truth+10,truth,valid)
    assert scalar_summary(wrong_dc['raw_level'])['native_mse']==100
    assert scalar_summary(wrong_dc['centered_dynamics'])['native_mse']==0


def test_intensity_mask_poison_ignored_and_observed_nonfinite_or_bad_mask_rejected():
    truth=torch.tensor([1.,3.,float('nan')])[None,:,None]
    valid=torch.tensor([True,True,False])[None,:,None]
    result=audit.intensity_statistics(truth,truth,valid)
    assert scalar_summary(result['centered_dynamics'])['native_mse']==0
    for mask in (valid[...,0],valid.float()):
        with pytest.raises(ValueError,match='mask must match'):audit.intensity_statistics(truth,truth,mask)
    valid[:]=True
    with pytest.raises(ValueError,match='Nonfinite observed'):audit.intensity_statistics(truth,truth,valid)


def test_domain_report_observed_clamp_empty_group_and_shape_checks():
    ref=reference();q=ref['q'];value=q['motion'].clone();value[0,0,41]=-0.5;value[0,1,41]=1.5
    q['valid'][0,3]=False;value[0,3]=float('nan')
    result=audit.domain_report({'42':{'full':value}},q,{'one':[0],'empty':[]})['full']
    assert result['one']['brows']['clamp_fraction']==pytest.approx(2/15)
    assert result['one']['brows']['mean_outside_magnitude']==pytest.approx(1/15)
    assert result['empty']['brows']['clamp_fraction'] is None
    assert result['empty']['brows']['observed_values_per_seed']==0
    json.dumps(result,allow_nan=False)
    with pytest.raises(ValueError,match='domain prediction'):
        audit.domain_report({'42':{'full':value[:1]}},q,{'one':[0]})


def test_intensity_report_separates_predictions_drives_levels_and_dynamics():
    arms,targets,ref=arms_and_targets()
    truth=targets['splits']['validation']['intensity']
    arms['text']['final']['intensity']['full']['predicted']=truth.clone()
    arms['text']['final']['intensity']['static_intensity']['predicted']=truth.clone()
    arms['text']['final']['intensity']['static_intensity']['driving']=torch.full_like(truth,2.)
    report=audit.intensity_report(arms,targets,ref,samples=20)
    compare=report['comparisons']['text/full_vs_target_clip_mean_oracle']['centered_dynamics']['all']
    assert compare['r2_improvement']==pytest.approx(1)
    assert compare['sentences']==2 and compare['clips']==4
    assert report['predicted_scores']['text/static_intensity']['centered_dynamics']['all']['native_mse']==0
    assert report['driving_scores']['text/static_intensity']['centered_dynamics']['all']['native_mse']==1
    assert 'speaker_0/nonneutral' in report['comparisons']['text/full_vs_no_text/full']['raw_level']
    json.dumps(report,allow_nan=False)


def test_analyze_keeps_epoch0_eight_seeds_and_interventions_single_seed():
    arms,targets,ref=arms_and_targets();report=audit.analyze(arms,targets,ref,samples=5)
    assert report['distribution']['sample_count']==8
    assert 'epoch0/full' in report['distribution']['modes']
    assert 'text/oracle_intensity' not in report['distribution']['modes']
    assert report['scope']['intervention_noise_seeds']==[42]
    assert 'text/full_vs_epoch0/full' in report['multi_seed_comparisons']
    assert 'distribution' not in report['single_seed_interventions']['text/full_vs_oracle_intensity']
    assert report['domain_single_seed']['text/full']['all']['brows']['noise_seeds']==1
    json.dumps(report,allow_nan=False)


def test_epoch0_common_all_seeds_both_arms_required():
    arms,_,_=arms_and_targets();assert audit.validate_common_initial(arms)
    arms['no_text']['epoch0']['motion']['997']['zero'][0,0,0]+=1
    with pytest.raises(ValueError,match='Epoch0'):audit.validate_common_initial(arms)


def test_curve_sidecar_authenticates_initial_checkpoint(tmp_path):
    curve=tmp_path/'epoch000_curves.pt';checkpoint=tmp_path/'epoch000.pt'
    curve.write_bytes(b'curves');checkpoint.write_bytes(b'checkpoint')
    recipe={'input_sha256':{'cache':'abc'}}
    curve_binding(curve,checkpoint,recipe,'abc')
    audit.validate_curve_binding(tmp_path,'epoch000_curves','epoch000.pt',recipe)
    checkpoint.write_bytes(b'modified')
    with pytest.raises(ValueError,match='sidecar binding'):
        audit.validate_curve_binding(tmp_path,'epoch000_curves','epoch000.pt',recipe)


def test_inventory_must_include_required_evidence(tmp_path):
    (tmp_path/'output_hashes.json').write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='inventory omits'):audit.load_arm(tmp_path,'text')


def test_target_audit_reference_order_independent_and_masks_bound(tmp_path):
    from scripts.prepare_expression_intensity_targets import prepare_targets
    native=tmp_path/'native';native.mkdir();enrollment=tmp_path/'enrollment.jsonl';cache_path=tmp_path/'cache.pt'
    ref=reference();ref['q']['speaker']=['a','a','b','b'];cache={'schema':'predictable_renderer_cache_v1','splits':{}}
    for role in ('train','validation'):
        value=copy.deepcopy(ref)
        for key in ('clip_id','sentence_id'):value['q'][key]=[role+s for s in value['q'][key]]
        cache['splits'][role]=value
    # Query clips must be globally unique and reference sentences independent.
    cache['splits']['train']['q']['clip_id']=['t'+str(i) for i in range(4)]
    torch.save(cache,cache_path);rows=[]
    for i,(speaker,sid) in enumerate((('b',1),('a',0),('b',1),('a',0))):
        path=native/f'r{i}.npz'
        np.savez(path,motion=np.full((3,52),.1),mask=np.ones(3,dtype=bool),channel_mask=np.ones(52,dtype=bool),
            provenance=np.asarray(json.dumps({'schema':'native_affect_style_v4','clock_evidence':'embedded_video'})))
        rows.append({'split':'train','emotion_id':0,'speaker':speaker,'speaker_id':sid,'clip_id':f'r{i}',
            'sentence':f'r{i}','artifact':path.name,'artifact_sha256':sha(path)})
    enrollment.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf8')
    target=prepare_targets(cache_path,enrollment,native,tmp_path/'out')
    assert audit.audit_target_values(target,cache)['all_target_values_rebuilt']
    target['splits']['validation']['anchor_valid'][0,41]=False
    with pytest.raises(ValueError,match='anchor mask'):audit.audit_target_values(target,cache)
