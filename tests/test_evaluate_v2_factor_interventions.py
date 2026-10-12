import numpy as np
import pytest
import torch

from scripts.evaluate_v2_factor_interventions import (balanced_validation_ids,content_change,
    descriptor_delta,directional_control,donor_speaker,emotion_indices,motion_descriptors,
    per_query_seed,predict_variants,schedule_change,validate_reference)


def names():
    from pathlib import Path
    with np.load(Path(__file__).resolve().parents[1]/'final_experiment/rig/fixed_arkit_rig_20260923.npz') as rig:
        return [str(s) for s in rig['channels']]


def test_selection_round_robin_emotion_and_people_not_first_speaker():
    q={'_lengths':torch.ones(120,dtype=torch.long),'speaker_id':torch.tensor([s for s in range(5) for e in range(8) for i in range(3)]),
       'emotion_id':torch.tensor([e for s in range(5) for e in range(8) for i in range(3)])}
    ids=balanced_validation_ids(q,32,47)
    assert len(ids)==32 and len(set(ids))==32
    assert len({int(q['speaker_id'][i]) for i in ids})==5
    assert all(sum(int(q['emotion_id'][i])==e for i in ids)==4 for e in range(8))
    assert ids==balanced_validation_ids(q,32,47)


def test_reference_independence_seed_and_emotion_aliases():
    ref={'clip_id':['ref_a','ref_b'],'sentence_id':['r1','r2'],'valid':torch.ones(2,5,dtype=torch.bool)}
    validate_reference('q','q1',ref)
    with pytest.raises(ValueError):validate_reference('ref_a','q1',ref)
    with pytest.raises(ValueError):validate_reference('q','r1',ref)
    assert donor_speaker(1,[1,2,3],47,'q')!=1
    assert per_query_seed('q',47)==per_query_seed('q',47)
    assert emotion_indices(['neutral','happy','sad','angry'])=={'happy':1,'sad':2,'angry':3}


def test_content_amplitude_can_change_without_being_declared_failure():
    ns=names();x=np.zeros((20,52));y=x.copy();jaw=ns.index('jawOpen');signal=.2+.1*np.sin(np.arange(20))
    x[:,jaw]=signal;y[:,jaw]=signal*1.2
    score=content_change(x,y,np.ones(20,bool),np.arange(20)/25,np.ones(52,bool),ns)
    assert score['jaw_zero_lag_correlation']==pytest.approx(1.)
    assert score['jaw_range_ratio']==pytest.approx(1.2)
    assert score['mouth_coefficient_pairwise_mae']>0
    assert 'disentanglement_passed' not in score


def test_range_ratio_with_only_outlier_is_undefined_not_infinite():
    ns=names();x=np.zeros((100,52));y=x.copy();jaw=ns.index('jawOpen')
    x[-1,jaw]=1.;y[-1,jaw]=.5
    score=content_change(x,y,np.ones(100,bool),np.arange(100)/25,np.ones(52,bool),ns)
    assert score['jaw_range_ratio'] is None


def test_descriptors_and_signed_control_use_real_motion_and_missing_masks():
    ns=names();base=np.zeros((10,52));happy=base.copy();sad=base.copy()
    happy[:,ns.index('mouthSmileLeft')]=.4;happy[:,ns.index('mouthSmileRight')]=.2
    sad[:,ns.index('mouthFrownLeft')]=.3;sad[:,ns.index('mouthFrownRight')]=.3
    mask=np.ones(52,bool);v=np.ones(10,bool);t=np.arange(10)/25
    dh=motion_descriptors(happy,base,v,t,mask,ns);ds=motion_descriptors(sad,base,v,t,mask,ns)
    assert dh['axes']['smile_minus_frown']['mean']==pytest.approx(.3)
    assert dh['asymmetry']['mouthsmile']['signed_mean']==pytest.approx(.2)
    d=descriptor_delta(ds,dh)
    assert directional_control(d,'happy')['smile_minus_frown_increase']==pytest.approx(.6)
    mask[ns.index('mouthSmileLeft')]=False
    assert not motion_descriptors(happy,base,v,t,mask,ns)['axes']['smile_minus_frown']['observed']


def test_sample_schedule_change_does_not_pretend_seed_fixed_events():
    a=np.zeros((5,12));b=a.copy();b[2,0]=1
    score=schedule_change(a,b,np.ones(5,bool))
    assert not score['events_bitwise_equal'] and score['active_group_disagreement_fraction']>0
    assert score['same_seed_does_not_guarantee_same_conditional_event_schedule']


def test_real_model_predict_interventions_no_teacher_inputs_or_mutation():
    from kinetalk_v2.unified_model import UnifiedConfig,UnifiedExpressionModel
    from scripts.train_expression_response import state_digest
    torch.set_num_threads(2);torch.manual_seed(47)
    cfg=UnifiedConfig(content_dim=8,hidden=16,decoder_hidden=24)
    model=UnifiedExpressionModel(cfg,torch.zeros(772),torch.ones(772),torch.ones(52)*.1).eval()
    valid=torch.ones(1,12,dtype=torch.bool);valid[:,6]=False
    batch={'audio_features':torch.randn(1,12,1540),'h0':torch.randn(1,12,8),
           'b0':torch.zeros(1,12,52),'valid':valid,'times':torch.arange(12,dtype=torch.float64)[None]/25}
    ra={'motion':torch.rand(1,12,52),'valid':torch.ones_like(valid),'channel_mask':torch.ones(1,52,dtype=torch.bool)}
    rb={**ra,'motion':torch.rand(1,12,52)}
    before=state_digest(model)
    result=predict_variants(model,batch,ra,rb,{'happy':1,'sad':2,'angry':3},seed=91)
    assert len(result)==11 and state_digest(model)==before
    for out in result.values():
        assert torch.equal(out['student']['state'],result['auto_A']['student']['state'])
        assert torch.isfinite(out['motion']).all()
    assert torch.equal(result['happy_A']['schedule']['events'],result['auto_A']['schedule']['events'])
    assert not torch.equal(result['happy_A']['motion'],result['auto_A']['motion'])
    assert torch.equal(result['inference_sensitivity_no_events']['schedule']['events'],torch.zeros(1,12,12))
