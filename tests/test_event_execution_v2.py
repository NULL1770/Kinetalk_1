from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from kinetalk_v2.event_execution import EventExecutionConfig, event_execution_loss


def fixture():
    with np.load(Path(__file__).resolve().parents[1]/'final_experiment/rig/fixed_arkit_rig_20260923.npz') as rig:
        channels=[str(n) for n in rig['channels']]
    lookup={n.lower():i for i,n in enumerate(channels)}
    b,t=1,16
    valid=torch.ones(b,t,dtype=torch.bool);times=torch.arange(t,dtype=torch.float64)[None]/25
    mask=torch.ones(b,52,dtype=torch.bool);mask[:,-1]=False
    base=torch.full((b,t,52),.05)
    zero=base.clone();zero[...,lookup['mouthsmileleft']]=.45;zero[...,lookup['mouthsmileright']]=.45
    state=torch.zeros(b,t,10);state[...,2]=.10;state[...,3]=.05;state[...,5]=.08;state[...,1]=.08
    events=torch.zeros(b,t,4,3)
    amp=torch.tensor([.60,.12,.10,.12])
    events[:,3:11,:,0]=1.;events[:,3:11,:,1]=torch.arange(8)[:,None]/7;events[:,3:11,:,2]=amp
    generated=zero.clone()
    shape=.8+.2*torch.sin(torch.pi*events[...,1]);target=events[...,2]*shape
    for name in ('eyeBlinkLeft','eyeBlinkRight'):generated[...,lookup[name.lower()]]+=target[...,0]
    for name in ('browInnerUp','browOuterUpLeft','browOuterUpRight'):generated[...,lookup[name.lower()]]+=target[...,1]*4/3
    for name in ('browDownLeft','browDownRight'):generated[...,lookup[name.lower()]]+=target[...,1]*2/3
    for name in ('cheekPuff','cheekSquintLeft','cheekSquintRight','noseSneerLeft','noseSneerRight'):generated[...,lookup[name.lower()]]+=target[...,2]
    for name in ('mouthStretchLeft','mouthStretchRight','mouthPressLeft','mouthPressRight'):generated[...,lookup[name.lower()]]+=target[...,3]
    return generated,zero,events,state,base,valid,times,mask,channels


def call(items,**kw):
    return event_execution_loss(*items,**kw)


def test_real_local_execution_beats_ignored_schedule_without_gt_motion():
    data=fixture();correct=call(data)
    ignored=list(data);ignored[0]=data[1].clone();wrong=call(ignored)
    assert correct['total']<1e-6
    assert wrong['total']>.1
    assert all(v['eligible_events']==1 and v['quality_active_frames']==8 for v in correct['groups'].values())
    assert correct['observed_increment'].shape==(1,16,4)


def test_same_closure_baseline_cannot_impersonate_causal_event():
    data=list(fixture());frozen_quality=data[1].clone()
    data[1]=data[0].clone()  # eyelids already closed in no-event output
    result=call(data)
    assert result['groups']['eyelid_closure']['ambiguous_direction_events']==1
    assert not result['quality'][...,0].any()
    assert result['total']>.1
    fixed=call(data,quality_reference_motion=frozen_quality)
    assert fixed['groups']['eyelid_closure']['eligible_events']==1
    assert fixed['groups']['eyelid_closure']['active']>.1
    assert fixed['quality_reference_policy']=='frozen_external_zero_event_motion'


def test_gradients_only_reach_event_decoder_not_zero_reference_state_or_schedule():
    data=list(fixture());data[0]=data[1].clone()
    for i in (0,1,2,3,4):data[i]=data[i].clone().requires_grad_()
    report=call(data);report['total'].backward()
    assert data[0].grad is not None and data[0].grad.abs().sum()>0
    assert all(data[i].grad is None for i in (1,2,3,4))
    with np.load(Path(__file__).resolve().parents[1]/'final_experiment/rig/fixed_arkit_rig_20260923.npz') as rig:
        corner=list(rig['channels']).index('mouthSmileLeft')
    assert data[0].grad[...,corner].abs().sum()==0


def test_masked_gaps_and_clock_gaps_censor_events_without_cross_gap_phase_loss():
    data=list(fixture());data[0]=data[1].clone();data[5][:,7]=False
    data[0][:,7]=float('nan');data[1][:,7]=float('nan');data[4][:,7]=float('nan')
    data[0].requires_grad_();result=call(data)
    assert result['total']==0 and not result['quality'].any()
    assert all(v['censored_events']==2 for v in result['groups'].values())
    result['total'].backward();assert not data[0].grad.nan_to_num().any()
    data=list(fixture());data[6][:,7:]+=.04
    result=call(data)
    assert result['total']==0 and not result['quality'].any()


def test_missing_channels_and_ambiguous_direction_receive_zero_weight():
    data=list(fixture());lookup={n.lower():i for i,n in enumerate(data[-1])}
    data[7][:,lookup['browinnerup']]=False;data[0][...,lookup['browinnerup']]=float('nan')
    data[1][...,lookup['browinnerup']]=float('nan');data[4][...,lookup['browinnerup']]=float('nan')
    data[3][...,5]=0.;data[3][...,1]=0.
    report=call(data)
    assert report['groups']['brow_emphasis']['unobserved_events']==1
    assert report['groups']['cheek_nose_emphasis']['ambiguous_direction_events']==1
    assert report['groups']['lip_tension']['ambiguous_direction_events']==1
    assert not report['quality'][...,1:].any()


def test_lip_emotion_and_jaw_are_free_and_unmeasurable_amplitudes_are_not_inflated():
    data=list(fixture());report=call(data)
    lookup={n.lower():i for i,n in enumerate(data[-1])}
    altered=list(data);altered[0]=data[0].clone()
    altered[0][...,lookup['mouthsmileleft']]=.9;altered[0][...,lookup['jawopen']]=.8
    changed=call(altered)
    assert torch.equal(report['total'],changed['total'])
    data[2][:,3:11,0,2]=.05  # below the active eyelid weak-label range
    result=call(data)
    assert result['groups']['eyelid_closure']['out_of_range_events']==1
    assert not result['quality'][...,0].any()


def test_negative_brow_and_lip_state_direction_is_honored_and_unstable_sign_is_disabled():
    data=list(fixture());data[3][:,:,2:4]*=-1;data[3][:,:,1]*=-1
    # Mirror only event-induced brow/lip effects around the fixed baseline.
    lookup={n.lower():i for i,n in enumerate(data[-1])}
    ix=[lookup[n.lower()] for n in ('browInnerUp','browOuterUpLeft','browOuterUpRight','browDownLeft','browDownRight',
                                 'mouthStretchLeft','mouthStretchRight','mouthPressLeft','mouthPressRight')]
    data[0][...,ix]+=.30;data[1][...,ix]+=.30
    data[0][...,ix]=2*data[1][...,ix]-data[0][...,ix]
    assert call(data)['total']<1e-6
    data[3][:,3:7,1]=.08;data[3][:,7:11,1]=-.08
    result=call(data)
    assert result['groups']['lip_tension']['ambiguous_direction_events']==1
    assert not result['quality'][...,3].any()


def test_phase_execution_and_off_support_nuisance_are_local():
    data=list(fixture())
    lookup={n.lower():i for i,n in enumerate(data[-1])}
    ix=[lookup['eyeblinkleft'],lookup['eyeblinkright']]
    # Reverse a non-symmetric schedule waveform to create the wrong phase.
    data[2][:,3:11,0,1]=torch.linspace(0,.6,8)
    data[0][:,3:11,ix]=data[1][:,3:11,ix]+.6*torch.linspace(1.,.8,8)[None,:,None]
    result=call(data)
    assert result['groups']['eyelid_closure']['phase']>0
    # An unrelated eyebrow change far away from events is a nuisance delta.
    data=list(fixture());data[0][:,14,lookup['browinnerup']]+=.2
    assert call(data)['groups']['brow_emphasis']['off_support']>0


def test_zero_config_weights_keep_loss_differentiable_zero():
    data=list(fixture());data[0].requires_grad_()
    cfg=replace(EventExecutionConfig(),weights=(0.,0.,0.,0.))
    report=call(data,config=cfg);assert report['total']==0
    report['total'].backward();assert not data[0].grad.any()


def test_insufficient_physical_direction_headroom_gets_zero_quality():
    data=list(fixture());data[3][...,1]=-.08
    # A .05 standing tension cannot decrease by a .12 coefficient excursion.
    result=call(data)
    assert result['groups']['lip_tension']['insufficient_headroom_events']==1
    assert not result['quality'][...,3].any()


def test_real_decoder_can_learn_execution_with_frozen_conditioning_and_no_gt_l2():
    from kinetalk_v2.unified_model import UnifiedConfig,UnifiedExpressionModel
    torch.set_num_threads(2);torch.manual_seed(47)
    cfg=UnifiedConfig(hidden=16,decoder_hidden=24,content_dim=8)
    model=UnifiedExpressionModel(cfg,torch.zeros(772),torch.ones(772),torch.full((52,),.1))
    data=fixture();_,_,events,state,base,valid,times,mask,channels=data
    events=events.clone();events[:,:,1:]=0.  # isolate the identifiable blink stream
    content=torch.randn(1,16,8);style=torch.randn(1,cfg.style_dim);emotion=torch.ones(1,8)/8
    y_event=model.decode(content.detach(),base.detach(),valid,state.detach(),emotion.detach(),style.detach(),events.detach().flatten(-2))['motion']
    with torch.no_grad():
        y_zero=model.decode(content,base,valid,state,emotion,style,torch.zeros_like(events).flatten(-2))['motion']
    result=event_execution_loss(y_event,y_zero,events,state,base,valid,times,mask,channels)
    assert result['groups']['eyelid_closure']['eligible_events']==1
    result['total'].backward()
    assert model.decoder.output.weight.grad.abs().sum()>0
    for branch in (model.student,model.teacher,model.events,model.style):
        assert all(p.grad is None for p in branch.parameters())
