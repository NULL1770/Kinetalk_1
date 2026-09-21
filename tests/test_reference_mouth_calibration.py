"""Reference-only neutral offsets preserve the protected B0 temporal signal."""
import copy

import pytest
import torch

from kinetalk_b0.reference_mouth_calibration import (
    MOUTH, fit_reference_mouth_calibration, apply_reference_mouth_calibration,
)
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(23)
    yield
    torch.set_num_threads(previous)


def example():
    ids = ['a', 'a', 'b', 'c']
    x = torch.tensor([.10, .10, .25, .40], dtype=torch.float64)[:, None].expand(-1, 27).clone()
    y = .35*x+.025
    binding = {'split': 'train', 'query_clip_ids': ['a_q0','a_q1','b_q0','c_q0'],
               'query_sentence_ids': ['query0','query1','query0','query0'],
               'reference_clip_ids': [['a_r0','a_r1']]*2+[['b_r0','b_r1'],['c_r0','c_r1']],
               'reference_sentence_ids': [['ref0','ref1']]*4}
    return y,x,ids,binding


def fitted():
    y,x,ids,binding=example()
    return fit_reference_mouth_calibration(y,x,ids,provenance=binding)


def model_config(calibration):
    residual = [True]*52
    for i in MOUTH: residual[i] = False
    return {'data': {'content_dim':8,'motion_dim':52,'neutral_output_indices':list(MOUTH),
                     'emotion_classes':['neutral','happy'],'num_intensity_levels':3,'audio_emotion_dim':5},
            'model': {'content_dim':8,'emotion_dim':8,'style_dim':8,'hidden_dim':8,
                      'heads':2,'dropout':0.,'dit_dim':8,'dit_depth':1,'affect_rank':3,
                      'residual_support':residual,'mouth_reference_calibration':calibration}}


def test_known_static_mapping_is_recovered_and_equal_speaker_weight_ignores_clip_count():
    y,x,ids,binding=example()
    fit=fit_reference_mouth_calibration(y,x,ids,ridge=1e-10,bias_ridge=1e-10,provenance=binding)
    assert torch.tensor(fit['gain']).mean().item() == pytest.approx(.35, abs=1e-6)
    assert torch.tensor(fit['bias']).mean().item() == pytest.approx(.025, abs=1e-6)
    # Give a's queries a different mean and replicate all a rows as additional
    # clips. It must not increase speaker a's total weight.
    y[:2] += .08
    first=fit_reference_mouth_calibration(y,x,ids,provenance=binding)
    duplicate=[0,1,0,1,0,1]
    changed=copy.deepcopy(binding)
    changed['query_clip_ids'] += [f'a_extra{i}' for i in range(len(duplicate))]
    for key in ('query_sentence_ids','reference_clip_ids','reference_sentence_ids'):
        changed[key] += [binding[key][i] for i in duplicate]
    second=fit_reference_mouth_calibration(torch.cat([y,y[duplicate]]),torch.cat([x,x[duplicate]]),
        ids+['a']*len(duplicate),provenance=changed)
    torch.testing.assert_close(torch.tensor(first['gain']),torch.tensor(second['gain']),rtol=0,atol=0)
    torch.testing.assert_close(torch.tensor(first['bias']),torch.tensor(second['bias']),rtol=0,atol=0)


@pytest.mark.parametrize('mutation',['same_clip','same_sentence','development','reference_changes'])
def test_independence_and_train_scope_are_enforced(mutation):
    y,x,ids,binding=example()
    if mutation=='same_clip':binding['reference_clip_ids'][0]=['a_q0']
    elif mutation=='same_sentence':binding['reference_sentence_ids'][0]=['query0']
    elif mutation=='development':binding['split']='validation'
    else:x[1,0]+=.1
    with pytest.raises(ValueError):
        fit_reference_mouth_calibration(y,x,ids,provenance=binding)


def test_independence_is_speaker_local_for_sentences():
    y,x,ids,binding=example()
    binding['reference_sentence_ids'][2]=['query1']  # Spoken by a, never queried for b.
    fit_reference_mouth_calibration(y,x,ids,provenance=binding)


def test_masked_values_do_not_fit_and_missing_mouth_channels_cannot_be_calibrated():
    y,x,ids,binding=example()
    mask=torch.ones_like(y,dtype=torch.bool);mask[0,0]=False
    expected=fit_reference_mouth_calibration(y,x,ids,query_mask=mask,provenance=binding)
    y[0,0]=float('nan')
    actual=fit_reference_mouth_calibration(y,x,ids,query_mask=mask,provenance=binding)
    assert actual['gain']==expected['gain'] and actual['bias']==expected['bias']
    mask[:,1]=False
    with pytest.raises(ValueError,match='Every calibrated mouth channel'):
        fit_reference_mouth_calibration(y,x,ids,query_mask=mask,provenance=binding)
    ref_mask=torch.ones_like(x,dtype=torch.bool);ref_mask[0,0]=False
    with pytest.raises(ValueError,match='absent'):
        apply_reference_mouth_calibration(x,actual,reference_mask=ref_mask)


def test_gain_bounds_are_constrained_solution_not_posthoc_output_clamping():
    _,x,ids,binding=example()
    for slope,want in ((-2.,0.),(3.,1.)):
        fit=fit_reference_mouth_calibration(slope*x+.03,x,ids,ridge=1e-10,bias_ridge=1e-10,provenance=binding)
        assert all(abs(v-want)<1e-6 for v in fit['gain'])
        # With the slope fixed at its boundary, solve the intercept from the
        # equal-speaker mean (.1,.25,.4), not the unequal-clip average.
        assert fit['bias'][0] == pytest.approx((slope-want)*.25+.03,abs=1e-8)


def test_model_uses_reference_mean_only_fixed_parameters_and_protects_b0_timing():
    calibration=fitted();cfg=model_config(calibration)
    system=NeutralAffectSystem(cfg).eval()
    refs=torch.randn(2,2,8,52)*.025
    mask=torch.ones(2,2,8,dtype=torch.bool);mask[:,1,6:]=False
    ref_channels=torch.ones(2,2,52,dtype=torch.bool)
    identity=system.encode_identity(refs,mask,reference_channel_mask=ref_channels)
    want=apply_reference_mouth_calibration(identity['neutral_mean'][...,list(MOUTH)],calibration,
        reference_mask=torch.ones(2,27,dtype=torch.bool))
    torch.testing.assert_close(identity['baseline'][...,list(MOUTH)],want,rtol=0,atol=0)
    # A learned identity-bias head update cannot alter the fixed mouth map.
    with torch.no_grad():system.identity_bias.weight.add_(.2);system.identity_bias.bias.add_(.2)
    updated=system.encode_identity(refs,mask,reference_channel_mask=ref_channels)
    torch.testing.assert_close(updated['baseline'][...,list(MOUTH)],want,rtol=0,atol=0)
    assert all('calibration' not in name for name,_ in system.named_parameters())
    assert all('calibration' not in name for name in system.state_dict())
    valid=torch.ones(2,8,dtype=torch.bool);content=torch.randn(2,8,8)
    affect=system.encode_audio(torch.randn(2,8,5),valid)
    base=system.base(content,valid)
    for seed in (42,123,2026):
        pred=system.generate(content,valid,updated,affect,
            torch.randn(2,8,52,generator=torch.Generator().manual_seed(seed)),steps=3,base=base)['motion']
        mouth=pred[...,list(MOUTH)];b0=base['b0'][...,list(MOUTH)]
        torch.testing.assert_close(mouth,b0+want[:,None],rtol=0,atol=0)
        # Mathematically exact temporal preservation, within floating roundoff.
        torch.testing.assert_close(torch.diff(mouth,dim=1),torch.diff(b0,dim=1),rtol=0,atol=2e-7)
    restored=NeutralAffectSystem(cfg).eval();restored.load_state_dict(system.state_dict(),strict=True)
    same=restored.encode_identity(refs,mask,reference_channel_mask=ref_channels)
    torch.testing.assert_close(same['baseline'],updated['baseline'],rtol=0,atol=0)


def test_model_rejects_unobserved_reference_mouth_and_unprotected_residual():
    cfg=model_config(fitted());system=NeutralAffectSystem(cfg)
    refs=torch.randn(1,2,5,52);mask=torch.ones(1,2,5,dtype=torch.bool)
    channels=torch.ones(1,2,52,dtype=torch.bool)
    with pytest.raises(ValueError,match='explicit Boolean'):
        system.encode_identity(refs,mask)
    channels[0,1,14]=False
    with pytest.raises(ValueError,match='absent mouth'):
        system.encode_identity(refs,mask,reference_channel_mask=channels)
    mask[:,1]=False;refs[:,1]=float('nan')
    assert torch.isfinite(system.encode_identity(refs,mask,reference_channel_mask=channels)['baseline']).all()
    cfg['model']['residual_support'][14]=True
    with pytest.raises(ValueError,match='protected mouth'):
        NeutralAffectSystem(cfg)
