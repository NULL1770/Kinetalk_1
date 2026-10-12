from dataclasses import asdict
import torch
from kinetalk_v6.factor_paths import FactorPathKineTalkV6,distribution_losses
from tests.test_dual_timescale_v5 import _batch


def setup():
    parent,_,refs,batch,targets=_batch()
    model=FactorPathKineTalkV6.from_parent(dict(config=asdict(parent.cfg),model=parent.state_dict())).eval()
    return model,refs,batch,targets


def test_content_code_has_no_expression_style_event_input_or_joint_bypass():
    model,refs,b,_=setup();d=model.decoder
    assert not hasattr(d,'output') and not hasattr(d,'geometry_delta')
    with torch.no_grad():
        d.expression_response.weight.normal_(std=.02)
        d.content_response.weight.normal_(std=.02)
    state=torch.randn(2,19,10);emotion=torch.randn(2,8).softmax(-1)
    style=torch.randn(2,model.cfg.style_dim);events=torch.rand(2,19,12)
    args=(b['h0'],b['b0'],b['valid'])
    first=d(*args,state,emotion,torch.ones(2),style,events,model.scales)
    second=d(*args,state+1,emotion.flip(-1),torch.ones(2)*1.5,style+1,events+1,model.scales)
    assert torch.equal(first['content_code'],second['content_code'])
    assert torch.equal(first['content_response'],second['content_response'])
    assert not torch.equal(first['expression_response'],second['expression_response'])


def test_algebraic_zero_anchor_and_full_output_support():
    model,_,b,_=setup();d=model.decoder
    with torch.no_grad():
        d.expression_response.weight.normal_(std=.02);d.interaction_response.weight.normal_(std=.02)
    out=d(b['h0'],b['b0'],b['valid'],torch.zeros(2,19,10),torch.zeros(2,8),
        torch.zeros(2),torch.zeros(2,model.cfg.style_dim),torch.zeros(2,19,12),model.scales)
    assert torch.equal(out['expression_response'],torch.zeros_like(out['expression_response']))
    assert torch.equal(out['interaction_response'],torch.zeros_like(out['interaction_response']))
    assert d.expression_response.weight.shape[0]==56


def test_teacher_missing_in_inference_and_content_audio_isolation():
    model,refs,b,_=setup()
    model.motion_teacher=lambda _: (_ for _ in ()).throw(AssertionError('teacher used'))
    args=(b['h0'],b['b0'],b['valid'],refs)
    altered=b['audio_features'].clone();altered[...,:768]+=100
    first=model.predict(b['audio_features'],*args,times=b['times'])
    second=model.predict(altered,*args,times=b['times'])
    assert torch.equal(first['motion'],second['motion'])
    assert torch.equal(first['conditioned']['state'],second['conditioned']['state'])
    assert torch.equal(first['motion'][~b['valid']],torch.zeros_like(first['motion'][~b['valid']]))
    assert torch.equal(model.condition(first['student'],b['valid'])['state'],first['student']['state'])


def test_distribution_mask_is_nan_safe_with_separate_q_p_gradients():
    model,refs,b,targets=setup()
    with torch.no_grad():
        model.decoder.expression_response.weight.normal_(std=.02)
    out=model.training_forward(b,refs,targets)
    mask=targets['quality'].bool()&b['valid'][...,None]
    targets={**targets,'state':torch.where(mask,targets['state'],float('nan'))}
    losses=distribution_losses(model,out,targets,b)
    assert torch.isfinite(losses['total'])
    (losses['total']+(out['posterior_motion']-b['motion']).square().mean()).backward()
    for head in (model.student_logvar,model.teacher_logvar):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum()>0


def test_strict_checkpoint_roundtrip_and_seeded_correlated_state():
    from scripts.load_candidate_model import load_candidate_model
    model,refs,b,_=setup()
    loaded=load_candidate_model(dict(model=model.state_dict(),config=asdict(model.cfg))).eval()
    args=(b['audio_features'],b['h0'],b['b0'],b['valid'],refs)
    first=model.predict(*args,times=b['times'],seed=91)
    assert torch.equal(first['motion'],loaded.predict(*args,times=b['times'],seed=91)['motion'])
    other=model.predict(*args,times=b['times'],seed=92)
    assert not torch.equal(first['conditioned']['state'],other['conditioned']['state'])


def test_correlated_noise_respects_invalid_and_clock_boundaries():
    model,_,b,_=setup()
    valid=torch.tensor([[True,True,False,True,True]])
    prediction=dict(state=torch.zeros(1,5,10),logvar=torch.zeros(1,5,10))
    times=torch.tensor([[0.,.04,.08,1.,1.04]],dtype=torch.float64)
    g=torch.Generator().manual_seed(5)
    result=model.distribution_sample(prediction,valid,times,g)['state']
    raw=torch.randn(1,5,10,generator=torch.Generator().manual_seed(5))
    assert torch.allclose(result[:,0],(raw[:,0]+raw[:,1])/2**.5)
    assert torch.allclose(result[:,3],(raw[:,3]+raw[:,4])/2**.5)
    assert result[:,2].abs().sum()==0


def test_complete_objective_reaches_all_new_response_heads():
    import numpy as np
    from pathlib import Path
    from scripts.train_factor_paths_v6 import v6_objective
    model,refs,b,targets=setup();b['speaker_id']=torch.tensor([1,2])
    asset=Path(__file__).resolve().parents[1]/'final_experiment/rig/lip_constraints_v2_20261010.npz'
    with np.load(asset,allow_pickle=False) as z:channels=z['channels'].tolist()
    loss,_,logs=v6_objective(model,b,refs,targets,channels)
    loss.backward()
    assert {'posterior','prior_kl','content_anchor','reference_consistency','execution'}<=logs.keys()
    for head in (model.decoder.content_response,model.decoder.expression_response,
                 model.decoder.interaction_response,model.decoder.style_posture,model.student_logvar):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum()>0


def test_expression_and_style_can_change_mouth_without_changing_content_code():
    """Input isolation must not become a hard mouth-output exclusion."""
    import numpy as np
    from pathlib import Path
    model,_,b,_=setup();d=model.decoder
    asset=Path(__file__).resolve().parents[1]/'final_experiment/rig/lip_constraints_v2_20261010.npz'
    with np.load(asset,allow_pickle=False) as z:channels=z['channels'].tolist()
    mouth=[channels.index(n) for n in ('jawOpen','mouthSmileLeft','mouthStretchLeft','mouthPressLeft')]
    with torch.no_grad():
        d.expression_response.weight.normal_(std=.02)
        d.interaction_response.weight.normal_(std=.02)
        d.style_posture.weight.normal_(std=.02)
    state=torch.randn(2,19,10);emotion=torch.randn(2,8).softmax(-1)
    style=torch.randn(2,model.cfg.style_dim);events=torch.zeros(2,19,12)
    def forward(s,ref):
        return d(b['h0'],b['b0'],b['valid'],s,emotion,torch.ones(2),ref,events,model.scales)
    first=forward(state,style);affect=forward(state+1,style);reference=forward(state,style+1)
    assert torch.equal(first['content_code'],affect['content_code'])
    assert torch.equal(first['content_code'],reference['content_code'])
    for second in (affect,reference):
        changed=(second['motion']-first['motion'])[b['valid']][:,mouth].abs().sum(0)
        assert (changed>0).all()
