import torch
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig
from kinetalk_b0.models.static_expression import StaticExpressionCorrection
from scripts.evaluate_static_expression import StaticEvaluationModel
from scripts.train_expression_response import state_digest


@torch.no_grad()
def test_evaluator_matches_real_deployment_and_keeps_parent_oracle():
    torch.manual_seed(13);torch.set_num_threads(2)
    parent=ExpressionResponse(ResponseConfig(hidden=16,decoder_hidden=24),torch.zeros(772),torch.ones(772),torch.ones(52)).eval()
    b,t=2,12
    refs=dict(motion=torch.rand(b,2,t,52),b0=torch.rand(b,2,t,52),
        valid=torch.ones(b,2,t,dtype=torch.bool),channel_mask=torch.ones(b,2,52,dtype=torch.bool))
    audio=torch.rand(b,t,772);base=torch.rand(b,t,52);valid=torch.ones(b,t,dtype=torch.bool)
    times=torch.arange(t,dtype=torch.float64)[None].expand(b,-1)/25
    channels=torch.ones(b,52,dtype=torch.bool);gt=torch.rand(b,t,52)
    before=state_digest(parent)
    for mode,dim in [('latent',96),('reference',564)]:
        c=StaticExpressionCorrection(mode,torch.zeros(dim),torch.ones(dim),torch.rand(dim,52)*.01,torch.zeros(52))
        m=StaticEvaluationModel(parent,c).eval();s=m.encode_style(refs)['code'];p=m.audio_prior(audio,valid)
        torch.testing.assert_close(m.decode(base,p,s,valid),c.predict(parent,audio,base,valid,refs),rtol=0,atol=0)
        q=m.motion_posterior(gt,base,s,valid,channels,times)
        expected=parent.decode(base,q,s[:,:64],valid)
        torch.testing.assert_close(m.decode(base,q,s,valid),expected,rtol=0,atol=0)
        # The evaluator's wrong-audio control slices explicit context by query.
        one={k:v[:1] for k,v in p.items()}
        a=m.decode(base[:1],one,s[:1],valid[:1])
        r={k:v[:1] for k,v in refs.items()}
        torch.testing.assert_close(a,c.predict(parent,audio[:1],base[:1],valid[:1],r),atol=2e-6,rtol=1e-5)
    assert state_digest(parent)==before
