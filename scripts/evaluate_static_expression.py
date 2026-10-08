"""Reuse the unchanged native evaluator for a frozen-parent static response.

The oracle diagnostic remains the original parent oracle; only deployment prior
outputs and their prior interventions receive the TRAIN-fitted mean correction.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.static_expression import (
    StaticExpressionCorrection,neutral_reference_mean)
from scripts.refine_expression_prior import restore_model
from scripts.train_expression_response import configure,load_runtime,cache_base,reference_batch,sha,write
from scripts.evaluate_expression_response import evaluate


class StaticEvaluationModel(nn.Module):
    """Explicit context transports the neutral mean alongside the frozen style.

    No encoder/decoder parameter is changed. Context slicing also works for the
    original evaluator's wrong-audio and one-reference diagnostic controls.
    """
    def __init__(self,parent,correction):
        super().__init__();self.parent=parent;self.correction=correction

    @property
    def cfg(self):return self.parent.cfg

    @property
    def scales(self):return self.parent.scales

    @property
    def emotion_head(self):return self.parent.emotion_head

    def audio_prior(self,audio,valid):return self.parent.audio_prior(audio,valid)

    def encode_style(self,refs):
        result=self.parent.encode_style(refs)
        return dict(result,code=torch.cat((result['code'],neutral_reference_mean(refs,self.scales)),-1))

    def motion_posterior(self,motion,base,style,valid,channels,times):
        result=self.parent.motion_posterior(motion,base,style[:,:self.cfg.style_dim],valid,channels,times)
        return dict(result,_parent_oracle=True)

    def decode(self,base,distribution,style,valid,sample=False,generator=None):
        if style.shape[-1]!=self.cfg.style_dim+52:
            raise ValueError('Explicit style+neutral reference context required')
        code,neutral=style.split((self.cfg.style_dim,52),-1)
        prediction=self.parent.decode(base,distribution,code,valid,sample,generator)
        if distribution.get('_parent_oracle',False):return prediction
        g=distribution['g_mean'].detach()
        features=torch.cat((g,code),-1)
        if self.correction.mode=='reference':
            probability=self.emotion_head(g).softmax(-1).detach()
            features=torch.cat((features,neutral,(probability[...,None]*neutral[:,None]).flatten(1)),-1)
        return self.correction(prediction,features,valid,self.scales)


@torch.no_grad()
def run(a):
    configure(47);device=torch.device(a.device)
    binding=json.loads(Path(a.binding).read_text(encoding='utf-8-sig'))
    for n,h in binding['source_files'].items():
        assert sha(Path(__file__).resolve().parents[1]/n)==h,n
    ck=torch.load(a.correction,map_location=device,weights_only=False)
    assert ck['parent_checkpoint_sha256']==binding['parent_checkpoint']['sha256']
    assert ck['data_manifest_sha256']==binding['data_manifest_sha256'] and not ck['test_loaded']
    parent,_=restore_model(binding,device);parent.eval().requires_grad_(False)
    correction=StaticExpressionCorrection(ck['mode'],**ck['state']).to(device)
    model=StaticEvaluationModel(parent,correction).eval().requires_grad_(False)
    data,base=load_runtime(binding,device)
    n=a.limit or len(data['splits']['validation']['valid'])
    cache_base(data,base,device,{'train':[],'validation':list(range(n))})
    # Compare the evaluator adapter to the actual deployment API on real clips.
    b=data['splits']['validation'].batch(torch.arange(min(n,16)),device)
    refs=reference_batch(data,b,device);p=model.audio_prior(b['audio_features'],b['valid'])
    actual=model.decode(b['b0'],p,model.encode_style(refs)['code'],b['valid'])
    expected=correction.predict(parent,b['audio_features'],b['b0'],b['valid'],refs)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    report=evaluate(binding,a.parent_run,a.output,a.device,a.limit,runtime=(model,data,base))
    receipt=dict(schema='phase47_static_response_evaluation_v1',
        parent_checkpoint_sha256=report['checkpoint_sha256'],correction_sha256=sha(a.correction),
        correction_mode=ck['mode'],fit_clips=ck['fit_clips'],ridge_strength=ck['strength'],
        evaluator_sha256=sha(Path(__file__).with_name('evaluate_expression_response.py')),
        deployment_api_equal=True,oracle='unchanged parent posterior, diagnostic only',
        raw_dynamic_trajectory='unchanged parent centered trajectory, constant mean correction',
        clipping='original clip_all; range/boundary effects still evaluated',
        test_loaded=False,default_replaced=False)
    write(Path(a.output)/'static_response_receipt.json',receipt)
    # Bind the extra receipt as well without editing the original report.
    mp=Path(a.output)/'manifest.json';m=json.loads(mp.read_text())
    p=Path(a.output)/'static_response_receipt.json'
    m['files'][p.name]=dict(size=p.stat().st_size,sha256=sha(p));write(mp,m)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--correction',required=True)
    p.add_argument('--parent-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int)
    run(p.parse_args())
