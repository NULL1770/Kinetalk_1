"""Controlled full-audio versus PCA24 ridge experiment on the current protocol.

Same signed four-state target and protected Stage4 renderer. Only representation
and ridge strength are selected, on TRAIN speaker/sentence holdout. No learned
generative prior or development-based gain/epoch selection is introduced.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.relative_audio_timing import relative_features, masked_center
from kinetalk_b0.models.slow_state_affect import masked_slow_state, lift_slow_state
from scripts.train_isolated_audio_state import load_context, state_metrics
from scripts.train_relative_audio_timing import final_evaluation, fit_statistics, set_status
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, canonical_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.train_predictable_renderer import state_hash

ALPHAS = (.01, .1, 1., 10., 100.)


class SupervisedAudioReadout(nn.Module):
    def __init__(self, mean, std, coefficient, scales):
        super().__init__()
        for k, v in [('mean', mean), ('std', std), ('coefficient', coefficient), ('scales', scales)]:
            self.register_buffer(k, v.detach().float().clone())

    def forward(self, features, valid, mode='audio'):
        if mode not in ('audio', 'static', 'reverse'): raise ValueError(mode)
        x = relative_features(features, valid, self.mean, self.std)
        state = masked_center(masked_slow_state(x @ self.coefficient, valid, stride=16)['state'], valid)
        if mode == 'static': state = torch.zeros_like(state)
        if mode == 'reverse':
            state = state.clone()
            for i in range(len(state)):
                ix = valid[i].nonzero().flatten()
                state[i, ix] = state[i, ix.flip(0)].clone()
        return {'state': state, 'delta': lift_slow_state(state, self.scales)}


@torch.no_grad()
def sufficient_statistics(q, ids, mean, std):
    """Clip-weighted normal equations of P(center(audio)); no dev statistics."""
    gram = torch.zeros(1540, 1540, device='cuda', dtype=torch.float64)
    rhs = torch.zeros(1540, 4, device='cuda', dtype=torch.float64)
    mean, std = mean.cuda(), std.cuda()
    for ix in ids.split(16):
        valid = q['valid'][ix].cuda()
        end = int(valid.any(0).nonzero()[-1]) + 1
        valid = valid[:, :end]
        x = relative_features(q['audio_features'][ix, :end].cuda(), valid, mean, std)
        # P is linear and channel-separable.  Project the real feature block
        # directly; constructing an identity [B,T,T] would make the helper's
        # [B,D,T,K] Gram tensor cubic in sequence length and can OOM.
        x = masked_center(masked_slow_state(x, valid, stride=16)['state'], valid)
        y = q['target_centered_state'][ix, :end].cuda()
        weight = (valid.float() / valid.sum(1)[:, None]).sqrt()[..., None]
        x = (x * weight).reshape(-1, 1540)
        y = (y * weight).reshape(-1, 4)
        gram += (x.T @ x).double()
        rhs += (x.T @ y).double()
    return gram / len(ids), rhs / len(ids)


def solve_path(gram, rhs, projection=None):
    transform = torch.eye(1540, device='cuda', dtype=torch.float64)
    if projection is not None:
        transform = torch.zeros(1540, 28, device='cuda', dtype=torch.float64)
        transform[:1536, :24] = projection.double().cuda()
        transform[-4:, -4:] = torch.eye(4, device='cuda', dtype=torch.float64)
    g = transform.T @ gram @ transform
    h = transform.T @ rhs
    scale = g.diag().clamp_min(1e-8).sqrt()
    g = g / scale[:, None] / scale[None]
    h = h / scale[:, None]
    eig, vec = torch.linalg.eigh((g + g.T) / 2)
    return {alpha: (transform @ ((vec @ ((vec.T @ h) / (eig.clamp_min(0)[:, None] + alpha))) / scale[:, None])).float()
            for alpha in ALPHAS}


@torch.no_grad()
def score(q, ids, model):
    values=[]
    for ix in ids.split(32):
        values.append(model(q['audio_features'][ix].cuda(), q['valid'][ix].cuda())['state'].cpu())
    pred=torch.cat(values); target=q['target_centered_state'][ids]; valid=q['valid'][ids]
    report=state_metrics(pred,target,valid)
    report['equal_clip_mse']=float((((pred-target).square().sum((1,2)))/(valid.sum(1)*4)).mean())
    return report


def main():
    p=argparse.ArgumentParser()
    for key in ('data','source','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True); start=time.monotonic(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    set_status(a.output,status='loading_full_protocol')
    data,system,audio,identities,_=load_context(a.data,a.source,'cuda',47)
    data['_directory']=str(a.data); q=data['splits']['train']
    frozen={k:state_hash(m.state_dict()) for k,m in [('system',system),('audio',audio)]}
    if [len(data['splits'][r]['valid']) for r in ('train','validation')]!=[4098,446]: raise ValueError('Full protocol required')
    held=set(sorted(set(q['speaker']))[::5])
    chosen=set(sorted({q['sentence_id'][i] for i,s in enumerate(q['speaker']) if s in held})[::4])
    fit=torch.tensor([i for i,s in enumerate(q['speaker']) if s not in held and q['sentence_id'][i] not in chosen])
    cal=torch.tensor([i for i,s in enumerate(q['speaker']) if s in held and q['sentence_id'][i] in chosen])
    if a.smoke: fit,cal=fit[:32],cal[:8]
    protocol={'schema':'supervised_audio_readout_v1','data':data['provenance'],'source_sha256':sha(a.source),
        'selection':'TRAIN speaker+sentence holdout; representations full1540/PCA24+prosody4; fixed ridge grid',
        'fit_ids':[q['clip_id'][i] for i in fit], 'calibration_ids':[q['clip_id'][i] for i in cal],
        'alphas':list(ALPHAS),'smoke':a.smoke,'test_loaded':False,'default_replaced':False,
        'source_sha256_script':sha(__file__), 'target':'existing centered signed four-state spline16',
        'limitations':'diagnostic ablation; comparison with old TCN changes both readout and bottleneck'}
    save_json(a.output/'protocol.json',protocol)
    set_status(a.output,status='fitting_train_only_statistics')
    mean,std,projection,_=fit_statistics(q,fit)
    gram,rhs=sufficient_statistics(q,fit,mean,std); candidates=[]
    for representation,proj in [('full1540',None),('pca24',projection)]:
        for alpha,coeff in solve_path(gram,rhs,proj).items():
            model=SupervisedAudioReadout(mean,std,coeff,data['target_scales']).cuda()
            result=score(q,cal,model)
            candidates.append({'representation':representation,'alpha':alpha,'calibration':result})
    chosen=min(candidates,key=lambda x:x['calibration']['equal_clip_mse'])
    save_json(a.output/'selection.json',{'candidates':candidates,'selected':chosen,'development_used':False})
    ids=fit if a.smoke else torch.arange(len(q['valid']))
    set_status(a.output,status='full_train_refit',clips=len(ids),representation=chosen['representation'],alpha=chosen['alpha'])
    mean,std,projection,_=fit_statistics(q,ids)
    gram,rhs=sufficient_statistics(q,ids,mean,std)
    coeff=solve_path(gram,rhs,projection if chosen['representation']=='pca24' else None)[chosen['alpha']]
    model=SupervisedAudioReadout(mean,std,coeff,data['target_scales']).cuda().eval()
    save_checkpoint(a.output/'final.pt',{'model':model.state_dict(),'protocol':protocol,'protocol_sha256':canonical_hash(protocol),'selection':chosen})
    if a.smoke:
        from scripts.train_full_staged import subset
        data['splits']['validation']=subset(data['splits']['validation'],torch.arange(4),'cpu')
    set_status(a.output,status='evaluating_full_development')
    report=final_evaluation(model,data,system,audio,identities,a.output)
    for k,m in [('system',system),('audio',audio)]:
        if frozen[k]!=state_hash(m.state_dict()): raise RuntimeError('Frozen source changed')
    state_passed=report['passed'] and not a.smoke
    temporal_passed=all(v['passed'] for pair in report['temporal_pairs'].values() for v in pair.values())
    save_json(a.output/'acceptance.json',{'state_gate_passed':state_passed,
        'temporal_gate_passed':temporal_passed,'quantitative_passed':state_passed and temporal_passed,
        'independent_emotion_AV_visual_pending':True,'test_loaded':False,'default_replaced':False,
        'interpretation':'state readout diagnostic; full animation success requires all temporal and independent checks'})
    set_status(a.output,status='complete',state_gate_passed=state_passed,
        quantitative_passed=state_passed and temporal_passed,elapsed_seconds=time.monotonic()-start,
        independent_emotion_AV_visual_pending=True)


if __name__=='__main__': main()
