"""Fixed-budget, full-train deterministic mouth recovery with a fail-closed gate.

Continues the provenance-known full_v1 articulation checkpoint. It deliberately
changes the base contract from neutral-only articulation to an all-emotion,
audio-conditioned mouth. No validation gradients, test targets or test refs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_paper_full_data import load_paper_data
from scripts.train_full_staged import base_forward, subset, obs, huber, MOUTH, sha
from scripts.train_formal_predictable_projection import save_json, save_checkpoint
from scripts.train_predictable_renderer import state_hash
from scripts.mouth_protection import mouth_metrics, protection_report


def intervene(content, valid, mode):
    """Operate on the actual native input clock, retaining holes and padding."""
    clean = torch.where(valid[..., None], content, 0.)
    if mode == 'real': return clean
    if mode == 'static':
        mean = clean.sum(1, keepdim=True) / valid.sum(1)[:, None, None]
        return torch.where(valid[..., None], mean, 0.)
    if mode != 'reverse': raise ValueError('Unknown content intervention')
    result = clean.clone()
    for i in range(len(content)):
        ix = valid[i].nonzero(as_tuple=True)[0]
        result[i, ix] = clean[i, ix.flip(0)]
    return result


def grouped(pred, q):
    answer = {}
    for name, keep in [('overall', torch.ones(len(pred), dtype=torch.bool)),
                       ('neutral', q['emotion_id'] == 0), ('nonneutral', q['emotion_id'] != 0)]:
        answer[name] = mouth_metrics(pred[keep], q['motion'][keep], q['valid'][keep], q['channel_mask'][keep])
    return answer


@torch.no_grad()
def predict(system, q, device, batch, mode='real'):
    system.eval(); result = []
    for ids in torch.arange(len(q['valid'])).split(batch):
        b = subset(q, ids, device)
        result.append(base_forward(system, intervene(b['content'], b['valid'], mode), b['valid'])['b0'].cpu())
    return torch.cat(result)


def centered_per_clip(pred, q):
    cc = list(MOUTH); mask = obs(q)[..., cc]
    x = torch.where(mask, pred[..., cc].double(), 0.)
    y = torch.where(mask, q['motion'][..., cc].double(), 0.)
    count = mask.sum(1, keepdim=True).clamp_min(1)
    x = torch.where(mask, x-x.sum(1, keepdim=True)/count, 0.)
    y = torch.where(mask, y-y.sum(1, keepdim=True)/count, 0.)
    return ((x-y).square().sum((1, 2))/mask.sum((1, 2))).numpy()


def paired_interval(real, other, q):
    delta = centered_per_clip(real, q)-centered_per_clip(other, q)
    clusters = {}
    for sentence, value in zip(q['sentence_id'], delta): clusters.setdefault(str(sentence), []).append(float(value))
    means = np.array([np.mean(x) for x in clusters.values()]); counts = np.array([len(x) for x in clusters.values()])
    rng = np.random.default_rng(20260920); ix = rng.integers(len(means), size=(2000, len(means)))
    boot = (means[ix]*counts[ix]).sum(1)/counts[ix].sum(1)
    ci = np.quantile(boot, [.025, .975]).tolist()
    return {'real_minus_control': float(delta.mean()), 'sentence_cluster_95ci': ci,
            'clusters': len(clusters), 'clips': len(delta), 'passed': bool(len(clusters)>1 and ci[1]<0)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--continue-protected', action='store_true')
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    if a.output.exists() and not a.resume: raise FileExistsError('Use a fresh recovery output')
    a.output.mkdir(parents=True, exist_ok=True)
    if not 1 <= a.epochs <= 60: raise ValueError('Fixed epoch budget must be 1..60')
    started = time.monotonic(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(47); np.random.seed(47); torch.manual_seed(47)
    save_json(a.output/'status.json', {'status':'loading_data','test_loaded':False})
    data = load_paper_data(a.data, seed=47)
    system = data['system'].to(a.device); cfg = data['config']
    source = torch.load(a.source, map_location='cpu', weights_only=False)
    if source.get('stage') != 'articulation' or source.get('data_manifest_sha256') != data['provenance']['manifest_sha256']:
        raise ValueError('Require provenance-matched articulation checkpoint')
    system.load_state_dict(source['system'], strict=True)
    support = data['splits']['train']['channel_mask'].any(0).bool()
    residual = support.clone(); residual[list(MOUTH)] = False
    cfg['model'].update(motion_support=support.tolist(), residual_support=residual.tolist())
    system.set_motion_support(support.to(a.device)); system.set_residual_support(residual.to(a.device))
    train = data['splits']['train']; val = data['splits']['validation']
    if a.smoke:
        # Keep both strata while remaining metadata-only; never select by quality.
        for role in ('train','validation'):
            q = data['splits'][role]; ids = torch.cat([(q['emotion_id']==0).nonzero(as_tuple=True)[0][:8],
                (q['emotion_id']!=0).nonzero(as_tuple=True)[0][:8]]).sort().values
            data['splits'][role] = subset(q, ids, 'cpu')
        train, val = data['splits']['train'], data['splits']['validation']
    fit_ids = torch.randperm(len(train['valid']), generator=torch.Generator().manual_seed(20260920))[:min(256,len(train['valid']))].sort().values
    fit = subset(train, fit_ids, 'cpu')
    protocol = {'schema':'paper_mouth_recovery_v1','source_sha256':sha(a.source),
        'manifest_sha256':data['provenance']['manifest_sha256'],
        'prepared_index_sha256':data['provenance'].get('index_sha256'),
        'prepared_config_sha256':data['provenance'].get('config_sha256'),'train_clips':len(train['valid']),
        'train_valid_frames':int(train['valid'].sum()),'validation_clips':len(val['valid']),
        'train_diagnostic_clip_ids':fit['clip_id'],'epochs':1 if a.smoke else a.epochs,'batch_size':a.batch_size,
        'continue_protected':a.continue_protected,'device':a.device,'smoke':a.smoke,
        'base_contract':'all-emotion deterministic audio mouth; not emotion-neutral',
        'objective':'existing scale-normalized raw Huber + 0.1 displacement Huber; no added loss',
        'selection':'fixed endpoint; intermediate validation diagnostic only','motion_support':support.tolist(),
        'residual_support':residual.tolist(),'test_loaded':False,'default_replaced':False,
        'sources':{name:sha(Path(__file__).parents[1]/name) for name in ['scripts/recover_paper_mouth.py','scripts/train_full_staged.py',
            'scripts/mouth_protection.py','kinetalk_b0/models/neutral_affect.py','kinetalk_b0/models/dit.py',
            'kinetalk_b0/models/model.py','kinetalk_b0/models/encoders.py','scripts/prepare_paper_full_data.py']}}
    if a.resume:
        if json.loads((a.output/'protocol.json').read_text()) != protocol: raise ValueError('Recovery recipe changed')
    else: save_json(a.output/'protocol.json', protocol)
    for name in protocol['sources']:
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((Path(__file__).parents[1]/name).read_bytes())
    old_val = predict(system, val, a.device, a.batch_size); old_fit = predict(system, fit, a.device, a.batch_size)
    save_json(a.output/'baseline.json', {'validation':grouped(old_val,val),'fit_probe':grouped(old_fit,fit)})
    system.requires_grad_(False); system.stage1.requires_grad_(True)
    frozen = state_hash({k:v for k,v in system.state_dict().items() if not k.startswith('stage1.')})
    optimizer = torch.optim.AdamW(system.stage1.parameters(), lr=1e-4, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(47); first = 0
    if a.resume:
        last = torch.load(a.output/'last.pt', map_location=a.device, weights_only=False)
        system.load_state_dict(last['system']); optimizer.load_state_dict(last['optimizer'])
        first = last['epoch']; gen.set_state(last['generator'].cpu()); torch.set_rng_state(last['torch_rng'].cpu())
        if a.device.startswith('cuda'): torch.cuda.set_rng_state_all([x.cpu() for x in last['cuda_rng']])
    scales = data['target_scales'].to(a.device)[list(MOUTH)].clamp_min(.05)
    for epoch in range(first, protocol['epochs']):
        begin = time.monotonic(); losses = []; system.eval(); system.stage1.train()
        order = torch.randperm(len(train['valid']), generator=gen)
        for ids in order.split(a.batch_size):
            b = subset(train, ids, a.device); mask = obs(b)[..., list(MOUTH)]
            out = base_forward(system,b['content'],b['valid'],gradients=True)['b0'][...,list(MOUTH)]
            target = b['motion'][...,list(MOUTH)]; pair = mask[:,1:] & mask[:,:-1]
            raw = huber(out/scales,target/scales,mask)
            velocity = huber((out[:,1:]-out[:,:-1])/scales,(target[:,1:]-target[:,:-1])/scales,pair)
            loss = raw + .1*velocity
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite mouth training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(system.stage1.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach()))
        record = {'status':'training','epoch':epoch+1,'epochs':protocol['epochs'],'clips':len(train['valid']),
            'loss':float(np.mean(losses)),'seconds':time.monotonic()-begin,'elapsed_seconds':time.monotonic()-started,'test_loaded':False}
        save_json(a.output/f'epoch{epoch+1:03d}.json',record); save_json(a.output/'status.json',record)
        save_checkpoint(a.output/'last.pt',{'system':system.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
            'generator':gen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []})
        print(json.dumps(record),flush=True)
        if (epoch+1)%10==0:
            save_json(a.output/f'validation_epoch{epoch+1:03d}.json',grouped(predict(system,val,a.device,a.batch_size),val))
    if state_hash({k:v for k,v in system.state_dict().items() if not k.startswith('stage1.')}) != frozen:
        raise RuntimeError('Non-mouth-base module drift')
    save_json(a.output/'status.json',{'status':'evaluating','elapsed_seconds':time.monotonic()-started,'test_loaded':False})
    preds = {mode:predict(system,val,a.device,a.batch_size,mode) for mode in ('real','static','reverse')}
    new_fit = predict(system,fit,a.device,a.batch_size)
    protection = protection_report(preds['real'],old_val,val['motion'],val['valid'],val['channel_mask'],val['emotion_id'])
    paired = {mode:paired_interval(preds['real'],preds[mode],val) for mode in ('static','reverse')}
    quality = grouped(preds['real'],val)
    useful = all(quality[g].get('centered_r2') is not None and quality[g]['centered_r2']>0 for g in ('overall','neutral','nonneutral'))
    passed = bool(protection['passed'] and useful and all(v['passed'] for v in paired.values()) and not a.smoke)
    result = {'passed':passed,'protection':protection,'positive_centered_r2_all_strata':useful,'paired':paired,
        'validation':{m:grouped(v,val) for m,v in preds.items()},'fit_probe':grouped(new_fit,fit),
        'test_loaded':False,'default_replaced':False,'scope':'development B0 gate, not fullface success'}
    save_json(a.output/'evaluation.json',result)
    save_checkpoint(a.output/'curves.pt',{'clip_id':val['clip_id'],'sentence_id':val['sentence_id'],
        'target':val['motion'],'valid':val['valid'],'times':val['times'],'channel_mask':val['channel_mask'],
        'emotion_id':val['emotion_id'],'predictions':{'old':old_val,**preds}})
    updated = {**source,'system':system.state_dict(),'config':cfg,'stage':'articulation',
        'recovery_protocol':protocol,'base_contract':protocol['base_contract'],'recovery_passed':passed,
        'completed_epochs':source['completed_epochs']+protocol['epochs'],'test_loaded':False}
    if a.resume and (a.output/'final.pt').exists():
        existing=torch.load(a.output/'final.pt',map_location='cpu',weights_only=False)
        if (existing.get('recovery_protocol')!=protocol or existing.get('recovery_passed')!=passed
                or state_hash(existing['system'])!=state_hash(updated['system'])):
            raise ValueError('Completed recovery checkpoint differs from replay; preserve original prerequisite')
        # The downstream recipe binds the serialized SHA, not just weights.
        # Never rewrite a completed prerequisite when resuming its child.
    else:save_checkpoint(a.output/'final.pt',updated)
    save_json(a.output/'status.json',{'status':'complete' if passed else 'gate_rejected','passed':passed,
        'elapsed_seconds':time.monotonic()-started,'test_loaded':False,'protected_training_started':False})
    print('MOUTH_RECOVERY_GATE',passed,flush=True)
    if passed and a.continue_protected:
        code = Path(__file__).resolve().parents[1]
        cmd = [sys.executable,'-u',str(code/'scripts/train_full_staged.py'),'--paper-data',str(a.data),
            '--output',str(a.output/'protected'),'--start-stage','identity','--end-stage','audio',
            '--stage-checkpoint',str(a.output/'final.pt'),'--protect-mouth','--epochs','12',
            '--identity-epochs','200','--compact','--artifact-dir',str(a.output/'protected_artifacts'),
            '--device',a.device,'--batch-size',str(a.batch_size)]
        if (a.output/'protected/summary.json').exists():
            previous=json.loads((a.output/'protected/summary.json').read_text())
            if previous.get('status')=='complete':
                save_json(a.output/'status.json',{'status':'protected_complete','mouth_base_passed':True,'test_loaded':False})
                return
        if (a.output/'protected/last.pt').exists():cmd.append('--resume')
        save_json(a.output/'status.json',{'status':'protected_training','command':cmd,'test_loaded':False})
        with (a.output/'protected.log').open('a') as log:
            rc = subprocess.call(cmd,cwd=code,stdout=log,stderr=subprocess.STDOUT)
        save_json(a.output/'status.json',{'status':'protected_complete' if rc==0 else 'protected_failed',
            'exit_code':rc,'mouth_base_passed':True,'dynamic_success_claim':False,'test_loaded':False})


if __name__ == '__main__': main()
