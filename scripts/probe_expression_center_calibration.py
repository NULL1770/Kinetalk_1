"""Fast, frozen-motion probe for reference-conditioned expression centers.

This probe does not retrain the audio student or empirical motion bank. It fits a
small static correction from frozen global audio, neutral identity and anchor to
the clip-level upper9 mean error. The correction is applied in bounded coefficient
space to every frame, so temporal residuals, mouth and non-upper channels remain
unchanged. TRAIN fit/calibration are used for selection; validation is loaded once.
The probe is diagnostic and does not replace the reference decoder checkpoint.
"""
from __future__ import annotations

import argparse, copy, json, time
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kinetalk_b0.models.reference_intensity_decoder import ReferenceIntensityDecoder, ReferenceIntensityStudent
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face
from scripts.reference_decoder_data import load_reference_context
from scripts.train_reference_intensity_decoder import (SOURCE_SHA, fit_statistics, prepare_cache,
    batch, predict, _fold, _diverse_ids, mean_valid, scores)
from scripts.train_formal_predictable_projection import save_json
from scripts.extract_emotion2vec_pilot import sha

UPPER = list(UPPER_INDICES)

class CenterCalibrator(nn.Module):
    def __init__(self, context_dim=201, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim, hidden), nn.SiLU(), nn.Linear(hidden, 9))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, global_code, identity_code, anchor):
        return self.net(torch.cat((global_code, identity_code, anchor[..., UPPER]), -1))

def train_epoch(model, cache, ids, scales, device, lr=.003):
    model.train(); opt = getattr(model, '_opt', None)
    if opt is None:
        model._opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01); opt=model._opt
    order = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(20260922))]
    total=0.
    for ix in order.split(64):
        b=batch(cache, ix, device)
        target=mean_valid(b['upper'], b['valid'])
        pred=model(b['global_code'], b['identity_code'], b['anchor'])
        loss=((pred-target)/scales).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        total += float(loss.detach())*len(ix)
    return total/len(ids)

@torch.no_grad()
def corrected(decoder, student, calibrator, cache, ids, stats, device, mode='full'):
    base, requested=predict(decoder, student, cache, ids, device, batch_size=32, mode=mode)
    b=batch(cache, ids, 'cpu'); predicted_center=calibrator(b['global_code'].to(device), b['identity_code'].to(device), b['anchor'].to(device)).cpu()
    valid=b['valid']; up=base; delta=predicted_center-mean_valid(up,valid)
    room=torch.where(delta[:,None]>=0,1-up,up)
    # Correction is static in coefficient space and bounded; no target enters inference.
    out=up + room*torch.tanh(delta[:,None]/room.clamp_min(1e-6))
    out=torch.where(valid[...,None],out,up)
    return out, delta

def main():
    p=argparse.ArgumentParser()
    for n in ('data','source','reference','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--epochs',type=int,default=12)
    args=p.parse_args();
    if args.output.exists(): raise FileExistsError('fresh output required')
    args.output.mkdir(parents=True); started=time.monotonic()
    data,system,audio,identities=load_reference_context(args.data,args.source,args.device,role='train',expected_source_sha256=SOURCE_SHA)
    q=data['splits']['train']; ck=torch.load(args.reference/'final.pt',map_location='cpu',weights_only=True)
    if ck.get('schema')!='reference_intensity_decoder_v1' or ck['protocol']['source_sha256']!=SOURCE_SHA: raise ValueError('reference provenance')
    # Keep cache statistics on CPU; only model inputs move to the selected device.
    stats=ck['stats']
    cache=prepare_cache(q,stats); decoder=ReferenceIntensityDecoder(**ck['decoder_config']).to(args.device).eval(); student=ReferenceIntensityStudent(**ck['student_config']).to(args.device).eval()
    decoder.load_state_dict(ck['decoder']); student.load_state_dict(ck['student']); decoder.requires_grad_(False);student.requires_grad_(False)
    fit,cal,_=_fold(q); calibrator=CenterCalibrator().to(args.device)
    best=None; bestscore=float('inf'); history=[]
    for epoch in range(1,args.epochs+1):
        loss=train_epoch(calibrator,cache,fit,stats['scales9'].to(args.device),args.device)
        pred,_=corrected(decoder,student,calibrator,cache,cal,stats,args.device)
        report=scores(pred,torch.zeros_like(pred[...,0:2]),cache,cal,stats)
        mean=float(report['normalized_mean_upper_mse']); centered=float(report['normalized_centered_upper_mse'])
        history.append({'epoch':epoch,'train_center_loss':loss,'cal_mean_mse':mean,'cal_centered_mse':centered,'cal_raw_mse':report['normalized_upper_mse']})
        score=mean+max(0.,centered-.037)*2
        if score<bestscore: bestscore=score; best=copy.deepcopy(calibrator.state_dict())
    calibrator.load_state_dict(best); save_json(args.output/'history.json',history)
    pred,_=corrected(decoder,student,calibrator,cache,cal,stats,args.device)
    base,_=predict(decoder,student,cache,cal,args.device,32,'full')
    calreport=scores(pred,torch.zeros_like(pred[...,0:2]),cache,cal,stats); basereport=scores(base,torch.zeros_like(base[...,0:2]),cache,cal,stats)
    # Development is loaded only after fit/cal selection.
    data,system,audio,identities=load_reference_context(args.data,args.source,args.device,role='validation',expected_source_sha256=SOURCE_SHA)
    qv=data['splits']['validation']; cachev=prepare_cache(qv,stats); ids=torch.arange(len(qv['valid']))
    predv,_=corrected(decoder,student,calibrator,cachev,ids,stats,args.device); basev,_=predict(decoder,student,cachev,ids,args.device,32,'full')
    base_report=scores(basev,torch.zeros_like(basev[...,0:2]),cachev,ids,stats); new_report=scores(predv,torch.zeros_like(predv[...,0:2]),cachev,ids,stats)
    save_json(args.output/'evaluation.json',{'schema':'expression_center_calibration_probe_v1','reference_sha256':sha(args.reference/'final.pt'),'source_sha256':SOURCE_SHA,'fit_count':len(fit),'cal_count':len(cal),'dev_count':len(ids),'history':history,'train_calibration':{'base':basereport,'calibrated':calreport},'development':{'base':base_report,'calibrated':new_report},'test_loaded':False,'default_replaced':False,'paper_success_established':False,'elapsed_seconds':time.monotonic()-started})
    torch.save({'schema':'expression_center_calibration_probe_v1','calibrator':calibrator.state_dict(),'reference_sha256':sha(args.reference/'final.pt')},args.output/'calibrator.pt')
    save_json(args.output/'status.json',{'status':'complete','test_loaded':False,'default_replaced':False,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'status':'complete','output':str(args.output)}))
if __name__=='__main__': main()
