"""Full-data deterministic mean/state learning without generator gradients."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_paper_full_data import load_paper_data
from scripts.train_full_staged import (subset,obs,cache_current_base,identity_cache,batch_identity,
    targets,region_report,identity_report,MOUTH,NOT_UPPER)
from scripts.train_formal_predictable_projection import save_json,save_checkpoint,capture_rng,restore_rng,canonical_hash
from scripts.train_predictable_renderer import state_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.compact_native_curves import compact_curves
from scripts.mouth_protection import protection_report
from kinetalk_b0.models.slow_state_affect import SlowStateAffect,UPPER_INDICES,compose_upper_face
from kinetalk_b0.models.isolated_audio_state import IsolatedAudioState
from scripts.joint_motion_metrics import _runs


def center(value,valid):
    clean=torch.where(valid[...,None],value,0.)
    mean=clean.sum(1,keepdim=True)/valid.sum(1)[:,None,None].clamp_min(1)
    return torch.where(valid[...,None],clean-mean,0.)


def masked_mse(x,y,valid):
    m=valid[...,None].expand_as(x)
    return (x[m]-y[m]).square().mean()


def trim(q,ids,device):
    b=subset(q,ids,device)
    end=int((b['valid']*torch.arange(1,b['valid'].shape[1]+1,device=device)).amax())
    temporal={'valid','motion','content','audio_features','times','b0','h0','target_centered_state','frozen_local'}
    for key,value in list(b.items()):
        if key in temporal and torch.is_tensor(value) and value.ndim>=2:
            b[key]=value[:,:end]
    return b


def load_context(data_path,source_path,device,seed=47):
    data=load_paper_data(data_path,seed=seed)
    saved=torch.load(source_path,map_location='cpu',weights_only=False)
    if saved.get('data_manifest_sha256')!=data['provenance']['manifest_sha256']:
        raise ValueError('Frozen source manifest mismatch')
    data['provenance']={**data['provenance'],
        'initialization':'frozen protected Stage4 system/audio restored; only new isolated modules initialize randomly',
        'frozen_source_sha256':sha(source_path)}
    system=data['system'];system.load_state_dict(saved['system'],strict=True)
    cfg=copy.deepcopy(saved['config']);system.set_motion_support(torch.tensor(cfg['model']['motion_support']))
    system.set_residual_support(torch.tensor(cfg['model']['residual_support']))
    calibration=cfg['model'].get('mouth_reference_calibration')
    if calibration is None:raise ValueError('Verified mouth calibration required')
    system.set_mouth_reference_calibration(calibration)
    system.to(device).requires_grad_(False).eval()
    stats=data['feature_stats'];audio=SlowStateAffect(stats['mean'],stats['std']).to(device)
    audio.load_state_dict(saved['audio'],strict=True);audio.requires_grad_(False).eval()
    data['config']=cfg;cache_current_base(system,data,device)
    identities=identity_cache(system,data,device)
    del saved
    # Cache frozen conditioning and target decomposition separately. Inference
    # module only receives explicit audio/global/reference inputs.
    with torch.no_grad():
        for q in data['splits'].values():
            globals_=[];logits=[];intensity=[];teacher_logits=[];slow=[]
            for ids in torch.arange(len(q['valid'])).split(32):
                b=subset(q,ids,device);affect=audio(b['audio_features'],b['valid'])
                globals_.append(affect['global'].cpu());logits.append(affect['emotion_logits'].cpu())
                intensity.append(affect['intensity_value'].cpu())
                truth,_=targets(b,data['target_scales'].to(device),16)
                slow.append(center(truth['state'],b['valid']).cpu())
                ident=batch_identity(identities,b)
                gt=system.encode_motion(torch.where(obs(b),b['motion']-b['b0']-ident['baseline'][:,None],0.),b['valid'])
                teacher_logits.append(gt['emotion_logits'].cpu())
            q['global_code']=torch.cat(globals_);q['emotion_logits']=torch.cat(logits)
            q['intensity_value']=torch.cat(intensity);q['teacher_logits']=torch.cat(teacher_logits)
            q['target_centered_state']=torch.cat(slow)
            q['identity_code']=torch.cat([identities[int(s)]['code'].cpu() for s in q['speaker_id']])
            m=obs(q)[...,list(UPPER_INDICES)]
            q['target_mean']=(torch.where(m,q['motion'][...,list(UPPER_INDICES)],0.).sum(1)/m.sum(1).clamp_min(1))
    train=data['splits']['train'];v=train['valid'][...,None]
    dynamic_scales=(train['target_centered_state'].square().sum((0,1))/v.sum()).sqrt().clamp_min(.02)
    return data,system,audio,identities,dynamic_scales


def make_model(data,dynamic_scales):
    return IsolatedAudioState(data['feature_stats']['mean'],data['feature_stats']['std'],
        data['target_scales'],dynamic_scales)


def forward(model,b,mode='audio'):
    return model(b['audio_features'],b['valid'],b['global_code'],b['identity_code'],b['anchors'],mode=mode)


@torch.no_grad()
def old_prediction(system,b,identities,noise):
    ident=batch_identity(identities,b)
    affect={'global':b['global_code'],'intensity_value':b['intensity_value'],
            'local':torch.zeros(*b['valid'].shape,64,device=b['valid'].device)}
    # Frozen audio local is added by caller for exact original Stage4 output.
    affect['local']=b['frozen_local']
    return system.generate(b['content'],b['valid'],ident,affect,initial_noise=noise,steps=12,
        base={'b0':b['b0'],'h0':b['h0']})['motion']


def paired_ci(delta,sentences):
    if len(delta)!=len(sentences):raise ValueError('Paired values and sentences differ in length')
    groups={}
    for v,s in zip(delta,sentences):
        if not np.isfinite(v):raise ValueError('Nonfinite paired statistic')
        groups.setdefault(s,[]).append(float(v))
    if not groups:raise ValueError('Empty paired comparison')
    n=np.array([len(v) for v in groups.values()]);means=np.array([np.mean(v) for v in groups.values()])
    ix=np.random.default_rng(20260921).integers(len(n),size=(2000,len(n)))
    boot=(n[ix]*means[ix]).sum(1)/n[ix].sum(1);ci=np.quantile(boot,[.025,.975]).tolist()
    return {'delta':float(np.mean(delta)),'sentence_cluster_95ci':ci,'clips':len(delta),'clusters':len(n),
            'passed':bool(len(n)>1 and ci[1]<0)}


def state_metrics(pred,target,valid):
    x=torch.where(valid[...,None],pred,0.).double();y=torch.where(valid[...,None],target,0.).double()
    mse=float((x-y).square().sum()/(valid.sum()*x.shape[-1]))
    energy=y.square().sum();ss=(x-y).square().sum()
    return {'mse':mse,'correlation':float((x*y).sum()/(x.square().sum()*energy).sqrt().clamp_min(1e-12)),
            'r2':float(1-ss/energy.clamp_min(1e-12))}


@torch.no_grad()
def evaluate(model,data,system,audio,identities,device,mode='audio',ids=None,fullface=True):
    q=data['splits']['validation']
    if ids is not None:q=subset(q,ids,'cpu')
    states={k:[] for k in ('full','static','reverse')};upper={k:[] for k in states};means=[]
    for ix in torch.arange(len(q['valid'])).split(16):
        b=subset(q,ix,device)
        for key,intervention in [('full',mode),('static','static'),('reverse','reverse')]:
            out=forward(model,b,intervention);states[key].append(out['state'].cpu());upper[key].append(out['upper'].cpu())
            if key=='full':means.append(out['mean'].cpu())
    states={k:torch.cat(v) for k,v in states.items()};upper={k:torch.cat(v) for k,v in upper.items()}
    target=q['target_centered_state'];valid=q['valid'];mask=valid[...,None]
    per={k:((v-target).square().sum((1,2))/(valid.sum(1)*4)).tolist() for k,v in states.items()}
    report={'clips':len(valid),'mode':mode,'states':{k:state_metrics(v,target,valid) for k,v in states.items()},
        'state_pairs':{k:paired_ci(np.array(per['full'])-per[k],q['sentence_id']) for k in ('static','reverse')},
        'mean_mse':float((torch.cat(means)-q['target_mean']).square().mean()),
        'per_clip_state_mse':{k:[{'clip_id':c,'sentence':s,'mse':x,
            'valid_frames':int(valid[i].sum()),'valid_runs':[[int(l),int(r)] for l,r in _runs(valid[i].numpy())]}
            for i,(c,s,x) in enumerate(zip(q['clip_id'],q['sentence_id'],v))] for k,v in per.items()},
        'test_loaded':False,'scope':'full development' if ids is None else 'fixed subset diagnostic'}
    if not fullface:return report,None
    predictions={};gen_emotion={};mouth={};old_means=[]
    for seed in (42,123,2026):
        noise=torch.randn(*q['motion'].shape,generator=torch.Generator().manual_seed(seed))
        old=[]
        for ix in torch.arange(len(valid)).split(16):
            b=subset(q,ix,device);b['frozen_local']=audio(b['audio_features'],b['valid'])['local']
            old.append(old_prediction(system,b,identities,noise[ix].to(device)).cpu())
        base=torch.cat(old);predictions[f'{seed}/base']=base
        old_means.append((base[...,list(UPPER_INDICES)]*mask).sum(1)/mask.sum(1))
        for k in upper:
            composed=compose_upper_face(base,upper[k],valid)
            if not torch.equal(composed[...,list(NOT_UPPER)],base[...,list(NOT_UPPER)]):
                raise RuntimeError('Non-upper identity/mouth protection failed')
            predictions[f'{seed}/{k}']=composed
    for key,pred in predictions.items():
        ref=predictions[key.split('/')[0]+'/base']
        mouth[key]=protection_report(pred,ref,q['motion'],valid,q['channel_mask'],q['emotion_id'])
        correct=0
        for ix in torch.arange(len(valid)).split(16):
            b=subset(q,ix,device);ident=batch_identity(identities,b)
            teacher=system.encode_motion(torch.where(obs(b),pred[ix].to(device)-b['b0']-ident['baseline'][:,None],0.),b['valid'])
            correct+=int((teacher['emotion_logits'].argmax(-1)==b['emotion_id']).sum())
        gen_emotion[key]=correct/len(valid)
    report.update(regions={k:region_report(v,q) for k,v in predictions.items()},
        mouth_protection=mouth,generated_teacher_accuracy_nonindependent=gen_emotion,
        mean_baseline_mse=float(torch.stack([(x-q['target_mean']).square().mean() for x in old_means]).mean()),
        input_audio_accuracy=float((q['emotion_logits'].argmax(-1)==q['emotion_id']).float().mean()),
        teacher_gt_accuracy=float((q['teacher_logits'].argmax(-1)==q['emotion_id']).float().mean()),
        identity=identity_report(system,data,device))
    curves={'clip_id':q['clip_id'],'target':q['motion'],'valid':valid,'times':q['times'],
        'channel_mask':q['channel_mask'],'b0':q['b0'],'predictions':predictions,'noise_seeds':[42,123,2026]}
    return report,curves


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','source','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--mode',choices=('audio','static'),default='audio');p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=47)
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1:raise ValueError('Positive budget required')
    if a.output.exists() and not a.resume:raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    data,system,audio,identities,ds=load_context(a.data,a.source,a.device,a.seed)
    if a.smoke:
        for role,q in data['splits'].items():data['splits'][role]=subset(q,torch.arange(min(32,len(q['valid']))),'cpu')
    train=data['splits']['train'];torch.manual_seed(a.seed);model=make_model(data,ds).to(a.device)
    frozen={'system':state_hash(system.state_dict()),'audio':state_hash(audio.state_dict())}
    root=Path(__file__).resolve().parents[1]
    sources=['scripts/train_isolated_audio_state.py','kinetalk_b0/models/isolated_audio_state.py',
        'kinetalk_b0/models/slow_state_affect.py','scripts/prepare_paper_full_data.py','scripts/train_full_staged.py',
        'scripts/mouth_protection.py','scripts/compact_native_curves.py','scripts/paper_generation_report.py']
    protocol={'schema':'isolated_mean_state_v1','source_sha256':sha(a.source),'data':data['provenance'],
        'epochs':1 if a.smoke else a.epochs,'mode':a.mode,'batch_size':a.batch_size,'seed':a.seed,'smoke':a.smoke,
        'state_dynamic_scales':ds.tolist(),'model_config':model.export_config(),'objective':'independent mean and centered state MSE, TRAIN scales',
        'sources':{s:sha(root/s) for s in sources},'test_loaded':False,'default_replaced':False,'selection':'fixed final epoch',
        'train_clips':len(train['valid']),'validation_clip_ids':data['splits']['validation']['clip_id'],
        'noise_seeds':[42,123,2026],'evaluation_modes':['full','base','static','reverse']}
    if a.resume:
        if json.loads((a.output/'protocol.json').read_text())!=protocol:raise ValueError('Resume protocol mismatch')
    else:
        save_json(a.output/'protocol.json',protocol)
        for s in sources:
            dest=a.output/'source'/s;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/s).read_bytes())
    mean_opt=torch.optim.AdamW(model.mean_branch.parameters(),lr=2e-4,weight_decay=1e-5)
    state_opt=torch.optim.AdamW(model.state_branch.parameters(),lr=2e-4,weight_decay=1e-5)
    gen=torch.Generator().manual_seed(a.seed);first=0;elapsed_before=0.
    if a.resume:
        ck=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
        if ck['protocol_sha256']!=canonical_hash(protocol):raise ValueError('Resume checkpoint protocol differs')
        mean_opt.load_state_dict(ck['mean_optimizer']);state_opt.load_state_dict(ck['state_optimizer'])
        first=ck['epoch'];elapsed_before=ck['elapsed_seconds'];restore_rng(ck['rng'],gen)
    scales=data['target_scales'][list(UPPER_INDICES)].to(a.device)
    for epoch in range(first,protocol['epochs']):
        tick=time.monotonic();model.train();sums={'mean':[],'state':[]}
        for ids in torch.randperm(len(train['valid']),generator=gen).split(a.batch_size):
            b=trim(train,ids,a.device);out=forward(model,b,a.mode)
            ml=((out['mean']-b['target_mean'])/scales).square().mean()
            sl=masked_mse(out['state']/ds.to(a.device),b['target_centered_state']/ds.to(a.device),b['valid'])
            if not torch.isfinite(ml+sl):raise FloatingPointError('Nonfinite state objective')
            mean_opt.zero_grad(set_to_none=True);ml.backward()
            nn.utils.clip_grad_norm_(model.mean_branch.parameters(),1.,error_if_nonfinite=True);mean_opt.step()
            state_opt.zero_grad(set_to_none=True)
            if a.mode=='audio':
                sl.backward();nn.utils.clip_grad_norm_(model.state_branch.parameters(),1.,error_if_nonfinite=True);state_opt.step()
            sums['mean'].append(float(ml.detach()));sums['state'].append(float(sl.detach()))
        record={'status':'training','epoch':epoch+1,'epochs':protocol['epochs'],'mode':a.mode,'clips':len(train['valid']),
            'losses':{k:float(np.mean(v)) for k,v in sums.items()},'seconds':time.monotonic()-tick,
            'elapsed_seconds':elapsed_before+time.monotonic()-started,'test_loaded':False}
        save_json(a.output/f'epoch{epoch+1:03d}.json',record);save_json(a.output/'status.json',record);print(json.dumps(record),flush=True)
        save_checkpoint(a.output/'last.pt',{'model':model.state_dict(),'mean_optimizer':mean_opt.state_dict(),
            'state_optimizer':state_opt.state_dict(),'epoch':epoch+1,'rng':capture_rng(gen),'elapsed_seconds':record['elapsed_seconds'],
            'protocol_sha256':canonical_hash(protocol)})
        if (epoch+1)%10==0:
            report,_=evaluate(model.eval(),data,system,audio,identities,a.device,a.mode,fullface=False)
            save_json(a.output/f'dev_epoch{epoch+1:03d}.json',report)
    model.eval();report,curves=evaluate(model,data,system,audio,identities,a.device,a.mode)
    probe_ids=torch.randperm(len(train['valid']),generator=torch.Generator().manual_seed(20260921))[:64].sort().values
    probe={**data,'splits':{**data['splits'],'validation':subset(train,probe_ids,'cpu')}}
    fit,_=evaluate(model,probe,system,audio,identities,a.device,a.mode,fullface=False)
    fit['scope']='fixed TRAIN diagnostic, not independent test';fit['clip_ids']=probe['splits']['validation']['clip_id']
    save_json(a.output/'training_probe.json',fit)
    from scripts.paper_generation_report import report_generation
    report_generation(curves,data,a.output,'state',SimpleNamespace(condition_mode=a.mode,artifact_dir=a.output/'scores'))
    benchmark=json.loads((a.output/'benchmark_summary.json').read_text())
    full_em=np.mean([report['generated_teacher_accuracy_nonindependent'][f'{s}/full'] for s in (42,123,2026)])
    base_em=np.mean([report['generated_teacher_accuracy_nonindependent'][f'{s}/base'] for s in (42,123,2026)])
    checks={'mean_no_worse':report['mean_mse']<=report['mean_baseline_mse'],
        'mouth_preserved':all(x['passed'] for x in report['mouth_protection'].values()),
        'mbe_no_worse':benchmark['full']['coefficient']['arkit_mbe']['value']<=benchmark['base']['coefficient']['arkit_mbe']['value'],
        'internal_emotion_no_large_drop':bool(full_em>=base_em-.03),
        'train_state_fit':fit['states']['full']['r2']>0 and fit['states']['full']['correlation']>0,
        'val_state_better_than_static':report['state_pairs']['static']['passed'],
        'val_state_better_than_reverse':report['state_pairs']['reverse']['passed']}
    report.update(checks=checks,passed=all(checks.values()) and not a.smoke,independent_emotion_AV_pending=True)
    save_json(a.output/'evaluation.json',report)
    manifest=json.loads((a.data/'manifest.json').read_text());lengths={r['clip_id']:r['frames'] for r in manifest['roles']['val']['query']}
    save_checkpoint(a.output/'native_curves.pt',compact_curves(curves,lengths))
    save_checkpoint(a.output/'final.pt',{'model':model.state_dict(),'config':model.export_config(),
        'dynamic_scales':ds,'protocol':protocol,'protocol_sha256':canonical_hash(protocol),'test_loaded':False})
    if state_hash(system.state_dict())!=frozen['system'] or state_hash(audio.state_dict())!=frozen['audio']:
        raise RuntimeError('Frozen identity/global/base drift')
    save_json(a.output/'status.json',{'status':'complete','passed':report['passed'],'elapsed_seconds':elapsed_before+time.monotonic()-started,'test_loaded':False,
        'final_sha256':sha(a.output/'final.pt'),'native_curves_sha256':sha(a.output/'native_curves.pt')})


if __name__=='__main__':main()
