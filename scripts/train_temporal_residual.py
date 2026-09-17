"""Free stochastic correction around the frozen aligned audio trajectory.

This exploratory successor changes both the baseline and local condition to
the fixed aligned predictor. No target motion enters deployment. Base noise
is zero in both fit and inference; only the correction is sampled.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import hashlib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_temporal_repair as r
from scripts.train_formal_predictable_projection import save_json,save_checkpoint,capture_rng,canonical_hash
from scripts.train_predictable_renderer import state_hash
from kinetalk_b0.models.temporal_upper import TemporalUpperFlow,UPPER_INDICES,compose_upper_face


@torch.no_grad()
def cache_base(system,adapter,data,identities,args,steps):
    for role,q in data['splits'].items():
        outputs=[];locals=[]
        for ix in torch.arange(len(q['valid'])).split(args.batch_size):
            b=r.subset(q,ix,args.device);ident=r.batch_identity(identities,b);affect=r.projected_affect(system,adapter,b)
            pred=system.generate(b['content'],b['valid'],ident,affect,initial_noise=torch.zeros_like(b['motion']),
                steps=steps,base={'b0':b['b0'],'h0':b['h0']})['motion']
            outputs.append(pred.cpu());locals.append(affect['local'].cpu())
        q['aligned_base']=torch.cat(outputs);q['aligned_local']=torch.cat(locals)


@torch.no_grad()
def evaluate(system,upper,data,identities,args,scales,steps,full=True):
    q=data['splits']['validation'];ids=torch.arange(len(q['valid'])) if full else torch.randperm(len(q['valid']),generator=torch.Generator().manual_seed(20260917))[:64].sort().values
    reference=r.subset(q,ids,'cpu');metrics={};predictions={};seeds=r.SEEDS if full else (42,)
    for seed in seeds:
        noises=torch.randn(len(q['valid']),q['valid'].shape[1],9,generator=torch.Generator().manual_seed(seed))
        modes=['full']+(['base','static','reverse'] if full and seed==42 else [])
        for mode in modes:
            outs=[];emotions=[]
            for ix in ids.split(args.batch_size):
                b=r.subset(q,ix,args.device);ident=r.batch_identity(identities,b);pred=b['aligned_base']
                if mode!='base':
                    local=b['aligned_local']
                    if mode=='static':local=(local.sum(1,keepdim=True)/b['valid'].sum(1)[:,None,None])*b['valid'][...,None]
                    if mode=='reverse':
                        local=local.clone()
                        for i in range(len(local)):
                            good=b['valid'][i].nonzero(as_tuple=True)[0];local[i,good]=local[i,good.flip(0)]
                    correction=upper.decode(b['valid'],b['h0'],ident['code'],{'global':b['audio_global'],'intensity_value':b['audio_intensity']},
                        local,None,noises[ix].to(args.device),steps=steps)
                    pred=compose_upper_face(pred,pred[...,list(UPPER_INDICES)]+scales*correction,b['valid'])
                    if not torch.equal(pred[...,list(r.NOT_UPPER)],b['aligned_base'][...,list(r.NOT_UPPER)]):raise RuntimeError('Nonupper changed')
                t=system.encode_motion(torch.where(r.obs(b),pred-b['b0']-ident['baseline'][:,None],0.),b['valid'])
                emotions.extend((t['emotion_logits'].argmax(-1)==b['emotion_id']).cpu().tolist());outs.append(pred.cpu())
            name=f'{seed}/{mode}';predictions[name]=torch.cat(outs)
            metrics[name]={'populations':r.populations(predictions[name],reference),'generated_emotion_accuracy_nonindependent':sum(emotions)/len(emotions)}
    report={'schema':'aligned_temporal_residual_v1','clips':len(ids),'modes':metrics,'noise_seeds':list(seeds),'test_loaded':False}
    curves={'schema':'aligned_temporal_residual_v1','clip_id':reference['clip_id'],'target':reference['motion'],'valid':reference['valid'],
            'times':reference['times'],'channel_mask':reference['channel_mask'],'emotion_id':reference['emotion_id'],'speaker_id':reference['speaker_id'],
            'b0':reference['b0'],'predictions':predictions,'noise_seeds':list(seeds),'decode_steps':steps}
    return report,curves


def main():
    p=argparse.ArgumentParser()
    for name in ('source-run','audio','targets','enrollment','native-root','trained-run','align-run','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=59);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();start=time.monotonic()
    if args.output.exists():raise FileExistsError('Fresh output required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    data=r.load_training_inputs(args.source_run,args.audio,args.targets,args.enrollment,args.native_root)
    ck,bindings,steps,stride=r.matching_checkpoints(args.trained_run,data)
    system=data['system'].to(args.device).eval().requires_grad_(False);system.load_state_dict(ck['teacher']['system'])
    audio=r._make_audio(ck['audio']['audio'],stride,args.device)
    alignpath=args.align_run/'final.pt';completion=json.loads((args.align_run/'complete.json').read_text())
    if r.sha(alignpath)!=completion['final_sha256']:raise ValueError('Alignment checkpoint differs')
    aligned=torch.load(alignpath,map_location='cpu',weights_only=False)
    if aligned['phase']!='align' or aligned['source_bindings']!=bindings:raise ValueError('Alignment source differs')
    adapter=r.AlignedAudioLocal(audio.feature_mean,audio.feature_std,hidden=audio.input.out_features,rank=system.motion_teacher.rank,stride=system.motion_teacher.stride).to(args.device).eval().requires_grad_(False)
    adapter.load_state_dict(aligned['adapter'])
    r.cache_current_base(system,data,args.device,args.batch_size);identities=r.identity_cache(system,data,args.device)
    r.cache_conditions(system,audio,data,identities,args.device,args.batch_size);cache_base(system,adapter,data,identities,args,steps)
    train=data['splits']['train'];cc=list(UPPER_INDICES)
    mask=r.obs(train)[...,cc];residual=train['motion'][...,cc]-train['aligned_base'][...,cc]
    scales=((torch.where(mask,residual,0.).square().sum((0,1))/mask.sum((0,1))).sqrt().clamp_min(.02)).to(args.device)
    upper=TemporalUpperFlow(data['config'],use_state=False).to(args.device).eval();params=list(upper.parameters());optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-5)
    frozen={k:state_hash(v.state_dict()) for k,v in [('system',system),('audio',audio),('adapter',adapter)]}
    args.output.mkdir(parents=True)
    recipe={'schema':'aligned_temporal_residual_v1','args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'source_bindings':bindings,'alignment_sha256':r.sha(alignpath),'frozen':frozen,'scales':scales.cpu().tolist(),
            'objective':'flow matching of complete residual to frozen zero-noise aligned base',
            'condition':'Frozen aligned rank8 teacher-coordinate local plus audio global/content/identity; no slow state',
            'comparison_limit':'Exploratory changes to baseline parameterization and condition, not a single-factor matched ablation',
            'data_provenance':data['provenance'],'source_sha256':r.sha(__file__),'test_loaded':False,'default_replaced':False}
    save_json(args.output/'provenance.json',{'recipe':recipe,'recipe_sha256':canonical_hash(recipe)})
    gen=torch.Generator().manual_seed(args.seed);ids=torch.arange(min(len(train['valid']),32) if args.smoke else len(train['valid']));epochs=1 if args.smoke else args.epochs
    for epoch in range(epochs):
        values=[];t0=time.monotonic();draws=hashlib.sha256()
        for ix in ids[torch.randperm(len(ids),generator=gen)].split(args.batch_size):
            b=r.subset(train,ix,args.device);ident=r.batch_identity(identities,b)
            target=torch.where(b['valid'][...,None],(b['motion'][...,cc]-b['aligned_base'][...,cc])/scales,0.)
            noise=torch.randn(target.shape,generator=gen);ft=torch.rand(len(ix),generator=gen)
            for tensor in (ix,noise,ft):draws.update(tensor.numpy().tobytes())
            loss=upper.flow_loss(target,b['valid'],b['h0'],ident['code'],{'global':b['audio_global'],'intensity_value':b['audio_intensity']},
                b['aligned_local'],None,noise.to(args.device),ft.to(args.device))
            r.optimize(loss,optimizer,params);values.append(float(loss.detach()))
        record={'epoch':epoch+1,'loss':sum(values)/len(values),'seconds':time.monotonic()-t0,'batch_noise_time_sha256':draws.hexdigest()}
        save_json(args.output/f'epoch{epoch+1:03d}.json',record);save_json(args.output/'status.json',{'status':'running',**record})
        payload={'schema':'aligned_temporal_residual_v1','upper':upper.state_dict(),'scales':scales.cpu(),'completed_epochs':epoch+1,'recipe_sha256':canonical_hash(recipe),
            'optimizer':optimizer.state_dict(),'rng':capture_rng(gen)};save_checkpoint(args.output/'last.pt',payload)
        print(json.dumps(record),flush=True)
        if (epoch+1)%4==0 or args.smoke:
            report,_=evaluate(system,upper,data,identities,args,scales,steps,False);save_json(args.output/f'dev_epoch{epoch+1:03d}.json',report)
    for key,module in [('system',system),('audio',audio),('adapter',adapter)]:
        if state_hash(module.state_dict())!=frozen[key] or any(p.grad is not None for p in module.parameters()):raise RuntimeError('Frozen subsystem changed')
    report,curves=evaluate(system,upper,data,identities,args,scales,steps,not args.smoke)
    save_json(args.output/'evaluation.json',report);save_checkpoint(args.output/'curves.pt',curves)
    save_checkpoint(args.output/'final.pt',{k:v for k,v in payload.items() if k not in ('optimizer','rng')})
    save_json(args.output/'complete.json',{'final_sha256':r.sha(args.output/'final.pt'),'curves_sha256':r.sha(args.output/'curves.pt'),'frozen':frozen,'completed_epochs':epochs})
    save_json(args.output/'status.json',{'status':'complete','elapsed_seconds':time.monotonic()-start});print('RESIDUAL_COMPLETE',flush=True)


if __name__=='__main__':main()
