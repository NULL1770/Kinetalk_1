"""Mixed MEAD/HDTF training after audited source features are available.

Frozen neutral B0 never receives HDTF neutral targets. HDTF emotion/intensity
are missing. Same-source references require a source-time gap. This runner
fine-tunes a v2 MEAD checkpoint and stores both domain sample counts explicitly.
"""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
from dataclasses import asdict
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v2.hdtf_dataset import HDTFSplit
from kinetalk_v3.unified_model_v3 import (UnifiedV3Config,UnifiedExpressionV3Model,teacher_targets,
    v3_supervision_losses,relation_style_loss,anchor_matrix)
from kinetalk_b0.models.expression_response import motion_objective
from scripts.train_expression_response import load_runtime,development_fold,configure,state_digest,write,save
from scripts.train_full_staged import base_forward
from scripts.train_unified_v3 import select_balanced,build_base_cache,attach_base,reference_ids,make_refs,evaluate


def objective(model,b,refs,seed,target=None):
    if target is None:target=teacher_targets(b['motion'],b['b0'],b['valid'],b['channel_mask'],b['times'])
    out=model.training_forward(b,refs,target)
    reconstruction,parts=motion_objective(out['motion'],b['motion'],b['valid'],b['channel_mask'],b['times'],model.scales)
    supervised=v3_supervision_losses(model,out,target,b)
    codes=model.style(b['motion'],b['valid'],b['channel_mask'],model.scales)
    relation=relation_style_loss(torch.cat((codes,out['style'])),b['speaker_id'].repeat(2))
    same=F.smooth_l1_loss(codes,out['style'])
    generated=model.predict(b['audio_features'],b['h0'],b['b0'],b['valid'],refs,times=b['times'],seed=seed)
    measured=(generated['motion']-b['b0'])@anchor_matrix().to(b['motion'])
    quality=target['quality'].bool()
    adherence=torch.where(quality,F.smooth_l1_loss(measured,generated['student']['state'].detach(),reduction='none'),0.).sum()/quality.sum().clamp_min(1)
    loss=reconstruction+supervised['total']+.05*relation+.05*same+.1*adherence
    return loss,{'reconstruction':float(reconstruction),'position':float(parts['position']),
        'hazard':float(supervised['hazard']),'relation':float(relation),'adherence':float(adherence)}


@torch.no_grad()
def hdtf_batch(hdtf,ids,base,device):
    # Parse on surrounding source context before cutting query windows. An
    # event that continues through a crop is not treated as ending at the cut.
    context=hdtf.context_batch(ids,device)
    context.update(base_forward(base,context['content'],context['valid']))
    targets=teacher_targets(context['motion'],context['b0'],context['valid'],context['channel_mask'],context['times'])
    b=hdtf.batch(ids,device)
    b.update({k:hdtf.crop_context_tensor(context[k],context) for k in ('b0','h0')})
    return b,{k:hdtf.crop_context_tensor(v,context) for k,v in targets.items()}


def choose_hdtf(hdtf,ids,limit,rng):
    groups={}
    for i in ids:groups.setdefault(hdtf['speaker_group'][i],{}).setdefault(hdtf['source_id'][i],[]).append(i)
    for sources in groups.values():
        for values in sources.values():rng.shuffle(values)
    selected=[]
    while any(any(values for values in sources.values()) for sources in groups.values()):
        for person in sorted(groups):
            for source in sorted(groups[person]):
                if groups[person][source]:selected.append(groups[person][source].pop())
                if limit and len(selected)>=limit:return selected
    return selected


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(a):
    if min(a.epochs,a.steps,a.batch_size)<=0:raise ValueError('Positive epochs/steps/batch size required')
    configure(a.seed);device=torch.device(a.device);root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    binding=json.loads(Path(a.binding).read_text());data,base=load_runtime(binding,device);frozen=state_digest(base.stage1)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=UnifiedExpressionV3Model(UnifiedV3Config(**checkpoint['config']),checkpoint['model']['feature_mean'],checkpoint['model']['feature_std'],checkpoint['model']['scales']).to(device)
    model.load_state_dict(checkpoint['model']);model.teacher.eval().requires_grad_(False)
    hdtf=HDTFSplit(a.hdtf_manifest,a.hdtf_features,split='train',source_root=a.hdtf_source_root,
        motion_policy=a.hdtf_motion_policy,postprocess_manifest=a.hdtf_postprocess_manifest)
    hids=hdtf.eligible_reference_indices()
    # Source/speaker-balanced HDTF windows, not thousands of overlapping copies
    # from the first long recording. Metadata is selected before reading targets.
    rng=random.Random(a.seed)
    selected=choose_hdtf(hdtf,hids,a.hdtf_limit,rng)
    if not selected:raise ValueError('No HDTF training windows with independent references and features')
    q=data['splits']['train'];fold=development_fold(q,data['fit_sids']);mids=select_balanced(q,fold['train'],a.mead_limit,a.seed)
    pool={}
    for i in fold['train']:pool.setdefault(int(q['speaker_id'][i]),[]).append(i)
    refs_by_id={i:reference_ids(q,[i],pool,rng)[0] for i in mids}
    held=select_balanced(q,fold['speaker_dev']+fold['sentence_dev'],a.eval_limit,a.seed)
    cache=build_base_cache(q,base,sorted(set(mids+held+list(refs_by_id.values()))),device)
    protocol={'schema':'unified_v3_mixed_pilot_v1','source_checkpoint':str(a.checkpoint),'mead_fit_clips':len(mids),
        'source_checkpoint_sha256':file_sha(a.checkpoint),'source_sha256':file_sha(__file__),
        'seed':a.seed,'learning_rate':a.lr,'batch_size_per_domain':a.batch_size,
        'hdtf_fit_windows':len(selected),'hdtf_sources':len({hdtf['source_id'][i] for i in selected}),
        'hdtf_identities':len({hdtf['speaker_group'][i] for i in selected}),'hdtf_provenance':hdtf.provenance,
        'hdtf_neutral_targets':0,'hdtf_emotion_targets':0,'mixing':'equal MEAD/HDTF objectives per optimizer update',
        'epochs':a.epochs,'steps_per_epoch':a.steps,'teacher_frozen':True,'neutral_frozen':True,'phonetic_enabled':False,
        'event_labels':'surrounding source context parsed before crop; outer-context events censored',
        'test_loaded':False,'paper_result':False,'remaining':'matched-data baseline and multi-seed attribution'}
    write(root/'protocol.json',protocol);write(root/'selection.json',{'mead':mids,'hdtf':selected})
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr,weight_decay=.01)
    start=0;step=0;history=[]
    if a.resume:
        saved=torch.load(a.resume,map_location=device,weights_only=False)
        if saved['protocol']!=protocol:raise ValueError('Resume protocol differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['epoch'];step=saved['step'];history=saved['history']
        rng.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if not a.resume:
        write(root/'before_mead_dev.json',evaluate(model,q,held,cache,data,base,device,a.seed,False,pool))
    for epoch in range(start,a.epochs):
        model.train();model.teacher.eval();begin=time.monotonic();sums={}
        for update in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            mead_ids=rng.sample(mids,min(a.batch_size,len(mids)));b=attach_base(q.batch(torch.tensor(mead_ids),device),mead_ids,cache)
            refs=make_refs(q,[refs_by_id[i] for i in mead_ids],device)
            ml,mp=objective(model,b,refs,a.seed+step);(ml*.5).backward()
            hd_ids=rng.sample(selected,min(a.batch_size,len(selected)));hb,ht=hdtf_batch(hdtf,hd_ids,base,device)
            hr=hdtf.single_references(hd_ids,device,seed=a.seed+step)
            if not hr['reference_valid'].all():raise ValueError('Invalid independent HDTF reference')
            hl,hp=objective(model,hb,hr,a.seed+step+1,ht);(hl*.5).backward()
            if not torch.isfinite(hl+ml):raise RuntimeError('Nonfinite mixed objective')
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();step+=1
            state={'status':'training','epoch':epoch+1,'step':step,'mead_loss':float(ml),'hdtf_loss':float(hl),'gradient_norm':float(grad),'mead':mp,'hdtf':hp}
            for k,v in [('mead_loss',float(ml)),('hdtf_loss',float(hl))]:sums[k]=sums.get(k,0.)+v
            if update%10==0:write(root/'state.json',state);print(json.dumps(state),flush=True)
        if state_digest(base.stage1)!=frozen:raise RuntimeError('Frozen B0 changed')
        dev=evaluate(model,q,held,cache,data,base,device,a.seed,False,pool)
        entry={'epoch':epoch+1,'updates':step,'seconds':time.monotonic()-begin,'mead_dev':dev,**{k:v/a.steps for k,v in sums.items()}};history.append(entry)
        saved={'model':model.state_dict(),'config':asdict(model.cfg),'optimizer':optimizer.state_dict(),'protocol':protocol,
            'epoch':epoch+1,'step':step,'history':history,'python_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        save(root/'resume.pt',saved);write(root/'history.json',history)
    save(root/'final.pt',{k:v for k,v in saved.items() if k!='optimizer'})
    write(root/'complete.json',{'status':'complete','updates':step,'protocol':protocol,'history':history,'frozen_b0_sha256':frozen,'improvement_verified':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--hdtf-manifest',required=True);p.add_argument('--hdtf-features',required=True);p.add_argument('--hdtf-source-root')
    p.add_argument('--hdtf-motion-policy',choices=('raw','sg5'),default='raw');p.add_argument('--hdtf-postprocess-manifest')
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--steps',type=int,default=40);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--hdtf-limit',type=int,default=512);p.add_argument('--mead-limit',type=int,default=512);p.add_argument('--eval-limit',type=int,default=64);p.add_argument('--resume');run(p.parse_args())

