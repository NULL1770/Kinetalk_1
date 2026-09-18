"""Matched six-arm semantic-condition pilot; no updates to full-face base weights."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from kinetalk_b0.models.semantic_upper_flow import SemanticUpperFlow, semantic_intervention
from scripts.joint_motion_metrics import score_clip, summarize
from scripts.fit_visual_semantic_student import audio_windows

UPPER=[41,42,43,44,45,5,6,12,13]
ARMS=('va_oracle','va_audio','va_static','posterior_oracle','posterior_audio','posterior_static')
SEEDS=(42,123,2026,77)
SCHEMA='visual_semantic_condition_pilot_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save_json(path,value):
    tmp=Path(path).with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)


def logit(x):return torch.logit(x.clamp(.001,.999))


def stats(clips):
    train=[c for c in clips if c['split']=='train']
    if not train:raise ValueError('No training clips')
    target=torch.cat([logit(c['motion'][c['valid']&c['semantic_valid']][:,UPPER]) for c in train])
    result={'motion_mean':target.mean(0),'motion_std':target.std(0).clamp_min(.2)}
    for key in ('va','posterior'):
        x=torch.cat([c[key][c['valid']&c['semantic_valid']] for c in train])
        result[key+'_mean']=x.mean(0);result[key+'_std']=x.std(0).clamp_min(.05)
    for key in ('affect_global','identity_code'):
        x=torch.stack([c[key] for c in train]);result[key+'_mean']=x.mean(0);result[key+'_std']=x.std(0).clamp_min(.05)
    raw=torch.cat([c['motion'][c['valid']&c['semantic_valid']][:,UPPER] for c in train])
    result['metric_scale']=raw.std(0).clamp_min(.02)
    return result


def validate_inputs(data,students):
    clips=data['clips'];sp=students['clips']
    if [c['clip_id'] for c in clips]!=[c['clip_id'] for c in sp]:raise ValueError('Student/dataset membership differs')
    train={c['sentence'] for c in clips if c['split']=='train'}
    hold={c['sentence'] for c in clips if c['split']=='holdout'}
    if train&hold:raise ValueError('Sentence overlap')
    for c,p in zip(clips,sp):
        if p['sentence']!=c['sentence'] or p['split']!=c['split']:raise ValueError('Student metadata differs')
        expected='sentence_oof' if c['split']=='train' else 'all_train'
        if p['prediction_source']!=expected:raise ValueError('Deployment-condition leakage')
        if not torch.equal(p['semantic_valid'],c['semantic_valid']):raise ValueError('Student semantic support differs')
        if not torch.equal(p['audio_valid'],c['valid']):raise ValueError('Student acoustic support differs')
        m=c['valid']&c['semantic_valid']
        if m.sum()<25:raise ValueError('Too few common frames')
        if c['motion'].shape!=(len(m),52) or c['baseline52'].shape!=(len(m),52):raise ValueError('Native52 required')
        for key,width in [('va',2),('posterior',8)]:
            for mode in ('actual','static','reverse'):
                v=p[key][mode]
                if v.shape!=(len(m),width) or not torch.isfinite(v[c['valid']]).all():raise ValueError('Invalid student condition')
                if key=='va' and (v[c['valid']].abs()>1.00001).any():raise ValueError('Student VA out of bounds')
                if key=='posterior' and ((v[c['valid']]<0).any() or not torch.allclose(v[c['valid']].sum(1),torch.ones(int(c['valid'].sum())),atol=1e-5)):
                    raise ValueError('Student posterior not normalized')
    return clips,sp


def semantic(c,p,arm,scales):
    kind,mode=arm.rsplit('_',1)
    if mode=='oracle':
        # Match the student's five-frame window averaging and block expansion.
        # The dense interpolated teacher remains unchanged in the source data.
        x=torch.zeros_like(c[kind])
        common=c['valid']&c['semantic_valid']
        if not common.any():raise ValueError('Oracle requires observed visual semantics')
        # Only the explicitly oracle arm may use a visual clip summary.
        # It fills unobserved teacher windows without changing the generation
        # clock; audio/static arms never consult this value or teacher mask.
        fallback=c[kind][common].mean(0)
        for left,right in audio_windows(c)['spans']:
            observed=c['semantic_valid'][left:right]
            x[left:right]=c[kind][left:right][observed].mean(0) if observed.any() else fallback
    else:x=p[kind]['actual' if mode=='audio' else 'static']
    x=(x-scales[kind+'_mean'])/scales[kind+'_std']
    # All arms share8-dimensional interface and identical initialization.
    return torch.nn.functional.pad(x,(0,8-x.shape[-1]))


def batch(clips,preds,ids,arm,scales,device):
    frames=max(len(clips[i]['valid']) for i in ids);n=len(ids)
    valid=torch.zeros(n,frames,dtype=torch.bool,device=device)
    loss_mask=torch.zeros_like(valid)
    sem=torch.zeros(n,frames,8,device=device);target=torch.zeros(n,frames,9,device=device)
    gs=[];ident=[]
    for j,i in enumerate(ids):
        c=clips[i];m=c['valid'];count=len(m)
        valid[j,:count]=m.to(device)
        loss_mask[j,:count]=(m&c['semantic_valid']).to(device)
        x=(logit(c['motion'][:,UPPER])-scales['motion_mean'])/scales['motion_std']
        target[j,:count]=torch.where(m[:,None],x,0).to(device)
        sem[j,:count]=torch.where(m[:,None],semantic(c,preds[i],arm,scales),0).to(device)
        gs.append((c['affect_global']-scales['affect_global_mean'])/scales['affect_global_std'])
        ident.append((c['identity_code']-scales['identity_code_mean'])/scales['identity_code_std'])
    return target,valid,sem,torch.stack(gs).to(device),torch.stack(ident).to(device),loss_mask


def decode_values(z,scales):
    return torch.sigmoid(z*scales['motion_std'].to(z)+scales['motion_mean'].to(z))


@torch.no_grad()
def evaluate(model,clips,preds,ids,arm,scales,device,steps=16,intervention=None):
    records=[];curves={}
    model.eval()
    for i in ids:
        c=clips[i];_,valid,sem,g,ident,score_mask=batch(clips,preds,[i],arm,scales,device)
        if intervention:sem=semantic_intervention(sem,valid,intervention)
        samples=[]
        for seed in SEEDS:
            key=int.from_bytes(hashlib.sha256(f'visual_semantic:{seed}:{c["clip_id"]}'.encode()).digest()[:8],'little')%(2**63-1)
            z=model.decode(valid,sem,g,ident,steps=steps,seed=key)
            v=decode_values(z,scales)[0].cpu().numpy()
            # Only acoustic/native-invalid frames keep the shared baseline.
            # Visual-semantic missing frames are generated but not scored.
            v[~valid[0].cpu().numpy()]=c['baseline52'][~valid[0].cpu(),:][:,UPPER].numpy()
            samples.append(v)
        samples=np.stack(samples);mask=score_mask[0].cpu().numpy();truth=c['motion'][:,UPPER].numpy()
        metric=score_clip(samples,truth,mask,scales['metric_scale'].numpy())
        records.append({'clip_id':c['clip_id'],'sentence':c['sentence'],'speaker':c['speaker'],'emotion':c['emotion'],**metric})
        curves[c['clip_id']]=samples
    return {'summary':summarize(records),'rows':records},curves


def run(args):
    if args.output.exists():raise FileExistsError('Fresh pilot run required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20260918);np.random.seed(20260918);random.seed(20260918)
    data=torch.load(args.dataset,map_location='cpu',weights_only=False)
    student=torch.load(args.student,map_location='cpu',weights_only=False)
    provenance=json.loads(args.student.with_name('provenance.json').read_text())
    if provenance['dataset_sha256']!=sha(args.dataset) or provenance['outputs']['predictions.pt']!=sha(args.student):raise ValueError('Student provenance differs')
    model_path=args.student.with_name('models.pt')
    if provenance['outputs'].get('models.pt')!=sha(model_path):raise ValueError('Student fitted models differ')
    fitted=torch.load(model_path,map_location='cpu',weights_only=False)
    clips,preds=validate_inputs(data,student);scales=stats(clips)
    allowed={c['clip_id'] for c in clips if c['split']=='train'}
    if set(fitted['all_train']['fit_clip_ids'])!=allowed:raise ValueError('Full student fitted wrong members')
    for c,p in zip(clips,preds):
        if c['split']=='train':
            fold_model=fitted['oof'][p['fold']]
            expected={d['clip_id'] for d in clips if d['split']=='train' and student['fold_by_sentence'][d['sentence']]!=p['fold']}
            if c['sentence'] in fold_model['fit_sentences'] or set(fold_model['fit_clip_ids'])!=expected:
                raise ValueError('Student OOF fitting members leaked')
    train=[i for i,c in enumerate(clips) if c['split']=='train'];hold=[i for i,c in enumerate(clips) if c['split']=='holdout']
    # Four metadata-first train examples permit distinguishing underfit from holdout failure.
    fit_examples=[min((i for i in train if clips[i]['emotion']==e),key=lambda i:clips[i]['clip_id']) for e in sorted({clips[i]['emotion'] for i in train})]
    config={'semantic_dim':8,'global_dim':clips[0]['affect_global'].numel(),'identity_dim':clips[0]['identity_code'].numel(),
            'motion_dim':9,'hidden':64,'depth':4}
    source=SemanticUpperFlow(**config);initial=copy.deepcopy(source.state_dict())
    protocol={'schema':SCHEMA,'dataset_sha256':sha(args.dataset),'student_sha256':sha(args.student),
              'script_sha256':sha(__file__),'model_sha256':sha(Path(__file__).parents[1]/'kinetalk_b0/models/semantic_upper_flow.py'),
              'epochs':args.epochs,'batch_size':args.batch_size,'lr':.0003,'seed':20260918,'sample_seeds':list(SEEDS),
              'config':config,'arms':list(ARMS),'normalization_fit_only':True,'initialization_shared':True,
              'flow_noise_order_matched':True,'all_train_audio_conditions':'sentence-levelOOF',
              'generation_mask':'native acoustic valid only; independent of visual teacher missingness',
              'loss_and_scoring_mask':'native valid AND visual semantic valid, common across six arms',
              'oracle_missing_windows':'oracle-only observed clip teacher mean; audio/static never use visual summary',
              'semantic_clock':'all arms use five-native-frame window averages and block expansion',
              'semantic_clock':'all arms use same five-native-frame windows and block expansion; oracle also window-averaged',
              'protected_channels':'other43 copied exactly from shared frozen baseline',
              'target_logit_clip':[.001,.999],'default_replaced':False,'dev405_indexed':False,'sealed_test_loaded':False,
              'scope':'small192/64 inner-fit sentence-disjoint pilot; upstream historically exposed; not final generalization',
              'fit_example_ids':[clips[i]['clip_id'] for i in fit_examples],
              'display_selection':'first lexicographic holdout clip per emotion, seed42, no outcome selection'}
    save_json(args.output/'protocol.json',protocol);torch.save(scales,args.output/'scales.pt')
    all_curves={};reports={};started=time.time()
    for arm in ARMS:
        model=SemanticUpperFlow(**config).to(args.device);model.load_state_dict(initial)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        rng=np.random.default_rng(20260918);noise_rng=torch.Generator(device=args.device).manual_seed(20260918)
        losses=[];steps=0;order_hash=hashlib.sha256()
        for epoch in range(args.epochs):
            model.train();order=np.asarray(train)[rng.permutation(len(train))];order_hash.update(order.tobytes())
            total=0.;count=0;start=time.time()
            for left in range(0,len(order),args.batch_size):
                ids=order[left:left+args.batch_size].tolist()
                target,valid,sem,g,ident,loss_mask=batch(clips,preds,ids,arm,scales,args.device)
                noise=torch.randn(target.shape,generator=noise_rng,device=args.device)
                t=torch.rand(len(ids),generator=noise_rng,device=args.device)
                loss=model.flow_loss(target,valid,sem,g,ident,noise,t,loss_mask=loss_mask)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                total+=float(loss.detach())*len(ids);count+=len(ids);steps+=1
            losses.append({'epoch':epoch+1,'flow_loss':total/count,'seconds':time.time()-start,'steps':steps})
            save_json(args.output/(arm+'_losses.json'),losses)
            checkpoint={'schema':SCHEMA,'state':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                        'config':config,'scales':scales,'protocol':protocol,'numpy_rng':rng.bit_generator.state,
                        'noise_rng':noise_rng.get_state(),'torch_rng':torch.get_rng_state(),'order_sha256':order_hash.hexdigest()}
            temp=args.output/(arm+'_last.tmp');torch.save(checkpoint,temp);temp.replace(args.output/(arm+'_last.pt'))
            save_json(args.output/'status.json',{'status':'training','arm':arm,'epoch':epoch+1,'steps':steps,'elapsed_seconds':time.time()-started})
            print('EPOCH',arm,epoch+1,total/count,round(time.time()-start,2),flush=True)
        arm_reports={}
        for role,ids in [('holdout',hold),('fit_examples',fit_examples)]:
            r,curves=evaluate(model,clips,preds,ids,arm,scales,args.device)
            arm_reports[role]=r
            for cid,value in curves.items():all_curves.setdefault(cid,{})[arm]=value
        # Same trained receiver: reverse semantic order, not a separately trained model.
        if arm.endswith('_audio') or arm.endswith('_oracle'):
            r,_=evaluate(model,clips,preds,hold,arm,scales,args.device,intervention='reverse')
            arm_reports['holdout_reverse_semantic']=r
        reports[arm]=arm_reports;save_json(args.output/'reports.json',reports)
        print('ARM_COMPLETE',arm,json.dumps(arm_reports['holdout']['summary']),flush=True)
    packed={}
    for c in clips:
        cid=c['clip_id']
        if cid not in all_curves:continue
        packed[cid]={'metadata':{k:c[k] for k in ('clip_id','sentence','split','speaker','emotion')},
                     'valid':(c['valid']&c['semantic_valid']).numpy(),'native_valid':c['valid'].numpy(),
                     'times':c['times'].numpy(),'target':c['motion'][:,UPPER].numpy(),
                     'target52':c['motion'].numpy(),'baseline52':c['baseline52'].numpy(),
                     'samples':all_curves[cid],'seeds':list(SEEDS)}
    torch.save({'schema':SCHEMA,'clips':packed,'protocol':protocol},args.output/'predictions.pt')
    save_json(args.output/'status.json',{'schema':SCHEMA,'status':'complete','epochs':args.epochs,'arms':list(ARMS),
              'elapsed_seconds':time.time()-started,'default_replaced':False,'naturalness_certified':False})
    save_json(args.output/'manifest.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='manifest.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('dataset','student','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cuda')
    run(p.parse_args())
