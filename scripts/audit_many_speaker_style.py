"""25-enrollment style audit on non-fit queries; no test or model fitting.

Seen speakers on held sentences and unseen speakers are reported separately.
Exact sentence/emotion/intensity matches support target *statistics* only.
They are not synchronized performances or counterfactual frame targets.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig
from scripts.train_expression_response import (configure,load_runtime,cache_base,development_fold,
    reference_batch,state_digest,write,sha)
from scripts.refine_expression_prior import restore_model
from scripts.audit_reference_style_swap import summary,change,target_direction,refs_for,export
from scripts.evaluate_expression_response import aggregate,finite
from kinetalk_b0.emotion_probe import MotionEmotionProbe,motion_features


def select_queries(data, display_clips=()):
    q=data['splits']['train'];fold=development_fold(q,data['fit_sids'])
    candidates=[]
    for role,indices in [('unseen_internal',fold['speaker_dev']),('seen_held_sentence',fold['sentence_dev'])]:
        candidates.extend(('train',i,role) for i in indices)
    candidates.extend(('validation',i,'unseen_development') for i in range(len(data['splits']['validation']['clip_id'])))
    groups=defaultdict(list)
    for split,i,role in candidates:
        q=data['splits'][split]
        groups[(int(q['speaker_id'][i]),int(q['emotion_id'][i]))].append((split,i,role))
    # One reproducible utterance per person/emotion, without inspecting errors.
    display_clips=set(display_clips)
    selected=[min(items,key=lambda v:(data['splits'][v[0]]['clip_id'][v[1]] not in display_clips,
                  hashlib.sha256(data['splits'][v[0]]['clip_id'][v[1]].encode()).hexdigest()))
              for _,items in sorted(groups.items())]
    assert all(i not in set(fold['train']) for split,i,_ in selected if split=='train')
    return selected,candidates,fold


def key(q,i):
    return str(q['sentence_id'][i]),int(q['emotion_id'][i]),int(q['intensity_id'][i])


def summaries(rows):
    if not rows:return {'n':0,'metrics':{}}
    ignored={'clip_id','source','target','role','emotion','kind','method','policy'}
    return dict(n=len(rows),metrics=aggregate([{k:v for k,v in row.items() if k not in ignored} for row in rows]))


@torch.no_grad()
def run(a):
    configure(47);device=torch.device(a.device);out=Path(a.output)
    if out.exists():raise FileExistsError('Preserve completed or failed audit')
    out.mkdir(parents=True);(out/'render_inputs').mkdir()
    bnd=json.loads(Path(a.binding).read_text())
    for name,h in bnd['source_files'].items():assert sha(Path(__file__).resolve().parents[1]/name)==h,name
    assert sha(a.candidate)==bnd['style_audit']['candidate_checkpoint']['sha256']
    parent,_=restore_model(bnd,device);parent.eval().requires_grad_(False)
    ck=torch.load(a.candidate,map_location=device,weights_only=False)
    assert not ck['test_loaded'] and ck['parent_checkpoint_sha256']==bnd['parent_checkpoint']['sha256']
    model=ExpressionResponse(ResponseConfig(**ck['config']),ck['model']['feature_mean'],ck['model']['feature_std'],ck['model']['scales']).to(device)
    model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False)
    assert state_digest(parent.prior)==state_digest(model.prior)
    assert state_digest(parent.style)==state_digest(model.style)
    models={'parent':parent,'candidate':model}
    data,base=load_runtime(bnd,device);selected,pool,fold=select_queries(data,bnd['display_clips'].values())
    probes=[]
    for spec in bnd['probes']:
        assert sha(spec['path'])==spec['sha256']
        z=torch.load(spec['path'],map_location='cpu',weights_only=False)
        assert z['train_manifest_sha256']==bnd['data_manifest_sha256']
        assert not z.get('test_used_for_selection') and not z.get('generator_outputs_used_for_fitting')
        assert torch.equal(z['channel_support'].bool(),torch.tensor([True]*51+[False]))
        net=MotionEmotionProbe(z['feature_dim'],z['hidden'],8).eval()
        net.load_state_dict(z['model']);probes.append((net,z.get('feature_mask',torch.ones(z['feature_dim'],dtype=torch.bool))))
    def classify(x,valid):
        f=motion_features(torch.as_tensor(x)[:,:51],torch.as_tensor(valid))
        return [int(net(f[None,mask]).argmax(-1)) for net,mask in probes]
    people=sorted(data['refs']);assert len(people)==25
    all_query_ids={cid for q in data['splits'].values() for cid in q['clip_id']}
    for sid in people:
        refs=data['refs'][sid]
        assert len(set(refs['clip_id']))==2 and not all_query_ids.intersection(refs['clip_id'])
        assert all('_neutral_' in cid for cid in refs['clip_id'])
    # Keep canonical B0 batch context; only required batches enter the cache.
    selection={split:sorted({j for s,i,_ in selected if s==split
        for j in range(i//32*32,min((i//32+1)*32,len(data['splits'][split]['clip_id'])))}) for split in data['splits']}
    cache_base(data,base,device,selection)
    before={k:state_digest(v) for k,v in models.items()}|{'base':state_digest(base.stage1)}
    roles={int(data['splits'][s]['speaker_id'][i]):role for s,i,role in selected}
    targets={};ambiguous=set()
    for split,i,role in pool:
        q=data['splits'][split];k=(int(q['speaker_id'][i]),*key(q,i))
        if k in targets:ambiguous.add(k);continue
        batch=q.batch(torch.tensor([i]),keys=('motion','valid','channel_mask','times'))
        targets[k]=(summary(batch['motion'][0].numpy(),batch['valid'][0].numpy(),
            batch['channel_mask'][0].numpy(),batch['times'][0].numpy()),batch['channel_mask'][0].numpy())
    for k in ambiguous:targets.pop(k,None)
    anchors={};meta={}
    for sid in people:
        ref=data['refs'][sid];values=[];supports=[]
        for j in range(2):
            values.append(summary(ref['motion'][j].numpy(),ref['valid'][j].numpy(),ref['channel_mask'][j].numpy(),ref['times'][j].numpy())[0])
            supports.append(ref['channel_mask'][j].numpy())
        sup=np.stack(supports);anchors[sid]=(np.stack(values)*sup).sum(0)/sup.sum(0).clip(1)
        meta[str(sid)]=dict(speaker=ref['speaker'][0],role=roles[sid],references=ref['clip_id'])
    records=[];display={};scales=model.scales.cpu().numpy()
    # Fixed four donors span held-out and seen enrollment people. No donor is
    # selected by resemblance, metric score, or the candidate's response.
    donors=sorted(fold['held_sids'])+sorted(set(data['fit_sids'])-set(fold['held_sids']))[:2]
    for number,(split,i,role) in enumerate(selected):
        q=data['splits'][split];b=q.batch(torch.tensor([i]),device)
        sid=int(q['speaker_id'][i]);emotion=int(q['emotion_id'][i]);cid=q['clip_id'][i]
        valid=b['valid'][0].cpu().numpy();channels=b['channel_mask'][0].cpu().numpy();times=b['times'][0].cpu().numpy()
        gt=b['motion'][0].cpu().numpy();gt_stats=summary(gt,valid,channels,times)
        p=model.audio_prior(b['audio_features'],b['valid']);ref=reference_batch(data,b,device)
        style=model.encode_style(ref)['code']
        own={name:m.decode(b['b0'],p,style,b['valid'])[0] for name,m in models.items()}
        views={}
        for slot,start in [('A',0),('B',1)]:
            code=model.encode_style({k:v[:,start:start+1] for k,v in ref.items()})['code']
            views[slot]={name:m.decode(b['b0'],p,code,b['valid'])[0] for name,m in models.items()}
        rendered={}
        for target in people:
            code=model.encode_style(refs_for(data,b,device,target))['code']
            swapped={name:m.decode(b['b0'],p,code,b['valid'])[0] for name,m in models.items()}
            if target in donors:rendered[target]=swapped['candidate'].cpu().numpy()
            for name in models:
                for policy in ('raw','clip_all'):
                    convert=lambda v:(v.clamp(0,1) if policy=='clip_all' else v).cpu().numpy()
                    y=convert(own[name]);z=convert(swapped[name])
                    info=dict(clip_id=cid,source=sid,target=target,role=role,emotion=emotion,method=name,policy=policy)
                    if target==sid:
                        torch.testing.assert_close(own[name],swapped[name],rtol=0,atol=0)
                        records.append(dict(**info,kind='own_A_B',**change(convert(views['A'][name]),convert(views['B'][name]),valid,channels,times)))
                    else:
                        own_labels,swap_labels=classify(y,valid),classify(z,valid)
                        semantic={f'probe{k+1}_label_changed':float(x!=v) for k,(x,v) in enumerate(zip(own_labels,swap_labels))}
                        semantic.update({f'probe{k+1}_source_emotion_correct':float(v==emotion) for k,v in enumerate(swap_labels)})
                        records.append(dict(**info,kind='cross_AB',**change(y,z,valid,channels,times),**semantic))
                        donor=targets.get((target,*key(q,i)))
                        if donor is not None:
                            ch=channels&donor[1];ch[51]=False
                            records.append(dict(**info,kind='target_direction',**target_direction(
                                summary(y,valid,channels,times),summary(z,valid,channels,times),gt_stats,donor[0],
                                anchors[sid],anchors[target],ch,scales)))
        # One fixed held-out source per emotion, chosen by selection hash above.
        if cid in bnd['display_clips'].values():
            labels=['GT',meta[str(sid)]['speaker']+' AB']+[meta[str(t)]['speaker']+' AB' for t in donors]+['Own A','Own B']
            motions=[gt,own['candidate'].cpu().numpy()]+[rendered[t] for t in donors]+[views[s]['candidate'].cpu().numpy() for s in ('A','B')]
            references=['none','|'.join(meta[str(sid)]['references'])]+['|'.join(meta[str(t)]['references']) for t in donors]+meta[str(sid)]['references']
            path=out/'render_inputs'/f'multi_{emotion}.npz'
            display[emotion]=dict(path=path.name,**export(path,motions,valid,times,channels,cid,labels,references))
        write(out/'state.json',dict(status='auditing',processed=number+1,total=len(selected),test_loaded=False))
    grouped={}
    for method in models:
        for policy in ('raw','clip_all'):
            for kind in ('own_A_B','cross_AB','target_direction'):
                rows=[r for r in records if r['method']==method and r['policy']==policy and r['kind']==kind]
                grouped[f'{method}/{policy}/{kind}']=dict(**summaries(rows),
                    by_source_role={role:summaries([r for r in rows if r['role']==role]) for role in sorted(set(roles.values()))},
                    by_source_speaker={str(sid):summaries([r for r in rows if r['source']==sid]) for sid in people},
                    by_target_speaker={str(sid):summaries([r for r in rows if r['target']==sid]) for sid in people})
    assert before=={k:state_digest(v) for k,v in models.items()}|{'base':state_digest(base.stage1)}
    report=dict(schema='many_speaker_style_v1',people=25,selected_queries=len(selected),references=meta,
        candidate_checkpoint_sha256=sha(a.candidate),parent_checkpoint_sha256=bnd['parent_checkpoint']['sha256'],
        selection='one clip per person/emotion: original eight display clips when present, otherwise SHA-smallest; non-fit folds/development only',
        ambiguous_target_keys_excluded=len(ambiguous),results=grouped,render_inputs=display,
        frozen_exact=True,test_loaded=False,training_performed=False,
        limits=['Twenty seen people are held-sentence robustness, not unseen-person generalization.',
                'Five unseen people include two internal development people previously used for diagnostics.',
                'Different performances permit clip-statistic target direction only; no aligned counterfactual GT.',
                'Four frozen clip-stat probes are auxiliary emotion-preservation diagnostics, not temporal truth.',
                'Zero lag/closure agreement is not independent phoneme recognition.'])
    write(out/'report.json',finite(report))
    with gzip.open(out/'per_query.json.gz','wt',encoding='utf8') as f:json.dump(finite(records),f,allow_nan=False,separators=(',',':'))
    write(out/'manifest.json',dict(files={p.relative_to(out).as_posix():dict(sha256=sha(p),size=p.stat().st_size)
        for p in out.rglob('*') if p.is_file() and p.name!='state.json'},test_loaded=False))
    write(out/'state.json',dict(status='complete',selected_queries=len(selected),people=25,test_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');run(p.parse_args())
