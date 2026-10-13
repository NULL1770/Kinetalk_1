"""Bounded, source-disjoint real data pilot; TRAIN fit and unseen validation.

MEAD references use different sentence and emotion from the query. HDTF has
unknown emotion; only buffered nonoverlapping same-person windows are used.
No test/quarantine file is read. Existing B0 supplies h0 as an auxiliary anchor,
not the v7 output base. The pilot has fewer sources than the formal dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import configure, load_runtime, development_fold
from scripts.train_full_staged import base_forward
from kinetalk_v2.hdtf_dataset import HDTFSplit

def select(q, pool, speakers, count, rng):
    groups={}
    for i in pool:
        sid,emo=int(q['speaker_id'][i]),int(q['emotion_id'][i])
        if sid in speakers and bool(q['valid'][i].any()):
            groups.setdefault((sid,emo),[]).append(i)
    chosen=[]
    for key in sorted(groups):
        values=groups[key];rng.shuffle(values);chosen+=values[:count]
    return chosen

def ref_ids(q, i, pool, rng):
    candidates=[j for j in pool if int(q['speaker_id'][j])==int(q['speaker_id'][i])
                and q['clip_id'][i]!=q['clip_id'][j] and q['sentence_id'][i]!=q['sentence_id'][j]
                and int(q['emotion_id'][i])!=int(q['emotion_id'][j])]
    rng.shuffle(candidates)
    chosen=[]
    for j in candidates:
        if chosen and (q['sentence_id'][j]==q['sentence_id'][chosen[0]] or
                       int(q['emotion_id'][j])==int(q['emotion_id'][chosen[0]])):
            continue
        chosen.append(j)
        if len(chosen)==2:return chosen
    raise ValueError('No legal cross-emotion supports for '+q['clip_id'][i])

def main(a):
    configure(a.seed);rng=random.Random(a.seed);device=torch.device(a.device)
    binding=json.loads(Path(a.binding).read_text())
    data,base=load_runtime(binding,device);q=data['splits']['train'];fold=development_fold(q,data['fit_sids'])
    people=sorted(set(int(q['speaker_id'][i]) for i in fold['train']))[:a.speakers]
    roles={'fit':(q,select(q,fold['train'],people,a.per_emotion,rng),fold['train']),
           'held_sentence':(q,select(q,fold['sentence_dev'],people,1,rng),fold['train'])}
    val=data['splits']['validation']
    roles['unseen_speaker']=(val,select(val,range(len(val['_lengths'])),data['dev_sids'],1,rng),range(len(val['_lengths'])))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    fmean=data['feature_stats']['mean'];fstd=data['feature_stats']['std']
    records=[]
    def cropped(split,i,role,reference=False):
        b=split.batch(torch.tensor([i]),device)
        valid=b['valid'][0].bool();n=len(valid);width=a.length
        start=rng.randint(0,max(0,n-width)) if role=='fit' else max(0,(n-width)//2)
        # B0 evaluated on full native clock before cropping, for fair h0 anchor.
        with torch.no_grad():old=base_forward(base,b['content'],b['valid'])
        values={}
        for key,x in [('audio_features',b['audio_features'][0]),('motion',b['motion'][0]),
                      ('valid',valid),('times',b['times'][0]),('h0',old['h0'][0]),('b0',old['b0'][0])]:
            x=x[start:start+width].cpu();shape=(width,*x.shape[1:]);y=torch.zeros(shape,dtype=x.dtype)
            y[:len(x)]=x
            if key=='audio_features':y=torch.where(values.get('valid',torch.ones(width,dtype=torch.bool))[:,None],(y-fmean)/fstd,0.)
            values[key]=y.numpy()
        values['audio_features'][~values['valid']]=0
        values['channel_mask']=b['channel_mask'][0].cpu().numpy()
        return values,{'clip_id':split['clip_id'][i],'sentence_id':split['sentence_id'][i],
                       'speaker':split['speaker'][i],'start_frame':start,'end_frame':min(start+width,n)}
    def pack_item(split,i,pool,role,hdtf=False):
        use=split.reference_candidates(i) if hdtf else ref_ids(split,i,pool,rng)
        if hdtf:
            rng.shuffle(use);use=use[:2]
            if not use:return None
            if len(use)==1:use=use+[use[0]]
        values,meta=cropped(split,i,role)
        refs=[cropped(split,j,role,True) for j in use]
        if hdtf and use[0]==use[1]:
            refs[1][0]['valid'][:]=False
        for key in ('audio_features','motion','valid','channel_mask'):
            values['ref_'+key]=np.stack([v[key] for v,_ in refs])
        values['speaker_id']=np.array(int(split['speaker_id'][i]),np.int64)
        values['emotion_id']=np.array(-1 if hdtf else int(split['emotion_id'][i]),np.int64)
        values['intensity']=np.array(-1. if hdtf else float(split['intensity_id'][i])/3.,np.float32)
        values['intensity_valid']=np.array(False if hdtf else bool(split['intensity_valid'][i]))
        values['dataset_id']=np.array(1 if hdtf else 0,np.int64)
        meta.update(role=role,references=[m for _,m in refs],dataset='HDTF' if hdtf else 'MEAD',
                    cross_emotion_reference=not hdtf, independent_reference=True)
        return values,meta
    packs={}
    for role,(split,ids,pool) in roles.items():
        if role=='unseen_speaker':pool=[i for i in pool if i not in ids]
        items=[pack_item(split,i,list(pool),role) for i in ids]
        packs[role]=items;print(json.dumps({'role':role,'clips':len(items)}),flush=True)
    if a.hdtf_manifest:
        h=HDTFSplit(a.hdtf_manifest,a.hdtf_features,split='train',motion_policy='raw')
        sources=sorted(set(h['source_id']))[:a.hdtf_sources]
        for source in sources:
            ids=[i for i,row in enumerate(h.rows) if row['source_id']==source]
            for i in ids[::max(1,len(ids)//a.hdtf_per_source)][:a.hdtf_per_source]:
                item=pack_item(h,i,ids,'fit',True)
                if item:packs['fit'].append(item)
    for role,items in packs.items():
        arrays={key:np.stack([v[key] for v,_ in items]) for key in items[0][0]}
        np.savez(out/(role+'.npz'),**arrays)
        for j,(_,m) in enumerate(items):records.append({**m,'row':j})
    provenance={'schema':'v7_real_pilot_data_v2','test_loaded':False,'fit_speakers':people,
                'roles':{k:len(v) for k,v in packs.items()},'length':a.length,'seed':a.seed,
                'feature_normalization':'existing TRAIN-only feature stats','h0_dimension':int(packs['fit'][0][0]['h0'].shape[-1]),
                'query_reference_sentence_disjoint_mead':True,'hdtf_emotion_supervision':False,
                'hdtf_motion_policy':'raw','hdtf_style_limit':'same-source buffered windows; not cross-source or cross-emotion proof',
                'records':records}
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in provenance.items() if k!='records'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--speakers',type=int,default=8);p.add_argument('--per-emotion',type=int,default=4)
    p.add_argument('--length',type=int,default=96);p.add_argument('--hdtf-manifest');p.add_argument('--hdtf-features')
    p.add_argument('--hdtf-sources',type=int,default=8);p.add_argument('--hdtf-per-source',type=int,default=4)
    main(p.parse_args())
