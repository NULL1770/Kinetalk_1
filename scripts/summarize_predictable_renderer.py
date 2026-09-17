"""Audit fixed-budget generation with paired sentence bootstrap across noise seeds.

The noise seeds are repeats of the same clips, not extra independent samples.
Average clip sufficient statistics across seeds before resampling sentences.
This never tunes amplitude, selects a checkpoint, or chooses favorable seeds.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_ray import NUISANCE_CHANNELS_52
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.train_predictable_renderer import center, sha


def averaged_statistics(predictions, target, weight, channels):
    if not predictions:
        raise ValueError('Require at least one noise seed')
    # Averaging predictions would hide stochastic error. Average errors instead.
    return np.stack([clip_statistics(p, target, weight, channels) for p in predictions]).mean(0)


def verify_reference(first, second):
    for group, keys in [('q', ('motion','valid','channel_mask','emotion_id','speaker_id','times')),
                        ('base', ('b0',)), ('identity', ('code','baseline'))]:
        for key in keys:
            if not torch.equal(first[group][key], second[group][key]):
                raise ValueError(f'Unpaired reference {group}.{key}')
    for key in ('clip_id','sentence_id'):
        if first['q'][key] != second['q'][key]:
            raise ValueError(f'Unpaired metadata {key}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('rrr','pca','zero','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--samples',type=int,default=5000)
    p.add_argument('--seed',type=int,default=45)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use fresh audit output')
    torch.set_num_threads(4)
    paths={k:getattr(args,k) for k in ('rrr','pca','zero')}
    summaries={k:json.loads((v/'summary.json').read_text()) for k,v in paths.items()}
    refs={k:torch.load(v/'validation_reference.pt',map_location='cpu',weights_only=False) for k,v in paths.items()}
    reference=refs['rrr'];q=reference['q']
    for value in refs.values():verify_reference(reference,value)
    for key in ('minibatch_sha256','noise_time_choice_sha256','final_rng_sha256','frozen_after'):
        if len({s[key] for s in summaries.values()})!=1:raise ValueError(f'Unmatched experiment {key}')
    for s in summaries.values():
        if not s['frozen_unchanged'] or s['new_test_loaded']:raise ValueError('Frozen/test contract failed')
        if s['frozen_after']!=s['provenance']['frozen_before']:raise ValueError('Frozen hash changed')
    seeds=(42,123,2026)
    curves={};original=None
    for arm,path in paths.items():
        for seed in seeds:
            old=torch.load(path/f'original_seed{seed}_curves.pt',map_location='cpu',weights_only=False)
            if old['noise_seed']!=seed or old['decode_steps']!=12:raise ValueError('Unexpected generation settings')
            key=('original','full',seed)
            if key in curves and not torch.equal(curves[key],old['motion']['full']):
                raise ValueError('Original checkpoint predictions differ across arms')
            curves[key]=old['motion']['full']
            value=torch.load(path/f'final_seed{seed}_curves.pt',map_location='cpu',weights_only=False)
            if value['noise_seed']!=seed or value['decode_steps']!=12:raise ValueError('Unmatched noise/decode steps')
            for mode,pred in value['motion'].items():curves[(arm,mode,seed)]=pred
    common=q['channel_mask'].all(0)
    if not torch.equal(q['channel_mask'],common[None].expand_as(q['channel_mask'])):raise ValueError('Inconsistent observed channels')
    groups={'all_expression':[i for i in range(52) if common[i] and i not in NUISANCE_CHANNELS_52],
        'upper_expression':[5,6,12,13,41,42,43,44,45],'brows':list(range(41,46)),
        'eyes_expression':[5,6,12,13],'mouth':list(range(14,41)),'jaw17':[17]}
    groups={k:[i for i in values if common[i]] for k,values in groups.items()}
    target=q['motion'].float();weight=q['valid'].float()
    base=reference['base']['b0']+reference['identity']['baseline'][:,None]
    settings=sorted({(arm,mode) for arm,mode,seed in curves})
    stats={}
    scores={}
    populations={'nonneutral':(q['emotion_id']!=0).nonzero(as_tuple=True)[0].tolist(),
                 'neutral':(q['emotion_id']==0).nonzero(as_tuple=True)[0].tolist()}
    for arm,mode in settings:
        name=arm+':'+mode
        scores[name]={}
        for kind in ('raw_motion','centered_residual','centered_motion'):
            yy=target if kind=='raw_motion' else center(target-base if kind=='centered_residual' else target,weight)
            pp=[curves[(arm,mode,seed)] for seed in seeds]
            if kind!='raw_motion':pp=[center(v-base if kind=='centered_residual' else v,weight) for v in pp]
            scores[name][kind]={}
            for group,channels in groups.items():
                st=averaged_statistics(pp,yy,weight,channels)
                stats[(name,kind,group)]=st
                scores[name][kind][group]={pop:scalar_summary(st[ids]) for pop,ids in populations.items() if ids}
    specs=[('rrr:full',v) for v in ('rrr:zero','rrr:reverse','pca:full','zero:full','original:full')]
    specs += [('pca:full','pca:zero'),('pca:full','pca:reverse'),('zero:full','original:full'),('rrr:oracle','rrr:zero')]
    comparisons={}
    for left,right in specs:
        name=left+'__vs__'+right
        comparisons[name]={}
        for pop,ids in populations.items():
            comparisons[name][pop]={kind:{group:paired_summary(stats[(left,kind,group)],stats[(right,kind,group)],
                q['sentence_id'],ids,samples=args.samples,seed=args.seed) for group in groups}
                for kind in ('raw_motion','centered_residual','centered_motion')}
    per_seed={}
    for arm,s in summaries.items():
        per_seed[arm]={seed:{mode:{pop:{g:{'raw_mse':row[g]['raw_motion']['native_mse'],
            'residual_r2':row[g]['centered_residual']['r2_against_zero'],
            'raw_centered_corr':row[g]['raw_motion']['pooled_centered_correlation'],
            'velocity_mse':row[g]['velocity_mse_per_second']} for g in groups} for pop,row in values.items() if pop in populations and row is not None}
            for mode,values in seed_values.items()} for seed,seed_values in s['after'].items()}
    result={'schema':'predictable_renderer_paired_audit_v1','hashes':{arm:sha(path/'summary.json') for arm,path in paths.items()},
        'integrity':'Matched minibatches/noise/time/teacher-choice RNG, exact reference and original generation, frozen hashes verified',
        'seeds':seeds,'clips':len(target),'sentences':len(set(q['sentence_id'])),'bootstrap_samples':args.samples,
        'noise_aggregation':'Average sufficient statistics per clip, then resample whole sentence clusters; seeds not treated as additional samples',
        'caveat':'Exploratory development functionality test, no independent perceptual or audiovisual lip-sync evaluator; oracle reads motion',
        'groups':groups,'scores':scores,'comparisons':comparisons,'per_seed':per_seed,
        'frozen_teacher_accuracy':{a:{s:{m:v.get('frozen_teacher_emotion_accuracy') for m,v in modes.items()} for s,modes in report['after'].items()} for a,report in summaries.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    for name in scores:
        print(json.dumps({'setting':name,'nonneutral':{g:scores[name]['centered_residual'][g]['nonneutral']['r2_against_zero'] for g in ('upper_expression','brows','eyes_expression','mouth','jaw17')},
            'neutral_mouth_mse':scores[name]['raw_motion']['mouth']['neutral']['native_mse']}),flush=True)
    for name,values in comparisons.items():
        row=values['nonneutral']['centered_residual']['upper_expression']
        print(json.dumps({'comparison':name,'upper_delta_r2':row['r2_improvement'],'ci95':row['r2_improvement_ci95']}),flush=True)


if __name__=='__main__':main()
