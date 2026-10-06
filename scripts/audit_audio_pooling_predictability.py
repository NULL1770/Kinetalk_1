"""Fixed TRAIN speaker-fold test of frozen acoustic mean vs mean/std readouts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts.audit_reference_condition_predictability import fit_predict,score
from scripts.diagnose_flow_sampling import sha
from scripts.packed_trainval_cache import load_packed
from scripts.train_full_staged import audio_affect


def masked_population_std(hidden,valid,mean):
    if hidden.ndim!=3 or valid.shape!=hidden.shape[:2] or mean.shape!=(len(hidden),hidden.shape[-1]):
        raise ValueError('Native hidden/valid/mean shape mismatch')
    if not valid.any(1).all() or not torch.isfinite(hidden[valid]).all():
        raise ValueError('Finite nonempty valid hidden frames required')
    centered=torch.where(valid[...,None],hidden-mean[:,None],0.)
    return (centered.square().sum(1)/valid.sum(1,keepdim=True)).clamp_min(0.).sqrt()


@torch.no_grad()
def run(a):
    if a.output.exists():raise FileExistsError('Fresh pooling diagnostic required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    ckpath=a.run_root/'audio/final.pt'
    assert sha(ckpath)==a.checkpoint_sha256
    cached_report=json.loads((a.source/'report.json').read_text())
    assert cached_report['checkpoint_sha256']==a.checkpoint_sha256 and not cached_report['test_loaded']
    assert cached_report['training_performed'] is False
    z=np.load(a.source/'per_clip_conditions.npz',allow_pickle=False)
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    recipe=json.loads((a.run_root/'provenance.json').read_text())
    assert ck['recipe_sha256']==recipe['recipe_sha256'] and not recipe['recipe']['test_loaded']
    data=load_packed(a.data,materialize=False,with_refs=False)
    manifest=data['provenance']['manifest_sha256']
    assert set(data['splits'])=={'train','validation'} and not data['provenance']['test_loaded']
    assert manifest==ck['data_manifest_sha256']==cached_report['data_manifest_sha256']
    audio=SlowStateAffect(ck['feature_stats']['mean'],ck['feature_stats']['std'],
                         stride=recipe['recipe']['args']['stride']).to(a.device).eval()
    audio.load_state_dict(ck['audio'],strict=True);audio.requires_grad_(False)
    captured={}
    def hidden_hook(module,args,output):captured['hidden']=output
    def mean_hook(module,args):captured['mean']=args[0]
    hh=audio.blocks[-1].register_forward_hook(hidden_hook)
    mh=audio.global_head.register_forward_pre_hook(mean_hook)
    arrays={};parity={}
    for role,q in data['splits'].items():
        np.testing.assert_array_equal(np.asarray(q['clip_id']),z[role+'__clip_id'])
        np.testing.assert_array_equal(q['speaker_id'].numpy(),z[role+'__speaker'])
        assert len(q['_lengths'])==(12536 if role=='train' else 1367)
        means=[];stds=[];codes=[]
        for batch_no,ids in enumerate(torch.arange(len(q['_lengths'])).split(16)):
            b=q.batch(ids,a.device,keys=('audio_features','valid'))
            output=audio_affect(audio,b['audio_features'],b['valid'])
            mean=captured['mean'];std=masked_population_std(captured['hidden'],b['valid'],mean)
            means.append(mean.cpu().numpy());stds.append(std.cpu().numpy());codes.append(output['global'].cpu().numpy())
            if batch_no%100==0:print(json.dumps({'event':'pooling','role':role,'clips':int(ids[-1])+1}),flush=True)
        arrays[role+'__mean']=np.concatenate(means);arrays[role+'__std']=np.concatenate(stds)
        code=np.concatenate(codes);diff=code-z[role+'__audio_global']
        np.testing.assert_allclose(code,z[role+'__audio_global'],rtol=1e-5,atol=1e-4)
        parity[role]={'max_abs_code_difference':float(np.max(np.abs(diff)))}
    hh.remove();mh.remove();del audio
    y=z['train__teacher_global'].astype(np.float64);vy=z['validation__teacher_global'].astype(np.float64)
    speaker=z['train__speaker'];folds=speaker%5
    report={'schema':'frozen_audio_pooling_predictability_v1','test_loaded':False,
            'generator_training_performed':False,'generation_performed':False,'real_train_ridge_fitted':True,
            'ridge_lambda':.001,'feature_std_floor':1e-4,'source_checkpoint_sha256':a.checkpoint_sha256,
            'source_npz_sha256':sha(a.source/'per_clip_conditions.npz'),'data_manifest_sha256':manifest,
            'script_sha256':sha(Path(__file__)),'scope':'Extra readouts only; encoder already trained on all TRAIN speakers',
            'folds':'TRAIN speaker_id modulo5; no validation fit','native_gap_policy':'All valid frames, never gap compression',
            'cached_audio_code_parity':parity,'results':{}}
    for use_std in (False,True):
        name='mean_and_std' if use_std else 'mean_only'
        def features(role):
            mean=arrays[role+'__mean']
            return np.concatenate([mean,arrays[role+'__std']],axis=1).astype(np.float64) if use_std else mean.astype(np.float64)
        x,vx=features('train'),features('validation');out=np.empty_like(y);rows=[]
        for fold in range(5):
            query=folds==fold
            out[query]=fit_predict(x[~query],y[~query],x[query])
            rows.append({'fold':fold,'query_speakers':np.unique(speaker[query]).tolist(),
                         'query_clips':int(query.sum()),**score(out[query],y[query])})
        vp=fit_predict(x,y,vx)
        report['results'][name]={'speaker_fold_out_of_fit_train':score(out,y),'fold_rows':rows,
                                 'all_train_fit_validation':score(vp,vy),'features':x.shape[1]}
        arrays[name+'__train_fold_prediction']=out;arrays[name+'__validation_prediction']=vp
        print(json.dumps({'event':'ridge_result','method':name,**report['results'][name]}),flush=True)
    baseline=report['results']['mean_only'];candidate=report['results']['mean_and_std']
    passed=all(e['mse']<c['mse'] for c,e in zip(baseline['fold_rows'],candidate['fold_rows']))
    report['improves_every_train_speaker_fold']=passed
    report['decision']='Structure candidate only, no promotion or deployable ridge' if passed else 'Do not train pooled-std branch based on validation alone'
    a.output.mkdir(parents=True)
    arrays.update(train_clip_id=z['train__clip_id'],validation_clip_id=z['validation__clip_id'])
    np.savez_compressed(a.output/'features_predictions.npz',**arrays)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run-root','data','source','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--device',default='cuda');run(p.parse_args())
