"""Frozen readout comparison on real coefficient means, not teacher coordinates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_reference_condition_predictability import fit_predict
from scripts.diagnose_flow_sampling import sha
from scripts.packed_trainval_cache import load_packed


def native_offset_mean(motion, valid, channels, anchor, anchor_valid):
    observed=valid[...,None]&channels[:,None,:]&anchor_valid[:,None,:]
    if not torch.isfinite(motion[observed]).all() or not torch.isfinite(anchor[channels&anchor_valid]).all():
        raise ValueError('Finite observed target and independent anchor required')
    count=observed.sum(1)
    clean=torch.where(observed,motion-anchor[:,None],0.)
    return clean.sum(1)/count.clamp_min(1),count>0


def metrics(pred,target,support):
    delta=(pred-target)**2
    def region(cols):
        cols=[i for i in cols if support[i]]
        return float(delta[:,cols].mean())
    return dict(mse=region(range(52)),mouth_mse=region(range(14,41)),
                brow_mse=region(range(41,46)),eyes_mse=region(range(14)),
                jaw_mean_mse=region((17,)))


@torch.no_grad()
def run(a):
    if a.output.exists():raise FileExistsError('Fresh diagnostic required')
    torch.set_num_threads(2)
    raw_report=json.loads((a.raw/'report.json').read_text())
    pool_report=json.loads((a.pooling/'report.json').read_text())
    assert not raw_report['test_loaded'] and not pool_report['test_loaded']
    assert raw_report['source_checkpoint_sha256']==pool_report['source_checkpoint_sha256']
    raw=np.load(a.raw/'features_predictions.npz',allow_pickle=False)
    pool=np.load(a.pooling/'features_predictions.npz',allow_pickle=False)
    z=np.load(a.conditions/'per_clip_conditions.npz',allow_pickle=False)
    condition_report=json.loads((a.conditions/'report.json').read_text())
    assert condition_report['checkpoint_sha256']==raw_report['source_checkpoint_sha256']
    assert not condition_report['test_loaded']
    data=load_packed(a.data,materialize=False,with_refs=False)
    manifest=data['provenance']['manifest_sha256']
    assert set(data['splits'])=={'train','validation'} and not data['provenance']['test_loaded']
    assert manifest==raw_report['data_manifest_sha256']==pool_report['data_manifest_sha256']==condition_report['data_manifest_sha256']
    arrays={};masks={};speakers={}
    for role,q in data['splits'].items():
        ids=np.asarray(q['clip_id'])
        np.testing.assert_array_equal(ids,z[role+'__clip_id'])
        np.testing.assert_array_equal(ids,pool[role+'_clip_id'])
        np.testing.assert_array_equal(q['speaker_id'].numpy(),z[role+'__speaker'])
        target=[];seen=[]
        for ix in torch.arange(len(q['_lengths'])).split(32):
            b=q.batch(ix,'cpu',keys=('motion','valid','channel_mask','anchors','anchor_valid'))
            mean,mask=native_offset_mean(b['motion'],b['valid'],b['channel_mask'],b['anchors'],b['anchor_valid'])
            target.append(mean.numpy());seen.append(mask.numpy())
        arrays[role+'__target_offset_mean']=np.concatenate(target)
        masks[role]=np.concatenate(seen);speakers[role]=q['speaker_id'].numpy()
    support=masks['train'].all(0)
    assert int(support.sum())==51 and masks['validation'][:,support].all()
    assert np.array_equal(support,np.arange(52)!=51), 'Expected fixed TRAIN support'
    report=dict(schema='physical_mean_predictability_v1',test_loaded=False,
        generator_training_performed=False,generation_performed=False,real_train_ridge_fitted=True,
        source_checkpoint_sha256=raw_report['source_checkpoint_sha256'],data_manifest_sha256=manifest,
        raw_npz_sha256=sha(a.raw/'features_predictions.npz'),pooling_npz_sha256=sha(a.pooling/'features_predictions.npz'),
        script_sha256=sha(Path(__file__)),ridge_lambda=.001,feature_std_floor=1e-4,
        target='Observed real clip coefficient mean minus independent enrollment anchor52',
        scope='Extra physical readout only; frozen encoder has seen all TRAIN speakers; no deployable correction',
        folds='TRAIN speaker_id modulo5; validation never fit',support=support.tolist(),results={},
        decision_gate='All five TRAIN-fold total MSEs must improve and overall mouth/brow MSE must not regress')
    y=arrays['train__target_offset_mean'].astype(np.float64)
    vy=arrays['validation__target_offset_mean'].astype(np.float64)
    folds=speakers['train']%5
    for extra in (False,True):
        name='hidden_mean' if not extra else 'hidden_plus_raw_emotion'
        def features(role):
            x=pool[role+'__mean']
            if extra:x=np.concatenate([x,raw[role+'__raw_emotion_mean']],1)
            return x.astype(np.float64)
        x,vx=features('train'),features('validation');out=np.empty_like(y);rows=[]
        for fold in range(5):
            query=folds==fold;out[query]=fit_predict(x[~query],y[~query],x[query])
            rows.append(dict(fold=fold,query_speakers=np.unique(speakers['train'][query]).tolist(),
                             query_clips=int(query.sum()),**metrics(out[query],y[query],support)))
        vp=fit_predict(x,y,vx)
        report['results'][name]=dict(features=x.shape[1],fold_rows=rows,
            speaker_fold_out_of_fit_train=metrics(out,y,support),all_train_fit_validation=metrics(vp,vy,support))
        arrays[name+'__train_prediction']=out;arrays[name+'__validation_prediction']=vp
        print(json.dumps(dict(event='physical_result',method=name,**report['results'][name])),flush=True)
    base,candidate=(report['results'][k] for k in ('hidden_mean','hidden_plus_raw_emotion'))
    passed=all(e['mse']<c['mse'] for c,e in zip(base['fold_rows'],candidate['fold_rows']))
    passed=passed and all(candidate['speaker_fold_out_of_fit_train'][k]<=base['speaker_fold_out_of_fit_train'][k]
                          for k in ('mouth_mse','brow_mse'))
    report['passes_train_gate']=passed
    report['decision']='Physical-target structural trial candidate only' if passed else 'No raw-mean bypass training; this fixed readout failed'
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'targets_predictions.npz',**arrays)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(event='complete',passes_train_gate=passed)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','raw','pooling','conditions','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
