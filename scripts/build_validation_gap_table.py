"""Re-score existing native validation predictions with one frozen metric protocol.

This development artifact is not a paper test table or an official reproduction.
It never loads a sealed/test split and never fits a classifier or model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_probe import MotionEmotionProbe, classification_metrics, motion_features
from scripts.arkit_benchmark_report import score_fullface
from scripts.audit_matched_motion_curves import region_values, RC
from scripts.diagnose_flow_sampling import sha
from scripts.evaluate_vertex_lve import evaluate as vertex_evaluate
from scripts.packed_trainval_cache import load_packed

VERTEX=['lve_mean_mm_mean','eve_mean_mm_mean','vertex_lve_sqrt_mean',
        'eye_forehead_eve_sqrt_mean','facediffuser_lve_sq_mean','vertex_fdd_absolute_mm2_mean']
COEFFICIENT=['arkit_mbe','arkit_lbe','arkit_fdd_signed','arkit_fdd_absolute']


def curve_clip(curves, index, length, offsets):
    """Return native frames, preserving gaps, from either historical storage schema."""
    compact=curves.get('compaction_schema')=='native_curve_pack_v1'
    if compact:
        if curves['native_lengths'][index]!=length:raise ValueError('Curve native length differs')
        sl=slice(offsets[index],offsets[index+1])
        item={k:curves[k][sl] for k in ('target','valid','times')}
        item['predictions']=[v[sl] for v in curves['predictions'].values()]
    else:
        item={k:curves[k][index,:length] for k in ('target','valid','times')}
        item['predictions']=[v[index,:length] for v in curves['predictions'].values()]
    item['channel_mask']=curves['channel_mask'][index]
    return item


def check_contract(item, target):
    for name in ('valid','times','channel_mask'):
        torch.testing.assert_close(item[name],target[name],rtol=0,atol=0,equal_nan=True)
    if 'target' in item:
        torch.testing.assert_close(item['target'],target['motion'],rtol=0,atol=0,equal_nan=True)
    for pred in item['predictions']:
        if pred.shape!=target['motion'].shape:raise ValueError('Prediction native shape differs')


@torch.no_grad()
def run(a):
    if a.output.exists():raise FileExistsError('Fresh validation gap output required')
    torch.set_num_threads(2)
    data=load_packed(a.data,materialize=False,with_refs=False)
    assert set(data['splits'])=={'train','validation'} and not data['provenance']['test_loaded']
    manifest=data['provenance']['manifest_sha256'];q=data['splits']['validation']
    assert len(q['_lengths'])==1367 and len(data['splits']['train']['_lengths'])==12536
    support=torch.tensor([True]*51+[False])
    assert torch.equal(data['splits']['train']['channel_mask'].any(0),support)
    spec=json.loads(a.spec.read_text());assert spec['test_loaded'] is False
    methods=spec['methods'];probes=[];probe_meta=[];names=data['config']['data']['emotion_classes']
    for path in a.probe:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['train_manifest_sha256']==manifest and ck['classes']==names
        assert not ck.get('test_used_for_selection') and not ck.get('generator_outputs_used_for_fitting')
        assert torch.equal(ck['channel_support'].bool(),support)
        model=MotionEmotionProbe(ck['feature_dim'],ck['hidden'],len(names)).eval()
        model.load_state_dict(ck['model'],strict=True)
        probes.append((model,ck.get('feature_mask',torch.ones(6*int(support.sum()),dtype=torch.bool))))
        probe_meta.append({'path':str(path),'sha256':sha(path),'kind':ck['kind']})
    with np.load(a.rig,allow_pickle=False) as z:rig={k:np.array(z[k]) for k in z.files}
    if rig['blendshape_deltas'].shape[0]!=52:raise ValueError('ARKit52 rig required')
    ids=list(q['clip_id']);labels=q['emotion_id'].cpu();targets=[]
    for i,n in enumerate(q['_lengths']):
        b=q.batch(torch.tensor([i]),keys=('motion','valid','times','channel_mask'))
        targets.append({k:v[0,:int(n)] if k!='channel_mask' else v[0] for k,v in b.items()})
    gt_features=torch.stack([motion_features(t['motion'][:,support],t['valid']) for t in targets])
    report={'schema':'same_protocol_validation_gap_v1','test_loaded':False,'training_performed':False,
            'scope':'1367 development validation clips; not final test/paper table',
            'data_manifest_sha256':manifest,'rig_sha256':sha(a.rig),'probes':probe_meta,
            'aggregation':'equal native clips, mean per-draw metrics/F1; no best draw or mean-motion inference',
            'coefficient_support':support.tolist(),'classes':names,
            'gt_probes':[classification_metrics(labels,m(gt_features[:,fm]).argmax(-1),names) for m,fm in probes],
            'method_adaptation_caution':'Shared-audio/ARKit adaptations, not original official end-to-end reproductions.',
            'methods':{},'sources':{p:sha(Path(__file__).resolve().parents[1]/p) for p in (
                'scripts/build_validation_gap_table.py','scripts/evaluate_vertex_lve.py',
                'scripts/arkit_benchmark_report.py','scripts/evaluate_arkit_literature_metrics.py',
                'kinetalk_b0/emotion_probe.py')}}
    arrays={'clip_id':np.asarray(ids),'labels':labels.numpy()};a.output.mkdir(parents=True)
    columns=COEFFICIENT+VERTEX+['mouth_displacement_mse','jaw_centered_correlation','jaw_q90_q10']
    report['metric_columns']=columns
    for method in methods:
        name=method['name'];path=Path(method['predictions']);checkpoint=Path(method['checkpoint'])
        assert 'sealed' not in str(path).lower() and 'test' not in str(path).lower()
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        pred_data=torch.load(path,map_location='cpu',weights_only=False)
        meta={**method,'checkpoint_sha256':sha(checkpoint),'prediction_sha256':sha(path),
              'historical_prediction_checkpoint_hash_recorded':False}
        if method['storage']=='baseline':
            protocol=json.loads((checkpoint.parent/'protocol.json').read_text())
            completed=json.loads((checkpoint.parent/'training_complete.json').read_text())
            assert ck['protocol']==protocol and not protocol['test_loaded'] and not protocol['smoke']
            assert protocol['data']['manifest_sha256']==manifest and completed['validation_clips']==1367
            assert completed['train_clips']==12536 and completed['epochs']==method['epochs']
            assert pred_data['test_loaded'] is False and pred_data['manifest_sha256']==manifest
            assert pred_data['scope']=='development validation' and set(pred_data['clips'])==set(ids)
            # The old artifact did not record its checkpoint hash. Verify its
            # first original batch numerically and retain this limitation.
            from scripts.baseline_training import make_model,predict
            # Match historical CUDA/TF32: autoregression amplifies backend rounding.
            torch.backends.cuda.matmul.allow_tf32=True
            model=make_model(method['method'],ck['config']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
            batch=q.batch(torch.arange(protocol['batch_size']),'cuda')
            numerical=predict(model,method['method'],batch)[0].cpu()
            maximum=0.
            for j in range(protocol['batch_size']):
                n=int(q['_lengths'][j]);original=pred_data['clips'][ids[j]]['prediction']
                maximum=max(maximum,float((numerical[j,:n]-original).abs().max()))
            if maximum>1e-3:raise ValueError(f'{name}: saved checkpoint spot-check mismatch {maximum}')
            meta['checkpoint_spot_check']={'clips':protocol['batch_size'],'device':'cuda','allow_tf32':True,
                                           'max_abs_difference_same_backend':maximum}
            del model,batch,numerical
            def get_item(i,n):
                item=pred_data['clips'][ids[i]]
                return {**item,'predictions':[item['prediction']]}
            draws=[42]
        else:
            assert set(pred_data['clip_id'])==set(ids) and pred_data['noise_seeds']==[42,123,2026]
            assert list(pred_data['predictions'])==['42/full','123/full','2026/full']
            lookup={cid:i for i,cid in enumerate(pred_data['clip_id'])}
            offsets=np.r_[0,np.cumsum(pred_data.get('native_lengths',[]))]
            if method['storage']=='kinetalk':
                complete=json.loads((checkpoint.parent/'complete.json').read_text())
                assert ck['stage']=='audio' and ck['data_manifest_sha256']==manifest and ck['test_loaded'] is False
                assert complete['final_sha256']==sha(checkpoint) and complete['curves_sha256']==sha(path)
                meta['historical_prediction_checkpoint_hash_recorded']=True
            else:
                assert ck['protocol']['data']['manifest_sha256']==manifest and ck['protocol']['test_loaded'] is False
                assert ck['protocol']['epochs']==method['epochs'] and not ck['protocol']['smoke']
                meta['condition_source']=ck['protocol']['condition_source']
                meta['historical_binding_limitation']='Curve file has no checkpoint hash; validated native target/mask/clock and final protocol, no full resampling.'
            def get_item(i,n):return curve_clip(pred_data,lookup[ids[i]],n,offsets)
            draws=[42,123,2026]
        del ck
        results={}
        for policy in ('raw','clip_all'):
            per_draw_metrics=[[] for _ in draws];per_draw_features=[[] for _ in draws]
            for i,t in enumerate(targets):
                item=get_item(i,len(t['valid']));check_contract(item,t)
                for j,raw in enumerate(item['predictions']):
                    pred=raw if policy=='raw' else raw.clamp(0,1)
                    scored=score_fullface(pred.numpy()[None],{'target52':t['motion'].numpy(),
                        'valid':t['valid'].numpy(),'times':t['times'].numpy(),
                        'channel_mask':np.broadcast_to(t['channel_mask'].numpy(),t['motion'].shape)})
                    vertex=vertex_evaluate(pred.numpy(),t['motion'].numpy(),t['valid'].numpy(),t['channel_mask'].numpy(),
                        rig['neutral_vertices'],rig['blendshape_deltas'],lip_mask=rig['lip_mask'],
                        expression_mask=rig['expression_mask'],fdd_mask=rig.get('fdd_mask'),
                        coordinate_scale_to_mm=float(rig['coordinate_scale_to_mm']),
                        coordinate_unit=str(rig['coordinate_unit'].item()),coefficient_support=support.numpy())
                    mouth=region_values(pred,t['motion'],t['valid'],t['channel_mask'],list(range(14,41)))
                    jaw=region_values(pred,t['motion'],t['valid'],t['channel_mask'],[17])
                    per_draw_metrics[j].append([scored['metrics'][k]['value'] for k in COEFFICIENT]+
                        [vertex[k] for k in VERTEX]+[mouth[RC.index('displacement_mse')],
                        jaw[RC.index('centered_correlation')],jaw[RC.index('pred_q90_q10')]])
                    per_draw_features[j].append(motion_features(pred[:,support],t['valid']))
                if i%300==0:print(json.dumps({'event':'scoring','method':name,'policy':policy,'clips':i+1}),flush=True)
            metrics=np.asarray(per_draw_metrics,dtype=np.float64)
            feature=torch.stack([torch.stack(f) for f in per_draw_features])
            classes=[torch.stack([m(f[:,fm]).argmax(-1) for f in feature]) for m,fm in probes]
            per_draw_probes=[[classification_metrics(labels,p[j],names) for p in classes] for j in range(len(draws))]
            arrays[name+'__'+policy+'__metrics']=metrics
            arrays[name+'__'+policy+'__class_predictions']=torch.stack(classes).numpy()
            results[policy]={'metrics':{k:float(np.nanmean(metrics[...,j])) for j,k in enumerate(columns)},
                             'F1':np.mean([[p['macro_f1'] for p in row] for row in per_draw_probes],axis=0).tolist(),
                             'per_draw_probes':per_draw_probes}
            report['vertex_protocol']={k:vertex[k] for k in ('rig_identity','region_masks','coordinate_units',
                                                             'coordinate_scale_to_mm','definitions')}
        report['methods'][name]={'provenance':meta,'draw_seeds':draws,'results':results}
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        del pred_data
    np.savez_compressed(a.output/'per_clip_metrics.npz',**arrays)
    text=['# Validation comparison gap (not a final test table)','',
          'Same 1,367 clips / native masks / fixed rig / frozen real-TRAIN probes. All baselines are declared shared-audio ARKit adaptations.',
          '','Clipped-to-[0,1] for every method; raw results retained in report.json.',
          '','| Method | MBE ↓ | LBE ↓ | Lip mean mm ↓ | Expression mean mm ↓ | Original F1 128/64 ↑ | Mouth displacement MSE ↓ |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in report['methods'].items():
        v=row['results']['clip_all'];m=v['metrics'];f=v['F1']
        text.append(f"| {name} | {m['arkit_mbe']:.4f} | {m['arkit_lbe']:.4f} | {m['lve_mean_mm_mean']:.3f} | {m['eve_mean_mm_mean']:.3f} | {f[0]:.4f}/{f[1]:.4f} | {m['mouth_displacement_mse']:.7f} |")
    text+=['','Stochastic methods use mean scores of three fixed draws, never best draw or mean generated motion.',
           'Auxiliary stable probes, alternate/max/squared vertex metrics and coefficient/vertex FDD are retained in the full report.',
           'Historical baseline prediction files did not store a checkpoint hash. The three deterministic methods passed a fixed first-batch CUDA/TF32 checkpoint spot check; FaceDiffuser retains a provenance limitation until full resampling.',
           'These scores do not establish official benchmark SOTA.']
    (a.output/'README.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    (a.output/'complete.json').write_text(json.dumps({'test_loaded':False,'report_sha256':sha(a.output/'report.json'),
                                                   'per_clip_sha256':sha(a.output/'per_clip_metrics.npz')},indent=2))
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','spec','rig','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--probe',type=Path,nargs='+',required=True);run(p.parse_args())
