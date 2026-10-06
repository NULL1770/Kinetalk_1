"""Replay matched saved validation curves: frozen probes, regions and diversity."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_probe import MotionEmotionProbe, classification_metrics, motion_features
from scripts.diagnose_flow_sampling import clip_metrics, sha
from scripts.packed_trainval_cache import load_packed
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES

REGIONS = {'eyes': list(range(14)), 'mouth_jaw': list(range(14,41)),
           'brows': list(range(41,46)), 'cheek_nose': list(range(46,51)), 'jawOpen': [17]}
RC = ['mse','mean_bias_mse','centered_mse','pred_centered_rms','gt_centered_rms',
      'centered_correlation','displacement_mse','pred_q90_q10','gt_q90_q10']
MC = ['MBE','LBE','outside_fraction','mouth_absolute_rms',
      'mouth_centered_rms','mouth_displacement_mse']

def region_values(p,t,v,ch,ix):
    ix=[j for j in ix if ch[j]]
    x,y=p[v][:,ix],t[v][:,ix]
    xc,yc=x-x.mean(0),y-y.mean(0)
    adj=v[1:] & v[:-1]
    dx,dy=(p[1:]-p[:-1])[adj][:,ix],(t[1:]-t[:-1])[adj][:,ix]
    den=xc.square().sum().sqrt()*yc.square().sum().sqrt()
    corr=float((xc*yc).sum()/den) if den>1e-12 else float('nan')
    return [float((x-y).square().mean()),float((x.mean(0)-y.mean(0)).square().mean()),
            float((xc-yc).square().mean()),float(xc.square().mean().sqrt()),
            float(yc.square().mean().sqrt()),corr,float((dx-dy).square().mean()),
            float((torch.quantile(x,.9,dim=0)-torch.quantile(x,.1,dim=0)).mean()),
            float((torch.quantile(y,.9,dim=0)-torch.quantile(y,.1,dim=0)).mean())]

def means(arr,cols):
    return {k:float(np.nanmean(arr[...,j])) if np.isfinite(arr[...,j]).any() else None
            for j,k in enumerate(cols)}

def longest_valid_span(valid):
    """Longest native contiguous observed interval; earliest interval breaks ties."""
    v=np.asarray(valid,dtype=bool)
    edges=np.diff(np.r_[False,v,False].astype(np.int8))
    starts,stops=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
    if not len(starts): raise ValueError('No observed render frames')
    j=int(np.argmax(stops-starts))
    return int(starts[j]),int(stops[j])

@torch.no_grad()
def run(a):
    if a.output.exists(): raise FileExistsError('Fresh audit required')
    torch.set_num_threads(2)
    control_name=getattr(a,'control_name','control')
    roots={'control':a.pair_root/'checkpoints'/control_name,
           a.experiment_name:a.pair_root/'checkpoints'/a.experiment_name}
    recipes={k:json.loads((r/'provenance.json').read_text())['recipe'] for k,r in roots.items()}
    ca,ea=[r['args'] for r in recipes.values()]
    diff={k:[ca.get(k),ea.get(k)] for k in ca.keys()|ea.keys() if ca.get(k)!=ea.get(k)}
    expected_diff={
        'class_prototype': {'global_distill_target':['clip','class-prototype']},
        'temporal_adapter': {'renderer_temporal_adapter':[False,True]},
        'source_spread': {'flow_source_noise':['standard','train-residual-std']},
        'source_temporal': {'flow_source_noise':[
            'train-residual-std' if control_name=='source_spread' else 'standard','train-residual-ar']},
        'native_b0': {},
    }[a.experiment_name]
    assert diff==expected_diff,diff
    assert recipes['control']['source_sha256']==recipes[a.experiment_name]['source_sha256']
    assert recipes['control']['data_provenance']==recipes[a.experiment_name]['data_provenance']
    if a.experiment_name in ('source_spread','source_temporal'):
        assert (recipes['control']['flow_source_noise']['statistics'] ==
                recipes[a.experiment_name]['flow_source_noise']['statistics'])
    if a.experiment_name=='source_temporal':
        assert recipes['control']['flow_source_noise']['temporal_statistics']==recipes[a.experiment_name]['flow_source_noise']['temporal_statistics']
    for r in recipes.values():
        assert not r['test_loaded'] and not r['independent_probe']['enabled']
    data=load_packed(a.data,materialize=False,with_refs=False)
    assert set(data['splits'])=={'train','validation'} and not data['provenance']['test_loaded']
    manifest=data['provenance']['manifest_sha256']; q=data['splits']['validation']
    lookup={str(v):i for i,v in enumerate(q['clip_id'])}
    curves={}
    for branch,root in roots.items():
        path=root/'audio/curves.pt'
        complete=json.loads((root/'audio/complete.json').read_text())
        assert sha(path)==complete['curves_sha256']
        assert recipes[branch]['data_provenance']['manifest_sha256']==manifest
        curves[branch]=torch.load(path,map_location='cpu',weights_only=False)
    c,e=curves.values()
    assert c['clip_id']==e['clip_id'] and len(c['clip_id'])==len(lookup)
    shared_keys=('target','valid','channel_mask','times') if a.experiment_name=='native_b0' else ('target','b0','valid','channel_mask','times')
    for key in shared_keys:
        torch.testing.assert_close(c[key],e[key],rtol=0,atol=0,equal_nan=True)
    if a.experiment_name=='native_b0':
        assert not torch.equal(c['b0'],e['b0']), 'Different registered B0s must be consumed'
        assert recipes['control']['stages']==recipes['native_b0']['stages']==['teacher','audio']
        assert recipes['control']['stage_checkpoint_sha256']!=recipes['native_b0']['stage_checkpoint_sha256']
    assert list(c['predictions'])==list(e['predictions'])==['42/full','123/full','2026/full']
    ids=[lookup[str(cid)] for cid in c['clip_id']]
    labels=q['emotion_id'][ids].cpu()
    groups={'emotion':labels.numpy(),'speaker':q['speaker_id'][ids].cpu().numpy(),
            'intensity':q['intensity_id'][ids].cpu().numpy()}
    names=data['config']['data']['emotion_classes']
    models=[]; pm=[]; support=None
    for path in a.probe:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['train_manifest_sha256']==manifest and ck['classes']==names
        assert not ck.get('test_used_for_selection') and not ck.get('generator_outputs_used_for_fitting')
        s=ck['channel_support'].bool()
        if support is not None: assert torch.equal(support,s)
        support=s
        m=MotionEmotionProbe(ck['feature_dim'],ck['hidden'],len(names)).eval()
        m.load_state_dict(ck['model'],strict=True)
        models.append((m,ck.get('feature_mask',torch.ones(6*int(s.sum()),dtype=torch.bool))))
        pm.append({'sha256':sha(path),'kind':ck['kind'],'path':str(path)})
    assert c['channel_mask'][:,support].all()
    a.output.mkdir(parents=True)
    report={'schema':'matched_curve_audit_v1','test_loaded':False,'training_performed':False,
            'clips':len(ids),'data_manifest_sha256':manifest,'probes':pm,'classes':names,
            'arg_difference':diff,'control_name':control_name,'regions':REGIONS,'region_columns':RC,
            'noise_policy':'saved native-padded seeds42/123/2026, distinct from trimmed-batch diagnostic',
            'aggregation':'equal clips; arithmetic mean draws, no best-seed selection',
            'diversity_policy':'pairwise RMS among three draws; descriptive, not monotonically better',
            'results':{}}
    arrays={'clip_id':np.asarray(c['clip_id']),'labels':labels.numpy(),**groups}
    f=torch.stack([motion_features(t[:,support],v) for t,v in zip(c['target'],c['valid'])])
    report['gt_probes']=[classification_metrics(labels,m(f[:,fm]).argmax(-1),names) for m,fm in models]
    for branch,z in curves.items():
      for policy in ('raw','clip_all'):
        ms=[]; rs=[]; fs=[]; probes=[]
        samples=[p if policy=='raw' else p.clamp(0,1) for p in z['predictions'].values()]
        for seed,pred in zip(z['noise_seeds'],samples):
            metric=[]; region=[]; feature=[]
            for i in range(len(ids)):
                p,t,v,ch,tm=pred[i],z['target'][i],z['valid'][i],z['channel_mask'][i],z['times'][i]
                metric.append(clip_metrics(p,t,v,ch,tm))
                region.append([region_values(p,t,v,ch,ix) for ix in REGIONS.values()])
                feature.append(motion_features(p[:,support],v))
            f=torch.stack(feature)
            ms.append(metric); rs.append(region); fs.append(f.numpy())
            probes.append([classification_metrics(labels,m(f[:,fm]).argmax(-1),names) for m,fm in models])
            print(json.dumps({'event':'scored','branch':branch,'policy':policy,'seed':seed}),flush=True)
        ms,rs=np.asarray(ms),np.asarray(rs)
        diversity=[]
        for i in range(len(ids)):
            row=[]
            for ix in REGIONS.values():
                ix=[j for j in ix if z['channel_mask'][i,j]]
                row.append(float(torch.stack([(pa[i]-pb[i])[z['valid'][i]][:,ix].square().mean().sqrt()
                           for pa,pb in itertools.combinations(samples,2)]).mean()))
            diversity.append(row)
        diversity=np.asarray(diversity); prefix=branch+'__'+policy
        arrays.update({prefix+'__metrics':ms,prefix+'__regions':rs,
                       prefix+'__features':np.asarray(fs),prefix+'__diversity':diversity})
        row={'metrics':means(ms,MC),
             'by_seed':{str(s):{'metrics':means(ms[j],MC),'probes':probes[j]}
                        for j,s in enumerate(z['noise_seeds'])},
             'probe_mean_f1':[float(np.mean([p[j]['macro_f1'] for p in probes])) for j in range(len(models))],
             'regions':{k:means(rs[:,:,j],RC) for j,k in enumerate(REGIONS)},
             'diversity':dict(zip(REGIONS,diversity.mean(0).tolist())),'groups':{}}
        for group,values in groups.items():
            row['groups'][group]={str(k):{'n':int((values==k).sum()),'metrics':means(ms[:,values==k],MC),
                  'regions':{name:means(rs[:,values==k,j],RC) for j,name in enumerate(REGIONS)},
                  'diversity':dict(zip(REGIONS,diversity[values==k].mean(0).tolist()))}
                  for k in np.unique(values)}
        report['results'][prefix]=row
    # Persist expensive scores before optional display export.
    np.savez_compressed(a.output/'per_clip_audit.npz',**arrays)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    export=a.output/'render_inputs'; export.mkdir(); selected=[]
    for name in names:
        cid=f'mead_M025_{name}_L{1 if name=="neutral" else 3}_005'
        if cid not in c['clip_id']: raise ValueError('Fixed render clip missing: '+cid)
        i=c['clip_id'].index(cid); start,stop=longest_valid_span(c['valid'][i].numpy())
        sl=slice(start,stop)
        motions=torch.stack([c['target'][i,sl],c['predictions']['42/full'][i,sl],e['predictions']['42/full'][i,sl]])
        np.savez_compressed(export/(name+'.npz'),channels=np.asarray(ARKIT_NAMES),
              times=c['times'][i,sl].numpy(),valid=c['valid'][i,sl].numpy(),
              channel_mask=c['channel_mask'][i].numpy(),motions=motions.numpy(),
              mode_names=np.asarray(['GT','Matched control',a.experiment_name+' candidate']),
              clip_id=np.asarray(cid),noise_seed=np.asarray(42))
        selected.append({'clip_id':cid,'frames':stop-start,'native_frame_start':start,'native_frame_stop':stop,
                         'selection':'fixed M025 utterance005 L3; neutral L1; longest contiguous valid interval, tie earliest; scores use ALL valid frames'})
    report['render_inputs']=selected
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-root',type=Path,required=True)
    p.add_argument('--experiment-name',choices=['class_prototype','temporal_adapter','source_spread','source_temporal','native_b0'],default='class_prototype')
    p.add_argument('--control-name',choices=['control','source_spread'],default='control')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe',type=Path,nargs='+',required=True)
    run(p.parse_args())
