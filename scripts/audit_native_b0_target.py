"""Full native validation audit of matched B0 supervision; no residual rollout."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.emotion_probe import MotionEmotionProbe,classification_metrics,motion_features
from scripts.audit_matched_motion_curves import RC,MC,region_values,means
from scripts.audit_stage1_dropout import paired_ci
from scripts.diagnose_flow_sampling import clip_metrics,sha
from scripts.packed_trainval_cache import load_packed
from scripts.train_full_staged import base_forward,subset

WARM='/root/autodl-tmp/kinetalk_final_20260922/checkpoints/phase2_fullmouth_timing000_20261004/audio/final.pt'
WARM_SHA='e17659536a6fcaaaec2d9c22f99403fe4d9f1c8690c3cd182583894545f5ab45'


@torch.no_grad()
def run(a):
    if a.output.exists():raise FileExistsError('Fresh audit required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    data=load_packed(a.data,materialize=False,with_refs=False)
    assert set(data['splits'])=={'train','validation'} and not data['provenance']['test_loaded']
    manifest=data['provenance']['manifest_sha256'];q=data['splits']['validation'];assert len(q['_lengths'])==1367
    names=data['config']['data']['emotion_classes'];labels=q['emotion_id'];speakers=q['speaker_id'].numpy()
    assert sha(WARM)==WARM_SHA
    warm=torch.load(WARM,map_location='cpu',weights_only=False)
    values_only = getattr(a, 'study', 'membership') == 'target-values'
    report=dict(schema='b0_target_values_audit_v1' if values_only else 'native_b0_target_audit_v1',test_loaded=False,residual_generation_performed=False,
        base_prediction_performed=True,training_performed=False,downstream_adaptation_required=True,
        interpretation='B0 only; expressive native-target B0 is an audio motion base, not a neutral base or full generator',
        warm_sha256=WARM_SHA,data_manifest_sha256=manifest,script_sha256=sha(Path(__file__)),results={},probes=[])
    models=[];support=None
    for path in sorted(a.probes.glob('*.pt')):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['train_manifest_sha256']==manifest and ck['classes']==names
        assert not ck.get('test_used_for_selection') and not ck.get('generator_outputs_used_for_fitting')
        s=ck['channel_support'].bool()
        if support is not None:assert torch.equal(support,s)
        support=s
        model=MotionEmotionProbe(ck['feature_dim'],ck['hidden'],len(names)).eval()
        model.load_state_dict(ck['model'],strict=True);model.requires_grad_(False)
        models.append((model,ck.get('feature_mask',torch.ones(6*int(s.sum()),dtype=torch.bool))))
        report['probes'].append(dict(path=str(path),sha256=sha(path),kind=ck['kind']))
    assert len(models)==4 and int(support.sum())==51
    arrays={'clip_id':np.asarray(q['clip_id']),'emotion':labels.numpy(),'speaker':speakers,
            'intensity':q['intensity_id'].numpy()}
    recipes={};native_truth=None;contracts={}
    for mode in ('control','native'):
        root=a.pair_root/'checkpoints'/mode
        recipe=json.loads((root/'provenance.json').read_text())['recipe'];recipes[mode]=recipe
        summary=json.loads((root/'summary.json').read_text())
        assert summary['total_steps']==1568 and not summary['test_loaded']
        assert recipe['stages']==['articulation'] and recipe['stage_checkpoint_sha256']==WARM_SHA
        assert recipe['args']['articulation_updates']==1568 and recipe['args']['batch_size']==16
        path=root/'articulation/final.pt';complete=json.loads((root/'articulation/complete.json').read_text())
        assert sha(path)==complete['final_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=False)
        changed=[]
        for key,v in ck['system'].items():
            if not torch.equal(v,warm['system'][key]):assert key.startswith('stage1.'),key;changed.append(key)
        assert changed and all(torch.equal(v,warm['audio'][k]) for k,v in ck['audio'].items())
        curve_path=Path(complete['curves']);assert sha(curve_path)==complete['curves_sha256']
        z=torch.load(curve_path,map_location='cpu',weights_only=False)
        truth={k:z[k] for k in ('target','valid','channel_mask','times')}
        if native_truth is None:native_truth=truth
        else:assert all(torch.equal(truth[k],native_truth[k]) for k in truth),'Native evaluation truth/masks/times differ'
        if values_only:
            contract=json.loads((a.pair_root/(mode+'_training_contract.json')).read_text())
            assert contract['updates']==1568 and contract['clip_inputs']==24990
            assert contract['target_values_mode']==('safe-teacher' if mode=='control' else 'native-query')
            assert contract['all_target_value_assertions_passed'] and contract['all_observation_assertions_passed']
            assert not contract['test_loaded'];contracts[mode]=contract
        assert list(z['clip_id'])==list(q['clip_id']) and len(z['target'])==1367
        preds=list(z['predictions'].values());assert len(preds)==3 and all(torch.equal(preds[0],x) for x in preds[1:])
        assert z['channel_mask'][:,support].all()
        report['results'][mode]=dict(changed_b0_tensors=len(changed),frozen_other_state_exact=True,
            final_checkpoint_sha256=sha(path),actual_epochs=summary['epochs_per_stage'],updates=1568,
            clip_inputs=sum(json.loads(p.read_text())['samples'] for p in (root/'articulation').glob('epoch*.json')))
        for policy in ('raw','clip_all'):
            pred=preds[0] if policy=='raw' else preds[0].clamp(0,1)
            rows=[];mouth=[];jaw=[];full=[];features=[]
            for i in range(1367):
                p,t,v,ch,tm=pred[i],z['target'][i],z['valid'][i],z['channel_mask'][i],z['times'][i]
                rows.append(clip_metrics(p,t,v,ch,tm));mouth.append(region_values(p,t,v,ch,list(range(14,41))))
                jaw.append(region_values(p,t,v,ch,[17]));full.append(float((p[v][:,ch]-t[v][:,ch]).square().mean()))
                features.append(motion_features(p[:,support],v))
            f=torch.stack(features);mouth=np.asarray(mouth);jaw=np.asarray(jaw);rows=np.asarray(rows);full=np.asarray(full)
            probes=[classification_metrics(labels,m(f[:,fm]).argmax(-1),names) for m,fm in models]
            result=dict(metrics=means(rows,MC),full_coefficient_mse=float(full.mean()),mouth=means(mouth,RC),jaw=means(jaw,RC),
                neutral_mouth=means(mouth[labels.numpy()==0],RC),neutral_jaw=means(jaw[labels.numpy()==0],RC),
                base_motion_probes=probes,groups={})
            for key,g in (('emotion',labels.numpy()),('speaker',speakers),('intensity',q['intensity_id'].numpy())):
                result['groups'][key]={str(k):dict(clips=int((g==k).sum()),metrics=means(rows[g==k],MC),mouth=means(mouth[g==k],RC),
                    jaw=means(jaw[g==k],RC),full_coefficient_mse=float(full[g==k].mean())) for k in np.unique(g)}
            report['results'][mode][policy]=result
            prefix=mode+'__'+policy
            arrays.update({prefix+'__metrics':rows,prefix+'__mouth':mouth,prefix+'__jaw':jaw,prefix+'__full_mse':full,prefix+'__features':f.numpy()})
        # Neutral TRAIN query timing is evaluated separately, never fit.
        model=NeutralAffectSystem(ck['config']).to(a.device).eval();model.load_state_dict(ck['system'],strict=True)
        tq=data['splits']['train'];ids=(tq['emotion_id']==0).nonzero(as_tuple=True)[0];assert len(ids)==715
        tr=[];tj=[]
        for ix in ids.split(16):
            b=subset(tq,ix,a.device,keys=('content','motion','valid','channel_mask'))
            p=base_forward(model,b['content'],b['valid'])['b0'].cpu().clamp(0,1)
            for j in range(len(ix)):
                t,v,ch=b['motion'][j].cpu(),b['valid'][j].cpu(),b['channel_mask'][j].cpu()
                tr.append(region_values(p[j],t,v,ch,list(range(14,41))));tj.append(region_values(p[j],t,v,ch,[17]))
        report['results'][mode]['train_neutral']=dict(mouth=means(np.asarray(tr),RC),jaw=means(np.asarray(tj),RC))
        del model;torch.cuda.empty_cache()
    c,e=recipes['control'],recipes['native']
    diff={k:[c['args'].get(k),e['args'].get(k)] for k in c['args'].keys()|e['args'].keys() if c['args'].get(k)!=e['args'].get(k)}
    expected_diff=({'articulation_target_values':['safe-teacher','native-query']} if values_only
                   else {'articulation_scope':['neutral','all-emotions']})
    assert diff==expected_diff,diff
    for key in ('source_sha256','data_provenance','stage_checkpoint_sha256','motion_support','residual_support'):
        assert c[key]==e[key],key
    assert c['articulation_scope']['effective_articulation_clip_count']==3298
    if values_only:
        assert c['articulation_scope']==e['articulation_scope'] and c['paths']==e['paths']
        for key in ('safe_dtw_manifest_sha256','safe_dtw_artifacts_sha256','articulation_update_budget'):
            assert c[key]==e[key],key
        for key in ('sample_stream_sha256','observation_stream_sha256','noise_stream_sha256'):
            assert contracts['control'][key]==contracts['native'][key],key
        assert contracts['control']['target_value_stream_sha256']!=contracts['native']['target_value_stream_sha256']
        report['training_contracts']=contracts
        report['conceptual_difference']='Same 3298 approved clips, batches, observations and RNG; only neutral-teacher target values replaced by own native GT'
    else:
        assert e['articulation_scope']['clip_count']==12536
        assert c['paths']['safe_dtw_root']=='/root/autodl-tmp/kinetalk_data/processed/safe_supervision_v1' and 'safe_dtw_root' not in e['paths']
        report['conceptual_difference']='Supervision targets/membership only: approved neutral teachers versus own synchronous native GT'
    report['native_evaluation_truth_masks_times_exact']=True
    report['argument_difference']=diff;report['comparisons']={}
    for policy in ('raw','clip_all'):
        prefix='__'+policy
        report['comparisons'][policy]={region:paired_ci(arrays['native'+prefix+'__'+region]-arrays['control'+prefix+'__'+region],speakers)
            for region in ('mouth','jaw')}
    c,e=(report['results'][k]['clip_all'] for k in ('control','native'))
    report['passes_geometry_timing_gate']=(e['full_coefficient_mse']<c['full_coefficient_mse'] and e['mouth']['mse']<c['mouth']['mse']
        and e['neutral_mouth']['displacement_mse']<=c['neutral_mouth']['displacement_mse']
        and e['neutral_jaw']['centered_correlation']>=c['neutral_jaw']['centered_correlation'])
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'per_clip_metrics.npz',**arrays)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(event='complete',passes_gate=report['passes_geometry_timing_gate'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pair-root','data','probes','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--study',choices=('membership','target-values'),default='membership')
    p.add_argument('--device',default='cuda');run(p.parse_args())
