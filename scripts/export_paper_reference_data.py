"""Read-only Phase53 figure data from existing development predictions.

No training, test input, new generation, or fitted emotion classifier.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.packed_trainval_cache import load_packed
from scripts.train_expression_response import sha,configure,write,state_digest
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig
from kinetalk_b0.emotion_probe import MotionEmotionProbe,motion_features
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES


@torch.no_grad()
def run(a):
    configure(47)
    root,out=Path(a.phase53),Path(a.output)
    if out.exists():raise FileExistsError('Fresh figure export required')
    out.mkdir(parents=True)
    binding=json.loads((root/'binding.json').read_text())
    data=load_packed(binding['data'],materialize=False,with_refs=False)
    assert not data['provenance']['test_loaded']
    assert data['provenance']['manifest_sha256']==binding['data_manifest_sha256']
    q=data['splits']['validation'];assert len(q['_lengths'])==1367
    curves={scope:torch.load(root/scope/'evaluation/curves.pt',map_location='cpu',weights_only=False) for scope in ('style_only','joint')}
    for scope,curve in curves.items():
        assert curve['clip_id']==q['clip_id'] and curve['test_loaded'] is False
        manifest=json.loads((root/scope/'evaluation/manifest.json').read_text())
        assert sha(root/scope/'evaluation/curves.pt')==manifest['files']['curves.pt']['sha256']
    checkpoint=root/'joint/seed47/final.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=ResponseConfig(**ck['config'])
    model=ExpressionResponse(config,ck['model']['feature_mean'],ck['model']['feature_std'],ck['model']['scales']).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(ck['model'],strict=True);before=state_digest(model)
    spec=binding['probes'][0];assert sha(spec['path'])==spec['sha256']
    probe_ck=torch.load(spec['path'],map_location='cpu',weights_only=False)
    assert not probe_ck.get('generator_outputs_used_for_fitting') and not probe_ck.get('test_used_for_selection')
    probe=MotionEmotionProbe(probe_ck['feature_dim'],probe_ck['hidden'],len(probe_ck['classes'])).eval().requires_grad_(False)
    probe.load_state_dict(probe_ck['model'],strict=True)
    feature_mask=probe_ck.get('feature_mask',torch.ones(306,dtype=torch.bool))
    arrays={'label':q['emotion_id'].numpy(),'speaker':q['speaker_id'].numpy(),'clip_id':np.array(q['clip_id'])}
    gathered={k:[] for k in ('audio_g','gt_features','gt_embedding','style_only_embedding','joint_embedding')}
    baseline_root=Path(a.baseline_root)
    method_dirs={'VOCA-core (adapted)':'voca_core_arkit_20260923',
        'EmoTalk-core (adapted)':'emotalk_core_arkit_20260923',
        'FaceFormer (adapted)':'faceformer_arkit_memmap_20260923'}
    baselines={name:torch.load(baseline_root/folder/'native_predictions.pt',map_location='cpu',weights_only=False) for name,folder in method_dirs.items()}
    names=data['config']['data']['emotion_classes'];assert names==probe_ck['classes']
    support=torch.tensor([True]*51+[False])
    display={};sources={}
    for name,folder in method_dirs.items():
        value=baselines[name]
        assert value['test_loaded'] is False and value['manifest_sha256']==binding['data_manifest_sha256']
        assert value['scope']=='development validation'
        sources[name]=dict(predictions_sha256=sha(baseline_root/folder/'native_predictions.pt'),checkpoint_sha256=sha(baseline_root/folder/'checkpoint.pt'),adapted=True)
    for sub in torch.arange(1367).split(16):
        batch=q.batch(sub,a.device,keys=('audio_features','motion','valid','times','channel_mask'))
        prior=model.audio_prior(batch['audio_features'],batch['valid'])
        gathered['audio_g'].extend(prior['g_mean'].cpu().numpy())
        for j,i in enumerate(sub.tolist()):
            length=int(q['_lengths'][i]);cid=q['clip_id'][i]
            gt=batch['motion'][j,:length].cpu();valid=batch['valid'][j,:length].cpu()
            feat=motion_features(gt[:,support],valid)
            gathered['gt_features'].append(feat.numpy())
            gathered['gt_embedding'].append(probe.network[:2]((feat[feature_mask]-probe.mean)/probe.scale).numpy())
            for scope in ('style_only','joint'):
                y=curves[scope]['predictions'][i]['prior_mean'].clamp(0,1)
                assert y.shape==gt.shape
                feat_y=motion_features(y[:,support],valid)
                gathered[scope+'_embedding'].append(probe.network[:2]((feat_y[feature_mask]-probe.mean)/probe.scale).numpy())
            emotion=next((e for e,c in binding['display_clips'].items() if c==cid),None)
            if emotion:
                ref=root/'joint/evaluation/render_inputs'/f'{emotion}.npz'
                with np.load(ref,allow_pickle=False) as z: saved={k:z[k].copy() for k in z.files}
                start,stop=int(saved['source_start_frame']),int(saved['source_stop_frame'])
                np.testing.assert_array_equal(saved['times'],batch['times'][j,start:stop].cpu().numpy())
                np.testing.assert_array_equal(saved['motions'][0],gt[start:stop].numpy())
                methods=['GT','Neutral B0',*method_dirs,'KineTalk Phase53 joint']
                values=[gt[start:stop],curves['joint']['predictions'][i]['neutral_b0'][start:stop]]
                for method in method_dirs:
                    item=baselines[method]['clips'][cid]
                    for key,value in [('target',gt),('valid',valid),('times',batch['times'][j,:length].cpu()),('channel_mask',batch['channel_mask'][j].cpu())]:
                        if key=='target' and key not in item:continue  # Historical baseline saves predictions/masks, not query GT.
                        torch.testing.assert_close(item[key],value,rtol=0,atol=0,equal_nan=True)
                    values.append(item['prediction'][start:stop])
                values.append(curves['joint']['predictions'][i]['prior_mean'][start:stop])
                saved.update(mode_names=np.array(methods),motions=torch.stack(values).numpy())
                np.savez_compressed(out/f'comparison_{emotion}.npz',**saved)
                display[emotion]=dict(clip_id=cid,sentence_id=str(q['sentence_id'][i]),speaker=int(q['speaker_id'][i]),
                    start_frame=start,stop_frame=stop,source_sha256=sha(ref),comparison_sha256=sha(out/f'comparison_{emotion}.npz'))
        if int(sub[-1])%256<16:print(json.dumps(dict(clips=int(sub[-1])+1)),flush=True)
    assert state_digest(model)==before
    arrays.update({k:np.asarray(v,dtype=np.float32) for k,v in gathered.items()})
    np.savez_compressed(out/'embeddings.npz',**arrays)
    write(out/'metadata.json',dict(classes=names,display=display,checkpoint_sha256=sha(checkpoint),probe_sha256=spec['sha256'],
        data_manifest_sha256=binding['data_manifest_sha256'],source_sha256=sha(__file__),baseline_sources=sources,
        prediction_sources={scope:sha(root/scope/'evaluation/curves.pt') for scope in curves},
        scope='1367 development clips; fixed already-generated Phase53 candidates, not final/test/SOTA',
        feature_definition='audio prior global mean32; frozen real-TRAIN probe first ReLU128 on full-face native statistics306',
        labels_used_for_prediction=False,neural_training_performed=False,query_gt_used_for_generation=False,
        frozen_model_exact=True,test_loaded=False,baseline_replay='Native GT/mask/clock exact; inherited historical checkpoint spot-check in validation_gap',
        embedding_sha256=sha(out/'embeddings.npz')))
    print(json.dumps(dict(status='complete',clips=1367,output=str(out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('phase53','baseline-root','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda')
    run(p.parse_args())
