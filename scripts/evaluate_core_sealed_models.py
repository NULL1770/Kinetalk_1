"""Frozen inference first, sealed native-target scoring second.

Supports the current four-stage KineTalk model and the three cached-audio
ARKit adapters. No test statistic is fitted or used for checkpoint selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_formal_predictable_projection import save_json, canonical_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.baseline_training import make_model, predict
from scripts.train_full_staged import base_forward, audio_affect, generator_inputs
from scripts.arkit_benchmark_report import score_fullface, build_report, write_report
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect


class SealedInputs:
    def __init__(self, root):
        self.root=Path(root)
        self.complete=json.loads((self.root/'complete.json').read_text())
        if self.complete.get('query_motion_loaded') is not False:
            raise ValueError('Query motion was loaded during input preparation')
        self.rows=json.loads((self.root/'rows.json').read_text())
        audit_record=json.loads((self.root/'manifest_audit.json').read_text())
        manifest_path=Path(audit_record['manifest'])
        if sha(manifest_path)!=self.complete['manifest_file_sha256']:
            raise ValueError('Sealed manifest file changed')
        manifest=json.loads(manifest_path.read_text())
        if (self.rows!=manifest['roles']['test']['query'] or
                canonical_hash(manifest['roles']['test'])!=self.complete['test_role_sha256']):
            raise ValueError('Sealed query membership changed')
        for name,digest in self.complete['files'].items():
            if sha(self.root/name)!=digest:raise ValueError('Sealed audio input hash changed')
        if sha(self.root/'enrollment.pt')!=self.complete['enrollment_sha256']:
            raise ValueError('Sealed enrollment changed')
        self.offsets=np.load(self.root/'offsets.npy',mmap_mode='r')
        self.audio=np.load(self.root/'audio_features.npy',mmap_mode='r')
        self.times=np.load(self.root/'times.npy',mmap_mode='r')
        self.valid=np.load(self.root/'valid.npy',mmap_mode='r')
        self.refs=torch.load(self.root/'enrollment.pt',map_location='cpu',weights_only=False)
        self.anchors={}
        for person,refs in self.refs.items():
            means=[];available=[]
            for ref in refs:
                obs=ref['valid'][:,None]&ref['channel_mask'][None]
                means.append(torch.where(obs,ref['motion'],0.).sum(0)/obs.sum(0).clamp_min(1))
                available.append(obs.sum(0)>0)
            self.anchors[person]=torch.where(torch.stack(available).all(0),torch.stack(means).quantile(.5,dim=0),0.)

    def batch(self, indices, device):
        lengths=[int(self.offsets[i+1]-self.offsets[i]) for i in indices]
        width=max(lengths)
        features=torch.zeros(len(indices),width,1540)
        valid=torch.zeros(len(indices),width,dtype=torch.bool)
        for j,i in enumerate(indices):
            lo,hi=int(self.offsets[i]),int(self.offsets[i+1])
            features[j,:hi-lo]=torch.from_numpy(np.array(self.audio[lo:hi],dtype=np.float32))
            valid[j,:hi-lo]=torch.from_numpy(np.array(self.valid[lo:hi]))
        return {'audio_features':features.to(device),'content':features[...,:768].to(device),
                'valid':valid.to(device),'anchors':torch.stack([self.anchors[self.rows[i]['speaker']] for i in indices]).to(device)},lengths


@torch.no_grad()
def encode_references(system, inputs, device):
    identities={}
    for person,refs in inputs.refs.items():
        width=max(len(ref['valid']) for ref in refs)
        motion=torch.zeros(len(refs),width,52,device=device)
        content=torch.zeros(len(refs),width,768,device=device)
        valid=torch.zeros(len(refs),width,dtype=torch.bool,device=device)
        channel=torch.stack([r['channel_mask'] for r in refs]).to(device)
        for i,ref in enumerate(refs):
            n=len(ref['valid']);motion[i,:n]=ref['motion'].to(device)
            content[i,:n]=ref['content'].to(device);valid[i,:n]=ref['valid'].to(device)
        base=base_forward(system,content,valid)
        residual=torch.where(valid[...,None]&channel[:,None],motion-base['b0'],0.)
        identities[person]=system.encode_identity(residual[None],valid[None],reference_channel_mask=channel[None])
    return identities


def sealed_target(row, native_root):
    path=(native_root/row['artifact']).resolve()
    if not path.is_relative_to(native_root.resolve()) or sha(path)!=row['artifact_sha256']:
        raise ValueError('Scoring target path/hash changed')
    with np.load(path,allow_pickle=False) as z:
        return {key:np.array(z[key]) for key in ('motion','mask','channel_mask','times')}


@torch.no_grad()
def run(args):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    inputs=SealedInputs(args.inputs)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if args.method=='kinetalk':
        run_root=args.checkpoint.parent.parent
        summary=json.loads((run_root/'summary.json').read_text())
        if summary['status']!='complete' or summary['smoke']:raise ValueError('Incomplete/smoke KineTalk checkpoint')
        provenance=json.loads((run_root/'provenance.json').read_text())
        recipe=provenance['recipe'];ablation=recipe['args'].get('ablation','none')
        if checkpoint['stage']!='audio' or checkpoint['recipe_sha256']!=provenance['recipe_sha256']:
            raise ValueError('Fixed audio-stage checkpoint required')
        binding=checkpoint['data_manifest_sha256']
        system=NeutralAffectSystem(checkpoint['config']).to(args.device).eval()
        system.load_state_dict(checkpoint['system'],strict=True)
        stats=checkpoint['feature_stats']
        audio=SlowStateAffect(stats['mean'],stats['std'],stride=recipe['args']['stride']).to(args.device).eval()
        audio.load_state_dict(checkpoint['audio'],strict=True)
        identities=encode_references(system,inputs,args.device)
        seeds=[42,123,2026];decode_steps=recipe['args']['decode_steps']
        model=None
    elif args.method=='facediffuser':
        from kinetalk_b0.models.facediffuser_arkit import FaceDiffuserARKit
        import importlib.util
        import importlib
        if not args.condition_checkpoint or not args.legacy_source:
            raise ValueError('FaceDiffuser requires its original frozen condition checkpoint and source tree')
        protocol=checkpoint['protocol']
        if protocol['smoke'] or sha(args.condition_checkpoint)!=protocol['source_checkpoint_sha256']:
            raise ValueError('FaceDiffuser frozen condition binding differs')
        binding=protocol['data']['manifest_sha256']
        # Load the exact historical conditioning implementation under a private
        # package name: the refactored identity pooling must not alter an
        # already-trained baseline's input distribution.
        legacy=args.legacy_source/'kinetalk_b0'
        spec=importlib.util.spec_from_file_location('kinetalk_legacy',legacy/'__init__.py',submodule_search_locations=[str(legacy)])
        package=importlib.util.module_from_spec(spec);sys.modules['kinetalk_legacy']=package;spec.loader.exec_module(package)
        LegacySystem=importlib.import_module('kinetalk_legacy.models.neutral_affect').NeutralAffectSystem
        LegacyAudio=importlib.import_module('kinetalk_legacy.models.slow_state_affect').SlowStateAffect
        source=torch.load(args.condition_checkpoint,map_location='cpu',weights_only=False)
        if source['data_manifest_sha256']!=binding:raise ValueError('FaceDiffuser source train binding differs')
        system=LegacySystem(source['config']).to(args.device).eval();system.load_state_dict(source['system'],strict=True)
        audio=LegacyAudio(checkpoint['feature_stats']['mean'],checkpoint['feature_stats']['std']).to(args.device).eval()
        audio.load_state_dict(source['audio'],strict=True)
        identities=encode_references(system,inputs,args.device)
        model=FaceDiffuserARKit(**checkpoint['config']).to(args.device).eval();model.load_state_dict(checkpoint['model'])
        stats=checkpoint['feature_stats'];seeds=[42,123,2026];ablation='none'
    else:
        if checkpoint.get('report',{}).get('smoke') is not False or not (args.checkpoint.parent/'training_complete.json').is_file():
            raise ValueError('Fixed completed non-smoke baseline required')
        if checkpoint['method']!=args.method:raise ValueError('Checkpoint method differs')
        binding=checkpoint['protocol']['data']['manifest_sha256']
        model=make_model(args.method,checkpoint['config']).to(args.device).eval()
        model.load_state_dict(checkpoint['model'],strict=True)
        seeds=[42];ablation='none'
    if binding!=inputs.complete['train_manifest_sha256']:raise ValueError('Frozen checkpoint train manifest differs')
    recipe=dict(schema='sealed_test_inference_score_v1',method=args.method,ablation=ablation,
        checkpoint_sha256=sha(args.checkpoint),manifest_sha256=inputs.complete['manifest_sha256'],
        test_role_sha256=inputs.complete['test_role_sha256'],inputs_complete_sha256=sha(args.inputs/'complete.json'),
        seeds=seeds,clips=len(inputs.rows),batch_size=args.batch_size,selection='fixed final epoch',
        evaluator_sha256=sha(__file__),test_statistics_fitted=False,
        rig_sha256=sha(args.rig) if args.rig else None,probe_sha256=sha(args.probe) if args.probe else None,
        reference_clip_ids={p:[r['clip_id'] for r in refs] for p,refs in inputs.refs.items()})
    # This native MEAD conversion observes all channels except TongueOut in
    # every TRAIN clip (audited before test scoring). Apply the same fixed
    # neutral exclusion to every method and the vertex reconstruction.
    coefficient_support=np.array([True]*51+[False])
    recipe['coefficient_support']=coefficient_support.tolist()
    recipe['coefficient_support_source']='complete train channel-mask audit: TongueOut absent in 12536/12536 clips'
    root=Path(__file__).resolve().parents[1]
    recipe['source_sha256']={name:sha(root/name) for name in (
        'scripts/evaluate_sealed_models.py','scripts/faceformer_arkit_model.py',
        'scripts/voca_emotalk_arkit_models.py','kinetalk_b0/models/neutral_affect.py',
        'kinetalk_b0/models/slow_state_affect.py','scripts/arkit_benchmark_report.py',
        'scripts/evaluate_arkit_literature_metrics.py','scripts/evaluate_vertex_lve.py')}
    if checkpoint.get('config',{}).get('adapter_version') == 'core_arkit_v1':
        recipe['baseline_adaptation'] = checkpoint['protocol']['baseline_adaptation']
        for name in ('scripts/baseline_core_arkit.py', 'scripts/baseline_training.py'):
            recipe['source_sha256'][name] = sha(root/name)
    if args.method=='facediffuser':
        recipe.update(condition_checkpoint_sha256=sha(args.condition_checkpoint),
                      legacy_source=str(args.legacy_source),support=model.support.cpu().tolist(),
                      diffusion_steps=model.diffusion_steps,official_beat_reproduction=False)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'recipe.json').is_file() and json.loads((args.output/'recipe.json').read_text())!=recipe:
        raise ValueError('Existing sealed evaluation recipe differs')
    save_json(args.output/'recipe.json',recipe)
    path=args.output/'predictions.npy';seal=args.output/'predictions_sealed.json'
    total=int(inputs.offsets[-1]);shape=(len(seeds),total,52)
    if not seal.exists():
        values=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=shape)
        progress=args.output/'inference_progress.json'
        done=json.loads(progress.read_text())['clips'] if progress.exists() else 0
        for start in range(done,len(inputs.rows),args.batch_size):
            indices=list(range(start,min(start+args.batch_size,len(inputs.rows))))
            b,lengths=inputs.batch(indices,args.device)
            if args.method in ('kinetalk','facediffuser'):
                identity={key:torch.cat([identities[inputs.rows[i]['speaker']][key] for i in indices]) for key in ('code','baseline')}
                if args.method=='kinetalk':
                    base=base_forward(system,b['content'],b['valid']);b.update(base)
                    b,identity=generator_inputs(b,identity,ablation)
                    affect=audio_affect(audio,b['audio_features'],b['valid'])
                else:
                    affect=audio(b['audio_features'],b['valid'])
                    condition=torch.cat((affect['global'],identity['code'],b['anchors'],affect['intensity_value']),-1)
                    mean=stats['mean'].to(args.device);std=stats['std'].to(args.device)
                    clean=torch.where(b['valid'][...,None],b['audio_features'],mean)
                    normalized=(clean-mean)/std
                    fd_audio=torch.cat((torch.where(b['valid'][...,None],b['content'],0.),normalized),-1)
            for s,seed in enumerate(seeds):
                if args.method=='kinetalk':
                    noise=torch.zeros(len(indices),b['valid'].shape[1],52)
                    for j,i in enumerate(indices):
                        stable=int.from_bytes(hashlib.sha256(f"{seed}:{inputs.rows[i]['clip_id']}".encode()).digest()[:8],'big')%(2**63-1)
                        noise[j,:lengths[j]]=torch.randn(lengths[j],52,generator=torch.Generator().manual_seed(stable))
                    pred=system.generate(b['content'],b['valid'],identity,affect,initial_noise=noise.to(args.device),
                        steps=decode_steps,base={'b0':b['b0'],'h0':b['h0']})['motion'].cpu()
                elif args.method=='facediffuser':
                    generator=torch.Generator(device=args.device).manual_seed(seed+1000003+start)
                    initial=torch.randn(len(indices),b['valid'].shape[1],52,generator=torch.Generator().manual_seed(seed+start)).to(args.device)
                    pred=model.sample(fd_audio,b['valid'],initial_noise=initial,generator=generator,clip_condition=condition).cpu()
                else:pred=predict(model,args.method,b)[0].cpu()
                if not torch.isfinite(pred).all():raise ValueError('Nonfinite sealed prediction')
                for j,i in enumerate(indices):values[s,int(inputs.offsets[i]):int(inputs.offsets[i+1])]=pred[j,:lengths[j]].numpy()
            values.flush();save_json(progress,dict(clips=indices[-1]+1,total=len(inputs.rows),query_targets_loaded=False))
            if start%(args.batch_size*25)==0:print(json.dumps(dict(event='sealed_inference',clips=indices[-1]+1)),flush=True)
        del values
        save_json(seal,dict(predictions_sha256=sha(path),recipe_sha256=canonical_hash(recipe),query_targets_loaded=False,clips=len(inputs.rows)))
    sealed=json.loads(seal.read_text())
    if sealed['recipe_sha256']!=canonical_hash(recipe) or sha(path)!=sealed['predictions_sha256']:
        raise ValueError('Sealed predictions changed')
    predictions=np.load(path,mmap_mode='r');rows=[];vertex=[];labels=[];emotion=[]
    rig=None;probe=None
    if args.rig:
        with np.load(args.rig,allow_pickle=False) as z:rig={k:np.array(z[k]) for k in z.files}
        if any(k not in rig for k in ('neutral_vertices','blendshape_deltas','lip_mask','expression_mask','coordinate_scale_to_mm','coordinate_unit')):
            raise ValueError('Rig requires declared geometry, region masks and physical units')
    if args.probe:
        from kinetalk_b0.emotion_probe import MotionEmotionProbe, classification_metrics, motion_features
        ck=torch.load(args.probe,map_location='cpu',weights_only=False)
        if ck['train_manifest_sha256']!=binding or ck['test_used_for_selection']:raise ValueError('Invalid independent emotion probe')
        probe=MotionEmotionProbe(ck['feature_dim'],ck['hidden'],len(ck['classes'])).eval();probe.load_state_dict(ck['model'])
    # Only after the complete prediction file has been hashed and sealed are
    # query motion arrays opened, exclusively by the scoring section below.
    for i,row in enumerate(inputs.rows):
        truth=sealed_target(row,args.native_root);lo,hi=int(inputs.offsets[i]),int(inputs.offsets[i+1])
        np.testing.assert_array_equal(truth['mask'].astype(bool),inputs.valid[lo:hi])
        np.testing.assert_allclose(truth['times'],inputs.times[lo:hi],rtol=0,atol=0)
        sample=np.asarray(predictions[:,lo:hi]);valid=truth['mask'].astype(bool);channel=truth['channel_mask'].astype(bool)&coefficient_support
        rows.append(score_fullface(sample,dict(clip_id=row['clip_id'],speaker=row['speaker'],sentence=row['sentence'],
            emotion=row['emotion'],target52=truth['motion'],valid=valid,times=truth['times'],
            channel_mask=np.broadcast_to(channel,truth['motion'].shape))))
        if rig:
            from scripts.evaluate_vertex_lve import evaluate
            result=evaluate(sample,truth['motion'],valid,channel,rig['neutral_vertices'],rig['blendshape_deltas'],
                lip_mask=rig['lip_mask'],expression_mask=rig['expression_mask'],fdd_mask=rig.get('fdd_mask'),
                coordinate_scale_to_mm=float(rig['coordinate_scale_to_mm']),coordinate_unit=str(rig['coordinate_unit'].item()),
                coefficient_support=coefficient_support)
            vertex.append({'clip_id':row['clip_id'],**{k:result[k] for k in ('vertex_lve_sqrt_mean','eye_forehead_eve_sqrt_mean','vertex_fdd_absolute_mm2_mean')}})
        if probe:
            support=ck['channel_support']
            if not torch.from_numpy(channel)[support].all():raise ValueError('Probe channels absent from test')
            for sample_i in sample:
                features=motion_features(torch.from_numpy(sample_i.copy())[:,support],torch.from_numpy(valid))
                emotion.append(int(probe(features[None]).argmax(-1)));labels.append(row['emotion'])
        if i%100==0:
            print(json.dumps(dict(event='sealed_scoring',clips=i+1,total=len(inputs.rows))),flush=True)
    report=build_report(rows,scope='sealed MEAD eight-emotion test; frozen checkpoints',sources=recipe)
    report.update(test_loaded=True,method=args.method,ablation=ablation,clips=len(rows),predictions_sha256=sealed['predictions_sha256'],
                  pending_metrics=['vertex LVE/EVE/FDD: fixed rig required'] if not rig else [])
    if args.method in ('voca', 'emotalk'):
        report['baseline_adaptation'] = checkpoint['protocol'].get('baseline_adaptation',
            dict(version='legacy_simplified', official_end_to_end_reproduction=False,
                 label=args.method+'-style internal simplified model'))
    elif args.method == 'faceformer':
        report['baseline_adaptation'] = dict(version='faceformer_arkit_v1',
            label='FaceFormer (ARKit, shared audio)', official_end_to_end_reproduction=False,
            retained=['autoregressive Transformer decoder','periodic positional encoding',
                      'periodic causal attention bias','aligned audio cross attention'],
            deviations=['frozen shared1540 audio replaces wav2vec2 encoder',
                        'neutral52 identity replaces categorical speaker code','ARKit52 output'])
    if vertex:
        report['vertex']={k:float(np.mean([v[k] for v in vertex])) if all(v[k] is not None for v in vertex) else None for k in vertex[0] if k!='clip_id'}
        report['vertex_per_clip']=vertex
        report['vertex_protocol']={k:result[k] for k in ('rig_identity','region_masks','coordinate_units','coordinate_scale_to_mm','definitions')}
    if probe:
        report['emotion']=classification_metrics(torch.tensor(labels),torch.tensor(emotion),ck['classes'])
        report['emotion_macro_f1']=report['emotion']['macro_f1']
        report['emotion_scope']='independent real-motion statistics probe; equally weighted clips and stochastic draws'
    write_report(args.output/'report.json',report)
    save_json(args.output/'complete.json',dict(status='complete',test_loaded=True,report_sha256=sha(args.output/'report.json'),clips=len(rows)))
    print(json.dumps(dict(event='sealed_evaluation_complete',clips=len(rows))),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ('inputs','checkpoint','native-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--method',choices=('voca','emotalk','faceformer','kinetalk','facediffuser'),required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--rig',type=Path);p.add_argument('--probe',type=Path)
    p.add_argument('--condition-checkpoint',type=Path);p.add_argument('--legacy-source',type=Path)
    run(p.parse_args())


if __name__=='__main__':main()
