"""Independent fixed-epoch audit of matched audio/text and no-text pilots.

Authenticates sources, caches, all checkpoints and RNG draws; regenerates the
first 32 locked validation examples at seed42 in every saved mode. All saved
validation curves are scored, with eight-seed distributions only for full/zero.
Interventions with a single stored seed remain single-seed diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.label_guided_intensity import fit_intensity_scales, regional_intensity, regional_window_activity
from scripts import audit_direct_audio_dynamics as common
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.audit_projection_readiness import relative_error_audit
from scripts.audit_teacher_schedule_probe import GROUPS, audit_populations
from scripts.named_motion_distribution_metrics import audit_named_motion_samples
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.train_predictable_renderer import batch_to_device, observed, sha, state_hash
from scripts import train_audio_text_dynamics as runner

SCHEMA = 'audio_text_dynamics_v1'
AUDIT_SCHEMA = 'audio_text_dynamics_audit_v1'
ARMS = ('text', 'no_text')
SEEDS = list(common.NOISE_SEEDS)
HASH_KEYS = ('minibatch_sha256', 'noise_time_sha256')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def validate_curve_binding(run, stem, checkpoint, recipe):
    sidecar = run/(stem+'.provenance.json')
    if read(sidecar) != {'schema': 'projection_schedule_curves_provenance_v1',
            'curve_sha256': sha(run/(stem+'.pt')), 'checkpoint_sha256': sha(run/checkpoint),
            'recipe_sha256': canonical_hash(recipe), 'cache_sha256': recipe['input_sha256']['cache']}:
        raise ValueError('Curve/checkpoint/cache sidecar binding differs: '+stem)
    return sidecar


def validate_checkpoint(value, recipe, epoch):
    if (value.get('schema') != SCHEMA or value.get('recipe') != recipe
            or value.get('recipe_sha256') != canonical_hash(recipe)
            or value.get('completed_epochs') != epoch
            or value.get('protected_sha256') != recipe.get('protected_sha256')):
        raise ValueError('Checkpoint schema/recipe/epoch/protected binding differs')
    for key in ('renderer', 'encoder'):
        common._verify_state(value.get(key), value.get(key+'_sha256'), key)
    if epoch == 0:
        if value.get('step') != 0 or value['encoder_sha256'] != recipe['initial_encoder_sha256']:
            raise ValueError('Initial checkpoint is not initial')
        if any(value.get(k) != hashlib.sha256().hexdigest() for k in HASH_KEYS):
            raise ValueError('Initial draw hashes are not empty')


def load_arm(path, arm):
    run = Path(path).resolve(); hashes = common._verify_inventory(run)
    required = {'provenance.json', 'summary.json', 'epoch000_curves.pt', 'final_curves.pt',
        'epoch000_curves.provenance.json', 'final_curves.provenance.json'}
    required.update(f'epoch{epoch:03d}.pt' for epoch in range(9))
    required.update(f'epoch{epoch:03d}.json' for epoch in range(1,9))
    if not required.issubset(hashes):
        raise ValueError('Output inventory omits required audit evidence')
    provenance, summary = read(run/'provenance.json'), read(run/'summary.json')
    recipe = provenance['recipe']; digest = canonical_hash(recipe)
    expected = {'schema': SCHEMA, 'arm': arm, 'seed': 46, 'epochs': 8, 'batch_size': 16,
        'optimizer': 'Adam', 'renderer_lr': 1e-5, 'encoder_lr': 1e-4, 'decode_steps': 12,
        'noise_seeds': SEEDS, 'smoke_steps': 0, 'teacher_intensity_probability': 0.,
        'online_global_distillation': False, 'test_loaded': False, 'default_replaced': False,
        'checkpoint_selection_performed': False, 'loss_weights': runner.LOSS_WEIGHTS}
    if (any(recipe.get(k) != v for k,v in expected.items())
            or provenance.get('recipe_sha256') != digest or summary.get('recipe_sha256') != digest
            or summary.get('schema') != SCHEMA or summary.get('arm') != arm
            or summary.get('completed_epochs') != 8 or summary.get('protected_unchanged') is not True
            or summary.get('test_loaded') is not False or summary.get('default_replaced') is not False):
        raise ValueError('Fixed audio/text protocol differs')
    checkpoints, records = {}, {}
    for epoch in range(9):
        cp = torch.load(run/f'epoch{epoch:03d}.pt', map_location='cpu', weights_only=False)
        validate_checkpoint(cp, recipe, epoch)
        if epoch:
            row = read(run/f'epoch{epoch:03d}.json'); records[epoch] = row
            if row.get('step') != cp.get('step') or any(row.get(k) != cp.get(k) for k in HASH_KEYS):
                raise ValueError('Epoch record/checkpoint draw binding differs')
        if epoch in (0, 8): checkpoints[epoch] = cp
    if summary.get('optimizer_steps') != checkpoints[8]['step']:
        raise ValueError('Summary optimizer budget differs')
    validate_curve_binding(run, 'epoch000_curves', 'epoch000.pt', recipe)
    sidecar = validate_curve_binding(run, 'final_curves', 'epoch008.pt', recipe)
    if (summary['curve_provenance']['sha256'] != sha(sidecar)
            or Path(summary['curve_provenance']['path']).resolve() != sidecar):
        raise ValueError('Final curve sidecar binding differs')
    root = Path(runner.__file__).resolve().parents[1]
    for filename, digest in recipe['source_sha256'].items():
        source = Path(filename).resolve()
        snapshot = Path('source')/source.relative_to(root) if source.is_relative_to(root) else None
        if (snapshot is None or (str(snapshot) not in hashes and snapshot.as_posix() not in hashes) or sha(source) != digest
                or sha(run/snapshot) != digest):
            raise ValueError('Executed/saved training source differs: ' + filename)
    for key, filename in recipe['derived_paths'].items():
        if sha(filename) != recipe['derived_sha256'][key]:
            raise ValueError('Derived training cache differs: ' + key)
    return {'run': run, 'recipe': recipe, 'summary': summary, 'hashes': hashes,
        'records': records, 'checkpoints': checkpoints,
        'epoch0': torch.load(run/'epoch000_curves.pt', map_location='cpu', weights_only=False, mmap=True),
        'final': torch.load(run/'final_curves.pt', map_location='cpu', weights_only=False, mmap=True)}


def validate_saved_curves(curves, reference, arm, *, initial=False):
    seeds = SEEDS; q = reference['q']; mask = observed(q)
    if (curves.get('schema') != SCHEMA or curves.get('arm') != arm
            or curves.get('noise_seeds') != seeds or curves.get('decode_steps') != 12
            or curves.get('clip_id') != list(q['clip_id'])
            or set(curves.get('motion', {})) != set(map(str,seeds))
            or curves.get('oracle_is_target_conditioned') is not True):
        raise ValueError('Saved curve metadata differs')
    for seed in seeds:
        modes = ('full', 'zero') if initial or seed != 42 else runner.MODES
        if set(curves['motion'][str(seed)]) != set(modes):
            raise ValueError('Wrong saved mode/seed layout')
        for value in curves['motion'][str(seed)].values():
            if value.shape != q['motion'].shape or not torch.isfinite(value[mask]).all():
                raise ValueError('Invalid observed saved motion')
    modes = ('full','zero') if initial else runner.MODES
    if set(curves.get('intensity', {})) != set(modes):
        raise ValueError('Intensity intervention layout differs')
    for values in curves['intensity'].values():
        if set(values) != {'predicted','driving'}:
            raise ValueError('Missing intensity field')
        for value in values.values():
            if (value.shape != (*q['valid'].shape,1) or not torch.isfinite(value[q['valid']]).all()
                    or (value[q['valid']] < 0).any()):
                raise ValueError('Invalid predicted/driving intensity')
    if not torch.equal(curves['mismatched_text_indices'], runner.mismatch_indices(q)):
        raise ValueError('Text mismatch schedule differs')


def audit_target_values(targets, cache):
    """Recompute scales/readouts and independent neutral anchors from bound refs."""
    from scripts.prepare_expression_intensity_targets import _query_metadata, _reference_mean
    prov = targets['provenance']; sources = prov['source_sha256']
    for key in ('cache','enrollment'):
        if sha(prov['sources'][key]) != sources[key]: raise ValueError('Intensity source differs')
    rows = [json.loads(line) for line in Path(prov['sources']['enrollment']).read_text(encoding='utf8').splitlines() if line.strip()]
    mapping,query_ids,query_sentences=_query_metadata(cache)
    means={}; bindings={};sentences={speaker:set() for speaker in mapping}
    for row in rows:
        if (row.get('split') != 'train' or row.get('emotion_id',row.get('emotion')) != 0
                or ('emotion' in row and row['emotion'] not in (0,'neutral'))
                or row['clip_id'] in query_ids or row['sentence'] in query_sentences):
            raise ValueError('Intensity references leak query or nontraining role')
        speaker=row['speaker']
        if speaker not in mapping or int(row['speaker_id']) != mapping[speaker]:
            raise ValueError('Intensity reference speaker ID mapping differs')
        if row['clip_id'] in bindings or row['sentence'] in sentences[speaker]:
            raise ValueError('Repeated intensity reference clip/per-person sentence')
        mean,binding=_reference_mean(row,prov['sources']['native_root'])
        means.setdefault(speaker,[]).append(mean);bindings[row['clip_id']]=binding;sentences[speaker].add(row['sentence'])
    saved_bindings={row['clip_id']:row for row in prov['references']}
    if len(saved_bindings)!=len(prov['references']) or bindings != saved_bindings:
        raise ValueError('Saved reference binding differs')
    if set(means)!=set(mapping) or set(targets['by_speaker'])!=set(mapping):
        raise ValueError('Intensity reference/query identity coverage differs')
    for speaker, rows in means.items():
        values=np.stack(rows);expected=np.full(52,np.nan)
        for c in range(52):
            valid=np.isfinite(values[:,c])
            if valid.sum()>=prov['minimum_references_per_channel']:expected[c]=np.median(values[valid,c])
        torch.testing.assert_close(targets['by_speaker'][speaker]['anchor'],torch.from_numpy(expected).float(),rtol=0,atol=0,equal_nan=True)
        if targets['by_speaker'][speaker]['speaker_id']!=mapping[speaker]:raise ValueError('Saved anchor identity differs')
        if not torch.equal(targets['by_speaker'][speaker]['anchor_valid'],torch.from_numpy(np.isfinite(expected))):
            raise ValueError('Saved anchor mask differs')
    for name,split in targets['splits'].items():
        q=cache['splits'][name]['q'];mask=observed(q)&split['anchor_valid'][:,None]
        if (split['clip_id']!=list(q['clip_id']) or split['speaker']!=list(q['speaker'])
                or not torch.equal(split['speaker_id'],q['speaker_id'])):
            raise ValueError('Intensity target identity/order differs')
        anchors=torch.stack([targets['by_speaker'][s]['anchor'] for s in q['speaker']])
        torch.testing.assert_close(split['anchors'],anchors,rtol=0,atol=0,equal_nan=True)
        if not torch.equal(split['anchor_valid'],torch.isfinite(anchors)):raise ValueError('Query anchor mask differs')
        values,valid=regional_intensity(q['motion'],mask,anchors,targets['scales'])
        torch.testing.assert_close(values,split['intensity'],rtol=0,atol=0)
        if not torch.equal(valid,split['valid']):raise ValueError('Saved intensity mask differs')
        activity,activity_valid=regional_window_activity(q['motion'],mask,anchors,targets['scales'])
        torch.testing.assert_close(activity,split['activity'],rtol=0,atol=0)
        if not torch.equal(activity_valid,split['activity_valid']):raise ValueError('Saved activity mask differs')
        if name=='train':
            scale=fit_intensity_scales(q['motion'],mask,anchors,floor=prov['scale_floor'])
            torch.testing.assert_close(scale,targets['scales'],rtol=0,atol=0)
    return {'independent_reference_anchors_rebuilt':True,'training_only_scales_rebuilt':True,'all_target_values_rebuilt':True}


@torch.no_grad()
def regenerate_prefix(data, loaded, audio, text, targets, device):
    system=loaded['system'];encoder=runner.make_encoder(audio,text,targets,device)
    # Initial model was constructed with seed46 immediately after load_source.
    runner.configure(system)
    if (state_hash(system.state_dict())!=data['recipe']['initial_system_sha256']
            or state_hash(encoder.state_dict())!=data['recipe']['initial_encoder_sha256']
            or state_hash(system.renderer.state_dict())!=data['checkpoints'][0]['renderer_sha256']):
        raise ValueError('Fit-only initialization reconstruction differs')
    reference=loaded['cache']['splits']['validation'];n=min(32,len(reference['q']['motion']));ids=torch.arange(n)
    batch=batch_to_device(reference,ids,device)
    noise=torch.randn(reference['q']['motion'].shape,generator=torch.Generator().manual_seed(42))[:n].to(device)
    result={}
    for epoch,key in ((0,'epoch0'),(8,'final')):
        runner.restore(system,encoder,data['checkpoints'][epoch]);curves=data[key]
        result[str(epoch)]={}
        for mode,saved in curves['motion']['42'].items():
            inputs=runner.slice_inputs(audio['splits']['validation'],text['splits']['validation'],targets['splits']['validation'],
                ids,device,text_ids=curves['mismatched_text_indices'][ids] if mode=='shuffled_text' else None)
            affect,output=runner.build_affect(encoder,batch,inputs,data['recipe']['arm'],mode)
            q=batch['q'];pred=system.generate(q['content'],q['valid'],batch['identity'],affect,initial_noise=noise,steps=12,base=batch['base'])['motion'].cpu()
            torch.testing.assert_close(pred,saved[:n],rtol=0,atol=0)
            for field,returned in (('predicted','predicted_intensity'),('driving','driving_intensity')):
                torch.testing.assert_close(output[returned].cpu(),curves['intensity'][mode][field][:n],rtol=0,atol=0)
            result[str(epoch)][mode]={'clips':n,'motion_max_abs':float((pred-saved[:n]).abs().max()),'intensity_exact':True}
    return result


def _condition_statistics(motion,reference,seeds):
    # Existing scorer averages sufficient statistics, not generated trajectories.
    if (len(set(seeds))!=len(seeds) or set(motion)!=set(map(str,seeds)) or not set(seeds).issubset(SEEDS)
            or any(not rows for rows in motion.values())):
        raise ValueError('Statistics seed evidence differs')
    curves={'motion':{str(seed):motion.get(str(seed),{}) for seed in SEEDS}}
    return common.statistics_for_conditions(curves,reference)


def _score_table(stats,populations):
    names=sorted({k[0] for k in stats})
    return {name:{kind:{group:{pop:scalar_summary(stats[name,kind,group][ids]) for pop,ids in populations.items()}
        for group in GROUPS} for kind in common.KINDS} for name in names}


def _paired(stats,velocity,components,candidate,baseline,query,populations,samples):
    return {'candidate':candidate,'baseline':baseline,'centered':{pop:{g:paired_summary(stats[candidate,'centered_residual',g],
        stats[baseline,'centered_residual',g],query['sentence_id'],ids,samples=samples) for g in GROUPS} for pop,ids in populations.items()},
        'raw_relative':{pop:{g:relative_error_audit(stats[candidate,'raw_motion',g][:,[0,3]],stats[baseline,'raw_motion',g][:,[0,3]],
            query['sentence_id'],ids,samples=samples) for g in GROUPS} for pop,ids in populations.items()},
        'mean_relative':{pop:{g:relative_error_audit(components[candidate,g][:,[2,3]],components[baseline,g][:,[2,3]],
            query['sentence_id'],ids,samples=samples) for g in GROUPS} for pop,ids in populations.items()},
        'velocity_relative':{pop:{g:relative_error_audit(velocity[candidate,g],velocity[baseline,g],query['sentence_id'],ids,samples=samples)
            for g in GROUPS} for pop,ids in populations.items()}}


def domain_report(motion,query,populations):
    mask=observed(query);result={}
    for condition in sorted({name for rows in motion.values() for name in rows}):
        predictions=[rows[condition] for rows in motion.values() if condition in rows]
        if any(p.shape!=mask.shape or not torch.isfinite(p[mask]).all() for p in predictions):
            raise ValueError('Invalid observed domain prediction')
        result[condition]={}
        for pop,ids in populations.items():
            result[condition][pop]={}
            for group,channels in GROUPS.items():
                obs=mask[ids][:,:,channels];rows=[];count=int(obs.sum())
                for p in predictions:
                    value=p[ids][:,:,channels];outside=(value<0)|(value>1)
                    if count:
                        rows.append([float(outside[obs].double().mean()),float((torch.relu(-value)+torch.relu(value-1))[obs].double().mean())])
                result[condition][pop][group]={'clamp_fraction':float(np.mean(rows,0)[0]) if count else None,
                    'mean_outside_magnitude':float(np.mean(rows,0)[1]) if count else None,
                    'observed_values_per_seed':count,'noise_seeds':len(predictions)}
    return result


def intensity_statistics(prediction,truth,valid):
    if (prediction.shape!=truth.shape or valid.shape!=truth.shape or valid.dtype!=torch.bool
            or truth.ndim!=3 or truth.shape[-1]!=1):
        raise ValueError('Intensity prediction/target/Boolean mask must match [B,T,1]')
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(truth[valid]).all():
        raise ValueError('Nonfinite observed intensity')
    weight=valid[...,0].float()
    count=valid.sum(1,keepdim=True).clamp_min(1)
    def center(value):
        clean=torch.where(valid,value,0.)
        return torch.where(valid,clean-clean.sum(1,keepdim=True)/count,0.)
    return {'raw_level':clip_statistics(prediction,truth,weight,[0]),
        'centered_dynamics':clip_statistics(center(prediction),center(truth),weight,[0])}


def masked_intensity_mean(value,valid):
    if value.shape!=valid.shape or valid.dtype!=torch.bool:
        raise ValueError('Intensity mean mask must match')
    return (torch.where(valid,value,0.).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)).expand_as(value)


def intensity_report(arms,targets,reference,samples):
    tg=targets['splits']['validation'];valid=tg['valid'];truth=tg['intensity']
    training=targets['splits']['train'];train_mean=float(training['intensity'][training['valid']].mean())
    population=audit_populations(reference['q']);values={'fit_constant':torch.full_like(truth,train_mean),
        'target_clip_mean_oracle':masked_intensity_mean(truth,valid)}
    driving={}
    for arm,data in arms.items():
        for mode,fields in data['final']['intensity'].items():
            values[arm+'/'+mode]=fields['predicted']
            driving[arm+'/'+mode]=fields['driving']
        values[arm+'/static_predicted']=masked_intensity_mean(data['final']['intensity']['full']['predicted'],valid)
    stats={name:intensity_statistics(value,truth,valid) for name,value in values.items()}
    drive_stats={name:intensity_statistics(value,truth,valid) for name,value in driving.items()}
    pairs=[('text/full','no_text/full')]
    for arm in ARMS:
        pairs.extend((arm+'/full',baseline) for baseline in ('fit_constant','target_clip_mean_oracle',arm+'/static_predicted'))
        pairs.extend((arm+'/full',arm+'/'+mode) for mode in ('no_text','shuffled_text','reverse_audio'))
    kinds=('raw_level','centered_dynamics')
    comparisons={candidate+'_vs_'+baseline:{kind:{pop:paired_summary(stats[candidate][kind],stats[baseline][kind],
        reference['q']['sentence_id'],ids,samples=samples) for pop,ids in population.items()} for kind in kinds}
        for candidate,baseline in pairs}
    def table(rows):
        return {name:{kind:{pop:scalar_summary(value[ids]) for pop,ids in population.items()}
            for kind,value in fields.items()} for name,fields in rows.items()}
    return {'fit_constant':train_mean,'predicted_scores':table(stats),'driving_scores':table(drive_stats),
        'comparisons':comparisons,'target_clip_mean_oracle_is_target_conditioned':True,
        'centering':'Subtract each trajectory own observed clip mean using the same valid intensity mask; no query normalization in training.',
        'note':'Intensity regression is deterministic. Static/oracle interventions change driving intensity; predicted readouts can remain identical. Target clip mean is a diagnostic using GT, never an inference input.'}


def validate_common_initial(arms):
    baseline=arms['text']['epoch0']['motion']
    for seed in SEEDS:
        common_motion=baseline[str(seed)]['full']
        for arm in ARMS:
            for mode in ('full','zero'):
                if not torch.equal(arms[arm]['epoch0']['motion'][str(seed)][mode],common_motion):
                    raise ValueError('Epoch0 zero-init full/zero or matched-arm curves differ')
    return True


def analyze(arms,targets,reference,samples=2000):
    validate_common_initial(arms)
    multi={'noise_seeds':SEEDS,'decode_steps':12,'motion':{str(seed):{} for seed in SEEDS}}
    single={'42':{}}
    for seed in SEEDS:
        for mode in ('full','zero'):
            multi['motion'][str(seed)]['epoch0/'+mode]=arms['text']['epoch0']['motion'][str(seed)][mode]
    for arm,data in arms.items():
        for seed in SEEDS:
            for mode in ('full','zero'):multi['motion'][str(seed)][arm+'/'+mode]=data['final']['motion'][str(seed)][mode]
        for mode,value in data['final']['motion']['42'].items():single['42'][arm+'/'+mode]=value
    q=reference['q'];populations=audit_populations(q)
    stats,velocity,components=_condition_statistics(multi['motion'],reference,SEEDS)
    one_stats,one_velocity,one_components=_condition_statistics(single,reference,[42])
    distributions=audit_named_motion_samples(multi,reference,expected_seeds=SEEDS)
    comparisons={}
    for candidate,baseline in (('text/full','no_text/full'),('text/full','text/zero'),('no_text/full','no_text/zero'),
            ('text/full','epoch0/full'),('no_text/full','epoch0/full'),
            ('text/zero','epoch0/zero'),('no_text/zero','epoch0/zero')):
        name=candidate+'_vs_'+baseline
        comparisons[name]=_paired(stats,velocity,components,candidate,baseline,q,populations,samples)
        comparisons[name]['distribution']=common.distribution_pair(distributions,candidate,baseline,q,samples)
    interventions={}
    for arm in ARMS:
        for mode in ('no_text','static_intensity','shuffled_text','reverse_audio','oracle_intensity'):
            interventions[arm+'/full_vs_'+mode]=_paired(one_stats,one_velocity,one_components,arm+'/full',arm+'/'+mode,q,populations,samples)
    return {'multi_seed_scores':_score_table(stats,populations),'single_seed_scores':_score_table(one_stats,populations),
        'multi_seed_comparisons':comparisons,'single_seed_interventions':interventions,
        'distribution':distributions,'domain_multi_seed':domain_report(multi['motion'],q,populations),
        'domain_single_seed':domain_report(single,q,populations),'intensity':intensity_report(arms,targets,reference,samples),
        'scope':{'full_zero_noise_seeds':SEEDS,'epoch0_noise_seeds':SEEDS,'intervention_noise_seeds':[42],
            'epoch0_common_full_zero_and_arms_exact':True,'single_seed_is_not_distribution':True,
            'oracle_is_target_conditioned':True,'checkpoint_selection_performed':False,'test_loaded':False}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text-run',type=Path,required=True);p.add_argument('--no-text-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--samples',type=int,default=2000)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh audit output required')
    if args.samples<0:raise ValueError('Bootstrap samples must be nonnegative')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    arms={arm:load_arm(path,arm) for arm,path in (('text',args.text_run),('no_text',args.no_text_run))}
    recipes=[copy.deepcopy(v['recipe']) for v in arms.values()]
    for recipe in recipes:recipe.pop('arm')
    if recipes[0]!=recipes[1]:raise ValueError('Matched arm recipes differ beyond arm')
    recipe=arms['text']['recipe'];derived={k:torch.load(v,map_location='cpu',weights_only=False,mmap=True) for k,v in recipe['derived_paths'].items()}
    audio,text,targets=(derived[k] for k in ('audio','text','targets'))
    loaded=None;reconstruction={}
    for arm,data in arms.items():
        random.seed(46);np.random.seed(46);torch.manual_seed(46)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(46)
        loaded=runner.load_source(recipe['source_run'],args.device)
        for key in ('input_sha256','source_adapter_sha256','data_scope'):
            if loaded[key]!=recipe[key]:raise ValueError('Rebuilt source differs: '+key)
        runner.validate_inputs(loaded['cache'],audio,text,targets,recipe['input_sha256']['cache'])
        for initial,key in ((True,'epoch0'),(False,'final')):validate_saved_curves(data[key],loaded['cache']['splits']['validation'],arm,initial=initial)
        reconstruction[arm]=regenerate_prefix(data,loaded,audio,text,targets,args.device)
        if arm=='text':target_audit=audit_target_values(targets,loaded['cache'])
        if arm=='text':rng=common.validate_rng(arms,loaded)
        del loaded['system']
        if torch.cuda.is_available():torch.cuda.empty_cache()
    report=analyze(arms,targets,loaded['cache']['splits']['validation'],args.samples)
    report.update(schema=AUDIT_SCHEMA,reconstruction=reconstruction,target_audit=target_audit,training_rng=rng,
        run_hashes={arm:data['hashes'] for arm,data in arms.items()},audit_script_sha256=sha(__file__),
        bootstrap_samples=args.samples,
        audit_source_sha256={str(Path(module.__file__).resolve()):sha(module.__file__) for module in list(sys.modules.values())
            if module is not None and getattr(module,'__name__','').startswith(('scripts.','kinetalk_b0.'))
            and getattr(module,'__file__',None) and Path(module.__file__).is_file()},
        reconstruction_scope='First32 validation clips, seed42, epoch000 and final all stored modes; remaining saved curves hash-bound, not regenerated.')
    save_json(args.output,report)
    print(json.dumps({'output':str(args.output),'schema':AUDIT_SCHEMA,'reconstruction':'exact_first32_all_modes'}))


if __name__=='__main__':main()
