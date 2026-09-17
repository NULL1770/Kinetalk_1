"""Fixed-budget three-arm incremental adaptation on the historical sentence split.

All components inherit prior supervision on the full fit pool. Held-out sentences
are held out only from these new updates, never called unseen-encoder evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_audio_prefix_adaptation as a
from scripts import train_context_mechanism as context
from scripts import train_prefix_upper as p
from scripts.audit_temporal_repair import read, sha, load_pt, metadata_equal
from scripts.evaluate_prefix_formal import _same_bits
from scripts.probe_dynamic_predictability import sentence_split
from kinetalk_b0.models.temporal_local_adapter import TemporalLocalAdapter

SCHEMA = 'temporal_adapter_incremental_transfer_v1'
ARMS = ('frozen_local', 'rank8_adapter', 'full_local')
SPLIT_SEED = 2026091708
TRAIN_SEED = 89


def partition(q, historical):
    fit, hold = sentence_split(list(q['sentence_id']), SPLIT_SEED, heldout_fraction=.2)
    selected = {}
    for name, ix in (('fit', fit), ('internal_sentence_holdout', hold)):
        selected[name] = {'clips': [q['clip_id'][int(i)] for i in ix],
                         'sentences': sorted({str(q['sentence_id'][int(i)]) for i in ix})}
    if selected != historical['split'] or set(selected['fit']['sentences']) & set(selected['internal_sentence_holdout']['sentences']):
        raise ValueError('Historical metadata-only sentence partition differs')
    if [len(fit), len(hold)] != [1707, 608] or len(q['valid']) != 2315:
        raise ValueError('Expected original 2315 split into 1707/608')
    return fit, hold, {'schema': SCHEMA, 'seed': SPLIT_SEED, 'split': selected,
        'fit_indices': fit.tolist(), 'holdout_indices': hold.tolist(),
        'role': 'Held out from NEW updates only; inherited sources saw the full fit pool',
        'uses_motion_for_selection': False, 'test_loaded': False}


def configure(arm, original_local, device):
    if arm not in ARMS:
        raise ValueError('Unknown adaptation policy')
    local = a.configure_local(copy.deepcopy(original_local), arm == 'full_local')
    adapter = TemporalLocalAdapter(64, 8, init_seed=TRAIN_SEED).to(device).eval()
    adapter.requires_grad_(arm == 'rank8_adapter')
    return local, adapter


def conditioned_features(local, adapter, b, arm):
    native = a.local_features(local, b['audio_features'], b['valid'])
    return adapter(native, b['valid']) if arm == 'rank8_adapter' else native


@torch.no_grad()
def verify_initial_features(native, original_local, b):
    """Compare identical current-batch forward paths; log old-cache roundoff."""
    expected = a.local_features(original_local, b['audio_features'], b['valid'])
    if not _same_bits(native.detach(), expected):
        raise RuntimeError('Initial local differs from same-forward frozen source')
    delta = (expected-b['prefix_local'])[b['valid']]
    return {'same_forward_bit_exact': True, 'old_cache_max_abs': float(delta.abs().max()),
            'old_cache_rms': float(delta.double().square().mean().sqrt())}


@torch.no_grad()
def cache_condition(local, adapter, q, args, arm):
    parts, rows = [], []
    for ix in torch.arange(len(q['valid'])).split(args.batch_size):
        b = p.r.subset(q, ix, args.device)
        direct = a.local_features(local, b['audio_features'], b['valid'])
        native = adapter(direct, b['valid']) if arm == 'rank8_adapter' else direct
        original = b['prefix_local']
        for j in range(len(ix)):
            delta = (native[j][b['valid'][j]] - original[j][b['valid'][j]]).double()
            mean = delta.mean(0)
            rows.append({'clip_id': b['clip_id'][j], 'sentence_id': str(b['sentence_id'][j]),
                         'correction_rms': float(delta.square().mean().sqrt()),
                         'mean_correction_max_abs': float(mean.abs().max()),
                         'same_forward_mean_correction_max_abs': float((native[j][b['valid'][j]].double()-direct[j][b['valid'][j]].double()).mean(0).abs().max()),
                         'centered_correction_rms': float((delta-mean).square().mean().sqrt())})
        parts.append(native.cpu())
    return {**q, 'prefix_local': torch.cat(parts)}, rows


def compact_deployment(curves, bases, baseline_path, baseline_hash):
    """Save upper9 only after exact full-face preservation checks; bind the base."""
    metadata_equal(curves, bases)
    for key, value in curves['predictions'].items():
        baseline = bases['predictions'][key.split('/')[0]+'/base']
        if (not _same_bits(value[..., list(p.r.NOT_UPPER)], baseline[..., list(p.r.NOT_UPPER)])
                or not _same_bits(value[~curves['valid']], baseline[~curves['valid']])):
            raise ValueError('Protected baseline channels or invalid frames differ')
    return {**{k:v for k,v in curves.items() if k != 'predictions'},
        'storage_schema': 'upper9_with_exact_bound_base_v1',
        'upper_predictions9': {k:v[..., p.CC].clone() for k,v in curves['predictions'].items()},
        'upper_indices': list(p.CC), 'baseline_path': str(baseline_path), 'baseline_sha256': baseline_hash,
        'nonupper_invalid_exact_before_storage': True,
        'reconstruction': 'Clone seed-specific bound base; replace upper9 on valid frames only'}


def restore_deployment(compact, bases, baseline_hash):
    if (compact['storage_schema'] != 'upper9_with_exact_bound_base_v1'
            or compact['upper_indices'] != list(p.CC) or compact['baseline_sha256'] != baseline_hash):
        raise ValueError('Compact base binding differs')
    metadata_equal(compact, bases)
    predictions = {}
    for key, upper in compact['upper_predictions9'].items():
        baseline = bases['predictions'][key.split('/')[0]+'/base']
        predictions[key] = p.compose_upper_face(baseline, upper, compact['valid'])
    return {**compact, 'predictions': predictions}


def gradient_record(local, adapter):
    modules = {name:getattr(local, name) for name in ('input','blocks','local_head')}
    modules.update(adapter_down=adapter.down, adapter_up=adapter.up)
    return {name: {'absolute_sum': sum(float(v.grad.abs().sum()) for v in module.parameters() if v.grad is not None),
                   'has_gradient': any(v.grad is not None for v in module.parameters())}
            for name,module in modules.items()}


def check_gradients(arm, records):
    for row in records.values():
        for name in ('input','blocks','local_head'):
            expected = arm == 'full_local'
            if row[name]['has_gradient'] != expected or (expected and row[name]['absolute_sum'] <= 0):
                raise RuntimeError('Local gradient policy differs: '+name)
        if arm != 'rank8_adapter' and any(row[n]['has_gradient'] for n in ('adapter_up','adapter_down')):
            raise RuntimeError('Disabled adapter received gradient')
    if arm == 'rank8_adapter':
        if records['1']['adapter_up']['absolute_sum'] <= 0 or records['2']['adapter_down']['absolute_sum'] <= 0:
            raise RuntimeError('Adapter gradient path missing after zero-initialized first step')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','audio','targets','enrollment','native-root','trained-run','history-run',
                 'centered-run','context-run','split-report','prior-audit','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=TRAIN_SEED)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); args.pilot = False; args.pilot_selection = None
    if args.output.exists(): raise FileExistsError('Fresh experiment output required')
    if (args.epochs,args.batch_size,args.seed)!=(12,16,TRAIN_SEED): raise ValueError('Fixed recipe')
    if shutil.disk_usage(args.output.parent).free < 1200*1024**2:
        raise RuntimeError('Need1.2GB free, retaining all historical artifacts')
    prior = read(args.prior_audit/'complete.json')
    if sha(args.prior_audit/'report.json') != prior['report_sha256']:
        raise ValueError('Prerequisite diagnostic binding differs')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True; began=time.monotonic()
    data, system, audio, original_local, identities, old_source, source_recipe, _, steps, _, frozen = p.load_context(args)
    if steps != 12 or len(data['splits']['validation']['valid']) != 405:
        raise ValueError('Data/solver contract differs')
    sourcepath=args.context_run/'chunk_teacher/final.pt'; source=load_pt(sourcepath)
    complete=read(sourcepath.with_name('complete.json')); previous=read(sourcepath.with_name('provenance.json'))
    if (sha(sourcepath)!=complete['final_sha256'] or source['schema']!=context.SCHEMA or source['arm']!='chunk_teacher'
            or source['completed_epochs']!=12 or source['recipe_sha256']!=complete['recipe_sha256']
            or source['recipe_sha256']!=previous['recipe_sha256'] or previous['recipe_sha256']!=p.canonical_hash(previous['recipe'])
            or source['frozen']!=frozen or not torch.equal(source['scales'],old_source['scales'])
            or previous['recipe']['source_recipe_sha256']!=p.canonical_hash(source_recipe)):
        raise ValueError('Context source contract differs')
    basepath=args.centered_run/'white/curves.pt'; basehash=sha(basepath); bases=load_pt(basepath)
    if basehash != source_recipe['baseline_curves_sha256']: raise ValueError('Base file changed')
    metadata_equal({**data['splits']['validation'],'target':data['splits']['validation']['motion']},bases)
    fit_ids,hold_ids,selection=partition(data['splits']['train'],read(args.split_report))
    train=p.r.subset(data['splits']['train'],fit_ids[:32] if args.smoke else fit_ids,'cpu')
    hold=p.r.subset(data['splits']['train'],hold_ids[:16] if args.smoke else hold_ids,'cpu')
    fit_diag_ids,fit_selection=context.fixed_fit_selection(train,count=8 if args.smoke else 128)
    fit_diag=p.r.subset(train,fit_diag_ids,'cpu')
    if args.smoke:
        ix=torch.arange(32)
        data['splits']['validation']=p.r.subset(data['splits']['validation'],ix,'cpu')
        bases={**bases,**p.r.subset({k:bases[k] for k in ('clip_id','target','valid','times','channel_mask','b0','emotion_id','speaker_id')},ix,'cpu'),
               'predictions':{k:v[:32] for k,v in bases['predictions'].items()}}
    scales=source['scales'].to(args.device); epochs=1 if args.smoke else 12
    args.output.mkdir(); p.save_json(args.output/'sentence_split.json',selection)
    p.save_json(args.output/'fit_selection.json',fit_selection)
    from scripts.evaluate_temporal_adapter_transfer import evaluate_transfer
    from scripts.evaluate_audio_prefix_adaptation import evaluate_compositions
    matched={}
    for arm in ARMS:
        torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
        upper=p.PrefixUpperFlow(data['config']).to(args.device).eval(); upper.load_state_dict(source['upper'],strict=True)
        local,adapter=configure(arm,original_local,args.device)
        before={'upper':p.state_hash(upper.state_dict()),'local':p.state_hash(local.state_dict()),'adapter':p.state_hash(adapter.state_dict())}
        protected=p.state_hash(a.protected_local_state(local)); output=args.output/arm;output.mkdir()
        if arm==ARMS[0]:
            p.save_checkpoint(args.output/'initial.pt',{'upper':upper.state_dict(),'local':local.state_dict(),'adapter':adapter.state_dict()})
            # All arms are identical at step0, including the zero adapter. Bind
            # the same held-out seed streams to the shared source checkpoint.
            p.save_json(args.output/'status.json',{'status':'step0_transfer','smoke':args.smoke})
            hold0,hold0_drift=cache_condition(local,adapter,hold,args,arm)
            step_report,step_curves=evaluate_transfer(upper,hold0,identities,scales,args,steps)
            p.save_json(args.output/'step0_feature_drift.json',{'transfer':hold0_drift,
                'scope':'Old cache versus identical specialized batch forward used at step0 and endpoints'})
            p.save_json(args.output/'step0_transfer.json',step_report);p.save_checkpoint(args.output/'step0_transfer.pt',step_curves)
            source_curves_path=args.context_run/'chunk_teacher/curves.pt'
            if sha(source_curves_path)!=complete['curves_sha256']:raise ValueError('Source curve binding changed')
            source_curves=load_pt(source_curves_path);q0=data['splits']['validation'];maximum=0.
            noise=torch.randn(len(q0['valid']),96,9,generator=torch.Generator().manual_seed(42))
            with torch.no_grad():
                for ix in torch.arange(min(32,len(q0['valid']))).split(16):
                    b=p.r.subset(q0,ix,args.device);ident=p.r.batch_identity(identities,b)
                    native=conditioned_features(local,adapter,b,arm)
                    if not _same_bits(adapter(native,b['valid']),native):raise RuntimeError('Adapter step0 is not exact')
                    conditions={k:b[k] for k in ('valid','h0','audio_global','audio_intensity')}
                    dyn=context.decode_context(upper,conditions,ident,native,noise[ix].to(args.device),arm='chunk_teacher')
                    diff=(b['static_upper'][:,None]+dyn*scales-source_curves['predictions']['42/full'][ix][...,p.CC].to(args.device))[b['valid']]
                    maximum=max(maximum,float(diff.abs().max()))
            if maximum>2e-6:raise RuntimeError('Step0 source replay differs')
            p.save_json(args.output/'step0_replay.json',{'clips':32,'seed':42,'max_abs':maximum,'adapter_step0_bit_exact':True,
                'source_curves_sha256':sha(source_curves_path)})
            del source_curves,step_curves
        recipe={'schema':SCHEMA,'arm':arm,'epochs':epochs,'requested_epochs':12,'batch_size':16,'seed':TRAIN_SEED,
            'rank':8,'adapter_parameters':1024,'fit_update_clips':len(train['valid']),'transfer_clips':len(hold['valid']),
            'source_sha256':sha(sourcepath),'source_recipe_sha256':source['recipe_sha256'],'frozen':frozen,
            'initial':before,'initial_file_sha256':sha(args.output/'initial.pt'),'scales':scales.cpu().tolist(),
            'data_provenance':data['provenance'],'baseline_curves_sha256':basehash,'config':data['config'],'solver_steps':steps,
            'sentence_split_sha256':sha(args.output/'sentence_split.json'),'historical_split_sha256':sha(args.split_report),
            'prior_diagnostic_sha256':sha(args.prior_audit/'report.json'),'fit_selection_sha256':sha(args.output/'fit_selection.json'),
            'smoke':args.smoke,'test_loaded':False,'default_replaced':False,'fixed_final_epoch':True,
            'upper_trainable':True,'local_trainable':arm=='full_local','adapter_trainable':arm=='rank8_adapter',
            'objective':'Unchanged unknown-only FM; GT strictly-past training; generated past deployment',
            'transfer_scope':selection['role'],'DC_composition':'Separate offline audio-mean translation, no GT or history feedback',
            'code_sha256':{n:sha(Path(__file__).resolve().parents[1]/n) for n in (
                'scripts/train_temporal_adapter_transfer.py','scripts/evaluate_temporal_adapter_transfer.py',
                'scripts/train_audio_prefix_adaptation.py','scripts/evaluate_audio_prefix_adaptation.py',
                'scripts/train_context_mechanism.py','scripts/evaluate_context_mechanism.py',
                'kinetalk_b0/models/temporal_local_adapter.py','kinetalk_b0/models/prefix_upper_flow.py',
                'docs/TEMPORAL_ADAPTER_TRANSFER_PROTOCOL_20260917.md')}}
        digest=p.canonical_hash(recipe);p.save_json(output/'provenance.json',{'recipe':recipe,'recipe_sha256':digest})
        params=list(upper.parameters())+[v for v in local.parameters() if v.requires_grad]+[v for v in adapter.parameters() if v.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-5)
        gen=torch.Generator().manual_seed(TRAIN_SEED);draws=[];total_steps=0;gradients={};initial_features=None
        for epoch in range(epochs):
            t0=time.monotonic();losses=[];draw=hashlib.sha256()
            for ix in torch.randperm(len(train['valid']),generator=gen).split(16):
                b=p.r.subset(train,ix,args.device);ident=p.r.batch_identity(identities,b)
                native=conditioned_features(local,adapter,b,arm)
                if total_steps==0:
                    initial_features=verify_initial_features(native,original_local,b)
                loss,randoms=context.context_batch(upper,b,ident,native,scales,gen,'chunk_teacher')
                for value in (ix,*randoms):draw.update(value.numpy().tobytes())
                p.r.optimize(loss,optimizer,params);losses.append(float(loss.detach()));total_steps+=1
                if total_steps<=2:gradients[str(total_steps)]=gradient_record(local,adapter)
            check_gradients(arm,gradients)
            row={'arm':arm,'epoch':epoch+1,'loss':sum(losses)/len(losses),'seconds':time.monotonic()-t0,
                 'total_steps':total_steps,'draw_sha256':draw.hexdigest(),'first_two_step_gradients':gradients,
                 'initial_features':initial_features}
            draws.append(draw.hexdigest());p.save_json(output/f'epoch{epoch+1:03d}.json',row)
            p.save_json(args.output/'status.json',{'status':'training',**row});print(row,flush=True)
            payload={'schema':SCHEMA,'arm':arm,'upper':upper.state_dict(),'local':local.state_dict(),'adapter':adapter.state_dict(),
                     'scales':scales.cpu(),'completed_epochs':epoch+1,'total_steps':total_steps,'recipe_sha256':digest,'frozen':frozen}
            if shutil.disk_usage(args.output).free<512*1024**2:raise RuntimeError('Storage below512MB; old artifacts untouched')
            p.save_checkpoint(output/'last.pt',{**payload,'optimizer':optimizer.state_dict(),'rng':p.capture_rng(gen)})
        for name,module in (('system',system),('audio',audio),('local',original_local)):
            if p.state_hash(module.state_dict())!=frozen[name] or any(v.grad is not None for v in module.parameters()):
                raise RuntimeError('Original frozen module changed: '+name)
        if p.state_hash(a.protected_local_state(local))!=protected:raise RuntimeError('Protected local head changed')
        if (p.state_hash(local.state_dict())!=before['local'])!=(arm=='full_local'):raise RuntimeError('Local update policy differs')
        if (p.state_hash(adapter.state_dict())!=before['adapter'])!=(arm=='rank8_adapter'):raise RuntimeError('Adapter update policy differs')
        p.save_json(args.output/'status.json',{'status':'evaluating','arm':arm,'completed_epochs':epochs})
        if shutil.disk_usage(args.output).free<700*1024**2:
            raise RuntimeError('Need700MB before compact evaluation writes; old artifacts untouched')
        q,dev_drift=cache_condition(local,adapter,data['splits']['validation'],args,arm)
        h,hold_drift=cache_condition(local,adapter,hold,args,arm)
        f,fit_drift=cache_condition(local,adapter,fit_diag,args,arm)
        if arm=='rank8_adapter' and max(r['same_forward_mean_correction_max_abs'] for r in dev_drift+hold_drift+fit_drift)>2e-6:
            raise RuntimeError('Adapter local mean shifted')
        p.save_json(output/'feature_drift.json',{'development':dev_drift,'transfer':hold_drift,'fit':fit_drift})
        report,curves=context.evaluate(upper,system,q,identities,scales,bases,args,steps,'chunk_teacher',context.decode_context)
        a.extra_local_interventions(upper,system,q,identities,scales,bases,args,steps,report,curves)
        report.update(schema=SCHEMA,adaptation_arm=arm,recipe_sha256=digest)
        curves.update(schema=SCHEMA,adaptation_arm=arm,recipe_sha256=digest,static_upper=q['static_upper'])
        dc_report,dc_curves=evaluate_compositions(system,q,identities,bases,curves,args,a.compose_dc)
        p.save_json(output/'evaluation.json',report);p.save_json(output/'dc_evaluation.json',dc_report)
        for label,value in (('curves',curves),('dc_curves',dc_curves)):
            compact=compact_deployment(value,bases,basepath,basehash)
            reconstructed=restore_deployment(compact,bases,basehash)
            if any(not _same_bits(v,reconstructed['predictions'][k]) for k,v in value['predictions'].items()):
                raise RuntimeError('Compact representation changed output')
            p.save_checkpoint(output/(label+'.pt'),compact)
        del curves,dc_curves,compact,reconstructed
        transfer_report,transfer_curves=evaluate_transfer(upper,h,identities,scales,args,steps)
        p.save_json(output/'transfer_evaluation.json',transfer_report);p.save_checkpoint(output/'transfer_curves.pt',transfer_curves)
        fit_report,fit_curves=context.fit_diagnostic(upper,f,identities,scales,args,steps,'chunk_teacher')
        p.save_json(output/'fit_evaluation.json',fit_report);p.save_checkpoint(output/'fit_curves.pt',fit_curves)
        p.save_checkpoint(output/'final.pt',payload)
        files={name:sha(output/name) for name in ('final.pt','curves.pt','dc_curves.pt','transfer_curves.pt','fit_curves.pt',
                'evaluation.json','dc_evaluation.json','transfer_evaluation.json','fit_evaluation.json','feature_drift.json')}
        p.save_json(output/'complete.json',{'status':'complete','files':files,'completed_epochs':epochs,'total_steps':total_steps,
            'recipe_sha256':digest,'frozen':frozen,'nonupper_invalid_exact':True,'compact_roundtrip_exact':True,
            'gradient_audit':gradients,'test_loaded':False})
        matched[arm]={'initial':before,'draws':draws,'updates':total_steps}
    if any(v!=matched[ARMS[0]] for v in matched.values()):raise RuntimeError('Matched initialization/RNG/update budget differs')
    p.save_json(args.output/'matched_audit.json',{'equal':True,'arms':matched})
    p.save_json(args.output/'status.json',{'status':'complete','smoke':args.smoke,'epochs_per_arm':epochs,
        'seconds':time.monotonic()-began,'transfer_scope':selection['role']})
    print('TEMPORAL_ADAPTER_TRANSFER_COMPLETE',flush=True)


if __name__=='__main__':main()
