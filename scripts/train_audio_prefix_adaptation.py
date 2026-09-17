"""Paired continuation: fixed versus trainable upper-only acoustic features."""
from __future__ import annotations

import argparse
import copy
import hashlib
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_context_mechanism as context
from scripts import train_prefix_upper as p
from scripts.audit_temporal_repair import read, sha, load_pt, metadata_equal
from kinetalk_b0.models.mean_preserving_upper import compose_mean_preserving_upper

SCHEMA = 'audio_prefix_adaptation_v1'
ARMS = ('frozen_local', 'adapt_local')


def configure_local(local, trainable):
    if type(trainable) is not bool:
        raise ValueError('Explicit Boolean local training policy required')
    local.eval().requires_grad_(False)
    for name in ('input', 'blocks', 'local_head'):
        getattr(local, name).requires_grad_(trainable)
    return local


def local_features(local, features, valid):
    """Equivalent native features; unused global/state heads never execute."""
    if (features.ndim != 3 or valid.shape != features.shape[:2] or valid.dtype != torch.bool
            or features.shape[-1] != len(local.feature_mean) or not valid.any(1).all()
            or not torch.isfinite(features[valid]).all()):
        raise ValueError('Finite observed native acoustic features required')
    clean = torch.where(valid[..., None], features, local.feature_mean)
    value = (clean-local.feature_mean)/local.feature_std
    hidden = torch.where(valid[..., None], F.silu(local.input(value)), 0.)
    for block in local.blocks:
        hidden = block(hidden, valid)
    return torch.where(valid[..., None], local.local_head(hidden), 0.)


def compose_dc(baseline, raw_upper, static_upper, valid):
    if static_upper.shape != (len(baseline), 9) or not torch.isfinite(static_upper).all():
        raise ValueError('Finite deployable static mean [B,9] required')
    fixed = baseline.clone()
    fixed[..., p.CC] = static_upper[:, None]
    fixed[~valid] = baseline[~valid]
    return compose_mean_preserving_upper(fixed, raw_upper, valid)


@torch.no_grad()
def cache_local(local, q, args):
    values = []
    for ix in torch.arange(len(q['valid'])).split(args.batch_size):
        b = p.r.subset(q, ix, args.device)
        values.append(local_features(local, b['audio_features'], b['valid']).cpu())
    return {**q, 'prefix_local': torch.cat(values)}


def protected_local_state(local):
    return {key: value for key, value in local.state_dict().items()
            if not key.startswith(('input.', 'blocks.', 'local_head.'))}


@torch.no_grad()
def extra_local_interventions(upper, system, q, identities, scales, bases, args, steps, report, curves):
    from scripts.evaluate_prefix_formal import chunk_diagnostics
    noise = torch.randn(len(q['valid']), 96, 9, generator=torch.Generator().manual_seed(42))
    for mode in ('local_static', 'local_reverse'):
        outputs, correct = [], 0
        for ix in torch.arange(len(q['valid'])).split(args.batch_size):
            b = p.r.subset(q, ix, args.device); ident = p.r.batch_identity(identities, b)
            local, _ = p.time_intervention(b['prefix_local'], b['h0'], b['valid'], mode.removeprefix('local_'))
            condition = {key: b[key] for key in ('valid', 'h0', 'audio_global', 'audio_intensity')}
            dynamic = context.decode_context(upper, condition, ident, local, noise[ix].to(args.device),
                steps=steps, mode='full', arm='chunk_teacher')
            baseline = bases['predictions']['42/base'][ix].to(args.device)
            pred = p.compose_upper_face(baseline, b['static_upper'][:, None]+dynamic*scales, b['valid'])
            if not torch.equal(pred[..., list(p.r.NOT_UPPER)], baseline[..., list(p.r.NOT_UPPER)]) or not torch.equal(pred[~b['valid']], baseline[~b['valid']]):
                raise RuntimeError('Local intervention changed protected output')
            logits = system.encode_motion(torch.where(p.r.obs(b), pred-b['b0']-ident['baseline'][:, None], 0.), b['valid'])['emotion_logits']
            correct += int((logits.argmax(-1)==b['emotion_id']).sum()); outputs.append(pred.cpu())
        value = torch.cat(outputs); key = '42/'+mode; curves['predictions'][key] = value
        report['modes'][key] = {'populations': p.r.populations(value, q),
            'temporal': p.temporal_stats(value, q['motion'], q['valid']),
            'boundaries': p.h.boundary_report(value, q['motion'], q['valid']),
            'chunk_diagnostics': chunk_diagnostics(value, q['motion'], q['valid'], q['channel_mask']),
            'generated_emotion_accuracy_nonindependent': correct/len(q['valid']),
            'intervention': 'Only upper-local feature time order; h0/global/static/identity unchanged'}
    # Existing evaluator separates oracle before aggregation. Preserve that
    # boundary while adding the two local-only intervention scores.
    deploy = {**curves, 'predictions': {key: value for key, value in curves['predictions'].items() if '/oracle_' not in key}}
    summary = p.summarize(deploy, q['emotion_id'])
    report['deployable_interventions_seed42'] = summary.pop('single_seed_interventions')
    report['distribution'] = summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'audio', 'targets', 'enrollment', 'native-root', 'trained-run',
                 'history-run', 'centered-run', 'context-run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=89)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); args.pilot = False; args.pilot_selection = None
    if args.output.exists(): raise FileExistsError('Fresh adaptation output required')
    if (args.epochs, args.batch_size, args.seed) != (12, 16, 89): raise ValueError('Fixed paired recipe')
    if shutil.disk_usage(args.output.parent).free < 1200*1024**2:
        raise RuntimeError('Need at least 1.2GB free for paired artifacts without deleting old runs')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True; began = time.monotonic()
    data, system, audio, original_local, identities, old_source, source_recipe, _, steps, _, frozen = p.load_context(args)
    if steps != 12 or [len(data['splits'][k]['valid']) for k in ('train','validation')] != [2315,405]:
        raise ValueError('Source contract mismatch')
    sourcepath = args.context_run/'chunk_teacher/final.pt'
    source = load_pt(sourcepath); complete = read(sourcepath.with_name('complete.json'))
    previous = read(sourcepath.with_name('provenance.json'))
    if (sha(sourcepath) != complete['final_sha256'] or source['schema'] != context.SCHEMA
            or source['arm'] != 'chunk_teacher' or source['completed_epochs'] != 12
            or source['recipe_sha256'] != p.canonical_hash(previous['recipe'])
            or previous['recipe_sha256'] != source['recipe_sha256']
            or source['frozen'] != frozen or not torch.equal(source['scales'], old_source['scales'])):
        raise ValueError('Context checkpoint lineage mismatch')
    if previous['recipe']['source_recipe_sha256'] != p.canonical_hash(source_recipe):
        raise ValueError('Context frozen acoustic source differs')
    basepath = args.centered_run/'white/curves.pt'; bases = load_pt(basepath)
    if sha(basepath) != source_recipe['baseline_curves_sha256']:
        raise ValueError('Complete baseline binding mismatch')
    metadata_equal({**data['splits']['validation'], 'target': data['splits']['validation']['motion']}, bases)
    selection_ids, selection = context.fixed_fit_selection(data['splits']['train'])
    if selection != read(args.context_run/'fit_selection.json'):
        raise ValueError('Historical fixed fit selection changed')
    fit = p.r.subset(data['splits']['train'], selection_ids[:8] if args.smoke else selection_ids, 'cpu')
    if args.smoke:
        data['splits'] = {key: p.r.subset(q, torch.arange(32), 'cpu') for key,q in data['splits'].items()}
        bases = {**bases, **p.r.subset({key:bases[key] for key in ('clip_id','target','valid','times','channel_mask','b0','emotion_id','speaker_id')},torch.arange(32),'cpu'),
                 'predictions': {key:value[:32] for key,value in bases['predictions'].items()}}
    scales = source['scales'].to(args.device); train = data['splits']['train']
    args.output.mkdir(); p.save_json(args.output/'fit_selection.json', selection)
    epochs = 1 if args.smoke else 12; matched = {}
    for arm in ARMS:
        torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
        upper = p.PrefixUpperFlow(data['config']).to(args.device).eval()
        upper.load_state_dict(source['upper'], strict=True)
        local = configure_local(copy.deepcopy(original_local), arm=='adapt_local')
        before = {'upper':p.state_hash(upper.state_dict()), 'local':p.state_hash(local.state_dict())}
        protected = p.state_hash(protected_local_state(local)); output = args.output/arm; output.mkdir()
        if arm == ARMS[0]:
            p.save_checkpoint(args.output/'initial.pt', {'upper':upper.state_dict(),'local':local.state_dict()})
            source_curves_path = args.context_run/'chunk_teacher/curves.pt'
            if sha(source_curves_path) != complete['curves_sha256']:
                raise ValueError('Source curves binding changed')
            source_curves = load_pt(source_curves_path)
            # Replay a fixed metadata prefix only to verify the unchanged
            # receiver/feature interface; never select a seed or checkpoint.
            maximum = 0.
            with torch.no_grad():
                q0 = data['splits']['validation']
                initial_noise = torch.randn(len(q0['valid']),96,9,generator=torch.Generator().manual_seed(42))
                for ix in torch.arange(min(32,len(q0['valid']))).split(16):
                    b = p.r.subset(q0,ix,args.device); ident=p.r.batch_identity(identities,b)
                    native=local_features(local,b['audio_features'],b['valid'])
                    conditions={key:b[key] for key in ('valid','h0','audio_global','audio_intensity')}
                    value=context.decode_context(upper,conditions,ident,native,initial_noise[ix].to(args.device),arm='chunk_teacher')
                    raw=b['static_upper'][:,None]+value*scales
                    difference=(raw-source_curves['predictions']['42/full'][ix][...,p.CC].to(args.device))[b['valid']]
                    maximum=max(maximum,float(difference.abs().max()))
            if maximum>2e-6: raise RuntimeError('Step0 source replay differs: '+str(maximum))
            p.save_json(args.output/'step0_replay.json',{'clips':32,'seed':42,'max_abs':maximum,
                'source_curves_sha256':sha(source_curves_path),'no_fitting':True})
            del source_curves
        recipe = {'schema':SCHEMA, 'arm':arm, 'epochs':epochs, 'requested_epochs':12, 'seed':89, 'batch_size':16,
            'source_sha256':sha(sourcepath), 'source_recipe_sha256':source['recipe_sha256'], 'scales':scales.cpu().tolist(),
            'initial':before, 'initial_file_sha256':sha(args.output/'initial.pt'), 'frozen':frozen,
            'data_provenance':data['provenance'], 'baseline_curves_sha256':sha(basepath),
            'fit_selection_sha256':sha(args.output/'fit_selection.json'), 'smoke':args.smoke,
            'config':data['config'], 'solver_steps':steps, 'same_noise_time_and_order':True,
            'test_loaded':False,'default_replaced':False,'fixed_final_epoch':True,
            'upper_trainable':True,'local_trainable':arm=='adapt_local', 'local_protected':protected,
            'objective':'Existing standard unknown-only FM, GT strictly-past training, generated-past deployment',
            'DC_composition':'Separate offline prediction-only mean translation; never enters training or oracle continuation',
            'code_sha256':{name:sha(Path(__file__).resolve().parents[1]/name) for name in (
                'scripts/train_audio_prefix_adaptation.py','scripts/evaluate_audio_prefix_adaptation.py',
                'scripts/train_context_mechanism.py','scripts/evaluate_context_mechanism.py',
                'kinetalk_b0/models/prefix_upper_flow.py','kinetalk_b0/models/mean_preserving_upper.py',
                'docs/AUDIO_PREFIX_ADAPTATION_PROTOCOL_20260917.md')}}
        digest = p.canonical_hash(recipe); p.save_json(output/'provenance.json',{'recipe':recipe,'recipe_sha256':digest})
        params = list(upper.parameters())+[param for param in local.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
        gen = torch.Generator().manual_seed(89); draws = []; total_steps = 0; gradient_audit = {}
        for epoch in range(epochs):
            t0 = time.monotonic(); losses = []; draw = hashlib.sha256()
            for selected in torch.randperm(len(train['valid']), generator=gen).split(16):
                b = p.r.subset(train, selected, args.device); ident = p.r.batch_identity(identities,b)
                native = local_features(local,b['audio_features'],b['valid'])
                if total_steps == 0 and not torch.allclose(native.detach(),b['prefix_local'],atol=2e-6,rtol=1e-5):
                    raise RuntimeError('Specialized local forward differs from source cache')
                loss, randoms = context.context_batch(upper,b,ident,native,scales,gen,'chunk_teacher')
                for tensor in (selected,*randoms): draw.update(tensor.numpy().tobytes())
                p.r.optimize(loss,optimizer,params); losses.append(float(loss.detach()));total_steps+=1
                if total_steps == 1:
                    for name in ('input','blocks','local_head'):
                        values=[param.grad for param in getattr(local,name).parameters()]
                        energy=sum(float(value.abs().sum()) for value in values if value is not None)
                        gradient_audit[name]={'absolute_sum':energy,'has_gradient':any(value is not None for value in values)}
                        if arm=='adapt_local' and not energy>0: raise RuntimeError('Missing local gradient: '+name)
                        if arm=='frozen_local' and any(value is not None for value in values): raise RuntimeError('Frozen local received gradient')
            row={'arm':arm,'epoch':epoch+1,'loss':sum(losses)/len(losses),'seconds':time.monotonic()-t0,
                 'total_steps':total_steps,'draw_sha256':draw.hexdigest(),'gradient_audit_first_step':gradient_audit}
            draws.append(draw.hexdigest());p.save_json(output/f'epoch{epoch+1:03d}.json',row)
            p.save_json(args.output/'status.json',{'status':'training',**row});print(row,flush=True)
            payload={'schema':SCHEMA,'arm':arm,'upper':upper.state_dict(),'local':local.state_dict(),
                     'scales':scales.cpu(),'completed_epochs':epoch+1,'total_steps':total_steps,'recipe_sha256':digest,'frozen':frozen}
            if shutil.disk_usage(args.output).free < 512*1024**2: raise RuntimeError('Free storage below512MB; old experiments untouched')
            p.save_checkpoint(output/'last.pt',{**payload,'optimizer':optimizer.state_dict(),'rng':p.capture_rng(gen)})
        for name,module in (('system',system),('audio',audio),('original_local',original_local)):
            expected=frozen['local' if name=='original_local' else name]
            if p.state_hash(module.state_dict())!=expected or any(param.grad is not None for param in module.parameters()):
                raise RuntimeError('Frozen module changed: '+name)
        if p.state_hash(protected_local_state(local))!=protected:
            raise RuntimeError('Unused local global/state head or normalization changed')
        for name,param in local.named_parameters():
            if not name.startswith(('input.','blocks.','local_head.')) and param.grad is not None:
                raise RuntimeError('Unused head received gradients')
        local_changed=p.state_hash(local.state_dict())!=before['local']
        if local_changed != (arm=='adapt_local'): raise RuntimeError('Acoustic update policy failed')
        p.save_json(args.output/'status.json',{'status':'evaluating','arm':arm,'completed_epochs':epochs})
        q = cache_local(local,data['splits']['validation'],args)
        report,curves=context.evaluate(upper,system,q,identities,scales,bases,args,steps,'chunk_teacher',context.decode_context)
        extra_local_interventions(upper,system,q,identities,scales,bases,args,steps,report,curves)
        report.update(schema=SCHEMA,adaptation_arm=arm,recipe_sha256=digest)
        curves.update(schema=SCHEMA,adaptation_arm=arm,recipe_sha256=digest,static_upper=q['static_upper'])
        from scripts.evaluate_audio_prefix_adaptation import evaluate_compositions
        dc_report,dc_curves=evaluate_compositions(system,q,identities,bases,curves,args,compose_dc)
        p.save_json(output/'evaluation.json',report);p.save_checkpoint(output/'curves.pt',curves)
        p.save_json(output/'dc_evaluation.json',dc_report);p.save_checkpoint(output/'dc_curves.pt',dc_curves)
        fit_report,fit_curves=context.fit_diagnostic(upper,cache_local(local,fit,args),identities,scales,args,steps,'chunk_teacher')
        p.save_json(output/'fit_evaluation.json',fit_report);p.save_checkpoint(output/'fit_curves.pt',fit_curves)
        p.save_checkpoint(output/'final.pt',payload)
        p.save_json(output/'complete.json',{'final_sha256':sha(output/'final.pt'),'curves_sha256':sha(output/'curves.pt'),
            'dc_curves_sha256':sha(output/'dc_curves.pt'),'fit_curves_sha256':sha(output/'fit_curves.pt'),
            'completed_epochs':epochs,'total_steps':total_steps,'recipe_sha256':digest,'frozen':frozen,
            'local_protected_unchanged':True,'local_changed':local_changed,'gradient_audit':gradient_audit})
        matched[arm]={'initial':before,'draws':draws,'updates':total_steps}
    if matched[ARMS[0]]!=matched[ARMS[1]]: raise RuntimeError('Paired initialization/randomness/budget failed')
    p.save_json(args.output/'matched_audit.json',{'equal':True,'arms':matched})
    p.save_json(args.output/'status.json',{'status':'complete','epochs_per_arm':epochs,'smoke':args.smoke,'seconds':time.monotonic()-began})
    print('AUDIO_PREFIX_ADAPTATION_COMPLETE',flush=True)


if __name__=='__main__':main()
