"""Matched thirty-epoch center versus full-native continuation experiment."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_prefix_upper as p
from scripts import train_audio_prefix_adaptation as a
from scripts import train_context_mechanism as context
from scripts.audit_temporal_repair import read, sha, load_pt, metadata_equal
from scripts.train_temporal_adapter_transfer import partition
from scripts.full_native_context_data import NativeContextStore
from scripts.native_context_runtime import encode_full_conditions, variable_context_loss, paired_noise
from scripts.train_formal_predictable_projection import restore_rng

SCHEMA = 'native_context_coverage30_v1'
ARMS = ('center96', 'full_native')
TRAIN_SEED = 97
ENCODED = ('h0', 'audio_global', 'audio_intensity', 'static_upper')
CODE_FILES = (
    'scripts/train_native_context.py', 'scripts/evaluate_native_context.py',
    'scripts/full_native_context_data.py', 'scripts/prepare_full_native_audio_delta.py',
    'scripts/native_context_runtime.py', 'scripts/train_prefix_upper.py',
    'scripts/train_context_mechanism.py', 'scripts/train_audio_prefix_adaptation.py',
    'kinetalk_b0/models/prefix_upper_flow.py', 'kinetalk_b0/models/temporal_upper.py',
    'docs/NATIVE_CONTEXT30_PROTOCOL_20260918.md',
)


def cast_batch(b, device):
    return {k: v.to(device=device, dtype=torch.float32 if v.is_floating_point() and k != 'times' else v.dtype)
            if torch.is_tensor(v) else v for k, v in b.items()}


def subset_query(q, ids):
    """Retain on-disk dtypes until native/cache overlap has been verified.

    The historical subset helper eagerly expands FP16 targets to FP32. That
    loses the dtype needed to verify native FP32 values against their original
    quantized cache. Model inputs are cast separately, after store validation.
    """
    count = len(q['valid'])
    indices = torch.as_tensor(ids, dtype=torch.long, device='cpu')
    if (indices.ndim != 1 or not len(indices) or (indices < 0).any()
            or (indices >= count).any() or len(indices.unique()) != len(indices)):
        raise ValueError('Unique in-range query indices required')
    result = {}
    for key, value in q.items():
        if torch.is_tensor(value) and value.ndim and len(value) == count:
            result[key] = value[indices.to(value.device)].detach().cpu().clone()
        elif isinstance(value, (list, tuple)) and len(value) == count:
            result[key] = [copy.deepcopy(value[int(i)]) for i in indices]
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_delta_policy(stores, smoke):
    if not smoke and any(s.index.get('config', {}).get('smoke') is not False for s in stores.values()):
        raise ValueError('Formal training cannot consume smoke-only or unclassified deltas')


def attach_encoded(b, ids, encoded):
    """Only declared audio/content-derived fields can enter full conditioning."""
    rows = [encoded[int(i)] for i in ids]
    frames = b['valid'].shape[1]
    if (not rows or len(rows) != len(b['valid'])
            or any(any(key not in row for key in ENCODED) for row in rows)):
        raise ValueError('One complete encoded record per selected native clip required')
    h0 = torch.zeros(len(ids), frames, rows[0]['h0'].shape[-1], dtype=rows[0]['h0'].dtype)
    for j, row in enumerate(rows):
        n = len(row['h0'])
        if n > frames or n != int(b['native_lengths'][j]):
            raise ValueError('Full encoded content differs from native clip length')
        h0[j, :n] = row['h0']
    return {**b, 'h0': h0, **{k: torch.stack([row[k] for row in rows]) for k in ENCODED if k != 'h0'}}


@torch.no_grad()
def cache_full(system, audio, store, target_scales, args):
    result = []
    for ids in torch.arange(len(store.clip_ids)).split(args.batch_size):
        b = cast_batch(store.batch(ids, mode='full'), args.device)
        fresh = encode_full_conditions(system, audio, b, target_scales)
        for j, n in enumerate(b['native_lengths']):
            result.append({k: (fresh[k][j, :int(n)] if k == 'h0' else fresh[k][j]).detach().cpu().clone()
                           for k in ENCODED})
    return result


def subset_bases(bases, ids):
    result = {}
    count = len(bases['valid'])
    for key, value in bases.items():
        if key == 'predictions':
            result[key] = {k: v[ids] for k, v in value.items()}
        elif torch.is_tensor(value) and value.ndim and len(value) == count:
            result[key] = value[ids]
        elif isinstance(value, list) and len(value) == count:
            result[key] = [value[int(i)] for i in ids]
        else:
            result[key] = value
    return result


def disk_guard(output, minimum_mib=512):
    if shutil.disk_usage(output).free < minimum_mib * 1024**2:
        raise RuntimeError(f'Need {minimum_mib} MiB free for atomic persistent checkpoints/results')


def verify_resume(payload, recipe_hash, arm, expected_epochs, *, frozen=None,
                  scales=None, protected_local_sha256=None, expected_updates=None):
    completed = payload.get('completed_epochs')
    rows, draws = payload.get('epochs'), payload.get('draws')
    if (payload.get('schema') != SCHEMA or payload.get('arm') != arm
            or payload.get('recipe_sha256') != recipe_hash
            or type(completed) is not int or not 1 <= completed <= expected_epochs
            or not isinstance(rows, list) or not isinstance(draws, list)
            or len(rows) != completed or len(draws) != completed):
        raise ValueError('Resume requires matching complete-epoch recipe/state')
    total = 0
    for epoch, (row, draw) in enumerate(zip(rows, draws), 1):
        updates = row.get('updates')
        if (row.get('arm') != arm or row.get('epoch') != epoch
                or type(updates) is not int or updates < 1
                or (expected_updates is not None and updates != expected_updates)
                or not isinstance(draw, str) or len(draw) != 64
                or any(c not in '0123456789abcdef' for c in draw)
                or row.get('draw_sha256') != draw):
            raise ValueError('Resume epoch/update/draw history is incomplete')
        total += updates
        if row.get('total_steps') != total:
            raise ValueError('Resume cumulative update history differs')
    if (payload.get('total_steps') != total or any(key not in payload for key in ('upper', 'local', 'optimizer', 'rng'))
            or (frozen is not None and payload.get('frozen') != frozen)
            or (scales is not None and (not torch.is_tensor(payload.get('scales'))
                or not torch.equal(payload['scales'].cpu(), scales.detach().cpu())))):
        raise ValueError('Resume optimizer/state/source coordinates differ')
    if protected_local_sha256 is not None:
        protected = {key:value for key,value in payload['local'].items()
                     if not key.startswith(('input.', 'blocks.', 'local_head.'))}
        if p.state_hash(protected) != protected_local_sha256:
            raise ValueError('Resume protected local heads differ from original source')


def completed_files(epochs):
    return {'final.pt', 'last.pt', 'matched.json', 'step0_holdout.json', 'step0_holdout_upper9.pt',
            'hold_evaluation.json', 'hold_upper9.pt', 'dev_evaluation.json', 'dev_upper9.pt',
            *(f'epoch{epoch:03d}.json' for epoch in range(1, epochs+1))}


def verify_completed(out, recipe_hash, arm, epochs, frozen, initial, protected_local_sha256, expected_updates):
    done = read(out/'complete.json')
    if (done.get('status') != 'complete' or done.get('schema') != SCHEMA
            or done.get('recipe_sha256') != recipe_hash or done.get('completed_epochs') != epochs
            or done.get('total_steps') != epochs*expected_updates or done.get('frozen') != frozen
            or set(done.get('files', {})) != completed_files(epochs)):
        raise ValueError('Completed arm recipe/output manifest differs')
    for name, digest in done['files'].items():
        if not (out/name).is_file() or sha(out/name) != digest:
            raise ValueError('Completed arm artifact changed: '+name)
    last = load_pt(out/'last.pt')
    verify_resume(last, recipe_hash, arm, epochs, frozen=frozen,
                  protected_local_sha256=protected_local_sha256, expected_updates=expected_updates)
    matched = read(out/'matched.json')
    if (last['completed_epochs'] != epochs
            or matched != {'initial':initial, 'draws':last['draws'], 'updates':last['total_steps']}):
        raise ValueError('Completed arm paired-state audit differs')
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','audio','targets','enrollment','native-root','native-manifest',
                 'trained-run','history-run','centered-run','context-run','split-report',
                 'delta-dir','initial-diagnostic','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=TRAIN_SEED)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=0, help='Smoke-only checkpoint-resume exercise')
    args = parser.parse_args()
    args.pilot = False; args.pilot_selection = None
    if (args.epochs, args.batch_size, args.seed) != (30, 16, TRAIN_SEED):
        raise ValueError('Fixed30epoch, batch16, seed97 recipe')
    if args.stop_after_epoch and (not args.smoke or args.stop_after_epoch != 1):
        raise ValueError('Only smoke may stop after its first complete epoch')
    if args.output.exists() != args.resume:
        raise ValueError('Use a fresh output or explicit resume of the existing output')
    disk_guard(args.output if args.output.exists() else args.output.parent, 1500)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    data, system, audio, original_local, identities, old_source, source_recipe, _, steps, _, frozen = p.load_context(args)
    if steps != 12 or [len(data['splits'][x]['valid']) for x in ('train','validation')] != [2315,405]:
        raise ValueError('Only locked native2315/405 source and12solver steps supported')
    sourcepath = args.context_run/'chunk_teacher/final.pt'
    source = load_pt(sourcepath); done = read(sourcepath.with_name('complete.json'))
    previous = read(sourcepath.with_name('provenance.json'))
    if (sha(sourcepath) != done['final_sha256'] or source['schema'] != context.SCHEMA
            or source['arm'] != 'chunk_teacher' or source['completed_epochs'] != 12
            or source['recipe_sha256'] != done['recipe_sha256'] or source['frozen'] != frozen
            or previous['recipe_sha256'] != source['recipe_sha256']
            or p.canonical_hash(previous['recipe']) != source['recipe_sha256']
            or previous['recipe']['source_recipe_sha256'] != p.canonical_hash(source_recipe)
            or not torch.equal(source['scales'], old_source['scales'])):
        raise ValueError('Context source lineage/coordinate mismatch')
    diag = read(args.initial_diagnostic/'complete.json')
    if (diag['status'] != 'complete' or diag['smoke'] or not diag['no_training']
            or diag['files']['fit128.json'] != sha(args.initial_diagnostic/'fit128.json')):
        raise ValueError('Complete fixed fit128 initialization diagnostic required')
    fit_ids, hold_ids, selection = partition(data['splits']['train'], read(args.split_report))
    ix = {'fit': fit_ids[:32] if args.smoke else fit_ids,
          'hold': hold_ids[:16] if args.smoke else hold_ids,
          'dev': torch.arange(32 if args.smoke else 405)}
    queries = {name: subset_query(data['splits']['validation' if name=='dev' else 'train'], ids)
               for name, ids in ix.items()}
    stores = {name: NativeContextStore(q, args.native_root, args.native_manifest, args.delta_dir,
                                      role='validation' if name=='dev' else 'train') for name,q in queries.items()}
    validate_delta_policy(stores, args.smoke)
    basepath = args.centered_run/'white/curves.pt'; basehash = sha(basepath)
    if basehash != source_recipe['baseline_curves_sha256']:
        raise ValueError('Bound seed-specific mouth/base changed')
    bases = subset_bases(load_pt(basepath), ix['dev'])
    metadata_equal({**queries['dev'],'target':queries['dev']['motion']}, bases)
    scales = source['scales'].to(args.device)
    epochs = 2 if args.smoke else 30
    code_root = Path(__file__).resolve().parents[1]
    common = {'schema':SCHEMA, 'epochs':epochs, 'requested_epochs':30, 'batch_size':16, 'seed':TRAIN_SEED,
              'source_sha256':sha(sourcepath), 'source_recipe_sha256':source['recipe_sha256'], 'frozen':frozen,
              'data_provenance':data['provenance'], 'baseline_sha256':basehash,
              'delta_manifest_sha256':sha(args.delta_dir/'manifest.json'),
              'initial_diagnostic_sha256':sha(args.initial_diagnostic/'fit128.json'),
              'native_manifest_sha256':sha(args.native_manifest), 'split':selection,
              'active_members':{k:v['clip_id'] for k,v in queries.items()},
              'config':data['config'], 'scales':scales.cpu().tolist(), 'decode_steps':12,
              'code_sha256':{f:sha(code_root/f) for f in CODE_FILES},
              'smoke':args.smoke, 'test_loaded':False, 'default_replaced':False,
              'local_policy':'Both arms train only input/blocks/local_head; frozen global/state and original base untouched',
              'objective':'Original unknown-only FM, strictly past GT training / generated past deployment',
              'comparison':'Equal updates, different supervised frame exposure and compute; offline full context/coverage combined',
              'evaluation':'Common historical center96,raw primary,DC separate secondary,no best-epoch/seed selection'}
    common_hash = p.canonical_hash(common)
    if args.resume:
        if read(args.output/'provenance.json') != {'recipe':common,'recipe_sha256':common_hash}:
            raise ValueError('Input/code/membership changed since original run')
    else:
        args.output.mkdir(parents=True)
        p.save_json(args.output/'provenance.json', {'recipe':common,'recipe_sha256':common_hash})
        p.save_json(args.output/'sentence_split.json', selection)
    p.save_json(args.output/'status.json', {'status':'encoding_full_context','smoke':args.smoke})
    encoded = {name:cache_full(system,audio,store,data['target_scales'],args) for name,store in stores.items()}
    coverage = {name:{'clips':len(s.clip_ids),'center_valid':int(s.q['valid'].sum()),
                     'full_valid':sum(int(s.clip(i)['valid'].sum()) for i in range(len(s.clip_ids))),
                     'native_max_length':int(s.native_lengths.max())} for name,s in stores.items()}
    p.save_json(args.output/'coverage.json', coverage)
    from scripts.evaluate_native_context import evaluate_native
    matched = {}
    source_protected = p.state_hash(a.protected_local_state(original_local))
    expected_updates = (len(stores['fit'].clip_ids)+15)//16
    for arm in ARMS:
        mode = 'center' if arm == 'center96' else 'full'
        out = args.output/arm
        torch.manual_seed(TRAIN_SEED); random.seed(TRAIN_SEED); np.random.seed(TRAIN_SEED)
        upper = p.PrefixUpperFlow(data['config']).to(args.device).eval()
        upper.load_state_dict(source['upper'], strict=True)
        local = a.configure_local(copy.deepcopy(original_local), True)
        initial = {'upper':p.state_hash(upper.state_dict()),'local':p.state_hash(local.state_dict())}
        recipe = {'common_sha256':common_hash,'arm':arm,'initial':initial}
        digest = p.canonical_hash(recipe)
        if (out/'complete.json').exists():
            matched[arm] = verify_completed(out,digest,arm,epochs,frozen,initial,source_protected,expected_updates)
            continue
        out.mkdir(exist_ok=True)
        if (out/'provenance.json').exists():
            if read(out/'provenance.json') != {'recipe':recipe,'recipe_sha256':digest}:
                raise ValueError('Arm provenance changed')
        else:
            p.save_json(out/'provenance.json', {'recipe':recipe,'recipe_sha256':digest})
        params = list(upper.parameters())+[v for v in local.parameters() if v.requires_grad]
        optimizer = torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-5)
        generator = torch.Generator().manual_seed(TRAIN_SEED)
        draws=[]; rows=[]; total_steps=0; start_epoch=0
        if (out/'last.pt').exists():
            last=load_pt(out/'last.pt')
            verify_resume(last,digest,arm,epochs,frozen=frozen,scales=scales,
                          protected_local_sha256=source_protected,expected_updates=expected_updates)
            upper.load_state_dict(last['upper']);local.load_state_dict(last['local'])
            optimizer.load_state_dict(last['optimizer']);restore_rng(last['rng'],generator)
            start_epoch=last['completed_epochs'];total_steps=last['total_steps'];draws=last['draws'];rows=last['epochs']
            del last
            for row in rows:
                # Atomic last.pt is the commit point. Repair a missing/truncated
                # lightweight epoch JSON after an interruption between writes.
                p.save_json(out/f"epoch{row['epoch']:03d}.json",row)
        else:
            # Input-context change can affect step0; report it rather than
            # pretending the full-native input is the center source forward.
            p.save_json(args.output/'status.json',{'status':'step0_holdout','arm':arm,'smoke':args.smoke})
            report, curves = evaluate_native(upper,local,system,stores['hold'],identities,scales,args,steps,
                                             mode=mode,encoded_full=encoded['hold'])
            p.save_json(out/'step0_holdout.json',report)
            p.save_checkpoint(out/'step0_holdout_upper9.pt',curves)
            del report,curves
        if p.state_hash(a.protected_local_state(local))!=source_protected:
            raise RuntimeError('Protected local heads differ from original source')
        for epoch in range(start_epoch,epochs):
            began=time.monotonic(); losses=[]; draw=hashlib.sha256(); supervised=0; grad_record={}
            for ids in torch.randperm(len(stores['fit'].clip_ids),generator=generator).split(16):
                b=stores['fit'].batch(ids,mode=mode)
                if mode=='full':b=attach_encoded(b,ids,encoded['fit'])
                b=cast_batch(b,args.device)
                noise,times,raw_draws=paired_noise(stores['fit'].native_lengths[ids],stores['fit'].center_starts[ids],generator,mode=mode)
                if mode=='full':noise=noise[:,:b['valid'].shape[1]]
                for value in (ids,raw_draws['full_noise'],times):draw.update(value.numpy().tobytes())
                ident=p.r.batch_identity(identities,b)
                native=a.local_features(local,b['audio_features'],b['valid'])
                loss=variable_context_loss(upper,b,ident,native,scales,noise,times)
                p.r.optimize(loss,optimizer,params)
                if not losses:
                    grad_record={n:sum(float(v.grad.abs().sum()) for v in getattr(local,n).parameters() if v.grad is not None)
                                 for n in ('input','blocks','local_head')}
                    if any(v<=0 for v in grad_record.values()):raise RuntimeError('Trainable local gradient missing')
                losses.append(float(loss.detach()));total_steps+=1;supervised+=int(b['valid'].sum())
            draws.append(draw.hexdigest())
            row={'arm':arm,'epoch':epoch+1,'updates':len(losses),'total_steps':total_steps,
                 'loss':sum(losses)/len(losses),'supervised_frames':supervised,'seconds':time.monotonic()-began,
                 'draw_sha256':draws[-1],'first_batch_local_gradients':grad_record}
            rows.append(row);disk_guard(args.output)
            payload={'schema':SCHEMA,'arm':arm,'upper':upper.state_dict(),'local':local.state_dict(),
                     'scales':scales.cpu(),'completed_epochs':epoch+1,'total_steps':total_steps,
                     'recipe_sha256':digest,'frozen':frozen,'draws':draws,'epochs':rows}
            p.save_checkpoint(out/'last.pt',{**payload,'optimizer':optimizer.state_dict(),'rng':p.capture_rng(generator)})
            p.save_json(out/f'epoch{epoch+1:03d}.json',row)
            p.save_json(args.output/'status.json',{'status':'training','smoke':args.smoke,**row})
            print(row,flush=True)
            if args.stop_after_epoch and epoch+1==args.stop_after_epoch:
                p.save_json(args.output/'status.json',{'status':'smoke_checkpoint_stop','arm':arm,'epoch':epoch+1})
                return
        if p.state_hash(a.protected_local_state(local))!=source_protected:
            raise RuntimeError('Protected local global/state heads changed')
        for name,module in (('system',system),('audio',audio),('local',original_local)):
            if p.state_hash(module.state_dict())!=frozen[name] or any(v.grad is not None for v in module.parameters()):
                raise RuntimeError('Frozen original changed: '+name)
        final={'schema':SCHEMA,'arm':arm,'upper':upper.state_dict(),'local':local.state_dict(),
               'scales':scales.cpu(),'completed_epochs':epochs,'total_steps':total_steps,'recipe_sha256':digest,'frozen':frozen}
        p.save_checkpoint(out/'final.pt',final)
        for name in ('hold','dev'):
            p.save_json(args.output/'status.json',{'status':'evaluating','arm':arm,'population':name,'epochs':epochs})
            report,curves=evaluate_native(upper,local,system,stores[name],identities,scales,args,steps,
                mode=mode,encoded_full=encoded[name],bases=bases if name=='dev' else None)
            report.update(recipe_sha256=digest,arm=arm,population=name)
            curves.update(recipe_sha256=digest,arm=arm,baseline_sha256=basehash if name=='dev' else None,
                          baseline_path=str(basepath) if name=='dev' else None)
            p.save_json(out/(name+'_evaluation.json'),report)
            p.save_checkpoint(out/(name+'_upper9.pt'),curves)
            del report,curves
        matched[arm]={'initial':initial,'draws':draws,'updates':total_steps}
        p.save_json(out/'matched.json',matched[arm])
        files={name:sha(out/name) for name in sorted(completed_files(epochs))}
        p.save_json(out/'complete.json',{'status':'complete','schema':SCHEMA,'recipe_sha256':digest,'files':files,
                    'completed_epochs':epochs,'total_steps':total_steps,'frozen':frozen,'test_loaded':False})
    if matched[ARMS[0]]!=matched[ARMS[1]]:
        raise RuntimeError('Paired initial weights/order/noise/time/update counts differ')
    p.save_json(args.output/'matched_audit.json',{'equal':True,'arms':matched})
    p.save_json(args.output/'status.json',{'status':'complete','smoke':args.smoke,'epochs_per_arm':epochs,
                'seconds_this_invocation':time.monotonic()-started,'test_loaded':False,'default_replaced':False})
    print('NATIVE_CONTEXT30_COMPLETE',flush=True)


if __name__=='__main__':main()
