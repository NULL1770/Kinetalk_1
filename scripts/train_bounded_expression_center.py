"""Nested speaker OOF of an independent bounded expression-center head.

Frozen upstream models have seen full TRAIN: this is module-level OOF only.
Every inner fold fits its own new statistics and selects epoch 0 (an actual
reference-generator bypass) or 1..12. Each outer speaker is evaluated once,
after selection and refit on the other speakers. Final epoch selection uses
a separate predeclared three-fold pass on all TRAIN, never the outer scores.
Development is opened only after the final checkpoint has been saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetalk_b0.models.bounded_expression_center import BoundedExpressionCenter, compose_bounded_center
from kinetalk_b0.models.reference_intensity_decoder import ReferenceIntensityDecoder, ReferenceIntensityStudent
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face
from scripts.reference_decoder_data import load_reference_context
from scripts.train_reference_intensity_decoder import SOURCE_SHA, prepare_cache, predict
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, canonical_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.train_full_staged import subset, NOT_UPPER
from scripts.train_isolated_audio_state import old_prediction

SCHEMA = 'bounded_expression_center_nested_oof_v1'
UPPER = list(UPPER_INDICES)
MODEL_CONFIG = {'global_dim': 64, 'identity_dim': 128, 'hidden': 64,
                'max_logit_delta': 6., 'anchor_eps': .01}


def long_ids(values):
    return torch.as_tensor(values, dtype=torch.long, device='cpu')


def speaker_folds(speakers, ids, groups=3):
    """Partition complete metadata by speaker before any smoke subsampling."""
    ids = long_ids(ids)
    people = sorted({speakers[int(i)] for i in ids})
    if groups < 2 or len(people) < groups:
        raise ValueError('Not enough speakers for nonempty speaker folds')
    answer = []
    for fold in range(groups):
        held = set(people[fold::groups])
        fit = long_ids([int(i) for i in ids if speakers[int(i)] not in held])
        val = long_ids([int(i) for i in ids if speakers[int(i)] in held])
        if not len(fit) or not len(val):
            raise ValueError('Empty speaker fold')
        answer.append((fit, val))
    return answer


def outer_folds(speakers):
    return speaker_folds(speakers, torch.arange(len(speakers)), len(set(speakers)))


def smoke_ids(speakers, ids, per_speaker=2):
    counts = Counter(); answer = []
    for i in long_ids(ids).tolist():
        s = speakers[i]
        if counts[s] < per_speaker:
            answer.append(i); counts[s] += 1
    return long_ids(answer)


def masked_mean(value, valid):
    return torch.where(valid[..., None], value, 0.).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


def compact_targets(q):
    valid = q['valid']
    if (not valid.any(1).all() or
            not (q['channel_mask'][:, UPPER] & q['anchor_valid'][:, UPPER]).all()):
        raise ValueError('Every query requires observed upper9 and an independent anchor')
    upper = q['motion'][..., UPPER]
    anchor = q['anchors'][:, UPPER]
    if not torch.isfinite(upper[valid]).all() or not torch.isfinite(anchor).all():
        raise ValueError('Observed target and independent anchors must be finite')
    if ((anchor < 0) | (anchor > 1)).any():
        raise ValueError('Independent anchor must be in [0,1]')
    relative = torch.where(valid[..., None], upper-anchor[:, None], 0.).double()
    return {'global': q['global_code'].detach().cpu(), 'identity': q['identity_code'].detach().cpu(),
            'anchor': anchor.detach().cpu(), 'target': masked_mean(upper, valid).detach().cpu(),
            'relative_sq': relative.square().sum(1).cpu(), 'frame_count': valid.sum(1).cpu(),
            'speaker': list(q['speaker']), 'clip_id': list(q['clip_id']),
            'sentence': list(q['sentence_id'])}


def fit_statistics(cache, ids):
    ids = long_ids(ids)
    if not len(ids):
        raise ValueError('Statistics need a nonempty fit fold')
    stats = {}
    for key in ('global', 'identity'):
        x = cache[key][ids].double()
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite fit conditions')
        stats[key+'_mean'] = x.mean(0).float()
        stats[key+'_std'] = x.std(0, unbiased=False).clamp_min(.05).float()
    stats['scales9'] = (cache['relative_sq'][ids].sum(0) /
                        cache['frame_count'][ids].sum()).sqrt().clamp_min(.02).float()
    stats['fit_ids'] = ids.clone()
    stats['fit_clip_ids_sha256'] = canonical_hash([cache['clip_id'][int(i)] for i in ids])
    stats['fit_speakers'] = sorted({cache['speaker'][int(i)] for i in ids})
    return stats


def model_inputs(cache, ids, stats, device):
    """Deployment input whitelist: no query motion/label/mean/baseline center."""
    ids = long_ids(ids)
    return tuple(value.to(device) for value in (
        (cache['global'][ids]-stats['global_mean'])/stats['global_std'],
        (cache['identity'][ids]-stats['identity_mean'])/stats['identity_std'],
        cache['anchor'][ids]))


def make_model(seed, device, config=None):
    torch.manual_seed(seed)
    return BoundedExpressionCenter(**(MODEL_CONFIG if config is None else config)).to(device)


def speaker_weights(speakers, ids):
    selected = [speakers[int(i)] for i in ids]
    count = Counter(selected)
    return torch.tensor([len(ids)/(len(count)*count[s]) for s in selected], dtype=torch.float32)


def train_epoch(model, cache, ids, stats, optimizer, device, batch_size, seed):
    ids = long_ids(ids); model.train()
    weights = speaker_weights(cache['speaker'], ids)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
    scale = stats['scales9'].to(device)
    total = 0.
    for local in order.split(batch_size):
        ix = ids[local]
        pred = model(*model_inputs(cache, ix, stats, device))
        error = ((pred-cache['target'][ix].to(device))/scale).square().mean(-1)
        # Fixed batch denominator: the last short batch does not gain extra
        # weight in this speaker-balanced empirical objective.
        loss = (error*weights[local].to(device)).sum()/batch_size
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite center loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); total += float((error.detach().cpu()*weights[local]).sum())
    return total/len(ids)


@torch.no_grad()
def predict_centers(model, cache, ids, stats, device, batch_size=64):
    model.eval()
    return torch.cat([model(*model_inputs(cache, ix, stats, device)).cpu()
                      for ix in long_ids(ids).split(batch_size)])


def per_speaker_center_error(center, cache, ids, scales):
    values = ((center.double()-cache['target'][ids].double())/scales.double()).square().mean(-1)
    people = [cache['speaker'][int(i)] for i in ids]
    return {s: float(values[long_ids([i for i,x in enumerate(people) if x == s])].mean())
            for s in sorted(set(people))}


def select_epoch(scores):
    """Epoch zero is the real bypass; ties select it or the earlier epoch."""
    if not scores or any(not np.isfinite(x) for x in scores):
        raise ValueError('Finite baseline and candidate scores are required')
    return min(range(len(scores)), key=lambda i: (scores[i], i))


def inner_selection(cache, base_center, ids, args, output, seed):
    output.mkdir(parents=True, exist_ok=True)
    combined = [dict() for _ in range(args.epochs+1)]
    folds = []
    for k, (fit_full, val_full) in enumerate(speaker_folds(cache['speaker'], ids, 3)):
        fit = smoke_ids(cache['speaker'], fit_full) if args.smoke else fit_full
        val = smoke_ids(cache['speaker'], val_full) if args.smoke else val_full
        stats = fit_statistics(cache, fit)
        model = make_model(seed+k, args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
        values = per_speaker_center_error(base_center[val], cache, val, stats['scales9'])
        combined[0].update(values)
        history = [{'epoch': 0, 'source': 'external_reference_bypass', 'per_speaker': values}]
        for epoch in range(1, args.epochs+1):
            loss = train_epoch(model, cache, fit, stats, optimizer, args.device,
                               args.batch_size, seed+1000*k+epoch)
            predicted = predict_centers(model, cache, val, stats, args.device, args.batch_size)
            values = per_speaker_center_error(predicted, cache, val, stats['scales9'])
            combined[epoch].update(values)
            history.append({'epoch': epoch, 'train_loss': loss, 'per_speaker': values})
        record = {'fold': k, 'fit_ids': fit.tolist(), 'validation_ids': val.tolist(),
                  'full_metadata_fit_ids': fit_full.tolist(), 'full_metadata_validation_ids': val_full.tolist(),
                  'fit_speakers': stats['fit_speakers'],
                  'validation_speakers': sorted({cache['speaker'][int(i)] for i in val}),
                  'stats_file': f'inner{k}_stats.pt', 'model_seed': seed+k,
                  'history': history}
        save_checkpoint(output/f'inner{k}_stats.pt', stats)
        save_json(output/f'inner{k}_history.json', record); folds.append(record)
    scores = [sum(row.values())/len(row) for row in combined]
    selected = select_epoch(scores)
    result = {'selected_epoch': selected, 'speaker_macro_normalized_center_mse': scores,
              'per_epoch_per_speaker': combined, 'inner_folds': folds,
              'selection_uses_outer_targets': False, 'bypass_candidate': 0}
    save_json(output/'selection.json', result)
    return selected, result


def refit(cache, ids, epoch_count, args, output, seed):
    stats = fit_statistics(cache, ids)
    model = None
    history = []
    if epoch_count:
        model = make_model(seed, args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
        for epoch in range(1, epoch_count+1):
            loss = train_epoch(model, cache, ids, stats, optimizer, args.device,
                               args.batch_size, seed+epoch)
            history.append({'epoch': epoch, 'loss': loss})
    checkpoint = {'schema': SCHEMA, 'selected_epoch': epoch_count,
                  'bypass': epoch_count == 0, 'model': None if model is None else model.state_dict(),
                  'model_config': MODEL_CONFIG, 'stats': stats, 'seed': seed,
                  'training_history': history, 'test_loaded': False}
    save_checkpoint(output, checkpoint)
    return model, stats


def bit_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def trajectory_rows(upper, baseline, target, valid, scales, clip_ids, speakers, *, center=None, alpha=None):
    """Error decomposition uses one shared valid mask and each fit's scales."""
    if not valid.any(1).all() or not torch.isfinite(upper[valid]).all():
        raise ValueError('Metrics require finite, nonempty observed output')
    x = upper.double(); y = target.double(); v = valid[..., None]
    difference = torch.where(v, x-y, 0.)
    mean = masked_mean(difference, valid)
    dynamic = torch.where(v, difference-mean[:, None], 0.)
    count = valid.sum(1)*upper.shape[-1]
    rows = []
    for raw, prefix in ((torch.ones_like(scales).double(), 'raw'), (scales.double(), 'normalized')):
        total = (difference/raw).square().sum((1,2))/count
        center_mse = (mean/raw).square().mean(-1)
        centered = (dynamic/raw).square().sum((1,2))/count
        if not torch.allclose(total, center_mse+centered, atol=1e-10, rtol=1e-8):
            raise RuntimeError('Mean + centered error decomposition failed')
        if not rows:
            rows = [{'clip_id': c, 'speaker': s} for c,s in zip(clip_ids,speakers)]
        for i, row in enumerate(rows):
            row.update({prefix+'_total_mse': float(total[i]), prefix+'_mean_mse': float(center_mse[i]),
                        prefix+'_centered_mse': float(centered[i])})
    means = masked_mean(x, valid)
    for i,row in enumerate(rows):
        observed = x[i,valid[i]]
        row.update(upper_min=float(observed.min()), upper_max=float(observed.max()),
                   outside_fraction=float(((observed < 0)|(observed > 1)).double().mean()),
                   invalid_upper_bit_exact=bit_equal(upper[i,~valid[i]],baseline[i,~valid[i]]))
        if center is not None:
            row['requested_center_max_error'] = float((means[i]-center[i].double()).abs().max())
        if alpha is not None:
            row['carrier_scale'] = alpha[i].tolist()
            row['carrier_scaled_channel_fraction'] = float((alpha[i]<1-1e-6).float().mean())
    return rows


def aggregate_rows(rows):
    metrics = [k for k in rows[0] if k.endswith('_mse')]
    people = sorted({row['speaker'] for row in rows})
    per_speaker = {s: {k: float(np.mean([r[k] for r in rows if r['speaker']==s])) for k in metrics}
                   for s in people}
    return {'clip_count': len(rows), 'speaker_count': len(people),
            'clip_average': {k: float(np.mean([r[k] for r in rows])) for k in metrics},
            'speaker_macro': {k: float(np.mean([per_speaker[s][k] for s in people])) for k in metrics},
            'per_speaker': per_speaker,
            'outside_fraction': float(np.mean([r['outside_fraction'] for r in rows])),
            'nonupper43_bit_exact': all(r.get('nonupper43_bit_exact',True) for r in rows),
            'invalid_frames_bit_exact': all(r['invalid_upper_bit_exact'] for r in rows)}


def paired_comparison(baseline, selected, seed=53):
    """Speaker-level paired uncertainty; frames are never independent units."""
    if [r['clip_id'] for r in baseline] != [r['clip_id'] for r in selected]:
        raise ValueError('Paired scores must have identical clip order')
    people=sorted({r['speaker'] for r in baseline})
    result={}
    rng=np.random.default_rng(seed)
    resamples=rng.integers(0,len(people),(2000,len(people)))
    for key in ('raw_mean_mse','raw_total_mse','raw_centered_mse'):
        delta={s:float(np.mean([b[key]-a[key] for a,b in zip(baseline,selected)
                               if a['speaker']==s])) for s in people}
        values=np.asarray(list(delta.values()))
        bootstrap=values[resamples].mean(-1)
        result[key]={'difference_selected_minus_baseline':float(values.mean()),
                     'speaker_differences':delta,'improved_speakers':int((values<0).sum()),
                     'worsened_speakers':int((values>0).sum()),
                     'largest_speaker_degradation':float(values.max()),
                     'speaker_bootstrap_95ci':np.quantile(bootstrap,[.025,.975]).tolist()}
    return {'speaker_count':len(people),'bootstrap_seed':seed,'bootstrap_draws':2000,
            'scope':'paired speaker means; conditions on frozen upstream and fixed training protocol',
            'small_speaker_count_warning':len(people)<10,'metrics':result}


@torch.no_grad()
def evaluate_head(q, cache, ids, base_upper, full_base, model, stats, args):
    ids = long_ids(ids); base = base_upper[ids]; valid = q['valid'][ids]
    target = q['motion'][ids][..., UPPER]
    centers = masked_mean(base, valid) if model is None else predict_centers(
        model, cache, ids, stats, args.device, args.batch_size)
    if model is None:
        generated = base.clone(); alpha = torch.ones(len(ids),9)
    else:
        composed = compose_bounded_center(base, centers, valid)
        generated, alpha = composed['upper'], composed['scale']
    names = [q['clip_id'][int(i)] for i in ids]
    speakers = [q['speaker'][int(i)] for i in ids]
    report = {}
    # Isolate any finite-precision effect of the composition operation itself.
    own_center=masked_mean(base,valid)
    recomposed=compose_bounded_center(base,own_center,valid)
    for name, value in (('baseline',base),('composition_only',recomposed['upper']),('selected',generated)):
        rows = trajectory_rows(value,base,target,valid,stats['scales9'],names,speakers,
                               center=centers if name=='selected' else None,
                               alpha=alpha if name=='selected' else None)
        full = compose_upper_face(full_base[ids], value, valid)
        for j,row in enumerate(rows):
            row['nonupper43_bit_exact'] = bit_equal(full[j,...,NOT_UPPER],full_base[ids[j],...,NOT_UPPER])
            if 'emotion_id' in q: row['emotion_id'] = int(q['emotion_id'][ids[j]])
        report[name] = {'summary': aggregate_rows(rows), 'per_clip': rows}
    if model is None and not bit_equal(generated,base):
        raise RuntimeError('Epoch zero must be an exact external baseline bypass')
    report['paired']=paired_comparison(report['baseline']['per_clip'],report['selected']['per_clip'],args.seed if hasattr(args,'seed') else 53)
    return report, {'upper':generated, 'center':centers, 'scale':alpha,
                    'baseline_upper':base,'target_upper':target,'valid':valid,
                    'ids':ids, 'clip_id':names, 'bypass':model is None}


@torch.no_grad()
def frozen_baselines(q, system, identities, reference, args):
    stats = reference['stats']
    decoder = ReferenceIntensityDecoder(**reference['decoder_config']).to(args.device).eval()
    student = ReferenceIntensityStudent(**reference['student_config']).to(args.device).eval()
    decoder.load_state_dict(reference['decoder']); student.load_state_dict(reference['student'])
    decoder.requires_grad_(False); student.requires_grad_(False)
    cache = prepare_cache(q, stats, args.batch_size)
    ids = torch.arange(len(q['valid']))
    upper, _ = predict(decoder, student, cache, ids, args.device, args.batch_size, 'full')
    del cache, decoder, student
    prior = []
    rng = torch.Generator().manual_seed(48)
    for ix in ids.split(args.batch_size):
        b = subset(q,ix,args.device)
        noise = torch.randn((*b['valid'].shape,52),generator=rng).to(args.device)
        prior.append(old_prediction(system,b,identities,noise).cpu())
    base = compose_upper_face(torch.cat(prior), upper, q['valid'])
    return upper, base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data','source','reference','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=64); p.add_argument('--seed',type=int,default=53)
    p.add_argument('--smoke',action='store_true')
    args = p.parse_args()
    if args.epochs != 12 and not args.smoke: p.error('Formal budget is fixed at 12 epochs')
    if args.epochs<1 or args.batch_size<1: p.error('Positive epochs and batch size required')
    if args.smoke: args.epochs=min(args.epochs,2)
    if args.output.exists(): raise FileExistsError('A fresh output directory is required')
    args.output.mkdir(parents=True); started=time.monotonic()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    ref_path = args.reference/'final.pt' if args.reference.is_dir() else args.reference
    reference = torch.load(ref_path,map_location='cpu',weights_only=False)
    if (reference.get('schema') != 'reference_intensity_decoder_v1' or
            reference.get('protocol',{}).get('source_sha256') != SOURCE_SHA):
        raise ValueError('Reference checkpoint provenance differs')
    save_json(args.output/'status.json',{'status':'loading_train','test_loaded':False})
    data,system,audio,identities = load_reference_context(args.data,args.source,args.device,
        role='train',seed=args.seed,expected_source_sha256=SOURCE_SHA,batch_size=args.batch_size)
    q=data['splits']['train']
    if reference['protocol'].get('data',{}).get('manifest_sha256') != data['provenance']['manifest_sha256']:
        raise ValueError('Reference and new center training require the same data manifest')
    complete_folds=outer_folds(q['speaker'])
    if not args.smoke and len(complete_folds)!=22: raise ValueError('Formal protocol expects 22 TRAIN speakers')
    if args.smoke:
        # Complete metadata determines split first; sample two clips per person
        # only afterwards. All speakers remain represented in inner folds.
        chosen=smoke_ids(q['speaker'],torch.arange(len(q['valid'])))
        q=subset(q,chosen,'cpu'); complete_folds=outer_folds(q['speaker'])[:1]
    cache=compact_targets(q)
    base_upper,full_base=frozen_baselines(q,system,identities,reference,args)
    base_center=masked_mean(base_upper,q['valid'])
    sources={str(path.relative_to(ROOT)):sha(path) for path in (
        Path(__file__),ROOT/'kinetalk_b0/models/bounded_expression_center.py',
        ROOT/'scripts/reference_decoder_data.py',ROOT/'scripts/train_reference_intensity_decoder.py')}
    protocol={'schema':SCHEMA,'smoke':args.smoke,'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,
        'model_config':MODEL_CONFIG,'optimizer':{'name':'AdamW','lr':.003,'weight_decay':.01,'grad_clip':1.},
        'source_sha256':SOURCE_SHA,'reference_sha256':sha(ref_path),'source_files':sources,
        'data':data['provenance'],'train_clip_ids':q['clip_id'],'train_speakers':q['speaker'],
        'scope':'new center head nested speaker OOF; frozen Stage4 and reference saw all TRAIN',
        'outer':'leave one speaker out, inner 3 sorted round-robin speaker folds',
        'inner_selection':'speaker-macro center normalized MSE, epoch0 actual external bypass, ties earlier',
        'final_selection':'separate 3-fold speaker selection on all TRAIN, then full TRAIN refit',
        'normalization':'global/identity standardization and upper RMS scales fitted only on current fit identities',
        'inputs':'global64, identity128, raw independent anchor9; no c0, local audio, query target or labels',
        'composition':'static requested center plus channel-constant bounded shrinkage of deterministic reference carrier',
        'dynamic_timing_improvement_claim':False,'empirical_combination_included':False,
        'upstream_seen_full_train':True,'sentence_disjoint_oof':False,'test_loaded':False,
        'development_previously_used':True,'default_replaced':False}
    protocol['runtime']={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                         'device':args.device,'cuda':torch.version.cuda}
    save_json(args.output/'protocol.json',protocol)
    del data,system,audio,identities
    oof={'baseline':[],'composition_only':[],'selected':[]}; fold_records=[]; seen=[]
    for k,(fit,held) in enumerate(complete_folds):
        out=args.output/f'outer_{k:02d}';out.mkdir()
        selected,selection=inner_selection(cache,base_center,fit,args,out,args.seed+10000*k)
        model,stats=refit(cache,fit,selected,args,out/'refit.pt',args.seed+10000*k+9000)
        report,curves=evaluate_head(q,cache,held,base_upper,full_base,model,stats,args)
        save_json(out/'evaluation.json',report);save_checkpoint(out/'curves.pt',curves)
        record={'fold':k,'fit_ids':fit.tolist(),'held_ids':held.tolist(),
                'held_speakers':sorted({q['speaker'][int(i)] for i in held}),'selected_epoch':selected,
                'selection_file':str((out/'selection.json').relative_to(args.output)),
                'refit_sha256':sha(out/'refit.pt')}
        fold_records.append(record);seen.extend(held.tolist())
        for name in oof:oof[name].extend(report[name]['per_clip'])
        save_json(args.output/'status.json',{'status':'outer_oof','completed_folds':k+1,
            'total_folds':len(complete_folds),'elapsed_seconds':time.monotonic()-started,'test_loaded':False})
        print(json.dumps({'outer_fold':k,'selected_epoch':selected,'held_speakers':record['held_speakers']}),flush=True)
    if not args.smoke and sorted(seen)!=list(range(len(q['valid']))):
        raise RuntimeError('Every TRAIN query must be outer-evaluated exactly once')
    oof_report={'scope':protocol['scope'],'smoke':args.smoke,'folds':fold_records,
                'cross_fold_primary_unit':'raw coefficient MSE; normalized scores use fold-specific fit scales',
                'modes':{name:{'summary':aggregate_rows(rows),'per_clip':rows} for name,rows in oof.items()},
                'test_loaded':False,'default_replaced':False}
    oof_report['paired']=paired_comparison(oof['baseline'],oof['selected'],args.seed)
    save_json(args.output/'oof_evaluation.json',oof_report)
    if args.smoke:
        save_json(args.output/'status.json',{'status':'complete','scope':'TRAIN smoke only; no formal inference',
                  'development_loaded':False,'test_loaded':False,'elapsed_seconds':time.monotonic()-started})
        return
    all_ids=torch.arange(len(q['valid']))
    final_dir=args.output/'final_selection'
    selected,_=inner_selection(cache,base_center,all_ids,args,final_dir,args.seed+500000)
    model,stats=refit(cache,all_ids,selected,args,args.output/'final.pt',args.seed+600000)
    final=torch.load(args.output/'final.pt',map_location='cpu',weights_only=False)
    final.update(protocol=protocol,protocol_sha256=canonical_hash(protocol))
    save_checkpoint(args.output/'final.pt',final)
    del q,cache,base_upper,full_base,base_center
    if torch.cuda.is_available():torch.cuda.empty_cache()
    save_json(args.output/'status.json',{'status':'evaluating_development','selected_epoch':selected,'test_loaded':False})
    data,system,audio,identities=load_reference_context(args.data,args.source,args.device,
        role='validation',seed=args.seed,expected_source_sha256=SOURCE_SHA,batch_size=args.batch_size)
    q=data['splits']['validation'];cache=compact_targets(q)
    base_upper,full_base=frozen_baselines(q,system,identities,reference,args)
    report,curves=evaluate_head(q,cache,torch.arange(len(q['valid'])),base_upper,full_base,model,stats,args)
    report.update(scope='development once after nested OOF and separately selected full TRAIN refit',
                  selected_epoch=selected,test_loaded=False,development_provenance=data['provenance'],
                  upstream_seen_full_train=True,default_replaced=False,medtalk_parity_established=False)
    save_json(args.output/'evaluation.json',report)
    save_checkpoint(args.output/'development_curves.pt',curves)
    save_json(args.output/'status.json',{'status':'complete','selected_epoch':selected,
              'development_loaded':True,'test_loaded':False,'elapsed_seconds':time.monotonic()-started})
    print('BOUNDED_CENTER_NESTED_OOF_COMPLETE',flush=True)


if __name__=='__main__':main()
