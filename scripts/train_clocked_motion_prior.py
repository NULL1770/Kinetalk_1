"""Paired fixed-clock acoustic versus static-acoustic motion-shape development."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_motion_process as old
from scripts import train_joint_motion_prior as previous
from scripts import joint_motion_metrics as metrics
from scripts import clocked_motion_dictionary as shapes
from scripts.joint_prior_audio_context import load_frozen_audio, assert_source_binding, encode_clips
from kinetalk_b0.models.clocked_motion_prior import ClockedMotionPrior, categorical_energy_score

SCHEMA = 'clocked_motion_shape_development_v1'
HORIZON, HOP, SEED = 32, 16, 20260918


def static_acoustics(features, valid):
    """Whole-clip audio mean on its original clock; no target or alignment input."""
    clean = torch.where(valid[:, None], features, 0.)
    mean = clean.sum(0) / valid.sum().clamp_min(1)
    return torch.where(valid[:, None], mean[None], 0.)


def acoustic_windows(features, valid):
    """Full H32 windows, end-anchored last window; mask only runs shorter than H."""
    arrays, masks, locations = [], [], []
    for left, right in old.runs(valid.cpu().numpy()):
        starts = list(range(left, max(left+1, right-HORIZON+1), HOP))
        if right-left>=HORIZON and starts[-1]!=right-HORIZON:
            starts.append(right-HORIZON)
        for start in starts:
            count = min(HORIZON, right - start)
            x = features.new_zeros(HORIZON, features.shape[-1])
            mask = torch.zeros(HORIZON, dtype=torch.bool)
            x[:count] = features[start:start+count]
            mask[:count] = True
            arrays.append(x); masks.append(mask); locations.append((start, count))
    if not arrays:
        raise ValueError('No observed acoustic frames')
    return torch.stack(arrays), torch.stack(masks), locations


def fit_equilibrium(clips, ids):
    x = np.stack([np.r_[clips[i]['global'], clips[i]['anchor_upper']] for i in ids])
    y = np.stack([clips[i]['upper'][clips[i]['valid'].numpy()].mean(0)
                  - clips[i]['anchor_upper'] for i in ids])
    mean, std = x.mean(0), np.maximum(x.std(0), .01)
    z = np.column_stack([np.ones(len(x)), (x-mean)/std])
    penalty = np.eye(z.shape[1])*10.; penalty[0, 0] = 0.
    coef = np.linalg.solve(z.T@z + penalty, z.T@y)
    return {'mean': mean, 'std': std, 'coef': coef, 'alpha': 10.,
            'fit_clip_ids': [clips[i]['clip_id'] for i in ids]}


def equilibrium(global_vec, anchor, fitted):
    z = np.r_[1., (np.r_[global_vec, anchor]-fitted['mean'])/fitted['std']]
    return np.asarray(anchor) + z @ fitted['coef']


def build_training(clips, windows, dictionary):
    by_id = {c['clip_id']: c for c in clips}
    counts = Counter(w['clip_id'] for w in windows)
    features, static, global_, weights = [], [], [], []
    cached_static = {}
    for w in windows:
        c = by_id[w['clip_id']]; start = w['start']
        if c['clip_id'] not in cached_static:
            cached_static[c['clip_id']] = static_acoustics(c['features'], c['valid'])
        features.append(c['features'][start:start+HORIZON])
        static.append(cached_static[c['clip_id']][start:start+HORIZON])
        global_.append(c['global']); weights.append(1./counts[c['clip_id']])
    distances = shapes.target_distances(np.stack([w['future'] for w in windows]),
                                       dictionary['shapes'], dictionary['scales'])
    weights = np.asarray(weights, np.float32); weights /= weights.mean()
    return {'temporal': torch.stack(features), 'static': torch.stack(static),
            'global': torch.tensor(np.stack(global_), dtype=torch.float32),
            'distance': torch.tensor(distances, dtype=torch.float32),
            'weight': torch.tensor(weights),
            'valid': torch.ones(len(windows), HORIZON, dtype=torch.bool)}


def train_pair(data, stats, dictionary, output, device, epochs):
    torch.manual_seed(SEED)
    model = ClockedMotionPrior(*stats, horizon=HORIZON, hidden=64, k=len(dictionary['shapes']))
    initial = copy.deepcopy(model.state_dict())
    distance = torch.tensor(shapes.pairwise_distances(dictionary['shapes'], dictionary['scales']),
                            dtype=torch.float32, device=device)
    tensors = {k: v.to(device) for k, v in data.items()}
    models, losses = {}, {}
    for arm in ('static', 'temporal'):
        model = ClockedMotionPrior(*stats, horizon=HORIZON, hidden=64, k=len(dictionary['shapes'])).to(device)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
        rng = np.random.default_rng(SEED); rows = []; order_hash = hashlib.sha256()
        for epoch in range(epochs):
            model.train(); order = rng.permutation(len(tensors['valid'])); order_hash.update(order.tobytes())
            total, count = 0., 0
            for start in range(0, len(order), 128):
                ix = torch.as_tensor(order[start:start+128], device=device)
                logits = model(tensors[arm][ix], tensors['valid'][ix], tensors['global'][ix])
                score = categorical_energy_score(logits.softmax(-1), tensors['distance'][ix], distance)
                loss = (score*tensors['weight'][ix]).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                total += float(loss.detach())*len(ix); count += len(ix)
            rows.append({'epoch': epoch+1, 'energy_score': total/count,
                         'updates': (len(order)+127)//128})
            old.save_json(output/(arm+'_losses.json'), rows)
            old.save_json(output/'status.json', {'schema': SCHEMA, 'status': 'training',
                'arm': arm, 'epoch': epoch+1, 'epochs': epochs})
            print('EPOCH', arm, epoch+1, total/count, flush=True)
        model.eval(); models[arm] = model
        torch.save({'schema': SCHEMA, 'state': model.state_dict(), 'epochs': epochs,
                    'order_sha256': order_hash.hexdigest(), 'optimizer': optimizer.state_dict()}, output/(arm+'_final.pt'))
        losses[arm] = rows
    del tensors
    return models, losses


@torch.no_grad()
def predict_probabilities(model, features, valid, global_vec, device):
    x, mask, locations = acoustic_windows(features, valid)
    g = torch.as_tensor(global_vec, dtype=torch.float32, device=device)[None].expand(len(x), -1)
    result = []
    for start in range(0, len(x), 128):
        result.append(model(x[start:start+128].to(device), mask[start:start+128].to(device),
                            g[start:start+128]).softmax(-1).cpu().numpy())
    return np.concatenate(result), locations


def sample_shapes(probabilities, locations, dictionary, level, valid, clip_id, seeds=old.SEEDS):
    """Only learned support/probabilities, independent level and audio clock enter."""
    values = []; chosen = []
    for seed in seeds:
        key = int.from_bytes(hashlib.sha256(f'clocked:{seed}:{clip_id}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(key)
        tokens = np.minimum((np.cumsum(probabilities, axis=1) < rng.random((len(probabilities), 1))).sum(1),
                            probabilities.shape[1]-1)
        draw = np.zeros((len(valid), 9), dtype=np.float64)
        for left, right in old.runs(valid):
            ix = [j for j, (start, _) in enumerate(locations) if left <= start < right]
            chunks = dictionary['shapes'][tokens[ix]]
            starts = [locations[j][0]-left for j in ix]
            wave, coverage = shapes.overlap_add(chunks, starts, right-left, horizon=HORIZON)
            if not coverage.all():
                raise ValueError('Fixed-clock overlap has an uncovered observed frame')
            draw[left:right] = wave + level
        values.append(draw); chosen.append(tokens)
    return np.asarray(values), np.asarray(chosen)


def acceleration_stats(values, target, valid, scales):
    squares, reference = [], []
    for left, right in old.runs(valid):
        if right-left >= 3:
            squares.extend(np.square(np.diff(values[:, left:right]/scales, n=2, axis=1)).reshape(-1))
            reference.extend(np.square(np.diff(target[left:right]/scales, n=2, axis=0)).reshape(-1))
    return {'sum_squares': float(np.sum(squares)), 'count': len(squares),
            'reference_sum_squares': float(np.sum(reference)), 'reference_count': len(reference)}


def report_rows(rows):
    if not rows:
        return {'summary': None, 'acceleration': None, 'rows': [], 'by_speaker': {},
                'by_emotion': {}, 'low_activity': None}
    summary = metrics.summarize(rows)
    acc = {k: sum(r['acceleration'][k] for r in rows) for k in rows[0]['acceleration']}
    acc['rms_ratio'] = float(np.sqrt((acc['sum_squares']/max(acc['count'], 1))/
                                    max(acc['reference_sum_squares']/max(acc['reference_count'], 1), 1e-15)))
    return {'summary': summary, 'acceleration': acc, 'rows': rows,
            'by_speaker': {str(s): metrics.summarize([r for r in rows if r['speaker']==s])
                           for s in sorted({r['speaker'] for r in rows})},
            'by_emotion': {str(s): metrics.summarize([r for r in rows if r['emotion']==s])
                           for s in sorted({r['emotion'] for r in rows})},
            'low_activity': metrics.summarize([r for r in rows if r['low_activity']])
                            if any(r['low_activity'] for r in rows) else None}


def intervention_audio(c, intervention, donor=None):
    x = c['features'].clone(); valid = c['valid']
    if intervention == 'static':
        x = static_acoustics(x, valid)
    elif intervention == 'reverse':
        for left, right in old.runs(valid.numpy()):
            x[left:right] = x[left:right].flip(0)
    elif intervention == 'mismatch':
        source = donor['features'][donor['valid']]
        for left, right in old.runs(valid.numpy()):
            x[left:right] = torch.nn.functional.interpolate(source.T[None], size=right-left,
                                mode='linear', align_corners=True)[0].T
    elif intervention != 'real':
        raise ValueError('Unknown acoustic intervention')
    return x


def evaluate(clips, ids, dictionary, fitted, scale, low_cut, model, device, *, intervention='real', uniform=False):
    rows, curves, donors = [], {}, {}
    if intervention == 'mismatch':
        for i in ids:
            options = [j for j in ids if clips[j]['speaker']==clips[i]['speaker']
                       and clips[j]['emotion']==clips[i]['emotion'] and clips[j]['sentence']!=clips[i]['sentence']
                       and len(old.runs(clips[j]['valid'].numpy()))==1]
            if options:
                donors[i] = min(options, key=lambda j: clips[j]['clip_id'])
        ids = [i for i in ids if i in donors]
    for i in ids:
        c = clips[i]; valid = c['valid'].numpy()
        x = intervention_audio(c, intervention, clips[donors[i]] if i in donors else None)
        prob, locations = predict_probabilities(model, x, c['valid'], c['global'], device)
        if uniform:
            prob[:] = 1./len(dictionary['shapes'])
        level = equilibrium(c['global'], c['anchor_upper'], fitted)
        values, tokens = sample_shapes(prob, locations, dictionary, level, valid, c['clip_id'])
        score = metrics.score_clip(values, c['upper'], valid, scale)
        target_energy = float(np.mean((old.center_runs(c['upper'], valid)[valid]/scale)**2))
        rows.append({'clip_id': c['clip_id'], 'sentence': c['sentence'], 'speaker': c['speaker'],
                     'emotion': c['emotion'], 'low_activity': target_energy<=low_cut,
                     'acceleration': acceleration_stats(values, c['upper'], valid, scale), **score})
        curves[c['clip_id']] = {'samples': values.astype(np.float32), 'target': c['upper'], 'valid': valid,
                               'probabilities': prob, 'tokens': tokens, 'locations': locations, 'level': level}
    report = report_rows(rows)
    report.update(intervention=intervention, uniform=uniform,
        donor_mapping={clips[i]['clip_id']: clips[j]['clip_id'] for i, j in donors.items()})
    return report, curves


def assessment(reports):
    required = ('temporal', 'static_trained', 'static', 'reverse', 'mismatch')
    if any(k not in reports or not reports[k].get('rows') for k in required):
        return {'timing_passed': False, 'quality_passed': False,
                'reason': 'missing_intervention_support', 'development_only': True,
                'generator_integrated': False, 'default_replaced': False}
    support = set.intersection(*[set(r['clip_id'] for r in reports[k]['rows']) for k in required])
    coverage = len(support)/len(reports['temporal']['rows'])
    sentences = {r['sentence'] for r in reports['temporal']['rows'] if r['clip_id'] in support}
    if len(support)<2 or coverage<.70 or len(sentences)<2:
        return {'timing_passed': False, 'quality_passed': False,
                'reason': 'insufficient_common_intervention_support', 'support': len(support),
                'support_fraction': coverage, 'support_sentences': len(sentences),
                'development_only': True, 'generator_integrated': False, 'default_replaced': False}
    paired = {k: previous.paired_subset(reports[k], support) for k in required}
    gains = {k: previous.bootstrap_gain(paired['temporal'], paired[k]) for k in required[1:]}
    real, base = paired['temporal']['summary'], paired['static_trained']['summary']
    relative = {'raw_es': previous._relative_not_worse(real['joint_fair_es']['raw'], base['joint_fair_es']['raw']),
                'variogram': previous._relative_not_worse(real['variogram']['aggregate'], base['variogram']['aggregate']),
                'velocity_covariance': previous._relative_not_worse(real['covariance_distance']['velocity'],
                                                                  base['covariance_distance']['velocity'])}
    rms = np.asarray(reports['temporal']['summary']['rms_ratio'], dtype=float)
    speed = reports['temporal']['summary']['speed']
    reference_speed = speed['reference_all']['rms']
    predicted_speed = speed['all']['rms']
    speed_ratio = (predicted_speed/reference_speed if predicted_speed is not None
                   and reference_speed is not None and reference_speed>0 else None)
    summary = reports['temporal']['summary']
    acc = reports['temporal']['acceleration']
    quality = {'group_rms': bool(np.isfinite(rms).all() and ((rms>=.65)&(rms<=1.5)).all()),
               'speed': bool(speed_ratio is not None and np.isfinite(speed_ratio)
                             and speed['reference_all']['count']>0 and speed_ratio<=1.5),
               'acceleration': bool(acc['count']>0 and acc['reference_count']>0 and
                                    acc['reference_sum_squares']>0 and np.isfinite(acc['rms_ratio']) and acc['rms_ratio']<=1.5),
               'raw_domain': max(summary['raw_oob'])<=.01,
               'clamp_retention': min(summary['clamp_rms_retention'])>=.95}
    gain = gains['static_trained']
    timing = bool(gain['ci95'][0]>0 and gain['gain']>=.01*base['joint_fair_es']['centered']
                  and all(gains[k]['gain']>0 for k in ('static','reverse','mismatch')) and all(relative.values()))
    return {'timing_passed': timing, 'quality_passed': all(quality.values()), 'gains_baseline_minus_real': gains,
            'relative_protection': relative, 'quality_checks': quality, 'speed_ratio': speed_ratio,
            'rms_ratio': rms.tolist(), 'support': len(support),
            'support_fraction': coverage, 'support_sentences': len(sentences),
            'development_only': True, 'generator_integrated': False, 'default_replaced': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio','targets','native-root','native-manifest','delta-dir','audio-checkpoint','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh output required')
    args.output.mkdir(parents=True); started = time.monotonic()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.manual_seed(SEED)
    old.save_json(args.output/'status.json', {'schema': SCHEMA, 'status': 'loading'})
    loading = copy.copy(args); loading.smoke=False
    clips, original, lineage = old.load_clips(loading); split = previous.split_inner(clips, original)
    if args.smoke:
        split = {k: v[:12] for k, v in split.items()}
    ids = sorted(set(i for k in ('fit','calibration','confirmation') for i in split[k]))
    target = torch.load(args.targets, map_location='cpu', weights_only=False, mmap=True)['splits']['train']
    anchors = {cid: row[previous.CC].numpy() for cid,row in zip(target['clip_id'],target['anchors'])}
    frozen = load_frozen_audio(args.audio_checkpoint,args.device); binding=assert_source_binding(frozen,lineage)
    encoded = encode_clips(frozen,[clips[i] for i in ids],batch_size=16,device=args.device)
    for i, context in zip(ids, encoded):
        clips[i]['global']=np.r_[context['global'].numpy(),context['intensity'].numpy()]
        clips[i]['anchor_upper']=anchors[clips[i]['clip_id']]
    del frozen, encoded
    root=Path(__file__).resolve().parents[1]
    code=['scripts/train_clocked_motion_prior.py','scripts/clocked_motion_dictionary.py',
          'kinetalk_b0/models/clocked_motion_prior.py','scripts/joint_motion_metrics.py',
          'scripts/train_joint_motion_prior.py','scripts/train_motion_process.py',
          'scripts/joint_prior_audio_context.py','scripts/full_native_context_data.py',
          'scripts/full_staged_data.py','kinetalk_b0/models/slow_state_affect.py']
    protocol={'schema':SCHEMA,'source':lineage,'frozen_audio':binding,'smoke':args.smoke,
        'split':{k:[{m:clips[i][m] for m in ('clip_id','sentence','speaker','emotion')} for i in split[k]]
                 for k in ('fit','calibration','confirmation')},
        'confirmation_scope':'Consumed development regression, not untouched',
        'code_sha256':{f:old.sha(root/f) for f in code},
        'protocol_sha256':old.sha(root/'docs/CLOCKED_MOTION_PRIOR_PROTOCOL_20260918.md'),
        'horizon':HORIZON,'hop':HOP,'epochs':2 if args.smoke else 30,'test_loaded':False}
    old.save_json(args.output/'protocol.json',protocol)
    source_dir=args.output/'source'; source_dir.mkdir()
    for f in code:
        destination=source_dir/f; destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((root/f).read_bytes())
    (source_dir/'protocol.md').write_bytes((root/'docs/CLOCKED_MOTION_PRIOR_PROTOCOL_20260918.md').read_bytes())
    windows=shapes.extract_windows(clips,split['fit'],horizon=HORIZON,hop=HOP)
    dictionary=shapes.fit_shape_dictionary(windows,k=16 if args.smoke else 128,seed=SEED)
    torch.save(dictionary,args.output/'dictionary.pt')
    fitted=fit_equilibrium(clips,split['fit']); torch.save(fitted,args.output/'equilibrium.pt')
    scale=previous.raw_scales(clips,split['fit']); old.save_json(args.output/'scales.json',scale.tolist())
    low_cut=float(np.quantile([np.mean((old.center_runs(clips[i]['upper'],clips[i]['valid'].numpy())
                                      [clips[i]['valid'].numpy()]/scale)**2) for i in split['fit']],.2))
    fm,fs=old.fit_feature_stats(clips,split['fit'])
    globals_=np.stack([clips[i]['global'] for i in split['fit']])
    stats=(fm,fs,torch.tensor(globals_.mean(0),dtype=torch.float32),
           torch.tensor(np.maximum(globals_.std(0),.01),dtype=torch.float32))
    data=build_training(clips,windows,dictionary)
    old.save_json(args.output/'training_data.json',{'windows':len(windows),'clips':len(split['fit']),
        'low_activity_train_quantile20':low_cut,'clip_balanced':True,'feature_dim':data['temporal'].shape[-1]})
    models,losses=train_pair(data,stats,dictionary,args.output,args.device,2 if args.smoke else 30)
    del data
    old.save_json(args.output/'short_run_support.json',{
        cell: {'runs':sum(len(old.runs(clips[i]['valid'].numpy())) for i in split[cell]),
               'short_runs':sum(right-left<HORIZON for i in split[cell]
                                for left,right in old.runs(clips[i]['valid'].numpy()))}
        for cell in ('fit','calibration','confirmation')})
    for cell in ('calibration','confirmation'):
        old.save_json(args.output/'status.json',{'schema':SCHEMA,'status':'evaluating','cell':cell})
        reports={}
        arms=[('static_trained','static','static',False),('temporal','temporal','real',False),
              ('static','temporal','static',False),('reverse','temporal','reverse',False),
              ('mismatch','temporal','mismatch',False),('uniform','temporal','real',True)]
        for label,net,intervention,uniform in arms:
            report,curves=evaluate(clips,split[cell],dictionary,fitted,scale,low_cut,models[net],args.device,
                                   intervention=intervention,uniform=uniform)
            reports[label]=report
            old.save_json(args.output/(cell+'_'+label+'.json'),report)
            torch.save(curves,args.output/(cell+'_'+label+'.pt'))
            print('EVAL',cell,label,report['summary']['joint_fair_es'] if report['summary'] else 'NO_SUPPORT',
                  report['summary']['rms_ratio'] if report['summary'] else None,flush=True)
        gate=assessment(reports)
        old.save_json(args.output/(cell+'_assessment.json'),gate)
        print('ASSESSMENT',cell,json.dumps(gate),flush=True)
    old.save_json(args.output/'status.json',{'schema':SCHEMA,'status':'complete','seconds':time.monotonic()-started,
        'epochs_per_arm':len(losses['temporal']),'development_only':True,'smoke':args.smoke,
        'timing_passed':gate['timing_passed'],'quality_passed':gate['quality_passed'],
        'generator_integrated':False,'default_replaced':False,'test_loaded':False})


if __name__=='__main__':
    main()
