"""Read-only likelihood/calibration decomposition of a completed process pilot.

Loads the saved normalization, teacher plans and final priors. No segmentation,
normalization, model fitting, checkpoint selection or free rollout is performed.
All target-motion use is explicitly teacher-forced scoring, never deployment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_motion_process as p
from scripts.motion_process_representation import render_motion_process
from kinetalk_b0.models.motion_process_prior import MotionProcessPrior, joint_event_nll, initial_nll


SCHEMA = 'motion_process_failure_diagnosis_v1'


def normal_components(loc, log_scale, target):
    """Unreduced Gaussian diagnostics in the saved normalization's units."""
    if loc.shape != log_scale.shape or loc.shape != target.shape:
        raise ValueError('Matching distribution/target shapes required')
    if not all(torch.isfinite(x).all() for x in (loc, log_scale, target)):
        raise ValueError('Nonfinite Gaussian diagnostic input')
    std = log_scale.exp()
    err = loc-target
    # Match the original Gaussian's FP32 operation order. Dividing by exp(s)
    # is mathematically equal but can differ noticeably for a very large NLL.
    z2 = ((target-loc)*torch.exp(-log_scale)).square()
    return {'gaussian_nll': .5*z2+log_scale+.5*math.log(2*math.pi),
        'squared_error': err.square(), 'absolute_error': err.abs(),
        'standardized_squared_error': z2, 'predicted_std': std,
        'predicted_variance': std.square(), 'log_std': log_scale,
        'coverage_1std': (z2 <= 1).to(loc.dtype),
        'coverage_2std': (z2 <= 4).to(loc.dtype),
        'std_at_lower_bound': (log_scale <= -3.+1e-6).to(loc.dtype),
        'std_at_upper_bound': (log_scale >= 2.-1e-6).to(loc.dtype),
        'predicted_mean': loc, 'target_mean': target,
        'target_squared': target.square()}


def event_components(out, duration_index, delta):
    """Duration CE plus true-duration delta NLL, with sign calibration.

    The conditional sign score knows the observed duration, just like joint
    likelihood. The marginal sign score integrates the predicted durations.
    Exact/near-zero target deltas are counted but excluded from sign summaries.
    """
    joint = joint_event_nll(out, duration_index, delta)
    index = duration_index[:, None]
    logits = out['duration_logits']
    loc = out['loc'].gather(1, index).squeeze(1)
    log_std = out['log_scale'].gather(1, index).squeeze(1)
    parts = normal_components(loc, log_std, delta)
    probs = logits.softmax(-1)
    positive = .5 * (1 + torch.erf(out['loc'] / (out['log_scale'].exp()*math.sqrt(2))))
    conditional_positive = positive.gather(1, index).squeeze(1)
    marginal_positive = (positive*probs).sum(-1)
    truth = (delta > 0).to(delta.dtype)
    parts.update(joint_nll=joint,
        duration_ce=-logits.log_softmax(-1).gather(1, index).squeeze(1),
        duration_brier=(probs-torch.nn.functional.one_hot(duration_index, logits.shape[1])).square().sum(-1),
        direction_brier_conditional=(conditional_positive-truth).square(),
        direction_brier_marginal=(marginal_positive-truth).square(),
        direction_accuracy_conditional=((conditional_positive >= .5) == (delta > 0)).to(delta.dtype),
        direction_accuracy_marginal=((marginal_positive >= .5) == (delta > 0)).to(delta.dtype),
        zero_delta=(delta.abs() <= 1e-8).to(delta.dtype))
    if not torch.allclose(parts['joint_nll'], parts['duration_ce']+parts['gaussian_nll'], atol=1e-5, rtol=1e-6):
        raise RuntimeError('Joint likelihood does not decompose exactly')
    return parts


def attach_saved_targets(clips, plans, scales):
    """Recreate training event records from immutable plans without re-fitting."""
    if set(plans) != {c['clip_id'] for c in clips}:
        raise ValueError('Saved plan membership differs from declared queries')
    scales = np.asarray(scales, dtype=np.float64)
    for c in clips:
        plan = plans[c['clip_id']]
        if (not np.array_equal(np.asarray(plan['scales']), scales)
                or tuple(plan['durations']) != p.DURATIONS
                or not np.array_equal(plan['mask'], c['groupmask'])):
            raise ValueError('Saved plan scale/grid/clock mismatch: '+c['clip_id'])
        render_motion_process(plan)  # Validation only; no segmentation or fit.
        events = []
        initial = {}
        for group, segments in enumerate(plan['segments']):
            previous_delta, previous_duration, previous_end = 0., 0., -1
            for s in segments:
                start, duration = s['start'], s['duration']
                if start != previous_end:
                    previous_delta, previous_duration = 0., 0.
                    initial.setdefault(start, np.full(4, np.nan))[group] = (
                        s['start_value']-c['anchor'][group])/scales[group]
                if (not np.isclose(s['start_value'], c['state'][start, group], rtol=0, atol=1e-7)
                        or not np.isclose(s['end_value'], c['state'][start+duration, group], rtol=0, atol=1e-7)):
                    raise ValueError('Saved teacher endpoint differs from source motion')
                delta = (s['end_value']-s['start_value'])/scales[group]
                if duration in p.DURATIONS and not s['right_censored']:
                    events.append([start, group, (s['start_value']-c['anchor'][group])/scales[group],
                        previous_delta, previous_duration, p.DURATIONS.index(duration), delta])
                previous_delta, previous_duration, previous_end = delta, float(duration), start+duration
        if sorted(initial) != [start for start, _ in p.runs(c['valid'].numpy())]:
            raise ValueError('Saved initial states do not match independent valid runs')
        initial_y = np.asarray([initial[start] for start in sorted(initial)], dtype=np.float32)
        if not np.isfinite(initial_y).all():
            raise ValueError('All four groups require saved initial states')
        c['events'] = np.asarray(events, dtype=np.float32).reshape(-1, 7)
        c['initial'] = initial_y


def summarize(table, selected=None):
    if not table:
        return {'count': 0}
    n = len(next(iter(table.values())))
    select = np.ones(n, bool) if selected is None else np.asarray(selected, bool)
    if select.shape != (n,):
        raise ValueError('Summary selection differs from table length')
    result = {'count': int(select.sum())}
    for key, value in table.items():
        if key.startswith('_'):
            continue
        use = select & (table['zero_delta'] == 0) if key.startswith('direction_') else select
        result[key] = float(np.mean(value[use])) if use.any() else None
    if 'zero_delta' in table:
        result['direction_count'] = int((select & (table['zero_delta'] == 0)).sum())
    return result


def _append_table(destination, components, **indices):
    for key, value in {**components, **indices}.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        destination.setdefault(key, []).append(np.asarray(value).reshape(-1))


def _flatten(parts):
    return {key: np.concatenate(value) for key, value in parts.items()}


@torch.no_grad()
def score(model, clips, ids, mean, std, *, device, local_enabled, mode):
    event_parts, initial_parts, exact_replay = {}, {}, {}
    for start in range(0, len(ids), 16):
        selected = ids[start:start+16]
        f, valid, ev, initial_ix, initial_y = p.batch(clips, selected, mean, std, device)
        hidden, pooled = model.forward_context(f, valid, local_enabled=local_enabled)
        if mode == 'static':
            hidden = (hidden*valid[..., None]).sum(1, keepdim=True)/valid.sum(1)[:, None, None]
            hidden = hidden.expand(-1, valid.shape[1], -1)
        elif mode != 'real':
            raise ValueError('Only saved-model real/static diagnostic modes allowed')
        batch_ids = torch.as_tensor(selected, dtype=torch.long, device=device)
        if len(ev):
            b, t, group = ev[:, 0].long(), ev[:, 1].long(), ev[:, 2].long()
            out = model.event_distribution(hidden[b, t], pooled[b], group, ev[:, 3], ev[:, 4], ev[:, 5])
            components = event_components(out, ev[:, 6].long(), ev[:, 7])
            _append_table(event_parts, components, _clip=batch_ids[b], _group=group,
                _duration=ev[:, 6].long(), _initial=(ev[:, 5] == 0).long())
        ini = model.initial_distribution(pooled[initial_ix])
        ic = normal_components(ini['loc'], ini['log_scale'], initial_y)
        # Reproduce original score reductions on-device, before moving anything
        # to NumPy. Mean(group NLL)*4 can round differently from sum(groups)
        # then mean(runs), especially when the initial NLL is very large.
        ini_exact = initial_nll(ini, initial_y)
        for batch_index, i in enumerate(selected):
            em = ev[:, 0].long() == batch_index
            im = initial_ix == batch_index
            exact_replay[i] = {'event_nll': float(components['joint_nll'][em].mean()) if em.any() else None,
                'initial_nll': float(ini_exact[im].mean())}
        _append_table(initial_parts, ic, _clip=batch_ids[initial_ix, None].expand(-1, 4),
            _group=torch.arange(4, device=device)[None].expand(len(initial_ix), -1))
    events, initial = _flatten(event_parts), _flatten(initial_parts)
    clip_rows = []
    for i in ids:
        c = clips[i]
        clip_rows.append({k: c[k] for k in ('clip_id', 'speaker', 'sentence', 'emotion')} | {
            'events': summarize(events, events['_clip'] == i) if events else {'count': 0},
            'initial': summarize(initial, initial['_clip'] == i),
            'exact_original_reduction': exact_replay[i]})
    event_summary = summarize(events)
    for key in ('duration_ce', 'gaussian_nll', 'joint_nll'):
        values = [row['events'][key] for row in clip_rows if row['events']['count']]
        event_summary['clip_mean_'+key] = float(np.mean(values)) if values else None
    return {'clips': len(ids), 'event_weighted': event_summary, 'initial_group_weighted': summarize(initial),
        'by_group': {name: {'events': summarize(events, events['_group'] == g) if events else {'count': 0},
            'initial': summarize(initial, initial['_group'] == g)} for g, name in enumerate(p.GROUPS)},
        'by_duration': {str(duration): summarize(events, events['_duration'] == d) if events else {'count': 0}
            for d, duration in enumerate(p.DURATIONS)},
        'by_segment_position': {name: summarize(events, events['_initial'] == flag) if events else {'count': 0}
            for name, flag in (('initial_left_censored', 1), ('interior', 0))},
        'rows': clip_rows}


def verify_saved_nll(diagnostic, saved_path):
    saved = json.loads(Path(saved_path).read_text(encoding='utf8'))
    rows = saved['rows']
    if [r['clip_id'] for r in diagnostic['rows']] != [r['clip_id'] for r in rows]:
        raise ValueError('Original and diagnostic evaluation memberships differ')
    event_errors, initial_errors = [], []
    for a, b in zip(diagnostic['rows'], rows):
        if a['events']['count'] != b['event_count']:
            raise ValueError('Completed event counts changed')
        if b['event_nll'] is not None:
            event_errors.append(abs(a['exact_original_reduction']['event_nll']-b['event_nll']))
        initial_errors.append(abs(a['exact_original_reduction']['initial_nll']-b['initial_nll']))
    errors = {'max_event_nll_abs_error': max(event_errors, default=0.),
              'max_initial_nll_abs_error': max(initial_errors, default=0.)}
    if max(errors.values()) > 2e-4:
        raise ValueError('Recomputed saved likelihood differs: '+str(errors))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'targets', 'native-root', 'native-manifest', 'delta-dir', 'run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh read-only diagnosis output required')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads((args.run/'protocol.json').read_text(encoding='utf8'))
    status = json.loads((args.run/'status.json').read_text(encoding='utf8'))
    if protocol['smoke'] or status.get('status') != 'complete':
        raise ValueError('A completed formal run is required')
    for name, expected in protocol['code_sha256'].items():
        if p.sha(Path(__file__).resolve().parents[1]/name) != expected:
            raise ValueError('Scoring source changed from formal run: '+name)
    args.smoke = False
    clips, split, lineage = p.load_clips(args)
    if lineage != protocol['source']:
        raise ValueError('Source lineage changed')
    for cell, ids in split.items():
        observed = [{k: clips[i][k] for k in ('clip_id', 'speaker', 'sentence', 'emotion')} for i in ids]
        if observed != protocol['split'][cell]:
            raise ValueError('Fixed diagnostic split changed')
    normalization_path = args.run/'normalization.pt'
    normal = torch.load(normalization_path, map_location='cpu', weights_only=False)
    if normal['fit_clip_ids'] != [clips[i]['clip_id'] for i in split['fit']]:
        raise ValueError('Saved normalization fitting scope changed')
    plans = torch.load(args.run/'teacher_plans.pt', map_location='cpu', weights_only=False)
    attach_saved_targets(clips, plans, normal['scales'])
    del plans
    report = {'schema': SCHEMA, 'scope': 'Teacher-forced final-model diagnosis only; no refit or new training',
        'sign_policy': 'Positive delta versus negative; abs(delta)<=1e-8 excluded from direction scores',
        'normalization': 'All deltas and initial levels use the original fit-only scales and independent neutral anchors',
        'weighting': 'event_weighted pools events; clip_mean_* reproduces original per-clip equal weighting; initial rows pool runs and groups',
        'limitations': 'Teacher likelihood is not free-rollout quality. Static intervention may be outside training conditions. Fit/holdout contrasts do not identify whether acoustics contain sufficient information.',
        'source': {'protocol_sha256': p.sha(args.run/'protocol.json'),
            'normalization_sha256': p.sha(normalization_path), 'plans_sha256': p.sha(args.run/'teacher_plans.pt'),
            'diagnostic_code_sha256': p.sha(__file__)}, 'scores': {}, 'replay': {}}
    for arm, local_enabled in (('global_only', False), ('local_audio', True)):
        path = args.run/arm/'final.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved['normalization_sha256'] != p.sha(normalization_path) or saved['arm'] != arm
                or saved['epochs'] != protocol['epochs']):
            raise ValueError('Final prior lineage differs')
        model = MotionProcessPrior(feature_dim=1540, hidden=96, groups=4, durations=p.DURATIONS).to(args.device)
        model.load_state_dict(saved['model'])
        model.eval().requires_grad_(False)
        report['source'][arm+'_final_sha256'] = p.sha(path)
        report['scores'][arm] = {}
        for mode in (('real', 'static') if local_enabled else ('real',)):
            report['scores'][arm][mode] = {}
            for cell in p.CELLS:
                result = score(model, clips, split[cell], normal['mean'], normal['std'],
                    device=args.device, local_enabled=local_enabled, mode=mode)
                report['scores'][arm][mode][cell] = result
                if mode == 'real' and cell != 'fit':
                    report['replay'][arm+'/'+cell] = verify_saved_nll(result, args.run/arm/(cell+'.json'))
                p.save_json(args.output/'report.json', report)
                print('DIAGNOSED', arm, mode, cell, json.dumps(result['event_weighted']), flush=True)
        del model
    report['seconds'] = time.monotonic()-started
    p.save_json(args.output/'report.json', report)
    p.save_json(args.output/'complete.json', {'schema': SCHEMA, 'report_sha256': p.sha(args.output/'report.json'),
        'seconds': report['seconds'], 'training_started': False, 'normalization_refitted': False,
        'segmentation_refitted': False, 'checkpoint_selected': False})
    print('DIAGNOSIS_COMPLETE', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
