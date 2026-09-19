"""Frozen AE/prior controls on the exact event receiver development clips.

This is a post-hoc bottleneck audit, not a training or model-promotion script.
Oracle reconstruction is deterministic and reads observed target motion;
source-prior generation uses the same four declared draws as the event arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousUpperAE, ContinuousLatentFlow
from scripts import train_continuous_motion_latent as common
from scripts import evaluate_continuous_motion_latent as native
from scripts.arkit_benchmark_report import build_report, score_fullface, write_report
from scripts.probe_motion_condition_predictability import split_train_pool, sha
from scripts.train_prior_audio_adapter import diverse_diagnostics
from scripts.joint_motion_metrics import summarize, GROUP_NAMES, GROUPS

SCHEMA = 'event_receiver_bottleneck_audit_v1'
SEEDS = (42, 123, 2026, 77)
EXISTING = {'event_oracle': 'control/oracle', 'event_null': 'control/null',
            'event_empty': 'control/empty', 'event_shifted': 'control/shifted',
            'event_audio': 'generation/audio', 'event_static': 'generation/matched_static',
            'event_reverse': 'generation/reverse'}
SOURCES = ('scripts/audit_event_receiver_bottleneck.py', 'scripts/evaluate_continuous_motion_latent.py',
           'scripts/joint_motion_metrics.py', 'scripts/arkit_benchmark_report.py',
           'scripts/evaluate_arkit_literature_metrics.py', 'scripts/train_prior_audio_adapter.py',
           'scripts/probe_motion_condition_predictability.py', 'scripts/train_continuous_motion_latent.py',
           'kinetalk_b0/models/continuous_upper_motion.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def verify_curve_arm(curves, clips, *, stochastic):
    ids = [c['clip_id'] for c in clips]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate audit clip ID')
    if set(curves) != set(ids): raise ValueError('Audit arm clip membership differs')
    for clip in clips:
        row = curves[clip['clip_id']]; valid = array(clip['valid'])
        observed = array(clip['motion_mask'])
        if valid.ndim != 1 or valid.dtype != bool or observed.shape != (len(valid), 9) or observed.dtype != bool:
            raise ValueError('Boolean native valid[T] and motion_mask[T,9] required')
        joint = valid & observed.all(1)
        if not np.array_equal(row['native_valid'], valid) or not np.array_equal(row['score_mask'], joint):
            raise ValueError('Audit arm native/score mask differs: '+clip['clip_id'])
        if not np.array_equal(row['target'], array(clip['motion9']), equal_nan=True):
            raise ValueError('Audit arm reference differs: '+clip['clip_id'])
        samples = np.asarray(row['samples'])
        expected = len(SEEDS) if stochastic else 1
        if samples.shape != (expected, len(valid), 9): raise ValueError('Unexpected audit sample shape')
        if stochastic and row['seeds'] != list(SEEDS): raise ValueError('Event arm draw seeds differ')
        if not np.isfinite(samples[:, valid]).all(): raise ValueError('Nonfinite audit samples')
        if stochastic and not np.array_equal(row['generated_mask'], valid):
            raise ValueError('Generation support must use native audio clock')


def signal_audit(samples, clip):
    """Raw coefficient diagnostics, independent of normalized motion scales."""
    valid = array(clip['valid']) & array(clip['motion_mask']).all(1)
    target = array(clip['motion9']); rows = []
    spans = native.contiguous_runs(valid)
    for name, group in zip(GROUP_NAMES, GROUPS):
        p = samples[:, valid][:, :, group].astype(np.float64)
        y = target[valid][:, group].astype(np.float64)
        if not len(y): continue
        prediction_mean = p.mean(1); truth_mean = y.mean(0)
        bias = prediction_mean-truth_mean[None]
        pred_energy = 0.; reference_energy = 0.; frame_channels = 0
        for start, stop in spans:
            px = samples[:, start:stop][:, :, group].astype(np.float64)
            yy = target[start:stop][:, group].astype(np.float64)
            px = px-px[:, :1]; px = px-px.mean(1, keepdims=True)
            yy = yy-yy[:1]; yy = yy-yy.mean(0, keepdims=True)
            pred_energy += float(np.square(px).sum()/len(samples))
            reference_energy += float(np.square(yy).sum())
            frame_channels += (stop-start)*len(group)
        rows.append({'group': name, 'prediction_mean': float(prediction_mean.mean()),
                     'reference_mean': float(truth_mean.mean()),
                     'mean_bias_rmse': float(np.sqrt(np.square(bias).mean())),
                     'raw_mse': float(np.square(p-y[None]).mean()),
                     'raw_centered_prediction_rms': float(np.sqrt(pred_energy/frame_channels)),
                     'raw_centered_reference_rms': float(np.sqrt(reference_energy/frame_channels)),
                     'raw_centered_rms_ratio': float(np.sqrt(pred_energy/reference_energy)) if reference_energy > 0 else None,
                     'prediction_below_zero_fraction': float((p < 0).mean()),
                     'prediction_above_one_fraction': float((p > 1).mean()),
                     'reference_oob_fraction': float(((y < 0) | (y > 1)).mean()),
                     'audio_b9_mean': float(array(clip['b9'])[list(group)].mean())})
    return {'clip_id': clip['clip_id'], 'groups': rows,
            'raw_centered_rms_definition': 'coefficient units; center each common observed run per channel; pool time/channels and mean sample energy; no fit scales'}


def rescore_arm(curves, clips, stats, destination, *, deterministic, scope):
    verify_curve_arm(curves, clips, stochastic=not deterministic)
    metrics = []; literature = []; numerics = []; signals = []
    for clip in clips:
        samples = np.asarray(curves[clip['clip_id']]['samples'])
        metric, _ = native._summary_scores(samples, clip, stats, deterministic=deterministic)
        if metric is not None: metrics.append(metric)
        full = native.compose_full(samples, array(clip['baseline52']))
        literature.append(score_fullface(full, clip))
        numerics.append({'clip_id': clip['clip_id'], **native._numerics(samples, full,
                         array(clip['baseline52']), array(clip['valid']))})
        signals.append(signal_audit(samples, clip))
    result = {'scope': scope, 'deterministic': deterministic,
              'summary': summarize(metrics) if metrics else None, 'per_clip_scores': metrics,
              'numerics': numerics, 'raw_signal_diagnostics': signals,
              'motion_score_support': 'native valid AND all nine motion_mask channels; identical across arms',
              'arkit_score_support': 'native valid AND per-channel channel_mask; AE oracle keeps baseline on partially observed upper-nine frames it cannot reconstruct',
              'rms_definition': 'sqrt(pooled mean-sample within-run centered energy / pooled target energy); each channel divided by common fit metric_scale (or residual_scale fallback)',
              'naturalness_certified': False, 'default_replaced': False}
    common._write(destination/'audit.json', result)
    report = build_report(literature, scope=scope)
    write_report(destination/'arkit_benchmark.json', report)
    return result, report


def baseline_curves(clips):
    return {c['clip_id']: {'samples': array(c['baseline52'])[:, native.UPPER][None],
             'target': array(c['motion9']), 'native_valid': array(c['valid']),
             'score_mask': array(c['valid']) & array(c['motion_mask']).all(1),
             'generated_mask': array(c['valid']), 'seeds': ['baseline']}
            for c in clips}


def table_row(name, audit, arkit):
    summary = audit['summary'] or {}; scores = arkit['summary']
    return {'arm': name, 'deterministic': audit['deterministic'],
            'rms_ratio': summary.get('rms_ratio'), 'raw_oob': summary.get('raw_oob'),
            'clamp_rms_retention': summary.get('clamp_rms_retention'),
            'joint_fair_es': summary.get('joint_fair_es'),
            'arkit': {k: scores[k] for k in ('arkit_mbe', 'arkit_lbe', 'arkit_fdd_absolute',
                                            'supp_upper9_fdd_absolute')}}


def run(args):
    if args.output.exists(): raise FileExistsError('Fresh audit directory required')
    torch.set_num_threads(4); started = time.monotonic()
    args.output.mkdir(parents=True)
    def status(state, **detail):
        common._write(args.output/'status.json', {'schema': SCHEMA, 'state': state,
                     'updated': time.time(), 'default_replaced': False, **detail})
    status('loading')
    reference = read(args.source_run/'protocol.json'); event = read(args.event_run/'protocol.json')
    if event.get('smoke'): raise ValueError('Formal event run required')
    if sha(args.dataset) != reference['dataset_sha256'] or event['dataset_sha256'] != reference['dataset_sha256']:
        raise ValueError('Dataset binding mismatch')
    if event['source_protocol_sha256'] != sha(args.source_run/'protocol.json'):
        raise ValueError('Source protocol binding mismatch')
    for key, file in [('fit_stats_sha256', 'fit_stats.pt'), ('ae_sha256', 'ae_final.pt'), ('prior_sha256', 'prior_final.pt')]:
        if event[key] != sha(args.source_run/file): raise ValueError('Source file binding mismatch: '+file)
    payload = torch.load(args.dataset, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('schema') != 'continuous_motion_dataset_v1': raise ValueError('Unexpected dataset schema')
    fit, validation = split_train_pool(payload['clips'], reference); del payload
    if event['train_ids'] != [c['clip_id'] for c in fit] or event['validation_ids'] != [c['clip_id'] for c in validation]:
        raise ValueError('Source event split mismatch')
    # Both selection and its verification read metadata, never outer targets.
    clips = diverse_diagnostics(validation, 24); ids = [c['clip_id'] for c in clips]
    if len(ids) != 24: raise ValueError('Expected exact formal 24-clip diagnostics')
    if read(args.event_run/'control/oracle/result.json')['selected_clip_ids'] != ids:
        raise ValueError('Event receiver selection differs from fixed metadata rule')
    stats = common._load(args.source_run/'fit_stats.pt')
    if set(stats['train_clip_ids']) != {c['clip_id'] for c in fit}: raise ValueError('Statistics fitting members differ')
    ae_ck = common._load(args.source_run/'ae_final.pt'); prior_ck = common._load(args.source_run/'prior_final.pt')
    expected = common._value_sha(reference)
    if any(c.get('binding', {}).get('protocol_sha256') != expected for c in (ae_ck, prior_ck)):
        raise ValueError('Source checkpoint protocol binding differs')
    if (prior_ck['binding'].get('ae_sha256') != common._value_sha(ae_ck['state']) or
            prior_ck['binding'].get('stats_sha256') != common._value_sha(stats)):
        raise ValueError('Source prior AE/statistics binding differs')
    source_paths = {name: args.event_run/path/'curves.pt' for name, path in EXISTING.items()}
    sources = {name: sha(path) for name, path in source_paths.items()}
    code = Path(__file__).resolve().parents[1]
    common._write(args.output/'protocol.json', {'schema': SCHEMA, 'posthoc': True,
       'dataset_sha256': reference['dataset_sha256'], 'event_protocol_sha256': sha(args.event_run/'protocol.json'),
       'source_protocol_sha256': sha(args.source_run/'protocol.json'), 'fit_stats_sha256': sha(args.source_run/'fit_stats.pt'),
       'ae_sha256': sha(args.source_run/'ae_final.pt'), 'prior_sha256': sha(args.source_run/'prior_final.pt'),
       'event_curve_sha256': sources, 'selected_ids': ids, 'seeds': list(SEEDS), 'steps': 24,
       'code_sha256': {p: sha(code/p) for p in SOURCES}, 'outer_tensors_indexed': False,
       'oracle_note': 'One deterministic observed-motion reconstruction; no sampled seed or audio predictability claim',
       'prior_note': 'Frozen source prior with use_audio=False; identical four seeds/native noise keys to event arms',
       'baseline_note': 'Original frozen fullface baseline; deterministic duplicate for metric API only',
       'scope': 'same24 inner-development posthoc bottleneck diagnosis; not sealed test or acceptance',
       'default_replaced': False})
    for path in SOURCES:
        dest = args.output/'source'/path; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((code/path).read_bytes())
    ae = ContinuousUpperAE(**ae_ck['config']).to(args.device)
    ae.load_state_dict(ae_ck['state']); ae.eval().requires_grad_(False)
    prior = ContinuousLatentFlow(**prior_ck['config']).to(args.device)
    prior.load_state_dict(prior_ck['state']); prior.eval().requires_grad_(False)
    state_before = (common._value_sha(ae.state_dict()), common._value_sha(prior.state_dict()))
    status('evaluating', arm='ae_oracle')
    native.evaluate_ae(ae, clips, stats, args.output/'ae_oracle', args.device)
    status('evaluating', arm='source_prior')
    native.evaluate_generation(prior, ae, clips, stats, args.output/'source_prior', args.device,
                               use_audio=False, seeds=SEEDS, steps=24)
    if state_before != (common._value_sha(ae.state_dict()), common._value_sha(prior.state_dict())):
        raise RuntimeError('Frozen source state changed during audit')
    arms = {'baseline': (baseline_curves(clips), True),
            'ae_oracle': (common._load(args.output/'ae_oracle/curves.pt')['clips'], True),
            'source_prior': (common._load(args.output/'source_prior/curves.pt')['clips'], False)}
    arms.update({name: (common._load(path)['clips'], False) for name, path in source_paths.items()})
    table = []
    for name, (curves, deterministic) in arms.items():
        status('rescoring', arm=name)
        audit, arkit = rescore_arm(curves, clips, stats, args.output/name,
                    deterministic=deterministic, scope=name+'; same24 inner-development receiver audit')
        table.append(table_row(name, audit, arkit))
    common._write(args.output/'summary.json', {'schema': SCHEMA, 'selected_ids': ids,
        'group_order': list(GROUP_NAMES), 'table': table, 'frozen_source_state_exact': True,
        'naturalness_certified': False, 'default_replaced': False,
        'interpretation': ['Oracle event sensitivity is not motion-amplitude or naturalness acceptance.',
          'Weak AE oracle reconstruction localizes a representation/decoder bottleneck.',
          'Good AE reconstruction but weak source prior localizes prior learning/sampling.',
          'Better source prior than event oracle suggests event-receiver adaptation degraded generation.',
          'Raw out-of-range and clamp RMS retention must be read with centered amplitude; no clipping was used for scoring.']})
    lines = ['# Event receiver bottleneck audit', '',
             'Post-hoc same24 inner-development comparison. No training, promotion, or success claim.', '',
             '| Arm | Up/down/squint/wide RMS ratio | Raw OOB fraction | Clamp RMS retention | MBE | LBE |',
             '| --- | --- | --- | --- | ---: | ---: |']
    def seq(x): return ' / '.join('pending' if y is None else f'{y:.4f}' for y in x) if x else 'pending'
    def val(x): return 'pending' if x.get('value') is None else f"{x['value']:.6f}"
    for row in table:
        lines.append(f"| {row['arm']} | {seq(row['rms_ratio'])} | {seq(row['raw_oob'])} | {seq(row['clamp_rms_retention'])} | {val(row['arkit']['arkit_mbe'])} | {val(row['arkit']['arkit_lbe'])} |")
    lines += ['', 'AE oracle and baseline are deterministic. All free-generation arms use the same four seeds.',
              'AE reconstructs jointly observed target runs; generation uses native audio runs. Scoring support is identical.',
              'RMS is a ratio of pooled within-run centered energy, after each channel is divided by its common fit scale. It is not pointwise amplitude or motion alignment.',
              'Raw coefficient centered RMS/ratios, mean/bias and OOB diagnostics are in each audit.json. Zero raw motion has clamp-retention 1 by convention, not evidence of useful animation.',
              'ARKit uses per-channel observation masks; partially observed upper-nine frames use the unchanged baseline for AE oracle. Joint-motion RMS/ES uses only fully observed upper-nine frames.',
              'The engineering event-control gate does not certify amplitude, timing predictability, or naturalness.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    status('complete', seconds=time.monotonic()-started, evaluated_clips=len(clips), naturalness_certified=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'source-run', 'event-run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try: run(args)
    except Exception as exc:
        if args.output.exists() and not isinstance(exc, FileExistsError):
            common._write(args.output/'status.json', {'schema': SCHEMA, 'state': 'failed',
                         'error': repr(exc), 'updated': time.time(), 'default_replaced': False})
        raise
