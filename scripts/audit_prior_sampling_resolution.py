"""Fixed-checkpoint Euler/Heun diagnostic on the receiver audit's same 24 clips.

Post-hoc sampling diagnosis only: no training, best-arm selection, model
promotion, query-driven calibration, or changes to previous result files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from kinetalk_b0.models.continuous_upper_motion import (
    ContinuousLatentFlow, ContinuousUpperAE, _clean_sequence, _prefix_mask,
)
from scripts import audit_event_receiver_bottleneck as audit

common = audit.common
native = audit.native
SCHEMA = 'prior_sampling_resolution_audit_v1'
ARMS = (('euler24', 'euler', 24), ('euler96', 'euler', 96),
        ('heun24', 'heun', 24), ('heun48', 'heun', 48))
ATOL = 1e-6
RTOL = 1e-6
SOURCES = tuple(dict.fromkeys(('scripts/audit_prior_sampling_resolution.py', *audit.SOURCES)))


class SamplingWrapper(nn.Module):
    """Only replace integration; the checkpoint's vector field stays frozen."""
    def __init__(self, prior, method):
        super().__init__()
        if method not in ('euler', 'heun'):
            raise ValueError('Expected euler or heun')
        self.prior = prior
        self.method = method
        self.latent_dim = prior.latent_dim
        self.block_size = prior.block_size

    @torch.no_grad()
    def sample(self, valid, context, audio_blocks, noise, steps=24,
               use_audio=False, *, audio_frame_valid=None):
        if use_audio is not False:
            raise ValueError('This fixed prior diagnostic requires use_audio=False')
        if self.method == 'euler':
            # Call the original method so the historical Euler24 check detects
            # changed data/model/evaluation conditions, not a duplicate solver.
            return self.prior.sample(valid, context, audio_blocks, noise, steps=steps,
                                     use_audio=False, audio_frame_valid=audio_frame_valid)
        if type(steps) is not int or steps < 1:
            raise ValueError('steps must be a positive integer')
        _prefix_mask(valid)
        x = _clean_sequence(noise, valid, self.latent_dim, 'noise')
        for step in range(steps):
            left = x.new_full((len(x),), step / steps)
            right = x.new_full((len(x),), (step + 1) / steps)
            v0 = self.prior.velocity(x, left, valid, context, audio_blocks, False,
                                     audio_frame_valid=audio_frame_valid)
            predicted = torch.where(valid[..., None], x + v0 / steps, 0.)
            v1 = self.prior.velocity(predicted, right, valid, context, audio_blocks, False,
                                     audio_frame_valid=audio_frame_valid)
            x = torch.where(valid[..., None], x + (v0 + v1) / (2 * steps), 0.)
        return x


def verify_same_draws(curves, reference, clips, *, compare_samples=False):
    """Check support, target, seeds and native per-run RNG keys before scoring."""
    audit.verify_curve_arm(curves, clips, stochastic=True)
    audit.verify_curve_arm(reference, clips, stochastic=True)
    exact = True
    max_absolute_difference = 0.
    for clip in clips:
        cid = clip['clip_id']
        row, original = curves[cid], reference[cid]
        records = []
        for run_index, (start, stop) in enumerate(native.contiguous_runs(audit.array(clip['valid']))):
            for seed in audit.SEEDS:
                records.append({'seed': seed, 'run_index': run_index, 'start': start, 'stop': stop,
                                'noise_seed': native._seed(cid, run_index, seed),
                                'latent_frames': (stop-start+clip['_audit_block_size']-1)//clip['_audit_block_size']})
        if row.get('run_records') != records or original.get('run_records') != records:
            raise ValueError('Native noise keys or generation runs differ: '+cid)
        if compare_samples:
            values, expected = np.asarray(row['samples']), np.asarray(original['samples'])
            if values.dtype != expected.dtype:
                raise ValueError('Euler24 reference sample dtype differs: '+cid)
            if not np.allclose(values, expected, atol=ATOL, rtol=RTOL, equal_nan=True):
                raise ValueError('Euler24 does not reproduce saved source_prior: '+cid)
            finite = np.isfinite(values) & np.isfinite(expected)
            difference = np.abs(values[finite].astype(np.float64)-expected[finite].astype(np.float64))
            max_absolute_difference = max(max_absolute_difference, float(difference.max(initial=0.)))
            exact = exact and np.array_equal(values, expected, equal_nan=True)
    return {'same_native_runs_seeds_and_noise_keys': True, 'samples_compared': compare_samples,
            'allclose': True if compare_samples else None,
            'bitwise_equal': exact if compare_samples else None,
            'max_absolute_difference': max_absolute_difference if compare_samples else None,
            'atol': ATOL, 'rtol': RTOL}


def signal_summary(result):
    """Keep group means/bias alongside the shared pooled-RMS/ES/ARKit table."""
    fields = ('prediction_mean', 'reference_mean', 'mean_bias_rmse', 'raw_mse',
              'prediction_below_zero_fraction', 'prediction_above_one_fraction',
              'reference_oob_fraction', 'audio_b9_mean')
    return {'aggregation': 'unweighted mean of clip-level diagnostics; not pooled RMS',
            'groups': [dict(group=group, **{
                field: float(np.mean([g[field] for c in result['raw_signal_diagnostics']
                                      for g in c['groups'] if g['group'] == group]))
                for field in fields}) for group in audit.GROUP_NAMES]}


def run(args):
    if args.output.exists():
        raise FileExistsError('Fresh sampling audit directory required')
    torch.set_num_threads(4)
    started = time.monotonic()
    args.output.mkdir(parents=True)

    def status(state, **details):
        common._write(args.output/'status.json', {'schema': SCHEMA, 'state': state,
                     'updated': time.time(), 'default_replaced': False, **details})

    status('loading')
    source = audit.read(args.source_run/'protocol.json')
    reference = audit.read(args.reference_audit/'protocol.json')
    if reference.get('schema') != audit.SCHEMA or reference.get('steps') != 24:
        raise ValueError('Expected completed Euler24 receiver bottleneck audit')
    if audit.read(args.reference_audit/'status.json').get('state') != 'complete':
        raise ValueError('Reference audit is not complete')
    if reference.get('seeds') != list(audit.SEEDS):
        raise ValueError('Reference draw seeds differ')
    if audit.sha(args.dataset) != source['dataset_sha256'] or source['dataset_sha256'] != reference['dataset_sha256']:
        raise ValueError('Dataset binding differs')
    if reference['source_protocol_sha256'] != audit.sha(args.source_run/'protocol.json'):
        raise ValueError('Source protocol binding differs')
    bound_files = {'fit_stats_sha256': 'fit_stats.pt', 'ae_sha256': 'ae_final.pt',
                   'prior_sha256': 'prior_final.pt'}
    for key, filename in bound_files.items():
        if reference[key] != audit.sha(args.source_run/filename):
            raise ValueError('Frozen source file binding differs: '+filename)
    code = Path(__file__).resolve().parents[1]
    for relative in audit.SOURCES:
        if reference['code_sha256'].get(relative) != audit.sha(code/relative):
            raise ValueError('Reference audit evaluation source differs: '+relative)

    payload = torch.load(args.dataset, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('schema') != 'continuous_motion_dataset_v1':
        raise ValueError('Unexpected dataset schema')
    fit, validation = audit.split_train_pool(payload['clips'], source)
    del payload
    clips = audit.diverse_diagnostics(validation, 24)
    ids = [c['clip_id'] for c in clips]
    if len(ids) != 24 or ids != reference['selected_ids']:
        raise ValueError('Fixed 24-clip selection differs from receiver audit')
    stats = common._load(args.source_run/'fit_stats.pt')
    if set(stats['train_clip_ids']) != {c['clip_id'] for c in fit}:
        raise ValueError('Statistics fitting members differ')
    ae_ck = common._load(args.source_run/'ae_final.pt')
    prior_ck = common._load(args.source_run/'prior_final.pt')
    expected = common._value_sha(source)
    if any(c.get('binding', {}).get('protocol_sha256') != expected for c in (ae_ck, prior_ck)):
        raise ValueError('Source checkpoint protocol binding differs')
    if (prior_ck['binding'].get('ae_sha256') != common._value_sha(ae_ck['state']) or
            prior_ck['binding'].get('stats_sha256') != common._value_sha(stats)):
        raise ValueError('Prior AE/statistics binding differs')
    ae = ContinuousUpperAE(**ae_ck['config']).to(args.device)
    ae.load_state_dict(ae_ck['state']); ae.eval().requires_grad_(False)
    prior = ContinuousLatentFlow(**prior_ck['config']).to(args.device)
    prior.load_state_dict(prior_ck['state']); prior.eval().requires_grad_(False)
    clips = [{**c, '_audit_block_size': ae.block_size} for c in clips]
    reference_curves_path = args.reference_audit/'source_prior/curves.pt'
    reference_curves = common._load(reference_curves_path)['clips']
    audit.verify_curve_arm(reference_curves, clips, stochastic=True)
    reference_result = audit.read(args.reference_audit/'source_prior/result.json')
    if reference_result.get('use_audio') is not False or reference_result.get('steps') != 24:
        raise ValueError('Reference source prior must be no-audio Euler24')
    before = (common._value_sha(ae.state_dict()), common._value_sha(prior.state_dict()))
    protocol = {'schema': SCHEMA, 'posthoc': True, 'dataset_sha256': source['dataset_sha256'],
                'source_protocol_sha256': audit.sha(args.source_run/'protocol.json'),
                'reference_audit_protocol_sha256': audit.sha(args.reference_audit/'protocol.json'),
                'reference_source_prior_curves_sha256': audit.sha(reference_curves_path),
                **{key: reference[key] for key in bound_files},
                'selected_ids': ids, 'seeds': list(audit.SEEDS), 'use_audio': False,
                'arms': [{'name': name, 'solver': solver, 'steps': steps,
                          'nfe_per_draw_per_native_run': steps*(2 if solver == 'heun' else 1)}
                         for name, solver, steps in ARMS],
                'noise_policy': 'native._seed(clip_id, run_index, seed), identical native runs and shape in every arm',
                'heun_policy': 'explicit trapezoidal predictor-corrector; 2 velocity calls per step including t=1; no terminal shortcut',
                'euler24_reproduction_tolerance': {'atol': ATOL, 'rtol': RTOL},
                'generation_scope': 'same24 inner-development; observed motion only used after generation for scoring',
                'code_sha256': {p: audit.sha(code/p) for p in SOURCES},
                'outer_tensors_indexed': False, 'training_performed': False,
                'best_arm_selection': False, 'default_replaced': False,
                'interpretation': 'Post-hoc fixed-checkpoint sampling diagnosis; no test or naturalness certification; no model promotion.'}
    common._write(args.output/'protocol.json', protocol)
    for relative in SOURCES:
        destination = args.output/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((code/relative).read_bytes())
    table = []
    reproduction = None
    for name, solver, steps in ARMS:
        status('evaluating', arm=name)
        arm_started = time.monotonic()
        wrapper = SamplingWrapper(prior, solver).eval()
        result = native.evaluate_generation(wrapper, ae, clips, stats, args.output/name,
                    args.device, use_audio=False, seeds=audit.SEEDS, steps=steps)
        if not result['numerical_gate']['passed']:
            raise RuntimeError('Numerical/protection gate failed: '+name)
        curves = common._load(args.output/name/'curves.pt')['clips']
        comparison = verify_same_draws(curves, reference_curves, clips, compare_samples=(name == 'euler24'))
        common._write(args.output/name/'sampling_audit.json', {'arm': name, 'solver': solver,
                     'steps': steps, 'nfe_per_draw_per_native_run': steps*(2 if solver == 'heun' else 1),
                     'reference_check': comparison, 'default_replaced': False})
        if name == 'euler24':
            reproduction = comparison
        rescored, arkit = audit.rescore_arm(curves, clips, stats, args.output/name,
                    deterministic=False, scope=name+'; same24 post-hoc fixed-checkpoint sampling diagnosis')
        row = audit.table_row(name, rescored, arkit)
        row.update(solver=solver, steps=steps, nfe=steps*(2 if solver == 'heun' else 1),
                   seconds=time.monotonic()-arm_started, raw_signal_summary=signal_summary(rescored))
        table.append(row)
        if before != (common._value_sha(ae.state_dict()), common._value_sha(prior.state_dict())):
            raise RuntimeError('Frozen source state changed')
    for key, filename in bound_files.items():
        if reference[key] != audit.sha(args.source_run/filename):
            raise RuntimeError('Frozen source file changed during audit: '+filename)
    if protocol['reference_source_prior_curves_sha256'] != audit.sha(reference_curves_path):
        raise RuntimeError('Reference curves changed during audit')
    common._write(args.output/'summary.json', {'schema': SCHEMA, 'selected_ids': ids,
        'group_order': list(audit.GROUP_NAMES), 'table': table,
        'euler24_reproduction': reproduction, 'frozen_source_state_exact': True,
        'training_performed': False, 'best_arm_selection': False,
        'naturalness_certified': False, 'default_replaced': False})
    lines = ['# Frozen prior sampling resolution audit', '',
             'Post-hoc same24 inner-development diagnosis. No training, best-arm selection, or promotion.', '',
             '| Arm | NFE | Up/down/squint/wide RMS ratio | Raw OOB | Clamp RMS retention | MBE | LBE |',
             '| --- | ---: | --- | --- | --- | ---: | ---: |']
    def seq(value):
        return ' / '.join('pending' if x is None else f'{x:.4f}' for x in value) if value else 'pending'
    def metric(value):
        return 'pending' if value['value'] is None else f"{value['value']:.6f}"
    for row in table:
        lines.append(f"| {row['arm']} | {row['nfe']} | {seq(row['rms_ratio'])} | {seq(row['raw_oob'])} | {seq(row['clamp_rms_retention'])} | {metric(row['arkit']['arkit_mbe'])} | {metric(row['arkit']['arkit_lbe'])} |")
    lines += ['', 'All arms use the same frozen checkpoint, four seeds, native runs, and latent noise keys; use_audio=False.',
              'Euler24 must reproduce the saved source-prior curves before subsequent arms run.',
              'Heun includes the corrector at t=1: Heun24 uses 48 NFE; Heun48 uses 96 NFE per draw and native run.',
              'RMS is pooled within-run centered energy normalized with common fit scales; it is not temporal alignment.',
              'Raw mean/bias/OOB and coefficient-unit RMS are in each audit.json; joint fair ES and ARKit metrics are retained.',
              'Raw predictions are scored without clipping. Display-only [0,1] clipping is unchanged.',
              'Sampling differences diagnose integration sensitivity, not better training, audio timing, or naturalness.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    status('complete', seconds=time.monotonic()-started, evaluated_clips=len(clips),
           frozen_source_state_exact=True, naturalness_certified=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'source-run', 'reference-audit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.output.exists() and not isinstance(exc, FileExistsError):
            common._write(args.output/'status.json', {'schema': SCHEMA, 'state': 'failed',
                         'error': repr(exc), 'updated': time.time(), 'default_replaced': False})
        raise
