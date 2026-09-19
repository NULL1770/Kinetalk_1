"""Fixed-budget null-receiver control restoring source-prior training support.

Only the supervised-support policy changes: all jointly observed runs, max200
and min5, replace all-four teacher-known runs with min10. No audio predictor or
event-conditioned candidate is trained or promoted. This is inner development.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousUpperAE, ContinuousLatentFlow
from kinetalk_b0.models.event_conditioned_flow import EventConditionedFlow
from scripts import audit_event_receiver_bottleneck as audit
from scripts import audit_event_training_support as support
from scripts import train_event_schedule_pipeline as event

common = audit.common
SCHEMA = 'event_receiver_support_repair_v1'
SEED = 2026091909
BUDGET = 2000
BATCH_SIZE = 24
STAGE = 'receiver_null_full_support'
EXPECTED_FIT_CLIPS = 613
EXPECTED_FIT_FRAMES = 66272
EXPECTED_VALID_CLIPS = 206
EXPECTED_OLD_CLIPS = 348
EXPECTED_OLD_FRAMES = 14898
SOURCES = tuple(dict.fromkeys(('scripts/repair_event_receiver_support.py',
    'scripts/audit_event_training_support.py', 'scripts/audit_event_validation_scope.py',
    *event.SOURCES, *audit.SOURCES)))


def prepare_full_null_segments(clips, stats, ae, device):
    """Source segmentation/encoding exactly, followed by 12D zero conditions.

    No event teacher, label or known mask is accepted by this function.
    """
    if len({c['clip_id'] for c in clips}) != len(clips) or any(c['split'] != 'train' for c in clips):
        raise ValueError('Unique fit-only training clips required')
    for clip in clips:
        valid, observed = audit.array(clip['valid']), audit.array(clip['motion_mask'])
        if valid.dtype != bool or observed.dtype != bool or observed.shape != (len(valid), 9):
            raise ValueError('Boolean native and motion masks required')
    segments, coverage = common.prepare_segments(clips, stats, 'train', max_frames=200)
    if not segments:
        raise ValueError('No source-supported training segments')
    encoded = common._encode_segments(ae, segments, device)
    items = [{**item, 'audio': torch.zeros(len(item['audio']), 12, dtype=torch.float32)}
             for item in encoded]
    expected_intervals = []
    for clip in clips:
        joint = audit.array(clip['valid']) & audit.array(clip['motion_mask']).all(1)
        _, spans, _ = support.chunk_support(joint, 5, 200)
        expected_intervals.extend((clip['clip_id'], start, end) for start, end in spans)
    actual_intervals = [(x['metadata']['clip_id'], x['metadata']['start'], x['metadata']['end']) for x in items]
    if actual_intervals != expected_intervals:
        raise RuntimeError('Restored support differs from source segmentation')
    for item in items:
        n = item['metadata']['end']-item['metadata']['start']
        if (item['audio'].shape != (n, 12) or torch.count_nonzero(item['audio']).item() or
                item['z'].shape != ((n+ae.block_size-1)//ae.block_size, ae.latent_dim)):
            raise RuntimeError('Null condition or encoded clock differs')
    coverage.update(retained_clip_ids=sorted({x['metadata']['clip_id'] for x in items}),
                    segment_intervals=[list(x) for x in actual_intervals], condition_width=12,
                    conditions_all_zero=True, teacher_read_for_training=False)
    return items, coverage


def verify_production_coverage(coverage, fit, validation):
    expected_ids = {c['clip_id'] for c in fit}
    if len(fit) != EXPECTED_FIT_CLIPS or len(validation) != EXPECTED_VALID_CLIPS:
        raise ValueError('Expected exact 613 fit / 206 inner-validation population')
    if (set(coverage['retained_clip_ids']) != expected_ids or
            coverage['totals']['kept_frames'] != EXPECTED_FIT_FRAMES):
        raise ValueError('Full support must retain all 613 fit clips and 66272 frames')
    if expected_ids & {c['clip_id'] for c in validation}:
        raise ValueError('Fit and validation overlap')
    if {c['sentence'] for c in fit} & {c['sentence'] for c in validation}:
        raise ValueError('Fit and validation sentences overlap')


def verify_reference_row(row, expected):
    """Numerical identity with the already published same-support audit table."""
    def check(value, original, key):
        if isinstance(original, dict):
            if not isinstance(value, dict) or set(value) != set(original):
                raise ValueError('Reference table keys differ: '+key)
            for name in original:
                check(value[name], original[name], key+'/'+name)
        elif isinstance(original, list):
            if not isinstance(value, (tuple, list)) or len(value) != len(original):
                raise ValueError('Reference table shape differs: '+key)
            for index, (a, b) in enumerate(zip(value, original)):
                check(a, b, key+'/'+str(index))
        elif isinstance(original, (float, int)) and not isinstance(original, bool):
            if not np.isclose(value, original, atol=1e-10, rtol=1e-10):
                raise ValueError('Reference table numerical value differs: '+key)
        elif value != original:
            raise ValueError('Reference table value differs: '+key)
    check(row, expected, row['arm'])


def preserved_hashes(args):
    paths = {'dataset': args.dataset, 'source_protocol': args.source_run/'protocol.json',
             'ae': args.source_run/'ae_final.pt', 'stats': args.source_run/'fit_stats.pt',
             'prior': args.source_run/'prior_final.pt', 'event_protocol': args.event_run/'protocol.json',
             'old_null_checkpoint': args.event_run/'receiver_null_final.pt',
             'old_receiver_coverage': args.event_run/'receiver_coverage.json',
             'teacher': args.event_run/'teacher.json',
             'reference_protocol': args.reference_audit/'protocol.json',
             'reference_table': args.reference_audit/'summary.json',
             'source_prior_curves': args.source_run/'prior_generation/inner_validation/curves.pt',
             'old_null_curves': args.reference_audit/'event_null/curves.pt'}
    return {name: audit.sha(path) for name, path in paths.items()}


def run(args):
    if args.output.exists():
        raise FileExistsError('Fresh support-repair output directory required')
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True)
    device = torch.device(args.device)

    def status(state, **details):
        common._write(args.output/'status.json', {'schema': SCHEMA, 'state': state,
                     'updated': time.time(), 'default_replaced': False, **details})

    status('binding_inputs')
    before_files = preserved_hashes(args)
    source = audit.read(args.source_run/'protocol.json')
    previous = audit.read(args.event_run/'protocol.json')
    reference = audit.read(args.reference_audit/'protocol.json')
    if previous.get('schema') != event.SCHEMA or previous.get('smoke'):
        raise ValueError('Formal event run required')
    if reference.get('schema') != 'event_validation_scope_audit_v1':
        raise ValueError('Full inner-validation audit required')
    if audit.read(args.reference_audit/'status.json').get('state') != 'complete':
        raise ValueError('Reference all206 audit not complete')
    if (source['dataset_sha256'] != before_files['dataset'] or previous['dataset_sha256'] != before_files['dataset'] or
            reference['dataset_sha256'] != before_files['dataset']):
        raise ValueError('Dataset hash binding mismatch')
    if (previous['source_protocol_sha256'] != before_files['source_protocol'] or
            reference['source_protocol_sha256'] != before_files['source_protocol'] or
            reference['event_protocol_sha256'] != before_files['event_protocol']):
        raise ValueError('Source/event protocol hash binding mismatch')
    if (reference['receiver_sha256']['null'] != before_files['old_null_checkpoint'] or
            reference['source_curves_sha256']['source_prior'] != before_files['source_prior_curves']):
        raise ValueError('Reference checkpoint/curve hash binding mismatch')
    if reference['seeds'] != list(audit.SEEDS) or reference['steps'] != 24:
        raise ValueError('Reference sampling protocol differs')
    for key, name in [('ae_sha256', 'ae'), ('prior_sha256', 'prior'), ('fit_stats_sha256', 'stats')]:
        if previous[key] != before_files[name]:
            raise ValueError('Frozen source binding mismatch: '+name)
    code = Path(__file__).resolve().parents[1]
    for relative in event.SOURCES:
        value = audit.sha(code/relative)
        if previous['source_sha256'][relative] != value or reference['code_sha256'][relative] != value:
            raise ValueError('Historical executed source differs: '+relative)
    segment_code = 'scripts/prepare_continuous_motion_dataset.py'
    if source['source_sha256'][segment_code] != audit.sha(code/segment_code):
        raise ValueError('Source prior segmentation code differs')
    payload = torch.load(args.dataset, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('schema') != 'continuous_motion_dataset_v1':
        raise ValueError('Unexpected dataset schema')
    fit, validation = audit.split_train_pool(payload['clips'], source)
    del payload
    fit_ids = [c['clip_id'] for c in fit]
    valid_ids = [c['clip_id'] for c in validation]
    if (previous['train_ids'] != fit_ids or previous['validation_ids'] != valid_ids or
            reference['selected_ids'] != valid_ids):
        raise ValueError('Exact source/event/all206 split binding mismatch')
    if len(fit) != EXPECTED_FIT_CLIPS or len(validation) != EXPECTED_VALID_CLIPS:
        raise ValueError('Expected formal 613/206 split')
    stats = common._load(args.source_run/'fit_stats.pt')
    ae_ck = common._load(args.source_run/'ae_final.pt')
    prior_ck = common._load(args.source_run/'prior_final.pt')
    old_null = common._load(args.event_run/'receiver_null_final.pt')
    if set(stats['train_clip_ids']) != set(fit_ids):
        raise ValueError('Frozen statistics fitting members differ')
    source_content_hash = common._value_sha(source)
    if any(ck['binding']['protocol_sha256'] != source_content_hash for ck in (ae_ck, prior_ck)):
        raise ValueError('Source checkpoint protocol content binding differs')
    if (prior_ck['binding'].get('ae_sha256') != common._value_sha(ae_ck['state']) or
            prior_ck['binding'].get('stats_sha256') != common._value_sha(stats)):
        raise ValueError('Prior AE/statistics content binding differs')
    if old_null['binding'] != {'protocol_sha256': before_files['event_protocol'], 'teacher_sha256': before_files['teacher']}:
        raise ValueError('Historical null receiver binding differs')
    if (previous['receiver_seed'] != SEED or previous['receiver_steps_per_arm'] != BUDGET or
            old_null['budget'] != BUDGET or old_null['step'] != BUDGET or
            old_null['stage'] != 'receiver_null' or old_null['use_audio'] is not True):
        raise ValueError('Historical null receiver training contract differs')
    if common._value_sha(old_null['stats']) != common._value_sha(stats):
        raise ValueError('Historical null receiver used different statistics')
    for group in old_null['optimizer']['param_groups']:
        if group['lr'] != 3e-4 or group['weight_decay'] != 1e-4:
            raise ValueError('Historical null receiver optimizer differs')
    teacher = audit.read(args.event_run/'teacher.json')
    ae = ContinuousUpperAE(**ae_ck['config']).to(device)
    ae.load_state_dict(ae_ck['state']); ae.eval().requires_grad_(False)
    prior = ContinuousLatentFlow(**prior_ck['config']).to(device)
    prior.load_state_dict(prior_ck['state']); prior.eval().requires_grad_(False)
    if ae.block_size != 5 or prior.block_size != 5:
        raise ValueError('Original event receiver evaluator requires five-frame blocks')
    frozen_before = (common._value_sha(ae.state_dict()), common._value_sha(prior.state_dict()), common._value_sha(stats))
    status('preparing_full_support')
    items, coverage = prepare_full_null_segments(fit, stats, ae, device)
    verify_production_coverage(coverage, fit, validation)
    old_coverage = audit.read(args.event_run/'receiver_coverage.json')
    old_counts = {'clips': 0, 'frames': 0, 'segments': 0}
    for clip in fit:
        label = event.schedule_for(clip, teacher)
        mask = audit.array(clip['valid']) & audit.array(clip['motion_mask']).all(1) & label['known'].all(1)
        kept, spans, _ = support.chunk_support(mask, 10, 200)
        old_counts['clips'] += bool(spans)
        old_counts['frames'] += int(kept.sum())
        old_counts['segments'] += len(spans)
    if any(old_counts[k] != old_coverage[k] for k in old_counts):
        raise ValueError('Historical support coverage cannot be reproduced')
    if old_counts['clips'] != EXPECTED_OLD_CLIPS or old_counts['frames'] != EXPECTED_OLD_FRAMES:
        raise ValueError('Unexpected formal historical null support')
    common._write(args.output/'receiver_coverage.json', {'restored': coverage, 'historical': old_counts})
    # Exact old initialized receiver; never start from the adapted old null.
    torch.manual_seed(SEED)
    model = EventConditionedFlow.from_prior(prior)
    initial_sha = common._value_sha(model.state_dict())
    if initial_sha != old_null['initial_state_sha256'] or model.config != old_null['config']:
        raise ValueError('Null receiver initialization/config does not reproduce old control')
    reference_table = {row['arm']: row for row in audit.read(args.reference_audit/'summary.json')['table']}
    old_curves = {
        'source_prior': common._load(args.source_run/'prior_generation/inner_validation/curves.pt')['clips'],
        'event_null': common._load(args.reference_audit/'event_null/curves.pt')['clips'],
    }
    table = []
    for name, curves in old_curves.items():
        status('checking_reference_scores', arm=name)
        scored, report = audit.rescore_arm(curves, validation, stats, args.output/name,
                            deterministic=False, scope=name+'; all206 support repair control')
        row = audit.table_row(name, scored, report)
        verify_reference_row(row, reference_table[name])
        table.append(row)
    protocol = {'schema': SCHEMA, 'posthoc_development': True, 'input_sha256': before_files,
                'train_ids': fit_ids, 'validation_ids': valid_ids,
                'initial_state_sha256': initial_sha, 'initialization_matches_old_null': True,
                'model_config': model.config, 'receiver_seed': SEED, 'receiver_steps': BUDGET,
                'batch_size': BATCH_SIZE, 'optimizer': {'name': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4},
                'gradient_clip_norm': 1., 'use_audio': True, 'condition': 'native-frame zero12; no audio predictor and no event values',
                'supervised_support_changed': {'old': 'native valid & joint9 & known4; nonoverlap max200/min10',
                                               'new': 'native valid & joint9; source nonoverlap max200/min5'},
                'causal_scope': 'Effect of restoring complete source support policy, including short-segment retention; cannot isolate known4 alone.',
                'sampling': 'uniform retained clip then uniform segment; same seed, not paired training draws because support and batch shapes differ',
                'loss': 'unchanged latent flow matching; equal valid latent-token coordinates',
                'coverage_sha256': audit.sha(args.output/'receiver_coverage.json'),
                'encoded_items_sha256': common._value_sha(items),
                'evaluation_seeds': list(audit.SEEDS), 'evaluation_steps': 24,
                'evaluation_solver': 'original Euler; no Heun intervention',
                'evaluation_population': 'all206 inner-development, inherited upstream exposure; not sealed test',
                'teacher_usage': 'historical coverage reproduction and evaluation only; no teacher filtering or target schedule in training',
                'reference_table_reproduced': True, 'checkpoint_selection': 'fixed final update2000; no best selection',
                'frozen': ['AE weights', 'source prior weights', 'fit statistics', 'identity/global-emotion feature extractor outputs', 'baseline52'],
                'source_sha256': {p: audit.sha(code/p) for p in SOURCES},
                'audio_predictor_trained': False, 'outer_targets_indexed': False, 'default_replaced': False}
    common._write(args.output/'protocol.json', protocol)
    for relative in SOURCES:
        destination = args.output/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((code/relative).read_bytes())
    binding = {'protocol_sha256': audit.sha(args.output/'protocol.json'),
               'source_prior_sha256': before_files['prior'], 'ae_sha256': before_files['ae'],
               'stats_sha256': before_files['stats'], 'teacher_sha256': before_files['teacher']}
    status('training_receiver', stage=STAGE, steps=BUDGET)
    checkpoint = common._train_stage(model, items, stats, stage=STAGE, budget=BUDGET,
        batch_size=BATCH_SIZE, device=device, output=args.output, binding=binding, seed=SEED, use_audio=True)
    if checkpoint['step'] != BUDGET or checkpoint['initial_state_sha256'] != initial_sha:
        raise RuntimeError('Training budget or initialization changed')
    if frozen_before != (common._value_sha(ae.state_dict()), common._value_sha(prior.state_dict()), common._value_sha(stats)):
        raise RuntimeError('Frozen AE/source/statistics changed')
    model.eval().requires_grad_(False)
    trained_state = common._value_sha(model.state_dict())
    empty = {c['clip_id']: [np.zeros((len(c['valid']), 12), np.float32)] for c in validation}
    status('evaluating', arm=STAGE, clips=len(validation))
    result = event.evaluate_receiver(model, ae, validation, stats, empty, args.output/STAGE,
                                     device, STAGE, teacher)
    if not result['numerical_gate']['passed']:
        raise RuntimeError('Generated numerical/protection checks failed')
    curves = common._load(args.output/STAGE/'curves.pt')['clips']
    scored, report = audit.rescore_arm(curves, validation, stats, args.output/STAGE,
                        deterministic=False, scope=STAGE+'; all206 support repair control')
    table.append(audit.table_row(STAGE, scored, report))
    if trained_state != common._value_sha(model.state_dict()):
        raise RuntimeError('Final receiver changed during evaluation')
    if before_files != preserved_hashes(args):
        raise RuntimeError('Bound input files changed during experiment')
    if frozen_before != (common._value_sha(ae.state_dict()), common._value_sha(prior.state_dict()), common._value_sha(stats)):
        raise RuntimeError('Frozen AE/source/statistics changed during evaluation')
    decision = {'schema': SCHEMA, 'default_replaced': False, 'promoted': False,
                'audio_predictor_trained': False, 'audio_timing_success_claimed': False,
                'naturalness_certified': False, 'reference_table_reproduced': True,
                'restored_support_verified': True, 'frozen_source_state_exact': True,
                'interpretation': 'Fixed-seed null-receiver support-policy control only. Numerical changes require distribution/range and visual review; no generalization or audio-timing certification.'}
    common._write(args.output/'decision.json', decision)
    common._write(args.output/'summary.json', {'schema': SCHEMA, 'clips': len(validation),
        'group_order': list(audit.GROUP_NAMES), 'table': table, 'decision': decision,
        'scope': 'all206 inner-development; support-policy restoration control'})
    lines = ['# Null receiver source-support restoration', '',
             'Fixed-budget development control. Only the support policy is restored; no audio predictor is trained or promoted.', '',
             f'Historical: {old_counts["clips"]} clips / {old_counts["frames"]} frames. Restored: {len(coverage["retained_clip_ids"])} clips / {coverage["totals"]["kept_frames"]} frames.', '',
             '| Arm | Up/down/squint/wide RMS | Raw OOB | Clamp RMS retention | MBE | LBE |',
             '| --- | --- | --- | --- | ---: | ---: |']
    def seq(value):
        return ' / '.join('pending' if x is None else f'{x:.4f}' for x in value) if value else 'pending'
    def metric(value):
        return 'pending' if value['value'] is None else f"{value['value']:.6f}"
    for row in table:
        lines.append(f"| {row['arm']} | {seq(row['rms_ratio'])} | {seq(row['raw_oob'])} | {seq(row['clamp_rms_retention'])} | {metric(row['arkit']['arkit_mbe'])} | {metric(row['arkit']['arkit_lbe'])} |")
    lines += ['', 'Identical prior-derived receiver initialization, seed, AdamW settings, batch size24 and 2000 updates; original Euler24 and four evaluation seeds.',
              'The support-policy intervention includes removing all-four-known filtering AND restoring minimum length5 instead of10. Effects cannot be attributed to the known mask alone.',
              'Training draws are not paired: altered support changes eligible clips, segment lengths and noise/time tensor shapes.',
              'Global emotion/identity and audio-derived baseline contexts remain; the local12D condition is always zero.',
              'AE/statistics/source-prior weights and old files remain unchanged. New receiver parameters alone are trained.',
              'All206 source-prior and old-null metrics reproduce the previous audit from original saved curves. Raw predictions are scored without clipping.',
              'Motion amplitude, range and fair ES require joint interpretation and visual review; this control cannot demonstrate audio timing.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    status('complete', seconds=time.monotonic()-started, clips=len(validation),
           receiver_updates=BUDGET, naturalness_certified=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'source-run', 'event-run', 'reference-audit', 'output'):
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
