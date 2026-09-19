"""Read-only posthoc summary of event scheduling and mouth candidate reports.

Reads existing JSON reports only. It never reads target tensors, trains a
model, edits a run, or promotes checkpoints. Missing/in-progress artifacts are
explicitly pending; rerun into a fresh output directory to obtain a new snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = 'event_mouth_posthoc_summary_v1'
SEEDS = ('42', '123', '2026')
METRICS = ('arkit_mbe', 'arkit_lbe', 'arkit_fdd_absolute', 'supp_upper9_fdd_absolute')


def pending(reason):
    return {'status': 'pending', 'reason': reason}


def read_json(path, provenance):
    path = Path(path)
    if not path.is_file(): return None
    raw = path.read_bytes()
    provenance[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    try: return json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        provenance[str(path)]['read_error'] = str(exc)
        return None


def field_value(value):
    if isinstance(value, dict):
        return value.get('value') if value.get('status') in ('computed', 'partial') else None
    return value


def event_arm(root, provenance):
    result = read_json(root/'result.json', provenance)
    metrics = read_json(root/'arkit_benchmark.json', provenance)
    if result is None: return pending('result.json missing or incomplete')
    event_scores = result.get('shared_truth_event_scores', [])
    supported = [r for r in event_scores if r.get('brier') is not None]
    return {'status': 'computed', 'mode': result.get('mode'), 'scope': result.get('scope'),
            'clips': len(result.get('selected_clip_ids', [])),
            'event_brier_clip_equal': float(np.mean([r['brier'] for r in supported])) if supported else None,
            'event_scored_clips': len(supported), 'event_scores': event_scores,
            'numerical_gate': result.get('numerical_gate'),
            'motion_summary': result.get('summary'),
            'arkit': metrics.get('summary', {}) if metrics else pending('arkit_benchmark.json missing or incomplete'),
            'event_condition_diagnostics': result.get('event_condition_diagnostics', [])}


def paired_event_delta(first, second):
    if first.get('status') != 'computed' or second.get('status') != 'computed':
        return pending('Both event arms must have completed reports')
    a = {r['clip_id']: r for r in first['event_scores']}
    b = {r['clip_id']: r for r in second['event_scores']}
    if set(a) != set(b): return pending('Arm clip membership differs; no unpaired comparison calculated')
    groups = {}; omitted = []
    for cid in sorted(a):
        x, y = a[cid], b[cid]
        if x.get('target_known_positions') != y.get('target_known_positions') or x.get('sentence') != y.get('sentence'):
            return pending('Arm target support/metadata differs; no comparison calculated')
        if x.get('brier') is None or y.get('brier') is None:
            omitted.append(cid); continue
        groups.setdefault(str(x.get('sentence')), []).append(x['brier']-y['brier'])
    if not groups: return pending('No shared supported event scores')
    values = np.array([np.mean(groups[s]) for s in sorted(groups)], dtype=float)
    boot = np.random.default_rng(20260919).choice(values, (4096, len(values))).mean(1)
    return {'status': 'computed', 'delta': float(values.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'sentences': len(values), 'sentences_first_better': int((values < 0).sum()),
            'weighting': 'equal sentences, equal clips within sentence', 'omitted_clips': omitted,
            'ci_scope': 'Exploratory sentence bootstrap; development set, not sealed significance'}


def mouth_arm(result, root, provenance):
    arkit = (result or {}).get('arkit')
    diagnostics = (result or {}).get('diagnostics')
    if arkit is None:
        file = read_json(root/'arkit_benchmark.json', provenance)
        arkit = file.get('summary') if file else None
    if diagnostics is None:
        file = read_json(root/'mouth_diagnostics.json', provenance)
        diagnostics = file.get('summary') if file else None
    if arkit is None and diagnostics is None: return pending('Mouth evaluation not available')
    return {'status': 'computed' if arkit is not None and diagnostics is not None else 'partial',
            'arkit': arkit or pending('ARKit report missing'),
            'diagnostics': diagnostics or pending('Mouth diagnostic report missing')}


def collect(root):
    provenance = {}; event = root/'event'; mouth = root/'mouth'
    supervisor = read_json(root/'status.json', provenance)
    event_protocol = read_json(event/'protocol.json', provenance)
    mouth_protocol = read_json(mouth/'protocol.json', provenance)
    predictor = read_json(event/'predictor_results.json', provenance) or {}
    controls = {name: event_arm(event/'control'/name, provenance) for name in ('oracle', 'empty', 'shifted', 'null')}
    generated = {name: event_arm(event/'generation'/name, provenance) for name in ('audio', 'matched_static', 'reverse')}
    mouth_results = read_json(mouth/'results.json', provenance) or {}
    mouth_seeds = {}
    for seed in SEEDS:
        values = mouth_results.get('seeds', {}).get(seed, {})
        mouth_seeds[seed] = {}
        for name in ('audio', 'matched_static', 'audio_reverse'):
            path = mouth/f'seed_{seed}'/('audio' if name == 'audio_reverse' else name)
            path /= 'validation_reverse' if name == 'audio_reverse' else 'validation'
            mouth_seeds[seed][name] = mouth_arm(values.get(name), path, provenance)
    return {'schema': SCHEMA, 'created_utc_epoch': time.time(), 'run_root': str(root),
            'snapshot_only': True, 'default_replaced': False,
            'supervisor': supervisor or pending('Supervisor status unavailable'),
            'scope': 'Inner development; inherited upstream exposure. No external baseline ranking or sealed test.',
            'event': {'status': read_json(event/'status.json', provenance) or pending('Event status unavailable'),
                      'protocol': event_protocol or pending('Event protocol unavailable'),
                      'decision': read_json(event/'decision.json', provenance) or pending('Event decision not yet available'),
                      'receiver_gate': read_json(event/'receiver_gate.json', provenance) or pending('Receiver gate not yet available'),
                      'teacher_coverage': read_json(event/'teacher_coverage.json', provenance) or pending('Teacher coverage unavailable'),
                      'receiver_coverage': read_json(event/'receiver_coverage.json', provenance) or pending('Receiver coverage unavailable'),
                      'duration_quantization': read_json(event/'duration_quantization.json', provenance) or pending('Duration coverage unavailable'),
                      'predictor_seeds': {s: ({'status': 'computed', **predictor[s]} if s in predictor else pending('Seed not fully evaluated')) for s in SEEDS},
                      'controls': controls, 'generation': generated,
                      'comparisons': {'oracle_minus_null': paired_event_delta(controls['oracle'], controls['null']),
                                      'oracle_minus_empty': paired_event_delta(controls['oracle'], controls['empty']),
                                      'audio_minus_matched_static': paired_event_delta(generated['audio'], generated['matched_static']),
                                      'audio_minus_reverse': paired_event_delta(generated['audio'], generated['reverse']),
                                      'audio_minus_null_receiver': paired_event_delta(generated['audio'], controls['null'])},
                      'null_comparison_caution': 'Null is a separately trained receiver with an empty schedule. Audio/static/reverse use the same event-conditioned receiver.'},
            'mouth': {'status': read_json(mouth/'status.json', provenance) or pending('Mouth status unavailable'),
                      'protocol': mouth_protocol or pending('Mouth protocol unavailable'),
                      'baseline': mouth_arm(mouth_results.get('baseline'), mouth/'baseline', provenance),
                      'seeds': mouth_seeds}, 'input_files': provenance}


def fmt(value):
    value = field_value(value)
    if value is None: return 'pending'
    if isinstance(value, (float, int)): return f'{value:.6f}'
    return str(value)


def report(summary):
    event, mouth = summary['event'], summary['mouth']
    lines = ['# Event scheduling and mouth candidate snapshot', '',
             'This report summarizes saved JSON only. Pending means unfinished or unavailable; it is not zero.',
             'All results are internal development diagnostics. No checkpoint is promoted and no external-method ranking is supported.', '',
             '## Event predictor: all predefined training seeds', '',
             '| Seed | Arm | Onset Brier | Onset NLL | Joint NLL | Duration NLL per event |',
             '|---|---|---:|---:|---:|---:|']
    for seed, row in event['predictor_seeds'].items():
        for arm in ('audio', 'matched_static', 'reverse', 'own_static'):
            values = row.get('summary', {}).get(arm, {})
            lines.append('| '+seed+' | '+arm+' | '+' | '.join(fmt(values.get(k)) for k in ('brier', 'onset_nll', 'joint_nll', 'duration_nll_per_event'))+' |')
    lines += ['', 'Saved paired predictor comparisons (all available statistics, including confidence intervals):', '```json',
              json.dumps({s: {k: v for k, v in r.items() if k not in ('summary',)} for s, r in event['predictor_seeds'].items()}, indent=2), '```', '',
              '## Receiver controls and audio generation', '',
              '| Arm | Shared-target event Brier | ARKit MBE | ARKit LBE | Absolute FDD | Upper9 absolute FDD |',
              '|---|---:|---:|---:|---:|---:|']
    for name, row in list(event['controls'].items())+list(event['generation'].items()):
        lines.append('| '+name+' | '+fmt(row.get('event_brier_clip_equal'))+' | '+
                     ' | '.join(fmt(row.get('arkit', {}).get(k)) for k in METRICS)+' |')
    lines += ['', 'Brier table uses equal clips; paired comparisons below use equal sentences. Negative deltas favor the first arm.',
              '```json', json.dumps(event['comparisons'], indent=2), '```', '',
              'Receiver gate:', '```json', json.dumps(event['receiver_gate'], indent=2), '```', '',
              'Oracle success establishes that a real-motion-derived schedule can control the receiver. It does not establish that audio can predict that schedule.',
              'A small motion response to shifting or removing a condition proves sensitivity, not correct event timing or naturalness. The oracle-vs-empty interval must remain visible even if the engineering gate passed.',
              'Null is a separately trained empty-condition receiver; audio and matched-static generation share the event receiver. FDD measures amplitude statistics, not chronological alignment.', '',
              '## Mouth candidate: all predefined training seeds', '',
              '| Seed | Arm | MBE | LBE | Lip23 velocity MAE/s | Lip23 std pred | Lip23 std GT | Lip23 absolute std gap |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    rows = [('baseline', 'baseline', mouth['baseline'])]
    rows += [(s, a, r) for s, arms in mouth['seeds'].items() for a, r in arms.items()]
    for seed, arm, row in rows:
        diag = row.get('diagnostics', {}); arkit = row.get('arkit', {})
        values = [arkit.get('arkit_mbe'), arkit.get('arkit_lbe')]+[diag.get(k) for k in
            ('lip23_velocity_mae_per_second', 'lip23_temporal_std_pred', 'lip23_temporal_std_gt', 'lip23_temporal_std_absolute_gap')]
        lines.append('| '+seed+' | '+arm+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines += ['', 'Full lip23/mouth27 MAE, velocity, temporal standard deviation and raw out-of-range diagnostics are preserved in summary.json.',
              'Training seeds are independent fitted models, not stochastic motion draws. Reduced LBE alone does not certify lip synchronization, identity, or emotion quality.', '',
              '## Current completion and decision', '', '```json',
              json.dumps({'event_status': event['status'], 'event_decision': event['decision'], 'mouth_status': mouth['status']}, indent=2), '```', '',
              'Shared AV offset/confidence, Multimodality, FD/WInD remain pending wherever the stored evaluator report marks them pending. No substitute score is fabricated.', '']
    return '\n'.join(lines)


def run(args):
    root = args.run_root.resolve(); output = args.output.resolve()
    if not root.is_dir(): raise NotADirectoryError(root)
    if output.exists(): raise FileExistsError('Fresh summary output directory required')
    if output == root: raise ValueError('Summary output must not replace run root')
    data = collect(root)
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    (output/'report.md').write_text(report(data), encoding='utf8')
    return data


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    return p


if __name__ == '__main__': run(parser().parse_args())
