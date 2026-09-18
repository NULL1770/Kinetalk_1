"""Audit saved process-prior results without training or neural inference.

Checks exact source-code/protocol bindings, split membership, matched epoch/order
records, teacher decoding, and all stored heldout draws. Event likelihoods can
only be checked for bookkeeping/aggregation: per-event model outputs were not
saved. The reconstructed normalized draws contain a documented FP32 raw-storage
round trip. This is a four-group pilot, not a 52-channel generator audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_motion_process as p
from scripts.full_native_context_data import sha
from scripts.motion_process_representation import render_motion_process

SCHEMA = 'saved_motion_process_audit_v1'
ARMS = ('global_only', 'local_audio')
HOLDS = ('sentence', 'speaker', 'joint')
CODE_FILES = ('scripts/train_motion_process.py', 'scripts/motion_process_representation.py',
              'kinetalk_b0/models/motion_process_prior.py')
METRIC_ATOL = 2e-5
METRIC_RTOL = 2e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def compare(expected, observed, path='root', *, atol=1e-10, rtol=1e-8):
    """Strict structural comparison, with explicit finite numeric tolerance."""
    if isinstance(expected, dict):
        require(isinstance(observed, dict) and set(expected) == set(observed), 'Keys differ: '+path)
        for key in expected:
            compare(expected[key], observed[key], path+'/'+key, atol=atol, rtol=rtol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(observed, (list, tuple)) and len(expected) == len(observed), 'Length differs: '+path)
        for i, (left, right) in enumerate(zip(expected, observed)):
            compare(left, right, path+'/'+str(i), atol=atol, rtol=rtol)
    elif isinstance(expected, (float, int, np.floating, np.integer)) and not isinstance(expected, bool):
        require(isinstance(observed, (float, int, np.floating, np.integer))
                and not isinstance(observed, bool) and np.isfinite(expected) and np.isfinite(observed)
                and np.isclose(expected, observed, atol=atol, rtol=rtol), 'Value differs: '+path)
    else:
        require(expected == observed, 'Value differs: '+path)


def validate_split(protocol, normalization, plan_ids):
    split = protocol['split']
    require(set(split) == set(p.CELLS), 'Four declared split cells required')
    ids = [row['clip_id'] for cell in p.CELLS for row in split[cell]]
    require(len(ids) == len(set(ids)) and set(ids) == set(plan_ids) and len(plan_ids) == len(set(plan_ids)),
            'Duplicate or incomplete split/plan membership')
    fit = [row['clip_id'] for row in split['fit']]
    require(normalization['fit_clip_ids'] == fit, 'Normalization fit IDs/order differ from split.fit')
    table = {row['clip_id']: row for cell in p.CELLS for row in split[cell]}
    if not protocol['smoke']:
        reconstructed = p.fixed_split([table[cid]['speaker'] for cid in plan_ids],
                                      [table[cid]['sentence'] for cid in plan_ids])
        for cell, indices in reconstructed.items():
            require([plan_ids[i] for i in indices] == [row['clip_id'] for row in split[cell]],
                    'Metadata hash split/order differs: '+cell)
        require({key: len(value) for key, value in split.items()} ==
                {'fit': 1605, 'sentence': 275, 'speaker': 375, 'joint': 60}, 'Formal cell counts differ')
    fit_speaker = {r['speaker'] for r in split['fit']}
    fit_sentence = {r['sentence'] for r in split['fit']}
    for cell in HOLDS:
        for row in split[cell]:
            require((row['speaker'] in fit_speaker) == (cell == 'sentence') and
                    (row['sentence'] in fit_sentence) == (cell == 'speaker'), 'Split isolation differs: '+cell)
    order = {cid: i for i, cid in enumerate(plan_ids)}
    return [order[cid] for cid in fit]


def expected_order(fit_indices, epochs):
    rng = np.random.default_rng(20260918)
    digest = hashlib.sha256()
    for _ in range(epochs):
        digest.update(np.asarray(rng.permutation(fit_indices), dtype=np.int64).tobytes())
    return digest.hexdigest()


def validate_budget(losses, matched, fit_indices, epochs):
    updates = (len(fit_indices) + 23)//24
    require(len(losses) == epochs, 'Incomplete epoch history')
    for epoch, row in enumerate(losses, 1):
        require(row['epoch'] == epoch and row['updates'] == updates and np.isfinite(row['loss'])
                and np.isfinite(row['seconds']) and row['seconds'] >= 0, 'Invalid epoch/update/loss record')
    expected = {'order_sha256': expected_order(fit_indices, epochs), 'updates': epochs*updates}
    compare(expected, matched, 'matched_budget', atol=0, rtol=0)
    return expected


def validate_draws(saved, scales, plan, where):
    required = {'samples', 'target', 'teacher', 'valid', 'anchor'}
    require(set(saved) == required, 'Draw payload differs: '+where)
    x, y, teacher, valid, anchor = [np.asarray(saved[key]) for key in
        ('samples', 'target', 'teacher', 'valid', 'anchor')]
    require(valid.dtype == np.bool_ and valid.ndim == 1 and valid.any(), 'Invalid draw mask: '+where)
    require(y.shape == teacher.shape == (len(valid), 4) and anchor.shape == (4,)
            and x.shape == (len(p.SEEDS), len(valid), 4) and x.dtype == np.float32,
            'Draw shape/seed count/storage differs: '+where)
    require(np.isfinite(x).all() and np.isfinite(y[valid]).all() and np.isfinite(teacher).all()
            and np.isfinite(anchor).all(), 'Nonfinite draw metadata: '+where)
    require(np.array_equal(plan['mask'], np.repeat(valid[:, None], 4, axis=1))
            and np.array_equal(np.asarray(plan['scales']), scales), 'Plan mask/scale mismatch: '+where)
    require(np.array_equal(render_motion_process(plan), teacher), 'Saved teacher decode mismatch: '+where)
    if (~valid).any():
        expected = np.broadcast_to(anchor.astype(np.float32), x[:, ~valid].shape)
        require(np.array_equal(x[:, ~valid], expected), 'Invalid raw draws should equal enrollment anchor: '+where)
    return x, y, teacher, valid, anchor


def rescore_draw(saved, scales, row, where):
    x, y, valid, anchor = [np.asarray(saved[key]) for key in ('samples', 'target', 'valid', 'anchor')]
    # Original evaluation scored normalized FP32 draws, then stored raw FP32.
    # Restoring from raw needs this inverse transform and the bounded tolerance.
    normalized = (x.astype(np.float64)-anchor)/scales
    target = (y-anchor)/scales
    metrics = p.distribution_metrics(normalized, target, valid)
    compare(metrics, {key: row[key] for key in metrics}, where, atol=METRIC_ATOL, rtol=METRIC_RTOL)
    max_error = max(float(np.max(np.abs(np.asarray(metrics[key])-np.asarray(row[key])))) for key in metrics)
    raw = x[:, valid]
    out_of_domain = ((raw < 0)|(raw > 1)).mean((0, 1))
    # A pre-save value just outside the domain can round onto exactly 0 or 1.
    # No other classification discrepancy is excused by raw FP32 storage.
    ambiguous = ((raw == 0)|(raw == 1)).mean((0, 1))
    recorded = np.asarray(row['out_of_domain'])
    require(recorded.shape == (4,) and np.isfinite(recorded).all()
            and np.all(recorded >= out_of_domain-1e-12)
            and np.all(recorded <= out_of_domain+ambiguous+1e-12), 'Out-of-domain mismatch: '+where)
    return {**row, **metrics, 'out_of_domain': out_of_domain.tolist()}, max_error, ambiguous.tolist()


def validate_aggregates(report, where):
    rows = report['rows']
    compare(p.aggregate(rows), report['summary'], where+'/summary')
    for field in ('speaker', 'emotion'):
        expected = {str(group): p.aggregate([r for r in rows if r[field] == group])
                    for group in sorted({r[field] for r in rows})}
        compare(expected, report['by_'+field], where+'/by_'+field)


def representation_gate(cells):
    return all(row['segments_per_second'] <= 12 and all(
        g['centered_r2'] >= .5 and g['correlation'] >= .8 and .75 <= g['rms_ratio'] <= 1.25
        for g in row['groups'].values()) for cell in HOLDS for row in [cells[cell]['reconstructed']])


def audit_run(run, code_root):
    run, code_root = Path(run).resolve(), Path(code_root).resolve()
    protocol = read(run/'protocol.json'); status = read(run/'status.json')
    require(protocol['schema'] == p.SCHEMA and protocol['epochs'] == 30 and protocol['smoke'] is False
            and protocol['test_loaded'] is False and protocol['dev_indexed'] is False, 'Formal protocol scope differs')
    require(status['status'] == 'complete' and status['smoke'] is False and status['generator_started'] is False
            and status['default_replaced'] is False and status['test_loaded'] is False, 'Completion/scope differs')
    require(sha(code_root/'docs/MOTION_PROCESS_PROTOCOL_20260918.md') == protocol['protocol_sha256'],
            'Protocol document SHA mismatch')
    require(set(protocol['code_sha256']) == set(CODE_FILES), 'Training code inventory differs')
    for name, digest in protocol['code_sha256'].items():
        require(sha(code_root/name) == digest, 'Training code SHA mismatch: '+name)
    normalization = load(run/'normalization.pt'); representation = read(run/'representation.json')
    scales = np.asarray(representation['scales'])
    require(scales.shape == (4,) and np.isfinite(scales).all() and (scales > 0).all()
            and np.array_equal(scales, np.asarray(normalization['scales'])), 'Motion scales differ')
    for key in ('mean', 'std'):
        value = normalization[key]
        require(torch.is_tensor(value) and value.shape == (1540,) and torch.isfinite(value).all(),
                'Invalid normalization '+key)
    require((normalization['std'] > 0).all(), 'Nonpositive feature standard deviation')
    plans = load(run/'teacher_plans.pt'); plan_ids = list(plans)
    fit_indices = validate_split(protocol, normalization, plan_ids)
    matching = read(run/'matching.json'); require(set(matching) == set(ARMS), 'Matching arms differ')
    reports, recomputed, reference, summaries = {}, {}, {}, {}
    max_roundtrip = 0.; ambiguous_count = 0
    source_files = {'protocol.json', 'status.json', 'representation.json', 'normalization.pt',
                    'teacher_plans.pt', 'matching.json', 'audio_gate.json'}
    reconstructed_cells = {}
    for arm in ARMS:
        folder = run/arm; reports[arm] = {}; recomputed[arm] = {}; summaries[arm] = {}
        losses = read(folder/'losses.json')
        budget = validate_budget(losses, matching[arm], fit_indices, protocol['epochs'])
        require(budget['updates'] == 2010, 'Expected 2010 formal updates')
        final, last = load(folder/'final.pt'), load(folder/'last.pt')
        require(final['schema'] == last['schema'] == p.SCHEMA and final['arm'] == last['arm'] == arm
                and final['epochs'] == last['epoch'] == 30
                and final['normalization_sha256'] == sha(run/'normalization.pt'), 'Checkpoint identity differs')
        require(set(final['model']) == set(last['model']) and all(torch.equal(final['model'][key], value)
                for key, value in last['model'].items()), 'Final/last model tensors differ')
        require(last['optimizer']['state'] and all(int(value['step']) == 2010
                for value in last['optimizer']['state'].values()), 'Optimizer update history differs')
        source_files |= {arm+'/'+name for name in ('losses.json', 'final.pt', 'last.pt')}
        for cell in HOLDS:
            where = arm+'/'+cell; report = read(folder/(cell+'.json')); draws = load(folder/(cell+'.pt'))
            wanted = protocol['split'][cell]
            require(list(draws) == [r['clip_id'] for r in wanted] == [r['clip_id'] for r in report['rows']],
                    'Evaluation IDs/order differ: '+where)
            require(report['intervention'] == 'real' and report['donor_mapping'] == {}, 'Primary intervention differs')
            validate_aggregates(report, where)
            rescored = []; clips = []
            for meta, row in zip(wanted, report['rows']):
                cid = row['clip_id']; compare(meta, {key: row[key] for key in meta}, where+'/'+cid+'/metadata', atol=0, rtol=0)
                x, y, teacher, valid, anchor = validate_draws(draws[cid], scales, plans[cid], where+'/'+cid)
                if arm == ARMS[0]:
                    reference[cid] = {key: np.asarray(draws[cid][key]).copy() for key in ('target','teacher','valid','anchor')}
                else:
                    for key, expected in reference[cid].items():
                        require(np.array_equal(expected, np.asarray(draws[cid][key])), 'Paired '+key+' differs: '+cid)
                updated, error, ambiguous = rescore_draw(draws[cid], scales, row, where+'/'+cid)
                max_roundtrip = max(max_roundtrip, error); ambiguous_count += int(any(ambiguous)); rescored.append(updated)
                plan = plans[cid]
                event_count = sum(s['duration'] in p.DURATIONS and not s['right_censored']
                                  for group in plan['segments'] for s in group)
                require(row['event_count'] == event_count and ((row['event_nll'] is None) == (event_count == 0)),
                        'Teacher completed-event count differs: '+cid)
                if arm == ARMS[0]:
                    clip = {'state': y, 'valid': torch.from_numpy(valid), 'plan': plan, 'reconstructed': teacher}
                    clip['uniform'] = p.uniform_reconstruction(clip)
                    clip['mean'] = np.zeros_like(y)
                    for start, end in p.runs(valid): clip['mean'][start:end] = y[start:end].mean(0)
                    clips.append(clip)
            reports[arm][cell] = report
            recomputed[arm][cell] = {'rows': rescored, 'summary': p.aggregate(rescored)}
            summaries[arm][cell] = recomputed[arm][cell]['summary']
            if arm == ARMS[0]:
                reconstructed_cells[cell] = {}
                for key in ('reconstructed', 'uniform', 'mean'):
                    actual = p.reconstruction_summary(clips, list(range(len(clips))), key)
                    # Raw upper9 targets were not saved; group reconstruction is independently scored.
                    actual.pop('upper9_centered_mse')
                    expected = {k: v for k, v in representation['cells'][cell][key].items() if k != 'upper9_centered_mse'}
                    compare(actual, expected, 'representation/'+cell+'/'+key)
                    reconstructed_cells[cell][key] = actual
            source_files |= {where+'.json', where+'.pt'}
    compare(matching[ARMS[0]], matching[ARMS[1]], 'paired_budget', atol=0, rtol=0)
    gate = p.audio_gate(reports); compare(gate, read(run/'audio_gate.json'), 'audio_gate')
    restored_gate = p.audio_gate(recomputed)
    require(restored_gate['passed'] == gate['passed'], 'FP32 restored draws change audio gate decision')
    rep_passed = representation_gate(reconstructed_cells)
    require(rep_passed == representation['passed'] == status['representation_passed']
            and gate['passed'] == status['audio_passed'], 'Recomputed gate/status differs')
    # Intervention predictions were not archived; audit report aggregation/common support only.
    intervention_reports = {'real': reports['local_audio']['sentence']}
    for mode in ('static', 'reverse', 'mismatch'):
        report = read(run/'local_audio'/(mode+'.json'))
        require(report['intervention'] == mode, 'Intervention label differs')
        validate_aggregates(report, 'local_audio/'+mode); intervention_reports[mode] = report
        source_files.add('local_audio/'+mode+'.json')
    common = read(run/'local_audio/common_interventions.json')
    support = set(intervention_reports['mismatch']['donor_mapping'])
    compare({'clip_ids': sorted(support), 'summary': {mode: p.aggregate([r for r in report['rows'] if r['clip_id'] in support])
        for mode, report in intervention_reports.items()}}, common, 'common_interventions')
    source_files.add('local_audio/common_interventions.json')
    report = {'schema': SCHEMA, 'passed': True, 'run': str(run), 'training_epochs_per_arm': 30,
        'updates_per_arm': 2010, 'cells': {cell: len(protocol['split'][cell]) for cell in p.CELLS},
        'saved_draw_sets_recomputed': 6, 'seeds_per_clip': len(p.SEEDS), 'max_metric_roundtrip_abs_error': max_roundtrip,
        'metric_roundtrip_tolerance': {'atol': METRIC_ATOL, 'rtol': METRIC_RTOL},
        'clip_arm_pairs_with_boundary_rounding_ambiguity': ambiguous_count,
        'representation_gate_recomputed': rep_passed, 'audio_gate_recomputed': gate,
        'restored_draw_audio_gate': restored_gate, 'free_run_summaries_recomputed': summaries,
        'scope': {'saved_artifacts_only': True, 'neural_forward_rerun': False, 'generator_started': False,
            'nonupper_or_identity_verified': False, 'normalization_values_refit': False,
            'source_data_files_rehashed': False, 'likelihood_values_recomputed_from_logits': False,
            'initial_model_tensor_matching_independently_verified': False,
            'upper9_representation_error_recomputed': False, 'intervention_draws_recomputed': False}}
    manifest = {'schema': SCHEMA, 'files': {name: sha(run/name) for name in sorted(source_files)},
                'auditor_sha256': sha(__file__), 'code_sha256': protocol['code_sha256'],
                'protocol_sha256': protocol['protocol_sha256']}
    return report, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); torch.set_num_threads(4)
    output = args.output or args.run/'result_audit'
    require(not output.exists(), 'Use a fresh audit output directory')
    report, manifest = audit_run(args.run, args.code_root)
    output.mkdir(parents=True)
    p.save_json(output/'audit.json', report)
    manifest['audit_sha256'] = sha(output/'audit.json')
    p.save_json(output/'manifest.json', manifest)
    print('MOTION_PROCESS_AUDIT_PASSED', json.dumps({'output': str(output), 'max_roundtrip_error': report['max_metric_roundtrip_abs_error']}), flush=True)


if __name__ == '__main__':
    main()
