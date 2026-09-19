"""Small independent post-run summary; never trains, selects, or replaces a model.

Inner validation acceptance is rechecked with the original auditor. Outer
archives are recomputed separately and are descriptive diagnostics only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_prior_audio_adapter import (
    EVALUATION_SCHEMA, _sha, audit, half_pair_spread, sentence_cluster_bootstrap,
)
from scripts.joint_motion_metrics import score_clip, summarize
from scripts.train_continuous_motion_latent import _value_sha

SCHEMA = 'bounded_prior_audio_summary_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def evaluation_directory(canonical):
    """Respect interrupted-evaluation recovery pointers, including local copies."""
    canonical = Path(canonical)
    pointer = canonical.with_name(canonical.name + '_artifact.json')
    if pointer.exists():
        stored = Path(read(pointer)['directory'])
        destination = stored if stored.is_dir() else canonical.parent / stored.name
    else:
        destination = canonical
    require((destination / 'result.json').is_file() and (destination / 'curves.pt').is_file(),
            'Complete evaluation archive missing: ' + str(destination))
    return destination


def verify_checkpoints(root, protocol):
    """Hash every saved prior submodule and recompute endpoint pair assertions."""
    root = Path(root); prior = load(root / 'prior_final.pt')
    prior_sha = _value_sha(prior['state'])
    protocol_sha = _value_sha(protocol)
    require(prior['stage'] == 'prior' and prior['step'] == prior['budget'] == protocol['budgets'][1],
            'Prior checkpoint does not finish the declared budget')
    require(prior['binding']['protocol_sha256'] == protocol_sha,
            'Prior checkpoint protocol binding differs')
    points = []
    for step in protocol['milestones']:
        point = root / f'point_{step}'
        arms = {arm: load(point / (arm + '.pt')) for arm in ('audio', 'matched_static')}
        hashes = {}
        for arm, checkpoint in arms.items():
            require(checkpoint['stage'] == arm and checkpoint['step'] == step
                    and checkpoint['budget'] == protocol['budgets'][2],
                    f'Checkpoint stage/step/budget mismatch at {step}/{arm}')
            frozen = {key[len('prior.'):]: value for key, value in checkpoint['state'].items()
                      if key.startswith('prior.')}
            hashes[arm] = _value_sha(frozen)
            require(bool(frozen) and hashes[arm] == prior_sha
                    and checkpoint['binding']['prior_sha256'] == prior_sha,
                    f'Frozen prior changed or binding differs at {step}/{arm}')
            require(checkpoint['binding'] == {**prior['binding'], 'prior_sha256': prior_sha},
                    f'Adapter AE/stat/protocol binding differs at {step}/{arm}')
        matched = {key: arms['audio'][key] == arms['matched_static'][key]
                   for key in ('initial_state_sha256', 'order_sha256', 'step')}
        require(all(matched.values()), f'Matched training pairing differs at {step}')
        require(arms['audio']['binding'] == arms['matched_static']['binding']
                and arms['audio']['config'] == arms['matched_static']['config'],
                f'Matched checkpoint binding/config differs at {step}')
        require(read(point / 'pairing.json') == matched, f'Saved pairing claim differs at {step}')
        points.append({'step': step, 'paired': matched, 'prior_state_sha256': hashes,
                       'checkpoint_sha256': {arm: _sha(point / (arm + '.pt')) for arm in arms}})
    return {'prior_state_sha256': prior_sha, 'all_saved_priors_exact': True,
            'prior_checkpoint_sha256': _sha(root / 'prior_final.pt'), 'points': points}


def load_outer_evaluation(directory, role):
    """Recompute holdout scores without relabeling data or invoking inner gates."""
    directory = evaluation_directory(directory)
    result = read(directory / 'result.json'); packed = load(directory / 'curves.pt')
    require(result['schema'] == packed['schema'] == EVALUATION_SCHEMA
            and result['mode'] == 'free_generation', role + ': free evaluator archive required')
    clips = packed['clips']; saved_rows = result['per_clip_scores']
    saved = {row['clip_id']: row for row in saved_rows}
    require(len(saved_rows) == len(saved) == len(clips) == result['clips'] == result['scored_clips']
            and set(saved) == set(clips), role + ': incomplete or duplicate scored membership')
    rows = {}; scale = None; seeds = None
    for cid, curve in clips.items():
        meta = curve['metadata']
        require(meta['clip_id'] == cid and meta['split'] == 'holdout'
                and meta.get('sentence') is not None, role + ': exclusively holdout metadata required')
        require(all(saved[cid].get(key) == meta.get(key) for key in ('clip_id', 'split', 'sentence')),
                role + ': score/curve metadata differs')
        samples = np.asarray(curve['samples']); target = np.asarray(curve['target'])
        native = np.asarray(curve['native_valid']); mask = np.asarray(curve['score_mask'])
        generated = np.asarray(curve['generated_mask'])
        require(samples.ndim == 3 and samples.shape[0] >= 4 and samples.shape[2] == 9
                and target.shape == samples.shape[1:], role + ': four native draws required')
        require(all(value.dtype == bool and value.shape == samples.shape[1:2]
                    for value in (native, mask, generated)) and np.all(~mask | native)
                and np.array_equal(generated, native), role + ': mask contract differs')
        require(np.isfinite(samples[:, native]).all() and curve['raw_coefficient_prediction'] is True,
                role + ': finite raw native predictions required')
        row_seeds = curve['seeds']; row_scale = np.asarray(saved[cid]['scales'])
        if seeds is None:
            seeds, scale = row_seeds, row_scale
        require(len(row_seeds) == len(samples) and len(set(row_seeds)) == len(row_seeds)
                and all(type(seed) is int for seed in row_seeds) and seeds == row_seeds
                and np.array_equal(scale, row_scale), role + ': shared unique seeds and training scales required')
        metric = score_clip(samples, target, mask, row_scale)
        require(all(np.isclose(metric['joint_fair_es'][kind], saved[cid]['joint_fair_es'][kind],
                               rtol=1e-6, atol=1e-9) for kind in ('raw', 'centered')),
                role + ': saved fair ES differs from raw curves')
        rows[cid] = {'metric': metric, 'spread': half_pair_spread(samples, mask, row_scale),
                     'target': target, 'native_valid': native, 'score_mask': mask,
                     'generated_mask': generated, 'metadata': meta, 'seeds': row_seeds,
                     'run_records': curve.get('run_records')}
    checks = result['numerics']
    require(len(checks) == len(clips) and {check['clip_id'] for check in checks} == set(clips),
            role + ': incomplete numerical checks')
    numerical = bool(result['numerical_gate']['passed'] and all(
        check['nonfinite_values'] == 0 and check['other43_exact']
        and check['native_invalid_exact_baseline'] and not check['prediction_clipped'] for check in checks))
    total = summarize([row['metric'] for row in rows.values()]) if rows else None
    if total is not None:
        total['half_pair_centered_spread'] = float(np.mean([row['spread'] for row in rows.values()]))
        numerator, denominator = total['speed']['all']['rms'], total['speed']['reference_all']['rms']
        total['speed_ratio_to_reference'] = float(numerator / denominator) if denominator else None
    return {'directory': str(directory), 'result': result, 'rows': rows, 'summary': total,
            'scales': scale, 'numerical_passed': numerical,
            'sha256': {name: _sha(directory / name) for name in ('result.json', 'curves.pt')}}


def paired_comparison(prior, audio, *, allow_subset=False):
    require(set(audio['rows']) <= set(prior['rows']) if allow_subset else set(audio['rows']) == set(prior['rows']),
            'Paired outer membership differs')
    require(prior['result']['steps'] == audio['result']['steps'], 'Paired outer solver steps differ')
    if not audio['rows']:
        return {'supported': False, 'reason': 'No matched donor/audio support'}
    require(np.array_equal(prior['scales'], audio['scales']), 'Paired outer training scales differ')
    ids = sorted(audio['rows']); sentences = []
    for cid in ids:
        a, p = audio['rows'][cid], prior['rows'][cid]
        require(a['metadata'] == p['metadata'] and a['seeds'] == p['seeds']
                and a['run_records'] == p['run_records'], 'Paired outer metadata/noise differs: ' + cid)
        for key in ('target', 'native_valid', 'score_mask', 'generated_mask'):
            require(np.array_equal(a[key], p[key], equal_nan=True), 'Paired outer ' + key + ' differs: ' + cid)
        sentences.append(str(a['metadata']['sentence']))
    result = {'supported': True, 'clip_count': len(ids), 'sentence_count': len(set(sentences))}
    for kind in ('raw', 'centered'):
        deltas = [audio['rows'][cid]['metric']['joint_fair_es'][kind]
                  - prior['rows'][cid]['metric']['joint_fair_es'][kind] for cid in ids]
        if len(set(sentences)) >= 2:
            result[kind] = sentence_cluster_bootstrap(deltas, sentences)
        else:
            result[kind] = {'mean_delta': float(np.mean(deltas)), 'ci95': None,
                            'reason': 'Fewer than two supported sentences; cluster CI unavailable'}
    return result


def compact_metrics(summary, reference=None, numerical_passed=None):
    if summary is None:
        return {'supported': False}
    spread = summary['half_pair_centered_spread']
    denominator = reference['half_pair_centered_spread'] if reference else None
    return {'supported': True, 'clips': summary['clips'],
            'centered_fair_es': summary['joint_fair_es']['centered'],
            'raw_fair_es': summary['joint_fair_es']['raw'], 'centered_half_pair_spread': spread,
            'spread_ratio_to_prior': float(spread / denominator) if denominator else None,
            'speed_ratio_to_reference': summary['speed_ratio_to_reference'],
            'group_rms_ratio_to_reference': summary['rms_ratio'], 'group_raw_oob': summary['raw_oob'],
            'numerical_passed': numerical_passed}


def summarize_run(root, output=None):
    root = Path(root); protocol = read(root / 'protocol.json'); status = read(root / 'status.json')
    decision = read(root / 'decision.json')
    require(status['state'] == 'complete', 'Training/evaluation must complete before final summary')
    require(protocol['schema'] == 'bounded_audio_experiment_v1'
            and len(protocol['milestones']) == 2, 'Expected fixed two-milestone bounded experiment')
    require(protocol.get('default_replaced') is False and decision.get('default_replaced') is False
            and status.get('default_replaced') is False, 'Default model must remain unchanged')
    require(decision['selection_data'] == 'inner_validation only', 'Decision selection scope differs')
    checkpoint_checks = verify_checkpoints(root, protocol)
    prior_inner = evaluation_directory(root / 'prior_generation/inner_validation')
    inner = []; records = []
    for step in protocol['milestones']:
        point = root / f'point_{step}'; saved = read(point / 'acceptance.json')
        current = audit(prior_inner, evaluation_directory(point / 'audio_generation/inner_validation'),
                        evaluation_directory(point / 'audio_static/inner_validation'),
                        matched_static_dir=evaluation_directory(point / 'matched_static_generation/inner_validation'))
        require(current['accepted'] == saved['accepted'] and current['failures'] == saved['failures']
                and current['metrics'] == saved['metrics'], 'Saved inner acceptance differs from independent recomputation')
        reference = current['metrics']['arms']['prior']
        inner.append({'step': step, 'accepted': current['accepted'], 'failures': current['failures'],
                      'arms': {role: compact_metrics(value, reference, not any(
                                  failure.startswith(role + ': numerical/') for failure in current['failures']))
                               for role, value in current['metrics']['arms'].items()},
                      'audio_minus_prior': current['metrics']['comparisons']['audio_minus_prior'],
                      'audio_minus_static': current['metrics']['comparisons']['audio_minus_static'],
                      'audio_minus_matched_static': current['metrics']['comparisons']['audio_minus_matched_static']})
        records.append({'step': step, 'accepted': current['accepted']})
    require([{key: row[key] for key in ('step', 'accepted')} for row in decision['records']] == records,
            'Decision records differ from inner acceptance')
    passing = [row for row in records if row['accepted']]
    expected = min((row['step'] for row in passing), default=None)
    actual = decision['chosen']['step'] if decision['chosen'] else None
    require(actual == expected, 'Decision must use earliest accepted inner endpoint')
    chosen_step = expected if expected is not None else protocol['milestones'][-1]
    outer = {'scope': 'Historically exposed diagnostics only; never used for endpoint selection',
             'evaluated_step': chosen_step, 'used_for_selection': False, 'arms': {}, 'comparisons': {}}
    mapping = {'prior': 'prior_generation', 'audio': 'audio_generation',
               'matched_static': 'matched_static_generation', 'static': 'audio_static',
               'reverse': 'audio_reverse', 'mismatch': 'audio_mismatch'}
    archives = {}
    if protocol['outer_ids']:
        for role, directory in mapping.items():
            archive = load_outer_evaluation(root / 'outer' / directory / 'holdout', role)
            expected_ids = set(protocol['outer_ids'])
            if role == 'mismatch':
                excluded = set(archive['result'].get('mismatch_excluded_no_donor', []))
                require(not (set(archive['rows']) & excluded) and set(archive['rows']) | excluded == expected_ids,
                        'Mismatch donor exclusions do not account for outer membership')
            else:
                require(set(archive['rows']) == expected_ids, role + ': protocol outer membership differs')
            require(archive['result']['use_audio'] is (role != 'prior')
                    and archive['result']['intervention'] == ('real' if role in ('audio', 'prior') else 'static' if role == 'matched_static' else role),
                    role + ': outer intervention differs')
            archives[role] = archive
        for role, archive in archives.items():
            outer['arms'][role] = compact_metrics(archive['summary'], archives['prior']['summary'], archive['numerical_passed'])
            if role != 'prior':
                outer['comparisons'][role + '_minus_prior'] = paired_comparison(archives['prior'], archive, allow_subset=role == 'mismatch')
        outer['comparisons']['audio_minus_matched_static'] = paired_comparison(archives['matched_static'], archives['audio'])
        outer['comparisons']['audio_minus_static'] = paired_comparison(archives['static'], archives['audio'])
    result = {'schema': SCHEMA, 'run_directory': str(root.resolve()), 'state': status['state'],
              'default_replaced': False, 'naturalness_certified': False, 'needs_visual_review': True,
              'accepted_inner': expected is not None, 'selected_step': expected,
              'diagnostic_endpoint_step': chosen_step, 'inner': inner, 'outer': outer,
              'checkpoint_checks': checkpoint_checks,
              'notes': ['AE and prior are newly trained on inner training only; inherited feature/base exposure remains.',
                        'A frozen prior and bounded velocity correction do not guarantee diversity, naturalness, or useful timing.',
                        'Outer results are reported after a locked inner decision; no outer acceptance gate or reselection.',
                        'Group order: up, down, squint, wide. ES is lower-better; all fixed draws contribute.',
                        'Intervals use fixed-seed 4096-resample paired sentence-cluster bootstrap.'],
              'sources': {'protocol_sha256': _sha(root / 'protocol.json'), 'decision_sha256': _sha(root / 'decision.json'),
                          'outer': {role: {'directory': archive['directory'], 'sha256': archive['sha256']}
                                    for role, archive in archives.items()}}}
    destination = Path(output) if output is not None else root
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    (destination / 'summary.md').write_text(markdown(result), encoding='utf8')
    return result


def _number(value):
    return 'n/a' if value is None else f'{value:.5f}'


def _table(arms):
    lines = ['| Arm | Centered ES | Raw ES | Spread | Spread/prior | Speed/GT | RMS up/down/squint/wide | OOB up/down/squint/wide |',
             '|---|---:|---:|---:|---:|---:|---|---|']
    for role, row in arms.items():
        if not row['supported']:
            lines.append(f'| {role} | unsupported | | | | | | |'); continue
        values = [_number(row[key]) for key in ('centered_fair_es', 'raw_fair_es', 'centered_half_pair_spread',
                                                'spread_ratio_to_prior', 'speed_ratio_to_reference')]
        lines.append('| ' + role + ' | ' + ' | '.join(values) + ' | '
                     + '/'.join(_number(value) for value in row['group_rms_ratio_to_reference']) + ' | '
                     + '/'.join(_number(value) for value in row['group_raw_oob']) + ' |')
    return '\n'.join(lines)


def _comparison(value):
    parts = []
    for kind in ('centered', 'raw'):
        row = value[kind]; ci = row['ci95']
        parts.append(f'{kind} delta {_number(row["mean_delta"])}; 95% sentence CI '
                     + (f'[{_number(ci[0])}, {_number(ci[1])}]' if ci else 'unavailable'))
    return '; '.join(parts)


def markdown(result):
    lines = ['# Bounded prior audio experiment', '',
             f'Complete. Inner accepted: {result["accepted_inner"]}; selected step: {result["selected_step"]}. '
             'Default model unchanged. Naturalness still requires visual review.', '']
    for point in result['inner']:
        lines += [f'## Inner validation · step {point["step"]}', '',
                  f'Acceptance: {point["accepted"]}.', '', _table(point['arms']), '',
                  'Audio − prior: ' + _comparison(point['audio_minus_prior']) + '.', '']
        if point['failures']:
            lines += ['Failures: ' + '; '.join(point['failures']) + '.', '']
    outer = result['outer']
    lines += [f'## Outer diagnostic · locked step {outer["evaluated_step"]}', '', outer['scope'] + '.', '',
              _table(outer['arms']), '']
    if 'audio_minus_prior' in outer['comparisons']:
        lines += ['Audio − prior: ' + _comparison(outer['comparisons']['audio_minus_prior']) + '.', '']
    lines += ['Saved prior submodules and two-endpoint training pairing independently verified.', '',
              'Frozen weights and bounded correction do not guarantee motion quality or audio timing.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = summarize_run(args.root, args.output)
    print(json.dumps({'accepted_inner': result['accepted_inner'], 'selected_step': result['selected_step'],
                      'default_replaced': False, 'output': str((args.output or args.root).resolve())}))


if __name__ == '__main__':
    main()
