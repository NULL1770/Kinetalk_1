"""Internal validation acceptance checks for bounded audio-prior adaptation.

The gate is an engineering guard, not a claim of naturalness, independent
generalization, or reliable alignment of a particular brow event to speech.
Only evaluator archives explicitly marked ``inner_validation`` are accepted.
All scores are recomputed from raw arrays with the saved training-only scale.
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
from scripts.joint_motion_metrics import GROUP_NAMES, score_clip, summarize

SCHEMA = 'prior_audio_adapter_audit_v1'
EVALUATION_SCHEMA = 'continuous_motion_latent_evaluation_v1'
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_RESAMPLES = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _runs(mask):
    edge = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))


def half_pair_spread(samples, mask, scales):
    """Half the mean pairwise normalized RMS-L2 of centered trajectories.

    Center independently inside each observed run so gaps create no artificial
    motion. This is exactly the finite-ensemble spread term in centered fair
    ES, not a variance of the ensemble mean or an average framewise distance.
    """
    value = np.asarray(samples, dtype=np.float64) / np.asarray(scales)
    centered = np.zeros_like(value)
    for left, right in _runs(mask):
        part = value[:, left:right] - value[:, left:left + 1]
        centered[:, left:right] = part - part.mean(axis=1, keepdims=True)
    flat = centered[:, mask].reshape(len(value), -1)
    pairs = [np.sqrt(np.square(flat[i] - flat[j]).mean())
             for i in range(len(flat)) for j in range(i)]
    return float(np.mean(pairs) / 2)


def sentence_cluster_bootstrap(deltas, sentences):
    """Resample sentences, retaining all clips, then compute clip-equal mean."""
    deltas = np.asarray(deltas, dtype=np.float64)
    names = sorted(set(sentences))
    _require(len(names) >= 2, 'At least two internal validation sentences are required')
    _require(len(deltas) == len(sentences) and np.isfinite(deltas).all(),
             'Finite paired delta per clip required')
    groups = [deltas[np.asarray(sentences) == name] for name in names]
    sums = np.asarray([group.sum() for group in groups])
    counts = np.asarray([len(group) for group in groups])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(groups), size=(BOOTSTRAP_RESAMPLES, len(groups)))
    estimates = sums[draws].sum(1) / counts[draws].sum(1)
    return {'mean_delta': float(deltas.mean()),
            'ci95': np.quantile(estimates, [.025, .975]).tolist(),
            'sentence_count': len(names), 'clip_count': len(deltas),
            'resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED,
            'method': 'percentile sentence-cluster paired bootstrap; clip-equal mean after each draw',
            'sentence_mean_deltas': {name: float(group.mean()) for name, group in zip(names, groups)}}


def _load(directory, role):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text(encoding='utf8'))
    packed = torch.load(directory / 'curves.pt', map_location='cpu', weights_only=False)
    _require(result.get('schema') == packed.get('schema') == EVALUATION_SCHEMA,
             f'{role}: unsupported evaluator schema')
    _require(result.get('mode') == 'free_generation', f'{role}: free generation required')
    clips = packed.get('clips')
    _require(isinstance(clips, dict) and bool(clips), f'{role}: nonempty clips required')
    score_rows = result.get('per_clip_scores', [])
    score_map = {row['clip_id']: row for row in score_rows}
    _require(len(score_rows) == len(score_map) == len(clips)
             and set(score_map) == set(clips), f'{role}: incomplete or duplicate scored membership')
    _require(result.get('clips') == result.get('scored_clips') == len(clips),
             f'{role}: inconsistent clip counts')
    scales = np.asarray(score_rows[0]['scales'], dtype=np.float64)
    _require(scales.shape == (9,) and np.isfinite(scales).all() and (scales > 0).all(),
             f'{role}: positive finite training scales required')
    _require(all(np.array_equal(row['scales'], scales) for row in score_rows),
             f'{role}: differing metric scales')
    rows = {}; common_seeds = None; native_finite = True
    for cid, source in clips.items():
        meta = source.get('metadata', {})
        _require(meta.get('clip_id') == cid and meta.get('split') == 'inner_validation',
                 f'{role}: only inner_validation clips are permitted; mixed/external splits rejected')
        _require(meta.get('sentence') is not None and str(meta['sentence']) != '',
                 f'{role}: sentence metadata required')
        _require(all(score_map[cid].get(key) == meta.get(key)
                     for key in ('clip_id', 'split', 'sentence')),
                 f'{role}: score and curve metadata differ')
        samples, target = _array(source['samples']), _array(source['target'])
        native, score_mask = _array(source['native_valid']), _array(source['score_mask'])
        generated = _array(source['generated_mask'])
        _require(samples.ndim == 3 and samples.shape[0] >= 4 and samples.shape[2] == 9
                 and target.shape == samples.shape[1:], f'{role}: four or more native [K,T,9] draws required')
        _require(all(mask.dtype == np.bool_ and mask.shape == samples.shape[1:2]
                     for mask in (native, score_mask, generated))
                 and score_mask.any() and np.all(~score_mask | native)
                 and np.array_equal(generated, native), f'{role}: native/score/generation masks invalid')
        _require(np.isfinite(target[score_mask]).all(), f'{role}: nonfinite observed reference')
        seeds = list(source['seeds'])
        _require(len(seeds) == len(samples) and len(set(seeds)) == len(seeds)
                 and all(type(seed) is int for seed in seeds), f'{role}: unique integer seed per draw required')
        if common_seeds is None:
            common_seeds = seeds
        _require(seeds == common_seeds, f'{role}: seed order must be constant across clips')
        _require(source.get('raw_coefficient_prediction') is True, f'{role}: raw prediction archive required')
        finite = bool(np.isfinite(samples[:, native]).all()); native_finite &= finite
        metric = score_clip(samples, target, score_mask, scales) if finite else None
        if metric is not None:
            for kind in ('raw', 'centered'):
                _require(np.isclose(metric['joint_fair_es'][kind], score_map[cid]['joint_fair_es'][kind],
                                    rtol=1e-6, atol=1e-9), f'{role}: saved {kind} score differs from raw curves')
        rows[cid] = {'metadata': meta, 'samples': samples, 'target': target,
                     'native_valid': native, 'score_mask': score_mask, 'generated_mask': generated,
                     'seeds': seeds, 'run_records': source.get('run_records'), 'metric': metric,
                     'spread': half_pair_spread(samples, score_mask, scales) if finite else None}
    _require(len({str(row['metadata']['sentence']) for row in rows.values()}) >= 2,
             f'{role}: at least two internal validation sentences required')
    checks = result.get('numerics', [])
    check_ids = [check.get('clip_id') for check in checks]
    _require(len(check_ids) == len(set(check_ids)) == len(rows) and set(check_ids) == set(rows),
             f'{role}: incomplete numerical/protection checks')
    numerical = (result.get('numerical_gate', {}).get('passed') is True and native_finite
                 and all(check.get('nonfinite_values') == 0
                         and check.get('other43_exact') is True
                         and check.get('native_invalid_exact_baseline') is True
                         and check.get('prediction_clipped') is False for check in checks))
    return {'path': str(directory.resolve()), 'rows': rows, 'scales': scales,
            'numerical_passed': bool(numerical), 'result': result,
            'hashes': {name: _sha(directory / name) for name in ('result.json', 'curves.pt')}}


def _match(reference, candidate, role):
    _require(reference['rows'].keys() == candidate['rows'].keys(), f'{role}: clip membership differs')
    _require(np.array_equal(reference['scales'], candidate['scales']), f'{role}: metric scales differ')
    _require(reference['result'].get('steps') == candidate['result'].get('steps'),
             f'{role}: sampling step count differs')
    for cid, original in reference['rows'].items():
        other = candidate['rows'][cid]
        _require(original['metadata'] == other['metadata'], f'{role}/{cid}: metadata differs')
        _require(original['seeds'] == other['seeds'], f'{role}/{cid}: paired seeds/order differ')
        _require(original['run_records'] == other['run_records'], f'{role}/{cid}: noise/run records differ')
        for name in ('target', 'native_valid', 'score_mask', 'generated_mask'):
            _require(np.array_equal(original[name], other[name], equal_nan=True),
                     f'{role}/{cid}: {name} differs')


def _ratio(value, reference):
    return float(value / reference) if value is not None and reference is not None and reference > 0 else None


def audit(prior_dir, audio_dir, static_dir, output=None, *, matched_static_dir=None):
    """Audit paired internal validation archives, never fit or select examples."""
    paths = {'prior': prior_dir, 'audio': audio_dir, 'static': static_dir}
    if matched_static_dir is not None:
        paths['matched_static'] = matched_static_dir
    arms = {role: _load(path, role) for role, path in paths.items()}
    for role, arm in arms.items():
        _match(arms['prior'], arm, role)
    _require(arms['prior']['result'].get('use_audio') is False,
             'prior must be the global-only generation arm')
    _require(arms['audio']['result'].get('use_audio') is True
             and arms['audio']['result'].get('intervention') == 'real',
             'audio must use real audio generation')
    for role in ('static', 'matched_static'):
        if role in arms:
            _require(arms[role]['result'].get('use_audio') is True
                     and arms[role]['result'].get('intervention') == 'static',
                     f'{role}: static audio intervention required')
    ids = sorted(arms['prior']['rows'])
    sentences = [str(arms['prior']['rows'][cid]['metadata']['sentence']) for cid in ids]
    failures = [f'{role}: numerical/protected43 gate failed'
                for role, arm in arms.items() if not arm['numerical_passed']]
    metrics = {'arms': {}, 'comparisons': {}}
    usable = all(row['metric'] is not None for arm in arms.values() for row in arm['rows'].values())
    per_clip = []
    if usable:
        for role, arm in arms.items():
            summary = summarize([arm['rows'][cid]['metric'] for cid in ids])
            summary['half_pair_centered_spread'] = float(np.mean([arm['rows'][cid]['spread'] for cid in ids]))
            summary['speed_ratio_to_reference'] = _ratio(summary['speed']['all']['rms'], summary['speed']['reference_all']['rms'])
            metrics['arms'][role] = summary
        for control in ('prior', 'static', 'matched_static'):
            if control not in arms:
                continue
            metrics['comparisons']['audio_minus_' + control] = {
                kind: sentence_cluster_bootstrap(
                    [arms['audio']['rows'][cid]['metric']['joint_fair_es'][kind]
                     - arms[control]['rows'][cid]['metric']['joint_fair_es'][kind] for cid in ids], sentences)
                for kind in ('raw', 'centered')}
        a, p = metrics['arms']['audio'], metrics['arms']['prior']
        comparison = metrics['comparisons']['audio_minus_prior']['centered']
        metrics['spread_ratio_to_prior'] = _ratio(a['half_pair_centered_spread'], p['half_pair_centered_spread'])
        if not (comparison['mean_delta'] < 0 and comparison['ci95'][1] < 0):
            failures.append('centered ES: audio must improve prior mean and sentence-cluster CI upper must be below zero')
        if not a['joint_fair_es']['raw'] <= p['joint_fair_es']['raw'] * 1.02:
            failures.append('raw ES: audio exceeds 1.02 times prior')
        if metrics['spread_ratio_to_prior'] is None or metrics['spread_ratio_to_prior'] < .8:
            failures.append('diversity: centered half-pair spread must retain at least 0.8 of prior')
        speed = a['speed_ratio_to_reference']
        if speed is None or not .5 <= speed <= 1.5:
            failures.append('speed: audio RMS speed/reference must be in [0.5,1.5]')
        if any(value is None or not .5 <= value <= 1.5 for value in a['rms_ratio']):
            failures.append('amplitude: every upper group RMS/reference must be in [0.5,1.5]')
        for control in ('static', 'matched_static'):
            if control in arms and not metrics['comparisons']['audio_minus_' + control]['centered']['mean_delta'] < 0:
                failures.append(f'centered ES: real audio must beat {control} mean; this is auxiliary timing evidence')
        per_clip = [{'clip_id': cid, 'sentence': sentence,
                     'arms': {role: {'joint_fair_es': arm['rows'][cid]['metric']['joint_fair_es'],
                                     'half_pair_centered_spread': arm['rows'][cid]['spread']}
                              for role, arm in arms.items()}}
                    for cid, sentence in zip(ids, sentences)]
    else:
        failures.append('raw numerical failure prevents complete paired score recomputation')
    result = {'schema': SCHEMA, 'accepted': not failures, 'failures': failures,
              'scope': 'internal acceptance gate only; not naturalness certification or independent final evaluation',
              'naturalness_certified': False, 'split': 'inner_validation',
              'clip_count': len(ids), 'sentence_count': len(set(sentences)),
              'seeds': arms['prior']['rows'][ids[0]]['seeds'],
              'metrics': metrics, 'per_clip': per_clip,
              'gate_contract': {'centered_es': 'audio mean < prior and sentence-cluster 95% CI upper < 0',
                                'raw_es_max_ratio_to_prior': 1.02, 'min_spread_ratio_to_prior': .8,
                                'speed_ratio_to_reference_range': [.5, 1.5],
                                'each_group_rms_ratio_to_reference_range': [.5, 1.5],
                                'static_mean_improvement_required': True,
                                'matched_static_mean_improvement_required': matched_static_dir is not None,
                                'group_order': list(GROUP_NAMES), 'numerical_and_protected43_required': True},
              'metric_notes': {'spread': 'half mean unordered seed-pair RMS-L2; train-scale normalized and observed-run centered',
                               'scores': 'all fixed draws; recomputed raw arrays; equal weight per clip',
                               'bootstrap': 'sentences resampled with replacement; clips retained as paired clusters',
                               'warning': 'Repeated model selection on this internal split can overfit; no untouched outer targets read here'},
              'sources': {role: {'directory': arm['path'], 'sha256': arm['hashes']}
                          for role, arm in arms.items()}}
    if output is not None:
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--static', required=True)
    parser.add_argument('--matched-static')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.prior, args.audio, args.static, args.output,
                   matched_static_dir=args.matched_static)
    print(json.dumps({'accepted': result['accepted'], 'failures': result['failures'],
                      'output': str(Path(args.output).resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
