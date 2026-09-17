"""Read-only local-time and display-clamping audit of saved prefix trajectories.

All comparisons use saved seed-42 samples. Nothing is fitted, resampled,
smoothed, amplified, or used to select a checkpoint or a visual example.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import load_pt, metadata_equal, read, sha
from scripts.full_staged_data import _validate_split_lock
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_projection_schedule_ablation import read_allowlist


ARMS = ('frozen_local', 'adapt_local')
REGIONS = {'brows': (41, 42, 43, 44, 45), 'eyes_expression': (5, 6, 12, 13)}
EMOTIONS = ('neutral', 'angry', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise')
EPS = 1e-15


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_ratio(numerator, denominator, *, sqrt=False):
    if denominator <= EPS:
        return None
    value = numerator / denominator
    return float(max(value, 0.) ** .5 if sqrt else value)


def sufficient_statistics(pred, target, valid):
    """Per-clip sufficient statistics; only truly adjacent valid pairs count."""
    require(pred.shape == target.shape and pred.ndim == 3 and valid.shape == pred.shape[:2]
            and valid.dtype == torch.bool and bool(valid.any(1).all()), 'Invalid metric tensors')
    require(bool(torch.isfinite(pred[valid]).all() and torch.isfinite(target[valid]).all()),
            'Nonfinite observed motion')
    pred, target = pred.double(), target.double()
    mask = valid[..., None]
    count = valid.sum(1).double() * pred.shape[-1]
    clean = lambda value: torch.where(mask, value, 0.)
    mean = lambda value: clean(value).sum(1, keepdim=True) / valid.sum(1)[:, None, None]
    pc, tc = clean(pred - mean(pred)), clean(target - mean(target))
    pair = (valid[:, 1:] & valid[:, :-1])[..., None]
    pd = torch.where(pair, pred[:, 1:] - pred[:, :-1], 0.)
    td = torch.where(pair, target[:, 1:] - target[:, :-1], 0.)
    total = lambda value: value.sum((1, 2))
    fields = {
        'count': count, 'pair_count': pair.sum((1, 2)).double() * pred.shape[-1],
        'raw_sse': total(clean(pred - target).square()),
        'centered_sse': total((pc - tc).square()),
        'pred_energy': total(pc.square()), 'target_energy': total(tc.square()),
        'cross': total(pc * tc), 'velocity_sse': total((pd - td).square()),
        'pred_velocity_energy': total(pd.square()), 'target_velocity_energy': total(td.square()),
        'below_zero': total((pred < 0) & mask).double(),
        'above_one': total((pred > 1) & mask).double(),
        'at_lower_bound': total((pred <= 0) & mask).double(),
        'at_upper_bound': total((pred >= 1) & mask).double(),
    }
    return [{key: float(value[i]) for key, value in fields.items()} for i in range(len(pred))]


def finish_metrics(stats):
    count, pairs = stats['count'], stats['pair_count']
    pe, te = stats['pred_energy'], stats['target_energy']
    correlation = safe_ratio(stats['cross'], (pe * te) ** .5)
    return {
        'observed_values': int(count), 'observed_adjacent_values': int(pairs),
        'raw_mse': safe_ratio(stats['raw_sse'], count),
        'centered_mse': safe_ratio(stats['centered_sse'], count),
        'centered_correlation': correlation,
        'prediction_centered_rms': safe_ratio(pe, count, sqrt=True),
        'target_centered_rms': safe_ratio(te, count, sqrt=True),
        'rms_ratio': safe_ratio(pe, te, sqrt=True),
        'frame_displacement_mse': safe_ratio(stats['velocity_sse'], pairs),
        'prediction_velocity_rms': safe_ratio(stats['pred_velocity_energy'], pairs, sqrt=True),
        'target_velocity_rms': safe_ratio(stats['target_velocity_energy'], pairs, sqrt=True),
        'velocity_rms_ratio': safe_ratio(stats['pred_velocity_energy'], stats['target_velocity_energy'], sqrt=True),
        'below_zero_fraction': safe_ratio(stats['below_zero'], count),
        'above_one_fraction': safe_ratio(stats['above_one'], count),
        'outside_fraction': safe_ratio(stats['below_zero'] + stats['above_one'], count),
        'display_saturated_fraction': safe_ratio(stats['at_lower_bound'] + stats['at_upper_bound'], count),
    }


def pool(stats, indices):
    return {key: sum(stats[i][key] for i in indices) for key in stats[0]}


def describe(values):
    actual = [x for x in values if x is not None]
    return {'defined': len(actual), 'undefined': len(values) - len(actual),
            'mean': float(np.mean(actual)) if actual else None,
            'quantiles_p10_p25_p50_p75_p90': np.quantile(actual, [.1, .25, .5, .75, .9]).tolist() if actual else None}


def fraction(values, predicate):
    actual = [x for x in values if x is not None]
    return {'count': sum(bool(predicate(x)) for x in actual), 'denominator': len(actual),
            'fraction': sum(bool(predicate(x)) for x in actual) / len(actual) if actual else None}


def paired_benefits(full, static, indices):
    output = {}
    for name, lower in (('centered_mse', True), ('centered_correlation', False),
                        ('frame_displacement_mse', True)):
        changes = [(full[i][name] - static[i][name])
                   if full[i][name] is not None and static[i][name] is not None else None for i in indices]
        output[name] = {'full_minus_local_static': describe(changes),
                        'full_better': fraction(changes, (lambda x: x < 0) if lower else (lambda x: x > 0)),
                        'equal': fraction(changes, lambda x: x == 0)}
    return output


def make_groups(metadata):
    groups = {'all': list(range(len(metadata)))}
    for i, row in enumerate(metadata):
        for category in ('emotion', 'sentence_id', 'speaker'):
            groups.setdefault(category + '/' + str(row[category]), []).append(i)
    return groups


def analyze_curves(curves, metadata, *, require_local_static):
    """Analyze full and, where saved, local-only constant-time intervention."""
    predictions = curves['predictions']
    require('42/full' in predictions, 'Missing full seed-42 trajectory')
    has_static = '42/local_static' in predictions
    require(not require_local_static or has_static, 'Missing deployable local_static intervention')
    modes = ('full', 'local_static') if has_static else ('full',)
    valid, target = curves['valid'], curves['target']
    require(len(metadata) == len(valid), 'Metric metadata size differs')
    groups = make_groups(metadata)
    stats, metrics = {}, {}
    for mode in modes:
        pred = predictions['42/' + mode]
        require(pred.shape == target.shape, 'Prediction shape differs')
        for region, channels in REGIONS.items():
            require(bool(curves['channel_mask'][:, list(channels)].all()), 'Unobserved upper channel')
            values = pred[..., list(channels)]
            truth = target[..., list(channels)]
            for display, x in (('raw', values), ('display_clamped', values.clamp(0, 1))):
                key = mode + '/' + region + '/' + display
                stats[key] = sufficient_statistics(x, truth, valid)
                metrics[key] = [finish_metrics(row) for row in stats[key]]
    rows = []
    for i, meta in enumerate(metadata):
        row = {**meta, 'metrics': {key: value[i] for key, value in metrics.items()}, 'clamp_retention': {}}
        for mode in modes:
            for region in REGIONS:
                raw = stats[mode + '/' + region + '/raw'][i]
                clamped = stats[mode + '/' + region + '/display_clamped'][i]
                row['clamp_retention'][mode + '/' + region] = {
                    'centered_rms_retained': safe_ratio(clamped['pred_energy'], raw['pred_energy'], sqrt=True),
                    'velocity_rms_retained': safe_ratio(clamped['pred_velocity_energy'], raw['pred_velocity_energy'], sqrt=True)}
        rows.append(row)
    grouped = {}
    for name, indices in groups.items():
        summary = {'clips': len(indices), 'pooled': {key: finish_metrics(pool(value, indices)) for key, value in stats.items()},
                   'per_clip': {}, 'clamp_retention': {}, 'full_vs_local_static': {}}
        for key, records in metrics.items():
            summary['per_clip'][key] = {
                metric: describe([records[i][metric] for i in indices])
                for metric in ('centered_mse', 'centered_correlation', 'rms_ratio', 'velocity_rms_ratio', 'outside_fraction')}
            ratios = [records[i]['rms_ratio'] for i in indices]
            summary['per_clip'][key]['amplitude_above_GT'] = fraction(ratios, lambda x: x > 1.)
            summary['per_clip'][key]['amplitude_above_1_5x_GT'] = fraction(ratios, lambda x: x > 1.5)
        for mode in modes:
            for region in REGIONS:
                key = mode + '/' + region
                raw = pool(stats[key + '/raw'], indices)
                clamp = pool(stats[key + '/display_clamped'], indices)
                retention = [rows[i]['clamp_retention'][key]['centered_rms_retained'] for i in indices]
                summary['clamp_retention'][key] = {
                    'pooled_centered_rms_retained': safe_ratio(clamp['pred_energy'], raw['pred_energy'], sqrt=True),
                    'pooled_velocity_rms_retained': safe_ratio(clamp['pred_velocity_energy'], raw['pred_velocity_energy'], sqrt=True),
                    'per_clip_centered_rms_retained': describe(retention),
                    'clips_below_half_raw_rms': fraction(retention, lambda x: x < .5)}
        if has_static:
            for region in REGIONS:
                for display in ('raw', 'display_clamped'):
                    key = region + '/' + display
                    summary['full_vs_local_static'][key] = paired_benefits(
                        metrics['full/' + key], metrics['local_static/' + key], indices)
        grouped[name] = summary
    return {'clips': len(rows), 'seed': 42, 'local_static_available': has_static,
            'fit_intervention_limitation': None if has_static else 'No saved local_static; static also changes h0 and is not substituted.',
            'groups': grouped, 'per_clip': rows}


def bind_metadata(curves, query):
    lookup = {cid: i for i, cid in enumerate(query['clip_id'])}
    require(len(lookup) == len(query['clip_id']), 'Duplicate source IDs')
    require(len(set(curves['clip_id'])) == len(curves['clip_id']), 'Duplicate curve IDs')
    require(all(cid in lookup for cid in curves['clip_id']), 'Curve outside permitted split')
    ids = torch.tensor([lookup[cid] for cid in curves['clip_id']], dtype=torch.long)
    for curve_key, query_key in (('target', 'motion'), ('valid', 'valid'), ('times', 'times'),
                                  ('channel_mask', 'channel_mask'), ('emotion_id', 'emotion_id'), ('speaker_id', 'speaker_id')):
        require(torch.equal(curves[curve_key], query[query_key][ids]), 'Native metadata differs: ' + curve_key)
    times = curves['times']
    require(torch.allclose(times[:, 1:] - times[:, :-1], torch.full_like(times[:, 1:], .04), atol=1e-7, rtol=1e-5),
            'Expected native 25 Hz clock')
    rows = []
    for i, index in enumerate(ids.tolist()):
        sentence = query['sentence_id'][index]
        require(isinstance(sentence, str) and bool(sentence), 'Sentence metadata missing')
        emotion = int(curves['emotion_id'][i])
        require(0 <= emotion < len(EMOTIONS), 'Unknown emotion')
        rows.append({'clip_id': curves['clip_id'][i], 'sentence_id': sentence,
                     'speaker': str(query['speaker'][index]), 'speaker_id': int(curves['speaker_id'][i]),
                     'emotion_id': emotion, 'emotion': EMOTIONS[emotion]})
    return rows


def comparison_report(first, second):
    require([x['clip_id'] for x in first['per_clip']] == [x['clip_id'] for x in second['per_clip']],
            'Cross-arm clip order differs')
    result = {}
    for group in first['groups']:
        ids = make_groups(first['per_clip'])[group]
        result[group] = {}
        for region in REGIONS:
            key = 'full/' + region + '/raw'
            aa = [x['metrics'][key] for x in first['per_clip']]
            bb = [x['metrics'][key] for x in second['per_clip']]
            # This routine only computes paired numeric changes, not a causal attribution.
            record = paired_benefits(bb, aa, ids)
            for value in record.values():
                value['adapt_minus_frozen'] = value.pop('full_minus_local_static')
                value['adapt_better'] = value.pop('full_better')
            result[group][region] = record
    return result


def markdown(report):
    lines = ['# Saved Local Temporal Transfer Audit', '',
             'Seed 42, all saved internal-development clips. No fitting, inference, gain, smoothing, or sample selection.', '',
             '| Arm / composition | Region | Corr | RMS/GT | Clips RMS > GT | Full better than local-static (centered MSE) | Clamp RMS retained | Saturated |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    fmt = lambda x: 'undefined' if x is None else f'{x:.4f}'
    for arm in ARMS:
        for composition in ('raw', 'dc'):
            group = report['arms'][arm]['development_' + composition]['groups']['all']
            for region in REGIONS:
                key = 'full/' + region + '/raw'
                values = group['pooled'][key]
                above = group['per_clip'][key]['amplitude_above_GT']['fraction']
                better = group['full_vs_local_static'][region + '/raw']['centered_mse']['full_better']['fraction']
                retained = group['clamp_retention']['full/' + region]['pooled_centered_rms_retained']
                lines.append('| ' + ' | '.join([arm + '/' + composition, region, fmt(values['centered_correlation']),
                    fmt(values['rms_ratio']), fmt(above), fmt(better), fmt(retained), fmt(values['display_saturated_fraction'])]) + ' |')
    lines += ['', 'Fit diagnostics include the fixed 128 training clips. They do not contain local_static, so no fit local-time benefit is claimed.', '',
              'Pooled metrics sum clip-centered sufficient statistics; fractions count clips equally. Group tables and every clip are in report.json.', '',
              'Display clamping is inspected only as a diagnostic; saved curves and raw metrics are untouched. A retained coefficient RMS is not a perceptual motion rating.', '',
              'Local-static keeps h0, global audio, static means, identity, and seed unchanged; generated history can then diverge. This measures total rollout response, not a history-controlled direct effect.', '',
              'Sentence groups are descriptive and can share sentences across fit/development. These repeatedly used internal development data do not establish untouched-test generalization.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    run = args.root / 'audio_prefix12'
    output = args.output or run / 'local_temporal_transfer_audit'
    require(not output.exists(), 'Fresh report output required')
    records = {arm: read(run / arm / 'provenance.json') for arm in ARMS}
    data = records[ARMS[0]]['recipe']['data_provenance']
    require(data == records[ARMS[1]]['recipe']['data_provenance'], 'Paired source data provenance differs')
    paths = {key: Path(data['source_paths'][key]) for key in ('cache', 'fit_ids', 'validation_ids', 'split_lock')}
    hashes = {key: sha(path) for key, path in paths.items()}
    require(all(value == data['input_sha256'][key] for key, value in hashes.items()), 'Source metadata binding differs')
    cache = load_pt(paths['cache'])
    require(cache.get('schema') == 'predictable_renderer_cache_v1'
            and set(cache.get('splits', {})) == {'train', 'validation'}, 'Only train/validation metadata cache permitted')
    fit_ids, dev_ids = read_allowlist(paths['fit_ids']), read_allowlist(paths['validation_ids'])
    _validate_split_lock(cache, read(paths['split_lock']), fit_ids, dev_ids)
    require(len(fit_ids) == 2315 and len(dev_ids) == 405, 'Native population changed')
    source = args.root / 'context12/chunk_teacher/final.pt'
    source_complete = read(source.with_name('complete.json'))
    source_sha = sha(source)
    require(source_sha == source_complete['final_sha256'], 'Source checkpoint hash differs')
    fit_selection = read(run / 'fit_selection.json')
    report = {
        'schema': 'saved_local_temporal_transfer_audit_v1', 'seed': 42,
        'test_loaded': False, 'fitting_performed': False, 'inference_performed': False,
        'source_paths': {key: str(value) for key, value in paths.items()},
        'input_sha256': {**hashes, 'source_checkpoint': source_sha, 'audit_script': sha(__file__)},
        'scope': 'Existing internal speaker-holdout development, repeatedly used; sentence groups are descriptive.',
        'metric_definitions': {
            'centering': 'Each clip and channel centered over valid frames before pooling sufficient statistics.',
            'velocity': 'Native adjacent displacement per 0.04 seconds; missing-frame gaps are excluded.',
            'clamp': 'Prediction-only [0,1] clamp, matching display rig; GT remains original.',
            'undefined': 'Null correlation or RMS ratio when denominator energy <=1e-15; excluded from corresponding fraction denominator.',
            'local_static': 'Only local feature time is replaced by its valid mean; h0/global/static/identity/seed fixed. Autoregressive generated history may diverge.',
            'thresholds': 'RMS >1x and >1.5x GT and clamp retention <0.5 are descriptive bins, not success criteria or tuned gains.'},
        'arms': {}}
    for arm in ARMS:
        folder = run / arm
        record, complete = records[arm], read(folder / 'complete.json')
        recipe = record['recipe']
        digest = canonical_hash(recipe)
        require(digest == record['recipe_sha256'] == complete['recipe_sha256'], 'Recipe hash differs: ' + arm)
        require(recipe['source_sha256'] == source_sha
                and recipe['source_recipe_sha256'] == source_complete['recipe_sha256'], 'Warmstart binding differs')
        require(recipe['test_loaded'] is False and recipe['epochs'] == complete['completed_epochs'] == 12
                and complete['total_steps'] == 1740, 'Training contract differs')
        require(recipe['fit_selection_sha256'] == sha(run / 'fit_selection.json'), 'Fit selection binding differs')
        for name, digest in recipe['code_sha256'].items():
            require(sha(Path(__file__).resolve().parents[1] / name) == digest, 'Training source code changed: ' + name)
        loaded = {}
        for name in ('curves', 'dc_curves', 'fit_curves'):
            path = folder / (name + '.pt')
            require(sha(path) == complete[name + '_sha256'], 'Curve file hash differs: ' + arm + '/' + name)
            report['input_sha256'][arm + '/' + name] = complete[name + '_sha256']
            loaded[name] = load_pt(path)
        metadata_equal(loaded['curves'], loaded['dc_curves'])
        require(loaded['curves']['clip_id'] == dev_ids, 'Development order differs')
        require(loaded['fit_curves']['clip_id'] == [x['clip_id'] for x in fit_selection['clips']], 'Fit order differs')
        rows = {}
        for label, name, split in (('development_raw', 'curves', 'validation'),
                                   ('development_dc', 'dc_curves', 'validation'), ('fit_raw', 'fit_curves', 'train')):
            curves = loaded[name]
            metadata = bind_metadata(curves, cache['splits'][split]['q'])
            rows[label] = analyze_curves(curves, metadata, require_local_static=split == 'validation')
        report['arms'][arm] = rows
        print('AUDITED', arm, flush=True)
    report['adapt_vs_frozen'] = {label: comparison_report(report['arms']['frozen_local'][label], report['arms']['adapt_local'][label])
                               for label in ('development_raw', 'development_dc', 'fit_raw')}
    report['seconds'] = time.monotonic() - started
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')
    (output / 'report.md').write_text(markdown(report), encoding='utf8')
    (output / 'complete.json').write_text(json.dumps({'report_sha256': sha(output / 'report.json'),
        'markdown_sha256': sha(output / 'report.md'), 'seconds': report['seconds'], 'test_loaded': False}, indent=2) + '\n', encoding='utf8')
    print('LOCAL_TEMPORAL_TRANSFER_AUDIT_COMPLETE', str(output), flush=True)


if __name__ == '__main__':
    main()
