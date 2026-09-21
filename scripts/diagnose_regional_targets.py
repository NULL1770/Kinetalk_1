"""TRAIN-only target semantics and fixed regional-envelope ridge diagnostics.

No renderer, validation/test tensor, or neural training is loaded. The four
speaker/sentence cells are reported, never used to select hyperparameters.
The linear probes predict the current TWO-region activity target, not the
historical signed four-state/ray targets. No probe is a production model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES
from kinetalk_b0.models.regional_intensity_gain import regional_envelope

UPPER = list(UPPER_INDICES)
CELLS = ('fit', 'seen_speaker_new_sentence', 'new_speaker_seen_sentence', 'new_speaker_new_sentence')
SEED, ALPHA, PCA_RANK, WINDOW = 20260922, 10., 24, 5
OFFSETS = (-2, -1, 0, 1, 2)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def runs(valid):
    left = None
    result = []
    for position, flag in enumerate(valid.tolist() + [False]):
        if flag and left is None:
            left = position
        elif not flag and left is not None:
            result.append((left, position)); left = None
    return result


def train_inventory(directory):
    """Inspect metadata, then whitelist TRAIN paths before any tensor reads."""
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    index = json.loads((root / 'index.json').read_text(encoding='utf8'))
    unsigned = dict(manifest); declared_hash = unsigned.pop('manifest_sha256')
    if (canonical_hash(unsigned) != declared_hash
            or manifest.get('status') != 'approved_train_val_only'
            or manifest.get('sealed_test_targets_loaded') is not False
            or index.get('test_loaded') is not False
            or index['recipe']['manifest_sha256'] != declared_hash):
        raise ValueError('Approved manifest/index binding differs')
    role = manifest['roles']['train']
    expected = {}
    for kind in ('query', 'enrollment'):
        for row in role[kind]:
            if row.get('source_split') != 'train' or row['clip_id'] in expected:
                raise ValueError('TRAIN role must contain unique TRAIN clips')
            expected[row['clip_id']] = (kind, row)
    query_pairs = {(row['speaker'], row['sentence']) for row in role['query']}
    if any((row['speaker'], row['sentence']) in query_pairs for row in role['enrollment']):
        raise ValueError('Neutral enrollment and TRAIN query speaker/sentence overlap')
    records = []
    for rec in index['records']:
        if rec['role'] != 'train':
            continue  # No val/test path is resolved, hashed or opened.
        if rec['clip_id'] not in expected or rec['kind'] != expected[rec['clip_id']][0]:
            raise ValueError('TRAIN index member differs from manifest')
        records.append({**rec, 'row': expected[rec['clip_id']][1]})
    if len(records) != len(expected) or len({r['clip_id'] for r in records}) != len(expected):
        raise ValueError('TRAIN index coverage differs')
    return root, records, declared_hash


def load_train_shard(root, rec):
    if rec['role'] != 'train' or rec['row']['source_split'] != 'train':
        raise ValueError('Only TRAIN tensor reads are permitted')
    path = (root / rec['path']).resolve()
    if not path.is_relative_to(root) or digest(path) != rec['sha256']:
        raise ValueError('TRAIN shard path/hash mismatch: ' + rec['clip_id'])
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['row'] != rec['row']:
        raise ValueError('TRAIN shard metadata mismatch')
    valid, motion = saved['valid'], saved['motion']
    if (valid.dtype != torch.bool or valid.ndim != 1 or not valid.any()
            or motion.shape != (len(valid), 52) or not motion.is_floating_point()
            or len(valid) != rec['frames'] or int(valid.sum()) != rec['valid_frames']):
        raise ValueError('TRAIN motion/mask dimensions differ')
    if not saved['channel_mask'][UPPER].all() or not torch.isfinite(motion[valid][:, UPPER]).all():
        raise ValueError('All nine observed upper channels are required')
    times = saved['times'].double()
    if times.shape != valid.shape or not torch.isfinite(times).all() or (len(times) > 1 and not torch.allclose(
            times[1:] - times[:-1], torch.full_like(times[1:], .04), atol=1e-5, rtol=0)):
        raise ValueError('Expected native 25-Hz frame clock')
    return saved


def split_cells(rows):
    """Identical speaker/sentence sets to native runner _fold; keep mixed cells."""
    speakers = sorted({r['speaker'] for r in rows})
    sentences = sorted({r['sentence'] for r in rows})
    held_s, held_t = set(speakers[::5]), set(sentences[::4])
    cells = {}
    for row in rows:
        key = (row['speaker'] in held_s, row['sentence'] in held_t)
        cells[row['clip_id']] = {(False, False): CELLS[0], (False, True): CELLS[1],
                                (True, False): CELLS[2], (True, True): CELLS[3]}[key]
    return cells, {'held_speakers': sorted(held_s), 'held_sentences': sorted(held_t)}


def center(upper, valid):
    safe = torch.where(valid[:, None], upper, 0.)
    mean = safe.sum(0, keepdim=True) / valid.sum().clamp_min(1)
    return torch.where(valid[:, None], safe - mean, 0.)


def smooth_fields(value, valid):
    result = torch.zeros_like(value)
    for left, right in runs(valid):
        x = value[left:right].T[None]
        kernel = x.new_ones(2, 1, WINDOW)
        numerator = F.conv1d(x, kernel, padding=WINDOW // 2, groups=2)
        support = F.conv1d(x.new_ones(1, 1, right - left), x.new_ones(1, 1, WINDOW), padding=WINDOW // 2)
        result[left:right] = (numerator / support)[0].T
    return result


def target_fields(upper, valid, anchor, scales):
    """Same units/fit scales; only the centering or physical quantity differs."""
    current = regional_envelope(center(upper, valid)[None], valid[None], window=WINDOW,
                                 channel_scales=scales)[0]
    relative = torch.where(valid[:, None], upper - anchor, 0.)
    neutral = regional_envelope(relative[None], valid[None], window=WINDOW,
                                 channel_scales=scales)[0]
    pair_valid = valid.clone(); pair_valid[0] = False
    pair_valid[1:] &= valid[:-1]
    safe = torch.where(valid[:, None], upper, 0.)
    velocity = torch.zeros_like(upper)
    velocity[1:] = (safe[1:] - safe[:-1]) / .04 / scales
    velocity = torch.where(pair_valid[:, None], velocity, 0.)
    square_energy = torch.stack((velocity[:, :5].square().mean(-1), velocity[:, 5:].square().mean(-1)), -1)
    speed = smooth_fields(square_energy, pair_valid).clamp_min(0).sqrt()
    return {'clip_centered_rms': (current, valid), 'neutral_relative_rms': (neutral, valid),
            'velocity_rms_per_second': (speed, pair_valid)}


def crop_diagnostics(upper, valid, anchor, scales):
    eligible = [(left, right) for left, right in runs(valid) if right - left >= 16]
    if not eligible:
        return None
    left, right = max(eligible, key=lambda r: r[1] - r[0])
    margin = (right - left) // 4
    crop_valid = torch.zeros_like(valid); crop_valid[left + margin:right - margin] = True
    full = target_fields(upper, valid, anchor, scales)
    cropped = target_fields(upper, crop_valid, anchor, scales)
    # Exclude derivative/filter boundary changes. The retained frames refer
    # to exactly the same source motion; only the clip support changed.
    interior = torch.zeros_like(valid); interior[left + margin + 3:right - margin - 3] = True
    if not interior.any():
        return None
    report = {}
    for name, (target, mask) in full.items():
        changed, changed_mask = cropped[name]
        keep = interior & mask & changed_mask
        error = changed[keep] - target[keep]
        denominator = target[keep].square().mean().sqrt().clamp_min(1e-8)
        report[name] = {'mse': float(error.square().mean()), 'mae': float(error.abs().mean()),
                        'relative_rmse': float(error.square().mean().sqrt() / denominator),
                        'common_interior_frames': int(keep.sum())}
    return {'crop_bounds': [left + margin, right - margin], 'targets': report}


def audio_features(saved):
    features = torch.cat((saved['content'].float(), saved['middle'].float(), saved['prosody'].float()), -1)
    if features.shape != (len(saved['valid']), 1540) or not torch.isfinite(features[saved['valid']]).all():
        raise ValueError('Expected finite TRAIN content768+middle768+prosody4')
    return torch.where(saved['valid'][:, None], features, 0.)


def ordered_context(x, valid):
    """Five native frame offsets; edge replication stays inside valid runs."""
    out = x.new_zeros(len(x), x.shape[-1] * len(OFFSETS))
    for left, right in runs(valid):
        positions = torch.arange(left, right)
        out[left:right] = torch.cat([x[(positions + offset).clamp(left, right - 1)] for offset in OFFSETS], -1)
    return out


def reverse_features(x, valid):
    out = torch.zeros_like(x)
    for left, right in runs(valid):
        out[left:right] = x[left:right].flip(0)
    return out


def donor_features(x, source_valid, target_valid):
    """Map the longest continuous donor run to each recipient run, features only."""
    left, right = max(runs(source_valid), key=lambda r: r[1] - r[0])
    segment = x[left:right].T[None]
    out = x.new_zeros(len(target_valid), x.shape[-1])
    for start, end in runs(target_valid):
        out[start:end] = F.interpolate(segment, size=end - start, mode='linear', align_corners=False)[0].T
    return out


def fit_ridge(examples, static=False):
    """Equal clip weighted least squares, fixed alpha10, unpenalized intercept."""
    widths = examples[0][0].shape[-1]
    means = torch.stack([x.double().mean(0) for x, _ in examples])
    second = torch.stack([x.double().square().mean(0) for x, _ in examples])
    mean = means.mean(0); scale = (second.mean(0) - mean.square()).clamp_min(0).sqrt().clamp_min(1e-5)
    gram = torch.zeros(widths + 1, widths + 1, dtype=torch.float64)
    rhs = torch.zeros(widths + 1, 2, dtype=torch.float64)
    for x, y in examples:
        z = (x.double() - mean) / scale
        design = torch.cat((z, torch.ones(len(z), 1, dtype=torch.float64)), -1)
        weight = 1. / len(design)
        gram += weight * design.T @ design
        rhs += weight * design.T @ y.double()
    regularizer = torch.eye(widths + 1, dtype=torch.float64) * ALPHA; regularizer[-1, -1] = 0
    coefficient = torch.linalg.solve(gram + regularizer, rhs)
    return {'mean': mean, 'scale': scale, 'coefficient': coefficient, 'static': static}


def ridge_predict(model, x):
    z = (x.double() - model['mean']) / model['scale']
    return torch.cat((z, torch.ones(len(z), 1, dtype=z.dtype)), -1) @ model['coefficient']


def metrics(prediction, target, valid):
    x, y = prediction[valid].double(), target[valid].double()
    xc, yc = x - x.mean(0), y - y.mean(0)
    denominator = (xc.square().sum(0) * yc.square().sum(0)).sqrt()
    correlation = [float((xc[:, k] * yc[:, k]).sum() / denominator[k]) if denominator[k] > 1e-10 else None for k in range(2)]
    defined = [r for r in correlation if r is not None]
    return {'mse': float((x - y).square().mean()), 'centered_mse': float((xc - yc).square().mean()),
            'target_centered_energy': float(yc.square().mean()),
            'correlation': float(np.mean(defined)) if defined else None, 'region_correlation': correlation,
            'temporal_std': float(x.std(0, unbiased=False).mean()),
            'target_temporal_std': float(y.std(0, unbiased=False).mean()),
            'negative_prediction_fraction': float((x < 0).double().mean())}


def summarize(items):
    if not items:
        return {'clips': 0}
    answer = {'clips': len(items), 'reduction': 'equal clip, undefined constant correlations excluded'}
    for key in ('mse', 'centered_mse', 'correlation', 'temporal_std', 'target_temporal_std', 'negative_prediction_fraction'):
        values = [item[key] for item in items if item[key] is not None]
        answer[key] = float(np.mean(values)) if values else None
    energy = float(np.mean([item['target_centered_energy'] for item in items]))
    answer['centered_r2'] = 1 - answer['centered_mse'] / energy if energy > 1e-12 else None
    return answer


def select_examples(records, count):
    """Round-robin speaker/emotion metadata groups, never sort by a score."""
    groups = {}
    for i, item in sorted(enumerate(records), key=lambda pair: pair[1]['clip_id']):
        groups.setdefault(item['speaker'], {}).setdefault(item['emotion'], []).append(i)
    selected = []
    cycle = 0
    while len(selected) < min(count, len(records)):
        for speaker_index, speaker in enumerate(sorted(groups)):
            emotions = sorted(groups[speaker])
            for offset in range(len(emotions)):
                emotion = emotions[(speaker_index + cycle + offset) % len(emotions)]
                if groups[speaker][emotion]:
                    selected.append(groups[speaker][emotion].pop(0))
                    break
            if len(selected) == min(count, len(records)):
                break
        cycle += 1
    return selected


def run(args):
    started = time.monotonic(); torch.set_num_threads(args.threads); torch.manual_seed(SEED)
    root, inventory, manifest_hash = train_inventory(args.data)
    if args.output.exists():
        raise FileExistsError('Use a fresh diagnostic output directory')
    args.output.mkdir(parents=True)
    queries = [rec for rec in inventory if rec['kind'] == 'query']
    enrollment = [rec for rec in inventory if rec['kind'] == 'enrollment']
    cells, fold = split_cells([rec['row'] for rec in queries])
    if not any(cell == 'fit' for cell in cells.values()):
        raise ValueError('TRAIN fitting cell is empty')
    reference_means, reference_sentences, read_log = {}, {}, []
    for rec in enrollment:
        row = rec['row']
        if row['emotion'] != 0:
            raise ValueError('Neutral enrollment required')
        if row['sentence'] in reference_sentences.setdefault(row['speaker'], set()):
            raise ValueError('Repeated enrollment sentence')
        reference_sentences[row['speaker']].add(row['sentence'])
        saved = load_train_shard(root, rec); read_log.append(rec['clip_id'])
        reference_means.setdefault(row['speaker'], []).append(saved['motion'][saved['valid']][:, UPPER].float().mean(0))
    anchors = {}
    for speaker in {q['row']['speaker'] for q in queries}:
        if len(reference_means.get(speaker, [])) < 2:
            raise ValueError('Two independent neutral references required: ' + speaker)
        anchors[speaker] = torch.stack(reference_means[speaker]).quantile(.5, dim=0)

    total = torch.zeros(1540, dtype=torch.float64); squares = torch.zeros_like(total)
    upper_square = torch.zeros(9, dtype=torch.float64); frame_count = 0
    records, pca_samples = [], []
    fit_count = sum(cell == 'fit' for cell in cells.values())
    per_clip = max(1, math.ceil(args.pca_frames / fit_count))
    for n, rec in enumerate(queries):
        saved = load_train_shard(root, rec); read_log.append(rec['clip_id'])
        features = audio_features(saved); valid = saved['valid']; upper = saved['motion'][:, UPPER].float()
        row = rec['row']; cell = cells[rec['clip_id']]
        records.append({'clip_id': row['clip_id'], 'speaker': row['speaker'], 'sentence': row['sentence'],
                        'emotion': row['emotion'], 'cell': cell, 'upper': upper, 'valid': valid,
                        'times': saved['times'].double(), 'anchor': anchors[row['speaker']]})
        if cell == 'fit':
            observed = features[valid].double()
            total += observed.sum(0); squares += observed.square().sum(0); frame_count += len(observed)
            upper_square += center(upper.double(), valid).square().sum(0)
            take = torch.linspace(0, len(observed) - 1, min(per_clip, len(observed))).round().long().unique()
            pca_samples.append(observed[take, :1536].float())
        if (n + 1) % 250 == 0:
            print(json.dumps({'stage': 'train_statistics', 'clips': n + 1}), flush=True)
    mean = (total / frame_count).float()
    std = (squares / frame_count - (total / frame_count).square()).clamp_min(0).sqrt().clamp_min(1e-4).float()
    channel_scales = (upper_square / frame_count).sqrt().clamp_min(.02).float()
    sampled = torch.cat(pca_samples); del pca_samples
    if len(sampled) > args.pca_frames:
        sampled = sampled[torch.randperm(len(sampled), generator=torch.Generator().manual_seed(SEED))[:args.pca_frames]]
    sampled = ((sampled - mean[:1536]) / std[:1536]).to(args.device)
    rank = min(PCA_RANK, len(sampled), 1536)
    _, _, projection = torch.pca_lowrank(sampled, q=rank, center=False, niter=2)
    projection = projection.cpu(); del sampled
    for n, (item, rec) in enumerate(zip(records, queries)):
        saved = load_train_shard(root, rec); read_log.append(rec['clip_id'])
        features = (audio_features(saved) - mean) / std
        item['pca_audio'] = torch.cat((features[:, :1536] @ projection, features[:, -4:]), -1)
        item['prosody'] = features[:, -4:]
        item['targets'] = target_fields(item['upper'], item['valid'], item['anchor'], channel_scales)
        item['crop'] = crop_diagnostics(item['upper'], item['valid'], item['anchor'], channel_scales)
        if (n + 1) % 250 == 0:
            print(json.dumps({'stage': 'train_projection', 'clips': n + 1}), flush=True)
    fit_ids = [i for i, r in enumerate(records) if r['cell'] == 'fit']
    fit_values = torch.cat([records[i]['targets']['clip_centered_rms'][0][records[i]['valid']] for i in fit_ids])
    envelope_scale = torch.from_numpy(np.quantile(fit_values.numpy(), .95, axis=0)).float().clamp_min(.05)
    for item in records:
        item['target'] = item['targets']['clip_centered_rms'][0] / envelope_scale
    example_ids = select_examples(records, args.example_clips); example_outputs = {i: {} for i in example_ids}

    groups = {cell: [i for i, item in enumerate(records) if item['cell'] == cell] for cell in CELLS}
    donors = {}
    for cell, members in groups.items():
        for i in members:
            candidates = [j for j in members if j != i]
            if candidates:
                donors[i] = min(candidates, key=lambda j: (
                    records[j]['sentence'] == records[i]['sentence'], records[j]['speaker'] != records[i]['speaker'],
                    records[j]['emotion'] != records[i]['emotion'], abs(len(records[j]['valid']) - len(records[i]['valid'])),
                    records[j]['clip_id']))
    all_results = {}
    for source in ('prosody', 'pca_audio'):
        dynamic_examples, static_examples = [], []
        for i in fit_ids:
            item = records[i]; valid = item['valid']; x = item[source]; y = item['target'][valid]
            dynamic_examples.append((ordered_context(x, valid)[valid], y))
            static_examples.append((x[valid].mean(0, keepdim=True), y.mean(0, keepdim=True)))
        dynamic_model = fit_ridge(dynamic_examples); static_model = fit_ridge(static_examples, static=True)
        del dynamic_examples, static_examples
        cell_results = {}
        for cell, members in groups.items():
            scored = {key: [] for key in ('full', 'reverse', 'mismatch', 'trained_static', 'own_static')}
            per_clip = []
            for i in members:
                item = records[i]; valid = item['valid']; x = item[source]
                conditions = {'full': x, 'reverse': reverse_features(x, valid)}
                if i in donors:
                    donor = records[donors[i]]
                    conditions['mismatch'] = donor_features(donor[source], donor['valid'], valid)
                outputs = {key: ridge_predict(dynamic_model, ordered_context(value, valid)) for key, value in conditions.items()}
                outputs['trained_static'] = ridge_predict(static_model, x[valid].mean(0, keepdim=True)).expand(len(x), -1)
                outputs['own_static'] = outputs['full'][valid].mean(0, keepdim=True).expand(len(x), -1)
                result = {key: metrics(value, item['target'], valid) for key, value in outputs.items()}
                for key, value in result.items():
                    scored[key].append(value)
                per_clip.append({'clip_id': item['clip_id'], 'speaker': item['speaker'], 'sentence': item['sentence'],
                                 'donor_clip_id': records[donors[i]]['clip_id'] if i in donors else None, 'conditions': result})
                if i in example_outputs:
                    example_outputs[i].update({source + '_' + key: value.float().numpy() for key, value in outputs.items()})
            cell_results[cell] = {'conditions': {key: summarize(value) for key, value in scored.items()}, 'per_clip': per_clip,
                                 'by_speaker': {speaker: {key: summarize([r['conditions'][key] for r in per_clip
                                            if r['speaker'] == speaker and key in r['conditions']]) for key in scored}
                                            for speaker in sorted({r['speaker'] for r in per_clip})}}
            cell_results[cell]['paired_mse_delta_full_minus_control'] = {
                key: float(np.mean([r['conditions']['full']['mse'] - r['conditions'][key]['mse']
                                    for r in per_clip if key in r['conditions']]))
                for key in ('reverse', 'mismatch', 'trained_static', 'own_static') if scored[key]}
        all_results[source] = cell_results
        print(json.dumps({'stage': 'ridge_completed', 'source': source}), flush=True)

    curve_dir = args.output / 'examples'; curve_dir.mkdir()
    examples = []
    for index in example_ids:
        item = records[index]
        payload = {'times': item['times'].numpy(), 'valid': item['valid'].numpy(), 'upper9': item['upper'].numpy(),
                   'neutral_anchor9': item['anchor'].numpy(), 'normalized_current_target': item['target'].numpy(),
                   'train_channel_scales': channel_scales.numpy(), 'envelope_scale': envelope_scale.numpy(),
                   **example_outputs[index]}
        for key, (value, mask) in item['targets'].items():
            payload[key] = value.numpy(); payload[key + '_valid'] = mask.numpy()
        if item['crop'] is not None:
            left, right = item['crop']['crop_bounds']
            mask = torch.zeros_like(item['valid']); mask[left:right] = True
            for key, (value, keep) in target_fields(item['upper'], mask, item['anchor'], channel_scales).items():
                payload['cropped_' + key] = value.numpy(); payload['cropped_' + key + '_valid'] = keep.numpy()
        path = curve_dir / (item['clip_id'] + '.npz'); np.savez_compressed(path, **payload)
        examples.append({'clip_id': item['clip_id'], 'speaker': item['speaker'], 'emotion': item['emotion'],
                         'cell': item['cell'], 'path': str(path.relative_to(args.output)), 'sha256': digest(path),
                         'crop': item['crop']})
    crop_report = {}
    for cell, members in groups.items():
        reports = [records[i]['crop']['targets'] for i in members if records[i]['crop'] is not None]
        crop_report[cell] = {name: {'clips': len(reports), **{
            key: float(np.mean([rec[name][key] for rec in reports])) for key in ('mse', 'mae', 'relative_rmse')}}
            for name in ('clip_centered_rms', 'neutral_relative_rms', 'velocity_rms_per_second')} if reports else {}
    protocol = {'schema': 'train_regional_target_diagnostic_v1', 'manifest_sha256': manifest_hash,
                'index_sha256': digest(root / 'index.json'),
                'script_sha256': digest(__file__), 'train_queries': len(queries), 'train_enrollment': len(enrollment),
                'fold': {**fold, 'cells': {cell: len(ids) for cell, ids in groups.items()}},
                'ridge_alpha': ALPHA, 'ridge_objective': 'sum of equal-clip average squared errors plus alpha*weight_norm_squared; intercept unpenalized',
                'feature_stats': 'fit-cell only, all native observed frames', 'pca_rank': rank,
                'pca_sample_budget': args.pca_frames, 'pca_sampling': 'evenly spaced fit frames per clip, deterministic seed cap',
                'dynamic_context_native_offsets': list(OFFSETS), 'target_window_frames': WINDOW,
                'probe_target': 'two-region clip-centered RMS, five-frame box smoothing; same definition as native runner',
                'diagnostic_targets': {'neutral_relative_rms': 'RMS from independent neutral enrollment, same fit channel scales; keeps sustained expression level',
                    'velocity_rms_per_second': 'sqrt of five-frame mean regional normalized squared finite-difference velocity; adjacent valid pairs only'},
                'reference_scope': 'TRAIN neutral enrollment; at least two unique reference sentences; no query/reference speaker-sentence overlap',
                'static_baseline': 'independently fitted pooled-audio ridge to clip mean target, repeated through time',
                'negative_outputs': 'raw ridge predictions retained; never clamped for score',
                'reverse': 'feature frames reversed within each contiguous valid run, not waveform reversal',
                'mismatch': 'nonself donor within same cell, prefer different sentence then same speaker/emotion; longest donor run resampled per recipient run',
                'examples_selection': 'metadata-only round-robin speakers, rotate emotions by speaker index, then clip_id order',
                'target_fit_scales': channel_scales.tolist(), 'target_envelope_scale': envelope_scale.tolist(),
                'validation_tensors_loaded': False, 'test_tensors_loaded': False, 'renderers_loaded': False,
                'cell_scores_used_for_hyperparameter_selection': False, 'default_model_changed': False,
                'read_tensor_clip_ids': sorted(set(read_log))}
    save_json(args.output / 'protocol.json', protocol)
    save_json(args.output / 'evaluation.json', {'protocol': protocol, 'crop_sensitivity': crop_report,
              'ridge': all_results, 'examples': examples, 'elapsed_seconds': time.monotonic() - started,
              'interpretation': 'TRAIN diagnostic only. Conditional correlation is not reliable audio control or generated-motion quality.'})
    save_json(args.output / 'status.json', {'status': 'complete', 'elapsed_seconds': time.monotonic() - started,
              'validation_tensors_loaded': False, 'test_tensors_loaded': False})
    print(json.dumps({'status': 'complete', 'output': str(args.output), 'seconds': time.monotonic() - started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', help='PCA execution device; all data inspection/ridge is CPU')
    parser.add_argument('--pca-frames', type=int, default=12000)
    parser.add_argument('--example-clips', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.pca_frames < PCA_RANK or not 12 <= args.example_clips <= 24 or args.threads < 1:
        parser.error('PCA needs at least 24 frames; example clips 12..24; positive threads')
    run(args)


if __name__ == '__main__':
    main()
