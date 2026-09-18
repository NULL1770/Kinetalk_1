"""Fit a reference-controlled continuous stochastic prior and save diagnostics.

This is explicitly extra-condition generation, not an audio timing predictor.
Motion targets are restricted to fitting, independent reference encoding, and
post-generation scoring. The sampling API sees only level, style and a seed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import controlled_motion_process as motion
from scripts import train_clocked_motion_prior as c
from scripts import joint_motion_dictionary as coordinates

SCHEMA = 'independent_reference_continuous_prior_v1'
SEEDS = (42, 123, 2026, 77, 91, 301, 509, 997)
GAINS = (0., .5, 1., 1.5)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, data):
    def convert(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=convert)+'\n', encoding='utf8')


def picks_by_metadata(clips, ids):
    names = sorted({'_'.join(clips[i]['clip_id'].split('_')[:2]) for i in ids})
    names = [name for name in names if name.startswith('mead_M')][:2]+[name for name in names if name.startswith('mead_W')][:2]
    result = []
    for name in names:
        for emotion in (0, 1, 5, 6):
            pool = [i for i in ids if clips[i]['clip_id'].startswith(name+'_') and clips[i]['emotion'] == emotion]
            result += sorted(pool, key=lambda i: clips[i]['clip_id'])[:2]
    if not result:
        raise ValueError('No fixed metadata queries')
    return result


def choose_references(item, clips, fit_ids):
    """Same-person references; predicted emotion is an AUDIO output, not GT."""
    candidates = [i for i in fit_ids if clips[i]['speaker'] == item['speaker']
        and clips[i]['sentence'] != item['sentence'] and clips[i]['clip_id'] != item['clip_id']]
    same_affect = [i for i in candidates if clips[i]['emotion'] == item['predicted_emotion']]
    matched = len({clips[i]['sentence'] for i in same_affect}) >= 2
    candidates = same_affect if matched else candidates
    candidates = sorted(candidates, key=lambda i: clips[i]['clip_id'])
    if len(candidates) < 2:
        raise ValueError('Two independent fit references are required')
    first = candidates[0]
    second = next((i for i in candidates[1:] if clips[i]['sentence'] != clips[first]['sentence']), None)
    if second is None:
        raise ValueError('Reference swap must use another sentence')
    return (first, second), matched


def rollout(level, style, fitted, valid, seed, key, gain=1.):
    valid = np.asarray(valid, bool); values = np.zeros((len(valid), 9))
    for left, right in c.old.runs(valid):
        state = motion.initialize(level, style, fitted, seed, f'{key}:run:{left}',
                                  stationary=True, activity_gain=gain)
        values[left:right] = motion.sample(state, right-left)
    return values


def trajectory_stats(raw):
    x = np.asarray(raw, np.float64)
    centered = x-x.mean(0)
    velocity = np.diff(x, axis=0)*25.
    acceleration = np.diff(x, n=2, axis=0)*25.**2
    rms = [float(np.sqrt(np.mean(centered[:, group]**2))) for group in motion.GROUPS]
    power = np.abs(np.fft.rfft(centered, axis=0))**2
    freq = np.fft.rfftfreq(len(x), d=.04)
    total = power[1:].sum()
    ac = []
    for lag in (1, 4, 8, 16):
        a, b = centered[:-lag], centered[lag:]
        denom = float(np.sqrt(np.sum(a*a)*np.sum(b*b)))
        ac.append(float(np.sum(a*b)/denom) if denom > 1e-12 else None)
    activity = np.linalg.norm(velocity, axis=1) > .05
    runs = []
    edges = np.flatnonzero(np.diff(np.r_[False, activity, False]))
    runs = ((edges[1::2]-edges[::2])*.04).tolist()
    return {'group_rms': rms, 'velocity_p95': float(np.quantile(np.abs(velocity), .95)),
        'acceleration_p95': float(np.quantile(np.abs(acceleration), .95)),
        'above_3hz_energy_fraction': float(power[freq > 3].sum()/total) if total > 1e-14 else 0.,
        'autocorrelation_lags_1_4_8_16': ac, 'activity_fraction_speed_norm_gt_005': float(activity.mean()),
        'activity_run_seconds': runs, 'finite': bool(np.isfinite(x).all()),
        'in_domain': bool(((x >= 0) & (x <= 1)).all())}


def long_controls(level, styles, fitted, key):
    records, arrays = [], {}
    for seed in SEEDS:
        for gain in GAINS:
            state = motion.initialize(level, styles[0], fitted, seed, key+':long', activity_gain=gain)
            raw = motion.sample(state, 1500)
            records.append({'seed': seed, 'gain': gain, 'kind': 'stationary_60s', **trajectory_stats(raw),
                'quarter_group_rms': [trajectory_stats(part)['group_rms'] for part in np.array_split(raw, 4)]})
            if seed == 42: arrays['gain_'+str(gain)] = raw.astype(np.float32)
        state = motion.initialize(level, styles[0], fitted, seed, key+':controls')
        chunks = [motion.sample(state, 128), motion.sample(state, 64, mode='hold'),
                  motion.sample(state, 64, mode='release'), motion.sample(state, 128, mode='run'),
                  motion.sample(state, 128, mode='run', style=styles[1])]
        raw = np.concatenate(chunks)
        hold = raw[128+8:192]; release = raw[192+16:256]
        hold_error = float(np.max(np.abs(np.diff(hold, axis=0))))
        release_error = float(np.max(np.abs(release-expit(level))))
        velocity = np.abs(np.diff(raw, axis=0))*25.
        acceleration = np.abs(np.diff(raw, n=2, axis=0))*25.**2
        # Pre-fixed +/-8-frame neighborhoods include both sides of switches.
        edge_indices = np.unique(np.concatenate([np.arange(t-8, t+8) for t in (128, 192, 256, 384)]))
        records.append({'seed': seed, 'kind': 'run_hold_release_resume_swap',
            'hold_speed_max': hold_error, 'release_equilibrium_max_error': release_error,
            'boundary_velocity_p95': float(np.quantile(velocity[edge_indices], .95)),
            'boundary_acceleration_p95': float(np.quantile(acceleration[edge_indices], .95)),
            'control_times_frames': [128, 192, 256, 384], **trajectory_stats(raw)})
        if seed == 42: arrays['controls'] = raw.astype(np.float32)
    return records, arrays


def fit_dynamics_reference(clips):
    """Clip-equal 32-frame native-clock diagnostic distribution; no tuning."""
    rows = []
    for item in clips:
        raw = np.asarray(item['upper']); valid = np.asarray(item['valid'], bool)
        windows = []
        for left, right in c.old.runs(valid):
            for start in range(left, right-31, 32):
                windows.append(trajectory_stats(raw[start:start+32]))
        if windows:
            rows.append({'clip_id': item['clip_id'], 'windows': len(windows), **{
                key: float(np.mean([w[key] for w in windows])) for key in
                ('velocity_p95', 'acceleration_p95', 'above_3hz_energy_fraction')}})
    if not rows: raise ValueError('No fit-native32-frame diagnostics')
    return {'window_frames': 32, 'clip_equal': True, 'rows': rows, 'quantiles': {
        key: {str(q): float(np.quantile([r[key] for r in rows], q)) for q in (.1, .5, .9, .95)}
        for key in ('velocity_p95', 'acceleration_p95', 'above_3hz_energy_fraction')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'targets', 'native-root', 'native-manifest', 'delta-dir', 'audio-checkpoint', 'source-run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists(): raise FileExistsError('Fresh output required')
    args.output.mkdir(parents=True); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    save(args.output/'status.json', {'schema': SCHEMA, 'status': 'loading'})
    source = read(args.source_run/'protocol.json'); status = read(args.source_run/'status.json')
    if source['schema'] != 'clocked_bounded_static_residual_v1' or status['status'] != 'complete' or status['smoke']:
        raise ValueError('Completed bounded source required')
    root = Path(__file__).resolve().parents[1]
    for relative, digest in source['code_sha256'].items():
        if c.old.sha(root/relative) != digest: raise ValueError('Source dependency differs: '+relative)
    loading = copy.copy(args); loading.smoke = False
    clips, original, lineage = c.old.load_clips(loading)
    split = c.previous.split_inner(clips, original)
    if lineage != source['source']: raise ValueError('Data lineage differs')
    for cell in ('fit', 'calibration', 'confirmation'):
        if [{k: clips[i][k] for k in ('clip_id', 'sentence', 'speaker', 'emotion')} for i in split[cell]] != source['split'][cell]:
            raise ValueError('Source split differs')
    # Both modes fit all1053; smoke only reduces query count, preserving refs.
    fit_ids = split['fit']
    query_ids = picks_by_metadata(clips, split['confirmation'])
    if args.smoke: query_ids = query_ids[:2]
    # Lock query metadata before fitting/reference encoding/scoring.
    save(args.output/'query_selection.json', {'rule': 'First2 male/first2 female lexicographic inner-dev speakers; first2 IDs per emotion',
        'queries': [{k: clips[i][k] for k in ('clip_id', 'speaker', 'sentence', 'emotion')} for i in query_ids]})
    audio = c.load_frozen_audio(args.audio_checkpoint, args.device)
    binding = c.assert_source_binding(audio, lineage)
    if binding != source['frozen_audio']: raise ValueError('Audio binding differs')
    # Keep source batch membership/padding: near-zero ridge outputs can expose
    # small GPU batch-shape roundoff when comparing a newly packed query set.
    source_ids = sorted(set(i for cell in ('fit', 'calibration', 'confirmation') for i in split[cell]))
    source_contexts = c.encode_clips(audio, [clips[i] for i in source_ids], batch_size=16, device=args.device)
    by_index = dict(zip(source_ids, source_contexts))
    contexts = [by_index[i] for i in query_ids]
    with torch.no_grad():
        for i, context in zip(query_ids, contexts):
            clips[i]['global'] = np.r_[context['global'].numpy(), context['intensity'].numpy()]
            clips[i]['predicted_emotion'] = int(audio.emotion_classifier(context['global'][None].to(args.device)).argmax(-1)[0])
    del audio, contexts, source_contexts, by_index
    anchors = torch.load(args.targets, map_location='cpu', weights_only=False, mmap=True)['splits']['train']
    by_id = {cid: row[c.previous.CC].numpy() for cid, row in zip(anchors['clip_id'], anchors['anchors'])}
    for i in query_ids: clips[i]['anchor_upper'] = by_id[clips[i]['clip_id']]
    del anchors
    fitted = motion.fit_prior([clips[i] for i in fit_ids])
    fit_reference = fit_dynamics_reference([clips[i] for i in fit_ids])
    save(args.output/'fit_dynamics_reference.json', fit_reference)
    torch.save(fitted, args.output/'prior.pt')
    save(args.output/'fit_summary.json', fitted)
    equilibrium = torch.load(args.source_run/'equilibrium.pt', map_location='cpu', weights_only=False)
    scales = np.asarray(read(args.source_run/'scales.json'))
    low_cut = float(read(args.source_run/'training_data.json')['low_activity_train_quantile20'])
    source_curves = torch.load(args.source_run/'confirmation_static_trained.pt', map_location='cpu', weights_only=False)
    files = sorted(set(source['code_sha256']) | {'scripts/run_controlled_prior.py', 'scripts/controlled_motion_process.py'})
    protocol_file = root/'docs/SUPERVISION_NATURAL_PRIOR_PROTOCOL_20260918.md'
    protocol = {'schema': SCHEMA, 'source': lineage, 'frozen_audio': binding, 'smoke': args.smoke,
        'fit_clip_ids': [clips[i]['clip_id'] for i in fit_ids], 'query_clip_ids': [clips[i]['clip_id'] for i in query_ids],
        'source_run': str(args.source_run), 'source_files_sha256': {name: c.old.sha(args.source_run/name) for name in
        ('protocol.json', 'status.json', 'equilibrium.pt', 'scales.json', 'training_data.json', 'confirmation_static_trained.pt')},
        'code_sha256': {f: c.old.sha(root/f) for f in files}, 'protocol_sha256': c.old.sha(protocol_file),
        'extra_expression_reference': True, 'audio_timing_claim': False, 'default_replaced': False,
        'test_loaded': False, 'dev405_loaded': False, 'seeds': list(SEEDS), 'activity_gains': list(GAINS)}
    save(args.output/'protocol.json', protocol)
    for relative in files:
        dest = args.output/'source'/relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(root/relative, dest)
    shutil.copyfile(protocol_file, args.output/'source/protocol.md')
    rows = {arm: [] for arm in ('medoid', 'process', 'reference_swap')}
    curves, references, long_rows, long_arrays = {}, {}, {}, {}
    for i in query_ids:
        item = clips[i]; cid = item['clip_id']; valid = item['valid'].numpy()
        pair, matched = choose_references(item, clips, fit_ids)
        encoded = [motion.encode_reference(clips[j], fitted) for j in pair]
        styles = [e['style'] for e in encoded]
        references[cid] = {'references': encoded, 'reference_ids': [clips[j]['clip_id'] for j in pair],
            'predicted_emotion': item['predicted_emotion'], 'reference_emotion_matched': matched,
            'reference_metadata': [{k: clips[j][k] for k in ('clip_id', 'sentence', 'speaker', 'emotion')} for j in pair]}
        level = c.equilibrium(item['global'], coordinates.logit(item['anchor_upper']), equilibrium)
        baseline = source_curves[cid]
        if not np.array_equal(valid, baseline['valid']) or not np.allclose(level, baseline['logit_level'], atol=1e-6):
            raise ValueError(f'Frozen source level/clock mismatch: {cid}, level maxabs={np.max(np.abs(level-baseline["logit_level"]))}')
        variants = {'medoid': baseline['samples']}
        for arm, style in zip(('process', 'reference_swap'), styles):
            variants[arm] = np.stack([rollout(level, style, fitted, valid, seed, cid) for seed in SEEDS])
        controls = {str(g): rollout(level, styles[0], fitted, valid, 42, cid, g).astype(np.float32) for g in GAINS}
        for arm, samples in variants.items():
            score = c.metrics.score_clip(samples, item['upper'], valid, scales)
            rows[arm].append({**{k: item[k] for k in ('clip_id', 'sentence', 'speaker', 'emotion')},
                'low_activity': bool(np.mean((c.old.center_runs(item['upper'], valid)[valid]/scales)**2) <= low_cut),
                'acceleration': c.acceleration_stats(samples, item['upper'], valid, scales), **score})
        curves[cid] = {'samples': {k: np.asarray(v, np.float32) for k, v in variants.items()},
            'gain_controls_seed42': controls, 'target': item['upper'], 'valid': valid, 'level': level,
            'metadata': {k: item[k] for k in ('clip_id', 'sentence', 'speaker', 'emotion')}}
        # One metadata-first query per selected person for all8seeds60s diagnostics.
        speaker_key = str(item['speaker'])
        if speaker_key not in long_rows:
            long_rows[speaker_key], long_arrays[cid] = long_controls(level, styles, fitted, cid)
            for row in long_rows[speaker_key]:
                if row['kind'] == 'run_hold_release_resume_swap':
                    row['boundary_to_fit_median_ratio'] = {
                        key: row['boundary_'+key]/max(fit_reference['quantiles'][key]['0.5'], 1e-12)
                        for key in ('velocity_p95', 'acceleration_p95')}
        print('GENERATED', cid, len(valid), references[cid]['reference_ids'], flush=True)
    save(args.output/'references.json', references)
    save(args.output/'reports.json', {arm: c.report_rows(values) for arm, values in rows.items()})
    save(args.output/'long_controls.json', long_rows)
    torch.save({'schema': SCHEMA, 'curves': curves, 'long': long_arrays}, args.output/'predictions.pt')
    control_rows = [row for group in long_rows.values() for row in group if row['kind'] == 'run_hold_release_resume_swap']
    control_exact = all(row['hold_speed_max'] < 1e-12 and row['release_equilibrium_max_error'] < 1e-6 for row in control_rows)
    all_finite = all(row['finite'] and row['in_domain'] for group in long_rows.values() for row in group)
    save(args.output/'status.json', {'schema': SCHEMA, 'status': 'complete', 'seconds': time.monotonic()-started,
        'smoke': args.smoke, 'queries': len(query_ids), 'fit_clips': len(fit_ids), 'control_exact': control_exact,
        'finite_bounded': all_finite, 'extra_expression_reference': True, 'audio_timing_proven': False,
        'perceptual_naturalness_verified': False, 'default_replaced': False, 'test_loaded': False})


if __name__ == '__main__':
    main()
