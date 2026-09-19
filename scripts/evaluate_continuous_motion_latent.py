"""Native-run reconstruction and freely sampled continuous-latent diagnostics.

Raw residual = (motion9-b9)/fit_residual_scale. No sigmoid, clipping, target
centering or target-driven amplitude calibration is applied to predictions.
Numerical checks never certify naturalness; every result needs visual review.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.joint_motion_metrics import score_clip, summarize
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES
from scripts.arkit_benchmark_report import build_report, score_fullface, write_report

SCHEMA = 'continuous_motion_latent_evaluation_v1'
UPPER = [41, 42, 43, 44, 45, 5, 6, 12, 13]
OTHER = [i for i in range(52) if i not in UPPER]


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def contiguous_runs(valid):
    """Half-open observed runs. Empty/missing regions are never connected."""
    valid = np.asarray(valid)
    if valid.ndim != 1 or valid.dtype != bool:raise ValueError('Boolean native mask required')
    edge = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(edge == 1).tolist(), np.flatnonzero(edge == -1).tolist()))


def _metadata(clip):
    embedded = clip.get('metadata', {})
    return {key: clip.get(key, embedded.get(key)) for key in ('clip_id', 'split', 'sentence', 'speaker', 'emotion')}


def _vector(stats, key, size, device):
    value = torch.as_tensor(stats[key], dtype=torch.float32, device=device)
    if value.shape != (size,) or not torch.isfinite(value).all():raise ValueError('invalid fit statistic: '+key)
    if ('scale' in key) and not (value > 0).all():raise ValueError('positive fit scale required: '+key)
    return value


def _audio_blocks(clip, start, stop, stats, block_size, device, intervention):
    raw = torch.as_tensor(_array(clip['features'])[start:stop], dtype=torch.float32, device=device)
    if raw.ndim != 2 or raw.shape[1] != 1540 or not torch.isfinite(raw).all():
        raise ValueError('native observed features must be finite [T,1540]')
    if intervention == 'static':raw = raw.mean(0, keepdim=True).expand_as(raw)
    elif intervention == 'reverse':raw = raw.flip(0)
    elif intervention != 'real':raise ValueError('intervention must be real/static/reverse')
    values = (raw-_vector(stats, 'audio_mean', 1540, device))/_vector(stats, 'audio_scale', 1540, device)
    blocks = (len(values)+block_size-1)//block_size
    packed = values.new_zeros(1, blocks*block_size, 1540)
    observed = torch.zeros(1, blocks*block_size, dtype=torch.bool, device=device)
    packed[0, :len(values)] = values; observed[0, :len(values)] = True
    return packed.reshape(1, blocks, block_size, 1540), observed.reshape(1, blocks, block_size)


def _seed(clip_id, run_index, seed):
    digest = hashlib.sha256(f'continuous_motion_latent:{seed}:{clip_id}:run{run_index}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2**63-1)


def _mismatched_audio(clips):
    """Metadata-fixed same-emotion/different-sentence donors, within each split.

    Only donor audio is read. Its longest valid run is linearly resampled onto
    each query audio run. This is a distribution-shifting diagnostic, not a
    matched-training baseline. Query context, b9, masks and RNG keys are kept.
    """
    replacements, records, excluded = {}, [], []
    for clip in clips:
        meta = _metadata(clip)
        donors = [c for c in clips if c['clip_id'] != clip['clip_id']
                  and _metadata(c)['split'] == meta['split']
                  and _metadata(c)['emotion'] == meta['emotion']
                  and _metadata(c)['sentence'] != meta['sentence']
                  and contiguous_runs(_array(c['valid']))]
        if not donors:
            excluded.append(clip['clip_id']); continue
        donor = min(donors, key=lambda c: (_metadata(c)['speaker'] != meta['speaker'], c['clip_id']))
        left, right = max(contiguous_runs(_array(donor['valid'])), key=lambda x: (x[1]-x[0], -x[0]))
        source = torch.as_tensor(_array(donor['features'])[left:right], dtype=torch.float32)
        if source.ndim != 2 or source.shape[1] != 1540 or not torch.isfinite(source).all():
            raise ValueError('Invalid mismatch donor audio')
        features = torch.as_tensor(_array(clip['features']), dtype=torch.float32).clone()
        for start, stop in contiguous_runs(_array(clip['valid'])):
            features[start:stop] = torch.nn.functional.interpolate(
                source.T[None], size=stop-start, mode='linear', align_corners=False)[0].T
        replacements[clip['clip_id']] = {**clip, 'features': features}
        records.append({'clip_id': clip['clip_id'], 'donor_clip_id': donor['clip_id'],
                        'donor_native_run': [left, right], 'same_speaker': _metadata(donor)['speaker'] == meta['speaker'],
                        'resampling': 'longest native donor run linearly resampled per query native run'})
    return replacements, records, excluded


@torch.no_grad()
def generate_clip(flow, ae, clip, stats, device='cpu', *, use_audio=True,
                  intervention='real', seeds=(42, 123, 2026, 77), steps=24):
    """Generation is independent of all query motion values and motion masks."""
    valid = _array(clip['valid'])
    runs = contiguous_runs(valid)
    if not runs:raise ValueError('native generation requires an observed run')
    baseline = _array(clip['baseline52'])
    if baseline.shape != (len(valid), 52) or not np.isfinite(baseline[valid]).all():
        raise ValueError('finite native baseline52 required')
    if not seeds or any(type(seed) is not int for seed in seeds):raise ValueError('explicit integer seeds required')
    latent_dim, block_size = int(ae.latent_dim), int(ae.block_size)
    if getattr(flow, 'latent_dim', latent_dim) != latent_dim or getattr(flow, 'block_size', block_size) != block_size:
        raise ValueError('flow and AE latent clock differ')
    context = torch.as_tensor(clip['context'], dtype=torch.float32, device=device)
    if context.ndim != 1:raise ValueError('context must be one global/identity vector')
    context = ((context-_vector(stats, 'context_mean', len(context), device))
               /_vector(stats, 'context_scale', len(context), device))[None]
    b9 = torch.as_tensor(clip['b9'], dtype=torch.float32, device=device)
    if b9.shape != (9,) or not torch.isfinite(b9).all():raise ValueError('finite independently predicted b9 required')
    scale = _vector(stats, 'residual_scale', 9, device)
    latent_mean = _vector(stats, 'latent_mean', latent_dim, device)
    latent_scale = _vector(stats, 'latent_scale', latent_dim, device)
    samples = np.broadcast_to(baseline[:, UPPER], (len(seeds), len(valid), 9)).copy()
    flow.eval(); ae.eval(); seed_records = []
    for run_index, (start, stop) in enumerate(runs):
        native_valid = torch.ones(1, stop-start, dtype=torch.bool, device=device)
        blocks, frame_valid = _audio_blocks(clip, start, stop, stats, block_size, device, intervention)
        latent_valid = frame_valid.any(-1)
        for sample_index, seed in enumerate(seeds):
            key = _seed(clip['clip_id'], run_index, seed)
            noise = torch.randn(1, latent_valid.shape[1], latent_dim,
                                generator=torch.Generator(device=device).manual_seed(key), device=device)
            normalized = flow.sample(latent_valid, context, blocks, noise, steps=steps,
                                     use_audio=use_audio, audio_frame_valid=frame_valid)
            decoded = ae.decode(normalized*latent_scale+latent_mean, native_valid)
            if decoded.shape != (1, stop-start, 9):raise ValueError('AE decode returned a different native clock')
            samples[sample_index, start:stop] = (decoded[0]*scale+b9).cpu().numpy()
            seed_records.append({'seed': seed, 'run_index': run_index, 'start': start, 'stop': stop,
                                 'noise_seed': key, 'latent_frames': latent_valid.shape[1]})
    return samples, seed_records


@torch.no_grad()
def reconstruct_clip(ae, clip, stats, device='cpu'):
    """Oracle AE reconstruction uses only jointly observed motion runs."""
    valid = _array(clip['valid']); observed = _array(clip['motion_mask'])
    target = _array(clip['motion9']); baseline = _array(clip['baseline52'])
    if observed.shape != (len(valid), 9) or observed.dtype != bool or target.shape != observed.shape:
        raise ValueError('native target9 and Boolean channel observation mask required')
    joint = valid & observed.all(1)
    if not np.isfinite(target[joint]).all():raise ValueError('observed reconstruction target must be finite')
    samples = baseline[:, UPPER].copy()[None]
    scale = _vector(stats, 'residual_scale', 9, device)
    b9 = torch.as_tensor(clip['b9'], dtype=torch.float32, device=device)
    runs = contiguous_runs(joint); ae.eval(); records = []
    for index, (start, stop) in enumerate(runs):
        residual = (torch.as_tensor(target[start:stop], dtype=torch.float32, device=device)-b9)/scale
        mask = torch.ones(1, stop-start, dtype=torch.bool, device=device)
        latent, zvalid = ae.encode(residual[None], mask)
        decoded = ae.decode(latent, mask)
        if decoded.shape != residual[None].shape:raise ValueError('AE reconstruction clock differs')
        samples[0, start:stop] = (decoded[0]*scale+b9).cpu().numpy()
        records.append({'run_index': index, 'start': start, 'stop': stop, 'latent_frames': int(zvalid.sum())})
    return samples, records, joint


def _summary_scores(samples, clip, stats, deterministic=False):
    valid = _array(clip['valid']); observed = _array(clip['motion_mask'])
    if observed.shape != (len(valid), 9) or observed.dtype != bool:raise ValueError('Boolean motion_mask[T,9] required')
    joint = valid & observed.all(1); target = _array(clip['motion9'])
    if target.shape != (len(valid), 9) or not np.isfinite(target[joint]).all():raise ValueError('finite observed target9 required')
    scale_key = 'metric_scale' if 'metric_scale' in stats else 'residual_scale'
    scales = _array(stats[scale_key])
    metric = None
    if joint.any() and np.isfinite(samples[:, joint]).all():
        scored = np.repeat(samples, 2, axis=0) if deterministic else samples
        metric = score_clip(scored, target, joint, scales)
        metric.update(_metadata(clip))
    return metric, joint


def compose_full(samples, baseline52):
    baseline52 = np.asarray(baseline52)
    result = np.broadcast_to(baseline52, (len(samples), *baseline52.shape)).copy()
    result[:, :, UPPER] = samples
    return result


def _numerics(samples, full, baseline, valid):
    observed = samples[:, valid]; finite = np.isfinite(observed)
    safe = observed[finite]
    outside = (observed < 0) | (observed > 1)
    full_observed = full[:, valid]
    return {'native_generated_values': int(observed.size), 'nonfinite_values': int((~finite).sum()),
            'raw_upper_min': float(safe.min()) if len(safe) else None,
            'raw_upper_max': float(safe.max()) if len(safe) else None,
            'raw_upper_rms': float(np.sqrt(np.square(safe.astype(float)).mean())) if len(safe) else None,
            'raw_upper_out_of_range_count': int(outside.sum()), 'raw_upper_out_of_range_fraction': float(outside.mean()),
            'display_upper_clamp_changed_count': int(outside.sum()),
            'display_full52_clamp_fraction': float(((full_observed < 0) | (full_observed > 1)).mean()),
            'other43_exact': bool(np.array_equal(full[:, :, OTHER], np.broadcast_to(baseline[:, OTHER], full[:, :, OTHER].shape), equal_nan=True)),
            'native_invalid_exact_baseline': bool(np.array_equal(full[:, ~valid], np.broadcast_to(baseline[~valid], full[:, ~valid].shape), equal_nan=True)),
            'prediction_clipped': False, 'display_only_clip_policy': '[0,1]; not applied to raw scoring arrays'}


def _selected(clips):
    """One lexical clip per first four emotions, then missing identities."""
    ordered = sorted(clips, key=lambda c: c['clip_id']); selected = []
    emotions = sorted({_metadata(c)['emotion'] for c in ordered}, key=str)[:4]
    for emotion in emotions:selected.append(next(c for c in ordered if _metadata(c)['emotion'] == emotion))
    speakers = {_metadata(c)['speaker'] for c in selected}
    for speaker in sorted({_metadata(c)['speaker'] for c in ordered}, key=str):
        if speaker not in speakers:
            selected.append(next(c for c in ordered if _metadata(c)['speaker'] == speaker)); speakers.add(speaker)
        if len(selected) >= 8:break
    return list({c['clip_id']: c for c in selected}.values())


def _svg(target, baseline, samples, valid, score_mask, destination):
    width, height = 1100, 610
    colors = ('#dc6b26', '#469b79', '#9066c0', '#bd6262')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>']
    groups = [('brow raise', [2, 3, 4]), ('brow down', [0, 1]), ('squint', [5, 7]), ('wide', [6, 8])]
    for panel, (name, channels) in enumerate(groups):
        top = 35+panel*140
        parts.append(f'<text x="15" y="{top}" font-family="sans-serif" font-size="15">{name} · coefficient axis 0–1</text>')
        parts.append(f'<path d="M70 {top+95}H1080M70 {top+5}H1080" stroke="#ddd" fill="none"/>')
        lines = [(target[:, channels].mean(1), score_mask, '#183f6d'),
                 (baseline[:, channels].mean(1), valid, '#999')]
        lines += [(sample[:, channels].mean(1), valid, colors[i % len(colors)]) for i, sample in enumerate(samples)]
        for values, mask, color in lines:
            mask = mask & np.isfinite(values)
            for left, right in contiguous_runs(mask):
                points = ' '.join(f'{70+1010*i/max(len(valid)-1,1):.2f},{top+95-90*values[i]:.2f}' for i in range(left, right))
                parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    parts.append(f'<text x="70" y="600" font-family="sans-serif" font-size="13">Full native duration: {len(valid)/25:.2f}s; blue=target observed, grey=baseline, colored=all fixed draws; no amplitude rescaling</text></svg>')
    destination.write_text(''.join(parts), encoding='utf8')


def _write_artifacts(clips, curves, output, result, mode):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/'npz').mkdir(exist_ok=True); (output/'svg').mkdir(exist_ok=True)
    jobs = []; sections = []; literature_rows = []; selected = {c['clip_id'] for c in _selected(clips)}
    for clip in clips:
        cid = clip['clip_id']
        if Path(cid).name != cid or any(x in cid for x in ('/', '\\', ':')):raise ValueError('unsafe clip_id')
        row = curves[cid]; samples = row['samples']; baseline = _array(clip['baseline52'])
        target52 = _array(clip['target52']); valid = row['native_valid']; score_mask = row['score_mask']
        if target52.shape != baseline.shape:raise ValueError('target52/baseline shape differs')
        full = compose_full(samples, baseline)
        literature_rows.append(score_fullface(full, clip))
        # Preserve the full native output clock in the renderer input. Missing
        # target rows are explicitly marked and baseline-filled for display
        # only; generation and scoring always use their separate raw arrays.
        ref = target52.copy()
        missing_reference = (~score_mask[:, None]) | ~np.isfinite(ref)
        if 'channel_mask' in clip:
            missing_reference |= ~_array(clip['channel_mask'])
        ref = np.where(missing_reference, baseline, ref)
        times = _array(clip['times']) if 'times' in clip else np.arange(len(valid))/25
        names = ['target reference (missing values baseline-filled; not inference)', 'frozen baseline']+[mode+' '+str(s) for s in row['seeds']]
        path = output/'npz'/(cid+'.npz')
        payload = {'channels': np.asarray(ARKIT_NAMES), 'mode_names': np.asarray(names),
                   'motions': np.concatenate((ref[None], baseline[None], full), axis=0),
                   'valid': valid, 'native_valid': valid, 'reference_valid': score_mask,
                   'reference_display_baseline_filled': missing_reference,
                   'times': times, 'clip_id': np.asarray(cid), 'noise_seed': np.asarray(row['seeds'][0])}
        # Forward optional exporter audio metadata without guessing paths or
        # changing offsets. No audio is synthesized or copied by evaluation.
        embedded = clip.get('metadata', {})
        for name in ('audio_relative_path', 'audio_sha256', 'audio_offset_seconds'):
            value = clip.get(name, embedded.get(name))
            if value is not None:payload[name] = np.asarray(value)
        np.savez_compressed(path, **payload)
        job = {'input': 'npz/'+cid+'.npz', 'output': cid, 'fps': 25,
               'columns': len(names), 'tile_size': 320, 'samples': 16, 'max_frames': 0,
               'rendered': False, 'render_mask': 'native valid; missing target values explicitly baseline-filled for display only'}
        audio_path = clip.get('audio_path', embedded.get('audio_path'))
        if audio_path is not None:
            job['audio'] = str(audio_path)
            if 'audio_offset_seconds' in payload:job['audio_offset_seconds'] = float(payload['audio_offset_seconds'])
            if 'audio_sha256' in payload:job['audio_sha256'] = str(payload['audio_sha256'])
        jobs.append(job)
        if cid in selected:
            svg = output/'svg'/(cid+'.svg')
            _svg(row['target'], baseline[:, UPPER], samples, valid, score_mask, svg)
            sections.append('<section><h2>'+html.escape(cid)+'</h2><p>'+html.escape(json.dumps(_metadata(clip), ensure_ascii=False))+'</p><img src="svg/'+html.escape(cid)+'.svg"></section>')
    torch.save({'schema': SCHEMA, 'mode': mode, 'clips': curves, 'result': result}, output/'curves.pt')
    # Separate report: historical model-selection gates and their metrics stay
    # unchanged. No dependency is fabricated when render/encoder inputs lack.
    write_report(output/'arkit_benchmark.json', build_report(
        literature_rows, scope=mode+'; same clips as native evaluation; split status inherited'))
    (output/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    (output/'render_jobs.json').write_text(json.dumps({'driver': 'scripts/render_dynamic_rig_comparison.py', 'jobs': jobs, 'rendered': False}, indent=2)+'\n', encoding='utf8')
    intro = '<h1>Continuous latent · '+html.escape(mode)+'</h1><p>Status: needs_visual_review. Full native curves, fixed metadata examples, no best seed. Numerical checks do not certify naturalness.</p>'
    intro += '<p><a href="arkit_benchmark.json">ARKit literature metrics and pending evaluator dependencies</a></p>'
    if mode == 'ae_reconstruction':intro += '<p>Oracle motion reconstruction only; not audio prediction or free generation.</p>'
    page = '<!doctype html><html><meta charset="utf-8"><title>Continuous latent diagnostics</title><style>body{max-width:1200px;margin:24px auto;font:16px/1.5 sans-serif;background:#edf1f5}section{background:white;padding:20px;margin:20px 0}img{width:100%}</style>'+intro+''.join(sections)+'</html>'
    (output/'index.html').write_text(page, encoding='utf8')


def _evaluate(ae, clips, stats, output, device, *, flow=None, use_audio=True,
              intervention='real', seeds=(42, 123, 2026, 77), steps=24):
    output = Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('fresh evaluation output required')
    if not clips or len({c['clip_id'] for c in clips}) != len(clips):raise ValueError('nonempty unique clip list required')
    if flow is not None and len(seeds) < 2:raise ValueError('at least two fixed draws required for fair ES')
    curves = {}; metrics = []; checks = []; deterministic = flow is None
    mismatches, mismatch_records, mismatch_excluded = {}, [], []
    if intervention == 'mismatch' and not deterministic:
        mismatches, mismatch_records, mismatch_excluded = _mismatched_audio(clips)
        clips = [c for c in clips if c['clip_id'] in mismatches]
    for clip in clips:
        if deterministic:
            samples, records, generated = reconstruct_clip(ae, clip, stats, device)
            row_seeds = ['oracle']
        else:
            samples, records = generate_clip(flow, ae, mismatches.get(clip['clip_id'], clip), stats, device, use_audio=use_audio,
                                             intervention='real' if intervention == 'mismatch' else intervention, seeds=seeds, steps=steps)
            generated = _array(clip['valid']); row_seeds = list(seeds)
        metric, score_mask = _summary_scores(samples, clip, stats, deterministic=deterministic)
        if metric is not None:metrics.append(metric)
        baseline = _array(clip['baseline52']); valid = _array(clip['valid'])
        full = compose_full(samples, baseline)
        check = _numerics(samples, full, baseline, valid); check['clip_id'] = clip['clip_id']; checks.append(check)
        curves[clip['clip_id']] = {'metadata': _metadata(clip), 'samples': samples, 'target': _array(clip['motion9']),
            'native_valid': valid.copy(), 'score_mask': score_mask, 'generated_mask': generated.copy(),
            'seeds': row_seeds, 'run_records': records, 'raw_coefficient_prediction': True}
    finite = all(row['nonfinite_values'] == 0 for row in checks)
    protection = all(row['other43_exact'] and row['native_invalid_exact_baseline'] for row in checks)
    result = {'schema': SCHEMA, 'mode': 'ae_reconstruction' if deterministic else 'free_generation',
              'status': 'needs_visual_review', 'clips': len(clips), 'scored_clips': len(metrics),
              'summary': summarize(metrics) if metrics else None, 'per_clip_scores': metrics, 'numerics': checks,
              'numerical_gate': {'passed': bool(curves) and finite and protection, 'finite': finite, 'protected43': protection,
                                 'scope': 'catastrophic numerical/protection checks only; raw out-of-range and RMS reported separately'},
              'naturalness_certified': False, 'use_audio': use_audio if not deterministic else None,
              'intervention': intervention if not deterministic else None, 'steps': steps if not deterministic else None,
              'generation_support': 'native audio valid runs only' if not deterministic else 'joint motion observation runs (oracle reconstruction)',
              'score_support': 'native valid AND all nine motion_mask channels',
              'metric_scale_source': 'metric_scale' if 'metric_scale' in stats else 'residual_scale',
              'deterministic_score_note': 'one AE reconstruction repeated twice for score API; zero spread, not stochastic generation evidence' if deterministic else None,
              'prediction_composition': 'b9 + AE_raw_residual * fit_residual_scale; no logit, sigmoid, clipping or query statistics',
              'selected_clip_ids': [c['clip_id'] for c in _selected(clips)],
              'mismatch_donors': mismatch_records, 'mismatch_excluded_no_donor': mismatch_excluded,
              'mismatch_note': 'Within split, same emotion, different sentence; speaker matched when available. Resampling may shift the input distribution; compare matched_global as the main control.' if intervention == 'mismatch' else None}
    _write_artifacts(clips, curves, output, result, 'ae_reconstruction' if deterministic else 'generation_'+intervention)
    return result


def evaluate_ae(ae, clips, stats, output, device='cpu'):
    return _evaluate(ae, clips, stats, output, device)


def evaluate_generation(flow, ae, clips, stats, output, device='cpu', *, use_audio=True,
                        intervention='real', seeds=(42, 123, 2026, 77), steps=24):
    return _evaluate(ae, clips, stats, output, device, flow=flow, use_audio=use_audio,
                     intervention=intervention, seeds=seeds, steps=steps)
