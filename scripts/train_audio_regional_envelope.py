"""Native-rate audio control of a frozen upper-face trajectory.

Targets are five-frame smoothed RMS activity of clip-centered brow/eye
coefficients. They are NOT absolute emotional intensity or a MEDTalk EIE.
Fit-fold normalization and epoch selection precede one full-TRAIN refit and
one development comparison. The frozen Stage4 source already saw TRAIN:
the held-out fold validates the new adapter, not the whole pretrained system.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetalk_b0.models.audio_regional_envelope import AudioRegionalEnvelope, compose_prior_with_envelope
from kinetalk_b0.models.regional_intensity_gain import regional_envelope
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES
from scripts.extract_emotion2vec_pilot import sha
from scripts.joint_motion_metrics import _runs
from scripts.mouth_protection import protection_report
from scripts.train_formal_predictable_projection import canonical_hash, save_checkpoint, save_json
from scripts.train_full_staged import NOT_UPPER, subset
from scripts.train_isolated_audio_state import load_context, old_prediction

WINDOW = 5
GAIN_MIN, GAIN_MAX = .25, 2.5


def _center(value, valid):
    mask = valid[..., None]
    first = valid.long().argmax(1, keepdim=True)[..., None].expand(-1, -1, value.shape[-1])
    relative = torch.where(mask, value - value.gather(1, first), 0.)
    mean = relative.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)[..., None]
    return torch.where(mask, relative - mean, 0.)


def _upper_valid(q):
    valid = q['valid']
    channels = q.get('channel_mask')
    if channels is not None:
        eligible = channels[..., list(UPPER_INDICES)].all(-1)
        valid = valid & (eligible[:, None] if eligible.ndim == 1 else eligible)
    if not valid.any(1).all():
        raise ValueError('Every retained clip needs observed upper-face frames')
    return valid


def _fold(q):
    speakers = sorted(set(q['speaker']))
    sentences = sorted(set(q['sentence_id']))
    held_s, held_t = set(speakers[::5]), set(sentences[::4])
    pairs = list(zip(q['speaker'], q['sentence_id']))
    fit = torch.tensor([i for i, (s, t) in enumerate(pairs) if s not in held_s and t not in held_t], dtype=torch.long)
    cal = torch.tensor([i for i, (s, t) in enumerate(pairs) if s in held_s and t in held_t], dtype=torch.long)
    if min(len(fit), len(cal)) < 2:
        raise ValueError('Speaker-by-sentence fold needs at least two fit and calibration clips')
    return fit, cal, {'held_speakers': sorted(held_s), 'held_sentences': sorted(held_t),
                      'fit_clips': len(fit), 'calibration_clips': len(cal),
                      'mixed_cells_unused_for_selection': len(pairs) - len(fit) - len(cal)}


def _diverse_ids(q, ids, limit, seed):
    """Round-robin speaker groups after randomization; never truncate before split."""
    if len(ids) <= limit:
        return ids
    rng = np.random.default_rng(seed)
    groups = {}
    for i in rng.permutation(ids.numpy()).tolist():
        groups.setdefault(q['speaker'][i], []).append(i)
    keys = sorted(groups)
    rng.shuffle(keys)
    out = []
    while len(out) < limit:
        for key in keys:
            if groups[key]:
                out.append(groups[key].pop())
                if len(out) == limit:
                    break
    return torch.tensor(out, dtype=torch.long)


@torch.no_grad()
def _cache_source(data, system, audio, identities, split, ids, device, seed, batch_size):
    q = data['splits'][split]
    system.eval(); audio.eval()
    rows = []
    generator = torch.Generator().manual_seed(seed)
    for ix in ids.split(batch_size):
        b = subset(q, ix, device)
        b['frozen_local'] = audio(b['audio_features'], b['valid'])['local']
        noise = torch.randn((*b['valid'].shape, 52), generator=generator).to(device)
        base = old_prediction(system, b, identities, noise)
        valid = _upper_valid(b)
        upper = base[..., list(UPPER_INDICES)]
        mean = torch.where(valid[..., None], upper, 0.).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        rows.append({'base_full': base.cpu(), 'valid': valid.cpu(), 'mean': mean.cpu()})
    cache = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    # Reuse the loaded full-split feature tensors, and avoid copying all legacy
    # teacher/content caches merely to retain a small list of output fields.
    is_all = len(ids) == len(q['valid']) and torch.equal(ids, torch.arange(len(ids)))
    needed = ('audio_features', 'motion', 'channel_mask', 'emotion_id', 'valid',
              'clip_id', 'speaker', 'sentence_id', 'times', 'b0')
    selected = {}
    for key in needed:
        if key not in q:
            continue
        value = q[key]
        selected[key] = (value if is_all else value[ids] if torch.is_tensor(value)
                         else [value[int(i)] for i in ids])
    cache.update(features=selected['audio_features'], target_full=selected['motion'],
                 channel_mask=selected['channel_mask'], emotion=selected['emotion_id'],
                 original_valid=selected['valid'], clip_id=selected['clip_id'],
                 speaker=selected['speaker'], sentence_id=selected['sentence_id'])
    for key in ('times', 'b0'):
        if key in selected:
            cache[key] = selected[key]
    return cache


def _take_cache(cache, ids):
    return {k: (v[ids] if torch.is_tensor(v) and v.ndim and len(v) == len(cache['valid'])
                else [v[int(i)] for i in ids] if isinstance(v, list) and len(v) == len(cache['valid'])
                else v) for k, v in cache.items()}


def _apply_carrier(cache, window, batch_size):
    """Keep the exact original prior and an explicitly disclosed smooth carrier."""
    if window not in (1, 3, 5, 7, 9):
        raise ValueError('carrier window must be one of 1, 3, 5, 7, 9')
    original = cache['base_full']
    if window == 1:
        carrier = original
    else:
        from kinetalk_b0.models.temporal_motion_carrier import smooth_motion_carrier
        carrier = torch.cat([smooth_motion_carrier(original[ix], cache['valid'][ix], window=window)['smoothed']
                             for ix in torch.arange(len(cache['valid'])).split(batch_size)])
    if not torch.equal(carrier[..., list(NOT_UPPER)].contiguous().view(torch.int32),
                       original[..., list(NOT_UPPER)].contiguous().view(torch.int32)):
        raise RuntimeError('Carrier smoothing altered non-upper channels')
    upper = carrier[..., list(UPPER_INDICES)]
    mean = torch.where(cache['valid'][..., None], upper, 0.).sum(1) / cache['valid'].sum(1, keepdim=True)
    if not torch.allclose(mean, cache['mean'], atol=1e-5, rtol=0):
        raise RuntimeError('Carrier smoothing changed the protected upper mean')
    return {**cache, 'original_base_full': original, 'base_full': carrier,
            'mean': mean, 'carrier_window': window}


def _fit_stats(cache, ids, batch_size):
    count = 0
    total = torch.zeros(cache['features'].shape[-1], dtype=torch.float64)
    squares = torch.zeros_like(total)
    upper_squares = torch.zeros(9, dtype=torch.float64)
    for ix in ids.split(batch_size):
        valid = cache['valid'][ix]
        values = cache['features'][ix][valid].double()
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite observed features')
        total += values.sum(0); squares += values.square().sum(0); count += len(values)
        centered = _center(cache['target_full'][ix][..., list(UPPER_INDICES)].double(), valid)
        upper_squares += centered.square().sum((0, 1))
    if not count:
        raise ValueError('Cannot fit statistics on an empty fold')
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
    return {'feature_mean': mean.float(), 'feature_std': std.float(),
            'channel_scales': (upper_squares / count).sqrt().clamp_min(.02).float()}


def _envelopes(motion, valid, channel_scales, batch_size):
    rows = []
    for ids in torch.arange(len(valid)).split(batch_size):
        centered = _center(motion[ids][..., list(UPPER_INDICES)], valid[ids])
        rows.append(regional_envelope(centered, valid[ids], window=WINDOW,
                                      channel_scales=channel_scales.to(centered)))
    return torch.cat(rows)


def _prepare_cache(cache, stats, fit_ids, batch_size):
    target = _envelopes(cache['target_full'], cache['valid'], stats['channel_scales'], batch_size)
    prior = _envelopes(cache['base_full'], cache['valid'], stats['channel_scales'], batch_size)
    if fit_ids is not None:
        values = target[fit_ids][cache['valid'][fit_ids]].numpy()
        stats = {**stats, 'envelope_scale': torch.from_numpy(np.quantile(values, .95, axis=0)).float().clamp_min(.05)}
    scale = stats['envelope_scale']
    return {**cache, 'target_env': target / scale, 'prior_env': prior / scale}, stats


def _feature_reverse(features, valid):
    out = torch.zeros_like(features)
    for row in range(len(features)):
        for left, right in _runs(valid[row].cpu().numpy()):
            out[row, left:right] = features[row, left:right].flip(0)
    return out


def _feature_shuffle(features, valid, row_ids, seed):
    out = torch.zeros_like(features)
    for row, row_id in enumerate(row_ids.tolist()):
        rng = np.random.default_rng(seed + int(row_id))
        for left, right in _runs(valid[row].cpu().numpy()):
            order = np.arange(left, right); rng.shuffle(order)
            out[row, left:right] = features[row, torch.as_tensor(order, device=features.device)]
    return out


def _donors(cache):
    """Prefer same speaker and unequal sentence, then unequal sentence, never self."""
    n = len(cache['valid'])
    if n < 2:
        raise ValueError('Mismatch requires at least two clips')
    lengths = cache['valid'].sum(1).tolist()
    result = []
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        result.append(min(candidates, key=lambda j: (
            cache['sentence_id'][j] == cache['sentence_id'][i],
            cache['speaker'][j] != cache['speaker'][i], abs(lengths[j] - lengths[i]), j)))
    return torch.tensor(result, dtype=torch.long)


def _mismatch_features(cache, ids, donors):
    """Duration-normalize donor valid audio onto recipient valid slots (diagnostic)."""
    out = torch.zeros_like(cache['features'][ids])
    for row, i in enumerate(ids.tolist()):
        j = int(donors[i])
        if i == j:
            raise ValueError('A mismatch donor cannot be the recipient')
        dst = cache['valid'][i].nonzero(as_tuple=True)[0]
        src = cache['valid'][j].nonzero(as_tuple=True)[0]
        values = cache['features'][j, src].T[None]
        out[row, dst] = F.interpolate(values, size=len(dst), mode='linear', align_corners=False)[0].T
    return out


@torch.no_grad()
def _predict(model, cache, ids, device, batch_size, mode='full', seed=20260921, donors=None):
    model.eval()
    rows = []
    for ix in ids.split(batch_size):
        valid = cache['valid'][ix].to(device)
        features = (_mismatch_features(cache, ix, donors) if mode == 'mismatch'
                    else cache['features'][ix]).to(device)
        if mode == 'reverse':
            features = _feature_reverse(features, valid)
        elif mode == 'shuffle':
            features = _feature_shuffle(features, valid, ix, seed)
        elif mode not in ('full', 'mismatch'):
            raise ValueError(mode)
        pred = model(features, valid)
        if pred.shape != (*valid.shape, 2) or not torch.isfinite(pred[valid]).all():
            raise ValueError('Invalid native-rate envelope prediction')
        rows.append(pred.cpu())
    return torch.cat(rows)


def _static_envelope(envelope, valid):
    mean = torch.where(valid[..., None], envelope, 0.).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)[..., None]
    return torch.where(valid[..., None], mean.expand_as(envelope), 0.)


def _gain_from_envelope(pred_env, prior_env, valid):
    gain = (pred_env / prior_env.clamp_min(.02)).clamp(GAIN_MIN, GAIN_MAX)
    return torch.where(valid[..., None], gain, torch.ones_like(gain))


def _train_epoch(model, cache, ids, optimizer, device, batch_size, seed):
    model.train()
    order = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))]
    total, frames = 0., 0
    for ix in order.split(batch_size):
        valid = cache['valid'][ix].to(device)
        target = cache['target_env'][ix].to(device)
        pred = model(cache['features'][ix].to(device), valid)
        mse = torch.where(valid[..., None], (pred - target).square(), 0.).sum() / (valid.sum() * 2)
        adjacent = valid[:, 1:] & valid[:, :-1]
        error = (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])
        delta = torch.where(adjacent[..., None], error.square(), 0.).sum() / (adjacent.sum() * 2).clamp_min(1)
        loss = mse + .25 * delta
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite envelope training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        count = int(valid.sum()); total += float(loss.detach()) * count; frames += count
    return total / frames


def _finite_mean(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def _envelope_metrics(prediction, target, valid, *, light=False):
    per_clip = []
    for row in range(len(valid)):
        x, y = prediction[row, valid[row]].double(), target[row, valid[row]].double()
        xc, yc = x - x.mean(0), y - y.mean(0)
        den = (xc.square().sum(0) * yc.square().sum(0)).sqrt()
        correlations = [float((xc[:, r] * yc[:, r]).sum() / den[r]) if den[r] > 1e-10 else None for r in range(2)]
        delta_errors, lags = [], []
        for left, right in _runs(valid[row].numpy()):
            xr, yr = prediction[row, left:right].double(), target[row, left:right].double()
            if len(xr) > 1:
                delta_errors.extend(((xr[1:] - xr[:-1]) - (yr[1:] - yr[:-1])).square().mean(-1).tolist())
            if light:
                continue
            # This is the peak of cross-correlation, not matched expression events.
            for region in range(2):
                best = None
                for lag in range(-min(15, max(0, len(xr) - 3)), min(15, max(0, len(xr) - 3)) + 1):
                    a, b = (xr[lag:, region], yr[:len(xr)-lag, region]) if lag >= 0 else (xr[:len(xr)+lag, region], yr[-lag:, region])
                    if len(a) < 3:
                        continue
                    a, b = a-a.mean(), b-b.mean()
                    d = (a.square().sum() * b.square().sum()).sqrt()
                    if d > 1e-10:
                        score = float((a*b).sum()/d)
                        candidate = (score, -abs(lag), -lag)
                        if best is None or candidate > best[0]:
                            best = (candidate, lag)
                if best is not None:
                    lags.append(abs(best[1]))
        per_clip.append({'mse': float((x-y).square().mean()), 'centered_mse': float((xc-yc).square().mean()),
                         'correlation': _finite_mean(correlations), 'region_correlation': correlations,
                         'temporal_std': float(x.std(0, unbiased=False).mean()),
                         'target_temporal_std': float(y.std(0, unbiased=False).mean()),
                         'temporal_difference_mse': _finite_mean(delta_errors),
                         'cross_correlation_peak_abs_lag_frames': _finite_mean(lags),
                         'valid_frames': len(x)})
    keys = ('mse', 'centered_mse', 'correlation', 'temporal_std', 'target_temporal_std',
            'temporal_difference_mse', 'cross_correlation_peak_abs_lag_frames')
    summary = {key: _finite_mean([r[key] for r in per_clip]) for key in keys}
    summary.update(clips=len(per_clip), defined_correlation_clips=sum(r['correlation'] is not None for r in per_clip),
                   lag_evaluated=not light,
                   reduction='equal clip weighting; undefined constant correlations omitted, never set to zero',
                   per_clip=per_clip)
    return summary


def _paired_cluster_ci(full, control, speakers, sentences, seed=20260921, draws=2000):
    delta = np.asarray(full, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    if len(delta) != len(speakers) or len(delta) != len(sentences) or not np.isfinite(delta).all():
        raise ValueError('Cluster CI needs finite paired clip values and matching metadata')
    result = {'delta_full_minus_control': float(delta.mean()), 'clips': len(delta),
              'negative_favors_full': True, 'method': 'paired cluster bootstrap, equal clip estimand', 'draws': draws}
    rng = np.random.default_rng(seed)
    for label, group in (('speaker', speakers), ('sentence', sentences)):
        names = sorted(set(group)); sums, counts = [], []
        for name in names:
            values = delta[np.asarray([g == name for g in group])]
            sums.append(values.sum()); counts.append(len(values))
        if len(names) < 2:
            result[label + '_cluster'] = {'clusters': len(names), 'ci95': None, 'improvement_supported': False}
            continue
        ix = rng.integers(len(names), size=(draws, len(names)))
        values = np.asarray(sums)[ix].sum(1) / np.asarray(counts)[ix].sum(1)
        ci = np.quantile(values, [.025, .975]).tolist()
        result[label + '_cluster'] = {'clusters': len(names), 'ci95': ci, 'improvement_supported': ci[1] < 0}
    return result


def _comparisons(metrics, cache, main, controls):
    return {name: {metric: _paired_cluster_ci(
        [r[metric] for r in metrics[main]['per_clip']], [r[metric] for r in metrics[name]['per_clip']],
        cache['speaker'], cache['sentence_id']) for metric in ('mse', 'centered_mse')}
        for name in controls}


def _upper_motion_diagnostics(motion, valid):
    """Observe display clamping separately; raw curves are never replaced."""
    upper = motion[..., list(UPPER_INDICES)].double()
    report = {}
    for name, values in (('raw', upper), ('clamped_0_1', upper.clamp(0, 1))):
        centered = _center(values, valid)
        std = (centered.square().sum(1) / valid.sum(1, keepdim=True)).sqrt()
        energy = centered[valid].square().mean(-1)
        velocity = []; acceleration = []
        for row in range(len(valid)):
            for left, right in _runs(valid[row].numpy()):
                x = values[row, left:right]
                if len(x) > 1:
                    velocity.extend((x[1:]-x[:-1]).square().mean(-1).tolist())
                if len(x) > 2:
                    acceleration.extend((x[2:]-2*x[1:-1]+x[:-2]).square().mean(-1).tolist())
        report[name] = {'brow_temporal_std_equal_clip_mean': float(std[:, :5].mean()),
                        'eye_temporal_std_equal_clip_mean': float(std[:, 5:].mean()),
                        'per_channel_temporal_std': std.mean(0).tolist(),
                        'centered_energy_quantiles': dict(zip(('p05', 'p25', 'p50', 'p75', 'p95'),
                            np.quantile(energy.numpy(), [.05, .25, .5, .75, .95]).tolist())),
                        'frame_displacement_energy_mean': _finite_mean(velocity),
                        'frame_acceleration_energy_mean': _finite_mean(acceleration),
                        'scope': 'native-frame differences; acceleration is a jitter diagnostic, not a naturalness score'}
    observed = upper[valid]
    means = torch.where(valid[..., None], upper, 0.).sum(1) / valid.sum(1, keepdim=True)
    report['bounds'] = {'observed_outside_0_1_fraction': float(((observed < 0) | (observed > 1)).double().mean()),
                        'clip_channel_mean_within_005_of_boundary_fraction': float(((means <= .05) | (means >= .95)).double().mean()),
                        'clip_channel_mean_min': float(means.min()), 'clip_channel_mean_max': float(means.max()),
                        'per_channel_mean': means.mean(0).tolist(),
                        'clamp_is_diagnostic_only': True}
    return report


@torch.no_grad()
def _compose_conditions(envelopes, cache, stats, batch_size, oracle=False):
    gains = {name: _gain_from_envelope(env, cache['prior_env'], cache['valid']) for name, env in envelopes.items()}
    main = 'oracle_envelope' if oracle else 'full'
    static = 'oracle_static_gain' if oracle else 'static_gain'
    gains[static] = _static_envelope(gains[main], cache['valid'])
    gains['prior'] = torch.ones_like(gains[main])
    gains['original_prior'] = torch.ones_like(gains[main])
    predictions, actual, protection, gain_metrics = {}, {}, {}, {}
    for name, gain in gains.items():
        rows = []
        for ix in torch.arange(len(cache['valid'])).split(batch_size):
            rows.append(compose_prior_with_envelope(cache['base_full'][ix], gain[ix], cache['mean'][ix], cache['valid'][ix]))
        generated = torch.cat(rows)
        # Preserve the exact baseline for the reference condition (no arithmetic).
        if name == 'prior':
            generated = cache['base_full']
        elif name == 'original_prior':
            generated = cache.get('original_base_full', cache['base_full'])
        predictions[name] = generated
        actual[name] = _envelopes(generated, cache['valid'], stats['channel_scales'], batch_size) / stats['envelope_scale']
        upper_mean = torch.where(cache['valid'][..., None], generated[..., list(UPPER_INDICES)], 0.).sum(1) / cache['valid'].sum(1, keepdim=True)
        exact = torch.equal(generated[..., list(NOT_UPPER)].contiguous().view(torch.int32),
                            cache['base_full'][..., list(NOT_UPPER)].contiguous().view(torch.int32))
        protection[name] = {'nonupper43_bit_exact': exact,
                            'max_upper_mean_drift': float((upper_mean-cache['mean']).abs().max()),
                            'mouth': protection_report(generated, cache['base_full'], cache['target_full'],
                                cache['original_valid'], cache['channel_mask'], cache['emotion'])}
        observed = gain[cache['valid']]
        gain_metrics[name] = {'min': float(observed.min()), 'max': float(observed.max()),
                              'lower_saturated_fraction': float((observed <= GAIN_MIN + 1e-7).float().mean()),
                              'upper_saturated_fraction': float((observed >= GAIN_MAX - 1e-7).float().mean())}
    return predictions, actual, protection, gain_metrics


def _evaluate(envelopes, cache, stats, batch_size, oracle=False):
    predicted_metrics = {name: _envelope_metrics(env, cache['target_env'], cache['valid']) for name, env in envelopes.items()}
    predictions, actual, protection, gain_metrics = _compose_conditions(envelopes, cache, stats, batch_size, oracle)
    actual_metrics = {name: _envelope_metrics(env, cache['target_env'], cache['valid']) for name, env in actual.items()}
    main = 'oracle_envelope' if oracle else 'full'
    comparisons = _comparisons(actual_metrics, cache, main, [k for k in actual_metrics if k != main])
    protected = all(p['nonupper43_bit_exact'] and p['max_upper_mean_drift'] < 1e-5 for p in protection.values())
    required = ('original_prior', 'prior', 'oracle_static_gain') if oracle else ('original_prior', 'prior', 'static_gain', 'reverse', 'shuffle', 'mismatch')
    checks = {'fullface_structurally_protected': protected,
              'actual_envelope_correlation_positive': (actual_metrics[main]['correlation'] or -1.) > 0.}
    for name in required:
        for metric in ('mse', 'centered_mse'):
            pair = comparisons[name][metric]
            checks[f'{metric}_beats_{name}_both_cluster_ci'] = all(pair[g+'_cluster']['improvement_supported'] for g in ('speaker', 'sentence'))
    report = {'predicted_envelope_metrics': predicted_metrics, 'composed_envelope_metrics': actual_metrics,
              'composed_paired_cluster_comparisons': comparisons,
              'predicted_paired_cluster_comparisons': _comparisons(predicted_metrics, cache, main, [k for k in predicted_metrics if k != main]),
              'protection': protection, 'gain_diagnostics': gain_metrics,
              'upper_motion_diagnostics': {name: _upper_motion_diagnostics(value, cache['valid']) for name, value in predictions.items()},
              'target_upper_motion_diagnostics': _upper_motion_diagnostics(cache['target_full'], cache['valid']),
              'diagnostic_gate': {'checks': checks, 'passed': all(checks.values())},
              'paper_success_established': False,
              'scope': 'single fixed prior noise seed; activity-envelope diagnostics, not emotional fidelity or naturalness'}
    curves = {key: cache[key] for key in ('clip_id', 'speaker', 'sentence_id', 'times', 'b0') if key in cache}
    curves.update(target=cache['target_full'], valid=cache['original_valid'], upper_valid=cache['valid'],
                  channel_mask=cache['channel_mask'], predictions=predictions,
                  predicted_envelopes=envelopes, composed_envelopes=actual, target_envelope=cache['target_env'],
                  envelope_scale=stats['envelope_scale'], channel_scales=stats['channel_scales'],
                  native_rate=True, postprocessed=cache.get('carrier_window', 1) > 1,
                  carrier_smoothing={'window_frames': cache.get('carrier_window', 1),
                      'applied': cache.get('carrier_window', 1) > 1,
                      'stage': 'frozen upper-motion carrier before regional gain composition',
                      'raw_target_smoothed': False, 'original_prior_key': 'original_prior',
                      'carrier_prior_key': 'prior', 'post_composition_smoothing': False})
    return report, curves


def _new_model(stats, device, seed):
    torch.manual_seed(seed)
    model = AudioRegionalEnvelope(stats['feature_mean'].to(device), stats['feature_std'].to(device), stride=1).to(device)
    return model, torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.02)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'source', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--seed', type=int, default=47)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--smoke-epochs', type=int, default=3)
    p.add_argument('--oracle-only', action='store_true')
    p.add_argument('--carrier-window', type=int, choices=(1, 3, 5, 7, 9), default=1,
                   help='Fixed upper-motion carrier smoothing window; 1 leaves the prior unchanged')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.smoke_epochs) < 1:
        raise ValueError('Positive training budgets and batch size required')
    if args.output.exists():
        raise FileExistsError('Fresh output directory required')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    args.output.mkdir(parents=True); started = time.monotonic()
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data, system, audio, identities, _ = load_context(args.data, args.source, device, args.seed)
    if 'test' in data['splits']:
        raise ValueError('A test split must not be loaded by this experiment')
    q = data['splits']['train']; fit, cal, fold = _fold(q)
    if args.smoke:
        fit = _diverse_ids(q, fit, 32, args.seed)
        cal = _diverse_ids(q, cal, 32, args.seed+1)
    selected_ids = torch.unique(torch.cat((fit, cal)), sorted=True) if (args.smoke or args.oracle_only) else torch.arange(len(q['valid']))
    positions = {int(i): j for j, i in enumerate(selected_ids)}
    fit_ids = torch.tensor([positions[int(i)] for i in fit], dtype=torch.long)
    cal_ids = torch.tensor([positions[int(i)] for i in cal], dtype=torch.long)
    raw = _cache_source(data, system, audio, identities, 'train', selected_ids, device, args.seed, args.batch_size)
    raw = _apply_carrier(raw, args.carrier_window, args.batch_size)
    stats = _fit_stats(raw, fit_ids, args.batch_size)
    cache, stats = _prepare_cache(raw, stats, fit_ids, args.batch_size)
    protocol = {'schema': 'native_regional_activity_v2', 'target': 'five-frame box-smoothed regional RMS of per-clip centered observed upper9; activity, not absolute emotion intensity',
                'window_frames': WINDOW, 'stride': 1, 'gain_bounds': [GAIN_MIN, GAIN_MAX],
                'carrier_smoothing': {'window_frames': args.carrier_window, 'applied': args.carrier_window > 1,
                    'fixed_before_selection': True, 'target_motion_remains_raw': True,
                    'stage': 'upper-only frozen carrier transform before gain composition',
                    'original_prior_retained': True, 'post_composition_smoothing': False},
                'gain_denominator_floor_normalized': .02, 'source_sha256': sha(args.source),
                'data_manifest_sha256': data['provenance'].get('manifest_sha256'),
                'runner_sha256': sha(Path(__file__)), 'fold': fold,
                'fit_clip_ids': [q['clip_id'][int(i)] for i in fit], 'calibration_clip_ids': [q['clip_id'][int(i)] for i in cal],
                'fit_normalization': {k: v.tolist() for k, v in stats.items() if k in ('channel_scales', 'envelope_scale')},
                'normalization_scope': 'feature/channel/envelope scales fit only on selection-fit clips; refit scales recomputed on full TRAIN',
                'frozen_source_scope': 'Stage4 pretrained on full TRAIN; fold isolation applies only to the new audio-envelope predictor',
                'loader_scope': 'legacy load_context precomputes development-derived tensors; none are used by selection, fit statistics, or oracle-only evaluation',
                'selection_budget': args.smoke_epochs if args.smoke else args.epochs,
                'selection_metric': 'equal-clip calibration envelope MSE with eval mode, fixed epoch budget',
                'smoke': args.smoke, 'oracle_only': args.oracle_only,
                'control_semantics': {'static_envelope': 'constant per-clip full-prediction mean; ratio gain can remain time-varying',
                    'static_gain': 'constant per-clip mean of full gain; frozen-prior timing retained',
                    'reverse': 'precomputed audio features reversed inside contiguous valid runs, not waveform reversal; frozen prior unchanged',
                    'shuffle': 'precomputed audio feature frames shuffled inside valid runs with deterministic clip seeds; frozen prior unchanged',
                    'mismatch': 'nonself donor precomputed audio features, prefer different sentence and same speaker, duration-normalized to recipient valid slots; frozen prior unchanged',
                    'zero': 'zero requested activity, bounded ratio yields positive minimum gain; not zero face motion'},
                'test_loaded': False, 'default_replaced': False, 'prior_noise_seed': args.seed}
    save_json(args.output/'protocol.json', protocol)
    if args.oracle_only:
        cal_cache = _take_cache(cache, cal_ids)
        truth = cal_cache['target_env']
        envelopes = {'oracle_envelope': truth, 'oracle_static_envelope': _static_envelope(truth, cal_cache['valid']),
                     'oracle_reverse': _feature_reverse(truth, cal_cache['valid']), 'zero': torch.zeros_like(truth)}
        report, curves = _evaluate(envelopes, cal_cache, stats, args.batch_size, oracle=True)
        report.update(protocol=protocol, audio_predictor_trained=False, evaluated_split='TRAIN held-out speaker-by-sentence calibration')
        save_json(args.output/'evaluation.json', report)
        save_checkpoint(args.output/'native_curves.pt', curves)
        save_json(args.output/'status.json', {'status': 'oracle_diagnostic_complete', 'diagnostic_gate_passed': report['diagnostic_gate']['passed'],
                                            'paper_success_established': False, 'elapsed_seconds': time.monotonic()-started})
        print(json.dumps({'output': str(args.output), 'oracle_gate': report['diagnostic_gate']}, ensure_ascii=False), flush=True)
        return

    model, optimizer = _new_model(stats, device, args.seed)
    history = []; best, best_epoch = float('inf'), 1
    budget = args.smoke_epochs if args.smoke else args.epochs
    for epoch in range(1, budget+1):
        loss = _train_epoch(model, cache, fit_ids, optimizer, device, args.batch_size, args.seed+epoch)
        oof = _predict(model, cache, cal_ids, device, args.batch_size)
        metrics = _envelope_metrics(oof, cache['target_env'][cal_ids], cache['valid'][cal_ids], light=True)
        row = {'epoch': epoch, 'train_loss': loss, 'calibration_mse': metrics['mse'],
               'calibration_correlation': metrics['correlation'], 'calibration_temporal_std': metrics['temporal_std']}
        if args.smoke:
            pred_train = _predict(model, cache, fit_ids, device, args.batch_size)
            train_metrics = _envelope_metrics(pred_train, cache['target_env'][fit_ids], cache['valid'][fit_ids], light=True)
            row['train_prediction'] = {k: train_metrics[k] for k in ('mse', 'correlation', 'temporal_std', 'target_temporal_std')}
        history.append(row)
        if metrics['mse'] < best:
            best, best_epoch = metrics['mse'], epoch
        save_json(args.output/'selection_history.json', history)
        print(json.dumps(row), flush=True)
    selection = {'chosen_epochs': best_epoch, 'best_calibration_mse': best,
                 'development_used_for_selection': False, 'budget_epochs': budget}
    save_json(args.output/'selection.json', selection)
    if args.smoke:
        save_checkpoint(args.output/'smoke.pt', {'model': model.state_dict(), 'stats': stats, 'protocol': protocol})
        save_json(args.output/'status.json', {'status': 'smoke_complete', 'scope': '32 diverse fit clips; interface and short training diagnostic only',
                                            'formal_refit_run': False, 'development_evaluated': False,
                                            'paper_success_established': False, 'elapsed_seconds': time.monotonic()-started})
        return

    # Discard fold statistics and weights before full-TRAIN refit.
    all_ids = torch.arange(len(raw['valid']))
    refit_stats = _fit_stats(raw, all_ids, args.batch_size)
    train_cache, refit_stats = _prepare_cache(raw, refit_stats, all_ids, args.batch_size)
    model, optimizer = _new_model(refit_stats, device, args.seed+100000)
    refit_history = []
    for epoch in range(1, best_epoch+1):
        loss = _train_epoch(model, train_cache, all_ids, optimizer, device, args.batch_size, args.seed+epoch)
        refit_history.append({'epoch': epoch, 'train_loss': loss})
        save_json(args.output/'refit_history.json', refit_history)
        print(json.dumps({'refit_epoch': epoch, 'train_loss': loss}), flush=True)
    save_checkpoint(args.output/'final.pt', {'model': model.state_dict(), 'stats': refit_stats,
        'protocol': protocol, 'protocol_sha256': canonical_hash(protocol), 'selection': selection, 'test_loaded': False})
    val_ids = torch.arange(len(data['splits']['validation']['valid']))
    val_raw = _cache_source(data, system, audio, identities, 'validation', val_ids, device, args.seed+1, args.batch_size)
    val_raw = _apply_carrier(val_raw, args.carrier_window, args.batch_size)
    val_cache, _ = _prepare_cache(val_raw, refit_stats, None, args.batch_size)
    donors = _donors(val_cache)
    envelopes = {mode: _predict(model, val_cache, val_ids, device, args.batch_size, mode=mode, donors=donors)
                 for mode in ('full', 'reverse', 'shuffle', 'mismatch')}
    envelopes['static_envelope'] = _static_envelope(envelopes['full'], val_cache['valid'])
    report, curves = _evaluate(envelopes, val_cache, refit_stats, args.batch_size)
    report.update(protocol=protocol, selection=selection, evaluated_split='development, one comparison after refit',
                  mismatch_donor_clip_ids=[val_cache['clip_id'][int(i)] for i in donors],
                  refit_normalization={k: v.tolist() for k, v in refit_stats.items() if k in ('channel_scales', 'envelope_scale')},
                  prior_noise_seed=args.seed+1, test_loaded=False, default_replaced=False)
    curves.update(noise_seeds=[args.seed+1], source_sha256=protocol['source_sha256'], selected_checkpoint_sha256=sha(args.output/'final.pt'))
    save_json(args.output/'evaluation.json', report)
    save_checkpoint(args.output/'native_curves.pt', curves)
    save_json(args.output/'status.json', {'status': 'complete', 'diagnostic_gate_passed': report['diagnostic_gate']['passed'],
        'paper_success_established': False, 'test_loaded': False, 'default_replaced': False, 'elapsed_seconds': time.monotonic()-started})
    print(json.dumps({'output': str(args.output), 'diagnostic_gate': report['diagnostic_gate']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
