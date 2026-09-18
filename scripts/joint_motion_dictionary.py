"""Train-only empirical joint upper-motion support and deployment continuation.

Tokens are real nine-channel motion units, not discovered semantic actions.
The caller supplies the fitting allowlist; this module never loads a dataset.
Only explicitly named ``oracle_*`` routines can inspect a query future.
"""
from __future__ import annotations

import copy
import hashlib
from collections import Counter

import numpy as np


SCHEMA = 'joint_motion_medoid_dictionary_v1'
EPS = 1e-4
UPPER_INDICES = (41, 42, 43, 44, 45, 5, 6, 12, 13)


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _finite(value, shape=None, name='value'):
    value = _array(value)
    if (shape is not None and value.shape != shape) or not np.isfinite(value).all():
        raise ValueError(f'{name} must have finite values and shape {shape}')
    return value


def logit(value, eps=EPS):
    """Explicit bounded-coordinate transform; callers report its clipping."""
    if not 0 < eps < .5:
        raise ValueError('eps must be in (0,.5)')
    value = _finite(value)
    bounded = np.clip(value, eps, 1-eps)
    return np.log(bounded)-np.log1p(-bounded)


def sigmoid(value):
    value = _finite(value)
    magnitude = np.exp(-np.abs(value))
    return np.where(value >= 0, 1/(1+magnitude), magnitude/(1+magnitude))


def to_residual(raw, anchor_upper, eps=EPS):
    raw = _finite(raw)
    anchor = _finite(anchor_upper, (9,), 'independent anchor')
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError('Raw trajectory must be [T,9]')
    return logit(raw, eps)-logit(anchor, eps)


def resample_trajectory(values, frames):
    """Endpoint-preserving time resampling for descriptors/oracle coverage only."""
    values = _finite(values)
    if values.ndim != 2 or values.shape[1] != 9 or len(values) == 0:
        raise ValueError('Trajectory must be nonempty [T,9]')
    if type(frames) is not int or frames < 1:
        raise ValueError('frames must be a positive integer')
    if len(values) == frames:
        return values.copy()
    return np.stack([np.interp(np.linspace(0, 1, frames),
        np.linspace(0, 1, len(values)), values[:, j]) for j in range(9)], -1)


def _runs(valid):
    edge = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def clipping_statistics(upper, valid, anchor_upper, eps=EPS):
    """Count unique observed source samples, including clamp-induced energy loss."""
    raw = _array(upper)
    valid = np.asarray(valid)
    anchor = _finite(anchor_upper, (9,), 'independent anchor')
    if raw.ndim != 2 or raw.shape[1] != 9 or valid.dtype != np.bool_ or valid.shape != raw.shape[:1]:
        raise ValueError('Expected [T,9] motion and Boolean [T] mask')
    x = _finite(raw[valid]); bounded = np.clip(x, eps, 1-eps)
    energy, clipped_energy = np.zeros(9), np.zeros(9)
    for start, end in _runs(valid):
        y = raw[start:end]; z = np.clip(y, eps, 1-eps)
        energy += np.square(y-y.mean(0)).sum(0)
        clipped_energy += np.square(z-z.mean(0)).sum(0)
    return {'observed_frames': int(valid.sum()), 'observed_values': int(x.size),
        'clipped_values': int(((x < eps) | (x > 1-eps)).sum()),
        'out_of_domain_values': int(((x < 0) | (x > 1)).sum()),
        'raw_range': [float(x.min()), float(x.max())] if x.size else None,
        'absolute_clipping_error_sum': float(np.abs(x-bounded).sum()),
        'dynamic_energy': energy.tolist(), 'clipped_dynamic_energy': clipped_energy.tolist(),
        'anchor_clipped_values': int(((anchor < eps) | (anchor > 1-eps)).sum()),
        'anchor_out_of_domain_values': int(((anchor < 0) | (anchor > 1)).sum())}


def extract_units(clips, fit_ids, durations=(16, 32, 48), prefix=8, stride=8):
    """Extract training-only complete native windows; no gap/tail compression.

    A hash of source clip and future start selects one available duration. The
    future starts at ``start``; the prefix is exactly [start-prefix,start).
    Recording-end truncated futures are not units or duration observations.
    """
    durations = tuple(durations); fit_ids = list(fit_ids)
    if (not durations or durations != tuple(sorted(set(durations)))
            or any(type(x) is not int or x < 1 for x in durations)
            or type(prefix) is not int or prefix < 1 or type(stride) is not int or stride < 1):
        raise ValueError('Positive prefix/stride and increasing unique durations required')
    if len(fit_ids) != len(set(fit_ids)):
        raise ValueError('Fitting allowlist contains duplicate indices')
    units, seen = [], set()
    for i in fit_ids:
        c = clips[i]; clip_id = str(c['clip_id'])
        if clip_id in seen:
            raise ValueError('Fitting sources must have unique clip_id')
        seen.add(clip_id)
        upper = _array(c['upper']); valid = np.asarray(c['valid'])
        if upper.ndim != 2 or upper.shape[1] != 9 or valid.dtype != np.bool_ or valid.shape != upper.shape[:1]:
            raise ValueError('Expected source [T,9] and Boolean [T] validity')
        if not np.isfinite(upper[valid]).all():
            raise ValueError('Observed upper motion must be finite')
        anchor = _finite(c['anchor_upper'], (9,), 'independent anchor')
        global_vec = _finite(c['global'])
        if global_vec.ndim != 1 or not len(global_vec):
            raise ValueError('Global emotion vector must be nonempty [D]')
        stats = clipping_statistics(upper, valid, anchor)
        for left, right in _runs(valid):
            for start in range(left+prefix, right, stride):
                available = [d for d in durations if start+d <= right]
                if not available:
                    break
                key = f'joint_motion_unit_20260918:{clip_id}:{start}'.encode()
                d = available[int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % len(available)]
                raw_prefix = upper[start-prefix:start].copy()
                raw_future = upper[start:start+d].copy()
                units.append({'clip_id': clip_id, 'start': int(start), 'duration': int(d),
                    'prefix': to_residual(raw_prefix, anchor), 'future': to_residual(raw_future, anchor),
                    'raw_prefix': raw_prefix, 'raw_future': raw_future,
                    'anchor_upper': anchor.copy(), 'global': global_vec.copy(),
                    'source_clipping': stats, 'eps': EPS})
    return units


def _validate_units(units):
    if not units:
        raise ValueError('At least one fitting unit required')
    p = len(units[0]['prefix']); dim = len(units[0]['global'])
    for unit in units:
        _finite(unit['prefix'], (p, 9), 'unit prefix')
        _finite(unit['future'], (unit['duration'], 9), 'unit future')
        _finite(unit['global'], (dim,), 'unit global')
        if p < 1 or type(unit['duration']) is not int or unit['duration'] < 1:
            raise ValueError('Nonempty prefix and positive native duration required')
    return p, dim


def fit_dictionary(units, k=128, seed=20260918):
    """Clip-balanced clustering with unique real medoids, using supplied units only.

    Scaling/clustering give every source clip total weight one. A cluster center
    is never decoded: the nearest not-yet-used real source unit is retained.
    Exact waveform duplicates may remain if the fitting corpus contains them.
    """
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits

    units = list(units); prefix_frames, global_dim = _validate_units(units)
    if type(k) is not int or k < 1 or len(units) < k:
        raise ValueError('k must be positive and no larger than fitting unit count')
    counts = Counter(str(u['clip_id']) for u in units)
    weights = np.asarray([1/counts[str(u['clip_id'])] for u in units], dtype=np.float64)
    weights /= weights.sum()
    wave = np.stack([np.concatenate((u['prefix'], resample_trajectory(u['future'], 32)), 0)
        for u in units])
    mu = np.einsum('n,ntc->c', weights, wave)/(prefix_frames+32)
    var = np.einsum('n,ntc->c', weights, np.square(wave-mu))/(prefix_frames+32)
    scales = np.maximum(np.sqrt(var), .05)
    descriptors = (wave/scales).reshape(len(units), -1).astype(np.float32)
    globals_ = np.stack([u['global'] for u in units])
    global_mean = (weights[:, None]*globals_).sum(0)
    global_scales = np.maximum(np.sqrt((weights[:, None]*(globals_-global_mean)**2).sum(0)), .05)
    with threadpool_limits(limits=1):
        clustering = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=min(2048, len(units)),
            n_init=3, max_iter=100, reassignment_ratio=0.)
        clustering.fit(descriptors, sample_weight=weights*len(units))
    selected, distances = [], []
    for center in clustering.cluster_centers_:
        d = np.square(descriptors-center).mean(1)
        d[selected] = np.inf
        ix = int(np.argmin(d)); selected.append(ix); distances.append(float(d[ix]))
    medoids = [copy.deepcopy(units[i]) for i in selected]
    by_source = {}
    for unit in units:
        if 'source_clipping' in unit:
            by_source.setdefault(str(unit['clip_id']), copy.deepcopy(unit['source_clipping']))
    clipping = {'unique_source_count': len(by_source), 'by_source': by_source}
    for key in ('observed_frames', 'observed_values', 'clipped_values', 'out_of_domain_values',
                'absolute_clipping_error_sum', 'anchor_clipped_values', 'anchor_out_of_domain_values'):
        clipping[key] = sum(v[key] for v in by_source.values())
    energy = np.sum([v['dynamic_energy'] for v in by_source.values()], axis=0) if by_source else np.zeros(9)
    bounded_energy = np.sum([v['clipped_dynamic_energy'] for v in by_source.values()], axis=0) if by_source else np.zeros(9)
    clipping['pooled_dynamic_rms_retention'] = np.sqrt(bounded_energy/np.maximum(energy, 1e-15)).tolist()
    return {'schema': SCHEMA, 'eps': EPS, 'k': k, 'requested_k': k, 'seed': seed,
        'prefix_frames': prefix_frames, 'descriptor_future_frames': 32, 'global_dim': global_dim,
        'medoids': medoids, 'scales': scales, 'global_mean': global_mean, 'global_scales': global_scales,
        'descriptors': descriptors[selected].copy(), 'prefix_descriptors': wave[selected, :prefix_frames]/scales,
        'future_descriptors': wave[selected, prefix_frames:]/scales,
        'medoid_global': globals_[selected].copy(), 'medoid_unit_indices': selected,
        'medoid_center_mse': distances, 'source_unit_count': len(units),
        'source_clip_ids': sorted(counts), 'source_clip_count': len(counts),
        'source_unit_counts': dict(sorted(counts.items())), 'clip_balanced_fit': True,
        'input_clipping': clipping}


def _check_dictionary(dictionary):
    if dictionary.get('schema') != SCHEMA or not dictionary.get('medoids'):
        raise ValueError('Unknown or empty joint motion dictionary')
    return len(dictionary['medoids'])


def prior_logits(dictionary, global_vec, history_residual=None, *, use_global=True,
                 history_weight=1., global_weight=1.):
    """Deployment prior using global audio emotion and generated past only.

    No query motion, future, per-query normalizer, or teacher token is accepted.
    The empirical global vector is the frozen audio embedding, not a GT label.
    """
    k = _check_dictionary(dictionary)
    if not np.isfinite([history_weight, global_weight]).all() or min(history_weight, global_weight) < 0:
        raise ValueError('Prior weights must be finite and nonnegative')
    result = np.zeros(k, dtype=np.float64)
    if use_global:
        global_vec = _finite(global_vec, (dictionary['global_dim'],), 'global emotion vector')
        result -= global_weight*np.square((dictionary['medoid_global']-global_vec)/dictionary['global_scales']).mean(1)
    if history_residual is not None:
        history = _finite(history_residual, (dictionary['prefix_frames'], 9), 'generated history')
        result -= history_weight*np.square(dictionary['prefix_descriptors']-history/dictionary['scales']).mean((1, 2))
    return result


def decode_unit(dictionary, index, anchor9, previous_raw=None, blend_frames=8):
    """Decode native-duration absolute trajectory, with a finite seam correction.

    The first generated frame equals the bounded previous raw endpoint. At frame
    ``blend_frames-1`` correction is exactly zero; subsequent frames match the
    medoid under the target independent anchor. No accumulated delta is retained.
    """
    k = _check_dictionary(dictionary)
    if not isinstance(index, (int, np.integer)) or not 0 <= index < k:
        raise ValueError('Medoid index out of range')
    if type(blend_frames) is not int or blend_frames < 0:
        raise ValueError('blend_frames must be a nonnegative integer')
    anchor = _finite(anchor9, (9,), 'independent anchor')
    logits = np.asarray(dictionary['medoids'][index]['future']).copy()+logit(anchor, dictionary['eps'])
    if previous_raw is not None and blend_frames:
        previous = _finite(previous_raw, (9,), 'generated previous endpoint')
        count = min(blend_frames, len(logits))
        if count == 1:
            raise ValueError('A seam correction needs at least two frames to decay to zero')
        correction = logit(previous, dictionary['eps'])-logits[0]
        logits[:count] += np.linspace(1., 0., count)[:, None]*correction
    return sigmoid(logits)


def oracle_nearest(dictionary, prefix_residual, future_residual, *, prefix_weight=1., future_weight=1.):
    """GT oracle coverage only: nearest normalized full joint waveform.

    All tokens use the same provided prefix frame count and 32 resampled future
    frames. Native token durations are not evaluated as teacher event labels.
    """
    _check_dictionary(dictionary)
    prefix = _finite(prefix_residual, (dictionary['prefix_frames'], 9), 'oracle GT prefix')
    future = resample_trajectory(future_residual, dictionary['descriptor_future_frames'])
    if not np.isfinite([prefix_weight, future_weight]).all() or min(prefix_weight, future_weight) < 0 or prefix_weight+future_weight <= 0:
        raise ValueError('Oracle weights must be nonnegative and at least one positive')
    prefix_count = dictionary['prefix_frames']; future_count = dictionary['descriptor_future_frames']
    prefix_error = np.square(dictionary['prefix_descriptors']-prefix/dictionary['scales']).mean((1, 2))
    future_error = np.square(dictionary['future_descriptors']-future/dictionary['scales']).mean((1, 2))
    distances = (prefix_weight*prefix_count*prefix_error+future_weight*future_count*future_error)/(prefix_weight*prefix_count+future_weight*future_count)
    return {'index': int(np.argmin(distances)), 'distances': distances,
        'prefix_mse': prefix_error, 'future_mse': future_error,
        'prefix_frames': prefix_count, 'future_descriptor_frames': future_count,
        'uses_query_future': True}


def oracle_reconstruct(dictionary, prefix_residual, future_residual, anchor9, *, prefix_weight=1., future_weight=1.):
    """Nearest-unit oracle with GT future length; never a deployment routine."""
    result = oracle_nearest(dictionary, prefix_residual, future_residual,
        prefix_weight=prefix_weight, future_weight=future_weight)
    motion = resample_trajectory(dictionary['medoids'][result['index']]['future'], len(future_residual))
    result['raw'] = sigmoid(motion+logit(_finite(anchor9, (9,), 'independent anchor'), dictionary['eps']))
    result['uses_query_length'] = True
    return result
