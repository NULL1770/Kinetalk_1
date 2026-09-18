"""Training-only clip-centered motion shapes on a fixed native frame clock.

Clip centering is a motion-teacher operation, performed only when extracting
training targets. A single observed-clip mean is subtracted, preserving slow
excursions within and between windows. Deployment receives coefficient chunks and
their frame starts; overlap-add never reads a query target, estimates its mean
or amplitude, or clips the generated coefficients. The caller supplies the
separate predicted slow level and owns missing-run and final-tail scheduling.
"""
from __future__ import annotations

from collections import Counter
from numbers import Integral

import numpy as np


SCHEMA = 'clocked_clip_centered_motion_dictionary_v1'
CHANNELS = 9


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _finite(value, *, shape=None, name='value'):
    value = _array(value)
    if (shape is not None and value.shape != shape) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return value


def _runs(valid):
    edge = np.diff(np.r_[False, valid, False].astype(np.int8))
    return zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))


def extract_windows(clips, fit_ids, horizon=32, hop=16):
    """Read only the fitting allowlist, retaining full uninterrupted windows.

    Each valid run has its own clock: starts are ``run_start + n * hop``,
    including its first frame. Incomplete tails are excluded from fitting, not
    stretched or padded. ``future`` is raw upper9 minus the training clip's
    observed-frame mean, shared across all valid runs and windows. Missing
    frames do not enter the mean or compress the frame clock. ``raw_mean`` and
    ``global`` are retained for training provenance; the mean is not an audio
    condition. No anchor or motion history is required.
    """
    horizon = _integer(horizon, 'horizon')
    hop = _integer(hop, 'hop')
    ids = [_integer(i, 'fit index', minimum=0) for i in fit_ids]
    if len(ids) != len(set(ids)):
        raise ValueError('Fitting allowlist contains duplicate indices')
    if any(i >= len(clips) for i in ids):
        raise ValueError('Fitting index outside clips')
    windows, seen = [], set()
    global_dim = None
    for i in ids:
        clip = clips[i]
        clip_id = clip['clip_id']
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
            raise ValueError('Fitting sources require unique nonempty clip_id')
        seen.add(clip_id)
        raw = _array(clip['upper'])
        valid = clip['valid']
        if hasattr(valid, 'detach'):
            valid = valid.detach().cpu().numpy()
        valid = np.asarray(valid)
        if (raw.ndim != 2 or raw.shape[1] != CHANNELS or valid.dtype != np.bool_
                or valid.shape != raw.shape[:1] or not np.isfinite(raw[valid]).all()):
            raise ValueError('Expected finite observed upper[T,9] and Boolean valid[T]')
        global_vec = _finite(clip['global'], name='global')
        if global_vec.ndim != 1 or not len(global_vec):
            raise ValueError('global must be a nonempty vector')
        if global_dim is not None and len(global_vec) != global_dim:
            raise ValueError('All fitting global vectors must have the same width')
        global_dim = len(global_vec)
        observed = raw[valid]
        if not len(observed):
            continue
        # A single training-clip DC origin preserves low-frequency waveform
        # structure. Per-window means would erase slow expressive excursions.
        shifted = observed-observed[:1]
        raw_mean = observed[0]+shifted.mean(axis=0)
        if not np.isfinite(raw_mean).all():
            raise ValueError('Clip motion mean overflowed')
        for left, right in _runs(valid):
            for start in range(int(left), int(right)-horizon+1, hop):
                future = raw[start:start+horizon]
                centered = future-raw_mean
                if not np.isfinite(centered).all() or not np.isfinite(raw_mean).all():
                    raise ValueError('Centered motion window overflowed')
                windows.append({'clip_id': clip_id, 'start': int(start),
                    'horizon': horizon, 'hop': hop, 'future': centered.copy(),
                    'raw_mean': raw_mean.copy(), 'global': global_vec.copy(),
                    'centering_scope': 'training_clip_observed_frames'})
    return windows


def fit_shape_dictionary(windows, k=128, seed=20260918):
    """Fit clip-balanced scales/clusters and retain distinct real shape medoids.

    Every source clip contributes total weight one, independent of its number
    of complete windows. Clustering sees clip-centered coefficient shapes;
    expressive levels and global audio vectors are not clustering features.
    Shape scale is channel RMS with a fixed .005 floor. Selected waveform
    values are exact real training windows, including their nonzero local mean;
    the k-means cluster center is never a decoded trajectory. There is no
    second per-window centering operation during fitting or decoding.
    """
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits

    windows = list(windows)
    k = _integer(k, 'k')
    seed = _integer(seed, 'seed', minimum=0)
    if not windows or k > len(windows):
        raise ValueError('k must be no larger than a nonempty fitting window count')
    horizon = _integer(windows[0]['horizon'], 'horizon')
    hop = _integer(windows[0]['hop'], 'hop')
    values, sources = [], []
    for window in windows:
        if window.get('horizon') != horizon or window.get('hop') != hop:
            raise ValueError('All training windows require the same fixed horizon and hop')
        clip_id = window.get('clip_id')
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError('Training window requires a nonempty clip_id')
        start = _integer(window.get('start'), 'window start', minimum=0)
        value = _finite(window['future'], shape=(horizon, CHANNELS), name='future')
        if window.get('centering_scope') != 'training_clip_observed_frames':
            raise ValueError('Training shape requires explicit observed-clip centering provenance')
        _finite(window['raw_mean'], shape=(CHANNELS,), name='training clip mean')
        values.append(value.copy())
        sources.append({'clip_id': clip_id, 'start': start})
    shapes = np.stack(values)
    counts = Counter(source['clip_id'] for source in sources)
    weights = np.asarray([1/counts[source['clip_id']] for source in sources], dtype=np.float64)
    weights /= weights.sum()
    channel_energy = np.einsum('n,ntc->c', weights, np.square(shapes))/horizon
    scales = np.maximum(np.sqrt(channel_energy), .005)
    descriptors = (shapes/scales).reshape(len(shapes), -1)
    if not np.isfinite(descriptors).all() or not np.isfinite(scales).all():
        raise ValueError('Shape descriptors or scales overflowed')
    with threadpool_limits(limits=1):
        clustering = MiniBatchKMeans(n_clusters=k, random_state=seed,
            batch_size=min(2048, len(windows)), n_init=3, max_iter=100,
            reassignment_ratio=0.)
        clustering.fit(descriptors, sample_weight=weights*len(windows))
    selected, distances = [], []
    for center in clustering.cluster_centers_:
        error = np.square(descriptors-center).mean(axis=1)
        error[selected] = np.inf
        index = int(np.argmin(error))
        selected.append(index)
        distances.append(float(error[index]))
    return {'schema': SCHEMA, 'horizon': horizon, 'hop': hop, 'k': k, 'seed': seed,
        'scales': scales, 'shapes': shapes[selected].copy(),
        'source_clip_ids': sorted(counts), 'source_clip_count': len(counts),
        'source_window_count': len(windows), 'source_window_counts': dict(sorted(counts.items())),
        'window_weights': weights, 'clip_balanced_fit': True,
        'medoid_window_indices': selected, 'medoid_sources': [sources[i] for i in selected],
        'medoid_center_mse': distances,
        'coordinate_system': 'raw coefficients minus training clip observed-frame mean; no window recentering',
        'centering_scope': 'training_clip_observed_frames',
        'scale_floor': .005}


def overlap_add(chunks, starts, total_frames, horizon=32):
    """Positive triangular normalized overlap-add, returning values and support.

    ``chunks`` is [N,horizon,9]; starts are unique increasing frame indices.
    Chunks may extend beyond the requested output tail and are truncated there.
    Uncovered frames are zero with a false coverage flag. This function neither
    generates windows across missing runs nor modifies coefficient magnitudes.
    Endpoints have weight 1/(ceil(horizon/2)), so even a single-frame tail is
    covered. The caller adds its independently predicted slow level afterward.
    """
    horizon = _integer(horizon, 'horizon')
    total_frames = _integer(total_frames, 'total_frames', minimum=0)
    chunks = _finite(chunks, name='chunks')
    if chunks.ndim != 3 or chunks.shape[1:] != (horizon, CHANNELS):
        raise ValueError('chunks must have shape [N,horizon,9]')
    starts = [_integer(start, 'chunk start', minimum=0) for start in starts]
    if (len(starts) != len(chunks) or any(a >= b for a, b in zip(starts, starts[1:]))
            or any(start >= total_frames for start in starts)):
        raise ValueError('One unique increasing in-range start is required per chunk')
    clock = np.arange(horizon, dtype=np.float64)
    weights = np.minimum(clock+1, horizon-clock)
    weights /= weights.max()
    output = np.zeros((total_frames, CHANNELS), dtype=np.float64)
    mass = np.zeros(total_frames, dtype=np.float64)
    for chunk, start in zip(chunks, starts):
        end = min(start+horizon, total_frames)
        weight = weights[:end-start]
        output[start:end] += chunk[:end-start]*weight[:, None]
        mass[start:end] += weight
    covered = mass > 0
    output[covered] /= mass[covered, None]
    if not np.isfinite(output).all():
        raise ValueError('Overlap-add coefficients overflowed')
    return output, covered


def target_distances(targets, shapes, scales):
    """Normalized Euclidean RMS distances [B,K] without target preprocessing.

    Targets must already use the caller's chosen coordinate system. In a
    supervised target construction they may be clip-centered teacher windows. This
    routine does not subtract their means, fit scales, or choose a token.
    """
    from scipy.spatial.distance import cdist

    shapes = _finite(shapes, name='shapes')
    targets = _finite(targets, name='targets')
    scales = _finite(scales, shape=(CHANNELS,), name='scales')
    if (shapes.ndim != 3 or not len(shapes) or shapes.shape[1] < 1
            or shapes.shape[2] != CHANNELS or targets.ndim != 3
            or targets.shape[1:] != shapes.shape[1:] or (scales <= 0).any()):
        raise ValueError('Expected targets[B,H,9], nonempty shapes[K,H,9] and positive scales[9]')
    width = int(np.prod(shapes.shape[1:]))
    x = (targets/scales).reshape(len(targets), width)
    y = (shapes/scales).reshape(len(shapes), width)
    result = cdist(x, y, metric='euclidean')/np.sqrt(width)
    if not np.isfinite(result).all():
        raise ValueError('Normalized distances overflowed')
    return result


def pairwise_distances(shapes, scales):
    """Symmetric [K,K] normalized Euclidean RMS distances of shape prototypes."""
    return target_distances(shapes, shapes, scales)
