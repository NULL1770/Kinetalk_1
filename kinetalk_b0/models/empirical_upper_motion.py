"""Nonparametric TRAIN motion-distribution baseline, not an audio timing model.

The bank retains synchronized upper9 trajectories at their native 25 fps.
Every continuous valid run is smoothed with a five-frame triangular filter
whose boundary weights are renormalized, then centered per run/channel.
Sampling selects one sufficiently long TRAIN run and one uniform contiguous
crop. It never changes polarity, concatenates runs, interpolates time, or fits
a query target. Conditional retrieval uses clip-level audio/global and identity
similarity; a moving result is not evidence of audio-to-event prediction.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


SCHEMA = 'empirical_upper_motion_v1'
STAT_FLOOR = 1e-6


def _validate_sequence(value, valid, name):
    if (not torch.is_tensor(value) or value.ndim != 3 or value.shape[-1] != 9
            or value.dtype not in (torch.float32, torch.float64) or min(value.shape[:2]) < 1):
        raise ValueError(f'{name} must be nonempty float32/float64 [B,T,9]')
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != value.shape[:2]
            or valid.device != value.device):
        raise ValueError('valid must be Boolean [B,T] on the sequence device')
    if not torch.isfinite(value[valid]).all():
        raise ValueError(f'Observed {name} must be finite')


def _codes(value, rows, width, name):
    if (not torch.is_tensor(value) or value.shape != (rows, width)
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite floating [{rows},{width}]')
    return value.detach().to(device='cpu', dtype=torch.float64).clone()


def _metadata(values, rows, name, *, required=False):
    if not isinstance(values, (tuple, list)) or len(values) != rows:
        raise ValueError(f'{name} must contain one metadata value per clip')
    if required:
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != rows:
            raise ValueError(f'{name} must contain unique nonempty string IDs')
    elif any(value is not None and (type(value) not in (str, int) or value == '') for value in values):
        raise ValueError(f'{name} must contain nonempty strings, integer IDs or None')
    return list(values)


def _runs(valid):
    for row, flags in enumerate(valid.detach().cpu().tolist()):
        left = None
        for position, observed in enumerate(flags + [False]):
            if observed and left is None:
                left = position
            elif not observed and left is not None:
                yield row, left, position
                left = None


@torch.no_grad()
def fit_empirical_bank(upper, valid, global_code, identity, speakers, sentences, clip_ids):
    """Build a serializable bank from TRAIN-only clips supplied by the caller.

    Global/identity mean and population standard deviation weight each input
    clip once, irrespective of its duration or number of runs. Statistics use
    float64 and a 1e-6 standard-deviation floor. No automatic train/dev partition
    inference occurs: the caller must restrict inputs to its declared fit fold.
    Runs are stored once in a packed CPU tensor; no invalid frame is retained.
    """
    _validate_sequence(upper, valid, 'upper')
    if not valid.any(1).all():
        raise ValueError('Every fit clip must contain an observed frame')
    count = len(upper)
    global_code = _codes(global_code, count, 64, 'global_code')
    identity = _codes(identity, count, 128, 'identity')
    speakers = _metadata(speakers, count, 'speakers')
    sentences = _metadata(sentences, count, 'sentences')
    clip_ids = _metadata(clip_ids, count, 'clip_ids', required=True)
    stats = {}
    standardized = {}
    for name, values in (('global', global_code), ('identity', identity)):
        mean = values.mean(0)
        std = values.std(0, unbiased=False).clamp_min(STAT_FLOOR)
        stats[name + '_mean'], stats[name + '_std'] = mean, std
        standardized[name] = (values - mean) / std
    source = upper.detach().cpu()
    groups = {}
    descriptors = list(_runs(valid))
    for index, (row, left, right) in enumerate(descriptors):
        groups.setdefault(right-left, []).append((index, row, left, right))
    residuals = [None] * len(descriptors)
    weights = source.new_tensor([1., 2., 3., 2., 1.]) / 9
    kernel = weights.reshape(1, 1, 5).expand(9, 1, 5).contiguous()
    for length, runs in groups.items():
        batch = torch.stack([source[row, left:right] for _, row, left, right in runs])
        smoothed = F.conv1d(batch.transpose(1, 2), kernel, padding=2, groups=9)
        support = F.conv1d(source.new_ones(1, 1, length), weights.reshape(1, 1, 5), padding=2)
        smoothed = (smoothed / support).transpose(1, 2)
        centered = smoothed - smoothed.mean(1, keepdim=True)
        for index, value in zip((run[0] for run in runs), centered):
            residuals[index] = value
    lengths = torch.tensor([right-left for _, left, right in descriptors], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]))
    return {'schema': SCHEMA, 'fps': 25, 'window': 5,
            'smoothing': 'triangular [1,2,3,2,1]/9 with available-frame boundary renormalization',
            'centering': 'per donor valid run and channel before cropping; crops are not recentered',
            'statistics_scope': 'caller-supplied fit clips only, equal clip weighting',
            'residuals': torch.cat(residuals), 'run_offsets': offsets, 'run_lengths': lengths,
            'run_clip_indices': torch.tensor([row for row, _, _ in descriptors], dtype=torch.long),
            'run_starts': torch.tensor([left for _, left, _ in descriptors], dtype=torch.long),
            'global_standardized': standardized['global'], 'identity_standardized': standardized['identity'],
            'stats': stats, 'speakers': speakers, 'sentences': sentences, 'clip_ids': clip_ids}


def _validate_bank(bank):
    if (not isinstance(bank, dict) or bank.get('schema') != SCHEMA
            or bank.get('fps') != 25 or bank.get('window') != 5):
        raise ValueError('Unsupported empirical motion bank')
    clip_ids = bank.get('clip_ids')
    if not isinstance(clip_ids, list) or not clip_ids:
        raise ValueError('Empty or invalid empirical bank clip metadata')
    count = len(clip_ids)
    _metadata(clip_ids, count, 'bank clip_ids', required=True)
    _metadata(bank.get('speakers'), count, 'bank speakers')
    _metadata(bank.get('sentences'), count, 'bank sentences')
    values = bank.get('residuals')
    if (not torch.is_tensor(values) or values.ndim != 2 or values.shape[1] != 9
            or values.device.type != 'cpu' or values.dtype not in (torch.float32, torch.float64)
            or not len(values) or not torch.isfinite(values).all()):
        raise ValueError('Bank residuals must be packed finite CPU upper9')
    lengths = bank.get('run_lengths')
    if not torch.is_tensor(lengths) or lengths.ndim != 1 or not len(lengths):
        raise ValueError('Invalid bank run lengths')
    for key in ('run_offsets', 'run_lengths', 'run_clip_indices', 'run_starts'):
        value = bank.get(key)
        if (not torch.is_tensor(value) or value.shape != lengths.shape
                or value.dtype != torch.long or value.device.type != 'cpu'):
            raise ValueError(f'Invalid bank index: {key}')
    expected = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]))
    if ((lengths < 1).any() or int(lengths.sum()) != len(values)
            or not torch.equal(bank['run_offsets'], expected) or (bank['run_starts'] < 0).any()
            or (bank['run_clip_indices'] < 0).any() or (bank['run_clip_indices'] >= count).any()):
        raise ValueError('Inconsistent empirical bank run indices')
    stats = bank.get('stats')
    if not isinstance(stats, dict):
        raise ValueError('Missing bank fit statistics')
    for name, width in (('global', 64), ('identity', 128)):
        code = bank.get(name + '_standardized')
        if (not torch.is_tensor(code) or code.shape != (count, width) or code.device.type != 'cpu'
                or code.dtype != torch.float64 or not torch.isfinite(code).all()):
            raise ValueError(f'Invalid standardized bank {name}')
        for suffix in ('mean', 'std'):
            value = stats.get(name + '_' + suffix)
            if (not torch.is_tensor(value) or value.shape != (width,) or value.dtype != torch.float64
                    or value.device.type != 'cpu' or not torch.isfinite(value).all()
                    or (suffix == 'std' and (value <= 0).any())):
                raise ValueError(f'Invalid bank {name} {suffix}')


@torch.no_grad()
def sample_empirical_motion(bank, center, valid, global_code, identity, speakers, sentences,
                            seeds, temperature, top_k=32, mode='conditional'):
    """Return (samples[S,B,T,9], donor_metadata[S][B][query_run]).

    Eligible runs must be long enough and must not share a known query speaker
    OR sentence. None means unavailable metadata, so that exclusion is skipped.
    Conditional distance is standardized global MSE + .25 identity MSE; ties
    retain bank order. The nearest top_k eligible runs are sampled uniformly.
    Unconditional mode ignores code distances and samples the entire same pool.

    Every seed owns a CPU Generator; global RNG state is unchanged. A selected
    crop uses an independent uniform integer offset and the same native frames
    for all nine channels. Crop means are not removed. Sign-specific headroom
    tanh bounds valid output around its supplied center. At temperature zero,
    the output is an exact center copy and donor metadata is empty. Nonzero
    sampling raises if any query run has no eligible donor; no fallback warping
    or hidden relaxation of speaker/sentence exclusion is performed.
    """
    _validate_bank(bank)
    _validate_sequence(center, valid, 'center')
    if ((center[valid] < 0) | (center[valid] > 1)).any():
        raise ValueError('Observed center must lie in [0,1]')
    count = len(center)
    global_code = _codes(global_code, count, 64, 'global_code')
    identity = _codes(identity, count, 128, 'identity')
    speakers = _metadata(speakers, count, 'speakers')
    sentences = _metadata(sentences, count, 'sentences')
    if (not isinstance(seeds, (list, tuple)) or not seeds
            or any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds)):
        raise ValueError('seeds must be a nonempty list of integer seeds in [0,2**63)')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if type(top_k) is not int or top_k < 1 or mode not in ('conditional', 'unconditional'):
        raise ValueError('Require positive top_k and conditional/unconditional mode')
    output = center.unsqueeze(0).expand(len(seeds), *center.shape).clone()
    metadata = [[[] for _ in range(count)] for _ in seeds]
    if temperature == 0:
        return output, metadata
    query_global = (global_code-bank['stats']['global_mean']) / bank['stats']['global_std']
    query_identity = (identity-bank['stats']['identity_mean']) / bank['stats']['identity_std']
    plans = []
    for row, left, right in _runs(valid):
        length = right-left
        eligible = []
        for run in (bank['run_lengths'] >= length).nonzero(as_tuple=True)[0].tolist():
            clip = int(bank['run_clip_indices'][run])
            if speakers[row] is not None and bank['speakers'][clip] == speakers[row]:
                continue
            if sentences[row] is not None and bank['sentences'][clip] == sentences[row]:
                continue
            eligible.append(run)
        if not eligible:
            raise ValueError(f'No eligible native donor run for query {row} [{left}:{right}] length={length} '
                             'after speaker/sentence exclusions; no stitching or time warp is allowed')
        distance = None
        candidates = torch.tensor(eligible, dtype=torch.long)
        if mode == 'conditional':
            clip = bank['run_clip_indices'][candidates]
            distance = ((bank['global_standardized'][clip]-query_global[row]).square().mean(-1)
                        + .25*(bank['identity_standardized'][clip]-query_identity[row]).square().mean(-1))
            order = torch.argsort(distance, stable=True)[:top_k]
            candidates, distance = candidates[order], distance[order]
        plans.append((row, left, right, candidates, distance, len(eligible)))
    epsilon = torch.finfo(center.dtype).eps
    for sample, seed in enumerate(seeds):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        for row, left, right, candidates, distance, eligible_count in plans:
            pick = int(torch.randint(len(candidates), (1,), generator=generator))
            run = int(candidates[pick])
            clip = int(bank['run_clip_indices'][run])
            length = right-left
            donor_length = int(bank['run_lengths'][run])
            crop_offset = int(torch.randint(donor_length-length+1, (1,), generator=generator))
            packed_start = int(bank['run_offsets'][run])+crop_offset
            delta = bank['residuals'][packed_start:packed_start+length].to(center)*temperature
            base = center[row, left:right]
            room = torch.where(delta >= 0, 1-base, base)
            output[sample, row, left:right] = base + room*torch.tanh(delta/room.clamp_min(epsilon))
            metadata[sample][row].append({'query_start': left, 'query_length': length,
                'donor_run_index': run, 'donor_clip_id': bank['clip_ids'][clip],
                'donor_speaker': bank['speakers'][clip], 'donor_sentence': bank['sentences'][clip],
                'donor_run_start': int(bank['run_starts'][run]), 'donor_run_length': donor_length,
                'crop_offset': crop_offset, 'native_start': int(bank['run_starts'][run])+crop_offset,
                'eligible_run_count': eligible_count, 'sampling_pool_count': len(candidates),
                'conditional_rank': pick if mode == 'conditional' else None,
                'condition_distance': float(distance[pick]) if distance is not None else None,
                'seed': seed, 'mode': mode, 'temperature': float(temperature)})
    return output, metadata


__all__ = ['fit_empirical_bank', 'sample_empirical_motion']
