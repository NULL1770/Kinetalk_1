"""Small clip-balanced Student-t event prior, never an audio timing model.

Fitting consumes the fixed extractor's complete events only. It fails closed on
insufficient event support; unsupported or left-censored intervals are not
converted into known waiting times. A fitted distribution is not a semantic
teacher validation or evidence of perceptual naturalness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp


DF = 5.
ITERATIONS = 20
MIN_EVENTS = 8
MIN_CLIPS = 4
DIM = 13


def event_vector(event):
    p, r = np.asarray(event['p5'], float), np.asarray(event['r5'], float)
    durations = [event[k] for k in ('d_onset', 'd_apex', 'd_release')]
    if (p.shape != (5,) or r.shape != (5,) or not np.isfinite(np.r_[p, r]).all()
            or not event.get('source_id') or event.get('left_censored')
            or event.get('right_censored') or any(int(d) != d for d in durations)
            or durations[0] < 4 or durations[1] < 0 or durations[2] < 4):
        raise ValueError('Complete finite raw-coefficient event with native durations required')
    return np.r_[p, r, np.log(durations[0]), np.log1p(durations[1]), np.log(durations[2])]


def clip_weights(events):
    ids = [e['source_id'] for e in events]
    counts = {cid: ids.count(cid) for cid in set(ids)}
    return np.array([1/counts[cid]/len(counts) for cid in ids])


def t_logpdf(x, loc, scale):
    delta = np.asarray(x)-loc
    chol = np.linalg.cholesky(scale)
    squared = np.square(np.linalg.solve(chol, delta.T)).sum(0)
    d = delta.shape[-1]
    logp = (gammaln((DF+d)/2)-gammaln(DF/2)-d/2*np.log(DF*np.pi)
            -np.log(np.diag(chol)).sum()-(DF+d)/2*np.log1p(squared/DF))
    return logp, squared


def _regularize(cov):
    # Small sample shrinkage specified before fitting; never tuned on dev.
    return .8*cov+.2*np.diag(np.diag(cov))+np.eye(len(cov))*1e-4


def fit_marks(events):
    if len(events) < MIN_EVENTS or len({e['source_id'] for e in events}) < MIN_CLIPS:
        raise ValueError('Insufficient complete events: need at least8 events from4 clips')
    x = np.stack([event_vector(e) for e in events])
    weights = clip_weights(events)
    center = (weights[:, None]*x).sum(0)
    scale = np.maximum(np.sqrt((weights[:, None]*(x-center)**2).sum(0)),
                       np.r_[np.full(10, .005), np.full(3, .1)])
    z = (x-center)/scale
    k = min(8, max(1, len(x)//16))
    # Deterministic farthest-first seeds, ties retain extractor order.
    indices = [int(np.argmax(np.sum(z*z, axis=1)))]
    for _ in range(1, k):
        distances = np.min([np.sum((z-z[i])**2, axis=1) for i in indices], axis=0)
        indices.append(int(np.argmax(distances)))
    loc = z[indices].copy()
    initial_cov = _regularize((z*weights[:, None]).T@z)
    cov = np.stack([initial_cov.copy() for _ in range(k)])
    mix = np.full(k, 1/k)
    trace = []
    for _ in range(ITERATIONS):
        logp, distance = zip(*(t_logpdf(z, loc[j], cov[j]) for j in range(k)))
        logp, distance = np.array(logp).T, np.array(distance).T
        joint = logp+np.log(mix)[None]
        marginal = logsumexp(joint, axis=1)
        resp = np.exp(joint-marginal[:, None])
        weighted = weights[:, None]*resp
        mass = weighted.sum(0)
        u = (DF+DIM)/(DF+distance)
        mix = np.maximum(mass, 1e-8); mix /= mix.sum()
        for j in range(k):
            wu = weighted[:, j]*u[:, j]
            loc[j] = (wu[:, None]*z).sum(0)/max(wu.sum(), 1e-12)
            dz = z-loc[j]
            cov[j] = _regularize((dz*wu[:, None]).T@dz/max(mass[j], 1e-12))
        trace.append(float(-np.sum(weights*marginal)+np.log(scale).sum()))
    components = [{'weight': float(mix[j]), 'loc': (center+scale*loc[j]).tolist(),
        'scale': (scale[:, None]*cov[j]*scale[None]).tolist(), 'df': DF} for j in range(k)]
    final = logsumexp(np.stack([t_logpdf(x, np.array(c['loc']), np.array(c['scale']))[0]
                               +np.log(c['weight']) for c in components]), axis=0)
    return components, {'events': len(events), 'clips': len(set(e['source_id'] for e in events)),
        'components': k, 'iterations': ITERATIONS, 'nll_trace_penalized_em': trace,
        'fit_clip_equal_nll': float(-np.dot(weights, final)),
        'evaluation_kind': 'in_sample_diagnostic_not_generalization',
        'normalization_center': center.tolist(), 'normalization_scale': scale.tolist(),
        'clip_equal': True, 'shrinkage': .2}


def fit_waits(waits):
    records = [w for w in waits if w['supported_hold'] and not w['left_censored']
               and w['duration_frames'] >= 4]
    completed = [w for w in records if not w['right_censored']]
    if len(completed) < 4:
        return None, {'status': 'insufficient_known_wait_endings', 'observed': len(completed),
                      'right_censored': len(records)-len(completed), 'learned_waiting': False}
    ids = [w['source_id'] for w in records]
    counts = {cid: ids.count(cid) for cid in set(ids)}
    weights = np.array([1/counts[cid] for cid in ids])
    bins = max(1, int(np.ceil(max(w['duration_frames']-4 for w in records)/12.5)))
    # Integer13frames at25Hz is .52s; the exact sampler contract uses13.
    bin_frames = 13
    bins = min(20, max(1, bins))
    exposure, endings = np.zeros(bins), np.zeros(bins)
    for w, weight in zip(records, weights):
        duration = int(w['duration_frames'])-4
        for frame in range(duration+int(not w['right_censored'])):
            exposure[min(frame//bin_frames, bins-1)] += weight
        if not w['right_censored']:
            endings[min(duration//bin_frames, bins-1)] += weight
    p_frame = (endings+.5)/(exposure+25.)
    hazards = 1-(1-p_frame)**bin_frames
    return hazards.tolist(), {'status': 'fitted', 'observed': len(completed),
        'right_censored': len(records)-len(completed), 'learned_waiting': True,
        'exposure_clip_balanced_frames': exposure.tolist(), 'endings_clip_balanced': endings.tolist(),
        'frame_hazards': p_frame.tolist(), 'bin_frames': bin_frames,
        'left_censored_excluded': sum(w['left_censored'] for w in waits)}


def fit(events, waits, require_waiting=True):
    hazards, wait_diagnostics = fit_waits(waits)
    if hazards is None and require_waiting:
        raise ValueError('Insufficient reliable waiting times; formal rhythm prior not fitted')
    components, diagnostics = fit_marks(events)
    x = np.stack([event_vector(e) for e in events])
    # Diagnostic fallback is explicit, never presented as fitted natural rhythm.
    model = {'schema_version': 'sparse_brow_event_v1', 'fps': 25,
        'mark_space': 'coefficient_delta', 'components': components,
        'wait_hazard': hazards if hazards is not None else [1-(1-1/25.)**13],
        'wait_bin_frames': 13, 'minimum_hold_frames': 4,
        'duration_bounds': {'onset': [4, 75], 'apex': [0, 50], 'release': [4, 75]},
        'max_peak_offset': np.maximum(np.quantile(np.abs(x[:, :5]), .99, axis=0), .035).tolist(),
        'max_return_offset': np.maximum(np.quantile(np.abs(x[:, 5:10]), .99, axis=0), .01).tolist(),
        'fit_clip_ids': sorted({e['source_id'] for e in events}),
        'fit_diagnostics': diagnostics, 'waiting_diagnostics': wait_diagnostics,
        'verified_semantic_ground_truth': False, 'audio_timing_learned': False,
        'naturalness_verified': False, 'prototype_only': True}
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-diagnostic-wait', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh output required')
    teacher = json.loads(args.teacher.read_text(encoding='utf8'))
    model = fit(teacher['events'], teacher['waits'], not args.allow_diagnostic_wait)
    model['teacher_sha256'] = hashlib.sha256(args.teacher.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(model['fit_diagnostics']))


if __name__ == '__main__':
    main()
