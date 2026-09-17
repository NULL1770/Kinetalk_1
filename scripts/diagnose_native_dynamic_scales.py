"""Fixed ridge diagnostics inside the existing fit pool, without dev selection.

Three fixed time resolutions and five hypothetical feature lags are reported.
They are sensitivity checks, never estimates of physical audio/video offset.
All preprocessing and fitting use only the new fitting sentence groups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.full_staged_data import sha, _renderer_cache_binding
from scripts.probe_dynamic_predictability import bin_frames, center_controls, sentence_split
from scripts.train_formal_predictable_projection import save_json

CHANNELS = (41, 42, 43, 44, 45, 5, 6, 12, 13, 17)
GROUPS = {'brows': list(range(5)), 'eyes_expression': list(range(5, 9)), 'jaw': [9]}
LAGS = (-8, -4, 0, 4, 8)
STRIDES = (4, 8, 16)


def lagged_common_support(features, valid, lag, radius=8):
    """Positive lag supplies future audio at the target time; no wraparound."""
    if features.ndim != 3 or valid.shape != features.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('Expected features and Boolean native masks')
    if type(lag) is not int or abs(lag) > radius or features.shape[1] <= 2 * radius:
        raise ValueError('Invalid lag or insufficient native frames')
    positions = torch.arange(radius, features.shape[1] - radius, device=valid.device)
    # Every lag uses exactly the same observed frames, including internal gaps.
    common = valid[:, positions].clone()
    for offset in range(-radius, radius + 1):
        common &= valid[:, positions + offset]
    selected = features[:, positions + lag]
    return torch.where(common[..., None], selected, 0.), common, positions


def feature_scale(x, weight, ids):
    xx, ww = x[ids].double(), weight[ids].double()[..., None]
    return ((xx.square() * ww).sum((0, 1)) / ww.sum()).sqrt().clamp_min(.001).float()


def metrics(pred, target, weight):
    rows = {}
    for name, cc in GROUPS.items():
        p = center_controls(pred[..., cc], weight).double()
        t = center_controls(target[..., cc], weight).double()
        w = weight.double()[..., None]
        error, energy, penergy = ((p-t).square()*w).sum(), (t.square()*w).sum(), (p.square()*w).sum()
        rows[name] = {'centered_mse': float(error / (w.sum()*len(cc))),
                     'r2_vs_zero': float(1-error/energy.clamp_min(1e-15)),
                     'correlation': float((p*t*w).sum()/(penergy*energy).sqrt().clamp_min(1e-15)),
                     'rms_ratio': float((penergy/energy.clamp_min(1e-15)).sqrt()),
                     'target_energy': float(energy/(w.sum()*len(cc)))}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'audio', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = torch.load(args.cache, map_location='cpu', weights_only=False, mmap=True)
    audio = torch.load(args.audio, map_location='cpu', weights_only=False, mmap=True)
    if _renderer_cache_binding(audio) != sha(args.cache):
        raise ValueError('Audio and renderer cache binding differ')
    q, a = cache['splits']['train']['q'], audio['splits']['train']
    if q['clip_id'] != a['clip_id'] or a['sentence_id'] != list(q['sentence_id']):
        raise ValueError('Fit clip or sentence order differs')
    for key in ('valid', 'times'):
        if not torch.equal(q[key], a[key]):
            raise ValueError('Native audio/motion clock or mask differs')
    if not q['channel_mask'][:, list(CHANNELS)].all():
        raise ValueError('Require observed diagnostic channels')
    valid = q['valid'].to(args.device)
    if not torch.allclose(q['times'][:, 1:]-q['times'][:, :-1], torch.full_like(q['times'][:, 1:], .04), atol=1e-7, rtol=1e-5):
        raise ValueError('Expected native 25Hz clock')
    raw = a['features'].to(args.device)
    truth = q['motion'][..., list(CHANNELS)].float().to(args.device)
    _, support, positions = lagged_common_support(raw, valid, 0)
    keep = support.sum(1) >= 16
    eligible = keep.nonzero(as_tuple=True)[0].cpu()
    sentences = [str(q['sentence_id'][i]) for i in eligible]
    fit, hold = sentence_split(sentences, 2026091708, heldout_fraction=.2)
    fit, hold = fit.to(args.device), hold.to(args.device)
    raw, truth, valid = raw[keep], truth[keep], valid[keep]
    split = {name: {'clips': [q['clip_id'][int(eligible[i])] for i in indices.cpu()],
                    'sentences': sorted({sentences[i] for i in indices.cpu()})}
             for name, indices in (('fit', fit), ('internal_sentence_holdout', hold))}
    if set(split['fit']['sentences']) & set(split['internal_sentence_holdout']['sentences']):
        raise RuntimeError('Sentence split overlap')
    report = {'schema': 'native_scale_lag_diagnostic_v1', 'seed': 2026091708,
              'scope': 'Existing train membership only; internal sentence holdout, not untouched test. No pretrained trunk.',
              'fit_pool_clips': len(q['clip_id']), 'eligible_clips': int(keep.sum()),
              'omitted_insufficient_common_support': int((~keep).sum()),
              'dev_or_test_tensor_indexed': False, 'split': split, 'strides': list(STRIDES), 'lags': list(LAGS),
              'lag_definition': 'Positive uses future audio; identical common observation support for all lags. Not physical AV offset estimation.',
              'regression': 'All 1540 raw features, weighted clip-centered; inner-fit RMS only; fixed ridge alpha=1, no model selection.',
              'limitations': 'A fixed linear diagnostic is neither an upper bound on nonlinear predictability nor a trained generator.',
              'input_sha256': {'cache': sha(args.cache), 'audio': sha(args.audio), 'script': sha(__file__)}, 'results': {}}
    args.output.mkdir(parents=True)
    save_json(args.output/'protocol.json', report)
    for stride in STRIDES:
        for lag in LAGS:
            shifted, mask, positions = lagged_common_support(raw, valid, lag)
            x, weight = bin_frames(shifted, mask, stride)
            y, yw = bin_frames(truth[:, positions], mask, stride)
            assert torch.equal(weight, yw)
            x, y = center_controls(x, weight), center_controls(y, weight)
            scale = feature_scale(x, weight, fit)
            x = x / scale
            xf, yf, wf = x[fit].reshape(-1, x.shape[-1]), y[fit].reshape(-1, y.shape[-1]), weight[fit].reshape(-1, 1)
            # Normalize the empirical risk by observed frame count before alpha=1.
            covariance = xf.T @ (xf * wf) / wf.sum()
            covariance.diagonal().add_(1.)
            cross = xf.T @ (yf * wf) / wf.sum()
            coefficients = torch.linalg.solve(covariance, cross)
            pred = x @ coefficients
            key = f'stride{stride}/lag{lag:+d}'
            report['results'][key] = {name: metrics(pred[ix], y[ix], weight[ix]) for name, ix in (('fit', fit), ('internal_sentence_holdout', hold))}
            print(json.dumps({'case': key, 'heldout': report['results'][key]['internal_sentence_holdout']}), flush=True)
            save_json(args.output/'report.json', report)
    report['seconds'] = time.monotonic()-started
    save_json(args.output/'report.json', report)
    print('NATIVE_SCALE_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
