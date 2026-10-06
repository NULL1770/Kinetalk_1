"""Frozen TRAIN-only within-clip residual lag moments; no inference fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.diagnose_flow_sampling import sha
from scripts.packed_trainval_cache import load_packed
from scripts.phase1_condition_diagnostic import _identity_cache
from scripts.train_full_staged import base_forward, batch_identity


def clip_lag_moments(values, observed):
    """Count/cross/left-energy/right-energy/delta-energy after clip centering."""
    value, mask = np.asarray(values, dtype=np.float64), np.asarray(observed, dtype=bool)
    if value.ndim != 2 or value.shape != mask.shape or not np.isfinite(value[mask]).all():
        raise ValueError('Finite observed native [T,C] values required')
    clean = np.where(mask, value, 0.)
    mean = clean.sum(0) / np.maximum(mask.sum(0), 1)
    centered = np.where(mask, clean-mean, 0.)
    adjacent = mask[1:] & mask[:-1]
    left, right = centered[:-1], centered[1:]
    return np.stack([adjacent.sum(0), np.where(adjacent, left*right, 0.).sum(0),
                     np.where(adjacent, left**2, 0.).sum(0), np.where(adjacent, right**2, 0.).sum(0),
                     np.where(adjacent, (right-left)**2, 0.).sum(0)])


def summarize_lags(moments):
    count, cross, left, right, delta = moments
    denominator = np.sqrt(left*right)
    rho = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator>0)
    return {'adjacent_count': count.astype(np.int64).tolist(), 'cross_sum': cross.tolist(),
            'left_square_sum': left.tolist(), 'right_square_sum': right.tolist(),
            'raw_rho': rho.tolist(), 'source_rho': np.clip(rho, -1+1e-4, 1-1e-4).tolist(),
            'demeaned_delta_mean_square': (delta/np.maximum(count, 1)).tolist()}


@torch.no_grad()
def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh temporal audit required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert sha(a.checkpoint) == a.checkpoint_sha256
    spread = json.loads(a.source_stats.read_text())
    assert spread['checkpoint_sha256'] == a.checkpoint_sha256 and spread['source_fit_split'] == 'train'
    assert not spread['test_loaded'] and not spread['validation_motion_used']
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    data = load_packed(a.data, materialize=False, with_refs=True)
    assert set(data['splits']) == {'train', 'validation'} and not data['provenance']['test_loaded']
    manifest = data['provenance']['manifest_sha256']
    assert manifest == ck['data_manifest_sha256'] == spread['data_manifest_sha256']
    system = NeutralAffectSystem(ck['config']).to(a.device).eval()
    system.load_state_dict(ck['system'], strict=True); system.requires_grad_(False)
    assert system.residual_support.cpu().tolist() == spread['residual_support']
    q = data['splits']['train']; speakers = set(q['speaker_id'].tolist())
    identities = _identity_cache(system, {**data, 'refs': {k:v for k,v in data['refs'].items() if k in speakers}}, a.device)
    rows, ids, speaker_ids, emotions = [], [], [], []
    for batch_no, ix in enumerate(torch.arange(len(q['_lengths'])).split(a.batch_size)):
        b = q.batch(ix, a.device)
        base = base_forward(system, b['content'], b['valid']); ident = batch_identity(identities, b)
        observed = b['valid'][..., None] & b['channel_mask'][:, None] & system.residual_support[None, None]
        residual = (b['motion']-base['b0']-ident['baseline'][:, None]).cpu().numpy()
        masks = observed.cpu().numpy()
        for j in range(len(ix)): rows.append(clip_lag_moments(residual[j], masks[j]))
        ids.extend(b['clip_id']); speaker_ids.extend(b['speaker_id'].cpu().tolist()); emotions.extend(b['emotion_id'].cpu().tolist())
        if batch_no%50 == 0: print(json.dumps({'event':'batch','clips':len(ids),'total':len(q['_lengths'])}),flush=True)
    rows = np.asarray(rows); speaker_ids, emotions = np.asarray(speaker_ids), np.asarray(emotions)
    report = {'schema':'frozen_train_demeaned_residual_lag_v1', 'test_loaded':False,
              'validation_motion_used':False,'training_performed':False,'source_fit_split':'train',
              'train_clips':len(ids),'checkpoint_sha256':a.checkpoint_sha256,'data_manifest_sha256':manifest,
              'source_stats_sha256':sha(a.source_stats),'script_sha256':sha(Path(__file__)),
              'residual_support':spread['residual_support'],'nondegeneracy_rho_margin':1e-4,
              'policy':'Within-clip observed-frame centered; native lag1 adjacent valid frames only; no interpolation/gap compression. Pearson moments pooled across TRAIN; rho clipped only for nonsingular AR innovations.',
              'results':summarize_lags(rows.sum(0)),
              'groups':{kind:{str(k):{'clips':int((values==k).sum()),**summarize_lags(rows[values==k].sum(0))}
                             for k in np.unique(values)} for kind,values in [('speaker',speaker_ids),('emotion',emotions)]}}
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    np.savez_compressed(a.output/'per_clip.npz',moments=rows,clip_id=np.asarray(ids),speaker=speaker_ids,emotion=emotions)
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--source-stats',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=16);run(p.parse_args())
