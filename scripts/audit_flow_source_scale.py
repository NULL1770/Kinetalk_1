"""Fit diagonal residual spread from frozen real TRAIN observations only."""
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


class ChannelMoments:
    def __init__(self, channels):
        self.count = np.zeros(channels, dtype=np.int64)
        self.total = np.zeros(channels, dtype=np.float64)
        self.square = np.zeros(channels, dtype=np.float64)

    def add(self, value, observed):
        value, observed = np.asarray(value, dtype=np.float64), np.asarray(observed, dtype=bool)
        if value.shape != observed.shape or value.shape[-1] != len(self.count):
            raise ValueError('Values and observations must match')
        if not np.isfinite(value[observed]).all():
            raise ValueError('Nonfinite observed value')
        clean = np.where(observed, value, 0.).reshape(-1, len(self.count))
        self.count += observed.reshape(-1, len(self.count)).sum(0)
        self.total += clean.sum(0)
        self.square += np.square(clean).sum(0)

    def result(self, residual_scale):
        if not np.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError('Positive normalization required')
        mean = self.total / np.maximum(self.count, 1)
        variance = np.maximum(self.square / np.maximum(self.count, 1) - mean ** 2, 0.)
        std = np.sqrt(variance)
        # Keep a nonsingular Gaussian even for constant observed targets.
        # Unobserved channels retain unit source scale and fixed support.
        normalized = np.where(self.count > 0, np.maximum(std / residual_scale, 1e-4), 1.)
        return {'count': self.count.tolist(), 'residual_mean': mean.tolist(),
                'residual_std': std.tolist(), 'normalized_source_std': normalized.tolist(),
                'normalization': float(residual_scale), 'nondegeneracy_floor_normalized': 1e-4}


@torch.no_grad()
def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert sha(a.checkpoint) == a.checkpoint_sha256
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    data = load_packed(a.data, materialize=False, with_refs=True)
    assert set(data['splits']) == {'train', 'validation'} and not data['provenance']['test_loaded']
    manifest = data['provenance']['manifest_sha256']
    assert manifest == ck['data_manifest_sha256']
    system = NeutralAffectSystem(ck['config']).to(a.device).eval()
    system.load_state_dict(ck['system'], strict=True)
    system.requires_grad_(False)
    q = data['splits']['train']
    speakers = set(q['speaker_id'].tolist())
    identities = _identity_cache(system, {**data, 'refs': {k: v for k, v in data['refs'].items() if k in speakers}}, a.device)
    moments = ChannelMoments(int(data['config']['data']['motion_dim']))
    for batch_no, ix in enumerate(torch.arange(len(q['_lengths'])).split(a.batch_size)):
        b = q.batch(ix, a.device)
        base = base_forward(system, b['content'], b['valid'])
        ident = batch_identity(identities, b)
        observed = b['valid'][..., None] & b['channel_mask'][:, None] & system.residual_support[None, None]
        residual = b['motion'] - base['b0'] - ident['baseline'][:, None]
        moments.add(residual.cpu().numpy(), observed.cpu().numpy())
        if batch_no % 50 == 0:
            print(json.dumps({'event': 'batch', 'clips': int(ix[-1])+1, 'total': len(q['_lengths'])}), flush=True)
    report = {'schema': 'frozen_train_residual_source_spread_v1', 'test_loaded': False,
              'validation_motion_used': False, 'training_performed': False, 'source_fit_split': 'train',
              'train_clips': len(q['_lengths']), 'checkpoint_sha256': a.checkpoint_sha256,
              'data_manifest_sha256': manifest, 'script_sha256': sha(Path(__file__)),
              'residual_support': system.residual_support.cpu().tolist(),
              'policy': 'Population residual std across observed TRAIN frames including clip means; no conditional/oracle inference, source centering or output scaling',
              'decision': 'Diagnostic only; scale skew alone is not proof of generation failure',
              **moments.result(system.residual_scale)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'event': 'complete', 'output': str(a.output)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    run(p.parse_args())
