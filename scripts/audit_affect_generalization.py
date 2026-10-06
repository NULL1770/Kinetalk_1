"""Frozen train/validation condition audit; no fitting or generation selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_probe import classification_metrics
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts.audit_matched_motion_curves import RC, region_values, means
from scripts.diagnose_flow_sampling import sha
from scripts.packed_trainval_cache import load_packed
from scripts.phase1_condition_diagnostic import _identity_cache
from scripts.train_full_staged import audio_affect, base_forward, batch_identity, teacher_affect


def error_partition(error, labels):
    """Exact per-clip error identity: group-mean error plus within-group error.

    Group means are an attribution, never an inference correction. Held-out
    labels and teacher observations are allowed only inside this diagnostic.
    """
    error = np.asarray(error, dtype=np.float64)
    labels = np.asarray(labels)
    total = float(np.mean(error ** 2))
    offset, within, rows = 0., 0., {}
    for k in np.unique(labels):
        selected = labels == k
        values = error[selected]
        bias = values.mean(0)
        a = float(np.mean(bias ** 2))
        b = float(np.mean((values - bias) ** 2))
        weight = float(selected.mean())
        offset += weight * a
        within += weight * b
        rows[str(k)] = {'clips': int(selected.sum()), 'mse': float(np.mean(values ** 2)),
                        'group_bias_mse': a, 'centered_error_mse': b}
    np.testing.assert_allclose(total, offset + within, rtol=1e-9, atol=1e-12)
    return {'mse': total, 'group_bias_mse': offset, 'centered_error_mse': within,
            'group_bias_fraction': offset / total if total > 0 else 0., 'groups': rows}


def joint_labels(speaker, emotion, intensity):
    return np.asarray([f'{s}/{e}/{i}' for s, e, i in zip(speaker, emotion, intensity)])


def summarize(z, classes, train_means):
    x, y = z['audio_global'], z['teacher_global']
    error = x.astype(np.float64) - y
    labels = z['emotion']
    joint = z['emotion'] * 4 + z['intensity']
    target_residual = y - train_means[joint]
    counts = {str(k): int((labels == k).sum()) for k in np.unique(labels)}
    out = {'clips': len(labels), 'emotion_counts': counts,
           'audio_condition_readout': classification_metrics(torch.from_numpy(labels),
                torch.from_numpy(z['audio_emotion']), classes),
           'teacher_condition_readout': classification_metrics(torch.from_numpy(labels),
                torch.from_numpy(z['teacher_emotion']), classes),
           'audio_mean_square': float(np.mean(x ** 2)), 'teacher_mean_square': float(np.mean(y ** 2)),
           'global_error': {'speaker': error_partition(error, z['speaker']),
                            'emotion': error_partition(error, labels),
                            'speaker_emotion_intensity': error_partition(error,
                                  joint_labels(z['speaker'], labels, z['intensity']))},
           'teacher_deviation_from_train_emotion_intensity': error_partition(target_residual, z['speaker']),
           'neutral_b0_jaw': {}}
    for s in ['all', *np.unique(z['speaker'])]:
        selected = labels == 0
        if s != 'all':
            selected &= z['speaker'] == s
        out['neutral_b0_jaw'][str(s)] = {'clips': int(selected.sum()),
                                       **means(z['b0_jaw'][selected], RC)}
    return out


@torch.no_grad()
def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    ckpath = a.run_root / 'audio/final.pt'
    if sha(ckpath) != a.checkpoint_sha256:
        raise ValueError('Checkpoint differs from preregistered reference')
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    provenance = json.loads((a.run_root / 'provenance.json').read_text())
    recipe = provenance['recipe']
    assert ck['recipe_sha256'] == provenance['recipe_sha256'] and not recipe['test_loaded']
    data = load_packed(a.data, with_refs=True)
    assert set(data['splits']) == {'train', 'validation'} and not data['provenance']['test_loaded']
    assert data['provenance']['manifest_sha256'] == ck['data_manifest_sha256']
    system = NeutralAffectSystem(ck['config']).to(a.device).eval()
    system.load_state_dict(ck['system'], strict=True)
    audio = SlowStateAffect(ck['feature_stats']['mean'], ck['feature_stats']['std'],
                           stride=recipe['args']['stride']).to(a.device).eval()
    audio.load_state_dict(ck['audio'], strict=True)
    identities = _identity_cache(system, data, a.device)
    roles = {}
    for role, q in data['splits'].items():
        rows = {k: [] for k in ['audio_global', 'teacher_global', 'audio_emotion',
                'teacher_emotion', 'audio_intensity', 'teacher_intensity', 'identity_baseline',
                'gt_mean', 'b0_mean', 'b0_jaw']}
        for bn, ix in enumerate(torch.arange(len(q['_lengths'])).split(a.batch_size)):
            b = q.batch(ix, a.device)
            b.update(base_forward(system, b['content'], b['valid']))
            identity = batch_identity(identities, b)
            ao = audio_affect(audio, b['audio_features'], b['valid'])
            mt = teacher_affect(system, b, identity)
            for k, value in [('audio_global', ao['global']), ('teacher_global', mt['global']),
                 ('audio_emotion', ao['emotion_logits'].argmax(-1)),
                 ('teacher_emotion', mt['emotion_logits'].argmax(-1)),
                 ('audio_intensity', ao['intensity_logits'].argmax(-1)),
                 ('teacher_intensity', mt['intensity_logits'].argmax(-1)),
                 ('identity_baseline', identity['baseline'])]:
                rows[k].append(value.cpu().numpy())
            weight = b['valid'][..., None]
            for name, value in [('gt_mean', b['motion']), ('b0_mean', b['b0'])]:
                rows[name].append((torch.where(weight, value, 0.).sum(1) /
                                   weight.sum(1).clamp_min(1)).cpu().numpy())
            jaw = []
            for j in range(len(ix)):
                jaw.append(region_values(b['b0'][j].clamp(0, 1).cpu(), b['motion'][j].cpu(),
                                         b['valid'][j].cpu(), b['channel_mask'][j].cpu(), [17]))
            rows['b0_jaw'].append(np.asarray(jaw))
            if bn % 50 == 0:
                print(json.dumps({'role': role, 'clips': int(ix[-1]) + 1, 'total': len(q['_lengths'])}), flush=True)
        z = {k: np.concatenate(v) for k, v in rows.items()}
        for k, source in [('emotion', 'emotion_id'), ('intensity', 'intensity_id'), ('speaker', 'speaker_id')]:
            z[k] = q[source].numpy()
        z['clip_id'] = np.asarray(q['clip_id'])
        roles[role] = z
    train = roles['train']
    joint = train['emotion'] * 4 + train['intensity']
    class_mean = np.stack([train['teacher_global'][train['emotion'] == k].mean(0) for k in range(8)])
    joint_mean = np.stack([train['teacher_global'][joint == k].mean(0) if (joint == k).any()
                           else class_mean[k // 4] for k in range(32)])
    report = {'schema': 'frozen_affect_generalization_v1', 'test_loaded': False,
              'training_performed': False, 'generation_performed': False,
              'checkpoint_sha256': sha(ckpath), 'data_manifest_sha256': ck['data_manifest_sha256'],
              'source_sha256': {k: sha(p) for k, p in [('audit', Path(__file__)),
                                                        ('runner', Path(__file__).with_name('train_full_staged.py'))]},
              'b0_scope': 'Neutral clips only for speech-base accuracy; all other B0 rows descriptive',
              'partition_scope': 'Observed error attribution; group offsets are never deployable corrections',
              'centroid_scope': 'Real train teacher per emotion/intensity only; missingjoint falls back class mean',
              'results': {role: summarize(z, data['config']['data']['emotion_classes'], joint_mean)
                          for role, z in roles.items()}}
    a.output.mkdir(parents=True)
    arrays = {role + '__' + k: v for role, z in roles.items() for k, v in z.items()}
    np.savez_compressed(a.output / 'per_clip_conditions.npz', **arrays)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'complete': str(a.output)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
