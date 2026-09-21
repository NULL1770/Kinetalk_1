"""Read-only decomposition of paper-v1 mouth regression, including saved video inputs.

Uses saved validation curves and six validation enrollment shards only. No fitting,
test read, temporal realignment, or prediction modification is performed. Lag is
reported as a diagnostic and never applied to a generated trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_full_staged import base_forward

MOUTH = list(range(14, 41))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def mouth_metrics(x, y, valid, channels):
    x, y = x.double()[..., MOUTH], y.double()[..., MOUTH]
    m = valid[..., None] & channels[:, None, MOUTH]
    n = m.sum(1, keepdim=True).clamp_min(1)
    xc = torch.where(m, x - torch.where(m, x, 0.).sum(1, keepdim=True) / n, 0.)
    yc = torch.where(m, y - torch.where(m, y, 0.).sum(1, keepdim=True) / n, 0.)
    pair = m[:, 1:] & m[:, :-1]
    return {
        'raw_mse': float((x - y)[m].square().mean()),
        'centered_mse': float((xc - yc)[m].square().mean()),
        'centered_r2': float(1 - (xc - yc).square().sum() / yc.square().sum().clamp_min(1e-12)),
        'centered_correlation': float((xc * yc).sum() / (xc.square().sum() * yc.square().sum()).sqrt().clamp_min(1e-12)),
        'rms_ratio': float((xc.square().sum() / yc.square().sum().clamp_min(1e-12)).sqrt()),
        'velocity_mse': float((x[:, 1:] - x[:, :-1] - y[:, 1:] + y[:, :-1])[pair].square().mean()),
        'outside_fraction': float(((x[m] < 0) | (x[m] > 1)).double().mean()),
    }


def lag_scan(x, y, valid, channel=17, radius=10):
    result = []
    for lag in range(-radius, radius + 1):
        if lag < 0:
            a, b, keep = x[-lag:, channel], y[:lag, channel], valid[-lag:] & valid[:lag]
        elif lag > 0:
            a, b, keep = x[:-lag, channel], y[lag:, channel], valid[:-lag] & valid[lag:]
        else:
            a, b, keep = x[:, channel], y[:, channel], valid
        a, b = a[keep].double(), b[keep].double()
        if len(a) < 12 or a.std() < 1e-6 or b.std() < 1e-6:
            corr = None
        else:
            a, b = a - a.mean(), b - b.mean()
            corr = float((a * b).sum() / (a.square().sum() * b.square().sum()).sqrt())
        result.append({'lag_frames': lag, 'correlation': corr})
    defined = [r for r in result if r['correlation'] is not None]
    return {'zero_lag_correlation': result[radius]['correlation'],
            'best_lag': max(defined, key=lambda r: r['correlation']) if defined else None,
            'convention': 'positive lag compares prediction[t] with target[t+lag]; diagnostic only',
            'target_std': float(y[valid, channel].std()), 'prediction_std': float(x[valid, channel].std())}


@torch.no_grad()
def audit(root, output):
    torch.set_num_threads(4)
    run = root / 'run_backup/audio'
    curve_path = run / 'dynamics/curves.pt'
    checkpoint_path = run / 'audio/final.pt'
    for path, stage, key in ((curve_path, 'dynamics', 'curves_sha256'),
                             (checkpoint_path, 'audio', 'final_sha256')):
        complete = json.loads((run / stage / 'complete.json').read_text(encoding='utf8'))
        if sha(path) != complete[key]:
            raise ValueError('Completed run binding differs: ' + str(path))
    curves = torch.load(curve_path, map_location='cpu', weights_only=False, mmap=True)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    system = NeutralAffectSystem(saved['config']).eval()
    system.load_state_dict(saved['system'], strict=True)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    index = json.loads((root / 'data_index.json').read_text(encoding='utf8'))
    refs = manifest['roles']['val']['enrollment']
    rows = {r['clip_id']: r for r in manifest['roles']['val']['query']}
    records = {r['clip_id']: r for r in index['records']}
    baseline = {}
    ref_audit = []
    with tarfile.open(root / 'data_backup.tar') as archive:
        for person in sorted({r['speaker'] for r in refs}):
            part = [r for r in refs if r['speaker'] == person]
            shards = []
            for row in part:
                rec = records[row['clip_id']]
                member = 'kinetalk_paper_full_v1/' + rec['path']
                blob = archive.extractfile(member).read()
                if hashlib.sha256(blob).hexdigest() != rec['sha256']:
                    raise ValueError('Enrollment shard digest differs: ' + row['clip_id'])
                d = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
                if d['row'] != row:
                    raise ValueError('Enrollment metadata differs')
                shards.append(d)
                ref_audit.append({'clip_id': row['clip_id'], 'sha256': rec['sha256']})
            length = max(len(d['valid']) for d in shards)
            content = torch.zeros(len(shards), length, 768)
            motion = torch.zeros(len(shards), length, 52)
            valid = torch.zeros(len(shards), length, dtype=torch.bool)
            channel_mask = torch.stack([d['channel_mask'] for d in shards])
            for j, d in enumerate(shards):
                n = len(d['valid'])
                content[j, :n] = d['content']; motion[j, :n] = d['motion']; valid[j, :n] = d['valid']
            base = base_forward(system, content, valid)
            residual = torch.where(valid[..., None] & channel_mask[:, None], motion - base['b0'], 0.)
            baseline[person] = system.encode_identity(residual[None], valid[None])['baseline'][0]
    identity = torch.stack([baseline[rows[cid]['speaker']] for cid in curves['clip_id']])
    b0 = curves['b0']
    stages = {'b0': b0, 'b0_plus_identity': torch.where(curves['valid'][..., None], b0 + identity[:, None], 0.),
              'stage4_seed42': curves['predictions']['42/base'],
              'stage5_seed42': curves['predictions']['42/full']}
    target, valid, channels = curves['target'], curves['valid'], curves['channel_mask']
    result = {'schema': 'paper_v1_mouth_regression_audit_v1', 'test_loaded': False,
        'fit_performed': False, 'validation_clips': len(curves['clip_id']),
        'mouth_channels': MOUTH, 'curves_sha256': sha(curve_path), 'stage4_checkpoint_sha256': sha(checkpoint_path),
        'reference_shards': ref_audit, 'aggregate': {}, 'by_emotion_scope': {}, 'jaw_open_lag_summary': {}, 'selected_video_examples': {},
        'stage_evaluations': {}}
    for name, prediction in stages.items():
        result['aggregate'][name] = mouth_metrics(prediction, target, valid, channels)
        for emotion_scope in ('neutral', 'nonneutral'):
            ids = [i for i, cid in enumerate(curves['clip_id']) if (rows[cid]['emotion'] == 0) == (emotion_scope == 'neutral')]
            group = result['by_emotion_scope'].setdefault(emotion_scope, {'clips': len(ids), 'metrics': {}})
            group['metrics'][name] = mouth_metrics(prediction[ids], target[ids], valid[ids], channels[ids])
        lags = [lag_scan(prediction[i], target[i], valid[i]) for i in range(len(valid))]
        defined = [r for r in lags if r['best_lag'] is not None]
        zero = [r['zero_lag_correlation'] for r in defined if r['zero_lag_correlation'] is not None]
        result['jaw_open_lag_summary'][name] = {
            'clips_with_nonconstant_jaw': len(defined),
            'median_zero_lag_correlation': float(np.median(zero)),
            'mean_zero_lag_correlation': float(np.mean(zero)),
            'median_best_lag_frames': float(np.median([r['best_lag']['lag_frames'] for r in defined])),
            'best_lag_histogram': {str(lag): sum(r['best_lag']['lag_frames'] == lag for r in defined) for lag in range(-10, 11)}}
    for stage in ('articulation', 'identity', 'teacher', 'audio', 'dynamics'):
        d = json.loads((run / stage / 'evaluation.json').read_text(encoding='utf8'))
        result['stage_evaluations'][stage] = {'condition_source': d['condition_source'],
            'mouth': d['modes']['42/full']['metrics']['mouth'],
            'interpretation': 'Includes as-yet-untrained renderer; not an identity quality or audio generation metric.' if stage == 'identity' else 'Teacher uses target-motion oracle, not audio-only inference.' if stage == 'teacher' else 'Saved stage-specific development evaluation.'}
    for path in sorted((root / 'recovery/visual_v1').glob('*.npz')):
        z = np.load(path, allow_pickle=False)
        cid = str(z['clip_id'].item()); i = curves['clip_id'].index(cid); n = len(z['valid'])
        if not np.array_equal(z['motions'][0], target[i, :n].numpy()):
            raise ValueError('Visual target differs from curve')
        if not np.array_equal(z['motions'][1], curves['predictions']['42/base'][i, :n].numpy()):
            raise ValueError('Visual stage4 differs from curve')
        speaker = rows[cid]['speaker']; report_path = root / 'videos' / speaker / 'display_report.json'
        display_report = json.loads(report_path.read_text(encoding='utf8'))
        display_z = np.load(report_path.with_name('display_curves.npz'), allow_pickle=False)
        # Reproduce the historical renderer input transformation independently
        # of today's renderer fixes. This is an audit of saved old artifacts.
        good = np.flatnonzero(z['valid'])
        nearest = good[np.abs(np.arange(n)[:, None] - good[None]).argmin(1)]
        expected_display = np.clip(z['motions'][:, nearest], 0, 1)
        observed = np.broadcast_to(z['channel_mask'], z['motions'].shape)
        corrected_display = np.where(observed, expected_display, 0.)
        item = {'frames': n, 'raw_video_input_sha256': sha(path),
            'legacy_unmasked_display_replay_max_error': float(np.max(np.abs(expected_display - display_z['motions']))),
            'corrected_supported_display_delta_max': float(np.max(np.abs(corrected_display - display_z['motions']))),
            'corrected_observed_channels_display_max_error': float(np.max(np.abs(corrected_display - display_z['motions'])[observed])),
            'audio_trim_start_seconds': display_report['audio_trim_start_seconds'],
            'mouth_exact_stage4_copy': {str(z['mode_names'][j]): bool(np.array_equal(z['motions'][j, :, MOUTH], z['motions'][1, :, MOUTH])) for j in range(2, 6)},
            'metrics': {}, 'jaw_open': {}}
        for name, prediction in stages.items():
            item['metrics'][name] = mouth_metrics(prediction[i:i+1], target[i:i+1], valid[i:i+1], channels[i:i+1])
            item['jaw_open'][name] = lag_scan(prediction[i, :n], target[i, :n], valid[i, :n])
        result['selected_video_examples'][cid] = item
    result['interpretation'] = [
        'Five rendered prediction modes share exactly the same mouth coefficients from stage4.',
        'A teacher-oracle evaluation is not an audio-only quality certificate.',
        'The stage4 mouth weakness predates the stage5 upper-face branch.',
        'A diagnostic lag scan does not certify audiovisual synchronization and is not applied to output.',
        'Raw stored predictions are scored without display clamping.']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'output': str(output), 'aggregate': result['aggregate'], 'jaw_open': result['jaw_open_lag_summary']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/paper_training_20260920'))
    p.add_argument('--output', type=Path, default=Path('artifacts/paper_training_20260920/mouth_regression_audit_20260920.json'))
    args = p.parse_args(); audit(args.root, args.output)
