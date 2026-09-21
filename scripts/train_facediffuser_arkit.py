"""Train the declared cached-audio FaceDiffuser ARKit adaptation on full data.

Random decoder initialization; no KineTalk prediction, mouth bypass, or query
target is used during sampling. Frozen clip conditions match the declared
global/reference information. This is not the original BEAT reproduction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_isolated_audio_state import load_context, trim
from scripts.train_full_staged import subset, obs, batch_identity, region_report
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, capture_rng, restore_rng, canonical_hash
from scripts.train_predictable_renderer import state_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.compact_native_curves import compact_curves
from scripts.paper_generation_report import report_generation
from kinetalk_b0.models.facediffuser_arkit import FaceDiffuserARKit, SOURCE_COMMIT


SEEDS = (42, 123, 2026)
CONDITION_LAYOUT = {'global_code': 64, 'identity_code': 128, 'anchors': 52, 'intensity_value': 1}


def clip_condition(batch):
    values = []
    for name, width in CONDITION_LAYOUT.items():
        value = batch[name]
        if value.shape != (len(batch['valid']), width) or not torch.isfinite(value).all():
            raise ValueError('Invalid frozen clip condition: ' + name)
        values.append(value.detach())
    return torch.cat(values, -1)


def audio_input(batch, feature_stats):
    """Expose every cached audio input using the same TRAIN-only statistics.

    Native content is preserved raw as in the articulation input; the complete
    audio_features stream (including its content copy) is normalized as in the
    audio/state branches. No target, label, or batch/query statistics are read.
    """
    raw, features, valid = batch['content'], batch['audio_features'], batch['valid']
    if (raw.ndim != 3 or features.ndim != 3 or raw.shape[:2] != valid.shape
            or features.shape[:2] != valid.shape or raw.dtype != features.dtype
            or raw.device != features.device or valid.dtype != torch.bool
            or valid.device != raw.device or not valid.any(1).all()
            or not torch.isfinite(raw[valid]).all() or not torch.isfinite(features[valid]).all()):
        raise ValueError('Finite observed aligned native content/audio_features required')
    mean = feature_stats['mean'].detach().to(device=features.device, dtype=features.dtype)
    std = feature_stats['std'].detach().to(device=features.device, dtype=features.dtype)
    if (mean.shape != (features.shape[-1],) or std.shape != mean.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Valid TRAIN-fitted audio feature statistics required')
    clean = torch.where(valid[..., None], features.detach(), mean)
    normalized = (clean - mean) / std
    content = torch.where(valid[..., None], raw.detach(), 0.)
    return torch.cat((content, normalized), -1)


def make_model(total_input_dim):
    # Explicit official main.py dimensions, rather than the class defaults.
    return FaceDiffuserARKit(content_dim=total_input_dim, latent_dim=512, gru_hidden=512,
                             num_layers=2, diffusion_steps=1000, dropout=.3,
                             clip_condition_dim=sum(CONDITION_LAYOUT.values()))


def training_step(model, optimizer, batch, generator, feature_stats):
    device = batch['motion'].device
    times = torch.randint(model.diffusion_steps, (len(batch['valid']),), generator=generator).to(device)
    noise = torch.randn(batch['motion'].shape, generator=generator).to(device)
    result = model.training_losses(batch['motion'], times, audio_input(batch, feature_stats), batch['valid'],
                                   noise=noise, channel_mask=batch['channel_mask'],
                                   clip_condition=clip_condition(batch))
    loss = result['loss'].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite FaceDiffuser x0 objective')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # No clipping modification to official Adam training; reject nonfinite grads.
    norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def status(output, record):
    save_json(output / 'status.json', record)
    print(json.dumps(record, allow_nan=False), flush=True)


@torch.no_grad()
def evaluate(model, data, system, identities, device, output, *, batch_size=16, smoke=False):
    model.eval()
    q = data['splits']['validation']
    predictions, seconds, emotions = {}, {}, {}
    started = time.monotonic()
    for seed in SEEDS:
        initial = torch.randn(q['motion'].shape, generator=torch.Generator().manual_seed(seed))
        generator = torch.Generator(device=device).manual_seed(seed + 1000003)
        chunks = []
        count = (len(q['valid']) + batch_size - 1) // batch_size
        for batch_index, ids in enumerate(torch.arange(len(q['valid'])).split(batch_size)):
            tick = time.monotonic()
            batch = trim(q, ids, device)
            status(output, {'status': 'sampling', 'seed': seed, 'batch': batch_index + 1,
                            'batches': count, 'batch_complete': False, 'diffusion_steps': model.diffusion_steps,
                            'clips_completed': batch_index * batch_size, 'clips': len(q['valid']),
                            'test_loaded': False})
            value = model.sample(audio_input(batch, data['feature_stats']), batch['valid'],
                                 initial_noise=initial[ids, :batch['valid'].shape[1]].to(device),
                                 generator=generator, channel_mask=batch['channel_mask'],
                                 clip_condition=clip_condition(batch)).cpu()
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite FaceDiffuser sampled motion')
            chunks.append(F.pad(value, (0, 0, 0, q['valid'].shape[1] - value.shape[1])))
            record = {'status': 'sampling', 'seed': seed, 'batch': batch_index + 1, 'batches': count,
                      'batch_complete': True, 'seconds': time.monotonic() - tick,
                      'diffusion_steps': model.diffusion_steps, 'clips_completed': min((batch_index + 1) * batch_size, len(q['valid'])),
                      'clips': len(q['valid']), 'test_loaded': False}
            seconds[f'{seed}/{batch_index + 1}'] = record['seconds']
            save_json(output / f'sampling_{seed}_{batch_index + 1:03d}.json', record)
            status(output, record)
        prediction = torch.cat(chunks)
        predictions[f'{seed}/full'] = prediction
        correct = 0
        for ids in torch.arange(len(q['valid'])).split(batch_size):
            batch = subset(q, ids, device)
            identity = batch_identity(identities, batch)
            residual = torch.where(obs(batch), prediction[ids].to(device) - batch['b0'] - identity['baseline'][:, None], 0.)
            logits = system.encode_motion(residual, batch['valid'])['emotion_logits']
            correct += int((logits.argmax(-1) == batch['emotion_id']).sum())
        emotions[f'{seed}/full'] = correct / len(q['valid'])
    report = {'schema': 'facediffuser_arkit_cached_evaluation_v1', 'clips': len(q['valid']),
              'scope': '4 TRAIN / 2 development smoke only' if smoke else 'complete 446-clip development; not final test',
              'noise_seeds': list(SEEDS), 'sampling_batch_seconds': seconds,
              'sampling_total_seconds': time.monotonic() - started,
              'regions': {key: region_report(pred, q) for key, pred in predictions.items()},
              'generated_teacher_accuracy_nonindependent': emotions,
              'official_beat_reproduction': False, 'test_loaded': False, 'default_replaced': False,
              'no_mouth_or_baseline_bypass': True, 'independent_emotion_AV_pending': True}
    curves = {'clip_id': q['clip_id'], 'target': q['motion'], 'valid': q['valid'], 'times': q['times'],
              'channel_mask': q['channel_mask'], 'b0': q['b0'], 'predictions': predictions, 'noise_seeds': list(SEEDS)}
    return report, curves


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'source', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('Positive epoch and batch budgets required')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh FaceDiffuser output required')
    if args.resume and not (args.output / 'protocol.json').exists():
        raise FileNotFoundError('Resume requires the original protocol')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, system, audio, identities, _ = load_context(args.data, args.source, args.device, args.seed)
    if set(data['splits']) != {'train', 'validation'} or data['provenance'].get('test_loaded', False):
        raise ValueError('Only approved TRAIN and development data allowed')
    if args.smoke:
        for role, size in [('train', 4), ('validation', 2)]:
            q = data['splits'][role]
            if len(q['valid']) < size:
                raise ValueError('Insufficient smoke coverage')
            data['splits'][role] = subset(q, torch.arange(size), 'cpu')
    elif [len(data['splits'][role]['valid']) for role in ('train', 'validation')] != [4098, 446]:
        raise ValueError('Formal FaceDiffuser run requires all 4098 TRAIN and 446 development clips')
    frozen = {'system': state_hash(system.state_dict()), 'audio': state_hash(audio.state_dict())}
    torch.manual_seed(args.seed)
    feature_stats = {key: data['feature_stats'][key].detach().cpu().clone() for key in ('mean', 'std')}
    content_dim = data['splits']['train']['content'].shape[-1]
    total_input_dim = content_dim + feature_stats['mean'].numel()
    model = make_model(total_input_dim).to(args.device)
    root = Path(__file__).resolve().parents[1]
    sources = ['scripts/train_facediffuser_arkit.py', 'kinetalk_b0/models/facediffuser_arkit.py',
               'scripts/train_isolated_audio_state.py', 'scripts/prepare_paper_full_data.py',
               'scripts/train_full_staged.py', 'scripts/paper_generation_report.py',
               'scripts/compact_native_curves.py', 'scripts/arkit_benchmark_report.py',
               'scripts/evaluate_arkit_literature_metrics.py', 'scripts/joint_motion_metrics.py',
               'third_party/facediffuser/provenance.json', 'third_party/facediffuser/models.py',
               'third_party/facediffuser/utils.py', 'third_party/facediffuser/LICENSE']
    protocol = {'schema': 'facediffuser_arkit_matched_audio_train_v2', 'source_checkpoint_sha256': sha(args.source),
                'data': data['provenance'], 'model_config': model.export_config(), 'seed': args.seed,
                'epochs': 1 if args.smoke else args.epochs, 'requested_epochs': args.epochs,
                'batch_size': args.batch_size, 'smoke': args.smoke, 'device_type': torch.device(args.device).type,
                'train_clips': len(data['splits']['train']['valid']), 'validation_clips': len(data['splits']['validation']['valid']),
                'optimizer': {'class': 'Adam', 'lr': .0001, 'weight_decay': 0, 'gradient_clipping': False},
                'diffusion': {'schedule': 'cosine', 'mean_type': 'START_X', 'variance': 'FIXED_SMALL',
                              'clip_denoised': False, 'steps': 1000, 'sample_timestep_skipping': False},
                'objective': 'x0 observed MSE, equal clips', 'condition_layout': CONDITION_LAYOUT,
                'condition_source': 'frozen source global64 + neutral-reference identity128 + anchor52 + audio intensity1',
                'audio_input': 'native25fps raw HuBERT content768 + TRAIN-normalized audio_features1540 (HuBERT768 + emotion2vec768 + prosody4); cached frozen encoders',
                'audio_input_layout': {'raw_content': content_dim, 'normalized_audio_features': feature_stats['mean'].numel(),
                                       'total': total_input_dim, 'content_in_both_representations': True},
                'feature_stats_source': 'same complete TRAIN query fit as KineTalk; no development fit',
                'feature_stats_sha256': state_hash(feature_stats),
                'decoder_initialization': 'random; no source decoder weights',
                'no_mouth_or_baseline_bypass': True, 'official_commit': SOURCE_COMMIT,
                'official_beat_reproduction': False, 'selection': 'fixed final epoch',
                'noise_seeds': list(SEEDS), 'ancestral_seed_offset': 1000003,
                'sampling_batch_policy': 'fixed order, native batch trim, all1000 DDPM steps',
                'sources': {name: sha(root / name) for name in sources},
                'test_loaded': False, 'default_replaced': False}
    digest = canonical_hash(protocol)
    if args.resume:
        if json.loads((args.output / 'protocol.json').read_text()) != protocol:
            raise ValueError('FaceDiffuser resume protocol mismatch')
    else:
        save_json(args.output / 'protocol.json', protocol)
        for name in sources:
            destination = args.output / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((root / name).read_bytes())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    first, elapsed_before = 0, 0.
    if args.resume:
        checkpoint = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint.get('protocol_sha256') != digest:
            raise ValueError('FaceDiffuser last checkpoint protocol differs')
        if state_hash(checkpoint.get('feature_stats', {})) != protocol['feature_stats_sha256']:
            raise ValueError('FaceDiffuser checkpoint audio statistics differ')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        first, elapsed_before = checkpoint['epoch'], checkpoint['elapsed_seconds']
        if not 0 <= first <= protocol['epochs']:
            raise ValueError('Invalid FaceDiffuser resume epoch')
        restore_rng(checkpoint['rng'], generator)
    train = data['splits']['train']
    for epoch in range(first, protocol['epochs']):
        tick = time.monotonic()
        model.train()
        losses, counts, norms = [], [], []
        for ids in torch.randperm(len(train['valid']), generator=generator).split(args.batch_size):
            batch = trim(train, ids, args.device)
            loss, norm = training_step(model, optimizer, batch, generator, feature_stats)
            losses.append(loss)
            counts.append(len(ids))
            norms.append(norm)
        record = {'status': 'training', 'epoch': epoch + 1, 'epochs': protocol['epochs'],
                  'clips': len(train['valid']), 'loss': float(np.average(losses, weights=counts)),
                  'mean_gradient_norm': float(np.mean(norms)), 'seconds': time.monotonic() - tick,
                  'elapsed_seconds': elapsed_before + time.monotonic() - started,
                  'test_loaded': False}
        save_json(args.output / f'epoch{epoch + 1:03d}.json', record)
        save_checkpoint(args.output / 'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1, 'rng': capture_rng(generator), 'elapsed_seconds': record['elapsed_seconds'],
                        'protocol_sha256': digest, 'feature_stats': feature_stats})
        status(args.output, record)
    final = {'model': model.state_dict(), 'config': model.export_config(), 'protocol': protocol,
             'protocol_sha256': digest, 'feature_stats': feature_stats, 'test_loaded': False}
    save_checkpoint(args.output / 'final.pt', final)
    report, curves = evaluate(model, data, system, identities, args.device, args.output,
                              batch_size=args.batch_size, smoke=args.smoke)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    lengths = {row['clip_id']: row['frames'] for row in manifest['roles']['val']['query']}
    save_checkpoint(args.output / 'native_curves.pt', compact_curves(curves, lengths))
    benchmark = report_generation(curves, data, args.output, 'facediffuser',
                                   SimpleNamespace(condition_mode='matched_all_audio_frozen_clip', artifact_dir=args.output / 'scores'))
    # Common scorer's default scope is overwritten explicitly for this variant.
    arkit_path = args.output / 'arkit_full.json'
    arkit = json.loads(arkit_path.read_text())
    arkit.update(scope=report['scope'], adaptation=protocol['audio_input'],
                 clip_condition_layout=CONDITION_LAYOUT, official_beat_reproduction=False,
                 smoke=args.smoke, model='FaceDiffuser-ARKit-cached+matched-all-audio-frozen-clip',
                 audio_input_layout=protocol['audio_input_layout'], feature_stats_sha256=protocol['feature_stats_sha256'])
    save_json(arkit_path, arkit)
    report.update(benchmark=benchmark, protocol_sha256=digest, smoke=args.smoke,
                  training_epochs=protocol['epochs'], formal_comparison_ready=not args.smoke)
    if state_hash(system.state_dict()) != frozen['system'] or state_hash(audio.state_dict()) != frozen['audio']:
        raise RuntimeError('Frozen condition source changed')
    save_json(args.output / 'evaluation.json', report)
    status(args.output, {'status': 'complete', 'training_epochs': protocol['epochs'],
                         'evaluation_clips': len(data['splits']['validation']['valid']),
                         'smoke': args.smoke, 'quality_pass_claimed': False,
                         'elapsed_seconds': elapsed_before + time.monotonic() - started,
                         'test_loaded': False, 'default_replaced': False})


if __name__ == '__main__':
    main()
