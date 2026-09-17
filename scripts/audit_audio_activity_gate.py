"""Read-only paired intervention on an existing frozen projection adapter.

The activity gate is one minus the frozen audio classifier's neutral
probability. Its constant control is the unweighted training-clip mean of
that probability, fitted without labels. No temperature/threshold is tuned.
Noise repetitions are averaged as errors, never as generated predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_ray import NUISANCE_CHANNELS_52
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import observed
from scripts.train_predictable_renderer import (
    PredictableAudioHead, audio_features, batch_to_device, center,
    projected_affect, reverse_controls, sha, validate_inputs,
)


STAT_COLUMNS = ['sse', 'target_energy', 'prediction_energy', 'weighted_values',
                'centered_covariance', 'centered_prediction_energy', 'centered_target_energy']
KINDS = ('raw_motion', 'centered_residual', 'centered_motion')
MODES = ('full', 'gated', 'constant', 'zero', 'reverse_gated', 'oracle_gated')
SEEDS = (42, 123, 2026)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def activity_gate(logits, emotion_classes):
    """Inference input contains audio logits only; neutral order is explicit."""
    if len(set(emotion_classes)) != len(emotion_classes) or emotion_classes.count('neutral') != 1:
        raise ValueError('Require one uniquely named neutral class')
    if logits.ndim != 2 or logits.shape[1] != len(emotion_classes) or not torch.isfinite(logits).all():
        raise ValueError('Finite audio logits must match configured emotion classes')
    return (1 - logits.detach().float().softmax(-1)[:, emotion_classes.index('neutral')]).clamp(0, 1)


def gate_diagnostics(split, classes):
    logits = split['affect']['emotion_logits'].float()
    gate = activity_gate(logits, classes)
    labels = split['q']['emotion_id'].long()
    prediction = logits.argmax(-1)
    neutral = classes.index('neutral')
    groups = {'all': torch.ones_like(labels, dtype=torch.bool),
              'neutral': labels == neutral, 'nonneutral': labels != neutral}
    groups.update({name: labels == i for i, name in enumerate(classes)})
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        values = gate[mask]
        result[name] = {'clips': int(mask.sum()), 'mean': float(values.mean()),
            'quantiles_05_25_50_75_95': torch.quantile(values, torch.tensor([.05, .25, .5, .75, .95])).tolist(),
            'audio_classification_accuracy': float((prediction[mask] == labels[mask]).float().mean()),
            'predicted_neutral_fraction': float((prediction[mask] == neutral).float().mean())}
    return result


def aggregate_sentence_statistics(statistics, sentences, populations):
    result = {}
    for population, indices in populations.items():
        rows = {}
        for i in indices:
            row = rows.setdefault(str(sentences[i]), np.zeros(len(STAT_COLUMNS), dtype=np.float64))
            row += statistics[i]
        result[population] = {sentence: row.tolist() for sentence, row in rows.items()}
    return result


def intervention_controls(mode, audio, teacher, weight, gate, constant):
    if mode == 'full':
        return audio
    if mode == 'gated':
        return audio * gate[:, None, None]
    if mode == 'constant':
        return audio * constant
    if mode == 'zero':
        return torch.zeros_like(audio)
    if mode == 'reverse_gated':
        return reverse_controls(audio, weight) * gate[:, None, None]
    if mode == 'oracle_gated':
        return teacher * gate[:, None, None]
    raise ValueError(f'Unknown intervention: {mode}')


def validate_adapter(adapter, hashes, head):
    if adapter.get('schema') != 'predictable_projection_adapter_v1' or adapter.get('mode') != 'rrr':
        raise ValueError('Require fixed RRR projection-only adapter')
    provenance = adapter['provenance']
    if not provenance.get('args', {}).get('projection_only'):
        raise ValueError('Adapter must have trained only the shared projection')
    if provenance.get('args', {}).get('audio_activity_gate', False):
        raise ValueError('This intervention requires the original ungated adapter')
    for key in ('checkpoint', 'config', 'cache', 'bundle', 'weights'):
        if provenance['hashes'].get(key) != hashes[key]:
            raise ValueError(f'Adapter {key} hash mismatch')
    expected = state_hash(head.state_dict())
    head.load_state_dict(adapter['head'], strict=True)
    if state_hash(head.state_dict()) != expected:
        raise ValueError('Adapter audio head/basis/scales differ from the fixed train-only ridge initialization')


def summarize_teacher_logits(logits, labels):
    """Same frozen training-teacher class readout as the renderer experiment."""
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()
    if logits.ndim != 2 or logits.shape[0] != len(labels) or not torch.isfinite(logits).all():
        raise ValueError('Invalid frozen teacher logits')
    predicted = logits.argmax(-1)
    return {'accuracy': float((predicted == labels).float().mean()),
            'mean_class_probability': logits.softmax(-1).mean(0).tolist(),
            'predicted_class_counts': torch.bincount(predicted, minlength=logits.shape[-1]).tolist(),
            'clips': len(labels),
            'note': 'Same frozen training motion teacher; not an independent emotion-quality evaluator'}


@torch.no_grad()
def readout_saved_prediction(system, split, prediction, *, device, batch_size=32):
    if prediction.shape != split['q']['motion'].shape:
        raise ValueError('Saved prediction shape differs from reference motion')
    logits = []
    for start in range(0, len(prediction), batch_size):
        ids = torch.arange(start, min(start + batch_size, len(prediction)))
        batch = batch_to_device(split, ids, device)
        residual = torch.where(observed(batch['q']), prediction[ids].to(device) -
            batch['base']['b0'] - batch['identity']['baseline'][:, None], 0)
        logits.append(system.motion_teacher(residual, batch['q']['valid'])['emotion_logits'].cpu())
    return summarize_teacher_logits(torch.cat(logits), split['q']['emotion_id'])


def velocity_clip_statistics(prediction, split, channels):
    """Per-clip velocity error sufficient statistics on observed adjacencies."""
    q = split['q']
    if prediction.shape != q['motion'].shape:
        raise ValueError('Prediction/reference shape mismatch')
    valid = q['valid'].bool()
    adjacent = valid[:, 1:] & valid[:, :-1]
    dt = q['times'][:, 1:] - q['times'][:, :-1]
    if not (dt[adjacent] > 0).all():
        raise ValueError('Positive native time intervals required')
    error = prediction.double()[..., channels] - q['motion'].double()[..., channels]
    velocity = torch.diff(error, dim=1) / dt.double().clamp_min(1e-9)[..., None]
    velocity = torch.where(adjacent[..., None], velocity, 0)
    return torch.stack([velocity.square().sum((1, 2)), adjacent.sum(1) * len(channels)], -1).numpy()


@torch.no_grad()
def evaluate(system, head, split, bundle, classes, constant, *, device, batch_size,
             with_original=True, curves_dir=None, teacher_readout=None):
    q = split['q']
    features, weight = audio_features(bundle), bundle['weight'].float()
    target, frame_weight = q['motion'].float(), q['valid'].float()
    common = q['channel_mask'].all(0)
    if not torch.equal(q['channel_mask'], common[None].expand_as(q['channel_mask'])):
        raise ValueError('Require audited common observed channels')
    groups = {'all_expression': [i for i in range(len(common)) if common[i] and i not in NUISANCE_CHANNELS_52],
              'upper_expression': [5, 6, 12, 13, 41, 42, 43, 44, 45], 'brows': list(range(41, 46)),
              'eyes_expression': [5, 6, 12, 13], 'mouth': list(range(14, 41)), 'jaw17': [17]}
    groups = {name: [i for i in channels if i < len(common) and common[i]] for name, channels in groups.items()}
    groups = {name: channels for name, channels in groups.items() if channels}
    baseline = split['base']['b0'].float() + split['identity']['baseline'].float()[:, None]
    targets = {'raw_motion': target, 'centered_residual': center(target - baseline, frame_weight),
               'centered_motion': center(target, frame_weight)}
    modes = MODES + (('original',) if with_original else ())
    statistics, per_seed, condition_rms = {}, {}, {}
    for seed in SEEDS:
        noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed))
        curves = {mode: [] for mode in modes}
        for start in range(0, len(features), batch_size):
            ids = torch.arange(start, min(start + batch_size, len(features)))
            batch = batch_to_device(split, ids, device)
            w = weight[ids].to(device)
            audio = head(features[ids].to(device), w)
            teacher = head.teacher(bundle['motion_bins'][ids].float().to(device), w)
            gate = activity_gate(batch['affect']['emotion_logits'], classes)
            for mode in modes:
                if mode == 'original':
                    affect = batch['affect']
                else:
                    controls = intervention_controls(mode, audio, teacher, w, gate, constant)
                    affect = projected_affect(system, batch, controls, w, zero=mode == 'zero')
                prediction = system.generate(batch['q']['content'], batch['q']['valid'], batch['identity'], affect,
                    initial_noise=noise[ids].to(device), steps=12, base=batch['base'])['motion']
                curves[mode].append(prediction.cpu())
                if seed == SEEDS[0]:
                    local = affect['local'].float()
                    mask = batch['q']['valid']
                    squared = (torch.where(mask[..., None], local, 0).square().sum((1, 2)) /
                               (mask.sum(1) * local.shape[-1]).clamp_min(1)).cpu()
                    condition_rms.setdefault(mode, []).append(squared)
        curves = {mode: torch.cat(parts) for mode, parts in curves.items()}
        if teacher_readout is not None:
            teacher_readout[str(seed)] = {mode: readout_saved_prediction(system, split, prediction,
                device=device, batch_size=batch_size) for mode, prediction in curves.items()}
        per_seed[str(seed)] = {}
        for mode, prediction in curves.items():
            values = {'raw_motion': prediction,
                      'centered_residual': center(prediction - baseline, frame_weight),
                      'centered_motion': center(prediction, frame_weight)}
            per_seed[str(seed)][mode] = {}
            for kind, pred in values.items():
                per_seed[str(seed)][mode][kind] = {}
                for group, channels in groups.items():
                    stats = clip_statistics(pred, targets[kind], frame_weight, channels)
                    key = (mode, kind, group)
                    statistics[key] = statistics.get(key, np.zeros_like(stats)) + stats / len(SEEDS)
                    per_seed[str(seed)][mode][kind][group] = stats
        if curves_dir is not None:
            torch.save({'motion': curves, 'noise_seed': seed, 'decode_steps': 12}, curves_dir / f'gate_seed{seed}_curves.pt')
        print(json.dumps({'stage': 'evaluation', 'seed': seed, 'modes': modes, 'clips': len(features)}), flush=True)
    condition_rms = {mode: torch.cat(parts).sqrt().numpy() for mode, parts in condition_rms.items()}
    return statistics, per_seed, condition_rms, groups, modes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'adapter', 'cache', 'bundle', 'weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--skip-original', action='store_true')
    parser.add_argument('--curves-dir', type=Path, help='Optional fresh directory, e.g. /dev/shm; no large curves saved by default')
    args = parser.parse_args()
    if args.batch_size < 1 or args.samples < 0:
        raise ValueError('Invalid batch size/bootstrap count')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.curves_dir is not None:
        args.curves_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = yaml.safe_load(args.config.read_text(encoding='utf8'))
    classes = config['data']['emotion_classes']
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=False)
    fitted = torch.load(args.weights, map_location='cpu', weights_only=False)
    adapter = torch.load(args.adapter, map_location='cpu', weights_only=False)
    validate_inputs(cache, bundle, fitted['states'], checkpoint, config)
    hashes = {name: sha(getattr(args, name)) for name in ('config', 'checkpoint', 'adapter', 'cache', 'bundle', 'weights')}
    for name in ('checkpoint', 'config', 'bundle'):
        if cache['provenance'].get(name + '_sha256') != hashes[name]:
            raise ValueError(f'Cache {name} provenance mismatch')
    if fitted['provenance'].get('bundle_sha256') != hashes['bundle']:
        raise ValueError('Basis weights came from a different motion bundle')
    train = bundle['bundles']['internal']
    head = PredictableAudioHead(fitted['states']['rrr_rank8'], train['motion_bins'], train['weight'])
    validate_adapter(adapter, hashes, head)
    head = head.to(args.device).eval()
    system = NeutralAffectSystem(config).to(args.device).eval()
    system.load_state_dict(checkpoint['model'], strict=True)
    system.local_projection.load_state_dict(adapter['local_projection'], strict=True)
    freeze_module(system); freeze_module(head)
    before = {'system': state_hash(system.state_dict()), 'head': state_hash(head.state_dict())}
    constant = float(activity_gate(cache['splits']['train']['affect']['emotion_logits'], classes).mean())
    validation = cache['splits']['validation']; q = validation['q']
    if args.curves_dir is not None:
        torch.save({'q': {k: v for k, v in q.items() if k != 'content'}, 'base': {'b0': validation['base']['b0']},
                    'identity': validation['identity']}, args.curves_dir / 'validation_reference.pt')
    teacher_readout = {}
    stats, seed_stats, local_rms, groups, modes = evaluate(system, head, validation,
        bundle['bundles']['external_dev'], classes, constant, device=args.device, batch_size=args.batch_size,
        with_original=not args.skip_original, curves_dir=args.curves_dir, teacher_readout=teacher_readout)
    after = {'system': state_hash(system.state_dict()), 'head': state_hash(head.state_dict())}
    if before != after or any(p.grad is not None for module in (system, head) for p in module.parameters()):
        raise RuntimeError('Frozen audit mutated weights or accumulated gradients')
    neutral = classes.index('neutral')
    populations = {'all': list(range(len(q['emotion_id']))),
        'nonneutral': (q['emotion_id'] != neutral).nonzero(as_tuple=True)[0].tolist(),
        'neutral': (q['emotion_id'] == neutral).nonzero(as_tuple=True)[0].tolist()}
    populations.update({'emotion_' + name: (q['emotion_id'] == i).nonzero(as_tuple=True)[0].tolist()
                        for i, name in enumerate(classes) if (q['emotion_id'] == i).any()})
    populations = {name: ids for name, ids in populations.items() if ids}
    scores = {mode: {kind: {group: {pop: scalar_summary(stats[(mode, kind, group)][ids])
        for pop, ids in populations.items()} for group in groups} for kind in KINDS} for mode in modes}
    pairs = [('gated', mode) for mode in ('full', 'constant', 'zero', 'reverse_gated')]
    pairs += [('constant', 'full'), ('oracle_gated', 'zero')]
    if 'original' in modes:
        pairs += [('gated', 'original')]
    comparisons = {left + '__vs__' + right: {pop: {kind: {group: paired_summary(
        stats[(left, kind, group)], stats[(right, kind, group)], q['sentence_id'], ids,
        samples=args.samples if pop in ('neutral', 'nonneutral') else 0, seed=45)
        for group in groups} for kind in KINDS} for pop, ids in populations.items()} for left, right in pairs}
    per_seed = {seed: {mode: {kind: {group: {pop: scalar_summary(values[ids]) for pop, ids in populations.items()}
        for group, values in group_values.items()} for kind, group_values in kind_values.items()}
        for mode, kind_values in mode_values.items()} for seed, mode_values in seed_stats.items()}
    sentence_stats = {'columns': STAT_COLUMNS, 'noise_aggregation': 'mean clip sufficient statistics over all three noise seeds',
        'settings': {mode: {kind: {group: aggregate_sentence_statistics(stats[(mode, kind, group)], q['sentence_id'],
            {pop: populations[pop] for pop in ('all', 'neutral', 'nonneutral') if pop in populations})
            for group in groups} for kind in KINDS} for mode in modes}}
    write_json(args.output / 'per_sentence_statistics.json', sentence_stats)
    np.savez_compressed(args.output / 'clip_statistics.npz', **{'__'.join(key): value for key, value in stats.items()})
    write_json(args.output / 'clip_metadata.json', {'clip_id': q['clip_id'], 'sentence_id': q['sentence_id'],
        'emotion_id': q['emotion_id'].tolist(), 'speaker_id': q['speaker_id'].tolist(), 'statistic_columns': STAT_COLUMNS})
    provenance = {'schema': 'audio_activity_gate_audit_v1', 'hashes': hashes,
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'seeds': SEEDS, 'decode_steps': 12, 'constant_train_clip_mean': constant,
        'gate': '1 - frozen audio softmax probability of configured neutral class; no temperature or threshold',
        'selection': 'Read-only fixed run33 adapter; no fit or development tuning; constant mean uses only training audio logits',
        'label_usage': 'Metadata emotion labels used exclusively for evaluation groups and diagnostics, never conditions',
        'oracle': 'Real motion controls with exactly the same audio-derived gate, diagnostic only',
        'new_test_loaded': False, 'frozen_before': before, 'frozen_after': after,
        'frozen_unchanged': True, 'noise_aggregation': 'Average clip error sufficient statistics before sentence-cluster bootstrap; seeds are not extra samples',
        'scope': 'Repeatedly inspected development; pretrained/global exposure possible; no independent perceptual or audiovisual lip-sync evaluator',
        'source_sha256': {str(path): sha(path) for path in (Path(__file__), Path(__file__).with_name('train_predictable_renderer.py'),
            Path(__file__).with_name('audit_predictable_motion_predictions.py'))}}
    write_json(args.output / 'provenance.json', provenance)
    summary = {'provenance': provenance, 'clips': len(q['emotion_id']), 'sentences': len(set(q['sentence_id'])),
        'groups': groups, 'scores': scores, 'comparisons': comparisons, 'per_seed': per_seed,
        'gate_statistics': {name: gate_diagnostics(split, classes) for name, split in cache['splits'].items()},
        'projection_weight_norm': float(system.local_projection.weight.norm()),
        'frozen_teacher_readout': teacher_readout,
        'frozen_teacher_accuracy': {seed: {mode: values['accuracy'] for mode, values in rows.items()}
                                   for seed, rows in teacher_readout.items()},
        'local_rms': {mode: {pop: float(np.sqrt(np.square(values[ids]).mean())) for pop, ids in populations.items()}
                      for mode, values in local_rms.items()}}
    write_json(args.output / 'summary.json', summary)
    (args.output / 'source').mkdir()
    for path in (Path(__file__), Path(__file__).with_name('train_predictable_renderer.py'),
                 Path(__file__).with_name('audit_predictable_motion_predictions.py')):
        shutil.copyfile(path, args.output / 'source' / path.name)
    for mode in modes:
        print(json.dumps({'mode': mode, 'nonneutral_upper_r2': scores[mode]['centered_residual']['upper_expression']['nonneutral']['r2_against_zero'],
            'neutral_mouth_mse': scores[mode]['raw_motion']['mouth']['neutral']['native_mse']}), flush=True)
    print('COMPLETE: frozen audio activity gate intervention; no training or default changes', flush=True)


if __name__ == '__main__':
    main()
