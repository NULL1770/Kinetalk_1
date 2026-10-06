"""Frozen train/validation diagnostic: pure-noise rollout vs GT-assisted endpoints.

No generator fitting, checkpoint writes, or sealed-test reads. Optional
condition statistics/calibration use real TRAIN only. A flow endpoint with t>0
contains target motion in its input and must never be reported as inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.packed_trainval_cache import load_packed
from scripts.phase1_condition_diagnostic import _identity_cache
from scripts.train_full_staged import (
    audio_affect, base_forward, batch_identity, merge_teacher_audio, teacher_affect,
)
from scripts.arkit_benchmark_report import score_fullface
from kinetalk_b0.emotion_probe import (
    MotionEmotionProbe, classification_metrics, motion_features,
)
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clip_metrics(pred, target, valid, channel, times):
    observed = valid[:, None] & channel[None]
    if not torch.isfinite(pred[observed]).all():
        raise ValueError('Nonfinite observed generation')
    scored = score_fullface(pred[None].numpy(), {
        'target52': target.numpy(), 'valid': valid.numpy(),
        'times': times.numpy(),
        'channel_mask': np.broadcast_to(channel.numpy(), target.shape),
    })
    x = pred[valid][:, 14:41][:, channel[14:41]]
    adjacent = valid[1:] & valid[:-1]
    velocity_error = (pred[1:] - pred[:-1]) - (target[1:] - target[:-1])
    ve = velocity_error[adjacent][:, 14:41][:, channel[14:41]]
    return [scored['metrics']['arkit_mbe']['value'],
            scored['metrics']['arkit_lbe']['value'],
            float(((pred[observed] < 0) | (pred[observed] > 1)).float().mean()),
            float(x.square().mean().sqrt()),
            float((x - x.mean(0)).square().mean().sqrt()),
            float(ve.square().mean())]


@torch.no_grad()
def fit_global_prototypes(system, data, identities, device, batch_size):
    """Frozen teacher codes averaged by REAL TRAIN emotion/intensity only."""
    train = data['splits']['train']
    codes = []
    for ix in torch.arange(len(train['_lengths'])).split(batch_size):
        b = train.batch(ix, device)
        b.update(base_forward(system, b['content'], b['valid']))
        ident = batch_identity(identities, b)
        codes.append(teacher_affect(system, b, ident)['global'].cpu())
    codes = torch.cat(codes)
    labels, intensity = train['emotion_id'], train['intensity_id']
    classes = len(data['config']['data']['emotion_classes'])
    levels = int(data['config']['data']['num_intensity_levels'])
    class_mean = torch.stack([codes[labels == k].mean(0) for k in range(classes)])
    joint_mean = class_mean[:, None].expand(-1, levels, -1).clone()
    counts = torch.zeros(classes, levels, dtype=torch.long)
    for k in range(classes):
        for l in range(levels):
            selected = (labels == k) & (intensity == l) & train['intensity_valid']
            counts[k, l] = selected.sum()
            if selected.any():
                joint_mean[k, l] = codes[selected].mean(0)
    total_sse = (codes - codes.mean(0)).square().sum().clamp_min(1e-12)
    within_sse = (codes - class_mean[labels]).square().sum()
    return class_mean.to(device), joint_mean.to(device), {
        'train_clips': len(codes), 'source': 'frozen teacher on real TRAIN motion only',
        'class_intensity_counts': counts.tolist(),
        'class_explained_code_variance': float(1 - within_sse / total_sse),
        'missing_joint_policy': 'fall back to same-emotion train class mean',
        'selection_policy': 'audio argmax or explicitly labeled validation-label oracle',
        'changed_condition': 'global only; intensity scalar, temporal audio and identity unchanged',
    }


@torch.no_grad()
def fit_global_calibration(system, audio, data, identities, device, batch_size):
    """Predeclared ridge map: frozen audio globals -> frozen teacher globals.

    Real TRAIN data only, lambda=.001 after train-only input standardization.
    No validation/generated values choose the strength or fitted coefficients.
    """
    train = data['splits']['train']
    xs, ys = [], []
    for ix in torch.arange(len(train['_lengths'])).split(batch_size):
        b = train.batch(ix, device)
        b.update(base_forward(system, b['content'], b['valid']))
        ident = batch_identity(identities, b)
        xs.append(audio_affect(audio, b['audio_features'], b['valid'])['global'].cpu())
        ys.append(teacher_affect(system, b, ident)['global'].cpu())
    x, y = torch.cat(xs).double(), torch.cat(ys).double()
    xm, ym = x.mean(0), y.mean(0)
    scale = x.std(0, correction=0).clamp_min(1e-4)
    z = (x - xm) / scale
    ridge = .001
    matrix = torch.linalg.solve(z.T @ z / len(z) + ridge * torch.eye(z.shape[1]),
                                z.T @ (y - ym) / len(z))
    pred = z @ matrix + ym
    meta = {'source': 'real TRAIN frozen audio/teacher codes only', 'train_clips': len(x),
            'ridge_lambda': ridge, 'normalization': 'train input mean/std, floor1e-4',
            'train_original_mse': float((x - y).square().mean()),
            'train_calibrated_mse': float((pred - y).square().mean()),
            'train_audio_mean_square': float(x.square().mean()),
            'train_teacher_mean_square': float(y.square().mean()),
            'selection': 'fixed lambda=.001; no validation or generated selection',
            'changed_condition': 'global only; all other conditions and parameters frozen'}
    return tuple(v.float().to(device) for v in (xm, scale, matrix, ym)), meta


@torch.no_grad()
def decode_with_sampler(system, content, mask, identity, affect, base, initial_noise,
                        steps, sampler):
    """Train-free ODE sampler comparison; default model path remains Euler."""
    if sampler not in {'euler', 'midpoint', 'heun'}:
        raise ValueError(f'Unknown sampler: {sampler}')
    if steps < 1:
        raise ValueError('Positive sampler steps required')
    expected = (content.shape[0], content.shape[1], system.renderer.output.out_features)
    if tuple(initial_noise.shape) != expected:
        raise ValueError(f'initial_noise must have shape {expected}')
    support = mask[..., None] & system.residual_support[None, None]
    state = system.source_noise(torch.where(support, initial_noise.to(content), 0.), mask)
    times = torch.linspace(0., 1., steps + 1, device=content.device, dtype=content.dtype)
    temporal = affect.get('u_a', affect.get('temporal', affect.get('local')))

    def velocity(x, t):
        v = system.renderer(x, t, base['h0'], affect['global'], affect['intensity_value'],
                            identity['code'], mask, temporal_condition=temporal,
                            condition_dropout=False)
        return torch.where(support, v, 0.)

    for index in range(steps):
        h = times[index + 1] - times[index]
        t0 = torch.full((content.shape[0],), times[index], device=content.device,
                        dtype=content.dtype)
        k1 = velocity(state, t0)
        if sampler == 'euler':
            state = state + h * k1
        elif sampler == 'midpoint':
            tm = torch.full((content.shape[0],), times[index] + h / 2,
                            device=content.device, dtype=content.dtype)
            state = state + h * velocity(state + h / 2 * k1, tm)
        else:
            t1 = torch.full((content.shape[0],), times[index + 1], device=content.device,
                            dtype=content.dtype)
            k2 = velocity(state + h * k1, t1)
            state = state + h / 2 * (k1 + k2)
        state = torch.where(support, state, 0.)
    return state * system.residual_scale


@torch.no_grad()
def run(args):
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic directory required')
    actual_sha = sha(args.checkpoint)
    if actual_sha != args.checkpoint_sha256:
        raise ValueError('Checkpoint checksum mismatch')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    provenance = json.loads((args.run_root / 'provenance.json').read_text(encoding='utf8'))
    recipe = provenance['recipe']
    if ck['stage'] != 'audio' or ck['recipe_sha256'] != provenance['recipe_sha256']:
        raise ValueError('Expected provenance-bound audio checkpoint')
    data = load_packed(args.data, materialize=False, with_refs=True)
    manifest = data['provenance']['manifest_sha256']
    if manifest != ck['data_manifest_sha256'] or data['provenance'].get('test_loaded'):
        raise ValueError('Manifest mismatch or test data present')
    if set(data['splits']) != {'train', 'validation'}:
        raise ValueError('Only train and validation splits allowed')
    system = NeutralAffectSystem(ck['config']).to(device).eval()
    system.load_state_dict(ck['system'], strict=True)
    audio = SlowStateAffect(ck['feature_stats']['mean'], ck['feature_stats']['std'],
                           stride=recipe['args']['stride']).to(device).eval()
    audio.load_state_dict(ck['audio'], strict=True)
    identities = _identity_cache(system, data, device)
    prototype_meta = None
    if args.global_prototypes:
        class_mean, joint_mean, prototype_meta = fit_global_prototypes(
            system, data, identities, device, args.batch_size)
        print(json.dumps({'event': 'train_prototypes', **prototype_meta}), flush=True)
    calibration_meta = None
    if args.global_calibration:
        calibration, calibration_meta = fit_global_calibration(
            system, audio, data, identities, device, args.batch_size)
        print(json.dumps({'event': 'train_global_calibration', **calibration_meta}), flush=True)
    probes, probe_meta = [], []
    support = None
    names = None
    for path in args.probe:
        pc = torch.load(path, map_location='cpu', weights_only=False)
        if (pc['train_manifest_sha256'] != manifest or pc.get('test_used_for_selection')
                or pc.get('generator_outputs_used_for_fitting')):
            raise ValueError('Probe must be fitted to real train motion only')
        ps = pc['channel_support'].bool()
        if support is not None and (not torch.equal(support, ps) or names != pc['classes']):
            raise ValueError('Probes must share feature support and class order')
        support, names = ps, pc['classes']
        p = MotionEmotionProbe(pc['feature_dim'], pc['hidden'], len(pc['classes'])).eval()
        p.load_state_dict(pc['model'], strict=True)
        probes.append(p)
        probe_meta.append({'path': str(path), 'sha256': sha(path),
                           'used_in_this_checkpoint_training': bool(
                               recipe['args'].get('independent_probe_weight', 0) > 0)})
    q = data['splits']['validation']
    if not q['channel_mask'][:, support].all():
        raise ValueError('Missing probe support in validation')
    ids = torch.arange(len(q['_lengths']))
    if args.limit_per_class:
        selected = []
        for c in range(len(names)):
            eligible = (q['emotion_id'] == c).nonzero().flatten()
            # Spread across the ordered list, including different speakers/levels.
            indices = torch.linspace(0, len(eligible) - 1,
                                     min(args.limit_per_class, len(eligible))).long()
            selected.extend(eligible[indices].tolist())
        ids = torch.tensor(sorted(selected))
    metrics, features, teacher_predictions, labels, clip_ids = {}, {}, {}, [], []
    noise_gen = torch.Generator().manual_seed(args.noise_seed)
    cases = {'GT': {'contains_target': True, 'type': 'reference'}}
    code_errors = {'original_squared_sum': 0., 'calibrated_squared_sum': 0.,
                   'audio_squared_sum': 0., 'teacher_squared_sum': 0., 'values': 0}

    def collect(mode, pred, b, ident, ix):
        metrics.setdefault(mode, {'raw': [], 'clip_all': []})
        features.setdefault(mode, {'raw': [], 'clip_all': []})
        residual = pred - b['b0'] - ident['baseline'][:, None]
        observed = b['valid'][..., None] & b['channel_mask'][:, None]
        encoded = system.encode_motion(torch.where(observed, residual, 0.), b['valid'])
        teacher_predictions.setdefault(mode, []).extend(encoded['emotion_logits'].argmax(-1).cpu().tolist())
        pred = pred.cpu()
        for j, i in enumerate(ix.tolist()):
            length = int(q['_lengths'][i])
            target = b['motion'][j, :length].cpu()
            valid = b['valid'][j, :length].cpu()
            channel = b['channel_mask'][j].cpu()
            times = b['times'][j, :length].cpu()
            for variant, x in [('raw', pred[j, :length]),
                               ('clip_all', pred[j, :length].clamp(0., 1.))]:
                metrics[mode][variant].append(clip_metrics(x, target, valid, channel, times))
                features[mode][variant].append(motion_features(x[:, support], valid).numpy())

    for batch_no, ix in enumerate(ids.split(args.batch_size)):
        b = q.batch(ix, device)
        ident = batch_identity(identities, b)
        base = base_forward(system, b['content'], b['valid'])
        b.update(base)
        ao = audio_affect(audio, b['audio_features'], b['valid'])
        oracle = merge_teacher_audio(teacher_affect(system, b, ident), ao)
        noise = torch.randn(b['motion'].shape, generator=noise_gen).to(device)
        labels.extend(b['emotion_id'].cpu().tolist())
        clip_ids.extend(b['clip_id'])
        collect('GT', b['motion'], b, ident, ix)
        conditions = [('audio', ao), ('oracle', oracle)]
        if args.global_calibration:
            xm, xs, matrix, ym = calibration
            mapped = (ao['global'] - xm) / xs @ matrix + ym
            code_errors['original_squared_sum'] += float((ao['global'] - oracle['global']).square().sum())
            code_errors['calibrated_squared_sum'] += float((mapped - oracle['global']).square().sum())
            code_errors['audio_squared_sum'] += float(ao['global'].square().sum())
            code_errors['teacher_squared_sum'] += float(oracle['global'].square().sum())
            code_errors['values'] += ao['global'].numel()
            conditions.append(('calibrated_audio', {**ao, 'global': mapped}))
        if args.global_prototypes:
            predicted_class = ao['emotion_logits'].argmax(-1)
            predicted_level = ao['intensity_logits'].argmax(-1)
            true_class = b['emotion_id']
            true_level = b['intensity_id'].clamp(0, joint_mean.shape[1] - 1)
            conditions += [
                ('prototype_class_audio', {**ao, 'global': class_mean[predicted_class]}),
                ('prototype_joint_audio', {**ao, 'global': joint_mean[predicted_class, predicted_level]}),
                ('prototype_class_label_oracle', {**ao, 'global': class_mean[true_class]}),
                ('prototype_joint_label_oracle', {**ao, 'global': joint_mean[true_class, true_level]}),
            ]
        for condition, affect in conditions:
            for steps in args.decode_steps:
                for sampler in args.samplers:
                    mode = f'{condition}_{sampler}{steps}'
                    cases[mode] = {'contains_target': condition == 'oracle' or condition.endswith('label_oracle'), 'type': 'rollout',
                                   'noise_scale': 1., 'steps': steps, 'sampler': sampler,
                                   'vector_field_evaluations': steps * (1 if sampler == 'euler' else 2)}
                    if sampler == 'euler':
                        pred_residual = system.generate(b['content'], b['valid'], ident, affect,
                                                        initial_noise=noise, steps=steps,
                                                        base=base)['raw_residual']
                        if args.verify_euler and batch_no == 0:
                            replay = decode_with_sampler(system, b['content'], b['valid'],
                                                         ident, affect, base, noise, steps, 'euler')
                            torch.testing.assert_close(replay, pred_residual, rtol=1e-5, atol=1e-6)
                    else:
                        pred_residual = decode_with_sampler(system, b['content'], b['valid'],
                                                            ident, affect, base, noise, steps, sampler)
                    pred_support = b['valid'][..., None] & system.motion_support[None, None]
                    pred = torch.where(pred_support, base['b0'] + ident['baseline'][:, None] + pred_residual, 0.)
                    collect(mode, pred, b, ident, ix)
                    if args.residual_gains:
                        anchor = base['b0'] + ident['baseline'][:, None]
                        for gain in args.residual_gains:
                            gain_mode = f'{mode}_gain{gain:g}'
                            cases[gain_mode] = {**cases[mode], 'residual_gain': gain,
                                               'note': 'Diagnostic residual scaling; not a selected inference policy.'}
                            gain_pred = torch.where(pred_support, anchor + gain * (pred - anchor), 0.)
                            collect(gain_mode, gain_pred, b, ident, ix)
            for t in ([] if condition.startswith(('prototype_', 'calibrated_')) else args.times):
                mode = f'{condition}_endpoint_t{t:g}'
                cases[mode] = {'contains_target': t > 0 or condition == 'oracle',
                               'type': 'one_step_endpoint', 'time': t}
                out = system.flow(b['motion'], b['content'], b['valid'], ident, affect,
                                  noise=noise, time=torch.full((len(ix),), t, device=device),
                                  base=base, observation_mask=b['channel_mask'])
                collect(mode, out['raw_motion'], b, ident, ix)
        mode = 'audio_zero_noise'
        cases[mode] = {'contains_target': False, 'type': 'rollout',
                       'noise_scale': 0., 'steps': args.decode_steps[0],
                       'note': 'Degenerate source distribution; diagnostic only, not a proposed inference rule.'}
        pred = system.generate(b['content'], b['valid'], ident, ao,
                               initial_noise=torch.zeros_like(noise),
                               steps=args.decode_steps[0], base=base)['raw_motion']
        collect(mode, pred, b, ident, ix)
        if batch_no % 10 == 0:
            print(json.dumps({'event': 'batch', 'batch': batch_no,
                              'clips': len(labels), 'total': len(ids)}), flush=True)

    metric_names = ['MBE', 'LBE', 'outside_fraction', 'mouth_absolute_rms',
                    'mouth_centered_rms', 'mouth_displacement_mse']
    results, arrays = {}, {'labels': np.asarray(labels), 'clip_id': np.asarray(clip_ids)}
    label_tensor = torch.tensor(labels)
    for mode in features:
        results[mode] = {'case': cases[mode], 'variants': {},
                        'teacher_readout_nonindependent': classification_metrics(
                            label_tensor, torch.tensor(teacher_predictions[mode]), names)}
        for variant in ['raw', 'clip_all']:
            f = np.stack(features[mode][variant])
            m = np.asarray(metrics[mode][variant], dtype=np.float64)
            arrays[f'{mode}__{variant}__features'] = f
            arrays[f'{mode}__{variant}__metrics'] = m
            readouts = []
            for probe in probes:
                prediction = probe(torch.from_numpy(f)).argmax(-1)
                readouts.append(classification_metrics(label_tensor, prediction, names))
            results[mode]['variants'][variant] = {
                'metrics': dict(zip(metric_names, m.mean(0).tolist())),
                'probes': readouts,
            }
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'per_clip_features_metrics.npz', **arrays)
    if args.global_calibration:
        np.savez_compressed(args.output / 'train_fitted_global_calibration.npz',
                            **{k: v.cpu().numpy() for k, v in zip(('mean', 'scale', 'matrix', 'target_mean'), calibration)})
        calibration_meta['validation_original_mse'] = code_errors['original_squared_sum'] / code_errors['values']
        calibration_meta['validation_calibrated_mse'] = code_errors['calibrated_squared_sum'] / code_errors['values']
        calibration_meta['validation_audio_mean_square'] = code_errors['audio_squared_sum'] / code_errors['values']
        calibration_meta['validation_teacher_mean_square'] = code_errors['teacher_squared_sum'] / code_errors['values']
    if args.global_prototypes:
        np.savez_compressed(args.output / 'train_fitted_global_prototypes.npz',
                            class_mean=class_mean.cpu().numpy(), joint_mean=joint_mean.cpu().numpy())
    report = {'schema': 'flow_sampling_diagnostic_v1', 'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': actual_sha, 'data_manifest_sha256': manifest,
              'clips': len(ids), 'noise_seed': args.noise_seed, 'batch_size': args.batch_size,
              'classes': names, 'metric_columns': metric_names, 'probes': probe_meta,
              'selection': 'complete validation' if not args.limit_per_class else 'evenly spaced within each class',
              'test_loaded': False, 'training_performed': False,
              'model_output_projection': system.output_projection,
              'global_prototypes': prototype_meta,
              'global_calibration': calibration_meta,
              'variant_policy': 'raw=pre-projection; clip_all=[0,1] final clamp on same outputs',
              'oracle_policy': 'GT/oracle and endpoints t>0 are diagnostics, never deployable scores.',
              'results': results, 'source_sha256': sha(__file__)}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({'event': 'complete', 'output': str(args.output), 'clips': len(ids)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--probe', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--noise-seed', type=int, default=42)
    p.add_argument('--limit-per-class', type=int, default=0)
    p.add_argument('--decode-steps', type=int, nargs='+', default=[12, 48])
    p.add_argument('--samplers', nargs='+', choices=['euler', 'midpoint', 'heun'],
                   default=['euler'], help='ODE samplers for train-free comparison')
    p.add_argument('--verify-euler', action='store_true',
                   help='Verify helper parity with the model Euler implementation on first batch')
    p.add_argument('--residual-gains', type=float, nargs='*', default=[],
                   help='Train-free gains applied to generated residual around B0+identity')
    p.add_argument('--global-prototypes', action='store_true',
                   help='Train-only teacher class/joint prototype global conditions; no parameter fitting')
    p.add_argument('--global-calibration', action='store_true',
                   help='Fixed train-only ridge calibration of frozen audio global codes; diagnostic only')
    p.add_argument('--times', type=float, nargs='+', default=[0., .25, .5, .75, .9])
    args = p.parse_args()
    if min(args.decode_steps) < 1 or any(not 0 <= t < 1 for t in args.times):
        raise ValueError('Positive decode steps and endpoint times in [0,1) required')
    if args.limit_per_class < 0 or args.batch_size < 1 or args.threads < 1:
        raise ValueError('Invalid diagnostic limits')
    if any(not np.isfinite(gain) or gain < 0 for gain in args.residual_gains):
        raise ValueError('Residual gains must be finite and nonnegative')
    run(args)


if __name__ == '__main__':
    main()
