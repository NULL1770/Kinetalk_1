"""Conventional direct acoustic conditioning and closed-loop motion supervision.

This isolated development baseline uses cached low-rate acoustic features; it
is not a reproduction of SubtleTalk's WavLM extractor or full training recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.utils import freeze_module
from scripts.train_audio_conditioned_flow_probe import load_source, NOISE_SEEDS, source_paths
from scripts.train_formal_predictable_projection import canonical_hash, capture_rng, save_checkpoint, save_json
from scripts.train_predictable_renderer import (
    audio_features, audio_activity_gate, batch_to_device, basic_metrics, center,
    observed, optimize, reverse_controls, sha, state_hash,
)
from scripts.train_projection_schedule_ablation import draws, curve_binding

SCHEMA = 'direct_audio_dynamics_v1'
UPPER = [5, 6, 12, 13, 41, 42, 43, 44, 45]
LOSS_WEIGHTS = {'centered_upper': 1.}
MODES = ('full', 'zero', 'reverse')


def feature_statistics(features, weight):
    """Fit only on training frames, with native bin occupancy weights."""
    x, w = features.double(), weight.double()[..., None]
    x = torch.where(w > 0, x, 0)
    mean = (x * w).sum((0, 1)) / w.sum().clamp_min(1)
    variance = (torch.where(w > 0, x - mean, 0).square() * w).sum((0, 1)) / w.sum().clamp_min(1)
    return mean.float(), variance.sqrt().clamp_min(1e-3).float()


class TemporalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)

    def forward(self, x, valid):
        value = torch.where(valid[..., None], F.silu(self.norm(x)), 0)
        value = self.conv(value.transpose(1, 2)).transpose(1, 2)
        return torch.where(valid[..., None], x + F.silu(value), 0)


class DirectAudioEncoder(nn.Module):
    def __init__(self, mean, std, output_dim=64, hidden=128):
        super().__init__()
        self.register_buffer('feature_mean', mean.detach().clone())
        self.register_buffer('feature_std', std.detach().clone())
        self.input = nn.Linear(len(mean), hidden)
        self.blocks = nn.ModuleList([TemporalBlock(hidden, d) for d in (1, 2, 4)])
        self.output = nn.Linear(hidden, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, weight):
        valid = weight > 0
        normalized = torch.where(valid[..., None], (features - self.feature_mean) / self.feature_std, 0)
        hidden = torch.where(valid[..., None], F.silu(self.input(normalized)), 0)
        for block in self.blocks:
            hidden = block(hidden, valid)
        return center(self.output(hidden), weight)


def interpolate_local(local, weight, valid, stride):
    """Same fixed bin centers as source; adding padding cannot retime speech."""
    position = ((torch.arange(valid.shape[1], device=local.device, dtype=local.dtype) - (stride - 1) / 2) / stride).clamp_min(0)
    last = ((weight > 0) * torch.arange(weight.shape[1], device=local.device)).amax(1)
    lower = torch.minimum(position.floor().long()[None].expand(len(local), -1), last[:, None])
    upper = torch.minimum(lower + 1, last[:, None])
    fraction = (position - position.floor())[None, :, None]
    left = local.gather(1, lower[..., None].expand(-1, -1, local.shape[-1]))
    right = local.gather(1, upper[..., None].expand(-1, -1, local.shape[-1]))
    return torch.where(valid[..., None], left * (1 - fraction) + right * fraction, 0)


def direct_affect(system, batch, local, weight, mode='full'):
    if mode not in MODES:
        raise ValueError('Unknown direct audio mode')
    if mode == 'reverse':
        local = reverse_controls(local, weight)
    if mode == 'zero':
        local = torch.zeros_like(local)
    local = local * audio_activity_gate(batch['affect'])[:, None, None]
    local = interpolate_local(local, weight, batch['q']['valid'], system.motion_teacher.stride)
    return {**batch['affect'], 'local': local}


def protected_hash(system):
    return state_hash({k: v for k, v in system.state_dict().items() if not k.startswith('renderer.')})


def configure(system):
    freeze_module(system)
    system.renderer.requires_grad_(True)
    system.eval()
    return list(system.renderer.parameters())


def masked_mean_channel(values, mask):
    clean = torch.where(mask, values, 0)
    return clean.sum((0, 1)) / mask.sum((0, 1)).clamp_min(1)


def fit_motion_scales(train):
    q = train['q']
    valid, target = q['valid'], q['motion'].float()
    mask = observed(q)
    baseline = train['base']['b0'].float() + train['identity']['baseline'].float()[:, None]
    residual = center(target - baseline, valid.float())
    dynamic = masked_mean_channel(residual.square(), mask).sqrt().clamp_min(.02)
    return {'dynamic': dynamic.float()}


def generated_losses(prediction, batch, scales):
    q = batch['q']
    target, valid, mask = q['motion'], q['valid'], observed(q)
    # Remove unobserved targets before centering or differences. NaN padding
    # must not leak into observed losses, including adjacent-frame terms.
    target = torch.where(mask, target, 0)
    prediction = torch.where(mask, prediction, 0)
    baseline = batch['base']['b0'] + batch['identity']['baseline'][:, None]
    baseline = torch.where(mask, baseline, 0)
    pc = center(prediction - baseline, valid.float())
    tc = center(target - baseline, valid.float())
    upper = mask[..., UPPER]
    centered = ((pc - tc)[..., UPPER].abs() / scales['dynamic'][UPPER])[upper].mean()
    return {'centered_upper': centered}


def training_loss(system, encoder, batch, features, weight, scales, noise, flow_time, arm):
    if arm not in ('flow', 'dynamics'):
        raise ValueError('Unknown training arm')
    q = batch['q']
    affect = direct_affect(system, batch, encoder(features, weight), weight)
    flow = system.flow(q['motion'], q['content'], q['valid'], batch['identity'], affect,
                       noise=noise, time=flow_time, base=batch['base'])
    losses = {'flow': (flow['prediction'] - flow['velocity_target'])[observed(q)].square().mean()}
    total = losses['flow']
    if arm == 'dynamics':
        # Genuine deployment rollout from noise: no target state enters decode.
        pred = system.generate(q['content'], q['valid'], batch['identity'], affect,
                               initial_noise=noise, steps=12, base=batch['base'])['motion']
        losses.update(generated_losses(pred, batch, scales))
        total = total + sum(LOSS_WEIGHTS[k] * losses[k] for k in LOSS_WEIGHTS)
    return total, losses


def restore_direct(system, encoder, payload):
    if payload.get('schema') != SCHEMA or payload.get('recipe_sha256') != canonical_hash(payload['recipe']):
        raise ValueError('Checkpoint schema/recipe differs')
    if payload['protected_sha256'] != protected_hash(system):
        raise ValueError('Protected source weights differ')
    for key in ('renderer', 'encoder'):
        if state_hash(payload[key]) != payload[key + '_sha256']:
            raise ValueError('Corrupt ' + key + ' state')
    system.renderer.load_state_dict(payload['renderer'], strict=True)
    encoder.load_state_dict(payload['encoder'], strict=True)
    system.eval(); encoder.eval()


@torch.no_grad()
def evaluate(system, encoder, split, bundle, device, path, seeds=NOISE_SEEDS, batch_size=32):
    features, weight = audio_features(bundle), bundle['weight'].float()
    reports, curves = {}, {}
    for seed in seeds:
        noise = torch.randn(split['q']['motion'].shape, generator=torch.Generator().manual_seed(seed))
        predictions, logits = {m: [] for m in MODES}, {m: [] for m in MODES}
        for start in range(0, len(features), batch_size):
            ids = torch.arange(start, min(start + batch_size, len(features)))
            batch, w = batch_to_device(split, ids, device), weight[ids].to(device)
            q = batch['q']
            local = encoder(features[ids].to(device), w)
            for mode in MODES:
                affect = direct_affect(system, batch, local, w, mode)
                pred = system.generate(q['content'], q['valid'], batch['identity'], affect,
                    initial_noise=noise[ids].to(device), steps=12, base=batch['base'])['motion']
                predictions[mode].append(pred.cpu())
                residual = torch.where(observed(q), pred - batch['base']['b0'] - batch['identity']['baseline'][:, None], 0)
                logits[mode].append(system.motion_teacher(residual, q['valid'])['emotion_logits'].cpu())
        curves[str(seed)], reports[str(seed)] = {}, {}
        for mode in MODES:
            pred = torch.cat(predictions[mode])
            curves[str(seed)][mode] = pred
            report = basic_metrics(pred, split)
            report['frozen_teacher_emotion_accuracy'] = float((torch.cat(logits[mode]).argmax(-1) == split['q']['emotion_id']).float().mean())
            report['emotion_measurement_note'] = 'Training teacher readout only; not independent emotion quality.'
            reports[str(seed)][mode] = report
        print(json.dumps({'stage': 'direct_eval', 'seed': seed}), flush=True)
    save_checkpoint(path, {'motion': curves, 'noise_seeds': list(seeds), 'decode_steps': 12})
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('flow', 'dynamics'), required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh isolated output required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(46); np.random.seed(46); torch.manual_seed(46)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(46)
    loaded = load_source(args.source_run, args.device)
    system, cache, bundle = (loaded[k] for k in ('system', 'cache', 'bundle'))
    train, validation = cache['splits']['train'], cache['splits']['validation']
    tr, dev = bundle['bundles']['internal'], bundle['bundles']['external_dev']
    features, weight = audio_features(tr), tr['weight'].float()
    mean, std = feature_statistics(features, weight)
    encoder = DirectAudioEncoder(mean, std, system.local_projection.out_features).to(args.device).eval()
    renderer_params = configure(system)
    parameters = renderer_params + list(encoder.parameters())
    protected = protected_hash(system)
    scales = fit_motion_scales(train)
    root = Path(__file__).resolve().parents[1]
    paths = list(dict.fromkeys(source_paths() + [Path(__file__), root/'docs/DIRECT_AUDIO_DYNAMICS_PROTOCOL.md', root/'tests/test_direct_audio_dynamics.py']))
    recipe = {'schema': SCHEMA, 'arm': args.arm, 'source_run': str(args.source_run.resolve()),
        'input_sha256': loaded['input_sha256'], 'source_adapter_sha256': loaded['source_adapter_sha256'],
        'data_scope': loaded['data_scope'], 'source_sha256': {str(x.resolve()): sha(x) for x in paths},
        'seed': 46, 'epochs': 8, 'batch_size': 16, 'optimizer': 'Adam', 'renderer_lr': 1e-5,
        'encoder_lr': 1e-4, 'clip_grad_norm': 1., 'loss_weights': LOSS_WEIGHTS,
        'initial_system_sha256': state_hash(system.state_dict()), 'initial_encoder_sha256': state_hash(encoder.state_dict()),
        'protected_sha256': protected, 'motion_scales_sha256': state_hash(scales),
        'decode_steps': 12, 'eval_noise_seeds': list(NOISE_SEEDS), 'eval_modes': list(MODES),
        'source_features': 'cached content768+middle(aggregate2/4/6)768+prosody4, stride4 bins; not full WavLM extraction',
        'initial_local': 'zero; equals old zero-local, not old full-local',
        'trainable_parameters': sum(x.numel() for x in parameters),
        'default_replaced': False, 'test_loaded': False, 'checkpoint_selection_performed': False}
    args.output.mkdir(parents=True)
    for path in paths:
        dest = args.output/'source'/path.resolve().relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    save_json(args.output/'provenance.json', {'recipe': recipe, 'recipe_sha256': canonical_hash(recipe)})
    optimizer = torch.optim.Adam([{'params': renderer_params, 'lr': 1e-5}, {'params': encoder.parameters(), 'lr': 1e-4}])
    generator = torch.Generator().manual_seed(46)
    batch_hash, noise_hash = hashlib.sha256(), hashlib.sha256()
    epoch_done, step, started = 0, 0, time.time()

    def payload():
        if protected_hash(system) != protected:
            raise RuntimeError('Protected weights changed')
        return {'schema': SCHEMA, 'recipe': recipe, 'recipe_sha256': canonical_hash(recipe),
            'renderer': system.renderer.state_dict(), 'encoder': encoder.state_dict(),
            'renderer_sha256': state_hash(system.renderer.state_dict()), 'encoder_sha256': state_hash(encoder.state_dict()),
            'protected_sha256': protected, 'motion_scales': scales, 'optimizer': optimizer.state_dict(),
            'rng': capture_rng(generator), 'completed_epochs': epoch_done, 'step': step,
            'minibatch_sha256': batch_hash.hexdigest(), 'noise_time_sha256': noise_hash.hexdigest()}

    save_checkpoint(args.output/'epoch000.pt', payload())
    # Epoch0 is algebraically the prior zero-local model for all three modes.
    # Keep one seed evidence; final conclusions use the prescribed eight seeds.
    initial = evaluate(system, encoder, validation, dev, args.device, args.output/'epoch000_curves.pt', seeds=(42,))
    save_json(args.output/'epoch000_evaluation.json', initial)
    gpu_scales = {k: v.to(args.device) for k, v in scales.items()}
    for epoch in range(1, 9):
        totals, seen = {}, 0
        for ids in torch.randperm(len(weight), generator=generator).split(16):
            noise, flow_time, _ = draws(generator, len(ids), train['q']['motion'].shape[1:])
            batch_hash.update(ids.numpy().tobytes())
            noise_hash.update(noise.numpy().tobytes()); noise_hash.update(flow_time.numpy().tobytes())
            batch = batch_to_device(train, ids, args.device)
            total, losses = training_loss(system, encoder, batch, features[ids].to(args.device), weight[ids].to(args.device),
                                         gpu_scales, noise.to(args.device), flow_time.to(args.device), args.arm)
            norm = optimize(total, optimizer, parameters)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * len(ids)
            seen += len(ids); step += 1
            if step % 100 == 0:
                print(json.dumps({'arm': args.arm, 'step': step, 'losses': {k: float(v.detach()) for k,v in losses.items()}, 'grad_norm': norm}), flush=True)
        epoch_done = epoch
        save_checkpoint(args.output/'last.pt', payload())
        row = {'epoch': epoch, 'step': step, 'losses': {k: v/seen for k,v in totals.items()},
            'elapsed_seconds': time.time()-started, 'minibatch_sha256': batch_hash.hexdigest(), 'noise_time_sha256': noise_hash.hexdigest()}
        save_json(args.output/f'epoch{epoch:03d}.json', row); print(json.dumps(row), flush=True)
    checkpoint = args.output/'final_epoch008.pt'
    save_checkpoint(checkpoint, payload())
    curve_path = args.output/'final_epoch008_curves.pt'
    final = evaluate(system, encoder, validation, dev, args.device, curve_path)
    binding = curve_binding(curve_path, checkpoint, recipe, loaded['input_sha256']['cache'])
    save_json(args.output/'summary.json', {'schema': SCHEMA, 'recipe_sha256': canonical_hash(recipe), 'arm': args.arm,
        'completed_epochs': epoch_done, 'optimizer_steps': step, 'final': final, 'curve_provenance': binding,
        'protected_unchanged': protected_hash(system) == protected, 'elapsed_seconds': time.time()-started,
        'test_loaded': False, 'default_replaced': False, 'checkpoint_selection_performed': False})
    save_json(args.output/'output_hashes.json', {str(x.relative_to(args.output)): sha(x) for x in args.output.rglob('*') if x.is_file() and x.name != 'output_hashes.json'})
    print('DIRECT_AUDIO_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
