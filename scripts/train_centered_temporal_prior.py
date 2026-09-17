"""Matched white/AR1 centered upper flow; preserve frozen baseline means."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_temporal_repair as r
from scripts.audit_temporal_repair import summarize, paired_metrics
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, capture_rng, canonical_hash
from scripts.train_predictable_renderer import state_hash
from kinetalk_b0.models.centered_upper_flow import CenteredUpperFlow, center_valid, fit_dynamic_statistics
from kinetalk_b0.models.mean_preserving_upper import compose_mean_preserving_upper

SCHEMA = 'mean_preserving_temporal_prior_v1'
CC = list(r.UPPER_INDICES)


@torch.no_grad()
def cache_evaluation_bases(system, adapter, data, identities, args, steps):
    q = data['splits']['validation']
    for seed in r.SEEDS:
        noise = torch.randn(len(q['valid']), q['valid'].shape[1], 52, generator=torch.Generator().manual_seed(seed))
        original, aligned = [], []
        for ix in torch.arange(len(q['valid'])).split(args.batch_size):
            b = r.subset(q, ix, args.device); ident = r.batch_identity(identities, b)
            affect = {'global': b['audio_global'], 'intensity_value': b['audio_intensity'], 'local': b['audio_local']}
            base = {'b0': b['b0'], 'h0': b['h0']}
            original.append(system.generate(b['content'], b['valid'], ident, affect, initial_noise=noise[ix].to(args.device), steps=steps, base=base)['motion'].cpu())
            affect = r.projected_affect(system, adapter, b)
            aligned.append(system.generate(b['content'], b['valid'], ident, affect, initial_noise=noise[ix].to(args.device), steps=steps, base=base)['motion'].cpu())
        q[f'original_{seed}'] = torch.cat(original)
        q[f'aligned_{seed}'] = torch.cat(aligned)


def time_intervention(local, content, valid, mode):
    if mode == 'static':
        def static(x):
            return torch.where(valid[..., None], torch.where(valid[..., None], x, 0.).sum(1, keepdim=True)/valid.sum(1)[:, None, None], 0.)
        return static(local), static(content)
    if mode == 'reverse':
        local, content = local.clone(), content.clone()
        for row in range(len(valid)):
            ids = valid[row].nonzero(as_tuple=True)[0]
            local[row, ids] = local[row, ids.flip(0)]
            content[row, ids] = content[row, ids.flip(0)]
    return local, content


def temporal_stats(pred, target, valid):
    result = {}
    pairs = valid[:, 1:] & valid[:, :-1]
    for name, indices in [('brows', CC[:5]), ('eyes_expression', CC[5:])]:
        values = []
        for x in (pred[..., indices], target[..., indices]):
            centered = center_valid(x.double(), valid)
            left, right = centered[:, :-1][pairs], centered[:, 1:][pairs]
            values.append({'adjacent_correlation': float((left*right).sum()/(left.square().sum()*right.square().sum()).sqrt().clamp_min(1e-15)),
                           'velocity_mse_energy': float((right-left).square().mean())})
        result[name] = {'prediction': values[0], 'target': values[1],
                        'velocity_rms_ratio': (values[0]['velocity_mse_energy']/max(values[1]['velocity_mse_energy'], 1e-15))**.5}
    return result


@torch.no_grad()
def evaluate(system, upper, local, data, identities, args, scales, steps):
    q = data['splits']['validation']
    ids = torch.arange(min(32, len(q['valid'])) if args.smoke else len(q['valid']))
    reference = r.subset(q, ids, 'cpu'); predictions = {}; metrics = {}; mean_error = 0.
    seeds = (42,) if args.smoke else r.SEEDS
    for seed in seeds:
        white = torch.randn(len(q['valid']), q['valid'].shape[1], 9, generator=torch.Generator().manual_seed(seed))
        modes = ['full', 'base', 'aligned_centered'] + (['static', 'reverse'] if seed == 42 else [])
        for mode in modes:
            outputs, emotions = [], []
            for ix in ids.split(args.batch_size):
                b = r.subset(q, ix, args.device); ident = r.batch_identity(identities, b)
                baseline = b[f'original_{seed}']
                if mode == 'base':
                    pred = baseline
                elif mode == 'aligned_centered':
                    pred = compose_mean_preserving_upper(baseline, b[f'aligned_{seed}'][..., CC], b['valid'])
                else:
                    native = local(b['audio_features'], b['valid'])['local']
                    native, content = time_intervention(native, b['h0'], b['valid'], mode)
                    noise = upper.prior(white[ix].to(args.device), b['valid'])
                    dynamic = upper.decode(b['valid'], content, ident['code'],
                        {'global': b['audio_global'], 'intensity_value': b['audio_intensity']}, native, None, noise, steps=steps)
                    pred = compose_mean_preserving_upper(baseline, dynamic*scales, b['valid'])
                if not torch.equal(pred[..., list(r.NOT_UPPER)], baseline[..., list(r.NOT_UPPER)]):
                    raise RuntimeError('Nonupper baseline changed')
                mean_delta = torch.where(b['valid'][..., None], pred[..., CC]-baseline[..., CC], 0.).sum(1)/b['valid'].sum(1)[:, None]
                mean_error = max(mean_error, float(mean_delta.abs().max()))
                teacher = system.encode_motion(torch.where(r.obs(b), pred-b['b0']-ident['baseline'][:, None], 0.), b['valid'])
                emotions.extend((teacher['emotion_logits'].argmax(-1) == b['emotion_id']).cpu().tolist())
                outputs.append(pred.cpu())
            key = f'{seed}/{mode}'; predictions[key] = torch.cat(outputs)
            metrics[key] = {'populations': r.populations(predictions[key], reference),
                            'temporal': temporal_stats(predictions[key], reference['motion'], reference['valid']),
                            'generated_emotion_accuracy_nonindependent': sum(emotions)/len(emotions)}
    if mean_error > 2e-6:
        raise RuntimeError('Upper mean preservation tolerance exceeded')
    curves = {k: reference[k] for k in ('clip_id', 'motion', 'valid', 'times', 'channel_mask', 'b0', 'emotion_id', 'speaker_id')}
    curves['target'] = curves.pop('motion')
    curves.update(schema=SCHEMA, predictions=predictions, noise_seeds=list(seeds), decode_steps=steps)
    report = {'schema': SCHEMA, 'clips': len(ids), 'modes': metrics, 'upper_mean_max_abs_difference': mean_error,
              'nonupper_baseline_exact': True, 'test_loaded': False,
              'identity': r.identity_report(system, data, args.device)}
    if not args.smoke:
        report['distribution'] = summarize(curves, curves['emotion_id'])
    return report, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'audio', 'targets', 'enrollment', 'native-root', 'trained-run', 'align-run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=61)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise FileExistsError('Fresh output directory required')
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('Positive epochs and batch size required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    data = r.load_training_inputs(args.source_run, args.audio, args.targets, args.enrollment, args.native_root)
    ck, bindings, steps, stride = r.matching_checkpoints(args.trained_run, data)
    system = data['system'].to(args.device).eval().requires_grad_(False)
    system.load_state_dict(ck['teacher']['system'])
    audio = r._make_audio(ck['audio']['audio'], stride, args.device)
    alignpath = args.align_run/'final.pt'
    if r.sha(alignpath) != json.loads((args.align_run/'complete.json').read_text())['final_sha256']:
        raise ValueError('Alignment checkpoint binding differs')
    aligned = torch.load(alignpath, map_location='cpu', weights_only=False)
    if aligned['source_bindings'] != bindings or aligned['config'] != data['config'] or aligned['phase'] != 'align':
        raise ValueError('Alignment lineage differs')
    adapter = r.AlignedAudioLocal(audio.feature_mean, audio.feature_std, hidden=audio.input.out_features,
        rank=system.motion_teacher.rank, stride=system.motion_teacher.stride).to(args.device).eval().requires_grad_(False)
    adapter.load_state_dict(aligned['adapter'])
    r.cache_current_base(system, data, args.device, args.batch_size)
    identities = r.identity_cache(system, data, args.device)
    r.cache_conditions(system, audio, data, identities, args.device, args.batch_size)
    cache_evaluation_bases(system, adapter, data, identities, args, steps)
    train = data['splits']['train']
    if not train['channel_mask'][:, CC].all():
        raise ValueError('All nine fit channels must be observed')
    scales, fitted_rho = fit_dynamic_statistics(train['motion'][..., CC], train['valid'])
    scales = scales.to(args.device)
    frozen = {k: state_hash(v.state_dict()) for k, v in [('system', system), ('audio', audio), ('adapter', adapter)]}
    args.output.mkdir(parents=True)
    initial_hashes, draw_hashes = {}, {}
    for arm in ('white', 'ar1'):
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        local = copy.deepcopy(audio).requires_grad_(False)
        local.input.load_state_dict(adapter.input.state_dict()); local.blocks.load_state_dict(adapter.blocks.state_dict())
        torch.nn.init.zeros_(local.local_head.weight); torch.nn.init.zeros_(local.local_head.bias)
        for module in (local.input, local.blocks, local.local_head):
            module.requires_grad_(True)
        rho = fitted_rho if arm == 'ar1' else torch.zeros_like(fitted_rho)
        upper = CenteredUpperFlow(data['config'], rho).to(args.device).eval()
        parameters = list(upper.parameters())+[p for p in local.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
        initial_hashes[arm] = {'upper': state_hash({k: v for k, v in upper.state_dict().items() if k != 'prior_rho'}), 'local': state_hash(local.state_dict())}
        output = args.output/arm; output.mkdir()
        recipe = {'schema': SCHEMA, 'arm': arm, 'epochs': 1 if args.smoke else args.epochs, 'seed': args.seed,
            'batch_size': args.batch_size, 'smoke': args.smoke, 'frozen': frozen, 'source_bindings': bindings,
            'alignment_sha256': r.sha(alignpath), 'data_provenance': data['provenance'], 'config': data['config'],
            'scales': scales.cpu().tolist(), 'fitted_rho': fitted_rho.tolist(), 'prior_rho': rho.tolist(),
            'initial': initial_hashes[arm], 'test_loaded': False, 'default_replaced': False, 'fixed_final_epoch': True,
            'objective': 'Centered normalized raw upper motion conditional flow matching only',
            'source_sha256': {name: r.sha(Path(__file__).resolve().parents[1]/name) for name in (
                'scripts/train_centered_temporal_prior.py', 'kinetalk_b0/models/centered_upper_flow.py',
                'kinetalk_b0/models/mean_preserving_upper.py', 'kinetalk_b0/models/temporal_upper.py')},
            'protocol_sha256': r.sha(Path(__file__).resolve().parents[1]/'docs/CENTERED_TEMPORAL_PRIOR_PROTOCOL_20260917.md')}
        digest = canonical_hash(recipe)
        save_json(output/'provenance.json', {'recipe': recipe, 'recipe_sha256': digest})
        gen = torch.Generator().manual_seed(args.seed)
        ids = torch.arange(min(32, len(train['valid'])) if args.smoke else len(train['valid']))
        draw_hashes[arm] = []; total_steps = 0
        for epoch in range(recipe['epochs']):
            t0 = time.monotonic(); losses = []; draws = hashlib.sha256()
            for ix in ids[torch.randperm(len(ids), generator=gen)].split(args.batch_size):
                b = r.subset(train, ix, args.device); ident = r.batch_identity(identities, b)
                target = center_valid(b['motion'][..., CC], b['valid'])/scales
                raw_noise = torch.randn(target.shape, generator=gen); flow_time = torch.rand(len(ix), generator=gen)
                for value in (ix, raw_noise, flow_time):
                    draws.update(value.numpy().tobytes())
                noise = upper.prior(raw_noise.to(args.device), b['valid'])
                native = local(b['audio_features'], b['valid'])['local']
                loss = upper.flow_loss(target, b['valid'], b['h0'], ident['code'],
                    {'global': b['audio_global'], 'intensity_value': b['audio_intensity']}, native, None, noise, flow_time.to(args.device))
                r.optimize(loss, optimizer, parameters)
                losses.append(float(loss.detach())); total_steps += 1
            draw_hashes[arm].append(draws.hexdigest())
            record = {'arm': arm, 'epoch': epoch+1, 'loss': sum(losses)/len(losses), 'seconds': time.monotonic()-t0,
                      'batch_raw_noise_time_sha256': draws.hexdigest(), 'total_steps': total_steps}
            save_json(output/f'epoch{epoch+1:03d}.json', record)
            save_json(args.output/'status.json', {'status': 'training', **record})
            payload = {'schema': SCHEMA, 'arm': arm, 'upper': upper.state_dict(), 'local': local.state_dict(),
                'scales': scales.cpu(), 'completed_epochs': epoch+1, 'total_steps': total_steps, 'recipe_sha256': digest,
                'optimizer': optimizer.state_dict(), 'rng': capture_rng(gen)}
            save_checkpoint(output/'last.pt', payload)
            print(json.dumps(record), flush=True)
        for name, module in [('system', system), ('audio', audio), ('adapter', adapter)]:
            if state_hash(module.state_dict()) != frozen[name] or any(p.grad is not None for p in module.parameters()):
                raise RuntimeError('Frozen subsystem changed: '+name)
        report, curves = evaluate(system, upper, local, data, identities, args, scales, steps)
        save_json(output/'evaluation.json', report); save_checkpoint(output/'curves.pt', curves)
        save_checkpoint(output/'final.pt', {k: v for k, v in payload.items() if k not in ('optimizer', 'rng')})
        save_json(output/'complete.json', {'final_sha256': r.sha(output/'final.pt'), 'curves_sha256': r.sha(output/'curves.pt'),
            'completed_epochs': recipe['epochs'], 'frozen': frozen, 'recipe_sha256': digest})
    if initial_hashes['white'] != initial_hashes['ar1'] or draw_hashes['white'] != draw_hashes['ar1']:
        raise RuntimeError('Paired initialization or random streams differ')
    save_json(args.output/'matched_audit.json', {'initial_parameters_equal': True, 'raw_random_streams_equal': True,
        'initial_sha256': initial_hashes, 'epoch_draw_sha256': draw_hashes, 'frozen': frozen})
    save_json(args.output/'status.json', {'status': 'complete', 'elapsed_seconds': time.monotonic()-started,
        'epochs_per_arm': 1 if args.smoke else args.epochs, 'smoke': args.smoke})
    print('CENTERED_PRIOR_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
