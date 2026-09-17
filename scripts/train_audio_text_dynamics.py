"""Matched frozen-ASR text/no-text expression intensity pilot; no GT conditions."""
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
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.audio_text_affect import AudioTextAffect
from kinetalk_b0.label_guided_intensity import regional_intensity
from scripts.train_audio_conditioned_flow_probe import load_source, NOISE_SEEDS, source_paths
from scripts.train_direct_audio_dynamics import configure, protected_hash
from scripts.train_formal_predictable_projection import canonical_hash, capture_rng, save_checkpoint, save_json
from scripts.train_predictable_renderer import batch_to_device, observed, optimize, sha, state_hash
from scripts.train_projection_schedule_ablation import draws, curve_binding

SCHEMA = 'audio_text_dynamics_v1'
GROUPS = {'brows': list(range(41, 46)), 'eyes_expression': [5, 6, 12, 13], 'mouth': list(range(14, 41))}
LOSS_WEIGHTS = {'predicted_intensity': .25, 'raw_motion': .1, 'output_intensity': .1, 'domain': .1}
MODES = ('full', 'zero', 'no_text', 'static_intensity', 'shuffled_text', 'reverse_audio', 'oracle_intensity')


def validate_inputs(cache, audio, text, targets, cache_hash):
    for obj, schema in ((audio, 'label_guided_audio_cache_v1'), (text, 'audio_text_semantics_v1'),
                        (targets, 'expression_intensity_targets_v1')):
        if obj.get('schema') != schema or set(obj.get('splits', {})) != {'train', 'validation'}:
            raise ValueError('Unexpected cache schema/splits: ' + str(obj.get('schema')))
        prov = obj['provenance']
        bound = prov.get('renderer_cache_sha256', prov.get('cache_sha256', prov.get('source_sha256', {}).get('cache')))
        if bound != cache_hash:
            raise ValueError('Derived cache binding differs: ' + schema)
    for role in ('train', 'validation'):
        q = cache['splits'][role]['q']
        for obj in (audio, text, targets):
            if list(map(str, obj['splits'][role]['clip_id'])) != list(map(str, q['clip_id'])):
                raise ValueError('Cache clip order differs')
        a, tx, tg = (v['splits'][role] for v in (audio, text, targets))
        if not torch.equal(a['valid'], q['valid']) or not torch.equal(a['times'], q['times']):
            raise ValueError('Native audio clock differs')
        if a['features'].shape != (*q['valid'].shape, 1540):
            raise ValueError('Expected complete frame features')
        if (tx['tokens'].shape[:2] != tx['token_valid'].shape or len(tx['tokens']) != len(q['valid'])
                or tx['token_valid'].dtype != torch.bool or not torch.isfinite(tx['tokens'][tx['token_valid']]).all()):
            raise ValueError('Text tokens/masks differ')
        if tg['intensity'].shape != (*q['valid'].shape, 1) or tg['valid'].shape != tg['intensity'].shape:
            raise ValueError('Intensity target clock differs')
        if not torch.isfinite(tg['intensity'][tg['valid']]).all() or (tg['intensity'][tg['valid']] < 0).any():
            raise ValueError('Invalid observed intensity')
    if audio['feature_stats']['fit_clip_ids'] != audio['splits']['train']['clip_id']:
        raise ValueError('Feature statistics fit IDs differ')


def make_encoder(audio, text, targets, device):
    tg = targets['splits']['train']
    initial = max(.001, float(tg['intensity'][tg['valid']].mean()))
    return AudioTextAffect(audio['feature_stats']['mean'], audio['feature_stats']['std'],
        text_dim=text['splits']['train']['tokens'].shape[-1], initial_intensity=initial).to(device).eval()


def slice_inputs(audio, text, targets, ids, device, *, text_ids=None):
    tids = ids if text_ids is None else text_ids
    return {'features': audio['features'][ids].to(device).float(),
        'tokens': text['tokens'][tids].to(device).float(), 'token_valid': text['token_valid'][tids].to(device),
        'target_intensity': targets['intensity'][ids].to(device).float(),
        'intensity_valid': targets['valid'][ids].to(device), 'anchors': targets['anchors'][ids].to(device).float(),
        'anchor_valid': targets['anchor_valid'][ids].to(device)}


def reverse_valid(features, valid):
    result = torch.zeros_like(features)
    for row in range(len(features)):
        ix = valid[row].nonzero(as_tuple=True)[0]
        result[row, ix] = features[row, ix.flip(0)]
    return result


def build_affect(encoder, batch, data, arm, mode='full'):
    if arm not in ('text', 'no_text') or mode not in MODES:
        raise ValueError('Unknown arm/intervention')
    valid = batch['q']['valid']
    features = reverse_valid(data['features'], valid) if mode == 'reverse_audio' else data['features']
    override = None
    if mode == 'oracle_intensity':
        # Explicit target-conditioned diagnostic; never selected by training.
        predicted = encoder(features, valid, batch['affect']['global'], data['tokens'], data['token_valid'],
                            use_text=arm == 'text')['predicted_intensity']
        override = torch.where(data['intensity_valid'], data['target_intensity'], predicted)
    output = encoder(features, valid, batch['affect']['global'], data['tokens'], data['token_valid'],
        use_text=arm == 'text' and mode != 'no_text', intensity_override=override,
        static_intensity=mode == 'static_intensity')
    local = torch.zeros_like(output['local']) if mode == 'zero' else output['local']
    return {**batch['affect'], 'local': local}, output


def masked_huber(prediction, target, mask):
    if mask.dtype != torch.bool or mask.shape != prediction.shape or target.shape != prediction.shape:
        raise ValueError('Huber masks must match')
    if not mask.any():
        return torch.where(mask, prediction, 0.).sum() * 0
    return F.smooth_l1_loss(prediction[mask], target[mask], beta=1.)


def generated_losses(prediction, batch, data, scales):
    mask = observed(batch['q'])
    target = batch['q']['motion']
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite observed motion')
    raw = torch.stack([masked_huber(prediction[..., cc] / scales[cc], target[..., cc] / scales[cc], mask[..., cc])
                       for cc in GROUPS.values()]).mean()
    intensity, imask = regional_intensity(prediction, mask & data['anchor_valid'][:, None], data['anchors'], scales)
    intensity_loss = masked_huber(intensity, data['target_intensity'], imask & data['intensity_valid'])
    domain = (F.relu(-prediction[mask]) + F.relu(prediction[mask] - 1.)).mean() / .02
    return {'raw_motion': raw, 'output_intensity': intensity_loss, 'domain': domain}


def training_loss(system, encoder, batch, data, scales, noise, flow_time, arm):
    affect, output = build_affect(encoder, batch, data, arm, 'full')
    q = batch['q']
    safe_target = torch.where(observed(q), q['motion'], 0.)
    flow = system.flow(safe_target, q['content'], q['valid'], batch['identity'], affect,
                       noise=noise, time=flow_time, base=batch['base'])
    losses = {'flow': (flow['prediction'] - flow['velocity_target'])[observed(q)].square().mean(),
        'predicted_intensity': masked_huber(output['predicted_intensity'], data['target_intensity'], data['intensity_valid'])}
    generated = system.generate(q['content'], q['valid'], batch['identity'], affect,
                               initial_noise=noise, steps=12, base=batch['base'])['motion']
    losses.update(generated_losses(generated, batch, data, scales))
    return losses['flow'] + sum(LOSS_WEIGHTS[k] * losses[k] for k in LOSS_WEIGHTS), losses


def mismatch_indices(q):
    # Prefer same speaker/emotion, but never use the same sentence. Fixed without metrics.
    sentences, result = list(map(str, q['sentence_id'])), []
    for i, sentence in enumerate(sentences):
        choices = [j for j in range(len(sentences)) if sentences[j] != sentence]
        if not choices: raise ValueError('Mismatched-text audit needs multiple sentences')
        same = [j for j in choices if int(q['speaker_id'][i]) == int(q['speaker_id'][j])
                and int(q['emotion_id'][i]) == int(q['emotion_id'][j])]
        result.append((same or choices)[i % len(same or choices)])
    return torch.tensor(result)


@torch.no_grad()
def evaluate(system, encoder, split, audio, text, targets, arm, device, path, *, seeds=NOISE_SEEDS, full_audit=True):
    curves, intensities = {}, {}
    mismatch = mismatch_indices(split['q'])
    for seed in seeds:
        modes = MODES if full_audit and seed == seeds[0] else ('full', 'zero')
        noise = torch.randn(split['q']['motion'].shape, generator=torch.Generator().manual_seed(seed))
        curves[str(seed)] = {}
        for mode in modes:
            values, predicted_i, driving_i = [], [], []
            for ids in torch.arange(len(noise)).split(32):
                batch = batch_to_device(split, ids, device)
                data = slice_inputs(audio, text, targets, ids, device, text_ids=mismatch[ids] if mode == 'shuffled_text' else None)
                affect, output = build_affect(encoder, batch, data, arm, mode)
                q = batch['q']
                values.append(system.generate(q['content'], q['valid'], batch['identity'], affect,
                    initial_noise=noise[ids].to(device), steps=12, base=batch['base'])['motion'].cpu())
                if seed == seeds[0]:
                    predicted_i.append(output['predicted_intensity'].cpu()); driving_i.append(output['driving_intensity'].cpu())
            curves[str(seed)][mode] = torch.cat(values)
            if predicted_i:
                intensities[mode] = {'predicted': torch.cat(predicted_i), 'driving': torch.cat(driving_i)}
        print(json.dumps({'stage': 'evaluate', 'arm': arm, 'seed': seed, 'modes': modes}), flush=True)
    save_checkpoint(path, {'schema': SCHEMA, 'motion': curves, 'intensity': intensities, 'arm': arm,
        'noise_seeds': list(seeds), 'decode_steps': 12, 'mismatched_text_indices': mismatch,
        'clip_id': list(split['q']['clip_id']), 'oracle_is_target_conditioned': True})


def restore(system, encoder, payload):
    if payload.get('schema') != SCHEMA or canonical_hash(payload['recipe']) != payload.get('recipe_sha256'):
        raise ValueError('Checkpoint recipe invalid')
    if protected_hash(system) != payload['protected_sha256']:
        raise ValueError('Frozen source differs')
    for key, module in (('renderer', system.renderer), ('encoder', encoder)):
        if state_hash(payload[key]) != payload[key + '_sha256']: raise ValueError('Checkpoint tensor corruption')
        module.load_state_dict(payload[key], strict=True)
    system.eval(); encoder.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'audio-cache', 'text-cache', 'targets', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--arm', choices=('text', 'no_text'), required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke-steps', type=int, default=0, help='Separate fit-only smoke output; never counted as experiment')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh isolated output required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(46); np.random.seed(46); torch.manual_seed(46)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(46)
    loaded = load_source(args.source_run, args.device)
    system, cache = loaded['system'], loaded['cache']
    paths = {'audio': args.audio_cache, 'text': args.text_cache, 'targets': args.targets}
    derived_hashes = {k: sha(v) for k, v in paths.items()}
    audio, text, targets = [torch.load(paths[k], map_location='cpu', weights_only=False, mmap=True) for k in paths]
    validate_inputs(cache, audio, text, targets, loaded['input_sha256']['cache'])
    encoder = make_encoder(audio, text, targets, args.device)
    renderer_params = configure(system)
    params = renderer_params + list(encoder.parameters())
    protected = protected_hash(system)
    root = Path(__file__).resolve().parents[1]
    sources = list(dict.fromkeys(source_paths() + [Path(__file__), root/'scripts/train_direct_audio_dynamics.py',
        root/'kinetalk_b0/models/audio_text_affect.py', root/'kinetalk_b0/models/label_guided_affect.py',
        root/'kinetalk_b0/label_guided_intensity.py', root/'scripts/prepare_label_guided_audio_cache.py',
        root/'scripts/prepare_audio_text_semantics.py', root/'scripts/prepare_expression_intensity_targets.py',
        root/'docs/AUDIO_TEXT_DYNAMICS_PROTOCOL_20260917.md', root/'tests/test_audio_text_affect.py',
        root/'tests/test_audio_text_training.py']))
    recipe = {'schema': SCHEMA, 'arm': args.arm, 'source_run': str(args.source_run.resolve()),
        'input_sha256': loaded['input_sha256'], 'source_adapter_sha256': loaded['source_adapter_sha256'],
        'data_scope': loaded['data_scope'], 'derived_paths': {k: str(v.resolve()) for k,v in paths.items()},
        'derived_sha256': derived_hashes, 'source_sha256': {str(v.resolve()): sha(v) for v in sources},
        'seed': 46, 'epochs': 8, 'batch_size': 16, 'optimizer': 'Adam', 'renderer_lr': 1e-5, 'encoder_lr': 1e-4,
        'loss_weights': LOSS_WEIGHTS, 'decode_steps': 12, 'noise_seeds': list(NOISE_SEEDS),
        'initial_system_sha256': state_hash(system.state_dict()), 'initial_encoder_sha256': state_hash(encoder.state_dict()),
        'protected_sha256': protected, 'teacher_intensity_probability': 0., 'online_global_distillation': False,
        'trainable_parameters': sum(v.numel() for v in params), 'smoke_steps': args.smoke_steps,
        'test_loaded': False, 'default_replaced': False, 'checkpoint_selection_performed': False}
    args.output.mkdir(parents=True)
    for path in sources:
        dest = args.output/'source'/path.relative_to(root); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    save_json(args.output/'provenance.json', {'recipe': recipe, 'recipe_sha256': canonical_hash(recipe)})
    optimizer = torch.optim.Adam([{'params': renderer_params, 'lr': 1e-5}, {'params': encoder.parameters(), 'lr': 1e-4}])
    generator = torch.Generator().manual_seed(46)
    batch_hash, noise_hash = hashlib.sha256(), hashlib.sha256()
    epoch_done, step, started = 0, 0, time.time()
    train, val = cache['splits']['train'], cache['splits']['validation']
    scales = targets['scales'].to(args.device).float()

    def payload():
        if protected_hash(system) != protected: raise RuntimeError('Protected weights changed')
        return {'schema': SCHEMA, 'recipe': recipe, 'recipe_sha256': canonical_hash(recipe),
            'renderer': system.renderer.state_dict(), 'encoder': encoder.state_dict(),
            'renderer_sha256': state_hash(system.renderer.state_dict()), 'encoder_sha256': state_hash(encoder.state_dict()),
            'protected_sha256': protected, 'optimizer': optimizer.state_dict(), 'rng': capture_rng(generator),
            'completed_epochs': epoch_done, 'step': step, 'minibatch_sha256': batch_hash.hexdigest(), 'noise_time_sha256': noise_hash.hexdigest()}

    save_checkpoint(args.output/'epoch000.pt', payload())
    if not args.smoke_steps:
        evaluate(system, encoder, val, audio['splits']['validation'], text['splits']['validation'], targets['splits']['validation'],
                 args.arm, args.device, args.output/'epoch000_curves.pt', seeds=NOISE_SEEDS, full_audit=False)
        curve_binding(args.output/'epoch000_curves.pt', args.output/'epoch000.pt', recipe, loaded['input_sha256']['cache'])
    for epoch in range(1, 9):
        totals, seen = {}, 0
        for ids in torch.randperm(len(train['q']['valid']), generator=generator).split(16):
            noise, flow_time, _ = draws(generator, len(ids), train['q']['motion'].shape[1:])
            batch_hash.update(ids.numpy().tobytes()); noise_hash.update(noise.numpy().tobytes()); noise_hash.update(flow_time.numpy().tobytes())
            batch = batch_to_device(train, ids, args.device)
            data = slice_inputs(audio['splits']['train'], text['splits']['train'], targets['splits']['train'], ids, args.device)
            total, losses = training_loss(system, encoder, batch, data, scales, noise.to(args.device), flow_time.to(args.device), args.arm)
            norm = optimize(total, optimizer, params)
            for k, v in losses.items(): totals[k] = totals.get(k, 0.) + float(v.detach()) * len(ids)
            seen += len(ids); step += 1
            if step % 50 == 0 or step <= 3:
                print(json.dumps({'arm': args.arm, 'step': step, 'losses': {k: float(v.detach()) for k,v in losses.items()}, 'grad_norm': norm}), flush=True)
            if args.smoke_steps and step >= args.smoke_steps:
                save_checkpoint(args.output/'smoke.pt', payload())
                save_json(args.output/'smoke.json', {'steps': step, 'losses': {k:v/seen for k,v in totals.items()},
                    'protected_unchanged': protected_hash(system) == protected, 'finite_gradients': True, 'elapsed_seconds': time.time()-started})
                print('AUDIO_TEXT_SMOKE_COMPLETE', flush=True); return
        epoch_done = epoch
        save_checkpoint(args.output/f'epoch{epoch:03d}.pt', payload())
        row = {'epoch': epoch, 'step': step, 'losses': {k:v/seen for k,v in totals.items()}, 'elapsed_seconds': time.time()-started,
               'minibatch_sha256': batch_hash.hexdigest(), 'noise_time_sha256': noise_hash.hexdigest()}
        save_json(args.output/f'epoch{epoch:03d}.json', row); print(json.dumps(row), flush=True)
    checkpoint, curves = args.output/'epoch008.pt', args.output/'final_curves.pt'
    evaluate(system, encoder, val, audio['splits']['validation'], text['splits']['validation'], targets['splits']['validation'],
             args.arm, args.device, curves)
    binding = curve_binding(curves, checkpoint, recipe, loaded['input_sha256']['cache'])
    save_json(args.output/'summary.json', {'schema': SCHEMA, 'arm': args.arm, 'completed_epochs': epoch_done, 'optimizer_steps': step,
        'recipe_sha256': canonical_hash(recipe), 'curve_provenance': binding, 'protected_unchanged': protected_hash(system) == protected,
        'elapsed_seconds': time.time()-started, 'test_loaded': False, 'default_replaced': False})
    save_json(args.output/'output_hashes.json', {str(v.relative_to(args.output)): sha(v) for v in args.output.rglob('*')
        if v.is_file() and v.name != 'output_hashes.json'})
    print('AUDIO_TEXT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
