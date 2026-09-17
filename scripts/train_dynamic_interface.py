"""Matched local-only training with frozen global/identity/B0 and zero-local behavior.

Audio head-only is the control; joint also adapts the existing local condition
projection. Optional refiner changes only the audio control head. No new motion
head, regional routing, or additional loss family is introduced.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.train_neutral_affect_audio_ablation import state_hash, summarize
from scripts.train_neutral_affect_feature_probe import prepare_feature_batch
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, derivative_loss, device_batch, masked_mse,
    observed, optimize, prepare, select, sha, write_json,
)

GLOBAL_KEYS = ('global', 'emotion_logits', 'intensity_logits', 'intensity_value')


def configure_trainable(system, joint):
    freeze_module(system)
    allowed = {'audio_encoder.control_head.weight', 'audio_encoder.control_head.bias'}
    allowed.update(n for n, _ in system.named_parameters() if n.startswith('audio_encoder.control_refiner.'))
    if joint:
        allowed.add('renderer.local_emotion.weight')
    for name, param in system.named_parameters():
        param.requires_grad_(name in allowed)
    system.eval()
    return allowed


def dynamic_objective(system, query, base, identity, teacher, rng, teacher_probability):
    audio = system.encode_audio(query['audio'], query['valid'])
    size, device = len(query['motion']), query['motion'].device
    use_teacher = torch.rand(size, device=device, generator=rng) < teacher_probability
    # Keep an audio item even for the frozen-renderer control's rare all-teacher batch.
    if use_teacher.all():
        use_teacher[0] = False
    affect = {**audio, 'local': torch.where(use_teacher[:, None, None], teacher['local'], audio['local'])}
    times = torch.rand(size, device=device, generator=rng)
    times[torch.rand(size, device=device, generator=rng) < .2] = 0
    noise = torch.randn(query['motion'].shape, device=device, generator=rng)
    output = system.flow(query['motion'], query['content'], query['valid'], identity, affect,
                         noise=noise, time=times, base=base)
    flow = masked_mse(output['prediction'], output['velocity_target'], observed(query))
    displacement = derivative_loss(output['motion'], query['motion'], query) / system.residual_scale ** 2
    # The same motion objectives already used for teacher/run05. Global CE and
    # global latent loss are constants because the entire global path is frozen.
    return flow + .1 * displacement, {'flow': float(flow.detach()), 'displacement': float(displacement.detach()),
                                      'teacher_fraction': float(use_teacher.float().mean())}, (use_teacher, times, noise)


def build_system(cfg, checkpoint, refiner, seed):
    original = NeutralAffectSystem(cfg)
    original.load_state_dict(checkpoint['model'], strict=True)
    effective = copy.deepcopy(cfg)
    if not refiner:
        return original, effective
    effective['model']['audio_control_refiner'] = True
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        system = NeutralAffectSystem(effective)
    state = system.state_dict()
    added = set(state) - set(checkpoint['model'])
    if not added or any(not n.startswith('audio_encoder.control_refiner.') for n in added):
        raise ValueError('Only explicitly initialized control_refiner weights may be added')
    state.update(checkpoint['model'])
    system.load_state_dict(state, strict=True)
    return system, effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=('head', 'joint', 'refiner_joint'), required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--audio-lr', type=float, default=.0005)
    parser.add_argument('--interface-lr', type=float, default=.0001)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError('Positive training steps required')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf8'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('stage') != 'audio' or checkpoint['provenance']['config'] != cfg:
        raise ValueError('Exact audio checkpoint config required')
    for split in ('train', 'heldout'):
        if sha(args.data / (split + '.pt')) != checkpoint['provenance'][split + '_sha256']:
            raise ValueError('Continuation must use the original checkpoint caches')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    system, effective = build_system(cfg, checkpoint, args.mode == 'refiner_joint', args.seed)
    device = torch.device(cfg.get('device', 'cuda'))
    system.to(device)
    allowed = configure_trainable(system, args.mode != 'head')
    frozen = lambda: {n: p for n, p in system.state_dict().items() if n not in allowed}
    frozen_hash = state_hash(frozen())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'effective_config.yaml').write_text(yaml.safe_dump(effective, sort_keys=False), encoding='utf8')
    datasets = {s: NeutralAffectDataset(args.data / (s + '.pt')) for s in ('train', 'heldout')}
    prepared = {s: prepare(system, ds, device) for s, ds in datasets.items()}
    queries, bases = {s: p[0] for s, p in prepared.items()}, {s: p[1] for s, p in prepared.items()}
    if 'feature_stats' in checkpoint:
        queries = {s: prepare_feature_batch(q, checkpoint['audio_source'], checkpoint['feature_stats']) for s, q in queries.items()}
    refs = datasets['train'].identity_references
    if sorted(refs) != list(range(len(refs))) or len({len(v) for v in refs.values()}) != 1:
        raise ValueError('Enrollment IDs/counts invalid')
    reference = device_batch([r for s in sorted(refs) for r in refs[s]], device)
    with torch.no_grad():
        residual = torch.where(observed(reference), reference['motion'] - system.base(reference['content'], reference['valid'])['b0'], 0)
        identities = cached_identity(system, {'residual': residual.reshape(len(refs), len(refs[0]), *residual.shape[1:]),
                                             'valid': reference['valid'].reshape(len(refs), len(refs[0]), -1)})
        targets = {s: system.encode_motion(affect_residual(q, select(identities, q['speaker_id'])), q['valid']) for s, q in queries.items()}
        fixed_global = {s: {k: v.clone() for k, v in system.encode_audio(q['audio'], q['valid']).items() if k in GLOBAL_KEYS}
                        for s, q in queries.items()}
        probe = select(queries['heldout'], torch.arange(3, device=device))
        probe_base = select(bases['heldout'], torch.arange(3, device=device))
        probe_identity = select(identities, probe['speaker_id'])
        probe_affect = system.encode_audio(probe['audio'], probe['valid'])
        probe_affect['local'] = torch.zeros_like(probe_affect['local'])
        probe_noise = torch.randn(probe['motion'].shape, device=device, generator=torch.Generator(device=device).manual_seed(9281))
        zero_before = system.generate(probe['content'], probe['valid'], probe_identity, probe_affect, probe_noise, 12, base=probe_base)['motion']
    scales = {k: v.to(device) for k, v in checkpoint['scales'].items()}
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/'kinetalk_b0/models/neutral_affect.py', root/'kinetalk_b0/models/dit.py',
               root/'scripts/train_neutral_affect_pilot.py', root/'scripts/train_neutral_affect_feature_probe.py']
    provenance = {'schema': 'neutral_affect_local_interface_v1', 'mode': args.mode, 'config': effective,
                  'seed': args.seed, 'steps': args.steps, 'checkpoint_sha256': sha(args.checkpoint),
                  'train_sha256': sha(args.data/'train.pt'), 'heldout_sha256': sha(args.data/'heldout.pt'),
                  'initial_state_sha256': state_hash(system.state_dict()), 'initial_frozen_sha256': frozen_hash,
                  'trainable_names': sorted(allowed), 'trainable_parameters': sum(p.numel() for p in system.parameters() if p.requires_grad),
                  'audio_lr': args.audio_lr, 'interface_lr': args.interface_lr,
                  'condition_schedule': 'Each clip teacher local probability .5 -> .1; audio global/intensity always fixed; at least one audio clip per batch',
                  'objective': 'Existing flow + .1 displacement; no constant global/CE terms',
                  'source_sha256': {str(p.relative_to(root)): sha(p) for p in sources},
                  'scope': 'Previously used development sentences; not untouched test; zero-local function fixed, active-local mouth still requires evaluation'}
    write_json(args.output/'provenance.json', provenance)
    report = {'before': {s: summarize(system, q, targets[s], scales) for s, q in queries.items()}}
    groups = [{'params': [p for n, p in system.named_parameters() if p.requires_grad and n.startswith('audio_encoder.')], 'lr': args.audio_lr}]
    if args.mode != 'head':
        groups.append({'params': [system.renderer.local_emotion.weight], 'lr': args.interface_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    params = [p for p in system.parameters() if p.requires_grad]
    batches = torch.Generator().manual_seed(args.seed)
    flow_rng = torch.Generator(device=device).manual_seed(args.seed+1000003)
    batch_hash, flow_hash = hashlib.sha256(), hashlib.sha256()
    for step in range(args.steps):
        ids_cpu = torch.randint(len(datasets['train']), (int(cfg['training']['batch_size']),), generator=batches)
        batch_hash.update(ids_cpu.numpy().tobytes())
        ids = ids_cpu.to(device)
        q = select(queries['train'], ids)
        probability = .5 - .4 * step / max(args.steps - 1, 1)
        loss, parts, random_values = dynamic_objective(system, q, select(bases['train'], ids), select(identities, q['speaker_id']),
                                                      select(targets['train'], ids), flow_rng, probability)
        for value in random_values:
            flow_hash.update(value.detach().cpu().numpy().tobytes())
        norm = optimize(loss, optimizer, params)
        if step % 100 == 0 or step == args.steps - 1:
            if state_hash(frozen()) != frozen_hash:
                raise RuntimeError('Frozen parameter changed')
            record = {'step': step+1, 'time': time.time(), 'loss': float(loss), 'grad_norm': norm, **parts}
            print(record, flush=True)
    with torch.no_grad():
        global_unchanged = all(torch.equal(system.encode_audio(q['audio'], q['valid'])[k], fixed_global[s][k])
                               for s, q in queries.items() for k in GLOBAL_KEYS)
        zero_after = system.generate(probe['content'], probe['valid'], probe_identity, probe_affect, probe_noise, 12, base=probe_base)['motion']
    integrity = {'frozen_parameters_unchanged': state_hash(frozen()) == frozen_hash,
                 'global_outputs_bitwise_unchanged': global_unchanged,
                 'zero_local_generation_bitwise_unchanged': torch.equal(zero_before, zero_after)}
    if not all(integrity.values()):
        raise RuntimeError(f'Preservation contract failed: {integrity}')
    report.update(after={s: summarize(system, q, targets[s], scales) for s, q in queries.items()}, integrity=integrity,
                  minibatch_sha256=batch_hash.hexdigest(), flow_random_sha256=flow_hash.hexdigest())
    write_json(args.output/'summary.json', report)
    saved = {k: checkpoint[k] for k in ('scales', 'audio_stats', 'audio_source', 'feature_stats') if k in checkpoint}
    torch.save({**saved, 'model': system.state_dict(), 'config': effective, 'provenance': provenance, 'stage': 'audio',
                'steps': args.steps, 'optimizer': optimizer.state_dict()}, args.output/'audio.pt')


if __name__ == '__main__':
    main()
