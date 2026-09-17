"""Matched renderer adaptation with fixed versus task-adapting motion controls.

Both arms open the same audio local head/refiner and the entire residual DiT.
Only ``coupled`` also trains the motion control head through teacher-conditioned
flow. Audio/global emotion, motion/global emotion, identity and B0 stay fixed.
Unlike the smaller interface experiment, zero-local generation may change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.train_dynamic_interface import GLOBAL_KEYS, build_system
from scripts.train_neutral_affect_audio_ablation import state_hash, summarize
from scripts.train_neutral_affect_feature_probe import prepare_feature_batch
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, derivative_loss, device_batch, masked_mse,
    observed, prepare, select, sha, write_json,
)
from kinetalk_b0.dynamic_energy import fit_control_probe, projected_energy, upper_l1_field


def configure_trainable(system, mode):
    if mode not in ('renderer', 'coupled'):
        raise ValueError('mode must be renderer or coupled')
    freeze_module(system)
    prefixes = ('audio_encoder.control_head.', 'audio_encoder.control_refiner.', 'renderer.')
    if mode == 'coupled':
        prefixes += ('motion_teacher.control_head.',)
    allowed = {name for name, _ in system.named_parameters() if name.startswith(prefixes)}
    for name, parameter in system.named_parameters():
        parameter.requires_grad_(name in allowed)
    # Deterministic conditions in both arms; gradients do not require train().
    system.eval()
    return allowed


def teacher_probability(step, steps):
    return .5 if step < steps / 2 else .1


def coupled_objective(system, query, base, identity, scales, rng, probability,
                      energy_target=None, energy_probe=None, energy_weight=0.0):
    """Teacher targets are live, but alignment never pushes the teacher.

    Teacher controls receive only the flow/displacement gradient on selected
    teacher-local examples. The renderer always receives frozen audio global
    and intensity, so neither arm uses GT emotion labels at generation time.
    """
    audio = system.encode_audio(query['audio'], query['valid'])
    teacher = system.encode_motion(affect_residual(query, identity), query['valid'])
    size, device = len(query['motion']), query['motion'].device
    use_teacher = torch.rand(size, device=device, generator=rng) < probability
    if use_teacher.all():
        use_teacher[0] = False
    affect = {**audio, 'local': torch.where(use_teacher[:, None, None], teacher['local'], audio['local'])}
    times = torch.rand(size, device=device, generator=rng)
    times[torch.rand(size, device=device, generator=rng) < .2] = 0
    noise = torch.randn(query['motion'].shape, device=device, dtype=query['motion'].dtype, generator=rng)
    output = system.flow(query['motion'], query['content'], query['valid'], identity, affect,
                         noise=noise, time=times, base=base)
    flow = masked_mse(output['prediction'], output['velocity_target'], observed(query))
    displacement = derivative_loss(output['motion'], query['motion'], query) / system.residual_scale ** 2
    error = ((audio['controls'] - teacher['controls'].detach()) / scales['controls']).square()
    weight = teacher['control_weight'].detach().unsqueeze(-1)
    alignment = (error * weight).sum() / (weight.sum() * error.shape[-1]).clamp_min(1)
    loss = flow + .1 * displacement + .1 * alignment
    energy_alignment = torch.zeros((), device=loss.device, dtype=loss.dtype)
    if energy_target is not None or energy_probe is not None or energy_weight:
        if energy_target is None or energy_probe is None or energy_weight <= 0:
            raise ValueError('energy_target, energy_probe and positive energy_weight are required together')
        predicted_energy = projected_energy(audio['controls'], energy_probe)
        valid_energy = audio['control_weight'] > 0
        energy_alignment = (((predicted_energy - energy_target.squeeze(-1)).square()
                             * valid_energy.to(predicted_energy.dtype)).sum()
                            / valid_energy.sum().clamp_min(1))
        loss = loss + float(energy_weight) * energy_alignment
    parts = {'flow': float(flow.detach()), 'displacement': float(displacement.detach()),
             'control_alignment': float(alignment.detach()),
             'energy_alignment': float(energy_alignment.detach()),
             'teacher_fraction': float(use_teacher.float().mean())}
    return loss, parts, (use_teacher, times, noise)


def optimize_coupled(loss, optimizer, params, system, used_teacher):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite coupled training loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not used_teacher:
        # torch.where still produces a zero gradient on the unused branch.
        # None additionally prevents AdamW momentum/decay from moving a live
        # teacher on batches containing no teacher-conditioned flow examples.
        for parameter in system.motion_teacher.control_head.parameters():
            parameter.grad = None
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


@torch.no_grad()
def teacher_targets(system, queries, identities):
    return {split: system.encode_motion(affect_residual(query, select(identities, query['speaker_id'])), query['valid'])
            for split, query in queries.items()}


@torch.no_grad()
def global_snapshots(system, queries, identities):
    result = {}
    for split, query in queries.items():
        audio = system.encode_audio(query['audio'], query['valid'])
        teacher = system.encode_motion(affect_residual(query, select(identities, query['speaker_id'])), query['valid'])
        result[split] = {source: {key: values[key].clone() for key in GLOBAL_KEYS}
                         for source, values in (('audio', audio), ('teacher', teacher))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=('renderer', 'coupled'), required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--audio-lr', type=float, default=.0005)
    parser.add_argument('--teacher-lr', type=float, default=.0005)
    parser.add_argument('--renderer-lr', type=float, default=.0001)
    parser.add_argument('--energy-mode', choices=('none', 'upper_l1'), default='none',
                        help='Optional single upper-face expression-energy alignment probe')
    parser.add_argument('--energy-weight', type=float, default=.05)
    args = parser.parse_args()
    if args.steps < 1 or min(args.audio_lr, args.teacher_lr, args.renderer_lr) <= 0:
        raise ValueError('Positive steps and learning rates required')
    if args.energy_weight < 0:
        raise ValueError('energy-weight must be nonnegative')
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
    system, effective = build_system(cfg, checkpoint, True, args.seed)
    device = torch.device(cfg.get('device', 'cuda'))
    system.to(device)
    allowed = configure_trainable(system, args.mode)
    frozen = lambda: {name: value for name, value in system.state_dict().items() if name not in allowed}
    frozen_hash = state_hash(frozen())
    initial_teacher_hash = state_hash(system.motion_teacher.control_head.state_dict())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'effective_config.yaml').write_text(yaml.safe_dump(effective, sort_keys=False), encoding='utf8')
    datasets = {split: NeutralAffectDataset(args.data / (split + '.pt')) for split in ('train', 'heldout')}
    prepared = {split: prepare(system, dataset, device) for split, dataset in datasets.items()}
    queries, bases = {split: item[0] for split, item in prepared.items()}, {split: item[1] for split, item in prepared.items()}
    if 'feature_stats' in checkpoint:
        queries = {split: prepare_feature_batch(query, checkpoint['audio_source'], checkpoint['feature_stats'])
                   for split, query in queries.items()}
    refs = datasets['train'].identity_references
    if sorted(refs) != list(range(len(refs))) or len({len(values) for values in refs.values()}) != 1:
        raise ValueError('Enrollment IDs/counts invalid')
    reference = device_batch([ref for speaker in sorted(refs) for ref in refs[speaker]], device)
    with torch.no_grad():
        residual = torch.where(observed(reference), reference['motion'] - system.base(reference['content'], reference['valid'])['b0'], 0)
        identities = cached_identity(system, {'residual': residual.reshape(len(refs), len(refs[0]), *residual.shape[1:]),
                                             'valid': reference['valid'].reshape(len(refs), len(refs[0]), -1)})
        fixed_global = global_snapshots(system, queries, identities)
        probe_ids = torch.arange(min(3, len(datasets['heldout'])), device=device)
        probe, probe_base = select(queries['heldout'], probe_ids), select(bases['heldout'], probe_ids)
        probe_identity = select(identities, probe['speaker_id'])
        probe_affect = system.encode_audio(probe['audio'], probe['valid'])
        probe_affect['local'] = torch.zeros_like(probe_affect['local'])
        probe_noise = torch.randn(probe['motion'].shape, device=device, generator=torch.Generator(device=device).manual_seed(9281))
        zero_before = system.generate(probe['content'], probe['valid'], probe_identity, probe_affect, probe_noise, 12, base=probe_base)['motion']
    energy_probe = None
    energy_targets = None
    if args.energy_mode == 'upper_l1':
        with torch.no_grad():
            initial_teacher = teacher_targets(system, queries, identities)
            # The target comes from motion residuals; the fitted probe maps the
            # existing teacher controls to this one interpretable field.
            residuals = {split: affect_residual(query, select(identities, query['speaker_id']))
                         for split, query in queries.items()}
            fields = {split: upper_l1_field(residuals[split], queries[split]['valid'], system.motion_teacher.stride,
                                            queries[split].get('channel_mask'))
                      for split in queries}
            energy_probe = fit_control_probe(initial_teacher['train']['controls'], fields['train'][0],
                                              fields['train'][1])
            energy_targets = {split: fields[split][0].detach() for split in queries}
    scales = {key: value.to(device) for key, value in checkpoint['scales'].items()}
    if any(not torch.isfinite(scales[key]).all() or not (scales[key] > 0).all() for key in ('global', 'controls')):
        raise ValueError('Checkpoint scales must be finite and positive')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/'scripts/train_dynamic_interface.py', root/'kinetalk_b0/models/neutral_affect.py',
               root/'kinetalk_b0/models/dit.py', root/'scripts/train_neutral_affect_pilot.py',
               root/'scripts/train_neutral_affect_feature_probe.py', root/'scripts/train_neutral_affect_audio_ablation.py']
    provenance = {
        'schema': 'neutral_affect_coupled_v1', 'mode': args.mode, 'config': effective,
        'seed': args.seed, 'steps': args.steps, 'checkpoint_sha256': sha(args.checkpoint),
        'train_sha256': sha(args.data/'train.pt'), 'heldout_sha256': sha(args.data/'heldout.pt'),
        'initial_state_sha256': state_hash(system.state_dict()), 'initial_frozen_sha256': frozen_hash,
        'trainable_names': sorted(allowed), 'trainable_parameters': sum(p.numel() for p in system.parameters() if p.requires_grad),
        'audio_lr': args.audio_lr, 'teacher_lr': args.teacher_lr, 'renderer_lr': args.renderer_lr,
        'condition_schedule': 'Teacher local probability .5 for first half, .1 for second half; at least one audio clip per batch; always frozen audio global/intensity',
        'objective': 'Both arms: flow + .1 displacement + .1 controls alignment; optional upper_l1 energy probe; alignment target detached; teacher flow only updates coupled teacher head',
        'energy_mode': args.energy_mode, 'energy_weight': args.energy_weight,
        'normalization': 'Original checkpoint scales/feature_stats kept fixed; teacher targets refreshed each batch',
        'source_sha256': {str(path.relative_to(root)): sha(path) for path in sources},
        'scope': 'Previously used development sentences; entire residual renderer trainable; zero-local generation may change and jaw/global-expression quality needs evaluation',
    }
    write_json(args.output/'provenance.json', provenance)
    initial_targets = teacher_targets(system, queries, identities)
    report = {'before': {split: summarize(system, query, initial_targets[split], scales) for split, query in queries.items()}}
    groups = [
        {'params': [p for name, p in system.named_parameters() if p.requires_grad and name.startswith('audio_encoder.')], 'lr': args.audio_lr},
        {'params': list(system.renderer.parameters()), 'lr': args.renderer_lr},
    ]
    if args.mode == 'coupled':
        groups.append({'params': list(system.motion_teacher.control_head.parameters()), 'lr': args.teacher_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    params = [parameter for parameter in system.parameters() if parameter.requires_grad]
    batches = torch.Generator().manual_seed(args.seed)
    flow_rng = torch.Generator(device=device).manual_seed(args.seed + 1000003)
    batch_hash, flow_hash = hashlib.sha256(), hashlib.sha256()
    for step in range(args.steps):
        ids_cpu = torch.randint(len(datasets['train']), (int(cfg['training']['batch_size']),), generator=batches)
        batch_hash.update(ids_cpu.numpy().tobytes())
        ids = ids_cpu.to(device)
        query = select(queries['train'], ids)
        loss, parts, random_values = coupled_objective(
            system, query, select(bases['train'], ids), select(identities, query['speaker_id']), scales,
            flow_rng, teacher_probability(step, args.steps),
            energy_target=(energy_targets['train'][ids] if energy_targets is not None else None),
            energy_probe=energy_probe, energy_weight=(args.energy_weight if energy_targets is not None else 0.0))
        for value in random_values:
            flow_hash.update(value.detach().cpu().numpy().tobytes())
        norm = optimize_coupled(loss, optimizer, params, system, bool(random_values[0].any()))
        if step % 100 == 0 or step == args.steps - 1:
            if state_hash(frozen()) != frozen_hash or any(p.grad is not None for name, p in system.named_parameters() if name not in allowed):
                raise RuntimeError('Frozen parameter changed or received gradient')
            record = {'step': step + 1, 'time': time.time(), 'loss': float(loss), 'grad_norm': norm, **parts}
            with (args.output/'training.jsonl').open('a', encoding='utf8') as handle:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)
    with torch.no_grad():
        final_global = global_snapshots(system, queries, identities)
        global_same = {source: all(torch.equal(fixed_global[split][source][key], final_global[split][source][key])
                                   for split in queries for key in GLOBAL_KEYS) for source in ('audio', 'teacher')}
        zero_after = system.generate(probe['content'], probe['valid'], probe_identity, probe_affect, probe_noise, 12, base=probe_base)['motion']
    integrity = {'frozen_parameters_unchanged': state_hash(frozen()) == frozen_hash,
                 'audio_global_outputs_bitwise_unchanged': global_same['audio'],
                 'teacher_global_outputs_bitwise_unchanged': global_same['teacher']}
    if not all(integrity.values()):
        raise RuntimeError(f'Preservation contract failed: {integrity}')
    if args.mode == 'renderer' and state_hash(system.motion_teacher.control_head.state_dict()) != initial_teacher_hash:
        raise RuntimeError('Fixed teacher head changed')
    final_targets = teacher_targets(system, queries, identities)
    report.update(
        after={split: summarize(system, query, final_targets[split], scales) for split, query in queries.items()},
        after_against_initial_teacher={split: summarize(system, query, initial_targets[split], scales) for split, query in queries.items()},
        integrity=integrity, teacher_head_changed=state_hash(system.motion_teacher.control_head.state_dict()) != initial_teacher_hash,
        zero_local_probe={'bitwise_unchanged': torch.equal(zero_before, zero_after),
                          'max_abs_change': float((zero_after-zero_before).abs().max()),
                          'mse_change': float(masked_mse(zero_after, zero_before, observed(probe)))},
        minibatch_sha256=batch_hash.hexdigest(), flow_random_sha256=flow_hash.hexdigest())
    write_json(args.output/'summary.json', report)
    saved = {key: checkpoint[key] for key in ('scales', 'audio_stats', 'audio_source', 'feature_stats') if key in checkpoint}
    torch.save({**saved, 'model': system.state_dict(), 'config': effective, 'provenance': provenance, 'stage': 'audio',
                'steps': args.steps, 'optimizer': optimizer.state_dict(), 'minibatch_sha256': batch_hash.hexdigest(),
                'flow_random_sha256': flow_hash.hexdigest()}, args.output/'audio.pt')


if __name__ == '__main__':
    main()
