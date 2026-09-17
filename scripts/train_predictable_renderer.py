"""Isolated fixed-budget generator test for a frozen predictable-motion basis.

Only the existing renderer/local projection and one bias-free audio linear
head train. B0, neutral identity, audio global affect and the previous motion
teacher stay frozen. Real motion is used only for training/oracle local
controls; every generated condition retains the same cached audio global code.
This is a development experiment, not a default model or a publication claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from torch import nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.emotion_ray import NUISANCE_CHANNELS_52
from kinetalk_b0.utils import freeze_module
from scripts.emotion_ray_metrics import field_metrics
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import observed, optimize, write_json


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def center(values, weight):
    """Differentiable masked centering; never detaches the audio prediction."""
    valid = weight > 0
    clean = torch.where(valid[..., None], values, 0)
    mean = (clean * weight[..., None]).sum(1, keepdim=True) / weight.sum(1)[:, None, None].clamp_min(1)
    return torch.where(valid[..., None], clean - mean, 0)


class PredictableAudioHead(nn.Module):
    """Trainable linear controls in a fixed motion basis and train-only scale."""
    def __init__(self, state, train_motion, train_weight):
        super().__init__()
        basis = state['basis'].detach().double().cpu()
        channels = torch.as_tensor(state['motion_channel_indices'], dtype=torch.long)
        target = center(train_motion.double()[..., channels], train_weight.double()) @ basis
        scale = ((target.square() * train_weight.double()[..., None]).sum((0, 1)) /
                 train_weight.double().sum()).sqrt().clamp_min(1e-6)
        self.register_buffer('basis', basis.float())
        self.register_buffer('channels', channels)
        self.register_buffer('feature_std', state['std'].detach().float().cpu())
        self.register_buffer('target_scale', scale.float())
        self.linear = nn.Linear(len(self.feature_std), basis.shape[1], bias=False)
        with torch.no_grad():
            self.linear.weight.copy_((state['weights'].double() / scale).T.float())

    def forward(self, features, weight):
        return center(self.linear(center(features, weight) / self.feature_std), weight)

    def teacher(self, motion_bins, weight):
        return center(motion_bins[..., self.channels], weight) @ self.basis / self.target_scale


def configure_trainable(system, *, seed, zero=False, projection_only=False):
    freeze_module(system)
    # Local coordinates changed. Reinitialize only their shared projection,
    # identically across arms; every renderer starts at the same checkpoint.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        weight = torch.empty(system.local_projection.weight.shape)
        nn.init.kaiming_uniform_(weight, a=5 ** .5)
        if projection_only:
            weight.zero_()
    with torch.no_grad():
        system.local_projection.weight.copy_(weight)
    system.renderer.requires_grad_(not projection_only)
    system.local_projection.requires_grad_(not zero)
    # Eval keeps frozen modules and condition dropout deterministic; gradients
    # are still enabled on the two deliberately trainable modules.
    system.eval()


def frozen_state(system):
    return {k: v for k, v in system.state_dict().items()
            if not k.startswith(('renderer.', 'local_projection.'))}


def batch_to_device(split, ids, device):
    result = {}
    for group in ('q', 'base', 'identity', 'affect'):
        result[group] = {}
        for key, value in split[group].items():
            if torch.is_tensor(value):
                sliced = value[ids].to(device)
                # Cached content may be half, but the cached B0/h0 must be
                # retained exactly: never invoke system.base on this content.
                result[group][key] = sliced.float() if sliced.is_floating_point() and key != 'times' else sliced
            elif isinstance(value, (list, tuple)):
                result[group][key] = [value[int(i)] for i in ids]
    return result


def audio_activity_gate(affect):
    """One shared scalar from frozen audio probabilities; never reads labels."""
    logits = affect['emotion_logits']
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('Finite [batch,emotion] audio logits required')
    return (1 - logits.detach().softmax(-1)[:, 0]).clamp(0, 1)


def projected_affect(system, batch, controls, weight, *, zero=False, audio_gate=False):
    cached = batch['affect']
    if zero:
        return {**cached, 'local': torch.zeros_like(cached['local'])}
    if audio_gate:
        controls = controls * audio_activity_gate(cached)[:, None, None]
    raw = {**cached, 'controls': controls, 'control_mask': weight > 0, 'control_weight': weight}
    return system.project_affect(raw, batch['q']['valid'])


def cached_flow(system, batch, controls, weight, noise, flow_time, *, zero=False, audio_gate=False):
    q = batch['q']
    affect = projected_affect(system, batch, controls, weight, zero=zero, audio_gate=audio_gate)
    return system.flow(q['motion'], q['content'], q['valid'], batch['identity'], affect,
                       noise=noise, time=flow_time, base=batch['base'])


def reverse_controls(controls, weight):
    result = torch.zeros_like(controls)
    for row in range(len(controls)):
        ids = (weight[row] > 0).nonzero(as_tuple=True)[0]
        result[row, ids] = controls[row, ids.flip(0)]
    return center(result, weight)


def audio_features(bundle):
    f = bundle['features']
    return torch.cat([f['content'], f['middle'], f['prosody']], -1).float()


def basic_metrics(prediction, split):
    """Native curves and sentence SSE; no lag search or tuned calibration."""
    q, base = split['q'], split['base']['b0']
    target, valid, cm = q['motion'].float(), q['valid'], q['channel_mask']
    if not torch.equal(cm, cm[:1].expand_as(cm)):
        raise ValueError('This compact scorer requires the audited common observed layout')
    groups = {'all_expression': [c for c in range(target.shape[-1]) if cm[0, c] and c not in NUISANCE_CHANNELS_52],
              'upper_expression': [5, 6, 12, 13, 41, 42, 43, 44, 45],
              'brows': list(range(41, 46)), 'eyes_expression': [5, 6, 12, 13],
              'mouth': list(range(14, 41)), 'jaw17': [17]}
    groups = {k: [c for c in v if c < target.shape[-1] and cm[0, c]] for k, v in groups.items()}
    baseline = base.float() + split['identity']['baseline'].float()[:, None]
    pred_residual = center(prediction - baseline, valid.float())
    target_residual = center(target - baseline, valid.float())
    report = {}
    for population, ids in {
        'all': torch.arange(len(target)),
        'nonneutral': (q['emotion_id'] != 0).nonzero(as_tuple=True)[0],
        'neutral': (q['emotion_id'] == 0).nonzero(as_tuple=True)[0],
    }.items():
        if not len(ids):
            report[population] = None
            continue
        sentences = [q['sentence_id'][int(i)] for i in ids]
        part = {}
        for name, channels in groups.items():
            if not channels:
                continue
            raw = field_metrics(prediction[ids][..., channels], target[ids][..., channels],
                                valid[ids].float(), sentences, bootstrap_samples=0)
            dynamic = field_metrics(pred_residual[ids][..., channels], target_residual[ids][..., channels],
                                    valid[ids].float(), sentences, bootstrap_samples=0)
            adjacent = valid[ids, 1:] & valid[ids, :-1]
            dt = q['times'][ids, 1:] - q['times'][ids, :-1]
            if not (dt[adjacent] > 0).all():
                raise ValueError('Timestamps must increase on valid adjacent frames')
            error = torch.diff(prediction[ids][..., channels] - target[ids][..., channels], dim=1)
            velocity = error / dt.clamp_min(1e-9)[..., None]
            velocity_mse = float(velocity[adjacent].square().mean()) if adjacent.any() else None
            part[name] = {'raw_motion': raw, 'centered_residual': dynamic, 'velocity_mse_per_second': velocity_mse}
        report[population] = part
    return report


@torch.no_grad()
def evaluate(system, head, split, bundle, output, name, *, zero_arm, original, device,
             seeds=(42, 123, 2026), batch_size=32, steps=12, audio_gate=False):
    system.eval()
    features, weight = audio_features(bundle), bundle['weight'].float()
    modes = ('full',) if original or zero_arm else ('full', 'zero', 'reverse', 'oracle')
    results = {}
    for seed in seeds:
        noise = torch.randn(split['q']['motion'].shape, generator=torch.Generator().manual_seed(seed))
        curves = {mode: [] for mode in modes}
        emotion_logits = {mode: [] for mode in modes}
        for start in range(0, len(features), batch_size):
            ids = torch.arange(start, min(start + batch_size, len(features)))
            batch = batch_to_device(split, ids, device)
            q = batch['q']
            w = weight[ids].to(device)
            controls = head(features[ids].to(device), w) if head is not None else None
            for mode in modes:
                if original:
                    affect = batch['affect']
                else:
                    z = controls
                    if mode == 'reverse':
                        z = reverse_controls(z, w)
                    elif mode == 'oracle':
                        z = head.teacher(bundle['motion_bins'][ids].float().to(device), w)
                    affect = projected_affect(system, batch, z, w, zero=zero_arm or mode == 'zero', audio_gate=audio_gate)
                pred = system.generate(q['content'], q['valid'], batch['identity'], affect,
                                       initial_noise=noise[ids].to(device), steps=steps, base=batch['base'])['motion']
                curves[mode].append(pred.cpu())
                residual = torch.where(observed(q), pred - batch['base']['b0'] - batch['identity']['baseline'][:, None], 0)
                emotion_logits[mode].append(system.motion_teacher(residual, q['valid'])['emotion_logits'].cpu())
        curves = {mode: torch.cat(values) for mode, values in curves.items()}
        report = {}
        for mode, prediction in curves.items():
            report[mode] = basic_metrics(prediction, split)
            classes = torch.cat(emotion_logits[mode]).argmax(-1)
            report[mode]['frozen_teacher_emotion_accuracy'] = float((classes == split['q']['emotion_id']).float().mean())
            report[mode]['emotion_measurement_note'] = 'Same frozen training motion teacher, not an independent emotion-quality evaluator.'
            report[mode]['change_from_full_mse'] = float((prediction - curves['full'])[observed(split['q'])].square().mean())
        torch.save({'motion': curves, 'noise_seed': seed, 'decode_steps': steps}, output / f'{name}_seed{seed}_curves.pt')
        write_json(output / f'{name}_seed{seed}.json', report)
        results[str(seed)] = report
        print(json.dumps({'stage': 'evaluation', 'name': name, 'seed': seed, 'modes': modes}), flush=True)
    return results


def validate_inputs(cache, bundle, states, checkpoint, config):
    if cache.get('schema') != 'predictable_renderer_cache_v1':
        raise ValueError('Wrong renderer cache schema')
    original = checkpoint.get('config', checkpoint.get('provenance', {}).get('config', {}))
    if any(original.get(k) != config.get(k) for k in ('data', 'model')):
        raise ValueError('Checkpoint and config data/model differ')
    if len(bundle['heldout_ids']) or not torch.equal(bundle['train_ids'], torch.arange(len(bundle['bundles']['internal']['weight']))):
        raise ValueError('This expanded experiment requires all locked training IDs and no internal heldout')
    for split, source in (('train', 'internal'), ('validation', 'external_dev')):
        q, b = cache['splits'][split]['q'], bundle['bundles'][source]
        if list(map(str, q['clip_id'])) != list(map(str, b['clip_id'])):
            raise ValueError('Cache/bundle clip order mismatch')
        if not torch.equal(q['channel_mask'], b['channel_mask']):
            raise ValueError('Cache/bundle observed channel mismatch')
        stride = int(config['model'].get('affect_stride', 4))
        from kinetalk_b0.predictable_motion import bin_centered_frames
        cached = cache['splits'][split]
        residual = torch.where(observed(q), q['motion'] - cached['base']['b0'] - cached['identity']['baseline'][:, None], 0)
        bins, weight = bin_centered_frames(residual, q['valid'], stride)
        if not torch.equal(weight.float(), b['weight'].float()):
            raise ValueError('Cache/bundle bin weights differ')
        channels = states['rrr_rank8']['motion_channel_indices']
        torch.testing.assert_close(bins[..., channels].float(), b['motion_bins'][..., channels].float(), rtol=2e-5, atol=2e-6)
    for key in ('rrr_rank8', 'pca_rank8'):
        state = states[key]
        if state['rank'] != 8 or not torch.equal(state['train_ids'], bundle['train_ids']):
            raise ValueError('Must use fixed rank8 train-only basis')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'bundle', 'weights', 'checkpoint', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--mode', choices=('rrr', 'pca', 'zero'), required=True)
    parser.add_argument('--projection-only', action='store_true',
                        help='Freeze the validated audio head and entire renderer; train zero-initialized local projection only')
    parser.add_argument('--audio-activity-gate', action='store_true',
                        help='Scale all shared local controls by frozen audio nonneutral probability')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('Positive steps, batch size and learning rate required')
    if args.projection_only and args.mode == 'zero':
        raise ValueError('Projection-only zero arm has no trainable parameters')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = yaml.safe_load(args.config.read_text(encoding='utf8'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=False)
    fitted = torch.load(args.weights, map_location='cpu', weights_only=False)
    validate_inputs(cache, bundle, fitted['states'], checkpoint, cfg)
    hashes = {k: sha(getattr(args, k)) for k in ('cache', 'bundle', 'weights', 'checkpoint', 'config')}
    for name in ('checkpoint', 'config', 'bundle'):
        if cache['provenance'].get(name + '_sha256') != hashes[name]:
            raise ValueError(f'Renderer cache {name} provenance mismatch')
    if fitted['provenance'].get('bundle_sha256') != hashes['bundle']:
        raise ValueError('Fixed-basis weights came from a different motion bundle')
    if cfg['data']['emotion_classes'][0] != 'neutral':
        raise ValueError('Activity gate requires audited neutral index zero')
    system = NeutralAffectSystem(cfg).to(args.device).eval()
    system.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    freeze_module(system)
    frozen_before = state_hash(frozen_state(system))
    source_paths = [Path(__file__), Path(__file__).parents[1] / 'kinetalk_b0/models/neutral_affect.py',
                    Path(__file__).parents[1] / 'kinetalk_b0/models/dit.py', Path(__file__).with_name('emotion_ray_metrics.py')]
    provenance = {'schema': 'predictable_renderer_experiment_v1', 'mode': args.mode,
                  'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'hashes': hashes,
                  'source_sha256': {str(p): sha(p) for p in source_paths}, 'frozen_before': frozen_before,
                  'selection': 'Fixed rank8, fixed 600-step default budget, final checkpoint only; no development selection',
                  'loss': 'observed flow MSE + 0.1 weighted train-RMS-normalized control MSE (zero arm: flow only)',
                  'optimizer': 'Adam, no weight decay; existing ridge regularization is initialization only',
                  'teacher_schedule': 'per-item probability 0.5 to 0 linearly in first half; second half all audio',
                  'global_condition': 'original frozen audio global/intensity cache in every training/evaluation condition',
                  'scope': 'Development generation functionality test; validation previously inspected and pretrained exposure possible',
                  'new_test_loaded': False, 'cached_base_required': True, 'torch': torch.__version__}
    provenance['audio_activity_gate'] = '1-softmax(frozen audio emotion_logits)[neutral], applied identically to predicted and oracle controls' if args.audio_activity_gate else 'disabled'
    if args.projection_only:
        provenance.update(loss='Observed flow MSE only; fixed audio and true-motion controls, no constant alignment objective',
            adaptation='Only zero-initialized local_projection trains. Entire renderer and audio dynamic head frozen; zero-local generation preserves loaded zero-local function.')
    write_json(args.output / 'provenance.json', provenance)
    (args.output / 'source').mkdir()
    for path in source_paths:
        shutil.copyfile(path, args.output / 'source' / path.name)
    validation = cache['splits']['validation']
    # Save target/reference once per run, not once per noise seed/condition.
    torch.save({'q': {k: v for k, v in validation['q'].items() if k != 'content'},
                'base': {'b0': validation['base']['b0']}, 'identity': validation['identity']},
               args.output / 'validation_reference.pt')
    before = evaluate(system, None, validation, bundle['bundles']['external_dev'], args.output, 'original',
                      zero_arm=False, original=True, device=args.device)
    state = fitted['states']['pca_rank8' if args.mode == 'pca' else 'rrr_rank8']
    tr = bundle['bundles']['internal']
    head = PredictableAudioHead(state, tr['motion_bins'], tr['weight']).to(args.device)
    zero = args.mode == 'zero'
    configure_trainable(system, seed=args.seed, zero=zero, projection_only=args.projection_only)
    head.requires_grad_(not zero and not args.projection_only)
    renderer_before = state_hash(system.renderer.state_dict())
    head_before = state_hash(head.state_dict())
    params = [p for p in system.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    features, weight = audio_features(tr), tr['weight'].float()
    teacher = head.teacher(tr['motion_bins'].float().to(args.device), weight.to(args.device)).detach().cpu()
    rng = torch.Generator().manual_seed(args.seed)
    batch_hash, noise_hash = hashlib.sha256(), hashlib.sha256()
    shape = cache['splits']['train']['q']['motion'].shape[1:]
    started = time.time()
    for step in range(args.steps):
        ids = torch.randint(len(features), (args.batch_size,), generator=rng)
        noise = torch.randn((len(ids), *shape), generator=rng)
        flow_time = torch.rand(len(ids), generator=rng)
        at_zero = torch.rand(len(ids), generator=rng) < .2
        flow_time[at_zero] = 0
        choose_teacher = torch.rand(len(ids), generator=rng)
        batch_hash.update(ids.numpy().tobytes())
        for value in (noise, flow_time, choose_teacher):
            noise_hash.update(value.numpy().tobytes())
        b = batch_to_device(cache['splits']['train'], ids, args.device)
        w = weight[ids].to(args.device)
        audio = head(features[ids].to(args.device), w)
        truth = teacher[ids].to(args.device)
        probability = .5 * max(0., 1. - step / max(args.steps / 2 - 1, 1))
        use_teacher = (choose_teacher.to(args.device) < probability)[:, None, None]
        controls = torch.where(use_teacher, truth, audio)
        result = cached_flow(system, b, controls, w, noise.to(args.device), flow_time.to(args.device), zero=zero, audio_gate=args.audio_activity_gate)
        flow_loss = (result['prediction'] - result['velocity_target'])[observed(b['q'])].square().mean()
        alignment = ((audio - truth).square() * w[..., None]).sum() / (w.sum() * audio.shape[-1]).clamp_min(1)
        loss = flow_loss if zero or args.projection_only else flow_loss + .1 * alignment
        norm = optimize(loss, optimizer, params)
        record = {'step': step + 1, 'loss': float(loss), 'flow': float(flow_loss),
                  'alignment': None if zero or args.projection_only else float(alignment), 'teacher_probability': probability,
                  'teacher_fraction': 0. if zero else float(use_teacher.float().mean()),
                  'time_zero_fraction': float(at_zero.float().mean()), 'grad_norm': norm,
                  'elapsed_seconds': time.time() - started}
        with (args.output / 'training.jsonl').open('a', encoding='utf8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        if step % 100 == 0 or step + 1 == args.steps:
            print(json.dumps(record, allow_nan=False), flush=True)
    frozen_after = state_hash(frozen_state(system))
    if frozen_after != frozen_before:
        raise RuntimeError('Frozen B0/identity/audio-global/motion-teacher parameters changed')
    if args.projection_only and (state_hash(system.renderer.state_dict()) != renderer_before or state_hash(head.state_dict()) != head_before):
        raise RuntimeError('Projection-only frozen renderer/head changed')
    after = evaluate(system, head, validation, bundle['bundles']['external_dev'], args.output, 'final',
                     zero_arm=zero, original=False, device=args.device, audio_gate=args.audio_activity_gate)
    report = {'provenance': provenance, 'before': before, 'after': after,
              'frozen_after': frozen_after, 'frozen_unchanged': True,
              'minibatch_sha256': batch_hash.hexdigest(), 'noise_time_choice_sha256': noise_hash.hexdigest(),
              'final_rng_sha256': state_hash({'rng': rng.get_state()}),
              'trainable_system_parameters': [name for name, p in system.named_parameters() if p.requires_grad],
              'basis_method': state['method'], 'basis_rank': state['rank'], 'basis_alpha': state['alpha'],
              'adaptation': 'projection_only' if args.projection_only else 'joint',
              'renderer_unchanged': state_hash(system.renderer.state_dict()) == renderer_before,
              'head_unchanged': state_hash(head.state_dict()) == head_before,
              'new_test_loaded': False}
    write_json(args.output / 'summary.json', report)
    torch.save({'schema': 'predictable_renderer_delta_v1', 'renderer': {k: v.detach().cpu() for k, v in system.renderer.state_dict().items()},
                'local_projection': {k: v.detach().cpu() for k, v in system.local_projection.state_dict().items()},
                'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                'mode': args.mode, 'provenance': provenance, 'steps': args.steps,
                'minibatch_sha256': batch_hash.hexdigest(), 'noise_time_choice_sha256': noise_hash.hexdigest()},
               args.output / 'renderer_delta.pt')
    print('COMPLETE: isolated renderer experiment; default checkpoints unchanged', flush=True)


if __name__ == '__main__':
    main()
