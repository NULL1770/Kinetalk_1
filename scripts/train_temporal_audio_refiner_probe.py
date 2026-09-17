"""Fixed eight-epoch, train-internal OOF audio-only residual refiner comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.predictable_motion import predict_motion, weighted_clip_center
from kinetalk_b0.temporal_motion_refiner import FrozenRidgeRefiner, fit_control_scale
from scripts.probe_scaled_motion_basis import training_inputs, motion_groups, write_json
from scripts.probe_predictable_motion import intervene_input
from scripts.train_predictable_renderer import audio_features, sha, state_hash
from scripts.audit_predictable_motion_predictions import clip_statistics, scalar_summary

ARMS = ('pointwise', 'temporal')
SCHEMA = 'temporal_audio_refiner_oof_v1'


def native_loss(prediction, target, weight):
    if prediction.shape != target.shape or weight.shape != target.shape[:2]:
        raise ValueError('Prediction/target/weight shape mismatch')
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('Weights must be finite and nonnegative')
    valid = weight > 0
    error = torch.where(valid[..., None], prediction - target, 0)
    if not torch.isfinite(error).all() or not weight.sum() > 0:
        raise ValueError('Nonfinite observed error or empty training batch')
    return (error.square() * weight[..., None]).sum() / (weight.sum() * target.shape[-1] * .25 ** 2)


def matched_models(state, scale, seed):
    """Exact common parameters; temporal side taps start zero, center matches."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        pointwise = FrozenRidgeRefiner(state, scale, 'pointwise')
        torch.manual_seed(seed)
        temporal = FrozenRidgeRefiner(state, scale, 'temporal')
    a, b = pointwise.state_dict(), temporal.state_dict()
    copied, kernels = [], []
    for key in b:
        if key not in a:
            raise ValueError(f'Unmatched parameter/buffer {key}')
        if a[key].shape == b[key].shape:
            b[key].copy_(a[key]); copied.append(key)
        elif a[key].ndim == 3 and a[key].shape[-1] == 1 and b[key].shape == (*a[key].shape[:-1], 3):
            b[key].zero_(); b[key][..., 1:2].copy_(a[key]); kernels.append(key)
        else:
            raise ValueError(f'Unexpected unmatched shape {key}')
    temporal.load_state_dict(b, strict=True)
    if len(kernels) != 2:
        raise ValueError('Exactly two temporal kernels required')
    return {'pointwise': pointwise, 'temporal': temporal}, {'shared_state_keys': copied,
        'center_matched_zero_side_kernels': kernels,
        'parameter_counts': {name: sum(p.numel() for p in model.parameters())
                             for name, model in [('pointwise', pointwise), ('temporal', temporal)]}}


def buffer_hash(model):
    return state_hash(dict(model.named_buffers()))


def atomic_save(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


@torch.no_grad()
def predict_batches(model, x, weight, device, batch_size=64):
    model.eval()
    return torch.cat([model(x[s:s + batch_size].to(device), weight[s:s + batch_size].to(device)).cpu()
                      for s in range(0, len(x), batch_size)])


def train_arm(model, x, y, weight, fit_ids, *, seed, device, output, provenance, fold, arm):
    """Index fit before centering/finite checks or using target values."""
    model = model.to(device)
    xf = x.index_select(0, fit_ids).to(device)
    wf = weight.index_select(0, fit_ids).double().to(device)
    yf = weighted_clip_center(y.index_select(0, fit_ids), wf.cpu()).to(device)
    with torch.no_grad():
        s = {'std': model.feature_std.cpu(), 'basis': model.basis.cpu(),
             'weights': model.ridge_weights.cpu()}
        expected = predict_motion(xf[:8].cpu(), wf[:8].cpu(), s)
        torch.testing.assert_close(model(xf[:8], wf[:8]).cpu(), expected, rtol=1e-8, atol=1e-10)
    before = buffer_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    generator = torch.Generator().manual_seed(seed)
    batch_hash = hashlib.sha256()
    output.mkdir(parents=True, exist_ok=False)
    records, step = [], 0
    started = time.time()
    for epoch in range(1, 9):
        model.train()
        order = torch.randperm(len(fit_ids), generator=generator)
        batch_hash.update(fit_ids[order].numpy().tobytes())
        total, denominator = 0., 0.
        for ids in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xf[ids], wf[ids])
            loss = native_loss(prediction, yf[ids], wf[ids])
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite trainable gradients')
            optimizer.step()
            count = float(wf[ids].sum()) * y.shape[-1]
            total += float(loss.detach()) * count
            denominator += count
            step += 1
        if buffer_hash(model) != before:
            raise ValueError('Frozen ridge/basis/std/scale changed')
        row = {'fold': fold, 'arm': arm, 'epoch': epoch, 'step': step,
               'online_training_native_mse': total / denominator * .25 ** 2,
               'batch_order_sha256': batch_hash.hexdigest(), 'elapsed_seconds': time.time() - started}
        records.append(row)
        atomic_save(output / 'last.pt', {'schema': SCHEMA, 'provenance': provenance,
            'fold': fold, 'arm': arm, 'epoch': epoch, 'step': step,
            'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'generator_rng': generator.get_state(),
            'torch_rng': torch.random.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
            'frozen_buffer_sha256': before, 'fit_ids': fit_ids,
            'batch_order_sha256': batch_hash.hexdigest(), 'history': records,
            'checkpoint_selection': 'fixed epoch8 only, no held-out selection'})
        write_json(output / 'history.json', records)
        print(json.dumps(row), flush=True)
    return model, records


def load_sources(bundle_path, weights_path, basis_dir):
    binding = json.loads((basis_dir / 'output_hashes.json').read_text())
    for name, digest in binding.items():
        if sha(basis_dir / name) != digest:
            raise ValueError(f'Prior OOF artifact changed: {name}')
    previous = torch.load(basis_dir / 'oof_predictions.pt', map_location='cpu', weights_only=False, mmap=True)
    saved = torch.load(basis_dir / 'fold_states.pt', map_location='cpu', weights_only=False, mmap=True)
    reference = saved['provenance']
    hashes = {'bundle': sha(bundle_path), 'weights': sha(weights_path)}
    if reference['input_sha256'] != hashes or previous['provenance'] != reference:
        raise ValueError('Prior OOF/bundle/state provenance changed')
    fitted = torch.load(weights_path, map_location='cpu', weights_only=False, mmap=True)
    source = torch.load(bundle_path, map_location='cpu', weights_only=False, mmap=True)
    if fitted['provenance']['bundle_sha256'] != hashes['bundle']:
        raise ValueError('Weights bundle binding changed')
    bundle, channels, speakers, folds = training_inputs(source, fitted)
    if previous['clip_id'] != bundle['clip_id'] or previous['sentence_id'] != bundle['sentence_id']:
        raise ValueError('Prior OOF order differs')
    if not torch.equal(previous['weight'], bundle['weight'].double()):
        raise ValueError('Prior clock changed')
    for fold, (fit, val) in enumerate(folds):
        s = saved['states'][f'fold{fold}_native']
        if not torch.equal(s['fit_ids'], fit) or not torch.equal(s['validation_ids'], val):
            raise ValueError('Prior native fold changed')
        if not torch.equal(s['metric']['sqrt_metric'], torch.ones(len(channels), dtype=torch.float64)):
            raise ValueError('Native basis required')
    return bundle, channels, speakers, folds, saved, previous, hashes, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'weights', 'basis-study', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh output required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(46); np.random.seed(46); torch.manual_seed(46)
    bundle, channels, speakers, folds, saved, previous, hashes, binding = load_sources(
        args.bundle, args.weights, args.basis_study)
    args.output.mkdir(parents=True, exist_ok=False)
    source_dir = args.output / 'source'; source_dir.mkdir()
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / 'kinetalk_b0/temporal_motion_refiner.py',
        root / 'kinetalk_b0/predictable_motion.py', root / 'scripts/probe_scaled_motion_basis.py',
        root / 'scripts/probe_predictable_motion.py', root / 'scripts/train_predictable_renderer.py',
        root / 'scripts/train_neutral_affect_audio_ablation.py',
        root / 'scripts/audit_predictable_motion_predictions.py',
        root / 'docs/TEMPORAL_AUDIO_REFINER_PROBE.md']
    sources = {str(p.resolve()): sha(p) for p in paths}
    for path in paths:
        dest = source_dir / path.relative_to(root); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    provenance = {'schema': SCHEMA, 'args': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_sha256': sources, 'input_sha256': hashes, 'prior_oof_output_sha256': binding,
        'seed': '46+fold', 'epochs': 8, 'batch_size': 32, 'lr': .0002, 'optimizer': 'Adam',
        'loss': 'weighted_mean_native_centered_residual_MSE / .25**2 only',
        'native_rank': 8, 'alpha': 1., 'hidden': 64, 'arms': list(ARMS),
        'checkpoint_selection': 'none; fixed epoch8', 'audio_gate': False,
        'source_roles_accessed': ['bundles.internal'], 'heldout405_targets_accessed': False,
        'outer280_targets_accessed': False, 'new439_targets_accessed': False, 'test_targets_accessed': False,
        'renderer_trained': False, 'default_changed': False, 'torch': str(torch.__version__),
        'motion_channel_indices': channels, 'clip_count': len(speakers), 'speaker_count': len(set(speakers)),
        'scope': 'Descriptive cross-sentence OOF on existing nineteen training identities; alpha/rank previously selected. Audio features already contextual; pointwise means no additional inter-bin mixing. No renderer or whole-system generalization claim.'}
    write_json(args.output / 'provenance.json', provenance)
    x, y, w = audio_features(bundle), bundle['motion_bins'][..., channels], bundle['weight']
    target = torch.zeros_like(previous['target'])
    predictions = {'ridge': {mode: previous['predictions']['native'][mode].clone()
                             for mode in ('full', 'reverse', 'zero', 'metric_projection_oracle')}}
    for arm in ARMS:
        predictions[arm] = {mode: torch.zeros_like(target) for mode in ('full', 'reverse')}
    fold_details = []
    seen = torch.zeros(len(y), dtype=torch.long)
    for fold, (fit, val) in enumerate(folds):
        state = copy.deepcopy(saved['states'][f'fold{fold}_native']['transformed_model'])
        scale = fit_control_scale(y, w, fit, state)
        models, init_report = matched_models(state, scale, 46 + fold)
        reference = predict_motion(x[fit[:8]], w[fit[:8]], state)
        for model in models.values():
            with torch.no_grad():
                torch.testing.assert_close(model(x[fit[:8]], w[fit[:8]]), reference, rtol=1e-12, atol=1e-12)
        histories = {}
        for arm in ARMS:
            models[arm], histories[arm] = train_arm(models[arm], x, y, w, fit, seed=46 + fold,
                device=args.device, output=args.output / f'fold{fold}_{arm}', provenance=provenance, fold=fold, arm=arm)
            models[arm].cpu()
        for ra, rb in zip(histories['pointwise'], histories['temporal']):
            if ra['batch_order_sha256'] != rb['batch_order_sha256'] or ra['step'] != rb['step']:
                raise ValueError('Unmatched batch budget/order')
        target[val] = weighted_clip_center(y[val], w[val])
        torch.testing.assert_close(target[val], previous['target'][val], rtol=0, atol=0)
        ridge = predict_motion(x[val], w[val], state)
        torch.testing.assert_close(ridge, predictions['ridge']['full'][val], rtol=0, atol=0)
        reversed_x = intervene_input(x[val], w[val], 'reverse')
        for arm in ARMS:
            model = models[arm].to(args.device)
            predictions[arm]['full'][val] = predict_batches(model, x[val], w[val], args.device)
            predictions[arm]['reverse'][val] = predict_batches(model, reversed_x, w[val], args.device)
            model.cpu()
        seen[val] += 1
        fold_details.append({'fold': fold, 'initialization': init_report,
            'control_scale': scale.tolist(), 'control_scale_sha256': state_hash({'scale': scale}),
            'native_state_sha256': state_hash({k: v for k, v in state.items() if torch.is_tensor(v)}),
            'fit_ids': fit.tolist(), 'validation_ids': val.tolist(), 'history': histories})
        del models
    if not torch.all(seen == 1):
        raise ValueError('OOF coverage failed')
    native_groups, local_groups = motion_groups(channels)
    statistics = {arm: {mode: {name: clip_statistics(p, target, w, cc) for name, cc in local_groups.items()}
                        for mode, p in modes.items()} for arm, modes in predictions.items()}
    output = {'provenance': provenance, 'target': target, 'weight': w.double(),
        'predictions': predictions, 'statistics': statistics, 'groups': native_groups,
        'clip_id': bundle['clip_id'], 'sentence_id': bundle['sentence_id'], 'emotion_id': bundle['emotion_id'],
        'speaker_id': speakers, 'source_speaker_id': bundle['speaker_id'], 'oof_fold': previous['oof_fold'],
        'folds': fold_details}
    torch.save(output, args.output / 'oof_predictions.pt')
    write_json(args.output / 'folds.json', fold_details)
    inventory = {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob('*')
                 if p.is_file() and p.name != 'output_hashes.json'}
    write_json(args.output / 'output_hashes.json', inventory)
    print('TEMPORAL_REFINER_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
