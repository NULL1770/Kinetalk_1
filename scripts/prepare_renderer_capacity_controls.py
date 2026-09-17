"""Fixed-basis sentence cross-fit student controls for renderer training.

Only ridge coefficients are cross-fitted. U, feature scale and target scale
are frozen from all fit data, so this is explicitly NOT full-pipeline OOF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.predictable_motion import weighted_clip_center
from scripts.probe_scaled_motion_basis import validate_folds
from scripts.train_predictable_renderer import PredictableAudioHead, audio_features, sha, state_hash
from scripts.train_formal_predictable_projection import canonical_hash, save_checkpoint, save_json
from scripts.train_projection_schedule_ablation import read_allowlist

SCHEMA = 'renderer_capacity_training_controls_v1'
INPUT_NAMES = ('cache', 'bundle', 'weights', 'checkpoint', 'config', 'fit_ids', 'validation_ids', 'split_lock')


def fit_fixed_coordinates(features, motion, weight, fit_ids, *, basis, channels, feature_std, alpha=1.):
    """Slice before accessing values; held-out targets cannot enter coefficients."""
    ids = torch.as_tensor(fit_ids, dtype=torch.long)
    if ids.ndim != 1 or not len(ids) or len(ids.unique()) != len(ids):
        raise ValueError('Distinct nonempty fit indices required')
    if ids.min() < 0 or ids.max() >= len(features):
        raise ValueError('Fit indices out of range')
    std, u = feature_std.detach().cpu().double(), basis.detach().cpu().double()
    if not (std > 0).all() or not torch.isfinite(std).all() or not torch.isfinite(u).all():
        raise ValueError('Fixed coordinates must be finite with positive feature scale')
    if alpha != 1.:
        raise ValueError('Prespecified alpha1 only')
    w = weight[ids].detach().cpu().double()
    x = weighted_clip_center(features[ids], w) / std
    y = weighted_clip_center(motion[ids][..., channels], w) @ u
    total = w.sum()
    if total <= 0:
        raise ValueError('Positive fit weight required')
    x, y, w = x.reshape(-1, len(std)), y.reshape(-1, u.shape[1]), w.reshape(-1, 1)
    gram = x.T @ (x * w) / total
    cross = x.T @ (y * w) / total
    gram = (gram + gram.T) * .5 + alpha * torch.eye(len(std), dtype=torch.float64)
    result = torch.linalg.solve(gram, cross)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite fixed-basis ridge coefficients')
    return result


def predict_fixed_coordinates(features, weight, coefficients, feature_std, target_scale):
    """Prediction takes no target/identity/label input."""
    scale = target_scale.detach().cpu().double()
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Finite positive target scale required')
    x = weighted_clip_center(features, weight) / feature_std.detach().cpu().double()
    return weighted_clip_center(x @ coefficients.detach().cpu().double() / scale, weight).float()


def crossfit_controls(features, motion, weight, folds, *, basis, channels, feature_std, target_scale):
    result = torch.zeros((*weight.shape, basis.shape[1]), dtype=torch.float32)
    counts = torch.zeros(len(weight), dtype=torch.long)
    states = []
    for number, (fit_ids, heldout_ids) in enumerate(folds):
        if set(fit_ids.tolist()) & set(heldout_ids.tolist()):
            raise ValueError('Cross-fit partitions overlap')
        coeff = fit_fixed_coordinates(features, motion, weight, fit_ids, basis=basis, channels=channels,
                                      feature_std=feature_std)
        result[heldout_ids] = predict_fixed_coordinates(features[heldout_ids], weight[heldout_ids],
                                                        coeff, feature_std, target_scale)
        counts[heldout_ids] += 1
        states.append({'fold': number, 'fit_ids': fit_ids, 'heldout_ids': heldout_ids, 'coefficients': coeff})
    if not torch.equal(counts, torch.ones_like(counts)):
        raise ValueError('Every fit clip must receive exactly one cross-fit prediction')
    return result, states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh controls output required')
    torch.set_num_threads(4)
    provenance = json.loads((args.source_run / 'provenance.json').read_text(encoding='utf8'))
    recipe = provenance['recipe']
    if canonical_hash(recipe) != provenance['recipe_sha256']:
        raise ValueError('Source recipe hash mismatch')
    if recipe['schema'] != 'projection_scaled_centered_probe_v1' or recipe['args']['arm'] != 'uniform':
        raise ValueError('Expected fixed uniform source')
    paths = {k: Path(recipe['args'][k]) for k in INPUT_NAMES}
    hashes = {k: sha(v) for k, v in paths.items()}
    if hashes != recipe['input_sha256']:
        raise ValueError('Source inputs changed')
    source = torch.load(paths['bundle'], map_location='cpu', weights_only=False)
    fitted = torch.load(paths['weights'], map_location='cpu', weights_only=False)
    adapter = torch.load(args.source_run / 'final_epoch008.pt', map_location='cpu', weights_only=False)
    tr = source['bundles']['internal']
    if len(tr['clip_id']) != 2315:
        raise ValueError('Expected 2315 fit clips')
    allow = read_allowlist(paths['fit_ids'])
    if list(tr['clip_id']) != allow:
        raise ValueError('Fit clip order mismatch')
    state = fitted['states']['rrr_rank8']
    if state['alpha'] != 1. or state['rank'] != 8 or not torch.equal(state['train_ids'], torch.arange(2315)):
        raise ValueError('Require fixed all-fit rank8 alpha1 state')
    head = PredictableAudioHead(state, tr['motion_bins'], tr['weight']).eval()
    if state_hash(head.state_dict()) != adapter['head_sha256']:
        raise ValueError('Head differs from immutable source')
    folds = validate_folds(fitted['selection']['folds'], tr['sentence_id'])
    features, motion, weight = audio_features(tr), tr['motion_bins'], tr['weight'].float()
    # Exact original double fitting coordinates for numerical reproduction;
    # all deployed teacher/control coordinates remain the same common basis.
    kwargs = dict(basis=state['basis'], channels=state['motion_channel_indices'], feature_std=state['std'])
    allfit = fit_fixed_coordinates(features, motion, weight, torch.arange(len(weight)), **kwargs)
    coefficient_error = float((allfit - state['weights'].double()).abs().max())
    if coefficient_error > 2e-6:
        raise ValueError(f'All-fit ridge differs from fixed predictor: {coefficient_error}')
    controls, states = crossfit_controls(features, motion, weight, folds, target_scale=head.target_scale, **kwargs)
    with torch.no_grad():
        deployment = head(features, weight)
    if not torch.isfinite(controls).all():
        raise ValueError('Nonfinite controls')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance_out = {
        'source_recipe_sha256': canonical_hash(recipe), 'source_adapter_sha256': sha(args.source_run/'final_epoch008.pt'),
        'preparation_source_sha256': sha(Path(__file__)), 'allfit_coefficient_max_abs': coefficient_error,
        'shared_basis_feature_scale_target_scale': 'fixed from all 2315 fit clips; not full pipeline OOF',
        'student_coefficients': 'three saved sentence folds; alpha1; fit-only coefficients per fold',
        'external_development_scored_or_fit': False,
        'bundle_note': 'Container includes dev tensors; preparation indexes only bundles.internal. No dev scoring/fitting.',
        'training_control_rms': float(controls.square().mean().sqrt()),
        'deployment_control_rms': float(deployment.square().mean().sqrt()),
        'crossfit_vs_deployment_mse': float((controls-deployment).square().mean()),
        'protocol': 'Conventional renderer capacity experiment, no novelty claim; frozen deployment head retained.',
    }
    payload = {'schema': SCHEMA, 'controls': controls, 'clip_ids': list(tr['clip_id']), 'weight': weight,
        'source_input_sha256': hashes, 'head_sha256': state_hash(head.state_dict()),
        'provenance': provenance_out, 'fold_states': states,
        'sentence_folds': fitted['selection']['folds'], 'basis': state['basis'],
        'feature_std': state['std'], 'target_scale': head.target_scale}
    save_checkpoint(args.output, payload)
    save_json(args.output.with_suffix('.json'), {'schema': SCHEMA, 'output_sha256': sha(args.output),
        'shape': list(controls.shape), 'head_sha256': payload['head_sha256'], 'provenance': provenance_out,
        'folds': [{'fold': s['fold'], 'fit_count': len(s['fit_ids']), 'heldout_count': len(s['heldout_ids']),
                   'coefficient_sha256': state_hash({'coefficient': s['coefficients']})} for s in states]})
    print(json.dumps({'complete': True, 'output': str(args.output), **provenance_out}), flush=True)


if __name__ == '__main__':
    main()
