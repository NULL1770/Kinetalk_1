"""Read-only fixed-final audit of aligned-local and matched free-upper runs.

Only saved metadata, state dictionaries and generated curves are read. No
inference, fitting, sample selection or checkpoint selection is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audio_flow_metrics import trajectory_energy_score, adjacent_variogram_score
from scripts.train_formal_predictable_projection import canonical_hash
from kinetalk_b0.models.temporal_upper import UPPER_INDICES

SCHEMA = 'aligned_local_free_upper_v1'
SEEDS = (42, 123, 2026)
GROUPS = {'brows': (41, 42, 43, 44, 45), 'eyes_expression': (5, 6, 12, 13), 'mouth': tuple(range(14, 41))}
NOT_UPPER = tuple(i for i in range(52) if i not in UPPER_INDICES)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(2**20), b''): digest.update(part)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def load_pt(path): return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def tensor_state_equal(a, b, description):
    if not isinstance(a, dict) or not isinstance(b, dict) or not a or set(a) != set(b):
        raise ValueError(description + ': state keys differ')
    for name, value in a.items():
        other = b[name]
        if not torch.is_tensor(value) or not torch.is_tensor(other) or value.dtype != other.dtype or not torch.equal(value, other):
            raise ValueError(description + ': tensor changed ' + name)


def validate_curves(curves, *, phase=None):
    ids, target = curves.get('clip_id', []), curves['target']
    valid, channels, times = (curves[k] for k in ('valid', 'channel_mask', 'times'))
    if (target.ndim != 3 or target.shape[-1] != 52 or target.shape[0] != len(ids)
            or len(ids) != len(set(ids)) or not ids or valid.shape != target.shape[:2]
            or valid.dtype != torch.bool or not valid.any(1).all()
            or channels.shape != (len(ids), 52) or channels.dtype != torch.bool
            or times.shape != valid.shape or not torch.isfinite(times).all()):
        raise ValueError('Invalid curve IDs, shape or mask')
    if not torch.allclose(times[:, 1:] - times[:, :-1], torch.full_like(times[:, 1:], .04), atol=1e-7, rtol=1e-5):
        raise ValueError('Native 25fps clock changed')
    if curves.get('noise_seeds') != list(SEEDS):
        raise ValueError('Expected all three fixed seeds')
    expected = {f'{seed}/full' for seed in SEEDS}
    if phase in ('direct', 'soft'): expected |= {'42/base', '42/static', '42/reverse', '42/zero_state'}
    if phase == 'align': expected |= {'42/original_local', '42/zero_local', '42/oracle_local'}
    if not expected <= curves.get('predictions', {}).keys():
        raise ValueError('Missing required fixed-seed conditions')
    observed = valid[..., None] & channels[:, None]
    for name, value in {'target': target, 'b0': curves['b0'], **curves['predictions']}.items():
        if value.shape != target.shape or not torch.isfinite(value[observed]).all():
            raise ValueError('Invalid observed saved curve ' + name)
    if phase in ('direct', 'soft'):
        base = curves['predictions']['42/base'][..., list(NOT_UPPER)]
        for mode in ('full', 'static', 'reverse', 'zero_state'):
            if not torch.equal(curves['predictions']['42/' + mode][..., list(NOT_UPPER)], base):
                raise ValueError('Same-stage non-upper channels changed: ' + phase + '/' + mode)
        if phase == 'direct' and not torch.equal(curves['predictions']['42/full'], curves['predictions']['42/zero_state']):
            raise ValueError('Direct arm unexpectedly uses state')


def load_run(path, phase):
    path = Path(path).resolve()
    provenance, complete = read(path / 'provenance.json'), read(path / 'complete.json')
    recipe = provenance['recipe']; digest = canonical_hash(recipe)
    if (recipe.get('schema') != SCHEMA or recipe.get('phase') != phase
            or provenance.get('recipe_sha256') != digest or complete.get('schema') != SCHEMA
            or complete.get('phase') != phase or recipe.get('fixed_final_epoch') is not True
            or recipe.get('smoke') is not False or recipe.get('test_loaded') is not False
            or recipe.get('default_replaced') is not False):
        raise ValueError('Run protocol/provenance differs: ' + phase)
    bindings = {}
    for name in ('final', 'curves'):
        p = path / (name + '.pt'); bindings[name] = sha(p)
        if complete.get(name + '_sha256') != bindings[name]:
            raise ValueError('Completed file hash differs: ' + phase + '/' + name)
    final, curves = load_pt(path / 'final.pt'), load_pt(path / 'curves.pt')
    epochs = recipe.get('epochs')
    if (type(epochs) is not int or epochs < 1 or complete.get('completed_epochs') != epochs
            or final.get('completed_epochs') != epochs or final.get('recipe_sha256') != digest
            or final.get('schema') != SCHEMA or final.get('phase') != phase
            or complete.get('frozen') != recipe.get('frozen')
            or final.get('source_bindings') != recipe.get('source_bindings')
            or curves.get('schema') != SCHEMA or curves.get('phase') != phase):
        raise ValueError('Incomplete or differently bound fixed final run: ' + phase)
    rows = [read(path / f'epoch{i:03d}.json') for i in range(1, epochs + 1)]
    for i, row in enumerate(rows, 1):
        if row.get('phase') != phase or row.get('epoch') != i or not row.get('batch_noise_time_sha256'):
            raise ValueError('Missing complete epoch draw history: ' + phase)
    if rows[-1]['total_steps'] != final['total_steps']:
        raise ValueError('Final step count differs from final epoch')
    validate_curves(curves, phase=phase)
    initial = load_pt(path / 'initial.pt')
    bindings.update(initial=sha(path / 'initial.pt'), provenance=sha(path / 'provenance.json'))
    return {'path': str(path), 'recipe': recipe, 'final': final, 'curves': curves,
            'initial': initial, 'epochs': rows, 'binding': bindings}


def validate_matched(direct, soft, align=None):
    aa, bb = dict(direct['recipe']), dict(soft['recipe'])
    aa.pop('phase'); bb.pop('phase')
    if aa != bb: raise ValueError('Matched direct/soft recipes differ beyond phase')
    for key in ('upper', 'local', 'adapter'):
        tensor_state_equal(direct['initial'][key], soft['initial'][key], 'Matched initial ' + key)
    draws = []
    if len(direct['epochs']) != len(soft['epochs']): raise ValueError('Matched epoch counts differ')
    for a, b in zip(direct['epochs'], soft['epochs']):
        for key in ('epoch', 'batches', 'total_steps', 'batch_noise_time_sha256'):
            if a[key] != b[key]: raise ValueError('Matched epoch RNG/budget differs: ' + key)
        draws.append(a['batch_noise_time_sha256'])
    for run in (direct, soft):
        tensor_state_equal(run['initial']['adapter'], run['final']['adapter'], 'Frozen adapter ' + run['recipe']['phase'])
        if align is not None:
            tensor_state_equal(run['initial']['adapter'], align['final']['adapter'], 'Completed aligned adapter')
            if run['recipe']['alignment']['sha256'] != align['binding']['final']:
                raise ValueError('Aligned checkpoint lineage differs')
    return {'initial_upper_local_adapter_equal_exactly': True, 'frozen_adapter_equal_exactly': True,
            'epochs': len(draws), 'epoch_batch_noise_time_sha256': draws,
            'initial_file_sha256': {'direct': direct['binding']['initial'], 'soft': soft['binding']['initial']},
            'initial_binding_limit': 'Initial snapshots are compared and hashed now; the training complete manifest did not originally bind their hashes.'}


def metadata_equal(reference, candidate, *, compare_b0=True):
    if reference['clip_id'] != candidate['clip_id']: raise ValueError('Cross-run clip order differs')
    for key in ('target', 'valid', 'channel_mask', 'times') + (('b0',) if compare_b0 else ()):
        if not torch.equal(reference[key], candidate[key]): raise ValueError('Cross-run native metadata differs: ' + key)
    for key in ('emotion_id', 'speaker_id'):
        if key in candidate and not torch.equal(reference[key], candidate[key]): raise ValueError('Cross-run labels differ: ' + key)


def center(x, observed):
    clean = torch.where(observed, x, 0.)
    mean = clean.sum(-2, keepdim=True) / observed.sum(-2, keepdim=True).clamp_min(1)
    return torch.where(observed, x - mean, 0.)


def paired_metrics(pred, target, observed):
    pred, target = pred.double(), target.double()
    pc, tc = center(pred, observed), center(target, observed)
    energy, penergy = tc.square().sum(), pc.square().sum()
    sse = (pc - tc).square().sum()
    raw = pred[observed] - target[observed]
    pairs = observed[:, 1:] & observed[:, :-1]
    velocity = (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])
    return {'raw_mse': float(raw.square().mean()), 'centered_mse': float(sse / observed.sum()),
            'centered_r2': float(1. - sse / energy.clamp_min(1e-12)),
            'centered_correlation': float((pc * tc).sum() / (penergy * energy).sqrt().clamp_min(1e-12)),
            'rms_ratio': float((penergy / energy.clamp_min(1e-12)).sqrt()),
            'frame_displacement_mse': float(velocity[pairs].square().mean()) if pairs.any() else None,
            'outside_fraction': float(((pred[observed] < 0) | (pred[observed] > 1)).double().mean())}


def compact_score(score):
    # Retain exact equal-clip aggregate and pair coverage without giant arrays.
    return {key: ({k: v for k, v in value.items() if k != 'per_clip'} if isinstance(value, dict) else value)
            for key, value in score.items() if key not in ('included_clip_indices', 'per_clip_observed_pairs')}


def summarize(curves, labels):
    target = curves['target']; observed = curves['valid'][..., None] & curves['channel_mask'][:, None]
    stack = torch.stack([curves['predictions'][f'{seed}/full'] for seed in SEEDS])
    populations = {'all': torch.ones(len(target), dtype=torch.bool), 'neutral': labels == 0, 'nonneutral': labels != 0}
    output = {}
    for population, rows in populations.items():
        if not rows.any(): continue
        groups = {}
        for name, cc in GROUPS.items():
            mask = observed[rows][..., list(cc)]; truth = target[rows][..., list(cc)]
            samples = stack[:, rows][..., list(cc)]
            if not mask.flatten(1).any(1).all(): raise ValueError('A scored clip has no observed region: ' + name)
            per_seed = [paired_metrics(value, truth, mask) for value in samples]
            mean = {k: sum(v[k] for v in per_seed) / len(SEEDS) if per_seed[0][k] is not None else None for k in per_seed[0]}
            distribution = {}
            for kind, x, y in (('raw', samples, truth), ('centered_coefficients', center(samples.double(), mask[None]), center(truth.double(), mask))):
                distribution[kind] = {'trajectory_energy_score': compact_score(trajectory_energy_score(x, y, mask)),
                                      'adjacent_variogram_score': compact_score(adjacent_variogram_score(x, y, mask, power=.5))}
            groups[name] = {'mean_over_three_seeds': mean, 'per_seed': dict(zip(map(str, SEEDS), per_seed)), 'distribution': distribution}
        output[population] = {'clips': int(rows.sum()), 'groups': groups}
    interventions = {}
    full = curves['predictions']['42/full']
    for key, value in curves['predictions'].items():
        if not key.startswith('42/') or key == '42/full': continue
        rows = {}
        for name, cc in GROUPS.items():
            mask = observed[..., list(cc)]
            stats = paired_metrics(value[..., list(cc)], target[..., list(cc)], mask)
            response = (value[..., list(cc)] - full[..., list(cc)])[mask].double()
            stats['matched_noise_response_rms'] = float(response.square().mean().sqrt())
            reference = paired_metrics(full[..., list(cc)], target[..., list(cc)], mask)
            stats['delta_raw_mse_vs_full'] = stats['raw_mse'] - reference['raw_mse']
            stats['delta_centered_r2_vs_full'] = stats['centered_r2'] - reference['centered_r2']
            rows[name] = stats
        interventions[key] = rows
    return {'noise_seeds': list(SEEDS), 'populations': output, 'single_seed_interventions': interventions}


def compare(new, old):
    output = {}
    for population, row in new['populations'].items():
        if population not in old['populations']: continue
        groups = {}
        for name, metrics in row['groups'].items():
            a = metrics['mean_over_three_seeds']; b = old['populations'][population]['groups'][name]['mean_over_three_seeds']
            values = {key: a[key] - b[key] if a[key] is not None and b[key] is not None else None for key in a}
            values['raw_mse_percent_change'] = 100. * (a['raw_mse'] / b['raw_mse'] - 1.) if b['raw_mse'] else None
            for kind in ('raw', 'centered_coefficients'):
                for metric in ('trajectory_energy_score', 'adjacent_variogram_score'):
                    left = metrics['distribution'][kind][metric]['fair']['mean']
                    right = old['populations'][population]['groups'][name]['distribution'][kind][metric]['fair']['mean']
                    values[kind + '_' + metric + '_fair_delta'] = left - right if left is not None and right is not None else None
            groups[name] = values
        output[population] = groups
    return output


def audit(align_run, direct_run, soft_run, old_run, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('Fresh audit output required')
    runs = {phase: load_run(path, phase) for phase, path in (('align', align_run), ('direct', direct_run), ('soft', soft_run))}
    matched = validate_matched(runs['direct'], runs['soft'], runs['align'])
    reference = runs['align']['curves']
    for run in runs.values(): metadata_equal(reference, run['curves'])
    old_root = Path(old_run).resolve(); old_curves = {}; old_binding = {}; base_recomputation = {}
    for stage in ('audio', 'dynamics'):
        p = old_root / stage / 'curves.pt'; complete = read(p.with_name('complete.json'))
        final_path = p.with_name('final.pt')
        if (complete.get('stage') != stage or sha(p) != complete['curves_sha256']
                or sha(final_path) != complete['final_sha256']):
            raise ValueError('Old run completed final/curve hash differs')
        curves = load_pt(p)
        if curves.get('schema') != 'full_staged_spline_innovation_v1' or curves.get('stage') != stage:
            raise ValueError('Old full-staged curve schema/stage differs')
        validate_curves(curves); metadata_equal(reference, curves, compare_b0=False)
        # B0 is recomputed model output, not target metadata. run12 cached at
        # batch32 and repair at batch16; raw/centered coefficient scores never
        # subtract B0. Record the difference rather than alter any curve.
        difference=(reference['b0']-curves['b0']).double()
        base_recomputation[stage]={'exact':bool(torch.equal(reference['b0'],curves['b0'])),
            'max_abs':float(difference.abs().max()),'rms':float(difference.square().mean().sqrt()),
            'used_in_comparison_scoring':False,'source':'Saved independently recomputed model outputs; not a target or timestamp binding'}
        old_curves[stage] = curves; old_binding[stage] = {'path': str(p), 'sha256': sha(p),
                                                        'final_sha256': sha(final_path)}
    labels = reference['emotion_id']
    reports = {phase: summarize(run['curves'], labels) for phase, run in runs.items()}
    old_reports = {stage: summarize(curves, labels) for stage, curves in old_curves.items()}
    result = {'schema': 'temporal_repair_fixed_final_audit_v1', 'matched_training': matched,
        'run_bindings': {phase: {'path': run['path'], **run['binding'], 'epochs': run['final']['completed_epochs'],
                               'steps': run['final']['total_steps']} for phase, run in runs.items()},
        'old_run_bindings': old_binding, 'same_stage_nonupper_exact': {'direct': True, 'soft': True},
        'native_metadata_equal_exactly': True, 'cross_run_b0_recomputation':base_recomputation, 'reports': reports, 'old_reports': old_reports,
        'comparisons': {phase + '_minus_run12_' + stage: compare(report, old_reports[stage])
                        for phase, report in reports.items() for stage in old_reports},
        'soft_minus_direct': compare(reports['soft'], reports['direct']),
        'distribution_definition': {
            'energy': 'Per clip full-trajectory L2/sqrt(observed elements): mean_k ||Xk-y|| - sum_{i<j}||Xi-Xj||/[K(K-1)], K=3; equal clip mean.',
            'variogram': 'Same-channel adjacent valid pairs only; power=.5. Mean (mean_k abs(deltaXk)^.5-abs(deltaGT)^.5)^2 minus sample_variance/K; equal clip mean.',
            'centered': 'Each predicted and GT coefficient separately loses its own valid-frame temporal mean. No B0 subtraction; not historical centered-residual scoring.',
            'limits': 'Three fixed seeds, one target trajectory per audio; fair estimates assume iid seeds. Finite fair VS can be negative. No best-of-K, significance claim, perceptual certification or calibration guarantee.'},
        'intervention_scope': 'static/reverse affect the new local and optional state; frozen content/global/base remain. oracle_local in align reads GT motion and is diagnostic only.',
        'checkpoint_selection_performed': False, 'model_inference_performed': False, 'test_loaded': False,
        'script_sha256': sha(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('align-run', 'direct-run', 'soft-run', 'old-run', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); result = audit(args.align_run, args.direct_run, args.soft_run, args.old_run, args.output)
    print(json.dumps({'output': str(args.output), 'matched_epochs': result['matched_training']['epochs'], 'fixed_seeds': list(SEEDS)}))


if __name__ == '__main__': main()
