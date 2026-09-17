"""Matched centered-vs-raw rollout audit; all scores use unchanged raw output."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_audio_activity_gate import SEEDS, write_json
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.audit_projection_rollout_probe import (
    ROLLOUT_ARM, ROLLOUT_LOSS, ROLLOUT_SCHEMA, audit_two_arm_epoch, load_rollout,
    validate_matched_rng, validate_recipe_pair,
)
from scripts.audit_projection_mean_dynamic_tradeoff import analyze_saved_curves
from scripts.audit_teacher_schedule_probe import (
    EPOCHS, MODES, load_arm, validate_curves, validate_training_inputs,
)
from scripts.train_projection_centered_rollout_probe import ARM, LOSS, SCHEMA
from scripts.train_predictable_renderer import basic_metrics, sha

ARMS = ('centered_rollout', 'raw_rollout')


def validate_centered_pair(centered, raw):
    """Reject changes other than loss definition and its independent entry."""
    if (centered.get('schema'), centered.get('loss'), centered['args'].get('arm')) != (SCHEMA, LOSS, ARM):
        raise ValueError('Wrong centered objective/schema/arm')
    if (raw.get('schema'), raw.get('loss'), raw['args'].get('arm')) != (ROLLOUT_SCHEMA, ROLLOUT_LOSS, ROLLOUT_ARM):
        raise ValueError('Control must be the actual raw rollout')
    if centered.get('mean_anchoring_loss') is not False or centered.get('loss_centering') != (
        'Per-clip/per-channel mean of raw prediction-target error over observed frames only; no detach; raw generated output untouched'):
        raise ValueError('Unexpected centering or extra loss')
    left, right = copy.deepcopy(centered), copy.deepcopy(raw)
    for recipe, entry in ((left, 'train_projection_centered_rollout_probe.py'), (right, 'train_projection_rollout_probe.py')):
        entry_keys = [p for p in recipe['source_sha256'] if p.replace('\\', '/').endswith('/scripts/' + entry)]
        if len(entry_keys) != 1:
            raise ValueError('Missing independent objective source')
        recipe['source_sha256'].pop(entry_keys[0])
        for key in ('schema', 'loss'):
            recipe.pop(key)
        for key in ('arm', 'output'):
            recipe['args'].pop(key)
    for key in ('mean_anchoring_loss', 'loss_centering'):
        left.pop(key)
    if left != right:
        raise ValueError('Centered/raw differ beyond the single authorized loss change')


def teacher_summary(arms, epoch):
    values = {arm: {mode: [float(data['reports'][epoch][str(seed)][mode]['frozen_teacher_emotion_accuracy'])
        for seed in SEEDS] for mode in MODES} for arm, data in arms.items()}
    if any(not np.isfinite(v) or not 0 <= v <= 1 for modes in values.values() for seeds in modes.values() for v in seeds):
        raise ValueError('Invalid frozen-teacher readout')
    means = {arm: {mode: float(np.mean(v)) for mode, v in modes.items()} for arm, modes in values.items()}
    return {'mean_accuracy': means, 'per_noise_accuracy': values,
        'full_minus_zero': {arm: row['full'] - row['zero'] for arm, row in means.items()},
        'note': 'Frozen training motion teacher only; not independent emotion/perceptual quality evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('centered', 'raw', 'flow-control', 'output', 'decomposition-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists() or args.decomposition_output.exists() or args.samples < 1:
        raise ValueError('Fresh outputs and positive bootstrap count required')
    torch.set_num_threads(4)
    centered = load_rollout(args.centered, schema=SCHEMA, arm=ARM)
    raw = load_rollout(args.raw)
    flow = load_arm(args.flow_control, 'constant_teacher')
    validate_centered_pair(centered['recipe'], raw['recipe'])
    validate_recipe_pair(raw['recipe'], flow['recipe'])
    validate_matched_rng(centered['records'], raw['records'])
    validate_matched_rng(raw['records'], flow['records'])
    if centered['summary'].get('generated_output_centered') is not False:
        raise ValueError('Audit requires unmodified generated output')
    reference, lock, scope, frozen = validate_training_inputs({'constant_teacher': flow})
    arms = dict(zip(ARMS, (centered, raw)))
    for data in arms.values():
        for epoch in EPOCHS:
            ck = data['checkpoints'][epoch]
            if ck['head_sha256'] != frozen['fixed_head'] or ck['frozen_state_sha256'] != frozen['frozen_backbone']:
                raise ValueError('Frozen head/backbone differs')
            validate_curves(data['curves'][epoch], reference)
            for seed in SEEDS:
                for mode in MODES:
                    assert_metric_agreement(basic_metrics(data['curves'][epoch]['motion'][str(seed)][mode], reference),
                        data['reports'][epoch][str(seed)][mode])
    epochs = {}
    for epoch in EPOCHS:
        epochs[str(epoch)] = audit_two_arm_epoch({arm: data['curves'][epoch] for arm, data in arms.items()},
            reference, samples=args.samples, arms=ARMS)
        epochs[str(epoch)]['frozen_teacher_emotion_readout'] = teacher_summary(arms, epoch)
    source_paths = [Path(__file__), Path(__file__).with_name('audit_projection_rollout_probe.py'),
        Path(__file__).with_name('audit_projection_mean_dynamic_tradeoff.py'),
        Path(__file__).with_name('audit_teacher_schedule_probe.py')]
    common = {'source_sha256': {p.name: sha(p) for p in source_paths},
        'arm_hashes': {arm: data['hashes'] for arm, data in arms.items()}, 'frozen_hashes': frozen,
        'split_lock': lock, 'data_scope': scope, 'primary_epoch': 18, 'auxiliary_epoch': 2,
        'same_rng_all18epochs': True, 'same_data_parameters_teacher_schedule': True,
        'single_factor_change': 'Subtract observed temporal error mean in the sole rollout MSE; raw generated output unchanged',
        'bootstrap_samples': args.samples, 'bootstrap_seed': 45, 'noise_seeds': SEEDS,
        'checkpoint_or_arm_selected': False, 'outer280_loaded': False, 'new_identity439_loaded': False,
        'test_loaded': False, 'default_replaced': False,
        'scope': 'Internal19fit/3heldout dynamic-adapter identity diagnosis; inherited B0/global exposure and shared scripts. Not independent novel-sentence or whole-system unseen-person test.'}
    report = {'schema': 'projection_centered_vs_raw_rollout_audit_v1', **common, 'epochs': epochs}
    report['same_arm_checks_scope'] = ('Existing dynamic/neutral/mouth engineering checks only; passing does not certify nonneutral mean/global preservation or deployment. '
        'Inspect raw/mean decomposition for every population and frozen-teacher readout separately; no post-hoc mean threshold is introduced.')
    report['promotion_decision'] = 'none; diagnostic experiment only'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print('PAIRED_AUDIT_COMPLETE', flush=True)
    decomposition = analyze_saved_curves(arms, reference, samples=args.samples)
    write_json(args.decomposition_output, {'schema': 'projection_centered_rollout_decomposition_v1', **common,
        'equation': 'raw SSE = observed-time-centered error SSE + sum(n_observed * mean_error²)',
        'identity_and_B0_cancel_verified': True, 'per_clip_channel_decomposition_verified': True, **decomposition})
    main_epoch = epochs['18']
    print(json.dumps({'checks': main_epoch['same_arm_checks']['centered_rollout'],
        'nonneutral_full_vs_zero': {group: {'delta_r2': row['r2_improvement'], 'ci95': row['r2_improvement_ci95']}
            for group, row in main_epoch['full_vs_same_arm']['centered_rollout']['zero']['nonneutral'].items()}}), flush=True)


if __name__ == '__main__':
    main()
