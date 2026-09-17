import copy
import pytest
import torch

from scripts.audit_projection_centered_probe import ARMS, validate_centered_pair
from scripts.audit_projection_rollout_probe import audit_two_arm_epoch, ROLLOUT_ARM, ROLLOUT_LOSS, ROLLOUT_SCHEMA
from scripts.train_projection_centered_rollout_probe import ARM, LOSS, SCHEMA


def test_centered_recipe_rejects_extra_changes_and_losses():
    raw = {'schema': ROLLOUT_SCHEMA, 'loss': ROLLOUT_LOSS,
        'args': {'arm': ROLLOUT_ARM, 'output': 'raw', 'lr': .0002},
        'source_sha256': {'/r/scripts/train_projection_rollout_probe.py': 'raw-source', '/r/model.py': 'fixed'},
        'teacher_probability': .5, 'input_sha256': {'cache': 'same'}}
    centered = copy.deepcopy(raw)
    centered.update(schema=SCHEMA, loss=LOSS, mean_anchoring_loss=False,
        loss_centering='Per-clip/per-channel mean of raw prediction-target error over observed frames only; no detach; raw generated output untouched')
    centered['args'].update(arm=ARM, output='centered')
    del centered['source_sha256']['/r/scripts/train_projection_rollout_probe.py']
    centered['source_sha256']['/r/scripts/train_projection_centered_rollout_probe.py'] = 'centered-source'
    validate_centered_pair(centered, raw)
    for mutation in ('lr', 'source', 'teacher', 'extra_loss'):
        changed = copy.deepcopy(centered)
        if mutation == 'lr':
            changed['args']['lr'] = .001
        elif mutation == 'source':
            changed['source_sha256']['/r/model.py'] = 'changed'
        elif mutation == 'teacher':
            changed['teacher_probability'] = 0
        else:
            changed['mean_anchoring_loss'] = True
        with pytest.raises(ValueError):
            validate_centered_pair(changed, raw)


def test_dynamic_improvement_cannot_hide_constant_mean_regression():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        torch.manual_seed(55)
        target = torch.randn(9, 12, 52) * .1
        q = {'motion': target, 'valid': torch.ones(9, 12, dtype=torch.bool),
            'channel_mask': torch.ones(9, 52, dtype=torch.bool), 'speaker_id': torch.tensor([4] * 3 + [5] * 3 + [6] * 3),
            'emotion_id': torch.tensor([0, 1, 1] * 3), 'sentence_id': ['n', 'a', 'b'] * 3,
            'times': torch.arange(12).double()[None].expand(9, -1) / 25}
        reference = {'q': q, 'base': {'b0': torch.zeros_like(target)}, 'identity': {'baseline': torch.zeros(9, 52)}}
        curves = {}
        for arm in ARMS:
            full = target + 1 if arm == 'centered_rollout' else target * .2
            modes = {'full': full, 'zero': target * .5, 'reverse': full.flip(1), 'oracle': target}
            curves[arm] = {'noise_seeds': [42, 123, 2026], 'decode_steps': 12,
                'motion': {str(seed): {mode: value.clone() for mode, value in modes.items()} for seed in (42, 123, 2026)}}
        result = audit_two_arm_epoch(curves, reference, samples=100, arms=ARMS)
        assert set(result['scores']) == set(ARMS)
        assert 'centered_rollout_minus_raw_rollout' in result
        checks = result['same_arm_checks']['centered_rollout']
        assert checks['upper_positive_vs_zero']
        assert not checks['neutral_mouth_one_sided90_within_3pct']
        assert not result['same_arm_checks_all_pass']['centered_rollout']
    finally:
        torch.set_num_threads(previous)
