"""A less bad loss arm must not be presented as dynamic success."""
import copy

import pytest
import torch

from scripts.audit_projection_rollout_probe import ARMS, ROLLOUT_LOSS, audit_two_arm_epoch, validate_matched_rng, validate_recipe_pair


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_rng_audit_includes_teacher_actual_fraction_and_unused_time_draws():
    records = {e: {'minibatch_sha256': 'a' * 64, 'noise_time_sha256': 'b' * 64,
        'teacher_choice_draw_sha256': 'c' * 64, 'step': e * 100, 'samples_seen': 2315, 'teacher_fraction': .5}
        for e in range(1, 19)}
    validate_matched_rng(records, copy.deepcopy(records))
    changed = copy.deepcopy(records); changed[2]['noise_time_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='epoch2'):
        validate_matched_rng(changed, records)
    changed = copy.deepcopy(records); changed[18]['teacher_fraction'] = .4
    with pytest.raises(ValueError, match='schedule'):
        validate_matched_rng(changed, records)


def test_recipe_allows_only_authorized_loss_and_entrypoint_differences():
    flow = {'schema': 'projection_schedule_ablation_v1', 'loss': 'observed_flow_mse_only',
            'args': {'arm': 'constant_teacher', 'output': 'flow', 'lr': .0002},
            'source_sha256': {'/repo/scripts/train_projection_schedule_ablation.py': 'shared'}, 'input_sha256': {'cache': 'same'}}
    rollout = copy.deepcopy(flow)
    rollout.update(schema='projection_rollout_probe_v1', loss=ROLLOUT_LOSS, teacher_probability=.5, decode_steps=12,
                   flow_time_draws='consumed and hashed identically to constant_teacher, unused in rollout loss')
    rollout['args'].update(arm='constant_teacher_rollout', output='rollout')
    rollout['source_sha256']['/repo/scripts/train_projection_rollout_probe.py'] = 'new'
    validate_recipe_pair(rollout, flow)
    changed = copy.deepcopy(rollout); changed['args']['lr'] = .001
    with pytest.raises(ValueError, match='authorized'):
        validate_recipe_pair(changed, flow)
    changed = copy.deepcopy(rollout); changed['source_sha256']['/repo/scripts/train_projection_schedule_ablation.py'] = 'changed'
    with pytest.raises(ValueError, match='Shared'):
        validate_recipe_pair(changed, flow)


def test_improved_rollout_vs_bad_flow_still_fails_own_zero_baseline():
    torch.manual_seed(51)
    target = torch.randn(9, 12, 52) * .1
    q = {'motion': target, 'valid': torch.ones(9, 12, dtype=torch.bool),
         'channel_mask': torch.ones(9, 52, dtype=torch.bool), 'speaker_id': torch.tensor([4] * 3 + [5] * 3 + [6] * 3),
         'emotion_id': torch.tensor([0, 1, 1] * 3), 'sentence_id': ['n', 'a', 'b'] * 3,
         'times': torch.arange(12).double()[None].expand(9, -1) / 25}
    reference = {'q': q, 'base': {'b0': torch.zeros_like(target)}, 'identity': {'baseline': torch.zeros(9, 52)}}
    arms = {}
    for arm in ARMS:
        full = target * (-.2 if arm == 'rollout' else -.5)
        values = {'zero': target * .5, 'full': full, 'reverse': full.flip(1), 'oracle': target * .9}
        arms[arm] = {'noise_seeds': [42, 123, 2026], 'decode_steps': 12,
                    'motion': {str(seed): {mode: value.clone() for mode, value in values.items()} for seed in (42, 123, 2026)}}
    result = audit_two_arm_epoch(arms, reference, samples=100)
    between = result['rollout_minus_flow_constant_teacher']['full']['nonneutral']['centered_residual']['upper_expression']
    assert between['r2_improvement'] > 0
    assert not result['same_arm_checks']['rollout']['upper_positive_vs_zero']
    assert not result['same_arm_checks_all_pass']['rollout']
    assert set(result['scores']) == set(ARMS)
