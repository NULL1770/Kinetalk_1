"""Fixed-epoch schedule comparison must preserve pairing and negative results."""
import copy

import pytest
import torch

from scripts.audit_teacher_schedule_probe import ARMS, SEEDS, audit_epoch, same_experiment_recipe, validate_rng_evidence


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_schedule_only_recipe_difference_and_every_epoch_random_draw_pairing():
    recipe = {'args': {'arm': 'constant_teacher', 'output': 'a', 'seed': 46}, 'input_sha256': {'cache': 'same'}}
    other = copy.deepcopy(recipe); other['args'].update(arm='audio_only', output='b')
    assert same_experiment_recipe(recipe) == same_experiment_recipe(other)
    other['args']['seed'] = 47
    assert same_experiment_recipe(recipe) != same_experiment_recipe(other)
    records = {arm: {epoch: {'minibatch_sha256': 'a' * 64, 'noise_time_sha256': 'b' * 64,
        'teacher_choice_draw_sha256': 'c' * 64, 'step': epoch * 100, 'samples_seen': 2300}
        for epoch in range(1, 19)} for arm in ARMS}
    validate_rng_evidence(records)
    records['audio_only'][7]['teacher_choice_draw_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='epoch7'):
        validate_rng_evidence(records)


def test_oracle_only_improvement_does_not_become_audio_improvement():
    torch.manual_seed(51)
    target = torch.randn(9, 12, 52) * .1
    q = {'motion': target, 'valid': torch.ones(9, 12, dtype=torch.bool),
         'channel_mask': torch.ones(9, 52, dtype=torch.bool), 'speaker_id': torch.tensor([4] * 3 + [5] * 3 + [6] * 3),
         'emotion_id': torch.tensor([0, 1, 1] * 3), 'sentence_id': ['n', 'a', 'b'] * 3,
         'times': torch.arange(12).double()[None].expand(9, -1) / 25}
    reference = {'q': q, 'base': {'b0': torch.zeros_like(target)}, 'identity': {'baseline': torch.zeros(9, 52)}}
    arms = {}
    for arm in ARMS:
        values = {'zero': target * .5, 'full': target * .6, 'reverse': target.flip(1) * .6,
                  'oracle': target * (.95 if arm == 'constant_teacher' else .7)}
        arms[arm] = {'noise_seeds': list(SEEDS), 'decode_steps': 12,
                     'motion': {str(seed): {mode: value.clone() for mode, value in values.items()} for seed in SEEDS}}
    result = audit_epoch(arms, reference, samples=100)
    difference = result['schedule_contrasts']['constant_teacher__vs__decay_teacher']
    assert difference['full']['nonneutral']['upper_expression']['r2_improvement'] == pytest.approx(0)
    assert difference['oracle']['nonneutral']['upper_expression']['r2_improvement'] > 0
    assert result['full_vs_same_arm']['constant_teacher']['oracle']['nonneutral']['upper_expression']['r2_improvement'] < 0
    assert 'neutral' in result['velocity_relative_vs_zero']['constant_teacher']
    arms['audio_only']['motion']['42']['zero'] += .001
    with pytest.raises(ValueError, match='Zero-local'):
        audit_epoch(arms, reference, samples=100)
