import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow, ContinuousUpperAE
from kinetalk_b0.models.prior_audio_multiscale import MultiScalePriorAudioResidual
from scripts import audit_multiscale_prior_audio as audit
from scripts import train_multiscale_prior_audio as runner
from scripts.prepare_continuous_motion_dataset import fit_statistics, prepare_segments
from tests.test_multiscale_prior_audio import clips


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_independent_support_matches_canonical_gaps_and_partial_tails():
    data = clips()
    data[0]['motion_mask'][4:8] = False
    data[1]['valid'][3:5] = False
    stats = fit_statistics(data)
    expected, coverage = prepare_segments(data, stats, 'train', 200)
    coverage['retained_clip_ids'] = sorted({s['metadata']['clip_id'] for s in expected})
    actual, report = audit.support_from_masks(data)
    assert report == coverage
    assert [s['metadata'] for s in expected] == [s['metadata'] for s in actual]
    # Event supervision never narrows this independently reconstructed support.
    for clip in data: clip['known'] = torch.zeros(len(clip['valid']), 4, dtype=torch.bool)
    assert audit.support_from_masks(data)[1] == report


def test_replay_matches_training_rng_and_rejects_perturbed_order():
    data = clips(); stats = fit_statistics(data)
    segments, _ = audit.support_from_masks(data)
    seed, batch_size, steps = 19, 4, 7
    rng = np.random.default_rng(seed+1)
    groups = runner.common._clip_groups(segments)
    bank = runner.DonorBank(data, stats)
    order = donors = '0'*64; paired = same = total = 0
    import hashlib
    import json
    for step in range(1, steps+1):
        ids = runner.common._sample_indices(groups, batch_size, rng)
        order = hashlib.sha256(bytes.fromhex(order)+np.asarray(ids, dtype='<i8').tobytes()).hexdigest()
        items = [segments[i] for i in ids]
        reference = torch.zeros(batch_size, 4, 5, 1540)
        frames = torch.ones(batch_size, 4, 5, dtype=torch.bool)
        _, eligible, draw, same_count = bank.batch(items, reference, frames, seed=seed+3, step=step)
        donors = hashlib.sha256(bytes.fromhex(donors)+json.dumps(draw).encode()).hexdigest()
        paired += int(eligible.sum()); same += same_count; total += batch_size
    result = audit.replay_draws(data, segments, seed, batch_size, steps)
    assert result == {'order_sha256': order, 'donor_sha256': donors,
                      'step': steps, 'donor_counts': [paired, same, total]}
    assert audit.replay_draws(data, list(reversed(segments)), seed, batch_size, steps)['order_sha256'] != order


def test_trained_null_and_seed42_raw_example_reproduction(tmp_path):
    torch.manual_seed(43)
    data = clips(); stats = fit_statistics(data)
    ae = ContinuousUpperAE(hidden=8, depth=1, latent_dim=4).eval()
    prior = ContinuousLatentFlow(3, latent_dim=4, hidden=8, depth=1).eval()
    model = MultiScalePriorAudioResidual(prior, hidden=8, depth=1, fast_hidden=4).eval()
    with torch.no_grad(): model.output.weight.normal_(std=.1)
    stats.update(latent_mean=torch.zeros(4), latent_scale=torch.ones(4))
    clip = {**data[0], 'split': 'inner_validation'}
    null = audit.verify_null(model, clip, stats, torch.device('cpu'))
    assert null['velocity_exact_null'] and null['sample3_exact_null']
    arm_dir = tmp_path/'audio'
    audit.native.evaluate_generation(model, ae, [clip], stats, arm_dir, seeds=(42, 123), steps=2)
    row = audit.verify_example(model, ae, clip, stats, tmp_path, 'audio', torch.device('cpu'), 2)
    assert row['raw_max_absolute_difference'] == 0 and row['other43_exact']
    source = arm_dir/'npz'/(clip['clip_id']+'.npz')
    with np.load(source, allow_pickle=False) as z: changed = {k:z[k] for k in z.files}
    changed['motions'][2, 0, 41] += .01
    np.savez_compressed(source, **changed)
    with pytest.raises(ValueError, match='differs from saved raw'):
        audit.verify_example(model, ae, clip, stats, tmp_path, 'audio', torch.device('cpu'), 2)


def test_lightweight_loading_rejects_prior_weights_and_mismatched_binding():
    torch.manual_seed(42)
    prior = ContinuousLatentFlow(3, latent_dim=4, hidden=8, depth=1)
    model = MultiScalePriorAudioResidual(prior, hidden=8, depth=1, fast_hidden=4)
    state = runner.adapter_state(model); initial = runner.common._value_sha(state)
    checkpoint = {'schema':runner.SCHEMA, 'binding':{'protocol_sha256':'bound'},
                  'config':model.config, 'initial_sha256':initial, 'step':2, 'state':state}
    restored = audit.load_adapter(prior, model.config, checkpoint, 'bound', initial, 2)
    assert runner.common._value_sha(restored.prior.state_dict()) == runner.common._value_sha(prior.state_dict())
    bad = copy.deepcopy(checkpoint); bad['state']['prior.output.bias'] = prior.output.bias.detach().clone()
    with pytest.raises(ValueError, match='contains prior weights'):
        audit.load_adapter(prior, model.config, bad, 'bound', initial, 2)
    with pytest.raises(ValueError, match='protocol binding'):
        audit.load_adapter(prior, model.config, checkpoint, 'wrong', initial, 2)
