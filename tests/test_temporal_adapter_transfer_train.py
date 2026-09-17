"""Methodological contracts for matched incremental sentence adaptation."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts import train_temporal_adapter_transfer as runner


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def source_fixture(frames=29):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(191)
        source = SlowStateAffect(torch.zeros(5), torch.ones(5), hidden=8,
                                 local_dim=64, global_dim=64,
                                 num_emotions=4, num_levels=2, stride=4).eval()
        with torch.no_grad():
            source.local_head.weight.normal_(std=.1)
            source.local_head.bias.normal_(std=.03)
        features = torch.randn(3, frames, 5)
    valid = torch.ones(3, frames, dtype=torch.bool)
    valid[0, 3:5] = False
    valid[1, -3:] = False
    valid[2, :2] = False
    features[~valid] = float('nan')
    q = {'audio_features': features, 'valid': valid,
         'clip_id': ['a', 'b', 'c'], 'sentence_id': ['s1', 's2', 's3']}
    with torch.no_grad():
        q['prefix_local'] = runner.a.local_features(source, features, valid)
    return source, q


def assert_bits(left, right):
    assert runner._same_bits(left, right)


def test_three_arms_start_identically_without_mutating_source_or_random_stream():
    source, q = source_fixture()
    source.train().requires_grad_(True)
    original = copy.deepcopy(source.state_dict())
    before_rng = torch.random.get_rng_state().clone()
    for arm in runner.ARMS:
        local, adapter = runner.configure(arm, source, 'cpu')
        assert not local.training and not adapter.training
        assert local is not source
        assert_bits(runner.conditioned_features(local, adapter, q, arm), q['prefix_local'])
        assert sum(p.numel() for p in adapter.parameters()) == 1024
        for name, parameter in local.named_parameters():
            permitted = name.startswith(('input.', 'blocks.', 'local_head.'))
            assert parameter.requires_grad == (arm == 'full_local' and permitted)
        assert all(p.requires_grad == (arm == 'rank8_adapter') for p in adapter.parameters())
        for key, value in original.items():
            assert_bits(value, local.state_dict()[key])
            assert_bits(value, source.state_dict()[key])
    assert source.training and all(p.requires_grad for p in source.parameters())
    assert_bits(before_rng, torch.random.get_rng_state())
    with pytest.raises(ValueError, match='Unknown adaptation'):
        runner.configure('invented_arm', source, 'cpu')


def test_initial_identity_uses_same_forward_and_reports_cache_difference():
    source, q = source_fixture()
    local, adapter = runner.configure('rank8_adapter', source, 'cpu')
    native = runner.conditioned_features(local, adapter, q, 'rank8_adapter')
    q['prefix_local'] = q['prefix_local'] + .001
    report = runner.verify_initial_features(native, source, q)
    assert report['same_forward_bit_exact']
    assert report['old_cache_max_abs'] > .0009
    with pytest.raises(RuntimeError, match='same-forward frozen source'):
        runner.verify_initial_features(native + .001, source, q)


@pytest.mark.parametrize('arm', runner.ARMS)
def test_conditioning_never_executes_global_heads_or_consumes_target_metadata(arm):
    source, q = source_fixture()
    local, adapter = runner.configure(arm, source, 'cpu')

    def forbidden(*args):
        raise AssertionError('Source/global/state/full-forward execution is forbidden')

    hooks = [source.register_forward_pre_hook(forbidden), local.register_forward_pre_hook(forbidden)]
    hooks += [getattr(local, name).register_forward_pre_hook(forbidden) for name in
              ('global_head', 'state_head', 'emotion_classifier', 'intensity_classifier')]
    if arm != 'rank8_adapter':
        hooks.append(adapter.register_forward_pre_hook(forbidden))
    try:
        dirty = {**q, 'motion': object(), 'static_upper': object(), 'audio_global': object(),
                 'emotion_id': object(), 'target': object()}
        actual = runner.conditioned_features(local, adapter, dirty, arm)
        assert_bits(actual, q['prefix_local'])
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize('arm', runner.ARMS)
def test_real_upper_loss_routes_gradients_only_to_the_declared_arm(arm):
    source, q = source_fixture(frames=96)
    source.eval().requires_grad_(False)
    source_before = copy.deepcopy(source.state_dict())
    local, adapter = runner.configure(arm, source, 'cpu')
    local_before = copy.deepcopy(local.state_dict())
    adapter_before = copy.deepcopy(adapter.state_dict())
    cfg = {'model': {'content_dim': 4, 'emotion_dim': 64, 'style_dim': 2,
                     'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.}}
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(341)
        upper = PrefixUpperFlow(cfg).eval()
        b = {**q, 'motion': torch.randn(3, 96, 52), 'static_upper': torch.randn(3, 9),
             'h0': torch.randn(3, 96, 4), 'audio_global': torch.randn(3, 64),
             'audio_intensity': torch.randn(3, 1)}
        identity = {'code': torch.randn(3, 2)}
    params = list(upper.parameters()) + [p for p in local.parameters() if p.requires_grad]
    params += [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0)
    generator = torch.Generator().manual_seed(91)
    records = {}
    for step in range(1, 3):
        native = runner.conditioned_features(local, adapter, b, arm)
        loss, _ = runner.context.context_batch(upper, b, identity, native,
                                               torch.ones(9), generator, 'chunk_teacher')
        runner.p.r.optimize(loss, optimizer, params)
        records[str(step)] = runner.gradient_record(local, adapter)
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    runner.check_gradients(arm, records)
    for name, value in local.state_dict().items():
        permitted = name.startswith(('input.', 'blocks.', 'local_head.'))
        if not (arm == 'full_local' and permitted):
            assert_bits(value, local_before[name])
        assert_bits(source.state_dict()[name], source_before[name])
    assert any(not runner._same_bits(v, local_before[k]) for k, v in local.state_dict().items()) == (arm == 'full_local')
    assert any(not runner._same_bits(v, adapter_before[k]) for k, v in adapter.state_dict().items()) == (arm == 'rank8_adapter')
    assert all(p.grad is None for p in source.parameters())
    if arm == 'rank8_adapter':
        assert records['1']['adapter_down']['absolute_sum'] == 0
        assert records['1']['adapter_up']['absolute_sum'] > 0
        assert records['2']['adapter_down']['absolute_sum'] > 0
    corrupted = copy.deepcopy(records)
    corrupted['1']['input']['has_gradient'] = arm != 'full_local'
    with pytest.raises(RuntimeError, match='Local gradient policy'):
        runner.check_gradients(arm, corrupted)


def deployment_fixture():
    generator = torch.Generator().manual_seed(253)
    valid = torch.tensor([[True, False, True, True, False], [False, True, True, False, True]])
    common = {'clip_id': ['a', 'b'], 'valid': valid, 'channel_mask': torch.ones(2, 52, dtype=torch.bool),
              'target': torch.randn(2, 5, 52, generator=generator),
              'times': torch.arange(5, dtype=torch.float64)[None].repeat(2, 1) / 25,
              'b0': torch.randn(2, 5, 52, generator=generator),
              'speaker_id': torch.tensor([2, 7]), 'emotion_id': torch.tensor([1, 3])}
    bases, generated = {}, {}
    for seed in (42, 123):
        base = torch.randn(2, 5, 52, generator=generator)
        base[~valid] = float('nan')
        base[0, 1, 0] = -0.
        base[1, 0, 1] = float('inf')
        base[0, 0, runner.p.r.NOT_UPPER[0]] = -0.
        bases[f'{seed}/base'] = base
        for mode in ('full', 'local_static'):
            upper = torch.randn(2, 5, 9, generator=generator)
            upper[~valid] = float('nan')
            generated[f'{seed}/{mode}'] = runner.p.compose_upper_face(base, upper, valid)
    return {**common, 'predictions': generated}, {**common, 'predictions': bases}


def test_compact_roundtrip_keeps_seed_specific_43_channels_and_all_invalid_bits():
    curves, bases = deployment_fixture()
    before = copy.deepcopy(curves)
    compact = runner.compact_deployment(curves, bases, Path('/bound/base.pt'), 'verified-sha')
    assert 'predictions' not in compact
    assert compact['baseline_path'] == str(Path('/bound/base.pt'))
    assert compact['baseline_sha256'] == 'verified-sha'
    assert all(v.shape == (2, 5, 9) for v in compact['upper_predictions9'].values())
    result = runner.restore_deployment(compact, bases, 'verified-sha')
    for key, expected in curves['predictions'].items():
        assert_bits(result['predictions'][key], expected)
        assert_bits(expected, before['predictions'][key])
        assert result['predictions'][key].data_ptr() != bases['predictions'][key.split('/')[0] + '/base'].data_ptr()


@pytest.mark.parametrize('corruption', ['nonupper', 'invalid_upper', 'signed_zero'])
def test_compaction_rejects_even_bit_level_protected_changes(corruption):
    curves, bases = deployment_fixture()
    pred = curves['predictions']['42/full']
    if corruption == 'nonupper':
        pred[0, 0, runner.p.r.NOT_UPPER[1]] += .01
    elif corruption == 'invalid_upper':
        pred[0, 1, runner.p.CC[0]] = 0
    else:
        pred[0, 0, runner.p.r.NOT_UPPER[0]] = +0.
    with pytest.raises(ValueError, match='Protected baseline'):
        runner.compact_deployment(curves, bases, 'base.pt', 'sha')


@pytest.mark.parametrize('corruption', ['hash', 'schema', 'channels', 'clip_order', 'clock', 'target'])
def test_restore_rejects_wrong_base_binding_and_native_metadata(corruption):
    curves, bases = deployment_fixture()
    compact = runner.compact_deployment(curves, bases, 'base.pt', 'sha')
    altered = copy.deepcopy(compact)
    if corruption == 'hash':
        altered['baseline_sha256'] = 'different-file'
    elif corruption == 'schema':
        altered['storage_schema'] = 'other'
    elif corruption == 'channels':
        altered['upper_indices'] = list(reversed(altered['upper_indices']))
    elif corruption == 'clip_order':
        altered['clip_id'].reverse()
    elif corruption == 'clock':
        altered['times'][0, 2] += .001
    else:
        altered['target'][0, 2, 0] += .01
    with pytest.raises(ValueError, match='Compact base binding|Cross-run'):
        runner.restore_deployment(altered, bases, 'sha')


def synthetic_historical_partition():
    sentence_names = ['s0', 's1', 's2', 's3', 's4']
    train_groups, hold_groups = runner.sentence_split(sentence_names, runner.SPLIT_SEED)
    counts = {sentence_names[int(hold_groups[0])]: 608}
    counts.update({sentence_names[int(i)]: n for i, n in zip(train_groups, (426, 426, 427, 428))})
    sentences = [name for name in sentence_names for _ in range(counts[name])]
    q = {'sentence_id': sentences, 'clip_id': [f'clip{i}' for i in range(2315)],
         'valid': torch.ones(2315, 1, dtype=torch.bool), 'motion': object()}
    fit, hold = runner.sentence_split(sentences, runner.SPLIT_SEED)
    split = {name: {'clips': [q['clip_id'][int(i)] for i in ids],
                    'sentences': sorted({str(sentences[int(i)]) for i in ids})}
             for name, ids in (('fit', fit), ('internal_sentence_holdout', hold))}
    return q, {'split': split}


def test_partition_is_metadata_only_and_excludes_entire_sentences_from_new_updates():
    q, historical = synthetic_historical_partition()
    fit, hold, record = runner.partition(q, historical)
    assert (len(fit), len(hold)) == (1707, 608)
    assert sorted(fit.tolist() + hold.tolist()) == list(range(2315))
    assert not set(q['sentence_id'][i] for i in fit) & set(q['sentence_id'][i] for i in hold)
    assert record['uses_motion_for_selection'] is False and record['test_loaded'] is False
    assert record['role'] == 'Held out from NEW updates only; inherited sources saw the full fit pool'
    assert record['split'] == historical['split']


@pytest.mark.parametrize('corruption', ['historical_clip_order', 'historical_sentence', 'pool_size'])
def test_partition_rejects_silent_resplits_or_changed_fit_pool(corruption):
    q, historical = synthetic_historical_partition()
    if corruption == 'historical_clip_order':
        historical['split']['fit']['clips'].reverse()
    elif corruption == 'historical_sentence':
        historical['split']['fit']['sentences'].append('new-sentence')
    else:
        q['valid'] = q['valid'][:-1]
    with pytest.raises(ValueError, match='Historical metadata|original 2315'):
        runner.partition(q, historical)


@pytest.mark.parametrize('arm', runner.ARMS)
def test_cache_uses_final_native_audio_path_without_changing_source_cache(arm):
    source, q = source_fixture()
    local, adapter = runner.configure(arm, source, 'cpu')
    if arm == 'rank8_adapter':
        with torch.no_grad():
            adapter.up.weight.copy_(torch.linspace(-.1, .2, 512).reshape(64, 8))
    elif arm == 'full_local':
        with torch.no_grad():
            local.local_head.bias.add_(.1)
    before = q['prefix_local'].clone()
    expected = runner.conditioned_features(local, adapter, q, arm).detach()
    cached, rows = runner.cache_condition(local, adapter, q, SimpleNamespace(batch_size=2, device='cpu'), arm)
    # Batched cache and full-batch GEMMs may choose different kernels; use
    # the same2e-6 numerical bound as the real source-feature replay.
    torch.testing.assert_close(cached['prefix_local'], expected, atol=2e-6, rtol=1e-5)
    assert_bits(q['prefix_local'], before)
    assert not cached['prefix_local'].requires_grad
    assert [r['clip_id'] for r in rows] == q['clip_id']
    assert [r['sentence_id'] for r in rows] == q['sentence_id']
    for i, row in enumerate(rows):
        observed = q['valid'][i]
        delta = (cached['prefix_local'][i, observed] - before[i, observed]).double()
        assert row['correction_rms'] == pytest.approx(float(delta.square().mean().sqrt()))
        assert row['mean_correction_max_abs'] == pytest.approx(float(delta.mean(0).abs().max()))
        if arm == 'rank8_adapter':
            assert row['mean_correction_max_abs'] < 2e-7
            assert row['same_forward_mean_correction_max_abs'] < 2e-7
            assert row['centered_correction_rms'] > 0
        elif arm == 'full_local':
            assert row['mean_correction_max_abs'] > .09
        else:
            assert row['correction_rms'] < 1e-7
    assert cached['prefix_local'][~q['valid']].count_nonzero() == 0
