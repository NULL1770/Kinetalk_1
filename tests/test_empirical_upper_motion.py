import io

import pytest
import torch

from kinetalk_b0.models.empirical_upper_motion import fit_empirical_bank, sample_empirical_motion


def fit_fixture(dtype=torch.float64):
    frames = 21
    clock = torch.arange(frames, dtype=dtype)[None, :, None]
    channel = torch.arange(9, dtype=dtype)[None, None]
    phase = torch.arange(4, dtype=dtype)[:, None, None]
    upper = .4 + .12*torch.sin(clock*.43+channel*.37+phase)
    valid = torch.ones(4, frames, dtype=torch.bool)
    valid[0, 5:8] = False
    valid[1, 19:] = False
    upper[~valid] = float('nan')
    global_code = torch.arange(4, dtype=dtype)[:, None].expand(4, 64).clone()
    identity = torch.arange(4, dtype=dtype)[:, None].expand(4, 128).clone()*.3
    speakers = ['speaker0', 'speaker1', 'speaker2', 'speaker3']
    sentences = ['sentence0', 'sentence1', 'sentence2', 'sentence3']
    bank = fit_empirical_bank(upper, valid, global_code, identity, speakers, sentences,
                              ['clip0', 'clip1', 'clip2', 'clip3'])
    return bank, upper, valid, global_code, identity


def query_fixture(dtype=torch.float64):
    center = torch.full((2, 14, 9), .45, dtype=dtype)
    valid = torch.ones(2, 14, dtype=torch.bool)
    valid[0, 4:7] = False
    valid[1, 10:] = False
    center[~valid] = float('nan')
    global_code = torch.full((2, 64), 1.8, dtype=dtype)
    identity = torch.full((2, 128), .55, dtype=dtype)
    return center, valid, global_code, identity


def bit_equal(left, right):
    integer = torch.int64 if left.dtype == torch.float64 else torch.int32
    return torch.equal(left.contiguous().view(integer), right.contiguous().view(integer))


def test_bank_native_runs_equal_clip_statistics_and_serializable():
    bank, upper, valid, global_code, identity = fit_fixture()
    assert bank['fps'] == 25 and bank['window'] == 5
    assert bank['run_lengths'].tolist() == [5, 13, 19, 21, 21]
    assert bank['run_starts'].tolist() == [0, 8, 0, 0, 0]
    assert bank['run_clip_indices'].tolist() == [0, 0, 1, 2, 3]
    assert len(bank['residuals']) == int(valid.sum())
    for key, code in (('global', global_code), ('identity', identity)):
        torch.testing.assert_close(bank['stats'][key+'_mean'], code.mean(0))
        torch.testing.assert_close(bank['stats'][key+'_std'], code.std(0, unbiased=False))
    for offset, length in zip(bank['run_offsets'], bank['run_lengths']):
        torch.testing.assert_close(bank['residuals'][offset:offset+length].mean(0), torch.zeros(9, dtype=upper.dtype),
                                   atol=2e-16, rtol=0)
    serialized = io.BytesIO()
    torch.save(bank, serialized)
    serialized.seek(0)
    recovered = torch.load(serialized, weights_only=True)
    assert recovered['clip_ids'] == bank['clip_ids']
    assert torch.equal(recovered['residuals'], bank['residuals'])


def test_bank_smoothing_and_centering_do_not_cross_gaps():
    bank, upper, valid, global_code, identity = fit_fixture()
    for run in range(len(bank['run_lengths'])):
        row = int(bank['run_clip_indices'][run])
        left = int(bank['run_starts'][run])
        length = int(bank['run_lengths'][run])
        isolated = fit_empirical_bank(upper[row:row+1, left:left+length],
            torch.ones(1, length, dtype=torch.bool), global_code[row:row+1], identity[row:row+1],
            ['x'], ['y'], ['independent'])
        offset = int(bank['run_offsets'][run])
        torch.testing.assert_close(bank['residuals'][offset:offset+length], isolated['residuals'], atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_bounds_invalid_exact_and_zero_temperature_exact(dtype):
    bank, *_ = fit_fixture(dtype)
    center, valid, global_code, identity = query_fixture(dtype)
    center[:, 0, :3] = torch.tensor([0., .5, 1.], dtype=dtype)
    zero, metadata = sample_empirical_motion(bank, center, valid, global_code, identity,
        [None, None], [None, None], seeds=[3, 4], temperature=0.)
    assert bit_equal(zero, center[None].expand_as(zero))
    assert metadata == [[[], []], [[], []]]
    samples, _ = sample_empirical_motion(bank, center, valid, global_code, identity,
        [None, None], [None, None], seeds=[3, 4], temperature=100.)
    assert samples.shape == (2, 2, 14, 9)
    assert torch.isfinite(samples[:, valid]).all()
    assert ((samples[:, valid] >= 0) & (samples[:, valid] <= 1)).all()
    assert bit_equal(samples[:, ~valid], center[None].expand_as(samples)[:, ~valid])


def test_seed_reproducibility_without_global_rng_changes():
    bank, *_ = fit_fixture()
    center, valid, global_code, identity = query_fixture()
    state = torch.random.get_rng_state().clone()
    first, first_metadata = sample_empirical_motion(bank, center, valid, global_code, identity,
        [None, None], [None, None], seeds=[123, 456, 123], temperature=.8)
    assert torch.equal(state, torch.random.get_rng_state())
    assert bit_equal(first[0], first[2]) and first_metadata[0] == first_metadata[2]
    alone, metadata = sample_empirical_motion(bank, center, valid, global_code, identity,
        [None, None], [None, None], seeds=[456], temperature=.8)
    assert bit_equal(first[1], alone[0]) and first_metadata[1] == metadata[0]
    assert not bit_equal(first[0], first[1])


def test_unconditional_is_condition_independent():
    bank, *_ = fit_fixture()
    center, valid, global_code, identity = query_fixture()
    first, metadata = sample_empirical_motion(bank, center, valid, global_code, identity,
        [None, None], [None, None], seeds=[7], temperature=.8, mode='unconditional')
    changed, changed_metadata = sample_empirical_motion(bank, center, valid, global_code+1e5, identity-1e4,
        [None, None], [None, None], seeds=[7], temperature=.8, mode='unconditional', top_k=1)
    assert bit_equal(first, changed) and metadata == changed_metadata
    for sample in metadata:
        for runs in sample:
            assert all(run['condition_distance'] is None and run['conditional_rank'] is None for run in runs)


def test_conditional_top_one_responds_to_condition_and_excludes_or():
    bank, *_ = fit_fixture()
    center = torch.full((1, 10, 9), .5, dtype=torch.float64)
    valid = torch.ones(1, 10, dtype=torch.bool)
    global_code = torch.full((1, 64), 1., dtype=torch.float64)
    identity = torch.full((1, 128), .3, dtype=torch.float64)
    _, match = sample_empirical_motion(bank, center, valid, global_code, identity,
        [None], [None], seeds=[4], temperature=1., top_k=1)
    assert match[0][0][0]['donor_clip_id'] == 'clip1'
    _, changed = sample_empirical_motion(bank, center, valid, global_code+2., identity+.6,
        [None], [None], seeds=[4], temperature=1., top_k=1)
    assert changed[0][0][0]['donor_clip_id'] == 'clip3'
    # Exclude speaker1 OR sentence2: neither donor can enter either sampling pool.
    for mode in ('conditional', 'unconditional'):
        _, metadata = sample_empirical_motion(bank, center, valid, global_code, identity,
            ['speaker1'], ['sentence2'], seeds=list(range(30)), temperature=1., top_k=4, mode=mode)
        for sample in metadata:
            donor = sample[0][0]
            assert donor['donor_speaker'] != 'speaker1'
            assert donor['donor_sentence'] != 'sentence2'
            assert donor['eligible_run_count'] == 2


def test_native_crop_exact_same_offsets_for_all_channels_and_no_recenter():
    bank, *_ = fit_fixture()
    center, valid, global_code, identity = query_fixture()
    samples, metadata = sample_empirical_motion(bank, center, valid, global_code, identity,
        [None, None], [None, None], seeds=[7, 13], temperature=.9)
    any_nonzero_crop_mean = False
    for sample, rows in enumerate(metadata):
        for row, runs in enumerate(rows):
            for record in runs:
                run = record['donor_run_index']
                length = record['query_length']
                offset = record['crop_offset']
                assert 0 <= offset <= record['donor_run_length']-length
                assert record['native_start'] == record['donor_run_start']+offset
                packed = int(bank['run_offsets'][run])+offset
                residual = bank['residuals'][packed:packed+length]
                delta = residual*.9
                left = record['query_start']
                base = center[row, left:left+length]
                room = torch.where(delta >= 0, 1-base, base)
                expected = base + room*torch.tanh(delta/room.clamp_min(torch.finfo(base.dtype).eps))
                torch.testing.assert_close(samples[sample, row, left:left+length], expected, atol=0, rtol=0)
                any_nonzero_crop_mean |= bool(residual.mean(0).abs().max() > 1e-6)
    assert any_nonzero_crop_mean


def test_missing_sufficient_length_or_exclusion_raises_without_fallback():
    bank, *_ = fit_fixture()
    center = torch.full((1, 22, 9), .5, dtype=torch.float64)
    valid = torch.ones(1, 22, dtype=torch.bool)
    global_code, identity = torch.zeros(1, 64), torch.zeros(1, 128)
    with pytest.raises(ValueError, match='No eligible native donor run'):
        sample_empirical_motion(bank, center, valid, global_code, identity, [None], [None], [3], 1.)
    one = fit_empirical_bank(center[:, :10], valid[:, :10], global_code, identity, ['same'], ['s'], ['only'])
    with pytest.raises(ValueError, match='exclusions'):
        sample_empirical_motion(one, center[:, :5], valid[:, :5], global_code, identity, ['same'], [None], [3], 1.)
    # The zero-temperature baseline never requires a donor and remains an exact copy.
    zero, _ = sample_empirical_motion(one, center, valid, global_code, identity, ['same'], [None], [3], 0.)
    assert torch.equal(zero[0], center)


@pytest.mark.parametrize('failure', ['temperature', 'negative_temperature', 'seeds', 'top_k', 'mode', 'center', 'bank'])
def test_invalid_sampling_contract_rejected(failure):
    bank, *_ = fit_fixture()
    center, valid, global_code, identity = query_fixture()
    kwargs = dict(seeds=[3], temperature=1., top_k=32, mode='conditional')
    if failure == 'temperature':
        kwargs['temperature'] = float('nan')
    elif failure == 'negative_temperature':
        kwargs['temperature'] = -1.
    elif failure == 'seeds':
        kwargs['seeds'] = []
    elif failure == 'top_k':
        kwargs['top_k'] = 0
    elif failure == 'mode':
        kwargs['mode'] = 'warped'
    elif failure == 'center':
        center[0, 0, 0] = 1.1
    else:
        bank['run_offsets'][1] += 1
    with pytest.raises(ValueError):
        sample_empirical_motion(bank, center, valid, global_code, identity,
            [None, None], [None, None], **kwargs)
