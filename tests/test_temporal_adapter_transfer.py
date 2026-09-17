"""Leakage, pairing, compact scoring, and DC contracts for transfer evaluation."""
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts import evaluate_temporal_adapter_transfer as evaluator


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    gen = torch.Generator().manual_seed(519)
    valid = torch.ones(3, 96, dtype=torch.bool)
    valid[0, 8:10] = False
    valid[1, :3] = False
    valid[1, 31:33] = False
    valid[2, -6:] = False
    q = {'valid': valid, 'motion': torch.randn(3, 96, 52, generator=gen)*.15+.4,
         'channel_mask': torch.ones(3, 52, dtype=torch.bool),
         'times': torch.arange(96, dtype=torch.float64)[None].expand(3, -1)*.04,
         'clip_id': ['clip_a', 'clip_b', 'clip_c'], 'sentence_id': ['sentence_a', 'sentence_b', 'sentence_a'],
         'emotion_id': torch.tensor([0, 1, 2]), 'speaker_id': torch.tensor([0, 1, 0]),
         'static_upper': torch.randn(3, 9, generator=gen)*.1+.4,
         'prefix_local': torch.randn(3, 96, 64, generator=gen),
         'h0': torch.randn(3, 96, 4, generator=gen),
         'audio_global': torch.randn(3, 3, generator=gen), 'audio_intensity': torch.ones(3, 1)}
    for key in ('motion', 'prefix_local', 'h0'):
        q[key][~valid] = float('nan')
    identities = {sid: {'code': torch.tensor([[float(sid), .2]]), 'baseline': torch.zeros(1, 52)} for sid in (0, 1)}
    return SimpleNamespace(training=False), q, identities, torch.arange(1, 10).float()*.03, SimpleNamespace(batch_size=2, device='cpu', smoke=True)


def decoder_stub(calls=None):
    def decode(upper, conditions, identity, native, noise, *, steps, mode, arm):
        assert set(conditions) == {'valid', 'h0', 'audio_global', 'audio_intensity'}
        assert set(identity) == {'code'}
        assert mode == 'full' and arm == 'chunk_teacher' and steps == 12
        assert not torch.is_grad_enabled()
        if calls is not None:
            calls.append({key: value.clone() for key, value in conditions.items()} | {'noise': noise.clone(), 'native': native.clone()})
        # A deterministic nonlinear conditional response; global/static, native
        # timing and noise can each be independently checked by the test.
        return native[..., :9]*.3+noise*.02+conditions['h0'][..., :1]*.05+conditions['audio_global'][:, None, :1]*.01
    return decode


def run(monkeypatch, **changes):
    upper, q, identities, scales, args = fixture()
    for key, value in changes.items():
        setattr(args, key, value)
    monkeypatch.setattr(evaluator.context, 'decode_context', decoder_stub())
    return evaluator.evaluate_transfer(upper, q, identities, scales, args, 12), q


def test_stripped_decoder_and_compact_outputs_have_no_oracle_or_fullface_claim(monkeypatch):
    (report, curves), q = run(monkeypatch)
    expected = {f'{seed}/full' for seed in evaluator.SEEDS} | {'42/local_static', '42/local_reverse'}
    assert set(curves['upper_predictions9']) == expected
    assert curves['target_upper9'].shape == (3, 96, 9)
    assert curves['channel_mask_upper'].shape == (3, 9)
    assert 'target' not in curves and 'predictions' not in curves and 'b0' not in curves
    assert not report['GT_was_input'] and not report['oracle_modes']
    assert not report['fullface_deployment_evaluation'] and not report['nonupper_protection_checked']
    assert not report['teacher_readout_scored'] and not report['nonupper_scored']
    for composition in report['compositions'].values():
        assert set(composition['modes']) == expected
        for row in composition['modes'].values():
            assert set(row['paired']) == {'brows', 'eyes_expression'}
            assert set(row['temporal']) == {'brows', 'eyes_expression'}
            assert set(row['boundaries']) == {'brows', 'eyes_expression'}
            assert len(row['per_clip']['brows']['raw_mse']) == 3
        for group in composition['distribution']['populations']['all']['groups'].values():
            for score in group['distribution'].values():
                assert score['trajectory_energy_score']['sample_count'] == 3
                assert score['adjacent_variogram_score']['sample_count'] == 3
    assert curves['sentence_id'] == q['sentence_id'] and curves['clip_id'] == q['clip_id']
    assert report['per_clip_order'] == q['clip_id']
    for value in curves['upper_predictions9'].values():
        assert value.shape == (3, 96, 9) and torch.isfinite(value).all()
        assert value[~q['valid']].count_nonzero() == 0


def test_paired_global_noise_and_local_interventions_preserve_other_conditions(monkeypatch):
    upper, q, identities, scales, args = fixture()
    args.batch_size = 3
    calls = []
    monkeypatch.setattr(evaluator.context, 'decode_context', decoder_stub(calls))
    random_before = torch.random.get_rng_state().clone()
    _, curves = evaluator.evaluate_transfer(upper, q, identities, scales, args, 12)
    assert torch.equal(torch.random.get_rng_state(), random_before)
    assert len(calls) == 5
    for call in calls[:3]:
        torch.testing.assert_close(call['noise'], calls[0]['noise'], atol=0, rtol=0)
        for key in evaluator.CONDITION_KEYS:
            torch.testing.assert_close(call[key], calls[0][key], atol=0, rtol=0, equal_nan=True)
    expected = torch.randn(3, 96, 9, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(calls[0]['noise'], expected, atol=0, rtol=0)
    assert not torch.equal(calls[3]['noise'], calls[0]['noise'])
    for row in range(3):
        observed = q['valid'][row]
        static = calls[1]['native'][row, observed]
        torch.testing.assert_close(static, static[0:1].expand_as(static))
        torch.testing.assert_close(calls[2]['native'][row, observed], calls[0]['native'][row, observed].flip(0))
    assert not torch.equal(curves['upper_predictions9']['42/full'], curves['upper_predictions9']['42/local_static'])


def test_batch_size_invariance_and_target_label_changes_never_change_generation(monkeypatch):
    upper, q, identities, scales, args = fixture()
    monkeypatch.setattr(evaluator.context, 'decode_context', decoder_stub())
    _, before = evaluator.evaluate_transfer(upper, q, identities, scales, args, 12)
    q['motion'][q['valid']] += 500.
    q['emotion_id'] = torch.tensor([3, 0, 2])
    args.batch_size = 1
    _, after = evaluator.evaluate_transfer(upper, q, identities, scales, args, 12)
    for key in before['upper_predictions9']:
        torch.testing.assert_close(before['upper_predictions9'][key], after['upper_predictions9'][key], atol=0, rtol=0)
    assert not torch.equal(before['target_upper9'], after['target_upper9'])


def test_dc_mean_dynamics_and_raw_scores_are_separate_and_inputs_unmodified(monkeypatch):
    upper, q, identities, scales, args = fixture()
    before = copy.deepcopy(q)
    monkeypatch.setattr(evaluator.context, 'decode_context', decoder_stub())
    report, curves = evaluator.evaluate_transfer(upper, q, identities, scales, args, 12)
    for key, value in q.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, before[key], atol=0, rtol=0, equal_nan=True)
        else:
            assert value == before[key]
    assert not curves['dc_saved']
    for key, raw in curves['upper_predictions9'].items():
        dc = evaluator.compose_dc_upper(raw, curves['static_upper'], curves['valid'])
        for row in range(3):
            mask = curves['valid'][row]
            torch.testing.assert_close(dc[row, mask].mean(0), curves['static_upper'][row], atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(dc[row, mask]-dc[row, mask].mean(0), raw[row, mask]-raw[row, mask].mean(0), atol=2e-7, rtol=1e-5)
        for name in evaluator.GROUPS:
            a = report['compositions']['raw']['modes'][key]['paired'][name]
            b = report['compositions']['dc']['modes'][key]['paired'][name]
            assert b['centered_mse'] == pytest.approx(a['centered_mse'], abs=1e-8)
            assert b['frame_displacement_mse'] == pytest.approx(a['frame_displacement_mse'], abs=1e-8)
        assert report['dc_invariants'][key]['valid_mean_max_abs_error'] <= 2e-6
    # Compact reference/metadata must be independent snapshots, not views that
    # silently change if the caller reuses its input tensors for a later arm.
    q['static_upper'].fill_(7.)
    assert not torch.equal(curves['static_upper'], q['static_upper'])


@pytest.mark.parametrize('corruption', ['sentence', 'observed_nan', 'mask', 'clock', 'scale', 'training', 'frames'])
def test_bad_transfer_contract_is_rejected_before_decode(monkeypatch, corruption):
    upper, q, identities, scales, args = fixture()
    if corruption == 'sentence':
        q.pop('sentence_id')
    elif corruption == 'observed_nan':
        q['prefix_local'][0, 0, 0] = float('nan')
    elif corruption == 'mask':
        q['channel_mask'][0, evaluator.p.CC[0]] = False
    elif corruption == 'clock':
        q['times'] = q['times']*2
    elif corruption == 'scale':
        scales[0] = 0.
    elif corruption == 'training':
        upper.training = True
    else:
        q['valid'] = q['valid'][:, :-1]
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid query reached decoder')
    monkeypatch.setattr(evaluator.context, 'decode_context', forbidden)
    with pytest.raises(ValueError):
        evaluator.evaluate_transfer(upper, q, identities, scales, args, 12)


def test_nonfinite_decoder_output_is_rejected(monkeypatch):
    upper, q, identities, scales, args = fixture()
    monkeypatch.setattr(evaluator.context, 'decode_context', lambda upper, b, ident, native, noise, **kw: torch.full_like(noise, float('nan')))
    with pytest.raises(ValueError, match='Decoder must return finite'):
        evaluator.evaluate_transfer(upper, q, identities, scales, args, 12)


def test_real_prefix_receiver_accepts_stripped_interface_with_generated_history():
    from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
    _, q, identities, scales, args = fixture()
    q = evaluator.p.r.subset(q, torch.tensor([0]), 'cpu')
    q['audio_global'] = torch.zeros(1, 64)
    upper = PrefixUpperFlow({'model': {'content_dim': 4, 'emotion_dim': 64,
        'style_dim': 2, 'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.}}).eval()
    report, curves = evaluator.evaluate_transfer(upper, q, identities, scales, args, 12)
    assert report['clips'] == 1 and report['decode_steps'] == 12
    for value in curves['upper_predictions9'].values():
        assert value.shape == (1, 96, 9) and torch.isfinite(value).all()
        assert value[~q['valid']].count_nonzero() == 0
