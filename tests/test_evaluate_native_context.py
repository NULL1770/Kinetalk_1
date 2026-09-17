"""Common-clock, target isolation, and baseline protection for native rollout."""
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts import evaluate_native_context as e


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Store:
    def __init__(self):
        gen = torch.Generator().manual_seed(337)
        self.native_lengths = torch.tensor([121, 57, 144])
        self.center_starts = torch.tensor([12, 0, 24])
        self.clip_ids = ['long_a', 'short_b', 'long_c']
        mask = torch.arange(144)[None] < self.native_lengths[:, None]
        mask[0, 28:31] = False
        mask[1, 14:18] = False
        mask[2, 75:81] = False
        h0 = torch.randn(3, 144, 4, generator=gen).half()
        audio = torch.randn(3, 144, 1540, generator=gen)
        motion = (torch.randn(3, 144, 52, generator=gen)*.1+.4).half()
        # Invalid acoustic slots must never contaminate generation.
        h0[~mask] = float('nan')
        audio[~mask] = float('nan')
        times = torch.arange(144, dtype=torch.float64)[None].expand(3, -1)*.04
        center = lambda value: e.extract_center(value, self.center_starts)
        self.q = {'motion': center(motion), 'valid': center(mask),
            'times': center(times), 'channel_mask': torch.ones(3, 52, dtype=torch.bool),
            'clip_id': list(self.clip_ids), 'sentence_id': ['s0', 's1', 's2'],
            'speaker_id': torch.tensor([0, 1, 0]), 'emotion_id': torch.tensor([0, 1, 2]),
            'h0': center(h0), 'audio_features': center(audio),
            'audio_global': torch.randn(3, 64, generator=gen),
            'audio_intensity': torch.ones(3, 1),
            'static_upper': (torch.randn(3, 9, generator=gen)*.03+.4).half(),
            'b0': torch.randn(3, 96, 52, generator=gen)}
        self.full = {'valid': mask, 'times': times, 'audio_features': audio,
                     'motion': motion, 'emotion_id': self.q['emotion_id']}
        self.encoded = [{'h0': h0[i, :n].clone(),
                         **{k: self.q[k][i].clone() for k in
                            ('audio_global', 'audio_intensity', 'static_upper')}}
                        for i, n in enumerate(self.native_lengths)]

    def batch(self, ids, mode, device):
        rows = self.q if mode == 'center' else self.full
        frames = ((int(self.native_lengths[ids].max())+15)//16)*16
        out = {}
        for key, value in rows.items():
            if torch.is_tensor(value):
                value = value[ids].clone()
                if mode == 'full' and key in ('valid', 'times', 'audio_features', 'motion'):
                    value = value[:, :frames]
                out[key] = value.to(device)
            else:
                out[key] = [value[int(i)] for i in ids]
        return out


def setup():
    store = Store()
    models = [SimpleNamespace(training=False) for _ in range(3)]
    identities = {sid: {'code': torch.tensor([[float(sid), .2]]),
                        'baseline': torch.zeros(1, 52)} for sid in (0, 1)}
    return (*models, store, identities, torch.arange(1, 10).float()*.03,
            SimpleNamespace(batch_size=2, device='cpu'))


def patch(monkeypatch, calls=None):
    def local(module, audio, valid):
        assert audio.dtype == torch.float32
        assert torch.isfinite(audio).all()
        return audio[..., :64]

    def decoder(upper, b, identity, native, noise, *, steps, mode, arm):
        assert set(b) == set(e.CONDITION_KEYS)
        assert set(identity) == {'code'}
        assert not torch.is_grad_enabled()
        assert mode == 'full' and arm == 'chunk_teacher' and steps == 12
        assert all(torch.isfinite(value).all() for value in b.values())
        assert b['h0'].dtype == torch.float32
        if calls is not None:
            calls.append({k: v.clone() for k, v in b.items()} |
                         {'noise': noise.clone(), 'native': native.clone()})
        return native[..., :9]*.3+noise*.02+b['h0'][..., :1]*.05+b['audio_global'][:, None, :1]*.01

    monkeypatch.setattr(e.adaptation, 'local_features', local)
    monkeypatch.setattr(e.context, 'decode_context', decoder)


def evaluate(args, *, mode='full', bases=None):
    return e.evaluate_native(*args, 12, mode=mode, encoded_full=args[3].encoded, bases=bases)


def bases_for(store):
    q = store.q
    bases = {key: copy.deepcopy(q[source]) for key, source in
             (('target', 'motion'), ('valid', 'valid'), ('channel_mask', 'channel_mask'),
              ('times', 'times'), ('b0', 'b0'), ('emotion_id', 'emotion_id'),
              ('speaker_id', 'speaker_id'), ('clip_id', 'clip_id'))}
    bases['target'] = bases['target'].float()
    bases['noise_seeds'] = list(e.SEEDS)
    bases['predictions'] = {}
    for seed in e.SEEDS:
        value = torch.randn(3, 96, 52, generator=torch.Generator().manual_seed(seed))
        value[~q['valid']] = float('nan')
        bases['predictions'][f'{seed}/base'] = value
    return bases


def test_absolute_noise_and_common_center_match_across_contexts(monkeypatch):
    args = setup()
    args[-1].batch_size = 3
    calls = []
    patch(monkeypatch, calls)
    before_rng = torch.random.get_rng_state().clone()
    _, full = evaluate(args)
    _, center = evaluate(args, mode='center')
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert len(calls) == 10
    for i in range(5):
        assert torch.equal(e.extract_center(calls[i]['noise'], args[3].center_starts), calls[5+i]['noise'])
    for index in (0, 3, 4):
        seed = e.SEEDS[(0, 3, 4).index(index)]
        assert torch.equal(full['upper_predictions9'][f'{seed}/full'], center['upper_predictions9'][f'{seed}/full'])
    # Intervention changes only native local conditions, with matched noise.
    for i in (1, 2):
        for key in (*e.CONDITION_KEYS, 'noise'):
            assert torch.equal(calls[0][key], calls[i][key])
    for row in range(3):
        observed = calls[0]['valid'][row]
        static = calls[1]['native'][row, observed]
        torch.testing.assert_close(static, static[:1].expand_as(static))
        assert torch.equal(calls[2]['native'][row, observed], calls[0]['native'][row, observed].flip(0))
    assert full['population_noise_draw_frames'] == center['population_noise_draw_frames'] == 144


@pytest.mark.parametrize('mode', ['center', 'full'])
def test_generation_is_target_label_and_batch_size_independent(monkeypatch, mode):
    args = setup()
    patch(monkeypatch)
    _, before = evaluate(args, mode=mode)
    args[3].q['motion'].add_(300.)
    args[3].q['emotion_id'] = torch.tensor([2, 0, 1])
    args[3].full['motion'].fill_(800.)
    args[3].full['emotion_id'] = torch.tensor([1, 1, 1])
    args[-1].batch_size = 1
    _, after = evaluate(args, mode=mode)
    for key in before['upper_predictions9']:
        assert torch.equal(before['upper_predictions9'][key], after['upper_predictions9'][key])
    assert not torch.equal(before['target_upper9'], after['target_upper9'])


def test_common_dc_archive_and_protected43_preserve_seed_specific_base(monkeypatch):
    args = setup()
    patch(monkeypatch)
    bases = bases_for(args[3])
    original = copy.deepcopy(bases)
    report, curves = evaluate(args, bases=bases)
    assert report['nonupper_protection_checked']
    assert len(report['protection_checks']) == 10
    assert report['primary_composition'] == 'raw' and report['secondary_composition'] == 'dc'
    assert not report['GT_was_input'] and not report['oracle_modes']
    assert not report['fullface_deployment_evaluation'] and not report['nonupper_scored']
    assert not curves['dc_saved'] and 'predictions' not in curves and 'b0' not in curves
    for key, raw in curves['upper_predictions9'].items():
        dc = e.compose_dc_upper(raw, curves['static_used'], curves['valid'])
        assert raw[~curves['valid']].count_nonzero() == 0
        for row, valid in enumerate(curves['valid']):
            torch.testing.assert_close(dc[row, valid].mean(0), curves['static_used'][row], atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(dc[row, valid]-dc[row, valid].mean(0),
                                       raw[row, valid]-raw[row, valid].mean(0), atol=2e-7, rtol=1e-5)
        assert report['dc_invariants'][key]['valid_mean_max_abs_error'] < 2e-6
        for group in e.GROUPS:
            a = report['compositions']['raw']['modes'][key]['paired'][group]
            b = report['compositions']['dc']['modes'][key]['paired'][group]
            assert a['centered_mse'] == pytest.approx(b['centered_mse'], abs=1e-8)
        baseline = bases['predictions'][key.split('/')[0]+'/base']
        result = e.p.compose_upper_face(baseline, raw, curves['valid'])
        assert e._same_bits(result[..., list(e.p.r.NOT_UPPER)], baseline[..., list(e.p.r.NOT_UPPER)])
        assert e._same_bits(result[~curves['valid']], baseline[~curves['valid']])
    for key, value in bases['predictions'].items():
        assert e._same_bits(value, original['predictions'][key])


def test_baseline_rejects_real_numeric_mismatch_and_protection_detects_mutation(monkeypatch):
    args = setup()
    patch(monkeypatch)
    bases = bases_for(args[3])
    # Exact FP16 -> FP32 promotion is supported, but any numerical change is not.
    e._validate_bases(args[3].q, bases)
    bases['target'][0, 0, 0] += 1e-7
    with pytest.raises(ValueError, match='Baseline native metadata differs: target'):
        evaluate(args, bases=bases)
    bases = bases_for(args[3])
    compose = e.p.compose_upper_face
    def corrupt(baseline, upper, valid):
        out = compose(baseline, upper, valid)
        out[..., e.p.r.NOT_UPPER[0]] += .01
        return out
    monkeypatch.setattr(e.p, 'compose_upper_face', corrupt)
    with pytest.raises(RuntimeError, match='protected43'):
        evaluate(args, bases=bases)


def test_decoder_seams_use_native_offset_and_do_not_bridge_gaps():
    valid = torch.ones(1, 96, dtype=torch.bool)
    valid[0, 19] = False  # Exclude native31 -> native32.
    target = torch.zeros(1, 96, 9)
    pred = torch.zeros_like(target)
    pred[:, 4:] = 1.  # Native15 -> native16 is a decoder seam.
    starts = torch.tensor([12])
    full = e.actual_decoder_seams(pred, target, valid, starts, mode='full')
    center = e.actual_decoder_seams(pred, target, valid, starts, mode='center')
    assert full['observed_seam_pairs'] == 5
    assert center['observed_seam_pairs'] == 5
    for group in e.GROUPS:
        assert full['groups'][group]['seam']['displacement_mse'] == pytest.approx(1/5)
        assert center['groups'][group]['seam']['displacement_mse'] == 0
        assert full['groups'][group]['nonseam']['displacement_mse'] == 0


@pytest.mark.parametrize('corrupt', ['native_mask', 'native_time', 'missing_encoded', 'nonfinite_encoded'])
def test_full_source_misalignment_rejected_before_decoding(monkeypatch, corrupt):
    args = setup()
    patch(monkeypatch)
    store = args[3]
    if corrupt == 'native_mask':
        store.full['valid'][0, 12] = False
    elif corrupt == 'native_time':
        store.full['times'] = store.full['times']+.01
    elif corrupt == 'missing_encoded':
        store.encoded[0].pop('audio_global')
    else:
        store.encoded[0]['h0'][12] = float('nan')
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid native reference reached decoder')
    monkeypatch.setattr(e.context, 'decode_context', forbidden)
    with pytest.raises(ValueError):
        evaluate(args)


def test_real_decoder_handles_native_prefix_long_clip_and_short_padding(monkeypatch):
    from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
    args = list(setup())
    args[0] = PrefixUpperFlow({'model': {'content_dim': 4, 'emotion_dim': 64,
        'style_dim': 2, 'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.}}).eval()
    monkeypatch.setattr(e.adaptation, 'local_features', lambda local, audio, valid: audio[..., :64])
    report, curves = evaluate(args)
    assert report['clips'] == 3 and report['decode_steps'] == 12
    for value in curves['upper_predictions9'].values():
        assert value.shape == (3, 96, 9) and torch.isfinite(value).all()
        assert value[~curves['valid']].count_nonzero() == 0

