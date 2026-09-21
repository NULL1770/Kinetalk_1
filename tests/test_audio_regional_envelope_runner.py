import json

import pytest
import torch

from kinetalk_b0.models.audio_regional_envelope import AudioRegionalEnvelope
from scripts import train_audio_regional_envelope as runner


def cache_fixture(n=24, frames=15, features=4):
    generator = torch.Generator().manual_seed(41)
    valid = torch.ones(n, frames, dtype=torch.bool)
    valid[:, 6] = False
    time = torch.arange(frames).float()[None, :, None]
    target = .5 + .12 * torch.sin(time * .45 + torch.arange(52)[None, None, :] * .3)
    target = target.expand(n, -1, -1).clone()
    base = .5 + .08 * torch.cos(time * .35 + torch.arange(52)[None, None, :] * .3)
    base = base.expand(n, -1, -1).clone()
    upper = base[..., list(runner.UPPER_INDICES)]
    mean = torch.where(valid[..., None], upper, 0.).sum(1) / valid.sum(1, keepdim=True)
    return {'features': torch.randn(n, frames, features, generator=generator),
            'target_full': target, 'base_full': base, 'valid': valid,
            'original_valid': valid, 'channel_mask': torch.ones(n, 52, dtype=torch.bool),
            'emotion': torch.arange(n) % 2, 'mean': mean,
            'speaker': ['s' + str(i // 6) for i in range(n)],
            'sentence_id': ['t' + str(i % 6) for i in range(n)],
            'clip_id': ['c' + str(i) for i in range(n)],
            'times': torch.arange(frames).double()[None].expand(n, -1) / 30,
            'b0': base.clone()}


def test_fit_statistics_and_scales_ignore_calibration_data():
    cache = cache_fixture()
    fit = torch.arange(12)
    stats = runner._fit_stats(cache, fit, 3)
    prepared, stats = runner._prepare_cache(cache, stats, fit, 3)
    contaminated = {**cache, 'features': cache['features'].clone(), 'target_full': cache['target_full'].clone()}
    contaminated['features'][12:] *= 10000
    contaminated['target_full'][12:] *= 10000
    stats2 = runner._fit_stats(contaminated, fit, 3)
    prepared2, stats2 = runner._prepare_cache(contaminated, stats2, fit, 3)
    for key in stats:
        torch.testing.assert_close(stats[key], stats2[key], rtol=0, atol=0)
    torch.testing.assert_close(prepared['target_env'][fit], prepared2['target_env'][fit], rtol=0, atol=0)
    assert runner.WINDOW == 5


def test_mismatch_never_selects_self_and_preserves_recipient_mask():
    cache = cache_fixture()
    # Unique durations cannot force fallback to the recipient.
    cache['valid'][0, -3:] = False
    donors = runner._donors(cache)
    assert (donors != torch.arange(len(donors))).all()
    for i in range(len(donors)):
        cache['features'][i].fill_(float(i + 1))
    output = runner._mismatch_features(cache, torch.arange(len(donors)), donors)
    for i, donor in enumerate(donors):
        torch.testing.assert_close(output[i, cache['valid'][i]], torch.full_like(output[i, cache['valid'][i]], float(donor + 1)))
    assert (output[~cache['valid']] == 0).all()


def test_prediction_is_batched_deterministic_and_eval():
    cache = cache_fixture(n=4)

    class Checked(AudioRegionalEnvelope):
        def forward(self, features, valid):
            assert not self.training
            assert len(features) <= 2
            return super().forward(features, valid)

    model = Checked(torch.zeros(4), torch.ones(4), hidden=8)
    torch.nn.init.normal_(model.head.weight)
    model.train()
    ids = torch.arange(4)
    a = runner._predict(model, cache, ids, 'cpu', 2, 'shuffle')
    b = runner._predict(model, cache, ids, 'cpu', 1, 'shuffle')
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    assert not model.training


def test_constant_correlation_undefined_and_static_gain_is_constant():
    cache = cache_fixture(n=4)
    stats = runner._fit_stats(cache, torch.arange(4), 2)
    cache, stats = runner._prepare_cache(cache, stats, torch.arange(4), 2)
    target = cache['target_env']
    constant = runner._static_envelope(target, cache['valid'])
    metrics = runner._envelope_metrics(constant, target, cache['valid'])
    assert metrics['correlation'] is None
    predictions, _, protection, gains = runner._compose_conditions({'full': target}, cache, stats, 2)
    assert protection['full']['nonupper43_bit_exact']
    assert gains['full']['min'] >= runner.GAIN_MIN
    # A constant region gain scales the entire prior waveform by one factor.
    before = runner._center(cache['base_full'][..., list(runner.UPPER_INDICES)], cache['valid'])
    after = runner._center(predictions['static_gain'][..., list(runner.UPPER_INDICES)], cache['valid'])
    for row in range(4):
        x, y = before[row, cache['valid'][row], 0], after[row, cache['valid'][row], 0]
        factor = (x*y).sum() / x.square().sum()
        torch.testing.assert_close(y, x * factor, rtol=1e-4, atol=1e-6)


def test_cluster_ci_requires_more_than_one_cluster():
    report = runner._paired_cluster_ci([0., 0.], [1., 1.], ['s', 's'], ['a', 'b'], draws=100)
    assert report['speaker_cluster']['ci95'] is None
    assert not report['speaker_cluster']['improvement_supported']
    assert report['sentence_cluster']['improvement_supported']


def test_light_metrics_match_selection_statistics_without_lag_scan():
    cache = cache_fixture(n=4)
    stats = runner._fit_stats(cache, torch.arange(4), 2)
    cache, _ = runner._prepare_cache(cache, stats, torch.arange(4), 2)
    target = cache['target_env']
    pred = .2 + .7 * target
    full = runner._envelope_metrics(pred, target, cache['valid'])
    light = runner._envelope_metrics(pred, target, cache['valid'], light=True)
    for key in ('mse', 'centered_mse', 'correlation', 'temporal_std', 'target_temporal_std'):
        assert light[key] == full[key]
    assert not light['lag_evaluated']
    assert light['cross_correlation_peak_abs_lag_frames'] is None


def test_smooth_carrier_preserves_original_and_raw_target():
    cache = cache_fixture(n=4)
    original = cache['base_full'].clone()
    raw_target = cache['target_full'].clone()
    changed = runner._apply_carrier(cache, 5, 2)
    torch.testing.assert_close(changed['original_base_full'], original, rtol=0, atol=0)
    torch.testing.assert_close(changed['target_full'], raw_target, rtol=0, atol=0)
    assert not torch.equal(changed['base_full'], original)
    torch.testing.assert_close(changed['mean'], cache['mean'], rtol=0, atol=1e-6)
    stats = runner._fit_stats(changed, torch.arange(4), 2)
    prepared, stats = runner._prepare_cache(changed, stats, torch.arange(4), 2)
    report, curves = runner._evaluate({'oracle_envelope': prepared['target_env']}, prepared, stats, 2, oracle=True)
    assert 'original_prior' in report['composed_envelope_metrics']
    torch.testing.assert_close(curves['predictions']['original_prior'], original, rtol=0, atol=0)
    assert curves['postprocessed'] and curves['carrier_smoothing']['window_frames'] == 5
    assert not curves['carrier_smoothing']['raw_target_smoothed']


@pytest.mark.parametrize('mode', ['oracle', 'smoke', 'train'])
def test_runner_end_to_end_with_synthetic_context(tmp_path, monkeypatch, mode):
    cache = cache_fixture()
    q = {'valid': cache['valid'], 'speaker': cache['speaker'], 'sentence_id': cache['sentence_id'],
         'clip_id': cache['clip_id']}
    data = {'splits': {'train': q, 'validation': q}, 'provenance': {'manifest_sha256': 'synthetic'}}
    monkeypatch.setattr(runner, 'load_context', lambda *args: (data, None, None, None, None))
    calls = []

    def fake_cache(data, system, audio, identities, split, ids, device, seed, batch_size):
        calls.append(split)
        return runner._take_cache(cache, ids)

    monkeypatch.setattr(runner, '_cache_source', fake_cache)
    source = tmp_path / 'source.pt'; source.write_bytes(b'synthetic source')
    out = tmp_path / mode
    argv = ['runner', '--data', str(tmp_path), '--source', str(source), '--output', str(out),
            '--device', 'cpu', '--epochs', '1', '--batch-size', '4', '--smoke-epochs', '1']
    if mode == 'oracle':
        argv.append('--oracle-only')
    elif mode == 'smoke':
        argv.append('--smoke')
    monkeypatch.setattr('sys.argv', argv)
    runner.main()
    status = json.loads((out / 'status.json').read_text())
    assert not status['paper_success_established']
    if mode in ('oracle', 'smoke'):
        assert calls == ['train']
    else:
        assert calls == ['train', 'validation']
        result = json.loads((out / 'evaluation.json').read_text())
        assert result['protection']['full']['nonupper43_bit_exact']
        assert 'static_gain' in result['composed_envelope_metrics']
        saved = torch.load(out / 'native_curves.pt', weights_only=False)
        assert saved['predictions']['full'].shape[-1] == 52
        assert saved['native_rate'] and not saved['postprocessed']
