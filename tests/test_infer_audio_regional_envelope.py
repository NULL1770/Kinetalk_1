import json

import numpy as np
import pytest
import torch

from kinetalk_b0.models.audio_regional_envelope import AudioRegionalEnvelope
from scripts import infer_audio_regional_envelope as inference
from scripts import train_audio_regional_envelope as runner
from scripts.render_dynamic_rig_comparison import inspect_input


def fixture(tmp_path, carrier_window=5):
    generator = torch.Generator().manual_seed(12)
    frames, width = 19, 4
    valid = torch.ones(1, frames, dtype=torch.bool)
    valid[:, [0, 7, 8, 18]] = False
    phase = torch.arange(frames)[None, :, None] * .42 + torch.arange(52)[None, None] * .24
    prior = .5 + .12 * torch.sin(phase)
    prior[~valid] = float('nan')
    features = torch.randn(1, frames, width, generator=generator)
    features[~valid] = float('nan')
    stats = {'feature_mean': torch.tensor([.1, -.2, .3, .1]), 'feature_std': torch.tensor([.5, 1., 2., 1.5]),
             'channel_scales': torch.linspace(.02, .15, 9), 'envelope_scale': torch.tensor([.4, 1.6])}
    model = AudioRegionalEnvelope(stats['feature_mean'], stats['feature_std'], hidden=8).eval()
    with torch.no_grad():
        model.head.weight.copy_(torch.randn(2, 8, generator=generator) * .2)
    protocol = {'schema': 'native_regional_activity_v2', 'window_frames': 5, 'stride': 1,
                'gain_bounds': [.25, 2.5], 'gain_denominator_floor_normalized': .02,
                'carrier_smoothing': {'window_frames': carrier_window, 'applied': carrier_window > 1,
                                      'post_composition_smoothing': False}}
    checkpoint = {'model': model.state_dict(), 'stats': stats, 'protocol': protocol,
                  'protocol_sha256': runner.canonical_hash(protocol)}
    checkpoint_path = tmp_path / 'final.pt'
    torch.save(checkpoint, checkpoint_path)
    input_path = tmp_path / 'input.npz'
    np.savez(input_path, prior=prior[0].numpy(), audio_features=features[0].numpy(), valid=valid[0].numpy(),
             times=np.arange(frames, dtype=np.float64) / 25, channels=np.asarray(inference.ARKIT_NAMES),
             clip_id=np.asarray('synthetic'), noise_seed=np.asarray(12),
             # Reading any object target would fail under allow_pickle=False.
             motion=np.array([{'forbidden': True}], dtype=object),
             emotion=np.array([{'forbidden': True}], dtype=object),
             target=np.array([{'forbidden': True}], dtype=object))
    return checkpoint_path, input_path, checkpoint, prior, features, valid, model


@pytest.mark.parametrize('window', [1, 5, 9])
def test_inference_matches_runner_scaling_carrier_and_static_gain(tmp_path, window):
    checkpoint_path, input_path, checkpoint, prior, features, valid, model = fixture(tmp_path, window)
    payload, report = inference.infer(checkpoint_path, input_path)
    upper = prior[..., list(runner.UPPER_INDICES)]
    mean = torch.where(valid[..., None], upper, 0.).sum(1) / valid.sum(1, keepdim=True)
    cache = {'base_full': prior, 'valid': valid, 'mean': mean}
    cache = runner._apply_carrier(cache, window, 1)
    stats = checkpoint['stats']
    source_env = runner._envelopes(cache['base_full'], valid, stats['channel_scales'], 1) / stats['envelope_scale']
    with torch.no_grad():
        predicted = model(features, valid)
    gain = runner._gain_from_envelope(predicted, source_env, valid)
    static = runner._static_envelope(gain, valid)
    expected = {'original_prior': prior, 'prior': cache['base_full'],
                'full': runner.compose_prior_with_envelope(cache['base_full'], gain, cache['mean'], valid),
                'static_gain': runner.compose_prior_with_envelope(cache['base_full'], static, cache['mean'], valid)}
    for name, values in expected.items():
        actual = torch.from_numpy(payload['motions'][inference.MODE_NAMES.index(name)])[None]
        assert torch.equal(actual.contiguous().view(torch.int32), values.contiguous().view(torch.int32))
        assert report['protection'][name]['passed']
    np.testing.assert_array_equal(payload['predicted_envelope_normalized'], predicted[0].numpy())
    np.testing.assert_array_equal(payload['prior_envelope_normalized'], source_env[0].numpy())
    np.testing.assert_array_equal(payload['gain'], gain[0].numpy())
    np.testing.assert_array_equal(payload['static_gain'], static[0].numpy())
    assert not report['query_motion_read'] and not report['query_emotion_read'] and not report['query_target_read']
    assert report['gain']['min'] >= .25 and report['gain']['max'] <= 2.5


def test_cli_writes_renderer_compatible_npz_and_provenance(tmp_path, monkeypatch):
    checkpoint, source, *_ = fixture(tmp_path)
    output = tmp_path / 'animation.npz'
    monkeypatch.setattr('sys.argv', ['infer', '--checkpoint', str(checkpoint), '--input', str(source),
                                     '--output', str(output), '--device', 'cpu'])
    inference.main()
    channels, times, valid, modes, display, renderer_report = inspect_input(output, 25)
    assert channels == inference.ARKIT_NAMES and modes == list(inference.MODE_NAMES)
    assert display.shape == (4, len(times), 52) and np.isfinite(display).all()
    assert renderer_report['metadata']['clip_id'] == 'synthetic'
    report = json.loads(output.with_suffix('.report.json').read_text(encoding='utf8'))
    assert report['output_sha256'] == inference._sha(output)
    assert report['checkpoint_sha256'] == inference._sha(checkpoint)
    assert all(value['passed'] for value in report['protection'].values())
    with pytest.raises(FileExistsError):
        inference.main()


def test_partially_observed_upper_frames_remain_unchanged(tmp_path):
    checkpoint, path, _, prior, features, valid, _ = fixture(tmp_path)
    support = np.ones((prior.shape[1], 52), dtype=bool)
    support[4, runner.UPPER_INDICES[0]] = False
    np.savez(path, prior=prior[0].numpy(), audio_features=features[0].numpy(), valid=valid[0].numpy(),
             times=np.arange(prior.shape[1], dtype=np.float64) / 25, channel_mask=support)
    payload, report = inference.infer(checkpoint, path)
    assert payload['valid'][4] and not payload['upper_valid'][4]
    assert report['upper_valid_frames'] == int(valid.sum()) - 1
    np.testing.assert_array_equal(payload['channel_mask'], support)
    for value in payload['motions']:
        np.testing.assert_array_equal(value[4].view(np.int32), prior[0, 4].numpy().view(np.int32))
    output = tmp_path / 'masked.npz'
    np.savez(output, **payload)
    inspect_input(output, 25)


@pytest.mark.parametrize('change', ['feature_stats', 'protocol_hash', 'carrier_conflict', 'wrong_window'])
def test_checkpoint_rejects_changed_training_contract(tmp_path, change):
    path, source, checkpoint, *_ = fixture(tmp_path)
    if change == 'feature_stats':
        checkpoint['stats']['feature_std'] = checkpoint['stats']['feature_std'] * 2
    elif change == 'protocol_hash':
        checkpoint['protocol_sha256'] = 'invalid'
    elif change == 'carrier_conflict':
        checkpoint['protocol']['carrier_window'] = 3
        checkpoint['protocol_sha256'] = runner.canonical_hash(checkpoint['protocol'])
    else:
        checkpoint['protocol']['window_frames'] = 13
        checkpoint['protocol_sha256'] = runner.canonical_hash(checkpoint['protocol'])
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        inference.infer(path, source)


@pytest.mark.parametrize('change', ['clock', 'channel_order', 'width', 'nonfinite', 'prior_dtype'])
def test_invalid_deployment_inputs_are_rejected(tmp_path, change):
    checkpoint, path, _, prior, features, valid, _ = fixture(tmp_path)
    values = {'prior': prior[0].numpy(), 'audio_features': features[0].numpy(), 'valid': valid[0].numpy(),
              'times': np.arange(len(valid[0]), dtype=np.float64) / 25,
              'channels': np.asarray(inference.ARKIT_NAMES)}
    if change == 'clock':
        values['times'][4] += .01
    elif change == 'channel_order':
        values['channels'] = values['channels'][::-1]
    elif change == 'width':
        values['audio_features'] = values['audio_features'][:, :2]
    elif change == 'prior_dtype':
        values['prior'] = values['prior'].astype(np.float64)
    else:
        values['audio_features'][2, 0] = np.nan
    np.savez(path, **values)
    with pytest.raises(ValueError):
        inference.infer(checkpoint, path)
