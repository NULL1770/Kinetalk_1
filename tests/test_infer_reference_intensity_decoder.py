import json

import numpy as np
import pytest
import torch

from kinetalk_b0.models.reference_intensity_decoder import ReferenceIntensityDecoder, ReferenceIntensityStudent
from kinetalk_b0.models.slow_state_affect import compose_upper_face
from kinetalk_b0.models.temporal_motion_carrier import smooth_motion_carrier
from scripts import infer_reference_intensity_decoder as inference
from scripts import train_reference_intensity_decoder as runner
from scripts.render_dynamic_rig_comparison import inspect_input


def fixture(tmp_path):
    torch.manual_seed(74)
    frames, rank = 21, 4
    valid = torch.ones(1, frames, dtype=torch.bool)
    valid[:, [0, 7, 8, 20]] = False
    prior = .55 + .12 * torch.sin(torch.arange(frames)[None, :, None] * .42
                                + torch.arange(52)[None, None] * .24)
    prior[~valid] = float('nan')
    features = torch.randn(1, frames, 1540)
    features[~valid] = float('nan')
    global_code, identity_code = torch.randn(1, 64), torch.randn(1, 128)
    anchor = torch.full((1, 52), .2)
    decoder = ReferenceIntensityDecoder(hidden=12).eval()
    student = ReferenceIntensityStudent(input_dim=rank+4, hidden=12).eval()
    stats = {'feature_mean': torch.linspace(-.2, .2, 1540), 'feature_std': torch.linspace(.7, 1.3, 1540),
             'projection': torch.randn(1536, rank) / 40,
             'reduced_mean': torch.linspace(-.1, .1, rank+4), 'reduced_std': torch.linspace(.8, 1.2, rank+4),
             'global_code_mean': torch.linspace(-.2, .2, 64), 'global_code_std': torch.linspace(.8, 1.2, 64),
             'identity_code_mean': torch.linspace(-.2, .2, 128), 'identity_code_std': torch.linspace(.8, 1.2, 128),
             'scales9': torch.linspace(.04, .12, 9)}
    protocol = {'schema': inference.SCHEMA, 'window': 5, 'pca_rank': rank, 'upper_mean_protected': False}
    checkpoint = {'schema': inference.SCHEMA, 'decoder': decoder.state_dict(), 'student': student.state_dict(),
                  'decoder_config': decoder.export_config(), 'student_config': student.export_config(),
                  'stats': stats, 'protocol': protocol, 'protocol_sha256': runner.canonical_hash(protocol)}
    checkpoint_path = tmp_path / 'final.pt'
    torch.save(checkpoint, checkpoint_path)
    values = {'prior': prior[0].numpy(), 'audio_features': features[0].numpy(),
              'global_code': global_code[0].numpy(), 'identity_code': identity_code[0].numpy(),
              'anchor': anchor[0].numpy(), 'valid': valid[0].numpy(), 'times': np.arange(frames, dtype=np.float64)/25,
              'channels': np.asarray(inference.ARKIT_NAMES), 'clip_id': np.asarray('synthetic'), 'noise_seed': np.asarray(48),
              'motion': np.array([{'forbidden': True}], dtype=object),
              'emotion': np.array([{'forbidden': True}], dtype=object),
              'target': np.array([{'forbidden': True}], dtype=object)}
    input_path = tmp_path / 'input.npz'
    np.savez(input_path, **values)
    q = {'audio_features': features, 'global_code': global_code, 'identity_code': identity_code,
         'anchors': anchor, 'valid': valid, 'motion': torch.zeros_like(prior),
         'channel_mask': torch.ones(1, 52, dtype=torch.bool), 'anchor_valid': torch.ones(1, 52, dtype=torch.bool),
         'clip_id': ['synthetic'], 'speaker': ['synthetic'], 'sentence_id': ['synthetic']}
    return checkpoint_path, input_path, checkpoint, values, q, decoder, student


def test_inference_matches_runner_preprocessing_and_outputs(tmp_path):
    checkpoint_path, input_path, checkpoint, values, q, decoder, student = fixture(tmp_path)
    inputs = inference.load_input(input_path)
    prepared = inference.prepare_inputs(inputs, checkpoint['stats'])
    cache = runner.prepare_cache(q, checkpoint['stats'])
    for key in ('features', 'valid', 'anchor', 'global_code', 'identity_code'):
        torch.testing.assert_close(prepared[key], cache[key], atol=0, rtol=0)
    payload, report = inference.infer(checkpoint_path, input_path)
    base = torch.from_numpy(values['prior'])[None]
    expected = {'original_prior': base, 'smooth_prior': smooth_motion_carrier(base, q['valid'], window=5)['smoothed']}
    for mode in ('full', 'static'):
        upper, intensity = runner.predict(decoder, student, cache, torch.tensor([0]), 'cpu', mode=mode)
        expected[mode] = compose_upper_face(base, upper, q['valid'])
        key = 'predicted_intensity' if mode == 'full' else 'static_intensity'
        np.testing.assert_array_equal(payload[key], intensity[0].numpy())
    for mode, value in expected.items():
        observed = torch.from_numpy(payload['motions'][inference.MODE_NAMES.index(mode)])[None]
        assert torch.equal(observed.contiguous().view(torch.int32), value.contiguous().view(torch.int32))
        assert report['protection'][mode]['passed']
    assert report['upper_mean_protected'] is False
    assert report['upper_mean_change']['full']['max_abs_change_from_original_prior'] > .1
    assert report['raw_upper_range']['full']['outside_unit_interval_fraction'] == 0
    assert report['query_motion_read'] is False and report['query_emotion_read'] is False and report['query_target_read'] is False
    static_upper = payload['motions'][3, values['valid']][:, list(inference.UPPER_INDICES)]
    np.testing.assert_allclose(static_upper, np.broadcast_to(static_upper[:1], static_upper.shape), atol=1e-7, rtol=1e-6)


def test_cli_provenance_renderer_output_and_no_overwrite(tmp_path, monkeypatch):
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
    assert report['checkpoint_sha256'] == inference._sha(checkpoint)
    assert report['output_sha256'] == inference._sha(output)
    assert report['protocol_sha256'] == inference._canonical_hash(report['protocol'])
    assert report['range_clipping_applied'] is False
    assert report['post_composition_smoothing'] is False
    with pytest.raises(FileExistsError):
        inference.main()


def test_partial_upper_mask_copies_entire_inactive_frame(tmp_path):
    checkpoint, source, _, values, *_ = fixture(tmp_path)
    support = np.ones((len(values['valid']), 52), dtype=bool)
    support[4, inference.UPPER_INDICES[0]] = False
    values['channel_mask'] = support
    np.savez(source, **values)
    payload, report = inference.infer(checkpoint, source)
    assert payload['valid'][4] and not payload['upper_valid'][4]
    assert report['upper_valid_frames'] == int(values['valid'].sum())-1
    for motion in payload['motions']:
        np.testing.assert_array_equal(motion[4].view(np.int32), values['prior'][4].view(np.int32))
    output = tmp_path / 'partial.npz'
    np.savez(output, **payload)
    inspect_input(output, 25)


@pytest.mark.parametrize('change', ['schema', 'protocol_hash', 'window', 'mean_contract', 'pca_rank',
                                   'stats_width', 'negative_std', 'nonfinite_projection', 'state_nonfinite', 'config'])
def test_bad_checkpoint_contract_is_rejected(tmp_path, change):
    path, source, checkpoint, *_ = fixture(tmp_path)
    if change == 'schema':
        checkpoint['schema'] = 'unsupported'
    elif change == 'protocol_hash':
        checkpoint['protocol_sha256'] = 'invalid'
    elif change == 'window':
        checkpoint['protocol']['window'] = 3
    elif change == 'mean_contract':
        checkpoint['protocol']['upper_mean_protected'] = True
    elif change == 'pca_rank':
        checkpoint['protocol']['pca_rank'] = 5
    elif change == 'stats_width':
        checkpoint['stats']['reduced_mean'] = torch.zeros(12)
    elif change == 'negative_std':
        checkpoint['stats']['global_code_std'][0] = -1
    elif change == 'nonfinite_projection':
        checkpoint['stats']['projection'][0, 0] = float('nan')
    elif change == 'state_nonfinite':
        checkpoint['student']['head.weight'][0, 0] = float('inf')
    elif change == 'config':
        checkpoint['decoder_config']['global_dim'] = 128
    if change != 'protocol_hash':
        checkpoint['protocol_sha256'] = inference._canonical_hash(checkpoint['protocol'])
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        inference.infer(path, source)


@pytest.mark.parametrize('change', ['missing_condition', 'global_shape', 'anchor_nonfinite', 'nonfinite_audio',
                                   'clock', 'wrong_fps', 'channel_order', 'prior_dtype', 'audio_width', 'invalid_mask'])
def test_bad_deployment_input_is_rejected(tmp_path, change):
    checkpoint, path, _, values, *_ = fixture(tmp_path)
    if change == 'missing_condition':
        del values['identity_code']
    elif change == 'global_shape':
        values['global_code'] = values['global_code'][None]
    elif change == 'anchor_nonfinite':
        values['anchor'][0] = np.nan
    elif change == 'nonfinite_audio':
        values['audio_features'][2, 0] = np.nan
    elif change == 'clock':
        values['times'][2] += .001
    elif change == 'wrong_fps':
        values['times'] = np.arange(len(values['valid'])) / 30
    elif change == 'channel_order':
        values['channels'] = values['channels'][::-1]
    elif change == 'prior_dtype':
        values['prior'] = values['prior'].astype(np.float64)
    elif change == 'audio_width':
        values['audio_features'] = values['audio_features'][:, :1536]
    elif change == 'invalid_mask':
        values['valid'] = values['valid'].astype(np.int32)
    np.savez(path, **values)
    with pytest.raises(ValueError):
        inference.infer(checkpoint, path)


def test_weights_only_loading_is_explicit(tmp_path, monkeypatch):
    checkpoint, source, *_ = fixture(tmp_path)
    original_load = torch.load
    called = []
    def tracked_load(*args, **kwargs):
        called.append(kwargs.copy())
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', tracked_load)
    inference.infer(checkpoint, source)
    assert len(called) == 1 and called[0]['weights_only'] is True and called[0]['map_location'] == 'cpu'
