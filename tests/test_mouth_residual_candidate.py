import argparse
import copy
import json

import numpy as np
import pytest
import torch

from scripts import train_mouth_residual_candidate as candidate
from scripts import probe_motion_condition_predictability as protocol_base


def clip(clip_id='sample', sentence='sentence', frames=8, offset=0.):
    rng = np.random.default_rng(3)
    baseline = torch.tensor(rng.uniform(.1, .4, (frames, 52)), dtype=torch.float32)
    target = baseline.clone()
    target[:, candidate.MOUTH] += .08
    mask = torch.ones(frames, 52, dtype=torch.bool)
    mask[:, 51] = False
    return {'clip_id': clip_id, 'sentence': sentence, 'split': 'train',
            'speaker': 'speaker', 'emotion': 'happy',
            'features': torch.tensor(rng.normal(size=(frames, 1540)), dtype=torch.float32) + offset,
            'context': torch.arange(5, dtype=torch.float32) + offset,
            'baseline52': baseline, 'target52': target, 'channel_mask': mask,
            'valid': torch.ones(frames, dtype=torch.bool),
            'times': torch.arange(frames, dtype=torch.float64) / 25}


def test_statistics_fit_split_and_target_independent():
    clips = [clip(f'c{i}', f's{i}', offset=i) for i in range(8)]
    poison = {'clip_id': 'outer', 'sentence': 'outer', 'split': 'holdout'}
    fit, validation = protocol_base.split_train_pool([*clips, poison])
    original = candidate.fit_input_statistics(fit)
    for row in validation:
        row['features'].fill_(1e20)
        row['context'].fill_(1e20)
    for row in fit:
        row.pop('target52')
        row.pop('channel_mask')
    changed = candidate.fit_input_statistics(fit)
    assert torch.equal(original['audio_mean'], changed['audio_mean'])
    assert set(changed['fit_ids']).isdisjoint(c['clip_id'] for c in validation)
    assert 'outer' not in changed['fit_ids']
    bad = {**fit[0], 'split': 'holdout'}
    with pytest.raises(ValueError, match='fit-pool'):
        candidate.fit_input_statistics([bad])


def test_raw_observation_mask_excludes_nan_and_missing_values():
    raw = clip()
    stats = candidate.fit_input_statistics([raw])
    raw['target52'][1, 20] = float('nan')
    raw['channel_mask'][1, 20] = False
    raw['target52'][2, 21] = 10000
    raw['channel_mask'][2, 21] = False
    row = candidate.prepare_clip(raw, stats, supervised=True)
    packed = candidate.pack_runs([row], 'audio', 'cpu')
    loss = candidate.reconstruction_loss(torch.zeros_like(packed[1]), packed[1], packed[4], [row])
    assert loss.item() == pytest.approx(.08 ** 2, abs=1e-8)
    assert not row['observed'][1, 20 - 14]
    raw['channel_mask'][1, 20] = True
    with pytest.raises(ValueError, match='Nonfinite observed'):
        candidate.prepare_clip(raw, stats, supervised=True)
    raw.pop('channel_mask')
    # Inference does not read raw masks or target values.
    candidate.prepare_clip(raw, stats, supervised=False)
    with pytest.raises(KeyError):
        candidate.prepare_clip(raw, stats, supervised=True)


def test_loss_weights_clips_equally_and_uses_observed_values():
    a, b = clip('a', frames=4), clip('b', frames=12)
    a['target52'][:, candidate.MOUTH] = a['baseline52'][:, candidate.MOUTH] + .1
    b['target52'][:, candidate.MOUTH] = b['baseline52'][:, candidate.MOUTH] + .2
    stats = candidate.fit_input_statistics([a, b])
    rows = [candidate.prepare_clip(c, stats, supervised=True) for c in (a, b)]
    packed = candidate.pack_runs(rows, 'audio', 'cpu')
    actual = candidate.reconstruction_loss(torch.zeros_like(packed[1]), packed[1], packed[4], rows)
    assert actual.item() == pytest.approx((.1 ** 2 + .2 ** 2) / 2, abs=1e-7)


def test_native_runs_and_velocity_do_not_bridge_time_or_valid_gaps():
    raw = clip(frames=8)
    raw['valid'][2] = False
    raw['times'][5:] += 1
    assert candidate.native_runs(raw['valid'], raw['times']) == [(0, 2), (3, 5), (5, 8)]
    raw['target52'].zero_()
    raw['baseline52'].zero_()
    pred = raw['baseline52'].clone()
    pred[5:, candidate.LIPS] = 10
    result = candidate.mouth_diagnostics(pred, raw)
    assert result['lip23_velocity_mae_per_second'] == 0
    # Missing observations also cannot become velocity bridges.
    raw['channel_mask'][4:6] = False
    assert candidate.mouth_diagnostics(pred, raw)['lip23_velocity_mae_per_second'] == 0


def test_zero_init_bounded_output_preserves_25_nonmouth_and_invalid_frames():
    raw = clip()
    raw['valid'][3] = False
    stats = candidate.fit_input_statistics([raw])
    raw['baseline52'][3, 0] = float('nan')
    row = candidate.prepare_clip(raw, stats, supervised=False)
    model = candidate.MouthResidualTCN(5, width=8, max_delta=.15)
    pred = candidate.predict_clip(model, row, 'audio', 'cpu')
    assert torch.equal(pred.view(torch.int32), raw['baseline52'].view(torch.int32))
    with torch.no_grad():
        model.output.bias.fill_(2.)
    pred = candidate.predict_clip(model, row, 'audio', 'cpu')
    assert torch.equal(pred[:, candidate.NONMOUTH].view(torch.int32), raw['baseline52'][:, candidate.NONMOUTH].view(torch.int32))
    assert torch.equal(pred[3].view(torch.int32), raw['baseline52'][3].view(torch.int32))
    delta = pred[raw['valid']][:, candidate.MOUTH] - raw['baseline52'][raw['valid']][:, candidate.MOUTH]
    assert bool((delta > 0).all())
    assert float(delta.abs().max()) <= .1500001
    assert len(candidate.NONMOUTH) == 25


def test_batch_padding_and_unrelated_run_cannot_influence_prediction():
    torch.manual_seed(2)
    a, b = clip('a', frames=7), clip('b', frames=18)
    stats = candidate.fit_input_statistics([a, b])
    prepared = [candidate.prepare_clip(c, stats, supervised=False) for c in (a, b)]
    model = candidate.MouthResidualTCN(5, width=8)
    nn = torch.nn
    nn.init.normal_(model.output.weight, std=.1)
    short = candidate.pack_runs(prepared[:1], 'audio', 'cpu')
    batch = candidate.pack_runs(prepared, 'audio', 'cpu')
    with torch.no_grad():
        first = model(*short[:4])[0]
        together = model(*batch[:4])[0, :7]
    assert torch.allclose(first, together, atol=2e-7)


def test_static_control_changes_only_audio_and_ignores_target_mask():
    raw = clip()
    raw['valid'][3] = False
    stats = candidate.fit_input_statistics([raw])
    prepared = candidate.prepare_clip(raw, stats, supervised=False)
    audio = candidate.pack_runs([prepared], 'audio', 'cpu')
    static = candidate.pack_runs([prepared], 'matched_static', 'cpu')
    assert audio[4] == static[4]
    assert torch.equal(audio[1], static[1])
    assert torch.equal(audio[2], static[2])
    for j, (_, left, right) in enumerate(static[4]):
        n = right - left
        assert torch.allclose(static[0][j, :n], audio[0][j, :n].mean(0).expand(n, -1))


def test_cli_smoke_atomic_report_and_no_outer_tensors(tmp_path):
    torch.set_num_threads(1)
    clips = [clip(f'c{i}', f's{i}') for i in range(7)]
    clips.append({'clip_id': 'outer', 'sentence': 'outer', 'split': 'holdout'})
    path = tmp_path / 'dataset.pt'
    torch.save({'schema': protocol_base.DATASET_SCHEMA, 'clips': clips}, path)
    fit, valid = protocol_base.split_train_pool(clips)
    ref = {'schema': 'bounded_audio_experiment_v1', 'dataset_sha256': protocol_base.sha(path),
           'inner_train_ids': [c['clip_id'] for c in fit],
           'inner_validation_ids': [c['clip_id'] for c in valid],
           'validation_sentences': sorted({c['sentence'] for c in valid})}
    ref_path = tmp_path / 'reference.json'
    ref_path.write_text(json.dumps(ref))
    args = argparse.Namespace(dataset=path, reference_protocol=ref_path, output=tmp_path / 'run',
                              steps=2, batch_size=2, width=8, max_delta=.15,
                              learning_rate=.001, seeds=[42], log_every=1,
                              save_every=1, device='cpu')
    result = candidate.run(args)
    assert set(result['seeds']['42']) == {'audio', 'matched_static', 'audio_reverse'}
    status = json.loads((args.output / 'status.json').read_text())
    assert status['state'] == 'completed'
    assert status['default_replaced'] is False
    assert status['lip_sync_certified'] is False
    report = json.loads((args.output / 'seed_42/audio/validation/arkit_benchmark.json').read_text())
    assert report['summary']['arkit_lbe']['status'] == 'computed'
    assert report['summary']['av_offset']['status'] == 'pending'
    assert len(report['per_clip']) == 5
    assert not list(args.output.rglob('*.tmp'))
    with pytest.raises(FileExistsError):
        candidate.run(args)
