"""Integration checks for isolated mean/state training and its held-out controls."""
import copy
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn

from scripts import train_isolated_audio_state as runner
from scripts.compact_native_curves import expand_curves
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, lift_slow_state


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FakeAudio(nn.Module):
    def __init__(self, *unused):
        super().__init__()
        self.register_buffer('tag', torch.tensor(3.))

    def forward(self, features, valid):
        clean = torch.where(valid[..., None], features, 0.)
        local = clean[..., :1].expand(-1, -1, 64) * .02
        pooled = clean.sum((1, 2)) / (valid.sum(1) * features.shape[-1])
        return {'local': local, 'global': pooled[:, None].expand(-1, 64),
                'emotion_logits': pooled[:, None].expand(-1, 8),
                'intensity_value': pooled[:, None] * 0 + .5}


class FakeSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('tag', torch.tensor(7.))
        self.last_affect = None

    def set_motion_support(self, support):
        pass

    def set_residual_support(self, support):
        pass

    def set_mouth_reference_calibration(self, calibration):
        pass

    def generate(self, content, valid, identity, affect, *, initial_noise, steps, base):
        assert steps == 12
        self.last_affect = {k: v.clone() for k, v in affect.items()}
        correction = affect['local'][..., :1] * .1 + initial_noise * .002
        motion = base['b0'] + identity['baseline'][:, None] + correction
        return {'motion': torch.where(valid[..., None], motion, 0.)}

    def encode_motion(self, residual, valid):
        value = torch.where(valid[..., None], residual, 0.).mean((1, 2))
        return {'emotion_logits': value[:, None].expand(-1, 8)}


def synthetic_data():
    torch.manual_seed(101)
    scales = torch.linspace(.1, .3, 52)
    frames, clips = 40, 4
    t = torch.arange(frames, dtype=torch.float32)
    identities = {sid: {'code': torch.randn(1, 128), 'baseline': torch.full((1, 52), .01 * sid)}
                  for sid in (0, 1)}

    def split(prefix):
        valid = torch.ones(clips, frames, dtype=torch.bool)
        valid[0, [0, 17]] = False
        valid[1, 35:] = False
        valid[2, 37:] = False
        anchor = torch.full((clips, 52), .3)
        slope = (t / 40 - .4)[None, :, None].expand(clips, -1, 4) * torch.tensor([.3, -.2, .1, .5])
        motion = anchor[:, None] + lift_slow_state(slope, scales)
        motion = motion + .025 * torch.sin(t / 3)[None, :, None]
        motion = torch.where(valid[..., None], motion, 0.)
        b0 = torch.where(valid[..., None], motion * .9 + .01, 0.)
        features = torch.randn(clips, frames, 12)
        features[~valid] = float('nan')
        return {'valid': valid, 'motion': motion, 'b0': b0, 'h0': torch.randn(clips, frames, 128),
                'content': torch.randn(clips, frames, 128), 'audio_features': features,
                'anchors': anchor, 'anchor_valid': torch.ones(clips, 52, dtype=torch.bool),
                'channel_mask': torch.ones(clips, 52, dtype=torch.bool),
                'speaker_id': torch.tensor([0, 1, 0, 1]), 'emotion_id': torch.tensor([0, 1, 0, 1]),
                'sentence_id': ['s0', 's1', 's2', 's3'], 'speaker': ['a', 'b', 'a', 'b'],
                'clip_id': [f'{prefix}_{i}' for i in range(clips)],
                'times': (t.double() / 25)[None].expand(clips, -1).clone()}

    data = {'system': FakeSystem(), 'splits': {'train': split('train'), 'validation': split('val')},
            'feature_stats': {'mean': torch.zeros(12), 'std': torch.ones(12)},
            'target_scales': scales, 'provenance': {'manifest_sha256': 'fixture_manifest', 'index_sha256': 'fixture_index'}}
    return data, identities


def install_context(monkeypatch, tmp_path):
    data, identities = synthetic_data()
    source = tmp_path / 'source.pt'
    torch.save({'data_manifest_sha256': data['provenance']['manifest_sha256'],
                'system': data['system'].state_dict(), 'audio': FakeAudio().state_dict(),
                'config': {'model': {'motion_support': [True] * 52, 'residual_support': [True] * 52,
                                     'mouth_reference_calibration': {'fixture': True}}}}, source)
    monkeypatch.setattr(runner, 'load_paper_data', lambda path, seed: copy.deepcopy(data))
    monkeypatch.setattr(runner, 'SlowStateAffect', FakeAudio)
    monkeypatch.setattr(runner, 'cache_current_base', lambda *args: None)
    monkeypatch.setattr(runner, 'identity_cache', lambda *args: copy.deepcopy(identities))
    monkeypatch.setattr(runner, 'identity_report', lambda *args: {'fixture': True})
    return data, source


def test_targets_are_native_mask_centered_and_dynamic_scales_use_train_only(monkeypatch, tmp_path):
    raw, source = install_context(monkeypatch, tmp_path)
    context = runner.load_context(tmp_path, source, 'cpu')
    data, _, _, _, dynamic_scales = context
    train = data['splits']['train']
    expected = (train['target_centered_state'].square().sum((0, 1)) / train['valid'].sum()).sqrt().clamp_min(.02)
    torch.testing.assert_close(dynamic_scales, expected, rtol=0, atol=0)
    for q in data['splits'].values():
        torch.testing.assert_close(q['target_centered_state'].sum(1), torch.zeros(len(q['valid']), 4), atol=1e-6, rtol=0)
        assert q['target_centered_state'][~q['valid']].count_nonzero() == 0
        assert not q['global_code'].requires_grad and not q['target_centered_state'].requires_grad
        expected_mean = (q['motion'][..., list(UPPER_INDICES)] * q['valid'][..., None]).sum(1) / q['valid'].sum(1, keepdim=True)
        torch.testing.assert_close(q['target_mean'], expected_mean, rtol=0, atol=0)
    raw['splits']['validation']['motion'] += 50 * torch.arange(40)[None, :, None]
    changed = runner.load_context(tmp_path, source, 'cpu')
    torch.testing.assert_close(changed[-1], dynamic_scales, rtol=0, atol=0)
    assert not torch.equal(changed[0]['splits']['validation']['target_mean'], data['splits']['validation']['target_mean'])


def test_center_and_objective_ignore_invalid_payload_and_remove_only_dc():
    valid = torch.tensor([[True, False, True, True], [False, True, True, False]])
    value = torch.tensor([[[1.], [float('nan')], [4.], [7.]], [[float('nan')], [2.], [6.], [float('nan')]]])
    centered = runner.center(value, valid)
    expected = torch.tensor([[[-3.], [0.], [0.], [3.]], [[0.], [-2.], [2.], [0.]]])
    torch.testing.assert_close(centered, expected, rtol=0, atol=0)
    torch.testing.assert_close(runner.center(value + 100, valid), expected, rtol=0, atol=0)
    loss = runner.masked_mse(centered, torch.zeros_like(centered), valid)
    torch.testing.assert_close(loss, torch.tensor(26 / 5))


@pytest.mark.parametrize('storage_width', [64, 128])
def test_trim_preserves_non_temporal_conditions_when_width_matches_time(storage_width):
    valid = torch.zeros(2, storage_width, dtype=torch.bool)
    valid[:, :7] = True
    q = {'valid': valid, 'audio_features': torch.randn(2, storage_width, 12),
         'target_centered_state': torch.randn(2, storage_width, 4),
         'global_code': torch.randn(2, 64), 'identity_code': torch.randn(2, 128),
         'anchors': torch.randn(2, 52), 'channel_mask': torch.ones(2, 52, dtype=torch.bool)}
    out = runner.trim(q, torch.tensor([1]), 'cpu')
    assert out['valid'].shape == (1, 7)
    assert out['target_centered_state'].shape == (1, 7, 4)
    assert out['global_code'].shape == (1, 64)
    assert out['identity_code'].shape == (1, 128)
    torch.testing.assert_close(out['identity_code'], q['identity_code'][[1]], rtol=0, atol=0)


def test_paired_statistics_do_not_award_ties_or_single_sentence_clusters():
    assert runner.paired_ci([-1., -2., -1.], ['a', 'b', 'c'])['passed']
    assert not runner.paired_ci([0., 0., 0.], ['a', 'b', 'c'])['passed']
    assert not runner.paired_ci([-1., -1.], ['a', 'a'])['passed']
    with pytest.raises(ValueError, match='Nonfinite'):
        runner.paired_ci([np.nan], ['a'])
    with pytest.raises(ValueError, match='Empty'):
        runner.paired_ci([], [])


def test_protected_baseline_uses_original_audio_local_and_caller_noise(monkeypatch, tmp_path):
    _, source = install_context(monkeypatch, tmp_path)
    data, system, audio, identities, _ = runner.load_context(tmp_path, source, 'cpu')
    b = runner.subset(data['splits']['validation'], torch.arange(4), 'cpu')
    b['frozen_local'] = audio(b['audio_features'], b['valid'])['local']
    noise = torch.randn_like(b['motion'])
    actual = runner.old_prediction(system, b, identities, noise)
    torch.testing.assert_close(system.last_affect['local'], b['frozen_local'], rtol=0, atol=0)
    ident = runner.batch_identity(identities, b)
    expected = system.generate(b['content'], b['valid'], ident,
                              {'global': b['global_code'], 'local': b['frozen_local'], 'intensity_value': b['intensity_value']},
                              initial_noise=noise, steps=12, base={'b0': b['b0'], 'h0': b['h0']})['motion']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['audio', 'static'])
def test_cpu_smoke_runs_independent_backward_and_writes_complete_reports(monkeypatch, tmp_path, mode):
    raw, source = install_context(monkeypatch, tmp_path)
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    manifest = {'roles': {'val': {'query': [{'clip_id': cid, 'frames': 40}
                                          for cid in raw['splits']['validation']['clip_id']]}}}
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    output = tmp_path / mode
    monkeypatch.setattr(sys, 'argv', ['train_isolated_audio_state.py', '--data', str(prepared),
                                    '--source', str(source), '--output', str(output), '--device', 'cpu',
                                    '--batch-size', '2', '--mode', mode, '--smoke'])
    # Exercise the actual optimizer sequence, report generation, native packing,
    # checkpointing, and all condition interventions; only frozen external context
    # is synthetic, so no private dataset or GPU is required.
    runner.main()
    status = json.loads((output / 'status.json').read_text())
    report = json.loads((output / 'evaluation.json').read_text())
    assert status['status'] == 'complete'
    assert not status['passed'] and not report['test_loaded']
    assert report['checks']['mouth_preserved']
    assert report['states']['static']['r2'] == pytest.approx(0.)
    saved = torch.load(output / 'last.pt', weights_only=False)
    assert saved['mean_optimizer']['state']
    assert bool(saved['state_optimizer']['state']) == (mode == 'audio')
    curves = expand_curves(torch.load(output / 'native_curves.pt', weights_only=False))
    for seed in (42, 123, 2026):
        baseline = curves['predictions'][f'{seed}/base']
        for intervention in ('full', 'static', 'reverse'):
            pred = curves['predictions'][f'{seed}/{intervention}']
            torch.testing.assert_close(pred[..., list(runner.NOT_UPPER)], baseline[..., list(runner.NOT_UPPER)], rtol=0, atol=0)
    if mode == 'static':
        assert not report['checks']['train_state_fit']
        assert not report['checks']['val_state_better_than_static']
