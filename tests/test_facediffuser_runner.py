import copy
import json
import sys

import pytest
import torch
from torch import nn

from scripts import train_facediffuser_arkit as runner
from scripts.compact_native_curves import expand_curves
from kinetalk_b0.models.facediffuser_arkit import FaceDiffuserARKit


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FrozenSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('tag', torch.tensor(1.))

    def encode_motion(self, motion, valid):
        logits = torch.zeros(len(motion), 8)
        logits[:, 0] = 1
        return {'emotion_logits': logits}

    def generate(self, *args, **kwargs):
        raise AssertionError('FaceDiffuser must never use the KineTalk generator')


def context():
    torch.manual_seed(7)

    def split(name, clips):
        frames = 6
        valid = torch.ones(clips, frames, dtype=torch.bool)
        valid[0, 0] = False
        valid[-1, -1] = False
        channel = torch.ones(clips, 52, dtype=torch.bool)
        channel[:, 51] = False
        target = torch.rand(clips, frames, 52)
        target = torch.where(valid[..., None] & channel[:, None], target, 0.)
        return {'valid': valid, 'channel_mask': channel, 'motion': target,
                'content': torch.randn(clips, frames, 7), 'audio_features': torch.randn(clips, frames, 11),
                'global_code': torch.randn(clips, 64),
                'identity_code': torch.randn(clips, 128), 'anchors': torch.zeros(clips, 52),
                'intensity_value': torch.ones(clips, 1), 'speaker_id': torch.zeros(clips, dtype=torch.long),
                'emotion_id': torch.arange(clips) % 2, 'speaker': ['a'] * clips,
                'sentence_id': [f's{i}' for i in range(clips)],
                'clip_id': [f'{name}{i}' for i in range(clips)],
                'b0': torch.full_like(target, 99.), 'times': (torch.arange(frames).double() / 25)[None].expand(clips, -1)}

    data = {'splits': {'train': split('train', 4), 'validation': split('val', 2)},
            'target_scales': torch.ones(52), 'feature_stats': {'mean': torch.full((11,), .3), 'std': torch.full((11,), 1.4)},
            'provenance': {'manifest_sha256': 'fake_manifest', 'index_sha256': 'fake_index', 'test_loaded': False}}
    identities = {0: {'code': torch.zeros(1, 128), 'baseline': torch.zeros(1, 52)}}
    return data, FrozenSource(), FrozenSource(), identities, torch.ones(4)


def small_model(total_input_dim):
    return FaceDiffuserARKit(content_dim=total_input_dim, latent_dim=8, gru_hidden=8, num_layers=2,
                             diffusion_steps=1000, dropout=.3, clip_condition_dim=245)


def prepare(monkeypatch, tmp_path):
    ctx = context()
    monkeypatch.setattr(runner, 'load_context', lambda *args: copy.deepcopy(ctx))
    monkeypatch.setattr(runner, 'make_model', small_model)
    prepared = tmp_path / 'data'
    prepared.mkdir()
    (prepared / 'manifest.json').write_text(json.dumps({'roles': {'val': {'query': [
        {'clip_id': cid, 'frames': 6} for cid in ctx[0]['splits']['validation']['clip_id']]}}}))
    source = tmp_path / 'source.pt'
    source.write_bytes(b'fixture source hash')
    return prepared, source


def test_formal_defaults_match_official_main_and_declared_conditions():
    model = runner.make_model(2308)
    assert model.latent_dim == 512 and model.gru_hidden == 512 and model.num_layers == 2
    assert model.diffusion_steps == 1000 and model.dropout == .3 and model.content_dim == 2308
    assert model.clip_condition_dim == 245
    args = runner.parser().parse_args(['--data', 'x', '--source', 'y', '--output', 'z'])
    assert (args.epochs, args.batch_size, args.seed) == (100, 16, 47)


def test_clip_condition_excludes_query_target_and_freezes_encoders():
    q = context()[0]['splits']['train']
    for name in runner.CONDITION_LAYOUT:
        q[name].requires_grad_()
    original = runner.clip_condition(q)
    q['motion'].fill_(float('nan'))
    q['emotion_id'].fill_(7)
    q['b0'].fill_(-1e30)
    torch.testing.assert_close(runner.clip_condition(q), original, rtol=0, atol=0)
    assert not original.requires_grad
    assert original.shape == (4, 245)


def test_actual_optimizer_changes_random_decoder_with_observed_target_only():
    data = context()[0]
    q = data['splits']['train']
    model = small_model(18)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.final_layer.weight.detach().clone()
    loss, norm = runner.training_step(model, optimizer, q, torch.Generator().manual_seed(4), data['feature_stats'])
    assert loss > 0 and norm > 0
    assert not torch.equal(before, model.final_layer.weight)
    assert all(value.grad is None for value in q.values() if torch.is_tensor(value))


def test_cpu_smoke_and_resume_cover_full_1000_step_sampling_and_no_mouth_bypass(monkeypatch, tmp_path):
    prepared, source = prepare(monkeypatch, tmp_path)
    output = tmp_path / 'run'
    argv = ['train_facediffuser_arkit.py', '--data', str(prepared), '--source', str(source),
            '--output', str(output), '--device', 'cpu', '--batch-size', '2', '--smoke']
    monkeypatch.setattr(sys, 'argv', argv)
    real_evaluate = runner.evaluate

    def interrupt_after_training(*args, **kwargs):
        raise RuntimeError('synthetic interruption before sampling')

    monkeypatch.setattr(runner, 'evaluate', interrupt_after_training)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        runner.main()
    last = torch.load(output / 'last.pt', weights_only=False)
    weights = {key: value.clone() for key, value in last['model'].items()}
    assert last['epoch'] == 1 and last['rng']['training_generator'].device.type == 'cpu'
    monkeypatch.setattr(runner, 'evaluate', real_evaluate)
    monkeypatch.setattr(sys, 'argv', argv + ['--resume'])
    runner.main()
    protocol = json.loads((output / 'protocol.json').read_text())
    report = json.loads((output / 'evaluation.json').read_text())
    status = json.loads((output / 'status.json').read_text())
    assert protocol['train_clips'] == 4 and protocol['validation_clips'] == 2
    assert protocol['diffusion']['steps'] == 1000
    assert protocol['audio_input_layout']['total'] == 18
    assert protocol['optimizer']['class'] == 'Adam' and protocol['optimizer']['lr'] == .0001
    assert not status['test_loaded'] and status['status'] == 'complete'
    assert report['noise_seeds'] == [42, 123, 2026] and len(report['sampling_batch_seconds']) == 3
    assert not report['formal_comparison_ready']
    final = torch.load(output / 'final.pt', weights_only=False)
    assert runner.state_hash(final['feature_stats']) == protocol['feature_stats_sha256']
    for key, value in weights.items():
        torch.testing.assert_close(final['model'][key], value, rtol=0, atol=0)
    curves = expand_curves(torch.load(output / 'native_curves.pt', weights_only=False))
    for seed in (42, 123, 2026):
        value = curves['predictions'][f'{seed}/full']
        assert value[..., 51].count_nonzero() == 0
        assert not torch.equal(value[..., 14:41], curves['b0'][..., 14:41])
    arkit = json.loads((output / 'arkit_full.json').read_text())
    assert arkit['smoke'] and 'smoke' in arkit['scope']
    # Protocol edits fail before any resumed updates or new sampling.
    protocol['seed'] += 1
    (output / 'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='protocol mismatch'):
        runner.main()


def test_formal_run_cannot_silently_use_partial_dataset(monkeypatch, tmp_path):
    prepared, source = prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, 'argv', ['runner', '--data', str(prepared), '--source', str(source),
                                    '--output', str(tmp_path / 'formal'), '--device', 'cpu'])
    with pytest.raises(ValueError, match='4098 TRAIN and 446'):
        runner.main()


def test_all_audio_condition_uses_train_stats_and_ignores_targets_and_masked_nan():
    data = context()[0]
    batch = data['splits']['train']
    valid = batch['valid']
    batch['content'][~valid] = float('nan')
    batch['audio_features'][~valid] = float('nan')
    batch['content'].requires_grad_()
    batch['audio_features'].requires_grad_()
    result = runner.audio_input(batch, data['feature_stats'])
    assert result.shape == (4, 6, 18) and not result.requires_grad
    torch.testing.assert_close(result[..., :7][valid], batch['content'][valid], rtol=0, atol=0)
    torch.testing.assert_close(result[..., 7:][valid], ((batch['audio_features'] - .3) / 1.4)[valid], rtol=0, atol=0)
    assert result[~valid].count_nonzero() == 0 and torch.isfinite(result).all()
    batch['motion'].fill_(float('nan'))
    batch['emotion_id'].fill_(7)
    batch['b0'].fill_(1e30)
    torch.testing.assert_close(runner.audio_input(batch, data['feature_stats']), result, rtol=0, atol=0)
    changed = {**batch, 'audio_features': batch['audio_features'].detach().clone()}
    changed['audio_features'][0, 1, 10] += 1
    assert runner.audio_input(changed, data['feature_stats'])[0, 1, 17] != result[0, 1, 17]
