import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kinetalk_b0.models.aligned_audio_local import AlignedAudioLocal
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect, compose_upper_face
from kinetalk_b0.models.temporal_upper import TemporalUpperFlow
from scripts import train_temporal_repair as runner
from scripts.train_full_staged import NOT_UPPER, cache_current_base, identity_cache
from scripts.train_predictable_renderer import state_hash


@pytest.fixture(autouse=True)
def single_thread():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def cfg():
    return {'data': {'motion_dim': 52, 'content_dim': 8, 'audio_dim': 10,
                     'neutral_output_indices': [17, 18], 'emotion_classes': ['neutral', 'angry'],
                     'num_intensity_levels': 4},
            'model': {'content_dim': 8, 'emotion_dim': 8, 'style_dim': 8, 'hidden_dim': 8,
                      'heads': 2, 'dit_dim': 8, 'dit_depth': 1, 'dropout': 0.,
                      'residual_scale': .25, 'affect_hidden_dim': 8, 'affect_rank': 3, 'affect_stride': 4}}


def fixture():
    torch.manual_seed(19)
    system = NeutralAffectSystem(cfg()).eval().requires_grad_(False)
    audio = SlowStateAffect(torch.zeros(10), torch.ones(10), global_dim=8, hidden=8,
                            local_dim=8, num_emotions=2, stride=4).eval().requires_grad_(False)
    splits = {}
    for role, sids in (('train', [7, 19]), ('validation', [99, 99])):
        valid = torch.ones(2, 12, dtype=torch.bool); valid[0, -2:] = False
        splits[role] = {'motion': torch.rand(2, 12, 52) * .3,
                        'content': torch.randn(2, 12, 8), 'audio_features': torch.randn(2, 12, 10),
                        'valid': valid, 'channel_mask': torch.ones(2, 52, dtype=torch.bool),
                        'anchors': torch.full((2, 52), .1), 'anchor_valid': torch.ones(2, 52, dtype=torch.bool),
                        'emotion_id': torch.tensor([0, 1]), 'speaker_id': torch.tensor(sids),
                        'times': torch.arange(12)[None].expand(2, -1) / 25,
                        'clip_id': [role + '0', role + '1']}
    refs = {sid: {'motion': torch.rand(2, 12, 52) * .2, 'content': torch.randn(2, 12, 8),
                  'valid': torch.ones(2, 12, dtype=torch.bool), 'channel_mask': torch.ones(2, 52, dtype=torch.bool)}
            for sid in (7, 19, 99)}
    data = {'system': system, 'config': cfg(), 'splits': splits, 'refs': refs, 'fit_sids': [7, 19],
            'dev_sids': [99], 'target_scales': torch.full((52,), .1),
            'provenance': {'test_loaded': False, 'input_sha256': {'fixture': 'only'}}}
    cache_current_base(system, data, 'cpu', 2)
    identities = identity_cache(system, data, 'cpu')
    runner.cache_conditions(system, audio, data, identities, 'cpu', 2)
    return data, audio, identities


def test_control_scale_and_loss_use_valid_frame_weights_not_bin_counts():
    q = {'teacher_controls': torch.tensor([[[1., 2.], [-4., -8.], [100., 100.]]]),
         'control_weight': torch.tensor([[4., 1., 0.]])}
    scale = runner.control_scale(q)
    torch.testing.assert_close(scale, torch.tensor([2., 4.]), rtol=0, atol=0)
    pred = torch.zeros_like(q['teacher_controls'], requires_grad=True)
    loss = runner.control_loss(pred, q['teacher_controls'], q['control_weight'], scale)
    assert float(loss.detach()) == pytest.approx(1., abs=1e-7)
    loss.backward()
    assert not pred.grad[:, 2].any()
    assert torch.isfinite(pred.grad).all()


def test_projected_affect_and_upper_decode_ignore_query_gt_and_copy_nonupper():
    data, audio, identities = fixture(); system = data['system']; q = data['splits']['validation']
    adapter = AlignedAudioLocal(audio.feature_mean, audio.feature_std, hidden=8, rank=3, stride=4)
    adapter.initialize_from_slow(audio).eval()
    with torch.no_grad(): adapter.control_head.weight.normal_(std=.1)
    a = runner.projected_affect(system, adapter, q)
    changed = copy.deepcopy(q)
    for key in ('motion', 'teacher_controls', 'teacher_local', 'teacher_global', 'teacher_intensity'):
        changed[key].fill_(999.)
    changed['emotion_id'].fill_(123)
    b = runner.projected_affect(system, adapter, changed)
    for key in a:
        assert torch.equal(a[key], b[key])
    upper = TemporalUpperFlow(cfg(), use_state=True).eval()
    local = copy.deepcopy(audio)
    identity = runner.batch_identity(identities, q)
    noise = torch.randn(2, 12, 52)
    with torch.no_grad():
        first = runner.upper_decode(upper, local, q, identity, a, data['target_scales'], noise, 2)
        again = runner.upper_decode(upper, local, changed, identity, b, data['target_scales'], noise, 2)
        assert torch.equal(first, again)
        baseline = system.generate(q['content'], q['valid'], identity, a, initial_noise=noise,
                                   steps=2, base={k: q[k] for k in ('b0', 'h0')})['motion']
        composed = compose_upper_face(baseline, first, q['valid'])
    assert torch.equal(composed[..., list(NOT_UPPER)], baseline[..., list(NOT_UPPER)])


def test_matched_upper_initialization_and_draws_with_frozen_alignment_base():
    data, audio, identities = fixture(); system = data['system']; q = data['splits']['train']
    adapter = AlignedAudioLocal(audio.feature_mean, audio.feature_std, hidden=8, rank=3, stride=4)
    adapter.initialize_from_slow(audio).eval().requires_grad_(False)
    frozen = {name: state_hash(module.state_dict()) for name, module in [('system', system), ('audio', audio), ('adapter', adapter)]}
    init, sequences = [], []
    for arm in ('direct', 'soft'):
        torch.manual_seed(53)
        local = copy.deepcopy(audio).requires_grad_(False)
        local.input.load_state_dict(adapter.input.state_dict()); local.blocks.load_state_dict(adapter.blocks.state_dict())
        torch.nn.init.zeros_(local.local_head.weight); torch.nn.init.zeros_(local.local_head.bias)
        for module in (local.input, local.blocks, local.local_head): module.requires_grad_(True)
        upper = TemporalUpperFlow(cfg(), use_state=arm == 'soft').eval()
        init.append((state_hash(upper.state_dict()), state_hash(local.state_dict())))
        rng = torch.Generator().manual_seed(53)
        ids = torch.randperm(2, generator=rng); noise = torch.randn(2, 12, 9, generator=rng); t = torch.rand(2, generator=rng)
        sequences.append(hashlib.sha256(ids.numpy().tobytes() + noise.numpy().tobytes() + t.numpy().tobytes()).hexdigest())
        b = runner.subset(q, ids, 'cpu'); identity = runner.batch_identity(identities, b)
        affect = {'global': b['audio_global'], 'intensity_value': b['audio_intensity']}
        pred = local(b['audio_features'], b['valid'])['local']
        params = list(upper.parameters()) + [p for p in local.parameters() if p.requires_grad]
        loss = upper.flow_loss(runner.upper_target(b, data['target_scales']), b['valid'], b['h0'], identity['code'],
                               affect, pred, b['audio_state'], noise, t)
        runner.optimize(loss, torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5), params)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in upper.parameters())
        assert local.local_head.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in local.global_head.parameters())
        for name, module in [('system', system), ('audio', audio), ('adapter', adapter)]:
            assert state_hash(module.state_dict()) == frozen[name]
            assert all(p.grad is None for p in module.parameters())
    assert init[0] == init[1] and sequences[0] == sequences[1]


def test_completed_alignment_resume_exports_without_another_optimizer_step(tmp_path, monkeypatch):
    data, audio, _ = fixture()
    system_state = copy.deepcopy(data['system'].state_dict())
    source_audio_state = copy.deepcopy(audio.state_dict())
    bindings = {'teacher': {'path': 'fixture_teacher', 'sha256': 'teacher'}, 'audio': {'path': 'fixture_audio', 'sha256': 'audio'}}
    def load(*args, **kwargs):
        fresh = copy.deepcopy(data)
        fresh['system'].load_state_dict(system_state)
        return fresh
    def matching(run, loaded):
        return {'teacher': {'system': system_state}, 'audio': {'audio': source_audio_state}}, bindings, 2, 4
    monkeypatch.setattr(runner, 'load_training_inputs', load)
    monkeypatch.setattr(runner, 'matching_checkpoints', matching)
    args = SimpleNamespace(source_run=Path('source'), audio=Path('audio'), targets=Path('targets'),
                           enrollment=Path('enrollment'), native_root=Path('native'), trained_run=Path('run12'),
                           output=tmp_path/'align', phase='align', align_run=None, epochs=1,
                           batch_size=2, seed=53, device='cpu', smoke=True, resume=False)
    monkeypatch.setattr(runner, 'arguments', lambda: args)
    runner.main()
    first = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
    expected_adapter = state_hash(first['adapter'])
    assert first['completed_epochs'] == 1 and first['total_steps'] == 1
    args.resume = True
    runner.main()
    final = torch.load(args.output/'final.pt', map_location='cpu', weights_only=False)
    assert final['completed_epochs'] == 1 and final['total_steps'] == 1
    assert state_hash(final['adapter']) == expected_adapter
    assert json.loads((args.output/'status.json').read_text())['status'] == 'complete'
