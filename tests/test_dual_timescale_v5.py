"""Unit checks for the zero-start dual-timescale v5 candidate."""
from dataclasses import asdict

import torch

from kinetalk_v2.unified_model import teacher_targets
from kinetalk_v4.unified_model_v4 import KineTalkV4Config, KineTalkV4Model
from kinetalk_v5.dual_timescale_model import (
    DualTimescaleKineTalkV5, KineTalkV5Config, temporal_lowpass, v5_losses,
    envelope_guidance, supported_envelope,
)
from tests.test_unified_v4 import fixture


def _batch():
    old, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    times = torch.arange(motion.shape[1], dtype=torch.float64)[None].expand(2, -1) / 25
    batch = dict(audio_features=audio, h0=torch.randn_like(audio[..., :cfg.content_dim]),
        b0=base, motion=motion, valid=valid, times=times,
        channel_mask=torch.ones(2, 52, dtype=torch.bool),
        emotion_id=torch.tensor([1, 2]), intensity_id=torch.tensor([1, 2]),
        intensity_valid=torch.ones(2, dtype=torch.bool))
    targets = teacher_targets(motion, base, valid, batch['channel_mask'], times)
    checkpoint = dict(model=old.state_dict(), config=asdict(cfg))
    model = DualTimescaleKineTalkV5.from_v4_checkpoint(checkpoint)
    return old, model, refs, batch, targets


def test_lowpass_does_not_cross_invalid_gap():
    value = torch.tensor([[[1.], [1.], [100.], [3.], [3.]]])
    valid = torch.tensor([[True, True, False, True, True]])
    out = temporal_lowpass(value, valid, 5)
    assert torch.allclose(out[0, 0], torch.tensor([1.]))
    assert torch.allclose(out[0, 1], torch.tensor([1.]))
    assert torch.allclose(out[0, 3], torch.tensor([3.]))
    assert torch.allclose(out[0, 4], torch.tensor([3.]))
    assert torch.equal(out[~valid], torch.zeros(1, 1))


def test_v4_checkpoint_is_exact_zero_start_continuation():
    old, new, refs, batch, _ = _batch()
    old.eval(); new.eval()
    with torch.no_grad():
        old_out = old.predict(batch['audio_features'], batch['h0'], batch['b0'],
                              batch['valid'], refs, times=batch['times'], seed=47)
        new_out = new.predict(batch['audio_features'], batch['h0'], batch['b0'],
                              batch['valid'], refs, times=batch['times'], seed=47)
    for key in ('motion', 'events', 'geometry_target'):
        assert torch.allclose(old_out[key], new_out[key], atol=2e-6, rtol=0)


def test_new_residual_heads_receive_finite_nonzero_gradients():
    _, model, refs, batch, targets = _batch()
    out = model.training_forward(batch, refs, targets)
    losses = v5_losses(model, out, targets, batch)
    losses['total'].backward()
    for head in (model.decoder.affect_adapter, model.decoder.phonetic_adapter,
                 model.envelope_head):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0


def test_predict_does_not_require_motion_teacher_or_query_motion():
    _, model, refs, batch, _ = _batch()
    model.eval()
    model.motion_teacher = lambda _: (_ for _ in ()).throw(AssertionError('teacher used'))
    with torch.no_grad():
        out = model.predict(batch['audio_features'], batch['h0'], batch['b0'],
                            batch['valid'], refs, times=batch['times'])
    assert out['motion'].shape == batch['motion'].shape
    assert 'teacher' not in out


def test_lowpass_respects_timestamp_gap_without_invalid_frame():
    x = torch.tensor([[[1.], [1.], [3.], [3.]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    times = torch.tensor([[0., .04, 1., 1.04]], dtype=torch.float64)
    assert torch.equal(temporal_lowpass(x, mask, 3, times), x)


def test_unsupported_coordinate_nan_does_not_contaminate_envelope():
    x = torch.tensor([[[1., 4.], [float('nan'), 4.], [3., 4.]]])
    quality = torch.tensor([[[1, 1], [0, 1], [1, 1]]], dtype=torch.bool)
    result = supported_envelope(x, quality, torch.ones(1, 3, dtype=torch.bool), 3)
    assert torch.equal(result, torch.tensor([[[1., 4.], [0., 4.], [3., 4.]]]))


def test_reliability_is_supported_fraction_and_teacher_is_detached():
    valid = torch.ones(1, 4, dtype=torch.bool)
    quality = torch.ones(1, 4, 1, dtype=torch.bool)
    student = torch.ones(1, 4, 1, requires_grad=True)
    teacher = torch.tensor([[[0.], [2.], [2.], [2.]]], requires_grad=True)
    guide = envelope_guidance(student, teacher, torch.zeros_like(teacher), quality, valid, 1)
    assert guide['reliable_fraction'].item() == .25
    assert guide['mask'].sum().item() == 1
    assert not guide['teacher'].requires_grad


def test_strict_loader_preserves_both_checkpoint_families():
    from scripts.load_candidate_model import load_candidate_model
    old, new, refs, batch, _ = _batch()
    for original in (old, new):
        checkpoint = dict(model=original.state_dict(), config=asdict(original.cfg))
        loaded = load_candidate_model(checkpoint)
        assert type(loaded) is type(original)
        original.eval(); loaded.eval()
        args = (batch['audio_features'], batch['h0'], batch['b0'], batch['valid'], refs)
        assert torch.equal(original.predict(*args, times=batch['times'])['motion'],
                           loaded.predict(*args, times=batch['times'])['motion'])


def _full_batch():
    import numpy as np
    from pathlib import Path
    old, new, refs, batch, targets = _batch()
    batch['speaker_id'] = torch.tensor([1, 2])
    asset = Path(__file__).resolve().parents[1] / 'final_experiment/rig/lip_constraints_v2_20261010.npz'
    with np.load(asset, allow_pickle=False) as z:
        channels = z['channels'].tolist()
    return old, new, refs, batch, targets, channels


def test_complete_control_objective_matches_previous_runner():
    from scripts.train_dual_timescale_v5 import full_objective
    from scripts.train_v5_mixed import training_objective
    old, _, refs, batch, targets, channels = _full_batch()
    torch.manual_seed(47)
    expected, _ = training_objective(old, batch, refs, targets, channels, .5, posture_weight=1.)
    torch.manual_seed(47)
    actual, _ = full_objective(old, batch, refs, targets, channels)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_complete_candidate_objective_trains_all_new_heads():
    from scripts.train_dual_timescale_v5 import full_objective
    _, model, refs, batch, targets, channels = _full_batch()
    loss, logs = full_objective(model, batch, refs, targets, channels)
    loss.backward()
    assert {'execution', 'relation', 'rollout_mean', 'envelope_distill'} <= set(logs)
    for head in (model.envelope_head, model.decoder.affect_adapter, model.decoder.phonetic_adapter):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum().item() > 0
