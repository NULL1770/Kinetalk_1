from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from kinetalk_v2.unified_model import teacher_targets
from kinetalk_v4.geometry import load_rig_geometry
from kinetalk_v4.unified_model_v4 import (
    KineTalkV4Config, KineTalkV4Model, affine_geometry, v4_losses,
)


def fixture():
    torch.manual_seed(13)
    cfg = KineTalkV4Config(content_dim=12, hidden=24, decoder_hidden=32, style_dim=8)
    jac = torch.randn(4, 52) * .02
    off = torch.randn(4)
    model = KineTalkV4Model(cfg, torch.zeros(772), torch.ones(772), torch.ones(52) * .1, jac, off)
    b, t = 2, 19
    valid = torch.ones(b, t, dtype=torch.bool)
    valid[1, -3:] = False
    base = torch.rand(b, t, 52)
    motion = torch.clamp(base + torch.randn_like(base) * .02, 0, 1)
    audio = torch.randn(b, t, 1540)
    refs = {'motion': motion.flip(1), 'valid': valid.clone(),
            'channel_mask': torch.ones(b, 52, dtype=torch.bool)}
    return model, cfg, valid, base, motion, audio, refs, jac, off


def test_v4_zero_start_is_b0_and_has_full_support():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    content = torch.randn(2, 19, cfg.content_dim)
    out = model.predict(audio, content, base, valid, refs)
    assert torch.allclose(out['motion'][valid], base[valid], atol=1e-5)
    assert out['motion'].shape == motion.shape
    assert torch.equal(out['motion'][~valid], torch.zeros_like(out['motion'][~valid]))


def test_v4_geometry_projector_moves_measured_geometry_without_channel_mask():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    model.eval()
    with torch.no_grad():
        model.decoder.geometry_delta.bias.fill_(.5)
    content = torch.randn(2, 19, cfg.content_dim)
    out = model.predict(audio, content, base, valid, refs)
    before = affine_geometry(base, jac, off)
    after = affine_geometry(out['motion'], jac, off)
    assert (after[valid] - before[valid]).abs().sum() > 0
    # A geometry correction must be able to touch mouth/corner channels and
    # cannot be implemented as a fixed category mask.
    assert (out['motion'][valid] - base[valid]).abs().sum() > 0


def test_v4_student_path_and_factor_gradients():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    content = torch.randn(2, 19, cfg.content_dim)
    channel_mask = torch.ones(2, 52, dtype=torch.bool)
    times = torch.arange(19, dtype=torch.float64)[None].expand(2, -1) / 25
    # Give the target motion one complete, observable blink event so the
    # teacher-forced decoder and stochastic rollout have distinct schedules.
    motion = motion.clone()
    motion[..., 0] = 0.
    motion[..., 7] = 0.
    motion[:, 4:10, 0] = .7
    motion[:, 4:10, 7] = .7
    batch = {'audio_features': audio, 'h0': content, 'b0': base, 'motion': motion,
             'valid': valid, 'times': times, 'channel_mask': channel_mask,
             'emotion_id': torch.tensor([1, -1]), 'intensity_valid': torch.tensor([True, False]),
             'intensity_id': torch.tensor([2, -1])}
    targets = teacher_targets(motion, base, valid, channel_mask, times)
    out = model.training_forward(batch, refs, targets)
    assert torch.equal(out['events'], targets['events'])
    assert out['sampled_motion'].shape == motion.shape
    losses = v4_losses(model, out, targets, batch)
    sampled_motion = out['sampled_motion']
    out['sampled_motion'] = sampled_motion + 100.
    perturbed_sample_loss = v4_losses(model, out, targets, batch)
    assert torch.allclose(losses['position'], perturbed_sample_loss['position'])
    assert torch.allclose(losses['total'], perturbed_sample_loss['total'])
    out['sampled_motion'] = sampled_motion
    losses['total'].backward()
    assert model.student.state.weight.grad is not None
    assert model.decoder.output.weight.grad is not None
    assert model.decoder.output.weight.grad.abs().sum() > 0
    assert model.teacher.state.weight.grad is not None
    assert model.teacher.state.weight.grad.abs().sum() > 0


def test_v4_full_1540_input_has_no_text_branch():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    out = model.predict(audio, torch.randn(2, 19, cfg.content_dim), base, valid, refs)
    assert out['student']['state'].shape == (2, 19, cfg.state_dim)


def test_v4_marked_duration_sampler_is_native_and_seeded():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(8)
    content = torch.randn(2, 19, cfg.content_dim)
    a = model.predict(audio, content, base, valid, refs, generator=g1)
    b = model.predict(audio, content, base, valid, refs, generator=g2)
    assert a['events'].shape == (2, 19, 12)
    assert not torch.equal(a['events'], b['events'])
    assert torch.equal(a['events'][~valid], torch.zeros_like(a['events'][~valid]))


def test_v4_real_rig_geometry_matches_deformed_named_group_measurements():
    path = Path(__file__).resolve().parents[1] / 'final_experiment/rig/lip_constraints_v2_20261010.npz'
    jac, off, scale, channels, metadata = load_rig_geometry(path)
    coefficients = torch.rand(3, 52)
    with np.load(path, allow_pickle=False) as z:
        mesh = z['neutral_mm'][None] + np.einsum(
            'bc,cva->bva', coefficients.numpy(), z['blendshape_deltas_mm'])
        groups = z['group_names'].tolist()
        centroids = {}
        for name in ('lLipCorner', 'rLipCorner', 'LipUpperMiddle', 'LipLowerMiddle'):
            weights = z['lip_group_weights'][:, groups.index(name)]
            centroids[name] = np.einsum('v,bva->ba', weights / weights.sum(), mesh)
        left, right = centroids['lLipCorner'], centroids['rLipCorner']
        upper, lower = centroids['LipUpperMiddle'], centroids['LipLowerMiddle']
        sign = np.sign(off[0].item())
        expected = np.stack((
            sign * (left[:, 0] - right[:, 0]),
            (left[:, 2] + right[:, 2] - upper[:, 2] - lower[:, 2]) / 2,
            upper[:, 2] - lower[:, 2],
            (left[:, 1] + right[:, 1]) / 2,
        ), -1)
    assert torch.allclose(affine_geometry(coefficients, jac, off),
                          torch.from_numpy(expected.astype(np.float32)), atol=4e-4, rtol=1e-5)
    assert channels.shape == (52,)
    assert scale.shape == (4,)
    assert metadata['physiological_contact_claim'] is False
    assert metadata['scale_is_train_fitted'] is False


def test_v4_no_style_ablation_removes_reference_influence_in_motion_and_schedule():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    cfg = replace(cfg, use_style=False)
    model = KineTalkV4Model(cfg, torch.zeros(772), torch.ones(772),
                           torch.ones(52) * .1, jac, off).eval()
    # Randomize the output so equality cannot pass solely because of zero start.
    with torch.no_grad():
        model.decoder.output.weight.normal_(std=.01)
    alternate = {**refs, 'motion': refs['motion'] + .2}
    content = torch.randn(2, 19, cfg.content_dim)
    with torch.no_grad():
        first = model.predict(audio, content, base, valid, refs, seed=8)
        second = model.predict(audio, content, base, valid, alternate, seed=8)
    assert torch.equal(first['events'], second['events'])
    assert torch.equal(first['motion'], second['motion'])
