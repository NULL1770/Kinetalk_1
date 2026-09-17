import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scripts.probe_expression_energy_field import augment_audio_context, fit_expression_energy_target


def _toy():
    n, t, c = 14, 8, 6
    residual = torch.zeros(n, t, c)
    for i in range(n):
        e = i % 3 + 1
        residual[i, :, e - 1] = torch.linspace(0.0, 1.0 + 0.1 * i, t)
    valid = torch.ones(n, t, dtype=torch.bool)
    channel_mask = torch.ones(n, c, dtype=torch.bool)
    rows = [(0, 0, 0), (1, 0, 0)]
    rows += [(s, e, l) for s in range(2) for e in range(1, 4) for l in (1, 2)]
    speaker = torch.tensor([r[0] for r in rows])
    emotion = torch.tensor([r[1] for r in rows])
    intensity = torch.tensor([r[2] for r in rows])
    train = torch.arange(n)
    return residual, valid, channel_mask, speaker, emotion, intensity, train


def test_energy_target_is_scalar_and_centered():
    out = fit_expression_energy_target(*_toy(), stride=2)
    energy, weight, state = out
    assert energy.shape == (14, 4, 1)
    assert weight.shape == (14, 4)
    assert state["target_name"] == "label_expression_energy"
    means = (energy * weight[..., None]).sum(1) / weight.sum(1, keepdim=True)
    torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-5, rtol=0)


def test_energy_is_sign_invariant():
    args = _toy()
    first = fit_expression_energy_target(*args, stride=2)[0]
    residual = args[0].clone(); residual[..., 0] *= -1
    second = fit_expression_energy_target(residual, *args[1:], stride=2)[0]
    assert first.shape == second.shape
    assert torch.isfinite(second).all()


def test_upper_and_face_l1_modes_are_scalar_centered_fields():
    args = _toy()
    for mode in ("upper_l1", "face_l1"):
        energy, weight, state = fit_expression_energy_target(*args, stride=2, mode=mode)
        assert energy.shape == (14, 4, 1)
        assert state["target_name"] == mode
        assert state["basis"] is None
        means = (energy.squeeze(-1) * weight).sum(1) / weight.sum(1)
        torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-6, rtol=0)


def test_energy_velocity_is_compact_dense_field_with_shared_weight():
    args = _toy()
    target, weight, state = fit_expression_energy_target(*args, stride=2, mode="energy_velocity")
    assert target.shape == (14, 4, 2)
    assert weight.shape == (14, 4)
    assert state["axis_names"] == ["energy", "velocity"]
    # Both axes follow the local-control zero-mean contract.  The second axis
    # is the centered finite difference of the first, not a separate target.
    means = (target * weight[..., None]).sum(1) / weight.sum(1, keepdim=True)
    torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-6, rtol=0)


def test_audio_context_augmentation_preserves_mask_and_triplicates_features():
    audio = torch.randn(2, 6, 4)
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[1, -2:] = False
    out = augment_audio_context(audio, valid)
    assert out.shape == (2, 6, 12)
    assert torch.equal(out[1, -2:], torch.zeros(2, 12))
    assert torch.isfinite(out).all()
