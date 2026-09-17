import torch

from kinetalk_b0.predictable_motion import fit_motion_path, predict_motion
from kinetalk_b0.temporal_motion_refiner import ConstrainedRidgeRefiner, fit_control_scale
from scripts.train_constrained_audio_refiner_probe import train_arm
from scripts.train_temporal_audio_refiner_probe import buffer_hash


def sample():
    rng = torch.Generator().manual_seed(17)
    x = torch.randn(5, 9, 10, generator=rng, dtype=torch.float64)
    y = x @ (.1 * torch.randn(10, 12, generator=rng, dtype=torch.float64))
    y += .1 * torch.randn(y.shape, generator=rng, dtype=torch.float64)
    w = torch.full((5, 9), 4., dtype=torch.float64)
    w[:, -1] = 0
    fit = torch.arange(3)
    state = fit_motion_path(x, y, w, fit, [1.], [8], methods=("rrr",))[0]
    return x, y, w, fit, state


def test_exact_ridge_init_and_masked_temporal_dependency():
    x, y, w, fit, state = sample()
    scale = fit_control_scale(y, w, fit, state)
    for temporal, parameters in ((False, 64), (True, 192)):
        model = ConstrainedRidgeRefiner(state, scale, temporal=temporal)
        assert sum(p.numel() for p in model.parameters()) == parameters
        torch.testing.assert_close(model(x, w), predict_motion(x, w, state), rtol=0, atol=0)
        with torch.no_grad():
            model.correction.weight.fill_(.05)
        clean = model(x, w)
        poisoned = x.clone(); poisoned[w == 0] = float("nan")
        torch.testing.assert_close(model(poisoned, w), clean, rtol=0, atol=0)
        torch.testing.assert_close((clean * w[..., None]).sum(1), torch.zeros_like(clean[:, 0]), atol=1e-7, rtol=0)


def test_eight_epoch_only_fit_targets_and_frozen_buffers(tmp_path):
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        x, y, w, fit, state = sample()
        scale = fit_control_scale(y, w, fit, state)
        models = [ConstrainedRidgeRefiner(state, scale) for _ in range(2)]
        before = buffer_hash(models[0])
        reference, history = train_arm(models[0], x, y, w, fit, seed=46, device="cpu",
            output=tmp_path / "normal", provenance={}, fold=0, arm="temporal")
        assert len(history) == 8 and history[-1]["step"] == 8
        assert buffer_hash(reference) == before
        assert reference.correction.weight.abs().sum() > 0
        x[3:], y[3:], w[3:] = float("nan"), float("nan"), float("nan")
        repeat, replay = train_arm(models[1], x, y, w, fit, seed=46, device="cpu",
            output=tmp_path / "poison", provenance={}, fold=0, arm="temporal")
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, repeat.state_dict()[key], rtol=0, atol=0)
        assert [r["online_native_mse"] for r in history] == [r["online_native_mse"] for r in replay]
    finally:
        torch.set_num_threads(prior)
