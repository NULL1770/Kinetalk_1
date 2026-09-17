import torch
import pytest

from scripts.prepare_renderer_capacity_controls import (
    fit_fixed_coordinates, predict_fixed_coordinates, crossfit_controls,
)
from kinetalk_b0.predictable_motion import fit_motion_path, weighted_clip_center


def fixture():
    g = torch.Generator().manual_seed(19)
    x, y = torch.randn(9, 5, 7, generator=g), torch.randn(9, 5, 6, generator=g)
    w = torch.ones(9, 5); w[2, -2:] = 0; w[5, -1:] = 0
    state = fit_motion_path(x, y, w, torch.arange(9), [1.], [3], methods=('rrr',))[0]
    return x, y, w, state


def test_allfit_coefficients_reproduce_existing_regularized_rrr():
    x, y, w, s = fixture()
    coeff = fit_fixed_coordinates(x, y, w, torch.arange(9), basis=s['basis'], channels=list(range(6)), feature_std=s['std'])
    torch.testing.assert_close(coeff, s['weights'], atol=1e-12, rtol=1e-10)


def test_fold_target_poison_cannot_enter_fit_or_audio_prediction():
    x, y, w, s = fixture()
    fit = torch.arange(6)
    kwargs = dict(basis=s['basis'], channels=list(range(6)), feature_std=s['std'])
    first = fit_fixed_coordinates(x, y, w, fit, **kwargs)
    poisoned = y.clone(); poisoned[6:] = float('nan')
    second = fit_fixed_coordinates(x, poisoned, w, fit, **kwargs)
    assert torch.equal(first, second)
    pred = predict_fixed_coordinates(x[6:], w[6:], first, s['std'], torch.ones(3))
    assert torch.isfinite(pred).all()


def test_crossfit_common_coordinates_and_exact_coverage_with_padding():
    x, y, w, s = fixture()
    folds = []
    for k in range(3):
        heldout = torch.arange(k*3, (k+1)*3)
        fit = torch.tensor([i for i in range(9) if i not in heldout])
        folds.append((fit, heldout))
    kwargs = dict(basis=s['basis'], channels=list(range(6)), feature_std=s['std'], target_scale=torch.ones(3))
    out, states = crossfit_controls(x, y, w, folds, **kwargs)
    assert out.shape == (9, 5, 3)
    assert torch.equal(out[w == 0], torch.zeros_like(out[w == 0]))
    torch.testing.assert_close((out*w[..., None]).sum(1), torch.zeros(9,3), atol=2e-7, rtol=0)
    for state in states:
        ids = state['heldout_ids']
        expected = predict_fixed_coordinates(x[ids], w[ids], state['coefficients'], s['std'], torch.ones(3))
        assert torch.equal(out[ids], expected)
    with pytest.raises(ValueError, match='exactly one'):
        crossfit_controls(x, y, w, folds[:2], **kwargs)
