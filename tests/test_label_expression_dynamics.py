"""Unit tests for the label-associated proxy target."""
import pytest
import torch

from scripts.probe_label_expression_dynamics import (
    RANK, basis_diagnostics, fit_label_basis,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        yield
    torch.set_num_threads(old)


def synthetic_labels():
    # Three speakers, three nonneutral emotions, two levels per cell.
    rows, values = [], []
    directions = torch.tensor([[2., 0., 0., 0., 0.], [0., 3., 0., 0., 0.], [0., 0., 4., 0., 0.]])
    for speaker in range(3):
        for emotion in range(4):
            levels = (0,) if emotion == 0 else (1, 3)
            for level in levels:
                for repeat in range(2):
                    rows.append((speaker, emotion, level))
                    offset = torch.tensor([.1 * speaker, -.05 * speaker, .02 * speaker, 0., 0.])
                    base = offset if emotion == 0 else offset + directions[emotion - 1] * (1. + .1 * level)
                    values.append(base.repeat(8, 1) + torch.randn(8, 5) * .01)
    residual = torch.stack(values)
    valid = torch.ones(len(values), 8, dtype=torch.bool)
    mask = torch.ones(len(values), 5, dtype=torch.bool)
    return residual, valid, mask, torch.tensor([r[0] for r in rows]), torch.tensor([r[1] for r in rows]), torch.tensor([r[2] for r in rows])


def test_label_basis_uses_neutral_anchor_and_has_rank_three_proxy():
    residual, valid, mask, speaker, emotion, level = synthetic_labels()
    train = torch.arange(len(residual))
    target, weight, state = fit_label_basis(residual, valid, mask, speaker, emotion, level, train, stride=4)
    assert target.shape == (len(residual), 2, RANK)
    assert weight.shape == (len(residual), 2)
    assert state["basis"].shape == (5, RANK)
    assert torch.allclose((target * weight[..., None]).sum(1), torch.zeros_like(target[:, 0]), atol=1e-5)
    assert len(state["training_cells"]) == 3 * (1 + 3 * 2)
    diag = basis_diagnostics(state["basis"], state["channel_indices"], target, weight, train)
    assert sum(diag["energy_weighted_squared_loading_mass"].values()) == pytest.approx(1., abs=1e-5)


def test_label_basis_rejects_missing_emotion_or_rank_deficiency():
    residual, valid, mask, speaker, emotion, level = synthetic_labels()
    train = torch.arange(len(residual))
    with pytest.raises(ValueError, match="exactly three"):
        fit_label_basis(residual, valid, mask, speaker, emotion.clamp_max(2), level, train, stride=4)
    # Keep three labels but make the third motion direction equal to the first.
    collapsed = torch.zeros_like(residual)
    collapsed[emotion == 1, :, 0] = 2.
    collapsed[emotion == 2, :, 1] = 3.
    collapsed[emotion == 3, :, 0] = 2.
    with pytest.raises(ValueError, match="rank deficient"):
        fit_label_basis(collapsed, valid, mask, speaker, emotion, level, train, stride=4)
