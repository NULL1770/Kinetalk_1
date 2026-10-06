import pytest
import torch
from torch.nn import functional as F
from scripts.train_full_staged import global_distillation, parser


def test_default_exactly_preserves_existing_mse_and_gradient():
    p = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    t = torch.tensor([[0., 2.], [5., 3.]])
    logits = torch.tensor([[1., 0.], [1., 0.]])
    loss, retained = global_distillation(p, t, logits, torch.tensor([0, 1]))
    torch.testing.assert_close(loss, F.mse_loss(p, t), rtol=0, atol=0)
    assert retained == 1
    assert parser().parse_args(['--output', 'tmp']).global_distill_gate == 'all'


def test_gate_omits_wrong_teacher_gradient_without_upweighting_valid_clip():
    p = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    t = torch.zeros_like(p, requires_grad=True)
    logits = torch.tensor([[2., 0.], [2., 0.]], requires_grad=True)
    loss, retained = global_distillation(p, t, logits, torch.tensor([0, 1]),
                                       gate='teacher-agreement')
    assert retained == .5
    torch.testing.assert_close(loss, (p[0] ** 2).mean() / 2)
    loss.backward()
    torch.testing.assert_close(p.grad[0], p.detach()[0] / 2)
    assert p.grad[1].count_nonzero() == 0
    assert logits.grad is None and t.grad is None


def test_all_rejected_batch_has_finite_zero_gradient_and_unknown_gate_fails():
    p = torch.ones(2, 3, requires_grad=True)
    logits = torch.tensor([[1., 0.], [1., 0.]])
    loss, retained = global_distillation(p, torch.zeros_like(p), logits,
                                       torch.tensor([1, 1]), gate='teacher-agreement')
    assert torch.isfinite(loss) and loss == 0 and retained == 0
    loss.backward()
    assert p.grad.count_nonzero() == 0
    with pytest.raises(ValueError):
        global_distillation(p, p, logits, torch.tensor([1, 1]), gate='invalid')
