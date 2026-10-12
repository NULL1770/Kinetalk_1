import torch
from kinetalk_v4.rollout_statistics import expression_mean_loss, pooled_state_loss


def test_rollout_mean_accepts_different_event_times_without_pointwise_fit():
    target = torch.zeros(1, 12, 3)
    target[:, 2:4, 0] = .8
    prediction = target.roll(5, 1).requires_grad_()
    valid = torch.ones(1, 12, dtype=torch.bool)
    channels = torch.ones(1, 3, dtype=torch.bool)
    assert not torch.equal(prediction, target)
    assert expression_mean_loss(prediction, target, valid, channels, torch.ones(3)) == 0


def test_standing_expression_bias_receives_gradient_with_missing_values_masked():
    target = torch.zeros(2, 10, 3)
    prediction = torch.ones_like(target, requires_grad=True)
    valid = torch.ones(2, 10, dtype=torch.bool); valid[1, -2:] = False
    channels = torch.tensor([[True, True, False], [True, True, False]])
    with torch.no_grad(): prediction[:, :, 2] = float('nan')
    loss = expression_mean_loss(prediction, target, valid, channels, torch.ones(3))
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad[:, :, :2].abs().sum() > 0
    assert prediction.grad[:, :, 2].eq(0).all()
    assert prediction.grad[1, -2:].eq(0).all()


def test_pooled_state_quality_does_not_impute_unobserved_axes():
    valid = torch.ones(1, 8, dtype=torch.bool)
    target = torch.zeros(1, 8, 2)
    quality = torch.ones_like(target, dtype=torch.bool); quality[:, :, 1] = False
    prediction = target.clone(); prediction[:, :, 1] = 100
    assert pooled_state_loss(prediction, target, valid, quality) == 0
