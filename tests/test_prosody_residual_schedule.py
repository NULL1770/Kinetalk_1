import copy

import torch

from kinetalk_b0.models.prosody_residual_schedule import (
    ProsodyResidualSchedule,
    StaticScheduleHead,
)


def test_zero_condition_is_exact_static_and_base_is_frozen():
    torch.manual_seed(19)
    base = StaticScheduleHead(context_dim=6, hidden=12)
    model = ProsodyResidualSchedule(base, condition_dim=10, hidden=12)
    condition = torch.randn(3, 27, 10)
    valid = torch.ones(3, 27, dtype=torch.bool)
    context = torch.randn(3, 6)
    static = base(context, 27)
    result = model(torch.zeros_like(condition), context, valid)
    torch.testing.assert_close(result['onset_logits'], static['onset_logits'], rtol=0, atol=0)
    torch.testing.assert_close(result['duration_logits'], static['duration_logits'], rtol=0, atol=0)
    assert all(not p.requires_grad for p in model.base.parameters())


def test_residual_updates_without_mutating_static_base_and_respects_gaps():
    torch.manual_seed(23)
    base = StaticScheduleHead(context_dim=4, hidden=8)
    model = ProsodyResidualSchedule(base, condition_dim=10, hidden=8)
    before = copy.deepcopy(base.state_dict())
    condition = torch.randn(2, 31, 10)
    valid = torch.ones(2, 31, dtype=torch.bool)
    valid[:, 13:17] = False
    context = torch.randn(2, 4)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    out = model(condition, context, valid)
    target = torch.zeros_like(out['onset_logits'])
    loss = out['onset_logits'][valid].square().mean() + out['duration_logits'][valid].square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert any(p.grad is not None for p in model.input.parameters())
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not model.training or not model.base.training
    assert torch.equal(model(condition, context, valid)['onset_logits'][:, 13:17],
                       static_zero := model.base(context, 31)['onset_logits'][:, 13:17])

