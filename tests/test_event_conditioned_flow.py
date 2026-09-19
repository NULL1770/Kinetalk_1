import numpy as np
import torch
from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow
from kinetalk_b0.models.event_conditioned_flow import EventConditionedFlow, shift_schedule


def test_prior_copy_starts_identical_and_receives_condition_gradients():
    torch.manual_seed(8)
    prior = ContinuousLatentFlow(context_dim=3, hidden=16, depth=1)
    saved = {k: v.clone() for k, v in prior.state_dict().items()}
    model = EventConditionedFlow.from_prior(prior)
    valid = torch.ones(2, 4, dtype=torch.bool)
    context = torch.randn(2, 3); noise = torch.randn(2, 4, 16)
    event = torch.rand(2, 4, 5, 12); audio = torch.zeros(2, 4, 5, 1540)
    a = prior.sample(valid, context, audio, noise, steps=2, use_audio=False)
    b = model.sample(valid, context, event, noise, steps=2)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    loss = model.flow_loss(torch.randn_like(noise), valid, context, event, noise, torch.full((2,), .5))
    loss.backward()
    assert model.audio_block_projection.weight.grad.abs().sum() > 0
    for k, v in prior.state_dict().items():
        torch.testing.assert_close(v, saved[k], rtol=0, atol=0)


def test_shift_is_run_local_and_never_wraps():
    x = np.zeros((15, 12)); x[2] = 1; x[11] = 2
    valid = np.ones(15, bool); valid[6:9] = False
    y = shift_schedule(x, valid, 3)
    assert (y[5] == 1).all() and (y[14] == 2).all()
    assert not y[6:9].any()
    z = shift_schedule(x, valid, -4)
    assert not z.any()
