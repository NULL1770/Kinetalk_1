"""Numerical and gradient contracts for checkpointed audio residual decoding.

These tests compare the memory-saving rollout with the pre-checkpoint direct
rollout under the same stochastic conditions.  The objective contains the
two-draw raw and centered fair energy scores as well as flow matching, so an
apparently equivalent forward pass cannot hide a missing parameter or
condition gradient.
"""

from __future__ import annotations

import copy

import pytest
import torch

import kinetalk_b0.models.audio_residual_flow as audio_flow
from kinetalk_b0.models.audio_residual_flow import AudioResidualFlow, fair_trajectory_es


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _config():
    return {
        "content_dim": 8,
        "emotion_dim": 8,
        "style_dim": 8,
        "dit_dim": 12,
        "dit_depth": 1,
        "heads": 3,
        # Nonzero dropout makes the RNG-preservation part of checkpointing
        # observable while remaining small enough for a unit test.
        "dropout": 0.2,
    }


def _inputs():
    torch.manual_seed(904)
    batch, frames = 2, 7
    valid = torch.tensor(
        [[True, True, False, True, True, True, False],
         [True, False, True, True, True, False, True]],
        dtype=torch.bool,
    )
    q = {
        "valid": valid,
        "h0": torch.randn(batch, frames, 8),
        "channel_mask": torch.ones(batch, 52, dtype=torch.bool),
    }
    # Conditions at invalid native positions are irrelevant, but are made
    # explicit zeros so the test isolates the decoder's frame mask.
    q["h0"] = torch.where(valid[..., None], q["h0"], torch.zeros_like(q["h0"]))
    identity = {"code": torch.randn(batch, 8)}
    affect = {"global": torch.randn(batch, 8), "intensity_value": torch.randn(batch, 1)}
    local = torch.randn(batch, frames, 8)
    local = torch.where(valid[..., None], local, torch.zeros_like(local)).requires_grad_()
    state = torch.randn(batch, frames, 4)
    state = torch.where(valid[..., None], state, torch.zeros_like(state)).requires_grad_()

    noise1 = torch.randn(batch, frames, 9)
    noise2 = torch.randn(batch, frames, 9)
    target = torch.randn(batch, frames, 9)
    # NaNs in unobserved payloads ensure every loss and rollout operation uses
    # the Boolean frame mask instead of accidentally reading padding.
    noise1 = noise1.masked_fill(~valid[..., None], float("nan"))
    noise2 = noise2.masked_fill(~valid[..., None], float("nan"))
    target = target.masked_fill(~valid[..., None], float("nan"))
    times = torch.tensor([0.23, 0.81])
    return q, identity, affect, local, state, noise1, noise2, target, times


def _run_objective(model, values, *, checkpoint_steps):
    q, identity, affect, local, state, noise1, noise2, target, times = values
    # Deliberately keep the first noise draw detached: this is the case where
    # a reentrant checkpoint implementation can silently drop model grads.
    assert not noise1.requires_grad
    draw1 = model.decode(q, identity, affect, local, state, noise1, 3,
                         checkpoint_steps=checkpoint_steps)
    draw2 = model.decode(q, identity, affect, local, state, noise2, 3,
                         checkpoint_steps=checkpoint_steps)
    samples = torch.stack((draw1, draw2))
    raw = fair_trajectory_es(samples, target, q["valid"])
    centered = fair_trajectory_es(samples, target, q["valid"], centered=True)
    fm = model.flow_loss(target, q, identity, affect, local, state, noise1, times)
    return raw + centered + fm, (draw1, draw2, raw, centered, fm)


def _grad_snapshot(model, local, state):
    return {
        "model": {name: None if parameter.grad is None else parameter.grad.detach().clone()
                  for name, parameter in model.named_parameters()},
        "local": None if local.grad is None else local.grad.detach().clone(),
        "state": None if state.grad is None else state.grad.detach().clone(),
    }


def test_checkpoint_rollout_matches_direct_fair_and_flow_objective_with_masks():
    """Checkpointing preserves two-draw losses and all trainable gradients."""
    values = _inputs()
    direct = AudioResidualFlow(_config(), stride=4).train()
    checkpointed = AudioResidualFlow(_config(), stride=4).train()
    checkpointed.load_state_dict(copy.deepcopy(direct.state_dict()))

    # A common seed makes dropout masks identical in the direct and
    # checkpointed forward passes. preserve_rng_state must then reproduce the
    # same masks during checkpoint backward recomputation as well.
    torch.manual_seed(2217)
    direct_loss, direct_parts = _run_objective(direct, values, checkpoint_steps=False)
    direct_loss.backward()
    direct_grads = _grad_snapshot(direct, values[3], values[4])

    # Inputs must be fresh because the first backward populated local/state
    # gradients in the direct run.
    values_checkpointed = _inputs()
    torch.manual_seed(2217)
    checkpoint_loss, checkpoint_parts = _run_objective(
        checkpointed, values_checkpointed, checkpoint_steps=True
    )
    checkpoint_loss.backward()
    checkpoint_grads = _grad_snapshot(checkpointed, values_checkpointed[3], values_checkpointed[4])

    torch.testing.assert_close(checkpoint_loss, direct_loss, rtol=2e-5, atol=2e-6)
    for got, expected in zip(checkpoint_parts, direct_parts):
        torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)
    assert direct_grads["model"].keys() == checkpoint_grads["model"].keys()
    for name in direct_grads["model"]:
        got, expected = checkpoint_grads["model"][name], direct_grads["model"][name]
        assert (got is None) == (expected is None), name
        if got is not None:
            torch.testing.assert_close(got, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(checkpoint_grads["local"], direct_grads["local"], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(checkpoint_grads["state"], direct_grads["state"], rtol=3e-5, atol=3e-6)
    assert checkpoint_grads["local"].abs().sum() > 0
    assert checkpoint_grads["state"].abs().sum() > 0


def test_checkpoint_is_nonreentrant_per_step_and_no_grad_bypasses_it(monkeypatch):
    """Training uses non-reentrant checkpoint once per step; inference does not."""
    values = _inputs()
    q, identity, affect, local, state, noise1, *_ = values
    model = AudioResidualFlow(_config(), stride=4).train()
    original = audio_flow.checkpoint
    calls = []

    def spy(function, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(function, *args, **kwargs)

    monkeypatch.setattr(audio_flow, "checkpoint", spy)
    torch.manual_seed(77)
    result = model.decode(q, identity, affect, local, state, noise1, 4)
    assert result.shape == noise1.shape
    assert len(calls) == 4
    assert all(call.get("use_reentrant") is False for call in calls)
    assert all(call.get("preserve_rng_state") is True for call in calls)

    # Inference must take the direct path even when checkpoint_steps remains
    # at its training default.  It also has to agree with the explicit legacy
    # path, including masked frames.
    calls.clear()
    model.eval()
    with torch.no_grad():
        torch.manual_seed(77)
        inferred = model.decode(q, identity, affect, local, state, noise1, 4)
        torch.manual_seed(77)
        legacy = model.decode(q, identity, affect, local, state, noise1, 4,
                              checkpoint_steps=False)
    assert calls == []
    torch.testing.assert_close(inferred, legacy, rtol=0, atol=0)
    assert inferred[~q["valid"]].count_nonzero() == 0
