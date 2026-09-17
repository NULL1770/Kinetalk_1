"""CPU regression of the real staged optimizer/checkpoint/RNG mechanism."""
import copy
import random

import numpy as np
import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect, compose_upper_face
from scripts.train_formal_predictable_projection import capture_rng, restore_rng, save_checkpoint
from scripts.train_full_staged import (
    NOT_UPPER, UpperFlow, base_forward, huber, optimize, subset, targets, upper_motion,
)
from scripts.train_predictable_renderer import state_hash
from tests.test_full_staged_runner import config


@pytest.fixture(autouse=True)
def one_cpu_thread_and_restore_rng():
    threads = torch.get_num_threads()
    rng = torch.Generator().manual_seed(501)
    state = capture_rng(rng)
    torch.set_num_threads(1)
    yield
    restore_rng(state, rng)
    torch.set_num_threads(threads)


def make_modules():
    cfg = config()
    system = NeutralAffectSystem(cfg).eval().requires_grad_(False)
    global_audio = SlowStateAffect(torch.zeros(10), torch.ones(10), global_dim=8,
                                  local_dim=8, hidden=8, num_emotions=2, stride=4).eval().requires_grad_(False)
    local_audio = copy.deepcopy(global_audio).requires_grad_(True)
    upper = UpperFlow(cfg, stride=4).eval()
    modules = {"system": system, "audio": global_audio, "local_audio": local_audio, "upper": upper}
    optimizer = torch.optim.AdamW([
        {"params": local_audio.parameters(), "lr": 1e-4},
        {"params": upper.parameters(), "lr": 1e-4},
    ], weight_decay=1e-5)
    return modules, optimizer


def fixture():
    random.seed(81); np.random.seed(82); torch.manual_seed(83)
    modules, optimizer = make_modules()
    rng = torch.Generator().manual_seed(84)
    data_rng = torch.Generator().manual_seed(85)
    valid = torch.ones(3, 12, dtype=torch.bool)
    valid[0, 5] = False; valid[1, -2:] = False
    anchor = torch.full((3, 52), .2)
    q = {"motion": anchor[:, None] + .03 * torch.randn(3, 12, 52, generator=data_rng),
         "content": torch.randn(3, 12, 8, generator=data_rng),
         "audio_features": torch.randn(3, 12, 10, generator=data_rng),
         "valid": valid, "channel_mask": torch.ones(3, 52, dtype=torch.bool),
         "anchors": anchor, "anchor_valid": torch.ones(3, 52, dtype=torch.bool)}
    with torch.no_grad():
        q.update(base_forward(modules["system"], q["content"], valid))
        identity = modules["system"].encode_identity(torch.randn(3, 2, 12, 52, generator=data_rng))
    identity = {k: identity[k] for k in ("code", "baseline")}
    return modules, optimizer, rng, q, identity, torch.linspace(.03, .12, 52)


def stage5_step(modules, optimizer, rng, q, identity, scales):
    """Use the runner's flow/state objective and explicit random draw order."""
    ids = torch.randperm(len(q["valid"]), generator=rng)[:2]
    batch = subset(q, ids, "cpu")
    ident = {k: v[ids] for k, v in identity.items()}
    noise = torch.randn(len(ids), q["valid"].shape[1], 9, generator=rng)
    flow_time = torch.rand(len(ids), generator=rng)
    with torch.no_grad():
        affect = modules["audio"](batch["audio_features"], batch["valid"])
    dynamic = modules["local_audio"](batch["audio_features"], batch["valid"])
    truth, innovation = targets(batch, scales, 4)
    flow = modules["upper"].flow_loss(innovation, batch, ident, affect, dynamic["local"],
                                       dynamic["state"], noise, flow_time)
    loss = flow + .5 * huber(dynamic["state"], truth["state"], truth["state_mask"])
    parameters = list(modules["local_audio"].parameters()) + list(modules["upper"].parameters())
    grad_norm = optimize(loss, optimizer, parameters)
    return loss.detach().clone(), grad_norm


def next_draws(rng):
    return {"python": random.random(), "numpy": np.random.rand(4),
            "torch": torch.rand(4), "training_generator": torch.rand(4, generator=rng)}


def assert_same_tree(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_same_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b): assert_same_tree(left, right)
    else:
        assert a == b


def test_optimizer_model_and_all_rng_resume_matches_uninterrupted_next_step(tmp_path):
    modules, optimizer, rng, q, identity, scales = fixture()
    stage5_step(modules, optimizer, rng, q, identity, scales)
    # Checkpoint state must include nonzero Adam moments, not just weights.
    assert optimizer.state and any(state["exp_avg"].abs().sum() > 0 for state in optimizer.state.values())
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"models": {k: m.state_dict() for k, m in modules.items()},
                           "optimizer": optimizer.state_dict(), "rng": capture_rng(rng),
                           "stage_index": 4, "completed_epochs": 1})
    expected_loss, expected_norm = stage5_step(modules, optimizer, rng, q, identity, scales)
    expected_models = {k: copy.deepcopy(m.state_dict()) for k, m in modules.items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_draws = next_draws(rng)

    # Reconstructing modules consumes global Torch RNG. Restoration must occur
    # after constructors, model loading and optimizer creation/loading.
    restored, restored_optimizer = make_modules()
    restored_rng = torch.Generator().manual_seed(9999)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for name, model in restored.items(): model.load_state_dict(payload["models"][name], strict=True)
    restored_optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"], restored_rng)
    actual_loss, actual_norm = stage5_step(restored, restored_optimizer, restored_rng, q, identity, scales)
    assert torch.equal(actual_loss, expected_loss)
    assert actual_norm == expected_norm
    assert_same_tree({k: m.state_dict() for k, m in restored.items()}, expected_models)
    assert_same_tree(restored_optimizer.state_dict(), expected_optimizer)
    actual_draws = next_draws(restored_rng)
    assert actual_draws["python"] == expected_draws["python"]
    np.testing.assert_array_equal(actual_draws["numpy"], expected_draws["numpy"])
    for key in ("torch", "training_generator"):
        assert torch.equal(actual_draws[key], expected_draws[key])
    assert not path.with_name("last.pt.tmp").exists()


def test_stage5_updates_dynamic_branch_preserves_frozen_global_system_and_nonupper():
    modules, optimizer, rng, q, identity, scales = fixture()
    frozen_before = {name: state_hash(modules[name].state_dict()) for name in ("system", "audio")}
    dynamic_before = {name: state_hash(modules[name].state_dict()) for name in ("local_audio", "upper")}
    with torch.no_grad():
        affect_before = modules["audio"](q["audio_features"], q["valid"])
    for _ in range(2): stage5_step(modules, optimizer, rng, q, identity, scales)
    for name in ("system", "audio"):
        assert state_hash(modules[name].state_dict()) == frozen_before[name]
        assert all(p.grad is None for p in modules[name].parameters())
        assert not any(p.requires_grad for p in modules[name].parameters())
    for name in ("local_audio", "upper"):
        assert state_hash(modules[name].state_dict()) != dynamic_before[name]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in modules[name].parameters())
    with torch.no_grad():
        affect_after = modules["audio"](q["audio_features"], q["valid"])
        assert_same_tree(affect_before, affect_after)
        dynamic = modules["local_audio"](q["audio_features"], q["valid"])
        noise = torch.randn(3, 12, 52, generator=torch.Generator().manual_seed(42))
        base = {k: q[k] for k in ("b0", "h0")}
        baseline = modules["system"].generate(q["content"], q["valid"], identity, affect_after,
                                              initial_noise=noise, steps=2, base=base)["motion"]
        innovation = modules["upper"].decode(q, identity, affect_after, dynamic["local"], dynamic["state"],
                                              noise[..., :9], 2)
        composed = compose_upper_face(baseline, upper_motion(q, scales, dynamic["state"], innovation), q["valid"])
    assert torch.equal(composed[..., list(NOT_UPPER)], baseline[..., list(NOT_UPPER)])
