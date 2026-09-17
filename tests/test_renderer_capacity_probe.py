import copy

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.train_predictable_renderer import PredictableAudioHead, projected_affect, state_hash
from scripts.train_projection_schedule_ablation import draws
from scripts.train_renderer_capacity_probe import (
    ARMS, CONTROL_SCHEMA, PREFIXES, assert_frozen, capacity_flow_loss,
    capacity_state, checkpoint_payload, configure_capacity, frozen_capacity_hash,
    restore_capacity_checkpoint, select_training_controls, validate_training_controls,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(91)
    system = NeutralAffectSystem({
        "data": {"content_dim": 8, "motion_dim": 52, "neutral_output_indices": [17, 18],
                 "emotion_classes": ["neutral", "angry", "happy", "sad"],
                 "num_intensity_levels": 4, "audio_emotion_dim": 5},
        "model": {"content_dim": 8, "emotion_dim": 64, "style_dim": 8, "hidden_dim": 8,
                  "heads": 2, "dropout": .2, "dit_dim": 8, "dit_depth": 2,
                  "residual_scale": .25, "affect_stride": 4, "affect_rank": 8,
                  "global_condition_dropout": .2, "style_condition_dropout": .2}}).eval()
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[1, -2:] = False
    content = torch.randn(2, 8, 8)
    with torch.no_grad():
        base = system.base(content, valid)
        identity = system.encode_identity(torch.randn(2, 2, 8, 52))
        affect = system.encode_audio(torch.randn(2, 8, 5), valid)
    q = {"content": content, "motion": torch.randn(2, 8, 52), "valid": valid,
         "channel_mask": torch.ones(2, 52, dtype=torch.bool)}
    q["channel_mask"][:, 0] = False
    weight = torch.tensor([[4., 4.], [4., 2.]])
    motion_bins = torch.randn(2, 2, 52)
    basis = torch.linalg.qr(torch.randn(52, 8)).Q.double()
    head = PredictableAudioHead({"basis": basis, "motion_channel_indices": list(range(52)),
        "std": torch.ones(5), "weights": torch.randn(5, 8) * .05}, motion_bins, weight).eval()
    freeze_module(head)
    controls = head(torch.randn(2, 2, 5), weight).detach()
    return system, head, {"q": q, "base": base, "identity": identity, "affect": affect}, controls, weight, motion_bins


def train_step(system, head, batch, controls, weight, motion_bins, arm, optimizer, generator):
    noise, time, _ = draws(generator, 2, (8, 52))
    local = select_training_controls(head, motion_bins, weight, controls, arm)
    loss = capacity_flow_loss(system, batch, local, weight, noise, time, arm)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def test_two_steps_change_only_capacity_and_distinguish_three_matched_arms(monkeypatch):
    original, head, batch, controls, weight, motion_bins = fixture()
    initial = copy.deepcopy(original.state_dict())
    frozen, head_hash = frozen_capacity_hash(original), state_hash(head.state_dict())
    changed_states, losses = {}, {}
    for arm in ARMS:
        model = copy.deepcopy(original)
        names, parameters = configure_capacity(model)
        assert state_hash(model.state_dict()) == state_hash(initial)
        assert all(n.startswith(PREFIXES) for n in names)
        monkeypatch.setattr(model, "base", lambda *a, **k: (_ for _ in ()).throw(AssertionError("B0 recomputed")))
        monkeypatch.setattr(model.motion_teacher, "forward", lambda *a, **k: (_ for _ in ()).throw(AssertionError("neural motion teacher read")))
        optimizer = torch.optim.Adam(parameters, lr=1e-5)
        generator = torch.Generator().manual_seed(46)
        losses[arm] = [train_step(model, head, batch, controls, weight, motion_bins, arm, optimizer, generator)
                       for _ in range(2)]
        assert_frozen(model, head, frozen, head_hash)
        changed = [n for n, v in model.state_dict().items() if not torch.equal(v, initial[n])]
        assert changed and all(n.startswith(PREFIXES) for n in changed)
        assert any(n.startswith("renderer.") for n in changed)
        assert ("local_projection.weight" in changed) == (arm != "zero_local")
        assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith(PREFIXES))
        assert all(p.grad is None for p in head.parameters())
        changed_states[arm] = state_hash(capacity_state(model))
    assert len(set(changed_states.values())) == 3
    assert len({values[0] for values in losses.values()}) == 3


def test_oracle_and_zero_ignore_external_student_values_and_no_target_reads_for_zero():
    model, head, batch, controls, weight, motion_bins = fixture()
    configure_capacity(model)
    noise, time, _ = draws(torch.Generator().manual_seed(46), 2, (8, 52))
    poison = torch.full_like(controls, float("nan"))
    for arm in ("oracle_local", "zero_local"):
        first = select_training_controls(head, motion_bins if arm == "oracle_local" else None, weight, controls, arm)
        second = select_training_controls(head, motion_bins if arm == "oracle_local" else None, weight, poison, arm)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        a = capacity_flow_loss(model, batch, first, weight, noise, time, arm)
        b = capacity_flow_loss(model, batch, second, weight, noise, time, arm)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unknown"):
        select_training_controls(head, motion_bins, weight, controls, "bad")


def test_checkpoint_restore_reproduces_generation_and_next_adam_step(tmp_path):
    original, head, batch, controls, weight, motion_bins = fixture()
    model = copy.deepcopy(original)
    _, parameters = configure_capacity(model)
    optimizer, generator = torch.optim.Adam(parameters, lr=1e-5), torch.Generator().manual_seed(46)
    recipe = {"initial_head_sha256": state_hash(head.state_dict()),
              "initial_system_sha256": state_hash(original.state_dict())}
    frozen = frozen_capacity_hash(model)
    train_step(model, head, batch, controls, weight, motion_bins, "audio_local", optimizer, generator)
    saved = checkpoint_payload(model, head, optimizer, generator, recipe, step=1,
                               completed_epochs=0, frozen_sha256=frozen)
    path = tmp_path / "compact.pt"
    torch.save(saved, path)
    restored = copy.deepcopy(original)
    _, restored_parameters = configure_capacity(restored)
    new_optimizer = torch.optim.Adam(restored_parameters, lr=1e-5)
    new_generator = torch.Generator().manual_seed(99)
    restore_capacity_checkpoint(restored, head, torch.load(path, weights_only=False), recipe,
                                new_optimizer, new_generator)
    assert state_hash(model.state_dict()) == state_hash(restored.state_dict())
    affect = projected_affect(model, batch, controls, weight, audio_gate=True)
    initial_noise = torch.randn(batch["q"]["motion"].shape, generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        a = model.generate(batch["q"]["content"], batch["q"]["valid"], batch["identity"], affect,
                           initial_noise, steps=2, base=batch["base"])["motion"]
        b = restored.generate(batch["q"]["content"], batch["q"]["valid"], batch["identity"], affect,
                              initial_noise, steps=2, base=batch["base"])["motion"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    first = train_step(model, head, batch, controls, weight, motion_bins, "audio_local", optimizer, generator)
    second = train_step(restored, head, batch, controls, weight, motion_bins, "audio_local", new_optimizer, new_generator)
    assert first == second
    assert state_hash(model.state_dict()) == state_hash(restored.state_dict())
    broken = copy.deepcopy(saved)
    broken["capacity"]["local_projection.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="keys/hash"):
        restore_capacity_checkpoint(copy.deepcopy(original), head, broken)
    with torch.no_grad():
        original.identity_bias.bias[0] += 1
    with pytest.raises(ValueError, match="Frozen source"):
        restore_capacity_checkpoint(original, head, saved)


def test_training_control_contract_rejects_reordering_wrong_coordinates_and_noncentered():
    _, head, _, controls, weight, _ = fixture()
    loaded = {"bundle": {"bundles": {"internal": {"clip_id": ["a", "b"], "weight": weight}}},
              "head": head, "input_sha256": {"cache": "abc"}}
    record = {"schema": CONTROL_SCHEMA, "clip_ids": ["a", "b"], "controls": controls,
              "weight": weight, "source_input_sha256": loaded["input_sha256"],
              "head_sha256": state_hash(head.state_dict())}
    torch.testing.assert_close(validate_training_controls(record, loaded), controls, rtol=0, atol=0)
    with pytest.raises(ValueError, match="clip order"):
        validate_training_controls({**record, "clip_ids": ["b", "a"]}, loaded)
    with pytest.raises(ValueError, match="different fixed head"):
        validate_training_controls({**record, "head_sha256": "bad"}, loaded)
    with pytest.raises(ValueError, match="clip-centered"):
        validate_training_controls({**record, "controls": controls + 1}, loaded)
