"""Check supervision isolation without closing the mouth forward pathway."""
import copy

import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect, UPPER_INDICES
from scripts.train_full_staged import (
    audio_affect, expression_gradient_view, flow_vector_mse, optimize,
    parser, student_gradient_overrides,
)


def scene():
    torch.set_num_threads(1)
    torch.manual_seed(83)
    cfg = {'data': {'motion_dim': 52, 'content_dim': 8, 'audio_dim': 8,
                    'neutral_output_indices': [17, 18],
                    'emotion_classes': ['neutral', 'happy'], 'num_intensity_levels': 4},
           'model': {'content_dim': 8, 'emotion_dim': 8, 'style_dim': 8,
                     'hidden_dim': 8, 'heads': 2, 'dit_dim': 8, 'dit_depth': 1,
                     'dropout': 0., 'residual_scale': .25, 'affect_hidden_dim': 8}}
    system = NeutralAffectSystem(cfg).eval()
    student = SlowStateAffect(torch.zeros(772), torch.ones(772),
                             hidden=8, global_dim=8, local_dim=8, num_emotions=2).eval()
    with torch.no_grad():
        system.renderer.output.weight.normal_(0, .1)
        student.local_head.weight.normal_(0, .1)
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[0, -2:] = False
    content = torch.randn(2, 7, 8)
    motion = torch.randn(2, 7, 52)
    identity = {'code': torch.randn(2, 8), 'baseline': torch.randn(2, 52)}
    base = {'b0': torch.randn(2, 7, 52), 'h0': torch.randn(2, 7, 8)}
    features = torch.randn(2, 7, 1540)
    affect = audio_affect(student, features, valid)
    noise, time = torch.randn_like(motion), torch.tensor([.3, .8])
    output = system.flow(motion, content, valid, identity, affect,
                         noise=noise, time=time, base=base)
    return system, student, output, affect, valid


def test_mouth_error_cannot_train_student_but_renderer_and_mouth_response_remain():
    system, student, output, affect, valid = scene()
    view = expression_gradient_view(output['prediction'])
    assert torch.equal(view, output['prediction'])
    mouth_only = view[..., 14:41].square().sum()
    student_grads = torch.autograd.grad(mouth_only, list(student.parameters()),
                                      allow_unused=True, retain_graph=True)
    assert all(g is None or not g.count_nonzero() for g in student_grads)
    # The renderer uses the untouched objective, rather than the student view.
    renderer_grads = torch.autograd.grad(output['prediction'][..., 14:41].square().sum(), list(system.renderer.parameters()),
                                       allow_unused=True, retain_graph=True)
    assert any(g is not None and g.abs().sum() > 0 for g in renderer_grads)
    expression_only = view[..., list(UPPER_INDICES)].square().sum()
    local_grad = torch.autograd.grad(expression_only, student.local_head.weight,
                                     retain_graph=True)[0]
    assert torch.isfinite(local_grad).all() and local_grad.abs().sum() > 0
    # Original forward jaw explicitly responds to audio timing/global inputs.
    local_jaw_grad = torch.autograd.grad(output['prediction'][..., 17].square().sum(),
                                        affect['u_a'], retain_graph=True)[0]
    assert local_jaw_grad[valid].abs().sum() > 0
    assert not local_jaw_grad[~valid].count_nonzero()


def test_replacement_preserves_full_renderer_gradient_and_expression_student_gradient():
    system, student, output, affect, valid = scene()
    full = flow_vector_mse(output['prediction'], output['velocity_target'], output['observation_mask'])
    student_loss = flow_vector_mse(expression_gradient_view(output['prediction']),
                                  output['velocity_target'], output['observation_mask'])
    assert torch.equal(full, student_loss)
    renderer_params = list(system.renderer.parameters())
    student_params = list(student.parameters())
    renderer_expected = torch.autograd.grad(full, renderer_params, retain_graph=True, allow_unused=True)
    student_expected = torch.autograd.grad(student_loss, student_params, retain_graph=True, allow_unused=True)
    overrides = student_gradient_overrides(student_loss, student)
    params = renderer_params + student_params
    optimizer = torch.optim.SGD(params, lr=1e-4)
    norm = optimize(full, optimizer, params, gradient_overrides=overrides)
    factor = min(1., 1. / (norm + 1e-6))
    for p, g in zip(params, list(renderer_expected) + list(student_expected)):
        if g is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, g * factor, rtol=1e-6, atol=1e-8)
    assert any(g is not None and g.abs().sum() > 0 for g in student_expected)


def test_default_optimizer_is_exact_previous_update_and_rng():
    torch.manual_seed(17)
    module = torch.nn.Linear(4, 3)
    previous = copy.deepcopy(module)
    inputs = torch.randn(2, 4)
    rng = torch.get_rng_state().clone()
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    old_optimizer = torch.optim.Adam(previous.parameters(), lr=1e-4)
    norm = optimize(module(inputs).square().sum(), optimizer, list(module.parameters()))
    old_optimizer.zero_grad(set_to_none=True)
    previous(inputs).square().sum().backward()
    old_norm = torch.nn.utils.clip_grad_norm_(previous.parameters(), 1., error_if_nonfinite=True)
    old_optimizer.step()
    assert norm == float(old_norm)
    assert all(torch.equal(p, q) for p, q in zip(module.parameters(), previous.parameters()))
    assert torch.equal(torch.get_rng_state(), rng)


def test_default_gradient_scope_and_native_expression_contract():
    assert parser().parse_args(['--output', 'unused']).audio_gradient_scope == 'full'
    assert set(UPPER_INDICES).isdisjoint(range(14, 41))
    assert len(set(UPPER_INDICES)) == 9
