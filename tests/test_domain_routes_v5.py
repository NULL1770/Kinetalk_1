from types import SimpleNamespace
import pytest
import torch
from torch import nn
from kinetalk_v4.domain_routes import content_style_gradient_scope


def model():
    return SimpleNamespace(teacher=nn.Linear(3,3), student=nn.Linear(3,3),
        events=nn.Linear(3,3), emotion_to_state=nn.Linear(3,3),
        decoder=SimpleNamespace(affect=nn.Linear(3,3), output=nn.Linear(3,3)),
        style=nn.Linear(3,3))


def test_hdtf_scope_keeps_existing_mead_gradients_and_routes_new_gradients():
    m = model(); x = torch.ones(2,3)
    (m.student(x).sum() + m.decoder.affect(x).sum()).backward()
    student_before = m.student.weight.grad.clone()
    affect_before = m.decoder.affect.weight.grad.clone()
    with content_style_gradient_scope(m):
        signal = m.student(x) + m.decoder.affect(x) + m.events(x) + m.teacher(x) + m.emotion_to_state(x)
        m.decoder.output(signal + m.style(x)).sum().backward()
    assert torch.equal(m.student.weight.grad, student_before)
    assert torch.equal(m.decoder.affect.weight.grad, affect_before)
    assert m.events.weight.grad is None and m.teacher.weight.grad is None
    assert m.emotion_to_state.weight.grad is None
    assert m.style.weight.grad.abs().sum() > 0
    assert m.decoder.output.weight.grad.abs().sum() > 0
    assert m.student.weight.requires_grad and m.decoder.affect.weight.requires_grad


def test_scope_restores_mixed_requires_grad_flags_even_after_failure():
    m = model(); m.teacher.weight.requires_grad_(False)
    with pytest.raises(ValueError):
        with content_style_gradient_scope(m):
            assert not m.student.weight.requires_grad
            raise ValueError('diagnostic interruption')
    assert not m.teacher.weight.requires_grad
    assert m.teacher.bias.requires_grad and m.student.weight.requires_grad
