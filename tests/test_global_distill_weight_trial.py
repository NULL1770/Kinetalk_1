import sys

import pytest
import torch

from scripts.train_full_staged import global_distillation, main, parser


def test_default_weight_and_zero_coordinate_gradient_preserve_other_path():
    assert parser().parse_args(['--output', 'trial']).global_distill_weight == .5
    code = torch.ones(2, 3, requires_grad=True)
    target = torch.zeros_like(code)
    logits = torch.tensor([[1., 0.], [1., 0.]])
    distill, _ = global_distillation(code, target, logits, torch.tensor([0, 1]))
    torch.testing.assert_close(.5 * distill, .5 * torch.nn.functional.mse_loss(code, target), rtol=0, atol=0)
    (0. * distill + (code * 3).sum()).backward()
    torch.testing.assert_close(code.grad, torch.full_like(code, 3), rtol=0, atol=0)


@pytest.mark.parametrize('extra', [
    ['--global-distill-weight', 'nan'],
    ['--global-distill-weight', '-1'],
    ['--global-distill-weight', 'inf'],
    ['--global-distill-weight', '0'],
    ['--global-distill-weight', '0', '--start-stage', 'audio', '--end-stage', 'audio'],
    ['--global-distill-weight', '0', '--start-stage', 'audio', '--end-stage', 'audio',
     '--stage-checkpoint', 'unused.pt', '--global-distill-gate', 'teacher-agreement'],
])
def test_invalid_or_nonisolated_trial_fails_before_filesystem_access(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['train_full_staged.py', '--output', 'unused', *extra])
    with pytest.raises(ValueError):
        main()
