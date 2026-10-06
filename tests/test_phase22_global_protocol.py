"""Guard the single intervention and the native full-evaluation protocol."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('phase22', Path(__file__).resolve().parents[1] /
                                            '.codex-finalizer/run_phase22_global_diagnostic.py')
phase22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase22)


def test_padded_noise_matches_original_full_evaluate_and_does_not_depend_on_batches():
    valid = torch.ones(5, 11, dtype=torch.bool)
    before = torch.random.get_rng_state().clone()
    noise = phase22.native_padded_noise(valid, 123)
    expected = torch.randn(5, 11, 52, generator=torch.Generator().manual_seed(123))
    assert torch.equal(noise, expected) and torch.equal(before, torch.random.get_rng_state())
    # A later native batch takes a slice of the full tensor; it must not redraw
    # at the smaller batch width (which changes every subsequent sample).
    ids = torch.tensor([2, 4])
    assert torch.equal(noise[ids, :6], expected[ids, :6])
    wrong = torch.randn(2, 6, 52, generator=torch.Generator().manual_seed(123))
    assert not torch.equal(noise[ids, :6], wrong)


def test_global_only_retains_scalar_temporal_logits_and_original_dictionary():
    audio = {'global': torch.randn(2, 64), 'u_a': torch.randn(2, 8, 64),
             'intensity_value': torch.randn(2, 1), 'emotion_logits': torch.randn(2, 8),
             'intensity_logits': torch.randn(2, 4)}
    teacher = {'global': torch.randn(2, 64), 'intensity_value': torch.ones(2, 1) * 3}
    original = audio['global'].clone()
    changed = phase22.replace_global(audio, teacher)
    assert changed is not audio and changed['global'] is teacher['global']
    assert torch.equal(audio['global'], original)
    assert all(changed[k] is v for k, v in audio.items() if k != 'global')
    with pytest.raises(ValueError):
        phase22.replace_global(audio, {'global': torch.randn(3, 64)})


def test_canonical_dispatch_flags_require_no_grad_and_cannot_change_parameters():
    system = torch.nn.Module()
    system.stage1 = torch.nn.Linear(2, 2)
    system.renderer = torch.nn.Linear(2, 2)
    system.motion_teacher = torch.nn.Linear(2, 2)
    audio = torch.nn.Linear(2, 2)
    saved = {k: v.clone() for k, v in system.state_dict().items()}
    with pytest.raises(RuntimeError):
        phase22.canonical_audio_evaluation_flags(system, audio)
    with torch.no_grad():
        phase22.canonical_audio_evaluation_flags(system, audio)
        assert not torch.is_grad_enabled()
        assert all(p.requires_grad for p in system.renderer.parameters())
        assert all(p.requires_grad for p in audio.parameters())
        assert all(not p.requires_grad for p in system.stage1.parameters())
        assert all(not p.requires_grad for p in system.motion_teacher.parameters())
        assert all(p.grad is None for m in (system, audio) for p in m.parameters())
    assert all(torch.equal(v, saved[k]) for k, v in system.state_dict().items())


def fixture():
    saved = {'clip_id': ['a', 'b', 'c'], 'target': torch.randn(3, 9, 52),
             'valid': torch.ones(3, 9, dtype=torch.bool), 'b0': torch.randn(3, 9, 52),
             'times': torch.arange(9, dtype=torch.float64)[None].repeat(3, 1),
             'channel_mask': torch.ones(3, 52, dtype=torch.bool)}
    ids = torch.tensor([0, 2])
    batch = {'clip_id': ['a', 'c'], **{k: saved[v][ids, :6].clone() if k != 'channel_mask'
             else saved[v][ids].clone() for k, v in (('motion', 'target'), ('valid', 'valid'),
                 ('times', 'times'), ('b0', 'b0'), ('channel_mask', 'channel_mask'))}}
    return batch, saved, ids


def test_native_batch_accepts_identical_reference_after_trimming():
    phase22.assert_reference_batch(*fixture())


@pytest.mark.parametrize('key', ('motion', 'valid', 'times', 'channel_mask', 'b0', 'clip_id'))
def test_replay_rejects_target_clock_observation_base_or_clip_drift(key):
    b, saved, ids = fixture()
    if key == 'clip_id':
        b[key].reverse()
    elif b[key].dtype == torch.bool:
        b[key].view(-1)[0] = False
    else:
        b[key].view(-1)[0] += .01
    with pytest.raises(AssertionError):
        phase22.assert_reference_batch(b, saved, ids)
