import copy

import numpy as np
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
    with torch.no_grad():
        model.onset_head.weight.normal_()
        model.onset_head.bias.normal_()
        model.duration_head.weight.normal_()
        model.duration_head.bias.normal_()
    changed = model(torch.zeros_like(condition), context, valid)
    torch.testing.assert_close(changed['onset_logits'], static['onset_logits'], rtol=0, atol=0)
    torch.testing.assert_close(changed['duration_logits'], static['duration_logits'], rtol=0, atol=0)


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
    loss = out['onset_logits'][valid].square().mean() + out['duration_logits'][valid].square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert any(p.grad is not None for p in model.input.parameters())
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not model.training or not model.base.training
    assert torch.equal(model(condition, context, valid)['onset_logits'][:, 13:17],
                       model.base(context, 31)['onset_logits'][:, 13:17])


def test_optimizer_keeps_frozen_prior_grad_free_and_null_exact_after_training():
    torch.manual_seed(37)
    base = StaticScheduleHead(context_dim=4, hidden=8)
    model = ProsodyResidualSchedule(base, condition_dim=10, hidden=8)
    condition = torch.randn(3, 29, 10)
    context = torch.randn(3, 4)
    valid = torch.ones(3, 29, dtype=torch.bool)
    before = copy.deepcopy(base.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003)
    # Both outputs train against varied targets, rather than only checking an
    # untouched zero-initialized model.  This must not update the static head.
    target_onset = (condition[..., :4] > .25).float()
    target_duration = condition[..., :1, None].expand(-1, -1, 4, 7)
    for _ in range(4):
        model.train()
        out = model(condition, context, valid)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out['onset_logits'], target_onset)
        loss = loss + (out['duration_logits']-target_duration).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None for p in model.base.parameters())
        optimizer.step()
    assert model.onset_head.weight.abs().max() > 0
    assert model.duration_head.weight.abs().max() > 0
    assert not model.base.training
    for name, value in model.base.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    null = model(torch.zeros_like(condition), context, valid)
    expected = model.base(context, condition.shape[1])
    for key in ('onset_logits', 'duration_logits'):
        torch.testing.assert_close(null[key], expected[key], rtol=0, atol=0)


def test_pretrained_static_head_gradients_are_cleared_when_freezing():
    torch.manual_seed(39)
    base = StaticScheduleHead(context_dim=4, hidden=8)
    optimizer = torch.optim.AdamW(base.parameters(), lr=.003)
    out = base(torch.randn(2, 4), 19)
    loss = out['onset_logits'].square().mean() + out['duration_logits'].square().mean()
    loss.backward()
    optimizer.step()
    assert any(p.grad is not None for p in base.parameters())
    # This is the actual runner lifecycle: the last static training step has
    # populated .grad before the trained head is passed to the residual.
    model = ProsodyResidualSchedule(base, condition_dim=10, hidden=8)
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())


def test_local_prosody_cannot_cross_native_gap():
    torch.manual_seed(41)
    model = ProsodyResidualSchedule(StaticScheduleHead(context_dim=4, hidden=8), hidden=8).eval()
    with torch.no_grad():
        model.onset_head.weight.normal_()
        model.duration_head.weight.normal_()
    context = torch.randn(1, 4)
    valid = torch.ones(1, 43, dtype=torch.bool)
    valid[:, 17:21] = False
    first = torch.randn(1, 43, 10)
    # Even dilation-8 taps must not jump over the gap into the changed run.
    second = first.clone()
    second[:, 21:] = torch.randn_like(second[:, 21:])*200
    first[:, 17:21] = float('nan')
    second[:, 17:21] = float('inf')
    left = model(first, context, valid)
    right = model(second, context, valid)
    for key in ('onset_logits', 'duration_logits'):
        torch.testing.assert_close(left[key][:, :17], right[key][:, :17], rtol=0, atol=0)
        assert torch.isfinite(left[key]).all()
        assert torch.isfinite(right[key]).all()


def prepare_fixture():
    frames = 50
    time = torch.arange(frames).float()
    feature = torch.zeros(frames, 1540)
    feature[:, 1536:] = torch.stack((5+.1*torch.sin(time/3), -3+.5*torch.sin(time/5),
                                   .5+.3*torch.cos(time/4), (time.remainder(7) > 1).float()), -1)
    valid = torch.ones(frames, dtype=torch.bool)
    valid[19:23] = False
    feature[~valid] = float('nan')
    clip = {'clip_id': 'fixture', 'sentence': 'held_sentence', 'features': feature,
            'valid': valid, 'context': torch.linspace(-1, 1, 202),
            'motion9': torch.randn(frames, 9), 'motion_mask': torch.ones(frames, 9, dtype=torch.bool)}
    schedule = np.zeros((frames, 12), dtype=np.float32)
    onset = np.zeros((frames, 4), dtype=bool)
    known = np.ones((frames, 4), dtype=bool)
    labels = {'fixture': {'schedule': schedule, 'onset': onset, 'known': known, 'events': []}}
    stats = {'audio_mean': torch.zeros(1540), 'audio_scale': torch.ones(1540),
             'context_mean': torch.zeros(202), 'context_scale': torch.ones(202)}
    return clip, labels, stats


def test_prepare_inputs_independent_of_motion_teacher_known_and_events():
    from scripts.train_prosody_residual_event import prepare
    clip, labels, stats = prepare_fixture()
    reference = prepare(clip, labels, stats)
    changed_clip, changed_labels = copy.deepcopy(clip), copy.deepcopy(labels)
    changed_clip['motion9'][:] = float('nan')
    changed_clip['motion_mask'][:] = False
    changed_labels['fixture']['known'][:] = False
    changed_labels['fixture']['schedule'][4:10, 0] = 1
    changed_labels['fixture']['onset'][4, 0] = True
    changed_labels['fixture']['events'] = [{'start': 4, 'end': 10, 'group_index': 0}]
    changed = prepare(changed_clip, changed_labels, stats)
    for key in ('condition', 'context', 'valid'):
        torch.testing.assert_close(changed[key], reference[key], rtol=0, atol=0)
    assert not torch.equal(changed['risk'], reference['risk'])
    assert torch.isfinite(reference['condition']).all()
    assert torch.isfinite(reference['context']).all()


def test_prepare_static_reverse_keep_context_and_native_support():
    from scripts.train_prosody_residual_event import prepare
    from scripts.event_schedule_teacher import valid_runs
    clip, labels, stats = prepare_fixture()
    real = prepare(clip, labels, stats)
    null = prepare(clip, labels, stats, mode='null')
    reverse = prepare(clip, labels, stats, mode='reverse')
    for changed in (null, reverse):
        torch.testing.assert_close(changed['context'], real['context'], rtol=0, atol=0)
        torch.testing.assert_close(changed['valid'], real['valid'], rtol=0, atol=0)
        for key in ('onset', 'risk', 'duration'):
            torch.testing.assert_close(changed[key], real[key], rtol=0, atol=0)
    assert torch.equal(null['condition'], torch.zeros_like(null['condition']))
    assert not torch.equal(reverse['condition'][real['valid']], real['condition'][real['valid']])
    for start, end in valid_runs(real['valid'].numpy()):
        torch.testing.assert_close(real['condition'][start:end].mean(0), torch.zeros(10), rtol=0, atol=1e-6)
        torch.testing.assert_close(reverse['condition'][start:end].mean(0), torch.zeros(10), rtol=0, atol=1e-6)
    assert torch.equal(real['condition'][~real['valid']], torch.zeros_like(real['condition'][~real['valid']]))


def test_uniform_batch_crops_do_not_depend_on_event_target_locations():
    from scripts.train_prosody_residual_event import prepare, batch
    clip, labels, stats = prepare_fixture()
    # Long native runs force random cropping.  Copying onset labels must not
    # change which acoustic frames are sampled when the RNG seed is fixed.
    clip['features'] = clip['features'].repeat(5, 1)
    clip['valid'] = torch.ones(250, dtype=torch.bool)
    clip['valid'][100:110] = False
    clip['features'][:] = torch.nan_to_num(clip['features'])
    labels['fixture'] = {'schedule': np.zeros((250, 12), np.float32),
                         'onset': np.zeros((250, 4), bool),
                         'known': np.ones((250, 4), bool), 'events': []}
    first = prepare(clip, labels, stats)
    second = copy.deepcopy(first)
    second['onset'][::3] = 1
    second['risk'][::2] = False
    x1, c1, v1, *_ = batch([first], np.random.default_rng(701), 'cpu', frames=70, size=8)
    x2, c2, v2, *_ = batch([second], np.random.default_rng(701), 'cpu', frames=70, size=8)
    for original, changed in ((x1, x2), (c1, c2), (v1, v2)):
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
