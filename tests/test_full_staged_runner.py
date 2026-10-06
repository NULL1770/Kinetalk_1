import copy

import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_full_staged import (
    STAGES, SCHEMA, base_forward, identity_pairs, targets, UpperFlow,
    upper_motion, merge_teacher_audio, content_timing_alignment,
    class_balanced_weights, independent_probe_consistency,
    motion_statistics_consistency, class_global_means,
)
from kinetalk_b0.emotion_probe import MotionEmotionProbe
from kinetalk_b0.models.slow_state_affect import compose_upper_face, UPPER_INDICES, readout_slow_state, masked_slow_state
from scripts.train_full_staged import parser


def config():
    return {'data':{'motion_dim':52,'content_dim':8,'audio_dim':8,'neutral_output_indices':[17,18],
                    'emotion_classes':['neutral','angry'],'num_intensity_levels':4},
            'model':{'content_dim':8,'emotion_dim':8,'style_dim':8,'hidden_dim':8,'heads':2,
                     'dit_dim':8,'dit_depth':1,'dropout':0.,'residual_scale':.25,'affect_hidden_dim':8}}


def test_b0_training_has_gradients_and_padding_invariance():
    torch.set_num_threads(1);torch.manual_seed(10)
    system=NeutralAffectSystem(config()).eval();system.stage1.requires_grad_(True)
    x=torch.randn(2,12,8);mask=torch.ones(2,12,dtype=torch.bool);mask[0,9:]=False
    y=base_forward(system,x,mask,gradients=True)
    padded=base_forward(system,torch.nn.functional.pad(x,(0,0,0,4)),torch.nn.functional.pad(mask,(0,4)),gradients=True)
    torch.testing.assert_close(y['b0'],padded['b0'][:,:12],rtol=0,atol=0)
    y['b0'].square().mean().backward()
    grads=[p.grad for p in system.stage1.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0


def test_class_balance_and_external_probe_critic_are_finite_and_differentiable():
    weights = class_balanced_weights(torch.tensor([0, 0, 0, 1, 1, 2]), 3, power=.5)
    assert weights[2] > weights[1] > weights[0] > 0
    probe = MotionEmotionProbe(12, 8, 3).eval()
    motion = torch.randn(2, 6, 2, requires_grad=True)
    valid = torch.ones(2, 6, dtype=torch.bool)
    loss = independent_probe_consistency(
        probe, motion, valid, torch.tensor([True, True]), torch.tensor([0, 1]),
        class_weights=weights,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert motion.grad is not None and torch.isfinite(motion.grad).all()


def test_motion_statistics_consistency_is_finite_and_ignores_padding():
    target = torch.randn(2, 6, 2)
    generated = target.clone().requires_grad_(True)
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[0, 4:] = False
    generated.data[0, 4:] = 1000.
    loss = motion_statistics_consistency(generated, target, valid, torch.tensor([True, True]))
    assert torch.isfinite(loss) and loss.item() == 0
    (loss + generated.square().sum() * 0).backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()


def test_identity_epoch_uses_only_fit_disjoint_views():
    refs={7:{'valid':torch.ones(4,12,dtype=torch.bool)},19:{'valid':torch.ones(2,12,dtype=torch.bool)},99:{'valid':torch.ones(4,12,dtype=torch.bool)}}
    pairs=identity_pairs(refs,[7,19])
    assert len(pairs)==4
    for sid,a,b in pairs:
        assert sid in (7,19) and not set(a)&set(b)
        assert set(a)|set(b)==set(range(len(refs[sid]['valid'])))


def test_runner_flow_target_and_nonupper_contract():
    torch.set_num_threads(1);torch.manual_seed(8)
    b,t=2,20;valid=torch.ones(b,t,dtype=torch.bool);valid[0,5]=False
    anchor=torch.full((b,52),.2);scales=torch.linspace(.03,.12,52)
    q={'motion':anchor[:,None]+torch.randn(b,t,52)*.02,'valid':valid,'channel_mask':torch.ones(b,52,dtype=torch.bool),
       'anchors':anchor,'anchor_valid':torch.ones(b,52,dtype=torch.bool),'h0':torch.randn(b,t,8)}
    truth,innovation=targets(q,scales,8)
    identity={'code':torch.randn(b,8)};affect={'global':torch.randn(b,8),'intensity_value':torch.ones(b,1)}
    local=torch.randn(b,t,8,requires_grad=True);model=UpperFlow(config(),stride=8)
    noise=torch.randn(b,t,9);flowtime=torch.rand(b)
    loss=model.flow_loss(innovation,q,identity,affect,local,truth['state'],noise,flowtime)
    loss.backward();assert torch.isfinite(local.grad).all() and local.grad.abs().sum()>0
    residual=model.decode(q,identity,affect,local.detach(),truth['state'],noise,3)
    uv=upper_motion(q,scales,truth['state'],residual)
    base=torch.randn(b,t,52);generated=compose_upper_face(base,uv,valid)
    others=[i for i in range(52) if i not in UPPER_INDICES]
    assert torch.equal(generated[...,others],base[...,others])
    read,mask=readout_slow_state(generated,valid[...,None].expand_as(generated),anchor,scales)
    back=masked_slow_state(read,mask,stride=8)['state']
    torch.testing.assert_close(back,truth['state'],atol=2e-6,rtol=2e-5)


def test_teacher_global_and_audio_temporal_contract_with_gradient():
    """Stage 3 keeps teacher semantics while learning audio timing."""
    teacher_global = torch.nn.Parameter(torch.randn(2, 8))
    teacher_local = torch.nn.Parameter(torch.randn(2, 5, 8))
    audio_timing = torch.nn.Parameter(torch.randn(2, 5, 8))
    teacher = {'global': teacher_global, 'local': teacher_local,
               'emotion_logits': teacher_global, 'intensity_logits': teacher_global,
               'intensity_value': teacher_global[:, :1]}
    audio = {'global': torch.randn(2, 8), 'u_a': audio_timing,
             'emotion_logits': torch.randn(2, 2), 'intensity_logits': torch.randn(2, 4),
             'intensity_value': torch.randn(2, 1)}
    merged = merge_teacher_audio(teacher, audio)
    assert merged['global'] is teacher_global
    assert merged['u_a'] is audio_timing
    assert merged['temporal'] is audio_timing
    assert merged['local'] is audio_timing
    loss = merged['global'].square().mean() + merged['u_a'].square().mean()
    loss.backward()
    assert teacher_global.grad is not None and teacher_global.grad.abs().sum() > 0
    assert audio_timing.grad is not None and audio_timing.grad.abs().sum() > 0
    assert teacher_local.grad is None


def test_default_runner_protocol_has_no_stage5():
    assert STAGES == ('articulation', 'identity', 'teacher', 'audio')
    assert 'v2' in SCHEMA


def test_support_expansion_is_explicitly_opt_in():
    args = parser().parse_args([
        '--output', 'tmp-run', '--stage-checkpoint', 'audio.pt',
        '--start-stage', 'audio', '--end-stage', 'audio',
    ])
    assert args.allow_residual_support_expansion is False
    args = parser().parse_args([
        '--output', 'tmp-run', '--stage-checkpoint', 'audio.pt',
        '--start-stage', 'audio', '--end-stage', 'audio',
        '--allow-residual-support-expansion',
    ])
    assert args.allow_residual_support_expansion is True


def test_bounded_output_is_explicitly_opt_in():
    args = parser().parse_args([
        '--output', 'tmp-run', '--stage-checkpoint', 'audio.pt',
        '--start-stage', 'audio', '--end-stage', 'audio',
    ])
    assert args.bounded_output is False
    args = parser().parse_args([
        '--output', 'tmp-run', '--stage-checkpoint', 'audio.pt',
        '--start-stage', 'audio', '--end-stage', 'audio', '--bounded-output',
    ])
    assert args.bounded_output is True


def test_class_prototype_targets_are_order_independent_and_complete():
    import pytest
    codes = torch.tensor([[1., 2.], [8., 10.], [3., 4.], [6., 8.]])
    labels = torch.tensor([0, 1, 0, 1])
    means, counts = class_global_means(codes, labels, 2)
    torch.testing.assert_close(means, torch.tensor([[2., 3.], [7., 9.]]))
    assert counts.tolist() == [2, 2]
    ix = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(class_global_means(codes[ix], labels[ix], 2)[0], means)
    with pytest.raises(ValueError, match='Every declared emotion'):
        class_global_means(codes, labels, 3)
    args = parser().parse_args(['--output', 'tmp-run'])
    assert args.global_distill_target == 'clip'


def test_output_projection_preserves_flow_targets_masks_and_protected_channels():
    torch.set_num_threads(1); torch.manual_seed(12)
    system = NeutralAffectSystem(config()).eval()
    assert system.output_projection is False
    # A zero vector field makes the raw source/endpoint exactly reproducible.
    with torch.no_grad():
        system.renderer.output.weight.zero_(); system.renderer.output.bias.zero_()
    support = torch.ones(52, dtype=torch.bool); support[51] = False
    system.set_motion_support(support)
    residual_support = support.clone(); residual_support[17] = False
    system.set_residual_support(residual_support)
    b, t = 2, 6
    content = torch.randn(b, t, 8)
    valid = torch.ones(b, t, dtype=torch.bool); valid[0, -2:] = False
    channel = support[None].expand(b, -1).clone(); channel[0, 3] = False
    motion = torch.full((b, t, 52), .5)
    baseline = torch.full((b, 52), .1)
    base = {'b0': torch.full((b, t, 52), .2), 'h0': torch.randn(b, t, 8)}
    base['b0'][..., 17] = 1.2  # A protected legacy channel is never projected.
    identity = {'code': torch.randn(b, 8), 'baseline': baseline}
    affect = {'global': torch.randn(b, 8), 'intensity_value': torch.ones(b, 1)}
    noise = torch.zeros(b, t, 52); noise[..., 0] = 8.; noise[..., 1] = -8.
    time = torch.full((b,), .2)
    raw = system.flow(motion, content, valid, identity, affect, noise=noise, time=time,
                      base=base, observation_mask=channel)
    raw_generation = system.generate(content, valid, identity, affect, noise, 2, base=base)
    system.set_output_projection(True)
    bounded = system.flow(motion, content, valid, identity, affect, noise=noise, time=time,
                          base=base, observation_mask=channel)
    bounded_generation = system.generate(content, valid, identity, affect, noise, 2, base=base)
    for key in ('prediction', 'velocity_target', 'x_t', 'target_residual', 'predicted_residual'):
        torch.testing.assert_close(bounded[key], raw[key], rtol=0, atol=0)
    active = bounded['observation_mask']
    torch.testing.assert_close(bounded['raw_motion'], raw['motion'], rtol=0, atol=0)
    torch.testing.assert_close(bounded['motion'][active], raw['motion'][active].clamp(0, 1))
    assert ((raw['motion'][active] < 0) | (raw['motion'][active] > 1)).any()
    assert not ((bounded['motion'][active] < 0) | (bounded['motion'][active] > 1)).any()
    torch.testing.assert_close(bounded['motion'][~active], raw['motion'][~active], rtol=0, atol=0)
    assert bounded['motion'][0, :, 3].count_nonzero() == 0
    assert bounded['motion'][~valid].count_nonzero() == 0
    assert bounded_generation['motion'][~valid].count_nonzero() == 0
    torch.testing.assert_close(bounded_generation['raw_motion'], raw_generation['motion'], rtol=0, atol=0)
    gactive = valid[..., None] & residual_support[None, None]
    torch.testing.assert_close(bounded_generation['motion'][gactive], raw_generation['motion'][gactive].clamp(0, 1))
    torch.testing.assert_close(bounded_generation['motion'][..., 17], raw_generation['motion'][..., 17], rtol=0, atol=0)
    observed = valid[..., None] & channel[:, None]
    anchor = torch.where(observed, base['b0'] + baseline[:, None], 0.)
    torch.testing.assert_close(anchor + bounded['residual'], bounded['motion'])
    bounded['motion'][active].sum().backward()
    assert torch.isfinite(system.renderer.output.bias.grad).all()
    assert system.renderer.output.bias.grad.abs().sum() > 0


def test_content_timing_alignment_allows_gain_but_penalizes_direction_change():
    valid = torch.ones(1, 4, dtype=torch.bool)
    channel = torch.ones(1, 52, dtype=torch.bool)
    times = torch.arange(4, dtype=torch.float32).view(1, -1) / 25
    base = torch.zeros(1, 4, 52)
    base[0, :, 17] = torch.tensor([0., 0.2, 0.4, 0.6])
    same = base * 2
    flipped = base.clone(); flipped[0, :, 17] = torch.tensor([0.6, 0.4, 0.2, 0.])
    aligned = content_timing_alignment(same, base, valid, channel, times, channels=(17,))
    changed = content_timing_alignment(flipped, base, valid, channel, times, channels=(17,))
    assert float(aligned) < 1e-3
    assert float(changed) > 1.0
